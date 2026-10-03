"""CPU matrix-product policy; no opaque ops::Matmul forwarding."""

from dataclasses import fields, replace
from types import MappingProxyType
from ... import ir as I
from ...passes.manager import register_pass
from ...ir.traversal import walk
from ...lowering.tile_to_vector import lower_stores
from .transpose import plan_transpose
from ...lowering.transpose import register_transpose, x86_dword_transpose
from ...cost import estimate
from .policies import recompute_limit


@register_pass("cpu-plan-matmul", stage="cpu-memory", after=("cpu-plan-transpose",))
def plan_matmul(prepared, context):
  plans, dense_packs, seen, grouped_epilogues = {}, {}, set(), set()
  costs = context.environment.costs
  stores, transposes = dict(prepared.stores), dict(prepared.transposes)
  fusion = context.options.get("cpu_matmul_fusion", "auto")
  if fusion not in {"auto", "always", "never"}:
    raise ValueError("cpu_matmul_fusion must be auto, always or never")
  gemv_columns = context.options.get("cpu_gemv_columns", 4)
  if type(gemv_columns) is not int or not 1 <= gemv_columns <= 8:
    raise ValueError("cpu_gemv_columns must be an integer in [1,8]")
  k_unroll = context.options.get("cpu_gemv_k_unroll", int(costs.policy("gemv_k_unroll", 2)))
  if type(k_unroll) is not int or k_unroll not in (1, 2, 4):
    raise ValueError("cpu_gemv_k_unroll must be 1, 2 or 4")

  def product_k_unroll(product):
    known_k = I.constant(product.args[0].type.shape[1])
    if "cpu_gemv_k_unroll" not in context.options:
      # Unbounded K can be long. Keep the conservative latency-hiding bank
      # count instead of imposing the short, compile-time-specialized choice.
      if known_k is None or known_k >= costs.policy("gemv_k_unroll_long_min", 512):
        result = int(costs.policy("gemv_k_unroll_long", k_unroll))
        if result not in (1, 2, 4):
          raise ValueError("gemv_k_unroll_long must be 1, 2 or 4")
        return result
    return k_unroll

  def amx_traversal(product):
    # A tall product's small B remains resident across neighboring M tiles.
    # N-major walks its much larger A again for every output-column group.
    # Keep this target policy independent of cache blocking and packed ABI.
    maximum = int(costs.policy("amx_m_major_n_max", 0))
    ratio = int(costs.policy("amx_m_major_ratio", 4))
    if maximum <= 0:
      return I.literal(False)
    if ratio <= 0:
      raise ValueError("amx_m_major_ratio must be positive")
    m, n = product.type.shape
    selected = I.binary(
      "and",
      I.binary("le", n, I.literal(maximum)),
      I.binary("ge", I.binary("floordiv", m, I.literal(ratio)), n),
    )
    if product.data.tiled and I.constant(selected) is True:
      # A constant tall *problem* does not imply tall cache leaves: MC/NC
      # shortening often makes these panels nearly square and resident in L2.
      # Retain the cache driver's N-major order until a panel-level profitability
      # proof is available. Runtime traversal keeps its separately measured
      # coordinate-selection path; direct dot still gets whole-tile M-major.
      return I.literal(False)
    return selected

  def operand_policy(product, index):
    expression = product.args[index]
    summary = estimate(expression, costs, boundaries=prepared.vectors.kernel.materializations)
    cost = summary.compute_cycles
    if expression.op == "packed_load" or not cost:
      return I.OperandFusion(expression, None, None, cost, "direct/packed operand; no transform to recompute")
    packed = (
      product.data.tiled
      and product.data.packing != "none"
      and (product.data.packing == "auto" or ("A" if index == 0 else "B") in product.data.packing)
      and not any(I.constant(v) == 1 for v in product.type.shape)
    )
    packing_active = (
      I.binary(
        "and", I.binary("gt", product.type.shape[0], I.literal(1)), I.binary("gt", product.type.shape[1], I.literal(1))
      )
      if packed
      else I.literal(False)
    )
    if fusion == "always" or index == 1 and I.constant(packing_active) is True and fusion != "never":
      return I.OperandFusion(
        expression, None, None, cost, "fuse into loads / B packing outside MC: no duplicated B production"
      )
    reuse = product.type.shape[1 if index == 0 else 0]
    # Compare total transform throughput against temporary traffic/setup.
    # This is a continuous region cost, not a cheap/expensive opcode bucket.
    raw_condition = I.binary("gt", reuse, I.literal(int(recompute_limit(cost, costs))))
    if packed:
      panel = product.data.block_n or 512
      reuse = I.binary("floordiv", I.binary("add", reuse, I.literal(panel - 1)), I.literal(panel))
      packed_condition = (
        I.literal(False)
        if index == 1
        else I.binary("gt", reuse, I.literal(int(recompute_limit(cost, costs, packed=True))))
      )
      # Runtime GEMV bypasses packing. In particular, B is no longer produced
      # once outside MC when N=1, so an expensive B tail can be recomputed for
      # every row group. Select the matching cost model on the actual shape.
      condition = (
        packed_condition
        if I.constant(packing_active) is True
        else I.Expr(I.ScalarType("bool"), "where", (packing_active, packed_condition, raw_condition))
      )
    else:
      condition = raw_condition
    if fusion == "never":
      condition = I.literal(True)
    if I.constant(condition) is False:
      return I.OperandFusion(expression, None, None, cost, "few reloads: saved tile traffic pays for recomputation")
    name = f"dsl_o_dot_input_{len(plans)}_{index}"
    shape = expression.type.shape
    view = I.View(name, expression.type.dtype, shape, (shape[1], I.literal(1)), I.literal(0), owned=True)
    statement = I.Store(view, expression)
    stores[id(statement)] = lower_stores((statement,), manager=prepared.vectors.manager)
    transpose = plan_transpose(statement, context.options)
    if transpose is not None:
      transposes[id(statement)] = transpose
    return I.OperandFusion(
      expression, statement, condition, cost, "reload cost exceeds saved tile traffic; hoist input computation"
    )

  def visit(node):
    if isinstance(node, I.Node):
      if id(node) in seen:
        return
      seen.add(id(node))
      if isinstance(node, I.Pack):
        packed = node.destination
        spatial, k = packed.shape if packed.side == "A" else packed.shape[::-1]
        destination = replace(packed.view, shape=(spatial, k), strides=(k, I.literal(1)))
        memo = {}
        source = I.transpose_matrix(node.source, memo) if packed.side == "B" else node.source
        statement = I.Store(destination, source)
        stores[id(statement)] = lower_stores((statement,), manager=prepared.vectors.manager)
        transpose = plan_transpose(statement, context.options)
        if transpose is not None:
          transposes[id(statement)] = transpose
        aliases = tuple((load, memo[id(load)]) for load in I.loads(node.source) if id(load) in memo)
        dense_packs[id(node)] = (statement, aliases)
      if isinstance(node, I.Store):
        products = tuple(id(e) for e in walk(node.value) if e.op == "dot")
        if products:
          scalar = estimate(node.value, costs, boundaries=products, implementation="scalar")
          vector = estimate(node.value, costs, boundaries=products)
          overhead = costs.policy("gemv_epilogue_group_overhead", 8)
          if scalar.compute_cycles * gemv_columns > vector.compute_cycles + overhead:
            grouped_epilogues.add(id(node))
      if isinstance(node, I.Expr) and node.op == "dot":
        operands = tuple(operand_policy(node, index) for index in range(2))
        dense_transposes = []
        for index, expression in enumerate(node.args[:2]):
          if expression.op == "packed_load":
            dense_transposes.append(None)
            continue
          source = I.transpose_matrix(expression) if index == 1 else expression
          shape = source.type.shape
          destination = I.View(
            "dense_panel", source.type.dtype, shape, (shape[1], I.literal(1)), I.literal(0), owned=True
          )
          dense_transposes.append(
            plan_transpose(I.Store(destination, source, implementation="vector"), context.options)
          )
        plans[id(node)] = I.MatmulPlan(
          node,
          node.data,
          operands=operands,
          vector_columns=gemv_columns,
          k_unroll=product_k_unroll(node),
          unblocked_bytes=int(costs.policy("matmul_unblocked_bytes", 512 * 1024)),
          runtime_unblocked_bytes=int(costs.policy("matmul_runtime_unblocked_bytes", 128 * 1024)),
          amx_m_major=amx_traversal(node),
          amx_native_b_pack_min_m=int(costs.policy("amx_native_b_pack_min_m", 0)),
          amx_b_permutation=register_transpose(16, 16, 4),
          dense_transposes=tuple(dense_transposes),
        )
      for field in fields(node):
        visit(getattr(node, field.name))
    elif isinstance(node, (tuple, list)):
      for child in node:
        visit(child)

  visit(prepared.vectors.kernel.body)
  return replace(
    prepared,
    matmuls=MappingProxyType(plans),
    stores=MappingProxyType(stores),
    transposes=MappingProxyType(transposes),
    packing_permutation=register_transpose(16, 16, 4),
    packing_permutation_x86=x86_dword_transpose(),
    dense_packs=MappingProxyType(dense_packs),
    grouped_epilogues=frozenset(grouped_epilogues),
  )
