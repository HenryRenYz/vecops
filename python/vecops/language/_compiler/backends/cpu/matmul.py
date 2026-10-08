# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU matrix-product policy; no opaque ops::Matmul forwarding."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from ... import ir as I
from ...analysis.affine import bind_instruction_metadata, simplify_loop_metadata
from ...cost import estimate
from ...ir.traversal import walk
from ...lowering.tile_to_vector import lower_stores
from ...operations import Trait
from ...passes.manager import register_pass
from .cost.sme import NativeAPackingPolicy, PackedABTraversalPolicy
from .policies import recompute_limit
from .registers import register_transpose, x86_dword_transpose
from .transpose import plan_transpose


@dataclass(frozen=True)
class CPUMatmulPlan:
  """Physical CPU family policies; none are universal logical-dot semantics."""

  expression: object
  options: I.DotOptions
  vector_rows: int = 4
  vector_columns: int = 4
  k_unroll: int = 2
  unblocked_bytes: int = 512 * 1024
  runtime_unblocked_bytes: int = 128 * 1024
  amx_m_major: object | None = None
  amx_native_b_pack_min_m: int = 0
  amx_cache_b_panel_divisor: int = 0
  amx_cache_b_panel_max_m: int = 0
  amx_shared_cache_dispatch: bool = False
  # Experimental SVE quantized/prepacked-B driver, still profitability-gated.
  sve_runtime_quant: bool = False
  # Evaluated against actual panel extents and captured SME streaming length.
  sme_native_a_packing: object = None
  sme_packed_ab_cache_bypass: bool = False
  sme_packed_ab_traversal: object = None
  # ZA keeps native geometry even for narrowed singleton scalar/Vec traffic.
  sme_narrow_singleton_slices: bool = False
  operands: tuple = ()
  dense_transposes: tuple = ()
  schedule: object | None = None
  pass_trace: tuple = ("cpu-plan-matmul",)


def runtime_quant_candidate(a, b, accumulator_dtype, operations):
  """Recognize a lane-local FP32 -> U8 DAG, never a helper/function name.

  The ordinary-SVE leaf consumes four adjacent logical K values and a pure
  SME-packed signed-byte B. It must not cross reductions, register shuffles,
  or stateful/materialized producer boundaries. Broadcasting and alias
  snapshots remain the ordinary vector emitter's responsibility.
  """
  if (
    accumulator_dtype != "int32"
    or a.type.dtype != "uint8"
    or a.op != "cast"
    or a.args[0].type.dtype != "float32"
    or len(a.type.shape) != 2
    or b.type.dtype != "int8"
    or b.op != "packed_load"
    or b.data.side != "B"
    or b.data.format not in {"sme", "auto"}
  ):
    return False
  primary = False
  for value in walk(a):
    if not isinstance(value.type, I.TileType):
      continue
    if value is not a and value.type.dtype not in {"float32", "int32", "bool"}:
      # Four FP32/I32 lanes fit the architectural minimum ordinary VL.
      # Wider or staged narrow conversions retain the general legalizer;
      # this family must never hide a multiword fixed descriptor in C++.
      return False
    if len(value.type.shape) > 2:
      return False
    if value.op == "load":
      if value.args or value.data.dtype not in {"float32", "int32"}:
        return False
      primary |= (
        value.data.dtype == "float32"
        and len(value.type.shape) == 2
        and all(I.same_expr(x, y) for x, y in zip(value.type.shape, a.type.shape))
      )
    elif not operations.has(value, Trait.ELEMENTWISE):
      return False
  return primary


@register_pass("cpu-plan-matmul", stage="cpu-memory", after=("cpu-plan-transpose",))
def plan_matmul(prepared, context):
  plans, dense_packs, seen, grouped_epilogues = {}, {}, set(), set()
  costs = context.environment.costs
  stores, transposes = dict(prepared.stores), dict(prepared.transposes)
  fusion = context.options.get("cpu_matmul_fusion", "auto")
  if fusion not in {"auto", "always", "never"}:
    raise ValueError("cpu_matmul_fusion must be auto, always or never")
  runtime_quant = context.options.get("cpu_sve_runtime_quant", False)
  if type(runtime_quant) is not bool:
    raise ValueError("cpu_sve_runtime_quant must be bool")
  amx_shared_cache = context.options.get("cpu_amx_shared_cache_dispatch", False)
  if type(amx_shared_cache) is not bool:
    raise ValueError("cpu_amx_shared_cache_dispatch must be bool")
  runtime_quant_available = (
    runtime_quant
    and fusion != "never"
    and context.environment.codegen_target.architecture != "x86"
    and context.environment.codegen_target.isa != "scalar"
  )
  sme_native_a_packing = NativeAPackingPolicy.from_costs(costs)
  packed_ab_default = costs.policy("sme_packed_ab_cache_bypass", 0)
  if type(packed_ab_default) is not int or packed_ab_default not in (0, 1):
    raise ValueError("sme_packed_ab_cache_bypass policy must be 0 or 1")
  sme_packed_ab_cache_bypass = context.options.get("cpu_sme_packed_ab_cache_bypass", bool(packed_ab_default))
  if type(sme_packed_ab_cache_bypass) is not bool:
    raise ValueError("cpu_sme_packed_ab_cache_bypass must be bool")
  sme_packed_ab_traversal = PackedABTraversalPolicy.from_costs(costs)
  sme_narrow_singleton_slices = context.options.get("cpu_sme_narrow_singleton_slices", False)
  if type(sme_narrow_singleton_slices) is not bool:
    raise ValueError("cpu_sme_narrow_singleton_slices must be bool")
  gemv_columns = context.options.get("cpu_gemv_columns", 4)
  if type(gemv_columns) is not int or not 1 <= gemv_columns <= 8:
    raise ValueError("cpu_gemv_columns must be an integer in [1,8]")
  k_unroll = context.options.get("cpu_gemv_k_unroll", int(costs.policy("gemv_k_unroll", 2)))
  if type(k_unroll) is not int or k_unroll not in (1, 2, 4):
    raise ValueError("cpu_gemv_k_unroll must be 1, 2 or 4")
  b_panel_divisor = costs.policy("amx_cache_b_panel_divisor", 0)
  if type(b_panel_divisor) is not int or b_panel_divisor < 0:
    raise ValueError("amx_cache_b_panel_divisor must be a nonnegative integer")
  b_panel_max_m = costs.policy("amx_cache_b_panel_max_m", 0)
  if type(b_panel_max_m) is not int or b_panel_max_m < 0:
    raise ValueError("amx_cache_b_panel_max_m must be a nonnegative integer")

  def product_k_unroll(product):
    known_k = I.constant(product.args[0].type.shape[1])
    # Unbounded K keeps the conservative latency-hiding bank count.
    if "cpu_gemv_k_unroll" not in context.options and (
      known_k is None or known_k >= costs.policy("gemv_k_unroll_long_min", 512)
    ):
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
        plans[id(node)] = CPUMatmulPlan(
          node,
          node.data,
          operands=operands,
          vector_columns=gemv_columns,
          k_unroll=product_k_unroll(node),
          unblocked_bytes=int(costs.policy("matmul_unblocked_bytes", 512 * 1024)),
          runtime_unblocked_bytes=int(costs.policy("matmul_runtime_unblocked_bytes", 128 * 1024)),
          amx_m_major=amx_traversal(node),
          amx_native_b_pack_min_m=int(costs.policy("amx_native_b_pack_min_m", 0)),
          amx_cache_b_panel_divisor=b_panel_divisor,
          amx_cache_b_panel_max_m=b_panel_max_m,
          amx_shared_cache_dispatch=amx_shared_cache,
          sve_runtime_quant=runtime_quant_available,
          sme_native_a_packing=sme_native_a_packing,
          sme_packed_ab_cache_bypass=sme_packed_ab_cache_bypass,
          sme_packed_ab_traversal=sme_packed_ab_traversal,
          sme_narrow_singleton_slices=sme_narrow_singleton_slices,
          dense_transposes=tuple(dense_transposes),
        )
      for field in fields(node):
        visit(getattr(node, field.name))
    elif isinstance(node, (tuple, list)):
      for child in node:
        visit(child)

  visit(prepared.vectors.kernel.body)
  packing_programs = {}
  staged_packing = context.options.get("cpu_sme_packing_dsl", False)
  if type(staged_packing) is not bool:
    raise TypeError("cpu_sme_packing_dsl must be bool")
  if staged_packing:
    from ...._library.sme_packing import pack_raw
    from ...frontend.helpers import expand_helper
    from ...ir.traversal import nodes

    graph = tuple(nodes(prepared.vectors.kernel.body))
    dtypes = {node.destination.view.dtype for node in graph if isinstance(node, I.Pack)}
    for node in graph:
      if isinstance(node, I.Expr) and node.op == "dot" and node.data.tiled:
        dtypes.update(operand.type.dtype for operand in node.args[:2])
    for dtype in sorted(dtypes):
      if dtype == "bool":
        continue

      def raw(name):
        return I.Expr(I.ScalarType("int64"), "raw", data="sme_packing." + name)

      source = I.View(
        "sme_packing_input", dtype, (raw("m"), raw("k")), (raw("pitch"), I.literal(1)), I.literal(0), writable=False
      )
      destination = I.View("sme_packing_output", dtype, (raw("capacity"),), (I.literal(1),), I.literal(0))
      _, body = expand_helper(
        pack_raw,
        (source, destination),
        catalog=context.environment.operations,
        prefix="sme_packing_" + dtype,
        hardware_kind="sme",
      )
      packing_programs[dtype] = bind_instruction_metadata(
        simplify_loop_metadata(I.Block(body)), prefix="sme_packing_metadata_" + dtype
      )
  return replace(
    prepared,
    matmuls=MappingProxyType(plans),
    stores=MappingProxyType(stores),
    transposes=MappingProxyType(transposes),
    # Only matrix/packing emission consumes these closed register recipes.
    # Pure elementwise/reduction kernels must not parse unrelated algorithms;
    # all products share the same prepared permutation instead of retaining
    # an unused duplicate in every family plan.
    packing_permutation=register_transpose(16, 16, 4) if plans or dense_packs else None,
    packing_permutation_x86=x86_dword_transpose() if plans or dense_packs else None,
    dense_packs=MappingProxyType(dense_packs),
    grouped_epilogues=frozenset(grouped_epilogues),
    sme_packing_programs=MappingProxyType(packing_programs),
  )
