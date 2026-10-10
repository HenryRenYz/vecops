# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Specialize source-parsed library leaves before ordinary kernel/vector passes.

Adapters select a measured legal region, not its arithmetic or instruction
schedule. The recipe's control, state and producer/consumer DAGs then use the
same optimization/legalization pipeline as user-written DSL code.
"""

from dataclasses import replace

from ... import ir as I
from ...analysis.affine import simplify_loop_metadata
from ...analysis.projection import pointwise_row_projection
from ...cost import estimate
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import ValueFunction, expand_helper
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ...passes.manager import PassResult, register_pass
from .policies import recompute_limit


@register_pass("cpu-expand-dot-recipes", after="cpu-stage-matrix-values", before="fuse-elementwise")
def expand_dot_recipes(ir, context):
  from ...._library.matmul import blocked_dot_into, vector_dot_into, vector_tile_dot

  environment = context.environment
  target = environment.codegen_target
  vector_state = environment.vector_state
  native_bytes = vector_state.sve_bytes or 16 if target.isa in {"sve", "sme"} else (target.vector_bits or 128) // 8
  budget = context.options.get("cpu_state_register_budget", 12)
  serial, rewritten, sme_rewritten = 0, set(), set()
  fusion = context.options.get("cpu_matmul_fusion", "auto")
  if fusion not in {"auto", "always", "never"}:
    raise ValueError("cpu_matmul_fusion must be auto, always or never")
  pipeline = context.options.get("cpu_matmul_pipeline", "auto")
  if pipeline not in {"auto", "off", "prefetch", "staged"}:
    raise ValueError("cpu_matmul_pipeline must be auto, off, prefetch or staged")
  if pipeline == "auto":
    policy = environment.costs.policy("matmul_pipeline", 0)
    if policy not in (0, 1, 2):
      raise ValueError("target matmul_pipeline policy must be 0, 1 or 2")
    pipeline = ("off", "prefetch", "staged")[int(policy)]
  distance = context.options.get("cpu_matmul_prefetch_distance", 1)
  if type(distance) is not int or not 1 <= distance <= 16:
    raise ValueError("cpu_matmul_prefetch_distance must be in [1,16]")
  unroll = context.options.get("cpu_gemv_k_unroll", 2)
  if type(unroll) is not int or unroll not in (1, 2, 4):
    raise ValueError("cpu_gemv_k_unroll must be 1, 2 or 4")

  def input_policy(product, ordinal, prefix):
    expression = product.args[ordinal]
    cost = estimate(expression, environment.costs, boundaries=ir.materializations).compute_cycles
    if not cost or fusion == "always":
      return expression, []
    condition = (
      I.literal(True)
      if fusion == "never"
      else I.binary("gt", product.type.shape[1 - ordinal], I.literal(int(recompute_limit(cost, environment.costs))))
    )
    if I.constant(condition) is False:
      return expression, []
    shape = expression.type.shape
    view = I.View(
      prefix + "_input_cache", expression.type.dtype, shape, (shape[1], I.literal(1)), I.literal(0), owned=True
    )
    body = I.Block([I.Allocate(view), I.Store(view, expression)])
    if I.constant(condition) is True:
      return I.Expr(expression.type, "load", data=view), body.statements
    flag = prefix + "_cache_enabled"
    enabled = I.Expr(I.ScalarType("bool"), "local", data=flag)
    return I.Expr(expression.type, "cached_value", (expression,), (view, enabled)), [
      I.Assign(flag, condition),
      I.If(enabled, body, I.Block([])),
    ]

  def eligible(product):
    a_dtype, b_dtype = (value.type.dtype for value in product.args[:2])
    native_narrow = (
      a_dtype == b_dtype
      and a_dtype in {"float16", "bfloat16"}
      and product.type.dtype == "float32"
      or a_dtype in {"int8", "uint8"}
      and b_dtype in {"int8", "uint8"}
      and product.type.dtype == "int32"
    )
    if (
      product.data.implementation == "amx"
      and not native_narrow
      or product.data.implementation == "sme"
      and not (
        native_narrow or a_dtype == b_dtype == product.type.dtype and product.type.dtype in {"float32", "float64"}
      )
    ):
      raise TypeError(f"unsupported {product.data.implementation} dot dtype combination")
    if product.type.dtype not in {"float32", "float64", "int32"}:
      return False
    if any(value.op == "dot" for arg in product.args for value in walk_unique(arg)):
      return False
    if product.data.tiled:
      return "blocked"
    if pipeline != "off":
      return "streaming"
    m, n = map(I.constant, product.type.shape)
    widest = max(
      ELEMENT_BYTES[value.type.dtype] for value in walk_unique(product) if isinstance(value.type, I.TileType)
    )
    lanes = max(1, native_bytes // widest)
    return (
      "tile"
      if type(m) is int and 0 < m <= 4 and type(n) is int and 0 < n <= lanes and m * n.bit_count() <= budget
      else "streaming"
    )

  def replace_product(value, product, result, memo):
    if value is product:
      return result
    if id(value) not in memo:
      args = tuple(replace_product(arg, product, result, memo) for arg in value.args)
      memo[id(value)] = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
    return memo[id(value)]

  def packed_checks(product, destination):
    """Packed ABI validation precedes all source-algorithm reads/writes.

    Packing offsets remain a physical-layout primitive. A bare T.packed
    declaration is not evidence of capacity, density or nonaliasing.
    """
    output, seen = [], set()
    for operand in product.args:
      for value in walk_unique(operand):
        if value.op != "packed_load" or id(value.data) in seen:
          continue
        seen.add(id(value.data))
        packed, expected = value.data, I.literal(1)
        view = packed.view
        output.append(
          I.Assert(
            I.binary("ge", I.Expr(I.ScalarType("int64"), "checked_shape", view.shape), I.literal(0)),
            "invalid packed buffer shape",
          )
        )
        for extent, stride in zip(reversed(view.shape), reversed(view.strides)):
          output.append(I.Assert(I.binary("eq", stride, expected), "packed input must be dense"))
          expected = I.binary("mul", expected, extent)
        spatial, k = packed.shape if packed.side == "A" else packed.shape[::-1]
        required = I.Expr(I.ScalarType("int64"), "packing_size", (spatial, k), (view.dtype, packed.format))
        # The ABI decoder is disabled if the physical view is not a valid
        # dense buffer. Shape metadata is nonnegative by the tensor schema;
        # the packing-size helper additionally checks byte-count overflow.
        output.append(I.Assert(I.binary("ge", expected, required), "packed input buffer too small"))
        if not view.owned:
          output.append(
            I.Assert(
              I.Expr(I.ScalarType("bool"), "view_disjoint", data=(view, destination)),
              "packed input and matrix output may not overlap",
            )
          )
    return output

  # A register result is scoped to its sink. Do not replace a product consumed
  # by another sink or a nonlocal/state/conditional consumer with that local.
  consumers = {}
  for statement in nodes(ir.body):
    if isinstance(statement, I.BindTile):
      continue
    for name in ("value", "source", "condition", "mask"):
      value = getattr(statement, name, None)
      if isinstance(value, I.Expr):
        for product in walk_unique(value):
          if product.op == "dot":
            consumers.setdefault(id(product), set()).add(id(statement))

  def visit(block, *, top=False):
    nonlocal serial
    output = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      if isinstance(statement, I.Store) and statement.mask is None:
        products = {id(value): value for value in walk_unique(statement.value) if value.op == "dot"}
        if len(products) == 1:
          product = next(iter(products.values()))
          operands = {id(value) for arg in product.args for value in walk_unique(arg)}
          epilogue = all(
            value is product
            or environment.operations.has(value, Trait.ELEMENTWISE | Trait.MEMORY_READ)
            or isinstance(value.type, I.ScalarType)
            for value in walk_unique(statement.value)
            if id(value) not in operands
          )
          family = eligible(product)
          raw_sme = (
            top
            and target.architecture == "arm"
            and target.isa == "sme"
            and not product.data.tiled
            and product.data.implementation in {"auto", "sme"}
            and product.type.dtype in {"float32", "float64"}
            and len(product.args) == 2
            and statement.value is product
            and I.constant(statement.destination.strides[-1]) == 1
            and statement.destination.dtype == product.type.dtype
            and all(
              operand.op == "load"
              and not operand.args
              and operand.data.dtype == operand.type.dtype == product.type.dtype
              and operand.data.region is not None
              for operand in product.args
            )
            # Passing views to a raw instruction recipe must not turn an old
            # captured read into a fresh read after any possible earlier write.
            # More general captures use the existing identity-bearing path.
            and all(isinstance(item, (I.BindTile, I.Assert, I.Assign)) for item in output)
          )
          if raw_sme and consumers[id(product)] == {id(statement)}:
            from ...._library.sme_matmul import direct_dot

            serial += 1
            output.append(
              I.Assert(
                I.Expr(I.ScalarType("bool"), "view_nonoverlap", data=statement.destination),
                "matrix output layout overlaps",
              )
            )
            _, body = expand_helper(
              direct_dot,
              (*[operand.data for operand in product.args], statement.destination),
              catalog=environment.operations,
              prefix=f"dsl_recipe_sme_dot_{serial}",
            )
            output.extend(body)
            rewritten.add(id(product))
            sme_rewritten.add(id(product))
            continue
          if family and epilogue:
            serial += 1
            output.append(
              I.Assert(
                I.Expr(I.ScalarType("bool"), "view_nonoverlap", data=statement.destination),
                "matrix output layout overlaps",
              )
            )
            output.extend(packed_checks(product, statement.destination))
            kwargs = {"acc": product.args[2] if len(product.args) > 2 else None, "acc_dtype": product.type.dtype}
            kwargs["unroll"] = unroll
            arguments = (product.args[:2],)
            if family in {"streaming", "blocked"}:
              kwargs.update(pipeline=pipeline, distance=distance)
              inputs = []
              for ordinal in range(2):
                operand, preamble = input_policy(product, ordinal, f"dsl_recipe_dot_{serial}_{ordinal}")
                output.extend(preamble)
                inputs.append(operand)
              arguments = (tuple(inputs),)
              primary = next(
                (
                  load
                  for load in I.loads_unique(product.args[1])
                  if len(load.type.shape) == 2
                  and all(I.same_expr(a, b) for a, b in zip(load.type.shape, product.args[1].type.shape))
                ),
                None,
              )
              kwargs["prefer_k"] = (
                I.literal(True)
                if product.args[1].op == "packed_load" and product.args[1].data.format == "dense"
                else I.literal(False)
                if primary is None
                else I.binary("eq", primary.data.strides[0], I.literal(1))
              )
              extent = I.Expr(I.ScalarType("int64"), "region_argument", data="prefix")
              reference = I.Expr(
                I.VectorType(product.type.dtype, (extent,), tag=I.VectorTag(basis_dtype=product.type.dtype)),
                "region_argument",
                data="result",
              )
              row, column = (I.Expr(I.ScalarType("int64"), "region_argument", data=name) for name in ("row", "column"))
              kwargs["epilogue"] = ValueFunction(
                (reference, row, column),
                pointwise_row_projection(
                  statement.value, product, reference, row, column, extent, environment.operations
                ),
              )
              arguments += (statement.destination,)
              if family == "blocked":
                kwargs.update(
                  BM=product.data.block_m,
                  BN=product.data.block_n,
                  BK=product.data.block_k,
                  packing=product.data.packing,
                )
            result, body = expand_helper(
              vector_tile_dot if family == "tile" else blocked_dot_into if family == "blocked" else vector_dot_into,
              arguments,
              keyword_arguments=kwargs,
              catalog=environment.operations,
              prefix=f"dsl_recipe_dot_{serial}",
            )
            if family != "blocked" and all(
              type(I.constant(v)) is int for v in (*product.type.shape, product.args[0].type.shape[1])
            ):
              body = simplify_loop_metadata(I.Block(body)).statements
            output.extend(body)
            rewritten.add(id(product))
            if family in {"streaming", "blocked"}:
              continue
            statement = replace(statement, value=replace_product(statement.value, product, result, {}))
      output.append(statement)
    block.statements = output

  visit(ir.body, top=True)

  # Keep original lazy-load provenance, but discard unused opaque products.
  def provenance(block):
    output = []
    for statement in block.statements:
      for child in children(statement):
        provenance(child)
      if isinstance(statement, I.BindTile) and any(id(value) in rewritten for value in walk_unique(statement.value)):
        output.extend(I.BindTile(load) for load in I.loads_unique(statement.value))
      else:
        output.append(statement)
    block.statements = output

  provenance(ir.body)
  if rewritten:
    # This adapter follows epilogue discovery, after the first layout fold.
    # Normalize newly staged windows with that same generic pass, rather than
    # teach the terminal emitter a recipe-specific projection algorithm.
    from ...passes.kernel.transforms import fold_value_transforms

    ir = fold_value_transforms(ir, context)
    if len(rewritten) > len(sme_rewritten):
      ir.diagnostics.append(f"staged {len(rewritten) - len(sme_rewritten)} vector dot leaf/leaves from DSL library")
    if sme_rewritten:
      ir.diagnostics.append(f"staged {len(sme_rewritten)} direct SME dot leaf/leaves from DSL library")
  return PassResult(ir, changed=bool(rewritten))
