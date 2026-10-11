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
  serial, rewritten, sme_rewritten, native_rewritten = 0, set(), set(), set()
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
    wide_integer = (a_dtype in {"int16", "uint16"} and b_dtype in {"int16", "uint16"}
                    and (product.type.dtype == "int64" or product.type.dtype == "int32" and a_dtype == b_dtype))
    if (
      product.data.implementation == "amx"
      and not native_narrow
      or product.data.implementation == "sme"
      and not (
        native_narrow or wide_integer or a_dtype == b_dtype == product.type.dtype and product.type.dtype in {"float32", "float64"}
      )
    ):
      raise TypeError(f"unsupported {product.data.implementation} dot dtype combination")
    if product.type.dtype not in {"float32", "float64", "int32", "int64"}:
      return False
    if any(value.op == "dot" for arg in product.args for value in walk_unique(arg)):
      return False
    hint = product.data.implementation
    if hint in {"auto", "amx"} and native_narrow and target.architecture == "x86" and target.isa != "scalar":
      return "amx"
    if hint in {"auto", "sme"} and target.isa == "sme" and (
      native_narrow or wide_integer or a_dtype == b_dtype == product.type.dtype and product.type.dtype in {"float32", "float64"}
    ):
      return "sme"
    if hint in {"amx", "sme"}:
      raise TypeError(f"{hint} dot requires a matching compilation target")
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
            guard = ("defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || "
                     "(defined(HAS_FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))")
            if product.type.dtype == "float64":
              guard += " && defined(HAS_SME_F64F64)"
            if product.data.implementation == "auto":
              _, fallback = expand_helper(vector_dot_into, (product.args[:2], statement.destination),
                                           keyword_arguments={"acc_dtype": product.type.dtype},
                                           catalog=environment.operations, prefix=f"dsl_recipe_sme_fallback_{serial}")
            else:
              fallback = [I.Assert(I.literal(False), "sme dot instruction family is not enabled for this target")]
            body = [I.If(I.Expr(I.ScalarType("bool"), "compile_feature", data=guard), I.Block(body), I.Block(fallback))]
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
            if family in {"streaming", "blocked", "amx", "sme"}:
              native = family in {"amx", "sme"}
              if not native:
                kwargs.update(pipeline=pipeline, distance=distance)
              inputs = []
              for ordinal in range(2):
                operand, preamble = ((product.args[ordinal], []) if native else
                                    input_policy(product, ordinal, f"dsl_recipe_dot_{serial}_{ordinal}"))
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
              prefer_k = (
                I.literal(True)
                if product.args[1].op == "packed_load" and product.args[1].data.format == "dense"
                else I.literal(False)
                if primary is None
                else I.binary("eq", primary.data.strides[0], I.literal(1))
              )
              if not native:
                kwargs["prefer_k"] = prefer_k
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
              if statement.value is product:
                kwargs["epilogue"] = None
              arguments += (statement.destination,)
              if family == "blocked":
                kwargs.update(
                  BM=product.data.block_m,
                  BN=product.data.block_n,
                  BK=product.data.block_k,
                  packing=product.data.packing,
                )
            recipe = vector_tile_dot if family == "tile" else blocked_dot_into if family == "blocked" else vector_dot_into
            native_bindings = []
            if family in {"amx", "sme"}:
              from ...._library.amx_matmul import dot_into as amx_dot
              from ...._library.sme_matmul import dot_into as sme_dot

              recipe = amx_dot if family == "amx" else sme_dot
              kwargs.pop("unroll")
              # Broadcast-domain validation can throw. Evaluate such shape
              # metadata before entering an architectural state owner; its
              # bounded logical domain remains identical for projection.
              dimensions = (product.args[0].type.shape[0], product.args[1].type.shape[1], product.args[0].type.shape[1])
              if any(not environment.operations.has(v, Trait.HARDWARE_SCALAR)
                     and v.op not in {"floordiv", "mod"} for dim in dimensions for v in walk_unique(dim)):
                bound = []
                for index, dim in enumerate(dimensions):
                  name = f"dsl_recipe_dot_{serial}_dimension_{index}"
                  native_bindings.append(I.Assign(name, dim))
                  bound.append(I.Expr(I.ScalarType("int64"), "local", data=name))
                m, n, k = bound
                inputs = tuple(replace(v, type=replace(v.type, shape=shape)) for v, shape in
                               zip(arguments[0], ((m, k), (k, n))))
                arguments = (inputs, replace(arguments[1], shape=(m, n)))
              if product.data.tiled:
                from ...._library.matrix_recipe import blocked_matrix
                from ...frontend.capture import capture

                arguments += (capture(recipe.function),)
                recipe = blocked_matrix
                kwargs.update(BM=product.data.block_m, BN=product.data.block_n, BK=product.data.block_k,
                              packing=product.data.packing,
                              quantum=(64 // ELEMENT_BYTES[product.args[0].type.dtype] if family == "amx" else
                                       ELEMENT_BYTES[product.type.dtype] // ELEMENT_BYTES[product.args[0].type.dtype]),
                              span=I.literal(32) if family == "amx" else I.binary("mul", I.literal(2),
                                    I.Expr(I.ScalarType("int64"), "streaming_lanes", data=product.type.dtype)))
              elif all(isinstance(item, (I.BindTile, I.Assert, I.Assign, I.Allocate)) for item in output):
                a = product.args[0]
                b = product.args[1]
                raw = lambda v: v.op == "load" and not v.args and v.data.dtype == v.type.dtype and v.data.region is not None
                if family == "amx":
                  if raw(a):
                    kwargs["raw_a"] = a.data
                  if b.op == "packed_load" and b.data.format in {"amx", "auto"}:
                    view = b.data.view
                    count = I.Expr(I.ScalarType("int64"), "checked_shape", view.shape)
                    name = f"dsl_recipe_dot_{serial}_packed_b_capacity"
                    native_bindings.append(I.Assign(name, count))
                    capacity = I.Expr(I.ScalarType("int64"), "local", data=name)
                    kwargs["packed_b"] = replace(view, shape=(capacity,), strides=(I.literal(1),), region=None)
                elif raw(a) and raw(b) and a.type.dtype in {"float32", "float64"}:
                  kwargs.update(raw_a=a.data, raw_b=b.data)
            result, body = expand_helper(
              recipe,
              arguments,
              keyword_arguments=kwargs,
              catalog=environment.operations,
              prefix=f"dsl_recipe_dot_{serial}",
            )
            if family in {"amx", "sme"}:
              body = native_bindings + body
              dtype = product.args[0].type.dtype
              if family == "amx":
                feature = "HAS_AMX_FP16" if dtype == "float16" else "HAS_AMX_BF16" if dtype == "bfloat16" else "HAS_AMX_INT8"
                guard = f"defined(HAS_AMX_TILE) && defined({feature})"
              else:
                extra = "HAS_SME_F64F64" if dtype == "float64" else (
                  "HAS_SME_I16I64" if product.type.dtype == "int64" else "HAS_SME2"
                ) if dtype in {"int16", "uint16"} else None
                guard = ("defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || "
                         "(defined(HAS_FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))")
                guard += f" && defined({extra})" if extra else ""
              if product.data.implementation == "auto":
                def expand_fallback(prefix):
                  fallback_inputs, fallback_preamble = [], []
                  for ordinal in range(2):
                    operand, preparation = input_policy(product, ordinal, f"{prefix}_{ordinal}")
                    fallback_inputs.append(operand)
                    fallback_preamble.extend(preparation)
                  fallback_kwargs = {name: kwargs[name] for name in ("acc", "acc_dtype", "epilogue")}
                  fallback_kwargs.update(unroll=unroll, pipeline=pipeline, distance=distance, prefer_k=prefer_k)
                  if product.data.tiled:
                    fallback_kwargs.update(BM=product.data.block_m, BN=product.data.block_n,
                                           BK=product.data.block_k, packing=product.data.packing)
                  _, fallback = expand_helper(
                    blocked_dot_into if product.data.tiled else vector_dot_into,
                    (tuple(fallback_inputs), statement.destination),
                    keyword_arguments=fallback_kwargs,
                    catalog=environment.operations, prefix=prefix,
                  )
                  return fallback_preamble + fallback

                fallback = expand_fallback(f"dsl_recipe_dot_fallback_{serial}")
                # Measured AttentionBench decode/spec-decode shapes lose to
                # SIMD below one spatial tile: physical packing dominates.
                # Keep automatic small matrices on their original vector
                # recipe, including its producer caches. Explicit requests
                # still exercise matrix instructions and their fringes.
                # GCC 13's complete versioned kernel currently costs about
                # 2–3% on single-row AttentionBench decode versus the former
                # vector-only source. Unconditionally selecting AMX was ~8x
                # slower there. On 920f-4, some decode pairs measured up to
                # ~4.5% overhead. Retain this gate and producer-cache policy
                # until a replacement passes paired shape/meta benchmarks;
                # deleting the gate is not a valid performance fix.
                use_native = I.binary("and", I.binary("ge", product.type.shape[0], I.literal(16)),
                                      I.binary("ge", product.type.shape[1], I.literal(16)))
                body = [I.If(use_native, I.Block(body), I.Block(expand_fallback(f"dsl_recipe_dot_small_{serial}")))]
              else:
                fallback = [I.Assert(I.literal(False), f"{family} dot instruction family is not enabled for this target")]
              body = [I.If(I.Expr(I.ScalarType("bool"), "compile_feature", data=guard), I.Block(body), I.Block(fallback))]
              ir.diagnostics.append(f"staged {family.upper()} matrix instruction recipe from DSL library")
              native_rewritten.add(id(product))
            if family != "blocked" and all(
              type(I.constant(v)) is int for v in (*product.type.shape, product.args[0].type.shape[1])
            ):
              body = simplify_loop_metadata(I.Block(body)).statements
            output.extend(body)
            rewritten.add(id(product))
            if family in {"streaming", "blocked", "amx", "sme"}:
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
    vector_count = len(rewritten - sme_rewritten - native_rewritten)
    if vector_count:
      ir.diagnostics.append(f"staged {vector_count} vector dot leaf/leaves from DSL library")
    if sme_rewritten:
      ir.diagnostics.append(f"staged {len(sme_rewritten)} direct SME dot leaf/leaves from DSL library")
  return PassResult(ir, changed=bool(rewritten))
