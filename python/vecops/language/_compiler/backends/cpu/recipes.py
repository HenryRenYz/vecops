# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Specialize source-parsed library leaves before ordinary kernel/vector passes.

Adapters select a measured legal region, not its arithmetic or instruction
schedule. The recipe's control, state and producer/consumer DAGs then use the
same optimization/legalization pipeline as user-written DSL code.
"""

from copy import deepcopy
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
from .packed_views import aligned_packed_view
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

  native_schedule = context.options.get("cpu_matmul_native_schedule", "default")
  outline_vector_fallback = context.options.get("cpu_outline_vector_fallback", target.isa == "sme")
  if type(outline_vector_fallback) is not bool:
    raise TypeError("cpu_outline_vector_fallback must be bool")
  if native_schedule not in {"default", "shared-owner", "prefetch", "stream-b", "paired-fringe"}:
    raise ValueError("cpu_matmul_native_schedule must be default, shared-owner, prefetch, stream-b or paired-fringe")
  if native_schedule in {"shared-owner", "prefetch"} and target.isa != "sme":
    raise ValueError("native shared-owner/prefetch requires SME")
  if native_schedule == "stream-b" and target.architecture != "x86":
    raise ValueError("native stream-b requires x86")

  packed_dot = context.options.get("cpu_packed_dot", False)
  if type(packed_dot) is not bool:
    raise TypeError("cpu_packed_dot must be bool")

  runtime_quant = context.options.get("cpu_runtime_quant_int8", False)
  if type(runtime_quant) is not bool:
    raise TypeError("cpu_runtime_quant_int8 must be bool")

  def quantized_affine(product):
    """Match the public FP32 row-affine -> U8 boundary without rewriting it."""
    if not runtime_quant or target.isa != "sme" or product.type.dtype != "int32":
      return False
    a, b = product.args[:2]
    if (a.op != "cast" or a.type.dtype != "uint8" or b.op != "packed_load" or
        b.type.dtype != "int8" or b.data.format not in {"sme", "auto"} or b.data.origins or
        product.data.implementation != "auto" or product.data.tiled and (
          product.data.packing != "auto" or any(v is not None for v in
            (product.data.block_m, product.data.block_n, product.data.block_k)))):
      return False
    value = a.args[0]
    if value.op == "max" and I.constant(value.args[0]) == 0:
      # Recognize the public quantize_per_row_u8 helper's exact NaN/clamp
      # wrapper as well. This only classifies the schedule: quant_dot still
      # evaluates the original producer, including every selection and cast.
      upper = value.args[1]
      if upper.op != "min" or I.constant(upper.args[0]) != 255:
        return False
      selected = upper.args[1]
      if selected.op != "where" or I.constant(selected.args[1]) != 0:
        return False
      condition, _, affine = selected.args
      if condition.op != "gt" or I.constant(condition.args[1]) != 2139095040:
        return False
      masked = condition.args[0]
      if masked.op != "bit_and" or I.constant(masked.args[1]) != 2147483647:
        return False
      bits = masked.args[0]
      if bits.op != "bitreinterpret" or bits.type.dtype != "uint32" or not I.same_expr(bits.args[0], affine):
        return False
      value = affine
    if value.op != "add" or value.type.dtype != "float32" or value.args[0].op != "mul":
      return False
    data, scale = value.args[0].args
    if scale.op == "cast" and scale.type.dtype == "float32" and scale.args[0].type.dtype == "float32":
      scale = scale.args[0]
    if (data.op != "load" or data.args or data.type.dtype != "float32" or data.data.dtype != "float32"
        or scale.type.dtype != "float32"):
      return False
    shape = getattr(scale.type, "shape", ())
    zero = value.args[1]
    if zero.op == "cast" and zero.type.dtype == "float32":
      zero = zero.args[0]
    zero_shape = getattr(zero.type, "shape", ())
    return (len(shape) == 2 and I.constant(shape[1]) == 1 and
            I.same_expr(shape[0], a.type.shape[0]) and scale.op == "load" and not scale.args and
            zero.op == "load" and not zero.args and zero.data.dtype == "int32" and
            zero.type.dtype in {"int32", "float32"} and
            all(I.constant(v) == 1 for v in zero_shape) and
            (len(product.args) < 3 or product.args[2].type.dtype == "int32"))

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

  def statically_small(product):
    # Match the automatic native gate before expanding either recipe. One
    # known-small extent suffices even when the other extent is dynamic.
    return I.constant(automatic_native_condition(product)) is False

  def automatic_native_condition(product):
    """Cost the logical problem and physical operand contract together.

    Raw BF16 decode retains its grouped-dot recipe. Native packed inputs
    avoid the expensive logical lane reconstruction; byte products amortize
    instruction ownership once they fill a meaningful reduction footprint.
    Divide instead of multiplying dynamic extents to avoid signed overflow.
    """
    def conjunction(*conditions):
      # IR short-circuit folding is left-biased. A known-small second axis
      # must prune the owner just as a known-small first axis does.
      if any(I.constant(condition) is False for condition in conditions):
        return I.literal(False)
      result = I.literal(True)
      for condition in conditions:
        result = I.binary("and", result, condition)
      return result

    m, n = product.type.shape
    k = product.args[0].type.shape[1]
    large = conjunction(I.binary("ge", m, I.literal(16)), I.binary("ge", n, I.literal(16)))
    a, b = product.args[:2]
    narrow = (a.type.dtype == b.type.dtype and a.type.dtype in {"bfloat16", "float16"}
              and product.type.dtype == "float32") or (
      a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"}
      and product.type.dtype == "int32")
    family = "amx" if target.architecture == "x86" else "sme"
    sites = product_sites.get(id(product), ())
    physical_operands = [value.op == "packed_load" and not value.data.origins and
                         value.data.format in {family, "auto"} and sites and
                         all(current_physical_read(value, block, index) for block, index in sites)
                         for value in (a, b)]
    physical = any(physical_operands)
    byte = (a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"}
            and not any(value.op == "packed_load" for operand in (a, b) for value in walk_unique(operand)))
    if not narrow or not (physical or byte):
      return large
    # Raw small SME products pay online packing costs that are not captured
    # by MNK alone. Keep the established vector domain until the native
    # packing schedule wins with the actual source SDK (including FA64).
    if not physical and target.architecture != "x86":
      return large
    threshold = 512 if physical else 16384
    footprint = I.binary("gt", k, I.binary("floordiv",
                         I.binary("floordiv", I.literal(threshold - 1), I.binary("max", I.literal(1), m)),
                         I.binary("max", I.literal(1), n)))
    footprint = conjunction(I.binary("ge", k, I.literal(32)), footprint,
                            I.binary("gt", m, I.literal(0)), I.binary("gt", n, I.literal(0)))
    if not physical:
      # Raw byte GEMV often wins with grouped SIMD even at long K. Retain it
      # while enabling the measured small-M reuse domain (including batch folds).
      footprint = conjunction(footprint, I.binary("ge", m, I.literal(4)), I.binary("ge", n, I.literal(16)))
    return I.binary("or", large, footprint)

  def tiny_unconfigured(product):
    # Constant tiny products need no cache traversal. Keep explicit blocking
    # and packing authoritative, and leave empty/large/dynamic domains on
    # their existing wrapper rather than expanding both sides speculatively.
    if (
      not product.data.tiled
      or product.data.packing != "auto"
      or any(v is not None for v in (product.data.block_m, product.data.block_n, product.data.block_k))
    ):
      return False
    m, n, k = map(I.constant, (*product.type.shape, product.args[0].type.shape[1]))
    return (
      all(type(v) is int and v > 0 for v in (m, n, k))
      and m * n <= 256
      and k <= 256
      and k * (m * ELEMENT_BYTES[product.args[0].type.dtype] + n * ELEMENT_BYTES[product.args[1].type.dtype]) <= 32768
    )

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
    wide_integer = (
      a_dtype in {"int16", "uint16"}
      and b_dtype in {"int16", "uint16"}
      and (product.type.dtype == "int64" or product.type.dtype == "int32" and a_dtype == b_dtype)
    )
    if (
      product.data.implementation == "amx"
      and not native_narrow
      or product.data.implementation == "sme"
      and not (
        native_narrow
        or wide_integer
        or a_dtype == b_dtype == product.type.dtype
        and product.type.dtype in {"float32", "float64"}
      )
    ):
      raise TypeError(f"unsupported {product.data.implementation} dot dtype combination")
    if product.type.dtype not in {"float32", "float64", "int32", "int64"}:
      return False
    if any(value.op == "dot" for arg in product.args for value in walk_unique(arg)):
      return False
    hint = product.data.implementation
    native_auto = hint == "auto" and not statically_small(product)
    if (native_auto or hint == "amx") and native_narrow and target.architecture == "x86" and target.isa != "scalar":
      return "amx"
    if (
      (native_auto or hint == "sme")
      and target.isa == "sme"
      and (
        native_narrow
        or wide_integer
        or a_dtype == b_dtype == product.type.dtype
        and product.type.dtype in {"float32", "float64"}
      )
    ):
      return "sme"
    if hint in {"amx", "sme"}:
      raise TypeError(f"{hint} dot requires a matching compilation target")
    if (
      hint == "auto"
      and statically_small(product)
      and not product.data.tiled
      and (
        native_narrow
        and target.architecture == "x86"
        and target.isa != "scalar"
        or target.isa == "sme"
        and (
          native_narrow
          or wide_integer
          or a_dtype == b_dtype == product.type.dtype
          and product.type.dtype in {"float32", "float64"}
        )
      )
    ):
      return "streaming"
    if product.data.tiled:
      return "streaming" if tiny_unconfigured(product) else "blocked"
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

  # Physical input bindings bypass the logical capture DAG. A write before
  # a fresh capture (notably online packing) is harmless; one after that
  # capture may alias any caller buffer and requires the snapshot decoder.
  # Keep lexical owners so a capture outside a loop/branch is never mistaken
  # for a fresh read inside it. Unknown effects remain conservative barriers.
  captures, lexical_blocks, product_sites = {}, {}, {}

  def collect_captures(block):
    lexical_blocks[id(block)] = tuple(block.statements)
    for index, statement in enumerate(block.statements):
      if isinstance(statement, I.Store):
        for product in walk_unique(statement.value):
          if product.op == "dot":
            product_sites.setdefault(id(product), []).append((block, index))
      if isinstance(statement, I.BindTile):
        for load in walk_unique(statement.value):
          if load.op != "packed_load" and not environment.operations.has(load, Trait.MEMORY_READ):
            continue
          captures.setdefault(id(load), (id(block), index))
      for child in children(statement):
        collect_captures(child)

  collect_captures(ir.body)

  def current_physical_read(operand, block, index):
    for load in walk_unique(operand):
      if load.op != "packed_load" and not environment.operations.has(load, Trait.MEMORY_READ):
        continue
      capture = captures.get(id(load))
      if capture is None:
        continue  # Inline producer: evaluated at this consumer.
      owner, first = capture
      if owner != id(block) or any(
        not isinstance(previous, (I.BindTile, I.Assert, I.Assign, I.Allocate))
        for previous in lexical_blocks[owner][first:index]
      ):
        return False
    return True

  def visit(block, *, top=False):
    nonlocal serial
    output = []
    for statement_index, statement in enumerate(block.statements):
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
            and native_schedule == "default"
            and target.architecture == "arm"
            and target.isa == "sme"
            and not product.data.tiled
            and product.data.implementation in {"auto", "sme"}
            and (product.data.implementation != "auto" or not statically_small(product))
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
            # Expanding a whole-matrix store into row stores must preserve
            # inline epilogue reads as captures at the original store point.
            # An unbound broadcast Prior that aliases C's first row otherwise
            # observes that row's updated value in every subsequent row.
            product_loads = {id(load) for load in I.loads_unique(product)}
            output.extend(I.BindTile(load) for load in I.loads_unique(statement.value)
                          if id(load) not in product_loads)
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
                operand, preamble = ((product.args[ordinal], []) if native or quantized_affine(product) else
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
                I.TileType(product.type.dtype, (extent,)) if native else
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

              if family == "sme" and native_schedule == "shared-owner":
                from ...._library.sme_matmul import shared_dot_into as sme_dot
              elif family == "sme" and native_schedule == "prefetch":
                from ...._library.sme_matmul import prefetched_dot_into as sme_dot
              elif family == "amx" and native_schedule == "stream-b":
                from ...._library.amx_matmul import streamed_dot_into as amx_dot
              elif native_schedule == "paired-fringe":
                if family == "amx":
                  from ...._library.amx_matmul import paired_dot_into as amx_dot
                else:
                  from ...._library.sme_matmul import paired_dot_into as sme_dot
              recipe = amx_dot if family == "amx" else sme_dot
              kwargs.pop("unroll")
              # Capture the original initializer's broadcast domain before
              # source expansion turns it into a full matrix projection.
              if family == "sme" and len(product.args) > 2:
                shape = getattr(product.args[2].type, "shape", ())
                kwargs["broadcast_acc"] = len(shape) < 2 or I.constant(shape[0]) == 1
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
                # Rebinding dimensions must retain a historical read's exact
                # DAG edge. Cloning its load would invent a new lexical read,
                # so later layout folding could bypass the old snapshot.
                inputs = tuple(
                  replace(v, type=replace(v.type, shape=shape))
                  if current_physical_read(v, block, statement_index)
                  else I.Expr(replace(v.type, shape=shape), "identity", (v,))
                  for v, shape in zip(arguments[0], ((m, k), (k, n)))
                )
                arguments = (inputs, replace(arguments[1], shape=(m, n)))
              if product.data.tiled:
                from ...._library.amx_matmul import prepare_b as amx_prepare_b
                from ...._library.matrix_recipe import blocked_matrix
                from ...._library.sme_matmul import prepare_a as sme_prepare_a
                from ...._library.sme_matmul import prepare_b as sme_prepare_b
                from ...frontend.capture import capture

                arguments += (capture(recipe.function),)
                recipe = blocked_matrix
                kwargs.update(BM=product.data.block_m, BN=product.data.block_n, BK=product.data.block_k,
                              packing=product.data.packing,
                              prepare_a=(capture(sme_prepare_a.function) if family == "sme" and
                                         product.type.dtype != "int64" and ELEMENT_BYTES[product.type.dtype] >
                                         ELEMENT_BYTES[product.args[0].type.dtype] else None),
                              prepare_b=(capture((amx_prepare_b if family == "amx" else sme_prepare_b).function)
                                         if product.type.dtype != "int64" and
                                         (family == "amx" or ELEMENT_BYTES[product.type.dtype] >
                                          ELEMENT_BYTES[product.args[0].type.dtype]) else None),
                              pack_raw_inputs=family == "sme" and product.data.packing == "auto",
                              quantum=(64 // ELEMENT_BYTES[product.args[0].type.dtype] if family == "amx" else
                                       ELEMENT_BYTES[product.type.dtype] // ELEMENT_BYTES[product.args[0].type.dtype]),
                              span=I.literal(32) if family == "amx" else I.binary("mul", I.literal(2),
                                    I.Expr(I.ScalarType("int64"), "streaming_lanes", data=product.type.dtype)))
              a, b = product.args[:2]
              current = lambda v: current_physical_read(v, block, statement_index)
              raw = lambda v: current(v) and v.op == "load" and not v.args and v.data.dtype == v.type.dtype and v.data.region is not None
              if family == "amx":
                if raw(a):
                  kwargs["raw_a"] = a.data
                # A cast chain preserves source coordinates and capture
                # identity. Pass only its layout as a packing proof; the
                # recipe still evaluates every conversion on the original
                # producer, rather than loading raw_b as native storage.
                source_b = b
                while source_b.op == "cast":
                  source_b = source_b.args[0]
                # Direct dots share their matrix consumer across packing
                # layouts. Dynamic cache traversal keeps the materialized
                # fallback: extra runtime layouts grew GCC's frame and
                # regressed small BiasRelu kernels by 10-19%, also at a
                # 1.8 GHz cap. Exact full-panel metadata avoids that cost.
                # Every fast branch still retains the original producer,
                # conversion boundaries and captured-read identity.
                static_full_b = (type(I.constant(b.type.shape[0])) is int
                                 and type(I.constant(b.type.shape[1])) is int
                                 and I.constant(b.type.shape[0]) % (64 // ELEMENT_BYTES[a.type.dtype]) == 0
                                 and I.constant(b.type.shape[1]) % 16 == 0)
                # A raw automatic BF16 resident leaf can use B's layout
                # proof just like dot. Keep converted/transformed inputs,
                # initializers, fused epilogues and explicit cache policies
                # on the established path: broad exposure previously grew
                # GCC's dynamic BiasRelu frame and regressed those kernels.
                raw_resident_b = (
                  product.data.tiled and product.data.implementation == "auto"
                  and product.data.packing == "auto"
                  and all(v is None for v in (product.data.block_m, product.data.block_n, product.data.block_k))
                  and raw(a) and raw(b) and len(product.args) == 2
                  and a.type.dtype == b.type.dtype == "bfloat16" and product.type.dtype == "float32"
                  and statement.value is product
                )
                if current(b) and (not product.data.tiled or static_full_b or raw_resident_b) and source_b.op == "load" and not source_b.args and source_b.data.region is not None:
                  kwargs["raw_b"] = source_b.data
                if current(b) and product.data.tiled and source_b.op == "load" and not source_b.args and source_b.data.region is not None:
                  # Layout evidence for the cache packer only. Keep the
                  # resident leaf's deliberately narrower raw-B contract.
                  kwargs["cache_raw_b"] = source_b.data
                  kwargs["whole_b_panel"] = (raw(a) and raw(b) and a.type.dtype == "bfloat16" and
                                             b.type.dtype == "bfloat16" and product.type.dtype == "float32")
              elif family == "sme":
                if raw(a):
                  kwargs["raw_a"] = a.data
                if raw(b):
                  kwargs["raw_b"] = b.data
              # Native aligned windows borrow physical panels with the
              # original K pitch. Nonaligned/converted windows keep their
              # logical decoder; never reinterpret them using child K.
              prepared = {side: aligned_packed_view(operand, family, product.type.dtype,
                                                    vector_state.sme_bytes) if current(operand) else None
                          for side, operand in (("a", a), ("b", b))}
              direct_windows = any(v is not None for v in prepared.values()) and (
                not product.data.tiled or product.data.packing == "auto" and
                all(prepared[side] is not None or operand.op == "packed_load" and
                    operand.data.format in {family, "auto"} for side, operand in (("a", a), ("b", b))) and
                all(v is None for v in (product.data.block_m, product.data.block_n, product.data.block_k)))
              if direct_windows:
                if product.data.tiled:
                  recipe = amx_dot if family == "amx" else sme_dot
                  arguments = arguments[:2]
                  for name in ("BM", "BN", "BK", "packing", "prepare_a", "prepare_b", "pack_raw_inputs",
                               "quantum", "span", "cache_raw_b", "whole_b_panel"):
                    kwargs.pop(name, None)
                kwargs.update({f"prepared_{side}": view for side, view in prepared.items() if view is not None})
                ir.diagnostics.append("borrowed aligned native packed subview with original panel pitch")
              for side, operand in (("a", a), ("b", b)):
                if (
                  current(operand)
                  and operand.op == "packed_load"
                  and operand.data.format in {family, "auto"}
                  and (family == "sme" and product.type.dtype != "int64" or family == "amx")
                ):
                  view = operand.data.view
                  count = I.Expr(I.ScalarType("int64"), "checked_shape", view.shape)
                  name = f"dsl_recipe_dot_{serial}_packed_{side}_capacity"
                  native_bindings.append(I.Assign(name, count))
                  capacity = I.Expr(I.ScalarType("int64"), "local", data=name)
                  kwargs[f"packed_{side}"] = replace(view, shape=(capacity,), strides=(I.literal(1),), region=None)
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
                def expand_fallback(prefix, *, product=product, kwargs=kwargs, prefer_k=prefer_k,
                                    family=family, destination=statement.destination):
                  fallback_inputs, fallback_preamble = [], []
                  for ordinal in range(2):
                    operand, preparation = input_policy(product, ordinal, f"{prefix}_{ordinal}")
                    fallback_inputs.append(operand)
                    fallback_preamble.extend(preparation)
                  fallback_kwargs = {name: kwargs[name] for name in ("acc", "acc_dtype", "epilogue")}
                  fallback_kwargs.update(unroll=unroll, pipeline=pipeline, distance=distance, prefer_k=prefer_k)
                  # Keep the measured GCC dynamic AMX owner unchanged: the
                  # expanded grouped-dot fallback can regress its hot native
                  # branch even when that fallback is not taken. Static
                  # small/GEMV and explicit vector recipes retain it. On SME,
                  # a wide producer converted to bytes also makes this cold
                  # fallback prohibitively large for LLVM's inliner; retain
                  # its ordinary vector contraction, not expanded grouped dot.
                  fallback_kwargs["widening"] = not (
                    any(type(I.constant(size)) is not int for size in product.type.shape)
                    and (
                      family == "amx"
                      or family == "sme"
                      and any(
                        ELEMENT_BYTES[operand.type.dtype] == 1
                        and any(
                          isinstance(value.type, I.TileType) and ELEMENT_BYTES[value.type.dtype] > 1
                          for value in walk_unique(operand)
                        )
                        for operand in product.args[:2]
                      )
                    )
                  )
                  if product.data.tiled:
                    fallback_kwargs.update(BM=product.data.block_m, BN=product.data.block_n,
                                           BK=product.data.block_k, packing=product.data.packing,
                                           packed_pair=kwargs.get("packed_a") is not None and kwargs.get("packed_b") is not None)
                  fallback_recipe = blocked_dot_into if product.data.tiled else vector_dot_into
                  if outline_vector_fallback:
                    from ....dsl import DSLFunction

                    fallback_recipe = DSLFunction(fallback_recipe.function, inline=False)
                  _, fallback = expand_helper(
                    fallback_recipe,
                    (tuple(fallback_inputs), destination),
                    keyword_arguments=fallback_kwargs,
                    catalog=environment.operations, prefix=prefix,
                  )
                  return fallback_preamble + fallback

                fallback = expand_fallback(f"dsl_recipe_dot_fallback_{serial}")
                # Measured AttentionBench decode/spec-decode shapes lose to
                # SIMD below one spatial tile: physical packing dominates.
                # Raw narrow-float decode keeps its original vector recipe
                # and producer caches. Physical packed and sufficiently
                # substantial byte products use the same cost predicate as
                # static recipe pruning; explicit requests remain native.
                # GCC 13's complete versioned kernel currently costs about
                # 2–3% on single-row AttentionBench decode versus the former
                # vector-only source. Unconditionally selecting AMX was ~8x
                # slower there. On 920f-4, some decode pairs measured up to
                # ~4.5% overhead. Retain this gate and producer-cache policy
                # until a replacement passes paired shape/meta benchmarks;
                # retaining raw BF16 skinny selection is therefore required.
                use_native = automatic_native_condition(product)
                if I.constant(use_native) is not True:
                  body = [I.If(use_native, I.Block(body), I.Block(expand_fallback(f"dsl_recipe_dot_small_{serial}")))]
              else:
                fallback = [I.Assert(I.literal(False), f"{family} dot instruction family is not enabled for this target")]
              body = [I.If(I.Expr(I.ScalarType("bool"), "compile_feature", data=guard), I.Block(body), I.Block(fallback))]
              ir.diagnostics.append(f"staged {family.upper()} matrix instruction recipe from DSL library")
              native_rewritten.add(id(product))
            if quantized_affine(product) and all(
              isinstance(item, (I.BindTile, I.Assert, I.Assign, I.Allocate)) for item in output
            ):
              from ...._library.runtime_quant_int8 import dot_into as quant_dot

              view = product.args[1].data.view
              name = f"dsl_quant_dot_{serial}_capacity"
              capacity = I.Expr(I.ScalarType("int64"), "local", data=name)
              physical = replace(view, shape=(capacity,), strides=(I.literal(1),), region=None)
              extent = I.Expr(I.ScalarType("int64"), "region_argument", data="prefix")
              reference = I.Expr(I.TileType(product.type.dtype, (extent,)), "region_argument", data="result")
              row, column = (I.Expr(I.ScalarType("int64"), "region_argument", data=name) for name in ("row", "column"))
              finish = None if statement.value is product else ValueFunction(
                (reference, row, column), pointwise_row_projection(
                  statement.value, product, reference, row, column, extent, environment.operations))
              _, specialized = expand_helper(
                quant_dot, (product.args[:2], statement.destination),
                keyword_arguments={"packed_b": physical, "acc": product.args[2] if len(product.args) > 2 else None,
                                   "epilogue": finish},
                catalog=environment.operations, prefix=f"dsl_quant_dot_{serial}")
              specialized.insert(0, I.Assign(name, I.Expr(I.ScalarType("int64"), "checked_shape", view.shape)))
              # This grouped GEMV avoids the generic converted-input fallback
              # on small shapes. Once both axes qualify for native SME,
              # quantize/pack plus MOPA is much faster than a row-wise GEMV.
              small_quant = I.binary("or", I.binary("lt", product.type.shape[0], I.literal(16)),
                                     I.binary("lt", product.type.shape[1], I.literal(16)))
              # Keep compile_feature as a preprocessor condition; it is not
              # a runtime scalar expression. The portable DSL grouped-dot
              # body itself is legal even without this native ISA feature.
              selected = f"dsl_quant_dot_{serial}_selected"
              body = [
                I.Assign(selected, I.literal(False)),
                I.If(I.Expr(I.ScalarType("bool"), "compile_feature",
                            data="defined(__ARM_FEATURE_SVE_MATMUL_INT8)"),
                     I.Block([I.Assign(selected, small_quant, declare=False)]), I.Block([])),
                I.If(I.Expr(I.ScalarType("bool"), "local", data=selected),
                     I.Block(specialized), I.Block(body)),
              ]
              ir.diagnostics.append("staged exact row-affine U8 quantization with packed I8 grouped GEMV")
            # Ordinary SVE consumes the same public SME packed ABI. It is
            # an explicitly gated small-shape schedule, not online packing or
            # a change to numerical dtype. Preserve the established fallback
            # for unsupported geometry, long K, casts and logical windows.
            a, b = product.args[:2]
            packed_candidate = (
              packed_dot and target.isa == "sme" and product.data.implementation == "auto"
              and a.type.dtype == b.type.dtype
              and (a.type.dtype == "bfloat16" and product.type.dtype == "float32" or
                   a.type.dtype in {"int8", "uint8"} and product.type.dtype == "int32")
              and all(v.op == "packed_load" and not v.data.origins and v.data.format in {"sme", "auto"} for v in (a, b))
              and vector_state.sve_bytes == vector_state.sme_bytes
              and (vector_state.sve_bytes or 0) >= 64
              and all(isinstance(item, (I.BindTile, I.Assert, I.Assign, I.Allocate)) for item in output)
              and (not product.data.tiled or product.data.packing == "auto" and
                   all(v is None for v in (product.data.block_m, product.data.block_n, product.data.block_k)))
            )
            if packed_candidate:
              from ...._library.packed_dot import dot_into as packed_sve_dot
              from ....dsl import DSLFunction

              m, n = product.type.shape
              k = a.type.shape[1]
              eq = lambda v, number: I.binary("eq", v, I.literal(number))
              land = lambda x, y: I.binary("and", x, y)
              lor = lambda x, y: I.binary("or", x, y)
              shape_ok = lor(land(eq(m, 2), lor(eq(n, 2), lor(eq(n, 4), eq(n, 8)))),
                             lor(land(eq(m, 4), lor(eq(n, 2), eq(n, 4))), land(eq(m, 8), eq(n, 2))))
              k_ok = land(I.binary("ge", k, I.literal(0)),
                          lor(land(eq(m, 4), I.binary("le", k, I.literal(257))),
                              land(I.binary("ne", m, I.literal(4)),
                                   I.binary("le", k, I.literal(1025 if a.type.dtype == "bfloat16" else 513)))))
              use_packed = land(shape_ok, k_ok)
              if I.constant(use_packed) is not False:
                physical, preparation = [], []
                for side, operand in (("a", a), ("b", b)):
                  view = operand.data.view
                  name = f"dsl_packed_sve_{serial}_{side}_capacity"
                  preparation.append(I.Assign(name, I.Expr(I.ScalarType("int64"), "checked_shape", view.shape)))
                  physical.append(replace(view, shape=(I.Expr(I.ScalarType("int64"), "local", data=name),),
                                          strides=(I.literal(1),), region=None))
                extent = I.Expr(I.ScalarType("int64"), "region_argument", data="prefix")
                reference = I.Expr(I.TileType(product.type.dtype, (extent,)), "region_argument", data="result")
                row, column = (I.Expr(I.ScalarType("int64"), "region_argument", data=name) for name in ("row", "column"))
                finish = None if statement.value is product else ValueFunction(
                  (reference, row, column), pointwise_row_projection(
                    statement.value, product, reference, row, column, extent, environment.operations))
                _, packed_body = expand_helper(
                  DSLFunction(packed_sve_dot.function, inline=False), (product.args[:2], statement.destination),
                  keyword_arguments={"packed_a": physical[0], "packed_b": physical[1],
                                     "acc": product.args[2] if len(product.args) > 2 else None,
                                     "acc_dtype": product.type.dtype, "epilogue": finish},
                  catalog=environment.operations, prefix=f"dsl_packed_sve_{serial}")
                instruction = "__ARM_FEATURE_SVE_BF16" if a.type.dtype == "bfloat16" else "__ARM_FEATURE_SVE_MATMUL_INT8"
                selected = I.If(use_packed, I.Block(preparation + packed_body), I.Block(body))
                body = [I.If(I.Expr(I.ScalarType("bool"), "compile_feature", data=f"defined({instruction})"),
                             I.Block([selected]), I.Block(deepcopy(body)))]
                ir.diagnostics.append("staged ordinary-SVE packed MMLA small-shape recipe")
            captured_svl = vector_state.sme_bytes if family == "sme" else None
            if family != "blocked" and all(
              type(I.constant(v)) is int for v in (*product.type.shape, product.args[0].type.shape[1])
            ):
              # The adapter validates SVL before entering this generated
              # kernel. Propagate it through immutable recipe geometry before
              # proving full-panel slices, rather than asking LLVM to recover
              # captured widths through a noinline streaming trampoline.
              # Ordinary VL and unknown cross-target SVL remain unrelated.
              body = simplify_loop_metadata(I.Block(body), streaming_bytes=captured_svl).statements
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
