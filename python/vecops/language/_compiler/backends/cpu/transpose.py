"""CPU layout recognition and target-specific register/state policies."""

from dataclasses import fields, replace
from types import MappingProxyType
from ... import ir as I
from ...ir.transpose import TransposePlan
from ...ir.traversal import walk
from ...dtypes import ELEMENT_BYTES
from ...lowering.transpose import register_transpose
from ...passes.manager import register_pass
from ...operations import Trait, snapshot


@register_pass("cpu-plan-transpose", stage="cpu-memory")
def plan_memory(prepared, context):
  plans = {}
  seen = set()

  def visit(node):
    if isinstance(node, I.Node):
      if id(node) in seen:
        return
      seen.add(id(node))
    if isinstance(node, I.Store):
      plan = plan_transpose(node, context.options)
      if node.implementation == "sme" and (plan is None or not plan.sme):
        raise TypeError("explicit SME transpose requires a simple unmasked last-two-axis load and supported strides")
      if plan is not None:
        plans[id(node)] = plan
    if isinstance(node, I.Node):
      for field in fields(node):
        visit(getattr(node, field.name))
    elif isinstance(node, (tuple, list)):
      for child in node:
        visit(child)

  visit(prepared.vectors.kernel.body)
  return replace(prepared, transposes=MappingProxyType(plans))


def plan_transpose(statement, options=None):
  options = options or {}
  catalog = options["_compiler_environment"].operations if "_compiler_environment" in options else snapshot()
  destination, value = statement.destination, statement.value
  rank = len(destination.shape)
  if statement.implementation == "gather" or rank < 2 or statement.mask is not None:
    return None
  # Broadcasts, reductions and masks need the general streaming scheduler.
  loads = tuple(I.loads(value))
  if not loads or any(
    not catalog.has(v, Trait.TRANSPOSE_STREAMABLE) and isinstance(v.type, I.TileType) or getattr(v.type, "mask", False)
    for v in walk(value)
  ):
    return None
  if not isinstance(value.type, I.TileType) or len(value.type.shape) != rank:
    return None
  if not all(I.same_expr(a, b) for a, b in zip(value.type.shape, destination.shape)):
    return None
  # Initially transpose the last two axes; preceding axes are batch planes.
  axis = rank - 2
  if I.constant(destination.strides[-1]) not in (None, 1):
    return None
  if statement.implementation != "sme" and all(I.constant(v.data.strides[-1]) == 1 for v in loads):
    return None
  for load in loads:
    view = load.data
    if (
      load.args
      or len(view.shape) != rank
      or not all(I.same_expr(a, b) for a, b in zip(view.shape, destination.shape))
      or I.constant(view.strides[axis]) not in (None, 1)
    ):
      return None
  if any(v.op == "axis_reduce" for v in walk(value)):
    return None
  # A final cast can be fused into store_convert after the permutation.
  compute = value.args[0] if value.op == "cast" and value.type.dtype == destination.dtype else value
  dtype = compute.type.dtype
  width = ELEMENT_BYTES[dtype]
  lane_bytes = max(
    (
      width,
      ELEMENT_BYTES[destination.dtype],
      *(ELEMENT_BYTES[v.data.dtype] for v in loads),
      *(ELEMENT_BYTES[v.type.dtype] for v in walk(value) if isinstance(v.type, I.TileType)),
    )
  )
  if dtype == "bool":
    return None
  variants = []
  for bytes_ in (64, 32, 16):
    # VLA FixedTag must fit every rebound memory/compute domain into its
    # guaranteed 128-bit prefix (e.g. 4*f32 cannot load 4*f64 on VLA SVE).
    columns = min(bytes_ // (lane_bytes if bytes_ == 16 else width), 16)
    rows = min(columns, 8)
    guard = (
      f"defined(CPU_CAPABILITY_SVE) && defined(FIXED_SVE_BITS) && FIXED_SVE_BITS >= {bytes_ * 8}"
      if bytes_ > 16
      else "defined(CPU_CAPABILITY_SVE)"
    )
    variants.append((guard, register_transpose(rows, columns, width)))
  # The GCC register-pressure cap follows the measured C++ baseline. Clang
  # can use 16 rows; SVE caps rows at 8, including the named-register VLA path.
  for bytes_, guard in ((64, "VEC_WIDTH >= 512"), (32, "VEC_WIDTH >= 256"), (16, "1")):
    move_conversion = (
      compute.op == "load"
      and not compute.args
      and destination.dtype == dtype
      and ELEMENT_BYTES[compute.data.dtype] < width
    )
    deferred = options.get("cpu_transpose_defer_conversion")
    if deferred is not None and not isinstance(deferred, bool):
      raise ValueError("cpu_transpose_defer_conversion must be bool")
    if deferred is None:
      # An eightfold widening with unknown strides creates many wide stores
      # and address chains. Paired measurements favor the compute-domain
      # network there; constants retain narrow-bit permutation's advantage.
      deferred = not (
        compute.op == "load"
        and ELEMENT_BYTES[compute.data.dtype] == 1
        and width == 8
        and any(I.constant(v) is None for v in destination.shape[-2:])
      )
    move_conversion = move_conversion and deferred
    network_dtype = compute.data.dtype if move_conversion else dtype
    network_width = ELEMENT_BYTES[network_dtype]
    columns = min(bytes_ // network_width, 16)
    rows = min(columns, 8 if network_width >= 4 else 16)
    variants.append((guard, replace(register_transpose(rows, columns, network_width), dtype=network_dtype)))
  simple = compute.op == "load" and not compute.args and len(loads) == 1
  preferred = True
  pairs = False
  extents = tuple(I.constant(v) for v in destination.shape[-2:])
  if simple and all(isinstance(v, int) for v in extents):
    elements = extents[0] * extents[1]
    memory = loads[0].data.dtype
    converts = memory != dtype or destination.dtype != dtype
    # Mirror the measured C++ selection thresholds; do not guess new SME
    # thresholds without a machine on which to benchmark streaming state.
    preferred = not (converts and (elements <= 256 or ELEMENT_BYTES[memory] == 8 and elements <= 8192))
    if memory == dtype:
      if destination.dtype == dtype:
        pairs = elements >= 1024 * 1024 and (width in (1, 4) or width == 8 and extents[0] == extents[1])
      elif width == 4 and ELEMENT_BYTES[destination.dtype] == 2:
        pairs = elements >= 8000
  # Input-major microtiles collect adjacent partial output cache lines. This
  # matters especially for narrowing stores; a 128-square cache tile also
  # amortizes the extra loop/state bookkeeping on current x86.
  cache = options.get("cpu_transpose_cache_block", 128)
  order = options.get("cpu_transpose_loop_order", "input")
  if not isinstance(cache, int) or isinstance(cache, bool) or cache < 16 or cache % 16:
    raise ValueError("cpu_transpose_cache_block must be a positive multiple of 16")
  if order not in {"input", "output"}:
    raise ValueError("cpu_transpose_loop_order must be 'input' or 'output'")
  expected, dense = I.literal(1), True
  for extent, stride in zip(reversed(destination.shape), reversed(destination.strides)):
    dense = dense and I.same_expr(stride, expected)
    expected = I.binary("mul", expected, extent)
  return TransposePlan(
    statement,
    dtype,
    axis,
    tuple(variants),
    sme=simple,
    sme_preferred=preferred,
    sme_pairs=pairs,
    cache_block=cache,
    loop_order=order,
    dense_destination=dense,
    lane_bytes=lane_bytes,
  )
