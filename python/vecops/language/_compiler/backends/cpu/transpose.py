# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU layout recognition and target-specific register/state policies."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from ... import ir as I
from ...analysis.affine import simplify_loop_metadata
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import expand_helper
from ...ir.traversal import walk
from ...ir.vector import RegisterProgram
from ...lowering.tile_to_vector import attach_stores, lower_region, store_roots, store_value
from ...operations import Trait, snapshot
from ...passes.manager import register_pass
from .registers import register_transpose

MAX_STATIC_HALF_INPUT_ROWS = 16


@dataclass(frozen=True)
class TransposePlan:
  """CPU-only physical layout schedule; not a portable IR contract."""

  statement: object
  dtype: str
  source_axis: int
  variants: tuple[tuple[str, RegisterProgram], ...]
  sme: bool = False
  sme_preferred: bool = True
  sme_pairs: bool = False
  cache_block: int = 64
  loop_order: str = "output"
  dense_destination: bool = False
  lane_bytes: int = 0
  # Captured SVL proves these bounded schedules; ordinary VL cannot.
  sme_const_half_output_columns: int | None = None
  sme_const_half_full_input_rows: int | None = None
  sme_program: I.Block | None = None
  pass_trace: tuple[str, ...] = ("cpu-plan-transpose", "lower-register-permutations")


@register_pass("cpu-plan-transpose", stage="cpu-memory")
def plan_memory(prepared, context):
  plans = {}
  gathers = {}
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
      if plan is not None and small_vla_conversion(plan, context.options):
        # Reuse the normal memory/alias/conversion lowering, but do not form
        # a two-word group for this conversion gather. A half-full grouped tail would
        # need multiword masks even when one ordinary SVE vector fits a row.
        region = lower_region(
          node.destination.shape,
          store_value(node).type.dtype,
          store_roots((node,)),
          unroll=1,
          opaque=prepared.vectors.kernel.materializations,
          manager=prepared.vectors.manager,
          run_passes=False,
        )
        attach_stores(region, (node,))
        region = prepared.vectors.manager.run(region)
        gathers[id(node)] = region
        if small_conversion_rolled_rows(plan):
          # Alias fallbacks and snapshot copies are mutually exclusive, but
          # their unrolled address chains can still enlarge the hot function's
          # frame. Keep every alternative's row loop rolled, not its explicit
          # vector schedule. Snapshot rebinding preserves this schedule hint.
          alternatives = (
            region,
            prepared.vectors.plans.get(id(node)),
            prepared.stores.get(id(node)),
            *(prepared.snapshots.get(id(load)) for load in I.loads_unique(node.value)),
          )
          for alternative in alternatives:
            if alternative is not None:
              alternative.outer_unroll = 1
      elif plan is not None:
        plans[id(node)] = plan
    if isinstance(node, I.Node):
      for field in fields(node):
        visit(getattr(node, field.name))
    elif isinstance(node, (tuple, list)):
      for child in node:
        visit(child)

  visit(prepared.vectors.kernel.body)
  stores = dict(prepared.stores)
  for identity, region in gathers.items():
    previous = prepared.vectors.plans.get(identity)
    if previous is None:
      stores[identity] = region
    else:
      prepared.vectors.plans[identity] = region
      # The frontend IR and public VectorKernel share this list. Keep that
      # identity so debug output reports the actual scheduled vector region.
      prepared.vectors.regions[:] = [region if item is previous else item for item in prepared.vectors.regions]
  return replace(prepared, transposes=MappingProxyType(plans), stores=MappingProxyType(stores))


def small_conversion_rolled_rows(plan):
  """Bound address-register pressure in tiny native SVE conversion gathers.

  Called only after the captured-VL, simple-conversion gather proof above.
  Tiny rows otherwise get aggressively expanded by Clang, retaining one
  address per row across hot/cold alias alternatives. A rolled row loop keeps
  the same converting memory operations and inner vector groups. Medium
  eight-byte gathers, dynamic shapes and other backends keep their schedules.
  """
  extents = tuple(I.constant(extent) for extent in plan.statement.destination.shape)
  return len(extents) == 2 and all(isinstance(extent, int) and 1 < extent <= 16 for extent in extents)


def small_vla_conversion(plan, options):
  """Prefer one native gather over a 128-bit-prefix permutation network.

  This policy is limited to automatic, simple, constant conversions that the
  shared transpose plan keeps outside SME. Its existing selection thresholds
  include tiny conversions and medium conversions with eight-byte inputs;
  reuse that decision rather than maintaining another size/dtype table here.
  The kernel compiler uses sizeless SVE when an ordinary vector length is
  captured. A wider native gather then avoids many small permutation tiles.
  Unknown VL retains the conservative existing path; neither SME SVL nor a
  user-selected cost profile establishes the ordinary vector representation.
  """
  environment = options.get("_compiler_environment")
  if environment is None or environment.codegen_target.architecture != "arm":
    return False
  if environment.codegen_target.isa not in {"sve", "sme"}:
    return False
  state = environment.vector_state
  if state is None or state.sve_bytes is None:
    return False
  statement = plan.statement
  if statement.implementation != "auto" or not plan.sme or plan.sme_preferred or len(statement.destination.shape) != 2:
    return False
  extents = tuple(I.constant(extent) for extent in statement.destination.shape)
  if not all(isinstance(extent, int) for extent in extents) or extents[0] * extents[1] <= 0:
    return False
  load = next(I.loads_unique(statement.value))
  if load.data.dtype == plan.dtype == statement.destination.dtype:
    return False
  prefix_lanes = min(16 // plan.lane_bytes, 16)
  return state.sve_bytes // plan.lane_bytes > prefix_lanes


def plan_transpose(statement, options=None):
  options = options or {}
  environment = options.get("_compiler_environment")
  default = 0 if environment is None else environment.costs.policy("sme_const_half_transpose", 0)
  if type(default) is not int or default not in (0, 1):
    raise ValueError("sme_const_half_transpose policy must be 0 or 1")
  static_half = options.get("cpu_transpose_sme_const_half_output", bool(default))
  if not isinstance(static_half, bool):
    raise TypeError("cpu_transpose_sme_const_half_output must be bool")
  # Disabling output also disables the implicit combined schedule. An explicit
  # input override remains independent for code-generation diagnostics.
  static_input = options.get("cpu_transpose_sme_const_half_full_input", static_half and bool(default))
  if not isinstance(static_input, bool):
    raise TypeError("cpu_transpose_sme_const_half_full_input must be bool")
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
    variants.append((guard, replace(register_transpose(rows, columns, width, dtype=dtype), dtype=dtype)))
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
    variants.append(
      (guard, replace(register_transpose(rows, columns, network_width, dtype=network_dtype), dtype=network_dtype))
    )
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
  plan = TransposePlan(
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
    sme_const_half_output_columns=(
      constant_half_output_columns(compute, destination, options) if simple and static_half else None
    ),
    sme_const_half_full_input_rows=(
      constant_half_full_input_rows(compute, destination, options) if simple and static_input else None
    ),
  )
  staged = options.get("cpu_transpose_sme_dsl", False)
  if not isinstance(staged, bool):
    raise TypeError("cpu_transpose_sme_dsl must be bool")
  if staged and simple:
    from ....sme import _transpose_scheduled_matrix

    dynamic_unroll = options.get("cpu_transpose_sme_dynamic_unroll", False)
    if not isinstance(dynamic_unroll, bool):
      raise TypeError("cpu_transpose_sme_dynamic_unroll must be bool")

    def raw(name):
      return I.Expr(I.ScalarType("int64"), "raw", data="sme_library." + name)

    m, n, source_pitch, target_pitch = (raw(name) for name in ("m", "n", "source_pitch", "target_pitch"))

    def view(storage, dtype, shape, pitch):
      strides = pitch, I.literal(1)
      return I.View(
        storage,
        dtype,
        shape,
        strides,
        I.literal(0),
        region=I.ViewRegion(shape, strides, (I.literal(0), I.literal(0)), shape, (1, 1), (0, 1), (False, False)),
      )

    # Preserve constant pitches in the source program. Dynamic geometry is
    # bound to fresh owner-local SSA by the terminal boundary printer.
    source_pitch = (
      I.literal(I.constant(compute.data.strides[-1]))
      if I.constant(compute.data.strides[-1]) is not None
      else source_pitch
    )
    target_pitch = (
      I.literal(I.constant(destination.strides[-2]))
      if I.constant(destination.strides[-2]) is not None
      else target_pitch
    )
    source = view("sme_library_input", compute.data.dtype, (m, n), source_pitch)
    target = view("sme_library_output", destination.dtype, (n, m), target_pitch)
    _, body = expand_helper(
      _transpose_scheduled_matrix,
      (source, target),
      catalog=catalog,
      prefix="sme_library",
      keyword_arguments={
        "compute_dtype": dtype,
        "pairs": pairs,
        "input_unroll": 2 if dynamic_unroll and I.constant(source_pitch) is None else 8,
        "output_unroll": 2 if dynamic_unroll and I.constant(target_pitch) is None else 8,
        "full_input": plan.sme_const_half_full_input_rows,
        "full_output": plan.sme_const_half_output_columns,
      },
      hardware_kind="sme",
    )
    plan = replace(plan, sme_program=simplify_loop_metadata(I.Block(body)))
  return plan


def constant_half_output_columns(compute, destination, options):
  """Prepare a narrow, bounded straight-line output schedule, never an ISA.

  This target policy is limited to the dtype/constant-address family with a
  measured positive combined result. It does not identify a library or shape.
  At most sixteen F32 column conversions bound code size and scalar address
  live ranges; this is a conservative code-budget heuristic, not a latency
  measurement or a claim that every such case is faster. The original input,
  masks, typed fringe and alias branch are unchanged. Explicit pass options
  can independently select the input/output schedules for diagnostics.
  """
  environment = options.get("_compiler_environment")
  if environment is None or environment.codegen_target.architecture != "arm":
    return None
  if environment.codegen_target.isa != "sme":
    return None
  if compute.type.dtype != "float32" or compute.data.dtype != "float32" or destination.dtype != "float16":
    return None
  pitch = I.constant(destination.strides[-2])
  if type(pitch) is not int or pitch == 0:
    return None
  state = environment.vector_state
  svl = None if state is None else state.sme_bytes
  if type(svl) is not int or not 16 <= svl <= 256 or svl % 16:
    return None
  columns = svl // ELEMENT_BYTES[compute.type.dtype]
  # A static pointer offset must itself fit the signed C++ index domain;
  # otherwise keep runtime layout validation and the original leaf. This
  # also avoids emitting an out-of-range literal for a dead zero-size view.
  if columns > 16 or not -(2**63 - 1) <= (columns - 1) * pitch <= 2**63 - 1:
    return None
  return columns


def constant_half_full_input_rows(compute, destination, options):
  """Prepare named full-input pairs only for the existing narrow family.

  Ordinary VL does not establish the ZA row count. Reuse the captured-SVL,
  simple same-memory/compute F32 -> constant-pitch F16 legality above, then
  additionally require a known, nonzero, signed source row pitch. The emitter
  guards both active extents and the actual streaming row count before any
  unmasked load. Tails, converting inputs and protected captures keep their
  existing paths. MAX_STATIC_HALF_INPUT_ROWS is only a code-size/liveness
  heuristic, not a hardware vector-length limit or measured profit claim.
  """
  rows = constant_half_output_columns(compute, destination, options)
  if rows is None or rows > MAX_STATIC_HALF_INPUT_ROWS:
    return None
  pitch = I.constant(compute.data.strides[-1])
  if type(pitch) is not int or pitch == 0:
    return None
  if not -(2**63 - 1) <= (rows - 1) * pitch <= 2**63 - 1:
    return None
  return rows
