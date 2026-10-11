# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Checked preflight for structured no-throw hardware instruction regions.

All potentially throwing metadata checks are planned once, before ownership.
Loop intervals prove every indexed access without replaying the instruction
loops. Clipped slices keep their frontend bounds; lost provenance is rejected.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...analysis.affine import affine_contains, expand_definitions, interval, positive_extent
from ...analysis.loops import trivial_true
from ...dtypes import ELEMENT_BYTES
from ...ir.traversal import nodes
from ...operations import current
from ...passes.manager import register_pass


@dataclass(frozen=True)
class HardwarePreflight:
  checks: tuple
  layouts: tuple
  writes: tuple


def preflight(region, catalog=None):
  catalog = catalog or current()
  checks, layouts, writes, definitions = [], [], [], {}

  def bound(value, bounds):
    result = interval(expand_definitions(value, definitions), bounds)
    if result is None:
      raise TypeError("hardware access has no provable integer loop interval")
    return result

  def require(condition, active, message):
    checks.append((I.binary("or", I.Expr(I.ScalarType("bool"), "not", (active,)), condition), message))

  def shape_interval(view, axis, bounds):
    root = view.region.axes[axis] if view.region is not None else None
    if root is not None and view.region.clipped[root]:
      requested = bound(view.region.widths[root], bounds)[1]
      upper = I.binary("min", I.binary("max", I.literal(0), requested), view.region.root_shape[root])
      return I.literal(0), upper
    return bound(view.shape[axis], bounds)

  def memory(statement, bounds, active, *, hint=False):
    view = statement.view
    if view is None:
      return
    if view.region is None and any(
      affine_contains(v, bounds, definitions) for v in (view.offset, *view.shape, *view.strides)
    ):
      raise TypeError("structured hardware memory access requires root-coordinate provenance")
    writable = statement.operation in {
      "amx.store",
      "sme.store",
      "sme.store_bits",
      "sme.store_tile",
      "sme.store_hor",
      "sme.store_ver",
    }
    if writable:
      writes.append(view)
    # Non-clipped axes include scalar indices: unlike region slices those
    # are not clamped by the frontend and must be proved within the root.
    axes = (
      ()
      if view.region is None
      else zip(
        view.region.exact_starts or view.region.starts, view.region.widths, view.region.root_shape, view.region.clipped
      )
    )
    for start, width, extent, clipped in axes:
      if clipped:
        continue
      lower, upper = bound(start, bounds)
      end = bound(I.binary("add", start, width), bounds)[1]
      require(
        I.binary("and", I.binary("ge", lower, I.literal(0)), I.binary("le", end, extent)),
        active,
        "hardware view index is outside its tensor",
      )
    if hint:
      # Prefetch does not consume a vector/tile, but its C++ pointer must
      # still be formed within the proven storage region.
      return
    # Requested width retains correlation at a fully clipped/empty fringe.
    shape_bounds = [shape_interval(view, axis, bounds) for axis in range(len(view.shape))]
    if region.kind == "sme":
      # Numeric memory conversion may consume/produce several storage words
      # while retaining one compute vector. Raw ZA moves use storage width.
      word_dtype = statement.vector.dtype if statement.vector is not None else view.dtype
      if statement.operation in {"sme.load_hor", "sme.load_ver", "sme.store_hor", "sme.store_ver"}:
        word_dtype = statement.operands[0]
        # Raw reinterpretation transfers only complete words. Fractional
        # terminal words are untouched, never rounded up into an overread.
        shape_bounds = [
          tuple(
            I.binary(
              "floordiv",
              I.binary("mul", endpoint, I.literal(ELEMENT_BYTES[view.dtype])),
              I.literal(ELEMENT_BYTES[word_dtype]),
            )
            for endpoint in pair
          )
          for pair in shape_bounds
        ]
      elif statement.operation in {"sme.load_bits", "sme.store_bits"}:
        # Storage-element masking rounds a partial word up after zeroing its
        # absent bytes, unlike raw ZA memory moves which transfer whole words.
        size = ELEMENT_BYTES[word_dtype]
        shape_bounds = [
          tuple(
            I.binary(
              "floordiv",
              I.binary("add", I.binary("mul", endpoint, I.literal(ELEMENT_BYTES[view.dtype])), I.literal(size - 1)),
              I.literal(size),
            )
            for endpoint in pair
          )
          for pair in shape_bounds
        ]
      lanes = I.Expr(I.ScalarType("int64"), "streaming_lanes", data=word_dtype)
      if not (statement.operation == "sme.load" and statement.operands == (True,)):
        require(
          I.binary("eq", expand_definitions(view.strides[-1], definitions), I.literal(1)),
          active,
          "SME operand must have unit column stride",
        )
      for lower, upper in shape_bounds:
        require(
          I.binary("and", I.binary("ge", lower, I.literal(0)), I.binary("le", upper, lanes)),
          active,
          "SME operand must fit one streaming word/tile",
        )
      if statement.slice_index is not None:
        lower, upper = bound(statement.slice_index, bounds)
        require(
          I.binary("and", I.binary("ge", lower, I.literal(0)), I.binary("lt", upper, lanes)),
          active,
          "SME ZA slice index is outside its streaming tile",
        )
    else:
      rows, columns = region.configuration[statement.tile]
      needed = (rows, (columns + ELEMENT_BYTES[view.dtype] - 1) // ELEMENT_BYTES[view.dtype])
      # An AMX tile always reads its configured rectangle, not the clipped
      # slice width. A lower bound is necessary here, not an upper bound.
      for shape, size in zip(view.shape, needed):
        require(
          I.binary("ge", bound(shape, bounds)[0], I.literal(size)),
          active,
          "AMX memory view is smaller than its configured tile",
        )
      require(
        I.binary("eq", expand_definitions(view.strides[-1], definitions), I.literal(1)),
        active,
        "AMX memory view is not row-contiguous",
      )
    if writable and not (region.kind == "sme" and len(view.shape) == 1):
      # A validated nonnegative, unit-stride row needs no separate layout
      # check. It still needs every access/capacity check above and must be
      # recorded as a write for historical-load snapshots.
      # Check the actual subview rather than demand a non-overlapping root:
      # disjoint writable subregions of a broadcast root can still be legal.
      if any(affine_contains(v, bounds, definitions) for v in (*view.shape, *view.strides)):
        if region.kind != "amx" or any(affine_contains(v, bounds, definitions) for v in view.strides):
          raise TypeError("hardware store layout must be invariant within its instruction loops")
        # AMX stores exactly the configured rectangle. The capacity checks
        # above prove it fits even when a clipped source slice retains loop-
        # dependent shape expressions. Check that invariant rectangle's
        # layout, rather than an unused larger logical slice.
        view = replace(view, shape=tuple(I.literal(v) for v in needed))
      checked_view = replace(
        view,
        shape=tuple(expand_definitions(v, definitions) for v in view.shape),
        strides=tuple(expand_definitions(v, definitions) for v in view.strides),
      )
      layouts.append((checked_view, active))

  def visit(block, bounds, active):
    for statement in block.statements:
      if isinstance(statement, I.Assign):
        definitions[statement.target] = expand_definitions(statement.value, definitions)
      elif isinstance(statement, I.For):
        step_expression = expand_definitions(statement.step, definitions)
        step = I.constant(step_expression)
        native_step = step is None and positive_extent(step_expression, catalog)
        if not native_step and (type(step) is not int or step == 0):
          raise TypeError("hardware instruction loops require a nonzero constant step")
        ascending = native_step or step > 0
        start, stop = (expand_definitions(v, definitions) for v in (statement.start, statement.stop))
        if native_step:
          last_iteration = I.binary("sub", stop, I.literal(1))
        else:
          distance = I.binary("sub", stop, start) if ascending else I.binary("sub", start, stop)
          steps = I.binary(
            "floordiv", I.binary("max", I.literal(0), I.binary("sub", distance, I.literal(1))), I.literal(abs(step))
          )
          last_iteration = I.binary("add", start, I.binary("mul", steps, I.literal(step)))
        first, last = (start, last_iteration) if ascending else (last_iteration, start)
        nested = {**bounds, statement.target: (first, last)}
        start_range, stop_range = bound(start, bounds), bound(stop, bounds)
        executes = (
          I.binary("lt", start_range[0], stop_range[1]) if ascending else I.binary("gt", start_range[1], stop_range[0])
        )
        visit(statement.body, nested, I.binary("and", active, executes))
      elif isinstance(statement, I.If):
        if affine_contains(statement.condition, bounds, definitions):
          raise TypeError("hardware branches must be invariant in their enclosing instruction loops")
        condition = expand_definitions(statement.condition, definitions)
        visit(statement.body, bounds, I.binary("and", active, condition))
        visit(
          statement.otherwise,
          bounds,
          I.binary("and", active, I.Expr(I.ScalarType("bool"), "not", (condition,))),
        )
      elif isinstance(statement, I.ArchitectureCall):
        memory(statement, bounds, active)
        if statement.vector is not None:
          vector = statement.vector
          lanes = I.Expr(I.ScalarType("int64"), "streaming_lanes", data=vector.dtype)
          lower, upper = (
            shape_interval(statement.view, 0, bounds)
            if statement.operation == "sme.load" and I.same_expr(vector.extent, statement.view.shape[0])
            else bound(vector.extent, bounds)
          )
          require(
            I.binary("and", I.binary("ge", lower, I.literal(0)), I.binary("le", upper, lanes)),
            active,
            "SME vector must fit one streaming word",
          )
          if statement.slice_index is not None:
            lower, upper = bound(statement.slice_index, bounds)
            require(
              I.binary("and", I.binary("ge", lower, I.literal(0)), I.binary("lt", upper, lanes)),
              active,
              "SME ZA slice index is outside its streaming tile",
            )
          if statement.operation == "sme.store" and not I.same_expr(statement.view.shape[0], vector.extent):
            require(
              I.binary("ge", bound(I.binary("sub", statement.view.shape[0], vector.extent), bounds)[0], I.literal(0)),
              active,
              "SME store destination is shorter than its streaming vector",
            )
      elif isinstance(statement, I.Prefetch):
        memory(I.ArchitectureCall("prefetch", view=statement.view), bounds, active, hint=True)
      elif isinstance(statement, I.Assert):
        if trivial_true(statement.condition):
          continue
        if affine_contains(statement.condition, bounds, definitions):
          raise TypeError("hardware assertion must be invariant in its instruction loops")
        require(expand_definitions(statement.condition, definitions), active, statement.message)

  visit(region.body, {}, I.literal(True))
  return HardwarePreflight(tuple(checks), tuple(layouts), tuple(writes))


@register_pass("cpu-plan-hardware", stage="cpu-memory", after="cpu-plan-register-programs")
def plan_hardware(prepared, context):
  plans = {
    id(value): preflight(value, context.environment.operations)
    for value in nodes(prepared.vectors.kernel.body)
    if isinstance(value, I.HardwareRegion)
  }
  return replace(prepared, hardware=MappingProxyType(plans))
