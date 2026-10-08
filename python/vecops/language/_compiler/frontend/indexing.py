# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Tensor/view clipping and logical producer-window indexing."""

import ast
from dataclasses import replace

from .. import ir as I


class IndexingFrontend:
  def index(self, node):
    obj = self.expression(node.value)
    if isinstance(obj, tuple):
      if isinstance(node.slice, ast.Slice):
        bounds = []
        for part in (node.slice.lower, node.slice.upper, node.slice.step):
          value = None if part is None else self.expression(part)
          bound = None if value is None else I.constant(self.value(value, node))
          if value is not None and type(bound) is not int:
            self.error(node, "tuple slice bounds must be compile-time integers")
          bounds.append(bound)
        if bounds[2] == 0:
          self.error(node, "tuple slice bounds/step must be compile-time integers with a nonzero step")
        return obj[slice(*bounds)]
      index = I.constant(self.value(self.expression(node.slice), node))
      if not isinstance(index, int):
        self.error(node, "tuple indices must be compile-time integers")
      return obj[index]
    value_source = obj if isinstance(obj, I.Expr) and isinstance(obj.type, I.TileType) else None
    if value_source is not None:
      # Reuse the view's clipping/index normalization, but retain the value
      # producer as an argument edge: its captured loads/state must not be
      # replaced with a newly timed memory read.
      shape = obj.type.shape
      strides, extent = [None] * len(shape), I.literal(1)
      for axis in reversed(range(len(shape))):
        strides[axis], extent = extent, I.binary("mul", extent, shape[axis])
      obj = I.View(
        "logical_value",
        obj.type.dtype,
        shape,
        tuple(strides),
        I.literal(0),
        writable=False,
        region=I.ViewRegion(
          shape,
          tuple(strides),
          (I.literal(0),) * len(shape),
          shape,
          (1,) * len(shape),
          tuple(range(len(shape))),
          (False,) * len(shape),
        ),
      )
    if not isinstance(obj, I.View):
      self.error(node, "only tensor/tile views and tuples can be indexed")
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    shape, strides, offset = [], [], obj.offset
    region = obj.region
    starts, widths, steps, region_clipped, axes = (
      (list(region.starts), list(region.widths), list(region.steps), list(region.clipped), [])
      if region is not None
      else ([], [], [], [], [])
    )
    exact_starts = list(region.exact_starts or region.starts) if region is not None else []
    axis = 0
    for item in items:
      if isinstance(item, ast.Constant) and item.value is Ellipsis:
        remaining = len(obj.shape) - (len(items) - 1)
        shape.extend(obj.shape[axis : axis + remaining])
        strides.extend(obj.strides[axis : axis + remaining])
        if region is not None:
          axes.extend(region.axes[axis : axis + remaining])
        axis += remaining
        continue
      if axis >= len(obj.shape):
        self.error(node, "too many tensor indices")
      extent, stride = obj.shape[axis], obj.strides[axis]
      full_slice = isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None
      if region is not None and region.clipped[region.axes[axis]] and not full_slice:
        # Keep exact bounds, but retain the previous conservative behavior
        # for parallel/fusion ownership: requested affine starts no longer
        # describe this nested clamped partition.
        region = replace(region, affine_partition=False)
      if isinstance(item, ast.Slice):
        if full_slice:
          shape.append(extent)
          strides.append(stride)
          if region is not None:
            axes.append(region.axes[axis])
          axis += 1
          continue
        step = self.value(self.expression(item.step), node) if item.step else I.literal(1)
        if step.op != "literal" or not isinstance(step.data, int) or isinstance(step.data, bool) or step.data <= 0:
          self.error(node, "slice steps must be positive constants; negative tensor strides are supported")
        if value_source is not None and step.data != 1:
          self.error(node, "logical value windows currently require unit slice steps; use gather for explicit indices")
        start = self.value(self.expression(item.lower), node) if item.lower else I.literal(0)
        stop = self.value(self.expression(item.upper), node) if item.upper else extent
        if any(not isinstance(v.type, I.ScalarType) or "int" not in v.type.dtype for v in (start, stop)):
          self.error(node, "slice bounds must be scalar integers")
        # Region slices clip to bounds. Negative Python indices are normalized.
        if I.constant(start) is not None and I.constant(start) < 0:
          start = I.binary("add", extent, start)
        if I.constant(stop) is not None and I.constant(stop) < 0:
          stop = I.binary("add", extent, stop)
        if region is not None:
          root_axis = region.axes[axis]
          starts[root_axis] = I.binary("add", starts[root_axis], I.binary("mul", start, I.literal(steps[root_axis])))
          widths[root_axis] = I.binary("mul", I.binary("sub", stop, start), I.literal(steps[root_axis]))
          steps[root_axis] *= step.data
          region_clipped[root_axis] = True
          axes.append(root_axis)
        start = I.binary("max", I.literal(0), I.binary("min", start, extent))
        if region is not None:
          exact_starts[root_axis] = I.binary(
            "add", exact_starts[root_axis], I.binary("mul", start, I.literal(steps[root_axis] // step.data))
          )
        clipped = I.binary("min", stop, extent)
        # Native lane extents are positive; preserve a canonical min for
        # descriptor loads and matching region stores.
        if I.constant(start) == 0 and stop.op == "lanes":
          stop = clipped
        else:
          stop = I.binary("max", start, clipped)
        length = I.binary(
          "floordiv", I.binary("add", I.binary("sub", stop, start), I.binary("sub", step, I.literal(1))), step
        )
        offset = I.binary("add", offset, I.binary("mul", start, stride))
        shape.append(length)
        strides.append(I.binary("mul", stride, step))
      else:
        index = self.value(self.expression(item), node)
        if not isinstance(index.type, I.ScalarType) or "int" not in index.type.dtype:
          self.error(node, "view indices must be scalar integers; use strided regions for vector access")
        if I.constant(index) is not None and I.constant(index) < 0:
          index = I.binary("add", extent, index)
        if region is not None:
          root_axis = region.axes[axis]
          starts[root_axis] = I.binary("add", starts[root_axis], I.binary("mul", index, I.literal(steps[root_axis])))
          exact_starts[root_axis] = I.binary(
            "add", exact_starts[root_axis], I.binary("mul", index, I.literal(steps[root_axis]))
          )
          widths[root_axis] = I.literal(1)
          region_clipped[root_axis] = False
        offset = I.binary("add", offset, I.binary("mul", index, stride))
      axis += 1
    shape.extend(obj.shape[axis:])
    strides.extend(obj.strides[axis:])
    if region is not None:
      axes.extend(region.axes[axis:])
      region = replace(
        region,
        starts=tuple(starts),
        widths=tuple(widths),
        steps=tuple(steps),
        axes=tuple(axes),
        clipped=tuple(region_clipped),
        exact_starts=tuple(exact_starts),
      )
    if value_source is not None:
      if not axes:
        self.error(node, "scalar tile extraction uses T.element_at; retain a slice for a logical value window")
      starts = region.exact_starts
      for axis, (index, size) in enumerate(zip(starts, value_source.type.shape)):
        if axis not in axes:
          self.block.statements.append(
            I.Assert(
              I.binary("and", I.binary("ge", index, I.literal(0)), I.binary("lt", index, size)),
              "logical tile index out of bounds",
            )
          )
      # Arbitrary windows are logical tiles, not a claim of closed register
      # geometry. Use lower_half/upper_half for physical register splits.
      return I.Expr(
        I.TileType(value_source.type.dtype, tuple(shape), value_source.type.mask),
        "value_transform",
        (value_source, *starts),
        ("slice", tuple(axes)),
      )
    return replace(obj, shape=tuple(shape), strides=tuple(strides), offset=offset, region=region)
