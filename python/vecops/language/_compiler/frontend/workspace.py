# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Workspace sites are declared serially; storage contents are not persistent."""

from .. import ir as I
from ..dtypes import dtype_name
from ..ir.traversal import children, walk
from ..operations import Trait


class WorkspaceFrontend:
  def call_allocation(self, args, kwargs, node):
    """Allocate from parsed values, preserving the original source location.

    Shared and double-buffer storage use the same scalar-storage contract as
    alloc. Re-parsing synthetic ASTs would lose locations and evaluate shape
    expressions twice, including helper calls with a capture prelude.
    """
    if set(kwargs) - {"dtype"}:
      self.error(node, "allocation only accepts the dtype keyword")
    if not 1 <= len(args) <= 2 or len(args) == 2 and "dtype" in kwargs:
      self.error(node, "allocation expects a shape and optional dtype")
    shape = self.shape(args[0], node)
    dtype = dtype_name(kwargs.get("dtype", args[1] if len(args) == 2 else "float32"))
    strides = [I.literal(1)] * len(shape)
    for axis in reversed(range(len(shape) - 1)):
      strides[axis] = I.binary("mul", strides[axis + 1], shape[axis + 1])
    view = I.View(
      self.unique("scratch"),
      dtype,
      shape,
      tuple(strides),
      I.literal(0),
      owned=True,
      region=I.ViewRegion(
        shape,
        tuple(strides),
        (I.literal(0),) * len(shape),
        shape,
        (1,) * len(shape),
        tuple(range(len(shape))),
        (False,) * len(shape),
        root_offset=I.literal(0),
      ),
    )
    self.block.statements.append(I.Allocate(view))
    return view

  def workspace_operation(self, operation, args, kwargs, node):
    if operation == "worker_capacity":
      if args or kwargs:
        self.error(node, "worker_capacity takes no arguments")
      return I.Expr(I.ScalarType("int64"), "worker_capacity")
    if operation != "workspace":
      return NotImplemented
    if self.loop_depth or self.control_depth or self.teams or self.hardware_regions:
      self.error(node, "workspace sites must be declared before conditional/loop/parallel/ISA regions")
    if not 1 <= len(args) <= 2 or len(args) == 2 and "dtype" in kwargs:
      self.error(node, "workspace expects a shape and optional dtype")
    shape = self.shape(args[0], node)
    definitions = {s.target: s.value for s in self.block.statements if isinstance(s, I.Assign) and s.declare}

    def invalidate_updates(block):
      # Replay keys do not contain scalar payloads. A local first initialized
      # from metadata can later be mutated using such a payload, including
      # inside a branch/loop; its original declaration is no longer a proof.
      for statement in block.statements:
        if isinstance(statement, I.Assign) and not statement.declare:
          definitions.pop(statement.target, None)
        for child in children(statement):
          invalidate_updates(child)

    invalidate_updates(self.block)

    def check(value, seen=frozenset()):
      for e in walk(value):
        if e.op in {"local", "frozen"}:
          if e.data not in definitions or e.data in seen:
            self.error(node, "workspace extent must be derived from tensor metadata or worker capacity")
          check(definitions[e.data], seen | {e.data})
        elif e.op == "raw" or self.operations.get(e).effects != "none" or self.operations.has(e, Trait.REDUCTION):
          self.error(node, "workspace extent must be derived from tensor metadata or worker capacity")

    for extent in shape:
      if not isinstance(extent.type, I.ScalarType) or "int" not in extent.type.dtype:
        self.error(node, "workspace extents must be scalar integers")
      check(extent)
    dtype = dtype_name(kwargs.get("dtype", args[1] if len(args) == 2 else "float32"))
    strides, count = [None] * len(shape), I.literal(1)
    for axis in reversed(range(len(shape))):
      strides[axis], count = count, I.binary("mul", count, shape[axis])
    view = I.View(
      self.unique("workspace"),
      dtype,
      shape,
      tuple(strides),
      I.literal(0),
      owned=True,
      region=I.ViewRegion(
        shape,
        tuple(strides),
        (I.literal(0),) * len(shape),
        shape,
        (1,) * len(shape),
        tuple(range(len(shape))),
        (False,) * len(shape),
        root_offset=I.literal(0),
      ),
    )
    self.block.statements.append(I.WorkspaceAllocate(view))
    return view
