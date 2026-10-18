# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove which scalar view geometry depends only on immutable entry metadata.

This is an eligibility analysis, not permission to evaluate a trapping span
earlier. The emitter retains the first evaluation's original control flow.
Scratch pointers, captured loop coordinates and mutable scalar state are not
entry metadata. Query caches belong to this prepared kernel and retain their
expression keys, including transient views created during emission.
"""

from dataclasses import replace
from types import MappingProxyType

from ....types import TensorType
from ... import ir as I
from ...operations import Trait
from ...passes.manager import register_pass


class EntryGeometry:
  def __init__(self, kernel, catalog):
    self.kernel = kernel
    self.parameters = MappingProxyType(dict(kernel.parameters))
    self.bindings = MappingProxyType(dict(kernel.bindings))
    self.catalog = catalog
    self.tensors = frozenset(name for name, typ in self.parameters.items() if type(typ) is TensorType)
    self.raw = frozenset(
      f"{name}.{kind}({axis})" for name in self.tensors
      for kind in ("size", "stride") for axis in range(len(self.parameters[name].shape))
    )
    self.memo = {}

  def __repr__(self):
    return f"EntryGeometry(parameters={len(self.parameters)}, bindings={len(self.bindings)})"

  def matches(self, kernel, catalog):
    """A later extension may replace bindings; it must not reuse an old proof."""
    return (
      self.kernel is kernel and self.catalog is catalog
      and self.parameters.keys() == kernel.parameters.keys()
      and self.bindings.keys() == kernel.bindings.keys()
      and all(value is kernel.parameters[name] for name, value in self.parameters.items())
      and all(value is kernel.bindings[name] for name, value in self.bindings.items())
    )

  def scalar(self, value, active=None):
    if (type(value) is not I.Expr or type(value.type) is not I.ScalarType
        or value.type.dtype not in {"int64", "bool"}):
      return False
    if value in self.memo:
      return self.memo[value]
    # In-progress results stay query-local: concurrent readers of a prepared
    # plan must not observe a temporary false result from another traversal.
    active = set() if active is None else active
    if value in active:
      return False
    active.add(value)
    if value.op == "literal":
      result = not value.args and type(value.data) in (int, bool)
    elif value.op == "raw":
      result = not value.args and type(value.data) is str and value.data in self.raw
    elif value.op == "symbol":
      result = (
        not value.args and type(value.data) is str and value.data in self.bindings
        and self.scalar(self.bindings[value.data], active)
      )
    else:
      result = (
        bool(value.args)
        and self.catalog.get(value).effects == "none"
        and not self.catalog.has(
          value, Trait.REFERENCE | Trait.MEMORY_READ | Trait.STATE_REFERENCE | Trait.CAPTURE_SCALAR
        )
        and all(self.scalar(arg, active) for arg in value.args)
      )
    active.remove(value)
    self.memo[value] = bool(result)
    return self.memo[value]

  def view(self, view):
    return (
      type(view) is I.View and type(view.storage) is str and view.storage in self.tensors
      and view.owned is False and type(view.shape) is tuple and type(view.strides) is tuple
      and all(self.scalar(value) for value in (*view.shape, *view.strides, view.offset))
    )


@register_pass("cpu-plan-entry-geometry", stage="cpu-memory", after="cpu-plan-private-captures")
def plan_entry_geometry(prepared, context):
  return replace(prepared, entry_geometry=EntryGeometry(prepared.vectors.kernel, context.environment.operations))
