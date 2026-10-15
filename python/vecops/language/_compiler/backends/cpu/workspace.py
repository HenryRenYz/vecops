# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Opt-in replay storage for top-level, immutable-metadata CPU scratch sites.

Keep loop/branch/ISA allocations invocation-local. Their placement and lifetime
need a separate proof: requesting a live workspace site twice is invalid, and
payload-dependent topology is not covered by the runtime replay fingerprint.
"""

import re

from ... import ir as I
from ...ir.traversal import nodes
from ...operations import Trait
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-promote-workspace-scratch", stage="cpu-memory", before="cpu-plan-transpose")
def promote_workspace_scratch(prepared, context):
  """Reuse existing WorkspaceContext authority; default emission is unchanged."""
  enabled = context.options.get("cpu_workspace_scratch", False)
  if type(enabled) is not bool:
    raise TypeError("cpu_workspace_scratch must be bool")
  if not enabled:
    return PassResult(prepared, changed=False)
  kernel = prepared.vectors.kernel
  catalog = context.environment.operations
  # A later update invalidates the original declaration even when the extent
  # currently appears metadata-derived. Replay keys do not include payloads.
  mutable = {s.target for s in nodes(kernel.body) if isinstance(s, I.Assign) and not s.declare}
  mutable.update(s.target for s in nodes(kernel.body) if isinstance(s, I.For))
  definitions = {}
  referenced = {
    view.storage
    for s in kernel.body.statements
    if not isinstance(s, I.Allocate)
    for view in nodes(s)
    if isinstance(view, I.View)
  }

  def metadata_raw(value):
    match = re.fullmatch(r"([A-Za-z_]\w*)\.(size|stride)\((\d+)\)", value)
    if not match:
      return False
    annotation = kernel.parameters.get(match[1])
    return hasattr(annotation, "shape") and int(match[3]) < len(annotation.shape)

  def metadata_uncached(value, seen=frozenset()):
    if not isinstance(value, I.Expr) or not isinstance(value.type, I.ScalarType):
      return False
    if value.op == "literal":
      return True
    if value.op == "raw":
      return isinstance(value.data, str) and metadata_raw(value.data)
    if value.op in {"symbol", "local", "frozen"}:
      name = value.data
      definition = kernel.bindings.get(name, definitions.get(name))
      return name not in mutable and name not in seen and definition is not None and metadata(definition, seen | {name})
    if value.op == "worker_capacity":
      return True
    definition = catalog.get(value)
    if definition.effects != "none" or catalog.has(
      value, Trait.MEMORY_READ | Trait.REDUCTION | Trait.STATE_REFERENCE | Trait.UNSTABLE_EXTENT
    ):
      return False
    # Pure scalar arithmetic is eligible only when every leaf has been traced
    # to literals, validated tensor metadata, or the replay-keyed worker count.
    return bool(value.args) and all(metadata(argument, seen) for argument in value.args)

  memo, active = {}, set()

  def metadata(value, seen=frozenset()):
    # Per-allocation memoization preserves definition-cycle context without
    # expanding a shared scalar DAG as a tree. Dominating definitions change
    # between sites, so no proof result is retained across allocations.
    key = id(value), seen
    if key in memo:
      return memo[key]
    if id(value) in active:
      return False
    active.add(id(value))
    try:
      result = metadata_uncached(value, seen)
    finally:
      active.remove(id(value))
    memo[key] = result
    return result

  changed = 0
  statements = []
  for statement in kernel.body.statements:
    if isinstance(statement, I.Assign) and statement.declare and statement.target not in mutable:
      definitions[statement.target] = statement.value
    if isinstance(statement, I.Allocate):
      memo.clear()
    if (
      isinstance(statement, I.Allocate)
      and statement.view.storage in referenced
      and all(metadata(v) for v in statement.view.shape)
    ):
      statement = I.WorkspaceAllocate(statement.view)
      changed += 1
    statements.append(statement)
  if changed:
    kernel.body.statements = statements
    kernel.diagnostics.append(f"promoted {changed} top-level metadata scratch site(s) to replay workspace")
  return PassResult(prepared, changed=bool(changed))
