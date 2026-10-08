# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit resident teams with conservative, compile-time uniformity rules."""

import ast

from .. import ir as I


class CollectiveFrontend:
  def collective(self, operation, args, kwargs, node):
    family = self.operations.get(operation).family
    if operation == "team":
      if len(args) > 1 or args and "workers" in kwargs:
        self.error(node, "team accepts one workers argument")
      workers = self.value(kwargs.get("workers", args[0] if args else 0), node)
      if not isinstance(workers.type, I.ScalarType) or "int" not in workers.type.dtype:
        self.error(node, "team workers must be a scalar integer")
      return ("team-region", workers)
    if family == "team":
      if not self.teams:
        self.error(node, "worker queries and collectives require T.team")
      name, depth = self.teams[-1]
      if operation in {"worker_id", "worker_count"}:
        if args or kwargs:
          self.error(node, "worker queries take no arguments")
        return I.Expr(
          I.ScalarType("int64"), "raw", data=f"{name}.{'worker_id' if operation == 'worker_id' else 'worker_count'}()"
        )
      if self.control_depth != depth:
        self.error(node, "collectives must be at team top level; branch/loop uniformity is not proven")
      if operation == "barrier":
        if args or kwargs:
          self.error(node, "barrier takes no arguments")
        self.block.statements.append(I.Barrier(name))
        return None
      if len(args) != 1:
        self.error(node, "team_reduce expects one scalar")
      value = self.value(args[0], node)
      op = kwargs.get("op", "add")
      if (
        not isinstance(value.type, I.ScalarType)
        or value.type.dtype not in {"float32", "float64", "int32", "int64", "uint32", "uint64"}
        or op not in {"add", "min", "max"}
      ):
        self.error(node, "unsupported team reduction type/operation")
      target = self.unique("collective")
      self.block.statements.append(I.TeamReduce(name, target, value, op))
      return I.Expr(value.type, "local", data=target)
    if family == "atomic":
      count = 3 if operation == "atomic_compare_exchange" else 2
      if len(args) != count or not isinstance(args[0], I.View) or args[0].shape or not args[0].writable:
        self.error(node, "atomics require a writable scalar view and scalar operands")
      view = args[0]
      if view.dtype not in {"float32", "float64", "int32", "int64", "uint32", "uint64"}:
        self.error(node, "atomics support 32/64-bit integers and floats")
      values = tuple(self.value(v, node) for v in args[1:])
      if any(not isinstance(v.type, I.ScalarType) for v in values):
        self.error(node, "atomic operands must be scalars")
      order = kwargs.get("order", "relaxed")
      if order not in {"relaxed", "acq_rel", "seq_cst"}:
        self.error(node, "atomic order must be relaxed, acq_rel or seq_cst")
      self.materialize(view.storage)
      result = self.unique("atomic")
      self.block.statements.append(I.Assign(result, I.Expr(I.ScalarType(view.dtype), operation, values, (view, order))))
      return I.Expr(I.ScalarType(view.dtype), "local", data=result)
    if operation == "shared":
      if self.teams:
        self.error(node, "declare shared storage before entering T.team")
      return self.call_allocation(args, kwargs, node)
    return NotImplemented

  def team_region(self, node, descriptor):
    if self.teams or self.loop_depth:
      self.error(node, "teams cannot nest inside loops, grids or other teams")
    # Non-local exits can abandon a barrier and are rejected before lowering.
    if any(isinstance(n, (ast.Return, ast.Break, ast.Continue)) for s in node.body for n in ast.walk(s)):
      self.error(node, "team body cannot return/break/continue")
    # Initialize shared owned values before workers can observe or overwrite
    # them; forwarding/materializing them inside callbacks would race.
    for storage in tuple(self.promoted):
      self.materialize(storage)
    name = self.unique("team")
    before = dict(self.env)
    outer = node.items[0].optional_vars
    if outer is not None:
      if not isinstance(outer, ast.Name):
        self.error(node, "team binding must be one worker-id name")
      self.env[outer.id] = I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.worker_id()")
    self.teams.append((name, self.control_depth + 1))
    try:
      body = self.nested(node.body)
    finally:
      self.teams.pop()
      self.env = before
    from ..ir.traversal import children

    local, states = set(), set()

    def validate(block):
      for s in block.statements:
        if isinstance(s, I.Assign):
          if s.declare:
            local.add(s.target)
          elif s.target not in local:
            self.error(node, "team cannot mutate an enclosing scalar; use shared storage/atomics")
        if isinstance(s, I.DeclareState):
          states.add(s.state.name)
        if isinstance(s, I.UpdateState) and s.state.name not in states:
          self.error(node, "team cannot mutate an enclosing tile state")
        for child in children(s):
          validate(child)

    validate(body)
    self.block.statements.append(I.Team(name, descriptor[1], body))
