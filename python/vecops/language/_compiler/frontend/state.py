"""Typed control-flow state and branch joins; no CPU storage decisions."""

from .. import ir as I
from ..operations import Trait
from .control import assigned_names


class StateFrontend:
  def declare_state(self, value):
    state = I.TileState(self.unique("state"), value.type)
    self.block.statements.append(I.DeclareState(state, value))
    return I.Expr(state.type, "state_ref", data=state)

  def snapshot_state(self, value):
    from ..ir.traversal import walk

    if (
      isinstance(value, I.Expr)
      and isinstance(value.type, I.TileType)
      and any(self.operations.has(e, Trait.STATE_REFERENCE) for e in walk(value))
    ):
      return self.declare_state(value)
    return value

  def check_state_type(self, typ, value, node):
    if not isinstance(value, I.Expr) or not isinstance(value.type, I.TileType):
      self.error(node, "tile state must be updated with a tile/vector value")
    other = value.type
    if typ.dtype != other.dtype or typ.mask != other.mask:
      self.error(node, "tile state dtype/predicate kind changes; use an explicit cast")
    if len(typ.shape) != len(other.shape) or any(
      I.constant(a) is not None and I.constant(b) is not None and I.constant(a) != I.constant(b)
      for a, b in zip(typ.shape, other.shape)
    ):
      self.error(node, "tile state logical shape changes across control flow")
    if isinstance(typ, I.VectorType) != isinstance(other, I.VectorType) or (
      isinstance(typ, I.VectorType) and typ.tag != other.tag
    ):
      self.error(node, "vector state tag changes across control flow")

  def promote_control_state(self, nodes, node, *, ignore=()):
    for name in sorted(assigned_names(nodes) - set(ignore)):
      value = self.env.get(name)
      if isinstance(value, I.Expr) and isinstance(value.type, I.TileType) and value.op != "state_ref":
        self.env[name] = self.declare_state(value)
      elif isinstance(value, I.Expr) and isinstance(value.type, I.ScalarType) and value.op == "literal":
        target = self.unique(name)
        self.block.statements.append(I.Assign(target, value))
        self.env[name] = I.Expr(value.type, "local", data=target)

  def available_scalar_names(self, environment):
    names, seen = set(), set()

    def visit(value):
      if isinstance(value, tuple):
        for v in value:
          visit(v)
      elif isinstance(value, I.Expr) and id(value) not in seen:
        seen.add(id(value))
        if value.op in {"local", "frozen"}:
          names.add(value.data)
        if isinstance(value.type, I.TileType):
          visit(value.type.shape)
        visit(value.args)
        if isinstance(value.data, I.View):
          visit(value.data)
      elif isinstance(value, I.View):
        visit((value.offset, *value.shape, *value.strides))

    for value in environment.values():
      visit(value)
    return names

  def join_branch_values(self, a, b, body, otherwise, before, node):
    """Create typed result slots outside an if; tuple slots are recursive."""
    if isinstance(a, tuple) or isinstance(b, tuple):
      if not isinstance(a, tuple) or not isinstance(b, tuple) or len(a) != len(b):
        self.error(node, "branch result tuple arity changes")
      return tuple(self.join_branch_values(x, y, body, otherwise, before, node) for x, y in zip(a, b))
    if not isinstance(a, I.Expr) or not isinstance(b, I.Expr):
      self.error(node, "a branch result must be a scalar/tile value defined on both paths")
    if isinstance(a.type, I.TileType):
      self.check_state_type(a.type, b, node)
      # Branch-local captured dimensions cannot be hoisted into a result slot.
      from ..ir.traversal import walk

      available = self.available_scalar_names(before)
      if any(e.op in {"local", "frozen"} and e.data not in available for s in a.type.shape for e in walk(s)):
        self.error(node, "branch tile shape must be available before the dynamic if")
      initial = I.Expr(a.type, "fill", (I.literal(False if a.type.mask else 0, a.type.dtype),))
      reference = self.declare_state(initial)
      body.statements.append(I.UpdateState(reference.data, a))
      otherwise.statements.append(I.UpdateState(reference.data, b))
      return reference
    if not isinstance(b.type, I.ScalarType) or a.type.dtype != b.type.dtype:
      self.error(node, "branch result dtype changes; use an explicit cast")
    target = self.unique("branch_value")
    self.block.statements.append(I.Assign(target, I.Expr(a.type, "cast", (I.literal(0),))))
    body.statements.append(I.Assign(target, a, False))
    otherwise.statements.append(I.Assign(target, b, False))
    return I.Expr(a.type, "local", data=target)
