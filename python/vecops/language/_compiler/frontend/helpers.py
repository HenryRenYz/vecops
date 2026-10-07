"""Source-parsed T.func binding, typed results and lexical helper exits."""

import ast

from .. import ir as I
from .control import helper_returns


class HelperFrontend:
  def helper(self, body, node, *, arguments=None):
    if body.tree.args.vararg or body.tree.args.kwarg or body.tree.args.posonlyargs:
      self.error(node, "DSL helpers currently require named positional/keyword-only parameters")
    if self.helper_depth >= 32:
      self.error(node, "DSL helper expansion depth exceeds 32")
    positional = [a.arg for a in body.tree.args.args]
    parameters = positional + [a.arg for a in body.tree.args.kwonlyargs]
    args = [self.expression(a) for a in node.args] if arguments is None else list(arguments)
    kwargs = {k.arg: self.expression(k.value) for k in node.keywords} if arguments is None else {}
    if len(args) > len(positional):
      self.error(node, "too many helper arguments")
    bound = dict(zip(parameters, args))
    for name, value in kwargs.items():
      if name not in parameters or name in bound:
        self.error(node, "invalid or duplicate helper argument")
      bound[name] = value
    defaults = dict(zip(positional[len(positional) - len(body.tree.args.defaults) :], body.tree.args.defaults))
    defaults.update({a.arg: v for a, v in zip(body.tree.args.kwonlyargs, body.tree.args.kw_defaults) if v is not None})
    previous_env, previous_body = self.env, self.body
    self.env = dict(body.environment)
    self.env.update(bound)
    self.body = body
    self.helper_depth += 1
    try:
      for name in parameters:
        if name not in self.env:
          if name not in defaults:
            self.error(node, f"missing helper argument {name!r}")
          self.env[name] = self.expression(defaults[name])
      loop_returns = any(
        isinstance(s, (ast.For, ast.While)) and any(isinstance(r, ast.Return) for r in ast.walk(s))
        for s in ast.walk(body.tree)
      )
      if loop_returns:
        return self.helper_exit_region(body, node)
      if any(isinstance(s, ast.Return) and s.value is not None for s in ast.walk(body.tree)):
        result = self.unique("helper_result")
        self.statements(helper_returns(body.tree.body, result, self.error))
        if result not in self.env:
          self.error(node, "T.func did not produce a result on every path")
        return self.env[result]
      self.statements(helper_returns(body.tree.body, None, self.error))
      return None
    finally:
      self.helper_depth -= 1
      self.env, self.body = previous_env, previous_body

  def helper_exit_region(self, body, node):
    parent = self.block
    for storage in tuple(self.promoted):
      self.materialize(storage)
    context = {
      "name": self.unique("helper_exit"),
      "prefix": [],
      "result": None,
      "available": dict(self.env),
      "value_returning": any(isinstance(s, ast.Return) and s.value is not None for s in ast.walk(body.tree)),
    }
    self.helper_exits.append(context)
    self.block = I.Block()
    self.control_depth += 1
    try:
      self.statements(body.tree.body)
      region = self.block
    finally:
      self.block = parent
      self.helper_exits.pop()
      self.control_depth -= 1
    if context["result"] is None and context["value_returning"]:
      self.error(node, "loop-returning T.func requires a value-returning path")
    if not context["value_returning"]:
      parent.statements.append(I.ExitRegion(context["name"], region))
      return None
    # Runtime verification distinguishes a missing return from a valid zero.
    flag = self.unique("helper_returned")
    parent.statements.append(I.Assign(flag, I.literal(False)))
    for block in self.exit_blocks(region, context["name"]):
      for index, statement in enumerate(block.statements):
        if isinstance(statement, I.Exit) and statement.region == context["name"]:
          block.statements.insert(index, I.Assign(flag, I.literal(True), False))
          break
    parent.statements.extend(context["prefix"])
    parent.statements.append(I.ExitRegion(context["name"], region))
    parent.statements.append(
      I.Assert(I.Expr(I.ScalarType("bool"), "local", data=flag), "T.func reached end without returning")
    )
    return context["result"]

  def exit_blocks(self, block, name):
    from ..ir.traversal import children

    yield block
    for statement in block.statements:
      if isinstance(statement, I.ExitRegion):
        continue
      for child in children(statement):
        yield from self.exit_blocks(child, name)

  def helper_return(self, value, node):
    context = self.helper_exits[-1]
    if value is None:
      if context["value_returning"]:
        self.error(node, "value-returning T.func must return a scalar/tile value on every path")
      self.block.statements.append(I.Exit(context["name"]))
      return

    def slot(v):
      if isinstance(v, tuple):
        return tuple(slot(x) for x in v)
      v = self.value(v, node)
      if isinstance(v.type, I.TileType):
        from ..ir.traversal import walk

        available = self.available_scalar_names(context["available"])
        if any(e.op in {"local", "frozen"} and e.data not in available for s in v.type.shape for e in walk(s)):
          self.error(node, "helper result tile shape must be available before its exit region")
        state = I.TileState(self.unique("helper_value"), v.type)
        initial = I.Expr(v.type, "fill", (I.literal(False if v.type.mask else 0, v.type.dtype),))
        context["prefix"].append(I.DeclareState(state, initial))
        return I.Expr(v.type, "state_ref", data=state)
      name = self.unique("helper_value")
      # A result slot has a storage dtype, not a literal-promotion hint.
      # Returning a weak literal must not make subsequent typed values of
      # the same dtype incompatible, or leave a mutable local marked weak.
      typ = I.ScalarType(v.type.dtype)
      context["prefix"].append(I.Assign(name, I.Expr(typ, "cast", (I.literal(0),))))
      return I.Expr(typ, "local", data=name)

    if context["result"] is None:
      context["result"] = slot(value)

    def update(target, source):
      if isinstance(target, tuple):
        if not isinstance(source, tuple) or len(source) != len(target):
          self.error(node, "T.func return tuple arity changes")
        for a, b in zip(target, source):
          update(a, b)
      elif target.op == "state_ref":
        self.check_state_type(target.type, source, node)
        self.block.statements.append(I.UpdateState(target.data, self.freeze_tile(source)))
      else:
        source = self.value(source, node)
        if not isinstance(source.type, I.ScalarType) or target.type.dtype != source.type.dtype:
          self.error(node, "T.func return dtype changes; use an explicit cast")
        self.block.statements.append(I.Assign(target.data, source, False))

    update(context["result"], value)
    self.block.statements.append(I.Exit(context["name"]))
