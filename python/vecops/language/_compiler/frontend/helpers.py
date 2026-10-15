# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-parsed T.func binding, typed results and lexical helper exits."""

import ast
from dataclasses import dataclass, fields, replace

from .. import ir as I
from ..operations import Trait
from .control import helper_returns


@dataclass(frozen=True)
class ValueFunction:
  """Compiler-staged pure value helper, with no executable host callback.

  This bridges a caller's already captured DAG into a source-parsed library
  algorithm. Substitution retains untouched read identities and binds formal
  shape variables to actual vector prefixes. It does not print native code.
  """

  parameters: tuple
  result: I.Expr

  @property
  def streaming_selection(self):
    from ..analysis.projection import streaming_selection_supported

    return streaming_selection_supported(self.result, self.parameters[0].type.dtype, staged=True)

  def bind(self, arguments):
    if len(arguments) != len(self.parameters):
      raise TypeError("value helper argument count mismatch")
    replacements = {}
    for parameter, argument in zip(self.parameters, arguments):
      if not isinstance(argument, I.Expr) or parameter.type.dtype != argument.type.dtype:
        raise TypeError("value helper argument dtype mismatch")
      replacements[id(parameter)] = argument
      if isinstance(parameter.type, I.TileType):
        if (
          not isinstance(argument.type, I.TileType)
          or parameter.type.mask != argument.type.mask
          or len(parameter.type.shape) != len(argument.type.shape)
        ):
          raise TypeError("value helper argument rank mismatch")
        replacements.update((id(a), b) for a, b in zip(parameter.type.shape, argument.type.shape))
        if isinstance(parameter.type, I.VectorType) and isinstance(argument.type, I.VectorType):
          replacements[id(parameter.type.tag)] = argument.type.tag
      elif not isinstance(argument.type, I.ScalarType):
        raise TypeError("value helper requires a scalar argument")
    memo = {}

    def visit(value):
      if id(value) in replacements:
        return replacements[id(value)]
      if isinstance(value, tuple):
        changed = tuple(map(visit, value))
        return value if all(a is b for a, b in zip(changed, value)) else changed
      if isinstance(value, I.Node):
        if id(value) not in memo:
          changes = {field.name: visit(getattr(value, field.name)) for field in fields(value)}
          memo[id(value)] = (
            value
            if all(changed is getattr(value, name) for name, changed in changes.items())
            else replace(value, **changes)
          )
        return memo[id(value)]
      return value

    return visit(self.result)


class HelperFrontend:
  def helper(self, body, node, *, arguments=None, keyword_arguments=None):
    if body.tree.args.vararg or body.tree.args.kwarg or body.tree.args.posonlyargs:
      self.error(node, "DSL helpers currently require named positional/keyword-only parameters")
    if self.helper_depth >= 32:
      self.error(node, "DSL helper expansion depth exceeds 32")
    positional = [a.arg for a in body.tree.args.args]
    parameters = positional + [a.arg for a in body.tree.args.kwonlyargs]
    args = [self.expression(a) for a in node.args] if arguments is None else list(arguments)
    kwargs = (
      {k.arg: self.expression(k.value) for k in node.keywords} if arguments is None else dict(keyword_arguments or {})
    )
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
    previous_mutable_names, self.mutable_names = self.mutable_names, set()
    self.env = dict(body.environment)
    self.env.update(bound)
    self.body = body
    self.helper_depth += 1
    self.helper_scopes.append((body, self.control_depth))
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
      if not body.inline and self.hardware_regions:
        self.error(node, "private T.func calls cannot cross an active hardware instruction region")
      if loop_returns or not body.inline:
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
      self.helper_scopes.pop()
      self.helper_depth -= 1
      self.env, self.body = previous_env, previous_body
      self.mutable_names = previous_mutable_names

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
      parent.statements.append(I.ExitRegion(context["name"], region, outlined=not body.inline))
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
    parent.statements.append(I.ExitRegion(context["name"], region, outlined=not body.inline))
    parent.statements.append(
      I.Assert(I.Expr(I.ScalarType("bool"), "local", data=flag), "T.func reached end without returning")
    )
    return context["result"]

  def map_rows(self, args, kwargs, node):
    """Stage a pure rank-one helper over a rank-two logical input.

    Scalar options are captured in the caller before staging. Reject effects
    rather than permitting writes to escape an independently scheduled row.
    The compiler never identifies a helper by its name or implementation.
    """
    from .capture import Body

    if len(args) != 2 or not isinstance(args[0], Body):
      self.error(node, "map_rows expects a T.func helper and a rank-two tile")
    source = self.matrix_operand(args[1], node)
    argument = I.Expr(I.TileType(source.type.dtype, (source.type.shape[1],), source.type.mask), "region_argument")
    if any(
      isinstance(v, I.View) or isinstance(v, I.Expr) and not isinstance(v.type, I.ScalarType) for v in kwargs.values()
    ):
      self.error(node, "map_rows options must be scalar values or immutable compile-time metadata")
    options = {key: self.freeze_tile(value) if isinstance(value, I.Expr) else value for key, value in kwargs.items()}
    parent, promoted = self.block, dict(self.promoted)
    self.block = I.Block()
    try:
      result = self.helper(args[0], node, arguments=(argument,), keyword_arguments=options)
      body = self.block
    finally:
      self.block, self.promoted = parent, promoted
    if (
      not isinstance(result, I.Expr)
      or not isinstance(result.type, I.TileType)
      or len(result.type.shape) != 1
      or not I.same_expr(result.type.shape[0], argument.type.shape[0])
      or any(
        not isinstance(s, (I.Assign, I.BindTile)) or isinstance(s, I.Assign) and not s.declare for s in body.statements
      )
    ):
      self.error(node, "map_rows helper must be a pure straight-line program returning one same-width tile")
    from ..ir.traversal import nodes

    if any(
      isinstance(value, I.Expr)
      and (
        self.operations.get(value).effects not in {"none", "read"} or self.operations.has(value, Trait.STATE_REFERENCE)
      )
      for value in nodes((body, result))
    ):
      self.error(node, "map_rows helper cannot mutate or capture tile state")
    typ = I.TileType(result.type.dtype, source.type.shape, result.type.mask)
    return I.Expr(typ, "map_rows", (source,), I.RowMap(argument, body, result))

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


def expand_helper(function, arguments, *, catalog, prefix, keyword_arguments=None, hardware_kind=None):
  """Expand a DSL helper into an executable region with a unique namespace.

  Unlike an expression-only adapter, this entry preserves ordinary control,
  effects and typed results. The caller must place the region at a legal
  execution site; helper source/captures are still handled by the frontend.
  """
  import inspect

  from ...template import TemplateIR
  from ..operations import use_catalog
  from .capture import capture
  from .parser import Frontend

  body = capture(function.function, inline=function.inline)
  ir = TemplateIR(prefix, inspect.Signature(), {}, (), None, (), None, body.filename, body)
  frontend = Frontend(ir, catalog=catalog)
  if hardware_kind is not None:
    frontend.hardware_regions.append((hardware_kind, ()))

  def unique(name="v"):
    frontend.counter += 1
    return f"{prefix}_{name}_{frontend.counter}"

  frontend.unique = unique
  call = ast.Call(func=ast.Name(id=function.__name__, ctx=ast.Load()), args=[], keywords=[])
  with use_catalog(catalog):
    value = frontend.helper(body, call, arguments=arguments, keyword_arguments=keyword_arguments)
    # A staged helper may borrow an owned caller buffer. Promotion is local
    # to this isolated frontend; publish those writes before returning its
    # statements, while leaving purely local/result values lazy.
    from ..ir.traversal import nodes

    borrowed = {
      v.storage for v in nodes((arguments, tuple((keyword_arguments or {}).values()))) if isinstance(v, I.View)
    }
    for storage in tuple(frontend.promoted):
      if storage in borrowed:
        frontend.materialize(storage)
  if hardware_kind is not None:
    frontend.validate_hardware_body(frontend.block, hardware_kind, call)
  return value, frontend.block.statements


def expand_register_helper(function, descriptors, *, catalog, prefix="register_library"):
  """Bind logical Vector inputs and stage a closed pure register program."""
  from ..dtypes import dtype_name
  from ..ir.vector import RegisterProgram

  inputs = []
  for ordinal, descriptor in enumerate(descriptors):
    dtype = dtype_name(descriptor.dtype)
    extent = (
      I.literal(descriptor.N)
      if descriptor.N is not None
      else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, descriptor.pow))
    )
    typ = I.VectorType(dtype, (extent,), tag=I.VectorTag(N=descriptor.N, pow=descriptor.pow, basis_dtype=dtype))
    inputs.append(I.Expr(typ, "register_input", data=ordinal))
  inputs = tuple(inputs)
  outputs, body = expand_helper(function, (inputs,), catalog=catalog, prefix=prefix)
  if any(not isinstance(statement, I.BindTile) for statement in body):
    raise TypeError("register helper must specialize to a pure expression tuple")
  if not isinstance(outputs, tuple) or not outputs:
    raise TypeError("register helper must return a nonempty expression tuple")
  operations, seen = [], set()

  def visit(expression):
    if id(expression) in seen:
      return
    if expression.op != "register_input" and catalog.get(expression).family != "register_geometry":
      raise TypeError("closed register helper requires register-geometry primitives")
    for argument in expression.args:
      visit(argument)
    seen.add(id(expression))
    operations.append(expression)

  # Retain operation-major staging order from the source helper before any
  # target scheduling. Output-only DFS can extend the lifetime of unconsumed
  # butterfly intermediates across many later output columns.
  for value in (*inputs, *(statement.value for statement in body), *outputs):
    visit(value)
  return RegisterProgram(inputs, tuple(operations), outputs)
