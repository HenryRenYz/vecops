"""Expand a pure DSL library helper for a semantic decomposition pass.

Keep algorithms source-parsed instead of maintaining a second hand-written
IR implementation. The prelude contains expression bindings and immutable
scalar captures, not observable stores or runtime control flow.
"""

import ast
import inspect

from ...template import TemplateIR
from .. import ir as I
from ..operations import use_catalog
from .capture import capture
from .diagnostics import DSLError
from .parser import Frontend


def expand_expression_helper(function, arguments, *, catalog, prefix):
  """Return a helper's expression and its capture/binding prelude.

  ``prefix`` must be unique within the enclosing kernel. This internal entry
  point intentionally rejects helpers with stores, state, or control flow;
  those require expansion in the original frontend's lexical environment.
  """
  body = capture(function.function)
  ir = TemplateIR(
    name=prefix,
    signature=inspect.Signature(),
    annotations={},
    captures=(),
    source=None,
    dependencies=(),
    recipe_id=None,
    origin=body.filename,
    body=body,
  )
  frontend = Frontend(ir, catalog=catalog)

  def unique(name="v"):
    frontend.counter += 1
    return f"{prefix}_{name}_{frontend.counter}"

  frontend.unique = unique
  call = ast.Call(func=ast.Name(id=function.__name__, ctx=ast.Load()), args=[], keywords=[])
  with use_catalog(catalog):
    value = frontend.helper(body, call, arguments=arguments)
  if not isinstance(value, I.Expr) or any(
    not isinstance(statement, (I.Assign, I.BindTile)) for statement in frontend.block.statements
  ):
    raise DSLError("expression helper must return an expression without side effects or runtime control flow")
  return value, frontend.block.statements
