"""Immutable source/helper capture and semantic cache identity."""

from __future__ import annotations
import ast
import hashlib
import inspect
import textwrap
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from ...dsl import DSLFunction, Intrinsic, Vector, ExecutionLayout, Reducer
from ...expr import Const, DType, Dynamic, IntExpr
from .. import ir as I
from ..dtypes import dtype_name
from .diagnostics import DSLError

VERSION = "vecops-dsl-8"
DSL_MODULES = {
  "vecops",
  "vecops.language",
  "vecops.language.dsl",
  "vecops.language.sme",
  "vecops.language.amx",
  "vecops.language.library",
}


def symbol(value):
  if isinstance(value, Const):
    return I.literal(value.value)
  if isinstance(value, Dynamic):
    return I.Expr(I.ScalarType("int64"), "symbol", data=value.name)
  if isinstance(value, IntExpr):
    return I.binary(value.op, symbol(value.left), symbol(value.right))
  if isinstance(value, (bool, int, float)):
    return I.literal(value)
  return value


@dataclass(frozen=True)
class Body:
  tree: ast.FunctionDef
  environment: object
  identity: str
  filename: str
  first_line: int


@dataclass(frozen=True)
class CapturedReducer:
  body: Body
  identity: object
  associative: bool


def capture(function, local_scope=None, seen=None):
  seen = set() if seen is None else seen
  if id(function) in seen:
    raise DSLError("recursive DSL helper dependency")
  seen = seen | {id(function)}
  try:
    lines, first_line = inspect.getsourcelines(function)
  except (OSError, TypeError) as error:
    raise DSLError("DSL kernels require inspectable Python source") from error
  tree = ast.parse(textwrap.dedent("".join(lines)))
  target = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
  if target is None:
    raise DSLError("DSL definition must be a regular Python function")
  closure = inspect.getclosurevars(function)
  namespace = dict(function.__globals__)
  namespace.update(local_scope or {})
  namespace.update(closure.nonlocals)
  environment = {}
  fingerprints = []
  # Python's closure/name inventory avoids capturing unrelated factory locals.
  references = {
    node.id
    for statement in target.body
    for node in ast.walk(statement)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
  }
  for name in sorted((set(function.__code__.co_freevars) | set(function.__code__.co_names)) & references):
    if name not in namespace:
      continue
    value = namespace[name]
    if isinstance(value, Reducer):
      body = capture(value.function, seen=seen)
      key = (body.identity, value.identity, value.associative)
      value = CapturedReducer(body, value.identity, value.associative)
    elif isinstance(value, DSLFunction):
      value = capture(value.function, seen=seen)
      key = value.identity
    elif isinstance(value, type) and value in (Vector, ExecutionLayout):
      key = value.__name__
    elif isinstance(value, (Const, Dynamic, IntExpr)):
      value = symbol(value)
      key = repr(value)
    elif isinstance(value, DType):
      value = dtype_name(value)
      key = ("dtype", value)
    elif isinstance(value, ModuleType):
      if value.__name__ not in DSL_MODULES:
        raise DSLError(f"unsupported captured module {value.__name__}")
      key = value.__name__
    elif isinstance(value, (str, bool, int, float, type(None), Intrinsic, Vector, ExecutionLayout)):
      key = repr(value)
    elif isinstance(value, tuple):

      def freeze_tuple(item):
        if isinstance(item, tuple):
          return tuple(freeze_tuple(v) for v in item)
        if isinstance(item, (Const, Dynamic, IntExpr)):
          return symbol(item)
        if isinstance(item, DType):
          return dtype_name(item)
        if isinstance(item, (str, bool, int, float, type(None), Vector, Intrinsic)):
          return item
        try:
          return dtype_name(item)
        except (TypeError, ValueError) as error:
          raise DSLError(f"capture {name!r} must be immutable kernel data") from error

      value = freeze_tuple(value)
      key = repr(value)
    else:
      try:
        value = dtype_name(value)
        key = ("dtype", value)
      except (TypeError, ValueError) as error:
        raise DSLError(f"unsupported mutable/host capture {name!r}: {type(value).__name__}") from error
    environment[name] = value
    fingerprints.append((name, key))
  # Exclude decorators and source locations from semantic identity.
  target.decorator_list = []
  for node in ast.walk(target):
    if not isinstance(node, ast.Attribute):
      continue
    attributes = []
    root = node
    while isinstance(root, ast.Attribute):
      attributes.append(root.attr)
      root = root.value
    if not isinstance(root, ast.Name):
      continue
    value = environment.get(root.id)
    for attribute in reversed(attributes):
      if not isinstance(value, ModuleType) or value.__name__ not in DSL_MODULES:
        value = None
        break
      value = getattr(value, attribute, None)
    if isinstance(value, DSLFunction):
      value = capture(value.function, seen=seen)
      key = value.identity
    elif isinstance(value, Intrinsic):
      key = value.name
    else:
      continue
    name = ".".join((root.id, *reversed(attributes)))
    environment[name] = value
    fingerprints.append((name, key))
  identity = hashlib.sha256(
    (VERSION + ast.dump(target, include_attributes=False) + repr(fingerprints)).encode()
  ).hexdigest()
  return Body(
    target, MappingProxyType(environment), identity, inspect.getsourcefile(function) or "<kernel>", first_line
  )
