"""Deterministic, bounded textual IR presentation; not an interchange format."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass


def format_ir(value, *, max_nodes=20000):
  from .._compiler import ir as I

  identifiers, definitions = {}, []

  def expression(e):
    key = id(e)
    if key in identifiers:
      return identifiers[key]
    name = f"%{len(identifiers)}"
    identifiers[key] = name
    if len(identifiers) > max_nodes:
      return "%truncated"
    arguments = ", ".join(render(a) for a in e.args)
    metadata = f" [{render(e.data)}]" if e.data is not None else ""
    definitions.append(f"{name}: {render(e.type)} = {e.op}({arguments}){metadata}")
    return name

  def render(v, depth=0):
    if depth > 24:
      return "..."
    if isinstance(v, I.Expr):
      return expression(v)
    if isinstance(v, I.ScalarType):
      return v.dtype
    if isinstance(v, I.TileType):
      tag = f", tag={render(v.tag)}" if isinstance(v, I.VectorType) else ""
      return f"{'mask' if v.mask else 'tile'}<{v.dtype}, ({', '.join(render(x) for x in v.shape)}){tag}>"
    if isinstance(v, I.Block):
      return "{\n" + "\n".join("  " + render(s, depth + 1).replace("\n", "\n  ") for s in v.statements) + "\n}"
    if isinstance(v, Mapping):

      def identity_key(k):
        return isinstance(k, int) and k > 2**32 or isinstance(k, tuple) and any(identity_key(x) for x in k)

      identity_keys = any(identity_key(k) for k in v)
      items = v.items() if identity_keys else sorted(v.items(), key=lambda p: str(p[0]))
      return (
        "{"
        + ", ".join(
          f"{'plan' + str(n) if identity_keys else render(k)}: {render(x, depth + 1)}" for n, (k, x) in enumerate(items)
        )
        + "}"
      )
    if isinstance(v, (tuple, list)):
      return "(" + ", ".join(render(x, depth + 1) for x in v) + ")"
    if isinstance(v, (set, frozenset)):
      return "{" + ", ".join(sorted(render(x, depth + 1) for x in v)) + "}"
    if is_dataclass(v):
      hidden = {"origins", "manager", "environment", "materializations", "plans", "pass_trace", "vector_regions"}
      parts = [f"{f.name}={render(getattr(v, f.name), depth + 1)}" for f in fields(v) if f.name not in hidden]
      return f"{type(v).__name__}(" + ", ".join(parts) + ")"
    if isinstance(v, (str, int, float, bool, type(None))):
      return repr(v)
    return f"<{type(v).__module__}.{type(v).__qualname__}>"

  body = render(value)
  return "\n".join(definitions) + ("\n\n" if definitions else "") + body + "\n"
