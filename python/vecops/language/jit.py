"""Python-metaprogram JIT wrapper with immutable call normalization."""

from __future__ import annotations

import functools
import inspect
import threading
from typing import Any, Callable

from .compile import compile
from .expr import cache_key, normalize_value
from .template import TemplateFunction


def _normalized_binding(signature: inspect.Signature, args, kwargs):
  bound = signature.bind(*args, **kwargs)
  bound.apply_defaults()
  for name, parameter in signature.parameters.items():
    value = bound.arguments[name]
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
      bound.arguments[name] = tuple(normalize_value(item, name=f"{name}_{index}") for index, item in enumerate(value))
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
      bound.arguments[name] = {key: normalize_value(item, name=key) for key, item in value.items()}
    else:
      bound.arguments[name] = normalize_value(value, name=name)
  key_items = []
  for name, parameter in signature.parameters.items():
    value = bound.arguments[name]
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
      key_items.append((name, tuple((key, cache_key(item)) for key, item in sorted(value.items()))))
    else:
      key_items.append((name, cache_key(value)))
  return bound, tuple(key_items)


def jit(
  function: Callable[..., Any] | None = None,
  *,
  out_idx: int | list[int] | tuple[int, ...] | None = None,
  compiler=None,
):
  """Cache a Python kernel factory and compile only returned TemplateFunctions.

  Outer-function annotations are intentionally ignored. Actual argument values
  are normalized recursively before both cache-key construction and execution.
  """

  def decorate(target: Callable[..., Any]):
    signature = inspect.signature(target)
    cache: dict[tuple[Any, ...], Any] = {}
    in_flight: dict[tuple[Any, ...], threading.Condition] = {}
    lock = threading.RLock()
    local = threading.local()

    @functools.wraps(target)
    def wrapped(*args, **kwargs):
      bound, key = _normalized_binding(signature, args, kwargs)
      active = getattr(local, "active", set())
      if key in active:
        raise RuntimeError(f"recursive T.jit elaboration cycle in {target.__qualname__}")
      with lock:
        while True:
          if key in cache:
            return cache[key]
          condition = in_flight.get(key)
          if condition is None:
            in_flight[key] = threading.Condition(lock)
            break
          condition.wait()
      local.active = active | {key}
      try:
        result = target(*bound.args, **bound.kwargs)
        if isinstance(result, TemplateFunction):
          result = compile(result, out_idx=out_idx, compiler=compiler)
      except BaseException:
        with lock:
          in_flight.pop(key).notify_all()
        raise
      finally:
        local.active = active
      with lock:
        cache[key] = result
        in_flight.pop(key).notify_all()
      return result

    def cache_clear():
      with lock:
        cache.clear()

    def cache_info():
      with lock:
        return {"size": len(cache), "in_flight": len(in_flight)}

    wrapped.__signature__ = signature
    wrapped.cache_clear = cache_clear
    wrapped.cache_info = cache_info
    return wrapped

  if function is not None:
    return decorate(function)
  return decorate


__all__ = ["jit"]
