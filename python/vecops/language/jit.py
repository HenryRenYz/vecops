# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
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
  pipeline=None,
  pass_options=None,
  debug=None,
):
  """Cache a Python kernel factory and compile only returned TemplateFunctions.

  Outer-function annotations are intentionally ignored. Actual argument values
  are normalized recursively before both cache-key construction and execution.
  ``pipeline`` and ``pass_options`` are forwarded to :func:`T.compile` only
  for returned templates. Pass options are copied at decoration time; their
  values, like a compile pipeline, must remain immutable for the decorator.
  """
  selected_options = None if pass_options is None else dict(pass_options)

  def decorate(target: Callable[..., Any]):
    from ._debug.report import DebugInfo, InspectionMixin

    signature = inspect.signature(target)
    cache: dict[tuple[Any, ...], Any] = {}
    in_flight: dict[tuple[Any, ...], threading.Condition] = {}
    lock = threading.RLock()
    local = threading.local()

    @functools.wraps(target)
    def wrapped(*args, **kwargs):
      bound, key = _normalized_binding(signature, args, kwargs)
      from .._arm_vector_state import get_vector_state
      from .._identity import callable_identity, implementation_identity
      from ._compiler.environment import registry_fingerprint

      key = (
        key,
        get_vector_state().env_token,
        registry_fingerprint(),
        implementation_identity(),
        callable_identity(target, include_state=False),
      )
      active = getattr(local, "active", set())
      if key in active:
        raise RuntimeError(f"recursive T.jit elaboration cycle in {target.__qualname__}")
      with lock:
        while True:
          if key in cache:
            candidate = cache[key]
            info = getattr(candidate, "debug", None)
            if info is None or info._validate_cache():
              if info is not None:
                info._cache["decorator_hit"] = True
                wrapped.debug = info
              return candidate
            del cache[key]
          condition = in_flight.get(key)
          if condition is None:
            in_flight[key] = threading.Condition(lock)
            break
          condition.wait()
      local.active = active | {key}
      try:
        result = target(*bound.args, **bound.kwargs)
        if isinstance(result, TemplateFunction):
          result = compile(
            result,
            out_idx=out_idx,
            compiler=compiler,
            pipeline=pipeline,
            pass_options=selected_options,
            debug=debug,
          )
        elif callable(result):
          from ._debug.report import attach_composite

          attach_composite(result)
        if hasattr(result, "debug"):
          wrapped.debug = result.debug
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
        wrapped.debug = DebugInfo(name=target.__qualname__)

    def cache_info():
      with lock:
        return {"size": len(cache), "in_flight": len(in_flight)}

    wrapped.__signature__ = signature
    wrapped.cache_clear = cache_clear
    wrapped.cache_info = cache_info
    wrapped.debug = DebugInfo(name=target.__qualname__)
    for method in (
      "get_kernel_source",
      "get_host_source",
      "get_ir",
      "get_pass_report",
      "get_cache_info",
      "get_build_report",
      "get_assembly",
      "explain",
      "export_debug",
    ):
      setattr(wrapped, method, getattr(InspectionMixin, method).__get__(wrapped))
    return wrapped

  if function is not None:
    return decorate(function)
  return decorate


__all__ = ["jit"]
