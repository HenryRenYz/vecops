# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit CPU timings, correctness summaries and Chrome/Perfetto traces."""

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from ..._dtype import bfloat16, normalize_dtype
from .resources import execution_environment


@dataclass(frozen=True)
class ProfileResult:
  samples_ns: tuple
  warmup: int
  mode: str
  environment: dict
  events: tuple
  allocations: tuple = ()
  memory_events: tuple = ()

  @property
  def median_ns(self):
    return statistics.median(self.samples_ns)

  @property
  def p95_ns(self):
    values = sorted(self.samples_ns)
    import math

    return values[max(0, math.ceil(len(values) * 0.95) - 1)]

  @property
  def memory_summary(self):
    live, peak = {}, 0
    for event in self.memory_events:
      if event.get("action", 0) >= 2:
        continue
      if event["released"]:
        live.pop(event["allocation_id"], None)
      else:
        live[event["allocation_id"]] = event["bytes"]
      peak = max(peak, sum(live.values()))
    return {
      "instrumented_peak_bytes": peak,
      "live_bytes_at_end": sum(live.values()),
      "workspace_allocated_bytes": sum(e.get("bytes", 0) * e.get("replicas", 1) for e in self.allocations),
      "dropped": sum(e.get("dropped", 0) for e in self.allocations),
      "coverage": "instrumented DSL scratch lifetimes and workspace allocation sites, not all process mallocs",
    }

  def export_trace(self, path):
    Path(path).write_text(
      json.dumps(
        {
          "traceEvents": self.events,
          "displayTimeUnit": "ns",
          "metadata": {
            "environment": self.environment,
            "allocations": self.allocations,
            "memory_summary": self.memory_summary,
          },
        },
        indent=2,
      )
    )
    return Path(path)


def profile(owner, args, *, warmup=10, repeat=100, mode="python", kernel_kwargs=None, trace=False, memory=False):
  """Compile outside timed samples. Timing changes/execution are opt-in only.

  native mode prebinds the native KernelCall and skips Python output allocation
  and argument binding, but retains provider dispatch; it is not pure ISA time.
  Traces describe host kernel calls/build phases. Worker events require native
  instrumentation and are not invented from scheduling estimates.
  """
  import contextlib

  if type(warmup) is not int or warmup < 0 or type(repeat) is not int or repeat < 1:
    raise ValueError("warmup must be nonnegative and repeat must be positive")
  if mode not in ("python", "native"):
    raise ValueError("profile mode must be python or native")
  if not hasattr(owner, "_core"):
    raise TypeError("profile requires a compiled callable kernel, not a template/lowered object")
  kwargs = dict(kernel_kwargs or {})
  events = []
  start = time.perf_counter_ns()
  owner.compile_for(*args, **kwargs)
  if trace:
    events.append(
      {
        "name": "prepare",
        "cat": "compile",
        "ph": "X",
        "ts": start / 1000,
        "dur": (time.perf_counter_ns() - start) / 1000,
        "pid": 1,
        "tid": 0,
      }
    )
  if mode == "native":
    from ... import _C

    ordered = owner._materialize_outputs(owner._arguments(args, kwargs))
    # Convert each NumPy/torch tensor to a non-owning ABI view once.
    parameters = owner.kernel_def.parameters
    from ..._schema import TensorDef

    if any(hasattr(value, "detach") for value in ordered):
      target = owner.explicit_torch_function()
      invoke = lambda: target(*ordered)
      mode = "native-bridge"
    else:
      values = [
        _C.TensorView(value, writable=parameter.output)
        if isinstance(parameter, TensorDef) and value is not None
        else value
        for value, parameter in zip(ordered, parameters)
      ]
      call = _C.KernelCall(values, owner._core.specialization_values)
      invoke = lambda: owner.operator.native.invoke(call)
  else:
    invoke = lambda: owner(*args, **kwargs)
  for _ in range(warmup):
    invoke()
  allocations, memory_events = (), ()
  if memory or trace:
    from ...workspace_observer import drain, drain_memory, tracking

    context = tracking()
  else:
    context = contextlib.nullcontext()
  samples = []
  with context:
    for index in range(repeat):
      begin = time.perf_counter_ns()
      invoke()
      finish = time.perf_counter_ns()
      samples.append(finish - begin)
      if trace:
        events.append(
          {
            "name": owner.template.ir.name,
            "cat": "kernel",
            "ph": "X",
            "ts": begin / 1000,
            "dur": (finish - begin) / 1000,
            "pid": 1,
            "tid": 0,
            "args": {"sample": index, "mode": mode},
          }
        )
    if memory or trace:
      memory_events = tuple(drain_memory())
      allocations = tuple(drain())
      if trace:
        live = {}
        for event in memory_events:
          if event.get("action", 0) >= 2:
            events.append(
              {
                "name": event["site"],
                "cat": "native",
                "ph": "B" if event["action"] == 2 else "E",
                "ts": event["timestamp_ns"] / 1000,
                "pid": 1,
                "tid": event["worker"],
              }
            )
            continue
          if event["released"]:
            live.pop(event["allocation_id"], None)
          else:
            live[event["allocation_id"]] = event["bytes"]
          events.append(
            {
              "name": "DSL scratch live bytes",
              "cat": "memory",
              "ph": "C",
              "ts": event["timestamp_ns"] / 1000,
              "pid": 1,
              "tid": event["worker"],
              "args": {"bytes": sum(live.values())},
            }
          )
  return ProfileResult(tuple(samples), warmup, mode, execution_environment(), tuple(events), allocations, memory_events)


def compare(actual, expected, *, rtol=1e-5, atol=1e-6, equal_nan=True):
  """Numerical report with explicit NaN/Inf policy and same-dtype FP ULPs."""
  import numpy as np

  def array(value):
    if hasattr(value, "detach"):
      value = value.detach().cpu()
      if normalize_dtype(value.dtype) == bfloat16:
        value = value.float()
      value = value.numpy()
    return np.asarray(value)

  a, b = array(actual), array(expected)
  if a.shape != b.shape:
    return {"ok": False, "reason": "shape mismatch", "actual_shape": a.shape, "expected_shape": b.shape}
  close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
  finite = np.isfinite(a) & np.isfinite(b)
  difference = np.zeros(a.shape, dtype=np.float64)
  np.subtract(a.astype(np.float64), b.astype(np.float64), where=finite, out=difference)
  np.abs(difference, out=difference)
  relative = difference / np.maximum(np.abs(b.astype(np.float64)), np.finfo(np.float64).tiny)
  ulp = None
  if a.dtype == b.dtype and a.dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
    uint = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize]
    sign = uint(1 << (8 * a.itemsize - 1))

    def ordered(x):
      bits = x.view(uint)
      return np.where(bits & sign, ~bits, bits | sign)

    x, y = ordered(a), ordered(b)
    distances = np.maximum(x, y) - np.minimum(x, y)
    ulp = int(distances[finite].max(initial=0))
  mismatch = tuple(int(x) for x in np.argwhere(~close)[0]) if not close.all() else None
  return {
    "ok": bool(close.all()),
    "max_abs": float(difference.max(initial=0)),
    "max_rel": float(relative.max(initial=0)),
    "max_ulp": ulp,
    "first_mismatch": mismatch,
    "nan_actual": int(np.isnan(a).sum()),
    "nan_expected": int(np.isnan(b).sum()),
    "inf_actual": int(np.isinf(a).sum()),
    "inf_expected": int(np.isinf(b).sum()),
    "rtol": rtol,
    "atol": atol,
  }
