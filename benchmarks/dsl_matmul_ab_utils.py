"""Shared exact-key selection and unfiltered three-label Matmul timing."""

from __future__ import annotations

import itertools
import json
import math
import statistics
import time

from dsl_matmul_cache_ab import native_calls
from dsl_matmul_source_catalog import load_cases, result_metadata, scenarios

_ORDERS = tuple(itertools.permutations(range(3)))


def balanced_order(round_):
  """Balance positions and directed neighbors over every six rounds."""
  return _ORDERS[round_ % len(_ORDERS)]


def select_cases(keys, *, svl_bytes=None):
  """Resolve original IDs exactly, without silently filtering a request."""
  if not keys or len(set(keys)) != len(keys):
    raise ValueError("execution IDs must be nonempty and unique")
  catalog = {case.execution_id: case for case in scenarios(load_cases(svl_bytes=svl_bytes), levels=("matmul",))}
  missing = set(keys) - catalog.keys()
  if missing:
    raise ValueError(f"original execution IDs unavailable: {sorted(missing)}")
  return tuple(catalog[key] for key in keys)


def time_case(scenario, compiled, values, rounds, stream, *, labels):
  """Keep native repeat, every clock reading and all original metadata.

  Numerical validation and offline packing belong outside these clocks.
  This records raw observations; it does not filter or frequency-adjust them.
  """
  if rounds <= 0 or rounds % len(_ORDERS):
    raise ValueError("positive rounds divisible by six are required")
  if len(labels) != 3 or len(set(labels)) != 3 or labels[-1] != "cpp":
    raise ValueError("two distinct DSL labels followed by cpp are required")
  repeat = max(1, min(10000, 8_000_000 // math.prod((scenario.case.m, scenario.case.n, scenario.case.k))))
  calls = native_calls(compiled, values, repeat)
  if len(calls) != len(labels):
    raise ValueError("exactly unchanged DSL, candidate DSL and C++ control required")
  for invoke in calls:
    invoke()
  samples = {label: [] for label in labels}
  for round_ in range(rounds):
    for position, index in enumerate(balanced_order(round_)):
      begin = time.perf_counter_ns()
      calls[index]()
      end = time.perf_counter_ns()
      if end <= begin:
        raise RuntimeError("nonpositive raw timing")
      ns = (end - begin) / repeat
      samples[labels[index]].append(ns)
      stream.write(
        json.dumps(
          {
            "execution_id": scenario.execution_id,
            "round": round_,
            "position": position,
            "label": labels[index],
            "repeat": repeat,
            "begin_ns": begin,
            "end_ns": end,
            "ns": ns,
          }
        )
        + "\n"
      )
      stream.flush()
  medians = {label: statistics.median(observations) for label, observations in samples.items()}
  return {
    **result_metadata(scenario),
    "median_ns": medians,
    "over_cpp": {label: value / medians["cpp"] for label, value in medians.items()},
  }
