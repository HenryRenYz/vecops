# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scheduling hints, separate from artifact validity or execution semantics."""

import json
import math
import os
import re
import shlex
from pathlib import Path


def order_requests(requests):
  """Start historically expensive sources first; use source size when unknown.

  Equal estimates retain collection order. The optional history is only a hint:
  stale values can affect latency, never cache keys, validation or results.
  """
  profile = cost_profile_path()
  try:
    costs = json.loads(profile.read_text())
  except (OSError, ValueError):
    costs = {}
  if not isinstance(costs, dict):
    costs = {}
  def estimate(request):
    if not request.source:
      return 0
    path = Path(request.source)
    cost = costs.get(str(path))
    if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
      return cost
    try:
      return path.stat().st_size / 1000
    except OSError:
      return 0
  return tuple(sorted(requests, key=estimate, reverse=True))


def observe_batch(log_path):
  """Read optional hints without throwing from a native build observer."""
  try:
    return _observe_batch(log_path)
  except (OSError, ValueError, TypeError, KeyError):
    return {}


def _observe_batch(log_path):
  """Extract source costs from Ninja's recorded object command durations."""
  build = Path(log_path).parent
  try:
    commands = json.loads((build / "compile_commands.json").read_text())
    rows = (build / ".ninja_log").read_text().splitlines()
  except (OSError, ValueError):
    return {}
  sources = {}
  def absolute(path, directory):
    path = Path(path)
    return str((path if path.is_absolute() else directory / path).resolve())
  for command in commands:
    output = command.get("output")
    if output is None:
      try:
        arguments = command.get("arguments") or shlex.split(command.get("command", ""))
        output = arguments[arguments.index("-o") + 1]
      except (ValueError, IndexError):
        continue
    directory = Path(command.get("directory", build))
    path = Path(absolute(command["file"], directory))
    try:
      includes = re.findall(r'#include "([^"]+\.cpp)"', path.read_text())
    except OSError:
      continue
    if includes:
      sources[absolute(output, directory)] = absolute(includes[0], path.parent)
  costs = {}
  for row in rows:
    fields = row.split("\t")
    if len(fields) >= 5 and (output := absolute(fields[3], build)) in sources:
      source = sources[output]
      seconds = (int(fields[1])-int(fields[0])) / 1000
      if seconds >= 0:
        costs[source] = max(costs.get(source, 0), seconds)
  return costs


def cost_profile_path():
  from ._compiler import default_cache_dir
  configured = os.environ.get("VECOPS_BUILD_COST_PROFILE")
  return Path(configured).expanduser() if configured else default_cache_dir() / "build-costs-v1.json"


def save_observed_costs(costs):
  """Update ordinary cache history; an explicit user-supplied profile is read-only."""
  if not costs or os.environ.get("VECOPS_BUILD_COST_PROFILE"):
    return
  from ._cache_lock import file_lock
  path = cost_profile_path()
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
      try:
        records = json.loads(path.read_text())
      except (OSError, ValueError):
        records = {}
      if not isinstance(records, dict):
        records = {}
      records.update(costs)
      temporary = path.with_suffix(f".{os.getpid()}.new")
      temporary.write_text(json.dumps(records, sort_keys=True, indent=2) + "\n")
      temporary.replace(path)
  except OSError:
    # An unavailable scheduling-hint file cannot invalidate a successful build.
    return
