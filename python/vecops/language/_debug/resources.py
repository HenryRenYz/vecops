# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU worker coverage, memory access plans and IR register estimates.

Never executes a kernel, allocates outputs or claims hardware occupancy. Plans
are conditional when runtime guards or data-dependent domains are unknown.
"""

import math
import os
import platform
from pathlib import Path

from .._compiler import ir as I
from .._compiler.ir.traversal import nodes
from .._compiler.ir.vector import VectorRegion
from .evaluate import bind, itemsize, scalar


def execution_environment():
  from ... import _C
  from ..._arm_vector_state import get_vector_state

  affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
  quota = None
  # cgroup-v2 paths are relative to their mounted hierarchy, not always root.
  try:
    group = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    root = Path("/sys/fs/cgroup")
    current = root / group.lstrip("/")
    quotas = []
    for directory in (current, *current.parents):
      if not directory.is_relative_to(root):
        break
      path = directory / "cpu.max"
      if path.is_file():
        q, period = path.read_text().split()
        if q != "max":
          quotas.append(int(q) / int(period))
    quota = min(quotas) if quotas else None
  except (OSError, ValueError, StopIteration):
    pass
  frequencies = {}
  if affinity:
    base = Path(f"/sys/devices/system/cpu/cpu{affinity[0]}/cpufreq")
    for key in ("scaling_cur_freq", "scaling_max_freq", "scaling_min_freq", "scaling_governor"):
      if (base / key).is_file():
        frequencies[key] = (base / key).read_text().strip()
  return {
    "machine": platform.machine(),
    "arm_vector_lengths": vars(get_vector_state()),
    "affinity": affinity,
    "quota_cpus": quota,
    "native_executor_capacity": _C.cpu_thread_pool_capacity(),
    "frequency": frequencies,
    "numa_nodes": tuple(p.name for p in Path("/sys/devices/system/node").glob("node[0-9]*")),
  }


def nonoverlapping(shape, strides):
  covered = 1
  for stride, extent in sorted((abs(s), n) for n, s in zip(shape, strides) if n > 1):
    if stride < covered:
      return False
    covered += (extent - 1) * stride
  return True


def layouts(owner, args=(), kwargs=None):
  _, tensors = bind(owner, args, kwargs or {})
  result = {}
  for name, descriptor in tensors.items():
    shape, strides = descriptor["shape"], descriptor["strides"]
    lo = sum(min(0, (n - 1) * s) for n, s in zip(shape, strides)) if all(shape) else 0
    hi = sum(max(0, (n - 1) * s) for n, s in zip(shape, strides)) + 1 if all(shape) else 0
    dense, expected = True, 1
    for n, s in reversed(tuple(zip(shape, strides))):
      dense &= n <= 1 or s == expected
      expected *= n
    dtype = descriptor["dtype"]
    access = descriptor["annotation"].access
    kernel = getattr(owner, "kernel_ir", None)
    if kernel is not None:
      access = kernel.parameters[name].access
    elif hasattr(owner, "out_idx"):
      parameters = tuple(owner.template.ir.signature.parameters)
      if parameters.index(name) in owner.out_idx and access == "input":
        access = "output"
    result[name] = {
      "shape": shape,
      "strides": strides,
      "dtype": dtype,
      "logical_bytes": math.prod(shape) * itemsize(dtype),
      "storage_span_bytes": (hi - lo) * itemsize(dtype),
      "dense_row_major": dense,
      "unit_stride_axes": tuple(i for i, s in enumerate(strides) if s == 1),
      "nonoverlapping": nonoverlapping(shape, strides),
      "access": access,
      "automatically_allocated": descriptor["value"] is None,
    }
  return result


def _liveness(region):
  last = {op.index: op.index for op in region.operations}
  for op in region.operations:
    for operand in op.operands:
      if operand is not None:
        last[operand] = max(last.get(operand, 0), op.index)
  for root in region.root_indices:
    if root is not None:
      last[root] = len(region.operations)
  live, peak, mask_peak = {}, 0, 0
  for op in region.operations:
    # Logical words, not a promise that the native allocator uses this many
    # registers. Subwords still occupy at least one physical register.
    words = max(1, math.ceil(2**region.base_power)) * region.unroll
    live[op.index] = (words, op.type.mask)
    peak = max(peak, sum(n for n, mask in live.values() if not mask))
    mask_peak = max(mask_peak, sum(n for n, mask in live.values() if mask))
    live = {index: value for index, value in live.items() if last[index] > op.index}
  return {
    "vector_words_peak": peak,
    "mask_words_peak": mask_peak,
    "unroll": region.unroll,
    "basis": "IR DAG liveness before native allocation; excludes address GPRs, backend temporaries and loop-carried accumulators",
  }


def analyze(owner, args=(), kwargs=None):
  from ... import _C

  if not hasattr(owner, "kernel_ir"):
    owner = owner.lower()
  if owner.kernel_ir is None:
    return {
      "kind": "cpu-resource-estimate",
      "unavailable": "source-backed kernel has no DSL IR",
      "environment": execution_environment(),
    }
  environment, _tensors = bind(owner, args, kwargs or {})
  report = {
    "kind": "cpu-resource-estimate",
    "environment": execution_environment(),
    "layouts": layouts(owner, args, kwargs),
    "phases": [],
    "scratch": [],
    "register_estimates": [],
    "notes": [
      "not GPU occupancy; costs are profile estimates",
      "nested/external executors and runtime alias guards can change worker selection",
    ],
  }
  backend = owner.backend_ir
  compiler_environment = backend.vectors.manager.options["_compiler_environment"]
  costs = compiler_environment.costs
  environment["__debug_cpu_target"] = compiler_environment.target
  environment["__debug_vector_state"] = compiler_environment.vector_state
  capacity = report["environment"]["native_executor_capacity"]
  for node in nodes(owner.kernel_ir.body):
    if isinstance(node, I.Grid):
      plan = backend.grids[id(node)]
      phase = {
        "kind": "grid",
        "schedule": "dynamic" if node.schedule == "dynamic" else "static",
        "chunk": node.chunk,
        "ownership_proven": plan.safe,
        "reason": plan.reason,
        "conditional_on_runtime_guards": bool(plan.guards),
      }
      try:
        count = math.prod(int(scalar(n, environment)) for n in node.extents)
        cycles = float(scalar(plan.cycles, environment))
        workers = (
          _C.grid_workers_available(capacity, 0, count, cycles, costs.policy("grid_worker_cycles", 2000))
          if plan.safe and node.parallel != "serial"
          else min(1, count)
        )
        phase.update(
          programs=count,
          selected_workers=workers,
          estimated_program_cycles=cycles,
          worker_coverage=workers / capacity if capacity else 0,
          equal_cost_wave_efficiency=count / (workers * math.ceil(count / workers)) if workers else 1,
        )
      except (ValueError, KeyError):
        phase["unknown"] = "data-dependent domain or scalable vector length"
      report["phases"].append(phase)
    view = node.view if isinstance(node, I.Allocate) else None
    if view:
      try:
        size = math.prod(int(scalar(n, environment)) for n in view.shape) * itemsize(view.dtype)
      except (ValueError, KeyError):
        size = None
      slot = backend.storage.slots.get(view.storage, view.storage) if backend.storage is not None else view.storage
      report["scratch"].append(
        {"storage": view.storage, "physical_slot": slot, "bytes": size, "scope": "invocation or worker-private"}
      )
  regions, seen = [], set()
  for region in (*owner.vector_ir.regions, *backend.stores.values(), *backend.state_regions.values()):
    if isinstance(region, VectorRegion) and id(region) not in seen:
      seen.add(id(region))
      regions.append(region)
  report["register_estimates"] = [_liveness(region) for region in regions]
  report["memory_lowering"] = {
    "transpose_plans": len(backend.transposes),
    "scratch_slots": len(set(backend.storage.slots.values())) if backend.storage is not None else 0,
    "snapshots": len(backend.snapshots),
    "materialized_values": len(owner.kernel_ir.materializations),
  }
  report["hardware_state"] = tuple(
    dict.fromkeys(getattr(n, "kind", "") for n in nodes(owner.kernel_ir.body) if isinstance(n, I.HardwareRegion))
  )
  return report
