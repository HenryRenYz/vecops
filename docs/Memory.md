# Heterogeneous CPU memory

`vecops::memory` is the framework-neutral physical-memory layer used to place
CPU buffers on heterogeneous NUMA targets. It discovers an immutable topology,
ranks initiator-to-target paths, performs explicit direct allocations, and
reports process-local telemetry. V1 has no reserve, pool, caching allocator,
compaction, page migration, or garbage collector.

The CMake target is `vecops::memory`. `VECOPS_BUILD_MEMORY=AUTO` builds with
hwloc 2.7 or newer when its development files are available; otherwise it
builds a portable single-target backend. The portable backend rejects a Linux
machine exposing multiple NUMA nodes instead of presenting inaccurate
placement semantics. `VECOPS_BUILD_MEMORY=ON` makes hwloc mandatory and `OFF`
omits the component.

## C++ API

```cpp
#include "vecops/memory/Memory.h"

using namespace vecops::memory;

auto memory = MemorySystem::discover();
auto allocation = memory.allocate({
  .bytes = 256 * 1024 * 1024,
  .domain = CpuDomainSelector::current(),
  .intent = PlacementIntent::HighBandwidth,
  .fallback = FallbackPolicy::ToDefault,
  .alignment = 64,
});
```

`Allocation` is move-only. Destruction releases the exact mapping directly to
the backend. It retains shared ownership of the backend/topology state, so it
may safely outlive the `MemorySystem` value that created it. `data()` is stable
until move, `reset()`, or destruction.

Placement intent is explicit:

- `Default` selects the lowest-latency exact-locality target, then capacity.
- `HighBandwidth` selects the highest-bandwidth exact-locality target.
- `LowLatency` selects the lowest-latency exact-locality target.
- `ExactTarget` requires an OS physical NUMA index.

`objective_rank` selects from the corresponding bandwidth/latency tier view.
The rank is meaningful only together with its objective and topology snapshot.
OS NUMA indexes, not hwloc logical indexes, are used by configuration and
diagnostics.

Target configuration is partial. `TargetOverride` supplies kind and allocation
budgets; `PathOverride` supplies initiator-relative bandwidth or latency. An
explicit configuration value takes precedence over a discovered hwloc value.
The `max_managed_bytes` and `min_free_bytes` fields are admission budgets, not
reservations and not guarantees that later page faults cannot fail.

## Python API

```python
from vecops import memory

system = memory.System()
print(system.describe())

owner = system.allocate(256 << 20, placement="high_bandwidth")
view = memoryview(owner)

array = memory.numpy.zeros(
  (4096, 4096),
  dtype="float32",
  system=system,
  placement="high_bandwidth",
)
```

`Buffer` implements the buffer protocol. NumPy arrays retain it as their base
owner; no address-only Python object is exposed. `topology()`, `tiers()`, and
`stats()` return dictionaries/lists suitable for logs and policy code.

## Workspace integration

`MemoryWorkspaceArenaProvider` maps logical `WorkspaceTier::Fast` to
`HighBandwidth` and `WorkspaceTier::Slow` to `Default`:

```cpp
auto provider = std::make_shared<MemoryWorkspaceArenaProvider>(
  memory,
  CpuDomainSelector::current(),
  fast_capacity);
WorkspaceReplayCache cache(4, provider);
```

The capacity is a placement limit. A cache entry allocates only the exact arena
sizes selected by `place_workspace()`. If allocation of a preferred fast arena
fails, the cache discards the partial owners, recomputes the complete plan with
zero fast capacity, and allocates it in slow/default memory.

The default provider remains aligned heap memory, preserving existing behavior.
The default replay cache also performs its first trace without a caller fast
arena; a `FastRequired` request therefore needs an externally managed
`WorkspaceContext` whose fast storage exists during the trace. It cannot use
the default cache's preferred-only fallback path.

## Lifetime and threading

Topology is immutable. Target counters are atomic and budget admission is
race-safe within one `MemorySystem` state. `os_free_bytes` is an instantaneous
kernel observation, while `managed_bytes` counts only live allocations made by
that state. Physical page placement should be validated after touching pages,
for example with `hwloc_get_area_memlocation()` or `/proc/self/numa_maps`.
