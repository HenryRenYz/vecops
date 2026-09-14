# Heterogeneous CPU memory

`vecops::memory` is the framework-neutral physical-memory layer used to place
CPU buffers on heterogeneous NUMA targets. It discovers an immutable topology,
ranks initiator-to-target paths, performs explicit direct allocations, and
reports process-local telemetry. V1 has no reserve, pool, caching allocator,
compaction, page migration, or garbage collector.

Large pages are preferred by default without exposing Linux backing mechanisms
in the public policy API. A Linux allocation reserves one contiguous virtual
range, maps as much of its aligned prefix as possible from target-local
HugeTLB pools, and fills the remainder with an anonymous mapping carrying a
THP hint. The result remains one pointer on one memory target even when its
physical backing mixes large and ordinary pages. `use_large_pages=false`
disables HugeTLB and applies `MADV_NOHUGEPAGE` to the ordinary mapping.

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
  // .use_large_pages = false,  // optional; the default is true
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

Admission accounts for both ordinary `MemFree` and free per-node HugeTLB
capacity when large pages are enabled. `managed_large_page_bytes` reports the
portion guaranteed to use explicit large pages. The remaining bytes are an
ordinary mapping and may still be promoted to THP asynchronously.

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

with system.workspace_session(fast_capacity=3 << 30, slow_capacity=8 << 30):
  # Registered Python and framework operators share these bounded arenas on
  # the calling thread.
  run_model()

# The only page-policy switch; both direct allocations and sessions default
# to True.
ordinary = system.allocate(64 << 20, large_pages=False)
```

`Buffer` implements the buffer protocol. NumPy arrays retain it as their base
owner; no address-only Python object is exposed. `topology()`, `tiers()`, and
`stats()` return dictionaries/lists suitable for logs and policy code.

`workspace_session()` constructs one fast and one slow physical arena, freezes
the selected CPU domain, and installs the session on the calling thread for one
non-overlapping execution region. The capacities are totals for the session,
not allowances multiplied by JIT artifacts or cached shapes. `__exit__()` and
`close()` release both physical allocations; a session is deliberately
single-use so stale DSO plan metadata can never reuse released addresses.

## Workspace integration

`execution::MemoryWorkspaceSession` maps logical `WorkspaceTier::Fast` to
`HighBandwidth` and `WorkspaceTier::Slow` to `Default`:

```cpp
auto provider = std::make_shared<execution::MemoryWorkspaceSession>(
  memory, execution::MemoryWorkspaceSessionConfig{
    .fast_capacity = fast_capacity,
    .slow_capacity = slow_capacity,
  });
WorkspaceReplayCache cache(4, provider);
```

The session allocates its bounded arenas once and every cache entry binds a
view of the same bases. Calls sharing a session must therefore be synchronous
and non-overlapping. If initial fast allocation fails for a recoverable reason
and fallback is enabled, its effective fast capacity becomes zero and plans are
placed wholly in the slow/default arena.

The default provider remains aligned heap memory. A shared-arena session also
supplies its bases to the initial trace, so the first real invocation uses the
selected physical memory. Cache materialization happens after that invocation;
failure to retain an optimization never changes an already-successful call
into an error.

Generated artifacts receive providers through `VecopsWorkspaceArenaProvider`,
a C-only retain/allocate/release callback table. This avoids relying on a C++
singleton across separately linked JIT DSOs. `VecopsExecutionContext` carries
the provider in a dedicated field, independently of embedding `user_data`.
Provider identity selects plan metadata in each DSO, while physical ownership
stays in the explicit session and ends at `close()`.

## Lifetime and threading

Topology is immutable. Target counters are atomic and budget admission is
race-safe within one `MemorySystem` state. `os_free_bytes` is an instantaneous
kernel observation, while `managed_bytes` counts only live allocations made by
that state. Physical page placement should be validated after touching pages,
for example with `hwloc_get_area_memlocation()` or `/proc/self/numa_maps`.
