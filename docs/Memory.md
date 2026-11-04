# Heterogeneous CPU memory

The Linux `vecops-memory-bench` tool measures FP32 scalar/Vec bandwidth,
NUMA route aggregates, dependent pointer-load latency, latency under load,
and coherent handoff round trips. It provides readable terminal summaries
and raw CSV/JSON Lines exports. See [the memory tool guide](../tools/memory/README.md)
for build instructions, placement/CPU syntax, timing and byte-count semantics,
and socket-1 examples for 920f-4. The tool performs strict OS binding and
page-residence checks independently of the optional hwloc backend. Its optional
`--interleave` mode cycles each private array over the selected memory nodes
at an explicit base-page or 2 MiB hugetlb-page granularity, with separate
source/destination pools and exact per-page sequence verification.

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

In eager Torch execution, `planner.PlanSession(..., enable_dynamic_factories=True)` uses the remaining
fast-tier budget (machine capacity minus the statically reserved arena) for
`planner.empty` and `planner.empty_like`. Dynamic factories and static factory
replay may be enabled together. Static sites take priority; unselected sites,
retained objects and escaping outputs continue to use storage-owned leases.
Suballocations are leased to Torch Storage through an owning buffer: aliases,
cached tensors, and asynchronous collectives keep their slots alive. Allocation
misses spill to the normal allocator; live objects are never evicted or reused.
The pool creates one byte memoryview over its arena and leases slices directly
to Torch, without constructing a NumPy array for each request. Returned ranges
are inserted in address order and coalesced with their immediate neighbours.
Escaping tensors retain their arena after session close. `dynamic_reserved_bytes`,
`dynamic_peak_bytes`, `dynamic_allocations`, and `dynamic_misses` distinguish the
physical reservation, peak live slots, successful requests, and capacity misses.
`planner.contiguous(tensor)` also uses this pool for layout materialization;
already-contiguous tensors preserve their storage and aliasing.
Pooled tensors have fixed, non-resizable storage; allocate a new tensor when
changing capacity.
This mode requires no lifetime prediction from a FakeTensor capture. General
ATen-internal allocations are unaffected unless their caller uses the factories.

For scoped eager replay, set `PlannerConfig(temporary_scope="diffusion",
only_explicit_factory_temporaries=True, require_replayable_temporaries=True)`.
The scope matches one named component of the captured allocation scope. Only
explicit planner factories in that scope may become static temporaries; this
does not redirect arbitrary ATen-internal allocations. Any retained or escaping
storage alias excludes the entire storage from temporary replay. Call ordinals
include excluded allocations, so a persistent call cannot take a neighbouring
temporary's slot.

Sampled loops require special liveness handling: a value allocated before a
steady sample and read by that sample remains live through the whole loop.
Loop traffic weights distribute only unobserved iterations across steady samples;
fully observed indexed layers each keep their own read counts and state identity.
Their temporary lifetimes end at the recorded last use; only sampled loops need
invariant widening. Runtime scopes use exact recorded iteration identities and
reject a changed concrete trip count or an extra iteration of a complete trace.
Symbolic trip counts retain conservative sampled-loop behavior.
Its last observed read in a two-iteration trace does not imply that later real
iterations stop using it. The planner widens these invariant lifetimes to the
loop exit, and validation rejects cached plans that release them earlier.

Decorated `graph.wrap_region` callables mirror their captured lexical scopes
automatically in a factory-replay session, so a nested region cannot accidentally
consume a parent scope's allocation slot. Dynamic-only sessions keep their
normal scope behavior.

When entering a replay session, eligible Tensor views, shapes and strides are
materialized once. Runtime factories return these cached views after checking
metadata; loop iteration cursors are released at each iteration boundary.
`strict_factory_replay=True` rejects incompatible metadata or extra invocations
at selected sites. Expected dynamic allocations at unselected sites do not count
as replay failures. A capture must faithfully record cache retention and async
lifetimes; asynchronous buffers must not be admitted as synchronous temporary
slots. Validate complete outputs and elapsed inference without profiling before
turning this optimization on for a new workload.

Keyword-free `empty_like` requests from ordinary CPU strided Torch tensors cache parsed
metadata within the session (up to 256 signatures). The signature includes
shape, strides, dtype and device; it stores no Tensor objects or storage
references. Keyword overrides, Tensor subclasses, other layouts and non-CPU tensors
retain the general validation path. Every request still selects its current scope/source and
loop epoch, advances its allocation ordinal and checks the selected slot's
strides. This cache does not authorize retained or escaping slots for static
replay. `stats()` reports `factory_metadata_cache_hits` and
`factory_metadata_cache_misses`; each new session starts with an empty cache.

Topology is immutable. Target counters are atomic and budget admission is
race-safe within one `MemorySystem` state. `os_free_bytes` is an instantaneous
kernel observation, while `managed_bytes` counts only live allocations made by
that state. Physical page placement should be validated after touching pages,
for example with `hwloc_get_area_memlocation()` or `/proc/self/numa_maps`.
`PlanSession.stats()["fast_target_verified"]` compares the allocation owner's
reported target with the topology ranking; it is not an exhaustive physical-page
query. On Linux, a read-only `get_mempolicy(NODE|ADDR)` query of an unwritten
anonymous tail can report the shared zero page's node. Inspect the VMA binding
and query pages after the workload has actually written them before diagnosing
a remote allocation.

## Native graph replay arena ownership

The executable extension of the existing `GraphCapture` uses the same capture
IR as the offline planner. `capture_torch` executes real kernels;
`compile_torch(memory_plan=...)` can materialize an exact plan's internal,
non-escaping empty-factory slots in a graph-private arena. Functional ATen
outputs still allocate through ATen; the executable reports unapplied slow
placements rather than claiming that all allocation is eliminated.

A graph captured inside an existing `PlanSession` cannot borrow that session's
mutable slots for independent execution. By default compilation copies its observed bound
view components into separately owned compact allocations with the same memory placement
and domain request. Original session reuse/close and outstanding old aliases
do not free or overwrite the graph arena. Relative view geometry within each
connected accessed component is preserved; gaps between components are removed;
absolute storage-offset introspection on isolated aliases is unsupported.
`replay.allocator_policy` records copy bytes, capacity, requested domain and
allocation target metadata. Validate physical placement after pages are touched
as for ordinary planner arenas. Captured arena weights are private snapshots by
default. The explicit immutable-state lease below can pin dedicated state slots;
non-arena bound weights retain live in-place content semantics.

Returned tensors may alias static graph slots or inputs. Such outputs are
borrowed and can be overwritten on the next call; clone them when a previous
state must survive. Distinct graph compilations own independent arenas.
Capturing an input that overlaps a borrowed planner arena requires an explicit
staging integration and currently raises instead of silently breaking aliases.
Capture and compile cost, copying, live storage and process/HBM peak allocation
must be measured separately from steady replay performance.

### Executable graph state leases

Executable GraphCapture defaults to private copies of captured planner storage.
Compaction removes gaps between disconnected accessed byte components while
preserving strides, overlapping aliases and original storage identity groups.
It does not compact holes inside a strided Tensor's accessed span. Ordinary
functional ATen allocations remain native ATen allocations.

The optional `borrow_readonly_state=True` compile policy pins only dedicated,
actually copied immutable parameter/buffer slots validated against the source
plan. Those slots cannot overlap another placement at any epoch. Their contents
must remain immutable for the graph lifetime; source session close may restore
module `.data`, but the fixed bound view and owner stay alive. All other arena
bindings remain isolated, and writes through aliases of borrowed state fail.
This avoids duplicating a model's HBM weight replicas while keeping temporary
buffers private. There is no implicit DDR fallback or weakening of free-memory
headroom. Inspect `captured_arena_borrowed_state_bytes`,
`captured_arena_copy_bytes` and `captured_arena_regions` for the actual policy.

### Persistent plan activation for prepared executables

`PlanSession.activation()` activates the same arena and module-state bindings
for sequential synchronous calls. Its first activation allocates the arena;
later activations reuse it. Unlike `with PlanSession(...)`, leaving an
activation does not restore module data or release storage. It only disables
context-local factory routing. This lets graph preparation and subsequent
replays share one HBM reservation rather than retaining the preparation arena
and allocating a second one for each inference.

Activations cannot overlap or nest. The owner must release graph executables,
finish outstanding communication, and then call `session.close()` to restore
module data and release session-owned resources. Storage-owned escaping Tensor
leases remain valid after close. Changing model state or the memory plan
requires a new session and new prepared executables.

Use `with vecops.planner.executable_factories():` around executable capture when
its temporaries should be planned by the executable. Within that scope,
`planner.empty` and `planner.empty_like` emit actual Torch factory operations
for the capture tape, rather than returning views of the active eager plan.
This avoids freezing a large sequence's intermediate buffers into external
arena snapshots. The scope has no effect on metadata-only precompile captures
or on ordinary eager allocation outside executable capture. Capture-time
buffers are temporary; compiled replay owns the resulting storage plan.

`GraphCapture.compile_torch(scratch_allocator=allocate)` requests its
nonescaping static scratch from `allocate(required_bytes)` during preparation.
The result must be a contiguous one-dimensional CPU uint8 Tensor at storage
offset zero with sufficient capacity. The executable retains the Tensor and
never calls the allocator during replay. `static_scratch_required_bytes`
reports the liveness-planned size separately from the supplied arena capacity.
Sharing a buffer across executables is valid only when their entire calls are
serialized and no asynchronous use escapes a call. Outputs and persistent
state retain separate ownership; the shared scratch never contains them.

The explicit `compile_torch(native_pointwise=True, borrow_outputs=True)` policy
also preallocates supported escaping factories and floating pointwise outputs
in a separate executable-owned arena. Its capacity and tensor count appear as
`retained_output_arena_bytes` and `retained_output_tensor_count`. The shared
scratch allocator never owns these outputs. Returned views preserve aliases,
but their values may change on the next replay; clone values that must survive.
Passing an output back as input must satisfy the original metadata/alias guards
and can be rejected. Unsupported functional ATen operators still allocate;
this flag does not claim to make arbitrary graphs allocation-free. The default
functional output semantics are unchanged.

An initializer feeding a steady graph must prepare that steady graph using the
initializer's actual returned buffers, including their capacities and aliases.
Ordinary eager prototypes can have different storage topology and are not a
valid substitute for those persistent output slots.

### Direct SDMA measurements

The memory benchmark has an optional `--engine sdma` copy backend using the
installed sdma-dk SDK, with explicit device selection independent of memory NUMA
and controller CPU placement. Device topology is discovered rather than inferred
from NUMA numbering; unknown physical engine count/locality stays unknown.
Bounded async request queues count completed bytes and include final draining in
the shared elapsed envelope. Payload counts bytes once; application read+write
counts twice. Generation/sentinel verification, page-residence checks, pin/setup
exclusion and per-worker CPU time are documented in `tools/memory/README.md`.

## Unified prepared buffer memory

`execution/BufferPlan.h` is the common pointer-free buffer IR and interval
colorer. `WorkspacePlan.h` now only provides source-compatible names. Python
`memory.BufferRequirement` and `memory.plan_buffers` use that same native
implementation; graph replay and model tensor planning no longer contain
independent copies of the interval allocator.

`memory.BufferSession` owns one fast/slow capacity authority. It installs a
source-kernel provider and also serves model state, explicit tensor factories,
graph intermediates and retained graph outputs. Capacities are **total managed
capacities**, not an allowance independently granted to each category. A plan
sees reservations of other live owners, including alignment and worker replica
padding. Buffers in one plan can independently select fast or slow memory.

A named `reuse_group` explicitly asserts serialized, non-overlapping execution
between those plans. Native kernel workspace uses the reserved `native` group;
prepared graph intermediates use a distinct group because they coexist with
kernel workspace. Escaping outputs use unique, storage-owned leases. Releasing
a larger graph lease shrinks its group's reservation to the remaining users.
Tensor aliases retain physical ownership after session close. Cached native
workspace metadata does not pin obsolete physical pools across model sessions.
Factory adapters explicitly report mutable backing arenas that capture must
isolate. Independent storage-owned leases have no such arena: captured Tensor
references retain their leases directly. When a captured planner arena does
need a private snapshot, its allocation uses the same active buffer budget.

Binding is prepare-time only. A new optional `place_buffers` callback on the
size-versioned arena provider ABI carries individual buffer sizes, traffic,
worker replication and half-open lifetimes into the same planner. Old provider
prefixes remain accepted. Prepared execution uses bound pointers, not Python
allocation callbacks. Moving a live allocation after binding is unsupported;
reprepare when changing capacity, placement policy or executable concurrency.

```python
from vecops.memory import BufferRequirement, BufferSession
with BufferSession(fast_capacity=3 << 30, slow_capacity=16 << 30) as memory:
  buffers = memory.bind([
      BufferRequirement('packed-a', 8 << 20, traffic_bytes=128 << 20),
      BufferRequirement('packed-b', 32 << 20, traffic_bytes=64 << 20),
  ])
  # Each buffers[name] owns a lease and exposes a writable buffer protocol.
  # Prepare/capture kernels using these owners, then replay them synchronously.
  print(memory.stats())
  print(memory.manifest())
```

This version performs conservative static placement as owners are prepared.
Within a batch it prioritizes required-fast buffers and traffic per allocated
byte, colors proved-disjoint lifetimes, and spills individual buffers to DDR.
Already bound owners are not evicted to admit later requests. The existing
model-state frontend still supplies its loop-weighted benefit/copy-cost
selection and may retain a compact arena as one layout-constrained lease.
Reuse across native and graph plans is conservative; their internal timelines
are not merged into one whole-model liveness schedule. Traffic hints count
total reads/writes across all replicas in the declared execution interval.
This is **not** a whole-model optimal solver, an automatic
prefetch/eviction scheduler, or a cross-process physical-capacity broker.
External ATen allocations and communication-owned mappings are not implicitly
captured by this managed allocator. Reserve platform headroom for them.
The legacy standalone `System.workspace_session` API remains available for
callers explicitly requesting an isolated workspace provider; AF3 no longer
uses it.

The common buffer placement preserves deterministic first-fit offsets: it resolves
conflicting allocations through a plan-local index, sorts their physical ranges,
and scans for the first aligned gap. It does not repeatedly search the full trace
at each candidate address. This is preparation-time work; replay uses bound slots.


Kernel workspace parts and outer tensors use the same `BufferPlan` allocation
and placement representation. Kernel grid effects and audited per-call scratch
contracts now supply narrower lifetimes to that planner. `KernelProgram` can
link serial C++ plans before arena allocation, while unknown retained states
remain pinned for its full lifetime. See `WorkspacePlanning.md` for the
`read`/`overwrite`/`consume`, `allow_inplace`, and opt-in graph donation contracts.
These capabilities do not imply automatic phase-dependent eviction of model
weights or cache restoration; such changes require an explicit global schedule
and prepared restore/copy nodes.

### Prepared readonly replicas in unused capacity

`TorchReplay.prepare_output_transfer` can bind individually selected readonly
initializer outputs into free fast leases without moving other allocations.
The slow canonical version remains alive, and its explicit vec copy runs once
between initializer and steady execution. Unknown/peer ownership and unconsumed
outputs are excluded. Placement uses steady logical access estimates rather
than initializer production volume. See WorkspacePlanning for the API and
immutability contract. This is opt-in; it does not implement transparent
weight eviction/restoration or a globally optimal phase schedule.
