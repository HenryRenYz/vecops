<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# Memory resources

The `vecops::memory::model` C++ API describes agents, memory pools, allocation
and capacity domains, access connections, transfer methods and shared service
resources. The planner has no built-in Socket, Die, NUMA or memory-tier enum.
Hardware names and topology belong to TOML templates and backend discovery.
The complete architecture and contracts are in
[the memory resource design](memory-resource-model/index.html).

`vecops::memory` is the CMake component. Public contracts live in
`vecops/memory/Model.h`; `ResourceModel.h` provides topology/placement/cost
queries. `LinuxRuntime.h` opens the Linux CPU allocation and mapping runtime.
Python exposes these same value objects through `vecops.memory.native`.
The Linux runtime provides CPU allocation/mapping and optional SDMA copies.
A800/H800 templates describe topology and costs; they do not install a GPU
allocation or execution backend.

## Resource views and budgets

`physical_graph()` retains the complete discovered machine. `current_scope()`
and `view()` identify executable agents, pools the process may allocate, pools
it can access, and execution-slot grants. Allocation permission is independent
of CPU execution permission. A shared DDR controller is one service resource;
OS NUMA partitions can remain distinct capacity pools within one allocation
domain. An eager arena can spill among explicitly permitted domain members.

`snapshot()` reports system availability, fixed limits, live/cache/retired roots,
and pending grants. Capacity groups and cross-group constraints prevent counting
aliases or shared physical capacity more than once. Same-host participants pass
one private job-local `capacity_directory` and consistent physical limits; the
C++ authority admits their physical roots atomically. `capacity_authority()`
returns its opaque launch identity, or no identity for independent accounting.
Same-host sharing participants must agree on this identity before exchanging
roots; it is excluded from persistent cache keys. Reusing an arena's cached
block does not reserve its capacity a second time.

Automatic root growth samples Linux free capacity and records its pending grant
inside one shared-authority transaction. Concurrent participants cannot invalidate
that admission by committing another root between sampling and reservation.
Pending grants continue to cover materializations until their physical roots are
committed. Explicit graph reservations retain the supplied budget epoch and
reject stale plans; automatic growth does not refresh an explicit plan's epoch.

`refresh_available()` takes a new Linux free-memory sample between physical
materialization transactions and advances the local budget epoch. Call it before
a new planning session when other allocations may have changed system capacity.
The independent-process ledger otherwise retains its last sample, minus its own
new commitments and pending grants. A shared authority also checks a fresh Linux
sample when admitting each new root. Neither a query nor an unbacked VA reservation
holds physical memory; use a capacity lease or arena preallocation for that.

Linux free-capacity samples exclude zone high watermarks, protected low-memory
reserves and the reported per-CPU counter uncertainty. They use the lower of
this zone headroom and node `MemFree`, without estimating reclaimable pages.
These are live capacity samples; physical commitments still require admission.
Rejected physical-free admissions report the group, requested bytes, sampled
free bytes and pending bytes from the same locked transaction.

`observe_external(view, range)` samples actual resident page placement without
touching or migrating pages and without a capacity charge. Its offsets are
relative to the supplied range. It ignores a caller's `known_pool` hint:
unfaulted pages, unavailable queries and ambiguous backend bindings leave gaps
and `complete=false`. Use this cold-stage observation when costing external
buffers; it is not a lasting placement or ownership guarantee.

Hardware TOML describes the machine; application-policy TOML describes limits,
initial reservation, growth and idle retention. Participant identity, process
CPU grants and the job-local authority directory are launch parameters.

## Allocation and mapping

Direct allocations and `MemoryArena` use the same native ledger as graph plans.
An allocation returns a page-aligned pointer, placement and export/remap
capabilities. Derived slices, mappings and in-flight operations retain their
physical root independently. `trim()` returns wholly idle arena growth chunks.
Logical page alignment and actual physical page sizes are reported separately.
`BackingMode` selects the mapping contract. `private_memory` uses private FD
backing with local aliases; `shareable_remappable` also supports same-host
export/import. `private_movable` owns anonymous pages with one active mapping
per whole root. It supports ordinary Tensor views at that mapping, quiescent
whole-root moves and transparent huge pages when the kernel provides them.
It excludes export, partial mappings, simultaneous VA aliases and fork
inheritance. Acquiring any mode materializes its pages before returning;
requesting a larger logical quantum does not promise physical huge pages.
Transparent coalescing does not increase the pool's declared mapping quantum;
when explicit small and large mapping formats coexist, that quantum remains
the largest declared format.

A planned allocation separates address reservation from physical commitment:

1. Validate a `MemoryRecipe` containing integer offsets, roots and mappings.
2. `prepare_addresses()` reserves inaccessible virtual ranges without backing.
3. `reserve()` admits the physical demand at an explicit budget epoch.
4. `acquire_backing()` commits the admitted roots, then `bind()` maps them.
5. Fence all users before `unbind()` or quiescent remapping.

Imported shared ranges are checked by backend-certified physical identity and
file offsets. Their virtual addresses alone are insufficient to establish
aliasing. Anonymous operands instead use the current process-private VA byte
identity, so an external Tensor at the same VA is recognized as overlapping.
The stable root identity used for capacity accounting remains separate.
Linux shared publication names use a PID and kernel UUID, so independent local
SDK consumers in one process do not compete for a DSO-local counter namespace.
Event failure or cancellation authorizes reuse only after accesses
have stopped. Serialized plans contain offsets and symbols, never live pointers.

An aborted communication operation can still complete the collective shutdown
handshake once all local work and page-exchange leases have stopped. Each peer
removes imported aliases before publishing its stopped acknowledgement; only
all acknowledgements authorize root refunds. Missing peers or active leases
keep their roots charged. Abort alone neither proves shutdown nor forbids it.
If participants exit without that proof, named shared backings can outlive their
processes. Recovery requires the job owner to establish that all participants
have exited and match each file's host boot, inode and creator receipt before
unlinking it. Dead-owner metadata alone never authorizes automatic refund.

`mapping_capabilities(view, pools, mode)` reports the selected contract.
`partial_backing_map` and `multiple_mappings` distinguish a single whole-root
move from arbitrary FD mappings. Unsupported movable-page kernels return an
unavailable capability, allowing planning to choose another mode. Replacement
errors preserve the previous usable mapping; an unrecoverable rollback stops
the process, including when restoring the reservation would overwrite an
unrelated mapping. No ordinary error returns a damaged address reservation.

## Python ownership

The Linux host adapters return CPU tensors, including empty views, regardless
of PyTorch's default device. Prototype construction and state relocation use
the same explicit CPU storage contract.

```python
from vecops.memory import Runtime, Pool, use_runtime

runtime = Runtime.open("hardware.toml", policy_file="application.toml")
with use_runtime(runtime):
    view = runtime.view()
    budgets = runtime.snapshot(view)
    # An opaque allocation-domain ID or a singleton pool ID.
    pool = Pool.create(runtime, "working-domain", hard_limit=0.75)
    tensor = pool.empty((1024, 1024))
    # Tensor Storage and its aliases retain the underlying C++ allocation.
    del tensor
    pool.trim()
```

`use_runtime` selects the same C++ allocation authority for ordinary native
workspaces, graph programs and eager pools. It is a synchronous, thread-affine
scope; nested scopes close in reverse order. Fallback native workspace caches
retire when entering or leaving a scope. Explicit graph bindings and live Tensor
owners retain their own allocations. `default_runtime()` returns the same native
default authority used by calls outside an application scope.

`Runtime.contract()` records stable topology, process grants and configured
limits for frontend caches. It excludes live usage, addresses and ledger epochs.
Every physical instantiation still performs fresh admission.

## Graph compiler boundary

`vecops.planner.GraphMemoryCompiler` is Python graph policy over public runtime
queries. Its `BufferRequirement` unifies tensor storage and operator workspace.
`CompiledMemoryProgram` contains a native recipe and buffer offsets;
`PreparedProgram` reserves addresses, materializes the recipe and creates Tensor
views that pin their actual binding, including escaped outputs.

`VirtualRegion.base_alignment_bytes` is an independent minimum absolute VA-base
alignment. Zero uses `page_bytes`; otherwise it must be a power of two and the
effective minimum is at least `page_bytes`. This changes only VA reservation,
not the logical mapping quantum, region byte size or physical backing charge.
The graph compiler propagates its strongest buffer-address alignment per slab,
so aligned offsets also produce aligned actual pointers for small regions.
The Linux page store's `reserve(bytes, page_bytes, base_alignment=0)`
implements this contract; its large-region THP preference is unchanged.
Serialized recipes require this field, including an explicit zero for the
default. An already prepared weaker region
cannot bind a recipe requiring a stronger base guarantee.

`GraphCapture.prepare_torch()` performs graph rewrites and describes native
workspace without allocation. Compile its final `memory_requirements()` before
binding that same prepared graph. `compile_model()` links explicitly named
storage across calls and reuses transient space only under a declared serialized
schedule. Provider state that survives calls is retained until its owner closes.
For unspecified backing, a single-pool slab without export, a sharing contract
or independent page mappings uses `private_movable` when its public capabilities
permit it. Explicit modes remain authoritative; shared, independently remapped
and interleaved slabs retain the shared FD mapping contract. Each movable slab
gets its own whole root; capacity is charged once, including while it is parked.
See [the Python guide](Python.md) for complete preparation and replay examples.

## Device copies

`LinuxRuntimeOptions.transfer_queues` explicitly binds device methods and graph
agents/controllers to an optional `LinuxSdma.h` executor. Copy registrations and
operand owners survive until SDK completion is proven. Failed submissions with
unknown quiescence remain unavailable for reuse and are visible through
`poll_linux_transfer_quarantine()`; no CPU fallback occurs. See
[the implementation guide](MemoryResourceMigration.md#optional-linux-sdma-execution)
for installation, progress and permanent-device-error lifetime rules.

## Measurement

The separate `vecops-memory-bench` tool measures NUMA routes, directional and
shared bandwidth, latency, contention, interleave and SDMA paths. Its
[tool guide](../tools/memory/README.md) defines placement, timing and byte counts.
Measured calibration and hardware specifications retain distinct provenance;
unknown model costs remain unknown.
The graph compiler caches exact access-query costs as primitive values and freezes
resource-view membership within each compile attempt. Every new compilation
refreshes these cold caches; admission still uses fresh ledger snapshots.


## Inspection

`vecops-memory-info` prints the current public agents, pools and capacity groups,
plus backend descriptors for diagnostics. Select allocations by an opaque pool
ID, for example `--pool 'host-memory[0]' --allocate '64 MiB'`. With a hardware
file, add `--hardware FILE`; policy and calibration file arguments have the same
meaning as the C++ runtime opener. Capacity displays use GiB, not raw rates.


## Reserving a compiled graph before mapping pages

`PreparedProgram.admit()` holds a pending capacity lease without acquiring
backing pages. `materialize()` uses that lease; `close()` releases an unused
lease. `GraphMemoryCompiler.prepare_model(..., serialized=...)` combines model
planning, address reservation and admission. If another participant consumes
capacity during planning, it may replan against changed available budgets
when placement, the final physical estimate, or admission rejects capacity,
before returning addresses to tensor bindings. Invalid recipes, permissions,
unchanged capacity failures and actual backing failures are not retried.
Pure `compile_model()` remains available when the caller manages admission.

## Runtime ownership across Python extensions

`MemoryRuntime.export_bridge()` returns a process-local capsule named
`vecops.memory.MemoryRuntimeBridge.v1`. The communication extension imports this
capsule rather than asking its own pybind registry to cast a runtime registered
by another extension. The descriptor in `memory/RuntimeBridge.h` checks its type,
descriptor version, C++ runtime interface revision, pointer width and standard
library ABI before acquiring an owner reference. Exporter callbacks retain and
release the native runtime; deleting the Python runtime or capsule cannot
invalidate an importing backend. No runtime address integer is accepted.

This bridge does not serialize runtime handles and is not an ABI-stable C memory
API. Its interface revision must change when the runtime vtable or argument and
result ABI changes. All participants use matching SDK interfaces and a compatible
C++ standard library; their Python binding registries may differ.
