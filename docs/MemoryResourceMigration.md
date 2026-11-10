<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# Generic memory infrastructure implementation

The resource model v0.7 is implemented in `vecops::memory::model`.
`Model.h` defines the resource/query/runtime contracts. `ResourceModel.h` owns
immutable, hardware-independent model queries. `LinuxBackend.h` contains the
OS binding and page-store backend; topology labels never enter generic planning.

The migration checkpoint is maintained in `tools/memory/migration/STATUS.md`.
The C++ runtime, Python graph compiler, native workspaces and AF3 consumers use
these contracts. The previous allocator and session APIs have been removed.
Target performance qualification is tracked separately from API migration.

## Resource queries

`ResourceModel` rejects unknown references, duplicate IDs and invalid numerical
properties during construction. A `Scope` produces a `ResourceView` with separate
execution, allocation and access selections. Services shared with invisible
agents retain their physical capacity. Allocation domains include only permitted
members. All user-defined IDs are opaque strings.

Placement specifies integer page stripes and weights. Resolved placement is a
sequence of runs; a contiguous 64 GiB placement uses one run. Recipe checking
validates complete byte coverage, page geometry, shared-backing permissions and
unique owner capacity. It checks coincident slots with a sorted contract sweep,
without enumerating all pairs. Executable lifetimes remain the upper compiler's
responsibility.

A command resolves its symbolic operand to physical intervals. Read/write rights
and physical overlap are checked across different slot IDs and virtual regions.
An unknown alias relation cannot establish a safe ordinary copy. Offline recipe
identities are for planning; the runtime will replace them with backend-certified
identities when submitting actual work.

The reference cost scheduler uses resource traffic, shared normalized read/write
services, fixed service time, executor slots, explicit occupancy and dependencies.
Max-min payload throughput redistributes bandwidth left by other bottlenecks.
Path caps supplement shared controller limits. Estimates without modeled service
are marked incomplete. Unknown physical bandwidth remains explicitly incomplete.
The optional calibration facade below adds measured path limits without
changing physical sharing. Model tests are not AF3 speedup measurements.

## Linux page store

`LinuxPageStore` allocates a shared page store, reserves aligned virtual space
without physical backing, maps backing offsets and rebinds a quiescent range.
Each live mapping retains its backing and reservation. Releasing a mapping
restores a `PROT_NONE` reservation; the VA remains reserved until its last owner
is gone. Logical page quantum may exceed the OS page size, as required by mixed
large/small-page planning.

Backend NUMA bindings are explicit. A nonempty binding uses strict `MPOL_BIND`;
a binding failure is an error. Export/import verifies host boot ID, device and
inode, so multiple process mappings refer to one certified physical allocation.
An exporter retains the name until peer imports complete. Read-only aliases are
enforced by the kernel. The runtime remains responsible for distributed lifetime
agreement, quiescence authorization, capacity admission and event ownership.

## Focused validation

`ResourceModelTest` covers arbitrary labels, projected shared services, 30/70
placement, 64 GiB metadata, complete mapping coverage, shared capacity, 4096 slots
at one offset, physical aliasing, permissions, invocation generations, range and
DAG failures, budget arithmetic, fluid resource contention and cost quality.
`LinuxBackendTest` exercises actual mappings, rebind, retained owners, backing
imports across processes, logical 2 MiB alignment and kernel read-only mappings.

## Capacity admission and templates

`CapacityLedger` admits all required capacity groups atomically and also enforces
shared constraints across groups. A reservation deducts unbacked quota; committing
backing converts quota to a uniquely owned physical charge. Aliased mappings
retain that charge. Eager cache state transitions move bytes between live, idle,
reserved and retired without charging the same pages twice. Failed admission and
unused grants leave no partial reservations. Refreshing an external free-memory
sample is coordinated between physical materialization transactions.

`HardwareTemplate` expands the compact `memory-model/0.3` TOML format entirely in
C++. Discovery fills generated IDs with capacities, page geometry, access and
backend metadata; `ResourceModel.view` applies process grants afterwards. Tests
use the LX2, A800 and H800 examples and arbitrary replacement names, plus directed
sparse links. Recipe/layout TOML uses only exact integers, with standard TOML's
signed 64-bit representable range checked explicitly.

## Physical topology and process grants

`LinuxDiscovery` keeps physical topology separate from execution/allocation
permission. `sched_getaffinity` selects execution agents; `Mems_allowed_list`
selects allocatable pools. Host pools outside the allocation grant can still be
represented for imports. An inactive CPU domain does not automatically hide its
memory from another agent allowed to allocate there.

The explicit LX2 Flat backend profile binds CPU groups to DDR NUMA0–15 and HBM
NUMA16–31, verifies the complete layout and rejects missing/cache-mode HBM.
Physical resource sharing comes from the template; the discovery adapter only
binds roles to OS entities. It preserves the raw CPU model as diagnostic evidence.
SDMA agents enter the executable scope only when device access is actually
permitted; discovery does not invent queue/channel capacity.

## Concrete host runtime

`LinuxRuntime` implements the byte-level runtime contract. Admission covers
standalone allocations, eager arenas and unbacked plan leases through one ledger.
Address preparation reserves inaccessible VA without committing physical pages.
Acquisition materializes backing; binding applies offset-only recipes to that VA.
Quiescence proofs authenticate the binding generation and last-use epoch before
unbind/rebind. FD operand descriptors use certified inode identities and file
offsets, including imported external aliases. Anonymous movable operands use
the same process-private address identity as external Tensor ranges, independently
of their stable capacity-accounting root identity.

`private_movable` is a separate whole-root, single-active-mapping contract.
Acquisition binds and touches anonymous pages before committing the root;
transparent huge pages are a kernel capability, separate from the logical page
quantum. The backing owns a parking reservation, keeps its capacity charge while
parked, and moves only at bind/unbind or proven quiescent rebind. The Linux
implementation retains the old VMA during a successful move, then makes that
empty VMA inaccessible. A failed move may remove the target placeholder;
restoration must not replace another mapping. A recoverable failure restores
the prior content and mapping; unrecoverable rollback terminates, as in the FD
replacement path. This mode excludes export, partial maps, concurrent VA aliases,
arena slices and inherited fork mappings. Ordinary Tensor views within its one
active mapping retain the same owner.

The graph compiler queries mode-specific public mapping capabilities. Only an
unspecified, single-pool slab without sharing or independent page mappings can
choose this mode automatically. Explicit backing choices, striped placement
and shared communication buffers keep their requested mapping contracts. The
hardware template and parser contain no new CPU-specific fields.

Eager arenas map each growing chunk once and suballocate page-aligned offsets.
Exported slices retain their chunk and return ticket; a block returns to the free
list only when its last local alias stops owning it. Explicitly offered caches
can supply plan backing without another physical charge. Estimation simulates
pool, page quantum, backing mode and contiguous-block requirements. Snapshot
availability includes only cache that the requesting path can actually reuse.

`HostEventService` authenticates invocation and generation, joins dependencies,
retains active owners and releases caller tokens explicitly. Runtime bookkeeping
holds stable event references, so releasing a finished caller token does not
invalidate quiescence. Callbacks run outside allocation/binding locks. Logical
failure or cancellation is distinct from stopped memory access.

CPU queues execute copy/fill/fence synchronously; optional device-copy queues
use the SDMA executor described below. Queue and method bindings are explicit;
route and endpoint checks precede acceptance.
Single-participant shared ranges implement begin/publish/acquire/retire epochs;
multiple participants require a control adapter with agreement and all-rank
retirement. The communication adapter owns payload roots through this same runtime.
The Python-to-Torch-extension handoff uses the versioned
`MemoryRuntimeBridge.v1` ownership capsule, so independently built pybind
registries do not need to recognize each other's runtime class. The descriptor
checks the runtime interface revision and C++ standard-library ABI; exporter
retain/release callbacks keep the runtime alive until the backend's final owner
is released. This changes no memory-core archive or physical capacity semantics.
Optional device copy execution is described below.

Only this memory infrastructure is required to be C++. The graph compiler stays
in Python by default for maintainability. It consumes public queries, recipes,
commands and events, without depending on OS NUMA IDs or allocator internals.

## Shared physical admission and explicit kernel buffers

`LinuxRuntimeOptions.capacity_directory` joins a job-scoped, same-host physical
capacity authority. Every participant uses identical physical groups and domain
limits; process scopes restrict candidate pools without cloning those limits.
Local ownership counters and globally remaining capacity are explicit in budget
snapshots. Root reserve/commit/release is transactional across processes; cached
arena allocate/free remains local. Imported aliases add no physical charge.

Recordable source kernels support metadata-only preparation and explicit buffer
activation. Names, shapes, replication and lifetimes are descriptions; placement
and offset packing belong to the Python graph compiler. Activation receives the
compiler's addresses and a physical lifetime owner. Suspension first destroys
prepared state, then permits backing unmap/reassignment after quiescence.


## Optional Linux SDMA execution

`LinuxSdma.h` uses the installed official `mdk_sdma.h` declarations and dynamically
loads `libsdma_dk.so`; no SDK implementation is vendored. Build selection is
`VECOPS_MEMORY_RUNTIME_SDMA=AUTO|ON|OFF`. `ON` requires the SDK header. A build
without it reports `sdma_unavailable`; it never executes a CPU substitute.

The application explicitly binds a queue symbol, graph device agent, host
controller and opaque method IDs through `LinuxRuntimeOptions.transfer_queues`:

```cpp
LinuxSdmaOptions sdma;
sdma.device = certified_device_path; // discovery backend descriptor
sdma.page_bytes = linux_base_page_bytes;
auto executor = make_linux_sdma_executor(sdma).value();
LinuxTransferQueue queue{agent_id, controller_id, {method_id}, executor};
options.transfer_queues.emplace("copy-queue", std::move(queue));
```

Only copy is implemented by this adapter. CPU fill/fence use explicit CPU
queues. The runtime checks process grants, controller host access, certified
operand pool spans and the exact directed agent/controller route before work
acceptance. Unknown external placement cannot authorize a device route. The
planner must explicitly choose the device method; listing a method in the
hardware graph alone does not install an implementation.

Each operation owns one SDK channel handle (shared channel 0 by default;
`shared_channel = nullopt` requests exclusive allocation), pins complete Linux base-page
envelopes, splits registration ranges below the SDK uint32 size limit, and copies
in bounded chunks. One descriptor is in flight per operation. Shared requests use their own
SDK receipt for completion; exclusive channels query and retire their descriptor
before unpin. Separate invocations have separate SDK handles and receipts,
while a shared physical channel can serve multiple clients. `progress()` therefore reports caller-driven execution, one in-flight
descriptor, no exposed chunk events and concurrent invocations. The current
implementation includes registration/channel setup in every operation; it does
not claim cached-registration or multi-channel peak performance.

`submit()` drives the command DAG until completion or
`transfer_wait_ms` (default 30000). A submit/query error may follow partial
hardware enqueue. Such a submission returns a failed event with
`accesses_stopped=false`, prevents binding reuse and retains all operand owners,
registrations, channel and SDK library. Remaining DAG commands are not accepted.
There is no CPU fallback or process termination inside the library.

`poll_linux_transfer_quarantine()` retries these uncertain operations and returns
address-free diagnostics: token, status/error, payload bytes, retained pin bytes
and registration count. Only an SDK-certified stopped result releases retained
resources. The registry also retains its event service outside callbacks, so
runtime destruction does not block indefinitely or reenter its own event poll.
The registry intentionally has no static destructor: permanent device failure
without a public abort/reset guarantee retains resources. Process exit releases
private kernel resources; named shared backings and their durable receipts may
require explicit cleanup after all exporting and importing participants exit.
Call this API again after recovery; query of a retained runtime token remains
valid as well. A failed event stays failed even if its device accesses later stop.

`LinuxSdmaSmoke` is an explicit compute-node correctness and measurement tool.
It selects a route by opaque ID, verifies actual source/target NUMA placement,
checks every copied byte, alternates CPU and SDMA samples, and reports effective
GiB/s including setup, submission and completion. Use `--list` to obtain route
IDs from the bound graph; do not run this workload on a cluster login node.
The optional final argument `private_movable` selects whole-root anonymous
backing for both endpoints; omitting it uses `shareable_remappable`. This lets
target qualification exercise actual SDMA registration and copying for both
mapping contracts without changing application placement policy.


For `SharingMode::read_write`, one physical Resource has one normalized service
budget. At rates `B_read` and `B_write` in GiB/s, an activity consumes
`read_bytes / (B_read * 2^30) + write_bytes / (B_write * 2^30)` seconds from that
same budget. Payload work and fixed `per_invocation_bytes` both participate.
Concurrent reads and writes compete for it; they cannot independently saturate
the two rates. Hardware with independent duplex service must declare two
separate Resource objects. Unknown required directional rates remain incomplete
estimates rather than becoming unlimited service.

Requester-executed CPU copies also consume the source access connection's
`read_fabric` and destination connection's `write_fabric`. `estimate` and
`cost_plan` materialize those traversals using each segment's fraction of the
copy payload. The route already supplies memory-controller work; adding fabric
does not add a second controller read/write. If reading and writing cross the
same fabric Resource, both byte demands count. Opposite directed Resources
remain independent, while concurrent copies sharing a Resource compete.

Explicit `Route::demands` (including a template transfer's `use`) describe
additional real work and add to these requester traversals, even when they
name the same Resource. IDs are not deduplication keys for traffic. A route's
fixed `per_invocation_bytes` remains once per used route, including fragmented
copies. `CostPlan` is the concrete demand representation: `estimate_schedule`
consumes it directly and does not reconstruct or add access traversals. The
calibration facade preserves those demands while applying private curve caps
and explicit copy write coefficients.

## Optional calibration overlays

`open_linux_runtime(hardware, policy, calibration, options)` accepts the supported
subset of `memory-calibration/0.4`. `Calibration.h` owns the parser and immutable
`CalibratedCostModel` facade; `ResourceModel` remains a standalone reference
model. Hardware and calibration selectors use the same temporary expansion and
runtime counts. Only concrete IDs survive in `PerformanceDatabase`. Scope
projection happens afterwards, so partial process grants do not renumber cases.

Supported profiles are access `read`/`write` curves, transfer size curves or
explicit `operation="copy"` cases, and resource `write_cost.copy`. Unknown
fields, unbound profiles, invalid selectors, conflicting cases, compute-only
`methods`, `extra_write`, and triad models fail explicitly. Historical files in
`docs/memory-resource-model/calibration/` preserve the broader design data and
are not all executable overlays. Extract only supported, qualified fields into
an application overlay; loading never silently discards the other fields.

For example, an overlay can bind an arbitrary access-rule name:

```toml
schema = "memory-calibration/0.4"
name = "workload access reference"
[units]
bandwidth = "GiB/s"
[models]
"access:ddr_local" = "local_stream"
[profiles.local_stream]
provenance = { quality = "reference", source = "920f-lx historical STREAM campaign" }
read = { cpu_cores = [16, 37], bandwidth = [149.29, 148.75] }
write = { cpu_cores = [16, 37], bandwidth = [141.13, 137.48] }
```

These are retained campaign values, not measurements of the new runtime
executor. A core-count curve is selected only with an explicitly supplied
`LinuxRuntimeOptions::calibration_controls[agent_id].cpu_cores` matching a sample.
Neither process affinity nor `CostContext::execution_slots` supplies this fact.
An optional provenance `implementation` requires the same opaque implementation
in that control. `quality="calibrated"` additionally requires nonempty `source`
and `implementation`; changing the executor requires requalification. SDMA case
selectors name the CPU controller, while the actual transfer executor remains
the DMA agent. The case resolver follows the route's controller field.

Size curves interpolate **service time**, with exact integer byte axes. They
never extrapolate, and core-count curves currently require exact sample counts.
An unmatched range or implementation uses the hardware reference estimate.
`CalibratedCostModel::explain_access` reports selected profile, provenance,
source, effective GiB/s and whether interpolation occurred. Python
`runtime.performance()` returns a detached database dictionary for auditing.

The facade adds private per-activity caps while preserving physical resource
sharing, directional read/write work, active activities and dependencies. Copy
write coefficients affect payload and fixed write traffic, exactly once through
`runtime.cost_plan`; an ordinary write access does not inherit a copy coefficient.
Unknown physical services remain incomplete. Aggregate estimates remain
`reference`, because physical service/startup/latency references still contribute;
the provenance of one selected curve does not promote an entire path to measured
accuracy. The public physical graph is unchanged by loading or estimating.


## Framework capacity domains

Ordinary framework allocations may have a permitted fallback domain wider than
one preferred pool. The application adapter translates its actual allocator
policy to an ordered list of opaque pool IDs. For Linux AF3 this uses the task
memory policy, cpuset memory mask, Runtime grants and public Linux bindings;
strict binding stays restricted, while preferred placement allows its real
fallback set. An explicit framework-pool override remains a single-pool policy.
The generic graph compiler never interprets Linux NUMA IDs or hardware names.

`reserve_budget_latest(runtime, view, bytes, pool_order)` reserves only pending
capacity and returns the owning lease plus the exact distributed recipe. Every
attempt rebuilds that recipe from a fresh snapshot. Page quanta, physical free
space, hard limits, capacity-group aliases and overlapping cuts all participate;
managed idle pages do not become ordinary malloc headroom. Only explicitly
rejected, unaccepted reservations can retry. The ordered policy can reject a
packing that another order could satisfy; it is not a global solver.

This declaration bounds aggregate framework demand without directing malloc's
physical page placement. Managed graph buffers remain governed by their own
placement and acquisition plans. AF3 includes the eligible-domain signature in
its replay contract and rejects a changed Linux policy. Metadata capture and
framework lifetime rules are unchanged by this capacity-domain correction.

AF3 admits graph and native workspace placements before this flexible framework
budget. The latter then uses the remaining eligible capacity, avoiding a greedy
framework reservation that consumes a constrained graph buffer's only pool.
Both admissions precede graph backing acquisition. The model owner retains both
reservations and closes them if either admission or later preparation fails.
This ordering is application policy over the public APIs; the memory foundation
does not distinguish framework tensors from graph buffers.

Planning caches belong to one compilation attempt: budget scalars, endpoint
identities and exact access queries are rebuilt for a new snapshot. Reusing
these immutable facts removes repeated binding conversions without changing
cost ordering, packing, placement, alias lifetimes or offsets. `PlanEstimate`
also reports each rejected capacity group, pool or overlapping constraint with
its requested and available bytes; a false feasibility result is auditable
without inferring its cause from requested backing sizes alone.

Distributed graph admission retains successful pending grants while unsuccessful
participants replan against the remaining capacity. Before collective resource
finalization, every grant is withdrawn at a barrier while its offset plan and
reserved VA survive. Resource growth invalidates all proposals. This preserves
joint spill placements without obstructing transport provisioning; fatal peer
failures close every local proposal and return all accepted pending capacity.

Physical free samples can decline during a large cold compilation even with an
unchanged ledger. Typed group shortfalls then provide an observed fresh-byte
margin for the next attempt. Learning requires unchanged accounting and full,
untruncated pool allowances; independent quota changes invalidate it. This
prepare-local policy also learns each untruncated group's observed physical
decline between the same-accounting snapshots, taking a maximum per group.
An unaccepted shared physical-free reservation rejection also carries primitive
facts from its last actual reserve snapshot. A local grant rollback can advance
the epoch while restoring every balance. Only this known rejection permits an
epoch rebase, and only when compilation, reserve and current snapshots agree on
all nonphysical group, pool and overlapping-constraint accounting. Peer quota
changes remain a reason to discard the observation. The error retains no native
snapshot owner or original exception context.
It preserves idle cache and never changes native admission
limits or public budgets. It prevents repeatedly filling the latest sample to
its exact boundary while retaining the existing bounded retry protocol.
