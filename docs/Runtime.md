# Runtime, declarative kernels, and artifact ABI

The runtime is framework-neutral.  It sees only non-owning tensor metadata,
scalars, a declarative `KernelDef`, and a caller-selected artifact provider;
it never depends on Torch, NumPy, or an owning tensor class.

## Main objects

| Object | Owns | Does not own |
| --- | --- | --- |
| `TensorView` / `ArgumentMetadata` | shape, stride, dtype, and positional metadata | Tensor allocation |
| `KernelDef` | declarative parameter rules and named value vocabulary | Invocation metadata or storage |
| `KernelCall` | supplied metadata and explicitly supplied values | inferred values or a compiled module |
| `BoundKernel` | normalized arguments and complete specialization identity | Tensor allocation |
| `Operator` | schema, recipes, dispatch policy, and provider | Executables returned to a caller |
| `Executable` | a shared `LoadedModule` DSO handle | Tensor allocation or workspace |

`TensorView::sizes` and `strides` use elements.  `byte_offset` and workspace
sizes use bytes.  A `TensorView` must grant `VECOPS_TENSOR_READ` for `Input`,
`VECOPS_TENSOR_WRITE` for `Output`, and both for `InOut`; the flags are access
declarations, not synchronization.  A caller must synchronize overlapping
writable storage.

Explicit DSL resident teams use the existing size-versioned team executor
capability, not barriers on logical task callbacks. A worker failure cancels
the native team's barriers. Team count respects current capacity/budgets;
collective scalar slots belong to one invocation and cannot be reused until
all readers finish. `execution/Atomic.h` exposes checked-alignment scalar
atomic_ref operations for 32/64-bit integers and float/double, with old-value
returns and documented NaN comparison semantics. These are thread-level
primitives, not a cross-process coordination API.

## KernelDef binding

A `TensorDef` coordinate is one of: fixed integer, named `ConstInt`, anonymous
`ConstInt`, anonymous `Dynamic`, or named `Dynamic`. Tensor dtype is fixed, a
named `DType`, or anonymous specialized dtype. Named specialization values
must be declared in `KernelDef::values`, even when they are referenced only by
kernel C++ through `vecops::spec`.

A named Dynamic enforces equality between all occurrences in one call while
retaining its alignment and inclusive bounds. Its observed value is kept in a
separate per-bind environment: it is never accepted through
`KernelCall::values` and never enters the specialization key. Declarations of
the same runtime symbol must have identical constraints.

`bind_kernel_call` works in two phases:

1. It starts with `KernelCall::values`, fills omitted optional/default
   positional arguments, and observes specialization values and named Dynamic
   values from every present tensor. Repeated observations must agree.
2. It verifies all values are now known, evaluates Dynamic alignment/bounds,
   validates metadata and access, and emits deterministic named/anonymous
   specialization identity.

This deliberately does not need `TensorView::data`.  It can therefore select
or compile an executable from `TensorMeta` before an output allocation exists.
Actual `Executable::invoke` checks storage before entering the artifact.

`Dynamic<alignment, lower, upper>` has inclusive bounds. Its expressions may
be fixed integers or named `ConstInt` values; alignment must resolve to a
positive power of two.

## Dispatch and cache policy

`KernelRecipe::match` and `bind` are cheap metadata-only operations. `bind`
accepts a call that has already passed `match`; `Operator` performs that
applicability check before binding instead of repeating it inside `bind`. An
`OrderedDispatchPolicy` tries recipes in declaration order.  Only an
`ExecutableProvider` may perform DSO loading, cache I/O, or compilation.

`ArtifactExecutableProvider` has three persistent-cache policies:

| Mode | Cache read | Build callback | Persistent writes |
| --- | --- | --- | --- |
| `CacheOnly` | Yes | Never | Never |
| `ReadWrite` | Yes | On a miss | Atomic staging-and-rename publication |
| `CompileOnly` | No | Every resolution | Never |

All modes may keep a successfully loaded executable in the provider's in-memory
memoization.  The persistent key includes the bound recipe identity and the
application namespace; applications must use distinct namespaces for
incompatible target, compiler, or SDK populations. Resolution is coordinated
per persistent key: identical concurrent misses share one in-flight result,
while different specializations may load or compile concurrently through the
same provider.

Batch preparation separates this boundary into `lookup()` and `adopt()`.
`lookup()` performs only memory/persistent-cache resolution and never invokes
the build callback. Cache misses are bound and compiled together; `adopt()`
then validates, atomically publishes in ReadWrite mode, and memoizes the
result. Ordinary `resolve()` retains its synchronous behavior for calls made
outside a batch.

## Execution parallelism and the Torch bridge

`VecopsThreadPoolV1` is a synchronous executor for dense logical task IDs. An
embedding supplies it through `VecopsExecutionContext::thread_pool`, together
with an optional `requested_threads`. The effective execution parallelism is
the requested value capped by the pool's reported capacity. Logical tasks are
not physical-thread identities: a pool may run several task IDs sequentially
on one physical worker.

V1 minor 1 appends an optional, size-checked `team_executor` pointer. A
`VecopsTeamExecutorV1` supplies concurrently resident workers with dense stable
ids and a cancellable, reusable happens-before barrier. Logical tasks alone
do **not** provide this capability: putting a barrier inside several tasks
that share one physical worker can deadlock. Old V1 providers stay valid.
The native provider implements teams; current Torch/OpenMP providers keep
their logical-task API and generated grids fall back to individual regions.

Linux AArch64 vector lengths are per-thread and SVE VL need not equal SME SVL.
Logical-task and resident-team dispatch capture the caller's pair. Worker entry
checks it using architectural reads; only a mismatch invokes `prctl` to select
the exact lengths. Worker exit restores its original lengths and inheritance
flags, including exceptions. Unsupported lengths or OS rounding are errors.
`execution/VectorLength.h` exposes `current_thread_vector_lengths()` and
`ScopedArmVectorLengths` for embedding code; the latter must run outside
streaming mode and with no live vector, predicate, or ZA state spanning it.
Generated source-kernel adapters cheaply validate their compile-time VL/SVL
pair on every entry. If the caller changes it, obtain the corresponding kernel
again through `T.jit` or `T.compile`; a stale fixed-geometry kernel never runs.

Direct Python/NumPy calls receive the lazy `vecops_cpu_thread_pool_v1()` facade.
Its capacity follows the calling thread's affinity, physical cores, cgroup v2
CPU quota (cached for at most one second), and `VECOPS_NUM_THREADS`. An explicit
budget may use SMT but cannot exceed the allowed CPU count/quota. A malformed
environment budget conservatively reports one worker. Pools are created only
for selected parallel work and cached by geometric capacity plus CPU set,
so varying task counts under a fixed budget do not create a separate pool for
every worker count. Only the requested prefix participates; default
pinning uses distinct physical cores before SMT siblings. The caller is
worker zero and may be temporarily pinned during a submission, with its
original affinity restored on return, including exceptions. Worker mailboxes
wake only selected participants. Inactive workers do not acknowledge each
submission; team barriers briefly spin then use atomic wait/notify.

`vecops_cpu_thread_limit(n)` returns the previous calling-thread budget;
zero selects automatic capacity. Python exposes a nested, exception-safe
`with vecops.cpu_thread_limit(n):` scope and the read-only
`vecops.cpu_thread_pool_capacity()`. These do not mutate global environment,
Torch's team, the legacy fixed native pool, or the caller's persistent affinity.
Separate processes should use spawn/forkserver with disjoint CPU affinities
or explicit per-process budgets. There is no cross-process work queue,
automatic NUMA placement, or support for inheriting live workers across fork.

The generated Torch extension is a typed, schema-shaped adapter artifact; it
does not bake a dispatcher namespace or operator name into its source. At load
time the DSO registers each namespace/name/schema separately, with a stateful
typed CPU functor holding that operator's runtime handle, specialization names,
allocation observer, and thread-pool context. Meta registration uses a typed
Meta functor. Thus multiple Torch names—including distinct read-write and
cache-only JIT wrappers—can share one bridge artifact without sharing operator
state. Re-registering the same name with a different handle is rejected. The
bridge cache identity includes the typed signature/spec ordering, Torch and
compiler identities, and parallel-runtime policy, but excludes the registration
name and cache mode; cache-only lookup checks its deterministic signature key
and never builds or writes a missing bridge.

The generated Torch extension uses Torch's intra-op runtime by default. It
reports `at::get_num_threads()` and detects an existing
`at::in_parallel_region()`. The default generated adapter uses
`at::parallel_for`. When Torch identifies its packaged GNU OpenMP runtime and
the selected bridge compiler is Clang, vecops emits a separate guarded adapter
that calls `GOMP_parallel` from that exact packaged runtime; it does not enable
compiler OpenMP or link a second runtime. This adapter matches ATen's
grain-size-one chunking, lazy thread initialization, thread-ID and parallel
guards, nested serialization, and post-join exception propagation. Before
using it, vecops verifies the packaged library is already loaded, checks with
`dladdr` that each required GOMP/standard `omp_*` symbol resolves to that exact
file, and confirms its maximum team size matches Torch's reported setting.
Those function addresses are installed directly into the bridge, so loader
symbol interposition cannot redirect worker IDs or nested-region checks to a
different OpenMP runtime. The library's content identity participates in the
bridge cache key and publication guard.
Unknown or ambiguous runtimes fail with an actionable error instead of
silently switching compilers or running serially. The direct GOMP entry point
is a GNU runtime ABI boundary, so it is intentionally limited to the
validated packaged runtime path. A bridge compiler override is available only
when both `VECOPS_TORCH_BRIDGE_CC` and `VECOPS_TORCH_BRIDGE_CXX` are explicitly
set.

Set `VECOPS_THREAD_POOL=native` before importing vecops to inject the optional
process-local persistent pool into every generated Torch bridge. The pool is
created once in the `_C` module, uses `VECOPS_NUM_THREADS` when set (otherwise
`torch.get_num_threads()`), and parks its worker threads after a short bounded
spin so idle vecops workers do not compete with Torch OpenMP work. On Linux,
workers 1..P-1 are pinned across the process's allowed CPU set; the submitting
thread's affinity is restored after each submission so later compiler
subprocesses do not inherit a single-CPU mask. `VECOPS_THREAD_POOL_CPU_LIST` overrides the detected set with
an explicit comma/range CPU list when the submitting thread was already pinned
before the legacy pool was created; the adaptive direct-call facade instead
requires that override to lie inside the caller's affinity. `VECOPS_THREAD_POOL_SPIN_COUNT` controls the bounded spin
before parking and defaults to 256. This mode is opt-in because a workload
with substantial interleaved ATen work can still prefer Torch's shared team.
Team barriers use the standard-library atomic wait facility when the selected
library provides it. Older Linux standard libraries park on a lock-free
32-bit generation word with private futexes after the bounded spin; other
platforms use a condition-variable fallback with predicate-based wakeup. This
keeps barrier waits parked rather than burning CPU on unsupported
`atomic<uint64_t>::wait` implementations.
Set `VECOPS_NATIVE_THREAD_POOL_OPS` to comma-separated qualified-name glob
patterns to opt in only selected bridges, for example
`alphafold3_native::diffusion_*`; unmatched bridges retain the Torch pool.

For a bounded model region containing many short native submissions,
`vecops_native_thread_pool_begin_active(threads)` and
`vecops_native_thread_pool_end_active()` keep that same worker team spinning
between phases. Calls may be nested. This avoids repeated park/wakeup latency,
but the region must not cover parallel Torch/OpenMP work on the same CPUs.
Python callers should use `vecops.native_thread_pool_region(threads)`, whose
`finally` path always balances the region.

A non-Torch embedding may provide another `VecopsThreadPoolV1`; the optional
standalone OpenMP provider is one such implementation, not an implicit
dependency of generated kernels.

For source kernels, effective parallelism is part of compilation:

This remains the default for existing C++ recipes. `KernelDef.runtime_parallelism`
opts a runtime scheduler out of this specialization: its generated
`spec::Parallelism` is one and artifact/hot-call identities do not vary with
the execution budget. CPU DSL definitions opt in automatically. The actual
size-versioned call context still reaches each invocation, so changing thread
budgets or affinity does not trigger JIT compilation. Such a kernel must not
use `spec::Parallelism` for compile-time workspace sharding.

```cpp
constexpr nint_t P = static_cast<nint_t>(vecops::spec::Parallelism);

workspace.parallel_lanes<P>(
    [&](execution::TaskContext<P> task) {
      // Exactly P stable logical lanes; task.lane_id() is a scratch identity.
    });

workspace.parallel_for<P>(nint_t{0}, end, cint<32>,
    [&](execution::TaskContext<P> task, const auto& item) {
      // item is [begin,end), clipped only for the final program.
    });
```

`parallel_lanes<P>` submits exactly `P` logical lanes. `parallel_for<P>` first
turns `[begin,end)` into fixed-size range programs, balances contiguous program
ID intervals over those same lanes, and lets each lane execute zero or more
programs sequentially. `RangeWorkItem` exposes `program_id/count`,
`begin/end/extent`, the Meta-preserving `chunk()`, and `is_full()`. Full and
tail items intentionally share one callback instantiation; this API does not
silently duplicate a tile kernel for a specialized tail.

The compile-time `P` is also the replica count expected by
`worker_tensor<T,P>` and prepared operators. `TaskContext<P>::local()` selects
that lane's stable replica. Do not substitute an OpenMP thread number: the ABI
does not promise a one-to-one mapping between logical lanes and physical
workers.

Runtime operator resolution specializes a source recipe with effective `P`
before artifact lookup/build. The suffix `$Parallelism=i:P` enters the binding,
specialization, and artifact keys, and generated `vecops_spec.h` exposes the
same value as `spec::Parallelism`. Changing Torch's intra-op thread count thus
selects a distinct compiled artifact rather than reusing code whose task split
and scratch replicas were compiled for another `P`.

Batch precompile distinguishes build concurrency from execution parallelism:
`parallelism` controls concurrent compiler jobs, while
`execution_parallelism` chooses `P` (defaulting to `torch.get_num_threads()`
when Torch is loaded). They are not interchangeable tuning knobs.

`SourceKernelRecipe` owns a cold-build input validator shared by its bound
specializations. A compiler batch checks each independent validator owner once
before building and freshly again before publication; separate recipes retain
their own checks even if their artifacts are deduplicated. These source and
toolchain checks never replace per-call metadata/relationship binding. Their
results are local to one boundary, not cached across boundaries or batches.

## C ABI split

`CallAbi.h` owns the shared data records (`VecopsCall`, tensor/scalar values,
errors, dtype, and status codes). Every caller sets each record's `struct_size`
and keeps pointers valid until its synchronous ABI call returns.

It also defines `VecopsWorkspaceArenaProvider`, a process-local retained
callback table stored in the dedicated
`VecopsExecutionContext::workspace_provider` field. A generated DSO copies the
table, retains its context, pairs every successful arena token with `release`,
and keys plan metadata by `identity`. Embedding `user_data` remains independent.

`KernelAbi.h` is the artifact-DSO protocol. A DSO exports exactly
`vecops_kernel_query_v1`, which returns a static `VecopsKernelDescriptorV1`.
The descriptor's major version, record sizes, parameter requirements, and
callback pointers are validated before it is used. `workspace` reports an
external byte count; source kernels generated by vecops report zero because
their `Workspace` is internal to `__kernel__`.

`OperatorBridgeAbi.h` is separate and process-local. Generated framework
extensions use an opaque handle to enter a Python-owned `Operator`; that
handle must not be stored after the owner goes away. It is not part of an
independently distributed kernel artifact's ABI.

The Python extension keeps up to eight exact call signatures per thread for
this process-local bridge. After a slow invocation has resolved and validated
an `Executable`, a matching context-free or arena-provider,
zero-external-workspace call may reuse that executable directly. The signature
excludes tensor data addresses and runtime scalar payloads, but includes tensor
kind, dtype, device, access, shape, stride, optional presence, explicit
specialization values, and effective execution parallelism. A shape,
specialization, or Torch thread-count change therefore returns to normal
operator resolution; operator destruction invalidates the cached weak owner.
Set `VECOPS_TORCH_PREPARED_CALL=0` before process startup to disable this hot
path for diagnostics.

Source kernels without an external workspace also use a small thread-local
workspace replay cache. Its identity includes the observed Dynamic axes,
decision fingerprint, and effective execution parallelism. This is a second,
workspace-specific guard in addition to the artifact's compile-time `P`: a
plan with `P` worker replicas cannot be replayed as though it had another lane
count.

DSL `T.workspace` uses this existing replay mechanism, without another ABI.
Sites are unconditional serial declarations; extents depend on tensor metadata,
compile-time values or effective worker capacity, not arbitrary runtime scalar
payloads or data. Replay retains allocation plans rather than tensor contents.
The public `WorkspaceContext::thread_pool()` accessor borrows the executor for
grid/team scheduling; it does not grant workers concurrent allocation authority.
Dense manually replicated worker buffers need explicit cache-line padding when
desired, and do not imply prepared-operator phase-lifetime storage reuse.

There is no `runtime/Abi.h` umbrella. Include the protocol header that a
component actually implements, avoiding unintended bridge dependencies in an
artifact DSO.

## Optional memory and timeline observation

`VecopsAllocationObserverV1` retains its legacy workspace-allocation callback
prefix and appends optional `on_memory_event`. Consumers must check `struct_size`
before reading the tail; old callers providing only the prefix remain valid.
`VecopsMemoryEventV1` carries action (`0` allocate, `1` release, `2` region begin,
`3` region end), allocation identity, byte count and site. Pointers are transient
identities only, never permission to dereference or export tensor contents.
Callbacks may run on worker threads, must not acquire the Python GIL and must
not throw through the C ABI. `WorkspaceContext::allocation_observer()` borrows
the observer for generated scratch and execution-region instrumentation.

Python collectors anonymize allocation IDs and bound event storage; dropped
events are reported. `vecops.workspace_observer.tracking()`, `drain()` and
`drain_memory()` feed profiling/Chrome traces. DSL scratch lifetimes and
workspace allocation sites are observed, not all allocator or process memory.
Tracing is opt-in and does not imply output ordering or extra synchronization.
