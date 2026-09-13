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

The generated Torch extension installs a bridge backed by Torch's intra-op
runtime. It reports `at::get_num_threads()`, detects an existing
`at::in_parallel_region()`, and executes the logical task set through the
active Torch/GNU OpenMP team where that integration is available, with a Torch
`at::parallel_for` fallback on other builds. Nested entry is serialized to
avoid oversubscription. Consequently a Torch operator does not discover an
unrelated "ambient vecops OpenMP maximum" or open a competing private vecops
pool. A non-Torch embedding may provide another `VecopsThreadPoolV1`; the
optional standalone OpenMP provider is one such implementation, not an
implicit dependency of generated kernels.

For source kernels, effective parallelism is part of compilation:

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

There is no `runtime/Abi.h` umbrella. Include the protocol header that a
component actually implements, avoiding unintended bridge dependencies in an
artifact DSO.
