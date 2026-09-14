# Workspace planning and prepared operators

Vecops separates logical storage, physical placement, and execution binding:

1. `LogicalWorkspacePlan` is pointer-free. It records stable allocation sites,
   Meta axis contracts, decision fingerprints, domains, and lexical lifetimes.
2. `WorkspacePlacement` assigns those sites to fast or slow memory and chooses
   offsets. Placement may be recomputed when HBM capacity changes without
   retracing the kernel.
3. `BoundWorkspacePlan` attaches concrete arena bases. Replay then resolves the
   stable sites to direct views; no allocation or placement search belongs in a
   tile loop.

`WorkspaceContext` is the source-kernel API joining these layers. Dynamic mode
serves requests from caller arenas and spills to aligned host allocation. Trace
mode performs the real call while recording its pointer-free plan. Replay mode
executes the same request topology against a bound placement.

## Typed workspace tensors

Kernel authors normally allocate Tensors rather than count bytes:

```cpp
void __kernel__(execution::WorkspaceContext& ctx, TensorLike auto input,
                TensorLike auto output) {
  constexpr nint_t P = static_cast<nint_t>(spec::Parallelism);
  auto lifetime = ctx.serial_scope("projection");

  auto global = ctx.tensor<bfloat16_t>(
      "global", make_shape(B, N, C));
  auto local = ctx.worker_tensor<bfloat16_t, P>(
      "local", make_shape(TileRows, C));

  ctx.parallel_for<P>(nint_t{0}, tasks, cint<1>,
      [&](execution::TaskContext<P> task, const auto& item) {
        auto tile = task.local(local);
        // `tile` is this logical lane's rank-two [TileRows,C] Tensor.
      });
}
```

`ctx.tensor<T>(name, shape)` infers a row-major contiguous layout.
`ctx.tensor<T>(name, layout)` and
`ctx.tensor<T>(name, shape, strides)` retain explicit element strides. The
corresponding `worker_tensor` overloads prepend one logical lane/replica axis.
`WorkspaceTensorOptions` controls base alignment and placement; the default is
`FastPreferred` with vecops' default vector alignment.

Typed workspace storage is writable. Its element type must be non-const,
trivially copyable, and have a power-of-two size. Explicit layouts may be
contiguous, padded, or permuted dense layouts, but must be non-negative and
provably non-overlapping. Negative strides and writable broadcast/overlapping
axes are rejected. Vecops computes the reachable storage span and converts it
to bytes internally; it does not incorrectly use logical `numel` for a padded
layout.

A worker tensor's leading element stride is derived from the planner's replica
stride. Every replica starts at a boundary aligned to at least 64 bytes, and the
replica stride is rounded up to that alignment. It is therefore not generally
equal to the inner Tensor's logical `numel`. Use `TaskContext<P>::local(worker)`
to select a replica; do not index it with an operating-system or OpenMP thread
number. A logical lane is the stable scratch identity even if an embedding runs
several lanes sequentially on one physical worker.

The low-level byte `request()`/`bind()` and `BoundWorkspaceSlot` APIs remain
implementation and compatibility tools. Ordinary source kernels should prefer
typed tensors and prepared operators.

## Scopes and placement

`serial_scope(name)` expresses lifetime explicitly. Requests in one scope are
simultaneously live and therefore do not overlap in placement. Ordered sibling
scopes have disjoint lifetimes and may reuse the same arena offsets. A
long-lived Tensor belongs to an enclosing scope that remains open across every
phase that consumes it.

Each site identity includes the recipe, scope path, local name, and domain.
Names must be stable parts of the kernel recipe; never derive them from worker
arrival order or a process-global counter. Repeated sites are checked for a
compatible contract.

`FastRequired` must fit in the fast arena. `FastPreferred` is prioritized using
estimated traffic density and spills when necessary. `SlowAllowed` begins in
slow memory. The placer interval-colors non-overlapping lifetimes separately in
each tier.

Dynamic Meta axes retain alignment and bounds in `AxisContract`. Replay also
checks the `DecisionFingerprint`, because compatible Dynamic types do not prove
that a matmul packing or attention-strategy decision stayed unchanged.

## Address-free prepared operators

The preferred operator form separates a storage-free planning pattern from the
actual tile addresses:

```cpp
auto op = ops::matmul(ops::MatmulConfig<Atom>{});
auto prepared = op.prepare<P>(
    ctx, "projection_scratch", max_rows, N, K,
    tensor::unbind(sample_a),
    tensor::unbind(sample_b),
    tensor::unbind(sample_output));

ctx.parallel_for<P>(nint_t{0}, rows, cint<TileRows>,
    [&](execution::TaskContext<P> task, const auto& item) {
      prepared(task, item.extent(), N, K,
               a(range(item.begin(), item.end()), reserve), b,
               output(range(item.begin(), item.end()), reserve));
    });
```

For an operation that needs scratch, `prepare<P>` registers one named
worker-local scratch Tensor. It retains unbound operand contracts and returns
an object invoked with `TaskContext<P>` plus the actual operands. Invocation
selects the lane's scratch internally, reconstructs the lightweight bound
operation, validates that active extents fit the planned capacity, and handles
its execution-resource scope. Scratch-free prepared operations use the same
surface without inventing storage. Kernel code does not call
`required_workspace`, create `WorkspaceView`/`ExecutionSession`, or reset a
byte cursor between prepared calls.

The stable address-free prepared surface is available for:

- architecture-family `Matmul`, including single-C and explicit
  accumulator-input forms;
- `LayerNorm`, including optional affine parameters;
- `Softmax`;
- `ScaledDotProductAttention`/SDPA, including its optional mask and bias
  operands;
- `MatmulPack`;
- `Transpose`.

Patterns and actual operands use the binding rules in [Tensor](Tensor.md).
In particular, Dynamic pattern values are capacities, strides stay exact, and
stateful transform values come from the actual Spec passed to the prepared
call. Matmul's legacy fully-bound `Prepared<Invocation>` overloads remain for
fixed addresses, but they are not the tile-rebinding surface described here.

For Matmul, the short invocation form `prepared(task, a, b, c)` derives the
active M/N extents from `c` but retains the logical K supplied to `prepare`.
Packed A/B layouts contain padded tile extents and cannot recover a logical K
tail. A caller whose active K changes within one prepared capacity must use the
explicit `prepared(task, active_m, active_n, active_k, ...)` form.

### Scratch accounting is explicit

Each scratch-using ordinary `prepare<P>(ctx, distinct_name, ...)` creates an
independent worker-tensor site. If several prepared objects are constructed in
the same `serial_scope`, their scratch lifetimes overlap and placement accounts
for their **sum**, even when calls happen sequentially inside each lane. No
shared scratch group or automatic `max(requirement...)` union is currently
implemented. Do not document or rely on such a group.

This conservative rule makes differently sized replica strides safe. It can
increase the arena peak relative to old code that manually reused one byte
buffer, but it keeps the high-level API free of byte arithmetic and ambiguous
aliasing.

## `serial_use`: barrier-separated whole-team phases only

`ctx.serial_use(name, factory)` is the narrow opt-in for prepared scratch that
may share placement with a sibling serial phase. The factory receives a typed
binding authority and returns the prepared object:

```cpp
auto phase_a = ctx.serial_use("phase_a", [&](auto& serial) {
  return op_a.prepare<P>(serial, "scratch", /* unbound patterns */);
});
auto phase_b = ctx.serial_use("phase_b", [&](auto& serial) {
  return op_b.prepare<P>(serial, "scratch", /* unbound patterns */);
});

ctx.parallel_lanes<P>([&](execution::TaskContext<P> task) {
  phase_a(task, /* actual operands */);
}); // synchronous whole-team completion
ctx.parallel_lanes<P>([&](execution::TaskContext<P> task) {
  phase_b(task, /* actual operands */);
}); // starts only after phase A has finished
```

The construction scopes are siblings, so placement may color their scratch at
the same offsets even though the returned objects remain callable. This is safe
only when every logical lane has left phase A before any lane enters phase B—an
explicit whole-team barrier, normally supplied by the return of the first
synchronous parallel region.

Never use `serial_use` for two prepared operations in a per-lane pipeline such
as `op_a(task); op_b(task);`. Lanes advance independently, so one lane may use
phase-B scratch while another still uses phase A. If differently sized sites
share an offset, their replica strides can map those lanes onto overlapping
addresses. Per-lane pipelines must use ordinary same-scope prepared sites, and
therefore pay the sum described above.

## Replay and physical arenas

When no external raw workspace is supplied, generated adapters use a small
calling-thread `WorkspaceReplayCache`: the first compatible call executes in
trace mode and subsequent calls replay a placed plan. A caller-provided ABI
workspace becomes the fast arena. A `WorkspaceArenaProvider` can instead bind
plans to explicit fast/slow storage; `MemoryWorkspaceSession` maps those tiers
to HighBandwidth and Default memory without coupling the memory core to
workspace types.

Shared-arena providers expose the same session-owned fast/slow bases to each
cached plan, so their capacities are session totals rather than per-entry
budgets. A changed provider identity replaces cached plan metadata. See
[Memory](Memory.md) and [Runtime](Runtime.md) for the provider and execution
context ABI.

`WorkspaceContext` itself is intentionally single-threaded. Allocate tensors
and prepare operators before entering a parallel region. Parallel callbacks may
use returned Tensor views, `TaskContext::local`, and prepared operations; they
must not issue new context allocation requests whose order would depend on lane
arrival.
