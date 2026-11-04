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

## KernelBuilder plans

`execution/KernelPlan.h` provides a persistent source-kernel model.
`KernelBuilder::tensor` and `worker_tensor` declare symbolic typed storage;
they expose layout metadata and cannot read capture-time tensor contents.
`prepare_operation<P>(name, op, patterns...)` accepts the existing address-free
operator patterns. Planning first declares all scratch; instantiation then
places buffers and scratch together, allocates one arena per tier, and binds
instance-owned prepared objects. No scratch declaration or placement lookup
belongs in replay. Allocation remains conservative: independently live sites
are not colored merely because their declarations occupy sibling C++ scopes.

Default storage prefers fast memory and falls back to slow memory when the
preferred arena is unavailable; required-fast storage must fit. A provider
that deliberately shares arenas between plans still requires its embedding's
serialization/lease discipline across instances. Within one plan, distinct
buffers and operator scratch have disjoint placed spans.

`grid` records a program callback. `worker_grid` records a worker callback
whose synchronous `worker.programs(...)` loop can be surrounded by per-worker
prologue/epilogue. Local captures inside that loop are valid only until it
returns. `single` records data work once per execution; ordinary C++ outside
these nodes executes only during construction. All recorded callbacks receive
explicit input/scalar/state bindings and must not retain references to the
builder's stack. SME/AMX resource scopes are entered on the executing worker,
not retained from plan construction.

## Common buffer placement and segmented operator storage

The implementation now lives in `execution/BufferPlan.h`; the workspace names
above are aliases to this common IR. `BufferMemorySession` accounts for graph
and tensor owners while placing kernel buffers. KernelBuilder uses this same
colorer rather than a second append-only allocator. Buffer names and traffic
hints are retained through the optional prepare-time provider ABI.

Prepared architecture Matmul declares backend scratch, packed A, packed B and
auxiliary materialization separately. Prepared SDPA declares backend workspace,
scores, probabilities, output accumulation, row statistics and cached Q/K/V
packs separately. These are ordinary typed worker buffers and may occupy
different memory tiers. Each sub-buffer's active capacity is checked before
execution, including bounded dynamic shapes. Current kernel-plan lifetimes
remain conservative for independently callable prepared operations; sibling
C++ declaration scopes alone do not prove storage reuse.

`kernel::WorkspaceView` can bind a table of separately placed segments. Its
mark/rewind captures each segment cursor. Nested operators use a scoped,
contiguous child workspace, preserving the parent's activated CPU resource
configuration and preventing an inner Matmul's packing sites from aliasing an
attention score buffer. Legacy contiguous views retain their existing meaning.
Segments are raw typed storage; no `torch.Tensor` wrappers occur inside native
workers. No placement lookup or allocation occurs on replay.


## Grid effects, consumed values and linked serial kernels

`KernelBuilder` uses the common `LogicalBufferPlan` for both named intermediates
and each separately declared prepared-operation scratch part. Bind a symbolic
tensor as `execution::read(x)`, `write(x)`, `read_write(x)` or `overwrite(x)`.
The callback receives the original bound tensor with unchanged Const/stride
metadata. `overwrite` is an author assertion that every subsequently read byte
is initialized in this grid on every replay, including dynamic tails. It must
cover a root buffer view. Plain bindings remain opaque and conservatively live
for the complete instance; `write` alone does not prove initialization.

Recording a grid returns `KernelStage`. `builder.consume(x, stage)` checks that
no later recorded use of this slot, its views, or a referencing prepared state
exists. The declaration changes no eager behavior and grants no external input
ownership. There is no additional runtime free or barrier. Internally owned
buffers with a defining overwrite and known effects are colored using their
grid intervals and existing completion barriers.

Prepared state remains pinned by default: constructors/destructors or retained
packing can access workspace outside a grid. An audited prepared-operation type
can declare `static constexpr bool transient_workspace = true`. This contract
requires constructors/destructors never to access scratch contents, all calls
to initialize scratch before reading it, and no scratch contents to survive a
call. Its declared sub-buffers use the union of calling-grid intervals, allowing
reuse in gaps between invocations. Opaque state retaining such an operation
pins its scratch transitively. `ops::Matmul::PatternPrepared` and
`ops::MatmulPack::PatternPrepared` satisfy this contract with fresh per-call
workspace cursors and `ExecutionSession`; persistent user states do not inherit
it automatically.

`KernelProgram({plan_a, plan_b}, pool, provider)` links serial C++ kernels through
one placement and one pair of arena allocations. Eligible internal intervals
are shifted into a common program timeline. Unknown/prepared persistent storage
stays live across the entire program, including construction and destruction.
Use `program.set(phase, argument, address_or_scalar)` before `program.replay()`.
The program owns its instances and submits one resident team. Its member frames
cannot be obtained for independent concurrent replay. `buffer_plan()` and
`placement()` expose cold evidence. This linker currently accepts C++
`KernelPlan` objects; it does not reinterpret independently compiled Torch
executables as global phases or release model weights automatically.

## Inplace permission and graph donation

`builder.allow_inplace(input, output)` declares an audited kernel capability for
identical complete argument views. Every old input read must finish before its
first overwrite, through same-index worker ordering or an existing completed
grid. Other input aliases must be excluded. It does not select an alias on its
own. The optional source-plan ABI exports call-value indices, including correct
mapping around missing optional arguments; legacy consumers receive only their
known prefix.

`GraphCapture.compile_torch(native_pointwise=True, native_donation=True)` opts
into dead-storage reuse for supported pointwise grids and source-kernel
capabilities. The graph separately verifies internal ownership, whole alias
liveness, dtype/layout, an uninitialized out buffer, and other-input alias
hazards. It extends the original allocation lifetime through every donated
logical version before common planning, and hoists views using relocated byte
spans. Inputs, persistent/escaping old values, other planners' bindings and
unproven operations retain their allocation semantics. Tensors touched by opaque
custom operators or specialized collective providers are protected as whole
alias groups; local last use does not prove peer completion. Borrowed graph outputs
still require `borrow_outputs=True`; eager schemas are unchanged. The default
is `native_donation=False` pending workload-specific correctness and performance
qualification. Memory reuse performs no DCE of mutable or void native calls.

Ownership follows storage through view/reshape aliases, rather than only the
Tensor ID originally produced by a kernel. Donation tests the final data use:
the readonly `other` argument of `view_as`, `reshape_as` and `expand_as` describes
shape, and does not keep old contents alive. Reads of `self`, writes, unknown
operations and escaping aliases remain conservative. Earlier uses of an out
buffer are also conservative, including metadata uses that could precede the
definition of the proposed input view. Rewritten JIT references resolve at use,
after static or ordinary view lowering establishes their targets.

## Explicit initializer-to-steady cache transfer

A Torch executable may expose cold `fixed_output_values(values)` descriptors,
then prepare `producer.prepare_output_transfer(values, consumer,
input_offset=..., expected_replays=...)`. This API requires an active common
`BufferSession`, `native_donation=True` ownership analysis, and a caller promise
that the cache tuple is immutable for the entire invocation, including other
graphs. It only selects privately owned retained initializer outputs in slow
memory that the steady executable actually reads. Consumer writes, opaque or
collective uses, and aliases outside the selected cache inputs exclude a storage.

The planner leaves all existing addresses, offsets and tiers unchanged, ranks
eligible complete storages by static read bytes per byte of capacity, and binds
individual fast leases into currently free space. It skips storages larger than
the largest free extent and caches with no reads. This is a conservative capacity
heuristic, not a bandwidth-optimal placement proof. Preparation must happen after
cold eager prototypes are released. It retains canonical slow storage, so peak
logical capacity includes both versions; it does not retire or migrate weights.

`transfer.replay(values)` is an explicit once-per-invocation native vec copy
after the initializer and before steady consumption. Four independent vector
streams preserve bits; no SDMA/libc fallback is used. The copy graph and target
views are prepared once; replay performs no allocation, mapping, capture or
placement. Full storage capacity, offset, strides and input alias groups remain
identical. A different source address or execution thread is rejected. The
caller must serialize the producer, transfer and consumer. C++ program scratch
reuse, graph donation and this transfer all draw from the common capacity
manager, with different explicit ownership/completion contracts.

Link-time alias closure includes both StorageImpl identity and overlapping
storage byte intervals. Captured input/state spans are cold, non-owning proof
metadata: an alias outside the rebound cache tuple, including a distinct
frombuffer StorageImpl or a bound module buffer, excludes promotion. Trailing
scalar specialization arguments remain distinct from runtime tensor call-value
indices; nested/custom argument paths stay conservative. Copy kernels have an
explicit worker upper bound, clamped to the compiled execution provider's
capacity, independently of compiler job count.

Logical pinning applies within a kernel/program. An explicitly shared workspace
provider also requires contents to be disposable between unrelated invocations;
serialization alone does not preserve hidden packing data. Values that survive
calls should be declared external tensors with owning leases, or kept in an
isolated provider/program covering all users. The AF3 native scratch paths use
invocation-local contents; opaque custom state is not made recyclable merely
by a consume marker.
