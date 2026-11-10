# Workspace planning and prepared operators

Vecops separates logical storage, physical placement, and execution binding:

1. `LogicalBufferPlan` is pointer-free. It records stable allocation sites,
   Meta axis contracts, decision fingerprints, domains, and lexical lifetimes.
2. `BufferPlacement` assigns those sites to fast or slow memory and chooses
   offsets. Placement may be recomputed when HBM capacity changes without
   retracing the kernel.
3. `BoundBufferPlan` attaches concrete arena bases. Replay then resolves the
   stable sites to direct views; no allocation or placement search belongs in a
   tile loop.

`WorkspaceContext` is the source-kernel API joining these layers. Dynamic mode
serves requests from borrowed caller arenas and sends owned spills through the
active resource-backed workspace provider. Those roots require Runtime capacity
admission; there is no uncharged heap fallback. Trace mode performs the real
call while recording its pointer-free plan. Replay mode
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

The low-level byte `request()`/`bind()` and `BoundBufferSlot` APIs remain
lower-level buffer interfaces. Ordinary source kernels should prefer
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

Generated adapters receive a call-scoped `WorkspaceArenaProvider`. The default
provider routes allocations through `MemoryRuntime` and its capacity ledger;
`resource_workspace_arena_provider(runtime, pool_ids)` binds a specific runtime
and permitted pools. Pool IDs are opaque. Native fast/slow placement hints are
translated by the provider; they are not hardware tiers in the resource model.
A caller-provided ABI workspace remains a borrowed arena with explicit capacity.

Cached plans retain their own backing owners, and every invocation receives the
current execution services. Dynamic spills are charged through the same runtime.
Graph-managed execution rejects undeclared owning scratch instead of allocating
an unplanned payload. The upper graph compiler can bind declared native buffers
together with tensor storage in an offset-only recipe. See
[Memory](Memory.md) and [Runtime](Runtime.md) for ownership and execution context.

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
places buffers and scratch together and binds instance-owned prepared objects.
Standalone C++ instances obtain their arenas through the resource provider;
metadata-only graph instances receive compiler-selected buffer addresses and
owners at activation, without first allocating a separate private arena. No
scratch declaration or placement lookup belongs in replay. Allocation remains conservative: independently live sites
are not colored merely because their declarations occupy sibling C++ scopes.

Native tier hints guide arena planning; the default provider resolves capacity
and allocation through the Runtime's authorized candidate pools. They do not
imply separate physical fast and slow pools. Required-fast storage must fit its
declared arena constraint. A provider that deliberately shares arenas between
plans still requires its embedding's serialization/lease discipline across
instances. Within one plan, simultaneously live buffers and operator scratch
have disjoint spans. Explicit consume/overwrite contracts may prove disjoint
lifetimes and permit reuse of the same offsets.

`grid` records a program callback. `worker_grid` records a worker callback
whose synchronous `worker.programs(...)` loop can be surrounded by per-worker
prologue/epilogue. Local captures inside that loop are valid only until it
returns. `single` records data work once per execution; ordinary C++ outside
these nodes executes only during construction. All recorded callbacks receive
explicit input/scalar/state bindings and must not retain references to the
builder's stack. SME/AMX resource scopes are entered on the executing worker,
not retained from plan construction.

## Common buffer placement and segmented operator storage

Native storage declarations use `execution/BufferPlan.h` directly.
`WorkspaceContext` and typed workspace tensors consume these same buffer records;
there is no compatibility `WorkspacePlan.h` header. The Python graph compiler converts
native declarations to the same resource-runtime requirements as tensor storage.
`MemoryRuntime` accounts for graph, eager and native owners through one ledger.
KernelBuilder uses this colorer for its local plan; buffer names and traffic
hints are retained through the prepare-time provider ABI.

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

## Initializer caches and explicit copies

A joint model program links an initializer's owned cache outputs and the steady
program's inputs to the same model storage identity. `compile_model()` combines
their lifetimes and read/write traffic, including the steady `repeat_count`,
before choosing placement. The initializer writes directly into that final
placement. AF3 therefore has no later cache-promotion/finalization pass and no
extra copy merely to change a cache from one memory tier to another. The
`cache_read_only` contract and ordinary capture/alias guards still apply.
Framework-owned outputs remain separately budgeted framework allocations; a
budget declaration does not claim that these outputs were physically relocated.

For a separately chosen destination, a Torch executable exposes cold
`fixed_output_values(values)` descriptors and an explicit copy preparation API:

```python
transfer = producer.prepare_output_transfer(
    producer.fixed_output_values(values), consumer,
    input_offset=cache_input_offset,
    target_program=destination_program,
    target_bindings={0: "destination-cache"},
)
replacement_values = transfer.replay(values)
```

`destination_program` must already be materialized. Its buffer requirements,
placement and capacity admission are the caller's responsibility. The binding
map names the destination logical buffer for each selected cache index; aliases
of one storage must name the same destination and be redirected together. Missing
or empty targets fail explicitly. This helper does not search free capacity,
choose a pool, infer profitability or accept an `expected_replays` promotion
threshold. Both source and destination coexist and remain charged while live.

Sources must be fixed, privately owned initializer outputs. The helper checks
consumer write/opaque effects and aliases outside the selected cache inputs,
including distinct StorageImpl objects with overlapping physical byte spans and
bound module state. Unsafe selections raise an error. Copies cover the complete
logical storage and preserve dtype, strides and nonzero view offsets; they do
not copy an unrelated shared slab envelope. Unselected cache values remain in
their original storage.

`transfer.replay(values)` performs the prepared, bit-preserving native copy with
no tensor payload allocation, placement search, mapping or capture. Targets and
escaped Tensor aliases retain their actual C++ physical binding, even if the
Python destination Program is closed. Source addresses and the preparation
thread are checked. The caller serializes the producer, transfer and consumer,
and promises that the selected caches stay immutable for the invocation across
other graphs. Execution width is part of the copy kernel specialization and is
independent of compiler job count.

Logical pinning applies within a kernel/program. An explicitly shared workspace
provider also requires contents to be disposable between unrelated invocations;
serialization alone does not preserve hidden packing data. Values that survive
calls should be declared external tensors with owning leases, or kept in an
isolated provider/program covering all users. The AF3 native scratch paths use
invocation-local contents; opaque custom state is not made recyclable merely
by a consume marker.


## Cold communication capacity and offset-only proposals

A registered collective describes its layouts and selected route. After earlier
broadcasts or barriers, registering a later graph does not itself certify that
all publication and relay roots already exist. A model closes that dependency
before admitting its graph capacity:

1. Compile an offset-only proposal and query `PreparedProgram.planned_placement`.
   This describes the recipe; `placement()` continues to require actual backing.
2. Call each graph plan's `configure_communication_placements` with that proposal.
   Provision each distinct transport through public
   `vecops.graph.provision_communication_capacity(plans)`.
3. If any participant added transport roots, refresh the budget snapshot and
   replan. Once provisioning reports no growth, call
   `plan.finalize_communication_contract()` for each plan.
4. Admit the graph, then materialize backing and compile the final executables.
   Final binding retains the closed routing contract; new observations of an
   external pointer do not silently select another route after backing.

`GraphMemoryCompiler.prepare_model(..., finalize_resources=callback,
preparation_consensus=agree)` implements this fixed point. The callback receives
an address-only `PreparedProgram` and returns a bool indicating changed external
resource capacity. Distributed callers provide an all-participant boolean AND
for `agree`; local compilation, resource growth and admission then follow the
same retry branch on every participant. The coordinator must cover the complete
model preparation, including any smaller communication groups. Graph backing is
never acquired inside that loop.

When only some participants obtain their pending grants, those participants
retain their proposals and grants while unsuccessful peers recompile against
the remaining capacity. This prevents repeated competition for the same preferred
pool. Before every collective finalizer, all participants withdraw graph grants
and synchronize while retaining their proposed layouts and reserved addresses.
No compilation occurs in that gap. Transport growth invalidates all proposals;
otherwise the fixed proposals are admitted again. A fatal peer failure closes
all local proposals and refunds their pending capacity.

If a final estimate observes less physical free space than the planning sample,
preparation can leave the observed loss as fresh-allocation headroom on its next
attempt. Ordinary observations require unchanged accounting, epoch and independent pool limits;
pool allowances must equal their complete group allowance. An already proved
margin survives unrelated resource accounting or global epoch changes. It is
invalidated by balance, cache, pool-quota or membership changes in the connected
capacity-group/pool/overlapping-constraint component; an incomplete component
also retains epoch protection. New typed learning still requires the original
complete same-epoch fingerprint, so temporary peer-grant ABA cannot teach a
spurious physical margin. The guard preserves reusable idle cache capacity and
ends with preparation. A typed physical rejection also learns each other untruncated group's
observed physical decline between those same-accounting snapshots, so several
declining pools need not fail one at a time. Per-group observations take a
maximum, never a sum. It changes only the compiler's private fit bound; native
capacity admission still checks the actual request. The compiler instance is
not reentrant. Compact fallback starts after the preferred-layout exception
handler has exited, releasing its failed layout metadata before the next compile.

A known, unaccepted shared physical-free reserve rejection is a separate
observation. Its error carries only primitive facts from the last actual reserve
snapshot. Local grant rollback can advance an epoch without changing balances;
this specific path may rebase the epoch when compilation, last reserve and
current snapshots agree on every nonphysical group, alias-pool limit and shared
constraint. Changed peer balances or independent limits prevent learning.
Pending is subtracted once from the raw failure sample, and idle cache remains
reusable. Other capacity, permission and uncertain-commit errors keep their
existing boundaries; reservation itself does not blindly retry capacity errors.

A standalone one-provider Torch graph uses its provider's cold consensus. A graph
with several independent transport groups uses explicit model-level preparation
and its common coordinator. Neither interface encodes CPU sockets, NUMA domains
or a fixed memory hierarchy.

The native communicator's `provision_registered_capacity()` drains at a caller
controlled cold boundary, proves that prior readers stopped, removes imported
aliases, provisions the registered size/relay union, and waits for new peer
mappings before returning. Its collective bool reports growth on any member.
`preparation_consensus(bool)` exchanges only preparation metadata. Both operations
are outside numeric replay; sealed replay uses the already provisioned roots.

Frontend resource descriptions must follow the actual execution roles. A rank
that only receives a result does not thereby require its producer's compute
graph or workspace. A frontend can use a shared pure scheduling function for
metadata discovery and execution, retain output geometry for receive-only
participants, and omit compute requirements only when that schedule proves
there are no local fragments. This is a frontend contract; the resource model
does not interpret sample counts, ranks' model roles or operator names.
