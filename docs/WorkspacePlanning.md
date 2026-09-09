# Workspace planning and prepared operators

Vecops separates workspace handling into three layers:

1. `LogicalWorkspacePlan` is pointer-free. It records semantic allocation
   sites, Meta axis contracts, decision fingerprints, domains, and structured
   lifetimes.
2. `WorkspacePlacement` assigns those logical slots to fast or slow memory
   and offsets. Placement can be recomputed when HBM availability changes
   without retracing the operator.
3. `BoundWorkspacePlan` attaches concrete arena bases. Prepared operators
   resolve each site once and retain a direct `BoundWorkspaceSlot`; kernels do
   not hash site names or search placement tables in their inner loops.

`WorkspaceContext` is the kernel-call authority joining these layers. In
dynamic mode it serves fast-preferred requests from the caller's arena and
spills to aligned host allocation. Trace mode performs real execution while
recording the logical plan; replay mode resolves the same stable sites from a
bound placement. Compiler-managed source kernels may take it as their first
argument:

```cpp
void __kernel__(WorkspaceContext& workspace, TensorLike auto x) {
  auto phase = workspace.serial_scope("projection");
  auto scratch = workspace.request("temporary", {.bytes = bytes});
  // use scratch.replica()
}
```

The generated adapter also accepts legacy kernels without that first
argument. A caller-provided ABI workspace becomes the context's fast arena.
When neither an external context nor raw arena is supplied, the easy path uses
a small calling-thread replay cache: the first call for a Meta-axis shape
executes in trace mode, and later identical calls use owned placed arenas.
This avoids turning migrated kernels into repeated heap allocation while
keeping explicit HBM/model-wide authorities in control. Framework and Python
operator bridges forward the original workspace and execution context instead
of discarding them.

`WorkspaceTrace::serial_scope()` expresses lifetime, rather than attempting to
infer last use from C++. Sibling scopes are ordered and may reuse storage;
allocations in the same scope overlap. A worker-local request has an explicit
logical replica count and receives at least 64-byte replica padding. Logical
worker IDs, not operating-system thread IDs, select replicas.

Each allocation site is identified by its recipe/scope/local-name path plus a
semantic hash. Names must therefore be stable parts of the operator recipe.
Do not derive them from a global counter, scheduling order, or the order in
which workers arrive. Hash collisions and an incompatible repeated contract
are checked against the retained canonical path.

Use `request()` for temporary storage whose pointer does not escape its
lexical scope. Use `bind()` when a prepared operator retains the pointer. In
dynamic mode a spilled bound allocation is owned until the context is
destroyed; in trace/replay it keeps the lexical lifetime used for placement.
Prepared operations assigned the same replay offset must therefore execute in
the same serial order expressed by their construction scopes.

Dynamic Meta axes retain alignment and bounds in `AxisContract`. Replay also
requires an exact `DecisionFingerprint`, because compatible Dynamic types do
not imply that matmul packing, attention strategy, or another allocation path
stays unchanged. Binding additionally checks actual requested bytes against
the slot capacity.

## Prepared Matmul

`ops::Matmul<Config>` remains the compatibility/configuration form returned by
`ops::matmul(config)`. For architecture-family matmuls its `prepare()` member
binds one scratch sub-slot from either `WorkspaceView` or `WorkspaceContext`
and returns the operator's nested `Prepared<Invocation>` state. This keeps the
legacy Matmul type and generated hot path unchanged while placing preparation
inside the public operator rather than adding a framework-level wrapper. The
prepared state retains the selected invocation, operand bindings, and scratch
address, so repeated calls do not
repeat family, orientation, packing decisions, or workspace-size planning.
The existing invocation still performs fixed-cost bump suballocations inside
the bound slot; converting each of those sites to direct `BoundWorkspaceSlot`
pointers is a later incremental migration.

That fully bound form is only valid while both its operand storage and parent
workspace stay alive. It is intentionally not a general tile-rebinding cache:
DataAccess bindings may contain concrete addresses and materialized-output
commit state. Long-lived model plans must cache layout/backend decisions and
scratch slots, then construct a fresh invocation for each new set of operand
addresses.

## Placement policy

`FastRequired` must fit in the fast arena. `FastPreferred` is considered by
benefit density and spills to slow memory when necessary. `SlowAllowed` starts
in slow memory. Overlapping lifetimes are interval-colored; non-overlapping
allocations may share an offset. Pointer binding aligns arbitrary arena bases,
so owners must provide capacity for the requested placement plus possible
leading alignment padding.

The current API is the pointer-free planning foundation. Migration of existing
operators should keep their legacy overloads, replace duplicated
`required_workspace` arithmetic with the same prepare recipe in counting mode,
and move allocations to stable sites incrementally.

## Parallel source-kernel migration

`WorkspaceContext` itself is deliberately not synchronized. A source kernel
must issue allocation requests before entering `parallel_for`, using
`WorkspaceDomain::WorkerLocal` and the actual logical worker count. Each worker
then creates its private `WorkspaceView` from `slot.replica(thread_id)`. This
keeps the trace independent of worker arrival order and gives every replica a
64-byte-aligned, 64-byte-padded region.

Long-lived intermediates belong to an enclosing scope that remains open across
all consuming phases. Sequential phases use sibling scopes, allowing the
placer to reuse their local and operator-scratch offsets. Data-dependent
branches may change computation but must not change the request sequence; a
choice that changes allocation topology belongs in the decision fingerprint.

Generated adapters reserve the high execution-context flag
`VECOPS_EXECUTION_CONTEXT_FLAG_WORKSPACE_CONTEXT`. With that flag set,
`user_data` points to a process-local C++ `WorkspaceContext`, allowing an outer
model trace/replay authority to span multiple JIT kernels. Without it, a raw
call-frame workspace selects dynamic mode; a null/zero raw workspace selects
the calling-thread trace/replay cache described above. The cache keys exact
observed Dynamic axes and retains four shapes by default. Allocation decisions
not represented by those axes must be included in the explicit decision
fingerprint. Legacy kernels do not construct any context because signature
selection is compile time.
