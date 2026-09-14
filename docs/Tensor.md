# The Tensor Module

The `tensor` module (in `vecops/tensor/`) connects raw memory, the
[Layout](Layout.md) vocabulary, and the [vec](Vec.md) SIMD layer. Its core
split:

| File | Role |
|---|---|
| `Tensor.h` | `Tensor<T, Shape, Strides, Binding>` — one typed layout/view abstraction with either a pointer or storage-free binding |
| `AccessOptions.h` | shared vocabulary: coordinates, axis tags, lane mappings, operand facts |
| `AccessPolicy.h` | kernel-owned compile-time policies (vector axis, read passes, access plan) and access defaults |
| `Transform.h` | pure vector transforms (prologue/epilogue functions) with a coordinate context |
| `DataAccess.h` | Spec/Policy binding, load/store lowering, materialization, unordered regions |
| `OptionalOperand.h` | `tensor::nullopt` — compile-time sentinel for an omitted operator operand |

The usage model is **Spec + Policy + Session**: a caller-side **Spec**
describes *what* an operand is (Tensor, compute boundary type, optional
transform, verifiable facts); a kernel-side **Policy** describes *how* the
algorithm uses it (vector axis, read passes, plan). Binding both creates a
scoped DataAccess **session** on which the kernel runs a hot loop.

```cpp
#include "vecops/kernel/Workspace.h"      // with_operands lives in kernel::
#include "vecops/tensor/DataAccess.h"
namespace tensor = vecops::tensor;
namespace kernel = vecops::kernel;

void scale_rows(float* out, const float* in, vecops::nint_t rows,
                vecops::nint_t cols, float alpha) {
  auto x_spec = tensor::input<float>(tensor::make_tensor<2>(in,  {rows, cols}));
  auto y_spec = tensor::output<float>(tensor::make_tensor<2>(out, {rows, cols}));
  using XPolicy = tensor::InputAccessPolicy<1>;   // lanes along cols
  using YPolicy = tensor::OutputAccessPolicy<1>;

  kernel::with_operands(workspace,                    // kernel::WorkspaceView
      tensor::operand(x_spec, XPolicy{}),
      tensor::operand(y_spec, YPolicy{}),
      [&](auto& x, auto& y) {
        auto tag = vecops::vec::ScalableTag<float>{};
        const auto alpha_vec = vecops::vec::fill(tag, alpha);
        for (vecops::nint_t r = 0; r < rows; ++r) {
          for (vecops::nint_t c = 0; c < cols; c += vecops::vec::size(tag)) {
            const auto v = x.load(tag, tensor::coord(r, c),
                                  vecops::vec::opt::first(cols - c));
            y.store(tag, tensor::coord(r, c),
                    vecops::vec::mul(v, alpha_vec),
                    vecops::vec::opt::first(cols - c));
          }
        }
        y.commit();  // unconditional: a no-op for direct outputs
      });
}
```

### Why DataAccess exists

DataAccess gives operators a **uniform input/output abstraction**: one
binding takes care of memory↔compute dtype conversion, prologue/epilogue
transform fusion, and layout restructuring (materialization) — the
mechanical groundwork every kernel would otherwise redo by hand. The kernel
body stays a plain vec computation loop while the surrounding machinery
still lowers to near-hand-written-intrinsic performance.

Two honest caveats on that "near-intrinsic" claim:

- It pays off most for operators whose inner loop is ordinary vec-module
  computation. Kernels built on special instruction tiers or accelerators
  (AMX matrix engines, SME streaming tiles) may bypass DataAccess entirely —
  their backend-specific paths (see the `gemm` and `kernel` modules) can
  outperform anything the generic access layer would generate.
- Even for pure vec kernels, DataAccess needs **cooperation** to hit peak:
  tile/block loops sensibly, batch complete vectors, pick a vector Tag that
  matches the data (multi-word tags where profitable), and tune the Policy
  (an accurate `ReadPasses`, the right plan) so the planner resolves to the
  intended path.

---

## Core Concepts

### Tensor: a typed, non-owning view

`Tensor<T, TShape, TStrides, Binding>` pairs one binding with a `Layout`.
For pointer-bound views, pointer cv-qualification is part of the type: a view
built from `const float*` is read-only forever (no `const_cast` anywhere), and
output Specs reject const elements at compile time. `Array<T, N>` is the
fully-dynamic bound spelling (all dimensions `Any`).

For fixed axes, prefer `tensor::size<I>(tensor_or_layout)` and
`tensor::stride<I>(tensor_or_layout)` over indexing `shape()[I]` or
`strides()[I]`. They return the corresponding `meta::Value` (rather than a
degraded `nint_t`), so `auto` preserves the layout's `meta::Const`/
`meta::Dynamic` metadata through adjacent shape-building and operator calls.
The resulting Value also converts implicitly to `nint_t` at runtime-only
boundaries such as pointer arithmetic, workspace byte counts, or a loop trip
count; retain it in an `auto` variable until such a boundary.

The fourth template argument is normally inferred. It selects the binding of
the same `Tensor` abstraction; it does not select a second descriptor class. A
bound tensor uses `PointerBinding<T>` and may access storage. An unbound tensor
uses `UnboundBinding`: it retains the element type, `Shape`, `Strides`, slicing,
transpose, and projection types, but contains only a logical element offset
and has no `data()` or scalar-dereference API. Consequently planning does not
need a separate `TensorDesc` hierarchy that duplicates Tensor and DataAccess
metadata.

```cpp
auto actual = make_tensor(data, make_shape(Dynamic<1, 1, 128>{rows}, cint<64>));
auto pattern = unbind(actual);              // no address
auto tile_pattern = pattern(range(0, 32), reserve);
auto bound_tile = bind(tile_pattern, data); // applies the retained slice offset
```

Use `make_unbound_tensor<T>(layout)` when no bound example exists. Calling
`unbind(tensor)` starts a pattern at logical offset zero; slicing the pattern
then accumulates an offset, and `bind(pattern, base_pointer)` reapplies it.

Input/output Specs use the same model. `unbind(spec)` removes the primary
Tensor address, preserves the transform *type* and safe structural metadata,
and deliberately discards transform state such as captured pointers and
scalars. There are then two execution-time paths:

- `bind(pattern, base_pointer)` and `rebind(pattern, actual_tensor)` are for
  stateless transforms whose type is empty and default-constructible.
- `rebind(pattern, actual_spec)` is the stateful path. It validates the actual
  Spec against the pattern, then retains the actual Spec—including its current
  transform state. State never comes back from the unbound pattern.

Both forms require matching element/compute, rank, transform, projection, and
fact types. Const extents and strides match exactly. A Dynamic extent must
satisfy its alignment/bounds and may not exceed the value stored in the pattern,
which is the prepared capacity. Strides remain exact because they may change
DataAccess lowering and scratch requirements. These are release-mode checks:
accepting an incompatible actual operand would invalidate prepared planning.

A stateful transform's C++ type can itself depend on the type of a captured
Tensor. The pattern and actual call must therefore construct that transform
from views with the same `Const`/`Dynamic` layout types. For a variable tail,
build the planning view with the same bounded `Dynamic` row type and store its
maximum value in the pattern; do not build a `Const<Max>` transform pattern and
expect it to accept a Dynamic tail. `unbind` removes the captured address, not
this type-level contract.

Projection and fact values are retained only when
`OperandPatternMetadata<T>` declares them storage-independent. Empty policy
objects are safe by default; stateful metadata must opt in explicitly. This is
separate from transforms, whose values are always replaced by
`TypeOnlyPatternMetadata<Transform>`.

`TensorOf<Element, Rank>` and `WritableTensorOf<Element, Rank>` express exact
rank/element requirements without introducing operator-local Tensor wrappers.
For Specs, `InputOperand`/`OutputOperand` describe only readability or
writability; `InputOperandOf<Rank>`/`OutputOperandOf<Rank>` add an exact rank.
`OptionalInputOperand` accepts an omitted input at any rank, while
`OptionalInputOperandOf<Rank>` makes the non-omitted branch rank-specific.

### Slicing

`operator()` / `operator[]` accept, per dimension:

| Argument | Effect | Consumes a source dim? |
|---|---|---|
| integer `i` | select index `i`; dimension disappears | yes |
| `reserve` | keep dimension unchanged | yes |
| `new_axis([n])` | insert a new axis of size `n` (default 1), **stride 0** (broadcast) | **no** |
| `range(s, e[, step])` | sub-range; result stride = `stride * step`; negative step reverses; step 0 asserts | yes |
| `ellipsis` | expands to as many `reserve`s as needed (at most one per call) | — |

**All-integer indexing returns an element reference, not a Tensor**; any
mixed form returns a new Tensor whose Shape/Strides *types* are computed at
compile time by `SlicedTraits`. A `range` preserves a statically provable
length (including bounded `Dynamic` ranges when the step sign is known) and
scales the stride type by the step type; `reserve` passes types through.
The omitted `range` step is `Const<1>`, and parameterless `new_axis()` inserts
`Const<1>`, so these common forms do not discard unit-size/unit-stride facts.

```cpp
float data[24] = {};
auto t = make_tensor<3>(data, {2, 3, 4});   // np.arange(24).reshape(2, 3, 4)

float e   = t(0, 1, 2);                     // t[0, 1, 2]        scalar element
auto  m   = t(0, reserve, reserve);         // t[0, :, :]        rank-2 view
auto  s   = t(reserve, range(1, 3), reserve);  // t[:, 1:3, :]    sub-range
auto  stp = t(reserve, range(0, 3, 2), reserve);  // t[:, 0:3:2, :]
auto  rev = t(reserve, range(3, 0, -1), reserve); // t[:, 3:0:-1, :]
auto  b   = t(new_axis(), reserve, reserve, reserve);  // t[None, ...]
auto  b4  = t(new_axis(cint<4>), reserve, reserve, reserve);  // repeat 4x
auto  e1  = t(ellipsis, 2);                 // t[..., 2]
auto  tt  = transpose<0, 2>(t);             // t.transpose(0, 2)
```

Marker-by-marker equivalence with NumPy:

| vecops expression | NumPy equivalent | Notes |
|---|---|---|
| `t(0, 1, 2)` | `t[0, 1, 2]` | returns `T&`, not a 0-d array |
| `t(0, reserve, reserve)` | `t[0, :, :]` | dimension removed / kept |
| `t(range(1, 3), ...)` | `t[1:3, ...]` | runtime bounds stay dynamic; the implicit step is `Const<1>` |
| `t(range(0, 3, 2), ...)` | `t[0:3:2, ...]` | stride scales by step |
| `t(range(3, 0, -1), ...)` | `t[3:0:-1, ...]` | negative step reverses (needs `start > end`) |
| `t(new_axis(), ...)` | `t[None, ...]` / `np.newaxis` | stride-0 broadcast axis; `new_axis(cint<4>)` has no NumPy spelling (repeat) |
| `t(ellipsis, 2)` | `t[..., 2]` | at most one per call |

One deliberate difference from NumPy: all type-level knowledge (`Const`
sizes, typed strides) is preserved through slicing where possible — the
result *type* records what happened, which is what later compile-time
planning feeds on.

### Spec, Policy, and why they are separate

A Spec (`tensor::input<Compute>(tensor, [transform,] [facts...])`,
`tensor::output<Compute>(...)`) carries caller knowledge; a Policy
(`InputAccessPolicy<VectorAxis, ReadPasses, Plan>`,
`OutputAccessPolicy<VectorAxis, Plan>`) carries kernel intent. The split
exists so callers **cannot** select unsafe strategies (lane reordering,
materialization) that only the kernel can validate — e.g. `ReadPasses` must
reflect the algorithm and cannot be inferred from actual calls.

### ComputeType versus the Tensor's element type

`tensor::input<float>(some_bf16_tensor)` mixes two element types on purpose:

- The **template argument is the compute boundary type** — the element type
  of the vectors your kernel computes with (`x.load` returns exactly
  `Vec` of that type, whatever the memory holds).
- The **Tensor's element type is the memory type** — what is actually stored
  on disk/in the buffer.

The two may differ freely. On load, DataAccess converts memory → ComputeType
with `load_convert` semantics (saturate by default, order per policy), runs
the transform pipeline in ComputeType space, and hands the kernel
full-precision-shaped vectors; on store the mirror image applies. The
classic use is narrow storage with wide arithmetic — bf16 or fp16 memory
with `float` compute — so accumulation happens at full precision while the
buffer stays small. Inside the kernel you never see the memory dtype (except
through `raw_data()`, which exposes it deliberately).

### Access plans

| `AccessPlan` | Meaning |
|---|---|
| `automatic` | planner rules below |
| `automatic_deferred` | same choice, but if after-transform materialization is selected, the fill pass is fused into the kernel's first pass via `materialize::populate` (input-only) |
| `direct` / `materialize_before_transform` / `materialize_after_transform` | forced paths |

Automatic planning resolves *outside* the hot loop (in `with_operands`), from
type-level facts only: readless transforms and provably unit-stride vector
axes stay direct; a multi-pass non-contiguous input materializes
after-transform; a non-contiguous output materializes only when another
unit-stride axis can commit a complete rectangular region, otherwise it
scatters directly.

### Transforms

A transform is a **pure, side-effect-free** vector function attached to a
Spec: a *prologue* on inputs (runs after the memory→Compute conversion), an
*epilogue* on outputs (runs before the Compute→memory conversion). Because
they are pure, DataAccess may run them during materialization preparation or
split one logical request into several internal calls without observable
difference.

Two factories and three built-ins:

```cpp
// Coordinate-aware: sees where each lane came from (original full-rank
// coordinates, even through sliced/transposed specs).
auto bias = tensor::make_vec_transform<float, float>(
    [](auto tag, auto x, const auto& ctx) {
      auto y = x;
      for (vecops::nint_t i = 0; i < vecops::vec::size(tag); ++i)
        if (ctx.is_active(i))
          y = vecops::vec::set(tag, y, i,
              vecops::vec::get(tag, x, i) +
              static_cast<float>(ctx.lane_coord(i)[0]));  // e.g. row bias
      return y;
    });

// Elementwise: no context, and a promise that lane order doesn't matter
// (permutation-equivariant) — required to combine with unordered conversion.
auto gelu = tensor::make_elementwise_vec_transform<float, float>(
    [](auto tag, auto x) { return gelu_vec(tag, x); });
```

- `NoTransform` (the default) is **not** an identity function — it is the
  marker that lets DataAccess fuse memory↔compute conversion directly into
  `vec::load_convert`/`store_convert` with no extra stage. Prefer it over
  `identity_transform` for plain operands; dtype conversion never belongs in
  a transform (DataAccess inserts it at the boundaries, keeping
  `adapt_vec_transform` free of conversion wrappers).
- `zeros_transform<float>` declares `reads_input = false`, so the planner
  can eliminate source reads and input workspace entirely.
- Trait reference — semantic promises the planner consumes (and cannot
  verify for arbitrary lambdas, so do not lie):

| Trait | Promise | Unlocks |
|---|---|---|
| `is_elementwise` | output lane `i` depends only on input lane `i` | unordered-region eligibility |
| `permutation_equivariant` | permuting input lanes permutes output lanes | same (implied by `is_elementwise`) |
| `reads_input = false` | result ignores the input values | source-read and workspace elimination |

The wrapper **adapts vector widths**: a callable implementing only some POW2
tags is first tried at the requested tag, then at larger tags via
`vec::bitcast`, then recursively split into halves with `context.subspan()`
keeping lane coordinates correct — a transform never needs to accept every
Tag a kernel may request (an `int8` load feeding a `double` transform may
become several invocations on SVE; each receives the right `subspan`, so
`lane_coord(0)` still names the true first lane). Tail masking is
DataAccess's job: the transform sees a full vector and uses
`ctx.is_active(i)` only when inactive lanes affect the formula.

---

## API Reference

### Coordinates, axes, facts (`AccessOptions.h`)

| Item | Meaning |
|---|---|
| `coord(i...)` → `Coord<Rank>` | fixed-rank signed coordinate (rank deduced) |
| `axis<D>` / `traversal_axis<D>` / `vector_axis<D>` | compile-time axis tags; the last two are semantic aliases distinguishing "the axis being advanced" from "the axis covered by lanes" |
| `ContiguousLaneMapping` / `AffineLaneMapping{stride}` | logical lane → axis-offset maps (elements, not bytes) |
| `assume_aligned<N>` | caller-asserted base-pointer alignment **fact** (checked at Spec construction in debug builds; it does not align anything — kernels still choose `mem::aligned/unaligned`) |

### Specs and their views

| Call | Notes |
|---|---|
| `input<Compute>(tensor[, transform][, facts...])` | input operand descriptor; a raw callable is normalized to a coordinate-aware transform; typed transforms keep their `TIn/TOut` and DataAccess inserts boundary conversions instead of wrapping them |
| `output<Compute>(tensor[, transform][, facts...])` | requires a mutable Tensor (const elements rejected at compile time); the transform is an *epilogue* applied at the store boundary |
| `slice_view<Dim>(spec, i)` / `transpose_view<I, J>(spec)` / `take_leading<N>(spec)` / `take_trailing<N>(spec)` | structure views; a `CoordinateProjection` is composed so coordinate-aware transforms still see original full-rank coordinates |
| `as_input_spec<Compute>(v)` / `as_output_spec<Compute>(v)` | normalize a Tensor-or-Spec into a Spec, forwarding existing Specs unchanged |

### Policies and defaults (`AccessPolicy.h`)

- `InputAccessPolicy<VectorAxis, ReadPasses = 1, Plan = automatic>`,
  `OutputAccessPolicy<VectorAxis, Plan = automatic>` (rejects
  `automatic_deferred`).
- `slice_access_policy_t` / `transpose_access_policy_t`: rebase axis numbers
  after slicing/transposing a view (slicing away the vector axis itself is a
  compile error — the result has no vector-view semantics).
- `access_defaults(opts...)`: stateless per-operand defaults for conversion
  order/value, temporality, packing, alignment (each at most one; defaults
  `Ordered/Saturate/Temporal/Packed/Unaligned`). Attach at
  `tensor::operand(...)`, not to the Policy: eager preparation and deferred
  commit run **outside** individual hot-loop calls, where no call-site option
  exists.

### Binding

| Entry | Purpose |
|---|---|
| `make_unbound_tensor<T>(layout_or_shape)` / `unbind(tensor_or_spec)` | create an address-free planning pattern using the same Tensor/Spec layout types |
| `bind(tensor_pattern, base)` | restore a Tensor pointer and apply any element offset accumulated by slicing the pattern |
| `bind(spec_pattern, base)` | pointer-bind a stateless Spec pattern; unavailable for stateful transforms |
| `rebind(spec_pattern, actual_tensor)` | validate capacity/layout and recreate a stateless bound Spec over the actual Tensor |
| `rebind(spec_pattern, actual_spec)` | validate the full typed contract while retaining the actual Spec's stateful transform value |
| `tensor::operand(spec, policy[, defaults])` | deferred binding record; the referenced Spec must outlive the `with_operands` call |
| `kernel::with_operands(workspace_or_scope, bindings..., fn)` (1–4 operands) / `kernel::with_operand_tuple(...)` | resolve dynamic choices, prepare inputs, hand **lvalue** sessions to `fn`; materializing groups run under a workspace mark that is rewound after the sessions die; all-direct groups never touch workspace state. The callback must not mutate the passed `WorkspaceView` |
| `tensor::required_workspace(spec, policy)` | byte estimate for planning (must use the same Spec/Layout/Policy as the later binding; direct and readless inputs are 0) |
| `tensor::bind(spec, policy, workspace)` | low-level direct-only binding (asserts the resolved plan is direct); prefer `with_operands` |

### Sessions: load / store

`x.load(tag, coord[, axis<D>][, options...])` returns exactly the requested
logical Compute Tag regardless of memory dtype or transform widths;
`y.store(tag, coord, value[, axis<D>][, options...])` writes it back. The
vector axis defaults to `Rank - 1`. Options (all compile-time validated):

- **active (≤1)**: `opt::unmasked` (default) / `opt::first(n)` /
  `opt::masked(m)` — masks use the **requested logical Tag**. `first(n)` is
  the preferred tail form; DataAccess derives every downstream mask from it.
- **addressing (≤1)**: `strided(s)` / `indexed(idx)` — indices are
  **logical element offsets, scale 0 only**; byte scales are rejected and
  Layout multiplies by the physical stride exactly once.
- **population (≤1, loads only)**: `opt::zero` / `opt::merge(vector|scalar)`
  fill inactive lanes of the *logical result* (stores never touch inactive
  addresses; duplicate active indexed lanes keep the backend's scatter
  conflict semantics).
- **default-class overrides (≤1 each)**: conversion order/value,
  temporality, packing (stores only), alignment — merged over the binding's
  `access_defaults` rather than passed down raw.

For exact point access, `x.load_scalar(coord[, conversion-options...])`
returns one `ComputeType`, while
`y.store_scalar(coord, value[, conversion-options...])` stores one. These
operations use Layout to compute the address and then issue direct scalar
pointer traffic; they do not construct a vector. Only conversion order/value
options are accepted (`materialize::populate` is additionally accepted by
scalar input loads). Lane masks, strided/indexed addressing, inactive
population, alignment, packing, and temporality are vector-only call options.
Operand-level memory defaults may still be shared with vector calls but have no
effect on scalar traffic. Conversion order likewise cannot reorder one value.

Scalar access is available only when the active boundary transform is
`NoTransform`, `IdentityVecTransform`, or `ZeroVecTransform`; a custom vector
transform makes the scalar member unavailable at compile time. `NoTransform`
performs one MemoryType↔ComputeType conversion. Identity retains its explicit
intermediate type and therefore performs both boundary conversions. A zero
input returns `ComputeType{}` without reading Tensor memory, and a zero output
stores a converted zero regardless of the supplied value.

Direct sessions expose `raw_data()`, `raw_strides()`, `spec()`, `policy()`.

A coordinate-aware `VecTransform` receives the same activity in its
`TransformContext`. For unmasked and prefix-active accesses,
`context.active_option(tag)` returns `opt::unmasked` or the subspan-adjusted
`opt::first(n)` respectively. Forward this option to auxiliary vector loads in
an epilogue instead of recomputing a tail from tensor extents. Arbitrary masked
contexts expose `is_active(lane)` but intentionally do not pretend to be a
contiguous prefix.

Matrix/tile kernels can query
`preferred_memory_access_power_v<Access>`. It is the number of `Twice<Tag>`
steps needed for a native-word `ComputeType` vector to cover at least one full
word of the actual `MemoryElement`; a wider memory element returns zero because
the compute input is already a complete word. The transform's declared output
type is deliberately irrelevant—the final boundary is always MemoryElement.
The AMX C path consumes this hint when adjacent accumulator tiles and a
physically contiguous row/column axis exist. Its automatic policy concatenates
the value and active predicate before invoking one wide transform. SME's
automatic policy currently keeps transforms and stores narrow on the tested
64-byte target: both the P1-transform path and the two-P0-transform/one-store
path lost to the smaller narrow loop with BiSheng 5.1. The coalesced primitive
still exists for backends or future compilers that benefit: it gives each P0
transform its own context and then concatenates only the results for one packed
converting store. A partial second tile becomes one combined prefix store.

Operator code normally leaves this automatic. An expert can override the
backend decision by wrapping only the transform, without naming a vector tag:

```cpp
auto epilogue = tensor::with_transform_store_mode<
    tensor::TransformStoreMode::coalesced>(my_epilogue);
```

The other explicit modes are `narrow` and `wide_transform`; an explicit mode
takes priority over the AMX/SME default. A kernel must still cap the requested
power by its resident tile shape and must not group across a non-contiguous
physical axis merely to obtain a wider access.

### Materialization: plans, populate, and commit

The plans differ in *what the workspace buffer holds* and *when it is
filled/written back*:

| Plan (input) | Buffer holds | Filled |
|---|---|---|
| `direct` | — (no buffer; accesses hit the Tensor itself) | — |
| `materialize_before_transform` | raw **MemoryElement** values | during preparation, before the kernel loop; the transform still runs on every load |
| `materialize_after_transform` | **ComputeType** values, transform already applied | during preparation; kernel loads are plain contiguous Compute reads |
| `automatic_deferred` | same as after-transform (ComputeType) | **not** during preparation — fused into the kernel's first pass via `materialize::populate` |

| Plan (output) | Buffer holds | `commit()` does |
|---|---|---|
| `direct` | — | no-op |
| `materialize_before_transform` | **ComputeType** results | replays epilogue + conversion from the buffer into the Tensor (Transpose2D-accelerated when the requested vector axis is not unit-stride and another provably unit-stride axis exists; degenerate layouts with multiple unit strides preserve the requested logical axis) |
| `materialize_after_transform` | **MemoryElement** results (epilogue already ran in the loop) | only remaps/copies the buffer into the Tensor |

Materialized output auxiliaries are zero-initialized because `commit()` writes
their complete extent and masked stores may intentionally leave elements
untouched. Other Workspace allocations remain raw and uninitialized.

So "before/after" names the transform stage the buffer sits at: an input
cached *before* the transform re-pays the transform per read but saves raw
bytes; an input cached *after* pays the transform once in preparation and
makes every kernel load a cheap contiguous read. The planner picks
automatically from type-level facts (multi-pass + non-contiguous →
after-transform, etc.); force a plan only with measured reason.

Canonical ComputeType storage also exposes `raw_data()`/`raw_strides()` when
its hot representation is `NoTransform` and memory type equals ComputeType.
This includes eager after-transform inputs, before-transform output auxiliary
storage, and their borrowed structural views. Matrix backends may therefore
load/store that already-prepared storage directly instead of copying it into a
second tile scratch buffer. Deferred inputs deliberately do not expose this
surface until their populate/reuse protocol has completed; original output
conversion and commit ownership remain on the materialized parent session.

**`materialize::populate`** is what makes `automatic_deferred` interesting:
instead of a separate preparation pass, the *kernel's own first traversal*
fills the canonical Compute buffer. Mark the loads of that first visit with
`tensor::materialize::populate` — the value is cached *and* returned; later
visits drop the marker and read the cache:

```cpp
using XPolicy = tensor::InputAccessPolicy<
    1, /*ReadPasses=*/2, tensor::AccessPlan::automatic_deferred>;

kernel::with_operands(workspace, tensor::operand(x_spec, XPolicy{}),
    [&](auto& x) {
      auto tag = vecops::vec::ScalableTag<float>{};
      // Pass 1 (populate): fills the canonical Compute buffer as a
      // side effect of the load.
      for (vecops::nint_t c = 0; c < cols; c += vecops::vec::size(tag))
        consume(x.load(tag, tensor::coord(r, c), vecops::vec::opt::first(cols - c),
                       tensor::materialize::populate));
      // Pass 2 (reuse): plain reads of the cached Compute values —
      // must be ordered; conversion order/value options are rejected here.
      for (vecops::nint_t c = 0; c < cols; c += vecops::vec::size(tag))
        consume(x.load(tag, tensor::coord(r, c), vecops::vec::opt::first(cols - c)));
    });
```

The kernel promises a complete populate pass before any unmarked reuse;
debug builds detect reuse-before-populate and populate-after-reuse, but keep
no per-element coverage bitmap. `populate` is a no-op on direct and eager
materialized inputs.

**Commit discipline.** `commit()` is idempotent and mandatory on materialized
outputs — destructors assert `"materialized output session was destroyed
without commit()"` and never write back implicitly. Call it unconditionally
(direct outputs no-op, so kernels need not branch). Sessions are move-only;
**moving transfers commit ownership**; slices of a session are
`BorrowedDataAccess` views with **no `commit()`** — the parent remains the
sole commit authority.

### Unordered regions

`tensor::with_unordered_access(accesses..., fn)` decides **for the whole
group, at compile time** whether every bound access may load/store with
`cvt::unordered`; `fn` receives order-injecting proxies (all-unordered or
all-ordered — never mixed, which is a compile error across a group). The
`ordered`/`unordered` semantics here are exactly those of
`vec::cvt::Ordered`/`Unordered` — the stable, compositional native-permutation
contract defined in `vecops/vec/Options.h` and motivated in
[Vec.md — Why so many conversion variants?](Vec.md#why-so-many-conversion-variants);
an unordered region simply licenses that cheaper native conversion layout
for a whole group at once, instead of per call.

A group qualifies only when: no canonical Compute buffer is involved,
transforms are permutation-equivariant, storage widths are compatible, and
all accesses share the same ComputeType and storage bytes. Within such a
region, sliced views reject per-call conversion-order options — the region
owns that knob.

---

## Implementation Notes

- **Planning is type-driven.** Unit-stride and continuity decisions come from
  meta Value types only; runtime stride values never add branches.
- **Chunked lowering.** When rebinding the requested Tag to a transform's
  `TIn`/`TOut` or to the memory dtype exceeds backend POW2 limits, DataAccess
  recursively partitions the logical lanes, invokes memory/transform per
  chunk (recombining with `vec::concat`, no per-lane scalar traffic on
  fixed-width backends), and hands each chunk a `context.subspan()` so
  lane coordinates stay correct. On sizeless SVE the transform chunk layer
  uses a per-lane reference path by backend selection, not a runtime branch.
- **Requests borrow, never own** vector values (masks, index vectors) — the
  same sizeless-SVE rule as `vec::opt::masked`; keep them alive across the
  call.
- **Workspace discipline.** Materializing groups run under `mark()` /
  `rewind()` around the callback; all-direct groups are mark-free.

---

## Pitfalls and FAQ

**Tensor is non-owning** and so is everything derived from it (Specs, direct
sessions, borrowed views). The backing memory — and any referenced Spec —
must outlive every view.

**All-integer slicing returns an element, not a Tensor.** `t(0, 1, 2)` is a
`float&`; insert a `reserve` if you wanted a view.

**`new_axis` does not consume a dimension** (stride 0 broadcast axis);
`reserve`/`range`/integers do. `range` step 0 asserts; negative steps
reverse but require `start > end`.

**Pointer mutability is frozen at construction.** Build from `const T*` and
no downstream cast will ever make it writable; outputs built that way fail
at compile time, not at runtime.

**`ReadPasses` and vector-axis choices are algorithm facts.** Wrong values
silently change the planner's cost decisions — they cannot be inferred, so
review them when reusing policies across kernels.

**`automatic_deferred` is a promise**: every element needed later must be
first visited by a `materialize::populate` load. Populate must be ordered;
reuse cannot reinterpret the cached Compute values.

**Commit discipline.** Call `y.commit()` unconditionally on outputs (safe —
direct outputs no-op); never drop a materialized output session without
committing (destructor asserts); never expect a borrowed slice to commit.

**`indexed` is element-scaled here.** Unlike raw `vec::opt::indexed`, tensor
indices are logical element offsets (scale 0 only) — Layout applies the
physical stride exactly once. Do not pre-multiply.

**`with_unordered_access` is all-or-nothing.** Mixing ordered and unordered
policies in one region is a compile error; canonical materialized buffers
and non-equivariant transforms force the whole region ordered.

**Layout fast paths need type-level facts.** A `Dynamic` stride that happens
to equal 1 at runtime keeps the general path — by design (no data-dependent
branches); spell strides as `cint<1>` / `dyn<...>` when you can prove them.
