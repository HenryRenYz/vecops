# The Layout Vocabulary

`vecops::tensor::Layout` (in `vecops/tensor/Layout.h`) maps an N-dimensional
logical coordinate to an element offset:

```
offset(i0, ..., iN) = i0 * stride(0) + ... + iN * stride(N)
```

Every shape and stride entry is a [meta](Meta.md) Value type —
`Const<N>` (compile-time), `Dynamic<A, Lo, Hi>` (constrained runtime), or
`Any` — so compile-time dimensions stay visible to optimizers while dynamic
dimensions carry their values at runtime. Strides are measured in
**elements**, never bytes. The data pointer and element type belong to
`Tensor`, not to a `Layout`.

```cpp
#include "vecops/tensor/Layout.h"
using namespace vecops::tensor;

auto sh = make_shape(cint<2>, cint<3>);         // Shape<Const<2>, Const<3>>
auto st = make_strides(cint<3>, cint<1>);       // Strides<Const<3>, Const<1>>
auto layout = make_layout(sh, st);
auto offset = offset_at(layout, 1, 2);          // 1*3 + 2*1 == 5

auto dynamic = make_layout(make_shape(8, 128)); // row-major strides inferred
auto transposed = transpose<1, 0>(layout);      // type info preserved
```

---

## Core Concepts

### ArrayMeta, Shape, Strides

`ArrayMeta<Is...>` is the fixed-rank base: one meta Value type per
dimension, values stored in a compressed `meta::details::PackedStorage`
(`Const` entries cost no storage). `Shape<Is...>` adds the constraint that
every dimension is non-negative (asserted at construction); `Strides<Is...>`
imposes no sign constraint — zero and negative strides are legal.
Explicit shape-contract access (`shape_extent<I>` /
`shape_extent_type_t<I,L>`) reifies that invariant: an otherwise unbounded
runtime entry is exposed as `Dynamic<A,0,Hi>`. `size<I>` / `size_type_t<I,L>`
keep the exact stored metadata type for source and code-generation
compatibility.

Factories `make_shape` / `make_strides` accept any mix of typed Values and
bare integers (bare integers promote to `Any`, exactly like meta arithmetic).

### Logical versus physical properties

Shape describes which coordinates are *logically valid*; strides describe how
a coordinate change moves the *physical* offset. Consequently:

- stride 1 → physically contiguous along that axis;
- stride 0 → broadcasting/aliasing along that axis;
- negative stride → walks memory backwards; the Tensor's base pointer must
  already refer to logical coordinate zero on that axis;
- arbitrary strides may overlap — Layout provides no ownership or
  non-aliasing guarantee, and never proves that the backing allocation covers
  every reachable offset.

### Typed transformations

Layout transformations are **compile-time type-level operations**: the result
type is computed from the input type, preserving per-dimension Const /
Dynamic information. `remove<I>` erases a dimension from the type;
`insert<I>(value)` adds one (with the Value type of the inserted argument);
`set<I>(value)` replaces one; `transpose<I, J>` swaps two. Runtime indices in
`transpose(layout, i, j)` cannot steer template parameters, so exact per-axis
types cannot survive. Its result instead preserves the guarantees shared by
all possible source axes: gcd alignment and the union of bounds; Shape entries
also retain non-negativity. If every axis denotes the same singleton value
(`Const` or singleton `Dynamic`), the result stays `Const`.

### Lenient matching and conversions

`Layout` converts **implicitly** toward equal-or-more-lenient metadata
(`Const<N>` → compatible `Dynamic` → `Any`, per dimension, for both Shape and
Strides). Tightening directions are explicit via `layout.as<TShape2,
TStrides2>()`, whose target-type constructors assert that the runtime values
actually satisfy the stricter types.

`is_lenient_v<Meta, Patterns...>` performs static pattern matching on Shape /
Strides *types*: rank must match, and each pattern is either the `meta::any`
wildcard (accepts every Value type) or a Value type accepted per
`IsMoreLenientValue`. This is the vocabulary consumers use to declare which
layout shapes they are specialized for.

---

## API Reference

### Construction

| Call | Result |
|---|---|
| `make_shape(vs...)` / `make_strides(vs...)` | typed Shape/Strides; bare ints → `Any` |
| `make_layout(shape, strides)` | Layout pairing the two (ranks must match) |
| `make_layout(shape)` | Strides **inferred as row-major contiguous, at the type level**: the stride type of dim `i` is the meta-product of the size types of dims `i+1..N-1`. `Shape<Const<2>, Any, Const<4>, Const<3>>` yields `Strides<Dynamic<4>, Const<12>, Const<3>, Const<1>>` — the last stride is statically `Const<1>` even when sizes are dynamic |
| `make_layout<N>({s...}, {t...})` / `make_layout<N>({s...})` | initializer-list forms; all dimensions `Any` (the shape-only form still infers typed row-major strides over those `Any`s) |

### Queries

| Call | Meaning |
|---|---|
| `get<I>(meta)` / `meta.get<I>()` / `meta[i]` | dimension value (compile-time for `Const` entries) |
| `is_const<I>` / `is_runtime<I>` | per-dimension staticness |
| `size<I>(layout)` / `stride<I>(layout)` | fixed-axis typed accessors; return the corresponding `meta::Value` and preserve `Const`/`Dynamic` metadata |
| `shape_extent<I>(layout)` / `tensor.shape_extent<I>()` | fixed-axis size with Shape's non-negative contract reified in its `Dynamic` lower bound |
| `layout.shape()[i]` / `layout.strides()[i]` | runtime-axis accessors; return `nint_t` after metadata erasure |
| `numel(layout)` | product of all dimension sizes as a raw `nint_t` compatibility boundary |
| `numel_value(layout)` / `tensor.numel_value()` | typed shape product preserving `Const`/`Dynamic` alignment and bounds |
| `offset_at(layout, i0, ..., iN)` | linear offset; **bounds-asserted** per dimension (debug) |
| `size_type_t<I, L>` / `shape_extent_type_t<I,L>` / `stride_type_t<I, L>` / `numel_type_t<L>` | stored size type / non-negative size type / stride type / typed element-count type |
| `is_array_meta_v` / `is_shape_v` / `is_strides_v` / `is_layout_v` (+ `...Like` concepts) | type classification |

### Dimension transformations

| Call | Effect |
|---|---|
| `remove<I>(meta)` / `remove<I>(layout)` | drop dimension `I` (rank −1) |
| `set<I>(meta, v)` / `set<I>(layout, size, stride)` | replace dimension `I` with the argument's Value type |
| `insert<I>(meta, v)` / `insert<I>(layout, size, stride)` | insert before position `I` (`I == Ndim` appends) |
| `take_leading<N>` / `take_trailing<N>` | keep the first/last N dimensions, discarding the others at coordinate 0 |
| `transpose<I, J>(layout)` | compile-time swap, full type information preserved |
| `transpose(layout, i, j)` | runtime swap; each axis retains constraints common to every source axis — prefer the template form for exact per-axis types |

### Continuity

| Call | Meaning |
|---|---|
| `is_ct_last_contiguous_v<L, N>` / `is_ct_contiguous_v<L>` | compile-time: are the last N (all) dimensions row-major contiguous, judged from exact singleton entries (`Const<N>` or `Dynamic<A,N,N>`); `false` is conservative ("not provable") |
| `is_last_contiguous<N>(layout)` / `is_contiguous(layout)` | runtime check; folds to a compile-time `true` when the static trait already proves it |

### Printing

`operator<<` prints Shape/Strides as `(2!, 128@4[0,512])` — `!` marks
`Const`, `@A` an alignment, `[lo,hi]` active bounds — and a Layout as
`Layout(s=(...), st=(...))`.

---

## Pitfalls and FAQ

**Shape entries must be non-negative; strides are unrestricted.** A negative
shape triggers a construction assertion; a negative stride is a legitimate
reversed view.

**Elements, not bytes.** All offsets and strides are element counts;
multiply by `sizeof(T)` only at a raw-byte boundary.

**Bare integers become `Any`**, erasing compile-time knowledge — the same
rule as [meta](Meta.md). `make_shape(4, 128)` cannot prove contiguity or
tail elimination that `make_shape(cint<4>, dyn<4>(128))` can.

**Continuity helpers describe the order they name.** `is_contiguous` tests
*row-major* contiguity; a transposed layout can be dense (no gaps) yet not
row-major contiguous.

**Transformations change the C++ type.** `remove`, `set`, and `insert`
return objects of different template types; code generic over layouts should
pass them through `auto`.

**Runtime strides never trigger hidden fast paths.** Continuity and planning
decisions in this library are made from *type-level* guarantees only
(`Const<1>` etc.); a `Dynamic` stride whose runtime value happens to be 1
selects the general path, deliberately — no per-call branch on data values.

**A Layout does not validate memory coverage.** Especially with negative,
zero, or overlapping strides, proving that the backing allocation covers
every reachable offset is the caller's job.
