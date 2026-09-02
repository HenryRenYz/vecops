# The `vec` Module

`vecops::vec` is a header-only SIMD abstraction layer. You describe *what*
vector you want with a **Tag**, obtain backend-selected **Vec/Mask**
representations from it, and apply operations written as small callable
objects (`vec::add`, `vec::load`, ...). Everything — backend selection,
masking, inactive-lane population, memory addressing — is resolved at compile
time; there is no virtual dispatch and no runtime cost over hand-written
intrinsics in the supported fast paths.

```cpp
#include "vecops/vec/Vec.h"  // umbrella; individual headers are self-sufficient too

namespace vec = vecops::vec;

void axpy(float* dst, const float* src, float alpha, vecops::nint_t n) {
  vec::ScalableTag<float> t;        // one native word of f32
  for (vecops::nint_t i = 0; i < n; i += vec::size(t)) {
    const auto tail = vec::mwhilelt(t, i, n);      // tail mask
    const auto x = vec::load(t, src + i, vec::opt::masked(tail));
    const auto y = vec::load(t, dst + i, vec::opt::masked(tail));
    const auto r = vec::fmadd(x, vec::fill(t, alpha), y);
    vec::store(t, dst + i, r, vec::opt::masked(tail));
  }
}
```

The same source compiles for the scalar, x86 (SSE–AVX-512) and SVE backends;
only the compile target changes.

### Positioning

`vec` occupies the same niche as `std::simd`, Google Highway, `xsimd`, and
PyTorch ATen's CPU vector wrappers: a portable SIMD abstraction over native
vector registers. Its differentiator is the combination of three goals:

1. **Near-intrinsic performance.** The architecture — compile-time dispatch,
   exposed layout choices, per-word batching — is deliberately engineered so
   that code written against the abstraction can perform on par with
   hand-written intrinsics: the library hides *which* instructions are used
   without adding work to hide *that* they differ.
2. **Architecture details stay hidden.** Backend selection, multi-word
   decomposition, register typing, and ISA-specific idioms are compile-time
   machinery the caller never touches.
3. **Maximum type coverage.** All 12 element types — including `bfloat16_t`,
   `float16_t`, and sub-word vectors — are first-class on every backend, with
   defined fallback semantics where an ISA lacks native support.

The price of goal 1 is that layout-affecting decisions (conversion lane
order, narrow-store packing, ...) are exposed as explicit options rather than
silently papered over — see [Options](#options). This is why, compared with
traditional SIMD libraries, `vec` covers noticeably more data types and more
instruction variants per operation (masked / aligned / indexed / estimate
forms, conversion layout phases, ...).

The Tag system is inspired by Google Highway's tag types.

---

## Core Concepts

### Elements and Tags

An **Element** is one of the 12 supported scalar types: `bfloat16_t`,
`float16_t`, `float32_t`, `float64_t`, `(u)int8/16/32/64_t`.

A **Tag** is a pure compile-time descriptor of "element type × logical lane
count". It carries no data.

| Spelling | Meaning |
|---|---|
| `FixedTag<T, N>` | Exactly `N` logical lanes. `N` must be a positive power of two. It need not fill a hardware word (sub-word Tags are fine). |
| `ScalableTag<T, P>` | A target-dependent vector covering `2^P` native words. Negative `P` describes a sub-word fraction. On sizeless SVE the lane count is a runtime value; on fixed-width targets it is a compile-time constant. |
| `ElementOf<Tag>` | The element type of a Tag. |
| `Rebind<U, Tag>` | Same logical lane count, element type changed to `U` (ScalePower adjusts automatically for scalable Tags). |
| `ViewAs<U, Tag>` | Same byte width, element type changed to `U`. |
| `Half<Tag>` / `Twice<Tag>` | Half / double the logical lanes. |
| `IndexTag<Tag>` | A Tag of signed integers used to index lanes of `Tag` (`f32 → i32`, `u32 → i32`, integers map to themselves). |

#### What `N` and `P` mean, and their ranges

**`FixedTag<T, N>` — `N` is the exact logical lane count.** It must be a
positive power of two (enforced by `static_assert`); any other value,
including negatives, is rejected at compile time. `N` need not fill a
hardware word: a sub-word Tag (say 8 f32 lanes on a 16-lane AVX-512 word)
still occupies one physical register but reports `size(tag) == N` and keeps
its logical semantics. On sizeless (VLA) SVE builds, `FixedTag` cannot be
materialized at all — the `static_assert` there points you to a
vector-length-specific build or `ScalableTag`.

**`ScalableTag<T, P>` — `P` is the base-2 exponent of the native word
count.** `P >= 0` gives `native_word_size << P` lanes; `P < 0` describes a
sub-word fraction (`word_lanes >> -P` lanes, asserted to keep at least one
lane). On sizeless SVE (`VEC_WIDTH == -1` marks the VLA mode) the lane count
is a runtime value; on fixed-width backends it is a compile-time constant.

`P` is bounded on both ends by what the backend can *represent well*, not by
an arbitrary policy:

- **Upper bound.** VLA SVE tops out at `P <= 2` (hard `static_assert`):
  sizeless words can only be grouped into the architectural tuples
  `sv<T>x2_t` / `sv<T>x3_t` / `sv<T>x4_t`, i.e. at most four words. (On
  Clang, multi-word masks in this mode have a known inlining limitation —
  see the pitfall note on SVE predicate tuples.) x86 and
  SVE vector-length-specific builds allow `P <= 5`: their words are sized
  types, so multi-word vectors are ordinary arrays with no tuple limit —
  the bound comes from the register file. The target machines expose 32
  vector registers, and `2^5 = 32` words already consume all of them; any
  wider Tag would force register spilling in every kernel that uses it.
  Since `vec` is designed for hand-intrinsic-level performance, that regime
  is excluded rather than supported.
- **Lower bound.** `VEC_HW_MIN_POW`: −2 on x86 (down to a quarter word) and
  0 on SVE/NEON, where one complete word is the smallest addressable unit;
  sub-word `P < 0` additionally requires the shrunken lane count to stay
  above zero.

The authoritative macro values per capability tier live in
`vec/Capabilities.h` (`VEC_WIDTH`, `MAX_VEC_WIDTH`, `VEC_MAX_POW`,
`VEC_HW_MIN_POW`).

Related queries (all take `Tag{}` and are `constexpr`):

| Query | Meaning |
|---|---|
| `size(tag)` | Exact logical lane count. |
| `native_word_size(tag)` | Physical lanes in one backend word. |
| `num_words(tag)` | Number of physical backend words the Tag spans (multi-word Tags exist, e.g. `ScalableTag<f32, 2>` on AVX-512 is two registers). |
| `memory_alignment(tag)` | Natural alignment in bytes for `mem::aligned` accesses. |
| `is_fixed_tag_v` / `is_scalable_tag_v` / `fixed_lanes_v` / `scale_power_v` | Compile-time Tag introspection. |
| `is_runtime_size_v<Tag>` | True only for scalable Tags whose size depends on runtime SVE vector length. |

### Vec and Mask Representations

`Vec<Tag>` and `Mask<Tag>` are aliases to whatever the active backend stores
for that Tag (`__m512` and `__mmask16` on AVX-512, sizeless `svfloat32_t` /
`svbool_t` on SVE, small arrays on scalar). Multi-word Tags are represented as
a short array (or SVE tuple); the library automatically batches elementwise
operations word by word.

- `get_word<I>(tag, v)` / `set_word` / `from_words(tag, w0, w1, ...)`
  (plus `Mask` counterparts and runtime-ordinal overloads) expose the physical
  words when you need them.
- `VecToTag<V>` maps a representation back to its canonical Tag. This mapping
  is **lossy for sub-word Tags** — a representation does not remember that it
  was created as, say, a quarter word. Pass the original Tag explicitly when
  sub-word semantics matter.

### Call Forms

Every operation object supports up to four forms (growing surface):

```cpp
vec::add(tag, a, b);                              // 1. explicit Tag
vec::add(a, b);                                   // 2. Tag-inferred (from operands)
vec::add(a, b, vec::opt::masked(m));              // 3. option pack (tag optional)
vec::add(tag, a, b, request);                     // 4. pre-resolved OpRequest
```

Generators (`fill`, `zeros`, `mwhilelt`, ...) have no operands to infer from,
so they always take the Tag first. `bitcast` and `convert` need both source
and destination Tags explicitly.

**Which operations get a tag-inferred form.** The criterion is
*lane-count-insensitivity*: the inferred Tag (`VecToTag<V>`, the full
physical Tag of the representation) must produce exactly the same backend
implementation as the original Tag. This holds for arithmetic, bit, and
comparison operations — each lane is computed independently, so a sub-word
origin is indistinguishable from a full-word one.

It deliberately does **not** hold for memory operations, reductions, and the
width-changing rearrangements (`lower`/`upper`/`concat`/`even`/`odd`/...):
there the logical lane count and its mapping onto physical words *is* the
semantics — `load` must know how many elements to access, `shuf` indices are
word-relative, a sub-word Tag selects a different backend path than its
inferred full-word Tag. Inferring would silently forward to the wrong
implementation, so these operations require the Tag explicitly.

### Options

Options are tag types passed after the operands, in any order. There are
three namespaces:

| Namespace | Options | Used by |
|---|---|---|
| `vec::opt` | `masked(m)`, `unmasked`, `first(n)`, `zero`, `merge(vec_or_scalar)` (three overloads: vector / mask / scalar), `wrap`, `saturate`, `indexed(idx[, scale])`, `strided(s)`, `lanes<I...>` | nearly everything; `wrap`/`saturate` are integer `add/sub` overflow policies |
| `vec::opt::math` | `strict` / `fast` / `estimate` / `accuracy<A>` | complex math functions only (see below) |
| `vec::cvt` | `ordered`, `unordered`, `lane<P>`, `saturate`, `wrap` | `convert`, `load_convert`, `store_convert` |
| `vec::mem` | `aligned`, `unaligned`, `temporal`, `non_temporal`, `packed`, `split`, prefetch hints (`prefetch_l1/l2/keep/stream/read/write`) | memory family |

**Active options** — which lanes participate:

- `opt::masked(m)` — lanes selected by mask `m`.
- `opt::unmasked` — every logical lane (the explicit spelling of the default).
- `opt::first(n)` — lanes `[0, clamp(n, 0, size(tag)))`; the idiomatic tail
  form for memory operations.

**Integer overflow options** — `opt::wrap` explicitly selects the default
modulo-2^N behavior of integer `add`/`sub`; `opt::saturate` instead evaluates
the mathematical result and clamps it to the element type's range. At most one
is accepted, floating Tags reject both, and an overflow-policy-only call is
unmasked. They compose normally with `opt::masked` and inactive-lane options.

**Population options** — what inactive *result* lanes contain:

- `opt::zero` — element zero.
- `opt::merge(v)` / `opt::merge(mask)` / `opt::merge(scalar)` — inactive
  lanes read the supplied vector / mask / broadcast scalar.

**Addressing options** (memory family only):

- `opt::indexed(idx[, scale])` — per-lane element offsets from an i32/i64
  index vector (gather/scatter). An explicit scale multiplies the index
  in **bytes**; scale 0 means "one memory element".
- `opt::strided(s)` — element offsets `lane * s`; the linear special case of
  indexed. `s` may be a runtime `nint_t`, `meta::Const<N>`, or a bounded
  `meta::Dynamic` — the static forms let backends pick address-generation
  idioms without runtime multiplies.

**Memory hints** (`mem::`):

- `mem::unaligned` (default) / `mem::aligned` — `aligned` *promises*
  `memory_alignment(tag)`-byte alignment of the base address (asserted in
  debug builds) and licenses the backend to use aligned-only instructions;
  `unaligned` makes no promise.
- `mem::temporal` (default) / `mem::non_temporal` — temporality is a *hint*:
  backends without a streaming store silently fall back to an ordinary one.
- `mem::packed` (default) / `mem::split` — layout of narrowing
  `store_convert` only: `packed` narrows all source words and emits one wide
  store (avoids partial cache-line writes); `split` stores each source word's
  narrowed result independently. See [Converting Memory](#converting-memory-vecconversionmemoryh).
- `mem::prefetch_*` — the three orthogonal prefetch dimensions
  (locality / temporality / intent); see [Prefetch](#prefetch-vecprefetchh).

**Math accuracy options** (`opt::math::`) apply **only to complex math
functions** — operations implemented as instruction *sequences* where
different algorithms trade accuracy for speed (`exp`, and estimate-tier
entries like `rcp`/`rsqrt`). Lane-wise primitive instructions (`add`, `mul`,
`fmadd`, ...) have exactly one hardware behavior and accept no accuracy
option; passing one is a compile error. See [Math](#math-vecmathh) for the
per-level error contracts.

Rules enforced at compile time:

- Exactly one *active* option: one of `masked` / `unmasked` / `first`.
  A call with no options is an ordinary unmasked call; as soon as you pass
  any option you must state which lanes are active.
- At most one *population* option for inactive result lanes: `zero` or one
  `merge`. Mutually exclusive with each other.
- `indexed` and `strided` are mutually exclusive and cannot be combined with
  alignment options.

#### Why so many conversion variants?

Most SIMD libraries expose one, maybe two conversion flavors (usually just
"convert", sometimes plus a saturating integer variant) and hide the lane
layout entirely. That hiding has a cost: a *widening or narrowing conversion
changes the data volume*, so the ISA's native convert instructions each fix
their own answer to "which input lanes land where" — e.g. x86
`vcvtneps2bf16` writes the narrowed result into the low half of the
destination, SVE narrowing instructions split even/odd input lanes into
separate registers. An abstraction that promises one fixed layout must
insert compensating shuffles on some backends — exactly the overhead this
library refuses to add.

`vec` instead makes the layout an explicit, orthogonal, *semantic* choice:

- `cvt::ordered` — the reference layout (`out[i] = in[i]` for equal lane
  shapes). Use it when lane positions matter downstream.
- `cvt::unordered` — allows the backend's native layout, requiring only that
  it be **stable and compositional**: `A→B→C` has the same lane provenance
  as `A→C`, and `A→B→A` restores positions. This lets each backend use its
  convert instructions with zero compensating shuffles. The guarantee is
  about ordering, never about numeric round-trips.
- `cvt::lane<P>` — the phase-selecting form for width conversion between
  equal-byte Tags, mirroring how hardware widen/narrow instruction pairs
  actually process lanes: a 2× conversion is consumed as two phases
  (`P = 0, 1`), so kernels can take one phase per iteration instead of
  recombining halves.

Combined with the `cvt::saturate` / `cvt::wrap` value policy (how
out-of-range narrowing results are formed — clamp, or keep the low bits),
this covers the real instruction-variant matrix instead of collapsing it to
a single "portable" behavior that no backend executes natively. Higher-level
consumers (e.g. the tensor layer's `with_unordered_access`) select the
cheapest legal layout for a whole region, so end-user code rarely touches
`unordered` directly.

### Inactive-Lane Defaults Differ by Family

When masked, lanes that are not active must still get *some* value. The
default differs by family — this is the single most common source of
surprises:

| Family | Inactive lanes default to | Override with |
|---|---|---|
| Arithmetic, bit, FMA | **first operand's lane** (`a[i]`, preserved) | `opt::zero`, `opt::merge(...)` |
| Comparison | **false** (`opt::zero` is also false) | `opt::merge(mask_value)` only |
| `fill` | zero | `opt::merge(...)` |
| `load`, `load_convert` | zero | `opt::merge(...)` |
| `store`, `store_convert`, reductions | n/a — inactive addresses are simply **not accessed**; reductions have no inactive output lanes | — |

---

## Operations Reference

Unless noted otherwise: `r[i]` denotes the result lane; operations are
lane-wise; tag-inferred forms exist exactly for the lane-count-insensitive
operations (see [Call Forms](#call-forms)); and the masked form accepts the
usual active + population options.

### Basic (`vec/Basic.h`)

**Generators**

| Call | Semantics | Notes |
|---|---|---|
| `fill(tag, v[, opts])` | `r[i] = v` | Bit-copies the scalar, preserving NaN payloads, infinities and **signed zero**. Inactive lanes default to zero. |
| `zeros(tag)` | `r[i] = T{}` | |
| `mfill(tag, b)` / `mtrue` / `mfalse` | mask generators | |

**Mask generation (`while` family — tail handling)**

| Call | Semantics | Notes |
|---|---|---|
| `mwhilelt(tag, a, b)` | `r[i] = (a + i < b)` | The idiomatic tail mask: `mwhilelt(tag, i, n)`. |
| `mwhilele` / `mwhilegt` / `mwhilege` | analogous `<=`, `>`, `>=` | `mwhilele` treats `b == nint_t::max` as always-true; `mwhilegt` treats it as always-false (overflow guard). |

**Mask logic** — `mask_and`, `mask_or`, `mask_xor`, `mask_andnot`
(`(!a) && b`), `mask_not`.

**Mask queries** — `mask_all`, `mask_any`, and `mask_none` test the complete
logical mask; `mask_count` returns its true-lane count; `mask_first` and
`mask_last` return the lowest/highest true lane index, or `-1` for an empty
mask. Multi-word queries operate across the complete Tag, not per word.

**Lane sequences** — `iota(tag, start)` returns `start + i`, and
`iota(tag, start, step)` returns `start + i*step`. Integer results wrap modulo
the element width. For floating elements the lane ordinal is first represented
in the destination element type, then multiplied and added in that type's
normal compute format.

**Lane access**

| Call | Semantics | Notes |
|---|---|---|
| `get(tag, v, i)` | extract lane `i` | asserts `0 <= i < size(tag)` |
| `set(tag, v, i, x)` | replace lane `i` | same assert |
| `bitcast(to, from, v)` | reinterpret bytes, no value conversion | Smaller destination drops high source bytes; larger destination zero-fills. Cannot mix fixed and scalable descriptors. See `convert` for value conversion. |

**Rearrangement**

`H = Half<Tag>` below. Note the two shapes: `even`/`odd`/`lower`/`upper`
*produce* a half vector from a full one, while `concat`/`interleave` take
halves and *produce* a full vector.

| Call | Input → Returns | Semantics |
|---|---|---|
| `lower` / `upper(tag, v)` | `Vec<Tag>` → **`Vec<H>`** (also `Mask` overloads) | take low / high half of `v` |
| `concat(tag, lo, hi)` | two `Vec<H>` → **`Vec<Tag>`** (also `Mask` overloads) | join two half vectors |
| `even` / `odd(tag, v)` | `Vec<Tag>` → **`Vec<H>`** | lanes `2i` / `2i+1` of `v` |
| `concat_even` / `concat_odd(tag, a, b)` | two `Vec<Tag>` → **`Vec<Tag>`** | `concat` of the `even`/`odd` halves of `a` and `b` |
| `interleave(tag, a, b)` | two `Vec<H>` → **`Vec<Tag>`** | `r[2i]=a[i]`, `r[2i+1]=b[i]` |
| `interleave_even` / `interleave_odd(tag, a, b)` | two `Vec<Tag>` → **`Vec<Tag>`** | `r[2i]=a[2i]`, `r[2i+1]=b[2i]` (resp. odd lanes); the inverse of `interleave` |
| `local_interleave_lower` / `_upper(tag, a, b)` | two `Vec<Tag>` → **`Vec<Tag>`** | interleave the low (resp. high) halves inside each **16-byte block** only; blocks never exchange lanes |
| `shuf(tag, v, indices)` | `Vec<Tag>` → `Vec<Tag>` | permute within each **native word**; indices in `[0, native_word_size)`; words never exchange lanes |
| `local_shuf(tag, v, indices)` | `Vec<Tag>` → `Vec<Tag>` | permute within each **16-byte block**; compile-time form: `opt::lanes<I...>` |

> **`shuf` vs `local_shuf` granularity.** `shuf` permutes per native word;
> `local_shuf` and the `local_interleave_*` family permute per fixed 16-byte
> block. On a 512-bit backend a four-word vector shuffles four times with
> `shuf` but eight times with `local_shuf`. Neither ever moves a lane across
> its word/block boundary.

### Arithmetic (`vec/Arithmetic.h`)

| Call | Semantics | Notes |
|---|---|---|
| `add` / `sub` / `mul(tag, a, b)` | lane-wise arithmetic | Integer wraps modulo 2^bits; `float16_t`/`bfloat16_t` round back to their formats. Integer `add/sub(..., opt::saturate)` instead clamp the mathematical result to the element range; `opt::wrap` explicitly selects the default. |
| `div(tag, a, b)` | floating division | **Floating Tags only.** No integer division — use `bit_shr` for power-of-two division. `div(0)` yields inf/NaN per backend. |
| `min` / `max(tag, a, b)` | lane-wise extrema | Floating NaN and signed-zero selection follow the active backend. |
| `clamp(tag, value, lower, upper)` | `min(max(value, lower), upper)` lane-wise | Bounds need not be ordered; floating NaN and signed-zero selection match the active backend's `max` then `min`. Masked calls preserve `value` by default. |
| `neg` / `abs(tag, v)` | negate / absolute value | `neg` flips only the sign bit (preserves NaN payloads, signed zeros); integer negation is modular; `abs(INT_MIN)` keeps the `INT_MIN` bit pattern; unsigned `abs` is a no-op. |
| `copysign(tag, magnitude, sign)` | replace only magnitude's sign bit | Floating only; preserves all exponent/significand bits, including NaN payloads and signed zero. |
| `sqrt(tag, v)` | square root | floating only |
| `rcp` / `rsqrt(tag, v)` | reciprocal / reciprocal sqrt | Math-module ops with accuracy tiers — see [Math](#math-vecmathh); default `Strict` is within 1 ULP. |
| `fmadd` / `fmsub` / `fnmadd` / `fnmsub(tag, a, b, c)` | `a*b±c` fused family | Native fusion when available; fallback paths may round the product separately (**no single-rounding guarantee**). `fnmsub` fallback preserves IEEE signed zero. Integer lanes wrap. |

### Rounding (`vec/Rounding.h`)

| Call | Direction | Notes |
|---|---|---|
| `floor` / `ceil` / `trunc(tag, v)` | toward -inf / +inf / zero | Returns an integral floating value in the original element type. |
| `round(tag, v)` | nearest, halfway away from zero | Independent of the current floating-point rounding mode. |
| `round_even(tag, v)` | nearest, halfway to even | Independent of the current floating-point rounding mode. |
| `nearbyint(tag, v)` | current floating-point environment | Suppresses the inexact exception. |
| `rint(tag, v)` | current floating-point environment | May raise the inexact exception. |

All entries support the arithmetic mask/zero/merge population options.
NaNs, infinities, and signed zeros remain in their corresponding IEEE class;
the exact NaN payload behavior follows the backend and element format.

### Widening dot (`vec/WideningDot.h`)

```cpp
using Out = vec::ScalableTag<float32_t>;
using F16 = vec::ViewAs<float16_t, Out>;
using BF16 = vec::ViewAs<bfloat16_t, Out>;

auto products = vec::widening_dot(Out{}, F16{}, BF16{}, a, b);
auto accumulated = vec::widening_dot(Out{}, a, b, c); // source Tags inferred
```

For widening calls, both source elements have the same byte width but may be
different types, such as `float16_t * bfloat16_t` or `int8_t * uint8_t`.
The destination is a wider type in the same arithmetic category and all three
Tags describe the same logical byte span. If
`G = sizeof(To) / sizeof(From)`, output lane `k` is:

```text
c[k] + sum(widen(a[G*k+j]) * widen(b[G*k+j]), j=0..G-1)
```

The overload without `c` starts from zero. When all three types and Tags are
identical, the two forms are exactly `mul` and `fmadd`, respectively. Integer
inputs retain their individual signedness while widening and destination-width
arithmetic wraps modulo 2^bits. Floating backends may fuse or reassociate the
per-group operations, so results need not be bit-identical across ISAs.

The inferred overloads reconstruct each source Tag with `ViewAs<Source,ToTag>`;
`ToTag` therefore retains fixed/scalable and subword extent information that
cannot be recovered from the Vec representation alone.

### Bit (`vec/Bit.h`)

Integer Tags only (`IntegerTag`). All operations act on the *unsigned bit
pattern* (signed elements participate via two's complement).

Shift names follow the standard SIMD vocabulary: use `shl` / `shr`. The old
`bit_shl` / `bit_shr` entry variables are not retained as aliases.

| Call | Semantics | Notes |
|---|---|---|
| `bit_and` / `bit_or` / `bit_xor` / `bit_andnot` | `(~a) & b` for andnot | |
| `bit_not(tag, v)` | bitwise complement | |
| `shl` / `shr(tag, v, count)` | shifts | Count is one of three forms: `vecops::meta::Const<N>` selects the compile-time immediate path, `int` selects the runtime-scalar path, and a same-Tag `Vec` supplies per-lane counts. Counts must be non-negative; a negative `Const` is rejected and negative runtime counts are UB. `shr` is logical for unsigned and **arithmetic for signed** elements; oversized counts zero (logical) or sign-fill (arithmetic). |
| `popcount(tag, v)` | number of one bits in each lane | The result retains the input Tag and lies in `[0, element_bits]`. |
| `countl_zero` / `countl_one(tag, v)` | leading zero/one count | Zero input returns `element_bits` for `countl_zero`; all-one input does so for `countl_one`. |
| `countr_zero` / `countr_one(tag, v)` | trailing zero/one count | Zero input returns `element_bits` for `countr_zero`; all-one input does so for `countr_one`. |
| `rotl` / `rotr(tag, v, count)` | lane-wise bit rotation | Accepts the same immediate, runtime-scalar, and per-lane count forms as shifts. Counts are reduced modulo the element width; negative counts rotate in the opposite direction. |

Signed values are interpreted through their unsigned bit representation for
counts and rotations. Masked operations preserve the input by default, or use
the normal `opt::zero` / scalar merge / vector merge policies.

### Comparison (`vec/Comparison.h`)

All comparisons produce a `Mask<Tag>`.

| Call | Semantics | Notes |
|---|---|---|
| `cmpeq/ne/lt/gt/le/ge(tag, a, b)` | lane-wise comparison | Floating comparisons use C++ *ordered* semantics: any comparison with NaN is false — **except `cmpne`, which is true whenever either operand is NaN** (including NaN != NaN). |
| `isnan` / `isinf` / `isposinf` / `isneginf(tag, v)` | NaN/infinity classification (floating only) | `isinf` = either sign; anything not matching is false. |
| `isfinite` / `isnormal(tag, v)` | finite / normal classification | `isfinite` includes zero and subnormals; `isnormal` excludes both. |
| `signbit(tag, v)` | test the IEEE sign bit | True for negative zero and negative-sign NaNs as well as ordinary negative values. |

Masked comparisons AND the result with the governing mask; inactive lanes
default to **false** (only `opt::merge(mask_value)` can preserve given bits).

### Conversion (`vec/Conversion.h`)

```cpp
auto y = vec::convert(to_tag, from_tag, x);              // value conversion
auto m = vec::convert(to_tag, from_tag, mask);           // mask granularity change
```

- Both Tags must be given explicitly — a `Vec` alone does not remember
  sub-word shapes.
- **Ordered/unordered conversion requires equal logical lane shapes** (the
  same lane count; use `Rebind` to change the element in place). Width
  conversion between equal-byte Tags goes through `cvt::lane<P>` instead,
  typically with a `ViewAs` destination:
- **Layout** (default `cvt::ordered`): ordered maps output lane `i` from
  input lane `i`. `cvt::unordered` permits a stable backend-native
  permutation with a compositional contract: converting `A→B→C` has the same
  lane provenance as `A→C`, so `A→B→A` restores lane positions (a statement
  about *ordering*, not numeric round-trips).
- `cvt::lane<P>` operates on equal-byte Tags: widening selects input lanes
  `ratio*i+P`; narrowing writes converted input lane `i` to output lane
  `ratio*i+P`. Phase 1 requires a 2× width ratio. Widening lane conversion
  rejects population options.

```cpp
using F32   = vec::ScalableTag<float>;
using BF16V = vec::ViewAs<vecops::bfloat16_t, F32>;   // equal-byte bf16 view

auto narrow = vec::convert(BF16V{}, F32{}, v, vec::cvt::lane<0>);  // f32 → bf16
auto widen  = vec::convert(F32{}, BF16V{}, narrow, vec::cvt::lane<0>);
```

- **Value policy** (default `cvt::saturate`): clamps out-of-range values to
  the destination range. `cvt::wrap` is valid only for integer narrowing and
  keeps the low destination-width bits.

### Memory (`vec/Memory.h`)

```cpp
auto a = vec::load(tag, ptr);                                // contiguous
auto b = vec::load(tag, ptr, vec::opt::first(n));            // tail-safe
auto c = vec::load(tag, ptr, vec::opt::indexed(idx));        // gather
vec::store(tag, ptr, v, vec::opt::masked(m), vec::mem::aligned);
```

- Default access is unaligned and temporal. `mem::aligned` promises
  `memory_alignment(tag)` bytes (asserted in debug builds);
  `mem::non_temporal` is a hint that may silently degrade to temporal.
- `opt::masked` / `opt::first` filter the *access itself*: inactive
  addresses are never read or written. `first(0)` touches no memory.
- Inactive load lanes read as zero unless `opt::merge` is supplied. Stores
  have no population options (no inactive output lanes).
- `opt::indexed(idx)` performs gathers/scatters with i32 or i64 index
  vectors; `opt::strided(s)` is the linear special case. Both are mutually
  exclusive with alignment options. An **explicit indexed scale is in
  bytes**; scale 0 (the default) means "one memory element".
- `load(tag, {v0, v1, ...})` takes an initializer list of exactly
  `size(tag)` values (asserted).
- Pre-resolved `LoadRequest` / `StoreRequest` overloads skip option parsing
  for hot loops that reuse the same access shape.

### Converting Memory (`vec/ConversionMemory.h`)

```cpp
auto x = vec::load_convert(f32_tag, bf16_ptr, vec::opt::first(n));
const auto mem_mask = vec::mtrue(vec::Rebind<vecops::bfloat16_t, decltype(f32_tag)>{});
vec::store_convert(f32_tag, bf16_ptr, y, vec::opt::masked(mem_mask),
                   vec::cvt::unordered);  // unordered masks in the bf16 domain
```

`load_convert` / `store_convert` fuse the memory access with an element
conversion while keeping the caller's logical lane count. The API promises
*semantics*, not a single opcode — backends may use a fused instruction or an
equivalent load+convert sequence. When memory and logical element types are
identical these calls forward straight to `load`/`store`.

Special behavior worth knowing:

- **Mask domains.** Ordered conversion masks in the caller-logical domain.
  Unordered conversion normally masks in the memory-side (rebound) domain
  because lanes may be permuted. When the rebound Tag is not representable,
  only the caller-logical mask type is valid. `opt::first(n)` sidesteps the
  distinction entirely and is the recommended tail interface here.
- **Oversized rebinding.** A legal caller Tag is accepted even when the
  memory-side rebind exceeds the backend's representable power-of-two range;
  the request is then recursively partitioned into legal chunks (and may run
  lane-by-lane for masked non-contiguous boundaries).
- `cvt::ordered/unordered` and `cvt::saturate/wrap` are **semantic choices**,
  not hints.
- `store_convert` narrowing layout defaults to `mem::packed` (merge all
  source words into one destination word before storing — one wide store);
  `mem::split` stores each source word's narrowed result independently.
- As with plain load/store: masked-off addresses are untouched, inactive
  load lanes are zero unless merged, and alignment/temporality are hints.

### Math (`vec/Math.h`)

The exponential family covers three bases: `exp` / `exp_neg` (e^x),
`exp2` / `exp2_neg` (2^x), and `exp10` / `exp10_neg` (10^x), each with the
fixed-accuracy spellings (`exp_strict`, `exp_fast`, `exp_est`,
`exp2_*`/`exp10_*` and their `_neg_*` counterparts). Floating Tags only.

| Accuracy | Normal-input error contract |
|---|---|
| `Strict` (default) | ≤ 1 ULP |
| `Fast` | ≤ 4 ULP |
| `Estimate` | ≤ 4 ULP or ≤ 0.006 relative |

- Accuracy combines orthogonally with one active option (`masked` / `first` /
  `unmasked`) and the usual population options, in any order.
- The fixed-accuracy CPOs (`exp_fast`, ...) deliberately **reject** a
  conflicting accuracy option; use the canonical CPO (`exp`, `exp2`, `exp10`)
  when forwarding a template-selected accuracy.
- The `*_neg` variants assume every active lane is `x <= 0` and skip the
  overflow test — **positive active lanes have unspecified results**. Defining
  `VECOPS_MATH_ASSUME_VALID_INPUTS` additionally skips NaN propagation on
  these paths.
- Fast/Estimate always flush subnormal outputs to zero; with
  `VECOPS_PRESERVE_SUBNORMALS` defined, Strict preserves them.

Implementation notes: the SVE backend derives all bases from the Arm
optimized-routines SVE exp kernels (FEXPA table + shared reduction
skeleton). The x86 backend runs one family kernel over the bases:
`n = rint(x*log2(base))` plus a two-part Cody-Waite residual, a per-base
minimax polynomial, and SCALEF reconstruction (integer exponent injection
before AVX-512) — base 2 degenerates to the exact `n = rint(x)`,
`r = x - n` reduction, which makes `exp2` the cheapest member of the
family; f64 coefficients come from SLEEF's u10 `xexp2`/`xexp10` sets and
the f32/f16 sets are bespoke Remez/grid fits. The scalar backend leans on
the platform libm per lane.

The reciprocal family covers `rcp` (1/x) and `rsqrt` (1/sqrt(x)) with the
same option grammar (`rcp_strict`, `rcp_fast`, `rcp_est`, `rsqrt_*`).
Their per-tier contracts follow the estimate-plus-Newton ladder the
hardware offers (SVE `FRSQRTTE`/`FRECPE` + `FRSQRTTS`/`FRECPS`; x86
`RCP14`/`RSQRT14`, legacy `RCPPS`/`RSQRTPS`):

| Accuracy | rcp / rsqrt contract (f32, f64) | f16, bf16 |
|---|---|---|
| `Strict` (default) | ULP error <= 1 | ULP error <= 1 |
| `Fast` | relative error <= 2^-15 | ULP error <= 2 |
| `Estimate` | relative error <= 2^-7 | relative error <= 2^-7 |

The ladder mirrors common industry tiers: **Estimate** is the raw hardware
estimate (architecturally 2^-8 on SVE, tighter on x86), **Fast** adds one
Newton step — the classic NEON/SSE estimate-plus-refinement tier used by
game math and ML normalization kernels — and **Strict** refines with
Markstein-style exact-residual FMA steps to at most 1 ULP (the SLEEF u10
class, tighter than CUDA's 2-ULP `rsqrtf`). The SVE backend follows the
Arm optimized-routines SVE `rsqrt` design (estimate + step instructions +
one `NOINLINE` scaled special path owning every tail); x86 uses
`RCP14`/`RSQRT14` refinement with a cold long-double repair loop, falling
back to IEEE `1/x` / `1/sqrt(x)` on tiers lacking FMA or estimates.

Special inputs are **tier-independent**: `rcp(+-0) = +-inf`,
`rcp(+-inf) = +-0` with the input's sign, `rsqrt(+-0) = +inf`,
`rsqrt(+inf) = +0`, `rsqrt(x < 0) = NaN` (including `-inf`), and NaN
propagates. rcp's Strict tier produces gradual subnormal results under
`VECOPS_PRESERVE_SUBNORMALS` (rcp of huge magnitudes); Fast/Estimate flush
them to signed zero.
- NaN inputs propagate through every tier and family, including mixed
  vectors whose special-tail lanes force the slow path.

The logarithm family covers `log`, `log2`, and `log10` with the same
option grammar (`log_strict`, `log_fast`, `log_est`, `log2_*`, `log10_*`).
Logarithms have no hardware estimate instruction to refine against, so
the tiers ladder along the remaining cost axes — table lookups and
polynomial degree — and the top tier targets the accuracy class of Arm's
own production SVE libm routines rather than the 1-ULP reciprocal class:

| Accuracy | log family (f32) | log family (f64) | f16, bf16 |
|---|---|---|---|
| `Strict` (default) | ULP error <= 1 | ULP error <= 4 | ULP error <= 1 |
| `Fast` | relative error <= 2^-13 | relative error <= 2^-13 | ULP error <= 2 |
| `Estimate` | relative error <= 2^-7 | relative error <= 2^-7 | relative error <= 2^-7 |

**f64 Strict** is the optimized-routines vector-libm class: the SVE
backend ports their 128-entry `invc`/`logc`-table kernels verbatim
(measured 2.64/2.58/2.46 ULP for log/log2/log10; SLEEF's relaxed "u35"
tier is the same class), and a 1-ULP log would need double-double
accumulation of the table terms. **f32 Strict** stays native: a
degree-13 minimax of `(log_b(1+r) - alpha*r)/r^2` on the symmetric
`[1/sqrt2, sqrt2)` reduction window, assembled Cody-Waite style —
`k*log_b(2)` is split into a `hi` quantized to 15 significant mantissa
bits (so `k*hi` stays exact inside the final FMA) plus a tiny-`lo`
correction folded ahead of it, and `|log_b(z)| <= |log_b(x)|` on every
`k` band keeps each rounding at or below the result binade. Measured 1
ULP on the SVE backend. **Fast** drops the table for a six-term Taylor
kernel on the [2/3, 4/3) reduction window (`r = z - 1` is exact by
Sterbenz); non-e bases scale once at the end, which preserves the
relative-error bound. **Estimate** keeps three terms for the same 2^-7
budget as the reciprocal Estimate tier.

Reduction and tails follow the upstream design: `bits(x) - off` splits
`x = 2^k * z` with the table index masked to the table size (any input
bit pattern gathers in range), and one `NOINLINE` special routine owns
subnormals (renormalized by an exact power of two, with the logarithmic
correction folded into the extracted exponent as an integer bias on the
f32/f16 polynomial kernels), zero, negatives, `+inf`, and NaN. Special
inputs are tier-independent: `log(+-0) = -inf`, `log(+inf) = +inf`,
`log(x < 0) = NaN` (including `-inf`), `log(1) = +0` exactly, NaN
propagates. Log outputs are never subnormal, so the
`VECOPS_PRESERVE_SUBNORMALS` build modes differ only in accuracy. On
SVE, f16 Fast/Estimate run native half-precision kernels of the same
skeleton (exhaustively validated over all 65536 bit patterns), while f16
Strict and bf16 widen through the f32 pipeline: pure f16 arithmetic
cannot absorb one 2^-11 rounding per FMA in the `k*log_b(2) + log_b(z)`
cancellation band, so the 1-ULP class rides the native f32 kernel and
narrows once.

The x86 backend evaluates the same shared-constant kernels as SVE
(`vec/details/Math.h` owns the tables and polynomial coefficients, so
both vector backends and all three bases run identical instruction
sequences with matching throughput): f64 Strict is the
optimized-routines table kernel with AVX2/AVX-512 hardware gathers for
the `invc`/`logc` pairs (128-bit words load each entry with one
16-byte load and unpack the halves), f32 Strict is the native
degree-13 kernel, and Fast/Estimate are the shared Taylor kernels.
Special routines mirror the SVE structure — f64 renormalizes subnormal
lanes by 2^52 with the additive correction, f32 folds the correction
into the extracted exponent as an integer bias on the renormalized
lanes — and every f16 tier and bf16 widen through the f32 pipeline.
Targets without AVX-512DQ convert the extracted exponent to f64 via
the exact exponent-bias trick (biased into one binade so negative
exponents cannot borrow into the exponent field). Only the scalar
backend still evaluates the IEEE libm logarithm per lane (through a
float intermediate for the narrow formats).

The trigonometric family provides `sin`, `cos`, `tan`, the C23
half-revolution forms `sinpi`, `cospi`, `tanpi`, and the fused
`sincos`/`sincospi` calls. Every canonical name has the usual
`*_strict`, `*_fast`, and `*_est` fixed-tier spellings. The fused API uses
two output references because scalable SVE values cannot be members of a
normal C++ pair:

```cpp
auto s = vec::zeros(tag);
auto c = vec::zeros(tag);
vec::sincos(tag, x, s, c, vec::opt::math::fast);
vec::sincospi(x, s, c);  // tag-inferred, Strict by default
```

Each fused component obeys the matching unary contract. For Fast and
Estimate, “mixed error” means
`abs(result-reference) <= eps * max(1, abs(reference))`; unlike pure
relative or ULP error this remains meaningful around roots and tangent
poles.

| Accuracy | f32 | f64 | f16 | bf16 |
|---|---|---|---|---|
| `Strict` (default) | <= 4 ULP | <= 4 ULP | <= 1 ULP | <= 1 ULP |
| `Fast` mixed error | <= 2^-12 | <= 2^-26 | <= 2^-5 | <= 2^-4 |
| `Estimate` mixed error | <= 2^-7 | <= 2^-13 | <= 2^-4 | <= 2^-3 |

All tiers cover the complete finite radian domain. Common magnitudes take a
vector Cody-Waite path; large arguments take a cold high-precision reduction
or per-lane libm repair path. Pi-scaled operations never multiply the input by
an approximate pi: they reduce the dyadic input around integers/half-integers,
so `sinpi` integer zeros, `cospi` integer/half-integer values, and `tanpi`
integer zeros/half-integer infinities have the C23-prescribed exact values and
signs. Infinities map to NaN and NaNs propagate in every tier.

The SVE hot path follows Arm optimized-routines and uses the base-SVE
`FTSSEL`/`FTSMUL`/`FTMAD` trigonometric-assist instructions for f32/f64. x86
uses FMA polynomial kernels and ISA-native masks/conversions; there is no x86
trigonometric instruction (Intel SVML entries are compiler/runtime functions,
not opcodes). f16 Strict and every bf16 tier widen through f32. SVE f16 Fast
and Estimate use the native half-precision `FTMAD`/`FTSMUL`/`FTSSEL` forms on
the hot domain and fall back to paired f32 only for large inputs and sensitive
tangent poles. AVX512-FP16 likewise uses native half arithmetic for every f16
Fast/Estimate operation; large arguments and sensitive tangent poles retain a
paired-f32 fallback. `sincos` and `sincospi` share one reduction and evaluate
both components together. The f16 Fast/Estimate contracts are exhaustively
checked over all 65536 input bit patterns.

### Reductions (`vec/Reduction.h`)

| Call | Semantics | Notes |
|---|---|---|
| `reduce_add(tag, v)` | sum of all lanes | Integer wraps; floating association is backend-defined; multi-word values fold with vector adds first, then one horizontal reduce. Empty masked selection → additive identity (zero). |
| `reduce_max` / `reduce_min(tag, v)` | greatest / least lane | NaN and signed-zero selection follow the backend. Empty selection → ∓infinity (floating) or lowest/greatest integer. |

Filtered forms accept exactly one `opt::masked(mask)` or `opt::unmasked`;
population options are invalid (no inactive output lanes). Reductions have
**no tag-inferred form** — the Tag argument is always required.

### Prefetch (`vec/Prefetch.h`)

`prefetch(tag, ptr, hints...)` prefetches the cache line containing `ptr`.
The three dimensions (locality, temporality, intent) default to
`prefetch_l1` / `prefetch_keep` / `prefetch_read`; unsupported combinations
map to the closest hint. **Purely advisory; the scalar backend makes it a
no-op.**

---

## How It Is Implemented

Three-layer header structure (see `AGENTS.md` for the conventions):

```
vec/X.h                      public API: Op declarations, doc, operator() entries
vec/details/X.h              backend-independent layer: option validation,
                             GenericImpl fallbacks, shared helpers
vec/details/{scalar,sve,x86}/X.h   ISA specializations (NativeImpl /
                             NativeWordImpl) for that backend tag
```

**Dispatch.** Each top-level header declares its Op functor types and the
`inline constexpr` entry variables, then includes the backend specializations,
then defines the `operator()` bodies. `details::execute()` probes, at compile
time and in this order:

```cpp
if constexpr (requires { NativeImpl<CurrentBackend, Op, Tag>::call(...); })
  // 1. whole-Tag native implementation (cross-word semantics, no-ops, ...)
else if constexpr (word_count == 1 &&
                   requires { NativeWordImpl<CurrentBackend, Op>::call<0>(...); })
  // 2. single physical word of native instructions
else if constexpr (requires { GenericImpl<CurrentBackend, Op, Tag>::call(...); })
  // 3. architecture-independent fallback (per-word batching, lane loops)
else static_assert(...);  // clear compile error naming Op and Tag
```

**Internal calls use the same short names.** Because the entry variables are
declared before the backend includes, implementations inside `vec/details/`
call other operations through the very same names external users see —
`mul(tag, a, b)`, `blend(tag, inactive, mask, computed)`. Every Op carries a
word-level `operator()` entry written inline in its own definition (via
forward declarations of `details::execute`/`execute_word` at the end of
`VecBase.h`) so the same call shape works on physical words: on single-word
Tags (where `Vec<Tag>` and `NativeWordVec<Tag>` name the same type) the
public whole-Tag overloads take over and dispatch routes into
`NativeWordImpl<0>` identically.
Inside the details layer the math tokens additionally expose
`details::exp<Base, Accuracy, NegativeOnly>(tag, v)`-style variable templates
that shadow the public variables and force an explicit accuracy tier; word
broadcast uses `fill_word`, and masked unary math word implementations share
the `masked_unary_word` sanitize→compute→blend skeleton. The explicit
`execute_word<Index>(op, ...)` spelling survives only where it means
something: index-sensitive conversion/memory word calls, and the
backend-agnostic batching loops of `GenericImpl`.

Consequences:

- A backend implements exactly what its ISA can express; everything else
  falls through to generic code automatically. That is why e.g. the scalar
  backend has no `Conversion.h` — the generic per-lane `GenericImpl` in
  `details/Conversion.h` already is the implementation.
- Multi-word Tags are batched word by word for every operation whose Op
  opted in (`EnableElementwiseWordBatching`); reductions and shuffles are
  combined/kept local as their semantics require.
- The backend is selected by target macros at include time
  (`ARCH_X86_FAMILY`, `CPU_CAPABILITY_SVE`, ...), fixed into the
  `CurrentBackend` tag alias, and never revisited at runtime.

**Options → Request, folded once.** An option pack is parsed exactly once
into a `LoadRequest` / `OpRequest` / ... descriptor (`Request.h`): path-selecting
kinds (masked? indexed? merge?) become template parameters so backends
branch with `if constexpr`, while runtime values (stride, counts, mask and
merge vectors) become plain fields. Every layer below the entry point
receives the struct instead of re-walking the pack — this keeps template
instantiation growth bounded. The request objects are also part of the
public API: you may construct one and call the request overload directly to
skip option parsing in hot loops.

---

## Pitfalls and FAQ

**Masks and merges must be lvalues.**
`opt::masked`, `opt::merge`, and request fields hold *references* because
sizeless SVE vectors cannot be stored as members. Passing a temporary is
deleted at compile time. Keep the mask in a named variable.

```cpp
const auto m = vec::mwhilelt(tag, i, n);
auto r = vec::add(a, b, vec::opt::masked(m));   // OK
// vec::add(a, b, vec::opt::masked(vec::mtrue(tag)));  // ERROR: rvalue
```

**Any option requires an active option.** `vec::add(a, b, vec::opt::zero)`
does not compile: once you pass options you must include exactly one of
`masked` / `unmasked` / `first`. A plain `vec::add(a, b)` needs none.

**Check the inactive-lane default per family.** `add`-family masked results
preserve `a[i]` in inactive lanes; comparisons yield false; loads yield
zero (see the table above). If you assumed one uniform rule, audit your
masked call sites.

**`rcp`/`rsqrt` live in `vec/Math.h` with accuracy tiers** (default
`Strict` is within 1 ULP — see [Math](#math-vecmathh)); `div` remains the
IEEE division. The raw hardware-estimate behavior is now the `estimate`
tier: `rcp_est`/`rsqrt_est` match the old bare instructions. Estimate-seed
instructions are not bit-deterministic across compiled call shapes (masked
vs unmasked forms), so two calls in one program may differ in the last
estimate bits while staying inside the tier bound.

**`cmpne` is true for NaN operands**, unlike every other comparison. Use
`!(a == b)`-style logic (i.e. `mask_not` over `cmpeq`) if you need the
"ordered unequal" meaning.

**`shr` on signed elements is an arithmetic shift.** Logical shifting
requires unsigned elements. Every scalar or per-lane shift count must be
non-negative; violating this precondition has undefined behavior. Use
`meta::cint<N>` when the count must reach the backend as an immediate;
an ordinary integer literal has type `int` and therefore uses the
runtime-scalar API category even if the compiler later constant-folds it.

**Rotation counts do not share the shift precondition.** `rotl` and `rotr`
reduce every count modulo the element width, and negative counts reverse the
direction, matching the standard C++ bit-rotation functions.

**Shuffles never cross their granularity boundary** — native word for
`shuf`, 16-byte block for `local_shuf` / `local_interleave_*`. Indices are
relative to that boundary, not to the whole vector.

**`bitcast` ≠ `convert`.** `bitcast` reinterprets bytes (dropping/zero-filling
on width change); `convert` performs value conversion with saturation and
lane-order policies. Mixing fixed and scalable descriptors in one `bitcast`
is rejected at compile time.

**Tag inference is lossy for sub-word vectors.** After
`VecToTag<V>`, a `FixedTag<i8, 8>` looks like a full `i8` word. When looping
sub-word chunks, keep the original Tag and pass it explicitly.

**`load_convert`/`store_convert` masks have two domains.** Ordered →
caller-logical; unordered → memory-side rebound. Prefer `opt::first(n)` for
tails to avoid thinking about domains at all.

**Indexed scale is in bytes** (0 = one element). The tensor layer assigns
different (logical) semantics on top; raw-vec users multiply byte offsets.

**Sizeless-SVE life cycle.** On SVE, `Vec` values are sizeless: they cannot
go in arrays, tuples or containers and have no `sizeof`. Build multi-word
values with `from_words(tag, ...)` / `concat`, and let the option system
hold references rather than copies. Portable code that follows the patterns
above needs no `#ifdef`.

**Clang + VLA + multi-word masks: out-of-line word accessors without
SVE2.1.** In a vector-length-agnostic build, a multi-word `Mask` is stored
as an architectural predicate tuple (`svboolx2_t` / `svboolx4_t`), and its
per-word accessors go through `svget2`/`svset2`/`svget4`/`svset4`. Clang
gates these builtins behind the SVE2.1 target feature even though they only
describe predicate-register grouping, so when compiling with Clang without
SVE2.1 enabled the library marks those accessors
`__attribute__((target("sve2p1")))` to keep the code correct — and calls
across a target-attribute boundary cannot be inlined. The result is real
function-call overhead on multi-word Mask word access, and a measurable
slowdown in kernels that use masked multi-word SVE vectors. Single-word
masks (`svbool_t`) and data-vector tuples are unaffected. Remedies, in
order of preference: compile with SVE2.1 enabled (e.g. `-msve2p1` or a
newer baseline), use GCC 13+ (which accepts the tuple operations under base
SVE and keeps the accessors always-inline), or build vector-length-specific
(`-msve-vector-bits=<N>`), where masks use sized array storage and the
tuple path is bypassed entirely.

**Store to inactive lanes never happens.** With `masked`/`first`, inactive
addresses are not touched — safe for tails, but it also means a masked store
cannot zero-fill trailing memory; do that with a separate store.

**Debugging failures.** Invalid option combinations, out-of-range lane
indices, misaligned `mem::aligned` pointers and short initializer lists are
reported by `static_assert` (compile time) or `VECOPS_ASSERT` (debug builds)
with the offending operation named. If dispatch finds no implementation at
all, the `static_assert` names the Op and Tag.

**Which header to include.** `vecops/vec/Vec.h` pulls in everything; every
operation header is also self-sufficient (`Arithmetic.h` alone compiles).
Include what you use; `details/` headers are internal — do not include them
outside the module (shared option traits live at the end of the public
`Options.h`).
