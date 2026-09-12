# The `meta` Module

`vecops::meta` (in `vecops/Meta.h`) provides **typed integer metadata**:
instead of passing bare `nint_t` around, an integer is represented by a
*type* that records what the compiler is allowed to know about it — whether
it is a compile-time constant, and for runtime values, what alignment and
lower/upper bounds hold. Arithmetic on these types **propagates every
constraint that can be proven from the operands**, so a `Dynamic<16, 0, 4096>`
multiplied by `Const<4>` is known — statically — to be a multiple of 64 in
`[0, 16384]`.

Higher layers consume these facts to pick specialized code paths at compile
time (a stride that is provably `Const<1>` needs no multiply; a byte offset
that is provably a multiple of 32 can use an aligned vector store).
`vec::opt::strided` accepts `meta::Const` / `meta::Dynamic` stride metadata
for exactly this purpose, and the tensor layer's shapes and strides are
built from the same vocabulary. `Meta.h` itself depends on nothing but
`CoreTypes.h` / `Assertion.h` / `util` — tensor and vector layers may both
use it with no dependency cycle.

```cpp
#include "vecops/Meta.h"
using namespace vecops::meta;

auto four  = cint<4>;                    // Const<4>, value known statically
auto n     = dyn<8, 0, 1024>(128);      // Dynamic<8, 0, 1024>: runtime, 8-multiple, [0,1024]
auto bytes = n * four;                   // Dynamic<32, 0, 4096>
static_assert(decltype(bytes)::aligns(32));

auto unknown = n + nint_t{3};            // raw integer → Any: all constraints dropped
static_assert(!decltype(unknown)::aligns(2));
```

---

## Core Concepts

### The Value hierarchy

Everything derives from `meta::Value`. Four spellings matter:

| Type | Meaning | Storage |
|---|---|---|
| `Const<N>` | The value is exactly `N`, known at compile time. `N` may be negative. | none — the value lives in the type |
| `Dynamic<A, Lo, Hi>` | A runtime value that is a multiple of `A` and lies in the inclusive range `[Lo, Hi]`. `A` must be a positive power of two. | one `nint_t` per instance |
| `Any` | `Dynamic<1>` — a runtime integer with no useful constraints. | one `nint_t` |
| `any` | **Not a value.** A type-level wildcard reserved for metadata pattern-matching traits. Lowercase `any` never holds a runtime value; use `Any` for that. | — |

Omitted bounds mean unbounded: `Dynamic<4>` is `Dynamic<4, kLoInf, kHiInf>`
where `kLoInf` / `kHiInf` are the `nint_t` extreme values used as "no bound"
sentinels.

Factories: `cint<N>` (variable template for `Const<N>{}`) and
`dyn<A, Lo, Hi>(v)` / `dyn<A>(v)` (function templates for `Dynamic`), so the
common cases never spell out the template arguments.

The `ValueType<T>` concept recognizes any `Value` subclass. `ValueInput<T>`
recognizes the broader API-input vocabulary: either a Value subtype or one of
vecops' fixed-width integer types. `to_value_t<T>` performs the corresponding
type normalization, while `to_value(value)` performs value normalization: raw
integers (including `nint_t`) become `Any`, and existing Value types preserve
their metadata. These helpers let APIs accept mixed bare and typed arguments
without defining local conversion concepts or overload sets:

```cpp
auto runtime = to_value(nint_t{17}); // Any{17}
auto fixed = to_value(cint<4>);      // Const<4>{}
auto strides = make_strides(1, 128, cint<64>);
```

Values also implicitly convert to `nint_t`, so they can cross into ordinary
runtime APIs, loop bounds, and pointer arithmetic without local casts. This
does not change the preferred propagation rule: retain a Value in an `auto`
variable while constructing shapes or calling Value-aware operators. When a
Value mixes with an ordinary integer type, the integer side is normalized to
`Any`; use `cint<N>` when that operand is itself a compile-time fact.

### Type guarantees vs. instance observations

Two related but distinct questions, deliberately separate APIs:

| Question | API | Example |
|---|---|---|
| Does *every* value of this type align to `k`? | `T::aligns(k)` (static) | `Dynamic<8>::aligns(4)` → true; `Any::aligns(4)` → false |
| Does *this particular instance* align to `k`? | `v.is_aligned(k)` (member) | `Any{4}.is_aligned(4)` → true |
| Does the compile-time value `v` satisfy this type's constraints? | `T::conforms(v)` (static) | `Dynamic<8, 0, 16>::conforms(10)` → false (10 is not an 8-multiple) |

`aligns` is **conservative**: `false` means "not guaranteed", never
"impossible". Dispatching on `aligns` selects fast paths; `is_aligned` is
for runtime checks and assertions.

---

## API Reference

### Constraint propagation through arithmetic

The operator set (`+`, `-`, unary `-`, `*`, `/`, `%`) is overloaded for
`Const`/`Dynamic` combinations and returns the strongest provable type:

| Expression | Result type |
|---|---|
| `Const<N> ± Const<M>` | `Const<N±M>` |
| `Const<N> ± Dynamic<A,L,H>` | `Dynamic<g, shifted bounds>` with `g = min(A, lsb(N))` (or `A` if `N == 0`); bounds shift by `±N` (swapped for `Const - Dyn`) |
| `Dynamic ± Dynamic` | `Dynamic<min(A1,A2), merged bounds>` — the weaker alignment survives, bounds add/subtract corner-wise |
| `Const<N> * Const<M>` | `Const<N*M>` |
| `Const<N> * Dynamic<A,...>` | `Const<0>` if `N == 0`; else `Dynamic<A * lsb(N), scaled bounds>`; available bounds propagate independently |
| `Dynamic * Dynamic` | `Dynamic<A1*A2, four-corner bounds>`; sign-proven one-sided ranges retain the bounds that can be established |
| `Dynamic<A,...> / Const<N>` | `Dynamic<A/abs(N),...>` if `A % N == 0`, else alignment degrades to 1; available bounds propagate independently and swap for negative `N` |
| `Const / Dynamic`, `Dynamic / Dynamic` | alignment always degrades to 1; bounds exact only when the denominator range does not cross zero (otherwise unbounded) |
| `Dynamic<A,...> % Const<N>` | **`Const<0>`** if `A % N == 0` (every value is a multiple of `N`); else `Dynamic<gcd(A,N), remainder range>` |
| `Const<N> % Dynamic<A,...>` | common alignment and sign survive; magnitude is bounded by `abs(N)` and, for a bounded divisor, by its maximum magnitude minus one; `N == 0` folds to `Const<0>` |
| `Dynamic % Dynamic` | common alignment survives; conservative one-sided remainder bounds survive even for otherwise unbounded operands |

Any operand that is a raw `nint_t` is wrapped as `Any` first, so that operand
contributes nothing: `dyn<8>(64) + nint_t{3}` is an unconstrained `Any`
result, not a `Dynamic<8>`. Information implied by the other operand and the
operation itself can still survive (for example, a remainder by a runtime
integer still has a finite range when the dividend is `Const`). If the raw
integer is actually known, say so with `cint<N>`.

Two invariant notes behind the rules: alignments are always powers of two,
and `min`/`gcd` of powers of two is again a power of two, so results stay
representable; and division propagation uses the piecewise monotonicity of
truncating division, which is why denominator ranges that cross zero lose
their bounds entirely.

### Value-aware math overloads (namespace `vecops`)

`min`, `max`, `clamp`, `ceil_div`, `floor_div`, `align_up`, `align_down` —
the scalar helpers from `util/Math.h` — have Value-aware overloads in the
parent `vecops` namespace. Const/Const folds at compile time; mixed cases
propagate:

| Call | Result | Notes |
|---|---|---|
| `min(Const, Dyn)` | folds to `Const<N>` when the whole Dynamic range is ≥ N (or to the Dynamic when ≤ N); otherwise `Dynamic<gcd(A,N), L, min(H,N)>` | result is one of the operands, hence gcd alignment |
| `max(Const, Dyn)` | dual of `min` | lower bound tightens to `max(L, N)` |
| `min/max(Dyn, Dyn)` | `Dynamic<gcd(A1,A2), merged bounds>` | bound sentinels compare naturally, no special case |
| `clamp(Dyn, Const<Lo>, Const<Hi>)` | intersection bounds; **folds to `Const`** when the intersection collapses or the Dynamic lies entirely outside `[Lo, Hi]` | alignment `gcd(gcd(A,Lo),Hi)`; **no `nint_t`-bound overload** — runtime bounds have nothing to propagate |
| `ceil_div` / `floor_div(Dyn, Const<N>)` | `Dynamic<A/N,...>` if `A % N == 0`, else alignment 1 | `N > 0` required (static_assert); runtime-divisor forms return plain `Any` |
| `align_up` / `align_down(Dyn, Const<N>)` | identity with alignment `A` preserved when `A % N == 0`; otherwise `Dynamic<lsb(N)>` because every result is an `N`-multiple | each available bound propagates independently; runtime-alignment forms return plain `Any` |

As with the arithmetic operators, every `nint_t` operand participates as
`Any`.

### Bound queries

For any `ValueType T` (a `Const` counts as both-bounded at its value):

| Variable | Meaning |
|---|---|
| `has_lower_bound_v<T>` / `has_upper_bound_v<T>` | whether the respective bound is not a sentinel |
| `lower_bound_v<T>` / `upper_bound_v<T>` | the bound (sentinel when absent) |
| `is_bounded_v<T>` | both bounds present |
| `range_within_v<T, Lo, Hi>` | bounded and entirely inside `[Lo, Hi]` |
| `lower_bound_at_least_v<T, Lo>` / `upper_bound_at_most_v<T, Hi>` | one-sided containment |

### `PackedStorage<Is...>` (namespace `meta::details`)

Compressed storage for a pack of Values: `Const` entries cost **zero**
storage (they live in the type), only `Dynamic` members occupy an `nint_t`
each. `PackedStorage<Const<2>, Dynamic<4>, Const<3>>` stores exactly one
runtime integer. This is how shape/stride tuples with many compile-time
dimensions stay small.

| Member | Purpose |
|---|---|
| `PackedStorage{Const<2>{}, Dynamic<4>{128}, Const<3>{}}` | construct from unpacked **Value instances** (not bare integers — the pack must match the type list); asserts conformance |
| `get<I>()` | value of dimension `I` — compile-time for Const entries |
| `operator[](i)` | runtime-indexed access, always a runtime read |
| `to_array()` | expand to the full `std::array<nint_t, n>` |
| `to_packed_array()` / `from_packed_array(p)` | raw access to / construction from the compressed layout (no validation) |

There is also a constructor from an already-compressed `const nint_t*`
array (validated) — the round-trip partner of `to_packed_array()`.

### Conversion safety (`details::IsMoreLenientValue<VSrc, VDst>`)

Defines which Value-type conversions are information-preserving: same type;
`Const<N>` → `Dynamic` when `N` satisfies the target's constraints;
`Dynamic<A1,...>` → `Dynamic<A2,...>` when the target is strictly weaker
(`A1 % A2 == 0`, bounds relaxed); anything → `Any`. Tightening directions
(e.g. `Dynamic` → `Const`) are not implicit — the value might violate them.
It lives in `details` because consumers (the tensor layer) use it for
metadata slot matching.

---

## How It Is Implemented

The constraints *are* the template parameters, so every operation computes
its result type with plain `constexpr` arithmetic before constructing the
instance:

- alignment propagation uses least-significant-bit analysis (`lsb(N)`) for
  add/sub/mul with constants, `min` for add/sub of two Dynamics, products
  for mul — all keeping the power-of-two invariant;
- bounds propagate by interval arithmetic: shifting for `±Const`, corner
  merging (plus sign-aware one-sided inference) for `*`, monotonic endpoint
  propagation for division, and sign/magnitude bounds for remainder. Each
  sentinel is handled independently so one missing side does not erase a
  provable bound on the other side;
- `PackedStorage` computes its compression offsets with a fold over the
  type pack at compile time (`offsets`, `num_stor`), so packing and
  unpacking are index remaps with no runtime dispatch.

Runtime validation (`Const` value match, `Dynamic` conformance,
`PackedStorage` entry conformance, `operator[]` range) goes through
`VECOPS_ASSERT`.

---

## Pitfalls and FAQ

**`Const<N>` construction asserts the value.** `Const<16> c(8)` is a runtime
assertion failure. The default constructor (`Const<16>{}` / `cint<16>`) is
the intended spelling; the value-taking constructor exists so unpacked
arrays can be rebuilt.

**`Dynamic` construction asserts its constraints.** `dyn<8>(10)` fails at
runtime: 10 is not an 8-multiple. Constraints are *promises* about a value
you supply, not sanitization of arbitrary input.

**One raw `nint_t` poisons the expression.** Any arithmetic mixing a bare
integer wraps it as `Any` and drops everything known on the other side.
Write `cint<4>` instead of `4`, and `dyn<A>(v)` instead of `v`, at the
earliest point where knowledge exists.

**`aligns(k)` false ≠ misaligned.** It means "not provable". `Any{64}` is a
perfectly 64-aligned instance; only `is_aligned` can see that. Conversely,
never use `is_aligned` to select code paths — it varies per call.

**"Alignment" is integer divisibility.** `Dynamic<16>` says "divisible by
16" in whatever unit the surrounding abstraction uses — elements, bytes,
tiles. It is not a C++ `alignof` and not tied to pointers.

**Division and remainder are deliberately conservative.** Quotients lose
alignment unless the divisor divides the operand alignment exactly;
denominator ranges crossing zero lose quotient bounds. Remainders retain the
common divisor alignment and any independently provable sign/magnitude bound.
Inspect the *result type* (`decltype`) before relying on a constraint in a
specialization.

**`clamp` has no runtime-bounds overload.** `clamp(d, lo_dyn, hi_dyn)` does
not exist — runtime bounds carry no compile-time information, so there is
nothing to propagate. Convert to `Any` explicitly if you need the scalar
behavior.

**`kLoInf` / `kHiInf` are real `nint_t` extremes**, not a separate "absent"
representation. Code doing raw arithmetic on bounds must handle the
sentinels explicitly (as this header's operators do); comparing them with
`<` / `>` works naturally.

**`any` is not `Any`.** `Any` is an unconstrained runtime Value; `any` is a
type-level wildcard that never holds a value. They are unrelated despite
the names.

**Assertions vanish in release builds.** `VECOPS_ASSERT` is debug-only
diagnostics. Constraints remain compile-time facts either way, but do not
treat the runtime checks as input validation.
