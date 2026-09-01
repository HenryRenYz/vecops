# Supported Element Types

Everything in vecops is parameterized by *element type*. Twelve are
supported (`vecops::CoreTypes.h`), and the `vec::Element` concept accepts
exactly these on every backend:

| Category | Types | Definition |
|---|---|---|
| Small floats | `vecops::float16_t`, `vecops::bfloat16_t` | custom storage wrappers (`util/Float16.h`, `util/BFloat16.h`) |
| IEEE floats | `vecops::float32_t` = `float`, `vecops::float64_t` = `double` | plain aliases |
| Integers | `vecops::int8_t` … `vecops::uint64_t` (4 widths × 2 signs) | aliases of the `<cstdint>` types |

Floats and integers flow through the same operation surface: `vec::add` on
a bf16 Tag and on an i32 Tag are the same API with backend-appropriate
lowering. Integer arithmetic wraps modulo 2^bits; the interesting design
questions are all in the two small-float types.

---

## `float16_t` and `bfloat16_t`: storage types, deliberately

Both are `struct alignas(2)` wrappers around one `uint16_t` of raw bits.
They are **not** the compiler's native `_Float16` / `__bf16`.

### Why not the compiler types

Compiler-provided fp16/bf16 support is uneven: `__bf16` on GCC/Clang is a
*storage* type with little or no arithmetic — you cannot reliably `+` two
of them, and what compiles on one toolchain fails on another; `_Float16`
support depends on target and compiler version, and neither gives a
portable `numeric_limits` story. vecops instead owns the format: a
two-byte storage wrapper with an explicit, controlled conversion surface, so
behavior is identical across x86, SVE, and scalar builds.

What the wrappers provide:

- Implicit conversion to/from `float` (the universal compute lane) and from
  `double`;
- Explicit bit-level access: `from_bits(uint16_t)` / `to_bits()`;
- Interop with the native types where they exist — constructing from, and
  converting to, `_Float16` / `__fp16` (ARM) / `__bf16` — as pure bitcasts;
- A full scalar operator surface (`+ - * /`, comparisons, `std::abs`,
  streaming) via the `SmallFloatOps` CRTP base, **every operation of which
  round-trips through `float`**: the wrapper stays a storage format, scalar
  math happens at f32 width, and the result is re-narrowed;
- Complete `std::numeric_limits` specializations (min/max/epsilon/denorm
  style, etc. — for `bfloat16_t`: same exponent range as `float`, 8
  significand bits; for `float16_t`: IEEE binary16 with its much smaller
  range).

### The two formats at a glance

| Property | `float16_t` (IEEE binary16) | `bfloat16_t` (truncated float) |
|---|---|---|
| Exponent / mantissa bits | 5 / 10 | 8 / 7 |
| Dynamic range | ±65504, normals down to ~6e-5 | same as `float` (±3.4e38) |
| Precision | ~3.3 decimal digits | ~2.3 decimal digits |
| Relationship to f32 | separate format, real conversion | literally the top 16 bits of an f32 (round-to-nearest-even on narrowing) |

`bfloat16_t` trades precision for range — that is why it dominates ML
storage, and why accumulating in it is a bad idea (below).

---

## How the vec backends compute on small floats

A backend uses native small-float instructions **where the target actually
has them**, and otherwise **widens to `float32`, computes, and narrows
back**:

| Backend | fp16 | bf16 |
|---|---|---|
| x86 | native arithmetic with AVX512-FP16 (`..._ph` intrinsics); otherwise f32-widened (with F16C conversions) or per-lane scalar emulation | f32-widened always; conversion via AVX512-BF16 intrinsics when present; dot products via `dpbf16ps` where applicable |
| SVE | native `..._f16` arithmetic where the width is covered; otherwise f32-widened | f32-widened as the norm: a bf16 word is split into lo/hi f32 pairs, computed, re-packed (`sve/Bf16.h`) |
| scalar | per-element, via the float round-trip above | same |

The rounding family follows the same rule. SVE uses native `FRINT*` for fp16,
fp32, and fp64, and AVX512-FP16 uses `VRNDSCALEPH`; other fp16 configurations
and every bf16 configuration widen each word to an f32 pair, round both halves,
and pack once. The rounded value is exactly representable in the source format,
so this final pack does not introduce a second rounding decision.

Narrowing back uses round-to-nearest-even (that is what F16C's
`_cvtss_sh` and the software bf16 rounding do), and the bf16 constructor
canonicalizes NaN inputs to `0x7fc0`.

### Performance characteristics

The widen-compute-narrow round trip is usually **cheap** — on the order of
one convert instruction per vector each way, and for bf16 essentially a
shift — so scattered small-float arithmetic performs fine. But it is real
work in **hot loops**: every small-float vector op pays two conversions the
native-width version doesn't, and on x86 without AVX512-FP16 a fp16
arithmetic fallback can degrade to per-lane scalar emulation, which is
visibly slow. When a kernel computes intensively over narrow storage, the
idiomatic remedy is to widen *once* and stay wide:

- use `vec::convert` to move the data into an f32 Tag for the arithmetic
  section and convert back at the end; or
- let the tensor layer do exactly that for you — `tensor::input<float>` /
  `output<float>` over a bf16/fp16 Tensor makes the kernel see `float`
  vectors end-to-end while memory stays narrow (see
  [Tensor.md — ComputeType versus the Tensor's element
  type](Tensor.md#comptype-versus-the-tensors-element-type)).

---

## Pitfalls and FAQ

**They are storage types; scalar math is f32 math.** `a * b + c` on
wrappers narrows after every step — three roundings, not one. Fine for
glue code; do not accumulate in it (especially `bfloat16_t`, whose 7-bit
mantissa loses fraction fast — accumulate in f32, store in bf16).

**`from_bits` / `to_bits` bypass everything**, including NaN
canonicalization. They are the honest way to move bits (I/O, hashing,
reinterpretation); arithmetic on arbitrary bit patterns then follows IEEE
rules via the f32 round-trip.

The vector classification and sign operations inspect these native 16-bit
encodings directly. In particular, `isnormal` distinguishes bf16/fp16
subnormals before any widening (a small-float subnormal can become a normal
f32 value), while `signbit` and `copysign` preserve NaN payloads and signed
zero without a float round trip.

**Conversions need hardware to be fast.** Without F16C / NEON / AVX512-BF16,
scalar conversion falls back to a software implementation (the fp16 one is
the classic bit-twiddling from PyTorch's `Half.h`). Bulk conversions should
go through `vec::convert` / `load_convert` so they use the vector path.

**Mixed comparisons with raw floats can be ambiguous.** The wrapper
operators are defined wrapper-to-wrapper; `fp16_value < 0.5f` may fail to
resolve between two implicit conversions. Cast one side to `float`
explicitly. Relatedly, `float16_t::operator float` is **not `constexpr`**
(the F16C intrinsic path cannot be), while `bfloat16_t`'s is — only bf16
conversions participate in constant expressions.

**`float16_t` has a small exponent.** Normals bottom out near 6e-5; values
below that are denormals (handled correctly, but slowly on some hardware
and flushed by some backends' estimate-tier math — see
[Vec.md](Vec.md#math-vecmathh)). `bfloat16_t` has `float`'s range and no
such cliff.

**Alignment is 2 bytes** (`alignas(2)`): arrays of wrappers pack tightly;
do not expect f32-style alignment, and use `vec::memory_alignment(tag)` for
aligned vector accesses rather than assumptions about the element.
