# Mask Tag Conversion Design

**Date:** 2026-06-13
**Status:** Approved
**Topic:** promote/demote/convert functions for `Mask<Tag>` across different element-type Tags

## 1. Motivation

The codebase already has `promote`/`demote`/`convert` for `Vec<Tag>` (vector data). However, `Mask<Tag>` (predicate/mask for those vectors) lacks equivalent cross-tag conversion functions. When masks need to be used with vectors of a different element type (e.g., applying a `float32` mask to `float16` operations), the mask register content must be reinterpreted at the target element width.

## 2. Semantics

**Same element count:** The number of mask elements (`N`) stays the same; only the per-element lane width changes. Examples:

- `Mask<float32_t, 8>` -> `promote` -> `Mask<float64_t, 8>`: each 32-bit mask lane (0xFFFFFFFF for true) widens to 64-bit (0xFFFFFFFFFFFFFFFF).
- `Mask<float32_t, 8>` -> `demote` -> `Mask<float16_t, 8>`: each 32-bit mask lane (0xFFFFFFFF) narrows to 16-bit (0xFFFF).
- `Mask<float32_t, 8>` -> `convert` -> `Mask<int32_t, 8>`: identity (same byte size).

**Constraint:** `sizeof(To)` vs `sizeof(Ti)` determines whether `promote` (`sizeof(To) > sizeof(Ti)`), `demote` (`sizeof(To) < sizeof(Ti)`), or `convert` (`sizeof(To) == sizeof(Ti)`) is invoked. SFINAE enforces this.

## 3. Function Signatures

Three-parameter form: both output tag `To` and input tag `Ti` are passed explicitly as tag objects, because `Mask<Tag>` does not carry enough type information to reliably reconstruct the original tag (unlike `Vec<Tag>` where `Vec2Tag` can extract it).

```cpp
// Promote: widen mask lanes  (e.g., int16 -> int32, float16 -> float32)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>))
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi);

// Demote: narrow mask lanes (e.g., int32 -> int16, float32 -> float16)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>))
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi);

// Convert: identity (e.g., int32 -> float32, same byte size)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) { return {mi}; }
```

## 4. File Layout

| File | Role |
|------|------|
| `include/vecops/vec/impl/x86_MaskConversions.h` | x86 `word::promote`/`word::demote`/`word::convert` single-word impls |
| `include/vecops/vec/impl/SVE_MaskConversions.h` | SVE `word::promote`/`word::demote`/`word::convert` single-word impls |
| `include/vecops/vec/Vec.h` | Top-level `vec::promote`/`vec::demote`/`vec::convert` with `num_words` multi-word dispatch |
| `include/vecops/vec/impl/Scalar.h` | Scalar `word::promote`/`word::demote`/`word::convert` fallback (all identity) |
| `tests/vec/MaskConversionsTest.cpp` | Unit tests for representative type pairs |

## 5. Architecture-Specific Strategy

### 5.1 x86 without AVX512DQ

Mask registers are `__m128i` or `__m256i` with per-element all-1s (true) or all-0s (false). Conversion is an integer lane-width change:

- **promote (widen):** Sign-extend using `_mm_cvtepi{8,16,32}_epi{16,32,64}` (SSE4.1). Mask values of 0 (all zeros) and -1 (all ones) are preserved perfectly. For AVX2 256-bit: `_mm256_*` variants.
  - int8->int16: `_mm_cvtepi8_epi16`
  - int8->int32: `_mm_cvtepi8_epi32`
  - int8->int64: `_mm_cvtepi8_epi64`
  - int16->int32: `_mm_cvtepi16_epi32`
  - int16->int64: `_mm_cvtepi16_epi64`
  - int32->int64: `_mm_cvtepi32_epi64`
  - Float masks treated as their integer-view equivalent.

- **demote (narrow):** Saturating pack using `_mm_packs_epi{16,32}`. -1 saturated to int16 = -1 (0xFFFF); 0 = 0. For non-doubling narrowing (e.g., int64->int32->int16), chain packs or use `_mm_cvtepi64_epi32` for int64->int32.
  - int16->int8: `_mm_packs_epi16`
  - int32->int16: `_mm_packs_epi32`
  - int64->int32: `_mm_cvtepi64_epi32` (AVX2)
  - int64->int16: int64->int32 + int32->int16
  - int64->int8: int64->int32 + int32->int16 + int16->int8

- **convert:** identity `reinterpret_cast`.

### 5.2 x86 with AVX512DQ

Mask registers are `__mmask{8,16,32,64}` -- 1 bit per element.

**Single-word (fits in one 512-bit register):** Same element count means identical bit positions in the `__mmask`, so **all three operations are identity**:

```cpp
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
Mask<To> promote(To to, Ti ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
Mask<To> demote(To to, Ti ti, Mask<Ti> mi)  { return Mask<To>{mi.v}; }
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
Mask<To> convert(To to, Ti ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }
```

**Multi-word (output exceeds 512-bit, e.g. Mask<float64, 16>):** Falls back to the multi-word dispatch in Vec.h (see Section 6), which splits into per-word masks and reassembles. Per-word conversions are still identity.

### 5.3 ARM SVE

Predicate register `svbool_t` stores 1 bit per byte of vector length. Conversion between granularities follows a systematic approach based on the byte-size ratio `R = sizeof(To) / sizeof(Ti)`:

**promote (R > 1, e.g. b16->b32, b16->b64):**

For a 2x widen (e.g. b16->b32, R=2): apply `svunpklo_b`/`svunpkhi_b` to double granularity, then interleave with `svzip1` to produce same-element-count output. Each input predicate bit at position 2k maps to output position 4k.

For a 4x widen (e.g. b16->b64, R=4): chain two 2x widen steps (b16->b32 then b32->b64).

For other ratios (e.g. b8->b64, R=8): chain three 2x steps (b8->b16->b32->b64).

**demote (R < 1, e.g. b32->b16, b64->b16):**

For a 2x narrow (e.g. b32->b16): use `svuzp1_b16`/`svuzp2_b16` to extract every other b32 predicate bit into b16 positions.

For a 4x narrow (e.g. b64->b16): chain two 2x narrow steps (b64->b32 then b32->b16).

For other ratios: chain 2x steps as needed.

**convert (R == 1):** identity.

**Multi-word SVE masks:** Use `get_word_mask<Tag>()` / `set_word_mask<Tag>()` for word extraction (NOT `svget2` directly, to avoid clang compile issues on 920f when the tuple subscript is not constant-folded).

SVE2p1 `svget2`/`svget4` are only used internally within the already-existing `VecDefs::get_mask`/`set_mask` infrastructure, not called directly in the conversion code.

### 5.4 Scalar

Identity for all cases (same element count means same bitset content, just reinterpreted).

## 6. Multi-Word Dispatch

Follow the exact same pattern as the vector `promote`/`demote`/`convert` in `Vec.h` (lines 2452-2516). The arch-specific implementations in `x86_MaskConversions.h` and `SVE_MaskConversions.h` define only single-word `word::promote`/`word::demote`/`word::convert`. The top-level `vec::promote`/`vec::demote`/`vec::convert` in `Vec.h` handle multi-word dispatch using the existing `num_words()` + `lower`/`upper`/`concat` pattern — exactly like vectors:

```cpp
// Mask promote — multi-word dispatch using num_words + lower/upper/concat
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (num_words(ti) > 1 && num_words(to) > 1) {
    Half<To> t_ho;
    Half<Ti> t_hi;
    auto lo = promote(t_ho, t_hi, lower(ti, mi));
    auto hi = promote(t_ho, t_hi, upper(ti, mi));
    return concat(to, lo, hi);
  } else {
    return word::promote(to, ti, mi);
  }
}

// demote — same pattern
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (num_words(ti) > 1 && num_words(to) > 1) {
    Half<To> t_ho;
    Half<Ti> t_hi;
    auto lo = demote(t_ho, t_hi, lower(ti, mi));
    auto hi = demote(t_ho, t_hi, upper(ti, mi));
    return concat(to, lo, hi);
  } else {
    return word::demote(to, ti, mi);
  }
}

// convert — uses vmap + ShardMask (like vector convert uses vmap + ShardVec)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  using namespace details;
  return vmap(
      to, [=](auto tt, auto&& mm) { return word::convert(tt, ti, mm); },
      ShardMask(ti, mi)
  );
}
```

Key points:
1. **`num_words(ti) > 1 && num_words(to) > 1`** — Same check as the vector version. No `is_runtime_size` or `POW2` checks.
2. **Single-word delegates to `word::promote`** (not `CPU_CAPABILITY::promote`). The `word` namespace is already inside the `CPU_CAPABILITY` namespace — same as how vector conversions work.
3. **`lower`/`upper` on masks** already exist in `Vec.h` (lines 2046-2090) and handle multi-word via `num_words` internally.
4. **`concat` on masks** already exists in `Vec.h` (lines 2112-2130).
5. **`ShardMask`** already exists in `VectorizedUtil.h` for the `vmap`-based convert dispatch.

## 7. Element Type Coverage

All 12 element types are supported: `bfloat16_t`, `float16_t`, `float32_t`, `float64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.

Since only element byte size matters for mask conversion (all-1s / all-0s patterns are type-agnostic), the implementation is generic over the element size ratio rather than explicit per-type-pair specializations (Approach 2: Generic Element-Size Dispatch).

## 8. Testing

### 8.1 Local x86 tests

`tests/vec/MaskConversionsTest.cpp` tests all size-ratio groups:

- `convert`: all same-size pairs (int32<->float32, int64<->float64, int16<->bf16, etc.)
- `promote`: int8->int16, int8->int32, int8->int64, int16->int32, int16->int64, int32->int64, float16->float32, float16->float64, bf16->float32, bf16->float64, float32->float64
- `demote`: inverse of above
- Multi-word cases (e.g., Mask<int8, 64> <-> Mask<int16, 64> which requires 2-word mask on AVX2)

### 8.2 Remote SVE tests (920f)

`ssh 920f` to compile and run tests with clang. Use the same test file compiled with SVE flags.

## 9. Implementation Notes

- The x86 without AVX512DQ promote path uses signed extension intrinsics. Since mask values are only 0x00...00 (zero) or 0xFF...FF (-1), both signed and unsigned extension produce correct results. We use signed extension for simplicity and because it's available in SSE4.1 baseline.
- For the demote path, `_mm_packs_epi32(lo, hi)` saturates -1 to -1 (0xFFFF) and 0 to 0. This works correctly for all mask values.
- The `_mm256` AVX2 variants of pack/extend intrinsics have different lane ordering (`_mm256_packs_epi32` interleaves results from two halves). The implementation must handle this correctly.
- SVE predicate operations (`svunpklo_b`, `svuzp1_b16`, etc.) operate at byte-granularity predicate register semantics. The `svzip1`/`svuzp1` variants with element-type suffixes (e.g., `svuzp1_b16`) operate at the specified element granularity.
- For remote SVE testing on 920f, the test binary should be compiled with `clang++ -march=armv8-a+sve ...`. No qemu or cross-compilation is needed since 920f is a native ARM machine.
