# Mask Tag Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `promote`/`demote`/`convert` for `Mask<Tag>` across different element-type Tags on x86 and ARM SVE.

**Architecture:** Three-parameter `(To to, Ti ti, Mask<Ti> mi)` → `Mask<To>`. Arch-specific `word::promote`/`word::demote`/`word::convert` handle single-word; `vec::promote`/`vec::demote`/`vec::convert` in Vec.h use `num_words` + `lower`/`upper`/`concat` for multi-word dispatch, exactly like the existing vector conversion pattern.

**Tech Stack:** C++20, SSE4.1/AVX2/AVX512DQ intrinsics (x86), ARM SVE intrinsics (SVE), GoogleTest.

---

### Task 1: Add scalar `word::promote`/`word::demote`/`word::convert` fallback (identity)

**Files:**
- Modify: `include/vecops/vec/impl/Scalar.h`

**Rationale:** Scalar mask all three are identity since same element count means same bitset.

- [ ] **Step 1: Add identity overloads to Scalar.h**

After the existing vector promote/demote/convert section (~line 1227), add:

```cpp
// ---- Mask promotion / demotion / conversion (all identity in scalar) ----
// Scalar masks store bits in std::bitset; same element count = same N,
// so copy bits between the underlying bitsets which have identical size.

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Mask<To> r;
  using Bits = std::bitset<(size_t)size(ti)>;
  static_cast<Bits&>(r) = static_cast<const Bits&>(mi);
  return r;
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Mask<To> r;
  using Bits = std::bitset<(size_t)size(ti)>;
  static_cast<Bits&>(r) = static_cast<const Bits&>(mi);
  return r;
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  Mask<To> r;
  using Bits = std::bitset<(size_t)size(ti)>;
  static_cast<Bits&>(r) = static_cast<const Bits&>(mi);
  return r;
}
```

- [ ] **Step 2: Build scalar test**

Run: `cd build && cmake --build . --target VecMaskConversionTest-Scalar 2>&1 | head -5`
Expected: target not found yet (test not created), but compilation succeeds.

---

### Task 2: Create x86 `word::promote`/`word::demote`/`word::convert` (single-word)

**Files:**
- Create: `include/vecops/vec/impl/x86_MaskConversions.h`

- [ ] **Step 1: Create x86_MaskConversions.h**

```cpp
#ifndef VECOPS_X86_MASK_CONVERSIONS_H
#define VECOPS_X86_MASK_CONVERSIONS_H

#include "./x86_Basic.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* =================================================================== */
/*              Identity (same tag)                                     */
/* =================================================================== */

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> promote(T to, T ti, Mask<T> mi) { return mi; }

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> demote(T to, T ti, Mask<T> mi) { return mi; }

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> convert(T to, T ti, Mask<T> mi) { return mi; }

/* =================================================================== */
/*              AVX512DQ (mmask) — all identity                         */
/* =================================================================== */

#ifdef HAS_AVX512DQ

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

#else // !HAS_AVX512DQ — vector register masks

/* =================================================================== */
/*              promote: sign-extend lanes                               */
/* =================================================================== */

// int8 -> int16
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi8_epi16(mi.v))};
}

// int8 -> int32
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi8_epi32(mi.v))};
}

// int8 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi8_epi64(mi.v))};
}

// int16 -> int32
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi16_epi32(mi.v))};
}

// int16 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi16_epi64(mi.v))};
}

// int32 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi32_epi64(mi.v))};
}

/* =================================================================== */
/*              demote: saturating pack lanes                            */
/* =================================================================== */

// int16 -> int8
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_packs_epi16(mi.v, _mm_setzero_si128())};
}

// int32 -> int16
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_packs_epi32(mi.v, _mm_setzero_si128())};
}

// int64 -> int32 (using _mm_cvtepi64_epi32 which truncates, works for mask values 0/-1)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  // _mm_cvtepi64_epi32 truncates int64 -> int32. For mask values:
  //   0xFFFFFFFFFFFFFFFF (int64 -1) -> 0xFFFFFFFF (int32 -1) ✓
  //   0x0000000000000000 (int64 0)  -> 0x00000000 (int32 0)   ✓
  return Mask<To>{vecops::bitcast<__m128i>(_mm_cvtepi64_epi32(mi.v))};
}

// int64 -> int16: chain int64->int32 then int32->int16
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, Tag<To>::N, Tag<To>::POW2> t32;
  auto m32 = demote(t32, ti, mi);
  return demote(to, t32, m32);
}

// int64 -> int8: chain int64->int32 then int32->int8
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, Tag<To>::N, Tag<To>::POW2> t32;
  auto m32 = demote(t32, ti, mi);
  return demote(to, t32, m32);
}

// int32 -> int8: chain int32->int16 then int16->int8
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int16_t, Tag<To>::N, Tag<To>::POW2> t16;
  auto m16 = demote(t16, ti, mi);
  return demote(to, t16, m16);
}

/* =================================================================== */
/*              convert: identity                                        */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{vecops::bitcast<__m128i>(mi.v)};
}

#endif // !HAS_AVX512DQ

/* =================================================================== */
/*    Multi-word fallback forward declarations (exclude from word::)     */
/*    These SFINAE out single-word cases so vec:: dispatch handles them   */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> promote(To to, Ti ti, Mask<Ti> mi);

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> demote(To to, Ti ti, Mask<Ti> mi);

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> convert(To to, Ti ti, Mask<Ti> mi);

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_X86_MASK_CONVERSIONS_H
```

- [ ] **Step 2: Include x86_MaskConversions.h from Vec.h**

Add after line 60 (after `#include "./impl/x86_Conversions.h"`):

```cpp
  #include "./impl/x86_MaskConversions.h"
```

- [ ] **Step 3: Build and check compilation**

Run: `cd build && cmake --build . --target VecBasicTest-Scalar 2>&1 | tail -5`
Expected: builds without errors.

---

### Task 3: Create SVE `word::promote`/`word::demote`/`word::convert` (single-word)

**Files:**
- Create: `include/vecops/vec/impl/SVE_MaskConversions.h`

- [ ] **Step 1: Create SVE_MaskConversions.h**

```cpp
#ifndef VECOPS_SVE_MASK_CONVERSIONS_H
#define VECOPS_SVE_MASK_CONVERSIONS_H

#include <arm_sve.h>
#include "./SVE_Basic.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* =================================================================== */
/*              Identity (same tag)                                     */
/* =================================================================== */

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> promote(T to, T ti, Mask<T> mi) { return mi; }

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> demote(T to, T ti, Mask<T> mi) { return mi; }

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> convert(T to, T ti, Mask<T> mi) { return mi; }

/* =================================================================== */
/*              convert: identity for same byte size                     */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{mi};
}

/* =================================================================== */
/*              promote: 2x widen (e.g. b16 -> b32)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 * sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  // svunpklo_b doubles granularity from Ti to intermediate To-size
  // svunpkhi_b does the same for the upper half of Ti bits
  // svzip1 interleaves lo and hi to regain full element count
  auto lo = svunpklo_b(mi);
  auto hi = svunpkhi_b(mi);
  if constexpr (sizeof(TypeOf<To>) == 2) {
    return svzip1_b16(lo, hi);
  } else if constexpr (sizeof(TypeOf<To>) == 4) {
    return svzip1_b32(lo, hi);
  } else { // sizeof == 8
    return svzip1_b64(lo, hi);
  }
}

/* =================================================================== */
/*              promote: 4x widen (e.g. b16 -> b64)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 * sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  // Chain two 2x promotes: Ti -> intermediate (2x) -> To (2x)
  using T_half = Tag<TypeOf<Ti>, Tag<Ti>::N, Tag<Ti>::POW2>;
  // Rebind intermediate to a type with 2x element size
  using TmElem = std::conditional_t<sizeof(TypeOf<Ti>) == 1, int16_t, int32_t>;
  Tag<TmElem, Tag<Ti>::N, Tag<Ti>::POW2> t_mid;
  auto m_mid = promote(t_mid, ti, mi);
  return promote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              promote: 8x widen (e.g. b8 -> b64)                       */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 * sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  using TmElem = std::conditional_t<sizeof(TypeOf<Ti>) == 1, int32_t, void>;
  Tag<TmElem, Tag<Ti>::N, Tag<Ti>::POW2> t_mid;
  auto m_mid = promote(t_mid, ti, mi);
  return promote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              demote: 2x narrow (e.g. b32 -> b16)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) * 2 == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  // svuzp1 extracts every-other predicate bit, halving granularity
  if constexpr (sizeof(TypeOf<Ti>) == 2) {
    return svuzp1_b8(mi, mi);
  } else if constexpr (sizeof(TypeOf<Ti>) == 4) {
    return svuzp1_b16(mi, mi);
  } else { // sizeof == 8
    return svuzp1_b32(mi, mi);
  }
}

/* =================================================================== */
/*              demote: 4x narrow (e.g. b64 -> b16)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) * 4 == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  // Chain two 2x demotes
  using TmElem = std::conditional_t<sizeof(TypeOf<To>) == 1, int16_t, int32_t>;
  Tag<TmElem, Tag<Ti>::N, Tag<Ti>::POW2> t_mid;
  auto m_mid = demote(t_mid, ti, mi);
  return demote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              demote: 8x narrow (e.g. b64 -> b8)                       */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) * 8 == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, Tag<Ti>::N, Tag<Ti>::POW2> t_mid;
  auto m_mid = demote(t_mid, ti, mi);
  return demote(to, t_mid, m_mid);
}

/* =================================================================== */
/*    Multi-word fallback forward declarations (exclude from word::)     */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> promote(To to, Ti ti, Mask<Ti> mi);

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> demote(To to, Ti ti, Mask<Ti> mi);

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1)>
Mask<To> convert(To to, Ti ti, Mask<Ti> mi);

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_SVE_MASK_CONVERSIONS_H
```

- [ ] **Step 2: Include SVE_MaskConversions.h from Vec.h**

Add after line 67 (after `#include "./impl/SVE_Conversions.h"`):

```cpp
    #include "./impl/SVE_MaskConversions.h"
```

---

### Task 4: Add `vec::promote`/`vec::demote`/`vec::convert` to Vec.h (multi-word dispatch)

**Files:**
- Modify: `include/vecops/vec/Vec.h`

- [ ] **Step 1: Add mask promote/demote/convert after xconvert (~line 2532)**

Add after `xconvert` function (after line 2532):

```cpp
/* ************************************************************************** */
//                    Mask Promotion / Demotion / Conversion                   //
/* ************************************************************************** */

/**
 * @brief Promote a mask to a wider element type.
 *
 * Each mask lane is widened to the target element width while preserving
 * the predicate value: all-1s stays all-1s, all-0s stays all-0s.
 * Same element count before and after.
 *
 * @tparam To Target element tag
 * @tparam Ti Source element tag
 * @param to Target tag
 * @param ti Source tag
 * @param mi Input mask
 * @return Promoted mask with wider lanes
 */
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

/**
 * @brief Demote a mask to a narrower element type.
 *
 * Each mask lane is narrowed to the target element width while preserving
 * the predicate value: all-1s stays all-1s, all-0s stays all-0s.
 * Same element count before and after.
 *
 * @tparam To Target element tag
 * @tparam Ti Source element tag
 * @param to Target tag
 * @param ti Source tag
 * @param mi Input mask
 * @return Demoted mask with narrower lanes
 */
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

/**
 * @brief Convert a mask between types of the same element size.
 *
 * Identity operation: same element count, same byte size per element.
 *
 * @tparam To Target element tag
 * @tparam Ti Source element tag
 * @param to Target tag
 * @param ti Source tag
 * @param mi Input mask
 * @return Converted mask
 */
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  using namespace details;
  return vmap(
      to, [=](auto tt, auto&& mm) { return word::convert(tt, ti, mm); },
      ShardMask(ti, mi)
  );
}
```

- [ ] **Step 2: Build and check**

Run: `cd build && cmake --build . --target VecMaskTest-Scalar 2>&1 | tail -5`
Expected: builds without errors.

---

### Task 5: Create MaskConversionsTest.cpp

**Files:**
- Create: `tests/vec/MaskConversionsTest.cpp`

- [ ] **Step 1: Create the test file**

```cpp
//
// MaskConversionsTest.cpp
// Comprehensive test for Mask promote/demote/convert operations
//
// Supports: x86 SSE/AVX/AVX-512, ARM SVE, and scalar fallback.
//

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

// Create a mask with a specific bool pattern
template <typename Tag>
Mask<Tag> make_mask(Tag tt, const std::vector<bool>& pattern) {
  using T = TypeOf<Tag>;
  auto* buf = static_cast<T*>(std::aligned_alloc(DEFAULT_ALIGNMENT,
                                                  (size_t)size(tt) * sizeof(T)));
  for (nint_t i = 0; i < size(tt); ++i) {
    buf[i] = pattern[(size_t)i] ? static_cast<T>(1) : static_cast<T>(0);
  }
  auto v = loadu(tt, buf);
  std::free(buf);
  return cmpeq(v, fill(tt, static_cast<T>(1)));
}

} // namespace

// ============================================================================
// Test fixture — runs on all element types with single-word tag
// ============================================================================

template <typename T>
struct MaskConvFixture : public ::testing::Test {
  using Elem = T;
  ScalableTag<T, 0> t;
  nint_t full_size = size(t);

  void SetUp() override {}
};

using TestedElemTypes = ::testing::Types<
    float32_t, float64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t
>;
TYPED_TEST_SUITE(MaskConvFixture, TestedElemTypes);

// ============================================================================
// convert: same-size identity
// ============================================================================

TYPED_TEST(MaskConvFixture, ConvertIdentity) {
  using T = TypeParam;
  auto& t = this->t;
  nint_t N = this->full_size;

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[(size_t)i] = (i % 2 == 0);

  auto m = make_mask(t, pattern);

  if constexpr (sizeof(T) == 4) {
    ScalableTag<int32_t, 0> t_i32;
    auto r = convert(t_i32, t, m);
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
  } else if constexpr (sizeof(T) == 8) {
    ScalableTag<int64_t, 0> t_i64;
    auto r = convert(t_i64, t, m);
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
  }
}

// ============================================================================
// promote: widen lanes
// ============================================================================

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt16) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N = size(t_i8);

  std::vector<bool> pattern(N);
  pattern[0] = true;
  pattern[1] = false;
  if (N > 2) pattern[2] = true;

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i16, t_i8, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt32) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N = size(t_i8);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 3 == 0);

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i32, t_i8, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt64) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i8);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i & 1) != 0;

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i64, t_i8, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt16ToInt32) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N = size(t_i16);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 2 == 1);

  auto m = make_mask(t_i16, pattern);
  auto r = promote(t_i32, t_i16, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt16ToInt64) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i16);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i < N / 2);

  auto m = make_mask(t_i16, pattern);
  auto r = promote(t_i64, t_i16, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt32ToInt64) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i32);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 4 == 0);

  auto m = make_mask(t_i32, pattern);
  auto r = promote(t_i64, t_i32, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

// ============================================================================
// demote: narrow lanes
// ============================================================================

TYPED_TEST(MaskConvFixture, DemoteInt16ToInt8) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int8_t, 0> t_i8;
  nint_t N = size(t_i16);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 3 == 1);

  auto m = make_mask(t_i16, pattern);
  auto r = demote(t_i8, t_i16, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i8, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt32ToInt16) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N = size(t_i32);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i & 1) != 0;

  auto m = make_mask(t_i32, pattern);
  auto r = demote(t_i16, t_i32, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt64ToInt32) {
  ScalableTag<int64_t, 0> t_i64;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N = size(t_i64);

  std::vector<bool> pattern(N, true);

  auto m = make_mask(t_i64, pattern);
  auto r = demote(t_i32, t_i64, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt32ToInt8) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int8_t, 0> t_i8;
  nint_t N = size(t_i32);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i & 1) == 0;

  auto m = make_mask(t_i32, pattern);
  auto r = demote(t_i8, t_i32, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i8, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt64ToInt16) {
  ScalableTag<int64_t, 0> t_i64;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N = size(t_i64);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 2 == 1);

  auto m = make_mask(t_i64, pattern);
  auto r = demote(t_i16, t_i64, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
}

// ============================================================================
// Edge cases
// ============================================================================

TYPED_TEST(MaskConvFixture, AllFalse) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N = size(t_i8);

  std::vector<bool> pattern(N, false);
  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i32, t_i8, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_FALSE(get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, AllTrue) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i8);

  std::vector<bool> pattern(N, true);
  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i64, t_i8, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, RoundTrip) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i16);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 5 < 2);

  auto m = make_mask(t_i16, pattern);
  auto promoted = promote(t_i64, t_i16, m);
  auto demoted = demote(t_i16, t_i64, promoted);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, demoted, i)) << "at " << i;
}
```

- [ ] **Step 2: Add test to CMakeLists.txt**

After line 139 (`set(VECOPS_VEC_TESTS ...)`), modify to include MaskConversionsTest:

Replace:
```cmake
set(VECOPS_VEC_TESTS BasicTest LoadStoreTest ArithTest ConversionTest ShuffleTest MaskTest)
```
With:
```cmake
set(VECOPS_VEC_TESTS BasicTest LoadStoreTest ArithTest ConversionTest ShuffleTest MaskTest MaskConversionsTest)
```

- [ ] **Step 3: Build and run tests**

Run: `cd build && cmake .. && cmake --build . --target VecMaskConversionsTest-Scalar -j$(nproc) 2>&1 | tail -10`
Run: `./build/VecMaskConversionsTest-Scalar 2>&1`
Expected: all tests PASS.

- [ ] **Step 4: Build and run for all architectures**

Run: `cd build && cmake .. && cmake --build . -j$(nproc) 2>&1 | grep -E "error:|warning:|VecMask" | tail -20`
Run each: `./build/VecMaskConversionsTest-Scalar && ./build/VecMaskConversionsTest-AVX && ./build/VecMaskConversionsTest-AVX2`
Expected: all tests PASS on all configured architectures.

---

### Task 6: Fix compilation issues and verify x86

**Files:**
- Modify: `include/vecops/vec/impl/x86_MaskConversions.h` (if needed)
- Modify: `include/vecops/vec/impl/SVE_MaskConversions.h` (if needed)
- Modify: `include/vecops/vec/Vec.h` (if needed)

- [ ] **Step 1: Handle `vecops::bitcast` availability**

Check if `vecops::bitcast` is available (it's needed for `__m128i` <-> `__m256i` reinterpretation). If `bitcast` is not in scope, use `_mm_castsi128_ps` / `_mm_castps_si128` style `reinterpret_cast` or the `RegMask` constructor directly.

For `__m128i` mask values, we can construct `Mask<To>` via:
```cpp
__m128i result_v = _mm_cvtepi8_epi16(mi.v);
return Mask<To>{result_v};
```
The `Mask<To>::v` member is typed as the underlying register type (e.g., `__m128i`), so direct assignment works.

- [ ] **Step 2: Handle tag construction for intermediate demotions**

In the x86 demote chain specializations (e.g., int64->int16), the intermediate tag construction `Tag<int32_t, Tag<To>::N>` needs to be correct. Verify:
```cpp
Tag<int32_t, Tag<To>::N, Tag<To>::POW2>
```

- [ ] **Step 3: Run full test suite**

Run: `ctest --test-dir build -R VecMaskConversionTest --output-on-failure 2>&1 | tail -30`
Expected: all tests PASS.

---

### Task 7: Remote SVE testing on 920f

- [ ] **Step 1: Copy files to 920f**

Run: `rsync -avz --exclude=build /home/renyz/project/vecops-neo/ 920f:~/vecops-neo/`

- [ ] **Step 2: Build on 920f**

Run: `ssh 920f "cd ~/vecops-neo && mkdir -p build && cd build && cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release && cmake --build . --target VecMaskConversionsTest-SVE -j\$(nproc) 2>&1 | tail -20"`

- [ ] **Step 3: Run SVE tests**

Run: `ssh 920f "cd ~/vecops-neo/build && ./VecMaskConversionsTest-SVE 2>&1"`

- [ ] **Step 4: Debug any SVE-specific failures**

Common SVE issues:
- `svzip1_b16` / `svzip1_b32` / `svzip1_b64` intrinsics availability — check SVE2 requirement
- Predicate type compatibility — `svbool_t` implicitly converts, but verify
- Multi-word tag construction for SVE scalable tags

If `svzip1_b16` etc. require SVE2 and 920f doesn't have it, fallback to manual interleave using `svzip1(p, p)` with reinterpretation, or chain `svunpklo_b` + `svunpkhi_b` differently.
