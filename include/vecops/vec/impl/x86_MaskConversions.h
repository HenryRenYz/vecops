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

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> convert(To, Ti, Mask<Ti> mi) { return Mask<To>{mi.v}; }

#else // !HAS_AVX512DQ — vector register masks

template <typename Reg>
static constexpr bool is_256 = (sizeof(Reg) == 32);

// Extract lower 128 bits regardless of 128/256-bit input
template <typename Mi>
VECOPS_VFUNC __m128i low128(Mi&& mi) {
  if constexpr (is_256<decltype(mi.v)>) return _mm256_castsi256_si128(mi.v);
  else return mi.v;
}
template <typename Mi>
VECOPS_VFUNC __m128i high128(Mi&& mi) {
  return _mm256_extracti128_si256(mi.v, 1);
}

// Output 256-bit from two 128-bit halves
template <typename M>
VECOPS_VFUNC M make_m256(__m128i lo, __m128i hi) {
  __m256i r = _mm256_castsi128_si256(lo);
  r = _mm256_inserti128_si256(r, hi, 1);
  return M{r};
}

/* =================================================================== */
/*              promote: int8 -> int16                                  */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    return Mask<To>{_mm256_cvtepi8_epi16(low128(mi))};
  } else {
    return Mask<To>{_mm_cvtepi8_epi16(mi.v)};
  }
}

/* =================================================================== */
/*              promote: int8 -> int32                                  */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    auto step1 = _mm256_cvtepi8_epi16(low128(mi));
    return Mask<To>{_mm256_cvtepi16_epi32(_mm256_castsi256_si128(step1))};
  } else {
    auto step1 = _mm_cvtepi8_epi16(mi.v);
    return Mask<To>{_mm_cvtepi16_epi32(step1)};
  }
}

/* =================================================================== */
/*              promote: int8 -> int64                                  */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    auto step1 = _mm256_cvtepi8_epi16(low128(mi));
    auto step2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(step1));
    return Mask<To>{_mm256_cvtepi32_epi64(_mm256_castsi256_si128(step2))};
  } else {
    auto step1 = _mm_cvtepi8_epi16(mi.v);
    auto step2 = _mm_cvtepi16_epi32(step1);
    return Mask<To>{_mm_cvtepi32_epi64(step2)};
  }
}

/* =================================================================== */
/*              promote: int16 -> int32                                 */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    return Mask<To>{_mm256_cvtepi16_epi32(low128(mi))};
  } else {
    return Mask<To>{_mm_cvtepi16_epi32(mi.v)};
  }
}

/* =================================================================== */
/*              promote: int16 -> int64                                 */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    auto step1 = _mm256_cvtepi16_epi32(low128(mi));
    return Mask<To>{_mm256_cvtepi32_epi64(_mm256_castsi256_si128(step1))};
  } else {
    auto step1 = _mm_cvtepi16_epi32(mi.v);
    return Mask<To>{_mm_cvtepi32_epi64(step1)};
  }
}

/* =================================================================== */
/*              promote: int32 -> int64                                 */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (To::Bytes >= 32) {
    return Mask<To>{_mm256_cvtepi32_epi64(low128(mi))};
  } else {
    return Mask<To>{_mm_cvtepi32_epi64(mi.v)};
  }
}

/* =================================================================== */
/*              demote: int16 -> int8                                    */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (is_256<decltype(mi.v)>) {
    if constexpr (To::Bytes >= 32) {
      auto lo = _mm_packs_epi16(low128(mi), _mm_setzero_si128());
      auto hi = _mm_packs_epi16(high128(mi), _mm_setzero_si128());
      auto packed = _mm_unpacklo_epi64(lo, hi);
      return Mask<To>{_mm256_castsi128_si256(packed)};
    } else {
      return Mask<To>{_mm_packs_epi16(low128(mi), _mm_setzero_si128())};
    }
  } else {
    return Mask<To>{_mm_packs_epi16(mi.v, _mm_setzero_si128())};
  }
}

/* =================================================================== */
/*              demote: int32 -> int16                                   */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (is_256<decltype(mi.v)>) {
    if constexpr (To::Bytes >= 32) {
      auto lo = _mm_packs_epi32(low128(mi), _mm_setzero_si128());
      auto hi = _mm_packs_epi32(high128(mi), _mm_setzero_si128());
      auto packed = _mm_unpacklo_epi64(lo, hi);
      return Mask<To>{_mm256_castsi128_si256(packed)};
    } else {
      return Mask<To>{_mm_packs_epi32(low128(mi), _mm_setzero_si128())};
    }
  } else {
    return Mask<To>{_mm_packs_epi32(mi.v, _mm_setzero_si128())};
  }
}

/* =================================================================== */
/*              demote: int64 -> int32                                   */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (is_256<decltype(mi.v)>) {
    __m256i shifted = _mm256_srli_epi64(mi.v, 32);
    auto r256 = _mm256_permutevar8x32_epi32(shifted,
        _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
    if constexpr (To::Bytes >= 32) {
      return Mask<To>{r256};
    } else {
      return Mask<To>{_mm256_castsi256_si128(r256)};
    }
  } else {
    __m128i shifted = _mm_srli_epi64(mi.v, 32);
    return Mask<To>{_mm_shuffle_epi32(shifted, _MM_SHUFFLE(0, 0, 2, 0))};
  }
}

/* =================================================================== */
/*              demote chain: multi-step narrow (single-word leaf only)  */
/* =================================================================== */

// int64 -> int16: chain int64->int32 then int32->int16
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, To::N, To::POW2> t32;
  return word::demote(to, t32, word::demote(t32, ti, mi));
}

// int64 -> int8: chain int64->int32 then int32->int8
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, To::N, To::POW2> t32;
  return word::demote(to, t32, word::demote(t32, ti, mi));
}

// int32 -> int8: chain int32->int16 then int16->int8
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int16_t, To::N, To::POW2> t16;
  return word::demote(to, t16, word::demote(t16, ti, mi));
}

/* =================================================================== */
/*              convert: identity                                        */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{mi.v};
}

#endif // !HAS_AVX512DQ

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_X86_MASK_CONVERSIONS_H
