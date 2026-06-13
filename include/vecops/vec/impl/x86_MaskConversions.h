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

// Helper to determine if a mask register is 256-bit
template <typename Reg>
static constexpr bool is_m256i_v = std::is_same_v<Reg, __m256i>;

/* =================================================================== */
/*              promote: sign-extend lanes                               */
/* =================================================================== */

// Helper for promote: split 256-bit, convert two 128-bit halves, recombine
#define VECOPS_PROMOTE_X86(ToElem, TiElem, SSE_INTR)                          \
  template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),                              \
            TL_IF(sizeof(TypeOf<To>) == sizeof(ToElem) &&                     \
                  sizeof(TypeOf<Ti>) == sizeof(TiElem))>                       \
  VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {                 \
    if constexpr (is_m256i_v<decltype(mi.v)>) {                              \
      auto lo_in  = _mm256_castsi256_si128(mi.v);                            \
      auto hi_in  = _mm256_extracti128_si256(mi.v, 1);                        \
      auto lo_out = SSE_INTR(lo_in);                                        \
      auto hi_out = SSE_INTR(hi_in);                                        \
      __m256i result = _mm256_castsi128_si256(lo_out);                       \
      result = _mm256_inserti128_si256(result, hi_out, 1);                   \
      return Mask<To>{result};                                               \
    } else {                                                                 \
      return Mask<To>{SSE_INTR(mi.v)};                                      \
    }                                                                        \
  }

VECOPS_PROMOTE_X86(int16_t, int8_t,  _mm_cvtepi8_epi16)
VECOPS_PROMOTE_X86(int32_t, int8_t,  _mm_cvtepi8_epi32)
VECOPS_PROMOTE_X86(int64_t, int8_t,  _mm_cvtepi8_epi64)
VECOPS_PROMOTE_X86(int32_t, int16_t, _mm_cvtepi16_epi32)
VECOPS_PROMOTE_X86(int64_t, int16_t, _mm_cvtepi16_epi64)
VECOPS_PROMOTE_X86(int64_t, int32_t, _mm_cvtepi32_epi64)

#undef VECOPS_PROMOTE_X86

/* =================================================================== */
/*              demote: saturating pack lanes                            */
/* =================================================================== */

// Helper for demote with pack intrinsics (128-bit and 256-bit variants)
#define VECOPS_DEMOTE_X86(ToElem, TiElem, PACK_128, PACK_256)                   \
  template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),                              \
            TL_IF(sizeof(TypeOf<To>) == sizeof(ToElem) &&                     \
                  sizeof(TypeOf<Ti>) == sizeof(TiElem))>                       \
  VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {                 \
    if constexpr (is_m256i_v<decltype(mi.v)>) {                              \
      return Mask<To>{PACK_256(mi.v, _mm256_setzero_si256())};              \
    } else {                                                                 \
      return Mask<To>{PACK_128(mi.v, _mm_setzero_si128())};                  \
    }                                                                        \
  }

VECOPS_DEMOTE_X86(int8_t,  int16_t, _mm_packs_epi16,  _mm256_packs_epi16)
VECOPS_DEMOTE_X86(int16_t, int32_t, _mm_packs_epi32,  _mm256_packs_epi32)

// int64 -> int32: mask narrowing (for mask values 0/-1, upper 32 bits carry info)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (is_m256i_v<decltype(mi.v)>) {
    __m256i shifted = _mm256_srli_epi64(mi.v, 32);
    return Mask<To>{_mm256_permutevar8x32_epi32(shifted,
        _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0))};
  } else {
    __m128i shifted = _mm_srli_epi64(mi.v, 32);
    return Mask<To>{_mm_shuffle_epi32(shifted, _MM_SHUFFLE(0, 0, 2, 0))};
  }
}

#undef VECOPS_DEMOTE_X86

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
