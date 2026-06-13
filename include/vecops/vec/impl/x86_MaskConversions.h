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
  return Mask<To>{_mm_cvtepi8_epi16(mi.v)};
}

// int8 -> int32
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi8_epi32(mi.v)};
}

// int8 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi8_epi64(mi.v)};
}

// int16 -> int32
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi16_epi32(mi.v)};
}

// int16 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi16_epi64(mi.v)};
}

// int32 -> int64
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi32_epi64(mi.v)};
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
  return Mask<To>{_mm_cvtepi64_epi32(mi.v)};
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
  return Mask<To>{mi.v};
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
