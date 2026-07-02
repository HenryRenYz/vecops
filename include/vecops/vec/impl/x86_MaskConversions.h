//
// x86_MaskConversions.h — x86 mask promote/demote/convert operations
//

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
/*              AVX512DQ (__mmask) — all identity                       */
/* =================================================================== */

#ifdef HAS_AVX512DQ

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> copy_mask_bits(To, Ti, Mask<Ti> mi) {
  using RawMask = decltype(Mask<To>{}.v);
  return Mask<To>{static_cast<RawMask>(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  return copy_mask_bits(to, ti, mi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  return copy_mask_bits(to, ti, mi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  return copy_mask_bits(to, ti, mi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1),
          TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h;
  Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1),
          TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h;
  Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h;
  Half<Ti> t_i_h;
  auto lo = word::convert(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::convert(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

#else // !HAS_AVX512DQ — vector register masks

/* =================================================================== */
/*                      8-bit  <=> 16-bit                               */
/* =================================================================== */

// ---- promote: int8 -> int16 (and uint8 -> uint16) -------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) <= 8)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi8_epi16(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 16)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi8_epi16(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

// ---- demote: int16 -> int8 (and uint16 -> uint8) --------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) <= 8)>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_packs_epi16(mi.v, _mm_setzero_si128())};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 16)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<Ti> t_i_h;
  auto lo = _mm_packs_epi16(word::lower(ti, mi).v, _mm_setzero_si128());
  auto hi = _mm_packs_epi16(word::upper(ti, mi).v, _mm_setzero_si128());
  auto packed = _mm_unpacklo_epi64(lo, hi);
  return Mask<To>{packed};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*                      8-bit  <=> 32-bit                               */
/* =================================================================== */

// ---- promote: int8 -> int32 (and uint8 -> uint32) -------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) <= 4)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi8_epi32(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 8)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi8_epi32(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*                      8-bit  <=> 64-bit                               */
/* =================================================================== */

// ---- promote: int8 -> int64 (and uint8 -> uint64) -------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) <= 2)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi8_epi64(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 4)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi8_epi64(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 1),
          TL_IF(size(To{}) == 8 || size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*                     16-bit  <=> 32-bit                               */
/* =================================================================== */

// ---- promote: int16 -> int32 (and uint16 -> uint32) -----------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) <= 4)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi16_epi32(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 8)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi16_epi32(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

// ---- demote: int32 -> int16 (and uint32 -> uint16) ------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) <= 4)>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_packs_epi32(mi.v, _mm_setzero_si128())};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) == 8)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<Ti> t_i_h;
  auto lo = _mm_packs_epi32(word::lower(ti, mi).v, _mm_setzero_si128());
  auto hi = _mm_packs_epi32(word::upper(ti, mi).v, _mm_setzero_si128());
  auto packed = _mm_unpacklo_epi64(lo, hi);
  return Mask<To>{packed};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*                     16-bit  <=> 64-bit                               */
/* =================================================================== */

// ---- promote: int16 -> int64 (and uint16 -> uint64) -----------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) <= 2)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi16_epi64(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 4)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi16_epi64(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 2),
          TL_IF(size(To{}) == 8 || size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*                     32-bit  <=> 64-bit                               */
/* =================================================================== */

// ---- promote: int32 -> int64 (and uint32 -> uint64) -----------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) <= 2)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm_cvtepi32_epi64(mi.v)};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) == 4)>
VECOPS_VFUNC Mask<To> promote(To, Ti, Mask<Ti> mi) {
  return Mask<To>{_mm256_cvtepi32_epi64(mi.v)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(size(To{}) == 8 || size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

// ---- demote: int64 -> int32 (and uint64 -> uint32) ------------------

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(size(To{}) <= 2)>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) {
  __m128i shifted = _mm_srli_epi64(mi.v, 32);
  return Mask<To>{_mm_shuffle_epi32(shifted, _MM_SHUFFLE(0, 0, 2, 0))};
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(size(To{}) == 4)>
VECOPS_VFUNC Mask<To> demote(To, Ti, Mask<Ti> mi) {
  __m256i shifted = _mm256_srli_epi64(mi.v, 32);
  auto r256 = _mm256_permutevar8x32_epi32(shifted,
      _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
  return Mask<To>{_mm256_castsi256_si128(r256)};
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(size(To{}) == 8 || size(To{}) == 16 || size(To{}) == 32 || size(To{}) == 64)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*              demote chains (multi-step narrow)                        */
/* =================================================================== */

// int32 -> int8: chain int32->int16 then int16->int8 (single-word)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(num_words(Ti{}) <= 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int16_t, To::N, To::POW2> t16;
  return word::demote(to, t16, word::demote(t16, ti, mi));
}

// int64 -> int8: chain int64->int32 then int32->int8 (single-word)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(num_words(Ti{}) <= 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, To::N, To::POW2> t32;
  return word::demote(to, t32, word::demote(t32, ti, mi));
}

// int64 -> int16: chain int64->int32 then int32->int16 (single-word)
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(num_words(Ti{}) <= 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, To::N, To::POW2> t32;
  return word::demote(to, t32, word::demote(t32, ti, mi));
}

// Multi-word chain variants (VEC >= 256 only — generic fallback handles VEC < 256)
#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 4),
          TL_IF(num_words(Ti{}) > 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 1 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(num_words(Ti{}) > 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 && sizeof(TypeOf<Ti>) == 8),
          TL_IF(num_words(Ti{}) > 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH >= 256

/* =================================================================== */
/*              convert: identity for same byte size                     */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> convert(To, Ti, Mask<Ti> mi) {
  return Mask<To>{mi.v};
}

/* =================================================================== */
/*              Generic multi-word fallback (VEC_WIDTH < 256)            */
/* =================================================================== */

#if VEC_WIDTH < 256
template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1),
          TL_IF(sizeof(TypeOf<To>) > sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  if constexpr (Ti::Bytes < 16) {
    // Sub-word input: promote to intermediate standard-word type, then split
    using TmElem = std::conditional_t<sizeof(TypeOf<Ti>) == 1, int16_t, int32_t>;
    Tag<TmElem, Ti::N, Ti::POW2> t_mid;
    auto m_mid = word::promote(t_mid, ti, mi);
    Half<To> t_h;
    Half<decltype(t_mid)> t_mid_h;
    auto lo = word::promote(t_h, t_mid_h, word::lower(t_mid, m_mid));
    auto hi = word::promote(t_h, t_mid_h, word::upper(t_mid, m_mid));
    return word::concat(to, lo, hi);
  } else {
    Half<To> t_h; Half<Ti> t_i_h;
    auto lo = word::promote(t_h, t_i_h, word::lower(ti, mi));
    auto hi = word::promote(t_h, t_i_h, word::upper(ti, mi));
    return word::concat(to, lo, hi);
  }
}

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(num_words(To{}) > 1 || num_words(Ti{}) > 1),
          TL_IF(sizeof(TypeOf<To>) < sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Half<To> t_h; Half<Ti> t_i_h;
  auto lo = word::demote(t_h, t_i_h, word::lower(ti, mi));
  auto hi = word::demote(t_h, t_i_h, word::upper(ti, mi));
  return word::concat(to, lo, hi);
}
#endif // VEC_WIDTH < 256

#endif // !HAS_AVX512DQ

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_X86_MASK_CONVERSIONS_H
