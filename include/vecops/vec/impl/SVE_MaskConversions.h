//
// SVE_MaskConversions.h — SVE mask promote/demote/convert operations
//

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
          TL_IF(sizeof(TypeOf<To>) == sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  return Mask<To>{mi};
}

/* =================================================================== */
/*              promote: 2x widen (e.g. b16 -> b32)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 2 * sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  // svunpklo_b takes the lower half of the bN predicate and widens
  // it to b(2N) granularity. For scalable tags, sizeof(To)==2*sizeof(Ti)
  // means N_output == N_input/2, so svunpklo_b gives exactly N_output elements.
  return svunpklo_b(mi);
}

/* =================================================================== */
/*              promote: 4x widen (e.g. b16 -> b64)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 4 * sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  using TmElem = std::conditional_t<sizeof(TypeOf<Ti>) == 1, int16_t, int32_t>;
  Tag<TmElem, Ti::N, Ti::POW2> t_mid;
  auto m_mid = word::promote(t_mid, ti, mi);
  return word::promote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              promote: 8x widen (e.g. b8 -> b64)                       */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) == 8 * sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> promote(To to, Ti ti, Mask<Ti> mi) {
  using TmElem = std::conditional_t<sizeof(TypeOf<Ti>) == 1, int32_t, void>;
  Tag<TmElem, Ti::N, Ti::POW2> t_mid;
  auto m_mid = word::promote(t_mid, ti, mi);
  return word::promote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              demote: 2x narrow (e.g. b32 -> b16)                      */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) * 2 == sizeof(TypeOf<Ti>)),
          TL_IF(num_words(To{}) == 1 && num_words(Ti{}) == 1)>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
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
  using TmElem = std::conditional_t<sizeof(TypeOf<To>) == 1, int16_t, int32_t>;
  Tag<TmElem, Ti::N, Ti::POW2> t_mid;
  auto m_mid = word::demote(t_mid, ti, mi);
  return word::demote(to, t_mid, m_mid);
}

/* =================================================================== */
/*              demote: 8x narrow (e.g. b64 -> b8)                       */
/* =================================================================== */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti),
          TL_IF(sizeof(TypeOf<To>) * 8 == sizeof(TypeOf<Ti>))>
VECOPS_VFUNC Mask<To> demote(To to, Ti ti, Mask<Ti> mi) {
  Tag<int32_t, Ti::N, Ti::POW2> t_mid;
  auto m_mid = word::demote(t_mid, ti, mi);
  return word::demote(to, t_mid, m_mid);
}

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_SVE_MASK_CONVERSIONS_H
