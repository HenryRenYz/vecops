//
// SVE_Bit.h — SVE real implementations for bitwise operations
//

#ifndef VECOPS_SVE_BIT_H
#define VECOPS_SVE_BIT_H

#include <arm_sve.h>

#include "CoreTypes.h"
#include "../VecBase.h"
#include "SVE_Basic.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* === bit_and === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_and(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return svand_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return svand_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return svand_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return svand_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return svand_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return svand_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return svand_u64_m(m, a, b);
  else return svand_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_and(V a, V b) {
  using E = TypeOf<T>;
  return word::bit_and(a, b, sve_detail::sve_ptrue<E>());
}

/* === bit_or === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_or(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return svorr_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return svorr_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return svorr_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return svorr_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return svorr_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return svorr_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return svorr_u64_m(m, a, b);
  else return svorr_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_or(V a, V b) {
  using E = TypeOf<T>;
  return word::bit_or(a, b, sve_detail::sve_ptrue<E>());
}

/* === bit_xor === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_xor(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return sveor_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return sveor_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return sveor_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return sveor_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return sveor_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return sveor_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return sveor_u64_m(m, a, b);
  else return sveor_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_xor(V a, V b) {
  using E = TypeOf<T>;
  return word::bit_xor(a, b, sve_detail::sve_ptrue<E>());
}

/* === bit_andnot === */
/* bic: b & ~a, so we swap arguments: bit_andnot(a, b) = b & ~a */
/* Note: cannot use _m here because bic's merge value is the first operand of
   the bic instruction (b), but the masked bit_andnot contract requires merging
   from a.  We use _x + blend instead. */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_andnot(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return word::blend(a, m, svbic_u8_x(pg, b, a));
  else if constexpr (std::is_same_v<E, int8_t>)   return word::blend(a, m, svbic_s8_x(pg, b, a));
  else if constexpr (std::is_same_v<E, uint16_t>) return word::blend(a, m, svbic_u16_x(pg, b, a));
  else if constexpr (std::is_same_v<E, int16_t>)  return word::blend(a, m, svbic_s16_x(pg, b, a));
  else if constexpr (std::is_same_v<E, uint32_t>) return word::blend(a, m, svbic_u32_x(pg, b, a));
  else if constexpr (std::is_same_v<E, int32_t>)  return word::blend(a, m, svbic_s32_x(pg, b, a));
  else if constexpr (std::is_same_v<E, uint64_t>) return word::blend(a, m, svbic_u64_x(pg, b, a));
  else return word::blend(a, m, svbic_s64_x(pg, b, a));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_andnot(V a, V b) {
  using E = TypeOf<T>;
  return word::bit_andnot(a, b, sve_detail::sve_ptrue<E>());
}

/* === bit_not === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_not(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return svnot_u8_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int8_t>)   return svnot_s8_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svnot_u16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int16_t>)  return svnot_s16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint32_t>) return svnot_u32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svnot_s32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint64_t>) return svnot_u64_m(default_v, m, v);
  else return svnot_s64_m(default_v, m, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_not(V v) {
  using E = TypeOf<T>;
  return word::bit_not(v, sve_detail::sve_ptrue<E>(), v);
}

/* === bit_shl (runtime shift) === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shl(V v, int shift, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return svlsl_n_u8_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int8_t>)   return svlsl_n_s8_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint16_t>) return svlsl_n_u16_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int16_t>)  return svlsl_n_s16_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint32_t>) return svlsl_n_u32_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int32_t>)  return svlsl_n_s32_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint64_t>) return svlsl_n_u64_m(m, v, shift);
  else return svlsl_n_s64_m(m, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  using E = TypeOf<T>;
  return word::bit_shl(v, shift, sve_detail::sve_ptrue<E>());
}

/* === bit_shr (runtime shift) === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shr(V v, int shift, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, uint8_t>)       return svlsr_n_u8_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int8_t>)   return svasr_n_s8_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint16_t>) return svlsr_n_u16_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int16_t>)  return svasr_n_s16_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint32_t>) return svlsr_n_u32_m(m, v, shift);
  else if constexpr (std::is_same_v<E, int32_t>)  return svasr_n_s32_m(m, v, shift);
  else if constexpr (std::is_same_v<E, uint64_t>) return svlsr_n_u64_m(m, v, shift);
  else return svasr_n_s64_m(m, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  using E = TypeOf<T>;
  return word::bit_shr(v, shift, sve_detail::sve_ptrue<E>());
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_BIT_H
