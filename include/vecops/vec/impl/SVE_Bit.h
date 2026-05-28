//
// SVE_Bit.h — SVE real implementations for bitwise operations
//

#ifndef VECOPS_SVE_BIT_H
#define VECOPS_SVE_BIT_H

#include <arm_sve.h>

#include "CoreDefs.h"
#include "../VecBase.h"
#include "SVE_Basic.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* === bit_and === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_and(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return svand_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return svand_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return svand_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return svand_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return svand_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return svand_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return svand_u64_x(pg, a, b);
  else return svand_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_and(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_and(a, b));
}

/* === bit_or === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_or(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return svorr_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return svorr_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return svorr_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return svorr_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return svorr_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return svorr_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return svorr_u64_x(pg, a, b);
  else return svorr_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_or(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_or(a, b));
}

/* === bit_xor === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_xor(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return sveor_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)   return sveor_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>) return sveor_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)  return sveor_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>) return sveor_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)  return sveor_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>) return sveor_u64_x(pg, a, b);
  else return sveor_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_xor(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_xor(a, b));
}

/* === bit_andnot === */
/* bic: b & ~a, so we swap arguments: bit_andnot(a, b) = b & ~a */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_andnot(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return svbic_u8_x(pg, b, a);
  else if constexpr (std::is_same_v<E, int8_t>)   return svbic_s8_x(pg, b, a);
  else if constexpr (std::is_same_v<E, uint16_t>) return svbic_u16_x(pg, b, a);
  else if constexpr (std::is_same_v<E, int16_t>)  return svbic_s16_x(pg, b, a);
  else if constexpr (std::is_same_v<E, uint32_t>) return svbic_u32_x(pg, b, a);
  else if constexpr (std::is_same_v<E, int32_t>)  return svbic_s32_x(pg, b, a);
  else if constexpr (std::is_same_v<E, uint64_t>) return svbic_u64_x(pg, b, a);
  else return svbic_s64_x(pg, b, a);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_andnot(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_andnot(a, b));
}

/* === bit_not === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_not(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, uint8_t>)       return svnot_u8_x(pg, v);
  else if constexpr (std::is_same_v<E, int8_t>)   return svnot_s8_x(pg, v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svnot_u16_x(pg, v);
  else if constexpr (std::is_same_v<E, int16_t>)  return svnot_s16_x(pg, v);
  else if constexpr (std::is_same_v<E, uint32_t>) return svnot_u32_x(pg, v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svnot_s32_x(pg, v);
  else if constexpr (std::is_same_v<E, uint64_t>) return svnot_u64_x(pg, v);
  else return svnot_s64_x(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_not(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::bit_not(v));
}

/* === bit_shl (runtime shift) === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint8_t>();
  return svlsl_n_u8_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int8_t>();
  return svlsl_n_s8_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint16_t>();
  return svlsl_n_u16_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int16_t>();
  return svlsl_n_s16_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint32_t>();
  return svlsl_n_u32_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int32_t>();
  return svlsl_n_s32_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint64_t>();
  return svlsl_n_u64_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V bit_shl(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int64_t>();
  return svlsl_n_s64_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shl(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shl(v, shift));
}

/* === bit_shr (runtime shift) === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint8_t>();
  return svlsr_n_u8_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int8_t>();
  return svasr_n_s8_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint16_t>();
  return svlsr_n_u16_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int16_t>();
  return svasr_n_s16_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint32_t>();
  return svlsr_n_u32_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int32_t>();
  return svasr_n_s32_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<uint64_t>();
  return svlsr_n_u64_x(pg, v, shift);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V bit_shr(V v, int shift) {
  auto pg = sve_detail::sve_ptrue<int64_t>();
  return svasr_n_s64_x(pg, v, shift);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC V bit_shr(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shr(v, shift));
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_BIT_H
