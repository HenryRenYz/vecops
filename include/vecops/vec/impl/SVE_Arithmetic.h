//
// SVE_Arithmetic.h — SVE real implementations for arithmetic operations
//

#ifndef VECOPS_SVE_ARITHMETIC_H
#define VECOPS_SVE_ARITHMETIC_H

#include <arm_sve.h>
#include <cmath>

#include "CoreTypes.h"
#include "../VecBase.h"
#include "SVE_Basic.h"
#include "SVE_Bit.h"
#include "SVE_Bf16.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* ================================================================ */
//                                  Add                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svadd_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svadd_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svadd_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svadd_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svadd_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svadd_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svadd_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svadd_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svadd_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svadd_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svadd_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svadd_u64_m(m, a, b);
  else return svadd_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b) {
  using E = TypeOf<T>;
  return word::add(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                  Sub                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svsub_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svsub_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svsub_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svsub_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svsub_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsub_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsub_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsub_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsub_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsub_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsub_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svsub_u64_m(m, a, b);
  else return svsub_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b) {
  using E = TypeOf<T>;
  return word::sub(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                  Mul                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svmul_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmul_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmul_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svmul_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svmul_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmul_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svmul_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmul_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmul_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmul_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmul_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svmul_u64_m(m, a, b);
  else return svmul_s64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b) {
  using E = TypeOf<T>;
  return word::mul(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                  FMA                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svmad_f32_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float64_t>) return svmad_f64_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float16_t>) return svmad_f16_m(m, a, b, c);
  else return word::blend(a, m, word::add(word::mul(a, b), c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c) {
  using E = TypeOf<T>;
  return word::fmadd(a, b, c, details::ptrue<E>());
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svnmsb_f32_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float64_t>) return svnmsb_f64_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float16_t>) return svnmsb_f16_m(m, a, b, c);
  else return word::blend(a, m, word::sub(word::mul(a, b), c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c) {
  using E = TypeOf<T>;
  return word::fmsub(a, b, c, details::ptrue<E>());
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svmsb_f32_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float64_t>) return svmsb_f64_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float16_t>) return svmsb_f16_m(m, a, b, c);
  else return word::blend(a, m, word::sub(c, word::mul(a, b)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c) {
  using E = TypeOf<T>;
  return word::fnmadd(a, b, c, details::ptrue<E>());
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svnmad_f32_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float64_t>) return svnmad_f64_m(m, a, b, c);
  else if constexpr (std::is_same_v<E, float16_t>) return svnmad_f16_m(m, a, b, c);
  else return word::blend(a, m, word::sub(word::sub(a, a), word::add(word::mul(a, b), c)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c) {
  using E = TypeOf<T>;
  return word::fnmsub(a, b, c, details::ptrue<E>());
}

/* ================================================================ */
//                                  Div (float only)                //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svdiv_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svdiv_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svdiv_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svdiv_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svdiv_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else return svdiv_f32_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V div(V a, V b) {
  using E = TypeOf<T>;
  return word::div(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                  Min                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svmin_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmin_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmin_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svmin_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svmin_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, int8_t>)    return svmin_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmin_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmin_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmin_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmin_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmin_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svmin_s64_m(m, a, b);
  else return svmin_u64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b) {
  using E = TypeOf<T>;
  return word::min(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                  Max                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svmax_f32_m(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmax_f64_m(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmax_f16_m(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svmax_f32_m(m_lo, a_lo, b_lo);
    auto r_hi = svmax_f32_m(m_hi, a_hi, b_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, int8_t>)    return svmax_s8_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmax_u8_m(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmax_s16_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmax_u16_m(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmax_s32_m(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmax_u32_m(m, a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svmax_s64_m(m, a, b);
  else return svmax_u64_m(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b) {
  using E = TypeOf<T>;
  return word::max(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                                 Neg                              //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svneg_f32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svneg_f64_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svneg_f16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto default_lo = details::bf16_to_f32_lo(default_v);
    auto default_hi = details::bf16_to_f32_hi(default_v);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svneg_f32_m(default_lo, m_lo, v_lo);
    auto r_hi = svneg_f32_m(default_hi, m_hi, v_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, int8_t>)    return svneg_s8_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint8_t>) { auto s = svreinterpret_s8_u8(v); auto s_def = svreinterpret_s8_u8(default_v); return svreinterpret_u8_s8(svneg_s8_m(s_def, m, s)); }
  else if constexpr (std::is_same_v<E, int16_t>)   return svneg_s16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint16_t>) { auto s = svreinterpret_s16_u16(v); auto s_def = svreinterpret_s16_u16(default_v); return svreinterpret_u16_s16(svneg_s16_m(s_def, m, s)); }
  else if constexpr (std::is_same_v<E, int32_t>)   return svneg_s32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, uint32_t>) { auto s = svreinterpret_s32_u32(v); auto s_def = svreinterpret_s32_u32(default_v); return svreinterpret_u32_s32(svneg_s32_m(s_def, m, s)); }
  else if constexpr (std::is_same_v<E, int64_t>)   return svneg_s64_m(default_v, m, v);
  else { auto s = svreinterpret_s64_u64(v); auto s_def = svreinterpret_s64_u64(default_v); return svreinterpret_u64_s64(svneg_s64_m(s_def, m, s)); }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v) {
  using E = TypeOf<T>;
  return word::neg(v, details::ptrue<E>(), v);
}

/* ================================================================ */
//                                 Abs                              //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svabs_f32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svabs_f64_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svabs_f16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto default_lo = details::bf16_to_f32_lo(default_v);
    auto default_hi = details::bf16_to_f32_hi(default_v);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svabs_f32_m(default_lo, m_lo, v_lo);
    auto r_hi = svabs_f32_m(default_hi, m_hi, v_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else if constexpr (std::is_same_v<E, int8_t>)    return svabs_s8_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svabs_s16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svabs_s32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svabs_s64_m(default_v, m, v);
  else return word::blend(default_v, m, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v) {
  using E = TypeOf<T>;
  return word::abs(v, details::ptrue<E>(), v);
}

/* ================================================================ */
//                                 Sqrt (float only)                //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svsqrt_f32_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svsqrt_f64_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svsqrt_f16_m(default_v, m, v);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto default_lo = details::bf16_to_f32_lo(default_v);
    auto default_hi = details::bf16_to_f32_hi(default_v);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto r_lo = svsqrt_f32_m(default_lo, m_lo, v_lo);
    auto r_hi = svsqrt_f32_m(default_hi, m_hi, v_hi);
    return details::f32x2_to_bf16(r_lo, r_hi);
  }
  else return svsqrt_f32_m(default_v, m, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V sqrt(V v) {
  using E = TypeOf<T>;
  return word::sqrt(v, details::ptrue<E>(), v);
}

/* ================================================================ */
//                             Rsqrt (float only, approximate)     //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return word::blend(default_v, m, svrsqrte_f32(v));
  else if constexpr (std::is_same_v<E, float64_t>) return word::blend(default_v, m, svrsqrte_f64(v));
  else if constexpr (std::is_same_v<E, float16_t>) return word::blend(default_v, m, svrsqrte_f16(v));
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto r_lo = svrsqrte_f32(v_lo);
    auto r_hi = svrsqrte_f32(v_hi);
    auto result = details::f32x2_to_bf16(r_lo, r_hi);
    return word::blend(default_v, m, result);
  }
  else return word::blend(default_v, m, svrsqrte_f32(v));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rsqrt(V v) {
  using E = TypeOf<T>;
  return word::rsqrt(v, details::ptrue<E>(), v);
}

/* ================================================================ */
//                             Rcp (float only, approximate)      //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return word::blend(default_v, m, svrecpe_f32(v));
  else if constexpr (std::is_same_v<E, float64_t>) return word::blend(default_v, m, svrecpe_f64(v));
  else if constexpr (std::is_same_v<E, float16_t>) return word::blend(default_v, m, svrecpe_f16(v));
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto r_lo = svrecpe_f32(v_lo);
    auto r_hi = svrecpe_f32(v_hi);
    auto result = details::f32x2_to_bf16(r_lo, r_hi);
    return word::blend(default_v, m, result);
  }
  else return word::blend(default_v, m, svrecpe_f32(v));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rcp(V v) {
  using E = TypeOf<T>;
  return word::rcp(v, details::ptrue<E>(), v);
}

/* ================================================================ */
//                             Comparisons                          //
/* ================================================================ */
/* cmpeq */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpeq_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpeq_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpeq_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmpeq_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmpeq_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpeq_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpeq_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpeq_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpeq_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpeq_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpeq_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpeq_u64(m, a, b);
  else return svcmpeq_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  using E = TypeOf<T>;
  return word::cmpeq(a, b, details::ptrue<E>());
}

/* cmpne */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpne_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpne_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpne_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmpne_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmpne_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpne_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpne_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpne_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpne_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpne_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpne_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpne_u64(m, a, b);
  else return svcmpne_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  using E = TypeOf<T>;
  return word::cmpne(a, b, details::ptrue<E>());
}

/* cmplt */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmplt_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmplt_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmplt_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmplt_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmplt_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmplt_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmplt_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmplt_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmplt_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmplt_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmplt_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmplt_u64(m, a, b);
  else return svcmplt_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  using E = TypeOf<T>;
  return word::cmplt(a, b, details::ptrue<E>());
}

/* cmpgt */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpgt_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpgt_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpgt_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmpgt_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmpgt_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpgt_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpgt_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpgt_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpgt_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpgt_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpgt_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpgt_u64(m, a, b);
  else return svcmpgt_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  using E = TypeOf<T>;
  return word::cmpgt(a, b, details::ptrue<E>());
}

/* cmple */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmple_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmple_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmple_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmple_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmple_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmple_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmple_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmple_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmple_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmple_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmple_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmple_u64(m, a, b);
  else return svcmple_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  using E = TypeOf<T>;
  return word::cmple(a, b, details::ptrue<E>());
}

/* cmpge */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpge_f32(m, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpge_f64(m, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpge_f16(m, a, b);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto a_lo = details::bf16_to_f32_lo(a);
    auto a_hi = details::bf16_to_f32_hi(a);
    auto b_lo = details::bf16_to_f32_lo(b);
    auto b_hi = details::bf16_to_f32_hi(b);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto cmp_lo = svcmpge_f32(m_lo, a_lo, b_lo);
    auto cmp_hi = svcmpge_f32(m_hi, a_hi, b_hi);
    return details::combine_f32_masks_to_bf16(cmp_lo, cmp_hi);
  }
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpge_u8(m, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpge_s8(m, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpge_u16(m, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpge_s16(m, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpge_u32(m, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpge_s32(m, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpge_u64(m, a, b);
  else return svcmpge_s64(m, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  using E = TypeOf<T>;
  return word::cmpge(a, b, details::ptrue<E>());
}

/* ================================================================ */
//                            Float Classification                  //
/* ================================================================ */
/* isnan: NaN != NaN returns true */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpne_f32(m, v, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpne_f64(m, v, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpne_f16(m, v, v);
  else if constexpr (std::is_same_v<E, bfloat16_t>) {
    auto v_lo = details::bf16_to_f32_lo(v);
    auto v_hi = details::bf16_to_f32_hi(v);
    auto m_lo = details::promote_mask_bf16_to_f32_lo(m);
    auto m_hi = details::promote_mask_bf16_to_f32_hi(m);
    auto nan_lo = svcmpne_f32(m_lo, v_lo, v_lo);
    auto nan_hi = svcmpne_f32(m_hi, v_hi, v_hi);
    return details::combine_f32_masks_to_bf16(nan_lo, nan_hi);
  }
  else return svcmpne_f32(m, v, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  using E = TypeOf<T>;
  return word::isnan(v, details::ptrue<E>());
}

/* isposinf: compare with +INFINITY */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)), m);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v) {
  using E = TypeOf<T>;
  return word::isposinf(v, details::ptrue<E>());
}

/* isneginf: compare with -INFINITY */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)), m);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v) {
  using E = TypeOf<T>;
  return word::isneginf(v, details::ptrue<E>());
}

/* isinf: abs(v) is posinf */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v, Mask<T> m) {
  return word::isposinf(word::abs(v), m);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v) {
  using E = TypeOf<T>;
  return word::isinf(v, details::ptrue<E>());
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_ARITHMETIC_H
