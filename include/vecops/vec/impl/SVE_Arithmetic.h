//
// SVE_Arithmetic.h — SVE real implementations for arithmetic operations
//

#ifndef VECOPS_SVE_ARITHMETIC_H
#define VECOPS_SVE_ARITHMETIC_H

#include <arm_sve.h>
#include <cmath>

#include "CoreDefs.h"
#include "../VecBase.h"
#include "SVE_Basic.h"
#include "SVE_Bit.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* ================================================================ */
//                                  Add                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svadd_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svadd_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svadd_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svadd_bf16_x(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svadd_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svadd_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svadd_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svadd_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svadd_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svadd_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svadd_u64_x(pg, a, b);
  else return svadd_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::add(a, b));
}

/* ================================================================ */
//                                  Sub                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svsub_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svsub_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svsub_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsub_bf16_x(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsub_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsub_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsub_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsub_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsub_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsub_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svsub_u64_x(pg, a, b);
  else return svsub_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::sub(a, b));
}

/* ================================================================ */
//                                  Mul                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svmul_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmul_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmul_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svmul_bf16_x(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmul_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svmul_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmul_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmul_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmul_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmul_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svmul_u64_x(pg, a, b);
  else return svmul_s64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}

/* ================================================================ */
//                                  Div (float only)                //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V div(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svdiv_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svdiv_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svdiv_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svdiv_bf16_x(pg, a, b);
#endif
  else return svdiv_f32_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::div(a, b));
}

/* ================================================================ */
//                                  Min                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svmin_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmin_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmin_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svmin_bf16_x(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, int8_t>)    return svmin_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmin_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmin_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmin_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmin_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmin_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svmin_s64_x(pg, a, b);
  else return svmin_u64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::min(a, b));
}

/* ================================================================ */
//                                  Max                             //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svmax_f32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svmax_f64_x(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svmax_f16_x(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svmax_bf16_x(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, int8_t>)    return svmax_s8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svmax_u8_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svmax_s16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svmax_u16_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svmax_s32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svmax_u32_x(pg, a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svmax_s64_x(pg, a, b);
  else return svmax_u64_x(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::max(a, b));
}

/* ================================================================ */
//                                 Neg                              //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svneg_f32_x(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svneg_f64_x(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svneg_f16_x(pg, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svneg_bf16_x(pg, v);
#endif
  else if constexpr (std::is_same_v<E, int8_t>)    return svneg_s8_x(pg, v);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svneg_u8_x(pg, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svneg_s16_x(pg, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svneg_u16_x(pg, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svneg_s32_x(pg, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svneg_u32_x(pg, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svneg_s64_x(pg, v);
  else return svneg_u64_x(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::neg(v));
}

/* ================================================================ */
//                                 Abs                              //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svabs_f32_x(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svabs_f64_x(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svabs_f16_x(pg, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svabs_bf16_x(pg, v);
#endif
  else if constexpr (std::is_same_v<E, int8_t>)    return svabs_s8_x(pg, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svabs_s16_x(pg, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svabs_s32_x(pg, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svabs_s64_x(pg, v);
  else return v;
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::abs(v));
}

/* ================================================================ */
//                                 Sqrt (float only)                //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V sqrt(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svsqrt_f32_x(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svsqrt_f64_x(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svsqrt_f16_x(pg, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsqrt_bf16_x(pg, v);
#endif
  else return svsqrt_f32_x(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sqrt(v));
}

/* ================================================================ */
//                             Rsqrt (float only, approximate)     //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rsqrt(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svrsqrte_f32_x(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svrsqrte_f64_x(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svrsqrte_f16_x(pg, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svrsqrte_bf16_x(pg, v);
#endif
  else return svrsqrte_f32_x(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rsqrt(v));
}

/* ================================================================ */
//                             Rcp (float only, approximate)      //
/* ================================================================ */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rcp(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svrecpe_f32_x(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svrecpe_f64_x(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svrecpe_f16_x(pg, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svrecpe_bf16_x(pg, v);
#endif
  else return svrecpe_f32_x(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rcp(v));
}

/* ================================================================ */
//                             Comparisons                          //
/* ================================================================ */
/* cmpeq */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpeq_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpeq_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpeq_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmpeq_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpeq_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpeq_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpeq_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpeq_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpeq_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpeq_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpeq_u64(pg, a, b);
  else return svcmpeq_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpeq(a, b));
}

/* cmpne */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpne_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpne_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpne_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmpne_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpne_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpne_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpne_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpne_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpne_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpne_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpne_u64(pg, a, b);
  else return svcmpne_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpne(a, b));
}

/* cmplt */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmplt_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmplt_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmplt_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmplt_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmplt_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmplt_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmplt_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmplt_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmplt_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmplt_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmplt_u64(pg, a, b);
  else return svcmplt_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmplt(a, b));
}

/* cmpgt */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpgt_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpgt_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpgt_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmpgt_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpgt_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpgt_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpgt_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpgt_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpgt_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpgt_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpgt_u64(pg, a, b);
  else return svcmpgt_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpgt(a, b));
}

/* cmple */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmple_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmple_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmple_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmple_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmple_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmple_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmple_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmple_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmple_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmple_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmple_u64(pg, a, b);
  else return svcmple_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmple(a, b));
}

/* cmpge */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpge_f32(pg, a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpge_f64(pg, a, b);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpge_f16(pg, a, b);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmpge_bf16(pg, a, b);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svcmpge_u8(pg, a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svcmpge_s8(pg, a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svcmpge_u16(pg, a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svcmpge_s16(pg, a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svcmpge_u32(pg, a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svcmpge_s32(pg, a, b);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svcmpge_u64(pg, a, b);
  else return svcmpge_s64(pg, a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpge(a, b));
}

/* ================================================================ */
//                            Float Classification                  //
/* ================================================================ */
/* isnan: NaN != NaN returns true */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  using E = TypeOf<T>;
  auto pg = sve_detail::sve_ptrue<E>();
  if constexpr (std::is_same_v<E, float32_t>)      return svcmpne_f32(pg, v, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svcmpne_f64(pg, v, v);
  else if constexpr (std::is_same_v<E, float16_t>) return svcmpne_f16(pg, v, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svcmpne_bf16(pg, v, v);
#endif
  else return svcmpne_f32(pg, v, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return word::bit_and(m, word::isnan(v));
}

/* isposinf: compare with +INFINITY */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)), m);
}

/* isneginf: compare with -INFINITY */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)), m);
}

/* isinf: abs(v) is posinf */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v) {
  return word::isposinf(word::abs(v));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v, Mask<T> m) {
  return word::isposinf(word::abs(v), m);
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_ARITHMETIC_H
