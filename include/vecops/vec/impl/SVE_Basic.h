//
// SVE_Basic.h — SVE register-only word-level operations
// Provides fill, zeros, get, set, mask ops, shuffle, interleave, etc.
// Stubs for load/store/arithmetic/comparison/conversion (needed by Vec.h).
//

#ifndef VECOPS_SVE_BASIC_H
#define VECOPS_SVE_BASIC_H

#include <arm_sve.h>
#include <cstring>
#include <cmath>

#include "CoreTypes.h"
#include "../VecBase.h"
#include "SVE_Types.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {

namespace details {

template <typename T> VECOPS_VFUNC constexpr nint_t max_elms() { return SVE::max_word_count<T>; }
template <typename T> VECOPS_VFUNC nint_t lane_count() {
  if constexpr (sizeof(T) == 1) return svcntb();
  else if constexpr (sizeof(T) == 2) return svcnth();
  else if constexpr (sizeof(T) == 4) return svcntw();
  else return svcntd();
}
template <typename T> VECOPS_VFUNC svbool_t ptrue() {
  if constexpr (sizeof(T) == 1) return svptrue_b8();
  else if constexpr (sizeof(T) == 2) return svptrue_b16();
  else if constexpr (sizeof(T) == 4) return svptrue_b32();
  else return svptrue_b64();
}
template <typename T> VECOPS_VFUNC svbool_t single_mask(nint_t idx) {
  if constexpr (sizeof(T) == 1)      return svcmpeq_n_s8(svptrue_b8(), svindex_s8(0, 1), (int8_t)idx);
  else if constexpr (sizeof(T) == 2) return svcmpeq_n_s16(svptrue_b16(), svindex_s16(0, 1), (int16_t)idx);
  else if constexpr (sizeof(T) == 4) return svcmpeq_n_s32(svptrue_b32(), svindex_s32(0, 1), (int32_t)idx);
  else return svcmpeq_n_s64(svptrue_b64(), svindex_s64(0, 1), (int64_t)idx);
}

template <typename E> VECOPS_VFUNC svuint8_t to_u8(auto v) {
  if constexpr (std::is_same_v<E, float32_t>) return svreinterpret_u8_f32(v);
  else if constexpr (std::is_same_v<E, float64_t>) return svreinterpret_u8_f64(v);
  else if constexpr (std::is_same_v<E, int8_t>)  return svreinterpret_u8_s8(v);
  else if constexpr (std::is_same_v<E, uint8_t>) return v;
  else if constexpr (std::is_same_v<E, int16_t>)  return svreinterpret_u8_s16(v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svreinterpret_u8_u16(v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svreinterpret_u8_s32(v);
  else if constexpr (std::is_same_v<E, uint32_t>) return svreinterpret_u8_u32(v);
  else if constexpr (std::is_same_v<E, int64_t>)  return svreinterpret_u8_s64(v);
  else if constexpr (std::is_same_v<E, float16_t>) return svreinterpret_u8_f16(v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_u8_bf16(v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_u8_u16(svreinterpret_u16_bf16(v));
#endif
  else return svreinterpret_u8_u64(v);
}

template <typename E> VECOPS_VFUNC auto from_u8(svuint8_t v) {
  if constexpr (std::is_same_v<E, float32_t>) return svreinterpret_f32_u8(v);
  else if constexpr (std::is_same_v<E, float64_t>) return svreinterpret_f64_u8(v);
  else if constexpr (std::is_same_v<E, int8_t>)  return svreinterpret_s8_u8(v);
  else if constexpr (std::is_same_v<E, uint8_t>) return v;
  else if constexpr (std::is_same_v<E, int16_t>)  return svreinterpret_s16_u8(v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svreinterpret_u16_u8(v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svreinterpret_s32_u8(v);
  else if constexpr (std::is_same_v<E, uint32_t>) return svreinterpret_u32_u8(v);
  else if constexpr (std::is_same_v<E, int64_t>)  return svreinterpret_s64_u8(v);
  else if constexpr (std::is_same_v<E, float16_t>) return svreinterpret_f16_u8(v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u8(v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svreinterpret_u16_u8(v));
#endif
  else return svreinterpret_u64_u8(v);
}

// ---- u32 canonical type for bitcast (x86-style, prevents combinatorial explosion) ----
template <typename E> VECOPS_VFUNC svuint32_t to_u32(auto v) {
  if constexpr (std::is_same_v<E, float32_t>) return svreinterpret_u32_f32(v);
  else if constexpr (std::is_same_v<E, float64_t>) return svreinterpret_u32_f64(v);
  else if constexpr (std::is_same_v<E, int8_t>)  return svreinterpret_u32_s8(v);
  else if constexpr (std::is_same_v<E, uint8_t>) return svreinterpret_u32_u8(v);
  else if constexpr (std::is_same_v<E, int16_t>)  return svreinterpret_u32_s16(v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svreinterpret_u32_u16(v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svreinterpret_u32_s32(v);
  else if constexpr (std::is_same_v<E, uint32_t>) return v;
  else if constexpr (std::is_same_v<E, int64_t>)  return svreinterpret_u32_s64(v);
  else if constexpr (std::is_same_v<E, float16_t>) return svreinterpret_u32_f16(v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_u32_bf16(v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_u32_u16(svreinterpret_u16_bf16(v));
#endif
  else return svreinterpret_u32_u64(v);
}

template <typename E> VECOPS_VFUNC auto from_u32(svuint32_t v) {
  if constexpr (std::is_same_v<E, float32_t>) return svreinterpret_f32_u32(v);
  else if constexpr (std::is_same_v<E, float64_t>) return svreinterpret_f64_u32(v);
  else if constexpr (std::is_same_v<E, int8_t>)  return svreinterpret_s8_u32(v);
  else if constexpr (std::is_same_v<E, uint8_t>) return svreinterpret_u8_u32(v);
  else if constexpr (std::is_same_v<E, int16_t>)  return svreinterpret_s16_u32(v);
  else if constexpr (std::is_same_v<E, uint16_t>) return svreinterpret_u16_u32(v);
  else if constexpr (std::is_same_v<E, int32_t>)  return svreinterpret_s32_u32(v);
  else if constexpr (std::is_same_v<E, uint32_t>) return v;
  else if constexpr (std::is_same_v<E, int64_t>)  return svreinterpret_s64_u32(v);
  else if constexpr (std::is_same_v<E, float16_t>) return svreinterpret_f16_u32(v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u32(v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svreinterpret_u16_u32(v));
#endif
  else return svreinterpret_u64_u32(v);
}

// ---- Typed intrinsic dispatchers ----
template <typename E, typename V, typename I>
VECOPS_VFUNC auto table_lookup(V v, I idx) {
  if constexpr (std::is_same_v<E, float32_t>) return svtbl_f32(v, idx);
  else if constexpr (std::is_same_v<E, float64_t>) return svtbl_f64(v, idx);
  else if constexpr (std::is_same_v<E, int8_t>)    return svtbl_s8(v, idx);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svtbl_u8(v, idx);
  else if constexpr (std::is_same_v<E, int16_t>)   return svtbl_s16(v, idx);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svtbl_u16(v, idx);
  else if constexpr (std::is_same_v<E, int32_t>)   return svtbl_s32(v, idx);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svtbl_u32(v, idx);
  else if constexpr (std::is_same_v<E, int64_t>)   return svtbl_s64(v, idx);
  else if constexpr (std::is_same_v<E, float16_t>)  return svtbl_f16(v, idx);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svtbl_bf16(v, idx);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svtbl_u16(svreinterpret_u16_bf16(v), idx));
#endif
  else return svtbl_u64(v, idx);
}

template <typename E, typename V>
VECOPS_VFUNC auto select(svbool_t m, V v1, V v0) {
  if constexpr (std::is_same_v<E, float32_t>) return svsel_f32(m, v1, v0);
  else if constexpr (std::is_same_v<E, float64_t>) return svsel_f64(m, v1, v0);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsel_s8(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsel_u8(m, v1, v0);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsel_s16(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsel_u16(m, v1, v0);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsel_s32(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsel_u32(m, v1, v0);
  else if constexpr (std::is_same_v<E, int64_t>)   return svsel_s64(m, v1, v0);
  else if constexpr (std::is_same_v<E, float16_t>)  return svsel_f16(m, v1, v0);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsel_bf16(m, v1, v0);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svsel_u16(m, svreinterpret_u16_bf16(v1), svreinterpret_u16_bf16(v0)));
#endif
  else return svsel_u64(m, v1, v0);
}

template <typename E>
VECOPS_VFUNC auto dup_n(decltype(E()) value) {
  if constexpr (std::is_same_v<E, float32_t>) return svdup_n_f32(value);
  else if constexpr (std::is_same_v<E, float64_t>) return svdup_n_f64(value);
  else if constexpr (std::is_same_v<E, int8_t>)    return svdup_n_s8(value);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svdup_n_u8(value);
  else if constexpr (std::is_same_v<E, int16_t>)   return svdup_n_s16(value);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svdup_n_u16(value);
  else if constexpr (std::is_same_v<E, int32_t>)   return svdup_n_s32(value);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svdup_n_u32(value);
  else if constexpr (std::is_same_v<E, int64_t>)   return svdup_n_s64(value);
  else if constexpr (std::is_same_v<E, float16_t>)  return svdup_n_f16(value);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svdup_n_bf16(value);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svdup_n_u16(value.to_bits()));
#endif
  else return svdup_n_u64(value);
}

template <typename E, typename V>
VECOPS_VFUNC auto lastb(svbool_t pg, V v) {
  if constexpr (std::is_same_v<E, float32_t>) return svlastb_f32(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svlastb_f64(pg, v);
  else if constexpr (std::is_same_v<E, int8_t>)    return svlastb_s8(pg, v);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svlastb_u8(pg, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svlastb_s16(pg, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svlastb_u16(pg, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svlastb_s32(pg, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svlastb_u32(pg, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svlastb_s64(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>)  return svlastb_f16(pg, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svlastb_bf16(pg, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return BFloat16::from_bits(svlastb_u16(pg, svreinterpret_u16_bf16(v)));
#endif
  else return svlastb_u64(pg, v);
}

template <typename E, typename V>
VECOPS_VFUNC auto zip1(V a, V b) {
  if constexpr (std::is_same_v<E, float32_t>) return svzip1_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svzip1_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svzip1_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svzip1_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svzip1_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svzip1_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svzip1_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svzip1_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svzip1_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)  return svzip1_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svzip1_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svzip1_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svzip1_u64(a, b);
}

template <typename E, typename V>
VECOPS_VFUNC auto uzp1(V a, V b) {
  if constexpr (std::is_same_v<E, float32_t>) return svuzp1_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp1_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svuzp1_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svuzp1_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svuzp1_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svuzp1_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svuzp1_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svuzp1_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svuzp1_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)  return svuzp1_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svuzp1_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svuzp1_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svuzp1_u64(a, b);
}
template <typename E, typename V>
VECOPS_VFUNC auto uzp2(V a, V b) {
  if constexpr (std::is_same_v<E, float32_t>) return svuzp2_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp2_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svuzp2_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svuzp2_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svuzp2_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svuzp2_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svuzp2_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svuzp2_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svuzp2_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)  return svuzp2_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svuzp2_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svuzp2_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svuzp2_u64(a, b);
}

template <typename E, typename V>
VECOPS_VFUNC auto trn1(V a, V b) {
  if constexpr (std::is_same_v<E, float32_t>) return svtrn1_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svtrn1_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svtrn1_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svtrn1_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svtrn1_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svtrn1_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svtrn1_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svtrn1_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svtrn1_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)  return svtrn1_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svtrn1_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svtrn1_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svtrn1_u64(a, b);
}
template <typename E, typename V>
VECOPS_VFUNC auto trn2(V a, V b) {
  if constexpr (std::is_same_v<E, float32_t>) return svtrn2_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svtrn2_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)    return svtrn2_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svtrn2_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)   return svtrn2_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svtrn2_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)   return svtrn2_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svtrn2_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)   return svtrn2_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)  return svtrn2_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svtrn2_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svtrn2_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svtrn2_u64(a, b);
}

template <typename E, typename V>
VECOPS_VFUNC auto splice(svbool_t pg, V lo, V hi) {
  if constexpr (std::is_same_v<E, float32_t>) return svsplice_f32(pg, lo, hi);
  else if constexpr (std::is_same_v<E, float64_t>) return svsplice_f64(pg, lo, hi);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsplice_s8(pg, lo, hi);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsplice_u8(pg, lo, hi);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsplice_s16(pg, lo, hi);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsplice_u16(pg, lo, hi);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsplice_s32(pg, lo, hi);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsplice_u32(pg, lo, hi);
  else if constexpr (std::is_same_v<E, int64_t>)   return svsplice_s64(pg, lo, hi);
  else if constexpr (std::is_same_v<E, float16_t>)  return svsplice_f16(pg, lo, hi);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsplice_bf16(pg, lo, hi);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svsplice_u16(pg, svreinterpret_u16_bf16(lo), svreinterpret_u16_bf16(hi)));
#endif
  else return svsplice_u64(pg, lo, hi);
}

}  // namespace details

// === bf16-safe multi-register tuple access helpers ===
// These abstract away the need for svget2/svcreate2/etc. on bf16 tuples,
// which require +bf16 target feature when used natively.
// Indices must be compile-time constants (template parameters).
namespace details {
template <typename T, uint64_t Index>
VECOPS_VFUNC auto tuple_get2(auto v) {
#if defined(HAS_BF16)
  return svget2(v, Index);
#else
  if constexpr (std::is_same_v<T, bfloat16_t>)
    return svreinterpret_bf16_u16(svget2(svreinterpret_u16_bf16_x2(v), Index));
  else
    return svget2(v, Index);
#endif
}
template <typename T, uint64_t Index>
VECOPS_VFUNC auto tuple_get4(auto v) {
#if defined(HAS_BF16)
  return svget4(v, Index);
#else
  if constexpr (std::is_same_v<T, bfloat16_t>)
    return svreinterpret_bf16_u16(svget4(svreinterpret_u16_bf16_x4(v), Index));
  else
    return svget4(v, Index);
#endif
}
template <typename T, typename WA, typename WB>
VECOPS_VFUNC auto tuple_create2(WA wa, WB wb) {
#if defined(HAS_BF16)
  return svcreate2(wa, wb);
#else
  if constexpr (std::is_same_v<T, bfloat16_t>)
    return svreinterpret_bf16_u16_x2(
      svcreate2(svreinterpret_u16_bf16(wa), svreinterpret_u16_bf16(wb)));
  else
    return svcreate2(wa, wb);
#endif
}
template <typename T, typename WA, typename WB, typename WC, typename WD>
VECOPS_VFUNC auto tuple_create4(WA wa, WB wb, WC wc, WD wd) {
#if defined(HAS_BF16)
  return svcreate4(wa, wb, wc, wd);
#else
  if constexpr (std::is_same_v<T, bfloat16_t>)
    return svreinterpret_bf16_u16_x4(
      svcreate4(svreinterpret_u16_bf16(wa), svreinterpret_u16_bf16(wb),
                svreinterpret_u16_bf16(wc), svreinterpret_u16_bf16(wd)));
  else
    return svcreate4(wa, wb, wc, wd);
#endif
}
}  // namespace details

/* === mfill / mwhilelt / mwhilege (used before fill_with_n) === */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mfill(T, bool value) {
  using E = TypeOf<T>;
  if constexpr (sizeof(E) == 1)      return svdup_b8(value);
  else if constexpr (sizeof(E) == 2) return svdup_b16(value);
  else if constexpr (sizeof(E) == 4) return svdup_b32(value);
  else return svdup_b64(value);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilelt(T, nint_t a, nint_t b) {
  using E = TypeOf<T>;
  if constexpr (sizeof(nint_t) == 4) {
    if constexpr (sizeof(E) == 1)      return svwhilelt_b8_s32((int32_t)a, (int32_t)b);
    else if constexpr (sizeof(E) == 2) return svwhilelt_b16_s32((int32_t)a, (int32_t)b);
    else if constexpr (sizeof(E) == 4) return svwhilelt_b32_s32((int32_t)a, (int32_t)b);
    else return svwhilelt_b64_s32((int32_t)a, (int32_t)b);
  } else {
    if constexpr (sizeof(E) == 1)      return svwhilelt_b8_s64((int64_t)a, (int64_t)b);
    else if constexpr (sizeof(E) == 2) return svwhilelt_b16_s64((int64_t)a, (int64_t)b);
    else if constexpr (sizeof(E) == 4) return svwhilelt_b32_s64((int64_t)a, (int64_t)b);
    else return svwhilelt_b64_s64((int64_t)a, (int64_t)b);
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilege(T, nint_t a, nint_t b) {
  using E = TypeOf<T>;
  if constexpr (sizeof(nint_t) == 4) {
    if constexpr (sizeof(E) == 1)      return svnot_b_z(svptrue_b8(),  svwhilelt_b8_s32((int32_t)a, (int32_t)b));
    else if constexpr (sizeof(E) == 2) return svnot_b_z(svptrue_b16(), svwhilelt_b16_s32((int32_t)a, (int32_t)b));
    else if constexpr (sizeof(E) == 4) return svnot_b_z(svptrue_b32(), svwhilelt_b32_s32((int32_t)a, (int32_t)b));
    else return svnot_b_z(svptrue_b64(), svwhilelt_b64_s32((int32_t)a, (int32_t)b));
  } else {
    if constexpr (sizeof(E) == 1)      return svnot_b_z(svptrue_b8(),  svwhilelt_b8_s64((int64_t)a, (int64_t)b));
    else if constexpr (sizeof(E) == 2) return svnot_b_z(svptrue_b16(), svwhilelt_b16_s64((int64_t)a, (int64_t)b));
    else if constexpr (sizeof(E) == 4) return svnot_b_z(svptrue_b32(), svwhilelt_b32_s64((int64_t)a, (int64_t)b));
    else return svnot_b_z(svptrue_b64(), svwhilelt_b64_s64((int64_t)a, (int64_t)b));
  }
}

/* === fill / zeros === */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T, TypeOf<T> value) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svdup_n_f32(value);
  else if constexpr (std::is_same_v<E, float64_t>) return svdup_n_f64(value);
  else if constexpr (std::is_same_v<E, int8_t>)    return svdup_n_s8(value);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svdup_n_u8(value);
  else if constexpr (std::is_same_v<E, int16_t>)   return svdup_n_s16(value);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svdup_n_u16(value);
  else if constexpr (std::is_same_v<E, int32_t>)   return svdup_n_s32(value);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svdup_n_u32(value);
  else if constexpr (std::is_same_v<E, int64_t>)   return svdup_n_s64(value);
  else if constexpr (std::is_same_v<E, float16_t>)  return svdup_n_f16(value);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svdup_n_bf16(value);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svdup_n_u16(value.to_bits()));
#endif
  else return svdup_n_u64(value);
}

/* === blend === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V blend(V v0, Mask<T> m, V v1) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svsel_f32(m, v1, v0);
  else if constexpr (std::is_same_v<E, float64_t>) return svsel_f64(m, v1, v0);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsel_s8(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsel_u8(m, v1, v0);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsel_s16(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsel_u16(m, v1, v0);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsel_s32(m, v1, v0);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsel_u32(m, v1, v0);
  else if constexpr (std::is_same_v<E, int64_t>)   return svsel_s64(m, v1, v0);
  else if constexpr (std::is_same_v<E, float16_t>)  return svsel_f16(m, v1, v0);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsel_bf16(m, v1, v0);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svsel_u16(m, svreinterpret_u16_bf16(v1), svreinterpret_u16_bf16(v0)));
#endif
  else return svsel_u64(m, v1, v0);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, Vec<T> default_v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svdup_n_f32_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, float64_t>) return svdup_n_f64_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, int8_t>)    return svdup_n_s8_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svdup_n_u8_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, int16_t>)   return svdup_n_s16_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svdup_n_u16_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, int32_t>)   return svdup_n_s32_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svdup_n_u32_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, int64_t>)   return svdup_n_s64_m(default_v, m, value);
  else if constexpr (std::is_same_v<E, float16_t>)  return svdup_n_f16_m(default_v, m, value);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svdup_n_bf16_m(default_v, m, value);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svdup_n_u16_m(svreinterpret_u16_bf16(default_v), m, value.to_bits()));
#endif
  else return svdup_n_u64_m(default_v, m, value);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  return word::fill(t, value, word::mwhilelt(t, 0, n), default_v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> zeros(T t) { return word::fill(t, TypeOf<T>()); }

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> make_mask(T t) {
  if constexpr (T::POW2 == 0) {
    return details::ptrue<TypeOf<T>>();
  } else {
    return word::mwhilelt(t, 0, size(t));
  }
}


/* === mask bit ops === */
template <TLV_DECL_MASK(M)> VECOPS_VFUNC M bit_and(M a, M b) { return svand_b_z(svptrue_b8(), a, b); }
template <TLV_DECL_MASK(M)> VECOPS_VFUNC M bit_or(M a, M b)  { return svorr_b_z(svptrue_b8(), a, b); }
template <TLV_DECL_MASK(M)> VECOPS_VFUNC M bit_xor(M a, M b) { return sveor_b_z(svptrue_b8(), a, b); }
template <TLV_DECL_MASK(M)> VECOPS_VFUNC M bit_andnot(M a, M b) { return svbic_b_z(svptrue_b8(), b, a); }
template <TLV_DECL_MASK(M)> VECOPS_VFUNC M bit_not(M a) { return svnot_b_z(svptrue_b8(), a); }

/* === get / set === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC TypeOf<T> get(V v, nint_t idx) {
  using E = TypeOf<T>;
  auto pg = details::single_mask<E>(idx);
  if constexpr (std::is_same_v<E, float32_t>)      return svlastb_f32(pg, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svlastb_f64(pg, v);
  else if constexpr (std::is_same_v<E, int8_t>)    return svlastb_s8(pg, v);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svlastb_u8(pg, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svlastb_s16(pg, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svlastb_u16(pg, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svlastb_s32(pg, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svlastb_u32(pg, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svlastb_s64(pg, v);
  else if constexpr (std::is_same_v<E, float16_t>)  return svlastb_f16(pg, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svlastb_bf16(pg, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return BFloat16::from_bits(svlastb_u16(pg, svreinterpret_u16_bf16(v)));
#endif
  else return svlastb_u64(pg, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V set(V v, nint_t idx, TypeOf<T> x) {
  using E = TypeOf<T>;
  auto pg = details::single_mask<E>(idx);
  auto bc = word::fill(T(), x);
  if constexpr (std::is_same_v<E, float32_t>)      return svsel_f32(pg, bc, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svsel_f64(pg, bc, v);
  else if constexpr (std::is_same_v<E, int8_t>)    return svsel_s8(pg, bc, v);
  else if constexpr (std::is_same_v<E, uint8_t>)   return svsel_u8(pg, bc, v);
  else if constexpr (std::is_same_v<E, int16_t>)   return svsel_s16(pg, bc, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svsel_u16(pg, bc, v);
  else if constexpr (std::is_same_v<E, int32_t>)   return svsel_s32(pg, bc, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svsel_u32(pg, bc, v);
  else if constexpr (std::is_same_v<E, int64_t>)   return svsel_s64(pg, bc, v);
  else if constexpr (std::is_same_v<E, float16_t>)  return svsel_f16(pg, bc, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svsel_bf16(pg, bc, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(
    svsel_u16(pg, svreinterpret_u16_bf16(bc), svreinterpret_u16_bf16(v)));
#endif
  else return svsel_u64(pg, bc, v);
}

template <TLV_DECL_TAG(TT), TLV_DECL_MASK(M)>
VECOPS_VFUNC bool get(TT, M m, nint_t idx) {
  using E = TypeOf<TT>;
  auto pg = details::single_mask<E>(idx);
  auto at = details::ptrue<E>();
  return svptest_any(at, svand_b_z(at, m, pg));
}

template <TLV_DECL_TAG(TT), TLV_DECL_MASK(M)>
VECOPS_VFUNC M set(TT, M m, nint_t idx, bool x) {
  using E = TypeOf<TT>;
  auto pg = details::single_mask<E>(idx);
  auto at = details::ptrue<E>();
  return x ? svorr_b_z(at, m, pg) : svbic_b_z(at, m, pg);
}

/* === bitcast === */
template <typename To, typename V, typename Ti = Vec2Tag<V>,
          TL_IF(is_word_vec(To())), TL_IF(is_word_vec(Ti()))>
VECOPS_VFUNC Vec<To> bitcast(To, V v) {
  using Eo = TypeOf<To>; using Ei = TypeOf<Ti>;
  if constexpr (std::is_same_v<Ti, To>) return v;
  return details::from_u32<Eo>(details::to_u32<Ei>(v));
}

// used for internal API
template <typename To, typename V, typename Ti = Vec2Tag<V>,
    TL_IF(num_words(To{}) == 2), TL_IF(num_words(Ti{}) == 2)>
VECOPS_VFUNC Vec<To> bitcast(To, V v) {
  Half<To> t1;
  return details::tuple_create2<TypeOf<To>>(
    word::bitcast(t1, details::tuple_get2<TypeOf<Ti>, 0>(v)),
    word::bitcast(t1, details::tuple_get2<TypeOf<Ti>, 1>(v)));
}

// used for internal API
template <typename To, typename V, typename Ti = Vec2Tag<V>,
    TL_IF(num_words(To{}) == 4), TL_IF(num_words(Ti{}) == 4)>
VECOPS_VFUNC Vec<To> bitcast(To, V v) {
  Half<Half<To>> t1;
  return details::tuple_create4<TypeOf<To>>(
    word::bitcast(t1, details::tuple_get4<TypeOf<Ti>, 0>(v)),
    word::bitcast(t1, details::tuple_get4<TypeOf<Ti>, 1>(v)),
    word::bitcast(t1, details::tuple_get4<TypeOf<Ti>, 2>(v)),
    word::bitcast(t1, details::tuple_get4<TypeOf<Ti>, 3>(v)));
}

/* === shuf / local_shuf === */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> vi) {
  using E = TypeOf<T>;
  using IdxT = Index<E>;
  if constexpr (std::is_same_v<IdxT, int8_t>) {
    auto uvi = svreinterpret_u8_s8(vi);
    return details::table_lookup<E, V, decltype(uvi)>(v, uvi);
  } else if constexpr (std::is_same_v<IdxT, int16_t>) {
    auto uvi = svreinterpret_u16_s16(vi);
    return details::table_lookup<E, V, decltype(uvi)>(v, uvi);
  } else if constexpr (std::is_same_v<IdxT, int32_t>) {
    auto uvi = svreinterpret_u32_s32(vi);
    return details::table_lookup<E, V, decltype(uvi)>(v, uvi);
  } else {
    auto uvi = svreinterpret_u64_s64(vi);
    return details::table_lookup<E, V, decltype(uvi)>(v, uvi);
  }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> vi) {
  using E = TypeOf<T>; using IdxT = Index<E>;
  constexpr nint_t group_el = 16 / (nint_t)sizeof(E);
  constexpr int shift = []{ int s=0; for(nint_t g=group_el; g>1; g>>=1) ++s; return s; }();
  constexpr auto ti = Rebind<IdxT, T>{};

  // Use unsigned for shifts (logical shift right)
  using UIdx = std::make_unsigned_t<IdxT>;
  auto pg = details::ptrue<IdxT>();

  // Generate per-element group base: (index >> shift) << shift
  auto ibase = Vec<decltype(ti)>{};
  if constexpr (sizeof(UIdx) == 1) {
    auto u = svindex_u8(0, 1);
    u = svlsr_n_u8_z(pg, u, shift);
    u = svlsl_n_u8_z(pg, u, shift);
    ibase = svreinterpret_s8_u8(u);
  } else if constexpr (sizeof(UIdx) == 2) {
    auto u = svindex_u16(0, 1);
    u = svlsr_n_u16_z(pg, u, shift);
    u = svlsl_n_u16_z(pg, u, shift);
    ibase = svreinterpret_s16_u16(u);
  } else if constexpr (sizeof(UIdx) == 4) {
    auto u = svindex_u32(0, 1);
    u = svlsr_n_u32_z(pg, u, shift);
    u = svlsl_n_u32_z(pg, u, shift);
    ibase = svreinterpret_s32_u32(u);
  } else {
    auto u = svindex_u64(0, 1);
    u = svlsr_n_u64_z(pg, u, shift);
    u = svlsl_n_u64_z(pg, u, shift);
    ibase = svreinterpret_s64_u64(u);
  }

  // Global indices = base + local offsets
  auto iglob = ibase;
  if constexpr (sizeof(IdxT) == 1)       iglob = svadd_s8_z(pg, ibase, vi);
  else if constexpr (sizeof(IdxT) == 2)  iglob = svadd_s16_z(pg, ibase, vi);
  else if constexpr (sizeof(IdxT) == 4)  iglob = svadd_s32_z(pg, ibase, vi);
  else                                   iglob = svadd_s64_z(pg, ibase, vi);

  return word::shuf(v, iglob);
}

template <int... Is, TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v) {
  using E = TypeOf<T>;
  static_assert(sizeof...(Is) * sizeof(E) == 16,
                "Number of compile-time indices must match elements per 16-byte lane");
  constexpr std::array<int, sizeof...(Is)> arr = {Is...};
  // arr[0] = I_{M-1}, arr[M-1] = I_0 -- reverse to ascending order for svdupq

  if constexpr (sizeof...(Is) == 2) {
    auto vi = svdupq_n_s64(arr[1], arr[0]);
    return word::local_shuf(v, vi);
  } else if constexpr (sizeof...(Is) == 4) {
    auto vi = svdupq_n_s32(arr[3], arr[2], arr[1], arr[0]);
    return word::local_shuf(v, vi);
  } else if constexpr (sizeof...(Is) == 8) {
    auto vi = svdupq_n_s16(arr[7], arr[6], arr[5], arr[4],
                           arr[3], arr[2], arr[1], arr[0]);
    return word::local_shuf(v, vi);
  } else if constexpr (sizeof...(Is) == 16) {
    auto vi = svdupq_n_s8(arr[15], arr[14], arr[13], arr[12],
                          arr[11], arr[10], arr[9], arr[8],
                          arr[7], arr[6], arr[5], arr[4],
                          arr[3], arr[2], arr[1], arr[0]);
    return word::local_shuf(v, vi);
  }
}

// Runtime scalar-index overloads — dispatch to vector version via svdupq
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v, int i3, int i2, int i1, int i0) {
  auto vi = svdupq_n_s32(i0, i1, i2, i3);
  return word::local_shuf(v, vi);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v, int i1, int i0) {
  auto vi = svdupq_n_s64(i0, i1);
  return word::local_shuf(v, vi);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  auto vi = svdupq_n_s16(i0, i1, i2, i3, i4, i5, i6, i7);
  return word::local_shuf(v, vi);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_shuf(V v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8,
                                         int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  auto vi = svdupq_n_s8(i0, i1, i2, i3, i4, i5, i6, i7,
                        i8, i9, i10, i11, i12, i13, i14, i15);
  return word::local_shuf(v, vi);
}

/* === upper / lower / even / odd === */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 1)>
VECOPS_VFUNC V upper(T t, Vec<T> v) {
  using E = TypeOf<T>; nint_t hn = word_size(t)/2;
  auto lo = word::mwhilelt(t, 0, hn);
  auto up = svnot_b_z(details::ptrue<E>(), lo);
  if constexpr (std::is_same_v<E, float32_t>)      return svsplice_f32(up, v, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svsplice_f64(up, v, v);
  else if constexpr (std::is_same_v<E, int8_t>)     return svsplice_s8(up, v, v);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svsplice_u8(up, v, v);
  else if constexpr (std::is_same_v<E, int16_t>)    return svsplice_s16(up, v, v);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svsplice_u16(up, v, v);
  else if constexpr (std::is_same_v<E, int32_t>)    return svsplice_s32(up, v, v);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svsplice_u32(up, v, v);
  else if constexpr (std::is_same_v<E, int64_t>)    return svsplice_s64(up, v, v);
  else if constexpr (std::is_same_v<E, float16_t>)   return svsplice_f16(up, v, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svsplice_bf16(up, v, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svsplice_u16(up, svreinterpret_u16_bf16(v), svreinterpret_u16_bf16(v)));
#endif
  else return svsplice_u64(up, v, v);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 2)>
VECOPS_VFUNC V upper(T t, Vec<T> v) { return details::tuple_get2<TypeOf<T>, 1>(v); }

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 4)>
VECOPS_VFUNC V upper(T t, Vec<T> v) {
  return details::tuple_create2<TypeOf<Half<T>>>(
    details::tuple_get4<TypeOf<T>, 2>(v), details::tuple_get4<TypeOf<T>, 3>(v));
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 1)>
VECOPS_VFUNC V lower(T t, Vec<T> v) { return v; }

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 2)>
VECOPS_VFUNC V lower(T t, Vec<T> v) { return details::tuple_get2<TypeOf<T>, 0>(v); }

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 4)>
VECOPS_VFUNC V lower(T t, Vec<T> v) {
  return details::tuple_create2<TypeOf<Half<T>>>(
    details::tuple_get4<TypeOf<T>, 0>(v), details::tuple_get4<TypeOf<T>, 1>(v));
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V even(T t, Vec<T> v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svuzp1_f32(v, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp1_f64(v, v);
  else if constexpr (std::is_same_v<E, int8_t>)     return svuzp1_s8(v, v);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svuzp1_u8(v, v);
  else if constexpr (std::is_same_v<E, int16_t>)    return svuzp1_s16(v, v);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svuzp1_u16(v, v);
  else if constexpr (std::is_same_v<E, int32_t>)    return svuzp1_s32(v, v);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svuzp1_u32(v, v);
  else if constexpr (std::is_same_v<E, int64_t>)    return svuzp1_s64(v, v);
  else if constexpr (std::is_same_v<E, float16_t>)   return svuzp1_f16(v, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svuzp1_bf16(v, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svuzp1_u16(svreinterpret_u16_bf16(v), svreinterpret_u16_bf16(v)));
#endif
  else return svuzp1_u64(v, v);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V odd(T t, Vec<T> v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svuzp2_f32(v, v);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp2_f64(v, v);
  else if constexpr (std::is_same_v<E, int8_t>)     return svuzp2_s8(v, v);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svuzp2_u8(v, v);
  else if constexpr (std::is_same_v<E, int16_t>)    return svuzp2_s16(v, v);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svuzp2_u16(v, v);
  else if constexpr (std::is_same_v<E, int32_t>)    return svuzp2_s32(v, v);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svuzp2_u32(v, v);
  else if constexpr (std::is_same_v<E, int64_t>)    return svuzp2_s64(v, v);
  else if constexpr (std::is_same_v<E, float16_t>)   return svuzp2_f16(v, v);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svuzp2_bf16(v, v);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svuzp2_u16(svreinterpret_u16_bf16(v), svreinterpret_u16_bf16(v)));
#endif
  else return svuzp2_u64(v, v);
}

/* === concat === */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 1)>
VECOPS_VFUNC Vec<T> concat(T t, V v_lo, V v_hi) {
  using E = TypeOf<T>; nint_t hn = size(t)/2;
  auto lo = word::mwhilelt(t, 0, hn);
  if constexpr (std::is_same_v<E, float32_t>)      return svsplice_f32(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float64_t>) return svsplice_f64(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int8_t>)     return svsplice_s8(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svsplice_u8(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int16_t>)    return svsplice_s16(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svsplice_u16(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int32_t>)    return svsplice_s32(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svsplice_u32(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int64_t>)    return svsplice_s64(lo, v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float16_t>)   return svsplice_f16(lo, v_lo, v_hi);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svsplice_bf16(lo, v_lo, v_hi);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svsplice_u16(lo, svreinterpret_u16_bf16(v_lo), svreinterpret_u16_bf16(v_hi)));
#endif
  else return svsplice_u64(lo, v_lo, v_hi);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 2)>
VECOPS_VFUNC Vec<T> concat(T t, V v_lo, V v_hi) {
  return details::tuple_create2<TypeOf<T>>(v_lo, v_hi);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(num_words(T{}) == 4)>
VECOPS_VFUNC Vec<T> concat(T t, V v_lo, V v_hi) {
  using E2 = TypeOf<Half<T>>;
  return details::tuple_create4<TypeOf<T>>(
    details::tuple_get2<E2, 0>(v_lo), details::tuple_get2<E2, 1>(v_lo),
    details::tuple_get2<E2, 0>(v_hi), details::tuple_get2<E2, 1>(v_hi));
}


template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> concat_even(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svuzp1_f32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp1_f64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int8_t>)     return svuzp1_s8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svuzp1_u8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int16_t>)    return svuzp1_s16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svuzp1_u16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int32_t>)    return svuzp1_s32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svuzp1_u32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int64_t>)    return svuzp1_s64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float16_t>)   return svuzp1_f16(v_lo, v_hi);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svuzp1_bf16(v_lo, v_hi);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svuzp1_u16(svreinterpret_u16_bf16(v_lo), svreinterpret_u16_bf16(v_hi)));
#endif
  else return svuzp1_u64(v_lo, v_hi);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> concat_odd(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svuzp2_f32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float64_t>) return svuzp2_f64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int8_t>)     return svuzp2_s8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svuzp2_u8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int16_t>)    return svuzp2_s16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svuzp2_u16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int32_t>)    return svuzp2_s32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svuzp2_u32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int64_t>)    return svuzp2_s64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float16_t>)   return svuzp2_f16(v_lo, v_hi);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svuzp2_bf16(v_lo, v_hi);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svuzp2_u16(svreinterpret_u16_bf16(v_lo), svreinterpret_u16_bf16(v_hi)));
#endif
  else return svuzp2_u64(v_lo, v_hi);
}

/* === interleave === */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC Vec<T> interleave(T t, V v_lo, V v_hi) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svzip1_f32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float64_t>) return svzip1_f64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int8_t>)     return svzip1_s8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svzip1_u8(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int16_t>)    return svzip1_s16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svzip1_u16(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int32_t>)    return svzip1_s32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svzip1_u32(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, int64_t>)    return svzip1_s64(v_lo, v_hi);
  else if constexpr (std::is_same_v<E, float16_t>)   return svzip1_f16(v_lo, v_hi);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svzip1_bf16(v_lo, v_hi);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svzip1_u16(svreinterpret_u16_bf16(v_lo), svreinterpret_u16_bf16(v_hi)));
#endif
  else return svzip1_u64(v_lo, v_hi);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V interleave_even(V a, V b) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svtrn1_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svtrn1_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)     return svtrn1_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svtrn1_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)    return svtrn1_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svtrn1_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)    return svtrn1_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svtrn1_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)    return svtrn1_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)   return svtrn1_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svtrn1_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svtrn1_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svtrn1_u64(a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V interleave_odd(V a, V b) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svtrn2_f32(a, b);
  else if constexpr (std::is_same_v<E, float64_t>) return svtrn2_f64(a, b);
  else if constexpr (std::is_same_v<E, int8_t>)     return svtrn2_s8(a, b);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svtrn2_u8(a, b);
  else if constexpr (std::is_same_v<E, int16_t>)    return svtrn2_s16(a, b);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svtrn2_u16(a, b);
  else if constexpr (std::is_same_v<E, int32_t>)    return svtrn2_s32(a, b);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svtrn2_u32(a, b);
  else if constexpr (std::is_same_v<E, int64_t>)    return svtrn2_s64(a, b);
  else if constexpr (std::is_same_v<E, float16_t>)   return svtrn2_f16(a, b);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svtrn2_bf16(a, b);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svtrn2_u16(svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
#endif
  else return svtrn2_u64(a, b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_interleave_lower(V a, V b) {
  using E = TypeOf<T>;
  auto a8 = details::to_u8<E>(a), b8 = details::to_u8<E>(b);
  auto a64 = svreinterpret_u64_u8(a8), b64 = svreinterpret_u64_u8(b8);
  auto alo = svuzp1_u64(a64, a64), blo = svuzp1_u64(b64, b64);
  auto aw = details::from_u8<E>(svreinterpret_u8_u64(alo));
  auto bw = details::from_u8<E>(svreinterpret_u8_u64(blo));
  if constexpr (std::is_same_v<E, float32_t>)      return svzip1_f32(aw, bw);
  else if constexpr (std::is_same_v<E, float64_t>) return svzip1_f64(aw, bw);
  else if constexpr (std::is_same_v<E, int8_t>)     return svzip1_s8(aw, bw);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svzip1_u8(aw, bw);
  else if constexpr (std::is_same_v<E, int16_t>)    return svzip1_s16(aw, bw);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svzip1_u16(aw, bw);
  else if constexpr (std::is_same_v<E, int32_t>)    return svzip1_s32(aw, bw);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svzip1_u32(aw, bw);
  else if constexpr (std::is_same_v<E, int64_t>)    return svzip1_s64(aw, bw);
  else if constexpr (std::is_same_v<E, float16_t>)   return svzip1_f16(aw, bw);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svzip1_bf16(aw, bw);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svzip1_u16(svreinterpret_u16_bf16(aw), svreinterpret_u16_bf16(bw)));
#endif
  else return svzip1_u64(aw, bw);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_interleave_upper(V a, V b) {
  using E = TypeOf<T>;
  auto a8 = details::to_u8<E>(a), b8 = details::to_u8<E>(b);
  auto a64 = svreinterpret_u64_u8(a8), b64 = svreinterpret_u64_u8(b8);
  auto ahi = svuzp2_u64(a64, a64), bhi = svuzp2_u64(b64, b64);
  auto aw = details::from_u8<E>(svreinterpret_u8_u64(ahi));
  auto bw = details::from_u8<E>(svreinterpret_u8_u64(bhi));
  if constexpr (std::is_same_v<E, float32_t>)      return svzip1_f32(aw, bw);
  else if constexpr (std::is_same_v<E, float64_t>) return svzip1_f64(aw, bw);
  else if constexpr (std::is_same_v<E, int8_t>)     return svzip1_s8(aw, bw);
  else if constexpr (std::is_same_v<E, uint8_t>)    return svzip1_u8(aw, bw);
  else if constexpr (std::is_same_v<E, int16_t>)    return svzip1_s16(aw, bw);
  else if constexpr (std::is_same_v<E, uint16_t>)   return svzip1_u16(aw, bw);
  else if constexpr (std::is_same_v<E, int32_t>)    return svzip1_s32(aw, bw);
  else if constexpr (std::is_same_v<E, uint32_t>)   return svzip1_u32(aw, bw);
  else if constexpr (std::is_same_v<E, int64_t>)    return svzip1_s64(aw, bw);
  else if constexpr (std::is_same_v<E, float16_t>)   return svzip1_f16(aw, bw);
#if defined(HAS_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svzip1_bf16(aw, bw);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>)  return svreinterpret_bf16_u16(
    svzip1_u16(svreinterpret_u16_bf16(aw), svreinterpret_u16_bf16(bw)));
#endif
  else return svzip1_u64(aw, bw);
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif
