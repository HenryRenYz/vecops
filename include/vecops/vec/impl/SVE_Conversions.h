//
// SVE_Conversions2.h — SVE register-level conversion operations (v2)
// Uses POW2-constrained Rebind signatures; multi-word fallback splits.
//

#ifndef VECOPS_SVE_CONVERSIONS_H
#define VECOPS_SVE_CONVERSIONS_H

#include <arm_sve.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

#include "./SVE_Basic.h"
#include "./SVE_Bf16.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* ======================================================================= */
/*     svcvtlt_* fallback for SVE without SVE2                              */
/*     svcvtlt converts the "long top" (odd elements).                      */
/*     Without SVE2, extract odds via svuzp2, then use regular svcvt.       */
/* ======================================================================= */
#if !defined(__ARM_FEATURE_SVE2)
VECOPS_VFUNC svfloat32_t sve_cvtlt_f32_f16(svbool_t pg, svfloat16_t v) {
  auto u16 = svreinterpret_u16_f16(v);
  auto odds_u16 = svuzp2_u16(u16, u16);
  auto full_u16 = svzip1_u16(odds_u16, odds_u16);
  return svcvt_f32_f16_x(pg, svreinterpret_f16_u16(full_u16));
}
VECOPS_VFUNC svfloat64_t sve_cvtlt_f64_f32(svbool_t pg, svfloat32_t v) {
  auto u32 = svreinterpret_u32_f32(v);
  auto odds_u32 = svuzp2_u32(u32, u32);
  auto full_u32 = svzip1_u32(odds_u32, odds_u32);
  return svcvt_f64_f32_x(pg, svreinterpret_f32_u32(full_u32));
}
#endif

/* ======================================================================= */
/*              Multi-word fallback forward declarations                    */
/* ======================================================================= */

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(Vec2Tag<V>{}) > 1 && num_words(T{}) > 1), bool> = true>
VECOPS_VFUNC Vec<T> promote(T t, V v);

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(Vec2Tag<V>{}) > 1 && num_words(T{}) > 1), bool> = true>
VECOPS_VFUNC Vec<T> demote(T t, V v);

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(T{}) > 1 && num_words(Vec2Tag<V>{}) > 1), bool> = true>
VECOPS_VFUNC Vec<T> convert(T t, V v);

/* ======================================================================= */
/*                         Identity / reshape                              */
/* ======================================================================= */

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<T> v) { return v; }
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<T> v) { return v; }
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> demote(T t, Vec<T> v) { return v; }

template <TLV_DECL_TAG(To), typename Vi, typename Ti = Vec2Tag<Vi>>
VECOPS_VFUNC Vec<To> reshape(To t_out, Vi v_in) {
  static_assert(std::is_same_v<TypeOf<To>, TypeOf<Ti>>, "Not same type");
  static_assert(std::is_same_v<Vec<To>, Vi>, "What");
  return v_in;
}

/* ======================================================================= */
/*         Convert (same-size pairs, POW2 <= 0, single-word only)          */
/* ======================================================================= */

// int64_t <=> float64_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  auto pg = details::ptrue<float64_t>();
  return svcvt_s64_f64_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  auto pg = details::ptrue<int64_t>();
  return svcvt_f64_s64_x(pg, v);
}

// uint64_t <=> float64_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  auto pg = details::ptrue<float64_t>();
  return svcvt_u64_f64_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  auto pg = details::ptrue<uint64_t>();
  return svcvt_f64_u64_x(pg, v);
}

// int64_t <=> uint64_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  auto pg = details::ptrue<uint64_t>();
  return svreinterpret_s64_u64(svmin_u64_z(pg, v, svdup_u64(std::numeric_limits<int64_t>::max())));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  auto pg = details::ptrue<int64_t>();
  return svreinterpret_u64_s64(svmax_s64_z(pg, v, svdup_s64(0)));
}

// int32_t <=> float32_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  return svcvt_s32_f32_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  auto pg = details::ptrue<int32_t>();
  return svcvt_f32_s32_x(pg, v);
}

// uint32_t <=> float32_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  return svcvt_u32_f32_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  auto pg = details::ptrue<uint32_t>();
  return svcvt_f32_u32_x(pg, v);
}

// int32_t <=> uint32_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  auto pg = details::ptrue<uint32_t>();
  return svreinterpret_s32_u32(svmin_u32_z(pg, v, svdup_u32(std::numeric_limits<int32_t>::max())));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  auto pg = details::ptrue<int32_t>();
  return svreinterpret_u32_s32(svmax_s32_z(pg, v, svdup_s32(0)));
}

// int16_t <=> uint16_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint16_t, T>> v) {
  auto pg = details::ptrue<uint16_t>();
  return svreinterpret_s16_u16(svmin_u16_z(pg, v, svdup_u16(std::numeric_limits<int16_t>::max())));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int16_t, T>> v) {
  auto pg = details::ptrue<int16_t>();
  return svreinterpret_u16_s16(svmax_s16_z(pg, v, svdup_s16(0)));
}

// int8_t <=> uint8_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint8_t, T>> v) {
  auto pg = details::ptrue<uint8_t>();
  return svreinterpret_s8_u8(svmin_u8_z(pg, v, svdup_u8(std::numeric_limits<int8_t>::max())));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int8_t, T>> v) {
  auto pg = details::ptrue<int8_t>();
  return svreinterpret_u8_s8(svmax_s8_z(pg, v, svdup_s8(0)));
}

// float16_t <=> int16_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int16_t, T>> v) {
  auto pg = details::ptrue<int16_t>();
  return svcvt_f16_s16_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float16_t, T>> v) {
  auto pg = details::ptrue<float16_t>();
  return svcvt_s16_f16_x(pg, v);
}

// float16_t <=> uint16_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint16_t, T>> v) {
  auto pg = details::ptrue<uint16_t>();
  return svcvt_f16_u16_x(pg, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float16_t, T>> v) {
  auto pg = details::ptrue<float16_t>();
  return svcvt_u16_f16_x(pg, v);
}

// float16_t <=> bfloat16_t
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<bfloat16_t, T>> v) {
  return details::f32x2_to_f16(details::bf16_to_f32_lo(v), details::bf16_to_f32_hi(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<float16_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  auto lo_f32 = svcvt_f32_f16_x(pg, v);
#if defined(__ARM_FEATURE_SVE2)
  auto hi_f32 = svcvtlt_f32_f16_x(pg, v);
#else
  auto hi_f32 = sve_cvtlt_f32_f16(pg, v);
#endif
#if defined(__ARM_FEATURE_SVE_BF16)
  auto pg_bf16 = details::ptrue<bfloat16_t>();
  auto lo_bf16 = svcvt_bf16_f32_x(pg_bf16, lo_f32);
  return svcvtnt_bf16_f32_x(lo_bf16, pg_bf16, hi_f32);
#else
  auto u32_lo = svlsr_n_u32_x(pg, svreinterpret_u32_f32(lo_f32), 16);
  auto u32_hi = svlsr_n_u32_x(pg, svreinterpret_u32_f32(hi_f32), 16);
  auto u16_lo = svreinterpret_u16_u32(u32_lo);
  auto u16_hi = svreinterpret_u16_u32(u32_hi);
  auto lo_compact = svuzp1_u16(u16_lo, svdup_n_u16(0));
  auto hi_compact = svuzp1_u16(u16_hi, svdup_n_u16(0));
  return svreinterpret_bf16_u16(svzip1_u16(lo_compact, hi_compact));
#endif
}

// bfloat16_t <=> int16_t (via f16)
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<bfloat16_t, T>> v) {
  auto f16_vec = details::f32x2_to_f16(details::bf16_to_f32_lo(v), details::bf16_to_f32_hi(v));
  auto pg = details::ptrue<float16_t>();
  return svcvt_s16_f16_x(pg, f16_vec);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<int16_t, T>> v) {
  auto pg_i32 = details::ptrue<int32_t>();
  auto lo_f32 = svcvt_f32_s32_x(pg_i32, svunpklo_s32(v));
  auto hi_f32 = svcvt_f32_s32_x(pg_i32, svunpkhi_s32(v));
  return details::f32x2_to_bf16(lo_f32, hi_f32);
}

// bfloat16_t <=> uint16_t (via f16)
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<bfloat16_t, T>> v) {
  auto f16_vec = details::f32x2_to_f16(details::bf16_to_f32_lo(v), details::bf16_to_f32_hi(v));
  auto pg = details::ptrue<float16_t>();
  return svcvt_u16_f16_x(pg, f16_vec);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> convert(T t, Vec<Rebind<uint16_t, T>> v) {
  auto pg_u32 = details::ptrue<uint32_t>();
  auto pg_f32 = details::ptrue<float32_t>();
  auto lo_f32 = svcvt_f32_u32_x(pg_f32, svunpklo_u32(v));
  auto hi_f32 = svcvt_f32_u32_x(pg_f32, svunpkhi_u32(v));
  return details::f32x2_to_bf16(lo_f32, hi_f32);
}

/* ======================================================================= */
/*              Promote (sizeof(Out) > sizeof(In), shift >= 1)              */
/* ======================================================================= */

/* ---------------  shift=1  (2x size)  ----------------- */
/* float32_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  return svcvt_f64_f32_x(pg, svzip1_f32(v, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  auto evens = svcvt_f64_f32_x(pg, v);
#if defined(__ARM_FEATURE_SVE2)
  auto odds  = svcvtlt_f64_f32_x(pg, v);
#else
  auto odds  = sve_cvtlt_f64_f32(pg, v);
#endif
  return word::reshape(t, svcreate2_f64(
    svzip1_f64(evens, odds), svzip2_f64(evens, odds)));
}

/* int32_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  auto pg = details::ptrue<int64_t>();
  auto lo64 = svunpklo_s64(v);
  return svcvt_f64_s64_x(pg, lo64);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  auto pg = details::ptrue<int64_t>();
  auto lo64 = svunpklo_s64(v);
  auto hi64 = svunpkhi_s64(v);
  auto lo_f64 = svcvt_f64_s64_x(pg, lo64);
  auto hi_f64 = svcvt_f64_s64_x(pg, hi64);
  return word::reshape(t, svcreate2_f64(lo_f64, hi_f64));
}

/* uint32_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  auto pg = details::ptrue<uint64_t>();
  auto lo64 = svunpklo_u64(v);
  return svcvt_f64_u64_x(pg, lo64);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  auto pg = details::ptrue<uint64_t>();
  auto lo64 = svunpklo_u64(v);
  auto hi64 = svunpkhi_u64(v);
  auto lo_f64 = svcvt_f64_u64_x(pg, lo64);
  auto hi_f64 = svcvt_f64_u64_x(pg, hi64);
  return word::reshape(t, svcreate2_f64(lo_f64, hi_f64));
}

/* int32_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return svunpklo_s64(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return word::reshape(t, svcreate2_s64(svunpklo_s64(v), svunpkhi_s64(v)));
}

/* uint32_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return svunpklo_u64(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return word::reshape(t, svcreate2_u64(svunpklo_u64(v), svunpkhi_u64(v)));
}

/* int32_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return svreinterpret_u64_s64(svunpklo_s64(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int64_t, T> t1;
  auto s64 = word::promote(t1, v);
  return word::reshape(t, svcreate2_u64(
    svreinterpret_u64_s64(svget2_s64(s64, 0)),
    svreinterpret_u64_s64(svget2_s64(s64, 1))));
}

/* uint32_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return svreinterpret_s64_u64(svunpklo_u64(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return word::reshape(t, svcreate2_s64(
    svreinterpret_s64_u64(svunpklo_u64(v)),
    svreinterpret_s64_u64(svunpkhi_u64(v))));
}

/* float32_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  return svcvt_s64_f32_x(pg, svzip1_f32(v, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg_f32 = details::ptrue<float32_t>();
  auto pg_f64 = details::ptrue<float64_t>();
  auto evens = svcvt_f64_f32_x(pg_f32, v);
#if defined(__ARM_FEATURE_SVE2)
  auto odds  = svcvtlt_f64_f32_x(pg_f32, v);
#else
  auto odds  = sve_cvtlt_f64_f32(pg_f32, v);
#endif
  auto lo = svcvt_s64_f64_x(pg_f64, svzip1_f64(evens, odds));
  auto hi = svcvt_s64_f64_x(pg_f64, svzip2_f64(evens, odds));
  return word::reshape(t, svcreate2_s64(lo, hi));
}

/* float32_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  return svcvt_u64_f32_x(pg, svzip1_f32(v, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  auto pg_f32 = details::ptrue<float32_t>();
  auto pg_f64 = details::ptrue<float64_t>();
  auto evens = svcvt_f64_f32_x(pg_f32, v);
#if defined(__ARM_FEATURE_SVE2)
  auto odds  = svcvtlt_f64_f32_x(pg_f32, v);
#else
  auto odds  = sve_cvtlt_f64_f32(pg_f32, v);
#endif
  auto lo = svcvt_u64_f64_x(pg_f64, svzip1_f64(evens, odds));
  auto hi = svcvt_u64_f64_x(pg_f64, svzip2_f64(evens, odds));
  return word::reshape(t, svcreate2_u64(lo, hi));
}

/* int16_t -> int32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return svunpklo_s32(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return word::reshape(t, svcreate2_s32(svunpklo_s32(v), svunpkhi_s32(v)));
}

/* uint16_t -> uint32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return svunpklo_u32(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return word::reshape(t, svcreate2_u32(svunpklo_u32(v), svunpkhi_u32(v)));
}

/* int16_t -> uint32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto v_i32 = word::promote(t1, v);
  return svreinterpret_u32_s32(v_i32);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto v_i32 = word::promote(t1, v);
  auto lo = svreinterpret_u32_s32(svget2_s32(v_i32, 0));
  auto hi = svreinterpret_u32_s32(svget2_s32(v_i32, 1));
  return word::reshape(t, svcreate2_u32(lo, hi));
}

/* uint16_t -> int32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return svreinterpret_s32_u32(svunpklo_u32(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return word::reshape(t, svcreate2_s32(
    svreinterpret_s32_u32(svunpklo_u32(v)),
    svreinterpret_s32_u32(svunpkhi_u32(v))));
}

/* int16_t -> float32_t (via int32_t) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* uint16_t -> float32_t (via int32_t) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* float16_t -> float32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  auto even_f32 = svcvt_f32_f16_x(pg, v);
#if defined(__ARM_FEATURE_SVE2)
  auto odd_f32  = svcvtlt_f32_f16_x(pg, v);
#else
  auto odd_f32  = sve_cvtlt_f32_f16(pg, v);
#endif
  return svzip1_f32(even_f32, odd_f32);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  auto pg = details::ptrue<float32_t>();
  auto even_f32 = svcvt_f32_f16_x(pg, v);
#if defined(__ARM_FEATURE_SVE2)
  auto odd_f32  = svcvtlt_f32_f16_x(pg, v);
#else
  auto odd_f32  = sve_cvtlt_f32_f16(pg, v);
#endif
  return word::reshape(t, svcreate2_f32(
    svzip1_f32(even_f32, odd_f32), svzip2_f32(even_f32, odd_f32)));
}

/* bfloat16_t -> float32_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  return details::bf16_to_f32_lo(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  return word::reshape(t, svcreate2_f32(
    details::bf16_to_f32_lo(v), details::bf16_to_f32_hi(v)));
}

/* int8_t -> int16_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return svunpklo_s16(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return word::reshape(t, svcreate2_s16(svunpklo_s16(v), svunpkhi_s16(v)));
}

/* uint8_t -> uint16_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return svunpklo_u16(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return word::reshape(t, svcreate2_u16(svunpklo_u16(v), svunpkhi_u16(v)));
}

/* int8_t -> uint16_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return svreinterpret_u16_s16(svunpklo_s16(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return word::reshape(t, svcreate2_u16(
    svreinterpret_u16_s16(svunpklo_s16(v)),
    svreinterpret_u16_s16(svunpkhi_s16(v))));
}

/* uint8_t -> int16_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return svreinterpret_s16_u16(svunpklo_u16(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return word::reshape(t, svcreate2_s16(
    svreinterpret_s16_u16(svunpklo_u16(v)),
    svreinterpret_s16_u16(svunpkhi_u16(v))));
}

/* ---------------  shift=2  (4x size)  ----------------- */
/* int16_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  auto lo32 = svunpklo_s32(v); auto hi32 = svunpkhi_s32(v);
  return word::reshape(t, svcreate4_s64(
    svunpklo_s64(lo32), svunpkhi_s64(lo32),
    svunpklo_s64(hi32), svunpkhi_s64(hi32)));
}

/* uint16_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  auto lo32 = svunpklo_u32(v); auto hi32 = svunpkhi_u32(v);
  return word::reshape(t, svcreate4_u64(
    svunpklo_u64(lo32), svunpkhi_u64(lo32),
    svunpklo_u64(hi32), svunpkhi_u64(hi32)));
}

/* int16_t -> uint64_t (via int64) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int64_t, T> t1;
  auto v_i64 = word::promote(t1, v);
  return svreinterpret_u64_s64(v_i64);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int64_t, T> t1;
  auto v_i64 = word::promote(t1, v);
  auto lo = svreinterpret_u64_s64(svget2_s64(v_i64, 0));
  auto hi = svreinterpret_u64_s64(svget2_s64(v_i64, 1));
  return word::reshape(t, svcreate2_u64(lo, hi));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int64_t, T> t1;
  auto v_i64 = word::promote(t1, v);
  return word::reshape(t, svcreate4_u64(
    svreinterpret_u64_s64(svget4_s64(v_i64, 0)),
    svreinterpret_u64_s64(svget4_s64(v_i64, 1)),
    svreinterpret_u64_s64(svget4_s64(v_i64, 2)),
    svreinterpret_u64_s64(svget4_s64(v_i64, 3))));
}

/* uint16_t -> int64_t (via int64) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  auto lo32 = svunpklo_u32(v); auto hi32 = svunpkhi_u32(v);
  return word::reshape(t, svcreate4_s64(
    svreinterpret_s64_u64(svunpklo_u64(lo32)),
    svreinterpret_s64_u64(svunpkhi_u64(lo32)),
    svreinterpret_s64_u64(svunpklo_u64(hi32)),
    svreinterpret_s64_u64(svunpkhi_u64(hi32))));
}

/* int16_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* uint16_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* float16_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* float16_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* float16_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* bfloat16_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* bfloat16_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* bfloat16_t -> float64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* int8_t -> int32_t (via int16) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  auto lo16 = svunpklo_s16(v); auto hi16 = svunpkhi_s16(v);
  return word::reshape(t, svcreate4_s32(
    svunpklo_s32(lo16), svunpkhi_s32(lo16),
    svunpklo_s32(hi16), svunpkhi_s32(hi16)));
}

/* uint8_t -> uint32_t (via uint16) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  auto lo16 = svunpklo_u16(v); auto hi16 = svunpkhi_u16(v);
  return word::reshape(t, svcreate4_u32(
    svunpklo_u32(lo16), svunpkhi_u32(lo16),
    svunpklo_u32(hi16), svunpkhi_u32(hi16)));
}

/* int8_t -> uint32_t (via int16->int32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto v_i32 = word::promote(t1, v);
  return svreinterpret_u32_s32(v_i32);
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto v_i32 = word::promote(t1, v);
  auto lo = svreinterpret_u32_s32(svget2_s32(v_i32, 0));
  auto hi = svreinterpret_u32_s32(svget2_s32(v_i32, 1));
  return word::reshape(t, svcreate2_u32(lo, hi));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto v_i32 = word::promote(t1, v);
  return word::reshape(t, svcreate4_u32(
    svreinterpret_u32_s32(svget4_s32(v_i32, 0)),
    svreinterpret_u32_s32(svget4_s32(v_i32, 1)),
    svreinterpret_u32_s32(svget4_s32(v_i32, 2)),
    svreinterpret_u32_s32(svget4_s32(v_i32, 3))));
}

/* uint8_t -> int32_t (via int16->int32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* int8_t -> float32_t (via int16->int32->f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  auto lo16 = svunpklo_s16(v); auto hi16 = svunpkhi_s16(v);
  auto lo32 = svunpklo_s32(lo16); auto hi32 = svunpklo_s32(hi16);
  auto pg = details::ptrue<int32_t>();
  return word::reshape(t, svcreate4_f32(
    svcvt_f32_s32_x(pg, lo32), svcvt_f32_s32_x(pg, svunpkhi_s32(lo16)),
    svcvt_f32_s32_x(pg, hi32),
    svcvt_f32_s32_x(pg, svunpkhi_s32(hi16))));
}

/* uint8_t -> float32_t (via uint16->uint32->f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  auto lo16 = svunpklo_u16(v); auto hi16 = svunpkhi_u16(v);
  auto lo32 = svunpklo_u32(lo16); auto hi32 = svunpklo_u32(hi16);
  auto pg = details::ptrue<uint32_t>();
  return word::reshape(t, svcreate4_f32(
    svcvt_f32_u32_x(pg, lo32), svcvt_f32_u32_x(pg, svunpkhi_u32(lo16)),
    svcvt_f32_u32_x(pg, hi32),
    svcvt_f32_u32_x(pg, svunpkhi_u32(hi16))));
}

/* float16_t -> int32_t (via f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* float16_t -> uint32_t (via f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<float16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* bfloat16_t -> int32_t (via f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* bfloat16_t -> uint32_t (via f32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<bfloat16_t, T>> v) {
  Rebind<float32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* ---------------  shift=3  (8x size)  max POW2=2 on SVE  -------------- */
/* int8_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* int8_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* uint8_t -> int64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* uint8_t -> uint64_t */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* int8_t -> float64_t (via int32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* uint8_t -> float64_t (via int32) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::promote(t, word::promote(t1, v));
}

/* float16_t -> int8_t / uint8_t (promote float16 to int8? no, that's demote direction) */
/* int8_t/uint8_t -> float16_t (promote, via int16_t/uint16_t) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* int8_t/uint8_t -> bfloat16_t (via float16_t) */
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<float16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<float16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<float16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::POW2 == 1), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<float16_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}

/* ======================================================================= */
/*    Demote base: sizeof(Out) < sizeof(In), POW2 <= -shift, single in     */
/* ======================================================================= */

/* ---------------  shift=1  (1/2 size)  ----------------- */
/* float64_t -> float32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<float64_t>();
  return svuzp1_f32(svcvt_f32_f64_x(pg, v), svcvt_f32_f64_x(pg, v));
}

/* float64_t -> int32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<float64_t>();
  return svuzp1_s32(svcvt_s32_f64_x(pg, v), svcvt_s32_f64_x(pg, v));
}

/* float64_t -> uint32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<float64_t>();
  return svuzp1_u32(svcvt_u32_f64_x(pg, v), svcvt_u32_f64_x(pg, v));
}

/* int64_t -> int32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int64_t>();
  auto clamped = svmax_s64_z(pg, svmin_s64_z(pg, v, svdup_s64(INT32_MAX)), svdup_s64(INT32_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s32(svqxtnb_s64(clamped), svqxtnb_s64(clamped));
#else
  auto u32 = svreinterpret_u32_s64(clamped);
  return svreinterpret_s32_u32(svuzp1_u32(u32, u32));
#endif
}

/* uint64_t -> uint32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint64_t>();
  auto clamped = svmin_u64_z(pg, v, svdup_u64(UINT32_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u32(svqxtnb_u64(clamped), svqxtnb_u64(clamped));
#else
  auto u32 = svreinterpret_u32_u64(clamped);
  return svuzp1_u32(u32, u32);
#endif
}

/* int64_t -> float32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int64_t>();
  return svuzp1_f32(svcvt_f32_s64_x(pg, v), svcvt_f32_s64_x(pg, v));
}

/* uint64_t -> float32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint64_t>();
  return svuzp1_f32(svcvt_f32_u64_x(pg, v), svcvt_f32_u64_x(pg, v));
}

/* uint64_t -> int32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint64_t>();
  auto clamped = svmin_u64_z(pg, v, svdup_u64(INT32_MAX));
  auto u32 = svreinterpret_u32_u64(clamped);
  return svreinterpret_s32_u32(svuzp1_u32(u32, u32));
}

/* int64_t -> uint32_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int64_t>();
  auto clamped = svmax_s64_z(pg, svmin_s64_z(pg, v, svdup_s64(UINT32_MAX)), svdup_s64(0));
  auto u32 = svreinterpret_u32_u64(svreinterpret_u64_s64(clamped));
  return svuzp1_u32(u32, u32);
}

/* int32_t -> int16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int32_t>();
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, v, svdup_s32(INT16_MAX)), svdup_s32(INT16_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s16(svqxtnb_s32(clamped), svqxtnb_s32(clamped));
#else
  auto u16 = svreinterpret_u16_s32(clamped);
  return svreinterpret_s16_u16(svuzp1_u16(u16, u16));
#endif
}

/* uint32_t -> uint16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint32_t>();
  auto clamped = svmin_u32_z(pg, v, svdup_u32(UINT16_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u16(svqxtnb_u32(clamped), svqxtnb_u32(clamped));
#else
  auto u16 = svreinterpret_u16_u32(clamped);
  return svuzp1_u16(u16, u16);
#endif
}

/* int32_t -> uint16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int32_t>();
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, v, svdup_s32(UINT16_MAX)), svdup_s32(0));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u16(svqxtunb_s32(clamped), svqxtunb_s32(clamped));
#else
  auto u16 = svreinterpret_u16_s32(clamped);
  return svuzp1_u16(u16, u16);
#endif
}

/* uint32_t -> int16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint32_t>();
  auto clamped = svmin_u32_z(pg, v, svdup_u32(INT16_MAX));
  auto u16 = svreinterpret_u16_u32(clamped);
  return svreinterpret_s16_u16(svuzp1_u16(u16, u16));
}

/* float32_t -> float16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<float16_t>();
  auto narrowed = svcvt_f16_f32_x(pg, v);
  return svreinterpret_f16_u16(svuzp1_u16(
    svreinterpret_u16_f16(narrowed), svreinterpret_u16_f16(narrowed)));
}

/* float32_t -> bfloat16_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  return details::f32x2_to_bf16(v, v);
}

/* int32_t -> bfloat16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* uint32_t -> bfloat16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* int32_t -> float16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* uint32_t -> float16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* int16_t -> int8_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int16_t>();
  auto clamped = svmax_s16_z(pg, svmin_s16_z(pg, v, svdup_s16(INT8_MAX)), svdup_s16(INT8_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s8(svqxtnb_s16(clamped), svqxtnb_s16(clamped));
#else
  auto u8 = svreinterpret_u8_s16(clamped);
  return svreinterpret_s8_u8(svuzp1_u8(u8, u8));
#endif
}

/* uint16_t -> uint8_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint16_t>();
  auto clamped = svmin_u16_z(pg, v, svdup_u16(UINT8_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u8(svqxtnb_u16(clamped), svqxtnb_u16(clamped));
#else
  auto u8 = svreinterpret_u8_u16(clamped);
  return svuzp1_u8(u8, u8);
#endif
}

/* int16_t -> uint8_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<int16_t>();
  auto clamped = svmax_s16_z(pg, svmin_s16_z(pg, v, svdup_s16(UINT8_MAX)), svdup_s16(0));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u8(svqxtunb_s16(clamped), svqxtunb_s16(clamped));
#else
  auto u8 = svreinterpret_u8_s16(clamped);
  return svuzp1_u8(u8, u8);
#endif
}

/* uint16_t -> int8_t */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  auto pg = details::ptrue<uint16_t>();
  auto clamped = svmin_u16_z(pg, v, svdup_u16(INT8_MAX));
  auto u8 = svreinterpret_u8_u16(clamped);
  return svreinterpret_s8_u8(svuzp1_u8(u8, u8));
}

/* float32_t -> int16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* float32_t -> uint16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* ======================================================================= */
/*    Demote shift=1: multi-word  (POW2 == -1 / POW2 == 0)                */
/* ======================================================================= */

/* --- POW2 == -1  (2-word input -> half-word output) --- */

/* float16_t -> int8_t (via int16_t)   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Tag<int16_t,-1,0> tc; auto lo = word::demote(t, word::convert(tc, word::lower(t1, v))); auto hi = word::demote(t, word::convert(tc, word::upper(t1, v))); return word::concat(t, lo, hi); }
/* float16_t -> uint8_t (via uint16_t) */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Tag<uint16_t,-1,0> tc; auto lo = word::demote(t, word::convert(tc, word::lower(t1, v))); auto hi = word::demote(t, word::convert(tc, word::upper(t1, v))); return word::concat(t, lo, hi); }
/* bfloat16_t -> int8_t (via float16_t->int16_t) */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Tag<float16_t,-1,0> tf; Tag<int16_t,-1,0> tc; auto lo = word::demote(t, word::convert(tc, word::convert(tf, word::lower(t1, v)))); auto hi = word::demote(t, word::convert(tc, word::convert(tf, word::upper(t1, v)))); return word::concat(t, lo, hi); }
/* bfloat16_t -> uint8_t (via float16_t->uint16_t)*/ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Tag<float16_t,-1,0> tf; Tag<uint16_t,-1,0> tc; auto lo = word::demote(t, word::convert(tc, word::convert(tf, word::lower(t1, v)))); auto hi = word::demote(t, word::convert(tc, word::convert(tf, word::upper(t1, v)))); return word::concat(t, lo, hi); }

/* --- POW2 == 0  (2-word input -> 1-word output) --- */

/* float64_t -> float32_t */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> int32_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> uint32_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> int32_t     */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> uint32_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> float32_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> int32_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> uint32_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> float32_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float32_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* int32_t -> int16_t     */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int32_t -> uint16_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> int16_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> uint16_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> float16_t */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> bfloat16_t*/ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> int16_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> uint16_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int32_t -> float16_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> float16_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int32_t -> bfloat16_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> bfloat16_t */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* int16_t -> int8_t      */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int16_t -> uint8_t     */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint16_t -> int8_t     */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint16_t -> uint8_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float16_t -> int8_t    */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; Tag<int16_t,-1,0> tc; auto lo = word::demote(t2, word::convert(tc, word::lower(t1, v))); auto hi = word::demote(t2, word::convert(tc, word::upper(t1, v))); return word::concat(t, lo, hi); }
/* float16_t -> uint8_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; Tag<uint16_t,-1,0> tc; auto lo = word::demote(t2, word::convert(tc, word::lower(t1, v))); auto hi = word::demote(t2, word::convert(tc, word::upper(t1, v))); return word::concat(t, lo, hi); }
/* bfloat16_t -> int8_t   */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; Tag<float16_t,-1,0> tf; Tag<int16_t,-1,0> tc; auto lo = word::demote(t2, word::convert(tc, word::convert(tf, word::lower(t1, v)))); auto hi = word::demote(t2, word::convert(tc, word::convert(tf, word::upper(t1, v)))); return word::concat(t, lo, hi); }
/* bfloat16_t -> uint8_t  */  template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; Tag<float16_t,-1,0> tf; Tag<uint16_t,-1,0> tc; auto lo = word::demote(t2, word::convert(tc, word::convert(tf, word::lower(t1, v)))); auto hi = word::demote(t2, word::convert(tc, word::convert(tf, word::upper(t1, v)))); return word::concat(t, lo, hi); }

/* ---------------  shift=2  (1/4 size)  ----------------- */
/* float64_t -> int16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float64_t -> uint16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float64_t -> float16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float64_t -> bfloat16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> int16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> uint16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> int16_t (via int32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> uint16_t (via uint32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<uint32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> float16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> float16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> bfloat16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> bfloat16_t (via float32_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float32_t -> int8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float32_t -> uint8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int32_t -> int8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int32_t -> uint8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint32_t -> int8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint32_t -> uint8_t (via uint16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<uint16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float16_t -> int8_t (via int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* float16_t -> uint8_t (via uint16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<uint16_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* bfloat16_t -> int8_t (via float16_t -> int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float16_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* bfloat16_t -> uint8_t (via float16_t -> uint16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, bfloat16_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<float16_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

/* ======================================================================= */
/*  Demote shift=2: multi-word  (POW2 == -1 / POW2 == 0)                  */
/* ======================================================================= */

/* --- POW2 == -1  (2-word input -> half-word output) --- */

/* float64_t -> int16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> uint16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> float16_t */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> bfloat16_t*/ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> int16_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> uint16_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> float16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> bfloat16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> int16_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> uint16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> float16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> bfloat16_t */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> int8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> uint8_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int32_t -> int8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Vec2Tag<V> t1; Half<T> t2;
  auto lo = word::demote(t2, word::lower(t1, v));
  auto hi = word::demote(t2, word::upper(t1, v));
  return word::concat(t, lo, hi);
}
/* int32_t -> uint8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> int8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> uint8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* --- POW2 == 0  (4-word input -> 1-word output) --- */

/* float64_t -> int16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> uint16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> float16_t */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> bfloat16_t*/ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> int16_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> uint16_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> float16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> bfloat16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> int16_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> uint16_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> float16_t  */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, float16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> bfloat16_t */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, bfloat16_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> int8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float32_t -> uint8_t   */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int32_t -> int8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Vec2Tag<V> t1; Half<T> t2;
  auto lo = word::demote(t2, word::lower(t1, v));
  auto hi = word::demote(t2, word::upper(t1, v));
  return word::concat(t, lo, hi);
}
/* int32_t -> uint8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> int8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint32_t -> uint8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint32_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* ---------------  shift=3  (1/8 size)  max POW2=2 on SVE  -------------- */
/* float64_t -> int8_t (via int32_t->int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* float64_t -> uint8_t (via int32_t->int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> int8_t (via int32_t->int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* int64_t -> uint8_t (via int32_t->int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> int8_t (via int32_t->int16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* uint64_t -> uint8_t (via uint32_t->uint16_t) */
template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 <= 0), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 1)>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Rebind<uint32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

/* ======================================================================= */
/*    Demote shift=3: multi-word  (POW2 == -2 / POW2 == -1)                */
/* ======================================================================= */

/* --- POW2 == -2  (2-word input -> quarter-word output) --- */

/* float64_t -> int8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> uint8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> int8_t       */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> uint8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> int8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> uint8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -2), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 2)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* --- POW2 == -1  (4-word input -> half-word output) --- */

/* float64_t -> int8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* float64_t -> uint8_t    */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, float64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> int8_t       */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* int64_t -> uint8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, int64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> int8_t      */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, int8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }
/* uint64_t -> uint8_t     */ template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>, TL_IF(T::POW2 == -1), TL_IF(is_any<TypeOf<T>, uint8_t>), TL_IF(is_any<TypeOf<Ti>, uint64_t>), TL_IF(num_words(Ti{}) == 4)> VECOPS_VFUNC Vec<T> demote(T t, V v) { Vec2Tag<V> t1; Half<T> t2; auto lo = word::demote(t2, word::lower(t1, v)); auto hi = word::demote(t2, word::upper(t1, v)); return word::concat(t, lo, hi); }

/* ======================================================================= */
/*              Multi-word fallback definitions (at bottom)                 */
/* ======================================================================= */

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(Vec2Tag<V>{}) > 1 && num_words(T{}) > 1), bool>>
VECOPS_VFUNC Vec<T> promote(T t, V v) {
  Vec2Tag<V> t1;
  Half<T> t2;
  auto lo = word::promote(t2, word::lower(t1, v));
  auto hi = word::promote(t2, word::upper(t1, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(Vec2Tag<V>{}) > 1 && num_words(T{}) > 1), bool>>
VECOPS_VFUNC Vec<T> demote(T t, V v) {
  Vec2Tag<V> t1;
  Half<T> t2;
  auto lo = word::demote(t2, word::lower(t1, v));
  auto hi = word::demote(t2, word::upper(t1, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V),
          std::enable_if_t<(num_words(T{}) > 1 && num_words(Vec2Tag<V>{}) > 1), bool>>
VECOPS_VFUNC Vec<T> convert(T t, V v) {
  Vec2Tag<V> t1;
  Half<T> t2;
  auto lo = word::convert(t2, word::lower(t1, v));
  auto hi = word::convert(t2, word::upper(t1, v));
  return word::concat(t, lo, hi);
}

/* ======================================================================= */
/*                   Interleaved native conversions                         */
/* ======================================================================= */

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote_even(T, Vec<ViewAs<float16_t, T>> v) {
  return svcvt_f32_f16_x(details::ptrue<float32_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote_even(T, Vec<ViewAs<float32_t, T>> v) {
  return svcvt_f64_f32_x(details::ptrue<float64_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> demote_even(T, Vec<ViewAs<float32_t, T>> v) {
  return svcvt_f16_f32_z(details::ptrue<float32_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> demote_even(T, Vec<ViewAs<float64_t, T>> v) {
  return svcvt_f32_f64_z(details::ptrue<float64_t>(), v);
}

#if defined(__ARM_FEATURE_SVE2)
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> promote_odd(T, Vec<ViewAs<float16_t, T>> v) {
  return svcvtlt_f32_f16_x(details::ptrue<float32_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> promote_odd(T, Vec<ViewAs<float32_t, T>> v) {
  return svcvtlt_f64_f32_x(details::ptrue<float64_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Vec<T> demote_odd(
    T, Vec<ViewAs<float32_t, T>> v, Vec<T> fallback
) {
  return svcvtnt_f16_f32_m(fallback, details::ptrue<float32_t>(), v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> demote_odd(
    T, Vec<ViewAs<float64_t, T>> v, Vec<T> fallback
) {
  return svcvtnt_f32_f64_m(fallback, details::ptrue<float64_t>(), v);
}

#define VECOPS_SVE_INTERLEAVED_INT(TO, TI, SHLLB, SHLLT, QXTNB, QXTNT)      \
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, TO>)>                    \
VECOPS_VFUNC Vec<T> promote_even(T, Vec<ViewAs<TI, T>> v) {                \
  return SHLLB(v, 0);                                                        \
}                                                                           \
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, TO>)>                    \
VECOPS_VFUNC Vec<T> promote_odd(T, Vec<ViewAs<TI, T>> v) {                 \
  return SHLLT(v, 0);                                                        \
}                                                                           \
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, TI>)>                    \
VECOPS_VFUNC Vec<T> demote_even(T, Vec<ViewAs<TO, T>> v) {                 \
  return QXTNB(v);                                                           \
}                                                                           \
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, TI>)>                    \
VECOPS_VFUNC Vec<T> demote_odd(                                             \
    T, Vec<ViewAs<TO, T>> v, Vec<T> fallback                               \
) {                                                                         \
  return QXTNT(fallback, v);                                                 \
}

VECOPS_SVE_INTERLEAVED_INT(
    int16_t, int8_t, svshllb_n_s16, svshllt_n_s16, svqxtnb_s16, svqxtnt_s16
)
VECOPS_SVE_INTERLEAVED_INT(
    int32_t, int16_t, svshllb_n_s32, svshllt_n_s32, svqxtnb_s32, svqxtnt_s32
)
VECOPS_SVE_INTERLEAVED_INT(
    int64_t, int32_t, svshllb_n_s64, svshllt_n_s64, svqxtnb_s64, svqxtnt_s64
)
VECOPS_SVE_INTERLEAVED_INT(
    uint16_t, uint8_t, svshllb_n_u16, svshllt_n_u16, svqxtnb_u16, svqxtnt_u16
)
VECOPS_SVE_INTERLEAVED_INT(
    uint32_t, uint16_t, svshllb_n_u32, svshllt_n_u32, svqxtnb_u32, svqxtnt_u32
)
VECOPS_SVE_INTERLEAVED_INT(
    uint64_t, uint32_t, svshllb_n_u64, svshllt_n_u64, svqxtnb_u64, svqxtnt_u64
)

#undef VECOPS_SVE_INTERLEAVED_INT
#endif

namespace interleaved_conversion_details {

template <typename To, typename Ti>
inline constexpr bool native_widen_float =
    (is_any<TypeOf<To>, float32_t> && is_any<TypeOf<Ti>, float16_t>) ||
    (is_any<TypeOf<To>, float64_t> && is_any<TypeOf<Ti>, float32_t>);

template <typename To, typename Ti>
inline constexpr bool native_narrow_float =
    native_widen_float<Ti, To>;

#if defined(__ARM_FEATURE_SVE2)
template <typename To, typename Ti>
inline constexpr bool native_widen_integer =
    std::is_integral_v<TypeOf<To>> &&
    std::is_integral_v<TypeOf<Ti>> &&
    std::is_signed_v<TypeOf<To>> == std::is_signed_v<TypeOf<Ti>> &&
    sizeof(TypeOf<To>) == 2 * sizeof(TypeOf<Ti>);

template <typename To, typename Ti>
inline constexpr bool native_narrow_integer =
    native_widen_integer<Ti, To>;
#else
template <typename To, typename Ti>
inline constexpr bool native_widen_integer = false;

template <typename To, typename Ti>
inline constexpr bool native_narrow_integer = false;
#endif

template <typename To, typename Ti>
inline constexpr bool native_promote_even =
    native_widen_float<To, Ti> || native_widen_integer<To, Ti>;

template <typename To, typename Ti>
inline constexpr bool native_promote_odd =
#if defined(__ARM_FEATURE_SVE2)
    native_widen_float<To, Ti> || native_widen_integer<To, Ti>;
#else
    false;
#endif

template <typename To, typename Ti>
inline constexpr bool native_demote_even =
    native_narrow_float<To, Ti> || native_narrow_integer<To, Ti>;

template <typename To, typename Ti>
inline constexpr bool native_demote_odd_fallback =
#if defined(__ARM_FEATURE_SVE2)
    native_narrow_float<To, Ti> || native_narrow_integer<To, Ti>;
#else
    false;
#endif

template <int Levels, TLV_DECL_TAG(T), TLV_DECL_VEC(V)>
VECOPS_VFUNC auto select_even(T t, V v) {
  if constexpr (Levels == 0) {
    return v;
  } else {
    return select_even<Levels - 1>(Half<T>{}, word::even(t, v));
  }
}

template <int Levels, TLV_DECL_TAG(T), TLV_DECL_VEC(V)>
VECOPS_VFUNC Vec<T> insert_even(T t, V values, Vec<T> fallback) {
  if constexpr (Levels == 0) {
    return values;
  } else {
    auto fallback_even = word::even(t, fallback);
    auto fallback_odd = word::odd(t, fallback);
    auto result_even = insert_even<Levels - 1>(Half<T>{}, values, fallback_even);
    return word::interleave(t, result_even, fallback_odd);
  }
}

} // namespace interleaved_conversion_details

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
  requires (!interleaved_conversion_details::native_promote_even<
            To, Vec2Tag<Vi>>)
VECOPS_VFUNC Vec<To> promote_even(To to, Vi vi) {
  using Ti = Vec2Tag<Vi>;
  constexpr Ti ti;
  constexpr int levels = log2_floor(sizeof(TypeOf<To>) / sizeof(TypeOf<Ti>));
  auto selected = interleaved_conversion_details::select_even<levels>(ti, vi);
  return word::promote(to, selected);
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
  requires (!interleaved_conversion_details::native_promote_odd<
            To, Vec2Tag<Vi>>)
VECOPS_VFUNC Vec<To> promote_odd(To to, Vi vi) {
  constexpr Vec2Tag<Vi> ti;
  return word::promote(to, word::odd(ti, vi));
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC Vec<To> demote_even(To to, Vi vi, Vec<To> fallback) {
  using Ti = Vec2Tag<Vi>;
  constexpr Ti ti;
  constexpr int levels = log2_floor(sizeof(TypeOf<Ti>) / sizeof(TypeOf<To>));
  using TCompact = Rebind<TypeOf<To>, Ti>;
  auto compact = word::demote(TCompact{}, vi);
  return interleaved_conversion_details::insert_even<levels>(to, compact, fallback);
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
  requires (!interleaved_conversion_details::native_demote_even<
            To, Vec2Tag<Vi>>)
VECOPS_VFUNC Vec<To> demote_even(To to, Vi vi) {
  return word::demote_even(to, vi, word::zeros(to));
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
  requires (!interleaved_conversion_details::native_demote_odd_fallback<
            To, Vec2Tag<Vi>>)
VECOPS_VFUNC Vec<To> demote_odd(To to, Vi vi, Vec<To> fallback) {
  using Ti = Vec2Tag<Vi>;
  constexpr Ti ti;
  using TCompact = Rebind<TypeOf<To>, Ti>;
  auto compact = word::demote(TCompact{}, vi);
  return word::interleave(to, word::even(to, fallback), compact);
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC Vec<To> demote_odd(To to, Vi vi) {
  return word::demote_odd(to, vi, word::zeros(to));
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif  // VECOPS_SVE_CONVERSIONS_H
