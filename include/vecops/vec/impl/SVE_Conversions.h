//
// SVE_Conversions.h — SVE register-level conversion operations
// Provides convert, promote, demote, and reshape for all type pairs.
// All functions use generic V parameters (no Rebind) to avoid POW2
// adjustments on SVE.  Source type is validated via Vec2Tag<V>.
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
/*                         Generic Identity / reshape                       */
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

// ---- short-hand macros for function signatures ----
#define _TV   template <TLV_DECL_TAG(T), typename V, typename Ti = Vec2Tag<std::remove_cvref_t<V>>,
#define _ED   >
#define _D(X) TL_IF(is_any<TypeOf<T>, X>)
#define _S(X) TL_IF(is_any<TypeOf<Ti>, X>)
#define _FN(N) VECOPS_VFUNC Vec<T> N(T t, V v)

/* ======================================================================= */
/*                          int64_t <=> float64_t                          */
/* ======================================================================= */
_TV _D(int64_t), _S(float64_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float64_t>(); return svcvt_s64_f64_x(pg, v); }
_TV _D(float64_t), _S(int64_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<int64_t>(); return svcvt_f64_s64_x(pg, v); }

/* ======================================================================= */
/*                         uint64_t <=> float64_t                          */
/* ======================================================================= */
_TV _D(uint64_t), _S(float64_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float64_t>(); return svcvt_u64_f64_x(pg, v); }
_TV _D(float64_t), _S(uint64_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<uint64_t>(); return svcvt_f64_u64_x(pg, v); }

/* ======================================================================= */
/*                          int64_t <=> uint64_t                           */
/* ======================================================================= */
_TV _D(int64_t), _S(uint64_t) _ED
_FN(convert) { return svreinterpret_s64_u64(v); }
_TV _D(uint64_t), _S(int64_t) _ED
_FN(convert) { return svreinterpret_u64_s64(v); }

/* ======================================================================= */
/*                        float32_t <=> float64_t                          */
/* ======================================================================= */
_TV _D(float32_t), _S(float64_t) _ED
_FN(demote) { auto pg = sve_detail::sve_ptrue<float64_t>(); return svuzp1_f32(svcvt_f32_f64_x(pg, v), svcvt_f32_f64_x(pg, v)); }
_TV _D(float64_t), _S(float32_t) _ED
_FN(promote) { auto pg = sve_detail::sve_ptrue<float32_t>(); return svcvt_f64_f32_x(pg, svzip1_f32(v, v)); }

/* ======================================================================= */
/*                         float32_t <=> int64_t                           */
/* ======================================================================= */
_TV _D(float32_t), _S(int64_t) _ED
_FN(demote) { auto pg = sve_detail::sve_ptrue<int64_t>(); return svuzp1_f32(svcvt_f32_s64_x(pg, v), svcvt_f32_s64_x(pg, v)); }
_TV _D(int64_t), _S(float32_t) _ED
_FN(promote) { auto pg = sve_detail::sve_ptrue<float32_t>(); return svcvt_s64_f32_x(pg, svzip1_f32(v, v)); }

/* ======================================================================= */
/*                        float32_t <=> uint64_t                           */
/* ======================================================================= */
_TV _D(float32_t), _S(uint64_t) _ED
_FN(demote) { auto pg = sve_detail::sve_ptrue<uint64_t>(); return svuzp1_f32(svcvt_f32_u64_x(pg, v), svcvt_f32_u64_x(pg, v)); }
_TV _D(uint64_t), _S(float32_t) _ED
_FN(promote) { auto pg = sve_detail::sve_ptrue<float32_t>(); return svcvt_u64_f32_x(pg, svzip1_f32(v, v)); }

/* ======================================================================= */
/*                         int32_t <=> float64_t                           */
/* ======================================================================= */
_TV _D(int32_t), _S(float64_t) _ED
_FN(demote) { auto pg = sve_detail::sve_ptrue<float64_t>(); return svuzp1_s32(svcvt_s32_f64_x(pg, v), svcvt_s32_f64_x(pg, v)); }
_TV _D(float64_t), _S(int32_t) _ED
_FN(promote) { auto pg = sve_detail::sve_ptrue<int32_t>(); return svcvt_f64_s32_x(pg, svzip1_s32(v, v)); }

/* ======================================================================= */
/*                         int32_t <=> int64_t                             */
/* ======================================================================= */
_TV _D(int32_t), _S(int64_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int64_t>();
  auto clamped = svmax_s64_z(pg, svmin_s64_z(pg, v, svdup_s64(INT32_MAX)), svdup_s64(INT32_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s32(svqxtnb_s64(clamped), svqxtnb_s64(clamped));
#else
  auto u32 = svreinterpret_u32_s64(clamped);
  return svreinterpret_s32_u32(svuzp1_u32(u32, u32));
#endif
}
_TV _D(int64_t), _S(int32_t) _ED
_FN(promote) { return svunpklo_s64(v); }

/* ======================================================================= */
/*                        int32_t <=> uint64_t                             */
/*   source=u64, target=i32: clamp u64 to [0, INT32_MAX], narrow, reinterpret */
/* ======================================================================= */
_TV _D(int32_t), _S(uint64_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint64_t>();
  auto clamped = svmin_u64_z(pg, v, svdup_u64(INT32_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svreinterpret_s32_u32(svuzp1_u32(svqxtnb_u64(clamped), svqxtnb_u64(clamped)));
#else
  auto u32 = svreinterpret_u32_u64(clamped);
  return svreinterpret_s32_u32(svuzp1_u32(u32, u32));
#endif
}
_TV _D(uint64_t), _S(int32_t) _ED
_FN(promote) { ScalableTag<int64_t> t1; return svreinterpret_u64_s64(word::promote(t1, v)); }

/* ======================================================================= */
/*                        uint32_t <=> float64_t                           */
/* ======================================================================= */
_TV _D(uint32_t), _S(float64_t) _ED
_FN(demote) { auto pg = sve_detail::sve_ptrue<float64_t>(); return svuzp1_u32(svcvt_u32_f64_x(pg, v), svcvt_u32_f64_x(pg, v)); }
_TV _D(float64_t), _S(uint32_t) _ED
_FN(promote) { auto pg = sve_detail::sve_ptrue<uint32_t>(); return svcvt_f64_u32_x(pg, svzip1_u32(v, v)); }

/* ======================================================================= */
/*                        uint32_t <=> int64_t                             */
/*   source=i64, target=u32: clamp i64 to [0, UINT32_MAX], narrow, reinterpret */
/* ======================================================================= */
_TV _D(uint32_t), _S(int64_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int64_t>();
  auto clamped = svmax_s64_z(pg, svmin_s64_z(pg, v, svdup_s64(UINT32_MAX)), svdup_s64(0));
  auto u32 = svreinterpret_u32_u64(svreinterpret_u64_s64(clamped));
  return svuzp1_u32(u32, u32);
}
_TV _D(int64_t), _S(uint32_t) _ED
_FN(promote) { return svreinterpret_s64_u64(svunpklo_u64(v)); }

/* ======================================================================= */
/*                       uint32_t <=> uint64_t                             */
/* ======================================================================= */
_TV _D(uint32_t), _S(uint64_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint64_t>();
  auto clamped = svmin_u64_z(pg, v, svdup_u64(UINT32_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u32(svqxtnb_u64(clamped), svqxtnb_u64(clamped));
#else
  auto u32 = svreinterpret_u32_u64(clamped);
  return svuzp1_u32(u32, u32);
#endif
}
_TV _D(uint64_t), _S(uint32_t) _ED
_FN(promote) { return svunpklo_u64(v); }

/* ======================================================================= */
/*                        float32_t <=> int32_t                            */
/* ======================================================================= */
_TV _D(float32_t), _S(int32_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<int32_t>(); return svcvt_f32_s32_x(pg, v); }
_TV _D(int32_t), _S(float32_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float32_t>(); return svcvt_s32_f32_x(pg, v); }

/* ======================================================================= */
/*                       float32_t <=> uint32_t                            */
/* ======================================================================= */
_TV _D(float32_t), _S(uint32_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<uint32_t>(); return svcvt_f32_u32_x(pg, v); }
_TV _D(uint32_t), _S(float32_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float32_t>(); return svcvt_u32_f32_x(pg, v); }

/* ======================================================================= */
/*                        int32_t <=> uint32_t                             */
/* ======================================================================= */
_TV _D(int32_t), _S(uint32_t) _ED  _FN(convert) { return svreinterpret_s32_u32(v); }
_TV _D(uint32_t), _S(int32_t) _ED  _FN(convert) { return svreinterpret_u32_s32(v); }

/* ======================================================================= */
/*                        int16_t <=> int32_t                              */
/* ======================================================================= */
_TV _D(int16_t), _S(int32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int32_t>();
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, v, svdup_s32(INT16_MAX)), svdup_s32(INT16_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s16(svqxtnb_s32(clamped), svqxtnb_s32(clamped));
#else
  auto u16 = svreinterpret_u16_s32(clamped);
  return svreinterpret_s16_u16(svuzp1_u16(u16, u16));
#endif
}
_TV _D(int32_t), _S(int16_t) _ED
_FN(promote) { return svunpklo_s32(v); }

/* ======================================================================= */
/*                        int16_t <=> uint32_t                             */
/*   source=u32, target=i16: clamp u32 to [0, INT16_MAX], narrow, reinterpret */
/* ======================================================================= */
_TV _D(int16_t), _S(uint32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint32_t>();
  auto clamped = svmin_u32_z(pg, v, svdup_u32(INT16_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svreinterpret_s16_u16(svuzp1_u16(svqxtnb_u32(clamped), svqxtnb_u32(clamped)));
#else
  auto u16 = svreinterpret_u16_u32(clamped);
  return svreinterpret_s16_u16(svuzp1_u16(u16, u16));
#endif
}
_TV _D(uint32_t), _S(int16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return svreinterpret_u32_s32(word::promote(t1, v)); }

/* ======================================================================= */
/*                       int16_t <=> float32_t (via int32)                 */
/* ======================================================================= */
_TV _D(int16_t), _S(float32_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(float32_t), _S(int16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::convert(t, word::promote(t1, v)); }

/* ======================================================================= */
/*                        uint16_t <=> int32_t                             */
/*   source=i32, target=u16: clamp i32 to [0, UINT16_MAX], narrow, reinterpret */
/* ======================================================================= */
_TV _D(uint16_t), _S(int32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int32_t>();
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, v, svdup_s32(UINT16_MAX)), svdup_s32(0));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u16(svqxtunb_s32(clamped), svqxtunb_s32(clamped));
#else
  auto u16 = svreinterpret_u16_s32(clamped);
  return svuzp1_u16(u16, u16);
#endif
}

_TV _D(int32_t), _S(uint16_t) _ED
_FN(promote) { return svreinterpret_s32_u32(svunpklo_u32(v)); }

/* ======================================================================= */
/*                       uint16_t <=> float32_t (via int32)                */
/* ======================================================================= */
/*                       uint16_t <=> uint32_t                             */
/* ======================================================================= */
_TV _D(uint16_t), _S(uint32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint32_t>();
  auto clamped = svmin_u32_z(pg, v, svdup_u32(UINT16_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u16(svqxtnb_u32(clamped), svqxtnb_u32(clamped));
#else
  auto u16 = svreinterpret_u16_u32(clamped);
  return svuzp1_u16(u16, u16);
#endif
}
_TV _D(uint32_t), _S(uint16_t) _ED
_FN(promote) { return svunpklo_u32(v); }

/* ======================================================================= */
/*                      uint16_t <=> float32_t (via int32)                */
/* ======================================================================= */
_TV _D(uint16_t), _S(float32_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(float32_t), _S(uint16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::convert(t, word::promote(t1, v)); }

/* ======================================================================= */
/*      int16_t / uint16_t  <=>  float64_t / int64_t / uint64_t           */
/*   (multi-step chains through int32_t or int64_t)                        */
/* ======================================================================= */
_TV _D(int16_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float64_t), _S(int16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(int16_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int64_t), _S(int16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(int16_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint64_t), _S(int16_t) _ED
_FN(promote) { ScalableTag<int64_t> t1; return svreinterpret_u64_s64(word::promote(t1, v)); }

_TV _D(uint16_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float64_t), _S(uint16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(uint16_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int64_t), _S(uint16_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(uint16_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<uint32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint64_t), _S(uint16_t) _ED
_FN(promote) { ScalableTag<int64_t> t1; return svreinterpret_u64_s64(word::promote(t1, v)); }

/* ======================================================================= */
/*                         int8_t <=> int16_t                              */
/* ======================================================================= */
_TV _D(int8_t), _S(int16_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int16_t>();
  auto clamped = svmax_s16_z(pg, svmin_s16_z(pg, v, svdup_s16(INT8_MAX)), svdup_s16(INT8_MIN));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_s8(svqxtnb_s16(clamped), svqxtnb_s16(clamped));
#else
  auto u8 = svreinterpret_u8_s16(clamped);
  return svreinterpret_s8_u8(svuzp1_u8(u8, u8));
#endif
}
_TV _D(int16_t), _S(int8_t) _ED
_FN(promote) { return svunpklo_s16(v); }

/* ======================================================================= */
/*                         int8_t <=> uint16_t                             */
/*   source=u16, target=i8: clamp u16 to [0, INT8_MAX], narrow, reinterpret */

_TV _D(int8_t), _S(uint16_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint16_t>();
  auto clamped = svmin_u16_z(pg, v, svdup_u16(INT8_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svreinterpret_s8_u8(svuzp1_u8(svqxtnb_u16(clamped), svqxtnb_u16(clamped)));
#else
  auto u8 = svreinterpret_u8_u16(clamped);
  return svreinterpret_s8_u8(svuzp1_u8(u8, u8));
#endif
}
_TV _D(uint16_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int16_t> t1; return svreinterpret_u16_s16(word::promote(t1, v)); }

/* ======================================================================= */
/*                        uint8_t <=> int16_t                              */
/*   source=i16, target=u8: clamp i16 to [0, UINT8_MAX], narrow, reinterpret */
/* ======================================================================= */
_TV _D(uint8_t), _S(int16_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<int16_t>();
  auto clamped = svmax_s16_z(pg, svmin_s16_z(pg, v, svdup_s16(UINT8_MAX)), svdup_s16(0));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u8(svqxtunb_s16(clamped), svqxtunb_s16(clamped));
#else
  auto u8 = svreinterpret_u8_s16(clamped);
  return svuzp1_u8(u8, u8);
#endif
}
_TV _D(int16_t), _S(uint8_t) _ED
_FN(promote) { return svreinterpret_s16_u16(svunpklo_u16(v)); }

/* ======================================================================= */
/*                        uint8_t <=> uint16_t                             */
/* ======================================================================= */
_TV _D(uint8_t), _S(uint16_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<uint16_t>();
  auto clamped = svmin_u16_z(pg, v, svdup_u16(UINT8_MAX));
#if defined(__ARM_FEATURE_SVE2)
  return svuzp1_u8(svqxtnb_u16(clamped), svqxtnb_u16(clamped));
#else
  auto u8 = svreinterpret_u8_u16(clamped);
  return svuzp1_u8(u8, u8);
#endif
}
_TV _D(uint16_t), _S(uint8_t) _ED
_FN(promote) { return svunpklo_u16(v); }

/* ======================================================================= */
/*        int8_t / uint8_t  <=>  float32_t / int32_t / uint32_t           */
/*   (multi-step chains through int16_t / int32_t)                         */
/* ======================================================================= */
_TV _D(int8_t), _S(float32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto i32 = svcvt_s32_f32_x(pg, v);
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, i32, svdup_s32(INT8_MAX)), svdup_s32(INT8_MIN));
  auto u16 = svreinterpret_u16_s32(clamped);
  auto u8 = svreinterpret_u8_u16(svuzp1_u16(u16, u16));
  return svreinterpret_s8_u8(svuzp1_u8(u8, u8));
}
_TV _D(float32_t), _S(int8_t) _ED
_FN(promote) {
  auto i16 = svunpklo_s16(v);
  auto i32 = svunpklo_s32(i16);
  auto pg = sve_detail::sve_ptrue<int32_t>();
  return svcvt_f32_s32_x(pg, i32);
}

_TV _D(uint8_t), _S(float32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto i32 = svcvt_s32_f32_x(pg, v);
  auto clamped = svmax_s32_z(pg, svmin_s32_z(pg, i32, svdup_s32(UINT8_MAX)), svdup_s32(0));
  auto u16 = svreinterpret_u16_s32(clamped);
  auto u8 = svuzp1_u8(svreinterpret_u8_u16(svuzp1_u16(u16, u16)), svreinterpret_u8_u16(svuzp1_u16(u16, u16)));
  return u8;
}
_TV _D(float32_t), _S(uint8_t) _ED
_FN(promote) {
  auto u16 = svunpklo_u16(v);
  auto u32 = svunpklo_u32(u16);
  auto pg = sve_detail::sve_ptrue<uint32_t>();
  return svcvt_f32_u32_x(pg, u32);
}

_TV _D(int8_t), _S(int32_t) _ED
_FN(demote) { ScalableTag<int16_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int32_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int16_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(int8_t), _S(uint32_t) _ED
_FN(demote) { ScalableTag<int16_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint32_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return svreinterpret_u32_s32(word::promote(t1, v)); }

_TV _D(uint8_t), _S(int32_t) _ED
_FN(demote) { ScalableTag<int16_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int32_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int16_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(uint8_t), _S(uint32_t) _ED
_FN(demote) { ScalableTag<uint16_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint32_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return svreinterpret_u32_s32(word::promote(t1, v)); }

/* ======================================================================= */
/*    int8_t / uint8_t  <=>  float64_t / int64_t / uint64_t               */
/*   (multi-step chains through int32_t / int64_t)                         */
/* ======================================================================= */
_TV _D(int8_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float64_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(uint8_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float64_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(int8_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int64_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(int8_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint64_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int64_t> t1; return svreinterpret_u64_s64(word::promote(t1, v)); }

_TV _D(uint8_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(int64_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::promote(t, word::promote(t1, v)); }

_TV _D(uint8_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<uint32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(uint64_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int64_t> t1; return svreinterpret_u64_s64(word::promote(t1, v)); }

/* ======================================================================= */
/*                        int8_t <=> uint8_t                               */
/* ======================================================================= */
_TV _D(int8_t), _S(uint8_t) _ED  _FN(convert) { return svreinterpret_s8_u8(v); }
_TV _D(uint8_t), _S(int8_t) _ED  _FN(convert) { return svreinterpret_u8_s8(v); }

/* ======================================================================= */
/*                      int16_t <=> uint16_t                               */
/* ======================================================================= */
_TV _D(int16_t), _S(uint16_t) _ED  _FN(convert) { return svreinterpret_s16_u16(v); }
_TV _D(uint16_t), _S(int16_t) _ED  _FN(convert) { return svreinterpret_u16_s16(v); }

/* ================================================================ */
/*              float16_t <=> float32_t  (foundation)               */
/* ================================================================ */
_TV _D(float16_t), _S(float32_t) _ED
_FN(demote) {
  auto pg = sve_detail::sve_ptrue<float16_t>();
  auto narrowed = svcvt_f16_f32_x(pg, v);
  return svreinterpret_f16_u16(svuzp1_u16(svreinterpret_u16_f16(narrowed), svreinterpret_u16_f16(narrowed)));
}
_TV _D(float32_t), _S(float16_t) _ED
_FN(promote) {
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto even_f32 = svcvt_f32_f16_x(pg, v);
  auto odd_f32 = svcvtlt_f32_f16_x(pg, v);
  return svzip1_f32(even_f32, odd_f32);
}

/* ================================================================ */
/*              float16_t <=> int16_t  (convert, same-size)         */
/* ================================================================ */
_TV _D(float16_t), _S(int16_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<int16_t>(); return svcvt_f16_s16_x(pg, v); }
_TV _D(int16_t), _S(float16_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float16_t>(); return svcvt_s16_f16_x(pg, v); }

/* ================================================================ */
/*            float16_t <=> uint16_t  (convert, same-size)          */
/* ================================================================ */
_TV _D(float16_t), _S(uint16_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<uint16_t>(); return svcvt_f16_u16_x(pg, v); }
_TV _D(uint16_t), _S(float16_t) _ED
_FN(convert) { auto pg = sve_detail::sve_ptrue<float16_t>(); return svcvt_u16_f16_x(pg, v); }

/* ================================================================ */
/*          float16_t <=> larger types  (demote / promote)           */
/*         float16_t from int32/uint32  (via f32)                   */
/* ================================================================ */
_TV _D(float16_t), _S(int32_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(float16_t), _S(uint32_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::convert(t1, v)); }

/* ================================================================ */
/*        float16_t from int64/uint64/float64  (via f32)            */
/* ================================================================ */
_TV _D(float16_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float16_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(float16_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }

/* ================================================================ */
/*           f16/bf16 -> int8/uint8  (demote, via f32)               */
/* ================================================================ */
_TV _D(int8_t), _S(float16_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; ScalableTag<int32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }
_TV _D(uint8_t), _S(float16_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; ScalableTag<int32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }

/* ================================================================ */
/*           int8/uint8 -> float16_t  (promote, via f32)             */
/* ================================================================ */
_TV _D(float16_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; ScalableTag<float32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }
_TV _D(float16_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; ScalableTag<float32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }

/* ================================================================ */
/*          float16_t -> larger types  (promote, via f32)           */
/*    float16 -> int32/uint32                                     */
/* ================================================================ */
_TV _D(int32_t), _S(float16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::convert(t, word::promote(t1, v)); }
_TV _D(uint32_t), _S(float16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::convert(t, word::promote(t1, v)); }

/* ================================================================ */
/*  float16 -> int64 / uint64 / float64             */
/* ================================================================ */
_TV _D(int64_t), _S(float16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }
_TV _D(uint64_t), _S(float16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }
_TV _D(float64_t), _S(float16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }

/* ================================================================ */
/*              bfloat16_t <=> float32_t  (foundation)              */
/* ================================================================ */
_TV _D(float32_t), _S(bfloat16_t) _ED
_FN(promote) { return sve_detail::bf16_to_f32_lo(v); }

_TV _D(bfloat16_t), _S(float32_t) _ED
_FN(demote) {
#if defined(__ARM_FEATURE_SVE_BF16)
  auto pg = sve_detail::sve_ptrue<bfloat16_t>();
  auto tmp = svcvt_bf16_f32_z(pg, v);
  return svreinterpret_bf16_u16(svuzp1_u16(svreinterpret_u16_bf16(tmp), svreinterpret_u16_bf16(tmp)));
#else
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto u32 = svlsr_n_u32_x(pg, svreinterpret_u32_f32(v), 16);
  auto u16 = svreinterpret_u16_u32(u32);
  return svreinterpret_bf16_u16(svuzp1_u16(u16, u16));
#endif
}

/* ================================================================ */
/*         bfloat16_t -> larger types  (promote, via f32)            */
/*      bfloat16 -> int32 / uint32 / float32                       */
/* ================================================================ */
_TV _D(int32_t), _S(bfloat16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::convert(t, word::promote(t1, v)); }
_TV _D(uint32_t), _S(bfloat16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::convert(t, word::promote(t1, v)); }

/* ================================================================ */
/*  bfloat16 -> int64 / uint64 / float64           */
/* ================================================================ */
_TV _D(int64_t), _S(bfloat16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }
_TV _D(uint64_t), _S(bfloat16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }
_TV _D(float64_t), _S(bfloat16_t) _ED
_FN(promote) { ScalableTag<float32_t> t1; return word::promote(t, word::promote(t1, v)); }

/* ================================================================ */
/*          int8/uint8 -> bfloat16_t  (promote, via f32)              */
/* ================================================================ */
_TV _D(bfloat16_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; ScalableTag<float32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }
_TV _D(bfloat16_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; ScalableTag<float32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }

/* ================================================================ */
/*         larger types -> bfloat16_t  (demote, via f32)             */
/*       int32 / uint32 / float32                                  */
/* ================================================================ */
_TV _D(bfloat16_t), _S(int32_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(bfloat16_t), _S(uint32_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::convert(t1, v)); }

/* ================================================================ */
/*  int64 / uint64 / float64 -> bfloat16   */
/* ================================================================ */
_TV _D(bfloat16_t), _S(int64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(bfloat16_t), _S(uint64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }
_TV _D(bfloat16_t), _S(float64_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; return word::demote(t, word::demote(t1, v)); }

/* ================================================================ */
/*     bfloat16_t -> int8/uint8  (demote, via f32)                   */
/* ================================================================ */
_TV _D(int8_t), _S(bfloat16_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; ScalableTag<int32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }
_TV _D(uint8_t), _S(bfloat16_t) _ED
_FN(demote) { ScalableTag<float32_t> t1; ScalableTag<int32_t> t2; return word::demote(t, word::convert(t2, word::promote(t1, v))); }

/* ================================================================ */
/*     bfloat16_t <=> int16_t  (convert, same-size, via f32)         */
/* ================================================================ */
_TV _D(int16_t), _S(bfloat16_t) _ED
_FN(convert) {
  auto lo_f32 = sve_detail::bf16_to_f32_lo(v);
  auto hi_f32 = sve_detail::bf16_to_f32_hi(v);
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto lo_i32 = svcvt_s32_f32_x(pg, lo_f32);
  auto hi_i32 = svcvt_s32_f32_x(pg, hi_f32);
  auto lo_clamped = svmax_s32_z(pg, svmin_s32_z(pg, lo_i32, svdup_s32(INT16_MAX)), svdup_s32(INT16_MIN));
  auto hi_clamped = svmax_s32_z(pg, svmin_s32_z(pg, hi_i32, svdup_s32(INT16_MAX)), svdup_s32(INT16_MIN));
  auto lo_u16 = svreinterpret_u16_s32(lo_clamped);
  auto hi_u16 = svreinterpret_u16_s32(hi_clamped);
  auto lo_s16 = svreinterpret_s16_u16(svuzp1_u16(lo_u16, lo_u16));
  auto hi_s16 = svreinterpret_s16_u16(svuzp1_u16(hi_u16, hi_u16));
  return svzip1_s16(lo_s16, hi_s16);
}

_TV _D(bfloat16_t), _S(int16_t) _ED
_FN(convert) {
  auto pg_u16 = sve_detail::sve_ptrue<uint16_t>();
  auto u16_v = svreinterpret_u16_s16(v);
  auto n = static_cast<uint16_t>(sve_detail::sve_cnt<int16_t>());
  auto q = static_cast<uint16_t>(n >> 2);
  auto rot_idx = svadd_u16_x(pg_u16, svindex_u16(0, 1), svdup_u16(q));
  auto v_rot_u16 = svtbl_u16(u16_v, rot_idx);
  auto v_rot = svreinterpret_s16_u16(v_rot_u16);
  auto pg_i32 = sve_detail::sve_ptrue<int32_t>();
  auto i32_0 = svunpklo_s32(v);
  auto i32_1 = svunpklo_s32(v_rot);
  auto i32_2 = svunpkhi_s32(v);
  auto i32_3 = svunpkhi_s32(v_rot);
  auto f32_0 = svcvt_f32_s32_x(pg_i32, i32_0);
  auto f32_1 = svcvt_f32_s32_x(pg_i32, i32_1);
  auto f32_2 = svcvt_f32_s32_x(pg_i32, i32_2);
  auto f32_3 = svcvt_f32_s32_x(pg_i32, i32_3);
  auto lo_f32 = svzip1_f32(f32_0, f32_1);
  auto hi_f32 = svzip1_f32(f32_2, f32_3);
  return sve_detail::f32x2_to_bf16(lo_f32, hi_f32);
}

/* ================================================================ */
/*    bfloat16_t <=> uint16_t  (convert, same-size, via f32)         */
/* ================================================================ */
_TV _D(uint16_t), _S(bfloat16_t) _ED
_FN(convert) {
  auto lo_f32 = sve_detail::bf16_to_f32_lo(v);
  auto hi_f32 = sve_detail::bf16_to_f32_hi(v);
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto lo_i32 = svcvt_s32_f32_x(pg, lo_f32);
  auto hi_i32 = svcvt_s32_f32_x(pg, hi_f32);
  auto lo_clamped = svmax_s32_z(pg, svmin_s32_z(pg, lo_i32, svdup_s32(UINT16_MAX)), svdup_s32(0));
  auto hi_clamped = svmax_s32_z(pg, svmin_s32_z(pg, hi_i32, svdup_s32(UINT16_MAX)), svdup_s32(0));
  auto lo_u16 = svreinterpret_u16_s32(lo_clamped);
  auto hi_u16 = svreinterpret_u16_s32(hi_clamped);
  return svzip1_u16(svuzp1_u16(lo_u16, lo_u16), svuzp1_u16(hi_u16, hi_u16));
}

_TV _D(bfloat16_t), _S(uint16_t) _ED
_FN(convert) {
  auto pg_u16 = sve_detail::sve_ptrue<uint16_t>();
  auto u16_v = v;
  auto n = static_cast<uint16_t>(sve_detail::sve_cnt<int16_t>());
  auto q = static_cast<uint16_t>(n >> 2);
  auto rot_idx = svadd_u16_x(pg_u16, svindex_u16(0, 1), svdup_u16(q));
  auto v_rot_u16 = svtbl_u16(u16_v, rot_idx);
  auto pg_u32 = sve_detail::sve_ptrue<uint32_t>();
  auto u32_0 = svunpklo_u32(u16_v);
  auto u32_1 = svunpklo_u32(v_rot_u16);
  auto u32_2 = svunpkhi_u32(u16_v);
  auto u32_3 = svunpkhi_u32(v_rot_u16);
  auto pg_f32 = sve_detail::sve_ptrue<float32_t>();
  auto f32_0 = svcvt_f32_u32_x(pg_f32, u32_0);
  auto f32_1 = svcvt_f32_u32_x(pg_f32, u32_1);
  auto f32_2 = svcvt_f32_u32_x(pg_f32, u32_2);
  auto f32_3 = svcvt_f32_u32_x(pg_f32, u32_3);
  auto lo_f32 = svzip1_f32(f32_0, f32_1);
  auto hi_f32 = svzip1_f32(f32_2, f32_3);
  return sve_detail::f32x2_to_bf16(lo_f32, hi_f32);
}

/* ================================================================ */
/*     bfloat16_t <=> float16_t  (convert, same-size, via f32)       */
/* ================================================================ */
_TV _D(float16_t), _S(bfloat16_t) _ED
_FN(convert) {
  auto lo_f32 = sve_detail::bf16_to_f32_lo(v);
  auto hi_f32 = sve_detail::bf16_to_f32_hi(v);
  auto pg = sve_detail::sve_ptrue<float32_t>();
  auto lo_f16 = svcvt_f16_f32_x(pg, lo_f32);
  auto hi_f16 = svcvt_f16_f32_x(pg, hi_f32);
  return svreinterpret_f16_u16(svzip1_u16(svreinterpret_u16_f16(lo_f16), svreinterpret_u16_f16(hi_f16)));
}

_TV _D(bfloat16_t), _S(float16_t) _ED
_FN(convert) {
  auto pg = sve_detail::sve_ptrue<float16_t>();
  auto lo_f32 = svcvt_f32_f16_x(pg, v);
  auto n = static_cast<uint16_t>(sve_detail::sve_cnt<float16_t>());
  auto half = static_cast<uint16_t>(n >> 1);
  auto pg_u16 = sve_detail::sve_ptrue<uint16_t>();
  auto u16_v = svreinterpret_u16_f16(v);
  auto hi_idx = svadd_u16_x(pg_u16, svindex_u16(0, 1), svdup_u16(half));
  auto hi_u16 = svtbl_u16(u16_v, hi_idx);
  auto hi_f16 = svreinterpret_f16_u16(hi_u16);
  auto hi_f32 = svcvt_f32_f16_x(pg, hi_f16);
  return sve_detail::f32x2_to_bf16(lo_f32, hi_f32);
}

#undef _TV
#undef _ED
#undef _D
#undef _S
#undef _FN

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif  // VECOPS_SVE_CONVERSIONS_H
