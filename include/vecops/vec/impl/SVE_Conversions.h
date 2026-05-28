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
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(float32_t), _S(int8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::convert(t, word::promote(t1, v)); }

_TV _D(uint8_t), _S(float32_t) _ED
_FN(demote) { ScalableTag<int32_t> t1; return word::demote(t, word::convert(t1, v)); }
_TV _D(float32_t), _S(uint8_t) _ED
_FN(promote) { ScalableTag<int32_t> t1; return word::convert(t, word::promote(t1, v)); }

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

#undef _TV
#undef _ED
#undef _D
#undef _S
#undef _FN

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif  // VECOPS_SVE_CONVERSIONS_H
