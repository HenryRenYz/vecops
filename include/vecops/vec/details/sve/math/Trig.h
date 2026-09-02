#ifndef VECOPS_VEC_DETAILS_SVE_MATH_TRIG_H
#define VECOPS_VEC_DETAILS_SVE_MATH_TRIG_H

/**
 * @file Trig.h
 * @brief SVE backend for the circular trigonometric families.
 *
 * The f32/f64 radian reduction, the use of SVE's FTSSEL/FTSMUL/FTMAD
 * instructions, and the pi-scaled polynomial data are derived from Arm
 * optimized-routines commit 67126040cf80f956676fbf473c2d9bebdb475283,
 * files math/aarch64/sve/{sin,cos,sincos,tan}{,f}.c,
 * math/aarch64/sve/{sinpi,cospi,sincospi,tanpi}{,f}.c and their common
 * headers. Copyright (c) 2019-2026, Arm Limited. Those sources are
 * dual-licensed under "MIT OR Apache-2.0 WITH LLVM-exception".
 *
 * Strict radians use the complete hardware polynomial and a three-part
 * Cody-Waite reduction on the hot domain. Large finite arguments take a
 * cold per-lane libm fallback; keeping it out of line preserves the hot
 * path while providing full-domain semantics until the considerably larger
 * vector Payne-Hanek tables are justified here. Fast and Estimate retain
 * fewer FTMAD stages. Pi-scaled functions never form pi*x: they reduce x
 * exactly modulo one and use progressively shorter direct polynomials.
 * Every grid value required by C23 (integers and half-integers) is repaired
 * explicitly after the approximation, including its prescribed sign.
 *
 * SVE's trigonometric-assist instructions also have f16 forms. Fast and
 * Estimate use them directly on the common small-argument domain; lanes whose
 * quadrant or pole handling needs more precision fall back to the paired-f32
 * route. Strict keeps that route for its 1-ULP narrow-format contract. bf16
 * always widens. SVE2 uses FCVTLT/FCVTNT for the upper half; base SVE uses the
 * same unzip/transpose route as the other widening math families.
 */

#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Radian f32/f64 reduction and hardware-polynomial evaluation              //
/* **************************************************************************** */

struct alignas(16) SVETrigF32Data {
  float neg_pio2_1, neg_pio2_2, neg_pio2_3, inv_pio2;
};

inline constexpr SVETrigF32Data kSVETrigF32{
    -0x1.921fb6p+0f, 0x1.777a5cp-25f, 0x1.ee59dap-50f,
    0x1.45f306p-1f};

struct alignas(16) SVETrigF64Data {
  double inv_pio2, pio2_1;
  double pio2_2, pio2_3;
};

inline constexpr SVETrigF64Data kSVETrigF64{
    0x1.45f306dc9c882p-1, 0x1.921fb54442d18p+0,
    0x1.1a62633145c07p-54, -0x1.f1976b7ed8fbcp-110};

VECOPS_ALWAYS_INLINE svfloat32x2_t sve_trig_reduce_f32(svfloat32_t x) {
  const auto pt = svptrue_b32();
  const auto constants = svld1rq_f32(pt, &kSVETrigF32.neg_pio2_1);
  const auto q = svmla_lane_f32(
      svdup_n_f32(0x1.8p+23f), x, constants, 3);
  const auto n = svsub_n_f32_x(pt, q, 0x1.8p+23f);
  auto r = svmla_lane_f32(x, n, constants, 0);
  r = svmla_lane_f32(r, n, constants, 1);
  r = svmla_lane_f32(r, n, constants, 2);
  return svcreate2_f32(r, q);
}

VECOPS_ALWAYS_INLINE svfloat64x2_t sve_trig_reduce_f64(svfloat64_t x) {
  const auto pt = svptrue_b64();
  const auto leading = svld1rq_f64(pt, &kSVETrigF64.inv_pio2);
  const auto trailing = svld1rq_f64(pt, &kSVETrigF64.pio2_2);
  const auto q = svmla_lane_f64(
      svdup_n_f64(0x1.8p+52), x, leading, 0);
  const auto n = svsub_n_f64_x(pt, q, 0x1.8p+52);
  auto r = svmls_lane_f64(x, n, leading, 1);
  r = svmls_lane_f64(r, n, trailing, 0);
  r = svmls_lane_f64(r, n, trailing, 1);
  return svcreate2_f64(r, q);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_ftmad_f32(
    svfloat32_t r2, svfloat32_t initial = svdup_n_f32(0.0f)) {
  auto y = initial;
  if constexpr (Tier == Accuracy::Strict) y = svtmad_f32(y, r2, 4);
  if constexpr (Tier != Accuracy::Estimate) y = svtmad_f32(y, r2, 3);
  y = svtmad_f32(y, r2, 2);
  y = svtmad_f32(y, r2, 1);
  return svtmad_f32(y, r2, 0);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_ftmad_f64(
    svfloat64_t r2, svfloat64_t initial = svdup_n_f64(0.0)) {
  auto y = initial;
  if constexpr (Tier == Accuracy::Strict) {
    y = svtmad_f64(y, r2, 7);
    y = svtmad_f64(y, r2, 6);
  }
  if constexpr (Tier != Accuracy::Estimate) {
    y = svtmad_f64(y, r2, 5);
    y = svtmad_f64(y, r2, 4);
  }
  y = svtmad_f64(y, r2, 3);
  y = svtmad_f64(y, r2, 2);
  y = svtmad_f64(y, r2, 1);
  return svtmad_f64(y, r2, 0);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_trig_component_f32(
    svfloat32_t r, svuint32_t quadrant) {
  const auto factor = svtssel_f32(r, quadrant);
  const auto r2 = svtsmul_f32(r, quadrant);
  return svmul_f32_x(
      svptrue_b32(), factor, sve_ftmad_f32<Tier>(r2));
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_trig_component_f64(
    svfloat64_t r, svuint64_t quadrant) {
  const auto factor = svtssel_f64(r, quadrant);
  const auto r2 = svtsmul_f64(r, quadrant);
  return svmul_f64_x(
      svptrue_b64(), factor, sve_ftmad_f64<Tier>(r2));
}

template <TrigKind Kind>
VECOPS_NOINLINE inline svfloat32_t sve_trig_scalar_fallback_f32(
    svbool_t special, svfloat32_t x, svfloat32_t current) {
  alignas(256) float input[64];
  alignas(256) float output[64];
  alignas(256) std::uint32_t selected[64];
  const auto pt = svptrue_b32();
  svst1_f32(pt, input, x);
  svst1_f32(pt, output, current);
  svst1_u32(pt, selected,
            svsel_u32(special, svdup_n_u32(1), svdup_n_u32(0)));
  const auto count = static_cast<std::uint64_t>(svcntw());
  for (std::uint64_t lane = 0; lane < count; ++lane) {
    if (selected[lane] == 0) continue;
    if constexpr (Kind == TrigKind::Sin) output[lane] = std::sin(input[lane]);
    else if constexpr (Kind == TrigKind::Cos)
      output[lane] = std::cos(input[lane]);
    else output[lane] = std::tan(input[lane]);
  }
  return svld1_f32(pt, output);
}

template <TrigKind Kind>
VECOPS_NOINLINE inline svfloat64_t sve_trig_scalar_fallback_f64(
    svbool_t special, svfloat64_t x, svfloat64_t current) {
  alignas(256) double input[32];
  alignas(256) double output[32];
  alignas(256) std::uint64_t selected[32];
  const auto pt = svptrue_b64();
  svst1_f64(pt, input, x);
  svst1_f64(pt, output, current);
  svst1_u64(pt, selected,
            svsel_u64(special, svdup_n_u64(1), svdup_n_u64(0)));
  const auto count = static_cast<std::uint64_t>(svcntd());
  for (std::uint64_t lane = 0; lane < count; ++lane) {
    if (selected[lane] == 0) continue;
    if constexpr (Kind == TrigKind::Sin) output[lane] = std::sin(input[lane]);
    else if constexpr (Kind == TrigKind::Cos)
      output[lane] = std::cos(input[lane]);
    else output[lane] = std::tan(input[lane]);
  }
  return svld1_f64(pt, output);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32x2_t sve_radian_sincos_f32(
    svfloat32_t x, svbool_t pg) {
  const auto reduced = sve_trig_reduce_f32(x);
  const auto r = svget2_f32(reduced, 0);
  const auto sin_q = svreinterpret_u32_f32(svget2_f32(reduced, 1));
  const auto cos_q = svadd_n_u32_x(svptrue_b32(), sin_q, 1);
  auto sin_y = sve_trig_component_f32<Tier>(r, sin_q);
  auto cos_y = sve_trig_component_f32<Tier>(r, cos_q);
  const auto special = svacge_n_f32(pg, x, 0x1p20f);
  if (svptest_any(pg, special)) {
    sin_y = sve_trig_scalar_fallback_f32<TrigKind::Sin>(special, x, sin_y);
    cos_y = sve_trig_scalar_fallback_f32<TrigKind::Cos>(special, x, cos_y);
  }
  const auto zero = svcmpeq_n_f32(pg, x, 0.0f);
  sin_y = svsel_f32(zero, x, sin_y);
  cos_y = svsel_f32(zero, svdup_n_f32(1.0f), cos_y);
  return svcreate2_f32(sin_y, cos_y);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64x2_t sve_radian_sincos_f64(
    svfloat64_t x, svbool_t pg) {
  const auto reduced = sve_trig_reduce_f64(x);
  const auto r = svget2_f64(reduced, 0);
  const auto sin_q = svreinterpret_u64_f64(svget2_f64(reduced, 1));
  const auto cos_q = svadd_n_u64_x(svptrue_b64(), sin_q, 1);
  auto sin_y = sve_trig_component_f64<Tier>(r, sin_q);
  auto cos_y = sve_trig_component_f64<Tier>(r, cos_q);
  const auto special = svacge_n_f64(pg, x, 0x1p23);
  if (svptest_any(pg, special)) {
    sin_y = sve_trig_scalar_fallback_f64<TrigKind::Sin>(special, x, sin_y);
    cos_y = sve_trig_scalar_fallback_f64<TrigKind::Cos>(special, x, cos_y);
  }
  const auto zero = svcmpeq_n_f64(pg, x, 0.0);
  sin_y = svsel_f64(zero, x, sin_y);
  cos_y = svsel_f64(zero, svdup_n_f64(1.0), cos_y);
  return svcreate2_f64(sin_y, cos_y);
}

template <TrigKind Kind, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_radian_trig_f32(
    svfloat32_t x, svbool_t pg) {
  const auto sc = sve_radian_sincos_f32<Tier>(x, pg);
  if constexpr (Kind == TrigKind::Sin) return svget2_f32(sc, 0);
  else if constexpr (Kind == TrigKind::Cos) return svget2_f32(sc, 1);
  else {
    auto y = svdiv_f32_x(pg, svget2_f32(sc, 0), svget2_f32(sc, 1));
    const auto special = svacge_n_f32(pg, x, 0x1p15f);
    if (svptest_any(pg, special))
      y = sve_trig_scalar_fallback_f32<TrigKind::Tan>(special, x, y);
    return svsel_f32(svcmpeq_n_f32(pg, x, 0.0f), x, y);
  }
}

template <TrigKind Kind, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_radian_trig_f64(
    svfloat64_t x, svbool_t pg) {
  const auto sc = sve_radian_sincos_f64<Tier>(x, pg);
  if constexpr (Kind == TrigKind::Sin) return svget2_f64(sc, 0);
  else if constexpr (Kind == TrigKind::Cos) return svget2_f64(sc, 1);
  else {
    auto y = svdiv_f64_x(pg, svget2_f64(sc, 0), svget2_f64(sc, 1));
    return svsel_f64(svcmpeq_n_f64(pg, x, 0.0), x, y);
  }
}

/* **************************************************************************** */
//    Pi-scaled direct kernels                                                  //
/* **************************************************************************** */

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_sinpi_poly_f32(
    svfloat32_t r, svbool_t pg) {
  const auto r2 = svmul_f32_x(pg, r, r);
  svfloat32_t p;
  if constexpr (Tier == Accuracy::Strict) {
    p = svdup_n_f32(-0x1.e30750p-8f);
    p = svmla_f32_x(pg, svdup_n_f32(0x1.50783p-4f), r2, p);
    p = svmla_f32_x(pg, svdup_n_f32(-0x1.32d2ccp-1f), r2, p);
  } else if constexpr (Tier == Accuracy::Fast) {
    p = svdup_n_f32(-0x1.32d2ccp-1f);
  } else {
    p = svdup_n_f32(0x1.466bc6p1f);
  }
  if constexpr (Tier != Accuracy::Estimate)
    p = svmla_f32_x(pg, svdup_n_f32(0x1.466bc6p1f), r2, p);
  p = svmla_f32_x(pg, svdup_n_f32(-0x1.4abbcep2f), r2, p);
  p = svmla_f32_x(pg, svdup_n_f32(0x1.921fb6p1f), r2, p);
  return svmul_f32_x(pg, r, p);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_sinpi_poly_f64(
    svfloat64_t r, svbool_t pg) {
  const auto r2 = svmul_f64_x(pg, r, r);
  svfloat64_t p;
  if constexpr (Tier == Accuracy::Strict) {
    p = svdup_n_f64(-0x1.012a9870eeb7dp-25);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.af86ae521260bp-21), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(-0x1.6fc0032b3c29fp-16), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.e8f48308acda4p-12), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(-0x1.e30750a28c88ep-8), r2, p);
  } else if constexpr (Tier == Accuracy::Fast) {
    p = svdup_n_f64(0x1.e8f48308acda4p-12);
  } else {
    p = svdup_n_f64(0x1.507834891188ep-4);
  }
  if constexpr (Tier == Accuracy::Fast)
    p = svmla_f64_x(pg, svdup_n_f64(-0x1.e30750a28c88ep-8), r2, p);
  if constexpr (Tier != Accuracy::Estimate)
    p = svmla_f64_x(pg, svdup_n_f64(0x1.507834891188ep-4), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(-0x1.32d2cce62dc33p-1), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.466bc6775ab16p1), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(-0x1.4abbce625be53p2), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.921fb54442d184p1), r2, p);
  return svmul_f64_x(pg, r, p);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32x2_t sve_pi_sincos_f32(
    svfloat32_t x, svbool_t pg) {
  const auto pt = svptrue_b32();
  const auto n = svrinta_f32_x(pg, x);
  auto sr = svsub_f32_x(pt, x, n);
  auto cr = svsubr_n_f32_x(pt, svabs_f32_x(pg, sr), 0.5f);
  const auto convertible = svaclt_n_f32(pg, x, 0x1p31f);
  const auto odd = svlsl_n_u32_z(
      convertible, svreinterpret_u32_s32(svcvt_s32_f32_z(pg, n)), 31);
  sr = svreinterpret_f32_u32(
      sveor_u32_x(pt, svreinterpret_u32_f32(sr), odd));
  cr = svreinterpret_f32_u32(
      sveor_u32_m(convertible, svreinterpret_u32_f32(cr), odd));
  auto sin_y = sve_sinpi_poly_f32<Tier>(sr, pg);
  auto cos_y = sve_sinpi_poly_f32<Tier>(cr, pg);

  const auto integer = svcmpeq_n_f32(pg, svsub_f32_x(pt, x, n), 0.0f);
  const auto half = svcmpeq_n_f32(
      pg, svabs_f32_x(pg, svsub_f32_x(pt, x, n)), 0.5f);
  const auto x_sign = svand_n_u32_x(
      pt, svreinterpret_u32_f32(x), 0x80000000u);
  sin_y = svsel_f32(integer, svreinterpret_f32_u32(x_sign), sin_y);
  sin_y = svsel_f32(
      half, svreinterpret_f32_u32(svorr_u32_x(
                pt, svand_n_u32_x(pt, svreinterpret_u32_f32(sin_y),
                                  0x80000000u),
                svdup_n_u32(0x3f800000u))), sin_y);
  cos_y = svsel_f32(integer,
      svreinterpret_f32_u32(svorr_u32_x(
          pt, odd, svdup_n_u32(0x3f800000u))), cos_y);
  cos_y = svsel_f32(half, svdup_n_f32(0.0f), cos_y);
  return svcreate2_f32(sin_y, cos_y);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64x2_t sve_pi_sincos_f64(
    svfloat64_t x, svbool_t pg) {
  const auto pt = svptrue_b64();
  const auto n = svrinta_f64_x(pg, x);
  auto sr0 = svsub_f64_x(pt, x, n);
  auto sr = sr0;
  auto cr = svsubr_n_f64_x(pt, svabs_f64_x(pg, sr), 0.5);
  const auto convertible = svaclt_n_f64(pg, x, 0x1p63);
  const auto odd = svlsl_n_u64_z(
      convertible, svreinterpret_u64_s64(svcvt_s64_f64_z(pg, n)), 63);
  sr = svreinterpret_f64_u64(
      sveor_u64_x(pt, svreinterpret_u64_f64(sr), odd));
  cr = svreinterpret_f64_u64(
      sveor_u64_m(convertible, svreinterpret_u64_f64(cr), odd));
  auto sin_y = sve_sinpi_poly_f64<Tier>(sr, pg);
  auto cos_y = sve_sinpi_poly_f64<Tier>(cr, pg);

  const auto integer = svcmpeq_n_f64(pg, sr0, 0.0);
  const auto half = svcmpeq_n_f64(pg, svabs_f64_x(pg, sr0), 0.5);
  const auto x_sign = svand_n_u64_x(
      pt, svreinterpret_u64_f64(x), 0x8000000000000000ull);
  sin_y = svsel_f64(integer, svreinterpret_f64_u64(x_sign), sin_y);
  sin_y = svsel_f64(
      half, svreinterpret_f64_u64(svorr_u64_x(
                pt, svand_n_u64_x(pt, svreinterpret_u64_f64(sin_y),
                                  0x8000000000000000ull),
                svdup_n_u64(0x3ff0000000000000ull))), sin_y);
  cos_y = svsel_f64(integer,
      svreinterpret_f64_u64(svorr_u64_x(
          pt, odd, svdup_n_u64(0x3ff0000000000000ull))), cos_y);
  cos_y = svsel_f64(half, svdup_n_f64(0.0), cos_y);
  return svcreate2_f64(sin_y, cos_y);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_tanpi_f32(
    svfloat32_t x, svbool_t pg) {
  const auto pt = svptrue_b32();
  const auto n = svrintn_f32_x(pg, x);
  const auto xr = svsub_f32_x(pt, x, n);
  const auto ar = svabs_f32_x(pg, xr);
  const auto flip = svcmpgt_n_f32(pg, ar, 0.25f);
  const auto r = svsel_f32(flip, svsubr_n_f32_x(pt, ar, 0.5f), ar);
  const auto r2 = svmul_f32_x(pt, r, r);
  svfloat32_t p;
  if constexpr (Tier == Accuracy::Strict) {
    p = svdup_n_f32(0x1.a52e08p16f);
    p = svmla_f32_x(pg, svdup_n_f32(0x1.e85558p11f), r2, p);
    p = svmla_f32_x(pg, svdup_n_f32(0x1.69e2c4p11f), r2, p);
    p = svmla_f32_x(pg, svdup_n_f32(0x1.42e9d4p9f), r2, p);
  } else if constexpr (Tier == Accuracy::Fast) {
    p = svdup_n_f32(0x1.69e2c4p11f);
  } else {
    p = svdup_n_f32(0x1.461c72p7f);
  }
  if constexpr (Tier == Accuracy::Fast)
    p = svmla_f32_x(pg, svdup_n_f32(0x1.42e9d4p9f), r2, p);
  if constexpr (Tier != Accuracy::Estimate) {
    p = svmla_f32_x(pg, svdup_n_f32(0x1.461c72p7f), r2, p);
  }
  p = svmla_f32_x(pg, svdup_n_f32(0x1.466b8p5f), r2, p);
  p = svmla_f32_x(pg, svdup_n_f32(0x1.4abbcep3f), r2, p);
  p = svmla_f32_x(pg, svdup_n_f32(0x1.921fb4p1f), r2, p);
  const auto poly = svmul_f32_x(pt, r, p);
  auto y = svsel_f32(flip, svdivr_n_f32_x(pg, poly, 1.0f), poly);
  const auto sign = sveor_u32_x(
      pt, svreinterpret_u32_f32(xr), svreinterpret_u32_f32(ar));
  y = svreinterpret_f32_u32(
      svorr_u32_x(pt, svreinterpret_u32_f32(y), sign));

  const auto integer = svcmpeq_n_f32(pg, ar, 0.0f);
  const auto half = svcmpeq_n_f32(pg, ar, 0.5f);
  const auto convertible = svaclt_n_f32(pg, x, 0x1p31f);
  const auto odd = svlsl_n_u32_z(
      convertible, svreinterpret_u32_s32(svcvt_s32_f32_z(pg, n)), 31);
  const auto integer_sign = sveor_u32_x(
      pt, svand_n_u32_x(pt, svreinterpret_u32_f32(x), 0x80000000u), odd);
  y = svsel_f32(integer,
      svreinterpret_f32_u32(integer_sign), y);
  y = svsel_f32(half,
      svreinterpret_f32_u32(svorr_u32_x(
          pt, svand_n_u32_x(pt, sign, 0x80000000u),
          svdup_n_u32(0x7f800000u))), y);
  return y;
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_tanpi_f64(
    svfloat64_t x, svbool_t pg) {
  const auto pt = svptrue_b64();
  const auto n = svrintn_f64_x(pg, x);
  const auto xr = svsub_f64_x(pt, x, n);
  const auto ar = svabs_f64_x(pg, xr);
  const auto flip = svcmpgt_n_f64(pg, ar, 0.25);
  const auto r = svsel_f64(flip, svsubr_n_f64_x(pt, ar, 0.5), ar);
  const auto r2 = svmul_f64_x(pt, r, r);
  svfloat64_t p;
  if constexpr (Tier != Accuracy::Estimate) {
    p = svdup_n_f64(0x1.1b76de7681424p32);
    p = svmla_f64_x(pg, svdup_n_f64(-0x1.a854d53ab6874p29), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.5d4e912bb8456p27), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(-0x1.89333f6acd922p19), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.927896baee627p21), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.3a6d958cdefaep19), r2, p);
  } else {
    p = svdup_n_f64(0x1.45f4234b330cap13);
  }
  if constexpr (Tier != Accuracy::Estimate) {
    p = svmla_f64_x(pg, svdup_n_f64(0x1.47283fc5eea69p17), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.45dca11be79ebp15), r2, p);
    p = svmla_f64_x(pg, svdup_n_f64(0x1.45f4234b330cap13), r2, p);
  }
  p = svmla_f64_x(pg, svdup_n_f64(0x1.45f3265994f85p11), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.45f4730dbca5cp9), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.45fff9b426f5ep7), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.466bc6775b0f9p5), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.4abbce625be52p3), r2, p);
  p = svmla_f64_x(pg, svdup_n_f64(0x1.921fb54442d18p1), r2, p);
  const auto poly = svmul_f64_x(pt, r, p);
  auto y = svsel_f64(flip, svdivr_n_f64_x(pg, poly, 1.0), poly);
  const auto sign = sveor_u64_x(
      pt, svreinterpret_u64_f64(xr), svreinterpret_u64_f64(ar));
  y = svreinterpret_f64_u64(
      svorr_u64_x(pt, svreinterpret_u64_f64(y), sign));

  const auto integer = svcmpeq_n_f64(pg, ar, 0.0);
  const auto half = svcmpeq_n_f64(pg, ar, 0.5);
  const auto convertible = svaclt_n_f64(pg, x, 0x1p63);
  const auto odd = svlsl_n_u64_z(
      convertible, svreinterpret_u64_s64(svcvt_s64_f64_z(pg, n)), 63);
  const auto integer_sign = sveor_u64_x(
      pt, svand_n_u64_x(
              pt, svreinterpret_u64_f64(x), 0x8000000000000000ull),
      odd);
  y = svsel_f64(integer,
      svreinterpret_f64_u64(integer_sign), y);
  y = svsel_f64(half,
      svreinterpret_f64_u64(svorr_u64_x(
          pt, svand_n_u64_x(pt, sign, 0x8000000000000000ull),
          svdup_n_u64(0x7ff0000000000000ull))), y);
  return y;
}

template <TrigKind Kind, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_pi_trig_f32(
    svfloat32_t x, svbool_t pg) {
  if constexpr (Kind == TrigKind::Tan) return sve_tanpi_f32<Tier>(x, pg);
  else {
    const auto sc = sve_pi_sincos_f32<Tier>(x, pg);
    if constexpr (Kind == TrigKind::Sin) return svget2_f32(sc, 0);
    else return svget2_f32(sc, 1);
  }
}

template <TrigKind Kind, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_pi_trig_f64(
    svfloat64_t x, svbool_t pg) {
  if constexpr (Kind == TrigKind::Tan) return sve_tanpi_f64<Tier>(x, pg);
  else {
    const auto sc = sve_pi_sincos_f64<Tier>(x, pg);
    if constexpr (Kind == TrigKind::Sin) return svget2_f64(sc, 0);
    else return svget2_f64(sc, 1);
  }
}

/* **************************************************************************** */
//    Narrow widening routes and dispatch                                      //
/* **************************************************************************** */

template <typename LowF32, typename HighF32>
VECOPS_ALWAYS_INLINE svfloat16_t sve_trig_f16_from_f32_pair(
    LowF32 low, HighF32 high) {
  const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), low);
#if defined(__ARM_FEATURE_SVE2)
  return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), high);
#else
  const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), high);
  return svtrn1_f16(packed_low, packed_high);
#endif
}

template <typename F>
VECOPS_ALWAYS_INLINE svfloat16_t sve_trig_f16_via_f32(
    svfloat16_t raw, F&& operation) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
  return sve_trig_f16_from_f32_pair(operation(low), operation(high));
}

template <typename F>
VECOPS_ALWAYS_INLINE svfloat16x2_t sve_sincos_f16_via_f32(
    svfloat16_t raw, F&& operation) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
  const auto low_sc = operation(low);
  const auto high_sc = operation(high);
  return svcreate2_f16(
      sve_trig_f16_from_f32_pair(
          svget2_f32(low_sc, 0), svget2_f32(high_sc, 0)),
      sve_trig_f16_from_f32_pair(
          svget2_f32(low_sc, 1), svget2_f32(high_sc, 1)));
}

/* **************************************************************************** */
//    Native f16 Fast/Estimate kernels                                          //
/* **************************************************************************** */

template <TrigUnit Unit>
VECOPS_ALWAYS_INLINE float16_t sve_trig_f16_hot_limit() {
  if constexpr (Unit == TrigUnit::Radians) return float16_t(32.0f);
  else return float16_t(1024.0f);
}

template <TrigUnit Unit>
VECOPS_ALWAYS_INLINE svfloat16x2_t sve_trig_reduce_f16(svfloat16_t x) {
  const auto pt = svptrue_b16();
  constexpr float shift = 0x1.8p+10f;
  svfloat16_t q;
  svfloat16_t r;
  if constexpr (Unit == TrigUnit::Radians) {
    q = svmla_n_f16_x(
        pt, svdup_n_f16(float16_t(shift)), x,
        float16_t(0.63661977236758134308f));
    const auto n = svsub_n_f16_x(pt, q, float16_t(shift));
    r = svmls_n_f16_x(pt, x, n, float16_t(1.5703125f));
    r = svmls_n_f16_x(
        pt, r, n, float16_t(0.00048351287841796875f));
  } else {
    const auto n = svrintn_f16_x(
        pt, svmul_n_f16_x(pt, x, float16_t(2.0f)));
    // FTS* consumes the quadrant in the low bits. Unlike the floating shift
    // trick, an explicit integer conversion remains valid when 1536+n crosses
    // an fp16 binade (x >= 256), extending the native pi path to 1024.
    q = svreinterpret_f16_u16(svreinterpret_u16_s16(
        svcvt_s16_f16_x(pt, n)));
    r = svmls_n_f16_x(pt, x, n, float16_t(0.5f));
    r = svmul_n_f16_x(pt, r, float16_t(3.14159265358979323846f));
  }
  return svcreate2_f16(r, q);
}

template <Accuracy Tier>
  requires (Tier != Accuracy::Strict)
VECOPS_ALWAYS_INLINE svfloat16_t sve_trig_component_f16(
    svfloat16_t r, svuint16_t quadrant) {
  const auto r2 = svtsmul_f16(r, quadrant);
  auto y = svdup_n_f16(float16_t(0.0f));
  if constexpr (Tier == Accuracy::Fast) y = svtmad_f16(y, r2, 3);
  y = svtmad_f16(y, r2, 2);
  y = svtmad_f16(y, r2, 1);
  y = svtmad_f16(y, r2, 0);
  return svmul_f16_x(
      svptrue_b16(), svtssel_f16(r, quadrant), y);
}

template <TrigUnit Unit, Accuracy Tier>
  requires (Tier != Accuracy::Strict)
VECOPS_ALWAYS_INLINE svfloat16x2_t sve_sincos_f16_native_hot(
    svfloat16_t x) {
  const auto pt = svptrue_b16();
  const auto reduced = sve_trig_reduce_f16<Unit>(x);
  const auto r = svget2_f16(reduced, 0);
  const auto sin_q = svreinterpret_u16_f16(svget2_f16(reduced, 1));
  const auto cos_q = svadd_n_u16_x(pt, sin_q, 1);
  auto sin_x = sve_trig_component_f16<Tier>(r, sin_q);
  auto cos_x = sve_trig_component_f16<Tier>(r, cos_q);
  if constexpr (Unit == TrigUnit::Pi) {
    const auto fraction = svabs_f16_x(
        pt, svsub_f16_x(pt, x, svrintn_f16_x(pt, x)));
    const auto integer =
        svcmpeq_n_f16(pt, fraction, float16_t(0.0f));
    const auto half = svcmpeq_n_f16(pt, fraction, float16_t(0.5f));
    const auto signed_zero = svreinterpret_f16_u16(svand_n_u16_x(
        pt, svreinterpret_u16_f16(x), 0x8000u));
    sin_x = svsel_f16(integer, signed_zero, sin_x);
    cos_x = svsel_f16(half, svdup_n_f16(float16_t(0.0f)), cos_x);
  }
  return svcreate2_f16(sin_x, cos_x);
}

template <TrigUnit Unit>
VECOPS_ALWAYS_INLINE svbool_t sve_trig_f16_cold_mask(
    svbool_t pg, svfloat16_t x) {
  auto cold = svacge_n_f16(pg, x, sve_trig_f16_hot_limit<Unit>());
  cold = svorr_b_z(pg, cold, svcmpne_f16(pg, x, x));
  return cold;
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier, typename F>
  requires (Tier != Accuracy::Strict)
VECOPS_ALWAYS_INLINE svfloat16_t sve_trig_f16_native(
    svfloat16_t x, svbool_t pg, F&& f32_operation) {
  const auto sc = sve_sincos_f16_native_hot<Unit, Tier>(x);
  const auto sin_x = svget2_f16(sc, 0);
  const auto cos_x = svget2_f16(sc, 1);
  auto y = [&] {
    if constexpr (Kind == TrigKind::Sin) return sin_x;
    else if constexpr (Kind == TrigKind::Cos) return cos_x;
    else return svdiv_f16_x(pg, sin_x, cos_x);
  }();
  if constexpr (Kind != TrigKind::Cos)
    y = svsel_f16(svcmpeq_n_f16(pg, x, float16_t(0.0f)), x, y);
  auto cold = sve_trig_f16_cold_mask<Unit>(pg, x);
  if constexpr (Kind == TrigKind::Tan) {
    auto near_pole = svaclt_n_f16(pg, cos_x, float16_t(0.125f));
    if constexpr (Unit == TrigUnit::Pi) {
      const auto fraction = svabs_f16_x(
          pg, svsub_f16_x(pg, x, svrintn_f16_x(pg, x)));
      const auto half =
          svcmpeq_n_f16(pg, fraction, float16_t(0.5f));
      near_pole = svand_b_z(pg, near_pole, svnot_b_z(pg, half));
    }
    cold = svorr_b_z(pg, cold, near_pole);
  }
  if (svptest_any(pg, cold)) {
    const auto widened = sve_trig_f16_via_f32(
        x, std::forward<F>(f32_operation));
    y = svsel_f16(cold, widened, y);
  }
  return y;
}

template <TrigUnit Unit, Accuracy Tier, typename F>
  requires (Tier != Accuracy::Strict)
VECOPS_ALWAYS_INLINE svfloat16x2_t sve_sincos_f16_native(
    svfloat16_t x, svbool_t pg, F&& f32_operation) {
  auto sc = sve_sincos_f16_native_hot<Unit, Tier>(x);
  auto sin_x = svget2_f16(sc, 0);
  auto cos_x = svget2_f16(sc, 1);
  sin_x = svsel_f16(
      svcmpeq_n_f16(pg, x, float16_t(0.0f)), x, sin_x);
  const auto cold = sve_trig_f16_cold_mask<Unit>(pg, x);
  if (svptest_any(pg, cold)) {
    const auto widened = sve_sincos_f16_via_f32(
        x, std::forward<F>(f32_operation));
    sin_x = svsel_f16(cold, svget2_f16(widened, 0), sin_x);
    cos_x = svsel_f16(cold, svget2_f16(widened, 1), cos_x);
  }
  return svcreate2_f16(sin_x, cos_x);
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_trig_dispatch(
    Tag, NativeWordVec<Tag> value, svbool_t pg) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  const auto f32_op = [](svfloat32_t x) VECOPS_INLINE_LAMBDA {
    if constexpr (Unit == TrigUnit::Radians)
      return sve_radian_trig_f32<Kind, Tier>(x, svptrue_b32());
    else return sve_pi_trig_f32<Kind, Tier>(x, svptrue_b32());
  };
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (Unit == TrigUnit::Radians)
      return sve_basic_wrap_word<Tag>(sve_radian_trig_f32<Kind, Tier>(raw, pg));
    else return sve_basic_wrap_word<Tag>(sve_pi_trig_f32<Kind, Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (Unit == TrigUnit::Radians)
      return sve_basic_wrap_word<Tag>(sve_radian_trig_f64<Kind, Tier>(raw, pg));
    else return sve_basic_wrap_word<Tag>(sve_pi_trig_f64<Kind, Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (Tier == Accuracy::Strict)
      return sve_basic_wrap_word<Tag>(sve_trig_f16_via_f32(raw, f32_op));
    else
      return sve_basic_wrap_word<Tag>(
          sve_trig_f16_native<Kind, Unit, Tier>(raw, pg, f32_op));
  } else {
    return sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        f32_op(sve_bf16_to_f32_lo(raw)), f32_op(sve_bf16_to_f32_hi(raw))));
  }
}

template <TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void sve_sincos_dispatch(
    Tag, NativeWordVec<Tag> value, NativeWordVec<Tag>& sin_out,
    NativeWordVec<Tag>& cos_out) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  const auto f32_op = [](svfloat32_t x) VECOPS_INLINE_LAMBDA {
    if constexpr (Unit == TrigUnit::Radians)
      return sve_radian_sincos_f32<Tier>(x, svptrue_b32());
    else return sve_pi_sincos_f32<Tier>(x, svptrue_b32());
  };
  if constexpr (std::same_as<T, float32_t>) {
    const auto sc = f32_op(raw);
    sin_out = sve_basic_wrap_word<Tag>(svget2_f32(sc, 0));
    cos_out = sve_basic_wrap_word<Tag>(svget2_f32(sc, 1));
  } else if constexpr (std::same_as<T, float64_t>) {
    const auto sc = [&] {
      if constexpr (Unit == TrigUnit::Radians)
        return sve_radian_sincos_f64<Tier>(raw, svptrue_b64());
      else return sve_pi_sincos_f64<Tier>(raw, svptrue_b64());
    }();
    sin_out = sve_basic_wrap_word<Tag>(svget2_f64(sc, 0));
    cos_out = sve_basic_wrap_word<Tag>(svget2_f64(sc, 1));
  } else if constexpr (std::same_as<T, float16_t>) {
    const auto sc = [&] {
      if constexpr (Tier == Accuracy::Strict)
        return sve_sincos_f16_via_f32(raw, f32_op);
      else
        return sve_sincos_f16_native<Unit, Tier>(
            raw, svptrue_b16(), f32_op);
    }();
    sin_out = sve_basic_wrap_word<Tag>(svget2_f16(sc, 0));
    cos_out = sve_basic_wrap_word<Tag>(svget2_f16(sc, 1));
  } else {
    const auto low_sc = f32_op(sve_bf16_to_f32_lo(raw));
    const auto high_sc = f32_op(sve_bf16_to_f32_hi(raw));
    sin_out = sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        svget2_f32(low_sc, 0), svget2_f32(high_sc, 0)));
    cos_out = sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        svget2_f32(low_sc, 1), svget2_f32(high_sc, 1)));
  }
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier>
struct SVETrigWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, Tier>, Tag tag, NativeWordVec<Tag> value) {
    return sve_trig_dispatch<Kind, Unit, Tier>(
        tag, value, sve_full_predicate<ElementOf<Tag>>());
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, Tier> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    if constexpr (sizeof(ElementOf<Tag>) >= 4) {
      const auto computed = sve_trig_dispatch<Kind, Unit, Tier>(
          tag, value, sve_basic_raw_word(mask));
      return blend(tag, inactive, mask, computed);
    } else {
      return masked_unary_word(
          tag, mask, inactive, value, [&](NativeWordVec<Tag> safe) {
            return call<Index>(op, tag, safe);
          });
    }
  }
};

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier>
struct NativeWordImpl<SVEBackend, TrigOp<Kind, Unit, Tier>>
    : SVETrigWordImpl<Kind, Unit, Tier> {};

template <TrigUnit Unit, Accuracy Tier>
struct NativeWordImpl<SVEBackend, SinCosOp<Unit, Tier>> {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE void call(
      SinCosOp<Unit, Tier>, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag>& sin_out, NativeWordVec<Tag>& cos_out) {
    sve_sincos_dispatch<Unit, Tier>(tag, value, sin_out, cos_out);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_TRIG_H
