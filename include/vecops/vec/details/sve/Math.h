#ifndef VECOPS_VEC_DETAILS_SVE_MATH_H
#define VECOPS_VEC_DETAILS_SVE_MATH_H

/**
 * @file Math.h
 * @brief SVE backend implementations for math operations (exp family).
 */

#include <arm_sve.h>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//    f32 accurate exp — sve_exp_f32_accurate                                //
/* **************************************************************************** */

struct alignas(16) SVEExpF32Constants {
  float lanes[8];
};

inline constexpr SVEExpF32Constants kSVEExpF32Constants{{
    -87.3365447505531f, 88.72283905206835f, 196735.0f,
    1.4426950408889634074f,
    -0.693145751953125f, -1.428606765330187045e-6f, 88.0f, 0.5f}};

/**
 * High-accuracy f32 exp via range reduction and 5-term minimax polynomial.
 * Used by Strict and Fast tiers; Estimate uses the hardware FEXPA path.
 */
VECOPS_ALWAYS_INLINE svfloat32_t sve_exp_f32_accurate(svfloat32_t x) {
  const auto pg = svptrue_b32();
  constexpr float overflow = 88.72283905206835f;
  constexpr float zero_limit = -103.972084045410f;
  auto xc = svminnm_n_f32_x(pg, x, overflow);
  xc = svmaxnm_n_f32_x(pg, xc, zero_limit);
  const auto qf = svrinta_f32_x(
      pg, svmul_n_f32_x(pg, xc, 1.4426950408889634074f));
  const auto qi = svcvt_s32_f32_x(pg, qf);
  auto r = svmla_n_f32_x(pg, xc, qf, -0.693145751953125f);
  r = svmla_n_f32_x(pg, r, qf, -1.428606765330187045e-6f);
  const auto r2 = svmul_f32_x(pg, r, r);
  const auto p0 = svmla_n_f32_x(
      pg, svdup_n_f32(0.5f), r, 0.166666671633720397949219f);
  const auto p1 = svmla_n_f32_x(
      pg, svdup_n_f32(0.0416664853692054748535156f), r,
      0.00833336077630519866943359f);
  const auto p2 = svmla_n_f32_x(
      pg, svdup_n_f32(0.00139304355252534151077271f), r,
      0.000198527617612853646278381f);
  const auto q = svmla_f32_x(
      pg, p0, svmla_f32_x(pg, p1, p2, r2), r2);
  const auto poly = svadd_n_f32_x(
      pg, svmla_f32_x(pg, r, q, r2), 1.0f);
  const auto qh = svasr_n_s32_x(pg, qi, 1);
  const auto qr = svsub_s32_x(pg, qi, qh);
  const auto sh = svreinterpret_f32_s32(
      svlsl_n_s32_x(pg, svadd_n_s32_x(pg, qh, 127), 23));
  const auto sr = svreinterpret_f32_s32(
      svlsl_n_s32_x(pg, svadd_n_s32_x(pg, qr, 127), 23));
  auto y = svmul_f32_x(pg, svmul_f32_x(pg, poly, sh), sr);
  y = svsel_f32(
      svcmpgt_n_f32(pg, x, overflow),
      svdup_n_f32(std::numeric_limits<float>::infinity()), y);
  y = svsel_f32(
      svcmplt_n_f32(pg, x, zero_limit), svdup_n_f32(0.0f), y);
  return svsel_f32(svcmpne_f32(pg, x, x), svadd_f32_x(pg, x, x), y);
}

/* **************************************************************************** */
//    f64 upper tail / tiered exp — f64, f32, f16                             //
/* **************************************************************************** */

VECOPS_NOINLINE inline svfloat64_t sve_exp_f64_upper_tail(svfloat64_t x) {
  const auto pg = svptrue_b64();
  const auto qf = svrinta_f64_x(
      pg, svmul_n_f64_x(pg, x, 1.4426950408889634074));
  const auto qi = svcvt_s64_f64_x(pg, qf);
  auto r = svmla_n_f64_x(
      pg, x, qf, -0.6931471805596629565116018);
  r = svmla_n_f64_x(
      pg, r, qf, -0.28235290563031577122588448175e-12);
  const auto r2 = svmul_f64_x(pg, r, r);
  constexpr double c[] = {
      0.5, 0.166666666666666851703837,
      0.0416666666666665047591422, 0.00833333333331652721664984,
      0.00138888888889774492207962, 0.000198412698960509205564975,
      2.4801587159235472998791e-5, 2.75572362911928827629423e-6,
      2.75573911234900471893338e-7, 2.51112930892876518610661e-8,
      2.08860621107283687536341e-9};
  const auto p0 = svmla_n_f64_x(pg, svdup_n_f64(c[0]), r, c[1]);
  const auto p1 = svmla_n_f64_x(pg, svdup_n_f64(c[2]), r, c[3]);
  const auto p2 = svmla_n_f64_x(pg, svdup_n_f64(c[4]), r, c[5]);
  const auto p3 = svmla_n_f64_x(pg, svdup_n_f64(c[6]), r, c[7]);
  const auto p4 = svmla_n_f64_x(pg, svdup_n_f64(c[8]), r, c[9]);
  const auto p5 = svdup_n_f64(c[10]);
  const auto r4 = svmul_f64_x(pg, r2, r2);
  const auto r8 = svmul_f64_x(pg, r4, r4);
  const auto q0 = svmla_f64_x(pg, p0, p1, r2);
  const auto q1 = svmla_f64_x(pg, p2, p3, r2);
  const auto q2 = svmla_f64_x(pg, p4, p5, r2);
  const auto q = svmla_f64_x(
      pg, svmla_f64_x(pg, q0, q1, r4), q2, r8);
  const auto poly = svadd_n_f64_x(
      pg, svmla_f64_x(pg, r, q, r2), 1.0);
  const auto qh = svasr_n_s64_x(pg, qi, 1);
  const auto qr = svsub_s64_x(pg, qi, qh);
  const auto sh = svreinterpret_f64_s64(
      svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qh, 1023), 52));
  const auto sr = svreinterpret_f64_s64(
      svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qr, 1023), 52));
  return svmul_f64_x(pg, svmul_f64_x(pg, poly, sh), sr);
}

template <ExpTier Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat32_t sve_exp_f32(svfloat32_t x) {
  constexpr bool strict = Tier == ExpTier::Strict;
  constexpr bool estimate = Tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
  const auto pg = svptrue_b32();
  const auto c0 = svld1rq_f32(pg, kSVEExpF32Constants.lanes);
  const auto c1 = svld1rq_f32(pg, kSVEExpF32Constants.lanes + 4);
  constexpr float zero_limit = -103.972084045410f;
  constexpr float normal_limit = -87.3365447505531f;
  constexpr float overflow = 88.72283905206835f;
  const auto subnormal = svcmplt_n_f32(pg, x, normal_limit);
  auto xc = x;
  if constexpr (gradual)
    xc = svsel_f32(
        subnormal, svadd_n_f32_x(pg, xc, 44.3614195558365f), xc);
  xc = svmaxnm_f32_x(pg, xc, svdup_lane_f32(c0, 0));
  if constexpr (!NegativeOnly)
    xc = svminnm_f32_x(pg, xc, svdup_lane_f32(c0, 1));
  const auto shift = svdup_lane_f32(c0, 2);
  const auto z = svmla_lane_f32(shift, xc, c0, 3);
  const auto n = svsub_f32_x(pg, z, shift);
  const auto scale = svexpa_f32(svreinterpret_u32_f32(z));
  auto r = svmla_lane_f32(xc, n, c1, 0);
  r = svmla_lane_f32(r, n, c1, 1);
  svfloat32_t y;
  if constexpr (estimate) {
    y = svmla_f32_x(pg, scale, scale, r);
  } else {
    const auto p = svmla_n_f32_x(
        pg, r, svmul_f32_x(pg, r, r), 0.5f);
    y = svmla_f32_x(pg, scale, scale, p);
  }
  if constexpr (gradual)
    y = svsel_f32(
        subnormal, svmul_n_f32_x(pg, y, 0x1.fffffcp-65f), y);
  if constexpr (!NegativeOnly) {
    const auto upper = svcmpgt_f32(pg, x, svdup_lane_f32(c1, 2));
    if (svptest_any(pg, upper))
      y = svsel_f32(upper, sve_exp_f32_accurate(x), y);
    y = svsel_f32(
        svcmpgt_n_f32(pg, x, overflow),
        svdup_n_f32(std::numeric_limits<float>::infinity()), y);
  }
  y = svsel_f32(
      svcmplt_n_f32(pg, x, gradual ? zero_limit : normal_limit),
      svdup_n_f32(0.0f), y);
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!NegativeOnly)
    y = svsel_f32(svcmpne_f32(pg, x, x), svadd_f32_x(pg, x, x), y);
#endif
  return y;
}

template <ExpTier Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat64_t sve_exp_f64(svfloat64_t x) {
  constexpr bool strict = Tier == ExpTier::Strict;
  constexpr bool estimate = Tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
  const auto pg = svptrue_b64();
  constexpr double overflow = 709.782712893384;
  constexpr double zero_limit = -745.1332191019411;
  constexpr double normal_limit = -708.3964185322641;
  const auto subnormal = svcmplt_n_f64(pg, x, normal_limit);
  auto xc = x;
  if constexpr (gradual)
    xc = svsel_f64(
        subnormal, svadd_n_f64_x(pg, xc, 354.891356446692), xc);
  xc = svmaxnm_n_f64_x(pg, xc, normal_limit);
  if constexpr (!NegativeOnly) xc = svminnm_n_f64_x(pg, xc, overflow);
  svfloat64_t y;
  if constexpr (estimate) {
    constexpr double inv_step = 92.332482616893656768;
    const auto nf = svrinta_f64_x(pg, svmul_n_f64_x(pg, xc, inv_step));
    const auto code = svcvt_u64_f64_x(
        pg, svadd_n_f64_x(pg, nf, 65472.0));
    const auto scale = svexpa_f64(code);
    auto r = svmla_n_f64_x(
        pg, xc, nf, -0.010830424696244733695493778125);
    r = svmla_n_f64_x(pg, r, nf, -4.411764150473684e-15);
    y = svmla_f64_x(pg, scale, scale, r);
  } else {
    constexpr double shift = 0x1.800000000ffc0p+46;
    const auto upper = svcmpgt_n_f64(pg, xc, 709.0);
    const auto xa = svsel_f64(
        upper, svsub_n_f64_x(pg, xc, 0.6931471805599453094), xc);
    const auto z = svmla_n_f64_x(
        pg, svdup_n_f64(shift), xa, 1.4426950408889634074);
    const auto n = svsub_n_f64_x(pg, z, shift);
    const auto scale = svexpa_f64(svreinterpret_u64_f64(z));
    auto r = svmla_n_f64_x(
        pg, xa, n, -0.6931471805596629565116018);
    r = svmla_n_f64_x(
        pg, r, n, -0.28235290563031577122588448175e-12);
    const auto r2 = svmul_f64_x(pg, r, r);
    const auto p01 = svmla_n_f64_x(pg, svdup_n_f64(0.5), r, 1.0 / 6.0);
    const auto p23 = svmla_n_f64_x(pg, svdup_n_f64(1.0 / 24.0), r, 1.0 / 120.0);
    const auto q = svmla_f64_x(pg, p01, r2, p23);
    y = svmla_f64_x(pg, scale, scale, svmla_f64_x(pg, r, r2, q));
    if (svptest_any(pg, upper))
      y = svsel_f64(upper, sve_exp_f64_upper_tail(xc), y);
  }
  if constexpr (gradual)
    y = svsel_f64(
        subnormal,
        svmul_n_f64_x(pg, y, 0x1.0000000000035p-512), y);
  if constexpr (!NegativeOnly)
    y = svsel_f64(
        svcmpgt_n_f64(pg, x, overflow),
        svdup_n_f64(std::numeric_limits<double>::infinity()), y);
  y = svsel_f64(
      svcmplt_n_f64(pg, x, gradual ? zero_limit : normal_limit),
      svdup_n_f64(0.0), y);
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!NegativeOnly)
    y = svsel_f64(svcmpne_f64(pg, x, x), svadd_f64_x(pg, x, x), y);
#endif
  return y;
}

template <ExpTier Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16(svfloat16_t x) {
  constexpr bool strict = Tier == ExpTier::Strict;
  constexpr bool estimate = Tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
  const auto pg = svptrue_b16();
  const float16_t overflow = float16_t(11.0859375);
  const float16_t zero_limit = float16_t(-17.328679513999);
  const float16_t normal_limit = float16_t(-9.704060527839);
  const auto subnormal = svcmplt_n_f16(pg, x, normal_limit);
  auto xc = x;
  if constexpr (gradual)
    xc = svsel_f16(
        subnormal, svadd_n_f16_x(pg, xc, float16_t(5.545177444480)), xc);
  xc = svmaxnm_n_f16_x(pg, xc, normal_limit);
  if constexpr (!NegativeOnly) xc = svminnm_n_f16_x(pg, xc, overflow);
  const auto nf = svrinta_f16_x(
      pg, svmul_n_f16_x(pg, xc, float16_t(46.16624130844683f)));
  const auto code = svcvt_u16_f16_x(
      pg, svadd_n_f16_x(pg, nf, float16_t(480)));
  const auto scale = svexpa_f16(code);
  svfloat16_t y;
  if constexpr (estimate) {
    const auto r = svmla_n_f16_x(
        pg, xc, nf, float16_t(-0.02166084939249829));
    y = svmla_f16_x(pg, scale, scale, r);
  } else {
    auto r = svmla_n_f16_x(
        pg, xc, nf, float16_t(-0.02166748046875));
    r = svmla_n_f16_x(
        pg, r, nf, float16_t(6.631076251709805e-6));
    y = svmla_f16_x(pg, scale, scale, r);
  }
  if constexpr (gradual)
    y = svsel_f16(
        subnormal, svmul_n_f16_x(pg, y, float16_t(0.00390625)), y);
  y = svsel_f16(
      svcmpeq_n_f16(pg, x, overflow),
      svdup_n_f16(estimate
          ? std::numeric_limits<float16_t>::max()
          : float16_t(65248.0)), y);
  if constexpr (!NegativeOnly)
    y = svsel_f16(
        svcmpgt_n_f16(pg, x, overflow),
        svdup_n_f16(std::numeric_limits<float16_t>::infinity()), y);
  y = svsel_f16(
      svcmplt_n_f16(pg, x, gradual ? zero_limit : normal_limit),
      svdup_n_f16(float16_t(0)), y);
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!NegativeOnly)
    y = svsel_f16(svcmpne_f16(pg, x, x), svadd_f16_x(pg, x, x), y);
#endif
  return y;
}

/* **************************************************************************** */
//    Exp dispatch and SVEExpWordImpl                                         //
/* **************************************************************************** */

template <ExpTier Tier, bool NegativeOnly, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_exp_dispatch(
    Tag, NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(sve_exp_f32<Tier, NegativeOnly>(raw));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(sve_exp_f64<Tier, NegativeOnly>(raw));
  } else if constexpr (std::same_as<T, float16_t>) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
    if constexpr (Tier == ExpTier::Strict) {
      const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
      const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
      const auto odd = svuzp2_u16(
          svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
      const auto high = svcvt_f32_f16_x(
          svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
      const auto result_low = sve_exp_f32<Tier, NegativeOnly>(low);
      const auto result_high = sve_exp_f32<Tier, NegativeOnly>(high);
      const auto packed_low =
          svcvt_f16_f32_z(svptrue_b32(), result_low);
#if defined(__ARM_FEATURE_SVE2)
      return sve_basic_wrap_word<Tag>(svcvtnt_f16_f32_m(
          packed_low, svptrue_b32(), result_high));
#else
      const auto packed_high =
          svcvt_f16_f32_z(svptrue_b32(), result_high);
      return sve_basic_wrap_word<Tag>(svtrn1_f16(
          packed_low, packed_high));
#endif
    }
#endif
    return sve_basic_wrap_word<Tag>(sve_exp_f16<Tier, NegativeOnly>(raw));
  } else {
    constexpr ExpTier compute_tier =
#ifdef VECOPS_PRESERVE_SUBNORMALS
        Tier == ExpTier::Strict ? ExpTier::Strict : ExpTier::Estimate;
#else
        ExpTier::Estimate;
#endif
    const auto low = sve_bfloat16_to_float32_low(raw);
    const auto high = sve_bfloat16_to_float32_high(raw);
    return sve_basic_wrap_word<Tag>(sve_float32_pair_to_bfloat16(
        sve_exp_f32<compute_tier, NegativeOnly>(low),
        sve_exp_f32<compute_tier, NegativeOnly>(high)));
  }
}

template <typename Op>
struct SVEExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    constexpr ExpTier tier = std::same_as<Op, ExpOp> ||
            std::same_as<Op, ExpNegOp>
        ? ExpTier::Strict
        : (std::same_as<Op, ExpFastOp> || std::same_as<Op, ExpNegFastOp>
            ? ExpTier::Fast : ExpTier::Estimate);
    constexpr bool negative_only =
        std::same_as<Op, ExpNegOp> ||
        std::same_as<Op, ExpNegFastOp> ||
        std::same_as<Op, ExpNegEstOp>;
    return sve_exp_dispatch<tier, negative_only>(tag, value);
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto zero = NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
        FillOp{}, tag, ElementOf<Tag>{});
    const auto safe = NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
        BlendOp{}, tag, zero, mask, value);
    const auto computed = call<Index>(op, tag, safe);
    return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

#define VECOPS_VEC_DEFINE_SVE_EXP(OpType)                              \
  template <>                                                          \
  struct NativeWordImpl<SVEBackend, OpType>                            \
      : SVEExpWordImpl<OpType> {}

VECOPS_VEC_DEFINE_SVE_EXP(ExpOp);
VECOPS_VEC_DEFINE_SVE_EXP(ExpFastOp);
VECOPS_VEC_DEFINE_SVE_EXP(ExpEstOp);
VECOPS_VEC_DEFINE_SVE_EXP(ExpNegOp);
VECOPS_VEC_DEFINE_SVE_EXP(ExpNegFastOp);
VECOPS_VEC_DEFINE_SVE_EXP(ExpNegEstOp);

#undef VECOPS_VEC_DEFINE_SVE_EXP

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_H
