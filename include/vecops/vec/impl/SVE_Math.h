// SIMD transcendental functions for Arm SVE.

#ifndef VECOPS_SVE_MATH_H
#define VECOPS_SVE_MATH_H

#include <arm_sve.h>
#include <cmath>
#include <limits>

#include "SVE_Arithmetic.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {
namespace math_details {

enum class ExpTier { Strict, Fast, Estimate };

struct alignas(16) ExpF32Constants {
  float lanes[8];
};

inline constexpr ExpF32Constants kExpF32Constants{{
    -87.3365447505531f, 88.72283905206835f, 196735.0f,
    1.4426950408889634074f,
    -0.693145751953125f, -1.428606765330187045e-6f, 88.0f, 0.5f}};

template <typename T>
VECOPS_VFUNC Vec<T> exp_f32_accurate(Vec<T> x) {
  const auto pg = details::ptrue<float32_t>();
  constexpr float overflow = 88.72283905206835f;
  constexpr float zero_limit = -103.972084045410f;
  auto xc = word::blend(x, word::cmpgt(x, word::fill(T{}, overflow)), word::fill(T{}, overflow));
  xc = word::blend(xc, word::cmplt(xc, word::fill(T{}, zero_limit)),
                   word::fill(T{}, zero_limit));
  const auto qf = svrinta_f32_x(pg, svmul_n_f32_x(pg, xc, 1.4426950408889634074f));
  const auto qi = svcvt_s32_f32_x(pg, qf);
  auto r = svmla_n_f32_x(pg, xc, qf, -0.693145751953125f);
  r = svmla_n_f32_x(pg, r, qf, -1.428606765330187045e-6f);
  const auto r2 = svmul_f32_x(pg, r, r);
  const auto p0 = svmla_n_f32_x(pg, svdup_n_f32(0.5f), r, 0.166666671633720397949219f);
  const auto p1 = svmla_n_f32_x(pg, svdup_n_f32(0.0416664853692054748535156f), r,
                                0.00833336077630519866943359f);
  const auto p2 = svmla_n_f32_x(pg, svdup_n_f32(0.00139304355252534151077271f), r,
                                0.000198527617612853646278381f);
  const auto q = svmla_f32_x(pg, p0, svmla_f32_x(pg, p1, p2, r2), r2);
  const auto poly = svadd_n_f32_x(pg, svmla_f32_x(pg, r, q, r2), 1.0f);
  const auto qh = svasr_n_s32_x(pg, qi, 1);
  const auto qr = svsub_s32_x(pg, qi, qh);
  const auto sh = svreinterpret_f32_s32(
      svlsl_n_s32_x(pg, svadd_n_s32_x(pg, qh, 127), 23));
  const auto sr = svreinterpret_f32_s32(
      svlsl_n_s32_x(pg, svadd_n_s32_x(pg, qr, 127), 23));
  auto y = svmul_f32_x(pg, svmul_f32_x(pg, poly, sh), sr);
  y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                  word::fill(T{}, std::numeric_limits<float32_t>::infinity()));
  y = word::blend(y, word::cmplt(x, word::fill(T{}, zero_limit)), word::fill(T{}, 0.0f));
  y = word::blend(y, word::cmpne(x, x), word::add(x, x));
  return y;
}

template <typename T>
VECOPS_NOINLINE Vec<T> exp_f64_upper_tail(Vec<T> x) {
  const auto pg = details::ptrue<float64_t>();
  const auto qf =
      svrinta_f64_x(pg, svmul_n_f64_x(pg, x, 1.4426950408889634074));
  const auto qi = svcvt_s64_f64_x(pg, qf);
  auto r = svmla_n_f64_x(pg, x, qf, -0.6931471805596629565116018);
  r = svmla_n_f64_x(
      pg, r, qf, -0.28235290563031577122588448175e-12);
  const auto r2 = svmul_f64_x(pg, r, r);
  constexpr double c[] = {
      0.5, 0.166666666666666851703837, 0.0416666666666665047591422,
      0.00833333333331652721664984, 0.00138888888889774492207962,
      0.000198412698960509205564975, 2.4801587159235472998791e-5,
      2.75572362911928827629423e-6, 2.75573911234900471893338e-7,
      2.51112930892876518610661e-8, 2.08860621107283687536341e-9};
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
  const auto poly_q =
      svmla_f64_x(pg, svmla_f64_x(pg, q0, q1, r4), q2, r8);
  const auto poly =
      svadd_n_f64_x(pg, svmla_f64_x(pg, r, poly_q, r2), 1.0);
  const auto qh = svasr_n_s64_x(pg, qi, 1);
  const auto qr = svsub_s64_x(pg, qi, qh);
  const auto sh = svreinterpret_f64_s64(
      svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qh, 1023), 52));
  const auto sr = svreinterpret_f64_s64(
      svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qr, 1023), 52));
  return svmul_f64_x(pg, svmul_f64_x(pg, poly, sh), sr);
}

template <ExpTier tier, bool negative_only, typename T>
VECOPS_VFUNC Vec<T> exp_sve(Vec<T> x) {
  using E = TypeOf<T>;
  constexpr bool strict = tier == ExpTier::Strict;
  constexpr bool estimate = tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
  const auto pg = details::ptrue<E>();

  const auto f32_pg = details::ptrue<float32_t>();
  const auto f32_c0 = svld1rq_f32(f32_pg, kExpF32Constants.lanes);
  const auto f32_c1 = svld1rq_f32(f32_pg, kExpF32Constants.lanes + 4);

  E overflow, zero_limit, normal_limit, sub_shift, sub_scale;
  if constexpr (std::is_same_v<E, float16_t>) {
    // Largest binary16 input whose exponential is still finite.
    overflow = E(11.0859375); zero_limit = E(-17.328679513999);
    normal_limit = E(-9.704060527839); sub_shift = E(5.545177444480);
    sub_scale = E(0.00390625);
  } else if constexpr (std::is_same_v<E, float32_t>) {
    overflow = E(88.72283905206835); zero_limit = E(-103.972084045410);
    normal_limit = E(-87.3365447505531); sub_shift = E(44.3614195558365);
    // Compensate for sub_shift being rounded to f32 rather than exactly 64*ln(2).
    sub_scale = E(0x1.fffffcp-65f);
  } else {
    overflow = E(709.782712893384); zero_limit = E(-745.1332191019411);
    normal_limit = E(-708.3964185322641); sub_shift = E(354.891356446692);
    // Compensate for sub_shift being rounded to f64 rather than exactly 512*ln(2).
    sub_scale = E(0x1.0000000000035p-512);
  }

  const auto subnormal = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (std::is_same_v<E, float32_t>)
      return svcmplt_f32(pg, x, svdup_lane_f32(f32_c0, 0));
    else
      return word::cmplt(x, word::fill(T{}, normal_limit));
  }();
  auto xc = x;
  if constexpr (gradual) {
    xc = word::blend(xc, subnormal, word::add(xc, word::fill(T{}, sub_shift)));
  }
  const E lower = normal_limit;
  if constexpr (std::is_same_v<E, float32_t>) {
    xc = svmaxnm_f32_x(pg, xc, svdup_lane_f32(f32_c0, 0));
    if constexpr (!negative_only)
      xc = svminnm_f32_x(pg, xc, svdup_lane_f32(f32_c0, 1));
  } else {
    xc = word::blend(xc, word::cmplt(xc, word::fill(T{}, lower)),
                     word::fill(T{}, lower));
    if constexpr (!negative_only) {
      xc = word::blend(xc, word::cmpgt(xc, word::fill(T{}, overflow)),
                       word::fill(T{}, overflow));
    }
  }

  Vec<T> y;
  if constexpr (std::is_same_v<E, float32_t>) {
    // The low six mantissa bits of z encode FEXPA's 64-entry table index.
    // This magic-bias form folds rounding and code generation into one FMA.
    const auto shift = svdup_lane_f32(f32_c0, 2);
    const auto z = svmla_lane_f32(shift, xc, f32_c0, 3);
    const auto n = svsub_f32_x(pg, z, shift);
    const auto scale = svexpa_f32(svreinterpret_u32_f32(z));
    auto r = svmla_lane_f32(xc, n, f32_c1, 0);
    r = svmla_lane_f32(r, n, f32_c1, 1);
    if constexpr (estimate) {
      y = svmla_f32_x(pg, scale, scale, r);
    } else {
      const auto p = svmla_lane_f32(r, svmul_f32_x(pg, r, r), f32_c1, 3);
      y = svmla_f32_x(pg, scale, scale, p);
    }
  } else if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (estimate) {
      constexpr double inv_step = 92.332482616893656768;
      const auto nf = svrinta_f64_x(pg, svmul_n_f64_x(pg, xc, inv_step));
      const auto code = svcvt_u64_f64_x(pg, svadd_n_f64_x(pg, nf, 65472.0));
      const auto scale = svexpa_f64(code);
      auto r = svmla_n_f64_x(pg, xc, nf, -0.010830424696244733695493778125);
      r = svmla_n_f64_x(pg, r, nf, -4.411764150473684e-15);
      y = svmla_f64_x(pg, scale, scale, r);
    } else {
      // The magic bias rounds x / ln(2) to a 1/64 grid. Its low mantissa
      // bits are already the FEXPA table code, avoiding float/int converts.
      constexpr double shift = 0x1.800000000ffc0p+46;
      const auto upper_tail = svcmpgt_n_f64(pg, xc, 709.0);
      const auto xa = svsel_f64(
          upper_tail, svsub_n_f64_x(pg, xc, 0.6931471805599453094), xc);
      const auto z =
          svmla_n_f64_x(pg, svdup_n_f64(shift), xa, 1.4426950408889634074);
      const auto n = svsub_n_f64_x(pg, z, shift);
      const auto scale = svexpa_f64(svreinterpret_u64_f64(z));
      auto r = svmla_n_f64_x(pg, xa, n, -0.6931471805596629565116018);
      r = svmla_n_f64_x(
          pg, r, n, -0.28235290563031577122588448175e-12);
      const auto r2 = svmul_f64_x(pg, r, r);
      const auto p01 =
          svmla_n_f64_x(pg, svdup_n_f64(0.5), r, 1.0 / 6.0);
      const auto p23 =
          svmla_n_f64_x(pg, svdup_n_f64(1.0 / 24.0), r, 1.0 / 120.0);
      const auto q = svmla_f64_x(pg, p01, r2, p23);
      const auto poly = svmla_f64_x(pg, r, r2, q);
      y = svmla_f64_x(pg, scale, scale, poly);
      if (svptest_any(pg, upper_tail)) {
        y = svsel_f64(upper_tail, exp_f64_upper_tail<T>(xc), y);
      }
    }
  } else {
    const E inv_step = E(46.16624130844683f);  // 32 / ln(2)
    const auto nf = svrinta_f16_x(pg, svmul_n_f16_x(pg, xc, inv_step));
    const auto code = svcvt_u16_f16_x(pg, svadd_n_f16_x(pg, nf, E(480)));
    const auto scale = svexpa_f16(code);
    if constexpr (estimate) {
      const auto r =
          svmla_n_f16_x(pg, xc, nf, E(-0.02166084939249829));
      y = svmla_f16_x(pg, scale, scale, r);
    } else {
      // Split ln(2)/32 so the range-reduction constant error is not magnified
      // by the (potentially large) integer grid index.
      auto r = svmla_n_f16_x(pg, xc, nf, E(-0.02166748046875));
      r = svmla_n_f16_x(pg, r, nf, E(6.631076251709805e-6));
      y = svmla_f16_x(pg, scale, scale, r);
    }
  }

  if constexpr (gradual) {
    y = word::blend(y, subnormal, word::mul(y, word::fill(T{}, sub_scale)));
  }
  if constexpr (std::is_same_v<E, float16_t>) {
    // FEXPA's top finite binary16 code maps to a NaN encoding. The only
    // affected finite input has a fixed correctly rounded result. Estimate
    // can retain its cheaper saturation, which remains within its contract.
    const E top_result =
        estimate ? std::numeric_limits<E>::max() : E(65248.0);
    y = word::blend(y, word::cmpeq(x, word::fill(T{}, overflow)),
                    word::fill(T{}, top_result));
  }

  if constexpr (!negative_only) {
    if constexpr (std::is_same_v<E, float32_t>) {
      // FEXPA's top finite code aliases a NaN encoding. Keep the common path
      // compact and use the accurate vector polynomial only for affected lanes.
      const auto upper_tail = svcmpgt_f32(pg, x, svdup_lane_f32(f32_c1, 2));
      if (svptest_any(pg, upper_tail)) {
        y = word::blend(y, upper_tail, exp_f32_accurate<T>(x));
      }
    }
    y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                    word::fill(T{}, std::numeric_limits<E>::infinity()));
  }
  if constexpr (gradual) {
    y = word::blend(y, word::cmplt(x, word::fill(T{}, zero_limit)),
                    word::fill(T{}, E(0)));
  } else {
    y = word::blend(y, subnormal, word::fill(T{}, E(0)));
  }
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!negative_only) {
    y = word::blend(y, word::cmpne(x, x), word::add(x, x));
  }
#endif
  return y;
}

template <ExpTier tier, bool negative_only, typename T>
VECOPS_VFUNC Vec<T> exp_f16_via_f32_unordered(Vec<T> x) {
  using TF = Rebind<float32_t, Half<T>>;
  constexpr TF tf;
  const auto even = word::promote_even(tf, x);
  const auto odd = word::promote_odd(tf, x);
  const auto y_even = exp_sve<tier, negative_only, TF>(even);
  const auto y_odd = exp_sve<tier, negative_only, TF>(odd);
  const auto packed_even = word::demote_even(T{}, y_even);
  return word::demote_odd(T{}, y_odd, packed_even);
}

template <ExpTier tier>
inline constexpr ExpTier kF16ComputeTier =
#ifdef VECOPS_PRESERVE_SUBNORMALS
    tier == ExpTier::Strict ? ExpTier::Strict : ExpTier::Estimate;
#else
    ExpTier::Estimate;
#endif

template <ExpTier tier>
inline constexpr ExpTier kBf16ComputeTier =
#ifdef VECOPS_PRESERVE_SUBNORMALS
    tier == ExpTier::Strict ? ExpTier::Strict : ExpTier::Estimate;
#else
    ExpTier::Estimate;
#endif

#if defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
template <ExpTier tier, bool negative_only, typename T>
VECOPS_VFUNC Vec<T> exp_bf16_via_f32_unordered(Vec<T> x) {
  using TF = Rebind<float32_t, Half<T>>;
  const auto pg = details::ptrue<float32_t>();
  const auto bits = svreinterpret_u16_bf16(x);
  const auto even = svreinterpret_f32_u32(
      svlsl_n_u32_x(pg, svshllb_n_u32(bits, 0), 16));
  const auto odd = svreinterpret_f32_u32(
      svlsl_n_u32_x(pg, svshllt_n_u32(bits, 0), 16));
  const auto y_even = exp_sve<tier, negative_only, TF>(even);
  const auto y_odd = exp_sve<tier, negative_only, TF>(odd);
  const auto packed_even = svcvt_bf16_f32_z(pg, y_even);
  return svcvtnt_bf16_f32_m(packed_even, pg, y_odd);
}
#endif

template <ExpTier tier, bool negative_only = false, typename T>
VECOPS_INLINE Vec<T> dispatch(Vec<T> x) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float16_t> &&
                tier != ExpTier::Estimate) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
    if constexpr (tier == ExpTier::Strict) {
      return exp_f16_via_f32_unordered<
          kF16ComputeTier<tier>, negative_only, T>(x);
    }
#endif
    return exp_sve<tier, negative_only, T>(x);
  } else if constexpr (std::is_same_v<E, bfloat16_t>) {
#if defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
    return exp_bf16_via_f32_unordered<
        kBf16ComputeTier<tier>, negative_only, T>(x);
#else
    constexpr Half<T> th;
    constexpr Rebind<float32_t, Half<T>> tf;
    const auto lo = word::promote(tf, word::lower(T{}, x));
    const auto hi = word::promote(tf, word::upper(T{}, x));
    const auto ylo = [&] {
      if constexpr (std::is_same_v<E, bfloat16_t>) {
        return exp_sve<kBf16ComputeTier<tier>, negative_only,
                       Rebind<float32_t, Half<T>>>(lo);
      } else {
        return exp_sve<tier, negative_only, Rebind<float32_t, Half<T>>>(lo);
      }
    }();
    const auto yhi = [&] {
      if constexpr (std::is_same_v<E, bfloat16_t>) {
        return exp_sve<kBf16ComputeTier<tier>, negative_only,
                       Rebind<float32_t, Half<T>>>(hi);
      } else {
        return exp_sve<tier, negative_only, Rebind<float32_t, Half<T>>>(hi);
      }
    }();
    return word::concat(T{}, word::demote(th, ylo), word::demote(th, yhi));
#endif
  } else {
    return exp_sve<tier, negative_only, T>(x);
  }
}

}  // namespace math_details

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v) { return math_details::dispatch<math_details::ExpTier::Strict, false, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v) { return math_details::dispatch<math_details::ExpTier::Fast, false, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v) { return math_details::dispatch<math_details::ExpTier::Estimate, false, T>(v); }

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg(V v) {
  return math_details::dispatch<math_details::ExpTier::Strict, true, T>(v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_fast(V v) {
  return math_details::dispatch<math_details::ExpTier::Fast, true, T>(v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_est(V v) {
  return math_details::dispatch<math_details::ExpTier::Estimate, true, T>(v);
}

#define VECOPS_SVE_MASKED_EXP(NAME) \
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)> \
VECOPS_VFUNC V NAME(V v, Mask<T> m, V default_v) { \
  const auto safe = word::blend(word::fill(T{}, TypeOf<T>(0)), m, v); \
  return word::blend(default_v, m, word::NAME(safe)); \
}
VECOPS_SVE_MASKED_EXP(exp)
VECOPS_SVE_MASKED_EXP(exp_fast)
VECOPS_SVE_MASKED_EXP(exp_est)
VECOPS_SVE_MASKED_EXP(exp_neg)
VECOPS_SVE_MASKED_EXP(exp_neg_fast)
VECOPS_SVE_MASKED_EXP(exp_neg_est)
#undef VECOPS_SVE_MASKED_EXP

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif
