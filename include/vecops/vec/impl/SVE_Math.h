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

template <ExpTier tier, typename T>
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

  E overflow, zero_limit, normal_limit, sub_shift, sub_scale;
  if constexpr (std::is_same_v<E, float16_t>) {
    // Largest binary16 input whose exponential is still finite.
    overflow = E(11.0859375); zero_limit = E(-17.328679513999);
    normal_limit = E(-9.704060527839); sub_shift = E(5.545177444480);
    sub_scale = E(0.00390625);
  } else if constexpr (std::is_same_v<E, float32_t>) {
    overflow = E(88.72283905206835); zero_limit = E(-103.972084045410);
    normal_limit = E(-87.3365447505531); sub_shift = E(44.3614195558365);
    sub_scale = E(0x1p-64f);
  } else {
    overflow = E(709.782712893384); zero_limit = E(-745.1332191019411);
    normal_limit = E(-708.3964185322641); sub_shift = E(354.891356446692);
    sub_scale = E(0x1p-512);
  }

  const auto subnormal = word::cmplt(x, word::fill(T{}, normal_limit));
  auto xc = x;
  if constexpr (gradual) {
    xc = word::blend(xc, subnormal, word::add(xc, word::fill(T{}, sub_shift)));
  }
  const E lower = normal_limit;
  xc = word::blend(xc, word::cmplt(xc, word::fill(T{}, lower)), word::fill(T{}, lower)); // TODO use clamp
  xc = word::blend(xc, word::cmpgt(xc, word::fill(T{}, overflow)), word::fill(T{}, overflow));

  Vec<T> y;
  if constexpr (std::is_same_v<E, float32_t>) {
    constexpr float inv_step = 92.33248261689366f;  // 64 / ln(2)
    const auto nf = svrinta_f32_x(pg, svmul_n_f32_x(pg, xc, inv_step));
    const auto code = svcvt_u32_f32_x(pg, svadd_n_f32_x(pg, nf, 8128.0f));
    const auto scale = svexpa_f32(code);
    auto r = svmla_n_f32_x(pg, xc, nf, -0.010830402374267578125f);
    r = svmla_n_f32_x(pg, r, nf, -2.232198101786e-8f);
    if constexpr (estimate) {
      y = svmla_f32_x(pg, scale, scale, r);
    } else {
      const auto p = svmla_n_f32_x(pg, r, svmul_f32_x(pg, r, r), 0.5f);
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
      const auto qf = svrinta_f64_x(pg, svmul_n_f64_x(pg, xc, 1.4426950408889634074));
      const auto qi = svcvt_s64_f64_x(pg, qf);
      auto r = svmla_n_f64_x(pg, xc, qf, -0.6931471805596629565116018);
      r = svmla_n_f64_x(pg, r, qf, -0.28235290563031577122588448175e-12);
      const auto r2 = svmul_f64_x(pg, r, r);
      constexpr double c[] = {
        0.5, 0.166666666666666851703837, 0.0416666666666665047591422,
        0.00833333333331652721664984, 0.00138888888889774492207962,
        0.000198412698960509205564975, 2.4801587159235472998791e-5,
        2.75572362911928827629423e-6, 2.75573911234900471893338e-7,
        2.51112930892876518610661e-8, 2.08860621107283687536341e-9
      };
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
      const auto poly_q = svmla_f64_x(pg, svmla_f64_x(pg, q0, q1, r4), q2, r8);
      const auto poly = svadd_n_f64_x(pg, svmla_f64_x(pg, r, poly_q, r2), 1.0);
      const auto qh = svasr_n_s64_x(pg, qi, 1);
      const auto qr = svsub_s64_x(pg, qi, qh);
      const auto sh = svreinterpret_f64_s64(
          svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qh, 1023), 52));
      const auto sr = svreinterpret_f64_s64(
          svlsl_n_s64_x(pg, svadd_n_s64_x(pg, qr, 1023), 52));
      y = svmul_f64_x(pg, svmul_f64_x(pg, poly, sh), sr);
    }
  } else {
    constexpr float inv_step_f = 46.16624130844683f;  // 32 / ln(2)
    const E inv_step = E(inv_step_f);
    const auto nf = svrinta_f16_x(pg, svmul_n_f16_x(pg, xc, inv_step));
    const auto code = svcvt_u16_f16_x(pg, svadd_n_f16_x(pg, nf, E(480)));
    const auto scale = svexpa_f16(code);
    const auto r = svmla_n_f16_x(pg, xc, nf, E(-0.02166084939249829));
    if constexpr (estimate) {
      y = svmla_f16_x(pg, scale, scale, r);
    } else {
      const auto p = svmla_n_f16_x(pg, r, svmul_f16_x(pg, r, r), E(0.5));
      y = svmla_f16_x(pg, scale, scale, p);
    }
  }

  if constexpr (gradual) {
    y = word::blend(y, subnormal, word::mul(y, word::fill(T{}, sub_scale)));
  }
  if constexpr (estimate && std::is_same_v<E, float16_t>) {
    // FEXPA's top finite binary16 code maps to a NaN encoding. The only
    // affected finite input rounds to a value within the estimate contract
    // when saturated to max finite.
    y = word::blend(y, word::cmpeq(x, word::fill(T{}, overflow)),
                    word::fill(T{}, std::numeric_limits<E>::max()));
  }

  y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                  word::fill(T{}, std::numeric_limits<E>::infinity()));
  y = word::blend(y, word::cmplt(x, word::fill(T{}, gradual ? zero_limit : normal_limit)),
                  word::fill(T{}, E(0)));
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  y = word::blend(y, word::cmpne(x, x), word::add(x, x));
#endif
  if constexpr (std::is_same_v<E, float32_t> || std::is_same_v<E, float64_t>) {
    const E repair_upper = std::is_same_v<E, float32_t> ? E(88.0) : E(709.0);
    const auto upper_tail = word::cmpgt(x, word::fill(T{}, repair_upper));
    const bool repair_subnormal = gradual && svptest_any(pg, subnormal);
    const bool repair_top = svptest_any(pg, upper_tail);
    if (repair_subnormal || repair_top) {
      for (nint_t i = 0; i < size(T{}); ++i) {
        const E xi = word::get(x, i);
        if (std::isfinite(xi) &&
            ((gradual && xi < normal_limit) || xi > repair_upper)) {
          E yi = std::exp(xi);
          if constexpr (!strict) {
            if (yi < std::numeric_limits<E>::min()) yi = E(0);
          }
          y = word::set(y, i, yi);
        }
      }
    }
  }
  return y;
}

template <ExpTier tier, typename T>
VECOPS_VFUNC Vec<T> dispatch(Vec<T> x) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, bfloat16_t> ||
                (std::is_same_v<E, float16_t> && tier != ExpTier::Estimate)) {
    constexpr Half<T> th;
    constexpr Rebind<float32_t, Half<T>> tf;
    const auto lo = word::promote(tf, word::lower(T{}, x));
    const auto hi = word::promote(tf, word::upper(T{}, x));
    const auto ylo = [&] {
      if constexpr (std::is_same_v<E, bfloat16_t> && tier == ExpTier::Estimate)
        return exp_f32_accurate<Rebind<float32_t, Half<T>>>(lo);
      else
        return exp_sve<tier, Rebind<float32_t, Half<T>>>(lo);
    }();
    const auto yhi = [&] {
      if constexpr (std::is_same_v<E, bfloat16_t> && tier == ExpTier::Estimate)
        return exp_f32_accurate<Rebind<float32_t, Half<T>>>(hi);
      else
        return exp_sve<tier, Rebind<float32_t, Half<T>>>(hi);
    }();
    return word::concat(T{}, word::demote(th, ylo), word::demote(th, yhi));
  } else {
    return exp_sve<tier, T>(x);
  }
}

}  // namespace math_details

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v) { return math_details::dispatch<math_details::ExpTier::Strict, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v) { return math_details::dispatch<math_details::ExpTier::Fast, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v) { return math_details::dispatch<math_details::ExpTier::Estimate, T>(v); }

// TODO: implement the x <= 0 specialization. It can omit the upper clamp,
// overflow repair, NaN handling, and the dynamic sign-dependent rounding used
// by a general exponential. The forwarding stubs keep the public API correct.
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg(V v) { return word::exp(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_fast(V v) { return word::exp_fast(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_est(V v) { return word::exp_est(v); }

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
