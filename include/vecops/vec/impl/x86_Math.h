// SIMD transcendental functions for x86.

#ifndef VECOPS_X86_MATH_H
#define VECOPS_X86_MATH_H

#include <cmath>
#include <limits>

#include "x86_Arithmetic.h"
#include "x86_Conversions.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {
namespace math_details {

enum class ExpTier { Strict, Fast, Estimate };

template <typename T>
VECOPS_VFUNC Vec<T> exp_poly_strict(Vec<T> r) {
  using E = TypeOf<T>;
  const auto r2 = word::mul(r, r);
  if constexpr (std::is_same_v<E, float32_t>) {
    const auto p0 = word::fmadd(word::fill(T{}, E(0.166666671633720397949219)), r,
                               word::fill(T{}, E(0.5)));
    const auto p1 = word::fmadd(word::fill(T{}, E(0.00833336077630519866943359)), r,
                               word::fill(T{}, E(0.0416664853692054748535156)));
    const auto p2 = word::fmadd(word::fill(T{}, E(0.000198527617612853646278381)), r,
                               word::fill(T{}, E(0.00139304355252534151077271)));
    const auto q = word::fmadd(word::fmadd(p2, r2, p1), r2, p0);
    return word::add(word::fmadd(q, r2, r), word::fill(T{}, E(1)));
  } else {
    constexpr double c[] = {
      0.5, 0.166666666666666851703837, 0.0416666666666665047591422,
      0.00833333333331652721664984, 0.00138888888889774492207962,
      0.000198412698960509205564975, 2.4801587159235472998791e-5,
      2.75572362911928827629423e-6, 2.75573911234900471893338e-7,
      2.51112930892876518610661e-8, 2.08860621107283687536341e-9
    };
    const auto p0 = word::fmadd(word::fill(T{}, E(c[1])), r, word::fill(T{}, E(c[0])));
    const auto p1 = word::fmadd(word::fill(T{}, E(c[3])), r, word::fill(T{}, E(c[2])));
    const auto p2 = word::fmadd(word::fill(T{}, E(c[5])), r, word::fill(T{}, E(c[4])));
    const auto p3 = word::fmadd(word::fill(T{}, E(c[7])), r, word::fill(T{}, E(c[6])));
    const auto p4 = word::fmadd(word::fill(T{}, E(c[9])), r, word::fill(T{}, E(c[8])));
    const auto p5 = word::fill(T{}, E(c[10]));
    const auto r4 = word::mul(r2, r2);
    const auto r8 = word::mul(r4, r4);
    const auto q0 = word::fmadd(p1, r2, p0);
    const auto q1 = word::fmadd(p3, r2, p2);
    const auto q2 = word::fmadd(p5, r2, p4);
    const auto q = word::fmadd(q2, r8, word::fmadd(q1, r4, q0));
    return word::add(word::fmadd(q, r2, r), word::fill(T{}, E(1)));
  }
}

template <typename T>
VECOPS_VFUNC Vec<T> exp_poly_est(Vec<T> r) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float16_t>) {
    auto p = word::fill(T{}, E(0.1666259765625));
    p = word::fmadd(p, r, word::fill(T{}, E(0.5)));
    p = word::fmadd(p, r, word::fill(T{}, E(1.0)));
    return word::fmadd(p, r, word::fill(T{}, E(1.0)));
  } else {
    const auto c1 = word::fill(T{}, E(1.0214147557098976));
    const auto c2 = word::fill(T{}, E(0.5160640924984561));
    return word::fmadd(c2, word::mul(r, r),
                       word::fmadd(c1, r, word::fill(T{}, E(1))));
  }
}

template <typename T>
VECOPS_VFUNC Vec<T> exp_poly_fast(Vec<T> r) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>) {
    // Degree-5 near-minimax approximation on [-ln(2)/2, ln(2)/2]. Horner's
    // scheme minimizes the instruction count for the throughput-oriented tier.
    auto p = word::fill(T{}, E(0.0083691484928131103515625));
    p = word::fmadd(p, r, word::fill(T{}, E(0.0419175066053867340087891)));
    p = word::fmadd(p, r, word::fill(T{}, E(0.166665047407150268554688)));
    p = word::fmadd(p, r, word::fill(T{}, E(0.499988704919815063476562)));
    p = word::fmadd(p, r, word::fill(T{}, E(1.0)));
    return word::fmadd(p, r, word::fill(T{}, E(1.0)));
  } else if constexpr (std::is_same_v<E, float16_t>) {
    auto p = word::fill(T{}, E(0.041656494140625));
    p = word::fmadd(p, r, word::fill(T{}, E(0.1666259765625)));
    p = word::fmadd(p, r, word::fill(T{}, E(0.5)));
    p = word::fmadd(p, r, word::fill(T{}, E(1.0)));
    return word::fmadd(p, r, word::fill(T{}, E(1.0)));
  } else {
    return exp_poly_strict<T>(r);
  }
}

template <typename T>
VECOPS_VFUNC Vec<T> round_nearest(Vec<T> x) {
  using E = TypeOf<T>;
  constexpr int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_round_ps(x.v, mode);
    else if constexpr (T::Bytes == 32) return _mm256_round_ps(x.v, mode);
    else return _mm512_roundscale_ps(x.v, mode);
  }
  if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_round_pd(x.v, mode);
    else if constexpr (T::Bytes == 32) return _mm256_round_pd(x.v, mode);
    else return _mm512_roundscale_pd(x.v, mode);
  }
#ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) {
      return _mm_castph_si128(_mm_roundscale_ph(_mm_castsi128_ph(x.v), mode));
    } else if constexpr (T::Bytes == 32) {
      return _mm256_castph_si256(_mm256_roundscale_ph(_mm256_castsi256_ph(x.v), mode));
    } else {
      return _mm512_castph_si512(_mm512_roundscale_ph(_mm512_castsi512_ph(x.v), mode));
    }
  }
#endif
}

template <typename T>
VECOPS_VFUNC Mask<T> isnan_mask(Vec<T> x) {
  using E = TypeOf<T>;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_cmp_ps_mask(x.v, x.v, _CMP_UNORD_Q);
    else if constexpr (T::Bytes == 32) return _mm256_cmp_ps_mask(x.v, x.v, _CMP_UNORD_Q);
    else return _mm512_cmp_ps_mask(x.v, x.v, _CMP_UNORD_Q);
  }
  if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_cmp_pd_mask(x.v, x.v, _CMP_UNORD_Q);
    else if constexpr (T::Bytes == 32) return _mm256_cmp_pd_mask(x.v, x.v, _CMP_UNORD_Q);
    else return _mm512_cmp_pd_mask(x.v, x.v, _CMP_UNORD_Q);
  }
#ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) {
      return _mm_cmp_ph_mask(_mm_castsi128_ph(x.v), _mm_castsi128_ph(x.v), _CMP_UNORD_Q);
    } else if constexpr (T::Bytes == 32) {
      return _mm256_cmp_ph_mask(
          _mm256_castsi256_ph(x.v), _mm256_castsi256_ph(x.v), _CMP_UNORD_Q);
    } else {
      return _mm512_cmp_ph_mask(
          _mm512_castsi512_ph(x.v), _mm512_castsi512_ph(x.v), _CMP_UNORD_Q);
    }
  }
#endif
#else
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_castps_si128(_mm_cmpunord_ps(x.v, x.v));
    else return _mm256_castps_si256(_mm256_cmp_ps(x.v, x.v, _CMP_UNORD_Q));
  } else {
    if constexpr (T::Bytes <= 16) return _mm_castpd_si128(_mm_cmpunord_pd(x.v, x.v));
    else return _mm256_castpd_si256(_mm256_cmp_pd(x.v, x.v, _CMP_UNORD_Q));
  }
#endif
}

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
VECOPS_VFUNC Vec<T> scale_pow2(Vec<T> x, Vec<T> qf) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_scalef_ps(x.v, qf.v);
    else if constexpr (T::Bytes == 32) return _mm256_scalef_ps(x.v, qf.v);
    else return _mm512_scalef_ps(x.v, qf.v);
  }
  if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_scalef_pd(x.v, qf.v);
    else if constexpr (T::Bytes == 32) return _mm256_scalef_pd(x.v, qf.v);
    else return _mm512_scalef_pd(x.v, qf.v);
  }
#ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) {
      return _mm_castph_si128(_mm_scalef_ph(_mm_castsi128_ph(x.v), _mm_castsi128_ph(qf.v)));
    } else if constexpr (T::Bytes == 32) {
      return _mm256_castph_si256(
          _mm256_scalef_ph(_mm256_castsi256_ph(x.v), _mm256_castsi256_ph(qf.v)));
    } else {
      return _mm512_castph_si512(
          _mm512_scalef_ph(_mm512_castsi512_ph(x.v), _mm512_castsi512_ph(qf.v)));
    }
  }
#endif
}
#endif

template <typename T>
VECOPS_VFUNC Vec<T> mul_pow2_split(Vec<T> x, Vec<Rebind<Index<TypeOf<T>>, T>> q) {
  using E = TypeOf<T>;
  using I = Index<E>;
  constexpr Rebind<I, T> ti;
  const auto half = word::bit_shr(q, 1);
  const auto rest = word::sub(q, half);
  if constexpr (std::is_same_v<E, float32_t>) {
    const auto bias = word::fill(ti, I(127));
    const auto a = word::bitcast(T{}, word::bit_shl<23>(word::add(half, bias)));
    const auto b = word::bitcast(T{}, word::bit_shl<23>(word::add(rest, bias)));
    return word::mul(word::mul(x, a), b);
  } else {
    const auto bias = word::fill(ti, I(1023));
    const auto a = word::bitcast(T{}, word::bit_shl<52>(word::add(half, bias)));
    const auto b = word::bitcast(T{}, word::bit_shl<52>(word::add(rest, bias)));
    return word::mul(word::mul(x, a), b);
  }
}

template <typename T>
VECOPS_VFUNC Vec<T> mul_pow2_direct(Vec<T> x, Vec<Rebind<Index<TypeOf<T>>, T>> q) {
  using E = TypeOf<T>;
  using I = Index<E>;
  constexpr Rebind<I, T> ti;
  if constexpr (std::is_same_v<E, float16_t>) {
    const auto bits = word::bit_shl<10>(word::add(q, word::fill(ti, I(15))));
    return word::mul(x, word::bitcast(T{}, bits));
  } else if constexpr (std::is_same_v<E, float32_t>) {
    const auto bits = word::bit_shl<23>(word::add(q, word::fill(ti, I(127))));
    return word::mul(x, word::bitcast(T{}, bits));
  } else {
    const auto bits = word::bit_shl<52>(word::add(q, word::fill(ti, I(1023))));
    return word::mul(x, word::bitcast(T{}, bits));
  }
}

template <ExpTier tier, bool negative_only, typename T>
VECOPS_VFUNC Vec<T> exp_float(Vec<T> x) {
  using E = TypeOf<T>;
  using I = Index<E>;
  constexpr Rebind<I, T> ti;
  constexpr bool strict = tier == ExpTier::Strict;
  constexpr bool estimate = tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif

#if defined(CPU_CAPABILITY_AVX512) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  // The default execution mode enables FTZ. Clamp to a finite exponent
  // sentinel and let VSCALEF produce zero naturally, avoiding a compare and
  // blend after the polynomial. Preserve-subnormal builds cannot rely on FTZ.
  constexpr bool natural_underflow = true;
#else
  constexpr bool natural_underflow = false;
#endif

  const E overflow = std::is_same_v<E, float16_t> ? E(11.0859375) :
      std::is_same_v<E, float32_t> ? E(88.72283905206835) : E(709.782712893384);
  const E zero_limit = std::is_same_v<E, float16_t> ? E(-17.328679513999) :
      std::is_same_v<E, float32_t> ? E(-103.972084045410) : E(-745.1332191019411);
  const E normal_limit = std::is_same_v<E, float16_t> ? E(-9.704060527839) :
      std::is_same_v<E, float32_t> ? E(-87.3365447505531) : E(-708.3964185322641);
  const E zero_sentinel = std::is_same_v<E, float16_t> ? E(-18.0) :
      std::is_same_v<E, float32_t> ? E(-104.0) : E(-746.0);
  const E lower = natural_underflow ? zero_sentinel :
      (gradual ? zero_limit : normal_limit);

  // MAX/MIN with the bound as the second operand also sanitizes NaNs before
  // conversion. The original NaN is restored at the end for the general API.
  auto xc = word::max(x, word::fill(T{}, lower));
  if constexpr (!negative_only) xc = word::min(xc, word::fill(T{}, overflow));

  const auto qf = round_nearest<T>(word::mul(
      xc, word::fill(T{}, E(1.442695040888963407359924681))));

  auto r = word::fmadd(qf, word::fill(T{}, !estimate
      ? (std::is_same_v<E, float16_t> ? E(-0.693359375) :
         std::is_same_v<E, float32_t> ? E(-0.693145751953125f) :
         E(-0.6931471805596629565116018))
      : E(-0.693147180559945309417232121458)), xc);
  if constexpr (!estimate) {
    r = word::fmadd(qf, word::fill(T{}, std::is_same_v<E, float16_t>
        ? E(0.000212192535400390625) : std::is_same_v<E, float32_t>
        ? E(-1.428606765330187045e-6f) : E(-0.28235290563031577122588448175e-12)), r);
  }

  const auto poly = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (estimate) return exp_poly_est<T>(r);
    else if constexpr (tier == ExpTier::Fast) return exp_poly_fast<T>(r);
    else if constexpr (std::is_same_v<E, float16_t>) return exp_poly_fast<T>(r);
    else return exp_poly_strict<T>(r);
  }();

  Vec<T> y;
#if defined(CPU_CAPABILITY_AVX512)
  y = scale_pow2<T>(poly, qf);
#else
  const auto qi = word::convert(ti, qf);
  if constexpr (negative_only && !gradual) y = mul_pow2_direct<T>(poly, qi);
  else y = mul_pow2_split<T>(poly, qi);
#endif

  if constexpr (!negative_only) {
    y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                    word::fill(T{}, std::numeric_limits<E>::infinity()));
  }
  if constexpr (!natural_underflow) {
    y = word::blend(y, word::cmplt(x, word::fill(T{}, gradual ? zero_limit : normal_limit)),
                    word::fill(T{}, E(0)));
  }
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!negative_only) {
    y = word::blend(y, isnan_mask<T>(x), word::add(x, x));
  }
#endif
  return y;
}

template <ExpTier tier, bool negative_only, typename T>
VECOPS_VFUNC Vec<T> exp_low_precision(Vec<T> x) {
  using E = TypeOf<T>;
  // A half word promoted to f32 occupies exactly one full hardware word.
  // Splitting first therefore avoids scalar extraction even for a full f16 or
  // bf16 register.
  constexpr Half<T> th;
  constexpr Rebind<float32_t, Half<T>> tf;
  const auto lo = word::promote(tf, word::lower(T{}, x));
  const auto hi = word::promote(tf, word::upper(T{}, x));
  constexpr ExpTier compute_tier =
      std::is_same_v<E, bfloat16_t> && tier == ExpTier::Estimate ? ExpTier::Fast : tier;
  const auto ylo = exp_float<compute_tier, negative_only, Rebind<float32_t, Half<T>>>(lo);
  const auto yhi = exp_float<compute_tier, negative_only, Rebind<float32_t, Half<T>>>(hi);
  return word::concat(T{}, word::demote(th, ylo), word::demote(th, yhi));
}

template <ExpTier tier, bool negative_only = false, typename T>
VECOPS_VFUNC Vec<T> dispatch(Vec<T> x) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t> || std::is_same_v<E, float64_t>) {
    return exp_float<tier, negative_only, T>(x);
#ifdef HAS_AVX512_FP16
  } else if constexpr (std::is_same_v<E, float16_t>) {
    return exp_float<tier, negative_only, T>(x);
#endif
  } else {
    return exp_low_precision<tier, negative_only, T>(x);
  }
}

}  // namespace math_details

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v) {
  return math_details::dispatch<math_details::ExpTier::Strict, false, T>(v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v) {
  return math_details::dispatch<math_details::ExpTier::Fast, false, T>(v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v) {
  return math_details::dispatch<math_details::ExpTier::Estimate, false, T>(v);
}
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

#define VECOPS_X86_MASKED_EXP(NAME) \
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)> \
VECOPS_VFUNC V NAME(V v, Mask<T> m, V default_v) { \
  const auto safe = word::blend(word::fill(T{}, TypeOf<T>(0)), m, v); \
  return word::blend(default_v, m, word::NAME(safe)); \
}
VECOPS_X86_MASKED_EXP(exp)
VECOPS_X86_MASKED_EXP(exp_fast)
VECOPS_X86_MASKED_EXP(exp_est)
VECOPS_X86_MASKED_EXP(exp_neg)
VECOPS_X86_MASKED_EXP(exp_neg_fast)
VECOPS_X86_MASKED_EXP(exp_neg_est)
#undef VECOPS_X86_MASKED_EXP

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif
