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
  const auto c1 = word::fill(T{}, E(1.0214147557098976));
  const auto c2 = word::fill(T{}, E(0.5160640924984561));
  return word::fmadd(c2, word::mul(r, r),
                     word::fmadd(c1, r, word::fill(T{}, E(1))));
}

template <typename T>
VECOPS_VFUNC Vec<T> mul_pow2_split(Vec<T> x, Vec<Rebind<Index<TypeOf<T>>, T>> q) {
  using E = TypeOf<T>;
  using I = Index<E>;
  constexpr Rebind<I, T> ti;
  const auto half = word::bit_shr(q, 1);
  const auto rest = word::sub(q, half);
  if constexpr (std::is_same_v<E, float32_t>) {
    const auto bias = word::fill(ti, I(127));
    const auto a = word::bitcast(T{}, word::bit_shl(word::add(half, bias), 23));
    const auto b = word::bitcast(T{}, word::bit_shl(word::add(rest, bias), 23));
    return word::mul(word::mul(x, a), b);
  } else {
    const auto bias = word::fill(ti, I(1023));
    const auto a = word::bitcast(T{}, word::bit_shl(word::add(half, bias), 52));
    const auto b = word::bitcast(T{}, word::bit_shl(word::add(rest, bias), 52));
    return word::mul(word::mul(x, a), b);
  }
}

template <ExpTier tier, typename T>
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

  constexpr E overflow = std::is_same_v<E, float32_t> ? E(88.72283905206835) : E(709.782712893384);
  constexpr E zero_limit = std::is_same_v<E, float32_t> ? E(-103.972084045410) : E(-745.1332191019411);
  constexpr E normal_limit = std::is_same_v<E, float32_t> ? E(-87.3365447505531) : E(-708.3964185322641);
  constexpr E lower = gradual ? zero_limit : normal_limit;

  auto xc = word::blend(x, word::cmpgt(x, word::fill(T{}, overflow)), word::fill(T{}, overflow));
  xc = word::blend(xc, word::cmplt(xc, word::fill(T{}, lower)), word::fill(T{}, lower));

  const auto half = word::fill(T{}, E(0.5));
  const auto neg_half = word::fill(T{}, E(-0.5));
  const auto offs = word::blend(half, word::cmplt(xc, word::fill(T{}, E(0))), neg_half);
  const auto z = word::fmadd(xc, word::fill(T{}, E(1.442695040888963407359924681)), offs);
  const auto qi = word::convert(ti, z);
  const auto qf = word::convert(T{}, qi);

  auto r = word::fmadd(qf, word::fill(T{}, std::is_same_v<E, float32_t>
      ? E(-0.693145751953125f) : E(-0.6931471805596629565116018)), xc);
  r = word::fmadd(qf, word::fill(T{}, std::is_same_v<E, float32_t>
      ? E(-1.428606765330187045e-6f) : E(-0.28235290563031577122588448175e-12)), r);

  const auto poly = estimate ? exp_poly_est<T>(r) : exp_poly_strict<T>(r);
  // Splitting also avoids constructing an infinity for q == max_exponent + 1;
  // values just below log(max) still have a finite result there.
  auto y = mul_pow2_split<T>(poly, qi);

#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!gradual) {
    y = word::blend(y, word::cmplt(x, word::fill(T{}, normal_limit)), word::fill(T{}, E(0)));
  }
  y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                  word::fill(T{}, std::numeric_limits<E>::infinity()));
  y = word::blend(y, word::cmplt(x, word::fill(T{}, gradual ? zero_limit : normal_limit)),
                  word::fill(T{}, E(0)));
  y = word::blend(y, word::cmpne(x, x), word::add(x, x));
#else
  if constexpr (strict) {
    y = word::blend(y, word::cmpgt(x, word::fill(T{}, overflow)),
                    word::fill(T{}, std::numeric_limits<E>::infinity()));
    y = word::blend(y, word::cmplt(x, word::fill(T{}, gradual ? zero_limit : normal_limit)),
                    word::fill(T{}, E(0)));
    y = word::blend(y, word::cmpne(x, x), word::add(x, x));
  }
#endif
  return y;
}

template <ExpTier tier, typename T>
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
  const auto ylo = exp_float<compute_tier, Rebind<float32_t, Half<T>>>(lo);
  const auto yhi = exp_float<compute_tier, Rebind<float32_t, Half<T>>>(hi);
  return word::concat(T{}, word::demote(th, ylo), word::demote(th, yhi));
}

template <ExpTier tier, typename T>
VECOPS_VFUNC Vec<T> dispatch(Vec<T> x) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t> || std::is_same_v<E, float64_t>) {
    return exp_float<tier, T>(x);
  } else {
    return exp_low_precision<tier, T>(x);
  }
}

}  // namespace math_details

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v) { return math_details::dispatch<math_details::ExpTier::Strict, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v) { return math_details::dispatch<math_details::ExpTier::Fast, T>(v); }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v) { return math_details::dispatch<math_details::ExpTier::Estimate, T>(v); }

#define VECOPS_X86_MASKED_EXP(NAME) \
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)> \
VECOPS_VFUNC V NAME(V v, Mask<T> m, V default_v) { \
  const auto safe = word::blend(word::fill(T{}, TypeOf<T>(0)), m, v); \
  return word::blend(default_v, m, word::NAME(safe)); \
}
VECOPS_X86_MASKED_EXP(exp)
VECOPS_X86_MASKED_EXP(exp_fast)
VECOPS_X86_MASKED_EXP(exp_est)
#undef VECOPS_X86_MASKED_EXP

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif
