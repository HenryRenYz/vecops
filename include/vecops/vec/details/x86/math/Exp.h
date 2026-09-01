#ifndef VECOPS_VEC_DETAILS_X86_MATH_EXP_H
#define VECOPS_VEC_DETAILS_X86_MATH_EXP_H

/**
 * @file Exp.h
 * @brief x86 backend implementations for the exp family (exp, exp2, exp10).
 *
 * Every base shares one reduction skeleton, parameterized by a per-base
 * constant set: base^x = 2^n * (1 + poly(r)) with n = rint(x * log2(base))
 * produced by a round-to-nearest step and r = x - n * log_base(2) recovered
 * through a two-part Cody-Waite subtraction (fused, so the hi part needs no
 * trailing zero bits). Base 2 degenerates to the cheapest reduction in the
 * family: n = rint(x) and r = x - n hold exactly, with no range multiplier
 * and no correction term. The reduced argument always satisfies
 * |r * ln(base)| <= ln(2)/2 + slack, so the per-base polynomials are the
 * same minimax problems rescaled, and the reconstruction multiplies by 2^n
 * through SCALEF on AVX-512 or integer exponent injection otherwise.
 *
 * Coefficient provenance:
 *  - f64 exp2/exp10 (Strict and Fast share one set): the degree-11 minimax
 *    coefficients published by the SLEEF project
 *    (https://github.com/shibatch/sleef), src/libm/sleefsimddp.c functions
 *    xexp2/xexp10, distributed under the Boost Software License 1.0.
 *  - f32 Strict/Fast and the Estimate sets for every width: Remez fits
 *    computed for this backend on the exact reachable r-range (the Estimate
 *    tier minimizes relative error to fit the 0.006 contract clause).
 *  - f16 sets: the tuned e-base coefficients scaled by ln(base)^k and
 *    re-quantized to float16, then validated against the full 65536-point
 *    f16 grid; the resulting bit patterns match the constants the Arm
 *    optimized-routines project ships in its SVE exp2/exp10 kernels.
 *  - Boundary constants (overflow / flush / gradual / sentinel) are derived
 *    from the IEEE-754 rounding midpoints of each format, snapped down (up
 *    for the flush bounds) to representable grid values.
 *
 * Narrow formats: f16 evaluates natively under AVX512-FP16 and widens
 * through paired f32 words otherwise; bf16 always widens. The widened paths
 * reuse the f32 kernels and their boundaries (the formats share the f32
 * exponent range), with the bf16 Estimate tier bumped to the f32 Fast poly.
 */

#include <cmath>
#include <cstring>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                Exponential helpers and word implementations                //
/* **************************************************************************** */


template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_round_nearest(
    NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  using Raw = decltype(value.value);
  constexpr int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_castph_si128(_mm_roundscale_ph(_mm_castsi128_ph(value.value), mode))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_castph_si256(_mm256_roundscale_ph(
            _mm256_castsi256_ph(value.value), mode))};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_castph_si512(_mm512_roundscale_ph(
        _mm512_castsi512_ph(value.value), mode))};
#endif
#endif
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_round_ps(value.value, mode)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_round_ps(value.value, mode)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_roundscale_ps(value.value, mode)};
#endif
  } else {
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_round_pd(value.value, mode)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_round_pd(value.value, mode)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_roundscale_pd(value.value, mode)};
#endif
  }
}

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<IndexTag<Tag>> x86_exp_to_index(
    Tag, NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  using Raw = decltype(value.value);
  using Result = NativeWordVec<IndexTag<Tag>>;
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) return Result{
        _mm_cvttps_epi32(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return Result{
        _mm256_cvttps_epi32(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Result{_mm512_cvttps_epi32(value.value)};
#endif
  } else {
#if defined(HAS_AVX512DQ)
    if constexpr (sizeof(Raw) == 16) return Result{
        _mm_cvttpd_epi64(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return Result{
        _mm256_cvttpd_epi64(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Result{_mm512_cvttpd_epi64(value.value)};
#endif
#else
    constexpr std::size_t lanes = sizeof(Raw) / sizeof(double);
    alignas(64) double input[lanes];
    alignas(64) int64_t output[lanes];
    std::memcpy(input, &value.value, sizeof(Raw));
    for (std::size_t lane = 0; lane < lanes; ++lane)
      output[lane] = static_cast<int64_t>(input[lane]);
    decltype(Result{}.value) raw;
    std::memcpy(&raw, output, sizeof(Raw));
    return Result{raw};
#endif
  }
}

/* **************************************************************************** */
//                Per-base constants                                            //
/* **************************************************************************** */

/**
 * Reduction constants for one base. log2_base scales x before the
 * round-to-nearest step; inv_hi/inv_lo are the two-part split of
 * log_base(2) = ln(2)/ln(base) used to recover the reduced argument
 * r = x - n*inv_hi - n*inv_lo. The split only has to hold under fused
 * multiplication-subtraction, so the hi part carries no forced trailing
 * zero bits. Base 2 needs neither: its step is exact.
 */
template <ExpBase Base>
struct X86ExpReduction;

template <>
struct X86ExpReduction<ExpBase::E> {
  static constexpr double log2_base = 1.442695040888963407359924681;
};

template <>
struct X86ExpReduction<ExpBase::Base2> {
  // n = rint(x), r = x - n exactly: no multiplier, no correction parts.
};

template <>
struct X86ExpReduction<ExpBase::Base10> {
  static constexpr double log2_base = 3.321928094887362347870319429489390;
  static constexpr double inv_hi = 0.30102999566383914498;
  static constexpr double inv_lo = 1.4205023227266099418e-13;
  static constexpr float inv_hi_f32 = 0.3010253906f;
  static constexpr float inv_lo_f32 = 4.605038981e-06f;
  static constexpr double inv_hi_f16 = 0.301025390625;
  static constexpr double inv_lo_f16 = 4.605038981e-06;
};

/**
 * Domain boundaries per base and storage type, as exact doubles (narrowed
 * by the caller). overflow is the largest input whose result stays finite
 * after one rounding; normal_limit flushes everything whose result would be
 * subnormal (used when gradual results are neither preserved nor produced
 * naturally); zero_limit is the true underflow boundary of the format; and
 * zero_sentinel is the clamp fed to computations that let the hardware
 * round subnormal results away. Values sit on the representable grid of the
 * storage type, chosen so no finite-result input is selected to a tail.
 */
template <ExpBase Base, typename T>
struct X86ExpDomain;

template <typename T>
struct X86ExpDomain<ExpBase::E, T> {
  static constexpr double overflow =
      std::same_as<T, float16_t> ? 11.0859375
      : std::same_as<T, float32_t> ? 88.72283905206835 : 709.782712893384;
  static constexpr double zero_limit =
      std::same_as<T, float16_t> ? -17.328679513999
      : std::same_as<T, float32_t> ? -103.972084045410
                                   : -745.1332191019411;
  static constexpr double normal_limit =
      std::same_as<T, float16_t> ? -9.704060527839
      : std::same_as<T, float32_t> ? -87.3365447505531
                                   : -708.3964185322641;
  static constexpr double zero_sentinel =
      std::same_as<T, float16_t> ? -18.0
      : std::same_as<T, float32_t> ? -104.0 : -746.0;
};

template <typename T>
struct X86ExpDomain<ExpBase::Base2, T> {
  static constexpr double overflow =
      std::same_as<T, float16_t> ? 15.9921875
      : std::same_as<T, float32_t> ? 127.99999237060547
                                   : 1023.9999999999999;
  static constexpr double zero_limit =
      std::same_as<T, float16_t> ? -25.0
      : std::same_as<T, float32_t> ? -150.0 : -1075.0;
  static constexpr double normal_limit =
      std::same_as<T, float16_t> ? -14.0
      : std::same_as<T, float32_t> ? -126.0 : -1022.0;
  static constexpr double zero_sentinel =
      std::same_as<T, float16_t> ? -26.0
      : std::same_as<T, float32_t> ? -151.0 : -1076.0;
};

template <typename T>
struct X86ExpDomain<ExpBase::Base10, T> {
  static constexpr double overflow =
      std::same_as<T, float16_t> ? 4.8125
      : std::same_as<T, float32_t> ? 38.531837463378906
                                   : 308.25471555991669;
  static constexpr double zero_limit =
      std::same_as<T, float16_t> ? -7.52734375
      : std::same_as<T, float32_t> ? -45.154499053955078
                                   : -323.60724533877976;
  static constexpr double normal_limit =
      std::same_as<T, float16_t> ? -4.21484375
      : std::same_as<T, float32_t> ? -37.929779052734375
                                   : -307.65265556858878;
  static constexpr double zero_sentinel =
      std::same_as<T, float16_t> ? -7.5625
      : std::same_as<T, float32_t> ? -46.0 : -324.0;
};

/** Estimate-tier coefficients [a1, a2]: 1 + a1*r + a2*r^2, per base. */
inline constexpr double kX86ExpEstimateE[2] = {
    1.0214147557098976, 0.5160640924984561};
inline constexpr double kX86ExpEstimateB2[2] = {
    0.70318234, 0.240194619};
inline constexpr double kX86ExpEstimateB10[2] = {
    2.33592129, 2.65059733};

template <ExpBase Base>
consteval const double* x86_exp_estimate() {
  if constexpr (Base == ExpBase::E) return kX86ExpEstimateE;
  else if constexpr (Base == ExpBase::Base2) return kX86ExpEstimateB2;
  else return kX86ExpEstimateB10;
}

/**
 * Strict/Fast polynomial coefficients per base and width, ascending
 * (k[0] = a1 is the lead). f64 Strict and Fast share one degree-11 set per
 * base (SLEEF's u10 coefficients); f32 splits into a degree-7 Strict set
 * and a degree-5 Fast set; f16 uses degree 4 for both tiers. The e-base
 * sets stay inline in x86_exp_poly to keep that path's generated code
 * byte-identical to the historically probe-validated form.
 */
inline constexpr double kX86ExpF64PolyB2[11] = {
    0.6931471805599452862, 0.2402265069591012214,
    0.5550410866482046596e-1, 0.9618129107597600536e-2,
    0.1333355814670499073e-2, 0.1540353045101147808e-3,
    0.1525273353517584730e-4, 0.1321543872511327615e-5,
    0.1017819260921760451e-6, 0.7073164598085707425e-8,
    0.4434359082926529454e-9};
inline constexpr double kX86ExpF64PolyB10[11] = {
    0.2302585092994045901e+1, 0.2650949055239205876e+1,
    0.2034678592293432953e+1, 0.1171255148908541655e+1,
    0.5393829292058536229e+0, 0.2069958494722676234e+0,
    0.6808936399446784138e-1, 0.1959762320720533080e-1,
    0.5013975546789733659e-2, 0.1157488415217187375e-2,
    0.2411463498334267652e-3};

inline constexpr double kX86ExpF32StrictB2[7] = {
    0.693147182, 0.240226507, 0.0555041023, 0.00961805787,
    0.00133334543, 0.000154614507, 1.53100809e-05};
inline constexpr double kX86ExpF32StrictB10[7] = {
    2.30258512, 2.650949, 2.0346787, 1.17124629,
    0.539378762, 0.207774192, 0.0683453679};

inline constexpr double kX86ExpF32FastB2[5] = {
    0.693147182, 0.240223482, 0.0555033311, 0.00966637861,
    0.00134004327};
inline constexpr double kX86ExpF32FastB10[5] = {
    2.30258512, 2.65091586, 2.03465009, 1.17713082,
    0.542088211};

/**
 * f16 coefficient sets (all exactly representable in float16): the strict
 * set is [a1..a4] and the estimate set is [a1..a3]. For bases 2 and 10 the
 * bit patterns coincide with the constants in Arm optimized-routines' SVE
 * exp2/exp10 f16 kernels (ln(base) scaling lands on the same grid points).
 */
inline constexpr double kX86ExpF16StrictB2[4] = {
    0.693359375, 0.240234375, 0.05548095703125, 0.009613037109375};
inline constexpr double kX86ExpF16StrictB10[4] = {
    2.302734375, 2.650390625, 2.03515625, 1.1708984375};
inline constexpr double kX86ExpF16StrictE[4] = {
    1.0, 0.5, 0.1666259765625, 0.041656494140625};
inline constexpr double kX86ExpF16EstimateB2[3] = {
    0.693359375, 0.240234375, 0.05548095703125};
inline constexpr double kX86ExpF16EstimateB10[3] = {
    2.302734375, 2.650390625, 2.03515625};
inline constexpr double kX86ExpF16EstimateE[3] = {
    1.0, 0.5, 0.1666259765625};

template <ExpBase Base>
consteval const double* x86_exp_f16_strict() {
  if constexpr (Base == ExpBase::E) return kX86ExpF16StrictE;
  else if constexpr (Base == ExpBase::Base2) return kX86ExpF16StrictB2;
  else return kX86ExpF16StrictB10;
}

template <ExpBase Base>
consteval const double* x86_exp_f16_estimate() {
  if constexpr (Base == ExpBase::E) return kX86ExpF16EstimateE;
  else if constexpr (Base == ExpBase::Base2) return kX86ExpF16EstimateB2;
  else return kX86ExpF16EstimateB10;
}

template <ExpBase Base, Accuracy Tier, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_poly(
    Tag tag, NativeWordVec<Tag> r) {
  using T = ElementOf<Tag>;
  const auto r2 = mul(tag, r, r);
  if constexpr (Tier == Accuracy::Estimate) {
    if constexpr (std::same_as<T, float16_t>) {
      constexpr const double* k = x86_exp_f16_estimate<Base>();
      auto p = fill_word(tag, T(k[2]));
      p = fmadd(tag, p, r, fill_word(tag, T(k[1])));
      p = fmadd(tag, p, r, fill_word(tag, T(k[0])));
      return fmadd(tag, p, r, fill_word(tag, T(1)));
    }
    constexpr const double* k = x86_exp_estimate<Base>();
    return fmadd(
        tag, fill_word(tag, T(k[1])), r2,
        fmadd(
            tag, fill_word(tag, T(k[0])), r,
            fill_word(tag, T(1))));
  } else if constexpr (std::same_as<T, float16_t>) {
    constexpr const double* k = x86_exp_f16_strict<Base>();
    auto p = fill_word(tag, T(k[3]));
    p = fmadd(tag, p, r, fill_word(tag, T(k[2])));
    p = fmadd(tag, p, r, fill_word(tag, T(k[1])));
    p = fmadd(tag, p, r, fill_word(tag, T(k[0])));
    return fmadd(tag, p, r, fill_word(tag, T(1)));
  } else if constexpr (Tier == Accuracy::Fast && std::same_as<T, float32_t>) {
    if constexpr (Base == ExpBase::E) {
      auto p = fill_word(tag, T(0.0083691484928131103515625));
      p = fmadd(tag, p, r, fill_word(tag, T(0.0419175066053867340087891)));
      p = fmadd(tag, p, r, fill_word(tag, T(0.166665047407150268554688)));
      p = fmadd(tag, p, r, fill_word(tag, T(0.499988704919815063476562)));
      p = fmadd(tag, p, r, fill_word(tag, T(1)));
      return fmadd(tag, p, r, fill_word(tag, T(1)));
    } else {
      constexpr const double* k =
          Base == ExpBase::Base2 ? kX86ExpF32FastB2 : kX86ExpF32FastB10;
      auto p = fill_word(tag, T(k[4]));
      p = fmadd(tag, p, r, fill_word(tag, T(k[3])));
      p = fmadd(tag, p, r, fill_word(tag, T(k[2])));
      p = fmadd(tag, p, r, fill_word(tag, T(k[1])));
      p = fmadd(tag, p, r, fill_word(tag, T(k[0])));
      return fmadd(tag, p, r, fill_word(tag, T(1)));
    }
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (Base == ExpBase::E) {
      const auto p0 = fmadd(
          tag, fill_word(tag, T(0.166666671633720397949219)), r,
          fill_word(tag, T(0.5)));
      const auto p1 = fmadd(
          tag, fill_word(tag, T(0.00833336077630519866943359)), r,
          fill_word(tag, T(0.0416664853692054748535156)));
      const auto p2 = fmadd(
          tag, fill_word(tag, T(0.000198527617612853646278381)), r,
          fill_word(tag, T(0.00139304355252534151077271)));
      const auto q = fmadd(
          tag, fmadd(tag, p2, r2, p1), r2, p0);
      return add(
          tag, fmadd(tag, q, r2, r), fill_word(tag, T(1)));
    } else {
      constexpr const double* k =
          Base == ExpBase::Base2 ? kX86ExpF32StrictB2 : kX86ExpF32StrictB10;
      const auto p0 = fmadd(
          tag, fill_word(tag, T(k[2])), r, fill_word(tag, T(k[1])));
      const auto p1 = fmadd(
          tag, fill_word(tag, T(k[4])), r, fill_word(tag, T(k[3])));
      const auto p2 = fmadd(
          tag, fill_word(tag, T(k[6])), r, fill_word(tag, T(k[5])));
      const auto q = fmadd(
          tag, fmadd(tag, p2, r2, p1), r2, p0);
      return fmadd(
          tag, fmadd(tag, q, r, fill_word(tag, T(k[0]))), r,
          fill_word(tag, T(1)));
    }
  } else if constexpr (Base == ExpBase::E) {
    constexpr double coefficients[] = {
        0.5, 0.166666666666666851703837,
        0.0416666666666665047591422, 0.00833333333331652721664984,
        0.00138888888889774492207962, 0.000198412698960509205564975,
        2.4801587159235472998791e-5, 2.75572362911928827629423e-6,
        2.75573911234900471893338e-7, 2.51112930892876518610661e-8,
        2.08860621107283687536341e-9};
    const auto pair = [&](int low) {
      return fmadd(
          tag, fill_word(tag, T(coefficients[low + 1])), r,
          fill_word(tag, T(coefficients[low])));
    };
    const auto p0 = pair(0);
    const auto p1 = pair(2);
    const auto p2 = pair(4);
    const auto p3 = pair(6);
    const auto p4 = pair(8);
    const auto p5 = fill_word(tag, T(coefficients[10]));
    const auto r4 = mul(tag, r2, r2);
    const auto r8 = mul(tag, r4, r4);
    const auto q0 = fmadd(tag, p1, r2, p0);
    const auto q1 = fmadd(tag, p3, r2, p2);
    const auto q2 = fmadd(tag, p5, r2, p4);
    const auto q = fmadd(
        tag, q2, r8, fmadd(tag, q1, r4, q0));
    return add(
        tag, fmadd(tag, q, r2, r), fill_word(tag, T(1)));
  } else {
    constexpr const double* k =
        Base == ExpBase::Base2 ? kX86ExpF64PolyB2 : kX86ExpF64PolyB10;
    const auto pair = [&](int low) {
      return fmadd(
          tag, fill_word(tag, T(k[low + 1])), r, fill_word(tag, T(k[low])));
    };
    const auto p0 = pair(1);
    const auto p1 = pair(3);
    const auto p2 = pair(5);
    const auto p3 = pair(7);
    const auto p4 = pair(9);
    const auto r4 = mul(tag, r2, r2);
    const auto r8 = mul(tag, r4, r4);
    const auto q = fmadd(
        tag, p4, r8,
        fmadd(
            tag, fmadd(tag, p3, r2, p2), r4,
            fmadd(tag, p1, r2, p0)));
    return fmadd(
        tag, fmadd(tag, q, r, fill_word(tag, T(k[0]))), r,
        fill_word(tag, T(1)));
  }
}

template <ExpBase Base, Accuracy Tier, bool NegativeOnly, nint_t Index,
          FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_family(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using D = X86ExpDomain<Base, T>;
  constexpr bool strict = Tier == Accuracy::Strict;
  constexpr bool estimate = Tier == Accuracy::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
#if defined(CPU_CAPABILITY_AVX512) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  constexpr bool natural_underflow = true;
#else
  constexpr bool natural_underflow = false;
#endif
  const T overflow = T(D::overflow);
  const T lower = natural_underflow
      ? T(D::zero_sentinel) : T(gradual ? D::zero_limit : D::normal_limit);

  auto xc = max(tag, x, fill_word(tag, lower));
  if constexpr (!NegativeOnly)
    xc = min(tag, xc, fill_word(tag, overflow));

  // Range reduction: n = rint(x*log2(base)), r = x - n*log_base(2).
  NativeWordVec<Tag> qf;
  NativeWordVec<Tag> r;
  if constexpr (Base == ExpBase::Base2) {
    qf = x86_exp_round_nearest<Tag>(xc);
    r = sub(tag, xc, qf);
  } else {
    constexpr double log2_base =
        Base == ExpBase::E ? X86ExpReduction<ExpBase::E>::log2_base
                           : X86ExpReduction<ExpBase::Base10>::log2_base;
    qf = x86_exp_round_nearest<Tag>(
        mul(tag, xc, fill_word(tag, T(log2_base))));
    if constexpr (Base == ExpBase::E) {
      r = fmadd(
          tag, qf,
          fill_word(tag, estimate ? T(-0.693147180559945309417232121458)
                                : std::same_as<T, float16_t>
                                      ? T(-0.693359375)
                                      : std::same_as<T, float32_t>
                                            ? T(-0.693145751953125F)
                                            : T(-0.6931471805596629565116018)),
          xc);
      if constexpr (!estimate) {
        r = fmadd(
            tag, qf,
            fill_word(tag, std::same_as<T, float16_t>
                             ? T(0.000212192535400390625)
                             : std::same_as<T, float32_t>
                                   ? T(-1.428606765330187045e-6F)
                                   : T(-0.28235290563031577122588448175e-12)),
            r);
      }
    } else {
      using R = X86ExpReduction<ExpBase::Base10>;
      const auto hi = std::same_as<T, float16_t> ? T(R::inv_hi_f16)
          : std::same_as<T, float32_t> ? T(R::inv_hi_f32)
                                       : T(R::inv_hi);
      const auto lo = std::same_as<T, float16_t> ? T(R::inv_lo_f16)
          : std::same_as<T, float32_t> ? T(R::inv_lo_f32)
                                       : T(R::inv_lo);
      r = fmadd(tag, qf, fill_word(tag, -hi), xc);
      if constexpr (!estimate)
        r = fmadd(tag, qf, fill_word(tag, -lo), r);
    }
  }
  const auto poly = x86_exp_poly<Base, Tier, Index>(tag, r);
  NativeWordVec<Tag> y;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_castph_si128(_mm_scalef_ph(
          _mm_castsi128_ph(poly.value), _mm_castsi128_ph(qf.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_castph_si256(_mm256_scalef_ph(
          _mm256_castsi256_ph(poly.value), _mm256_castsi256_ph(qf.value)))};
#endif
#if VEC_WIDTH >= 512
    else y = NativeWordVec<Tag>{_mm512_castph_si512(_mm512_scalef_ph(
        _mm512_castsi512_ph(poly.value), _mm512_castsi512_ph(qf.value)))};
#endif
#endif
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_scalef_ps(poly.value, qf.value)};
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_scalef_ps(poly.value, qf.value)};
    else y = NativeWordVec<Tag>{_mm512_scalef_ps(poly.value, qf.value)};
  } else {
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_scalef_pd(poly.value, qf.value)};
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_scalef_pd(poly.value, qf.value)};
    else y = NativeWordVec<Tag>{_mm512_scalef_pd(poly.value, qf.value)};
  }
#else
  constexpr IndexTag<Tag> index_tag{};
  const auto qi = x86_exp_to_index<Tag>(tag, qf);
  const auto half = shr(index_tag, qi, 1);
  const auto rest = sub(index_tag, qi, half);
  const auto bias = fill_word(
      index_tag, std::same_as<T, float32_t> ? 127 : 1023);
  const int shift = std::same_as<T, float32_t> ? 23 : 52;
  if constexpr (NegativeOnly && !gradual) {
    const auto scale = bitcast(
        tag, shl(index_tag, add(index_tag, qi, bias), shift));
    y = mul(tag, poly, scale);
  } else {
    const auto a = bitcast(
        tag, shl(index_tag, add(index_tag, half, bias), shift));
    const auto b = bitcast(
        tag, shl(index_tag, add(index_tag, rest, bias), shift));
    y = mul(tag, mul(tag, poly, a), b);
  }
#endif
  if constexpr (!NegativeOnly) {
    y = blend(
        tag, y, cmpgt(tag, x, fill_word(tag, overflow)),
        fill_word(tag, std::numeric_limits<T>::infinity()));
  }
  if constexpr (!natural_underflow) {
    y = blend(
        tag, y,
        cmplt(tag, x, fill_word(tag, gradual ? T(D::zero_limit)
                                           : T(D::normal_limit))),
        fill_word(tag, T(0)));
  }
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!NegativeOnly) {
    y = blend(
        tag, y, cmpne(tag, x, x), add(tag, x, x));
  }
#endif
  return y;
}

template <ExpBase Base, Accuracy Tier, bool NegativeOnly, nint_t Index,
          FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_low_precision(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t,
      static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
  constexpr Accuracy compute_tier =
      std::same_as<T, bfloat16_t> && Tier == Accuracy::Estimate
          ? Accuracy::Fast : Tier;
  if constexpr (sizeof(Raw) == 16) {
    __m128 low;
    __m128 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256 low;
    __m256 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
#if VEC_WIDTH >= 512
  else {
    __m512 low;
    __m512 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_family<
        Base, compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
}

/* **************************************************************************** */
//                    Exponential word implementation                        //
/* **************************************************************************** */

template <Accuracy A, bool NegativeOnly>
struct X86ExpWordImpl {
  template <nint_t Index, ExpBase Base, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<Base, A, NegativeOnly>, Tag tag, NativeWordVec<Tag> value) {
    if constexpr (
        std::same_as<ElementOf<Tag>, float32_t> ||
        std::same_as<ElementOf<Tag>, float64_t>
#if defined(HAS_AVX512_FP16)
        || std::same_as<ElementOf<Tag>, float16_t>
#endif
        )
      return x86_exp_family<Base, A, NegativeOnly, Index>(tag, value);
    else
      return x86_exp_low_precision<Base, A, NegativeOnly, Index>(tag, value);
  }

  template <nint_t Index, ExpBase Base, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<Base, A, NegativeOnly> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto zero = fill_word(tag, ElementOf<Tag>{});
    const auto safe = blend(
        tag, zero, mask, value);
    return blend(
        tag, inactive, mask, call<Index>(op, tag, safe));
  }
};

template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct NativeWordImpl<X86Backend, ExpOp<Base, A, NegativeOnly>>
    : X86ExpWordImpl<A, NegativeOnly> {};
} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_EXP_H
