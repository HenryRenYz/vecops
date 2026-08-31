#ifndef VECOPS_VEC_DETAILS_SVE_MATH_LOG_H
#define VECOPS_VEC_DETAILS_SVE_MATH_LOG_H

/**
 * @file Log.h
 * @brief SVE backend implementations for the log family (log, log2, log10).
 *
 * The f64 Strict kernels and the special-case structure are derived from
 * the Arm optimized-routines project
 * (https://github.com/ARM-software/optimized-routines), files
 * math/aarch64/sve/{log,log2,log10}.c (dual-licensed under "MIT OR
 * Apache-2.0 WITH LLVM-exception"); the minimax coefficients and the
 * 128-entry invc/logc tables live in the shared vec/details/Math.h.
 *
 * Shared design, mirroring the exp/reciprocal families: a branch-free
 * inline core covers the normal domain and one NOINLINE special routine
 * owns every tail (subnormals, zero, negatives, +inf, NaN). The special
 * routine renormalizes subnormal lanes by an exact power of two (2^23 for
 * f32, 2^10 for f16, 2^52 for f64) and reruns the exact-tier core with
 * the logarithmic correction folded into the extracted exponent as an
 * integer bias through a merging predicate (the f64 table kernel keeps
 * the upstream masked-subtract correction instead), so special lanes
 * return tier-grade results. Zero/inf/NaN lanes select their constants
 * through the same masked add, which also propagates NaN from negative
 * lanes.
 *
 * Reduction: bits(x) - off splits x into 2^k * z with z inside a fixed
 * window (f64 table kernels: the upstream invc/logc gather window; the
 * table-free f32/f16 kernels: the symmetric [1/sqrt2, sqrt2) window,
 * whose r = z - 1 (exact by Sterbenz) keeps |log_b(z)| <= |log_b(x)| on
 * every k band, bounding the assembly roundings; f32 Fast/Estimate keep
 * the [2/3, 4/3) window of their Taylor kernels). The f64 gather index is
 * masked to the table size, so any input bit pattern gathers in range.
 *
 * Tiers: f64 Strict is the upstream table algorithm (measured
 * 2.64/2.58/2.46 ULP for log/log2/log10) and Fast/Estimate drop to six /
 * three Taylor terms. f32 Strict is a native table-free kernel: a
 * degree-13 minimax of (log_b(1+r) - alpha*r)/r^2 evaluated with Estrin
 * FMAs and assembled Cody-Waite style (kflo = kf*lo in a tiny binade,
 * pf = r^2*Q + kflo, t = alpha*r + pf, y = kf*hi + t), where hi is
 * quantized to 15 significant mantissa bits so kf*hi stays exact inside
 * the final FMA (|kf| <= 150 including the subnormal bias); measured 1
 * ULP over dense full-mantissa sweeps of the near-1 exponents plus
 * all-exponent stratified scans and the subnormal path (coefficients and
 * the bit-exact simulation live in scripts/log-native-design/). Fast
 * keeps six Taylor terms (relative error ~2^-13.9), Estimate three
 * (~2^-10.3).
 *
 * f16 Fast and Estimate run the same skeleton natively in f16 arithmetic
 * (degree-6 / degree-2 Q with per-base hi/lo/alpha constants),
 * exhaustively validated over all 65536 bit patterns: max relative error
 * 8.7e-4 / 1.6e-3 against the 2^-10 / 2^-7 contracts. f16 Strict and all
 * of bf16 widen through the f32 kernels instead: the 1-ULP class is not
 * reachable in pure f16 arithmetic (each FMA rounds at 2^-11, which the
 * k*log_b(2) + log_b(z) cancellation band cannot absorb), while the
 * widened native f32 Strict kernel delivers it after one narrowing
 * rounding; every f16/bf16 input is normal in f32 and log outputs are
 * never subnormal, so the f32 special routine owns all narrow-format
 * tails exactly.
 */

#include <arm_sve.h>
#include <cstdint>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Math.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//    f64 families and kernels                                                 //
/* **************************************************************************** */


/**
 * f64 table kernel: log(x) = (logc + r*lead + k*kd_scale) +
 * r^2*(c0 + r*c1 + r^2*(c2 + r*c3 + r^2*c4)), r = invc*z - 1. Gather
 * indices are always in range by construction, so inactive or garbage
 * lanes can never form wild addresses.
 */
template <const LogF64Family& C>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64_table_core(
    svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const auto tmp = svsub_n_u64_x(pg, ix, 0x3fe6900900000000ull);
  const auto i = svand_n_u64_x(pg, svlsr_n_u64_x(pg, tmp, 44), 0xfeull);
  const auto k = svcvt_f64_s64_x(
      pg, svasr_n_s64_x(pg, svreinterpret_s64_u64(tmp), 52));
  const auto z = svreinterpret_f64_u64(
      svsub_u64_x(pg, ix, svand_n_u64_x(pg, tmp, 0xfffull << 52)));
  const auto invc = svld1_gather_u64index_f64(pg, &C.table[0].invc, i);
  const auto logc = svld1_gather_u64index_f64(pg, &C.table[0].logc, i);
  const auto r = svmad_n_f64_x(pg, invc, z, -1.0);
  const auto w = svmla_n_f64_x(pg, logc, r, C.lead);
  const auto hi = svmla_n_f64_x(pg, w, k, C.kd_scale);
  const auto r2 = svmul_f64_x(pg, r, r);
  auto y = svmla_n_f64_x(pg, svdup_n_f64(C.c2), r, C.c3);
  y = svmla_n_f64_x(pg, y, r2, C.c4);
  const auto p = svmla_n_f64_x(pg, svdup_n_f64(C.c0), r, C.c1);
  y = svmla_f64_x(pg, p, r2, y);
  return svmla_f64_x(pg, hi, r2, y);
}

/**
 * f64 table-free kernel for the Fast and Estimate tiers: r = z - 1 on the
 * [2/3, 4/3) window (exact by Sterbenz), six Taylor terms for Fast and
 * three for Estimate, and one final base scale for non-e bases.
 */
template <const LogF64PolyFamily& C, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64_poly_core(
    svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const auto tmp = svsub_n_u64_x(pg, ix, 0x3fe5555555555555ull);
  const auto n = svcvt_f64_s64_x(
      pg, svasr_n_s64_x(pg, svreinterpret_s64_u64(tmp), 52));
  const auto z = svreinterpret_f64_u64(
      svsub_u64_x(pg, ix, svand_n_u64_x(pg, tmp, 0xfffull << 52)));
  const auto r = svsub_n_f64_x(pg, z, 1.0);
  const auto r2 = svmul_f64_x(pg, r, r);
  svfloat64_t y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = svmla_n_f64_x(pg, svdup_n_f64(C.c0), r, C.c1);
    const auto b = svmla_n_f64_x(pg, svdup_n_f64(C.c2), r, C.c3);
    const auto d = svmla_n_f64_x(pg, svdup_n_f64(C.c4), r, C.c5);
    y = svmla_f64_x(pg, b, r2, d);
    y = svmla_f64_x(pg, a, r2, y);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = svmla_n_f64_x(pg, svdup_n_f64(C.c1), r, C.c2);
    y = svmla_f64_x(pg, svdup_n_f64(C.c0), r, y);
  }
  const auto hi = svmla_n_f64_x(pg, r, n, C.ln2);
  auto result = svmla_f64_x(pg, hi, r2, y);
  if constexpr (C.scale != 1.0)
    result = svmul_n_f64_x(pg, result, C.scale);
  return result;
}

/** Tails: subnormals renormalized by 2^52, constants for 0/inf/NaN. */
template <const LogF64Family& C>
VECOPS_NOINLINE inline svfloat64_t sve_log_f64_special(
    svbool_t pg, svbool_t special, svfloat64_t x) {
  const auto is_sub = svcmpgt_n_f64(pg, x, 0.0);
  const auto is_minf = svcmpeq_n_f64(pg, x, 0.0);
  const auto is_pinf =
      svcmpeq_n_f64(pg, x, std::numeric_limits<double>::infinity());
  const auto scaled = svmul_m(special, x, 0x1p52);
  auto y = sve_log_f64_table_core<C>(scaled, svptrue_b64());
  auto corr = svsel_f64(is_sub, svdup_n_f64(C.sub), svdup_n_f64(
                         std::numeric_limits<double>::quiet_NaN()));
  corr = svsel_f64(
      is_minf, svdup_n_f64(-std::numeric_limits<double>::infinity()),
      corr);
  corr = svsel_f64(
      is_pinf, svdup_n_f64(std::numeric_limits<double>::infinity()),
      corr);
  y = svadd_m(special, y, corr);
  return y;
}

template <const LogF64Family& C, const LogF64PolyFamily& P,
          Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64(svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const svbool_t special = svcmpge_n_u64(
      pg, svsub_n_u64_x(pg, ix, 0x0010000000000000ull), 0x7fe0000000000000ull);
  if (svptest_any(special, special))
    return sve_log_f64_special<C>(pg, special, x);
  if constexpr (Tier == Accuracy::Strict)
    return sve_log_f64_table_core<C>(x, pg);
  else
    return sve_log_f64_poly_core<P, Tier>(x, pg);
}


/* **************************************************************************** */
//    f32 families and kernels                                                 //
/* **************************************************************************** */


/**
 * f32 table-free kernel on the [2/3, 4/3) window; r = z - 1 is exact by
 * Sterbenz. Fast keeps six Taylor terms, Estimate three. KBias subtracts
 * from the extracted exponent on the special path (subnormal
 * renormalization by 2^23) through a merging predicate so normal lanes
 * are untouched; pass 0 with any mask on the normal path.
 */
template <const LogF32FastFamily& C, Accuracy Tier, int KBias>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32_poly_core(
    svfloat32_t x, svbool_t pg, svbool_t bias_lanes) {
  constexpr std::uint32_t kOff = 0x3f2aaaabu;
  const auto u_off = svsub_n_u32_x(pg, svreinterpret_u32_f32(x), kOff);
  const auto k_shift = svasr_n_s32_x(pg, svreinterpret_s32_u32(u_off), 23);
  const svint32_t k_int = [&]() {
    if constexpr (KBias == 0)
      return k_shift;
    else
      return svsub_m(bias_lanes, k_shift, KBias);
  }();
  const auto n = svcvt_f32_s32_x(pg, k_int);
  const auto z = svreinterpret_f32_u32(
      svadd_n_u32_x(pg, svand_n_u32_x(pg, u_off, 0x007fffffu), kOff));
  const auto r = svsub_n_f32_x(pg, z, 1.0f);
  const auto r2 = svmul_f32_x(pg, r, r);
  svfloat32_t y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = svmla_n_f32_x(pg, svdup_n_f32(C.c0), r, C.c1);
    const auto b = svmla_n_f32_x(pg, svdup_n_f32(C.c2), r, C.c3);
    const auto d = svmla_n_f32_x(pg, svdup_n_f32(C.c4), r, C.c5);
    y = svmla_f32_x(pg, b, r2, d);
    y = svmla_f32_x(pg, a, r2, y);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = svmla_n_f32_x(pg, svdup_n_f32(C.c1), r, C.c2);
    y = svmla_f32_x(pg, svdup_n_f32(C.c0), r, y);
  }
  const auto hi = svmla_n_f32_x(pg, r, n, C.ln2);
  auto result = svmla_f32_x(pg, hi, r2, y);
  if constexpr (C.scale != 1.0f)
    result = svmul_n_f32_x(pg, result, C.scale);
  return result;
}

/**
 * Native f32 Strict kernel on the [1/sqrt2, sqrt2) window. r = z - 1 is
 * exact by Sterbenz, Q is the Estrin evaluation of the degree-13 minimax
 * in C.d, and the tail keeps every rounding at or below the result
 * binade: kflo = kf*lo and pf = kflo + r^2*Q round in tiny binades,
 * t = alpha*r + pf rounds at |t| <= |log_b(z)| <= |y| on every k band,
 * and the final FMA rounds once at the result scale with kf*hi exact (hi
 * carries nine trailing zero mantissa bits). KBias serves the special
 * path exactly as in sve_log_f32_poly_core.
 */
template <const LogF32StrictFamily& C, int KBias>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32_strict_core(
    svfloat32_t x, svbool_t pg, svbool_t bias_lanes) {
  constexpr std::uint32_t kOff = 0x3f3504f3u; // fl32(1/sqrt2)
  const auto u_off = svsub_n_u32_x(pg, svreinterpret_u32_f32(x), kOff);
  const auto k_shift = svasr_n_s32_x(pg, svreinterpret_s32_u32(u_off), 23);
  const svint32_t k_int = [&]() {
    if constexpr (KBias == 0)
      return k_shift;
    else
      return svsub_m(bias_lanes, k_shift, KBias);
  }();
  const auto kf = svcvt_f32_s32_x(pg, k_int);
  const auto z = svreinterpret_f32_u32(
      svadd_n_u32_x(pg, svand_n_u32_x(pg, u_off, 0x007fffffu), kOff));
  const auto r = svsub_n_f32_x(pg, z, 1.0f);
  const auto r2 = svmul_f32_x(pg, r, r);
  const auto qa = svmla_n_f32_x(pg, svdup_n_f32(C.d[0]), r, C.d[1]);
  const auto qb = svmla_n_f32_x(pg, svdup_n_f32(C.d[2]), r, C.d[3]);
  const auto qc = svmla_n_f32_x(pg, svdup_n_f32(C.d[4]), r, C.d[5]);
  const auto qd = svmla_n_f32_x(pg, svdup_n_f32(C.d[6]), r, C.d[7]);
  const auto qe = svmla_n_f32_x(pg, svdup_n_f32(C.d[8]), r, C.d[9]);
  const auto qf = svmla_n_f32_x(pg, svdup_n_f32(C.d[10]), r, C.d[11]);
  const auto qg = svmla_n_f32_x(pg, svdup_n_f32(C.d[12]), r, C.d[13]);
  const auto qab = svmla_f32_x(pg, qa, r2, qb);
  const auto qcd = svmla_f32_x(pg, qc, r2, qd);
  const auto qef = svmla_f32_x(pg, qe, r2, qf);
  const auto r4 = svmul_f32_x(pg, r2, r2);
  const auto qabcd = svmla_f32_x(pg, qab, r4, qcd);
  const auto qefg = svmla_f32_x(pg, qef, r4, qg);
  const auto r8 = svmul_f32_x(pg, r4, r4);
  const auto q = svmla_f32_x(pg, qabcd, r8, qefg);
  const auto kflo = svmul_n_f32_x(pg, kf, C.lo);
  const auto pf = svmla_f32_x(pg, kflo, r2, q);
  const auto t = svmla_n_f32_x(pg, pf, r, C.alpha);
  return svmla_n_f32_x(pg, t, kf, C.hi);
}

/**
 * f32 special routine: renormalizes subnormal lanes by 2^23 (merging
 * predicate, so normal lanes recompute unchanged), reruns the exact-tier
 * core with the -23 exponent bias, and selects the IEEE constants for
 * zero/inf/NaN lanes through the masked add. Subnormal lanes need no
 * additive correction -- the bias already folded it -- so their select
 * value is +0.
 */
template <LogBase Base, Accuracy Tier>
VECOPS_NOINLINE inline svfloat32_t sve_log_f32_special(
    svbool_t pg, svbool_t special, svfloat32_t x) {
  const auto is_sub = svcmpgt_n_f32(pg, x, 0.0f);
  const auto is_minf = svcmpeq_n_f32(pg, x, 0.0f);
  const auto is_pinf =
      svcmpeq_n_f32(pg, x, std::numeric_limits<float>::infinity());
  const auto scaled = svmul_m(special, x, 0x1p23f);
  const svfloat32_t y = [&]() {
    if constexpr (Tier == Accuracy::Strict)
      return sve_log_f32_strict_core<log_f32_strict_family<Base>(), 23>(
          scaled, svptrue_b32(), special);
    else
      return sve_log_f32_poly_core<log_f32_fast_family<Base>(), Tier, 23>(
          scaled, svptrue_b32(), special);
  }();
  auto corr = svsel_f32(
      is_sub, svdup_n_f32(0.0f),
      svdup_n_f32(std::numeric_limits<float>::quiet_NaN()));
  corr = svsel_f32(
      is_minf, svdup_n_f32(-std::numeric_limits<float>::infinity()), corr);
  corr = svsel_f32(
      is_pinf, svdup_n_f32(std::numeric_limits<float>::infinity()), corr);
  return svadd_m(special, y, corr);
}

template <LogBase Base, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32(svfloat32_t x, svbool_t pg) {
  const auto u = svreinterpret_u32_f32(x);
  const svbool_t special = svcmpge_n_u32(
      pg, svsub_n_u32_x(pg, u, 0x00800000u), 0x7f000000u);
  if (svptest_any(special, special))
    return sve_log_f32_special<Base, Tier>(pg, special, x);
  if constexpr (Tier == Accuracy::Strict)
    return sve_log_f32_strict_core<log_f32_strict_family<Base>(), 0>(
        x, pg, special);
  else
    return sve_log_f32_poly_core<log_f32_fast_family<Base>(), Tier, 0>(
        x, pg, special);
}

/**
 * Native f16 kernel for the Fast and Estimate tiers, same skeleton as the
 * f32 Strict core but in pure f16 arithmetic: reduction on the u16 bits of
 * the [0.70703125, 1.4140625) window, a tier-degree Q, and the
 * kflo/pf/t/y tail with kf*hi exact (hi keeps enough trailing zero
 * mantissa bits for |kf| <= 25). KBias serves the special path exactly as
 * in the f32 cores.
 */
template <const LogF16Family& C, Accuracy Tier, int KBias>
VECOPS_ALWAYS_INLINE svfloat16_t sve_log_f16_core(
    svfloat16_t raw, svbool_t pg, svbool_t bias_lanes) {
  constexpr std::uint16_t kOff = 0x39a8u; // f16(0.70703125)
  const auto u_off = svsub_n_u16_x(pg, svreinterpret_u16_f16(raw), kOff);
  const auto k_shift = svasr_n_s16_x(pg, svreinterpret_s16_u16(u_off), 10);
  const svint16_t k_int = [&]() {
    if constexpr (KBias == 0)
      return k_shift;
    else
      return svsub_m(bias_lanes, k_shift, KBias);
  }();
  const auto kf = svcvt_f16_s16_x(pg, k_int);
  const auto z = svreinterpret_f16_u16(
      svadd_n_u16_x(pg, svand_n_u16_x(pg, u_off, 0x03ffu), kOff));
  const auto r = svsub_n_f16_x(pg, z, float16_t(1.0f));
  const auto r2 = svmul_f16_x(pg, r, r);
  svfloat16_t q;
  if constexpr (Tier == Accuracy::Fast) {
    const auto qa =
        svmla_n_f16_x(pg, svdup_n_f16(float16_t(C.d_fast[0])), r,
                      float16_t(C.d_fast[1]));
    const auto qb =
        svmla_n_f16_x(pg, svdup_n_f16(float16_t(C.d_fast[2])), r,
                      float16_t(C.d_fast[3]));
    const auto qc =
        svmla_n_f16_x(pg, svdup_n_f16(float16_t(C.d_fast[4])), r,
                      float16_t(C.d_fast[5]));
    const auto qab = svmla_f16_x(pg, qa, r2, qb);
    const auto qcd = svmla_n_f16_x(pg, qc, r2, float16_t(C.d_fast[6]));
    const auto r4 = svmul_f16_x(pg, r2, r2);
    q = svmla_f16_x(pg, qab, r4, qcd);
  } else {
    const auto inner =
        svmla_n_f16_x(pg, svdup_n_f16(float16_t(C.d_est[1])), r,
                      float16_t(C.d_est[2]));
    q = svmla_f16_x(pg, svdup_n_f16(float16_t(C.d_est[0])), r, inner);
  }
  const auto kflo = svmul_n_f16_x(pg, kf, float16_t(C.lo));
  const auto pf = svmla_f16_x(pg, kflo, r2, q);
  const auto t = svmla_n_f16_x(pg, pf, r, float16_t(C.alpha));
  return svmla_n_f16_x(pg, t, kf, float16_t(C.hi));
}

/**
 * f16 special routine: subnormals renormalized by 2^10 with the -10
 * exponent bias folded into the core (merging predicate), constants for
 * zero/inf/NaN through the masked add. Mirrors sve_log_f32_special.
 */
template <LogBase Base, Accuracy Tier>
VECOPS_NOINLINE inline svfloat16_t sve_log_f16_special(
    svbool_t pg, svbool_t special, svfloat16_t raw) {
  const auto is_sub = svcmpgt_n_f16(pg, raw, float16_t(0.0f));
  const auto is_minf = svcmpeq_n_f16(pg, raw, float16_t(0.0f));
  const auto is_pinf =
      svcmpeq_n_f16(pg, raw, std::numeric_limits<float16_t>::infinity());
  const auto scaled = svmul_m(special, raw, float16_t(1024.0f));
  const auto y =
      sve_log_f16_core<log_f16_family<Base>(), Tier, 10>(
          scaled, svptrue_b16(), special);
  auto corr = svsel_f16(
      is_sub, svdup_n_f16(float16_t(0.0f)),
      svdup_n_f16(std::numeric_limits<float16_t>::quiet_NaN()));
  corr = svsel_f16(
      is_minf,
      svdup_n_f16(float16_t(-std::numeric_limits<float16_t>::infinity())),
      corr);
  corr = svsel_f16(
      is_pinf, svdup_n_f16(std::numeric_limits<float16_t>::infinity()),
      corr);
  return svadd_m(special, y, corr);
}

/**
 * f16 tier entry: Fast and Estimate run the native core; Strict is
 * dispatched by sve_log_dispatch through the widening path below.
 */
template <LogBase Base, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_log_f16(svfloat16_t raw, svbool_t pg) {
  const auto u = svreinterpret_u16_f16(raw);
  // Subnormal-or-worse lanes: (u - min_normal_bits) >= (inf_bits -
  // min_normal_bits) wraps exactly for every tail class.
  const svbool_t special = svcmpge_n_u16(
      pg, svsub_n_u16_x(pg, u, 0x0400u), 0x7800u);
  if (svptest_any(special, special))
    return sve_log_f16_special<Base, Tier>(pg, special, raw);
  return sve_log_f16_core<log_f16_family<Base>(), Tier, 0>(
      raw, pg, special);
}

/**
 * Widens an f16 word to two f32 words for the Strict tier, evaluates the
 * native f32 Strict kernel on both halves, and narrows back. Pure f16
 * arithmetic cannot hold the 1-ULP contract in the k*log_b(2) + log_b(z)
 * cancellation band, while this detour rounds once into f16; every f16
 * value (including subnormals) is normal in f32 and log never produces
 * subnormal outputs, so the f32 special routine owns all f16 tails
 * exactly.
 */
template <LogBase Base, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_log_f16_via_f32(svfloat16_t raw) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd_value = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd_value, odd_value)));
#endif
  const auto result_low = sve_log_f32<Base, Tier>(low, svptrue_b32());
  const auto result_high = sve_log_f32<Base, Tier>(high, svptrue_b32());
  const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), result_low);
#if defined(__ARM_FEATURE_SVE2)
  return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), result_high);
#else
  const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), result_high);
  return svtrn1_f16(packed_low, packed_high);
#endif
}

template <LogBase Base, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_log_dispatch(
    Tag, NativeWordVec<Tag> value, svbool_t pg) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(sve_log_f32<Base, Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(
        sve_log_f64<log_f64_family<Base>(),
                    log_f64_poly_family<Base>(), Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (Tier == Accuracy::Strict)
      return sve_basic_wrap_word<Tag>(
          sve_log_f16_via_f32<Base, Tier>(raw));
    else
      return sve_basic_wrap_word<Tag>(sve_log_f16<Base, Tier>(raw, pg));
  } else {
    // bf16 widens exactly to f32 and the f32 exponent range is shared, so
    // the f32 pipeline owns every bf16 tail directly.
    const auto low = sve_bf16_to_f32_lo(raw);
    const auto high = sve_bf16_to_f32_hi(raw);
    return sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        sve_log_f32<Base, Tier>(low, svptrue_b32()),
        sve_log_f32<Base, Tier>(high, svptrue_b32())));
  }
}

/**
 * Shared word implementation for the log family. Wide element formats are
 * elementwise, so an explicit mask only guards the special-trigger compare
 * and the final blend. f16 Strict and bf16 widen through whole-word f32
 * conversions; f16 Fast/Estimate run native and only need inactive lanes
 * sanitized first so the special-tail tests cannot observe stale values.
 */
template <LogBase Base, Accuracy A>
struct SVELogWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A>, Tag tag, NativeWordVec<Tag> value) {
    return sve_log_dispatch<Base, A>(
        tag, value, sve_full_predicate<ElementOf<Tag>>());
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    if constexpr (sizeof(ElementOf<Tag>) >= 4) {
      const auto computed = sve_log_dispatch<Base, A>(
          tag, value, sve_basic_raw_word(mask));
      return blend(tag, inactive, mask, computed);
    } else {
      return masked_unary_word(
          tag, mask, inactive, value, [&](NativeWordVec<Tag> safe) {
            return sve_log_dispatch<Base, A>(
                tag, safe, sve_full_predicate<ElementOf<Tag>>());
          });
    }
  }
};

template <LogBase Base, Accuracy A>
struct NativeWordImpl<SVEBackend, LogOp<Base, A>> : SVELogWordImpl<Base, A> {
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_LOG_H
