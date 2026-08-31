#ifndef VECOPS_VEC_DETAILS_SVE_MATH_EXP_H
#define VECOPS_VEC_DETAILS_SVE_MATH_EXP_H

/**
 * @file Exp.h
 * @brief SVE backend implementations for the exp family (exp, exp2, exp10).
 *
 * The implementation structure and all f32/f64 minimax coefficients are
 * derived from the Arm optimized-routines project
 * (https://github.com/ARM-software/optimized-routines), files
 * math/aarch64/sve/{expf,exp2f,exp10f,exp,exp2,exp10}.c, which are
 * dual-licensed under "MIT OR Apache-2.0 WITH LLVM-exception".
 *
 * Design, shared by every base and width: a branch-free inline core covers
 * the FEXPA-safe normal domain, and one NOINLINE special-case routine owns
 * every tail. The special routine repeats the core with the FEXPA exponent
 * biased by a fixed offset (23 for f32, 53 for f64) and undoes it with an
 * exact power-of-two scaling, which keeps the intermediate result normal
 * for any input and produces correctly rounded gradual underflow when
 * subnormal preservation is enabled. NaN inputs propagate naturally
 * through the core arithmetic; the special predicate never matches them.
 *
 * All three bases share one reduction skeleton and differ only in
 * constants, so the f32/f64 cores are a single template over a per-base
 * constant set: base^x = 2^n * (1 + poly(r)) with n = round32/64(x *
 * log2(base)) produced by the FEXPA shift trick and r = x - n/log2(base).
 * exp2 degenerates to exact integer-step reduction (z = x + shift,
 * r = x - n); exp keeps the unit-lead polynomial form r + r^2/2 used by the
 * e-base cores upstream. The f16 paths build the FEXPA exponent code from
 * a rounded integer step count instead (no float-precision shift trick
 * survives f16 rounding) and pin their saturation edges with exact
 * constants; see SVEExpF16Family.
 */

#include <arm_sve.h>
#include <cstdint>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {

/** True for the tiers that must reproduce subnormals when enabled. */
template <Accuracy Tier>
constexpr bool sve_exp_gradual() {
#ifdef VECOPS_PRESERVE_SUBNORMALS
  return Tier == Accuracy::Strict;
#else
  return false;
#endif
}

/* **************************************************************************** */
//    exp family — per-base constants and shared cores (f32)                  //
/* **************************************************************************** */

/**
 * Reduction constants for one base. The first four floats form the LD1RW
 * quad serving the hot path: [red_hi, red_lo, lead, poly_c1]. unit_step
 * marks base 2, where z = x + shift and r = x - n hold exactly; unit_lead
 * marks base e, whose polynomial starts at r itself.
 */
struct alignas(16) SVEExpF32Family {
  float red_hi, red_lo, lead, poly_c1;
  float log2_base;
  float shift, special_bound, inf_bound, zero_bound;
  bool unit_step;
  bool unit_lead;
};

inline constexpr SVEExpF32Family kSVEExpF32E{
    0x1.62e4p-1f, 0x1.7f7d1cp-20f, 1.0f, 0.5f,
    0x1.715476p+0f, 196735.0f,
    87.3365447505531f, 88.72283905206835f, -103.972084045410f,
    false, true};

inline constexpr SVEExpF32Family kSVEExpF32B2{
    1.0f, 0.0f, 0x1.62e485p-1f, 0x1.ebfbe0p-3f,
    1.0f, 196735.0f,
    126.0f, 128.0f, -150.0f,
    true, false};

inline constexpr SVEExpF32Family kSVEExpF32B10{
    0x1.344136p-2f, -0x1.ec10cp-27f, 0x1.26bb62p1f, 0x1.53524cp1f,
    0x1.a934fp+1f, 196735.0f,
    0x1.2f702p+5f, 0x1.34ccccccccccdp+5f, -0x1.68ccccccccccdp+5f,
    false, false};

template <ExpBase Base>
consteval const SVEExpF32Family& sve_exp_f32_family() {
  if constexpr (Base == ExpBase::E) return kSVEExpF32E;
  else if constexpr (Base == ExpBase::Base2) return kSVEExpF32B2;
  else return kSVEExpF32B10;
}

/**
 * base^xc * 2^-offset by reduced-argument evaluation: the FEXPA shift trick
 * turns x*log2(base) into a table index plus exponent, r = x - n/base in two
 * parts, and base^r - 1 ~= r*(lead + r*poly_c1). The Estimate tier drops the
 * r^2 term and the red_lo correction. offset is subtracted from the FEXPA
 * input in exponent units; pass 0 and Adjusted = false on the normal path.
 */
template <const SVEExpF32Family& C, Accuracy Tier, bool Adjusted>
VECOPS_ALWAYS_INLINE svfloat32_t sve_exp_f32_core(
    svfloat32_t xc, svbool_t pg, svfloat32_t offset) {
  constexpr bool estimate = Tier == Accuracy::Estimate;
  // Constant loads must run unpredicated: LD1RW zeroes inactive lanes, which
  // would corrupt the reduction constants under a partial mask.
  const auto lanes = svld1rq_f32(svptrue_b32(), &C.red_hi);
  const auto z = [&]() -> svfloat32_t {
    if constexpr (C.unit_step) return svadd_n_f32_x(pg, xc, C.shift);
    else
      return svmla_n_f32_x(pg, svdup_n_f32(C.shift), xc, C.log2_base);
  }();
  const auto n = svsub_n_f32_x(pg, z, C.shift);
  svfloat32_t r;
  if constexpr (C.unit_step) {
    r = svsub_f32_x(pg, xc, n);
  } else {
    r = svmls_lane_f32(xc, n, lanes, 0);
    if constexpr (!estimate) r = svmls_lane_f32(r, n, lanes, 1);
  }
  const auto biased = [&]() -> svfloat32_t {
    if constexpr (Adjusted) return svsub_f32_x(pg, z, offset);
    else return z;
  }();
  const auto scale = svexpa_f32(svreinterpret_u32_f32(biased));
  const auto poly = [&]() -> svfloat32_t {
    if constexpr (estimate) {
      if constexpr (C.unit_lead) return r;
      else return svmul_n_f32_x(pg, r, C.lead);
    } else if constexpr (C.unit_lead)
      return svmla_n_f32_x(pg, r, svmul_f32_x(pg, r, r), C.poly_c1);
    else
      return svmul_f32_x(
          pg, svmla_n_f32_x(pg, svdup_n_f32(C.lead), r, C.poly_c1), r);
  }();
  return svmla_f32_x(pg, scale, scale, poly);
}

/**
 * Tails for |x| beyond the FEXPA-safe window: overflow, underflow, and
 * gradual results. Biases the FEXPA exponent by +/−23 towards the interior
 * of the normal range and rescales by the same exact power of two
 * afterwards, so the intermediate stays normal however extreme the input is.
 */
template <const SVEExpF32Family& C, Accuracy Tier>
VECOPS_NOINLINE inline svfloat32_t sve_exp_f32_special(
    svbool_t pg, svfloat32_t x) {
  constexpr float flush_bound =
      sve_exp_gradual<Tier>() ? C.zero_bound : -C.special_bound;
  const auto full = svptrue_b32();
  const auto is_negative = svcmplt_n_f32(pg, x, 0.0f);
  const auto offset =
      svsel_f32(is_negative, svdup_n_f32(-23.0f), svdup_n_f32(23.0f));
  const auto adjust =
      svsel_s32(is_negative, svdup_n_s32(-23), svdup_n_s32(23));
  // NaN lanes must survive the clamp, so use the propagating FMIN/FMAX.
  const auto xc = svmin_n_f32_x(
      full, svmax_n_f32_x(full, x, C.zero_bound), C.inf_bound);
  auto y = sve_exp_f32_core<C, Accuracy::Strict, true>(xc, full, offset);
  y = svscale_f32_x(full, y, adjust);
  y = svsel_f32(
      svcmpgt_n_f32(pg, x, C.inf_bound),
      svdup_n_f32(std::numeric_limits<float>::infinity()), y);
  y = svsel_f32(svcmplt_n_f32(pg, x, flush_bound), svdup_n_f32(0.0f), y);
  return y;
}

template <const SVEExpF32Family& C, Accuracy Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat32_t sve_exp_f32(svfloat32_t x, svbool_t pg) {
  const svbool_t special = NegativeOnly
      ? svcmplt_n_f32(pg, x, -C.special_bound)
      : svacgt_n_f32(pg, x, C.special_bound);
  if (svptest_any(special, special))
    return sve_exp_f32_special<C, Tier>(pg, x);
  const svfloat32_t zero = svdup_n_f32(0.0f);
  return sve_exp_f32_core<C, Tier, false>(x, pg, zero);
}

/* **************************************************************************** */
//    exp family — per-base constants and shared cores (f64)                  //
/* **************************************************************************** */

/** f64 counterpart of SVEExpF32Family; the poly is degree 5 in r. */
struct alignas(16) SVEExpF64Family {
  double red_hi, red_lo;
  double lead;
  double a, b, c, d;
  double log2_base, shift, special_bound, inf_bound, zero_bound;
  bool unit_step;
  bool unit_lead;
};

inline constexpr SVEExpF64Family kSVEExpF64E{
    0x1.62e42fefa3800p-1, 0x1.ef35793c76730p-45,
    1.0,
    0x1.fffffffffdbcdp-2, 0x1.555555555444cp-3, 0x1.555573c6a9f7dp-5,
    0x1.1111266d28935p-7,
    0x1.71547652b82fep+0, 0x1.800000000ffc0p+46,
    708.3964185322641, 709.782712893384, -745.1332191019411,
    false, true};

inline constexpr SVEExpF64Family kSVEExpF64B2{
    1.0, 0.0,
    0x1.62e42fefa39efp-1,
    0x1.ebfbdff82a31bp-3, 0x1.c6b08d706c8a5p-5, 0x1.3b2ad2ff7d2f3p-7,
    0x1.5d8761184beb3p-10,
    1.0, 0x1.800000000ffc0p+46,
    0x1.ff01p+9, 1024.0, -0x1.0d008p+10,
    true, false};

inline constexpr SVEExpF64Family kSVEExpF64B10{
    0x1.34413509f79ffp-2, -0x1.9dc1da994fd21p-59,
    0x1.26bb1bbb55516p1,
    0x1.53524c73cd32ap1, 0x1.0470591daeafbp1, 0x1.2bd77b1361ef6p0,
    0x1.142b5d54e9621p-1,
    0x1.a934f0979a371p+1, 0x1.800000000ffc0p+46,
    0x1.33a7ae900b507p+8, 0x1.35p+8, -0x1.439b746e36b53p+8,
    false, false};

template <ExpBase Base>
consteval const SVEExpF64Family& sve_exp_f64_family() {
  if constexpr (Base == ExpBase::E) return kSVEExpF64E;
  else if constexpr (Base == ExpBase::Base2) return kSVEExpF64B2;
  else return kSVEExpF64B10;
}

/**
 * base^xc * 2^-offset, structured like the f32 core. The degree-5 minimax
 * polynomial (coefficients from the Arm optimized-routines SVE sources)
 * covers the Strict and Fast tiers; Estimate keeps only the linear term and
 * a single-part reduction.
 */
template <const SVEExpF64Family& C, Accuracy Tier, bool Adjusted>
VECOPS_ALWAYS_INLINE svfloat64_t sve_exp_f64_core(
    svfloat64_t xc, svbool_t pg, svfloat64_t offset) {
  constexpr bool estimate = Tier == Accuracy::Estimate;
  const auto z = [&]() -> svfloat64_t {
    if constexpr (C.unit_step) return svadd_n_f64_x(pg, xc, C.shift);
    else
      return svmla_n_f64_x(pg, svdup_n_f64(C.shift), xc, C.log2_base);
  }();
  const auto n = svsub_n_f64_x(pg, z, C.shift);
  svfloat64_t r;
  if constexpr (C.unit_step) {
    r = svsub_f64_x(pg, xc, n);
  } else {
    const auto ln = svld1rq_f64(svptrue_b64(), &C.red_hi);
    r = svmls_lane_f64(xc, n, ln, 0);
    if constexpr (!estimate) r = svmls_lane_f64(r, n, ln, 1);
  }
  const auto biased = [&]() -> svfloat64_t {
    if constexpr (Adjusted) return svsub_f64_x(pg, z, offset);
    else return z;
  }();
  const auto scale = svexpa_f64(svreinterpret_u64_f64(biased));
  svfloat64_t poly;
  if constexpr (estimate) {
    if constexpr (C.unit_lead) poly = r;
    else poly = svmul_n_f64_x(pg, r, C.lead);
  } else {
    const auto r2 = svmul_f64_x(pg, r, r);
    const auto p01 = svmla_n_f64_x(pg, svdup_n_f64(C.a), r, C.b);
    const auto p23 = svmla_n_f64_x(pg, svdup_n_f64(C.c), r, C.d);
    const auto body = svmla_f64_x(pg, p01, p23, r2);
    if constexpr (C.unit_lead) poly = svmla_f64_x(pg, r, body, r2);
    else {
      poly = svmla_f64_x(
          pg, svmul_n_f64_x(pg, r, C.lead), body, r2);
    }
  }
  return svmla_f64_x(pg, scale, scale, poly);
}

/**
 * Tails for |x| beyond the FEXPA-safe window, mirroring
 * sve_exp_f32_special with a +/−53 exponent bias.
 */
template <const SVEExpF64Family& C, Accuracy Tier>
VECOPS_NOINLINE inline svfloat64_t sve_exp_f64_special(
    svbool_t pg, svfloat64_t x) {
  constexpr double flush_bound =
      sve_exp_gradual<Tier>() ? C.zero_bound : -C.special_bound;
  const auto full = svptrue_b64();
  const auto is_negative = svcmplt_n_f64(pg, x, 0.0);
  const auto offset =
      svsel_f64(is_negative, svdup_n_f64(-53.0), svdup_n_f64(53.0));
  const auto adjust =
      svsel_s64(is_negative, svdup_n_s64(-53), svdup_n_s64(53));
  const auto xc = svmin_n_f64_x(
      full, svmax_n_f64_x(full, x, C.zero_bound), C.inf_bound);
  auto y = sve_exp_f64_core<C, Accuracy::Strict, true>(xc, full, offset);
  y = svscale_f64_x(full, y, adjust);
  y = svsel_f64(
      svcmpgt_n_f64(pg, x, C.inf_bound),
      svdup_n_f64(std::numeric_limits<double>::infinity()), y);
  y = svsel_f64(svcmplt_n_f64(pg, x, flush_bound), svdup_n_f64(0.0), y);
  return y;
}

template <const SVEExpF64Family& C, Accuracy Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat64_t sve_exp_f64(svfloat64_t x, svbool_t pg) {
  const svbool_t special = NegativeOnly
      ? svcmplt_n_f64(pg, x, -C.special_bound)
      : svacge_n_f64(pg, x, C.special_bound);
  if (svptest_any(special, special))
    return sve_exp_f64_special<C, Tier>(pg, x);
  const svfloat64_t zero = svdup_n_f64(0.0);
  return sve_exp_f64_core<C, Tier, false>(x, pg, zero);
}

/* **************************************************************************** */
//    f16 exp (base e) — native core + special case                           //
/* **************************************************************************** */

/**
 * f16 FEXPA has no float-precision shift trick (the required table index
 * does not survive f16 rounding), so the exponent code is built from the
 * rounded step count as an integer instead. This native path always flushes
 * subnormal results to zero; Strict with VECOPS_PRESERVE_SUBNORMALS widens
 * to the f32 core in the dispatcher below.
 */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16_core(
    svfloat16_t xc, svbool_t pg) {
  constexpr bool estimate = Tier == Accuracy::Estimate;
  const auto nf = svrinta_f16_x(
      pg, svmul_n_f16_x(pg, xc, float16_t(46.16624130844683f)));
  const auto code =
      svcvt_u16_f16_x(pg, svadd_n_f16_x(pg, nf, float16_t(480)));
  const auto scale = svexpa_f16(code);
  svfloat16_t r;
  if constexpr (estimate) {
    r = svmla_n_f16_x(pg, xc, nf, float16_t(-0.02166084939249829));
  } else {
    r = svmla_n_f16_x(pg, xc, nf, float16_t(-0.02166748046875));
    r = svmla_n_f16_x(pg, r, nf, float16_t(6.631076251709805e-6));
  }
  return svmla_f16_x(pg, scale, scale, r);
}

/** Tails for the f16 domain: underflow below ln(2^-14), saturation above. */
VECOPS_NOINLINE inline svfloat16_t sve_exp_f16_special(
    svbool_t pg, svfloat16_t x) {
  const auto full = svptrue_b16();
  // exp(11.0859375) rounds to 65248; every larger finite input overflows.
  const float16_t overflow = float16_t(11.0859375);
  const float16_t normal_limit = float16_t(-9.704060527839);  // ln(2^-14)
  const auto xc = svmin_n_f16_x(
      full, svmax_n_f16_x(full, x, normal_limit), overflow);
  auto y = sve_exp_f16_core<Accuracy::Strict>(xc, full);
  y = svsel_f16(svcmpeq_n_f16(pg, x, overflow), svdup_n_f16(65248.0), y);
  y = svsel_f16(
      svcmpgt_n_f16(pg, x, overflow),
      svdup_n_f16(std::numeric_limits<float16_t>::infinity()), y);
  y = svsel_f16(svcmplt_n_f16(pg, x, normal_limit), svdup_n_f16(float16_t(0)), y);
  return y;
}

template <Accuracy Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16(svfloat16_t x, svbool_t pg) {
  // Asymmetric tails: the negative side must keep the FEXPA scale normal
  // while the positive side may run up to the saturation edge.
  const float16_t normal_limit = float16_t(-9.704060527839);
  const svbool_t special = [&] {
    if constexpr (NegativeOnly) return svcmplt_n_f16(pg, x, normal_limit);
    else {
      return svorr_b_z(
          pg, svcmplt_n_f16(pg, x, normal_limit),
          svcmpge_n_f16(pg, x, float16_t(11.0859375)));
    }
  }();
  if (svptest_any(special, special))
    return sve_exp_f16_special(pg, x);
  return sve_exp_f16_core<Tier>(x, pg);
}

/* **************************************************************************** */
//    f16 exp2/exp10 — native core + special case                              //
/* **************************************************************************** */

/**
 * The base-2/base-10 f16 families keep a native path: widening to f32 costs
 * four FCVTs per word, which dominate the runtime on current cores. The
 * trade-off is that the saturation edge is pinned by hand: above clamp_hi
 * the rounded step count would need a biased exponent of 31, which FEXPA
 * cannot produce, so the remaining grid points get exact selected values
 * and everything beyond overflows to infinity. clamp_hi itself must stay
 * hot-safe: lanes below the special bound can still be dragged into the
 * special path by a vector mate. Float16's narrowing constructor is not
 * constexpr, so the fields hold raw bit patterns.
 */
struct SVEExpF16Family {
  std::uint16_t log2_step;  // 32 * log2(base); nf = rinta(x * log2_step)
  std::uint16_t step_hi;    // (1 / log2(base)) / 32, split in two parts
  std::uint16_t step_lo;
  std::uint16_t lead, poly_c1;
  std::uint16_t normal_limit;  // below: the FEXPA scale would be subnormal
  std::uint16_t pos_bound;     // first grid point owned by the saturation fixups
  std::uint16_t clamp_hi;      // last grid point with a FEXPA-safe step count
  std::uint16_t sat_v;         // correctly rounded value at pos_bound
  bool two_part;
  std::uint16_t sat2_x, sat2_v;  // optional second saturation fixup
};

inline constexpr SVEExpF16Family kSVEExpF16B2{
    0x5000,  // 32
    0x2800,  // 0.03125
    0x0000,
    0x398C,  // ln2
    0x33B0,  // ln2^2 / 2
    0xCB00,  // -14 = log2(2^-14)
    0x4BFE,  // 15.984375: first grid point needing step count 512
    0x4BFD,  // 15.9765625
    0x7BEA,  // 64832 = correctly rounded 2^15.984375
    false,
    0x4BFF,  // 15.9921875: also needs step count 512
    0x7BF5};  // 65184 = correctly rounded 2^15.9921875

inline constexpr SVEExpF16Family kSVEExpF16B10{
    0x56A5,  // 106.3125 = 32 * log2(10) rounded to f16
    0x20D1,  // 0.00940704345703125 = log10(2)/32 rounded to f16
    0x0002,  // 1.1920928955078125e-07 second reduction part
    0x409B,  // 2.302734375 = ln10
    0x414D,  // 2.650390625 = ln10^2 / 2
    0xC437,  // -4.21484375 = log10(2^-14)
    0x44D0,  // 4.8125: first grid point needing step count 512
    0x44CF,  // 4.80859375
    0x7BED,  // 64928 = correctly rounded 10^4.8125
    true,
    0x0000,  // no second fixup: 10^4.81640625 already rounds to infinity
    0x0000};

template <const SVEExpF16Family& C, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16_family_core(
    svfloat16_t xc, svbool_t pg) {
  constexpr bool estimate = Tier == Accuracy::Estimate;
  const auto nf = svrinta_f16_x(
      pg, svmul_n_f16_x(pg, xc, float16_t::from_bits(C.log2_step)));
  const auto code =
      svcvt_u16_f16_x(pg, svadd_n_f16_x(pg, nf, float16_t(480)));
  const auto scale = svexpa_f16(code);
  auto r = svmls_n_f16_x(
      pg, xc, nf, float16_t::from_bits(C.step_hi));
  if constexpr (!estimate) {
    if constexpr (C.two_part)
      r = svmla_n_f16_x(pg, r, nf, float16_t::from_bits(C.step_lo));
  }
  const auto poly = [&]() -> svfloat16_t {
    if constexpr (estimate)
      return svmul_n_f16_x(pg, r, float16_t::from_bits(C.lead));
    else
      return svmul_f16_x(
          pg,
          svmla_n_f16_x(
              pg, svdup_n_f16(float16_t::from_bits(C.lead)), r,
              float16_t::from_bits(C.poly_c1)),
          r);
  }();
  return svmla_f16_x(pg, scale, scale, poly);
}

/** Tails: underflow below normal_limit and the pinned saturation edge. */
template <const SVEExpF16Family& C>
VECOPS_NOINLINE inline svfloat16_t sve_exp_f16_family_special(
    svbool_t pg, svfloat16_t x) {
  const auto full = svptrue_b16();
  // NaN lanes must survive the clamp, so use the propagating FMIN/FMAX.
  const auto xc = svmin_n_f16_x(
      full,
      svmax_n_f16_x(full, x, float16_t::from_bits(C.normal_limit)),
      float16_t::from_bits(C.clamp_hi));
  auto y = sve_exp_f16_family_core<C, Accuracy::Strict>(xc, full);
  y = svsel_f16(
      svcmpeq_n_f16(pg, x, float16_t::from_bits(C.pos_bound)),
      svdup_n_f16(float16_t::from_bits(C.sat_v)), y);
  if constexpr (C.sat2_x != 0) {
    y = svsel_f16(
        svcmpeq_n_f16(pg, x, float16_t::from_bits(C.sat2_x)),
        svdup_n_f16(float16_t::from_bits(C.sat2_v)), y);
    y = svsel_f16(
        svcmpgt_n_f16(pg, x, float16_t::from_bits(C.sat2_x)),
        svdup_n_f16(std::numeric_limits<float16_t>::infinity()), y);
  } else {
    y = svsel_f16(
        svcmpgt_n_f16(pg, x, float16_t::from_bits(C.pos_bound)),
        svdup_n_f16(std::numeric_limits<float16_t>::infinity()), y);
  }
  y = svsel_f16(
      svcmplt_n_f16(pg, x, float16_t::from_bits(C.normal_limit)),
      svdup_n_f16(float16_t(0)), y);
  return y;
}

template <const SVEExpF16Family& C, Accuracy Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16_family(
    svfloat16_t x, svbool_t pg) {
  // Asymmetric tails, like the e-base f16 path: the negative side keeps the
  // FEXPA scale normal, the positive side ends at the saturation fixup.
  const svbool_t special = [&] {
    if constexpr (NegativeOnly)
      return svcmplt_n_f16(pg, x, float16_t::from_bits(C.normal_limit));
    else {
      return svorr_b_z(
          pg, svcmplt_n_f16(pg, x, float16_t::from_bits(C.normal_limit)),
          svcmpge_n_f16(pg, x, float16_t::from_bits(C.pos_bound)));
    }
  }();
  if (svptest_any(special, special))
    return sve_exp_f16_family_special<C>(pg, x);
  return sve_exp_f16_family_core<C, Tier>(x, pg);
}

/* **************************************************************************** */
//    f16/bf16 widening routes and exp dispatch                               //
/* **************************************************************************** */

/**
 * Widens an f16 word to two f32 words, evaluates the f32 core on both
 * halves, and narrows back. Used for the e-base Strict subnormal-preserving
 * build and for every base-2/base-10 f16 evaluation (whose saturation edges
 * are cheaper to inherit from the f32 cores than to re-derive).
 */
template <const SVEExpF32Family& C, Accuracy Tier, bool NegativeOnly>
VECOPS_ALWAYS_INLINE svfloat16_t sve_exp_f16_via_f32(svfloat16_t raw) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd_value = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd_value, odd_value)));
#endif
  const auto result_low = sve_exp_f32<C, Tier, NegativeOnly>(
      low, svptrue_b32());
  const auto result_high = sve_exp_f32<C, Tier, NegativeOnly>(
      high, svptrue_b32());
  const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), result_low);
#if defined(__ARM_FEATURE_SVE2)
  return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), result_high);
#else
  const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), result_high);
  return svtrn1_f16(packed_low, packed_high);
#endif
}

/**
 * Flushes f16 subnormal outputs to zero. The widening route computes them
 * as f32 normals, so the documented Fast/Estimate (and non-preserving
 * Strict) flush contract needs this explicit select.
 */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_f16_flush_subnormals(svfloat16_t y) {
  if constexpr (!sve_exp_gradual<Tier>()) {
    return svsel_f16(
        svcmplt_n_f16(svptrue_b16(), y, float16_t(6.103515625e-05f)),
        svdup_n_f16(float16_t(0)), y);
  } else {
    return y;
  }
}

template <ExpBase Base, Accuracy Tier, bool NegativeOnly, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_exp_dispatch(
    Tag, NativeWordVec<Tag> value, svbool_t pg) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(sve_exp_f32<sve_exp_f32_family<Base>(),
                                    Tier, NegativeOnly>(raw, pg));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(sve_exp_f64<sve_exp_f64_family<Base>(),
                                    Tier, NegativeOnly>(raw, pg));
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (Base == ExpBase::E) {
      if constexpr (sve_exp_gradual<Tier>()) {
        return sve_basic_wrap_word<Tag>(
            sve_exp_f16_via_f32<sve_exp_f32_family<Base>(), Tier,
                                NegativeOnly>(raw));
      }
      return sve_basic_wrap_word<Tag>(sve_exp_f16<Tier, NegativeOnly>(raw, pg));
    } else {
      // The native families always flush; Strict under subnormal
      // preservation widens to the f32 cores like the e-base path.
      if constexpr (sve_exp_gradual<Tier>()) {
        return sve_basic_wrap_word<Tag>(
            sve_exp_f16_via_f32<sve_exp_f32_family<Base>(), Tier,
                                NegativeOnly>(raw));
      } else if constexpr (Base == ExpBase::Base2) {
        return sve_basic_wrap_word<Tag>(
            sve_exp_f16_family<kSVEExpF16B2, Tier, NegativeOnly>(raw, pg));
      } else {
        return sve_basic_wrap_word<Tag>(
            sve_exp_f16_family<kSVEExpF16B10, Tier, NegativeOnly>(raw, pg));
      }
    }
  } else {
    // bf16 shares the f32 exponent range, so the f32 core's own flush
    // boundaries already implement the subnormal contract.
    constexpr Accuracy compute_tier =
        sve_exp_gradual<Tier>() ? Accuracy::Strict : Accuracy::Estimate;
    const auto low = sve_bf16_to_f32_lo(raw);
    const auto high = sve_bf16_to_f32_hi(raw);
    return sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        sve_exp_f32<sve_exp_f32_family<Base>(), compute_tier, NegativeOnly>(
            low, svptrue_b32()),
        sve_exp_f32<sve_exp_f32_family<Base>(), compute_tier, NegativeOnly>(
            high, svptrue_b32())));
  }
}

template <ExpBase A, Accuracy Acc, bool NegativeOnly>
struct SVEExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<A, Acc, NegativeOnly>, Tag tag, NativeWordVec<Tag> value) {
    return sve_exp_dispatch<A, Acc, NegativeOnly>(
        tag, value, sve_full_predicate<ElementOf<Tag>>());
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<A, Acc, NegativeOnly> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    if constexpr (sizeof(ElementOf<Tag>) >= 4) {
      // The cores are elementwise with no cross-lane interaction, so the
      // mask can predicate the computation directly; only the inactive lanes
      // need the final blend.
      const auto computed = sve_exp_dispatch<A, Acc, NegativeOnly>(
          tag, value, sve_basic_raw_word(mask));
      return blend(tag, inactive, mask, computed);
    } else {
      // Narrow types widen through whole-word f32 conversions on the way to
      // the cores; sanitize inactive lanes first so the widening (and the
      // special-tail tests inside it) cannot observe stale values.
      return masked_unary_word(
          tag, mask, inactive, value, [&](NativeWordVec<Tag> safe) {
            return call<Index>(op, tag, safe);
          });
    }
  }
};

template <ExpBase A, Accuracy Acc, bool NegativeOnly>
struct NativeWordImpl<SVEBackend, ExpOp<A, Acc, NegativeOnly>>
    : SVEExpWordImpl<A, Acc, NegativeOnly> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_EXP_H
