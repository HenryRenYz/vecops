// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// The existing Arm rsqrt design reference is retained; no verbatim upstream
// function was identified. See THIRD_PARTY_NOTICES.md for reference scope.
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_MATH_RECIPROCAL_H
#define VECOPS_VEC_DETAILS_SVE_MATH_RECIPROCAL_H

/**
 * @file Reciprocal.h
 * @brief SVE backend implementations for rcp and rsqrt (all accuracy tiers).
 *
 * The refinement ladder follows the Arm optimized-routines SVE rsqrt design
 * (math/aarch64/sve/{rsqrtf,rsqrt}.c upstream): a branch-free inline core
 * built from the hardware estimate plus FRSQRTS/FRECPS Newton steps, and one
 * NOINLINE special routine that owns every tail. The tails re-run the core on
 * an exactly-scaled input (svscale) so every intermediate stays normal and
 * gradual subnormal outputs fall out of a final exact power-of-two scaling.
 *
 * Per-tier cores, with the documented contract each one must satisfy:
 *
 * | Tier     | Chain                                   | Contract          |
 * |----------|-----------------------------------------|-------------------|
 * | Estimate | FRSQRTE / FRECPE only                   | rel. error <= 2^-7|
 * | Fast     | estimate + 1 FRSQRTS/FRECPS step        | rel. error <= 2^-15 |
 * | Strict   | estimate + steps + FMA residual polish  | ULP error <= 1    |
 *
 * The estimate instructions are architecturally accurate to 2^-8, and one
 * step squares that to about 2^-16. The Strict polish is the Markstein-style
 * residual form: r = 1 - x*y^2 (rsqrt) or r = 1 - x*y (rcp) is near-exact
 * inside a single FMA, and one more fused multiply-add folds the correction
 * in with exactly one rounding, which pins the final error below 1 ULP.
 *
 * Special inputs (zero, subnormal, negative, infinity, NaN) produce the same
 * results in every tier: the special routine always runs the Strict core.
 * NaN never matches the special predicates and propagates through the inline
 * arithmetic. rcp(+-0) = +-inf, rcp(+-inf) = +-0, rsqrt(+-0) = +inf,
 * rsqrt(x < 0) = NaN, with signs carried by the estimate instructions.
 *
 * All cores evaluate unpredicated on the raw word; an explicit call mask only
 * guards the special-trigger compares and the final lane blend, so inactive
 * lanes may compute garbage that is blended away afterwards.
 */

#include <arm_sve.h>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {

/** True for the tiers that must produce gradual subnormal results if enabled. */
template <Accuracy Tier>
constexpr bool sve_recip_gradual() {
#ifdef VECOPS_PRESERVE_SUBNORMALS
  return Tier == Accuracy::Strict;
#else
  return false;
#endif
}

/* **************************************************************************** */
//    f32 / f64 cores                                                           //
/* **************************************************************************** */

/** rsqrt core for normal inputs: estimate, FRSQRTS step, [Strict] polish. */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_rsqrt_f32_core(svfloat32_t x) {
  const auto full = svptrue_b32();
  const auto y0 = svrsqrte_f32(x);
  if constexpr (Tier == Accuracy::Estimate) return y0;
  const auto y1 = svmul_f32_x(
      full, y0, svrsqrts_f32(x, svmul_f32_x(full, y0, y0)));
  if constexpr (Tier == Accuracy::Fast) return y1;
  const auto residual = svmls_f32_x(
      full, svdup_n_f32(1.0f), x, svmul_f32_x(full, y1, y1));
  return svmla_f32_x(
      full, y1, residual, svmul_n_f32_x(full, y1, 0.5f));
}

/** rcp core for normal inputs: estimate, FRECPS step, [Strict] polish. */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_rcp_f32_core(svfloat32_t x) {
  const auto full = svptrue_b32();
  const auto y0 = svrecpe_f32(x);
  if constexpr (Tier == Accuracy::Estimate) return y0;
  const auto y1 = svmul_f32_x(full, y0, svrecps_f32(x, y0));
  if constexpr (Tier == Accuracy::Fast) return y1;
  const auto residual = svmls_f32_x(full, svdup_n_f32(1.0f), x, y1);
  return svmla_f32_x(full, y1, residual, y1);
}

/**
 * f64 rsqrt core. Strict takes a second FRSQRTS step before the polish so
 * the residual the polish corrects is small enough for a <= 1 ULP bound.
 */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_rsqrt_f64_core(svfloat64_t x) {
  const auto full = svptrue_b64();
  const auto y0 = svrsqrte_f64(x);
  if constexpr (Tier == Accuracy::Estimate) return y0;
  auto y = svmul_f64_x(
      full, y0, svrsqrts_f64(x, svmul_f64_x(full, y0, y0)));
  if constexpr (Tier == Accuracy::Strict) {
    y = svmul_f64_x(full, y, svrsqrts_f64(x, svmul_f64_x(full, y, y)));
  }
  const auto residual = svmls_f64_x(
      full, svdup_n_f64(1.0), x, svmul_f64_x(full, y, y));
  return svmla_f64_x(full, y, residual, svmul_n_f64_x(full, y, 0.5));
}

/** f64 rcp core; Strict takes a second FRECPS step before the polish. */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_rcp_f64_core(svfloat64_t x) {
  const auto full = svptrue_b64();
  const auto y0 = svrecpe_f64(x);
  if constexpr (Tier == Accuracy::Estimate) return y0;
  auto y = svmul_f64_x(full, y0, svrecps_f64(x, y0));
  if constexpr (Tier == Accuracy::Strict) {
    y = svmul_f64_x(full, y, svrecps_f64(x, y));
  }
  const auto residual = svmls_f64_x(full, svdup_n_f64(1.0), x, y);
  return svmla_f64_x(full, y, residual, y);
}

/* **************************************************************************** */
//    special-case routines                                                     //
/* **************************************************************************** */

/**
 * rsqrt tails: magnitudes below the normal range (subnormals and zero) and
 * infinities. Small lanes are scaled up by an exact power of two so the
 * estimate sees a normal operand, and the result is scaled back down by the
 * square root of the same factor. rsqrt outputs never drop below the
 * smallest normal, so there is no output-side tail and no subnormal flush.
 */
VECOPS_NOINLINE inline svfloat32_t sve_rsqrt_f32_special(
    svbool_t pg, svfloat32_t x) {
  const auto full = svptrue_b32();
  // A partially-special word must not scale its normal lanes: x*2^64 can
  // overflow them to infinity, whose Strict polish forms 0*inf NaNs. Scale
  // only the small lanes; zero and infinity lanes take the raw estimate,
  // which is exact there for the same reason.
  const auto small = svnot_b_z(pg, svacge_n_f32(pg, x, 0x1p-126f));
  const auto adjust = svsel_s32(small, svdup_n_s32(64), svdup_n_s32(0));
  const auto scaled = svscale_f32_x(full, x, adjust);
  auto y = sve_rsqrt_f32_core<Accuracy::Strict>(scaled);
  y = svscale_f32_x(
      full, y, svsel_s32(small, svdup_n_s32(32), svdup_n_s32(0)));
  const auto is_zero = svcmpeq_n_f32(pg, x, 0.0f);
  // Only +infinity maps to +0; -infinity stays NaN through the core.
  const auto is_inf = svcmpeq_n_f32(
      pg, x, std::numeric_limits<float>::infinity());
  // rsqrt(+-0) = +inf and rsqrt(+inf) = +0; the raw estimate of -0 reads
  // -inf on this core, so constants are selected for both classes.
  y = svsel_f32(
      is_inf, svdup_n_f32(0.0f),
      svsel_f32(
          is_zero, svdup_n_f32(std::numeric_limits<float>::infinity()), y));
  return y;
}

VECOPS_NOINLINE inline svfloat64_t sve_rsqrt_f64_special(
    svbool_t pg, svfloat64_t x) {
  const auto full = svptrue_b64();
  // Same per-lane scaling discipline as the f32 routine above.
  const auto small = svnot_b_z(pg, svacge_n_f64(pg, x, 0x1p-1022));
  const auto adjust = svsel_s64(small, svdup_n_s64(512), svdup_n_s64(0));
  const auto scaled = svscale_f64_x(full, x, adjust);
  auto y = sve_rsqrt_f64_core<Accuracy::Strict>(scaled);
  y = svscale_f64_x(
      full, y, svsel_s64(small, svdup_n_s64(256), svdup_n_s64(0)));
  const auto is_zero = svcmpeq_n_f64(pg, x, 0.0);
  const auto is_inf = svcmpeq_n_f64(
      pg, x, std::numeric_limits<double>::infinity());
  y = svsel_f64(
      is_inf, svdup_n_f64(0.0),
      svsel_f64(
          is_zero, svdup_n_f64(std::numeric_limits<double>::infinity()), y));
  return y;
}

/**
 * rcp tails. Both sides need care: subnormal or tiny inputs make the
 * estimate overflow, and inputs beyond 2^126 / 2^1022 produce subnormal
 * outputs. One routine rescales either side by an exact power of two, always
 * runs the Strict core, and rescales the result, which also yields gradual
 * subnormal outputs when subnormal preservation is enabled. Lanes holding
 * zero or infinity select the raw estimate, which is exact on those lanes
 * and short-circuits the indeterminate 0*inf residuals the polish would
 * otherwise form. Non-preserving tiers flush subnormal outputs to signed
 * zero (subnormal * 0.0 rounds to a correctly signed zero).
 */
template <Accuracy Tier>
VECOPS_NOINLINE inline svfloat32_t sve_rcp_f32_special(
    svbool_t pg, svfloat32_t x) {
  const auto full = svptrue_b32();
  const auto small =
      svnot_b_z(pg, svacge_n_f32(pg, x, 0x1p-126f));
  const auto big = svacgt_n_f32(pg, x, 0x1p126f);
  const auto adjust = svsel_s32(
      small, svdup_n_s32(64),
      svsel_s32(big, svdup_n_s32(-64), svdup_n_s32(0)));
  const auto scaled = svscale_f32_x(full, x, adjust);
  auto y = sve_rcp_f32_core<Accuracy::Strict>(scaled);
  y = svscale_f32_x(full, y, adjust);
  const auto extreme = svorr_b_z(
      pg, svcmpeq_n_f32(pg, x, 0.0f),
      svcmpeq_n_f32(pg, svabs_f32_x(pg, x),
                    std::numeric_limits<float>::infinity()));
  y = svsel_f32(extreme, svrecpe_f32(x), y);
  if constexpr (!sve_recip_gradual<Tier>()) {
    const auto flush = svand_b_z(
        pg, svcmpne_n_f32(pg, y, 0.0f),
        svnot_b_z(pg, svacge_n_f32(pg, y, 0x1p-126f)));
    y = svsel_f32(flush, svmul_n_f32_x(full, y, 0.0f), y);
  }
  return y;
}

template <Accuracy Tier>
VECOPS_NOINLINE inline svfloat64_t sve_rcp_f64_special(
    svbool_t pg, svfloat64_t x) {
  const auto full = svptrue_b64();
  const auto small =
      svnot_b_z(pg, svacge_n_f64(pg, x, 0x1p-1022));
  const auto big = svacgt_n_f64(pg, x, 0x1p1022);
  const auto adjust = svsel_s64(
      small, svdup_n_s64(512),
      svsel_s64(big, svdup_n_s64(-512), svdup_n_s64(0)));
  const auto scaled = svscale_f64_x(full, x, adjust);
  auto y = sve_rcp_f64_core<Accuracy::Strict>(scaled);
  y = svscale_f64_x(full, y, adjust);
  const auto extreme = svorr_b_z(
      pg, svcmpeq_n_f64(pg, x, 0.0),
      svcmpeq_n_f64(pg, svabs_f64_x(pg, x),
                    std::numeric_limits<double>::infinity()));
  y = svsel_f64(extreme, svrecpe_f64(x), y);
  if constexpr (!sve_recip_gradual<Tier>()) {
    const auto flush = svand_b_z(
        pg, svcmpne_n_f64(pg, y, 0.0),
        svnot_b_z(pg, svacge_n_f64(pg, y, 0x1p-1022)));
    y = svsel_f64(flush, svmul_n_f64_x(full, y, 0.0), y);
  }
  return y;
}

/* **************************************************************************** */
//    word-level entries (f32 / f64)                                           //
/* **************************************************************************** */

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_rsqrt_f32(svfloat32_t x, svbool_t pg) {
  const svbool_t special =
      svorr_b_z(pg, svnot_b_z(pg, svacge_n_f32(pg, x, 0x1p-126f)),
                svacge_n_f32(pg, x, std::numeric_limits<float>::max()));
  if (svptest_any(special, special))
    return sve_rsqrt_f32_special(pg, x);
  return sve_rsqrt_f32_core<Tier>(x);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_rsqrt_f64(svfloat64_t x, svbool_t pg) {
  const svbool_t special =
      svorr_b_z(pg, svnot_b_z(pg, svacge_n_f64(pg, x, 0x1p-1022)),
                svacge_n_f64(pg, x, std::numeric_limits<double>::max()));
  if (svptest_any(special, special))
    return sve_rsqrt_f64_special(pg, x);
  return sve_rsqrt_f64_core<Tier>(x);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_rcp_f32(svfloat32_t x, svbool_t pg) {
  const svbool_t special =
      svorr_b_z(pg, svnot_b_z(pg, svacge_n_f32(pg, x, 0x1p-126f)),
                svacgt_n_f32(pg, x, 0x1p126f));
  if (svptest_any(special, special)) return sve_rcp_f32_special<Tier>(pg, x);
  return sve_rcp_f32_core<Tier>(x);
}

template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_rcp_f64(svfloat64_t x, svbool_t pg) {
  const svbool_t special =
      svorr_b_z(pg, svnot_b_z(pg, svacge_n_f64(pg, x, 0x1p-1022)),
                svacgt_n_f64(pg, x, 0x1p1022));
  if (svptest_any(special, special)) return sve_rcp_f64_special<Tier>(pg, x);
  return sve_rcp_f64_core<Tier>(x);
}

/* **************************************************************************** */
//    f16 / bf16 widening routes                                               //
/* **************************************************************************** */

/**
 * Widens an f16 word to two f32 words, evaluates the f32 pipeline on both
 * halves, and narrows back. Every f16 value (including subnormals) is normal
 * in f32, so the f32 special routine already owns all f16 tails exactly.
 */
template <Accuracy Tier, bool IsRsqrt>
VECOPS_ALWAYS_INLINE svfloat16_t sve_recip_f16_via_f32(svfloat16_t raw) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd_value = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd_value, odd_value)));
#endif
  svfloat32_t result_low;
  svfloat32_t result_high;
  if constexpr (IsRsqrt) {
    result_low = sve_rsqrt_f32<Tier>(low, svptrue_b32());
    result_high = sve_rsqrt_f32<Tier>(high, svptrue_b32());
  } else {
    result_low = sve_rcp_f32<Tier>(low, svptrue_b32());
    result_high = sve_rcp_f32<Tier>(high, svptrue_b32());
  }
  const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), result_low);
#if defined(__ARM_FEATURE_SVE2)
  return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), result_high);
#else
  const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), result_high);
  return svtrn1_f16(packed_low, packed_high);
#endif
}

/**
 * Flushes subnormal f16 outputs to (signed) zero for the tiers whose
 * contract does not preserve them. The widening route computes them as f32
 * normals, so the flush needs this explicit select. Negative results must
 * not be caught by the magnitude bound, hence the absolute compare.
 */
template <Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_recip_f16_flush(svfloat16_t y) {
  if constexpr (!sve_recip_gradual<Tier>()) {
    const auto flush = svand_b_z(
        svptrue_b16(), svcmpne_n_f16(svptrue_b16(), y, float16_t(0)),
        svnot_b_z(svptrue_b16(),
                  svacge_n_f16(svptrue_b16(), y, float16_t(6.103515625e-05f))));
    return svsel_f16(
        flush, svmul_n_f16_x(svptrue_b16(), y, float16_t(0)), y);
  } else {
    return y;
  }
}

/* **************************************************************************** */
//    dispatch and word-impl registration                                      //
/* **************************************************************************** */

template <Accuracy Tier, bool IsRsqrt, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_recip_dispatch(
    Tag, NativeWordVec<Tag> value, svbool_t pg) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (IsRsqrt) return sve_basic_wrap_word<Tag>(
        sve_rsqrt_f32<Tier>(raw, pg));
    else return sve_basic_wrap_word<Tag>(sve_rcp_f32<Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (IsRsqrt) return sve_basic_wrap_word<Tag>(
        sve_rsqrt_f64<Tier>(raw, pg));
    else return sve_basic_wrap_word<Tag>(sve_rcp_f64<Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float16_t>) {
    // The f32 pipeline covers every f16 input and output tail; only the
    // subnormal-flush contract is f16-specific.
    return sve_basic_wrap_word<Tag>(
        sve_recip_f16_flush<Tier>(sve_recip_f16_via_f32<Tier, IsRsqrt>(raw)));
  } else {
    // bf16 shares the f32 exponent range, so the f32 pipeline's own bounds
    // and flush behavior already implement the bf16 subnormal contract.
    const auto low = sve_bf16_to_f32_lo(raw);
    const auto high = sve_bf16_to_f32_hi(raw);
    svfloat32_t result_low;
    svfloat32_t result_high;
    if constexpr (IsRsqrt) {
      result_low = sve_rsqrt_f32<Tier>(low, svptrue_b32());
      result_high = sve_rsqrt_f32<Tier>(high, svptrue_b32());
    } else {
      result_low = sve_rcp_f32<Tier>(low, svptrue_b32());
      result_high = sve_rcp_f32<Tier>(high, svptrue_b32());
    }
    return sve_basic_wrap_word<Tag>(
        sve_f32_pair_to_bf16(result_low, result_high));
  }
}

/**
 * Shared word implementation for both reciprocal ops. Wide element formats
 * are elementwise, so an explicit mask only guards the special-trigger
 * compares and the final blend. Narrow formats widen through whole-word f32
 * conversions; sanitize inactive lanes first so the widening (and the
 * special-tail tests inside it) cannot observe stale values.
 */
template <Accuracy A, bool IsRsqrt>
struct SVERecipWordImpl {
  template <nint_t Index, FloatingTag Tag, typename Op>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    static_assert(
        std::same_as<Op, RsqrtOp<A>> || std::same_as<Op, RcpOp<A>>);
    return sve_recip_dispatch<A, IsRsqrt>(
        tag, value, sve_full_predicate<ElementOf<Tag>>());
  }

  template <nint_t Index, FloatingTag Tag, typename Op, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    if constexpr (sizeof(ElementOf<Tag>) >= 4) {
      const auto computed =
          sve_recip_dispatch<A, IsRsqrt>(tag, value, sve_basic_raw_word(mask));
      return blend(tag, inactive, mask, computed);
    } else {
      return masked_unary_word(
          tag, mask, inactive, value, [&](NativeWordVec<Tag> safe) {
            return sve_recip_dispatch<A, IsRsqrt>(
                tag, safe, sve_full_predicate<ElementOf<Tag>>());
          });
    }
  }
};

template <Accuracy A>
struct NativeWordImpl<SVEBackend, RsqrtOp<A>>
    : SVERecipWordImpl<A, true> {};

template <Accuracy A>
struct NativeWordImpl<SVEBackend, RcpOp<A>>
    : SVERecipWordImpl<A, false> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_RECIPROCAL_H
