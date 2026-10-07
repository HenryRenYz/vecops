// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_BF16_H
#define VECOPS_VEC_DETAILS_SVE_BF16_H

/**
 * @file Bf16.h
 * @brief SVE backend bfloat16 conversion helpers.
 */

#if !defined(HAS_SVE)
#error "This header requires an SVE target"
#endif

#include <arm_sve.h>

#include "vecops/CoreTypes.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                        bfloat16 conversion helpers                         //
/* **************************************************************************** */

VECOPS_ALWAYS_INLINE svfloat32_t sve_bf16_to_f32_lo(svbfloat16_t value) {
  const auto pg = svptrue_b32();
  const auto widened = svunpklo_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, widened, 16));
}

VECOPS_ALWAYS_INLINE svfloat32_t sve_bf16_to_f32_hi(svbfloat16_t value) {
  const auto pg = svptrue_b32();
  const auto widened = svunpkhi_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, widened, 16));
}

VECOPS_ALWAYS_INLINE svuint32_t sve_f32_to_bf16_rne_bits(
    svfloat32_t value) {
  const auto pg = svptrue_b32();
  const auto bits = svreinterpret_u32_f32(value);
  const auto lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const auto rounded = svlsr_n_u32_x(
      pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fffu)), 16);
  const auto magnitude = svand_n_u32_x(pg, bits, 0x7fffffffu);
  const auto nan = svcmpgt_n_u32(pg, magnitude, 0x7f800000u);
  return svsel_u32(nan, svdup_n_u32(0x7fc0u), rounded);
}

VECOPS_ALWAYS_INLINE svbfloat16_t sve_f32_to_bf16_lo(
    svfloat32_t value) {
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  // Keep UZP1 in the integer domain. The operation is bitwise-identical, but
  // avoids a bf16 vector_interleave that BiSheng 5.1 cannot select at -O2.
  const auto predicate = svptrue_b16();
  const auto converted = svcvt_bf16_f32_x(predicate, value);
  const auto bits = svreinterpret_u16_bf16(converted);
  return svreinterpret_bf16_u16(svuzp1_u16(bits, bits));
#else
  const auto bits = sve_f32_to_bf16_rne_bits(value);
  return svreinterpret_bf16_u16(svuzp1_u16(
      svreinterpret_u16_u32(bits), svreinterpret_u16_u32(bits)));
#endif
}

/** Packs the even (low) halves of two f32 vectors into one bf16 vector. */
VECOPS_ALWAYS_INLINE svbfloat16_t sve_f32_pair_to_bf16(
    svfloat32_t low, svfloat32_t high) {
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  // Keep UZP1 in the integer domain. The operation is bitwise-identical, but
  // avoids a bf16 vector_interleave that BiSheng 5.1 cannot select at -O2.
  const auto predicate = svptrue_b16();
  const auto low_converted = svcvt_bf16_f32_x(predicate, low);
  const auto high_converted = svcvt_bf16_f32_x(predicate, high);
  return svreinterpret_bf16_u16(svuzp1_u16(
      svreinterpret_u16_bf16(low_converted),
      svreinterpret_u16_bf16(high_converted)));
#else
  const auto low_bits = svreinterpret_u16_u32(sve_f32_to_bf16_rne_bits(low));
  const auto high_bits =
      svreinterpret_u16_u32(sve_f32_to_bf16_rne_bits(high));
  return svreinterpret_bf16_u16(svuzp1_u16(low_bits, high_bits));
#endif
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_BF16_H
