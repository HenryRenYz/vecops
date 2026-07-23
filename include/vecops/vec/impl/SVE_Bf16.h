//
// SVE_Bf16.h — bfloat16 <-> float32 conversion helpers
//

#ifndef VECOPS_SVE_BF16_H
#define VECOPS_SVE_BF16_H

#include <arm_sve.h>
#include "CoreTypes.h"
#include "SVE_Basic.h"

namespace vecops::vec::CPU_CAPABILITY {
namespace word {
namespace details {

/* ================================================================ */
//                      bf16 -> two f32 vectors                     //
/* ================================================================ */

VECOPS_VFUNC svfloat32_t bf16_to_f32_lo(svbfloat16_t v) {
  auto pg = ptrue<float32_t>();
  auto u16 = svreinterpret_u16_bf16(v);
  auto u32_lo = svunpklo_u32(u16);
  auto shifted = svlsl_n_u32_x(pg, u32_lo, 16);
  return svreinterpret_f32_u32(shifted);
}

VECOPS_VFUNC svfloat32_t bf16_to_f32_hi(svbfloat16_t v) {
  auto pg = ptrue<float32_t>();
  auto u16 = svreinterpret_u16_bf16(v);
  auto u32_hi = svunpkhi_u32(u16);
  auto shifted = svlsl_n_u32_x(pg, u32_hi, 16);
  return svreinterpret_f32_u32(shifted);
}

/* ================================================================ */
//                      two f32 vectors -> bf16                     //
//  lo_f32 = lower  half computation results (contiguous)           //
//  hi_f32 = upper  half computation results (contiguous)           //
/* ================================================================ */

VECOPS_VFUNC svuint32_t f32_to_bf16_rne_bits(svfloat32_t v) {
  const auto pg = ptrue<float32_t>();
  const auto bits = svreinterpret_u32_f32(v);
  const auto lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const auto rounded = svlsr_n_u32_x(
      pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fffu)), 16);
  const auto abs_bits = svand_n_u32_x(pg, bits, 0x7fffffffu);
  const auto nan = svcmpgt_n_u32(pg, abs_bits, 0x7f800000u);
  return svsel_u32(nan, svdup_n_u32(0x7fc0u), rounded);
}

VECOPS_VFUNC svbfloat16_t f32x2_to_bf16(svfloat32_t lo_f32, svfloat32_t hi_f32) {
  #if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  auto pg_bf16 = ptrue<bfloat16_t>();

  auto lo_bf16 = svcvt_bf16_x(pg_bf16, lo_f32);
  auto hi_bf16 = svcvt_bf16_x(pg_bf16, hi_f32);
  return svuzp1(lo_bf16, hi_bf16);
  #else
    auto u16_lo = svreinterpret_u16_u32(f32_to_bf16_rne_bits(lo_f32));
    auto u16_hi = svreinterpret_u16_u32(f32_to_bf16_rne_bits(hi_f32));
    return svreinterpret_bf16_u16(svuzp1_u16(u16_lo, u16_hi));
  #endif
}

/* ================================================================ */
//                      two f32 vectors -> f16                     //
/* ================================================================ */

VECOPS_VFUNC svfloat16_t f32x2_to_f16(svfloat32_t lo_f32, svfloat32_t hi_f32) {
  auto pg_f16 = ptrue<float16_t>();
  auto lo_f16 = svcvt_f16_f32_z(pg_f16, lo_f32);
  auto hi_f16 = svcvt_f16_f32_z(pg_f16, hi_f32);
  auto lo_u16 = svreinterpret_u16_f16(lo_f16);
  auto hi_u16 = svreinterpret_u16_f16(hi_f16);
  return svreinterpret_f16_u16(svuzp1_u16(lo_u16, hi_u16));
}

/* ================================================================ */
//                      mask conversion helpers                     //
//  bf16 mask (b16 granularity) <-> f32 mask (b32 granularity)      //
/* ================================================================ */

VECOPS_VFUNC svbool_t promote_mask_bf16_to_f32_lo(svbool_t m_bf16) {
  return svunpklo_b(m_bf16);
}

VECOPS_VFUNC svbool_t promote_mask_bf16_to_f32_hi(svbool_t m_bf16) {
  return svunpkhi_b(m_bf16);
}

// combine two f32 masks (lower/upper) back into one bf16 mask
VECOPS_VFUNC svbool_t combine_f32_masks_to_bf16(svbool_t m_lo, svbool_t m_hi) {
  return svuzp1_b16(m_lo, m_hi);
}

}  // namespace details
}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY

#endif // VECOPS_SVE_BF16_H
