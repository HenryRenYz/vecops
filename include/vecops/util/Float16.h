// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// The project copyright above applies to project-owned portions only.
// SPDX-FileCopyrightText: 2017 Facebook Inc.
// SPDX-FileCopyrightText: 2017 Georgia Institute of Technology
// SPDX-FileCopyrightText: 2019 Google LLC
// PyTorch-derived portions: BSD-3-Clause; overlapping FP16 portions: MIT.
// The complete original PyTorch project copyright statements, conditions
// and disclaimer are retained in LICENSES/PyTorch-BSD.txt; these project
// statements do not identify the original author of each imported function.
// Original FP16 notices and MIT terms: LICENSES/FP16-MIT.txt.
// SPDX-License-Identifier: MIT AND BSD-3-Clause

#ifndef VECOPS_FLOAT16_H
#define VECOPS_FLOAT16_H

#include <cmath>
#include <concepts>
#include <cstring>
#include <cstdint>

#include "vecops/CoreDefs.h"
#include "./Bitcast.h"
#include "./SmallFloat.h"

#ifdef ARCH_X86_FAMILY
#include <immintrin.h>
#endif

#if defined(ARCH_ARM64)
#include <arm_neon.h>
#endif

namespace vecops {

struct alignas(2) Float16 : SmallFloatOps<Float16> {

  VECOPS_INLINE constexpr Float16() = default;

  Float16(float x);

  VECOPS_INLINE Float16(double x) : Float16(float(x)) { }

  template <std::integral Int>
  VECOPS_INLINE explicit Float16(Int x) : Float16(float(x)) { }

  #if defined(ARCH_ARM64)
  VECOPS_INLINE constexpr Float16(__fp16 x) {
    this->x = bitcast<uint16_t>(x);
  }
  #else
  VECOPS_INLINE constexpr Float16(_Float16 x) {
    this->x = bitcast<uint16_t>(x);
  }
  #endif

  VECOPS_INLINE static constexpr Float16 from_bits(uint16_t x) {
    Float16 b;
    b.x = x;
    return b;
  }

  VECOPS_INLINE constexpr uint16_t to_bits() const {
    return x;
  }

  operator float() const;

  #if defined(ARCH_ARM64)

  VECOPS_INLINE constexpr operator __fp16() const {
    return bitcast<__fp16>(this->x);
  }

  #else
  VECOPS_INLINE constexpr operator _Float16() const {
    return bitcast<_Float16>(this->x);
  }
  #endif

  template <std::integral Int>
  VECOPS_INLINE explicit constexpr operator Int() const {
    return Int(float(*this));
  }

private:
  uint16_t x;
};

namespace details {
inline float fp16_to_fp32(uint16_t v) {
  #if defined(ARCH_X86_64) && defined(HAS_F16C)
  return _cvtsh_ss(v);
  #elif defined(ARCH_ARM64) && defined(HAS_NEON_FP16_ARITH)
  return float(bitcast<__fp16>(v));
  #else // HAS_F16C
  // Copied from https://github.com/pytorch/pytorch/blob/torch/headeronly/util/Half.h
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t) v << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  // const float exp_scale = 0x1.0p-112f;
  constexpr uint32_t scale_bits = (uint32_t) 15 << 23;
  float exp_scale_val = 0;
  #if defined(_MSC_VER) && defined(__clang__)
  __builtin_memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  #else
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  #endif

  const float exp_scale = exp_scale_val;
  const float normalized_value =
      bitcast<float>((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructed single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      bitcast<float>((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
                          (two_w < denormalized_cutoff ? bitcast<uint32_t>(denormalized_value)
                                                       : bitcast<uint32_t>(normalized_value));
  return bitcast<float>(result);
  #endif // HAS_F16C
}

inline uint16_t fp16_from_fp32(float v) {
  #if defined(ARCH_X86_64) && defined(HAS_F16C)
  return _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT);
  #elif defined(ARCH_ARM64) && defined(HAS_NEON_FP16_ARITH)
  return bitcast<uint16_t>(__fp16(v));
  #else // HAS_F16C
  // Copied from https://github.com/pytorch/pytorch/blob/torch/headeronly/util/Half.h
  // const float scale_to_inf = 0x1.0p+112f;
  // const float scale_to_zero = 0x1.0p-110f;
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val = 0, scale_to_zero_val = 0;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

  #if defined(_MSC_VER) && _MSC_VER == 1916
  float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
  #else
  float base = (std::fabs(v) * scale_to_inf) * scale_to_zero;
  #endif

  const uint32_t w = bitcast<uint32_t>(v);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = bitcast<float>((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = bitcast<uint32_t>(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
  #endif // HAS_F16C
}

} // namespace details

VECOPS_INLINE Float16::Float16(float x)
  : x(details::fp16_from_fp32(x)) { }

VECOPS_INLINE Float16::operator float() const {
  return details::fp16_to_fp32(x);
}

} // namespace vecops

namespace std {

VECOPS_DEFINE_STD_ABS(Float16)

VECOPS_DEFINE_SMALL_FLOAT_LIMITS(
    Float16, true, 11, 3, 5, -13, -4, 16, 4,
    0x0400, 0xFBFF, 0x7BFF, 0x1400, 0x3800, 0x7C00, 0x7E00, 0x7D00, 0x0001)

} // namespace std

#endif //VECOPS_FLOAT16_H
