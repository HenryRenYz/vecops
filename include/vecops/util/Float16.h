//
// Created by renyz on 2026/5/28.
//

#ifndef VECOPS_FLOAT16_H
#define VECOPS_FLOAT16_H

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>

#include "vecops/CoreDefs.h"

#ifdef ARCH_X86_FAMILY
#include <immintrin.h>
#endif

namespace vecops {

struct alignas(2) Float16 {

  VECOPS_INLINE constexpr Float16() = default;

  constexpr Float16(float x);

  VECOPS_INLINE constexpr Float16(double x) : Float16(float(x)) { }

  template <typename Int, std::enable_if_t<std::is_integral_v<Int>, bool> = false>
  VECOPS_INLINE explicit constexpr Float16(Int x) : Float16(float(x)) { }

  #if defined(__arm__) || defined(__aarch64__)
  VECOPS_INLINE constexpr Float16(__fp16 x) {
    union { __fp16 f; uint16_t u; } u{.f = x};
    this->x = u.u;
  }
  #else
  VECOPS_INLINE constexpr Float16(_Float16 x) {
    union { _Float16 f; uint16_t u; } u{.f = x};
    this->x = u.u;
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

  constexpr operator float() const;

  #if defined(__arm__) || defined(__aarch64__)

  VECOPS_INLINE constexpr operator __fp16() const {
    union { __fp16 b; uint16_t u; } u{.u = this->x};
    return u.b;
  }

  #else
  VECOPS_INLINE constexpr operator _Float16() const {
    union { _Float16 b; uint16_t u; } u{.u = this->x};
    return u.b;
  }
  #endif

  template <typename Int, std::enable_if_t<std::is_integral_v<Int>, bool> = false>
  VECOPS_INLINE explicit constexpr operator Int() const {
    return Int(float(*this));
  }

private:
  uint16_t x;
};

namespace details {

VECOPS_INLINE constexpr float fp32_from_bits(uint32_t x) {
  union { float f; uint32_t i; } u{.i = x};
  return u.f;
}

VECOPS_INLINE constexpr uint32_t fp32_to_bits(float x) {
  union { float f; uint32_t i; } u{.f = x};
  return u.i;
}

inline float fp16_to_fp32(uint16_t v) {
  #ifdef HAS_F16C
  return _cvtsh_ss(v);
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
      details::fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

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
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

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
                          (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                                       : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
  #endif // HAS_F16C
}

inline uint16_t fp16_from_fp32(float v) {
  #ifdef HAS_F16C
  return _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT);
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
  float base = (fabsf(v) * scale_to_inf) * scale_to_zero;
  #endif

  const uint32_t w = fp32_to_bits(v);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
  #endif // HAS_F16C
}

} // namespace details

VECOPS_INLINE constexpr Float16::Float16(float x)
  : x(details::fp16_from_fp32(x)) { }

VECOPS_INLINE constexpr Float16::operator float() const {
  return details::fp16_to_fp32(x);
}

VECOPS_INLINE std::ostream& operator<<(std::ostream& out, const Float16& v) {
  out << float(v);
  return out;
}


VECOPS_INLINE Float16 operator+(const Float16& a, const Float16& b) {
  return float(a) + float(b);
}

VECOPS_INLINE Float16 operator-(const Float16& a, const Float16& b) {
  return float(a) - float(b);
}

VECOPS_INLINE Float16 operator*(const Float16& a, const Float16& b) {
  return float(a) * float(b);
}

VECOPS_INLINE Float16 operator/(const Float16& a, const Float16& b) {
  return float(a) / float(b);
}

VECOPS_INLINE Float16 operator-(const Float16& a) {
  return -float(a);
}

VECOPS_INLINE Float16& operator+=(Float16& a, const Float16& b) {
  a = a + b;
  return a;
}

VECOPS_INLINE Float16& operator-=(Float16& a, const Float16& b) {
  a = a - b;
  return a;
}

VECOPS_INLINE Float16& operator*=(Float16& a, const Float16& b) {
  a = a * b;
  return a;
}

VECOPS_INLINE Float16& operator/=(Float16& a, const Float16& b) {
  a = a / b;
  return a;
}


VECOPS_INLINE bool operator<(const Float16& a, const Float16& b) {
  return float(a) < float(b);
}

VECOPS_INLINE bool operator>(const Float16& a, const Float16& b) {
  return float(a) > float(b);
}

VECOPS_INLINE bool operator<=(const Float16& a, const Float16& b) {
  return float(a) <= float(b);
}

VECOPS_INLINE bool operator>=(const Float16& a, const Float16& b) {
  return float(a) >= float(b);
}

VECOPS_INLINE bool operator==(const Float16& a, const Float16& b) {
  return float(a) == float(b);
}

VECOPS_INLINE bool operator!=(const Float16& a, const Float16& b) {
  return float(a) != float(b);
}

} // namespace vecops

namespace std {

VECOPS_INLINE constexpr vecops::Float16 fabs(vecops::Float16 x) {
  return vecops::Float16::from_bits(x.to_bits() & 0x7fff);
}

VECOPS_INLINE constexpr vecops::Float16 abs(vecops::Float16 x) {
  return std::fabs(x);
}

template <>
class numeric_limits<vecops::Float16> {
public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

  static constexpr vecops::Float16 min() {
    return vecops::Float16::from_bits(0x0400);
  }
  static constexpr vecops::Float16 lowest() {
    return vecops::Float16::from_bits(0xFBFF);
  }
  static constexpr vecops::Float16 max() {
    return vecops::Float16::from_bits(0x7BFF);
  }
  static constexpr vecops::Float16 epsilon() {
    return vecops::Float16::from_bits(0x1400);
  }
  static constexpr vecops::Float16 round_error() {
    return vecops::Float16::from_bits(0x3800);
  }
  static constexpr vecops::Float16 infinity() {
    return vecops::Float16::from_bits(0x7C00);
  }
  static constexpr vecops::Float16 quiet_NaN() {
    return vecops::Float16::from_bits(0x7E00);
  }
  static constexpr vecops::Float16 signaling_NaN() {
    return vecops::Float16::from_bits(0x7D00);
  }
  static constexpr vecops::Float16 denorm_min() {
    return vecops::Float16::from_bits(0x0001);
  }
};

} // namespace std

#endif //VECOPS_FLOAT16_H
