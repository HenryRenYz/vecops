//
// Created by renyz on 2026/5/28.
//

#ifndef VECOPS_BFLOAT16_H
#define VECOPS_BFLOAT16_H

#include <cmath>
#include <cstdint>
#include <iostream>

#include "vecops/CoreDefs.h"
#include "./Bitcast.h"

#if defined(ARCH_ARM64)
#include <arm_neon.h>
#endif

namespace vecops {

struct alignas(2) BFloat16 {

  VECOPS_INLINE constexpr BFloat16() = default;

  VECOPS_INLINE constexpr BFloat16(float x) {
    #if defined(ARCH_ARM64) && defined(HAS_BF16)
    this->x = bitcast<uint16_t>(vcvth_bf16_f32(x));
    #else
    if (std::isnan(x)) {
      this->x = uint16_t(0x7fc0);
    } else {
      uint32_t bits = bitcast<uint32_t>(x);
      auto bias = ((bits >> 16) & 1) + uint32_t(0x7fff);
      this->x = uint16_t((bits + bias) >> 16);
    }
    #endif
  }

  VECOPS_INLINE constexpr BFloat16(double x) : BFloat16(float(x)) { }

  template <typename Int>
    requires std::is_integral_v<Int>
  VECOPS_INLINE explicit constexpr BFloat16(Int x) : BFloat16(float(x)) { }

  VECOPS_INLINE constexpr BFloat16(__bf16 x) {
    this->x = bitcast<uint16_t>(x);
  }

  VECOPS_INLINE static constexpr BFloat16 from_bits(uint16_t x) {
    BFloat16 b;
    b.x = x;
    return b;
  }

  VECOPS_INLINE constexpr uint16_t to_bits() const {
    return x;
  }

  VECOPS_INLINE constexpr operator float() const {
    #if defined(ARCH_ARM64) && defined(HAS_BF16)
    return vcvtah_f32_bf16(bitcast<__bf16>(this->x));
    #else
    return bitcast<float>(uint32_t(this->x) << 16);
    #endif
  }

  VECOPS_INLINE constexpr operator __bf16() const {
    return bitcast<__bf16>(this->x);
  }

  template <typename Int>
    requires std::is_integral_v<Int>
  VECOPS_INLINE explicit constexpr operator Int() const {
    return Int(float(*this));
  }

private:
  uint16_t x;
};

VECOPS_INLINE std::ostream& operator<<(std::ostream& out, const BFloat16& v) {
  out << float(v);
  return out;
}

VECOPS_INLINE BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return float(a) + float(b);
}

VECOPS_INLINE BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return float(a) - float(b);
}

VECOPS_INLINE BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return float(a) * float(b);
}

VECOPS_INLINE BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return float(a) / float(b);
}

VECOPS_INLINE BFloat16 operator-(const BFloat16& a) {
  return -float(a);
}

VECOPS_INLINE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

VECOPS_INLINE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

VECOPS_INLINE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

VECOPS_INLINE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

VECOPS_INLINE bool operator<(const BFloat16& a, const BFloat16& b) {
  return float(a) < float(b);
}

VECOPS_INLINE bool operator>(const BFloat16& a, const BFloat16& b) {
  return float(a) > float(b);
}

VECOPS_INLINE bool operator<=(const BFloat16& a, const BFloat16& b) {
  return float(a) <= float(b);
}

VECOPS_INLINE bool operator>=(const BFloat16& a, const BFloat16& b) {
  return float(a) >= float(b);
}

VECOPS_INLINE bool operator==(const BFloat16& a, const BFloat16& b) {
  return float(a) == float(b);
}

VECOPS_INLINE bool operator!=(const BFloat16& a, const BFloat16& b) {
  return float(a) != float(b);
}

} // namespace vecops

namespace std {

VECOPS_INLINE constexpr vecops::BFloat16 fabs(vecops::BFloat16 x) {
  return vecops::BFloat16::from_bits(x.to_bits() & 0x7fff);
}

VECOPS_INLINE constexpr vecops::BFloat16 abs(vecops::BFloat16 x) {
  return std::fabs(x);
}

template <>
class numeric_limits<vecops::BFloat16> {
public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =numeric_limits<float>::tinyness_before;

  static constexpr vecops::BFloat16 min() {
    return vecops::BFloat16::from_bits(0x0080);
  }

  static constexpr vecops::BFloat16 lowest() {
    return vecops::BFloat16::from_bits(0xFF7F);
  }

  static constexpr vecops::BFloat16 max() {
    return vecops::BFloat16::from_bits(0x7F7F);
  }

  static constexpr vecops::BFloat16 epsilon() {
    return vecops::BFloat16::from_bits(0x3C00);
  }

  static constexpr vecops::BFloat16 round_error() {
    return vecops::BFloat16::from_bits(0x3F00);
  }

  static constexpr vecops::BFloat16 infinity() {
    return vecops::BFloat16::from_bits(0x7F80);
  }

  static constexpr vecops::BFloat16 quiet_NaN() {
    return vecops::BFloat16::from_bits(0x7FC0);
  }

  static constexpr vecops::BFloat16 signaling_NaN() {
    return vecops::BFloat16::from_bits(0x7F80);
  }

  static constexpr vecops::BFloat16 denorm_min() {
    return vecops::BFloat16::from_bits(0x0001);
  }
};

} // namespace std

#endif //VECOPS_BFLOAT16_H
