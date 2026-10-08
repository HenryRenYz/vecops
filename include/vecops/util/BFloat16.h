// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_BFLOAT16_H
#define VECOPS_BFLOAT16_H

#include <cmath>
#include <concepts>
#include <cstdint>

#include "vecops/CoreDefs.h"
#include "./Bitcast.h"
#include "./SmallFloat.h"

#if defined(ARCH_ARM64)
#include <arm_neon.h>
#endif

namespace vecops {

struct alignas(2) BFloat16 : SmallFloatOps<BFloat16> {

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

  template <std::integral Int>
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

  template <std::integral Int>
  VECOPS_INLINE explicit constexpr operator Int() const {
    return Int(float(*this));
  }

private:
  uint16_t x;
};

} // namespace vecops

namespace std {

VECOPS_DEFINE_STD_ABS(BFloat16)

VECOPS_DEFINE_SMALL_FLOAT_LIMITS(
    BFloat16, false, 8, 2, 4, -125, -37, 128, 38,
    0x0080, 0xFF7F, 0x7F7F, 0x3C00, 0x3F00, 0x7F80, 0x7FC0, 0x7F80, 0x0001)

} // namespace std

#endif //VECOPS_BFLOAT16_H
