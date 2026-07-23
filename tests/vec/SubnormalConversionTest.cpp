#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <vector>

#include "TestUtils.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

float f32_from_bits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t f32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint16_t bf16_rne_bits(uint32_t bits) {
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return static_cast<uint16_t>(
      (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

bool is_f32_subnormal(uint32_t bits) {
  return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
}

std::vector<uint32_t> demote_inputs() {
  std::vector<uint32_t> values = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
      0x00007fffu, 0x00008000u, 0x00008001u,
      0x0000ffffu, 0x00010000u, 0x00018000u,
      0x007e7fffu, 0x007e8000u, 0x007effffu,
      0x007f0000u, 0x007f7fffu, 0x007f8000u,
      0x007fffffu, 0x00800000u, 0x00808000u,
      0x3f7f7fffu, 0x3f7f8000u, 0x3f7f8001u,
      0x3f807fffu, 0x3f808000u, 0x3f808001u,
      0x7f7fffffu, 0x7f800000u, 0xff800000u,
      0x7f800001u, 0x7fffffffu, 0xff800001u,
  };
  for (uint32_t code = 0; code <= 0x80u; ++code) {
    for (uint32_t tail : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
      const uint32_t bits = (code << 16) | tail;
      values.push_back(bits);
      values.push_back(bits | 0x80000000u);
    }
  }
  return values;
}

template <int POW2>
void check_f32_to_bf16() {
  ScalableTag<vecops::bfloat16_t, POW2> to;
  using To = decltype(to);
  Rebind<float32_t, To> from;
  const nint_t input_lanes = size(from);
  std::vector<float32_t> input(static_cast<size_t>(input_lanes));
  std::vector<vecops::bfloat16_t> fused_output(
      static_cast<size_t>(input_lanes + 2),
      vecops::bfloat16_t::from_bits(0x3f80u));
  const auto values = demote_inputs();

  for (size_t pos = 0; pos < values.size(); pos += static_cast<size_t>(input_lanes)) {
    for (nint_t i = 0; i < input_lanes; ++i) {
      input[static_cast<size_t>(i)] =
          f32_from_bits(values[(pos + static_cast<size_t>(i)) % values.size()]);
    }
    const auto input_v = loadu(from, input.data());
    const auto output = demote(to, input_v);
    demote_storeu(from, fused_output.data() + 1, input_v);
    const size_t count = std::min(static_cast<size_t>(input_lanes), values.size() - pos);
    for (size_t i = 0; i < count; ++i) {
      const uint32_t input_bits = f32_bits(input[i]);
      const auto output_lane = get(to, output, static_cast<nint_t>(i));
      const uint16_t actual = output_lane.to_bits();
      const uint16_t fused_actual = fused_output[i + 1].to_bits();
      if ((input_bits & 0x7fffffffu) > 0x7f800000u) {
        EXPECT_TRUE(std::isnan(static_cast<float>(output_lane)));
        EXPECT_TRUE(std::isnan(static_cast<float>(fused_output[i + 1])));
        continue;
      }
      const uint16_t expected = bf16_rne_bits(input_bits);
#ifdef VECOPS_PRESERVE_SUBNORMALS
      EXPECT_EQ(actual, expected) << "input_bits=0x" << std::hex << input_bits;
      EXPECT_EQ(fused_actual, expected)
          << "fused input_bits=0x" << std::hex << input_bits;
#else
      if (is_f32_subnormal(input_bits)) {
        EXPECT_TRUE(actual == expected || (actual & 0x7fffu) == 0)
            << "input_bits=0x" << std::hex << input_bits
            << " expected=0x" << expected << " actual=0x" << actual;
        EXPECT_TRUE(fused_actual == expected || (fused_actual & 0x7fffu) == 0)
            << "fused input_bits=0x" << std::hex << input_bits
            << " expected=0x" << expected << " actual=0x" << fused_actual;
      } else {
        EXPECT_EQ(actual, expected) << "input_bits=0x" << std::hex << input_bits;
        EXPECT_EQ(fused_actual, expected)
            << "fused input_bits=0x" << std::hex << input_bits;
      }
#endif
    }
  }
}

template <int POW2>
void check_bf16_to_f32() {
  ScalableTag<vecops::bfloat16_t, POW2> from;
  using From = decltype(from);
  Rebind<float32_t, From> to;
  const nint_t output_lanes = size(to);
  std::vector<vecops::bfloat16_t> input(static_cast<size_t>(size(from)));

  for (uint32_t base = 0; base <= 0xffffu; base += static_cast<uint32_t>(output_lanes)) {
    for (nint_t i = 0; i < size(from); ++i) {
      const uint32_t code = std::min(0xffffu, base + static_cast<uint32_t>(i));
      input[static_cast<size_t>(i)] =
          vecops::bfloat16_t::from_bits(static_cast<uint16_t>(code));
    }
    const auto output = promote(to, loadu(from, input.data()));
    const auto fused_output = promote_loadu(to, input.data());
    const uint32_t count = std::min(
        static_cast<uint32_t>(output_lanes), 0x10000u - base);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t expected = (base + i) << 16;
      const uint32_t actual =
          f32_bits(get(to, output, static_cast<nint_t>(i)));
      const uint32_t fused_actual =
          f32_bits(get(to, fused_output, static_cast<nint_t>(i)));
      EXPECT_EQ(actual, expected) << "bf16_bits=0x" << std::hex << base + i;
      EXPECT_EQ(fused_actual, expected)
          << "fused bf16_bits=0x" << std::hex << base + i;
    }
    if (base + static_cast<uint32_t>(output_lanes) >= 0x10000u) break;
  }
}

}  // namespace

#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
TEST(VecSubnormalConversionTest, Float32ToBFloat16RoundingAndSubnormals) {
  check_f32_to_bf16<0>();
  check_f32_to_bf16<-1>();
}

TEST(VecSubnormalConversionTest, BFloat16ToFloat32AllBitPatterns) {
  check_bf16_to_f32<0>();
  check_bf16_to_f32<-1>();
}
#endif
