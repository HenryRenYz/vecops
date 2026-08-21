#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "vecops/vec/Conversion.h"
#include "vecops/vec/ConversionMemory.h"
#include "vecops/vec/Memory.h"
#include "vecops/util/ScalarConvert.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

float subnormal_f32_from_bits(uint32_t bits) {
  return vecops::bitcast<float>(bits);
}

uint32_t subnormal_f32_bits(float value) {
  return vecops::bitcast<uint32_t>(value);
}

uint16_t subnormal_bf16_rne_bits(uint32_t bits) {
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return static_cast<uint16_t>(
      (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

bool is_f32_subnormal(uint32_t bits) {
  return (bits & 0x7f800000u) == 0 &&
      (bits & 0x007fffffu) != 0;
}

std::vector<uint32_t> bf16_demote_boundary_inputs() {
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

template <int ScalePower>
void check_f32_to_bf16_boundaries() {
  using ToTag = vec::ScalableTag<vecops::bfloat16_t, ScalePower>;
  using FromTag = vec::Rebind<float, ToTag>;
  const auto lanes = vec::size(FromTag{});
  std::vector<float> input(static_cast<std::size_t>(lanes));
  std::vector<vecops::bfloat16_t> stored(
      static_cast<std::size_t>(lanes + 2),
      vecops::bfloat16_t::from_bits(0x3f80u));
  const auto values = bf16_demote_boundary_inputs();

  for (std::size_t pos = 0; pos < values.size();
       pos += static_cast<std::size_t>(lanes)) {
    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      input[static_cast<std::size_t>(lane)] = subnormal_f32_from_bits(
          values[(pos + static_cast<std::size_t>(lane)) % values.size()]);
    }
    const auto input_v = vec::load(FromTag{}, input.data());
    const auto output = vec::convert(ToTag{}, FromTag{}, input_v);
    vec::store_convert(FromTag{}, stored.data() + 1, input_v);
    const auto count = std::min(
        static_cast<std::size_t>(lanes), values.size() - pos);
    for (std::size_t lane = 0; lane < count; ++lane) {
      const uint32_t input_bits = subnormal_f32_bits(input[lane]);
      const auto converted = vec::get(
          ToTag{}, output, static_cast<vecops::nint_t>(lane));
      const uint16_t actual = converted.to_bits();
      const uint16_t stored_actual = stored[lane + 1].to_bits();
      if ((input_bits & 0x7fffffffu) > 0x7f800000u) {
        EXPECT_TRUE(std::isnan(static_cast<float>(converted)));
        EXPECT_TRUE(std::isnan(static_cast<float>(stored[lane + 1])));
        continue;
      }
      const uint16_t expected = subnormal_bf16_rne_bits(input_bits);
#if defined(VECOPS_PRESERVE_SUBNORMALS)
      EXPECT_EQ(expected, actual) << "input_bits=0x" << std::hex << input_bits;
      EXPECT_EQ(expected, stored_actual)
          << "stored input_bits=0x" << std::hex << input_bits;
#else
      if (is_f32_subnormal(input_bits)) {
        EXPECT_TRUE(actual == expected || (actual & 0x7fffu) == 0)
            << "input_bits=0x" << std::hex << input_bits;
        EXPECT_TRUE(
            stored_actual == expected || (stored_actual & 0x7fffu) == 0)
            << "stored input_bits=0x" << std::hex << input_bits;
      } else {
        EXPECT_EQ(expected, actual)
            << "input_bits=0x" << std::hex << input_bits;
        EXPECT_EQ(expected, stored_actual)
            << "stored input_bits=0x" << std::hex << input_bits;
      }
#endif
    }
  }
}

template <int ScalePower>
void check_bf16_to_f32_all_bits() {
  using FromTag = vec::ScalableTag<vecops::bfloat16_t, ScalePower>;
  using ToTag = vec::Rebind<float, FromTag>;
  const auto output_lanes = vec::size(ToTag{});
  std::vector<vecops::bfloat16_t> input(
      static_cast<std::size_t>(vec::size(FromTag{})));

  for (uint32_t base = 0; base <= 0xffffu;
       base += static_cast<uint32_t>(output_lanes)) {
    for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane) {
      input[static_cast<std::size_t>(lane)] =
          vecops::bfloat16_t::from_bits(static_cast<uint16_t>(
              std::min(0xffffu, base + static_cast<uint32_t>(lane))));
    }
    const auto loaded = vec::load(FromTag{}, input.data());
    const auto output = vec::convert(ToTag{}, FromTag{}, loaded);
    const auto fused = vec::load_convert(ToTag{}, input.data());
    const uint32_t count = std::min(
        static_cast<uint32_t>(output_lanes), 0x10000u - base);
    for (uint32_t lane = 0; lane < count; ++lane) {
      const uint32_t expected = (base + lane) << 16;
      EXPECT_EQ(expected, subnormal_f32_bits(vec::get(
          ToTag{}, output, static_cast<vecops::nint_t>(lane))));
      EXPECT_EQ(expected, subnormal_f32_bits(vec::get(
          ToTag{}, fused, static_cast<vecops::nint_t>(lane))));
    }
    if (base + static_cast<uint32_t>(output_lanes) >= 0x10000u) break;
  }
}

template <typename T, typename To>
T conversion_input(vecops::nint_t lane) {
  if constexpr (std::same_as<T, vecops::bfloat16_t> ||
                std::same_as<T, vecops::float16_t>) {
    if constexpr (std::unsigned_integral<To>)
      return T(static_cast<float>(lane % 5) * 1.25F);
    return T(static_cast<float>((lane % 5) - 2) * 1.25F);
  } else if constexpr (std::floating_point<T>) {
    if constexpr (std::unsigned_integral<To>)
      return static_cast<T>(lane % 5) * static_cast<T>(1.25);
    return static_cast<T>((lane % 5) - 2) * static_cast<T>(1.25);
  } else if constexpr (std::signed_integral<T>) {
    if (lane == 0) return std::numeric_limits<T>::lowest();
    if (lane == 1) return std::numeric_limits<T>::max();
    return static_cast<T>((lane % 5) - 2);
  } else {
    if (lane == 0) return std::numeric_limits<T>::max();
    return static_cast<T>(lane * 37 + 1);
  }
}

template <typename From, vec::Element To>
void verify_ordered_conversion(To) {
  SCOPED_TRACE(::testing::Message()
               << "From=" << typeid(From).name()
               << ", To=" << typeid(To).name());
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using FromTag = vec::ScalableTag<From, -1>;
#else
  using FromTag = vec::FixedTag<From, 8>;
#endif
  using ToTag = vec::Rebind<To, FromTag>;
  std::vector<From> input(static_cast<std::size_t>(vec::size(FromTag{})));
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane)
    input[static_cast<std::size_t>(lane)] = conversion_input<From, To>(lane);
  const auto source = vec::load(FromTag{}, input.data());
  const auto output = vec::convert(ToTag{}, FromTag{}, source);
  const auto named = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::ordered, vec::cvt::saturate);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    SCOPED_TRACE(::testing::Message() << "lane=" << lane);
    const To expected = ::vecops::convert<To, From>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, output, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, named, lane)));
  }
}

template <typename T>
class VecConversionTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecConversionTest, vec_test::AllElementTypes);

TYPED_TEST(VecConversionTest, OrderedSaturateAllDestinations) {
  using From = TypeParam;
  verify_ordered_conversion<From>(vecops::bfloat16_t{});
  verify_ordered_conversion<From>(vecops::float16_t{});
  verify_ordered_conversion<From>(vecops::float32_t{});
  verify_ordered_conversion<From>(vecops::float64_t{});
  verify_ordered_conversion<From>(vecops::int8_t{});
  verify_ordered_conversion<From>(vecops::uint8_t{});
  verify_ordered_conversion<From>(vecops::int16_t{});
  verify_ordered_conversion<From>(vecops::uint16_t{});
  verify_ordered_conversion<From>(vecops::int32_t{});
  verify_ordered_conversion<From>(vecops::uint32_t{});
  verify_ordered_conversion<From>(vecops::int64_t{});
  verify_ordered_conversion<From>(vecops::uint64_t{});
}

TEST(VecConversionSubnormalTest, Float32ToBFloat16RoundingBoundaries) {
  check_f32_to_bf16_boundaries<0>();
  check_f32_to_bf16_boundaries<-1>();
}

TEST(VecConversionSubnormalTest, BFloat16ToFloat32AllBitPatterns) {
  check_bf16_to_f32_all_bits<0>();
  check_bf16_to_f32_all_bits<-1>();
}

template <typename ToTag, typename FromTag, typename... Options>
concept ConvertsFilteredVector = requires(
    ToTag to, FromTag from, vec::Vec<FromTag> input,
    Options... options) {
  vec::convert(to, from, input, options...);
};

using FilterFromTag = vec::ScalableTag<int16_t, -1>;
using FilterToTag = vec::Rebind<int8_t, FilterFromTag>;
using FilterMask = vec::Mask<FilterToTag>;
static_assert(ConvertsFilteredVector<
              FilterToTag, FilterFromTag,
              vec::opt::Masked<FilterMask>, vec::opt::Zero>);
static_assert(ConvertsFilteredVector<
              FilterToTag, FilterFromTag, vec::cvt::Wrap,
              vec::opt::Masked<FilterMask>,
              vec::opt::ScalarMerge<int8_t>>);
static_assert(!ConvertsFilteredVector<
              FilterToTag, FilterFromTag, FilterMask>);
static_assert(!ConvertsFilteredVector<
              FilterToTag, FilterFromTag,
              vec::opt::Unmasked, vec::opt::Zero>);

TEST(VecConversionPolicyTest, MaskedConversionHonorsPopulationAndWrap) {
  using FromTag = vec::ScalableTag<int16_t, -1>;
  using ToTag = vec::Rebind<int8_t, FromTag>;
  auto source = vec::zeros(FromTag{});
  auto mask = vec::mfalse(ToTag{});
  auto fallback = vec::zeros(ToTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane) {
    source = vec::set(
        FromTag{}, source, lane, static_cast<int16_t>(0x120 + lane));
    mask = vec::set(ToTag{}, mask, lane, lane % 2 == 0);
    fallback = vec::set(
        ToTag{}, fallback, lane, static_cast<int8_t>(-30 + lane));
  }

  const auto zero = vec::convert(
      ToTag{}, FromTag{}, source, vec::opt::masked(mask), vec::opt::zero);
  const auto scalar = vec::convert(
      ToTag{}, FromTag{}, source, vec::opt::merge(int8_t{-7}),
      vec::opt::masked(mask));
  const auto merged = vec::convert(
      ToTag{}, FromTag{}, source, vec::opt::masked(mask),
      vec::opt::merge(fallback));
  const auto wrapped = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::wrap,
      vec::opt::masked(mask), vec::opt::merge(int8_t{11}));
  const auto explicitly_unmasked = vec::convert(
      ToTag{}, FromTag{}, source, vec::opt::unmasked);

  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    const auto input = vec::get(FromTag{}, source, lane);
    const auto saturated = ::vecops::convert<int8_t>(input);
    const bool active = lane % 2 == 0;
    EXPECT_EQ(active ? saturated : int8_t{}, vec::get(ToTag{}, zero, lane));
    EXPECT_EQ(active ? saturated : int8_t{-7},
              vec::get(ToTag{}, scalar, lane));
    EXPECT_EQ(active ? saturated : vec::get(ToTag{}, fallback, lane),
              vec::get(ToTag{}, merged, lane));
    EXPECT_EQ(active ? ::vecops::wrap_convert<int8_t>(input) : int8_t{11},
              vec::get(ToTag{}, wrapped, lane));
    EXPECT_EQ(saturated, vec::get(ToTag{}, explicitly_unmasked, lane));
  }
}

template <typename Narrow, typename Wide>
void verify_lane_widening() {
  static_assert(sizeof(Narrow) < sizeof(Wide));
  using FromTag = vec::ScalableTag<Narrow>;
  using ToTag = vec::ViewAs<Wide, FromTag>;
  constexpr vecops::nint_t ratio = sizeof(Wide) / sizeof(Narrow);
  auto source = vec::zeros(FromTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane)
    source = vec::set(FromTag{}, source, lane,
                      static_cast<Narrow>((lane % 13) - 6));
  const auto phase0 = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        ::vecops::convert<Wide>(vec::get(FromTag{}, source, ratio * lane)),
        vec::get(ToTag{}, phase0, lane)));
  if constexpr (ratio == 2) {
    const auto phase1 = vec::convert(
        ToTag{}, FromTag{}, source, vec::cvt::lane<1>);
    for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
      EXPECT_TRUE(vec_test::values_identical(
          ::vecops::convert<Wide>(vec::get(FromTag{}, source, 2 * lane + 1)),
          vec::get(ToTag{}, phase1, lane)));
  }
}

template <typename Wide, typename Narrow>
void verify_lane_narrowing() {
  static_assert(sizeof(Wide) > sizeof(Narrow));
  using FromTag = vec::ScalableTag<Wide>;
  using ToTag = vec::ViewAs<Narrow, FromTag>;
  constexpr vecops::nint_t ratio = sizeof(Wide) / sizeof(Narrow);
  const Narrow scalar_fallback = static_cast<Narrow>(41);
  auto source = vec::zeros(FromTag{});
  auto vector_fallback = vec::zeros(ToTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane)
    source = vec::set(FromTag{}, source, lane,
                      static_cast<Wide>((lane % 13) - 6));
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
    vector_fallback = vec::set(
        ToTag{}, vector_fallback, lane, static_cast<Narrow>(70 + lane % 11));
  const auto zero = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>, vec::opt::zero);
  const auto scalar = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>,
      vec::opt::merge(scalar_fallback));
  const auto merged = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>,
      vec::opt::merge(vector_fallback));
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    const bool active = lane % ratio == 0;
    const auto converted = ::vecops::convert<Narrow>(
        vec::get(FromTag{}, source, lane / ratio));
    if constexpr (std::is_arithmetic_v<Narrow>) {
      EXPECT_TRUE(vec_test::values_identical(
          active ? converted : Narrow{}, vec::get(ToTag{}, zero, lane)))
          << "lane=" << lane
          << ", expected=" << static_cast<long double>(
                 active ? converted : Narrow{})
          << ", actual=" << static_cast<long double>(
                 vec::get(ToTag{}, zero, lane));
    } else {
      EXPECT_TRUE(vec_test::values_identical(
          active ? converted : Narrow{}, vec::get(ToTag{}, zero, lane)));
    }
    EXPECT_TRUE(vec_test::values_identical(
        active ? converted : scalar_fallback,
        vec::get(ToTag{}, scalar, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        active ? converted : vec::get(ToTag{}, vector_fallback, lane),
        vec::get(ToTag{}, merged, lane)));
  }
  if constexpr (ratio == 2) {
    const auto phase1 = vec::convert(
        ToTag{}, FromTag{}, source, vec::cvt::lane<1>,
        vec::opt::merge(scalar_fallback));
    for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
      const bool active = lane % 2 == 1;
      const auto converted = ::vecops::convert<Narrow>(
          vec::get(FromTag{}, source, lane / 2));
      EXPECT_TRUE(vec_test::values_identical(
          active ? converted : scalar_fallback,
          vec::get(ToTag{}, phase1, lane)));
    }
  }
}

TEST(VecConversionPolicyTest, LaneCoversEveryWidthRatioAndFloatingTypes) {
  verify_lane_widening<int8_t, int16_t>();
  verify_lane_widening<vecops::float16_t, float>();
  verify_lane_widening<int8_t, int32_t>();
  verify_lane_widening<vecops::float16_t, double>();
  verify_lane_widening<int8_t, int64_t>();
  verify_lane_narrowing<int16_t, int8_t>();
  verify_lane_narrowing<float, vecops::float16_t>();
  verify_lane_narrowing<int32_t, int8_t>();
  verify_lane_narrowing<double, vecops::float16_t>();
  verify_lane_narrowing<int64_t, int8_t>();
}

TEST(VecConversionPolicyTest, LaneCoversEveryAdjacentTypePair) {
  vec_test::for_each_element_type([]<typename From>() {
    vec_test::for_each_element_type([]<typename To>() {
      SCOPED_TRACE(::testing::Message()
                   << "From=" << typeid(From).name()
                   << ", To=" << typeid(To).name());
      if constexpr (sizeof(To) == sizeof(From) * 2)
        verify_lane_widening<From, To>();
      else if constexpr (sizeof(From) == sizeof(To) * 2)
        verify_lane_narrowing<From, To>();
    });
  });
}

#if defined(CPU_CAPABILITY_SVE)
template <typename Narrow, typename Wide>
void verify_unordered_matches_adjacent_lanes() {
  static_assert(sizeof(Wide) == sizeof(Narrow) * 2);
  using NarrowTag = vec::ScalableTag<Narrow>;
  using WideTag = vec::Rebind<Wide, NarrowTag>;
  using LaneWideTag = vec::ViewAs<Wide, NarrowTag>;
  using HalfWideTag = vec::Half<WideTag>;
  static_assert(std::same_as<LaneWideTag, HalfWideTag>);

  auto source = vec::zeros(NarrowTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(NarrowTag{}); ++lane)
    source = vec::set(
        NarrowTag{}, source, lane, static_cast<Narrow>((lane % 17) - 8));

  const auto unordered_wide = vec::convert(
      WideTag{}, NarrowTag{}, source, vec::cvt::unordered);
  const auto phase0 = vec::convert(
      LaneWideTag{}, NarrowTag{}, source, vec::cvt::lane<0>);
  const auto phase1 = vec::convert(
      LaneWideTag{}, NarrowTag{}, source, vec::cvt::lane<1>);
  const auto lane_wide = vec::concat(WideTag{}, phase0, phase1);
  for (vecops::nint_t lane = 0; lane < vec::size(WideTag{}); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(WideTag{}, lane_wide, lane),
        vec::get(WideTag{}, unordered_wide, lane)))
        << "widen lane=" << lane;

  const auto lower = vec::lower(WideTag{}, unordered_wide);
  const auto upper = vec::upper(WideTag{}, unordered_wide);
  const auto bottom = vec::convert(
      NarrowTag{}, HalfWideTag{}, lower, vec::cvt::lane<0>, vec::opt::zero);
  const auto lane_narrow = vec::convert(
      NarrowTag{}, HalfWideTag{}, upper, vec::cvt::lane<1>,
      vec::opt::merge(bottom));
  const auto unordered_narrow = vec::convert(
      NarrowTag{}, WideTag{}, unordered_wide, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < vec::size(NarrowTag{}); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(NarrowTag{}, lane_narrow, lane),
        vec::get(NarrowTag{}, unordered_narrow, lane)))
        << "narrow lane=" << lane;

  if constexpr (std::integral<Narrow> && std::integral<Wide>) {
    const auto wrap_bottom = vec::convert(
        NarrowTag{}, HalfWideTag{}, lower, vec::cvt::lane<0>,
        vec::cvt::wrap, vec::opt::zero);
    const auto wrap_lane_narrow = vec::convert(
        NarrowTag{}, HalfWideTag{}, upper, vec::cvt::lane<1>,
        vec::cvt::wrap, vec::opt::merge(wrap_bottom));
    const auto wrap_unordered_narrow = vec::convert(
        NarrowTag{}, WideTag{}, unordered_wide,
        vec::cvt::unordered, vec::cvt::wrap);
    for (vecops::nint_t lane = 0; lane < vec::size(NarrowTag{}); ++lane)
      EXPECT_EQ(
          vec::get(NarrowTag{}, wrap_lane_narrow, lane),
          vec::get(NarrowTag{}, wrap_unordered_narrow, lane))
          << "wrap narrow lane=" << lane;
  }
}

TEST(VecConversionPolicyTest, UnorderedMatchesPublicLanesForAdjacentPairs) {
  vec_test::for_each_element_type([]<typename Narrow>() {
    vec_test::for_each_element_type([]<typename Wide>() {
      if constexpr (sizeof(Wide) == sizeof(Narrow) * 2)
        verify_unordered_matches_adjacent_lanes<Narrow, Wide>();
    });
  });
}
#endif

template <typename Visitor>
void for_each_unordered_contract_width(Visitor&& visitor) {
  visitor.template operator()<int8_t>();
  visitor.template operator()<int16_t>();
  visitor.template operator()<int32_t>();
  visitor.template operator()<int64_t>();
}

template <typename From, typename Middle, typename To>
void verify_unordered_composition() {
  // Power -1 keeps the complete 1/2/4/8-byte width family representable and
  // crosses the one-word boundary on SVE, where unordered phases are native.
  using BaseTag = vec::ScalableTag<int8_t, -1>;
  using FromTag = vec::Rebind<From, BaseTag>;
  using MiddleTag = vec::Rebind<Middle, BaseTag>;
  using ToTag = vec::Rebind<To, BaseTag>;

  auto source = vec::zeros(FromTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane) {
    // All contract types represent these values exactly, including when a
    // path temporarily narrows before widening again.
    source = vec::set(
        FromTag{}, source, lane, static_cast<From>(lane % 61 + 1));
  }

  const auto direct = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::unordered);
  const auto middle = vec::convert(
      MiddleTag{}, FromTag{}, source, vec::cvt::unordered);
  const auto indirect = vec::convert(
      ToTag{}, MiddleTag{}, middle, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(ToTag{}, direct, lane),
        vec::get(ToTag{}, indirect, lane)))
        << "lane=" << lane;
  }
}

TEST(VecConversionPolicyTest, UnorderedSameWidthMatchesOrdered) {
  using BaseTag = vec::ScalableTag<int8_t, -1>;
  vec_test::for_each_element_type([]<typename From>() {
    vec_test::for_each_element_type([]<typename To>() {
      if constexpr (sizeof(From) == sizeof(To)) {
        using FromTag = vec::Rebind<From, BaseTag>;
        using ToTag = vec::Rebind<To, BaseTag>;
        auto source = vec::zeros(FromTag{});
        for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane)
          source = vec::set(
              FromTag{}, source, lane,
              static_cast<From>(lane % 61 + 1));
        const auto ordered = vec::convert(
            ToTag{}, FromTag{}, source, vec::cvt::ordered);
        const auto unordered = vec::convert(
            ToTag{}, FromTag{}, source, vec::cvt::unordered);
        for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
          EXPECT_TRUE(vec_test::values_identical(
              vec::get(ToTag{}, ordered, lane),
              vec::get(ToTag{}, unordered, lane)))
              << "lane=" << lane;
        }
      }
    });
  });
}

TEST(VecConversionPolicyTest, UnorderedCompositionIsPathIndependent) {
  for_each_unordered_contract_width([]<typename From>() {
    for_each_unordered_contract_width([]<typename Middle>() {
      for_each_unordered_contract_width([]<typename To>() {
        SCOPED_TRACE(::testing::Message()
                     << "From=" << typeid(From).name()
                     << ", Middle=" << typeid(Middle).name()
                     << ", To=" << typeid(To).name());
        verify_unordered_composition<From, Middle, To>();
      });
    });
  });
}

template <typename From, typename To>
void verify_ordered_wrap_pair() {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using FromTag = vec::ScalableTag<
      From, vec_test::details::minimum_scalable_power<From>>;
#else
  using FromTag = vec::FixedTag<From, 8>;
#endif
  using ToTag = vec::Rebind<To, FromTag>;
  auto source = vec::zeros(FromTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane) {
    const auto value = static_cast<From>(
        (static_cast<std::uint64_t>(lane + 1) << (sizeof(To) * 8)) |
        static_cast<std::uint64_t>(lane * 17 + 3));
    source = vec::set(FromTag{}, source, lane, value);
  }
  const auto output = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::ordered, vec::cvt::wrap);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
    EXPECT_EQ(::vecops::wrap_convert<To>(vec::get(FromTag{}, source, lane)),
              vec::get(ToTag{}, output, lane));
}

template <typename From>
void verify_wrap_destinations() {
  if constexpr (sizeof(From) > 1) {
    verify_ordered_wrap_pair<From, int8_t>();
    verify_ordered_wrap_pair<From, uint8_t>();
  }
  if constexpr (sizeof(From) > 2) {
    verify_ordered_wrap_pair<From, int16_t>();
    verify_ordered_wrap_pair<From, uint16_t>();
  }
  if constexpr (sizeof(From) > 4) {
    verify_ordered_wrap_pair<From, int32_t>();
    verify_ordered_wrap_pair<From, uint32_t>();
  }
}

TEST(VecConversionPolicyTest, WrapCoversEveryIntegerNarrowingPair) {
  verify_wrap_destinations<int16_t>();
  verify_wrap_destinations<uint16_t>();
  verify_wrap_destinations<int32_t>();
  verify_wrap_destinations<uint32_t>();
  verify_wrap_destinations<int64_t>();
  verify_wrap_destinations<uint64_t>();
}

template <typename Narrow, typename Wide>
void verify_unordered_round_trip() {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using NarrowTag = vec::ScalableTag<
      Narrow, vec_test::details::minimum_scalable_power<Narrow>>;
#else
  using NarrowTag = vec::FixedTag<Narrow, 8>;
#endif
  using WideTag = vec::Rebind<Wide, NarrowTag>;
  auto original = vec::zeros(NarrowTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(NarrowTag{}); ++lane)
    original = vec::set(NarrowTag{}, original, lane,
                        static_cast<Narrow>((lane % 11) - 5));
  const auto wide = vec::convert(
      WideTag{}, NarrowTag{}, original, vec::cvt::unordered);
  const auto restored = vec::convert(
      NarrowTag{}, WideTag{}, wide, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < vec::size(NarrowTag{}); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(NarrowTag{}, original, lane),
        vec::get(NarrowTag{}, restored, lane)));
}

TEST(VecConversionPolicyTest, UnorderedRoundTripCoversWidthRatiosAndFloats) {
  verify_unordered_round_trip<int8_t, int8_t>();
  verify_unordered_round_trip<int16_t, int32_t>();
  verify_unordered_round_trip<int8_t, int64_t>();
  verify_unordered_round_trip<vecops::float16_t, float>();
  verify_unordered_round_trip<vecops::float16_t, double>();
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionPolicyTest, OrderedWrapKeepsLowBits) {
  using FromTag = vec::FixedTag<int16_t, 8>;
  using ToTag = vec::Rebind<int8_t, FromTag>;
  const std::array<int16_t, 8> input{
      0x123, -1, 0x7f, 0x80, -129, 511, -512, 42};
  const auto result = vec::convert(
      ToTag{}, FromTag{}, vec::load(FromTag{}, input.data()),
      vec::cvt::ordered, vec::cvt::wrap);
  for (vecops::nint_t lane = 0; lane < 8; ++lane)
    EXPECT_EQ(
        ::vecops::wrap_convert<int8_t>(input[static_cast<std::size_t>(lane)]),
        vec::get(ToTag{}, result, lane));
}

TEST(VecConversionPolicyTest, UnorderedRoundTripPreservesValues) {
  using NarrowTag = vec::FixedTag<int8_t, 8>;
  using WideTag = vec::Rebind<int32_t, NarrowTag>;
  const std::array<int8_t, 8> input{-7, -1, 0, 1, 2, 9, 17, 63};
  const auto original = vec::load(NarrowTag{}, input.data());
  const auto wide = vec::convert(
      WideTag{}, NarrowTag{}, original, vec::cvt::unordered);
  const auto restored = vec::convert(
      NarrowTag{}, WideTag{}, wide, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < 8; ++lane)
    EXPECT_EQ(input[static_cast<std::size_t>(lane)],
              vec::get(NarrowTag{}, restored, lane));
}

TEST(VecConversionPolicyTest, LaneWidenSelectsPhase) {
  using FromTag = vec::FixedTag<int8_t, 8>;
  using ToTag = vec::ViewAs<int16_t, FromTag>;
  const std::array<int8_t, 8> input{-8, 11, -6, 13, -4, 15, -2, 17};
  const auto source = vec::load(FromTag{}, input.data());
  const auto even = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>);
  const auto odd = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<1>);
  for (vecops::nint_t lane = 0; lane < 4; ++lane) {
    EXPECT_EQ(input[static_cast<std::size_t>(2 * lane)],
              vec::get(ToTag{}, even, lane));
    EXPECT_EQ(input[static_cast<std::size_t>(2 * lane + 1)],
              vec::get(ToTag{}, odd, lane));
  }
}

TEST(VecConversionPolicyTest, LaneNarrowHonorsPopulationOptions) {
  using FromTag = vec::FixedTag<int16_t, 4>;
  using ToTag = vec::ViewAs<int8_t, FromTag>;
  const std::array<int16_t, 4> input{-8, 13, -4, 17};
  const std::array<int8_t, 8> fallback{31, 32, 33, 34, 35, 36, 37, 38};
  const auto source = vec::load(FromTag{}, input.data());
  const auto fallback_vec = vec::load(ToTag{}, fallback.data());
  const auto zero = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>, vec::opt::zero);
  const auto scalar = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<1>, vec::opt::merge(int8_t{9}));
  const auto merged = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<0>,
      vec::opt::merge(fallback_vec));
  const auto wrapped = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::lane<1>, vec::cvt::wrap,
      vec::opt::merge(int8_t{3}));
  for (vecops::nint_t lane = 0; lane < 4; ++lane) {
    EXPECT_EQ(static_cast<int8_t>(input[static_cast<std::size_t>(lane)]),
              vec::get(ToTag{}, zero, 2 * lane));
    EXPECT_EQ(int8_t{}, vec::get(ToTag{}, zero, 2 * lane + 1));
    EXPECT_EQ(int8_t{9}, vec::get(ToTag{}, scalar, 2 * lane));
    EXPECT_EQ(static_cast<int8_t>(input[static_cast<std::size_t>(lane)]),
              vec::get(ToTag{}, scalar, 2 * lane + 1));
    EXPECT_EQ(fallback[static_cast<std::size_t>(2 * lane + 1)],
              vec::get(ToTag{}, merged, 2 * lane + 1));
    EXPECT_EQ(int8_t{3}, vec::get(ToTag{}, wrapped, 2 * lane));
    EXPECT_EQ(
        ::vecops::wrap_convert<int8_t>(
            input[static_cast<std::size_t>(lane)]),
        vec::get(ToTag{}, wrapped, 2 * lane + 1));
  }
}

template <typename ToTag, typename FromTag, typename... Options>
concept ConvertsVector = requires(
    ToTag to, FromTag from, vec::Vec<FromTag> input,
    Options... options) {
  vec::convert(to, from, input, options...);
};

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
using FromTag = vec::ScalableTag<int32_t, -1>;
#else
using FromTag = vec::FixedTag<int32_t, 8>;
#endif
using ToTag = vec::Rebind<int16_t, FromTag>;
static_assert(ConvertsVector<ToTag, FromTag>);
static_assert(ConvertsVector<
              ToTag, FromTag, vec::cvt::Ordered, vec::cvt::Saturate>);
static_assert(ConvertsVector<ToTag, FromTag, vec::cvt::Unordered>);
static_assert(ConvertsVector<ToTag, FromTag, vec::cvt::Wrap>);
static_assert(!ConvertsVector<ToTag, FromTag, vec::cvt::Ordered, vec::opt::Zero>);
static_assert(!ConvertsVector<
              ToTag, FromTag, vec::cvt::Ordered, vec::cvt::Unordered>);
using LaneFromTag = vec::FixedTag<int16_t, 4>;
using LaneToTag = vec::ViewAs<int8_t, LaneFromTag>;
static_assert(ConvertsVector<
              LaneToTag, LaneFromTag, vec::cvt::Lane<0>, vec::cvt::Wrap,
              vec::opt::ScalarMerge<int8_t>>);
static_assert(!ConvertsVector<
              LaneToTag, LaneFromTag, vec::cvt::Lane<2>>);
using WidenFromTag = vec::FixedTag<int8_t, 8>;
using WidenToTag = vec::ViewAs<int16_t, WidenFromTag>;
static_assert(!ConvertsVector<
              WidenToTag, WidenFromTag, vec::cvt::Lane<0>, vec::opt::Zero>);
#endif

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionPolicyTest, ScalablePolicies) {
  using NaturalFrom = vec::ScalableTag<int8_t, -1>;
  using NaturalTo = vec::Rebind<int16_t, NaturalFrom>;
  const auto original = vec::fill(NaturalFrom{}, int8_t{7});
  const auto wide = vec::convert(
      NaturalTo{}, NaturalFrom{}, original, vec::cvt::unordered);
  const auto restored = vec::convert(
      NaturalFrom{}, NaturalTo{}, wide, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < vec::size(NaturalFrom{}); ++lane)
    EXPECT_EQ(int8_t{7}, vec::get(NaturalFrom{}, restored, lane));

  using LaneFrom = vec::ScalableTag<int8_t>;
  using LaneTo = vec::ViewAs<int16_t, LaneFrom>;
  auto source = vec::zeros(LaneFrom{});
  for (vecops::nint_t lane = 0; lane < vec::size(LaneFrom{}); ++lane)
    source = vec::set(LaneFrom{}, source, lane, static_cast<int8_t>(lane + 1));
  const auto odd = vec::convert(
      LaneTo{}, LaneFrom{}, source, vec::cvt::lane<1>);
  for (vecops::nint_t lane = 0; lane < vec::size(LaneTo{}); ++lane)
    EXPECT_EQ(static_cast<int16_t>(2 * lane + 2),
              vec::get(LaneTo{}, odd, lane));
}
#endif

} // namespace
