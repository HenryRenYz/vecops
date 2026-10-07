// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vecops/vec/ConversionMemory.h"
#include "vecops/util/ScalarConvert.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

template <vec::Populate Population>
void verify_unordered_bfloat16_to_float_tail_population() {
  using Tag = vec::ScalableTag<vecops::float32_t>;
  using MemoryTag = vec::Rebind<vecops::bfloat16_t, Tag>;
  using Request = vec::LoadConvertRequest<
      Tag, vecops::bfloat16_t, vec::Active::First,
      vec::Addressing::Contiguous, Population, vec::mem::Unaligned,
      vec::mem::Temporal, 0, vec::Vec<vec::IndexTag<Tag>>,
      vec::cvt::Unordered, vec::cvt::Saturate, vec::Mask<Tag>>;
  const vecops::nint_t lanes = vec::size(Tag{});
  const vecops::nint_t active = std::max<vecops::nint_t>(1, lanes - 1);
  std::vector<vecops::bfloat16_t> input(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] =
        vecops::bfloat16_t{static_cast<float>(lane) - 17.0f};

  Request request{};
  request.first_count = active;
  auto merge = vec::zeros(Tag{});
  if constexpr (Population == vec::Populate::MergeScalar) {
    request.merge_scalar = -123.0f;
  } else if constexpr (Population == vec::Populate::MergeVector) {
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      merge = vec::set(Tag{}, merge, lane, 100.0f + static_cast<float>(lane));
    request.merge_vector = &merge;
  }

  const auto loaded = vec::details::execute_load_convert_request(
      vec::LoadConvertOp{}, Tag{}, input.data(), request);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const float expected = lane < active
        ? static_cast<float>(input[static_cast<std::size_t>(lane)])
        : Population == vec::Populate::MergeScalar
            ? -123.0f
            : Population == vec::Populate::MergeVector
                ? 100.0f + static_cast<float>(lane)
                : 0.0f;
    EXPECT_FLOAT_EQ(vec::get(Tag{}, loaded, lane), expected)
        << "population=" << static_cast<int>(Population)
        << " lane=" << lane;
  }

  const auto memory_mask = vec::mwhilelt(MemoryTag{}, 0, active);
  const auto logical_mask = vec::convert(Tag{}, MemoryTag{}, memory_mask);
  const auto round_trip = vec::convert(MemoryTag{}, Tag{}, logical_mask);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(vec::get(MemoryTag{}, memory_mask, lane), lane < active);
    EXPECT_EQ(vec::get(Tag{}, logical_mask, lane), lane < active);
    EXPECT_EQ(vec::get(MemoryTag{}, round_trip, lane), lane < active);
  }
}

TEST(VecConversionMemoryTest,
     ResolvedUnorderedBfloat16TailPreservesEveryPopulationMode) {
  verify_unordered_bfloat16_to_float_tail_population<vec::Populate::Zero>();
  verify_unordered_bfloat16_to_float_tail_population<
      vec::Populate::MergeScalar>();
  verify_unordered_bfloat16_to_float_tail_population<
      vec::Populate::MergeVector>();
}

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionMemoryTest, OversizedRebindPrefersSVEOrderedLowering) {
  using Tag = vec::ScalableTag<int8_t, VEC_MAX_POW>;
  static_assert(
      !vec::details::memory_rebind_supported<Tag, double>());
  static_assert(
      vec::details::has_oversized_memory_conversion_lowering_v<
          vec::details::SVEBackend, vec::LoadConvertOp, Tag, double>);
  static_assert(
      vec::details::has_oversized_memory_conversion_lowering_v<
          vec::details::SVEBackend, vec::StoreConvertOp, Tag, double,
          vec::cvt::Ordered, vec::cvt::Saturate,
          vec::opt::Unmasked>);
  static_assert(
      !vec::details::has_oversized_memory_conversion_lowering_v<
          vec::details::SVEBackend, vec::LoadConvertOp, Tag, double,
          vec::cvt::Unordered>);
  static_assert(
      !vec::details::has_oversized_memory_conversion_lowering_v<
          vec::details::SVEBackend, vec::StoreConvertOp, Tag, double,
          decltype(vec::strided(2))>);
  SUCCEED();
}
#endif

TEST(VecConversionMemoryTest, RebindBeyondBackendMaximumRecursivelySplits) {
  using Tag = vec::ScalableTag<int8_t, VEC_MAX_POW>;
  Tag tag{};
  const vecops::nint_t lanes = vec::size(tag);
  std::vector<double> input(static_cast<std::size_t>(lanes));
  std::vector<double> output(static_cast<std::size_t>(lanes), -1.0);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    input[static_cast<std::size_t>(lane)] =
        static_cast<double>((lane % 101) - 50);
  }

  const auto loaded = vec::load_convert(tag, input.data());
  vec::store_convert(tag, output.data(), loaded);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const auto expected = static_cast<int8_t>((lane % 101) - 50);
    EXPECT_EQ(vec::get(tag, loaded, lane), expected);
    EXPECT_DOUBLE_EQ(
        output[static_cast<std::size_t>(lane)],
        static_cast<double>(expected));
  }

  const vecops::nint_t active = lanes - 3;
  const auto tail = vec::load_convert(
      tag, input.data(), vec::opt::first(active),
      vec::opt::merge(int8_t{77}));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(
        vec::get(tag, tail, lane),
        lane < active ? static_cast<int8_t>((lane % 101) - 50)
                      : int8_t{77});
  }

  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    mask = vec::set(tag, mask, lane, lane % 3 == 0);
  }
  const auto unordered = vec::load_convert(
      tag, input.data(), vec::cvt::unordered, vec::opt::masked(mask));
  output.assign(static_cast<std::size_t>(lanes), -1.0);
  vec::store_convert(
      tag, output.data(), unordered, vec::cvt::unordered,
      vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const auto expected = static_cast<int8_t>((lane % 101) - 50);
    EXPECT_EQ(
        vec::get(tag, unordered, lane), lane % 3 == 0 ? expected : int8_t{});
    EXPECT_DOUBLE_EQ(
        output[static_cast<std::size_t>(lane)],
        lane % 3 == 0 ? static_cast<double>(expected) : -1.0);
  }
}

template <typename From, typename To>
From memory_conversion_input(vecops::nint_t lane) {
  if constexpr (std::same_as<From, vecops::bfloat16_t> ||
                std::same_as<From, vecops::float16_t>) {
    const float value = std::unsigned_integral<To>
        ? static_cast<float>(lane % 5) * 1.25F
        : static_cast<float>((lane % 5) - 2) * 1.25F;
    return From(value);
  } else if constexpr (std::floating_point<From>) {
    const From value = static_cast<From>(lane % 5) * static_cast<From>(1.25);
    if constexpr (std::unsigned_integral<To>) return value;
    else return value - static_cast<From>(2.5);
  } else if constexpr (std::signed_integral<From>) {
    return static_cast<From>((lane % 9) - 4);
  } else {
    return static_cast<From>(lane % 9);
  }
}

template <typename From, vec::Element To>
void verify_default_conversion_memory(To) {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using FromTag = vec::ScalableTag<From, -1>;
#else
  using FromTag = vec::FixedTag<From, 8>;
#endif
  using ToTag = vec::Rebind<To, FromTag>;
  const auto lanes = vec::size(FromTag{});
  std::vector<From> input(static_cast<std::size_t>(lanes));
  std::vector<From> round_trip(static_cast<std::size_t>(lanes));
  std::vector<From> unmasked_round_trip(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] =
        memory_conversion_input<From, To>(lane);

  const auto converted = vec::load_convert(ToTag{}, input.data());
  const auto unmasked_converted = vec::load_convert(
      ToTag{}, input.data(), vec::opt::unmasked);
  vec::store_convert(ToTag{}, round_trip.data(), converted);
  vec::store_convert(
      ToTag{}, unmasked_round_trip.data(), unmasked_converted,
      vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const To expected = ::vecops::convert<To>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, converted, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(ToTag{}, converted, lane),
        vec::get(ToTag{}, unmasked_converted, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        ::vecops::convert<From>(expected),
        round_trip[static_cast<std::size_t>(lane)]));
    EXPECT_TRUE(vec_test::values_identical(
        round_trip[static_cast<std::size_t>(lane)],
        unmasked_round_trip[static_cast<std::size_t>(lane)]));
  }
}

template <typename T>
class VecConversionMemoryTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecConversionMemoryTypedTest, vec_test::AllElementTypes);

TYPED_TEST(VecConversionMemoryTypedTest, DefaultLoadAndStoreAllDestinations) {
  using From = TypeParam;
  verify_default_conversion_memory<From>(vecops::bfloat16_t{});
  verify_default_conversion_memory<From>(vecops::float16_t{});
  verify_default_conversion_memory<From>(vecops::float32_t{});
  verify_default_conversion_memory<From>(vecops::float64_t{});
  verify_default_conversion_memory<From>(vecops::int8_t{});
  verify_default_conversion_memory<From>(vecops::uint8_t{});
  verify_default_conversion_memory<From>(vecops::int16_t{});
  verify_default_conversion_memory<From>(vecops::uint16_t{});
  verify_default_conversion_memory<From>(vecops::int32_t{});
  verify_default_conversion_memory<From>(vecops::uint32_t{});
  verify_default_conversion_memory<From>(vecops::int64_t{});
  verify_default_conversion_memory<From>(vecops::uint64_t{});
}

TEST(VecConversionMemoryTest, AlignmentAndTemporalityAreSoftHints) {
  using Tag = vec::ScalableTag<int32_t>;
  constexpr std::size_t capacity = 256;
  ASSERT_LT(static_cast<std::size_t>(vec::size(Tag{}) + 1), capacity);
  alignas(64) std::array<int16_t, capacity> input{};
  alignas(64) std::array<int16_t, capacity> output{};
  for (std::size_t lane = 0; lane < capacity; ++lane) {
    input[lane] = static_cast<int16_t>(lane * 3 + 1);
    output[lane] = int16_t{-77};
  }

  const auto aligned = vec::load_convert(
      Tag{}, input.data(), vec::mem::aligned, vec::mem::non_temporal);
  const auto unaligned = vec::load_convert(
      Tag{}, input.data() + 1, vec::mem::unaligned,
      vec::mem::non_temporal);
  auto mask = vec::mfalse(Tag{});
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    mask = vec::set(Tag{}, mask, lane, lane % 2 == 0);
  const auto filtered = vec::load_convert(
      Tag{}, input.data(), vec::mem::aligned, vec::mem::non_temporal,
      vec::opt::masked(mask), vec::opt::merge(int32_t{-9}));

  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(input[static_cast<std::size_t>(lane)],
              vec::get(Tag{}, aligned, lane));
    EXPECT_EQ(input[static_cast<std::size_t>(lane + 1)],
              vec::get(Tag{}, unaligned, lane));
    EXPECT_EQ(lane % 2 == 0 ? input[static_cast<std::size_t>(lane)] : -9,
              vec::get(Tag{}, filtered, lane));
  }

  vec::store_convert(
      Tag{}, output.data(), aligned, vec::mem::aligned,
      vec::mem::non_temporal);
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(input[static_cast<std::size_t>(lane)],
              output[static_cast<std::size_t>(lane)]);

  output.fill(int16_t{-77});
  vec::store_convert(
      Tag{}, output.data(), filtered, vec::mem::aligned,
      vec::mem::non_temporal, vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const int16_t expected = lane % 2 == 0
        ? input[static_cast<std::size_t>(lane)]
        : int16_t{-77};
    EXPECT_EQ(expected, output[static_cast<std::size_t>(lane)]);
  }
}

TEST(VecConversionMemoryTest, IdentityForwardsOrdinaryMemoryOptions) {
  using Tag = vec::ScalableTag<int32_t>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  constexpr std::size_t capacity = 4096;
  alignas(64) std::array<int32_t, capacity> input{};
  alignas(64) std::array<int32_t, capacity> output{};
  alignas(64) std::array<int32_t, capacity> index_values{};
  ASSERT_LE(static_cast<std::size_t>(vec::size(Tag{}) * 3), capacity);
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = static_cast<int32_t>(i * 7 + 3);
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    index_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(lane * 2 + 1);

  const auto active = std::max<vecops::nint_t>(0, vec::size(Tag{}) - 2);
  const auto tail = vec::load_convert(
      Tag{}, input.data(), vec::cvt::ordered, vec::cvt::saturate,
      vec::mem::aligned, vec::mem::non_temporal,
      vec::opt::first(active), vec::opt::merge(int32_t{-17}));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(
        lane < active ? input[static_cast<std::size_t>(lane)] : -17,
        vec::get(Tag{}, tail, lane));
  }

  output.fill(-1);
  vec::store_convert(
      Tag{}, output.data(), tail, vec::cvt::ordered, vec::cvt::saturate,
      vec::mem::split, vec::mem::aligned, vec::mem::non_temporal,
      vec::opt::first(active));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(
        lane < active ? input[static_cast<std::size_t>(lane)] : -1,
        output[static_cast<std::size_t>(lane)]);
  }

  const auto indices = vec::load(IndexTag{}, index_values.data());
  auto mask = vec::mfalse(Tag{});
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    mask = vec::set(Tag{}, mask, lane, lane % 2 == 0);
  const auto gathered = vec::load_convert(
      Tag{}, input.data(), vec::cvt::unordered, vec::cvt::saturate,
      vec::mem::non_temporal, vec::opt::masked(mask),
      vec::indexed(indices));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = lane % 2 == 0
        ? input[static_cast<std::size_t>(lane * 2 + 1)] : int32_t{};
    EXPECT_EQ(expected, vec::get(Tag{}, gathered, lane));
  }

  output.fill(-1);
  vec::store_convert(
      Tag{}, output.data(), gathered, vec::cvt::unordered,
      vec::cvt::saturate, vec::mem::packed, vec::mem::non_temporal,
      vec::opt::masked(mask), vec::indexed(indices));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto index = static_cast<std::size_t>(lane * 2 + 1);
    EXPECT_EQ(lane % 2 == 0 ? input[index] : -1, output[index]);
  }
}

TEST(VecConversionMemoryTest, IndexedAndStridedAddressingConvertsAndFilters) {
  using Tag = vec::ScalableTag<int32_t, -1>;
  using I32Tag = vec::Rebind<int32_t, Tag>;
  using I64Tag = vec::Rebind<int64_t, Tag>;
  constexpr std::size_t capacity = 4096;
  std::array<int8_t, capacity> input{};
  std::array<int8_t, capacity> output{};
  std::array<int32_t, capacity> index32_values{};
  std::array<int64_t, capacity> index64_values{};
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = static_cast<int8_t>((static_cast<int>(i) % 101) - 50);
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto index = lane * 5 + 3;
    index32_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(index);
    index64_values[static_cast<std::size_t>(lane)] =
        static_cast<int64_t>(index);
  }
  const auto indices32 = vec::load(I32Tag{}, index32_values.data());
  const auto indices64 = vec::load(I64Tag{}, index64_values.data());
  auto mask = vec::mfalse(Tag{});
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    mask = vec::set(Tag{}, mask, lane, lane % 2 == 0);

  const auto by_i32 = vec::load_convert(
      Tag{}, input.data(), vec::indexed(indices32));
  const auto by_i32_nt = vec::load_convert(
      Tag{}, input.data(), vec::indexed(indices32),
      vec::mem::non_temporal);
  const auto by_i64 = vec::load_convert(
      Tag{}, input.data(), vec::indexed(indices64));
  const auto by_stride = vec::load_convert(
      Tag{}, input.data(), vec::strided(5));
  const auto filtered = vec::load_convert(
      Tag{}, input.data(), vec::indexed(indices32),
      vec::opt::masked(mask), vec::opt::merge(int32_t{77}));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = static_cast<int32_t>(
        input[static_cast<std::size_t>(index32_values[lane])]);
    EXPECT_EQ(expected, vec::get(Tag{}, by_i32, lane));
    EXPECT_EQ(expected, vec::get(Tag{}, by_i32_nt, lane));
    EXPECT_EQ(expected, vec::get(Tag{}, by_i64, lane));
    EXPECT_EQ(static_cast<int32_t>(input[static_cast<std::size_t>(lane * 5)]),
              vec::get(Tag{}, by_stride, lane));
    EXPECT_EQ(lane % 2 == 0 ? expected : 77,
              vec::get(Tag{}, filtered, lane));
  }

  output.fill(int8_t{-99});
  vec::store_convert(
      Tag{}, output.data(), by_i32, vec::indexed(indices32),
      vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = lane % 2 == 0
        ? static_cast<int8_t>(vec::get(Tag{}, by_i32, lane))
        : int8_t{-99};
    EXPECT_EQ(expected,
              output[static_cast<std::size_t>(index32_values[lane])]);
  }

  output.fill(int8_t{-99});
  vec::store_convert(
      Tag{}, output.data(), by_i32, vec::indexed(indices32),
      vec::mem::non_temporal, vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = lane % 2 == 0
        ? static_cast<int8_t>(vec::get(Tag{}, by_i32, lane))
        : int8_t{-99};
    EXPECT_EQ(expected,
              output[static_cast<std::size_t>(index32_values[lane])]);
  }

  output.fill(int8_t{-99});
  vec::store_convert(
      Tag{}, output.data(), by_i32, vec::strided(5),
      vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = lane % 2 == 0
        ? static_cast<int8_t>(vec::get(Tag{}, by_i32, lane))
        : int8_t{-99};
    EXPECT_EQ(expected, output[static_cast<std::size_t>(lane * 5)]);
  }
}

template <vec::Element From, vec::Element To>
void verify_indexed_load_convert_pair() {
  using Tag = vec::ScalableTag<To>;
  using I32Tag = vec::Rebind<int32_t, Tag>;
  using I64Tag = vec::Rebind<int64_t, Tag>;
  constexpr std::size_t capacity = 1024;
  std::array<From, capacity> input{};
  std::array<int32_t, capacity> i32_values{};
  std::array<int64_t, capacity> i64_values{};
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = memory_conversion_input<From, To>(static_cast<vecops::nint_t>(i));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto index = lane * 3 + 1;
    i32_values[static_cast<std::size_t>(lane)] = static_cast<int32_t>(index);
    i64_values[static_cast<std::size_t>(lane)] = static_cast<int64_t>(index);
  }
  const auto i32 = vec::load(I32Tag{}, i32_values.data());
  const auto i64 = vec::load(I64Tag{}, i64_values.data());
  const auto by_i32 = vec::load_convert(Tag{}, input.data(), vec::indexed(i32));
  const auto by_i64 = vec::load_convert(Tag{}, input.data(), vec::indexed(i64));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = ::vecops::convert<To>(
        input[static_cast<std::size_t>(i32_values[lane])]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(Tag{}, by_i32, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(Tag{}, by_i64, lane)));
  }
}

template <vec::Element From, vec::Element To>
void verify_indexed_store_convert_pair() {
  SCOPED_TRACE(::testing::Message()
               << "From=" << typeid(From).name()
               << ", To=" << typeid(To).name());
  using Tag = vec::ScalableTag<From>;
  using I32Tag = vec::Rebind<int32_t, Tag>;
  using I64Tag = vec::Rebind<int64_t, Tag>;
  constexpr std::size_t capacity = 1024;
  std::array<From, capacity> input{};
  std::array<To, capacity> output32{};
  std::array<To, capacity> output64{};
  std::array<int32_t, capacity> i32_values{};
  std::array<int64_t, capacity> i64_values{};
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    input[static_cast<std::size_t>(lane)] =
        memory_conversion_input<From, To>(lane);
    const auto index = lane * 3 + 1;
    i32_values[static_cast<std::size_t>(lane)] = static_cast<int32_t>(index);
    i64_values[static_cast<std::size_t>(lane)] = static_cast<int64_t>(index);
  }
  const auto value = vec::load(Tag{}, input.data());
  const auto i32 = vec::load(I32Tag{}, i32_values.data());
  const auto i64 = vec::load(I64Tag{}, i64_values.data());
  vec::store_convert(Tag{}, output32.data(), value, vec::indexed(i32));
  vec::store_convert(Tag{}, output64.data(), value, vec::indexed(i64));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const auto expected = ::vecops::convert<To>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, output32[static_cast<std::size_t>(i32_values[lane])]));
    EXPECT_TRUE(vec_test::values_identical(
        expected, output64[static_cast<std::size_t>(i64_values[lane])]))
        << "lane=" << lane;
  }
}

TEST(VecConversionMemoryTest, IndexedSubwordConversionsCoverWidthPairs) {
  verify_indexed_load_convert_pair<int8_t, int16_t>();
  verify_indexed_load_convert_pair<int8_t, uint16_t>();
  verify_indexed_load_convert_pair<int8_t, int32_t>();
  verify_indexed_load_convert_pair<int8_t, uint32_t>();
  verify_indexed_load_convert_pair<int8_t, int64_t>();
  verify_indexed_load_convert_pair<int8_t, uint64_t>();
  verify_indexed_load_convert_pair<uint8_t, int16_t>();
  verify_indexed_load_convert_pair<uint8_t, uint16_t>();
  verify_indexed_load_convert_pair<uint8_t, int32_t>();
  verify_indexed_load_convert_pair<uint8_t, uint32_t>();
  verify_indexed_load_convert_pair<uint8_t, int64_t>();
  verify_indexed_load_convert_pair<uint8_t, uint64_t>();
  verify_indexed_load_convert_pair<int16_t, int32_t>();
  verify_indexed_load_convert_pair<int16_t, uint32_t>();
  verify_indexed_load_convert_pair<int16_t, int64_t>();
  verify_indexed_load_convert_pair<int16_t, uint64_t>();
  verify_indexed_load_convert_pair<uint16_t, int32_t>();
  verify_indexed_load_convert_pair<uint16_t, uint32_t>();
  verify_indexed_load_convert_pair<uint16_t, int64_t>();
  verify_indexed_load_convert_pair<uint16_t, uint64_t>();
  verify_indexed_load_convert_pair<vecops::float16_t, float>();
  verify_indexed_load_convert_pair<vecops::bfloat16_t, float>();

  verify_indexed_store_convert_pair<int16_t, int8_t>();
  verify_indexed_store_convert_pair<int16_t, uint8_t>();
  verify_indexed_store_convert_pair<uint16_t, int8_t>();
  verify_indexed_store_convert_pair<uint16_t, uint8_t>();
  verify_indexed_store_convert_pair<int32_t, int8_t>();
  verify_indexed_store_convert_pair<int32_t, uint8_t>();
  verify_indexed_store_convert_pair<int32_t, int16_t>();
  verify_indexed_store_convert_pair<int32_t, uint16_t>();
  verify_indexed_store_convert_pair<uint32_t, int8_t>();
  verify_indexed_store_convert_pair<uint32_t, uint8_t>();
  verify_indexed_store_convert_pair<uint32_t, int16_t>();
  verify_indexed_store_convert_pair<uint32_t, uint16_t>();
  verify_indexed_store_convert_pair<int64_t, int8_t>();
  verify_indexed_store_convert_pair<int64_t, uint8_t>();
  verify_indexed_store_convert_pair<int64_t, int16_t>();
  verify_indexed_store_convert_pair<int64_t, uint16_t>();
  verify_indexed_store_convert_pair<uint64_t, int8_t>();
  verify_indexed_store_convert_pair<uint64_t, uint8_t>();
  verify_indexed_store_convert_pair<uint64_t, int16_t>();
  verify_indexed_store_convert_pair<uint64_t, uint16_t>();
  verify_indexed_store_convert_pair<float, vecops::float16_t>();
  verify_indexed_store_convert_pair<float, vecops::bfloat16_t>();
}

template <typename F16>
void verify_filtered_indexed_f16_f32_round_trip() {
  using Tag = vec::ScalableTag<float>;
  using MemoryTag = vec::Rebind<F16, Tag>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  constexpr std::size_t capacity = 4096;
  std::array<F16, capacity> input{};
  std::array<F16, capacity> indexed_output{};
  std::array<F16, capacity> strided_output{};
  std::array<int32_t, capacity> index_values{};
  const auto lanes = vec::size(Tag{});
  const auto first = std::max<vecops::nint_t>(1, lanes / 2);
  const F16 canary{-91.0F};
  indexed_output.fill(canary);
  strided_output.fill(canary);
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = F16(static_cast<float>(static_cast<int>(i % 31) - 15) / 8.0F);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    index_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(lane * 3 + 1);

  const auto indices = vec::load(IndexTag{}, index_values.data());
  auto mask = vec::mfalse(Tag{});
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    mask = vec::set(Tag{}, mask, lane, lane % 2 == 0);
  const auto gathered = vec::load_convert(
      Tag{}, input.data(), vec::indexed(indices), vec::opt::masked(mask),
      vec::opt::merge(-77.0F), vec::mem::non_temporal);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const float expected = lane % 2 == 0
        ? static_cast<float>(input[static_cast<std::size_t>(index_values[lane])])
        : -77.0F;
    EXPECT_EQ(expected, vec::get(Tag{}, gathered, lane));
  }

  auto memory_mask = vec::mfalse(MemoryTag{});
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    memory_mask = vec::set(MemoryTag{}, memory_mask, lane, lane % 2 == 0);
  const auto unordered = vec::load_convert(
      Tag{}, input.data(), vec::cvt::unordered, vec::indexed(indices),
      vec::opt::masked(memory_mask));
  const auto unordered_all = vec::load_convert(
      Tag{}, input.data(), vec::cvt::unordered, vec::indexed(indices));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const float loaded =
        static_cast<float>(input[static_cast<std::size_t>(index_values[lane])]);
    const float expected = lane % 2 == 0
        ? loaded : 0.0F;
    EXPECT_EQ(expected, vec::get(Tag{}, unordered, lane));
    EXPECT_EQ(loaded, vec::get(Tag{}, unordered_all, lane));
  }

  vec::store_convert(
      Tag{}, indexed_output.data(), gathered, vec::indexed(indices),
      vec::opt::masked(mask), vec::mem::non_temporal);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const auto index = static_cast<std::size_t>(index_values[lane]);
    const F16 expected = lane % 2 == 0
        ? ::vecops::convert<F16>(vec::get(Tag{}, gathered, lane)) : canary;
    EXPECT_TRUE(vec_test::values_identical(expected, indexed_output[index]))
        << "lane=" << lane;
  }

  indexed_output.fill(canary);
  vec::store_convert(
      Tag{}, indexed_output.data(), gathered, vec::cvt::unordered,
      vec::indexed(indices), vec::opt::masked(memory_mask));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const auto index = static_cast<std::size_t>(index_values[lane]);
    const F16 expected = lane % 2 == 0
        ? ::vecops::convert<F16>(vec::get(Tag{}, gathered, lane)) : canary;
    EXPECT_TRUE(vec_test::values_identical(expected, indexed_output[index]))
        << "unordered lane=" << lane;
  }

  indexed_output.fill(canary);
  vec::store_convert(
      Tag{}, indexed_output.data(), unordered_all, vec::cvt::unordered,
      vec::indexed(indices));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const auto index = static_cast<std::size_t>(index_values[lane]);
    EXPECT_TRUE(vec_test::values_identical(
        ::vecops::convert<F16>(vec::get(Tag{}, unordered_all, lane)),
        indexed_output[index])) << "unordered unmasked lane=" << lane;
  }

  auto source = vec::fill(Tag{}, 3.25F);
  vec::store_convert(
      Tag{}, strided_output.data(), source, vec::strided(1),
      vec::opt::first(first));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const F16 expected = lane < first ? F16{3.25F} : canary;
    EXPECT_TRUE(vec_test::values_identical(
        expected, strided_output[static_cast<std::size_t>(lane)]))
        << "lane=" << lane;
  }
}

TEST(VecConversionMemoryTest, FilteredIndexedF16F32ConversionsPreserveAddresses) {
  verify_filtered_indexed_f16_f32_round_trip<vecops::float16_t>();
  verify_filtered_indexed_f16_f32_round_trip<vecops::bfloat16_t>();
}

TEST(VecConversionMemoryTest, WideOrderedF32ToBf16StorePreservesLaneOrder) {
  using FromTag = vec::ScalableTag<float, 2>;
  const auto lanes = vec::size(FromTag{});
  std::vector<float> input(static_cast<std::size_t>(lanes));
  std::vector<vecops::bfloat16_t> output(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] =
        static_cast<float>(lane) + static_cast<float>(lane % 7) / 8.0F;

  const auto value = vec::load(FromTag{}, input.data());
  vec::store_convert(FromTag{}, output.data(), value);

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        vecops::convert<vecops::bfloat16_t>(
            input[static_cast<std::size_t>(lane)]),
        output[static_cast<std::size_t>(lane)]))
        << "lane=" << lane;
  }
}

TEST(VecConversionMemoryTest, WideOrderedIntegerNarrowStorePreservesLaneOrder) {
  using FromTag = vec::ScalableTag<int32_t, 1>;
  const auto lanes = vec::size(FromTag{});
  std::vector<int32_t> input(static_cast<std::size_t>(lanes));
  std::vector<int16_t> output(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    input[static_cast<std::size_t>(lane)] =
        lane % 5 == 0 ? 100000 :
        lane % 5 == 1 ? -100000 : static_cast<int32_t>(lane * 37 - 400);
  }

  const auto value = vec::load(FromTag{}, input.data());
  vec::store_convert(FromTag{}, output.data(), value);

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(
        ::vecops::convert<int16_t>(input[static_cast<std::size_t>(lane)]),
        output[static_cast<std::size_t>(lane)])
        << "lane=" << lane;
  }
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionMemoryTest, OrderedSaturatingLoadPopulationAndStoreFilter) {
  using ToTag = vec::FixedTag<int16_t, 8>;
  const std::array<int32_t, 8> input{
      -100000, -31, 0, 17, 32767, 32768, 90000, -32769};
  const auto mask = [] {
    auto result = vec::mfalse(ToTag{});
    for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
      result = vec::set(ToTag{}, result, lane, lane % 3 != 1);
    return result;
  }();
  auto fallback = vec::fill(ToTag{}, int16_t{73});
  fallback = vec::set(ToTag{}, fallback, 6, int16_t{91});

  const auto zero = vec::load_convert(
      ToTag{}, input.data(), vec::opt::masked(mask), vec::opt::zero);
  const auto scalar = vec::load_convert(
      ToTag{}, input.data(), vec::opt::first(5),
      vec::opt::merge(int16_t{-9}));
#if defined(CPU_CAPABILITY_SVE)
  const auto merged = vec::load_convert(
      ToTag{}, input.data(), vec::opt::merge(fallback),
      vec::opt::masked(mask), vec::mem::non_temporal);
#else
  const auto merged = vec::load_convert(
      ToTag{}, input.data(), vec::opt::merge(fallback),
      vec::opt::masked(mask));
#endif

  for (vecops::nint_t lane = 0; lane < 8; ++lane) {
    const int16_t converted = ::vecops::convert<int16_t>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_EQ(lane % 3 != 1 ? converted : int16_t{},
              vec::get(ToTag{}, zero, lane));
    EXPECT_EQ(lane < 5 ? converted : int16_t{-9},
              vec::get(ToTag{}, scalar, lane));
    EXPECT_EQ(lane % 3 != 1 ? converted : vec::get(ToTag{}, fallback, lane),
              vec::get(ToTag{}, merged, lane));
  }

  std::array<int32_t, 8> round_trip{};
  round_trip.fill(1234567);
  vec::store_convert(
      ToTag{}, round_trip.data(), merged, vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < 8; ++lane) {
    const int32_t expected = lane % 3 != 1
        ? static_cast<int32_t>(vec::get(ToTag{}, merged, lane)) : 1234567;
    EXPECT_EQ(expected, round_trip[static_cast<std::size_t>(lane)]);
  }
}

TEST(VecConversionMemoryTest, WrapAndSplitStorePreserveLogicalOrder) {
  using FromTag = vec::FixedTag<int32_t, 8>;
  using ToTag = vec::Rebind<int8_t, FromTag>;
  const std::array<int32_t, 8> input{
      0x123, -1, 0x7f, 0x80, -129, 511, -512, 42};
  const auto source = vec::load(FromTag{}, input.data());
  std::array<int8_t, 8> wrapped{};
  std::array<int8_t, 8> split{};
  vec::store_convert(
      FromTag{}, wrapped.data(), source, vec::cvt::wrap);
  vec::store_convert(
      FromTag{}, split.data(), source, vec::mem::split);
  for (vecops::nint_t lane = 0; lane < 8; ++lane) {
    EXPECT_EQ(::vecops::wrap_convert<int8_t>(
                  input[static_cast<std::size_t>(lane)]),
              wrapped[static_cast<std::size_t>(lane)]);
    EXPECT_EQ(::vecops::convert<int8_t>(
                  input[static_cast<std::size_t>(lane)]),
              split[static_cast<std::size_t>(lane)]);
  }

  const auto loaded = vec::load_convert(
      ToTag{}, input.data(), vec::cvt::wrap);
  for (vecops::nint_t lane = 0; lane < 8; ++lane)
    EXPECT_EQ(wrapped[static_cast<std::size_t>(lane)],
              vec::get(ToTag{}, loaded, lane));
}

TEST(VecConversionMemoryTest, UnorderedUsesMemorySideMaskAndRoundTrips) {
  using NarrowTag = vec::FixedTag<vecops::float16_t, 8>;
  using WideTag = vec::Rebind<float, NarrowTag>;
  std::array<vecops::float16_t, 8> input{};
  std::array<vecops::float16_t, 8> output{};
  for (vecops::nint_t lane = 0; lane < 8; ++lane) {
    input[static_cast<std::size_t>(lane)] =
        vecops::float16_t(static_cast<float>(lane) - 3.0F);
    output[static_cast<std::size_t>(lane)] = vecops::float16_t(-91.0F);
  }
  const auto mask = vec::mwhilelt(NarrowTag{}, 0, 5);
  const auto value = vec::load_convert(
      WideTag{}, input.data(), vec::cvt::unordered,
      vec::opt::masked(mask));
  vec::store_convert(
      WideTag{}, output.data(), value, vec::cvt::unordered,
      vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < 8; ++lane) {
    const auto expected = lane < 5
        ? input[static_cast<std::size_t>(lane)] : vecops::float16_t(-91.0F);
    EXPECT_TRUE(vec_test::values_identical(
        expected, output[static_cast<std::size_t>(lane)]));
  }
}

TEST(VecConversionMemoryTest, FilteredLoadAndStoreDoNotCrossGuardPage) {
  using LoadTag = vec::FixedTag<int32_t, 8>;
  using StoreTag = vec::FixedTag<int32_t, 8>;
  constexpr vecops::nint_t active = 3;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  void* mapping = mmap(
      nullptr, static_cast<std::size_t>(page_size * 2),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(
      static_cast<std::byte*>(mapping) + page_size,
      static_cast<std::size_t>(page_size), PROT_NONE), 0);

  auto* load_pointer = reinterpret_cast<int16_t*>(
      static_cast<std::byte*>(mapping) + page_size) - active;
  for (vecops::nint_t lane = 0; lane < active; ++lane)
    load_pointer[lane] = static_cast<int16_t>(lane + 11);
  const auto loaded = vec::load_convert(
      LoadTag{}, load_pointer, vec::opt::first(active));
  for (vecops::nint_t lane = 0; lane < 8; ++lane)
    EXPECT_EQ(lane < active ? lane + 11 : 0,
              vec::get(LoadTag{}, loaded, lane));

  auto source = vec::fill(StoreTag{}, int32_t{37});
  auto* store_pointer = reinterpret_cast<int16_t*>(
      static_cast<std::byte*>(mapping) + page_size) - active;
  vec::store_convert(
      StoreTag{}, store_pointer, source, vec::opt::first(active));
  for (vecops::nint_t lane = 0; lane < active; ++lane)
    EXPECT_EQ(37, store_pointer[lane]);
  EXPECT_EQ(munmap(mapping, static_cast<std::size_t>(page_size * 2)), 0);
}

template <typename Memory, typename Vector>
void verify_conversion_guard_pair() {
  using Tag = vec::FixedTag<Vector, 8>;
  constexpr vecops::nint_t active = 3;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  void* mapping = mmap(
      nullptr, static_cast<std::size_t>(page_size * 2),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(static_cast<std::byte*>(mapping) + page_size,
                     static_cast<std::size_t>(page_size), PROT_NONE), 0);
  auto* protected_pointer = reinterpret_cast<Memory*>(
      static_cast<std::byte*>(mapping) + page_size);
  const auto empty = vec::load_convert(
      Tag{}, protected_pointer, vec::opt::first(0));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        Vector{}, vec::get(Tag{}, empty, lane)));
  auto* pointer = protected_pointer - active;
  for (vecops::nint_t lane = 0; lane < active; ++lane)
    pointer[lane] = static_cast<Memory>(lane + 11);
  const auto loaded = vec::load_convert(
      Tag{}, pointer, vec::opt::first(active));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    const Vector expected = lane < active
        ? ::vecops::convert<Vector>(pointer[lane]) : Vector{};
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(Tag{}, loaded, lane)));
  }
  vec::store_convert(Tag{}, pointer, loaded, vec::opt::first(active));
  vec::store_convert(Tag{}, protected_pointer, loaded, vec::opt::first(0));
  EXPECT_EQ(munmap(mapping, static_cast<std::size_t>(page_size * 2)), 0);
}

TEST(VecConversionMemoryTest, GuardPageCoversEqualWidenAndNarrow) {
  verify_conversion_guard_pair<int32_t, int32_t>();
  verify_conversion_guard_pair<int16_t, int32_t>();
  verify_conversion_guard_pair<int32_t, int16_t>();
}

TEST(VecConversionMemoryTest, FirstCoversEveryBoundaryCount) {
  using Tag = vec::ScalableTag<int32_t>;
  using Memory = int16_t;
  constexpr std::size_t capacity = 4096;
  std::array<Memory, capacity> input{};
  std::array<Memory, capacity> output{};
  for (std::size_t lane = 0; lane < capacity; ++lane)
    input[lane] = static_cast<Memory>(lane * 3 + 1);
  const std::array<vecops::nint_t, 4> counts{
      0, 1, std::max<vecops::nint_t>(0, vec::size(Tag{}) - 1),
      vec::size(Tag{})};
  for (const auto count : counts) {
    const auto loaded = vec::load_convert(
        Tag{}, input.data(), vec::opt::first(count),
        vec::opt::merge(int32_t{-71}));
    for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
      EXPECT_EQ(lane < count ? input[static_cast<std::size_t>(lane)] : -71,
                vec::get(Tag{}, loaded, lane));
    output.fill(Memory{-99});
    vec::store_convert(Tag{}, output.data(), loaded, vec::opt::first(count));
    for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
      const Memory expected = lane < count
          ? ::vecops::convert<Memory>(vec::get(Tag{}, loaded, lane))
          : Memory{-99};
      EXPECT_EQ(expected, output[static_cast<std::size_t>(lane)]);
    }
  }
}

TEST(VecConversionMemoryTest, FirstCountsFollowMwhileltSemantics) {
  using Tag = vec::ScalableTag<int32_t>;
  std::array<int16_t, 4096> storage{};
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    storage[static_cast<std::size_t>(lane)] =
        static_cast<int16_t>(lane + 11);
  const auto empty = vec::load_convert(
      Tag{}, storage.data(), vec::opt::first(-1));
  const auto full = vec::load_convert(
      Tag{}, storage.data(), vec::opt::first(vec::size(Tag{}) + 1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(vec::get(Tag{}, empty, lane), 0);
    EXPECT_EQ(
        vec::get(Tag{}, full, lane),
        static_cast<int32_t>(storage[static_cast<std::size_t>(lane)]));
  }

  std::array<int16_t, 4096> output;
  output.fill(int16_t{-1});
  const auto value = vec::fill(Tag{}, int32_t{37});
  vec::store_convert(Tag{}, output.data(), value, vec::opt::first(-1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(output[static_cast<std::size_t>(lane)], int16_t{-1});
  vec::store_convert(
      Tag{}, output.data(), value,
      vec::opt::first(vec::size(Tag{}) + 1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(output[static_cast<std::size_t>(lane)], int16_t{37});
}

template <typename Tag, typename From, typename... Options>
concept AcceptsLoadConvert = requires(
    Tag tag, const From* pointer, Options... options) {
  vec::load_convert(tag, pointer, options...);
};

template <typename Tag, typename To, typename... Options>
concept AcceptsStoreConvert = requires(
    Tag tag, To* pointer, vec::Vec<Tag> value, Options... options) {
  vec::store_convert(tag, pointer, value, options...);
};

using ConstraintTag = vec::FixedTag<int32_t, 8>;
using ConstraintMemoryTag = vec::Rebind<int16_t, ConstraintTag>;
using ConstraintIndices = vec::Vec<vec::Rebind<int32_t, ConstraintTag>>;
using ConstraintIndexed = vec::opt::Indexed<ConstraintIndices>;
// Cross a physical-word boundary so the negative check remains meaningful on
// fixed-length SVE, where all one-word i32 FixedTags share SVEFixedVector<i32>
// and therefore intentionally erase their subword logical extent.
using WrongConstraintIndices = vec::Vec<vec::FixedTag<int32_t, 32>>;
using WrongConstraintIndexed = vec::opt::Indexed<WrongConstraintIndices>;
static_assert(AcceptsLoadConvert<ConstraintTag, int16_t>);
static_assert(AcceptsLoadConvert<ConstraintTag, int16_t, ConstraintIndexed>);
static_assert(AcceptsStoreConvert<ConstraintTag, int16_t, ConstraintIndexed>);
static_assert(AcceptsLoadConvert<
              ConstraintTag, int16_t, decltype(vec::strided(3))>);
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t, ConstraintIndexed,
              vec::mem::Aligned>);
static_assert(!AcceptsStoreConvert<
              ConstraintTag, int16_t, ConstraintIndexed,
              decltype(vec::strided(3))>);
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t, WrongConstraintIndexed>);
static_assert(AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::cvt::Ordered,
              vec::cvt::Saturate, vec::mem::Aligned,
              vec::mem::NonTemporal>);
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::mem::Split>);
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::cvt::Ordered,
              vec::cvt::Unordered>);
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::opt::Zero>);
static_assert(AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::opt::First,
              vec::opt::ScalarMerge<int32_t>>);
#if !defined(CPU_CAPABILITY_SVE)
// SVE predicates with the same physical shape intentionally share one raw
// type, so a Mask does not retain enough Tag identity for this negative
// compile-time check. Explicit source/destination Tags still drive dispatch.
static_assert(!AcceptsLoadConvert<
              ConstraintTag, int16_t,
              vec::opt::Masked<vec::Mask<ConstraintMemoryTag>>>);
#endif
static_assert(AcceptsLoadConvert<
              ConstraintTag, int16_t, vec::cvt::Unordered,
              vec::opt::Masked<vec::Mask<ConstraintMemoryTag>>>);
static_assert(!AcceptsStoreConvert<
              ConstraintTag, int16_t, vec::opt::Zero>);
static_assert(AcceptsStoreConvert<
              ConstraintTag, int16_t, vec::mem::Split>);
static_assert(AcceptsStoreConvert<
              ConstraintTag, int16_t, vec::mem::Split,
              vec::mem::Aligned>);
static_assert(AcceptsStoreConvert<
              ConstraintTag, int16_t, vec::mem::Split,
              vec::mem::Aligned, vec::mem::NonTemporal>);
static_assert(!AcceptsStoreConvert<
              ConstraintTag, int16_t, vec::mem::Split,
              vec::cvt::Wrap>);
static_assert(!AcceptsStoreConvert<
              ConstraintTag, float, vec::cvt::Wrap>);
#endif

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionMemoryTest, ScalableFilteredRoundTrip) {
  using ToTag = vec::ScalableTag<int32_t, -1>;
  using FromTag = vec::Rebind<int16_t, ToTag>;
  const vecops::nint_t lanes = vec::size(ToTag{});
  const vecops::nint_t active = lanes - 1;
  std::vector<int16_t> input(static_cast<std::size_t>(lanes));
  std::vector<int16_t> output(static_cast<std::size_t>(lanes), int16_t{-77});
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] = static_cast<int16_t>(lane - 9);

  const auto loaded = vec::load_convert(
      ToTag{}, input.data(), vec::opt::first(active),
      vec::opt::merge(int32_t{41}));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    EXPECT_EQ(lane < active ? input[static_cast<std::size_t>(lane)] : 41,
              vec::get(ToTag{}, loaded, lane));

  const auto memory_mask = vec::mwhilelt(FromTag{}, 0, active);
  vec::store_convert(
      ToTag{}, output.data(), loaded, vec::cvt::unordered,
      vec::opt::masked(memory_mask));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    EXPECT_EQ(lane < active ? input[static_cast<std::size_t>(lane)] : -77,
              output[static_cast<std::size_t>(lane)]);
}
#endif

} // namespace
