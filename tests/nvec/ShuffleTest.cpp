#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/nvec/Basic.h"
#include "TestHelpers.h"

namespace vec = vecops::nvec;

template <typename T>
class NVecShuffleElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(NVecShuffleElementTest, nvec_test::AllElementTypes);

template <typename T>
T lane_value(vecops::nint_t lane) {
  if constexpr (std::same_as<T, vecops::bfloat16_t>) {
    return vecops::bfloat16_t(static_cast<float>(lane + 1));
  } else if constexpr (std::same_as<T, vecops::float16_t>) {
    return vecops::float16_t(static_cast<float>(lane + 1));
  } else {
    return static_cast<T>(lane + 1);
  }
}

using nvec_test::values_identical;

template <typename T, typename Visitor>
void for_each_halvable_rearrange_shape(Visitor&& visitor) {
  // The minimum native word is 16 bytes (including the AVX target, whose
  // public word width intentionally remains 128 bits). A 64-bit P=-1 tag has
  // one lane there and therefore cannot be halved again.
  if constexpr (sizeof(T) <= 4) {
    visitor.template operator()<vec::ScalableTag<T, -1>>();
  }
  visitor.template operator()<vec::ScalableTag<T, 0>>();
  visitor.template operator()<vec::ScalableTag<T, 1>>();
  visitor.template operator()<vec::ScalableTag<T, 2>>();
}

template <typename T, typename Visitor>
void for_each_same_width_rearrange_shape(Visitor&& visitor) {
  visitor.template operator()<vec::ScalableTag<T, -1>>();
  visitor.template operator()<vec::ScalableTag<T, 0>>();
  visitor.template operator()<vec::ScalableTag<T, 1>>();
  visitor.template operator()<vec::ScalableTag<T, 2>>();
}

template <vec::VectorTag Tag>
vec::Vec<Tag> make_lane_sequence(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    value = vec::set(tag, value, lane, lane_value<T>(lane));
  }
  return value;
}

template <vec::VectorTag Tag>
vec::Vec<Tag> make_lane_sequence(Tag tag, vecops::nint_t offset) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    value = vec::set(tag, value, lane, lane_value<T>(lane + offset));
  }
  return value;
}

constexpr bool mask_lane(vecops::nint_t lane, vecops::nint_t seed = 0) {
  return ((lane * 5 + seed) % 7) < 3;
}

template <vec::VectorTag Tag>
vec::Mask<Tag> make_mask_pattern(Tag tag, vecops::nint_t seed = 0) {
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    mask = vec::set(tag, mask, lane, mask_lane(lane, seed));
  }
  return mask;
}

template <vec::VectorTag Tag>
void expect_lane(
    Tag tag, vec::Vec<Tag> value, vecops::nint_t lane,
    vec::ElementOf<Tag> expected, const char* operation) {
  EXPECT_TRUE(values_identical(expected, vec::get(tag, value, lane)))
      << operation << " lane=" << lane;
}

template <vec::VectorTag Tag>
void expect_mask_lane(
    Tag tag, vec::Mask<Tag> value, vecops::nint_t lane,
    bool expected, const char* operation) {
  EXPECT_EQ(vec::get(tag, value, lane), expected)
      << operation << " lane=" << lane;
}

template <vec::VectorTag Tag>
void verify_halves_concat_and_selection(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using HalfTag = vec::Half<Tag>;
  const vecops::nint_t lanes = vec::size(tag);
  const vecops::nint_t half = lanes / 2;

  // This can occur only for a minimum-VL SVE subword. The corresponding
  // half descriptor has no runtime lanes, so none of these APIs is defined.
  if (lanes < 2) return;

  const auto a = make_lane_sequence(tag, 0);
  const auto b = make_lane_sequence(tag, lanes + 19);
  const auto mask = make_mask_pattern(tag, 2);
  const auto lower_a = vec::lower(tag, a);
  const auto upper_a = vec::upper(tag, a);
  const auto lower_mask = vec::lower(tag, mask);
  const auto upper_mask = vec::upper(tag, mask);
  const auto rebuilt_a = vec::concat(tag, lower_a, upper_a);
  const auto rebuilt_mask = vec::concat(tag, lower_mask, upper_mask);
  const auto evens = vec::even(tag, a);
  const auto odds = vec::odd(tag, a);
  const auto interleaved = vec::interleave(
      tag, lower_a, vec::lower(tag, b));

  EXPECT_EQ(vec::size(HalfTag{}), half);
  for (vecops::nint_t lane = 0; lane < half; ++lane) {
    expect_lane(
        HalfTag{}, lower_a, lane, lane_value<T>(lane), "lower");
    expect_lane(
        HalfTag{}, upper_a, lane, lane_value<T>(lane + half), "upper");
    expect_mask_lane(
        HalfTag{}, lower_mask, lane, mask_lane(lane, 2), "mask lower");
    expect_mask_lane(
        HalfTag{}, upper_mask, lane,
        mask_lane(lane + half, 2), "mask upper");
    expect_lane(
        HalfTag{}, evens, lane, lane_value<T>(2 * lane), "even");
    expect_lane(
        HalfTag{}, odds, lane, lane_value<T>(2 * lane + 1), "odd");
    expect_lane(
        tag, interleaved, 2 * lane,
        lane_value<T>(lane), "interleave a");
    expect_lane(
        tag, interleaved, 2 * lane + 1,
        lane_value<T>(lane + lanes + 19), "interleave b");
  }
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    expect_lane(tag, rebuilt_a, lane, lane_value<T>(lane), "concat roundtrip");
    expect_mask_lane(
        tag, rebuilt_mask, lane, mask_lane(lane, 2),
        "mask concat roundtrip");
  }
}

template <vec::VectorTag Tag>
void verify_same_width_rearrangements(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr vecops::nint_t block_lanes =
      16 / static_cast<vecops::nint_t>(sizeof(T));
  const vecops::nint_t lanes = vec::size(tag);
  if (lanes < 2) return;

  const vecops::nint_t b_offset = lanes + 37;
  const auto a = make_lane_sequence(tag, 0);
  const auto b = make_lane_sequence(tag, b_offset);
  const auto concat_evens = vec::concat_even(tag, a, b);
  const auto concat_odds = vec::concat_odd(tag, a, b);
  const auto interleaved_evens = vec::interleave_even(tag, a, b);
  const auto interleaved_odds = vec::interleave_odd(tag, a, b);

  for (vecops::nint_t lane = 0; lane < lanes / 2; ++lane) {
    expect_lane(
        tag, concat_evens, lane,
        lane_value<T>(2 * lane), "concat_even lower");
    expect_lane(
        tag, concat_evens, lane + lanes / 2,
        lane_value<T>(b_offset + 2 * lane), "concat_even upper");
    expect_lane(
        tag, concat_odds, lane,
        lane_value<T>(2 * lane + 1), "concat_odd lower");
    expect_lane(
        tag, concat_odds, lane + lanes / 2,
        lane_value<T>(b_offset + 2 * lane + 1), "concat_odd upper");
    expect_lane(
        tag, interleaved_evens, 2 * lane,
        lane_value<T>(2 * lane), "interleave_even a");
    expect_lane(
        tag, interleaved_evens, 2 * lane + 1,
        lane_value<T>(b_offset + 2 * lane), "interleave_even b");
    expect_lane(
        tag, interleaved_odds, 2 * lane,
        lane_value<T>(2 * lane + 1), "interleave_odd a");
    expect_lane(
        tag, interleaved_odds, 2 * lane + 1,
        lane_value<T>(b_offset + 2 * lane + 1), "interleave_odd b");
  }

  // Local interleave is defined on complete 16-byte blocks. Shorter logical
  // subwords have no complete block and therefore intentionally stop above.
  if (lanes < block_lanes || lanes % block_lanes != 0) return;
  const auto local_lower = vec::local_interleave_lower(tag, a, b);
  const auto local_upper = vec::local_interleave_upper(tag, a, b);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const vecops::nint_t block = (lane / block_lanes) * block_lanes;
    const vecops::nint_t within = lane % block_lanes;
    const vecops::nint_t selected = block + within / 2;
    const vecops::nint_t upper_selected = selected + block_lanes / 2;
    const vecops::nint_t stream_offset = (within & 1) ? b_offset : 0;
    expect_lane(
        tag, local_lower, lane,
        lane_value<T>(selected + stream_offset), "local_interleave_lower");
    expect_lane(
        tag, local_upper, lane,
        lane_value<T>(upper_selected + stream_offset),
        "local_interleave_upper");
  }
}

template <vec::VectorTag Tag>
void verify_word_local_vector_shuffle(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using Index = vec::IndexElement<T>;
  using IndexTag = vec::IndexTag<Tag>;
  const vecops::nint_t lanes = vec::size(tag);
  const vecops::nint_t word_lanes = vec::native_word_size(tag);
  const auto source = make_lane_sequence(tag);
  auto indices = vec::zeros(IndexTag{});

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const vecops::nint_t word_start = (lane / word_lanes) * word_lanes;
    const vecops::nint_t remaining = lanes - word_start;
    const vecops::nint_t valid =
        remaining < word_lanes ? remaining : word_lanes;
    const vecops::nint_t within = lane - word_start;
    indices = vec::set(
        IndexTag{}, indices, lane,
        static_cast<Index>(valid - 1 - within));
  }

  const auto result = vec::shuf(tag, source, indices);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const vecops::nint_t word_start = (lane / word_lanes) * word_lanes;
    const vecops::nint_t remaining = lanes - word_start;
    const vecops::nint_t valid =
        remaining < word_lanes ? remaining : word_lanes;
    const vecops::nint_t within = lane - word_start;
    const vecops::nint_t source_lane = word_start + valid - 1 - within;
    EXPECT_TRUE(values_identical(
        lane_value<T>(source_lane), vec::get(tag, result, lane)))
        << "lane=" << lane << ", word_start=" << word_start;
  }
}

template <vec::VectorTag Tag>
void verify_block_local_vector_shuffle(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using Index = vec::IndexElement<T>;
  using IndexTag = vec::IndexTag<Tag>;
  constexpr vecops::nint_t block_lanes =
      16 / static_cast<vecops::nint_t>(sizeof(T));
  const vecops::nint_t lanes = vec::size(tag);
  const auto source = make_lane_sequence(tag);
  auto indices = vec::zeros(IndexTag{});

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const vecops::nint_t block_start = (lane / block_lanes) * block_lanes;
    const vecops::nint_t remaining = lanes - block_start;
    const vecops::nint_t valid =
        remaining < block_lanes ? remaining : block_lanes;
    const vecops::nint_t within = lane - block_start;
    indices = vec::set(
        IndexTag{}, indices, lane,
        static_cast<Index>(valid - 1 - within));
  }

  const auto result = vec::local_shuf(tag, source, indices);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const vecops::nint_t block_start = (lane / block_lanes) * block_lanes;
    const vecops::nint_t remaining = lanes - block_start;
    const vecops::nint_t valid =
        remaining < block_lanes ? remaining : block_lanes;
    const vecops::nint_t within = lane - block_start;
    const vecops::nint_t source_lane = block_start + valid - 1 - within;
    EXPECT_TRUE(values_identical(
        lane_value<T>(source_lane), vec::get(tag, result, lane)))
        << "lane=" << lane << ", block_start=" << block_start;
  }
}

template <vec::VectorTag Tag, std::size_t... Index>
vec::Vec<Tag> runtime_reversing_pattern_local_shuffle(
    Tag tag, vec::Vec<Tag> source, std::index_sequence<Index...>) {
  constexpr vecops::nint_t block_lanes =
      16 / static_cast<vecops::nint_t>(sizeof(vec::ElementOf<Tag>));
  const vecops::nint_t pattern_lanes =
      vec::size(tag) < block_lanes ? vec::size(tag) : block_lanes;
  return vec::local_shuf(
      tag, source,
      static_cast<int>(pattern_lanes - 1 -
                       static_cast<vecops::nint_t>(Index) % pattern_lanes)...);
}

template <vec::VectorTag Tag, std::size_t... Index>
vec::Vec<Tag> compile_time_zero_pattern_local_shuffle(
    Tag tag, vec::Vec<Tag> source, std::index_sequence<Index...>) {
  return vec::local_shuf(
      tag, source, vec::opt::lanes<(static_cast<int>(Index) * 0)...>);
}

template <vec::VectorTag Tag>
void verify_repeating_local_shuffle_patterns(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr vecops::nint_t block_lanes =
      16 / static_cast<vecops::nint_t>(sizeof(T));
  const auto source = make_lane_sequence(tag);
  const auto sequence = std::make_index_sequence<
      static_cast<std::size_t>(block_lanes)>{};
  const auto runtime_result =
      runtime_reversing_pattern_local_shuffle(tag, source, sequence);
  const auto compile_time_result =
      compile_time_zero_pattern_local_shuffle(tag, source, sequence);

  // The runtime pattern reverses every block. For a subword shorter than one
  // block it reverses only the logical lanes, so it never reads padding.
  // The compile-time all-zero pattern is universally valid and broadcasts
  // the first lane of each independently shuffled block.
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const vecops::nint_t block_start = (lane / block_lanes) * block_lanes;
    const vecops::nint_t remaining = vec::size(tag) - block_start;
    const vecops::nint_t valid =
        remaining < block_lanes ? remaining : block_lanes;
    const vecops::nint_t within = lane - block_start;
    const T runtime_expected =
        lane_value<T>(block_start + valid - 1 - within);
    const T compile_time_expected = lane_value<T>(block_start);
    EXPECT_TRUE(values_identical(
        runtime_expected, vec::get(tag, runtime_result, lane)))
        << "runtime lane=" << lane;
    EXPECT_TRUE(values_identical(
        compile_time_expected, vec::get(tag, compile_time_result, lane)))
        << "compile-time lane=" << lane;
  }
}

TYPED_TEST(NVecShuffleElementTest, VectorIndicesUseTheirDocumentedDomains) {
  using T = TypeParam;
  nvec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_word_local_vector_shuffle(Tag{});
    verify_block_local_vector_shuffle(Tag{});
  });
}

TYPED_TEST(NVecShuffleElementTest, ScalarAndCompileTimePatternsRepeatPerBlock) {
  using T = TypeParam;
  nvec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_repeating_local_shuffle_patterns(Tag{});
  });
}

TYPED_TEST(NVecShuffleElementTest, HalvesConcatSelectionAndInterleaveKeepGlobalOrder) {
  using T = TypeParam;
  for_each_halvable_rearrange_shape<T>([]<vec::VectorTag Tag>() {
    verify_halves_concat_and_selection(Tag{});
  });
}

TYPED_TEST(NVecShuffleElementTest, SameWidthRearrangementsKeepDocumentedOrder) {
  using T = TypeParam;
  for_each_same_width_rearrange_shape<T>([]<vec::VectorTag Tag>() {
    verify_same_width_rearrangements(Tag{});
  });
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(NVecShuffleElementTest, FixedSVEArrayPathKeepsShuffleLocality) {
  using T = TypeParam;
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  // SVE VL is a 128-bit multiple, not necessarily a power of two. Select the
  // first power-of-two extent beyond four words so FixedTag remains valid
  // even when CMake requests (for example) a 384-bit fixed SVE vector.
  constexpr vecops::nint_t array_lanes = static_cast<vecops::nint_t>(
      std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
  using Tag = vec::FixedTag<T, array_lanes>;

  verify_word_local_vector_shuffle(Tag{});
  verify_block_local_vector_shuffle(Tag{});
  verify_repeating_local_shuffle_patterns(Tag{});
  verify_halves_concat_and_selection(Tag{});
  verify_same_width_rearrangements(Tag{});
}

#endif
