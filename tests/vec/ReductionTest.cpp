// @vecops-test-shards: 13

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "TestHelpers.h"
#include "TestShard.h"
#include "vecops/vec/Reduction.h"

namespace vec = vecops::vec;

template <typename T>
void run_scalable_reduction_test();
template <typename T>
void run_fixed_reduction_test();
void run_reduction_options_test();
void run_reduction_skeleton_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(
    VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size + 1);

namespace {

template <typename V>
concept HasTaglessReduceAdd = requires(V value) {
  vec::reduce_add(value);
};

template <typename V>
concept HasTaglessReduceMax = requires(V value) {
  vec::reduce_max(value);
};

template <typename V>
concept HasTaglessReduceMin = requires(V value) {
  vec::reduce_min(value);
};

template <typename V, typename Mask>
concept HasMaskedTaglessReduceAdd = requires(V value, Mask mask) {
  vec::reduce_add(value, vec::opt::masked(mask));
};

using CanonicalReductionTag = vec::ScalableTag<float, 0>;
using CanonicalReductionVec = vec::Vec<CanonicalReductionTag>;
using CanonicalReductionMask = vec::Mask<CanonicalReductionTag>;

static_assert(!HasTaglessReduceAdd<CanonicalReductionVec>);
static_assert(!HasTaglessReduceMax<CanonicalReductionVec>);
static_assert(!HasTaglessReduceMin<CanonicalReductionVec>);
static_assert(!HasMaskedTaglessReduceAdd<
              CanonicalReductionVec, CanonicalReductionMask>);

template <typename T>
T reduction_value(vecops::nint_t lane) {
  if constexpr (::vecops::IsFloatV<T>) {
    return static_cast<T>(static_cast<float>((lane % 5) - 2));
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>((lane % 5) - 2);
  } else {
    return static_cast<T>((lane % 5) + 1);
  }
}

template <typename T>
T wrapping_add(T a, T b) {
  if constexpr (::vecops::IsFloatV<T>) {
    return static_cast<T>(a + b);
  } else {
    using U = std::make_unsigned_t<T>;
    const auto bits = [](T value) -> U {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(value);
      else return static_cast<U>(value);
    };
    const U sum = static_cast<U>(bits(a) + bits(b));
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(sum);
    else return static_cast<T>(sum);
  }
}

template <typename T>
T max_identity() {
  if constexpr (::vecops::IsFloatV<T>)
    return static_cast<T>(-std::numeric_limits<double>::infinity());
  else
    return std::numeric_limits<T>::lowest();
}

template <typename T>
T min_identity() {
  if constexpr (::vecops::IsFloatV<T>)
    return static_cast<T>(std::numeric_limits<double>::infinity());
  else
    return std::numeric_limits<T>::max();
}

template <vec::VectorTag Tag>
void verify_reduction_shape(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const vecops::nint_t lanes = vec::size(tag);

  // fill writes the complete physical representation. Replacing only logical
  // lanes leaves nonzero padding behind and catches subword over-reduction.
  auto add_value = vec::fill(tag, static_cast<T>(7));
  const T high_padding = [] {
    if constexpr (::vecops::IsFloatV<T>) return static_cast<T>(100);
    else return std::numeric_limits<T>::max();
  }();
  const T low_padding = [] {
    if constexpr (::vecops::IsFloatV<T>) return static_cast<T>(-100);
    else return std::numeric_limits<T>::lowest();
  }();
  auto max_value = vec::fill(tag, high_padding);
  auto min_value = vec::fill(tag, low_padding);
  auto all = vec::mtrue(tag);
  auto none = vec::mfalse(tag);
  auto patterned = vec::mfalse(tag);
  auto high_word = vec::mfalse(tag);
  T expected{};
  T expected_max = max_identity<T>();
  T expected_min = min_identity<T>();
  T expected_pattern{};
  T expected_pattern_max = max_identity<T>();
  T expected_pattern_min = min_identity<T>();
  T expected_high{};
  T expected_high_max = max_identity<T>();
  T expected_high_min = min_identity<T>();
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const T lane_value = reduction_value<T>(lane);
    add_value = vec::set(tag, add_value, lane, lane_value);
    max_value = vec::set(tag, max_value, lane, lane_value);
    min_value = vec::set(tag, min_value, lane, lane_value);
    expected = wrapping_add(expected, lane_value);
    if (lane_value > expected_max) expected_max = lane_value;
    if (lane_value < expected_min) expected_min = lane_value;
    if ((lane % 3) != 1) {
      patterned = vec::set(tag, patterned, lane, true);
      expected_pattern = wrapping_add(expected_pattern, lane_value);
      if (lane_value > expected_pattern_max) expected_pattern_max = lane_value;
      if (lane_value < expected_pattern_min) expected_pattern_min = lane_value;
    }
    if (lane >= lanes - vec::native_word_size(tag)) {
      high_word = vec::set(tag, high_word, lane, true);
      expected_high = wrapping_add(expected_high, lane_value);
      if (lane_value > expected_high_max) expected_high_max = lane_value;
      if (lane_value < expected_high_min) expected_high_min = lane_value;
    }
  }

  EXPECT_TRUE(vec_test::values_identical(expected, vec::reduce_add(tag, add_value)))
      << "lanes=" << lanes << ", words=" << vec::num_words(tag);
  EXPECT_TRUE(vec_test::values_identical(
      expected_max, vec::reduce_max(tag, max_value)));
  EXPECT_TRUE(vec_test::values_identical(
      expected_min, vec::reduce_min(tag, min_value)));
  EXPECT_TRUE(vec_test::values_identical(
      expected,
      vec::reduce_add(tag, add_value, vec::opt::unmasked)));
  EXPECT_TRUE(vec_test::values_identical(
      expected_max,
      vec::reduce_max(tag, max_value, vec::opt::unmasked)));
  EXPECT_TRUE(vec_test::values_identical(
      expected_min,
      vec::reduce_min(tag, min_value, vec::opt::unmasked)));
  EXPECT_TRUE(vec_test::values_identical(
      expected, vec::reduce_add(tag, add_value, vec::opt::masked(all))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_max, vec::reduce_max(tag, max_value, vec::opt::masked(all))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_min, vec::reduce_min(tag, min_value, vec::opt::masked(all))));
  EXPECT_TRUE(vec_test::values_identical(
      T{}, vec::reduce_add(tag, add_value, vec::opt::masked(none))));
  EXPECT_TRUE(vec_test::values_identical(
      max_identity<T>(),
      vec::reduce_max(tag, max_value, vec::opt::masked(none))));
  EXPECT_TRUE(vec_test::values_identical(
      min_identity<T>(),
      vec::reduce_min(tag, min_value, vec::opt::masked(none))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_pattern,
      vec::reduce_add(tag, add_value, vec::opt::masked(patterned))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_pattern_max,
      vec::reduce_max(tag, max_value, vec::opt::masked(patterned))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_pattern_min,
      vec::reduce_min(tag, min_value, vec::opt::masked(patterned))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_high,
      vec::reduce_add(tag, add_value, vec::opt::masked(high_word))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_high_max,
      vec::reduce_max(tag, max_value, vec::opt::masked(high_word))));
  EXPECT_TRUE(vec_test::values_identical(
      expected_high_min,
      vec::reduce_min(tag, min_value, vec::opt::masked(high_word))));

  // Select each logical lane independently, including every word boundary.
  for (vecops::nint_t selected = 0; selected < lanes; ++selected) {
    auto one = vec::mfalse(tag);
    one = vec::set(tag, one, selected, true);
    EXPECT_TRUE(vec_test::values_identical(
        reduction_value<T>(selected),
        vec::reduce_add(tag, add_value, vec::opt::masked(one))))
        << "selected=" << selected << ", lanes=" << lanes
        << ", words=" << vec::num_words(tag);
    EXPECT_TRUE(vec_test::values_identical(
        reduction_value<T>(selected),
        vec::reduce_max(tag, max_value, vec::opt::masked(one))));
    EXPECT_TRUE(vec_test::values_identical(
        reduction_value<T>(selected),
        vec::reduce_min(tag, min_value, vec::opt::masked(one))));
  }
}

template <vec::VectorTag Tag>
void verify_integer_overflow(Tag tag) {
  using T = vec::ElementOf<Tag>;
  if constexpr (std::integral<T>) {
    const T large = std::numeric_limits<T>::max();
    auto value = vec::fill(tag, large);
    T expected{};
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      expected = wrapping_add(expected, large);
    }
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::reduce_add(tag, value)))
        << "overflow lanes=" << vec::size(tag)
        << ", words=" << vec::num_words(tag);
  }
}

} // namespace

template <typename T>
void run_scalable_reduction_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_reduction_shape(Tag{});
  });

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
  // Eight words exercises ordinary word arrays beyond four entries. Sizeless
  // SVE deliberately stops at its supported x4 tuple representation.
  using EightWords = vec::ScalableTag<T, 3>;
  static_assert(vec::num_words(EightWords{}) == 8);
  verify_reduction_shape(EightWords{});
  verify_integer_overflow(EightWords{});
#endif
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
template <typename T>
void run_fixed_reduction_test() {
  verify_reduction_shape(vec::FixedTag<T, 1>{});
  verify_reduction_shape(vec::FixedTag<T, 2>{});
  verify_reduction_shape(vec::FixedTag<T, 8>{});
  verify_reduction_shape(vec::FixedTag<T, 64>{});
  verify_integer_overflow(vec::FixedTag<T, 64>{});
}
#endif

#if VECOPS_TEST_SHARD_INDEX == 12
template <typename Tag, typename... Options>
concept CanReduceAddWith = requires(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  vec::reduce_add(tag, value, std::forward<Options>(options)...);
};

void run_reduction_options_test() {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using Tag = vec::ScalableTag<vecops::float32_t, 0>;
#else
  using Tag = vec::FixedTag<vecops::float32_t, 4>;
#endif
  using Mask = vec::Mask<Tag>;
  static_assert(CanReduceAddWith<Tag, vec::opt::Masked<Mask>>);
  static_assert(!CanReduceAddWith<Tag, Mask>);
  static_assert(!CanReduceAddWith<Tag, vec::opt::Zero>);
  static_assert(!CanReduceAddWith<Tag, vec::opt::ScalarMerge<vecops::float32_t>>);
  static_assert(!CanReduceAddWith<
      Tag, vec::opt::Masked<Mask>, vec::opt::Masked<Mask>>);
}

void run_reduction_skeleton_test() {
  static_assert(std::same_as<decltype(vec::reduce_add), const vec::ReduceAddOp>);
  static_assert(std::same_as<decltype(vec::reduce_max), const vec::ReduceMaxOp>);
  static_assert(std::same_as<decltype(vec::reduce_min), const vec::ReduceMinOp>);
}
#endif

#if VECOPS_TEST_SHARD_INDEX < 12
using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_scalable_reduction_test<ShardType>();
#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
template void run_fixed_reduction_test<ShardType>();
#endif
#endif

#else

template <typename T>
class VecReductionTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecReductionTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecReductionTest, AllScalableShapes) {
  run_scalable_reduction_test<TypeParam>();
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecReductionTest, FixedShapes) {
  run_fixed_reduction_test<TypeParam>();
}
#endif

TEST(VecReductionOptionsTest, RejectsPositionalMaskAndPopulationOptions) {
  run_reduction_options_test();
}

TEST(VecReductionSkeletonTest, DeclaresAllReductionCpos) {
  run_reduction_skeleton_test();
}

#endif
