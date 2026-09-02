// @vecops-test-shards: 12

#include <gtest/gtest.h>

#include <type_traits>

#include "TestHelpers.h"
#include "TestShard.h"
#include "vecops/vec/Sequence.h"

namespace vec = vecops::vec;

template <typename T>
void run_sequence_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size);

template <typename T>
T sequence_start() {
  if constexpr (std::is_signed_v<T> || ::vecops::is_float_v<T>)
    return static_cast<T>(-3);
  else
    return static_cast<T>(5);
}

template <typename T>
T sequence_step() {
  return static_cast<T>(3);
}

template <typename T>
T expected_iota(T start, T step, vecops::nint_t lane) {
  if constexpr (std::integral<T>) {
    using U = std::make_unsigned_t<T>;
    const U a = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(start);
      else return start;
    }();
    const U b = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(step);
      else return step;
    }();
    const U result = static_cast<U>(a + b * static_cast<U>(lane));
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
    else return result;
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    return static_cast<T>(start + static_cast<T>(lane) * step);
  } else {
    const T represented_lane = static_cast<T>(lane);
    return static_cast<T>(
        static_cast<float>(start) +
        static_cast<float>(represented_lane) * static_cast<float>(step));
  }
}

template <vec::VectorTag Tag>
void verify_iota(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const T start = sequence_start<T>();
  const T step = sequence_step<T>();
  const auto unit = vec::iota(tag, start);
  const auto strided = vec::iota(tag, start, step);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        expected_iota(start, static_cast<T>(1), lane),
        vec::get(tag, unit, lane))) << "unit lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected_iota(start, step, lane),
        vec::get(tag, strided, lane))) << "strided lane=" << lane;
  }
}

template <typename T>
void run_sequence_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_iota(Tag{});
  });
}

using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_sequence_test<ShardType>();

#else

template <typename T>
class VecSequenceTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecSequenceTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecSequenceTest, EveryLegalElementAndScalableShape) {
  run_sequence_test<TypeParam>();
}

#endif
