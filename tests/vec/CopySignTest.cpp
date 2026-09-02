// @vecops-test-shards: 4

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "TestHelpers.h"
#include "TestShard.h"
#include "vecops/vec/Arithmetic.h"

namespace vec = vecops::vec;

template <typename T>
void run_copysign_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::FloatingElements::size);

template <typename T>
T float_from_test_bits(uint64_t bits) {
  if constexpr (std::same_as<T, vecops::bfloat16_t> ||
                std::same_as<T, vecops::float16_t>)
    return T::from_bits(static_cast<uint16_t>(bits));
  else if constexpr (std::same_as<T, vecops::float32_t>)
    return ::vecops::bitcast<T>(static_cast<uint32_t>(bits));
  else
    return ::vecops::bitcast<T>(bits);
}

template <typename T>
uint64_t float_test_bits(T value) {
  if constexpr (std::same_as<T, vecops::bfloat16_t> ||
                std::same_as<T, vecops::float16_t>)
    return value.to_bits();
  else if constexpr (std::same_as<T, vecops::float32_t>)
    return ::vecops::bitcast<uint32_t>(value);
  else
    return ::vecops::bitcast<uint64_t>(value);
}

template <vec::VectorTag Tag>
void verify_copysign(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr uint64_t sign_mask = sizeof(T) == 2 ? 0x8000ull
                                  : sizeof(T) == 4 ? 0x80000000ull
                                                   : 0x8000000000000000ull;
  constexpr uint64_t value_mask = sizeof(T) == 2 ? 0xffffull
                                   : sizeof(T) == 4 ? 0xffffffffull
                                                    : ~uint64_t{0};
  constexpr uint64_t nan_payload =
      std::same_as<T, vecops::bfloat16_t> ? 0x7fc5ull
      : std::same_as<T, vecops::float16_t> ? 0x7e35ull
      : std::same_as<T, vecops::float32_t> ? 0x7fc12345ull
                                           : 0x7ff8123456789abcull;
  auto magnitude = vec::zeros(tag);
  auto sign = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const uint64_t mag_bits = (lane % 3 == 0 ? nan_payload
        : lane % 3 == 1 ? sign_mask : uint64_t{0x3}) & value_mask;
    const uint64_t sign_bits = lane % 2 == 0 ? sign_mask : 0;
    magnitude = vec::set(
        tag, magnitude, lane, float_from_test_bits<T>(mag_bits));
    sign = vec::set(tag, sign, lane, float_from_test_bits<T>(sign_bits));
    mask = vec::set(tag, mask, lane, lane % 2 == 0);
  }

  const auto result = vec::copysign(tag, magnitude, sign);
  const auto inferred = vec::copysign(magnitude, sign);
  const auto masked = vec::copysign(
      tag, magnitude, sign, vec::opt::masked(mask));
  const auto zeroed = vec::copysign(
      tag, magnitude, sign, vec::opt::masked(mask), vec::opt::zero);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const uint64_t magnitude_bits = float_test_bits(
        vec::get(tag, magnitude, lane));
    const uint64_t sign_bits = float_test_bits(vec::get(tag, sign, lane));
    const uint64_t expected =
        (magnitude_bits & ~sign_mask) | (sign_bits & sign_mask);
    EXPECT_EQ(expected, float_test_bits(vec::get(tag, result, lane)));
    EXPECT_EQ(expected, float_test_bits(vec::get(tag, inferred, lane)));
    EXPECT_EQ(
        lane % 2 == 0 ? expected : magnitude_bits,
        float_test_bits(vec::get(tag, masked, lane)));
    EXPECT_EQ(
        lane % 2 == 0 ? expected : uint64_t{0},
        float_test_bits(vec::get(tag, zeroed, lane)));
  }
}

template <typename T>
void run_copysign_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_copysign(Tag{});
  });
}

using ShardType = vec_test::FloatingElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_copysign_test<ShardType>();

#else

template <typename T>
class VecCopySignTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecCopySignTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecCopySignTest, PreservesMagnitudePayloadAndCopiesOnlySign) {
  run_copysign_test<TypeParam>();
}

#endif
