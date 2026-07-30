#include <gtest/gtest.h>

#include <type_traits>

#include "vecops/vec/Conversion.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

template <vec::VectorTag FromTag, vec::VectorTag ToTag>
void verify_mask_conversion(FromTag from, ToTag to) {
  ASSERT_EQ(vec::size(to), vec::size(from));
  const auto verify = [&](vec::Mask<FromTag> input, const char* pattern) {
    const auto output = vec::convert(to, from, input);
    for (vecops::nint_t lane = 0; lane < vec::size(to); ++lane)
      EXPECT_EQ(vec::get(from, input, lane), vec::get(to, output, lane))
          << "pattern=" << pattern << ", lane=" << lane
          << ", from_words=" << vec::num_words(from)
          << ", to_words=" << vec::num_words(to);
  };
  verify(vec::mfalse(from), "all-false");
  verify(vec::mtrue(from), "all-true");
  auto input = vec::mfalse(from);
  for (vecops::nint_t lane = 0; lane < vec::size(from); ++lane)
    input = vec::set(from, input, lane, lane % 3 != 1);
  verify(input, "alternating");
  auto boundary = vec::mfalse(from);
  boundary = vec::set(from, boundary, 0, true);
  boundary = vec::set(from, boundary, vec::size(from) - 1, true);
  const auto word_lanes = vec::native_word_size(from);
  if (word_lanes < vec::size(from)) {
    boundary = vec::set(from, boundary, word_lanes - 1, true);
    boundary = vec::set(from, boundary, word_lanes, true);
  }
  verify(boundary, "word-boundaries");
}

template <typename T>
class VecMaskConversionTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecMaskConversionTest, vec_test::AllElementTypes);

TYPED_TEST(VecMaskConversionTest, EveryDestinationAndScalableShape) {
  using From = TypeParam;
  vec_test::for_each_element_type([&]<typename To>() {
    vec_test::for_each_scalable_conversion_shape<From, To>(
        []<vec::VectorTag FromTag, vec::VectorTag ToTag>() {
          verify_mask_conversion(FromTag{}, ToTag{});
        });
  });
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
TEST(VecMaskConversionFixedTest, MultiwordAllLanes) {
  using FromTag = vec::FixedTag<int8_t, 64>;
  verify_mask_conversion(FromTag{}, vec::Rebind<vecops::float64_t, FromTag>{});
  using ReverseTag = vec::FixedTag<vecops::float64_t, 64>;
  verify_mask_conversion(ReverseTag{}, vec::Rebind<vecops::uint8_t, ReverseTag>{});
}
#endif

template <typename ToTag, typename FromTag>
concept ConvertsMask = requires(
    ToTag to, FromTag from, vec::Mask<FromTag> mask) {
  vec::convert(to, from, mask);
};

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
static_assert(ConvertsMask<
              vec::ScalableTag<int32_t, 2>, vec::ScalableTag<int8_t>>);
static_assert(!ConvertsMask<
              vec::ScalableTag<int32_t>, vec::ScalableTag<int8_t>>);
#else
static_assert(ConvertsMask<
              vec::ScalableTag<int64_t, 3>, vec::ScalableTag<int8_t>>);
static_assert(!ConvertsMask<
              vec::ScalableTag<int64_t>, vec::ScalableTag<int8_t>>);
static_assert(!ConvertsMask<
              vec::FixedTag<int64_t, 2>, vec::FixedTag<int8_t, 4>>);
#endif

} // namespace
