#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "vecops/nvec/Basic.h"
#include "TestHelpers.h"

namespace vec = vecops::nvec;

template <typename T>
class NVecBasicElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(NVecBasicElementTest, nvec_test::AllElementTypes);

template <typename T>
T test_value(int seed) {
  if constexpr (std::same_as<T, vecops::bfloat16_t>) {
    return vecops::bfloat16_t(static_cast<float>(seed) + 0.25F);
  } else if constexpr (std::same_as<T, vecops::float16_t>) {
    return vecops::float16_t(static_cast<float>(seed) + 0.25F);
  } else if constexpr (std::same_as<T, vecops::float32_t>) {
    return static_cast<vecops::float32_t>(seed) + 0.25F;
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    return static_cast<vecops::float64_t>(seed) + 0.25;
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>((seed % 31) - 15);
  } else {
    return static_cast<T>((seed * 3 + 1) % 61);
  }
}

using nvec_test::values_identical;

template <typename T>
struct BitcastPeer;

template <> struct BitcastPeer<vecops::bfloat16_t> { using Type = vecops::uint16_t; };
template <> struct BitcastPeer<vecops::float16_t> { using Type = vecops::int16_t; };
template <> struct BitcastPeer<vecops::float32_t> { using Type = vecops::uint32_t; };
template <> struct BitcastPeer<vecops::float64_t> { using Type = vecops::uint64_t; };
template <> struct BitcastPeer<vecops::int8_t> { using Type = vecops::uint8_t; };
template <> struct BitcastPeer<vecops::uint8_t> { using Type = vecops::int8_t; };
template <> struct BitcastPeer<vecops::int16_t> { using Type = vecops::float16_t; };
template <> struct BitcastPeer<vecops::uint16_t> { using Type = vecops::bfloat16_t; };
template <> struct BitcastPeer<vecops::int32_t> { using Type = vecops::float32_t; };
template <> struct BitcastPeer<vecops::uint32_t> { using Type = vecops::int32_t; };
template <> struct BitcastPeer<vecops::int64_t> { using Type = vecops::float64_t; };
template <> struct BitcastPeer<vecops::uint64_t> { using Type = vecops::int64_t; };

template <typename T>
using BitcastPeerOf = typename BitcastPeer<T>::Type;

using nvec_test::for_each_scalable_shape;

template <vec::VectorTag Tag>
void verify_fill_and_zeros(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const T filled_value = test_value<T>(19);
  const auto filled = vec::fill(tag, filled_value);
  const auto zero = vec::zeros(tag);

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(values_identical(filled_value, vec::get(tag, filled, lane)))
        << "fill lane=" << lane;
    EXPECT_TRUE(values_identical(T{}, vec::get(tag, zero, lane)))
        << "zero lane=" << lane;
  }
}

template <vec::VectorTag Tag>
void verify_lane_get_set(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const vecops::nint_t lanes = vec::size(tag);
  const vecops::nint_t word_lanes = vec::native_word_size(tag);
  auto value = vec::zeros(tag);
  std::vector<T> expected(static_cast<std::size_t>(lanes), T{});

  // Set every lane so runtime word dispatch necessarily crosses each word
  // boundary in the 2-word and 4-word cases.
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const T replacement = test_value<T>(static_cast<int>(lane + 3));
    value = vec::set(tag, value, lane, replacement);
    expected[static_cast<std::size_t>(lane)] = replacement;
  }
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_TRUE(values_identical(
        expected[static_cast<std::size_t>(lane)],
        vec::get(tag, value, lane)))
        << "lane=" << lane << ", word=" << lane / word_lanes;
  }

  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    mask = vec::set(tag, mask, lane, lane % 3 == 1);
  }
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(vec::get(tag, mask, lane), lane % 3 == 1)
        << "mask lane=" << lane << ", word=" << lane / word_lanes;
  }

  if (lanes > word_lanes) {
    const vecops::nint_t before_boundary = word_lanes - 1;
    const vecops::nint_t after_boundary = word_lanes;
    mask = vec::set(tag, mask, before_boundary, true);
    mask = vec::set(tag, mask, after_boundary, false);
    EXPECT_TRUE(vec::get(tag, mask, before_boundary));
    EXPECT_FALSE(vec::get(tag, mask, after_boundary));
  }
}

template <vec::VectorTag Tag>
void verify_mask_construction_and_logic(Tag tag) {
  const vecops::nint_t lanes = vec::size(tag);
  const auto all_from_fill = vec::mfill(tag, true);
  const auto none_from_fill = vec::mfill(tag, false);
  const auto all = vec::mtrue(tag);
  const auto none = vec::mfalse(tag);

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_TRUE(vec::get(tag, all_from_fill, lane));
    EXPECT_TRUE(vec::get(tag, all, lane));
    EXPECT_FALSE(vec::get(tag, none_from_fill, lane));
    EXPECT_FALSE(vec::get(tag, none, lane));
  }

  // Non-zero a is intentional: it fixes the public contract as a+i versus b,
  // rather than accidentally testing a mask over absolute lane indices.
  constexpr vecops::nint_t a = 7;
  const vecops::nint_t split = lanes / 2;
  const vecops::nint_t b = a + split;
  const auto lt = vec::mwhilelt(tag, a, b);
  const auto le = vec::mwhilele(tag, a, b);
  const auto ge = vec::mwhilege(tag, a, b);
  const auto gt = vec::mwhilegt(tag, a, b);
  const auto lt_all = vec::mwhilelt(tag, a, a + lanes);
  const auto lt_none = vec::mwhilelt(tag, a, a);
  const auto ge_all = vec::mwhilege(tag, a, a);
  const auto ge_none = vec::mwhilege(tag, a, a + lanes);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(vec::get(tag, lt, lane), a + lane < b) << "lt lane=" << lane;
    EXPECT_EQ(vec::get(tag, le, lane), a + lane <= b) << "le lane=" << lane;
    EXPECT_EQ(vec::get(tag, ge, lane), a + lane >= b) << "ge lane=" << lane;
    EXPECT_EQ(vec::get(tag, gt, lane), a + lane > b) << "gt lane=" << lane;
    EXPECT_TRUE(vec::get(tag, lt_all, lane));
    EXPECT_FALSE(vec::get(tag, lt_none, lane));
    EXPECT_TRUE(vec::get(tag, ge_all, lane));
    EXPECT_FALSE(vec::get(tag, ge_none, lane));
  }

  auto lhs = vec::mfalse(tag);
  auto rhs = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    lhs = vec::set(tag, lhs, lane, lane % 2 == 0);
    rhs = vec::set(tag, rhs, lane, lane % 3 != 0);
  }
  const auto both = vec::mask_and(tag, lhs, rhs);
  const auto either = vec::mask_or(tag, lhs, rhs);
  const auto different = vec::mask_xor(tag, lhs, rhs);
  const auto not_lhs_and_rhs = vec::mask_andnot(tag, lhs, rhs);
  const auto inverse = vec::mask_not(tag, lhs);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const bool l = lane % 2 == 0;
    const bool r = lane % 3 != 0;
    EXPECT_EQ(vec::get(tag, both, lane), l && r);
    EXPECT_EQ(vec::get(tag, either, lane), l || r);
    EXPECT_EQ(vec::get(tag, different, lane), l != r);
    EXPECT_EQ(vec::get(tag, not_lhs_and_rhs, lane), !l && r);
    EXPECT_EQ(vec::get(tag, inverse, lane), !l);
  }
}

template <vec::VectorTag Tag>
void verify_blend_and_conditional_fill(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const vecops::nint_t lanes = vec::size(tag);
  const T false_scalar = test_value<T>(5);
  const T true_scalar = test_value<T>(23);
  const T scalar_merge = test_value<T>(41);
  const auto false_value = vec::fill(tag, false_scalar);
  const auto true_value = vec::fill(tag, true_scalar);

  auto alternating = vec::mfalse(tag);
  auto vector_merge = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    alternating = vec::set(tag, alternating, lane, lane % 2 == 0);
    vector_merge = vec::set(
        tag, vector_merge, lane,
        test_value<T>(static_cast<int>(lane + 50)));
  }

  const auto blended =
      vec::blend(tag, false_value, alternating, true_value);
  const auto blended_all =
      vec::blend(tag, false_value, vec::mtrue(tag), true_value);
  const auto blended_none =
      vec::blend(tag, false_value, vec::mfalse(tag), true_value);
  const auto masked_zero =
      vec::fill(tag, true_scalar, vec::opt::masked(alternating));
  const auto masked_scalar = vec::fill(
      tag, true_scalar,
      vec::opt::masked(alternating), vec::opt::merge(scalar_merge));
  const auto masked_vector = vec::fill(
      tag, true_scalar,
      vec::opt::masked(alternating), vec::opt::merge(vector_merge));
  const auto all_mask = vec::mtrue(tag);
  const auto no_mask = vec::mfalse(tag);
  const auto masked_all = vec::fill(
      tag, true_scalar,
      vec::opt::masked(all_mask), vec::opt::merge(scalar_merge));
  const auto masked_none = vec::fill(
      tag, true_scalar,
      vec::opt::masked(no_mask), vec::opt::merge(scalar_merge));

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const bool active = lane % 2 == 0;
    EXPECT_TRUE(values_identical(
        active ? true_scalar : false_scalar,
        vec::get(tag, blended, lane)));
    EXPECT_TRUE(values_identical(
        true_scalar, vec::get(tag, blended_all, lane)));
    EXPECT_TRUE(values_identical(
        false_scalar, vec::get(tag, blended_none, lane)));
    EXPECT_TRUE(values_identical(
        active ? true_scalar : T{},
        vec::get(tag, masked_zero, lane)));
    EXPECT_TRUE(values_identical(
        active ? true_scalar : scalar_merge,
        vec::get(tag, masked_scalar, lane)));
    EXPECT_TRUE(values_identical(
        active ? true_scalar : vec::get(tag, vector_merge, lane),
        vec::get(tag, masked_vector, lane)));
    EXPECT_TRUE(values_identical(
        true_scalar, vec::get(tag, masked_all, lane)));
    EXPECT_TRUE(values_identical(
        scalar_merge, vec::get(tag, masked_none, lane)));
  }

  const vecops::nint_t count = lanes - lanes / 3;
  const auto first_zero =
      vec::fill(tag, true_scalar, vec::opt::first(count));
  const auto first_scalar = vec::fill(
      tag, true_scalar,
      vec::opt::first(count), vec::opt::merge(scalar_merge));
  const auto first_vector = vec::fill(
      tag, true_scalar,
      vec::opt::first(count), vec::opt::merge(vector_merge));
  const auto first_negative = vec::fill(
      tag, true_scalar,
      vec::opt::first(-3), vec::opt::merge(scalar_merge));
  const auto first_past_end = vec::fill(
      tag, true_scalar,
      vec::opt::first(lanes + 3), vec::opt::merge(scalar_merge));

  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const bool active = lane < count;
    EXPECT_TRUE(values_identical(
        active ? true_scalar : T{}, vec::get(tag, first_zero, lane)));
    EXPECT_TRUE(values_identical(
        active ? true_scalar : scalar_merge,
        vec::get(tag, first_scalar, lane)));
    EXPECT_TRUE(values_identical(
        active ? true_scalar : vec::get(tag, vector_merge, lane),
        vec::get(tag, first_vector, lane)));
    EXPECT_TRUE(values_identical(
        scalar_merge, vec::get(tag, first_negative, lane)));
    EXPECT_TRUE(values_identical(
        true_scalar, vec::get(tag, first_past_end, lane)));
  }
}

template <vec::VectorTag FromTag>
void verify_bitcast(FromTag from) {
  using From = vec::ElementOf<FromTag>;
  using To = BitcastPeerOf<From>;
  using ToTag = vec::Rebind<To, FromTag>;
  static_assert(sizeof(From) == sizeof(To));
  EXPECT_EQ(vec::size(ToTag{}), vec::size(FromTag{}));

  auto source = vec::zeros(from);
  std::vector<From> expected(
      static_cast<std::size_t>(vec::size(from)), From{});
  for (vecops::nint_t lane = 0; lane < vec::size(from); ++lane) {
    const From value = test_value<From>(static_cast<int>(lane + 11));
    source = vec::set(from, source, lane, value);
    expected[static_cast<std::size_t>(lane)] = value;
  }

  const auto result = vec::bitcast(ToTag{}, from, source);
  for (vecops::nint_t lane = 0; lane < vec::size(from); ++lane) {
    const To expected_bits =
        ::vecops::bitcast<To>(expected[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(values_identical(
        expected_bits, vec::get(ToTag{}, result, lane)))
        << "lane=" << lane;
  }
}

template <vec::VectorTag Tag>
void verify_extreme_values(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const vecops::nint_t lanes = vec::size(tag);

  if constexpr (std::is_integral_v<T>) {
    const T minimum = std::numeric_limits<T>::min();
    const T maximum = std::numeric_limits<T>::max();
    const auto minima = vec::fill(tag, minimum);
    const auto maxima = vec::fill(tag, maximum);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      EXPECT_EQ(vec::get(tag, minima, lane), minimum);
      EXPECT_EQ(vec::get(tag, maxima, lane), maximum);
    }
  } else {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T infinity = std::numeric_limits<T>::infinity();
    const T negative_infinity = -infinity;
    const T maximum = std::numeric_limits<T>::max();
    const T minimum_normal = std::numeric_limits<T>::min();
    const T subnormal = std::numeric_limits<T>::denorm_min();
    const T positive_zero = T(0.0F);
    const T negative_zero = T(-0.0F);

    const auto nan_values = vec::fill(tag, nan);
    const auto infinities = vec::fill(tag, infinity);
    const auto negative_infinities = vec::fill(tag, negative_infinity);
    const auto maxima = vec::fill(tag, maximum);
    const auto minima = vec::fill(tag, minimum_normal);
    const auto subnormals = vec::fill(tag, subnormal);
    const auto positive_zeros = vec::fill(tag, positive_zero);
    const auto negative_zeros = vec::fill(tag, negative_zero);

    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      EXPECT_TRUE(std::isnan(static_cast<double>(
          vec::get(tag, nan_values, lane))));
      EXPECT_TRUE(std::isinf(static_cast<double>(
          vec::get(tag, infinities, lane))));
      EXPECT_FALSE(std::signbit(static_cast<double>(
          vec::get(tag, infinities, lane))));
      EXPECT_TRUE(std::isinf(static_cast<double>(
          vec::get(tag, negative_infinities, lane))));
      EXPECT_TRUE(std::signbit(static_cast<double>(
          vec::get(tag, negative_infinities, lane))));
      EXPECT_TRUE(values_identical(maximum, vec::get(tag, maxima, lane)));
      EXPECT_TRUE(values_identical(
          minimum_normal, vec::get(tag, minima, lane)));
      EXPECT_TRUE(values_identical(
          subnormal, vec::get(tag, subnormals, lane)));
      EXPECT_FALSE(std::signbit(static_cast<double>(
          vec::get(tag, positive_zeros, lane))));
      EXPECT_TRUE(std::signbit(static_cast<double>(
          vec::get(tag, negative_zeros, lane))));
    }
  }
}

TYPED_TEST(NVecBasicElementTest, FillZerosAndLaneAccessCoverEveryShape) {
  using T = TypeParam;
  for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_fill_and_zeros(Tag{});
    verify_lane_get_set(Tag{});
  });
}

TYPED_TEST(NVecBasicElementTest, MasksAndLogicCoverEveryShape) {
  using T = TypeParam;
  for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_mask_construction_and_logic(Tag{});
  });
}

TYPED_TEST(NVecBasicElementTest, BlendAndConditionalFillCoverEveryShape) {
  using T = TypeParam;
  for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_blend_and_conditional_fill(Tag{});
  });
}

TYPED_TEST(NVecBasicElementTest, BitcastCoversEveryShape) {
  using T = TypeParam;
  for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_bitcast(Tag{});
  });
}

TYPED_TEST(NVecBasicElementTest, ExtremeValuesPreserveBitsAndSignedZero) {
  using T = TypeParam;
  // One and four words catch both the primitive and the multiword path while
  // avoiding redundant copies of the expensive NaN/Inf matrix.
  verify_extreme_values(vec::ScalableTag<T, 0>{});
  verify_extreme_values(vec::ScalableTag<T, 2>{});
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(NVecBasicElementTest, FixedSVEArrayPathCrossesTupleLimit) {
  using T = TypeParam;
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  using Tag = vec::FixedTag<T, word_lanes * 8>;

  verify_fill_and_zeros(Tag{});
  verify_lane_get_set(Tag{});
  verify_mask_construction_and_logic(Tag{});
  verify_blend_and_conditional_fill(Tag{});
  verify_bitcast(Tag{});
}

#endif
