// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "vecops/vec/VecBase.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;
namespace vec_details = vecops::vec::details;

static_assert(__cplusplus >= 202002L);

template <typename T>
class VecBaseElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecBaseElementTest, vec_test::AllElementTypes);

consteval int expected_rebind_scale(std::size_t from, std::size_t to) {
  int scale = 0;
  while (from < to) {
    from *= 2;
    ++scale;
  }
  while (from > to) {
    to *= 2;
    --scale;
  }
  return scale;
}

template <typename T, int ScalePower>
void expect_scalable_metadata() {
  using Tag = vec::ScalableTag<T, ScalePower>;

  const vecops::nint_t word_lanes =
      vec::native_word_size(vec::ScalableTag<T, 0>{});
  const vecops::nint_t expected_lanes = [&] {
    if constexpr (ScalePower >= 0) {
      return word_lanes << ScalePower;
    } else {
      return word_lanes >> (-ScalePower);
    }
  }();

  EXPECT_TRUE((std::same_as<vec::ElementOf<Tag>, T>));
  EXPECT_TRUE(vec::is_scalable_tag_v<Tag>);
  EXPECT_FALSE(vec::is_fixed_tag_v<Tag>);
  EXPECT_EQ(vec::scale_power_v<Tag>, ScalePower);
  EXPECT_EQ(vec::size(Tag{}), expected_lanes);
  EXPECT_EQ(vec::num_words(Tag{}),
            ScalePower > 0 ? (vecops::nint_t{1} << ScalePower) : 1);
  EXPECT_EQ(
      vec::is_subword_v<Tag>,
      vec::size(Tag{}) < vec::native_word_size(Tag{}));
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  EXPECT_TRUE(vec::is_runtime_size_v<Tag>);
#else
  EXPECT_FALSE(vec::is_runtime_size_v<Tag>);
#endif

  using V = vec::Vec<Tag>;
  using M = vec::Mask<Tag>;
  EXPECT_TRUE(vec::VectorValue<V>);
  EXPECT_TRUE(vec::MaskValue<M>);
  EXPECT_FALSE(vec_details::HasInferredTag<M>);
}

template <typename Tag>
concept HasSingleWordVectorAccess = requires(
    vec::Vec<Tag> value, vec::NativeWordVec<Tag> word) {
  { vec::get_word<0>(Tag{}, value) } ->
      std::same_as<vec::NativeWordVec<Tag>>;
  { vec::set_word<0>(Tag{}, value, word) } -> std::same_as<vec::Vec<Tag>>;
  { vec::from_words(Tag{}, word) } -> std::same_as<vec::Vec<Tag>>;
};

template <typename Tag>
concept HasSingleWordMaskAccess = requires(
    vec::Mask<Tag> value, vec::NativeWordMask<Tag> word) {
  { vec::get_word<0>(Tag{}, value) } ->
      std::same_as<vec::NativeWordMask<Tag>>;
  { vec::set_word<0>(Tag{}, value, word) } -> std::same_as<vec::Mask<Tag>>;
  { vec::mask_from_words(Tag{}, word) } -> std::same_as<vec::Mask<Tag>>;
};

template <typename Tag>
concept HasTwoWordVectorAccess = requires(
    vec::Vec<Tag> value, vec::NativeWordVec<Tag> word) {
  { vec::get_word<0>(Tag{}, value) } ->
      std::same_as<vec::NativeWordVec<Tag>>;
  { vec::get_word<1>(Tag{}, value) } ->
      std::same_as<vec::NativeWordVec<Tag>>;
  { vec::set_word<0>(Tag{}, value, word) } -> std::same_as<vec::Vec<Tag>>;
  { vec::set_word<1>(Tag{}, value, word) } -> std::same_as<vec::Vec<Tag>>;
  { vec::from_words(Tag{}, word, word) } -> std::same_as<vec::Vec<Tag>>;
};

template <typename Tag>
concept HasTwoWordMaskAccess = requires(
    vec::Mask<Tag> value, vec::NativeWordMask<Tag> word) {
  { vec::get_word<0>(Tag{}, value) } ->
      std::same_as<vec::NativeWordMask<Tag>>;
  { vec::get_word<1>(Tag{}, value) } ->
      std::same_as<vec::NativeWordMask<Tag>>;
  { vec::set_word<0>(Tag{}, value, word) } -> std::same_as<vec::Mask<Tag>>;
  { vec::set_word<1>(Tag{}, value, word) } -> std::same_as<vec::Mask<Tag>>;
  { vec::mask_from_words(Tag{}, word, word) } -> std::same_as<vec::Mask<Tag>>;
};

template <typename Tag>
concept HasFourWordVectorAccess = requires(
    vec::Vec<Tag> value, vec::NativeWordVec<Tag> word) {
  { vec::get_word<3>(Tag{}, value) } ->
      std::same_as<vec::NativeWordVec<Tag>>;
  { vec::set_word<3>(Tag{}, value, word) } -> std::same_as<vec::Vec<Tag>>;
  { vec::from_words(Tag{}, word, word, word, word) } ->
      std::same_as<vec::Vec<Tag>>;
};

template <typename Tag>
concept HasFourWordMaskAccess = requires(
    vec::Mask<Tag> value, vec::NativeWordMask<Tag> word) {
  { vec::get_word<3>(Tag{}, value) } ->
      std::same_as<vec::NativeWordMask<Tag>>;
  { vec::set_word<3>(Tag{}, value, word) } -> std::same_as<vec::Mask<Tag>>;
  { vec::mask_from_words(Tag{}, word, word, word, word) } ->
      std::same_as<vec::Mask<Tag>>;
};

template <typename Tag>
concept HasEightWordConstruction = requires(
    vec::NativeWordVec<Tag> vector_word,
    vec::NativeWordMask<Tag> mask_word) {
  { vec::from_words(
        Tag{}, vector_word, vector_word, vector_word, vector_word,
        vector_word, vector_word, vector_word, vector_word) } ->
      std::same_as<vec::Vec<Tag>>;
  { vec::mask_from_words(
        Tag{}, mask_word, mask_word, mask_word, mask_word,
        mask_word, mask_word, mask_word, mask_word) } ->
      std::same_as<vec::Mask<Tag>>;
};

TYPED_TEST(VecBaseElementTest, RecognizesEveryBuiltInElementType) {
  using T = TypeParam;

  EXPECT_TRUE(vec::Element<T>);
  EXPECT_FALSE(vec::Element<const T>);
  EXPECT_TRUE((vec::VectorTag<vec::FixedTag<T, 4>>));
  EXPECT_TRUE((vec::VectorTag<vec::ScalableTag<T, 0>>));
}

TEST(VecBaseTest, RejectsUnsupportedElementTypes) {
  EXPECT_FALSE(vec::Element<bool>);
  EXPECT_FALSE(vec::Element<char>);
  EXPECT_FALSE(vec::Element<long double>);
}

TYPED_TEST(VecBaseElementTest, FixedTagRetainsPositivePowerOfTwoExtents) {
  using T = TypeParam;
  using One = vec::FixedTag<T, 1>;
  using Two = vec::FixedTag<T, 2>;
  using Four = vec::FixedTag<T, 4>;

  EXPECT_TRUE(vec::is_fixed_tag_v<One>);
  EXPECT_TRUE(vec::is_fixed_tag_v<Two>);
  EXPECT_TRUE(vec::is_fixed_tag_v<Four>);
  EXPECT_FALSE(vec::is_scalable_tag_v<One>);
  EXPECT_TRUE((std::same_as<vec::ElementOf<One>, T>));
  EXPECT_EQ(vec::fixed_lanes_v<One>, 1);
  EXPECT_EQ(vec::fixed_lanes_v<Two>, 2);
  EXPECT_EQ(vec::fixed_lanes_v<Four>, 4);
  EXPECT_EQ(vec::size(One{}), 1);
  EXPECT_EQ(vec::size(Two{}), 2);
  EXPECT_EQ(vec::size(Four{}), 4);
}

TYPED_TEST(VecBaseElementTest, RebindAndViewAsPreserveTheirDistinctContracts) {
  using T = TypeParam;
  using Fixed = vec::FixedTag<T, 8>;
  using ReboundFixed = vec::Rebind<vecops::uint8_t, Fixed>;
  using ViewedFixed = vec::ViewAs<vecops::uint8_t, Fixed>;

  EXPECT_TRUE((std::same_as<
      ReboundFixed, vec::FixedTag<vecops::uint8_t, 8>>));
  EXPECT_TRUE((std::same_as<
      ViewedFixed,
      vec::FixedTag<
          vecops::uint8_t,
          8 * static_cast<vecops::nint_t>(sizeof(T))>>));

  using Scalable = vec::ScalableTag<T, 1>;
  using ReboundScalable = vec::Rebind<vecops::uint8_t, Scalable>;
  using ViewedScalable = vec::ViewAs<vecops::uint8_t, Scalable>;
  constexpr int expected_rebind_power =
      1 + expected_rebind_scale(sizeof(T), sizeof(vecops::uint8_t));

  EXPECT_TRUE((std::same_as<
      ReboundScalable,
      vec::ScalableTag<vecops::uint8_t, expected_rebind_power>>));
  EXPECT_TRUE((std::same_as<
      ViewedScalable, vec::ScalableTag<vecops::uint8_t, 1>>));
}

TYPED_TEST(VecBaseElementTest, TwiceAndIndexDescriptorsPreserveLaneSemantics) {
  using T = TypeParam;
  using ExpectedIndex = std::conditional_t<
      sizeof(T) == 1,
      vecops::int8_t,
      std::conditional_t<
          sizeof(T) == 2,
          vecops::int16_t,
          std::conditional_t<sizeof(T) == 4, vecops::int32_t, vecops::int64_t>>>;
  using Fixed = vec::FixedTag<T, 4>;
  using Scalable = vec::ScalableTag<T, 0>;
  using TwiceFixed = vec::Twice<Fixed>;
  using TwiceScalable = vec::Twice<Scalable>;
  using FixedIndices = vec::IndexTag<Fixed>;
  using ScalableIndices = vec::IndexTag<Scalable>;

  EXPECT_TRUE((std::same_as<TwiceFixed, vec::FixedTag<T, 8>>));
  EXPECT_TRUE((std::same_as<TwiceScalable, vec::ScalableTag<T, 1>>));
  EXPECT_TRUE((std::same_as<vec::Half<TwiceFixed>, Fixed>));
  EXPECT_TRUE((std::same_as<vec::Half<TwiceScalable>, Scalable>));
  EXPECT_EQ(vec::size(TwiceFixed{}), 2 * vec::size(Fixed{}));
  EXPECT_EQ(vec::size(TwiceScalable{}), 2 * vec::size(Scalable{}));

  EXPECT_TRUE((std::same_as<vec::IndexElement<T>, ExpectedIndex>));
  EXPECT_TRUE((std::same_as<
      FixedIndices, vec::FixedTag<ExpectedIndex, 4>>));
  EXPECT_TRUE((std::same_as<vec::ElementOf<ScalableIndices>, ExpectedIndex>));
  EXPECT_EQ(vec::size(ScalableIndices{}), vec::size(Scalable{}));
}

TYPED_TEST(VecBaseElementTest, ScalableMetadataCoversSubwordAndMultiwordPowers) {
  using T = TypeParam;

  expect_scalable_metadata<T, -1>();
  expect_scalable_metadata<T, 0>();
  expect_scalable_metadata<T, 1>();
  expect_scalable_metadata<T, 2>();

  // The architectural minimum SVE word is 16 bytes. These guards make the
  // more aggressive subword descriptors valid on every supported backend.
  if constexpr (sizeof(T) <= 4) {
    expect_scalable_metadata<T, -2>();
  }
  if constexpr (sizeof(T) <= 2) {
    expect_scalable_metadata<T, -3>();
  }
}

TYPED_TEST(VecBaseElementTest, TagMapsUnambiguouslyToVectorAndMask) {
  using T = TypeParam;
  using Single = vec::ScalableTag<T, 0>;
  using Subword = vec::ScalableTag<T, -1>;
  using Multiword = vec::ScalableTag<T, 1>;

  EXPECT_TRUE(vec::VectorValue<vec::Vec<Single>>);
  EXPECT_TRUE(vec::MaskValue<vec::Mask<Single>>);
  EXPECT_TRUE(vec::VectorValue<vec::Vec<Subword>>);
  EXPECT_TRUE(vec::MaskValue<vec::Mask<Subword>>);
  EXPECT_TRUE(vec::VectorValue<vec::Vec<Multiword>>);
  EXPECT_TRUE(vec::MaskValue<vec::Mask<Multiword>>);

  EXPECT_FALSE(vec_details::HasInferredTag<vec::Mask<Single>>);
  EXPECT_FALSE(vec_details::HasInferredTag<vec::Mask<Subword>>);
  EXPECT_FALSE(vec_details::HasInferredTag<vec::Mask<Multiword>>);
}

TYPED_TEST(VecBaseElementTest, VectorInferenceCanonicalizesPhysicalStorage) {
  using T = TypeParam;
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using LogicalSubword = vec::ScalableTag<T, -1>;
#else
  using LogicalSubword = vec::FixedTag<T, 1>;
#endif
  using SubwordVec = vec::Vec<LogicalSubword>;
  using SubwordPhysical = vec::InferredTagOf<SubwordVec>;

  EXPECT_TRUE(vec::is_subword_v<LogicalSubword>);
  EXPECT_GT(
      vec::native_word_size(LogicalSubword{}), vec::size(LogicalSubword{}));
  EXPECT_FALSE((std::same_as<LogicalSubword, SubwordPhysical>));
  EXPECT_TRUE((std::same_as<vec::Vec<SubwordPhysical>, SubwordVec>));

  using LogicalMultiword = vec::ScalableTag<T, 1>;
  using MultiwordVec = vec::Vec<LogicalMultiword>;
  using MultiwordPhysical = vec::InferredTagOf<MultiwordVec>;
  EXPECT_TRUE((std::same_as<vec::Vec<MultiwordPhysical>, MultiwordVec>));

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  EXPECT_TRUE(vec::is_scalable_tag_v<SubwordPhysical>);
  EXPECT_EQ(vec::scale_power_v<SubwordPhysical>, 0);
  EXPECT_TRUE(vec::is_scalable_tag_v<MultiwordPhysical>);
  EXPECT_EQ(vec::scale_power_v<MultiwordPhysical>, 1);
#else
  EXPECT_TRUE(vec::is_fixed_tag_v<SubwordPhysical>);
  EXPECT_EQ(
      vec::fixed_lanes_v<SubwordPhysical>,
      vec::native_word_size(LogicalSubword{}));
  EXPECT_TRUE(vec::is_fixed_tag_v<MultiwordPhysical>);
  EXPECT_EQ(
      vec::fixed_lanes_v<MultiwordPhysical>, vec::size(LogicalMultiword{}));
#endif
}

TYPED_TEST(VecBaseElementTest, ScalableVectorAndMaskHaveCompileTimeWordAccess) {
  using T = TypeParam;
  using SingleWord = vec::ScalableTag<T, 0>;
  using TwoWords = vec::ScalableTag<T, 1>;
  using FourWords = vec::ScalableTag<T, 2>;

  EXPECT_TRUE(HasSingleWordVectorAccess<SingleWord>);
  EXPECT_TRUE(HasSingleWordMaskAccess<SingleWord>);
  EXPECT_TRUE(HasTwoWordVectorAccess<TwoWords>);
  EXPECT_TRUE(HasTwoWordMaskAccess<TwoWords>);
  EXPECT_TRUE(HasFourWordVectorAccess<FourWords>);
  EXPECT_TRUE(HasFourWordMaskAccess<FourWords>);

#if defined(CPU_CAPABILITY_GENERIC)
  // Scalar words and masks are ordinary sized values, so exercise actual
  // routing as well as expression availability. This deliberately avoids
  // executing uninitialized SVE tuple access in VLA builds.
  vec::Vec<TwoWords> vectors{};
  vec::NativeWordVec<TwoWords> vector_word{};
  vector_word.lanes[0] = static_cast<T>(3.0f);
  auto vector_word0 = vec::NativeWordVec<TwoWords>{};
  vector_word0.lanes[0] = static_cast<T>(2.0f);
  vectors = vec::from_words(TwoWords{}, vector_word0, vector_word);
  EXPECT_EQ(
      static_cast<float>(vec::get_word<1>(TwoWords{}, vectors).lanes[0]),
      static_cast<float>(static_cast<T>(3.0f)));
  EXPECT_EQ(
      static_cast<float>(vec::get_word<0>(TwoWords{}, vectors).lanes[0]),
      static_cast<float>(static_cast<T>(2.0f)));

  vec::Mask<TwoWords> masks{};
  vec::NativeWordMask<TwoWords> mask_word{};
  mask_word.bits.set(0);
  auto mask_word0 = vec::NativeWordMask<TwoWords>{};
  mask_word0.bits.set(1);
  masks = vec::mask_from_words(TwoWords{}, mask_word0, mask_word);
  EXPECT_TRUE(vec::get_word<1>(TwoWords{}, masks).bits.test(0));
  EXPECT_TRUE(vec::get_word<0>(TwoWords{}, masks).bits.test(1));

  vec::Vec<SingleWord> single_vector{};
  vec::NativeWordVec<SingleWord> single_vector_word{};
  single_vector_word.lanes[0] = static_cast<T>(5.0f);
  single_vector =
      vec::set_word<0>(SingleWord{}, single_vector, single_vector_word);
  EXPECT_EQ(
      static_cast<float>(
          vec::get_word<0>(SingleWord{}, single_vector).lanes[0]),
      static_cast<float>(static_cast<T>(5.0f)));

  vec::Mask<SingleWord> single_mask{};
  vec::NativeWordMask<SingleWord> single_mask_word{};
  single_mask_word.bits.set(1);
  single_mask = vec::set_word<0>(SingleWord{}, single_mask, single_mask_word);
  EXPECT_TRUE(vec::get_word<0>(SingleWord{}, single_mask).bits.test(1));
#endif
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)

template <typename T>
inline constexpr vecops::nint_t fixed_backend_word_lanes =
#if defined(CPU_CAPABILITY_SVE)
    FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
#else
    vec::native_word_size(vec::ScalableTag<T, 0>{});
#endif

TYPED_TEST(VecBaseElementTest, FixedRepresentationCoversSubwordAndMultiword) {
  using T = TypeParam;
  constexpr vecops::nint_t native_lanes = fixed_backend_word_lanes<T>;
  using Subword = vec::FixedTag<T, 1>;
  constexpr vecops::nint_t subword_physical_lanes =
      vec::native_word_size(Subword{});
  using CompleteWord = vec::FixedTag<T, subword_physical_lanes>;
  using TwoWords = vec::FixedTag<T, native_lanes * 2>;
  using EightWords = vec::FixedTag<T, native_lanes * 8>;

  EXPECT_EQ(vec::size(Subword{}), 1);
  EXPECT_EQ(vec::num_words(Subword{}), 1);
  EXPECT_TRUE(vec::is_subword_v<Subword>);

  EXPECT_EQ(vec::size(CompleteWord{}), subword_physical_lanes);
  EXPECT_EQ(vec::num_words(CompleteWord{}), 1);
  EXPECT_FALSE(vec::is_subword_v<CompleteWord>);

  EXPECT_EQ(vec::size(TwoWords{}), native_lanes * 2);
  EXPECT_EQ(vec::native_word_size(TwoWords{}), native_lanes);
  EXPECT_EQ(vec::num_words(TwoWords{}), 2);
  EXPECT_FALSE(vec::is_subword_v<TwoWords>);
  EXPECT_TRUE(vec::VectorValue<vec::Vec<TwoWords>>);
  EXPECT_TRUE(vec::MaskValue<vec::Mask<TwoWords>>);
  EXPECT_TRUE(HasTwoWordVectorAccess<TwoWords>);
  EXPECT_TRUE(HasTwoWordMaskAccess<TwoWords>);
  EXPECT_TRUE(HasEightWordConstruction<EightWords>);

  using InferredSubword = vec::InferredTagOf<vec::Vec<Subword>>;
  EXPECT_TRUE((std::same_as<
      InferredSubword, vec::FixedTag<T, subword_physical_lanes>>));
  EXPECT_FALSE((std::same_as<InferredSubword, Subword>));
  EXPECT_TRUE((std::same_as<
      vec::Vec<InferredSubword>, vec::Vec<Subword>>));

  using InferredMultiword = vec::InferredTagOf<vec::Vec<TwoWords>>;
  EXPECT_TRUE((std::same_as<InferredMultiword, TwoWords>));
  EXPECT_FALSE(vec_details::HasInferredTag<vec::Mask<TwoWords>>);
}

#endif

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(VecBaseElementTest, FixedSVEUsesSizedArraysBeyondTupleLimits) {
  using T = TypeParam;
  constexpr vecops::nint_t word_lanes = fixed_backend_word_lanes<T>;
  using EightWords = vec::FixedTag<T, word_lanes * 8>;
  using V = vec::Vec<EightWords>;
  using M = vec::Mask<EightWords>;

  EXPECT_EQ(vec::size(EightWords{}), word_lanes * 8);
  EXPECT_EQ(vec::native_word_size(EightWords{}), word_lanes);
  EXPECT_EQ(vec::num_words(EightWords{}), 8);
  EXPECT_TRUE((requires { sizeof(V); }));
  EXPECT_TRUE((requires { sizeof(M); }));
  EXPECT_TRUE(vec_details::is_word_array_v<V>);
  EXPECT_TRUE(vec_details::is_word_array_v<M>);
  EXPECT_TRUE(vec::VectorValue<V>);
  EXPECT_TRUE(vec::MaskValue<M>);
  EXPECT_FALSE(vec_details::HasInferredTag<M>);

  EXPECT_TRUE((requires(
      V vectors,
      M masks,
      vec::NativeWordVec<EightWords> vector_word,
      vec::NativeWordMask<EightWords> mask_word) {
    { vec::get_word<7>(EightWords{}, vectors) } ->
        std::same_as<vec::NativeWordVec<EightWords>>;
    { vec::set_word<7>(EightWords{}, vectors, vector_word) } ->
        std::same_as<V>;
    { vec::get_word<7>(EightWords{}, masks) } ->
        std::same_as<vec::NativeWordMask<EightWords>>;
    { vec::set_word<7>(EightWords{}, masks, mask_word) } ->
        std::same_as<M>;
  }));

  // VLS values are sized and may safely pass through the public accessors.
  // Zero is sufficient here: array-index routing is covered structurally
  // above without inventing a representation for raw SVE predicate bits.
  V vectors{};
  const auto vector_word = vec::get_word<0>(EightWords{}, vectors);
  vectors = vec::set_word<7>(EightWords{}, vectors, vector_word);
  [[maybe_unused]] const auto copied_vector_word =
      vec::get_word<7>(EightWords{}, vectors);

  M masks{};
  const auto mask_word = vec::get_word<0>(EightWords{}, masks);
  masks = vec::set_word<7>(EightWords{}, masks, mask_word);
  [[maybe_unused]] const auto copied_mask_word =
      vec::get_word<7>(EightWords{}, masks);
}

#endif
