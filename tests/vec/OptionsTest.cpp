// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include "vecops/vec/Memory.h"

template <typename V>
concept CanIndexRvalue = requires(V value) {
  vecops::vec::opt::indexed(std::move(value));
};

template <typename V>
concept CanIndexScaledRvalue = requires(V value) {
  vecops::vec::opt::indexed(std::move(value), vecops::vec::opt::scale<1>);
};

namespace vec = vecops::vec;

TEST(VecOptionsTest, ClassifiesEveryPublicOptionFamily) {
  using namespace vecops::vec::details;

  using StrictAccuracy =
      std::remove_cvref_t<decltype(vec::opt::math::strict)>;
  using FastAccuracy =
      std::remove_cvref_t<decltype(vec::opt::math::fast)>;
  using EstimateAccuracy =
      std::remove_cvref_t<decltype(vec::opt::math::estimate)>;
  static_assert(IsMathAccuracyOption<StrictAccuracy>::value);
  static_assert(IsMathAccuracyOption<StrictAccuracy>::accuracy ==
                vec::Accuracy::Strict);
  static_assert(IsMathAccuracyOption<FastAccuracy>::accuracy ==
                vec::Accuracy::Fast);
  static_assert(IsMathAccuracyOption<EstimateAccuracy>::accuracy ==
                vec::Accuracy::Estimate);
  static_assert(std::same_as<
      std::remove_cvref_t<decltype(vec::opt::math::fast)>,
      std::remove_cvref_t<decltype(
          vec::opt::math::accuracy<vec::Accuracy::Fast>)>>);
  static_assert(IsUnmaskedOption<decltype(vec::opt::unmasked)>::value);
  static_assert(IsFirstOption<decltype(vec::opt::first(1))>::value);
  static_assert(IsZeroOption<decltype(vec::opt::zero)>::value);
  static_assert(IsOrderedOption<decltype(vec::cvt::ordered)>::value);
  static_assert(IsUnorderedOption<decltype(vec::cvt::unordered)>::value);
  using Lane = std::remove_cvref_t<decltype(vec::cvt::lane<1>)>;
  static_assert(IsLaneOption<Lane>::value);
  static_assert(IsLaneOption<Lane>::phase == 1);
  static_assert(IsSaturateOption<decltype(vec::cvt::saturate)>::value);
  static_assert(IsWrapOption<decltype(vec::cvt::wrap)>::value);
  static_assert(IsWrapOption<decltype(vec::cvt::truncate)>::value);
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using AddressTag = vec::ScalableTag<vecops::float32_t, 0>;
#else
  using AddressTag = vec::FixedTag<vecops::float32_t, 4>;
#endif
  using IndexVec = vec::Vec<vec::Rebind<int32_t, AddressTag>>;
  using Indexed = vec::opt::Indexed<IndexVec, 4>;
  using Strided = vec::opt::Strided<vecops::meta::Const<3>>;
  static_assert(IsIndexedOption<Indexed>::value);
  static_assert(IsIndexedOption<Indexed>::scale == 4);
  static_assert(IsStridedOption<Strided>::value);
  static_assert(!CanIndexRvalue<IndexVec>);
  static_assert(!CanIndexScaledRvalue<IndexVec>);
  static_assert(IsUnalignedOption<decltype(vec::mem::unaligned)>::value);
  static_assert(IsAlignedOption<decltype(vec::mem::aligned)>::value);
  static_assert(IsTemporalOption<decltype(vec::mem::temporal)>::value);
  static_assert(
      IsNonTemporalOption<decltype(vec::mem::non_temporal)>::value);
  static_assert(IsPackedOption<decltype(vec::mem::packed)>::value);
  static_assert(IsSplitOption<decltype(vec::mem::split)>::value);
  static_assert(
      IsPrefetchLocalityOption<decltype(vec::mem::prefetch_l1)>::value);
  static_assert(
      IsPrefetchLocalityOption<decltype(vec::mem::prefetch_l2)>::value);
  static_assert(
      IsPrefetchTemporalityOption<decltype(vec::mem::prefetch_keep)>::value);
  static_assert(IsPrefetchTemporalityOption<
                decltype(vec::mem::prefetch_stream)>::value);
  static_assert(
      IsPrefetchIntentOption<decltype(vec::mem::prefetch_read)>::value);
  static_assert(
      IsPrefetchIntentOption<decltype(vec::mem::prefetch_write)>::value);

  static_assert(valid_prefetch_options<>());
  static_assert(valid_prefetch_options<
      decltype(vec::mem::prefetch_l2),
      decltype(vec::mem::prefetch_stream),
      decltype(vec::mem::prefetch_write)>());
  static_assert(!valid_prefetch_options<
      decltype(vec::mem::prefetch_l1),
      decltype(vec::mem::prefetch_l2)>());
  static_assert(!valid_prefetch_options<
      decltype(vec::mem::prefetch_keep),
      decltype(vec::mem::prefetch_stream)>());
  static_assert(!valid_prefetch_options<
      decltype(vec::mem::prefetch_read),
      decltype(vec::mem::prefetch_write)>());

  SUCCEED();
}

TEST(VecOptionsTest, ValidatesIndexedAndStridedMemoryOptions) {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using Tag = vec::ScalableTag<vecops::float32_t, 0>;
#else
  using Tag = vec::FixedTag<vecops::float32_t, 4>;
#endif
  using I32 = vec::Vec<vec::Rebind<int32_t, Tag>>;
  using I64 = vec::Vec<vec::Rebind<int64_t, Tag>>;
  using U32 = vec::Vec<vec::Rebind<uint32_t, Tag>>;
  using Indexed32 = vec::opt::Indexed<I32, 0>;
  using Indexed64 = vec::opt::Indexed<I64, 1>;
  using UnsignedIndexed = vec::opt::Indexed<U32, 1>;
  using ConstantStride = vec::opt::Strided<vecops::meta::Const<4>>;
  using DynamicStride = vec::opt::Strided<vecops::meta::Dynamic<1>>;

  using namespace vecops::vec::details;
  static_assert(valid_memory_options<Tag, false, Indexed32>());
  static_assert(valid_memory_options<Tag, true, Indexed64>());
  static_assert(!valid_memory_options<Tag, false, UnsignedIndexed>());
  static_assert(valid_memory_options<Tag, false, ConstantStride>());
  static_assert(valid_memory_options<Tag, true, DynamicStride>());
  static_assert(valid_memory_options<
                Tag, false, Indexed32, decltype(vec::mem::non_temporal)>());
  static_assert(!valid_memory_options<Tag, false, Indexed32, Indexed64>());
  static_assert(!valid_memory_options<
                Tag, false, Indexed32, ConstantStride>());
  static_assert(!valid_memory_options<
                Tag, false, Indexed32, decltype(vec::mem::aligned)>());
  static_assert(!valid_memory_options<
                Tag, true, DynamicStride, decltype(vec::mem::aligned)>());
  SUCCEED();
}

TEST(VecOptionsTest, FindsAndCountsCategoryOptions) {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  using Tag = vec::ScalableTag<vecops::float32_t, 0>;
#else
  using Tag = vec::FixedTag<vecops::float32_t, 4>;
#endif
  auto mask = vec::Mask<Tag>{};
  auto value = vec::Vec<Tag>{};
  auto masked = vec::opt::masked(mask);
  auto unmasked = vec::opt::unmasked;
  auto first = vec::opt::first(3);
  auto merge = vec::opt::merge(value);
  auto mask_merge = vec::opt::merge(mask);

  static_assert(vecops::vec::details::is_masked_option_for_v<
                Tag, decltype(masked)>);
  static_assert(vecops::vec::details::is_vector_population_option_for_v<
                Tag, decltype(vec::opt::zero)>);
  static_assert(vecops::vec::details::is_vector_population_option_for_v<
                Tag, decltype(merge)>);
  static_assert(vecops::vec::details::is_mask_population_option_for_v<
                Tag, decltype(mask_merge)>);

  static_assert(vecops::vec::details::option_count_v<
      vecops::vec::details::IsMaskedOption,
      decltype(masked), decltype(first), decltype(merge)> == 1);
  static_assert(vecops::vec::details::option_count_v<
      vecops::vec::details::IsUnmaskedOption,
      decltype(masked), decltype(unmasked), decltype(merge)> == 1);
  static_assert(vecops::vec::details::option_count_v<
      vecops::vec::details::IsFirstOption,
      decltype(masked), decltype(first), decltype(merge)> == 1);
  EXPECT_EQ(
      &vecops::vec::details::find_option<
          vecops::vec::details::IsMaskedOption>(masked, first, merge).value,
      &mask);
  EXPECT_EQ(
      &vecops::vec::details::find_option<
          vecops::vec::details::IsVectorMergeOption>(
              masked, first, merge).value,
      &value);
  EXPECT_EQ(
      &vecops::vec::details::find_option<
          vecops::vec::details::IsMaskMergeOption>(mask_merge).value,
      &mask);
}
