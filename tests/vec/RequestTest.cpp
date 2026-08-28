#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>
#include <vector>

#include "TestHelpers.h"
#include "vecops/Features.h"
#if defined(HAS_SME)
#include "vecops/execution/details/arm/Resources.h"
#endif
#include "vecops/vec/Vec.h"

namespace vec = vecops::vec;
using vecops::nint_t;

namespace {

using Tag = vec::ScalableTag<vecops::float32_t, 0>;
Tag tag{};

std::vector<vecops::float32_t> iota_buffer(nint_t lanes, nint_t stride = 1) {
  std::vector<vecops::float32_t> values(
      static_cast<std::size_t>(lanes * stride + 8), -1.0f);
  for (nint_t i = 0; i < stride * lanes + 8; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<vecops::float32_t>(i);
  }
  return values;
}

void expect_same_lanes(Tag comparison_tag, vec::Vec<Tag> packed,
                       vec::Vec<Tag> resolved) {
  for (nint_t lane = 0; lane < vec::size(comparison_tag); ++lane) {
    ASSERT_TRUE(vec_test::values_identical(
        vec::get(comparison_tag, packed, lane),
        vec::get(comparison_tag, resolved, lane)))
        << "lane=" << lane;
  }
}

} // namespace

using ResolvedLoad = decltype(vec::details::resolve_load_request<Tag>());
using ResolvedStore = decltype(vec::details::resolve_store_request<Tag>());
using ResolvedOp = decltype(vec::details::resolve_op_request<Tag>());
using ResolvedFirst = decltype(
    vec::details::resolve_load_request<Tag>(vec::opt::first(0)));
using ResolvedStrided = decltype(
    vec::details::resolve_store_request<Tag>(vec::strided(nint_t{1})));
struct TestResource {};
using TestResources = vecops::execution::details::ResourceSet<TestResource>;
using ResolvedResourceLoad = decltype(
    vec::details::resolve_load_request<Tag, TestResources>(
        vec::strided(nint_t{1})));
using ResolvedOptionResourceLoad = decltype(
    vec::details::resolve_load_request<Tag>(
        vec::strided(nint_t{3}, vec::scale<1>),
        vec::resources<TestResources>));
using ResolvedResourceLoadConvert = decltype(
    vec::details::resolve_load_convert_request<
        Tag, int16_t, TestResources>(vec::strided(nint_t{1})));
using ResolvedResourceStoreConvert = decltype(
    vec::details::resolve_store_convert_request<
        Tag, int16_t, TestResources>(vec::strided(nint_t{1})));
static_assert(std::is_empty_v<ResolvedLoad>);
static_assert(std::is_empty_v<ResolvedStore>);
static_assert(std::is_empty_v<ResolvedOp>);
static_assert(
    sizeof(ResolvedFirst) == sizeof(nint_t));
static_assert(sizeof(ResolvedStrided) == sizeof(nint_t));
static_assert(std::same_as<
              typename ResolvedResourceLoad::ActiveResources,
              TestResources>);
static_assert(ResolvedOptionResourceLoad::index_scale == 1);
static_assert(std::same_as<
              typename ResolvedOptionResourceLoad::ActiveResources,
              TestResources>);
static_assert(std::same_as<
              typename ResolvedResourceLoadConvert::ActiveResources,
              TestResources>);
static_assert(std::same_as<
              typename ResolvedResourceStoreConvert::ActiveResources,
              TestResources>);

TEST(VecRequestTest, ResolvedLoadMatchesOptionPackLoad) {
  const nint_t lanes = vec::size(tag);
  const auto buffer = iota_buffer(lanes);

  {
    const auto packed = vec::load(tag, buffer.data(), vec::opt::unmasked);
    vec::LoadRequest<Tag> request{};
    const auto resolved = vec::load(tag, buffer.data(), request);
    expect_same_lanes(tag, packed, resolved);
  }
  {
    const auto packed = vec::load(
        tag, buffer.data(), vec::opt::first(3));
    vec::LoadRequest<Tag, vec::Active::First> request{};
    request.first_count = 3;
    const auto resolved = vec::load(tag, buffer.data(), request);
    expect_same_lanes(tag, packed, resolved);
  }
  {
    const auto mask = vec::mwhilelt(tag, 0, 2);
    const auto packed = vec::load(
        tag, buffer.data(), vec::opt::masked(mask), vec::opt::merge(7.0f));
    vec::LoadRequest<
        Tag, vec::Active::Masked, vec::Addressing::Contiguous,
        vec::Populate::MergeScalar>
        request{};
    request.mask = &mask;
    request.merge_scalar = 7.0f;
    const auto resolved = vec::load(tag, buffer.data(), request);
    expect_same_lanes(tag, packed, resolved);
    EXPECT_FLOAT_EQ(vec::get(tag, resolved, 1), 1.0f);
    EXPECT_FLOAT_EQ(vec::get(tag, resolved, 2), 7.0f);
  }
  {
    const auto packed =
        vec::load(tag, buffer.data(), vec::opt::strided(nint_t(2)));
    vec::LoadRequest<Tag, vec::Active::Unmasked, vec::Addressing::Strided>
        request{};
    request.stride = 2;
    const auto resolved = vec::load(tag, buffer.data(), request);
    expect_same_lanes(tag, packed, resolved);
    EXPECT_FLOAT_EQ(vec::get(tag, resolved, 3), 6.0f);
  }
}

TEST(VecRequestTest, ResolvedStoreMatchesOptionPackStore) {
  const nint_t lanes = vec::size(tag);
  const auto buffer = iota_buffer(lanes);
  const auto value = vec::load(tag, buffer.data(), vec::opt::unmasked);

  std::vector<vecops::float32_t> packed_out(
      static_cast<std::size_t>(lanes), -1.0f);
  std::vector<vecops::float32_t> resolved_out(
      static_cast<std::size_t>(lanes), -1.0f);

  vec::store(tag, packed_out.data(), value, vec::opt::first(5));
  vec::StoreRequest<Tag, vec::Active::First> request{};
  request.first_count = 5;
  vec::store(tag, resolved_out.data(), value, request);

  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(
        packed_out[static_cast<std::size_t>(lane)],
        resolved_out[static_cast<std::size_t>(lane)]);
    if (lane < 5) {
      EXPECT_FLOAT_EQ(
          resolved_out[static_cast<std::size_t>(lane)],
          static_cast<vecops::float32_t>(lane));
    } else {
      EXPECT_FLOAT_EQ(resolved_out[static_cast<std::size_t>(lane)], -1.0f);
    }
  }
}

TEST(VecRequestTest, ScalarConvertingRequestsPreserveStridesAndTails) {
  const nint_t lanes = vec::size(tag);
  std::vector<int16_t> input(
      static_cast<std::size_t>(lanes * 2), int16_t{-1});
  for (nint_t lane = 0; lane < lanes; ++lane) {
    input[static_cast<std::size_t>(lane * 2)] =
        static_cast<int16_t>(lane + 10);
  }

  using IndexVector = vec::Vec<vec::IndexTag<Tag>>;
  const auto active_mask = vec::mwhilelt(tag, 0, lanes - 1);
  using LoadRequest = vec::LoadConvertRequest<
      Tag, int16_t, vec::Active::Masked, vec::Addressing::Strided,
      vec::Populate::Zero, vec::mem::Unaligned, vec::mem::Temporal,
      0, IndexVector, vec::cvt::Ordered, vec::cvt::Saturate,
      vec::Mask<Tag>, TestResources>;
  LoadRequest load_request{};
  load_request.mask = &active_mask;
  load_request.stride = 2;
  const auto loaded = vec::details::execute_scalar_load_convert_request(
      tag, input.data(), load_request);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, loaded, lane),
        lane + 1 < lanes ? static_cast<float>(lane + 10) : 0.0f);
  }

  using StoreRequest = vec::StoreConvertRequest<
      Tag, int16_t, vec::Active::Masked, vec::Addressing::Strided,
      vec::mem::Unaligned, vec::mem::Temporal, 0, IndexVector,
      vec::cvt::Ordered, vec::cvt::Saturate, vec::mem::Packed,
      vec::Mask<Tag>, TestResources>;
  StoreRequest store_request{};
  store_request.mask = &active_mask;
  store_request.stride = 2;
  std::vector<int16_t> output(
      static_cast<std::size_t>(lanes * 2), int16_t{-1});
  vec::details::execute_scalar_store_convert_request(
      tag, output.data(), loaded, store_request);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(
        output[static_cast<std::size_t>(lane * 2)],
        lane + 1 < lanes ? static_cast<int16_t>(lane + 10) : int16_t{-1});
  }
}

#if defined(HAS_SME)
TEST(VecRequestTest, StreamingResourcesAcceptByteStridedOptions) {
  using WordTag = vec::ScalableTag<uint32_t, 0>;
  using StreamingResources = vecops::execution::details::ResourceSet<
      vecops::execution::details::arm::StreamingZA>;
  WordTag word_tag{};
  const nint_t lanes = vec::size(word_tag);
  constexpr nint_t ByteStride = 6;
  std::vector<unsigned char> storage(
      static_cast<std::size_t>(lanes * ByteStride + sizeof(uint32_t)), 0);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    const uint32_t value = static_cast<uint32_t>(lane + 100);
    std::memcpy(
        storage.data() + lane * ByteStride, &value, sizeof(value));
  }
  const auto loaded = vec::load(
      word_tag, reinterpret_cast<const uint32_t*>(storage.data()),
      vec::opt::first(lanes),
      vec::strided(ByteStride, vec::scale<1>),
      vec::resources<StreamingResources>);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(vec::get(word_tag, loaded, lane),
              static_cast<uint32_t>(lane + 100));
  }
}
#endif
