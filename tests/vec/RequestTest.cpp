#include <gtest/gtest.h>

#include <vector>

#include "TestHelpers.h"
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
    ASSERT_FLOAT_EQ(
        vec::get(comparison_tag, packed, lane),
        vec::get(comparison_tag, resolved, lane))
        << "lane=" << lane;
  }
}

} // namespace

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
