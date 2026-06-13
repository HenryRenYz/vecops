//
// MaskConversionsTest.cpp
// Comprehensive test for Mask promote/demote/convert operations
//
// Supports: x86 SSE/AVX/AVX-512, ARM SVE, and scalar fallback.
//

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

// Create a mask with a specific bool pattern
template <typename Tag>
Mask<Tag> make_mask(Tag tt, const std::vector<bool>& pattern) {
  using T = TypeOf<Tag>;
  auto* buf = static_cast<T*>(std::aligned_alloc(DEFAULT_ALIGNMENT,
                                                  (size_t)size(tt) * sizeof(T)));
  for (nint_t i = 0; i < size(tt); ++i) {
    buf[i] = pattern[(size_t)i] ? static_cast<T>(1) : static_cast<T>(0);
  }
  auto v = loadu(tt, buf);
  std::free(buf);
  return cmpeq(v, fill(tt, static_cast<T>(1)));
}

} // namespace

// ============================================================================
// Test fixture — runs on all element types with single-word tag
// ============================================================================

template <typename T>
struct MaskConvFixture : public ::testing::Test {
  using Elem = T;
  ScalableTag<T, 0> t;
  nint_t full_size = size(t);

  void SetUp() override {}
};

using TestedElemTypes = ::testing::Types<
    float32_t, float64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t
>;
TYPED_TEST_SUITE(MaskConvFixture, TestedElemTypes);

// ============================================================================
// convert: same-size identity
// ============================================================================

TYPED_TEST(MaskConvFixture, ConvertIdentity) {
  using T = typename TestFixture::Elem;
  auto& t = this->t;
  nint_t N = this->full_size;

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[(size_t)i] = (i % 2 == 0);

  auto m = make_mask(t, pattern);

  if constexpr (sizeof(T) == 4) {
    ScalableTag<int32_t, 0> t_i32;
    auto r = convert(t_i32, t, m);
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
  } else if constexpr (sizeof(T) == 8) {
    ScalableTag<int64_t, 0> t_i64;
    auto r = convert(t_i64, t, m);
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
  }
}

// ============================================================================
// promote: widen lanes (output element count shrinks for same byte width)
// ============================================================================

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt16) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N_in = size(t_i8);
  nint_t N_out = size(t_i16);

  std::vector<bool> pattern(N_in);
  pattern[0] = true;
  pattern[1] = false;
  if (N_in > 2) pattern[2] = true;

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i16, t_i8, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt32) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N_in = size(t_i8);
  nint_t N_out = size(t_i32);

  std::vector<bool> pattern(N_in);
  for (nint_t i = 0; i < N_in; ++i) pattern[i] = (i % 3 == 0);

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i32, t_i8, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt8ToInt64) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N_in = size(t_i8);
  nint_t N_out = size(t_i64);

  std::vector<bool> pattern(N_in);
  for (nint_t i = 0; i < N_in; ++i) pattern[i] = (i & 1) != 0;

  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i64, t_i8, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt16ToInt32) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N_in = size(t_i16);
  nint_t N_out = size(t_i32);

  std::vector<bool> pattern(N_in);
  for (nint_t i = 0; i < N_in; ++i) pattern[i] = (i % 2 == 1);

  auto m = make_mask(t_i16, pattern);
  auto r = promote(t_i32, t_i16, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt16ToInt64) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N_in = size(t_i16);
  nint_t N_out = size(t_i64);

  std::vector<bool> pattern(N_in);
  for (nint_t i = 0; i < N_in; ++i) pattern[i] = (i < N_in / 2);

  auto m = make_mask(t_i16, pattern);
  auto r = promote(t_i64, t_i16, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, PromoteInt32ToInt64) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N_in = size(t_i32);
  nint_t N_out = size(t_i64);

  std::vector<bool> pattern(N_in);
  for (nint_t i = 0; i < N_in; ++i) pattern[i] = (i % 4 == 0);

  auto m = make_mask(t_i32, pattern);
  auto r = promote(t_i64, t_i32, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_EQ(pattern[i], get(t_i64, r, i)) << "at " << i;
}

// ============================================================================
// demote: narrow lanes (output element count grows for same byte width)
// ============================================================================

TYPED_TEST(MaskConvFixture, DemoteInt16ToInt8) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int8_t, 0> t_i8;
  nint_t N = size(t_i16);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 3 == 1);

  auto m = make_mask(t_i16, pattern);
  auto r = demote(t_i8, t_i16, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i8, r, i)) << "at " << i;
  // Remaining output lanes should be false
  for (nint_t i = N; i < size(t_i8); ++i)
    EXPECT_FALSE(get(t_i8, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt32ToInt16) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N = size(t_i32);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i & 1) != 0;

  auto m = make_mask(t_i32, pattern);
  auto r = demote(t_i16, t_i32, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
  for (nint_t i = N; i < size(t_i16); ++i)
    EXPECT_FALSE(get(t_i16, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt64ToInt32) {
  ScalableTag<int64_t, 0> t_i64;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N = size(t_i64);

  std::vector<bool> pattern(N, true);

  auto m = make_mask(t_i64, pattern);
  auto r = demote(t_i32, t_i64, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(t_i32, r, i)) << "at " << i;
  for (nint_t i = N; i < size(t_i32); ++i)
    EXPECT_FALSE(get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt32ToInt8) {
  ScalableTag<int32_t, 0> t_i32;
  ScalableTag<int8_t, 0> t_i8;
  nint_t N = size(t_i32);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i & 1) == 0;

  auto m = make_mask(t_i32, pattern);
  auto r = demote(t_i8, t_i32, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i8, r, i)) << "at " << i;
  for (nint_t i = N; i < size(t_i8); ++i)
    EXPECT_FALSE(get(t_i8, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, DemoteInt64ToInt16) {
  ScalableTag<int64_t, 0> t_i64;
  ScalableTag<int16_t, 0> t_i16;
  nint_t N = size(t_i64);

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 2 == 1);

  auto m = make_mask(t_i64, pattern);
  auto r = demote(t_i16, t_i64, m);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, r, i)) << "at " << i;
  for (nint_t i = N; i < size(t_i16); ++i)
    EXPECT_FALSE(get(t_i16, r, i)) << "at " << i;
}

// ============================================================================
// Edge cases
// ============================================================================

TYPED_TEST(MaskConvFixture, AllFalse) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int32_t, 0> t_i32;
  nint_t N_in = size(t_i8);
  nint_t N_out = size(t_i32);

  std::vector<bool> pattern(N_in, false);
  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i32, t_i8, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_FALSE(get(t_i32, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, AllTrue) {
  ScalableTag<int8_t, 0> t_i8;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N_in = size(t_i8);
  nint_t N_out = size(t_i64);

  std::vector<bool> pattern(N_in, true);
  auto m = make_mask(t_i8, pattern);
  auto r = promote(t_i64, t_i8, m);

  for (nint_t i = 0; i < N_out; ++i)
    EXPECT_TRUE(get(t_i64, r, i)) << "at " << i;
}

TYPED_TEST(MaskConvFixture, RoundTrip) {
  ScalableTag<int16_t, 0> t_i16;
  ScalableTag<int64_t, 0> t_i64;
  nint_t N = size(t_i16);   // also = size(t_i64) * 4

  std::vector<bool> pattern(N);
  for (nint_t i = 0; i < N; ++i) pattern[i] = (i % 5 < 2);

  auto m = make_mask(t_i16, pattern);
  auto promoted = promote(t_i64, t_i16, m);
  auto demoted = demote(t_i16, t_i64, promoted);

  // Only first size(t_i64) bits survive the roundtrip
  nint_t N_survive = size(t_i64);
  for (nint_t i = 0; i < N_survive; ++i)
    EXPECT_EQ(pattern[i], get(t_i16, demoted, i)) << "at " << i;
  // Remaining positions should be false (demote fills with zeros)
  for (nint_t i = N_survive; i < N; ++i)
    EXPECT_FALSE(get(t_i16, demoted, i)) << "at " << i;
}
