//
// MaskConversionsTest.cpp
// Comprehensive test for Mask promote/demote/convert operations
// Mirrors ConversionTest.cpp structure: uses MaskXxxCase + Rebind
// for matching element counts, plus POW2 coverage macros.
//
// Supports: x86 SSE/AVX/AVX-512, ARM SVE, and scalar fallback.
//

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

// ============================================================================
// Helper utilities
// ============================================================================

namespace test_utils {

template <typename T>
T* alloc_aligned(size_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

// Create a bool pattern of `size` elements, where bit i is set
// if (i & mask_bits) == match_val.
std::vector<bool> make_pattern(nint_t size, int mask_bits, int match_val) {
  std::vector<bool> pattern((size_t)size);
  for (nint_t i = 0; i < size; ++i)
    pattern[(size_t)i] = ((int)i & mask_bits) == match_val;
  return pattern;
}

// Create a Mask<Tag> from a std::vector<bool> pattern.
// Uses the tag's element type to create data: 1 for true, 0 for false.
template <typename Tag>
Mask<Tag> make_mask(Tag tt, const std::vector<bool>& pattern) {
  using T = TypeOf<Tag>;
  auto* buf = alloc_aligned<T>((size_t)size(tt));
  for (nint_t i = 0; i < size(tt); ++i)
    buf[i] = pattern[(size_t)i] ? static_cast<T>(1) : static_cast<T>(0);
  auto v = loadu(tt, buf);
  std::free(buf);
  return cmpeq(v, fill(tt, static_cast<T>(1)));
}

} // namespace test_utils

// ============================================================================
// Case structs for parameterizing tests at specific POW2 vector-width levels
// ============================================================================

template <typename TIn_, typename TOut_, int POW2_In_>
struct MaskPromoteCase {
  using TIn    = TIn_;
  using TOut   = TOut_;
  using InTag  = ScalableTag<TIn,  POW2_In_>;
  using OutTag = Rebind<TOut, InTag>;
};

template <typename TIn_, typename TOut_, int POW2_Out_>
struct MaskDemoteCase {
  using TIn    = TIn_;
  using TOut   = TOut_;
  using OutTag = ScalableTag<TOut, POW2_Out_>;
  using InTag  = Rebind<TIn, OutTag>;
};

template <typename T1_, typename T2_, int POW2_>
struct MaskConvertCase {
  using TIn    = T1_;
  using TOut   = T2_;
  using InTag  = ScalableTag<TIn,  POW2_>;
  using OutTag = ScalableTag<TOut, POW2_>;
};

// ------------------------------------------------------------------
// Macros: enumerate all valid POW2 levels per shift category
//   shift = log2(sizeof(larger)/sizeof(smaller))
//   1 (ratio 2:1)  -> POW2 = 0, -1
//   2 (ratio 4:1)  -> POW2 = 0, -1, -2
//   3 (ratio 8:1)  -> POW2 = -1, -2, -3  (0 excluded: requires 8-word vectors)
// ------------------------------------------------------------------

#define MASK_PROMOTE_SHIFT1(T1, T2)  MaskPromoteCase<T1, T2, 0>, MaskPromoteCase<T1, T2, -1>
#define MASK_DEMOTE_SHIFT1(T1, T2)   MaskDemoteCase<T1, T2, 0>, MaskDemoteCase<T1, T2, -1>

#define MASK_PROMOTE_SHIFT2(T1, T2)  MaskPromoteCase<T1, T2, 0>, MaskPromoteCase<T1, T2, -1>, MaskPromoteCase<T1, T2, -2>
#define MASK_DEMOTE_SHIFT2(T1, T2)   MaskDemoteCase<T1, T2, 0>, MaskDemoteCase<T1, T2, -1>, MaskDemoteCase<T1, T2, -2>

#if VEC_MAX_POW >= 3
#define MASK_PROMOTE_SHIFT3(T1, T2)  MaskPromoteCase<T1, T2, 0>, MaskPromoteCase<T1, T2, -1>, MaskPromoteCase<T1, T2, -2>, MaskPromoteCase<T1, T2, -3>
#define MASK_DEMOTE_SHIFT3(T1, T2)   MaskDemoteCase<T1, T2, 0>, MaskDemoteCase<T1, T2, -1>, MaskDemoteCase<T1, T2, -2>, MaskDemoteCase<T1, T2, -3>
#else
#define MASK_PROMOTE_SHIFT3(T1, T2)  MaskPromoteCase<T1, T2, -1>, MaskPromoteCase<T1, T2, -2>, MaskPromoteCase<T1, T2, -3>
#define MASK_DEMOTE_SHIFT3(T1, T2)   MaskDemoteCase<T1, T2, -1>, MaskDemoteCase<T1, T2, -2>, MaskDemoteCase<T1, T2, -3>
#endif

#define MASK_CONVERT_POWS(T1, T2)    MaskConvertCase<T1, T2, 0>, MaskConvertCase<T1, T2, 1>, MaskConvertCase<T1, T2, 2>

// ============================================================================
// Mask Promote Tests: smaller type -> larger type
// ============================================================================

template <typename TCase>
class VecMaskPromoteTest : public ::testing::Test {
protected:
  using TIn    = typename TCase::TIn;
  using TOut   = typename TCase::TOut;
  using InTag  = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;

  InTag  t_in_;
  OutTag t_out_;

  nint_t in_elements()  { return size(t_in_); }
  nint_t out_elements() { return size(t_out_); }

  void SetUp() override {}
  void TearDown() override {}
};

using MaskPromoteTypes = ::testing::Types<
    // 8-bit -> 16-bit (shift=1)
    MASK_PROMOTE_SHIFT1(int8_t,  int16_t),
    MASK_PROMOTE_SHIFT1(uint8_t, uint16_t),
    // 8-bit -> 32-bit (shift=2)
    MASK_PROMOTE_SHIFT2(int8_t,  int32_t),
    MASK_PROMOTE_SHIFT2(uint8_t, uint32_t),
    // 8-bit -> 64-bit (shift=3)
    MASK_PROMOTE_SHIFT3(int8_t,  int64_t),
    MASK_PROMOTE_SHIFT3(uint8_t, uint64_t),
    // 16-bit -> 32-bit (shift=1)
    MASK_PROMOTE_SHIFT1(int16_t, int32_t),
    MASK_PROMOTE_SHIFT1(uint16_t, uint32_t),
    // 16-bit -> 64-bit (shift=2)
    MASK_PROMOTE_SHIFT2(int16_t, int64_t),
    MASK_PROMOTE_SHIFT2(uint16_t, uint64_t),
    // 32-bit -> 64-bit (shift=1)
    MASK_PROMOTE_SHIFT1(int32_t, int64_t),
    MASK_PROMOTE_SHIFT1(uint32_t, uint64_t)
>;

TYPED_TEST_SUITE(VecMaskPromoteTest, MaskPromoteTypes);

TYPED_TEST(VecMaskPromoteTest, BasicPromote) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  // Pattern: every other element true (i & 1) == 0
  auto pattern = test_utils::make_pattern(n_in, 1, 0);
  auto m_in    = test_utils::make_mask(this->t_in_, pattern);
  auto m_out   = promote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = pattern[(size_t)i];
    bool actual   = get(this->t_out_, m_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
}

TYPED_TEST(VecMaskPromoteTest, AllTrue) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  std::vector<bool> pattern((size_t)n_in, true);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = promote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(this->t_out_, m_out, i)) << "i=" << i;
}

TYPED_TEST(VecMaskPromoteTest, AllFalse) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  std::vector<bool> pattern((size_t)n_in, false);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = promote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_FALSE(get(this->t_out_, m_out, i)) << "i=" << i;
}

TYPED_TEST(VecMaskPromoteTest, AlternatingPattern) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  // Alternating T/F/T/F/...
  std::vector<bool> pattern((size_t)n_in);
  for (nint_t i = 0; i < n_in; ++i) pattern[(size_t)i] = (i % 2 == 0);

  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = promote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(pattern[(size_t)i], get(this->t_out_, m_out, i)) << "i=" << i;
  }
}

// ============================================================================
// Mask Demote Tests: larger type -> smaller type
// ============================================================================

template <typename TCase>
class VecMaskDemoteTest : public ::testing::Test {
protected:
  using TIn    = typename TCase::TIn;
  using TOut   = typename TCase::TOut;
  using OutTag = typename TCase::OutTag;
  using InTag  = typename TCase::InTag;

  InTag  t_in_;
  OutTag t_out_;

  nint_t in_elements()  { return size(t_in_); }
  nint_t out_elements() { return size(t_out_); }

  void SetUp() override {}
  void TearDown() override {}
};

using MaskDemoteTypes = ::testing::Types<
    // 16-bit -> 8-bit (shift=1)
    MASK_DEMOTE_SHIFT1(int16_t, int8_t),
    MASK_DEMOTE_SHIFT1(uint16_t, uint8_t),
    // 32-bit -> 16-bit (shift=1)
    MASK_DEMOTE_SHIFT1(int32_t, int16_t),
    MASK_DEMOTE_SHIFT1(uint32_t, uint16_t),
    // 64-bit -> 32-bit (shift=1)
    MASK_DEMOTE_SHIFT1(int64_t, int32_t),
    MASK_DEMOTE_SHIFT1(uint64_t, uint32_t),
    // 32-bit -> 8-bit (shift=2)
    MASK_DEMOTE_SHIFT2(int32_t, int8_t),
    MASK_DEMOTE_SHIFT2(uint32_t, uint8_t),
    // 64-bit -> 16-bit (shift=2)
    MASK_DEMOTE_SHIFT2(int64_t, int16_t),
    MASK_DEMOTE_SHIFT2(uint64_t, uint16_t)
>;

TYPED_TEST_SUITE(VecMaskDemoteTest, MaskDemoteTypes);

TYPED_TEST(VecMaskDemoteTest, BasicDemote) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  // Pattern: (i & 3) == 0
  auto pattern = test_utils::make_pattern(n_in, 3, 0);
  auto m_in    = test_utils::make_mask(this->t_in_, pattern);
  auto m_out   = demote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = pattern[(size_t)i];
    bool actual   = get(this->t_out_, m_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
}

TYPED_TEST(VecMaskDemoteTest, AllTrue) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  std::vector<bool> pattern((size_t)n_in, true);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = demote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(this->t_out_, m_out, i)) << "i=" << i;
}

TYPED_TEST(VecMaskDemoteTest, AllFalse) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  std::vector<bool> pattern((size_t)n_in, false);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = demote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_FALSE(get(this->t_out_, m_out, i)) << "i=" << i;
}

TYPED_TEST(VecMaskDemoteTest, AlternatingPattern) {
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  std::vector<bool> pattern((size_t)n_in);
  for (nint_t i = 0; i < n_in; ++i) pattern[(size_t)i] = (i & 1) != 0;

  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = demote(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(pattern[(size_t)i], get(this->t_out_, m_out, i)) << "i=" << i;
  }
}

TYPED_TEST(VecMaskDemoteTest, RoundTrip) {
  // promote then demote should restore original mask bits
  nint_t n_out = this->out_elements();
  nint_t N = n_out;

  // Create mask on OutTag (the smaller type), promote to InTag, demote back
  std::vector<bool> pattern((size_t)n_out);
  for (nint_t i = 0; i < n_out; ++i) pattern[(size_t)i] = (i % 5 < 2);

  auto m_small  = test_utils::make_mask(this->t_out_, pattern);
  auto promoted  = promote(this->t_in_, this->t_out_, m_small);
  auto demoted   = demote(this->t_out_, this->t_in_, promoted);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(pattern[(size_t)i], get(this->t_out_, demoted, i)) << "i=" << i;
  }
}

// ============================================================================
// Mask Convert Tests: same-size type conversions
// ============================================================================

template <typename TCase>
class VecMaskConvertTest : public ::testing::Test {
protected:
  using TIn    = typename TCase::TIn;
  using TOut   = typename TCase::TOut;
  using InTag  = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;
  static_assert(sizeof(TIn) == sizeof(TOut), "Convert requires same-size types");

  InTag  t_in_;
  OutTag t_out_;

  nint_t elements() { return size(t_in_); }

  void SetUp() override {}
  void TearDown() override {}
};

using MaskConvertTypes = ::testing::Types<
    // 8-bit
    MASK_CONVERT_POWS(int8_t, uint8_t),
    MASK_CONVERT_POWS(uint8_t, int8_t),
    // 16-bit
    MASK_CONVERT_POWS(int16_t, uint16_t),
    MASK_CONVERT_POWS(uint16_t, int16_t),
    // 32-bit
    MASK_CONVERT_POWS(int32_t, uint32_t),
    MASK_CONVERT_POWS(int32_t, float32_t),
    MASK_CONVERT_POWS(uint32_t, int32_t),
    MASK_CONVERT_POWS(uint32_t, float32_t),
    MASK_CONVERT_POWS(float32_t, int32_t),
    MASK_CONVERT_POWS(float32_t, uint32_t),
    // 64-bit
    MASK_CONVERT_POWS(int64_t, uint64_t),
    MASK_CONVERT_POWS(int64_t, float64_t),
    MASK_CONVERT_POWS(uint64_t, int64_t),
    MASK_CONVERT_POWS(uint64_t, float64_t),
    MASK_CONVERT_POWS(float64_t, int64_t),
    MASK_CONVERT_POWS(float64_t, uint64_t)
>;

TYPED_TEST_SUITE(VecMaskConvertTest, MaskConvertTypes);

TYPED_TEST(VecMaskConvertTest, BasicConvert) {
  nint_t N = this->elements();

  // Alternating pattern
  std::vector<bool> pattern((size_t)N);
  for (nint_t i = 0; i < N; ++i) pattern[(size_t)i] = (i % 2 == 0);

  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = convert(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(pattern[(size_t)i], get(this->t_out_, m_out, i)) << "i=" << i;
  }
}

TYPED_TEST(VecMaskConvertTest, AllTrue) {
  nint_t N = this->elements();

  std::vector<bool> pattern((size_t)N, true);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = convert(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_TRUE(get(this->t_out_, m_out, i)) << "i=" << i;
}

TYPED_TEST(VecMaskConvertTest, AllFalse) {
  nint_t N = this->elements();

  std::vector<bool> pattern((size_t)N, false);
  auto m_in  = test_utils::make_mask(this->t_in_, pattern);
  auto m_out = convert(this->t_out_, this->t_in_, m_in);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_FALSE(get(this->t_out_, m_out, i)) << "i=" << i;
}
