//
// VecConvertTest.cpp
// Comprehensive test for vector type conversions
// Covers: promote, demote, convert
//

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

#include "TestUtils.h"
#include "vecops/vec/Vec.h"
#include "vecops/util/ScalarConvert.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

template <typename TIn, typename TOut>
TIn get_promote_extreme_test_value(nint_t i) {
  if constexpr (vecops::is_float<TIn> && vecops::is_int<TOut>) {
    if constexpr (vecops::is_unsigned_int<TOut>) {
      const int values[] = {123, 1, 0, 7};
      return TIn(values[i % 4]);
    } else {
      const int values[] = {123, -123, 0, -1};
      return TIn(values[i % 4]);
    }
  } else {
    if (i % 4 == 0) return std::numeric_limits<TIn>::max();
    if (i % 4 == 1) return std::numeric_limits<TIn>::min();
    if (i % 4 == 2) return TIn(0);
    return TIn(-1);
  }
}

} // namespace

// ============================================================================
// Case structs for parameterizing tests at specific POW2 vector-width levels
// ============================================================================

template <typename T1_, typename T2_, int POW2_In_>
struct PromoteCase {
  using TIn    = T1_;
  using TOut   = T2_;
  using InTag  = ScalableTag<TIn,  POW2_In_>;
  using OutTag = Rebind<TOut, InTag>;
};

template <typename T1_, typename T2_, int POW2_Out_>
struct DemoteCase {
  using TIn    = T1_;
  using TOut   = T2_;
  using OutTag = ScalableTag<TOut, POW2_Out_>;
  using InTag  = Rebind<TIn, OutTag>;
};

template <typename T1_, typename T2_, int POW2_>
struct ConvertCase {
  using TIn    = T1_;
  using TOut   = T2_;
  using InTag  = ScalableTag<TIn,  POW2_>;
  using OutTag = ScalableTag<TOut, POW2_>;
};

template <typename T1_, typename T2_, int POW2_>
struct BitcastCase {
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

#define PROMOTE_SHIFT1(T1, T2)  PromoteCase<T1, T2, 0>, PromoteCase<T1, T2, -1>
#define DEMOTE_SHIFT1(T1, T2)   DemoteCase<T1, T2, 0>, DemoteCase<T1, T2, -1>

#define PROMOTE_SHIFT2(T1, T2)  PromoteCase<T1, T2, 0>, PromoteCase<T1, T2, -1>, PromoteCase<T1, T2, -2>
#define DEMOTE_SHIFT2(T1, T2)   DemoteCase<T1, T2, 0>, DemoteCase<T1, T2, -1>, DemoteCase<T1, T2, -2>

#if VEC_MAX_POW >= 3
#define PROMOTE_SHIFT3(T1, T2)  PromoteCase<T1, T2, 0>, PromoteCase<T1, T2, -1>, PromoteCase<T1, T2, -2>, PromoteCase<T1, T2, -3>
#define DEMOTE_SHIFT3(T1, T2)   PromoteCase<T1, T2, 0>, DemoteCase<T1, T2, -1>, DemoteCase<T1, T2, -2>, DemoteCase<T1, T2, -3>
#else
#define PROMOTE_SHIFT3(T1, T2)  PromoteCase<T1, T2, -1>, PromoteCase<T1, T2, -2>, PromoteCase<T1, T2, -3>
#define DEMOTE_SHIFT3(T1, T2)   DemoteCase<T1, T2, -1>, DemoteCase<T1, T2, -2>, DemoteCase<T1, T2, -3>
#endif

#define CONVERT_POWS(T1, T2)    ConvertCase<T1, T2, 0>, ConvertCase<T1, T2, 1>, ConvertCase<T1, T2, 2>
#define BITCAST_POWS(T1, T2)    BitcastCase<T1, T2, 0>, BitcastCase<T1, T2, 1>, BitcastCase<T1, T2, 2>

// ============================================================================
// Promote Tests: smaller type -> larger type
// ============================================================================

template <typename TCase>
class VecPromoteTest : public ::testing::Test {
protected:
  using TIn    = typename TCase::TIn;
  using TOut   = typename TCase::TOut;
  using InTag  = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;

  InTag  t_in_;
  OutTag t_out_;

  nint_t in_elements()  { return size(t_in_); }
  nint_t out_elements() { return size(t_out_); }

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

using PromoteTypes = ::testing::Types<
    // 8-bit -> 16-bit (shift=1)
    PROMOTE_SHIFT1(int8_t, int16_t),
    PROMOTE_SHIFT1(int8_t, uint16_t),
    PROMOTE_SHIFT1(int8_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    PROMOTE_SHIFT1(int8_t, vecops::bfloat16_t),
#endif
    PROMOTE_SHIFT1(uint8_t, uint16_t),
    PROMOTE_SHIFT1(uint8_t, int16_t),
    PROMOTE_SHIFT1(uint8_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    PROMOTE_SHIFT1(uint8_t, vecops::bfloat16_t),
#endif
    // 8-bit -> 32-bit (shift=2)
    PROMOTE_SHIFT2(int8_t, int32_t),
    PROMOTE_SHIFT2(int8_t, uint32_t),
    PROMOTE_SHIFT2(int8_t, float32_t),
    PROMOTE_SHIFT2(uint8_t, int32_t),
    PROMOTE_SHIFT2(uint8_t, uint32_t),
    PROMOTE_SHIFT2(uint8_t, float32_t),
    // 8-bit -> 64-bit (shift=3)
    PROMOTE_SHIFT3(int8_t, int64_t),
    PROMOTE_SHIFT3(int8_t, float64_t),
    PROMOTE_SHIFT3(uint8_t, int64_t),
    PROMOTE_SHIFT3(uint8_t, uint64_t),
    PROMOTE_SHIFT3(uint8_t, float64_t),
    // 16-bit -> 32-bit (shift=1)
    PROMOTE_SHIFT1(int16_t, int32_t),
    PROMOTE_SHIFT1(int16_t, uint32_t),
    PROMOTE_SHIFT1(int16_t, float32_t),
    PROMOTE_SHIFT1(uint16_t, int32_t),
    PROMOTE_SHIFT1(uint16_t, uint32_t),
    PROMOTE_SHIFT1(uint16_t, float32_t),
    PROMOTE_SHIFT1(vecops::float16_t, int32_t),
    PROMOTE_SHIFT1(vecops::float16_t, uint32_t),
    PROMOTE_SHIFT1(vecops::float16_t, float32_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    PROMOTE_SHIFT1(vecops::bfloat16_t, int32_t),
    PROMOTE_SHIFT1(vecops::bfloat16_t, uint32_t),
    PROMOTE_SHIFT1(vecops::bfloat16_t, float32_t),
#endif
    // 16-bit -> 64-bit (shift=2)
    PROMOTE_SHIFT2(int16_t, int64_t),
    PROMOTE_SHIFT2(int16_t, uint64_t),
    PROMOTE_SHIFT2(int16_t, float64_t),
    PROMOTE_SHIFT2(uint16_t, int64_t),
    PROMOTE_SHIFT2(uint16_t, uint64_t),
    PROMOTE_SHIFT2(uint16_t, float64_t),
    PROMOTE_SHIFT2(vecops::float16_t, int64_t),
    PROMOTE_SHIFT2(vecops::float16_t, uint64_t),
    PROMOTE_SHIFT2(vecops::float16_t, float64_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    PROMOTE_SHIFT2(vecops::bfloat16_t, int64_t),
    PROMOTE_SHIFT2(vecops::bfloat16_t, uint64_t),
    PROMOTE_SHIFT2(vecops::bfloat16_t, float64_t),
#endif
    // 32-bit -> 64-bit (shift=1)
    PROMOTE_SHIFT1(int32_t, int64_t),
    PROMOTE_SHIFT1(int32_t, uint64_t),
    PROMOTE_SHIFT1(int32_t, float64_t),
    PROMOTE_SHIFT1(uint32_t, uint64_t),
    PROMOTE_SHIFT1(uint32_t, int64_t),
    PROMOTE_SHIFT1(uint32_t, float64_t),
    PROMOTE_SHIFT1(float32_t, uint64_t),
    PROMOTE_SHIFT1(float32_t, int64_t),
    PROMOTE_SHIFT1(float32_t, float64_t)
>;

TYPED_TEST_SUITE(VecPromoteTest, PromoteTypes);

TYPED_TEST(VecPromoteTest, BasicPromote) {
  using TIn = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  auto v_in  = load(this->t_in_, this->in_data_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = vecops::convert<TOut>(this->in_data_[i]);
    TOut actual   = get(this->t_out_, v_out, i);
    EXPECT_TRUE(test_utils::values_near(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecPromoteTest, UnorderedRoundTripPreservesLaneOrder) {
  using TIn = typename TestFixture::TIn;
  nint_t N = this->in_elements();

  for (nint_t i = 0; i < N; ++i) {
    this->in_data_[i] = static_cast<TIn>(i + 1);
  }

  auto v_in = load(this->t_in_, this->in_data_);
  auto promoted = convert(this->t_out_, v_in, cvt::unordered);
  auto promoted_x = convert(this->t_out_, v_in, cvt::unordered);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_near(
        get(this->t_out_, promoted, i),
        get(this->t_out_, promoted_x, i)
    )) << "promoted i=" << i;
  }

#if !defined(CPU_CAPABILITY_SVE)
  auto promoted_ordered = convert(this->t_out_, v_in);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_near(
        get(this->t_out_, promoted_ordered, i),
        get(this->t_out_, promoted, i)
    )) << "ordered i=" << i;
  }
#endif

  auto restored = convert(this->t_in_, promoted, cvt::unordered);
  auto restored_x = convert(this->t_in_, promoted_x, cvt::unordered);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_near(
        this->in_data_[i], get(this->t_in_, restored, i)
    )) << "restored i=" << i;
    EXPECT_TRUE(test_utils::values_near(
        this->in_data_[i], get(this->t_in_, restored_x, i)
    )) << "restored_x i=" << i;
  }
}

TYPED_TEST(VecPromoteTest, PromoteWithZeroValues) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t N = this->out_elements();

  auto v_in  = zeros(this->t_in_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(TOut(0), get(this->t_out_, v_out, i)) << "i=" << i;
  }
}

TYPED_TEST(VecPromoteTest, PromoteWithMaxMinValues) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  auto data = test_utils::alloc_aligned<TIn>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = get_promote_extreme_test_value<TIn, TOut>(i);
  }

  auto v_in  = load(this->t_in_, data);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = vecops::convert<TOut>(data[i]);
    TOut actual   = get(this->t_out_, v_out, i);
    EXPECT_TRUE(test_utils::values_near(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(data[i]);
  }
  std::free(data);
}

// Sign extension test for signed types
TYPED_TEST(VecPromoteTest, SignExtensionTest) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  if constexpr (vecops::is_signed_int<TIn>) {
    auto data = test_utils::alloc_aligned<TIn>(N);
    for (nint_t i = 0; i < N; ++i) {
      data[i] = static_cast<TIn>(-(i + 1));
    }

    auto v_in  = load(this->t_in_, data);
    auto v_out = convert(this->t_out_, v_in);

    for (nint_t i = 0; i < N; ++i) {
      TOut actual   = get(this->t_out_, v_out, i);
      TIn  original = data[i];
      EXPECT_TRUE(test_utils::values_near(vecops::convert<TOut>(original), actual))
                << "i=" << i << " original=" << static_cast<long long>(original)
                << " actual=" << static_cast<long long>(actual);
    }
    std::free(data);
  }
}

// ============================================================================
// Demote Tests: larger type -> smaller type
// ============================================================================

template <typename TCase>
class VecDemoteTest : public ::testing::Test {
protected:
  using TIn    = typename TCase::TIn;
  using TOut   = typename TCase::TOut;
  using OutTag = typename TCase::OutTag;
  using InTag  = typename TCase::InTag;

  InTag  t_in_;
  OutTag t_out_;

  nint_t in_elements()  { return size(t_in_); }
  nint_t out_elements() { return size(t_out_); }

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (int i = 0; i < 256; ++i) {
      if constexpr (std::is_floating_point_v<TIn> ||
                    std::is_same_v<TIn, vecops::float16_t> ||
                    std::is_same_v<TIn, vecops::bfloat16_t>) {
        if constexpr (is_unsigned_int<TOut>) {
          in_data_[i] = static_cast<TIn>((i % 20) + 10 + 0.5); // make sure no negative value input (UB)
        } else {
          in_data_[i] = static_cast<TIn>((i % 20) - 10 + 0.5); // leading negative
        }
      } else if constexpr (std::is_signed_v<TIn>) {
        in_data_[i] = static_cast<TIn>((i % 100) - 50);
      } else {
        in_data_[i] = static_cast<TIn>(i % 100);
      }
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for demote tests
using DemoteTypes = ::testing::Types<
    // 16-bit -> 8-bit (shift=1)
    DEMOTE_SHIFT1(int16_t, int8_t),
    DEMOTE_SHIFT1(int16_t, uint8_t),
    DEMOTE_SHIFT1(uint16_t, int8_t),
    DEMOTE_SHIFT1(uint16_t, uint8_t),
    DEMOTE_SHIFT1(vecops::float16_t, int8_t),
    DEMOTE_SHIFT1(vecops::float16_t, uint8_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT1(vecops::bfloat16_t, int8_t),
    DEMOTE_SHIFT1(vecops::bfloat16_t, uint8_t),
#endif
    // 32-bit -> 16-bit (shift=1)
    DEMOTE_SHIFT1(int32_t, int16_t),
    DEMOTE_SHIFT1(int32_t, uint16_t),
    DEMOTE_SHIFT1(int32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT1(int32_t, vecops::bfloat16_t),
#endif
    DEMOTE_SHIFT1(uint32_t, int16_t),
    DEMOTE_SHIFT1(uint32_t, uint16_t),
    DEMOTE_SHIFT1(uint32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT1(uint32_t, vecops::bfloat16_t),
#endif
    DEMOTE_SHIFT1(float32_t, int16_t),
    DEMOTE_SHIFT1(float32_t, uint16_t),
    DEMOTE_SHIFT1(float32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT1(float32_t, vecops::bfloat16_t),
#endif
    // 32-bit -> 8-bit (shift=2)
    DEMOTE_SHIFT2(int32_t, int8_t),
    DEMOTE_SHIFT2(int32_t, uint8_t),
    DEMOTE_SHIFT2(uint32_t, int8_t),
    DEMOTE_SHIFT2(uint32_t, uint8_t),
    DEMOTE_SHIFT2(float32_t, int8_t),
    DEMOTE_SHIFT2(float32_t, uint8_t),
    // 64-bit -> 32-bit (shift=1)
    DEMOTE_SHIFT1(int64_t, int32_t),
    DEMOTE_SHIFT1(int64_t, uint32_t),
    DEMOTE_SHIFT1(int64_t, float32_t),
    DEMOTE_SHIFT1(uint64_t, int32_t),
    DEMOTE_SHIFT1(uint64_t, uint32_t),
    DEMOTE_SHIFT1(uint64_t, float32_t),
    DEMOTE_SHIFT1(float64_t, int32_t),
    DEMOTE_SHIFT1(float64_t, uint32_t),
    DEMOTE_SHIFT1(float64_t, float32_t),
    // 64-bit -> 16-bit (shift=2)
    DEMOTE_SHIFT2(int64_t, int16_t),
    DEMOTE_SHIFT2(int64_t, uint16_t),
    DEMOTE_SHIFT2(int64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT2(int64_t, vecops::bfloat16_t),
#endif
    DEMOTE_SHIFT2(uint64_t, int16_t),
    DEMOTE_SHIFT2(uint64_t, uint16_t),
    DEMOTE_SHIFT2(uint64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT2(uint64_t, vecops::bfloat16_t),
#endif
    DEMOTE_SHIFT2(float64_t, int16_t),
    DEMOTE_SHIFT2(float64_t, uint16_t),
    DEMOTE_SHIFT2(float64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DEMOTE_SHIFT2(float64_t, vecops::bfloat16_t),
#endif
    // 64-bit -> 8-bit (shift=3)
    DEMOTE_SHIFT3(int64_t, int8_t),
    DEMOTE_SHIFT3(int64_t, uint8_t),
    DEMOTE_SHIFT3(uint64_t, int8_t),
    DEMOTE_SHIFT3(uint64_t, uint8_t),
    DEMOTE_SHIFT3(float64_t, int8_t),
    DEMOTE_SHIFT3(float64_t, uint8_t)
>;

TYPED_TEST_SUITE(VecDemoteTest, DemoteTypes);

TYPED_TEST(VecDemoteTest, BasicDemote) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  auto v_in  = load(this->t_in_, this->in_data_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = vecops::convert<TOut>(this->in_data_[i]);
    TOut actual   = get(this->t_out_, v_out, i);
    EXPECT_TRUE(test_utils::values_near(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecDemoteTest, DemoteWithZeroValues) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  auto v_in  = zeros(this->t_in_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(TOut(0), get(this->t_out_, v_out, i)) << "i=" << i;
  }
}

// Truncation behavior test for overflow cases
TYPED_TEST(VecDemoteTest, TruncationBehavior) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t n_in  = this->in_elements();
  nint_t n_out = this->out_elements();
  nint_t N = n_in < n_out ? n_in : n_out;

  auto data = test_utils::alloc_aligned<TIn>(N);
  TIn max_out = static_cast<TIn>(std::numeric_limits<TOut>::max());
  TIn min_out = std::is_signed_v<TOut>
                ? static_cast<TIn>(std::numeric_limits<TOut>::min())
                : TIn(0);

  for (nint_t i = 0; i < N; ++i) {
    if (i % 3 == 0) data[i] = max_out;
    else if (i % 3 == 1 && std::is_signed_v<TOut>) data[i] = min_out;
    else data[i] = TIn(0);
  }

  auto v_in  = load(this->t_in_, data);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = vecops::convert<TOut>(data[i]);
    TOut actual   = get(this->t_out_, v_out, i);
    EXPECT_TRUE(test_utils::values_near(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(data[i]);
  }
  std::free(data);
}

// ============================================================================
// Convert Tests: same-size type conversions
// ============================================================================

template <typename TCase>
class VecConvertTest : public ::testing::Test {
protected:
  using TIn   = typename TCase::TIn;
  using TOut  = typename TCase::TOut;
  using InTag  = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;
  static_assert(sizeof(TIn) == sizeof(TOut), "Convert requires same-size types");

  InTag  t_in_;
  OutTag t_out_;

  nint_t elements() { return size(t_in_); }

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for convert tests (same size)
using ConvertTypes = ::testing::Types<
    // 8-bit conversions
    CONVERT_POWS(int8_t, uint8_t),
    CONVERT_POWS(uint8_t, int8_t),
    // 16-bit conversions
    CONVERT_POWS(int16_t, uint16_t),
    CONVERT_POWS(int16_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    CONVERT_POWS(int16_t, vecops::bfloat16_t),
#endif
    CONVERT_POWS(uint16_t, int16_t),
    CONVERT_POWS(uint16_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    CONVERT_POWS(uint16_t, vecops::bfloat16_t),
#endif
    CONVERT_POWS(vecops::float16_t, int16_t),
    CONVERT_POWS(vecops::float16_t, uint16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    CONVERT_POWS(vecops::float16_t, vecops::bfloat16_t),
    CONVERT_POWS(vecops::bfloat16_t, int16_t),
    CONVERT_POWS(vecops::bfloat16_t, uint16_t),
    CONVERT_POWS(vecops::bfloat16_t, vecops::float16_t),
#endif
    // 32-bit conversions
    CONVERT_POWS(int32_t, uint32_t),
    CONVERT_POWS(int32_t, float32_t),
    CONVERT_POWS(uint32_t, int32_t),
    CONVERT_POWS(uint32_t, float32_t),
    CONVERT_POWS(float32_t, int32_t),
    CONVERT_POWS(float32_t, uint32_t),
    // 64-bit conversions
    CONVERT_POWS(int64_t, uint64_t),
    CONVERT_POWS(int64_t, float64_t),
    CONVERT_POWS(uint64_t, int64_t),
    CONVERT_POWS(uint64_t, float64_t),
    CONVERT_POWS(float64_t, int64_t),
    CONVERT_POWS(float64_t, uint64_t)
>;

TYPED_TEST_SUITE(VecConvertTest, ConvertTypes);

TYPED_TEST(VecConvertTest, BasicConvert) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t N = this->elements();

  auto v_in  = load(this->t_in_, this->in_data_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = vecops::convert<TOut>(this->in_data_[i]);
    TOut actual   = get(this->t_out_, v_out, i);
    EXPECT_TRUE(test_utils::values_near(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecConvertTest, UnorderedConversionForwardsToConvert) {
  nint_t N = this->elements();

  auto v_in = load(this->t_in_, this->in_data_);
  auto expected = convert(this->t_out_, v_in);
  auto converted = convert(this->t_out_, v_in, cvt::unordered);
  auto converted_x = convert(this->t_out_, v_in, cvt::unordered);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_near(
        get(this->t_out_, expected, i),
        get(this->t_out_, converted, i)
    )) << "converted i=" << i;
    EXPECT_TRUE(test_utils::values_near(
        get(this->t_out_, expected, i),
        get(this->t_out_, converted_x, i)
    )) << "converted_x i=" << i;
  }
}

TYPED_TEST(VecConvertTest, ConvertWithZeroValues) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t N = this->elements();

  auto v_in  = zeros(this->t_in_);
  auto v_out = convert(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(TOut(0), get(this->t_out_, v_out, i)) << "i=" << i;
  }
}

// Test signed/unsigned reinterpretation
TYPED_TEST(VecConvertTest, SignedUnsignedConversion) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t N = this->elements();

  if constexpr (!std::is_floating_point_v<TIn>)
    if constexpr ((std::is_signed_v<TIn> && std::is_unsigned_v<TOut>) ||
                  (std::is_unsigned_v<TIn> && std::is_signed_v<TOut>)) {
      auto data = test_utils::alloc_aligned<TIn>(N);
      for (nint_t i = 0; i < N; ++i) {
        data[i] = static_cast<TIn>(~TIn(0) - i);
      }

      auto v_in  = load(this->t_in_, data);
      auto v_out = convert(this->t_out_, v_in);

      for (nint_t i = 0; i < N; ++i) {
        TOut expected = vecops::convert<TOut>(data[i]);
        TOut actual   = get(this->t_out_, v_out, i);
        EXPECT_TRUE(test_utils::values_near(expected, actual))
                  << "i=" << i << " input=" << static_cast<long long>(data[i])
                  << " expected=" << static_cast<long long>(expected);
      }
      std::free(data);
    }
}

// Float to int and int to float conversion test
TYPED_TEST(VecConvertTest, FloatIntConversion) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;
  nint_t N = this->elements();

  if constexpr ((std::is_floating_point_v<TIn> && std::is_integral_v<TOut>) ||
                (std::is_integral_v<TIn> && std::is_floating_point_v<TOut>)) {
    auto data = test_utils::alloc_aligned<TIn>(N);
    for (nint_t i = 0; i < N; ++i) {
      if constexpr (std::is_floating_point_v<TIn>) {
        if constexpr (std::is_unsigned_v<TOut>) {
          data[i] = static_cast<TIn>(i * 10 + 33);
        } else {
          data[i] = static_cast<TIn>(i * 10 - 50);
        }
      } else {
        data[i] = static_cast<TIn>(i * 100);
      }
    }

    auto v_in  = load(this->t_in_, data);
    auto v_out = convert(this->t_out_, v_in);

    for (nint_t i = 0; i < N; ++i) {
      TOut expected = vecops::convert<TOut>(data[i]);
      TOut actual   = get(this->t_out_, v_out, i);
      EXPECT_TRUE(test_utils::values_near(expected, actual))
                << "i=" << i << " input=" << static_cast<long long>(data[i]);
    }
    std::free(data);
  }
}

// ============================================================================
// Corner Case Tests
// ============================================================================

class VecConvertCornerCaseTest : public ::testing::Test {
protected:
  void SetUp() override {
    data_256_ = test_utils::alloc_aligned<uint8_t>(256);
    memset(data_256_, 0, 256);
  }

  void TearDown() override {
    std::free(data_256_);
  }

  uint8_t* data_256_{};
};

// Test: int8_t (-1) -> uint16_t (should be 65535 via sign extension)
TEST_F(VecConvertCornerCaseTest, PromoteNegativeToUnsigned) {
  ScalableTag<int8_t> t_in;
  Rebind<uint16_t, decltype(t_in)> t_out;
  nint_t N = std::min(size(t_in), size(t_out));

  auto data = test_utils::alloc_aligned<int8_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = -1;
  }

  auto v_in  = load(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    uint16_t actual = get(t_out, v_out, i);
    EXPECT_EQ(uint16_t(65535), actual) << "i=" << i;
  }
  std::free(data);
}

// Test: uint8_t (255) -> int16_t (should be 255, not -1)
TEST_F(VecConvertCornerCaseTest, PromoteUnsignedToSigned) {
  ScalableTag<uint8_t> t_in;
  Rebind<int16_t, decltype(t_in)> t_out;
  nint_t N = std::min(size(t_in), size(t_out));

  auto data = test_utils::alloc_aligned<uint8_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = 255;
  }

  auto v_in  = load(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    int16_t actual = get(t_out, v_out, i);
    EXPECT_EQ(int16_t(255), actual) << "i=" << i;
  }
  std::free(data);
}

// Test: int16_t (-1) -> int8_t (truncation to -1)
TEST_F(VecConvertCornerCaseTest, DemoteNegativeValue) {
  ScalableTag<int8_t>  t_out;
  Rebind<int16_t, decltype(t_out)> t_in;
  nint_t N = std::min(size(t_in), size(t_out));

  auto data = test_utils::alloc_aligned<int16_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = -1;
  }

  auto v_in  = load(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    int8_t actual = get(t_out, v_out, i);
    EXPECT_EQ(int8_t(-1), actual) << "i=" << i;
  }
  std::free(data);
}

// Test: Demotion with out-of-range unsigned source (unsigned -> smaller int)
// Verifies ScalarConvert correctly clamps in unsigned domain without signed wraparound
TEST_F(VecConvertCornerCaseTest, DemoteUnsignedSaturated) {
  // uint32_t -> int16_t: overshoot values
  {
    ScalableTag<int16_t>  t_out;
    Rebind<uint32_t, decltype(t_out)> t_in;
    nint_t N = std::min(size(t_in), size(t_out));

    auto data = test_utils::alloc_aligned<uint32_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      nint_t vals[] = {0, 100, 40000, 4294934528u};
      data[i] = vals[i % 4];
    }

    auto v_in  = load(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      int16_t expected = vecops::convert<int16_t>(data[i]);
      int16_t actual   = get(t_out, v_out, i);
      EXPECT_EQ(expected, actual) << "i=" << i << " input=" << data[i];
    }
    std::free(data);
  }

  // uint64_t -> uint32_t: large values should clamp to UINT32_MAX
  {
    ScalableTag<uint32_t> t_out;
    Rebind<uint64_t, decltype(t_out)> t_in;
    nint_t N = std::min(size(t_in), size(t_out));

    auto data = test_utils::alloc_aligned<uint64_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      uint64_t vals[] = {0, 100, 0x100000000ULL, 0xFFFFFFFF00000000ULL};
      data[i] = vals[i % 4];
    }

    auto v_in  = load(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      uint32_t expected = vecops::convert<uint32_t>(data[i]);
      uint32_t actual   = get(t_out, v_out, i);
      EXPECT_EQ(expected, actual) << "i=" << i << " input=" << data[i];
    }
    std::free(data);
  }

  // uint64_t -> int32_t: large unsigned values clamped to INT32_MAX
  {
    ScalableTag<int32_t>  t_out;
    Rebind<uint64_t, decltype(t_out)> t_in;
    nint_t N = std::min(size(t_in), size(t_out));

    auto data = test_utils::alloc_aligned<uint64_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      uint64_t vals[] = {0, 100, 0x100000000ULL, 0xFFFFFFFF00000000ULL};
      data[i] = vals[i % 4];
    }

    auto v_in  = load(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      int32_t expected = vecops::convert<int32_t>(data[i]);
      int32_t actual   = get(t_out, v_out, i);
      EXPECT_EQ(expected, actual) << "i=" << i << " input=" << data[i];
    }
    std::free(data);
  }
}

// Test: Demotion with out-of-range signed source (int64_t -> int32_t)
// Verifies ScalarConvert uses a wide enough intermediate type to avoid truncation
TEST_F(VecConvertCornerCaseTest, DemoteSignedSaturated) {
  // int64_t -> int32_t: values outside int32_t range
  {
    ScalableTag<int32_t> t_out;
    Rebind<int64_t, decltype(t_out)> t_in;
    nint_t N = std::min(size(t_in), size(t_out));

    auto data = test_utils::alloc_aligned<int64_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      int64_t vals[] = {
          0,
          INT64_MAX,
          INT64_MIN,
          (int64_t)0x800000000LL,  // > INT32_MAX
          (int64_t)(-0x800000001LL) // < INT32_MIN
      };
      data[i] = vals[i % 5];
    }

    auto v_in  = load(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      int32_t expected = vecops::convert<int32_t>(data[i]);
      int32_t actual   = get(t_out, v_out, i);
      EXPECT_EQ(expected, actual) << "i=" << i << " input=" << data[i];
    }
    std::free(data);
  }

  // int64_t -> int16_t
  {
    ScalableTag<int16_t> t_out;
    Rebind<int64_t, decltype(t_out)> t_in;
    nint_t N = std::min(size(t_in), size(t_out));

    auto data = test_utils::alloc_aligned<int64_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      int64_t vals[] = {0, INT64_MAX, INT64_MIN, (int64_t)0x800000000LL};
      data[i] = vals[i % 4];
    }

    auto v_in  = load(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      int16_t expected = vecops::convert<int16_t>(data[i]);
      int16_t actual   = get(t_out, v_out, i);
      EXPECT_EQ(expected, actual) << "i=" << i << " input=" << data[i];
    }
    std::free(data);
  }
}

// Test: int32_t <-> float32_t for boundary values
TEST_F(VecConvertCornerCaseTest, Int32Float32Boundary) {
  ScalableTag<int32_t>   t_int;
  ScalableTag<float32_t> t_float;
  nint_t N = size(t_int);

  // int32 -> float32
  {
    auto int_data = test_utils::alloc_aligned<int32_t>(N);
    for (nint_t i = 0; i < N; ++i) {
      nint_t vals[] = {0, 1, -1, 123456};
      int_data[i] = vals[i % 4];
    }

    auto v_int   = load(t_int, int_data);
    auto v_float = convert(t_float, v_int);

    for (nint_t i = 0; i < N; ++i) {
      float32_t expected = static_cast<float32_t>(int_data[i]);
      float32_t actual = get(t_float, v_float, i);
      EXPECT_FLOAT_EQ(expected, actual) << "i=" << i;
    }
    std::free(int_data);
  }

  // float32 -> int32
  {
    auto float_data = test_utils::alloc_aligned<float32_t>(N);
    float32_t fvals[] = {0.0f, 1.0f, -1.0f, 123456.0f};
    for (nint_t i = 0; i < N; ++i) {
      float_data[i] = fvals[i % 4];
    }

    auto v_f = load(t_float, float_data);
    auto v_i = convert(t_int, v_f);

    for (nint_t i = 0; i < N; ++i) {
      int32_t expected = static_cast<int32_t>(float_data[i]);
      int32_t actual = get(t_int, v_i, i);
      EXPECT_EQ(expected, actual) << "i=" << i;
    }
    std::free(float_data);
  }
}

// Test: Large int32_t values to float32_t (precision loss expected)
TEST_F(VecConvertCornerCaseTest, LargeInt32ToFloat32) {
  ScalableTag<int32_t>   t_int;
  ScalableTag<float32_t> t_float;
  nint_t N = size(t_int);

  auto int_data = test_utils::alloc_aligned<int32_t>(N);
  int32_t vals[] = {
      0x7FFFFFFF,
      (int32_t)0x80000000,
      1234567890,
      -1234567890
  };
  for (nint_t i = 0; i < N; ++i) {
    int_data[i] = vals[i % 4];
  }

  auto v_int   = load(t_int, int_data);
  auto v_float = convert(t_float, v_int);

  for (nint_t i = 0; i < N; ++i) {
    float32_t actual   = get(t_float, v_float, i);
    float32_t expected = static_cast<float32_t>(int_data[i]);
    EXPECT_NEAR(expected, actual, std::abs(expected) * 1e-6f)
              << "i=" << i << " input=" << int_data[i];
  }
  std::free(int_data);
}

TEST_F(VecConvertCornerCaseTest, MultiWordDemote) {
  ScalableTag<int8_t>  t_out;
  Rebind<int16_t, decltype(t_out)> t_in;
  nint_t N = std::min(size(t_in), size(t_out));

  auto data = test_utils::alloc_aligned<int16_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = static_cast<int16_t>((i - 8) * 10);
  }

  auto v_in  = load(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    int8_t expected = vecops::convert<int8_t>(data[i]);
    int8_t actual   = get(t_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(data);
}

TEST_F(VecConvertCornerCaseTest, MultiWordConvert) {
  ScalableTag<int32_t>  t_in;
  ScalableTag<uint32_t> t_out;
  nint_t N = size(t_in);

  auto data = test_utils::alloc_aligned<int32_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = (i - 4) * 1000;
  }

  auto v_in  = load(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    uint32_t expected = vecops::convert<uint32_t>(data[i]);
    uint32_t actual   = get(t_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(data);
}

// Test: Float64 <-> Float32 conversion
TEST_F(VecConvertCornerCaseTest, Float64Float32RoundTrip) {
  ScalableTag<float32_t> t_f32;
  Rebind<float64_t, decltype(t_f32)> t_f64;
  nint_t N = std::min(size(t_f64), size(t_f32));

  // Float64 -> Float32 (demote)
  {
    auto f64_data = test_utils::alloc_aligned<float64_t>(N);
    float64_t vals[] = {1.5, -2.5};
    for (nint_t i = 0; i < N; ++i) {
      f64_data[i] = vals[i % 2];
    }

    auto v_f64 = load(t_f64, f64_data);
    auto v_f32 = convert(t_f32, v_f64);

    for (nint_t i = 0; i < N; ++i) {
      float32_t expected = vecops::convert<float32_t>(f64_data[i]);
      float32_t actual = get(t_f32, v_f32, i);
      EXPECT_FLOAT_EQ(expected, actual) << "i=" << i;
    }
    std::free(f64_data);
  }
}

TEST_F(VecConvertCornerCaseTest, Float32Float64RoundTrip) {
  ScalableTag<float32_t> t_f32;
  Rebind<float64_t, decltype(t_f32)> t_f64;
  nint_t N = std::min(size(t_f32), size(t_f64));

  // Float32 -> Float64 (promote)
  {
    auto f32_data = test_utils::alloc_aligned<float32_t>(N);
    float32_t vals[] = {1.5f, -2.5f, 0.0f, 100.5f};
    for (nint_t i = 0; i < N; ++i) {
      f32_data[i] = vals[i % 4];
    }

    auto v_f32 = load(t_f32, f32_data);
    auto v_f64 = convert(t_f64, v_f32);

    for (nint_t i = 0; i < N; ++i) {
      float64_t expected = vecops::convert<float64_t>(f32_data[i]);
      float64_t actual = get(t_f64, v_f64, i);
      EXPECT_DOUBLE_EQ(expected, actual) << "i=" << i;
    }
    std::free(f32_data);
  }
}

// Test: All bits set patterns
TEST_F(VecConvertCornerCaseTest, AllBitsSetPattern) {
  ScalableTag<int8_t>   t_i8;
  Rebind<uint16_t, decltype(t_i8)> t_u16;
  nint_t N = std::min(size(t_i8), size(t_u16));

  auto data = test_utils::alloc_aligned<int8_t>(N);
  memset(data, 0xFF, N * sizeof(int8_t));

  auto v_in  = load(t_i8, data);
  auto v_out = convert(t_u16, v_in);

  for (nint_t i = 0; i < N; ++i) {
    uint16_t actual = get(t_u16, v_out, i);
    EXPECT_EQ(uint16_t(0xFFFF), actual) << "i=" << i;
  }
  std::free(data);
}

// ============================================================================
// Bitcast Tests: same-size type reinterpretation
// ============================================================================

// Helper to compare bit patterns (handles NaN correctly)
template <typename T1, typename T2>
::testing::AssertionResult bits_equal(T1 expected, T2 actual) {
  static_assert(sizeof(T1) == sizeof(T2), "Size mismatch");
  if (std::memcmp(&expected, &actual, sizeof(T1)) == 0) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
      << "Bit pattern mismatch: expected " << static_cast<long long>(expected)
      << ", got " << static_cast<long long>(actual);
}

template <typename TCase>
class VecBitcastTest : public ::testing::Test {
protected:
  using TIn   = typename TCase::TIn;
  using TOut  = typename TCase::TOut;
  using InTag  = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;
  static_assert(sizeof(TIn) == sizeof(TOut), "Bitcast requires same-size types");

  InTag  t_in_;
  OutTag t_out_;

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for bitcast tests (same size)
using BitcastTypes = ::testing::Types<
    // 8-bit bitcast
    BITCAST_POWS(int8_t, uint8_t),
    BITCAST_POWS(uint8_t, int8_t),
    // 16-bit bitcast
    BITCAST_POWS(int16_t, uint16_t),
    BITCAST_POWS(uint16_t, int16_t),
    // 32-bit bitcast
    BITCAST_POWS(int32_t, uint32_t),
    BITCAST_POWS(int32_t, float32_t),
    BITCAST_POWS(uint32_t, int32_t),
    BITCAST_POWS(uint32_t, float32_t),
    BITCAST_POWS(float32_t, int32_t),
    BITCAST_POWS(float32_t, uint32_t),
    // 64-bit bitcast
    BITCAST_POWS(int64_t, uint64_t),
    BITCAST_POWS(int64_t, float64_t),
    BITCAST_POWS(uint64_t, int64_t),
    BITCAST_POWS(uint64_t, float64_t),
    BITCAST_POWS(float64_t, int64_t),
    BITCAST_POWS(float64_t, uint64_t)
>;

TYPED_TEST_SUITE(VecBitcastTest, BitcastTypes);

TYPED_TEST(VecBitcastTest, BasicBitcast) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;

  nint_t N = size(this->t_in_);

  auto v_in  = load(this->t_in_, this->in_data_);
  auto v_out = bitcast(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TIn original = this->in_data_[i];
    TOut actual = get(this->t_out_, v_out, i);
    TOut expected;
    std::memcpy(&expected, &original, sizeof(TOut));
    EXPECT_TRUE(bits_equal(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(original);
  }
}

TYPED_TEST(VecBitcastTest, BitcastRoundTrip) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;

  nint_t N = size(this->t_in_);

  auto v_in  = load(this->t_in_, this->in_data_);
  auto v_mid = bitcast(this->t_out_, v_in);
  auto v_out = bitcast(this->t_in_, v_mid);

  for (nint_t i = 0; i < N; ++i) {
    TIn expected = this->in_data_[i];
    TIn actual = get(this->t_in_, v_out, i);
    EXPECT_TRUE(bits_equal(expected, actual)) << "i=" << i;
  }
}

TYPED_TEST(VecBitcastTest, BitcastWithZeroValues) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;

  nint_t N = size(this->t_in_);

  auto v_in  = zeros(this->t_in_);
  auto v_out = bitcast(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut actual = get(this->t_out_, v_out, i);
    EXPECT_TRUE(bits_equal(TOut(0), actual)) << "i=" << i;
  }
}

// Test special bit patterns (all ones = NaN for floats)
TYPED_TEST(VecBitcastTest, BitcastWithAllOnesPattern) {
  using TIn  = typename TestFixture::TIn;
  using TOut = typename TestFixture::TOut;

  nint_t N = size(this->t_in_);

  auto data = test_utils::alloc_aligned<TIn>(N);
  std::memset(data, 0xFF, N * sizeof(TIn));

  auto v_in  = load(this->t_in_, data);
  auto v_out = bitcast(this->t_out_, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut actual = get(this->t_out_, v_out, i);
    TOut expected;
    std::memset(&expected, 0xFF, sizeof(TOut));
    EXPECT_TRUE(bits_equal(expected, actual)) << "i=" << i;
  }
  std::free(data);
}

// Corner case tests for bitcast using ScalableTag
TEST_F(VecConvertCornerCaseTest, BitcastFloat32ToInt32) {
  ScalableTag<float32_t> t_f32;
  ScalableTag<int32_t>   t_i32;
  nint_t N = size(t_f32);

  auto f32_data = test_utils::alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) f32_data[i] = 0.0f;
  f32_data[0] = 0.0f;
  f32_data[1] = -0.0f;
  if (N > 2) f32_data[2] = 1.0f;
  if (N > 3) f32_data[3] = -1.0f;

  auto v_f32 = load(t_f32, f32_data);
  auto v_i32 = bitcast(t_i32, v_f32);

  EXPECT_EQ(uint32_t(0x00000000), uint32_t(get(t_i32, v_i32, 0)));
  EXPECT_EQ(uint32_t(0x80000000), uint32_t(get(t_i32, v_i32, 1)));
  if (N > 2) EXPECT_EQ(uint32_t(0x3F800000), uint32_t(get(t_i32, v_i32, 2)));
  if (N > 3) EXPECT_EQ(uint32_t(0xBF800000), uint32_t(get(t_i32, v_i32, 3)));

  std::free(f32_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastInt32ToFloat32) {
  ScalableTag<int32_t>   t_i32;
  ScalableTag<float32_t> t_f32;
  nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  for (nint_t i = 0; i < N; ++i) i32_data[i] = 0;
  i32_data[0] = 0x00000000;
  i32_data[1] = (int32_t)0x80000000;
  if (N > 2) i32_data[2] = 0x3F800000;
  if (N > 3) i32_data[3] = (int32_t)0xBF800000;

  auto v_i32 = load(t_i32, i32_data);
  auto v_f32 = bitcast(t_f32, v_i32);

  EXPECT_FLOAT_EQ(0.0f, get(t_f32, v_f32, 0));
  EXPECT_FLOAT_EQ(-0.0f, get(t_f32, v_f32, 1));
  if (N > 2) EXPECT_FLOAT_EQ(1.0f, get(t_f32, v_f32, 2));
  if (N > 3) EXPECT_FLOAT_EQ(-1.0f, get(t_f32, v_f32, 3));

  std::free(i32_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastFloat64ToInt64) {
  ScalableTag<float64_t> t_f64;
  ScalableTag<int64_t>   t_i64;
  nint_t N = size(t_f64);

  auto f64_data = test_utils::alloc_aligned<float64_t>(N);
  for (nint_t i = 0; i < N; ++i) f64_data[i] = 0.0;
  f64_data[0] = 0.0;
  if (N > 1) f64_data[1] = 1.0;

  auto v_f64 = load(t_f64, f64_data);
  auto v_i64 = bitcast(t_i64, v_f64);

  EXPECT_EQ(int64_t(0x0000000000000000LL), get(t_i64, v_i64, 0));
  if (N > 1) EXPECT_EQ(int64_t(0x3FF0000000000000LL), get(t_i64, v_i64, 1));

  std::free(f64_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastSignedUnsigned) {
  ScalableTag<int32_t>   t_i32;
  ScalableTag<uint32_t>  t_u32;
  nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  for (nint_t i = 0; i < N; ++i) i32_data[i] = 0;
  i32_data[0] = -1;
  i32_data[1] = -2;
  if (N > 2) i32_data[2] = 0x7FFFFFFF;
  if (N > 3) i32_data[3] = (int32_t)0x80000000;

  auto v_i32 = load(t_i32, i32_data);
  auto v_u32 = bitcast(t_u32, v_i32);

  EXPECT_EQ(uint32_t(0xFFFFFFFF), get(t_u32, v_u32, 0));
  EXPECT_EQ(uint32_t(0xFFFFFFFE), get(t_u32, v_u32, 1));
  if (N > 2) EXPECT_EQ(uint32_t(0x7FFFFFFF), get(t_u32, v_u32, 2));
  if (N > 3) EXPECT_EQ(uint32_t(0x80000000), get(t_u32, v_u32, 3));

  std::free(i32_data);
}

// Test bitcast with multi-word vectors (same size)
TEST_F(VecConvertCornerCaseTest, BitcastMultiWordSameSize) {
  ScalableTag<int32_t, 1>  t_i32;
  ScalableTag<uint32_t, 1> t_u32;
  nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    i32_data[i] = (i - N/2) * 0x11111111;
  }

  auto v_i32 = load(t_i32, i32_data);
  auto v_u32 = bitcast(t_u32, v_i32);

  for (nint_t i = 0; i < N; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &i32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32, v_u32, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(i32_data);
}

// Test bitcast with larger input vector (input words > output words)
TEST_F(VecConvertCornerCaseTest, BitcastInputLarger) {
  ScalableTag<int32_t, 1>  t_i32_in;
  ScalableTag<uint32_t, 0> t_u32_out;
  nint_t N_IN  = size(t_i32_in);
  nint_t N_OUT = size(t_u32_out);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N_IN);
  for (nint_t i = 0; i < N_IN; ++i) {
    i32_data[i] = i * 12345;
  }

  auto v_in  = load(t_i32_in, i32_data);
  auto v_out = bitcast(t_u32_out, v_in);

  for (nint_t i = 0; i < N_OUT; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &i32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(i32_data);
}

// Test bitcast with larger output vector (input words < output words)
TEST_F(VecConvertCornerCaseTest, BitcastOutputLarger) {
  ScalableTag<float32_t, 0> t_f32_in;
  ScalableTag<uint32_t, 1>  t_u32_out;
  nint_t N_IN  = size(t_f32_in);
  nint_t N_OUT = size(t_u32_out);

  auto f32_data = test_utils::alloc_aligned<float32_t>(N_IN);
  for (nint_t i = 0; i < N_IN; ++i) {
    f32_data[i] = static_cast<float32_t>(i * 1.5f + 0.5f);
  }

  auto v_in  = load(t_f32_in, f32_data);
  auto v_out = bitcast(t_u32_out, v_in);

  for (nint_t i = 0; i < N_IN; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &f32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(f32_data);
}

// Test bitcast with 4-word vectors
TEST_F(VecConvertCornerCaseTest, BitcastFourWords) {
  ScalableTag<float64_t, 2> t_f64;
  ScalableTag<uint64_t, 2>  t_u64;
  nint_t N = size(t_f64);

  auto f64_data = test_utils::alloc_aligned<float64_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    f64_data[i] = static_cast<float64_t>(i * 2.5 + 1.0);
  }

  auto v_f64 = load(t_f64, f64_data);
  auto v_u64 = bitcast(t_u64, v_f64);

  for (nint_t i = 0; i < N; ++i) {
    uint64_t expected;
    std::memcpy(&expected, &f64_data[i], sizeof(uint64_t));
    uint64_t actual = get(t_u64, v_u64, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  std::free(f64_data);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
