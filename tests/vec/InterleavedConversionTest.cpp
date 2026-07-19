#include <gtest/gtest.h>

#include <cstdlib>
#include <type_traits>

#include "TestUtils.h"
#include "vecops/util/ScalarConvert.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

using float16_t = vecops::float16_t;
using bfloat16_t = vecops::bfloat16_t;

template <typename T>
bool conversion_values_equal(T expected, T actual) {
  return expected == actual || test_utils::values_near(expected, actual);
}

template <typename TIn_, typename TOut_, int Pow2_>
struct ConversionCase {
  using TIn = TIn_;
  using TOut = TOut_;
  using InTag = ScalableTag<TIn, Pow2_>;
  using OutTag = ViewAs<TOut, InTag>;
};

#define INTERLEAVED_POWS(TI, TO) \
  ConversionCase<TI, TO, 0>, ConversionCase<TI, TO, 1>, ConversionCase<TI, TO, 2>

template <typename TCase>
class InterleavedConversionTest : public ::testing::Test {
 protected:
  using TIn = typename TCase::TIn;
  using TOut = typename TCase::TOut;
  using InTag = typename TCase::InTag;
  using OutTag = typename TCase::OutTag;

  void SetUp() override {
    input = test_utils::alloc_aligned<TIn>(size(in_tag));
    fallback = test_utils::alloc_aligned<TOut>(size(out_tag));

    for (nint_t i = 0; i < size(in_tag); ++i) {
      if constexpr (is_unsigned_int<TOut>) {
        input[i] = static_cast<TIn>((i % 31) + 1);
      } else {
        input[i] = test_utils::get_test_value<TIn>(static_cast<size_t>(i + 1));
      }
    }
    for (nint_t i = 0; i < size(out_tag); ++i) {
      fallback[i] = static_cast<TOut>((i % 17) + 31);
    }
  }

  void TearDown() override {
    std::free(input);
    std::free(fallback);
  }

  InTag in_tag;
  OutTag out_tag;
  TIn* input{};
  TOut* fallback{};
};

using WidenCases = ::testing::Types<
    INTERLEAVED_POWS(int8_t, int16_t),
    INTERLEAVED_POWS(int8_t, uint16_t),
    INTERLEAVED_POWS(int8_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(int8_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(uint8_t, int16_t),
    INTERLEAVED_POWS(uint8_t, uint16_t),
    INTERLEAVED_POWS(uint8_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(uint8_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(int8_t, int32_t),
    INTERLEAVED_POWS(int8_t, uint32_t),
    INTERLEAVED_POWS(int8_t, float32_t),
    INTERLEAVED_POWS(uint8_t, int32_t),
    INTERLEAVED_POWS(uint8_t, uint32_t),
    INTERLEAVED_POWS(uint8_t, float32_t),
    INTERLEAVED_POWS(int8_t, int64_t),
    INTERLEAVED_POWS(int8_t, uint64_t),
    INTERLEAVED_POWS(int8_t, float64_t),
    INTERLEAVED_POWS(uint8_t, int64_t),
    INTERLEAVED_POWS(uint8_t, uint64_t),
    INTERLEAVED_POWS(uint8_t, float64_t),
    INTERLEAVED_POWS(int16_t, int32_t),
    INTERLEAVED_POWS(int16_t, uint32_t),
    INTERLEAVED_POWS(int16_t, float32_t),
    INTERLEAVED_POWS(uint16_t, int32_t),
    INTERLEAVED_POWS(uint16_t, uint32_t),
    INTERLEAVED_POWS(uint16_t, float32_t),
    INTERLEAVED_POWS(float16_t, int32_t),
    INTERLEAVED_POWS(float16_t, uint32_t),
    INTERLEAVED_POWS(float16_t, float32_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(bfloat16_t, int32_t),
    INTERLEAVED_POWS(bfloat16_t, uint32_t),
    INTERLEAVED_POWS(bfloat16_t, float32_t),
#endif
    INTERLEAVED_POWS(int16_t, int64_t),
    INTERLEAVED_POWS(int16_t, uint64_t),
    INTERLEAVED_POWS(int16_t, float64_t),
    INTERLEAVED_POWS(uint16_t, int64_t),
    INTERLEAVED_POWS(uint16_t, uint64_t),
    INTERLEAVED_POWS(uint16_t, float64_t),
    INTERLEAVED_POWS(float16_t, int64_t),
    INTERLEAVED_POWS(float16_t, uint64_t),
    INTERLEAVED_POWS(float16_t, float64_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(bfloat16_t, int64_t),
    INTERLEAVED_POWS(bfloat16_t, uint64_t),
    INTERLEAVED_POWS(bfloat16_t, float64_t),
#endif
    INTERLEAVED_POWS(int32_t, int64_t),
    INTERLEAVED_POWS(int32_t, uint64_t),
    INTERLEAVED_POWS(int32_t, float64_t),
    INTERLEAVED_POWS(uint32_t, int64_t),
    INTERLEAVED_POWS(uint32_t, uint64_t),
    INTERLEAVED_POWS(uint32_t, float64_t),
    INTERLEAVED_POWS(float32_t, int64_t),
    INTERLEAVED_POWS(float32_t, uint64_t),
    INTERLEAVED_POWS(float32_t, float64_t)>;

using WidenBy2Cases = ::testing::Types<
    INTERLEAVED_POWS(int8_t, int16_t),
    INTERLEAVED_POWS(int8_t, uint16_t),
    INTERLEAVED_POWS(int8_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(int8_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(uint8_t, int16_t),
    INTERLEAVED_POWS(uint8_t, uint16_t),
    INTERLEAVED_POWS(uint8_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(uint8_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(int16_t, int32_t),
    INTERLEAVED_POWS(int16_t, uint32_t),
    INTERLEAVED_POWS(int16_t, float32_t),
    INTERLEAVED_POWS(uint16_t, int32_t),
    INTERLEAVED_POWS(uint16_t, uint32_t),
    INTERLEAVED_POWS(uint16_t, float32_t),
    INTERLEAVED_POWS(float16_t, int32_t),
    INTERLEAVED_POWS(float16_t, uint32_t),
    INTERLEAVED_POWS(float16_t, float32_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(bfloat16_t, int32_t),
    INTERLEAVED_POWS(bfloat16_t, uint32_t),
    INTERLEAVED_POWS(bfloat16_t, float32_t),
#endif
    INTERLEAVED_POWS(int32_t, int64_t),
    INTERLEAVED_POWS(int32_t, uint64_t),
    INTERLEAVED_POWS(int32_t, float64_t),
    INTERLEAVED_POWS(uint32_t, int64_t),
    INTERLEAVED_POWS(uint32_t, uint64_t),
    INTERLEAVED_POWS(uint32_t, float64_t),
    INTERLEAVED_POWS(float32_t, int64_t),
    INTERLEAVED_POWS(float32_t, uint64_t),
    INTERLEAVED_POWS(float32_t, float64_t)>;

using NarrowCases = ::testing::Types<
    INTERLEAVED_POWS(int16_t, int8_t),
    INTERLEAVED_POWS(int16_t, uint8_t),
    INTERLEAVED_POWS(uint16_t, int8_t),
    INTERLEAVED_POWS(uint16_t, uint8_t),
    INTERLEAVED_POWS(float16_t, int8_t),
    INTERLEAVED_POWS(float16_t, uint8_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(bfloat16_t, int8_t),
    INTERLEAVED_POWS(bfloat16_t, uint8_t),
#endif
    INTERLEAVED_POWS(int32_t, int16_t),
    INTERLEAVED_POWS(int32_t, uint16_t),
    INTERLEAVED_POWS(int32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(int32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(uint32_t, int16_t),
    INTERLEAVED_POWS(uint32_t, uint16_t),
    INTERLEAVED_POWS(uint32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(uint32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(float32_t, int16_t),
    INTERLEAVED_POWS(float32_t, uint16_t),
    INTERLEAVED_POWS(float32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(float32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(int32_t, int8_t),
    INTERLEAVED_POWS(int32_t, uint8_t),
    INTERLEAVED_POWS(uint32_t, int8_t),
    INTERLEAVED_POWS(uint32_t, uint8_t),
    INTERLEAVED_POWS(float32_t, int8_t),
    INTERLEAVED_POWS(float32_t, uint8_t),
    INTERLEAVED_POWS(int64_t, int32_t),
    INTERLEAVED_POWS(int64_t, uint32_t),
    INTERLEAVED_POWS(int64_t, float32_t),
    INTERLEAVED_POWS(uint64_t, int32_t),
    INTERLEAVED_POWS(uint64_t, uint32_t),
    INTERLEAVED_POWS(uint64_t, float32_t),
    INTERLEAVED_POWS(float64_t, int32_t),
    INTERLEAVED_POWS(float64_t, uint32_t),
    INTERLEAVED_POWS(float64_t, float32_t),
    INTERLEAVED_POWS(int64_t, int16_t),
    INTERLEAVED_POWS(int64_t, uint16_t),
    INTERLEAVED_POWS(int64_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(int64_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(uint64_t, int16_t),
    INTERLEAVED_POWS(uint64_t, uint16_t),
    INTERLEAVED_POWS(uint64_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(uint64_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(float64_t, int16_t),
    INTERLEAVED_POWS(float64_t, uint16_t),
    INTERLEAVED_POWS(float64_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(float64_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(int64_t, int8_t),
    INTERLEAVED_POWS(int64_t, uint8_t),
    INTERLEAVED_POWS(uint64_t, int8_t),
    INTERLEAVED_POWS(uint64_t, uint8_t),
    INTERLEAVED_POWS(float64_t, int8_t),
    INTERLEAVED_POWS(float64_t, uint8_t)>;

using NarrowBy2Cases = ::testing::Types<
    INTERLEAVED_POWS(int16_t, int8_t),
    INTERLEAVED_POWS(int16_t, uint8_t),
    INTERLEAVED_POWS(uint16_t, int8_t),
    INTERLEAVED_POWS(uint16_t, uint8_t),
    INTERLEAVED_POWS(float16_t, int8_t),
    INTERLEAVED_POWS(float16_t, uint8_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(bfloat16_t, int8_t),
    INTERLEAVED_POWS(bfloat16_t, uint8_t),
#endif
    INTERLEAVED_POWS(int32_t, int16_t),
    INTERLEAVED_POWS(int32_t, uint16_t),
    INTERLEAVED_POWS(int32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(int32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(uint32_t, int16_t),
    INTERLEAVED_POWS(uint32_t, uint16_t),
    INTERLEAVED_POWS(uint32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(uint32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(float32_t, int16_t),
    INTERLEAVED_POWS(float32_t, uint16_t),
    INTERLEAVED_POWS(float32_t, float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    INTERLEAVED_POWS(float32_t, bfloat16_t),
#endif
    INTERLEAVED_POWS(int64_t, int32_t),
    INTERLEAVED_POWS(int64_t, uint32_t),
    INTERLEAVED_POWS(int64_t, float32_t),
    INTERLEAVED_POWS(uint64_t, int32_t),
    INTERLEAVED_POWS(uint64_t, uint32_t),
    INTERLEAVED_POWS(uint64_t, float32_t),
    INTERLEAVED_POWS(float64_t, int32_t),
    INTERLEAVED_POWS(float64_t, uint32_t),
    INTERLEAVED_POWS(float64_t, float32_t)>;

template <typename TCase>
class PromoteEvenTest : public InterleavedConversionTest<TCase> {};
TYPED_TEST_SUITE(PromoteEvenTest, WidenCases);

TYPED_TEST(PromoteEvenTest, SelectsStridedLanes) {
  using TOut = typename TestFixture::TOut;
  constexpr nint_t ratio = sizeof(TOut) / sizeof(typename TestFixture::TIn);
  auto result = promote_even(this->out_tag, loadu(this->in_tag, this->input));

  for (nint_t i = 0; i < size(this->out_tag); ++i) {
    auto expected = vecops::convert<TOut>(this->input[ratio * i]);
    EXPECT_TRUE(conversion_values_equal(expected, get(this->out_tag, result, i))) << "i=" << i;
  }
}

template <typename TCase>
class PromoteOddTest : public InterleavedConversionTest<TCase> {};
TYPED_TEST_SUITE(PromoteOddTest, WidenBy2Cases);

TYPED_TEST(PromoteOddTest, SelectsOddLanes) {
  using TOut = typename TestFixture::TOut;
  auto result = promote_odd(this->out_tag, loadu(this->in_tag, this->input));

  for (nint_t i = 0; i < size(this->out_tag); ++i) {
    auto expected = vecops::convert<TOut>(this->input[2 * i + 1]);
    EXPECT_TRUE(conversion_values_equal(expected, get(this->out_tag, result, i))) << "i=" << i;
  }
}

template <typename TCase>
class DemoteEvenTest : public InterleavedConversionTest<TCase> {};
TYPED_TEST_SUITE(DemoteEvenTest, NarrowCases);

TYPED_TEST(DemoteEvenTest, InsertsValuesAndPreservesFallback) {
  using TOut = typename TestFixture::TOut;
  constexpr nint_t ratio = sizeof(typename TestFixture::TIn) / sizeof(TOut);
  auto input = loadu(this->in_tag, this->input);
  auto fallback = loadu(this->out_tag, this->fallback);
  auto zero_result = demote_even(this->out_tag, input);
  auto fallback_result = demote_even(this->out_tag, input, fallback);

  for (nint_t i = 0; i < size(this->out_tag); ++i) {
    const bool converted = i % ratio == 0;
    auto expected_value = converted
        ? vecops::convert<TOut>(this->input[i / ratio])
        : TOut{};
    auto expected_fallback = converted
        ? vecops::convert<TOut>(this->input[i / ratio])
        : this->fallback[i];
    EXPECT_TRUE(conversion_values_equal(expected_value, get(this->out_tag, zero_result, i))) << "i=" << i;
    EXPECT_TRUE(conversion_values_equal(expected_fallback, get(this->out_tag, fallback_result, i))) << "i=" << i;
  }
}

template <typename TCase>
class DemoteOddTest : public InterleavedConversionTest<TCase> {};
TYPED_TEST_SUITE(DemoteOddTest, NarrowBy2Cases);

TYPED_TEST(DemoteOddTest, InsertsValuesAndPreservesFallback) {
  using TOut = typename TestFixture::TOut;
  auto input = loadu(this->in_tag, this->input);
  auto fallback = loadu(this->out_tag, this->fallback);
  auto zero_result = demote_odd(this->out_tag, input);
  auto fallback_result = demote_odd(this->out_tag, input, fallback);

  for (nint_t i = 0; i < size(this->out_tag); ++i) {
    const bool converted = i % 2 == 1;
    auto expected_value = converted
        ? vecops::convert<TOut>(this->input[i / 2])
        : TOut{};
    auto expected_fallback = converted
        ? vecops::convert<TOut>(this->input[i / 2])
        : this->fallback[i];
    EXPECT_TRUE(conversion_values_equal(expected_value, get(this->out_tag, zero_result, i))) << "i=" << i;
    EXPECT_TRUE(conversion_values_equal(expected_fallback, get(this->out_tag, fallback_result, i))) << "i=" << i;
  }
}

template <typename To, typename Vi>
concept CanPromoteEven = requires(To to, Vi vi) { promote_even(to, vi); };

template <typename To, typename Vi>
concept CanPromoteOdd = requires(To to, Vi vi) { promote_odd(to, vi); };

template <typename To, typename Vi>
concept CanDemoteEven = requires(To to, Vi vi) { demote_even(to, vi); };

template <typename To, typename Vi>
concept CanDemoteOdd = requires(To to, Vi vi) { demote_odd(to, vi); };

using I16 = ScalableTag<int16_t>;
using I16Vec = Vec<I16>;
static_assert(CanPromoteEven<ViewAs<int32_t, I16>, I16Vec>);
static_assert(CanPromoteOdd<ViewAs<int32_t, I16>, I16Vec>);
static_assert(!CanPromoteEven<ViewAs<int16_t, I16>, I16Vec>);
static_assert(!CanPromoteOdd<ViewAs<int64_t, I16>, I16Vec>);
static_assert(!CanPromoteEven<Rebind<int32_t, I16>, I16Vec>);

using I32 = ScalableTag<int32_t>;
using I32Vec = Vec<I32>;
static_assert(CanDemoteEven<ViewAs<int16_t, I32>, I32Vec>);
static_assert(CanDemoteOdd<ViewAs<int16_t, I32>, I32Vec>);
static_assert(!CanDemoteEven<ViewAs<int32_t, I32>, I32Vec>);
static_assert(!CanDemoteOdd<ViewAs<int8_t, I32>, I32Vec>);
static_assert(!CanDemoteEven<Rebind<int16_t, I32>, I32Vec>);

#undef INTERLEAVED_POWS

} // namespace
