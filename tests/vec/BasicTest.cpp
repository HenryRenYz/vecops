//
// BasicTest.cpp — Register-only basic operations tests
// Covers fill, zeros, get, set, mask ops, blend, bitcast
// Tests single-word, half-size, and multi-word vectors
// Note: shuffle/interleave/upper/lower/even/odd/concat tests moved to ShuffleTest.cpp
//

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace test_utils {

template <typename T>
constexpr T get_test_value(int idx) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    return static_cast<vecops::bfloat16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    return static_cast<vecops::float16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float32_t>) {
    return static_cast<float32_t>(idx * 1.5f + 0.5f);
  } else if constexpr (std::is_same_v<T, float64_t>) {
    return static_cast<float64_t>(idx * 1.5 + 0.5);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return static_cast<int8_t>((idx * 7 + 3) % 127 - 64);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return static_cast<uint8_t>((idx * 7 + 3) % 256);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return static_cast<int16_t>((idx * 100 + 50) % 32767 - 16384);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return static_cast<uint16_t>((idx * 100 + 50) % 65536);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return static_cast<int32_t>(idx * 1000 + 500);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return static_cast<uint32_t>(idx * 1000 + 500);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<int64_t>(idx * 100000LL + 50000LL);
  } else {
    return static_cast<uint64_t>(idx * 100000ULL + 50000ULL);
  }
}

template <typename T>
::testing::AssertionResult values_equal(T expected, T actual) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::abs(expected - actual) < 1e-5f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::abs(expected - actual) < 1e-10) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else {
    if (expected == actual) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
        << "Expected " << static_cast<long long>(expected)
        << ", got " << static_cast<long long>(actual);
  }
}

} // namespace test_utils

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class VecBasicTest : public ::testing::Test {
protected:
  using Type = T;

  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  ScalableTag<T, 2> t4;

  nint_t full_size;
  nint_t multi2_size;
  nint_t multi4_size;

  void SetUp() override {
    full_size = size(t);
    multi2_size = size(t2);
    multi4_size = size(t4);
  }
};

using TestedTypes = ::testing::Types<
    float32_t, float64_t, int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t, vecops::float16_t
#if defined(__ARM_FEATURE_BF16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

TYPED_TEST_SUITE(VecBasicTest, TestedTypes);

// ============================================================================
// Size Verification Tests
// ============================================================================

TYPED_TEST(VecBasicTest, VerifySize) {
  auto& t = this->t;
  nint_t N = this->full_size;
  EXPECT_EQ(size(t), N);
  EXPECT_EQ(word_size(t), N);
  EXPECT_TRUE(is_word_vec(t));
}

TYPED_TEST(VecBasicTest, MultiWordSize) {
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  EXPECT_EQ(size(t), N);
#ifndef CPU_CAPABILITY_GENERIC
  EXPECT_EQ(word_size(t), this->full_size);
  EXPECT_EQ(num_words(t), 2);
  EXPECT_FALSE(is_word_vec(t));
#endif
}

// ============================================================================
// fill / zeros Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Fill) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, Zeros) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = zeros(t);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  nint_t n = N / 2;

  auto m = mwhilelt(t, 0, n);
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithMaskAll) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);

  auto m = mtrue(t);
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithMaskNone) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);

  auto m = mfalse(t);
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  nint_t n = N / 2;

  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithNZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);

  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, 0, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, FillWithNFull) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);

  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, N, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

// ============================================================================
// mfill / mtrue / mfalse Tests
// ============================================================================

TYPED_TEST(VecBasicTest, MfillTrue) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mfill(t, true);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TYPED_TEST(VecBasicTest, MfillFalse) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mfill(t, false);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecBasicTest, MtrueMfalse) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m_true = mtrue(t);
  auto m_false = mfalse(t);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m_true, i));
    EXPECT_FALSE(get(t, m_false, i));
  }
}

// ============================================================================
// mwhilelt / mwhilege / mwhilele / mwhilegt Tests
// ============================================================================

TYPED_TEST(VecBasicTest, MwhileltBasic) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhileltAll) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilelt(t, 0, N);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhileltNone) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilelt(t, 0, 0);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhileltVariousRanges) {
  auto& t = this->t;
  nint_t N = this->full_size;

  for (nint_t end = 0; end <= N; ++end) {
    auto m = mwhilelt(t, 0, end);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_EQ(get(t, m, i), i < end)
          << "end = " << end << ", i = " << i;
    }
  }
}

TYPED_TEST(VecBasicTest, MwhilegeBasic) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilege(t, 0, n);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhilegeAll) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilege(t, 0, 0);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhilegeNone) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilege(t, 0, N);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MwhilegeVariousRanges) {
  auto& t = this->t;
  nint_t N = this->full_size;

  for (nint_t start = 0; start <= N; ++start) {
    auto m = mwhilege(t, 0, start);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_EQ(get(t, m, i), i >= start)
          << "start = " << start << ", i = " << i;
    }
  }
}

TYPED_TEST(VecBasicTest, MwhileleMwhilegt) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t mid = N / 2;

  auto m_le = mwhilele(t, 0, mid);
  for (nint_t i = 0; i <= mid; ++i) {
    EXPECT_TRUE(get(t, m_le, i)) << "mwhilele: i = " << i;
  }
  for (nint_t i = mid + 1; i < N; ++i) {
    EXPECT_FALSE(get(t, m_le, i)) << "mwhilele: i = " << i;
  }

  auto m_gt = mwhilegt(t, 0, mid);
  for (nint_t i = 0; i <= mid; ++i) {
    EXPECT_FALSE(get(t, m_gt, i)) << "mwhilegt: i = " << i;
  }
  for (nint_t i = mid + 1; i < N; ++i) {
    EXPECT_TRUE(get(t, m_gt, i)) << "mwhilegt: i = " << i;
  }
}

// ============================================================================
// get / set element Tests
// ============================================================================

TYPED_TEST(VecBasicTest, GetElement) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) {
    v = set(v, i, test_utils::get_test_value<T>(i));
  }

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::get_test_value<T>(i);
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, SetElement) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = zeros(t);
  for (nint_t i = 0; i < N; ++i) {
    T val = test_utils::get_test_value<T>(i * 10 + 5);
    v = set(v, i, val);
    EXPECT_TRUE(test_utils::values_equal(val, get(v, i)));
    for (nint_t j = 0; j <= i; ++j) {
      EXPECT_TRUE(test_utils::values_equal(
          test_utils::get_test_value<T>(j * 10 + 5), get(v, j)));
    }
  }
}

TYPED_TEST(VecBasicTest, SetElementIndividual) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = zeros(t);
  for (nint_t i = 0; i < N; ++i) {
    T val = test_utils::get_test_value<T>(i * 10 + 5);
    v = set(v, i, val);
  }

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::get_test_value<T>(i * 10 + 5);
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, GetMaskElement) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecBasicTest, SetMaskElement) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mfalse(t);
  for (nint_t i = 0; i < N; i += 2) {
    m = set(t, m, i, true);
  }

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(get(t, m, i), (i % 2 == 0)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, SetMaskElementToggle) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mtrue(t);
  for (nint_t i = 0; i < N; i += 2) {
    m = set(t, m, i, false);
  }

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(get(t, m, i), (i % 2 != 0)) << "i = " << i;
  }
}

// ============================================================================
// Tag-based get/set Tests
// ============================================================================

TYPED_TEST(VecBasicTest, TagGetSetElement) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto v = zeros(t);
  T x0 = test_utils::get_test_value<T>(10);
  T x1 = test_utils::get_test_value<T>(20);

  v = set(t, v, 0, x0);
  v = set(t, v, N - 1, x1);

  EXPECT_TRUE(test_utils::values_equal(x0, get(t, v, 0)));
  EXPECT_TRUE(test_utils::values_equal(x1, get(t, v, N - 1)));
}

// ============================================================================
// Mask Bit Ops Tests
// ============================================================================

TYPED_TEST(VecBasicTest, MaskBitOps) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto a = mwhilelt(t, 0, N / 2 + 1);
  auto b = mwhilelt(t, N / 4, 3 * N / 4);
  auto c_and = word::bit_and(a, b);
  auto c_or = word::bit_or(a, b);
  auto c_xor = word::bit_xor(a, b);
  auto c_nand = word::bit_andnot(a, b);
  auto c_not = word::bit_not(a);

  for (nint_t i = 0; i < N; ++i) {
    bool va = get(t, a, i), vb = get(t, b, i);
    EXPECT_EQ(get(t, c_and, i), va && vb);
    EXPECT_EQ(get(t, c_or, i), va || vb);
    EXPECT_EQ(get(t, c_xor, i), va != vb);
    EXPECT_EQ(get(t, c_nand, i), !va && vb);
    EXPECT_EQ(get(t, c_not, i), !va);
  }
#endif
}

// ============================================================================
// Blend Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Blend) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T va = test_utils::get_test_value<T>(5);
  T vb = test_utils::get_test_value<T>(95);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mwhilelt(t, 0, N / 2);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(get(t, m, i) ? vb : va, get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, BlendAll) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T va = test_utils::get_test_value<T>(5);
  T vb = test_utils::get_test_value<T>(95);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mtrue(t);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(vb, get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, BlendNone) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T va = test_utils::get_test_value<T>(5);
  T vb = test_utils::get_test_value<T>(95);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mfalse(t);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(va, get(r, i)));
  }
}

// ============================================================================
// Bitcast Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Bitcast) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using TagT = std::remove_reference_t<decltype(t)>;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, test_utils::get_test_value<T>(i));

  if constexpr (sizeof(T) == 1) {
    // int8_t <-> uint8_t
    using OT = std::conditional_t<std::is_same_v<T, int8_t>, uint8_t, int8_t>;
    auto tt = Rebind<OT, TagT>{};
    auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i);
      OT exp; std::memcpy(&exp, &orig, 1);
      EXPECT_EQ(exp, get(r, i));
    }
  } else if constexpr (sizeof(T) == 2) {
    // int16_t/uint16_t/float16_t/bfloat16_t <-> int16_t
    using OT = std::conditional_t<std::is_same_v<T, int16_t>, uint16_t, int16_t>;
    auto tt = Rebind<OT, TagT>{};
    auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i);
      OT exp; std::memcpy(&exp, &orig, 2);
      EXPECT_EQ(exp, get(r, i));
    }
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_same_v<T, float32_t>) {
      using OT = int32_t;
      auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 4); EXPECT_EQ(exp, get(r, i));
      }
    } else {
      using OT = float32_t;
      auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 4); EXPECT_TRUE(test_utils::values_equal(exp, get(r, i)));
      }
    }
  } else if constexpr (sizeof(T) == 8) {
    if constexpr (std::is_same_v<T, float64_t>) {
      using OT = int64_t;
      auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 8); EXPECT_EQ(exp, get(r, i));
      }
    } else {
      using OT = float64_t;
      auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 8); EXPECT_TRUE(test_utils::values_equal(exp, get(r, i)));
      }
    }
  }
}

TYPED_TEST(VecBasicTest, BitcastMultiWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  using TagT = std::remove_reference_t<decltype(t)>;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, test_utils::get_test_value<T>(i));

  // bitcast multi-word: pick opposite signed/unsigned or int/float type
  if constexpr (sizeof(T) == 1) {
    using OT = std::conditional_t<std::is_same_v<T, int8_t>, uint8_t, int8_t>;
    auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 1); EXPECT_EQ(exp, get(r, i));
    }
  } else if constexpr (sizeof(T) == 2) {
    using OT = std::conditional_t<std::is_same_v<T, int16_t>, uint16_t, int16_t>;
    auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 2); EXPECT_EQ(exp, get(r, i));
    }
  } else if constexpr (sizeof(T) == 4) {
    using OT = std::conditional_t<std::is_same_v<T, float32_t>, int32_t, float32_t>;
    auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 4);
      if constexpr (std::is_integral_v<OT>) EXPECT_EQ(exp, get(r, i));
      else EXPECT_TRUE(test_utils::values_equal(exp, get(r, i)));
    }
  } else if constexpr (sizeof(T) == 8) {
    using OT = std::conditional_t<std::is_same_v<T, float64_t>, int64_t, float64_t>;
    auto tt = Rebind<OT, TagT>{}; auto r = bitcast(tt, v);
    for (nint_t i = 0; i < N; ++i) {
      T orig = test_utils::get_test_value<T>(i); OT exp; std::memcpy(&exp, &orig, 8);
      if constexpr (std::is_integral_v<OT>) EXPECT_EQ(exp, get(r, i));
      else EXPECT_TRUE(test_utils::values_equal(exp, get(r, i)));
    }
  }
}

// ============================================================================
// Half-size Tests (partial register usage on SVE)
// ============================================================================

TYPED_TEST(VecBasicTest, HalfSizeFill) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  nint_t n = N / 2;

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, HalfSizeZeros) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  nint_t n = N / 2;

  auto v = zeros(t);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, HalfSizeGetSet) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;
  nint_t n = N / 4;

  auto v = zeros(t);
  T val = test_utils::get_test_value<T>(123);
  v = set(v, n - 1, val);
  EXPECT_TRUE(test_utils::values_equal(val, get(v, n - 1)));
}

TYPED_TEST(VecBasicTest, HalfSizeMwhilelt) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  nint_t n = N / 4;
  auto m = mwhilelt(t, 0, n);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecBasicTest, HalfSizeMaskBitOps) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;
  nint_t n = N / 4;
  if (n < 2) return;

  auto a = mwhilelt(t, 0, n);
  auto b = mwhilelt(t, n / 2, n);
  auto c_and = word::bit_and(a, b);

  for (nint_t i = 0; i < N; ++i) {
    bool va = get(t, a, i), vb = get(t, b, i);
    EXPECT_EQ(get(t, c_and, i), va && vb);
  }
#endif
}

TYPED_TEST(VecBasicTest, HalfSizeBlend) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;
  nint_t n = N / 4;

  T va = test_utils::get_test_value<T>(5);
  T vb = test_utils::get_test_value<T>(95);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mwhilelt(t, 0, n);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(vb, get(r, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(va, get(r, i)));
  }
}

// ============================================================================
// Partial Register Tests (ScalableTag<T, POW2<0>)
// Tests half-word (-1), quarter-word (-2), eighth-word (-3) tags
// Critical for SVE and partial-register scenarios on x86
// ============================================================================

TYPED_TEST(VecBasicTest, PartialHalfWordFill) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(th, fill_val);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, PartialHalfWordZeros) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto v = zeros(th);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, PartialHalfWordGetSet) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto v = zeros(th);
  T val = test_utils::get_test_value<T>(123);
  v = set(v, N - 1, val);
  EXPECT_TRUE(test_utils::values_equal(val, get(v, N - 1)));
}

TYPED_TEST(VecBasicTest, PartialHalfWordMwhilelt) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilelt(th, 0, n);
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(get(th, m, i));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_FALSE(get(th, m, i));
  }
}

TYPED_TEST(VecBasicTest, PartialHalfWordBlend) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 2) return;

  T va = test_utils::get_test_value<T>(5);
  T vb = test_utils::get_test_value<T>(95);
  auto v0 = fill(th, va);
  auto v1 = fill(th, vb);
  auto m = mwhilelt(th, 0, N / 2);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(vb, get(r, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(va, get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, PartialQuarterWordFill) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 1) return;

    T fill_val = test_utils::get_test_value<T>(77);
    auto v = fill(tq, fill_val);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
    }
  }
}

TYPED_TEST(VecBasicTest, PartialQuarterWordZeros) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 1) return;

    auto v = zeros(tq);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
    }
  }
}

TYPED_TEST(VecBasicTest, PartialQuarterWordGetSet) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 2) return;

    auto v = zeros(tq);
    T val = test_utils::get_test_value<T>(99);
    v = set(v, N - 1, val);
    EXPECT_TRUE(test_utils::values_equal(val, get(v, N - 1)));
  }
}

TYPED_TEST(VecBasicTest, PartialEighthWordFill) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 8) {
    ScalableTag<T, -3> te;
    nint_t N = size(te);
    if (N < 1) return;

    T fill_val = test_utils::get_test_value<T>(88);
    auto v = fill(te, fill_val);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
    }
  }
}

TYPED_TEST(VecBasicTest, PartialEighthWordZeros) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 8) {
    ScalableTag<T, -3> te;
    nint_t N = size(te);
    if (N < 1) return;

    auto v = zeros(te);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
    }
  }
}

// ============================================================================
// Multi-word Tests (2-word vectors)
// ============================================================================

TYPED_TEST(VecBasicTest, MultiWordFill) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;

  T fill_val = test_utils::get_test_value<T>(77);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, MultiWordZeros) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;

  auto v = zeros(t);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, MultiWordGetSet) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  nint_t ws = this->full_size;
  if (ws < 2 || N < ws * 2) return;

  auto v = zeros(t);
  T val1 = test_utils::get_test_value<T>(111);
  T val2 = test_utils::get_test_value<T>(222);

  v = set(v, ws - 1, val1);
  v = set(v, ws, val2);

  EXPECT_TRUE(test_utils::values_equal(val1, get(v, ws - 1)));
  EXPECT_TRUE(test_utils::values_equal(val2, get(v, ws)));
}

TYPED_TEST(VecBasicTest, MultiWordMwhilelt) {
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  nint_t n = N - N / 4;
  auto m = mwhilelt(t, 0, n);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecBasicTest, MultiWordBlend) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  T va = test_utils::get_test_value<T>(10);
  T vb = test_utils::get_test_value<T>(90);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mwhilelt(t, 0, N / 2);
  auto r = blend(v0, m, v1);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(vb, get(r, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(va, get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, MultiWordFillWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  nint_t n = N - N / 4;

  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecBasicTest, MultiWordSetElement4) {
  using T = typename TestFixture::Type;
  auto& t = this->t4;
  nint_t N = this->multi4_size;
  nint_t ws = this->full_size;
  if (ws < 2 || N < ws * 2) return;

  auto v = zeros(t);
  T val1 = test_utils::get_test_value<T>(111);
  T val2 = test_utils::get_test_value<T>(222);

  v = set(v, ws - 1, val1);
  v = set(v, ws, val2);

  EXPECT_TRUE(test_utils::values_equal(val1, get(v, ws - 1)));
  EXPECT_TRUE(test_utils::values_equal(val2, get(v, ws)));
}

// ============================================================================
// Edge Cases
// ============================================================================

TYPED_TEST(VecBasicTest, VariousMaskPatterns) {
  auto& t = this->t;
  nint_t N = this->full_size;

  for (nint_t count = 0; count <= N; ++count) {
    auto m = mwhilelt(t, 0, count);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_EQ(get(t, m, i), i < count)
          << "count=" << count << ", i=" << i;
    }
  }
}

TYPED_TEST(VecBasicTest, ExtremeValues) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  if constexpr (std::is_integral_v<T>) {
    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();

    auto v_min = fill(t, min_val);
    auto v_max = fill(t, max_val);

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(min_val, get(v_min, i)));
      EXPECT_TRUE(test_utils::values_equal(max_val, get(v_max, i)));
    }
  }
}

TYPED_TEST(VecBasicTest, FloatExtremeValues) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  if constexpr (std::is_same_v<T, float32_t>) {
    auto v_nan  = fill(t, std::numeric_limits<float32_t>::quiet_NaN());
    auto v_inf  = fill(t, std::numeric_limits<float32_t>::infinity());
    auto v_ninf = fill(t, -std::numeric_limits<float32_t>::infinity());
    auto v_max  = fill(t, std::numeric_limits<float32_t>::max());
    auto v_min  = fill(t, std::numeric_limits<float32_t>::min());
    auto v_sub  = fill(t, std::numeric_limits<float32_t>::denorm_min());
    auto v_zero = fill(t, 0.0f);
    auto v_nzero= fill(t, -0.0f);

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(std::isnan(get(v_nan, i)));
      EXPECT_TRUE(std::isinf(get(v_inf, i)));
      EXPECT_TRUE(std::isinf(get(v_ninf, i)));
      EXPECT_TRUE(test_utils::values_equal(
          std::numeric_limits<float32_t>::max(), get(v_max, i)));
      EXPECT_TRUE(test_utils::values_equal(
          std::numeric_limits<float32_t>::denorm_min(), get(v_sub, i)));
      EXPECT_EQ(0.0f, get(v_zero, i));
      EXPECT_EQ(-0.0f, get(v_nzero, i));
    }
  } else if constexpr (std::is_same_v<T, float64_t>) {
    auto v_nan  = fill(t, std::numeric_limits<float64_t>::quiet_NaN());
    auto v_inf  = fill(t, std::numeric_limits<float64_t>::infinity());
    auto v_ninf = fill(t, -std::numeric_limits<float64_t>::infinity());
    auto v_max  = fill(t, std::numeric_limits<float64_t>::max());
    auto v_sub  = fill(t, std::numeric_limits<float64_t>::denorm_min());
    auto v_zero = fill(t, 0.0);
    auto v_nzero= fill(t, -0.0);

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(std::isnan(get(v_nan, i)));
      EXPECT_TRUE(std::isinf(get(v_inf, i)));
      EXPECT_TRUE(std::isinf(get(v_ninf, i)));
      EXPECT_TRUE(test_utils::values_equal(
          std::numeric_limits<float64_t>::max(), get(v_max, i)));
      EXPECT_TRUE(test_utils::values_equal(
          std::numeric_limits<float64_t>::denorm_min(), get(v_sub, i)));
      EXPECT_EQ(0.0, get(v_zero, i));
      EXPECT_EQ(-0.0, get(v_nzero, i));
    }
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    auto v_zero = fill(t, vecops::float16_t{});
    auto v_pos  = fill(t, static_cast<vecops::float16_t>(
        static_cast<float>(1.5f)));
    auto v_neg  = fill(t, static_cast<vecops::float16_t>(
        static_cast<float>(-3.25f)));

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(vecops::float16_t{}, get(v_zero, i)));
      EXPECT_TRUE(test_utils::values_equal(
          static_cast<vecops::float16_t>(static_cast<float>(1.5f)), get(v_pos, i)));
      EXPECT_TRUE(test_utils::values_equal(
          static_cast<vecops::float16_t>(static_cast<float>(-3.25f)), get(v_neg, i)));
    }
  } else if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    auto v_zero = fill(t, vecops::bfloat16_t{});
    auto v_pos  = fill(t, static_cast<vecops::bfloat16_t>(
        static_cast<float>(2.5f)));
    auto v_neg  = fill(t, static_cast<vecops::bfloat16_t>(
        static_cast<float>(-7.5f)));

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(vecops::bfloat16_t{}, get(v_zero, i)));
      EXPECT_TRUE(test_utils::values_equal(
          static_cast<vecops::bfloat16_t>(static_cast<float>(2.5f)), get(v_pos, i)));
      EXPECT_TRUE(test_utils::values_equal(
          static_cast<vecops::bfloat16_t>(static_cast<float>(-7.5f)), get(v_neg, i)));
    }
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
