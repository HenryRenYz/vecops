//
// BasicTest.cpp — Register-only basic operations tests
// Covers fill, zeros, get, set, mask ops, blend, shuffle,
// half-vector ops, interleave, bitcast
// Tests single-word, half-size, and multi-word vectors
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
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    return static_cast<bfloat16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return static_cast<float16_t>(static_cast<float>(idx * 1.5f + 0.5f));
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
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float16_t>) {
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
    int32_t, uint32_t, int64_t, uint64_t, float16_t
#if defined(__ARM_FEATURE_BF16)
    , bfloat16_t
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
// Shuffle Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Shuf) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  using IdxT = Index<T>;
  ScalableTag<IdxT, 0> ti;
  auto idx = fill(ti, IdxT(0));
  for (nint_t i = 0; i < N; ++i) idx = set(idx, i, IdxT(N - 1 - i));

  auto r = shuf(v, idx);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(N - 1 - i), get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, LocalShuf) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  constexpr nint_t group_el = 16 / (nint_t)sizeof(T);
  if (N < group_el || N % group_el != 0) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  using IdxT = Index<T>;
  ScalableTag<IdxT, 0> ti;
  auto idx = fill(ti, IdxT(0));
  nint_t n_groups = N / group_el;
  for (nint_t g = 0; g < n_groups; ++g)
    for (nint_t j = 0; j < group_el; ++j)
      idx = set(idx, g * group_el + j, IdxT(group_el - 1 - j));

  auto r = local_shuf(v, idx);
  for (nint_t g = 0; g < n_groups; ++g)
    for (nint_t j = 0; j < group_el; ++j)
      EXPECT_TRUE(test_utils::values_equal(
          T(g * group_el + group_el - 1 - j), get(r, g * group_el + j)));
}

// ============================================================================
// Upper / Lower / Even / Odd Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Upper) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto r = upper(t, v);
  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(half + i), get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, Lower) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto r = lower(t, v);
  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(i), get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, UpperLowerRoundTrip) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto lo = lower(t, v);
  auto hi = upper(t, v);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(i), get(lo, i)));
    EXPECT_TRUE(test_utils::values_equal(T(N / 2 + i), get(hi, i)));
  }
}

TYPED_TEST(VecBasicTest, Even) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto r = even(t, v);
  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(i * 2), get(r, i)));
  }
#endif
}

TYPED_TEST(VecBasicTest, Odd) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto r = odd(t, v);
  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(i * 2 + 1), get(r, i)));
  }
#endif
}

// ============================================================================
// Concat Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Concat) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto v = fill(t, test_utils::get_test_value<T>(0));
  for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

  auto lo = lower(t, v);
  auto hi = upper(t, v);
  auto r = concat(t, lo, hi);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(i), get(r, i)));
  }
#endif
}

// ============================================================================
// Interleave Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Interleave) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  // interleave(t, lo, hi) = [lo[0], hi[0], lo[1], hi[1], ...]
  T lo_val = test_utils::get_test_value<T>(1);
  T hi_val = test_utils::get_test_value<T>(100);

  using TagT = std::remove_reference_t<decltype(t)>;
  Half<TagT> th;
  auto v_lo = fill(th, lo_val);
  auto v_hi = fill(th, hi_val);

  auto r = interleave(t, v_lo, v_hi);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal((i % 2 == 0) ? lo_val : hi_val, get(r, i)));
  }
}

TYPED_TEST(VecBasicTest, InterleaveEven) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  // interleave_even = [a[0], b[0], a[2], b[2], a[4], b[4], ...]
  T val_a = test_utils::get_test_value<T>(10);
  T val_b = test_utils::get_test_value<T>(200);

  auto a = fill(t, val_a);
  auto b = fill(t, val_b);

  auto r = interleave_even(a, b);

  for (nint_t i = 0; i < N; i += 2) {
    EXPECT_TRUE(test_utils::values_equal(val_a, get(r, i)));
    EXPECT_TRUE(test_utils::values_equal(val_b, get(r, i + 1)));
  }
}

TYPED_TEST(VecBasicTest, InterleaveOdd) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  // interleave_odd = [a[1], b[1], a[3], b[3], a[5], b[5], ...]
  T val_a = test_utils::get_test_value<T>(30);
  T val_b = test_utils::get_test_value<T>(250);

  auto a = fill(t, val_a);
  auto b = fill(t, val_b);

  auto r = interleave_odd(a, b);

  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(val_a, get(r, i * 2)));
    EXPECT_TRUE(test_utils::values_equal(val_b, get(r, i * 2 + 1)));
  }
}

// ============================================================================
// Local Interleave Tests
// ============================================================================

TYPED_TEST(VecBasicTest, LocalInterleaveLower) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  // local_interleave_lower interleaves lower halves of each 16-byte block
  T val_a = test_utils::get_test_value<T>(15);
  T val_b = test_utils::get_test_value<T>(180);

  auto a = fill(t, val_a);
  auto b = fill(t, val_b);

  auto r = local_interleave_lower(a, b);

  for (nint_t i = 0; i < N; i += 2) {
    EXPECT_TRUE(test_utils::values_equal(val_a, get(r, i)));
    EXPECT_TRUE(test_utils::values_equal(val_b, get(r, i + 1)));
  }
}

TYPED_TEST(VecBasicTest, LocalInterleaveUpper) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  // local_interleave_upper interleaves upper halves of each 16-byte block
  T val_a = test_utils::get_test_value<T>(25);
  T val_b = test_utils::get_test_value<T>(220);

  auto a = fill(t, val_a);
  auto b = fill(t, val_b);

  auto r = local_interleave_upper(a, b);

  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(val_a, get(r, i * 2)));
    EXPECT_TRUE(test_utils::values_equal(val_b, get(r, i * 2 + 1)));
  }
}

// ============================================================================
// ConcatEven / ConcatOdd Tests
// ============================================================================

TYPED_TEST(VecBasicTest, ConcatEvenOdd) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  T val_a = test_utils::get_test_value<T>(1);
  T val_b = test_utils::get_test_value<T>(2);

  auto a = fill(t, val_a);
  auto b = fill(t, val_b);

  auto r_even = concat_even(t, a, b);
  auto r_odd = concat_odd(t, a, b);

  auto lo_even = even(t, a);
  auto hi_even = even(t, b);
  auto lo_odd = odd(t, a);
  auto hi_odd = odd(t, b);

  nint_t half = N / 2;
  for (nint_t i = 0; i < half; ++i) {
    EXPECT_TRUE(test_utils::values_equal(get(lo_even, i), get(r_even, i)));
    EXPECT_TRUE(test_utils::values_equal(get(hi_even, i), get(r_even, half + i)));
    EXPECT_TRUE(test_utils::values_equal(get(lo_odd, i), get(r_odd, i)));
    EXPECT_TRUE(test_utils::values_equal(get(hi_odd, i), get(r_odd, half + i)));
  }
#endif
}

// ============================================================================
// Bitcast Tests
// ============================================================================

TYPED_TEST(VecBasicTest, Bitcast) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  // Only test bitcast for 32-bit types (float32 <-> int32 <-> uint32)
  if constexpr (sizeof(T) == 4) {
    auto v = fill(t, test_utils::get_test_value<T>(0));
    for (nint_t i = 0; i < N; ++i) v = set(v, i, T(i));

    using TagT = std::remove_reference_t<decltype(t)>;

    if constexpr (std::is_same_v<T, float32_t>) {
      using OtherT = int32_t;
      auto tt = Rebind<OtherT, TagT>{};
      auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        int32_t expected;
        float f = static_cast<float>(i);
        std::memcpy(&expected, &f, sizeof(float));
        EXPECT_EQ(expected, get(r, i));
      }
    } else if constexpr (std::is_same_v<T, int32_t>) {
      using OtherT = float32_t;
      auto tt = Rebind<OtherT, TagT>{};
      auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        float expected;
        int32_t val = static_cast<int32_t>(i);
        std::memcpy(&expected, &val, sizeof(int32_t));
        EXPECT_TRUE(test_utils::values_equal(expected, get(r, i)));
      }
    } else {
      // T is uint32_t
      using OtherT = float32_t;
      auto tt = Rebind<OtherT, TagT>{};
      auto r = bitcast(tt, v);
      for (nint_t i = 0; i < N; ++i) {
        float expected;
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(&expected, &val, sizeof(uint32_t));
        EXPECT_TRUE(test_utils::values_equal(expected, get(r, i)));
      }
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

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
