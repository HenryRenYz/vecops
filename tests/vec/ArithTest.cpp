//
// VecArithTest.cpp
// Comprehensive test for basic arithmetic operations
// Covers: add, sub, mul, div, rcp, max, min,
//         bit_and, bit_or, bit_xor, bit_andnot, bit_not,
//         bit_shl, bit_shr,
//         neg, abs, sqrt, rsqrt,
//         cmpeq, cmpne, cmplt, cmpgt, cmple, cmpge,
//         isnan, isposinf, isneginf, isinf
//
// Supports: x86 SSE/AVX/AVX-512, ARM SVE/NEON, and scalar fallback.
//

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

// ============================================================================
// Helper utilities
// ============================================================================

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
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return static_cast<uint64_t>(idx * 100000ULL + 50000ULL);
  }
}

template <typename T>
constexpr T get_test_value_b(int idx) {
  return get_test_value<T>(idx + 50);
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

template <typename T>
T* alloc_aligned(size_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

// Arithmetic helpers: compute expected result element-wise
template <typename T> T scalar_add(T a, T b) { return a + b; }
template <typename T> T scalar_sub(T a, T b) { return a - b; }
template <typename T> T scalar_mul(T a, T b) { return a * b; }

template <typename T>
T scalar_div(T a, T b) {
  if constexpr (vecops::is_float<T>) return a / b;
  else return static_cast<T>(0);
}

template <typename T>
T scalar_max(T a, T b) {
  if constexpr (vecops::is_float<T>) return std::max(a, b);
  else return (a > b) ? a : b;
}

template <typename T>
T scalar_min(T a, T b) {
  if constexpr (vecops::is_float<T>) return std::min(a, b);
  else return (a < b) ? a : b;
}

template <typename T> T scalar_neg(T a) { return -a; }

template <typename T>
T scalar_abs(T a) {
  if constexpr (vecops::is_float<T>) return std::fabs(a);
  else return (a < T{}) ? -a : a;
}

template <typename T>
T scalar_bit_and(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) & static_cast<U>(b));
}

template <typename T>
T scalar_bit_or(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) | static_cast<U>(b));
}

template <typename T>
T scalar_bit_xor(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) ^ static_cast<U>(b));
}

template <typename T>
T scalar_bit_andnot(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>((~static_cast<U>(a)) & static_cast<U>(b));
}

template <typename T>
T scalar_bit_not(T a) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(~static_cast<U>(a));
}

template <typename T>
T scalar_bit_shl(T a, int count) {
  using U = std::make_unsigned_t<T>;
  if (count >= static_cast<int>(sizeof(T) * 8)) return T{0};
  if (count < 0) return a;
  return static_cast<T>(static_cast<U>(a) << count);
}

template <typename T>
T scalar_bit_shr(T a, int count) {
  if (count >= static_cast<int>(sizeof(T) * 8)) {
    if constexpr (std::is_signed_v<T>) return (a < 0) ? static_cast<T>(-1) : T{0};
    else return T{0};
  }
  if (count < 0) return a;
  return a >> count;
}

template <typename T>
T scalar_sqrt(T a) {
  if constexpr (vecops::is_float<T>) return std::sqrt((float64_t)a);
  else return static_cast<T>(0);
}

} // namespace test_utils

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class VecArithTest : public ::testing::Test {
protected:
  using Type = T;

  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  ScalableTag<T, 2> t4;

  nint_t full_size;
  nint_t multi2_size;
  nint_t multi4_size;

  void SetUp() override {
    full_size   = size(t);
    multi2_size = size(t2);
    multi4_size = size(t4);

    a_data_ = test_utils::alloc_aligned<T>(256);
    b_data_ = test_utils::alloc_aligned<T>(256);
    for (size_t i = 0; i < 256; ++i) {
      a_data_[i] = test_utils::get_test_value<T>(i);
      b_data_[i] = test_utils::get_test_value_b<T>(i);
    }
  }

  void TearDown() override {
    std::free(a_data_);
    std::free(b_data_);
  }

  T* a_data_{};
  T* b_data_{};
};

using AllTypes = ::testing::Types<
    float32_t, float64_t,
    int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t,
    vecops::float16_t
#if defined(__ARM_FEATURE_BF16)
    , vecops::bfloat16_t
#endif
>;

TYPED_TEST_SUITE(VecArithTest, AllTypes);

// ============================================================================
// add
// ============================================================================

TYPED_TEST(VecArithTest, AddBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, AddWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = add(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)))
        << "i=" << i << " (masked out, should be a)";
  }
}

// ============================================================================
// sub
// ============================================================================

TYPED_TEST(VecArithTest, SubBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, SubWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = sub(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
  }
}

// ============================================================================
// mul
// ============================================================================

TYPED_TEST(VecArithTest, MulBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto small_a = std::make_unique<T[]>(N);
  auto small_b = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    small_a[i] = test_utils::get_test_value<T>(i % 5);
    small_b[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t, small_a.get());
  auto vb = loadu(t, small_b.get());
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_mul(small_a[i], small_b[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MulWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto small_a = std::make_unique<T[]>(N);
  auto small_b = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    small_a[i] = test_utils::get_test_value<T>(i % 5);
    small_b[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t, small_a.get());
  auto vb = loadu(t, small_b.get());
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = mul(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_mul(small_a[i], small_b[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(small_a[i], get(t, vr, i)));
  }
}

// ============================================================================
// div (float only)
// ============================================================================

TYPED_TEST(VecArithTest, DivBasic) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    auto b = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 10.0 + 1.0);
      b[i] = static_cast<T>((i + 1) * 3.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto vb = loadu(t, b.get());
    auto vr = div(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_div(a[i], b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, DivWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    auto b = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 10.0 + 1.0);
      b[i] = static_cast<T>((i + 1) * 3.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto vb = loadu(t, b.get());
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = div(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_div(a[i], b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(a[i], get(t, vr, i)));
    }
  }
}

// ============================================================================
// rcp (float only, approximate)
// ============================================================================

TYPED_TEST(VecArithTest, RcpBasic) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 2.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto vr = rcp(va);

    for (nint_t i = 0; i < N; ++i) {
      float64_t expected = T{1} / a[i];
      float64_t actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01)
          << "i=" << i << " expected=" << expected << " got=" << actual;
    }
  }
}

TYPED_TEST(VecArithTest, RcpWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 2.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = rcp(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      float64_t expected = T{1} / a[i];
      float64_t actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01);
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

// ============================================================================
// max / min
// ============================================================================

TYPED_TEST(VecArithTest, MaxBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = vec::max(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MaxWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = vec::max(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
  }
}

TYPED_TEST(VecArithTest, MinBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = vec::min(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MinWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = vec::min(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
  }
}

// ============================================================================
// neg / abs
// ============================================================================

TYPED_TEST(VecArithTest, NegBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vr = neg(va);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, NegWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto default_v = fill(t, T(999));
  auto vr = neg(va, m, default_v);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
  }
}

TYPED_TEST(VecArithTest, AbsBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vr = abs(va);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, AbsWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto default_v = fill(t, T(999));
  auto vr = abs(va, m, default_v);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
  }
}

// ============================================================================
// sqrt / rsqrt (float only)
// ============================================================================

TYPED_TEST(VecArithTest, SqrtBasic) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto vr = sqrt(va);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_sqrt(a[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, SqrtWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = sqrt(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_sqrt(a[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, RsqrtBasic) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto vr = rsqrt(va);

    for (nint_t i = 0; i < N; ++i) {
      float64_t expected = 1.0 / std::sqrt((float64_t)a[i]);
      float64_t actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01)
          << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, RsqrtWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a.get());
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = rsqrt(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      float64_t expected = 1.0 / std::sqrt((float64_t)a[i]);
      float64_t actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01);
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

// ============================================================================
// Bitwise operations (integral types only)
// ============================================================================

TYPED_TEST(VecArithTest, BitAndBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, BitAndWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_and(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitOrBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_or(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, BitOrWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_or(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitXorBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_xor(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_xor(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, BitXorWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_xor(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_xor(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitAndnotBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_andnot(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_andnot(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, BitAndnotWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_andnot(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_andnot(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitNotBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto vr = bit_not(va);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_not(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, BitNotWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T{0x42});
    auto vr = bit_not(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_not(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T{0x42}, get(t, vr, i)));
    }
  }
}

// ============================================================================
// Bitwise shift operations (integral types only)
// ============================================================================

TYPED_TEST(VecArithTest, BitShlBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int shift_counts[] = {0, 1, 2, 3, 4, 7, 8};
    for (int shift : shift_counts) {
      auto va = loadu(t, this->a_data_);
      auto vr = bit_shl(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "i=" << i << " shift=" << shift
            << " a=" << static_cast<long long>(this->a_data_[i])
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(VecArithTest, BitShlWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int shift = 2;
    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_shl(va, shift, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitShlLargeShift) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int large_shift = sizeof(T) * 8;
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shl(va, large_shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], large_shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " shift=" << large_shift;
    }
  }
}

TYPED_TEST(VecArithTest, BitShrBasic) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int shift_counts[] = {0, 1, 2, 3, 4, 7, 8};
    for (int shift : shift_counts) {
      auto va = loadu(t, this->a_data_);
      auto vr = bit_shr(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "i=" << i << " shift=" << shift
            << " a=" << static_cast<long long>(this->a_data_[i])
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(VecArithTest, BitShrWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int shift = 2;
    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_shr(va, shift, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, BitShrArithmeticSignExtension) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      if (i % 2 == 0) test_data[i] = static_cast<T>(-1 - i);
      else            test_data[i] = static_cast<T>(1 + i);
    }

    int shift = 1;
    auto va = loadu(t, test_data.get());
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " input=" << static_cast<long long>(test_data[i])
          << " expected=" << static_cast<long long>(expected)
          << " actual=" << static_cast<long long>(get(t, vr, i));
      if (test_data[i] < 0) {
        EXPECT_LT(get(t, vr, i), T{0})
            << "Arithmetic right shift should preserve sign for negative values";
      }
    }
  }
}

TYPED_TEST(VecArithTest, BitShrLogicalZeroFill) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      test_data[i] = static_cast<T>(~T{0} - i);
    }

    int shift = 1;
    auto va = loadu(t, test_data.get());
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " input=" << static_cast<long long>(test_data[i])
          << " expected=" << static_cast<long long>(expected)
          << " actual=" << static_cast<long long>(get(t, vr, i));
    }
  }
}

TYPED_TEST(VecArithTest, BitShrLargeShift) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    int large_shift = sizeof(T) * 8;
    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      if constexpr (std::is_signed_v<T>)
        test_data[i] = (i % 2 == 0) ? static_cast<T>(-1 - i) : static_cast<T>(1 + i);
      else
        test_data[i] = static_cast<T>(i + 1);
    }

    auto va = loadu(t, test_data.get());
    auto vr = bit_shr(va, large_shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], large_shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " shift=" << large_shift;
    }
  }
}

TYPED_TEST(VecArithTest, BitShlPattern) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) test_data[i] = static_cast<T>(1);

    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8); ++shift) {
      auto va = loadu(t, test_data.get());
      auto vr = bit_shl(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shl(T{1}, shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "shift=" << shift
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(VecArithTest, BitShrPattern) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i)
      test_data[i] = static_cast<T>(T{1} << (sizeof(T) * 8 - 1));

    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8); ++shift) {
      auto va = loadu(t, test_data.get());
      auto vr = bit_shr(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shr(
            static_cast<T>(T{1} << (sizeof(T) * 8 - 1)), shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "shift=" << shift
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

// ============================================================================
// Half-size vector operations (using Half<T> tag pattern)
// ============================================================================

TYPED_TEST(VecArithTest, HalfSizeAdd) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = loadu(half_t, this->a_data_);
  auto vb = loadu(half_t, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeSub) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = loadu(half_t, this->a_data_);
  auto vb = loadu(half_t, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeMul) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto sa = std::make_unique<T[]>(half_n);
  auto sb = std::make_unique<T[]>(half_n);
  for (nint_t i = 0; i < half_n; ++i) {
    sa[i] = test_utils::get_test_value<T>(i % 5);
    sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(half_t, sa.get());
  auto vb = loadu(half_t, sb.get());
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_mul(sa[i], sb[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeBitShl) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto half_t = Half<decltype(this->t)>{};
    nint_t half_n = size(half_t);

    int shift = 3;
    auto va = loadu(half_t, this->a_data_);
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < half_n; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, HalfSizeBitShr) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto half_t = Half<decltype(this->t)>{};
    nint_t half_n = size(half_t);

    int shift = 3;
    auto va = loadu(half_t, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < half_n; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i)));
    }
  }
}

// ============================================================================
// Multi-word vector operations — POW2=1 (2 registers)
// ============================================================================

TYPED_TEST(VecArithTest, MultiWordAdd) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordSub) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordMul) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto sa = std::make_unique<T[]>(M);
  auto sb = std::make_unique<T[]>(M);
  for (nint_t i = 0; i < M; ++i) {
    sa[i] = test_utils::get_test_value<T>(i % 5);
    sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t2, sa.get());
  auto vb = loadu(t2, sb.get());
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_mul(sa[i], sb[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordDiv) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto a = std::make_unique<T[]>(M);
    auto b = std::make_unique<T[]>(M);
    for (nint_t i = 0; i < M; ++i) {
      a[i] = static_cast<T>((i + 1) * 10.0 + 1.0);
      b[i] = static_cast<T>((i + 1) * 3.0 + 1.0);
    }

    auto va = loadu(t2, a.get());
    auto vb = loadu(t2, b.get());
    auto vr = div(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_div(a[i], b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordMax) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto vr = vec::max(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordMin) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto vr = vec::min(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordNeg) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vr = neg(va);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordAbs) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vr = abs(va);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordBitShl) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    int shift = 3;
    auto va = loadu(t2, this->a_data_);
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitShr) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    int shift = 3;
    auto va = loadu(t2, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitAnd) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto va = loadu(t2, this->a_data_);
    auto vb = loadu(t2, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitOr) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto va = loadu(t2, this->a_data_);
    auto vb = loadu(t2, this->b_data_);
    auto vr = bit_or(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

// --- Multi-word masked operations (POW2=1) ---

TYPED_TEST(VecArithTest, MultiWordAddWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto m = mwhilelt(t2, 0, M / 2);
  auto vr = add(va, vb, m);

  for (nint_t i = 0; i < M / 2; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
  }
  for (nint_t i = M / 2; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t2, vr, i)))
        << "i=" << i << " (masked out, should be a)";
  }
}

TYPED_TEST(VecArithTest, MultiWordMulWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto sa = std::make_unique<T[]>(M);
  auto sb = std::make_unique<T[]>(M);
  for (nint_t i = 0; i < M; ++i) {
    sa[i] = test_utils::get_test_value<T>(i % 5);
    sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t2, sa.get());
  auto vb = loadu(t2, sb.get());
  auto m = mwhilelt(t2, 0, M / 2);
  auto vr = mul(va, vb, m);

  for (nint_t i = 0; i < M / 2; ++i) {
    T expected = test_utils::scalar_mul(sa[i], sb[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
  }
  for (nint_t i = M / 2; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sa[i], get(t2, vr, i)));
  }
}

TYPED_TEST(VecArithTest, MultiWordMaxWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto m = mwhilelt(t2, 0, M / 2);
  auto vr = vec::max(va, vb, m);

  for (nint_t i = 0; i < M / 2; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
  }
  for (nint_t i = M / 2; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t2, vr, i)));
  }
}

TYPED_TEST(VecArithTest, MultiWordBitShlWithMask) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    int shift = 3;
    auto va = loadu(t2, this->a_data_);
    auto m = mwhilelt(t2, 0, M / 2);
    auto vr = bit_shl(va, shift, m);

    for (nint_t i = 0; i < M / 2; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
    }
    for (nint_t i = M / 2; i < M; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t2, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpeq) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->a_data_);
  auto m = cmpeq(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    EXPECT_TRUE(get(t2, m, i)) << "i=" << i;
  }

  auto vb2 = loadu(t2, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < M; ++i) {
    EXPECT_FALSE(get(t2, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpeqWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->a_data_);
  auto m_pred = mwhilelt(t2, 0, M / 2);
  auto m_result = cmpeq(va, vb, m_pred);

  for (nint_t i = 0; i < M / 2; ++i) EXPECT_TRUE(get(t2, m_result, i)) << "i=" << i;
  for (nint_t i = M / 2; i < M; ++i) EXPECT_FALSE(get(t2, m_result, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, MultiWordCmpneWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = loadu(t2, this->a_data_);
  auto vb = loadu(t2, this->b_data_);
  auto m_pred = mwhilelt(t2, 0, M / 2);
  auto m_result = cmpne(va, vb, m_pred);

  for (nint_t i = 0; i < M / 2; ++i) EXPECT_TRUE(get(t2, m_result, i));
  for (nint_t i = M / 2; i < M; ++i) EXPECT_FALSE(get(t2, m_result, i));
}

// ============================================================================
// Multi-word vector operations — POW2=2 (4 registers)
// ============================================================================

TYPED_TEST(VecArithTest, MultiWordAdd4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = loadu(t4, this->a_data_);
  auto vb = loadu(t4, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordMul4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto sa = std::make_unique<T[]>(M);
  auto sb = std::make_unique<T[]>(M);
  for (nint_t i = 0; i < M; ++i) {
    sa[i] = test_utils::get_test_value<T>(i % 5);
    sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t4, sa.get());
  auto vb = loadu(t4, sb.get());
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_mul(sa[i], sb[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordBitShl4) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t4 = this->t4;
    nint_t M = this->multi4_size;

    int shift = 2;
    auto va = loadu(t4, this->a_data_);
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordFill4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t4, fill_val);

  for (nint_t i = 0; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t4, v, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordAddWithMask4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = loadu(t4, this->a_data_);
  auto vb = loadu(t4, this->b_data_);
  auto m = mwhilelt(t4, 0, M / 2);
  auto vr = add(va, vb, m);

  for (nint_t i = 0; i < M / 2; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
  for (nint_t i = M / 2; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t4, vr, i)))
        << "i=" << i << " (masked out, should be a)";
  }
}

// ============================================================================
// Comparison operations → Mask
// ============================================================================

TYPED_TEST(VecArithTest, CmpeqBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->a_data_);
  auto m = cmpeq(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i=" << i;
  }

  auto vb2 = loadu(t, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, CmpeqWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->a_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpeq(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(t, m_result, i)) << "i=" << i;
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, CmpneBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmpne(va, vb);

  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, m, i)) << "i=" << i;

  auto vsame = loadu(t, this->a_data_);
  auto m2 = cmpne(va, vsame);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, CmpneWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpne(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(t, m_result, i));
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i));
}

TYPED_TEST(VecArithTest, CmpltBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmplt(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] < this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i << " a=" << static_cast<long long>(this->a_data_[i])
        << " b=" << static_cast<long long>(this->b_data_[i]);
  }
}

TYPED_TEST(VecArithTest, CmpltWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmplt(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    bool expected = this->a_data_[i] < this->b_data_[i];
    EXPECT_EQ(expected, get(t, m_result, i)) << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, CmpgtBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmpgt(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] > this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i << " a=" << static_cast<long long>(this->a_data_[i])
        << " b=" << static_cast<long long>(this->b_data_[i]);
  }
}

TYPED_TEST(VecArithTest, CmpgtWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpgt(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    bool expected = this->a_data_[i] > this->b_data_[i];
    EXPECT_EQ(expected, get(t, m_result, i));
  }
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i));
}

TYPED_TEST(VecArithTest, CmpleBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmple(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] <= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, CmpleWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmple(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    bool expected = this->a_data_[i] <= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m_result, i));
  }
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i));
}

TYPED_TEST(VecArithTest, CmpgeBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmpge(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] >= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, CmpgeWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpge(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    bool expected = this->a_data_[i] >= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m_result, i));
  }
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i));
}

// ============================================================================
// Float-specific classification: isnan, isposinf, isneginf, isinf
// ============================================================================

using FloatTypes = ::testing::Types<float32_t, float64_t>;

template <typename T>
class VecFloatClassifyTest : public ::testing::Test {
protected:
  using Type = T;
  ScalableTag<T, 0> t;
  nint_t full_size;

  void SetUp() override { full_size = size(t); }
};

TYPED_TEST_SUITE(VecFloatClassifyTest, FloatTypes);

TYPED_TEST(VecFloatClassifyTest, IsNanBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  if constexpr (std::is_same_v<T, float32_t>) {
    a[0] = std::numeric_limits<float>::quiet_NaN();
    a[N - 1] = std::numeric_limits<float>::quiet_NaN();
  } else {
    a[0] = std::numeric_limits<double>::quiet_NaN();
    a[N - 1] = std::numeric_limits<double>::quiet_NaN();
  }

  auto va = loadu(t, a.get());
  auto m = isnan(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) EXPECT_FALSE(get(t, m, i)) << "i=" << i;
}

TYPED_TEST(VecFloatClassifyTest, IsNanWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  a[N / 2] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

  auto va = loadu(t, a.get());
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = isnan(va, m_pred);

  EXPECT_TRUE(get(t, m_result, 0));
  EXPECT_FALSE(get(t, m_result, N / 2));
  for (nint_t i = 1; i < N / 2; ++i) EXPECT_FALSE(get(t, m_result, i));
}

TYPED_TEST(VecFloatClassifyTest, IsPosInfBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[N - 1] = static_cast<T>(-INFINITY);

  auto va = loadu(t, a.get());
  auto m = isposinf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_FALSE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) EXPECT_FALSE(get(t, m, i)) << "i=" << i;
}

TYPED_TEST(VecFloatClassifyTest, IsPosInfWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[N - 1] = static_cast<T>(INFINITY);

  auto va = loadu(t, a.get());
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = isposinf(va, m_pred);

  EXPECT_TRUE(get(t, m_result, 0));
  EXPECT_FALSE(get(t, m_result, N - 1));
}

TYPED_TEST(VecFloatClassifyTest, IsNegInfBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(-INFINITY);
  a[N - 1] = static_cast<T>(INFINITY);

  auto va = loadu(t, a.get());
  auto m = isneginf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_FALSE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) EXPECT_FALSE(get(t, m, i)) << "i=" << i;
}

TYPED_TEST(VecFloatClassifyTest, IsNegInfWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(-INFINITY);
  a[N - 1] = static_cast<T>(-INFINITY);

  auto va = loadu(t, a.get());
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = isneginf(va, m_pred);

  EXPECT_TRUE(get(t, m_result, 0));
  EXPECT_FALSE(get(t, m_result, N - 1));
}

TYPED_TEST(VecFloatClassifyTest, IsInfBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[1] = static_cast<T>(-INFINITY);
  if (N > 2) a[2] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

  auto va = loadu(t, a.get());
  auto m = isinf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, 1));
  if (N > 2) EXPECT_FALSE(get(t, m, 2));
  for (nint_t i = 3; i < N; ++i) EXPECT_FALSE(get(t, m, i)) << "i=" << i;
}

TYPED_TEST(VecFloatClassifyTest, IsInfWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[N - 1] = static_cast<T>(-INFINITY);

  auto va = loadu(t, a.get());
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = isinf(va, m_pred);

  EXPECT_TRUE(get(t, m_result, 0));
  EXPECT_FALSE(get(t, m_result, N - 1));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
