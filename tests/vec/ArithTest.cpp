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

#include "TestUtils.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {
template <typename T>
T reduce_test_value(nint_t i) {
  if constexpr (is_signed_int<T> || is_float<T>) {
    return static_cast<T>(static_cast<float>((i % 11) - 5));
  } else {
    return static_cast<T>((i % 13) + 1);
  }
}

template <typename T>
T expected_reduce_add(const std::vector<T>& data, const std::vector<bool>* mask = nullptr) {
  T acc{};
  for (size_t i = 0; i < data.size(); ++i) {
    if (mask == nullptr || (*mask)[i]) acc = static_cast<T>(acc + data[i]);
  }
  return acc;
}

template <typename T>
T expected_reduce_max(const std::vector<T>& data, const std::vector<bool>* mask = nullptr) {
  T acc;
  if constexpr (is_float<T>) acc = static_cast<T>(-std::numeric_limits<double>::infinity());
  else acc = std::numeric_limits<T>::lowest();
  for (size_t i = 0; i < data.size(); ++i) {
    if ((mask == nullptr || (*mask)[i]) && data[i] > acc) acc = data[i];
  }
  return acc;
}

template <typename T>
T expected_reduce_min(const std::vector<T>& data, const std::vector<bool>* mask = nullptr) {
  T acc;
  if constexpr (is_float<T>) acc = static_cast<T>(std::numeric_limits<double>::infinity());
  else acc = std::numeric_limits<T>::max();
  for (size_t i = 0; i < data.size(); ++i) {
    if ((mask == nullptr || (*mask)[i]) && data[i] < acc) acc = data[i];
  }
  return acc;
}

template <typename Tag>
void run_reduce_checks(Tag t) {
  using T = TypeOf<Tag>;
  const nint_t N = size(t);
  std::vector<T> data((size_t)N);
  for (nint_t i = 0; i < N; ++i) data[(size_t)i] = reduce_test_value<T>(i);

  auto v = load(t, data.data());
  EXPECT_TRUE(test_utils::values_equal(expected_reduce_add(data), reduce_add(t, v)))
      << "reduce_add N=" << N;
  EXPECT_TRUE(test_utils::values_equal(expected_reduce_max(data), reduce_max(t, v)))
      << "reduce_max N=" << N;
  EXPECT_TRUE(test_utils::values_equal(expected_reduce_min(data), reduce_min(t, v)))
      << "reduce_min N=" << N;

  std::vector<std::vector<bool>> masks;
  masks.emplace_back((size_t)N, true);
  masks.emplace_back((size_t)N, false);
  masks.emplace_back((size_t)N, false);
  masks.emplace_back((size_t)N, false);
  for (nint_t i = 0; i < N; ++i) {
    masks[2][(size_t)i] = (i % 2) == 0;
    masks[3][(size_t)i] = (i % 5) == 1 || (i % 5) == 3;
  }

  for (size_t mi = 0; mi < masks.size(); ++mi) {
    auto m = test_utils::make_mask(t, masks[mi]);
    EXPECT_TRUE(test_utils::values_equal(expected_reduce_add(data, &masks[mi]), reduce_add(t, v, m)))
        << "masked reduce_add N=" << N << " mask=" << mi;
    EXPECT_TRUE(test_utils::values_equal(expected_reduce_max(data, &masks[mi]), reduce_max(t, v, m)))
        << "masked reduce_max N=" << N << " mask=" << mi;
    EXPECT_TRUE(test_utils::values_equal(expected_reduce_min(data, &masks[mi]), reduce_min(t, v, m)))
        << "masked reduce_min N=" << N << " mask=" << mi;
  }
}
} // namespace

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

TYPED_TEST_SUITE(VecArithTest, test_utils::AllVecDataTypes);

// ============================================================================
// add
// ============================================================================

TYPED_TEST(VecArithTest, AddBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

TYPED_TEST(VecArithTest, Reductions) {
  run_reduce_checks(this->t);
  run_reduce_checks(this->t2);
  run_reduce_checks(this->t4);
}

// ============================================================================
// sub
// ============================================================================

TYPED_TEST(VecArithTest, SubBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, small_a.get());
  auto vb = load(t, small_b.get());
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

  auto va = load(t, small_a.get());
  auto vb = load(t, small_b.get());
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
// fused multiply-add/subtract
// ============================================================================

TYPED_TEST(VecArithTest, FmaBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  auto b = std::make_unique<T[]>(N);
  auto c = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    a[i] = static_cast<T>((i % 5) + 1);
    b[i] = static_cast<T>((i % 3) + 2);
    c[i] = static_cast<T>((i % 7) + 3);
  }

  auto va = load(t, a.get());
  auto vb = load(t, b.get());
  auto vc = load(t, c.get());

  auto v_fmadd = fmadd(va, vb, vc);
  auto v_fmsub = fmsub(va, vb, vc);
  auto v_fnmadd = fnmadd(va, vb, vc);
  auto v_fnmsub = fnmsub(va, vb, vc);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fmadd(a[i], b[i], c[i]), get(t, v_fmadd, i))) << "fmadd i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fmsub(a[i], b[i], c[i]), get(t, v_fmsub, i))) << "fmsub i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fnmadd(a[i], b[i], c[i]), get(t, v_fnmadd, i))) << "fnmadd i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fnmsub(a[i], b[i], c[i]), get(t, v_fnmsub, i))) << "fnmsub i=" << i;
  }
}

TYPED_TEST(VecArithTest, FmaWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto a = std::make_unique<T[]>(N);
  auto b = std::make_unique<T[]>(N);
  auto c = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    a[i] = static_cast<T>((i % 5) + 1);
    b[i] = static_cast<T>((i % 3) + 2);
    c[i] = static_cast<T>((i % 7) + 3);
  }

  auto va = load(t, a.get());
  auto vb = load(t, b.get());
  auto vc = load(t, c.get());
  auto m = mwhilelt(t, 0, N / 2);

  auto v_fmadd = fmadd(va, vb, vc, m);
  auto v_fmsub = fmsub(va, vb, vc, m);
  auto v_fnmadd = fnmadd(va, vb, vc, m);
  auto v_fnmsub = fnmsub(va, vb, vc, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fmadd(a[i], b[i], c[i]), get(t, v_fmadd, i))) << "fmadd i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fmsub(a[i], b[i], c[i]), get(t, v_fmsub, i))) << "fmsub i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fnmadd(a[i], b[i], c[i]), get(t, v_fnmadd, i))) << "fnmadd i=" << i;
    EXPECT_TRUE(test_utils::values_equal(test_utils::scalar_fnmsub(a[i], b[i], c[i]), get(t, v_fnmsub, i))) << "fnmsub i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(a[i], get(t, v_fmadd, i))) << "fmadd inactive i=" << i;
    EXPECT_TRUE(test_utils::values_equal(a[i], get(t, v_fmsub, i))) << "fmsub inactive i=" << i;
    EXPECT_TRUE(test_utils::values_equal(a[i], get(t, v_fnmadd, i))) << "fnmadd inactive i=" << i;
    EXPECT_TRUE(test_utils::values_equal(a[i], get(t, v_fnmsub, i))) << "fnmsub inactive i=" << i;
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

    auto va = load(t, a.get());
    auto vb = load(t, b.get());
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

    auto va = load(t, a.get());
    auto vb = load(t, b.get());
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

    auto va = load(t, a.get());
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

    auto va = load(t, a.get());
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
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

  auto va = load(t, this->a_data_);
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

  auto va = load(t, this->a_data_);
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

  auto va = load(t, this->a_data_);
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

    auto va = load(t, a.get());
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

    auto va = load(t, a.get());
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

    auto va = load(t, a.get());
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

    auto va = load(t, a.get());
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
    auto vb = load(t, this->b_data_);
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

    auto va = load(t, this->a_data_);
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

    auto va = load(t, this->a_data_);
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
      auto va = load(t, this->a_data_);
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
    auto va = load(t, this->a_data_);
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
    auto va = load(t, this->a_data_);
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
      auto va = load(t, this->a_data_);
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
    auto va = load(t, this->a_data_);
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
    auto va = load(t, test_data.get());
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
    auto va = load(t, test_data.get());
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

    auto va = load(t, test_data.get());
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
      auto va = load(t, test_data.get());
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
      auto va = load(t, test_data.get());
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

  auto va = load(half_t, this->a_data_);
  auto vb = load(half_t, this->b_data_);
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

  auto va = load(half_t, this->a_data_);
  auto vb = load(half_t, this->b_data_);
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

  auto va = load(half_t, sa.get());
  auto vb = load(half_t, sb.get());
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
    auto va = load(half_t, this->a_data_);
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
    auto va = load(half_t, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < half_n; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, HalfSizeMax) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = load(half_t, this->a_data_);
  auto vb = load(half_t, this->b_data_);
  auto vr = vec::max(va, vb);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeMin) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = load(half_t, this->a_data_);
  auto vb = load(half_t, this->b_data_);
  auto vr = vec::min(va, vb);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeNeg) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = load(half_t, this->a_data_);
  auto vr = neg(va);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeAbs) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = load(half_t, this->a_data_);
  auto vr = abs(va);

  for (nint_t i = 0; i < half_n; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, HalfSizeBitAnd) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto half_t = Half<decltype(this->t)>{};
    nint_t half_n = size(half_t);

    auto va = load(half_t, this->a_data_);
    auto vb = load(half_t, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < half_n; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, HalfSizeBitOr) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto half_t = Half<decltype(this->t)>{};
    nint_t half_n = size(half_t);

    auto va = load(half_t, this->a_data_);
    auto vb = load(half_t, this->b_data_);
    auto vr = bit_or(va, vb);

    for (nint_t i = 0; i < half_n; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(half_t, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, HalfSizeCmpeq) {
  using T = typename TestFixture::Type;
  auto half_t = Half<decltype(this->t)>{};
  nint_t half_n = size(half_t);

  auto va = load(half_t, this->a_data_);
  auto vb = load(half_t, this->a_data_);
  auto m = cmpeq(va, vb);
  for (nint_t i = 0; i < half_n; ++i) EXPECT_TRUE(get(half_t, m, i));

  auto vb2 = load(half_t, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < half_n; ++i) EXPECT_FALSE(get(half_t, m2, i));
}

// ============================================================================
// Partial Register Tests (ScalableTag<T, POW2<0>)
// Tests half-word (-1), quarter-word (-2), eighth-word (-3) tags
// ============================================================================

TYPED_TEST(VecArithTest, PartialHalfWordAdd) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto va = load(th, this->a_data_);
  auto vb = load(th, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(th, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, PartialHalfWordSub) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto va = load(th, this->a_data_);
  auto vb = load(th, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(th, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, PartialHalfWordMul) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto sa = std::make_unique<T[]>(N);
  auto sb = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    sa[i] = test_utils::get_test_value<T>(i % 5);
    sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }
  auto va = load(th, sa.get());
  auto vb = load(th, sb.get());
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_mul(sa[i], sb[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(th, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, PartialHalfWordBitAnd) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    ScalableTag<T, -1> th;
    nint_t N = size(th);
    if (N < 1) return;

    auto va = load(th, this->a_data_);
    auto vb = load(th, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(th, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, PartialHalfWordCmpeq) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto va = load(th, this->a_data_);
  auto vb = load(th, this->a_data_);
  auto m = cmpeq(va, vb);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(th, m, i));

  auto vb2 = load(th, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(th, m2, i));
}

TYPED_TEST(VecArithTest, PartialQuarterWordAdd) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 1) return;

    auto va = load(tq, this->a_data_);
    auto vb = load(tq, this->b_data_);
    auto vr = add(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(tq, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, PartialQuarterWordSub) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 1) return;

    auto va = load(tq, this->a_data_);
    auto vb = load(tq, this->b_data_);
    auto vr = sub(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(tq, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, PartialEighthWordAdd) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 8) {
    ScalableTag<T, -3> te;
    nint_t N = size(te);
    if (N < 1) return;

    auto va = load(te, this->a_data_);
    auto vb = load(te, this->b_data_);
    auto vr = add(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(te, vr, i))) << "i=" << i;
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, sa.get());
  auto vb = load(t2, sb.get());
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

    auto va = load(t2, a.get());
    auto vb = load(t2, b.get());
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, this->a_data_);
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

  auto va = load(t2, this->a_data_);
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
    auto va = load(t2, this->a_data_);
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
    auto va = load(t2, this->a_data_);
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

    auto va = load(t2, this->a_data_);
    auto vb = load(t2, this->b_data_);
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

    auto va = load(t2, this->a_data_);
    auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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

  auto va = load(t2, sa.get());
  auto vb = load(t2, sb.get());
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
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
    auto va = load(t2, this->a_data_);
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

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->a_data_);
  auto m = cmpeq(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    EXPECT_TRUE(get(t2, m, i)) << "i=" << i;
  }

  auto vb2 = load(t2, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < M; ++i) {
    EXPECT_FALSE(get(t2, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpeqWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->a_data_);
  auto m_pred = mwhilelt(t2, 0, M / 2);
  auto m_result = cmpeq(va, vb, m_pred);

  for (nint_t i = 0; i < M / 2; ++i) EXPECT_TRUE(get(t2, m_result, i)) << "i=" << i;
  for (nint_t i = M / 2; i < M; ++i) EXPECT_FALSE(get(t2, m_result, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, MultiWordCmpneWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m_pred = mwhilelt(t2, 0, M / 2);
  auto m_result = cmpne(va, vb, m_pred);

  for (nint_t i = 0; i < M / 2; ++i) EXPECT_TRUE(get(t2, m_result, i));
  for (nint_t i = M / 2; i < M; ++i) EXPECT_FALSE(get(t2, m_result, i));
}

TYPED_TEST(VecArithTest, MultiWordSubWithMask) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m = mwhilelt(t2, 0, M / 2);
  auto vr = sub(va, vb, m);

  for (nint_t i = 0; i < M / 2; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i)));
  }
  for (nint_t i = M / 2; i < M; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t2, vr, i)));
  }
}

TYPED_TEST(VecArithTest, MultiWordBitXor) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto va = load(t2, this->a_data_);
    auto vb = load(t2, this->b_data_);
    auto vr = bit_xor(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_xor(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitAndnot) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto va = load(t2, this->a_data_);
    auto vb = load(t2, this->b_data_);
    auto vr = bit_andnot(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_andnot(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitNot) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto va = load(t2, this->a_data_);
    auto vr = bit_not(va);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_not(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordCmplt) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m = cmplt(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    bool expected = this->a_data_[i] < this->b_data_[i];
    EXPECT_EQ(expected, get(t2, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpgt) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m = cmpgt(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    bool expected = this->a_data_[i] > this->b_data_[i];
    EXPECT_EQ(expected, get(t2, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordCmple) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m = cmple(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    bool expected = this->a_data_[i] <= this->b_data_[i];
    EXPECT_EQ(expected, get(t2, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpge) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t M = this->multi2_size;

  auto va = load(t2, this->a_data_);
  auto vb = load(t2, this->b_data_);
  auto m = cmpge(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    bool expected = this->a_data_[i] >= this->b_data_[i];
    EXPECT_EQ(expected, get(t2, m, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordSqrt) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto a = std::make_unique<T[]>(M);
    for (nint_t i = 0; i < M; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = load(t2, a.get());
    auto vr = sqrt(va);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_sqrt(a[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t2, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordRsqrt) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t2 = this->t2;
    nint_t M = this->multi2_size;

    auto a = std::make_unique<T[]>(M);
    for (nint_t i = 0; i < M; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = load(t2, a.get());
    auto vr = rsqrt(va);

    for (nint_t i = 0; i < M; ++i) {
      float64_t expected = 1.0 / std::sqrt((float64_t)a[i]);
      float64_t actual = get(t2, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01) << "i=" << i;
    }
  }
}

// ============================================================================
// Multi-word vector operations — POW2=2 (4 registers)
// ============================================================================

TYPED_TEST(VecArithTest, MultiWordAdd4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
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

  auto va = load(t4, sa.get());
  auto vb = load(t4, sb.get());
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
    auto va = load(t4, this->a_data_);
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

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
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

TYPED_TEST(VecArithTest, MultiWordSub4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordMax4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
  auto vr = vec::max(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordMin4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
  auto vr = vec::min(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordNeg4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vr = neg(va);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordAbs4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vr = abs(va);

  for (nint_t i = 0; i < M; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, MultiWordBitAnd4) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t4 = this->t4;
    nint_t M = this->multi4_size;

    auto va = load(t4, this->a_data_);
    auto vb = load(t4, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitOr4) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t4 = this->t4;
    nint_t M = this->multi4_size;

    auto va = load(t4, this->a_data_);
    auto vb = load(t4, this->b_data_);
    auto vr = bit_or(va, vb);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i))) << "i=" << i;
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordBitShr4) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t4 = this->t4;
    nint_t M = this->multi4_size;

    int shift = 2;
    auto va = load(t4, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < M; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t4, vr, i)));
    }
  }
}

TYPED_TEST(VecArithTest, MultiWordCmpeq4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->a_data_);
  auto m = cmpeq(va, vb);
  for (nint_t i = 0; i < M; ++i) EXPECT_TRUE(get(t4, m, i));

  auto vb2 = load(t4, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < M; ++i) EXPECT_FALSE(get(t4, m2, i));
}

TYPED_TEST(VecArithTest, MultiWordCmplt4) {
  using T = typename TestFixture::Type;
  auto& t4 = this->t4;
  nint_t M = this->multi4_size;

  auto va = load(t4, this->a_data_);
  auto vb = load(t4, this->b_data_);
  auto m = cmplt(va, vb);

  for (nint_t i = 0; i < M; ++i) {
    bool expected = this->a_data_[i] < this->b_data_[i];
    EXPECT_EQ(expected, get(t4, m, i)) << "i=" << i;
  }
}

// ============================================================================
// Comparison operations → Mask
// ============================================================================

TYPED_TEST(VecArithTest, CmpeqBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->a_data_);
  auto m = cmpeq(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i=" << i;
  }

  auto vb2 = load(t, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(VecArithTest, CmpeqWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->a_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpeq(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(t, m_result, i)) << "i=" << i;
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, CmpneBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
  auto m = cmpne(va, vb);

  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, m, i)) << "i=" << i;

  auto vsame = load(t, this->a_data_);
  auto m2 = cmpne(va, vsame);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
}

TYPED_TEST(VecArithTest, CmpneWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpne(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(t, m_result, i));
  for (nint_t i = N / 2; i < N; ++i) EXPECT_FALSE(get(t, m_result, i));
}

TYPED_TEST(VecArithTest, CmpltBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

  auto va = load(t, this->a_data_);
  auto vb = load(t, this->b_data_);
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

using FloatTypes = ::testing::Types<float32_t, float64_t, vecops::float16_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
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

  auto va = load(t, a.get());
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = isinf(va, m_pred);

  EXPECT_TRUE(get(t, m_result, 0));
  EXPECT_FALSE(get(t, m_result, N - 1));
}

// ============================================================================
// Multi-word float classification
// ============================================================================

TYPED_TEST(VecFloatClassifyTest, MultiWordIsNan) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  a[N - 1] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

  auto va = load(t2, a.get());
  auto m = isnan(va);

  EXPECT_TRUE(get(t2, m, 0));
  EXPECT_TRUE(get(t2, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) EXPECT_FALSE(get(t2, m, i));
}

TYPED_TEST(VecFloatClassifyTest, MultiWordIsInf) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);

  auto a = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[1] = static_cast<T>(-INFINITY);

  auto va = load(t2, a.get());
  auto m = isinf(va);

  EXPECT_TRUE(get(t2, m, 0));
  EXPECT_TRUE(get(t2, m, 1));
  for (nint_t i = 2; i < N; ++i) EXPECT_FALSE(get(t2, m, i));
}

// ============================================================================
// Arithmetic corner cases
// ============================================================================

TYPED_TEST(VecArithTest, DivByZero) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    auto b = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 2.0);
      b[i] = static_cast<T>(0.0);
    }

    auto va = load(t, a.get());
    auto vb = load(t, b.get());
    auto vr = div(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      float64_t actual = get(t, vr, i);
      EXPECT_TRUE(std::isinf(actual)) << "i=" << i << " expected +inf, got " << actual;
      if (a[i] > T{0}) EXPECT_GT(actual, 0) << "i=" << i << " positive dividend should give +inf";
    }
  }
}

TYPED_TEST(VecArithTest, SqrtNegative) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(-1.0);

    auto va = load(t, a.get());
    auto vr = sqrt(va);

    for (nint_t i = 0; i < N; ++i) {
      float64_t actual = get(t, vr, i);
      EXPECT_TRUE(std::isnan(actual)) << "i=" << i << " sqrt(-1) should be NaN";
    }
  }
}

TYPED_TEST(VecArithTest, RcpZero) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(0.0);

    auto va = load(t, a.get());
    auto vr = rcp(va);

    for (nint_t i = 0; i < N; ++i) {
      float64_t actual = get(t, vr, i);
      EXPECT_TRUE(std::isinf(actual)) << "i=" << i << " rcp(0) should be +inf";
    }
  }
}

TYPED_TEST(VecArithTest, AddWithNaN) {
  using T = typename TestFixture::Type;
  if constexpr (vecops::is_float<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto a = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      a[i] = (i == 0) ? static_cast<T>(std::numeric_limits<double>::quiet_NaN())
                      : static_cast<T>(i + 1.0);
    }

    auto va = load(t, a.get());
    auto vb = load(t, this->b_data_);
    auto vr = add(va, vb);

    float64_t actual0 = get(t, vr, 0);
    EXPECT_TRUE(std::isnan(actual0)) << "NaN + x should be NaN";
    if (N > 1) {
      T expected = test_utils::scalar_add(a[1], this->b_data_[1]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, 1)));
    }
  }
}

TYPED_TEST(VecArithTest, BitShlOverflow) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) test_data[i] = static_cast<T>(~T{0});

    int shift = sizeof(T) * 8; // shift by full bit width
    auto va = load(t, test_data.get());
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shl(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(T{0}, get(t, vr, i)))
          << "i=" << i << " shl by full bit-width should zero";
    }
  }
}

TYPED_TEST(VecArithTest, BitShrFullWidthSignExt) {
  using T = typename TestFixture::Type;
  if constexpr (std::is_integral_v<T>) {
    auto& t = this->t;
    nint_t N = this->full_size;

    auto test_data = std::make_unique<T[]>(N);
    for (nint_t i = 0; i < N; ++i) {
      test_data[i] = (i % 2 == 0) ? static_cast<T>(-1) : static_cast<T>(~T{0} >> 1);
    }

    int shift = sizeof(T) * 8 - 1;
    auto va = load(t, test_data.get());
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " shr full-width-1";
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
