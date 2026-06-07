//
// LoadStoreTest.cpp — load/store/gather/scatter operation tests
// Supports both x86 (FixedTag) and SVE (ScalableTag)
//

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <limits>
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
    if (std::abs(e - a) < 0.01f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::abs(expected - actual) < 1e-5f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::abs(expected - actual) < 1e-10) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "Expected " << expected << ", got " << actual;
  } else {
    if (expected == actual) {
      return ::testing::AssertionSuccess();
    }
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

} // namespace test_utils

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class VecLoadStoreTest : public ::testing::Test {
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

    aligned_data_ = test_utils::alloc_aligned<T>(256);
    aligned_out_ = test_utils::alloc_aligned<T>(256);

    for (size_t i = 0; i < 256; ++i) {
      aligned_data_[i] = test_utils::get_test_value<T>(i);
      aligned_out_[i] = T{};
    }
  }

  void TearDown() override {
    std::free(aligned_data_);
    std::free(aligned_out_);
  }

  T* aligned_data_{};
  T* aligned_out_{};
};

using TestedTypes = ::testing::Types<
    float32_t, float64_t, int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t, vecops::float16_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

TYPED_TEST_SUITE(VecLoadStoreTest, TestedTypes);

// ============================================================================
// Size Verification Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, VerifySize) {
  auto& t = this->t;
  nint_t N = this->full_size;
  EXPECT_EQ(size(t), N);
  EXPECT_EQ(word_size(t), N);
  EXPECT_EQ(num_words(t), 1);
  EXPECT_TRUE(is_word_vec(t));
}

#ifndef CPU_CAPABILITY_SVE
TYPED_TEST(VecLoadStoreTest, VerifyHalfSize) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = ScalableTag<T, 0>::N;
  constexpr nint_t HALF_SIZE = FULL_SIZE / 2;

  if constexpr (HALF_SIZE >= 1) {
    FixedTag<T, HALF_SIZE> t;
    EXPECT_EQ(size(t), HALF_SIZE);
    EXPECT_EQ(word_size(t), HALF_SIZE);
  }
}
#endif

TYPED_TEST(VecLoadStoreTest, MultiWordSize) {
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

TYPED_TEST(VecLoadStoreTest, FillBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T zero_val = T{};
  auto v = fill(t, zero_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(zero_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, ZerosBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = zeros(t);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithMask) {
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

TYPED_TEST(VecLoadStoreTest, FillWithMaskAll) {
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

TYPED_TEST(VecLoadStoreTest, FillWithMaskNone) {
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

TYPED_TEST(VecLoadStoreTest, FillWithN) {
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

TYPED_TEST(VecLoadStoreTest, FillWithNZero) {
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

TYPED_TEST(VecLoadStoreTest, FillWithNFull) {
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

TYPED_TEST(VecLoadStoreTest, MfillTrue) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mfill(t, true);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TYPED_TEST(VecLoadStoreTest, MfillFalse) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mfill(t, false);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecLoadStoreTest, MtrueMfalse) {
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
// mwhilelt / mwhilege Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, MwhileltBasic) {
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

TYPED_TEST(VecLoadStoreTest, MwhileltAll) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilelt(t, 0, N);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileltNone) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilelt(t, 0, 0);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileltVariousRanges) {
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

TYPED_TEST(VecLoadStoreTest, MwhilegeBasic) {
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

TYPED_TEST(VecLoadStoreTest, MwhilegeAll) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilege(t, 0, 0);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeNone) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mwhilege(t, 0, N);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeVariousRanges) {
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

TYPED_TEST(VecLoadStoreTest, MwhileleMwhilegt) {
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
// loadu / storeu Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoaduBasic) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = loadu(t, this->aligned_data_);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuBasic) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = loadu(t, this->aligned_data_);
  storeu(t, this->aligned_out_, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduStoreuRoundTrip) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(123);
  auto v = fill(t, fill_val);
  storeu(t, this->aligned_out_, v);

  auto v2 = loadu(t, this->aligned_out_);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v2, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  nint_t n = N / 2;

  auto v = loadu(t, this->aligned_data_, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithNZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);

  auto v = loadu(t, this->aligned_data_, 0, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithNFull) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);

  auto v = loadu(t, this->aligned_data_, N, default_v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = test_utils::get_test_value<T>(-1);
  }

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  nint_t n = N / 2;

  storeu(t, this->aligned_out_, n, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithNZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  storeu(t, this->aligned_out_, 0, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

// ============================================================================
// load / store (aligned) Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoadAligned) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = load(t, this->aligned_data_);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreAligned) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = load(t, this->aligned_data_);
  store(t, this->aligned_out_, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, LoadWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  nint_t n = N / 2;

  auto v = load(t, this->aligned_data_, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = test_utils::get_test_value<T>(-1);
  }

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  nint_t n = N / 2;

  store(t, this->aligned_out_, n, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
}

// ============================================================================
// Masked load/store Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoaduWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);

  auto v = loadu(t, this->aligned_data_, m, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoadWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);

  auto v = load(t, this->aligned_data_, m, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  storeu(t, this->aligned_out_, m, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  nint_t n = N / 2;
  auto m = mwhilelt(t, 0, n);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  store(t, this->aligned_out_, m, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMaskAll) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto m = mtrue(t);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  storeu(t, this->aligned_out_, m, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMaskNone) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto m = mfalse(t);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);

  storeu(t, this->aligned_out_, m, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

// ============================================================================
// get / set element Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, GetElement) {
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = loadu(t, this->aligned_data_);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, SetElement) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto v = zeros(t);
  T new_val = test_utils::get_test_value<T>(123);

  for (nint_t i = 0; i < N; ++i) {
    v = set(v, i, new_val);
    EXPECT_TRUE(test_utils::values_equal(new_val, get(v, i)));
    for (nint_t j = 0; j <= i; ++j) {
      EXPECT_TRUE(test_utils::values_equal(new_val, get(v, j)));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, SetElementIndividual) {
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

TYPED_TEST(VecLoadStoreTest, GetMaskElement) {
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

TYPED_TEST(VecLoadStoreTest, SetMaskElement) {
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

TYPED_TEST(VecLoadStoreTest, SetMaskElementToggle) {
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
// Half-size vector Tests (SVE: uses partial elements of full register)
// ============================================================================

TYPED_TEST(VecLoadStoreTest, HalfSizeFill) {
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

TYPED_TEST(VecLoadStoreTest, HalfSizeLoadStore) {
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  nint_t n = N / 2;

  auto v = loadu(t, this->aligned_data_);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }

  storeu(t, this->aligned_out_, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, HalfSizeMwhilelt) {
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

TYPED_TEST(VecLoadStoreTest, HalfSizeLoadWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  nint_t n = N / 4;

  auto v = loadu(t, this->aligned_data_, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

// ============================================================================
// Partial Register Tests (ScalableTag<T, POW2<0>)
// Tests half-word (-1), quarter-word (-2), eighth-word (-3) tags
// ============================================================================

TYPED_TEST(VecLoadStoreTest, PartialHalfWordLoadStore) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 1) return;

  auto v = loadu(th, this->aligned_data_);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], get(v, i)));
  }

  for (nint_t i = 0; i < N; ++i) {
    this->aligned_out_[i] = T{};
  }
  storeu(th, this->aligned_out_, v);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, PartialHalfWordLoadWithN) {
  using T = typename TestFixture::Type;
  ScalableTag<T, -1> th;
  nint_t N = size(th);
  if (N < 2) return;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(th, default_val);
  nint_t n = N / 2;

  auto v = loadu(th, this->aligned_data_, n, default_v);
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, PartialHalfWordFill) {
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

TYPED_TEST(VecLoadStoreTest, PartialQuarterWordLoadStore) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 4) {
    ScalableTag<T, -2> tq;
    nint_t N = size(tq);
    if (N < 1) return;

    auto v = loadu(tq, this->aligned_data_);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], get(v, i)));
    }

    storeu(tq, this->aligned_out_, v);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[i]));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, PartialQuarterWordFill) {
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

TYPED_TEST(VecLoadStoreTest, PartialEighthWordLoadStore) {
  using T = typename TestFixture::Type;
  if constexpr (VEC_WIDTH < 0 || VEC_WIDTH / (8 * sizeof(T)) >= 8) {
    ScalableTag<T, -3> te;
    nint_t N = size(te);
    if (N < 1) return;

    auto v = loadu(te, this->aligned_data_);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], get(v, i)));
    }

    storeu(te, this->aligned_out_, v);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[i]));
    }
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TYPED_TEST(VecLoadStoreTest, VariousMaskPatterns) {
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

TYPED_TEST(VecLoadStoreTest, LoadStoreSequential) {
  auto& t = this->t;
  nint_t N = this->full_size;

  for (int offset = 0; offset < 64; offset += N) {
    auto v = loadu(t, this->aligned_data_ + offset);
    storeu(t, this->aligned_out_ + offset, v);
  }

  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, ExtremeValues) {
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

TYPED_TEST(VecLoadStoreTest, FloatExtremeValues) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  if constexpr (std::is_same_v<T, float32_t>) {
    auto v_nan  = fill(t, std::numeric_limits<float32_t>::quiet_NaN());
    auto v_inf  = fill(t, std::numeric_limits<float32_t>::infinity());
    auto v_ninf = fill(t, -std::numeric_limits<float32_t>::infinity());
    auto v_max  = fill(t, std::numeric_limits<float32_t>::max());
    auto v_sub  = fill(t, std::numeric_limits<float32_t>::denorm_min());
    auto v_zero = fill(t, 0.0f);
    auto v_nzero= fill(t, -0.0f);

    storeu(t, this->aligned_out_, v_nan);  EXPECT_TRUE(std::isnan(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_inf);  EXPECT_TRUE(std::isinf(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_ninf); EXPECT_TRUE(std::isinf(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_max);  EXPECT_TRUE(test_utils::values_equal(std::numeric_limits<float32_t>::max(), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_sub);  EXPECT_TRUE(test_utils::values_equal(std::numeric_limits<float32_t>::denorm_min(), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_zero); EXPECT_EQ(0.0f, this->aligned_out_[0]);
    storeu(t, this->aligned_out_, v_nzero);EXPECT_EQ(-0.0f, this->aligned_out_[0]);
  } else if constexpr (std::is_same_v<T, float64_t>) {
    auto v_nan  = fill(t, std::numeric_limits<float64_t>::quiet_NaN());
    auto v_inf  = fill(t, std::numeric_limits<float64_t>::infinity());
    auto v_ninf = fill(t, -std::numeric_limits<float64_t>::infinity());
    auto v_max  = fill(t, std::numeric_limits<float64_t>::max());
    auto v_sub  = fill(t, std::numeric_limits<float64_t>::denorm_min());
    auto v_zero = fill(t, 0.0);
    auto v_nzero= fill(t, -0.0);

    storeu(t, this->aligned_out_, v_nan);  EXPECT_TRUE(std::isnan(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_inf);  EXPECT_TRUE(std::isinf(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_ninf); EXPECT_TRUE(std::isinf(this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_max);  EXPECT_TRUE(test_utils::values_equal(std::numeric_limits<float64_t>::max(), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_sub);  EXPECT_TRUE(test_utils::values_equal(std::numeric_limits<float64_t>::denorm_min(), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_zero); EXPECT_EQ(0.0, this->aligned_out_[0]);
    storeu(t, this->aligned_out_, v_nzero);EXPECT_EQ(-0.0, this->aligned_out_[0]);
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    auto v_zero = fill(t, vecops::float16_t{});
    auto v_pos  = fill(t, static_cast<vecops::float16_t>(static_cast<float>(1.5f)));
    auto v_neg  = fill(t, static_cast<vecops::float16_t>(static_cast<float>(-3.25f)));

    storeu(t, this->aligned_out_, v_zero);
    EXPECT_TRUE(test_utils::values_equal(vecops::float16_t{}, this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_pos);
    EXPECT_TRUE(test_utils::values_equal(static_cast<vecops::float16_t>(static_cast<float>(1.5f)), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_neg);
    EXPECT_TRUE(test_utils::values_equal(static_cast<vecops::float16_t>(static_cast<float>(-3.25f)), this->aligned_out_[0]));
  } else if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    auto v_zero = fill(t, vecops::bfloat16_t{});
    auto v_pos  = fill(t, static_cast<vecops::bfloat16_t>(static_cast<float>(2.5f)));
    auto v_neg  = fill(t, static_cast<vecops::bfloat16_t>(static_cast<float>(-7.5f)));

    storeu(t, this->aligned_out_, v_zero);
    EXPECT_TRUE(test_utils::values_equal(vecops::bfloat16_t{}, this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_pos);
    EXPECT_TRUE(test_utils::values_equal(static_cast<vecops::bfloat16_t>(static_cast<float>(2.5f)), this->aligned_out_[0]));
    storeu(t, this->aligned_out_, v_neg);
    EXPECT_TRUE(test_utils::values_equal(static_cast<vecops::bfloat16_t>(static_cast<float>(-7.5f)), this->aligned_out_[0]));
  }
}

TYPED_TEST(VecLoadStoreTest, InitializerListLoad) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto data = std::make_unique<T[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    data[i] = test_utils::get_test_value<T>(i + 100);
  }

  auto v = loadu(t, data.get());

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(data[i], get(v, i)));
  }
}

// ============================================================================
// Multi-word Vector Tests (2 registers / 4 registers)
// ============================================================================

TYPED_TEST(VecLoadStoreTest, MultiWordFill2) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;

  T fill_val = test_utils::get_test_value<T>(77);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordLoadStore2) {
  auto& t = this->t2;
  nint_t N = this->multi2_size;

  auto v = loadu(t, this->aligned_data_);
  storeu(t, this->aligned_out_, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordMask2) {
#ifndef CPU_CAPABILITY_SVE
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
#endif
}

TYPED_TEST(VecLoadStoreTest, MultiWordLoadWithN2) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  nint_t n = N - N / 4;

  auto v = loadu(t, this->aligned_data_, n, default_v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordStoreWithN2) {
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < N; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  nint_t n = N - N / 4;

  storeu(t, this->aligned_out_, n, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordSetElement4) {
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
// Gather / Scatter Test Fixture
// ============================================================================

template <typename T>
class VecGatherScatterTest : public ::testing::Test {
protected:
  using Type = T;

  ScalableTag<T, 0> t;
  nint_t full_size;

  void SetUp() override {
    full_size = size(t);

    aligned_data_ = test_utils::alloc_aligned<T>(256);
    aligned_out_ = test_utils::alloc_aligned<T>(256);

    for (size_t i = 0; i < 256; ++i) {
      aligned_data_[i] = test_utils::get_test_value<T>(i);
      aligned_out_[i] = T{};
    }
  }

  void TearDown() override {
    std::free(aligned_data_);
    std::free(aligned_out_);
  }

  T* aligned_data_{};
  T* aligned_out_{};
};

using GatherScatterTypes = ::testing::Types<
    float32_t, float64_t, int32_t, uint32_t, int64_t, uint64_t
>;

TYPED_TEST_SUITE(VecGatherScatterTest, GatherScatterTypes);

// ============================================================================
// Gather Tests
// ============================================================================

TYPED_TEST(VecGatherScatterTest, GatherBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i * 2);
  }

  auto idx = loadu(it, indices.get());
  auto v = gather(t, this->aligned_data_, idx);

  for (nint_t i = 0; i < N; ++i) {
    T expected = this->aligned_data_[i * 2];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>((i * 3) % 128);
  }

  auto idx = loadu(it, indices.get());
  T default_val = test_utils::get_test_value<T>(999);
  nint_t n = N / 2;

  auto v = gather(t, this->aligned_data_, idx, n, default_val);

  for (nint_t i = 0; i < n; ++i) {
    T expected = this->aligned_data_[indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
  for (nint_t i = n; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithNZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  T default_val = test_utils::get_test_value<T>(999);

  auto v = gather(t, this->aligned_data_, idx, 0, default_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithNFull) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  T default_val = test_utils::get_test_value<T>(999);

  auto v = gather(t, this->aligned_data_, idx, N, default_val);

  for (nint_t i = 0; i < N; ++i) {
    T expected = this->aligned_data_[i];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>((i * 5 + 10) % 128);
  }

  auto idx = loadu(it, indices.get());
  auto m = mwhilelt(t, 0, N / 2);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = this->aligned_data_[indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMaskAll) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  auto m = mtrue(t);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  for (nint_t i = 0; i < N; ++i) {
    T expected = this->aligned_data_[i];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMaskNone) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  auto m = mfalse(t);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

// ============================================================================
// Scatter Tests
// ============================================================================

TYPED_TEST(VecGatherScatterTest, ScatterBasic) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i * 2);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithN) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>((i * 3) % 128);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);
  nint_t n = N / 2;

  scatter(t, this->aligned_out_, idx, n, v);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithNZero) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, 0, v);

  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithNFull) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i + 50);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, N, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMask) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>((i * 5 + 10) % 128);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);
  auto m = mwhilelt(t, 0, N / 2);

  scatter(t, this->aligned_out_, idx, m, v);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMaskAll) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i + 100);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);
  auto m = mtrue(t);

  scatter(t, this->aligned_out_, idx, m, v);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMaskNone) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t, this->aligned_data_);
  auto m = mfalse(t);

  scatter(t, this->aligned_out_, idx, m, v);

  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherScatterRoundTrip) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 0> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i + 128);
  }

  auto idx = loadu(it, indices.get());

  for (int i = 128; i < 128 + N; ++i) {
    this->aligned_data_[i] = test_utils::get_test_value<T>(i);
  }

  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = T{};
  }

  auto v = gather(t, this->aligned_data_, idx);

  auto scatter_indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    scatter_indices[i] = static_cast<IndexT>(i);
  }
  auto scatter_idx = loadu(it, scatter_indices.get());

  scatter(t, this->aligned_out_, scatter_idx, v);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::get_test_value<T>(static_cast<int>(indices[i]));
    EXPECT_TRUE(test_utils::values_equal(expected, this->aligned_out_[i]));
  }
}

// ============================================================================
// Multi-word Gather / Scatter Tests
// ============================================================================

TYPED_TEST(VecGatherScatterTest, MultiWordGather) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);
  nint_t ws = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 1> it;

  // Reverse within each word — tests that vmap offsets base per word
  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(ws - 1 - (i % ws));
  }

  auto idx = loadu(it, indices.get());
  auto v = gather(t2, this->aligned_data_, idx);

  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / ws;
    T expected = this->aligned_data_[word * ws + indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, MultiWordScatter) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);
  nint_t ws = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 1> it;

  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  // Reverse within each word — scatter input to reversed positions
  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(ws - 1 - (i % ws));
  }

  auto idx = loadu(it, indices.get());
  auto v = loadu(t2, this->aligned_data_);

  scatter(t2, this->aligned_out_, idx, v);

  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / ws;
    nint_t pos = i % ws;
    nint_t dst = static_cast<nint_t>(word * ws + indices[word * ws + pos]);
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[dst]));
  }
}

TYPED_TEST(VecGatherScatterTest, MultiWordGatherWithN) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);
  if (N < 2) return;
  nint_t ws = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 1> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>(i % ws);
  }

  auto idx = loadu(it, indices.get());
  auto m = mwhilelt(t2, 0, N / 2);
  T default_val = test_utils::get_test_value<T>(999);

  auto v = gather(t2, this->aligned_data_, idx, m, default_val);

  for (nint_t i = 0; i < N / 2; ++i) {
    nint_t word = i / ws;
    T expected = this->aligned_data_[word * ws + indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, MultiWordGatherWithMask) {
  using T = typename TestFixture::Type;
  ScalableTag<T, 1> t2;
  nint_t N = size(t2);
  if (N < 2) return;
  nint_t ws = this->full_size;
  using IndexT = Index<T>;
  ScalableTag<IndexT, 1> it;

  auto indices = std::make_unique<IndexT[]>(N);
  for (nint_t i = 0; i < N; ++i) {
    indices[i] = static_cast<IndexT>((i * 5 + 3) % ws);
  }

  auto idx = loadu(it, indices.get());
  auto m = mwhilelt(t2, 0, N / 2);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t2, this->aligned_data_, idx, m, default_val);

  for (nint_t i = 0; i < N / 2; ++i) {
    nint_t word = i / ws;
    T expected = this->aligned_data_[word * ws + indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(v, i)));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(v, i)));
  }
}

// ============================================================================
// ScalableTag Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, ScalableTagTest) {
  auto& t = this->t;
  nint_t N = this->full_size;

  EXPECT_EQ(size(t), N);
#ifndef CPU_CAPABILITY_GENERIC
  EXPECT_FALSE(is_default_impl(t));
#endif
}

TYPED_TEST(VecLoadStoreTest, ScalableTagFill) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  T fill_val = test_utils::get_test_value<T>(123);
  auto v = fill(t, fill_val);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(v, i)));
  }
}

// ============================================================================
// Utility function test (vectorized copy)
// ============================================================================

template <typename T>
VECOPS_NOINLINE
static void vectorized_copy(const T* from, T* to, nint_t len) {
  ScalableTag<T> t;
  nint_t vec_size = size(t);
  nint_t i;

  for (i = 0; i <= len - vec_size; i += vec_size) {
    auto v = loadu(t, from + i);
    storeu(t, to + i, v);
  }

  if (i < len) {
    auto m = mwhilelt(t, i, len);
    auto v = loadu(t, from + i, m, zeros(t));
    storeu(t, to + i, m, v);
  }
}

TYPED_TEST(VecLoadStoreTest, VectorizedCopyFunction) {
  vectorized_copy(this->aligned_data_, this->aligned_out_, 100);

  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, VectorizedCopyOddLength) {
  auto& t = this->t;
  nint_t N = this->full_size;
  nint_t test_len = N * 3 + N / 2;

  vectorized_copy(this->aligned_data_, this->aligned_out_, test_len);

  for (nint_t i = 0; i < test_len; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
