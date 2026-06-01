//
// ShuffleTest.cpp — Comprehensive shuffle operation tests
// Covers: shuf, local_shuf (CT indices, vector indices, scalar indices),
//         upper/lower, even/odd, concat, concat_even/odd,
//         interleave, interleave_even/odd, local_interleave_lower/upper
// Supports both x86 (FixedTag legacy) and ARM SVE (ScalableTag)
//

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <type_traits>

#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace test_utils {

template <typename T>
constexpr nint_t lane_size() { return 16 / static_cast<nint_t>(sizeof(T)); }

template <typename T> struct ShuffleIndex     { using type = T; };
template <> struct ShuffleIndex<float32_t>     { using type = int32_t; };
template <> struct ShuffleIndex<float64_t>     { using type = int64_t; };
template <> struct ShuffleIndex<uint32_t>      { using type = int32_t; };
template <> struct ShuffleIndex<uint64_t>      { using type = int64_t; };
template <> struct ShuffleIndex<uint8_t>       { using type = int8_t; };
template <> struct ShuffleIndex<uint16_t>      { using type = int16_t; };
template <> struct ShuffleIndex<vecops::float16_t>      { using type = int16_t; };
template <> struct ShuffleIndex<vecops::bfloat16_t>      { using type = int16_t; };
template <typename T> using shuffle_idx_t = typename ShuffleIndex<T>::type;

template <typename T>
void fill_seq(T* data, nint_t n) {
  for (nint_t i = 0; i < n; ++i)
    data[i] = static_cast<T>(i + 1);
}

template <typename I>
void fill_local_identity(I* idx, nint_t n) {
  constexpr nint_t M = 16 / sizeof(I);
  for (nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % M);
}

template <typename I>
void fill_shuf_identity(I* idx, nint_t n, nint_t word_size) {
  for (nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % word_size);
}

template <typename T>
::testing::AssertionResult values_equal(T expected, T actual) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    float e = static_cast<float>(expected), a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    float e = static_cast<float>(expected), a = static_cast<float>(actual);
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
// Type groups
// ============================================================================

using Types4  = ::testing::Types<float32_t, int32_t, uint32_t>;
using Types2  = ::testing::Types<float64_t, int64_t, uint64_t>;
using Types8  = ::testing::Types<int16_t, uint16_t>;
using Types16 = ::testing::Types<int8_t, uint8_t>;

using ShufTypes = ::testing::Types<
    float32_t, int32_t, uint32_t,
    float64_t, int64_t, uint64_t,
    int8_t,   uint8_t,
    int16_t,  uint16_t,
    float16_t
    #if defined(__ARM_FEATURE_BF16) || defined(ARCH_X86_FAMILY)
    , bfloat16_t
    #endif
>;

using AllTypes = ::testing::Types<
    float16_t,
    #if defined(__ARM_FEATURE_BF16) || defined(ARCH_X86_FAMILY)
    bfloat16_t,
    #endif
    float32_t, float64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t
>;

// ============================================================================
// Main typed test fixture (for most tests)
// ============================================================================

template <typename T>
class VecShuffleTest : public ::testing::Test {
protected:
  using Type = T;

  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;

  nint_t full_size;
  nint_t multi2_size;

  void SetUp() override {
    full_size = size(t);
    multi2_size = size(t2);
  }
};

template <typename T>
class VecShuffleAllTest : public VecShuffleTest<T> {};

// ============================================================================
// Compile-time local_shuf — 4 elements per lane (float32_t, int32_t, uint32_t)
// ============================================================================

template <typename T> class LocalShufCT4 : public ::testing::Test {
protected:
  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  nint_t N;
  nint_t N2;
  void SetUp() override { N = size(t); N2 = size(t2); }
};
TYPED_TEST_SUITE(LocalShufCT4, Types4);

TYPED_TEST(LocalShufCT4, Identity) {
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<3, 2, 1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT4, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1, 2, 3>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TYPED_TEST(LocalShufCT4, BroadcastFirst) {
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 0, 0, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[(i / M) * M]) << "i=" << i;
}

TYPED_TEST(LocalShufCT4, MixedPermutation) {
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<1, 3, 0, 2>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    static constexpr int perm[] = {2, 0, 3, 1};
    EXPECT_EQ(get(t, r, i), data[lane * M + perm[pos]]) << "i=" << i;
  }
}

// ============================================================================
// Compile-time local_shuf — 2 elements per lane (float64_t, int64_t, uint64_t)
// ============================================================================

template <typename T> class LocalShufCT2 : public ::testing::Test {
protected:
  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  nint_t N;
  nint_t N2;
  void SetUp() override { N = size(t); N2 = size(t2); }
};
TYPED_TEST_SUITE(LocalShufCT2, Types2);

TYPED_TEST(LocalShufCT2, Identity) {
  using T = TypeParam;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT2, Swap) {
  using T = TypeParam;
  constexpr auto M = 2;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// Compile-time local_shuf — 8 elements per lane (int16_t, uint16_t)
// ============================================================================

template <typename T> class LocalShufCT8 : public ::testing::Test {
protected:
  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  nint_t N;
  nint_t N2;
  void SetUp() override { N = size(t); N2 = size(t2); }
};
TYPED_TEST_SUITE(LocalShufCT8, Types8);

TYPED_TEST(LocalShufCT8, Identity) {
  using T = TypeParam;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<7, 6, 5, 4, 3, 2, 1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT8, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto M = 8;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1, 2, 3, 4, 5, 6, 7>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// Compile-time local_shuf — 16 elements per lane (int8_t, uint8_t)
// ============================================================================

template <typename T> class LocalShufCT16 : public ::testing::Test {
protected:
  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;
  nint_t N;
  nint_t N2;
  void SetUp() override { N = size(t); N2 = size(t2); }
};
TYPED_TEST_SUITE(LocalShufCT16, Types16);

TYPED_TEST(LocalShufCT16, Identity) {
  using T = TypeParam;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<15, 14, 13, 12, 11, 10, 9, 8,
                     7,  6,  5,  4,  3,  2,  1,  0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT16, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto M = 16;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10, 11, 12, 13, 14, 15>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TYPED_TEST(LocalShufCT16, BroadcastMiddle) {
  using T = TypeParam;
  constexpr auto M = 16;
  auto& t = this->t;
  nint_t N = this->N;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[(i / M) * M + 7]) << "i=" << i;
}

// ============================================================================
// local_shuf with vector indices
// ============================================================================

TYPED_TEST_SUITE(VecShuffleTest, ShufTypes);

TYPED_TEST(VecShuffleTest, LocalShufVI_Identity) {
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t;
  nint_t N = this->full_size;
  ScalableTag<I, 0> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  test_utils::fill_local_identity(idx.get(), N);
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(VecShuffleTest, LocalShufVI_ReverseWithinLane) {
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t;
  nint_t N = this->full_size;
  constexpr auto M = test_utils::lane_size<T>();
  ScalableTag<I, 0> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TYPED_TEST(VecShuffleTest, LocalShufVI_DifferentPerLane) {
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t;
  nint_t N = this->full_size;
  constexpr auto M = test_utils::lane_size<T>();
  ScalableTag<I, 0> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    idx[i] = (lane % 2 == 0)
        ? static_cast<I>(pos)
        : static_cast<I>(M - 1 - pos);
  }
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    T expected = (lane % 2 == 0)
        ? data[lane * M + pos]
        : data[lane * M + (M - 1 - pos)];
    EXPECT_EQ(get(t, r, i), expected) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with scalar indices (runtime int... parameters)
// ============================================================================

TEST(LocalShufScalar, Float32_Identity) {
  using T = float32_t;
  constexpr auto M = 4;
  ScalableTag<T, 0> t;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 3, 2, 1, 0);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufScalar, Float32_Reverse) {
  using T = float32_t;
  constexpr auto M = 4;
  ScalableTag<T, 0> t;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 0, 1, 2, 3);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TEST(LocalShufScalar, Float64_Swap) {
  using T = float64_t;
  constexpr auto M = 2;
  ScalableTag<T, 0> t;
  nint_t N = size(t);
  if (N < 2) return;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 0, 1);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TEST(LocalShufScalar, Int8_Identity) {
  using T = int8_t;
  constexpr auto M = 16;
  ScalableTag<T, 0> t;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 15, 14, 13, 12, 11, 10, 9, 8,
                         7,  6,  5,  4,  3,  2,  1,  0);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufScalar, Int16_Reverse) {
  using T = int16_t;
  constexpr auto M = 8;
  ScalableTag<T, 0> t;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 0, 1, 2, 3, 4, 5, 6, 7);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// shuf — identity for all type categories
// ============================================================================

TYPED_TEST(VecShuffleTest, ShufIdentity) {
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t;
  nint_t N = this->full_size;
  ScalableTag<I, 0> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  test_utils::fill_shuf_identity(idx.get(), N, N);
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

// ============================================================================
// shuf vs local_shuf — demonstrate cross-lane capability
// ============================================================================

TEST(ShufVsLocalShuf, Float32_FullWordReverse) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto M = 4;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  auto idx_local = std::make_unique<I[]>(N);
  auto idx_shuf = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);

  for (nint_t i = 0; i < N; ++i)
    idx_local[i] = static_cast<I>(M - 1 - (i % M));
  for (nint_t i = 0; i < N; ++i)
    idx_shuf[i] = static_cast<I>(N - 1 - i);

  auto v = loadu(t, data.get());
  auto r_local = local_shuf(v, loadu(ti, idx_local.get()));
  auto r_shuf  = shuf(v, loadu(ti, idx_shuf.get()));

  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r_local, i), data[lane * M + (M - 1 - pos)])
        << "local_shuf i=" << i;
    EXPECT_EQ(get(t, r_shuf, i), data[N - 1 - i])
        << "shuf i=" << i;
  }

  if (N == M) {
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(get(t, r_local, i), get(t, r_shuf, i));
  } else {
    bool differs = false;
    for (nint_t i = 0; i < N; ++i) {
      if (get(t, r_local, i) != get(t, r_shuf, i)) {
        differs = true;
        break;
      }
    }
    EXPECT_TRUE(differs) << "local_shuf and shuf should differ on multi-lane words";
  }
}

TEST(ShufVsLocalShuf, Int8_FullWordReverse) {
  using T = int8_t;
  using I = int8_t;
  constexpr auto M = 16;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  auto idx_local = std::make_unique<I[]>(N);
  auto idx_shuf = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);

  for (nint_t i = 0; i < N; ++i)
    idx_local[i] = static_cast<I>(M - 1 - (i % M));
  for (nint_t i = 0; i < N; ++i)
    idx_shuf[i] = static_cast<I>(N - 1 - i);

  auto v = loadu(t, data.get());
  auto r_local = local_shuf(v, loadu(ti, idx_local.get()));
  auto r_shuf  = shuf(v, loadu(ti, idx_shuf.get()));

  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r_local, i), data[lane * M + (M - 1 - pos)])
        << "local_shuf i=" << i;
    EXPECT_EQ(get(t, r_shuf, i), data[N - 1 - i])
        << "shuf i=" << i;
  }
}

// ============================================================================
// shuf — cross-lane swap for float64
// ============================================================================

TEST(Shuf, Float64_CrossLaneSwap) {
  using T = float64_t;
  using I = int64_t;
  constexpr auto M = 2;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>((i % M == 0) ? i + 1 : i - 1);
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    I expected_idx = static_cast<I>((i % M == 0) ? i + 1 : i - 1);
    EXPECT_EQ(get(t, r, i), data[expected_idx]) << "i=" << i;
  }
}

// ============================================================================
// upper / lower tests
// ============================================================================

TYPED_TEST_SUITE(VecShuffleAllTest, AllTypes);

TYPED_TEST(VecShuffleAllTest, UpperLower_SingleWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());

  auto lo = lower(t, v);
  auto hi = upper(t, v);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, lo, i), data[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, hi, i), data[i + N / 2]) << "upper i=" << i;
}

TYPED_TEST(VecShuffleAllTest, UpperLower_MultiWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());

  auto lo = lower(t, v);
  auto hi = upper(t, v);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, lo, i), data[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, hi, i), data[i + N / 2]) << "upper i=" << i;
#endif
}

// ============================================================================
// even / odd tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, EvenOdd_SingleWord) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());

  auto ev = even(t, v);
  auto od = odd(t, v);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data[2 * i]) << "even i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data[2 * i + 1]) << "odd i=" << i;
#endif
}

TYPED_TEST(VecShuffleAllTest, EvenOdd_MultiWord) {
#if !defined(CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY_GENERIC)
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());

  auto ev = even(t, v);
  auto od = odd(t, v);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data[2 * i]) << "even i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data[2 * i + 1]) << "odd i=" << i;
#endif
}

// ============================================================================
// concat tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, Concat_SingleWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data_lo = std::make_unique<T[]>(N / 2);
  auto data_hi = std::make_unique<T[]>(N / 2);
  test_utils::fill_seq(data_lo.get(), N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_hi[i] = static_cast<T>(data_lo[i]) + T(N / 2);

  auto v_lo = loadu(th, data_lo.get());
  auto v_hi = loadu(th, data_hi.get());
  auto v = concat(t, v_lo, v_hi);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i), data_lo[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i + N / 2), data_hi[i]) << "upper i=" << i;
}

TYPED_TEST(VecShuffleAllTest, Concat_MultiWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data_lo = std::make_unique<T[]>(N / 2);
  auto data_hi = std::make_unique<T[]>(N / 2);
  test_utils::fill_seq(data_lo.get(), N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_hi[i] = static_cast<T>(data_lo[i]) + T(N / 2);

  auto v_lo = loadu(th, data_lo.get());
  auto v_hi = loadu(th, data_hi.get());
  auto v = concat(t, v_lo, v_hi);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i), data_lo[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i + N / 2), data_hi[i]) << "upper i=" << i;
#endif
}

TYPED_TEST(VecShuffleAllTest, Concat_RoundTrip) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());

  auto lo = lower(t, v);
  auto hi = upper(t, v);
  auto v2 = concat(t, lo, hi);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, v2, i), data[i]) << "i=" << i;
}

// ============================================================================
// concat_even / concat_odd tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, ConcatEvenOdd_SingleWord) {
#ifndef CPU_CAPABILITY_SVE
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto ce = concat_even(t, a, b);
  auto co = concat_odd(t, a, b);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i), data_a[2 * i]) << "concat_even lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i + N / 2), data_b[2 * i]) << "concat_even upper i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i), data_a[2 * i + 1]) << "concat_odd lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i + N / 2), data_b[2 * i + 1]) << "concat_odd upper i=" << i;
#endif
}

TYPED_TEST(VecShuffleAllTest, ConcatEvenOdd_MultiWord) {
#if !defined(CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY_GENERIC)
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto ce = concat_even(t, a, b);
  auto co = concat_odd(t, a, b);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i), data_a[2 * i]) << "concat_even lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i + N / 2), data_b[2 * i]) << "concat_even upper i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i), data_a[2 * i + 1]) << "concat_odd lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i + N / 2), data_b[2 * i + 1]) << "concat_odd upper i=" << i;
#endif
}

// ============================================================================
// local_interleave_lower / local_interleave_upper tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, LocalInterleave_SingleWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  constexpr auto M = test_utils::lane_size<T>();
  if (N < M) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, lo, out_idx), data_a[idx]) << "lower a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, lo, out_idx + 1), data_b[idx]) << "lower b lane=" << lane << " i=" << i;
    }
  }

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, hi, out_idx), data_a[idx]) << "upper a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, hi, out_idx + 1), data_b[idx]) << "upper b lane=" << lane << " i=" << i;
    }
  }
}

TYPED_TEST(VecShuffleAllTest, LocalInterleave_MultiWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  constexpr auto M = test_utils::lane_size<T>();
  if (N < M) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, lo, out_idx), data_a[idx]) << "lower a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, lo, out_idx + 1), data_b[idx]) << "lower b lane=" << lane << " i=" << i;
    }
  }

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, hi, out_idx), data_a[idx]) << "upper a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, hi, out_idx + 1), data_b[idx]) << "upper b lane=" << lane << " i=" << i;
    }
  }
#endif
}

// ============================================================================
// interleave tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, Interleave_SingleWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data_a = std::make_unique<T[]>(N / 2);
  auto data_b = std::make_unique<T[]>(N / 2);
  test_utils::fill_seq(data_a.get(), N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N / 2);

  auto a = loadu(th, data_a.get());
  auto b = loadu(th, data_b.get());
  auto v = interleave(t, a, b);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, v, 2 * i), data_a[i]) << "a i=" << i;
    EXPECT_EQ(get(t, v, 2 * i + 1), data_b[i]) << "b i=" << i;
  }
}

TYPED_TEST(VecShuffleAllTest, Interleave_MultiWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data_a = std::make_unique<T[]>(N / 2);
  auto data_b = std::make_unique<T[]>(N / 2);
  test_utils::fill_seq(data_a.get(), N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N / 2);

  auto a = loadu(th, data_a.get());
  auto b = loadu(th, data_b.get());
  auto v = interleave(t, a, b);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, v, 2 * i), data_a[i]) << "a i=" << i;
    EXPECT_EQ(get(t, v, 2 * i + 1), data_b[i]) << "b i=" << i;
  }
#endif
}

// ============================================================================
// interleave_even / interleave_odd tests
// ============================================================================

TYPED_TEST(VecShuffleAllTest, InterleaveEvenOdd_SingleWord) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;
  if (N < 4) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto ie = interleave_even(a, b);
  auto io = interleave_odd(a, b);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, ie, 2 * i), data_a[2 * i]) << "ie a i=" << i;
    EXPECT_EQ(get(t, ie, 2 * i + 1), data_b[2 * i]) << "ie b i=" << i;
  }
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, io, 2 * i), data_a[2 * i + 1]) << "io a i=" << i;
    EXPECT_EQ(get(t, io, 2 * i + 1), data_b[2 * i + 1]) << "io b i=" << i;
  }
}

TYPED_TEST(VecShuffleAllTest, InterleaveEvenOdd_MultiWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  if (N < 4) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto ie = interleave_even(a, b);
  auto io = interleave_odd(a, b);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, ie, 2 * i), data_a[2 * i]) << "ie a i=" << i;
    EXPECT_EQ(get(t, ie, 2 * i + 1), data_b[2 * i]) << "ie b i=" << i;
  }
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, io, 2 * i), data_a[2 * i + 1]) << "io a i=" << i;
    EXPECT_EQ(get(t, io, 2 * i + 1), data_b[2 * i + 1]) << "io b i=" << i;
  }
#endif
}

// ============================================================================
// Multi-word local_shuf with compile-time indices
// ============================================================================

TYPED_TEST(LocalShufCT4, MultiWord_Identity) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<3, 2, 1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
#endif
}

TYPED_TEST(LocalShufCT4, MultiWord_Reverse) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1, 2, 3>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
#endif
}

TYPED_TEST(LocalShufCT16, MultiWord_Identity) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<15, 14, 13, 12, 11, 10, 9, 8,
                     7,  6,  5,  4,  3,  2,  1,  0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
#endif
}

TYPED_TEST(LocalShufCT2, MultiWord_Swap) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  constexpr auto M = 2;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf<0, 1>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
#endif
}

// ============================================================================
// Multi-word local_shuf with vector indices
// ============================================================================

TYPED_TEST(VecShuffleTest, LocalShufVI_MultiWord_Identity) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  ScalableTag<I, 1> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  test_utils::fill_local_identity(idx.get(), N);
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
#endif
}

TYPED_TEST(VecShuffleTest, LocalShufVI_MultiWord_Reverse) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  constexpr auto M = test_utils::lane_size<T>();
  ScalableTag<I, 1> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
#endif
}

// ============================================================================
// Multi-word shuf
// ============================================================================

TYPED_TEST(VecShuffleTest, Shuf_MultiWord_ReverseWithinWord) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  nint_t ws = this->full_size;
  ScalableTag<I, 1> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / ws;
    nint_t pos  = i % ws;
    idx[i] = static_cast<I>(ws - 1 - pos);
  }
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / ws;
    nint_t pos  = i % ws;
    EXPECT_EQ(get(t, r, i), data[word * ws + (ws - 1 - pos)]) << "i=" << i;
  }
#endif
}

TYPED_TEST(VecShuffleTest, Shuf_MultiWord_Identity) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  nint_t ws = this->full_size;
  ScalableTag<I, 1> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  test_utils::fill_shuf_identity(idx.get(), N, ws);
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
#endif
}

TYPED_TEST(VecShuffleTest, Shuf_MultiWord_Swap) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = typename TestFixture::Type;
  using I = test_utils::shuffle_idx_t<T>;
  auto& t = this->t2;
  nint_t N = this->multi2_size;
  nint_t ws = this->full_size;
  constexpr auto M = 2;
  ScalableTag<I, 1> ti;

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i) {
    nint_t pos = i % ws;
    idx[i] = static_cast<I>((pos % M == 0) ? pos + 1 : pos - 1);
  }
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    nint_t pos = i % ws;
    I expected_idx = static_cast<I>((pos % M == 0) ? pos + 1 : pos - 1);
    EXPECT_EQ(get(t, r, i), data[i / ws * ws + expected_idx]) << "i=" << i;
  }
#endif
}

// ============================================================================
// Multi-word local_shuf with scalar indices
// ============================================================================

TYPED_TEST(LocalShufCT4, MultiWord_ScalarReverse) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  constexpr auto M = 4;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 0, 1, 2, 3);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
#endif
}

TYPED_TEST(LocalShufCT2, MultiWord_ScalarSwap) {
#ifndef CPU_CAPABILITY_GENERIC
  using T = TypeParam;
  constexpr auto M = 2;
  auto& t = this->t2;
  nint_t N = this->N2;

  auto data = std::make_unique<T[]>(N);
  test_utils::fill_seq(data.get(), N);
  auto v = loadu(t, data.get());
  auto r = local_shuf(v, 0, 1);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
#endif
}

// ============================================================================
// Corner cases
// ============================================================================

TEST(ShufCornerCase, SingleLaneEqualsLocalShuf) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto M = 4;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  // Arbitrary permutation
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>((i % M == 0) ? i + 2 : i - 1);

  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r_local = local_shuf(v, vi);
  auto r_shuf  = shuf(v, vi);

  // On single-lane words, shuf == local_shuf
  if (N == M) {
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(get(t, r_local, i), get(t, r_shuf, i)) << "i=" << i;
  }
}

TEST(ShufCornerCase, ZeroVector) {
  using T = int32_t;
  using I = int32_t;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto v = zeros(t);
  auto idx = std::make_unique<I[]>(N);
  for (nint_t i = 0; i < N; ++i) idx[i] = static_cast<I>((N - 1 - i) % N);
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), T(0)) << "i=" << i;
}

TEST(ShufCornerCase, BroadcastFirst) {
  using T = float32_t;
  using I = int32_t;
  ScalableTag<T, 0> t;
  ScalableTag<I, 0> ti;
  nint_t N = size(t);

  auto data = std::make_unique<T[]>(N);
  auto idx = std::make_unique<I[]>(N);
  test_utils::fill_seq(data.get(), N);
  for (nint_t i = 0; i < N; ++i) idx[i] = 0;
  auto v = loadu(t, data.get());
  auto vi = loadu(ti, idx.get());
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[0]) << "i=" << i;
}

// ============================================================================
// Combined roundtrip tests
// ============================================================================

TEST(InterleaveRoundTrip, Float32) {
#ifndef CPU_CAPABILITY_SVE
  using T = float32_t;
  ScalableTag<T, 0> t;
  nint_t N = size(t);
  if (N < 2) return;
  Half<std::remove_reference_t<decltype(t)>> th;

  auto data_a = std::make_unique<T[]>(N / 2);
  auto data_b = std::make_unique<T[]>(N / 2);
  test_utils::fill_seq(data_a.get(), N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N / 2);

  auto a = loadu(th, data_a.get());
  auto b = loadu(th, data_b.get());

  auto v = interleave(t, a, b);
  auto ev = even(t, v);
  auto od = odd(t, v);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data_a[i]) << "even i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data_b[i]) << "odd i=" << i;
#endif
}

TEST(LocalInterleaveRoundTrip, Float64) {
  using T = float64_t;
  ScalableTag<T, 0> t;
  nint_t N = size(t);
  constexpr auto M = test_utils::lane_size<T>();
  if (N < M) return;

  auto data_a = std::make_unique<T[]>(N);
  auto data_b = std::make_unique<T[]>(N);
  test_utils::fill_seq(data_a.get(), N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i]) + T(N);

  auto a = loadu(t, data_a.get());
  auto b = loadu(t, data_b.get());

  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      EXPECT_EQ(get(t, lo, lane * M + 2 * i), data_a[idx]);
      EXPECT_EQ(get(t, lo, lane * M + 2 * i + 1), data_b[idx]);
    }
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      EXPECT_EQ(get(t, hi, lane * M + 2 * i), data_a[idx]);
      EXPECT_EQ(get(t, hi, lane * M + 2 * i + 1), data_b[idx]);
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
