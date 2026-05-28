//
// BasicTest.cpp — SVE_Basic.h operation tests
// Single-word (POW2=0) for all operations
//

#include <gtest/gtest.h>
#include <cmath>
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace test_utils {
template <typename T> T get_val(int idx) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    return static_cast<bfloat16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return static_cast<float16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float32_t>)      return T(idx * 1.5f + 0.5f);
  else if constexpr (std::is_same_v<T, float64_t>) return T(idx * 1.5 + 0.5);
  else if constexpr (std::is_same_v<T, int8_t>)    return T((idx * 13 + 7) % 127 - 64);
  else if constexpr (std::is_same_v<T, uint8_t>)   return T((idx * 13 + 7) % 255 + 1);
  else if constexpr (std::is_same_v<T, int16_t>)   return T((idx * 97 + 53) % 32000 - 16000);
  else if constexpr (std::is_same_v<T, uint16_t>)  return T((idx * 97 + 53) % 64000 + 100);
  else if constexpr (std::is_same_v<T, int32_t>)   return T(idx * 10000 + 500);
  else if constexpr (std::is_same_v<T, uint32_t>)  return T(idx * 10000u + 500u);
  else if constexpr (std::is_same_v<T, int64_t>)   return T(idx * 100000LL + 50000LL);
  else                                              return T(idx * 100000ULL + 50000ULL);
}
template <typename T>
::testing::AssertionResult eq(T expect, T actual) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    float e = static_cast<float>(expect);
    float a = static_cast<float>(actual);
    float diff = std::fabs(e - a);
    float tol = std::max(std::max(std::fabs(e), std::fabs(a)) * 1e-2f, 1e-3f);
    if (diff <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << e << " vs " << a << " diff=" << diff;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    float e = static_cast<float>(expect);
    float a = static_cast<float>(actual);
    float diff = std::fabs(e - a);
    float tol = std::max(std::max(std::fabs(e), std::fabs(a)) * 1e-2f, 1e-3f);
    if (diff <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << e << " vs " << a << " diff=" << diff;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::fabs(expect - actual) < 1e-5f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << expect << " vs " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::fabs(expect - actual) < 1e-12) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << expect << " vs " << actual;
  } else {
    if (expect == actual) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << (long long)expect << " vs " << (long long)actual;
  }
}
}  // namespace test_utils

template <typename T> class BasicTest : public ::testing::Test { protected: using Type = T; };
using TestedTypes = ::testing::Types<float32_t,float64_t,int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,int64_t,uint64_t,float16_t
#if defined(__ARM_FEATURE_BF16)
  ,bfloat16_t
#endif
>;
TYPED_TEST_SUITE(BasicTest, TestedTypes);

#define SW ScalableTag<typename TestFixture::Type, 0>

TYPED_TEST(BasicTest, Fill) {
  using T = typename TestFixture::Type;
  SW t; T v = test_utils::get_val<T>(42); auto r = fill(t, v);
  for (nint_t i = 0; i < size(t); ++i) ASSERT_TRUE(test_utils::eq(v, get(r, i)));
}
TYPED_TEST(BasicTest, Zeros) {
  using T = typename TestFixture::Type;
  SW t; auto r = zeros(t);
  for (nint_t i = 0; i < size(t); ++i) ASSERT_TRUE(test_utils::eq(T{}, get(r, i)));
}
TYPED_TEST(BasicTest, FillWithN) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 2) return; nint_t n = size(t) / 2;
  T va = test_utils::get_val<T>(1), vb = test_utils::get_val<T>(99);
  auto r = fill(t, va, n, fill(t, vb));
  for (nint_t i = 0; i < n; ++i) ASSERT_TRUE(test_utils::eq(va, get(r, i)));
  for (nint_t i = n; i < size(t); ++i) ASSERT_TRUE(test_utils::eq(vb, get(r, i)));
}
TYPED_TEST(BasicTest, FillWithMask) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 2) return; nint_t n = size(t) / 2;
  T va = test_utils::get_val<T>(1), vb = test_utils::get_val<T>(99);
  auto m = mwhilelt(t, 0, n);
  auto r = fill(t, va, m, fill(t, vb));
  for (nint_t i = 0; i < n; ++i) ASSERT_TRUE(test_utils::eq(va, get(r, i)));
  for (nint_t i = n; i < size(t); ++i) ASSERT_TRUE(test_utils::eq(vb, get(r, i)));
}
TYPED_TEST(BasicTest, GetSet) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 4) return;
  auto v = fill(t, test_utils::get_val<T>(0));
  T x0 = test_utils::get_val<T>(10), x1 = test_utils::get_val<T>(20);
  v = set(v, 0, x0); v = set(v, size(t) - 1, x1);
  ASSERT_TRUE(test_utils::eq(x0, get(v, 0)));
  ASSERT_TRUE(test_utils::eq(x1, get(v, size(t) - 1)));
}
TYPED_TEST(BasicTest, Mfill) {
  using T = typename TestFixture::Type;
  SW t; auto mt = mtrue(t), mf = mfalse(t);
  for (nint_t i = 0; i < size(t); ++i) { ASSERT_TRUE(get(t, mt, i)); ASSERT_FALSE(get(t, mf, i)); }
}
TYPED_TEST(BasicTest, Mwhilelt) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) <= 3) return;
  auto m = mwhilelt(t, 2, (nint_t)size(t) - 1);
  for (nint_t i = 0; i < size(t); ++i) ASSERT_EQ(get(t, m, i), (2 + i) < (size(t) - 1));
}
TYPED_TEST(BasicTest, Mwhilege) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) <= 3) return;
  auto m = mwhilege(t, 2, (nint_t)size(t) - 1);
  for (nint_t i = 0; i < size(t); ++i) ASSERT_EQ(get(t, m, i), (2 + i) >= (size(t) - 1));
}
TYPED_TEST(BasicTest, MaskSetGet) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 4) return;
  auto m = mfalse(t); m = set(t, m, 1, true); m = set(t, m, size(t)-2, true);
  ASSERT_TRUE(get(t, m, 1)); ASSERT_TRUE(get(t, m, size(t)-2));
  ASSERT_FALSE(get(t, m, 0)); ASSERT_FALSE(get(t, m, size(t)-1));
}
TYPED_TEST(BasicTest, MaskBitOps) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 4) return; nint_t n = size(t);
  auto a = mwhilelt(t, 0, n/2+1), b = mwhilelt(t, n/4, 3*n/4);
  auto c_and = word::bit_and(a,b), c_or = word::bit_or(a,b), c_xor = word::bit_xor(a,b);
  auto c_nand = word::bit_andnot(a,b), c_not = word::bit_not(a);
  for (nint_t i = 0; i < n; ++i) {
    bool va = get(t, a, i), vb = get(t, b, i);
    ASSERT_EQ(get(t, c_and, i), va && vb);
    ASSERT_EQ(get(t, c_or, i),  va || vb);
    ASSERT_EQ(get(t, c_xor, i), va != vb);
    ASSERT_EQ(get(t, c_nand, i),!va && vb);
    ASSERT_EQ(get(t, c_not, i), !va);
  }
}
TYPED_TEST(BasicTest, Blend) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 4) return; nint_t n = size(t);
  T va = test_utils::get_val<T>(5), vb = test_utils::get_val<T>(95);
  auto v0 = fill(t, va);
  auto v1 = fill(t, vb);
  auto m = mwhilelt(t, 0, n/2);
  auto r = word::blend(v0, m, v1);
  for (nint_t i = 0; i < n; ++i) ASSERT_TRUE(test_utils::eq(get(t, m, i) ? vb : va, get(r, i)));
}
TYPED_TEST(BasicTest, Shuf) {
  using T = typename TestFixture::Type;
  SW t; if (size(t) < 4) return; nint_t n = size(t);
  auto v = fill(t, test_utils::get_val<T>(0));
  for (nint_t i = 0; i < n; ++i) v = set(v, i, T(i));
  using IdxT = Index<T>; ScalableTag<IdxT, 0> ti; auto idx = fill(ti, IdxT(0));
  for (nint_t i = 0; i < n; ++i) idx = set(idx, i, IdxT(n-1-i));
  auto r = shuf(v, idx);
  for (nint_t i = 0; i < n; ++i) ASSERT_TRUE(test_utils::eq(T(n-1-i), get(r, i)));
}
TYPED_TEST(BasicTest, LocalShuf) {
  using T = typename TestFixture::Type;
  SW t; nint_t n = size(t);
  constexpr nint_t group_el = 16 / (nint_t)sizeof(T);
  if (n < group_el || n % group_el != 0) return;
  auto v = fill(t, test_utils::get_val<T>(0));
  for (nint_t i = 0; i < n; ++i) v = set(v, i, T(i));
  using IdxT = Index<T>; ScalableTag<IdxT, 0> ti; auto idx = fill(ti, IdxT(0));
  nint_t n_groups = n / group_el;
  for (nint_t g = 0; g < n_groups; ++g)
    for (nint_t j = 0; j < group_el; ++j)
      idx = set(idx, g*group_el+j, IdxT(group_el-1-j));
  auto r = local_shuf(v, idx);
  for (nint_t g = 0; g < n_groups; ++g)
    for (nint_t j = 0; j < group_el; ++j)
      ASSERT_TRUE(test_utils::eq(T(g*group_el+group_el-1-j), get(r, g*group_el+j)));
}
