//
// MaskTest.cpp
// Comprehensive test for Mask operations:
//   bit_and, bit_or, bit_xor, bit_andnot, bit_not,
//   lower, upper, concat
//
// Supports: x86 SSE/AVX/AVX-512, ARM SVE/NEON, and scalar fallback.
//

#include <gtest/gtest.h>
#include <cstring>
#include <limits>

#include "TestUtils.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class VecMaskTest : public ::testing::Test {
protected:
  using Type = T;

  ScalableTag<T, 0> t;
  ScalableTag<T, 1> t2;

  nint_t full_size;
  nint_t multi2_size;

  void SetUp() override {
    full_size   = size(t);
    multi2_size = size(t2);

    data_ = test_utils::alloc_aligned<T>(256);
    for (size_t i = 0; i < 256; ++i) {
      data_[i] = test_utils::get_test_value<T>(i);
    }
  }

  void TearDown() override {
    std::free(data_);
  }

  T* data_{};

  // Helper: create a mask from a bool pattern using comparisons
  template <typename Tag>
  Mask<Tag> make_mask_from_pattern(Tag tt, const std::vector<bool>& pattern) {
    return test_utils::make_mask(tt, pattern);
  }

  // Helper: verify mask matches expected pattern
  template <typename Tag>
  void verify_mask(Tag tt, Mask<Tag> m, const std::vector<bool>& expected, int line) {
    test_utils::verify_mask_pattern(tt, m, expected, line);
  }
};

TYPED_TEST_SUITE(VecMaskTest, test_utils::UnsignedIntTypes);


// ============================================================================
// Mask Bit Operations
// ============================================================================

TYPED_TEST(VecMaskTest, MaskBitAnd) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  // Pattern A: every other bit set
  auto pat_a = test_utils::make_pattern(N, 1, 1);
  // Pattern B: bits with i % 4 < 2 set
  auto pat_b = test_utils::make_pattern(N, 3, 0);  // 0,1,4,5,8,9,...

  auto ma = this->make_mask_from_pattern(t, pat_a);
  auto mb = this->make_mask_from_pattern(t, pat_b);

  auto mr = bit_and(t, ma, mb);
  auto expected = test_utils::scalar_mask_and(pat_a, pat_b);

  for (nint_t i = 0; i < N; ++i) {
    bool actual = get(t, mr, i);
    EXPECT_EQ(expected[(size_t)i], actual) << "MaskBitAnd i=" << i;
  }

  // Also test all-true and all-false
  auto v_all = loadu(t, this->data_);
  auto m_true = cmpeq(v_all, v_all);
  auto m_false = cmpne(v_all, v_all);

  auto mr_tt = bit_and(t, m_true, m_true);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, mr_tt, i));

  auto mr_ff = bit_and(t, m_false, m_true);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, mr_ff, i));
}

TYPED_TEST(VecMaskTest, MaskBitOr) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto pat_a = test_utils::make_pattern(N, 1, 1);  // odd bits
  auto pat_b = test_utils::make_pattern(N, 1, 0);  // even bits

  auto ma = this->make_mask_from_pattern(t, pat_a);
  auto mb = this->make_mask_from_pattern(t, pat_b);

  auto mr = bit_or(t, ma, mb);
  auto expected = test_utils::scalar_mask_or(pat_a, pat_b);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(expected[(size_t)i], get(t, mr, i)) << "MaskBitOr i=" << i;
  }

  // all-true OR anything = all-true
  auto v = loadu(t, this->data_);
  auto m_true = cmpeq(v, v);
  auto mr2 = bit_or(t, m_true, ma);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, mr2, i));
}

TYPED_TEST(VecMaskTest, MaskBitXor) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto pat_a = test_utils::make_pattern(N, 2, 0);  // 0,1,4,5,8,9,...
  auto pat_b = test_utils::make_pattern(N, 1, 0);  // even bits

  auto ma = this->make_mask_from_pattern(t, pat_a);
  auto mb = this->make_mask_from_pattern(t, pat_b);

  auto mr = bit_xor(t, ma, mb);
  auto expected = test_utils::scalar_mask_xor(pat_a, pat_b);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(expected[(size_t)i], get(t, mr, i)) << "MaskBitXor i=" << i;
  }

  // XOR with self = all false
  auto mr2 = bit_xor(t, ma, ma);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, mr2, i));
}

TYPED_TEST(VecMaskTest, MaskBitAndnot) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto pat_a = test_utils::make_pattern(N, 1, 1);   // odd
  auto pat_b = test_utils::make_pattern(N, 2, 0);   // 0,1,4,5,...

  auto ma = this->make_mask_from_pattern(t, pat_a);
  auto mb = this->make_mask_from_pattern(t, pat_b);

  auto mr = bit_andnot(t, ma, mb);
  auto expected = test_utils::scalar_mask_andnot(pat_a, pat_b);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(expected[(size_t)i], get(t, mr, i)) << "MaskBitAndnot i=" << i;
  }

  // andnot(a, a) = all false
  auto mr2 = bit_andnot(t, ma, ma);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, mr2, i));
}

TYPED_TEST(VecMaskTest, MaskBitNot) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto pat_a = test_utils::make_pattern(N, 1, 0);  // even
  auto ma = this->make_mask_from_pattern(t, pat_a);

  auto mr = bit_not(t, ma);
  auto expected = test_utils::scalar_mask_not(pat_a);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(expected[(size_t)i], get(t, mr, i)) << "MaskBitNot i=" << i;
  }

  // NOT(all-false) = all-true
  auto v = loadu(t, this->data_);
  auto m_false = cmpne(v, v);
  auto mr2 = bit_not(t, m_false);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, mr2, i));
}


// ============================================================================
// Mask Lower / Upper
// ============================================================================

TYPED_TEST(VecMaskTest, MaskLower) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  // Create a mask with known pattern
  auto pat = test_utils::make_pattern(N, 3, 0);  // 0,1,4,5,8,9,12,13,...
  auto m = this->make_mask_from_pattern(t, pat);
  auto m_lo = lower(t, m);

  // Verify lower half
  auto expected = test_utils::scalar_mask_lower(pat);
  for (nint_t i = 0; i < N / 2; ++i) {
    bool actual = get(Half<ScalableTag<T, 0>>{}, m_lo, i);
    EXPECT_EQ(expected[(size_t)i], actual) << "MaskLower i=" << i;
  }

  // Lower half size should be N/2
  EXPECT_EQ((size_t)(N / 2), (size_t)size(Half<ScalableTag<T, 0>>{}));

  // Also test with all-false and all-true
  auto v = loadu(t, this->data_);
  auto m_false = cmpne(v, v);
  auto m_true = cmpeq(v, v);

  auto mf_lo = lower(t, m_false);
  for (nint_t i = 0; i < N / 2; ++i) EXPECT_FALSE(get(Half<ScalableTag<T, 0>>{}, mf_lo, i));

  auto mt_lo = lower(t, m_true);
  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(Half<ScalableTag<T, 0>>{}, mt_lo, i));
}

TYPED_TEST(VecMaskTest, MaskUpper) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  auto pat = test_utils::make_pattern(N, 7, 3);  // specific pattern
  auto m = this->make_mask_from_pattern(t, pat);
  auto m_hi = upper(t, m);

  auto expected = test_utils::scalar_mask_upper(pat);
  for (nint_t i = 0; i < N / 2; ++i) {
    bool actual = get(Half<ScalableTag<T, 0>>{}, m_hi, i);
    EXPECT_EQ(expected[(size_t)i], actual) << "MaskUpper i=" << i;
  }

  // Test with all-true: upper of all-true = all-true
  auto v = loadu(t, this->data_);
  auto m_true = cmpeq(v, v);
  auto mt_hi = upper(t, m_true);
  for (nint_t i = 0; i < N / 2; ++i) EXPECT_TRUE(get(Half<ScalableTag<T, 0>>{}, mt_hi, i));

  // Test with all-false
  auto m_false = cmpne(v, v);
  auto mf_hi = upper(t, m_false);
  for (nint_t i = 0; i < N / 2; ++i) EXPECT_FALSE(get(Half<ScalableTag<T, 0>>{}, mf_hi, i));
}


// ============================================================================
// Mask Concat
// ============================================================================

TYPED_TEST(VecMaskTest, MaskConcat) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  // Create two half-size patterns
  auto pat_lo = test_utils::make_pattern(N / 2, 1, 1);  // all odd
  auto pat_hi = test_utils::make_pattern(N / 2, 1, 0);  // all even

  Half<ScalableTag<T, 0>> th;
  auto m_lo = this->make_mask_from_pattern(th, pat_lo);
  auto m_hi = this->make_mask_from_pattern(th, pat_hi);

  auto m = concat(t, m_lo, m_hi);

  // Verify concatenated result
  auto expected = test_utils::scalar_mask_concat(pat_lo, pat_hi);
  for (nint_t i = 0; i < N; ++i) {
    bool actual = get(t, m, i);
    EXPECT_EQ(expected[(size_t)i], actual) << "MaskConcat i=" << i;
  }
}

TYPED_TEST(VecMaskTest, MaskConcatAllFalse) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  Half<ScalableTag<T, 0>> th;
  auto v_half = loadu(th, this->data_);
  auto m_false = cmpne(v_half, v_half);

  auto m = concat(t, m_false, m_false);
  for (nint_t i = 0; i < N; ++i) EXPECT_FALSE(get(t, m, i));
}

TYPED_TEST(VecMaskTest, MaskConcatAllTrue) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  Half<ScalableTag<T, 0>> th;
  auto v_half = loadu(th, this->data_);
  auto m_true = cmpeq(v_half, v_half);

  auto m = concat(t, m_true, m_true);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(get(t, m, i));
}


// ============================================================================
// Roundtrip: concat(lower(m), upper(m)) == m
// ============================================================================

TYPED_TEST(VecMaskTest, MaskRoundtrip) {
  using T = typename TestFixture::Type;
  auto& t = this->t;
  nint_t N = this->full_size;

  // Test with several patterns
  std::vector<std::vector<bool>> patterns;
  patterns.push_back(test_utils::make_pattern(N, 1, 0));     // alternating
  patterns.push_back(test_utils::make_pattern(N, 3, 2));     // a specific pattern
  patterns.push_back(test_utils::make_pattern(N, 7, 3));     // another
  patterns.push_back(test_utils::make_pattern(N, 15, 5));    // yet another

  for (size_t p = 0; p < patterns.size(); ++p) {
    auto m = this->make_mask_from_pattern(t, patterns[p]);
    auto m_lo = lower(t, m);
    auto m_hi = upper(t, m);
    auto m_rebuilt = concat(t, m_lo, m_hi);

    for (nint_t i = 0; i < N; ++i) {
      EXPECT_EQ(get(t, m, i), get(t, m_rebuilt, i))
          << "Roundtrip pattern=" << p << " i=" << i;
    }
  }

  // Also test all-true and all-false roundtrip
  auto v = loadu(t, this->data_);
  auto m_true = cmpeq(v, v);
  auto m_false = cmpne(v, v);

  for (auto* pm : {&m_true, &m_false}) {
    auto m_lo = lower(t, *pm);
    auto m_hi = upper(t, *pm);
    auto m_rebuilt = concat(t, m_lo, m_hi);
    for (nint_t i = 0; i < N; ++i) {
      EXPECT_EQ(get(t, *pm, i), get(t, m_rebuilt, i)) << "Roundtrip all-same i=" << i;
    }
  }
}


// ============================================================================
// Multi-Word Mask Tests (NW=2, NW=4)
// ============================================================================

TYPED_TEST(VecMaskTest, MaskLowerUpperMultiWord) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t N2 = this->multi2_size;

  if (N2 < 4) GTEST_SKIP() << "Multi-word requires at least 4 elements";

  // Create pattern for multi-word mask
  auto pat = test_utils::make_pattern(N2, 3, 1);
  auto m = this->make_mask_from_pattern(t2, pat);

  auto m_lo = lower(t2, m);
  auto m_hi = upper(t2, m);

  auto expected_lo = test_utils::scalar_mask_lower(pat);
  auto expected_hi = test_utils::scalar_mask_upper(pat);

  Half<ScalableTag<T, 1>> th;
  for (nint_t i = 0; i < N2 / 2; ++i) {
    EXPECT_EQ(expected_lo[(size_t)i], get(th, m_lo, i)) << "MultiWordLower i=" << i;
    EXPECT_EQ(expected_hi[(size_t)i], get(th, m_hi, i)) << "MultiWordUpper i=" << i;
  }

  // Roundtrip
  auto m_rebuilt = concat(t2, m_lo, m_hi);
  for (nint_t i = 0; i < N2; ++i) {
    EXPECT_EQ(get(t2, m, i), get(t2, m_rebuilt, i)) << "MultiWordRoundtrip i=" << i;
  }
}

TYPED_TEST(VecMaskTest, MaskBitOpsMultiWord) {
  using T = typename TestFixture::Type;
  auto& t2 = this->t2;
  nint_t N2 = this->multi2_size;

  if (N2 < 4) GTEST_SKIP() << "Multi-word requires at least 4 elements";

  auto pat_a = test_utils::make_pattern(N2, 1, 1);
  auto pat_b = test_utils::make_pattern(N2, 3, 0);

  auto ma = this->make_mask_from_pattern(t2, pat_a);
  auto mb = this->make_mask_from_pattern(t2, pat_b);

  auto mr_and = bit_and(t2, ma, mb);
  auto mr_or  = bit_or(t2, ma, mb);
  auto mr_xor = bit_xor(t2, ma, mb);
  auto mr_andnot = bit_andnot(t2, ma, mb);

  auto expected_and = test_utils::scalar_mask_and(pat_a, pat_b);
  auto expected_or  = test_utils::scalar_mask_or(pat_a, pat_b);
  auto expected_xor = test_utils::scalar_mask_xor(pat_a, pat_b);
  auto expected_andnot = test_utils::scalar_mask_andnot(pat_a, pat_b);

  for (nint_t i = 0; i < N2; ++i) {
    EXPECT_EQ(expected_and[(size_t)i], get(t2, mr_and, i))    << "MW AND i=" << i;
    EXPECT_EQ(expected_or[(size_t)i],  get(t2, mr_or, i))     << "MW OR i=" << i;
    EXPECT_EQ(expected_xor[(size_t)i], get(t2, mr_xor, i))    << "MW XOR i=" << i;
    EXPECT_EQ(expected_andnot[(size_t)i], get(t2, mr_andnot, i)) << "MW ANDNOT i=" << i;
  }

  // bit_not
  auto mr_not = bit_not(t2, ma);
  auto expected_not = test_utils::scalar_mask_not(pat_a);
  for (nint_t i = 0; i < N2; ++i) {
    EXPECT_EQ(expected_not[(size_t)i], get(t2, mr_not, i)) << "MW NOT i=" << i;
  }
}
