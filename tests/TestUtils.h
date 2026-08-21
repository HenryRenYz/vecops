#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <type_traits>
#include <vector>

#include "vecops/vec/Vec.h"

namespace test_utils {

template <typename T>
constexpr T get_test_value(int idx) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    return static_cast<vecops::bfloat16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    return static_cast<vecops::float16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, vecops::float32_t>) {
    return static_cast<vecops::float32_t>(idx * 1.5f + 0.5f);
  } else if constexpr (std::is_same_v<T, vecops::float64_t>) {
    return static_cast<vecops::float64_t>(idx * 1.5 + 0.5);
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
constexpr T get_test_value_b(int idx) {
  return get_test_value<T>(idx + 50);
}

template <typename T>
::testing::AssertionResult values_equal(T expected, T actual) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (e == a) return ::testing::AssertionSuccess();
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (e == a) return ::testing::AssertionSuccess();
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float32_t>) {
    if (expected == actual) return ::testing::AssertionSuccess();
    if (std::abs(expected - actual) < 1e-5f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, vecops::float64_t>) {
    if (expected == actual) return ::testing::AssertionSuccess();
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
::testing::AssertionResult values_near(T expected, T actual, double tolerance = 0.01) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) <= std::max(std::abs(e), std::abs(a)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) <= std::max(std::abs(e), std::abs(a)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, vecops::float32_t>) {
    if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, vecops::float64_t>) {
    if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else {
    if (expected == actual) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
        << "Expected " << static_cast<long long>(expected)
        << ", got " << static_cast<long long>(actual);
  }
}

template <typename T>
T* alloc_aligned(
    size_t count, size_t alignment = vecops::vec::DEFAULT_ALIGNMENT) {
  size_t bytes = count * sizeof(T);
  size_t aligned_bytes =
      ((bytes + alignment - 1) / alignment) * alignment;
  void* ptr = std::aligned_alloc(alignment, aligned_bytes);
  if (ptr == nullptr) throw std::bad_alloc();
  return static_cast<T*>(ptr);
}

using AllVecDataTypes = ::testing::Types<
    vecops::float32_t, vecops::float64_t, int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t, vecops::float16_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

using UnsignedIntTypes = ::testing::Types<
    uint8_t, uint16_t, uint32_t, uint64_t
>;

template <typename T>
constexpr vecops::nint_t lane_size() {
  return 16 / static_cast<vecops::nint_t>(sizeof(T));
}

template <typename T> struct ShuffleIndex { using type = T; };
template <> struct ShuffleIndex<vecops::float32_t> { using type = int32_t; };
template <> struct ShuffleIndex<vecops::float64_t> { using type = int64_t; };
template <> struct ShuffleIndex<uint32_t> { using type = int32_t; };
template <> struct ShuffleIndex<uint64_t> { using type = int64_t; };
template <> struct ShuffleIndex<uint8_t> { using type = int8_t; };
template <> struct ShuffleIndex<uint16_t> { using type = int16_t; };
template <> struct ShuffleIndex<vecops::float16_t> { using type = int16_t; };
template <> struct ShuffleIndex<vecops::bfloat16_t> { using type = int16_t; };
template <typename T> using shuffle_idx_t = typename ShuffleIndex<T>::type;

template <typename T>
void fill_seq(T* data, vecops::nint_t n) {
  for (vecops::nint_t i = 0; i < n; ++i)
    data[i] = static_cast<T>(i + 1);
}

template <typename I>
void fill_local_identity(I* idx, vecops::nint_t n) {
  constexpr vecops::nint_t M = 16 / sizeof(I);
  for (vecops::nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % M);
}

template <typename I>
void fill_shuf_identity(I* idx, vecops::nint_t n, vecops::nint_t word_size) {
  for (vecops::nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % word_size);
}

using Types4 = ::testing::Types<vecops::float32_t, int32_t, uint32_t>;
using Types2 = ::testing::Types<vecops::float64_t, int64_t, uint64_t>;
using Types8 = ::testing::Types<int16_t, uint16_t, vecops::float16_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;
using Types16 = ::testing::Types<int8_t, uint8_t>;

using ShufTypes = ::testing::Types<
    vecops::float32_t, int32_t, uint32_t,
    vecops::float64_t, int64_t, uint64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    vecops::float16_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

using ShuffleAllTypes = ::testing::Types<
    vecops::float16_t,
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    vecops::bfloat16_t,
#endif
    vecops::float32_t, vecops::float64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t
>;

std::vector<bool> make_pattern(vecops::nint_t size, int mask_bits, int match_val);
std::vector<bool> scalar_mask_and(const std::vector<bool>& a, const std::vector<bool>& b);
std::vector<bool> scalar_mask_or(const std::vector<bool>& a, const std::vector<bool>& b);
std::vector<bool> scalar_mask_xor(const std::vector<bool>& a, const std::vector<bool>& b);
std::vector<bool> scalar_mask_andnot(const std::vector<bool>& a, const std::vector<bool>& b);
std::vector<bool> scalar_mask_not(const std::vector<bool>& a);
std::vector<bool> scalar_mask_lower(const std::vector<bool>& a);
std::vector<bool> scalar_mask_upper(const std::vector<bool>& a);
std::vector<bool> scalar_mask_concat(const std::vector<bool>& lo, const std::vector<bool>& hi);

template <typename Tag>
vecops::vec::Mask<Tag> make_mask(Tag tt, const std::vector<bool>& pattern) {
  auto m = vecops::vec::mfalse(tt);
  for (vecops::nint_t i = 0; i < size(tt); ++i) {
    if (pattern[(size_t)i]) {
      m = vecops::vec::set(tt, m, i, true);
    }
  }
  return m;
}

template <typename Tag>
void verify_mask_pattern(Tag tt, vecops::vec::Mask<Tag> m, const std::vector<bool>& expected, int line) {
  for (vecops::nint_t i = 0; i < size(tt); ++i) {
    bool actual = get(tt, m, i);
    EXPECT_EQ(expected[(size_t)i], actual)
        << "Mismatch at i=" << i << " line=" << line;
  }
}

template <typename T> T scalar_add(T a, T b) { return a + b; }
template <typename T> T scalar_sub(T a, T b) { return a - b; }
template <typename T> T scalar_mul(T a, T b) { return a * b; }
template <typename T> T scalar_neg(T a) { return -a; }
template <typename T> T scalar_fmadd(T a, T b, T c) { return scalar_add(scalar_mul(a, b), c); }
template <typename T> T scalar_fmsub(T a, T b, T c) { return scalar_sub(scalar_mul(a, b), c); }
template <typename T> T scalar_fnmadd(T a, T b, T c) { return scalar_add(scalar_neg(scalar_mul(a, b)), c); }
template <typename T> T scalar_fnmsub(T a, T b, T c) { return scalar_sub(scalar_neg(scalar_mul(a, b)), c); }

template <typename T>
T scalar_div(T a, T b) {
  if constexpr (vecops::is_float_v<T>) return a / b;
  else return static_cast<T>(0);
}

template <typename T>
T scalar_max(T a, T b) {
  if constexpr (vecops::is_float_v<T>) return std::max(a, b);
  else return (a > b) ? a : b;
}

template <typename T>
T scalar_min(T a, T b) {
  if constexpr (vecops::is_float_v<T>) return std::min(a, b);
  else return (a < b) ? a : b;
}

template <typename T>
T scalar_abs(T a) {
  if constexpr (vecops::is_float_v<T>) return std::fabs(a);
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
  if constexpr (vecops::is_float_v<T>) return std::sqrt((vecops::float64_t)a);
  else return static_cast<T>(0);
}

} // namespace test_utils
