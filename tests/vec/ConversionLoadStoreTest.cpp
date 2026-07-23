#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "TestUtils.h"
#include "vecops/vec/Vec.h"
#include "vecops/util/ScalarConvert.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

template <typename T>
T test_value(nint_t i) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t> ||
                std::is_same_v<T, vecops::float16_t> ||
                std::is_floating_point_v<T>) {
    return T(float((i % 9) - 4) * 0.625f);
  } else if constexpr (std::is_unsigned_v<T>) {
    return T((i * 37 + 5) % 113);
  } else {
    return T((i * 29) % 101 - 50);
  }
}

template <typename InTag, typename OutTag>
void check_conversion_load_store() {
  using Ti = TypeOf<InTag>;
  using To = TypeOf<OutTag>;
  constexpr OutTag to{};
  constexpr InTag ti{};
  const nint_t n = size(to);
  ASSERT_EQ(n, size(ti));

  std::vector<Ti> input(static_cast<size_t>(n + 2));
  for (nint_t i = 0; i < n; ++i) input[static_cast<size_t>(i + 1)] = test_value<Ti>(i);
  const Ti* input_p = input.data() + 1;

  const auto expected_load = vec::xconvert(to, vec::loadu(ti, input_p));
  const auto actual_load = vec::xconvert_loadu(to, input_p);
  Vec<OutTag> named_load;
  if constexpr (sizeof(Ti) < sizeof(To)) {
    named_load = vec::promote_loadu(to, input_p);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    named_load = vec::demote_loadu(to, input_p);
  } else {
    named_load = vec::convert_loadu(to, input_p);
  }
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_near(get(to, expected_load, i), get(to, actual_load, i)))
        << "xconvert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(get(to, expected_load, i), get(to, named_load, i)))
        << "named load lane=" << i;
  }

  const nint_t active = n > 1 ? n - 1 : 0;
  const auto mask = vec::mwhilelt(to, 0, active);
  const To default_value = test_value<To>(17);
  const auto default_v = vec::fill(to, default_value);
  const auto masked_load = vec::xconvert_loadu(to, input_p, mask, default_v);
  const auto zero_masked_load = vec::xconvert_loadu(to, input_p, mask);
  Vec<OutTag> named_masked_load;
  Vec<OutTag> named_zero_masked_load;
  if constexpr (sizeof(Ti) < sizeof(To)) {
    named_masked_load = vec::promote_loadu(to, input_p, mask, default_v);
    named_zero_masked_load = vec::promote_loadu(to, input_p, mask);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    named_masked_load = vec::demote_loadu(to, input_p, mask, default_v);
    named_zero_masked_load = vec::demote_loadu(to, input_p, mask);
  } else {
    named_masked_load = vec::convert_loadu(to, input_p, mask, default_v);
    named_zero_masked_load = vec::convert_loadu(to, input_p, mask);
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = i < active ? get(to, expected_load, i) : default_value;
    EXPECT_TRUE(test_utils::values_near(expected, get(to, masked_load, i)))
        << "masked xconvert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(expected, get(to, named_masked_load, i)))
        << "masked named load lane=" << i;
    const To zero_expected = i < active ? get(to, expected_load, i) : To{};
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, zero_masked_load, i)))
        << "zero masked xconvert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, named_zero_masked_load, i)))
        << "zero masked named load lane=" << i;
  }

  std::vector<To> output(static_cast<size_t>(n + 2), test_value<To>(31));
  std::vector<To> named_output(static_cast<size_t>(n + 2), test_value<To>(31));
  const auto input_v = vec::loadu(ti, input_p);
  vec::xconvert_storeu(ti, output.data() + 1, input_v);
  if constexpr (sizeof(Ti) < sizeof(To)) {
    vec::promote_storeu(ti, named_output.data() + 1, input_v);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::demote_storeu(ti, named_output.data() + 1, input_v);
  } else {
    vec::convert_storeu(ti, named_output.data() + 1, input_v);
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = vecops::convert<To>(input_p[i]);
    EXPECT_TRUE(test_utils::values_near(expected, output[static_cast<size_t>(i + 1)]))
        << "xconvert store lane=" << i;
    EXPECT_TRUE(test_utils::values_near(expected, named_output[static_cast<size_t>(i + 1)]))
        << "named store lane=" << i;
  }

  const auto input_mask = vec::mwhilelt(ti, 0, active);
  const To sentinel = test_value<To>(23);
  std::fill(output.begin(), output.end(), sentinel);
  std::fill(named_output.begin(), named_output.end(), sentinel);
  vec::xconvert_storeu(ti, output.data() + 1, input_mask, input_v);
  if constexpr (sizeof(Ti) < sizeof(To)) {
    vec::promote_storeu(ti, named_output.data() + 1, input_mask, input_v);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::demote_storeu(ti, named_output.data() + 1, input_mask, input_v);
  } else {
    vec::convert_storeu(ti, named_output.data() + 1, input_mask, input_v);
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = i < active ? vecops::convert<To>(input_p[i]) : sentinel;
    EXPECT_TRUE(test_utils::values_near(expected, output[static_cast<size_t>(i + 1)]))
        << "masked xconvert store lane=" << i;
    EXPECT_TRUE(test_utils::values_near(expected, named_output[static_cast<size_t>(i + 1)]))
        << "masked named store lane=" << i;
  }
  EXPECT_TRUE(test_utils::values_near(sentinel, output.front()));
  EXPECT_TRUE(test_utils::values_near(sentinel, output.back()));
  EXPECT_TRUE(test_utils::values_near(sentinel, named_output.front()));
  EXPECT_TRUE(test_utils::values_near(sentinel, named_output.back()));
}

template <typename Ti, typename To, int POW2_In>
void check_promote_load_store() {
  using InTag = ScalableTag<Ti, POW2_In>;
  using OutTag = Rebind<To, InTag>;
  check_conversion_load_store<InTag, OutTag>();
}

template <typename Ti, typename To, int POW2_Out>
void check_demote_load_store() {
  using OutTag = ScalableTag<To, POW2_Out>;
  using InTag = Rebind<Ti, OutTag>;
  check_conversion_load_store<InTag, OutTag>();
}

template <typename Ti, typename To, int POW2>
void check_convert_load_store() {
  using InTag = ScalableTag<Ti, POW2>;
  using OutTag = ScalableTag<To, POW2>;
  check_conversion_load_store<InTag, OutTag>();
}

#define CHECK_PROMOTE_SHIFT1(Ti, To) \
  check_promote_load_store<Ti, To, 0>(); \
  check_promote_load_store<Ti, To, -1>()

#define CHECK_DEMOTE_SHIFT1(Ti, To) \
  check_demote_load_store<Ti, To, 0>(); \
  check_demote_load_store<Ti, To, -1>()

#define CHECK_PROMOTE_SHIFT2(Ti, To) \
  check_promote_load_store<Ti, To, 0>(); \
  check_promote_load_store<Ti, To, -1>(); \
  check_promote_load_store<Ti, To, -2>()

#define CHECK_DEMOTE_SHIFT2(Ti, To) \
  check_demote_load_store<Ti, To, 0>(); \
  check_demote_load_store<Ti, To, -1>(); \
  check_demote_load_store<Ti, To, -2>()

#if VEC_MAX_POW >= 3
#define CHECK_PROMOTE_SHIFT3(Ti, To) \
  check_promote_load_store<Ti, To, 0>(); \
  check_promote_load_store<Ti, To, -1>(); \
  check_promote_load_store<Ti, To, -2>(); \
  check_promote_load_store<Ti, To, -3>()
#define CHECK_DEMOTE_SHIFT3(Ti, To) \
  check_demote_load_store<Ti, To, 0>(); \
  check_demote_load_store<Ti, To, -1>(); \
  check_demote_load_store<Ti, To, -2>(); \
  check_demote_load_store<Ti, To, -3>()
#else
#define CHECK_PROMOTE_SHIFT3(Ti, To) \
  check_promote_load_store<Ti, To, -1>(); \
  check_promote_load_store<Ti, To, -2>(); \
  check_promote_load_store<Ti, To, -3>()
#define CHECK_DEMOTE_SHIFT3(Ti, To) \
  check_demote_load_store<Ti, To, -1>(); \
  check_demote_load_store<Ti, To, -2>(); \
  check_demote_load_store<Ti, To, -3>()
#endif

#define CHECK_CONVERT_POWS(Ti, To) \
  check_convert_load_store<Ti, To, 0>(); \
  check_convert_load_store<Ti, To, 1>(); \
  check_convert_load_store<Ti, To, 2>()

template <typename Ti, typename To>
void check_integer_demote_saturation() {
  using InputTag = ScalableTag<Ti>;
  constexpr InputTag ti{};
  const nint_t n = size(ti);

  const Ti low = [] {
    if constexpr (std::is_signed_v<Ti>) {
      return std::numeric_limits<Ti>::lowest();
    } else {
      return Ti{0};
    }
  }();
  const Ti high = std::numeric_limits<Ti>::max();

  for (const Ti value : {low, high}) {
    const auto v = vec::fill(ti, value);
    const To sentinel = To{7};
    std::vector<To> output(static_cast<size_t>(n + 2), sentinel);
    vec::demote_storeu(ti, output.data() + 1, v);
    const To expected = vecops::convert<To>(value);
    for (nint_t i = 0; i < n; ++i) {
      EXPECT_EQ(expected, output[static_cast<size_t>(i + 1)]);
    }
    EXPECT_EQ(sentinel, output.front());
    EXPECT_EQ(sentinel, output.back());

    std::fill(output.begin(), output.end(), sentinel);
    const nint_t active = n > 1 ? n - 1 : 0;
    const auto mask = vec::mwhilelt(ti, 0, active);
    vec::demote_storeu(ti, output.data() + 1, mask, v);
    for (nint_t i = 0; i < n; ++i) {
      EXPECT_EQ(i < active ? expected : sentinel,
                output[static_cast<size_t>(i + 1)]);
    }
    EXPECT_EQ(sentinel, output.front());
    EXPECT_EQ(sentinel, output.back());
  }
}

#if defined(__unix__) || defined(__APPLE__)
template <typename Ti, typename To>
void check_masked_guard_page() {
  const long page_size_long = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size_long, 0);
  const size_t page_size = static_cast<size_t>(page_size_long);
  void* mapping = ::mmap(
      nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(
      ::mprotect(static_cast<char*>(mapping) + page_size,
                 page_size, PROT_NONE),
      0);

  using OutTag = ScalableTag<To>;
  constexpr OutTag to{};
  const nint_t load_active = std::max<nint_t>(1, size(to) / 2);
  auto* load_p = reinterpret_cast<Ti*>(
      static_cast<char*>(mapping) + page_size -
      static_cast<size_t>(load_active) * sizeof(Ti));
  for (nint_t i = 0; i < load_active; ++i) load_p[i] = test_value<Ti>(i);
  const auto load_mask = vec::mwhilelt(to, 0, load_active);
  const To default_value = test_value<To>(29);
  const auto default_v = vec::fill(to, default_value);

  Vec<OutTag> loaded;
  Vec<OutTag> zero_loaded;
  if constexpr (sizeof(Ti) < sizeof(To)) {
    loaded = vec::promote_loadu(to, load_p, load_mask, default_v);
    zero_loaded = vec::promote_loadu(to, load_p, load_mask);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    loaded = vec::demote_loadu(to, load_p, load_mask, default_v);
    zero_loaded = vec::demote_loadu(to, load_p, load_mask);
  } else {
    loaded = vec::convert_loadu(to, load_p, load_mask, default_v);
    zero_loaded = vec::convert_loadu(to, load_p, load_mask);
  }
  const auto zero_xloaded = vec::xconvert_loadu(to, load_p, load_mask);
  for (nint_t i = 0; i < size(to); ++i) {
    const To expected = i < load_active
        ? vecops::convert<To>(load_p[i]) : default_value;
    EXPECT_TRUE(test_utils::values_near(expected, get(to, loaded, i)))
        << "guarded masked load lane=" << i;
    const To zero_expected = i < load_active
        ? vecops::convert<To>(load_p[i]) : To{};
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, zero_loaded, i)))
        << "guarded zero masked named load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, zero_xloaded, i)))
        << "guarded zero masked xconvert load lane=" << i;
  }

  using InTag = ScalableTag<Ti>;
  constexpr InTag ti{};
  const nint_t store_active = std::max<nint_t>(1, size(ti) / 2);
  auto* store_p = reinterpret_cast<To*>(
      static_cast<char*>(mapping) + page_size -
      static_cast<size_t>(store_active) * sizeof(To));
  const To sentinel = test_value<To>(31);
  for (nint_t i = 0; i < store_active; ++i) store_p[i] = sentinel;
  const auto store_mask = vec::mwhilelt(ti, 0, store_active);
  const Ti input_value = test_value<Ti>(7);
  const auto input_v = vec::fill(ti, input_value);
  if constexpr (sizeof(Ti) < sizeof(To)) {
    vec::promote_storeu(ti, store_p, store_mask, input_v);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::demote_storeu(ti, store_p, store_mask, input_v);
  } else {
    vec::convert_storeu(ti, store_p, store_mask, input_v);
  }
  const To expected_store = vecops::convert<To>(input_value);
  for (nint_t i = 0; i < store_active; ++i) {
    EXPECT_TRUE(test_utils::values_near(expected_store, store_p[i]))
        << "guarded masked store lane=" << i;
  }

  EXPECT_EQ(::munmap(mapping, 2 * page_size), 0);
}
#endif

template <typename T>
void check_ordinary_zero_loads() {
  using Tag = ScalableTag<T>;
  constexpr Tag t{};
  const nint_t n = size(t);
  const nint_t active = std::max<nint_t>(1, n / 2);

  const size_t bytes = static_cast<size_t>(n) * sizeof(T);
  std::vector<unsigned char> storage(bytes + DEFAULT_ALIGNMENT);
  const auto raw = reinterpret_cast<uintptr_t>(storage.data());
  auto* aligned_p = reinterpret_cast<T*>(
      (raw + DEFAULT_ALIGNMENT - 1) & ~(uintptr_t(DEFAULT_ALIGNMENT - 1)));
  for (nint_t i = 0; i < n; ++i) aligned_p[i] = test_value<T>(i);

  const auto mask = vec::mwhilelt(t, 0, active);
  const auto aligned_masked = vec::load(t, aligned_p, mask);
  const auto aligned_count = vec::load(t, aligned_p, active);
  for (nint_t i = 0; i < n; ++i) {
    const T expected = i < active ? aligned_p[i] : T{};
    EXPECT_TRUE(test_utils::values_near(expected, get(t, aligned_masked, i)));
    EXPECT_TRUE(test_utils::values_near(expected, get(t, aligned_count, i)));
  }

#if defined(__unix__) || defined(__APPLE__)
  const long page_size_long = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size_long, 0);
  const size_t page_size = static_cast<size_t>(page_size_long);
  void* mapping = ::mmap(
      nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(::mprotect(static_cast<char*>(mapping) + page_size,
                       page_size, PROT_NONE), 0);
  auto* guarded_p = reinterpret_cast<T*>(
      static_cast<char*>(mapping) + page_size -
      static_cast<size_t>(active) * sizeof(T));
  for (nint_t i = 0; i < active; ++i) guarded_p[i] = test_value<T>(i);

  const auto unaligned_masked = vec::loadu(t, guarded_p, mask);
  const auto unaligned_count = vec::loadu(t, guarded_p, active);
  for (nint_t i = 0; i < n; ++i) {
    const T expected = i < active ? guarded_p[i] : T{};
    EXPECT_TRUE(test_utils::values_near(expected, get(t, unaligned_masked, i)));
    EXPECT_TRUE(test_utils::values_near(expected, get(t, unaligned_count, i)));
  }
  EXPECT_EQ(::munmap(mapping, 2 * page_size), 0);
#endif
}

} // namespace

TEST(VecConversionLoadStoreTest, IntegerPromote) {
  CHECK_PROMOTE_SHIFT1(int8_t, int16_t);
  CHECK_PROMOTE_SHIFT2(int8_t, uint32_t);
  CHECK_PROMOTE_SHIFT2(uint8_t, int32_t);
  CHECK_PROMOTE_SHIFT2(int16_t, int64_t);
  CHECK_PROMOTE_SHIFT1(uint16_t, uint32_t);
  CHECK_PROMOTE_SHIFT3(int8_t, int64_t);
}

TEST(VecConversionLoadStoreTest, IntegerDemoteSaturates) {
  CHECK_DEMOTE_SHIFT1(int16_t, int8_t);
  CHECK_DEMOTE_SHIFT2(uint32_t, uint8_t);
  CHECK_DEMOTE_SHIFT2(int64_t, int16_t);
  CHECK_DEMOTE_SHIFT1(uint64_t, int32_t);
  CHECK_DEMOTE_SHIFT3(int64_t, int8_t);
  check_integer_demote_saturation<int32_t, int8_t>();
  check_integer_demote_saturation<int32_t, uint8_t>();
  check_integer_demote_saturation<uint32_t, int8_t>();
  check_integer_demote_saturation<uint64_t, uint16_t>();
}

TEST(VecConversionLoadStoreTest, EqualWidthConvert) {
  CHECK_CONVERT_POWS(int32_t, float32_t);
  CHECK_CONVERT_POWS(float32_t, int32_t);
  CHECK_CONVERT_POWS(int16_t, uint16_t);
  CHECK_CONVERT_POWS(float64_t, uint64_t);
}

TEST(VecConversionLoadStoreTest, FloatingPointWidths) {
  CHECK_PROMOTE_SHIFT1(vecops::float16_t, float32_t);
  CHECK_DEMOTE_SHIFT1(float32_t, vecops::float16_t);
  CHECK_PROMOTE_SHIFT1(float32_t, float64_t);
  CHECK_DEMOTE_SHIFT1(float64_t, float32_t);
}

TEST(VecConversionLoadStoreTest, OrdinaryInt16MaskedLoadsZeroInactiveLanes) {
  check_ordinary_zero_loads<int16_t>();
}

TEST(VecConversionLoadStoreTest, OrdinaryInt32MaskedLoadsZeroInactiveLanes) {
  check_ordinary_zero_loads<int32_t>();
}

TEST(VecConversionLoadStoreTest, OrdinaryFloat32MaskedLoadsZeroInactiveLanes) {
  check_ordinary_zero_loads<float32_t>();
}

#if defined(__unix__) || defined(__APPLE__)
TEST(VecConversionLoadStoreTest, MaskedPromoteDoesNotCrossGuardPage) {
  check_masked_guard_page<int16_t, int32_t>();
}

TEST(VecConversionLoadStoreTest, MaskedDemoteDoesNotCrossGuardPage) {
  check_masked_guard_page<int32_t, int16_t>();
}

TEST(VecConversionLoadStoreTest, MaskedConvertDoesNotCrossGuardPage) {
  check_masked_guard_page<int32_t, float32_t>();
}
#endif

#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
TEST(VecConversionLoadStoreTest, Bfloat16Routes) {
  CHECK_PROMOTE_SHIFT1(vecops::bfloat16_t, float32_t);
  CHECK_DEMOTE_SHIFT1(float32_t, vecops::bfloat16_t);
  CHECK_CONVERT_POWS(vecops::bfloat16_t, int16_t);
  CHECK_CONVERT_POWS(int16_t, vecops::bfloat16_t);
  CHECK_PROMOTE_SHIFT2(vecops::bfloat16_t, float64_t);
  CHECK_DEMOTE_SHIFT2(float64_t, vecops::bfloat16_t);
#if defined(__unix__) || defined(__APPLE__)
  check_masked_guard_page<vecops::bfloat16_t, float32_t>();
  check_masked_guard_page<float32_t, vecops::bfloat16_t>();
#endif
}
#endif
