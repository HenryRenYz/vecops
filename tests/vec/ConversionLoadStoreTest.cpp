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

  const auto expected_load = vec::convert(to, vec::load(ti, input_p));
  const auto actual_load = vec::load_convert(to, input_p);
  Vec<OutTag> named_load;
  if constexpr (sizeof(Ti) < sizeof(To)) {
    named_load = vec::load_convert(to, input_p);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    named_load = vec::load_convert(to, input_p);
  } else {
    named_load = vec::load_convert(to, input_p);
  }
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_near(get(to, expected_load, i), get(to, actual_load, i)))
        << "convert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(get(to, expected_load, i), get(to, named_load, i)))
        << "named load lane=" << i;
  }

  const nint_t active = n > 1 ? n - 1 : 0;
  const auto mask = vec::mwhilelt(to, 0, active);
  const To default_value = test_value<To>(17);
  const auto default_v = vec::fill(to, default_value);
  const auto masked_load = vec::load_convert(
      to, input_p, opt::masked(mask), opt::merge(default_v));
  const auto zero_masked_load = vec::load_convert(
      to, input_p, opt::masked(mask));
  Vec<OutTag> named_masked_load;
  Vec<OutTag> named_zero_masked_load;
  if constexpr (sizeof(Ti) < sizeof(To)) {
    named_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask), opt::merge(default_v));
    named_zero_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask));
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    named_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask), opt::merge(default_v));
    named_zero_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask));
  } else {
    named_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask), opt::merge(default_v));
    named_zero_masked_load = vec::load_convert(
        to, input_p, opt::masked(mask));
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = i < active ? get(to, expected_load, i) : default_value;
    EXPECT_TRUE(test_utils::values_near(expected, get(to, masked_load, i)))
        << "masked convert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(expected, get(to, named_masked_load, i)))
        << "masked named load lane=" << i;
    const To zero_expected = i < active ? get(to, expected_load, i) : To{};
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, zero_masked_load, i)))
        << "zero masked convert load lane=" << i;
    EXPECT_TRUE(test_utils::values_near(zero_expected, get(to, named_zero_masked_load, i)))
        << "zero masked named load lane=" << i;
  }

  std::vector<To> output(static_cast<size_t>(n + 2), test_value<To>(31));
  std::vector<To> named_output(static_cast<size_t>(n + 2), test_value<To>(31));
  const auto input_v = vec::load(ti, input_p);
  vec::store_convert(ti, output.data() + 1, input_v);
  if constexpr (sizeof(Ti) < sizeof(To)) {
    vec::store_convert(ti, named_output.data() + 1, input_v);
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::store_convert(ti, named_output.data() + 1, input_v);
  } else {
    vec::store_convert(ti, named_output.data() + 1, input_v);
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = vecops::convert<To>(input_p[i]);
    EXPECT_TRUE(test_utils::values_near(expected, output[static_cast<size_t>(i + 1)]))
        << "convert store lane=" << i;
    EXPECT_TRUE(test_utils::values_near(expected, named_output[static_cast<size_t>(i + 1)]))
        << "named store lane=" << i;
  }

  const auto input_mask = vec::mwhilelt(ti, 0, active);
  const To sentinel = test_value<To>(23);
  std::fill(output.begin(), output.end(), sentinel);
  std::fill(named_output.begin(), named_output.end(), sentinel);
  vec::store_convert(
      ti, output.data() + 1, input_v, opt::masked(input_mask));
  if constexpr (sizeof(Ti) < sizeof(To)) {
    vec::store_convert(
        ti, named_output.data() + 1, input_v, opt::masked(input_mask));
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::store_convert(
        ti, named_output.data() + 1, input_v, opt::masked(input_mask));
  } else {
    vec::store_convert(
        ti, named_output.data() + 1, input_v, opt::masked(input_mask));
  }
  for (nint_t i = 0; i < n; ++i) {
    const To expected = i < active ? vecops::convert<To>(input_p[i]) : sentinel;
    EXPECT_TRUE(test_utils::values_near(expected, output[static_cast<size_t>(i + 1)]))
        << "masked convert store lane=" << i;
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
    vec::store_convert(ti, output.data() + 1, v);
    const To expected = vecops::convert<To>(value);
    for (nint_t i = 0; i < n; ++i) {
      EXPECT_EQ(expected, output[static_cast<size_t>(i + 1)]);
    }
    EXPECT_EQ(sentinel, output.front());
    EXPECT_EQ(sentinel, output.back());

    std::fill(output.begin(), output.end(), sentinel);
    const nint_t active = n > 1 ? n - 1 : 0;
    const auto mask = vec::mwhilelt(ti, 0, active);
    vec::store_convert(
        ti, output.data() + 1, v, opt::masked(mask));
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
    loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask), opt::merge(default_v));
    zero_loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask));
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask), opt::merge(default_v));
    zero_loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask));
  } else {
    loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask), opt::merge(default_v));
    zero_loaded = vec::load_convert(
        to, load_p, opt::masked(load_mask));
  }
  const auto zero_xloaded = vec::load_convert(
      to, load_p, opt::masked(load_mask));
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
        << "guarded zero masked convert load lane=" << i;
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
    vec::store_convert(ti, store_p, input_v, opt::masked(store_mask));
  } else if constexpr (sizeof(Ti) > sizeof(To)) {
    vec::store_convert(ti, store_p, input_v, opt::masked(store_mask));
  } else {
    vec::store_convert(ti, store_p, input_v, opt::masked(store_mask));
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
  const auto alignment = static_cast<uintptr_t>(memory_alignment(t));
  std::vector<unsigned char> storage(bytes + alignment);
  const auto raw = reinterpret_cast<uintptr_t>(storage.data());
  auto* aligned_p = reinterpret_cast<T*>(
      (raw + alignment - 1) & ~(alignment - 1));
  for (nint_t i = 0; i < n; ++i) aligned_p[i] = test_value<T>(i);

  const auto mask = vec::mwhilelt(t, 0, active);
  const auto aligned_masked = vec::load(
      t, aligned_p, mem::aligned, opt::masked(mask));
  const auto aligned_count = vec::load(
      t, aligned_p, mem::aligned, opt::first(active));
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

  const auto unaligned_masked = vec::load(
      t, guarded_p, opt::masked(mask));
  const auto unaligned_count = vec::load(
      t, guarded_p, opt::first(active));
  for (nint_t i = 0; i < n; ++i) {
    const T expected = i < active ? guarded_p[i] : T{};
    EXPECT_TRUE(test_utils::values_near(expected, get(t, unaligned_masked, i)));
    EXPECT_TRUE(test_utils::values_near(expected, get(t, unaligned_count, i)));
  }
  EXPECT_EQ(::munmap(mapping, 2 * page_size), 0);
#endif
}

template <typename Ti, typename To>
void check_integer_wrap() {
  SCOPED_TRACE(
      ::testing::Message()
      << "Ti=" << sizeof(Ti) << (std::is_signed_v<Ti> ? "s" : "u")
      << " To=" << sizeof(To) << (std::is_signed_v<To> ? "s" : "u"));
  using InTag = ScalableTag<Ti>;
  using OutTag = Rebind<To, InTag>;
  constexpr InTag ti{};
  constexpr OutTag to{};
  const nint_t n = size(ti);
  ASSERT_EQ(n, size(to));

  std::vector<Ti> input(static_cast<size_t>(n));
  const Ti values[] = {
      std::numeric_limits<Ti>::lowest(),
      std::numeric_limits<Ti>::max(),
      Ti{0}, Ti{1}, static_cast<Ti>(-1),
      static_cast<Ti>(0x7f), static_cast<Ti>(0x80),
      static_cast<Ti>(0xff), static_cast<Ti>(0x100),
      static_cast<Ti>(0xffff)};
  for (nint_t i = 0; i < n; ++i) {
    input[static_cast<size_t>(i)] =
        values[static_cast<size_t>(i) % std::size(values)];
  }

  const auto vi = load(ti, input.data());
  const auto wrapped = convert(to, vi, cvt::wrap);
  const auto truncated = convert(to, vi, cvt::truncate);
  std::vector<To> wrapped_memory(static_cast<size_t>(n));
  std::vector<To> truncated_memory(static_cast<size_t>(n));
  store(to, wrapped_memory.data(), wrapped);
  store(to, truncated_memory.data(), truncated);

  std::vector<To> fused_store(static_cast<size_t>(n));
  store_convert(ti, fused_store.data(), vi, cvt::wrap);
  const auto fused_load = load_convert(to, input.data(), cvt::truncate);

  for (nint_t i = 0; i < n; ++i) {
    const To expected =
        vecops::wrap_convert<To>(input[static_cast<size_t>(i)]);
    EXPECT_EQ(expected, wrapped_memory[static_cast<size_t>(i)]) << "lane=" << i;
    EXPECT_EQ(expected, truncated_memory[static_cast<size_t>(i)]) << "lane=" << i;
    EXPECT_EQ(expected, fused_store[static_cast<size_t>(i)]) << "lane=" << i;
    EXPECT_EQ(expected, get(to, fused_load, i)) << "lane=" << i;
  }
}

template <typename Ti, typename To, int Phase>
void check_lane_wrap() {
  using InTag = ScalableTag<Ti>;
  using OutTag = ViewAs<To, InTag>;
  constexpr InTag ti{};
  constexpr OutTag to{};
  constexpr int ratio = int(sizeof(Ti) / sizeof(To));
  const nint_t input_n = size(ti);
  const To fallback_value = static_cast<To>(0x5a);
  std::vector<Ti> input(static_cast<size_t>(input_n));
  for (nint_t i = 0; i < input_n; ++i) {
    input[static_cast<size_t>(i)] =
        static_cast<Ti>(i * 257 - 513);
  }
  const auto fallback = fill(to, fallback_value);
  const auto result = convert(
      to, load(ti, input.data()), cvt::lane<Phase>, cvt::truncate,
      opt::merge(fallback));
  for (nint_t i = 0; i < size(to); ++i) {
    const bool populated = i % ratio == Phase;
    const To expected = populated
        ? vecops::wrap_convert<To>(input[static_cast<size_t>(i / ratio)])
        : fallback_value;
    EXPECT_EQ(expected, get(to, result, i));
  }
}

template <typename To, typename Vi, typename... Options>
concept CanConvertWith = requires(To to, Vi vi, Options... options) {
  convert(to, vi, options...);
};

using ConstraintI32Tag = ScalableTag<int32_t>;
using ConstraintI16Tag = ViewAs<int16_t, ConstraintI32Tag>;
using ConstraintI8Tag = ViewAs<int8_t, ConstraintI32Tag>;
using ConstraintF32Tag = Rebind<float32_t, ConstraintI32Tag>;
using ConstraintI32Vec = Vec<ConstraintI32Tag>;
static_assert(CanConvertWith<
              ConstraintI16Tag, ConstraintI32Vec, cvt::lane_t<0>>);
static_assert(!CanConvertWith<
              ConstraintI8Tag, ConstraintI32Vec, cvt::lane_t<1>>);
static_assert(!CanConvertWith<
              ConstraintI16Tag, ConstraintI32Vec,
              cvt::ordered_t, cvt::unordered_t>);
static_assert(!CanConvertWith<
              ConstraintF32Tag, ConstraintI32Vec, cvt::wrap_t>);
static_assert(!CanConvertWith<
              ConstraintI32Tag, ConstraintI32Vec, cvt::wrap_t>);

template <typename T, typename V>
concept CanStoreWithMerge = requires(T t, TypeOf<T>* p, V v) {
  store(t, p, v, opt::merge(v));
};
static_assert(!CanStoreWithMerge<ConstraintI32Tag, ConstraintI32Vec>);

template <typename To, typename Ei, typename... Options>
concept CanLoadConvertWith =
    requires(To to, const Ei* p, Options... options) {
      load_convert(to, p, options...);
    };
static_assert(!CanLoadConvertWith<
              ConstraintI16Tag, int32_t, cvt::unordered_t>);

template <typename Ti, typename Eo, typename V, typename... Options>
concept CanStoreConvertWith =
    requires(Ti ti, Eo* p, V v, Options... options) {
      store_convert(ti, p, v, options...);
    };
static_assert(!CanStoreConvertWith<
              ConstraintI32Tag, int16_t, ConstraintI32Vec,
              opt::merge_t<ConstraintI32Vec>>);

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

TEST(VecConversionLoadStoreTest, IntegerWrapAndTruncateKeepLowBits) {
  check_integer_wrap<int16_t, int8_t>();
  check_integer_wrap<int16_t, uint8_t>();
  check_integer_wrap<uint16_t, int8_t>();
  check_integer_wrap<uint16_t, uint8_t>();
  check_integer_wrap<int32_t, int8_t>();
  check_integer_wrap<int32_t, uint16_t>();
  check_integer_wrap<uint32_t, int16_t>();
  check_integer_wrap<uint64_t, uint8_t>();
  check_lane_wrap<int16_t, int8_t, 0>();
  check_lane_wrap<int16_t, int8_t, 1>();
  check_lane_wrap<int32_t, uint8_t, 0>();
}

TEST(VecConversionLoadStoreTest, NonTemporalFallsBackToTemporal) {
  using T = ScalableTag<int32_t>;
  constexpr T t{};
  const nint_t n = size(t);
  std::vector<int32_t> input(static_cast<size_t>(n));
  std::vector<int32_t> output(static_cast<size_t>(n));
  for (nint_t i = 0; i < n; ++i) input[static_cast<size_t>(i)] = int32_t(i * 7 - 9);

  const auto temporal = load(t, input.data());
  const auto non_temporal = load(t, input.data(), mem::non_temporal);
  store(t, output.data(), non_temporal, mem::non_temporal);
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_EQ(get(t, temporal, i), get(t, non_temporal, i));
    EXPECT_EQ(input[static_cast<size_t>(i)], output[static_cast<size_t>(i)]);
  }

  using NarrowTag = ScalableTag<int16_t>;
  using WideTag = Rebind<int32_t, NarrowTag>;
  constexpr NarrowTag narrow_tag{};
  constexpr WideTag wide_tag{};
  std::vector<int16_t> narrow_input(static_cast<size_t>(size(narrow_tag)));
  std::vector<int32_t> wide_output(static_cast<size_t>(size(narrow_tag)));
  for (nint_t i = 0; i < size(narrow_tag); ++i) {
    narrow_input[static_cast<size_t>(i)] = int16_t(i * 13 - 27);
  }
  const auto fused_temporal =
      load_convert(wide_tag, narrow_input.data());
  const auto fused_non_temporal =
      load_convert(wide_tag, narrow_input.data(), mem::non_temporal);
  store_convert(
      narrow_tag, wide_output.data(),
      load(narrow_tag, narrow_input.data()), mem::non_temporal);
  for (nint_t i = 0; i < size(wide_tag); ++i) {
    EXPECT_EQ(get(wide_tag, fused_temporal, i),
              get(wide_tag, fused_non_temporal, i));
    EXPECT_EQ(int32_t(narrow_input[static_cast<size_t>(i)]),
              wide_output[static_cast<size_t>(i)]);
  }
}

TEST(VecConversionLoadStoreTest, AlignmentUsesOneMemorySideWord) {
  using WordTag = ScalableTag<int32_t>;
  constexpr WordTag word_tag{};
  constexpr Twice<WordTag> multi_word_tag{};
#if !defined(CPU_CAPABILITY_SVE)
  constexpr FixedTag<int32_t, 2> fixed_tag{};
#endif
  EXPECT_EQ(memory_alignment(word_tag),
            word_size(word_tag) * nint_t(sizeof(int32_t)));
  EXPECT_EQ(memory_alignment(multi_word_tag), memory_alignment(word_tag));
#if !defined(CPU_CAPABILITY_SVE)
  EXPECT_EQ(memory_alignment(fixed_tag),
            word_size(fixed_tag) * nint_t(sizeof(int32_t)));
#endif

#ifdef VECOPS_DEBUG
  const nint_t n = size(word_tag);
  std::vector<unsigned char> storage(
      static_cast<size_t>(n * sizeof(int32_t) +
                          memory_alignment(word_tag) + 1));
  const auto raw = reinterpret_cast<uintptr_t>(storage.data());
  const auto alignment = static_cast<uintptr_t>(memory_alignment(word_tag));
  auto* aligned = reinterpret_cast<int32_t*>(
      (raw + alignment - 1) & ~(alignment - 1));
  EXPECT_DEATH(
      (void)load(word_tag, aligned + 1, mem::aligned),
      "Not aligned");
#endif
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
