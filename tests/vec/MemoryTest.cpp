// @vecops-test-shards: 241

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vecops/vec/Memory.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <typename T>
void run_consecutive_memory_test();
template <typename T, int Slot>
void run_indexed_loads_test();
template <typename T>
void run_indexed_stores_test();
template <typename T>
void run_filtered_access_test();
template <typename T>
void run_filtered_indexed_access_test();
template <typename T>
void run_fixed_sve_memory_test();
void run_repeated_addressing_test();
void run_first_counts_test();
void run_initializer_list_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(
    VECOPS_TEST_SHARD_COUNT ==
        vec_test::AllElements::size * 20 + 1);

namespace {

template <typename T>
T memory_value(int seed) {
  if constexpr (std::same_as<T, vecops::bfloat16_t>) {
    const float scalar = static_cast<float>(seed) + 0.25F;
    const uint32_t bits = ::vecops::bitcast<uint32_t>(scalar);
    const uint32_t bias = ((bits >> 16) & 1U) + uint32_t{0x7fff};
    return vecops::bfloat16_t::from_bits(
        static_cast<uint16_t>((bits + bias) >> 16));
  }
  else if constexpr (std::same_as<T, vecops::float16_t>)
    return T(static_cast<float>(seed) + 0.25F);
  else if constexpr (std::floating_point<T>)
    return static_cast<T>(seed) + static_cast<T>(0.25);
  else if constexpr (std::signed_integral<T>)
    return static_cast<T>((seed % 61) - 30);
  else
    return static_cast<T>((seed * 3 + 1) % 127);
}

template <bool FullOptions = true, vec::VectorTag Tag>
void verify_memory_shape(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr std::size_t capacity = 4096;
  ASSERT_LT(static_cast<std::size_t>(vec::size(tag) + 2), capacity);
  alignas(64) std::array<T, capacity> input{};
  alignas(64) std::array<T, capacity> output{};
  for (std::size_t lane = 0; lane < capacity; ++lane) {
    input[lane] = memory_value<T>(static_cast<int>(lane + 3));
    output[lane] = memory_value<T>(901);
  }

  const auto unaligned = vec::load(tag, input.data() + 1);
  const auto aligned = vec::load(tag, input.data(), vec::mem::aligned);
  const auto explicit_unmasked =
      vec::load(tag, input.data(), vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        input[static_cast<std::size_t>(lane + 1)],
        vec::get(tag, unaligned, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        input[static_cast<std::size_t>(lane)],
        vec::get(tag, aligned, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, aligned, lane),
        vec::get(tag, explicit_unmasked, lane)));
  }

  vec::store(tag, output.data() + 1, unaligned);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        input[static_cast<std::size_t>(lane + 1)],
        output[static_cast<std::size_t>(lane + 1)]));
  }
  EXPECT_TRUE(vec_test::values_identical(memory_value<T>(901), output[0]));

  std::fill(output.begin(), output.end(), memory_value<T>(901));
  vec::store(
      tag, output.data(), explicit_unmasked, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        input[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(lane)]));
  }

  if constexpr (!FullOptions) return;
  auto mask = vec::mfalse(tag);
  auto vector_merge = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    mask = vec::set(tag, mask, lane, lane % 3 != 0);
    vector_merge = vec::set(
        tag, vector_merge, lane, memory_value<T>(lane + 500));
  }
  const T scalar_merge = memory_value<T>(777);
  const auto zero = vec::load(
      tag, input.data(), vec::opt::masked(mask), vec::opt::zero);
  const auto scalar = vec::load(
      tag, input.data(), vec::opt::merge(scalar_merge),
      vec::opt::masked(mask));
  const auto vector = vec::load(
      tag, input.data(), vec::opt::masked(mask),
      vec::opt::merge(vector_merge));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = lane % 3 != 0;
    EXPECT_TRUE(vec_test::values_identical(
        active ? input[static_cast<std::size_t>(lane)] : T{},
        vec::get(tag, zero, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        active ? input[static_cast<std::size_t>(lane)] : scalar_merge,
        vec::get(tag, scalar, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        active ? input[static_cast<std::size_t>(lane)]
               : vec::get(tag, vector_merge, lane),
        vec::get(tag, vector, lane)));
  }
  const std::array<vecops::nint_t, 4> first_counts{
      0, 1, std::max<vecops::nint_t>(0, vec::size(tag) - 1), vec::size(tag)};
  for (const auto first_count : first_counts) {
    const auto first = vec::load(
        tag, input.data(), vec::opt::first(first_count),
        vec::opt::merge(scalar_merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
      EXPECT_TRUE(vec_test::values_identical(
          lane < first_count ? input[static_cast<std::size_t>(lane)]
                             : scalar_merge,
          vec::get(tag, first, lane))) << "first=" << first_count;
  }

  std::fill(output.begin(), output.end(), memory_value<T>(901));
  vec::store(tag, output.data(), vector, vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = lane % 3 != 0
        ? vec::get(tag, vector, lane) : memory_value<T>(901);
    EXPECT_TRUE(vec_test::values_identical(
        expected, output[static_cast<std::size_t>(lane)]));
  }

  const auto filtered_stream = vec::load(
      tag, input.data(), vec::mem::non_temporal,
      vec::opt::masked(mask), vec::opt::merge(vector_merge));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, vector, lane),
        vec::get(tag, filtered_stream, lane)));
  }

  std::fill(output.begin(), output.end(), memory_value<T>(901));
  vec::store(
      tag, output.data(), vector, vec::mem::non_temporal,
      vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = lane % 3 != 0
        ? vec::get(tag, vector, lane) : memory_value<T>(901);
    EXPECT_TRUE(vec_test::values_identical(
        expected, output[static_cast<std::size_t>(lane)]));
  }

  for (const auto first_count : first_counts) {
    std::fill(output.begin(), output.end(), memory_value<T>(901));
    vec::store(tag, output.data(), aligned, vec::opt::first(first_count));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T expected = lane < first_count
          ? vec::get(tag, aligned, lane) : memory_value<T>(901);
      EXPECT_TRUE(vec_test::values_identical(
          expected, output[static_cast<std::size_t>(lane)]));
    }
  }

  const auto streamed = vec::load(
      tag, input.data(), vec::mem::aligned, vec::mem::non_temporal);
  vec::store(
      tag, output.data(), streamed,
      vec::mem::aligned, vec::mem::non_temporal);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        input[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(lane)]));
  }
}

template <int Scale, vec::VectorTag Tag>
void verify_scaled_indexed_load(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  const auto lanes = static_cast<std::size_t>(vec::size(tag));
  const auto last_index = (lanes - 1) * 16 + 9;
  const auto bytes = last_index * Scale + sizeof(T);
  std::vector<std::byte> storage(bytes);
  std::vector<int32_t> index_values(lanes);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const int32_t index = static_cast<int32_t>(lane * 16 + 9);
    index_values[static_cast<std::size_t>(lane)] = index;
    const T expected = memory_value<T>(static_cast<int>(lane + 81));
    std::memcpy(
        storage.data() + static_cast<std::size_t>(index * Scale),
        &expected, sizeof(expected));
  }
  const auto indices = vec::load(IndexTag{}, index_values.data());
  const auto loaded = vec::load(
      tag, reinterpret_cast<const T*>(storage.data()),
      vec::indexed(indices, vec::scale<Scale>));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        memory_value<T>(static_cast<int>(lane + 81)),
        vec::get(tag, loaded, lane)));
  }
}

template <typename Index, vec::VectorTag Tag>
inline constexpr bool memory_index_shape_supported = [] {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  return vec::scale_power_v<vec::Rebind<Index, Tag>> <= 2;
#else
  return true;
#endif
}();

template <int Case = 0, vec::VectorTag Tag>
void verify_indexed_loads(Tag tag) {
  if constexpr (Case == 0) {
    verify_indexed_loads<1>(tag);
    verify_indexed_loads<2>(tag);
    verify_indexed_loads<3>(tag);
    verify_indexed_loads<4>(tag);
    verify_indexed_loads<5>(tag);
    verify_indexed_loads<6>(tag);
    return;
  }

  using T = vec::ElementOf<Tag>;
  const auto lanes = static_cast<std::size_t>(vec::size(tag));
  const auto capacity = std::max<std::size_t>(20, (lanes - 1) * 7 + 4);
  std::vector<T> input(capacity);
  std::vector<int32_t> indices32_values(lanes);
  std::vector<int64_t> indices64_values(lanes);
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = memory_value<T>(static_cast<int>(i + 17));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const auto index = lane * 7 + 3;
    indices32_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(index);
    indices64_values[static_cast<std::size_t>(lane)] =
        static_cast<int64_t>(index);
  }
  using I32Tag = vec::Rebind<int32_t, Tag>;
  const auto indices32 = vec::load(I32Tag{}, indices32_values.data());

  if constexpr (Case == 1) {
    const auto loaded = vec::load(tag, input.data(), vec::indexed(indices32));
    const auto non_temporal = vec::load(
        tag, input.data(), vec::indexed(indices32), vec::mem::non_temporal);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T expected =
          input[static_cast<std::size_t>(indices32_values[lane])];
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, loaded, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, non_temporal, lane)));
    }
  }
  else if constexpr (Case == 2) {
    const auto constant =
        vec::load(tag, input.data(), vec::strided(vecops::meta::cint<7>));
    const auto dynamic =
        vec::load(tag, input.data(), vec::strided(vecops::meta::dyn<1>(7)));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T expected = input[static_cast<std::size_t>(lane * 7)];
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, constant, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, dynamic, lane)));
    }
  }
  else if constexpr (Case == 3) {
    auto mask = vec::mfalse(tag);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
      mask = vec::set(tag, mask, lane, lane % 2 == 0);
    const T merge = memory_value<T>(701);
    const auto filtered = vec::load(
        tag, input.data(), vec::indexed(indices32),
        vec::opt::masked(mask), vec::opt::merge(merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T expected =
          input[static_cast<std::size_t>(indices32_values[lane])];
      EXPECT_TRUE(vec_test::values_identical(
          lane % 2 == 0 ? expected : merge,
          vec::get(tag, filtered, lane)));
    }
  }
  else if constexpr (Case == 4) {
    if constexpr (memory_index_shape_supported<int64_t, Tag>) {
      using I64Tag = vec::Rebind<int64_t, Tag>;
      const auto indices64 = vec::load(I64Tag{}, indices64_values.data());
      const auto loaded = vec::load(tag, input.data(), vec::indexed(indices64));
      const auto non_temporal = vec::load(
          tag, input.data(), vec::indexed(indices64), vec::mem::non_temporal);
      for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
        const T expected =
            input[static_cast<std::size_t>(indices64_values[lane])];
        EXPECT_TRUE(vec_test::values_identical(
            expected, vec::get(tag, loaded, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            expected, vec::get(tag, non_temporal, lane)));
      }
    }
  }
  else if constexpr (Case == 5) {
    verify_scaled_indexed_load<1>(tag);
    verify_scaled_indexed_load<2>(tag);
  }
  else if constexpr (Case == 6) {
    verify_scaled_indexed_load<4>(tag);
    verify_scaled_indexed_load<8>(tag);
  }
}

template <int Scale, vec::VectorTag Tag>
void verify_scaled_indexed_store(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  const auto lanes = static_cast<std::size_t>(vec::size(tag));
  const auto last_index = (lanes - 1) * 16 + 9;
  const auto bytes = last_index * Scale + sizeof(T);
  std::vector<std::byte> storage(bytes);
  std::vector<int32_t> index_values(lanes);
  std::vector<T> values(lanes);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    index_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(lane * 16 + 9);
    values[static_cast<std::size_t>(lane)] =
        memory_value<T>(static_cast<int>(lane + 181));
  }
  const auto indices = vec::load(IndexTag{}, index_values.data());
  const auto stored = vec::load(tag, values.data());
  vec::store(
      tag, reinterpret_cast<T*>(storage.data()), stored,
      vec::indexed(indices, vec::scale<Scale>));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    T actual;
    std::memcpy(
        &actual,
        storage.data() + static_cast<std::size_t>(
                             index_values[static_cast<std::size_t>(lane)] *
                             Scale),
        sizeof(actual));
    EXPECT_TRUE(vec_test::values_identical(
        values[static_cast<std::size_t>(lane)], actual));
  }
}

template <vec::VectorTag Tag>
void verify_indexed_stores(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const auto lanes = static_cast<std::size_t>(vec::size(tag));
  const auto capacity = std::max<std::size_t>(20, (lanes - 1) * 7 + 4);
  const T untouched = memory_value<T>(991);
  std::vector<T> values(lanes);
  std::vector<T> output(capacity);
  std::vector<int32_t> indices32_values(lanes);
  std::vector<int64_t> indices64_values(lanes);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    values[static_cast<std::size_t>(lane)] =
        memory_value<T>(static_cast<int>(lane + 271));
    const auto index = lane * 7 + 3;
    indices32_values[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(index);
    indices64_values[static_cast<std::size_t>(lane)] =
        static_cast<int64_t>(index);
  }
  const auto value = vec::load(tag, values.data());
  using I32Tag = vec::Rebind<int32_t, Tag>;
  const auto indices32 = vec::load(I32Tag{}, indices32_values.data());
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    mask = vec::set(tag, mask, lane, lane % 2 == 0);

  std::fill(output.begin(), output.end(), untouched);
  vec::store(tag, output.data(), value, vec::indexed(indices32));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        values[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(indices32_values[lane])]));

  std::fill(output.begin(), output.end(), untouched);
  std::vector<int32_t> duplicate_index_values(lanes, 19);
  const auto duplicate_indices =
      vec::load(I32Tag{}, duplicate_index_values.data());
  vec::store(tag, output.data(), value, vec::indexed(duplicate_indices));
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, value, vec::size(tag) - 1), output[19]));

  std::fill(output.begin(), output.end(), untouched);
  vec::store(
      tag, output.data(), value, vec::indexed(indices32),
      vec::opt::masked(mask), vec::mem::non_temporal);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        lane % 2 == 0 ? values[static_cast<std::size_t>(lane)] : untouched,
        output[static_cast<std::size_t>(indices32_values[lane])]));

  std::fill(output.begin(), output.end(), untouched);
  vec::store(tag, output.data(), value, vec::strided(vecops::meta::cint<7>));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        values[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(lane * 7)]));

  std::fill(output.begin(), output.end(), untouched);
  vec::store(
      tag, output.data(), value,
      vec::strided(vecops::meta::dyn<1>(7)));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        values[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(lane * 7)]));

  if constexpr (memory_index_shape_supported<int64_t, Tag>) {
    using I64Tag = vec::Rebind<int64_t, Tag>;
    const auto indices64 = vec::load(I64Tag{}, indices64_values.data());
    std::fill(output.begin(), output.end(), untouched);
    vec::store(tag, output.data(), value, vec::indexed(indices64));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
      EXPECT_TRUE(vec_test::values_identical(
          values[static_cast<std::size_t>(lane)],
          output[static_cast<std::size_t>(indices64_values[lane])]));
  }

  verify_scaled_indexed_store<1>(tag);
  verify_scaled_indexed_store<2>(tag);
  verify_scaled_indexed_store<4>(tag);
  verify_scaled_indexed_store<8>(tag);
}

} // namespace

template <typename T>
void run_consecutive_memory_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_memory_shape<vec_test::exhaustive_options_shape<Tag>>(Tag{});
  });
}

template <typename T, int Slot>
void run_indexed_loads_test() {
  static_assert(Slot >= 0 && Slot < 17);
  constexpr int power = Slot < 11 ? Slot - 6 : 5;
  constexpr int test_case = Slot < 11 ? 0 : Slot - 10;
  if constexpr (
      power >= vec_test::details::minimum_scalable_power<T> &&
      power <= vec_test::details::maximum_scalable_power) {
    using Tag = vec::ScalableTag<T, power>;
    if constexpr (memory_index_shape_supported<int32_t, Tag>)
      verify_indexed_loads<test_case>(Tag{});
  }
}

template <typename T>
void run_indexed_stores_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    if constexpr (memory_index_shape_supported<int32_t, Tag>)
      verify_indexed_stores(Tag{});
  });
}

template <typename T>
void run_filtered_access_test() {
  vec::ScalableTag<T, 0> tag;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  void* mapping = mmap(
      nullptr, static_cast<std::size_t>(page_size * 2),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(
      static_cast<std::byte*>(mapping) + page_size,
      static_cast<std::size_t>(page_size), PROT_NONE), 0);

  const vecops::nint_t active_count =
      std::max<vecops::nint_t>(1, vec::size(tag) / 2);
  auto* pointer = reinterpret_cast<T*>(
      static_cast<std::byte*>(mapping) + page_size) - active_count;
  for (vecops::nint_t lane = 0; lane < active_count; ++lane)
    pointer[lane] = memory_value<T>(lane + 11);

  auto prefix = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < active_count; ++lane)
    prefix = vec::set(tag, prefix, lane, true);
  const auto by_count = vec::load(
      tag, pointer, vec::opt::first(active_count));
  const auto by_mask = vec::load(
      tag, pointer, vec::opt::masked(prefix));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = lane < active_count
        ? memory_value<T>(lane + 11) : T{};
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, by_count, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, by_mask, lane)));
  }

  vec::store(tag, pointer, by_count, vec::opt::first(active_count));
  vec::store(tag, pointer, by_mask, vec::opt::masked(prefix));
  EXPECT_EQ(munmap(mapping, static_cast<std::size_t>(page_size * 2)), 0);
}

template <typename T>
void run_filtered_indexed_access_test() {
  using Tag = vec::ScalableTag<T>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  Tag tag;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  void* mapping = mmap(
      nullptr, static_cast<std::size_t>(page_size * 2),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(
      static_cast<std::byte*>(mapping) + page_size,
      static_cast<std::size_t>(page_size), PROT_NONE), 0);
  auto* pointer = static_cast<T*>(mapping);
  const int32_t protected_index = static_cast<int32_t>(page_size / sizeof(T));
  const int32_t boundary_index = protected_index - 1;
  std::array<int32_t, 4096> index_values{};
  ASSERT_LT(static_cast<std::size_t>(vec::size(tag)), index_values.size());
  auto mask = vec::mfalse(tag);
  const vecops::nint_t active_count =
      std::max<vecops::nint_t>(1, vec::size(tag) / 2);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = lane < active_count;
    index_values[static_cast<std::size_t>(lane)] =
        active ? (lane == 0 ? boundary_index : static_cast<int32_t>(lane))
               : protected_index;
    mask = vec::set(tag, mask, lane, active);
    if (active)
      pointer[index_values[static_cast<std::size_t>(lane)]] =
          memory_value<T>(lane + 11);
  }
  const auto indices = vec::load(IndexTag{}, index_values.data());
  const auto loaded = vec::load(
      tag, pointer, vec::indexed(indices), vec::opt::masked(mask));
  vec::store(
      tag, pointer, loaded, vec::indexed(indices), vec::opt::masked(mask));
  EXPECT_EQ(munmap(mapping, static_cast<std::size_t>(page_size * 2)), 0);
}

#if VECOPS_TEST_SHARD_INDEX == 240
void run_repeated_addressing_test() {
  using Tag = vec::ScalableTag<int32_t>;
  using IndexTag = vec::Rebind<int32_t, Tag>;
  constexpr std::size_t capacity = 4096;
  std::array<int32_t, capacity> input{};
  std::array<int32_t, capacity> output{};
  std::array<int32_t, capacity> index_values{};
  for (std::size_t i = 0; i < capacity; ++i)
    input[i] = static_cast<int32_t>(i * 13 + 7);
  const auto zero_stride = vec::load(Tag{}, input.data(), vec::strided(0));
  auto* reverse_base = input.data() + vec::size(Tag{}) - 1;
  const auto negative_stride = vec::load(
      Tag{}, reverse_base, vec::strided(-1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(input[0], vec::get(Tag{}, zero_stride, lane));
    EXPECT_EQ(input[static_cast<std::size_t>(vec::size(Tag{}) - 1 - lane)],
              vec::get(Tag{}, negative_stride, lane));
    index_values[static_cast<std::size_t>(lane)] = lane % 3 == 0
        ? 0 : static_cast<int32_t>(vec::size(Tag{}) - 1 - lane);
  }
  const auto indices = vec::load(IndexTag{}, index_values.data());
  const auto indexed = vec::load(Tag{}, input.data(), vec::indexed(indices));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(input[static_cast<std::size_t>(index_values[lane])],
              vec::get(Tag{}, indexed, lane));
  const auto values = vec::load(Tag{}, input.data());
  auto* reverse_output = output.data() + vec::size(Tag{}) - 1;
  vec::store(Tag{}, reverse_output, values, vec::strided(-1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(input[static_cast<std::size_t>(lane)],
              output[static_cast<std::size_t>(vec::size(Tag{}) - 1 - lane)]);
}

void run_first_counts_test() {
  using Tag = vec::ScalableTag<int32_t>;
  std::array<int32_t, 4096> storage{};
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    storage[static_cast<std::size_t>(lane)] =
        static_cast<int32_t>(lane + 11);
  const auto empty = vec::load(
      Tag{}, storage.data(), vec::opt::first(-1));
  const auto full = vec::load(
      Tag{}, storage.data(), vec::opt::first(vec::size(Tag{}) + 1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane) {
    EXPECT_EQ(vec::get(Tag{}, empty, lane), 0);
    EXPECT_EQ(
        vec::get(Tag{}, full, lane),
        storage[static_cast<std::size_t>(lane)]);
  }

  std::array<int32_t, 4096> output;
  output.fill(-1);
  const auto value = vec::fill(Tag{}, int32_t{37});
  vec::store(Tag{}, output.data(), value, vec::opt::first(-1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(output[static_cast<std::size_t>(lane)], -1);
  vec::store(
      Tag{}, output.data(), value,
      vec::opt::first(vec::size(Tag{}) + 1));
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(output[static_cast<std::size_t>(lane)], 37);
}

void run_initializer_list_test() {
  using Tag = vec::ScalableTag<float>;
  const auto value = vec::load(
      Tag{},
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
       9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F});
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(vec::get(Tag{}, value, lane), static_cast<float>(lane + 1));
}
#endif

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template <typename T>
void run_fixed_sve_memory_test() {
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word memory type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_memory_shape<false>(Tag{});
    verify_indexed_loads<0>(Tag{});
    verify_indexed_stores(Tag{});
  }
}
#endif

#if VECOPS_TEST_SHARD_INDEX == 240
template <typename Tag>
concept AcceptsAlignedNonTemporalLoad =
    requires(Tag tag, const vec::ElementOf<Tag>* pointer) {
      vec::load(tag, pointer, vec::mem::aligned, vec::mem::non_temporal);
    };

template <typename Tag>
concept AcceptsUnalignedNonTemporalLoad =
    requires(Tag tag, const vec::ElementOf<Tag>* pointer) {
      vec::load(tag, pointer, vec::mem::non_temporal);
    };

template <typename Tag>
concept AcceptsFilteredNonTemporalLoad =
    requires(Tag tag, const vec::ElementOf<Tag>* pointer, vec::Mask<Tag> mask) {
      vec::load(tag, pointer, vec::mem::non_temporal, vec::opt::masked(mask));
    };

template <typename Tag>
concept AcceptsDuplicateAlignment =
    requires(Tag tag, const vec::ElementOf<Tag>* pointer) {
      vec::load(tag, pointer, vec::mem::aligned, vec::mem::unaligned);
    };

template <typename Tag>
concept AcceptsStorePopulation =
    requires(Tag tag, vec::ElementOf<Tag>* pointer, vec::Vec<Tag> value) {
      vec::store(tag, pointer, value, vec::opt::zero);
    };

template <typename Tag>
concept AcceptsRawIndexedMask = requires(
    Tag tag, const vec::ElementOf<Tag>* pointer,
    vec::Vec<vec::Rebind<int32_t, Tag>> indices, vec::Mask<Tag> mask) {
  vec::load(tag, pointer, vec::indexed(indices), mask);
};

template <typename Tag>
concept AcceptsDuplicateIndexedActive = requires(
    Tag tag, const vec::ElementOf<Tag>* pointer,
    vec::Vec<vec::Rebind<int32_t, Tag>> indices, vec::Mask<Tag> mask) {
  vec::load(
      tag, pointer, vec::indexed(indices),
      vec::opt::masked(mask), vec::opt::first(1));
};

template <typename Tag>
concept AcceptsIndexedStorePopulation = requires(
    Tag tag, vec::ElementOf<Tag>* pointer,
    vec::Vec<vec::Rebind<int32_t, Tag>> indices, vec::Vec<Tag> value) {
  vec::store(tag, pointer, value, vec::indexed(indices), vec::opt::zero);
};

template <typename Tag>
concept AcceptsNonTemporalIndexedAccess = requires(
    Tag tag, vec::ElementOf<Tag>* pointer,
    vec::Vec<vec::Rebind<int32_t, Tag>> indices, vec::Vec<Tag> value) {
  vec::load(tag, pointer, vec::indexed(indices), vec::mem::non_temporal);
  vec::store(
      tag, pointer, value, vec::indexed(indices), vec::mem::non_temporal);
};

using ConstraintTag = vec::ScalableTag<float>;
static_assert(AcceptsAlignedNonTemporalLoad<ConstraintTag>);
static_assert(AcceptsUnalignedNonTemporalLoad<ConstraintTag>);
static_assert(AcceptsFilteredNonTemporalLoad<ConstraintTag>);
static_assert(!AcceptsDuplicateAlignment<ConstraintTag>);
static_assert(!AcceptsStorePopulation<ConstraintTag>);
static_assert(!AcceptsRawIndexedMask<ConstraintTag>);
static_assert(!AcceptsDuplicateIndexedActive<ConstraintTag>);
static_assert(!AcceptsIndexedStorePopulation<ConstraintTag>);
static_assert(AcceptsNonTemporalIndexedAccess<ConstraintTag>);
#endif

#if VECOPS_TEST_SHARD_INDEX < 12
using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_consecutive_memory_test<ShardType>();
#elif VECOPS_TEST_SHARD_INDEX < 216
constexpr std::size_t shard_type_index =
    (VECOPS_TEST_SHARD_INDEX - 12) / 17;
constexpr int shard_slot = (VECOPS_TEST_SHARD_INDEX - 12) % 17;
using ShardType = vec_test::ElementAt<shard_type_index>;
template void run_indexed_loads_test<ShardType, shard_slot>();
#elif VECOPS_TEST_SHARD_INDEX < 228
using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX - 216>;
template void run_indexed_stores_test<ShardType>();
#elif VECOPS_TEST_SHARD_INDEX < 240
using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX - 228>;
template void run_filtered_access_test<ShardType>();
template void run_filtered_indexed_access_test<ShardType>();
#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template void run_fixed_sve_memory_test<ShardType>();
#endif
#endif

#else

template <typename T>
class VecMemoryTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecMemoryTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecMemoryTest, ConsecutiveOptionsCoverEveryShape) {
  run_consecutive_memory_test<TypeParam>();
}

template <typename T, std::size_t... Slots>
void run_all_indexed_loads(std::index_sequence<Slots...>) {
  (run_indexed_loads_test<T, static_cast<int>(Slots)>(), ...);
}

TYPED_TEST(VecMemoryTest, IndexedAndStridedLoadsCoverEveryShape) {
  run_all_indexed_loads<TypeParam>(std::make_index_sequence<17>{});
}

TYPED_TEST(VecMemoryTest, IndexedAndStridedStoresCoverEveryShape) {
  run_indexed_stores_test<TypeParam>();
}

TYPED_TEST(VecMemoryTest, FilteredAccessStopsAtProtectedPage) {
  run_filtered_access_test<TypeParam>();
}

TYPED_TEST(VecMemoryTest, FilteredIndexedAccessSkipsProtectedPage) {
  run_filtered_indexed_access_test<TypeParam>();
}

TEST(VecMemoryEdgeTest, ZeroNegativeAndRepeatedAddressing) {
  run_repeated_addressing_test();
}

TEST(VecMemoryEdgeTest, FirstCountsFollowMwhileltSemantics) {
  run_first_counts_test();
}

TEST(VecMemoryInitializerListTest, LoadsExactlyOneLogicalVector) {
  run_initializer_list_test();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecMemoryTest, FixedSVEBatchesBeyondTupleLimit) {
  run_fixed_sve_memory_test<TypeParam>();
}
#endif

#endif
