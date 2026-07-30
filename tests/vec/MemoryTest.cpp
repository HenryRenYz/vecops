#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vecops/vec/Memory.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

template <typename T>
T memory_value(int seed) {
  if constexpr (std::same_as<T, vecops::bfloat16_t>)
    return T(static_cast<float>(seed) + 0.25F);
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
  return vec::scale_power<vec::Rebind<Index, Tag>> <= 2;
#else
  return true;
#endif
}();

template <vec::VectorTag Tag>
void verify_indexed_loads(Tag tag) {
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
  const auto by_i32 = vec::load(tag, input.data(), vec::indexed(indices32));
  const auto by_i32_non_temporal = vec::load(
      tag, input.data(), vec::indexed(indices32), vec::mem::non_temporal);
  const auto by_constant_stride =
      vec::load(tag, input.data(), vec::strided(vecops::gemm::cint<7>));
  const auto by_dynamic_stride =
      vec::load(tag, input.data(), vec::strided(vecops::gemm::dyn<1>(7)));

  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    mask = vec::set(tag, mask, lane, lane % 2 == 0);
  const T merge = memory_value<T>(701);
  const auto filtered = vec::load(
      tag, input.data(), vec::indexed(indices32),
      vec::opt::masked(mask), vec::opt::merge(merge));

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T indexed_expected =
        input[static_cast<std::size_t>(indices32_values[lane])];
    const T strided_expected = input[static_cast<std::size_t>(lane * 7)];
    EXPECT_TRUE(vec_test::values_identical(
        indexed_expected, vec::get(tag, by_i32, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        indexed_expected, vec::get(tag, by_i32_non_temporal, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        strided_expected, vec::get(tag, by_constant_stride, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        strided_expected, vec::get(tag, by_dynamic_stride, lane)));
    EXPECT_TRUE(vec_test::values_identical(
        lane % 2 == 0 ? indexed_expected : merge,
        vec::get(tag, filtered, lane)));
  }
  if constexpr (memory_index_shape_supported<int64_t, Tag>) {
    using I64Tag = vec::Rebind<int64_t, Tag>;
    const auto indices64 = vec::load(I64Tag{}, indices64_values.data());
    const auto by_i64 = vec::load(tag, input.data(), vec::indexed(indices64));
    const auto by_i64_non_temporal = vec::load(
        tag, input.data(), vec::indexed(indices64), vec::mem::non_temporal);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      EXPECT_TRUE(vec_test::values_identical(
          input[static_cast<std::size_t>(indices64_values[lane])],
          vec::get(tag, by_i64, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          input[static_cast<std::size_t>(indices64_values[lane])],
          vec::get(tag, by_i64_non_temporal, lane)));
    }
  }

  verify_scaled_indexed_load<1>(tag);
  verify_scaled_indexed_load<2>(tag);
  verify_scaled_indexed_load<4>(tag);
  verify_scaled_indexed_load<8>(tag);
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
  vec::store(tag, output.data(), value, vec::strided(vecops::gemm::cint<7>));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_TRUE(vec_test::values_identical(
        values[static_cast<std::size_t>(lane)],
        output[static_cast<std::size_t>(lane * 7)]));

  std::fill(output.begin(), output.end(), untouched);
  vec::store(
      tag, output.data(), value,
      vec::strided(vecops::gemm::dyn<1>(7)));
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

template <typename T>
class VecMemoryTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecMemoryTest, vec_test::AllElementTypes);

TYPED_TEST(VecMemoryTest, ConsecutiveOptionsCoverEveryShape) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_memory_shape<vec_test::exhaustive_options_shape<Tag>>(Tag{});
  });
}

TYPED_TEST(VecMemoryTest, IndexedAndStridedLoadsCoverEveryShape) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    if constexpr (memory_index_shape_supported<int32_t, Tag>)
      verify_indexed_loads(Tag{});
  });
}

TYPED_TEST(VecMemoryTest, IndexedAndStridedStoresCoverEveryShape) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    if constexpr (memory_index_shape_supported<int32_t, Tag>)
      verify_indexed_stores(Tag{});
  });
}

TYPED_TEST(VecMemoryTest, FilteredAccessStopsAtProtectedPage) {
  using T = TypeParam;
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

TYPED_TEST(VecMemoryTest, FilteredIndexedAccessSkipsProtectedPage) {
  using T = TypeParam;
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

TEST(VecMemoryEdgeTest, ZeroNegativeAndRepeatedAddressing) {
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

#ifdef VECOPS_DEBUG
TEST(VecMemoryEdgeTest, FirstRejectsOutOfRangeCounts) {
  using Tag = vec::ScalableTag<int32_t>;
  std::array<int32_t, 4096> storage{};
  const auto value = vec::zeros(Tag{});
  EXPECT_DEATH((void)vec::load(
      Tag{}, storage.data(), vec::opt::first(-1)), "count");
  EXPECT_DEATH((void)vec::load(
      Tag{}, storage.data(), vec::opt::first(vec::size(Tag{}) + 1)), "count");
  EXPECT_DEATH(vec::store(
      Tag{}, storage.data(), value, vec::opt::first(-1)), "count");
  EXPECT_DEATH(vec::store(
      Tag{}, storage.data(), value,
      vec::opt::first(vec::size(Tag{}) + 1)), "count");
}
#endif

TEST(VecMemoryInitializerListTest, LoadsExactlyOneLogicalVector) {
  using Tag = vec::ScalableTag<float>;
  const auto value = vec::load(
      Tag{},
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
       9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F});
  for (vecops::nint_t lane = 0; lane < vec::size(Tag{}); ++lane)
    EXPECT_EQ(vec::get(Tag{}, value, lane), static_cast<float>(lane + 1));
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecMemoryTest, FixedSVEBatchesBeyondTupleLimit) {
  using T = TypeParam;
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
    verify_indexed_loads(Tag{});
    verify_indexed_stores(Tag{});
  }
}
#endif

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

} // namespace
