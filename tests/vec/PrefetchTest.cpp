#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "vecops/vec/Prefetch.h"

namespace vec = vecops::vec;

namespace {

template <typename Tag, typename... Options>
concept CanPrefetch = requires(
    Tag tag, const vec::ElementOf<Tag>* pointer, Options... options) {
  vec::prefetch(tag, pointer, options...);
};

using CompileTag = vec::ScalableTag<int32_t>;
static_assert(CanPrefetch<CompileTag>);
static_assert(CanPrefetch<
              CompileTag, vec::mem::PrefetchL2,
              vec::mem::PrefetchStream, vec::mem::PrefetchWrite>);
static_assert(!CanPrefetch<
              CompileTag, vec::mem::PrefetchL1,
              vec::mem::PrefetchL2>);
static_assert(!CanPrefetch<
              CompileTag, vec::mem::PrefetchKeep,
              vec::mem::PrefetchStream>);
static_assert(!CanPrefetch<
              CompileTag, vec::mem::PrefetchRead,
              vec::mem::PrefetchWrite>);
static_assert(!CanPrefetch<CompileTag, vec::mem::Temporal>);

template <vec::Element T>
void prefetch_all_combinations() {
  using Tag = vec::ScalableTag<T>;
  alignas(64) std::array<T, 256> values{};
  const T* pointer = values.data();

  vec::prefetch(Tag{}, pointer);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l1,
      vec::mem::prefetch_keep, vec::mem::prefetch_read);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l1,
      vec::mem::prefetch_stream, vec::mem::prefetch_read);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l2,
      vec::mem::prefetch_keep, vec::mem::prefetch_read);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l2,
      vec::mem::prefetch_stream, vec::mem::prefetch_read);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l1,
      vec::mem::prefetch_keep, vec::mem::prefetch_write);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l1,
      vec::mem::prefetch_stream, vec::mem::prefetch_write);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l2,
      vec::mem::prefetch_keep, vec::mem::prefetch_write);
  vec::prefetch(
      Tag{}, pointer, vec::mem::prefetch_l2,
      vec::mem::prefetch_stream, vec::mem::prefetch_write);
}

TEST(VecPrefetchTest, AllOptionsAndElementWidthsDoNotCrash) {
  prefetch_all_combinations<int8_t>();
  prefetch_all_combinations<int16_t>();
  prefetch_all_combinations<int32_t>();
  prefetch_all_combinations<int64_t>();
}

} // namespace
