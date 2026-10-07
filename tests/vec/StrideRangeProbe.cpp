// Sparse anonymous mappings exercise valid huge strides without huge RSS/files.
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <sys/mman.h>
#include <type_traits>

#include "vecops/vec/Vec.h"

namespace {
using Tag = vecops::vec::FixedTag<vecops::float32_t, 4>;

template <typename T, vecops::nint_t Step>
struct Sparse {
  static constexpr vecops::nint_t step = Step;
  static constexpr std::size_t bytes = 7 * step * sizeof(T) + 4096;
  void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

  ~Sparse() { if (mapping != MAP_FAILED) munmap(mapping, bytes); }

  T* base(bool negative) {
    const auto offset = (negative ? 6 : 3) * step * sizeof(T);
    return reinterpret_cast<T*>(static_cast<unsigned char*>(mapping) + offset);
  }

  void initialize() {
    T* pointer = base(false);
    for (vecops::nint_t i = 0; i < 4; ++i)
      pointer[i * step] = static_cast<T>(static_cast<float>(i) + 0.5f);
  }
};

bool equals(vecops::vec::Vec<Tag> value, bool negative,
            float increment = 0, bool partial = false) {
  const auto expected = [&](int lane) {
    if (partial && lane >= 2) return -8.0f;
    return static_cast<float>(negative ? 3 - lane : lane) + 0.5f + increment;
  };
  return vecops::vec::get(Tag{}, value, 0) == expected(0)
      && vecops::vec::get(Tag{}, value, 1) == expected(1)
      && vecops::vec::get(Tag{}, value, 2) == expected(2)
      && vecops::vec::get(Tag{}, value, 3) == expected(3);
}

template <typename Memory, vecops::nint_t Step = vecops::nint_t{1} << 30>
int run() {
  Sparse<Memory, Step> sparse;
  if (sparse.mapping == MAP_FAILED) {
    std::fprintf(stderr, "anonymous sparse mmap failed: errno=%d\n", errno);
    return 2;
  }
  for (bool negative : {false, true}) {
    sparse.initialize();
    Memory* pointer = sparse.base(negative);
    const auto stride = negative ? -Step : Step;
    auto value = vecops::vec::load_convert(Tag{}, pointer, vecops::vec::opt::strided(stride));
    if (!equals(value, negative)) return 41;
    auto partial = vecops::vec::load_convert(
        Tag{}, pointer, vecops::vec::opt::strided(stride),
        vecops::vec::opt::first(2), vecops::vec::opt::merge(-8.0f));
    if (!equals(partial, negative, 0, true)) return 42;
    value = vecops::vec::add(Tag{}, value, vecops::vec::fill(Tag{}, 2.0f));
    vecops::vec::store_convert(Tag{}, pointer, value, vecops::vec::opt::strided(stride));
    for (vecops::nint_t lane = 0; lane < 4; ++lane) {
      const float expected = static_cast<float>(negative ? 3 - lane : lane) + 2.5f;
      if (static_cast<float>(pointer[lane * stride]) != expected) return 43;
    }
    sparse.initialize();
    vecops::vec::store_convert(Tag{}, pointer, value,
                               vecops::vec::opt::strided(stride),
                               vecops::vec::opt::first(2));
    for (vecops::nint_t lane = 0; lane < 4; ++lane) {
      const float expected = static_cast<float>(negative ? 3 - lane : lane)
          + (lane < 2 ? 2.5f : 0.5f);
      if (static_cast<float>(pointer[lane * stride]) != expected) return 44;
    }
    if constexpr (std::same_as<Memory, vecops::float32_t>) {
      sparse.initialize();
      value = vecops::vec::load(Tag{}, pointer, vecops::vec::opt::strided(stride));
      if (!equals(value, negative)) return 45;
      vecops::vec::store(Tag{}, pointer, value, vecops::vec::opt::strided(stride));
      auto merged = vecops::vec::load(
          Tag{}, pointer, vecops::vec::opt::strided(stride),
          vecops::vec::opt::first(2), vecops::vec::opt::merge(-8.0f));
      if (!equals(merged, negative, 0, true)) return 46;
      value = vecops::vec::add(Tag{}, value, vecops::vec::fill(Tag{}, 2.0f));
      vecops::vec::store(Tag{}, pointer, value, vecops::vec::opt::strided(stride),
                         vecops::vec::opt::first(2));
      for (vecops::nint_t lane = 0; lane < 4; ++lane) {
        const float expected = static_cast<float>(negative ? 3 - lane : lane)
            + (lane < 2 ? 2.5f : 0.5f);
        if (pointer[lane * stride] != expected) return 47;
      }
    }
  }
  auto zero = vecops::vec::load_convert(
      Tag{}, static_cast<const Memory*>(nullptr),
      vecops::vec::opt::strided(Step),
      vecops::vec::opt::first(0), vecops::vec::opt::merge(-8.0f));
  if (vecops::vec::get(Tag{}, zero, 0) != -8.0f
      || vecops::vec::get(Tag{}, zero, 3) != -8.0f) return 48;
  vecops::vec::store_convert(
      Tag{}, static_cast<Memory*>(nullptr), zero,
      vecops::vec::opt::strided(Step),
      vecops::vec::opt::first(0));
  return 0;
}
}  // namespace

int main() {
  int result = run<vecops::float32_t>();
  if (result) {
    std::fprintf(stderr, "float32 stride probe failed: %d\n", result);
    return result;
  }
  result = run<vecops::float16_t>();
  if (result) {
    std::fprintf(stderr, "float16 stride probe failed: %d\n", result);
    return result;
  }
  // The unscaled i32 sequence fits here, but SVE's 4-byte offset scaling
  // would overflow. x86 gather uses its scale directly, so this is a target
  // semantic regression, not evidence that x86 has the same second defect.
  result = run<vecops::float32_t, vecops::nint_t{1} << 28>();
  if (result) {
    std::fprintf(stderr, "byte-offset stride probe failed: %d\n", result);
    return result;
  }
  std::puts("huge signed stride load/store/convert/first/merge/null gate passed");
  return 0;
}
