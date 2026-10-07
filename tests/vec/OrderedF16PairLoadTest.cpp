/** Standalone ordered converting-load regression; no Vec/Mask containers. */
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vecops/vec/ConversionMemory.h"
#include "vecops/vec/Vec.h"

namespace vec = vecops::vec;
using Half = vecops::float16_t;
using Index = vecops::nint_t;

template <typename Tag>
concept SplitLoad = vec::VectorTag<Tag> && requires(Tag tag, const Half* pointer) {
  vec::load_convert(tag, pointer, vec::mem::split);
};
static_assert(!SplitLoad<vec::ScalableTag<float, 1>>);

namespace {

Half input_value(Index lane) {
  switch (lane % 29) {
    case 0: return Half::from_bits(0x0000);
    case 1: return Half::from_bits(0x8000);
    case 2: return Half::from_bits(0x7c00);
    case 3: return Half::from_bits(0xfc00);
    case 4: return Half::from_bits(0x7e15);
    case 5: return Half::from_bits(0x0001);
    case 6: return Half::from_bits(0x8001);
    case 7: return Half::from_bits(0x03ff);
    case 8: return Half::from_bits(0x0400);
    case 9: return Half::from_bits(0x7bff);
    default: {
      const uint16_t sign = (lane & 1) ? 0x8000 : 0;
      return Half::from_bits(uint16_t(sign | (0x3400 + lane % 1023)));
    }
  }
}

float inactive_value(Index lane) {
  return lane & 1 ? -17.25f - float(lane) : 9.5f + float(lane);
}

void identical(float actual, float expected, const char* mode, Index lane) {
  if (std::isnan(expected)) {
    if (std::isnan(actual)) return;
  } else if (std::bit_cast<uint32_t>(actual) ==
             std::bit_cast<uint32_t>(expected)) {
    return;
  }
  throw std::runtime_error(std::string(mode) + " lane=" +
      std::to_string(lane) + " actual=" +
      std::to_string(std::bit_cast<uint32_t>(actual)) + " expected=" +
      std::to_string(std::bit_cast<uint32_t>(expected)));
}

template <vec::VectorTag Tag>
void check_value(Tag tag, vec::Vec<Tag> value,
                 const std::vector<float>& expected, const char* mode) {
  const Index lanes = vec::size(tag);
  constexpr float canary = -123.5f;
  std::vector<float> output(std::size_t(lanes + 8), canary);
  vec::store(tag, output.data() + 4, value);
  for (Index lane = -4; lane < lanes + 4; ++lane)
    identical(output[std::size_t(lane + 4)], lane >= 0 && lane < lanes
        ? expected[std::size_t(lane)] : canary, mode, lane);
}

template <vec::VectorTag Tag>
void verify_loads(Tag tag) {
  const Index lanes = vec::size(tag);
  std::vector<Half> input(std::size_t(lanes * 3 + 8), Half::from_bits(0x557b));
  auto* pointer = input.data() + 4;
  std::vector<float> expected(static_cast<std::size_t>(lanes));
  std::vector<float> inactive(static_cast<std::size_t>(lanes));
  std::vector<float> selected(static_cast<std::size_t>(lanes));
  for (Index lane = 0; lane < lanes; ++lane) {
    pointer[lane] = input_value(lane);
    expected[std::size_t(lane)] = float(pointer[lane]);
    inactive[std::size_t(lane)] = inactive_value(lane);
    selected[std::size_t(lane)] = (lane & 3) != 1 ? 1.0f : 0.0f;
  }
  const auto before = input;
  check_value(tag, vec::load_convert(tag, pointer), expected, "unmasked");
  check_value(tag, vec::load_convert(tag, pointer, vec::mem::non_temporal),
              expected, "non-temporal-fallback");
  const auto merge = vec::load(tag, inactive.data());
  for (Index count = 0; count <= lanes; ++count) {
    for (Index lane = 0; lane < lanes; ++lane)
      expected[std::size_t(lane)] = lane < count ? float(pointer[lane]) : 0.0f;
    check_value(tag, vec::load_convert(tag, pointer, vec::opt::first(count)),
                expected, "first-zero");
    for (Index lane = count; lane < lanes; ++lane)
      expected[std::size_t(lane)] = -17.25f;
    check_value(tag, vec::load_convert(tag, pointer, vec::opt::first(count),
                                      vec::opt::merge(-17.25f)),
                expected, "first-scalar-merge");
    for (Index lane = count; lane < lanes; ++lane)
      expected[std::size_t(lane)] = inactive[std::size_t(lane)];
    check_value(tag, vec::load_convert(tag, pointer, vec::opt::first(count),
                                      vec::opt::merge(merge)),
                expected, "first-vector-merge");
  }
  const auto mask = vec::cmpne(
      tag, vec::load(tag, selected.data()), vec::zeros(tag));
  for (Index lane = 0; lane < lanes; ++lane)
    expected[std::size_t(lane)] = selected[std::size_t(lane)] != 0
        ? float(pointer[lane]) : inactive[std::size_t(lane)];
  check_value(tag, vec::load_convert(tag, pointer, vec::opt::masked(mask),
                                    vec::opt::merge(merge)),
              expected, "masked-vector-merge");
  for (Index lane = 0; lane < lanes; ++lane)
    if (selected[std::size_t(lane)] == 0) expected[std::size_t(lane)] = 0;
  check_value(tag, vec::load_convert(tag, pointer, vec::opt::masked(mask)),
              expected, "masked-zero");
  for (std::size_t lane = 0; lane < input.size(); ++lane)
    if (input[lane].to_bits() != before[lane].to_bits())
      throw std::runtime_error("converting load changed its input");

  // Different scalar addresses must not be folded into the contiguous pair.
  for (Index lane = 0; lane < lanes; ++lane)
    pointer[2 * lane] = input_value(lane);
  for (Index stride : {Index{2}, Index{-2}}) {
    const auto* address = stride > 0 ? pointer : pointer + 2 * (lanes - 1);
    for (Index lane = 0; lane < lanes; ++lane)
      expected[std::size_t(lane)] = lane < lanes - 1
          ? float(address[lane * stride]) : inactive[std::size_t(lane)];
    check_value(tag, vec::load_convert(tag, address, vec::strided(stride),
                                      vec::opt::first(lanes - 1),
                                      vec::opt::merge(merge)),
                expected, "signed-stride-fallback");
  }
  using IndexTag = vec::IndexTag<Tag>;
  const auto indices = vec::mul(
      IndexTag{}, vec::iota(IndexTag{}, 0), vec::fill(IndexTag{}, 2));
  for (Index lane = 0; lane < lanes; ++lane)
    expected[std::size_t(lane)] = lane < lanes - 1
        ? float(pointer[2 * lane]) : inactive[std::size_t(lane)];
  check_value(tag, vec::load_convert(tag, pointer, vec::indexed(indices),
                                    vec::opt::first(lanes - 1),
                                    vec::opt::merge(merge)),
              expected, "indexed-fallback");

  // Neither an inactive input SNaN nor a merge SNaN may be converted as an
  // active value. Inspect the result bits, not a floating comparison.
  std::fill(input.begin(), input.end(), Half::from_bits(0x7c01));
  const float signaling = std::bit_cast<float>(uint32_t{0x7fa54321});
  std::feclearexcept(FE_ALL_EXCEPT);
  const auto untouched = vec::load_convert(
      tag, pointer, vec::opt::first(0), vec::opt::merge(signaling));
  vec::store(tag, inactive.data(), untouched);
  if (std::fetestexcept(FE_INVALID))
    throw std::runtime_error("inactive SNaN caused invalid FP conversion");
  for (float scalar : inactive)
    if (std::bit_cast<uint32_t>(scalar) != std::bit_cast<uint32_t>(signaling))
      throw std::runtime_error("inactive SNaN merge bits changed");
}

template <vec::VectorTag Tag>
void verify_guard_page(Tag tag) {
  const long page = sysconf(_SC_PAGESIZE);
  if (page < 1) throw std::runtime_error("sysconf page size failed");
  auto* allocation = static_cast<std::byte*>(mmap(
      nullptr, std::size_t(page) * 2, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (allocation == MAP_FAILED) throw std::runtime_error("mmap failed");
  if (mprotect(allocation + page, std::size_t(page), PROT_NONE) != 0) {
    munmap(allocation, std::size_t(page) * 2);
    throw std::runtime_error("mprotect failed");
  }
  const Index lanes = vec::size(tag);
  std::vector<float> expected(std::size_t(lanes), -33.5f);
  std::vector<float> selected(std::size_t(lanes), 0.0f);
  try {
    for (Index count = 0; count <= lanes; ++count) {
      auto* input = reinterpret_cast<Half*>(allocation + page) - count;
      for (Index lane = 0; lane < count; ++lane)
        input[lane] = input_value(lane);
      for (Index lane = 0; lane < lanes; ++lane)
        expected[std::size_t(lane)] = lane < count ? float(input[lane]) : -33.5f;
      check_value(tag, vec::load_convert(tag, input, vec::opt::first(count),
                                        vec::opt::merge(-33.5f)),
                  expected, "guard-first");
      for (Index lane = 0; lane < lanes; ++lane) {
        const bool enabled = lane < count && (lane & 3) != 1;
        selected[std::size_t(lane)] = enabled ? 1.0f : 0.0f;
        expected[std::size_t(lane)] = enabled ? float(input[lane]) : -33.5f;
      }
      const auto mask = vec::cmpne(
          tag, vec::load(tag, selected.data()), vec::zeros(tag));
      check_value(tag, vec::load_convert(tag, input, vec::opt::masked(mask),
                                        vec::opt::merge(-33.5f)),
                  expected, "guard-holes");
    }
  } catch (...) {
    munmap(allocation, std::size_t(page) * 2);
    throw;
  }
  munmap(allocation, std::size_t(page) * 2);
}

}  // namespace

int main() {
  try {
    verify_loads(vec::ScalableTag<float, -1>{});
    verify_loads(vec::ScalableTag<float, 0>{});
    verify_loads(vec::ScalableTag<float, 1>{});
    verify_loads(vec::ScalableTag<float, 2>{});
    verify_loads(vec::FixedTag<float, 1>{});
    verify_loads(vec::FixedTag<float, 4>{});
#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
    verify_loads(vec::FixedTag<float, 8>{});
    verify_loads(vec::FixedTag<float, 16>{});
    verify_loads(vec::FixedTag<float, 32>{});
#endif
    verify_guard_page(vec::ScalableTag<float, -1>{});
    verify_guard_page(vec::ScalableTag<float, 0>{});
    verify_guard_page(vec::ScalableTag<float, 1>{});
    verify_guard_page(vec::ScalableTag<float, 2>{});
    std::cout << "ordered FP16 pair loads: scalar-bit oracle, inactive lanes and guard pages passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
