/** Standalone public converting-store regression; no Vec/Mask containers. */
#include <algorithm>
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

namespace {

float input_value(Index lane) {
  switch (lane % 29) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return std::numeric_limits<float>::infinity();
    case 3: return -std::numeric_limits<float>::infinity();
    case 4: return std::numeric_limits<float>::quiet_NaN();
    case 5: return std::ldexp(1.0f, -24);
    case 6: return -std::ldexp(1.0f, -24);
    case 7: return std::ldexp(1.0f, -25);
    case 8: return -std::ldexp(1.0f, -25);
    case 9: return 65504.0f;
    case 10: return 65520.0f;
    default: {
      // Exact F32 half-ULP ties and their immediate neighbors exercise RNE,
      // while the lane-dependent offset detects ordering across word edges.
      const float tie = 1.0f + float(lane % 23) / 64.0f + 1.0f / 2048.0f;
      const float value = lane % 3 == 0 ? tie
          : std::nextafter(tie, lane % 3 == 1 ? 0.0f : 2.0f);
      return lane % 2 == 0 ? value : -value;
    }
  }
}

void identical(Half actual, Half expected, const char* mode, Index lane) {
  // NaN payload encoding is not part of the scalar/backend public contract.
  // NaN classification remains mandatory; all other values compare bits.
  if (std::isnan(float(expected))) {
    if (std::isnan(float(actual))) return;
  } else if (actual.to_bits() == expected.to_bits()) {
    return;
  }
  throw std::runtime_error(std::string(mode) + " lane=" +
                           std::to_string(lane) + " actual=" +
                           std::to_string(actual.to_bits()) + " expected=" +
                           std::to_string(expected.to_bits()));
}

template <vec::VectorTag Tag>
void verify_stores(Tag tag) {
  const Index lanes = vec::size(tag);
  const Half canary = Half::from_bits(0x557b);
  std::vector<float> input(static_cast<std::size_t>(lanes));
  for (Index lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] = input_value(lane);
  const auto value = vec::load(tag, input.data());
  std::vector<Half> backing(static_cast<std::size_t>(lanes * 3 + 8), canary);
  auto* output = backing.data() + 4;

  auto check_contiguous = [&](Index active, bool masked) {
    std::fill(backing.begin(), backing.end(), canary);
    if (masked) {
      std::vector<float> selection(static_cast<std::size_t>(lanes));
      for (Index lane = 0; lane < lanes; ++lane)
        selection[std::size_t(lane)] = (lane & 3) != 1 ? 1.0f : 0.0f;
      const auto mask = vec::cmpne(
          tag, vec::load(tag, selection.data()), vec::zeros(tag));
      vec::store_convert(tag, output, value, vec::opt::masked(mask));
    } else {
      vec::store_convert(tag, output, value, vec::opt::first(active));
    }
    for (Index lane = -4; lane < Index(backing.size()) - 4; ++lane) {
      const bool enabled = lane >= 0 && lane < lanes &&
          (masked ? (lane & 3) != 1 : lane < active);
      identical(output[lane], enabled ? Half{input[std::size_t(lane)]} : canary,
                masked ? "masked" : "first", lane);
    }
  };
  // Every prefix crosses both the physical word boundary and the packed
  // output boundary. Empty/full prefixes are checked, not merely interior.
  for (Index active = 0; active <= lanes; ++active)
    check_contiguous(active, false);
  check_contiguous(0, true);

  for (bool split : {false, true}) {
    std::fill(backing.begin(), backing.end(), canary);
    if (split)
      vec::store_convert(tag, output, value, vec::mem::split);
    else
      vec::store_convert(tag, output, value);
    for (Index lane = -4; lane < Index(backing.size()) - 4; ++lane)
      identical(output[lane], lane >= 0 && lane < lanes
          ? Half{input[std::size_t(lane)]} : canary,
          split ? "split" : "unmasked", lane);
  }

  for (Index stride : {Index{2}, Index{-2}}) {
    std::fill(backing.begin(), backing.end(), canary);
    auto* pointer = stride > 0 ? output : output + 2 * (lanes - 1);
    vec::store_convert(tag, pointer, value, vec::strided(stride),
                       vec::opt::first(lanes - 1));
    for (Index position = -4; position < Index(backing.size()) - 4; ++position) {
      const Index lane = stride > 0 ? position / 2 : lanes - 1 - position / 2;
      const bool enabled = position >= 0 && position % 2 == 0 &&
                           lane >= 0 && lane < lanes - 1;
      identical(output[position], enabled ? Half{input[std::size_t(lane)]} : canary,
                "strided", position);
    }
  }

  std::fill(backing.begin(), backing.end(), canary);
  using IndexTag = vec::IndexTag<Tag>;
  const auto indices = vec::mul(
      IndexTag{}, vec::iota(IndexTag{}, 0), vec::fill(IndexTag{}, 2));
  vec::store_convert(tag, output, value, vec::indexed(indices),
                     vec::opt::first(lanes - 1));
  for (Index position = -4; position < Index(backing.size()) - 4; ++position) {
    const bool enabled = position >= 0 && position % 2 == 0 &&
                         position / 2 < lanes - 1;
    identical(output[position], enabled
        ? Half{input[std::size_t(position / 2)]} : canary,
        "indexed", position);
  }
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
  const auto value = vec::fill(tag, 1.00048828125f);
  try {
    for (Index active = 0; active <= vec::size(tag); ++active) {
      auto* output = reinterpret_cast<Half*>(allocation + page) - active;
      vec::store_convert(tag, output, value, vec::opt::first(active));
      for (Index lane = 0; lane < active; ++lane)
        identical(output[lane], Half{1.00048828125f}, "guard-first", lane);
      const auto mask = vec::mwhilelt(tag, Index{0}, active);
      vec::store_convert(tag, output, value, vec::opt::masked(mask));
      for (Index lane = 0; lane < active; ++lane)
        identical(output[lane], Half{1.00048828125f}, "guard-masked", lane);
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
    verify_stores(vec::ScalableTag<float, -1>{});
    verify_stores(vec::ScalableTag<float, 0>{});
    verify_stores(vec::ScalableTag<float, 1>{});
    verify_stores(vec::ScalableTag<float, 2>{});
    verify_stores(vec::FixedTag<float, 1>{});
    verify_stores(vec::FixedTag<float, 4>{});
#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
    verify_stores(vec::FixedTag<float, 8>{});
    verify_stores(vec::FixedTag<float, 32>{});
#endif
    verify_guard_page(vec::ScalableTag<float, 1>{});
    verify_guard_page(vec::ScalableTag<float, 2>{});
    std::cout << "ordered FP16 pair stores: scalar-bit oracle, canaries and guard pages passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
