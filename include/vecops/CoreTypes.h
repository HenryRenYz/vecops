// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_CORETYPES_H
#define VECOPS_CORETYPES_H

#include <cstddef>
#include <cstdint>  // for standard int defs

#include "vecops/platform/Features.h"
#include "vecops/CoreDefs.h"
#include "vecops/util/BFloat16.h"
#include "vecops/util/Float16.h"

namespace vecops {

namespace details {

/**
 * Fixed-size storage whose trivial accessors are part of the kernel ABI.
 *
 * Unlike std::array, operator[]/data/begin/end are unconditionally force
 * inlined even after a surrounding accelerator owner becomes very large.
 */
template <typename T, std::size_t N>
struct InlineArray {
  T storage[N == 0 ? 1 : N]{};

  VECOPS_ALWAYS_INLINE constexpr InlineArray() = default;

  template <typename... U>
    requires (N > 0 && sizeof...(U) == N)
  VECOPS_ALWAYS_INLINE constexpr InlineArray(U... values)
      : storage{static_cast<T>(values)...} {}

  VECOPS_ALWAYS_INLINE constexpr T& operator[](std::size_t index) {
    return storage[index];
  }

  VECOPS_ALWAYS_INLINE constexpr const T& operator[](
      std::size_t index) const {
    return storage[index];
  }

  VECOPS_ALWAYS_INLINE constexpr T* data() { return storage; }
  VECOPS_ALWAYS_INLINE constexpr const T* data() const { return storage; }
  VECOPS_ALWAYS_INLINE constexpr T* begin() { return storage; }
  VECOPS_ALWAYS_INLINE constexpr const T* begin() const { return storage; }
  VECOPS_ALWAYS_INLINE constexpr T* end() { return storage + N; }
  VECOPS_ALWAYS_INLINE constexpr const T* end() const { return storage + N; }
  VECOPS_ALWAYS_INLINE static constexpr std::size_t size() { return N; }
};

} // namespace details

/**
 * Float types
 */
using bfloat16_t = BFloat16;
using float16_t = Float16;
using float32_t = float;
using float64_t = double;
/**
 * Signed native int, having the same width as machine word.
 */
using nint_t = ptrdiff_t;
/**
 * Unsigned native int, having the same width as machine word.
 */
using nuint_t = size_t;

/**
 * Integer types
 */
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;

} // namespace vecops

#endif //VECOPS_CORETYPES_H
