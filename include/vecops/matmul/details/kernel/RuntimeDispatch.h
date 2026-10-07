// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_RUNTIME_DISPATCH_H
#define VECOPS_MATMUL_DETAILS_KERNEL_RUNTIME_DISPATCH_H

#include "vecops/CoreTypes.h"

namespace vecops::kernel::matmul_details {

/** Stable semantic classes consumed by the shared runtime shape selector. */
enum class SmallVectorShape {
  amx_bf16,
  amx_i8_same_sign,
  amx_i8_mixed_sign,
  sme_f16,
  sme_other,
};

namespace runtime_dispatch_rules {

constexpr bool area_at_most(nint_t m, nint_t n, nint_t limit) {
  return m > 0 && n > 0 && n <= limit / m;
}

constexpr bool skinny_at_most(nint_t m, nint_t n, nint_t limit) {
  return (m == 1 && 1 <= n && n <= limit) ||
      (n == 1 && 1 <= m && m <= limit);
}

constexpr bool large_m1(nint_t m, nint_t n) {
  return m == 1 && 128 <= n && n <= 4096;
}

constexpr bool large_n1(nint_t m, nint_t n) {
  return n == 1 && 128 <= m && m <= 4096;
}

constexpr bool amx_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k) {
  if (m < 1 || n < 1 || k < 1) return false;
  if (skinny_at_most(m, n, 64)) return true;
  switch (shape) {
    case SmallVectorShape::amx_bf16:
      return area_at_most(m, n, 16) ||
          (area_at_most(m, n, 32) && k <= 256) ||
          (area_at_most(m, n, 64) && k <= 64) ||
          (k >= 32 && k % 32 <= 1 &&
           (large_m1(m, n) || (large_n1(m, n) && k != 256)));
    case SmallVectorShape::amx_i8_same_sign:
      return area_at_most(m, n, 32) ||
          (k >= 64 && k % 64 <= 1 &&
           (large_m1(m, n) || large_n1(m, n)));
    case SmallVectorShape::amx_i8_mixed_sign:
      return area_at_most(m, n, 32) ||
          (area_at_most(m, n, 64) && k <= 128) ||
          (k >= 64 && k % 64 <= 1 &&
           (large_m1(m, n) || (large_n1(m, n) && k != 64)));
    default:
      return false;
  }
}

constexpr bool sme_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k) {
  if (m < 0 || n < 0 || k < 0) return false;
  const nint_t limit = shape == SmallVectorShape::sme_f16 ? 16 : 64;
  return (m == 1 && n <= limit) || (n == 1 && m <= limit);
}

constexpr bool amx_residual_split_profitable(
    nint_t m, nint_t n, nint_t k) {
  const bool selected_m = m == 17 || m == 18 || m == 20;
  const bool selected_n = n == 33 || n == 47 || n == 48;
  // The packed ABI and tile scheduler are symmetric in the two spatial
  // dimensions.  Keep the measured catalog but admit its transpose too.
  return ((selected_m && selected_n) ||
          ((n == 17 || n == 18 || n == 20) &&
           (m == 33 || m == 47 || m == 48))) &&
      k == 1024;
}

} // namespace runtime_dispatch_rules

/** Runtime form of the AMX SmallVector profitability rules. */
VECOPS_NOINLINE bool runtime_amx_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k);

/** Runtime form of the SME SmallVector profitability rules. */
VECOPS_NOINLINE bool runtime_sme_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k);

/** Runtime bounded-area predicate shared by fused vector leaves. */
VECOPS_NOINLINE bool runtime_small_area_profitable(
    nint_t m, nint_t n, nint_t k, nint_t limit);

/** Runtime form of the original AMX packed residual-split rule. */
VECOPS_NOINLINE bool runtime_amx_residual_split_profitable(
    nint_t m, nint_t n, nint_t k);

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_KERNEL_RUNTIME_DISPATCH_H
