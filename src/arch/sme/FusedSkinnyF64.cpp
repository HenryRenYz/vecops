// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//
// Explicit FP64 skinny compute instantiations for SME-capable multiarch
// targets.  This source is intentionally excluded from generic libvecops.a
// and compiled in an ISA-specific archive with the consumer target's -march.

#include "vecops/matmul/details/kernel/Backend.h"

namespace vecops::kernel::matmul_details::sme {

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_compute_f64_row(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k) {
  sve_skinny_fused_compute<false, float64_t, float64_t>(
      a_data, a_stride, b_data, b_stride,
      values, outputs, logical_k);
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_compute_f64_col(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k) {
  sve_skinny_fused_compute<true, float64_t, float64_t>(
      a_data, a_stride, b_data, b_stride,
      values, outputs, logical_k);
}

} // namespace vecops::kernel::matmul_details::sme
