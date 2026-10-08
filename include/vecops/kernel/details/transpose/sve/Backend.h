// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_SVE_BACKEND_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_SVE_BACKEND_H

#include <type_traits>

#include "vecops/kernel/details/transpose/generic/Transpose2D.h"

/** @file Backend.h @brief VLA/fixed-length SVE transpose backend. */

namespace vecops::kernel::transpose2d_details {

/**
 * @brief SVE transpose backend for VLA and fixed-vector-length compilation.
 *
 * VLA SVE cannot instantiate a C++ `std::array` containing one sizeless vector
 * per matrix row, so Automatic delegates to the generic gather path. A fixed
 * SVE vector length has sized vector types and can use the fixed butterfly.
 */
struct SVEBackend {
  template <int SrcRow, int SrcCol, int DstRow, int DstCol,
            typename Scope, typename M, typename N,
            typename Source, typename Destination, typename Policy>
  /** Execute one plane using the policy valid for the SVE compilation mode. */
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, M m, N n,
      Source& source,
      tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
      Destination& destination,
      tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
      Policy policy) {
    if constexpr (std::same_as<Policy, transpose2d_policy::Gather>) {
      generic::gather_transpose<
          M, N, Source, Destination, SrcRow, SrcCol, DstRow, DstCol>(
          m, n, source, src_origin, destination, dst_origin);
#if defined(HAS_FIXED_SVE_BITS)
    } else {
      static_assert(std::same_as<Policy, transpose2d_policy::Automatic>);
      using T = generic::ComputeOf<Source>;
      constexpr nint_t NativeBytes = FIXED_SVE_BITS / 8;
      constexpr nint_t NativeLanes =
          NativeBytes / static_cast<nint_t>(sizeof(T));
      constexpr nint_t Columns = NativeLanes < 16 ? NativeLanes : 16;
      // WORKAROUND/TUNING: a 16x16 fixed-SVE network substantially increases
      // compiler memory/time and keeps too many vectors live on current
      // BiSheng Clang and GCC. Eight rows retain 64-byte column loads without
      // triggering that pressure. Re-evaluate with assembly and compile-time
      // measurements before increasing this value.
      constexpr nint_t Rows = Columns < 8 ? Columns : 8;
      generic::fixed_transpose<
          Rows, Columns, M, N, Source, Destination,
          SrcRow, SrcCol, DstRow, DstCol>(
          m, n, source, src_origin, destination, dst_origin);
#else
    } else {
      static_assert(std::same_as<Policy, transpose2d_policy::Automatic>);
      // ACLE sizeless vectors cannot be C++ array elements in VLA mode. This is
      // a type-system restriction, not a runtime performance fallback.
      generic::gather_transpose<
          M, N, Source, Destination, SrcRow, SrcCol, DstRow, DstCol>(
          m, n, source, src_origin, destination, dst_origin);
#endif
    }
  }
};

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_SVE_BACKEND_H
