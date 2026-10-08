// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_SCALAR_BACKEND_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_SCALAR_BACKEND_H

#include <type_traits>

#include "vecops/kernel/details/transpose/generic/Transpose2D.h"

/** @file Backend.h @brief Portable transpose backend and tile selection. */

namespace vecops::kernel::transpose2d_details {

/**
 * @brief Portable fixed-width transpose backend.
 *
 * The backend delegates to the generic-vector implementation with a square
 * tile no wider than 16 bytes or 16 lanes. It is selected for explicit generic
 * builds and for targets whose vec implementation currently has no dedicated
 * transpose tuning. Delegation is explicit so a future platform backend can
 * replace the algorithm without changing the public entry point.
 */
struct ScalarBackend {
  template <int SrcRow, int SrcCol, int DstRow, int DstCol,
            typename Scope, typename M, typename N,
            typename Source, typename Destination, typename Policy>
  /** Execute one plane with compile-time axis mapping and policy dispatch. */
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, M m, N n,
      Source& source,
      tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
      Destination& destination,
      tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
      Policy policy) {
    using T = generic::ComputeOf<Source>;
    constexpr nint_t NativeLanes =
        16 / static_cast<nint_t>(sizeof(T));
    constexpr nint_t Tile = NativeLanes < 16 ? NativeLanes : 16;
    generic::fixed_or_gather_transpose<
        Tile, Tile, SrcRow, SrcCol, DstRow, DstCol>(
        m, n, source, src_origin, destination, dst_origin, policy);
  }
};

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_SCALAR_BACKEND_H
