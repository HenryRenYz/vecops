// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_X86_BACKEND_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_X86_BACKEND_H

#include <type_traits>

#include "vecops/kernel/details/transpose/generic/Transpose2D.h"
#include "vecops/vec/Capabilities.h"

/** @file Backend.h @brief x86 transpose backend and compiler-specific tuning. */

namespace vecops::kernel::transpose2d_details {

/**
 * @brief x86 tile selection over the shared generic-vector transpose network.
 *
 * Columns use the native vector byte width, capped at 16 lanes. Rows may be
 * smaller to control register pressure. The backend remains a distinct type so
 * future AVX/AMX-specific networks need not affect SVE or generic targets.
 */
struct X86Backend {
  template <int SrcRow, int SrcCol, int DstRow, int DstCol,
            typename Scope, typename M, typename N,
            typename Source, typename Destination, typename Policy>
  /** Execute one plane with x86 compile-time tile tuning. */
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, M m, N n,
      Source& source,
      tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
      Destination& destination,
      tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
      Policy policy) {
    using T = generic::ComputeOf<Source>;
#if VEC_WIDTH > 0
    constexpr nint_t NativeBytes = VEC_WIDTH / 8;
#else
    constexpr nint_t NativeBytes = 16;
#endif
    constexpr nint_t NativeLanes =
        NativeBytes / static_cast<nint_t>(sizeof(T));
    constexpr nint_t Columns = NativeLanes < 16 ? NativeLanes : 16;
#if defined(COMPILER_GCC)
    // WORKAROUND: GCC 13/15 spills the recursive 16x16 32-bit AVX-512
    // butterfly because all row vectors remain live across several unpack
    // stages. An 8x16 rectangle keeps full-width loads while halving live row
    // state. Clang allocates the 16x16 network without those spills. Do not
    // remove this compiler split without checking both generated assemblies.
    constexpr nint_t RegisterRows = sizeof(T) >= 4 ? 8 : 16;
#else
    constexpr nint_t RegisterRows = 16;
#endif
    constexpr nint_t Rows =
        Columns < RegisterRows ? Columns : RegisterRows;
    generic::fixed_or_gather_transpose<
        Rows, Columns, SrcRow, SrcCol, DstRow, DstCol>(
        m, n, source, src_origin, destination, dst_origin, policy);
  }
};

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_X86_BACKEND_H
