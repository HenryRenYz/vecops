// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_KERNEL_TRANSPOSE2D_H
#define VECOPS_KERNEL_TRANSPOSE2D_H

#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/details/transpose/Backend.h"
#include "vecops/kernel/details/transpose/Types.h"

/**
 * @file vecops/kernel/Transpose2D.h
 * @brief Backend-dispatched transpose between already-bound DataAccess views.
 *
 * This public layer owns validation and dispatch only. Platform kernels,
 * register-network algorithms, and hardware-resource requirements live below
 * kernel/details/transpose. No operand binding or cache blocking happens here,
 * so DataAccess materialization can call it without recursive binding.
 *
 * @code
 * // Inside a DataAccess materialization with bound accesses x (read) and
 * // y (write), transposing the logical MxN plane at (src_origin, dst_origin):
 * kernel::transpose2d_bound<1, 0, 0, 1>(
 *     scope, m, n, x, src_origin, y, dst_origin,
 *     kernel::transpose2d_policy::Automatic{});
 * // SrcRow=1, SrcCol=0: source rows run along access axis 1; the
 * // destination receives them on axis 0 (DstRow=0, DstCol=1).
 * @endcode
 */
namespace vecops::kernel {

namespace transpose2d_implementation {

template <typename Implementation>
/** Resource contract exported by a transpose implementation backend. */
using resource_requirements_t = typename transpose2d_details::
    ImplementationBackend<Implementation>::ResourceRequirements;

} // namespace transpose2d_implementation

/**
 * Transpose one logical MxN plane between two already-bound accesses.
 * The source and destination ComputeType must match. `src(i,j)` is written to
 * `dst(j,i)` according to the four compile-time axis arguments.
 *
 * @tparam SrcRow Source axis defining logical rows.
 * @tparam SrcCol Source axis defining logical columns.
 * @tparam DstRow Destination axis receiving logical rows.
 * @tparam DstCol Destination axis receiving logical columns.
 * @tparam Scope Execution scope proving resources required by Implementation.
 * @tparam M Meta row extent (`Const`, constrained `Dynamic`, or runtime value).
 * @tparam N Meta column extent.
 * @tparam Policy Automatic backend choice or forced Gather.
 * @tparam Implementation Vector or SME implementation tag.
 * @param scope Active execution scope.
 * @param m Source row extent.
 * @param n Source column extent.
 * @param source Bound readable DataAccess.
 * @param src_origin Logical coordinate corresponding to source `(0,0)`.
 * @param destination Bound writable DataAccess.
 * @param dst_origin Logical coordinate corresponding to destination `(0,0)`.
 * @param policy Compile-time dispatch policy value.
 *
 * The function neither binds operands nor allocates workspace. This prevents
 * recursion when DataAccess itself uses Transpose2D for materialization.
 */
template <int SrcRow, int SrcCol, int DstRow, int DstCol,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N,
          typename Source, typename Destination,
          typename Policy = transpose2d_policy::Automatic,
          typename Implementation = transpose2d_implementation::Vector>
  requires transpose2d_details::generic::
      CompatibleAccesses<Source, Destination>
VECOPS_ALWAYS_INLINE void transpose2d_bound(
    Scope& scope, M m, N n,
    Source& source,
    tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
    Policy policy = {}, Implementation = {}) {
  static_assert(SrcRow != SrcCol && DstRow != DstCol);
  static_assert(0 <= SrcRow && SrcRow < std::remove_cvref_t<Source>::Rank);
  static_assert(0 <= SrcCol && SrcCol < std::remove_cvref_t<Source>::Rank);
  static_assert(0 <= DstRow && DstRow < std::remove_cvref_t<Destination>::Rank);
  static_assert(0 <= DstCol && DstCol < std::remove_cvref_t<Destination>::Rank);
  VECOPS_ASSERT(static_cast<nint_t>(m) >= 0 && static_cast<nint_t>(n) >= 0,
                "transpose extents must be non-negative");

  using Backend =
      transpose2d_details::ImplementationBackend<Implementation>;
  Backend::template run<SrcRow, SrcCol, DstRow, DstCol>(
      scope, m, n, source, src_origin, destination, dst_origin, policy);
}

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_TRANSPOSE2D_H
