// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/**
 * @file vecops/matmul/details/packing/Kernel.h
 * @brief Kernel-layer entries of the matmul packing sublayer.
 *
 * The `matmul_pack_bound` / `matmul_pack_b_compensated_bound` functions sit
 * between the plan layer (packing/Plan.h) and the Backend dispatch surface
 * (packing/Backend.h): they resolve the operand's FormatType plus an
 * Implementation tag into a Backend specialization and forward to its
 * run()/run_compensated().  The plan layer always passes its
 * SelectedPackImplementation explicitly; the `Vector` default serves
 * callers that drive the kernel layer directly and want the base
 * implementation of the current format.
 */

#ifndef VECOPS_MATMUL_DETAILS_PACK_KERNEL_H
#define VECOPS_MATMUL_DETAILS_PACK_KERNEL_H

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/packing/Backend.h"

namespace vecops::kernel {

namespace matmul_pack_implementation {

/// Resource requirements the given Backend specialization reports for
/// packing Atom/Side: what execution resources its run() body needs to be
/// compiled under (see packing/Backend.h for the protocol).
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Implementation>
using resource_requirements_t = typename matmul_pack_details::Backend<
    typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
    Implementation>::ResourceRequirements;

} // namespace matmul_pack_implementation

/**
 * @brief Pack one operand through the resolved Backend specialization.
 *
 * Resolves `Backend<FormatType, Implementation>` from the Atom/Side packing
 * format and forwards to `Backend::run`.  The source/destination element
 * types must equal the packing's element type (statically checked here);
 * the block layout itself is validated by the plan layer.
 */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          execution::ExecutionScope Scope,
          typename Source, typename Destination,
          typename Implementation = matmul_pack_implementation::Vector>
VECOPS_ALWAYS_INLINE void matmul_pack_bound(
    Scope& scope, const Source& source, Destination& destination,
    Implementation = {}) {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;
  static_assert(std::same_as<typename Source::ComputeType,
                             typename Packing::Element>);
  static_assert(std::same_as<typename Destination::ComputeType,
                             typename Packing::Element>);
  using Backend = matmul_pack_details::Backend<
      typename Packing::FormatType, Implementation>;
  Backend::template run<Atom, Side>(scope, source, destination);
}

/**
 * @brief Pack signed-byte B and generate the asymmetric-A column sidecar.
 *
 * Requires a u8 x s8 -> i32 atom, an s8 source, and a Backend that sets
 * `supports_column_compensation` (only the fused/transpose-capable packing
 * backends do).  Forwards to `Backend::run_compensated`, which writes the
 * packed B panel and, per B row n, `compensation[n] = -a_zero_point *
 * sum_k(B[n][k])`.
 */
template <::vecops::matmul::Atom Atom,
          execution::ExecutionScope Scope,
          typename Source, typename Destination,
          typename CompensationDestination,
          typename Implementation = matmul_pack_implementation::Vector>
VECOPS_ALWAYS_INLINE void matmul_pack_b_compensated_bound(
    Scope& scope, const Source& source, Destination& destination,
    CompensationDestination& compensation, int32_t a_zero_point,
    Implementation = {}) {
  using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>;
  static_assert(std::same_as<typename Atom::TA, uint8_t>);
  static_assert(std::same_as<typename Atom::TB, int8_t>);
  static_assert(std::same_as<typename Atom::TAcc, int32_t>);
  static_assert(std::same_as<typename Source::ComputeType,
                             typename Packing::Element>);
  static_assert(std::same_as<typename Destination::ComputeType,
                             typename Packing::Element>);
  static_assert(std::same_as<
                typename CompensationDestination::ComputeType, int32_t>);
  using Backend = matmul_pack_details::Backend<
      typename Packing::FormatType, Implementation>;
  static_assert(
      Backend::supports_column_compensation,
      "the selected packing backend cannot generate a column sidecar");
  Backend::template run_compensated<Atom>(
      scope, source, destination, compensation, a_zero_point);
}

} // namespace vecops::kernel

#endif // VECOPS_MATMUL_DETAILS_PACK_KERNEL_H
