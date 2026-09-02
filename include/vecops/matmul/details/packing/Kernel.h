//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_KERNEL_H
#define VECOPS_MATMUL_DETAILS_PACK_KERNEL_H

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/packing/Backend.h"

namespace vecops::kernel {

namespace matmul_pack_implementation {

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Implementation>
using resource_requirements_t = typename matmul_pack_details::Backend<
    typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
    Implementation>::ResourceRequirements;

} // namespace matmul_pack_implementation

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
