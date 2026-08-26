//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_MATMUL_PACK_H
#define VECOPS_KERNEL_MATMUL_PACK_H

#include "vecops/gemm/Packing.h"
#include "vecops/kernel/details/matmul_pack/Backend.h"

namespace vecops::kernel {

namespace matmul_pack_implementation {

template <gemm::Atom Atom, gemm::Operand Side, typename Implementation>
using resource_requirements_t = typename matmul_pack_details::Backend<
    typename gemm::packing_t<Atom, Side>::FormatType,
    Implementation>::ResourceRequirements;

} // namespace matmul_pack_implementation

template <gemm::Atom Atom, gemm::Operand Side,
          execution::ExecutionScope Scope,
          typename Source, typename Destination,
          typename Implementation = matmul_pack_implementation::Vector>
VECOPS_ALWAYS_INLINE void matmul_pack_bound(
    Scope& scope, const Source& source, Destination& destination,
    Implementation = {}) {
  using Packing = gemm::packing_t<Atom, Side>;
  static_assert(std::same_as<typename Source::ComputeType,
                             typename Packing::Element>);
  static_assert(std::same_as<typename Destination::ComputeType,
                             typename Packing::Element>);
  using Backend = matmul_pack_details::Backend<
      typename Packing::FormatType, Implementation>;
  Backend::template run<Atom, Side>(scope, source, destination);
}

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_MATMUL_PACK_H
