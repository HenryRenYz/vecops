//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_MATMUL_H
#define VECOPS_KERNEL_MATMUL_H

#include "vecops/gemm/Atoms.h"
#include "vecops/kernel/details/matmul/Backend.h"

namespace vecops::kernel {

namespace matmul_implementation {

template <typename Implementation>
using resource_requirements_t =
    typename matmul_details::Backend<Implementation>::ResourceRequirements;

template <typename Implementation>
VECOPS_INLINE nint_t scratch_bytes() {
  return matmul_details::Backend<Implementation>::scratch_bytes();
}

} // namespace matmul_implementation

/**
 * Multiply logical A[M,K] by B[N,K]^T into C[M,N].
 *
 * A and B can be direct rank-two DataAccess objects or matching packed
 * layouts produced by matmul_pack. CInput supplies the accumulator prologue;
 * COutput supplies the epilogue and final memory conversion.
 */
template <gemm::Atom Atom, execution::ExecutionScope Scope,
          typename A, typename B, typename CInput, typename COutput,
          typename Implementation>
VECOPS_ALWAYS_INLINE void matmul_bound(
    Scope& scope, nint_t m, nint_t n, nint_t k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch, Implementation = {}) {
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(
      std::same_as<typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(
      std::same_as<typename COutput::ComputeType, typename Atom::TAcc>);
  matmul_details::Backend<Implementation>::template run<Atom>(
      scope, m, n, k, a, b, c_input, c_output, scratch);
}

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_MATMUL_H
