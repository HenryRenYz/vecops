//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_MATMUL_H
#define VECOPS_KERNEL_MATMUL_H

#include "vecops/gemm/Atoms.h"
#include "vecops/kernel/details/matmul/Backend.h"
#include "vecops/kernel/details/matmul/Traversal.h"

namespace vecops::kernel {

namespace matmul_implementation {

template <typename Implementation>
using resource_requirements_t =
    typename matmul_details::Backend<Implementation>::ResourceRequirements;

template <typename Implementation>
VECOPS_INLINE nint_t scratch_bytes() {
  return matmul_details::Backend<Implementation>::scratch_bytes();
}

template <typename Implementation>
inline constexpr int problem_rank_v =
    matmul_details::Backend<Implementation>::ProblemRank;

} // namespace matmul_implementation

/**
 * Multiply logical A[M,K] by B[N,K]^T into C[M,N].
 *
 * The current backends consume one rank-two leaf problem, but ProblemRank is a
 * backend contract so a future batched microkernel can retain another leading
 * dimension without changing the operator traversal. A and B can be direct
 * DataAccess objects or matching packed layouts produced by matmul_pack.
 */
template <gemm::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound(
    Scope& scope, M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch, Implementation = {})) {
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(
      std::same_as<typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(
      std::same_as<typename COutput::ComputeType, typename Atom::TAcc>);
  matmul_details::Backend<Implementation>::template run<Atom, Policy>(
      scope, m, n, k, a, b, c_input, c_output, scratch);
}

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_MATMUL_H
