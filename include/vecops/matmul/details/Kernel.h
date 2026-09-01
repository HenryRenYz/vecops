//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_H
#define VECOPS_MATMUL_DETAILS_KERNEL_H

#include <utility>

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/Backend.h"
#include "vecops/matmul/details/TileScheduler.h"

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
 * Enter one implementation-specific matrix-multiply configuration around a
 * group of leaf problems.  Backends without a separate configuration simply
 * forward the current scope.  PackedB describes the final leaf operand after
 * any operation-level packing, not necessarily the user's original input.
 */
template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          bool PackedB = false,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N,
          typename Implementation, typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_matmul_configuration(
    Scope& scope, M m, N n, Implementation, Fn&& fn) {
  using Backend = matmul_details::Backend<Implementation>;
  if constexpr (requires {
                  Backend::template with_configuration<
                      Atom, Policy, PackedB>(
                          scope, m, n, std::forward<Fn>(fn));
                }) {
    return Backend::template with_configuration<Atom, Policy, PackedB>(
        scope, m, n, std::forward<Fn>(fn));
  } else {
    return std::forward<Fn>(fn)(scope);
  }
}

/**
 * Multiply logical A[M,K] by B[N,K]^T into C[M,N].
 *
 * The current backends consume one rank-two leaf problem, but ProblemRank is a
 * backend contract so a future batched microkernel can retain another leading
 * dimension without changing the operator traversal. A and B can be direct
 * DataAccess objects or matching packed layouts produced by matmul_pack.
 * AllowTailSplit is intentionally supplied by the operation from the user's
 * original operand types, so an internal online-pack instantiation cannot
 * duplicate the narrow explicit-packed tail specialization.
 */
template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          bool AllowTailSplit = false,
          typename FamilyDispatch =
              ::vecops::matmul::details::AutomaticFamilyDispatch,
          typename Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound(
    Scope& scope, M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch, Implementation = {})) {
  static_assert(execution::ExecutionScope<Scope>);
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(
      std::same_as<typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(
      std::same_as<typename COutput::ComputeType, typename Atom::TAcc>);
  using Backend = matmul_details::Backend<Implementation>;
  if constexpr (requires {
                  Backend::template run<
                      Atom, Policy, AllowTailSplit, FamilyDispatch>(
                      scope, m, n, k, a, b,
                      c_input, c_output, scratch);
                }) {
    Backend::template run<Atom, Policy, AllowTailSplit, FamilyDispatch>(
        scope, m, n, k, a, b, c_input, c_output, scratch);
  } else {
    Backend::template run<Atom, Policy>(
        scope, m, n, k, a, b, c_input, c_output, scratch);
  }
}

/**
 * Execute a leaf under a configuration established by
 * with_matmul_configuration().  This explicit entry point prevents ordinary
 * nested calls from assuming that an equal configuration type also carries
 * equal runtime row counts.
 */
template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound_configured(
    Scope& scope, M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch, Implementation = {})) {
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(
      std::same_as<typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(
      std::same_as<typename COutput::ComputeType, typename Atom::TAcc>);
  using Backend = matmul_details::Backend<Implementation>;
  static_assert(requires {
    Backend::template run_configured<Atom, Policy>(
        scope, m, n, k, a, b, c_input, c_output, scratch);
  });
  Backend::template run_configured<Atom, Policy>(
      scope, m, n, k, a, b, c_input, c_output, scratch);
}

} // namespace vecops::kernel

#endif // VECOPS_MATMUL_DETAILS_KERNEL_H
