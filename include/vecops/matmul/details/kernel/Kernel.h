//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_H
#define VECOPS_MATMUL_DETAILS_KERNEL_H

/**
 * @file vecops/matmul/details/kernel/Kernel.h
 * @brief Kernel-layer entry points that bind a leaf matmul to a backend.
 *
 * Design intent: these are the only functions outside the backends that
 * talk to matmul_details::Backend directly. There are two entry styles:
 *
 *  - with_matmul_configuration() + matmul_bound_configured(): the leaf is
 *    entered inside an implementation-specific configuration scope, and
 *    nested calls must reuse the outer scope instead of opening a second
 *    one. The explicit "configured" spelling prevents an ordinary nested
 *    call from assuming that an equal configuration type also carries
 *    equal runtime row counts.
 *  - matmul_bound(): the self-contained whole-problem entry; it opens the
 *    configuration itself when the backend defines one.
 *
 * The matmul_implementation namespace re-exports backend metadata
 * (resource requirements, scratch size, problem rank) so the tiled layer
 * can size its workspace without naming a concrete backend type.
 */

#include <utility>

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/kernel/AccumulatorRoute.h"
#include "vecops/matmul/details/kernel/Backend.h"
#include "vecops/matmul/details/kernel/TileScheduler.h"

namespace vecops::kernel {

namespace matmul_implementation {

/// Resources an Implementation claims from its execution scope.
template <typename Implementation>
using resource_requirements_t =
    typename matmul_details::Backend<Implementation>::ResourceRequirements;

/// Scratch bytes the backend's microkernels require; the caller allocates.
template <typename Implementation>
VECOPS_INLINE nint_t scratch_bytes() {
  return matmul_details::Backend<Implementation>::scratch_bytes();
}

/// Rank of one leaf problem the backend consumes.
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
 * AllowTailSplit is supplied by the caller from the actual leaf plan.  Both
 * explicit and online/cache-packed operands may therefore reuse packed-only
 * residual kernels; candidate predicates still verify the bound access types.
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
  // Probe the extended run<> signature (family dispatch, tail-split
  // control) first and fall back to the older run<Atom, Policy> form, so
  // backends that predate FamilyDispatch keep compiling unchanged.
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
 * Execute one split-K leaf while carrying both the semantic C operands and
 * the native accumulator operands. Route is intentionally a runtime value for
 * a real K loop, allowing first/middle/last to share one backend instantiation.
 */
template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          typename Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B,
          typename CInput, typename COutput,
          typename AccInput, typename AccOutput,
          typename Route, typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound_phased(
    Scope& scope, M m, N n, K k,
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output,
    Route route, void* scratch, Implementation = {})) {
  static_assert(execution::ExecutionScope<Scope>);
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(std::same_as<
      typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename COutput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename AccInput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename AccOutput::ComputeType, typename Atom::TAcc>);
  using Backend = matmul_details::Backend<Implementation>;
  Backend::template run_phased<Atom, Policy>(
      scope, m, n, k, a, b,
      c_input, c_output, acc_input, acc_output, route, scratch);
}

/** N-major GenericTiled leaf; separate from matmul_bound() to preserve the
 * historical M-major template ABI and code layout. */
template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          bool AllowTailSplit = false,
          typename FamilyDispatch =
              ::vecops::matmul::details::AutomaticFamilyDispatch,
          typename Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound_n_major(
    Scope& scope, M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch, Implementation = {})) {
  static_assert(execution::ExecutionScope<Scope>);
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(std::same_as<
      typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename COutput::ComputeType, typename Atom::TAcc>);
  using Backend = matmul_details::Backend<Implementation>;
  Backend::template run_n_major<
      Atom, Policy, AllowTailSplit, FamilyDispatch>(
          scope, m, n, k, a, b, c_input, c_output, scratch);
}

template <::vecops::matmul::Atom Atom,
          typename Policy = matmul_policy::Automatic,
          typename Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename AccInput, typename AccOutput,
          typename Route, typename Implementation>
VECOPS_KERNEL_FUNCTION(void matmul_bound_phased_n_major(
    Scope& scope, M m, N n, K k, const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output,
    Route route, void* scratch, Implementation = {})) {
  static_assert(execution::ExecutionScope<Scope>);
  static_assert(std::same_as<typename A::ComputeType, typename Atom::TA>);
  static_assert(std::same_as<typename B::ComputeType, typename Atom::TB>);
  static_assert(std::same_as<
      typename CInput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename COutput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename AccInput::ComputeType, typename Atom::TAcc>);
  static_assert(std::same_as<
      typename AccOutput::ComputeType, typename Atom::TAcc>);
  using Backend = matmul_details::Backend<Implementation>;
  Backend::template run_phased_n_major<Atom, Policy>(
      scope, m, n, k, a, b, c_input, c_output,
      acc_input, acc_output, route, scratch);
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
