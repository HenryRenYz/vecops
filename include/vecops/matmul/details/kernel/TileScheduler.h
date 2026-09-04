//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H
#define VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H

/**
 * @file vecops/matmul/details/kernel/TileScheduler.h
 * @brief Backend-neutral traversal of one bound rank-two matrix product.
 *
 * Design intent: this header glues the generic kernel::loop::tile2d
 * traversal to four Backend hooks and nothing else: EffectivePolicy
 * (traversal policy after Automatic resolution), Catalog (per-Case kernel
 * selection), dispatch_plan (compile-time kernel plan, e.g. K guarantees),
 * and run_case (one Case instantiation). Keeping it backend-free lets the
 * AMX and SME backends share one traversal body; hardware-state ownership
 * (TILECFG image, streaming-ZA region) stays with the caller so a whole
 * traversal forms one lexical hardware-state interval.
 *
 * The two entry points differ in coordinate space: run_tiles() traverses
 * the whole logical problem (origin 0,0), while run_tiles_region()
 * traverses a sub-region of a larger problem and therefore carries both
 * the traversal extents and the logical/origin coordinates separately —
 * this is what lets a backend re-enter a shortened traversal (e.g. the
 * AMX packed-A/B tail split) under the same configuration.
 */

#include "vecops/matmul/Atom.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/matmul/details/kernel/AccumulatorRoute.h"

namespace vecops::kernel::matmul_details {

template <typename Dispatch>
struct NMajorFamilyDispatch : Dispatch {
  static constexpr bool n_major_traversal = true;
};

template <typename Dispatch>
inline constexpr bool n_major_family_dispatch_v = [] {
  if constexpr (requires { Dispatch::n_major_traversal; })
    return static_cast<bool>(Dispatch::n_major_traversal);
  else
    return false;
}();

/**
 * Backend-neutral traversal of one bound rank-two matrix product.
 *
 * A backend supplies only its catalog, effective policy, compile-time kernel
 * plan, and one-Case implementation.  Streaming/ZA or TILECFG ownership stays
 * outside this function so the backend can make this whole traversal one
 * lexical hardware-state interval.
 */
template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType TraversalM, meta::ValueType TraversalN,
          meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_KERNEL_FUNCTION(void run_tiles_region(
    TraversalM traversal_m, TraversalN traversal_n, K k,
    nint_t logical_m, nint_t logical_n,
    nint_t origin_m, nint_t origin_n,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, TraversalM, TraversalN, K, A, B, CInput, COutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, TraversalM, TraversalN, K,
      A, B, CInput, COutput>;
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan<Atom, A, B>(
      traversal_m, traversal_n, k,
      [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d<EffectivePolicy>(
            traversal_m, traversal_n,
            Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case<Atom, Case, Plan>(
                  a, b, c_input, c_output,
                  logical_m, logical_n, logical_k,
                  origin_m + mi, origin_n + ni,
                  active_m, active_n, scratch);
            });
      });
}

/**
 * Whole-problem form of the traversal above: origin is (0, 0) and the
 * traversal extents are the logical extents. Prefer run_tiles_region()
 * when re-entering a sub-region under an already-established
 * configuration.
 */
template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_KERNEL_FUNCTION(void run_tiles(
    M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, M, N, K, A, B, CInput, COutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, M, N, K, A, B, CInput, COutput>;
  const nint_t logical_m = static_cast<nint_t>(m);
  const nint_t logical_n = static_cast<nint_t>(n);
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan<Atom, A, B>(
      m, n, k, [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d<EffectivePolicy>(
            m, n, Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case<Atom, Case, Plan>(
                  a, b, c_input, c_output,
                  logical_m, logical_n, logical_k, mi, ni,
                  active_m, active_n, scratch);
            });
      });
}

/**
 * Split-K traversal carrying both semantic C endpoints and the native
 * accumulator endpoints. First/middle/last blocks share one Route type; the
 * backend selects only the prologue source and epilogue destination around a
 * single Case compute body.
 */
template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B,
          typename CInput, typename COutput,
          typename AccInput, typename AccOutput,
          typename Route>
VECOPS_KERNEL_FUNCTION(void run_tiles_phased(
    M m, N n, K k,
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output,
    Route route, void* scratch)) {
  // The accumulator is always native TAcc storage. Resolve the traversal from
  // that stable shape so all split phases choose one catalog/policy type even
  // when the original C carries a transform or a non-direct layout.
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, M, N, K, A, B, AccInput, AccOutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, M, N, K, A, B, AccInput, AccOutput>;
  const nint_t logical_m = static_cast<nint_t>(m);
  const nint_t logical_n = static_cast<nint_t>(n);
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan<Atom, A, B>(
      m, n, k, [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d<EffectivePolicy>(
            m, n, Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case_phased<Atom, Case, Plan>(
                  a, b, c_input, c_output, acc_input, acc_output, route,
                  logical_m, logical_n, logical_k, mi, ni,
                  active_m, active_n, scratch);
            });
      });
}

// N-major entries are deliberately separate from the historical functions
// above. Default M-major instantiations therefore retain their original
// template signatures, mangling, and code layout.
template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType TraversalM, meta::ValueType TraversalN,
          meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_KERNEL_FUNCTION(void run_tiles_region_n_major(
    TraversalM traversal_m, TraversalN traversal_n, K k,
    nint_t logical_m, nint_t logical_n,
    nint_t origin_m, nint_t origin_n,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, TraversalM, TraversalN, K, A, B, CInput, COutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, TraversalM, TraversalN, K,
      A, B, CInput, COutput>;
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan_n_major<Atom, A, B>(
      traversal_m, traversal_n, k,
      [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d_n_major<EffectivePolicy>(
            traversal_m, traversal_n, Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case<Atom, Case, Plan>(
                  a, b, c_input, c_output,
                  logical_m, logical_n, logical_k,
                  origin_m + mi, origin_n + ni,
                  active_m, active_n, scratch);
            });
      });
}

template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_KERNEL_FUNCTION(void run_tiles_n_major(
    M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, M, N, K, A, B, CInput, COutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, M, N, K, A, B, CInput, COutput>;
  const nint_t logical_m = static_cast<nint_t>(m);
  const nint_t logical_n = static_cast<nint_t>(n);
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan_n_major<Atom, A, B>(
      m, n, k, [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d_n_major<EffectivePolicy>(
            m, n, Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case<Atom, Case, Plan>(
                  a, b, c_input, c_output,
                  logical_m, logical_n, logical_k, mi, ni,
                  active_m, active_n, scratch);
            });
      });
}

template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename AccInput, typename AccOutput, typename Route>
VECOPS_KERNEL_FUNCTION(void run_tiles_phased_n_major(
    M m, N n, K k, const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output,
    Route route, void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, M, N, K, A, B, AccInput, AccOutput>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, M, N, K, A, B, AccInput, AccOutput>;
  const nint_t logical_m = static_cast<nint_t>(m);
  const nint_t logical_n = static_cast<nint_t>(n);
  const nint_t logical_k = static_cast<nint_t>(k);
  Backend::template dispatch_plan_n_major<Atom, A, B>(
      m, n, k, [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
        kernel::loop::tile2d_n_major<EffectivePolicy>(
            m, n, Atom::M_R, Atom::N_R, Catalog{},
            [&]<typename Case>(Case, nint_t mi, nint_t ni,
                               nint_t active_m, nint_t active_n)
                VECOPS_INLINE_LAMBDA_NOEXCEPT {
              Backend::template run_case_phased<Atom, Case, Plan>(
                  a, b, c_input, c_output, acc_input, acc_output, route,
                  logical_m, logical_n, logical_k, mi, ni,
                  active_m, active_n, scratch);
            });
      });
}

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H
