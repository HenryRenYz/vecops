//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H
#define VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H

#include "vecops/matmul/Atom.h"
#include "vecops/kernel/Tile2D.h"

namespace vecops::kernel::matmul_details {

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
      Atom, Policy, TraversalM, TraversalN, K, A, B>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, A, B>;
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

template <typename Backend, ::vecops::matmul::Atom Atom, typename Policy,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_KERNEL_FUNCTION(void run_tiles(
    M m, N n, K k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    void* scratch)) {
  using EffectivePolicy = typename Backend::template EffectivePolicy<
      Atom, Policy, M, N, K, A, B>;
  using Catalog = typename Backend::template Catalog<
      Atom, EffectivePolicy, A, B>;
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

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_TILE_SCHEDULER_H
