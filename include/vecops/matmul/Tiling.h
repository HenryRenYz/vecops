//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_TILING_H
#define VECOPS_MATMUL_TILING_H

#include <algorithm>
#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/util/Math.h"

namespace vecops::matmul {

namespace details {

template <typename T>
inline constexpr nint_t atom_alignment_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (V::is_const) return V::value;
  else return V::alignment;
}();

VECOPS_INLINE nint_t aligned_tile(nint_t budget, nint_t alignment) {
  VECOPS_ASSERT(alignment > 0, "matmul tile alignment must be positive");
  return std::max(alignment, vecops::align_down(budget, alignment));
}

} // namespace details

template <Atom AtomT>
using AutomaticCacheTilingFor = CacheTiling<
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::M_R)>,
                  details::atom_alignment_v<decltype(AtomT::M_R)>>,
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::N_R)>,
                  details::atom_alignment_v<decltype(AtomT::N_R)>>,
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::K_R)>,
                  details::atom_alignment_v<decltype(AtomT::K_R)>>>;

/** Runtime cache-based default using fixed L1/L2 occupancy budgets. */
template <Atom AtomT>
struct DefaultTilingPolicy {
  using Result = AutomaticCacheTilingFor<AtomT>;

  VECOPS_INLINE static Result select(const platform::CacheInfo& cache) {
    const nint_t MR = static_cast<nint_t>(AtomT::M_R);
    const nint_t NR = static_cast<nint_t>(AtomT::N_R);
    const nint_t KR = static_cast<nint_t>(AtomT::K_R);
    const nint_t l1_budget = std::max<nint_t>(cache.l1d_bytes / 2, 1);
    const nint_t k_bytes =
        MR * static_cast<nint_t>(sizeof(typename AtomT::TA)) +
        NR * static_cast<nint_t>(sizeof(typename AtomT::TB));
    const nint_t kc = details::aligned_tile(l1_budget / k_bytes, KR);

    const nint_t l2_budget = std::max<nint_t>(
        cache.l2_bytes * 3 / 4, cache.line_bytes);
    const nint_t a_budget = l2_budget / 3;
    const nint_t b_budget = l2_budget - a_budget;
    const nint_t mc = details::aligned_tile(
        a_budget /
            (kc * static_cast<nint_t>(sizeof(typename AtomT::TA))),
        MR);
    const nint_t nc = details::aligned_tile(
        b_budget /
            (kc * static_cast<nint_t>(sizeof(typename AtomT::TB))),
        NR);
    return Result{
        typename Result::MTile{mc},
        typename Result::NTile{nc},
        typename Result::KTile{kc}};
  }
};

template <typename Config>
VECOPS_INLINE auto resolve_cache_tiling(const Config& config) {
  using Tuning = typename Config::GenericTuning;
  if constexpr (std::same_as<
                    typename Tuning::CacheTiling, AutomaticCacheTiling>) {
    return DefaultTilingPolicy<typename Config::Atom>::select(
        config.cache_info_provider());
  } else {
    return config.generic_tiled.cache_tiling;
  }
}

template <typename Config>
#if defined(ARCH_X86_FAMILY)
using backend_default_loop_order_t = loop_order::NKM;
#else
using backend_default_loop_order_t = loop_order::MKN;
#endif

template <typename Config>
using resolved_loop_order_t = std::conditional_t<
    std::same_as<typename Config::GenericTuning::LoopOrder,
                 loop_order::Automatic>,
    backend_default_loop_order_t<Config>,
    typename Config::GenericTuning::LoopOrder>;

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_TILING_H
