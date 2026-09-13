//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_TILING_H
#define VECOPS_MATMUL_TILING_H

#include <algorithm>
#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/platform/CacheInfo.h"
#include "vecops/util/Math.h"

namespace vecops::matmul {

/**
 * @file vecops/matmul/Tiling.h
 * @brief Resolve `AutomaticCacheTiling` and `Automatic` loop order into
 *        concrete tile sizes and traversal order.
 *
 * Key components:
 *
 * | Component                  | Purpose                                        |
 * |----------------------------|------------------------------------------------|
 * | `AutomaticCacheTilingFor`  | Atom-aligned `CacheTiling` type factory        |
 * | `DefaultTilingPolicy`      | L1/L2-occupancy budget model                   |
 * | `resolve_cache_tiling`     | Automatic vs explicit tiling dispatch          |
 * | `backend_default_loop_order_t` / `resolved_loop_order_t` | Loop-order resolution |
 *
 * Users trigger this resolution implicitly by configuring
 * `AutomaticCacheTiling` / `loop_order::Automatic` in `GenericTiledTuning`;
 * the tiled driver (`matmul/details/tiled/Tiler.h`) is the only consumer.
 */

namespace details {

/** Native tile granularity of one Atom extent, in elements.
 *  A singleton extent contributes its unique value (e.g. AMX's 16 rows),
 *  whether represented by `Const` or a singleton `Dynamic`. A genuinely
 *  varying extent (such as SME's VL-scaled extents) contributes only its
 *  alignment constraint. */
template <typename T>
inline constexpr nint_t atom_alignment_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (meta::is_singleton_v<V>)
    return meta::singleton_value_v<V>;
  else return V::alignment;
}();

/** Round `budget` down to a multiple of `alignment` — but never below one
 *  full aligned block: when the budget cannot afford even one block, the
 *  `std::max` floor keeps the tile at `alignment`. Removing the floor would
 *  produce zero-sized tiles (and infinite tiling loops) on tiny caches. */
VECOPS_INLINE nint_t aligned_tile(nint_t budget, nint_t alignment) {
  VECOPS_ASSERT(alignment > 0, "matmul tile alignment must be positive");
  return std::max(alignment, vecops::align_down(budget, alignment));
}

} // namespace details

/**
 * @brief `CacheTiling` type whose per-axis tiles are runtime values aligned
 *        to the Atom's native extents.
 *
 * Every axis becomes `Dynamic<align, align>`: the tile is chosen at runtime
 * (by `DefaultTilingPolicy`), constrained to a multiple of the Atom's
 * M_R/N_R/K_R, with the lower bound equal to the alignment so at least one
 * native block always fits.
 *
 * @tparam AtomT  Atom whose register-block extents define the alignments.
 */
template <Atom AtomT>
using AutomaticCacheTilingFor = CacheTiling<
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::M_R)>,
                  details::atom_alignment_v<decltype(AtomT::M_R)>>,
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::N_R)>,
                  details::atom_alignment_v<decltype(AtomT::N_R)>>,
    meta::Dynamic<details::atom_alignment_v<decltype(AtomT::K_R)>,
                  details::atom_alignment_v<decltype(AtomT::K_R)>>>;

/**
 * @brief Runtime cache-based default using fixed L1/L2 occupancy budgets.
 *
 * Budget model (heuristic, tuned by measurement):
 * - `kc` first: half of L1d is split between one A row-block and one B
 *   row-block of K elements (`k_bytes` per K element pair), rounded down to
 *   a multiple of K_R.
 * - 3/4 of L2 is then split ~1/3 for A panels and ~2/3 for B panels — the
 *   A/B asymmetry leaves B (the shared/streamed operand) the larger share.
 *   `mc`/`nc` are the element counts fitting those budgets at the chosen
 *   `kc`, each aligned to M_R/N_R.
 *
 * The result is returned in the `AutomaticCacheTilingFor<AtomT>` shape so
 * it satisfies the alignment constraints by construction.
 */
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

/** Resolve the configured cache tiling: `AutomaticCacheTiling` runs the
 *  `DefaultTilingPolicy` budget model with the config's cache info; any
 *  explicitly configured `CacheTiling` is returned unchanged. */
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

// The unused `Config` template parameter exists only so this alias and
// `resolved_loop_order_t` below share one spelling that participates in
// dependent-name lookup; the value is arch-selected, never Config-selected.
// The arch split itself is a measured per-backend choice: x86 (AMX)
// defaults to NKM, every other backend to MKN; TilerTest pins both.
template <typename Config>
#if defined(ARCH_X86_FAMILY)
using backend_default_loop_order_t = loop_order::NKM;
#else
using backend_default_loop_order_t = loop_order::MKN;
#endif

/** Resolve the configured loop order: `loop_order::Automatic` maps to the
 *  backend default above; explicit orders pass through unchanged. */
template <typename Config>
using resolved_loop_order_t = std::conditional_t<
    std::same_as<typename Config::GenericTuning::LoopOrder,
                 loop_order::Automatic>,
    backend_default_loop_order_t<Config>,
    typename Config::GenericTuning::LoopOrder>;

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_TILING_H
