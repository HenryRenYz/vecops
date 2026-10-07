// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H
#define VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H

#include "vecops/matmul/Config.h"

/**
 * @file vecops/matmul/details/tiled/PolicyTraits.h
 * @brief Compile-time resolution of packing-policy `automatic` defaults
 *        against the resolved loop order.
 *
 * The generic tiler's packing knobs can be spelled `automatic`; those
 * spellings are resolved here, once, from the loop order -- the one piece of
 * context that determines whether a whole-K packed copy is worth its memory.
 * Everything evaluates at compile time, so the tiled loop nest never sees an
 * unresolved `automatic`.
 */

namespace vecops::matmul::details {

/// Position of `Target` in the traversal order: 0 = outermost, 2 = innermost.
template <Axis Target, typename Order>
inline constexpr int axis_position_v = [] {
  if constexpr (Order::first == Target) return 0;
  else if constexpr (Order::second == Target) return 1;
  else return 2;
}();

/// Whether `Target` is traversed outside the K loop.
template <Axis Target, typename Order>
inline constexpr bool axis_precedes_k_v =
    axis_position_v<Target, Order> < axis_position_v<Axis::K, Order>;

/** Spatial Tile2D visitation follows the relative M/N cache-loop order. */
template <typename Order>
inline constexpr auto spatial_traversal_order_v =
    axis_position_v<Axis::M, Order> < axis_position_v<Axis::N, Order>
        ? kernel::loop::Tile2DTraversalOrder::m_major
        : kernel::loop::Tile2DTraversalOrder::n_major;

/** Axis along which an operand is reused: A across N, B across M. */
template <Operand Side>
inline constexpr Axis reuse_axis_v =
    Side == Operand::A ? Axis::N : Axis::M;

/**
 * Whether a bounded spatial x KC copy can be placed immediately outside the
 * reuse loop. This is possible exactly when that reuse loop is innermost:
 * the other two axes have then selected all coordinates the operand needs.
 */
template <Operand Side, typename Order>
inline constexpr bool has_panel_lifetime_site_v =
    axis_position_v<reuse_axis_v<Side>, Order> == 2;

template <typename Placement>
inline constexpr bool internal_packing_disabled_v =
    Placement::allowed_sites == PackingSite::none ||
    Placement::prepared_input == PreparedInputRequirement::required;

template <typename Placement>
inline constexpr bool allows_inside_packing_v =
    !internal_packing_disabled_v<Placement> && allows_packing_site(
        Placement::allowed_sites, PackingSite::inside_reuse_loop);

template <typename Placement>
inline constexpr bool allows_outside_packing_v =
    !internal_packing_disabled_v<Placement> && allows_packing_site(
        Placement::allowed_sites, PackingSite::outside_reuse_loop);

/**
 * Bridge the legacy GenericTiled packing choice into the top-level placement
 * contract. The old mode remains the profitability heuristic for optional
 * packing; `required` upgrades it to `always`, while disabled/prepared-only
 * policies force `never`.
 */
template <typename LegacyPolicy, typename Placement>
inline constexpr PackingMode resolved_internal_packing_mode_v = [] {
  if constexpr (internal_packing_disabled_v<Placement>)
    return PackingMode::never;
  else if constexpr (
      Placement::requirement == PackingRequirement::required)
    return PackingMode::always;
  else
    return LegacyPolicy::mode;
}();

/**
 * Resolve the copy extent while respecting placement restrictions.
 * Explicit legacy extents remain source-compatible unless they name an
 * impossible site. For an automatic extent, an outside copy uses the bounded
 * panel lifetime when the reuse axis is innermost and otherwise becomes a
 * whole-operand copy; inside-only always remains a cache panel.
 */
template <typename LegacyPolicy, typename Placement,
          Operand Side, typename Order>
inline constexpr PackingExtent resolved_internal_packing_extent_v = [] {
  if constexpr (internal_packing_disabled_v<Placement>) {
    return PackingExtent::cache_k;
  } else if constexpr (!allows_outside_packing_v<Placement>) {
    return PackingExtent::cache_k;
  } else if constexpr (
      !allows_inside_packing_v<Placement> &&
      !has_panel_lifetime_site_v<Side, Order>) {
    // Outside-only has no bounded panel site unless the reuse axis is
    // innermost. Promote even an explicit cache_k spelling to the only
    // realizable outside representation: one whole-operand copy.
    return PackingExtent::full_k;
  } else if constexpr (LegacyPolicy::extent != PackingExtent::automatic) {
    return LegacyPolicy::extent;
  } else if constexpr (has_panel_lifetime_site_v<Side, Order>) {
    return PackingExtent::cache_k;
  } else {
    return PackingExtent::full_k;
  }
}();

/** Concrete legacy-shaped policy consumed by OperandController. */
template <typename LegacyPolicy, typename Placement,
          Operand Side, typename Order>
using ResolvedInternalPackingPolicy = PackingPolicy<
    resolved_internal_packing_mode_v<LegacyPolicy, Placement>,
    resolved_internal_packing_extent_v<
        LegacyPolicy, Placement, Side, Order>>;

/**
 * @brief Map a logical M/N block origin into split-K accumulator storage.
 *
 * An output-backed accumulator is a full logical M x N view and always keeps
 * the real origin. A workspace accumulator may collapse an axis that precedes
 * K to one reusable stripe; only that representation rebases the origin to
 * zero. Keeping this rule in one helper prevents accumulator placement and
 * Tiler block narrowing from independently interpreting the loop order.
 */
template <bool OutputAcc, Axis Target, typename Order>
VECOPS_INLINE constexpr nint_t accumulator_block_origin(
    nint_t logical_origin) {
  if constexpr (OutputAcc) return logical_origin;
  else if constexpr (axis_precedes_k_v<Target, Order>) return 0;
  else return logical_origin;
}

/**
 * @brief Resolve a `PackingExtent` (honoring an explicit choice) for one
 *        operand under the given loop order.
 *
 * Automatic resolution chain, per operand side:
 *
 * 1. The operand's spatial axis is M for A, N for B.
 * 2. If the spatial axis precedes K, the nest completes one spatial block per
 *    K sweep (each spatial block's full K accumulation finishes before the
 *    next spatial block starts), so a per-KC-block packed copy -- rebuilt each
 *    K step, bounded by one panel stripe -- is `cache_k`: a whole-K packed
 *    copy would instead have to cover the entire spatial axis at once.
 * 3. If K precedes the spatial axis, every spatial block revisits the same K
 *    blocks, so packing the operand's full K extent once and hoisting the
 *    packed copy out of the whole traversal (`full_k`) amortizes the packing
 *    work instead of repeating it per block.
 */
template <typename Policy, Operand Side, typename Order>
inline constexpr PackingExtent resolved_packing_extent_v = [] {
  if constexpr (Policy::extent != PackingExtent::automatic) {
    return Policy::extent;
  } else {
    constexpr Axis Spatial = Side == Operand::A ? Axis::M : Axis::N;
    return axis_position_v<Spatial, Order> < axis_position_v<Axis::K, Order>
        ? PackingExtent::cache_k : PackingExtent::full_k;
  }
}();

/// Whether one operand's packing is hoisted to cover the whole K extent.
template <typename Policy, Operand Side, typename Order>
inline constexpr bool full_k_packing_v =
    resolved_packing_extent_v<Policy, Side, Order> == PackingExtent::full_k;

template <typename Policy, typename Placement,
          Operand Side, typename Order>
inline constexpr bool uses_panel_lifetime_packing_v =
    allows_outside_packing_v<Placement> &&
    has_panel_lifetime_site_v<Side, Order> &&
    resolved_packing_extent_v<Policy, Side, Order> == PackingExtent::cache_k;

/**
 * A whole-operand representation must be selected statically: explicit
 * full-K, legacy `always`, or a mandatory outside placement. Optional
 * automatic cases fall back to an allowed bounded panel rather than
 * generating raw/packed representations behind a runtime cache-size branch.
 */
template <typename LegacyPolicy, typename Placement,
          typename Policy, Operand Side, typename Order>
inline constexpr bool uses_whole_operand_packing_v =
    allows_outside_packing_v<Placement> &&
    full_k_packing_v<Policy, Side, Order> &&
    (LegacyPolicy::extent == PackingExtent::full_k ||
     LegacyPolicy::mode == PackingMode::always ||
     Placement::requirement == PackingRequirement::required);

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H
