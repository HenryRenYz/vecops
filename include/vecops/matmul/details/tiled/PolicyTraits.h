//
// Copyright (c) vecops contributors.
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

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H
