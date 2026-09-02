//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H
#define VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H

#include "vecops/matmul/Config.h"

namespace vecops::matmul::details {

template <Axis Target, typename Order>
inline constexpr int axis_position_v = [] {
  if constexpr (Order::first == Target) return 0;
  else if constexpr (Order::second == Target) return 1;
  else return 2;
}();

template <Axis Target, typename Order>
inline constexpr bool axis_precedes_k_v =
    axis_position_v<Target, Order> < axis_position_v<Axis::K, Order>;

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

template <typename Policy, Operand Side, typename Order>
inline constexpr bool full_k_packing_v =
    resolved_packing_extent_v<Policy, Side, Order> == PackingExtent::full_k;

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_POLICY_TRAITS_H
