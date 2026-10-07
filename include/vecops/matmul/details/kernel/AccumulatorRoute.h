// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_ACCUMULATOR_ROUTE_H
#define VECOPS_MATMUL_DETAILS_KERNEL_ACCUMULATOR_ROUTE_H

#include <type_traits>

#include "vecops/CoreDefs.h"

namespace vecops::kernel::matmul_details {

/**
 * C-endpoint routing for a block that does not participate in split-K.
 *
 * Static routing deliberately has no data members: the backend can discard
 * accumulator operands and both endpoint decisions with `if constexpr`.
 */
template <bool UseAccumulatorInput, bool WriteAccumulatorOutput>
struct StaticAccumulatorRoute {
  static constexpr bool is_static = true;
  static constexpr bool use_accumulator_input = UseAccumulatorInput;
  static constexpr bool write_accumulator_output = WriteAccumulatorOutput;
};

/** One runtime type shared by first, middle, and last split-K blocks. */
struct DynamicAccumulatorRoute {
  static constexpr bool is_static = false;
  bool use_accumulator_input;
  bool write_accumulator_output;
};

template <typename Route>
inline constexpr bool static_accumulator_route_v =
    std::remove_cvref_t<Route>::is_static;

template <typename Route>
VECOPS_ALWAYS_INLINE constexpr bool route_uses_accumulator_input(
    const Route& route) {
  if constexpr (static_accumulator_route_v<Route>)
    return std::remove_cvref_t<Route>::use_accumulator_input;
  else
    return route.use_accumulator_input;
}

template <typename Route>
VECOPS_ALWAYS_INLINE constexpr bool route_writes_accumulator_output(
    const Route& route) {
  if constexpr (static_accumulator_route_v<Route>)
    return std::remove_cvref_t<Route>::write_accumulator_output;
  else
    return route.write_accumulator_output;
}

using UnsplitAccumulatorRoute = StaticAccumulatorRoute<false, false>;

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_KERNEL_ACCUMULATOR_ROUTE_H
