//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H
#define VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H

#include <type_traits>

#include "vecops/matmul/Config.h"

namespace vecops::matmul::details {

template <typename Selection>
struct SelectedFamily;

template <>
struct SelectedFamily<family_selection::Automatic> {
  // Preserve the proven default machine code while WholeProblem is split into
  // individually enumerable architecture families.  Crucially, generic-tiled
  // tuning no longer changes this choice.
  using type = kernel_family::WholeProblem;
};

template <typename Family>
struct SelectedFamily<family_selection::Prefer<Family>> {
  using type = Family;
};

template <typename Family>
struct SelectedFamily<family_selection::Require<Family>> {
  using type = Family;
};

template <typename Config>
using selected_family_t =
    typename SelectedFamily<typename Config::FamilySelection>::type;

template <typename Config>
inline constexpr auto family_selection_mode_v =
    Config::FamilySelection::mode;

template <typename Config>
inline constexpr bool uses_generic_tiler_v =
    std::same_as<selected_family_t<Config>, kernel_family::GenericTiled>;

template <typename Config>
inline constexpr bool uses_whole_problem_v =
    std::same_as<selected_family_t<Config>, kernel_family::WholeProblem>;

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H
