//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H
#define VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H

/**
 * @file vecops/matmul/details/planning/FamilySelector.h
 * @brief Resolves Config::FamilySelection into the single family type that
 *        plans and executes the operation.
 *
 * Selection is resolved exactly once, here, before any planning happens, so
 * that family-local tuning parameters stored in the Config (generic-tiled
 * cache tiling, loop order, packing policy) cannot influence which family
 * runs.  The companion helpers expose the selection mode and the two family
 * identities that downstream planning branches on.
 */

#include <type_traits>

#include "vecops/matmul/Config.h"

namespace vecops::matmul::details {

/// Maps a family_selection policy to the family type to plan with.
template <typename Selection>
struct SelectedFamily;

template <>
struct SelectedFamily<family_selection::Automatic> {
  // Preserve the architecture route while WholeProblem is split into
  // individually enumerable leaves. Instantiating ArchitectureFamily and
  // GenericTiled side by side at every call site nearly doubles matmul text;
  // a future cross-family runtime selector must call a backend/atom-specific,
  // type-erased GenericTiled ABI leaf instead. Generic-tiled tuning therefore
  // does not implicitly change this choice.
  using type = kernel_family::WholeProblem;
};

/// A preferred family is kept as the selection; its per-rank applicability
/// (and fallback) is enforced later through FamilyDispatch in the invocation.
template <typename Family>
struct SelectedFamily<family_selection::Prefer<Family>> {
  using type = Family;
};

/// A required family is kept as the selection; an inapplicable one is a
/// configuration error, asserted where the leaf problem shape is known.
template <typename Family>
struct SelectedFamily<family_selection::Require<Family>> {
  using type = Family;
};

/// The single family type selected for a matmul Config.
template <typename Config>
using selected_family_t =
    typename SelectedFamily<typename Config::FamilySelection>::type;

/// Selection mode carried into the invocation so backend leaves can enforce
/// require semantics.
template <typename Config>
inline constexpr auto family_selection_mode_v =
    Config::FamilySelection::mode;

/// True when the plan is the explicitly selected generic cache-tiled family.
template <typename Config>
inline constexpr bool uses_generic_tiler_v =
    std::same_as<selected_family_t<Config>, kernel_family::GenericTiled>;

/// True when the plan is the composite automatic whole-problem family.
template <typename Config>
inline constexpr bool uses_whole_problem_v =
    std::same_as<selected_family_t<Config>, kernel_family::WholeProblem>;

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILY_SELECTOR_H
