//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_SCALAR_BACKEND_H
#define VECOPS_EXECUTION_DETAILS_SCALAR_BACKEND_H

#include <utility>

#include "vecops/CoreDefs.h"
#include "vecops/execution/details/ResourceSet.h"
#include "vecops/platform/Target.h"

/** @file Backend.h @brief Stateless generic execution backend. */

namespace vecops::execution::details {

/**
 * @brief No-state execution backend used by explicit generic builds.
 *
 * Resource entry is a forced-inline callback invocation. Hardware
 * configuration is rejected at compile time rather than ignored.
 */
template <>
struct Backend<platform::GenericTarget> {
  using DefaultRequirements = ResourceSet<>;

  template <typename>
  /** Accept every empty/generic resource combination. */
  static consteval void validate() {}

  template <typename Current, typename Required, typename Fn>
  /** Invoke `fn` directly; generic targets own no hardware state. */
  VECOPS_ALWAYS_INLINE static decltype(auto) enter(Fn&& fn) {
    return std::forward<Fn>(fn)();
  }

  template <typename Resources, typename Configuration, typename Fn>
  /** Reject hardware configuration on a stateless backend. */
  VECOPS_ALWAYS_INLINE static decltype(auto) configure(
      const Configuration&, Fn&&) {
    static_assert(
        dependent_false_v<Configuration>,
        "scalar execution backend has no hardware configuration");
  }
};

} // namespace vecops::execution::details

#endif // VECOPS_EXECUTION_DETAILS_SCALAR_BACKEND_H
