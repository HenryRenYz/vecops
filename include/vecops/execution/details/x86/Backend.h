//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_X86_BACKEND_H
#define VECOPS_EXECUTION_DETAILS_X86_BACKEND_H

#include <utility>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/x86/Resources.h"
#include "vecops/platform/Target.h"

/** @file vecops/execution/details/x86/Backend.h
 *  @brief x86 AMX execution-state backend. */

namespace vecops::execution::details {

/**
 * @brief x86 execution backend for Intel AMX ownership and configuration.
 *
 * Entering a new `x86::Tiles` resource installs one lexical TILERELEASE guard.
 * TILECFG is loaded separately by `configure()`, allowing one configuration to
 * cover a complete micro-kernel loop instead of every tile invocation.
 */
template <>
struct Backend<platform::X86Target> {
  using DefaultRequirements = ResourceSet<>;

  template <typename>
  /** x86 resource tags currently have no mutually exclusive combination. */
  static consteval void validate() {}

  template <typename Current, typename Required, typename Fn>
  /**
   * @brief Acquire AMX tile ownership if it is not already active.
   * @return The callback result; TILERELEASE runs after normal callback return.
   */
  VECOPS_ALWAYS_INLINE static decltype(auto) enter(Fn&& fn) {
    constexpr bool EnterTiles =
        has_resource_v<x86::Tiles, Required> &&
        !has_resource_v<x86::Tiles, Current>;
    if constexpr (EnterTiles) {
#if defined(HAS_AMX_TILE)
      x86::TileReleaseGuard release;
      return std::forward<Fn>(fn)();
#else
      static_assert(!EnterTiles, "AMX tile resource requires an AMX target");
#endif
    } else {
      return std::forward<Fn>(fn)();
    }
  }

  template <typename Resources, typename Configuration, typename Fn>
    requires requires(const Configuration& configuration) {
      typename Configuration::RequiredResource;
      configuration.load();
    }
  /**
   * @brief Load one configuration under its required active resource.
   * @param configuration Object exposing `RequiredResource` and `load()`.
   * @param fn Callback invoked once after the load.
   * @return The callback result.
   */
  VECOPS_ALWAYS_INLINE static decltype(auto) configure(
      const Configuration& configuration, Fn&& fn) {
    using Resource = typename Configuration::RequiredResource;
    static_assert(
        has_resource_v<Resource, Resources>,
        "configuration resource is not active in this execution scope");
    configuration.load();
    return std::forward<Fn>(fn)();
  }
};

} // namespace vecops::execution::details

#endif // VECOPS_EXECUTION_DETAILS_X86_BACKEND_H
