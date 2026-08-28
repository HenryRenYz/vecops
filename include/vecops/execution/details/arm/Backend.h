//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_ARM_BACKEND_H
#define VECOPS_EXECUTION_DETAILS_ARM_BACKEND_H

#include <type_traits>
#include <utility>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/platform/Target.h"
#if defined(HAS_SME_FA64)
#include "vecops/vec/details/sme/State.h"
#endif

/** @file Backend.h @brief AArch64 execution-state backend. */

namespace vecops::execution::details {

/**
 * @brief AArch64 execution backend for manual Streaming+ZA ownership.
 */
template <>
struct Backend<platform::AArch64Target> {
  using DefaultRequirements = ResourceSet<>;

  template <typename Resources>
  /** Manual StreamingZA is the only ARM execution-state resource. */
  static consteval void validate() {}

  template <typename Current, typename Required, typename Fn>
  /**
   * @brief Enter Streaming+ZA only when newly required, then invoke `fn`.
   * @return The callback result.
   */
  VECOPS_ALWAYS_INLINE static decltype(auto) enter(Fn&& fn) {
    constexpr bool EnterStreamingZA =
        has_resource_v<arm::StreamingZA, Required> &&
        !has_resource_v<arm::StreamingZA, Current>;
    if constexpr (EnterStreamingZA) {
#if defined(HAS_SME_FA64)
      static_assert(std::is_nothrow_invocable_v<Fn&&>,
                    "manual SME region callback must be noexcept");
      return vec::details::sme::with_streaming_za(std::forward<Fn>(fn));
#else
      static_assert(!EnterStreamingZA,
                    "Arm StreamingZA requires an SME+FA64 target");
#endif
    } else {
      return std::forward<Fn>(fn)();
    }
  }

  template <typename Resources, typename Configuration, typename Fn>
  /** Emit a compile-time diagnostic for unsupported AArch64 configurations. */
  VECOPS_ALWAYS_INLINE static decltype(auto) configure(
      const Configuration&, Fn&&) {
    static_assert(
        dependent_false_v<Configuration>,
        "AArch64 execution configuration is not implemented");
  }
};

} // namespace vecops::execution::details

#endif // VECOPS_EXECUTION_DETAILS_ARM_BACKEND_H
