//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_ARM_BACKEND_H
#define VECOPS_EXECUTION_DETAILS_ARM_BACKEND_H

#include <utility>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/platform/Target.h"

/** @file Backend.h @brief AArch64 execution-state backend. */

namespace vecops::execution::details {

/**
 * @brief AArch64 execution backend for streaming-mode resource ownership.
 *
 * Ordinary operators default to `arm::NonStreaming`. A region requesting
 * `arm::Streaming` is entered through the ACLE locally-streaming trampoline.
 * Combining both requirements is a compile-time error; no runtime mode test or
 * transition schedule is synthesized.
 */
template <>
struct Backend<platform::AArch64Target> {
  using DefaultRequirements = ResourceSet<arm::NonStreaming>;

  template <typename Resources>
  /** Reject a region containing mutually exclusive ARM mode requirements. */
  static consteval void validate() {
    static_assert(
        !(has_resource_v<arm::Streaming, Resources> &&
          has_resource_v<arm::NonStreaming, Resources>),
        "streaming and non-streaming-only operators cannot share a region");
  }

  template <typename Current, typename Required, typename Fn>
  /**
   * @brief Enter streaming mode only when newly required, then invoke `fn`.
   * @return The callback result.
   */
  VECOPS_ALWAYS_INLINE static decltype(auto) enter(Fn&& fn) {
    constexpr bool EnterStreaming =
        has_resource_v<arm::Streaming, Required> &&
        !has_resource_v<arm::Streaming, Current>;
    if constexpr (EnterStreaming) {
#if defined(HAS_ARM_LOCALLY_STREAMING)
      return arm::invoke_streaming(std::forward<Fn>(fn));
#else
      static_assert(!EnterStreaming,
                    "Arm streaming resource requires SME ACLE support");
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
