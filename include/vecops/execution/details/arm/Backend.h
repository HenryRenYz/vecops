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
#if defined(HAS_SME)
#include "vecops/vec/details/sme/State.h"
#endif

/** @file vecops/execution/details/arm/Backend.h
 *  @brief AArch64 execution-state backend. */

namespace vecops::execution::details {

/**
 * @brief AArch64 execution backend for manual Streaming+ZA ownership.
 *
 * "Manual" means this backend owns the PSTATE.SM/ZA lifecycle itself: it
 * emits SMSTART/SMSTOP around the callback through
 * `vec::details::sme::with_streaming_za` (a lexical region), rather than
 * relying on compiler-generated streaming functions. A nested region whose
 * requirements are already active compiles to a plain callback invocation.
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
   * @pre `fn` must be `noexcept` when a Streaming+ZA transition is needed:
   *      an exception unwinding through the SMSTOP epilogue would leave the
   *      processor in streaming mode with ZA storage still enabled. This is
   *      enforced by static_assert, not caught at runtime.
   */
  VECOPS_ALWAYS_INLINE static decltype(auto) enter(Fn&& fn) {
    constexpr bool EnterStreamingZA =
        has_resource_v<arm::StreamingZA, Required> &&
        !has_resource_v<arm::StreamingZA, Current>;
    if constexpr (EnterStreamingZA) {
#if defined(HAS_SME)
      static_assert(std::is_nothrow_invocable_v<Fn&&>,
                    "manual SME region callback must be noexcept");
      return vec::details::sme::with_streaming_za(std::forward<Fn>(fn));
#else
      static_assert(!EnterStreamingZA,
                    "Arm StreamingZA requires an SME target");
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
