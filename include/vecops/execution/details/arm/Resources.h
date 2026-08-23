//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H
#define VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H

#include <utility>

#include "vecops/CoreDefs.h"

/**
 * @file Resources.h
 * @brief ARM streaming-mode resource tags and ACLE transition trampoline.
 */

namespace vecops::execution::details::arm {

/** Resource tag proving code in the scope executes with PSTATE.SM set. */
struct Streaming {};

/** Resource tag restricting an implementation to PSTATE.SM clear. */
struct NonStreaming {};

#if defined(HAS_ARM_LOCALLY_STREAMING)

/**
 * @brief Invoke a callback through an ACLE locally-streaming boundary.
 * @param fn Callback compiled in streaming context and invoked exactly once.
 * @return The callback result, preserving references and `void`.
 *
 * A plain C++ guard containing SMSTART is not sufficient: the compiler must
 * also compile the body as streaming code. This trampoline is intentionally
 * `noinline`; inlining a locally-streaming function into a non-streaming caller
 * is not a valid substitute for an ACLE mode boundary. One call per outer
 * resource region is preferred to annotating every operator and helper.
 */
template <typename Fn>
__arm_locally_streaming VECOPS_NOINLINE decltype(auto)
invoke_streaming(Fn&& fn) {
  return std::forward<Fn>(fn)();
}

#endif

} // namespace vecops::execution::details::arm

#endif // VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H
