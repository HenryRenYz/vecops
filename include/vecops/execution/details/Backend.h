//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_BACKEND_H
#define VECOPS_EXECUTION_DETAILS_BACKEND_H

#include "vecops/platform/Target.h"

/**
 * @file Backend.h
 * @brief Select the execution-state backend for the current platform target.
 *
 * Execution backends manage coarse hardware state such as ARM streaming mode
 * and Intel AMX tile ownership. They are independent of `vec` backends and are
 * only instantiated for the compile-time target; no runtime dispatch exists.
 */

namespace vecops::execution::details {
template <typename Target>
struct Backend;
} // namespace vecops::execution::details

#if defined(CPU_CAPABILITY_GENERIC)
#include "vecops/execution/details/scalar/Backend.h"
#elif defined(ARCH_X86_FAMILY)
#include "vecops/execution/details/x86/Backend.h"
#elif defined(ARCH_ARM_FAMILY)
#include "vecops/execution/details/arm/Backend.h"
#else
#include "vecops/execution/details/scalar/Backend.h"
#endif

namespace vecops::execution::details {
/** Execution-state backend selected for this translation unit. */
using CurrentBackend = Backend<platform::CurrentTarget>;
} // namespace vecops::execution::details

#endif // VECOPS_EXECUTION_DETAILS_BACKEND_H
