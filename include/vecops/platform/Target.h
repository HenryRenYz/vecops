//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_PLATFORM_TARGET_H
#define VECOPS_PLATFORM_TARGET_H

#include "vecops/platform/Capabilities.h"

/**
 * @file Target.h
 * @brief Architecture-family identity shared by independent CPU subsystems.
 *
 * A target is deliberately not a vector or execution backend. The vec,
 * execution, transpose, and future modules map the same `current_target_t` to
 * their own backend interfaces without depending on one another.
 */

namespace vecops::platform {

/** Portable target selected by explicit generic builds or unknown hosts. */
struct GenericTarget {};

/** x86 architecture-family target; ISA level remains in capability macros. */
struct X86Target {};

/** AArch64 architecture-family target, including SVE/SME capability variants. */
struct AArch64Target {};

/** Architecture-family target selected for the current translation unit. */
#if defined(CPU_CAPABILITY_GENERIC)
using current_target_t = GenericTarget;
#elif defined(ARCH_X86_FAMILY)
using current_target_t = X86Target;
#elif defined(ARCH_ARM_FAMILY)
using current_target_t = AArch64Target;
#else
using current_target_t = GenericTarget;
#endif

} // namespace vecops::platform

#endif // VECOPS_PLATFORM_TARGET_H
