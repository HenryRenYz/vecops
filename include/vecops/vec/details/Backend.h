// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_BACKEND_H
#define VECOPS_VEC_DETAILS_BACKEND_H

#include "vecops/platform/Target.h"
#include "vecops/vec/Capabilities.h"

/**
 * @file Backend.h
 * @brief Map shared platform targets to vec representation backends.
 *
 * The aliases preserve existing vec specializations. Execution and kernel
 * modules map the same target identity to their own independent interfaces.
 */

namespace vecops::vec::details {

/** Compatibility alias for portable vec specializations. */
using ScalarBackend = platform::GenericTarget;
/** Compatibility alias for x86 vec specializations. */
using X86Backend = platform::X86Target;
/** Compatibility alias for SVE vec specializations. */
using SVEBackend = platform::AArch64Target;

#if defined(CPU_CAPABILITY_SVE)
/** Vec backend selected for the current translation unit. */
using CurrentBackend = SVEBackend;
#elif defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
using CurrentBackend = X86Backend;
#else
using CurrentBackend = ScalarBackend;
#endif

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_BACKEND_H
