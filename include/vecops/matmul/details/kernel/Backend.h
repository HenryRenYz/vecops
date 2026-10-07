// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_BACKEND_H
#define VECOPS_MATMUL_DETAILS_BACKEND_H

/**
 * @file vecops/matmul/details/kernel/Backend.h
 * @brief Primary Backend template and the arch-gated backend assembly point.
 *
 * Design intent: this header is the normative statement of the kernel-side
 * backend contract. A matmul Implementation is only ever reached through
 * matmul_details::Backend<Implementation>, which supplies:
 *
 *  - ResourceRequirements: execution resources the implementation claims
 *    (x86 Tiles, arm StreamingZA, ...) so a scope can be configured before
 *    any leaf runs.
 *  - ProblemRank: rank of one leaf problem the backend consumes.
 *  - scratch_bytes(): scratch the microkernels expect the caller to pass to
 *    run/run_configured; the tiled layer allocates it from the workspace.
 *  - with_configuration<Atom, Policy, PackedB>(scope, m, n, fn): optional
 *    (detected via `requires`) hardware-state scope wrapping a group of
 *    leaves, e.g. one TILECFG image for AMX. Backends without a separate
 *    configuration simply do not define it.
 *  - run<Atom, Policy, ...> / run_configured<Atom, Policy>: whole-problem
 *    entry points, the latter for leaves already inside a configuration.
 *  - Catalog / EffectivePolicy / dispatch_plan / run_case: the hooks that
 *    TileScheduler.h drives for the generic tile traversal.
 *
 * The primary template doubles as the "backend unavailable" answer: run()
 * is a dependent-false static_assert trap instead of an undefined symbol,
 * so a misconfigured target fails with a message at instantiation rather
 * than a linker error. Concrete specializations are selected by the arch
 * guards below, which is the single assembly point for kernel backends
 * (same convention as the vec module's arch guard blocks).
 */

#include "vecops/platform/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Family.h"
#include "vecops/matmul/details/kernel/Types.h"

namespace vecops::kernel::matmul_details {

/// Backend contract holder (see the file header for the full contract).
/// The primary template is the "unavailable" implementation: a target only
/// gets a working backend when an arch guard below has included one.
template <typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 0; }

  template <typename... Args>
  VECOPS_ALWAYS_INLINE static void run(Args&&...) {
    // Dependent-false on the template parameters: the assertion fires only
    // at instantiation, so merely including this template never breaks an
    // unrelated target.
    static_assert(
        execution::details::dependent_false_v<Implementation, Args...>,
        "matrix-multiply backend is unavailable for this target");
  }
};

} // namespace vecops::kernel::matmul_details

#if defined(HAS_AMX_TILE)
#include "vecops/matmul/details/kernel/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/kernel/sme/Backend.h"
#endif

#endif // VECOPS_MATMUL_DETAILS_BACKEND_H
