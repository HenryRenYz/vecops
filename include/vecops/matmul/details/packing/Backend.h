// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/**
 * @file vecops/matmul/details/packing/Backend.h
 * @brief Dispatch surface of the matmul packing sublayer: the
 *        `Backend<Format, Implementation>` protocol and its assembly point.
 *
 * Every packing capability is a specialization of
 * `Backend<Format, Implementation>`, where Format comes from
 * `packing_t<Atom, Side>::FormatType` (amx::Format / sme::Format) and
 * Implementation is a tag from packing/Types.h (Vector, SME, ...).  Each
 * specialization provides the following protocol members:
 *
 * - `ResourceRequirements` — execution resources (streaming mode, ZA
 *   state, ...) under which the specialization's run() body must be
 *   compiled; the plan layer acquires them before calling.
 * - `eligible<InputSpec, OutputSpec>` — constexpr probe used by
 *   Plan.h::SelectedPackImplementation to walk its fallback chain.
 * - `run<Atom, Side>(scope, source, destination)` — pack `source` into
 *   `destination` following the block format of the Format.
 * - `supports_column_compensation` — whether the specialization offers
 *   `run_compensated<Atom>(scope, source, destination, compensation,
 *   a_zero_point)`, which packs signed-byte B and additionally emits the
 *   asymmetric-A column sidecar (see Plan.h::MatmulPackBCompensatedPlan).
 *
 * The primary template below is the "unavailable on this target" sentinel:
 * it fails eligibility probes and turns any direct run() call into a
 * static_assert.  The arch guards at the bottom include the amx/sme
 * specializations when their instruction sets are compiled in; this is the
 * same per-header arch-selection pattern used across the library.
 */

#ifndef VECOPS_MATMUL_DETAILS_PACK_BACKEND_H
#define VECOPS_MATMUL_DETAILS_PACK_BACKEND_H

#include "vecops/platform/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/packing/Types.h"

namespace vecops::kernel::matmul_pack_details {

/**
 * @brief "Unavailable on this target" sentinel for a Format/Implementation
 *        pair with no specialization.
 *
 * `eligible` is false so the plan-layer fallback chain skips it, and run()
 * is a compile-time dead end (`dependent_false_v` keeps the assert
 * unsatisfied until instantiation) for callers that bypass the chain.
 * See the file header for the full Backend protocol.
 */
template <typename Format, typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;
  static constexpr bool supports_column_compensation = false;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename... Args>
  VECOPS_ALWAYS_INLINE static void run(Args&&...) {
    static_assert(
        execution::details::dependent_false_v<Format, Implementation, Args...>,
        "matmul packing backend is unavailable for this target");
  }
};

} // namespace vecops::kernel::matmul_pack_details

// Arch-selected Backend specializations: exactly one of these provides the
// native implementations for each compiled-in instruction set.
#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/packing/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/packing/sme/Backend.h"
#endif

#endif // VECOPS_MATMUL_DETAILS_PACK_BACKEND_H
