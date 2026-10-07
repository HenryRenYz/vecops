// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_DETAILS_PARALLEL_H
#define VECOPS_EXECUTION_DETAILS_PARALLEL_H

#include "vecops/CoreTypes.h"
#include "vecops/runtime/CallAbi.h"

namespace vecops::execution {

struct ParallelContext;

namespace details {

/** Type-erased worker callback used by the compiled parallel backend. */
using ParallelBody = void (*)(void*, ParallelContext);

/**
 * @brief Execute @p body once on every worker in one synchronous team.
 *
 * OpenMP and its runtime API are deliberately confined to the out-of-line
 * implementation. Public kernel headers only instantiate a callback
 * trampoline and therefore do not expose backend-specific types or pragmas.
 */
void parallel_for_erased(nint_t requested_threads, void* object, ParallelBody body);

/** Execute exactly @p task_count logical tasks using an optional embedding pool. */
void parallel_tasks_erased(const VecopsThreadPoolV1* pool, nint_t task_count, void* object, ParallelBody body);

/** Validate and query a pool, falling back to the built-in backend when null. */
[[nodiscard]] nint_t max_parallelism(const VecopsThreadPoolV1* pool) noexcept;

} // namespace details
} // namespace vecops::execution

#endif // VECOPS_EXECUTION_DETAILS_PARALLEL_H
