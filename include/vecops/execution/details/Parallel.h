//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_PARALLEL_H
#define VECOPS_EXECUTION_DETAILS_PARALLEL_H

#include "vecops/CoreTypes.h"

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

} // namespace details
} // namespace vecops::execution

#endif // VECOPS_EXECUTION_DETAILS_PARALLEL_H
