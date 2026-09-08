//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_PARALLEL_H
#define VECOPS_EXECUTION_PARALLEL_H

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/execution/details/Parallel.h"

/**
 * @file vecops/execution/Parallel.h
 * @brief Synchronous worker-team parallelism with a backend-neutral surface.
 *
 * `parallel_for()` opens one parallel region and invokes the supplied callback
 * exactly once on every worker that actually joins the team. The callback
 * receives its dense worker id and the actual team size, and owns loop
 * scheduling below that boundary. It is therefore the programmatic equivalent
 * of an OpenMP `parallel` region, not an OpenMP `for` work-sharing construct.
 *
 * The initial backend uses OpenMP, but no OpenMP header, type, or pragma is
 * exposed here. Calls are synchronous: the callback object and all referenced
 * state need only remain alive until `parallel_for()` returns.
 *
 * ## Usage
 *
 * @code
 * const auto workers = std::min(num_tasks, execution::max_parallelism());
 * execution::parallel_for(workers, [&](execution::ParallelContext context) {
 *   for (nint_t task = context.thread_id; task < num_tasks;
 *        task += context.num_threads) {
 *     run_task(task);
 *   }
 * });
 * @endcode
 *
 * ## Semantics and pitfalls
 *
 * - `requested_threads <= 0` selects the backend default.
 * - The actual `context.num_threads` may be smaller than requested; scheduling
 *   must always use the value supplied to the callback.
 * - Nested calls are serialized and receive context `{0, 1}` to avoid
 *   oversubscription.
 * - One callback object is invoked concurrently by all workers. Mutable shared
 *   captures require caller-provided synchronization.
 * - Exceptions are collected by the backend and the first one is rethrown
 *   after every worker has left the parallel region.
 */

namespace vecops::execution {

/** @brief Identity of one worker in the current parallel region. */
struct ParallelContext {
  /** Dense worker id in `[0, num_threads)`. */
  nint_t thread_id;
  /** Actual number of workers participating in this region. */
  nint_t num_threads;
};

/**
 * @brief Return the current call site's available parallelism.
 *
 * Outside a parallel region this is OpenMP's current maximum thread count.
 * Inside a parallel region it is one because nested vecops regions serialize.
 * A build without the optional OpenMP backend always returns one.
 */
[[nodiscard]] nint_t max_parallelism() noexcept;

/**
 * @brief Open a synchronous worker team and invoke @p body once per worker.
 * @param requested_threads Desired team size; non-positive uses the backend
 *                          default and positive values are capped by it.
 * @param body Thread-safe worker callback accepting `ParallelContext`.
 */
template <typename Fn>
void parallel_for(nint_t requested_threads, Fn&& body) {
  using Body = std::remove_reference_t<Fn>;
  auto* object = const_cast<void*>(static_cast<const void*>(std::addressof(body)));
  details::parallel_for_erased(requested_threads, object, [](void* erased, ParallelContext context) {
    std::invoke(*static_cast<Body*>(erased), context);
  });
}

/** @brief Open a synchronous worker team using the backend thread default. */
template <typename Fn>
void parallel_for(Fn&& body) {
  parallel_for(nint_t{0}, std::forward<Fn>(body));
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_PARALLEL_H
