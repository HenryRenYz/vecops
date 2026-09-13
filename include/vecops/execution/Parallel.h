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
 * @brief Synchronous logical-task parallelism with a backend-neutral surface.
 *
 * `parallel_for()` opens one parallel region and invokes the supplied callback
 * exactly once on every worker that actually joins the team. The callback
 * receives its dense worker id and the actual team size, and owns loop
 * scheduling below that boundary. It is therefore the programmatic equivalent
 * of an OpenMP `parallel` region, not an OpenMP `for` work-sharing construct.
 *
 * The core has a serial fallback and accepts an embedding-owned pool through
 * `VecopsExecutionContext`. The optional OpenMP provider is a separate DSO, so
 * generated kernels do not acquire an OpenMP runtime dependency. Calls are
 * synchronous.
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

/** Logical-task identity whose task count is part of the C++ specialization. */
template <nint_t Parallelism>
struct StaticParallelContext {
  static_assert(Parallelism > 0, "static parallelism must be positive");
  nint_t thread_id;
  static constexpr nint_t num_threads = Parallelism;
};

/**
 * @brief Return the current call site's available parallelism.
 *
 * Outside a parallel region this is OpenMP's current maximum thread count.
 * Inside a parallel region it is one because nested vecops regions serialize.
 * A build without the optional OpenMP backend always returns one.
 */
[[nodiscard]] nint_t max_parallelism() noexcept;

/** Return the capacity of an embedding pool, or the built-in backend. */
[[nodiscard]] inline nint_t max_parallelism(const VecopsThreadPoolV1* pool) noexcept {
  return details::max_parallelism(pool);
}

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

/**
 * Execute exactly @p task_count dense logical tasks. Unlike `parallel_for`,
 * the callback count is independent of the number of physical workers.
 */
template <typename Fn>
void parallel_tasks(const VecopsThreadPoolV1* pool, nint_t task_count, Fn&& body) {
  using Body = std::remove_reference_t<Fn>;
  auto* object = const_cast<void*>(static_cast<const void*>(std::addressof(body)));
  details::parallel_tasks_erased(pool, task_count, object, [](void* erased, ParallelContext context) {
    std::invoke(*static_cast<Body*>(erased), context);
  });
}

/** Execute a compile-time fixed number of logical tasks. */
template <nint_t Parallelism, typename Fn>
void parallel_tasks(const VecopsThreadPoolV1* pool, Fn&& body) {
  static_assert(Parallelism > 0, "static parallelism must be positive");
  using Body = std::remove_reference_t<Fn>;
  auto* object = const_cast<void*>(static_cast<const void*>(std::addressof(body)));
  details::parallel_tasks_erased(pool, Parallelism, object, [](void* erased, ParallelContext context) {
    std::invoke(*static_cast<Body*>(erased), StaticParallelContext<Parallelism>{context.thread_id});
  });
}

/** One compile-time-sized contiguous shard. */
template <nint_t Extent>
struct StaticShard {
  static_assert(Extent > 0, "a static shard must be non-empty");
  nint_t begin;
  static constexpr nint_t extent = Extent;
  [[nodiscard]] constexpr nint_t end() const noexcept {
    return begin + Extent;
  }
};

/**
 * Split a compile-time extent across a compile-time task count. The callback
 * is instantiated for at most the main and tail extents; empty shards vanish.
 */
template <nint_t Total, nint_t Parallelism, typename Fn>
void balanced_shard(StaticParallelContext<Parallelism> worker, Fn&& body) {
  static_assert(Total >= 0, "static work extent must be non-negative");
  constexpr nint_t smaller = Total / Parallelism;
  constexpr nint_t larger_tasks = Total % Parallelism;
  if constexpr (larger_tasks > 0) {
    if (worker.thread_id < larger_tasks) {
      constexpr nint_t extent = smaller + 1;
      std::invoke(std::forward<Fn>(body), StaticShard<extent>{worker.thread_id * extent});
      return;
    }
  }
  if constexpr (smaller > 0) {
    const nint_t begin = larger_tasks * (smaller + 1) + (worker.thread_id - larger_tasks) * smaller;
    std::invoke(std::forward<Fn>(body), StaticShard<smaller>{begin});
  }
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_PARALLEL_H
