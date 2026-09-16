//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_PARALLEL_H
#define VECOPS_EXECUTION_PARALLEL_H

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/Meta.h"
#include "vecops/execution/TaskPartition.h"
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
 * Identity of one logical lane in a compile-time-sized execution region.
 *
 * A logical lane is not an operating-system thread. An embedding executor may
 * run several lanes sequentially on one physical worker. Keeping that
 * distinction explicit lets kernels allocate one stable scratch replica per
 * lane without depending on a backend-specific worker identity.
 */
template <nint_t Parallelism>
class TaskContext {
public:
  static_assert(Parallelism > 0, "static parallelism must be positive");

  explicit constexpr TaskContext(nint_t lane) noexcept
    : lane_(lane) {
  }

  /** Dense logical lane id in `[0, lane_count())`. */
  [[nodiscard]] constexpr nint_t lane_id() const noexcept {
    return lane_;
  }

  /** Logical lane count carried by this C++ specialization. */
  [[nodiscard]] static constexpr nint_t lane_count() noexcept {
    return Parallelism;
  }

  /**
   * Select this lane from the leading replica axis of a bound worker tensor.
   *
   * The tensor-aware definition lives with the typed-workspace layer so this
   * backend-neutral header does not acquire a dependency on Tensor. Rank and
   * replica-count contracts are checked there.
   */
  template <typename WorkerTensor>
  [[nodiscard]] constexpr decltype(auto) local(WorkerTensor&& tensor) const;

private:
  nint_t lane_;
};

/**
 * One chunk of a one-dimensional range operation.
 *
 * Program ids are dense over the logical range and independent from lane ids:
 * one lane may execute zero, one, or several work items sequentially.
 */
template <meta::ValueType Chunk = meta::Any, meta::ValueType ProgramCount = meta::Any>
class RangeWorkItem {
public:
  using chunk_type = Chunk;
  using program_count_type = ProgramCount;

  constexpr RangeWorkItem(nint_t program, ProgramCount programs, nint_t first, nint_t last,
                          Chunk chunk) noexcept
    : program_(program)
    , programs_(programs)
    , first_(first)
    , last_(last)
    , chunk_(chunk) {
  }

  [[nodiscard]] constexpr nint_t program_id() const noexcept {
    return program_;
  }

  [[nodiscard]] constexpr nint_t program_count() const noexcept {
    return static_cast<nint_t>(programs_);
  }

  /** Program count with the range expression's retained Meta value type. */
  [[nodiscard]] constexpr ProgramCount program_count_value() const noexcept {
    return programs_;
  }

  [[nodiscard]] constexpr nint_t begin() const noexcept {
    return first_;
  }

  [[nodiscard]] constexpr nint_t end() const noexcept {
    return last_;
  }

  [[nodiscard]] constexpr nint_t extent() const noexcept {
    return last_ - first_;
  }

  /** Requested chunk size, preserving `Const<N>` when supplied by the caller. */
  [[nodiscard]] constexpr Chunk chunk() const noexcept {
    return chunk_;
  }

  /** Whether this item covers a complete chunk rather than the clipped tail. */
  [[nodiscard]] constexpr bool is_full() const noexcept {
    return extent() == static_cast<nint_t>(chunk_);
  }

private:
  nint_t program_;
  ProgramCount programs_;
  nint_t first_;
  nint_t last_;
  Chunk chunk_;
};

/** Short names for APIs that treat a range work item as a tile/program. */
using WorkItem = RangeWorkItem<>;
using Tile = RangeWorkItem<>;

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

/**
 * Execute exactly @p Parallelism logical lanes.
 *
 * This is the lane-named replacement for static `parallel_tasks`. It preserves
 * the V1 thread-pool contract: the backend receives exactly `Parallelism`
 * logical tasks, regardless of how many physical workers it owns.
 */
template <nint_t Parallelism, typename Fn>
void parallel_lanes(const VecopsThreadPoolV1* pool, Fn&& body) {
  static_assert(Parallelism > 0, "static parallelism must be positive");
  constexpr auto partition = plan_task_partition(meta::cint<Parallelism>, meta::cint<Parallelism>);
  static_assert(decltype(partition.active_lane_count)::value == Parallelism);
  static_assert(decltype(partition.small_task_count)::value == 1);
  static_assert(decltype(partition.large_lane_count)::value == 0);
  using Body = std::remove_reference_t<Fn>;
  auto* object = const_cast<void*>(static_cast<const void*>(std::addressof(body)));
  details::parallel_tasks_erased(pool, static_cast<nint_t>(partition.active_lane_count), object,
                                 [](void* erased, ParallelContext context) {
                                   std::invoke(*static_cast<Body*>(erased),
                                               TaskContext<Parallelism>{context.thread_id});
                                 });
}

/** Execute compile-time-fixed logical lanes using the built-in backend. */
template <nint_t Parallelism, typename Fn>
void parallel_lanes(Fn&& body) {
  parallel_lanes<Parallelism>(nullptr, std::forward<Fn>(body));
}

/**
 * Execute `[begin, end)` as dense fixed-size programs on static logical lanes.
 *
 * At most `Parallelism` logical tasks are submitted to the thread-pool ABI;
 * when the range contains fewer programs, empty lanes are omitted. Each
 * active lane receives a balanced contiguous interval of program ids and runs
 * its programs sequentially. Consequently, per-lane scratch remains valid
 * even when an embedding maps several lanes onto one physical worker.
 *
 * The callback receives `(TaskContext<Parallelism>, RangeWorkItem)`. The final
 * item is clipped to @p end and may be shorter than @p chunk. Full and tail
 * items deliberately have the same C++ type and callback instantiation.
 */
namespace details {

template <nint_t Parallelism, meta::ValueType ActiveLaneCount, typename Fn>
void parallel_active_lanes(const VecopsThreadPoolV1* pool, ActiveLaneCount active_lane_count, Fn&& body) {
  if constexpr (ActiveLaneCount::is_const) {
    static_assert(ActiveLaneCount::value >= 0 && ActiveLaneCount::value <= Parallelism,
                  "active lane count must fit the static parallelism");
    if constexpr (ActiveLaneCount::value == 0)
      return;
  } else {
    const nint_t active = static_cast<nint_t>(active_lane_count);
    if (active < 0 || active > Parallelism)
      throw std::invalid_argument("active lane count must fit the static parallelism");
    if (active == 0)
      return;
  }

  using Body = std::remove_reference_t<Fn>;
  auto* object = const_cast<void*>(static_cast<const void*>(std::addressof(body)));
  parallel_tasks_erased(pool, static_cast<nint_t>(active_lane_count), object,
                        [](void* erased, ParallelContext context) {
                          std::invoke(*static_cast<Body*>(erased), TaskContext<Parallelism>{context.thread_id});
                        });
}

template <nint_t Parallelism, meta::ValueType Chunk, meta::ValueType ProgramCount, typename Fn>
void parallel_range(const VecopsThreadPoolV1* pool, nint_t begin, nint_t end, Chunk chunk,
                    ProgramCount program_count, Fn&& body) {
  const nint_t chunk_size = static_cast<nint_t>(chunk);
  const auto partition = plan_task_partition(meta::cint<Parallelism>, program_count);
  parallel_active_lanes<Parallelism>(pool, partition.active_lane_count, [&](TaskContext<Parallelism> task) {
    using LaneIndex = meta::Dynamic<1, 0, Parallelism - 1>;
    visit_task_partition_shard(partition, LaneIndex{task.lane_id()}, [&](const auto& shard) {
      const nint_t first_program = static_cast<nint_t>(shard.begin);
      const nint_t lane_programs = static_cast<nint_t>(shard.task_count);
      for (nint_t local_program = 0; local_program < lane_programs; ++local_program) {
        const nint_t program = first_program + local_program;
        const nint_t first = begin + program * chunk_size;
        const nint_t last = std::min(first + chunk_size, end);
        std::invoke(body, task,
                    RangeWorkItem<Chunk, ProgramCount>{program, program_count, first, last, chunk});
      }
    });
  });
}

} // namespace details

/** Runtime-integer range form retained for source compatibility. */
template <nint_t Parallelism, typename Fn>
void parallel_for(const VecopsThreadPoolV1* pool, nint_t begin, nint_t end, nint_t chunk, Fn&& body) {
  static_assert(Parallelism > 0, "static parallelism must be positive");
  if (chunk <= 0)
    throw std::invalid_argument("vecops parallel_for chunk must be positive");
  if (end < begin)
    throw std::invalid_argument("vecops parallel_for end precedes begin");

  const nint_t distance = end - begin;
  const nint_t programs = distance / chunk + (distance % chunk != 0 ? 1 : 0);
  details::parallel_range<Parallelism>(pool, begin, end, meta::Any{chunk}, meta::Any{programs},
                                       std::forward<Fn>(body));
}

/**
 * Meta range form. Raw integer arguments normalize to `Any`, while Const and
 * Dynamic inputs retain their constraints. At least one Meta argument is
 * required so all-raw-integer calls continue to select the compatibility form.
 */
template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Fn>
  requires (meta::ValueType<std::remove_cvref_t<Begin>> || meta::ValueType<std::remove_cvref_t<End>> ||
            meta::ValueType<std::remove_cvref_t<Chunk>>)
void parallel_for(const VecopsThreadPoolV1* pool, Begin&& begin, End&& end, Chunk&& chunk, Fn&& body) {
  static_assert(Parallelism > 0, "static parallelism must be positive");
  auto begin_value = meta::to_value(std::forward<Begin>(begin));
  auto end_value = meta::to_value(std::forward<End>(end));
  auto chunk_value = meta::to_value(std::forward<Chunk>(chunk));
  using ChunkValue = decltype(chunk_value);
  if constexpr (ChunkValue::is_const)
    static_assert(ChunkValue::value > 0, "vecops parallel_for chunk must be positive");

  const nint_t raw_begin = static_cast<nint_t>(begin_value);
  const nint_t raw_end = static_cast<nint_t>(end_value);
  const nint_t raw_chunk = static_cast<nint_t>(chunk_value);
  if (raw_chunk <= 0)
    throw std::invalid_argument("vecops parallel_for chunk must be positive");
  if (raw_end < raw_begin)
    throw std::invalid_argument("vecops parallel_for end precedes begin");

  auto program_count = ::vecops::ceil_div(end_value - begin_value, chunk_value);
  details::parallel_range<Parallelism>(pool, raw_begin, raw_end, chunk_value, program_count,
                                       std::forward<Fn>(body));
}

/** Execute a range through compile-time-fixed lanes using the built-in backend. */
template <nint_t Parallelism, typename Fn>
void parallel_for(nint_t begin, nint_t end, nint_t chunk, Fn&& body) {
  parallel_for<Parallelism>(nullptr, begin, end, chunk, std::forward<Fn>(body));
}

/** Meta range form using the built-in backend. */
template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Fn>
  requires (meta::ValueType<std::remove_cvref_t<Begin>> || meta::ValueType<std::remove_cvref_t<End>> ||
            meta::ValueType<std::remove_cvref_t<Chunk>>)
void parallel_for(Begin&& begin, End&& end, Chunk&& chunk, Fn&& body) {
  parallel_for<Parallelism>(nullptr, std::forward<Begin>(begin), std::forward<End>(end),
                            std::forward<Chunk>(chunk), std::forward<Fn>(body));
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
  constexpr auto partition = plan_task_partition(meta::cint<Parallelism>, meta::cint<Total>);
  using LaneIndex = meta::Dynamic<1, 0, Parallelism - 1>;
  visit_task_partition_shard(partition, LaneIndex{worker.thread_id}, [&](const auto& shard) {
    using TaskCount = typename std::remove_cvref_t<decltype(shard)>::task_count_type;
    static_assert(TaskCount::is_const);
    std::invoke(std::forward<Fn>(body),
                StaticShard<TaskCount::value>{static_cast<nint_t>(shard.begin)});
  });
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_PARALLEL_H
