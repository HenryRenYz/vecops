/** @file Grid.h
 * @brief CPU logical-grid scheduling with optional resident worker teams.
 *
 * A program is indivisible. Static assignment balances contiguous programs;
 * dynamic assignment claims chunks, not individual vector operations. Team
 * barriers must never be constructed on top of a logical-task-only provider.
 */
#ifndef VECOPS_EXECUTION_GRID_H
#define VECOPS_EXECUTION_GRID_H

#include <atomic>
#include <cmath>
#include <limits>
#include "vecops/CoreDefs.h"
#include "vecops/execution/Parallel.h"
#include "vecops/execution/FloatControl.h"
#include "vecops/execution/VectorLength.h"

namespace vecops::execution {

class GridTeam {
public:
  explicit GridTeam(const VecopsTeamContextV1& team)
    : team_(team) {
  }
  [[nodiscard]] nint_t worker_id() const noexcept {
    return team_.worker_id;
  }
  [[nodiscard]] nint_t worker_count() const noexcept {
    return team_.worker_count;
  }
  void barrier() const {
    if (team_.worker_count <= 1)
      return;
    char message[256]{};
    VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
    if (team_.barrier == nullptr || team_.barrier(team_.context, &error) != VECOPS_STATUS_OK)
      throw std::runtime_error(message[0] ? message : "grid team cancelled");
  }

private:
  VecopsTeamContextV1 team_;
};

[[nodiscard]] inline bool supports_grid_team(const VecopsThreadPoolV1* pool) noexcept {
  return pool != nullptr &&
         pool->struct_size >= offsetof(VecopsThreadPoolV1, team_executor) + sizeof(pool->team_executor) &&
         pool->abi_major == VECOPS_THREAD_POOL_ABI_MAJOR && pool->team_executor != nullptr &&
         pool->team_executor->struct_size >= sizeof(VecopsTeamExecutorV1) && pool->team_executor->abi_major == 1 &&
         pool->team_executor->run != nullptr;
}

/** Respect an existing provider parallel region before selecting workers. */
[[nodiscard]] inline nint_t grid_capacity(const VecopsThreadPoolV1* pool) noexcept {
  if (pool != nullptr && pool->struct_size >= offsetof(VecopsThreadPoolV1, flags) + sizeof(pool->flags) &&
      pool->abi_major == VECOPS_THREAD_POOL_ABI_MAJOR && pool->in_parallel_region != nullptr &&
      pool->in_parallel_region(pool->context))
    return 1;
  return max_parallelism(pool);
}

/** Choose active workers, not pool capacity. Costs are relative cycle
 * estimates from the target profile. Equal-cost indivisible tasks retain a
 * wave tail; using fewer workers can reduce dispatch cost, not remove it.
 */
[[nodiscard]] inline nint_t grid_workers_available(nint_t capacity, nint_t requested, nint_t programs,
                                                   double program_cycles, double worker_cycles) {
  if (programs <= 0)
    return 0;
  capacity = std::max<nint_t>(1, capacity);
  const auto limit = std::min(programs, requested > 0 ? std::min(requested, capacity) : capacity);
  if (!(program_cycles > 0) || !std::isfinite(program_cycles))
    program_cycles = 1;
  if (!(worker_cycles >= 0) || !std::isfinite(worker_cycles))
    worker_cycles = 2000;
  nint_t best = 1;
  double score = static_cast<double>(programs) * program_cycles;
  for (nint_t workers = 2; workers <= limit; ++workers) {
    const auto waves = programs / workers + (programs % workers != 0);
    const double candidate = static_cast<double>(waves) * program_cycles + workers * worker_cycles;
    if (candidate < score) {
      score = candidate;
      best = workers;
    }
  }
  return best;
}

[[nodiscard]] inline nint_t grid_workers(const VecopsThreadPoolV1* pool, nint_t requested, nint_t programs,
                                         double program_cycles, double worker_cycles) {
  return grid_workers_available(grid_capacity(pool), requested, programs, program_cycles, worker_cycles);
}

template <class Fn>
void grid_team(const VecopsThreadPoolV1* pool, nint_t workers, Fn&& body) {
  if (workers <= 1) {
    apply_process_flush_subnormals();
    const VecopsTeamContextV1 team{sizeof(VecopsTeamContextV1), 0, 1, nullptr, nullptr};
    body(GridTeam(team));
    return;
  }
  if (!supports_grid_team(pool))
    throw std::invalid_argument("executor has no resident-team capability");
  if (workers > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("grid worker count overflow");
  using Body = std::remove_reference_t<Fn>;
  struct CallbackState {
    Body* body;
    ArmVectorLengths vector_lengths;
  } state{std::addressof(body), current_thread_vector_lengths()};
  char message[512]{};
  VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
  const auto status = pool->team_executor->run(
    pool->context, static_cast<uint32_t>(workers), &state,
    [](void* opaque, const VecopsTeamContextV1* team) VECOPS_INLINE_LAMBDA {
      auto& callback = *static_cast<CallbackState*>(opaque);
      const ScopedArmVectorLengths vector_lengths(callback.vector_lengths);
      apply_process_flush_subnormals();
      std::invoke(*callback.body, GridTeam(*team));
    },
    &error);
  if (status != VECOPS_STATUS_OK)
    throw std::runtime_error(message[0] ? message : "grid team failed");
}

template <class Fn>
void grid_programs(nint_t programs, nint_t worker, nint_t workers, bool dynamic, nint_t chunk,
                   std::atomic<nint_t>& cursor, Fn&& body) {
  if (programs <= 0)
    return;
  if (workers <= 0 || worker < 0 || worker >= workers || chunk <= 0)
    throw std::invalid_argument("invalid grid scheduling parameters");
  if (workers == 1) {
    for (nint_t program = 0; program < programs; ++program)
      body(program);
    return;
  }
  if (dynamic) {
    for (;;) {
      auto first = cursor.load(std::memory_order_relaxed);
      if (first >= programs)
        break;
      const auto last = first + std::min(chunk, programs - first);
      if (!cursor.compare_exchange_weak(first, last, std::memory_order_relaxed))
        continue;
      for (auto program = first; program < last; ++program)
        body(program);
    }
  } else {
    const auto small = programs / workers, large = programs % workers;
    const auto first = worker * small + std::min(worker, large);
    const auto last = first + small + (worker < large);
    for (auto program = first; program < last; ++program)
      body(program);
  }
}

/** One callback per logical lane. It owns reusable scratch across programs.
 * Logical lanes need not be simultaneously resident; no barriers are legal
 * in this callback unless using the separate grid_team capability.
 */
template <class Fn>
void grid_lanes(const VecopsThreadPoolV1* pool, nint_t workers, Fn&& body) {
  if (workers <= 1) {
    apply_process_flush_subnormals();
    body(ParallelContext(0, 1));
    return;
  }
  parallel_tasks(pool, workers, std::forward<Fn>(body));
}

template <class Fn>
void grid_tasks(const VecopsThreadPoolV1* pool, nint_t programs, nint_t workers, bool dynamic, nint_t chunk,
                Fn&& body) {
  if (programs <= 0)
    return;
  if (workers <= 1) {
    apply_process_flush_subnormals();
    for (nint_t p = 0; p < programs; ++p)
      body(p);
    return;
  }
  std::atomic<nint_t> cursor{0};
  parallel_tasks(pool, workers, [&](ParallelContext lane) VECOPS_INLINE_LAMBDA {
    grid_programs(programs, lane.thread_id, lane.num_threads, dynamic, chunk, cursor, body);
  });
}

} // namespace vecops::execution
#endif
