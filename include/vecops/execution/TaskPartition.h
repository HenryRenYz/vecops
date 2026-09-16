//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_TASKPARTITION_H
#define VECOPS_EXECUTION_TASKPARTITION_H

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/Meta.h"

/**
 * @file vecops/execution/TaskPartition.h
 * @brief Metadata-preserving cost model for logical-task decomposition.
 *
 * A task count merely greater than the worker count is not necessarily well
 * balanced. Forty indivisible tasks on 38 workers require two waves and use
 * only 40 of 76 lane slots. This planner compares alternative decompositions
 * phase by phase using `ceil(tasks / lanes) * work_per_task`, plus an optional
 * fixed synchronization/setup cost.
 *
 * All integer inputs are `meta::ValueInput`. `Const` inputs remain `Const`
 * through phase construction, analysis, arithmetic, and candidate selection;
 * constrained `Dynamic` values retain their bounds/alignment according to
 * `Meta.h` arithmetic. Raw integers deliberately normalize to `meta::Any`.
 * When every candidate score is compile-time constant, the selected candidate
 * is itself a `meta::Const<Index>` and can drive `if constexpr` kernel codegen.
 * A genuinely dynamic score returns `meta::Any` and uses a runtime branch.
 *
 * Work units are caller-defined but must be comparable between candidates.
 * They normally count repeated inner operations (heads, panels, matmuls), not
 * bytes or instructions. The planner selects a decomposition but does not
 * launch it; kernels implement the selected phases with `parallel_for` or
 * `parallel_lanes`.
 */

namespace vecops::execution {

/** One synchronous phase containing equal-cost, indivisible logical tasks. */
template <meta::ValueType TaskCount, meta::ValueType WorkPerTask>
struct TaskPartitionPhase {
  using task_count_type = TaskCount;
  using work_per_task_type = WorkPerTask;

  TaskCount task_count;
  WorkPerTask work_per_task;
};

/** Construct a phase without discarding `meta::Value` constraints. */
template <meta::ValueInput TaskCount, meta::ValueInput WorkPerTask>
[[nodiscard]] constexpr auto task_partition_phase(TaskCount&& task_count, WorkPerTask&& work_per_task) {
  auto tasks = meta::to_value(std::forward<TaskCount>(task_count));
  auto work = meta::to_value(std::forward<WorkPerTask>(work_per_task));
  return TaskPartitionPhase<decltype(tasks), decltype(work)>{tasks, work};
}

template <typename T>
concept TaskPartitionPhaseType =
  requires {
    typename std::remove_cvref_t<T>::task_count_type;
    typename std::remove_cvref_t<T>::work_per_task_type;
  } && meta::ValueType<typename std::remove_cvref_t<T>::task_count_type> &&
  meta::ValueType<typename std::remove_cvref_t<T>::work_per_task_type>;

/** One complete candidate decomposition of an operation. */
template <typename Phases, meta::ValueType FixedCost>
struct TaskPartitionCandidate {
  using phases_type = Phases;
  using fixed_cost_type = FixedCost;

  Phases phases;
  FixedCost fixed_cost;
};

/** Construct a candidate whose heterogeneous phase metadata stays typed. */
template <meta::ValueInput FixedCost, TaskPartitionPhaseType... Phases>
[[nodiscard]] constexpr auto task_partition_candidate(FixedCost&& fixed_cost, Phases&&... phases) {
  auto fixed = meta::to_value(std::forward<FixedCost>(fixed_cost));
  auto values = std::tuple<std::remove_cvref_t<Phases>...>{std::forward<Phases>(phases)...};
  return TaskPartitionCandidate<decltype(values), decltype(fixed)>{values, fixed};
}

template <typename T>
concept TaskPartitionCandidateType = requires {
  typename std::remove_cvref_t<T>::phases_type;
  typename std::remove_cvref_t<T>::fixed_cost_type;
} && meta::ValueType<typename std::remove_cvref_t<T>::fixed_cost_type>;

/** Metadata-preserving aggregate scheduling metrics for one candidate. */
template <meta::ValueType LaneCount, meta::ValueType PhaseCount, meta::ValueType TotalTasks, meta::ValueType TotalWork,
          meta::ValueType CriticalWork, meta::ValueType LaneCapacityWork, meta::ValueType IdleLaneWork>
struct TaskPartitionMetrics {
  using lane_count_type = LaneCount;
  using phase_count_type = PhaseCount;
  using total_tasks_type = TotalTasks;
  using total_work_type = TotalWork;
  using critical_work_type = CriticalWork;
  using lane_capacity_work_type = LaneCapacityWork;
  using idle_lane_work_type = IdleLaneWork;

  LaneCount lane_count;
  PhaseCount phase_count;
  TotalTasks total_tasks;
  TotalWork total_work;
  CriticalWork critical_work;
  LaneCapacityWork lane_capacity_work;
  IdleLaneWork idle_lane_work;

  /** Weighted lane utilization in `[0, 1]`; one when modeled capacity is zero. */
  [[nodiscard]] constexpr double efficiency() const noexcept {
    const nint_t capacity = static_cast<nint_t>(lane_capacity_work);
    if (capacity == 0)
      return 1.0;
    return static_cast<double>(static_cast<nint_t>(total_work)) / static_cast<double>(capacity);
  }
};

/** Candidate selection; `CandidateIndex` is `Const<I>` when statically known. */
template <meta::ValueType CandidateIndex>
struct TaskPartitionPlan {
  using candidate_index_type = CandidateIndex;
  CandidateIndex candidate_index;
};

template <meta::ValueInput LaneCount, TaskPartitionCandidateType Candidate>
[[nodiscard]] constexpr auto analyze_task_partition(LaneCount&& lane_count, const Candidate& candidate);

namespace task_partition_details {

template <meta::ValueType Value>
constexpr void validate_nonnegative(Value value, const char* message) {
  if constexpr (Value::is_const) {
    static_assert(Value::value >= 0, "task partition value cannot be negative");
  } else if (static_cast<nint_t>(value) < 0) {
    throw std::invalid_argument(message);
  }
}

template <meta::ValueType Value>
constexpr void validate_positive(Value value, const char* message) {
  if constexpr (Value::is_const) {
    static_assert(Value::value > 0, "task partition value must be positive");
  } else if (static_cast<nint_t>(value) <= 0) {
    throw std::invalid_argument(message);
  }
}

template <std::size_t Index = 0, typename Tuple, typename Fn>
[[nodiscard]] constexpr auto tuple_sum(const Tuple& tuple, Fn function) {
  if constexpr (Index == std::tuple_size_v<std::remove_cvref_t<Tuple>>) {
    return meta::cint<0>;
  } else {
    return std::invoke(function, std::get<Index>(tuple)) + tuple_sum<Index + 1>(tuple, function);
  }
}

template <typename LaneCount, typename Candidate>
using metrics_t = decltype(analyze_task_partition(std::declval<LaneCount>(), std::declval<const Candidate&>()));

template <typename LaneCount, typename Candidates, std::size_t... Indices>
[[nodiscard]] consteval bool all_scores_const(std::index_sequence<Indices...>) {
  return ((metrics_t<LaneCount, std::tuple_element_t<Indices, Candidates>>::critical_work_type::is_const &&
           metrics_t<LaneCount, std::tuple_element_t<Indices, Candidates>>::idle_lane_work_type::is_const) &&
          ...);
}

template <typename LaneCount, typename Candidates, std::size_t Best, std::size_t Index>
struct StaticBestCandidate {
private:
  using BestMetrics = metrics_t<LaneCount, std::tuple_element_t<Best, Candidates>>;
  using CurrentMetrics = metrics_t<LaneCount, std::tuple_element_t<Index, Candidates>>;
  static constexpr nint_t best_critical = BestMetrics::critical_work_type::value;
  static constexpr nint_t current_critical = CurrentMetrics::critical_work_type::value;
  static constexpr nint_t best_idle = BestMetrics::idle_lane_work_type::value;
  static constexpr nint_t current_idle = CurrentMetrics::idle_lane_work_type::value;
  static constexpr nint_t best_phases = BestMetrics::phase_count_type::value;
  static constexpr nint_t current_phases = CurrentMetrics::phase_count_type::value;
  static constexpr bool better =
    current_critical < best_critical ||
    (current_critical == best_critical &&
     (current_idle < best_idle || (current_idle == best_idle && current_phases < best_phases)));
  static constexpr std::size_t next_best = better ? Index : Best;

public:
  static constexpr std::size_t value = StaticBestCandidate<LaneCount, Candidates, next_best, Index + 1>::value;
};

template <typename LaneCount, typename Candidates, std::size_t Best>
struct StaticBestCandidate<LaneCount, Candidates, Best, std::tuple_size_v<Candidates>> {
  static constexpr std::size_t value = Best;
};

} // namespace task_partition_details

/**
 * Analyze one candidate without erasing metadata from lanes, tasks, or work.
 *
 * @throws std::invalid_argument for invalid dynamic values. Invalid `Const`
 *         inputs fail at compile time.
 */
template <meta::ValueInput LaneCount, TaskPartitionCandidateType Candidate>
[[nodiscard]] constexpr auto analyze_task_partition(LaneCount&& lane_count, const Candidate& candidate) {
  auto lanes = meta::to_value(std::forward<LaneCount>(lane_count));
  task_partition_details::validate_positive(lanes, "task partition lane count must be positive");
  task_partition_details::validate_nonnegative(candidate.fixed_cost, "task partition fixed cost cannot be negative");
  std::apply(
    [](const auto&... phase) {
      ((task_partition_details::validate_nonnegative(phase.task_count, "task partition task count cannot be negative"),
        task_partition_details::validate_positive(phase.work_per_task, "task partition work must be positive")),
       ...);
    },
    candidate.phases);

  auto total_tasks =
    task_partition_details::tuple_sum(candidate.phases, [](const auto& phase) { return phase.task_count; });
  auto total_work = task_partition_details::tuple_sum(
    candidate.phases, [](const auto& phase) { return phase.task_count * phase.work_per_task; });
  auto phase_critical_work = task_partition_details::tuple_sum(
    candidate.phases, [&lanes](const auto& phase) { return ceil_div(phase.task_count, lanes) * phase.work_per_task; });
  auto critical_work = candidate.fixed_cost + phase_critical_work;
  auto lane_capacity_work = lanes * critical_work;
  auto idle_lane_work = lane_capacity_work - total_work;
  using PhaseCount = meta::Const<std::tuple_size_v<typename Candidate::phases_type>>;
  const PhaseCount phase_count{};

  return TaskPartitionMetrics<decltype(lanes), PhaseCount, decltype(total_tasks), decltype(total_work),
                              decltype(critical_work), decltype(lane_capacity_work), decltype(idle_lane_work)>{
    lanes, phase_count, total_tasks, total_work, critical_work, lane_capacity_work, idle_lane_work};
}

/**
 * Choose the candidate with the shortest estimated critical path.
 *
 * Ties prefer less idle lane work, then fewer phases, then the earlier
 * candidate. The heterogeneous tuple is intentional: every phase and
 * candidate retains its own `Const`/`Dynamic` types.
 */
template <meta::ValueInput LaneCount, TaskPartitionCandidateType... Candidates>
[[nodiscard]] constexpr auto choose_task_partition(LaneCount&& lane_count,
                                                   const std::tuple<Candidates...>& candidates) {
  static_assert(sizeof...(Candidates) > 0, "task partition requires at least one candidate");
  auto lanes = meta::to_value(std::forward<LaneCount>(lane_count));
  using LaneValue = decltype(lanes);
  using CandidateTuple = std::tuple<Candidates...>;
  constexpr bool static_scores =
    task_partition_details::all_scores_const<LaneValue, CandidateTuple>(std::index_sequence_for<Candidates...>{});

  if constexpr (static_scores) {
    constexpr std::size_t index = task_partition_details::StaticBestCandidate<LaneValue, CandidateTuple, 0, 1>::value;
    return TaskPartitionPlan<meta::Const<static_cast<nint_t>(index)>>{meta::Const<static_cast<nint_t>(index)>{}};
  } else {
    std::size_t candidate_index = 0;
    std::size_t best_index = 0;
    nint_t best_critical = 0;
    nint_t best_idle = 0;
    nint_t best_phases = 0;
    bool initialized = false;
    std::apply(
      [&](const auto&... candidate) {
        auto consider = [&](const auto& value) {
          const auto metrics = analyze_task_partition(lanes, value);
          const nint_t critical = static_cast<nint_t>(metrics.critical_work);
          const nint_t idle = static_cast<nint_t>(metrics.idle_lane_work);
          const nint_t phases = static_cast<nint_t>(metrics.phase_count);
          const bool better =
            !initialized || critical < best_critical ||
            (critical == best_critical && (idle < best_idle || (idle == best_idle && phases < best_phases)));
          if (better) {
            initialized = true;
            best_index = candidate_index;
            best_critical = critical;
            best_idle = idle;
            best_phases = phases;
          }
          ++candidate_index;
        };
        (consider(candidate), ...);
      },
      candidates);
    return TaskPartitionPlan<meta::Any>{meta::Any{static_cast<nint_t>(best_index)}};
  }
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_TASKPARTITION_H
