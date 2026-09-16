#include <gtest/gtest.h>

#include <tuple>
#include <type_traits>

#include "vecops/execution/TaskPartition.h"

namespace {

using namespace vecops;

TEST(TaskPartitionTest, PreservesConstMetricsForOneWaveTail) {
  constexpr auto candidate =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<40>, meta::cint<1>));
  constexpr auto metrics = execution::analyze_task_partition(meta::cint<38>, candidate);

  static_assert(std::same_as<decltype(metrics.lane_count), meta::Const<38>>);
  static_assert(std::same_as<decltype(metrics.phase_count), meta::Const<1>>);
  static_assert(std::same_as<decltype(metrics.total_tasks), meta::Const<40>>);
  static_assert(std::same_as<decltype(metrics.total_work), meta::Const<40>>);
  static_assert(std::same_as<decltype(metrics.critical_work), meta::Const<2>>);
  static_assert(std::same_as<decltype(metrics.lane_capacity_work), meta::Const<76>>);
  static_assert(std::same_as<decltype(metrics.idle_lane_work), meta::Const<36>>);
  EXPECT_NEAR(metrics.efficiency(), 40.0 / 76.0, 1e-12);
}

TEST(TaskPartitionTest, StaticInputsProduceAConstCandidateChoice) {
  constexpr auto coarse =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<40>, meta::cint<66>),
                                        execution::task_partition_phase(meta::cint<40>, meta::cint<17>));
  constexpr auto fine =
    execution::task_partition_candidate(meta::cint<2>, execution::task_partition_phase(meta::cint<40>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<640>, meta::cint<5>),
                                        execution::task_partition_phase(meta::cint<640>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<40>, meta::cint<1>));
  constexpr auto candidates = std::tuple{coarse, fine};
  constexpr auto plan = execution::choose_task_partition(meta::cint<38>, candidates);

  static_assert(std::same_as<decltype(plan.candidate_index), meta::Const<1>>);
  EXPECT_EQ(static_cast<nint_t>(plan.candidate_index), 1);
}

TEST(TaskPartitionTest, DynamicTasksRetainBoundsAndUseRuntimeChoice) {
  const auto tasks = meta::dyn<1, 0, 128>(40);
  const auto coarse =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(tasks, meta::cint<32>));
  const auto fine =
    execution::task_partition_candidate(meta::cint<3>, execution::task_partition_phase(tasks, meta::cint<2>),
                                        execution::task_partition_phase(tasks * meta::cint<12>, meta::cint<1>),
                                        execution::task_partition_phase(tasks * meta::cint<12>, meta::cint<1>),
                                        execution::task_partition_phase(tasks * meta::cint<6>, meta::cint<1>));
  const auto candidates = std::tuple{coarse, fine};
  const auto metrics = execution::analyze_task_partition(meta::cint<38>, coarse);
  const auto plan = execution::choose_task_partition(meta::cint<38>, candidates);

  using TotalTasks = std::remove_cvref_t<decltype(metrics.total_tasks)>;
  static_assert(TotalTasks::is_runtime);
  static_assert(meta::has_lower_bound_v<TotalTasks>);
  static_assert(meta::has_upper_bound_v<TotalTasks>);
  static_assert(meta::lower_bound_v<TotalTasks> == 0);
  static_assert(meta::upper_bound_v<TotalTasks> == 128);
  static_assert(std::same_as<decltype(plan.candidate_index), meta::Any>);
  EXPECT_EQ(static_cast<nint_t>(plan.candidate_index), 1);
}

TEST(TaskPartitionTest, ExactTwoWavesPreferLocalCoarseCandidate) {
  constexpr auto coarse =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<76>, meta::cint<32>));
  constexpr auto fine =
    execution::task_partition_candidate(meta::cint<3>, execution::task_partition_phase(meta::cint<76>, meta::cint<2>),
                                        execution::task_partition_phase(meta::cint<912>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<912>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<456>, meta::cint<1>));
  constexpr auto plan = execution::choose_task_partition(meta::cint<38>, std::tuple{coarse, fine});

  static_assert(std::same_as<decltype(plan.candidate_index), meta::Const<0>>);
  EXPECT_EQ(static_cast<nint_t>(plan.candidate_index), 0);
}

TEST(TaskPartitionTest, TaskImmediatelyAfterExactWaveUsesFineCandidate) {
  constexpr auto coarse =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<77>, meta::cint<32>));
  constexpr auto fine =
    execution::task_partition_candidate(meta::cint<3>, execution::task_partition_phase(meta::cint<77>, meta::cint<2>),
                                        execution::task_partition_phase(meta::cint<924>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<924>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<462>, meta::cint<1>));
  constexpr auto plan = execution::choose_task_partition(meta::cint<38>, std::tuple{coarse, fine});

  static_assert(std::same_as<decltype(plan.candidate_index), meta::Const<1>>);
}

TEST(TaskPartitionTest, TieBreaksTowardFewerPhasesThenEarlierCandidate) {
  constexpr auto first =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<38>, meta::cint<2>));
  constexpr auto second =
    execution::task_partition_candidate(meta::cint<0>, execution::task_partition_phase(meta::cint<38>, meta::cint<1>),
                                        execution::task_partition_phase(meta::cint<38>, meta::cint<1>));
  constexpr auto third = first;
  constexpr auto plan = execution::choose_task_partition(meta::cint<38>, std::tuple{first, second, third});

  static_assert(std::same_as<decltype(plan.candidate_index), meta::Const<0>>);
}

TEST(TaskPartitionTest, RawIntegersDeliberatelyNormalizeToAny) {
  const auto candidate = execution::task_partition_candidate(0, execution::task_partition_phase(40, 1));
  const auto metrics = execution::analyze_task_partition(38, candidate);
  const auto plan = execution::choose_task_partition(38, std::tuple{candidate});

  static_assert(std::same_as<decltype(metrics.lane_count), meta::Any>);
  static_assert(std::same_as<decltype(metrics.total_tasks), meta::Any>);
  static_assert(std::same_as<decltype(plan.candidate_index), meta::Any>);
  EXPECT_EQ(static_cast<nint_t>(metrics.critical_work), 2);
}

TEST(TaskPartitionTest, RejectsInvalidDynamicDescriptions) {
  const auto valid =
    execution::task_partition_candidate(meta::Any{0}, execution::task_partition_phase(meta::Any{1}, meta::Any{1}));
  const auto negative_tasks =
    execution::task_partition_candidate(meta::Any{0}, execution::task_partition_phase(meta::Any{-1}, meta::Any{1}));
  const auto zero_work =
    execution::task_partition_candidate(meta::Any{0}, execution::task_partition_phase(meta::Any{1}, meta::Any{0}));

  EXPECT_THROW((void)execution::analyze_task_partition(meta::Any{0}, valid), std::invalid_argument);
  EXPECT_THROW((void)execution::analyze_task_partition(meta::Any{1}, negative_tasks), std::invalid_argument);
  EXPECT_THROW((void)execution::analyze_task_partition(meta::Any{1}, zero_work), std::invalid_argument);
}

} // namespace
