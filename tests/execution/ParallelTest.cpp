#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vecops/execution/Parallel.h"

namespace {

using namespace vecops;

struct RecordingThreadPool {
  static std::uint32_t maximum(void*) {
    return 16;
  }

  static std::uint32_t in_parallel(void*) {
    return 0;
  }

  static std::int32_t run(void* opaque, std::uint32_t count, void* body_context,
                          VecopsParallelTaskFn body, VecopsError*) {
    auto& self = *static_cast<RecordingThreadPool*>(opaque);
    ++self.calls;
    self.submitted = count;
    // Deliberately differ from lane order to ensure range assignment depends
    // only on logical lane ids, not backend scheduling order.
    for (std::uint32_t lane = count; lane-- > 0;)
      body(body_context, lane, count);
    return VECOPS_STATUS_OK;
  }

  [[nodiscard]] VecopsThreadPoolV1 abi() {
    return VecopsThreadPoolV1{
      sizeof(VecopsThreadPoolV1), VECOPS_THREAD_POOL_ABI_MAJOR, VECOPS_THREAD_POOL_ABI_MINOR,
      91, this, maximum, in_parallel, run, nullptr, nullptr, 0};
  }

  std::uint32_t calls = 0;
  std::uint32_t submitted = 0;
};

struct NativePoolProbe {
  static constexpr std::size_t task_count = 11;

  static void visit(void* opaque, std::uint32_t task, std::uint32_t count) {
    auto& self = *static_cast<NativePoolProbe*>(opaque);
    if (count != task_count || task >= task_count) {
      self.invalid.store(true, std::memory_order_relaxed);
      return;
    }
    self.visits[task].fetch_add(1, std::memory_order_relaxed);
  }

  std::array<std::atomic<int>, task_count> visits{};
  std::atomic<bool> invalid{false};
};

struct NestedNativePoolProbe {
  static void inner(void* opaque, std::uint32_t, std::uint32_t) {
    static_cast<NestedNativePoolProbe*>(opaque)->inner_visits.fetch_add(1, std::memory_order_relaxed);
  }

  static void outer(void* opaque, std::uint32_t task, std::uint32_t) {
    auto& self = *static_cast<NestedNativePoolProbe*>(opaque);
    self.outer_visits.fetch_add(1, std::memory_order_relaxed);
    if (self.pool->in_parallel_region(self.pool->context) != 1)
      self.invalid.store(true, std::memory_order_relaxed);
    if (task == 0) {
      VecopsError error{sizeof(VecopsError)};
      if (self.pool->parallel_for(self.pool->context, 3, &self, inner, &error) != VECOPS_STATUS_OK)
        self.invalid.store(true, std::memory_order_relaxed);
    }
  }

  const VecopsThreadPoolV1* pool = nullptr;
  std::atomic<int> outer_visits{0};
  std::atomic<int> inner_visits{0};
  std::atomic<bool> invalid{false};
};

TEST(ParallelTest, PersistentNativePoolRunsEveryLogicalTaskExactlyOnce) {
  const auto* pool = vecops_native_thread_pool_v1(4);
  ASSERT_NE(pool, nullptr);
  ASSERT_EQ(pool->max_parallelism(pool->context), 4U);
  NativePoolProbe probe;
  VecopsError error{sizeof(VecopsError)};
  for (int iteration = 0; iteration < 100; ++iteration)
    ASSERT_EQ(pool->parallel_for(pool->context, NativePoolProbe::task_count, &probe, NativePoolProbe::visit, &error),
              VECOPS_STATUS_OK);
  EXPECT_FALSE(probe.invalid.load(std::memory_order_relaxed));
  for (const auto& visits : probe.visits)
    EXPECT_EQ(visits.load(std::memory_order_relaxed), 100);
}

TEST(ParallelTest, PersistentNativePoolKeepsSubmissionAliveForIdleWorkers) {
  const auto* pool = vecops_native_thread_pool_v1(4);
  std::atomic<int> visits{0};
  auto visit = [](void* opaque, std::uint32_t task, std::uint32_t count) {
    auto& value = *static_cast<std::atomic<int>*>(opaque);
    if (task == 0 && count == 1)
      value.fetch_add(1, std::memory_order_relaxed);
  };
  VecopsError error{sizeof(VecopsError)};
  // A one-task submission leaves every native worker idle. Repetition makes
  // late generation observers race with stack reuse in implementations that
  // only wait for the participant count.
  constexpr int iterations = 10000;
  for (int iteration = 0; iteration < iterations; ++iteration)
    ASSERT_EQ(pool->parallel_for(pool->context, 1, &visits, visit, &error), VECOPS_STATUS_OK);
  EXPECT_EQ(visits.load(std::memory_order_relaxed), iterations);
}

TEST(ParallelTest, PersistentNativePoolSerializesNestedSubmission) {
  const auto* pool = vecops_native_thread_pool_v1(4);
  NestedNativePoolProbe probe;
  probe.pool = pool;
  VecopsError error{sizeof(VecopsError)};
  ASSERT_EQ(pool->parallel_for(pool->context, 4, &probe, NestedNativePoolProbe::outer, &error), VECOPS_STATUS_OK);
  EXPECT_FALSE(probe.invalid.load(std::memory_order_relaxed));
  EXPECT_EQ(probe.outer_visits.load(std::memory_order_relaxed), 4);
  EXPECT_EQ(probe.inner_visits.load(std::memory_order_relaxed), 3);
}

TEST(ParallelTest, PersistentNativePoolSupportsNestedActiveRegions) {
  vecops_native_thread_pool_begin_active(4);
  vecops_native_thread_pool_begin_active(4);
  const auto* pool = vecops_native_thread_pool_v1(4);
  NativePoolProbe probe;
  VecopsError error{sizeof(VecopsError)};
  ASSERT_EQ(pool->parallel_for(pool->context, NativePoolProbe::task_count,
                               &probe, NativePoolProbe::visit, &error),
            VECOPS_STATUS_OK);
  vecops_native_thread_pool_end_active();
  vecops_native_thread_pool_end_active();
  EXPECT_FALSE(probe.invalid.load(std::memory_order_relaxed));
  for (const auto& visits : probe.visits)
    EXPECT_EQ(visits.load(std::memory_order_relaxed), 1);
}

TEST(ParallelTest, MaximumIsPositiveAndRequestedWorkersHaveDenseIds) {
  const nint_t maximum = execution::max_parallelism();
  ASSERT_GE(maximum, 1);
  const nint_t requested = std::min<nint_t>(maximum, 4);
  std::vector<std::atomic<int>> visits(static_cast<std::size_t>(requested));
  std::atomic<nint_t> observed_threads{0};
  std::atomic<bool> invalid_context{false};

  execution::parallel_for(requested, [&](execution::ParallelContext context) {
    if (context.num_threads < 1 || context.num_threads > requested || context.thread_id < 0 ||
        context.thread_id >= context.num_threads) {
      invalid_context.store(true, std::memory_order_relaxed);
      return;
    }
    observed_threads.store(context.num_threads, std::memory_order_relaxed);
    visits[static_cast<std::size_t>(context.thread_id)].fetch_add(1, std::memory_order_relaxed);
  });

  EXPECT_FALSE(invalid_context.load(std::memory_order_relaxed));
  const auto actual = observed_threads.load(std::memory_order_relaxed);
  ASSERT_GE(actual, 1);
  ASSERT_LE(actual, requested);
  for (nint_t thread = 0; thread < actual; ++thread) {
    EXPECT_EQ(visits[static_cast<std::size_t>(thread)].load(std::memory_order_relaxed), 1);
  }
}

TEST(ParallelTest, NestedRegionsSerialize) {
  std::atomic<nint_t> outer_visits{0};
  std::atomic<nint_t> inner_visits{0};
  std::atomic<bool> invalid_inner_context{false};

  execution::parallel_for(std::min<nint_t>(execution::max_parallelism(), 4), [&](execution::ParallelContext) {
    outer_visits.fetch_add(1, std::memory_order_relaxed);
    if (execution::max_parallelism() != 1)
      invalid_inner_context.store(true, std::memory_order_relaxed);
    execution::parallel_for(4, [&](execution::ParallelContext inner) {
      if (inner.thread_id != 0 || inner.num_threads != 1)
        invalid_inner_context.store(true, std::memory_order_relaxed);
      inner_visits.fetch_add(1, std::memory_order_relaxed);
    });
  });

  EXPECT_FALSE(invalid_inner_context.load(std::memory_order_relaxed));
  EXPECT_EQ(inner_visits.load(std::memory_order_relaxed), outer_visits.load(std::memory_order_relaxed));
}

TEST(ParallelTest, WorkerExceptionIsRethrownAfterJoin) {
  EXPECT_THROW(execution::parallel_for(std::min<nint_t>(execution::max_parallelism(), 4),
                                       [](execution::ParallelContext context) {
                                         if (context.thread_id == 0)
                                           throw std::runtime_error("parallel worker failure");
                                       }),
               std::runtime_error);
}

TEST(ParallelTest, StaticLanesUseLogicalNamesAndSubmitExactlyTheSpecializedCount) {
  RecordingThreadPool recording;
  auto pool = recording.abi();
  std::array<int, 5> visits{};

  execution::parallel_lanes<5>(&pool, [&](execution::TaskContext<5> task) {
    static_assert(decltype(task)::lane_count() == 5);
    ASSERT_GE(task.lane_id(), 0);
    ASSERT_LT(task.lane_id(), task.lane_count());
    ++visits[static_cast<std::size_t>(task.lane_id())];
  });

  EXPECT_EQ(recording.calls, 1U);
  EXPECT_EQ(recording.submitted, 5U);
  EXPECT_EQ(visits, (std::array<int, 5>{1, 1, 1, 1, 1}));
}

TEST(ParallelTest, RangeProgramsAreBalancedAcrossLanesAndCoverClippedTailOnce) {
  RecordingThreadPool recording;
  auto pool = recording.abi();
  std::array<int, 23> visits{};
  std::array<int, 4> programs_per_lane{};
  std::array<nint_t, 5> observed_begin{};
  std::array<nint_t, 5> observed_end{};

  execution::parallel_for<4>(&pool, 3, 26, 5, [&](auto task, execution::WorkItem item) {
    ASSERT_EQ(item.program_count(), 5);
    ASSERT_GE(item.program_id(), 0);
    ASSERT_LT(item.program_id(), item.program_count());
    ++programs_per_lane[static_cast<std::size_t>(task.lane_id())];
    observed_begin[static_cast<std::size_t>(item.program_id())] = item.begin();
    observed_end[static_cast<std::size_t>(item.program_id())] = item.end();
    EXPECT_EQ(item.extent(), item.end() - item.begin());
    for (nint_t index = item.begin(); index < item.end(); ++index)
      ++visits[static_cast<std::size_t>(index - 3)];
  });

  EXPECT_EQ(recording.calls, 1U);
  EXPECT_EQ(recording.submitted, 4U);
  EXPECT_EQ(programs_per_lane, (std::array<int, 4>{2, 1, 1, 1}));
  EXPECT_EQ(observed_begin, (std::array<nint_t, 5>{3, 8, 13, 18, 23}));
  EXPECT_EQ(observed_end, (std::array<nint_t, 5>{8, 13, 18, 23, 26}));
  EXPECT_EQ(visits, (std::array<int, 23>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ParallelTest, RangeWithFewerProgramsOmitsEmptyLanes) {
  RecordingThreadPool recording;
  auto pool = recording.abi();
  std::array<int, 2> visits{};

  execution::parallel_for<8>(&pool, 0, 2, 1, [&](auto task, auto item) {
    EXPECT_LT(task.lane_id(), 2);
    ++visits[static_cast<std::size_t>(item.begin())];
  });

  EXPECT_EQ(recording.calls, 1U);
  EXPECT_EQ(recording.submitted, 2U);
  EXPECT_EQ(visits, (std::array<int, 2>{1, 1}));
}

TEST(ParallelTest, EmptyRangeDoesNotSubmitAThreadPoolCall) {
  RecordingThreadPool recording;
  auto pool = recording.abi();
  bool visited = false;

  execution::parallel_for<8>(&pool, meta::cint<0>, meta::cint<0>, meta::cint<1>,
                             [&](auto, auto) { visited = true; });

  EXPECT_EQ(recording.calls, 0U);
  EXPECT_EQ(recording.submitted, 0U);
  EXPECT_FALSE(visited);
}

TEST(ParallelTest, MetaRangeRetainsConstChunkWithoutSplittingFullAndTailTypes) {
  RecordingThreadPool recording;
  auto pool = recording.abi();
  std::array<int, 5> visits{};
  int full = 0;
  int tail = 0;

  execution::parallel_for<3>(&pool, meta::cint<3>, meta::cint<26>, meta::cint<5>,
                             [&](auto task, auto item) {
    using Item = std::remove_cvref_t<decltype(item)>;
    static_assert(std::same_as<typename Item::chunk_type, meta::Const<5>>);
    static_assert(std::same_as<typename Item::program_count_type, meta::Const<5>>);
    static_assert(decltype(item.chunk())::is_const);
    static_assert(decltype(item.program_count_value())::is_const);
    EXPECT_EQ(static_cast<nint_t>(item.chunk()), 5);
    EXPECT_EQ(item.program_count(), 5);
    EXPECT_GE(task.lane_id(), 0);
    ++visits[static_cast<std::size_t>(item.program_id())];
    if (item.is_full())
      ++full;
    else
      ++tail;
  });

  EXPECT_EQ(recording.calls, 1U);
  EXPECT_EQ(recording.submitted, 3U);
  EXPECT_EQ(visits, (std::array<int, 5>{1, 1, 1, 1, 1}));
  EXPECT_EQ(full, 4);
  EXPECT_EQ(tail, 1);
}

TEST(ParallelTest, DynamicMetaBoundsRetainChunkContractAndCoverTheRange) {
  std::array<int, 17> visits{};
  auto begin = meta::dyn<1, 0, 64>(7);
  auto end = meta::dyn<1, 0, 64>(24);

  execution::parallel_for<4>(begin, end, meta::cint<4>, [&](auto, auto item) {
    static_assert(std::same_as<typename decltype(item)::chunk_type, meta::Const<4>>);
    for (nint_t index = item.begin(); index < item.end(); ++index)
      ++visits[static_cast<std::size_t>(index - 7)];
  });

  EXPECT_EQ(visits, (std::array<int, 17>{1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ParallelTest, RangeRejectsInvalidBoundsAndChunk) {
  EXPECT_THROW(execution::parallel_for<2>(nullptr, 0, 10, 0, [](auto, auto) {}), std::invalid_argument);
  EXPECT_THROW(execution::parallel_for<2>(nullptr, 4, 3, 1, [](auto, auto) {}), std::invalid_argument);
}

} // namespace
