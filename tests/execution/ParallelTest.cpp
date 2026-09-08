#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <vector>

#include "vecops/execution/Parallel.h"

namespace {

using namespace vecops;

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

} // namespace
