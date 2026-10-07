#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "vecops/execution/Grid.h"
#include "vecops/execution/VectorLength.h"

namespace {

using vecops::execution::ArmVectorLengths;
using vecops::execution::ScopedArmVectorLengths;
using vecops::execution::current_thread_vector_lengths;

TEST(VectorLengthTest, UnchangedScopeAndKernelCheckPreserveCaller) {
  const auto original = current_thread_vector_lengths();
  {
    const ScopedArmVectorLengths unchanged(original);
    EXPECT_EQ(current_thread_vector_lengths(), original);
    EXPECT_TRUE(vecops::execution::current_thread_vector_lengths_match(original));
    EXPECT_NO_THROW(vecops::execution::require_current_thread_vector_lengths(original));
  }
  EXPECT_EQ(current_thread_vector_lengths(), original);
}

TEST(VectorLengthTest, InvalidLengthNeverChangesThread) {
  const auto original = current_thread_vector_lengths();
  EXPECT_THROW(ScopedArmVectorLengths(ArmVectorLengths{7, 0}), std::exception);
  EXPECT_EQ(current_thread_vector_lengths(), original);
  EXPECT_THROW(vecops::execution::require_current_thread_vector_lengths({7, 0}), std::runtime_error);
}

#if defined(__linux__) && defined(__aarch64__)
TEST(VectorLengthTest, NestedIndependentLengthsRestoreOnException) {
  const auto original = current_thread_vector_lengths();
  if (original.sve_bytes == 0 || original.sme_bytes == 0)
    GTEST_SKIP() << "requires Linux SVE and SME vector-length controls";
  {
    const ScopedArmVectorLengths short_sve({16, original.sme_bytes});
    EXPECT_EQ(current_thread_vector_lengths(), (ArmVectorLengths{16, original.sme_bytes}));
    try {
      const ScopedArmVectorLengths short_sme({16, 16});
      EXPECT_EQ(current_thread_vector_lengths(), (ArmVectorLengths{16, 16}));
      throw std::runtime_error("restore probe");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(current_thread_vector_lengths(), (ArmVectorLengths{16, original.sme_bytes}));
  }
  EXPECT_EQ(current_thread_vector_lengths(), original);
}

struct DifferentLengthPool {
  std::atomic<bool> restored{false};

  static std::uint32_t maximum(void*) {
    return 1;
  }

  static std::int32_t run(void* opaque, std::uint32_t count, void* context, VecopsParallelTaskFn body, VecopsError*) {
    auto& self = *static_cast<DifferentLengthPool*>(opaque);
    std::thread worker([&] {
      const ScopedArmVectorLengths short_lengths({16, 16});
      for (std::uint32_t task = 0; task < count; ++task)
        body(context, task, count);
      self.restored.store(current_thread_vector_lengths() == ArmVectorLengths{16, 16});
    });
    worker.join();
    return VECOPS_STATUS_OK;
  }
};

TEST(VectorLengthTest, LogicalTasksAdaptPreexistingWorkerAndRestoreItsLengths) {
  const auto original = current_thread_vector_lengths();
  if (original.sve_bytes == 0 || original.sme_bytes == 0)
    GTEST_SKIP() << "requires Linux SVE and SME vector-length controls";
  DifferentLengthPool probe;
  const VecopsThreadPoolV1 pool{sizeof(VecopsThreadPoolV1),
                                VECOPS_THREAD_POOL_ABI_MAJOR,
                                VECOPS_THREAD_POOL_ABI_MINOR,
                                1,
                                &probe,
                                probe.maximum,
                                nullptr,
                                probe.run,
                                nullptr,
                                nullptr,
                                0};
  std::atomic<int> visits{0};
  vecops::execution::parallel_tasks(&pool, 3, [&](vecops::execution::ParallelContext) {
    EXPECT_EQ(current_thread_vector_lengths(), original);
    ++visits;
  });
  EXPECT_EQ(visits, 3);
  EXPECT_TRUE(probe.restored.load());
  EXPECT_EQ(current_thread_vector_lengths(), original);
}

TEST(VectorLengthTest, ResidentTeamSynchronizesDistinctVLAndSVL) {
  const auto original = current_thread_vector_lengths();
  if (original.sve_bytes == 0 || original.sme_bytes == 0)
    GTEST_SKIP() << "requires Linux SVE and SME vector-length controls";
  const auto* pool = vecops_native_thread_pool_v1(3);
  const ScopedArmVectorLengths distinct_lengths({16, original.sme_bytes});
  const ArmVectorLengths expected{16, original.sme_bytes};
  std::atomic<int> visits{0};
  vecops::execution::grid_team(pool, 3, [&](vecops::execution::GridTeam team) {
    EXPECT_EQ(current_thread_vector_lengths(), expected);
    ++visits;
    team.barrier();
    EXPECT_TRUE(vecops::execution::current_thread_vector_lengths_match(expected));
  });
  EXPECT_EQ(visits, 3);
  EXPECT_EQ(current_thread_vector_lengths(), expected);
}
#endif

} // namespace
