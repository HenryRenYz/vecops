#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

#include "vecops/execution/WorkspaceContext.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using namespace vecops;
using namespace vecops::execution;

TEST(WorkspaceContextTest, DynamicModeUsesFastArenaAndRewindsScopes) {
  alignas(64) std::array<std::byte, 256> fast{};
  WorkspaceContext workspace{"kernel", fast.data(), nint_t(fast.size())};
  void* first = nullptr;
  {
    auto phase = workspace.serial_scope("phase");
    auto slot = workspace.request("scratch", {.bytes = 128, .placement = WorkspacePlacementPolicy::FastRequired});
    first = slot.replica();
    EXPECT_EQ(first, fast.data());
  }
  {
    auto phase = workspace.serial_scope("next");
    auto slot = workspace.request("scratch", {.bytes = 128, .placement = WorkspacePlacementPolicy::FastRequired});
    EXPECT_EQ(slot.replica(), first);
  }
}

TEST(WorkspaceContextTest, PreferredRequestSpillsWithoutMovingLiveFastData) {
  alignas(64) std::array<std::byte, 64> fast{};
  WorkspaceContext workspace{"kernel", fast.data(), nint_t(fast.size())};
  auto phase = workspace.serial_scope("phase");
  auto resident = workspace.request("resident", {.bytes = 64, .placement = WorkspacePlacementPolicy::FastRequired});
  auto spilled = workspace.request("spilled", {.bytes = 128, .placement = WorkspacePlacementPolicy::FastPreferred});
  EXPECT_EQ(resident.replica(), fast.data());
  EXPECT_NE(spilled.replica(), nullptr);
  EXPECT_NE(spilled.replica(), resident.replica());
  *static_cast<std::byte*>(resident.replica()) = std::byte{17};
  EXPECT_EQ(fast.front(), std::byte{17});
}

TEST(WorkspaceContextTest, PreparedBindingOwnsSpilledStoragePastConstructionScope) {
  WorkspaceContext workspace{"kernel"};
  float* bound = nullptr;
  {
    auto construction = workspace.serial_scope("prepared");
    auto slot = workspace.bind("scratch", {.bytes = 4 * nint_t(sizeof(float)), .alignment = alignof(float)});
    bound = static_cast<float*>(slot.replica());
    bound[0] = 7.0f;
  }
  auto transient = workspace.serial_scope("later");
  auto other = workspace.request("scratch", {.bytes = 4096, .alignment = 64});
  EXPECT_NE(other.replica(), bound);
  EXPECT_FLOAT_EQ(bound[0], 7.0f);
}

TEST(WorkspaceContextTest, TracePlacementAndReplayShareStableSites) {
  alignas(64) std::array<std::byte, 256> trace_fast{};
  WorkspaceContext tracing{trace_workspace, "kernel", {31, 47}, trace_fast.data(), nint_t(trace_fast.size())};
  tracing.add_axis(AxisContract::from(meta::dyn<8, 8, 256>(128), 192));
  {
    auto phase = tracing.serial_scope("projection");
    auto lane =
      tracing.request("scratch", {.bytes = 33, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 4});
    EXPECT_EQ(lane.replica_stride, 64);
  }
  auto logical = tracing.finish_trace();
  auto placement = place_workspace(logical);
  std::vector<std::byte> arena(static_cast<std::size_t>(placement.fast_bytes + 64));
  BoundWorkspacePlan bound{placement, arena.data(), nint_t(arena.size()), nullptr, 0};
  ASSERT_TRUE(bound.accepts({31, 47}, {192}));

  WorkspaceContext replay{"kernel", bound};
  replay.observe_axis(AxisContract::from(meta::dyn<8, 8, 256>(192), 192));
  auto phase = replay.serial_scope("projection");
  auto lane =
    replay.request("scratch", {.bytes = 33, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 4});
  EXPECT_EQ(lane.replica_stride, 64);
  EXPECT_EQ(static_cast<std::byte*>(lane.replica(3)) - static_cast<std::byte*>(lane.replica()), 192);
  phase.close();
  replay.finish_replay();
}

TEST(WorkspaceContextTest, DefaultReplayCacheTracesEachAxisShapeOnce) {
  WorkspaceReplayCache cache{2};
  int traces = 0;
  int replays = 0;
  auto run = [&](nint_t rows) {
    cache.invoke(
      "kernel", [&](auto& workspace) { workspace.observe_axis(AxisContract::from(meta::dyn<1, 1, 256>(rows))); },
      [&](WorkspaceContext& workspace) {
        if (workspace.is_tracing())
          ++traces;
        if (workspace.is_replaying())
          ++replays;
        auto phase = workspace.serial_scope("phase");
        auto slot = workspace.request("scratch", {.bytes = rows * nint_t(sizeof(float))});
        ASSERT_NE(slot.replica(), nullptr);
      });
  };

  run(32);
  run(32);
  run(64);
  run(32);
  EXPECT_EQ(traces, 2);
  EXPECT_EQ(replays, 2);
}

#if defined(_OPENMP)
TEST(WorkspaceContextTest, ReplayCacheSeparatesAmbientParallelism) {
  WorkspaceReplayCache cache{2};
  int traces = 0;
  int replays = 0;
  const int previous_threads = omp_get_max_threads();
  auto run = [&] {
    cache.invoke(
      "parallel-kernel", [](auto&) {},
      [&](WorkspaceContext& workspace) {
        if (workspace.is_tracing())
          ++traces;
        if (workspace.is_replaying())
          ++replays;
        const nint_t workers = max_parallelism();
        auto phase = workspace.serial_scope("phase");
        auto slot = workspace.request(
          "scratch", {.bytes = 64,
                      .domain = WorkspaceDomain::WorkerLocal,
                      .replicas = workers});
        for (nint_t worker = 0; worker < workers; ++worker)
          EXPECT_NE(slot.replica(worker), nullptr);
      });
  };

  omp_set_num_threads(1);
  run();
  run();
  omp_set_num_threads(2);
  run();
  run();
  omp_set_num_threads(previous_threads);

  EXPECT_EQ(traces, 2);
  EXPECT_EQ(replays, 2);
}
#endif

} // namespace
