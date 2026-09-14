#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "vecops/execution/WorkspaceContext.h"

#if defined(_OPENMP)
#  include <omp.h>
#endif

namespace {

using namespace vecops;
using namespace vecops::execution;

class FailingFastArenaProvider final : public WorkspaceArenaProvider {
public:
  [[nodiscard]] nint_t capacity(WorkspaceTier) const noexcept override {
    return std::numeric_limits<nint_t>::max();
  }

  WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t alignment) override {
    if (tier == WorkspaceTier::Fast && bytes != 0) {
      ++fast_attempts;
      throw std::bad_alloc();
    }
    if (tier == WorkspaceTier::Slow && bytes != 0)
      ++slow_allocations;
    return heap.allocate(tier, bytes, alignment);
  }

  int fast_attempts = 0;
  int slow_allocations = 0;
  HeapWorkspaceArenaProvider heap;
};

class SharedArenaProvider final : public WorkspaceArenaProvider {
public:
  [[nodiscard]] nint_t capacity(WorkspaceTier) const noexcept override {
    return 256;
  }

  [[nodiscard]] bool shares_arenas_between_plans() const noexcept override {
    return true;
  }

  WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t) override {
    if (bytes == 0)
      return {};
    auto& storage = tier == WorkspaceTier::Fast ? fast : slow;
    std::shared_ptr<void> owner(storage.data(), [](void*) {});
    return {storage.data(), nint_t(storage.size()), std::move(owner)};
  }

  alignas(64) std::array<std::byte, 256> fast{};
  alignas(64) std::array<std::byte, 256> slow{};
};

class UnavailableArenaProvider final : public WorkspaceArenaProvider {
public:
  [[nodiscard]] nint_t capacity(WorkspaceTier) const noexcept override {
    return 4096;
  }

  WorkspaceArena allocate(WorkspaceTier, nint_t bytes, nint_t) override {
    if (bytes != 0)
      throw WorkspaceArenaUnavailable("unavailable for cache retention");
    return {};
  }
};

struct AbiArenaProviderState {
  static std::uint64_t capacity(void*, std::uint32_t) {
    return 4096;
  }

  static std::int32_t allocate(void* context, std::uint32_t tier, std::uint64_t bytes, std::uint64_t alignment,
                               VecopsWorkspaceArena* result, VecopsError*) {
    auto& self = *static_cast<AbiArenaProviderState*>(context);
    if (bytes == 0) {
      *result = {sizeof(VecopsWorkspaceArena), 0, nullptr, 0, nullptr};
      return VECOPS_STATUS_OK;
    }
    const auto logical_tier = tier == VECOPS_WORKSPACE_TIER_FAST ? WorkspaceTier::Fast : WorkspaceTier::Slow;
    auto* owner =
      new WorkspaceArena(self.heap.allocate(logical_tier, static_cast<nint_t>(bytes), static_cast<nint_t>(alignment)));
    *result = {sizeof(VecopsWorkspaceArena), 0, owner->data, static_cast<std::uint64_t>(owner->capacity), owner};
    ++self.allocations;
    return VECOPS_STATUS_OK;
  }

  static void release_arena(void* context, void* owner) {
    ++static_cast<AbiArenaProviderState*>(context)->releases;
    delete static_cast<WorkspaceArena*>(owner);
  }

  static void retain(void* context) {
    ++static_cast<AbiArenaProviderState*>(context)->references;
  }

  static void release_context(void* context) {
    --static_cast<AbiArenaProviderState*>(context)->references;
  }

  HeapWorkspaceArenaProvider heap;
  int references = 1;
  int allocations = 0;
  int releases = 0;
};

struct AbiThreadPoolState {
  static std::uint32_t maximum(void* context) {
    return static_cast<AbiThreadPoolState*>(context)->maximum_threads;
  }

  static std::uint32_t in_parallel(void*) {
    return 0;
  }

  static std::int32_t parallel_for(void* context, std::uint32_t count, void* body_context, VecopsParallelTaskFn body,
                                   VecopsError*) {
    auto& self = *static_cast<AbiThreadPoolState*>(context);
    ++self.calls;
    self.last_task_count = count;
    for (std::uint32_t task = count; task-- > 0;)
      body(body_context, task, count);
    return VECOPS_STATUS_OK;
  }

  std::uint32_t maximum_threads = 8;
  std::uint32_t calls = 0;
  std::uint32_t last_task_count = 0;
};

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

TEST(WorkspaceContextTest, ParallelTasksUseCallPoolAndKeepStaticTaskCount) {
  AbiThreadPoolState state;
  const VecopsThreadPoolV1 pool{sizeof(VecopsThreadPoolV1),
                                VECOPS_THREAD_POOL_ABI_MAJOR,
                                VECOPS_THREAD_POOL_ABI_MINOR,
                                17,
                                &state,
                                AbiThreadPoolState::maximum,
                                AbiThreadPoolState::in_parallel,
                                AbiThreadPoolState::parallel_for,
                                nullptr,
                                nullptr,
                                0};
  const VecopsExecutionContext execution{
    .struct_size = sizeof(VecopsExecutionContext),
    .requested_threads = 6,
    .thread_pool = &pool,
  };
  WorkspaceContext workspace{"parallel", nullptr, 0, nullptr, 0, &execution};
  EXPECT_EQ(workspace.parallelism(), 6);

  std::array<int, 4> visits{};
  workspace.parallel_tasks<4>([&](auto worker) {
    static_assert(decltype(worker)::num_threads == 4);
    ++visits[static_cast<std::size_t>(worker.thread_id)];
  });
  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(state.last_task_count, 4);
  EXPECT_EQ(visits, (std::array<int, 4>{1, 1, 1, 1}));
}

TEST(WorkspaceContextTest, ParallelLanesAndRangesUseTheBoundCallPool) {
  AbiThreadPoolState state;
  const VecopsThreadPoolV1 pool{sizeof(VecopsThreadPoolV1),
                                VECOPS_THREAD_POOL_ABI_MAJOR,
                                VECOPS_THREAD_POOL_ABI_MINOR,
                                17,
                                &state,
                                AbiThreadPoolState::maximum,
                                AbiThreadPoolState::in_parallel,
                                AbiThreadPoolState::parallel_for,
                                nullptr,
                                nullptr,
                                0};
  const VecopsExecutionContext execution{
    .struct_size = sizeof(VecopsExecutionContext),
    .requested_threads = 6,
    .thread_pool = &pool,
  };
  WorkspaceContext workspace{"parallel", nullptr, 0, nullptr, 0, &execution};

  std::array<int, 3> lane_visits{};
  workspace.parallel_lanes<3>([&](TaskContext<3> task) {
    static_assert(TaskContext<3>::lane_count() == 3);
    ++lane_visits[static_cast<std::size_t>(task.lane_id())];
  });
  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(state.last_task_count, 3);
  EXPECT_EQ(lane_visits, (std::array<int, 3>{1, 1, 1}));

  std::array<int, 11> range_visits{};
  workspace.parallel_for<3>(meta::cint<0>, meta::cint<11>, meta::cint<4>, [&](auto task, auto item) {
    static_assert(decltype(task)::lane_count() == 3);
    static_assert(std::remove_cvref_t<decltype(item.chunk())>::value == 4);
    for (nint_t index = item.begin(); index < item.end(); ++index)
      ++range_visits[static_cast<std::size_t>(index)];
  });
  EXPECT_EQ(state.calls, 2);
  EXPECT_EQ(state.last_task_count, 3);
  EXPECT_EQ(range_visits, (std::array<int, 11>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(WorkspaceContextTest, BalancedShardHasCompileTimeMainAndTailExtents) {
  WorkspaceContext workspace{"balanced"};
  std::array<int, 10> visits{};
  workspace.parallel_tasks<4>([&](auto worker) {
    balanced_shard<10>(worker, [&](auto shard) {
      static_assert(decltype(shard)::extent == 3 || decltype(shard)::extent == 2);
      for (nint_t index = shard.begin; index < shard.end(); ++index)
        ++visits[static_cast<std::size_t>(index)];
    });
  });
  EXPECT_EQ(visits, (std::array<int, 10>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
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

TEST(WorkspaceContextTest, SerialUsesColorSequentialWorkerScratchToMaximum) {
  WorkspaceContext tracing{trace_workspace, "serial_operators"};
  auto first = tracing.serial_use("first", [](auto& scratch) {
    return scratch.template worker_tensor<std::byte, 2>("scratch", tensor::make_shape(meta::cint<80>));
  });
  auto second = tracing.serial_use("second", [](auto& scratch) {
    return scratch.template worker_tensor<std::byte, 2>("scratch", tensor::make_shape(meta::cint<144>));
  });
  (void)first;
  (void)second;

  const auto logical = tracing.finish_trace();
  ASSERT_EQ(logical.allocations.size(), 2u);
  const auto placement = place_workspace(logical);
  ASSERT_EQ(placement.entries.size(), 2u);
  EXPECT_EQ(placement.fast_bytes, 384);
  EXPECT_EQ(placement.entries[0].offset, placement.entries[1].offset);
}

TEST(WorkspaceContextTest, IndependentWorkerScratchInSameScopeAddsItsPeak) {
  WorkspaceContext tracing{trace_workspace, "independent_operators"};
  auto scope = tracing.serial_scope("pipeline");
  auto first = tracing.worker_tensor<std::byte, 2>("first", tensor::make_shape(meta::cint<80>));
  auto second = tracing.worker_tensor<std::byte, 2>("second", tensor::make_shape(meta::cint<144>));
  (void)first;
  (void)second;
  scope.close();

  const auto logical = tracing.finish_trace();
  ASSERT_EQ(logical.allocations.size(), 2u);
  const auto placement = place_workspace(logical);
  ASSERT_EQ(placement.entries.size(), 2u);
  EXPECT_EQ(placement.fast_bytes, 640);
  EXPECT_NE(placement.entries[0].offset, placement.entries[1].offset);
}

TEST(WorkspaceContextTest, DifferentStrideWorkerSitesIsolateLaneSkew) {
  alignas(64) std::array<std::byte, 1024> fast{};
  WorkspaceContext workspace{"lane_skew", fast.data(), static_cast<nint_t>(fast.size())};
  auto small = workspace.worker_tensor<std::byte, 2>("small", tensor::make_shape(meta::cint<80>));
  auto large = workspace.worker_tensor<std::byte, 2>("large", tensor::make_shape(meta::cint<144>));

  std::fill_n(small.data() + small.stride(0), 80, std::byte{0x35});
  std::fill_n(large.data(), 144, std::byte{0x6a});
  for (nint_t index = 0; index < 80; ++index)
    EXPECT_EQ(small(1, index), std::byte{0x35});
  for (nint_t index = 0; index < 144; ++index)
    EXPECT_EQ(large(0, index), std::byte{0x6a});
}

TEST(WorkspaceContextTest, SerialUseOwnsDynamicSpillAfterConstructionScope) {
  WorkspaceContext workspace{"dynamic_serial_operator"};
  auto scratch = workspace.serial_use("prepared", [](auto& authority) {
    return authority.template worker_tensor<float, 2>("scratch", tensor::make_shape(meta::cint<4>));
  });
  scratch(0, 0) = 3.0f;
  scratch(1, 3) = 9.0f;

  {
    auto later = workspace.serial_scope("later");
    auto transient = workspace.request("scratch", {.bytes = 4096});
    ASSERT_NE(transient.replica(), nullptr);
    EXPECT_NE(transient.replica(), scratch.data());
  }
  EXPECT_FLOAT_EQ(scratch(0, 0), 3.0f);
  EXPECT_FLOAT_EQ(scratch(1, 3), 9.0f);
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

TEST(WorkspaceContextTest, PreferredArenaFallsBackAsACompleteSlowPlan) {
  auto provider = std::make_shared<FailingFastArenaProvider>();
  WorkspaceReplayCache cache{1, provider};
  cache.invoke(
    "fallback", [](auto&) {},
    [](WorkspaceContext& workspace) {
      auto phase = workspace.serial_scope("phase");
      auto slot = workspace.request("preferred", {.bytes = 128});
      ASSERT_NE(slot.replica(), nullptr);
    });

  EXPECT_EQ(provider->fast_attempts, 1);
  EXPECT_EQ(provider->slow_allocations, 1);
}

TEST(WorkspaceContextTest, SharedSessionArenaServesFirstTraceAndMultiplePlans) {
  auto provider = std::make_shared<SharedArenaProvider>();
  WorkspaceReplayCache cache{2, provider};
  std::array<void*, 2> observed{};
  for (nint_t shape = 1; shape <= 2; ++shape) {
    cache.invoke(
      "shared", [&](auto& workspace) { workspace.observe_axis(AxisContract::from(shape)); },
      [&](WorkspaceContext& workspace) {
        auto phase = workspace.serial_scope("phase");
        observed[static_cast<std::size_t>(shape - 1)] = workspace.request("buffer", {.bytes = 128}).replica();
      });
  }
  EXPECT_EQ(observed[0], provider->fast.data());
  EXPECT_EQ(observed[1], provider->fast.data());
}

TEST(WorkspaceContextTest, CacheRetentionFailureDoesNotFailCompletedInvocation) {
  auto provider = std::make_shared<UnavailableArenaProvider>();
  WorkspaceReplayCache cache{1, provider};
  int completed = 0;
  EXPECT_NO_THROW(cache.invoke(
    "nonfatal-cache", [](auto&) {},
    [&](WorkspaceContext& workspace) {
      auto phase = workspace.serial_scope("phase");
      auto slot = workspace.request("buffer", {.bytes = 128});
      ASSERT_NE(slot.replica(), nullptr);
      ++completed;
    }));
  EXPECT_EQ(completed, 1);
}

TEST(WorkspaceContextTest, CAbiProviderRetainsContextAndArenaOwnership) {
  AbiArenaProviderState state;
  VecopsWorkspaceArenaProvider abi{sizeof(VecopsWorkspaceArenaProvider),
                                   0,
                                   17,
                                   &state,
                                   AbiArenaProviderState::capacity,
                                   AbiArenaProviderState::allocate,
                                   AbiArenaProviderState::release_arena,
                                   AbiArenaProviderState::retain,
                                   AbiArenaProviderState::release_context,
                                   0};
  {
    WorkspaceReplayCache cache{1, workspace_arena_provider_from_abi(abi)};
    EXPECT_EQ(state.references, 2);
    cache.invoke(
      "abi", [](auto&) {},
      [](WorkspaceContext& workspace) {
        auto phase = workspace.serial_scope("phase");
        auto slot = workspace.request("preferred", {.bytes = 128});
        ASSERT_NE(slot.replica(), nullptr);
      });
    EXPECT_EQ(state.allocations, 1);
    EXPECT_EQ(state.releases, 0);
  }
  EXPECT_EQ(state.releases, 1);
  EXPECT_EQ(state.references, 1);
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
        auto slot =
          workspace.request("scratch", {.bytes = 64, .domain = WorkspaceDomain::WorkerLocal, .replicas = workers});
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
