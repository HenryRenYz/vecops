#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "vecops/execution/WorkspaceContext.h"

namespace {

using namespace vecops;
using namespace vecops::execution;
using namespace vecops::meta;
using namespace vecops::tensor;

TEST(WorkspaceTensorTest, ShapeOnlyAllocationReturnsContiguousTensor) {
  alignas(64) std::array<std::byte, 256> arena{};
  WorkspaceContext workspace{"typed", arena.data(), nint_t(arena.size())};

  auto value = workspace.tensor<float>("dense", make_shape(cint<2>, cint<3>));

  static_assert(decltype(value)::Ndim == 2);
  EXPECT_EQ(value.data(), reinterpret_cast<float*>(arena.data()));
  EXPECT_EQ(value.size(0), 2);
  EXPECT_EQ(value.size(1), 3);
  EXPECT_EQ(value.stride(0), 3);
  EXPECT_EQ(value.stride(1), 1);
  value(1, 2) = 7.0f;
  EXPECT_FLOAT_EQ(reinterpret_cast<float*>(arena.data())[5], 7.0f);
}

TEST(WorkspaceTensorTest, ExplicitStridesAllocateThePhysicalStorageSpan) {
  WorkspaceContext tracing{trace_workspace, "typed"};
  auto value = tracing.tensor<float>(
    "padded", make_shape(cint<2>, cint<3>), make_strides(cint<5>, cint<1>));
  EXPECT_EQ(value.stride(0), 5);

  auto logical = tracing.finish_trace();
  ASSERT_EQ(logical.allocations.size(), 1u);
  // Reachable offsets are [0, 7], not merely the six logical elements.
  EXPECT_EQ(logical.allocations.front().request.bytes, 8 * nint_t(sizeof(float)));
}

TEST(WorkspaceTensorTest, WorkerTensorAddsAlignedPaddedLeadingDimension) {
  alignas(64) std::array<std::byte, 512> arena{};
  WorkspaceContext workspace{"workers", arena.data(), nint_t(arena.size())};

  auto value = workspace.worker_tensor<float, 4>("local", make_shape(cint<3>));

  static_assert(std::remove_cvref_t<decltype(value.size<0>())>::is_const);
  EXPECT_EQ(value.size(0), 4);
  EXPECT_EQ(value.size(1), 3);
  EXPECT_EQ(value.stride(0), 64 / nint_t(sizeof(float)));
  EXPECT_EQ(value.stride(1), 1);
  for (nint_t worker = 0; worker < 4; ++worker) {
    auto local = value(worker, reserve);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(local.data()) % 64, 0u);
    local[2] = static_cast<float>(worker + 1);
  }
  EXPECT_FLOAT_EQ(value(3, 2), 4.0f);
}

TEST(WorkspaceTensorTest, WorkerTensorReplicaFollowsTaskContextLane) {
  alignas(64) std::array<std::byte, 512> arena{};
  WorkspaceContext workspace{"workers", arena.data(), nint_t(arena.size())};
  auto value = workspace.worker_tensor<std::uint32_t, 4>("local", make_shape(cint<3>));

  workspace.parallel_lanes<4>([&](TaskContext<4> task) {
    auto local = task.local(value);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(local.data()) % 64, 0u);
    local[0] = static_cast<std::uint32_t>(task.lane_id() + 10);
    local[2] = static_cast<std::uint32_t>(task.lane_count());
  });

  for (nint_t lane = 0; lane < 4; ++lane) {
    EXPECT_EQ(value(lane, 0), static_cast<std::uint32_t>(lane + 10));
    EXPECT_EQ(value(lane, 2), 4u);
  }
}

TEST(WorkspaceTensorTest, TaskContextRejectsMismatchedRuntimeReplicaCount) {
  alignas(64) std::array<std::byte, 512> arena{};
  WorkspaceContext workspace{"workers", arena.data(), nint_t(arena.size())};
  auto value = workspace.worker_tensor<std::uint32_t>(
    "local", nint_t{3}, make_shape(cint<3>));

  EXPECT_THROW((void)TaskContext<4>{0}.local(value), std::runtime_error);
}

TEST(WorkspaceTensorTest, RuntimeReplicasAndPaddedInnerLayoutCompose) {
  WorkspaceContext tracing{trace_workspace, "workers"};
  const nint_t replicas = 3;
  auto value = tracing.worker_tensor<std::uint16_t>(
    "local", replicas, make_shape(cint<2>, cint<3>), make_strides(cint<8>, cint<2>));

  EXPECT_EQ(value.size(0), replicas);
  EXPECT_EQ(value.stride(0), 64 / nint_t(sizeof(std::uint16_t)));
  EXPECT_EQ(value.stride(1), 8);
  EXPECT_EQ(value.stride(2), 2);

  auto logical = tracing.finish_trace();
  ASSERT_EQ(logical.allocations.size(), 1u);
  // Inner reachable offsets are [0, 12], hence thirteen uint16_t elements.
  EXPECT_EQ(logical.allocations.front().request.bytes, 13 * nint_t(sizeof(std::uint16_t)));
  auto placement = place_workspace(logical);
  ASSERT_EQ(placement.entries.size(), 1u);
  EXPECT_EQ(placement.entries.front().replica_stride, 64);
  EXPECT_EQ(placement.fast_bytes, 3 * 64);
}

TEST(WorkspaceTensorTest, RejectsUnsafeWritableLayouts) {
  WorkspaceContext workspace{"invalid"};
  EXPECT_THROW((workspace.tensor<float>(
                 "overlap", make_shape(cint<2>, cint<2>), make_strides(cint<1>, cint<1>))),
               std::runtime_error);
  EXPECT_THROW((workspace.tensor<float>(
                 "negative", make_shape(cint<2>, cint<2>), make_strides(cint<2>, meta::Any{-1}))),
               std::runtime_error);
}

TEST(WorkspaceTensorTest, TypedRequestsReplayThroughTheBytePlanner) {
  WorkspaceContext tracing{trace_workspace, "typed-replay"};
  auto traced = tracing.worker_tensor<std::uint8_t, 2>(
    "local", make_shape(cint<17>), WorkspaceTensorOptions{.placement = WorkspacePlacementPolicy::SlowAllowed});
  ASSERT_NE(traced.data(), nullptr);
  auto logical = tracing.finish_trace();
  auto placement = place_workspace(logical, 0, 1024);
  std::vector<std::byte> slow(static_cast<std::size_t>(placement.slow_bytes + 64));
  BoundWorkspacePlan bound{placement, nullptr, 0, slow.data(), nint_t(slow.size())};
  WorkspaceContext replay{"typed-replay", bound};

  auto replayed = replay.worker_tensor<std::uint8_t, 2>(
    "local", make_shape(cint<17>), WorkspaceTensorOptions{.placement = WorkspacePlacementPolicy::SlowAllowed});
  EXPECT_EQ(replayed.size(0), 2);
  EXPECT_EQ(replayed.stride(0), 64);
  EXPECT_NE(replayed.data(), nullptr);
  replay.finish_replay();
}

} // namespace
