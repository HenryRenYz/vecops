// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vecops/execution/WorkspacePlan.h"

namespace {

using namespace vecops;
using namespace vecops::execution;

TEST(WorkspacePlanTest, SequentialScopesReuseButOverlappingSitesDoNot) {
  WorkspaceTrace trace{"kernel", {11, 22}};
  WorkspaceSiteId first;
  WorkspaceSiteId second;
  {
    auto phase = trace.serial_scope("first");
    first = trace.request("scratch", {.bytes = 256});
    trace.request("peer", {.bytes = 128});
  }
  {
    auto phase = trace.serial_scope("second");
    second = trace.request("scratch", {.bytes = 192});
  }
  auto logical = trace.finish();
  auto placement = place_workspace(logical, 1024, 1024);
  ASSERT_EQ(placement.entries.size(), 3u);
  EXPECT_EQ(placement.fast_bytes, 384);

  std::vector<std::byte> fast(static_cast<std::size_t>(placement.fast_bytes + 64));
  BoundWorkspacePlan bound{placement, fast.data(), nint_t(fast.size()), nullptr, 0};
  EXPECT_NE(bound.find(first).base, nullptr);
  EXPECT_NE(bound.find(second).base, nullptr);
}

TEST(WorkspacePlanTest, WorkerLocalSlotsArePaddedAndReplicated) {
  WorkspaceTrace trace{"parallel"};
  auto workers = trace.serial_scope("workers");
  const auto site =
    trace.request("lane", {.bytes = 33, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 4});
  workers.close();
  auto placement = place_workspace(trace.finish());
  ASSERT_EQ(placement.entries.size(), 1u);
  EXPECT_EQ(placement.entries[0].replica_stride, 64);
  EXPECT_EQ(placement.entries[0].alignment, 64);
  EXPECT_EQ(placement.fast_bytes, 256);

  std::vector<std::byte> fast(placement.fast_bytes + 64);
  BoundWorkspacePlan bound{placement, fast.data(), nint_t(fast.size()), nullptr, 0};
  const auto slot =
    bound.find(site, {.bytes = 33, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 4});
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(slot.replica()) % 64, 0u);
  EXPECT_EQ(static_cast<std::byte*>(slot.replica(3)) - static_cast<std::byte*>(slot.replica(0)), 192);
}

TEST(WorkspacePlanTest, PlacementSpillsPreferredButKeepsRequiredFast) {
  WorkspaceTrace trace{"tiers"};
  auto phase = trace.serial_scope("all");
  const auto required = trace.request("required", {.bytes = 128, .placement = WorkspacePlacementPolicy::FastRequired});
  const auto preferred =
    trace.request("preferred", {.bytes = 128, .placement = WorkspacePlacementPolicy::FastPreferred});
  phase.close();
  auto placement = place_workspace(trace.finish(), 128, 128);
  ASSERT_EQ(placement.entries.size(), 2u);
  for (const auto& entry : placement.entries) {
    if (entry.site == required)
      EXPECT_EQ(entry.tier, WorkspaceTier::Fast);
    if (entry.site == preferred)
      EXPECT_EQ(entry.tier, WorkspaceTier::Slow);
  }
}

TEST(WorkspacePlanTest, RepeatedDynamicSiteWidensCapacityAndReplicas) {
  WorkspaceTrace trace{"dynamic-repeat"};
  WorkspaceSiteId site;
  {
    auto phase = trace.serial_scope("phase");
    site =
      trace.request("scratch", {.bytes = 33, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 2});
  }
  {
    auto phase = trace.serial_scope("phase");
    trace.request("scratch", {.bytes = 97, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 4});
  }
  auto logical = trace.finish();
  ASSERT_EQ(logical.allocations.size(), 1u);
  EXPECT_EQ(logical.allocations.front().request.bytes, 97);
  EXPECT_EQ(logical.allocations.front().request.replicas, 4);
  EXPECT_EQ(logical.allocations.front().lifetimes.size(), 2u);

  auto placement = place_workspace(logical);
  std::vector<std::byte> fast(placement.fast_bytes + 64);
  BoundWorkspacePlan bound{placement, fast.data(), nint_t(fast.size()), nullptr, 0};
  auto smaller =
    bound.find(site, {.bytes = 65, .alignment = 16, .domain = WorkspaceDomain::WorkerLocal, .replicas = 3});
  EXPECT_GE(smaller.bytes, 97);
  EXPECT_GE(smaller.replicas, 4);
}

TEST(WorkspacePlanTest, SimultaneouslyLiveDuplicateSiteIsRejectedInReleaseBuilds) {
  WorkspaceTrace trace{"duplicate-live-site"};
  auto phase = trace.serial_scope("phase");
  trace.request("scratch", {.bytes = 64});
  EXPECT_THROW(trace.request("scratch", {.bytes = 64}), std::runtime_error);
}

TEST(WorkspacePlanTest, DynamicAxisAndDecisionFingerprintGuardReplay) {
  WorkspaceTrace trace{"guard", {7, 9}};
  trace.add_axis(AxisContract::from(meta::dyn<8, 8, 256>(128), 192));
  {
    auto phase = trace.serial_scope("phase");
    trace.request("scratch", {.bytes = 64});
  }
  auto placement = place_workspace(trace.finish());
  std::vector<std::byte> fast(placement.fast_bytes + 64);
  BoundWorkspacePlan bound{placement, fast.data(), nint_t(fast.size()), nullptr, 0};
  EXPECT_TRUE(bound.accepts({7, 9}, {192}));
  EXPECT_FALSE(bound.accepts({7, 9}, {200}));
  EXPECT_FALSE(bound.accepts({7, 10}, {128}));
}

} // namespace
