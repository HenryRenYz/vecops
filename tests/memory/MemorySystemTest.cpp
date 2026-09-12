//
// Copyright (c) vecops contributors.
//

#include "vecops/memory/Memory.h"
#include "vecops/execution/MemoryWorkspaceSession.h"
#include "vecops/execution/WorkspaceContext.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <cstring>

#include <gtest/gtest.h>

namespace vecops::memory {
namespace {

MemorySystem system_backend(MemoryConfig config = {}) {
  config.backend = BackendPreference::System;
  return MemorySystem::discover(config);
}

MemorySystem test_backend(MemoryConfig config = {}) {
  config.backend = BackendPreference::Auto;
  return MemorySystem::discover(config);
}

std::uint64_t total_managed(const MemorySystem& memory) {
  std::uint64_t result = 0;
  for (const auto& stats : memory.stats())
    result += stats.managed_bytes;
  return result;
}

#if defined(__linux__)
std::map<unsigned, std::uint64_t> resident_nodes(const void* pointer) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  std::uintptr_t closest = 0;
  std::string mapping;
  std::ifstream input("/proc/self/numa_maps");
  for (std::string line; std::getline(input, line);) {
    std::istringstream parser(line);
    std::string begin_text;
    parser >> begin_text;
    const auto begin = static_cast<std::uintptr_t>(std::stoull(begin_text, nullptr, 16));
    if (begin <= address && begin >= closest) {
      closest = begin;
      mapping = std::move(line);
    }
  }

  std::map<unsigned, std::uint64_t> result;
  std::istringstream parser(mapping);
  for (std::string token; parser >> token;) {
    if (token.size() < 4 || token.front() != 'N')
      continue;
    const auto equal = token.find('=');
    if (equal == std::string::npos)
      continue;
    result.emplace(static_cast<unsigned>(std::stoul(token.substr(1, equal - 1))), std::stoull(token.substr(equal + 1)));
  }
  return result;
}
#endif

TEST(MemorySystemTest, SystemBackendHasOneUsableDomainAndTarget) {
  MemorySystem memory;
  try {
    memory = system_backend();
  } catch (const MemoryError& error) {
    if (error.code() == MemoryErrc::UnsupportedBinding)
      GTEST_SKIP() << error.what();
    throw;
  }
  const auto& topology = memory.topology();

  ASSERT_EQ(topology.backend, "system");
  ASSERT_EQ(topology.cpu_domains.size(), 1u);
  ASSERT_EQ(topology.memory_targets.size(), 1u);
  ASSERT_EQ(topology.memory_paths.size(), 1u);
  EXPECT_EQ(memory.current_cpu_domain(), 0u);
  EXPECT_TRUE(topology.memory_paths.front().exact_locality);
  EXPECT_FALSE(memory.describe().empty());
}

TEST(MemorySystemTest, AutoBackendDiscoversAndAllocates) {
  const auto memory = MemorySystem::discover();
  ASSERT_FALSE(memory.topology().memory_targets.empty());
  ASSERT_FALSE(memory.topology().cpu_domains.empty());

  const auto target = memory.topology().memory_targets.front().os_numa_id;
  auto allocation = memory.allocate({
    .bytes = 8192,
    .intent = PlacementIntent::ExactTarget,
    .exact_os_numa_id = target,
    .fallback = FallbackPolicy::None,
    .alignment = 4096,
  });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(allocation.data()) % 4096, 0u);
}

#if defined(__linux__)
TEST(MemorySystemTest, TouchedPagesResideOnTheSelectedTarget) {
  const auto memory = MemorySystem::discover();
  constexpr std::size_t bytes = 8 * 1024 * 1024;
  auto allocation = memory.allocate({
    .bytes = bytes,
    .intent = PlacementIntent::HighBandwidth,
    .fallback = FallbackPolicy::None,
    .alignment = 4096,
  });
  ASSERT_TRUE(allocation.target().has_value());
  auto* data = static_cast<std::byte*>(allocation.data());
  for (std::size_t offset = 0; offset < bytes; offset += 4096)
    data[offset] = std::byte{1};

  const auto nodes = resident_nodes(data);
  ASSERT_FALSE(nodes.empty());
  const auto expected = memory.topology().memory_targets[allocation.target().value()].os_numa_id;
  EXPECT_EQ(nodes.size(), 1u);
  EXPECT_NE(nodes.find(expected), nodes.end());
}
#endif

TEST(MemorySystemTest, DirectAllocationIsAlignedAndUpdatesStats) {
  const auto memory = test_backend();
  {
    auto allocation = memory.allocate({.bytes = 4097, .alignment = 4096});
    ASSERT_TRUE(allocation);
    EXPECT_EQ(allocation.size(), 4097u);
    ASSERT_TRUE(allocation.target().has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(allocation.data()) % 4096, 0u);

    const auto stats = memory.stats();
    const auto& target_stats = stats[allocation.target().value()];
    EXPECT_EQ(target_stats.managed_bytes, 4097u);
    EXPECT_EQ(target_stats.peak_managed_bytes, 4097u);
    EXPECT_EQ(target_stats.allocation_count, 1u);
  }
  EXPECT_EQ(total_managed(memory), 0u);
}

TEST(MemorySystemTest, AllocationMoveTransfersOwnership) {
  const auto memory = test_backend();
  auto first = memory.allocate({.bytes = 1024});
  auto second = std::move(first);

  EXPECT_FALSE(first);
  EXPECT_EQ(first.size(), 0u);
  EXPECT_TRUE(second);
  EXPECT_EQ(total_managed(memory), 1024u);

  second.reset();
  EXPECT_EQ(total_managed(memory), 0u);
}

TEST(MemorySystemTest, AllocationKeepsBackendAlive) {
  Allocation allocation;
  {
    const auto memory = test_backend();
    allocation = memory.allocate({.bytes = 1024});
  }
  EXPECT_TRUE(allocation);
  allocation.reset();
  EXPECT_FALSE(allocation);
}

TEST(MemorySystemTest, ZeroBytesDoesNotTouchBackend) {
  const auto memory = test_backend();
  const auto allocation = memory.allocate({.bytes = 0});
  EXPECT_FALSE(allocation);
  for (const auto& stats : memory.stats())
    EXPECT_EQ(stats.allocation_count, 0u);
}

TEST(MemorySystemTest, InvalidAlignmentIsTyped) {
  const auto memory = test_backend();
  try {
    (void)memory.allocate({.bytes = 64, .alignment = 3});
    FAIL() << "allocation unexpectedly succeeded";
  } catch (const MemoryError& error) {
    EXPECT_EQ(error.code(), MemoryErrc::InvalidRequest);
  }
}

TEST(MemorySystemTest, TierRankRequiresAnObjectiveIntent) {
  const auto memory = test_backend();
  try {
    (void)memory.allocate({.bytes = 64, .objective_rank = 0});
    FAIL() << "allocation unexpectedly succeeded";
  } catch (const MemoryError& error) {
    EXPECT_EQ(error.code(), MemoryErrc::InvalidRequest);
  }
}

TEST(MemorySystemTest, ManagedBudgetRejectsOversizedAllocation) {
  MemoryConfig config;
  config.target_overrides.push_back(TargetOverride{.max_managed_bytes = 1024});
  const auto memory = test_backend(std::move(config));

  try {
    (void)memory.allocate({.bytes = 1025});
    FAIL() << "allocation unexpectedly succeeded";
  } catch (const MemoryError& error) {
    EXPECT_EQ(error.code(), MemoryErrc::BudgetExceeded);
  }
  std::uint64_t failures = 0;
  for (const auto& stats : memory.stats()) {
    EXPECT_EQ(stats.managed_bytes, 0u);
    failures += stats.failed_allocation_count;
  }
  EXPECT_EQ(failures, 1u);
}

TEST(MemorySystemTest, PartialOverridesAreReported) {
  MemoryConfig config;
  config.target_overrides.push_back(TargetOverride{.os_numa_id = 0, .kind = MemoryKind::HBM});
  config.path_overrides.push_back(PathOverride{
    .target_os_numa_id = 0,
    .bandwidth_mib_s = 123456,
    .latency_ns = 42,
  });
  const auto memory = test_backend(std::move(config));

  const auto target = std::find_if(memory.topology().memory_targets.begin(), memory.topology().memory_targets.end(),
                                   [](const auto& item) { return item.os_numa_id == 0; });
  ASSERT_NE(target, memory.topology().memory_targets.end());
  EXPECT_EQ(target->kind, MemoryKind::HBM);
  const auto path = std::find_if(memory.topology().memory_paths.begin(), memory.topology().memory_paths.end(),
                                 [&](const auto& item) { return item.target == target->id; });
  ASSERT_NE(path, memory.topology().memory_paths.end());
  EXPECT_EQ(path->bandwidth_mib_s, 123456u);
  EXPECT_EQ(path->latency_ns, 42u);
  EXPECT_EQ(path->bandwidth_source, AttributeSource::Configuration);
  EXPECT_EQ(path->latency_source, AttributeSource::Configuration);

  const auto bandwidth = memory.tiers(0, RankingObjective::Bandwidth);
  ASSERT_FALSE(bandwidth.ranks.empty());
}

TEST(MemorySystemTest, ExplicitUnknownTargetIsTyped) {
  const auto memory = test_backend();
  try {
    (void)memory.allocate({
      .bytes = 64,
      .intent = PlacementIntent::ExactTarget,
      .exact_os_numa_id = std::numeric_limits<unsigned>::max(),
    });
    FAIL() << "allocation unexpectedly succeeded";
  } catch (const MemoryError& error) {
    EXPECT_EQ(error.code(), MemoryErrc::UnknownTarget);
  }
}

TEST(MemorySystemTest, WorkspaceReplayCacheUsesInjectedMemoryProvider) {
  const auto memory = test_backend();
  auto provider = std::make_shared<execution::MemoryWorkspaceSession>(
    memory, execution::MemoryWorkspaceSessionConfig{.fast_capacity = 4096, .slow_capacity = 4096});
  execution::WorkspaceReplayCache cache(1, provider);

  auto invoke = [&] {
    cache.invoke(
      "memory-provider-test", [](auto&) {},
      [](execution::WorkspaceContext& workspace) {
        auto scope = workspace.serial_scope("body");
        auto slot =
          workspace.request("fast", {.bytes = 1024, .placement = execution::WorkspacePlacementPolicy::FastPreferred});
        std::memset(slot.replica(), 0x5a, static_cast<std::size_t>(slot.bytes));
      });
  };
  invoke();
  invoke();

  std::uint64_t managed = 0;
  std::uint64_t allocations = 0;
  const auto stats = memory.stats();
  for (const auto& item : stats) {
    managed += item.managed_bytes;
    allocations += item.allocation_count;
  }
  EXPECT_EQ(managed, 8192u);
  EXPECT_EQ(allocations, 2u);
  const auto domain = memory.current_cpu_domain();
  const auto bandwidth = memory.tiers(domain, RankingObjective::Bandwidth);
  ASSERT_FALSE(bandwidth.ranks.empty());
  ASSERT_FALSE(bandwidth.ranks.front().targets.empty());
  EXPECT_GE(stats[bandwidth.ranks.front().targets.front()].managed_bytes, 4096u);

  provider->close();
  EXPECT_EQ(total_managed(memory), 0u);
}

} // namespace
} // namespace vecops::memory
