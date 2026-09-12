//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MEMORY_TYPES_H
#define VECOPS_MEMORY_TYPES_H

/**
 * @file vecops/memory/Types.h
 * @brief Framework-neutral topology, placement, and telemetry value types.
 *
 * OS NUMA indexes are the stable identifiers used by configuration and
 * diagnostics. Dense CpuDomainId and MemoryTargetId values are valid only
 * inside the immutable TopologySnapshot that created them.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace vecops::memory {

using CpuDomainId = std::uint32_t;
using MemoryTargetId = std::uint32_t;

enum class MemoryKind : std::uint8_t { Unknown, DRAM, HBM, CXL, PMEM };
enum class AttributeSource : std::uint8_t { Unknown, Hwloc, Configuration };
enum class RankingObjective : std::uint8_t { Bandwidth, Latency };
enum class PlacementIntent : std::uint8_t { Default, HighBandwidth, LowLatency, ExactTarget };
enum class FallbackPolicy : std::uint8_t { None, ToDefault };
enum class BackendPreference : std::uint8_t { Auto, System, Hwloc };

enum class MemoryErrc : std::uint8_t {
  InvalidRequest,
  UnknownCpuDomain,
  UnknownTarget,
  UnsupportedBinding,
  BudgetExceeded,
  OutOfMemory,
  BindingFailed,
  ConfigurationInvalid,
};

/** Select the current execution domain or a dense domain in one snapshot. */
struct CpuDomainSelector {
  enum class Kind : std::uint8_t { Current, Specific };

  Kind kind = Kind::Current;
  CpuDomainId id = 0;

  static constexpr CpuDomainSelector current() noexcept {
    return {};
  }

  static constexpr CpuDomainSelector specific(CpuDomainId value) noexcept {
    return {Kind::Specific, value};
  }
};

struct CpuDomainInfo {
  CpuDomainId id = 0;
  std::optional<unsigned> os_numa_id;
  std::vector<unsigned> cpu_ids;
};

struct MemoryTargetInfo {
  MemoryTargetId id = 0;
  unsigned os_numa_id = 0;
  MemoryKind kind = MemoryKind::Unknown;
  std::uint64_t capacity_bytes = 0;
};

struct MemoryPathInfo {
  CpuDomainId initiator = 0;
  MemoryTargetId target = 0;
  std::optional<std::uint64_t> bandwidth_mib_s;
  std::optional<std::uint64_t> latency_ns;
  AttributeSource bandwidth_source = AttributeSource::Unknown;
  AttributeSource latency_source = AttributeSource::Unknown;
  bool exact_locality = false;
};

struct MemoryTierRank {
  unsigned rank = 0;
  std::vector<MemoryTargetId> targets;
  std::optional<std::uint64_t> representative_value;
};

struct MemoryTierView {
  CpuDomainId initiator = 0;
  RankingObjective objective = RankingObjective::Bandwidth;
  std::vector<MemoryTierRank> ranks;
};

struct TopologySnapshot {
  std::string backend;
  std::vector<CpuDomainInfo> cpu_domains;
  std::vector<MemoryTargetInfo> memory_targets;
  std::vector<MemoryPathInfo> memory_paths;
};

/** Partial target override. A missing OS index applies to every target. */
struct TargetOverride {
  std::optional<unsigned> os_numa_id;
  std::optional<MemoryKind> kind;
  std::optional<std::uint64_t> max_managed_bytes;
  std::optional<std::uint64_t> min_free_bytes;
};

/** Partial initiator-to-target performance override. */
struct PathOverride {
  std::optional<unsigned> initiator_os_numa_id;
  unsigned target_os_numa_id = 0;
  std::optional<std::uint64_t> bandwidth_mib_s;
  std::optional<std::uint64_t> latency_ns;
};

struct MemoryConfig {
  BackendPreference backend = BackendPreference::Auto;
  std::vector<TargetOverride> target_overrides;
  std::vector<PathOverride> path_overrides;
};

struct AllocationRequest {
  std::size_t bytes = 0;
  CpuDomainSelector domain = CpuDomainSelector::current();
  PlacementIntent intent = PlacementIntent::Default;
  std::optional<unsigned> objective_rank;
  std::optional<unsigned> exact_os_numa_id;
  FallbackPolicy fallback = FallbackPolicy::ToDefault;
  std::size_t alignment = 64;
};

struct TargetRuntimeStats {
  MemoryTargetId target = 0;
  unsigned os_numa_id = 0;
  std::uint64_t managed_bytes = 0;
  std::uint64_t peak_managed_bytes = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t failed_allocation_count = 0;
  std::uint64_t fallback_count = 0;
  std::optional<std::uint64_t> os_free_bytes;
  std::optional<std::uint64_t> budget_remaining_bytes;
};

std::string to_string(MemoryKind value);
std::string to_string(RankingObjective value);
std::string to_string(PlacementIntent value);

} // namespace vecops::memory

#endif // VECOPS_MEMORY_TYPES_H
