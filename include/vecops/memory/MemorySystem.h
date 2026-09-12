//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MEMORY_MEMORY_SYSTEM_H
#define VECOPS_MEMORY_MEMORY_SYSTEM_H

/**
 * @file vecops/memory/MemorySystem.h
 * @brief Direct OS allocations selected from a heterogeneous CPU topology.
 *
 * MemorySystem owns an immutable topology snapshot and process-local budget
 * counters. Allocation is move-only and releases directly to the backend; V1
 * has no arena, cache, reserve, compaction, or garbage collection layer.
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vecops/memory/Types.h"

namespace vecops::memory {

namespace details {
struct MemoryState;
struct AllocationHeader;
struct AllocationAccess;
} // namespace details

class MemoryError : public std::runtime_error {
public:
  MemoryError(MemoryErrc code, std::string message);

  [[nodiscard]] MemoryErrc code() const noexcept {
    return code_;
  }

private:
  MemoryErrc code_;
};

/** Move-only ownership of one direct backend allocation. */
class Allocation {
public:
  Allocation() noexcept = default;
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&& other) noexcept;
  Allocation& operator=(Allocation&& other) noexcept;
  ~Allocation();

  [[nodiscard]] void* data() const noexcept {
    return data_;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] std::optional<MemoryTargetId> target() const noexcept {
    return target_;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return data_ != nullptr;
  }

  void reset() noexcept;

private:
  friend class MemorySystem;
  friend struct details::AllocationAccess;
  Allocation(void* data, std::size_t size, MemoryTargetId target) noexcept;

  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::optional<MemoryTargetId> target_;
};

/** Discover topology, resolve placement, and perform direct allocations. */
class MemorySystem {
public:
  static MemorySystem discover(const MemoryConfig& config = {});

  MemorySystem() noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return state_ != nullptr;
  }

  Allocation allocate(const AllocationRequest& request) const;

  [[nodiscard]] const TopologySnapshot& topology() const;
  [[nodiscard]] CpuDomainId current_cpu_domain() const;
  [[nodiscard]] MemoryTierView tiers(CpuDomainId domain, RankingObjective objective) const;
  [[nodiscard]] std::vector<TargetRuntimeStats> stats() const;
  [[nodiscard]] std::string describe() const;

private:
  explicit MemorySystem(std::shared_ptr<details::MemoryState> state) noexcept;

  std::shared_ptr<details::MemoryState> state_;
};

} // namespace vecops::memory

#endif // VECOPS_MEMORY_MEMORY_SYSTEM_H
