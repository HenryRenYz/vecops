//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MEMORY_WORKSPACE_ARENA_H
#define VECOPS_MEMORY_WORKSPACE_ARENA_H

/**
 * @file vecops/memory/WorkspaceArena.h
 * @brief Physical-memory provider for vecops logical workspace plans.
 */

#include <limits>
#include <memory>
#include <utility>

#include "vecops/execution/WorkspaceArena.h"
#include "vecops/memory/MemorySystem.h"

namespace vecops::memory {

/**
 * Map Fast workspace to HighBandwidth memory and Slow workspace to Default.
 *
 * `fast_capacity` is a placement budget, not a reservation. Each replay-cache
 * entry allocates exactly the arena bytes selected by the logical placer. If
 * the strict HighBandwidth allocation fails, WorkspaceReplayCache may rebuild
 * a plan with zero fast capacity when no allocation is FastRequired.
 */
class MemoryWorkspaceArenaProvider final : public execution::WorkspaceArenaProvider {
public:
  MemoryWorkspaceArenaProvider(MemorySystem memory, CpuDomainSelector domain, nint_t fast_capacity,
                               nint_t slow_capacity = std::numeric_limits<nint_t>::max())
    : memory_(std::move(memory))
    , domain_(domain)
    , fast_capacity_(fast_capacity)
    , slow_capacity_(slow_capacity) {
    if (!memory_.valid())
      throw MemoryError(MemoryErrc::ConfigurationInvalid, "workspace arena provider has no MemorySystem");
    if (fast_capacity_ < 0 || slow_capacity_ < 0)
      throw MemoryError(MemoryErrc::InvalidRequest, "workspace arena capacity must be non-negative");
  }

  [[nodiscard]] nint_t capacity(execution::WorkspaceTier tier) const noexcept override {
    return tier == execution::WorkspaceTier::Fast ? fast_capacity_ : slow_capacity_;
  }

  execution::WorkspaceArena allocate(execution::WorkspaceTier tier, nint_t bytes, nint_t alignment) override {
    if (bytes == 0)
      return {};
    AllocationRequest request{
      .bytes = static_cast<std::size_t>(bytes),
      .domain = domain_,
      .intent = tier == execution::WorkspaceTier::Fast ? PlacementIntent::HighBandwidth : PlacementIntent::Default,
      .fallback = FallbackPolicy::None,
      .alignment = static_cast<std::size_t>(alignment),
    };
    std::shared_ptr<Allocation> allocation;
    try {
      allocation = std::make_shared<Allocation>(memory_.allocate(request));
    } catch (const MemoryError& error) {
      throw execution::WorkspaceArenaUnavailable(error.what());
    }
    void* data = allocation->data();
    std::shared_ptr<void> owner(allocation, data);
    return {data, bytes, std::move(owner)};
  }

private:
  MemorySystem memory_;
  CpuDomainSelector domain_;
  nint_t fast_capacity_;
  nint_t slow_capacity_;
};

} // namespace vecops::memory

#endif // VECOPS_MEMORY_WORKSPACE_ARENA_H
