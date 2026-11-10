// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_WORKSPACE_ARENA_H
#define VECOPS_EXECUTION_WORKSPACE_ARENA_H

/**
 * @file vecops/execution/WorkspaceArena.h
 * @brief Type-erased ownership for placed fast and slow workspace arenas.
 *
 * The provider boundary keeps logical workspace planning independent of any
 * NUMA, HBM, framework, or allocator implementation. The default provider obtains page-backed storage through MemoryRuntime
 * capacity admission. Caller-supplied ABI providers remain explicit authorities.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/BufferPlan.h"
#include "vecops/runtime/CallAbi.h"

namespace vecops::execution {

/** Recoverable physical-arena failure eligible for preferred-tier fallback. */
class WorkspaceArenaUnavailable : public std::runtime_error {
public:
  explicit WorkspaceArenaUnavailable(std::string message)
    : std::runtime_error(std::move(message)) {
  }
};

struct WorkspaceArena {
  void* data = nullptr;
  nint_t capacity = 0;
  std::shared_ptr<void> owner;
};

class WorkspaceArenaProvider {
public:
  virtual ~WorkspaceArenaProvider() = default;

  [[nodiscard]] virtual nint_t capacity(BufferTier tier) const noexcept = 0;
  virtual WorkspaceArena allocate(BufferTier tier, nint_t bytes, nint_t alignment) = 0;

  /** Plan all logical buffers together. Called only while preparing bindings.
   * Providers may account for other live owners and serialized reuse groups.
   */
  virtual BufferPlacement place(const LogicalBufferPlan& logical) {
    return place_buffers(logical, capacity(BufferTier::Fast), capacity(BufferTier::Slow));
  }

  /** Whether every plan receives a view of the same session-owned arenas. */
  [[nodiscard]] virtual bool shares_arenas_between_plans() const noexcept {
    return false;
  }
};

/** Synchronous calling-thread provider scope, including generated DSO entry. */
class WorkspaceProviderScope {
public:
  explicit WorkspaceProviderScope(const VecopsWorkspaceArenaProvider*);
  ~WorkspaceProviderScope();
  WorkspaceProviderScope(const WorkspaceProviderScope&)=delete;
  WorkspaceProviderScope& operator=(const WorkspaceProviderScope&)=delete;
private:
  bool active_=false;
  std::shared_ptr<WorkspaceArenaProvider> previous_;
};
/** Reject undeclared payload allocations during a recorded grid action. */
class PlannedWorkspaceScope {
public:
  PlannedWorkspaceScope();
  ~PlannedWorkspaceScope();
  PlannedWorkspaceScope(const PlannedWorkspaceScope&)=delete;
  PlannedWorkspaceScope& operator=(const PlannedWorkspaceScope&)=delete;
private:
  bool previous_;
};

/** Export a provider to generated DSOs. The returned ABI owner and every
 * retained consumer keep the allocation authority alive independently.
 */
std::shared_ptr<const VecopsWorkspaceArenaProvider> workspace_arena_provider_to_abi(
  std::shared_ptr<WorkspaceArenaProvider> provider);

/** Adapt the versioned process-local C ABI to WorkspaceArenaProvider. */
class AbiWorkspaceArenaProvider final : public WorkspaceArenaProvider {
public:
  explicit AbiWorkspaceArenaProvider(const VecopsWorkspaceArenaProvider& provider)
    : provider_{} {
    std::memcpy(&provider_, &provider, std::min<std::size_t>(provider.struct_size, sizeof(provider_)));
    if (provider_.struct_size < offsetof(VecopsWorkspaceArenaProvider, place_buffers) || provider_.identity == 0 ||
        provider_.capacity == nullptr || provider_.allocate == nullptr || provider_.release == nullptr ||
        provider_.retain == nullptr || provider_.release_context == nullptr) {
      throw WorkspaceArenaUnavailable("workspace arena provider ABI is incomplete");
    }
    provider_.retain(provider_.context);
  }

  AbiWorkspaceArenaProvider(const AbiWorkspaceArenaProvider&) = delete;
  AbiWorkspaceArenaProvider& operator=(const AbiWorkspaceArenaProvider&) = delete;

  ~AbiWorkspaceArenaProvider() override {
    provider_.release_context(provider_.context);
  }

  [[nodiscard]] nint_t capacity(BufferTier tier) const noexcept override {
    const auto value = provider_.capacity(provider_.context, abi_tier(tier));
    return value > static_cast<std::uint64_t>(std::numeric_limits<nint_t>::max()) ? std::numeric_limits<nint_t>::max()
                                                                                  : static_cast<nint_t>(value);
  }

  WorkspaceArena allocate(BufferTier tier, nint_t bytes, nint_t alignment) override {
    VecopsWorkspaceArena arena{sizeof(VecopsWorkspaceArena), 0, nullptr, 0, nullptr};
    std::array<char, 512> message{};
    VecopsError error{sizeof(VecopsError), VECOPS_STATUS_OK, message.data(), message.size(), 0};
    const auto status = provider_.allocate(provider_.context, abi_tier(tier), static_cast<std::uint64_t>(bytes),
                                           static_cast<std::uint64_t>(alignment), &arena, &error);
    if (status != VECOPS_STATUS_OK) {
      if (arena.owner != nullptr)
        provider_.release(provider_.context, arena.owner);
      throw WorkspaceArenaUnavailable(message.front() == '\0' ? "workspace arena provider allocation failed"
                                                              : std::string(message.data()));
    }
    if (arena.struct_size < sizeof(VecopsWorkspaceArena) || arena.capacity < static_cast<std::uint64_t>(bytes) ||
        (bytes != 0 && (arena.data == nullptr || arena.owner == nullptr))) {
      if (arena.owner != nullptr)
        provider_.release(provider_.context, arena.owner);
      throw WorkspaceArenaUnavailable("workspace arena provider returned an invalid arena");
    }
    std::shared_ptr<void> owner;
    if (arena.owner != nullptr) {
      const auto provider = provider_;
      provider.retain(provider.context);
      owner = std::shared_ptr<void>(arena.owner, [provider](void* value) {
        provider.release(provider.context, value);
        provider.release_context(provider.context);
      });
    }
    return {arena.data, static_cast<nint_t>(arena.capacity), std::move(owner)};
  }

  BufferPlacement place(const LogicalBufferPlan& logical) override {
    if (provider_.struct_size <
          offsetof(VecopsWorkspaceArenaProvider, place_buffers) + sizeof(provider_.place_buffers) ||
        provider_.place_buffers == nullptr)
      return WorkspaceArenaProvider::place(logical);
    std::vector<VecopsBufferRequirementV1> requests;
    std::vector<std::vector<VecopsBufferLifetimeV1>> lifetimes;
    lifetimes.reserve(logical.allocations.size());
    requests.reserve(logical.allocations.size());
    for (const auto& allocation : logical.allocations) {
      auto& times = lifetimes.emplace_back();
      for (const auto& t : allocation.lifetimes)
        times.push_back({t.begin, t.end});
      const auto& r = allocation.request;
      requests.push_back({allocation.canonical_path.c_str(), static_cast<std::uint64_t>(r.bytes),
                          static_cast<std::uint64_t>(r.alignment), static_cast<std::uint64_t>(r.replicas),
                          static_cast<std::uint32_t>(r.domain), static_cast<std::uint32_t>(r.placement),
                          r.estimated_traffic_bytes, times.data(), times.size()});
    }
    std::vector<VecopsBufferPlacementV1> placements(requests.size());
    std::array<char, 512> message{};
    VecopsError error{sizeof(error), 0, message.data(), message.size(), 0};
    const auto status =
      provider_.place_buffers(provider_.context, requests.data(), requests.size(), placements.data(), &error);
    if (status != VECOPS_STATUS_OK)
      throw WorkspaceArenaUnavailable(message.data());
    BufferPlacement result{logical.fingerprint, logical.axes, {}, 0, 0};
    for (std::size_t i = 0; i < placements.size(); ++i) {
      const auto& p = placements[i];
      const auto& a = logical.allocations[i];
      const auto candidate = buffer_plan_details::candidate(logical, i);
      if (p.tier != VECOPS_WORKSPACE_TIER_FAST && p.tier != VECOPS_WORKSPACE_TIER_SLOW)
        throw WorkspaceArenaUnavailable("buffer provider returned an invalid tier");
      const auto tier = p.tier == VECOPS_WORKSPACE_TIER_FAST ? BufferTier::Fast : BufferTier::Slow;
      if (p.offset > static_cast<std::uint64_t>(capacity(tier)) ||
          static_cast<std::uint64_t>(candidate.bytes) > static_cast<std::uint64_t>(capacity(tier)) - p.offset ||
          p.offset % candidate.alignment ||
          (a.request.placement == BufferPlacementPolicy::FastRequired && tier != BufferTier::Fast) ||
          (a.request.placement == BufferPlacementPolicy::SlowAllowed && tier != BufferTier::Slow))
        throw WorkspaceArenaUnavailable("buffer provider violated the allocation contract");
      result.entries.push_back({a.site, tier, static_cast<nint_t>(p.offset), a.request.bytes, candidate.stride,
                                a.request.replicas, candidate.alignment, a.request.domain, a.request.alignment,
                                a.request.placement});
      auto& size = tier == BufferTier::Fast ? result.fast_bytes : result.slow_bytes;
      size = std::max(size, buffer_plan_details::checked_add(p.offset, candidate.bytes));
    }
    return result;
  }

  [[nodiscard]] std::uint64_t identity() const noexcept {
    return provider_.identity;
  }

  [[nodiscard]] bool shares_arenas_between_plans() const noexcept override {
    return (provider_.flags & VECOPS_WORKSPACE_ARENA_PROVIDER_FLAG_SHARED_ARENAS) != 0;
  }

private:
  static constexpr std::uint32_t abi_tier(BufferTier tier) noexcept {
    return tier == BufferTier::Fast ? VECOPS_WORKSPACE_TIER_FAST : VECOPS_WORKSPACE_TIER_SLOW;
  }

  VecopsWorkspaceArenaProvider provider_;
};

inline std::shared_ptr<WorkspaceArenaProvider>
workspace_arena_provider_from_abi(const VecopsWorkspaceArenaProvider& provider) {
  return std::make_shared<AbiWorkspaceArenaProvider>(provider);
}

/** Resource-backed default; its lifetime owns every admitted allocation. */
std::shared_ptr<WorkspaceArenaProvider> default_workspace_arena_provider();

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_ARENA_H
