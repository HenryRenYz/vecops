//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_WORKSPACE_ARENA_H
#define VECOPS_EXECUTION_WORKSPACE_ARENA_H

/**
 * @file vecops/execution/WorkspaceArena.h
 * @brief Type-erased ownership for placed fast and slow workspace arenas.
 *
 * The provider boundary keeps logical workspace planning independent of any
 * NUMA, HBM, framework, or allocator implementation. The default provider is
 * ordinary aligned heap storage and preserves the historical behavior.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/WorkspacePlan.h"
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

  [[nodiscard]] virtual nint_t capacity(WorkspaceTier tier) const noexcept = 0;
  virtual WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t alignment) = 0;

  /** Whether every plan receives a view of the same session-owned arenas. */
  [[nodiscard]] virtual bool shares_arenas_between_plans() const noexcept {
    return false;
  }
};

/** Heap-backed provider used when no physical placement authority is given. */
class HeapWorkspaceArenaProvider final : public WorkspaceArenaProvider {
public:
  [[nodiscard]] nint_t capacity(WorkspaceTier) const noexcept override {
    return std::numeric_limits<nint_t>::max();
  }

  WorkspaceArena allocate(WorkspaceTier, nint_t bytes, nint_t alignment) override {
    VECOPS_ASSERT(bytes >= 0, "workspace arena bytes must be non-negative");
    VECOPS_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0,
                  "workspace arena alignment must be a positive power of two");
    if (bytes == 0)
      return {};
    VECOPS_ASSERT(bytes <= std::numeric_limits<nint_t>::max() - alignment, "workspace arena allocation size overflow");
    const auto storage_bytes = static_cast<std::size_t>(bytes + alignment - 1);
    auto* storage = new std::byte[storage_bytes];
    std::shared_ptr<void> owner(storage, [](void* pointer) { delete[] static_cast<std::byte*>(pointer); });
    const auto raw = reinterpret_cast<std::uintptr_t>(storage);
    const auto aligned =
      (raw + static_cast<std::uintptr_t>(alignment - 1)) & ~static_cast<std::uintptr_t>(alignment - 1);
    return {reinterpret_cast<void*>(aligned), bytes, std::move(owner)};
  }
};

/** Adapt the versioned process-local C ABI to WorkspaceArenaProvider. */
class AbiWorkspaceArenaProvider final : public WorkspaceArenaProvider {
public:
  explicit AbiWorkspaceArenaProvider(const VecopsWorkspaceArenaProvider& provider)
    : provider_(provider) {
    if (provider_.struct_size < sizeof(VecopsWorkspaceArenaProvider) || provider_.identity == 0 ||
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

  [[nodiscard]] nint_t capacity(WorkspaceTier tier) const noexcept override {
    const auto value = provider_.capacity(provider_.context, abi_tier(tier));
    return value > static_cast<std::uint64_t>(std::numeric_limits<nint_t>::max()) ? std::numeric_limits<nint_t>::max()
                                                                                  : static_cast<nint_t>(value);
  }

  WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t alignment) override {
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
      owner =
        std::shared_ptr<void>(arena.owner, [provider](void* value) { provider.release(provider.context, value); });
    }
    return {arena.data, static_cast<nint_t>(arena.capacity), std::move(owner)};
  }

  [[nodiscard]] std::uint64_t identity() const noexcept {
    return provider_.identity;
  }

  [[nodiscard]] bool shares_arenas_between_plans() const noexcept override {
    return (provider_.flags & VECOPS_WORKSPACE_ARENA_PROVIDER_FLAG_SHARED_ARENAS) != 0;
  }

private:
  static constexpr std::uint32_t abi_tier(WorkspaceTier tier) noexcept {
    return tier == WorkspaceTier::Fast ? VECOPS_WORKSPACE_TIER_FAST : VECOPS_WORKSPACE_TIER_SLOW;
  }

  VecopsWorkspaceArenaProvider provider_;
};

inline std::shared_ptr<WorkspaceArenaProvider>
workspace_arena_provider_from_abi(const VecopsWorkspaceArenaProvider& provider) {
  return std::make_shared<AbiWorkspaceArenaProvider>(provider);
}

inline std::shared_ptr<WorkspaceArenaProvider> default_workspace_arena_provider() {
  static auto provider = std::make_shared<HeapWorkspaceArenaProvider>();
  return provider;
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_ARENA_H
