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

inline std::shared_ptr<WorkspaceArenaProvider> default_workspace_arena_provider() {
  static auto provider = std::make_shared<HeapWorkspaceArenaProvider>();
  return provider;
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_ARENA_H
