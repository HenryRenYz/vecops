// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_MEMORY_WORKSPACE_SESSION_H
#define VECOPS_EXECUTION_MEMORY_WORKSPACE_SESSION_H

/**
 * @file vecops/execution/MemoryWorkspaceSession.h
 * @brief Explicit, bounded fast/slow arenas backed by vecops::memory.
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "vecops/execution/WorkspaceArena.h"
#include "vecops/memory/MemorySystem.h"

namespace vecops::execution {

struct MemoryWorkspaceSessionConfig {
  memory::CpuDomainSelector domain = memory::CpuDomainSelector::current();
  nint_t fast_capacity = 0;
  nint_t slow_capacity = 0;
  nint_t arena_alignment = 2 * 1024 * 1024;
  bool allow_fast_fallback = true;
  bool use_large_pages = true;
};

/**
 * One explicitly bounded workspace lifetime.
 *
 * The session freezes its CPU domain and owns exactly one fast and one slow
 * arena. Every per-artifact replay plan receives views of those same arenas,
 * so capacities are session totals rather than per-cache-entry allowances.
 * Calls using one session must be synchronous and non-overlapping. `close()`
 * releases both physical allocations even if generated DSOs still retain
 * obsolete plan metadata; the closed identity must never be used again.
 */
class MemoryWorkspaceSession final : public WorkspaceArenaProvider {
public:
  MemoryWorkspaceSession(memory::MemorySystem memory, MemoryWorkspaceSessionConfig config)
    : memory_(std::move(memory))
    , config_(config) {
    if (!memory_.valid())
      throw memory::MemoryError(memory::MemoryErrc::ConfigurationInvalid, "workspace session has no MemorySystem");
    if (config_.fast_capacity < 0 || config_.slow_capacity < 0)
      throw memory::MemoryError(memory::MemoryErrc::InvalidRequest,
                                "workspace session capacities must be non-negative");
    if (config_.arena_alignment <= 0 || (config_.arena_alignment & (config_.arena_alignment - 1)) != 0)
      throw memory::MemoryError(memory::MemoryErrc::InvalidRequest,
                                "workspace session alignment must be a positive power of two");
    domain_ = config_.domain.kind == memory::CpuDomainSelector::Kind::Specific ? config_.domain.id
                                                                               : memory_.current_cpu_domain();
    open();
  }

  MemoryWorkspaceSession(const MemoryWorkspaceSession&) = delete;
  MemoryWorkspaceSession& operator=(const MemoryWorkspaceSession&) = delete;

  ~MemoryWorkspaceSession() override {
    close();
  }

  [[nodiscard]] nint_t capacity(WorkspaceTier tier) const noexcept override {
    if (closed_.load(std::memory_order_acquire))
      return 0;
    return tier == WorkspaceTier::Fast ? fast_capacity_ : slow_capacity_;
  }

  [[nodiscard]] bool shares_arenas_between_plans() const noexcept override {
    return true;
  }

  WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t alignment) override {
    std::lock_guard lock(mutex_);
    if (closed_.load(std::memory_order_relaxed))
      throw WorkspaceArenaUnavailable("workspace session is closed");
    if (bytes < 0 || alignment <= 0 || (alignment & (alignment - 1)) != 0)
      throw WorkspaceArenaUnavailable("workspace session received an invalid arena request");
    const auto available = tier == WorkspaceTier::Fast ? fast_capacity_ : slow_capacity_;
    if (bytes > available)
      throw WorkspaceArenaUnavailable("workspace session arena capacity was exceeded");
    if (bytes == 0)
      return {};
    if (alignment > config_.arena_alignment)
      throw WorkspaceArenaUnavailable("workspace session request exceeds the arena alignment");
    auto& allocation = tier == WorkspaceTier::Fast ? fast_ : slow_;
    void* data = allocation.data();
    if (data == nullptr || reinterpret_cast<std::uintptr_t>(data) % static_cast<std::uintptr_t>(alignment) != 0)
      throw WorkspaceArenaUnavailable("workspace session arena has insufficient alignment");
    // WorkspaceReplayCache itself retains the provider. The token keeps the
    // generic arena ownership contract without extending physical lifetime
    // beyond an explicit session close.
    std::shared_ptr<void> owner(data, [](void*) {});
    return {data, available, std::move(owner)};
  }

  void close() noexcept {
    std::lock_guard lock(mutex_);
    if (closed_.exchange(true, std::memory_order_acq_rel))
      return;
    slow_.reset();
    fast_.reset();
  }

  [[nodiscard]] bool closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
  }

  [[nodiscard]] memory::CpuDomainId domain() const noexcept {
    return domain_;
  }

  [[nodiscard]] nint_t fast_capacity() const noexcept {
    return fast_capacity_;
  }

  [[nodiscard]] nint_t slow_capacity() const noexcept {
    return slow_capacity_;
  }

private:
  static bool recoverable_fast_failure(memory::MemoryErrc code) noexcept {
    return code == memory::MemoryErrc::BudgetExceeded || code == memory::MemoryErrc::OutOfMemory ||
           code == memory::MemoryErrc::BindingFailed || code == memory::MemoryErrc::UnsupportedBinding;
  }

  memory::Allocation allocate_physical(nint_t bytes, memory::PlacementIntent intent) {
    if (bytes == 0)
      return {};
    return memory_.allocate({
      .bytes = static_cast<std::size_t>(bytes),
      .domain = memory::CpuDomainSelector::specific(domain_),
      .intent = intent,
      .fallback = memory::FallbackPolicy::None,
      .alignment = static_cast<std::size_t>(config_.arena_alignment),
      .use_large_pages = config_.use_large_pages,
    });
  }

  void open() {
    if (config_.fast_capacity != 0) {
      try {
        fast_ = allocate_physical(config_.fast_capacity, memory::PlacementIntent::HighBandwidth);
        fast_capacity_ = config_.fast_capacity;
      } catch (const memory::MemoryError& error) {
        if (!config_.allow_fast_fallback || !recoverable_fast_failure(error.code()))
          throw;
      }
    }
    slow_ = allocate_physical(config_.slow_capacity, memory::PlacementIntent::Default);
    slow_capacity_ = config_.slow_capacity;
    closed_.store(false, std::memory_order_release);
  }

  memory::MemorySystem memory_;
  MemoryWorkspaceSessionConfig config_;
  memory::CpuDomainId domain_ = 0;
  nint_t fast_capacity_ = 0;
  nint_t slow_capacity_ = 0;
  memory::Allocation fast_;
  memory::Allocation slow_;
  mutable std::mutex mutex_;
  std::atomic<bool> closed_{true};
};

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_MEMORY_WORKSPACE_SESSION_H
