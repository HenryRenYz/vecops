// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_EXECUTION_BUFFER_MEMORY_SESSION_H
#define VECOPS_EXECUTION_BUFFER_MEMORY_SESSION_H
/** @file BufferMemorySession.h
 * One physical capacity authority for tensors, graph buffers and operator
 * temporaries. All use LogicalBufferPlan, the common buffer/alias-group IR.
 * Named reuse groups are an explicit nonconcurrency contract. They share
 * offsets across plans, never with other groups or storage-owned leases.
 * Planning and binding are cold operations; replay has no allocator callback.
 */
#include <map>
#include <mutex>
#include <atomic>
#include "vecops/execution/WorkspaceArena.h"
#include "vecops/memory/MemorySystem.h"

namespace vecops::execution {
struct BufferMemorySessionConfig {
  memory::CpuDomainSelector domain = memory::CpuDomainSelector::current();
  nint_t fast_capacity = 0;
  nint_t slow_capacity = 0;
  bool allow_fast_fallback = false;
  bool use_large_pages = true;
};

class BufferMemorySession final
  : public WorkspaceArenaProvider
  , public std::enable_shared_from_this<BufferMemorySession> {
  struct Physical {
    std::shared_ptr<memory::Allocation> fast, slow;
  };

public:
  struct Binding {
    BufferPlacement placement;
    WorkspaceArena fast, slow;
    std::shared_ptr<void> lease;
  };
  struct Stats {
    nint_t fast_capacity, slow_capacity, fast_reserved, slow_reserved;
    std::size_t groups, plans;
    nint_t largest_fast_free;
  };
  struct ManifestEntry {
    std::size_t plan;
    std::string group, name;
    BufferPlacementEntry placement;
    double traffic_bytes;
  };
  std::vector<ManifestEntry> manifest() const {
    std::lock_guard lock(mutex_);
    return manifest_;
  }
  std::pair<std::optional<memory::MemoryTargetId>, nint_t> metadata(BufferTier tier, nint_t offset,
                                                                    nint_t bytes) const {
    std::lock_guard lock(mutex_);
    if (!physical_)
      throw WorkspaceArenaUnavailable("buffer session is closed");
    const auto& a = tier == BufferTier::Fast ? physical_->fast : physical_->slow;
    if (!a) return {std::nullopt, 0};
    return {a->target(), std::max<nint_t>(0, std::min<nint_t>(offset + bytes, a->large_page_bytes()) - offset)};
  }
  std::uintptr_t arena_address(BufferTier tier) const {
    std::lock_guard lock(mutex_);
    if (!physical_)
      throw WorkspaceArenaUnavailable("buffer session is closed");
    const auto& owner = tier == BufferTier::Fast ? physical_->fast : physical_->slow;
    return owner ? reinterpret_cast<std::uintptr_t>(owner->data()) : 0;
  }
  BufferMemorySession(memory::MemorySystem system, BufferMemorySessionConfig config)
    : system_(std::move(system))
    , config_(config)
    , physical_(std::make_shared<Physical>()) {
    if (config.fast_capacity < 0 || config.slow_capacity < 0)
      throw std::invalid_argument("buffer capacities must be nonnegative");
    domain_ =
      config.domain.kind == memory::CpuDomainSelector::Kind::Specific ? config.domain.id : system_.current_cpu_domain();
    auto alloc = [&](nint_t bytes, memory::PlacementIntent intent, bool large) {
      return system_.allocate({.bytes = static_cast<std::size_t>(bytes),
                               .domain = memory::CpuDomainSelector::specific(domain_),
                               .intent = intent,
                               .fallback = memory::FallbackPolicy::None,
                               .alignment = 2 * 1024 * 1024,
                               .use_large_pages = large});
    };
    if (config_.fast_capacity) {
      try {
        physical_->fast = std::make_shared<memory::Allocation>(alloc(config_.fast_capacity, memory::PlacementIntent::HighBandwidth, config.use_large_pages));
      } catch (const memory::MemoryError& e) {
        const auto c = e.code();
        if (!config.allow_fast_fallback ||
            (c != memory::MemoryErrc::BudgetExceeded && c != memory::MemoryErrc::OutOfMemory &&
             c != memory::MemoryErrc::BindingFailed && c != memory::MemoryErrc::UnsupportedBinding))
          throw;
        config_.fast_capacity = 0;
      }
    }
    // Default memory is demand paged; do not reserve a huge-page DDR pool.
    if (config_.slow_capacity)
      physical_->slow = std::make_shared<memory::Allocation>(alloc(config_.slow_capacity, memory::PlacementIntent::Default, false));
  }
  nint_t capacity(BufferTier tier) const noexcept override {
    if (closed_.load())
      return 0;
    return tier == BufferTier::Fast ? config_.fast_capacity : config_.slow_capacity;
  }
  WorkspaceArena allocate(BufferTier tier, nint_t bytes, nint_t alignment) override {
    std::lock_guard lock(mutex_);
    return arena(tier, bytes, alignment);
  }
  BufferPlacement place(const LogicalBufferPlan& logical) override {
    std::lock_guard lock(mutex_);
    // Source-kernel instances in one session are explicitly serialized by
    // the embedding; their workspace never aliases live graph intermediates.
    return place_locked(logical, "serial:native");
  }
  Binding bind(const LogicalBufferPlan& logical, std::string reuse_group = {}) {
    auto self = shared_from_this();
    std::unique_lock lock(mutex_);
    if (reuse_group == "native")
      throw std::invalid_argument("native is a reserved reuse group");
    const auto id = ++next_lease_;
    auto group = reuse_group.empty() ? "lease:" + std::to_string(id) : "serial:" + reuse_group;
    auto placement = place_locked(logical, group);
    Binding result{std::move(placement), {}, {}, {}};
    result.fast = arena(BufferTier::Fast, result.placement.fast_bytes, 64);
    result.slow = arena(BufferTier::Slow, result.placement.slow_bytes, 64);
    if (result.fast.data)
      result.fast.owner = physical_->fast;
    if (result.slow.data)
      result.slow.owner = physical_->slow;
    auto& ranges = lease_ranges_[group][id];
    for (const auto& e : result.placement.entries)
      if (e.bytes)
        ranges.push_back({e.tier, e.offset, buffer_plan_details::checked_mul(e.replica_stride, e.replicas)});
    lock.unlock();
    result.lease = std::shared_ptr<void>(self.get(), [self, group, id](void*) {
      std::lock_guard guard(self->mutex_);
      auto it = self->lease_ranges_.find(group);
      if (it == self->lease_ranges_.end())
        return;
      it->second.erase(id);
      if (it->second.empty()) {
        self->lease_ranges_.erase(it);
        self->groups_.erase(group);
      } else {
        // Preserve the conservative reservation if memory pressure prevents
        // bookkeeping compaction in a storage destructor.
        try {
          std::vector<BufferReservation> live;
          for (const auto& [_, ranges] : it->second)
            live.insert(live.end(), ranges.begin(), ranges.end());
          self->groups_[group] = merge(std::move(live));
        } catch (const std::bad_alloc&) {
        }
      }
    });
    return result;
  }
  Stats stats() const {
    std::lock_guard lock(mutex_);
    nint_t fast = 0, slow = 0;
    for (const auto& [_, ranges] : groups_)
      for (const auto& r : ranges)
        (r.tier == BufferTier::Fast ? fast : slow) += r.bytes;
    std::vector<std::pair<nint_t, nint_t>> ranges;
    for (const auto& [_, rs] : groups_)
      for (const auto& r : rs)
        if (r.tier == BufferTier::Fast)
          ranges.emplace_back(r.offset, r.offset + r.bytes);
    std::sort(ranges.begin(), ranges.end());
    nint_t cursor = 0, largest = 0;
    for (const auto& [begin, end] : ranges) {
      largest = std::max(largest, begin - cursor);
      cursor = std::max(cursor, end);
    }
    largest = std::max(largest, config_.fast_capacity - cursor);
    return {config_.fast_capacity, config_.slow_capacity, fast, slow, groups_.size(), plans_, largest};
  }
  void close() {
    std::lock_guard lock(mutex_);
    closed_.store(true);
    groups_.clear();
    lease_ranges_.clear();
    physical_.reset(); // Existing owners retain mappings, including escaped Tensor aliases.
  }
  bool closed() const noexcept {
    return closed_.load();
  }
  memory::CpuDomainId domain() const noexcept {
    return domain_;
  }

private:
  WorkspaceArena arena(BufferTier tier, nint_t bytes, nint_t alignment) {
    if (closed_.load())
      throw WorkspaceArenaUnavailable("buffer session is closed");
    if (bytes < 0 || bytes > capacity(tier) || alignment < 1 || (alignment & (alignment - 1)) ||
        alignment > 2 * 1024 * 1024)
      throw WorkspaceArenaUnavailable("invalid buffer arena request");
    if (!bytes)
      return {};
    auto& allocation = tier == BufferTier::Fast ? physical_->fast : physical_->slow;
    // Native cached bindings are invalid after close; they must not pin the
    // entire model pool through process-global operator caches. bind() gives
    // storage-owned Tensor leases real physical ownership separately.
    return {allocation->data(), capacity(tier), std::shared_ptr<void>(allocation->data(), [](void*) {})};
  }
  BufferPlacement place_locked(const LogicalBufferPlan& logical, const std::string& group) {
    if (closed_.load())
      throw WorkspaceArenaUnavailable("buffer session is closed");
    for (const auto& a : logical.allocations)
      if (a.request.alignment > 2 * 1024 * 1024)
        throw WorkspaceArenaUnavailable("buffer alignment exceeds session backing alignment");
    std::vector<BufferReservation> occupied;
    for (const auto& [key, ranges] : groups_)
      if (key != group)
        occupied.insert(occupied.end(), ranges.begin(), ranges.end());
    auto result = place_buffers(logical, capacity(BufferTier::Fast), capacity(BufferTier::Slow), occupied);
    auto ranges = groups_[group];
    for (const auto& e : result.entries)
      if (e.bytes)
        ranges.push_back({e.tier, e.offset, buffer_plan_details::checked_mul(e.replica_stride, e.replicas)});
    groups_[group] = merge(std::move(ranges));
    for (const auto& e : result.entries) {
      const auto it = std::find_if(logical.allocations.begin(), logical.allocations.end(),
                                   [&](const auto& a) { return a.site == e.site; });
      manifest_.push_back({plans_, group, it->canonical_path, e, it->request.estimated_traffic_bytes});
    }
    ++plans_;
    return result;
  }
  static std::vector<BufferReservation> merge(std::vector<BufferReservation> ranges) {
    std::sort(ranges.begin(), ranges.end(),
              [](const auto& a, const auto& b) { return a.tier != b.tier ? a.tier < b.tier : a.offset < b.offset; });
    std::vector<BufferReservation> result;
    for (const auto& r : ranges) {
      if (!result.empty() && result.back().tier == r.tier && r.offset <= result.back().offset + result.back().bytes)
        result.back().bytes =
          std::max(result.back().offset + result.back().bytes, r.offset + r.bytes) - result.back().offset;
      else
        result.push_back(r);
    }
    return result;
  }
  memory::MemorySystem system_;
  BufferMemorySessionConfig config_;
  memory::CpuDomainId domain_;
  std::shared_ptr<Physical> physical_;
  mutable std::mutex mutex_;
  std::atomic<bool> closed_{false};
  std::map<std::string, std::vector<BufferReservation>> groups_;
  std::map<std::string, std::map<std::uint64_t, std::vector<BufferReservation>>> lease_ranges_;
  std::vector<ManifestEntry> manifest_;
  std::uint64_t next_lease_ = 0;
  std::size_t plans_ = 0;
};
} // namespace vecops::execution
#endif
