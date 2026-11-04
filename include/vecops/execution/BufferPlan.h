// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_BUFFER_PLAN_H
#define VECOPS_EXECUTION_BUFFER_PLAN_H

/**
 * @file vecops/execution/BufferPlan.h
 * @brief Common pointer-free buffer traces, deterministic placement, and binding.
 *
 * Planning is deliberately separate from storage ownership. A trace records
 * stable semantic allocation sites and structured lifetimes. Placement maps
 * logical buffers to fast or slow memory. Binding adds arena base pointers
 * and produces direct slots for prepared operators. `find()` is a prepare-time
 * API, not an inner-loop allocation mechanism.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/Meta.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"

namespace vecops::execution {

namespace buffer_plan_details {

constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

constexpr std::uint64_t hash(std::string_view text, std::uint64_t seed = fnv_offset) {
  auto value = seed;
  for (const unsigned char byte : text) {
    value ^= byte;
    value *= fnv_prime;
  }
  return value;
}

constexpr std::uint64_t mix(std::uint64_t left, std::uint64_t right) {
  left ^= right + 0x9e3779b97f4a7c15ull + (left << 6) + (left >> 2);
  return left;
}

inline nint_t checked_add(nint_t left, nint_t right) {
  VECOPS_CHECK(left >= 0 && right >= 0, "workspace sizes must be non-negative");
  VECOPS_CHECK(left <= std::numeric_limits<nint_t>::max() - right, "workspace size addition overflow");
  return left + right;
}

inline nint_t checked_mul(nint_t left, nint_t right) {
  VECOPS_CHECK(left >= 0 && right >= 0, "workspace sizes must be non-negative");
  VECOPS_CHECK(left == 0 || right <= std::numeric_limits<nint_t>::max() / left,
                "workspace size multiplication overflow");
  return left * right;
}

inline nint_t checked_align_up(nint_t value, nint_t alignment) {
  VECOPS_CHECK(value >= 0, "workspace size must be non-negative");
  VECOPS_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0,
                "workspace alignment must be a positive power of two");
  return checked_add(value, alignment - 1) & ~(alignment - 1);
}

} // namespace buffer_plan_details

/** Stable, plan-local allocation identity. */
struct BufferSiteId {
  std::uint64_t path_hash = 0;
  std::uint64_t semantic_hash = 0;

  friend constexpr bool operator==(BufferSiteId, BufferSiteId) = default;
};

/** Compile-time/run-time contract for one shape or stride axis. */
struct AxisContract {
  enum class Kind : std::uint8_t { Constant, Dynamic };

  Kind kind = Kind::Dynamic;
  nint_t exact = 0;
  nint_t alignment = 1;
  nint_t lower = meta::kLoInf;
  nint_t upper = meta::kHiInf;
  nint_t recorded = 0;
  nint_t bucket_upper = 0;

  template <meta::ValueInput T>
  static AxisContract from(T&& input, nint_t bucket = -1) {
    auto value = meta::to_value(std::forward<T>(input));
    using V = decltype(value);
    const nint_t actual = static_cast<nint_t>(value);
    if constexpr (V::is_const) {
      return {Kind::Constant, actual, 0, actual, actual, actual, actual};
    } else {
      const nint_t bucket_upper = bucket < 0 ? actual : bucket;
      VECOPS_CHECK(actual <= bucket_upper, "workspace axis bucket is smaller than the recorded value");
      return {Kind::Dynamic, 0, V::alignment, meta::lower_bound_v<V>, meta::upper_bound_v<V>, actual, bucket_upper};
    }
  }

  [[nodiscard]] bool accepts(nint_t actual) const {
    if (kind == Kind::Constant)
      return actual == exact;
    if (actual < lower || actual > upper || actual > bucket_upper)
      return false;
    return alignment > 0 && actual % alignment == 0;
  }
};

/** Decisions that can alter allocation paths despite compatible Meta types. */
struct DecisionFingerprint {
  std::uint64_t recipe = 0;
  std::uint64_t decision = 0;

  friend constexpr bool operator==(DecisionFingerprint, DecisionFingerprint) = default;
};

enum class BufferDomain : std::uint8_t { Global, WorkerLocal };

/** Build the stable ID shared by trace, dynamic execution, and replay. */
inline BufferSiteId buffer_site_id(std::string_view canonical_path, std::string_view local_name,
                                         BufferDomain domain) {
  auto path_hash = buffer_plan_details::hash(canonical_path);
  path_hash = buffer_plan_details::hash("/", path_hash);
  path_hash = buffer_plan_details::hash(local_name, path_hash);
  return {path_hash,
          buffer_plan_details::mix(buffer_plan_details::hash(local_name), static_cast<std::uint64_t>(domain))};
}

/** Build a site ID from an incrementally maintained scope-path hash. */
inline BufferSiteId buffer_site_id(std::uint64_t scope_path_hash, std::string_view local_name,
                                         BufferDomain domain) {
  auto path_hash = buffer_plan_details::hash("/", scope_path_hash);
  path_hash = buffer_plan_details::hash(local_name, path_hash);
  return {path_hash,
          buffer_plan_details::mix(buffer_plan_details::hash(local_name), static_cast<std::uint64_t>(domain))};
}

enum class BufferPlacementPolicy : std::uint8_t {
  FastRequired,
  FastPreferred,
  SlowAllowed,
};

enum class BufferTier : std::uint8_t { Fast, Slow };

struct BufferAllocationRequest {
  nint_t bytes = 0;
  nint_t alignment = vec::DEFAULT_ALIGNMENT;
  BufferDomain domain = BufferDomain::Global;
  nint_t replicas = 1;
  BufferPlacementPolicy placement = BufferPlacementPolicy::FastPreferred;
  /** Estimated bytes transferred through this buffer; zero defaults to bytes. */
  double estimated_traffic_bytes = 0.0;
};

struct BufferLifetime {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct LogicalBufferAllocation {
  BufferSiteId site;
  std::string canonical_path;
  BufferAllocationRequest request;
  std::vector<BufferLifetime> lifetimes;
};

/** Pointer-free trace result; safe to cache across arena rebinding. */
struct LogicalBufferPlan {
  DecisionFingerprint fingerprint;
  std::vector<AxisContract> axes;
  std::vector<LogicalBufferAllocation> allocations;
};

/** Structured trace builder with lexical scratch lifetimes. */
class BufferTrace {
public:
  class Scope {
  public:
    Scope() = default;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    Scope(Scope&& other) noexcept
      : owner_(std::exchange(other.owner_, nullptr))
      , depth_(other.depth_) {
    }

    Scope& operator=(Scope&& other) noexcept {
      if (this != &other) {
        close();
        owner_ = std::exchange(other.owner_, nullptr);
        depth_ = other.depth_;
      }
      return *this;
    }

    ~Scope() {
      close();
    }

    void close() {
      if (owner_ != nullptr) {
        owner_->close_scope(depth_);
        owner_ = nullptr;
      }
    }

  private:
    friend class BufferTrace;
    Scope(BufferTrace* owner, std::size_t depth)
      : owner_(owner)
      , depth_(depth) {
    }

    BufferTrace* owner_ = nullptr;
    std::size_t depth_ = 0;
  };

  explicit BufferTrace(std::string recipe, DecisionFingerprint fingerprint = {})
    : recipe_(std::move(recipe))
    , fingerprint_(fingerprint) {
    VECOPS_CHECK(!recipe_.empty(), "workspace recipe name must not be empty");
    frames_.push_back(Frame{recipe_, ++clock_, {}});
  }

  BufferTrace(const BufferTrace&) = delete;
  BufferTrace& operator=(const BufferTrace&) = delete;

  Scope serial_scope(std::string_view name) {
    VECOPS_CHECK(!finished_, "cannot add a scope to a finished workspace trace");
    VECOPS_CHECK(!name.empty(), "workspace scope name must not be empty");
    std::string path = frames_.back().path;
    path.push_back('/');
    path.append(name);
    frames_.push_back(Frame{std::move(path), ++clock_, {}});
    return Scope{this, frames_.size() - 1};
  }

  void add_axis(AxisContract contract) {
    VECOPS_CHECK(!finished_, "cannot add an axis to a finished workspace trace");
    axes_.push_back(contract);
  }

  BufferSiteId request(std::string_view local_name, BufferAllocationRequest request) {
    VECOPS_CHECK(!finished_, "cannot allocate from a finished workspace trace");
    VECOPS_CHECK(!local_name.empty(), "workspace allocation name must not be empty");
    VECOPS_CHECK(request.bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_CHECK(request.alignment > 0 && (request.alignment & (request.alignment - 1)) == 0,
                  "workspace allocation alignment must be a positive power of two");
    VECOPS_CHECK(request.replicas > 0, "workspace replica count must be positive");
    VECOPS_CHECK(request.estimated_traffic_bytes >= 0.0,
                  "workspace estimated traffic must be non-negative");
    if (request.domain == BufferDomain::Global) {
      VECOPS_CHECK(request.replicas == 1, "global workspace cannot have replicas");
    }

    std::string path = frames_.back().path;
    path.push_back('/');
    path.append(local_name);
    const BufferSiteId site = buffer_site_id(frames_.back().path, local_name, request.domain);

    std::size_t allocation_index = allocations_.size();
    for (std::size_t index = 0; index < allocations_.size(); ++index) {
      if (allocations_[index].site == site) {
        allocation_index = index;
        break;
      }
    }
    if (allocation_index == allocations_.size()) {
      allocations_.push_back(LogicalBufferAllocation{site, path, request, {}});
    } else {
      auto& existing = allocations_[allocation_index];
      VECOPS_CHECK(existing.canonical_path == path, "workspace site hash collision");
      VECOPS_CHECK(compatible(existing.request, request),
                    "workspace site was replayed with a different allocation contract");
      for (const auto& pending : pending_) {
        // A duplicate live site would make two prepared objects alias the
        // same replay slot.  This is a caller contract violation rather than
        // a debug-only internal invariant, so keep the check in release
        // builds as well.
        VECOPS_CHECK(pending.closed || pending.allocation_index != allocation_index,
                     "the same workspace site is live more than once");
      }
      // A Dynamic axis may revisit the same semantic site with a different
      // extent during one trace. Retain the maximum capacity and worker count;
      // replay validates each actual request against these upper bounds.
      existing.request.bytes = std::max(existing.request.bytes, request.bytes);
      existing.request.replicas = std::max(existing.request.replicas, request.replicas);
    }

    const std::size_t pending_index = pending_.size();
    pending_.push_back(Pending{allocation_index, frames_.back().begin, false});
    frames_.back().pending.push_back(pending_index);
    return site;
  }

  LogicalBufferPlan finish() {
    VECOPS_CHECK(!finished_, "workspace trace can only be finished once");
    VECOPS_CHECK(frames_.size() == 1, "workspace trace still has an open nested scope");
    close_scope(0);
    finished_ = true;
    return LogicalBufferPlan{fingerprint_, std::move(axes_), std::move(allocations_)};
  }

private:
  struct Pending {
    std::size_t allocation_index;
    std::uint64_t begin;
    bool closed;
  };

  struct Frame {
    std::string path;
    std::uint64_t begin;
    std::vector<std::size_t> pending;
  };

  static bool compatible(const BufferAllocationRequest& left, const BufferAllocationRequest& right) {
    return left.alignment == right.alignment && left.domain == right.domain && left.placement == right.placement &&
           left.estimated_traffic_bytes == right.estimated_traffic_bytes;
  }

  void close_scope(std::size_t depth) {
    VECOPS_CHECK(depth + 1 == frames_.size(), "workspace scopes must close in LIFO order");
    const auto end = ++clock_;
    for (const auto pending_index : frames_.back().pending) {
      auto& pending = pending_[pending_index];
      VECOPS_CHECK(!pending.closed, "workspace lifetime was closed twice");
      allocations_[pending.allocation_index].lifetimes.push_back(BufferLifetime{pending.begin, end});
      pending.closed = true;
    }
    frames_.pop_back();
  }

  std::string recipe_;
  DecisionFingerprint fingerprint_;
  std::uint64_t clock_ = 0;
  bool finished_ = false;
  std::vector<Frame> frames_;
  std::vector<Pending> pending_;
  std::vector<AxisContract> axes_;
  std::vector<LogicalBufferAllocation> allocations_;
};

struct BufferPlacementEntry {
  BufferSiteId site;
  BufferTier tier = BufferTier::Slow;
  nint_t offset = 0;
  nint_t bytes = 0;
  nint_t replica_stride = 0;
  nint_t replicas = 1;
  nint_t alignment = vec::DEFAULT_ALIGNMENT;
  BufferDomain domain = BufferDomain::Global;
  nint_t request_alignment = vec::DEFAULT_ALIGNMENT;
  BufferPlacementPolicy placement = BufferPlacementPolicy::FastPreferred;
};

/** Already committed ranges belonging to other live buffer owners. */
struct BufferReservation {
  BufferTier tier;
  nint_t offset;
  nint_t bytes;
};

struct BufferPlacement {
  DecisionFingerprint fingerprint;
  std::vector<AxisContract> axes;
  std::vector<BufferPlacementEntry> entries;
  nint_t fast_bytes = 0;
  nint_t slow_bytes = 0;
};

struct BoundBufferSlot {
  std::byte* base = nullptr;
  nint_t bytes = 0;
  nint_t replica_stride = 0;
  nint_t replicas = 1;

  [[nodiscard]] void* replica(nint_t index = 0) const {
    VECOPS_CHECK(0 <= index && index < replicas, "workspace replica is out of range");
    return base == nullptr ? nullptr : base + index * replica_stride;
  }
};

class BoundBufferPlan {
public:
  BoundBufferPlan() = default;

  BoundBufferPlan(const BufferPlacement& placement, void* fast_base, nint_t fast_capacity, void* slow_base,
                     nint_t slow_capacity)
    : placement_(&placement)
    , fast_(static_cast<std::byte*>(fast_base))
    , slow_(static_cast<std::byte*>(slow_base)) {
    bind_tier(BufferTier::Fast, fast_, fast_capacity, placement.fast_bytes);
    bind_tier(BufferTier::Slow, slow_, slow_capacity, placement.slow_bytes);
  }

  [[nodiscard]] BoundBufferSlot find(BufferSiteId site, nint_t actual_bytes = -1) const {
    VECOPS_CHECK(placement_ != nullptr, "workspace plan is not bound");
    for (const auto& entry : placement_->entries) {
      if (entry.site == site) {
        if (actual_bytes >= 0) {
          VECOPS_CHECK(actual_bytes <= entry.bytes, "workspace replay request exceeds its planned capacity");
        }
        auto* tier = entry.tier == BufferTier::Fast ? fast_ : slow_;
        return {tier == nullptr ? nullptr : tier + entry.offset, entry.bytes, entry.replica_stride, entry.replicas};
      }
    }
    VECOPS_CHECK(false, "workspace site is absent from the bound plan");
    return {};
  }

  /** Resolve a replay site while validating its complete allocation contract. */
  [[nodiscard]] BoundBufferSlot find(BufferSiteId site, const BufferAllocationRequest& request) const {
    VECOPS_CHECK(placement_ != nullptr, "workspace plan is not bound");
    for (const auto& entry : placement_->entries) {
      if (entry.site == site) {
        VECOPS_CHECK(request.bytes <= entry.bytes, "workspace replay request exceeds its planned capacity");
        VECOPS_CHECK(request.alignment == entry.request_alignment,
                      "workspace replay alignment differs from the recorded contract");
        VECOPS_CHECK(request.replicas <= entry.replicas,
                      "workspace replay replica count exceeds its planned capacity");
        VECOPS_CHECK(request.domain == entry.domain, "workspace replay domain differs from the recorded contract");
        VECOPS_CHECK(request.placement == entry.placement,
                      "workspace replay placement policy differs from the recorded contract");
        auto* tier = entry.tier == BufferTier::Fast ? fast_ : slow_;
        return {tier == nullptr ? nullptr : tier + entry.offset, entry.bytes, entry.replica_stride, entry.replicas};
      }
    }
    VECOPS_CHECK(false, "workspace site is absent from the bound plan");
    return {};
  }

  [[nodiscard]] bool accepts(DecisionFingerprint fingerprint, const std::vector<nint_t>& actual_axes) const {
    if (placement_ == nullptr || placement_->fingerprint != fingerprint ||
        placement_->axes.size() != actual_axes.size()) {
      return false;
    }
    for (std::size_t index = 0; index < actual_axes.size(); ++index) {
      if (!placement_->axes[index].accepts(actual_axes[index]))
        return false;
    }
    return true;
  }

  /** Validate one observed axis in trace order, including its Meta contract. */
  [[nodiscard]] bool accepts_axis(std::size_t index, const AxisContract& actual) const {
    if (placement_ == nullptr || index >= placement_->axes.size())
      return false;
    const auto& expected = placement_->axes[index];
    if (expected.kind != actual.kind || expected.alignment != actual.alignment || expected.lower != actual.lower ||
        expected.upper != actual.upper)
      return false;
    return expected.accepts(actual.recorded);
  }

  [[nodiscard]] std::size_t axis_count() const {
    return placement_ == nullptr ? 0 : placement_->axes.size();
  }

private:
  void bind_tier(BufferTier tier, std::byte*& base, nint_t capacity, nint_t required) const {
    VECOPS_CHECK(capacity >= 0, "workspace arena capacity is negative");
    VECOPS_CHECK(required == 0 || base != nullptr, "workspace arena is null");
    if (required == 0)
      return;
    nint_t alignment = 1;
    for (const auto& entry : placement_->entries) {
      if (entry.tier == tier)
        alignment = std::max(alignment, entry.alignment);
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(base);
    const auto aligned =
      (raw + static_cast<std::uintptr_t>(alignment - 1)) & ~static_cast<std::uintptr_t>(alignment - 1);
    const nint_t padding = static_cast<nint_t>(aligned - raw);
    VECOPS_CHECK(padding <= capacity && required <= capacity - padding,
                  "workspace arena is smaller than the aligned placement");
    base += padding;
  }

  const BufferPlacement* placement_ = nullptr;
  std::byte* fast_ = nullptr;
  std::byte* slow_ = nullptr;
};

namespace buffer_plan_details {

inline bool overlaps(const LogicalBufferAllocation& left, const LogicalBufferAllocation& right) {
  for (const auto& a : left.lifetimes) {
    for (const auto& b : right.lifetimes) {
      if (a.begin < b.end && b.begin < a.end)
        return true;
    }
  }
  return false;
}

struct PlacementCandidate {
  std::size_t logical_index;
  nint_t bytes;
  nint_t stride;
  nint_t alignment;
};

inline PlacementCandidate candidate(const LogicalBufferPlan& logical, std::size_t index) {
  const auto& request = logical.allocations[index].request;
  const nint_t alignment =
    request.domain == BufferDomain::WorkerLocal ? std::max(request.alignment, nint_t{64}) : request.alignment;
  const nint_t stride = request.domain == BufferDomain::WorkerLocal
                          ? checked_align_up(request.bytes, alignment)
                          : checked_align_up(request.bytes, request.alignment);
  return {index, request.domain == BufferDomain::WorkerLocal ? checked_mul(stride, request.replicas) : stride,
          stride, alignment};
}

/** First aligned gap in the physical intervals of live conflicts.
 * Keep placement order unchanged. Resolve each already placed allocation by
 * its stable logical index once, rather than searching the trace repeatedly
 * at every candidate address. Sorting physical conflicts makes first-fit a
 * single scan; this produces the same earliest offset as repeated jumps.
 */
inline nint_t place_in_tier(const LogicalBufferPlan& logical, const PlacementCandidate& candidate,
                            BufferTier tier, const std::vector<BufferPlacementEntry>& placed,
                            const std::vector<std::size_t>& placed_logical, nint_t capacity,
                            const std::vector<BufferReservation>& reserved = {}) {
  const auto& allocation = logical.allocations[candidate.logical_index];
  VECOPS_CHECK(placed.size() == placed_logical.size(), "invalid buffer placement index");
  std::vector<std::pair<nint_t, nint_t>> blocked;
  blocked.reserve(placed.size() + reserved.size());
  for (const auto& entry : reserved) {
    if (entry.tier == tier)
      blocked.emplace_back(entry.offset, checked_add(entry.offset, entry.bytes));
  }
  for (std::size_t index = 0; index < placed.size(); ++index) {
    const auto& entry = placed[index];
    if (entry.tier != tier || !overlaps(allocation, logical.allocations[placed_logical[index]]))
      continue;
    blocked.emplace_back(entry.offset, checked_add(entry.offset, checked_mul(entry.replica_stride, entry.replicas)));
  }
  std::sort(blocked.begin(), blocked.end());
  nint_t cursor = 0;
  for (const auto& [begin, end] : blocked) {
    if (end <= cursor)
      continue;
    if (checked_add(cursor, candidate.bytes) <= begin)
      break;
    cursor = checked_align_up(end, candidate.alignment);
  }
  const nint_t end = checked_add(cursor, candidate.bytes);
  return end <= capacity ? cursor : -1;
}

} // namespace buffer_plan_details

/** Deterministically color logical lifetimes into fast and slow arenas. */
inline BufferPlacement place_buffers(const LogicalBufferPlan& logical,
                                          nint_t fast_capacity = std::numeric_limits<nint_t>::max(),
                                          nint_t slow_capacity = std::numeric_limits<nint_t>::max(),
                                          const std::vector<BufferReservation>& reserved = {}) {
  VECOPS_CHECK(fast_capacity >= 0 && slow_capacity >= 0, "workspace capacities must be non-negative");
  std::vector<buffer_plan_details::PlacementCandidate> candidates;
  candidates.reserve(logical.allocations.size());
  for (std::size_t index = 0; index < logical.allocations.size(); ++index)
    candidates.push_back(buffer_plan_details::candidate(logical, index));
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
    const auto& l = logical.allocations[left.logical_index].request;
    const auto& r = logical.allocations[right.logical_index].request;
    if (l.placement != r.placement)
      return l.placement < r.placement;
    const double l_traffic = l.estimated_traffic_bytes == 0.0 ? double(left.bytes) : l.estimated_traffic_bytes;
    const double r_traffic = r.estimated_traffic_bytes == 0.0 ? double(right.bytes) : r.estimated_traffic_bytes;
    const double ld = left.bytes == 0 ? l_traffic : l_traffic / double(left.bytes);
    const double rd = right.bytes == 0 ? r_traffic : r_traffic / double(right.bytes);
    if (ld != rd)
      return ld > rd;
    return logical.allocations[left.logical_index].canonical_path <
           logical.allocations[right.logical_index].canonical_path;
  });

  BufferPlacement result{logical.fingerprint, logical.axes, {}, 0, 0};
  result.entries.reserve(candidates.size());
  std::vector<std::size_t> placed_logical;
  placed_logical.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const auto& allocation = logical.allocations[candidate.logical_index];
    const bool may_use_fast = allocation.request.placement != BufferPlacementPolicy::SlowAllowed;
    nint_t offset = may_use_fast ? buffer_plan_details::place_in_tier(logical, candidate, BufferTier::Fast,
                                                                         result.entries, placed_logical, fast_capacity, reserved)
                                 : -1;
    BufferTier tier = BufferTier::Fast;
    if (offset < 0) {
      VECOPS_CHECK(allocation.request.placement != BufferPlacementPolicy::FastRequired,
                    "required-fast workspace does not fit in the fast arena");
      tier = BufferTier::Slow;
      offset = buffer_plan_details::place_in_tier(logical, candidate, tier, result.entries, placed_logical, slow_capacity, reserved);
      VECOPS_CHECK(offset >= 0, "workspace plan does not fit in either arena");
    }
    result.entries.push_back(BufferPlacementEntry{
      allocation.site, tier, offset, allocation.request.bytes, candidate.stride, allocation.request.replicas,
      candidate.alignment, allocation.request.domain, allocation.request.alignment, allocation.request.placement});
    placed_logical.push_back(candidate.logical_index);
    const nint_t end = buffer_plan_details::checked_add(offset, candidate.bytes);
    if (tier == BufferTier::Fast)
      result.fast_bytes = std::max(result.fast_bytes, end);
    else
      result.slow_bytes = std::max(result.slow_bytes, end);
  }
  return result;
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_BUFFER_PLAN_H
