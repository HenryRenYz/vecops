//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_WORKSPACE_PLAN_H
#define VECOPS_EXECUTION_WORKSPACE_PLAN_H

/**
 * @file vecops/execution/WorkspacePlan.h
 * @brief Pointer-free workspace traces, deterministic placement, and binding.
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

namespace workspace_plan_details {

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
  VECOPS_ASSERT(left >= 0 && right >= 0, "workspace sizes must be non-negative");
  VECOPS_ASSERT(left <= std::numeric_limits<nint_t>::max() - right, "workspace size addition overflow");
  return left + right;
}

inline nint_t checked_mul(nint_t left, nint_t right) {
  VECOPS_ASSERT(left >= 0 && right >= 0, "workspace sizes must be non-negative");
  VECOPS_ASSERT(left == 0 || right <= std::numeric_limits<nint_t>::max() / left,
                "workspace size multiplication overflow");
  return left * right;
}

inline nint_t checked_align_up(nint_t value, nint_t alignment) {
  VECOPS_ASSERT(value >= 0, "workspace size must be non-negative");
  VECOPS_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0,
                "workspace alignment must be a positive power of two");
  return checked_add(value, alignment - 1) & ~(alignment - 1);
}

} // namespace workspace_plan_details

/** Stable, plan-local allocation identity. */
struct WorkspaceSiteId {
  std::uint64_t path_hash = 0;
  std::uint64_t semantic_hash = 0;

  friend constexpr bool operator==(WorkspaceSiteId, WorkspaceSiteId) = default;
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
      VECOPS_ASSERT(actual <= bucket_upper, "workspace axis bucket is smaller than the recorded value");
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

enum class WorkspaceDomain : std::uint8_t { Global, WorkerLocal };

/** Build the stable ID shared by trace, dynamic execution, and replay. */
inline WorkspaceSiteId workspace_site_id(std::string_view canonical_path, std::string_view local_name,
                                         WorkspaceDomain domain) {
  auto path_hash = workspace_plan_details::hash(canonical_path);
  path_hash = workspace_plan_details::hash("/", path_hash);
  path_hash = workspace_plan_details::hash(local_name, path_hash);
  return {path_hash,
          workspace_plan_details::mix(workspace_plan_details::hash(local_name), static_cast<std::uint64_t>(domain))};
}

/** Build a site ID from an incrementally maintained scope-path hash. */
inline WorkspaceSiteId workspace_site_id(std::uint64_t scope_path_hash, std::string_view local_name,
                                         WorkspaceDomain domain) {
  auto path_hash = workspace_plan_details::hash("/", scope_path_hash);
  path_hash = workspace_plan_details::hash(local_name, path_hash);
  return {path_hash,
          workspace_plan_details::mix(workspace_plan_details::hash(local_name), static_cast<std::uint64_t>(domain))};
}

enum class WorkspacePlacementPolicy : std::uint8_t {
  FastRequired,
  FastPreferred,
  SlowAllowed,
};

enum class WorkspaceTier : std::uint8_t { Fast, Slow };

struct WorkspaceAllocationRequest {
  nint_t bytes = 0;
  nint_t alignment = vec::DEFAULT_ALIGNMENT;
  WorkspaceDomain domain = WorkspaceDomain::Global;
  nint_t replicas = 1;
  WorkspacePlacementPolicy placement = WorkspacePlacementPolicy::FastPreferred;
  double benefit = 1.0;
};

struct WorkspaceLifetime {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct LogicalWorkspaceAllocation {
  WorkspaceSiteId site;
  std::string canonical_path;
  WorkspaceAllocationRequest request;
  std::vector<WorkspaceLifetime> lifetimes;
};

/** Pointer-free trace result; safe to cache across arena rebinding. */
struct LogicalWorkspacePlan {
  DecisionFingerprint fingerprint;
  std::vector<AxisContract> axes;
  std::vector<LogicalWorkspaceAllocation> allocations;
};

/** Structured trace builder with lexical scratch lifetimes. */
class WorkspaceTrace {
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
    friend class WorkspaceTrace;
    Scope(WorkspaceTrace* owner, std::size_t depth)
      : owner_(owner)
      , depth_(depth) {
    }

    WorkspaceTrace* owner_ = nullptr;
    std::size_t depth_ = 0;
  };

  explicit WorkspaceTrace(std::string recipe, DecisionFingerprint fingerprint = {})
    : recipe_(std::move(recipe))
    , fingerprint_(fingerprint) {
    VECOPS_ASSERT(!recipe_.empty(), "workspace recipe name must not be empty");
    frames_.push_back(Frame{recipe_, ++clock_, {}});
  }

  WorkspaceTrace(const WorkspaceTrace&) = delete;
  WorkspaceTrace& operator=(const WorkspaceTrace&) = delete;

  Scope serial_scope(std::string_view name) {
    VECOPS_ASSERT(!finished_, "cannot add a scope to a finished workspace trace");
    VECOPS_ASSERT(!name.empty(), "workspace scope name must not be empty");
    std::string path = frames_.back().path;
    path.push_back('/');
    path.append(name);
    frames_.push_back(Frame{std::move(path), ++clock_, {}});
    return Scope{this, frames_.size() - 1};
  }

  void add_axis(AxisContract contract) {
    VECOPS_ASSERT(!finished_, "cannot add an axis to a finished workspace trace");
    axes_.push_back(contract);
  }

  WorkspaceSiteId request(std::string_view local_name, WorkspaceAllocationRequest request) {
    VECOPS_ASSERT(!finished_, "cannot allocate from a finished workspace trace");
    VECOPS_ASSERT(!local_name.empty(), "workspace allocation name must not be empty");
    VECOPS_ASSERT(request.bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_ASSERT(request.alignment > 0 && (request.alignment & (request.alignment - 1)) == 0,
                  "workspace allocation alignment must be a positive power of two");
    VECOPS_ASSERT(request.replicas > 0, "workspace replica count must be positive");
    if (request.domain == WorkspaceDomain::Global) {
      VECOPS_ASSERT(request.replicas == 1, "global workspace cannot have replicas");
    }

    std::string path = frames_.back().path;
    path.push_back('/');
    path.append(local_name);
    const WorkspaceSiteId site = workspace_site_id(frames_.back().path, local_name, request.domain);

    std::size_t allocation_index = allocations_.size();
    for (std::size_t index = 0; index < allocations_.size(); ++index) {
      if (allocations_[index].site == site) {
        allocation_index = index;
        break;
      }
    }
    if (allocation_index == allocations_.size()) {
      allocations_.push_back(LogicalWorkspaceAllocation{site, path, request, {}});
    } else {
      auto& existing = allocations_[allocation_index];
      VECOPS_ASSERT(existing.canonical_path == path, "workspace site hash collision");
      VECOPS_ASSERT(compatible(existing.request, request),
                    "workspace site was replayed with a different allocation contract");
      for (const auto& pending : pending_) {
        VECOPS_ASSERT(pending.closed || pending.allocation_index != allocation_index,
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

  LogicalWorkspacePlan finish() {
    VECOPS_ASSERT(!finished_, "workspace trace can only be finished once");
    VECOPS_ASSERT(frames_.size() == 1, "workspace trace still has an open nested scope");
    close_scope(0);
    finished_ = true;
    return LogicalWorkspacePlan{fingerprint_, std::move(axes_), std::move(allocations_)};
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

  static bool compatible(const WorkspaceAllocationRequest& left, const WorkspaceAllocationRequest& right) {
    return left.alignment == right.alignment && left.domain == right.domain && left.placement == right.placement &&
           left.benefit == right.benefit;
  }

  void close_scope(std::size_t depth) {
    VECOPS_ASSERT(depth + 1 == frames_.size(), "workspace scopes must close in LIFO order");
    const auto end = ++clock_;
    for (const auto pending_index : frames_.back().pending) {
      auto& pending = pending_[pending_index];
      VECOPS_ASSERT(!pending.closed, "workspace lifetime was closed twice");
      allocations_[pending.allocation_index].lifetimes.push_back(WorkspaceLifetime{pending.begin, end});
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
  std::vector<LogicalWorkspaceAllocation> allocations_;
};

struct WorkspacePlacementEntry {
  WorkspaceSiteId site;
  WorkspaceTier tier = WorkspaceTier::Slow;
  nint_t offset = 0;
  nint_t bytes = 0;
  nint_t replica_stride = 0;
  nint_t replicas = 1;
  nint_t alignment = vec::DEFAULT_ALIGNMENT;
  WorkspaceDomain domain = WorkspaceDomain::Global;
  nint_t request_alignment = vec::DEFAULT_ALIGNMENT;
  WorkspacePlacementPolicy placement = WorkspacePlacementPolicy::FastPreferred;
};

struct WorkspacePlacement {
  DecisionFingerprint fingerprint;
  std::vector<AxisContract> axes;
  std::vector<WorkspacePlacementEntry> entries;
  nint_t fast_bytes = 0;
  nint_t slow_bytes = 0;
};

struct BoundWorkspaceSlot {
  std::byte* base = nullptr;
  nint_t bytes = 0;
  nint_t replica_stride = 0;
  nint_t replicas = 1;

  [[nodiscard]] void* replica(nint_t index = 0) const {
    VECOPS_ASSERT(0 <= index && index < replicas, "workspace replica is out of range");
    return base == nullptr ? nullptr : base + index * replica_stride;
  }
};

class BoundWorkspacePlan {
public:
  BoundWorkspacePlan() = default;

  BoundWorkspacePlan(const WorkspacePlacement& placement, void* fast_base, nint_t fast_capacity, void* slow_base,
                     nint_t slow_capacity)
    : placement_(&placement)
    , fast_(static_cast<std::byte*>(fast_base))
    , slow_(static_cast<std::byte*>(slow_base)) {
    bind_tier(WorkspaceTier::Fast, fast_, fast_capacity, placement.fast_bytes);
    bind_tier(WorkspaceTier::Slow, slow_, slow_capacity, placement.slow_bytes);
  }

  [[nodiscard]] BoundWorkspaceSlot find(WorkspaceSiteId site, nint_t actual_bytes = -1) const {
    VECOPS_ASSERT(placement_ != nullptr, "workspace plan is not bound");
    for (const auto& entry : placement_->entries) {
      if (entry.site == site) {
        if (actual_bytes >= 0) {
          VECOPS_ASSERT(actual_bytes <= entry.bytes, "workspace replay request exceeds its planned capacity");
        }
        auto* tier = entry.tier == WorkspaceTier::Fast ? fast_ : slow_;
        return {tier == nullptr ? nullptr : tier + entry.offset, entry.bytes, entry.replica_stride, entry.replicas};
      }
    }
    VECOPS_ASSERT(false, "workspace site is absent from the bound plan");
    return {};
  }

  /** Resolve a replay site while validating its complete allocation contract. */
  [[nodiscard]] BoundWorkspaceSlot find(WorkspaceSiteId site, const WorkspaceAllocationRequest& request) const {
    VECOPS_ASSERT(placement_ != nullptr, "workspace plan is not bound");
    for (const auto& entry : placement_->entries) {
      if (entry.site == site) {
        VECOPS_ASSERT(request.bytes <= entry.bytes, "workspace replay request exceeds its planned capacity");
        VECOPS_ASSERT(request.alignment == entry.request_alignment,
                      "workspace replay alignment differs from the recorded contract");
        VECOPS_ASSERT(request.replicas <= entry.replicas,
                      "workspace replay replica count exceeds its planned capacity");
        VECOPS_ASSERT(request.domain == entry.domain, "workspace replay domain differs from the recorded contract");
        VECOPS_ASSERT(request.placement == entry.placement,
                      "workspace replay placement policy differs from the recorded contract");
        auto* tier = entry.tier == WorkspaceTier::Fast ? fast_ : slow_;
        return {tier == nullptr ? nullptr : tier + entry.offset, entry.bytes, entry.replica_stride, entry.replicas};
      }
    }
    VECOPS_ASSERT(false, "workspace site is absent from the bound plan");
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
  void bind_tier(WorkspaceTier tier, std::byte*& base, nint_t capacity, nint_t required) const {
    VECOPS_ASSERT(capacity >= 0, "workspace arena capacity is negative");
    VECOPS_ASSERT(required == 0 || base != nullptr, "workspace arena is null");
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
    VECOPS_ASSERT(padding <= capacity && required <= capacity - padding,
                  "workspace arena is smaller than the aligned placement");
    base += padding;
  }

  const WorkspacePlacement* placement_ = nullptr;
  std::byte* fast_ = nullptr;
  std::byte* slow_ = nullptr;
};

namespace workspace_plan_details {

inline bool overlaps(const LogicalWorkspaceAllocation& left, const LogicalWorkspaceAllocation& right) {
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

inline PlacementCandidate candidate(const LogicalWorkspacePlan& logical, std::size_t index) {
  const auto& request = logical.allocations[index].request;
  const nint_t alignment =
    request.domain == WorkspaceDomain::WorkerLocal ? std::max(request.alignment, nint_t{64}) : request.alignment;
  const nint_t stride = request.domain == WorkspaceDomain::WorkerLocal
                          ? checked_align_up(request.bytes, alignment)
                          : checked_align_up(request.bytes, request.alignment);
  return {index, request.domain == WorkspaceDomain::WorkerLocal ? checked_mul(stride, request.replicas) : stride,
          stride, alignment};
}

inline nint_t place_in_tier(const LogicalWorkspacePlan& logical, const PlacementCandidate& candidate,
                            WorkspaceTier tier, const std::vector<WorkspacePlacementEntry>& placed, nint_t capacity) {
  const auto& allocation = logical.allocations[candidate.logical_index];
  const nint_t alignment = candidate.alignment;
  nint_t cursor = 0;
  while (true) {
    cursor = checked_align_up(cursor, alignment);
    nint_t next_cursor = -1;
    bool conflict = false;
    for (const auto& entry : placed) {
      if (entry.tier != tier)
        continue;
      const auto existing = std::find_if(logical.allocations.begin(), logical.allocations.end(),
                                         [&](const auto& item) { return item.site == entry.site; });
      VECOPS_ASSERT(existing != logical.allocations.end(), "invalid workspace placement entry");
      if (!overlaps(allocation, *existing))
        continue;
      const nint_t candidate_end = checked_add(cursor, candidate.bytes);
      const nint_t entry_bytes = checked_mul(entry.replica_stride, entry.replicas);
      const nint_t entry_end = checked_add(entry.offset, entry_bytes);
      if (cursor < entry_end && entry.offset < candidate_end) {
        conflict = true;
        next_cursor = std::max(next_cursor, entry_end);
      }
    }
    if (!conflict)
      break;
    cursor = next_cursor;
  }
  const nint_t end = checked_add(cursor, candidate.bytes);
  return end <= capacity ? cursor : -1;
}

} // namespace workspace_plan_details

/** Deterministically color logical lifetimes into fast and slow arenas. */
inline WorkspacePlacement place_workspace(const LogicalWorkspacePlan& logical,
                                          nint_t fast_capacity = std::numeric_limits<nint_t>::max(),
                                          nint_t slow_capacity = std::numeric_limits<nint_t>::max()) {
  VECOPS_ASSERT(fast_capacity >= 0 && slow_capacity >= 0, "workspace capacities must be non-negative");
  std::vector<workspace_plan_details::PlacementCandidate> candidates;
  candidates.reserve(logical.allocations.size());
  for (std::size_t index = 0; index < logical.allocations.size(); ++index)
    candidates.push_back(workspace_plan_details::candidate(logical, index));
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
    const auto& l = logical.allocations[left.logical_index].request;
    const auto& r = logical.allocations[right.logical_index].request;
    if (l.placement != r.placement)
      return l.placement < r.placement;
    const double ld = left.bytes == 0 ? l.benefit : l.benefit / double(left.bytes);
    const double rd = right.bytes == 0 ? r.benefit : r.benefit / double(right.bytes);
    if (ld != rd)
      return ld > rd;
    return logical.allocations[left.logical_index].canonical_path <
           logical.allocations[right.logical_index].canonical_path;
  });

  WorkspacePlacement result{logical.fingerprint, logical.axes, {}, 0, 0};
  result.entries.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const auto& allocation = logical.allocations[candidate.logical_index];
    const bool may_use_fast = allocation.request.placement != WorkspacePlacementPolicy::SlowAllowed;
    nint_t offset = may_use_fast ? workspace_plan_details::place_in_tier(logical, candidate, WorkspaceTier::Fast,
                                                                         result.entries, fast_capacity)
                                 : -1;
    WorkspaceTier tier = WorkspaceTier::Fast;
    if (offset < 0) {
      VECOPS_ASSERT(allocation.request.placement != WorkspacePlacementPolicy::FastRequired,
                    "required-fast workspace does not fit in the fast arena");
      tier = WorkspaceTier::Slow;
      offset = workspace_plan_details::place_in_tier(logical, candidate, tier, result.entries, slow_capacity);
      VECOPS_ASSERT(offset >= 0, "workspace plan does not fit in either arena");
    }
    result.entries.push_back(WorkspacePlacementEntry{
      allocation.site, tier, offset, allocation.request.bytes, candidate.stride, allocation.request.replicas,
      candidate.alignment, allocation.request.domain, allocation.request.alignment, allocation.request.placement});
    const nint_t end = workspace_plan_details::checked_add(offset, candidate.bytes);
    if (tier == WorkspaceTier::Fast)
      result.fast_bytes = std::max(result.fast_bytes, end);
    else
      result.slow_bytes = std::max(result.slow_bytes, end);
  }
  return result;
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_PLAN_H
