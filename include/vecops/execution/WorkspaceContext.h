//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_WORKSPACE_CONTEXT_H
#define VECOPS_EXECUTION_WORKSPACE_CONTEXT_H

/**
 * @file vecops/execution/WorkspaceContext.h
 * @brief Kernel-call workspace authority with dynamic, trace, and replay modes.
 *
 * `WorkspaceContext` is the object intended to become the first argument of a
 * compiler-managed `__kernel__`. A default context services requests from an
 * optional caller-provided fast arena and spills preferred requests to aligned
 * host storage. Trace mode performs the same real allocations while recording
 * a pointer-free plan. Replay mode resolves stable sites from a bound plan.
 *
 * The context is intentionally single-threaded. Parallel regions must fork one
 * logical worker context per worker; they must not race on this object.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vecops/Assertion.h"
#include "vecops/execution/Parallel.h"
#include "vecops/execution/WorkspaceArena.h"
#include "vecops/execution/WorkspacePlan.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/runtime/CallAbi.h"

namespace vecops::execution {

struct TraceWorkspaceTag {};
inline constexpr TraceWorkspaceTag trace_workspace{};

class WorkspaceContext {
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
    friend class WorkspaceContext;
    Scope(WorkspaceContext* owner, std::size_t depth)
      : owner_(owner)
      , depth_(depth) {
    }

    WorkspaceContext* owner_ = nullptr;
    std::size_t depth_ = 0;
  };

  /** Dynamic mode. Preferred allocations use `fast_base` while it fits. */
  explicit WorkspaceContext(std::string recipe = "kernel", void* fast_base = nullptr, nint_t fast_capacity = 0)
    : mode_(Mode::Dynamic)
    , recipe_(std::move(recipe))
    , fast_(fast_base, fast_capacity) {
    initialize_root();
  }

  /** Trace mode: execute dynamically and retain a logical allocation plan. */
  WorkspaceContext(TraceWorkspaceTag, std::string recipe, DecisionFingerprint fingerprint = {},
                   void* fast_base = nullptr, nint_t fast_capacity = 0)
    : mode_(Mode::Trace)
    , recipe_(std::move(recipe))
    , fast_(fast_base, fast_capacity)
    , trace_(std::make_unique<WorkspaceTrace>(recipe_, fingerprint)) {
    initialize_root();
  }

  /** Replay mode over an already placed and bound plan. */
  WorkspaceContext(std::string recipe, const BoundWorkspacePlan& replay)
    : mode_(Mode::Replay)
    , recipe_(std::move(recipe))
    , replay_(&replay) {
    initialize_root();
  }

  WorkspaceContext(const WorkspaceContext&) = delete;
  WorkspaceContext& operator=(const WorkspaceContext&) = delete;
  WorkspaceContext(WorkspaceContext&&) = delete;
  WorkspaceContext& operator=(WorkspaceContext&&) = delete;

  Scope serial_scope(std::string_view name) {
    VECOPS_ASSERT(!finished_, "cannot enter a finished workspace context");
    VECOPS_ASSERT(!name.empty(), "workspace scope name must not be empty");
    auto path_hash = workspace_plan_details::hash("/", frames_.back().path_hash);
    path_hash = workspace_plan_details::hash(name, path_hash);
    std::optional<WorkspaceTrace::Scope> trace_scope;
    if (trace_ != nullptr)
      trace_scope.emplace(trace_->serial_scope(name));
    frames_.push_back(Frame{path_hash, fast_.mark(), owned_.size(), std::move(trace_scope)});
    return Scope{this, frames_.size() - 1};
  }

  /** Allocate one stable semantic site in the active lexical scope. */
  BoundWorkspaceSlot request(std::string_view local_name, WorkspaceAllocationRequest request) {
    return allocate_site(local_name, request, false);
  }

  /**
   * Bind scratch retained by a prepared operator.
   *
   * Trace and replay retain the same lexical lifetime used for placement. In
   * dynamic mode a spilled block is owned until the context dies, because the
   * prepared object may outlive the construction scope that selected it.
   */
  BoundWorkspaceSlot bind(std::string_view local_name, WorkspaceAllocationRequest request) {
    return allocate_site(local_name, request, true);
  }

  /** Convenience binding for operators using the default scratch contract. */
  BoundWorkspaceSlot bind(std::string_view local_name, nint_t bytes) {
    return bind(local_name, WorkspaceAllocationRequest{.bytes = bytes});
  }

  /** Add one Meta axis contract to a trace before publishing it. */
  void add_axis(AxisContract contract) {
    VECOPS_ASSERT(trace_ != nullptr, "axis contracts require trace mode");
    trace_->add_axis(contract);
  }

  /**
   * Observe a Meta axis on every execution mode. Trace records it, replay
   * validates it in order, and unplanned dynamic execution has no bookkeeping.
   */
  void observe_axis(AxisContract contract) {
    VECOPS_ASSERT(!finished_, "cannot observe an axis on a finished workspace context");
    if (mode_ == Mode::Trace) {
      trace_->add_axis(contract);
    } else if (mode_ == Mode::Replay) {
      VECOPS_ASSERT(replay_->accepts_axis(observed_axes_, contract),
                    "workspace replay axis differs from the recorded Meta contract");
      ++observed_axes_;
    }
  }

  /** Finish and take the pointer-free trace. No later request is permitted. */
  LogicalWorkspacePlan finish_trace() {
    VECOPS_ASSERT(trace_ != nullptr, "workspace context is not tracing");
    VECOPS_ASSERT(frames_.size() == 1, "workspace context still has an open scope");
    finished_ = true;
    return trace_->finish();
  }

  /** Finish one replay region and verify that every recorded axis was seen. */
  void finish_replay() {
    VECOPS_ASSERT(mode_ == Mode::Replay, "workspace context is not replaying");
    VECOPS_ASSERT(frames_.size() == 1, "workspace context still has an open scope");
    VECOPS_ASSERT(observed_axes_ == replay_->axis_count(), "workspace replay did not observe every recorded Meta axis");
    finished_ = true;
  }

  /** Begin a pristine replay or reset a completed one while retaining storage. */
  void restart_replay() {
    VECOPS_ASSERT(mode_ == Mode::Replay, "only replay contexts can be restarted");
    VECOPS_ASSERT(frames_.size() == 1, "workspace replay can restart only at its root scope");
    VECOPS_ASSERT(finished_ || observed_axes_ == 0, "an incomplete workspace replay cannot be restarted");
    observed_axes_ = 0;
    finished_ = false;
  }

  [[nodiscard]] bool is_tracing() const {
    return mode_ == Mode::Trace;
  }
  [[nodiscard]] bool is_replaying() const {
    return mode_ == Mode::Replay;
  }

private:
  enum class Mode : std::uint8_t { Dynamic, Trace, Replay };

  BoundWorkspaceSlot allocate_site(std::string_view local_name, WorkspaceAllocationRequest request, bool persistent) {
    VECOPS_ASSERT(!finished_, "cannot allocate from a finished workspace context");
    validate_request(local_name, request);
    const auto site = workspace_site_id(frames_.back().path_hash, local_name, request.domain);
    if (trace_ != nullptr) {
      const auto traced = trace_->request(local_name, request);
      VECOPS_ASSERT(traced == site, "workspace trace and execution site IDs differ");
    }
    if (mode_ == Mode::Replay) {
      return replay_->find(site, request);
    }

    const nint_t allocation_alignment =
      request.domain == WorkspaceDomain::WorkerLocal ? std::max(request.alignment, nint_t{64}) : request.alignment;
    const nint_t stride = request.domain == WorkspaceDomain::WorkerLocal
                            ? workspace_plan_details::checked_align_up(request.bytes, allocation_alignment)
                            : workspace_plan_details::checked_align_up(request.bytes, request.alignment);
    const nint_t total = request.domain == WorkspaceDomain::WorkerLocal
                           ? workspace_plan_details::checked_mul(stride, request.replicas)
                           : stride;
    if (total == 0)
      return {nullptr, request.bytes, stride, request.replicas};

    if (request.placement != WorkspacePlacementPolicy::SlowAllowed && fast_.can_allocate(total, allocation_alignment)) {
      auto* data = static_cast<std::byte*>(fast_.allocate(total, allocation_alignment));
      return {data, request.bytes, stride, request.replicas};
    }
    VECOPS_ASSERT(request.placement != WorkspacePlacementPolicy::FastRequired,
                  "required-fast workspace request does not fit in the fast arena");
    const nint_t heap_alignment = std::max(allocation_alignment, static_cast<nint_t>(alignof(std::max_align_t)));
    VECOPS_ASSERT(static_cast<std::uint64_t>(total) <=
                    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                  "workspace request exceeds the host address space");
    void* data =
      ::operator new(static_cast<std::size_t>(total), std::align_val_t(static_cast<std::size_t>(heap_alignment)));
    auto& owner = persistent ? persistent_owned_ : owned_;
    owner.emplace_back(data, heap_alignment);
    return {static_cast<std::byte*>(data), request.bytes, stride, request.replicas};
  }

  struct OwnedBlock {
    OwnedBlock(void* pointer, nint_t alignment)
      : pointer(pointer)
      , alignment(alignment) {
    }
    OwnedBlock(const OwnedBlock&) = delete;
    OwnedBlock& operator=(const OwnedBlock&) = delete;
    OwnedBlock(OwnedBlock&& other) noexcept
      : pointer(std::exchange(other.pointer, nullptr))
      , alignment(other.alignment) {
    }
    OwnedBlock& operator=(OwnedBlock&&) = delete;
    ~OwnedBlock() {
      if (pointer != nullptr) {
        ::operator delete(pointer, std::align_val_t(static_cast<std::size_t>(alignment)));
      }
    }
    void* pointer;
    nint_t alignment;
  };

  struct Frame {
    std::uint64_t path_hash;
    kernel::WorkspaceView::Mark fast_mark;
    std::size_t owned_mark;
    std::optional<WorkspaceTrace::Scope> trace_scope;
  };

  void initialize_root() {
    VECOPS_ASSERT(!recipe_.empty(), "workspace recipe name must not be empty");
    VECOPS_ASSERT(mode_ != Mode::Replay || replay_ != nullptr, "workspace replay plan is null");
    frames_.push_back(Frame{workspace_plan_details::hash(recipe_), fast_.mark(), 0, std::nullopt});
  }

  static void validate_request(std::string_view local_name, const WorkspaceAllocationRequest& request) {
    VECOPS_ASSERT(!local_name.empty(), "workspace allocation name must not be empty");
    VECOPS_ASSERT(request.bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_ASSERT(request.alignment > 0 && (request.alignment & (request.alignment - 1)) == 0,
                  "workspace allocation alignment must be a positive power of two");
    VECOPS_ASSERT(request.replicas > 0, "workspace replica count must be positive");
    if (request.domain == WorkspaceDomain::Global) {
      VECOPS_ASSERT(request.replicas == 1, "global workspace request cannot have replicas");
    }
  }

  void close_scope(std::size_t depth) {
    VECOPS_ASSERT(depth + 1 == frames_.size() && depth != 0, "workspace scopes must close in LIFO order");
    auto& frame = frames_.back();
    if (frame.trace_scope.has_value())
      frame.trace_scope->close();
    fast_.rewind(frame.fast_mark);
    while (owned_.size() > frame.owned_mark)
      owned_.pop_back();
    frames_.pop_back();
  }

  Mode mode_;
  std::string recipe_;
  kernel::WorkspaceView fast_;
  const BoundWorkspacePlan* replay_ = nullptr;
  std::unique_ptr<WorkspaceTrace> trace_;
  std::vector<OwnedBlock> owned_;
  std::vector<OwnedBlock> persistent_owned_;
  std::vector<Frame> frames_;
  std::size_t observed_axes_ = 0;
  bool finished_ = false;
};

/**
 * @brief Small per-calling-thread cache for default source-kernel replay.
 *
 * Framework bridges commonly invoke a kernel without an explicit workspace
 * authority. Reconstructing a Dynamic WorkspaceContext in that path would
 * allocate and free every spilled site on every call. This cache executes a
 * new Meta/decision shape once in trace mode, owns the resulting placed
 * arenas, and uses replay for later identical calls. It is intentionally an
 * execution-thread object: callers that need cross-thread ownership, HBM
 * placement, or model-wide lifetime coloring should pass their own context.
 * The ambient maximum parallelism is folded into every decision fingerprint
 * so worker-local replica counts remain valid when a framework changes its
 * thread setting between calls.
 *
 * `Setup` must accept an object exposing `observe_axis(AxisContract)` and
 * `Invoke` must synchronously accept `WorkspaceContext&`. Allocation topology
 * may depend only on the observed axes and supplied decision fingerprint.
 */
class WorkspaceReplayCache {
public:
  explicit WorkspaceReplayCache(
      std::size_t capacity = 4,
      std::shared_ptr<WorkspaceArenaProvider> arena_provider = default_workspace_arena_provider())
    : capacity_(capacity)
    , arena_provider_(std::move(arena_provider)) {
    VECOPS_ASSERT(capacity_ > 0, "workspace replay cache capacity must be positive");
    VECOPS_ASSERT(arena_provider_ != nullptr, "workspace replay cache arena provider is null");
  }

  WorkspaceReplayCache(const WorkspaceReplayCache&) = delete;
  WorkspaceReplayCache& operator=(const WorkspaceReplayCache&) = delete;

  template <typename Setup, typename Invoke>
  void invoke(std::string_view recipe, DecisionFingerprint fingerprint, Setup&& setup, Invoke&& invoke) {
    // Worker-local allocation counts commonly depend on the ambient OpenMP
    // team size. Tensor metadata alone therefore cannot identify a safe
    // replay plan when a framework changes its thread count between calls.
    fingerprint.decision = workspace_plan_details::mix(
      fingerprint.decision, static_cast<std::uint64_t>(max_parallelism()));
    AxisCollector observed;
    setup(observed);
    auto found = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
      return entry.fingerprint == fingerprint && same_axes(entry.axes, observed);
    });
    if (found != entries_.end()) {
      entries_.splice(entries_.begin(), entries_, found);
      auto& entry = entries_.front();
      entry.replay->restart_replay();
      try {
        setup(*entry.replay);
        invoke(*entry.replay);
        entry.replay->finish_replay();
      } catch (...) {
        entry.rebuild_replay(recipe);
        throw;
      }
      return;
    }

    WorkspaceContext tracing{trace_workspace, std::string(recipe), fingerprint};
    setup(tracing);
    invoke(tracing);
    auto logical = tracing.finish_trace();
    VECOPS_ASSERT(logical.fingerprint == fingerprint && same_axes(logical.axes, observed),
                  "workspace trace observations differ from the replay cache key");
    entries_.emplace_front(recipe, std::move(logical), arena_provider_);
    if (entries_.size() > capacity_)
      entries_.pop_back();
  }

  template <typename Setup, typename Invoke>
  void invoke(std::string_view recipe, Setup&& setup, Invoke&& invoke) {
    this->invoke(recipe, {}, std::forward<Setup>(setup), std::forward<Invoke>(invoke));
  }

private:
  struct AxisCollector {
    static constexpr std::size_t InlineCapacity = 16;

    void observe_axis(AxisContract contract) {
      if (size_ < InlineCapacity) {
        inline_axes[size_] = contract;
      } else {
        overflow.push_back(contract);
      }
      ++size_;
    }

    [[nodiscard]] std::size_t size() const {
      return size_;
    }

    [[nodiscard]] const AxisContract& operator[](std::size_t index) const {
      return index < InlineCapacity ? inline_axes[index] : overflow[index - InlineCapacity];
    }

    std::array<AxisContract, InlineCapacity> inline_axes{};
    std::vector<AxisContract> overflow;
    std::size_t size_ = 0;
  };

  struct Entry {
    Entry(std::string_view recipe, LogicalWorkspacePlan logical,
          const std::shared_ptr<WorkspaceArenaProvider>& arena_provider)
      : fingerprint(logical.fingerprint)
      , axes(logical.axes)
      , placement(place_workspace(logical, arena_provider->capacity(WorkspaceTier::Fast),
                                  arena_provider->capacity(WorkspaceTier::Slow))) {
      try {
        reserve(arena_provider);
      } catch (const WorkspaceArenaUnavailable&) {
        fallback_to_slow(logical, arena_provider);
      } catch (const std::bad_alloc&) {
        fallback_to_slow(logical, arena_provider);
      }
      rebuild_replay(recipe);
    }

    Entry(const Entry&) = delete;
    Entry& operator=(const Entry&) = delete;
    Entry(Entry&&) = delete;
    Entry& operator=(Entry&&) = delete;

    static nint_t tier_alignment(const WorkspacePlacement& placement, WorkspaceTier tier) {
      nint_t result = vec::DEFAULT_ALIGNMENT;
      for (const auto& entry : placement.entries) {
        if (entry.tier == tier)
          result = std::max(result, entry.alignment);
      }
      return result;
    }

    void fallback_to_slow(const LogicalWorkspacePlan& logical,
                          const std::shared_ptr<WorkspaceArenaProvider>& arena_provider) {
      const bool requires_fast = std::any_of(logical.allocations.begin(), logical.allocations.end(),
                                             [](const auto& allocation) {
                                               return allocation.request.placement ==
                                                      WorkspacePlacementPolicy::FastRequired;
                                             });
      if (requires_fast)
        throw;
      placement = place_workspace(logical, 0, arena_provider->capacity(WorkspaceTier::Slow));
      fast = {};
      slow = {};
      reserve(arena_provider);
    }

    void reserve(const std::shared_ptr<WorkspaceArenaProvider>& arena_provider) {
      fast = arena_provider->allocate(WorkspaceTier::Fast, placement.fast_bytes,
                                      tier_alignment(placement, WorkspaceTier::Fast));
      slow = arena_provider->allocate(WorkspaceTier::Slow, placement.slow_bytes,
                                      tier_alignment(placement, WorkspaceTier::Slow));
      bound.emplace(placement, fast.data, fast.capacity, slow.data, slow.capacity);
    }

    void rebuild_replay(std::string_view recipe) {
      replay = std::make_unique<WorkspaceContext>(std::string(recipe), *bound);
    }

    DecisionFingerprint fingerprint;
    std::vector<AxisContract> axes;
    WorkspacePlacement placement;
    WorkspaceArena fast;
    WorkspaceArena slow;
    std::optional<BoundWorkspacePlan> bound;
    std::unique_ptr<WorkspaceContext> replay;
  };

  template <typename Axes>
  static bool same_axes(const std::vector<AxisContract>& left, const Axes& right) {
    if (left.size() != right.size())
      return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
      const auto& a = left[index];
      const auto& b = right[index];
      if (a.kind != b.kind || a.exact != b.exact || a.alignment != b.alignment || a.lower != b.lower ||
          a.upper != b.upper || a.recorded != b.recorded || a.bucket_upper != b.bucket_upper)
        return false;
    }
    return true;
  }

  std::size_t capacity_;
  std::shared_ptr<WorkspaceArenaProvider> arena_provider_;
  std::list<Entry> entries_;
};

/**
 * Build a non-owning ABI context that lets a generated source-kernel adapter
 * use this exact trace/replay/dynamic authority. The returned record and the
 * WorkspaceContext must both remain alive for the synchronous call.
 */
inline VecopsExecutionContext workspace_execution_context(WorkspaceContext& workspace,
                                                          std::uint32_t requested_threads = 0, void* stream = nullptr,
                                                          std::uint64_t extra_flags = 0) {
  constexpr auto reserved_flags = VECOPS_EXECUTION_CONTEXT_FLAG_WORKSPACE_CONTEXT |
                                  VECOPS_EXECUTION_CONTEXT_FLAG_WORKSPACE_ARENA_PROVIDER;
  VECOPS_ASSERT((extra_flags & reserved_flags) == 0, "workspace execution-context flags are managed by vecops");
  return {sizeof(VecopsExecutionContext), requested_threads, stream, &workspace,
          extra_flags | VECOPS_EXECUTION_CONTEXT_FLAG_WORKSPACE_CONTEXT};
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_CONTEXT_H
