// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file KernelPlan.cpp @brief Instance ownership and one-submission replay. */
#include "vecops/execution/KernelPlan.h"
#include <algorithm>

namespace vecops::execution {

KernelBuilder::KernelBuilder(nint_t workers)
  : data_(std::make_shared<kernel_plan_details::PlanData>()) {
  if (workers <= 0)
    throw std::invalid_argument("kernel team width must be positive");
  data_->workers = workers;
}
void KernelBuilder::check_open() const {
  if (finished_)
    throw std::logic_error("kernel builder is already finished");
}
KernelStage KernelBuilder::record(std::shared_ptr<const kernel_plan_details::Action> action,
                                 kernel_plan_details::Resources resources) {
  const auto stage = data_->actions.size();
  if (data_->actions.capacity() == stage)
    data_->actions.reserve(std::max<std::size_t>(stage + 1, stage * 2));
  if (data_->action_resources.capacity() == stage)
    data_->action_resources.reserve(std::max<std::size_t>(stage + 1, stage * 2));
  data_->actions.push_back(std::move(action));
  data_->action_resources.push_back(std::move(resources));
  return {data_.get(), stage};
}
KernelPlan KernelBuilder::finish() {
  check_open();
  nint_t active = 1;
  for (const auto& action : data_->actions)
    active = std::max(active, action->active);
  data_->workers = active;
  // Factories can retain pointers or initialized packing across replays.
  // Without a separately audited state contract these buffers remain pinned.
  using namespace kernel_plan_details;
  std::vector<bool> retained(data_->slots.size(), false);
  auto retain = [&](auto&& self, const Resources& resources) -> void {
    for (const auto& access : resources.buffers) retained[access.slot] = true;
    for (auto state : resources.states) self(self, data_->factories[state]->resources);
  };
  for (const auto& factory : data_->factories)
    if (!factory->transient_workspace) retain(retain, factory->resources);
  std::vector<std::vector<std::pair<std::size_t, ResourceAccess>>> uses(data_->slots.size());
  auto collect = [&](auto&& self, const Resources& resources, std::size_t stage) -> void {
    for (const auto& access : resources.buffers) {
      if (access.mode == AccessMode::Overwrite && data_->slots[access.slot].kind == SlotKind::Buffer &&
          (access.offset != 0 || access.bytes != data_->slots[access.slot].bytes))
        throw std::invalid_argument("overwrite must cover the root buffer, not a partial alias");
      uses[access.slot].emplace_back(stage, access);
    }
    for (auto state : resources.states)
      self(self, data_->factories[state]->resources, stage);
  };
  for (std::size_t stage = 0; stage < data_->actions.size(); ++stage)
    collect(collect, data_->action_resources[stage], stage);
  for (auto [slot, stage] : data_->consumes)
    for (const auto& [use, access] : uses[slot])
      if (use > stage)
        throw std::invalid_argument("kernel resource is used after consume (including aliases/state)");
  const auto end = std::max<std::size_t>(1, data_->actions.size());
  data_->reusable.resize(data_->slots.size(), false);
  for (std::size_t slot = 0; slot < data_->slots.size(); ++slot) {
    const auto& spec = data_->slots[slot];
    if (spec.kind != SlotKind::Buffer)
      continue;
    const auto& accesses = uses[slot];
    bool known = !retained[slot] && !accesses.empty();
    bool initialized = false;
    bool call_temporary = !accesses.empty();
    for (const auto& [stage, access] : accesses) {
      known &= access.mode != AccessMode::Unknown;
      call_temporary &= access.mode == AccessMode::CallTemporary;
      if (stage == accesses.front().first) {
        initialized |= access.mode == AccessMode::Overwrite || access.mode == AccessMode::CallTemporary;
        // An alias read in the defining grid requires pre-existing contents.
        known &= access.mode != AccessMode::Read && access.mode != AccessMode::ReadWrite;
      }
    }
    const bool reusable = known && initialized;
    data_->reusable[slot] = reusable;
    std::vector<BufferLifetime> lifetimes;
    if (reusable && call_temporary) {
      for (const auto& [stage, access] : accesses) {
        if (lifetimes.empty() || lifetimes.back().end < stage)
          lifetimes.push_back({stage, stage + 1});
        else
          lifetimes.back().end = stage + 1;
      }
    } else {
      lifetimes.push_back({reusable ? accesses.front().first : 0, reusable ? accesses.back().first + 1 : end});
    }
    data_->buffers.allocations.push_back({{slot + 1, 0}, spec.name,
      {spec.replicas == 1 ? spec.bytes : spec.worker_bytes, spec.alignment,
       spec.replicas == 1 ? BufferDomain::Global : BufferDomain::WorkerLocal,
       spec.replicas, spec.placement, spec.estimated_traffic_bytes}, std::move(lifetimes)});
  }
  finished_ = true;
  return KernelPlan{std::move(data_)};
}

KernelInstance::KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool,
                               std::shared_ptr<WorkspaceArenaProvider> provider)
  : plan_(std::move(plan))
  , pool_(pool ? pool : vecops_cpu_thread_pool_v1())
  , provider_(std::move(provider)) {
  initialize_frame();
  // Place all declared intermediates and operator scratch together before
  // binding state. One allocation per tier also respects session providers
  // that return the same shared arena on repeated allocation requests.
  const auto& logical = plan_.buffer_plan();
  auto placement = provider_->place(logical);
  nint_t alignment = 64;
  bool required_fast = false;
  for (const auto& slot : plan_.data_->slots) {
    alignment = std::max(alignment, slot.alignment);
    required_fast |= slot.placement == WorkspacePlacementPolicy::FastRequired;
  }
  WorkspaceArena fast, slow;
  try {
    fast = provider_->allocate(WorkspaceTier::Fast, placement.fast_bytes, alignment);
  } catch (const std::bad_alloc&) {
    if (required_fast)
      throw;
    placement = place_workspace(logical, 0, provider_->capacity(WorkspaceTier::Slow));
  } catch (const WorkspaceArenaUnavailable&) {
    if (required_fast)
      throw;
    placement = place_workspace(logical, 0, provider_->capacity(WorkspaceTier::Slow));
  }
  slow = provider_->allocate(WorkspaceTier::Slow, placement.slow_bytes, alignment);
  bind_storage(placement, std::move(fast), std::move(slow));
}
void KernelInstance::initialize_frame() {
  if (!provider_)
    throw std::invalid_argument("kernel workspace provider is null");
  // Logical-task providers can virtualize lanes, but must never execute a
  // resident barrier. They retain an explicit synchronous-grid fallback.
  if (supports_grid_team(pool_) && plan_.workers() > max_parallelism(pool_))
    throw std::invalid_argument("kernel instance exceeds resident team capacity");
  frame_.slots.resize(plan_.data_->slots.size());
  frame_.owner = plan_.data_.get();
  frame_.provider = provider_;
}
KernelInstance::KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool,
    std::shared_ptr<WorkspaceArenaProvider> provider, const WorkspacePlacement& placement,
    WorkspaceArena fast, WorkspaceArena slow)
  : plan_(std::move(plan)), pool_(pool ? pool : vecops_cpu_thread_pool_v1()), provider_(std::move(provider)) {
  initialize_frame();
  bind_storage(placement, std::move(fast), std::move(slow));
}
void KernelInstance::bind_storage(const WorkspacePlacement& placement, WorkspaceArena fast, WorkspaceArena slow) {
  for (const auto& entry : placement.entries) {
    auto* base = static_cast<std::byte*>(entry.tier == WorkspaceTier::Fast ? fast.data : slow.data);
    auto& slot = frame_.slots[entry.site.path_hash - 1];
    slot.address = base == nullptr ? nullptr : base + entry.offset;
    slot.bound = true;
  }
  buffers_.reserve(2);
  buffers_.push_back(std::move(fast));
  buffers_.push_back(std::move(slow));
  frame_.states.reserve(plan_.data_->factories.size());
  for (const auto& factory : plan_.data_->factories)
    frame_.states.push_back(factory->make(frame_));
  if (pool_->retain)
    pool_->retain(pool_->context);
}
KernelInstance::~KernelInstance() {
  // Prepared-state destructors may still refer to their scratch arenas.
  frame_.states.clear();
  if (pool_ && pool_->release)
    pool_->release(pool_->context);
}
void KernelInstance::acquire() {
  if (running_.test_and_set(std::memory_order_acquire))
    throw std::logic_error("kernel instance is already in use");
}
void KernelInstance::release() noexcept {
  running_.clear(std::memory_order_release);
}
void KernelInstance::validate_bindings() const {
  for (std::size_t i = 0; i < frame_.slots.size(); ++i) {
    const auto& slot = frame_.slots[i];
    const auto& spec = plan_.data_->slots[i];
    if (spec.kind == kernel_plan_details::SlotKind::Buffer && spec.bytes &&
        provider_->capacity(WorkspaceTier::Fast) == 0 && provider_->capacity(WorkspaceTier::Slow) == 0)
      throw WorkspaceArenaUnavailable("kernel buffer provider is closed");
    if (!slot.bound || (spec.kind == kernel_plan_details::SlotKind::Tensor && spec.bytes && !slot.address))
      throw std::invalid_argument("kernel input slot is unbound or has no data");
  }
}
void KernelInstance::run_lane(nint_t lane, const GridTeam& team) const {
  if (team.worker_count() < plan_.workers())
    throw std::runtime_error("resident executor supplied fewer workers than the plan requires");
  for (const auto& action : plan_.data_->actions) {
    if (lane < action->active)
      action->run(*action, frame_, lane);
    team.barrier();
  }
}
void KernelInstance::replay() {
  acquire();
  try {
    validate_bindings();
    run_synchronous();
    ++replay_count_;
  } catch (...) {
    release();
    throw;
  }
  release();
}
void KernelInstance::run_synchronous() {
  if (plan_.workers() == 1 || supports_grid_team(pool_)) {
    grid_team(pool_, plan_.workers(), [&](GridTeam team) { run_lane(team.worker_id(), team); });
  } else {
    for (const auto& action : plan_.data_->actions) {
      parallel_tasks(pool_, action->active,
                     [&](ParallelContext task) { action->run(*action, frame_, task.thread_id); });
    }
  }
}
void KernelInstance::enter() {
  acquire();
  try {
    validate_bindings();
  } catch (...) {
    release();
    throw;
  }
}
void KernelInstance::execute_on_team(const GridTeam& team) const {
  run_lane(team.worker_id(), team);
}
void KernelInstance::leave(bool successful) noexcept {
  if (successful)
    ++replay_count_;
  release();
}

void KernelSequence::append(std::shared_ptr<KernelInstance> instance) {
  if (running_.test_and_set(std::memory_order_acquire))
    throw std::logic_error("kernel sequence is already in use");
  try {
    if (!instance)
      throw std::invalid_argument("null kernel sequence instance");
    if (!instances_.empty() && instance->pool_ != instances_[0]->pool_)
      throw std::invalid_argument("kernel sequence requires one provider");
    // Strong exception guarantee: reserve before updating either collection.
    instances_.reserve(instances_.size() + 1);
    if (std::find(unique_.begin(), unique_.end(), instance.get()) == unique_.end()) {
      unique_.reserve(unique_.size() + 1);
      unique_.push_back(instance.get());
    }
    instances_.push_back(std::move(instance));
  } catch (...) {
    running_.clear(std::memory_order_release);
    throw;
  }
  running_.clear(std::memory_order_release);
}
void KernelSequence::replay() {
  if (running_.test_and_set(std::memory_order_acquire))
    throw std::logic_error("kernel sequence is already in use");
  std::size_t acquired = 0;
  try {
    for (auto* instance : unique_) {
      instance->acquire();
      ++acquired;
      instance->validate_bindings();
    }
    if (!instances_.empty()) {
      auto& first = *instances_.front();
      nint_t workers = 1;
      for (auto* instance : unique_)
        workers = std::max(workers, instance->plan_.workers());
      if (workers == 1 || supports_grid_team(first.pool_)) {
        grid_team(first.pool_, workers, [&](GridTeam team) {
          for (const auto& instance : instances_)
            instance->run_lane(team.worker_id(), team);
        });
      } else {
        for (const auto& instance : instances_)
          instance->run_synchronous();
      }
      for (const auto& instance : instances_)
        ++instance->replay_count_;
    }
  } catch (...) {
    while (acquired)
      unique_[--acquired]->release();
    running_.clear(std::memory_order_release);
    throw;
  }
  while (acquired)
    unique_[--acquired]->release();
  running_.clear(std::memory_order_release);
}
KernelProgram::KernelProgram(std::vector<KernelPlan> plans, const VecopsThreadPoolV1* pool,
    std::shared_ptr<WorkspaceArenaProvider> provider) {
  if (!provider || plans.empty())
    throw std::invalid_argument("kernel program requires a provider and at least one phase");
  std::uint64_t end = 0;
  for (const auto& plan : plans)
    end += std::max<std::size_t>(1, plan.grid_count());
  std::uint64_t clock = 0;
  std::vector<std::size_t> slots;
  nint_t alignment = 64;
  bool required_fast = false;
  for (std::size_t phase = 0; phase < plans.size(); ++phase) {
    const auto& plan = plans[phase];
    for (auto allocation : plan.buffer_plan().allocations) {
      const auto slot = allocation.site.path_hash - 1;
      if (plan.data_->reusable[slot]) {
        for (auto& lifetime : allocation.lifetimes) {
          lifetime.begin += clock;
          lifetime.end += clock;
        }
      } else {
        allocation.lifetimes = {{0, end}};
      }
      slots.push_back(slot);
      allocation.site = {buffers_.allocations.size() + 1, phase};
      allocation.canonical_path = "phase" + std::to_string(phase) + "/" + allocation.canonical_path;
      alignment = std::max(alignment, allocation.request.alignment);
      required_fast |= allocation.request.placement == BufferPlacementPolicy::FastRequired;
      buffers_.allocations.push_back(std::move(allocation));
    }
    clock += std::max<std::size_t>(1, plan.grid_count());
  }
  placement_ = provider->place(buffers_);
  WorkspaceArena fast, slow;
  try {
    fast = provider->allocate(WorkspaceTier::Fast, placement_.fast_bytes, alignment);
  } catch (const std::bad_alloc&) {
    if (required_fast) throw;
    placement_ = place_buffers(buffers_, 0, provider->capacity(WorkspaceTier::Slow));
  } catch (const WorkspaceArenaUnavailable&) {
    if (required_fast) throw;
    placement_ = place_buffers(buffers_, 0, provider->capacity(WorkspaceTier::Slow));
  }
  slow = provider->allocate(WorkspaceTier::Slow, placement_.slow_bytes, alignment);
  // The common placer may reorder entries. Site IDs, not entry positions,
  // establish the map back to typed frame slots.
  std::vector<WorkspacePlacement> local(plans.size());
  for (const auto& entry : placement_.entries) {
    auto mapped = entry;
    mapped.site = {slots[entry.site.path_hash - 1] + 1, 0};
    local[entry.site.semantic_hash].entries.push_back(mapped);
  }
  for (std::size_t phase = 0; phase < plans.size(); ++phase) {
    auto instance = std::shared_ptr<KernelInstance>(new KernelInstance(
      std::move(plans[phase]), pool, provider, local[phase], fast, slow));
    sequence_.append(instance);
    instances_.push_back(std::move(instance));
  }
}

} // namespace vecops::execution
