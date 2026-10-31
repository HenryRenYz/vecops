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
KernelPlan KernelBuilder::finish() {
  check_open();
  nint_t active = 1;
  for (const auto& action : data_->actions)
    active = std::max(active, action->active);
  data_->workers = active;
  finished_ = true;
  return KernelPlan{std::move(data_)};
}

KernelInstance::KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool,
                               std::shared_ptr<WorkspaceArenaProvider> provider)
  : plan_(std::move(plan))
  , pool_(pool ? pool : vecops_cpu_thread_pool_v1())
  , provider_(std::move(provider)) {
  if (!provider_)
    throw std::invalid_argument("kernel workspace provider is null");
  // Logical-task providers can virtualize lanes, but must never execute a
  // resident barrier. They retain an explicit synchronous-grid fallback.
  if (supports_grid_team(pool_) && plan_.workers() > max_parallelism(pool_))
    throw std::invalid_argument("kernel instance exceeds resident team capacity");
  frame_.slots.resize(plan_.data_->slots.size());
  frame_.owner = plan_.data_.get();
  frame_.provider = provider_;
  // Place all declared intermediates and operator scratch together before
  // binding state. One allocation per tier also respects session providers
  // that return the same shared arena on repeated allocation requests.
  struct Placement {
    WorkspaceTier tier = WorkspaceTier::Slow;
    nint_t offset = 0;
  };
  std::vector<Placement> placement(frame_.slots.size());
  nint_t fast_bytes = 0, slow_bytes = 0, alignment = 64;
  bool required_fast = false;
  auto place = [&](bool prefer_fast) {
    fast_bytes = slow_bytes = 0;
    for (std::size_t i = 0; i < plan_.data_->slots.size(); ++i) {
      const auto& slot = plan_.data_->slots[i];
      if (slot.kind != kernel_plan_details::SlotKind::Buffer)
        continue;
      alignment = std::max(alignment, slot.alignment);
      required_fast |= slot.placement == WorkspacePlacementPolicy::FastRequired;
      const auto fast_offset = workspace_plan_details::checked_align_up(fast_bytes, slot.alignment);
      const bool fast = prefer_fast && slot.placement != WorkspacePlacementPolicy::SlowAllowed &&
                        fast_offset <= provider_->capacity(WorkspaceTier::Fast) &&
                        slot.bytes <= provider_->capacity(WorkspaceTier::Fast) - fast_offset;
      if (!fast && slot.placement == WorkspacePlacementPolicy::FastRequired)
        throw WorkspaceArenaUnavailable("required-fast kernel storage does not fit");
      auto& size = fast ? fast_bytes : slow_bytes;
      const auto offset = workspace_plan_details::checked_align_up(size, slot.alignment);
      size = workspace_tensor_details::checked_add(offset, slot.bytes);
      if (size > provider_->capacity(fast ? WorkspaceTier::Fast : WorkspaceTier::Slow))
        throw WorkspaceArenaUnavailable("kernel storage exceeds arena capacity");
      placement[i] = {fast ? WorkspaceTier::Fast : WorkspaceTier::Slow, offset};
    }
  };
  place(true);
  WorkspaceArena fast, slow;
  try {
    fast = provider_->allocate(WorkspaceTier::Fast, fast_bytes, alignment);
  } catch (const std::bad_alloc&) {
    if (required_fast)
      throw;
    place(false);
  } catch (const WorkspaceArenaUnavailable&) {
    if (required_fast)
      throw;
    place(false);
  }
  slow = provider_->allocate(WorkspaceTier::Slow, slow_bytes, alignment);
  for (std::size_t i = 0; i < plan_.data_->slots.size(); ++i) {
    if (plan_.data_->slots[i].kind != kernel_plan_details::SlotKind::Buffer)
      continue;
    const auto& entry = placement[i];
    auto* base = static_cast<std::byte*>(entry.tier == WorkspaceTier::Fast ? fast.data : slow.data);
    frame_.slots[i].address = entry.offset ? base + entry.offset : base;
    frame_.slots[i].bound = true;
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
} // namespace vecops::execution
