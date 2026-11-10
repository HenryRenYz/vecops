// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "details/RuntimeInternal.h"
#include <unistd.h>

namespace vecops::memory::model::details {
namespace {
std::vector<PoolBudget> budgets(const LinuxDiscovery& d, const LinuxRuntimeOptions& o) {
  std::vector<PoolBudget> result;
  for (const auto& g : d.graph.capacity_groups) {
    require(g.capacity_bytes.has_value(), "runtime requires discovered capacity");
    U free = *g.capacity_bytes;
    for (const auto& p : d.graph.pools)
      if (p.capacity_group == g.id) {
        auto it = d.pool_available_bytes.find(p.id);
        require(it != d.pool_available_bytes.end(), "missing discovered free capacity");
        free = std::min(free, it->second);
      }
    U limit = *g.capacity_bytes;
    if (auto it = o.group_hard_limits.find(g.id); it != o.group_hard_limits.end()) {
      require(it->second <= limit, "group hard limit exceeds physical capacity");
      limit = it->second;
    }
    PoolBudget b{};
    b.capacity_group = g.id;
    b.hard_limit_bytes = limit;
    b.system_available_bytes = free;
    result.push_back(b);
  }
  for (const auto& [id, n] : o.group_hard_limits) {
    (void)n;
    find(d.graph.capacity_groups, id);
  }
  return result;
}
} // namespace
Core::Core(LinuxDiscovery d, LinuxRuntimeOptions o)
  : model(d.graph)
  , pages(d.pool_bindings)
  , pool_bindings(std::move(d.pool_bindings))
  , scope(d.scope)
  , options(std::move(o))
  , descriptors(std::move(d.descriptors))
  , ledger(std::make_shared<CapacityLedger>(budgets(d, options), options.shared_limits)) {
  if (options.participant.empty())
    options.participant = "process/" + std::to_string(::getpid());
  canonical_view = std::make_shared<const ResourceView>(take(model.view(scope, 1)));
  if (options.capacity_directory) {
    std::vector<SharedCapacityGroup> groups;
    for (const auto& b : ledger->snapshot().groups)
      groups.push_back({b.capacity_group, *find(model.graph().capacity_groups, b.capacity_group).capacity_bytes,
                        b.hard_limit_bytes});
    authority = take(SharedCapacityAuthority::open(*options.capacity_directory, std::move(groups),
                                                   options.shared_limits, options.participant));
  }
  for (const auto& [queue, agent] : options.cpu_queues) {
    require(!queue.empty() && has(scope.executable_agents, agent), "queue executor not executable");
    const auto& a = find(model.graph().agents, agent);
    require(has(a.capabilities, std::string("host_load_store")), "host queue executor lacks host capability");
  }
  for (const auto& id : options.cpu_copy_methods)
    find(model.graph().methods, id);
  std::set<std::pair<std::string, std::string>> device_methods;
  for (const auto& [queue, binding] : options.transfer_queues) {
    require(!queue.empty() && !options.cpu_queues.contains(queue) && binding.executor,
            "invalid/duplicate device queue");
    require(has(scope.executable_agents, binding.agent) && has(scope.executable_agents, binding.controller),
            "device queue executor/controller not executable");
    require(has(find(model.graph().agents, binding.agent).capabilities, std::string("dma")),
            "device queue executor lacks DMA capability");
    require(has(find(model.graph().agents, binding.controller).capabilities, std::string("host_load_store")),
            "device queue controller lacks host capability");
    require(!binding.methods.empty(), "device queue has no methods");
    for (const auto& method : binding.methods) {
      require(!has(options.cpu_copy_methods, method) && device_methods.emplace(binding.agent, method).second,
              "ambiguous device method binding");
      const auto& m = find(model.graph().methods, method);
      require(std::any_of(m.routes.begin(), m.routes.end(), [&](const auto& r) {
                return r.agent == binding.agent && r.controller == binding.controller;
              }), "device method has no matching agent/controller route");
    }
  }
  require(options.transfer_wait_ms > 0, "invalid device completion window");
}
ResourceView Core::authorize(const ResourceView& v) const {
  return *authorize_shared(v);
}
std::shared_ptr<const ResourceView> Core::authorize_shared(const ResourceView& v) const {
  require(v.epoch == 1, "stale runtime resource view");
  for (const auto& a : v.executable_agents)
    require(has(scope.executable_agents, a), "view grants an unavailable agent");
  for (const auto& p : v.allocatable_pools)
    require(has(scope.allocatable_pools, p), "view grants unavailable allocation memory");
  for (const auto& p : v.accessible_pools)
    require(has(scope.accessible_pools, p), "view grants unavailable accessible memory");
  for (const auto& [agent, n] : v.execution_slots) {
    auto it = scope.execution_slots.find(agent);
    require(it != scope.execution_slots.end() && n <= it->second, "view overgrants executor slots");
  }
  // Compare grants only. Caller-provided graph values are never authoritative.
  // The launch scope and model are immutable for this runtime epoch.
  if (v.executable_agents == canonical_view->executable_agents &&
      v.allocatable_pools == canonical_view->allocatable_pools &&
      v.accessible_pools == canonical_view->accessible_pools &&
      v.execution_slots == canonical_view->execution_slots)
    return canonical_view;
  return std::make_shared<const ResourceView>(
      take(model.view({v.executable_agents, v.allocatable_pools, v.accessible_pools, v.execution_slots}, 1)));
}
std::unique_ptr<PoolQuota> Core::reserve_pools(const std::vector<BackingRequest>& requests) {
  std::lock_guard lock(admission);
  std::map<std::string, U> asked, committed;
  for (const auto& r : requests)
    if (r.owner == options.participant)
      asked[r.pool] = add(asked[r.pool], r.bytes);
  for (const auto& [identity, weak] : observed)
    if (auto b = weak.lock()) {
      (void)identity;
      for (const auto& s : b->info().segments)
        committed[s.pool] = add(committed[s.pool], s.size_bytes);
    }
  for (const auto& [pool, n] : asked) {
    const auto& p = find(model.graph().pools, pool);
    require(p.capacity_bytes.has_value(), "unknown pool capacity");
    const U used = add(committed[pool], pool_pending[pool]);
    require(used <= *p.capacity_bytes && n <= *p.capacity_bytes - used, "pool capacity exhausted");
  }
  auto q = std::make_unique<PoolQuota>();
  q->core = shared_from_this();
  q->remaining = asked;
  for (const auto& [pool, n] : asked)
    pool_pending[pool] += n;
  return q;
}
PoolQuota::~PoolQuota() {
  if (!core)
    return;
  std::lock_guard lock(core->admission);
  for (const auto& [pool, n] : remaining)
    core->pool_pending[pool] -= n;
}
void PoolQuota::consume(const BackingRequest& r) {
  auto it = remaining.find(r.pool);
  require(it != remaining.end() && it->second >= r.bytes, "backing exceeds pool grant");
  core->pool_pending[r.pool] -= r.bytes;
  it->second -= r.bytes;
}
std::shared_ptr<SharedBacking> Core::materialize(const BackingRequest& request, RootGrant& grant, PoolQuota& quota,
                                                 CapacityState initial, std::shared_ptr<CapacityCharge>* charge) {
  std::lock_guard lock(admission);
  for (auto it = observed.begin(); it != observed.end();)
    if (it->second.expired()) {
      records.erase(it->first);
      it = observed.erase(it);
    } else
      ++it;
  require(request.owner == options.participant, "cannot materialize another participant's backing");
  const auto& p = find(model.graph().pools, request.pool);
  auto owner = std::make_shared<ChargeSlot>();
  auto b = take(pages.allocate(request, owner));
  owner->charge = take(grant.local->commit(p.capacity_group, request.bytes, initial));
  auto identity = take(LinuxPageStore::physical_identity(*b));
  if (grant.shared)
    owner->shared_charge = take(grant.shared->commit(p.capacity_group, request.bytes, identity));
  observed[identity] = b;
  records[identity] = {identity, options.participant, p.capacity_group, request.bytes, b->info().segments, true};
  quota.consume(request);
  if (charge)
    *charge = owner->charge;
  return b;
}
std::map<std::string, U> Core::physical_free(bool require_bindings) const {
  // Synthetic discovery fixtures deliberately have no OS binding. Their free
  // sample comes from the local ledger; certified live pools always query Linux.
  const bool has_nodes = std::any_of(pool_bindings.begin(), pool_bindings.end(),
                                     [](const auto& b) { return !b.numa_nodes.empty(); });
  std::map<std::string, U> free;
  if (!has_nodes && !require_bindings) {
    for (const auto& group : ledger->snapshot().groups)
      free[group.capacity_group] = add(group.system_available_bytes, group.unbacked_reserved_bytes);
    return free;
  }
  const auto physical = take(probe_linux_topology());
  for (const auto& group : model.graph().capacity_groups) {
    require(group.capacity_bytes.has_value(), "unknown physical capacity");
    U available = *group.capacity_bytes;
    for (const auto& pool : model.graph().pools)
      if (pool.capacity_group == group.id) {
        auto binding = std::find_if(pool_bindings.begin(), pool_bindings.end(),
                                    [&](const auto& b) { return b.pool == pool.id; });
        require(binding != pool_bindings.end() && !binding->numa_nodes.empty(),
                "pool has no certified Linux free-capacity source");
        U pool_free = 0;
        std::set<unsigned> seen;
        for (auto node : binding->numa_nodes) {
          require(seen.insert(node).second, "duplicate pool NUMA binding");
          const auto it = std::find_if(physical.nodes.begin(), physical.nodes.end(),
                                       [&](const auto& n) { return n.os_numa == node; });
          require(it != physical.nodes.end(), "physical NUMA binding disappeared");
          pool_free = add(pool_free, it->available_bytes);
        }
        available = std::min(available, pool_free);
      }
    free.emplace(group.id, available);
  }
  return free;
}
std::unique_ptr<RootGrant> Core::reserve_capacity(const std::map<std::string, U>& bytes,
                                                 std::optional<U> shared_epoch) {
  std::lock_guard lock(admission);
  auto grant = std::make_unique<RootGrant>();
  grant->local = take(ledger->reserve_current(bytes));
  if (authority) {
    // Explicit plans retain their snapshot contract. Automatic root growth
    // samples and admits under one authority lock; no competing epoch refresh
    // or retry can hide a materialized root or repeat an uncertain transaction.
    grant->shared = shared_epoch
      ? take(authority->reserve(bytes, *shared_epoch, physical_free()))
      : take(authority->reserve_current(bytes, [&] { return protect([&] { return physical_free(); }); }));
  }
  return grant;
}
BudgetSnapshot Core::snapshot(const ResourceView& input) const {
  std::lock_guard lock(admission);
  const auto v = authorize(input);
  auto result = ledger->snapshot();
  if (authority) {
    const auto shared = take(authority->snapshot());
    const auto free = physical_free();
    for (auto& group : result.groups) {
      const auto& global = find(shared.groups, group.capacity_group);
      group.shared_remaining_bytes = global.hard_remaining_bytes;
      const U sample = free.at(group.capacity_group);
      const U remaining = sample > global.pending_bytes ? sample - global.pending_bytes : 0;
      group.system_available_bytes = std::min(group.system_available_bytes, remaining);
    }
    for (auto& c : result.constraints)
      c.remaining_bytes = std::min(c.remaining_bytes, find(shared.constraints, c.id).remaining_bytes);
    const auto signature = std::pair{result.epoch, shared.epoch};
    if (signature != budget_signature) {
      require(budget_epoch != std::numeric_limits<U>::max(), "budget epoch exhausted");
      ++budget_epoch;
      budget_signature = signature;
    }
    result.epoch = budget_epoch;
    snapshot_shared_epoch = shared.epoch;
  }
  std::set<std::string> visible;
  for (const auto& p : v.graph.pools)
    visible.insert(p.capacity_group);
  result.groups.erase(std::remove_if(result.groups.begin(), result.groups.end(),
                                     [&](const auto& b) { return !visible.contains(b.capacity_group); }),
                      result.groups.end());
  for (const auto& p : v.graph.pools) {
    const auto& g = *std::find_if(result.groups.begin(), result.groups.end(),
                                  [&](const auto& b) { return b.capacity_group == p.capacity_group; });
    U available = has(v.allocatable_pools, p.id) ? take(budget_available(g)) : 0;
    U committed = 0;
    for (const auto& [identity, w] : observed)
      if (auto root = w.lock()) {
        (void)identity;
        for (const auto& seg : root->info().segments)
          if (seg.pool == p.id)
            committed = add(committed, seg.size_bytes);
      }
    const auto pending = pool_pending.contains(p.id) ? pool_pending.at(p.id) : 0;
    const auto idle = pool_idle.contains(p.id) ? pool_idle.at(p.id) : 0;
    if (p.capacity_bytes) {
      const U used = add(committed, pending);
      const U remaining = used >= *p.capacity_bytes ? 0 : *p.capacity_bytes - used;
      available = std::min(available, add(remaining, idle));
    }
    PoolAvailability a{p.id, p.capacity_group, available, g.largest_cached_block_bytes, {}};
    for (const auto& c : result.constraints)
      if (has(c.capacity_groups, p.capacity_group)) {
        a.shared_constraints.push_back(c.id);
        a.allocatable_bytes = std::min(a.allocatable_bytes, add(g.idle_available_bytes, c.remaining_bytes));
      }
    result.pools.push_back(std::move(a));
  }
  return result;
}
} // namespace vecops::memory::model::details
