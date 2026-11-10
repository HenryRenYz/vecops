// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "details/RuntimeInternal.h"
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <tuple>

namespace vecops::memory::model {
namespace {
using namespace details;
struct Lease final : PlanLease {
  std::shared_ptr<Core> issuer;
  std::string canonical;
  bool consumed = false;
  std::unique_ptr<RootGrant> grant;
  std::unique_ptr<PoolQuota> quota;
  std::map<std::string, CachedBacking> cached;
};
struct Backings final : BackingSet {
  std::shared_ptr<Core> issuer;
  std::string canonical;
  std::map<std::string, std::shared_ptr<SharedBacking>> objects;
  std::vector<AllocationInfo> owned;
  std::span<const AllocationInfo> owned_allocations() const override {
    return owned;
  }
  std::vector<std::string> accessible_backings() const override {
    std::vector<std::string> ids;
    for (const auto& [id, b] : objects) {
      (void)b;
      ids.push_back(id);
    }
    return ids;
  }
};
struct Prepared final : PreparedAddresses {
  std::shared_ptr<Core> issuer;
  AddressLayout layout;
  std::map<std::string, std::unique_ptr<VirtualReservation>> regions;
  // A region set can be bound only once until all old mappings are removed.
  mutable std::mutex mutex;
  bool bound = false;
  Result<VirtualAddress> slot(const std::string& id) const override {
    return protect([&] {
      const auto& s = find(layout.slots, id);
      const auto& r = find(layout.regions, s.region);
      auto base = take(LinuxPageStore::address(*regions.at(r.id)));
      return VirtualAddress{r.address_space, add(reinterpret_cast<std::uintptr_t>(base), s.offset_bytes)};
    });
  }
};
struct BindingStorage {
  std::shared_ptr<Prepared> prepared;
  std::shared_ptr<Backings> backings;
  std::vector<std::unique_ptr<MappedRange>> mappings;
  ~BindingStorage() {
    mappings.clear();
    if (prepared) {
      std::lock_guard lock(prepared->mutex);
      prepared->bound = false;
    }
  }
};
struct Binding final : MemoryBinding {
  std::shared_ptr<Core> core;
  std::shared_ptr<HostEventService> events;
  MemoryRecipe recipe;
  U version, epoch = 1, active = 0;
  mutable std::mutex mutex;
  std::shared_ptr<BindingStorage> storage;
  std::vector<std::shared_ptr<HostEventReference>> uses;
  ~Binding() override = default;
  Result<void*> slot(const std::string& id) const override {
    return protect([&] {
      std::lock_guard lock(mutex);
      require(storage != nullptr, "binding is unbound");
      find(recipe.addresses.slots, id);
      auto p = take(storage->prepared->slot(id));
      return reinterpret_cast<void*>(p.value);
    });
  }
  U generation() const override {
    return version;
  }
  Result<void> record_use(CompletionToken t) override {
    return protect([&] {
      require(t.binding_generation == version, "stale binding-use generation");
      std::shared_ptr<BindingStorage> owner;
      std::vector<std::shared_ptr<HostEventReference>> previous, finished;
      {
        std::lock_guard lock(mutex);
        require(storage != nullptr, "binding is unbound");
        owner = storage;
        previous = uses;
      }
      auto reference = take(events->retain_event(t));
      for (const auto& old : previous)
        if (stopped(take(old->query())))
          finished.push_back(old);
      const bool done = stopped(take(reference->query()));
      if (!done) {
        auto retained = events->retain_until(t, owner);
        if (!retained && !stopped(take(reference->query())))
          take(retained);
      }
      std::lock_guard lock(mutex);
      require(storage == owner, "binding changed during use registration");
      uses.erase(std::remove_if(uses.begin(), uses.end(), [&](const auto& old) { return has(finished, old); }),
                 uses.end());
      if (!done)
        uses.push_back(std::move(reference));
      require(epoch != std::numeric_limits<U>::max(), "binding use epoch exhausted");
      ++epoch;
    });
  }
};
struct Agreement final : LayoutAgreement {
  std::shared_ptr<Core> issuer;
  DistributedLayout layout;
  std::string canonical;
  const std::string& digest() const override {
    return canonical;
  }
  U generation() const override {
    return layout.generation;
  }
};
struct Imported final : ImportedMemory {
  ExternalMemoryView external;
  const ExternalMemoryView& view() const override {
    return external;
  }
  Result<TransferEndpoint> endpoint() const override {
    return protect([&] {
      require(external.known_pool.has_value(), "external placement is unknown");
      return TransferEndpoint{external.address.address_space, {{0, external.bytes, *external.known_pool}}};
    });
  }
  Result<MemoryDescriptor> describe(const std::string& symbol) const override {
    return LinuxPageStore::describe_external(external, symbol, 0);
  }
  MappingCapabilities mapping_capabilities() const override {
    return {1, false, false, false, false, false, {}, {}, {}, false};
  }
};
struct RuntimeMapping final : MappedRange {
  std::shared_ptr<Core> issuer;
  std::unique_ptr<MappedRange> native;
  U version = 0, epoch = 1;
  mutable std::mutex mutex;
  Result<void*> data() const override {
    std::lock_guard lock(mutex);
    return native->data();
  }
  U generation() const override {
    std::lock_guard lock(mutex);
    return version;
  }
};
struct FenceProof {
  const void* object;
  U generation, epoch;
};
class Runtime final : public MemoryRuntime {
  std::shared_ptr<Core> c_;
  std::shared_ptr<HostEventService> events_ = std::make_shared<HostEventService>();
  PerformanceDatabase performance_;
  std::unique_ptr<CalibratedCostModel> calibrated_costs_;
  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<MemoryArena>> arenas_;
  std::map<std::string, FenceProof> proofs_;
  std::map<std::string, DistributedLayout> layouts_;
  struct SharedEpoch {
    U epoch;
    enum Phase { begun, published, retiring } phase = begun;
    std::shared_ptr<HostEventReference> publication, retirement;
  };
  using SharedKey = std::tuple<std::string, U, U, std::string, U>;
  std::map<SharedKey, std::shared_ptr<SharedEpoch>> shared_;
  U shared_revision_ = 0;
  CompletionToken reference_event(std::vector<std::shared_ptr<HostEventReference>> references,
                                  const InvocationContext& context) {
    auto poll = [references](bool wait) -> Result<EventStatus> {
      return protect([&] {
        EventStatus out{EventState::succeeded, true, {}};
        for (const auto& reference : references) {
          const auto s = take(wait ? reference->wait() : reference->query());
          out.accesses_stopped = out.accesses_stopped && s.accesses_stopped;
          if (s.state == EventState::failed) {
            out.state = s.state;
            out.error = s.error;
          } else if (s.state == EventState::cancelled && out.state != EventState::failed)
            out.state = s.state;
          else if (s.state == EventState::pending && out.state == EventState::succeeded)
            out.state = s.state;
        }
        return out;
      });
    };
    HostEventCallbacks callbacks{[poll] { return poll(false); }, [poll] { return poll(true); }, {}};
    return take(events_->import_event({"host-callback", callbacks, {}}, context));
  }
  ResourceView authorized(const ResourceView& v) const {
    return c_->authorize(v);
  }
  std::vector<ArenaCache*> caches(std::span<MemoryArena* const> list) const {
    std::vector<ArenaCache*> result;
    for (auto* a : list) {
      auto* cache = dynamic_cast<ArenaCache*>(a);
      require(cache && cache->core() == c_.get(), "foreign arena cache");
      require(!has(result, cache), "duplicate arena cache");
      result.push_back(cache);
    }
    return result;
  }
  Binding& binding(MemoryBinding& b) const {
    auto* p = dynamic_cast<Binding*>(&b);
    require(p && p->core == c_, "foreign memory binding");
    return *p;
  }
  const LinuxTransferQueue* device_queue(const MemoryCommand& cmd) const {
    for (const auto& [symbol, queue] : c_->options.transfer_queues)
      if ((cmd.queue == symbol || cmd.queue == queue.agent) && has(queue.methods, cmd.method)) return &queue;
    return nullptr;
  }
  MemoryCommandPlan commands(const ResourceView& v, const MemoryCommandPlan& input) const {
    auto p = input;
    for (auto& cmd : p.commands) {
      if (const auto* device = device_queue(cmd)) {
        require(cmd.opcode == MemoryOpcode::copy, "device queue implements copy only");
        require(has(v.executable_agents, device->agent) && has(v.executable_agents, device->controller),
                "device queue executor/controller unavailable");
        cmd.queue = device->agent;
      } else {
        auto q = c_->options.cpu_queues.find(cmd.queue);
        require(q != c_->options.cpu_queues.end() && has(v.executable_agents, q->second),
                "unknown/unavailable host queue");
        cmd.queue = q->second;
        require(cmd.opcode != MemoryOpcode::register_range && cmd.opcode != MemoryOpcode::unregister_range,
                "host runtime has no registration method");
        if (cmd.opcode != MemoryOpcode::fence)
          require(has(c_->options.cpu_copy_methods, cmd.method), "method is not bound to a host implementation");
      }
      if (cmd.opcode != MemoryOpcode::fence) find(v.graph.methods, cmd.method);
    }
    return p;
  }
  void host_access(const ResourceView& v, const MemoryCommandPlan& p, const OperandEnvironment& env) const {
    auto check = [&](const MemoryCommand& cmd, const ByteRangeRef& r, AccessMode mode) {
      auto it = std::find_if(env.memories.begin(), env.memories.end(),
                             [&](const auto& d) { return d.kind == r.kind && d.symbol == r.memory; });
      require(it != env.memories.end(), "missing memory descriptor");
      const auto* device = device_queue(cmd);
      const auto host = device ? device->controller : cmd.queue;
      const auto& a = find(v.graph.agents, host);
      require(a.address_space && *a.address_space == it->address_space, "host command address space mismatch");
      for (const auto& s : it->spans)
        if (s.offset_bytes < add(r.offset_bytes, r.size_bytes) && r.offset_bytes < add(s.offset_bytes, s.size_bytes) &&
            s.pool) {
          bool usable = false;
          for (const auto& e : v.graph.access)
            if (e.agent == host && e.pool == *s.pool && has(e.access, mode) && e.mapping == MappingMode::direct &&
                e.prerequisites.empty())
              usable = true;
          require(usable, "host executor has no direct authorized memory access");
        }
    };
    for (const auto& cmd : p.commands) {
      if (cmd.source)
        check(cmd, *cmd.source, AccessMode::read);
      if (cmd.target)
        check(cmd, *cmd.target, AccessMode::write);
      if (cmd.opcode != MemoryOpcode::fence) {
        const auto& method = find(v.graph.methods, cmd.method);
        if (cmd.route)
          find(method.routes, *cmd.route);
        if (cmd.source && cmd.target) {
          auto descriptor = [&](const ByteRangeRef& ref) -> const MemoryDescriptor& {
            auto it = std::find_if(env.memories.begin(), env.memories.end(),
                                   [&](const auto& d) { return d.kind == ref.kind && d.symbol == ref.memory; });
            require(it != env.memories.end(), "unknown operand");
            return *it;
          };
          const auto& source = descriptor(*cmd.source);
          const auto& target = descriptor(*cmd.target);
          U done = 0;
          while (done < cmd.source->size_bytes) {
            const U so = add(cmd.source->offset_bytes, done), to = add(cmd.target->offset_bytes, done);
            const auto& a = *std::find_if(source.spans.begin(), source.spans.end(), [&](const auto& s) {
              return s.offset_bytes <= so && so - s.offset_bytes < s.size_bytes;
            });
            const auto& b = *std::find_if(target.spans.begin(), target.spans.end(), [&](const auto& s) {
              return s.offset_bytes <= to && to - s.offset_bytes < s.size_bytes;
            });
            if (device_queue(cmd)) require(a.pool && b.pool, "device copy requires certified endpoint pools");
            if (a.pool && b.pool) {
              const Route* selected = nullptr;
              for (const auto& route : method.routes)
                if (route.source == *a.pool && route.target == *b.pool && (!cmd.route || route.id == *cmd.route) &&
                    (!route.agent || *route.agent == cmd.queue) &&
                    (!route.controller || has(v.executable_agents, *route.controller)) &&
                    (!device_queue(cmd) || route.controller == device_queue(cmd)->controller)) {
                  require(selected == nullptr, "ambiguous host copy route");
                  selected = &route;
                }
              require(selected != nullptr, "no executable host copy route");
            } else
              require(!cmd.route, "explicit copy route requires known endpoint pools");
            done += std::min(
              {cmd.source->size_bytes - done, a.offset_bytes + a.size_bytes - so, b.offset_bytes + b.size_bytes - to});
          }
        }
      }
    }
  }
  static MemoryRecipe project(const DistributedLayout& l, const std::string& participant) {
    const auto& p = *std::find_if(l.participants.begin(), l.participants.end(),
                                  [&](const auto& p) { return p.participant == participant; });
    MemoryRecipe r{participant, l.id, l.generation, l.sharing_contracts, p.addresses, {}, p.mappings};
    for (const auto& b : l.backings)
      if (b.owner == participant ||
          std::any_of(p.mappings.begin(), p.mappings.end(), [&](const auto& m) { return m.backing == b.id; }))
        r.backings.push_back(b);
    return r;
  }
  static void compatible(const AddressLayout& requested, const AddressLayout& prepared) {
    for (const auto& r : requested.regions) {
      const auto& p = find(prepared.regions, r.id);
      require(r.address_space == p.address_space && r.page_bytes == p.page_bytes && r.bytes <= p.bytes &&
                std::max(r.page_bytes, r.base_alignment_bytes) <= std::max(p.page_bytes, p.base_alignment_bytes),
              "prepared region contract mismatch");
    }
    for (const auto& s : requested.slots) {
      const auto& p = find(prepared.slots, s.id);
      require(s.region == p.region && s.offset_bytes == p.offset_bytes && s.size_bytes == p.size_bytes &&
                s.sharing_contract == p.sharing_contract && s.placement.key == p.placement.key &&
                s.placement.address_space == p.placement.address_space &&
                s.placement.page_bytes == p.placement.page_bytes,
              "prepared slot contract mismatch");
      require(s.placement.segments.size() == p.placement.segments.size(), "prepared placement mismatch");
      for (std::size_t i = 0; i < s.placement.segments.size(); ++i) {
        const auto& a = s.placement.segments[i];
        const auto& b = p.placement.segments[i];
        require(a.offset_bytes == b.offset_bytes && a.size_bytes == b.size_bytes && a.pool == b.pool,
                "prepared placement mismatch");
      }
    }
  }

public:
  explicit Runtime(std::shared_ptr<Core> c)
    : c_(std::move(c)) {
    if (c_->options.calibration)
      calibrated_costs_ = std::make_unique<CalibratedCostModel>(c_->model, c_->options.calibration,
                                                               c_->options.calibration_controls);
  }
  const std::string& participant() const override {
    return c_->options.participant;
  }
  std::optional<std::string> capacity_authority() const override {
    if (c_->authority) return c_->authority->namespace_id();
    return {};
  }
  const ResourceGraph& physical_graph() const override {
    return c_->model.graph();
  }
  const PerformanceDatabase& performance() const override {
    return c_->options.calibration ? calibration_database(*c_->options.calibration) : performance_;
  }
  EventService& events() override {
    return *events_;
  }
  Result<MemoryObservation> observe(const ResourceView& input) const override {
    return protect([&] {
      auto v = authorized(input);
      std::lock_guard lock(c_->admission);
      MemoryObservation out{};
      std::set<std::string> groups;
      for (const auto& p : v.graph.pools)
        groups.insert(p.capacity_group);
      for (const auto& [identity, weak] : c_->observed)
        if (!weak.expired()) {
          const auto& r = c_->records.at(identity);
          if (groups.contains(r.capacity_group))
            out.allocations.push_back(r);
        }
      return out;
    });
  }
  Result<Scope> current_scope() const override {
    return {c_->scope};
  }
  Result<ResidencyObservation> observe_external(const ResourceView& input,
                                                const ExternalMemoryView& memory) const override {
    return protect([&] {
      const auto v=authorized(input);
      require(memory.address.address_space=="host", "Linux residency requires host address space");
      return take(c_->pages.observe_external(memory,v.accessible_pools));
    });
  }
  Result<void> refresh_available() override {
    return protect([&] {
      std::lock_guard lock(c_->admission);
      const auto free = c_->physical_free(true);
      take(c_->ledger->refresh_available(free));
    });
  }
  Result<ResourceView> view(const Scope& s) const override {
    return protect([&] { return authorized(take(c_->model.view(s, 1))); });
  }
  std::vector<BackendDescriptor> describe_bindings() const override {
    return c_->descriptors;
  }
  std::vector<std::string> arena_ids() const override {
    std::lock_guard lock(mutex_);
    std::vector<std::string> ids;
    for (const auto& [id, a] : arenas_) {
      (void)a;
      ids.push_back(id);
    }
    return ids;
  }
  Result<std::shared_ptr<MemoryArena>> arena(const std::string& id) override {
    return protect([&] {
      std::lock_guard lock(mutex_);
      auto i = arenas_.find(id);
      require(i != arenas_.end(), "unknown arena");
      return i->second;
    });
  }
  Result<std::shared_ptr<MemoryArena>> create_arena(const ResourceView& v, const ArenaOptions& options) override {
    return protect([&] {
      auto a = make_arena(c_, events_, v, options);
      std::lock_guard lock(mutex_);
      arenas_.emplace(a->id(), a);
      return a;
    });
  }
  BudgetSnapshot snapshot(const ResourceView& v, std::span<MemoryArena* const> list = {}) const override {
    std::lock_guard lock(c_->admission);
    auto selected = caches(list);
    auto b = c_->snapshot(v);
    std::map<std::string, U> reusable;
    for (auto* cache : selected)
      for (const auto& block : cache->blocks())
        reusable[block.pool] = add(reusable[block.pool], block.bytes);
    for (auto& p : b.pools) {
      p.largest_cached_block_bytes = 0;
      for (auto* a : selected)
        p.largest_cached_block_bytes = std::max(p.largest_cached_block_bytes, a->largest(p.pool));
      const U inaccessible = c_->pool_idle[p.pool] - reusable[p.pool];
      p.allocatable_bytes = p.allocatable_bytes > inaccessible ? p.allocatable_bytes - inaccessible : 0;
    }
    for (auto& g : b.groups) {
      g.largest_cached_block_bytes = 0;
      for (const auto& p : b.pools)
        if (p.capacity_group == g.capacity_group)
          g.largest_cached_block_bytes = std::max(g.largest_cached_block_bytes, p.largest_cached_block_bytes);
    }
    return b;
  }
  Result<void> validate_recipe(const ResourceView& v, const MemoryRecipe& r) const override {
    return protect([&] {
      take(c_->model.validate_recipe(authorized(v), r));
      require(r.participant == participant(), "recipe participant mismatch");
    });
  }
  Result<PlanEstimate> estimate(const ResourceView& v, const MemoryRecipe& r,
                                std::span<MemoryArena* const> list = {}) const override {
    return protect([&] {
      std::lock_guard lock(c_->admission);
      const auto canonical_view = authorized(v);
      take(validate_recipe(canonical_view, r));
      auto selected = caches(list);
      auto b = snapshot(v, list);
      PlanEstimate e{};
      e.feasible = true;
      std::map<std::string, U> reusable;
      std::vector<CachedBlock> blocks;
      for (auto* cache : selected) {
        auto own = cache->blocks();
        blocks.insert(blocks.end(), own.begin(), own.end());
        for (const auto& [g, n] : cache->idle_by_group())
          reusable[g] = add(reusable[g], n);
      }
      std::map<std::string, U> pool_need;
      for (const auto& req : r.backings)
        if (req.owner == participant()) {
          const auto& p = find(canonical_view.graph.pools, req.pool);
          e.peak_bytes_by_capacity_group[p.capacity_group] =
            add(e.peak_bytes_by_capacity_group[p.capacity_group], req.bytes);
          pool_need[req.pool] = add(pool_need[req.pool], req.bytes);
          U cached = 0;
          for (auto& block : blocks)
            if (block.pool == req.pool && block.bytes >= req.bytes && block.page_bytes == req.page_bytes &&
                block.mode == req.mode) {
              block.bytes -= req.bytes;
              cached = req.bytes;
              break;
            }
          e.cached_reuse_bytes_by_capacity_group[p.capacity_group] =
            add(e.cached_reuse_bytes_by_capacity_group[p.capacity_group], cached);
          e.new_backing_bytes_by_capacity_group[p.capacity_group] =
            add(e.new_backing_bytes_by_capacity_group[p.capacity_group], req.bytes - cached);
        }
      for (const auto& [g, n] : e.peak_bytes_by_capacity_group) {
        const auto& budget =
          *std::find_if(b.groups.begin(), b.groups.end(), [&](const auto& x) { return x.capacity_group == g; });
        const U fresh = take(budget_available(budget)) - budget.idle_available_bytes;
        if (e.new_backing_bytes_by_capacity_group[g] > fresh) {
          e.feasible = false;
          e.shortfalls.push_back({CapacityLimitKind::capacity_group, g,
                                  e.new_backing_bytes_by_capacity_group[g], fresh});
        }
        (void)n;
      }
      for (const auto& [p, n] : pool_need) {
        const auto& available =
          *std::find_if(b.pools.begin(), b.pools.end(), [&](const auto& x) { return x.pool == p; });
        if (n > available.allocatable_bytes) {
          e.feasible = false;
          e.shortfalls.push_back({CapacityLimitKind::pool, p, n, available.allocatable_bytes});
        }
      }
      for (const auto& constraint : b.constraints) {
        U need = 0;
        for (const auto& g : constraint.capacity_groups)
          need = add(need, e.new_backing_bytes_by_capacity_group[g]);
        if (need > constraint.remaining_bytes) {
          e.feasible = false;
          e.shortfalls.push_back({CapacityLimitKind::constraint, constraint.id, need, constraint.remaining_bytes});
        }
      }
      return e;
    });
  }
  const MemoryCostModel& costs() const override {
    return calibrated_costs_ ? static_cast<const MemoryCostModel&>(*calibrated_costs_) : c_->model;
  }
  Result<ProgressCapabilities> progress(const ResourceView& v, const std::string& method,
                                        const std::optional<std::string>& route) const override {
    return protect([&] {
      const auto checked = authorized(v);
      const auto& m = find(checked.graph.methods, method);
      if (route)
        find(m.routes, *route);
      if (has(c_->options.cpu_copy_methods, method))
        return ProgressCapabilities{ProgressMode::caller_driven, {}, 1, false, true};
      for (const auto& [symbol, queue] : c_->options.transfer_queues) {
        (void)symbol;
        if (has(queue.methods, method) && has(checked.executable_agents, queue.agent) &&
            has(checked.executable_agents, queue.controller)) return queue.executor->progress();
      }
      return ProgressCapabilities{ProgressMode::unknown, {}, 0, false, false};
    });
  }
  Result<OperandEnvironment> describe_operands(const ResourceView& v, const MemoryRecipe& r,
                                               std::span<const MemoryDescriptor> e) const override {
    return protect([&] { return take(c_->model.describe_operands(authorized(v), r, e)); });
  }
  Result<void> validate_commands(const ResourceView& v, const MemoryCommandPlan& p,
                                 const OperandEnvironment& e) const override {
    return protect([&] {
      auto a = authorized(v);
      auto normalized = commands(a, p);
      take(c_->model.validate_commands(a, normalized, e));
      host_access(a, normalized, e);
    });
  }
  Result<CostPlan> cost_plan(const ResourceView& v, const MemoryCommandPlan& p,
                             const OperandEnvironment& e) const override {
    return protect([&] {
      auto a = authorized(v);
      return calibrated_costs_ ? take(calibrated_costs_->cost_plan(a, commands(a, p), e))
                               : take(c_->model.cost_plan(a, commands(a, p), e));
    });
  }
  Result<MappingCapabilities> mapping_capabilities(const ResourceView& v,
                                                   std::span<const std::string> pools, BackingMode mode) const override {
    return protect([&] {
      const auto a = c_->authorize_shared(v);
      require(!pools.empty(), "no pools for mapping capabilities");
      U q = 0;
      std::optional<std::string> space;
      for (const auto& id : pools) {
        require(has(a->accessible_pools, id), "pool not accessible");
        const auto& p = find(a->graph.pools, id);
        require(p.address_space.has_value(), "unknown pool address space");
        if (space)
          require(space == p.address_space, "mixed address spaces");
        space = p.address_space;
        q = std::max(q, quantum(p));
      }
      require(mode == BackingMode::private_memory || mode == BackingMode::shareable_remappable ||
                mode == BackingMode::private_movable, "invalid backing mode");
      if (mode == BackingMode::private_movable) {
        const bool available = pools.size() == 1 && LinuxPageStore::private_movable_supported();
        return MappingCapabilities{q, available, false, available, available, false, {}, {}, {}, false};
      }
      return MappingCapabilities{q, true, mode == BackingMode::shareable_remappable, true, true, true, {}, {}, {}};
    });
  }
  Result<ResolvedPlacement> resolve_placement(const ResourceView& v, const PlacementPlan& p, U bytes, BackingMode mode,
                                              std::span<const AccessMode> access) const override {
    return protect([&] {
      const auto a = c_->authorize_shared(v);
      require(mode == BackingMode::private_memory || mode == BackingMode::shareable_remappable || mode == BackingMode::private_movable, "invalid backing mode");
      if (mode == BackingMode::private_movable) require(LinuxPageStore::private_movable_supported(), "private_movable unsupported by this Linux kernel");
      for (auto m : access)
        require(m == AccessMode::read || m == AccessMode::write, "invalid access mode");
      auto resolved = take(c_->model.resolve_placement(*a, p, bytes));
      if (mode == BackingMode::private_movable)
        require(resolved.segments.size() == 1 && resolved.segments.front().offset_bytes == 0 &&
                  resolved.segments.front().size_bytes == add(bytes, (resolved.page_bytes - bytes % resolved.page_bytes) % resolved.page_bytes), "private_movable requires one contiguous pool placement");
      for (const auto& s : resolved.segments)
        require(has(a->allocatable_pools, s.pool), "placement pool not allocatable");
      return resolved;
    });
  }
  Result<bool> compatible_overlap(const ResolvedPlacement& a, U ao, const ResolvedPlacement& b, U bo,
                                  U bytes) const override {
    return c_->model.compatible_overlap(a, ao, b, bo, bytes);
  }
  Result<Allocation> allocate(const ResourceView& input, const BackingRequest& r) override {
    return protect([&] {
      auto v = authorized(input);
      require(r.owner == participant() && has(v.allocatable_pools, r.pool), "allocation owner/pool not authorized");
      const auto& p = find(v.graph.pools, r.pool);
      require(r.page_bytes >= quantum(p), "allocation page quantum below pool quantum");
      std::lock_guard lock(c_->admission);
      auto quota = c_->reserve_pools({r});
      auto grant = c_->reserve_capacity({{p.capacity_group, r.bytes}});
      auto s = std::make_shared<AllocationStorage>();
      s->backing = c_->materialize(r, *grant, *quota, CapacityState::live);
      s->reservation = take(c_->pages.reserve(r.bytes, r.page_bytes));
      s->mapping = take(c_->pages.map(*s->reservation, s->backing, {0, 0, r.bytes}));
      auto info = s->backing->info();
      info.generation = c_->serial();
      return AllocationAccess::make(std::move(info), std::move(s), events_);
    });
  }
  Result<std::unique_ptr<PlanLease>> reserve(const ResourceView& input, const MemoryRecipe& r, U epoch,
                                             std::span<MemoryArena* const> list = {}) override {
    return protect([&]() -> std::unique_ptr<PlanLease> {
      auto v = authorized(input);
      take(validate_recipe(v, r));
      std::lock_guard lock(c_->admission);
      require(c_->snapshot(v).epoch == epoch, "stale budget snapshot");
      const auto shared_epoch = c_->authority ? std::optional<U>(c_->snapshot_shared_epoch) : std::nullopt;
      auto selected = caches(list);
      auto lease = std::make_unique<Lease>();
      lease->issuer = c_;
      lease->canonical = take(recipe_to_toml(r));
      std::vector<BackingRequest> fresh;
      std::map<std::string, U> need;
      for (const auto& req : r.backings)
        if (req.owner == participant()) {
          bool reused = false;
          for (auto* cache : selected)
            if (auto b = cache->claim(req)) {
              lease->cached.emplace(req.id, std::move(*b));
              reused = true;
              break;
            }
          if (!reused) {
            fresh.push_back(req);
            const auto& p = find(v.graph.pools, req.pool);
            need[p.capacity_group] = add(need[p.capacity_group], req.bytes);
          }
        }
      lease->quota = c_->reserve_pools(fresh);
      if (!need.empty())
        lease->grant = c_->reserve_capacity(need, shared_epoch);
      return lease;
    });
  }
  Result<std::shared_ptr<BackingSet>> acquire_backing(const ResourceView& input, const MemoryRecipe& r,
                                                      PlanLease& raw) override {
    return protect([&]() -> std::shared_ptr<BackingSet> {
      auto v = authorized(input);
      take(validate_recipe(v, r));
      std::lock_guard lock(c_->admission);
      auto* lease = dynamic_cast<Lease*>(&raw);
      require(lease && lease->issuer == c_ && !lease->consumed && lease->canonical == take(recipe_to_toml(r)),
              "foreign/consumed/mismatched plan lease");
      lease->consumed = true;
      auto b = std::make_shared<Backings>();
      b->issuer = c_;
      b->canonical = lease->canonical;
      for (const auto& req : r.backings)
        if (req.owner == participant()) {
          std::shared_ptr<SharedBacking> object;
          if (auto i = lease->cached.find(req.id); i != lease->cached.end()) {
            object = i->second.backing;
            i->second.activate();
          } else {
            require(lease->grant != nullptr, "missing plan grant");
            object = c_->materialize(req, *lease->grant, *lease->quota, CapacityState::live);
          }
          b->owned.push_back(object->info());
          b->objects.emplace(req.id, std::move(object));
        }
      lease->cached.clear();
      lease->grant.reset();
      lease->quota.reset();
      return b;
    });
  }
  Result<std::shared_ptr<ImportedMemory>> import_memory(const ResourceView& input,
                                                        const ExternalMemoryView& e) override {
    return protect([&]() -> std::shared_ptr<ImportedMemory> {
      auto v = authorized(input);
      if (e.known_pool) {
        require(has(v.accessible_pools, *e.known_pool), "external pool not accessible");
        const auto& p = find(v.graph.pools, *e.known_pool);
        require(p.address_space && *p.address_space == e.address.address_space, "external pool address space mismatch");
      }
      take(LinuxPageStore::describe_external(e, "import-check", 0));
      auto p = std::make_shared<Imported>();
      p->external = e;
      return p;
    });
  }
  Result<std::shared_ptr<PreparedAddresses>> prepare_addresses(const ResourceView& input,
                                                               const AddressLayout& layout) override {
    return protect([&]() -> std::shared_ptr<PreparedAddresses> {
      auto v = authorized(input);
      auto p = std::make_shared<Prepared>();
      p->issuer = c_;
      p->layout = layout;
      std::set<std::string> slots;
      for (const auto& r : layout.regions) {
        require(!r.id.empty() && !p->regions.contains(r.id), "duplicate/empty region");
        bool space = false;
        for (const auto& pool : v.graph.pools)
          if (pool.address_space && *pool.address_space == r.address_space)
            space = true;
        require(space, "unknown region address space");
        p->regions.emplace(r.id, take(c_->pages.reserve(r.bytes, r.page_bytes, r.base_alignment_bytes)));
      }
      for (const auto& s : layout.slots) {
        require(!s.id.empty() && slots.insert(s.id).second, "duplicate/empty slot");
        const auto& r = find(layout.regions, s.region);
        require(s.size_bytes && s.offset_bytes <= r.bytes && s.size_bytes <= r.bytes - s.offset_bytes,
                "invalid address slot range");
        require(s.placement.address_space == r.address_space && s.placement.page_bytes == r.page_bytes,
                "slot page/address contract mismatch");
      }
      return p;
    });
  }
  Result<std::unique_ptr<LayoutAgreement>> agree_layout(const DistributedLayout& l) override {
    return protect([&]() -> std::unique_ptr<LayoutAgreement> {
      auto canonical = take(layout_to_toml(l));
      require(std::any_of(l.participants.begin(), l.participants.end(),
                          [&](const auto& p) { return p.participant == participant(); }),
              "participant absent from distributed layout");
      // Global validation sees physical topology; each participant's own admission
      // remains restricted by its actual scope when it materializes backing.
      Scope all{};
      for (const auto& a : c_->model.graph().agents)
        all.executable_agents.push_back(a.id);
      for (const auto& p : c_->model.graph().pools) {
        all.allocatable_pools.push_back(p.id);
        all.accessible_pools.push_back(p.id);
      }
      auto full = take(c_->model.view(all, 1));
      std::map<std::string, U> group_bytes;
      for (const auto& req : l.backings) {
        const auto& p = find(full.graph.pools, req.pool);
        group_bytes[p.capacity_group] = add(group_bytes[p.capacity_group], req.bytes);
      }
      for (const auto& [id, n] : group_bytes) {
        const auto& g = find(full.graph.capacity_groups, id);
        require(!g.capacity_bytes || n <= *g.capacity_bytes, "distributed backing exceeds shared capacity");
      }
      for (const auto& p : l.participants)
        take(c_->model.validate_recipe(full, project(l, p.participant)));
      if (l.participants.size() > 1) {
        require(c_->options.control != nullptr, "distributed layout requires a control adapter");
        take(c_->options.control->agree(l, canonical));
      }
      auto a = std::make_unique<Agreement>();
      a->issuer = c_;
      a->layout = l;
      a->canonical = canonical;
      {
        std::lock_guard lock(mutex_);
        const auto key = l.id + "/" + std::to_string(l.generation);
        auto i = layouts_.find(key);
        if (i != layouts_.end())
          require(take(layout_to_toml(i->second)) == canonical, "conflicting layout generation");
        else
          layouts_.emplace(key, l);
      }
      return a;
    });
  }
  Result<std::shared_ptr<BackingSet>> exchange_backing(const LayoutAgreement& raw,
                                                       std::shared_ptr<BackingSet> owned) override {
    return protect([&]() -> std::shared_ptr<BackingSet> {
      auto* a = dynamic_cast<const Agreement*>(&raw);
      auto b = std::dynamic_pointer_cast<Backings>(owned);
      require(a && a->issuer == c_ && b && b->issuer == c_, "foreign agreement/backings");
      require(b->canonical == take(recipe_to_toml(project(a->layout, participant()))), "backings disagree with layout");
      if (a->layout.participants.size() == 1)
        return owned;
      std::map<std::string, LinuxBackingHandle> handles;
      for (const auto& [id, backing] : b->objects) {
        auto h = take(backing->export_handle());
        require(h.type() == typeid(LinuxBackingHandle), "non-Linux backing export");
        handles.emplace(id, std::any_cast<LinuxBackingHandle>(std::move(h)));
      }
      auto received = take(c_->options.control->exchange(a->layout, handles));
      auto out = std::make_shared<Backings>(*b);
      const auto r = project(a->layout, participant());
      for (const auto& req : r.backings)
        if (req.owner != participant()) {
          auto i = received.find(req.id);
          require(i != received.end(), "missing foreign backing export");
          auto object = take(c_->pages.import_backing(i->second));
          const auto& info = object->info();
          require(info.id == req.id && info.mapped_bytes == req.bytes && info.page_bytes == req.page_bytes &&
                    info.segments.size() == 1 && info.segments.front().pool == req.pool,
                  "foreign backing contract mismatch");
          out->objects.emplace(req.id, std::move(object));
        }
      return out;
    });
  }
  Result<std::unique_ptr<MemoryBinding>> bind(const ResourceView& input, const MemoryRecipe& r,
                                              std::shared_ptr<PreparedAddresses> prepared,
                                              std::shared_ptr<BackingSet> backing,
                                              const LayoutAgreement* raw = nullptr) override {
    return protect([&]() -> std::unique_ptr<MemoryBinding> {
      auto v = authorized(input);
      take(validate_recipe(v, r));
      auto p = std::dynamic_pointer_cast<Prepared>(prepared);
      auto b = std::dynamic_pointer_cast<Backings>(backing);
      require(p && p->issuer == c_ && b && b->issuer == c_ && b->canonical == take(recipe_to_toml(r)),
              "foreign/mismatched addresses or backing");
      compatible(r.addresses, p->layout);
      if (r.layout_id) {
        auto* a = dynamic_cast<const Agreement*>(raw);
        require(a && a->issuer == c_ && a->layout.id == *r.layout_id && a->layout.generation == r.layout_generation &&
                  take(recipe_to_toml(project(a->layout, participant()))) == b->canonical,
                "layout agreement mismatch");
      } else
        require(raw == nullptr, "unexpected distributed agreement");
      auto storage = std::make_shared<BindingStorage>();
      storage->backings = b;
      {
        std::lock_guard lock(p->mutex);
        require(!p->bound, "prepared addresses already bound");
        p->bound = true;
        storage->prepared = p;
      }
      for (const auto& m : r.mappings) {
        auto i = b->objects.find(m.backing);
        require(i != b->objects.end(), "backing not acquired/exchanged");
        storage->mappings.push_back(
          take(c_->pages.map(*p->regions.at(m.region), i->second,
                             {m.backing_offset_bytes, m.virtual_offset_bytes, m.size_bytes}, m.access)));
      }
      auto bound = std::make_unique<Binding>();
      bound->core = c_;
      bound->events = events_;
      bound->recipe = r;
      bound->version = c_->serial();
      bound->storage = std::move(storage);
      return bound;
    });
  }
  Result<QuiescenceToken> quiesce(MemoryBinding& raw, std::span<const CompletionToken> tokens) override {
    return protect([&] {
      auto& b = binding(raw);
      std::vector<std::shared_ptr<HostEventReference>> all;
      for (const auto& token : tokens) {
        require(token.binding_generation == b.version, "foreign quiescence generation");
        all.push_back(take(events_->retain_event(token)));
      }
      U epoch;
      {
        std::lock_guard lock(b.mutex);
        require(b.storage && b.active == 0, "binding is unbound or submitting");
        epoch = b.epoch;
        all.insert(all.end(), b.uses.begin(), b.uses.end());
      }
      for (const auto& t : all) {
        while (!stopped(take(t->wait())))
          std::this_thread::yield();
      }
      {
        std::lock_guard lock(b.mutex);
        require(b.storage && b.active == 0 && epoch == b.epoch, "binding changed during quiescence");
      }
      auto id = c_->id("binding-quiescence");
      std::lock_guard lock(mutex_);
      proofs_.emplace(id, FenceProof{&b, b.version, epoch});
      return QuiescenceToken{id};
    });
  }
  Result<void> unbind(MemoryBinding& raw, const QuiescenceToken& token) override {
    return protect([&] {
      auto& b = binding(raw);
      std::shared_ptr<BindingStorage> storage;
      {
        std::scoped_lock lock(mutex_, b.mutex);
        auto p = proofs_.find(token.id);
        require(p != proofs_.end() && p->second.object == &b && p->second.generation == b.version &&
                  p->second.epoch == b.epoch && b.active == 0 && b.storage,
                "invalid/stale quiescence proof");
        proofs_.erase(p);
        storage = std::move(b.storage);
        b.uses.clear();
      }
      storage.reset();
    });
  }
  Result<CompletionToken> shared(SharedOperation op, const SharedRange& range,
                                 std::span<const CompletionToken> dependencies) {
    return protect([&] {
      DistributedLayout l;
      {
        std::lock_guard lock(mutex_);
        bool found = false;
        for (const auto& [id, layout] : layouts_) {
          (void)id;
          if (layout.generation == range.layout_generation &&
              std::any_of(layout.backings.begin(), layout.backings.end(),
                          [&](const auto& b) { return b.id == range.backing; })) {
            require(!found, "ambiguous shared backing generation");
            l = layout;
            found = true;
          }
        }
        require(found, "unknown shared range");
      }
      const auto& b = find(l.backings, range.backing);
      const auto& contract = find(l.sharing_contracts, range.sharing_contract);
      require(has(contract.participants, participant()), "participant outside shared contract");
      if (op == SharedOperation::begin || op == SharedOperation::publish)
        require(has(contract.producers, participant()), "shared operation requires a producer");
      if (op == SharedOperation::acquire)
        require(has(contract.consumers, participant()), "shared acquire requires a consumer");
      require(range.size_bytes && range.offset_bytes <= b.bytes && range.size_bytes <= b.bytes - range.offset_bytes,
              "shared range outside backing");
      if (l.participants.size() > 1)
        return take(c_->options.control->shared(op, range, dependencies, *events_));
      InvocationContext context{{c_->id("shared"), range.layout_generation}, {}, {}};
      const SharedKey key{range.backing, range.offset_bytes, range.size_bytes, range.sharing_contract,
                          range.layout_generation};
      if (op == SharedOperation::begin) {
        std::vector<std::shared_ptr<HostEventReference>> retirements;
        U revision;
        {
          std::lock_guard lock(mutex_);
          revision = shared_revision_;
          for (const auto& [previous_key, previous] : shared_) {
            const auto& [backing, offset, bytes, contract, generation] = previous_key;
            (void)contract;
            if (backing == range.backing && generation == range.layout_generation &&
                offset < add(range.offset_bytes, range.size_bytes) && range.offset_bytes < add(offset, bytes)) {
              require(range.data_epoch > previous->epoch && previous->phase == SharedEpoch::retiring,
                      "overlapping shared epoch still in use or not increasing");
              retirements.push_back(previous->retirement);
            }
          }
        }
        for (const auto& retirement : retirements)
          require(stopped(take(retirement->query())), "overlapping shared epoch has active accesses");
        std::lock_guard lock(mutex_);
        require(shared_revision_ == revision, "concurrent shared begin");
        auto next = std::make_shared<SharedEpoch>();
        next->epoch = range.data_epoch;
        auto token = take(events_->complete(context.stamp, {EventState::succeeded, true, {}}));
        shared_[key] = std::move(next);
        ++shared_revision_;
        return token;
      }
      std::lock_guard lock(mutex_);
      auto it = shared_.find(key);
      require(it != shared_.end() && it->second->epoch == range.data_epoch, "shared epoch has not begun");
      auto& epoch = *it->second;
      if (op == SharedOperation::publish) {
        require(epoch.phase == SharedEpoch::begun && dependencies.size() == 1, "shared epoch already published");
        auto token = take(events_->join(dependencies, context));
        epoch.publication = take(events_->retain_event(token));
        epoch.phase = SharedEpoch::published;
        return token;
      }
      require(epoch.phase == SharedEpoch::published, "shared epoch is not published or already retiring");
      if (op == SharedOperation::acquire)
        return reference_event({epoch.publication}, context);
      require(op == SharedOperation::retire, "invalid shared operation");
      std::vector<std::shared_ptr<HostEventReference>> refs{epoch.publication};
      for (const auto& dependency : dependencies)
        refs.push_back(take(events_->retain_event(dependency)));
      auto token = reference_event(std::move(refs), context);
      epoch.retirement = take(events_->retain_event(token));
      epoch.phase = SharedEpoch::retiring;
      return token;
    });
  }
  Result<CompletionToken> begin_shared(const SharedRange& r) override {
    return shared(SharedOperation::begin, r, {});
  }
  Result<CompletionToken> publish_shared(const SharedRange& r, CompletionToken t) override {
    return shared(SharedOperation::publish, r, {&t, 1});
  }
  Result<CompletionToken> acquire_shared(const SharedRange& r) override {
    return shared(SharedOperation::acquire, r, {});
  }
  Result<CompletionToken> retire_shared(const SharedRange& r, std::span<const CompletionToken> t) override {
    return shared(SharedOperation::retire, r, t);
  }
  Result<std::unique_ptr<VirtualReservation>> reserve_address(const std::string& space, U bytes, U q) override {
    return protect([&] {
      bool known = false;
      for (const auto& p : c_->model.graph().pools)
        if (p.address_space && *p.address_space == space)
          known = true;
      require(known, "unknown address space");
      return take(c_->pages.reserve(bytes, q));
    });
  }
  Result<std::unique_ptr<MappedRange>> map(VirtualReservation& va, std::shared_ptr<SharedBacking> backing,
                                           const MapRequest& r) override {
    return protect([&]() -> std::unique_ptr<MappedRange> {
      auto m = std::make_unique<RuntimeMapping>();
      m->issuer = c_;
      m->version = c_->serial();
      m->native = take(c_->pages.map(va, std::move(backing), r));
      return m;
    });
  }
  Result<QuiescenceToken> quiesce(MappedRange& raw, std::span<const CompletionToken> tokens) override {
    return protect([&] {
      auto* m = dynamic_cast<RuntimeMapping*>(&raw);
      require(m && m->issuer == c_, "foreign mapping");
      U version, epoch;
      {
        std::lock_guard lock(m->mutex);
        version = m->version;
        epoch = m->epoch;
      }
      for (const auto& t : tokens) {
        require(t.binding_generation == version, "stale mapped-range event");
        wait_stopped(*events_, t);
      }
      auto id = c_->id("mapping-quiescence");
      {
        std::scoped_lock lock(mutex_, m->mutex);
        require(version == m->version && epoch == m->epoch, "mapping changed during quiescence");
        proofs_.emplace(id, FenceProof{m, version, epoch});
      }
      return QuiescenceToken{id};
    });
  }
  Result<void> rebind(MappedRange& raw, std::shared_ptr<SharedBacking> backing, const MapRequest& r,
                      const QuiescenceToken& token) override {
    return protect([&] {
      auto* m = dynamic_cast<RuntimeMapping*>(&raw);
      require(m && m->issuer == c_, "foreign mapping");
      std::scoped_lock lock(mutex_, m->mutex);
      auto p = proofs_.find(token.id);
      require(p != proofs_.end() && p->second.object == m && p->second.generation == m->version &&
                p->second.epoch == m->epoch,
              "invalid mapping quiescence proof");
      take(c_->pages.replace_quiescent(*m->native, std::move(backing), r));
      proofs_.erase(p);
      m->version = c_->serial();
      ++m->epoch;
    });
  }
  Result<OperandEnvironment> resolve_operands(const ResourceView& input, MemoryBinding& raw,
                                              const InvocationContext& context) const override {
    return protect([&] {
      auto v = authorized(input);
      auto& b = binding(raw);
      require(!context.stamp.id.empty() && context.stamp.binding_generation == b.version,
              "stale/empty invocation stamp");
      std::shared_ptr<BindingStorage> storage;
      {
        std::lock_guard lock(b.mutex);
        require(b.storage != nullptr, "binding is unbound");
        storage = b.storage;
      }
      auto env = take(c_->model.describe_operands(v, b.recipe));
      env.invocation = context.stamp;
      // Offline recipe identity is replaced by backend-certified inode + file offset.
      for (auto& d : env.memories) {
        d.generation = b.version;
        const auto& slot = find(b.recipe.addresses.slots, d.symbol);
        for (auto& span : d.spans) {
          U virtual_offset = add(slot.offset_bytes, span.offset_bytes);
          auto m = std::find_if(b.recipe.mappings.begin(), b.recipe.mappings.end(), [&](const auto& m) {
            return m.region == slot.region && virtual_offset >= m.virtual_offset_bytes &&
                   virtual_offset - m.virtual_offset_bytes < m.size_bytes;
          });
          require(m != b.recipe.mappings.end(), "missing resolved page mapping");
          auto backing = storage->backings->objects.at(m->backing);
          if (backing->info().mode == BackingMode::private_movable) {
            // Root accounting identity survives moves; byte identity must match
            // imported same-VA private aliases, not the synthetic root receipt.
            ExternalMemoryView external{};
            external.address = {d.address_space, reinterpret_cast<U>(take(LinuxPageStore::address(*storage->prepared->regions.at(slot.region)))) + virtual_offset};
            external.bytes = span.size_bytes;
            external.owner = storage;
            auto actual = take(LinuxPageStore::describe_external(external, d.symbol, b.version));
            require(actual.spans.size() == 1, "private_movable span crosses unexpected VMA boundary");
            span.physical_identity = actual.spans.front().physical_identity;
            span.physical_offset_bytes = actual.spans.front().physical_offset_bytes;
            continue;
          }
          span.physical_identity = take(LinuxPageStore::physical_identity(*backing));
          span.physical_offset_bytes = add(take(LinuxPageStore::physical_offset(*backing)),
                                           add(m->backing_offset_bytes, virtual_offset - m->virtual_offset_bytes));
        }
      }
      std::set<std::string> symbols;
      for (const auto& e : context.external) {
        require(e.memory && symbols.insert(e.symbol).second, "null/duplicate external binding");
        auto imported = std::dynamic_pointer_cast<Imported>(e.memory);
        require(imported != nullptr, "foreign external memory implementation");
        auto d = take(LinuxPageStore::describe_external(imported->external, e.symbol, b.version));
        env.memories.push_back(std::move(d));
      }
      return env;
    });
  }
  Result<Submission> submit(const ResourceView& input, const MemoryCommandPlan& p, MemoryBinding& raw,
                            const InvocationContext& context) override {
    return protect([&] {
      auto v = authorized(input);
      auto& b = binding(raw);
      std::shared_ptr<BindingStorage> storage;
      {
        std::lock_guard lock(b.mutex);
        require(b.storage != nullptr, "binding is unbound");
        storage = b.storage;
        ++b.active;
        ++b.epoch;
      }
      struct Active {
        Binding& b;
        ~Active() {
          std::lock_guard lock(b.mutex);
          --b.active;
        }
      } active{b};
      require(context.stamp.binding_generation == b.version, "stale submission binding generation");
      auto env = take(resolve_operands(v, b, context));
      auto normalized = commands(v, p);
      take(c_->model.validate_commands(v, normalized, env));
      host_access(v, normalized, env);
      std::map<std::string, CompletionToken> ports;
      for (const auto& e : context.events) {
        require(!e.port.empty() && ports.emplace(e.port, e.event).second, "duplicate event port");
        take(events_->query(e.event));
      }
      for (const auto& cmd : p.commands)
        for (const auto& port : cmd.wait_ports)
          require(ports.contains(port), "unbound command event port");
      auto pointer = [&](const ByteRangeRef& r) {
        if (r.kind == MemoryRefKind::slot)
          return static_cast<unsigned char*>(take(b.slot(r.memory))) + r.offset_bytes;
        auto e = std::find_if(context.external.begin(), context.external.end(),
                              [&](const auto& e) { return e.symbol == r.memory; });
        require(e != context.external.end(), "missing external operand");
        return reinterpret_cast<unsigned char*>(e->memory->view().address.value) + r.offset_bytes;
      };
      // Mint/record the completion before accepting any device descriptors.
      // The state owns all external imports as well as the binding mappings.
      struct ExecutionState {
        std::mutex mutex;
        EventStatus status{EventState::pending, false, {}};
        std::shared_ptr<LinuxTransferOperation> operation;
        std::shared_ptr<BindingStorage> storage;
        std::vector<ExternalBinding> external;
        EventStatus poll() {
          std::lock_guard lock(mutex);
          if (operation) {
            auto r = operation->query();
            if (!r) {
              status = {EventState::failed, false, std::get<Error>(r.state)};
            } else if (stopped(r.value())) {
              status.accesses_stopped = true;
              if (status.state == EventState::pending) status = r.value();
              operation.reset();
            } else if (r.value().state == EventState::failed) status = r.value();
          }
          return status;
        }
      };
      auto state = std::make_shared<ExecutionState>();
      state->storage = storage;
      state->external = context.external;
      HostEventCallbacks callbacks;
      callbacks.query = [state]() -> Result<EventStatus> { return {state->poll()}; };
      callbacks.wait = callbacks.query;
      auto token = take(events_->import_event({"host-callback", callbacks, state}, context));
      auto recorded = b.record_use(token);
      if (!recorded) {
        { std::lock_guard lock(state->mutex); state->status = {EventState::failed, true, *recorded.error}; }
        take(events_->query(token));
        take(events_->release(std::span<const CompletionToken>(&token, 1)));
        take(recorded);
      }
      Submission out{{}, {}, token, token, token};
      out.accepted_commands.reserve(p.commands.size());
      std::set<std::string> done;
      try {
        while (done.size() < p.commands.size())
          for (const auto& cmd : p.commands)
            if (!done.contains(cmd.id) && std::all_of(cmd.dependencies.begin(), cmd.dependencies.end(),
                                                      [&](const auto& id) { return done.contains(id); })) {
              for (const auto& port : cmd.wait_ports) {
                auto s = take(events_->wait(ports.at(port)));
                while (!stopped(s))
                  s = take(events_->wait(ports.at(port)));
                if (s.state != EventState::succeeded)
                  throw Failure{
                    s.error.value_or(Error{"dependency_failed", "memory command dependency did not succeed"})};
              }
              out.accepted_commands.push_back(cmd.id);
              if (cmd.opcode == MemoryOpcode::copy) {
                if (const auto* device = device_queue(cmd)) {
                  auto operation = take(device->executor->copy(pointer(*cmd.target), pointer(*cmd.source),
                                                               cmd.target->size_bytes, state->storage));
                  { std::lock_guard lock(state->mutex); state->operation = operation; }
                  const auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(c_->options.transfer_wait_ms);
                  for (;;) {
                    auto current = take(operation->query());
                    if (stopped(current)) {
                      { std::lock_guard lock(state->mutex); state->operation.reset(); }
                      if (current.state != EventState::succeeded)
                        throw Failure{current.error.value_or(Error{"sdma_error", "device copy failed"})};
                      break;
                    }
                    if (current.state == EventState::failed || std::chrono::steady_clock::now() >= deadline)
                      throw Failure{current.error.value_or(Error{"sdma_quiescence_unknown",
                                                               "device completion deadline; accesses not stopped"})};
                    std::this_thread::yield();
                  }
                } else std::memcpy(pointer(*cmd.target), pointer(*cmd.source), cmd.target->size_bytes);
              } else if (cmd.opcode == MemoryOpcode::fill)
                std::memset(pointer(*cmd.target), static_cast<int>(cmd.fill_byte), cmd.target->size_bytes);
              done.insert(cmd.id);
            }
        std::lock_guard lock(state->mutex);
        state->status = {EventState::succeeded, true, {}};
      } catch (const Failure& e) {
        out.submission_error = e.error;
        std::lock_guard lock(state->mutex);
        state->status = {EventState::failed, !state->operation, e.error};
        if (state->operation) {
          state->operation->stop_submitting(*out.submission_error);
          quarantine_transfer(events_, token, state->operation);
        }
      } catch (const std::exception& e) {
        out.submission_error = Error{"execution_failed", e.what()};
        std::lock_guard lock(state->mutex);
        state->status = {EventState::failed, !state->operation, out.submission_error};
        if (state->operation) {
          state->operation->stop_submitting(*out.submission_error);
          quarantine_transfer(events_, token, state->operation);
        }
      }
      take(events_->query(token));
      return out;
    });
  }
};
} // namespace
Result<std::unique_ptr<MemoryRuntime>> make_linux_runtime(LinuxDiscovery d, LinuxRuntimeOptions o) {
  return details::protect([&]() -> std::unique_ptr<MemoryRuntime> {
    return std::make_unique<Runtime>(std::make_shared<Core>(std::move(d), std::move(o)));
  });
}
} // namespace vecops::memory::model
