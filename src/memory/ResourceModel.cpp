// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/ResourceModel.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
constexpr double gib = 1073741824.0;
void require(bool ok, const char* message) {
  if (!ok)
    throw std::invalid_argument(message);
}
U add(U a, U b) {
  if (b > std::numeric_limits<U>::max() - a)
    throw std::overflow_error("byte range overflow");
  return a + b;
}
bool power2(U x) {
  return x && !(x & (x - 1));
}
void range(U offset, U size, U capacity) {
  require(size && add(offset, size) <= capacity, "byte range outside storage");
}
template <class F>
auto checked(F&& f) -> Result<decltype(f())> {
  using T = decltype(f());
  try {
    if constexpr (std::is_void_v<T>) {
      f();
      return {};
    } else
      return {f()};
  } catch (const std::invalid_argument& e) {
    return {Error{"invalid_argument", e.what()}};
  } catch (const std::overflow_error& e) {
    return {Error{"overflow", e.what()}};
  }
}
template <class T>
auto index(const std::vector<T>& objects) {
  std::map<std::string, const T*> result;
  for (const auto& o : objects)
    require(!o.id.empty() && result.emplace(o.id, &o).second, "duplicate or empty identity");
  return result;
}
template <class Map>
const auto& at(const Map& m, const std::string& k) {
  const auto it = m.find(k);
  require(it != m.end(), "unknown resource reference");
  return *it->second;
}
template <class T>
bool contains(const std::vector<T>& v, const T& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}
void unique(const std::vector<std::string>& v) {
  std::set<std::string> seen;
  for (const auto& x : v)
    require(!x.empty() && seen.insert(x).second, "duplicate or empty selection");
}
void rights(const std::vector<AccessMode>& v) {
  require(!v.empty() && v.size() <= 2, "invalid access rights");
  for (auto mode : v)
    require(mode == AccessMode::read || mode == AccessMode::write, "unknown access right");
  if (v.size() == 2)
    require(v[0] != v[1], "duplicate access right");
}
U quantum(const MemoryPool& p) {
  U q = p.granularity_bytes.value_or(1);
  require(power2(q), "invalid pool granularity");
  require(!p.page_sizes_bytes.empty(), "pool has no page sizes");
  for (auto n : p.page_sizes_bytes) {
    require(power2(n), "invalid pool page size");
    q = std::max(q, n);
  }
  return q;
}
void positive(std::optional<double> x) {
  if (x)
    require(std::isfinite(*x) && *x > 0, "invalid bandwidth");
}
void demand_valid(const ResourceDemand& d, const std::map<std::string, const Resource*>& resources) {
  const auto& r = at(resources, d.resource);
  require(d.service == DemandKind::read || d.service == DemandKind::write || d.service == DemandKind::traffic ||
            d.service == DemandKind::occupancy,
          "unknown demand service");
  require(std::isfinite(d.amount) && d.amount >= 0, "invalid resource demand");
  if (d.service == DemandKind::occupancy)
    require(r.kind == ResourceKind::execution_slots && d.amount >= 1 && std::floor(d.amount) == d.amount &&
              !d.per_invocation_bytes,
            "invalid slot demand");
  else
    require(r.kind == ResourceKind::bandwidth, "traffic requires bandwidth resource");
}
void validate_graph(const ResourceGraph& g) {
  const auto agents = index(g.agents);
  const auto pools = index(g.pools);
  const auto resources = index(g.resources);
  const auto groups = index(g.capacity_groups);
  index(g.domains);
  index(g.access);
  index(g.methods);
  for (const auto& p : g.pools) {
    at(groups, p.capacity_group);
    quantum(p);
    for (const auto& s : {p.read_service, p.write_service})
      if (s)
        require(at(resources, *s).kind == ResourceKind::bandwidth, "invalid pool service");
  }
  for (const auto& r : g.resources) {
    require(r.sharing == SharingMode::shared || r.sharing == SharingMode::read_write, "unknown sharing mode");
    require(r.kind == ResourceKind::bandwidth || r.kind == ResourceKind::execution_slots, "unknown resource kind");
    positive(r.bandwidth);
    positive(r.read);
    positive(r.write);
    if (r.kind == ResourceKind::execution_slots)
      require(r.slots && *r.slots, "missing execution slots");
  }
  for (const auto& a : g.access) {
    at(agents, a.agent);
    at(pools, a.pool);
    rights(a.access);
    if (a.bandwidth)
      for (const auto& [key, v] : *a.bandwidth) {
        require(key == "read" || key == "write", "unknown access bandwidth key");
        positive(v);
      }
    if (a.latency_ns)
      require(std::isfinite(*a.latency_ns) && *a.latency_ns >= 0, "invalid latency");
    for (const auto& s : {a.read_service, a.write_service})
      if (s)
        at(resources, *s);
    for (const auto& s : a.read_fabric)
      at(resources, s);
    for (const auto& s : a.write_fabric)
      at(resources, s);
  }
  for (const auto& d : g.domains) {
    unique(d.members);
    require(!d.members.empty(), "empty allocation domain");
    for (const auto& p : d.members)
      at(pools, p);
  }
  for (const auto& m : g.methods) {
    positive(m.bandwidth);
    if (m.startup_us)
      require(std::isfinite(*m.startup_us) && *m.startup_us >= 0, "invalid method startup");
    index(m.routes);
    for (const auto& r : m.routes) {
      at(pools, r.source);
      at(pools, r.target);
      if (r.agent)
        at(agents, *r.agent);
      if (r.controller)
        at(agents, *r.controller);
      for (const auto& d : r.demands)
        demand_valid(d, resources);
    }
  }
}
void placement_valid(const ResolvedPlacement& p, U bytes) {
  require(!p.key.empty() && !p.address_space.empty() && power2(p.page_bytes) && bytes && bytes % p.page_bytes == 0,
          "invalid resolved placement");
  U end = 0;
  for (const auto& s : p.segments) {
    require(!s.pool.empty() && s.offset_bytes == end && s.size_bytes && s.size_bytes % p.page_bytes == 0,
            "placement gap/overlap/alignment");
    end = add(end, s.size_bytes);
    require(end <= bytes, "placement exceeds slot");
  }
  require(end == bytes, "incomplete placement");
}
std::vector<std::size_t> order(const std::vector<std::string>& ids, const std::vector<std::vector<std::string>>& deps) {
  require(ids.size() == deps.size(), "inconsistent dependency graph");
  unique(ids);
  std::unordered_map<std::string, std::size_t> by_id;
  for (std::size_t i = 0; i < ids.size(); ++i)
    by_id.emplace(ids[i], i);
  std::vector<std::vector<std::size_t>> next(ids.size());
  std::vector<std::size_t> degree(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    unique(deps[i]);
    degree[i] = deps[i].size();
    for (const auto& d : deps[i]) {
      auto it = by_id.find(d);
      require(it != by_id.end(), "unknown dependency");
      next[it->second].push_back(i);
    }
  }
  std::deque<std::size_t> ready;
  for (std::size_t i = 0; i < ids.size(); ++i)
    if (!degree[i])
      ready.push_back(i);
  std::vector<std::size_t> result;
  result.reserve(ids.size());
  while (!ready.empty()) {
    const auto i = ready.front();
    ready.pop_front();
    result.push_back(i);
    for (auto n : next[i])
      if (!--degree[n])
        ready.push_back(n);
  }
  require(result.size() == ids.size(), "cyclic dependency graph");
  return result;
}
using DescriptorKey = std::pair<MemoryRefKind, std::string>;
using Descriptors = std::map<DescriptorKey, const MemoryDescriptor*>;
Descriptors descriptors(const OperandEnvironment& env) {
  Descriptors out;
  if (env.invocation)
    require(!env.invocation->id.empty(), "empty invocation identity");
  for (const auto& d : env.memories) {
    require(!d.symbol.empty() && !d.address_space.empty() && d.size_bytes &&
              out.emplace(DescriptorKey{d.kind, d.symbol}, &d).second,
            "invalid operand identity");
    if (env.invocation)
      require(d.generation == env.invocation->binding_generation, "stale operand generation");
    U end = 0;
    for (const auto& s : d.spans) {
      rights(s.access);
      require(s.offset_bytes == end && s.size_bytes, "operand span gap/overlap");
      end = add(end, s.size_bytes);
      add(s.physical_offset_bytes, s.size_bytes);
      require(end <= d.size_bytes && (!s.physical_identity || !s.physical_identity->empty()), "invalid physical span");
    }
    require(end == d.size_bytes, "incomplete operand descriptor");
  }
  return out;
}
std::vector<ResolvedMemorySpan> resolve(const ByteRangeRef& ref, const Descriptors& env, AccessMode access) {
  const auto it = env.find({ref.kind, ref.memory});
  require(it != env.end(), "unknown command operand");
  const auto& d = *it->second;
  range(ref.offset_bytes, ref.size_bytes, d.size_bytes);
  const U end = ref.offset_bytes + ref.size_bytes;
  std::vector<ResolvedMemorySpan> out;
  for (const auto& s : d.spans) {
    if (s.offset_bytes >= end)
      break;
    const U a = std::max(s.offset_bytes, ref.offset_bytes), b = std::min(add(s.offset_bytes, s.size_bytes), end);
    if (a >= b)
      continue;
    require(contains(s.access, access), "command access denied");
    auto r = s;
    r.offset_bytes = a - ref.offset_bytes;
    r.size_bytes = b - a;
    r.physical_offset_bytes = add(s.physical_offset_bytes, a - s.offset_bytes);
    out.push_back(std::move(r));
  }
  return out;
}
void nonoverlap(const std::vector<ResolvedMemorySpan>& a, const std::vector<ResolvedMemorySpan>& b) {
  std::map<std::string, std::vector<std::pair<U, U>>> intervals;
  for (const auto& s : a) {
    require(s.physical_identity.has_value(), "unknown source alias relation");
    intervals[*s.physical_identity].emplace_back(s.physical_offset_bytes, add(s.physical_offset_bytes, s.size_bytes));
  }
  for (auto& [id, ranges] : intervals) {
    (void)id;
    std::sort(ranges.begin(), ranges.end());
    U end = 0;
    for (auto& interval : ranges) {
      end = std::max(end, interval.second);
      interval.second = end;
    }
  }
  for (const auto& s : b) {
    require(s.physical_identity.has_value(), "unknown target alias relation");
    const auto it = intervals.find(*s.physical_identity);
    if (it == intervals.end())
      continue;
    const auto stop = std::lower_bound(it->second.begin(), it->second.end(),
                                       std::pair<U, U>{add(s.physical_offset_bytes, s.size_bytes), 0});
    if (stop != it->second.begin())
      require(std::prev(stop)->second <= s.physical_offset_bytes, "copy operands physically overlap");
  }
}
std::string physical_key(const MemoryRecipe& r, const BackingRequest& b) {
  // Length-prefixed tuple encoding has no ambiguity when user IDs contain ':' etc.
  std::string out = "planned:";
  for (const auto& s : {b.owner, r.layout_id.value_or(""), std::to_string(r.layout_generation), b.id})
    out += std::to_string(s.size()) + ":" + s;
  return out;
}
} // namespace

U page_charge(U bytes, U q) {
  require(bytes && power2(q), "invalid page charge");
  return add(bytes, q - 1) / q * q;
}
Result<U> budget_available(const PoolBudget& b) {
  return checked([&] {
    const U used = add(add(b.live_bytes, b.idle_available_bytes), add(b.idle_reserved_bytes, b.retired_bytes));
    require(used == b.committed_bytes, "inconsistent committed budget");
    const U granted = add(b.committed_bytes, b.unbacked_reserved_bytes);
    U headroom = granted < b.hard_limit_bytes ? b.hard_limit_bytes - granted : 0;
    if (b.shared_remaining_bytes)
      headroom = std::min(headroom, *b.shared_remaining_bytes);
    return add(b.idle_available_bytes, std::min(headroom, b.system_available_bytes));
  });
}
ResourceModel::ResourceModel(ResourceGraph g)
  : graph_(std::move(g)) {
  validate_graph(graph_);
}

Result<ResourceView> ResourceModel::view(const Scope& scope, U epoch) const {
  return checked([&] {
    unique(scope.executable_agents);
    unique(scope.allocatable_pools);
    unique(scope.accessible_pools);
    const auto agents = index(graph_.agents);
    const auto pools = index(graph_.pools);
    const std::set<std::string> selected(scope.executable_agents.begin(), scope.executable_agents.end());
    const std::set<std::string> allocate(scope.allocatable_pools.begin(), scope.allocatable_pools.end());
    std::set<std::string> visible(scope.accessible_pools.begin(), scope.accessible_pools.end());
    visible.insert(allocate.begin(), allocate.end());
    for (const auto& a : selected)
      at(agents, a);
    for (const auto& p : visible)
      at(pools, p);
    for (const auto& [a, n] : scope.execution_slots)
      require(selected.contains(a) && n, "invalid scope execution slots");
    ResourceView v{};
    v.epoch = epoch;
    v.executable_agents = scope.executable_agents;
    v.allocatable_pools = scope.allocatable_pools;
    v.accessible_pools.assign(visible.begin(), visible.end());
    v.execution_slots = scope.execution_slots;
    auto& g = v.graph;
    for (const auto& a : graph_.agents)
      if (selected.contains(a.id))
        g.agents.push_back(a);
    std::set<std::string> needed, groups;
    for (const auto& p : graph_.pools)
      if (visible.contains(p.id)) {
        g.pools.push_back(p);
        groups.insert(p.capacity_group);
        if (p.read_service)
          needed.insert(*p.read_service);
        if (p.write_service)
          needed.insert(*p.write_service);
      }
    for (const auto& e : graph_.access)
      if (selected.contains(e.agent) && visible.contains(e.pool)) {
        g.access.push_back(e);
        needed.insert(e.read_fabric.begin(), e.read_fabric.end());
        needed.insert(e.write_fabric.begin(), e.write_fabric.end());
        if (e.read_service)
          needed.insert(*e.read_service);
        if (e.write_service)
          needed.insert(*e.write_service);
      }
    for (const auto& m : graph_.methods) {
      // Copy scalar metadata first: routes outside this scope can be large.
      Method copy{m.id, m.request_executor, m.bandwidth, m.startup_us, m.registration_bits, {}};
      for (const auto& r : m.routes)
        if (visible.contains(r.source) && visible.contains(r.target) && (!r.agent || selected.contains(*r.agent)) &&
            (!r.controller || selected.contains(*r.controller))) {
          copy.routes.push_back(r);
          for (const auto& d : r.demands)
            needed.insert(d.resource);
        }
      if (!copy.routes.empty())
        g.methods.push_back(std::move(copy));
    }
    for (const auto& d : graph_.domains) {
      AllocationDomain copy{d.id, d.rule, {}};
      for (const auto& p : d.members)
        if (allocate.contains(p))
          copy.members.push_back(p);
      if (!copy.members.empty())
        g.domains.push_back(std::move(copy));
    }
    for (const auto& r : graph_.resources)
      if (needed.contains(r.id))
        g.resources.push_back(r);
    for (const auto& c : graph_.capacity_groups)
      if (groups.contains(c.id))
        g.capacity_groups.push_back(c);
    return v;
  });
}

Result<ResolvedPlacement> ResourceModel::resolve_placement(const ResourceView& v, const PlacementPlan& p,
                                                           U bytes) const {
  return checked([&] {
    const auto pools = index(v.graph.pools);
    require(!p.address_space.empty() && !p.weights.empty(), "empty placement policy");
    U q = 1, sum = 0;
    for (const auto& [id, w] : p.weights) {
      const auto& pool = at(pools, id);
      require(w && contains(v.allocatable_pools, id), "placement pool not allocatable");
      require(pool.address_space && *pool.address_space == p.address_space, "pool address space mismatch");
      q = std::max(q, quantum(pool));
      sum = add(sum, w);
    }
    require(p.stripe_bytes && p.stripe_bytes % q == 0, "invalid placement stripe");
    // A contiguous policy has constant-size metadata irrespective of allocation size.
    const U n = page_charge(bytes, q);
    std::vector<std::string> cycle = p.cycle;
    if (cycle.empty()) {
      require(sum <= 1048576, "weighted placement cycle too large; supply explicit cycle");
      for (const auto& [id, w] : p.weights)
        for (U i = 0; i < w; ++i)
          cycle.push_back(id);
    }
    std::map<std::string, U> counts;
    for (const auto& id : cycle) {
      require(p.weights.contains(id), "unknown cycle pool");
      ++counts[id];
    }
    require(counts == p.weights, "placement cycle differs from weights");
    require(p.phase < cycle.size(), "invalid placement phase");
    std::string key = std::to_string(p.address_space.size()) + ":" + p.address_space + "/" + std::to_string(q) + "/" +
                      std::to_string(p.stripe_bytes) + "/" + std::to_string(p.phase);
    for (const auto& id : cycle)
      key += "/" + std::to_string(id.size()) + ":" + id;
    ResolvedPlacement result{key, p.address_space, q, {}};
    if (p.weights.size() == 1)
      result.segments.push_back({0, n, p.weights.begin()->first});
    else
      for (U pos = 0, i = p.phase; pos < n; i = (i + 1) % cycle.size()) {
        const U count = std::min(p.stripe_bytes, n - pos);
        if (!result.segments.empty() && result.segments.back().pool == cycle[i])
          result.segments.back().size_bytes = add(result.segments.back().size_bytes, count);
        else
          result.segments.push_back({pos, count, cycle[i]});
        pos += count;
      }
    return result;
  });
}

Result<bool> ResourceModel::compatible_overlap(const ResolvedPlacement& a, U ao, const ResolvedPlacement& b, U bo,
                                               U n) const {
  return checked([&] {
    auto length = [](const auto& p) {
      require(!p.segments.empty(), "empty placement");
      return add(p.segments.back().offset_bytes, p.segments.back().size_bytes);
    };
    const U al = length(a), bl = length(b);
    placement_valid(a, al);
    placement_valid(b, bl);
    range(ao, n, al);
    range(bo, n, bl);
    if (a.key != b.key || a.address_space != b.address_space || a.page_bytes != b.page_bytes || ao % a.page_bytes ||
        bo % b.page_bytes || n % a.page_bytes)
      return false;
    std::size_t i = 0, j = 0;
    U done = 0;
    while (done < n) {
      while (a.segments[i].offset_bytes + a.segments[i].size_bytes <= ao + done)
        ++i;
      while (b.segments[j].offset_bytes + b.segments[j].size_bytes <= bo + done)
        ++j;
      const auto& x = a.segments[i];
      const auto& y = b.segments[j];
      if (x.pool != y.pool)
        return false;
      done +=
        std::min({n - done, x.offset_bytes + x.size_bytes - ao - done, y.offset_bytes + y.size_bytes - bo - done});
    }
    return true;
  });
}

Result<void> ResourceModel::validate_recipe(const ResourceView& v, const MemoryRecipe& r) const {
  return checked([&] {
    require(!r.participant.empty(), "empty recipe participant");
    const auto pools = index(v.graph.pools);
    const auto groups = index(v.graph.capacity_groups);
    const auto regions = index(r.addresses.regions);
    const auto backings = index(r.backings);
    const auto contracts = index(r.sharing_contracts);
    index(r.addresses.slots);
    for (const auto& c : r.sharing_contracts) {
      unique(c.participants);
      unique(c.producers);
      unique(c.consumers);
      require(!c.protocol.empty(), "empty sharing protocol");
      for (const auto& p : c.producers)
        require(contains(c.participants, p), "producer outside sharing participants");
      for (const auto& p : c.consumers)
        require(contains(c.participants, p), "consumer outside sharing participants");
    }
    for (const auto& region : r.addresses.regions) {
      require(!region.address_space.empty() && power2(region.page_bytes) && region.bytes &&
                region.bytes % region.page_bytes == 0,
              "invalid virtual region");
      require(region.base_alignment_bytes == 0 || power2(region.base_alignment_bytes),
              "invalid virtual base alignment");
    }
    std::map<std::string, U> charges;
    for (const auto& b : r.backings) {
      const auto& p = at(pools, b.pool);
      at(groups, p.capacity_group);
      require(!b.owner.empty() && power2(b.page_bytes) && b.page_bytes >= quantum(p) && b.bytes &&
                b.bytes % b.page_bytes == 0,
              "invalid backing geometry");
      require(b.mode == BackingMode::private_memory || b.mode == BackingMode::shareable_remappable ||
                b.mode == BackingMode::private_movable, "unknown backing mode");
      unique(b.export_to);
      require(b.export_to.empty() || b.mode == BackingMode::shareable_remappable, "export needs shareable backing");
      if (b.owner == r.participant) {
        require(contains(v.allocatable_pools, b.pool), "backing pool outside allocation scope");
        charges[p.capacity_group] = add(charges[p.capacity_group], b.bytes);
      } else
        require(contains(b.export_to, r.participant), "backing not exported to participant");
    }
    std::map<std::string, std::vector<const PageMapping*>> maps;
    for (const auto& m : r.mappings) {
      const auto& region = at(regions, m.region);
      const auto& b = at(backings, m.backing);
      const auto& p = at(pools, b.pool);
      const U q = std::max(region.page_bytes, b.page_bytes);
      rights(m.access);
      require(p.address_space && *p.address_space == region.address_space, "mapping address space mismatch");
      require(region.page_bytes >= b.page_bytes && m.virtual_offset_bytes % q == 0 && m.backing_offset_bytes % q == 0 &&
                m.size_bytes % q == 0,
              "unaligned mapping");
      range(m.virtual_offset_bytes, m.size_bytes, region.bytes);
      range(m.backing_offset_bytes, m.size_bytes, b.bytes);
      maps[m.region].push_back(&m);
    }
    for (const auto& backing : r.backings) {
      if (backing.mode != BackingMode::private_movable) continue;
      std::size_t count = 0;
      for (const auto& mapping : r.mappings) if (mapping.backing == backing.id) {
        ++count;
        require(mapping.backing_offset_bytes == 0 && mapping.size_bytes == backing.bytes,
                "private_movable requires one whole-root mapping");
        for (const auto& slot : r.addresses.slots) if (slot.region == mapping.region)
          require(!slot.sharing_contract.has_value(), "private_movable cannot have a sharing contract");
      }
      require(count <= 1, "private_movable cannot have multiple active mappings");
    }
    for (auto& [id, entries] : maps) {
      (void)id;
      std::sort(entries.begin(), entries.end(),
                [](auto a, auto b) { return a->virtual_offset_bytes < b->virtual_offset_bytes; });
      U end = 0;
      for (auto m : entries) {
        require(m->virtual_offset_bytes >= end, "overlapping virtual mappings");
        end = add(m->virtual_offset_bytes, m->size_bytes);
      }
    }
    // Contract sweep is independent of slot pairs, including thousands of slots
    // at one reusable offset. End events precede start events at equal offsets.
    using Contract = std::tuple<std::string, std::string, U, std::optional<std::string>>;
    struct Event {
      U at;
      bool begin;
      Contract key;
    };
    std::map<std::string, std::vector<Event>> events;
    for (const auto& slot : r.addresses.slots) {
      const auto& region = at(regions, slot.region);
      const auto& p = slot.placement;
      placement_valid(p, slot.size_bytes);
      require(p.address_space == region.address_space && p.page_bytes == region.page_bytes &&
                slot.offset_bytes % region.page_bytes == 0,
              "slot placement/address alignment mismatch");
      range(slot.offset_bytes, slot.size_bytes, region.bytes);
      if (slot.sharing_contract)
        require(contains(at(contracts, *slot.sharing_contract).participants, r.participant),
                "slot sharing participant mismatch");
      const auto& entries = maps[slot.region];
      for (const auto& s : p.segments) {
        at(pools, s.pool);
        U cursor = add(slot.offset_bytes, s.offset_bytes);
        const U end = add(cursor, s.size_bytes);
        auto it = std::upper_bound(entries.begin(), entries.end(), cursor,
                                   [](U x, auto m) { return x < m->virtual_offset_bytes; });
        if (it != entries.begin())
          --it;
        while (cursor < end && it != entries.end()) {
          const auto& m = **it;
          const U stop = add(m.virtual_offset_bytes, m.size_bytes);
          if (stop <= cursor) {
            ++it;
            continue;
          }
          require(m.virtual_offset_bytes <= cursor && at(backings, m.backing).pool == s.pool,
                  "slot mapping hole or placement mismatch");
          cursor = std::min(end, stop);
          ++it;
        }
        require(cursor == end, "slot not completely mapped");
      }
      Contract key{p.key, p.address_space, p.page_bytes, slot.sharing_contract};
      events[slot.region].push_back({slot.offset_bytes, true, key});
      events[slot.region].push_back({add(slot.offset_bytes, slot.size_bytes), false, key});
    }
    for (auto& [id, entries] : events) {
      (void)id;
      std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return std::tie(a.at, a.begin) < std::tie(b.at, b.begin); });
      std::map<Contract, std::size_t> active;
      for (const auto& e : entries)
        if (e.begin) {
          require(active.empty() || active.contains(e.key), "overlapping incompatible slot contracts");
          ++active[e.key];
        } else {
          auto it = active.find(e.key);
          require(it != active.end(), "invalid contract sweep");
          if (!--it->second)
            active.erase(it);
        }
    }
    for (const auto& [id, n] : charges) {
      const auto& c = at(groups, id);
      require(!c.capacity_bytes || n <= *c.capacity_bytes, "recipe exceeds unique capacity");
    }
  });
}

Result<OperandEnvironment> ResourceModel::describe_operands(const ResourceView& v, const MemoryRecipe& r,
                                                            std::span<const MemoryDescriptor> external) const {
  auto valid = validate_recipe(v, r);
  if (!valid)
    return {valid.error.value()};
  return checked([&] {
    const auto backings = index(r.backings);
    const auto regions = index(r.addresses.regions);
    std::map<std::string, std::vector<const PageMapping*>> maps;
    for (const auto& m : r.mappings)
      maps[m.region].push_back(&m);
    for (auto& [id, entries] : maps) {
      (void)id;
      std::sort(entries.begin(), entries.end(),
                [](auto a, auto b) { return a->virtual_offset_bytes < b->virtual_offset_bytes; });
    }
    OperandEnvironment env{};
    env.memories.assign(external.begin(), external.end());
    for (const auto& d : external)
      require(d.kind == MemoryRefKind::external, "external descriptor has slot kind");
    for (const auto& s : r.addresses.slots) {
      MemoryDescriptor d{MemoryRefKind::slot, s.id, at(regions, s.region).address_space, s.size_bytes,
                         r.layout_generation, {}};
      const U end = add(s.offset_bytes, s.size_bytes);
      const auto& entries = maps[s.region];
      auto it = std::upper_bound(entries.begin(), entries.end(), s.offset_bytes,
                                 [](U x, auto m) { return x < m->virtual_offset_bytes; });
      if (it != entries.begin())
        --it;
      for (; it != entries.end(); ++it) {
        const auto& m = **it;
        if (m.virtual_offset_bytes >= end)
          break;
        const U lo = std::max(s.offset_bytes, m.virtual_offset_bytes),
                hi = std::min(end, add(m.virtual_offset_bytes, m.size_bytes));
        if (lo >= hi)
          continue;
        const auto& b = at(backings, m.backing);
        d.spans.push_back({lo - s.offset_bytes, hi - lo, physical_key(r, b),
                           add(m.backing_offset_bytes, lo - m.virtual_offset_bytes), m.access, b.pool});
      }
      env.memories.push_back(std::move(d));
    }
    descriptors(env);
    return env;
  });
}

Result<void> ResourceModel::validate_commands(const ResourceView& v, const MemoryCommandPlan& plan,
                                              const OperandEnvironment& env) const {
  return checked([&] {
    (void)v;
    const auto ds = descriptors(env);
    std::vector<std::string> ids;
    std::vector<std::vector<std::string>> deps;
    for (const auto& c : plan.commands) {
      ids.push_back(c.id);
      deps.push_back(c.dependencies);
    }
    order(ids, deps);
    for (const auto& c : plan.commands) {
      require(!c.queue.empty(), "unbound command queue");
      unique(c.wait_ports);
      switch (c.opcode) {
      case MemoryOpcode::copy: {
        require(c.source && c.target && c.source->size_bytes == c.target->size_bytes,
                "copy needs equal sized operands");
        nonoverlap(resolve(*c.source, ds, AccessMode::read), resolve(*c.target, ds, AccessMode::write));
        break;
      }
      case MemoryOpcode::fill:
        require(!c.source && c.target && c.fill_byte <= 255, "invalid fill command");
        resolve(*c.target, ds, AccessMode::write);
        break;
      case MemoryOpcode::register_range:
      case MemoryOpcode::unregister_range:
        require(!c.source && c.target, "registration needs target only");
        {
          auto it = ds.find({c.target->kind, c.target->memory});
          require(it != ds.end(), "unknown registration operand");
          range(c.target->offset_bytes, c.target->size_bytes, it->second->size_bytes);
        }
        break;
      case MemoryOpcode::fence:
        require(!c.source && !c.target, "fence has no operands");
        break;
      default:
        require(false, "invalid memory opcode");
      }
    }
  });
}

namespace {
const Method& method(const ResourceView& v, const std::string& id) {
  return at(index(v.graph.methods), id);
}
const AccessConnection& access_to(const ResourceView& v, const std::string& agent, const std::string& pool,
                                  AccessMode mode) {
  const AccessConnection* found = nullptr;
  for (const auto& e : v.graph.access)
    if (e.agent == agent && e.pool == pool && contains(e.access, mode)) {
      require(found == nullptr, "ambiguous access connection; resolve access rule before query");
      found = &e;
    }
  require(found != nullptr, "missing access connection");
  return *found;
}
void append_access_fabric(std::vector<ResourceDemand>& demands, const AccessConnection& edge,
                           AccessMode mode, double fraction) {
  // These are actual requester load/store traversals, in addition to the
  // method route's declared work. Read and write may traverse the same service
  // and must both be counted; an explicit route use is not a deduplication key.
  for (const auto& resource : mode == AccessMode::read ? edge.read_fabric : edge.write_fabric)
    demands.push_back({resource, DemandKind::traffic, fraction, 0});
}
std::vector<ResourceDemand> access_demands(const ResourceView& v, const AccessConnection& e, AccessMode mode) {
  const auto pool_index = index(v.graph.pools);
  const auto& p = at(pool_index, e.pool);
  const bool read = mode == AccessMode::read;
  const auto service = read ? e.read_service.value_or(p.read_service.value_or(""))
                            : e.write_service.value_or(p.write_service.value_or(""));
  std::vector<ResourceDemand> d;
  if (!service.empty())
    d.push_back({service, read ? DemandKind::read : DemandKind::write, 1, 0});
  for (const auto& r : read ? e.read_fabric : e.write_fabric)
    d.push_back({r, DemandKind::traffic, 1, 0});
  return d;
}
const Route* route_for(const ResourceView& v, const Method& m, const std::string& source, const std::string& target,
                       const std::optional<std::string> selected, const std::optional<std::string> executor) {
  const Route* found = nullptr;
  for (const auto& r : m.routes)
    if (r.source == source && r.target == target && (!selected || r.id == *selected)) {
      if (m.request_executor && (!executor || !contains(v.executable_agents, *executor)))
        continue;
      if (r.agent && executor && *r.agent != *executor)
        continue;
      require(found == nullptr, "ambiguous transfer route; select route explicitly");
      found = &r;
    }
  require(found != nullptr, "no usable transfer route");
  return found;
}
U endpoint_size(const TransferEndpoint& ep, const ResourceView& v) {
  require(!ep.address_space.empty(), "empty endpoint address space");
  const auto pools = index(v.graph.pools);
  U end = 0;
  for (const auto& s : ep.segments) {
    const auto& p = at(pools, s.pool);
    require(p.address_space && *p.address_space == ep.address_space && s.offset_bytes == end && s.size_bytes,
            "invalid transfer endpoint");
    end = add(end, s.size_bytes);
  }
  require(end, "empty transfer endpoint");
  return end;
}
} // namespace

Result<std::vector<TransferCandidate>> ResourceModel::candidates(const ResourceView& v, const TransferEndpoint& source,
                                                                 const TransferEndpoint& target) const {
  return checked([&] {
    const U source_bytes = endpoint_size(source, v), target_bytes = endpoint_size(target, v);
    require(source_bytes == target_bytes, "candidate endpoints have different lengths");
    std::vector<TransferCandidate> result;
    for (const auto& m : v.graph.methods) {
      TransferCandidate c{m.id, {}, {}};
      bool usable = true;
      U cursor = 0;
      std::size_t i = 0, j = 0;
      while (cursor < source_bytes) {
        const auto& a = source.segments[i];
        const auto& b = target.segments[j];
        bool pair = false;
        for (const auto& r : m.routes)
          if (r.source == a.pool && r.target == b.pool) {
            pair = true;
            if (!contains(c.routes, r.id))
              c.routes.push_back(r.id);
          }
        usable = usable && pair;
        cursor = std::min(a.offset_bytes + a.size_bytes, b.offset_bytes + b.size_bytes);
        if (cursor == a.offset_bytes + a.size_bytes)
          ++i;
        if (cursor == b.offset_bytes + b.size_bytes)
          ++j;
      }
      if (usable)
        result.push_back(std::move(c));
    }
    return result;
  });
}

Result<CostPlan> ResourceModel::cost_plan(const ResourceView& v, const MemoryCommandPlan& plan,
                                          const OperandEnvironment& env) const {
  auto valid = validate_commands(v, plan, env);
  if (!valid)
    return {valid.error.value()};
  return checked([&] {
    const auto ds = descriptors(env);
    CostPlan result;
    for (const auto& c : plan.commands) {
      Activity a{c.id, c.method, c.route, 0, c.dependencies, {}, 0, c.queue};
      if (c.opcode == MemoryOpcode::fence) {
        result.activities.push_back(std::move(a));
        continue;
      }
      const auto& m = method(v, c.method);
      a.fixed_us = m.startup_us.value_or(0);
      if (c.opcode == MemoryOpcode::register_range || c.opcode == MemoryOpcode::unregister_range) {
        result.activities.push_back(std::move(a));
        continue;
      }
      const auto target = resolve(*c.target, ds, AccessMode::write);
      a.payload_bytes = c.target->size_bytes;
      if (c.opcode == MemoryOpcode::fill) {
        for (const auto& t : target) {
          require(t.pool.has_value(), "missing target placement for costing");
          auto demands = access_demands(v, access_to(v, c.queue, *t.pool, AccessMode::write), AccessMode::write);
          for (auto& d : demands)
            d.amount *= static_cast<double>(t.size_bytes) / static_cast<double>(a.payload_bytes);
          a.demands.insert(a.demands.end(), demands.begin(), demands.end());
        }
      } else {
        const auto source = resolve(*c.source, ds, AccessMode::read);
        std::size_t i = 0, j = 0;
        U done = 0;
        std::set<std::string> used_routes;
        while (done < a.payload_bytes) {
          const auto& s = source[i];
          const auto& t = target[j];
          require(s.pool && t.pool, "unknown placement for copy costing");
          const U count = std::min(s.offset_bytes + s.size_bytes, t.offset_bytes + t.size_bytes) - done;
          const std::optional<std::string> executor = c.queue;
          const auto* route = route_for(v, m, *s.pool, *t.pool, c.route, executor);
          const auto& r = *route;
          const bool first = used_routes.insert(r.id).second;
          for (auto d : r.demands) {
            if (d.service != DemandKind::occupancy)
              d.amount *= static_cast<double>(count) / static_cast<double>(a.payload_bytes);
            if (!first)
              d.per_invocation_bytes = 0;
            a.demands.push_back(std::move(d));
          }
          if (m.request_executor)
            for (const auto mode : {AccessMode::read, AccessMode::write}) {
              const auto& edge = access_to(v, c.queue, mode == AccessMode::read ? *s.pool : *t.pool, mode);
              append_access_fabric(a.demands, edge, mode,
                                   static_cast<double>(count) / static_cast<double>(a.payload_bytes));
            }
          done += count;
          if (done == s.offset_bytes + s.size_bytes)
            ++i;
          if (done == t.offset_bytes + t.size_bytes)
            ++j;
        }
        if (used_routes.size() == 1)
          a.route = *used_routes.begin();
      }
      result.activities.push_back(std::move(a));
    }
    return result;
  });
}

Result<CostEstimate> ResourceModel::estimate(const ResourceView& v, const TransferQuery& q,
                                             const CostContext& context) const {
  return checked([&] {
    require(q.payload_bytes && q.payload_bytes <= endpoint_size(q.source, v) &&
              q.payload_bytes <= endpoint_size(q.target, v),
            "transfer payload outside endpoints");
    const auto& m = method(v, q.method);
    Activity a{"transfer", m.id, {}, q.payload_bytes, {}, {}, m.startup_us.value_or(0), q.executor};
    std::size_t i = 0, j = 0;
    U done = 0;
    std::set<std::string> overhead_routes;
    std::map<std::pair<std::string, AccessMode>, std::pair<U, double>> access_work;
    while (done < q.payload_bytes) {
      const auto& s = q.source.segments[i];
      const auto& t = q.target.segments[j];
      const U count = std::min({q.payload_bytes, s.offset_bytes + s.size_bytes, t.offset_bytes + t.size_bytes}) - done;
      const auto* route = route_for(v, m, s.pool, t.pool, std::nullopt, q.executor);
      const auto& r = *route;
      const bool first = overhead_routes.insert(r.id).second;
      for (auto d : r.demands) {
        if (d.service != DemandKind::occupancy)
          d.amount *= static_cast<double>(count) / static_cast<double>(q.payload_bytes);
        if (!first)
          d.per_invocation_bytes = 0;
        a.demands.push_back(std::move(d));
      }
      if (m.request_executor) {
        require(q.executor.has_value(), "transfer requires executor");
        for (auto mode : {AccessMode::read, AccessMode::write}) {
          const auto& edge = access_to(v, *q.executor, mode == AccessMode::read ? s.pool : t.pool, mode);
          append_access_fabric(a.demands, edge, mode,
                               static_cast<double>(count) / static_cast<double>(q.payload_bytes));
          const std::string mode_name = mode == AccessMode::read ? "read" : "write";
          if (edge.bandwidth && edge.bandwidth->contains(mode_name)) {
            auto& work = access_work[{edge.id, mode}];
            work.first = add(work.first, count);
            work.second = edge.bandwidth->at(mode_name);
          }
          for (const auto& p : edge.prerequisites)
            require(contains(q.satisfied_prerequisites, p), "unsatisfied transfer prerequisite");
        }
      }
      done += count;
      if (done == s.offset_bytes + s.size_bytes)
        ++i;
      if (done == t.offset_bytes + t.size_bytes)
        ++j;
    }
    ResourceView local = v;
    if (!access_work.empty()) {
      Method capped = m;
      for (const auto& [id, w] : access_work) {
        (void)id;
        const double cap = w.second * static_cast<double>(q.payload_bytes) / static_cast<double>(w.first);
        capped.bandwidth = std::min(capped.bandwidth.value_or(std::numeric_limits<double>::infinity()), cap);
      }
      U index = 0;
      do {
        capped.id = "query-method/" + std::to_string(index++);
      } while (std::any_of(local.graph.methods.begin(), local.graph.methods.end(),
                           [&](const auto& candidate) { return candidate.id == capped.id; }));
      a.method = capped.id;
      local.graph.methods.push_back(std::move(capped));
    }
    auto cost = estimate_schedule(local, CostPlan{{a}}, context);
    if (!cost)
      throw std::invalid_argument(std::get<Error>(cost.state).message);
    return std::move(cost).value();
  });
}

Result<CostEstimate> ResourceModel::estimate_access(const ResourceView& v, const AccessQuery& q,
                                                    const CostContext& context) const {
  return checked([&] {
    require(q.payload_bytes && q.payload_bytes <= endpoint_size(q.memory, v), "access payload outside endpoint");
    require(contains(v.executable_agents, q.agent), "access agent outside executable scope");
    // Access caps are local query resources. They supplement shared physical
    // services, and do not overwrite them when projecting a smaller process view.
    ResourceView local = v;
    Activity a{"access", q.method.value_or(""), {}, q.payload_bytes, {}, {}, 0, q.agent};
    U done = 0;
    std::size_t cap_id = 0;
    std::map<std::pair<std::string, AccessMode>, std::string> path_caps;
    for (const auto& s : q.memory.segments) {
      if (done == q.payload_bytes)
        break;
      const U n = std::min(s.size_bytes, q.payload_bytes - done);
      const auto& e = access_to(v, q.agent, s.pool, q.access);
      auto demands = access_demands(v, e, q.access);
      const auto mode = q.access == AccessMode::read ? "read" : "write";
      if (e.bandwidth && e.bandwidth->contains(mode)) {
        auto& id = path_caps[{e.id, q.access}];
        if (id.empty()) {
          do {
            id = "query-cap/" + std::to_string(cap_id++);
          } while (std::any_of(local.graph.resources.begin(), local.graph.resources.end(),
                               [&](const auto& r) { return r.id == id; }));
          local.graph.resources.push_back(
            {id, "", e.bandwidth->at(mode), {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}});
        }
        demands.push_back({id, DemandKind::traffic, 1, 0});
      }
      for (auto& d : demands)
        d.amount *= static_cast<double>(n) / static_cast<double>(q.payload_bytes);
      a.demands.insert(a.demands.end(), demands.begin(), demands.end());
      a.fixed_us = std::max(a.fixed_us, e.latency_ns.value_or(0) / 1000.0);
      done += n;
    }
    auto cost = estimate_schedule(local, CostPlan{{a}}, context);
    if (!cost)
      throw std::invalid_argument(std::get<Error>(cost.state).message);
    return std::move(cost).value();
  });
}

Result<CostEstimate> ResourceModel::estimate_schedule(const ResourceView& v, const CostPlan& plan,
                                                      const CostContext& context) const {
  return checked([&] {
    require(context.view_epoch == v.epoch, "stale cost view");
    const auto resources = index(v.graph.resources);
    const auto methods = index(v.graph.methods);
    std::vector<Activity> activities = context.active;
    activities.insert(activities.end(), plan.activities.begin(), plan.activities.end());
    const std::size_t background = context.active.size();
    std::vector<std::string> ids;
    std::vector<std::vector<std::string>> deps;
    for (const auto& a : activities) {
      ids.push_back(a.id);
      deps.push_back(a.dependencies);
    }
    order(ids, deps);
    std::unordered_map<std::string, std::size_t> by_id;
    for (std::size_t i = 0; i < ids.size(); ++i)
      by_id.emplace(ids[i], i);
    // A normalized activity progress variable carries all its resource work.
    // Shared resource rates use max-min sharing among currently runnable work;
    // explicit occupancy resources and executor slots gate concurrent starts.
    struct Work {
      using ServiceKey = std::pair<std::string, DemandKind>;
      std::map<ServiceKey, double> needs;
      std::map<std::pair<bool, std::string>, U> occupancy;
      double fixed = 0, remaining = 1, private_rate = std::numeric_limits<double>::infinity(), finished = 0;
      bool started = false, done = false;
    };
    std::vector<Work> work(activities.size());
    std::map<Work::ServiceKey, double> capacities;
    std::map<std::pair<bool, std::string>, U> slot_capacity;
    for (const auto& r : v.graph.resources)
      if (r.kind == ResourceKind::execution_slots)
        slot_capacity[{false, r.id}] = *r.slots;
    for (const auto& [agent, slots] : v.execution_slots) {
      require(slots && contains(v.executable_agents, agent), "invalid view execution slots");
      slot_capacity[{true, agent}] = slots;
    }
    for (const auto& [agent, slots] : context.execution_slots) {
      require(slots && contains(v.executable_agents, agent), "invalid cost execution slots");
      slot_capacity[{true, agent}] = slots;
    }
    CostEstimate estimate{};
    estimate.quality = CostQuality::reference;
    for (std::size_t i = 0; i < activities.size(); ++i) {
      const auto& a = activities[i];
      auto& w = work[i];
      require(std::isfinite(a.fixed_us) && a.fixed_us >= 0, "invalid fixed service cost");
      w.fixed = a.fixed_us;
      if (a.executor) {
        require(contains(v.executable_agents, *a.executor), "activity executor outside view");
        if (slot_capacity.contains({true, *a.executor}))
          w.occupancy[{true, *a.executor}] = 1;
      }
      if (!a.method.empty()) {
        const auto& m = at(methods, a.method);
        if (m.bandwidth && a.payload_bytes)
          w.private_rate = *m.bandwidth * gib / 1e6 / static_cast<double>(a.payload_bytes);
        if (m.request_executor && a.route && a.payload_bytes) {
          const auto routes = index(m.routes);
          const auto& selected = at(routes, *a.route);
          require(a.executor.has_value(), "costed route needs executor");
          for (auto mode : {AccessMode::read, AccessMode::write}) {
            const auto& edge =
              access_to(v, *a.executor, mode == AccessMode::read ? selected.source : selected.target, mode);
            const std::string name = mode == AccessMode::read ? "read" : "write";
            if (edge.bandwidth && edge.bandwidth->contains(name))
              w.private_rate =
                std::min(w.private_rate, edge.bandwidth->at(name) * gib / 1e6 / static_cast<double>(a.payload_bytes));
          }
        }
      }
      for (const auto& d : a.demands) {
        demand_valid(d, resources);
        const auto& r = at(resources, d.resource);
        if (d.service == DemandKind::occupancy) {
          require(d.amount < std::ldexp(1.0, 64), "slot demand overflow");
          const U n = static_cast<U>(d.amount);
          w.occupancy[{false, d.resource}] = std::max(w.occupancy[{false, d.resource}], n);
          continue;
        }
        std::optional<double> cap;
        Work::ServiceKey key{d.resource, DemandKind::traffic};
        if (r.sharing == SharingMode::shared)
          cap = r.bandwidth;
        else if (d.service == DemandKind::read) {
          cap = r.read;
        } else if (d.service == DemandKind::write) {
          cap = r.write;
        } else {
          cap = r.bandwidth;
        }
        const double amount =
          static_cast<double>(a.payload_bytes) * d.amount + static_cast<double>(d.per_invocation_bytes);
        require(std::isfinite(amount), "resource work overflow");
        if (amount) {
          if (!cap) {
            estimate.quality = CostQuality::incomplete;
            estimate.missing_facts.push_back("missing bandwidth: " + d.resource + " for " + a.id);
          } else {
            if (r.sharing == SharingMode::read_write) {
              // Read and write share one normalized service budget. The
              // scheduler clock is microseconds, so each byte demand becomes
              // microseconds at that direction's isolated rate. Per-invocation
              // traffic participates in the same budget as payload traffic.
              capacities[key] = 1;
              w.needs[key] += amount / (*cap * gib / 1e6);
            } else {
              capacities[key] = *cap * gib / 1e6;
              w.needs[key] += amount;
            }
          }
        }
      }
      for (const auto& [id, n] : w.occupancy)
        require(slot_capacity.contains(id) && n <= slot_capacity[id], "activity exceeds execution capacity");
      if (a.payload_bytes && w.needs.empty() && !std::isfinite(w.private_rate)) {
        estimate.quality = CostQuality::incomplete;
        estimate.missing_facts.push_back("unmodeled data service: " + a.id);
      }
      if (i >= background)
        estimate.demands.insert(estimate.demands.end(), a.demands.begin(), a.demands.end());
    }
    double now = 0;
    std::size_t completed = 0;
    std::map<std::pair<bool, std::string>, U> occupied;
    while (completed < work.size()) {
      bool change = true;
      while (change) {
        change = false;
        for (std::size_t i = 0; i < work.size(); ++i) {
          auto& w = work[i];
          if (w.done)
            continue;
          if (!w.started) {
            bool ready = true;
            for (const auto& d : activities[i].dependencies)
              ready = ready && work[by_id.at(d)].done;
            for (const auto& [id, n] : w.occupancy)
              ready = ready && n <= slot_capacity[id] - occupied[id];
            if (!ready)
              continue;
            w.started = true;
            change = true;
            for (const auto& [id, n] : w.occupancy)
              occupied[id] += n;
          }
          if (w.fixed <= 0 && (w.remaining <= 1e-12 || (w.needs.empty() && !std::isfinite(w.private_rate)))) {
            w.done = true;
            w.finished = now;
            ++completed;
            change = true;
            for (const auto& [id, n] : w.occupancy)
              occupied[id] -= n;
          }
        }
      }
      if (completed == work.size())
        break;
      // Progressive filling redistributes service released by a different
      // bottleneck. Equal payload rates are frozen at private or shared caps.
      std::vector<double> rates(work.size()), throughput(work.size());
      std::set<std::size_t> filling;
      std::map<Work::ServiceKey, double> residual = capacities;
      double step = std::numeric_limits<double>::infinity();
      auto scale = [&](std::size_t i) { return static_cast<double>(std::max<U>(activities[i].payload_bytes, 1)); };
      for (std::size_t i = 0; i < work.size(); ++i) {
        const auto& w = work[i];
        if (!w.started || w.done)
          continue;
        if (w.fixed > 0)
          step = std::min(step, w.fixed);
        else
          filling.insert(i);
      }
      while (!filling.empty()) {
        std::map<Work::ServiceKey, double> slopes;
        double delta = std::numeric_limits<double>::infinity();
        for (auto i : filling) {
          delta = std::min(delta, work[i].private_rate * scale(i) - throughput[i]);
          for (const auto& [id, n] : work[i].needs)
            slopes[id] += n / scale(i);
        }
        for (const auto& [id, n] : slopes)
          if (n > 0)
            delta = std::min(delta, residual[id] / n);
        require(std::isfinite(delta) && delta >= 0, "unbounded resource service");
        for (auto i : filling)
          throughput[i] += delta;
        for (const auto& [id, n] : slopes)
          residual[id] = std::max(0.0, residual[id] - delta * n);
        std::vector<std::size_t> frozen;
        for (auto i : filling) {
          bool freeze = throughput[i] >= work[i].private_rate * scale(i) * (1 - 1e-12);
          for (const auto& [id, n] : work[i].needs)
            if (n > 0 && residual[id] <= capacities[id] * 1e-12)
              freeze = true;
          if (freeze)
            frozen.push_back(i);
        }
        require(!frozen.empty(), "resource filling made no progress");
        for (auto i : frozen)
          filling.erase(i);
      }
      for (std::size_t i = 0; i < work.size(); ++i)
        if (throughput[i] > 0) {
          rates[i] = throughput[i] / scale(i);
          step = std::min(step, work[i].remaining / rates[i]);
        }
      require(std::isfinite(step) && step >= 0, "resource schedule deadlock");
      now += step;
      require(std::isfinite(now), "schedule duration overflow");
      for (std::size_t i = 0; i < work.size(); ++i) {
        auto& w = work[i];
        if (!w.started || w.done)
          continue;
        if (w.fixed > 0)
          w.fixed = std::max(0.0, w.fixed - step);
        else
          w.remaining = std::max(0.0, w.remaining - rates[i] * step);
      }
    }
    for (std::size_t i = background; i < work.size(); ++i)
      estimate.total_us = std::max(estimate.total_us, work[i].finished);
    estimate.service_us = estimate.total_us;
    estimate.completion_us = estimate.total_us;
    // No empirical bounds are invented. Reference estimates report missing facts.
    return estimate;
  });
}
} // namespace vecops::memory::model
