// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "details/RuntimeInternal.h"
#include <cmath>

namespace vecops::memory::model::details {
namespace {
struct Chunk {
  std::shared_ptr<SharedBacking> root;
  std::shared_ptr<CapacityCharge> charge;
  std::string pool, group;
  U bytes;
  std::map<U, U> free;
  std::unique_ptr<VirtualReservation> reservation;
  std::unique_ptr<MappedRange> mapping;
};
// Eager chunks keep one stable mapping. Suballocation uses an aligned offset,
// without an mmap/munmap pair on every tensor allocation.
struct ChunkMapping final : MappedRange {
  std::shared_ptr<Chunk> chunk;
  U offset = 0, version = 0;
  Result<void*> data() const override {
    return protect([&]() -> void* { return static_cast<unsigned char*>(take(chunk->mapping->data())) + offset; });
  }
  U generation() const override {
    return version;
  }
};
struct ArenaState;
struct Ticket {
  std::weak_ptr<ArenaState> arena;
  std::shared_ptr<Chunk> chunk;
  U offset, bytes;
  CapacityState state;
  ~Ticket();
  void activate();
};
struct ArenaState : std::enable_shared_from_this<ArenaState> {
  std::shared_ptr<Core> core;
  std::weak_ptr<HostEventService> events;
  ResourceView view;
  ArenaOptions options;
  std::string id;
  std::vector<std::string> pools;
  std::vector<std::shared_ptr<Chunk>> chunks;
  U q = 0, limit = 0, committed = 0;
  ~ArenaState() {
    std::lock_guard lock(core->admission);
    for (const auto& c : chunks)
      for (const auto& [off, n] : c->free) {
        (void)off;
        core->pool_idle[c->pool] -= n;
      }
  }
  // All mutable arena state is serialized by core->admission. This also makes
  // chunk state changes atomic with runtime admission and capacity queries.
  U idle() const {
    U n = 0;
    for (const auto& c : chunks)
      for (const auto& [o, b] : c->free) {
        (void)o;
        n = add(n, b);
      }
    return n;
  }
  U trim(U retain) {
    U released = 0, remaining = idle();
    for (auto it = chunks.begin(); it != chunks.end();) {
      const auto& c = *it;
      if (c->free.size() == 1 && c->free.begin()->first == 0 && c->free.begin()->second == c->bytes &&
          remaining > retain) {
        remaining -= c->bytes;
        released = add(released, c->bytes);
        committed -= c->bytes;
        core->pool_idle[c->pool] -= c->bytes;
        it = chunks.erase(it);
      } else
        ++it;
    }
    return released;
  }
  void put(const std::shared_ptr<Chunk>& c, U off, U bytes, CapacityState from) {
    std::lock_guard lock(core->admission);
    take(c->charge->transition(from, CapacityState::idle_available, bytes));
    core->pool_idle[c->pool] = add(core->pool_idle[c->pool], bytes);
    auto next = c->free.lower_bound(off);
    if (next != c->free.begin()) {
      auto prev = std::prev(next);
      if (add(prev->first, prev->second) == off) {
        off = prev->first;
        bytes = add(bytes, prev->second);
        c->free.erase(prev);
      }
    }
    if (next != c->free.end() && add(off, bytes) == next->first) {
      bytes = add(bytes, next->second);
      c->free.erase(next);
    }
    c->free.emplace(off, bytes);
    trim(options.retain_idle_bytes);
  }
  std::shared_ptr<Chunk> grow(const std::string& pool, U bytes) {
    require(bytes <= limit - committed, "arena hard limit exhausted");
    BackingRequest r{core->id("arena-chunk"), core->options.participant, pool, bytes, q, options.backing, {}};
    const auto& p = find(core->model.graph().pools, pool);
    auto quota = core->reserve_pools({r});
    auto grant = core->reserve_capacity({{p.capacity_group, bytes}});
    auto c = std::make_shared<Chunk>();
    c->pool = pool;
    c->group = p.capacity_group;
    c->bytes = bytes;
    c->free.emplace(0, bytes);
    c->root = core->materialize(r, *grant, *quota, CapacityState::idle_available, &c->charge);
    c->reservation = take(core->pages.reserve(bytes, q));
    c->mapping = take(core->pages.map(*c->reservation, c->root, {0, 0, bytes}));
    chunks.push_back(c);
    committed += bytes;
    core->pool_idle[pool] = add(core->pool_idle[pool], bytes);
    return c;
  }
  std::optional<CachedBacking> claim(const BackingRequest& r, CapacityState state) {
    if (r.page_bytes != q || r.mode != options.backing || !has(pools, r.pool) || !r.bytes || r.bytes % q)
      return {};
    for (auto& c : chunks)
      if (c->pool == r.pool)
        for (auto it = c->free.begin(); it != c->free.end(); ++it)
          if (it->second >= r.bytes) {
            const U off = it->first, remaining = it->second - r.bytes;
            auto t = std::make_shared<Ticket>();
            t->arena = shared_from_this();
            t->chunk = c;
            t->offset = off;
            t->bytes = r.bytes;
            t->state = state;
            // Construct everything that can allocate before removing the free range.
            // The unarmed ticket must not return an interval until it was consumed.
            t->bytes = 0;
            auto b = take(core->pages.slice(c->root, off, r.bytes, r.id, t));
            auto after = c->free.end();
            if (remaining)
              after = c->free.emplace(add(off, r.bytes), remaining).first;
            try {
              take(c->charge->transition(CapacityState::idle_available, state, r.bytes));
            } catch (...) {
              if (remaining)
                c->free.erase(after);
              throw;
            }
            c->free.erase(it);
            core->pool_idle[r.pool] -= r.bytes;
            t->bytes = r.bytes;
            return CachedBacking{std::move(b), [t] { t->activate(); },
                                 [c, off](U version) -> std::unique_ptr<MappedRange> {
                                   auto view = std::make_unique<ChunkMapping>();
                                   view->chunk = c;
                                   view->offset = off;
                                   view->version = version;
                                   return view;
                                 }};
          }
    return {};
  }
};
Ticket::~Ticket() {
  if (!bytes)
    return;
  if (auto a = arena.lock()) {
    try {
      a->put(chunk, offset, bytes, state);
    } catch (...) {
      std::terminate();
    }
  }
}
void Ticket::activate() {
  if (auto a = arena.lock()) {
    std::lock_guard lock(a->core->admission);
    if (state == CapacityState::idle_reserved) {
      take(chunk->charge->transition(state, CapacityState::live, bytes));
      state = CapacityState::live;
    }
  } else {
    if (state == CapacityState::idle_reserved) {
      take(chunk->charge->transition(state, CapacityState::live, bytes));
      state = CapacityState::live;
    }
  }
}
U resolve(const std::variant<U, double>& value, U base) {
  if (auto n = std::get_if<U>(&value))
    return *n;
  const double f = std::get<double>(value);
  require(std::isfinite(f) && f >= 0 && f <= 1, "invalid capacity fraction");
  if (f == 1)
    return base;
  const long double n = static_cast<long double>(base) * f;
  require(n <= std::numeric_limits<U>::max(), "capacity fraction overflow");
  return static_cast<U>(n);
}
class Arena final
  : public MemoryArena
  , public ArenaCache {
  std::shared_ptr<ArenaState> s_;
  std::shared_ptr<HostEventService> events_;

public:
  Arena(std::shared_ptr<ArenaState> s, std::shared_ptr<HostEventService> e)
    : s_(std::move(s))
    , events_(std::move(e)) {
  }

  Core* core() const override {
    return s_->core.get();
  }
  const std::string& id() const override {
    return s_->id;
  }
  U page_bytes() const override {
    return s_->q;
  }
  BudgetSnapshot snapshot() const override {
    std::lock_guard lock(s_->core->admission);
    auto b = s_->core->snapshot(s_->view);
    const auto constraint_id = "arena-limit/" + s_->id;
    std::set<std::string> groups;
    for (const auto& pool : s_->pools)
      groups.insert(find(s_->view.graph.pools, pool).capacity_group);
    b.constraints.push_back({constraint_id, {groups.begin(), groups.end()}, s_->limit - s_->committed});
    for (auto& p : b.pools) {
      U own = 0;
      for (const auto& block : blocks())
        if (block.pool == p.pool)
          own = add(own, block.bytes);
      const U other = s_->core->pool_idle[p.pool] - own;
      p.allocatable_bytes = p.allocatable_bytes > other ? p.allocatable_bytes - other : 0;
      if (has(s_->pools, p.pool)) {
        p.allocatable_bytes = std::min(p.allocatable_bytes, add(own, s_->limit - s_->committed));
        p.shared_constraints.push_back(constraint_id);
      } else
        p.allocatable_bytes = 0;
      p.largest_cached_block_bytes = largest(p.pool);
    }
    for (auto& g : b.groups) {
      g.largest_cached_block_bytes = 0;
      for (const auto& p : b.pools)
        if (p.capacity_group == g.capacity_group)
          g.largest_cached_block_bytes = std::max(g.largest_cached_block_bytes, p.largest_cached_block_bytes);
    }
    return b;
  }
  std::map<std::string, U> idle_by_group() const override {
    std::lock_guard lock(s_->core->admission);
    std::map<std::string, U> n;
    for (const auto& c : s_->chunks)
      for (const auto& [off, b] : c->free) {
        (void)off;
        n[c->group] = add(n[c->group], b);
      }
    return n;
  }
  std::vector<CachedBlock> blocks() const override {
    std::lock_guard lock(s_->core->admission);
    std::vector<CachedBlock> result;
    for (const auto& c : s_->chunks)
      for (const auto& [off, n] : c->free) {
        (void)off;
        result.push_back({c->pool, n, s_->q, s_->options.backing});
      }
    return result;
  }
  U largest(const std::string& pool) const override {
    std::lock_guard lock(s_->core->admission);
    U n = 0;
    for (const auto& c : s_->chunks)
      if (c->pool == pool)
        for (const auto& [off, b] : c->free) {
          (void)off;
          n = std::max(n, b);
        }
    return n;
  }
  std::optional<CachedBacking> claim(const BackingRequest& r) override {
    std::lock_guard lock(s_->core->admission);
    return s_->claim(r, CapacityState::idle_reserved);
  }
  Result<Allocation> allocate(const AllocateRequest& r) override {
    return protect([&] {
      std::lock_guard lock(s_->core->admission);
      require(r.bytes > 0, "zero allocation");
      const U n = page_charge(r.bytes, s_->q);
      auto pools = s_->pools;
      if (r.preferred_pool) {
        require(has(pools, *r.preferred_pool), "preferred pool not allowed");
        pools.erase(std::find(pools.begin(), pools.end(), *r.preferred_pool));
        pools.insert(pools.begin(), *r.preferred_pool);
        if (!r.allow_domain_fallback)
          pools.resize(1);
      }
      std::optional<CachedBacking> chosen;
      std::optional<Error> last;
      // An explicit preference takes precedence over cache in a fallback pool.
      auto cached_pools = pools;
      if (r.preferred_pool && s_->options.allow_growth)
        cached_pools.resize(1);
      for (const auto& pool : cached_pools) {
        BackingRequest br{
          s_->core->id("arena-slice"), s_->core->options.participant, pool, n, s_->q, s_->options.backing, {}};
        chosen = s_->claim(br, CapacityState::live);
        if (chosen)
          break;
      }
      if (!chosen && s_->options.allow_growth)
        for (const auto& pool : pools) {
          BackingRequest cached{
            s_->core->id("arena-slice"), s_->core->options.participant, pool, n, s_->q, s_->options.backing, {}};
          chosen = s_->claim(cached, CapacityState::live);
          if (chosen)
            break;
          if (n > s_->limit - s_->committed)
            continue;
          U growth = std::min(page_charge(std::max(n, s_->options.grow_by_bytes), s_->q),
                              (s_->limit - s_->committed) / s_->q * s_->q);
          auto grow = protect([&] { s_->grow(pool, growth); });
          if (!grow && growth != n)
            grow = protect([&] { s_->grow(pool, n); });
          if (!grow) {
            last = *grow.error;
            continue;
          }
          BackingRequest br{
            s_->core->id("arena-slice"), s_->core->options.participant, pool, n, s_->q, s_->options.backing, {}};
          chosen = s_->claim(br, CapacityState::live);
          require(chosen.has_value(), "new arena chunk unavailable");
          break;
        }
      if (!chosen)
        throw Failure{last.value_or(Error{"capacity_exhausted", "no arena block can satisfy allocation"})};
      auto storage = std::make_shared<AllocationStorage>();
      storage->backing = chosen->backing;
      const auto generation = s_->core->serial();
      storage->mapping = chosen->mapping(generation);
      auto info = storage->backing->info();
      info.requested_bytes = r.bytes;
      info.generation = generation;
      return AllocationAccess::make(std::move(info), std::move(storage), events_, s_.get());
    });
  }
  Result<void> record_use(Allocation& a, CompletionToken token) override {
    return protect([&] {
      require(AllocationAccess::arena(a) == s_.get(), "allocation belongs to another arena");
      AllocationAccess::record(a, std::move(token));
    });
  }
  Result<void> release(Allocation&& a, CompletionToken token) override {
    return protect([&] {
      require(AllocationAccess::arena(a) == s_.get(), "allocation belongs to another arena");
      AllocationAccess::record(a, std::move(token));
      auto owned = std::move(a);
    });
  }
  Result<U> trim(U retain) override {
    return protect([&] {
      std::lock_guard lock(s_->core->admission);
      return s_->trim(retain);
    });
  }
};
} // namespace
std::shared_ptr<MemoryArena> make_arena(std::shared_ptr<Core> core, std::shared_ptr<HostEventService> events,
                                        const ResourceView& input, const ArenaOptions& o) {
  auto s = std::make_shared<ArenaState>();
  s->core = std::move(core);
  s->events = events;
  s->view = s->core->authorize(input);
  s->options = o;
  s->id = s->core->id("arena");
  std::vector<std::string> members;
  if (auto i = std::find_if(s->view.graph.pools.begin(), s->view.graph.pools.end(),
                            [&](const auto& p) { return p.id == o.domain; });
      i != s->view.graph.pools.end())
    members.push_back(i->id);
  else
    members = find(s->view.graph.domains, o.domain).members;
  for (const auto& pool : members)
    if (has(s->view.allocatable_pools, pool) && (o.allowed_pools.empty() || has(o.allowed_pools, pool)))
      s->pools.push_back(pool);
  for (const auto& pool : o.allowed_pools)
    require(has(s->pools, pool), "arena allowed pool outside allocation grant/domain");
  require(!s->pools.empty(), "arena has no allocatable pools");
  U capacity = 0;
  std::set<std::string> groups;
  std::optional<std::string> as;
  for (const auto& pool : s->pools) {
    const auto& p = find(s->view.graph.pools, pool);
    require(p.address_space.has_value(), "unknown pool address space");
    if (as)
      require(as == p.address_space, "arena pools have different address spaces");
    else
      as = p.address_space;
    s->q = std::max(s->q, quantum(p));
    if (groups.insert(p.capacity_group).second) {
      const auto& g = find(s->view.graph.capacity_groups, p.capacity_group);
      require(g.capacity_bytes.has_value(), "unknown capacity");
      capacity = add(capacity, *g.capacity_bytes);
    }
  }
  require(o.backing != BackingMode::private_movable, "private_movable does not support arena slices");
  if (o.page_bytes) {
    require(power2(o.page_bytes) && o.page_bytes >= s->q, "arena page quantum below pool quantum");
    s->q = o.page_bytes;
  }
  s->limit = resolve(o.hard_limit.bytes_or_capacity_fraction, capacity);
  require(s->limit <= capacity, "arena limit exceeds unique capacity");
  s->limit = s->limit / s->q * s->q;
  U initial = resolve(o.initial.bytes_or_limit_fraction, s->limit);
  require(initial <= s->limit, "preallocation exceeds hard limit");
  if (initial)
    initial = page_charge(initial, s->q);
  require(initial <= s->limit, "rounded preallocation exceeds hard limit");
  std::lock_guard lock(s->core->admission);
  // Preallocation is transactional: a failed construction drops every chunk.
  for (const auto& pool : s->pools) {
    if (!initial)
      break;
    auto snap = s->core->snapshot(s->view);
    auto p = std::find_if(snap.pools.begin(), snap.pools.end(), [&](const auto& p) { return p.pool == pool; });
    U n = std::min(initial, p->allocatable_bytes) / s->q * s->q;
    if (!n)
      continue;
    s->grow(pool, n);
    initial -= n;
  }
  require(!initial, "preallocation cannot be satisfied");
  return std::make_shared<Arena>(std::move(s), std::move(events));
}
} // namespace vecops::memory::model::details
