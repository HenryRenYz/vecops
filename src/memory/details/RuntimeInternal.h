// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/LinuxRuntime.h"
#include "vecops/memory/HostEvents.h"
#include "vecops/memory/ResourceModel.h"
#include "vecops/memory/details/CapacityLedger.h"
#include "vecops/memory/details/SharedCapacityAuthority.h"
#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <set>

namespace vecops::memory::model::details {
using U = std::uint64_t;
struct Failure : std::runtime_error {
  Error error;
  explicit Failure(Error e) : std::runtime_error(e.code + ": " + e.message), error(std::move(e)) {}
};
inline void require(bool yes, const char* message) {
  if (!yes)
    throw Failure{Error{"invalid_argument", message}};
}
inline U add(U a, U b) {
  require(b <= std::numeric_limits<U>::max() - a, "byte range overflow");
  return a + b;
}
inline bool power2(U q) {
  return q && !(q & (q - 1));
}
template <class T>
T take(Result<T> r) {
  if (!r)
    throw Failure{std::get<Error>(r.state)};
  return std::move(r).value();
}
inline void take(Result<void> r) {
  if (!r)
    throw Failure{*r.error};
}
template <class F>
auto protect(F&& fn) -> Result<decltype(fn())> {
  using T = decltype(fn());
  try {
    if constexpr (std::is_void_v<T>) {
      fn();
      return {};
    } else
      return {fn()};
  } catch (const Failure& e) {
    return {e.error};
  } catch (const std::invalid_argument& e) {
    return {Error{"invalid_argument", e.what()}};
  } catch (const std::overflow_error& e) {
    return {Error{"overflow", e.what()}};
  } catch (const std::system_error& e) {
    return {Error{"os_error", e.what()}};
  } catch (const std::bad_alloc&) {
    return {Error{"out_of_memory", "memory allocation failed"}};
  }
}
template <class T>
bool has(const std::vector<T>& v, const T& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}
template <class T>
const T& find(const std::vector<T>& v, const std::string& id) {
  auto i = std::find_if(v.begin(), v.end(), [&](const auto& x) { return x.id == id; });
  require(i != v.end(), "unknown object");
  return *i;
}
inline U quantum(const MemoryPool& p) {
  U q = p.granularity_bytes.value_or(1);
  for (auto n : p.page_sizes_bytes)
    q = std::max(q, n);
  require(power2(q), "invalid pool page quantum");
  return q;
}
inline bool stopped(const EventStatus& s) {
  return s.state != EventState::pending && s.accesses_stopped;
}
inline void wait_stopped(EventService& events, const CompletionToken& token) {
  for (;;) {
    if (stopped(take(events.wait(token))))
      return;
    std::this_thread::yield();
  }
}
void quarantine_transfer(std::shared_ptr<HostEventService>, CompletionToken,
                           std::shared_ptr<LinuxTransferOperation>);
struct ChargeSlot {
  std::shared_ptr<CapacityCharge> charge;
  std::shared_ptr<SharedCapacityCharge> shared_charge;
};
struct RootGrant {
  std::unique_ptr<CapacityGrant> local;
  std::unique_ptr<SharedCapacityGrant> shared;
};
struct PoolQuota;
struct Core : std::enable_shared_from_this<Core> {
  ResourceModel model;
  LinuxPageStore pages;
  std::vector<LinuxPoolBinding> pool_bindings;
  const Scope scope;
  // Immutable, trusted projection; public view values never alias this graph.
  std::shared_ptr<const ResourceView> canonical_view;
  LinuxRuntimeOptions options;
  std::vector<BackendDescriptor> descriptors;
  std::shared_ptr<CapacityLedger> ledger;
  std::shared_ptr<SharedCapacityAuthority> authority;
  mutable std::pair<U, U> budget_signature{0, 0};
  mutable U budget_epoch = 0;
  mutable U snapshot_shared_epoch = 0;
  mutable std::recursive_mutex admission;
  std::atomic<U> next{1};
  std::map<std::string, U> pool_pending;
  std::map<std::string, U> pool_idle;
  std::map<std::string, std::weak_ptr<SharedBacking>> observed;
  std::map<std::string, PhysicalAllocationRecord> records;
  Core(LinuxDiscovery, LinuxRuntimeOptions);
  U serial() {
    const auto n = next.fetch_add(1);
    require(n && n != std::numeric_limits<U>::max(), "runtime generation exhausted");
    return n;
  }
  std::string id(const std::string& prefix) {
    return options.participant + "/" + prefix + "/" + std::to_string(serial());
  }
  ResourceView authorize(const ResourceView&) const;
  std::shared_ptr<const ResourceView> authorize_shared(const ResourceView&) const;
  std::shared_ptr<SharedBacking> materialize(const BackingRequest&, RootGrant&, PoolQuota&, CapacityState,
                                             std::shared_ptr<CapacityCharge>* charge = nullptr);
  std::unique_ptr<PoolQuota> reserve_pools(const std::vector<BackingRequest>&);
  std::unique_ptr<RootGrant> reserve_capacity(const std::map<std::string, U>&,
                                              std::optional<U> shared_epoch = std::nullopt);
  std::map<std::string, U> physical_free(bool require_bindings = false) const;
  BudgetSnapshot snapshot(const ResourceView&) const;
};
struct PoolQuota {
  std::shared_ptr<Core> core;
  std::map<std::string, U> remaining;
  ~PoolQuota();
  void consume(const BackingRequest&);
};
struct AllocationStorage {
  std::shared_ptr<SharedBacking> backing;
  std::unique_ptr<VirtualReservation> reservation;
  std::unique_ptr<MappedRange> mapping; // destroyed first
};
struct AllocationAccess {
  static Allocation make(AllocationInfo, std::shared_ptr<AllocationStorage>, std::shared_ptr<HostEventService>,
                         const void* arena = nullptr);
  static const void* arena(const Allocation&);
  static std::shared_ptr<AllocationStorage> storage(const Allocation&);
  static void record(Allocation&, CompletionToken);
};
struct CachedBacking {
  std::shared_ptr<SharedBacking> backing;
  std::function<void()> activate;
  std::function<std::unique_ptr<MappedRange>(U)> mapping;
};
struct CachedBlock {
  std::string pool;
  U bytes, page_bytes;
  BackingMode mode;
};
class ArenaCache {
public:
  virtual ~ArenaCache() = default;
  virtual Core* core() const = 0;
  virtual std::map<std::string, U> idle_by_group() const = 0;
  virtual U largest(const std::string& pool) const = 0;
  virtual std::vector<CachedBlock> blocks() const = 0;
  virtual std::optional<CachedBacking> claim(const BackingRequest&) = 0;
};
std::shared_ptr<MemoryArena> make_arena(std::shared_ptr<Core>, std::shared_ptr<HostEventService>, const ResourceView&,
                                        const ArenaOptions&);
} // namespace vecops::memory::model::details
