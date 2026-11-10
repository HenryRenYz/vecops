// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/details/CapacityLedger.h"
#include "vecops/memory/ResourceModel.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <set>

namespace vecops::memory::model::details {
namespace {
using U = std::uint64_t;
U add(U a, U b) {
  if (b > std::numeric_limits<U>::max() - a)
    throw std::overflow_error("capacity arithmetic overflow");
  return a + b;
}
void require(bool ok, const char* message) {
  if (!ok)
    throw std::invalid_argument(message);
}
std::size_t bucket(CapacityState s) {
  const auto i = static_cast<std::size_t>(s);
  require(i < 4, "invalid capacity state");
  return i;
}
U& counter(PoolBudget& b, std::size_t i) {
  switch (i) {
  case 0:
    return b.live_bytes;
  case 1:
    return b.idle_available_bytes;
  case 2:
    return b.idle_reserved_bytes;
  case 3:
    return b.retired_bytes;
  }
  throw std::invalid_argument("invalid capacity state");
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
} // namespace
struct CapacityCharge::State {
  std::mutex mutex;
  std::map<std::string, PoolBudget> groups;
  std::vector<BudgetConstraint> constraints;
  U remaining(const BudgetConstraint& c) {
    U n = c.remaining_bytes;
    for (const auto& id : c.capacity_groups) {
      const auto& g = at(id);
      for (auto used : {g.committed_bytes, g.unbacked_reserved_bytes})
        n = used >= n ? 0 : n - used;
    }
    return n;
  }
  U epoch = 1;
  PoolBudget& at(const std::string& id) {
    auto it = groups.find(id);
    require(it != groups.end(), "unknown capacity group");
    return it->second;
  }
  void advance() {
    if (epoch == std::numeric_limits<U>::max())
      std::terminate();
    ++epoch;
  }
};
CapacityCharge::CapacityCharge(std::shared_ptr<State> state, std::string group, U bytes, CapacityState initial)
  : state_(std::move(state))
  , group_(std::move(group)) {
  bytes_[bucket(initial)] = bytes;
}
CapacityCharge::~CapacityCharge() {
  if (!activated_)
    return;
  std::lock_guard lock(state_->mutex);
  auto& group = state_->at(group_);
  U total = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    counter(group, i) -= bytes_[i];
    total += bytes_[i];
  }
  group.committed_bytes -= total;
  group.system_available_bytes = add(group.system_available_bytes, total);
  state_->advance();
}
Result<void> CapacityCharge::transition(CapacityState from, CapacityState to, U bytes) {
  return checked([&] {
    const auto a = bucket(from), b = bucket(to);
    require(bytes, "empty capacity transition");
    std::lock_guard lock(state_->mutex);
    require(bytes <= bytes_[a], "capacity transition exceeds owned backing state");
    if (a == b)
      return;
    auto& group = state_->at(group_);
    const U next = add(bytes_[b], bytes), global = add(counter(group, b), bytes);
    bytes_[a] -= bytes;
    bytes_[b] = next;
    counter(group, a) -= bytes;
    counter(group, b) = global;
    state_->advance();
  });
}
CapacityGrant::CapacityGrant(std::shared_ptr<CapacityCharge::State> state, std::map<std::string, U> remaining)
  : state_(std::move(state))
  , remaining_(std::move(remaining)) {
}
CapacityGrant::~CapacityGrant() {
  std::lock_guard lock(state_->mutex);
  bool changed = false;
  for (const auto& [id, bytes] : remaining_)
    if (bytes) {
      state_->at(id).unbacked_reserved_bytes -= bytes;
      changed = true;
    }
  if (changed)
    state_->advance();
}
Result<std::shared_ptr<CapacityCharge>> CapacityGrant::commit(const std::string& id, U bytes, CapacityState initial) {
  return checked([&] {
    const auto i = bucket(initial);
    require(bytes, "empty backing commitment");
    std::lock_guard lock(state_->mutex);
    const auto it = remaining_.find(id);
    require(it != remaining_.end() && it->second >= bytes, "commit exceeds grant");
    auto& group = state_->at(id);
    require(group.system_available_bytes >= bytes, "physical free sample exhausted");
    const U committed = add(group.committed_bytes, bytes), next = add(counter(group, i), bytes);
    // Construct ownership before changing counters, including shared_ptr's control
    // block. A construction failure must not run a charged destructor or leak a cap.
    auto charge = std::shared_ptr<CapacityCharge>(new CapacityCharge(state_, id, bytes, initial));
    charge->activated_ = true;
    group.committed_bytes = committed;
    counter(group, i) = next;
    group.system_available_bytes -= bytes;
    group.unbacked_reserved_bytes -= bytes;
    it->second -= bytes;
    state_->advance();
    return charge;
  });
}
CapacityLedger::CapacityLedger(std::vector<PoolBudget> groups, std::vector<BudgetConstraint> constraints)
  : state_(std::make_shared<CapacityCharge::State>()) {
  for (auto& group : groups) {
    require(!group.capacity_group.empty() && group.unbacked_reserved_bytes == 0, "invalid initial capacity group");
    require(static_cast<bool>(budget_available(group)), "inconsistent initial capacity ledger");
    // Initial counters must be owned by this ledger's charges, not anonymous.
    require(group.committed_bytes == 0, "initial commitments need explicit owners");
    require(state_->groups.emplace(group.capacity_group, std::move(group)).second, "duplicate capacity group");
  }
  std::set<std::string> ids;
  for (const auto& c : constraints) {
    require(!c.id.empty() && ids.insert(c.id).second && !c.capacity_groups.empty(), "invalid shared budget constraint");
    std::set<std::string> groups;
    for (const auto& id : c.capacity_groups) {
      state_->at(id);
      require(groups.insert(id).second, "duplicate constraint capacity group");
    }
  }
  state_->constraints = std::move(constraints);
}
BudgetSnapshot CapacityLedger::snapshot() const {
  std::lock_guard lock(state_->mutex);
  BudgetSnapshot s{};
  s.epoch = state_->epoch;
  for (const auto& [id, stored] : state_->groups) {
    (void)id;
    auto group = stored;
    group.system_available_bytes = group.system_available_bytes > group.unbacked_reserved_bytes
                                     ? group.system_available_bytes - group.unbacked_reserved_bytes
                                     : 0;
    s.groups.push_back(std::move(group));
  }
  for (auto c : state_->constraints) {
    c.remaining_bytes = state_->remaining(c);
    s.constraints.push_back(std::move(c));
  }
  return s;
}
Result<std::unique_ptr<CapacityGrant>> CapacityLedger::reserve(const std::map<std::string, U>& requests, U epoch) {
  return reserve_impl(requests, epoch);
}
Result<std::unique_ptr<CapacityGrant>> CapacityLedger::reserve_current(const std::map<std::string, U>& requests) {
  return reserve_impl(requests, {});
}
Result<std::unique_ptr<CapacityGrant>> CapacityLedger::reserve_impl(const std::map<std::string, U>& requests,
                                                                  std::optional<U> epoch) {
  return checked([&] {
    require(!requests.empty(), "empty capacity reservation");
    std::lock_guard lock(state_->mutex);
    require(!epoch || state_->epoch == *epoch, "stale capacity snapshot");
    for (const auto& [id, bytes] : requests) {
      require(bytes, "zero capacity reservation");
      const auto& g = state_->at(id);
      const U used = add(g.committed_bytes, g.unbacked_reserved_bytes);
      require(used <= g.hard_limit_bytes && bytes <= g.hard_limit_bytes - used, "hard capacity limit exceeded");
      require(g.unbacked_reserved_bytes <= g.system_available_bytes &&
                bytes <= g.system_available_bytes - g.unbacked_reserved_bytes,
              "physical free capacity exceeded");
    }
    for (const auto& c : state_->constraints) {
      U available = state_->remaining(c);
      for (const auto& id : c.capacity_groups)
        if (auto it = requests.find(id); it != requests.end()) {
          require(it->second <= available, "shared domain capacity limit exceeded");
          available -= it->second;
        }
    }
    auto grant = std::unique_ptr<CapacityGrant>(new CapacityGrant(state_, requests));
    for (const auto& [id, bytes] : requests)
      state_->at(id).unbacked_reserved_bytes += bytes;
    state_->advance();
    return grant;
  });
}
Result<void> CapacityLedger::refresh_available(const std::map<std::string, U>& free) {
  return checked([&] {
    std::lock_guard lock(state_->mutex);
    for (const auto& [id, bytes] : free) {
      add(bytes, state_->at(id).committed_bytes);
      require(state_->at(id).unbacked_reserved_bytes == 0, "free-capacity refresh requires no materialization grant");
    }
    for (const auto& [id, bytes] : free)
      state_->at(id).system_available_bytes = bytes;
    state_->advance();
  });
}
} // namespace vecops::memory::model::details
