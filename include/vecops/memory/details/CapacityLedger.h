// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"
#include <array>
#include <mutex>

namespace vecops::memory::model::details {
/** Ownership counters are per physical backing, never per imported VA alias. */
enum class CapacityState { live, idle_available, idle_reserved, retired };
class CapacityLedger;
class CapacityCharge {
public:
  ~CapacityCharge();
  CapacityCharge(const CapacityCharge&) = delete;
  CapacityCharge& operator=(const CapacityCharge&) = delete;
  Result<void> transition(CapacityState from, CapacityState to, std::uint64_t bytes);

private:
  friend class CapacityLedger;
  friend class CapacityGrant;
  struct State;
  CapacityCharge(std::shared_ptr<State>, std::string, std::uint64_t, CapacityState);
  std::shared_ptr<State> state_;
  std::string group_;
  std::array<std::uint64_t, 4> bytes_{};
  bool activated_ = false;
};
/** Atomic all-group admission. Unused grants cancel on destruction. */
class CapacityGrant {
public:
  ~CapacityGrant();
  CapacityGrant(const CapacityGrant&) = delete;
  CapacityGrant& operator=(const CapacityGrant&) = delete;
  Result<std::shared_ptr<CapacityCharge>> commit(const std::string& group, std::uint64_t bytes,
                                                 CapacityState initial = CapacityState::live);

private:
  friend class CapacityLedger;
  CapacityGrant(std::shared_ptr<CapacityCharge::State>, std::map<std::string, std::uint64_t>);
  std::shared_ptr<CapacityCharge::State> state_;
  std::map<std::string, std::uint64_t> remaining_;
};
class CapacityLedger {
public:
  /** system_available is currently free physical capacity, excluding existing
   * commitments. The ledger deducts its subsequent commits and grants from this
   * sample. Refresh only between materialization transactions (no pending grant).
   */
  explicit CapacityLedger(std::vector<PoolBudget> groups, std::vector<BudgetConstraint> constraints = {});
  BudgetSnapshot snapshot() const;
  Result<std::unique_ptr<CapacityGrant>> reserve(const std::map<std::string, std::uint64_t>& bytes,
                                                 std::uint64_t expected_epoch);
  /** Automatic root admission has no caller snapshot. Check and reserve under
   * one lock, including concurrent final-owner refunds. Explicit reserve()
   * continues to reject a stale expected_epoch.
   */
  Result<std::unique_ptr<CapacityGrant>> reserve_current(const std::map<std::string, std::uint64_t>& bytes);
  Result<void> refresh_available(const std::map<std::string, std::uint64_t>& physical_free);

private:
  Result<std::unique_ptr<CapacityGrant>> reserve_impl(const std::map<std::string, std::uint64_t>&,
                                                    std::optional<std::uint64_t>);
  std::shared_ptr<CapacityCharge::State> state_;
};
} // namespace vecops::memory::model::details
