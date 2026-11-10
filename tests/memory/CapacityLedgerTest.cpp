// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/details/CapacityLedger.h"
#include "vecops/memory/ResourceModel.h"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
namespace vecops::memory::model::details {
namespace {
PoolBudget group(std::string id, std::uint64_t limit = 1024, std::uint64_t free = 1024) {
  PoolBudget g{};
  g.capacity_group = std::move(id);
  g.hard_limit_bytes = limit;
  g.system_available_bytes = free;
  return g;
}
TEST(CapacityLedger, AllGroupReservationIsAtomicAndUnusedGrantCancels) {
  CapacityLedger ledger({group("a"), group("b", 1024, 100)});
  auto snap = ledger.snapshot();
  EXPECT_FALSE(ledger.reserve({{"a", 512}, {"b", 101}}, snap.epoch));
  EXPECT_EQ(ledger.snapshot().epoch, snap.epoch);
  EXPECT_EQ(ledger.snapshot().groups[0].unbacked_reserved_bytes, 0u);
  auto r = ledger.reserve({{"a", 512}, {"b", 100}}, snap.epoch);
  ASSERT_TRUE(r);
  auto pending = ledger.snapshot();
  EXPECT_EQ(pending.groups[0].unbacked_reserved_bytes, 512u);
  EXPECT_EQ(pending.groups[0].system_available_bytes, 512u);
  EXPECT_FALSE(ledger.refresh_available({{"a", 1000}}));
  r.value().reset();
  EXPECT_EQ(ledger.snapshot().groups[0].unbacked_reserved_bytes, 0u);
  EXPECT_EQ(ledger.snapshot().groups[0].system_available_bytes, 1024u);
}
TEST(CapacityLedger, ChargesAreOwnedOnceAndStateTransitionsPreserveCommitment) {
  CapacityLedger ledger({group("a")});
  auto grant = ledger.reserve({{"a", 800}}, ledger.snapshot().epoch);
  ASSERT_TRUE(grant);
  auto charge = grant.value()->commit("a", 600, CapacityState::idle_available);
  ASSERT_TRUE(charge);
  auto shared = charge.value();
  EXPECT_TRUE(shared->transition(CapacityState::idle_available, CapacityState::live, 200));
  EXPECT_TRUE(shared->transition(CapacityState::live, CapacityState::retired, 100));
  auto snap = ledger.snapshot().groups[0];
  EXPECT_EQ(snap.committed_bytes, 600u);
  EXPECT_EQ(snap.live_bytes, 100u);
  EXPECT_EQ(snap.idle_available_bytes, 400u);
  EXPECT_EQ(snap.retired_bytes, 100u);
  EXPECT_EQ(snap.unbacked_reserved_bytes, 200u);
  EXPECT_FALSE(shared->transition(CapacityState::live, CapacityState::retired, 101));
  grant.value().reset();
  snap = ledger.snapshot().groups[0];
  EXPECT_EQ(snap.system_available_bytes, 424u);
  EXPECT_EQ(budget_available(snap).value(), 824u);
  charge.value().reset();
  EXPECT_EQ(ledger.snapshot().groups[0].committed_bytes, 600u); // shared alias does not double-charge
  shared.reset();
  snap = ledger.snapshot().groups[0];
  EXPECT_EQ(snap.committed_bytes, 0u);
  EXPECT_EQ(snap.system_available_bytes, 1024u);
}
TEST(CapacityLedger, GrantsCannotOvercommitOrUseStaleSnapshots) {
  CapacityLedger ledger({group("a")});
  auto snap = ledger.snapshot();
  auto grant = ledger.reserve({{"a", 512}}, snap.epoch);
  ASSERT_TRUE(grant);
  EXPECT_FALSE(ledger.reserve({{"a", 1}}, snap.epoch));
  EXPECT_FALSE(grant.value()->commit("a", 513));
  auto charge = grant.value()->commit("a", 512);
  ASSERT_TRUE(charge);
  EXPECT_FALSE(grant.value()->commit("a", 1));
  EXPECT_FALSE(ledger.reserve({{"a", 513}}, ledger.snapshot().epoch));
  EXPECT_TRUE(ledger.reserve({{"a", 512}}, ledger.snapshot().epoch));
}
TEST(CapacityLedger, RefreshIsAllOrNothingAndPreservesObservedFreeMeaning) {
  CapacityLedger ledger({group("a"), group("b")});
  EXPECT_FALSE(ledger.refresh_available({{"a", 42}, {"unknown", 1}}));
  EXPECT_EQ(ledger.snapshot().groups[0].system_available_bytes, 1024u);
  auto grant = ledger.reserve({{"a", 100}}, ledger.snapshot().epoch);
  auto charge = grant.value()->commit("a", 100);
  grant.value().reset();
  EXPECT_TRUE(ledger.refresh_available({{"a", 500}}));
  EXPECT_EQ(ledger.snapshot().groups[0].system_available_bytes, 500u);
  EXPECT_FALSE(ledger.refresh_available({{"a", std::numeric_limits<std::uint64_t>::max()}}));
  charge.value().reset();
  EXPECT_EQ(ledger.snapshot().groups[0].system_available_bytes, 600u);
}
TEST(CapacityLedger, AutomaticAdmissionAndConcurrentFinalRefundsDoNotNeedAnEpochRetry) {
  CapacityLedger ledger({group("shared", 16, 16)});
  const auto old = ledger.snapshot().epoch;
  std::atomic<unsigned> failures = 0;
  std::vector<std::thread> workers;
  for (unsigned rank = 0; rank < 16; ++rank)
    workers.emplace_back([&] {
      for (unsigned i = 0; i < 200; ++i) {
        // At most 16 one-byte owners, so neither capacity exhaustion nor stale
        // admission is allowed while other threads commit and refund roots.
        auto grant = ledger.reserve_current({{"shared", 1}});
        if (!grant) { ++failures; continue; }
        auto root = grant.value()->commit("shared", 1);
        if (!root) ++failures;
        std::this_thread::yield();
      }
    });
  for (auto& worker : workers) worker.join();
  EXPECT_EQ(failures, 0u);
  const auto final = ledger.snapshot();
  EXPECT_EQ(final.groups[0].committed_bytes, 0u);
  EXPECT_EQ(final.groups[0].unbacked_reserved_bytes, 0u);
  EXPECT_EQ(final.groups[0].system_available_bytes, 16u);
  EXPECT_FALSE(ledger.reserve({{"shared", 1}}, old));
}
TEST(CapacityLedger, ConcurrentRequestsRespectHardLimitWithoutLeakingGrants) {
  CapacityLedger ledger({group("shared", 64, 64)});
  std::atomic<int> accepted = 0, failed = 0, violations = 0;
  std::vector<std::thread> workers;
  for (int t = 0; t < 8; ++t)
    workers.emplace_back([&] {
      for (int i = 0; i < 100; ++i) {
        auto r = ledger.reserve({{"shared", 16}}, ledger.snapshot().epoch);
        if (!r) {
          ++failed;
          continue;
        }
        auto c = r.value()->commit("shared", 16);
        if (!c) {
          ++failed;
          continue;
        }
        ++accepted;
        auto snap = ledger.snapshot().groups[0];
        if (snap.committed_bytes + snap.unbacked_reserved_bytes > 64)
          ++violations;
      }
    });
  for (auto& w : workers)
    w.join();
  EXPECT_GT(accepted.load(), 0);
  EXPECT_EQ(violations.load(), 0);
  EXPECT_EQ(ledger.snapshot().groups[0].committed_bytes, 0u);
  EXPECT_EQ(ledger.snapshot().groups[0].unbacked_reserved_bytes, 0u);
}
TEST(CapacityLedger, LiveChargesOutlivePublicLedgerOwner) {
  std::shared_ptr<CapacityCharge> charge;
  {
    CapacityLedger ledger({group("a")});
    auto grant = ledger.reserve({{"a", 100}}, ledger.snapshot().epoch);
    charge = std::move(grant.value()->commit("a", 100)).value();
  }
  EXPECT_TRUE(charge->transition(CapacityState::live, CapacityState::idle_available, 100));
  charge.reset();
}
TEST(CapacityLedger, SharedDomainLimitCoversAllMemberPoolsAndAllocationPaths) {
  CapacityLedger ledger({group("a"), group("b")}, {{"domain", {"a", "b"}, 512}});
  auto first = ledger.reserve({{"a", 300}}, ledger.snapshot().epoch);
  ASSERT_TRUE(first);
  EXPECT_EQ(ledger.snapshot().constraints[0].remaining_bytes, 212u);
  EXPECT_FALSE(ledger.reserve({{"b", 213}}, ledger.snapshot().epoch));
  auto charged = first.value()->commit("a", 300);
  ASSERT_TRUE(charged);
  first.value().reset();
  EXPECT_EQ(ledger.snapshot().constraints[0].remaining_bytes, 212u);
  auto next = ledger.reserve({{"b", 212}}, ledger.snapshot().epoch);
  ASSERT_TRUE(next);
  EXPECT_EQ(ledger.snapshot().constraints[0].remaining_bytes, 0u);
  next.value().reset();
  charged.value().reset();
  EXPECT_EQ(ledger.snapshot().constraints[0].remaining_bytes, 512u);
}
} // namespace
} // namespace vecops::memory::model::details
