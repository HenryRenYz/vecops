// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/details/SharedCapacityAuthority.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace vecops::memory::model::details {
namespace {
using U = std::uint64_t;
template <class T>
T take(Result<T> r) {
  if (!r) throw std::runtime_error(std::get<Error>(r.state).code + ": " + std::get<Error>(r.state).message);
  return std::move(r).value();
}
struct Directory {
  std::filesystem::path path;
  Directory() {
    const auto base = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                      "build/memory-migration";
    std::filesystem::create_directories(base);
    auto pattern = (base / "shared-capacity-XXXXXX").string();
    auto* result = ::mkdtemp(pattern.data());
    if (!result) throw std::runtime_error("mkdtemp failed");
    path = result;
  }
  ~Directory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
struct Fixture {
  Directory directory;
  std::vector<SharedCapacityGroup> groups{{"uma", 100, 80}, {"cowos_superdie", 100, 80}, {"other", 100, 80}};
  std::vector<BudgetConstraint> constraints;
  std::map<std::string, U> free{{"uma", 100}, {"cowos_superdie", 100}, {"other", 100}};
  std::shared_ptr<SharedCapacityAuthority> open(std::string participant = "rank-0") {
    return take(SharedCapacityAuthority::open(directory.path, groups, constraints, std::move(participant)));
  }
  std::unique_ptr<SharedCapacityGrant> reserve(const std::shared_ptr<SharedCapacityAuthority>& a,
                                               std::map<std::string, U> request) {
    return take(a->reserve(request, take(a->snapshot()).epoch, free));
  }
};
const SharedCapacityGroupSnapshot& group(const SharedCapacitySnapshot& snapshot, const std::string& id) {
  auto it = std::find_if(snapshot.groups.begin(), snapshot.groups.end(), [&](const auto& g) { return g.id == id; });
  if (it == snapshot.groups.end()) throw std::runtime_error("missing group");
  return *it;
}
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void write(const std::filesystem::path& path, const std::string& text) {
  std::ofstream file(path, std::ios::trunc); file << text;
  if (!file) throw std::runtime_error("metadata write failed");
}
void byte(int fd, char value) {
  for (;;) {
    const auto n = ::write(fd, &value, 1);
    if (n == 1) return;
    if (n < 0 && errno == EINTR) continue;
    throw std::runtime_error("pipe write failed");
  }
}
char byte(int fd) {
  char value;
  for (;;) {
    const auto n = ::read(fd, &value, 1);
    if (n == 1) return value;
    if (n < 0 && errno == EINTR) continue;
    throw std::runtime_error("pipe read failed");
  }
}
TEST(SharedCapacityAuthority, GrantCommitCancelAndFinalAliasReleaseHaveOnePhysicalCharge) {
  Fixture f;
  auto a = f.open(); auto b = f.open("rank-1");
  auto grant = f.reserve(a, {{"uma", 70}});
  auto snapshot = take(b->snapshot());
  EXPECT_EQ(group(snapshot, "uma").pending_bytes, 70u);
  EXPECT_EQ(group(snapshot, "uma").hard_remaining_bytes, 10u);
  auto root = take(grant->commit("uma", 30, "canonical-root"));
  snapshot = take(b->snapshot());
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 30u);
  EXPECT_EQ(group(snapshot, "uma").pending_bytes, 40u);
  EXPECT_EQ(snapshot.roots.front().owner.participant, "rank-0");
  EXPECT_EQ(snapshot.roots.front().owner.state, SharedCapacityOwnerState::alive);
  grant.reset();
  auto alias = root;
  const auto epoch = take(a->snapshot()).epoch;
  for (unsigned i = 0; i < 10000; ++i) { auto cache_slice = alias; (void)cache_slice; }
  root.reset();
  snapshot = take(a->snapshot());
  EXPECT_EQ(snapshot.epoch, epoch); // cache suballocation never touches authority
  EXPECT_EQ(group(snapshot, "uma").pending_bytes, 0u);
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 30u);
  alias.reset();
  snapshot = take(b->snapshot());
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 0u);
  EXPECT_TRUE(snapshot.roots.empty()); EXPECT_TRUE(snapshot.grants.empty());
  EXPECT_FALSE(a->cleanup_error());
}
TEST(SharedCapacityAuthority, CoordinatorIdentityUsesTheSameLockInodeAcrossPathAliases) {
  Fixture f;
  auto first = f.open();
  auto alias = take(SharedCapacityAuthority::open(f.directory.path / ".", f.groups, f.constraints, "rank-1"));
  Fixture other;
  auto independent = other.open();
  EXPECT_FALSE(first->namespace_id().empty());
  EXPECT_EQ(first->namespace_id(), alias->namespace_id());
  EXPECT_NE(first->namespace_id(), independent->namespace_id());
}
TEST(SharedCapacityAuthority, CurrentAdmissionSerializesSamplingWithMaterializationCommit) {
  using namespace std::chrono_literals;
  Fixture f;
  auto first = f.open(), second = f.open("rank-1");
  auto pending = f.reserve(first, {{"uma", 30}});
  std::promise<void> sampling, release_sample, committing;
  auto release = release_sample.get_future().share();
  auto admission = std::async(std::launch::async, [&] {
    return second->reserve_current({{"uma", 50}}, [&]() -> Result<std::map<std::string, U>> {
      sampling.set_value();
      release.wait();
      // The pending owner's physical pages already exist, but its commit is
      // blocked until sampling/admission complete: conservative double counting.
      return {std::map<std::string, U>{{"uma", 70}}};
    });
  });
  sampling.get_future().wait();
  auto commit = std::async(std::launch::async, [&] {
    committing.set_value();
    return pending->commit("uma", 30, "materializing-root");
  });
  committing.get_future().wait();
  const auto blocked = commit.wait_for(30ms);
  release_sample.set_value(); // always release before assertions/future cleanup
  EXPECT_EQ(blocked, std::future_status::timeout);
  auto rejected = admission.get();
  ASSERT_FALSE(rejected);
  EXPECT_EQ(std::get<Error>(rejected.state).code, "shared_capacity.physical_free");
  const auto& message=std::get<Error>(rejected.state).message;
  EXPECT_NE(message.find("group=uma"),std::string::npos);
  EXPECT_NE(message.find("requested_bytes=50"),std::string::npos);
  EXPECT_NE(message.find("sampled_free_bytes=70"),std::string::npos);
  EXPECT_NE(message.find("pending_bytes=30"),std::string::npos);
  auto root = take(commit.get());
  const auto snapshot = take(second->snapshot());
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 30u);
  EXPECT_EQ(group(snapshot, "uma").pending_bytes, 0u);
  // Use a request within both the current hard remainder and physical free.
  auto next = take(second->reserve_current({{"uma", 50}}, []() -> Result<std::map<std::string, U>> {
    return {std::map<std::string, U>{{"uma", 70}}};
  }));
  EXPECT_EQ(group(take(second->snapshot()), "uma").pending_bytes, 50u);
}
TEST(SharedCapacityAuthority, CurrentAdmissionPropagatesSamplerFailureWithoutMutationOrRetry) {
  Fixture f;
  auto a = f.open();
  const auto before = take(a->snapshot());
  unsigned calls = 0;
  auto rejected = a->reserve_current({{"uma", 1}}, [&]() -> Result<std::map<std::string, U>> {
    ++calls;
    return {Error{"probe.unavailable", "free-capacity source unavailable"}};
  });
  ASSERT_FALSE(rejected);
  EXPECT_EQ(std::get<Error>(rejected.state).code, "probe.unavailable");
  EXPECT_EQ(calls, 1u);
  const auto after = take(a->snapshot());
  EXPECT_EQ(after.epoch, before.epoch);
  EXPECT_TRUE(after.grants.empty());
  EXPECT_TRUE(after.roots.empty());
  EXPECT_FALSE(a->reserve_current({{"uma", 1}}, {}));
  auto grant = take(a->reserve_current({{"uma", 1}}, [&]() -> Result<std::map<std::string, U>> {
    return {f.free};
  }));
  // The automatic path must not weaken the explicit planner's stale contract.
  auto stale = a->reserve({{"uma", 1}}, before.epoch, f.free);
  ASSERT_FALSE(stale);
  EXPECT_EQ(std::get<Error>(stale.state).code, "shared_capacity.stale");
}
TEST(SharedCapacityAuthority, AllGroupsAndOverlappingConstraintsAreAtomic) {
  Fixture f;
  f.constraints = {{"fabric-A", {"uma", "cowos_superdie"}, 60},
                   {"fabric-B", {"cowos_superdie", "other"}, 50}};
  auto a = f.open(); auto b = f.open("rank-1");
  auto grant = f.reserve(a, {{"uma", 10}, {"cowos_superdie", 30}, {"other", 10}});
  auto before = take(a->snapshot());
  auto rejected = b->reserve({{"uma", 1}, {"other", 11}}, before.epoch, f.free);
  EXPECT_FALSE(rejected);
  auto after = take(a->snapshot());
  EXPECT_EQ(after.epoch, before.epoch);
  EXPECT_EQ(group(after, "uma").pending_bytes, 10u);
  EXPECT_EQ(group(after, "other").pending_bytes, 10u);
  EXPECT_EQ(after.constraints[0].remaining_bytes, 20u);
  EXPECT_EQ(after.constraints[1].remaining_bytes, 10u);
  EXPECT_TRUE(b->reserve({{"uma", 20}, {"other", 10}}, after.epoch, f.free));
}
TEST(SharedCapacityAuthority, PhysicalFreeSamplesSubtractPendingButNotCommittedAgain) {
  Fixture f; auto a = f.open(); auto b = f.open("rank-1");
  auto pending = take(a->reserve({{"uma", 30}}, take(a->snapshot()).epoch, {{"uma", 50}}));
  auto snapshot = take(b->snapshot());
  EXPECT_FALSE(b->reserve({{"uma", 21}}, snapshot.epoch, {{"uma", 50}}));
  auto second = take(b->reserve({{"uma", 20}}, snapshot.epoch, {{"uma", 50}}));
  auto root = take(pending->commit("uma", 30, "physical-30"));
  pending.reset(); second.reset();
  // OS reports 50 free after the committed 30; only new pending is subtracted.
  EXPECT_TRUE(b->reserve({{"uma", 50}}, take(b->snapshot()).epoch, {{"uma", 50}}));
  EXPECT_FALSE(b->reserve({{"uma", 1}}, take(b->snapshot()).epoch, {}));
  EXPECT_FALSE(b->reserve({{"uma", 1}}, take(b->snapshot()).epoch, {{"uma", 101}}));
}
TEST(SharedCapacityAuthority, StaleSnapshotsAndDuplicatePhysicalCommitAreRejected) {
  Fixture f; auto a = f.open(); auto b = f.open("rank-1");
  const auto epoch = take(b->snapshot()).epoch;
  auto first = f.reserve(a, {{"uma", 20}});
  auto stale = b->reserve({{"uma", 1}}, epoch, f.free);
  ASSERT_FALSE(stale); EXPECT_EQ(std::get<Error>(stale.state).code, "shared_capacity.stale");
  auto second = f.reserve(b, {{"uma", 20}});
  auto root = take(first->commit("uma", 20, "one-physical-backing"));
  EXPECT_FALSE(second->commit("uma", 20, "one-physical-backing"));
  EXPECT_FALSE(second->commit("uma", 21, "other-backing"));
  EXPECT_EQ(group(take(b->snapshot()), "uma").pending_bytes, 20u);
  EXPECT_FALSE(first->commit("uma", 1, "already-consumed"));
}
TEST(SharedCapacityAuthority, ActiveConfigurationMismatchFailsAndEmptyReconfigurationAdvancesEpoch) {
  Fixture f; auto a = f.open();
  auto grant = f.reserve(a, {{"uma", 10}});
  auto changed = f.groups; changed[0].hard_limit_bytes = 70;
  auto rejected = SharedCapacityAuthority::open(f.directory.path, changed, {}, "rank-1");
  ASSERT_FALSE(rejected); EXPECT_EQ(std::get<Error>(rejected.state).code, "shared_capacity.config_active");
  grant.reset(); const auto before = take(a->snapshot()).epoch;
  auto newer = take(SharedCapacityAuthority::open(f.directory.path, changed, {}, "rank-1"));
  EXPECT_GT(take(newer->snapshot()).epoch, before);
  EXPECT_FALSE(a->snapshot());
  // Configuration ordering is irrelevant when another rank joins.
  std::reverse(changed.begin(), changed.end());
  EXPECT_TRUE(SharedCapacityAuthority::open(f.directory.path, changed, {}, "rank-2"));
}
TEST(SharedCapacityAuthority, PrivateFilesAndStableLockInodeAreEnforced) {
  Fixture f; auto a = f.open();
  struct stat lock_before{}, lock_after{}, metadata{}, directory{};
  ASSERT_EQ(::stat((f.directory.path / "lock").c_str(), &lock_before), 0);
  { auto grant = f.reserve(a, {{"uma", 10}}); }
  ASSERT_EQ(::stat((f.directory.path / "lock").c_str(), &lock_after), 0);
  ASSERT_EQ(::stat((f.directory.path / "capacity.toml").c_str(), &metadata), 0);
  ASSERT_EQ(::stat(f.directory.path.c_str(), &directory), 0);
  EXPECT_EQ(lock_before.st_ino, lock_after.st_ino);
  EXPECT_EQ(metadata.st_mode & 0777, 0600u); EXPECT_EQ(directory.st_mode & 0777, 0700u);
  ASSERT_EQ(::chmod(f.directory.path.c_str(), 0755), 0);
  EXPECT_FALSE(SharedCapacityAuthority::open(f.directory.path, f.groups, {}, "rank-2"));
  ASSERT_EQ(::chmod(f.directory.path.c_str(), 0700), 0);
}
TEST(SharedCapacityAuthority, InvalidMetadataAndOtherHostBootAreNeverSilentlyReset) {
  Fixture f; auto a = f.open(); const auto path = f.directory.path / "capacity.toml";
  const auto good = read(path);
  write(path, "broken = ["); EXPECT_FALSE(a->snapshot());
  write(path, good + "\nunknown_metadata = true\n"); EXPECT_FALSE(a->snapshot());
  auto foreign = good;
  const auto start = foreign.find("boot_id = '");
  // toml++ may choose either quote style; replace the whole assignment.
  (void)start;
  const auto line = foreign.find("boot_id = "); ASSERT_NE(line, std::string::npos);
  foreign.replace(line, foreign.find('\n', line) - line, "boot_id = 'other-host-boot'");
  write(path, foreign);
  auto rejected = SharedCapacityAuthority::open(f.directory.path, f.groups, {}, "rank-1");
  ASSERT_FALSE(rejected); EXPECT_EQ(std::get<Error>(rejected.state).code, "shared_capacity.host");
  EXPECT_FALSE(a->snapshot());
  write(path, good); EXPECT_TRUE(a->snapshot());
}
TEST(SharedCapacityAuthority, ConcurrentThreadsCannotReserveMoreThanTheSharedCap) {
  Fixture f; auto a = f.open(); auto b = f.open("rank-1");
  std::vector<std::future<std::unique_ptr<SharedCapacityGrant>>> tasks;
  for (unsigned i = 0; i < 8; ++i)
    tasks.push_back(std::async(std::launch::async, [&, i]() -> std::unique_ptr<SharedCapacityGrant> {
      const auto& authority = i % 2 ? a : b;
      for (;;) {
        auto snap = take(authority->snapshot());
        auto grant = authority->reserve({{"uma", 20}}, snap.epoch, f.free);
        if (grant) return std::move(grant).value();
        if (std::get<Error>(grant.state).code != "shared_capacity.stale") return {};
      }
    }));
  std::vector<std::unique_ptr<SharedCapacityGrant>> winners;
  for (auto& task : tasks) if (auto grant = task.get()) winners.push_back(std::move(grant));
  EXPECT_EQ(winners.size(), 4u);
  EXPECT_EQ(group(take(a->snapshot()), "uma").pending_bytes, 80u);
}
TEST(SharedCapacityAuthority, ForkedRanksJointlyOwnOneGlobalHardCap) {
  Fixture f; auto parent = f.open("parent");
  std::array<int, 2> results{}, gate{};
  ASSERT_EQ(::pipe(results.data()), 0); ASSERT_EQ(::pipe(gate.data()), 0);
  std::vector<pid_t> children;
  for (unsigned rank = 0; rank < 8; ++rank) {
    const auto child = ::fork(); ASSERT_GE(child, 0);
    if (!child) {
      ::close(results[0]); ::close(gate[1]);
      const int status = [&] {
        try {
          auto authority = f.open("child-" + std::to_string(rank));
          std::unique_ptr<SharedCapacityGrant> grant;
          for (;;) {
            auto snapshot = take(authority->snapshot());
            auto attempt = authority->reserve({{"uma", 20}}, snapshot.epoch, f.free);
            if (attempt) { grant = std::move(attempt).value(); break; }
            if (std::get<Error>(attempt.state).code != "shared_capacity.stale") break;
          }
          std::shared_ptr<SharedCapacityCharge> root;
          if (grant && rank % 2) root = take(grant->commit("uma", 20, "child-root-" + std::to_string(rank)));
          byte(results[1], grant ? 'W' : 'L');
          byte(gate[0]);
          return 0;
        } catch (...) { byte(results[1], 'E'); return 1; }
      }();
      ::_exit(status);
    }
    children.push_back(child);
  }
  ::close(results[1]); ::close(gate[0]);
  unsigned winners = 0;
  for (unsigned i = 0; i < children.size(); ++i) {
    const auto result = byte(results[0]); EXPECT_NE(result, 'E'); winners += result == 'W';
  }
  EXPECT_EQ(winners, 4u);
  const auto snap = take(parent->snapshot());
  EXPECT_EQ(group(snap, "uma").committed_bytes + group(snap, "uma").pending_bytes, 80u);
  for (unsigned i = 0; i < children.size(); ++i) byte(gate[1], 'X');
  ::close(results[0]); ::close(gate[1]);
  for (const auto child : children) {
    int status; ASSERT_EQ(::waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  auto after = take(parent->snapshot());
  EXPECT_EQ(group(after, "uma").hard_remaining_bytes, 80u);
  EXPECT_TRUE(after.grants.empty()); EXPECT_TRUE(after.roots.empty());
}
TEST(SharedCapacityAuthority, DeadOwnerReceiptsRemainChargedAndInheritedHandlesCannotRefund) {
  Fixture f; auto parent = f.open("parent");
  const auto child = ::fork(); ASSERT_GE(child, 0);
  if (!child) {
    if (parent->snapshot()) ::_exit(2); // inherited authority is not child authority
    try {
      auto authority = f.open("dead-rank");
      auto grant = f.reserve(authority, {{"uma", 60}});
      auto root = take(grant->commit("uma", 40, "orphan-root"));
      (void)root;
      ::_exit(0); // simulate crash: neither destructor may refund
    } catch (...) { ::_exit(1); }
  }
  int status; ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  const auto snapshot = take(parent->snapshot());
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 40u);
  EXPECT_EQ(group(snapshot, "uma").pending_bytes, 20u);
  ASSERT_EQ(snapshot.roots.size(), 1u); ASSERT_EQ(snapshot.grants.size(), 1u);
  EXPECT_EQ(snapshot.roots.front().owner.state, SharedCapacityOwnerState::owner_dead);
  EXPECT_EQ(snapshot.grants.front().owner.state, SharedCapacityOwnerState::owner_dead);
  EXPECT_FALSE(parent->reserve({{"uma", 21}}, snapshot.epoch, f.free));
}
TEST(SharedCapacityAuthority, FullUint64CapacitiesRemainExactInMetadata) {
  Fixture f; const U max = std::numeric_limits<U>::max();
  f.groups = {{"opaque", max, max}}; f.free = {{"opaque", max}};
  auto a = f.open();
  auto grant = f.reserve(a, {{"opaque", max - 1}});
  EXPECT_EQ(group(take(a->snapshot()), "opaque").pending_bytes, max - 1);
  auto root = take(grant->commit("opaque", max - 1, "large-root"));
  grant.reset();
  EXPECT_EQ(group(take(a->snapshot()), "opaque").committed_bytes, max - 1);
  root.reset(); EXPECT_EQ(group(take(a->snapshot()), "opaque").hard_remaining_bytes, max);
}
TEST(SharedCapacityAuthority, ChargesKeepTheirAuthorityAliveAfterPublicHandleDestruction) {
  Fixture f; auto authority = f.open();
  auto grant = f.reserve(authority, {{"uma", 40}});
  auto root = take(grant->commit("uma", 40, "long-lived-root"));
  grant.reset(); authority.reset();
  auto observer = f.open("observer");
  EXPECT_EQ(group(take(observer->snapshot()), "uma").committed_bytes, 40u);
  root.reset();
  EXPECT_EQ(group(take(observer->snapshot()), "uma").hard_remaining_bytes, 80u);
}
TEST(SharedCapacityAuthority, RefundFailureReportsDiagnosticAndPreservesPersistedCharge) {
  Fixture f; auto authority = f.open();
  auto grant = f.reserve(authority, {{"uma", 40}});
  auto root = take(grant->commit("uma", 40, "retained-on-error"));
  grant.reset();
  const auto path = f.directory.path / "capacity.toml";
  const auto good = read(path);
  write(path, "broken = [");
  root.reset();
  EXPECT_TRUE(authority->cleanup_error());
  write(path, good);
  auto snapshot = take(authority->snapshot());
  EXPECT_EQ(group(snapshot, "uma").committed_bytes, 40u);
  EXPECT_EQ(snapshot.roots.front().physical_identity, "retained-on-error");
}
} // namespace
} // namespace vecops::memory::model::details
