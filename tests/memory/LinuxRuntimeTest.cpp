// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxRuntime.h"
#include "vecops/memory/HostEvents.h"
#include "vecops/memory/LinuxBackend.h"
#include "vecops/memory/RuntimeBridge.h"
#include "vecops/memory/details/SharedCapacityAuthority.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
template <class T>
T take(Result<T> r) {
  if (!r)
    throw std::runtime_error(std::get<Error>(r.state).message);
  return std::move(r).value();
}
void take(Result<void> r) {
  if (!r)
    throw std::runtime_error(r.error->message);
}
struct Fixture {
  U q = static_cast<U>(::sysconf(_SC_PAGESIZE));
  std::string participant;
  std::unique_ptr<MemoryRuntime> runtime;
  ResourceView view;
  Fixture(U pages = 32, std::optional<std::string> directory = {}, std::optional<U> shared_limit = {},
          std::string owner = "p0") : participant(std::move(owner)) {
    LinuxDiscovery d;
    d.graph.agents = {{"executor", {"host_load_store"}, "host"}};
    d.graph.capacity_groups = {{"cap-a", pages * q}, {"cap-b", pages * q}};
    d.graph.pools = {{"a", pages * q, "host", q, {q}, {}, {}, "cap-a"},
                     {"b", pages * q, "host", q, {q}, {}, {}, "cap-b"}};
    d.graph.domains = {{"domain", "siblings", {"a", "b"}}};
    for (const auto& p : d.graph.pools) {
      d.pool_bindings.push_back({p.id, {}});
      d.pool_available_bytes[p.id] = pages * q;
      d.graph.access.push_back({p.id,
                                "cpu-direct",
                                "executor",
                                p.id,
                                {AccessMode::read, AccessMode::write},
                                {},
                                {},
                                MappingMode::direct,
                                {},
                                {},
                                {},
                                {},
                                {}});
    }
    Method method{"copy", true, 10.0, 0.0, {}, {}};
    for (const auto& a : d.graph.pools)
      for (const auto& b : d.graph.pools)
        method.routes.push_back({a.id + b.id, {}, {}, a.id, b.id, {}});
    d.graph.methods.push_back(method);
    d.scope = {{"executor"}, {"a", "b"}, {"a", "b"}, {{"executor", 1}}};
    LinuxRuntimeOptions options;
    options.participant = participant;
    options.capacity_directory = std::move(directory);
    if (shared_limit)
      options.shared_limits = {{"joint", {"cap-a", "cap-b"}, *shared_limit}};
    options.cpu_queues = {{"main", "executor"}};
    options.cpu_copy_methods = {"copy"};
    runtime = take(make_linux_runtime(std::move(d), options));
    view = take(runtime->view(take(runtime->current_scope())));
  }
  BackingRequest request(std::string id = "backing", std::string pool = "a", U pages = 1) {
    return {id, participant, pool, pages * q, q, BackingMode::shareable_remappable, {}};
  }
  MemoryRecipe recipe(U pages = 2) {
    ResolvedPlacement placement{"placement-a", "host", q, {{0, pages * q, "a"}}};
    return {participant,
            {},
            1,
            {},
            {{{"region", "host", pages * q, q}}, {{"slot", "region", 0, pages * q, placement, {}}}},
            {request("backing", "a", pages)},
            {{"region", 0, "backing", 0, pages * q, {AccessMode::read, AccessMode::write}}}};
  }
  ArenaOptions options(U initial = 0, U hard = 16) {
    return {
      "domain", {}, Limit{hard * q}, Preallocate{initial * q}, 4 * q, 32 * q, q, BackingMode::shareable_remappable,
      true};
  }
  std::shared_ptr<MemoryArena> arena(U initial = 0, U hard = 16) {
    return take(runtime->create_arena(view, options(initial, hard)));
  }
  U committed() {
    U n = 0;
    for (const auto& b : runtime->snapshot(view).groups)
      n += b.committed_bytes;
    return n;
  }
  std::unique_ptr<MemoryBinding> bind(MemoryRecipe r) {
    auto p = take(runtime->prepare_addresses(view, r.addresses));
    auto lease = take(runtime->reserve(view, r, runtime->snapshot(view).epoch));
    auto backing = take(runtime->acquire_backing(view, r, *lease));
    return take(runtime->bind(view, r, p, backing));
  }
};
struct Directory {
  std::filesystem::path path;
  Directory() {
    const auto base = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                      "build/memory-migration";
    std::filesystem::create_directories(base);
    auto pattern = (base / "runtime-authority-XXXXXX").string();
    auto* result = ::mkdtemp(pattern.data());
    if (!result) throw std::runtime_error("mkdtemp failed");
    path = result;
  }
  ~Directory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

// Each child owns a separately opened runtime. The parent advances every rank
// together and inspects the physical authority only while every root is held.
// A failed child/expectation cannot leave another rank blocked in a barrier.
struct RuntimeRanks {
  std::vector<std::array<int, 2>> sockets;
  std::vector<pid_t> pids;
  explicit RuntimeRanks(unsigned count) : sockets(count, {-1, -1}) {
    for (auto& pair : sockets)
      if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair.data())) {
        for (auto& opened : sockets)
          for (auto fd : opened)
            if (fd >= 0) ::close(fd);
        throw std::runtime_error("runtime rank socketpair failed");
      }
  }
  ~RuntimeRanks() {
    for (auto pid : pids)
      if (pid > 0) ::kill(pid, SIGKILL);
    for (auto pid : pids)
      if (pid > 0)
        while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    for (auto& pair : sockets)
      for (auto fd : pair)
        if (fd >= 0) ::close(fd);
  }
  static void send(int fd, char value) {
    if (::send(fd, &value, 1, MSG_NOSIGNAL) != 1)
      throw std::runtime_error("runtime rank send failed");
  }
  static char receive(int fd) {
    pollfd poller{fd, POLLIN, 0};
    int ready;
    do { ready = ::poll(&poller, 1, 30000); } while (ready < 0 && errno == EINTR);
    char value = 0;
    if (ready != 1 || ::recv(fd, &value, 1, 0) != 1)
      throw std::runtime_error("runtime rank response missing/timed out");
    return value;
  }
  void broadcast(char value) {
    for (const auto& pair : sockets) send(pair[0], value);
  }
  void expect(char value) {
    for (unsigned i = 0; i < sockets.size(); ++i)
      if (receive(sockets[i][0]) != value)
        throw std::runtime_error("runtime rank " + std::to_string(i) + " failed its allocation phase");
  }
};

void concurrent_runtime_roots(unsigned rank_count) {
  constexpr unsigned rounds = 3, cycles = 2;
  Directory directory;
  const U q = static_cast<U>(::sysconf(_SC_PAGESIZE));
  const U group_capacity = rank_count * rounds * q;
  Fixture observer(rank_count * rounds, directory.path.string(), 2 * group_capacity, "observer");
  auto authority = take(details::SharedCapacityAuthority::open(
    directory.path, {{"cap-a", group_capacity, group_capacity}, {"cap-b", group_capacity, group_capacity}},
    {{"joint", {"cap-a", "cap-b"}, 2 * group_capacity}}, "auditor"));
  RuntimeRanks ranks(rank_count);
  for (unsigned rank = 0; rank < rank_count; ++rank) {
    const auto pid = ::fork();
    if (pid < 0) throw std::runtime_error("runtime rank fork failed");
    if (!pid) {
      for (unsigned i = 0; i < rank_count; ++i) {
        ::close(ranks.sockets[i][0]);
        if (i != rank) ::close(ranks.sockets[i][1]);
      }
      const int channel = ranks.sockets[rank][1];
      try {
        {
          Fixture own(rank_count * rounds, directory.path.string(), 2 * group_capacity,
                      "rank-" + std::to_string(rank));
          std::vector<Allocation> roots;
          RuntimeRanks::send(channel, 'I');
          for (unsigned cycle = 0; cycle < cycles; ++cycle) {
            for (unsigned round = 0; round < rounds; ++round) {
              if (RuntimeRanks::receive(channel) != 'A') throw std::runtime_error("unexpected allocate command");
              for (unsigned part = 0; part < 2; ++part) {
                // Both pools share one authority/constraint, but ranks reverse
                // their order so unrelated group commits also contend.
                const auto pool = (rank + round + part) % 2 ? "a" : "b";
                auto root = take(own.runtime->allocate(
                  own.view, own.request("root-" + std::to_string(cycle) + "-" + std::to_string(round) +
                                          "-" + std::to_string(part), pool)));
                std::memset(take(root.data()), static_cast<int>(rank + 1), q);
                roots.push_back(std::move(root));
              }
              RuntimeRanks::send(channel, 'H');
            }
            if (RuntimeRanks::receive(channel) != 'R') throw std::runtime_error("unexpected release command");
            roots.clear();
            RuntimeRanks::send(channel, 'F');
          }
        }
        ::close(channel);
        ::_exit(0);
      } catch (const std::exception& e) {
        ::dprintf(STDERR_FILENO, "runtime rank %u: %s\n", rank, e.what());
        ::send(channel, "E", 1, MSG_NOSIGNAL);
        ::_exit(1);
      }
    }
    ranks.pids.push_back(pid);
    ::close(ranks.sockets[rank][1]);
    ranks.sockets[rank][1] = -1;
  }
  ranks.expect('I');
  for (unsigned cycle = 0; cycle < cycles; ++cycle) {
    const auto explicit_epoch = observer.runtime->snapshot(observer.view).epoch;
    for (unsigned round = 0; round < rounds; ++round) {
      ranks.broadcast('A');
      ranks.expect('H');
      const auto snapshot = take(authority->snapshot());
      ASSERT_EQ(snapshot.groups.size(), 2u);
      for (const auto& group : snapshot.groups) {
        EXPECT_EQ(group.committed_bytes, rank_count * (round + 1) * q);
        EXPECT_EQ(group.pending_bytes, 0u);
        EXPECT_EQ(group.hard_remaining_bytes, rank_count * (rounds - round - 1) * q);
      }
      EXPECT_TRUE(snapshot.grants.empty());
      EXPECT_EQ(snapshot.roots.size(), 2 * rank_count * (round + 1));
      ASSERT_EQ(snapshot.constraints.size(), 1u);
      EXPECT_EQ(snapshot.constraints.front().remaining_bytes, 2 * rank_count * (rounds - round - 1) * q);
      if (!round) {
        auto stale = observer.runtime->reserve(observer.view, observer.recipe(1), explicit_epoch);
        ASSERT_FALSE(stale);
        EXPECT_EQ(std::get<Error>(stale.state).message, "stale budget snapshot");
      }
    }
    EXPECT_FALSE(observer.runtime->allocate(observer.view, observer.request("over-capacity")));
    ranks.broadcast('R');
    ranks.expect('F');
    const auto empty = take(authority->snapshot());
    for (const auto& group : empty.groups) {
      EXPECT_EQ(group.committed_bytes, 0u);
      EXPECT_EQ(group.pending_bytes, 0u);
      EXPECT_EQ(group.hard_remaining_bytes, group_capacity);
    }
    EXPECT_TRUE(empty.grants.empty());
    EXPECT_TRUE(empty.roots.empty());
    EXPECT_EQ(empty.constraints.front().remaining_bytes, 2 * group_capacity);
  }
  for (auto& pid : ranks.pids) {
    int status = 0;
    pid_t waited;
    do { waited = ::waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
    ASSERT_EQ(waited, pid);
    pid = -1;
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
  }
  EXPECT_EQ(observer.committed(), 0u);
}
TEST(LinuxRuntime, PrivateMovableCapabilityIsModeAwareAndBridgeRejectsOldDataAbi) {
  Fixture f;std::string pools[]={"a"};
  auto fd=take(f.runtime->mapping_capabilities(f.view,pools));EXPECT_TRUE(fd.multiple_mappings);EXPECT_TRUE(fd.shareable);
  auto c=take(f.runtime->mapping_capabilities(f.view,pools,BackingMode::private_movable));
  EXPECT_EQ(c.reserve_without_backing,LinuxPageStore::private_movable_supported());
  EXPECT_FALSE(c.multiple_mappings);EXPECT_FALSE(c.partial_backing_map);EXPECT_FALSE(c.shareable);
  EXPECT_FALSE(f.runtime->mapping_capabilities(f.view,pools,static_cast<BackingMode>(99)));
  auto options=f.options();options.backing=BackingMode::private_movable;EXPECT_FALSE(f.runtime->create_arena(f.view,options));
  bridge::Descriptor old{};old.struct_size=sizeof(old);old.bridge_version=1;old.runtime_interface_revision=1;
  old.runtime_type=bridge::type_tag;EXPECT_THROW(bridge::validate(old),std::invalid_argument);
}
TEST(LinuxRuntime, PrivateMovablePlacementPreservesLogicalSubpageRounding) {
  if(!LinuxPageStore::private_movable_supported())GTEST_SKIP();
  Fixture f;PlacementPlan plan{"host",f.q,{{"a",1}},{},0};
  const AccessMode access[]={AccessMode::read,AccessMode::write};
  auto p=take(f.runtime->resolve_placement(f.view,plan,256,BackingMode::private_movable,access));
  EXPECT_EQ(p.page_bytes,f.q);ASSERT_EQ(p.segments.size(),1u);EXPECT_EQ(p.segments[0].size_bytes,f.q);
}
TEST(LinuxRuntime, PrivateMovableRuntimeExternalAliasAndAcquireParkingRefund) {
  if(!LinuxPageStore::private_movable_supported())GTEST_SKIP();
  Fixture f;auto r=f.recipe();r.backings.front().mode=BackingMode::private_movable;
  auto prepared=take(f.runtime->prepare_addresses(f.view,r.addresses));
  auto lease=take(f.runtime->reserve(f.view,r,f.runtime->snapshot(f.view).epoch));
  auto backing=take(f.runtime->acquire_backing(f.view,r,*lease));
  EXPECT_EQ(f.runtime->snapshot(f.view).groups.front().committed_bytes,2*f.q);
  auto bound=take(f.runtime->bind(f.view,r,prepared,backing));
  auto* data=static_cast<unsigned char*>(take(bound->slot("slot")));std::memset(data,11,2*f.q);
  ExternalMemoryView ext{};ext.id="alias";ext.address={"host",reinterpret_cast<U>(data)};ext.bytes=f.q;ext.owner=backing;
  auto imported=take(f.runtime->import_memory(f.view,ext));
  InvocationContext context{{"same-va",bound->generation()},{},{{"input",imported}}};
  auto env=take(f.runtime->resolve_operands(f.view,*bound,context));
  ASSERT_EQ(env.memories.size(),2u);EXPECT_EQ(env.memories[0].spans[0].physical_identity,env.memories[1].spans[0].physical_identity);
  EXPECT_EQ(env.memories[0].spans[0].physical_offset_bytes,env.memories[1].spans[0].physical_offset_bytes);
  MemoryCommand command{"copy",MemoryOpcode::copy,ByteRangeRef{MemoryRefKind::external,"input",0,f.q},ByteRangeRef{MemoryRefKind::slot,"slot",0,f.q},0,"copy",{},"main",{}, {}};
  EXPECT_FALSE(f.runtime->submit(f.view,{{command}},*bound,context));
  take(f.runtime->unbind(*bound,take(f.runtime->quiesce(*bound,{}))));
  auto next=take(f.runtime->bind(f.view,r,prepared,backing));
  EXPECT_EQ(*static_cast<unsigned char*>(take(next->slot("slot"))),11);
  next.reset();bound.reset();backing.reset();imported.reset();ext.owner.reset();context.external.clear();lease.reset();
  for(const auto& group:f.runtime->snapshot(f.view).groups)EXPECT_EQ(group.committed_bytes,0u);
}

TEST(LinuxRuntime, PlacementQueriesUseTrustedMetadataAndValidateEveryScope) {
  Fixture f;
  auto view = f.view;
  view.graph = {}; // Mutable caller metadata cannot override the runtime model.
  PlacementPlan plan{"host", f.q, {{"a", 1}}, {}, 0};
  const std::vector<AccessMode> access{AccessMode::read, AccessMode::write};
  auto placement = f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access);
  ASSERT_TRUE(placement);
  EXPECT_EQ(placement.value().page_bytes, f.q);
  ASSERT_EQ(placement.value().segments.size(), 1);
  EXPECT_EQ(placement.value().segments[0].pool, "a");
  const std::vector<std::string> pools{"a"};
  auto capability = f.runtime->mapping_capabilities(view, pools);
  ASSERT_TRUE(capability);
  EXPECT_EQ(capability.value().page_bytes, f.q);
  view.epoch += 1;
  EXPECT_FALSE(f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access));
  EXPECT_FALSE(f.runtime->mapping_capabilities(view, pools));
  view = f.view;
  view.execution_slots["executor"] = 2;
  EXPECT_FALSE(f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access));
  EXPECT_FALSE(f.runtime->mapping_capabilities(view, pools));
  view = f.view;
  view.allocatable_pools = {"a", "a"};
  EXPECT_FALSE(f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access));
  view = f.view;
  view.executable_agents.push_back("executor");
  EXPECT_FALSE(f.runtime->mapping_capabilities(view, pools));
  view = take(f.runtime->view({{"executor"}, {"a"}, {"a"}, {{"executor", 1}}}));
  plan.weights = {{"b", 1}};
  EXPECT_FALSE(f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access));
  const std::vector<std::string> unavailable{"b"};
  EXPECT_FALSE(f.runtime->mapping_capabilities(view, unavailable));
  view = f.view;
  std::reverse(view.allocatable_pools.begin(), view.allocatable_pools.end());
  plan.weights = {{"a", 1}};
  EXPECT_TRUE(f.runtime->resolve_placement(view, plan, f.q, BackingMode::shareable_remappable, access));
  // Query results and public views still have independent value ownership.
  auto fresh = take(f.runtime->view(take(f.runtime->current_scope())));
  EXPECT_EQ(fresh.graph.pools.size(), 2);
}

TEST(LinuxRuntime, EightRanksImplicitRootAdmissionDoesNotFailOnSharedEpochContention) {
  concurrent_runtime_roots(8);
}
TEST(LinuxRuntime, SixteenRanksImplicitRootAdmissionDoesNotFailOnSharedEpochContention) {
  concurrent_runtime_roots(16);
}
TEST(LinuxRuntime, SharedAuthorityDeniesCompetingRootAndPreservesLocalOwnershipCounters) {
  Directory d;
  Fixture first(4, d.path.string()), second(4, d.path.string());
  auto allocation = take(first.runtime->allocate(first.view, first.request("first", "a", 3)));
  auto b = second.runtime->snapshot(second.view);
  EXPECT_EQ(b.groups.front().committed_bytes, 0u);
  EXPECT_EQ(b.groups.front().shared_remaining_bytes, second.q);
  EXPECT_EQ(b.pools.front().allocatable_bytes, second.q);
  EXPECT_FALSE(second.runtime->allocate(second.view, second.request("second", "a", 2)));
  { auto own = std::move(allocation); }
  auto next = take(second.runtime->allocate(second.view, second.request("second", "a", 4)));
  EXPECT_EQ(second.committed(), 4 * second.q);
}
TEST(LinuxRuntime, SharedCapacityCapabilityIsExplicitAndIndependentOfDiagnosticDescriptors) {
  Directory d, other;
  Fixture a(4, d.path.string()), b(4, (d.path / ".").string()), c(4, other.path.string()), local;
  ASSERT_TRUE(a.runtime->capacity_authority());
  EXPECT_EQ(a.runtime->capacity_authority(), b.runtime->capacity_authority());
  EXPECT_NE(a.runtime->capacity_authority(), c.runtime->capacity_authority());
  EXPECT_FALSE(local.runtime->capacity_authority());
}
TEST(LinuxRuntime, SharedAuthorityRejectsStalePlanAndAccountsGlobalPendingAndOverlappingCaps) {
  Directory d;
  const U q = static_cast<U>(::sysconf(_SC_PAGESIZE));
  Fixture first(8, d.path.string(), 4*q), second(8, d.path.string(), 4*q);
  const auto old = second.runtime->snapshot(second.view).epoch;
  auto r = first.recipe(3);
  auto lease = take(first.runtime->reserve(first.view, r, first.runtime->snapshot(first.view).epoch));
  EXPECT_FALSE(second.runtime->reserve(second.view, second.recipe(1), old));
  EXPECT_EQ(second.runtime->snapshot(second.view).constraints.front().remaining_bytes, q);
  EXPECT_FALSE(second.runtime->allocate(second.view, second.request("other-bank", "b", 2)));
  auto other = take(second.runtime->allocate(second.view, second.request("other-bank", "b", 1)));
  EXPECT_FALSE(second.runtime->estimate(second.view, second.recipe(2)).value().feasible);
  lease.reset();
  auto next = take(second.runtime->allocate(second.view, second.request("free-now", "a", 3)));
  EXPECT_EQ(second.committed(), 4*q);
}
TEST(LinuxRuntime, SharedAuthorityChargesArenaRootOnceAcrossCachedPlanAndExportedAliases) {
  Directory d;
  Fixture first(4, d.path.string()), second(4, d.path.string());
  auto arena = first.arena(4,4);
  MemoryArena* caches[] = {arena.get()};
  auto r = first.recipe(2);
  auto lease = take(first.runtime->reserve(first.view, r, first.runtime->snapshot(first.view).epoch, caches));
  auto roots = take(first.runtime->acquire_backing(first.view, r, *lease));
  EXPECT_EQ(second.runtime->snapshot(second.view).pools.front().allocatable_bytes, 0u);
  EXPECT_EQ(first.committed(), 4*first.q);
  EXPECT_EQ(take(arena->trim(0)), 0u);
  roots.reset();
  EXPECT_EQ(take(arena->trim(0)), 4*first.q);
  EXPECT_EQ(second.runtime->snapshot(second.view).pools.front().allocatable_bytes, 4*first.q);
}
TEST(LinuxRuntime, ArenaCacheHotPathDoesNotTouchSharedAuthorityMetadata) {
  Directory d;
  Fixture f(4, d.path.string());
  auto arena = f.arena(4,4);
  const auto path = d.path / "capacity.toml";
  std::ifstream input(path);
  const std::string original{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
  { std::ofstream file(path); file << "invalid = ["; }
  // Cached suballocation and return only touch the local free-list/ledger.
  {
    auto a = take(arena->allocate({f.q,"a",false}));
    std::memset(take(a.data()), 19, f.q);
  }
  { std::ofstream file(path); file << original; }
  EXPECT_EQ(take(arena->trim(0)),4*f.q);
}
TEST(LinuxRuntime, DirectOwnershipAliasesAndCapacityRelease) {
  Fixture f;
  auto allocation = take(f.runtime->allocate(f.view, f.request()));
  EXPECT_EQ(f.committed(), f.q);
  auto* p = take(allocation.data());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % f.q, 0u);
  auto exported = take(allocation.export_backing());
  std::memset(p, 17, f.q);
  auto va = take(f.runtime->reserve_address("host", f.q, f.q));
  auto map = take(f.runtime->map(*va, exported, {0, 0, f.q}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(map->data())), 17);
  EXPECT_EQ(take(f.runtime->observe(f.view)).allocations.size(), 1u);
  { auto owned = std::move(allocation); }
  EXPECT_EQ(f.committed(), f.q);
  exported.reset();
  map.reset();
  EXPECT_EQ(f.committed(), 0u);
}
TEST(LinuxRuntime, AdmissionReservationsAreTransactionalAndEpochChecked) {
  Fixture f(4);
  auto r = f.recipe(3);
  auto epoch = f.runtime->snapshot(f.view).epoch;
  auto lease = take(f.runtime->reserve(f.view, r, epoch));
  EXPECT_EQ(f.committed(), 0u);
  EXPECT_FALSE(f.runtime->reserve(f.view, r, epoch));
  EXPECT_EQ(f.runtime->snapshot(f.view).groups.front().unbacked_reserved_bytes, 3 * f.q);
  lease.reset();
  EXPECT_EQ(f.runtime->snapshot(f.view).groups.front().unbacked_reserved_bytes, 0u);
  auto now = f.runtime->snapshot(f.view).epoch;
  lease = take(f.runtime->reserve(f.view, r, now));
  auto wrong = r;
  wrong.backings.front().id = "other";
  EXPECT_FALSE(f.runtime->acquire_backing(f.view, wrong, *lease));
  auto backing = take(f.runtime->acquire_backing(f.view, r, *lease));
  EXPECT_EQ(f.committed(), 3 * f.q);
  EXPECT_FALSE(f.runtime->acquire_backing(f.view, r, *lease));
  backing.reset();
  EXPECT_EQ(f.committed(), 0u);
}
TEST(LinuxRuntime, UnbackedAddressesAndStableRebindAcrossPlans) {
  Fixture f;
  auto r = f.recipe();
  auto prepared = take(f.runtime->prepare_addresses(f.view, r.addresses));
  EXPECT_EQ(f.committed(), 0u);
  auto address = take(prepared->slot("slot"));
  auto lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch));
  auto backing = take(f.runtime->acquire_backing(f.view, r, *lease));
  auto b = take(f.runtime->bind(f.view, r, prepared, backing));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(take(b->slot("slot"))), address.value);
  EXPECT_FALSE(f.runtime->bind(f.view, r, prepared, backing));
  auto first = b->generation();
  EXPECT_FALSE(f.runtime->unbind(*b, {"forged"}));
  auto proof = take(f.runtime->quiesce(*b, {}));
  take(f.runtime->unbind(*b, proof));
  EXPECT_FALSE(b->slot("slot"));
  EXPECT_FALSE(f.runtime->unbind(*b, proof));
  auto second = take(f.runtime->bind(f.view, r, prepared, backing));
  EXPECT_NE(first, second->generation());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(take(second->slot("slot"))), address.value);
}
TEST(LinuxRuntime, StrongRegionBaseAlignmentIsIndependentOfCapacityAndMappingQuantum) {
  Fixture f;
  for (const U alignment : {U{1} << 16, U{1} << 22}) {
    auto r = f.recipe(1);
    r.addresses.regions.front().base_alignment_bytes = alignment;
    EXPECT_TRUE(f.runtime->validate_recipe(f.view, r));
    auto prepared = take(f.runtime->prepare_addresses(f.view, r.addresses));
    const auto address = take(prepared->slot("slot")).value;
    EXPECT_EQ(address % std::max(f.q, alignment), 0u);
    EXPECT_EQ(f.committed(), 0u);
    auto lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch));
    auto backing = take(f.runtime->acquire_backing(f.view, r, *lease));
    auto bound = take(f.runtime->bind(f.view, r, prepared, backing));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(take(bound->slot("slot"))), address);
    EXPECT_EQ(f.committed(), f.q);
    auto proof = take(f.runtime->quiesce(*bound, {}));
    take(f.runtime->unbind(*bound, proof));
    bound.reset();backing.reset();lease.reset();
    EXPECT_EQ(f.committed(), 0u);
    // Backings are canonicalized to their recipe; reserve a matching weaker
    // recipe before testing that the same stronger prepared VA may be reused.
    auto weaker = r;
    weaker.addresses.regions.front().base_alignment_bytes = 0;
    lease = take(f.runtime->reserve(f.view, weaker, f.runtime->snapshot(f.view).epoch));
    backing = take(f.runtime->acquire_backing(f.view, weaker, *lease));
    bound = take(f.runtime->bind(f.view, weaker, prepared, backing));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(take(bound->slot("slot"))), address);
    proof = take(f.runtime->quiesce(*bound, {}));
    take(f.runtime->unbind(*bound, proof));
    bound.reset();backing.reset();lease.reset();
    EXPECT_EQ(f.committed(), 0u);
  }
  auto invalid = f.recipe(1);
  invalid.addresses.regions.front().base_alignment_bytes = 3;
  EXPECT_FALSE(f.runtime->validate_recipe(f.view, invalid));
  EXPECT_FALSE(f.runtime->prepare_addresses(f.view, invalid.addresses));
}

TEST(LinuxRuntime, WeakerPreparedBaseCannotSatisfyStrongerDeclaredAlignment) {
  Fixture f;
  auto r = f.recipe(1);
  auto prepared = take(f.runtime->prepare_addresses(f.view, r.addresses));
  r.addresses.regions.front().base_alignment_bytes = U{1} << 22;
  auto lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch));
  auto backing = take(f.runtime->acquire_backing(f.view, r, *lease));
  // Reject by the declared guarantee even if this VA happened to be aligned.
  EXPECT_FALSE(f.runtime->bind(f.view, r, prepared, backing));
}

TEST(LinuxRuntime, ArenaSuballocationsRetainOffsetsExportsAndTrim) {
  Fixture f;
  auto arena = f.arena(4);
  EXPECT_EQ(f.committed(), 4 * f.q);
  auto a = take(arena->allocate({f.q, {}, true}));
  auto b = take(arena->allocate({f.q, {}, true}));
  auto ba = take(a.export_backing()), bb = take(b.export_backing());
  EXPECT_EQ(take(LinuxPageStore::physical_identity(*ba)), take(LinuxPageStore::physical_identity(*bb)));
  EXPECT_NE(take(LinuxPageStore::physical_offset(*ba)), take(LinuxPageStore::physical_offset(*bb)));
  std::memset(take(b.data()), 29, f.q);
  auto va = take(f.runtime->reserve_address("host", f.q, f.q));
  auto mapped = take(f.runtime->map(*va, bb, {0, 0, f.q}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(mapped->data())), 29);
  { auto own = std::move(a); }
  { auto own = std::move(b); }
  EXPECT_EQ(take(arena->trim(0)), 0u);
  ba.reset();
  bb.reset();
  EXPECT_EQ(take(arena->trim(0)), 0u);
  mapped.reset();
  EXPECT_EQ(take(arena->trim(0)), 4 * f.q);
  EXPECT_EQ(f.committed(), 0u);
}
TEST(LinuxRuntime, DomainFallbackAndArenaLimit) {
  Fixture f(4);
  auto arena = f.arena(0, 8);
  auto a = take(arena->allocate({4 * f.q, "a", false}));
  EXPECT_FALSE(arena->allocate({f.q, "a", false}));
  auto b = take(arena->allocate({4 * f.q, "a", true}));
  EXPECT_EQ(b.info().segments.front().pool, "b");
  EXPECT_EQ(f.committed(), 8 * f.q);
  EXPECT_FALSE(arena->allocate({f.q, {}, true}));
}
TEST(LinuxRuntime, CachedPlanLeaseUsesPhysicalCapacityOnlyOnce) {
  Fixture f;
  auto arena = f.arena(4);
  MemoryArena* list[] = {arena.get()};
  auto r = f.recipe(3);
  auto estimate = take(f.runtime->estimate(f.view, r, list));
  EXPECT_EQ(estimate.new_backing_bytes_by_capacity_group.at("cap-a"), 0u);
  auto lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch, list));
  EXPECT_EQ(f.committed(), 4 * f.q);
  EXPECT_EQ(f.runtime->snapshot(f.view).groups.front().idle_reserved_bytes, 3 * f.q);
  EXPECT_EQ(arena->snapshot().pools.front().largest_cached_block_bytes, f.q);
  lease.reset();
  EXPECT_EQ(arena->snapshot().pools.front().largest_cached_block_bytes, 4 * f.q);
  lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch, list));
  auto backings = take(f.runtime->acquire_backing(f.view, r, *lease));
  EXPECT_EQ(f.committed(), 4 * f.q);
  EXPECT_EQ(f.runtime->snapshot(f.view).groups.front().live_bytes, 3 * f.q);
  backings.reset();
  EXPECT_EQ(arena->snapshot().pools.front().largest_cached_block_bytes, 4 * f.q);
}
TEST(LinuxRuntime, AsyncFailureDoesNotAuthorizeArenaReuse) {
  Fixture f;
  auto arena = f.arena(1, 1);
  auto allocation = take(arena->allocate({f.q, {}, true}));
  auto flag = std::make_shared<std::atomic<bool>>(false);
  auto callback = [flag]() -> Result<EventStatus> {
    return {EventStatus{EventState::failed, flag->load(), Error{"test", "failed but active"}}};
  };
  InvocationContext context{{"use", allocation.info().generation}, {}, {}};
  auto token = take(
    f.runtime->events().import_event({"host-callback", HostEventCallbacks{callback, callback, {}}, flag}, context));
  take(arena->release(std::move(allocation), token));
  EXPECT_FALSE(arena->allocate({f.q, {}, true}));
  EXPECT_FALSE(f.runtime->events().release({&token, 1}));
  flag->store(true);
  take(f.runtime->events().wait(token));
  auto next = take(arena->allocate({f.q, {}, true}));
  EXPECT_NE(next.info().generation, context.stamp.binding_generation);
  take(f.runtime->events().release({&token, 1}));
}
TEST(LinuxRuntime, CpuCommandsUseInvocationLocalExternalBindings) {
  Fixture f;
  auto b = f.bind(f.recipe());
  auto owner = std::make_shared<std::vector<unsigned char>>(f.q, 73);
  ExternalMemoryView ext{
    "arbitrary-input-id", {"host", reinterpret_cast<std::uintptr_t>(owner->data())}, f.q, "b", owner, false, {}};
  auto imported = take(f.runtime->import_memory(f.view, ext));
  InvocationContext context{{"replay-1", b->generation()}, {}, {{"input", imported}}};
  MemoryCommand copy{"copy",
                     MemoryOpcode::copy,
                     ByteRangeRef{MemoryRefKind::external, "input", 0, f.q},
                     ByteRangeRef{MemoryRefKind::slot, "slot", 0, f.q},
                     0,
                     "copy",
                     {},
                     "main",
                     {},
                     {}};
  auto submission = take(f.runtime->submit(f.view, {{copy}}, *b, context));
  EXPECT_FALSE(submission.submission_error);
  EXPECT_EQ(submission.accepted_commands, (std::vector<std::string>{"copy"}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(b->slot("slot"))), 73);
  EXPECT_EQ(take(f.runtime->events().wait(submission.finished)).state, EventState::succeeded);
  take(f.runtime->events().release({&submission.finished, 1}));
  context.stamp.id = "replay-2";
  context.external.front().symbol = "different";
  EXPECT_FALSE(f.runtime->submit(f.view, {{copy}}, *b, context));
}
TEST(LinuxRuntime, BackendCertifiedExternalAliasBlocksOverlappingCopy) {
  Fixture f;
  auto b = f.bind(f.recipe());
  std::memset(take(b->slot("slot")), 11, f.q);
  auto owner = std::shared_ptr<void>(b.get(), [](void*) {});
  ExternalMemoryView ext{
    "alias", {"host", reinterpret_cast<std::uintptr_t>(take(b->slot("slot")))}, f.q, "a", owner, false, {}};
  auto imported = take(f.runtime->import_memory(f.view, ext));
  InvocationContext context{{"alias-check", b->generation()}, {}, {{"input", imported}}};
  MemoryCommand copy{"copy",
                     MemoryOpcode::copy,
                     ByteRangeRef{MemoryRefKind::external, "input", 0, f.q},
                     ByteRangeRef{MemoryRefKind::slot, "slot", 0, f.q},
                     0,
                     "copy",
                     {},
                     "main",
                     {},
                     {}};
  EXPECT_FALSE(f.runtime->submit(f.view, {{copy}}, *b, context));
  EXPECT_EQ(*static_cast<unsigned char*>(take(b->slot("slot"))), 11);
}
TEST(LinuxRuntime, UseInvalidatesQuiescenceAndStaleInvocation) {
  Fixture f;
  auto b = f.bind(f.recipe());
  auto proof = take(f.runtime->quiesce(*b, {}));
  auto* events = dynamic_cast<HostEventService*>(&f.runtime->events());
  auto token = take(events->complete({"touch", b->generation()}, {EventState::succeeded, true, {}}));
  take(b->record_use(token));
  EXPECT_FALSE(f.runtime->unbind(*b, proof));
  proof = take(f.runtime->quiesce(*b, {}));
  take(f.runtime->unbind(*b, proof));
  EXPECT_FALSE(f.runtime->resolve_operands(f.view, *b, {{"stale", b->generation()}, {}, {}}));
  take(events->release({&token, 1}));
}
TEST(LinuxRuntime, ViewsCannotGrantForeignPoolsExecutorsOrSlots) {
  Fixture f;
  auto v = f.view;
  v.execution_slots["executor"] = 2;
  EXPECT_FALSE(f.runtime->view({v.executable_agents, v.allocatable_pools, v.accessible_pools, v.execution_slots}));
  v = f.view;
  v.allocatable_pools.push_back("foreign");
  EXPECT_FALSE(f.runtime->allocate(v, f.request()));
  v = f.view;
  v.epoch = 0;
  EXPECT_FALSE(f.runtime->allocate(v, f.request()));
}
TEST(LinuxRuntime, DistributedLayoutsRequireRealControlAndGlobalCapacity) {
  Fixture f(4);
  auto r = f.recipe();
  DistributedLayout layout{"layout", 1, {}, r.backings, {{"p0", r.addresses, r.mappings}}};
  auto agreement = take(f.runtime->agree_layout(layout));
  r.layout_id = layout.id;
  auto prepared = take(f.runtime->prepare_addresses(f.view, r.addresses));
  auto lease = take(f.runtime->reserve(f.view, r, f.runtime->snapshot(f.view).epoch));
  auto backing = take(f.runtime->acquire_backing(f.view, r, *lease));
  backing = take(f.runtime->exchange_backing(*agreement, backing));
  auto b = take(f.runtime->bind(f.view, r, prepared, backing, agreement.get()));
  EXPECT_TRUE(b->slot("slot"));
  layout.participants.push_back({"p1", {}, {}});
  EXPECT_FALSE(f.runtime->agree_layout(layout));
}
} // namespace
TEST(LinuxRuntime, ReleasedCallerEventStillPermitsQuiescence) {
  Fixture f;
  auto b = f.bind(f.recipe());
  auto done = std::make_shared<std::atomic<bool>>(false);
  auto poll = [done]() -> Result<EventStatus> {
    return {EventStatus{done->load() ? EventState::succeeded : EventState::pending, done->load(), {}}};
  };
  InvocationContext context{{"registered-use", b->generation()}, {}, {}};
  auto token =
    take(f.runtime->events().import_event({"host-callback", HostEventCallbacks{poll, poll, {}}, done}, context));
  take(b->record_use(token));
  done->store(true);
  take(f.runtime->events().wait(token));
  take(f.runtime->events().release({&token, 1}));
  auto proof = take(f.runtime->quiesce(*b, {}));
  take(f.runtime->unbind(*b, proof));
}
TEST(LinuxRuntime, EventCallbacksMayInspectBindingWithoutDeadlock) {
  Fixture f;
  auto b = f.bind(f.recipe());
  auto done = std::make_shared<std::atomic<bool>>(false);
  auto poll = [done, ptr = b.get()]() -> Result<EventStatus> {
    if (!ptr->slot("slot"))
      return {Error{"test", "slot unavailable"}};
    return {EventStatus{done->load() ? EventState::succeeded : EventState::pending, done->load(), {}}};
  };
  InvocationContext context{{"callback-use", b->generation()}, {}, {}};
  auto token =
    take(f.runtime->events().import_event({"host-callback", HostEventCallbacks{poll, poll, {}}, done}, context));
  take(b->record_use(token));
  done->store(true);
  auto proof = take(f.runtime->quiesce(*b, {}));
  take(f.runtime->unbind(*b, proof));
  take(f.runtime->events().release({&token, 1}));
}
TEST(LinuxRuntime, InvalidRoutesAreRejectedBeforeAnyWrite) {
  Fixture f;
  auto b = f.bind(f.recipe());
  auto owner = std::make_shared<std::vector<unsigned char>>(f.q, 39);
  ExternalMemoryView ext{"source", {"host", reinterpret_cast<std::uintptr_t>(owner->data())}, f.q, "b", owner, false,
                         {}};
  InvocationContext context{
    {"bad-route", b->generation()}, {}, {{"source", take(f.runtime->import_memory(f.view, ext))}}};
  MemoryCommand command{"copy",
                        MemoryOpcode::copy,
                        ByteRangeRef{MemoryRefKind::external, "source", 0, f.q},
                        ByteRangeRef{MemoryRefKind::slot, "slot", 0, f.q},
                        0,
                        "copy",
                        "no-such-route",
                        "main",
                        {},
                        {}};
  std::memset(take(b->slot("slot")), 0, f.q);
  EXPECT_FALSE(f.runtime->submit(f.view, {{command}}, *b, context));
  command.route = "aa";
  EXPECT_FALSE(f.runtime->submit(f.view, {{command}}, *b, context));
  EXPECT_EQ(*static_cast<unsigned char*>(take(b->slot("slot"))), 0);
  command.route = "ba";
  auto result = take(f.runtime->submit(f.view, {{command}}, *b, context));
  EXPECT_FALSE(result.submission_error);
  EXPECT_EQ(*static_cast<unsigned char*>(take(b->slot("slot"))), 39);
  take(f.runtime->events().release({&result.finished, 1}));
}
TEST(LinuxRuntime, ForgedGraphMetadataNeverControlsAdmission) {
  Fixture f;
  auto v = f.view;
  v.graph.pools.front().capacity_group = "forged";
  v.graph.pools.front().capacity_bytes = ~U{0};
  auto estimate = take(f.runtime->estimate(v, f.recipe()));
  EXPECT_EQ(estimate.peak_bytes_by_capacity_group.at("cap-a"), 2 * f.q);
  EXPECT_FALSE(estimate.peak_bytes_by_capacity_group.contains("forged"));
}
TEST(LinuxRuntime, UnselectedArenaCacheDoesNotInflateAvailablePoolCapacity) {
  Fixture f(4);
  auto first = f.arena(4, 4), second = f.arena(0, 4);
  auto snap = second->snapshot();
  EXPECT_EQ(snap.pools.front().allocatable_bytes, 0u);
  EXPECT_EQ(f.runtime->snapshot(f.view).pools.front().allocatable_bytes, 0u);
  MemoryArena* list[] = {first.get()};
  EXPECT_EQ(f.runtime->snapshot(f.view, list).pools.front().allocatable_bytes, 4 * f.q);
  EXPECT_FALSE(f.runtime->reserve(f.view, f.recipe(), f.runtime->snapshot(f.view).epoch));
}
TEST(LinuxRuntime, FragmentationAndBackingModeAreIncludedInPlanEstimate) {
  Fixture f(4);
  auto arena = f.arena(4, 4);
  std::vector<Allocation> live;
  for (unsigned i = 0; i < 4; ++i)
    live.push_back(take(arena->allocate({f.q, "a", false})));
  std::optional<Allocation> zero{std::move(live[0])}, two{std::move(live[2])};
  zero.reset();
  two.reset();
  MemoryArena* caches[] = {arena.get()};
  auto estimate = take(f.runtime->estimate(f.view, f.recipe(2), caches));
  EXPECT_FALSE(estimate.feasible);
  EXPECT_EQ(estimate.new_backing_bytes_by_capacity_group.at("cap-a"), 2 * f.q);
  EXPECT_FALSE(f.runtime->reserve(f.view, f.recipe(2), f.runtime->snapshot(f.view).epoch, caches));
  auto r = f.recipe(1);
  r.backings.front().mode = BackingMode::private_memory;
  estimate = take(f.runtime->estimate(f.view, r, caches));
  EXPECT_FALSE(estimate.feasible);
  EXPECT_EQ(estimate.new_backing_bytes_by_capacity_group.at("cap-a"), f.q);
}
TEST(LinuxRuntime, LocalSharedAcquireWaitsForPublicationAndRetire) {
  Fixture f;
  auto recipe = f.recipe();
  recipe.sharing_contracts = {{"sharing", {"p0"}, {"p0"}, {"p0"}, "publish-retire"}};
  recipe.addresses.slots.front().sharing_contract = "sharing";
  DistributedLayout layout{
    "layout", 7, recipe.sharing_contracts, recipe.backings, {{"p0", recipe.addresses, recipe.mappings}}};
  auto agreement = take(f.runtime->agree_layout(layout));
  (void)agreement;
  SharedRange range{"backing", 0, 2 * f.q, "sharing", 7, 1};
  EXPECT_FALSE(f.runtime->acquire_shared(range));
  auto begin = take(f.runtime->begin_shared(range));
  EXPECT_FALSE(f.runtime->acquire_shared(range));
  auto stopped = std::make_shared<std::atomic<bool>>(false);
  auto poll = [stopped]() -> Result<EventStatus> {
    return {EventStatus{stopped->load() ? EventState::succeeded : EventState::pending, stopped->load(), {}}};
  };
  auto producer = take(f.runtime->events().import_event({"host-callback", HostEventCallbacks{poll, poll, {}}, stopped},
                                                        {{"producer", 17}, {}, {}}));
  auto publish = take(f.runtime->publish_shared(range, producer));
  auto acquire = take(f.runtime->acquire_shared(range));
  auto overlap = range;
  overlap.offset_bytes = f.q;
  overlap.size_bytes = f.q;
  overlap.data_epoch = 2;
  EXPECT_FALSE(f.runtime->begin_shared(overlap));
  EXPECT_EQ(take(f.runtime->events().query(acquire)).state, EventState::pending);
  auto retire = take(f.runtime->retire_shared(range, {&acquire, 1}));
  auto next = range;
  next.data_epoch = 2;
  EXPECT_FALSE(f.runtime->begin_shared(next));
  stopped->store(true);
  EXPECT_EQ(take(f.runtime->events().wait(acquire)).state, EventState::succeeded);
  take(f.runtime->events().wait(retire));
  CompletionToken tokens[] = {begin, producer, publish, acquire, retire};
  take(f.runtime->events().release(tokens));
  auto begin2 = take(f.runtime->begin_shared(next));
  EXPECT_FALSE(f.runtime->begin_shared(range));
  take(f.runtime->events().release({&begin2, 1}));
}
TEST(LinuxRuntime, LowLevelRemapRequiresCurrentProvenance) {
  Fixture f;
  auto a = take(f.runtime->allocate(f.view, f.request("a"))), b = take(f.runtime->allocate(f.view, f.request("b")));
  std::memset(take(a.data()), 12, f.q);
  std::memset(take(b.data()), 97, f.q);
  auto va = take(f.runtime->reserve_address("host", f.q, f.q));
  auto mapping = take(f.runtime->map(*va, take(a.export_backing()), {0, 0, f.q}));
  auto address = take(mapping->data());
  auto generation = mapping->generation();
  auto proof = take(f.runtime->quiesce(*mapping, {}));
  take(f.runtime->rebind(*mapping, take(b.export_backing()), {0, 0, f.q}, proof));
  EXPECT_EQ(take(mapping->data()), address);
  EXPECT_NE(mapping->generation(), generation);
  EXPECT_EQ(*static_cast<unsigned char*>(address), 97);
  EXPECT_FALSE(f.runtime->rebind(*mapping, take(a.export_backing()), {0, 0, f.q}, proof));
}

TEST(LinuxRuntime, PreferredPoolWinsOverFallbackCacheWhileItHasCapacity) {
  Fixture f(4);
  auto arena = f.arena(4, 8);
  auto allocation = take(arena->allocate({f.q, "b", true}));
  EXPECT_EQ(allocation.info().segments.front().pool, "b");
}

TEST(LinuxRuntime, RefreshFreePhysicalCapacityHasAnExplicitSamplingBoundary) {
  auto runtime = take(discover_linux_runtime());
  const auto before = runtime->snapshot(take(runtime->view(take(runtime->current_scope())))).epoch;
  take(runtime->refresh_available());
  const auto after = runtime->snapshot(take(runtime->view(take(runtime->current_scope())))).epoch;
  EXPECT_GT(after, before);
  Fixture synthetic;
  EXPECT_FALSE(synthetic.runtime->refresh_available());
}

TEST(LinuxRuntime, ArenaSnapshotReportsItsOwnDomainAndRemainingLimit) {
  Fixture f;
  auto options = f.options(1, 3);
  options.allowed_pools = {"a"};
  auto arena = take(f.runtime->create_arena(f.view, options));
  auto snapshot = arena->snapshot();
  EXPECT_EQ(snapshot.pools.front().allocatable_bytes, 3 * f.q);
  EXPECT_EQ(snapshot.pools.back().allocatable_bytes, 0u);
  auto a = take(arena->allocate({2 * f.q, "a", false}));
  snapshot = arena->snapshot();
  EXPECT_EQ(snapshot.pools.front().allocatable_bytes, f.q);
  auto b = take(arena->allocate({f.q, "a", false}));
  snapshot = arena->snapshot();
  EXPECT_EQ(snapshot.pools.front().allocatable_bytes, 0u);
  EXPECT_FALSE(snapshot.pools.front().shared_constraints.empty());
}
TEST(LinuxRuntime, PreparedAndBackingOwnersKeepTheirIssuerAliveAndRejectForeignRuntime) {
  Fixture f;
  auto recipe = f.recipe();
  auto prepared = take(f.runtime->prepare_addresses(f.view, recipe.addresses));
  auto lease = take(f.runtime->reserve(f.view, recipe, f.runtime->snapshot(f.view).epoch));
  auto backing = take(f.runtime->acquire_backing(f.view, recipe, *lease));
  f.runtime.reset();
  EXPECT_TRUE(prepared->slot("slot"));
  Fixture other;
  EXPECT_FALSE(other.runtime->bind(other.view, recipe, prepared, backing));
  prepared.reset();
  backing.reset();
  lease.reset();
}

} // namespace vecops::memory::model
