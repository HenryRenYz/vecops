// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/RuntimePolicy.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
template <class T>
T take(Result<T> result) {
  if (!result)
    throw std::runtime_error(std::get<Error>(result.state).message);
  return std::move(result).value();
}
std::string policy_text(U q, std::string hard = "hard_limit_fraction = 0.5",
                         std::string initial = "preallocate_bytes = \"0 B\"") {
  return "schema = \"memory-runtime/0.1\"\n[arena]\ndomain_rule = \"custom-rule\"\n" + hard + "\n" + initial +
         "\ngrow_by = \"" + std::to_string(q) + " B\"\nretain_idle = \"0 B\"\npage_size = \"" +
         std::to_string(q) + " B\"\nbacking = \"shareable_remappable\"\nallow_growth = true\n";
}
void replace(std::string& text, const std::string& old, const std::string& value) {
  const auto at = text.find(old);
  ASSERT_NE(at, std::string::npos);
  text.replace(at, old.size(), value);
}
struct Fixture {
  U q = static_cast<U>(::sysconf(_SC_PAGESIZE));
  LinuxDiscovery discovery() const {
    LinuxDiscovery d;
    d.graph.agents = {{"host-executor", {"host_load_store"}, "host"}};
    d.graph.capacity_groups = {{"physical-bank", 8 * q}, {"other-bank", 4 * q}};
    d.graph.pools = {{"a", 8 * q, "host", q, {q}, {}, {}, "physical-bank"},
                     {"alias-a", 8 * q, "host", q, {q}, {}, {}, "physical-bank"},
                     {"b", 4 * q, "host", q, {q}, {}, {}, "other-bank"}};
    d.graph.domains = {{"first", "custom-rule", {"a", "alias-a"}}, {"second", "custom-rule", {"b"}}};
    d.scope = {{"host-executor"}, {"a", "alias-a", "b"}, {"a", "alias-a", "b"}, {{"host-executor", 1}}};
    for (const auto& pool : d.graph.pools) {
      d.pool_bindings.push_back({pool.id, {}});
      d.pool_available_bytes[pool.id] = *pool.capacity_bytes;
      d.graph.access.push_back({pool.id, "direct", "host-executor", pool.id,
                                {AccessMode::read, AccessMode::write}, {}, {}, MappingMode::direct,
                                {}, {}, {}, {}, {}});
    }
    return d;
  }
  RuntimePolicy policy(std::string initial = "preallocate_bytes = \"0 B\"") const {
    return take(parse_runtime_policy(policy_text(q, "hard_limit_fraction = 0.5", std::move(initial))));
  }
  LinuxRuntimeOptions options() const {
    LinuxRuntimeOptions result;
    result.participant = "test-participant";
    return result;
  }
  ResourceView view(MemoryRuntime& runtime) const {
    return take(runtime.view(take(runtime.current_scope())));
  }
  BackingRequest request(std::string id, std::string pool, U pages) const {
    return {std::move(id), "test-participant", std::move(pool), pages * q, q,
            BackingMode::shareable_remappable, {}};
  }
};
TEST(RuntimePolicy, DocumentationPoliciesParseWithExplicitZeroAndEnums) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
  for (const auto* name : {"eager", "planned"}) {
    std::ifstream file(root / "docs/memory-resource-model/policies" / (std::string(name) + ".toml"));
    ASSERT_TRUE(file);
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    auto result = parse_runtime_policy(text);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.value().arena.domain_rule, "ddr_domain");
    EXPECT_EQ(std::get<double>(result.value().arena.hard_limit.bytes_or_capacity_fraction), 0.8);
    if (std::string(name) == "planned") {
      EXPECT_EQ(std::get<U>(result.value().arena.initial.bytes_or_limit_fraction), 0u);
      EXPECT_EQ(result.value().arena.backing, BackingMode::shareable_remappable);
    } else
      EXPECT_EQ(result.value().arena.backing, BackingMode::private_memory);
  }
}
TEST(RuntimePolicy, RejectsUnknownFieldsTypesFractionsAndAmbiguousAlternatives) {
  Fixture f;
  const auto valid = policy_text(f.q);
  for (const auto& [old, invalid] : std::vector<std::pair<std::string, std::string>>{
         {"schema = \"memory-runtime/0.1\"", "schema = \"memory-runtime/0.2\""},
         {"[arena]", "unknown = 1\n[arena]"},
         {"domain_rule = \"custom-rule\"", "domain_rule = \"custom-rule\"\nnuma = 0"},
         {"hard_limit_fraction = 0.5", "hard_limit_fraction = 0.5\nhard_limit_bytes = \"4 KiB\""},
         {"hard_limit_fraction = 0.5", "hard_limit_fraction = nan"},
         {"hard_limit_fraction = 0.5", "hard_limit_fraction = 1.1"},
         {"hard_limit_fraction = 0.5", "hard_limit_fraction = -0.1"},
         {"hard_limit_fraction = 0.5", "hard_limit_fraction = true"},
         {"preallocate_bytes = \"0 B\"", "preallocate_bytes = \"0 B\"\npreallocate_fraction = 0"},
         {"preallocate_bytes = \"0 B\"", ""},
         {"backing = \"shareable_remappable\"", "backing = \"shared\""},
         {"allow_growth = true", "allow_growth = 1"},
         {"page_size = \"" + std::to_string(f.q) + " B\"", "page_size = \"3 KiB\""},
         {"page_size = \"" + std::to_string(f.q) + " B\"", "page_size = \"0 B\""},
         {"retain_idle = \"0 B\"", "retain_idle = 0"}}) {
    auto text = valid;
    replace(text, old, invalid);
    EXPECT_FALSE(parse_runtime_policy(text)) << text;
  }
  EXPECT_FALSE(parse_runtime_policy("schema = ["));
}
TEST(RuntimePolicy, ResolvesUniqueCapacityAndScopedDomainMembersBeforeConstruction) {
  Fixture f;
  auto d = f.discovery();
  d.scope.allocatable_pools = {"a"};
  auto options = f.options();
  const auto arenas = take(apply_runtime_policy(d, options, f.policy()));
  ASSERT_EQ(arenas.size(), 1u);
  EXPECT_EQ(arenas.front().domain, "first");
  EXPECT_EQ(arenas.front().allowed_pools, std::vector<std::string>{"a"});
  EXPECT_EQ(std::get<U>(arenas.front().hard_limit.bytes_or_capacity_fraction), 4 * f.q);
  ASSERT_EQ(options.shared_limits.size(), 2u);
  EXPECT_EQ(options.shared_limits.front().capacity_groups, std::vector<std::string>{"physical-bank"});
  EXPECT_EQ(options.shared_limits.front().remaining_bytes, 4 * f.q);
  auto all = f.options();
  auto both = take(apply_runtime_policy(f.discovery(), all, f.policy()));
  EXPECT_EQ(std::get<U>(both.front().hard_limit.bytes_or_capacity_fraction), 4 * f.q);
}
TEST(RuntimePolicy, PhysicalDomainCapsDoNotDependOnEachParticipantsAllocationView) {
  Fixture f;
  auto all=f.discovery();
  all.graph.domains={{"combined","custom-rule",{"a","alias-a","b"}}};
  auto first=all,second=all;
  first.scope.allocatable_pools={"a"}; second.scope.allocatable_pools={"b"};
  auto a=f.options(),b=f.options();
  const auto arenas_a=take(apply_runtime_policy(first,a,f.policy()));
  const auto arenas_b=take(apply_runtime_policy(second,b,f.policy()));
  ASSERT_EQ(a.shared_limits.size(),1u);ASSERT_EQ(b.shared_limits.size(),1u);
  EXPECT_EQ(a.shared_limits[0].capacity_groups,b.shared_limits[0].capacity_groups);
  EXPECT_EQ(a.shared_limits[0].remaining_bytes,6*f.q);
  EXPECT_EQ(b.shared_limits[0].remaining_bytes,6*f.q);
  EXPECT_EQ(std::get<U>(arenas_a[0].hard_limit.bytes_or_capacity_fraction),6*f.q);
  EXPECT_EQ(std::get<U>(arenas_b[0].hard_limit.bytes_or_capacity_fraction),4*f.q);
}
TEST(RuntimePolicy, DirectAllocationAndAdditionalArenasCannotBypassDomainCap) {
  Fixture f;
  auto runtime = take(make_policy_linux_runtime(f.discovery(), f.policy(), f.options()));
  EXPECT_EQ(runtime->arena_ids().size(), 2u);
  auto view = f.view(*runtime);
  auto full = take(runtime->allocate(view, f.request("full", "a", 4)));
  EXPECT_FALSE(runtime->allocate(view, f.request("alias-bypass", "alias-a", 1)));
  ArenaOptions another{"first", {}, Limit{8 * f.q}, Preallocate{U{0}}, f.q, 0, f.q,
                        BackingMode::shareable_remappable, true};
  auto arena = take(runtime->create_arena(view, another));
  EXPECT_FALSE(arena->allocate({f.q, "a", false}));
  auto independent = runtime->allocate(view, f.request("other", "b", 2));
  EXPECT_TRUE(independent);
}
TEST(RuntimePolicy, PreallocationIsBasedOnResolvedLimitAndMaterializedBeforeReturn) {
  Fixture f;
  auto runtime = take(make_policy_linux_runtime(f.discovery(), f.policy("preallocate_fraction = 0.5"), f.options()));
  auto snapshot = runtime->snapshot(f.view(*runtime));
  U committed = 0;
  for (const auto& group : snapshot.groups)
    committed += group.committed_bytes;
  EXPECT_EQ(committed, 3 * f.q); // (8Q * .5 * .5) + (4Q * .5 * .5)
  EXPECT_EQ(take(runtime->observe(f.view(*runtime))).allocations.size(), 2u);
}
TEST(RuntimePolicy, ApplyIsTransactionalAndRespectsExistingLimits) {
  Fixture f;
  auto options = f.options();
  options.shared_limits = {{"existing", {"physical-bank"}, 3 * f.q}};
  auto invalid = f.policy();
  invalid.arena.domain_rule = "missing";
  EXPECT_FALSE(apply_runtime_policy(f.discovery(), options, invalid));
  EXPECT_EQ(options.shared_limits.size(), 1u);
  invalid = f.policy();
  invalid.arena.initial.bytes_or_limit_fraction = 9 * f.q;
  EXPECT_FALSE(apply_runtime_policy(f.discovery(), options, invalid));
  EXPECT_EQ(options.shared_limits.size(), 1u);
  auto runtime = take(make_policy_linux_runtime(f.discovery(), f.policy(), options));
  EXPECT_FALSE(runtime->allocate(f.view(*runtime), f.request("too-large", "a", 4)));
  EXPECT_TRUE(runtime->allocate(f.view(*runtime), f.request("allowed", "a", 3)));
}
TEST(RuntimePolicy, FailedPreallocationReturnsNoPartiallyConstructedRuntime) {
  Fixture f;
  auto d = f.discovery();
  d.pool_available_bytes["a"] = f.q;
  d.pool_available_bytes["alias-a"] = f.q;
  EXPECT_FALSE(make_policy_linux_runtime(d, f.policy("preallocate_fraction = 1"), f.options()));
  // A fresh independent construction succeeds; the failure owns no global state.
  EXPECT_TRUE(make_policy_linux_runtime(f.discovery(), f.policy("preallocate_fraction = 1"), f.options()));
}
TEST(RuntimePolicy, MissingCalibrationFileIsAnExplicitFileError) {
  auto result = open_runtime("does-not-exist.toml", {}, "calibration.toml");
  ASSERT_FALSE(result);
  EXPECT_EQ(std::get<Error>(result.state).code, "policy.schema");
  EXPECT_NE(std::get<Error>(result.state).message.find("cannot open TOML file"),std::string::npos);
}
TEST(RuntimePolicy, PlanEstimateReportsCapacityGroupFreshShortfallAcrossAliases) {
  Fixture f;
  auto options = f.options();
  options.group_hard_limits = {{"physical-bank", 4 * f.q}};
  auto runtime = take(make_linux_runtime(f.discovery(), options));
  MemoryRecipe recipe{};
  recipe.participant = options.participant;
  recipe.backings = {f.request("first", "a", 3), f.request("alias", "alias-a", 2)};
  const auto estimate = take(runtime->estimate(f.view(*runtime), recipe));
  EXPECT_FALSE(estimate.feasible);
  ASSERT_EQ(estimate.shortfalls.size(), 1u);
  const auto& shortage = estimate.shortfalls.front();
  EXPECT_EQ(shortage.kind, CapacityLimitKind::capacity_group);
  EXPECT_EQ(shortage.id, "physical-bank");
  EXPECT_EQ(shortage.requested_bytes, 5 * f.q);
  EXPECT_EQ(shortage.available_bytes, 4 * f.q);
  EXPECT_EQ(estimate.new_backing_bytes_by_capacity_group.at("physical-bank"), 5 * f.q);
}
TEST(RuntimePolicy, PlanEstimateReportsPoolQuotaWithoutGroupShortfall) {
  Fixture f;
  auto discovery = f.discovery();
  // A pool quota can be smaller than its aliased physical group's free sample.
  discovery.graph.pools.front().capacity_bytes = 2 * f.q;
  auto options = f.options();
  auto runtime = take(make_linux_runtime(std::move(discovery), options));
  MemoryRecipe recipe{};
  recipe.participant = options.participant;
  recipe.backings = {f.request("pool-quota", "a", 3)};
  const auto estimate = take(runtime->estimate(f.view(*runtime), recipe));
  EXPECT_FALSE(estimate.feasible);
  ASSERT_EQ(estimate.shortfalls.size(), 1u);
  const auto& shortage = estimate.shortfalls.front();
  EXPECT_EQ(shortage.kind, CapacityLimitKind::pool);
  EXPECT_EQ(shortage.id, "a");
  EXPECT_EQ(shortage.requested_bytes, 3 * f.q);
  EXPECT_EQ(shortage.available_bytes, 2 * f.q);
}
TEST(RuntimePolicy, PlanEstimateReportsSharedConstraintAcrossDistinctGroups) {
  Fixture f;
  auto options = f.options();
  options.shared_limits = {{"combined-cut", {"physical-bank", "other-bank"}, 5 * f.q}};
  auto runtime = take(make_linux_runtime(f.discovery(), options));
  MemoryRecipe recipe{};
  recipe.participant = options.participant;
  recipe.backings = {f.request("first", "a", 3), f.request("second", "b", 3)};
  const auto estimate = take(runtime->estimate(f.view(*runtime), recipe));
  EXPECT_FALSE(estimate.feasible);
  ASSERT_EQ(estimate.shortfalls.size(), 1u);
  const auto& shortage = estimate.shortfalls.front();
  EXPECT_EQ(shortage.kind, CapacityLimitKind::constraint);
  EXPECT_EQ(shortage.id, "combined-cut");
  EXPECT_EQ(shortage.requested_bytes, 6 * f.q);
  EXPECT_EQ(shortage.available_bytes, 5 * f.q);
}
TEST(RuntimePolicy, FeasiblePlanEstimateHasNoShortfallsOrAcceptedCapacity) {
  Fixture f;
  auto options = f.options();
  auto runtime = take(make_linux_runtime(f.discovery(), options));
  const auto view = f.view(*runtime);
  MemoryRecipe recipe{};
  recipe.participant = options.participant;
  recipe.backings = {f.request("fits", "a", 2)};
  const auto estimate = take(runtime->estimate(view, recipe));
  EXPECT_TRUE(estimate.feasible);
  EXPECT_TRUE(estimate.shortfalls.empty());
  for (const auto& group : runtime->snapshot(view).groups) {
    EXPECT_EQ(group.committed_bytes, 0u);
    EXPECT_EQ(group.unbacked_reserved_bytes, 0u);
  }
}
TEST(RuntimePolicy, PlanEstimateFreshShortfallExcludesSelectedCachedBacking) {
  Fixture f;
  auto options = f.options();
  options.group_hard_limits = {{"physical-bank", 4 * f.q}};
  auto runtime = take(make_linux_runtime(f.discovery(), options));
  const auto view = f.view(*runtime);
  ArenaOptions arena_options{"first", {"a"}, Limit{4 * f.q}, Preallocate{4 * f.q},
                            f.q, 0, f.q, BackingMode::shareable_remappable, true};
  auto arena = take(runtime->create_arena(view, arena_options));
  MemoryArena* caches[] = {arena.get()};
  MemoryRecipe recipe{};
  recipe.participant = options.participant;
  recipe.backings = {f.request("cached", "a", 4)};
  auto estimate = take(runtime->estimate(view, recipe, caches));
  EXPECT_TRUE(estimate.feasible);
  EXPECT_TRUE(estimate.shortfalls.empty());
  EXPECT_EQ(estimate.new_backing_bytes_by_capacity_group.at("physical-bank"), 0u);
  // An incompatible mode needs fresh backing even though total request <= idle.
  recipe.backings = {f.request("fresh", "a", 1)};
  recipe.backings.front().mode = BackingMode::private_memory;
  estimate = take(runtime->estimate(view, recipe, caches));
  EXPECT_FALSE(estimate.feasible);
  ASSERT_EQ(estimate.shortfalls.size(), 1u);
  const auto& shortage = estimate.shortfalls.front();
  EXPECT_EQ(shortage.kind, CapacityLimitKind::capacity_group);
  EXPECT_EQ(shortage.id, "physical-bank");
  EXPECT_EQ(shortage.requested_bytes, f.q);
  EXPECT_EQ(shortage.available_bytes, 0u);
}
TEST(RuntimePolicy, LocalDiscoveryDefaultsArbitraryRuntimeRootAndBindsHostQueues) {
  const auto snapshot = take(probe_linux_topology());
  U groups = 0;
  for (const auto& node : snapshot.nodes)
    if (!node.cpus.empty())
      ++groups;
  ASSERT_GT(groups, 0u);
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                    "build" / ("runtime-policy-template-" + std::to_string(::getpid()) + ".toml");
  std::filesystem::create_directories(path.parent_path());
  struct Remove {
    std::filesystem::path path;
    ~Remove() { std::error_code ec; std::filesystem::remove(path, ec); }
  } remove{path};
  {
    std::ofstream file(path);
    ASSERT_TRUE(file);
    file << "schema = \"memory-model/0.3\"\nname = \"local test\"\n[units]\nbandwidth = \"GiB/s\"\n"
            "[require]\n[groups]\ncustom_uma = {count = \"runtime\"}\n[agents]\ncompute = {per = \"custom_uma\", count = "
         << groups << ", bind = \"cpu-group\"}\n[pools]\nmemory = {per = \"compute\", capacity = \"discover\", bind = \"local-ddr\"}\n"
            "[resources]\n[paths]\n[transfers]\n";
  }
  auto runtime = take(open_runtime(path.string()));
  EXPECT_EQ(runtime->physical_graph().agents.size(), groups);
  EXPECT_FALSE(take(runtime->current_scope()).executable_agents.empty());
  LinuxRuntimeOptions invalid;
  invalid.runtime_counts = {{"not_a_group", 1}};
  EXPECT_FALSE(discover_linux_runtime(path.string(), LinuxCpuProfile::generic_cpu, invalid));
  invalid.runtime_counts = {{"custom_uma", 2}};
  EXPECT_FALSE(discover_linux_runtime(path.string(), LinuxCpuProfile::generic_cpu, invalid));
  EXPECT_TRUE(discover_linux_runtime());
}
} // namespace
} // namespace vecops::memory::model
