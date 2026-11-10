// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/ResourceModel.h"
#include "vecops/memory/Calibration.h"
#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace vecops::memory::model {
namespace {
constexpr std::uint64_t Q = 4096, GiB = 1ULL << 30;
ResourceGraph hardware() {
  ResourceGraph g;
  g.agents = {{"uma", {"execute"}, "host"}, {"cowos_superdie", {"execute"}, "host"}};
  g.capacity_groups = {{"shared_capacity", 16 * GiB}, {"other_capacity", 16 * GiB}};
  g.pools = {{"a", 8 * GiB, "host", Q, {Q}, "controller", "controller", "shared_capacity"},
             {"b", 8 * GiB, "host", Q, {Q}, "controller", "controller", "shared_capacity"},
             {"c", 16 * GiB, "host", Q, {Q}, "other", "other", "other_capacity"}};
  g.resources = {{"controller", "", 100.0, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}},
                 {"other", "", 100.0, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}},
                 {"infinity_fabric", "", 32.0, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}}};
  g.domains = {{"shared_domain", "arbitrary rule", {"a", "b"}}};
  for (const auto& agent : g.agents)
    for (const auto& pool : g.pools) {
      g.access.push_back({agent.id + ":" + pool.id,
                          "",
                          agent.id,
                          pool.id,
                          {AccessMode::read, AccessMode::write},
                          std::map<std::string, double>{{"read", 80}, {"write", 90}},
                          0.0,
                          MappingMode::direct,
                          {},
                          {},
                          {},
                          {},
                          {}});
    }
  Method copy{"cpu_copy", true, {}, 0.0, {}, {}};
  for (const auto& a : g.pools)
    for (const auto& b : g.pools)
      copy.routes.push_back(
        {a.id + ">" + b.id,
         {},
         {},
         a.id,
         b.id,
         {{*a.read_service, DemandKind::traffic, 1, 0}, {*b.write_service, DemandKind::traffic, 1, 0}}});
  g.methods.push_back(copy);
  return g;
}
ResourceView all(const ResourceModel& model) {
  return std::move(model.view(Scope{{"uma", "cowos_superdie"}, {"a", "b", "c"}, {}, {}})).value();
}
ResolvedPlacement place(std::string pool = "a", std::uint64_t n = 2 * Q) {
  return {"same-pattern", "host", Q, {{0, n, std::move(pool)}}};
}
MemoryRecipe recipe(std::size_t slots = 1) {
  MemoryRecipe r{};
  r.participant = "p0";
  r.layout_id = "layout";
  r.layout_generation = 1;
  r.addresses.regions = {{"va", "host", 4 * Q, Q}};
  r.backings = {{"storage", "p0", "a", 4 * Q, Q, BackingMode::shareable_remappable, {}}};
  r.mappings = {{"va", 0, "storage", 0, 4 * Q, {AccessMode::read, AccessMode::write}}};
  for (std::size_t i = 0; i < slots; ++i)
    r.addresses.slots.push_back({"slot" + std::to_string(i), "va", 0, 2 * Q, place(), {}});
  return r;
}
MemoryCommand copy(std::string source, std::uint64_t so, std::string target, std::uint64_t to,
                   std::uint64_t bytes = Q) {
  return {"copy",
          MemoryOpcode::copy,
          ByteRangeRef{MemoryRefKind::slot, std::move(source), so, bytes},
          ByteRangeRef{MemoryRefKind::slot, std::move(target), to, bytes},
          0,
          "cpu_copy",
          {},
          "uma",
          {},
          {}};
}
Activity traffic(std::string id, std::string resource = "controller", std::uint64_t size = GiB) {
  return {std::move(id), "", {}, size, {}, {{std::move(resource), DemandKind::traffic, 1, 0}}, 0, {}};
}
CostContext context(const ResourceView& v) {
  return {v.epoch, {}, {}, false};
}
ResourceGraph copy_fabric_hardware(bool same_service = false) {
  auto g = hardware();
  g.resources[0].bandwidth = 1000;
  g.resources[0].rule = "bank_rule";
  g.resources[1].bandwidth = 1000;
  g.resources[1].rule = "bank_rule";
  g.resources.push_back({"return_fabric", "", 64, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}});
  for (auto& edge : g.access) {
    edge.bandwidth.reset();
    if (edge.agent == "uma" && edge.pool == "a") {
      edge.read_fabric = {"infinity_fabric"};
      edge.write_fabric = {same_service ? "infinity_fabric" : "return_fabric"};
    }
  }
  for (auto& route : g.methods[0].routes) {
    route.demands[0].service = DemandKind::read;
    route.demands[1].service = DemandKind::write;
  }
  return g;
}
TransferQuery copy_query(TransferEndpoint source = {"host", {{0, GiB, "a"}}},
                         TransferEndpoint target = {"host", {{0, GiB, "c"}}}) {
  return {std::move(source), std::move(target), "cpu_copy", "uma", GiB, {}};
}
template <class CostModel>
CostPlan copy_plan(const CostModel& model, const ResourceView& v, const TransferQuery& query, std::size_t copies = 1) {
  MemoryCommandPlan commands;
  OperandEnvironment env{};
  for (std::size_t i = 0; i < copies; ++i) {
    const auto suffix = std::to_string(i);
    for (const auto& [name, endpoint, mode] : {std::tuple{"source" + suffix, query.source, AccessMode::read},
                                               std::tuple{"target" + suffix, query.target, AccessMode::write}}) {
      MemoryDescriptor memory{MemoryRefKind::external, name, endpoint.address_space, query.payload_bytes, 0, {}};
      for (std::size_t j = 0; j < endpoint.segments.size(); ++j) {
        const auto& s = endpoint.segments[j];
        memory.spans.push_back(
          {s.offset_bytes, s.size_bytes, name + "/physical/" + std::to_string(j), 0, {mode}, s.pool});
      }
      env.memories.push_back(std::move(memory));
    }
    commands.commands.push_back({"copy" + suffix,
                                 MemoryOpcode::copy,
                                 ByteRangeRef{MemoryRefKind::external, "source" + suffix, 0, query.payload_bytes},
                                 ByteRangeRef{MemoryRefKind::external, "target" + suffix, 0, query.payload_bytes},
                                 0,
                                 query.method,
                                 {},
                                 *query.executor,
                                 {},
                                 {}});
  }
  auto result = model.cost_plan(v, commands, env);
  if (!result)
    throw std::runtime_error(std::get<Error>(result.state).message);
  return std::move(result).value();
}
double payload_demand(const Activity& activity, const std::string& resource) {
  double out = 0;
  for (const auto& d : activity.demands)
    if (d.resource == resource)
      out += d.amount;
  return out;
}

TEST(ResourceModel, PrivateMovableRejectsPartialAliasExportAndSharingBeforeAdmission) {
  ResourceModel model(hardware()); auto view=all(model); auto r=recipe();
  r.backings.front().mode=BackingMode::private_movable;
  EXPECT_TRUE(model.validate_recipe(view,r));
  auto bad=r;bad.mappings.front().size_bytes=Q; EXPECT_FALSE(model.validate_recipe(view,bad));
  bad=r;bad.addresses.regions.push_back({"other","host",4*Q,Q});
  bad.mappings.push_back({"other",0,"storage",0,4*Q,{AccessMode::read}});EXPECT_FALSE(model.validate_recipe(view,bad));
  bad=r;bad.backings.front().export_to={"p1"};EXPECT_FALSE(model.validate_recipe(view,bad));
  bad=r;bad.addresses.slots.front().sharing_contract="contract";
  bad.sharing_contracts={{"contract",{"p0"},{"p0"},{"p0"},"publish"}};EXPECT_FALSE(model.validate_recipe(view,bad));
}

TEST(ResourceModel, ArbitraryLabelsAndSharedServiceSurviveScopeProjection) {
  ResourceModel m(hardware());
  auto projected = m.view({{"cowos_superdie"}, {"b"}, {"a"}, {{"cowos_superdie", 7}}}, 42);
  ASSERT_TRUE(projected);
  const auto& v = projected.value();
  EXPECT_EQ(v.epoch, 42);
  ASSERT_EQ(v.graph.resources.size(), 1);
  EXPECT_EQ(v.graph.resources[0].id, "controller");
  EXPECT_DOUBLE_EQ(*v.graph.resources[0].bandwidth, 100);
  ASSERT_EQ(v.graph.domains.size(), 1);
  EXPECT_EQ(v.graph.domains[0].members, (std::vector<std::string>{"b"}));
  EXPECT_EQ(*v.graph.capacity_groups[0].capacity_bytes, 16 * GiB);
  EXPECT_FALSE(m.view({{"missing"}, {"a"}, {}, {}}));
  EXPECT_FALSE(m.view({{"uma"}, {"a"}, {}, {{"cowos_superdie", 1}}}));
}
TEST(ResourceModel, ProjectionPreservesMethodMetadataAndIndependentRoutes) {
  auto g = hardware();
  g.methods[0].bandwidth = 72.5;
  g.methods[0].startup_us = 1.25;
  g.methods[0].registration_bits = 24;
  g.methods[0].routes[0].agent = "cowos_superdie";
  g.methods[0].routes[0].controller = "cowos_superdie";
  ResourceModel model(g);
  auto result = model.view({{"cowos_superdie"}, {"b"}, {"a"}, {}}, 42);
  ASSERT_TRUE(result);
  auto& v = result.value();
  ASSERT_EQ(v.graph.methods.size(), 1);
  const auto& method = v.graph.methods[0];
  EXPECT_EQ(method.id, g.methods[0].id);
  EXPECT_EQ(method.request_executor, g.methods[0].request_executor);
  EXPECT_EQ(method.bandwidth, g.methods[0].bandwidth);
  EXPECT_EQ(method.startup_us, g.methods[0].startup_us);
  EXPECT_EQ(method.registration_bits, g.methods[0].registration_bits);
  ASSERT_EQ(method.routes.size(), 4);
  const auto& route = method.routes[0];
  EXPECT_EQ(route.id, g.methods[0].routes[0].id);
  EXPECT_EQ(route.agent, g.methods[0].routes[0].agent);
  EXPECT_EQ(route.controller, g.methods[0].routes[0].controller);
  EXPECT_EQ(route.source, "a");
  EXPECT_EQ(route.target, "a");
  ASSERT_EQ(route.demands.size(), 2);
  EXPECT_EQ(route.demands[0].resource, "controller");
  EXPECT_EQ(route.demands[0].service, DemandKind::traffic);
  EXPECT_DOUBLE_EQ(route.demands[0].amount, 1);
  EXPECT_EQ(route.demands[0].per_invocation_bytes, 0);
  ASSERT_EQ(v.graph.domains.size(), 1);
  EXPECT_EQ(v.graph.domains[0].id, g.domains[0].id);
  EXPECT_EQ(v.graph.domains[0].rule, g.domains[0].rule);
  EXPECT_EQ(v.graph.domains[0].members, (std::vector<std::string>{"b"}));
  v.graph.methods[0].routes[0].demands.clear();
  v.graph.domains[0].members.clear();
  auto again = model.view({{"uma", "cowos_superdie"}, {"a", "b", "c"}, {}, {}});
  ASSERT_TRUE(again);
  EXPECT_EQ(again.value().graph.methods[0].routes.size(), 9);
  EXPECT_EQ(again.value().graph.methods[0].routes[0].demands.size(), 2);
  EXPECT_EQ(again.value().graph.domains[0].members, g.domains[0].members);
}
TEST(ResourceModel, InvalidHardwareRejectedBeforeUse) {
  auto g = hardware();
  g.agents.push_back(g.agents[0]);
  EXPECT_THROW(ResourceModel{g}, std::invalid_argument);
  g = hardware();
  g.resources[0].bandwidth = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(ResourceModel{g}, std::invalid_argument);
  g = hardware();
  g.pools[0].capacity_group = "missing";
  EXPECT_THROW(ResourceModel{g}, std::invalid_argument);
}
TEST(ResourceModel, WeightedPlacementAndPhaseUseExactPageUnits) {
  ResourceModel m(hardware());
  auto v = all(m);
  PlacementPlan p{"host", Q, {{"a", 3}, {"c", 7}}, {}, 0};
  auto resolved = m.resolve_placement(v, p, 10 * Q);
  ASSERT_TRUE(resolved);
  std::map<std::string, std::uint64_t> charge;
  for (const auto& s : resolved.value().segments)
    charge[s.pool] += s.size_bytes;
  EXPECT_EQ(charge["a"], 3 * Q);
  EXPECT_EQ(charge["c"], 7 * Q);
  p.phase = 3;
  auto shifted = m.resolve_placement(v, p, Q);
  ASSERT_TRUE(shifted);
  EXPECT_EQ(shifted.value().segments[0].pool, "c");
  p.cycle = {"a", "c"};
  EXPECT_FALSE(m.resolve_placement(v, p, 10 * Q));
  p.cycle.clear();
  p.stripe_bytes = 64;
  EXPECT_FALSE(m.resolve_placement(v, p, Q));
}
TEST(ResourceModel, ContiguousLargePlacementHasConstantMetadata) {
  auto g = hardware();
  g.capacity_groups[0].capacity_bytes = 128 * GiB;
  g.pools[0].capacity_bytes = 128 * GiB;
  ResourceModel m(g);
  auto result = m.resolve_placement(all(m), {"host", Q, {{"a", 1}}, {}, 0}, 64 * GiB);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().segments.size(), 1);
  EXPECT_EQ(result.value().segments[0].size_bytes, 64 * GiB);
  EXPECT_THROW(page_charge(std::numeric_limits<std::uint64_t>::max(), Q), std::overflow_error);
}
TEST(ResourceModel, OverlapCompatibilityChecksPhaseAndEverySegment) {
  ResourceModel m(hardware());
  auto p = place();
  EXPECT_TRUE(m.compatible_overlap(p, 0, p, 0, Q).value());
  auto b = p;
  b.segments = {{0, Q, "a"}, {Q, Q, "b"}};
  EXPECT_FALSE(m.compatible_overlap(p, 0, b, 0, 2 * Q).value());
  EXPECT_FALSE(m.compatible_overlap(p, 1, p, 0, Q).value());
  b = p;
  b.key = "different contract";
  EXPECT_FALSE(m.compatible_overlap(p, 0, b, 0, Q).value());
  EXPECT_FALSE(m.compatible_overlap(p, 0, p, 0, 3 * Q));
}
TEST(ResourceModel, RecipeRejectsHolesAndSubpagePoolMismatch) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto r = recipe();
  EXPECT_TRUE(m.validate_recipe(v, r));
  r.mappings[0].size_bytes = Q;
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe();
  r.addresses.slots[0].placement.segments = {{0, Q / 2, "a"}, {Q / 2, 3 * Q / 2, "b"}};
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe();
  r.mappings[0].backing_offset_bytes = Q;
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe();
  r.mappings.push_back(r.mappings[0]);
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe();
  r.addresses.slots[0].placement.segments[0].pool = "unknown";
  EXPECT_FALSE(m.validate_recipe(v, r));
}
TEST(ResourceModel, CapacityIsChargedOncePerOwnedBacking) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto r = recipe(4096);
  EXPECT_TRUE(m.validate_recipe(v, r));
  r.backings[0].bytes = 12 * GiB;
  r.backings.push_back({"another", "p0", "b", 12 * GiB, Q, BackingMode::private_memory, {}});
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe();
  r.backings[0].owner = "p1";
  EXPECT_FALSE(m.validate_recipe(v, r));
  r.backings[0].export_to = {"p0"};
  EXPECT_TRUE(m.validate_recipe(v, r));
  r.backings[0].mode = BackingMode::private_memory;
  EXPECT_FALSE(m.validate_recipe(v, r));
}
TEST(ResourceModel, SameOffsetSharingContractsMustMatch) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto r = recipe(2);
  r.addresses.slots[1].placement.key = "other";
  EXPECT_FALSE(m.validate_recipe(v, r));
  r = recipe(2);
  r.sharing_contracts = {{"publish", {"p0", "p1"}, {"p0"}, {"p1"}, "publish-acquire-retire"}};
  r.addresses.slots[1].sharing_contract = "publish";
  EXPECT_FALSE(m.validate_recipe(v, r));
  r.addresses.slots[0].sharing_contract = "publish";
  EXPECT_TRUE(m.validate_recipe(v, r));
}
TEST(ResourceModel, PhysicalAliasesAreComparedAcrossSlotAndVirtualRegionIds) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto r = recipe(2);
  r.addresses.regions.push_back({"other-va", "host", 4 * Q, Q});
  r.mappings.push_back({"other-va", 0, "storage", 0, 4 * Q, {AccessMode::read, AccessMode::write}});
  r.addresses.slots[1].region = "other-va";
  auto env = m.describe_operands(v, r);
  ASSERT_TRUE(env);
  EXPECT_FALSE(m.validate_commands(v, {{copy("slot0", 0, "slot1", 0)}}, env.value()));
  EXPECT_TRUE(m.validate_commands(v, {{copy("slot0", Q, "slot1", 0)}}, env.value()));
}
TEST(ResourceModel, ReadonlyMappingsAndStaleGenerationsRejectWrites) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto r = recipe();
  r.mappings[0].access = {AccessMode::read};
  auto env = m.describe_operands(v, r);
  ASSERT_TRUE(env);
  MemoryCommand fill{
    "fill", MemoryOpcode::fill, {}, ByteRangeRef{MemoryRefKind::slot, "slot0", 0, Q}, 0, "cpu_copy", {}, "uma", {}, {}};
  EXPECT_FALSE(m.validate_commands(v, {{fill}}, env.value()));
  env.value().memories[0].spans[0].access.push_back(AccessMode::write);
  env.value().invocation = InvocationStamp{"call", 2};
  EXPECT_FALSE(m.validate_commands(v, {{fill}}, env.value()));
  env.value().memories[0].generation = 2;
  EXPECT_TRUE(m.validate_commands(v, {{fill}}, env.value()));
}
TEST(ResourceModel, UnknownPhysicalAliasesAreNotProvenDisjoint) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto env = m.describe_operands(v, recipe(2));
  ASSERT_TRUE(env);
  env.value().memories[0].spans[0].physical_identity.reset();
  EXPECT_FALSE(m.validate_commands(v, {{copy("slot0", Q, "slot1", 0)}}, env.value()));
}
TEST(ResourceModel, ExternalPlanSymbolsAreInvocationLocalAndSeparateFromSlots) {
  ResourceModel m(hardware());
  auto v = all(m);
  MemoryDescriptor ext{MemoryRefKind::external, "slot0", "host", Q, 1, {{0, Q, "import1", 0, {AccessMode::read}, {}}}};
  auto env = m.describe_operands(v, recipe(), std::span{&ext, 1});
  ASSERT_TRUE(env);
  ASSERT_EQ(env.value().memories.size(), 2);
  auto c = copy("slot0", 0, "slot0", 0);
  c.source->kind = MemoryRefKind::external;
  EXPECT_TRUE(m.validate_commands(v, {{c}}, env.value()));
  ext.spans[0].physical_identity = env.value().memories[1].spans[0].physical_identity;
  env = m.describe_operands(v, recipe(), std::span{&ext, 1});
  ASSERT_TRUE(env);
  EXPECT_FALSE(m.validate_commands(v, {{c}}, env.value()));
}
TEST(ResourceModel, InvalidRangesFillAndDependencyCyclesAreRejected) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto env = m.describe_operands(v, recipe(2)).value();
  auto c = copy("slot0", std::numeric_limits<std::uint64_t>::max(), "slot1", 0);
  EXPECT_FALSE(m.validate_commands(v, {{c}}, env));
  c = copy("slot0", Q, "slot1", 0);
  c.dependencies = {c.id};
  EXPECT_FALSE(m.validate_commands(v, {{c}}, env));
  c.dependencies = {"unknown"};
  EXPECT_FALSE(m.validate_commands(v, {{c}}, env));
  c.dependencies.clear();
  c.source->size_bytes = 0;
  EXPECT_FALSE(m.validate_commands(v, {{c}}, env));
  c = copy("slot0", Q, "slot1", 0);
  EXPECT_FALSE(m.validate_commands(v, {{c, c}}, env));
}
TEST(ResourceModel, BudgetIncludesCacheAndSubtractsOutstandingGrants) {
  PoolBudget b{};
  b.hard_limit_bytes = 1000;
  b.system_available_bytes = 800;
  b.live_bytes = 200;
  b.idle_available_bytes = 100;
  b.retired_bytes = 100;
  b.committed_bytes = 400;
  b.unbacked_reserved_bytes = 300;
  ASSERT_TRUE(budget_available(b));
  EXPECT_EQ(budget_available(b).value(), 400);
  b.committed_bytes = 0;
  EXPECT_FALSE(budget_available(b));
}
TEST(ResourceModel, SharedBandwidthAndDependenciesHaveDifferentSchedules) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto c = context(v);
  auto a = traffic("a"), b = traffic("b");
  EXPECT_NEAR(m.estimate_schedule(v, {{a}}, c).value().total_us, 10000, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 20000, 1e-7);
  b.demands[0].resource = "other";
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 10000, 1e-7);
  b.dependencies = {"a"};
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 20000, 1e-7);
  a.dependencies = {"b"};
  EXPECT_FALSE(m.estimate_schedule(v, {{a, b}}, c));
}
TEST(ResourceModel, ReadWriteDirectionsConsumeOneNormalizedSharedService) {
  auto g = hardware();
  g.resources[0].sharing = SharingMode::read_write;
  g.resources[0].read = 100;
  g.resources[0].write = 50;
  g.resources[0].bandwidth.reset();
  ResourceModel m(g);
  auto v = all(m);
  auto a = traffic("a"), b = traffic("b");
  a.demands[0].service = DemandKind::read;
  b.demands[0].service = DemandKind::write;
  EXPECT_NEAR(m.estimate_schedule(v, {{a}}, context(v)).value().total_us, 10000, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, {{b}}, context(v)).value().total_us, 20000, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, context(v)).value().total_us, 30000, 1e-7);
  auto other_read = a;
  other_read.id = "second-read";
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b, other_read}}, context(v)).value().total_us, 40000, 1e-7);
  b.demands[0].service = DemandKind::read;
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, context(v)).value().total_us, 20000, 1e-7);
}
TEST(ResourceModel, SamePoolCopyAddsAsymmetricReadAndWriteServiceTimes) {
  auto g = hardware();
  g.resources[0].sharing = SharingMode::read_write;
  g.resources[0].read = 100;
  g.resources[0].write = 25;
  g.resources[0].bandwidth.reset();
  for (auto& access : g.access)
    access.bandwidth.reset();
  auto& routes = g.methods[0].routes;
  auto& route = *std::find_if(routes.begin(), routes.end(), [](const auto& r) { return r.id == "a>a"; });
  route.demands = {{"controller", DemandKind::read, 1, 0}, {"controller", DemandKind::write, 1, 0}};
  ResourceModel m(g);
  auto v = all(m);
  TransferEndpoint memory{"host", {{0, GiB, "a"}}};
  TransferQuery query{memory, memory, "cpu_copy", "uma", GiB, {}};
  auto cost = m.estimate(v, query, context(v));
  ASSERT_TRUE(cost);
  EXPECT_NEAR(cost.value().total_us, 10000 + 40000, 1e-7);
  // Exercise command materialization and explicit schedule costing as well.
  auto r = recipe();
  auto env = m.describe_operands(v, r).value();
  MemoryCommand command{"copy",
                        MemoryOpcode::copy,
                        ByteRangeRef{MemoryRefKind::slot, "slot0", 0, Q},
                        ByteRangeRef{MemoryRefKind::slot, "slot0", Q, Q},
                        0,
                        "cpu_copy",
                        {},
                        "uma",
                        {},
                        {}};
  auto plan = m.cost_plan(v, {{command}}, env);
  ASSERT_TRUE(plan);
  EXPECT_NEAR(m.estimate_schedule(v, plan.value(), context(v)).value().total_us, double(Q) / GiB * 50000, 1e-9);
}
TEST(ResourceModel, ReadWriteFixedTrafficUsesSameSharedServiceAndMissingRatesStayUnknown) {
  auto g = hardware();
  g.resources[0].sharing = SharingMode::read_write;
  g.resources[0].read = 100;
  g.resources[0].write = 50;
  g.resources[0].bandwidth.reset();
  ResourceModel m(g);
  auto v = all(m);
  auto a = traffic("fixed");
  a.fixed_us = 7;
  a.demands = {{"controller", DemandKind::read, 1, GiB / 2}, {"controller", DemandKind::write, 0, GiB / 4}};
  auto cost = m.estimate_schedule(v, {{a}}, context(v));
  ASSERT_TRUE(cost);
  EXPECT_NEAR(cost.value().total_us, 7 + 15000 + 5000, 1e-7);
  g.resources[0].write.reset();
  ResourceModel missing(g);
  auto unknown = missing.estimate_schedule(all(missing), {{a}}, context(all(missing)));
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown.value().quality, CostQuality::incomplete);
  EXPECT_FALSE(unknown.value().missing_facts.empty());
}
TEST(ResourceModel, IndependentDuplexRequiresSeparatePhysicalResources) {
  auto g = hardware();
  g.resources[0].sharing = SharingMode::read_write;
  g.resources[0].read = 100;
  g.resources[0].write = 50;
  g.resources[0].bandwidth.reset();
  g.resources[1].sharing = SharingMode::read_write;
  g.resources[1].read = 100;
  g.resources[1].write = 50;
  g.resources[1].bandwidth.reset();
  ResourceModel m(g);
  auto v = all(m);
  auto a = traffic("read"), b = traffic("write");
  a.demands = {{"controller", DemandKind::read, 1, 0}};
  b.demands = {{"other", DemandKind::write, 1, 0}};
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, context(v)).value().total_us, 20000, 1e-7);
  b.dependencies = {"read"};
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, context(v)).value().total_us, 30000, 1e-7);
}
TEST(ResourceModel, ProgressiveFillingRedistributesUnusedBandwidth) {
  auto g = hardware();
  g.resources[1].bandwidth = 25;
  ResourceModel m(g);
  auto v = all(m);
  auto c = context(v);
  auto background = traffic("background");
  background.demands.push_back({"other", DemandKind::traffic, 1, 0});
  c.active = {background};
  EXPECT_NEAR(m.estimate_schedule(v, {{traffic("foreground")}}, c).value().total_us, 1e6 / 75.0, 1e-6);
}
TEST(ResourceModel, OccupancyAndExecutorSlotsGateWork) {
  auto g = hardware();
  g.resources.push_back({"executor/uma", "", {}, {}, {}, SharingMode::shared, ResourceKind::execution_slots, 1});
  // Keep an occupancy-only resource in projected views via an explicit route.
  g.methods[0].routes[0].demands.push_back({"executor/uma", DemandKind::occupancy, 1, 0});
  ResourceModel m(g);
  auto v = all(m);
  v.execution_slots = {{"uma", 2}};
  auto c = context(v);
  Activity a{"a", "", {}, 0, {}, {{"executor/uma", DemandKind::occupancy, 1, 0}}, 10, "uma"}, b = a;
  b.id = "b";
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 20, 1e-7);
  a.demands.clear();
  b.demands.clear();
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 10, 1e-7);
  c.execution_slots = {{"uma", 1}};
  EXPECT_NEAR(m.estimate_schedule(v, {{a, b}}, c).value().total_us, 20, 1e-7);
}
TEST(ResourceModel, StaleContextAndUnknownCostsCannotPretendCalibration) {
  ResourceModel m(hardware());
  auto v = all(m);
  auto c = context(v);
  c.view_epoch = 2;
  EXPECT_FALSE(m.estimate_schedule(v, {{traffic("a")}}, c));
  c = context(v);
  auto a = traffic("a");
  a.demands.clear();
  auto e = m.estimate_schedule(v, {{a}}, c);
  ASSERT_TRUE(e);
  EXPECT_EQ(e.value().quality, CostQuality::incomplete);
  EXPECT_FALSE(e.value().missing_facts.empty());
}
TEST(ResourceModel, TransferCountsSourceAndTargetControllerTraffic) {
  ResourceModel m(hardware());
  auto v = all(m);
  TransferQuery q{{"host", {{0, GiB, "a"}}}, {"host", {{0, GiB, "b"}}}, "cpu_copy", "uma", GiB, {}};
  auto e = m.estimate(v, q, context(v));
  ASSERT_TRUE(e);
  EXPECT_NEAR(e.value().total_us, 20000, 1e-7);
  q.executor = "not_allowed";
  EXPECT_FALSE(m.estimate(v, q, context(v)));
}
TEST(ResourceModel, AccessUsesPathLimitAlongsideSharedController) {
  ResourceModel m(hardware());
  auto v = all(m);
  AccessQuery q{"uma", {"host", {{0, GiB, "a"}}}, AccessMode::read, GiB, {}};
  auto e = m.estimate_access(v, q, context(v));
  ASSERT_TRUE(e);
  EXPECT_NEAR(e.value().total_us, 12500, 1e-7);
}
TEST(ResourceModel, RepeatedSegmentsDoNotMultiplyAnAccessPathCap) {
  ResourceModel m(hardware());
  auto v = all(m);
  AccessQuery q{"uma",
                {"host", {{0, GiB / 4, "a"}, {GiB / 4, GiB / 2, "c"}, {3 * GiB / 4, GiB / 4, "a"}}},
                AccessMode::read,
                GiB,
                {}};
  auto result = m.estimate_access(v, q, context(v));
  ASSERT_TRUE(result);
  EXPECT_NEAR(result.value().total_us, 6250, 1e-7); // each path reads half at 80 GiB/s
}
TEST(ResourceModel, CopyToIndependentControllersStillRespectsTheMeasuredAccessPath) {
  ResourceModel m(hardware());
  auto v = all(m);
  TransferQuery q{{"host", {{0, GiB, "a"}}}, {"host", {{0, GiB, "c"}}}, "cpu_copy", "uma", GiB, {}};
  auto result = m.estimate(v, q, context(v));
  ASSERT_TRUE(result);
  EXPECT_NEAR(result.value().total_us, 12500, 1e-7);
}
TEST(ResourceModel, UnknownBandwidthReportsIncompleteCostInsteadOfInventingHardware) {
  auto g = hardware();
  g.resources[0].bandwidth.reset();
  ResourceModel m(g);
  auto v = all(m);
  auto result = m.estimate_schedule(v, {{traffic("unknown-service")}}, context(v));
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().quality, CostQuality::incomplete);
  EXPECT_FALSE(result.value().missing_facts.empty());
}
TEST(ResourceModel, CandidateRoutesFollowAlignedPiecesRatherThanCartesianProducts) {
  auto g = hardware();
  g.methods[0].routes.erase(std::remove_if(g.methods[0].routes.begin(), g.methods[0].routes.end(),
                                           [](const auto& r) { return r.source != r.target; }),
                            g.methods[0].routes.end());
  ResourceModel m(g);
  auto v = all(m);
  TransferEndpoint end{"host", {{0, GiB / 2, "a"}, {GiB / 2, GiB / 2, "c"}}};
  auto result = m.candidates(v, end, end);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0].routes.size(), 2u);
}
TEST(ResourceModel, CpuCopyConcretePlansCountRequesterFabricAndDoNotDuplicateControllers) {
  ResourceModel m(copy_fabric_hardware());
  const auto v = all(m);
  const auto query = copy_query();
  const auto plan = copy_plan(m, v, query);
  ASSERT_EQ(plan.activities.size(), 1u);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), 1);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "controller"), 1);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "other"), 1);
  auto direct = m.estimate(v, query, context(v));
  auto scheduled = m.estimate_schedule(v, plan, context(v));
  ASSERT_TRUE(direct);
  ASSERT_TRUE(scheduled);
  EXPECT_NEAR(direct.value().total_us, 31250, 1e-7);
  EXPECT_NEAR(scheduled.value().total_us, direct.value().total_us, 1e-7);
  // A concrete plan is reusable; the scheduler must not add its access work again.
  EXPECT_NEAR(m.estimate_schedule(v, plan, context(v)).value().total_us, 31250, 1e-7);
}
TEST(ResourceModel, CpuCopyConcurrentDirectionsShareOnlyTheirActualFabricServices) {
  ResourceModel m(copy_fabric_hardware());
  const auto v = all(m);
  auto query = copy_query();
  const auto two = copy_plan(m, v, query, 2);
  EXPECT_NEAR(m.estimate_schedule(v, two, context(v)).value().total_us, 62500, 1e-7);
  auto active = context(v);
  active.active = {two.activities[0]};
  EXPECT_NEAR(m.estimate(v, query, active).value().total_us, 62500, 1e-7);
  std::swap(query.source, query.target);
  const auto opposite = copy_plan(m, v, query);
  auto concurrent = two;
  concurrent.activities[1] = opposite.activities[0];
  concurrent.activities[1].id = "opposite";
  EXPECT_DOUBLE_EQ(payload_demand(concurrent.activities[1], "return_fabric"), 1);
  EXPECT_DOUBLE_EQ(payload_demand(concurrent.activities[1], "infinity_fabric"), 0);
  EXPECT_NEAR(m.estimate_schedule(v, concurrent, context(v)).value().total_us, 31250, 1e-7);
}
TEST(ResourceModel, CpuCopyReadAndWriteOnTheSameFabricAreTwoRealTraversals) {
  ResourceModel m(copy_fabric_hardware(true));
  const auto v = all(m);
  const auto query = copy_query({"host", {{0, GiB, "a"}}}, {"host", {{0, GiB, "a"}}});
  const auto plan = copy_plan(m, v, query);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), 2);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "controller"), 2);
  EXPECT_NEAR(m.estimate(v, query, context(v)).value().total_us, 62500, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, plan, context(v)).value().total_us, 62500, 1e-7);
}
TEST(ResourceModel, CpuCopySegmentWeightsAndRepeatedPathsPreserveExactTrafficFractions) {
  ResourceModel m(copy_fabric_hardware());
  const auto v = all(m);
  auto query = copy_query({"host", {{0, GiB / 4, "a"}, {GiB / 4, 3 * GiB / 4, "c"}}});
  auto plan = copy_plan(m, v, query);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), .25);
  EXPECT_NEAR(m.estimate(v, query, context(v)).value().total_us, 7812.5, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, plan, context(v)).value().total_us, 7812.5, 1e-7);
  query.source.segments = {{0, GiB / 4, "a"}, {GiB / 4, GiB / 2, "c"}, {3 * GiB / 4, GiB / 4, "a"}};
  plan = copy_plan(m, v, query);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), .5);
  EXPECT_NEAR(m.estimate(v, query, context(v)).value().total_us, 15625, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, plan, context(v)).value().total_us, 15625, 1e-7);
}
TEST(ResourceModel, ExplicitRouteFabricWorkAddsToAccessWorkAndFixedBytesOccurOncePerRoute) {
  auto g = copy_fabric_hardware(true);
  auto& route =
    *std::find_if(g.methods[0].routes.begin(), g.methods[0].routes.end(), [](const auto& r) { return r.id == "a>a"; });
  route.demands.push_back({"infinity_fabric", DemandKind::traffic, .5, GiB / 2});
  ResourceModel m(g);
  const auto v = all(m);
  const TransferEndpoint memory{"host", {{0, GiB / 2, "a"}, {GiB / 2, GiB / 2, "a"}}};
  const auto query = copy_query(memory, memory);
  const auto plan = copy_plan(m, v, query);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), 2.5);
  std::uint64_t fixed = 0;
  for (const auto& d : plan.activities[0].demands)
    if (d.resource == "infinity_fabric")
      fixed += d.per_invocation_bytes;
  EXPECT_EQ(fixed, GiB / 2);
  EXPECT_NEAR(m.estimate(v, query, context(v)).value().total_us, 93750, 1e-7);
  EXPECT_NEAR(m.estimate_schedule(v, plan, context(v)).value().total_us, 93750, 1e-7);
}
TEST(ResourceModel, CalibrationFacadePreservesConcreteCpuCopyFabricWithoutLosingOrReaddingIt) {
  ResourceModel m(copy_fabric_hardware(true));
  const auto v = all(m);
  auto overlay = parse_calibration(R"(
schema='memory-calibration/0.4'
name='synthetic copy contract'
[units]
bandwidth='GiB/s'
[models]
'transfer:cpu_copy'='copy_curve'
'resource:bank_rule'='write_work'
[profiles.copy_curve]
size=['1 GiB']
bandwidth=[128]
[profiles.write_work]
write_cost={copy=2}
)",
                                         m.graph(), {});
  ASSERT_TRUE(overlay);
  CalibratedCostModel calibrated(m, overlay.value());
  const auto query = copy_query({"host", {{0, GiB, "a"}}}, {"host", {{0, GiB, "a"}}});
  const auto plan = copy_plan(calibrated, v, query);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "infinity_fabric"), 2);
  EXPECT_DOUBLE_EQ(payload_demand(plan.activities[0], "controller"), 3);
  EXPECT_NEAR(calibrated.estimate(v, query, context(v)).value().total_us, 62500, 1e-7);
  EXPECT_NEAR(calibrated.estimate_schedule(v, plan, context(v)).value().total_us, 62500, 1e-7);
}
} // namespace
} // namespace vecops::memory::model
