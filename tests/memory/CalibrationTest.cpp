// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/Calibration.h"
#include "vecops/memory/LinuxRuntime.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
constexpr U GiB = 1ULL << 30, Q = 4096;
template <class T>
T take(Result<T> r) {
  if (!r)
    throw std::runtime_error(std::get<Error>(r.state).code + ": " + std::get<Error>(r.state).message);
  return std::move(r).value();
}
ResourceGraph hardware() {
  ResourceGraph g;
  g.agents = {{"opaque/requester", {"host_load_store"}, "host"}, {"opaque/device", {"dma"}, "host"}};
  g.capacity_groups = {{"a", 16 * GiB}, {"b", 16 * GiB}};
  g.pools = {{"a", 16 * GiB, "host", Q, {Q}, "bank", "bank", "a"},
             {"b", 16 * GiB, "host", Q, {Q}, "bank", "bank", "b"}};
  g.resources = {{"bank", "controller-rule", {}, 1000, 1000, SharingMode::read_write, ResourceKind::bandwidth, {}}};
  for (const auto& p : g.pools)
    g.access.push_back({"access/" + p.id,
                        "any-rule",
                        "opaque/requester",
                        p.id,
                        {AccessMode::read, AccessMode::write},
                        std::map<std::string, double>{{"read", 80}, {"write", 80}},
                        0,
                        MappingMode::direct,
                        {},
                        "bank",
                        "bank",
                        {},
                        {}});
  Method copy{"move", true, {}, 0, {}, {}}, dma{"dma", false, {}, 0, {}, {}};
  for (const auto& s : g.pools)
    for (const auto& t : g.pools) {
      std::vector<ResourceDemand> ds = {{"bank", DemandKind::read, 1, 0}, {"bank", DemandKind::write, 1, 0}};
      copy.routes.push_back({s.id + ">" + t.id, {}, {}, s.id, t.id, ds});
      dma.routes.push_back({"dma/" + s.id + ">" + t.id, "opaque/device", "opaque/requester", s.id, t.id, ds});
    }
  g.methods = {copy, dma};
  return g;
}
ResourceView view(const ResourceModel& model) {
  return take(model.view({{"opaque/requester", "opaque/device"}, {"a", "b"}, {"a", "b"}, {{"opaque/requester", 37}}}));
}
std::string overlay(const std::string& models, const std::string& profiles) {
  return "schema='memory-calibration/0.4'\nname='test'\n[units]\nbandwidth='GiB/s'\n[models]\n" + models + "\n" +
         profiles;
}
std::shared_ptr<const Calibration> parse(const std::string& text, const ResourceGraph& graph) {
  return take(parse_calibration(text, graph, {}));
}
AccessQuery access(U bytes = GiB) {
  return {"opaque/requester", {"host", {{0, bytes, "a"}}}, AccessMode::read, bytes, {}};
}
TransferQuery transfer(std::string method = "move", U bytes = GiB) {
  return {{"host", {{0, bytes, "a"}}},
          {"host", {{0, bytes, "b"}}},
          method,
          method == "dma" ? "opaque/device" : "opaque/requester",
          bytes,
          {}};
}
TEST(Calibration, ExplicitControlsAreNotExecutionSlotsAndPhysicalGraphStaysImmutable) {
  ResourceModel base(hardware());
  auto v = view(base);
  auto c =
    parse(overlay("'access:any-rule'='p'", "[profiles.p]\nread={cpu_cores=[16,37],bandwidth=[40,60]}"), base.graph());
  CalibratedCostModel unknown(base, c), known(base, c, {{"opaque/requester", {16, ""}}});
  CostContext ctx{v.epoch, {}, {{"opaque/requester", 16}}, false};
  EXPECT_NEAR(take(unknown.estimate_access(v, access(), ctx)).total_us, 12500, 1e-6);
  EXPECT_NEAR(take(known.estimate_access(v, access(), ctx)).total_us, 25000, 1e-6);
  EXPECT_EQ(take(known.estimate_access(v, access(), ctx)).quality, CostQuality::reference);
  EXPECT_EQ(v.graph.access.front().bandwidth->at("read"), 80);
  EXPECT_EQ(base.graph().access.front().bandwidth->at("read"), 80);
  EXPECT_TRUE(take(unknown.explain_access(v, access())).empty());
}
TEST(Calibration, ExactImplementationAndProvenanceAreAuditableWithoutPromotingAggregateQuality) {
  ResourceModel base(hardware());
  auto v = view(base);
  auto c = parse(overlay("'access:any-rule'='p'", R"(
[profiles.p]
provenance={quality='calibrated',source='sha256:measurement',implementation='stream-v7'}
read={cpu_cores=[16],bandwidth=[40]}
)"),
                 base.graph());
  CalibratedCostModel wrong(base, c, {{"opaque/requester", {16, "memcpy"}}});
  CalibratedCostModel right(base, c, {{"opaque/requester", {16, "stream-v7"}}});
  EXPECT_TRUE(take(wrong.explain_access(v, access())).empty());
  const auto selected = take(right.explain_access(v, access()));
  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0].provenance, CostQuality::calibrated);
  EXPECT_EQ(selected[0].source, "sha256:measurement");
  const auto cost = take(right.estimate_access(v, access(), {v.epoch, {}, {}, false}));
  EXPECT_EQ(cost.quality, CostQuality::reference);
  EXPECT_TRUE(cost.missing_facts.empty());
}
TEST(Calibration, SizeInterpolatesServiceTimeWithinBoundsAndNeverExtrapolates) {
  ResourceModel base(hardware());
  auto v = view(base);
  auto c = parse(overlay("'access:any-rule'='p'", "[profiles.p]\nread={size=['1 GiB','3 GiB'],bandwidth=[10,15]}"),
                 base.graph());
  CalibratedCostModel model(base, c);
  CostContext ctx{v.epoch, {}, {}, false};
  EXPECT_NEAR(take(model.estimate_access(v, access(2 * GiB), ctx)).total_us, 150000, 1e-6);
  EXPECT_TRUE(take(model.explain_access(v, access(2 * GiB)))[0].interpolated);
  EXPECT_NEAR(take(model.estimate_access(v, access(4 * GiB), ctx)).total_us, 50000, 1e-6);
  EXPECT_TRUE(take(model.explain_access(v, access(4 * GiB))).empty());
}
TEST(Calibration, TransferCapsRemainPrivateWhilePhysicalServicesRemainShared) {
  ResourceModel base(hardware());
  auto v = view(base);
  auto c = parse(overlay("'transfer:move'='p'", "[profiles.p]\nsize=['1 GiB']\nbandwidth=[32]"), base.graph());
  CalibratedCostModel model(base, c);
  const auto q = transfer();
  CostContext ctx{v.epoch, {}, {}, false};
  EXPECT_NEAR(take(model.estimate(v, q, ctx)).total_us, 31250, 1e-6);
  CostPlan p{{{"first",
               "move",
               "a>b",
               GiB,
               {},
               {{"bank", DemandKind::read, 1, 0}, {"bank", DemandKind::write, 1, 0}},
               0,
               "opaque/requester"},
              {"second",
               "move",
               "a>b",
               GiB,
               {},
               {{"bank", DemandKind::read, 1, 0}, {"bank", DemandKind::write, 1, 0}},
               0,
               "opaque/requester"}}};
  EXPECT_NEAR(take(model.estimate_schedule(v, p, ctx)).total_us, 31250, 1e-6);
  p.activities[1].dependencies = {"first"};
  EXPECT_NEAR(take(model.estimate_schedule(v, p, ctx)).total_us, 62500, 1e-6);
  auto slow = v;
  slow.graph.resources[0].read = 16;
  slow.graph.resources[0].write = 16;
  p.activities[1].dependencies.clear();
  EXPECT_NEAR(take(model.estimate_schedule(slow, p, ctx)).total_us, 250000, 1e-6);
}
TEST(Calibration, CopyWriteCostAppliesInBothDirectQueryAndCommandPlanExactlyOnce) {
  auto g = hardware();
  g.resources[0].read = 100;
  g.resources[0].write = 50;
  for (auto& e : g.access)
    e.bandwidth.reset();
  ResourceModel base(g);
  auto v = view(base);
  auto c = parse(overlay("'resource:controller-rule'='p'", "[profiles.p]\nwrite_cost={copy=2}"), g);
  CalibratedCostModel model(base, c);
  auto q = transfer();
  q.target = q.source;
  CostContext ctx{v.epoch, {}, {}, false};
  EXPECT_NEAR(take(model.estimate(v, q, ctx)).total_us, 50000, 1e-6);
  MemoryCommand cmd{"copy",
                    MemoryOpcode::copy,
                    ByteRangeRef{MemoryRefKind::external, "src", 0, GiB},
                    ByteRangeRef{MemoryRefKind::external, "dst", 0, GiB},
                    0,
                    "move",
                    "a>a",
                    "opaque/requester",
                    {},
                    {}};
  OperandEnvironment env{
    {},
    {{MemoryRefKind::external, "src", "host", GiB, 0, {{0, GiB, "root1", 0, {AccessMode::read}, "a"}}},
     {MemoryRefKind::external, "dst", "host", GiB, 0, {{0, GiB, "root2", 0, {AccessMode::write}, "a"}}}}};
  const auto plan = take(model.cost_plan(v, {{cmd}}, env));
  ASSERT_EQ(plan.activities[0].demands.size(), 2u);
  EXPECT_EQ(plan.activities[0].demands[1].amount, 2);
  EXPECT_NEAR(take(model.estimate_schedule(v, plan, ctx)).total_us, 50000, 1e-6);
  auto write = access();
  write.access = AccessMode::write;
  EXPECT_NEAR(take(model.estimate_access(v, write, ctx)).total_us, 20000, 1e-6);
}
TEST(Calibration, MissingPhysicalFactsRemainIncompleteAndBadActivityIsNotHidden) {
  auto g = hardware();
  g.resources[0].read.reset();
  ResourceModel base(g);
  auto v = view(base);
  auto c = parse(overlay("'transfer:move'='p'", "[profiles.p]\nsize=['1 GiB']\nbandwidth=[32]"), g);
  CalibratedCostModel model(base, c);
  const auto cost = take(model.estimate(v, transfer(), {v.epoch, {}, {}, false}));
  EXPECT_EQ(cost.quality, CostQuality::incomplete);
  EXPECT_FALSE(cost.missing_facts.empty());
  CostPlan bad{{{"bad", "move", "missing-route", GiB, {}, {}, 0, "opaque/requester"}}};
  EXPECT_FALSE(model.estimate_schedule(v, bad, {v.epoch, {}, {}, false}));
}
TEST(Calibration, StrictSchemaRejectsUnsupportedAmbiguousAndMalformedInputs) {
  const auto g = hardware();
  const auto good = overlay("'access:any-rule'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[40]}");
  for (const auto& tail : {std::string("\n[methods]\nread={profile='p'}"), std::string("\nunknown=1")})
    EXPECT_FALSE(parse_calibration(good + tail, g, {}));
  for (const auto& p : {"read={cpu_cores=[16,16],bandwidth=[40,50]}", "read={cpu_cores=[16],bandwidth=[nan]}",
                        "read={cpu_cores=[true],bandwidth=[40]}", "read={size=['1.1 B'],bandwidth=[40]}",
                        "read={size=['1 GiB'],cpu_cores=[16],bandwidth=[40]}", "triad={cpu_cores=[16],bandwidth=[40]}",
                        "read={cpu_cores=[16],bandwidth=[40]}\nprovenance={quality='calibrated'}"})
    EXPECT_FALSE(parse_calibration(overlay("'access:any-rule'='p'", std::string("[profiles.p]\n") + p), g, {})) << p;
  EXPECT_FALSE(
    parse_calibration(overlay("'access:missing'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[40]}"), g, {}));
  EXPECT_FALSE(parse_calibration(good + "\n[profiles.unused]\nsize=['1 GiB']\nbandwidth=[1]", g, {}));
}
const std::string topology = R"(
schema='memory-model/0.3'
name='arbitrary names'
[units]
bandwidth='GiB/s'
[groups]
uma={count='runtime'}
cowos_superdie={per='uma',count=2}
[agents]
requester={per='cowos_superdie',bind='custom-cpu',capabilities=['host_load_store']}
engine={per='cowos_superdie',bind='custom-device',capabilities=['dma']}
[pools]
bank={per='requester',bind='custom-bank',capacity='1 GiB',service='controller',page_sizes=['4 KiB'],granularity='4 KiB'}
[resources]
controller={per='requester',read=1000,write=1000,sharing='read_write'}
infinity_fabric={between='cowos_superdie',within='uma',topology='complete',bandwidth=200}
[paths]
access_rule={agent='requester',pool='bank',where='any',within='uma',bandwidth={read=80,write=80},mapping='direct',use_fabric=true}
[transfers]
async_move={from='bank',to='bank',by='engine',controller='requester[0]',within='uma',where='any'}
)";
TEST(Calibration, SameTemporaryRelationsResolveOpaqueTopologyAndDeviceController) {
  const auto cal = overlay("'transfer:async_move'='p'", R"(
[profiles.p]
samples=[{method='async_move',executor='requester[0]',source='bank:local',target='bank:same(uma)&!local',operation='copy',cpu_cores=16,bandwidth=32}]
)");
  const auto compiled = take(expand_calibrated_hardware_template(topology, cal, {{"uma", 1}}));
  const auto& db = calibration_database(*compiled.calibration);
  ASSERT_FALSE(db.cases.empty());
  const auto& point = std::any_cast<const std::map<std::string, std::any>&>(
    std::any_cast<const std::vector<std::any>&>(db.profiles.at("p").at("samples")).front());
  EXPECT_FALSE(point.contains("source"));
  EXPECT_FALSE(point.contains("executor"));
  for (const auto& c : db.cases) {
    const auto& m = compiled.hardware.graph.methods[0];
    const auto r = std::find_if(m.routes.begin(), m.routes.end(), [&](const auto& x) { return x.id == c.route; });
    ASSERT_NE(r, m.routes.end());
    EXPECT_EQ(c.executor, r->controller);
    EXPECT_NE(c.executor, r->agent);
  }
  // Runtime bindings are separate and no ancestry survives in the database.
  auto graph = compiled.hardware.graph;
  for (auto& a : graph.agents)
    a.address_space = "host";
  for (auto& p : graph.pools)
    p.address_space = "host";
  ResourceModel base(graph);
  Scope scope;
  for (const auto& a : graph.agents)
    scope.executable_agents.push_back(a.id);
  for (const auto& p : graph.pools)
    scope.allocatable_pools.push_back(p.id);
  auto v = take(base.view(scope));
  const auto& entry = db.cases.front();
  const auto& route = *std::find_if(graph.methods[0].routes.begin(), graph.methods[0].routes.end(),
                                    [&](const auto& x) { return x.id == entry.route; });
  CalibratedCostModel model(base, compiled.calibration, {{*route.controller, {16, ""}}});
  TransferQuery q{
    {"host", {{0, GiB, route.source}}}, {"host", {{0, GiB, route.target}}}, "async_move", route.agent, GiB, {}};
  EXPECT_NEAR(take(model.estimate(v, q, {v.epoch, {}, {}, false})).total_us, 31250, 1e-6);
  auto broken = cal;
  const auto pos = broken.find("same(uma)");
  broken.replace(pos, 9, "same(unknown)");
  EXPECT_FALSE(expand_calibrated_hardware_template(topology, broken, {{"uma", 1}}));
  const auto two = take(expand_calibrated_hardware_template(topology, cal, {{"uma", 2}}));
  EXPECT_EQ(two.hardware.graph.agents.size(), 8u);
}
TEST(Calibration, RuntimePublishesReadOnlyDatabaseAndUsesFacadeWithoutAllocation) {
  auto g = hardware();
  auto c = parse(overlay("'access:any-rule'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[40]}"), g);
  LinuxDiscovery d;
  d.graph = g;
  d.scope = {{"opaque/requester"}, {"a", "b"}, {"a", "b"}, {{"opaque/requester", 37}}};
  for (const auto& p : g.pools) {
    d.pool_bindings.push_back({p.id, {}});
    d.pool_available_bytes[p.id] = *p.capacity_bytes;
  }
  LinuxRuntimeOptions options;
  options.calibration = c;
  options.calibration_controls = {{"opaque/requester", {16, ""}}};
  auto runtime = take(make_linux_runtime(std::move(d), std::move(options)));
  EXPECT_EQ(runtime->performance().bindings.size(), 2u);
  const auto v = take(runtime->view(take(runtime->current_scope())));
  EXPECT_NEAR(take(runtime->costs().estimate_access(v, access(), {v.epoch, {}, {}, false})).total_us, 25000, 1e-6);
  EXPECT_EQ(runtime->physical_graph().access[0].bandwidth->at("read"), 80);
}
TEST(Calibration, ActiveTransfersUseTheirOwnControlsAndRetainPhysicalCompetition) {
  ResourceModel base(hardware());
  auto v = view(base);
  auto c = parse(overlay("'transfer:move'='p'", "[profiles.p]\nsize=['1 GiB']\nbandwidth=[32]"), base.graph());
  CalibratedCostModel model(base, c);
  CostContext context{v.epoch,
                      {{"active",
                        "move",
                        "a>b",
                        GiB,
                        {},
                        {{"bank", DemandKind::read, 1, 0}, {"bank", DemandKind::write, 1, 0}},
                        0,
                        "opaque/requester"}},
                      {},
                      false};
  EXPECT_NEAR(take(model.estimate(v, transfer(), context)).total_us, 31250, 1e-6);
  auto slow = v;
  slow.graph.resources[0].read = 16;
  slow.graph.resources[0].write = 16;
  EXPECT_NEAR(take(model.estimate(slow, transfer(), context)).total_us, 250000, 1e-6);
}
TEST(Calibration, CompiledOverlayCannotAttachToDifferentlyConnectedSameNames) {
  auto g = hardware();
  auto c = parse(overlay("'access:any-rule'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[40]}"), g);
  g.access[0].pool = "b";
  ResourceModel other(g);
  EXPECT_THROW(CalibratedCostModel(other, c), std::invalid_argument);
}
std::string read_document_file(const std::string& relative) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
  std::ifstream file(root / "docs/memory-resource-model" / relative);
  if (!file)
    throw std::runtime_error("missing document input: " + relative);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
TEST(Calibration, PortableDocumentHardwareSupportsQualifiedSubsetAndRejectsBroaderHistoricalProfiles) {
  for (const auto name : {"a800", "h800"}) {
    const auto hardware_text = read_document_file(std::string("configs/") + name + ".toml");
    auto compiled = take(expand_calibrated_hardware_template(
      hardware_text, overlay("'transfer:p2p'='p'", "[profiles.p]\nsize=['64 KiB','128 MiB']\nbandwidth=[13.95,21.16]"),
      {{"node", 1}}));
    ASSERT_EQ(calibration_database(*compiled.calibration).bindings.size(), 1u);
    EXPECT_EQ(calibration_database(*compiled.calibration).bindings[0].target, "p2p");
    const auto historic = expand_calibrated_hardware_template(
      hardware_text, read_document_file(std::string("calibration/") + name + ".toml"), {{"node", 1}});
    ASSERT_FALSE(historic);
    EXPECT_EQ(std::get<Error>(historic.state).code, "calibration.unsupported");
  }
  const auto compiled = take(expand_calibrated_hardware_template(
    read_document_file("configs/920f-lx.toml"),
    overlay("'access:ddr_local'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[149.29]}"), {{"node", 1}}));
  EXPECT_EQ(calibration_database(*compiled.calibration).bindings.size(), 16u);
}
TEST(Calibration, ArbitraryRenamingAsymmetricTopologyAndDuplicateCasesAreChecked) {
  const auto cal = overlay("'transfer:async_move'='p'", R"(
[profiles.p]
samples=[{method='async_move',executor='requester[0]',source='bank:local',target='bank:same(uma)&!local',operation='copy',bandwidth=32}]
)");
  auto h = topology, c = cal;
  const auto replace = [](std::string& text, const std::string& from, const std::string& to) {
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
      text.replace(at, from.size(), to);
      at += to.size();
    }
  };
  for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{{"uma", "coherence_domain"},
                                                                                 {"cowos_superdie", "region"},
                                                                                 {"requester", "compute_agent"},
                                                                                 {"engine", "copy_agent"},
                                                                                 {"bank", "storage"},
                                                                                 {"async_move", "opaque_transfer"}}) {
    replace(h, from, to);
    replace(c, from, to);
  }
  replace(h, "topology='complete',bandwidth=200", "topology='explicit',edges=[[0,1],[1,0]],bandwidth=[200,100]");
  const auto renamed = take(expand_calibrated_hardware_template(h, c, {{"coherence_domain", 1}}));
  const auto original = take(expand_calibrated_hardware_template(topology, cal, {{"uma", 1}}));
  EXPECT_EQ(calibration_database(*renamed.calibration).cases.size(),
            calibration_database(*original.calibration).cases.size());
  auto duplicate = cal;
  const auto begin = duplicate.find("{method=");
  const auto end = duplicate.find('}', begin);
  duplicate.insert(end + 1, "," + duplicate.substr(begin, end - begin + 1));
  EXPECT_FALSE(expand_calibrated_hardware_template(topology, duplicate, {{"uma", 1}}));
}
TEST(Calibration, SizeAxesKeepUint64PrecisionAboveDoubleIntegerRange) {
  auto g = hardware();
  for (auto& p : g.pools)
    p.capacity_bytes = 1ULL << 60;
  for (auto& c : g.capacity_groups)
    c.capacity_bytes = 1ULL << 60;
  ResourceModel base(g);
  const auto v = view(base);
  auto c =
    parse(overlay("'access:any-rule'='p'", "[profiles.p]\nread={size=['9007199254740993 B'],bandwidth=[10]}"), g);
  CalibratedCostModel model(base, c);
  EXPECT_EQ(take(model.explain_access(v, access(9007199254740993ULL))).size(), 1u);
  EXPECT_TRUE(take(model.explain_access(v, access(9007199254740992ULL))).empty());
}
TEST(Calibration, RequesterExecutedMethodUsesActualExecutorRatherThanDeviceControllerControls) {
  auto h = topology;
  const auto at = h.find("by='engine'");
  ASSERT_NE(at, std::string::npos);
  h.replace(at, 11, "by='request.agent'");
  const auto cal = overlay("'transfer:async_move'='p'", R"(
[profiles.p]
samples=[{method='async_move',executor='requester[1]',source='bank:any',target='bank:any',operation='copy',cpu_cores=16,bandwidth=32}]
)");
  auto compiled = take(expand_calibrated_hardware_template(h, cal, {{"uma", 1}}));
  auto graph = compiled.hardware.graph;
  for (auto& a : graph.agents)
    a.address_space = "host";
  for (auto& p : graph.pools)
    p.address_space = "host";
  ResourceModel base(graph);
  Scope scope;
  for (const auto& a : graph.agents)
    scope.executable_agents.push_back(a.id);
  for (const auto& p : graph.pools)
    scope.allocatable_pools.push_back(p.id);
  const auto v = take(base.view(scope));
  const auto& entry = calibration_database(*compiled.calibration).cases.front();
  const auto& route = *std::find_if(graph.methods[0].routes.begin(), graph.methods[0].routes.end(),
                                    [&](const auto& x) { return x.id == entry.route; });
  ASSERT_NE(entry.executor, route.controller);
  CalibratedCostModel model(base, compiled.calibration, {{*entry.executor, {16, ""}}, {*route.controller, {37, ""}}});
  TransferQuery q{
    {"host", {{0, GiB, route.source}}}, {"host", {{0, GiB, route.target}}}, "async_move", entry.executor, GiB, {}};
  EXPECT_NEAR(take(model.estimate(v, q, {v.epoch, {}, {}, false})).total_us, 31250, 1e-6);
}
TEST(Calibration, OpenRuntimeLoadsOverlayBeforeDiscoveryAndDoesNotAllocatePayload) {
  const auto probe = take(probe_linux_topology());
  const auto groups =
    std::count_if(probe.nodes.begin(), probe.nodes.end(), [](const auto& n) { return !n.cpus.empty(); });
  ASSERT_GT(groups, 0);
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "build" /
                    ("calibration-open-" + std::to_string(::getpid()));
  std::filesystem::create_directories(root);
  struct Remove {
    std::filesystem::path path;
    ~Remove() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } remove{root};
  const auto text =
    "schema='memory-model/0.3'\nname='generic live discovery'\n[units]\nbandwidth='GiB/s'\n"
    "[groups]\numa={count='runtime'}\n[agents]\ncompute={per='uma',count=" +
    std::to_string(groups) +
    ",bind='cpu-group',capabilities=['host_load_store']}\n[pools]\nbank={per='compute',capacity='discover',bind='local-"
    "ddr',service='controller'}\n"
    "[resources]\ncontroller={per='compute',read=1000,write=1000,sharing='read_write'}\n"
    "[paths]\nread_path={agent='compute',pool='bank',where='any',bandwidth={read=80,write=80},mapping='direct'}\n";
  const auto calibration = overlay("'access:read_path'='p'", "[profiles.p]\nread={cpu_cores=[16],bandwidth=[40]}");
  {
    std::ofstream file(root / "hardware.toml");
    file << text;
  }
  {
    std::ofstream file(root / "calibration.toml");
    file << calibration;
  }
  LinuxRuntimeOptions options;
  const auto expanded = take(expand_hardware_template(text, {{"uma", 1}}));
  for (const auto& a : expanded.graph.agents)
    options.calibration_controls[a.id] = {16, ""};
  auto runtime = take(open_linux_runtime((root / "hardware.toml").string(), {}, (root / "calibration.toml").string(),
                                         std::move(options)));
  const auto v = take(runtime->view(take(runtime->current_scope())));
  ASSERT_FALSE(v.executable_agents.empty());
  ASSERT_FALSE(v.allocatable_pools.empty());
  AccessQuery query{
    v.executable_agents.front(), {"host", {{0, Q, v.allocatable_pools.front()}}}, AccessMode::read, Q, {}};
  // The discovered address-space string is intentionally opaque.
  query.memory.address_space = *v.graph.pools.front().address_space;
  EXPECT_NEAR(take(runtime->costs().estimate_access(v, query, {v.epoch, {}, {}, false})).total_us,
              static_cast<double>(Q) * 1e6 / (40 * GiB), 1e-9);
  EXPECT_FALSE(runtime->performance().bindings.empty());
  {
    std::ofstream file(root / "calibration.toml");
    file << calibration << "\n[methods]\nread={profile='p'}\n";
  }
  const auto invalid =
    open_linux_runtime((root / "hardware.toml").string(), {}, (root / "calibration.toml").string(), {});
  ASSERT_FALSE(invalid);
  EXPECT_EQ(std::get<Error>(invalid.state).code, "calibration.unsupported");
}
} // namespace
} // namespace vecops::memory::model
