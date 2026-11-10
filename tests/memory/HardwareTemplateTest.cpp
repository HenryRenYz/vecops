// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/HardwareTemplate.h"
#include "vecops/memory/ResourceModel.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <set>

namespace vecops::memory::model {
namespace {
constexpr std::uint64_t GiB = 1ULL << 30;
const std::string custom = R"toml(
schema = 'memory-model/0.3'
name = 'arbitrary sparse asymmetric fabric'
[units]
bandwidth = 'GiB/s'
[require]
custom_fact = true
[groups]
cowos_superdie = {per='uma', count=3}
uma = {count='runtime'}
[agents]
compute = {per='cowos_superdie', bind='custom-executor', capabilities=['execute']}
[pools]
memory = {per='cowos_superdie', capacity='discover', bind='custom-pool', service='bank', granularity='4 KiB', page_sizes=['4 KiB']}
[domains]
shared_bank = {per='uma', members='memory'}
[resources]
bank = {per='cowos_superdie', bandwidth=200}
infinity_fabric = {between='cowos_superdie', within='uma', topology='explicit', edges=[[0,1],[1,0],[1,2],[2,1]], bandwidth=[100,90,80,70]}
queue = {per='compute', kind='execution_slots', slots=2}
[paths]
access = {agent='compute', pool='memory', where='any', within='uma', use_fabric=true, mapping='requires_mapping', prerequisites=['map']}
[transfers]
move = {from='memory', to='memory', by='source.compute', within='uma', traffic={read=2,write=0.5}, use=['executor.queue'], startup_us=1.5}
)toml";
std::string read_config(const std::string& name) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
  std::ifstream file(root / "docs/memory-resource-model/configs" / (name + ".toml"));
  EXPECT_TRUE(file.good());
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}
std::string replace_all(std::string source, std::string_view before, std::string_view after) {
  std::size_t at = 0;
  bool found = false;
  while ((at = source.find(before, at)) != std::string::npos) {
    source.replace(at, before.size(), after);
    at += after.size();
    found = true;
  }
  EXPECT_TRUE(found) << before;
  return source;
}
HardwareInventory inventory(const HardwareExpansion& value) {
  HardwareInventory i;
  i.facts = value.requirements;
  for (const auto& a : value.graph.agents) {
    HardwareBinding b;
    b.address_space = "process";
    b.capabilities = a.capabilities;
    i.bindings.emplace(a.id, std::move(b));
  }
  for (const auto& p : value.graph.pools) {
    HardwareBinding b;
    b.capacity_bytes = 8 * GiB;
    b.address_space = "process";
    b.granularity_bytes = 4096;
    b.page_sizes_bytes = {4096};
    i.bindings.emplace(p.id, std::move(b));
  }
  return i;
}
const Resource* find_resource(const ResourceGraph& graph, const std::string& id) {
  const auto it =
    std::find_if(graph.resources.begin(), graph.resources.end(), [&](const auto& r) { return r.id == id; });
  EXPECT_NE(it, graph.resources.end());
  return it == graph.resources.end() ? nullptr : &*it;
}
const AccessConnection* access(const ResourceGraph& graph, std::string_view agent, std::string_view pool) {
  const auto it = std::find_if(graph.access.begin(), graph.access.end(),
                               [&](const auto& a) { return a.agent == agent && a.pool == pool; });
  EXPECT_NE(it, graph.access.end());
  return it == graph.access.end() ? nullptr : &*it;
}
const Method* method(const ResourceGraph& graph, std::string_view name) {
  const auto it = std::find_if(graph.methods.begin(), graph.methods.end(), [&](const auto& m) { return m.id == name; });
  EXPECT_NE(it, graph.methods.end());
  return it == graph.methods.end() ? nullptr : &*it;
}
TEST(HardwareTemplate, ThreeDocumentTemplatesExpandAndBindToResourceModel) {
  struct Expected {
    std::string name;
    std::size_t agents, pools, resources, access, domains;
  };
  for (const auto& expected :
       {Expected{"920f-lx", 20, 32, 28, 384, 4}, Expected{"a800", 2, 3, 8, 2, 0}, Expected{"h800", 8, 9, 42, 8, 0}}) {
    SCOPED_TRACE(expected.name);
    auto expanded = expand_hardware_template(read_config(expected.name), {{"node", 1}});
    ASSERT_TRUE(expanded) << std::get<Error>(expanded.state).message;
    const auto& g = expanded.value().graph;
    EXPECT_EQ(g.agents.size(), expected.agents);
    EXPECT_EQ(g.pools.size(), expected.pools);
    EXPECT_EQ(g.resources.size(), expected.resources);
    EXPECT_EQ(g.access.size(), expected.access);
    EXPECT_EQ(g.domains.size(), expected.domains);
    EXPECT_EQ(expanded.value().binding_requests.size(), expected.agents + expected.pools);
    EXPECT_FALSE(g.pools.back().address_space);
    EXPECT_TRUE(g.pools.back().page_sizes_bytes.empty());
    auto bound = bind_hardware_inventory(expanded.value(), inventory(expanded.value()));
    ASSERT_TRUE(bound) << std::get<Error>(bound.state).message;
    EXPECT_NO_THROW(ResourceModel model(bound.value()));
    if (expected.name == "920f-lx") {
      ASSERT_NE(method(g, "cpu_copy"), nullptr);
      EXPECT_EQ(method(g, "cpu_copy")->routes.size(), 1024u);
      EXPECT_EQ(method(g, "sdma_0")->routes.size(), 1024u);
      EXPECT_TRUE(method(g, "network")->routes.empty());
      EXPECT_EQ(bound.value().pools[0].capacity_bytes, 8 * GiB); // DDR discovered; HBM explicit 4 GiB.
      for (const auto& p : bound.value().pools) {
        if (p.id.ends_with("hbm[0]")) {
          EXPECT_EQ(p.capacity_bytes, 4 * GiB);
        }
      }
      for (const auto& d : g.domains)
        EXPECT_EQ(d.members.size(), 4u);
    } else {
      EXPECT_EQ(method(g, "p2p")->routes.size(), expected.name == "a800" ? 2u : 56u);
      const auto* link = find_resource(g, "node[0]/gpu[0]/hbm");
      ASSERT_NE(link, nullptr);
      EXPECT_DOUBLE_EQ(*link->bandwidth, expected.name == "a800" ? 1074.18 : 2224.83);
    }
  }
}
TEST(HardwareTemplate, DirectionalFabricAndSharedDdrService) {
  auto result = expand_hardware_template(read_config("920f-lx"), {{"node", 1}});
  ASSERT_TRUE(result);
  const auto& g = result.value().graph;
  const std::string a = "node[0]/socket[0]/die[0]/cpu[0]", b = "node[0]/socket[0]/die[1]/cpu[0]/hbm[0]";
  const auto* edge = access(g, a, b);
  ASSERT_NE(edge, nullptr);
  EXPECT_EQ(edge->rule, "hbm_other_die");
  ASSERT_EQ(edge->read_fabric.size(), 1u);
  ASSERT_EQ(edge->write_fabric.size(), 1u);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->write_fabric[0])->bandwidth, 148.84);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->read_fabric[0])->bandwidth, 144.42);
  std::set<std::string> services;
  for (const auto& p : g.pools)
    if (p.id.starts_with("node[0]/socket[0]/die[0]/") && p.id.ends_with("ddr[0]"))
      services.insert(*p.read_service);
  ASSERT_EQ(services.size(), 1u);
  EXPECT_EQ(*services.begin(), "node[0]/socket[0]/die[0]/ddr");
  const auto* ddr = find_resource(g, *services.begin());
  ASSERT_NE(ddr, nullptr);
  EXPECT_EQ(ddr->sharing, SharingMode::read_write);
  EXPECT_DOUBLE_EQ(*ddr->read, 149.92);
  EXPECT_DOUBLE_EQ(*ddr->write, 141.37);
  const auto* sdma = method(g, "sdma_3");
  ASSERT_NE(sdma, nullptr);
  for (const auto& route : sdma->routes) {
    EXPECT_EQ(route.agent, "node[0]/dma[3]");
    EXPECT_EQ(route.controller, "node[0]/socket[1]/die[0]/cpu[0]");
  }
}
TEST(HardwareTemplate, MultipleRootsUsePerRootSelectorsAndExplicitNetworkResources) {
  auto expanded = expand_hardware_template(read_config("920f-lx"), {{"node", 2}});
  ASSERT_TRUE(expanded);
  const auto& g = expanded.value().graph;
  EXPECT_EQ(g.agents.size(), 40u);
  EXPECT_EQ(g.pools.size(), 64u);
  const auto* net = method(g, "network");
  ASSERT_NE(net, nullptr);
  EXPECT_EQ(net->routes.size(), 2048u);
  for (const auto& r : net->routes) {
    EXPECT_NE(r.source.substr(0, 7), r.target.substr(0, 7));
    ASSERT_EQ(r.demands.size(), 4u);
    EXPECT_TRUE(r.demands[2].resource.ends_with("/network_tx"));
    EXPECT_TRUE(r.demands[3].resource.ends_with("/network_rx"));
  }
  const auto* sdma = method(g, "sdma_0");
  ASSERT_NE(sdma, nullptr);
  EXPECT_EQ(sdma->routes.size(), 2048u);
  for (const auto& r : sdma->routes)
    EXPECT_EQ(r.controller->substr(0, 7), r.source.substr(0, 7));
}
TEST(HardwareTemplate, ArbitraryNamesSparseAsymmetryAndSlotDemands) {
  auto expanded = expand_hardware_template(custom, {{"uma", 1}});
  ASSERT_TRUE(expanded);
  const auto& g = expanded.value().graph;
  EXPECT_EQ(g.agents.size(), 3u);
  EXPECT_EQ(g.access.size(), 9u);
  const auto* edge = access(g, "uma[0]/cowos_superdie[0]/compute[0]", "uma[0]/cowos_superdie[2]/memory[0]");
  ASSERT_NE(edge, nullptr);
  ASSERT_EQ(edge->write_fabric.size(), 2u);
  ASSERT_EQ(edge->read_fabric.size(), 2u);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->write_fabric[0])->bandwidth, 100);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->write_fabric[1])->bandwidth, 80);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->read_fabric[0])->bandwidth, 70);
  EXPECT_DOUBLE_EQ(*find_resource(g, edge->read_fabric[1])->bandwidth, 90);
  EXPECT_EQ(edge->mapping, MappingMode::requires_mapping);
  const auto* move = method(g, "move");
  ASSERT_NE(move, nullptr);
  ASSERT_EQ(move->routes.size(), 9u);
  for (const auto& r : move->routes) {
    ASSERT_EQ(r.demands.size(), 3u);
    EXPECT_DOUBLE_EQ(r.demands[0].amount, 2);
    EXPECT_DOUBLE_EQ(r.demands[1].amount, 0.5);
    EXPECT_EQ(r.demands[2].service, DemandKind::occupancy);
    EXPECT_EQ(r.agent->substr(0, r.agent->find("/compute")), r.source.substr(0, r.source.find("/memory")));
  }
  auto renamed = replace_all(replace_all(custom, "cowos_superdie", "uncore"), "infinity_fabric", "any_link");
  auto renamed_result = expand_hardware_template(renamed, {{"uma", 1}});
  ASSERT_TRUE(renamed_result);
  EXPECT_EQ(renamed_result.value().graph.access.size(), g.access.size());
  EXPECT_NO_THROW(
    ResourceModel model(std::move(bind_hardware_inventory(expanded.value(), inventory(expanded.value()))).value()));
}
TEST(HardwareTemplate, ExactDecimalSizesAndOverflow) {
  for (const auto& [text, expected] : std::vector<std::pair<std::string, std::uint64_t>>{
         {"1.5 GiB", 3 * GiB / 2},
         {"0.125 KiB", 128},
         {"0.0009765625 KiB", 1},
         {"18446744073709551615 B", std::numeric_limits<std::uint64_t>::max()},
         {"1.000000000000000000000 B", 1}}) {
    auto size = parse_memory_size(text);
    ASSERT_TRUE(size) << text;
    EXPECT_EQ(size.value(), expected);
  }
  for (const auto* bad : {"0 B", "1.1 B", "1 GB", "4096", "-1 KiB", "nan GiB", "18446744073709551616 B",
                          "18446744073709551615 GiB", "0.00000000000000000001 B"})
    EXPECT_FALSE(parse_memory_size(bad)) << bad;
}
TEST(HardwareTemplate, UnknownFieldsInvalidSelectorsAndAmbiguitiesRejected) {
  for (const auto& bad : std::vector<std::string>{
         replace_all(custom, "memory-model/0.3", "memory-model/0.2"),
         replace_all(custom, "bandwidth = 'GiB/s'", "bandwidth = 'GB/s'"),
         replace_all(custom, "count=3", "count=3, die_only=true"),
         replace_all(custom, "count=3", "count=3.0"),
         replace_all(custom, "per='uma', count=3", "per='unknown', count=3"),
         replace_all(custom, "per='uma', count=3", "per='compute', count=3"),
         replace_all(custom, "agent='compute'", "agent='compute[3]'"),
         replace_all(custom, "where='any'", "where='same(socket)'"),
         replace_all(custom, "mapping='requires_mapping'", "mapping='magic'"),
         replace_all(custom, "bandwidth=[100,90,80,70]", "bandwidth=[100,90]"),
         replace_all(custom, "edges=[[0,1],[1,0],[1,2],[2,1]]", "edges=[[0,1],[0,1],[1,2],[2,1]]"),
         replace_all(custom, "edges=[[0,1],[1,0],[1,2],[2,1]]", "edges=[[0,1],[1,0],[1,2],[2,9]]"),
         replace_all(custom, "edges=[[0,1],[1,0],[1,2],[2,1]]", "edges=[[0,1],[1,0],[1,2],[0,2]]"),
         replace_all(custom, "bandwidth=[100,90,80,70]", "bandwidth=[100,nan,80,70]"),
         replace_all(custom, "from='memory'", "from='compute'"),
         replace_all(custom, "by='source.compute'", "by='source.memory'"),
         replace_all(custom, "use=['executor.queue']", "use=['executor.unknown']"),
         replace_all(custom, "service='bank'", "service='unknown'"),
         replace_all(custom, "slots=2", "slots=2, bandwidth=10"),
         replace_all(custom, "where='any'", "where='any', bandwidth={read=50, extra=10}"),
         replace_all(custom, "prerequisites=['map']", "prerequisites=['map'], access=['read','read']"),
         replace_all(custom, "[paths]",
                     "[paths]\nduplicate = {agent='compute',pool='memory',where='any',within='uma'}")}) {
    auto result = expand_hardware_template(bad, {{"uma", 1}});
    EXPECT_FALSE(result) << bad;
  }
  EXPECT_FALSE(expand_hardware_template(custom));
  EXPECT_FALSE(expand_hardware_template(custom, {{"uma", 0}}));
  EXPECT_FALSE(expand_hardware_template(custom, {{"uma", 1}, {"socket", 2}}));
  EXPECT_FALSE(expand_hardware_template(custom, {{"uma", 1000001}}));
}
TEST(HardwareTemplate, DiscoveryFactsCoveragePagesCapabilitiesAndCapacityAliases) {
  auto expanded = expand_hardware_template(custom, {{"uma", 1}});
  ASSERT_TRUE(expanded);
  auto good = inventory(expanded.value());
  for (const auto& p : expanded.value().graph.pools) {
    good.bindings[p.id].capacity_group = "one-physical-capacity";
    good.bindings[p.id].capacity_group_bytes = 8 * GiB;
  }
  auto bound = bind_hardware_inventory(expanded.value(), good);
  ASSERT_TRUE(bound);
  ASSERT_EQ(bound.value().capacity_groups.size(), 1u);
  EXPECT_EQ(bound.value().capacity_groups[0].capacity_bytes, 8 * GiB);
  auto bad = good;
  bad.facts.clear();
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings.erase(bad.bindings.begin());
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings["unknown"] = {};
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  const auto pool = expanded.value().graph.pools[0].id, agent = expanded.value().graph.agents[0].id;
  bad = good;
  bad.bindings[pool].capacity_bytes = 0;
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings[pool].address_space.reset();
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings[pool].page_sizes_bytes = {8192};
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings[pool].capacity_group_bytes = 16 * GiB;
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.bindings[agent].capabilities = std::vector<std::string>{};
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  bad = good;
  bad.access["unknown"] = {};
  EXPECT_FALSE(bind_hardware_inventory(expanded.value(), bad));
  good.access[expanded.value().graph.access[0].id] = {MappingMode::direct, std::vector<std::string>{}};
  bound = bind_hardware_inventory(expanded.value(), good);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound.value().access[0].mapping, MappingMode::direct);
  auto compiled = compile_hardware_template(custom, {{"uma", 1}}, good);
  ASSERT_TRUE(compiled);
}
} // namespace
} // namespace vecops::memory::model
