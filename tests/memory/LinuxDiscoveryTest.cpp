// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxDiscovery.h"
#include "../../src/memory/details/LinuxFreeCapacity.h"
#include "vecops/memory/ResourceModel.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace vecops::memory::model {
namespace {
constexpr std::uint64_t GiB = 1ULL << 30;
LinuxTopologySnapshot generic_topology() {
  return {{{10, {100, 101}, 32 * GiB, 16 * GiB}, {2, {20, 21}, 16 * GiB, 8 * GiB}, {99, {}, 4 * GiB, 3 * GiB}},
          {100},
          {2, 99},
          4096,
          "synthetic CPU",
          {}};
}
LinuxTopologySnapshot lx2() {
  LinuxTopologySnapshot s{};
  s.base_page_bytes = 4096;
  s.raw_cpu_model = "CPU implementer=0x48; CPU part=raw-test";
  for (unsigned i = 0; i < 16; ++i) {
    std::vector<unsigned> cpus;
    for (unsigned j = 0; j < 38; ++j)
      cpus.push_back(38 * i + j);
    s.nodes.push_back({i, cpus, 32 * GiB, 16 * GiB});
    s.nodes.push_back({16 + i, {}, 4 * GiB, 3 * GiB});
    s.allowed_memory_nodes.push_back(i);
    s.allowed_memory_nodes.push_back(16 + i);
    if (i < 7)
      for (unsigned j = 0; j < 37; ++j)
        s.allowed_cpus.push_back(38 * i + j);
  }
  s.dma_devices = {{0, "/dev/sdma0", false}, {2, "/dev/sdma2", true}};
  std::reverse(s.nodes.begin(), s.nodes.end()); // Probe order has no placement semantics.
  return s;
}
HardwareExpansion lx2_template(std::filesystem::path source = {}) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
  if (source.empty()) source = root / "docs/memory-resource-model/configs/920f-lx.toml";
  std::ifstream file(source);
  if (!file.good()) throw std::runtime_error("Missing LX2 test template: " + source.string());
  std::ostringstream text;
  text << file.rdbuf();
  auto result = expand_hardware_template(text.str(), {{"node", 1}});
  if (!result) throw std::runtime_error("Invalid LX2 test template: " + std::get<Error>(result.state).message);
  return std::move(result).value();
}
bool contains(const std::vector<std::string>& values, const std::string& id) {
  return std::find(values.begin(), values.end(), id) != values.end();
}
const BackendDescriptor* descriptor(const LinuxDiscovery& d, const std::string& id) {
  const auto it =
    std::find_if(d.descriptors.begin(), d.descriptors.end(), [&](const auto& row) { return row.object == id; });
  EXPECT_NE(it, d.descriptors.end());
  return it == d.descriptors.end() ? nullptr : &*it;
}
std::string zone_text(unsigned node, const std::string& name, std::uint64_t free,
                      std::uint64_t high, std::string protection = "(0, 0)",
                      const std::vector<std::uint64_t>& thresholds = {0}) {
  std::ostringstream out;
  out << "Node " << node << ", zone " << name << "\n  pages free " << free
      << "\n high " << high << "\n protection: " << protection << "\n";
  if (!thresholds.empty()) {
    out << " pagesets\n";
    for (std::size_t cpu = 0; cpu < thresholds.size(); ++cpu)
      out << " cpu: " << cpu << "\n high: 999999999\n vm stats threshold: " << thresholds[cpu] << "\n";
  }
  return out.str();
}
TEST(LinuxDiscovery, ZoneFreePagesCombineOpaqueZonesAndNodesWithoutDoubleSubtractingBoost) {
  const auto first = zone_text(9, "opaque-bank:@", 100, 10, "(0, 5, 2)", {2, 3});
  const auto second = zone_text(9, "arbitrary-coWoS", 50, 8, "(7)", {1});
  const auto empty = zone_text(9, "unpopulated", 0, 32, "(0, 0)", {});
  const auto remote = zone_text(3, "custom", 20, 25, "(0)", {0});
  const auto result = details::linux_zone_free_pages(first + " boost 123456\n nr_free_pages 9999\n" + second + empty + remote);
  EXPECT_EQ(result, (std::map<unsigned, std::uint64_t>{{3, 0}, {9, 114}}));
}
TEST(LinuxDiscovery, ZoneFreePagesSubtractBothHighatomicSpellingsAndEveryCpuThreshold) {
  for (const auto* spelling : {"reserved_highatomic", "nr_reserved_highatomic"}) {
    auto text = zone_text(2, "memory", 100, 10, "(3, 7, 2)", {2, 3});
    text.insert(text.find(" pagesets"), std::string(" ") + spelling + " 8\n");
    EXPECT_EQ(details::linux_zone_free_pages(text).at(2), 70u);
  }
  // A reduced excerpt of the sampled format: the CPU pageset high: value is
  // unrelated to the zone high watermark. Counter drift is one threshold/CPU.
  auto text = zone_text(16, "sampled-zone", 1031818, 2773, "(0, 0, 0, 0)", std::vector<std::uint64_t>(608, 125));
  EXPECT_EQ(details::linux_zone_free_pages(text).at(16), 953045u);
}
TEST(LinuxDiscovery, ZoneFreePagesAllowConsistentSingleCpuUpFormWithoutVmThreshold) {
  auto text = zone_text(0, "zone-a", 100, 10, "(0, 5)", {});
  text += " pagesets\n cpu: 0\n high: 400\n batch: 7\n";
  auto other = zone_text(1, "zone-b", 200, 20, "(0)", {});
  other += " pagesets\n cpu: 0\n high: 900\n";
  EXPECT_EQ(details::linux_zone_free_pages(text + other),
            (std::map<unsigned, std::uint64_t>{{0, 85}, {1, 180}}));
  // One zone with normal SMP counters proves an omitted cpu0 threshold in a
  // different zone is not an unambiguous UP snapshot.
  EXPECT_THROW(details::linux_zone_free_pages(text + zone_text(1, "zone-c", 100, 5)), std::invalid_argument);
  EXPECT_THROW(details::linux_zone_free_pages(text + " cpu: 1\n high: 5\n"), std::invalid_argument);
}
TEST(LinuxDiscovery, ZoneFreePagesRejectDuplicateMissingAndMalformedFields) {
  const auto good = zone_text(0, "memory", 100, 10, "(0, 5)", {3});
  auto erase_line = [&](const std::string& needle) {
    auto value = good;
    const auto begin = value.find(needle);
    value.erase(begin, value.find('\n', begin) - begin + 1);
    return value;
  };
  const std::vector<std::string> bad = {
      "", "unknown before header\n" + good,
      "Node 0 zone memory\n", "Node -1, zone memory\n", "Node 0, zone memory extra\n",
      good + good, erase_line("  pages free"), erase_line(" high 10"), erase_line(" protection:"),
      good + " vm stats threshold: 3\n", good + " cpu: 0\n vm stats threshold: 3\n",
      good + " cpu: 1\n high: 9\n", good + " cpu: 1\n high: 9\n vm stats threshold 5\n",
      "Node 0, zone memory\n pages free 100\n high 10\n protection: (0)\n",
      "Node 0, zone empty\n pages free 0\n high 0\n protection: (0)\n pagesets\n",
      good + " high 7\n", good + " pagesets\n",
      "Node 0, zone memory\n pages free -1\n high 0\n protection: (0)\n",
      "Node 0, zone memory\n pages free 1 extra\n high 0\n protection: (0)\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: ()\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (1,)\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (1 2)\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (0) trailing\n",
      "Node 0, zone memory\n pages free 0\n high 0\n high 0\n protection: (0)\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (0)\n reserved_highatomic 1\n nr_reserved_highatomic 1\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (0)\n vm stats threshold: 1\n",
      "Node 0, zone memory\n pages free 0\n high 0\n protection: (0)\n cpu: 0\n",
  };
  for (const auto& text : bad)
    EXPECT_THROW(details::linux_zone_free_pages(text), std::invalid_argument) << text;
  // Missing one threshold in a multi-CPU zone cannot silently infer zero drift.
  auto partial = good;
  const auto threshold = partial.find(" vm stats threshold: 3\n");
  partial.erase(threshold, std::string(" vm stats threshold: 3\n").size());
  partial += " cpu: 1\n vm stats threshold: 4\n";
  EXPECT_THROW(details::linux_zone_free_pages(partial), std::invalid_argument);
}
TEST(LinuxDiscovery, ZoneFreePagesCheckIntegerReservationAndNodeSumOverflow) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(details::linux_zone_free_pages(zone_text(0, "a", maximum, 0)).at(0), maximum);
  EXPECT_THROW(details::linux_zone_free_pages(zone_text(0, "a", maximum, 0) + zone_text(0, "b", 1, 0)), std::overflow_error);
  EXPECT_THROW(details::linux_zone_free_pages(zone_text(0, "a", 0, maximum, "(1)", {})), std::overflow_error);
  EXPECT_THROW(details::linux_zone_free_pages(zone_text(0, "a", 0, 0, "(0)", {maximum, 1})), std::overflow_error);
  EXPECT_THROW(details::linux_zone_free_pages("Node 0, zone a\n pages free 18446744073709551616\n"), std::overflow_error);
  EXPECT_THROW(details::linux_zone_free_pages("Node 4294967296, zone a\n"), std::overflow_error);
}
TEST(LinuxDiscovery, ZoneFreePagesReadOnlyLiveLocalSnapshot) {
  std::ifstream file("/proc/zoneinfo");
  ASSERT_TRUE(file.good());
  std::ostringstream buffer; buffer << file.rdbuf();
  const auto result = details::linux_zone_free_pages(buffer.str());
  EXPECT_FALSE(result.empty());
  for (const auto& [node, pages] : result) {
    (void)node;
    EXPECT_LE(pages, std::numeric_limits<std::uint64_t>::max() / 4096);
  }
}
TEST(LinuxDiscovery, LinuxIdRangesAreStrictExactSets) {
  auto value = parse_linux_id_list("0-2,5,9-10\n");
  ASSERT_TRUE(value);
  EXPECT_EQ(value.value(), (std::vector<unsigned>{0, 1, 2, 5, 9, 10}));
  ASSERT_TRUE(parse_linux_id_list(""));
  EXPECT_TRUE(parse_linux_id_list("").value().empty());
  for (const auto* bad : {"1,", "-1", "2-1", "1-2-3", "1.0", "1,,2", "1,1", "1-3,3-4", "4294967296", "0-1000000"})
    EXPECT_FALSE(parse_linux_id_list(bad)) << bad;
}
TEST(LinuxDiscovery, PhysicalOrderAndMemoryGrantsAreIndependentOfCpuAffinity) {
  auto result = discover_linux_cpu(generic_topology());
  ASSERT_TRUE(result);
  const auto& d = result.value();
  ASSERT_EQ(d.graph.agents.size(), 2u);
  ASSERT_EQ(d.graph.pools.size(), 3u);
  EXPECT_EQ(descriptor(d, "linux/cpu[0]")->labels.at("os_numa"), "2");
  EXPECT_EQ(descriptor(d, "linux/cpu[1]")->labels.at("os_numa"), "10");
  EXPECT_EQ(d.scope.executable_agents, (std::vector<std::string>{"linux/cpu[1]"}));
  EXPECT_EQ(d.scope.allocatable_pools, (std::vector<std::string>{"linux/pool[0]", "linux/pool[2]"}));
  EXPECT_EQ(d.scope.accessible_pools.size(), 3u);
  EXPECT_EQ(d.pool_bindings[0].numa_nodes, (std::vector<unsigned>{2}));
  EXPECT_EQ(d.pool_available_bytes.at("linux/pool[0]"), 8 * GiB);
  EXPECT_EQ(d.graph.access.size(), 6u);
  EXPECT_EQ(d.graph.methods[0].routes.size(), 9u);
  for (const auto& r : d.graph.resources) {
    EXPECT_FALSE(r.bandwidth);
    EXPECT_FALSE(r.read);
    EXPECT_FALSE(r.write);
  }
  for (const auto& a : d.graph.access) {
    EXPECT_FALSE(a.latency_ns);
    EXPECT_FALSE(a.bandwidth);
    EXPECT_EQ(a.mapping, MappingMode::direct);
  }
  ResourceModel model(d.graph);
  auto view = model.view(d.scope);
  ASSERT_TRUE(view);
  EXPECT_TRUE(contains(view.value().allocatable_pools, "linux/pool[0]"));
  EXPECT_FALSE(contains(view.value().executable_agents, "linux/cpu[0]"));
  EXPECT_EQ(descriptor(d, "linux/cpu[1]")->labels.at("raw_cpu_model"), "synthetic CPU");
}
TEST(LinuxDiscovery, MissingTemplateFixtureStopsBeforeUsingSyntheticTopology) {
  EXPECT_THROW(lx2_template(std::filesystem::path(__FILE__) / "not-a-template.toml"), std::runtime_error);
}
TEST(LinuxDiscovery, Lx2FlatWholePhysicalGraphAndPartialCpuScope) {
  const auto hardware = lx2_template();
  auto s = lx2();
  auto result = discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat);
  ASSERT_TRUE(result) << std::get<Error>(result.state).message;
  const auto& d = result.value();
  EXPECT_EQ(d.graph.agents.size(), 20u);
  EXPECT_EQ(d.graph.pools.size(), 32u);
  EXPECT_EQ(d.scope.executable_agents.size(), 8u); // 7 CPU groups and the one permitted SDMA role.
  EXPECT_EQ(d.scope.allocatable_pools.size(), 32u);
  EXPECT_EQ(d.pool_bindings.size(), 32u);
  const std::string cpu7 = "node[0]/socket[0]/die[1]/cpu[3]", ddr7 = cpu7 + "/ddr[0]", hbm7 = cpu7 + "/hbm[0]";
  EXPECT_FALSE(contains(d.scope.executable_agents, cpu7));
  EXPECT_TRUE(contains(d.scope.allocatable_pools, ddr7));
  EXPECT_TRUE(contains(d.scope.allocatable_pools, hbm7));
  EXPECT_EQ(descriptor(d, cpu7)->labels.at("os_numa"), "7");
  EXPECT_EQ(descriptor(d, hbm7)->labels.at("os_numa"), "23");
  EXPECT_FALSE(contains(d.scope.executable_agents, "node[0]/dma[0]"));
  EXPECT_FALSE(contains(d.scope.executable_agents, "node[0]/dma[1]"));
  EXPECT_TRUE(contains(d.scope.executable_agents, "node[0]/dma[2]"));
  EXPECT_FALSE(contains(d.scope.executable_agents, "node[0]/dma[3]"));
  EXPECT_TRUE(d.scope.execution_slots.empty());
  for (const auto& descriptor : d.descriptors)
    EXPECT_EQ(descriptor.labels.at("layout_profile"), "lx2_flat");
  EXPECT_EQ(descriptor(d, cpu7)->labels.at("raw_cpu_model"), s.raw_cpu_model);
  EXPECT_EQ(d.inventory.bindings.size(), hardware.binding_requests.size());
  EXPECT_EQ(d.inventory.facts.at("cpu_model"), HardwareFact{std::string("LX2")});
  EXPECT_EQ(d.pool_available_bytes.at(hbm7), 3 * GiB);
  ResourceModel model(d.graph);
  auto view = model.view(d.scope);
  ASSERT_TRUE(view);
  EXPECT_EQ(view.value().allocatable_pools.size(), 32u);
  // Restrict allocation policy independently: host-addressable imports stay visible.
  s.allowed_memory_nodes = {0, 1, 2, 3, 16, 17, 18, 19};
  result = discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().scope.allocatable_pools.size(), 8u);
  EXPECT_EQ(result.value().scope.accessible_pools.size(), 32u);
}
TEST(LinuxDiscovery, CpuGroupsUseLowestCpuRatherThanNumaOrLexicalId) {
  auto s = lx2();
  const auto hardware = lx2_template();
  auto n2 = std::find_if(s.nodes.begin(), s.nodes.end(), [](const auto& n) { return n.os_numa == 2; });
  auto n10 = std::find_if(s.nodes.begin(), s.nodes.end(), [](const auto& n) { return n.os_numa == 10; });
  std::swap(n2->cpus, n10->cpus);
  auto result = discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat);
  ASSERT_TRUE(result);
  const auto& d = result.value();
  const std::string logical2 = "node[0]/socket[0]/die[0]/cpu[2]";
  EXPECT_EQ(descriptor(d, logical2)->labels.at("os_numa"), "10");
  EXPECT_EQ(descriptor(d, logical2 + "/hbm[0]")->labels.at("os_numa"), "26");
}
TEST(LinuxDiscovery, Lx2CacheModeMissingHbmAndRoleMismatchAreErrors) {
  const auto hardware = lx2_template();
  auto s = lx2();
  std::erase_if(s.nodes, [](const auto& n) { return n.os_numa >= 16; });
  std::erase_if(s.allowed_memory_nodes, [](unsigned id) { return id >= 16; });
  EXPECT_FALSE(discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat));
  s = lx2();
  s.nodes[0].capacity_bytes = 3 * GiB;
  EXPECT_FALSE(discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat));
  s = lx2();
  s.nodes[0].cpus = {9999};
  EXPECT_FALSE(discover_linux_cpu(s, &hardware, LinuxCpuProfile::lx2_flat));
  EXPECT_FALSE(discover_linux_cpu(lx2(), &hardware));
  auto bad = hardware;
  bad.binding_requests.begin()->second = "unknown";
  EXPECT_FALSE(discover_linux_cpu(lx2(), &bad, LinuxCpuProfile::lx2_flat));
  bad = hardware;
  bad.graph.agents.pop_back();
  EXPECT_FALSE(discover_linux_cpu(lx2(), &bad, LinuxCpuProfile::lx2_flat));
  EXPECT_FALSE(discover_linux_cpu(lx2(), nullptr, LinuxCpuProfile::lx2_flat));
}
TEST(LinuxDiscovery, SyntheticInvalidProbeDataRejected) {
  auto s = generic_topology();
  s.nodes.push_back(s.nodes[0]);
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.nodes[0].cpus.push_back(20);
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.allowed_cpus = {9999};
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.allowed_memory_nodes = {99, 999};
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.nodes[0].available_bytes = 64 * GiB;
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.base_page_bytes = 3;
  EXPECT_FALSE(discover_linux_cpu(s));
  s = generic_topology();
  s.dma_devices = {{0, "x", false}, {0, "y", true}};
  EXPECT_FALSE(discover_linux_cpu(s));
}
TEST(LinuxDiscovery, ReadOnlyLiveLocalProbe) {
  auto snapshot = probe_linux_topology();
  ASSERT_TRUE(snapshot) << std::get<Error>(snapshot.state).message;
  EXPECT_GT(snapshot.value().base_page_bytes, 0u);
  EXPECT_FALSE(snapshot.value().allowed_cpus.empty());
  EXPECT_FALSE(snapshot.value().allowed_memory_nodes.empty());
  auto discovered = discover_linux_cpu(snapshot.value());
  ASSERT_TRUE(discovered);
  EXPECT_FALSE(discovered.value().scope.executable_agents.empty());
  EXPECT_FALSE(discovered.value().scope.allocatable_pools.empty());
  EXPECT_NO_THROW(ResourceModel model(discovered.value().graph));
}
} // namespace
} // namespace vecops::memory::model
