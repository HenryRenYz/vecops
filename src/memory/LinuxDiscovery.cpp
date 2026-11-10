// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxDiscovery.h"
#include "details/LinuxFreeCapacity.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <sched.h>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
constexpr U GiB = 1ULL << 30;
void check(bool ok, const std::string& message) {
  if (!ok)
    throw std::invalid_argument(message);
}
template <class F>
auto checked(F&& fn) -> Result<decltype(fn())> {
  try {
    return {fn()};
  } catch (const std::invalid_argument& e) {
    return {Error{"linux.discovery", e.what()}};
  } catch (const std::runtime_error& e) {
    return {Error{"linux.probe", e.what()}};
  }
}
std::string trim(std::string_view text) {
  const auto begin = text.find_first_not_of(" \t\n\r");
  if (begin == std::string_view::npos)
    return {};
  return std::string(text.substr(begin, text.find_last_not_of(" \t\n\r") - begin + 1));
}
U decimal(std::string_view text) {
  U value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  check(!text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "invalid Linux integer");
  return value;
}
std::vector<unsigned> ids(std::string_view input) {
  std::set<unsigned> result;
  const auto text = trim(input);
  if (text.empty())
    return {};
  std::size_t begin = 0;
  while (begin < text.size()) {
    const auto comma = text.find(',', begin);
    const auto token =
      std::string_view(text).substr(begin, comma == std::string::npos ? text.size() - begin : comma - begin);
    const auto dash = token.find('-');
    const U low = decimal(token.substr(0, dash));
    const U high = dash == std::string_view::npos ? low : decimal(token.substr(dash + 1));
    check(low <= high && high <= std::numeric_limits<unsigned>::max() && high - low < 1000000,
          "invalid/oversized Linux ID range");
    for (U value = low; value <= high; ++value)
      check(result.insert(static_cast<unsigned>(value)).second, "duplicate/overlapping Linux ID range");
    check(result.size() <= 1000000, "Linux ID set exceeds 1000000 entries");
    if (comma == std::string::npos)
      break;
    begin = comma + 1;
    check(begin < text.size(), "trailing comma in Linux ID list");
  }
  return {result.begin(), result.end()};
}
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot read " + path.string());
  std::ostringstream content;
  content << file.rdbuf();
  if (file.bad())
    throw std::runtime_error("failed reading " + path.string());
  return content.str();
}
std::vector<unsigned> affinity() {
  for (std::size_t bytes = 128; bytes <= 131072; bytes *= 2) {
    std::vector<unsigned long> mask((bytes + sizeof(unsigned long) - 1) / sizeof(unsigned long), 0);
    if (::sched_getaffinity(0, mask.size() * sizeof(unsigned long), reinterpret_cast<cpu_set_t*>(mask.data())) == 0) {
      std::vector<unsigned> result;
      for (std::size_t word = 0; word < mask.size(); ++word)
        for (unsigned bit = 0; bit < sizeof(unsigned long) * 8; ++bit)
          if (mask[word] & (1UL << bit))
            result.push_back(static_cast<unsigned>(word * sizeof(unsigned long) * 8 + bit));
      return result;
    }
    if (errno != EINVAL)
      throw std::runtime_error("sched_getaffinity failed, errno=" + std::to_string(errno));
  }
  throw std::runtime_error("affinity mask exceeds supported CPU count");
}
std::map<std::string, std::string> colon_fields(std::string_view source) {
  std::istringstream lines{std::string(source)};
  std::map<std::string, std::string> result;
  std::string line;
  while (std::getline(lines, line)) {
    const auto at = line.find(':');
    if (at != std::string::npos)
      result.emplace(trim(std::string_view(line).substr(0, at)), trim(std::string_view(line).substr(at + 1)));
  }
  return result;
}
U node_memory(std::string_view content, std::string_view field) {
  std::istringstream lines{std::string(content)};
  std::string line;
  while (std::getline(lines, line)) {
    const auto at = line.find(std::string(field) + ":");
    if (at == std::string::npos)
      continue;
    std::istringstream value(line.substr(at + field.size() + 1));
    std::string count, unit, extra;
    value >> count >> unit;
    check(!count.empty() && unit == "kB" && !(value >> extra), "invalid node memory quantity");
    const U kib = decimal(count);
    check(kib <= std::numeric_limits<U>::max() / 1024, "node memory overflow");
    return kib * 1024;
  }
  throw std::runtime_error("missing node memory field: " + std::string(field));
}
bool intersects(const std::vector<unsigned>& a, const std::set<unsigned>& b) {
  return std::any_of(a.begin(), a.end(), [&](unsigned id) { return b.contains(id); });
}
std::string list(const std::vector<unsigned>& values) {
  std::string result;
  for (auto value : values) {
    if (!result.empty())
      result += ',';
    result += std::to_string(value);
  }
  return result;
}
struct Physical {
  std::vector<const LinuxNumaNode*> cpu, memory;
  std::map<unsigned, const LinuxNumaNode*> nodes;
  std::set<unsigned> allowed_cpus, allowed_memory;
  std::map<unsigned, const LinuxDmaDevice*> dma;
};
Physical validate(const LinuxTopologySnapshot& input) {
  check(input.base_page_bytes && !(input.base_page_bytes & (input.base_page_bytes - 1)), "invalid base page size");
  Physical p;
  std::set<unsigned> all_cpus;
  for (const auto& node : input.nodes) {
    check(p.nodes.emplace(node.os_numa, &node).second, "duplicate OS NUMA node");
    check(node.available_bytes <= node.capacity_bytes, "node available bytes exceed capacity");
    for (auto cpu : node.cpus)
      check(all_cpus.insert(cpu).second, "duplicate CPU within/across NUMA nodes");
    if (!node.cpus.empty())
      p.cpu.push_back(&node);
    if (node.capacity_bytes)
      p.memory.push_back(&node);
  }
  check(!p.cpu.empty() && !p.memory.empty(), "missing physical CPU or memory nodes");
  std::sort(p.cpu.begin(), p.cpu.end(), [](const auto* a, const auto* b) {
    return *std::min_element(a->cpus.begin(), a->cpus.end()) < *std::min_element(b->cpus.begin(), b->cpus.end());
  });
  std::sort(p.memory.begin(), p.memory.end(), [](const auto* a, const auto* b) { return a->os_numa < b->os_numa; });
  for (auto cpu : input.allowed_cpus)
    check(all_cpus.contains(cpu) && p.allowed_cpus.insert(cpu).second, "unknown/duplicate affinity CPU");
  for (auto node : input.allowed_memory_nodes)
    check(p.nodes.contains(node) && p.allowed_memory.insert(node).second, "unknown/duplicate memory grant");
  check(!p.allowed_cpus.empty() && !p.allowed_memory.empty(), "empty process CPU/memory grant");
  for (const auto& device : input.dma_devices) {
    check(!device.path.empty() && p.dma.emplace(device.role, &device).second, "duplicate/invalid DMA role");
  }
  return p;
}
void lx2_layout(const Physical& p) {
  check(p.cpu.size() == 16, "LX2 flat profile requires 16 CPU NUMA groups");
  std::set<unsigned> cpu_nodes;
  for (const auto* n : p.cpu) {
    check(n->os_numa < 16 && n->cpus.size() == 38 && n->capacity_bytes > 0, "LX2 CPU-group layout mismatch");
    cpu_nodes.insert(n->os_numa);
  }
  check(cpu_nodes.size() == 16 && p.nodes.size() == 32, "LX2 flat profile requires CPU0-15 and memory-only HBM16-31");
  for (unsigned i = 16; i < 32; ++i) {
    const auto n = p.nodes.find(i);
    check(n != p.nodes.end() && n->second->cpus.empty() && n->second->capacity_bytes == 4 * GiB,
          "LX2 flat HBM node is absent or not 4 GiB");
  }
}
std::vector<std::string> cpu_capabilities() {
  return {"execute", "copy", "host_load_store"};
}
BackendDescriptor cpu_descriptor(const std::string& id, const LinuxNumaNode& node,
                                 const LinuxTopologySnapshot& snapshot, const Physical& p) {
  std::vector<unsigned> active;
  for (auto cpu : node.cpus)
    if (p.allowed_cpus.contains(cpu))
      active.push_back(cpu);
  std::sort(active.begin(), active.end());
  auto cpus = node.cpus;
  std::sort(cpus.begin(), cpus.end());
  return {id,
          "linux_cpu",
          {{"os_numa", std::to_string(node.os_numa)},
           {"cpus", list(cpus)},
           {"allowed_cpus", list(active)},
           {"raw_cpu_model", snapshot.raw_cpu_model}}};
}
void bind_pool(LinuxDiscovery& output, const std::string& id, const LinuxNumaNode& node,
               const LinuxTopologySnapshot& snapshot, const Physical& p) {
  output.pool_bindings.push_back({id, {node.os_numa}, "host"});
  output.pool_available_bytes.emplace(id, node.available_bytes);
  output.scope.accessible_pools.push_back(id);
  const bool grant = p.allowed_memory.contains(node.os_numa);
  if (grant)
    output.scope.allocatable_pools.push_back(id);
  output.descriptors.push_back({id,
                                "linux_numa",
                                {{"os_numa", std::to_string(node.os_numa)},
                                 {"allocation_granted", grant ? "true" : "false"},
                                 {"available_source", "Linux zone headroom (no reclaim)"}}});
  HardwareBinding binding;
  binding.capacity_bytes = node.capacity_bytes;
  binding.address_space = "host";
  binding.granularity_bytes = snapshot.base_page_bytes;
  binding.page_sizes_bytes = {snapshot.base_page_bytes};
  output.inventory.bindings.emplace(id, std::move(binding));
}
LinuxDiscovery generic(const LinuxTopologySnapshot& snapshot, const Physical& p) {
  LinuxDiscovery result;
  for (std::size_t i = 0; i < p.cpu.size(); ++i) {
    const auto& n = *p.cpu[i];
    const auto id = "linux/cpu[" + std::to_string(i) + "]";
    result.graph.agents.push_back({id, cpu_capabilities(), "host"});
    result.descriptors.push_back(cpu_descriptor(id, n, snapshot, p));
    if (intersects(n.cpus, p.allowed_cpus))
      result.scope.executable_agents.push_back(id);
  }
  for (std::size_t i = 0; i < p.memory.size(); ++i) {
    const auto& n = *p.memory[i];
    const auto id = "linux/pool[" + std::to_string(i) + "]", service = id + "/service";
    result.graph.pools.push_back(
      {id, n.capacity_bytes, "host", snapshot.base_page_bytes, {snapshot.base_page_bytes}, service, service, id});
    result.graph.capacity_groups.push_back({id, n.capacity_bytes});
    result.graph.resources.push_back(
      {service, "linux_memory", {}, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}});
    bind_pool(result, id, n, snapshot, p);
  }
  for (const auto& a : result.graph.agents)
    for (const auto& pool : result.graph.pools)
      result.graph.access.push_back({a.id + "->" + pool.id,
                                     "linux_host_access",
                                     a.id,
                                     pool.id,
                                     {AccessMode::read, AccessMode::write},
                                     {},
                                     {},
                                     MappingMode::direct,
                                     {},
                                     pool.read_service,
                                     pool.write_service,
                                     {},
                                     {}});
  Method copy{"cpu_copy", true, {}, {}, {}, {}};
  for (const auto& source : result.graph.pools)
    for (const auto& target : result.graph.pools)
      copy.routes.push_back(
        {"$request:" + source.id + "->" + target.id,
         {},
         {},
         source.id,
         target.id,
         {{*source.read_service, DemandKind::read, 1, 0}, {*target.write_service, DemandKind::write, 1, 0}}});
  result.graph.methods.push_back(std::move(copy));
  // No template was bound, so this is not a complete HardwareInventory contract.
  result.inventory.bindings.clear();
  return result;
}
const LinuxNumaNode& pool_owner(const std::string& pool, const std::map<std::string, const LinuxNumaNode*>& agents) {
  const LinuxNumaNode* owner = nullptr;
  std::size_t depth = 0;
  for (const auto& [id, node] : agents)
    if (pool.starts_with(id + "/") && id.size() > depth) {
      owner = node;
      depth = id.size();
    }
  check(owner, "CPU memory role is not a descendant of a bound cpu-group agent: " + pool);
  return *owner;
}
LinuxDiscovery from_template(const LinuxTopologySnapshot& snapshot, const Physical& p,
                             const HardwareExpansion& expanded, LinuxCpuProfile profile) {
  if (profile == LinuxCpuProfile::lx2_flat)
    lx2_layout(p);
  LinuxDiscovery result;
  std::map<std::string, const LinuxNumaNode*> cpu_agents;
  std::size_t cpu_index = 0, dma_index = 0;
  for (const auto& a : expanded.graph.agents) {
    const auto req = expanded.binding_requests.find(a.id);
    check(req != expanded.binding_requests.end(), "agent missing discovery role");
    const auto& role = req->second;
    HardwareBinding b;
    b.address_space = "host";
    if (role == "cpu-group") {
      check(cpu_index < p.cpu.size(), "template contains more CPU roles than physical CPU groups");
      const auto& node = *p.cpu[cpu_index++];
      cpu_agents.emplace(a.id, &node);
      b.capabilities = cpu_capabilities();
      result.descriptors.push_back(cpu_descriptor(a.id, node, snapshot, p));
      if (intersects(node.cpus, p.allowed_cpus))
        result.scope.executable_agents.push_back(a.id);
    } else if (role == "sdma-role") {
      b.capabilities = std::vector<std::string>{"copy", "dma"};
      check(profile == LinuxCpuProfile::lx2_flat, "sdma-role requires the explicit LX2 backend profile");
      check(dma_index < 4, "LX2 profile supports exactly four declared SDMA roles");
      const auto device = p.dma.find(static_cast<unsigned>(dma_index));
      const bool allowed = device != p.dma.end() && device->second->accessible;
      result.descriptors.push_back({a.id,
                                    "linux_sdma",
                                    {{"role", std::to_string(dma_index)},
                                     {"device", device == p.dma.end() ? "" : device->second->path},
                                     {"accessible", allowed ? "true" : "false"},
                                     {"layout_profile", "lx2_flat"}}});
      if (allowed)
        result.scope.executable_agents.push_back(a.id);
      ++dma_index;
    } else
      throw std::invalid_argument("unsupported Linux CPU agent binding role: " + role);
    result.inventory.bindings.emplace(a.id, std::move(b));
  }
  check(cpu_index == p.cpu.size(), "template CPU roles do not cover the complete physical CPU topology");
  if (profile == LinuxCpuProfile::lx2_flat)
    check(dma_index == 4, "LX2 profile requires all four SDMA roles");
  std::set<unsigned> ddr_nodes, hbm_nodes;
  for (const auto& pool : expanded.graph.pools) {
    const auto req = expanded.binding_requests.find(pool.id);
    check(req != expanded.binding_requests.end(), "pool missing discovery role");
    const auto& role = req->second;
    const auto& cpu = pool_owner(pool.id, cpu_agents);
    const LinuxNumaNode* node = nullptr;
    if (role == "local-ddr") {
      node = &cpu;
      check(ddr_nodes.insert(node->os_numa).second, "multiple DDR pools claim the same capacity partition");
    } else if (role == "local-hbm") {
      check(profile == LinuxCpuProfile::lx2_flat, "local-hbm requires the explicit LX2 flat profile");
      const auto at = p.nodes.find(cpu.os_numa + 16);
      check(at != p.nodes.end(), "local flat HBM node absent");
      node = at->second;
      check(hbm_nodes.insert(node->os_numa).second, "multiple HBM pools claim the same capacity partition");
    } else
      throw std::invalid_argument("unsupported Linux CPU pool binding role: " + role);
    bind_pool(result, pool.id, *node, snapshot, p);
  }
  check(ddr_nodes.size() == p.cpu.size(), "template DDR roles do not cover all physical CPU groups");
  if (profile == LinuxCpuProfile::lx2_flat)
    check(hbm_nodes.size() == 16, "LX2 template does not cover all flat HBM pools");
  // The profile is selected by the caller. These are compatibility facts for
  // that profile; raw_cpu_model remains separate, unmodified evidence.
  if (profile == LinuxCpuProfile::lx2_flat) {
    result.inventory.facts = {{"cpu_model", std::string("LX2")},
                              {"cpu_groups", U{16}},
                              {"cores_per_group", U{38}},
                              {"hbm_mode", std::string("flat")}};
    for (auto& d : result.descriptors)
      d.labels.emplace("layout_profile", "lx2_flat");
  } else {
    result.inventory.facts = {{"cpu_groups", static_cast<U>(p.cpu.size())}};
  }
  for (const auto& access : expanded.graph.access)
    result.inventory.access.emplace(access.id, AccessDiscovery{MappingMode::direct, std::vector<std::string>{}});
  auto graph = bind_hardware_inventory(expanded, result.inventory);
  if (!graph)
    throw std::invalid_argument(std::get<Error>(graph.state).message);
  result.graph = std::move(graph).value();
  // Explicit template capacities may limit usable memory below physical totals.
  for (const auto& pool : result.graph.pools)
    result.pool_available_bytes[pool.id] = std::min(result.pool_available_bytes.at(pool.id), *pool.capacity_bytes);
  return result;
}
} // namespace
Result<std::vector<unsigned>> parse_linux_id_list(std::string_view text) {
  return checked([&] { return ids(text); });
}
Result<LinuxTopologySnapshot> probe_linux_topology() {
  return checked([] {
    LinuxTopologySnapshot result;
    const long page = ::sysconf(_SC_PAGESIZE);
    check(page > 0, "cannot determine Linux base page size");
    result.base_page_bytes = static_cast<U>(page);
    result.allowed_cpus = affinity();
    const auto status = colon_fields(read("/proc/self/status"));
    const auto mems = status.find("Mems_allowed_list");
    check(mems != status.end(), "missing process Mems_allowed_list");
    result.allowed_memory_nodes = ids(mems->second);
    const auto online = ids(read("/sys/devices/system/node/online"));
    const auto online_cpu_list = ids(read("/sys/devices/system/cpu/online"));
    const std::set<unsigned> online_cpus(online_cpu_list.begin(), online_cpu_list.end());
    const auto zone_pages = details::linux_zone_free_pages(read("/proc/zoneinfo"));
    for (auto node : online) {
      const auto root = std::filesystem::path("/sys/devices/system/node") / ("node" + std::to_string(node));
      auto cpus = ids(read(root / "cpulist"));
      std::erase_if(cpus, [&](unsigned cpu) { return !online_cpus.contains(cpu); });
      const auto meminfo = read(root / "meminfo");
      const U capacity = node_memory(meminfo, "MemTotal");
      const U free = node_memory(meminfo, "MemFree");
      const auto zone = zone_pages.find(node);
      check(!capacity || zone != zone_pages.end(), "missing NUMA kernel free-capacity source");
      const U pages = zone == zone_pages.end() ? 0 : zone->second;
      check(pages <= std::numeric_limits<U>::max() / result.base_page_bytes, "zone free capacity overflow");
      // These are separate kernel samples. Take the lower free estimate and
      // never count reclaimable pages, watermarks or counter uncertainty.
      result.nodes.push_back({node, std::move(cpus), capacity,
                              std::min(free, pages * result.base_page_bytes)});
    }
    const auto cpuinfo = colon_fields(read("/proc/cpuinfo"));
    for (const auto* key : {"model name", "Hardware", "CPU implementer", "CPU part", "CPU variant", "CPU revision"})
      if (const auto found = cpuinfo.find(key); found != cpuinfo.end()) {
        if (!result.raw_cpu_model.empty())
          result.raw_cpu_model += "; ";
        result.raw_cpu_model += std::string(key) + "=" + found->second;
      }
    for (unsigned role = 0; role < 4; ++role) {
      const std::string path = "/dev/sdma" + std::to_string(role);
      struct stat info {};
      if (::stat(path.c_str(), &info) == 0) {
        // Opening verifies device-cgroup/LSM permissions as well as DAC mode.
        // No channel initialization, registration or DMA operation is issued.
        int fd = S_ISCHR(info.st_mode) ? ::open(path.c_str(), O_RDWR | O_CLOEXEC) : -1;
        const bool accessible = fd >= 0;
        if (fd >= 0)
          ::close(fd);
        result.dma_devices.push_back({role, path, accessible});
      }
    }
    (void)validate(result);
    return result;
  });
}
Result<LinuxDiscovery> discover_linux_cpu(const LinuxTopologySnapshot& snapshot, const HardwareExpansion* expansion,
                                          LinuxCpuProfile profile) {
  return checked([&] {
    const auto physical = validate(snapshot);
    if (!expansion) {
      check(profile == LinuxCpuProfile::generic_cpu, "LX2 profile requires its explicit hardware template");
      return generic(snapshot, physical);
    }
    return from_template(snapshot, physical, *expansion, profile);
  });
}
} // namespace vecops::memory::model
