// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/HardwareTemplate.h"
#include "vecops/memory/LinuxBackend.h"

namespace vecops::memory::model {
struct LinuxNumaNode {
  unsigned os_numa;
  std::vector<unsigned> cpus; // online logical CPUs; empty for memory-only NUMA
  std::uint64_t capacity_bytes;
  std::uint64_t available_bytes; // non-reclaiming headroom after kernel reserves and counter uncertainty
};
struct LinuxDmaDevice {
  unsigned role;
  std::string path;
  bool accessible; // discovery successfully opened a character device for read/write
};
/** Injectable physical topology + current process grants, without backend handles. */
struct LinuxTopologySnapshot {
  std::vector<LinuxNumaNode> nodes;
  std::vector<unsigned> allowed_cpus; // actual sched_getaffinity result
  std::vector<unsigned> allowed_memory_nodes; // /proc/self/status Mems_allowed_list
  std::uint64_t base_page_bytes;
  std::string raw_cpu_model;
  std::vector<LinuxDmaDevice> dma_devices;
};
enum class LinuxCpuProfile {
  generic_cpu,
  lx2_flat // explicit layout-compatibility profile; does not identify the silicon
};
struct LinuxDiscovery {
  ResourceGraph graph; // complete physical graph, before process-scope projection
  Scope scope;
  std::vector<LinuxPoolBinding> pool_bindings;
  std::vector<BackendDescriptor> descriptors;
  std::map<std::string, std::uint64_t> pool_available_bytes;
  HardwareInventory inventory; // populated for a supplied HardwareExpansion
};
/** Parse Linux comma/range lists. Empty input denotes the empty CPU/node set. */
Result<std::vector<unsigned>> parse_linux_id_list(std::string_view);
/** Lightweight probe of sysfs, /proc and process grants. SDMA device access is
 * checked by open/close without issuing DMA or driver ioctls. */
Result<LinuxTopologySnapshot> probe_linux_topology();
/** Generic CPU graph when expansion is null; otherwise bind all physical roles
 * before deriving the process scope. CPU affinity grants execution only; memory
 * allocation follows Mems_allowed and does not require a local active CPU agent.
 * Accessible host pools include importable memory outside the allocation grant.
 * The LX2 profile verifies all 16 x 38 CPU groups and 16 x 4 GiB flat-HBM nodes.
 */
Result<LinuxDiscovery> discover_linux_cpu(const LinuxTopologySnapshot&, const HardwareExpansion* expansion = nullptr,
                                          LinuxCpuProfile profile = LinuxCpuProfile::generic_cpu);
} // namespace vecops::memory::model
