// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"

namespace vecops::memory::model {
/** Hardware requirements are opaque facts. They never drive topology parsing. */
using HardwareFact = std::variant<std::string, std::uint64_t, bool>;
using HardwareFacts = std::map<std::string, HardwareFact>;
using RuntimeCounts = std::map<std::string, std::uint64_t>;
struct HardwareExpansion {
  std::string name;
  ResourceGraph graph;
  BindingRequests binding_requests; // generated agent/pool ID -> opaque discovery role
  HardwareFacts requirements;
};
/** Metadata returned by discovery for one generated agent/pool ID.
 * No OS pointers, file descriptors or device handles cross this interface.
 * A discovered capacity may exceed the template's explicit usable capacity.
 */
struct HardwareBinding {
  std::optional<std::uint64_t> capacity_bytes;
  std::optional<std::string> address_space;
  std::optional<std::uint64_t> granularity_bytes;
  std::vector<std::uint64_t> page_sizes_bytes;
  std::optional<std::vector<std::string>> capabilities;
  std::optional<std::string> capacity_group;
  std::optional<std::uint64_t> capacity_group_bytes;
};
struct AccessDiscovery {
  std::optional<MappingMode> mapping;
  std::optional<std::vector<std::string>> prerequisites;
};
struct HardwareInventory {
  HardwareFacts facts;
  std::map<std::string, HardwareBinding> bindings;
  std::map<std::string, AccessDiscovery> access; // optional generated AccessConnection IDs
};
/** Expand memory-model/0.3, with no hardware-name interpretation.
 * Every count="runtime" needs an explicit RuntimeCounts entry. Unknown entries
 * are errors. Unresolved discovered properties remain absent in the result;
 * call bind_hardware_inventory before using the graph for actual allocation.
 */
Result<HardwareExpansion> expand_hardware_template(std::string_view toml, const RuntimeCounts& counts = {});
/** Verify opaque require facts and exact discovery-role coverage, then resolve
 * pool capacities, shared-capacity aliases, pages, capabilities and mappings.
 * Pools require a discovered address space and allocation/page granularity.
 */
Result<ResourceGraph> bind_hardware_inventory(const HardwareExpansion&, const HardwareInventory&);
Result<ResourceGraph> compile_hardware_template(std::string_view toml, const RuntimeCounts&, const HardwareInventory&);
/** Parse an explicit positive whole-byte quantity, e.g. 1.5 GiB, 64 KiB, 4096 B.
 * Uses exact decimal arithmetic, accepts the full uint64 range, and rejects
 * fractional bytes, unitless numbers, unknown units and overflow.
 */
Result<std::uint64_t> parse_memory_size(std::string_view);
} // namespace vecops::memory::model
