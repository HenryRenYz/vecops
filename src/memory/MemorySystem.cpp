//
// Copyright (c) vecops contributors.
//

/** @file MemorySystem.cpp @brief Topology discovery and direct allocations. */

#include "vecops/memory/MemorySystem.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

#if defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#endif

#if defined(VECOPS_MEMORY_HAVE_HWLOC)
#  include <hwloc.h>
#  include <hwloc/helper.h>
#endif

namespace vecops::memory {

namespace details {

constexpr std::uint64_t allocation_magic = UINT64_C(0x5645434f50534d45);

struct TargetCounters {
  std::atomic<std::uint64_t> managed_bytes{0};
  std::atomic<std::uint64_t> peak_managed_bytes{0};
  std::atomic<std::uint64_t> allocation_count{0};
  std::atomic<std::uint64_t> failed_allocation_count{0};
  std::atomic<std::uint64_t> fallback_count{0};
  std::uint64_t max_managed_bytes = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t min_free_bytes = 0;
};

struct MemoryState {
  TopologySnapshot snapshot;
  BackendPreference backend = BackendPreference::System;
  std::vector<std::unique_ptr<TargetCounters>> counters;

#if defined(VECOPS_MEMORY_HAVE_HWLOC)
  hwloc_topology_t hwloc_topology = nullptr;
  std::vector<hwloc_obj_t> target_objects;
  std::vector<hwloc_bitmap_t> domain_cpusets;
#endif

  ~MemoryState() {
#if defined(VECOPS_MEMORY_HAVE_HWLOC)
    for (auto* cpuset : domain_cpusets)
      hwloc_bitmap_free(cpuset);
    if (hwloc_topology != nullptr)
      hwloc_topology_destroy(hwloc_topology);
#endif
  }

  MemoryState(const MemoryState&) = delete;
  MemoryState& operator=(const MemoryState&) = delete;
  MemoryState() = default;
};

struct AllocationHeader {
  std::uint64_t magic;
  std::shared_ptr<MemoryState> state;
  void* backend_base;
  std::size_t backend_bytes;
  std::size_t requested_bytes;
  MemoryTargetId target;
};

struct AllocationAccess {
  static Allocation make(void* data, std::size_t size, MemoryTargetId target) noexcept {
    return Allocation(data, size, target);
  }
};

static_assert(sizeof(AllocationHeader) % alignof(AllocationHeader) == 0);

std::optional<std::uint64_t> os_free_bytes(unsigned os_numa_id) {
#if defined(__linux__)
  std::ifstream input("/sys/devices/system/node/node" + std::to_string(os_numa_id) + "/meminfo");
  std::string token;
  while (input >> token) {
    if (token != "MemFree:")
      continue;
    std::uint64_t kib = 0;
    input >> kib;
    if (kib > std::numeric_limits<std::uint64_t>::max() / 1024)
      return std::nullopt;
    return kib * 1024;
  }
#else
  (void)os_numa_id;
#endif
  return std::nullopt;
}

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void update_peak(std::atomic<std::uint64_t>& peak, std::uint64_t value) {
  auto current = peak.load(std::memory_order_relaxed);
  while (current < value && !peak.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
}

void reserve_budget(TargetCounters& counters, const MemoryTargetInfo& target, std::size_t bytes) {
  const auto free = os_free_bytes(target.os_numa_id);
  if (free.has_value() && (free.value() <= counters.min_free_bytes || bytes > free.value() - counters.min_free_bytes)) {
    throw MemoryError(MemoryErrc::BudgetExceeded, "memory target OS node " + std::to_string(target.os_numa_id) +
                                                    " does not have the configured free-memory headroom");
  }

  auto current = counters.managed_bytes.load(std::memory_order_relaxed);
  while (true) {
    if (bytes > counters.max_managed_bytes || current > counters.max_managed_bytes - bytes) {
      throw MemoryError(MemoryErrc::BudgetExceeded, "memory target OS node " + std::to_string(target.os_numa_id) +
                                                      " exceeds its managed-memory budget");
    }
    if (counters.managed_bytes.compare_exchange_weak(current, current + bytes, std::memory_order_relaxed)) {
      update_peak(counters.peak_managed_bytes, current + bytes);
      return;
    }
  }
}

void* backend_allocate(MemoryState& state, MemoryTargetId target, std::size_t bytes) {
#if defined(VECOPS_MEMORY_HAVE_HWLOC)
  if (state.backend == BackendPreference::Hwloc) {
    auto* nodeset = hwloc_bitmap_alloc();
    if (nodeset == nullptr)
      return nullptr;
    hwloc_bitmap_only(nodeset, state.snapshot.memory_targets[target].os_numa_id);
    void* result = hwloc_alloc_membind(state.hwloc_topology, bytes, nodeset, HWLOC_MEMBIND_BIND,
                                       HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_NOCPUBIND);
    hwloc_bitmap_free(nodeset);
    return result;
  }
#else
  (void)target;
#endif
  return std::malloc(bytes);
}

void backend_deallocate(MemoryState& state, void* base, std::size_t bytes) noexcept {
#if defined(VECOPS_MEMORY_HAVE_HWLOC)
  if (state.backend == BackendPreference::Hwloc) {
    (void)hwloc_free(state.hwloc_topology, base, bytes);
    return;
  }
#else
  (void)bytes;
#endif
  std::free(base);
}

void release_allocation(void* data) noexcept {
  if (data == nullptr)
    return;
  auto* header = reinterpret_cast<AllocationHeader*>(static_cast<std::byte*>(data) - sizeof(AllocationHeader));
  if (header->magic != allocation_magic)
    std::terminate();

  auto state = std::move(header->state);
  const void* const base = header->backend_base;
  const auto backend_bytes = header->backend_bytes;
  const auto requested_bytes = header->requested_bytes;
  const auto target = header->target;
  header->magic = 0;
  header->~AllocationHeader();
  state->counters[target]->managed_bytes.fetch_sub(requested_bytes, std::memory_order_relaxed);
  backend_deallocate(*state, const_cast<void*>(base), backend_bytes);
}

std::uint64_t system_capacity() {
#if defined(__linux__)
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0 &&
      static_cast<std::uint64_t>(pages) <=
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(page_size)) {
    return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
  }
#endif
  return 0;
}

std::vector<unsigned> allowed_system_cpus() {
  std::vector<unsigned> result;
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
    for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &cpuset))
        result.push_back(cpu);
    }
  }
#endif
  if (result.empty()) {
    for (unsigned cpu = 0; cpu < std::thread::hardware_concurrency(); ++cpu)
      result.push_back(cpu);
  }
  return result;
}

MemoryKind kind_from_subtype(const char* subtype) {
  if (subtype == nullptr)
    return MemoryKind::Unknown;
  const std::string_view value(subtype);
  if (value == "DRAM")
    return MemoryKind::DRAM;
  if (value == "HBM")
    return MemoryKind::HBM;
  if (value == "CXL-DRAM" || value == "CXL")
    return MemoryKind::CXL;
  if (value == "NVM" || value == "PMEM")
    return MemoryKind::PMEM;
  return MemoryKind::Unknown;
}

#if defined(VECOPS_MEMORY_HAVE_HWLOC)

bool os_node_has_cpus(unsigned os_numa_id) {
#  if defined(__linux__)
  std::ifstream input("/sys/devices/system/node/node" + std::to_string(os_numa_id) + "/cpulist");
  if (input.is_open()) {
    std::string cpulist;
    std::getline(input, cpulist);
    return std::any_of(cpulist.begin(), cpulist.end(), [](unsigned char value) { return !std::isspace(value); });
  }
#  else
  (void)os_numa_id;
#  endif
  return true;
}

std::vector<unsigned> bitmap_cpu_ids(hwloc_const_bitmap_t cpuset) {
  std::vector<unsigned> result;
  int cpu = -1;
  while ((cpu = hwloc_bitmap_next(cpuset, cpu)) != -1)
    result.push_back(static_cast<unsigned>(cpu));
  return result;
}

std::optional<std::uint64_t> memory_attribute(hwloc_topology_t topology, hwloc_memattr_id_t attribute,
                                              hwloc_obj_t target, hwloc_const_bitmap_t initiator) {
  hwloc_location location{};
  location.type = HWLOC_LOCATION_TYPE_CPUSET;
  location.location.cpuset = const_cast<hwloc_cpuset_t>(initiator);
  hwloc_uint64_t value = 0;
  if (hwloc_memattr_get_value(topology, attribute, target, &location, 0, &value) != 0)
    return std::nullopt;
  return value;
}

std::shared_ptr<MemoryState> discover_hwloc() {
  auto state = std::make_shared<MemoryState>();
  if (hwloc_topology_init(&state->hwloc_topology) != 0)
    throw MemoryError(MemoryErrc::ConfigurationInvalid, "hwloc_topology_init failed");
  if (hwloc_topology_load(state->hwloc_topology) != 0)
    throw MemoryError(MemoryErrc::ConfigurationInvalid, "hwloc_topology_load failed");

  state->backend = BackendPreference::Hwloc;
  state->snapshot.backend = "hwloc";
  const auto* allowed_cpus = hwloc_topology_get_allowed_cpuset(state->hwloc_topology);
  const auto* allowed_nodes = hwloc_topology_get_allowed_nodeset(state->hwloc_topology);

  for (auto* object = hwloc_get_next_obj_by_type(state->hwloc_topology, HWLOC_OBJ_NUMANODE, nullptr); object != nullptr;
       object = hwloc_get_next_obj_by_type(state->hwloc_topology, HWLOC_OBJ_NUMANODE, object)) {
    if (object->os_index == HWLOC_UNKNOWN_INDEX ||
        (allowed_nodes != nullptr && !hwloc_bitmap_isset(allowed_nodes, static_cast<int>(object->os_index)))) {
      continue;
    }
    const auto id = static_cast<MemoryTargetId>(state->snapshot.memory_targets.size());
    state->snapshot.memory_targets.push_back(MemoryTargetInfo{id, object->os_index, kind_from_subtype(object->subtype),
                                                              object->attr->numanode.local_memory});
    state->target_objects.push_back(object);
  }
  if (state->snapshot.memory_targets.empty())
    throw MemoryError(MemoryErrc::ConfigurationInvalid, "hwloc found no allowed NUMA memory targets");

  for (std::size_t index = 0; index < state->snapshot.memory_targets.size(); ++index) {
    const auto& target = state->snapshot.memory_targets[index];
    auto* object = state->target_objects[index];
    if (!os_node_has_cpus(target.os_numa_id) || object->cpuset == nullptr || hwloc_bitmap_iszero(object->cpuset))
      continue;
    auto* cpuset = hwloc_bitmap_dup(object->cpuset);
    if (cpuset == nullptr)
      throw std::bad_alloc();
    if (allowed_cpus != nullptr)
      hwloc_bitmap_and(cpuset, cpuset, allowed_cpus);
    if (hwloc_bitmap_iszero(cpuset)) {
      hwloc_bitmap_free(cpuset);
      continue;
    }
    bool duplicate = false;
    for (const auto* existing : state->domain_cpusets) {
      if (hwloc_bitmap_isequal(existing, cpuset)) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      hwloc_bitmap_free(cpuset);
      continue;
    }
    const auto domain = static_cast<CpuDomainId>(state->snapshot.cpu_domains.size());
    state->snapshot.cpu_domains.push_back(CpuDomainInfo{domain, target.os_numa_id, bitmap_cpu_ids(cpuset)});
    state->domain_cpusets.push_back(cpuset);
  }

  if (state->snapshot.cpu_domains.empty()) {
    auto* cpuset = hwloc_bitmap_dup(allowed_cpus);
    if (cpuset == nullptr)
      throw std::bad_alloc();
    state->snapshot.cpu_domains.push_back(CpuDomainInfo{0, std::nullopt, bitmap_cpu_ids(cpuset)});
    state->domain_cpusets.push_back(cpuset);
  }

  for (std::size_t domain = 0; domain < state->snapshot.cpu_domains.size(); ++domain) {
    for (std::size_t target = 0; target < state->snapshot.memory_targets.size(); ++target) {
      auto* object = state->target_objects[target];
      auto* locality = hwloc_bitmap_dup(object->cpuset);
      if (locality == nullptr)
        throw std::bad_alloc();
      if (allowed_cpus != nullptr)
        hwloc_bitmap_and(locality, locality, allowed_cpus);
      const bool exact = hwloc_bitmap_isequal(locality, state->domain_cpusets[domain]);
      hwloc_bitmap_free(locality);
      const auto bandwidth =
        memory_attribute(state->hwloc_topology, HWLOC_MEMATTR_ID_BANDWIDTH, object, state->domain_cpusets[domain]);
      const auto latency =
        memory_attribute(state->hwloc_topology, HWLOC_MEMATTR_ID_LATENCY, object, state->domain_cpusets[domain]);
      state->snapshot.memory_paths.push_back(
        MemoryPathInfo{static_cast<CpuDomainId>(domain), static_cast<MemoryTargetId>(target), bandwidth, latency,
                       bandwidth.has_value() ? AttributeSource::Hwloc : AttributeSource::Unknown,
                       latency.has_value() ? AttributeSource::Hwloc : AttributeSource::Unknown, exact});
    }
  }
  return state;
}

#endif

std::shared_ptr<MemoryState> discover_system() {
#if defined(__linux__)
  unsigned numa_nodes = 0;
  for (unsigned node = 0; node < 1024; ++node) {
    std::ifstream cpulist("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    if (cpulist.good())
      ++numa_nodes;
  }
  if (numa_nodes > 1) {
    throw MemoryError(MemoryErrc::UnsupportedBinding,
                      "the portable memory backend cannot represent a multi-NUMA machine; enable hwloc");
  }
#endif
  auto state = std::make_shared<MemoryState>();
  state->backend = BackendPreference::System;
  state->snapshot.backend = "system";
  state->snapshot.cpu_domains.push_back(CpuDomainInfo{0, std::nullopt, allowed_system_cpus()});
  state->snapshot.memory_targets.push_back(MemoryTargetInfo{0, 0, MemoryKind::Unknown, system_capacity()});
  state->snapshot.memory_paths.push_back(
    MemoryPathInfo{0, 0, std::nullopt, std::nullopt, AttributeSource::Unknown, AttributeSource::Unknown, true});
  return state;
}

void apply_configuration(MemoryState& state, const MemoryConfig& config) {
  state.counters.reserve(state.snapshot.memory_targets.size());
  for (const auto& target : state.snapshot.memory_targets) {
    auto counters = std::make_unique<TargetCounters>();
    for (const auto& override : config.target_overrides) {
      if (override.os_numa_id.has_value() && override.os_numa_id.value() != target.os_numa_id)
        continue;
      if (override.max_managed_bytes.has_value())
        counters->max_managed_bytes = override.max_managed_bytes.value();
      if (override.min_free_bytes.has_value())
        counters->min_free_bytes = override.min_free_bytes.value();
    }
    state.counters.push_back(std::move(counters));
  }

  for (auto& target : state.snapshot.memory_targets) {
    for (const auto& override : config.target_overrides) {
      if (override.os_numa_id.has_value() && override.os_numa_id.value() != target.os_numa_id)
        continue;
      if (override.kind.has_value())
        target.kind = override.kind.value();
    }
  }

  for (auto& path : state.snapshot.memory_paths) {
    const auto& domain = state.snapshot.cpu_domains[path.initiator];
    const auto& target = state.snapshot.memory_targets[path.target];
    for (const auto& override : config.path_overrides) {
      if (override.target_os_numa_id != target.os_numa_id)
        continue;
      if (override.initiator_os_numa_id.has_value() &&
          (!domain.os_numa_id.has_value() || domain.os_numa_id.value() != override.initiator_os_numa_id.value())) {
        continue;
      }
      if (override.bandwidth_mib_s.has_value()) {
        path.bandwidth_mib_s = override.bandwidth_mib_s;
        path.bandwidth_source = AttributeSource::Configuration;
      }
      if (override.latency_ns.has_value()) {
        path.latency_ns = override.latency_ns;
        path.latency_source = AttributeSource::Configuration;
      }
    }
  }
}

const MemoryPathInfo& path_for(const MemoryState& state, CpuDomainId domain, MemoryTargetId target) {
  const auto found = std::find_if(state.snapshot.memory_paths.begin(), state.snapshot.memory_paths.end(),
                                  [&](const auto& path) { return path.initiator == domain && path.target == target; });
  if (found == state.snapshot.memory_paths.end())
    throw MemoryError(MemoryErrc::UnknownTarget, "memory path is absent from the topology snapshot");
  return *found;
}

std::vector<MemoryTargetId> local_targets(const MemoryState& state, CpuDomainId domain) {
  std::vector<MemoryTargetId> result;
  for (const auto& path : state.snapshot.memory_paths) {
    if (path.initiator == domain && path.exact_locality)
      result.push_back(path.target);
  }
  if (result.empty()) {
    for (const auto& target : state.snapshot.memory_targets)
      result.push_back(target.id);
  }
  return result;
}

MemoryTargetId default_target(const MemoryState& state, CpuDomainId domain) {
  auto candidates = local_targets(state, domain);
  auto better = [&](MemoryTargetId left, MemoryTargetId right) {
    const auto& a = path_for(state, domain, left);
    const auto& b = path_for(state, domain, right);
    if (a.latency_ns.has_value() != b.latency_ns.has_value())
      return a.latency_ns.has_value();
    if (a.latency_ns.has_value() && a.latency_ns.value() != b.latency_ns.value())
      return a.latency_ns.value() < b.latency_ns.value();
    return state.snapshot.memory_targets[left].capacity_bytes > state.snapshot.memory_targets[right].capacity_bytes;
  };
  return *std::min_element(candidates.begin(), candidates.end(),
                           [&](auto left, auto right) { return better(left, right); });
}

MemoryTargetId objective_target(const MemoryState& state, CpuDomainId domain, RankingObjective objective) {
  auto candidates = local_targets(state, domain);
  auto better = [&](MemoryTargetId left, MemoryTargetId right) {
    const auto& a = path_for(state, domain, left);
    const auto& b = path_for(state, domain, right);
    const auto av = objective == RankingObjective::Bandwidth ? a.bandwidth_mib_s : a.latency_ns;
    const auto bv = objective == RankingObjective::Bandwidth ? b.bandwidth_mib_s : b.latency_ns;
    if (av.has_value() != bv.has_value())
      return av.has_value();
    if (av.has_value() && av.value() != bv.value()) {
      return objective == RankingObjective::Bandwidth ? av.value() > bv.value() : av.value() < bv.value();
    }
    return state.snapshot.memory_targets[left].capacity_bytes > state.snapshot.memory_targets[right].capacity_bytes;
  };
  return *std::min_element(candidates.begin(), candidates.end(),
                           [&](auto left, auto right) { return better(left, right); });
}

MemoryTargetId target_by_os_index(const MemoryState& state, unsigned os_numa_id) {
  const auto found = std::find_if(state.snapshot.memory_targets.begin(), state.snapshot.memory_targets.end(),
                                  [&](const auto& target) { return target.os_numa_id == os_numa_id; });
  if (found == state.snapshot.memory_targets.end())
    throw MemoryError(MemoryErrc::UnknownTarget, "unknown OS NUMA node " + std::to_string(os_numa_id));
  return found->id;
}

Allocation allocate_on_target(const std::shared_ptr<MemoryState>& state, MemoryTargetId target,
                              const AllocationRequest& request) {
  const auto& target_info = state->snapshot.memory_targets[target];
  auto& counters = *state->counters[target];
  try {
    reserve_budget(counters, target_info, request.bytes);
  } catch (...) {
    counters.failed_allocation_count.fetch_add(1, std::memory_order_relaxed);
    throw;
  }

  const auto alignment = std::max(request.alignment, alignof(AllocationHeader));
  constexpr auto header_bytes = sizeof(AllocationHeader);
  if (request.bytes > std::numeric_limits<std::size_t>::max() - header_bytes - (alignment - 1)) {
    counters.managed_bytes.fetch_sub(request.bytes, std::memory_order_relaxed);
    counters.failed_allocation_count.fetch_add(1, std::memory_order_relaxed);
    throw MemoryError(MemoryErrc::InvalidRequest, "allocation size overflows the host address space");
  }
  const auto backend_bytes = request.bytes + header_bytes + alignment - 1;
  errno = 0;
  void* base = backend_allocate(*state, target, backend_bytes);
  if (base == nullptr) {
    counters.managed_bytes.fetch_sub(request.bytes, std::memory_order_relaxed);
    counters.failed_allocation_count.fetch_add(1, std::memory_order_relaxed);
    const auto code = errno == ENOMEM ? MemoryErrc::OutOfMemory : MemoryErrc::BindingFailed;
    throw MemoryError(code, "allocation on OS NUMA node " + std::to_string(target_info.os_numa_id) +
                              " failed: " + std::strerror(errno));
  }

  const auto raw = reinterpret_cast<std::uintptr_t>(base) + header_bytes;
  const auto aligned = (raw + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
  auto* data = reinterpret_cast<void*>(aligned);
  auto* header = reinterpret_cast<AllocationHeader*>(aligned - header_bytes);
  new (header) AllocationHeader{allocation_magic, state, base, backend_bytes, request.bytes, target};
  counters.allocation_count.fetch_add(1, std::memory_order_relaxed);
  return AllocationAccess::make(data, request.bytes, target);
}

} // namespace details

MemoryError::MemoryError(MemoryErrc code, std::string message)
  : std::runtime_error(std::move(message))
  , code_(code) {
}

Allocation::Allocation(void* data, std::size_t size, MemoryTargetId target) noexcept
  : data_(data)
  , size_(size)
  , target_(target) {
}

Allocation::Allocation(Allocation&& other) noexcept
  : data_(std::exchange(other.data_, nullptr))
  , size_(std::exchange(other.size_, 0))
  , target_(std::exchange(other.target_, std::nullopt)) {
}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    target_ = std::exchange(other.target_, std::nullopt);
  }
  return *this;
}

Allocation::~Allocation() {
  reset();
}

void Allocation::reset() noexcept {
  details::release_allocation(std::exchange(data_, nullptr));
  size_ = 0;
  target_.reset();
}

MemorySystem::MemorySystem(std::shared_ptr<details::MemoryState> state) noexcept
  : state_(std::move(state)) {
}

MemorySystem MemorySystem::discover(const MemoryConfig& config) {
  std::shared_ptr<details::MemoryState> state;
  if (config.backend == BackendPreference::Hwloc) {
#if defined(VECOPS_MEMORY_HAVE_HWLOC)
    state = details::discover_hwloc();
#else
    throw MemoryError(MemoryErrc::UnsupportedBinding, "this vecops::memory build does not include hwloc support");
#endif
  } else if (config.backend == BackendPreference::System) {
    state = details::discover_system();
  } else {
#if defined(VECOPS_MEMORY_HAVE_HWLOC)
    state = details::discover_hwloc();
#else
    state = details::discover_system();
#endif
  }
  details::apply_configuration(*state, config);
  return MemorySystem(std::move(state));
}

const TopologySnapshot& MemorySystem::topology() const {
  if (state_ == nullptr)
    throw MemoryError(MemoryErrc::ConfigurationInvalid, "MemorySystem is not initialized");
  return state_->snapshot;
}

CpuDomainId MemorySystem::current_cpu_domain() const {
  const auto& snapshot = topology();
  if (snapshot.cpu_domains.size() == 1)
    return snapshot.cpu_domains.front().id;

#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    std::optional<CpuDomainId> match;
    for (const auto& domain : snapshot.cpu_domains) {
      bool contains_all = true;
      bool contains_any = false;
      for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &affinity))
          continue;
        const bool contains = std::find(domain.cpu_ids.begin(), domain.cpu_ids.end(), cpu) != domain.cpu_ids.end();
        contains_all = contains_all && contains;
        contains_any = contains_any || contains;
      }
      if (contains_all && contains_any) {
        if (match.has_value()) {
          match.reset();
          break;
        }
        match = domain.id;
      }
    }
    if (match.has_value())
      return match.value();
  }

  const int cpu = sched_getcpu();
  if (cpu >= 0) {
    for (const auto& domain : snapshot.cpu_domains) {
      if (std::find(domain.cpu_ids.begin(), domain.cpu_ids.end(), static_cast<unsigned>(cpu)) != domain.cpu_ids.end())
        return domain.id;
    }
  }
#endif
  throw MemoryError(MemoryErrc::UnknownCpuDomain, "current thread does not resolve to a discovered CPU domain");
}

MemoryTierView MemorySystem::tiers(CpuDomainId domain, RankingObjective objective) const {
  const auto& snapshot = topology();
  if (domain >= snapshot.cpu_domains.size())
    throw MemoryError(MemoryErrc::UnknownCpuDomain, "CPU domain is outside the topology snapshot");

  std::vector<MemoryTargetId> targets;
  for (const auto& path : snapshot.memory_paths) {
    if (path.initiator == domain)
      targets.push_back(path.target);
  }
  auto value = [&](MemoryTargetId target) {
    const auto& path = details::path_for(*state_, domain, target);
    return objective == RankingObjective::Bandwidth ? path.bandwidth_mib_s : path.latency_ns;
  };
  std::stable_sort(targets.begin(), targets.end(), [&](auto left, auto right) {
    const auto a = value(left);
    const auto b = value(right);
    if (a.has_value() != b.has_value())
      return a.has_value();
    if (a == b) {
      const auto& ap = details::path_for(*state_, domain, left);
      const auto& bp = details::path_for(*state_, domain, right);
      if (ap.exact_locality != bp.exact_locality)
        return ap.exact_locality;
      return snapshot.memory_targets[left].os_numa_id < snapshot.memory_targets[right].os_numa_id;
    }
    return objective == RankingObjective::Bandwidth ? a.value() > b.value() : a.value() < b.value();
  });

  MemoryTierView result{domain, objective, {}};
  for (const auto target : targets) {
    const auto current = value(target);
    if (result.ranks.empty() || result.ranks.back().representative_value != current) {
      result.ranks.push_back(MemoryTierRank{static_cast<unsigned>(result.ranks.size()), {}, current});
    }
    result.ranks.back().targets.push_back(target);
  }
  return result;
}

Allocation MemorySystem::allocate(const AllocationRequest& request) const {
  const auto& snapshot = topology();
  if (!details::is_power_of_two(request.alignment))
    throw MemoryError(MemoryErrc::InvalidRequest, "allocation alignment must be a positive power of two");
  if (request.objective_rank.has_value() && request.intent != PlacementIntent::HighBandwidth &&
      request.intent != PlacementIntent::LowLatency) {
    throw MemoryError(MemoryErrc::InvalidRequest,
                      "objective_rank requires HighBandwidth or LowLatency placement intent");
  }
  if (request.intent != PlacementIntent::ExactTarget && request.exact_os_numa_id.has_value()) {
    throw MemoryError(MemoryErrc::InvalidRequest, "exact_os_numa_id requires ExactTarget placement intent");
  }
  if (request.bytes == 0)
    return {};

  const auto domain =
    request.domain.kind == CpuDomainSelector::Kind::Specific ? request.domain.id : current_cpu_domain();
  if (domain >= snapshot.cpu_domains.size())
    throw MemoryError(MemoryErrc::UnknownCpuDomain, "CPU domain is outside the topology snapshot");

  MemoryTargetId target = 0;
  if (request.intent == PlacementIntent::ExactTarget) {
    if (!request.exact_os_numa_id.has_value())
      throw MemoryError(MemoryErrc::InvalidRequest, "ExactTarget requires exact_os_numa_id");
    target = details::target_by_os_index(*state_, request.exact_os_numa_id.value());
  } else if (request.objective_rank.has_value()) {
    const auto objective =
      request.intent == PlacementIntent::LowLatency ? RankingObjective::Latency : RankingObjective::Bandwidth;
    const auto view = tiers(domain, objective);
    if (request.objective_rank.value() >= view.ranks.size() ||
        view.ranks[request.objective_rank.value()].targets.empty())
      throw MemoryError(MemoryErrc::UnknownTarget, "requested memory tier rank does not exist");
    target = view.ranks[request.objective_rank.value()].targets.front();
  } else if (request.intent == PlacementIntent::HighBandwidth) {
    target = details::objective_target(*state_, domain, RankingObjective::Bandwidth);
  } else if (request.intent == PlacementIntent::LowLatency) {
    target = details::objective_target(*state_, domain, RankingObjective::Latency);
  } else {
    target = details::default_target(*state_, domain);
  }

  try {
    return details::allocate_on_target(state_, target, request);
  } catch (const MemoryError& error) {
    const auto fallback = details::default_target(*state_, domain);
    const bool recoverable = error.code() == MemoryErrc::BudgetExceeded || error.code() == MemoryErrc::OutOfMemory ||
                             error.code() == MemoryErrc::BindingFailed ||
                             error.code() == MemoryErrc::UnsupportedBinding;
    if (request.fallback != FallbackPolicy::ToDefault || fallback == target || !recoverable)
      throw;
    state_->counters[target]->fallback_count.fetch_add(1, std::memory_order_relaxed);
    return details::allocate_on_target(state_, fallback, request);
  }
}

std::vector<TargetRuntimeStats> MemorySystem::stats() const {
  const auto& snapshot = topology();
  std::vector<TargetRuntimeStats> result;
  result.reserve(snapshot.memory_targets.size());
  for (const auto& target : snapshot.memory_targets) {
    const auto& counters = *state_->counters[target.id];
    const auto managed = counters.managed_bytes.load(std::memory_order_relaxed);
    std::optional<std::uint64_t> remaining;
    if (counters.max_managed_bytes != std::numeric_limits<std::uint64_t>::max())
      remaining = counters.max_managed_bytes > managed ? counters.max_managed_bytes - managed : 0;
    result.push_back(TargetRuntimeStats{
      target.id,
      target.os_numa_id,
      managed,
      counters.peak_managed_bytes.load(std::memory_order_relaxed),
      counters.allocation_count.load(std::memory_order_relaxed),
      counters.failed_allocation_count.load(std::memory_order_relaxed),
      counters.fallback_count.load(std::memory_order_relaxed),
      details::os_free_bytes(target.os_numa_id),
      remaining,
    });
  }
  return result;
}

std::string MemorySystem::describe() const {
  const auto& snapshot = topology();
  std::ostringstream output;
  output << "memory backend: " << snapshot.backend << '\n';
  for (const auto& domain : snapshot.cpu_domains) {
    output << "CPU domain " << domain.id;
    if (domain.os_numa_id.has_value())
      output << " (OS node " << domain.os_numa_id.value() << ')';
    output << ", CPUs=";
    for (std::size_t index = 0; index < domain.cpu_ids.size(); ++index) {
      if (index != 0)
        output << ',';
      output << domain.cpu_ids[index];
    }
    output << '\n';
    for (const auto& path : snapshot.memory_paths) {
      if (path.initiator != domain.id)
        continue;
      const auto& target = snapshot.memory_targets[path.target];
      output << "  target " << target.id << " (OS node " << target.os_numa_id << ", " << to_string(target.kind) << ", "
             << (target.capacity_bytes >> 20) << " MiB)";
      output << " bandwidth=";
      if (path.bandwidth_mib_s.has_value())
        output << path.bandwidth_mib_s.value() << " MiB/s";
      else
        output << "unknown";
      output << " latency=";
      if (path.latency_ns.has_value())
        output << path.latency_ns.value() << " ns";
      else
        output << "unknown";
      output << " local=" << (path.exact_locality ? "yes" : "no") << '\n';
    }
  }
  return output.str();
}

std::string to_string(MemoryKind value) {
  switch (value) {
  case MemoryKind::Unknown:
    return "unknown";
  case MemoryKind::DRAM:
    return "dram";
  case MemoryKind::HBM:
    return "hbm";
  case MemoryKind::CXL:
    return "cxl";
  case MemoryKind::PMEM:
    return "pmem";
  }
  return "unknown";
}

std::string to_string(RankingObjective value) {
  return value == RankingObjective::Bandwidth ? "bandwidth" : "latency";
}

std::string to_string(PlacementIntent value) {
  switch (value) {
  case PlacementIntent::Default:
    return "default";
  case PlacementIntent::HighBandwidth:
    return "high_bandwidth";
  case PlacementIntent::LowLatency:
    return "low_latency";
  case PlacementIntent::ExactTarget:
    return "exact_target";
  }
  return "default";
}

} // namespace vecops::memory
