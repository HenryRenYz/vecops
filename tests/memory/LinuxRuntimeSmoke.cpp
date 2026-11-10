// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Live-host check: all granted pools, actual placement, exported offset aliases.
#include "vecops/memory/LinuxRuntime.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cerrno>
#include <system_error>
#if defined(__linux__)
#  include <linux/mempolicy.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

using namespace vecops::memory::model;
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
int main(int argc, char** argv) {
  try {
    if (argc > 2)
      throw std::invalid_argument("usage: LinuxRuntimeSmoke [hardware.toml]");
    auto runtime = argc == 2 ? take(open_runtime(argv[1])) : take(discover_linux_runtime());
    auto view = take(runtime->view(take(runtime->current_scope())));
    const auto descriptors = runtime->describe_bindings();
    std::uint64_t bytes = 0;
    std::vector<Allocation> live;
    for (const auto& id : view.allocatable_pools) {
      std::string pool_ids[] = {id};
      auto caps = take(runtime->mapping_capabilities(view, pool_ids));
      auto allocation = take(runtime->allocate(view, {"smoke/" + id,
                                                      runtime->participant(),
                                                      id,
                                                      2 * caps.page_bytes,
                                                      caps.page_bytes,
                                                      BackingMode::shareable_remappable,
                                                      {}}));
      auto ptr = take(allocation.data());
      if (reinterpret_cast<std::uintptr_t>(ptr) % caps.page_bytes)
        throw std::runtime_error("unaligned allocation");
      std::memset(ptr, 61, 2 * caps.page_bytes);
      int observed_node = -1;
#if defined(__linux__)
      const auto descriptor = std::find_if(descriptors.begin(), descriptors.end(),
          [&](const auto& item) { return item.object == id && item.labels.contains("os_numa"); });
      if (descriptor != descriptors.end()) {
        const auto expected_node = std::stoi(descriptor->labels.at("os_numa"));
        for (const auto offset : {std::uint64_t{0}, caps.page_bytes}) {
          if (::syscall(SYS_get_mempolicy, &observed_node, nullptr, 0,
                        static_cast<unsigned char*>(ptr) + offset, MPOL_F_NODE | MPOL_F_ADDR) != 0)
            throw std::system_error(errno, std::generic_category(), "query actual NUMA placement");
          if (observed_node != expected_node)
            throw std::runtime_error("actual NUMA placement mismatch: " + id + " expected=" +
                std::to_string(expected_node) + " observed=" + std::to_string(observed_node));
        }
      }
#endif
      auto reservation =
        take(runtime->reserve_address(allocation.info().address_space, caps.page_bytes, caps.page_bytes));
      auto mapping =
        take(runtime->map(*reservation, take(allocation.export_backing()), {caps.page_bytes, 0, caps.page_bytes}));
      if (*static_cast<unsigned char*>(take(mapping->data())) != 61)
        throw std::runtime_error("export alias mismatch");
      const auto& info = allocation.info();
      bytes += info.mapped_bytes;
      std::cout << "pool=" << id << " mapped_bytes=" << info.mapped_bytes << " quantum=" << info.page_bytes
                << " observed_numa=" << observed_node << " actual_pages=";
      for (auto n : info.actual_page_sizes_bytes)
        std::cout << n << ',';
      std::cout << '\n';
      live.push_back(std::move(allocation));
    }
    take(runtime->refresh_available());
    std::uint64_t committed = 0;
    for (const auto& g : runtime->snapshot(view).groups)
      committed += g.committed_bytes;
    if (committed != bytes || take(runtime->observe(view)).allocations.size() != live.size())
      throw std::runtime_error("unique physical ownership mismatch");
    live.clear();
    for (const auto& g : runtime->snapshot(view).groups)
      if (g.committed_bytes)
        throw std::runtime_error("physical charge did not retire");
    std::cout << "PASS pools=" << view.allocatable_pools.size() << " physical_bytes=" << bytes
              << " executable_agents=" << view.executable_agents.size() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
