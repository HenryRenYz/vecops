// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file MemoryInfo.cpp @brief Inspect generic resources and allocate in an opaque pool. */
#include "vecops/memory/Memory.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace vecops::memory::model;
template <class T> T take(Result<T> result) {
  if (!result) {
    const auto& error = std::get<Error>(result.state);
    throw std::runtime_error(error.code + ": " + error.message);
  }
  return std::move(result).value();
}
struct Options {
  std::optional<std::string> hardware, policy, calibration, pool, allocate;
};
[[noreturn]] void usage(const char* program, int status) {
  std::cerr << "usage: " << program
      << " [--hardware FILE] [--policy FILE] [--calibration FILE]"
         " [--pool ID --allocate '64 MiB']\n";
  std::exit(status);
}
Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help") usage(argv[0], 0);
    if (i + 1 == argc) usage(argv[0], 2);
    const std::string value = argv[++i];
    if (key == "--hardware") options.hardware = value;
    else if (key == "--policy") options.policy = value;
    else if (key == "--calibration") options.calibration = value;
    else if (key == "--pool") options.pool = value;
    else if (key == "--allocate") options.allocate = value;
    else usage(argv[0], 2);
  }
  if ((options.policy || options.calibration) && !options.hardware)
    throw std::invalid_argument("policy/calibration require --hardware");
  if (options.allocate.has_value() != options.pool.has_value())
    throw std::invalid_argument("allocation requires both --pool and --allocate");
  return options;
}
double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30); }
}

int main(int argc, char** argv) try {
  const auto options = parse_options(argc, argv);
  auto runtime = options.hardware ? take(open_runtime(*options.hardware, options.policy, options.calibration))
                                  : take(discover_linux_runtime());
  const auto view = take(runtime->view(take(runtime->current_scope())));
  std::cout << std::fixed << std::setprecision(3);
  for (const auto& id : view.executable_agents)
    std::cout << "agent " << id << "\n";
  for (const auto& id : view.allocatable_pools)
    std::cout << "pool " << id << "\n";
  for (const auto& descriptor : runtime->describe_bindings()) {
    std::cout << "binding " << descriptor.object << " backend=" << descriptor.backend;
    for (const auto& [key, value] : descriptor.labels)
      std::cout << ' ' << key << '=' << value;
    std::cout << '\n';
  }
  const auto snapshot = runtime->snapshot(view);
  for (const auto& group : snapshot.groups)
    std::cout << "capacity " << group.capacity_group << " limit=" << gib(group.hard_limit_bytes)
              << " GiB committed=" << gib(group.committed_bytes)
              << " GiB system_available=" << gib(group.system_available_bytes) << " GiB"
              << (group.shared_remaining_bytes ? " shared_remaining=" +
                    std::to_string(gib(*group.shared_remaining_bytes)) + " GiB" : "") << '\n';
  if (options.allocate) {
    const auto bytes = take(parse_memory_size(*options.allocate));
    const std::string pools[] = {*options.pool};
    const auto caps = take(runtime->mapping_capabilities(view, pools));
    auto allocation = take(runtime->allocate(view, {"memory-info", runtime->participant(), *options.pool,
      bytes, caps.page_bytes, BackingMode::shareable_remappable, {}}));
    const auto& info = allocation.info();
    std::cout << "allocation pool=" << *options.pool << " mapped=" << gib(info.mapped_bytes)
              << " GiB quantum=" << info.page_bytes << " B actual_page_sizes=";
    for (const auto page : info.actual_page_sizes_bytes) std::cout << page << " B ";
    std::cout << '\n';
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "vecops-memory-info: " << error.what() << '\n';
  return 1;
}
