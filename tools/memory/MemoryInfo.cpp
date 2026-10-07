// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/** @file MemoryInfo.cpp @brief Inspect topology and smoke-test placement. */

#include "vecops/memory/Memory.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace {

using namespace vecops::memory;

struct Options {
  std::optional<std::size_t> allocate_mib;
  PlacementIntent placement = PlacementIntent::Default;
  std::optional<unsigned> target;
  bool touch = false;
  bool use_large_pages = true;
};

[[noreturn]] void usage(const char* program, int status) {
  std::cerr << "usage: " << program
            << " [--allocate-mib N] [--placement default|high-bandwidth|low-latency] [--target OS_NODE]"
               " [--touch] [--no-large-pages]\n";
  std::exit(status);
}

std::size_t parse_size(std::string_view text, const char* option) {
  std::size_t consumed = 0;
  const auto value = std::stoull(std::string(text), &consumed);
  if (consumed != text.size() || value > std::numeric_limits<std::size_t>::max()) {
    std::cerr << option << " is outside the host size range\n";
    std::exit(2);
  }
  return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help")
      usage(argv[0], 0);
    if (option == "--touch") {
      result.touch = true;
      continue;
    }
    if (option == "--no-large-pages") {
      result.use_large_pages = false;
      continue;
    }
    if (index + 1 >= argc)
      usage(argv[0], 2);
    const std::string_view value(argv[++index]);
    if (option == "--allocate-mib") {
      result.allocate_mib = parse_size(value, "--allocate-mib");
    } else if (option == "--target") {
      const auto parsed = parse_size(value, "--target");
      if (parsed > std::numeric_limits<unsigned>::max())
        usage(argv[0], 2);
      result.target = static_cast<unsigned>(parsed);
    } else if (option == "--placement") {
      if (value == "default")
        result.placement = PlacementIntent::Default;
      else if (value == "high-bandwidth")
        result.placement = PlacementIntent::HighBandwidth;
      else if (value == "low-latency")
        result.placement = PlacementIntent::LowLatency;
      else
        usage(argv[0], 2);
    } else {
      usage(argv[0], 2);
    }
  }
  return result;
}

} // namespace

int main(int argc, char** argv) try {
  const auto options = parse_options(argc, argv);
  const auto memory = MemorySystem::discover();
  std::cout << memory.describe();
  std::cout << "current CPU domain: " << memory.current_cpu_domain() << '\n';

  if (!options.allocate_mib.has_value())
    return 0;
  if (options.allocate_mib.value() > std::numeric_limits<std::size_t>::max() / (1024 * 1024)) {
    std::cerr << "--allocate-mib overflows bytes\n";
    return 2;
  }
  AllocationRequest request{
    .bytes = options.allocate_mib.value() * 1024 * 1024,
    .intent = options.target.has_value() ? PlacementIntent::ExactTarget : options.placement,
    .exact_os_numa_id = options.target,
    .fallback = FallbackPolicy::None,
    .alignment = 4096,
    .use_large_pages = options.use_large_pages,
  };
  auto allocation = memory.allocate(request);
  if (options.touch) {
    auto* data = static_cast<std::byte*>(allocation.data());
    for (std::size_t offset = 0; offset < allocation.size(); offset += 4096)
      data[offset] = std::byte{1};
  }
  const auto target = allocation.target().value();
  std::cout << "allocated " << allocation.size() << " bytes on target " << target << " (OS node "
            << memory.topology().memory_targets[target].os_numa_id << "), large-page bytes "
            << allocation.large_page_bytes() << ", regular bytes " << allocation.regular_page_bytes()
            << ", touched=" << (options.touch ? "yes" : "no") << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "vecops-memory-info: " << error.what() << '\n';
  return 1;
}
