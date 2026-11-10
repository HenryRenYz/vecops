// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "Bench.h"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  using namespace memory_bench;
  try {
    const auto options = parse_options(argc, argv);
    if (options.help) { print_help(); return 0; }
    const auto topology = discover();
    Reporter reporter(options, topology);
    if (options.command == "topology") {
      for (const auto& node : topology.nodes)
        std::cout << "  Node " << node.id << " | CPUs: " << (node.cpus.empty() ? "none" : join_cpus(node.cpus))
                  << " | Free: " << human_size(node.free_bytes) << " | Memory allowed: " << (node.memory_allowed ? "yes" : "no") << '\n';
      std::cout << "  Allowed CPUs: " << join_cpus(topology.allowed_cpus) << '\n';
      if (options.engine == "sdma") print_sdma_topology(topology);
    } else if (options.engine == "sdma") run_sdma(options, topology, reporter);
    else if (options.command == "compute") run_compute(options, topology, reporter);
    else if (options.command == "bandwidth") run_bandwidth(options, topology, reporter);
    else if (options.command == "latency") run_latency(options, topology, reporter);
    else run_pingpong(options, topology, reporter);
    return reporter.failed() ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "\n  ERROR: " << e.what() << "\n  Use --help for usage.\n"; return 2;
  }
}
