// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "Kernels.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace memory_bench {
struct Node {
  int id;
  std::vector<int> cpus;
  std::uint64_t free_bytes = 0;
  std::uint64_t huge_free_bytes = 0;
  bool memory_allowed = false;
};
struct Topology {
  std::vector<Node> nodes;
  std::vector<int> allowed_cpus;
  std::size_t page_bytes = 0, line_bytes = 64;
  std::string hostname, model;
  const Node& node(int id) const;
};
struct Route { int src, dst; bool paired = false; };
struct Options {
  std::string command = "bandwidth";
  std::vector<std::string> operations{"read", "write", "copy", "triad"};
  std::vector<std::string> kernels{"vec1", "vec2", "vec4"};
  std::vector<std::string> accesses{"temporal"};
  std::string numa = "0", exec = "auto", cpu = "0", cpu_kind = "logical";
  std::string group_cpus;
  // Compute shapes describe one global operation, partitioned by output rows.
  std::vector<std::size_t> shape;
  std::vector<int> input_nodes, output_nodes;
  int workspace_node = -1;
  bool ln_affine = true;
  double ln_eps = 1e-5;
  std::string size_scope = "thread-array", pages = "base";
  std::string load_policy = "auto", store_policy = "auto";
  std::string csv, json;
  std::string engine = "cpu", sdma_device, sdma_mode = "shared";
  std::vector<int> sdma_channels{1}, sdma_depths{32}, sdma_batches{8};
  std::vector<std::size_t> sdma_chunks{1024 * 1024};
  std::vector<std::size_t> sizes{64 * 1024 * 1024};
  double seconds = 0.2, warmup = 0.05, max_skew = 0.001;
  int samples = 7, retries = 2;
  std::uint64_t seed = 1;
  bool fma = true, absolute = false, dry_run = false, help = false;
  bool loaded = false, interleave = false;
  int latency_cpu = -1, peer_cpu = -1;
  std::size_t stride = 0;
};
struct Group {
  Route route;
  int exec;
  std::vector<int> cpus;
};
struct Row {
  std::string kind, operation, kernel, access, scope, cpus, status = "ok", detail;
  std::string load_policy, store_policy, instructions, page_policy, placement;
  std::vector<int> source_nodes, destination_nodes;
  std::map<int, std::size_t> source_pages, destination_pages;
  int config = 0, sample = 0, attempt = 0, group = -1, src = -1, dst = -1, exec = -1;
  std::size_t threads = 0, requested_bytes = 0, array_bytes = 0, working_bytes = 0;
  std::size_t step_bytes = 0, vector_bytes = 0, stride_bytes = 0;
  std::uint64_t read_bytes = 0, write_bytes = 0, accesses = 0;
  std::uint64_t start_ns = 0, end_ns = 0, start_skew_ns = 0, end_skew_ns = 0;
  std::uint64_t common_ns = 0, minor_faults = 0, major_faults = 0, switches = 0;
  double seconds = 0, read_gbs = 0, write_gbs = 0, gbs = 0, payload_gbs = 0, ns = 0;
  bool fma = true;
  std::string shape, dtype, compute_backend, verification, workspace_nodes, workspace_pages;
  std::size_t row_begin = 0, row_end = 0, workspace_bytes = 0;
  std::uint64_t iterations = 0, tensor_input_bytes = 0, tensor_output_bytes = 0;
  double warmup_actual_seconds = 0, calibration_seconds = 0;
  double operation_seconds = 0, gflops = 0, gelements = 0, logical_gbs = 0;
  std::string sdma_devices, sdma_channel_ids, sdma_library;
  std::size_t dma_channels = 0, dma_depth = 0, dma_batch = 0, dma_chunk = 0;
  std::uint64_t dma_descriptors = 0, dma_submissions = 0, drain_ns = 0;
  double cpu_seconds = 0, setup_seconds = 0;
};
struct Buffer {
  void* data = nullptr;
  std::size_t bytes = 0, mapped_bytes = 0, page_bytes = 0;
  int node = -1;
  std::vector<int> nodes;
  std::string page_policy;
  Buffer(std::size_t bytes, int node, const std::string& pages, std::size_t base_page);
  // Every array starts at nodes[0]; subsequent pages cycle through sorted nodes.
  // Interleaving requires explicit base or hugetlb pages (never THP auto).
  Buffer(std::size_t bytes, std::vector<int> nodes, const std::string& pages, std::size_t base_page);
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  // Query every resident page in bounded batches, without migrating it.
  void verify() const;
  std::map<int, std::size_t> page_counts() const;
};
class Reporter {
 public:
  Reporter(const Options&, const Topology&);
  void add(const Row&);
  void summary(const std::vector<Row>&, const std::string& label);
  bool failed() const { return failed_; }
 private:
  const Options& options_;
  const Topology& topology_;
  std::ofstream csv_, json_;
  bool failed_ = false;
};
Options parse_options(int argc, char** argv);
void print_help();
Topology discover();
std::vector<std::string> split(const std::string&, char delimiter);
std::vector<int> cpu_list(const std::string&);
std::string join_cpus(const std::vector<int>&);
std::vector<Group> groups(const Options&, const Topology&, const std::vector<int>& offsets);
std::vector<int> memory_nodes(const Options&, const std::vector<Group>&, std::size_t group, bool source);
std::string page_counts_text(const std::map<int, std::size_t>&);
void pin(int cpu);
std::uint64_t now_ns();
std::size_t checked_mul(std::size_t, std::size_t);
std::string human_size(std::size_t);
Operation operation(const std::string&);
void run_compute(const Options&, const Topology&, Reporter&);
void run_bandwidth(const Options&, const Topology&, Reporter&);
void run_latency(const Options&, const Topology&, Reporter&);
void run_pingpong(const Options&, const Topology&, Reporter&);
void print_sdma_topology(const Topology&);
void run_sdma(const Options&, const Topology&, Reporter&);
} // namespace memory_bench
