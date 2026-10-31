// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "Bench.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <linux/mempolicy.h>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace memory_bench {
namespace {
std::string read_text(const std::string& path) {
  std::ifstream stream(path);
  std::string text;
  std::getline(stream, text);
  return text;
}
std::uint64_t integer(const std::string& text) {
  std::uint64_t result;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::runtime_error("Invalid nonnegative integer: " + text);
  return result;
}
int index(const std::string& text) {
  const auto value = integer(text);
  if (value > 1000000) throw std::runtime_error("CPU/node/count exceeds supported range: " + text);
  return static_cast<int>(value);
}
double real(const std::string& text) {
  std::size_t end;
  const auto value = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(value)) throw std::runtime_error("Invalid number: " + text);
  return value;
}
std::size_t size_value(const std::string& text) {
  const auto pos = text.find_first_not_of("0123456789");
  const auto number = integer(text.substr(0, pos));
  const auto unit = pos == std::string::npos ? "" : text.substr(pos);
  std::size_t multiplier = 1;
  if (unit == "KiB" || unit == "K") multiplier = 1024;
  else if (unit == "MiB" || unit == "M") multiplier = 1024 * 1024;
  else if (unit == "GiB" || unit == "G") multiplier = 1024UL * 1024 * 1024;
  else if (unit != "" && unit != "B") throw std::runtime_error("Unknown binary size suffix: " + unit);
  const auto result = checked_mul(number, multiplier);
  if (!result) throw std::runtime_error("Buffer size must be positive");
  return result;
}
void choice(const std::string& value, std::initializer_list<const char*> allowed, const char* option) {
  for (auto item : allowed) if (value == item) return;
  throw std::runtime_error(std::string("Invalid ") + option + ": " + value);
}
std::string escape(const std::string& text, bool json) {
  std::string result = "\"";
  for (unsigned char ch : text) {
    if (ch == '"') result += json ? "\\\"" : "\"\"";
    else if (ch == '\\' && json) result += "\\\\";
    else if (ch < 32 && json) {
      const char hex[] = "0123456789abcdef";
      result += "\\u00"; result += hex[ch >> 4]; result += hex[ch & 15];
    } else result += static_cast<char>(ch);
  }
  return result + '"';
}
template <typename T> std::string number(T value) {
  std::ostringstream stream; stream << std::setprecision(12) << value; return stream.str();
}
using Fields = std::vector<std::pair<std::string, std::string>>;
Fields fields(const Row& r, const Options& o, const Topology& t) {
  Fields f;
  const auto add = [&](const char* key, const auto& value) { f.emplace_back(key, number(value)); };
  f.emplace_back("schema_version", o.engine == "sdma" ? "3" : "2");
  f.emplace_back("engine", o.engine);
  f.emplace_back("cpu_clock", o.engine == "sdma" ? "CLOCK_THREAD_CPUTIME_ID" : "unavailable");
  f.emplace_back("host", t.hostname); f.emplace_back("cpu_model", t.model);
  f.emplace_back("compiler", __VERSION__);
#ifdef VECOPS_MEMORY_BENCH_SOURCE_SHA
  f.emplace_back("tool_source_sha256", VECOPS_MEMORY_BENCH_SOURCE_SHA);
  f.emplace_back("build_flags", VECOPS_MEMORY_BENCH_FLAGS);
#endif
#ifdef VECOPS_MEMORY_BENCH_REVISION
  f.emplace_back("revision", VECOPS_MEMORY_BENCH_REVISION);
#endif
  f.emplace_back("clock", "CLOCK_MONOTONIC_RAW");
  f.emplace_back("backend", vector_backend()); f.emplace_back("kind", r.kind);
  f.emplace_back("operation", r.operation); f.emplace_back("kernel", r.kernel);
  f.emplace_back("requested_access", r.access); f.emplace_back("size_scope", o.size_scope);
  f.emplace_back("requested_load_policy", o.load_policy);
  f.emplace_back("requested_store_policy", o.store_policy);
  f.emplace_back("requested_cpu_lists", o.cpu); f.emplace_back("cpu_kind", o.cpu_kind);
  f.emplace_back("requested_group_cpus", o.group_cpus);
  f.emplace_back("scope", r.scope); f.emplace_back("cpus", r.cpus);
  f.emplace_back("load_policy", r.load_policy); f.emplace_back("store_policy", r.store_policy);
  f.emplace_back("instructions", r.instructions); f.emplace_back("page_policy", r.page_policy);
  f.emplace_back("placement", r.placement); f.emplace_back("status", r.status);
  f.emplace_back("memory_policy", o.interleave ? "interleave" : "bind");
  f.emplace_back("source_memory_nodes", join_cpus(r.source_nodes));
  f.emplace_back("destination_memory_nodes", join_cpus(r.destination_nodes));
  f.emplace_back("source_page_counts", page_counts_text(r.source_pages));
  f.emplace_back("destination_page_counts", page_counts_text(r.destination_pages));
  add("interleave_bytes", o.interleave ? (o.pages == "huge" ? 2 * 1024 * 1024UL : t.page_bytes) : 0);
  f.emplace_back("sdma_devices", r.sdma_devices);
  f.emplace_back("sdma_channel_ids", r.sdma_channel_ids);
  f.emplace_back("sdma_library", r.sdma_library);
  f.emplace_back("sdma_mode", o.engine == "sdma" ? o.sdma_mode : "");
  add("sdma_channels", r.dma_channels); add("sdma_depth", r.dma_depth);
  add("sdma_batch", r.dma_batch); add("sdma_chunk_bytes", r.dma_chunk);
  add("completed_descriptors", r.dma_descriptors); add("submissions", r.dma_submissions);
  add("drain_ns", r.drain_ns); add("cpu_seconds", r.cpu_seconds); add("setup_seconds", r.setup_seconds);
  f.emplace_back("detail", r.detail);
  add("config", r.config); add("sample", r.sample); add("attempt", r.attempt);
  add("group", r.group); add("source_node", r.src); add("destination_node", r.dst); add("execution_node", r.exec);
  add("threads", r.threads); add("requested_size_bytes", r.requested_bytes);
  add("per_thread_array_bytes", r.array_bytes); add("working_set_bytes", r.working_bytes);
  add("loop_step_bytes", r.step_bytes); add("vector_bytes", r.vector_bytes);
  add("stride_bytes", r.stride_bytes); add("base_page_bytes", t.page_bytes);
  add("read_bytes", r.read_bytes); add("write_bytes", r.write_bytes); add("accesses", r.accesses);
  add("start_ns", r.start_ns); add("end_ns", r.end_ns); add("elapsed_seconds", r.seconds);
  add("start_skew_ns", r.start_skew_ns); add("end_skew_ns", r.end_skew_ns); add("common_ns", r.common_ns);
  add("read_GB_s", r.read_gbs); add("write_GB_s", r.write_gbs); add("total_GB_s", r.gbs);
  add("payload_GB_s", r.payload_gbs); add("ns_per_access", r.ns);
  add("minor_faults", r.minor_faults); add("major_faults", r.major_faults); add("context_switches", r.switches);
  add("seed", o.seed); add("fma", r.fma ? 1 : 0); add("target_seconds", o.seconds);
  add("warmup_seconds", o.warmup);
  add("requested_samples", o.samples); add("max_skew_fraction", o.max_skew);
  add("max_retries", o.retries); add("cpu_absolute", o.absolute ? 1 : 0);
  return f;
}
} // namespace
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::runtime_error("Size/count overflow");
  return a * b;
}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::size_t start = 0;
  do {
    auto end = text.find(delimiter, start);
    auto token = text.substr(start, end == std::string::npos ? end : end - start);
    if (token.empty()) throw std::runtime_error("Empty list item in: " + text);
    result.push_back(token);
    if (end == std::string::npos) break;
    start = end + 1;
  } while (true);
  return result;
}
std::vector<int> cpu_list(const std::string& text) {
  std::vector<int> result;
  std::set<int> seen;
  for (const auto& token : split(text, ',')) {
    const auto range = split(token, '-');
    if (range.size() > 2) throw std::runtime_error("Invalid CPU range: " + token);
    const auto first = index(range[0]), last = range.size() == 1 ? first : index(range[1]);
    if (last < first) throw std::runtime_error("Reversed CPU range: " + token);
    for (int i = first; i <= last; ++i) {
      if (!seen.insert(i).second) throw std::runtime_error("Duplicate CPU: " + number(i));
      result.push_back(i);
    }
  }
  return result;
}
std::string join_cpus(const std::vector<int>& cpus) {
  std::string result;
  for (auto cpu : cpus) { if (!result.empty()) result += ','; result += number(cpu); }
  return result;
}
std::string human_size(std::size_t bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = bytes; int unit = 0;
  while (value >= 1024 && unit < 4) { value /= 1024; ++unit; }
  std::ostringstream stream; stream << std::setprecision(4) << value << ' ' << units[unit]; return stream.str();
}
Operation operation(const std::string& text) {
  if (text == "read") return Operation::Read;
  if (text == "write") return Operation::Write;
  if (text == "copy") return Operation::Copy;
  if (text == "triad") return Operation::Triad;
  throw std::runtime_error("Unknown operation: " + text);
}
Options parse_options(int argc, char** argv) {
  Options o;
  bool explicit_op = false, cpu_policy = false, sdma_option = false;
  int i = 1;
  if (i < argc && argv[i][0] != '-') o.command = argv[i++];
  for (; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") { o.help = true; return o; }
    if (key == "--cpu-absolute") { o.absolute = true; continue; }
    if (key == "--dry-run") { o.dry_run = true; continue; }
    if (key == "--loaded") { o.loaded = true; continue; }
    if (key == "--interleave") { o.interleave = true; continue; }
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--kernel" || key == "--access" || key == "--load-policy" || key == "--store-policy" || key == "--fma") cpu_policy = true;
    if (key == "--op") { o.operations = split(value, ','); explicit_op = true; }
    else if (key == "--engine") o.engine = value;
    else if (key == "--sdma-device") { o.sdma_device = value; sdma_option = true; }
    else if (key == "--sdma-mode") { o.sdma_mode = value; sdma_option = true; }
    else if (key == "--sdma-channels" || key == "--sdma-depth" || key == "--sdma-batch") {
      auto& v = key == "--sdma-channels" ? o.sdma_channels : key == "--sdma-depth" ? o.sdma_depths : o.sdma_batches;
      v.clear(); for (const auto& x : split(value, ',')) v.push_back(index(x)); sdma_option = true;
    }
    else if (key == "--sdma-chunk") {
      o.sdma_chunks.clear(); for (const auto& x : split(value, ',')) o.sdma_chunks.push_back(size_value(x)); sdma_option = true;
    }
    else if (key == "--kernel") o.kernels = split(value, ',');
    else if (key == "--access") o.accesses = split(value, ',');
    else if (key == "--numa") o.numa = value;
    else if (key == "--exec-node") o.exec = value;
    else if (key == "--cpu") o.cpu = value;
    else if (key == "--group-cpus") o.group_cpus = value;
    else if (key == "--cpu-kind") o.cpu_kind = value;
    else if (key == "--size-scope") o.size_scope = value;
    else if (key == "--pages") o.pages = value;
    else if (key == "--load-policy") o.load_policy = value;
    else if (key == "--store-policy") o.store_policy = value;
    else if (key == "--csv") o.csv = value;
    else if (key == "--json") o.json = value;
    else if (key == "--seconds") o.seconds = real(value);
    else if (key == "--warmup") o.warmup = real(value);
    else if (key == "--samples") o.samples = index(value);
    else if (key == "--retries") o.retries = index(value);
    else if (key == "--max-skew") o.max_skew = real(value);
    else if (key == "--seed") o.seed = integer(value);
    else if (key == "--stride") o.stride = size_value(value);
    else if (key == "--latency-cpu") o.latency_cpu = index(value);
    else if (key == "--peer-cpu") o.peer_cpu = index(value);
    else if (key == "--pattern") choice(value, {"chase"}, "--pattern");
    else if (key == "--fma") { choice(value, {"on", "off"}, "--fma"); o.fma = value == "on"; }
    else if (key == "--size") { o.sizes.clear(); for (const auto& item : split(value, ',')) o.sizes.push_back(size_value(item)); }
    else throw std::runtime_error("Unknown option: " + key);
  }
  choice(o.command, {"bandwidth", "latency", "pingpong", "topology"}, "command");
  choice(o.engine, {"cpu", "sdma"}, "--engine");
  choice(o.sdma_mode, {"shared", "exclusive"}, "--sdma-mode");
  if (o.engine == "cpu" && sdma_option) throw std::runtime_error("SDMA options require --engine sdma");
  if (o.engine == "sdma") {
    if (o.command != "bandwidth" && o.command != "topology") throw std::runtime_error("SDMA supports bandwidth and topology only");
    if (cpu_policy || o.loaded) throw std::runtime_error("CPU kernels, access hints, FMA and loaded chase do not apply to SDMA");
    if (!explicit_op) o.operations = {"copy"};
    if (o.operations != std::vector<std::string>{"copy"}) throw std::runtime_error("SDMA supports copy only");
    if (o.command == "bandwidth" && o.sdma_device.empty()) throw std::runtime_error("SDMA requires an explicit --sdma-device ID or per-group IDs separated by '|'");
    for (auto n : o.sdma_channels) if (n < 1 || n > 4096) throw std::runtime_error("SDMA channel count must be in [1,4096]");
    for (auto n : o.sdma_depths) if (n < 1 || n > 1024) throw std::runtime_error("SDMA depth must be in [1,1024]");
    for (auto n : o.sdma_batches) if (n < 1 || n > 1024) throw std::runtime_error("SDMA batch must be in [1,1024]");
    for (auto d : o.sdma_depths) for (auto b : o.sdma_batches) if (d % b) throw std::runtime_error("SDMA batch must divide every selected depth");
    for (auto n : o.sdma_chunks) if (n > UINT32_MAX || n % 4) throw std::runtime_error("SDMA chunk must be FP32-aligned and fit uint32_t");
  }
  choice(o.cpu_kind, {"logical", "physical"}, "--cpu-kind");
  choice(o.size_scope, {"thread-array", "group-array"}, "--size-scope");
  choice(o.pages, {"base", "auto", "huge"}, "--pages");
  if (o.interleave && o.pages == "auto")
    throw std::runtime_error("--interleave requires --pages base or huge; THP has no fixed interleave unit");
  if (o.interleave && o.command != "bandwidth" && o.command != "latency")
    throw std::runtime_error("--interleave supports bandwidth and latency only");
  choice(o.load_policy, {"auto", "temporal", "nt"}, "--load-policy");
  choice(o.store_policy, {"auto", "temporal", "nt"}, "--store-policy");
  for (auto& item : o.operations) operation(item);
  for (auto& item : o.kernels) choice(item, {"scalar16", "scalar32", "scalar64", "vec1", "vec2", "vec4"}, "--kernel");
  for (auto& item : o.accesses) choice(item, {"temporal", "stream"}, "--access");
  if (o.seconds < 0.01 || o.seconds > 3600 || o.warmup < 0 || o.warmup > 3600 ||
      o.samples < 1 || o.samples > 10000 || o.retries > 20 || o.max_skew < 0 || o.max_skew > 1)
    throw std::runtime_error("Require seconds in [0.01,3600], warmup in [0,3600], samples in [1,10000], retries <= 20, max-skew in [0,1]");
  for (auto& item : split(o.cpu, ';')) cpu_list(item);
  if (!o.group_cpus.empty()) {
    if (split(o.cpu, ';').size() != 1) throw std::runtime_error("--group-cpus cannot accompany a CPU sweep");
    for (const auto& item : split(o.group_cpus, '|')) cpu_list(item);
  }
  if (o.loaded && o.command != "latency") throw std::runtime_error("--loaded requires the latency command");
  if (o.command == "latency" && !o.loaded && (o.load_policy != "auto" || o.store_policy != "auto" || o.accesses != std::vector<std::string>{"temporal"}))
    throw std::runtime_error("Pointer chase uses ordinary dependent loads; access policies require --loaded");
  if (!o.csv.empty() && o.csv == o.json) throw std::runtime_error("CSV and JSON paths must differ");
  return o;
}
void print_help() {
  std::cout << R"(vecops-memory-bench - NUMA-aware FP32 bandwidth and dependent-access latency

Usage: vecops-memory-bench [bandwidth|latency|pingpong|topology] [options]

  --engine ENGINE        cpu (default), sdma (copy only; requires sdma-dk SDK)
  --sdma-device IDS      Explicit device per group, separated by '|'; one ID broadcasts
  --sdma-mode MODE       shared (default), exclusive; no automatic fallback
  --sdma-channels LIST   Channels per group (default: 1), comma-separated sweep
  --sdma-depth LIST      In-flight descriptors/channel (default: 32; <= 1024)
  --sdma-batch LIST      Descriptors/submission (default: 8; must divide depth)
  --sdma-chunk LIST      Bytes/descriptor, e.g. 4KiB,64KiB,1MiB (default: 1MiB)
                        Each worker owns private arrays and disjoint channels.
                        Buffer/channel must fit depth chunks; tail chunks are valid.
  --op LIST              read,write,copy,triad (SDMA default: copy; CPU: all)
  --kernel LIST          scalar16,scalar32,scalar64,vec1,vec2,vec4 (default: vec1,vec2,vec4)
  --access LIST          temporal,stream (default: temporal; stream means NT hints)
  --load-policy POLICY   auto,temporal,nt (override access preset)
  --store-policy POLICY  auto,temporal,nt
  --numa LIST            3,4 or source:destination routes such as 3:4,5:6
  --interleave           Cycle every private array's pages over the unique source
                        or destination nodes, separately; sorted, fixed phase zero
  --exec-node SPEC       auto,src,dst or one execution node per route
  --cpu LISTS            Relative offsets, e.g. '0;0-1;0-3;0-7'
  --cpu-kind KIND        logical (default) or physical (one sibling per core)
  --group-cpus LISTS     Per-group offset lists separated by '|'; overrides --cpu
                        for heterogeneous workers; cannot accompany a CPU sweep
  --cpu-absolute         Interpret --cpu as OS CPU IDs (one route only)
  --size LIST            Per-thread per-array bytes, e.g. 16KiB,1MiB,64MiB
  --size-scope SCOPE     thread-array (default) or group-array
  --pages POLICY         base (disable THP), auto (allow THP), huge (strict 2MiB)
  --seconds N            Target duration per sample (default: 0.2)
  --warmup N             Warmup duration (default: 0.05)
  --samples N            Samples per configuration (default: 7)
  --max-skew FRACTION    Allowed start/end spread / target duration (default: .001)
  --retries N            Extra attempts for disturbed samples (default: 2)
  --fma on|off           FP32 triad arithmetic (default: on)
  --csv PATH            Raw sample CSV, including rejected attempts
  --json PATH           Raw sample JSON Lines, including rejected attempts
  --dry-run             Print mappings and configurations without allocating

Latency:
  --pattern chase       Random single-cycle dependent native-pointer loads
  --stride BYTES        Node spacing (default: cache line; multiple of cache line)
  --seed N              Reproducible random permutation (default: 1)
  --loaded              Run FP32 bandwidth traffic alongside the chase
  --latency-cpu N        Reserved chase CPU offset per execution node; required
                        with --loaded, excluded from bandwidth --cpu

Ping-pong:
  --cpu N               Exactly one CPU; one route; memory placed at source node
  --peer-cpu N          Peer offset in destination node (required)
                        With --cpu-absolute, both are OS CPU IDs

Bandwidth counts application reads + writes, NOT physical bus traffic.
Copy payload GB/s counts copied bytes once. Latency reports ns/load; ping-pong
reports ns/round-trip, never an assumed directional one-way latency.
Timing uses a shared deadline, per-thread byte counts and interval envelopes.
All pages are initialized and their NUMA residence is verified before/after.
Sizes must divide exactly into complete kernel steps for each worker.
Pinning, binding and placement failures are fatal; no NUMA fallback is allowed.
Examples:
  vecops-memory-bench bandwidth --numa 0 --cpu '0;0-3' --size 16KiB,64MiB
  vecops-memory-bench bandwidth --op copy --numa 3:4,5:6 --cpu 0-7 --csv copy.csv
  vecops-memory-bench latency --numa 16 --exec-node 3 --cpu 0 --size 256MiB
  vecops-memory-bench latency --loaded --numa 3 --cpu 1-7 --latency-cpu 0
  vecops-memory-bench pingpong --numa 3:4 --cpu 0 --peer-cpu 0
)";
}
const Node& Topology::node(int id) const {
  for (const auto& n : nodes) if (n.id == id) return n;
  throw std::runtime_error("Unknown NUMA node: " + number(id));
}
Topology discover() {
  Topology t;
  t.page_bytes = sysconf(_SC_PAGESIZE);
  const auto online = cpu_list(read_text("/sys/devices/system/cpu/online"));
  const auto count = static_cast<std::size_t>(*std::max_element(online.begin(), online.end()) + 1);
  const auto mask_bytes = CPU_ALLOC_SIZE(count);
  auto* mask = CPU_ALLOC(count);
  if (!mask) throw std::bad_alloc();
  CPU_ZERO_S(mask_bytes, mask);
  const auto result = sched_getaffinity(0, mask_bytes, mask);
  if (result == 0) for (int cpu : online) if (CPU_ISSET_S(cpu, mask_bytes, mask)) t.allowed_cpus.push_back(cpu);
  CPU_FREE(mask);
  if (result != 0) throw std::runtime_error("sched_getaffinity failed");
  std::ifstream status("/proc/self/status"); std::string line, mems;
  while (std::getline(status, line)) if (line.starts_with("Mems_allowed_list:")) mems = line.substr(line.find_last_of("\t ") + 1);
  const auto allowed = cpu_list(mems);
  for (const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/node")) {
    const auto name = entry.path().filename().string();
    if (!name.starts_with("node") || name.size() == 4 || !std::isdigit(name[4])) continue;
    Node n; n.id = index(name.substr(4));
    const auto list = read_text(entry.path().string() + "/cpulist");
    if (!list.empty()) for (auto cpu : cpu_list(list))
      if (std::find(online.begin(), online.end(), cpu) != online.end()) n.cpus.push_back(cpu);
    std::sort(n.cpus.begin(), n.cpus.end());
    n.memory_allowed = std::find(allowed.begin(), allowed.end(), n.id) != allowed.end();
    std::ifstream meminfo(entry.path().string() + "/meminfo");
    while (std::getline(meminfo, line)) {
      auto pos = line.find("MemFree:");
      if (pos != std::string::npos) { std::istringstream s(line.substr(pos + 8)); s >> n.free_bytes; n.free_bytes *= 1024; }
    }
    const auto huge_free = read_text(entry.path().string() + "/hugepages/hugepages-2048kB/free_hugepages");
    if (!huge_free.empty()) n.huge_free_bytes = checked_mul(integer(huge_free), 2 * 1024 * 1024);
    t.nodes.push_back(n);
  }
  std::sort(t.nodes.begin(), t.nodes.end(), [](const Node& a, const Node& b) { return a.id < b.id; });
  if (t.nodes.empty() || t.allowed_cpus.empty()) throw std::runtime_error("No usable Linux NUMA topology");
  char host[256]{}; gethostname(host, sizeof(host) - 1); t.hostname = host;
  std::ifstream cpuinfo("/proc/cpuinfo");
  while (std::getline(cpuinfo, line)) if (line.starts_with("model name") || line.starts_with("CPU implementer")) {
    t.model = line.substr(line.find(':') + 2); break;
  }
  const auto line_size = read_text("/sys/devices/system/cpu/cpu" + number(t.allowed_cpus.front()) + "/cache/index0/coherency_line_size");
  if (!line_size.empty()) t.line_bytes = integer(line_size);
  return t;
}
std::vector<Group> groups(const Options& o, const Topology& t, const std::vector<int>& offsets) {
  const auto routes = split(o.numa, ',');
  const auto per_group = o.group_cpus.empty() ? std::vector<std::string>{} : split(o.group_cpus, '|');
  if (!per_group.empty() && per_group.size() != routes.size())
    throw std::runtime_error("--group-cpus requires one CPU list per group");
  const bool explicit_exec = o.exec != "auto" && o.exec != "src" && o.exec != "dst";
  const auto execs = explicit_exec ? split(o.exec, ',') : std::vector<std::string>{};
  if (explicit_exec && execs.size() != routes.size()) throw std::runtime_error("--exec-node needs one node per route");
  if (o.absolute && routes.size() != 1) throw std::runtime_error("--cpu-absolute requires exactly one route");
  std::vector<Group> result; std::set<int> used;
  for (std::size_t i = 0; i < routes.size(); ++i) {
    const auto pair = split(routes[i], ':');
    if (pair.size() > 2) throw std::runtime_error("Invalid NUMA route: " + routes[i]);
    Group g; g.route = {index(pair[0]), index(pair.back()), pair.size() == 2};
    for (int node : {g.route.src, g.route.dst}) if (!t.node(node).memory_allowed)
      throw std::runtime_error("NUMA memory node excluded by cpuset: " + number(node));
    g.exec = explicit_exec ? index(execs[i]) : o.exec == "src" ? g.route.src : g.route.dst;
    auto candidates = t.node(g.exec).cpus;
    if (candidates.empty()) throw std::runtime_error("Execution node has no CPUs; specify --exec-node: " + number(g.exec));
    if (o.cpu_kind == "physical") {
      std::set<std::string> cores; std::vector<int> physical;
      for (int cpu : candidates) {
        const auto base = "/sys/devices/system/cpu/cpu" + number(cpu) + "/topology/";
        const auto key = read_text(base + "physical_package_id") + ":" + read_text(base + "core_id");
        if (cores.insert(key).second) physical.push_back(cpu);
      }
      candidates = physical;
    }
    for (int offset : per_group.empty() ? offsets : cpu_list(per_group[i])) {
      if (!o.absolute && static_cast<std::size_t>(offset) >= candidates.size()) throw std::runtime_error("CPU offset out of range on node " + number(g.exec));
      const int cpu = o.absolute ? offset : candidates[offset];
      if (std::find(t.node(g.exec).cpus.begin(), t.node(g.exec).cpus.end(), cpu) == t.node(g.exec).cpus.end())
        throw std::runtime_error("Absolute CPU does not belong to execution node: " + number(cpu));
      if (std::find(t.allowed_cpus.begin(), t.allowed_cpus.end(), cpu) == t.allowed_cpus.end()) throw std::runtime_error("CPU excluded by affinity: " + number(cpu));
      if (!used.insert(cpu).second) throw std::runtime_error("Overlapping worker CPU: " + number(cpu));
      g.cpus.push_back(cpu);
    }
    result.push_back(g);
  }
  return result;
}
void pin(int cpu) {
  const auto bytes = CPU_ALLOC_SIZE(cpu + 1);
  auto* mask = CPU_ALLOC(cpu + 1); if (!mask) throw std::bad_alloc();
  CPU_ZERO_S(bytes, mask); CPU_SET_S(cpu, bytes, mask);
  const auto error = pthread_setaffinity_np(pthread_self(), bytes, mask);
  CPU_FREE(mask);
  if (error) throw std::runtime_error("Cannot pin CPU " + number(cpu) + ": " + std::strerror(error));
  if (sched_getcpu() != cpu) throw std::runtime_error("CPU affinity verification failed");
}
std::uint64_t now_ns() {
  timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts)) std::terminate();
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
std::vector<int> memory_nodes(const Options& o, const std::vector<Group>& mapping, std::size_t group, bool source) {
  if (!o.interleave) return {source ? mapping[group].route.src : mapping[group].route.dst};
  std::set<int> nodes;
  for (const auto& g : mapping) nodes.insert(source ? g.route.src : g.route.dst);
  return {nodes.begin(), nodes.end()};
}
std::string page_counts_text(const std::map<int, std::size_t>& counts) {
  std::string result;
  for (const auto& [node, count] : counts) {
    if (!result.empty()) result += '|';
    result += number(node) + ':' + number(count);
  }
  return result;
}
Buffer::Buffer(std::size_t requested, int target, const std::string& pages, std::size_t base_page)
    : Buffer(requested, std::vector<int>{target}, pages, base_page) {}
Buffer::Buffer(std::size_t requested, std::vector<int> targets, const std::string& pages, std::size_t base_page)
    : bytes(requested), page_bytes(pages == "huge" ? 2 * 1024 * 1024 : base_page), nodes(std::move(targets)), page_policy(pages) {
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  if (nodes.empty() || nodes.front() < 0) throw std::runtime_error("Buffer requires nonnegative memory nodes");
  node = nodes.front();
  if (nodes.size() > 1 && pages == "auto") throw std::runtime_error("Interleaving cannot use THP auto pages");
  if (bytes > std::numeric_limits<std::size_t>::max() - page_bytes) throw std::runtime_error("Mapping size overflow");
  mapped_bytes = (bytes + page_bytes - 1) / page_bytes * page_bytes;
  const auto extra = checked_mul(nodes.size() - 1, page_bytes);
  if (mapped_bytes > std::numeric_limits<std::size_t>::max() - extra) throw std::runtime_error("Interleave mapping size overflow");
  std::size_t reservation_bytes = mapped_bytes + extra;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (pages == "huge") flags |= MAP_HUGETLB | (21 << MAP_HUGE_SHIFT);
  data = mmap(nullptr, reservation_bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (data == MAP_FAILED) { data = nullptr; throw std::runtime_error("mmap: " + std::string(std::strerror(errno))); }
  try {
    constexpr auto bits = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(static_cast<std::size_t>(nodes.back()) / bits + 1, 0);
    for (const auto target : nodes) mask[target / bits] |= 1UL << (target % bits);
    // Linux get_nodes() consumes maxnode - 1 bits, as libnuma does when
    // passing a rounded bitmap size plus one (including node zero).
    const auto mode = nodes.size() > 1 ? MPOL_INTERLEAVE : MPOL_BIND;
    if (syscall(SYS_mbind, data, reservation_bytes, mode | MPOL_F_STATIC_NODES,
                mask.data(), mask.size() * bits + 1, 0))
      throw std::runtime_error("mbind nodes " + join_cpus(nodes) + ": " + std::strerror(errno));
    if (pages != "huge" && madvise(data, reservation_bytes, pages == "base" ? MADV_NOHUGEPAGE : MADV_HUGEPAGE))
      throw std::runtime_error("madvise: " + std::string(std::strerror(errno)));
    if (nodes.size() > 1) {
      // VMA/file offsets can change the kernel's initial interleave phase.
      // Probe one page, then retain the slice starting at nodes[0]. Trimming a
      // VMA preserves its page offset. No per-page policies or extra VMAs.
      *static_cast<volatile unsigned char*>(data) = 0;
      void* address = data; int observed = -1;
      if (syscall(SYS_move_pages, 0, 1UL, &address, nullptr, &observed, 0) < 0)
        throw std::runtime_error("Cannot query interleave phase: " + std::string(std::strerror(errno)));
      const auto found = std::find(nodes.begin(), nodes.end(), observed);
      if (found == nodes.end()) throw std::runtime_error("Interleave probe outside requested nodes: " + number(observed));
      const auto phase = static_cast<std::size_t>(found - nodes.begin());
      const auto prefix = ((nodes.size() - phase) % nodes.size()) * page_bytes;
      if (prefix) {
        if (munmap(data, prefix)) throw std::runtime_error("Cannot trim interleave prefix");
        data = static_cast<char*>(data) + prefix;
        reservation_bytes -= prefix;
      }
      const auto suffix = reservation_bytes - mapped_bytes;
      if (suffix) {
        if (munmap(static_cast<char*>(data) + mapped_bytes, suffix)) throw std::runtime_error("Cannot trim interleave suffix");
        reservation_bytes -= suffix;
      }
    }
    // Write the entire mapping: never benchmark the shared anonymous zero page.
    std::memset(data, 0, mapped_bytes);
  } catch (...) { munmap(data, reservation_bytes); data = nullptr; throw; }
}
Buffer::~Buffer() { if (data) munmap(data, mapped_bytes); }
std::map<int, std::size_t> Buffer::page_counts() const {
  std::map<int, std::size_t> counts;
  const auto pages = mapped_bytes / page_bytes;
  for (std::size_t i = 0; i < nodes.size(); ++i)
    counts[nodes[i]] = pages / nodes.size() + (i < pages % nodes.size());
  return counts;
}
void Buffer::verify() const {
  constexpr std::size_t batch = 1024;
  void* addresses[batch]; int status[batch];
  for (std::size_t offset = 0; offset < mapped_bytes;) {
    const auto first_page = offset / page_bytes;
    std::size_t n = 0;
    for (; n < batch && offset < mapped_bytes; ++n, offset += page_bytes)
      addresses[n] = static_cast<char*>(data) + offset;
    if (syscall(SYS_move_pages, 0, n, addresses, nullptr, status, 0) < 0)
      throw std::runtime_error("Cannot verify NUMA residence with move_pages: " + std::string(std::strerror(errno)));
    for (std::size_t i = 0; i < n; ++i) {
      const auto expected = nodes[(first_page + i) % nodes.size()];
      if (status[i] != expected)
        throw std::runtime_error("NUMA residence mismatch at page " + number(first_page + i) +
                                 ": requested " + number(expected) + ", observed " + number(status[i]));
    }
  }
}
Reporter::Reporter(const Options& o, const Topology& t) : options_(o), topology_(t) {
  if (!o.csv.empty()) { csv_.open(o.csv); if (!csv_) throw std::runtime_error("Cannot create CSV: " + o.csv); }
  if (!o.json.empty()) { json_.open(o.json); if (!json_) throw std::runtime_error("Cannot create JSON Lines: " + o.json); }
  if (csv_.is_open()) {
    const auto f = fields(Row{}, o, t);
    for (std::size_t i = 0; i < f.size(); ++i) csv_ << (i ? "," : "") << f[i].first;
    csv_ << '\n';
  }
  std::cout << "\n  vecops memory bench\n  " << std::string(76, '=')
            << "\n  Host: " << t.hostname << " | " << t.model
            << "\n  SIMD: " << vector_backend() << " / " << native_vector_bytes() << " B"
            << " | Base page: " << human_size(t.page_bytes)
            << "\n  Timing: " << o.samples << " x " << o.seconds << " s | Warmup: " << o.warmup
            << " s | Pages: " << o.pages
            << " | Memory policy: " << (o.interleave ? "interleave (fixed phase zero)" : "bind")
            << "\n  Bandwidth: application GB/s (10^9 B/s); not physical bus traffic.\n";
  if (!o.csv.empty()) std::cout << "  CSV: " << o.csv << '\n';
  if (!o.json.empty()) std::cout << "  JSON Lines: " << o.json << '\n';
  std::cout << std::flush;
}
void Reporter::add(const Row& row) {
  const auto f = fields(row, options_, topology_);
  if (csv_.is_open()) {
    for (std::size_t i = 0; i < f.size(); ++i) csv_ << (i ? "," : "") << escape(f[i].second, false);
    csv_ << '\n'; csv_.flush();
    if (!csv_) throw std::runtime_error("CSV write failed");
  }
  if (json_.is_open()) {
    json_ << '{';
    for (std::size_t i = 0; i < f.size(); ++i) json_ << (i ? "," : "") << escape(f[i].first, true) << ':' << escape(f[i].second, true);
    json_ << "}\n"; json_.flush();
    if (!json_) throw std::runtime_error("JSON Lines write failed");
  }
}
void Reporter::summary(const std::vector<Row>& rows, const std::string& label) {
  std::map<std::string, std::vector<const Row*>> buckets;
  for (const auto& r : rows) if (r.status == "ok") buckets[r.scope].push_back(&r);
  std::cout << "\n  " << label << '\n';
  if (buckets.empty()) {
    std::cout << "  No valid samples";
    if (!rows.empty()) std::cout << ": " << rows.back().status << " " << rows.back().detail;
    std::cout << '\n'; failed_ = true; return;
  }
  const bool latency = rows.front().kind != "bandwidth";
  std::cout << "  " << std::left << std::setw(17) << "Scope" << std::right << std::setw(8) << "Valid"
            << std::setw(13) << (latency ? "Median ns" : "Median GB/s") << std::setw(12) << "Min"
            << std::setw(12) << "Max" << std::setw(12) << "Spread %" << '\n'
            << "  " << std::string(74, '-') << '\n';
  for (const auto& [scope, samples] : buckets) {
    std::vector<double> values;
    for (auto* r : samples) values.push_back(latency ? r->ns : r->gbs);
    std::sort(values.begin(), values.end());
    const auto n = values.size(); const double median = (values[(n - 1) / 2] + values[n / 2]) / 2;
    const double spread = median ? 100 * (values.back() - values.front()) / median : 0;
    std::cout << "  " << std::left << std::setw(17) << scope << std::right
              << std::setw(8) << (number(n) + "/" + number(options_.samples)) << std::fixed << std::setprecision(3)
              << std::setw(13) << median << std::setw(12) << values.front() << std::setw(12) << values.back()
              << std::setw(12) << spread << '\n';
    if (n < static_cast<std::size_t>(options_.samples)) failed_ = true;
  }
  std::size_t rejected = 0; for (const auto& row : rows) if (row.status != "ok") ++rejected;
  if (rejected) std::cout << "  Rejected rows: " << rejected << " (retained in raw output)\n";
  std::cout << std::flush;
}
} // namespace memory_bench
