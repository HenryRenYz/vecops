#include <charconv>
#include <algorithm>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "InstructionBench.h"

namespace {

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

using vecops::bench::inst::InstructionCase;
using vecops::bench::inst::MemoryPattern;
using vecops::bench::inst::WorkingSetLevel;

uint64_t g_inner_loops = 10'000'000;

struct CacheInfo {
  uint32_t level = 0;
  uint64_t bytes = 0;
};

uint64_t parse_size(std::string text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }
  uint64_t multiplier = 1;
  if (!text.empty()) {
    const char suffix = text.back();
    if (suffix == 'K' || suffix == 'k') {
      multiplier = 1024;
      text.pop_back();
    } else if (suffix == 'M' || suffix == 'm') {
      multiplier = 1024 * 1024;
      text.pop_back();
    }
  }
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} ? value * multiplier : 0;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string value;
  std::getline(input, value);
  return value;
}

std::vector<CacheInfo> detect_caches() {
  const int cpu = std::max(0, sched_getcpu());
  const auto root = std::filesystem::path("/sys/devices/system/cpu") /
      ("cpu" + std::to_string(cpu)) / "cache";
  std::vector<CacheInfo> caches;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
    if (error || !entry.is_directory() ||
        !entry.path().filename().string().starts_with("index")) {
      continue;
    }
    const std::string type = read_text(entry.path() / "type");
    if (type != "Data" && type != "Unified") continue;
    const uint64_t level = parse_size(read_text(entry.path() / "level"));
    const uint64_t bytes = parse_size(read_text(entry.path() / "size"));
    if (level != 0 && bytes != 0) {
      caches.push_back({static_cast<uint32_t>(level), bytes});
    }
  }
  std::sort(caches.begin(), caches.end(), [](const auto& a, const auto& b) {
    return a.level < b.level;
  });
  return caches;
}

uint64_t floor_power_of_two(uint64_t value) {
  return value == 0 ? 0 : std::bit_floor(value);
}

uint64_t working_set_bytes(WorkingSetLevel level) {
  if (level == WorkingSetLevel::None) return 0;
  const auto caches = detect_caches();
  auto capacity = [&](uint32_t wanted) -> uint64_t {
    for (const auto& cache : caches) {
      if (cache.level == wanted) return cache.bytes;
    }
    return 0;
  };
  if (level == WorkingSetLevel::L1) return floor_power_of_two(capacity(1) / 2);
  if (level == WorkingSetLevel::L2) return floor_power_of_two(capacity(2) / 2);
  if (level == WorkingSetLevel::L3) return floor_power_of_two(capacity(3) / 4);

  uint64_t last = 0;
  for (const auto& cache : caches) last = std::max(last, cache.bytes);
  const uint64_t target = std::max<uint64_t>(256ULL << 20, last * 2);
  return std::bit_ceil(target);
}

bool level_available(WorkingSetLevel level) {
  if (level == WorkingSetLevel::None || level == WorkingSetLevel::BeyondLastCache) {
    return true;
  }
  const uint32_t wanted = level == WorkingSetLevel::L1 ? 1 :
      level == WorkingSetLevel::L2 ? 2 : 3;
  for (const auto& cache : detect_caches()) {
    if (cache.level == wanted) return true;
  }
  return false;
}

struct MemoryBuffer {
  std::vector<std::byte> storage;
  void* aligned = nullptr;
  uint64_t bytes = 0;
};

MemoryBuffer prepare_memory(const InstructionCase& c, uint64_t bytes) {
  bytes = std::max<uint64_t>(bytes, 4096);
  MemoryBuffer result;
  result.storage.resize(static_cast<size_t>(bytes + 128));
  auto address = reinterpret_cast<uintptr_t>(result.storage.data());
  address = (address + 63) & ~uintptr_t(63);
  result.aligned = reinterpret_cast<void*>(address);
  result.bytes = bytes;

  auto* data = static_cast<std::byte*>(result.aligned);
  for (uint64_t offset = 0; offset < bytes; offset += 4096) {
    data[offset] = std::byte{static_cast<unsigned char>(offset / 4096)};
  }

  if (c.memory_pattern == MemoryPattern::PointerChase) {
    const size_t lines = static_cast<size_t>(bytes / 64);
    std::vector<size_t> order(lines);
    std::iota(order.begin(), order.end(), size_t{0});
    std::mt19937_64 random(0x7665636f7073ULL);
    std::shuffle(order.begin(), order.end(), random);
    for (size_t i = 0; i < lines; ++i) {
      const uintptr_t next = reinterpret_cast<uintptr_t>(
          data + order[(i + 1) % lines] * 64);
      auto* line = reinterpret_cast<uintptr_t*>(data + order[i] * 64);
      for (size_t lane = 0; lane < 64 / sizeof(uintptr_t); ++lane) {
        line[lane] = next;
      }
    }
    if (order[0] != 0) {
      std::swap_ranges(data + order[0] * 64, data + order[0] * 64 + 64, data);
      for (size_t i = 0; i < lines; ++i) {
        auto* line = reinterpret_cast<uintptr_t*>(data + i * 64);
        for (size_t lane = 0; lane < 64 / sizeof(uintptr_t); ++lane) {
          uintptr_t value = line[lane];
          if (value == reinterpret_cast<uintptr_t>(data)) {
            line[lane] = reinterpret_cast<uintptr_t>(data + order[0] * 64);
          } else if (value == reinterpret_cast<uintptr_t>(data + order[0] * 64)) {
            line[lane] = reinterpret_cast<uintptr_t>(data);
          }
        }
      }
    }
  }
  return result;
}

std::vector<const InstructionCase*> available_cases() {
  std::vector<const InstructionCase*> result;
  for (const auto& c : vecops::bench::inst::arch_instruction_cases()) {
    if (level_available(c.working_set)) result.push_back(&c);
  }
  return result;
}

bool parse_u64(std::string_view text, uint64_t& value) {
  if (text.empty()) return false;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc{} && result.ptr == end && value != 0;
}

std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

std::string benchmark_name(const InstructionCase& c) {
  return "Instruction/" + std::string(VECOPS_BENCH_ARCH_CODE) + "/" + c.name + "/" +
      vecops::bench::inst::mode_name(c.mode);
}

void print_case_list_json() {
  const auto cases = available_cases();
  std::cout << "{\n  \"arch\": \"" << json_escape(VECOPS_BENCH_ARCH_CODE)
            << "\",\n  \"cases\": [\n";
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = *cases[i];
    const uint64_t memory_bytes = working_set_bytes(c.working_set);
    std::cout << "    {\"benchmark_name\": \"" << json_escape(benchmark_name(c))
              << "\", \"name\": \"" << json_escape(c.name)
              << "\", \"isa\": \"" << json_escape(c.isa)
              << "\", \"mode\": \"" << vecops::bench::inst::mode_name(c.mode)
              << "\", \"category\": \"" << vecops::bench::inst::category_name(c.category)
              << "\", \"description\": \"" << json_escape(c.description)
              << "\", \"sequences_per_loop\": " << c.sequences_per_loop
              << ", \"instructions_per_sequence\": " << c.instructions_per_sequence
              << ", \"dependency_chains\": " << c.dependency_chains
              << ", \"vector_bits\": " << c.vector_bits
              << ", \"memory_pattern\": \""
              << vecops::bench::inst::memory_pattern_name(c.memory_pattern)
              << "\", \"working_set_level\": \""
              << vecops::bench::inst::working_set_level_name(c.working_set)
              << "\", \"working_set_bytes\": " << memory_bytes
              << ", \"bytes_per_sequence\": " << c.bytes_per_sequence
              << ", \"baseline_name\": \"" << json_escape(c.baseline_name) << "\"}";
    if (i + 1 != cases.size()) std::cout << ',';
    std::cout << '\n';
  }
  std::cout << "  ]\n}\n";
}

void register_instruction_benchmarks() {
  for (const auto* case_ptr : available_cases()) {
    const auto& c = *case_ptr;
    const std::string name = benchmark_name(c);
    benchmark::RegisterBenchmark(name.c_str(), [case_ptr](benchmark::State& state) {
      const auto& c = *case_ptr;
      const uint64_t memory_bytes = working_set_bytes(c.working_set);
      auto memory = c.memory_kernel ? prepare_memory(c, memory_bytes) : MemoryBuffer{};
      if (c.memory_kernel) {
        c.memory_kernel(std::min<uint64_t>(g_inner_loops, 1024), memory.aligned, memory.bytes);
      }
      for (auto _ : state) {
        if (c.memory_kernel) {
          c.memory_kernel(g_inner_loops, memory.aligned, memory.bytes);
        } else {
          c.kernel(g_inner_loops);
        }
        benchmark::ClobberMemory();
      }

      const long double sequences = static_cast<long double>(state.iterations()) *
          static_cast<long double>(g_inner_loops) * c.sequences_per_loop;
      const long double instructions = sequences * c.instructions_per_sequence;
      state.SetItemsProcessed(static_cast<int64_t>(instructions));
      state.counters["chains"] = static_cast<double>(c.dependency_chains);
      state.counters["inner_loops"] = static_cast<double>(g_inner_loops);
      state.counters["inst_per_seq"] = static_cast<double>(c.instructions_per_sequence);
      state.counters["seq_per_loop"] = static_cast<double>(c.sequences_per_loop);
      state.counters["sequences"] = static_cast<double>(sequences);
      state.counters["uut_instructions"] = static_cast<double>(instructions);
      state.counters["vector_bits"] = static_cast<double>(c.vector_bits);
      state.counters["bytes_per_sequence"] = static_cast<double>(c.bytes_per_sequence);
      state.counters["working_set_bytes"] = static_cast<double>(memory_bytes);
    })->Unit(benchmark::kNanosecond);
  }
}

} // namespace

int main(int argc, char** argv) {
  bool list_json = false;
  int write_index = 1;
  for (int read_index = 1; read_index < argc; ++read_index) {
    const std::string_view arg(argv[read_index]);
    constexpr std::string_view loops_prefix = "--inst_loops=";
    if (arg == "--inst_list_json") {
      list_json = true;
      continue;
    }
    if (arg.starts_with(loops_prefix)) {
      if (!parse_u64(arg.substr(loops_prefix.size()), g_inner_loops)) {
        std::cerr << "Invalid --inst_loops value: " << arg << '\n';
        return 2;
      }
      continue;
    }
    argv[write_index++] = argv[read_index];
  }
  argc = write_index;

  if (list_json) {
    print_case_list_json();
    return 0;
  }

  register_instruction_benchmarks();
  const bool list_tests = vecops::bench::has_arg(argc, argv, "--benchmark_list_tests");
  const auto default_json = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "instruction", VECOPS_BENCH_ARCH_CODE, "json");
  std::vector<std::string> injected;
  if (!list_tests) {
    injected = vecops::bench::default_google_benchmark_output_args(
        argc, argv, default_json, "json");
  }

  std::vector<char*> args;
  args.reserve(static_cast<size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());

  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  if (!list_tests) {
    vecops::bench::print_default_output_path(argc, argv, default_json);
  }
  return 0;
}
