// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <charconv>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace vecops::bench {

inline std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream os;
  os << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return os.str();
}

inline bool has_arg(int argc, char** argv, const std::string& name) {
  const std::string prefix = name + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == name || arg.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

inline std::filesystem::path default_result_path(
    const char* source_dir,
    const std::string& op_name,
    const std::string& arch,
    const std::string& extension) {
  auto dir = std::filesystem::path(source_dir) / "benchmarks" / "results";
  std::filesystem::create_directories(dir);
  return dir / (op_name + "_" + arch + "_" + timestamp() + "." + extension);
}

inline std::vector<std::string> default_google_benchmark_output_args(
    int argc,
    char** argv,
    const std::filesystem::path& default_output,
    const std::string& format) {
  std::vector<std::string> args;
  if (!has_arg(argc, argv, "--benchmark_out")) {
    args.push_back("--benchmark_out=" + default_output.string());
  }
  if (!has_arg(argc, argv, "--benchmark_out_format")) {
    args.push_back("--benchmark_out_format=" + format);
  }
  return args;
}

inline void print_default_output_path(
    int argc,
    char** argv,
    const std::filesystem::path& path) {
  if (!has_arg(argc, argv, "--benchmark_out")) {
    std::cout << "Benchmark output: " << path.string() << '\n';
  }
}

inline std::optional<int64_t> positive_integer_environment(
    const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::nullopt;
  const std::string_view text{value};
  int64_t parsed = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() || parsed <= 0) {
    throw std::runtime_error(
        std::string{name} + " must be a positive integer");
  }
  return parsed;
}

/**
 * Apply the standard adaptive benchmark policy, or a fixed-work policy for
 * perf/ABBA when VECOPS_BENCH_ITERATIONS is set.  The environment is read at
 * registration time, before Google Benchmark starts calibration; this avoids
 * comparing different iteration counts when an optimization changes runtime.
 */
template <typename RegisteredBenchmark>
RegisteredBenchmark* configure_registered_benchmark(
    RegisteredBenchmark* registered,
    double default_min_time, int default_repetitions) {
  static const auto FixedIterations =
      positive_integer_environment("VECOPS_BENCH_ITERATIONS");
  static const auto ConfiguredRepetitions =
      positive_integer_environment("VECOPS_BENCH_REPETITIONS");
  if (FixedIterations) {
    registered->Iterations(*FixedIterations);
  } else {
    registered->MinTime(default_min_time);
  }
  const int64_t repetitions =
      ConfiguredRepetitions.value_or(default_repetitions);
  if (repetitions > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "VECOPS_BENCH_REPETITIONS exceeds the supported int range");
  }
  registered->Repetitions(static_cast<int>(repetitions));
  return registered;
}

} // namespace vecops::bench
