#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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

} // namespace vecops::bench
