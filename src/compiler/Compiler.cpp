/**
 * @file Compiler.cpp
 * @brief POSIX subprocess implementation of the CMake kernel compiler.
 *
 * Request validation and project generation happen in the caller process.
 * Configure and build/install commands are launched with `posix_spawnp`
 * without shell parsing; their combined output is retained in per-stage log
 * files and copied into `BuildResult`.
 */

#include "vecops/compiler/Compiler.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <spawn.h>
#include <sstream>
#include <system_error>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace vecops::compiler {
namespace {

namespace fs = std::filesystem;

std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string cmake_quote(std::string value) {
  std::string quoted = "\"";
  for (const char character : value) {
    switch (character) {
    case '\\':
      quoted += "\\\\";
      break;
    case '"':
      quoted += "\\\"";
      break;
    case '$':
      quoted += "\\$";
      break;
    case ';':
      quoted += "\\;";
      break;
    case '\n':
      quoted += "\\n";
      break;
    case '\r':
      quoted += "\\r";
      break;
    default:
      quoted += character;
      break;
    }
  }
  quoted += '"';
  return quoted;
}

void append_path_list(std::ostringstream& cmake, const char* keyword, const std::vector<fs::path>& paths) {
  if (paths.empty()) {
    return;
  }
  cmake << "    " << keyword << '\n';
  for (const auto& path : paths) {
    cmake << "        " << cmake_quote(fs::absolute(path).lexically_normal().string()) << '\n';
  }
}

void append_string_list(std::ostringstream& cmake, const char* keyword, const std::vector<std::string>& values) {
  if (values.empty()) {
    return;
  }
  cmake << "    " << keyword << '\n';
  for (const auto& value : values) {
    cmake << "        " << cmake_quote(value) << '\n';
  }
}

bool is_source_tree(const fs::path& path) {
  return fs::is_regular_file(path / "CMakeLists.txt") && fs::is_directory(path / "include" / "vecops") &&
         fs::is_regular_file(path / "cmake" / "VecopsKernelTargets.cmake");
}

std::optional<fs::path> package_config_directory(const fs::path& path) {
  if (fs::is_regular_file(path / "VecopsConfig.cmake")) {
    return path;
  }
  const auto lib_config = path / "lib" / "cmake" / "Vecops";
  if (fs::is_regular_file(lib_config / "VecopsConfig.cmake")) {
    return lib_config;
  }
  const auto lib64_config = path / "lib64" / "cmake" / "Vecops";
  if (fs::is_regular_file(lib64_config / "VecopsConfig.cmake")) {
    return lib64_config;
  }
  return std::nullopt;
}

std::optional<std::string> validate(const KernelBuildRequest& request, SdkLayout& resolved_layout,
                                    fs::path& resolved_sdk) {
  static const std::regex target_pattern{"^[A-Za-z_][A-Za-z0-9_.+-]*$"};
  if (!std::regex_match(request.target_name, target_pattern)) {
    return "target_name must be a non-empty CMake-safe identifier";
  }
  if (request.output_name && !std::regex_match(*request.output_name, target_pattern)) {
    return "output_name must be a CMake-safe filename stem";
  }
  if (request.toolchain.cmake_program.empty()) {
    return "toolchain.cmake_program is required";
  }
  if (request.sources.empty()) {
    return "at least one kernel source is required";
  }
  for (const auto& source : request.sources) {
    if (!fs::is_regular_file(source)) {
      return "kernel source does not exist: " + source.string();
    }
  }
  for (const auto& directory : request.include_directories) {
    if (!fs::is_directory(directory)) {
      return "include directory does not exist: " + directory.string();
    }
  }
  for (const auto& directory : request.link_directories) {
    if (!fs::is_directory(directory)) {
      return "link directory does not exist: " + directory.string();
    }
  }
  for (const auto& [name, value] : request.toolchain.environment) {
    (void)value;
    if (name.empty() || name.find('=') != std::string::npos) {
      return "toolchain environment contains an invalid variable name";
    }
  }
  if (request.sdk.path.empty()) {
    return "sdk.path is required";
  }
  if (request.generated_source_directory.empty() || request.build_directory.empty() ||
      request.artifact_directory.empty()) {
    return "generated, build, and artifact directories are required";
  }

  std::error_code error;
  const auto generated = fs::weakly_canonical(fs::absolute(request.generated_source_directory), error);
  error.clear();
  const auto build = fs::weakly_canonical(fs::absolute(request.build_directory), error);
  error.clear();
  const auto artifact = fs::weakly_canonical(fs::absolute(request.artifact_directory), error);
  if (generated == build || generated == artifact || build == artifact) {
    return "generated, build, and artifact directories must be distinct";
  }
  if (!request.reuse_generated_source_directory && fs::exists(request.generated_source_directory) &&
      !fs::is_empty(request.generated_source_directory)) {
    return "generated source directory must be absent or empty unless "
           "reuse_generated_source_directory is set";
  }
  if (!request.reuse_build_directory && fs::exists(request.build_directory) && !fs::is_empty(request.build_directory)) {
    return "build directory must be absent or empty unless "
           "reuse_build_directory is set";
  }
  if (fs::exists(request.artifact_directory) && !fs::is_empty(request.artifact_directory)) {
    return "artifact directory must be absent or empty";
  }

  resolved_sdk = fs::absolute(request.sdk.path);
  resolved_layout = request.sdk.layout;
  if (resolved_layout == SdkLayout::Auto) {
    resolved_layout = is_source_tree(resolved_sdk) ? SdkLayout::SourceTree : SdkLayout::Package;
  }
  if (resolved_layout == SdkLayout::SourceTree) {
    if (!is_source_tree(resolved_sdk)) {
      return "sdk.path is not a Vecops source tree: " + resolved_sdk.string();
    }
  } else if (!package_config_directory(resolved_sdk)) {
    return "sdk.path does not contain a Vecops package config: " + resolved_sdk.string();
  }
  return std::nullopt;
}

std::string generated_task(const KernelBuildRequest& request) {
  std::ostringstream cmake;
  cmake << "vecops_add_kernel_library(\n"
           "    NO_INSTALL\n"
           "    NAME "
        << cmake_quote(request.target_name) << '\n';
  if (request.output_name) {
    cmake << "    OUTPUT_NAME " << cmake_quote(*request.output_name) << '\n';
  }
  if (request.target_arch) {
    cmake << "    TARGET_ARCH " << cmake_quote(*request.target_arch) << '\n';
  }
  append_path_list(cmake, "SOURCES", request.sources);
  append_path_list(cmake, "INCLUDE_DIRECTORIES", request.include_directories);
  append_string_list(cmake, "COMPILE_DEFINITIONS", request.compile_definitions);
  append_string_list(cmake, "COMPILE_OPTIONS", request.compile_options);
  append_path_list(cmake, "LINK_DIRECTORIES", request.link_directories);
  append_string_list(cmake, "LINK_LIBRARIES", request.link_libraries);
  append_string_list(cmake, "LINK_OPTIONS", request.link_options);
  cmake << ")\n"
           "set_target_properties("
        << request.target_name << " PROPERTIES\n"
        << "    LIBRARY_OUTPUT_DIRECTORY " << cmake_quote(fs::absolute(request.artifact_directory).string()) << '\n'
        << "    RUNTIME_OUTPUT_DIRECTORY " << cmake_quote(fs::absolute(request.artifact_directory).string()) << '\n'
        << ")\n";
  return cmake.str();
}

std::string generated_batch_project(const KernelBuildBatchRequest& request, SdkLayout layout,
                                    const fs::path& sdk_path) {
  std::ostringstream cmake;
  cmake << "cmake_minimum_required(VERSION 3.22)\n"
           "project(VecopsGeneratedKernelBatch LANGUAGES C CXX)\n\n";
  if (layout == SdkLayout::SourceTree) {
    cmake << "set(VECOPS_BUILD_TESTING OFF CACHE BOOL \"\" FORCE)\n"
             "set(VECOPS_BUILD_BENCHMARKS OFF CACHE BOOL \"\" FORCE)\n"
             "set(VECOPS_BUILD_COMPILER OFF CACHE BOOL \"\" FORCE)\n"
             "add_subdirectory("
          << cmake_quote(sdk_path.string())
          << " vecops-sdk EXCLUDE_FROM_ALL)\n"
             "include("
          << cmake_quote((sdk_path / "cmake" / "VecopsKernelTargets.cmake").string()) << ")\n\n";
  } else {
    cmake << "find_package(Vecops CONFIG REQUIRED)\n\n";
  }
  std::vector<std::string> targets;
  targets.reserve(request.tasks.size());
  for (std::size_t index = 0; index < request.tasks.size(); ++index) {
    const auto task_source = fs::absolute(request.generated_source_directory / "tasks" / std::to_string(index));
    const auto task_binary = fs::absolute(request.build_directory / "tasks" / std::to_string(index));
    cmake << "add_subdirectory(" << cmake_quote(task_source.string()) << ' ' << cmake_quote(task_binary.string())
          << ")\n";
    targets.push_back(request.tasks[index].target_name);
  }
  cmake << "add_custom_target(vecops_batch_all DEPENDS";
  for (const auto& target : targets)
    cmake << ' ' << target;
  cmake << ")\n";
  return cmake.str();
}

std::string os_error(const std::string& prefix, int error) {
  return prefix + ": " + std::error_code(error, std::generic_category()).message();
}

CommandResult run_command(const std::vector<std::string>& arguments, const fs::path& log_path,
                          const std::map<std::string, std::string>& environment) {
  CommandResult result;
  result.arguments = arguments;
  result.log_path = log_path;

  const int log_fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (log_fd < 0) {
    result.output = os_error("cannot open subprocess log", errno);
    return result;
  }

  posix_spawn_file_actions_t actions;
  int spawn_error = ::posix_spawn_file_actions_init(&actions);
  const bool actions_initialized = spawn_error == 0;
  if (spawn_error == 0) {
    spawn_error = ::posix_spawn_file_actions_adddup2(&actions, log_fd, STDOUT_FILENO);
  }
  if (spawn_error == 0) {
    spawn_error = ::posix_spawn_file_actions_adddup2(&actions, log_fd, STDERR_FILENO);
  }
  if (spawn_error == 0) {
    spawn_error = ::posix_spawn_file_actions_addclose(&actions, log_fd);
  }
  if (spawn_error != 0) {
    if (actions_initialized) {
      ::posix_spawn_file_actions_destroy(&actions);
    }
    ::close(log_fd);
    result.output = os_error("cannot prepare subprocess", spawn_error);
    return result;
  }

  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (const auto& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  argv.push_back(nullptr);

  std::map<std::string, std::string> merged_environment;
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    const std::string value(*entry);
    const auto separator = value.find('=');
    if (separator != std::string::npos) {
      merged_environment[value.substr(0, separator)] = value.substr(separator + 1);
    }
  }
  for (const auto& [name, value] : environment) {
    merged_environment[name] = value;
  }
  std::vector<std::string> environment_storage;
  environment_storage.reserve(merged_environment.size());
  for (const auto& [name, value] : merged_environment) {
    environment_storage.push_back(name + "=" + value);
  }
  std::vector<char*> envp;
  envp.reserve(environment_storage.size() + 1);
  for (auto& entry : environment_storage) {
    envp.push_back(entry.data());
  }
  envp.push_back(nullptr);

  pid_t child = -1;
  spawn_error = ::posix_spawnp(&child, arguments.front().c_str(), &actions, nullptr, argv.data(), envp.data());
  ::posix_spawn_file_actions_destroy(&actions);
  ::close(log_fd);
  if (spawn_error != 0) {
    result.output = os_error("failed to launch command", spawn_error);
    result.exit_code = spawn_error == ENOENT ? 127 : 126;
    return result;
  }

  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      result.output = os_error("waitpid failed", errno);
      return result;
    }
  }
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  result.output = read_file(log_path);
  if (result.output.empty() && result.exit_code == 127) {
    result.output = "executable not found: " + arguments.front();
  } else if (result.output.empty() && result.exit_code == 126) {
    result.output = "failed to launch command: " + arguments.front();
  }
  return result;
}

void append_cache_argument(std::vector<std::string>& command, const std::string& name, const std::string& value) {
  command.push_back("-D" + name + "=" + value);
}

std::vector<fs::path> collect_artifacts(const fs::path& directory) {
  std::vector<fs::path> result;
  std::error_code error;
  for (fs::recursive_directory_iterator iterator(directory, error), end; !error && iterator != end;
       iterator.increment(error)) {
    if (iterator->is_regular_file(error)) {
      result.push_back(iterator->path());
    }
  }
  return result;
}

std::optional<fs::path> find_kernel_library(const std::vector<fs::path>& files, const KernelBuildRequest& request) {
  const std::string output_name = request.output_name.value_or(request.target_name);
  for (const auto& file : files) {
    const auto extension = file.extension().string();
    const auto filename = file.filename().string();
    if ((extension == ".so" || extension == ".dylib" || extension == ".dll") &&
        filename.find(output_name) != std::string::npos) {
      return file;
    }
  }
  return std::nullopt;
}

} // namespace

BuildResult Compiler::compile(const KernelBuildRequest& request) const {
  KernelBuildBatchRequest batch;
  batch.tasks.push_back(request);
  batch.generated_source_directory = request.generated_source_directory;
  batch.build_directory = request.build_directory;
  auto result = compile_batch(batch);
  if (!result.tasks.empty())
    return std::move(result.tasks.front());
  BuildResult failed;
  failed.stage = result.stage;
  failed.error = std::move(result.error);
  failed.configure = std::move(result.configure);
  failed.build_and_install = std::move(result.build);
  failed.artifact_directory = request.artifact_directory;
  return failed;
}

BuildBatchResult Compiler::compile_batch(const KernelBuildBatchRequest& request) const {
  BuildBatchResult result;
  result.stage = BuildStage::Validate;
  result.tasks.resize(request.tasks.size());
  for (std::size_t index = 0; index < request.tasks.size(); ++index)
    result.tasks[index].artifact_directory = request.tasks[index].artifact_directory;

  try {
    if (request.tasks.empty()) {
      result.error = "a build batch requires at least one task";
      return result;
    }
    if (request.generated_source_directory.empty() || request.build_directory.empty()) {
      result.error = "batch generated and build directories are required";
      return result;
    }
    if (fs::absolute(request.generated_source_directory).lexically_normal() ==
        fs::absolute(request.build_directory).lexically_normal()) {
      result.error = "batch generated and build directories must be distinct";
      return result;
    }
    const bool reuse_generated = std::all_of(request.tasks.begin(), request.tasks.end(),
                                             [](const auto& task) { return task.reuse_generated_source_directory; });
    const bool reuse_build = std::all_of(request.tasks.begin(), request.tasks.end(),
                                         [](const auto& task) { return task.reuse_build_directory; });
    if (!reuse_generated && fs::exists(request.generated_source_directory) &&
        !fs::is_empty(request.generated_source_directory)) {
      result.error = "batch generated source directory must be absent or empty unless every task enables reuse";
      return result;
    }
    if (!reuse_build && fs::exists(request.build_directory) && !fs::is_empty(request.build_directory)) {
      result.error = "batch build directory must be absent or empty unless every task enables reuse";
      return result;
    }

    const auto& common = request.tasks.front();
    SdkLayout sdk_layout = SdkLayout::Auto;
    fs::path sdk_path;
    std::set<std::string, std::less<>> target_names;
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
      auto task = request.tasks[index];
      task.generated_source_directory = request.generated_source_directory / "tasks" / std::to_string(index);
      task.build_directory = request.build_directory / "tasks" / std::to_string(index);
      SdkLayout task_layout = SdkLayout::Auto;
      fs::path task_sdk;
      if (const auto error = validate(task, task_layout, task_sdk)) {
        result.tasks[index].error = *error;
        result.error = "invalid build task " + std::to_string(index) + ": " + *error;
        return result;
      }
      if (!target_names.insert(task.target_name).second) {
        result.error = "batch target names must be unique: " + task.target_name;
        return result;
      }
      if (index == 0) {
        sdk_layout = task_layout;
        sdk_path = task_sdk;
      } else if (task_sdk != sdk_path || task_layout != sdk_layout ||
                 task.toolchain.cmake_program != common.toolchain.cmake_program ||
                 task.toolchain.c_compiler != common.toolchain.c_compiler ||
                 task.toolchain.cxx_compiler != common.toolchain.cxx_compiler ||
                 task.toolchain.toolchain_file != common.toolchain.toolchain_file ||
                 task.toolchain.generator != common.toolchain.generator ||
                 task.toolchain.build_type != common.toolchain.build_type ||
                 task.toolchain.parallel_jobs != common.toolchain.parallel_jobs ||
                 task.toolchain.environment != common.toolchain.environment ||
                 task.cmake_cache_variables != common.cmake_cache_variables) {
        result.error = "all tasks in one batch must share one SDK, toolchain, environment, and CMake cache";
        return result;
      }
    }

    result.stage = BuildStage::Generate;
    fs::create_directories(request.generated_source_directory / "tasks");
    fs::create_directories(request.build_directory);
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
      const auto task_directory = request.generated_source_directory / "tasks" / std::to_string(index);
      fs::create_directories(task_directory);
      fs::create_directories(request.tasks[index].artifact_directory);
      const auto task_file = task_directory / "CMakeLists.txt";
      std::ofstream task_project(task_file, std::ios::binary | std::ios::trunc);
      if (!task_project) {
        result.error = "cannot write generated task: " + task_file.string();
        return result;
      }
      task_project << generated_task(request.tasks[index]);
      task_project.close();
      if (!task_project) {
        result.error = "failed while writing generated task: " + task_file.string();
        return result;
      }
    }

    const auto project_file = request.generated_source_directory / "CMakeLists.txt";
    std::ofstream project(project_file, std::ios::binary | std::ios::trunc);
    if (!project) {
      result.error = "cannot write generated batch project: " + project_file.string();
      return result;
    }
    project << generated_batch_project(request, sdk_layout, sdk_path);
    project.close();
    if (!project) {
      result.error = "failed while writing generated batch project: " + project_file.string();
      return result;
    }

    std::vector<std::string> configure{common.toolchain.cmake_program.string(), "-S",
                                       fs::absolute(request.generated_source_directory).string(), "-B",
                                       fs::absolute(request.build_directory).string()};
    if (common.toolchain.generator) {
      configure.emplace_back("-G");
      configure.push_back(*common.toolchain.generator);
    }
    append_cache_argument(configure, "CMAKE_BUILD_TYPE", common.toolchain.build_type);
    if (common.toolchain.c_compiler)
      append_cache_argument(configure, "CMAKE_C_COMPILER", common.toolchain.c_compiler->string());
    if (common.toolchain.cxx_compiler)
      append_cache_argument(configure, "CMAKE_CXX_COMPILER", common.toolchain.cxx_compiler->string());
    if (common.toolchain.toolchain_file)
      append_cache_argument(configure, "CMAKE_TOOLCHAIN_FILE", common.toolchain.toolchain_file->string());
    if (sdk_layout == SdkLayout::Package)
      append_cache_argument(configure, "Vecops_DIR", package_config_directory(sdk_path)->string());
    for (const auto& [name, value] : common.cmake_cache_variables)
      append_cache_argument(configure, name, value);

    result.stage = BuildStage::Configure;
    result.configure =
      run_command(configure, request.build_directory / "vecops-configure.log", common.toolchain.environment);
    if (result.configure->exit_code != 0) {
      result.error = "CMake batch configure failed";
      for (auto& task : result.tasks) {
        task.stage = result.stage;
        task.configure = result.configure;
        task.error = result.error;
      }
      return result;
    }

    std::vector<std::string> build{common.toolchain.cmake_program.string(),
                                   "--build",
                                   fs::absolute(request.build_directory).string(),
                                   "--target",
                                   "vecops_batch_all",
                                   "--config",
                                   common.toolchain.build_type};
    if (common.toolchain.parallel_jobs != 0) {
      build.emplace_back("--parallel");
      build.push_back(std::to_string(common.toolchain.parallel_jobs));
    }
    result.stage = BuildStage::BuildAndInstall;
    result.build = run_command(build, request.build_directory / "vecops-build.log", common.toolchain.environment);
    if (result.build->exit_code != 0)
      result.error = "CMake batch build failed";

    bool all_tasks_succeeded = result.build->exit_code == 0;
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
      auto& task = result.tasks[index];
      task.stage = result.stage;
      task.configure = result.configure;
      task.build_and_install = result.build;
      task.artifact_files = collect_artifacts(request.tasks[index].artifact_directory);
      task.kernel_library = find_kernel_library(task.artifact_files, request.tasks[index]);
      if (task.kernel_library) {
        task.success = true;
        task.stage = BuildStage::Complete;
      } else {
        task.error = result.build->exit_code == 0 ? "CMake succeeded but the task library was not found"
                                                  : "CMake batch build failed before the task library was produced";
        all_tasks_succeeded = false;
      }
    }
    result.success = all_tasks_succeeded;
    if (result.success)
      result.stage = BuildStage::Complete;
    return result;
  } catch (const fs::filesystem_error& error) {
    result.error = error.what();
    return result;
  } catch (const std::exception& error) {
    result.error = error.what();
    return result;
  }
}

std::string Compiler::batch_key() const {
  if (!kernel_config_)
    return {};
  std::ostringstream output;
  auto append = [&](std::string_view value) { output << value.size() << ':' << value << ';'; };
  auto append_path = [&](const fs::path& value) { append(fs::absolute(value).lexically_normal().string()); };
  auto append_optional_path = [&](const std::optional<fs::path>& value) {
    append(value ? fs::absolute(*value).lexically_normal().string() : std::string{});
  };
  const auto& config = *kernel_config_;
  output << static_cast<int>(config.sdk.layout) << ';';
  append_path(config.sdk.path);
  append_path(config.work_directory);
  append(config.target_arch.value_or(std::string{}));
  append_path(config.toolchain.cmake_program);
  append_optional_path(config.toolchain.c_compiler);
  append_optional_path(config.toolchain.cxx_compiler);
  append_optional_path(config.toolchain.toolchain_file);
  append(config.toolchain.generator.value_or(std::string{}));
  append(config.toolchain.build_type);
  for (const auto& [name, value] : config.toolchain.environment) {
    append(name);
    append(value);
  }
  for (const auto& path : config.include_directories)
    append_path(path);
  for (const auto& value : config.compile_definitions)
    append(value);
  for (const auto& value : config.compile_options)
    append(value);
  for (const auto& path : config.link_directories)
    append_path(path);
  for (const auto& value : config.link_libraries)
    append(value);
  for (const auto& value : config.link_options)
    append(value);
  return output.str();
}

} // namespace vecops::compiler
