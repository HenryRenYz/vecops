// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/**
 * @file Compiler.h
 * @brief Isolated CMake-based compilation of vecops kernel artifact DSOs.
 *
 * `Compiler` generates one top-level project containing one or more task
 * fragments and launches CMake subprocesses without a shell. Requests describe
 * caller-owned filesystem locations; compilation creates or writes the
 * requested generated-source, build, artifact, and log paths. It does not
 * cache results or synchronize concurrent writers, so callers must provide
 * distinct directories or external locking.
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "vecops/runtime/Executable.h"
#include "vecops/runtime/KernelDefinition.h"

namespace vecops::runtime {
struct BoundKernelRecipe;
}

namespace vecops::compiler {

/** @brief How a generated kernel project consumes Vecops. */
enum class SdkLayout {
  /** @brief Detect a source tree, otherwise treat the path as a package location. */
  Auto,
  /** @brief Use `add_subdirectory()` on a vecops source checkout. */
  SourceTree,
  /** @brief Use `find_package(Vecops CONFIG)` with an installed/build-tree package. */
  Package,
};

/**
 * @brief Location and consumption mode for the vecops SDK.
 *
 * This structure owns its path value.  `Compiler::compile` only reads it, but
 * the referenced SDK files must remain accessible until all subprocesses
 * finish.
 */
struct SdkSpec {
  /** @brief Source root, install prefix, or directory containing `VecopsConfig.cmake`. */
  std::filesystem::path path;
  /** @brief Resolution mode applied to `path`. */
  SdkLayout layout = SdkLayout::Auto;
};

/**
 * @brief Native build tools and environment used by one compiler invocation.
 *
 * Explicit compiler paths override inherited `CC`/`CXX` in the generated CMake
 * configuration.  The environment map augments the current process
 * environment for child processes only; it does not mutate this process.
 */
struct ToolchainSpec {
  /** @brief CMake executable to invoke. */
  std::filesystem::path cmake_program = "cmake";
  /** @brief Optional C compiler passed as `CMAKE_C_COMPILER`. */
  std::optional<std::filesystem::path> c_compiler;
  /** @brief Optional C++ compiler passed as `CMAKE_CXX_COMPILER`. */
  std::optional<std::filesystem::path> cxx_compiler;
  /** @brief Optional CMake toolchain file. */
  std::optional<std::filesystem::path> toolchain_file;
  /** @brief Optional CMake generator name. */
  std::optional<std::string> generator;
  /** @brief CMake build type supplied to single-config generators. */
  std::string build_type = "Release";
  /** @brief Requested parallel jobs; zero leaves parallelism to CMake. */
  std::size_t parallel_jobs = 0;

  /**
   * @brief Additional environment entries for CMake subprocesses.
   *
   * The current process environment is inherited.  Explicit compiler arguments
   * above take precedence over inherited `CC` and `CXX` values.
   */
  std::map<std::string, std::string> environment;
};

/**
 * @brief Reusable policy for compiling declarative `__kernel__` source files.
 *
 * The config is copied into `Compiler`. `work_directory` is an attempt root,
 * not a cache: each `compile_kernel` call creates a fresh child project below
 * it. Persistent artifact lookup and publication remain a runtime-provider
 * responsibility.
 */
struct KernelCompilerConfig {
  /** @brief Vecops headers/libraries consumed by generated projects. */
  SdkSpec sdk;
  /** @brief CMake executable, compilers, and subprocess environment. */
  ToolchainSpec toolchain;
  /** Fresh per-attempt projects are created below this non-cache directory. */
  std::filesystem::path work_directory;
  /** @brief Optional vecops target architecture override for generated kernels. */
  std::optional<std::string> target_arch;
  /** @brief Extra include paths appended to the generated target. */
  std::vector<std::filesystem::path> include_directories;
  /** @brief Extra preprocessor definitions appended to the generated target. */
  std::vector<std::string> compile_definitions;
  /** @brief Extra compiler options appended to the generated target. */
  std::vector<std::string> compile_options;
  /** @brief Extra library search paths appended to the generated target. */
  std::vector<std::filesystem::path> link_directories;
  /** @brief Extra libraries appended to the generated target. */
  std::vector<std::string> link_libraries;
  /** @brief Extra linker options appended to the generated target. */
  std::vector<std::string> link_options;
};

/**
 * @brief Complete, cache-independent description of one kernel module build.
 *
 * The request is a value object.  Paths are interpreted by the local process;
 * source and include inputs are read, while generated/build/artifact paths may
 * be created and written by `compile`.  The three output directories must be
 * distinct, and non-empty reuse requires the corresponding explicit opt-in.
 */
struct KernelBuildRequest {
  /** @brief CMake-safe target identifier for the generated kernel library. */
  std::string target_name;
  /** @brief Optional output library filename stem. */
  std::optional<std::string> output_name;
  /**
   * @brief Target architecture accepted by vecops kernel CMake helpers.
   *
   * Examples include Scalar, AVX, AVX2, AVX512, Native, NEON, SVE, SVE2, and
   * supported fixed-vector variants for the current host architecture.
   */
  std::optional<std::string> target_arch;
  /** @brief vecops SDK location and consumption mode. */
  SdkSpec sdk;
  /** @brief Native tools and child-process environment. */
  ToolchainSpec toolchain;
  /** @brief Link the generated module to the Vecops runtime target. */
  bool link_runtime = true;

  /** @brief Kernel source files read during generation and compilation. */
  std::vector<std::filesystem::path> sources;
  /** @brief Include directories added to the generated kernel target. */
  std::vector<std::filesystem::path> include_directories;
  /** @brief Preprocessor definitions added to the generated kernel target. */
  std::vector<std::string> compile_definitions;
  /** @brief Compiler options added to the generated kernel target. */
  std::vector<std::string> compile_options;
  /** @brief Linker search directories added to the generated kernel target. */
  std::vector<std::filesystem::path> link_directories;
  /** @brief Link libraries added to the generated kernel target. */
  std::vector<std::string> link_libraries;
  /** @brief Linker options added to the generated kernel target. */
  std::vector<std::string> link_options;

  /**
   * @brief Compiler-owned generated project directory.
   *
   * `compile` writes the generated `CMakeLists.txt` here.
   */
  std::filesystem::path generated_source_directory;
  /** @brief CMake binary directory created or reused by `compile`. */
  std::filesystem::path build_directory;
  /** @brief Install destination populated by the generated project. */
  std::filesystem::path artifact_directory;

  /** @brief Extra `-D` entries passed as literal CMake cache values. */
  std::map<std::string, std::string> cmake_cache_variables;

  /**
   * @brief Permit reuse of an existing CMake binary tree.
   *
   * This is opt-in because compiler and SDK selection are sticky CMake cache
   * properties.
   */
  bool reuse_build_directory = false;
  /** @brief Permit overwriting an existing generated `CMakeLists.txt`. */
  bool reuse_generated_source_directory = false;
};

/** @brief Progress stage reached by a compiler invocation. */
enum class BuildStage {
  /** @brief Compilation did not begin validation. */
  None,
  /** @brief Request validation is running or failed. */
  Validate,
  /** @brief Generated CMake project is being written. */
  Generate,
  /** @brief CMake configuration is running or failed. */
  Configure,
  /** @brief CMake build (and any legacy install target) is running or failed. */
  BuildAndInstall,
  /** @brief Artifact discovery completed successfully. */
  Complete,
};

/**
 * @brief Captured result of one CMake subprocess.
 *
 * `output` is copied from `log_path` after the process exits.  No process is
 * left running when this value is returned.
 */
struct CommandResult {
  /** @brief Exact argument vector supplied without shell parsing. */
  std::vector<std::string> arguments;
  /** @brief Process exit code, or a launcher-specific failure code. */
  int exit_code = -1;
  /** @brief File written with combined subprocess stdout and stderr. */
  std::filesystem::path log_path;
  /** @brief Captured log contents available after process completion. */
  std::string output;
};

/**
 * @brief Complete outcome of `Compiler::compile`.
 *
 * This is returned for all outcomes instead of throwing.  On failure, `stage`
 * identifies the furthest reached phase and `error` explains validation or
 * filesystem failure; subprocess details are populated when available.
 */
struct BuildResult {
  /** @brief True only after build and artifact discovery succeed. */
  bool success = false;
  /** @brief Furthest phase reached. */
  BuildStage stage = BuildStage::None;
  /** @brief Human-readable failure detail, empty on success. */
  std::string error;
  /** @brief Configure subprocess result when configuration was attempted. */
  std::optional<CommandResult> configure;
  /** @brief Build subprocess result; name retained for source compatibility. */
  std::optional<CommandResult> build_and_install;
  /** @brief Requested artifact directory, whether or not build succeeded. */
  std::filesystem::path artifact_directory;
  /** @brief Files discovered under `artifact_directory` after a successful install. */
  std::vector<std::filesystem::path> artifact_files;
  /** @brief Located kernel DSO when exactly one suitable artifact is found. */
  std::optional<std::filesystem::path> kernel_library;
};

/**
 * @brief One CMake invocation containing independently-addressable kernel targets.
 *
 * Every task keeps its own generated CMake fragment, binary subdirectory, and
 * artifact directory.  The batch owns the only top-level CMake project and
 * therefore imports the Vecops SDK once and exposes every translation unit to
 * one native build-system scheduler.
 */
struct KernelBuildBatchRequest {
  /** @brief Non-empty task list built by one CMake project. */
  std::vector<KernelBuildRequest> tasks;
  /** @brief Directory receiving the top-level project and task fragments. */
  std::filesystem::path generated_source_directory;
  /** @brief Single CMake binary tree shared by every task. */
  std::filesystem::path build_directory;
};

/** @brief Outcome of one configure/build command plus per-task discovery. */
struct BuildBatchResult {
  /** @brief True when every task completed and exposed its requested DSO. */
  bool success = false;
  /** @brief Furthest batch-wide phase reached. */
  BuildStage stage = BuildStage::None;
  /** @brief Batch-wide validation or subprocess error. */
  std::string error;
  /** @brief Shared configure subprocess result. */
  std::optional<CommandResult> configure;
  /** @brief Shared build subprocess result. */
  std::optional<CommandResult> build;
  /** @brief Results in the same order as `KernelBuildBatchRequest::tasks`. */
  std::vector<BuildResult> tasks;
};

/**
 * @brief Generate a minimal CMake project and invoke CMake without a shell.
 *
 * The compiler performs no artifact lookup cache across calls.  It never reads
 * `CC`/`CXX` itself: without explicit compiler paths, CMake applies its normal
 * inherited environment and platform selection rules.  Instances hold no
 * mutable state, but calls targeting the same output paths must be externally
 * serialized.
 */
class Compiler {
public:
  Compiler() = default;
  explicit Compiler(KernelCompilerConfig config)
    : kernel_config_(std::move(config)) {
  }

  /**
   * @brief Validate, generate, configure, build, and discover one kernel DSO.
   * @param request Complete source, SDK, toolchain, and output-path description.
   * @return Detailed build result; errors are represented in the result.
   *
   * Creates/writes the request's generated, build, artifact, and log files and
   * launches CMake child processes.  It does not invoke a shell or mutate the
   * parent process environment.
   */
  [[nodiscard]] BuildResult compile(const KernelBuildRequest& request) const;

  /**
   * @brief Build many kernel module targets through one CMake/Ninja graph.
   *
   * Tasks must share one SDK and toolchain. Target-specific architecture,
   * sources, compile/link options, and output directories remain independent.
   * No per-task CMake subprocess is launched.
   */
  [[nodiscard]] BuildBatchResult compile_batch(const KernelBuildBatchRequest& request) const;

  /**
   * @brief Bind, generate, compile, and load a C++ file defining `__kernel__`.
   *
   * This is intentionally cache-independent. Every call creates a fresh build
   * attempt; cache lookup/publication is owned by an executable provider.
   * Generated kernels report zero external workspace because allocation is
   * wholly owned by the user kernel.
   */
  [[nodiscard]] runtime::Result<std::shared_ptr<runtime::Executable>>
  compile_kernel(const std::filesystem::path& kernel_file, const runtime::KernelDef& definition,
                 const runtime::KernelCall& call) const;

  /** @brief Compile the source instantiation carried by a bound recipe. */
  [[nodiscard]] runtime::Result<std::shared_ptr<runtime::Executable>>
  compile_kernel(const runtime::BoundKernelRecipe& recipe) const;

  /**
   * @brief Compile source-backed bound recipes through one generated project.
   * @return One result per input recipe, preserving input order.
   */
  [[nodiscard]] std::vector<runtime::Result<std::shared_ptr<runtime::Executable>>>
  compile_kernels(std::span<const runtime::BoundKernelRecipe> recipes, std::size_t parallel_jobs = 0) const;

  /**
   * @brief Return the optional source-kernel policy retained by this compiler.
   *
   * An empty value means `compile(request)` remains available but
   * `compile_kernel(...)` cannot construct its generated build request.
   */
  [[nodiscard]] const std::optional<KernelCompilerConfig>& kernel_config() const {
    return kernel_config_;
  }

  /**
   * @brief Canonical grouping key for one generated-project build domain.
   *
   * Compiler instances with equal non-empty keys can contribute source
   * recipes to the same batch even when they are distinct C++ objects.
   */
  [[nodiscard]] std::string batch_key() const;

private:
  std::optional<KernelCompilerConfig> kernel_config_;
};

} // namespace vecops::compiler
