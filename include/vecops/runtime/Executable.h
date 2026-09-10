/**
 * @file Executable.h
 * @brief Dynamic kernel-module loading and validated artifact invocation.
 *
 * `LoadedModule` owns a dynamic-library handle, and `Executable` retains that
 * ownership while exposing one immutable kernel descriptor.  Loading and
 * invocation may execute loader or kernel code; callers must synchronize
 * access to aliased writable tensors and any kernels that are not documented
 * as re-entrant.
 */
#ifndef VECOPS_RUNTIME_EXECUTABLE_H
#define VECOPS_RUNTIME_EXECUTABLE_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

#include "vecops/runtime/Argument.h"
#include "vecops/runtime/KernelAbi.h"
#include "vecops/runtime/Status.h"

namespace vecops::runtime {

/**
 * @brief RAII owner of a loaded kernel DSO and its immutable descriptor.
 *
 * A descriptor reference remains valid only while a shared owner of this
 * object exists. The destructor unloads the DSO, so no callback/function
 * pointer or descriptor reference may outlive the final `shared_ptr` owner.
 * `load` performs loader-visible side effects; immutable accessors are safe
 * for concurrent reads after successful construction.
 */
class LoadedModule {
public:
  /** @brief Unload the backing DSO after the final shared owner is destroyed. */
  ~LoadedModule();
  /** @brief Modules are not copyable because the native handle has unique ownership. */
  LoadedModule(const LoadedModule&) = delete;
  /** @brief Modules are not assignable because the native handle has unique ownership. */
  LoadedModule& operator=(const LoadedModule&) = delete;

  /**
   * @brief Load and validate one kernel artifact DSO.
   * @param library_path Path to the shared library exporting the v1 query symbol.
   * @return A shared module owner, or a load/ABI failure status.
   *
   * Invokes platform dynamic-loader APIs and the artifact's query function.
   * Those operations can have process-wide loader side effects and should not
   * race with external unloads of the same library.
   */
  static Result<std::shared_ptr<LoadedModule>> load(const std::filesystem::path& library_path);

  /**
   * @brief Return the validated immutable ABI descriptor.
   * @return Reference valid while this module remains alive.
   */
  [[nodiscard]] const VecopsKernelDescriptorV1& descriptor() const {
    return *descriptor_;
  }
  /** @brief Return the path used to load this module. */
  [[nodiscard]] const std::filesystem::path& path() const {
    return path_;
  }

private:
  LoadedModule(std::filesystem::path path, void* handle, const VecopsKernelDescriptorV1* descriptor)
    : path_(std::move(path))
    , handle_(handle)
    , descriptor_(descriptor) {
  }

  std::filesystem::path path_;
  void* handle_ = nullptr;
  const VecopsKernelDescriptorV1* descriptor_ = nullptr;
};

/**
 * @brief A validated specialization backed by a loaded artifact DSO.
 *
 * The object preserves module lifetime through `module()`.  Public execution
 * methods synchronously validate metadata before entering a kernel callback;
 * they do not make overlapping tensor writes or kernel-global mutable state
 * thread-safe.
 */
class Executable {
public:
  /**
   * @brief Retain a loaded module as one executable specialization.
   * @param module Valid loaded module; callers must not pass null.
   */
  explicit Executable(std::shared_ptr<LoadedModule> module)
    : module_(std::move(module)) {
  }

  /**
   * @brief Load an artifact and construct its executable wrapper.
   * @param library_path Shared library to load.
   * @return Executable retaining the module, or an error status.
   */
  static Result<std::shared_ptr<Executable>> load(const std::filesystem::path& library_path);

  /** @brief Return the immutable logical operator name declared by the DSO. */
  [[nodiscard]] std::string_view operator_name() const;
  /** @brief Return the immutable specialization identity declared by the DSO. */
  [[nodiscard]] std::string_view specialization_key() const;
  /**
   * @brief Return the shared owner that keeps descriptor storage and DSO code alive.
   * @return Immutable reference to this executable's module owner.
   */
  [[nodiscard]] const std::shared_ptr<LoadedModule>& module() const {
    return module_;
  }

  /**
   * @brief Validate a C++ argument sequence against this exact descriptor.
   * @param arguments Ordered non-owning tensor/scalar metadata.
   * @return Success when this specialization can accept the arguments.
   *
   * Pure with respect to tensor storage and does not invoke kernel code.
   */
  [[nodiscard]] Status can_invoke(const ArgumentMetadata& arguments) const;
  /**
   * @brief Query workspace required by this specialization for @p arguments.
   * @param arguments Ordered metadata to validate and pass to the callback.
   * @param context Optional non-owning context valid for the synchronous query.
   * @return Required byte count, or validation/callback failure.
   *
   * The descriptor's workspace callback must not modify tensor storage.
   */
  [[nodiscard]] Result<std::uint64_t> workspace_size(const ArgumentMetadata& arguments,
                                                     const VecopsExecutionContext* context = nullptr) const;
  /**
   * @brief Validate and synchronously run this specialization.
   * @param arguments Ordered metadata whose tensor storage outlives the call.
   * @param workspace Caller-owned workspace, or null when zero bytes are needed.
   * @param workspace_size Available workspace bytes.
   * @param context Optional caller-owned execution context valid for this call.
   * @return Success or validation/kernel failure.
   *
   * May write only argument tensors carrying `VECOPS_TENSOR_WRITE`; callers
   * synchronize aliases and concurrent kernel execution.
   */
  [[nodiscard]] Status invoke(const ArgumentMetadata& arguments, void* workspace = nullptr,
                              std::uint64_t workspace_size = 0, const VecopsExecutionContext* context = nullptr) const;

  /**
   * @brief Invoke an ABI frame already proven to match this specialization.
   * @param call Complete call frame whose metadata was validated against this
   * executable and has not changed except for tensor storage locations and
   * runtime scalar payloads.
   * @return Kernel status.
   *
   * This expert hot-path entry skips descriptor and workspace validation.  It
   * is intended for framework bridges that retain this Executable and guard
   * every invocation against the metadata of a previously successful call.
   * Calling it with different tensor metadata, scalar dtype,
   * optional-presence, specialization value, context, or workspace contract
   * is invalid.
   */
  [[nodiscard]] Status invoke_prevalidated(const VecopsCall& call) const;

private:
  [[nodiscard]] Status invoke_unchecked(const VecopsCall& call) const;
  std::shared_ptr<LoadedModule> module_;
};

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_EXECUTABLE_H
