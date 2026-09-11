/**
 * @file Provider.h
 * @brief Memory/disk artifact resolution with explicit cache side-effect modes.
 */
#ifndef VECOPS_RUNTIME_PROVIDER_H
#define VECOPS_RUNTIME_PROVIDER_H

#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "vecops/runtime/Operator.h"

namespace vecops::runtime {

/**
 * @brief Persistent-artifact side-effect policy for an executable provider.
 *
 * The mode applies only to the persistent cache. Every mode may memoize a
 * successfully loaded executable in memory for this provider's lifetime.
 */
enum class ArtifactCacheMode {
  /** Read existing artifacts only; never create locks, directories, or files. */
  CacheOnly,
  /** Read existing artifacts, compile misses, and atomically publish results. */
  ReadWrite,
  /** Compile misses without consulting or modifying the persistent cache. */
  CompileOnly,
};

/**
 * @brief Application-supplied JIT/AOT build callback for a bound recipe.
 *
 * The callback may compile and load an executable but must return one whose
 * declared specialization key equals the requested recipe key. It is invoked
 * only by `ReadWrite` and `CompileOnly`, never `CacheOnly`.
 */
using KernelBuildCallback = std::function<Result<std::shared_ptr<Executable>>(const BoundKernelRecipe&)>;

/**
 * @brief Configuration owned by `ArtifactExecutableProvider`.
 *
 * `cache_directory` is read in `CacheOnly` but is never created there. The
 * namespace separates incompatible SDK/toolchain/target populations sharing
 * a root. A missing build callback is an error only when a cache miss needs a
 * build in a mode that permits building.
 */
struct ArtifactProviderConfig {
  /** @brief Persistent-cache side-effect mode. */
  ArtifactCacheMode mode = ArtifactCacheMode::CacheOnly;
  /** @brief Root containing namespace-scoped artifact entries. */
  std::filesystem::path cache_directory;
  /** Target/toolchain/SDK namespace supplied by the application. */
  std::string namespace_key;
  /** Required by ReadWrite and CompileOnly; ignored by CacheOnly. */
  KernelBuildCallback build;
};

/**
 * @brief Thread-safe executable provider with strict cache-only behavior.
 *
 * CacheOnly performs only existence checks, directory enumeration, DSO reads,
 * and in-memory memoization. In particular it never calls `build` and never
 * creates or updates a filesystem object. Resolution is synchronized per
 * artifact key: identical concurrent misses share one result, while distinct
 * specializations may load or build concurrently.
 */
class ArtifactExecutableProvider final : public ExecutableProvider {
public:
  /** @brief Take ownership of immutable provider configuration. */
  explicit ArtifactExecutableProvider(ArtifactProviderConfig config);
  /** @copydoc ExecutableProvider::resolve */
  [[nodiscard]] Result<std::shared_ptr<Executable>> resolve(const BoundKernelRecipe& recipe) override;
  /**
   * @brief Resolve from memory or the persistent cache without compiling.
   *
   * CompileOnly always reports NotFound unless the specialization is already
   * memoized. This is the side-effect-free planning half used by batch builds.
   */
  [[nodiscard]] Result<std::shared_ptr<Executable>> lookup(const BoundKernelRecipe& recipe);
  /**
   * @brief Validate and publish an executable produced by a batch compiler.
   *
   * ReadWrite atomically publishes through the normal artifact cache;
   * CompileOnly memoizes the supplied executable without filesystem writes.
   */
  [[nodiscard]] Result<std::shared_ptr<Executable>> adopt(const BoundKernelRecipe& recipe,
                                                          std::shared_ptr<Executable> executable);
  /** @brief Return the immutable persistent-cache policy. */
  [[nodiscard]] ArtifactCacheMode mode() const {
    return config_.mode;
  }

private:
  struct InFlight {
    std::condition_variable ready;
    bool complete = false;
    std::optional<Result<std::shared_ptr<Executable>>> result;
  };

  [[nodiscard]] std::string key(const BoundKernelRecipe& recipe) const;
  [[nodiscard]] Result<std::shared_ptr<Executable>> load_cached(const BoundKernelRecipe& recipe,
                                                                std::string_view key) const;
  [[nodiscard]] Result<std::shared_ptr<Executable>> publish(const BoundKernelRecipe& recipe, std::string_view key,
                                                            std::shared_ptr<Executable> executable) const;

  ArtifactProviderConfig config_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<Executable>, std::less<>> memory_cache_;
  std::map<std::string, std::shared_ptr<InFlight>, std::less<>> in_flight_;
};

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_PROVIDER_H
