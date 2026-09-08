/**
 * @file Provider.h
 * @brief Memory/disk artifact resolution with explicit cache side-effect modes.
 */
#ifndef VECOPS_RUNTIME_PROVIDER_H
#define VECOPS_RUNTIME_PROVIDER_H

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
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
 * creates or updates a filesystem object.
 */
class ArtifactExecutableProvider final : public ExecutableProvider {
public:
  /** @brief Take ownership of immutable provider configuration. */
  explicit ArtifactExecutableProvider(ArtifactProviderConfig config);
  /** @copydoc ExecutableProvider::resolve */
  [[nodiscard]] Result<std::shared_ptr<Executable>> resolve(const BoundKernelRecipe& recipe) override;

private:
  [[nodiscard]] std::string key(const BoundKernelRecipe& recipe) const;
  [[nodiscard]] Result<std::shared_ptr<Executable>> load_cached(const BoundKernelRecipe& recipe,
                                                                std::string_view key) const;
  [[nodiscard]] Result<std::shared_ptr<Executable>> publish(const BoundKernelRecipe& recipe, std::string_view key,
                                                            std::shared_ptr<Executable> executable) const;

  ArtifactProviderConfig config_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<Executable>, std::less<>> memory_cache_;
};

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_PROVIDER_H
