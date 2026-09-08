/**
 * @file Provider.cpp
 * @brief Side-effect-controlled artifact cache provider implementation.
 *
 * Persistent publication builds in a unique staging directory then renames it
 * into its key path. CacheOnly uses the same lookup format but never creates a
 * directory, staging entry, or lock and never invokes the build callback.
 */

#include "vecops/runtime/Provider.h"

#include <atomic>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace vecops::runtime {
namespace {

namespace fs = std::filesystem;
std::atomic<std::uint64_t> temporary_counter{0};

Status status(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

std::uint64_t hash_text(std::uint64_t hash, std::string_view text) {
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

std::string hash_key(std::string_view first, std::string_view second) {
  auto hash = hash_text(UINT64_C(14695981039346656037), first);
  hash = hash_text(hash, ":");
  hash = hash_text(hash, std::to_string(first.size()));
  hash = hash_text(hash, ":");
  hash = hash_text(hash, second);
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

bool library_file(const fs::path& path) {
  const auto extension = path.extension().string();
  return extension == ".so" || extension == ".dylib" || extension == ".dll";
}

} // namespace

ArtifactExecutableProvider::ArtifactExecutableProvider(ArtifactProviderConfig config)
  : config_(std::move(config)) {
}

std::string ArtifactExecutableProvider::key(const BoundKernelRecipe& recipe) const {
  const auto recipe_key =
    recipe.artifact_key.empty() ? recipe.recipe_id + ";" + recipe.specialization_key : recipe.artifact_key;
  return hash_key(config_.namespace_key, recipe_key);
}

Result<std::shared_ptr<Executable>> ArtifactExecutableProvider::load_cached(const BoundKernelRecipe& recipe,
                                                                            std::string_view cache_key) const {
  if (config_.cache_directory.empty())
    return status(StatusCode::NotFound, "artifact cache directory is not configured");
  const auto directory =
    config_.cache_directory / hash_key("namespace", config_.namespace_key) / std::string(cache_key);
  std::error_code error;
  if (!fs::is_directory(directory, error) || error)
    return status(StatusCode::NotFound, "kernel artifact is not cached");
  fs::path library;
  for (fs::directory_iterator iterator(directory, error), end; !error && iterator != end; iterator.increment(error)) {
    if (iterator->is_regular_file(error) && library_file(iterator->path())) {
      if (!library.empty())
        return status(StatusCode::LoadError, "cache entry contains multiple kernel libraries");
      library = iterator->path();
    }
  }
  if (error)
    return status(StatusCode::LoadError, "cannot inspect kernel cache entry: " + error.message());
  if (library.empty())
    return status(StatusCode::NotFound, "kernel cache entry has no library");
  auto executable = Executable::load(library);
  if (!executable)
    return executable.status();
  if (executable.value()->specialization_key() != recipe.specialization_key)
    return status(StatusCode::AbiMismatch, "cached artifact specialization key mismatch");
  return std::move(executable).value();
}

Result<std::shared_ptr<Executable>> ArtifactExecutableProvider::publish(const BoundKernelRecipe& recipe,
                                                                        std::string_view cache_key,
                                                                        std::shared_ptr<Executable> executable) const {
  if (config_.cache_directory.empty())
    return status(StatusCode::InvalidArgument, "ReadWrite mode requires a cache directory");
  const auto namespace_directory = config_.cache_directory / hash_key("namespace", config_.namespace_key);
  const auto destination = namespace_directory / std::string(cache_key);
  std::error_code error;
  fs::create_directories(namespace_directory, error);
  if (error)
    return status(StatusCode::InternalError, "cannot create cache namespace: " + error.message());
  if (fs::exists(destination, error))
    return load_cached(recipe, cache_key);

  const auto temporary =
    namespace_directory /
    (std::string(cache_key) + ".tmp-" + std::to_string(temporary_counter.fetch_add(1, std::memory_order_relaxed)));
  if (!fs::create_directory(temporary, error) || error)
    return status(StatusCode::InternalError, "cannot create cache staging directory: " + error.message());
  const auto extension = executable->module()->path().extension();
  const auto staged_library = temporary / ("kernel" + extension.string());
  fs::copy_file(executable->module()->path(), staged_library, fs::copy_options::none, error);
  if (error) {
    fs::remove_all(temporary);
    return status(StatusCode::InternalError, "cannot stage compiled kernel: " + error.message());
  }
  fs::rename(temporary, destination, error);
  if (error) {
    fs::remove_all(temporary);
    if (fs::exists(destination))
      return load_cached(recipe, cache_key);
    return status(StatusCode::InternalError, "cannot publish compiled kernel: " + error.message());
  }
  return load_cached(recipe, cache_key);
}

Result<std::shared_ptr<Executable>> ArtifactExecutableProvider::resolve(const BoundKernelRecipe& recipe) {
  std::lock_guard lock(mutex_);
  const auto cache_key = key(recipe);
  if (const auto found = memory_cache_.find(cache_key); found != memory_cache_.end())
    return found->second;

  if (config_.mode != ArtifactCacheMode::CompileOnly) {
    auto cached = load_cached(recipe, cache_key);
    if (cached) {
      memory_cache_.emplace(cache_key, cached.value());
      return cached;
    }
    if (cached.status().code() != StatusCode::NotFound)
      return cached.status();
  }
  if (config_.mode == ArtifactCacheMode::CacheOnly)
    return status(StatusCode::NotFound, "kernel artifact is absent in cache-only mode");
  if (!config_.build)
    return status(StatusCode::NotFound, "kernel compilation is not configured");

  auto compiled = config_.build(recipe);
  if (!compiled)
    return compiled.status();
  if (compiled.value() == nullptr)
    return status(StatusCode::InternalError, "kernel builder returned null");
  if (compiled.value()->specialization_key() != recipe.specialization_key)
    return status(StatusCode::AbiMismatch, "compiled artifact specialization key mismatch");
  if (config_.mode == ArtifactCacheMode::ReadWrite) {
    auto cached = publish(recipe, cache_key, compiled.value());
    if (!cached)
      return cached.status();
    compiled = std::move(cached);
  }
  memory_cache_.emplace(cache_key, compiled.value());
  return compiled;
}

} // namespace vecops::runtime
