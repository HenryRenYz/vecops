// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "vecops/memory/Model.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace vecops::memory::bridge {
/** Process-local interop for extensions with independent Python type registries.
 *
 * The named capsule contains this descriptor, never an integer runtime address.
 * Only the exporting DSO touches its shared_ptr/control block. Each importer
 * retains the opaque owner and releases it through the exporter callback after
 * its final native user. Copying a Python capsule is not required for retention.
 *
 * This is a C-compatible function-table boundary to an explicitly versioned C++
 * interface, not a stable C API to memory operations. Increment interface_revision
 * when MemoryRuntime's vtable or argument/result data ABI changes. Exporter and
 * importer must also use the same standard-library ABI. Descriptors and callbacks
 * must originate from a trusted loaded extension and remain process-local.
 */
inline constexpr char capsule_name[] = "vecops.memory.MemoryRuntimeBridge.v1";
inline constexpr std::uint32_t interface_revision = 2;
inline constexpr std::uint64_t type_tag = 0x5645434f50534d52ULL;
#if defined(_LIBCPP_VERSION)
#  define VECOPS_MEMORY_BRIDGE_STRINGIFY_INNER(x) #x
#  define VECOPS_MEMORY_BRIDGE_STRINGIFY(x) VECOPS_MEMORY_BRIDGE_STRINGIFY_INNER(x)
inline constexpr char cxx_abi[] = "libc++:" VECOPS_MEMORY_BRIDGE_STRINGIFY(_LIBCPP_ABI_VERSION);
#  undef VECOPS_MEMORY_BRIDGE_STRINGIFY
#  undef VECOPS_MEMORY_BRIDGE_STRINGIFY_INNER
#elif defined(__GLIBCXX__) && _GLIBCXX_USE_CXX11_ABI
inline constexpr char cxx_abi[] = "libstdc++:cxx11=1";
#elif defined(__GLIBCXX__)
inline constexpr char cxx_abi[] = "libstdc++:cxx11=0";
#else
#  error "MemoryRuntime bridge requires a known C++ standard-library ABI"
#endif

struct Descriptor {
  std::uint32_t struct_size;
  std::uint32_t bridge_version;
  std::uint32_t runtime_interface_revision;
  std::uint32_t pointer_bytes;
  std::uint64_t runtime_type;
  const char* standard_library_abi;
  void* context;
  void* (*get_runtime)(void*) noexcept;
  void (*retain)(void*) noexcept;
  void (*release)(void*) noexcept;
};
static_assert(std::is_standard_layout_v<Descriptor>);

inline void validate(const Descriptor& value) {
  if (value.struct_size != sizeof(Descriptor) || value.bridge_version != 1 ||
      value.runtime_interface_revision != interface_revision || value.runtime_type != type_tag)
    throw std::invalid_argument("incompatible MemoryRuntime bridge type/version");
  if (value.pointer_bytes != sizeof(void*) || !value.standard_library_abi ||
      std::strcmp(value.standard_library_abi, cxx_abi))
    throw std::invalid_argument("incompatible MemoryRuntime C++ standard-library ABI");
  if (!value.context || !value.get_runtime || !value.retain || !value.release)
    throw std::invalid_argument("incomplete MemoryRuntime bridge owner callbacks");
}

inline std::shared_ptr<model::MemoryRuntime> acquire(const Descriptor& value) {
  validate(value);
  value.retain(value.context);
  auto* runtime = static_cast<model::MemoryRuntime*>(value.get_runtime(value.context));
  if (!runtime) {
    value.release(value.context);
    throw std::invalid_argument("empty MemoryRuntime bridge owner");
  }
  // shared_ptr invokes this deleter if allocating its local control block fails.
  return {runtime, [context = value.context, release = value.release](model::MemoryRuntime*) {
            release(context);
          }};
}

namespace details {
struct Owner {
  std::atomic<std::uint64_t> references{1};
  std::shared_ptr<model::MemoryRuntime> runtime;
  Descriptor descriptor;
  explicit Owner(std::shared_ptr<model::MemoryRuntime> value)
    : runtime(std::move(value)),
      descriptor{sizeof(Descriptor), 1, interface_revision, sizeof(void*), type_tag, cxx_abi, this,
                 [](void* p) noexcept -> void* { return static_cast<Owner*>(p)->runtime.get(); },
                 [](void* p) noexcept { static_cast<Owner*>(p)->references.fetch_add(1, std::memory_order_relaxed); },
                 [](void* p) noexcept {
                   auto* owner = static_cast<Owner*>(p);
                   if (owner->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
                     delete owner;
                 }} {
  }
};
} // namespace details
struct ReleaseDescriptor {
  void operator()(Descriptor* value) const noexcept {
    if (value)
      value->release(value->context);
  }
};
using Export = std::unique_ptr<Descriptor, ReleaseDescriptor>;
inline Export export_runtime(std::shared_ptr<model::MemoryRuntime> runtime) {
  if (!runtime)
    throw std::invalid_argument("cannot export an empty MemoryRuntime");
  auto* owner = new details::Owner(std::move(runtime));
  return Export(&owner->descriptor);
}
} // namespace vecops::memory::bridge
