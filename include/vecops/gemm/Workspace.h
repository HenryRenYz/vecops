//
// Created by renyz on 2026/7/9.
//

#ifndef VECOPS_GEMM_WORKSPACE_H
#define VECOPS_GEMM_WORKSPACE_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecops::gemm {

namespace details {

constexpr nint_t workspace_round_up(nint_t value, nint_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

} // namespace details

/**
 * @brief Non-owning single-thread workspace view.
 *
 * `WorkspaceView` is a small bump allocator for temporary kernel buffers.
 * It does not own memory and is intentionally not synchronized; callers should
 * give each thread its own view and call `reset()` between independent kernel
 * invocations.
 */
class WorkspaceView {
public:
  struct Mark {
    nint_t offset;
  };

  WorkspaceView() = default;

  WorkspaceView(void* data, nint_t capacity)
      : _base(static_cast<std::byte*>(data)), _capacity(capacity) {
    VECOPS_ASSERT(capacity >= 0, "workspace capacity must be non-negative");
  }

  void reset() { _offset = 0; }

  Mark mark() const { return Mark{_offset}; }

  void rewind(Mark mark) {
    VECOPS_ASSERT(0 <= mark.offset && mark.offset <= _offset,
                  "workspace mark is outside the active allocation range");
    _offset = mark.offset;
  }

  void* allocate(nint_t bytes, nint_t alignment = vec::DEFAULT_ALIGNMENT) {
    if (bytes == 0) return nullptr;
    VECOPS_ASSERT(_base != nullptr, "workspace buffer is null");
    VECOPS_ASSERT(bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0,
                  "workspace alignment must be a positive power of two");

    const auto raw = reinterpret_cast<std::uintptr_t>(_base + _offset);
    const auto aligned = (raw + static_cast<std::uintptr_t>(alignment - 1)) &
                         ~static_cast<std::uintptr_t>(alignment - 1);
    const nint_t aligned_offset =
        static_cast<nint_t>(aligned - reinterpret_cast<std::uintptr_t>(_base));
    const nint_t next = aligned_offset + bytes;
    VECOPS_ASSERT(next <= _capacity,
                  "workspace overflow: requested %td bytes with %td-byte alignment, capacity %td",
                  bytes, alignment, _capacity);
    _offset = next;
    if (_high_watermark < _offset) _high_watermark = _offset;
    return reinterpret_cast<void*>(aligned);
  }

  template <typename T>
  T* allocate(nint_t count) {
    return static_cast<T*>(allocate(count * static_cast<nint_t>(sizeof(T)), alignof(T)));
  }

  nint_t used() const { return _offset; }
  nint_t capacity() const { return _capacity; }
  nint_t high_watermark() const { return _high_watermark; }

private:
  std::byte* _base = nullptr;
  nint_t _capacity = 0;
  nint_t _offset = 0;
  nint_t _high_watermark = 0;
};

/**
 * @brief Owning single-thread workspace.
 */
class Workspace {
public:
  Workspace() = default;
  explicit Workspace(nint_t bytes) { reserve(bytes); }

  void reserve(nint_t bytes) {
    VECOPS_ASSERT(bytes >= 0, "workspace reserve size must be non-negative");
    _requested = bytes;
    _storage.resize(static_cast<size_t>(bytes + vec::DEFAULT_ALIGNMENT));
  }

  WorkspaceView view() {
    return WorkspaceView(_storage.data(), static_cast<nint_t>(_storage.size()));
  }

  nint_t requested_capacity() const { return _requested; }
  nint_t storage_size() const { return static_cast<nint_t>(_storage.size()); }

private:
  nint_t _requested = 0;
  std::vector<std::byte> _storage;
};

/**
 * @brief Owning workspace that provides one independent view per thread.
 */
class ParallelWorkspace {
public:
  ParallelWorkspace() = default;

  ParallelWorkspace(nint_t num_threads, nint_t per_thread_bytes) {
    reserve(num_threads, per_thread_bytes);
  }

  void reserve(nint_t num_threads, nint_t per_thread_bytes) {
    VECOPS_ASSERT(num_threads >= 0, "thread count must be non-negative");
    VECOPS_ASSERT(per_thread_bytes >= 0, "per-thread workspace size must be non-negative");
    _num_threads = num_threads;
    _per_thread_bytes = per_thread_bytes;
    _stride = details::workspace_round_up(
        per_thread_bytes + vec::DEFAULT_ALIGNMENT,
        vec::DEFAULT_ALIGNMENT);
    _storage.resize(static_cast<size_t>(_stride * num_threads + vec::DEFAULT_ALIGNMENT));
  }

  WorkspaceView thread_view(nint_t tid) {
    VECOPS_ASSERT(0 <= tid && tid < _num_threads, "thread id is out of range");
    return WorkspaceView(_storage.data() + tid * _stride, _stride);
  }

  nint_t num_threads() const { return _num_threads; }
  nint_t per_thread_bytes() const { return _per_thread_bytes; }
  nint_t stride() const { return _stride; }
  nint_t storage_size() const { return static_cast<nint_t>(_storage.size()); }

private:
  nint_t _num_threads = 0;
  nint_t _per_thread_bytes = 0;
  nint_t _stride = 0;
  std::vector<std::byte> _storage;
};

} // namespace vecops::gemm

#endif // VECOPS_GEMM_WORKSPACE_H
