//
// Created by renyz on 2026/7/9.
//

#ifndef VECOPS_KERNEL_WORKSPACE_H
#define VECOPS_KERNEL_WORKSPACE_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @file Workspace.h
 * @brief Aligned bump-allocation storage for temporary kernel sessions.
 *
 * `WorkspaceView` is the non-owning primitive used by Tensor DataAccess
 * materialization. `Workspace` and `ParallelWorkspace` are convenience owners
 * for one or several independent views. Allocations are monotonic until
 * `rewind()` or `reset()` and never run constructors or destructors.
 *
 * @code
 * kernel::Workspace storage(required_bytes);
 * auto ws = storage.view();
 * auto mark = ws.mark();
 * float* tmp = ws.allocate<float>(count);
 * // use tmp; destroy objects that refer to it
 * ws.rewind(mark);
 * @endcode
 *
 * ## DataAccess lifetime
 *
 * `kernel::with_operands` creates a mark, allocates all materialized operand
 * buffers, destroys the scoped sessions after the callback, then rewinds the
 * mark. Materialized outputs must be committed while their buffer is alive.
 * A borrowed DataAccess slice must not escape that callback.
 *
 * ## Pitfalls
 *
 * - Workspace memory is raw storage. Placement-created non-trivial objects
 *   must be explicitly destroyed before rewind/reset.
 * - A view does not own its buffer and is not thread-safe.
 * - `high_watermark()` is historical and is not reduced by rewind/reset.
 * - `Workspace::view()` exposes the padded storage size, while
 *   `requested_capacity()` reports the caller's requested payload.
 * - Per-thread views are independent only when each thread uses its own tid.
 */
namespace vecops::kernel {

/**
 * @brief Non-owning single-thread workspace view.
 *
 * `WorkspaceView` is a small aligned bump allocator for temporary kernel
 * buffers. It does not own memory and is intentionally not synchronized.
 * Allocation failure is reported through `VECOPS_ASSERT`; no heap fallback is
 * attempted.
 */
class WorkspaceView {
public:
  /** @brief Opaque rewind point within this view's current allocation stack. */
  struct Mark {
    nint_t offset;
  };

  WorkspaceView() = default;

  /** @brief Adopt `data` as the backing buffer without taking ownership.
   *
   * @param data      Buffer start; may be null only for a zero-capacity view.
   * @param capacity  Buffer size in bytes (non-negative, asserted).
   * @note The buffer must outlive every allocation made through this view.
   */
  WorkspaceView(void* data, nint_t capacity)
      : _base(static_cast<std::byte*>(data)), _capacity(capacity) {
    VECOPS_ASSERT(capacity >= 0, "workspace capacity must be non-negative");
  }

  /** @brief Release all bump allocations without changing high watermark. */
  void reset() { _offset = 0; }

  /** @brief Capture the current bump offset for later LIFO-style rewind. */
  Mark mark() const { return Mark{_offset}; }

  /**
   * @brief Release allocations made after `mark`.
   * @note The mark belongs to the same view and must not be ahead of `used()`.
   */
  void rewind(Mark mark) {
    VECOPS_ASSERT(0 <= mark.offset && mark.offset <= _offset,
                  "workspace mark is outside the active allocation range");
    _offset = mark.offset;
  }

  /**
   * @brief Allocate uninitialized aligned bytes from the view.
   * @param bytes Payload size; zero returns nullptr without advancing.
   * @param alignment Positive power-of-two byte alignment.
   * @return Pointer valid until a covering rewind/reset or owner destruction.
   */
  void* allocate(nint_t bytes, nint_t alignment = vec::DEFAULT_ALIGNMENT) {
    if (bytes == 0) return nullptr;
    VECOPS_ASSERT(_base != nullptr, "workspace buffer is null");
    VECOPS_ASSERT(bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0,
                  "workspace alignment must be a positive power of two");

    // Round the cursor up to the requested alignment; the skipped padding
    // bytes are never zeroed (the storage is raw).
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

  /**
   * @brief Allocate raw storage for `count` T objects.
   * @note This does not initialize T and is primarily intended for trivial
   * Tensor element buffers.
   */
  template <typename T>
  T* allocate(nint_t count) {
    return static_cast<T*>(allocate(count * static_cast<nint_t>(sizeof(T)), alignof(T)));
  }

  /** @brief Bytes currently bump-allocated (the cursor position). */
  nint_t used() const { return _offset; }
  /** @brief Total bytes this view may hand out. */
  nint_t capacity() const { return _capacity; }
  /** @brief Historical peak of `used()`; not reduced by rewind()/reset(). */
  nint_t high_watermark() const { return _high_watermark; }

private:
  std::byte* _base = nullptr;
  nint_t _capacity = 0;
  nint_t _offset = 0;
  nint_t _high_watermark = 0;
};

/**
 * @brief Owning single-thread workspace with alignment padding.
 *
 * Resizing through `reserve()` invalidates every previously returned view and
 * pointer. Despite its name, `reserve()` sets the requested size exactly and
 * may shrink the vector.
 */
class Workspace {
public:
  Workspace() = default;

  /** @brief Construct and reserve exactly `bytes` (see reserve()). */
  explicit Workspace(nint_t bytes) { reserve(bytes); }

  /**
   * @brief Set the requested payload size exactly (may shrink).
   *
   * The backing vector is resized to `bytes` plus one DEFAULT_ALIGNMENT of
   * slack so views can satisfy their first aligned allocation. Resizing
   * invalidates every previously returned view and pointer.
   */
  void reserve(nint_t bytes) {
    VECOPS_ASSERT(bytes >= 0, "workspace reserve size must be non-negative");
    _requested = bytes;
    _storage.resize(static_cast<size_t>(bytes + vec::DEFAULT_ALIGNMENT));
  }

  /** @brief Borrow a view over the (padded) storage; not synchronized. */
  WorkspaceView view() {
    return WorkspaceView(_storage.data(), static_cast<nint_t>(_storage.size()));
  }

  /** @brief Caller-requested payload, without the alignment padding. */
  nint_t requested_capacity() const { return _requested; }
  /** @brief Actual storage size including padding (what view() exposes). */
  nint_t storage_size() const { return static_cast<nint_t>(_storage.size()); }

private:
  nint_t _requested = 0;
  std::vector<std::byte> _storage;
};

/**
 * @brief Owning workspace that provides one padded independent view per thread.
 *
 * The stride prevents adjacent thread regions from sharing the requested
 * payload. This class allocates storage only; it performs no thread scheduling
 * and does not guard duplicate tid use.
 */
class ParallelWorkspace {
public:
  ParallelWorkspace() = default;

  /** @brief Construct and reserve `num_threads` views of `per_thread_bytes`. */
  ParallelWorkspace(nint_t num_threads, nint_t per_thread_bytes) {
    reserve(num_threads, per_thread_bytes);
  }

  /**
   * @brief (Re)allocate per-thread regions; invalidates all previous views.
   *
   * Each thread region gets one extra DEFAULT_ALIGNMENT of slack so its
   * usable payload survives the view's internal alignment rounding
   * (including any misalignment inherited from the vector's base pointer),
   * and the region stride keeps adjacent payloads separated.
   */
  void reserve(nint_t num_threads, nint_t per_thread_bytes) {
    VECOPS_ASSERT(num_threads >= 0, "thread count must be non-negative");
    VECOPS_ASSERT(per_thread_bytes >= 0, "per-thread workspace size must be non-negative");
    _num_threads = num_threads;
    _per_thread_bytes = per_thread_bytes;
    _stride = align_up(
        per_thread_bytes + vec::DEFAULT_ALIGNMENT,
        vec::DEFAULT_ALIGNMENT);
    _storage.resize(static_cast<size_t>(_stride * num_threads + vec::DEFAULT_ALIGNMENT));
  }

  /**
   * @brief Borrow thread `tid`'s independent region as a view.
   *
   * @param tid  Thread index in [0, num_threads()) (asserted).
   * @note Views are unsynchronized; independence holds only when each thread
   *       uses its own tid exclusively.
   */
  WorkspaceView thread_view(nint_t tid) {
    VECOPS_ASSERT(0 <= tid && tid < _num_threads, "thread id is out of range");
    return WorkspaceView(_storage.data() + tid * _stride, _stride);
  }

  /** @brief Number of per-thread regions reserved. */
  nint_t num_threads() const { return _num_threads; }
  /** @brief Caller-requested payload per thread, without alignment slack. */
  nint_t per_thread_bytes() const { return _per_thread_bytes; }
  /** @brief Region stride in bytes: payload plus alignment slack. */
  nint_t stride() const { return _stride; }
  /** @brief Total storage size including all padding. */
  nint_t storage_size() const { return static_cast<nint_t>(_storage.size()); }

private:
  nint_t _num_threads = 0;
  nint_t _per_thread_bytes = 0;
  nint_t _stride = 0;
  std::vector<std::byte> _storage;
};

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_WORKSPACE_H
