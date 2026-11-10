// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_KERNEL_WORKSPACE_H
#define VECOPS_KERNEL_WORKSPACE_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/WorkspaceArena.h"
#include "vecops/tensor/Tensor.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>

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
 * auto tmp = ws.allocate_tensor<float>(tensor::make_shape(meta::Any{count}));
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
  /** Logical sub-buffers may have different physical tiers. Their cursors
   * belong to the calling lane and retain ordinary mark/rewind semantics. */
  static constexpr unsigned max_segments = 16;
  struct Segment {
    void* data = nullptr;
    nint_t capacity = 0;
    nint_t used = 0;
  };
  using Segments = std::array<Segment, max_segments>;
  /** @brief Opaque rewind point within this view's current allocation stack. */
  struct Mark {
    nint_t offset;
    std::array<nint_t, max_segments> segments{};
  };

  WorkspaceView() = default;

  /** Minimum alignment used by typed workspace allocations. */
  template <typename T>
  static constexpr nint_t typed_alignment() {
    return std::max(vec::DEFAULT_ALIGNMENT, static_cast<nint_t>(alignof(T)));
  }

  /** Conservative byte budget for one typed allocation from this view. */
  template <typename T>
  static constexpr nint_t allocation_bytes(nint_t count) {
    return align_up(count * static_cast<nint_t>(sizeof(T)), typed_alignment<T>());
  }

  /** Conservative byte budget for `allocate_tensor<T>(shape)`. */
  template <typename T, typename Shape>
    requires tensor::is_shape_v<std::remove_cvref_t<Shape>>
  static nint_t tensor_bytes(const Shape& shape) {
    return allocation_bytes<T>(tensor::numel(tensor::make_layout(shape)));
  }

  /** @brief Adopt `data` as the backing buffer without taking ownership.
   *
   * @param data      Buffer start; may be null only for a zero-capacity view.
   * @param capacity  Buffer size in bytes (non-negative, asserted).
   * @note The buffer must outlive every allocation made through this view.
   */
  WorkspaceView(void* data, nint_t capacity)
    : _base(static_cast<std::byte*>(data))
    , _capacity(capacity) {
    VECOPS_ASSERT(capacity >= 0, "workspace capacity must be non-negative");
  }

  /** Adopt separately placed logical buffers; the segment table outlives this view. */
  explicit WorkspaceView(Segments& segments)
    : _segments(&segments) {
    for (const auto& s : segments)
      _capacity += s.capacity;
  }

  /** @brief Release all bump allocations without changing high watermark. */
  void reset() {
    _offset = 0;
    if (_segments)
      for (auto& s : *_segments)
        s.used = 0;
  }

  /** @brief Capture the current bump offset for later LIFO-style rewind. */
  Mark mark() const {
    Mark result{_offset};
    if (_segments)
      for (unsigned i = 0; i < max_segments; ++i)
        result.segments[i] = (*_segments)[i].used;
    return result;
  }

  /**
   * @brief Release allocations made after `mark`.
   * @note The mark belongs to the same view and must not be ahead of `used()`.
   */
  void rewind(Mark mark) {
    VECOPS_ASSERT(0 <= mark.offset && mark.offset <= _offset, "workspace mark is outside the active allocation range");
    _offset = mark.offset;
    if (_segments)
      for (unsigned i = 0; i < max_segments; ++i) {
        VECOPS_ASSERT(mark.segments[i] <= (*_segments)[i].used, "invalid segmented workspace mark");
        (*_segments)[i].used = mark.segments[i];
      }
  }

  /**
   * @brief Allocate uninitialized aligned bytes from the view.
   * @param bytes Payload size; zero returns nullptr without advancing.
   * @param alignment Positive power-of-two byte alignment.
   * @return Pointer valid until a covering rewind/reset or owner destruction.
   */
  void* allocate(nint_t bytes, nint_t alignment = vec::DEFAULT_ALIGNMENT, unsigned part = 0) {
    if (_segments) {
      VECOPS_CHECK(part < max_segments, "workspace segment index is out of range");
      auto& s = (*_segments)[part];
      VECOPS_CHECK(bytes >= 0 && s.used <= s.capacity && bytes <= s.capacity - s.used,
                   "workspace sub-buffer capacity exceeded");
      WorkspaceView view{s.data, s.capacity};
      view._offset = s.used;
      auto* result = view.allocate(bytes, alignment);
      _offset += view._offset - s.used;
      s.used = view._offset;
      _high_watermark = std::max(_high_watermark, _offset);
      return result;
    }
    if (bytes == 0)
      return nullptr;
    VECOPS_ASSERT(_base != nullptr, "workspace buffer is null");
    VECOPS_ASSERT(bytes >= 0, "workspace allocation size must be non-negative");
    VECOPS_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0,
                  "workspace alignment must be a positive power of two");

    // Round the cursor up to the requested alignment; the skipped padding
    // bytes are never zeroed (the storage is raw).
    const auto raw = reinterpret_cast<std::uintptr_t>(_base + _offset);
    const auto aligned =
      (raw + static_cast<std::uintptr_t>(alignment - 1)) & ~static_cast<std::uintptr_t>(alignment - 1);
    const nint_t aligned_offset = static_cast<nint_t>(aligned - reinterpret_cast<std::uintptr_t>(_base));
    const nint_t next = aligned_offset + bytes;
    VECOPS_ASSERT(next <= _capacity, "workspace overflow: requested %td bytes with %td-byte alignment, capacity %td",
                  bytes, alignment, _capacity);
    _offset = next;
    if (_high_watermark < _offset)
      _high_watermark = _offset;
    return reinterpret_cast<void*>(aligned);
  }

  /** Return whether the next allocation would fit without changing state. */
  [[nodiscard]] bool can_allocate(
      nint_t bytes,
      nint_t alignment = vec::DEFAULT_ALIGNMENT) const {
    if (_segments) {
      const auto& s = (*_segments)[0];
      WorkspaceView view{s.data, s.capacity};
      view._offset = s.used;
      return view.can_allocate(bytes, alignment);
    }
    if (bytes == 0)
      return true;
    if (_base == nullptr || bytes < 0 || alignment <= 0 ||
        (alignment & (alignment - 1)) != 0) {
      return false;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(_base);
    const auto raw = base + static_cast<std::uintptr_t>(_offset);
    const auto aligned =
      (raw + static_cast<std::uintptr_t>(alignment - 1)) &
      ~static_cast<std::uintptr_t>(alignment - 1);
    const auto aligned_offset = aligned - base;
    if (aligned_offset > static_cast<std::uintptr_t>(_capacity))
      return false;
    return static_cast<std::uintptr_t>(bytes) <=
      static_cast<std::uintptr_t>(_capacity) - aligned_offset;
  }

  /** Execute a nested operator in one logical segment. Nested allocations
   * cannot escape the callback; operand sessions commit before it returns. */
  template <typename Fn>
  decltype(auto) with_subworkspace(unsigned segment, Fn&& fn) {
    if (!_segments)
      return std::forward<Fn>(fn)(*this);
    VECOPS_CHECK(segment < max_segments, "workspace segment index is out of range");
    auto& s = (*_segments)[segment];
    auto* base = static_cast<std::byte*>(s.data);
    WorkspaceView child{s.used ? base + s.used : base, s.capacity - s.used};
    return std::forward<Fn>(fn)(child);
  }

  /**
   * @brief Allocate raw storage for `count` T objects.
   *
   * The returned pointer is aligned to at least `vec::DEFAULT_ALIGNMENT`;
   * over-aligned element types retain their stronger `alignof(T)` guarantee.
   * This does not initialize T and is primarily intended for trivial Tensor
   * element buffers.
   */
  template <typename T>
  T* allocate(nint_t count, unsigned segment = 0) {
    return static_cast<T*>(allocate(count * static_cast<nint_t>(sizeof(T)), typed_alignment<T>(), segment));
  }

  /**
   * @brief Allocate one row-major contiguous Tensor with the supplied Shape.
   *
   * The returned view is non-owning: it remains valid until a covering
   * `rewind()`/`reset()` or the backing Workspace owner is destroyed.  Shape
   * metadata is preserved, including `meta::Const` and `meta::Dynamic`
   * constraints.  Storage is raw and uninitialized.
   */
  template <typename T, typename Shape>
    requires tensor::is_shape_v<std::remove_cvref_t<Shape>>
  auto allocate_tensor(Shape shape, unsigned segment = 0) {
    auto layout = tensor::make_layout(std::move(shape));
    T* data = allocate<T>(tensor::numel(layout), segment);
    return tensor::make_tensor(data, std::move(layout));
  }

  /** @brief Bytes currently bump-allocated (the cursor position). */
  nint_t used() const {
    return _offset;
  }
  /** @brief Total bytes this view may hand out. */
  nint_t capacity() const {
    return _capacity;
  }
  /** @brief Historical peak of `used()`; not reduced by rewind()/reset(). */
  nint_t high_watermark() const {
    return _high_watermark;
  }

private:
  std::byte* _base = nullptr;
  nint_t _capacity = 0;
  nint_t _offset = 0;
  nint_t _high_watermark = 0;
  Segments* _segments = nullptr;
};

/**
 * @brief Owning single-thread workspace with alignment padding.
 *
 * Resizing through `reserve()` invalidates every previously returned view and
 * pointer. Despite its name, `reserve()` sets the requested size exactly and
 * replaces the backing allocation.
 */
class Workspace {
public:
  Workspace() = default;

  /** @brief Construct and reserve exactly `bytes` (see reserve()). */
  explicit Workspace(nint_t bytes) {
    reserve(bytes);
  }

  /**
   * @brief Set the requested payload size exactly (may shrink).
   *
   * The uninitialized backing storage passes MemoryRuntime admission and is
   * allocated as `bytes` plus one
   * DEFAULT_ALIGNMENT of slack so views can satisfy their first aligned
   * allocation. Calling reserve() invalidates every previously returned view
   * and pointer. An empty payload allocates no physical storage.
   */
  void reserve(nint_t bytes) {
    VECOPS_ASSERT(bytes >= 0, "workspace reserve size must be non-negative");
    VECOPS_CHECK(bytes <= std::numeric_limits<nint_t>::max() - vec::DEFAULT_ALIGNMENT,
                 "workspace size exceeds host address range");
    const nint_t size = bytes ? bytes + vec::DEFAULT_ALIGNMENT : 0;
    auto storage = size ? execution::default_workspace_arena_provider()->allocate(
      execution::BufferTier::Fast, size, vec::DEFAULT_ALIGNMENT) : execution::WorkspaceArena{};
    _storage = std::move(storage);
    _requested = bytes;
    _storage_size = size;
  }

  /** @brief Borrow a view over the (padded) storage; not synchronized. */
  WorkspaceView view() {
    return WorkspaceView(_storage.data, _storage_size);
  }

  /** @brief Caller-requested payload, without the alignment padding. */
  nint_t requested_capacity() const {
    return _requested;
  }
  /** @brief Actual storage size including padding (what view() exposes). */
  nint_t storage_size() const {
    return _storage_size;
  }

private:
  nint_t _requested = 0;
  nint_t _storage_size = 0;
  execution::WorkspaceArena _storage;
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
   * (including any misalignment inherited from the backing allocation),
   * and the region stride keeps adjacent payloads separated.
   */
  void reserve(nint_t num_threads, nint_t per_thread_bytes) {
    VECOPS_ASSERT(num_threads >= 0, "thread count must be non-negative");
    VECOPS_ASSERT(per_thread_bytes >= 0, "per-thread workspace size must be non-negative");
    VECOPS_CHECK(per_thread_bytes <= std::numeric_limits<nint_t>::max() - 2 * vec::DEFAULT_ALIGNMENT,
                 "per-thread workspace exceeds host address range");
    const nint_t stride = per_thread_bytes ? align_up(per_thread_bytes + vec::DEFAULT_ALIGNMENT, vec::DEFAULT_ALIGNMENT) : 0;
    VECOPS_CHECK(!num_threads || stride <= (std::numeric_limits<nint_t>::max() - vec::DEFAULT_ALIGNMENT) / num_threads,
                 "parallel workspace exceeds host address range");
    const nint_t size = stride && num_threads ? stride * num_threads + vec::DEFAULT_ALIGNMENT : 0;
    auto storage = size ? execution::default_workspace_arena_provider()->allocate(
      execution::BufferTier::Fast, size, vec::DEFAULT_ALIGNMENT) : execution::WorkspaceArena{};
    _num_threads = num_threads;
    _per_thread_bytes = per_thread_bytes;
    _stride = stride;
    _storage_size = size;
    _storage = std::move(storage);
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
    return WorkspaceView(_storage.data ? static_cast<std::byte*>(_storage.data) + tid * _stride : nullptr, _stride);
  }

  /** @brief Number of per-thread regions reserved. */
  nint_t num_threads() const {
    return _num_threads;
  }
  /** @brief Caller-requested payload per thread, without alignment slack. */
  nint_t per_thread_bytes() const {
    return _per_thread_bytes;
  }
  /** @brief Region stride in bytes: payload plus alignment slack. */
  nint_t stride() const {
    return _stride;
  }
  /** @brief Total storage size including all padding. */
  nint_t storage_size() const {
    return _storage_size;
  }

private:
  nint_t _num_threads = 0;
  nint_t _per_thread_bytes = 0;
  nint_t _stride = 0;
  nint_t _storage_size = 0;
  execution::WorkspaceArena _storage;
};

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_WORKSPACE_H
