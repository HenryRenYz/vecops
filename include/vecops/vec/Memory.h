#ifndef VECOPS_VEC_MEMORY_H
#define VECOPS_VEC_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/vec/Basic.h"
#include "vecops/vec/Options.h"
#include "vecops/vec/Request.h"

namespace vecops::vec {

namespace details {
struct StridedIndicesOp {};
}

namespace details {
template <VectorTag Tag, bool IsStore, typename... Options>
consteval bool valid_memory_options();
}

/* **************************************************************************** */
//    Contiguous load and store                                           //
/* **************************************************************************** */

struct LoadOp {
  template <VectorTag Tag, typename... Options>
    requires (details::valid_memory_options<Tag, false, Options...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, const ElementOf<Tag>* pointer, Options&&... options) const;

  /// Direct entry with an already-resolved request; skips option parsing.
  template <VectorTag Tag, Active A, Addressing Addr, Populate P,
            typename Alignment, typename Temporality, int IndexScale,
            VectorValue IndexVector, typename Resources>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, const ElementOf<Tag>* pointer,
      LoadRequest<Tag, A, Addr, P, Alignment, Temporality, IndexScale,
                  IndexVector, Resources> request) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, std::initializer_list<ElementOf<Tag>> values) const;
};

struct StoreOp {
  template <VectorTag Tag, typename... Options>
    requires (details::valid_memory_options<Tag, true, Options...>())
  VECOPS_ALWAYS_INLINE void operator()(
      Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      Options&&... options) const;

  /// Direct entry with an already-resolved request; skips option parsing.
  template <VectorTag Tag, Active A, Addressing Addr,
            typename Alignment, typename Temporality, int IndexScale,
            VectorValue IndexVector, typename Resources>
  VECOPS_ALWAYS_INLINE void operator()(
      Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      StoreRequest<Tag, A, Addr, Alignment, Temporality, IndexScale,
                   IndexVector, Resources> request) const;
};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Memory.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Memory.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Memory.h"
#endif

#include "vecops/vec/details/Memory.h"

namespace vecops::vec {

/**
 * Loads consecutive logical lanes from pointer.
 *
 * The default access is unaligned and temporal. `opt::masked(mask)` and
 * `opt::first(count)` filter memory accesses, so inactive addresses are never
 * read. `opt::first(0)` is a no-op that returns zeros/merge without accessing
 * memory. Inactive result lanes are zero unless a scalar/vector `opt::merge`
 * is supplied. `mem::aligned` requires `memory_alignment(tag)` byte alignment.
 * Alignment and temporality are optimization hints: when a requested native
 * form is unavailable, the backend uses the closest safe ordinary load.
 *
 * `opt::indexed(indices)` performs gather loads using i32 or i64 index
 * vectors. `opt::strided(stride)` emits a linear index sequence internally.
 * Indexed and strided addressing are mutually exclusive with alignment.
 *
 * @see store for contiguous stores.
 * @see load_convert for converting loads.
 * @see prefetch for cache-line prefetching.
 */
template <VectorTag Tag, typename... Options>
  requires (details::valid_memory_options<Tag, false, Options...>())
VECOPS_ALWAYS_INLINE Vec<Tag> LoadOp::operator()(
    Tag tag, const ElementOf<Tag>* pointer, Options&&... options) const {
  return details::execute_load_request(
      *this, tag, pointer,
      details::resolve_load_request<Tag>(
          std::forward<Options>(options)...));
}

template <VectorTag Tag, Active A, Addressing Addr, Populate P,
          typename Alignment, typename Temporality, int IndexScale,
          VectorValue IndexVector, typename Resources>
VECOPS_ALWAYS_INLINE Vec<Tag> LoadOp::operator()(
    Tag tag, const ElementOf<Tag>* pointer,
    LoadRequest<Tag, A, Addr, P, Alignment, Temporality, IndexScale,
                IndexVector, Resources> request) const {
  return details::execute_load_request(*this, tag, pointer, request);
}

/** Loads exactly size(tag) values from an initializer list. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LoadOp::operator()(
    Tag tag, std::initializer_list<ElementOf<Tag>> values) const {
  VECOPS_ASSERT(
      values.size() >= static_cast<std::size_t>(size(tag)),
      "insufficient elements: %zd v.s. %zd",
      static_cast<nint_t>(values.size()), size(tag));
  return (*this)(tag, values.begin());
}

/**
 * Stores consecutive logical lanes to pointer.
 *
 * The default access is unaligned and temporal. `opt::masked(mask)` and
 * `opt::first(count)` filter writes, and inactive addresses are untouched.
 * `opt::first(0)` is a no-op. Population options (opt::zero, opt::merge) are
 * invalid because a store has no inactive output lanes. `mem::aligned`
 * requires `memory_alignment(tag)` byte alignment. Alignment and temporality
 * are optimization hints; unsupported forms silently use a safe ordinary
 * store. Indexed and strided addressing are supported as with load.
 *
 * @see load for contiguous loads.
 * @see store_convert for converting stores.
 */
template <VectorTag Tag, typename... Options>
  requires (details::valid_memory_options<Tag, true, Options...>())
VECOPS_ALWAYS_INLINE void StoreOp::operator()(
    Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Options&&... options) const {
  details::execute_store_request(
      *this, tag, pointer, value,
      details::resolve_store_request<Tag>(
          std::forward<Options>(options)...));
}

template <VectorTag Tag, Active A, Addressing Addr,
          typename Alignment, typename Temporality, int IndexScale,
          VectorValue IndexVector, typename Resources>
VECOPS_ALWAYS_INLINE void StoreOp::operator()(
    Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    StoreRequest<Tag, A, Addr, Alignment, Temporality, IndexScale,
                 IndexVector, Resources> request) const {
  details::execute_store_request(*this, tag, pointer, value, request);
}

inline constexpr LoadOp load{};
inline constexpr StoreOp store{};

} // namespace vecops::vec

#endif // VECOPS_VEC_MEMORY_H
