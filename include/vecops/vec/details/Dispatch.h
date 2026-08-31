#ifndef VECOPS_VEC_DETAILS_DISPATCH_H
#define VECOPS_VEC_DETAILS_DISPATCH_H

#include <type_traits>
#include <utility>

#include "vecops/vec/VecBase.h"

namespace vecops::vec::details {

/**
 * Backend-owned implementation point for an operation and a complete Tag.
 *
 * The primary template deliberately has no members. A backend may specialize
 * either a single physical word or a complete multi-word Tag when the ISA has
 * a better representation-level implementation.
 */
template <typename Backend, typename Op, VectorTag Tag>
struct NativeImpl;

/**
 * Backend-owned implementation point for one physical word of an operation.
 *
 * Implementations expose a templated call<Index>(op, tag, words...) member.
 * Keeping Index and the complete parent Tag separate from the word values is
 * important: neither a vector word nor a mask word can reconstruct all of the
 * original logical-extent information.
 */
template <typename Backend, typename Op>
struct NativeWordImpl;

/** Architecture-independent fallback, including automatic word batching. */
template <typename Backend, typename Op, VectorTag Tag>
struct GenericImpl;

template <typename...>
inline constexpr bool dispatch_dependent_false = false;

#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
// GCC diagnoses attributes on builtin sizeless SVE values when a variadic
// dispatch boundary deduces them as template arguments. The backend retains
// the attributes, so suppress only this known false positive.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

/** Selects a whole-Tag native implementation before word or generic paths. */
template <VectorTag Tag, typename Op, typename... Args>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) execute(
    Op op, Tag tag, Args&&... args) {
  using CleanTag = std::remove_cvref_t<Tag>;
  if constexpr (requires {
                  NativeImpl<CurrentBackend, Op, CleanTag>::call(
                      op, tag, std::forward<Args>(args)...);
                }) {
    return NativeImpl<CurrentBackend, Op, CleanTag>::call(
        op, tag, std::forward<Args>(args)...);
  } else if constexpr (
      RepresentationTraits<CurrentBackend, CleanTag>::word_count == 1 &&
      requires {
        NativeWordImpl<CurrentBackend, Op>::template call<0>(
            op, tag, std::forward<Args>(args)...);
      }) {
    return NativeWordImpl<CurrentBackend, Op>::template call<0>(
        op, tag, std::forward<Args>(args)...);
  } else if constexpr (requires {
                         GenericImpl<CurrentBackend, Op, CleanTag>::call(
                             op, tag, std::forward<Args>(args)...);
                       }) {
    return GenericImpl<CurrentBackend, Op, CleanTag>::call(
        op, tag, std::forward<Args>(args)...);
  } else {
    static_assert(
        dispatch_dependent_false<Op, CleanTag>,
        "operation has neither a native implementation nor a generic fallback for this Tag");
  }
}

/**
 * Executes an operation on one physical word while retaining its parent Tag.
 *
 * Both template arguments are always spelled explicitly: shared batching
 * loops iterate a generic Backend (not CurrentBackend), and backend-internal
 * index-sensitive call sites keep their Index visible.
 */
template <nint_t Index, typename Backend, VectorTag Tag,
          typename Op, typename... Words>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) execute_word(
    Op op, Tag tag, Words&&... words) {
  using CleanTag = std::remove_cvref_t<Tag>;
  if constexpr (requires {
                  NativeWordImpl<Backend, Op>::template call<Index>(
                      op, tag, std::forward<Words>(words)...);
                }) {
    return NativeWordImpl<Backend, Op>::template call<Index>(
        op, tag, std::forward<Words>(words)...);
  } else {
    static_assert(
        dispatch_dependent_false<Backend, Op, CleanTag>,
        "operation has no native word implementation for this Tag");
  }
}

#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
#pragma GCC diagnostic pop
#endif

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_DISPATCH_H
