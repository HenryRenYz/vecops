#ifndef VECOPS_NVEC_VECBASE_H
#define VECOPS_NVEC_VECBASE_H

#include <concepts>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/nvec/Tag.h"
#include "vecops/nvec/details/Backend.h"
#include "vecops/nvec/details/Representation.h"
#include "vecops/nvec/details/scalar/Types.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/nvec/details/x86/Types.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/nvec/details/sve/Types.h"
#endif

namespace vecops::nvec {

namespace details {

template <VectorTag Tag>
using CurrentRepresentation = RepresentationTraits<
    CurrentBackend, std::remove_cvref_t<Tag>>;

template <typename Representation>
VECOPS_ALWAYS_INLINE constexpr nint_t representation_logical_lanes() {
  if constexpr (requires { Representation::logical_lanes(); }) {
    return Representation::logical_lanes();
  } else {
    return Representation::logical_lanes;
  }
}

template <typename Representation>
VECOPS_ALWAYS_INLINE constexpr nint_t representation_word_lanes() {
  if constexpr (requires { Representation::word_lanes(); }) {
    return Representation::word_lanes();
  } else {
    return Representation::word_lanes;
  }
}

} // namespace details

/** Vector representation selected unambiguously from a complete Tag. */
template <VectorTag Tag>
using Vec = typename details::CurrentRepresentation<Tag>::VecType;

/** Mask representation selected unambiguously from a complete Tag. */
template <VectorTag Tag>
using Mask = typename details::CurrentRepresentation<Tag>::MaskType;

template <VectorTag Tag>
using NativeWordVec = typename details::CurrentRepresentation<Tag>::WordVec;

template <VectorTag Tag>
using NativeWordMask = typename details::CurrentRepresentation<Tag>::WordMask;

/** Returns the exact logical lane count described by tag. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t size(Tag = {}) {
  if constexpr (is_fixed_tag<Tag>) {
    // FixedTag is usable as architecture-independent metadata even when the
    // current scalable SVE mode cannot materialize its Vec representation.
    return fixed_lanes<Tag>;
  } else {
    return details::representation_logical_lanes<
        details::CurrentRepresentation<Tag>>();
  }
}

/** Returns the physical lane count in one backend word. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t native_word_size(Tag = {}) {
  return details::representation_word_lanes<
      details::CurrentRepresentation<Tag>>();
}

/** Returns the number of physical backend words used by tag. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t num_words(Tag = {}) {
  return details::CurrentRepresentation<Tag>::word_count;
}

/** Whether size(tag) depends on the runtime SVE vector length. */
template <VectorTag Tag>
inline constexpr bool is_runtime_size =
    details::CurrentRepresentation<Tag>::is_runtime_size;

/** Whether the logical extent occupies only part of its physical word. */
template <VectorTag Tag>
inline constexpr bool is_subword =
    details::CurrentRepresentation<Tag>::is_subword;

/**
 * Returns a physical word from sized Scalar, x86, or VLS representations.
 * Sizeless SVE tuple access is provided by the SVE access layer.
 */
template <nint_t Index, VectorTag Tag>
  requires requires(Vec<Tag> value) {
    details::get_word<Index>(value);
  }
VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_word(
    Tag, Vec<Tag> value) {
  return details::get_word<Index>(value);
}

template <nint_t Index, VectorTag Tag>
  requires requires(Vec<Tag> value, NativeWordVec<Tag> word) {
    details::set_word<Index>(value, word);
  }
VECOPS_ALWAYS_INLINE constexpr Vec<Tag> set_word(
    Tag, Vec<Tag> value, NativeWordVec<Tag> word) {
  return details::set_word<Index>(value, word);
}

/** Returns a physical predicate word from a Mask selected by tag. */
template <nint_t Index, VectorTag Tag>
  requires requires(Mask<Tag> value) {
    details::get_word<Index>(value);
  }
VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_word(
    Tag, Mask<Tag> value) {
  return details::get_word<Index>(value);
}

/** Replaces a physical predicate word in a Mask selected by tag. */
template <nint_t Index, VectorTag Tag>
  requires requires(Mask<Tag> value, NativeWordMask<Tag> word) {
    details::set_word<Index>(value, word);
  }
VECOPS_ALWAYS_INLINE constexpr Mask<Tag> set_word(
    Tag, Mask<Tag> value, NativeWordMask<Tag> word) {
  return details::set_word<Index>(value, word);
}

} // namespace vecops::nvec

#endif // VECOPS_NVEC_VECBASE_H
