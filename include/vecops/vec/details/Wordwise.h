#ifndef VECOPS_VEC_DETAILS_WORDWISE_H
#define VECOPS_VEC_DETAILS_WORDWISE_H

/**
 * @file Wordwise.h
 * @brief Multi-word construction helpers: visit_runtime_word (expand a
 * runtime word index into a compile-time instantiation), construct_words
 * (iterate over physical words to build a Vec), and construct_mask_words
 * (same for Mask).
 *
 * The continuation-passing callback pattern used by construct_words is
 * required for sizeless SVE representations where words are stored in
 * tuple types (svfloat32x2_t etc.) and cannot be individually default-
 * constructed in an array.
 */

#include <cassert>
#include <type_traits>
#include <utility>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                        Runtime word index expansion                        //
/* **************************************************************************** */

/**
 * Lanes of word @p Index that lie inside the tag's logical lane count,
 * clamped to [0, word_lanes]. Backends whose representation completes every
 * physical word (for example non-subword scalable SVE tuples) override this
 * with a backend-specific fast path.
 */
template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t valid_word_lanes(Tag = {}) {
  const nint_t word_lanes = native_word_size(Tag{});
  const nint_t remaining = size(Tag{}) - Index * word_lanes;
  return remaining < 0 ? 0
                       : (remaining > word_lanes ? word_lanes : remaining);
}

template <nint_t Index, nint_t Count, typename Visitor>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) visit_runtime_word_impl(
    nint_t ordinal, Visitor&& visitor) {
  static_assert(Count > 0);
  static_assert(Index >= 0 && Index < Count);
  if (ordinal == Index) {
    return std::forward<Visitor>(visitor).template operator()<Index>();
  }
  if constexpr (Index + 1 < Count) {
    return visit_runtime_word_impl<Index + 1, Count>(
        ordinal, std::forward<Visitor>(visitor));
  } else {
    assert(false && "word ordinal is outside the representation");
    VECOPS_UNREACHABLE();
  }
}

/**
 * Invokes visitor.template operator()<Index>() for a runtime word ordinal.
 *
 * All Index instantiations must return the same type and value category. The
 * ordinal must be in [0, RepresentationTraits<Backend, Tag>::word_count).
 * This bridge is intended for lane APIs whose lane index is runtime data but
 * whose underlying get_word/set_word primitives require a compile-time index.
 */
template <typename Backend, VectorTag Tag, typename Visitor>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) visit_runtime_word(
    Tag, nint_t ordinal, Visitor&& visitor) {
  constexpr nint_t count = RepresentationTraits<Backend, Tag>::word_count;
  assert(ordinal >= 0 && ordinal < count);
  if constexpr (count == 1) {
    // Single-word fast path: no runtime dispatch, so the compiler cannot
    // fold the word chain into a predicated vector select. Some bf16
    // instruction selectors cannot lower such selects.
    return std::forward<Visitor>(visitor).template operator()<0>();
  } else {
    return visit_runtime_word_impl<0, count>(
        ordinal, std::forward<Visitor>(visitor));
  }
}

/** Backend hook for representations that cannot be default-constructed. */
template <typename Backend>
struct ConstructWordsHook;

/** Backend hook for predicate representations that cannot be initialized. */
template <typename Backend>
struct ConstructMaskWordsHook;

template <typename Value, VectorTag Tag, typename Builder,
          std::size_t... Index>
VECOPS_ALWAYS_INLINE constexpr Value construct_sized_value(
    Tag tag, Builder& builder, std::index_sequence<Index...>) {
  if constexpr (is_word_array_v<Value>) {
    // Keep this as one aggregate initialization. Repeated
    // `result = set_word(result, word)` creates partially-covered whole-value
    // copies; GCC 13 then fails SRA for four 64-byte words and spills them.
    // `--param=sra-max-scalarization-size-Ospeed=256` masks that failure, but
    // the construction itself must not depend on a compiler threshold.
    return Value{{
        builder.template operator()<static_cast<nint_t>(Index)>(tag)...}};
  } else {
    static_assert(sizeof...(Index) == 1);
    return (builder.template operator()<static_cast<nint_t>(Index)>(tag), ...);
  }
}

/**
 * Constructs a vector by requesting each physical word from builder.
 *
 * The default path supports representations that can be value-initialized.
 * Sizeless tuple representations must provide ConstructWordsHook<Backend>;
 * their first tuple value needs an ISA-specific creation operation and cannot
 * be expressed safely as a generic set_word sequence.
 */
template <typename Backend, VectorTag Tag, typename Builder>
VECOPS_ALWAYS_INLINE constexpr auto construct_words(
    Tag tag, Builder&& builder)
    -> typename RepresentationTraits<Backend, Tag>::VecType {
  using Traits = RepresentationTraits<Backend, Tag>;
  using Value = typename Traits::VecType;
  auto&& builder_ref = builder;
  if constexpr (requires {
                  ConstructWordsHook<Backend>::template call<Tag>(
                      tag, builder_ref);
                }) {
    return ConstructWordsHook<Backend>::template call<Tag>(tag, builder_ref);
  } else if constexpr (requires { Value{}; }) {
    return construct_sized_value<Value>(
        tag,
        builder_ref,
        std::make_index_sequence<
            static_cast<std::size_t>(Traits::word_count)>{});
  } else {
    static_assert(
        dispatch_dependent_false<Backend, Tag>,
        "sizeless vector construction requires a backend ConstructWordsHook");
  }
}

/** Constructs a mask from independently produced physical predicate words. */
template <typename Backend, VectorTag Tag, typename Builder>
VECOPS_ALWAYS_INLINE constexpr auto construct_mask_words(
    Tag tag, Builder&& builder)
    -> typename RepresentationTraits<Backend, Tag>::MaskType {
  using Traits = RepresentationTraits<Backend, Tag>;
  using Value = typename Traits::MaskType;
  auto&& builder_ref = builder;
  if constexpr (requires {
                  ConstructMaskWordsHook<Backend>::template call<Tag>(
                      tag, builder_ref);
                }) {
    return ConstructMaskWordsHook<Backend>::template call<Tag>(
        tag, builder_ref);
  } else if constexpr (requires { Value{}; }) {
    return construct_sized_value<Value>(
        tag,
        builder_ref,
        std::make_index_sequence<
            static_cast<std::size_t>(Traits::word_count)>{});
  } else {
    static_assert(
        dispatch_dependent_false<Backend, Tag>,
        "sizeless mask construction requires a backend hook");
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_WORDWISE_H
