#ifndef VECOPS_NVEC_DETAILS_ELEMENTWISE_H
#define VECOPS_NVEC_DETAILS_ELEMENTWISE_H

#include <type_traits>
#include <utility>

#include "vecops/nvec/details/Wordwise.h"

namespace vecops::nvec::details {

/**
 * Opts an operation into architecture-independent, same-shape word batching.
 *
 * This fallback is appropriate only when every output word depends solely on
 * the input words at the same index. Reductions, cross-word permutations, and
 * operations that change vector shape need a different GenericImpl.
 */
template <typename Op>
struct EnableElementwiseWordBatching : std::false_type {};

/** Opts an operation into same-shape Mask<Tag> word batching. */
template <typename Op>
struct EnableMaskElementwiseWordBatching : std::false_type {};

template <typename Value, VectorTag Tag>
inline constexpr bool is_elementwise_value =
    std::same_as<std::remove_cvref_t<Value>, Vec<Tag>> ||
    std::same_as<std::remove_cvref_t<Value>, Mask<Tag>>;

template <typename Backend, typename Op, VectorTag Tag, typename Value>
struct ElementwiseWordBatch {
 private:
  template <std::size_t Index, typename... Rest>
  static VECOPS_ALWAYS_INLINE decltype(auto) call_word(
      Op op, Tag tag, Value first, Rest... rest) {
    return execute_word<static_cast<nint_t>(Index), Backend>(
        op,
        tag,
        ::vecops::nvec::get_word<static_cast<nint_t>(Index)>(tag, first),
        ::vecops::nvec::get_word<static_cast<nint_t>(Index)>(tag, rest)...);
  }

  template <std::size_t... Index, typename... Rest>
  static VECOPS_ALWAYS_INLINE Value call_words(
      Op op,
      Tag tag,
      Value first,
      std::index_sequence<Index...>,
      Rest... rest) {
    Value result = first;
    ((result = ::vecops::nvec::set_word<static_cast<nint_t>(Index)>(
          tag,
          result,
          call_word<Index>(op, tag, first, rest...))),
     ...);
    return result;
  }

 public:
  template <typename... Rest>
    requires (is_elementwise_value<Rest, Tag> && ...)
  static VECOPS_ALWAYS_INLINE Value call(
      Op op, Tag tag, Value first, Rest&&... rest) {
    return call_words(
        op,
        tag,
        first,
        std::make_index_sequence<static_cast<std::size_t>(
            RepresentationTraits<Backend, Tag>::word_count)>{},
        std::forward<Rest>(rest)...);
  }
};

template <typename Backend, typename Op, VectorTag Tag>
  requires (EnableElementwiseWordBatching<Op>::value ||
            EnableMaskElementwiseWordBatching<Op>::value) &&
           (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, Op, Tag> {
  template <typename... Rest>
    requires EnableElementwiseWordBatching<Op>::value &&
             (is_elementwise_value<Rest, Tag> && ...)
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> first, Rest&&... rest) {
    return ElementwiseWordBatch<Backend, Op, Tag, Vec<Tag>>::call(
        op, tag, first, std::forward<Rest>(rest)...);
  }

  template <typename... Rest>
    requires EnableMaskElementwiseWordBatching<Op>::value &&
             (is_elementwise_value<Rest, Tag> && ...)
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      Op op, Tag tag, Mask<Tag> first, Rest&&... rest) {
    return ElementwiseWordBatch<Backend, Op, Tag, Mask<Tag>>::call(
        op, tag, first, std::forward<Rest>(rest)...);
  }
};

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_ELEMENTWISE_H
