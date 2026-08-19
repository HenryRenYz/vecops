#ifndef VECOPS_VEC_DETAILS_BIT_H
#define VECOPS_VEC_DETAILS_BIT_H

/**
 * @file Bit.h
 * @brief Bitwise operation infrastructure: inactive-lane policy tags
 * (PreserveBitLanes, ZeroBitLanes, MergeBitLanes), options validation
 * (validate_bit_options), and the multi-word GenericImpl fallback for
 * bitwise operations.
 */

#include <type_traits>
#include <utility>

#include "vecops/vec/details/Options.h"
#include "vecops/vec/details/Request.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Inactive-lane policy tags for bitwise operations                          //
/* **************************************************************************** */

/** Policy tag: preserve the first operand's lanes where inactive. */
struct PreserveBitLanes {};
/** Policy tag: set inactive lanes to zero. */
struct ZeroBitLanes {};
/** Policy tag: set inactive lanes from a supplied merge value. */
struct MergeBitLanes {};

template <VectorTag Tag, typename Option>
inline constexpr bool is_bit_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (IsUnmaskedOption<Clean>::value) {
    return true;
  } else if constexpr (is_masked_option<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else if constexpr (is_zero_option<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option<Clean>) {
    return std::same_as<typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option<Clean>) {
    return std::same_as<typename IsScalarMergeOption<Clean>::Value,
                        ElementOf<Tag>>;
  } else {
    return false;
  }
}();

template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> bit_inactive_value(
    Tag tag, Vec<Tag> preserve, Options&&... options) {
  constexpr std::size_t zero_count =
      option_count<IsZeroOption, Options...>;
  constexpr std::size_t vector_count =
      option_count<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_count =
      option_count<IsScalarMergeOption, Options...>;
  if constexpr (zero_count == 1) {
    return zeros(tag);
  } else if constexpr (vector_count == 1) {
    return find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
  } else if constexpr (scalar_count == 1) {
    return fill(tag, find_option<IsScalarMergeOption>(
        std::forward<Options>(options)...).value);
  } else {
    return preserve;
  }
}

template <VectorTag Tag, typename... Options>
consteval void validate_bit_options() {
  static_assert(
      (is_bit_option_for<Tag, Options> && ...),
      "bit operation received an option with the wrong kind or value type");
  constexpr std::size_t masked_count =
      option_count<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  static_assert(
      masked_count + unmasked_count == 1,
      "bit operation requires exactly one opt::masked or opt::unmasked");
  static_assert(
      masked_count * unmasked_count == 0,
      "opt::masked and opt::unmasked are mutually exclusive");
  static_assert(
      option_count<IsZeroOption, Options...> +
          option_count<IsVectorMergeOption, Options...> +
          option_count<IsScalarMergeOption, Options...> <= 1,
      "bit operation accepts at most one zero or merge option");
}

template <typename... Options>
VECOPS_ALWAYS_INLINE constexpr auto bit_inactive_policy() {
  if constexpr (option_count<IsZeroOption, Options...> == 1)
    return ZeroBitLanes{};
  else if constexpr (
      option_count<IsVectorMergeOption, Options...> +
      option_count<IsScalarMergeOption, Options...> == 1)
    return MergeBitLanes{};
  else
    return PreserveBitLanes{};
}

template <typename Op, IntegerTag Tag, typename Count, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_bit_shift_options(
    Op op, Tag tag, Vec<Tag> value, Count count, Options&&... options) {
  validate_bit_options<Tag, Options...>();
  if constexpr (option_count<IsUnmaskedOption, Options...> == 1) {
    return execute(op, tag, value, count);
  } else {
    const auto inactive = bit_inactive_value(
        tag, value, std::forward<Options>(options)...);
    return execute(
        op, tag, value, count,
        find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value,
        inactive, bit_inactive_policy<Options...>());
  }
}

/** Request-driven bit-shift dispatch. */
template <typename Op, IntegerTag Tag, typename Count, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_bit_shift_request(
    Op op, Tag tag, Vec<Tag> value, Count count,
    const OpRequest<Tag, A, I>& request) {
  static_assert(
      A != Active::First,
      "bit operations have no first-count form");
  if constexpr (A == Active::Unmasked) {
    return execute(op, tag, value, count);
  } else if constexpr (I == Inactive::Zero) {
    return execute(
        op, tag, value, count, *request.mask, zeros(tag),
        ZeroBitLanes{});
  } else if constexpr (I == Inactive::MergeScalar) {
    return execute(
        op, tag, value, count, *request.mask,
        fill(tag, request.merge_scalar), MergeBitLanes{});
  } else {
    return execute(
        op, tag, value, count, *request.mask, value,
        PreserveBitLanes{});
  }
}

#define VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY(OpType)                    \
  template <typename Backend, IntegerTag Tag>                            \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)       \
  struct GenericImpl<Backend, OpType, Tag> {                             \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> a, Vec<Tag> b) {                    \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, a),         \
            ::vecops::vec::get_word<Index>(tag, b));                    \
      });                                                                \
    }                                                                    \
    template <typename Policy>                                          \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,     \
        Vec<Tag> inactive, Policy policy) {                              \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, a),         \
            ::vecops::vec::get_word<Index>(tag, b),                     \
            ::vecops::vec::get_word<Index>(tag, mask),                  \
            ::vecops::vec::get_word<Index>(tag, inactive),             \
            policy);                                                     \
      });                                                                \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY(BitAndOp);
VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY(BitOrOp);
VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY(BitXorOp);
VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY(BitAndNotOp);

#undef VECOPS_VEC_DEFINE_GENERIC_BIT_BINARY

template <typename Backend, IntegerTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, BitNotOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      BitNotOp op, Tag tag, Vec<Tag> value) {
    return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {
      return execute_word<Index, Backend>(
          op, parent, ::vecops::vec::get_word<Index>(tag, value));
    });
  }

  template <typename Policy>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      BitNotOp op, Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) {
    return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {
      return execute_word<Index, Backend>(
          op, parent, ::vecops::vec::get_word<Index>(tag, value),
          ::vecops::vec::get_word<Index>(tag, mask),
          ::vecops::vec::get_word<Index>(tag, inactive), policy);
    });
  }
};

#define VECOPS_VEC_DEFINE_GENERIC_SHIFT(OpType)                          \
  template <typename Backend, IntegerTag Tag>                            \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)       \
  struct GenericImpl<Backend, OpType, Tag> {                             \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value, int count) {                 \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, value),    \
            count);                                                      \
      });                                                                \
    }                                                                    \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value, Vec<Tag> counts) {           \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, value),    \
            ::vecops::vec::get_word<Index>(tag, counts));              \
      });                                                                \
    }                                                                    \
    template <typename Policy>                                          \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value, int count, Mask<Tag> mask, \
        Vec<Tag> inactive, Policy policy) {                              \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, value),    \
            count, ::vecops::vec::get_word<Index>(tag, mask),          \
            ::vecops::vec::get_word<Index>(tag, inactive),             \
            policy);                                                     \
      });                                                                \
    }                                                                    \
    template <typename Policy>                                          \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value, Vec<Tag> counts,            \
        Mask<Tag> mask, Vec<Tag> inactive, Policy policy) {             \
      return construct_words<Backend>(tag, [&]<nint_t Index>(Tag parent) {\
        return execute_word<Index, Backend>(                             \
            op, parent, ::vecops::vec::get_word<Index>(tag, value),    \
            ::vecops::vec::get_word<Index>(tag, counts),               \
            ::vecops::vec::get_word<Index>(tag, mask),                 \
            ::vecops::vec::get_word<Index>(tag, inactive), policy);    \
      });                                                                \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_GENERIC_SHIFT(BitShiftLeftOp);
VECOPS_VEC_DEFINE_GENERIC_SHIFT(BitShiftRightOp);

#undef VECOPS_VEC_DEFINE_GENERIC_SHIFT

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_BIT_H
