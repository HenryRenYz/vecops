#ifndef VECOPS_VEC_DETAILS_ARITHMETIC_H
#define VECOPS_VEC_DETAILS_ARITHMETIC_H

/**
 * @file Arithmetic.h
 * @brief Generic (backend-independent) arithmetic infrastructure:
 * EnableElementwiseWordBatching opt-ins, options validation
 * (execute_*_arithmetic_options), and multi-word GenericImpl fallbacks
 * (ArithmeticGenericImpl, UnaryArithmeticGenericImpl,
 * TernaryArithmeticGenericImpl).
 */

#include "vecops/vec/details/Elementwise.h"
#include "vecops/vec/details/Options.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Word-batching opt-in: all arithmetic ops support same-shape word batch     //
/* **************************************************************************** */

template <>
struct EnableElementwiseWordBatching<AddOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SubOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MulOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<DivOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MinOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MaxOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<NegOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<AbsOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SqrtOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<RcpOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<RsqrtOp> : std::true_type {};

/* **************************************************************************** */
//    Option validation for arithmetic operations                               //
/* **************************************************************************** */

template <VectorTag Tag, typename Option>
inline constexpr bool is_arithmetic_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (IsUnmaskedOption<Clean>::value) {
    return true;
  } else if constexpr (is_masked_option<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else if constexpr (is_zero_option<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option<Clean>) {
    return std::same_as<
        typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option<Clean>) {
    return std::same_as<
        typename IsScalarMergeOption<Clean>::Value, ElementOf<Tag>>;
  } else {
    return false;
  }
}();

/* **************************************************************************** */
//    Multi-word GenericImpl: binary, unary, and ternary arithmetic             //
/* **************************************************************************** */

template <typename Backend, typename Op, VectorTag Tag>
struct ArithmeticGenericImpl {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> a, Vec<Tag> b) {
    return construct_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
          return execute_word<Index, Backend>(
              op,
              tag,
              ::vecops::vec::get_word<Index>(tag, a),
              ::vecops::vec::get_word<Index>(tag, b));
        });
  }

  template <typename Policy>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) {
    return construct_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
          return execute_word<Index, Backend>(
              op,
              tag,
              ::vecops::vec::get_word<Index>(tag, a),
              ::vecops::vec::get_word<Index>(tag, b),
              ::vecops::vec::get_word<Index>(tag, mask),
              ::vecops::vec::get_word<Index>(tag, inactive),
              policy);
        });
  }
};

#define VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(OpType)                    \
  template <typename Backend, VectorTag Tag>                            \
  struct GenericImpl<Backend, OpType, Tag>                              \
      : ArithmeticGenericImpl<Backend, OpType, Tag> {}

VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(AddOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(SubOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MulOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(DivOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MinOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MaxOp);

#undef VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC

template <typename Backend, typename Op, VectorTag Tag>
struct UnaryArithmeticGenericImpl {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> value) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag, ::vecops::vec::get_word<Index>(tag, value));
    });
  }

  template <typename Policy>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag, ::vecops::vec::get_word<Index>(tag, value),
          ::vecops::vec::get_word<Index>(tag, mask),
          ::vecops::vec::get_word<Index>(tag, inactive), policy);
    });
  }
};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, NegOp, Tag>
    : UnaryArithmeticGenericImpl<Backend, NegOp, Tag> {};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, AbsOp, Tag>
    : UnaryArithmeticGenericImpl<Backend, AbsOp, Tag> {};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, SqrtOp, Tag>
    : UnaryArithmeticGenericImpl<Backend, SqrtOp, Tag> {};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, RcpOp, Tag>
    : UnaryArithmeticGenericImpl<Backend, RcpOp, Tag> {};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, RsqrtOp, Tag>
    : UnaryArithmeticGenericImpl<Backend, RsqrtOp, Tag> {};

template <typename Backend, typename Op, VectorTag Tag>
struct TernaryArithmeticGenericImpl {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag,
          ::vecops::vec::get_word<Index>(tag, a),
          ::vecops::vec::get_word<Index>(tag, b),
          ::vecops::vec::get_word<Index>(tag, c));
    });
  }

  template <typename Policy>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,
      Mask<Tag> mask, Vec<Tag> inactive, Policy policy) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag,
          ::vecops::vec::get_word<Index>(tag, a),
          ::vecops::vec::get_word<Index>(tag, b),
          ::vecops::vec::get_word<Index>(tag, c),
          ::vecops::vec::get_word<Index>(tag, mask),
          ::vecops::vec::get_word<Index>(tag, inactive), policy);
    });
  }
};

#define VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(OpType)            \
  template <typename Backend, VectorTag Tag>                             \
  struct GenericImpl<Backend, OpType, Tag>                               \
      : TernaryArithmeticGenericImpl<Backend, OpType, Tag> {}

VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(FmaddOp);
VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(FmsubOp);
VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(FnmaddOp);
VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(FnmsubOp);

#undef VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC

/* **************************************************************************** */
//    Option dispatch: extract mask/merge from variadic pack, validate, invoke  //
/* **************************************************************************** */

/**
 * Collects and validates arithmetic Options, then dispatches to the
 * appropriate execute() overload.  The five policy branches are:
 *   unmasked → plain operation
 *   masked + zero → operation with zero-filled inactive lanes
 *   masked + vector_merge → operation preserving user-supplied inactive vector
 *   masked + scalar_merge → operation with scalar-broadcast inactive lanes
 *   masked (default) → operation preserving first operand's inactive lanes
 */
template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_arithmetic_options(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) {
  static_assert(
      (is_arithmetic_option_for<Tag, Options> && ...),
      "arithmetic received an option with the wrong kind or value type");
  constexpr std::size_t masked_count = option_count<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count<IsScalarMergeOption, Options...>;
  static_assert(
      masked_count + unmasked_count == 1,
      "arithmetic requires exactly one opt::masked or opt::unmasked option");
  static_assert(
      masked_count * unmasked_count == 0,
      "opt::masked and opt::unmasked are mutually exclusive");
  static_assert(zero_count <= 1, "arithmetic accepts at most one opt::zero");
  static_assert(
      vector_merge_count + scalar_merge_count <= 1,
      "arithmetic accepts at most one opt::merge option");
  static_assert(
      zero_count + vector_merge_count + scalar_merge_count <= 1,
      "opt::zero and opt::merge are mutually exclusive");

  if constexpr (unmasked_count == 1) {
    return execute(op, tag, a, b);
  } else if constexpr (zero_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, mask, zeros(tag), ZeroArithmeticInactive{});
  } else if constexpr (vector_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto& inactive = find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, mask, inactive, MergeArithmeticInactive{});
  } else if constexpr (scalar_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto inactive = fill(
        tag,
        find_option<IsScalarMergeOption>(
            std::forward<Options>(options)...).value);
    return execute(
        op, tag, a, b, mask, inactive, MergeArithmeticInactive{});
  } else {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, mask, a, PreserveArithmeticInactive{});
  }
}

/**
 * Same Option validation and dispatch pattern as execute_arithmetic_options,
 * but for unary operations (neg, abs, sqrt, rcp, rsqrt).
 */
template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_unary_arithmetic_options(
    Op op, Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(
      (is_arithmetic_option_for<Tag, Options> && ...),
      "unary arithmetic received an invalid option or value type");
  constexpr std::size_t masked_count = option_count<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count<IsScalarMergeOption, Options...>;
  static_assert(
      masked_count + unmasked_count == 1,
      "unary arithmetic requires exactly one opt::masked or opt::unmasked");
  static_assert(
      masked_count * unmasked_count == 0,
      "opt::masked and opt::unmasked are mutually exclusive");
  static_assert(
      zero_count + vector_merge_count + scalar_merge_count <= 1,
      "unary arithmetic accepts at most one zero or merge population");

  if constexpr (unmasked_count == 1) {
    return execute(op, tag, value);
  } else if constexpr (zero_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, value, mask, zeros(tag), ZeroArithmeticInactive{});
  } else if constexpr (vector_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto& inactive = find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, value, mask, inactive, MergeArithmeticInactive{});
  } else if constexpr (scalar_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto inactive = fill(
        tag, find_option<IsScalarMergeOption>(
                 std::forward<Options>(options)...).value);
    return execute(
        op, tag, value, mask, inactive, MergeArithmeticInactive{});
  } else {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, value, mask, value, PreserveArithmeticInactive{});
  }
}

/**
 * Same Option validation and dispatch as execute_arithmetic_options,
 * adapted for ternary operations (fmadd, fmsub, fnmadd, fnmsub).
 */
template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_ternary_arithmetic_options(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,
    Options&&... options) {
  static_assert(
      (is_arithmetic_option_for<Tag, Options> && ...),
      "ternary arithmetic received an invalid option or value type");
  constexpr std::size_t masked_count = option_count<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count<IsScalarMergeOption, Options...>;
  static_assert(
      masked_count + unmasked_count == 1,
      "ternary arithmetic requires exactly one opt::masked or opt::unmasked");
  static_assert(
      masked_count * unmasked_count == 0,
      "opt::masked and opt::unmasked are mutually exclusive");
  static_assert(
      zero_count + vector_merge_count + scalar_merge_count <= 1,
      "ternary arithmetic accepts at most one zero or merge population");

  if constexpr (unmasked_count == 1) {
    return execute(op, tag, a, b, c);
  } else if constexpr (zero_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, c, mask, zeros(tag), ZeroArithmeticInactive{});
  } else if constexpr (vector_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto& inactive = find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, c, mask, inactive, MergeArithmeticInactive{});
  } else if constexpr (scalar_merge_count == 1) {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto inactive = fill(
        tag, find_option<IsScalarMergeOption>(
                 std::forward<Options>(options)...).value);
    return execute(
        op, tag, a, b, c, mask, inactive, MergeArithmeticInactive{});
  } else {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(
        op, tag, a, b, c, mask, a, PreserveArithmeticInactive{});
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_ARITHMETIC_H
