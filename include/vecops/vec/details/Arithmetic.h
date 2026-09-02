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
#include "vecops/vec/Options.h"
#include "vecops/vec/details/Request.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Word-batching opt-in: all arithmetic ops support same-shape word batch     //
/* **************************************************************************** */

template <>
struct EnableElementwiseWordBatching<AddOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SubOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SaturatedAddOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SaturatedSubOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MulOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<DivOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MinOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<MaxOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<CopySignOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<NegOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<AbsOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<SqrtOp> : std::true_type {};

/* **************************************************************************** */
//    Option validation for arithmetic operations                               //
/* **************************************************************************** */

template <VectorTag Tag, typename Option>
inline constexpr bool is_arithmetic_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_unmasked_option_v<Clean>) {
    return true;
  } else if constexpr (is_masked_option_v<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else if constexpr (is_zero_option_v<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option_v<Clean>) {
    return std::same_as<
        typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option_v<Clean>) {
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
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(SaturatedAddOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(SaturatedSubOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MulOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(DivOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MinOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(MaxOp);
VECOPS_VEC_DEFINE_ARITHMETIC_GENERIC(CopySignOp);

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
VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC(ClampOp);

#undef VECOPS_VEC_DEFINE_TERNARY_ARITHMETIC_GENERIC

/* **************************************************************************** */
//    Backend-shared FMA synthesis fallback                                    //
/* **************************************************************************** */

/**
 * Synthesizes one FMA variant on a single physical word from the backend's
 * word-level Mul/Add/Sub/Fill ops. Used by backends whose instruction tier
 * has no native fused op for the element format. FnmsubOp preserves IEEE
 * signed zero by computing (-0) - product - c instead of flipping the
 * product's sign bit.
 */
template <typename Backend, typename Op, nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> synthesize_fma_word(
    Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
    NativeWordVec<Tag> c) {
  const auto product = NativeWordImpl<Backend, MulOp>::template call<Index>(
      MulOp{}, tag, a, b);
  if constexpr (std::same_as<Op, FmaddOp>)
    return NativeWordImpl<Backend, AddOp>::template call<Index>(
        AddOp{}, tag, product, c);
  else if constexpr (std::same_as<Op, FmsubOp>)
    return NativeWordImpl<Backend, SubOp>::template call<Index>(
        SubOp{}, tag, product, c);
  else if constexpr (std::same_as<Op, FnmaddOp>)
    return NativeWordImpl<Backend, SubOp>::template call<Index>(
        SubOp{}, tag, c, product);
  else if constexpr (std::same_as<Op, FnmsubOp>) {
    using T = ElementOf<Tag>;
    const T negative_zero = [] {
      if constexpr (is_float_v<T>) return static_cast<T>(-0.0F);
      else return T{};
    }();
    const auto zero = NativeWordImpl<Backend, FillOp>::template call<Index>(
        FillOp{}, tag, negative_zero);
    const auto negative_product =
        NativeWordImpl<Backend, SubOp>::template call<Index>(
            SubOp{}, tag, zero, product);
    return NativeWordImpl<Backend, SubOp>::template call<Index>(
        SubOp{}, tag, negative_product, c);
  } else
    static_assert(dispatch_dependent_false<Op>, "unsupported FMA variant");
}

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
      (is_arithmetic_option_for_v<Tag, Options> && ...),
      "arithmetic received an option with the wrong kind or value type");
  constexpr std::size_t masked_count = option_count_v<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count_v<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count_v<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count_v<IsScalarMergeOption, Options...>;
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

/** Invokes f with every integer-overflow option removed. */
template <typename F>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_arithmetic_overflow(
    F&& f) {
  return std::forward<F>(f)();
}

template <typename F, typename First, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_arithmetic_overflow(
    F&& f, First&& first, Rest&&... rest) {
  if constexpr (is_arithmetic_overflow_option_v<First>) {
    return invoke_without_arithmetic_overflow(
        std::forward<F>(f), std::forward<Rest>(rest)...);
  } else {
    return invoke_without_arithmetic_overflow(
        [&f, &first]<typename... Tail>(Tail&&... tail) -> decltype(auto) {
          return std::forward<F>(f)(
              std::forward<First>(first),
              std::forward<Tail>(tail)...);
        },
        std::forward<Rest>(rest)...);
  }
}

/** Returns the selected add/sub overflow mode; wrap is the default. */
template <typename... Options>
consteval OverflowMode selected_arithmetic_overflow_mode() {
  OverflowMode result = OverflowMode::Wrap;
  ([&]() VECOPS_INLINE_LAMBDA {
    using Option = std::remove_cvref_t<Options>;
    if constexpr (is_arithmetic_overflow_option_v<Option>)
      result = IsArithmeticOverflowOption<Option>::mode;
  }(), ...);
  return result;
}

/**
 * Selects wrap or saturating add/sub, removes that semantic option, then
 * reuses the ordinary mask/population dispatcher. A policy-only call is an
 * ordinary unmasked operation.
 */
template <typename WrapOp, typename SaturatedOp, VectorTag Tag,
          typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_add_sub_options(
    WrapOp wrap_op, SaturatedOp saturated_op, Tag tag,
    Vec<Tag> a, Vec<Tag> b, Options&&... options) {
  static_assert(
      ((is_arithmetic_option_for_v<Tag, Options> ||
        is_arithmetic_overflow_option_v<Options>) && ...),
      "add/sub received an option with the wrong kind or value type");
  constexpr std::size_t overflow_count =
      option_count_v<IsArithmeticOverflowOption, Options...>;
  static_assert(
      overflow_count <= 1,
      "add/sub accepts at most one opt::wrap or opt::saturate");
  if constexpr (overflow_count != 0) {
    static_assert(
        std::integral<ElementOf<Tag>>,
        "opt::wrap and opt::saturate require an integer element type");
  }

  constexpr OverflowMode mode =
      selected_arithmetic_overflow_mode<Options...>();
  const auto dispatch = [&]<typename... ArithmeticOptions>(
                            ArithmeticOptions&&... arithmetic_options)
      -> Vec<Tag> {
    if constexpr (sizeof...(ArithmeticOptions) == 0) {
      if constexpr (mode == OverflowMode::Saturate)
        return execute(saturated_op, tag, a, b);
      else
        return execute(wrap_op, tag, a, b);
    } else {
      if constexpr (mode == OverflowMode::Saturate)
        return execute_arithmetic_options(
            saturated_op, tag, a, b,
            std::forward<ArithmeticOptions>(arithmetic_options)...);
      else
        return execute_arithmetic_options(
            wrap_op, tag, a, b,
            std::forward<ArithmeticOptions>(arithmetic_options)...);
    }
  };
  return invoke_without_arithmetic_overflow(
      dispatch, std::forward<Options>(options)...);
}

/** Request-driven binary elementwise dispatch. */
template <typename Op, VectorTag Tag, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_arithmetic_request(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b,
    const OpRequest<Tag, A, I>& request) {
  static_assert(
      A != Active::First,
      "elementwise operations have no first-count form");
  if constexpr (A == Active::Unmasked) {
    return execute(op, tag, a, b);
  } else if constexpr (I == Inactive::Zero) {
    return execute(
        op, tag, a, b, *request.mask, zeros(tag),
        ZeroArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeVector) {
    return execute(
        op, tag, a, b, *request.mask, *request.merge_vector,
        MergeArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeScalar) {
    return execute(
        op, tag, a, b, *request.mask, fill(tag, request.merge_scalar),
        MergeArithmeticInactive{});
  } else {
    return execute(
        op, tag, a, b, *request.mask, a, PreserveArithmeticInactive{});
  }
}

/**
 * Same Option validation and dispatch pattern as execute_arithmetic_options,
 * but for unary operations (neg, abs, sqrt).
 */
template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_unary_arithmetic_options(
    Op op, Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(
      (is_arithmetic_option_for_v<Tag, Options> && ...),
      "unary arithmetic received an invalid option or value type");
  constexpr std::size_t masked_count = option_count_v<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count_v<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count_v<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count_v<IsScalarMergeOption, Options...>;
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
      (is_arithmetic_option_for_v<Tag, Options> && ...),
      "ternary arithmetic received an invalid option or value type");
  constexpr std::size_t masked_count = option_count_v<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count_v<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count_v<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count_v<IsScalarMergeOption, Options...>;
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

/** Request-driven ternary elementwise dispatch (FMA family). */
template <typename Op, VectorTag Tag, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_ternary_arithmetic_request(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,
    const OpRequest<Tag, A, I>& request) {
  static_assert(
      A != Active::First,
      "elementwise operations have no first-count form");
  if constexpr (A == Active::Unmasked) {
    return execute(op, tag, a, b, c);
  } else if constexpr (I == Inactive::Zero) {
    return execute(
        op, tag, a, b, c, *request.mask, zeros(tag),
        ZeroArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeVector) {
    return execute(
        op, tag, a, b, c, *request.mask, *request.merge_vector,
        MergeArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeScalar) {
    return execute(
        op, tag, a, b, c, *request.mask, fill(tag, request.merge_scalar),
        MergeArithmeticInactive{});
  } else {
    return execute(
        op, tag, a, b, c, *request.mask, a, PreserveArithmeticInactive{});
  }
}

/** Request-driven unary elementwise dispatch. */
template <typename Op, VectorTag Tag, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_unary_arithmetic_request(
    Op op, Tag tag, Vec<Tag> value,
    const OpRequest<Tag, A, I>& request) {
  static_assert(
      A != Active::First,
      "elementwise operations have no first-count form");
  if constexpr (A == Active::Unmasked) {
    return execute(op, tag, value);
  } else if constexpr (I == Inactive::Zero) {
    return execute(
        op, tag, value, *request.mask, zeros(tag),
        ZeroArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeVector) {
    return execute(
        op, tag, value, *request.mask, *request.merge_vector,
        MergeArithmeticInactive{});
  } else if constexpr (I == Inactive::MergeScalar) {
    return execute(
        op, tag, value, *request.mask, fill(tag, request.merge_scalar),
        MergeArithmeticInactive{});
  } else {
    return execute(
        op, tag, value, *request.mask, value,
        PreserveArithmeticInactive{});
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_ARITHMETIC_H
