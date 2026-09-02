//
// Created by renyz on 2026/7/8.
//

#ifndef VECOPS_KERNEL_LOOP_H
#define VECOPS_KERNEL_LOOP_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/CoreDefs.h"
#include "vecops/Assertion.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

/**
 * @file Loop.h
 * @brief Higher-order traversal helpers for Tensor-like nested iteration.
 *
 * This header defines small higher-order traversal primitives for applying a
 * callable to slices of one or more `Tensor` objects. The traversal dimensions
 * are selected at compile time, while runtime sizes are read from the input
 * Tensor layouts.
 *
 * ## Key components
 *
 * | Component                         | Purpose                                   |
 * |-----------------------------------|-------------------------------------------|
 * | `loop::for_each<Is...>`            | Traverse the listed logical dimensions    |
 * | `loop::for_each_dims<N>`           | Traverse logical dimensions `0..N - 1`    |
 * | `loop::for_each_with_index<Is...>` | Traverse and pass current indices to `fn` |
 * | `loop::for_each_dims_with_index<N>`| Traverse `0..N - 1` and pass indices      |
 * | `loop::*_with_index_tuple`         | Pass indices as one tuple before slices   |
 * | `loop::fold`                       | Vector-block fold with typed carries      |
 *
 * ## Logical dimensions and alignment
 *
 * Inputs are interpreted under trailing-dimension alignment, matching the
 * usual NumPy/PyTorch broadcasting convention. If the traversal logical rank is
 * `R` and a Tensor has rank `T`, the Tensor dimension corresponding to logical
 * dimension `I` is:
 *
 * @code
 * actual_dim = I - (R - T)
 * @endcode
 *
 * If `actual_dim` is outside `[0, T)`, that Tensor has no dimension at the
 * current logical axis and is forwarded unchanged into the recursive traversal.
 *
 * For `for_each<Is...>`, the logical rank is:
 *
 * @code
 * max(max(Is...) + 1, max_rank(inputs...))
 * @endcode
 *
 * For `for_each_dims<N>`, the selected dimensions are `0..N - 1`; the
 * logical rank is chosen the same way as `for_each<0, ..., N - 1>`, namely
 * large enough to include both the selected dimensions and all Tensor inputs.
 *
 * ## Broadcasting
 *
 * Only dimensions whose compile-time shape type is exactly `Const<1>` use
 * broadcast semantics. Those dimensions are always sliced at index 0 and do
 * not contribute to the loop extent. Runtime size-one dimensions such as
 * `Any{1}` are **not** treated as broadcast dimensions.
 *
 * Non-broadcast Tensor dimensions at the same logical axis must have the same
 * runtime extent. The check uses `VECOPS_ASSERT`, so it is intended to have no
 * release-build cost when assertions are disabled.
 *
 * Non-Tensor inputs have rank 0. They never affect extents and are forwarded
 * unchanged to every invocation of the user callable.
 *
 * ## Slicing behavior
 *
 * At each traversed axis, a Tensor is sliced with an integer index on the
 * selected dimension and `reserve` on all other dimensions. Therefore:
 *
 * - Traversing every dimension of a Tensor eventually passes element
 *   references to the callable.
 * - Traversing only some dimensions passes sub-Tensor views for the remaining
 *   dimensions.
 * - The sub-Tensor shapes passed for different inputs do not need to match.
 * - Writable Tensor inputs may be updated through the scalar references or
 *   sub-Tensor views passed to the callable.
 *
 * Tensor Specs and DataAccess sessions participate through the same logical
 * slicing operation. Slicing a Spec composes its CoordinateProjection so a
 * coordinate-aware transform still sees the original rank. Slicing DataAccess
 * returns a borrowed view: it can load/store, but it cannot commit a
 * materialized output. The parent session retains commit responsibility and
 * must outlive the traversal.
 *
 * ## Vector-block fold
 *
 * `fold` grows one base Tag into full and tail multi-word Tags. A callback
 * therefore sees one wide block. Reduction and invariant carries are created
 * as independent automatic variables, so the helper also supports sizeless
 * SVE vectors.
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/kernel/Loop.h"
 * using namespace vecops::kernel;
 *
 * std::vector<float> a(2 * 3);
 * std::vector<float> b(3);
 * std::vector<float> out(2 * 3);
 *
 * auto ta = make_tensor(a.data(), make_shape(cint<2>, cint<3>),
 *                       make_strides(cint<3>, cint<1>));
 * auto tb = make_tensor(b.data(), make_shape(cint<3>), make_strides(cint<1>));
 * auto to = make_tensor(out.data(), make_shape(cint<2>, cint<3>),
 *                       make_strides(cint<3>, cint<1>));
 *
 * // Traverses logical dimensions 0 and 1. `tb` is trailing-aligned with the
 * // last dimension and is reused for both rows.
 * loop::for_each_dims<2>([](auto&& dst, auto&& x, auto&& y) {
 *   dst = x + y;
 * }, to, ta, tb);
 *
 * // Traverse rows only. The callable receives rank-1 row views.
 * loop::for_each<0>([](auto&& row) {
 *   row(0) = 0;
 * }, to);
 * @endcode
 *
 * ## Pitfalls and limitations
 *
 * - Traversal dimensions must be non-negative and unique; violations are
 *   compile-time errors.
 * - `for_each_dims<N>` is shorthand for traversing dimensions `0..N - 1`;
 *   Tensor inputs may have higher rank, in which case the untraversed trailing
 *   dimensions are forwarded as sub-Tensor views.
 * - Broadcast is type-based (`Const<1>`), not value-based (`Any{1}`).
 * - Use the `_with_index` variants when the callable needs current traversal
 *   indices. The indices are passed before the sliced inputs.
 * - The helpers preserve `Tensor`'s non-owning view semantics. The underlying
 *   storage must outlive the traversal and must be mutable if the callable
 *   writes through the passed objects.
 * - A borrowed DataAccess view must not escape the parent operand session.
 *   Commit the owning output, not an individual loop slice.
 * - Traversal order follows `Is...`, and each selected dimension is removed
 *   before the next one is interpreted. Use the documented logical trailing
 *   alignment rather than assuming indices refer to the original local rank.
 */

namespace vecops::kernel::loop {

using namespace ::vecops::meta;
using namespace ::vecops::tensor;

enum class TailCarryPolicy {
  independent,
  reuse_prefix,
};

namespace details {

template <typename T>
struct ConstValue : std::false_type {
  static constexpr nint_t value = 0;
};

template <nint_t N>
struct ConstValue<Const<N>> : std::true_type {
  static constexpr nint_t value = N;
};

template <typename T>
inline constexpr bool is_const_value_v = ConstValue<std::remove_cvref_t<T>>::value;

// ======================== Vector Block Traversal ========================

template <::vecops::vec::VectorTag Tag, int Power>
struct GrowVectorTag {
  using type = typename GrowVectorTag<
      ::vecops::vec::Twice<Tag>, Power - 1>::type;
};

template <::vecops::vec::VectorTag Tag>
struct GrowVectorTag<Tag, 0> {
  using type = Tag;
};

template <::vecops::vec::VectorTag Tag, int Factor,
          bool Scalable = ::vecops::vec::is_scalable_tag_v<Tag>>
struct SuggestedFactorTag;

template <::vecops::vec::VectorTag Tag, int Factor>
struct SuggestedFactorTag<Tag, Factor, true> {
  static_assert(Factor > 0, "vector loop factor must be positive");
  static_assert(
      ::vecops::vec::scale_power_v<Tag> <= VEC_MAX_POW,
      "base vector tag exceeds the backend multi-word limit");
  static constexpr int requested_power = ::vecops::log2_floor(Factor);
  static constexpr int available_power =
      VEC_MAX_POW - ::vecops::vec::scale_power_v<Tag>;
  static constexpr int actual_power =
      requested_power < available_power ? requested_power : available_power;
  using type = typename GrowVectorTag<Tag, actual_power>::type;
};

template <::vecops::vec::VectorTag Tag, int Factor>
struct SuggestedFactorTag<Tag, Factor, false> {
  static_assert(Factor > 0, "vector loop factor must be positive");
  static constexpr int requested_power = ::vecops::log2_floor(Factor);
  static constexpr int actual_power = []() consteval {
    constexpr nint_t max_word_lanes =
        static_cast<nint_t>(MAX_VEC_WIDTH / 8) /
        static_cast<nint_t>(sizeof(::vecops::vec::ElementOf<Tag>));
    constexpr nint_t max_lanes = max_word_lanes << VEC_MAX_POW;
    nint_t lanes = ::vecops::vec::fixed_lanes_v<Tag>;
    int power = 0;
    while (power < requested_power && lanes <= max_lanes / 2) {
      lanes *= 2;
      ++power;
    }
    return power;
  }();
  using type = typename GrowVectorTag<Tag, actual_power>::type;
};

template <::vecops::vec::VectorTag Tag, int Factor>
using suggested_factor_tag_t = typename SuggestedFactorTag<Tag, Factor>::type;

template <::vecops::vec::VectorTag ParentTag,
          ::vecops::vec::VectorTag ChildTag>
VECOPS_ALWAYS_INLINE auto vector_prefix(
    ParentTag parent_tag, ChildTag child_tag,
    ::vecops::vec::Vec<ParentTag> value) {
  if constexpr (std::same_as<ParentTag, ChildTag>) {
    return value;
  } else {
    using HalfTag = ::vecops::vec::Half<ParentTag>;
    return vector_prefix(
        HalfTag{}, child_tag,
        ::vecops::vec::lower(parent_tag, value));
  }
}

template <::vecops::vec::VectorTag ParentTag,
          ::vecops::vec::VectorTag ChildTag>
VECOPS_ALWAYS_INLINE ::vecops::vec::Vec<ParentTag> replace_vector_prefix(
    ParentTag parent_tag, ChildTag child_tag,
    ::vecops::vec::Vec<ParentTag> value,
    ::vecops::vec::Vec<ChildTag> prefix) {
  if constexpr (std::same_as<ParentTag, ChildTag>) {
    return prefix;
  } else {
    using HalfTag = ::vecops::vec::Half<ParentTag>;
    auto lower = ::vecops::vec::lower(parent_tag, value);
    lower = replace_vector_prefix(HalfTag{}, child_tag, lower, prefix);
    return ::vecops::vec::concat(
        parent_tag, lower, ::vecops::vec::upper(parent_tag, value));
  }
}

template <typename T, typename Reduction>
struct VectorReductionDefinition {
  T& result;
  using ReductionType = Reduction;
};

template <typename T>
struct VectorInvariantDefinition {
  const T& value;
};

template <typename T>
struct IsVectorReductionDefinition : std::false_type {};

template <typename T, typename Reduction>
struct IsVectorReductionDefinition<VectorReductionDefinition<T, Reduction>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_vector_reduction_definition_v =
    IsVectorReductionDefinition<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsVectorInvariantDefinition : std::false_type {};

template <typename T>
struct IsVectorInvariantDefinition<VectorInvariantDefinition<T>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_vector_invariant_definition_v =
    IsVectorInvariantDefinition<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_vector_carry_definition_v =
    is_vector_reduction_definition_v<T> ||
    is_vector_invariant_definition_v<T>;

template <typename Reduction, ::vecops::vec::VectorTag Tag>
VECOPS_ALWAYS_INLINE ::vecops::vec::Vec<Tag> combine_reduction_carries(
    Tag, ::vecops::vec::Vec<Tag> lhs,
    ::vecops::vec::Vec<Tag> rhs) {
  if constexpr (std::same_as<Reduction, ::vecops::vec::ReduceAddOp>) {
    return ::vecops::vec::add(lhs, rhs);
  } else if constexpr (
      std::same_as<Reduction, ::vecops::vec::ReduceMaxOp>) {
    return ::vecops::vec::max(lhs, rhs);
  } else {
    static_assert(
        std::same_as<Reduction, ::vecops::vec::ReduceMinOp>,
        "unsupported vector loop reduction");
    return ::vecops::vec::min(lhs, rhs);
  }
}

template <::vecops::vec::VectorTag ToTag,
          ::vecops::vec::VectorTag FromTag>
VECOPS_ALWAYS_INLINE ::vecops::vec::Vec<ToTag> repeat_invariant_vector(
    ToTag to, FromTag from, ::vecops::vec::Vec<FromTag> value) {
  if constexpr (std::same_as<ToTag, FromTag>) {
    return value;
  } else {
    using HalfTag = ::vecops::vec::Half<ToTag>;
    auto half = repeat_invariant_vector(HalfTag{}, from, value);
    return ::vecops::vec::concat(to, half, half);
  }
}

template <::vecops::vec::VectorTag FullTag,
          ::vecops::vec::VectorTag BaseTag, typename Definition>
VECOPS_ALWAYS_INLINE auto make_full_invariant(
    FullTag full_tag, BaseTag base_tag, const Definition& definition) {
  using Value = std::remove_cvref_t<decltype(definition.value)>;
  if constexpr (std::same_as<Value, ::vecops::vec::ElementOf<BaseTag>>) {
    return ::vecops::vec::fill(full_tag, definition.value);
  } else {
    static_assert(
        std::same_as<Value, ::vecops::vec::Vec<BaseTag>>,
        "loop::invariant requires ElementOf<Tag> or Vec<Tag>");
    return repeat_invariant_vector(
        full_tag, base_tag, definition.value);
  }
}

#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
// GCC diagnoses the attributes on sizeless SVE types when they are deduced
// through these variadic carry helpers, even though the attributes are
// preserved by the underlying builtin types. Keep the workaround local so
// unrelated attribute diagnostics remain visible.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

template <std::size_t Wanted, std::size_t Current = 0,
          typename Fn, typename First, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_vector_carry(
    Fn&& fn, First& first, Rest&... rest) {
  if constexpr (Wanted == Current) {
    return std::forward<Fn>(fn)(first);
  } else {
    static_assert(sizeof...(Rest) > 0, "vector carry index out of range");
    return invoke_vector_carry<Wanted, Current + 1>(
        std::forward<Fn>(fn), rest...);
  }
}

template <std::size_t I, typename DefinitionTuple,
          ::vecops::vec::VectorTag BaseTag,
          ::vecops::vec::VectorTag FullTag,
          typename Use, typename... FullCarries>
VECOPS_ALWAYS_INLINE void with_vector_full_carries(
    DefinitionTuple& definitions, BaseTag base_tag, FullTag full_tag,
    Use& use, FullCarries&... full_carries) {
  if constexpr (I == std::tuple_size_v<std::remove_reference_t<DefinitionTuple>>) {
    use(full_carries...);
  } else {
    auto&& definition = std::get<I>(definitions);
    using Definition = std::remove_cvref_t<decltype(definition)>;
    if constexpr (is_vector_reduction_definition_v<Definition>) {
      using Reduction = typename Definition::ReductionType;
      using Result = std::remove_cvref_t<decltype(definition.result)>;
      static_assert(
          std::same_as<Result, ::vecops::vec::ElementOf<BaseTag>>,
          "vector reduction result must be ElementOf<Tag>");
      auto carry = ::vecops::vec::fill(
          full_tag,
          ::vecops::vec::details::reduction_identity<
              Reduction, ::vecops::vec::ElementOf<BaseTag>>());
      with_vector_full_carries<I + 1>(
          definitions, base_tag, full_tag, use,
          full_carries..., carry);
      definition.result = Reduction{}(full_tag, carry);
    } else {
      static_assert(is_vector_invariant_definition_v<Definition>);
      auto carry = make_full_invariant(full_tag, base_tag, definition);
      const auto& const_carry = carry;
      with_vector_full_carries<I + 1>(
          definitions, base_tag, full_tag, use,
          full_carries..., const_carry);
    }
  }
}

template <std::size_t I, TailCarryPolicy Policy,
          typename DefinitionTuple,
          ::vecops::vec::VectorTag FullTag,
          ::vecops::vec::VectorTag TailTag,
          typename FullDispatch, typename Use,
          typename... TailCarries>
VECOPS_ALWAYS_INLINE void with_vector_tail_carries(
    DefinitionTuple& definitions, FullTag full_tag, TailTag tail_tag,
    FullDispatch& full_dispatch, Use& use,
    TailCarries&... tail_carries) {
  if constexpr (I == std::tuple_size_v<std::remove_reference_t<DefinitionTuple>>) {
    use(tail_carries...);
  } else {
    auto&& definition = std::get<I>(definitions);
    using Definition = std::remove_cvref_t<decltype(definition)>;
    full_dispatch.template operator()<I>(
        [&](auto& full_carry) VECOPS_INLINE_LAMBDA {
          if constexpr (is_vector_reduction_definition_v<Definition>) {
            using Reduction = typename Definition::ReductionType;
            auto tail_carry = [&]() VECOPS_INLINE_LAMBDA {
              if constexpr (Policy == TailCarryPolicy::reuse_prefix) {
                return vector_prefix(full_tag, tail_tag, full_carry);
              } else {
                return ::vecops::vec::fill(
                    tail_tag,
                    ::vecops::vec::details::reduction_identity<
                        Reduction, ::vecops::vec::ElementOf<TailTag>>());
              }
            }();
            with_vector_tail_carries<I + 1, Policy>(
                definitions, full_tag, tail_tag,
                full_dispatch, use, tail_carries..., tail_carry);
            if constexpr (Policy == TailCarryPolicy::independent) {
              tail_carry = combine_reduction_carries<Reduction>(
                  tail_tag,
                  vector_prefix(full_tag, tail_tag, full_carry),
                  tail_carry);
            }
            full_carry = replace_vector_prefix(
                full_tag, tail_tag, full_carry, tail_carry);
          } else {
            static_assert(is_vector_invariant_definition_v<Definition>);
            auto tail_carry = vector_prefix(
                full_tag, tail_tag, full_carry);
            const auto& const_tail_carry = tail_carry;
            with_vector_tail_carries<I + 1, Policy>(
                definitions, full_tag, tail_tag,
                full_dispatch, use,
                tail_carries..., const_tail_carry);
          }
        });
  }
}

#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
#pragma GCC diagnostic pop
#endif

// ======================== Slice Traits ========================

template <typename T>
struct SliceTraits {
  static constexpr int rank = 0;
  static constexpr bool is_sliceable = false;
};

template <int I, typename T>
struct TensorShapeDim;

template <int I, typename T, typename... Ss, typename... Ts>
struct TensorShapeDim<I, Tensor<T, Shape<Ss...>, Strides<Ts...>>> {
  using type = std::tuple_element_t<I, std::tuple<Ss...>>;
};

template <typename T, typename TShape, typename TStrides>
struct SliceTraits<Tensor<T, TShape, TStrides>> {
  using TensorT = Tensor<T, TShape, TStrides>;
  static constexpr int rank = TensorT::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, TensorT>::type;

  static nint_t size(const TensorT& input, int dim) {
    return input.size(dim);
  }

  template <int ActualDim, typename U>
  VECOPS_ALWAYS_INLINE static constexpr decltype(auto) slice(U&& input, nint_t index);
};

template <typename Compute, typename TensorT, typename Transform,
          typename Projection, typename... Facts>
struct SliceTraits<
    InputSpec<Compute, TensorT, Transform, Projection, Facts...>> {
  using Spec = InputSpec<
      Compute, TensorT, Transform, Projection, Facts...>;
  static constexpr int rank = TensorT::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, TensorT>::type;

  static nint_t size(const Spec& spec, int dim) {
    return spec.input_layout().shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_ALWAYS_INLINE static constexpr auto slice(U&& spec, nint_t index) {
    return tensor::slice_view<ActualDim>(spec, index);
  }
};

template <typename Compute, typename TensorT, typename Transform,
          typename Projection, typename... Facts>
struct SliceTraits<
    OutputSpec<Compute, TensorT, Transform, Projection, Facts...>> {
  using Spec = OutputSpec<
      Compute, TensorT, Transform, Projection, Facts...>;
  static constexpr int rank = TensorT::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, TensorT>::type;

  static nint_t size(const Spec& spec, int dim) {
    return spec.output_layout().shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_ALWAYS_INLINE static constexpr auto slice(U&& spec, nint_t index) {
    return tensor::slice_view<ActualDim>(spec, index);
  }
};

template <typename Spec, bool IsInput = Spec::is_input>
struct SpecTensorType {
  using type = typename Spec::OutputTensor;
};

template <typename Spec>
struct SpecTensorType<Spec, true> {
  using type = typename Spec::InputTensor;
};

template <typename Access>
  requires requires(const Access& access, nint_t index) {
    Access::Rank;
    access.spec();
    tensor::slice_view<0>(access, index);
  }
struct SliceTraits<Access> {
  using Spec = std::remove_cvref_t<
      decltype(std::declval<const Access&>().spec())>;
  using TensorT = typename SpecTensorType<Spec>::type;
  static constexpr int rank = Access::Rank;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, TensorT>::type;

  static nint_t size(const Access& access, int dim) {
    return access.spec().tensor().size(dim);
  }

  template <int ActualDim, typename U>
  VECOPS_ALWAYS_INLINE static constexpr auto slice(
      U&& access, nint_t index) {
    return tensor::slice_view<ActualDim>(access, index);
  }
};

template <typename T>
inline constexpr int slice_rank_v = SliceTraits<std::remove_cvref_t<T>>::rank;

template <typename T>
inline constexpr bool is_sliceable_v = SliceTraits<std::remove_cvref_t<T>>::is_sliceable;

template <int I, typename T>
using slice_shape_dim_t = typename SliceTraits<std::remove_cvref_t<T>>::template shape_dim<I>;

template <typename T>
struct IsConstOne : std::false_type {};

template <>
struct IsConstOne<Const<1>> : std::true_type {};

template <int I, typename T>
inline constexpr bool is_const_one_dim_v =
    IsConstOne<slice_shape_dim_t<I, T>>::value;

constexpr int max2(int a, int b) {
  return a > b ? a : b;
}

template <int... Ns>
struct MaxInt;

template <>
struct MaxInt<> : std::integral_constant<int, 0> {};

template <int N, int... Ns>
struct MaxInt<N, Ns...> : std::integral_constant<int, max2(N, MaxInt<Ns...>::value)> {};

template <int... Is>
inline constexpr int max_index_plus_one_v = MaxInt<(Is + 1)...>::value;

template <typename... Ts>
inline constexpr int max_slice_rank_v = MaxInt<slice_rank_v<Ts>...>::value;

// ======================== Dimension Validation ========================

template <int I, int... Is>
struct ContainsDim : std::bool_constant<((I == Is) || ...)> {};

template <int... Is>
struct UniqueDims : std::true_type {};

template <int I, int... Is>
struct UniqueDims<I, Is...>
    : std::bool_constant<!ContainsDim<I, Is...>::value && UniqueDims<Is...>::value> {};

template <int I>
constexpr int adjust_dim_after_slice() {
  static_assert(I != 0, "duplicate traversal dimension");
  if constexpr (I < 0) {
    return I;
  } else {
    return I - 1;
  }
}

template <int SlicedDim, int I>
constexpr int adjust_dim_after_slice() {
  static_assert(I != SlicedDim, "duplicate traversal dimension");
  if constexpr (I < SlicedDim) {
    return I;
  } else {
    return I - 1;
  }
}

// ======================== Logical-to-Actual Dimension Mapping ========================

template <int LogicalRank, int LogicalDim, typename T>
inline constexpr int actual_dim_v =
    LogicalDim - (LogicalRank - slice_rank_v<T>);

template <int LogicalRank, int LogicalDim, typename T>
inline constexpr bool has_actual_dim_v =
    is_sliceable_v<T> && (0 <= actual_dim_v<LogicalRank, LogicalDim, T>) &&
    (actual_dim_v<LogicalRank, LogicalDim, T> < slice_rank_v<T>);

template <bool HasActualDim, int ActualDim, typename T>
struct IsBroadcastDim : std::true_type {};

template <int ActualDim, typename T>
struct IsBroadcastDim<true, ActualDim, T>
    : std::bool_constant<is_const_one_dim_v<ActualDim, T>> {};

template <int LogicalRank, int LogicalDim, typename T>
inline constexpr bool is_broadcast_dim_v =
    IsBroadcastDim<
        has_actual_dim_v<LogicalRank, LogicalDim, T>,
        actual_dim_v<LogicalRank, LogicalDim, T>,
        T>::value;

// ======================== Extent Resolution and Slicing ========================

template <int LogicalRank, int LogicalDim, typename T>
VECOPS_ALWAYS_INLINE void update_extent(nint_t& extent, bool& has_extent, const T& input) {
  if constexpr (has_actual_dim_v<LogicalRank, LogicalDim, T> &&
                !is_broadcast_dim_v<LogicalRank, LogicalDim, T>) {
    constexpr int actual_dim = actual_dim_v<LogicalRank, LogicalDim, T>;
    const nint_t current = SliceTraits<std::remove_cvref_t<T>>::size(input, actual_dim);
    if (has_extent) {
      VECOPS_ASSERT(extent == current, "broadcast extent mismatch");
    } else {
      extent = current;
      has_extent = true;
    }
  }
}

template <int ActualDim, int D>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) slice_arg(nint_t index) {
  if constexpr (D == ActualDim) {
    return index;
  } else {
    return reserve;
  }
}

template <int ActualDim, typename T, size_t... Ds>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) slice_at_actual_dim_impl(
    T&& input,
    nint_t index,
    std::index_sequence<Ds...>) {
  return std::forward<T>(input)(slice_arg<ActualDim, static_cast<int>(Ds)>(index)...);
}

template <typename T, typename TShape, typename TStrides>
template <int ActualDim, typename U>
VECOPS_ALWAYS_INLINE constexpr decltype(auto)
SliceTraits<Tensor<T, TShape, TStrides>>::slice(U&& input, nint_t index) {
  return slice_at_actual_dim_impl<ActualDim>(
      std::forward<U>(input),
      index,
      std::make_index_sequence<TensorT::Ndim>{});
}

template <int ActualDim, typename T>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) slice_at_actual_dim(T&& input, nint_t index) {
  return SliceTraits<std::remove_cvref_t<T>>::template slice<ActualDim>(
      std::forward<T>(input),
      index);
}

template <int LogicalRank, int LogicalDim, typename T>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) slice_one(T&& input, nint_t index) {
  if constexpr (!has_actual_dim_v<LogicalRank, LogicalDim, T>) {
    return std::forward<T>(input);
  } else {
    constexpr int actual_dim = actual_dim_v<LogicalRank, LogicalDim, T>;
    if constexpr (is_broadcast_dim_v<LogicalRank, LogicalDim, T>) {
      return slice_at_actual_dim<actual_dim>(std::forward<T>(input), 0);
    } else {
      return slice_at_actual_dim<actual_dim>(std::forward<T>(input), index);
    }
  }
}

// ======================== Recursive Traversal ========================

/** How a traversal leaf forwards the accumulated dimension indices: not at
 *  all, expanded as leading call arguments, or as one tuple argument. */
enum class for_each_index_mode { none, expand, tuple };

template <typename Fn, typename IndexTuple, typename... Inputs, size_t... Js>
VECOPS_ALWAYS_INLINE void invoke_with_index_impl(
    Fn& fn,
    const IndexTuple& indices,
    std::index_sequence<Js...>,
    Inputs&&... inputs) {
  fn(std::get<Js>(indices)..., std::forward<Inputs>(inputs)...);
}

template <typename Fn, typename IndexTuple, typename... Inputs>
VECOPS_ALWAYS_INLINE void invoke_with_index(
    Fn& fn,
    const IndexTuple& indices,
    Inputs&&... inputs) {
  constexpr size_t index_count = std::tuple_size_v<std::remove_cvref_t<IndexTuple>>;
  invoke_with_index_impl(
      fn,
      indices,
      std::make_index_sequence<index_count>{},
      std::forward<Inputs>(inputs)...);
}

// Leaf overload: no remaining sliced dimensions.
template <
    int LogicalRank,
    for_each_index_mode Mode,
    typename IndexTuple,
    typename Fn,
    typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_impl(
    Fn& fn, const IndexTuple& indices, Inputs&&... inputs) {
  static_assert(LogicalRank >= 0, "logical rank exhausted");
  if constexpr (Mode == for_each_index_mode::none) {
    fn(std::forward<Inputs>(inputs)...);
  } else if constexpr (Mode == for_each_index_mode::expand) {
    invoke_with_index(fn, indices, std::forward<Inputs>(inputs)...);
  } else {
    fn(indices, std::forward<Inputs>(inputs)...);
  }
}

// Recursive overload: one dimension at a time. A single traversal serves all
// three index modes; only the leaf differs.
template <
    int LogicalRank,
    for_each_index_mode Mode,
    int I0,
    int... Is,
    typename IndexTuple,
    typename Fn,
    typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_impl(
    Fn& fn, const IndexTuple& indices, Inputs&&... inputs) {
  static_assert(LogicalRank > 0, "logical rank exhausted");
  nint_t extent = 1;
  bool has_extent = false;
  (update_extent<LogicalRank, I0>(extent, has_extent, inputs), ...);

  for (nint_t i = 0; i < extent; ++i) {
    if constexpr (Mode == for_each_index_mode::none) {
      for_each_impl<LogicalRank - 1, Mode, adjust_dim_after_slice<I0, Is>()...>(
          fn, indices, slice_one<LogicalRank, I0>(inputs, i)...);
    } else {
      const auto next_indices = std::tuple_cat(indices, std::tuple<nint_t>{i});
      for_each_impl<LogicalRank - 1, Mode, adjust_dim_after_slice<I0, Is>()...>(
          fn, next_indices, slice_one<LogicalRank, I0>(inputs, i)...);
    }
  }
}

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDims;

template <int N, int... Is>
struct LeadingDims<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_impl<logical_rank, for_each_index_mode::none, Is...>(
        fn, std::tuple<>{}, std::forward<Inputs>(inputs)...);
  }
};

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDimsWithIndex;

template <int N, int... Is>
struct LeadingDimsWithIndex<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_impl<logical_rank, for_each_index_mode::expand, Is...>(
        fn, std::tuple<>{}, std::forward<Inputs>(inputs)...);
  }
};

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDimsWithIndexTuple;

template <int N, int... Is>
struct LeadingDimsWithIndexTuple<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_impl<logical_rank, for_each_index_mode::tuple, Is...>(
        fn, std::tuple<>{}, std::forward<Inputs>(inputs)...);
  }
};

} // namespace details

/** Defines an additive vector carry and overwrites `result` on completion. */
template <typename T>
VECOPS_ALWAYS_INLINE auto reduce_add(T& result) {
  return details::VectorReductionDefinition<
      T, ::vecops::vec::ReduceAddOp>{result};
}

/** Defines a maximum vector carry and overwrites `result` on completion. */
template <typename T>
VECOPS_ALWAYS_INLINE auto reduce_max(T& result) {
  return details::VectorReductionDefinition<
      T, ::vecops::vec::ReduceMaxOp>{result};
}

/** Defines a minimum vector carry and overwrites `result` on completion. */
template <typename T>
VECOPS_ALWAYS_INLINE auto reduce_min(T& result) {
  return details::VectorReductionDefinition<
      T, ::vecops::vec::ReduceMinOp>{result};
}

/**
 * Defines a read-only loop invariant. Scalars are filled into each block Tag;
 * a Vec<BaseTag> is repeated word-wise to cover the actual block Tag.
 */
template <typename T>
VECOPS_ALWAYS_INLINE auto invariant(const T& value) {
  return details::VectorInvariantDefinition<T>{value};
}

/**
 * @brief Vector block traversal with mutable reductions and read-only
 *        invariants.
 *
 * Reduction carries are independent automatic variables and are never stored
 * in an aggregate, so this overload supports sizeless SVE vectors. Definitions
 * are passed to the callback in declaration order after `(tag, i, active)`.
 */
#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

template <
    int FullFactor = 1, int TailFactor = 1,
    TailCarryPolicy Policy = TailCarryPolicy::independent,
    ::vecops::vec::VectorTag Tag, meta::ValueInput N, typename Fn,
    typename... Definitions>
VECOPS_ALWAYS_INLINE void fold(
    Tag base_tag, N n, Fn&& block, Definitions&&... definitions) {
  static_assert(FullFactor > 0, "fold full factor must be positive");
  static_assert(TailFactor > 0, "fold tail factor must be positive");
  static_assert(
      TailFactor <= FullFactor,
      "fold tail factor cannot exceed full factor");
  static_assert(
      sizeof...(Definitions) > 0,
      "fold requires at least one carry definition");
  static_assert(
      (details::is_vector_carry_definition_v<Definitions> && ...),
      "fold accepts only reduce_add/max/min or invariant definitions");

  using FullTag = details::suggested_factor_tag_t<Tag, FullFactor>;
  using TailTag = details::suggested_factor_tag_t<Tag, TailFactor>;
  auto n_value = meta::to_value(n);
  if constexpr (details::is_const_value_v<decltype(n_value)>) {
    static_assert(
        details::ConstValue<std::remove_cvref_t<decltype(n_value)>>::value >= 0,
        "fold n must be non-negative");
  }
  const nint_t n_int = static_cast<nint_t>(n_value);
  VECOPS_ASSERT(n_int >= 0, "fold n must be non-negative");

  FullTag full_tag{};
  TailTag tail_tag{};
  const nint_t full_lanes = ::vecops::vec::size(full_tag);
  const nint_t tail_lanes = ::vecops::vec::size(tail_tag);
  auto definition_tuple = std::forward_as_tuple(definitions...);
  auto&& block_ref = block;

  auto use_full_carries = [&](auto&... full_carries)
      VECOPS_INLINE_LAMBDA {
    nint_t i = 0;
    for (; i + full_lanes <= n_int; i += full_lanes) {
      block_ref(
          full_tag, i, ::vecops::vec::opt::unmasked,
          full_carries...);
    }
    if (i < n_int) {
      auto full_dispatch = [&]<std::size_t Wanted>(auto&& fn)
          VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return details::invoke_vector_carry<Wanted>(
            std::forward<decltype(fn)>(fn), full_carries...);
      };
      auto use_tail_carries = [&](auto&... tail_carries)
        VECOPS_INLINE_LAMBDA {
        for (; i + tail_lanes <= n_int; i += tail_lanes) {
          block_ref(
              tail_tag, i, ::vecops::vec::opt::unmasked,
              tail_carries...);
        }
        if (i < n_int) {
          block_ref(
              tail_tag, i,
              ::vecops::vec::opt::first(n_int - i),
              tail_carries...);
        }
      };
      details::with_vector_tail_carries<0, Policy>(
          definition_tuple, full_tag, tail_tag,
          full_dispatch, use_tail_carries);
    }
  };
  details::with_vector_full_carries<0>(
      definition_tuple, base_tag, full_tag, use_full_carries);
}

#if defined(COMPILER_GCC) && defined(CPU_CAPABILITY_SVE)
#pragma GCC diagnostic pop
#endif

/**
 * @brief Traverse selected logical dimensions and invoke `fn` on the resulting
 *        slices.
 *
 * The dimensions in `Is...` are interpreted in logical trailing-aligned space.
 * Each listed dimension is removed before descending to the next recursion
 * level, so later dimensions are adjusted to match the rank of the sliced
 * inputs.
 *
 * When all selected dimensions have been traversed, `fn` is invoked with one
 * argument per input. Each argument is either:
 *
 * - an element reference, if all dimensions of that Tensor were sliced;
 * - a sub-Tensor view, if some Tensor dimensions remain;
 * - the original non-Tensor input, for scalar or other non-Tensor arguments.
 *
 * @tparam Is      Logical dimensions to traverse. They must be non-negative
 *                 and unique.
 * @tparam Fn      Callable type. It must be invocable as `fn(auto&&...)`.
 * @tparam Inputs  Tensor and non-Tensor input types.
 *
 * @param fn      Callable invoked at the traversal leaves.
 * @param inputs  Inputs traversed together.
 *
 * @note Only `Const<1>` Tensor dimensions broadcast. Non-broadcast extents at
 *       the same logical dimension are checked with `VECOPS_ASSERT`.
 *
 * @see for_each_dims
 */
template <int... Is, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each(Fn&& fn, Inputs&&... inputs) {
  static_assert(((Is >= 0) && ...), "traversal dimensions must be non-negative");
  static_assert(details::UniqueDims<Is...>::value, "duplicate traversal dimension");
  constexpr int logical_rank = details::max2(
      details::max_index_plus_one_v<Is...>,
      details::max_slice_rank_v<Inputs...>);
  details::for_each_impl<logical_rank, details::for_each_index_mode::none, Is...>(
      fn, std::tuple<>{}, std::forward<Inputs>(inputs)...);
}

/**
 * @brief Traverse selected logical dimensions, passing current indices to
 *        `fn` before the resulting slices.
 *
 * This is the indexed variant of `for_each`. Traversal, trailing alignment,
 * broadcasting, and slicing behavior are identical to `for_each`.
 *
 * At each leaf, `fn` is invoked as:
 *
 * @code
 * fn(nint_t indices..., auto&& sub_inputs...)
 * @endcode
 *
 * The index order is exactly the order of `Is...`. For example,
 * `for_each_with_index<1, 2, 0>` calls `fn(dim1_idx, dim2_idx, dim0_idx, ...)`.
 *
 * @tparam Is      Logical dimensions to traverse. They must be non-negative
 *                 and unique.
 * @tparam Fn      Callable type. It must be invocable with the indices
 *                 followed by one argument per input.
 * @tparam Inputs  Tensor and non-Tensor input types.
 *
 * @param fn      Callable invoked at the traversal leaves.
 * @param inputs  Inputs traversed together.
 *
 * @see for_each
 * @see for_each_dims_with_index
 */
template <int... Is, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index(Fn&& fn, Inputs&&... inputs) {
  static_assert(((Is >= 0) && ...), "traversal dimensions must be non-negative");
  static_assert(details::UniqueDims<Is...>::value, "duplicate traversal dimension");
  constexpr int logical_rank = details::max2(
      details::max_index_plus_one_v<Is...>,
      details::max_slice_rank_v<Inputs...>);
  details::for_each_impl<logical_rank, details::for_each_index_mode::expand, Is...>(
      fn,
      std::tuple<>{},
      std::forward<Inputs>(inputs)...);
}

/**
 * @brief Traverse selected logical dimensions, passing current indices as one
 *        tuple before the resulting slices.
 *
 * This is equivalent to `for_each_with_index<Is...>` except that the indices
 * are grouped as `std::tuple<nint_t, ...>`:
 *
 * @code
 * fn(const auto& indices, auto&& sub_inputs...)
 * @endcode
 *
 * The tuple order is exactly the order of `Is...`.
 *
 * @see for_each_with_index
 * @see for_each_dims_with_index_tuple
 */
template <int... Is, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index_tuple(Fn&& fn, Inputs&&... inputs) {
  static_assert(((Is >= 0) && ...), "traversal dimensions must be non-negative");
  static_assert(details::UniqueDims<Is...>::value, "duplicate traversal dimension");
  constexpr int logical_rank = details::max2(
      details::max_index_plus_one_v<Is...>,
      details::max_slice_rank_v<Inputs...>);
  details::for_each_impl<logical_rank, details::for_each_index_mode::tuple, Is...>(
      fn,
      std::tuple<>{},
      std::forward<Inputs>(inputs)...);
}

/**
 * @brief Traverse logical dimensions `0, 1, ..., Ndim - 1`.
 *
 * This is the common prefix wrapper around `for_each<0, ..., Ndim - 1>`.
 * The logical rank is large enough to include both the selected dimensions and
 * all Tensor inputs. Inputs with smaller rank are trailing-aligned against that
 * logical rank.
 *
 * @tparam Ndim    Number of logical dimensions to traverse.
 * @tparam Fn      Callable type. It must be invocable as `fn(auto&&...)`.
 * @tparam Inputs  Tensor and non-Tensor input types.
 *
 * @param fn      Callable invoked at the traversal leaves.
 * @param inputs  Inputs traversed together.
 *
 * @code
 * loop::for_each_dims<2>([](auto&& dst, auto&& lhs, auto&& rhs) {
 *   dst = lhs + rhs;
 * }, out, a, b);
 * @endcode
 *
 * @see for_each
 */
template <int Ndim, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_dims(Fn&& fn, Inputs&&... inputs) {
  static_assert(Ndim >= 0, "Ndim must be non-negative");
  details::LeadingDims<Ndim>::run(fn, std::forward<Inputs>(inputs)...);
}

/**
 * @brief Traverse logical dimensions `0, 1, ..., Ndim - 1`, passing current
 *        indices to `fn` before the resulting slices.
 *
 * This is the indexed prefix wrapper around
 * `for_each_with_index<0, ..., Ndim - 1>`. The logical rank is large enough to
 * include both the selected dimensions and all Tensor inputs. Inputs with
 * smaller rank are trailing-aligned against that logical rank.
 *
 * At each leaf, `fn` is invoked as:
 *
 * @code
 * fn(nint_t dim0_idx, nint_t dim1_idx, ..., auto&& sub_inputs...)
 * @endcode
 *
 * @tparam Ndim    Number of logical dimensions to traverse.
 * @tparam Fn      Callable type. It must be invocable with `Ndim` indices
 *                 followed by one argument per input.
 * @tparam Inputs  Tensor and non-Tensor input types.
 *
 * @param fn      Callable invoked at the traversal leaves.
 * @param inputs  Inputs traversed together.
 *
 * @see for_each_with_index
 * @see for_each_dims
 */
template <int Ndim, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_dims_with_index(Fn&& fn, Inputs&&... inputs) {
  static_assert(Ndim >= 0, "Ndim must be non-negative");
  details::LeadingDimsWithIndex<Ndim>::run(fn, std::forward<Inputs>(inputs)...);
}

/**
 * @brief Traverse logical dimensions `0, 1, ..., Ndim - 1`, passing current
 *        indices as one tuple before the resulting slices.
 *
 * This is the tuple-index wrapper around
 * `for_each_with_index_tuple<0, ..., Ndim - 1>`.
 *
 * At each leaf, `fn` is invoked as:
 *
 * @code
 * fn(const auto& indices, auto&& sub_inputs...)
 * @endcode
 *
 * @see for_each_with_index_tuple
 * @see for_each_dims_with_index
 */
template <int Ndim, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_dims_with_index_tuple(Fn&& fn, Inputs&&... inputs) {
  static_assert(Ndim >= 0, "Ndim must be non-negative");
  details::LeadingDimsWithIndexTuple<Ndim>::run(fn, std::forward<Inputs>(inputs)...);
}

} // namespace vecops::kernel::loop

#endif // VECOPS_KERNEL_LOOP_H
