//
// Created by renyz on 2026/7/8.
//

#ifndef VECOPS_HOP_H
#define VECOPS_HOP_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/CoreDefs.h"
#include "vecops/Assertion.h"
#include "vecops/gemm/Tensor.h"

/**
 * @file HOP.h
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
 * | `hop::for_each<Is...>`            | Traverse the listed logical dimensions    |
 * | `hop::for_each_dims<N>`           | Traverse logical dimensions `0..N - 1`    |
 * | `hop::for_each_with_index<Is...>` | Traverse and pass current indices to `fn` |
 * | `hop::for_each_dims_with_index<N>`| Traverse `0..N - 1` and pass indices      |
 * | `hop::*_with_index_tuple`         | Pass indices as one tuple before slices   |
 * | `hop::scan`                       | Chunked 1D scan with carry                |
 * | `hop::map`                        | Chunked 1D map without carry              |
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
 * ## Usage overview
 *
 * @code
 * #include "vecops/gemm/HOP.h"
 * using namespace vecops::gemm;
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
 * hop::for_each_dims<2>([](auto&& dst, auto&& x, auto&& y) {
 *   dst = x + y;
 * }, to, ta, tb);
 *
 * // Traverse rows only. The callable receives rank-1 row views.
 * hop::for_each<0>([](auto&& row) {
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
 * - `scan` and `map` are one-dimensional chunk helpers. They accept
 *   `Const`/`Dynamic` Value objects for `n` and `step`; raw integers are
 *   promoted to `Any`.
 * - The helpers preserve `Tensor`'s non-owning view semantics. The underlying
 *   storage must outlive the traversal and must be mutable if the callable
 *   writes through the passed objects.
 */

namespace vecops::gemm::hop {
namespace details {

// ======================== Value Normalization ========================

template <typename T>
using HopValue = ::vecops::gemm::ToValue<std::remove_cvref_t<T>>;

template <typename T>
VECOPS_ALWAYS_INLINE constexpr HopValue<T> to_hop_value(T&& value) {
  return HopValue<T>{static_cast<nint_t>(value)};
}

template <typename T>
struct ConstValue : std::false_type {
  static constexpr nint_t value = 0;
};

template <nint_t N>
struct ConstValue<Const<N>> : std::true_type {
  static constexpr nint_t value = N;
};

template <typename T>
static constexpr bool is_const_value_v = ConstValue<std::remove_cvref_t<T>>::value;

template <typename N, typename Step>
struct HasNoTail : std::false_type {};

template <nint_t N, nint_t Step>
struct HasNoTail<Const<N>, Const<Step>>
    : std::bool_constant<(Step > 0) && (N % Step) == 0> {};

template <nint_t A, nint_t Lo, nint_t Hi, nint_t Step>
struct HasNoTail<Dynamic<A, Lo, Hi>, Const<Step>>
    : std::bool_constant<(Step > 0) && Dynamic<A, Lo, Hi>::aligns(Step)> {};

template <>
struct HasNoTail<Const<0>, Any> : std::true_type {};

template <nint_t A, nint_t Lo, nint_t Hi>
struct HasNoTail<Const<0>, Dynamic<A, Lo, Hi>> : std::true_type {};

template <typename N, typename Step>
static constexpr bool has_no_tail_v =
    HasNoTail<std::remove_cvref_t<N>, std::remove_cvref_t<Step>>::value;

template <typename N, typename Step>
struct HasConstTail : std::false_type {};

template <nint_t N, nint_t Step>
struct HasConstTail<Const<N>, Const<Step>>
    : std::bool_constant<(Step > 0) && (N % Step) != 0> {};

template <nint_t N>
struct HasConstTail<Const<N>, Const<0>> : std::false_type {};

template <typename N, typename Step>
static constexpr bool has_const_tail_v =
    HasConstTail<std::remove_cvref_t<N>, std::remove_cvref_t<Step>>::value;

template <typename N, typename Step>
VECOPS_ALWAYS_INLINE constexpr void validate_scan_args(const N& n, const Step& step) {
  if constexpr (is_const_value_v<N>) {
    static_assert(ConstValue<std::remove_cvref_t<N>>::value >= 0,
                  "scan n must be non-negative");
  }
  if constexpr (is_const_value_v<Step>) {
    static_assert(ConstValue<std::remove_cvref_t<Step>>::value > 0,
                  "scan step must be positive");
  }
  VECOPS_ASSERT(static_cast<nint_t>(n) >= 0, "scan n must be non-negative");
  VECOPS_ASSERT(static_cast<nint_t>(step) > 0, "scan step must be positive");
}

// Builds one independently initialized carry set per unrolled lane. Each set
// lives in a separate recursive call frame, so scalable vector types are never
// placed in an array or class object.
template <int Lane, int Unroll, typename MakeCarry, typename Body>
VECOPS_ALWAYS_INLINE decltype(auto) with_scan_carries(
    MakeCarry& make_carry,
    Body& body) {
  return make_carry([&](auto&... carries) VECOPS_INLINE_LAMBDA -> decltype(auto) {
    auto dispatch = [&]<int Wanted>(auto&& fn) VECOPS_INLINE_LAMBDA -> decltype(auto) {
      static_assert(Wanted == Lane);
      return std::forward<decltype(fn)>(fn)(carries...);
    };

    if constexpr (Lane + 1 == Unroll) {
      return body(dispatch);
    } else {
      auto next_body = [&](auto& next_dispatch) -> decltype(auto) VECOPS_INLINE_LAMBDA {
        auto combined_dispatch = [&]<int Wanted>(auto&& fn) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          if constexpr (Wanted == Lane) {
            return dispatch.template operator()<Wanted>(
                std::forward<decltype(fn)>(fn));
          } else {
            return next_dispatch.template operator()<Wanted>(
                std::forward<decltype(fn)>(fn));
          }
        };
        return body(combined_dispatch);
      };
      return with_scan_carries<Lane + 1, Unroll>(make_carry, next_body);
    }
  });
}

template <int Lane, int Unroll, typename Dispatch, typename Step, typename Fn>
VECOPS_ALWAYS_INLINE void invoke_full_scan_block(
    Dispatch& dispatch,
    nint_t base,
    const Step& step,
    Fn& fn) {
  dispatch.template operator()<Lane>([&](auto&... carries) VECOPS_INLINE_LAMBDA {
    fn(base + Lane * static_cast<nint_t>(step), step, carries...);
  });
  if constexpr (Lane + 1 < Unroll) {
    invoke_full_scan_block<Lane + 1, Unroll>(dispatch, base, step, fn);
  }
}

template <int Lane, int Unroll, typename Dispatch, typename Step, typename Fn>
VECOPS_ALWAYS_INLINE void invoke_remaining_scan_chunks(
    Dispatch& dispatch,
    nint_t base,
    nint_t remaining,
    const Step& step,
    Fn& fn) {
  if (Lane < remaining) {
    dispatch.template operator()<Lane>([&](auto&... carries) VECOPS_INLINE_LAMBDA {
      fn(base + Lane * static_cast<nint_t>(step), step, carries...);
    });
  }
  if constexpr (Lane + 1 < Unroll) {
    invoke_remaining_scan_chunks<Lane + 1, Unroll>(
        dispatch, base, remaining, step, fn);
  }
}

template <int Lane, int Unroll, typename Dispatch, typename Combine>
VECOPS_ALWAYS_INLINE void combine_scan_carries(
    Dispatch& dispatch,
    Combine& combine) {
  dispatch.template operator()<0>([&](auto&... dst) VECOPS_INLINE_LAMBDA {
    dispatch.template operator()<Lane>([&](auto&... src) VECOPS_INLINE_LAMBDA {
      combine(dst..., src...);
    });
  });
  if constexpr (Lane + 1 < Unroll) {
    combine_scan_carries<Lane + 1, Unroll>(dispatch, combine);
  }
}

template <int Lane, int Unroll, typename Step, typename Fn>
VECOPS_ALWAYS_INLINE void invoke_full_map_block(
    nint_t base,
    const Step& step,
    Fn& fn) {
  fn(base + Lane * static_cast<nint_t>(step), step);
  if constexpr (Lane + 1 < Unroll) {
    invoke_full_map_block<Lane + 1, Unroll>(base, step, fn);
  }
}

template <int Lane, int Unroll, typename Step, typename Fn>
VECOPS_ALWAYS_INLINE void invoke_remaining_map_chunks(
    nint_t base,
    nint_t remaining,
    const Step& step,
    Fn& fn) {
  if (Lane < remaining) {
    fn(base + Lane * static_cast<nint_t>(step), step);
  }
  if constexpr (Lane + 1 < Unroll) {
    invoke_remaining_map_chunks<Lane + 1, Unroll>(
        base, remaining, step, fn);
  }
}

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

template <typename T>
static constexpr int slice_rank_v = SliceTraits<std::remove_cvref_t<T>>::rank;

template <typename T>
static constexpr bool is_sliceable_v = SliceTraits<std::remove_cvref_t<T>>::is_sliceable;

template <int I, typename T>
using slice_shape_dim_t = typename SliceTraits<std::remove_cvref_t<T>>::template shape_dim<I>;

template <typename T>
struct IsConstOne : std::false_type {};

template <>
struct IsConstOne<Const<1>> : std::true_type {};

template <int I, typename T>
static constexpr bool is_const_one_dim_v =
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
static constexpr int max_index_plus_one_v = MaxInt<(Is + 1)...>::value;

template <typename... Ts>
static constexpr int max_slice_rank_v = MaxInt<slice_rank_v<Ts>...>::value;

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
static constexpr int actual_dim_v =
    LogicalDim - (LogicalRank - slice_rank_v<T>);

template <int LogicalRank, int LogicalDim, typename T>
static constexpr bool has_actual_dim_v =
    is_sliceable_v<T> && (0 <= actual_dim_v<LogicalRank, LogicalDim, T>) &&
    (actual_dim_v<LogicalRank, LogicalDim, T> < slice_rank_v<T>);

template <bool HasActualDim, int ActualDim, typename T>
struct IsBroadcastDim : std::true_type {};

template <int ActualDim, typename T>
struct IsBroadcastDim<true, ActualDim, T>
    : std::bool_constant<is_const_one_dim_v<ActualDim, T>> {};

template <int LogicalRank, int LogicalDim, typename T>
static constexpr bool is_broadcast_dim_v =
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

template <int LogicalRank, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_impl(Fn& fn, Inputs&&... inputs) {
  fn(inputs...);
}

template <int LogicalRank, int I0, int... Is, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_impl(Fn& fn, Inputs&&... inputs) {
  static_assert(LogicalRank > 0, "logical rank exhausted");
  nint_t extent = 1;
  bool has_extent = false;
  (update_extent<LogicalRank, I0>(extent, has_extent, inputs), ...);

  for (nint_t i = 0; i < extent; ++i) {
    for_each_impl<LogicalRank - 1, adjust_dim_after_slice<I0, Is>()...>(
        fn,
        slice_one<LogicalRank, I0>(inputs, i)...);
  }
}

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

template <int LogicalRank, typename IndexTuple, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index_impl(
    Fn& fn,
    const IndexTuple& indices,
    Inputs&&... inputs) {
  invoke_with_index(fn, indices, std::forward<Inputs>(inputs)...);
}

template <int LogicalRank, typename IndexTuple, typename Fn, typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index_tuple_impl(
    Fn& fn,
    const IndexTuple& indices,
    Inputs&&... inputs) {
  fn(indices, std::forward<Inputs>(inputs)...);
}

template <
    int LogicalRank,
    int I0,
    int... Is,
    typename IndexTuple,
    typename Fn,
    typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index_impl(
    Fn& fn,
    const IndexTuple& indices,
    Inputs&&... inputs) {
  static_assert(LogicalRank > 0, "logical rank exhausted");
  nint_t extent = 1;
  bool has_extent = false;
  (update_extent<LogicalRank, I0>(extent, has_extent, inputs), ...);

  for (nint_t i = 0; i < extent; ++i) {
    const auto next_indices = std::tuple_cat(indices, std::tuple<nint_t>{i});
    for_each_with_index_impl<
        LogicalRank - 1,
        adjust_dim_after_slice<I0, Is>()...>(
        fn,
        next_indices,
        slice_one<LogicalRank, I0>(inputs, i)...);
  }
}

template <
    int LogicalRank,
    int I0,
    int... Is,
    typename IndexTuple,
    typename Fn,
    typename... Inputs>
VECOPS_ALWAYS_INLINE void for_each_with_index_tuple_impl(
    Fn& fn,
    const IndexTuple& indices,
    Inputs&&... inputs) {
  static_assert(LogicalRank > 0, "logical rank exhausted");
  nint_t extent = 1;
  bool has_extent = false;
  (update_extent<LogicalRank, I0>(extent, has_extent, inputs), ...);

  for (nint_t i = 0; i < extent; ++i) {
    const auto next_indices = std::tuple_cat(indices, std::tuple<nint_t>{i});
    for_each_with_index_tuple_impl<
        LogicalRank - 1,
        adjust_dim_after_slice<I0, Is>()...>(
        fn,
        next_indices,
        slice_one<LogicalRank, I0>(inputs, i)...);
  }
}

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDims;

template <int N, int... Is>
struct LeadingDims<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_impl<logical_rank, Is...>(fn, std::forward<Inputs>(inputs)...);
  }
};

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDimsWithIndex;

template <int N, int... Is>
struct LeadingDimsWithIndex<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_with_index_impl<logical_rank, Is...>(
        fn,
        std::tuple<>{},
        std::forward<Inputs>(inputs)...);
  }
};

template <int N, typename Seq = std::make_integer_sequence<int, N>>
struct LeadingDimsWithIndexTuple;

template <int N, int... Is>
struct LeadingDimsWithIndexTuple<N, std::integer_sequence<int, Is...>> {
  template <typename Fn, typename... Inputs>
  VECOPS_ALWAYS_INLINE static void run(Fn& fn, Inputs&&... inputs) {
    constexpr int logical_rank = max2(N, max_slice_rank_v<Inputs...>);
    for_each_with_index_tuple_impl<logical_rank, Is...>(
        fn,
        std::tuple<>{},
        std::forward<Inputs>(inputs)...);
  }
};

} // namespace details

/**
 * @brief Run a chunked one-dimensional scan and return the final carry.
 *
 * `scan` walks the half-open range `[0, n)` in chunks of `step`. For each full
 * chunk it invokes:
 *
 * @code
 * carry = fn(carry, i, step);
 * @endcode
 *
 * where `i` is the chunk start. If the compiler cannot prove that the range
 * length is always divisible by `step`, `scan` emits one runtime tail branch:
 *
 * @code
 * if (i < n) carry = fn(carry, i, tail_n);
 * @endcode
 *
 * For a compile-time `Const` tail, `tail_n` preserves its `Const` type.
 * Otherwise `tail_n` is an `Any` value. The tail branch is removed only when
 * the type-level metadata proves that no non-empty tail can exist, for example
 * `Const<16>` with `Const<4>` or `Dynamic<4>` with `Const<4>`.
 *
 * `n` and `step` may be `Const`, `Dynamic`, or raw integer values. Raw integers
 * are promoted to `Any`, so their exact values are available at runtime and
 * their tail length is passed as `Any` when needed.
 *
 * @tparam Fn     Callable type. It must be invocable as
 *                `fn(Carry, nint_t, auto&&)` and return the next carry.
 * @tparam N      Range length type (`Const`, `Dynamic`, or integer).
 * @tparam Step   Chunk step type (`Const`, `Dynamic`, or integer).
 * @tparam Carry  Carry type.
 *
 * @param init  Initial carry value.
 * @param n     Number of elements in the logical range. Must be non-negative.
 * @param step  Full chunk length. Must be positive.
 * @param fn    Iteration function.
 *
 * @return The final carry value after all emitted chunks.
 *
 * @note Runtime validity checks use `VECOPS_ASSERT`. Compile-time `Const`
 *       invalid arguments are rejected with `static_assert`.
 */
template <int Unroll = 1, typename Fn, typename N, typename Step, typename Carry>
VECOPS_ALWAYS_INLINE auto scan(Carry&& init, N n, Step step, Fn&& fn)
    -> std::remove_cvref_t<Carry> {
  static_assert(Unroll == 1,
                "unrolled scan requires make_carry, combine, and finish callbacks");
  auto n_value = details::to_hop_value(n);
  auto step_value = details::to_hop_value(step);
  details::validate_scan_args(n_value, step_value);

  using NValue = std::remove_cvref_t<decltype(n_value)>;
  using StepValue = std::remove_cvref_t<decltype(step_value)>;
  using CarryT = std::remove_cvref_t<Carry>;

  CarryT carry = std::forward<Carry>(init);
  nint_t i = 0;
  const nint_t full_end = static_cast<nint_t>(n_value - step_value + cint<1>);
  const nint_t step_int = static_cast<nint_t>(step_value);
  for (; i < full_end; i += step_int) {
    carry = fn(carry, i, step_value);
  }

  if constexpr (!details::has_no_tail_v<NValue, StepValue>) {
    if (i < static_cast<nint_t>(n_value)) {
      if constexpr (details::has_const_tail_v<NValue, StepValue>) {
        carry = fn(carry, i, n_value % step_value);
      } else {
        carry = fn(carry, i, Any{static_cast<nint_t>(n_value) - i});
      }
    }
  }

  return carry;
}

/**
 * @brief Run an unrolled reduction with one or more independent carry values.
 *
 * `make_carry(use)` must create a fresh identity-valued carry set as automatic
 * variables and call `use(carry...)` synchronously. `update` is invoked as
 * `update(i, chunk_n, carry...)`. After traversal, `combine(dst..., src...)`
 * merges lanes 1..Unroll-1 into lane 0, and `finish(carry...)` produces the
 * result. Carry values are passed only by reference and are never aggregated,
 * so they may be SVE sizeless types.
 *
 * @tparam Unroll Number of independent carry sets. Must be positive.
 */
template <
    int Unroll = 1,
    typename N,
    typename Step,
    typename MakeCarry,
    typename Update,
    typename Combine,
    typename Finish>
VECOPS_ALWAYS_INLINE decltype(auto) scan(
    N n,
    Step step,
    MakeCarry&& make_carry,
    Update&& update,
    Combine&& combine,
    Finish&& finish) {
  static_assert(Unroll > 0, "scan unroll must be positive");

  auto n_value = details::to_hop_value(n);
  auto step_value = details::to_hop_value(step);
  details::validate_scan_args(n_value, step_value);

  using NValue = std::remove_cvref_t<decltype(n_value)>;
  using StepValue = std::remove_cvref_t<decltype(step_value)>;

  auto&& make_carry_ref = make_carry;
  auto&& update_ref = update;
  auto&& combine_ref = combine;
  auto&& finish_ref = finish;

  auto body = [&](auto& dispatch) -> decltype(auto) VECOPS_INLINE_LAMBDA {
    const nint_t n_int = static_cast<nint_t>(n_value);
    const nint_t step_int = static_cast<nint_t>(step_value);
    const nint_t full_chunks = n_int / step_int;
    const nint_t full_groups = full_chunks / Unroll;
    nint_t i = 0;

    for (nint_t group = 0; group < full_groups; ++group) {
      details::invoke_full_scan_block<0, Unroll>(
          dispatch, i, step_value, update_ref);
      i += Unroll * step_int;
    }

    const nint_t remaining = full_chunks - full_groups * Unroll;
    details::invoke_remaining_scan_chunks<0, Unroll>(
        dispatch, i, remaining, step_value, update_ref);
    i += remaining * step_int;

    if constexpr (!details::has_no_tail_v<NValue, StepValue>) {
      if (i < n_int) {
        dispatch.template operator()<0>([&](auto&... carries) VECOPS_INLINE_LAMBDA {
          if constexpr (details::has_const_tail_v<NValue, StepValue>) {
            update_ref(i, n_value % step_value, carries...);
          } else {
            update_ref(i, Any{n_int - i}, carries...);
          }
        });
      }
    }

    if constexpr (Unroll > 1) {
      details::combine_scan_carries<1, Unroll>(dispatch, combine_ref);
    }
    return dispatch.template operator()<0>(finish_ref);
  };

  return details::with_scan_carries<0, Unroll>(make_carry_ref, body);
}

/**
 * @brief Convenience overload for an unrolled reduction with one carry.
 *
 * `identity` is copied into every unrolled lane. `combine(lhs, rhs)` must
 * return the merged carry, and both callbacks must describe an associative
 * reduction with `identity` as its identity element.
 */
template <
    int Unroll = 1,
    typename Carry,
    typename N,
    typename Step,
    typename Fn,
    typename Combine>
VECOPS_ALWAYS_INLINE auto scan(
    Carry&& identity,
    N n,
    Step step,
    Fn&& fn,
    Combine&& combine) -> std::remove_cvref_t<Carry> {
  using CarryT = std::remove_cvref_t<Carry>;
  auto&& identity_ref = identity;
  auto&& fn_ref = fn;
  auto&& combine_ref = combine;

  return scan<Unroll>(
      n,
      step,
      [&](auto&& use) -> decltype(auto) VECOPS_INLINE_LAMBDA {
        CarryT carry = identity_ref;
        return use(carry);
      },
      [&](nint_t i, auto&& chunk_n, auto& carry) VECOPS_INLINE_LAMBDA {
        carry = fn_ref(carry, i, std::forward<decltype(chunk_n)>(chunk_n));
      },
      [&](auto& dst, auto& src) VECOPS_INLINE_LAMBDA {
        dst = combine_ref(dst, src);
      },
      [](auto& carry) VECOPS_INLINE_LAMBDA -> CarryT {
        return carry;
      });
}

/**
 * @brief Run a chunked one-dimensional map with no carry.
 *
 * This is the non-iterative form of `scan`. It uses the same chunk emission
 * rules, but invokes the user callable as:
 *
 * @code
 * fn(i, chunk_n);
 * @endcode
 *
 * @tparam Fn    Callable type. It must be invocable as `fn(nint_t, auto&&)`.
 * @tparam N     Range length type (`Const`, `Dynamic`, or integer).
 * @tparam Step  Chunk step type (`Const`, `Dynamic`, or integer).
 * @tparam Unroll Number of chunks emitted per loop iteration. Must be positive.
 *
 * @param n     Number of elements in the logical range. Must be non-negative.
 * @param step  Full chunk length. Must be positive.
 * @param fn    Mapping function.
 *
 * @see scan
 */
template <int Unroll = 1, typename Fn, typename N, typename Step>
VECOPS_ALWAYS_INLINE void map(N n, Step step, Fn&& fn) {
  static_assert(Unroll > 0, "map unroll must be positive");
  auto n_value = details::to_hop_value(n);
  auto step_value = details::to_hop_value(step);
  details::validate_scan_args(n_value, step_value);

  using NValue = std::remove_cvref_t<decltype(n_value)>;
  using StepValue = std::remove_cvref_t<decltype(step_value)>;

  const nint_t n_int = static_cast<nint_t>(n_value);
  const nint_t step_int = static_cast<nint_t>(step_value);
  const nint_t full_chunks = n_int / step_int;
  const nint_t full_groups = full_chunks / Unroll;
  nint_t i = 0;

  auto&& fn_ref = fn;
  for (nint_t group = 0; group < full_groups; ++group) {
    details::invoke_full_map_block<0, Unroll>(i, step_value, fn_ref);
    i += Unroll * step_int;
  }

  const nint_t remaining = full_chunks - full_groups * Unroll;
  details::invoke_remaining_map_chunks<0, Unroll>(
      i, remaining, step_value, fn_ref);
  i += remaining * step_int;

  if constexpr (!details::has_no_tail_v<NValue, StepValue>) {
    if (i < n_int) {
      if constexpr (details::has_const_tail_v<NValue, StepValue>) {
        fn_ref(i, n_value % step_value);
      } else {
        fn_ref(i, Any{n_int - i});
      }
    }
  }
}

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
  details::for_each_impl<logical_rank, Is...>(fn, std::forward<Inputs>(inputs)...);
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
  details::for_each_with_index_impl<logical_rank, Is...>(
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
  details::for_each_with_index_tuple_impl<logical_rank, Is...>(
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
 * hop::for_each_dims<2>([](auto&& dst, auto&& lhs, auto&& rhs) {
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

} // namespace vecops::gemm::hop

#endif // VECOPS_HOP_H
