//
// Created by renyz on 2026/6/11.
//

#ifndef VECOPS_TENSOR_H
#define VECOPS_TENSOR_H

#include <cstdint>
#include <type_traits>
#include <array>
#include <tuple>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/tensor/Layout.h"

/**
 * @file Tensor.h
 * @brief Non-owning multi-dimensional tensor view with slicing.
 *
 * This header defines the `Tensor` class — a lightweight, non-owning view
 * over externally-managed data, paired with a compile-time shape-and-stride
 * Layout. It supports rich slicing semantics via `operator()` and `operator[]`
 * using marker objects (`reserve`, `new_axis(n)`, `range(s,e,step)`).
 *
 * ## Key components
 *
 * | Component      | Purpose                                              |
 * |----------------|------------------------------------------------------|
 * | Tensor<T,S,St> | Multi-dimensional view preserving pointer mutability |
 * | Array<N,T>     | Alias for fully-dynamic Tensor of rank N             |
 * | make_tensor()  | Factory functions for creating Tensors               |
 * | Slicing markers| `reserve`, `new_axis(n)`, `range(s,e,step)`          |
 * | SlicedTraits   | Compile-time type computation for slicing results    |
 * | Continuity     | Convenience free functions for contiguity checks     |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/tensor/Tensor.h"
 * using namespace vecops::tensor;
 *
 * float data[24];
 *
 * // Create a 2x3x4 tensor with inferred row-major strides
 * auto t = make_tensor<3>(data, {2, 3, 4});
 *
 * // Slice with integer indices → element access
 * float val = t(0, 1, 2);
 *
 * // Slice with markers
 * auto sub = t(0, reserve, range(0, 3, 2));  // Tensor<float, Shape<Const<3>, Any>, ...>
 *
 * // Insert a new dimension
 * auto batched = t(new_axis(), reserve, reserve, reserve);  // Shape<Any=1, Shape<...>>
 *
 * // Transpose
 * auto t_t = transpose<0, 2>(t);
 * @endcode
 *
 * ## Pitfalls
 *
 * - Tensor is **non-owning**: the data pointer must outlive the Tensor view.
 * - Pointer cv-qualification is preserved: a Tensor created from `const T*`
 *   stays read-only, while one created from `T*` remains writable.
 * - Slicing with **only** integer indices returns an **element reference**,
 *   not a Tensor. Mixed slicing (integers + markers) returns a Tensor.
 * - `new_axis()` does NOT consume an existing dimension; `reserve` and `range` do.
 * - Range step can be negative (reversal); step=0 triggers an assertion.
 * - Slicing returns a new Tensor with potentially different Shape/Strides
 *   template parameters. The result type is statically computed by `SlicedTraitsImpl`.
 * - `as<TShape2,TStrides2>()` reinterprets the tensor with new type parameters;
 *   the caller must ensure the new types' constraints are satisfied.
 * - `Array<N,T>` always uses `Any` for all dimensions (fully dynamic, no Const info).
 */

namespace vecops::tensor {

// ======================== Slicing Markers ========================

namespace details {

/**
 * @brief Marker: keep the current dimension unchanged during slicing.
 *
 * When used as an `operator()` argument, `ReserveAxis` passes the
 * dimension through without modification (both the size and stride
 * are preserved in the result Tensor).
 *
 * @note This consumes one dimension from the source Tensor.
 *
 * @see reserve
 */
struct ReserveAxis {};

/**
 * @brief Marker: insert a new dimension of size `repeat`.
 *
 * `NewAxis<V>` inserts a new axis with size `repeat` into the result
 * Tensor. The stride for this new axis is always 0 (broadcast semantics
 * — all elements along the new axis share the same data).
 *
 * @note `NewAxis` does **not** consume a dimension from the source Tensor.
 *
 * @tparam V  Value type for the new dimension's size (default: `Any`).
 *
 * @see new_axis()
 */
template <typename V = Any>
struct NewAxis {
  static_assert(std::is_base_of_v<Value, V>, "NewAxis repeat must be a Value type");
  using value_type = V;
  nint_t repeat;

  constexpr explicit NewAxis(nint_t r) : repeat(r) {}
};

/**
 * @brief Marker: slice a dimension with start, end, and step.
 *
 * `Range<S,E,T>` selects a sub-range of the current dimension:
 * - `start`: inclusive start index (0-based).
 * - `end`: exclusive end index.
 * - `step`: stride within the slice (can be negative for reversal).
 *
 * The resulting dimension size is `ceil((end - start) / step)` (for positive
 * step) or `ceil((start - end) / (-step))` (for negative step).
 * The resulting stride is `original_stride * step`.
 *
 * @note `step` must not be zero (assertion).
 * @note This consumes one dimension from the source Tensor.
 *
 * @tparam Start  Value type for `start` (default: `Any`).
 * @tparam End    Value type for `end` (default: `Any`).
 * @tparam Step   Value type for `step` (default: `Any`).
 *
 * @see range()
 */
template <typename Start = Any, typename End = Any, typename Step = Any>
struct Range {
  static_assert(std::is_base_of_v<Value, Start>, "Range start must be a Value type");
  static_assert(std::is_base_of_v<Value, End>, "Range end must be a Value type");
  static_assert(std::is_base_of_v<Value, Step>, "Range step must be a Value type");
  using start_type = Start;
  using end_type = End;
  using step_type = Step;
  nint_t start, end, step;
};

/**
 * @brief Marker: ellipsis — fills remaining dimensions with `reserve`.
 *
 * When used in `operator()(...)`, the ellipsis expands to as many `reserve`
 * markers as needed to fill all source dimensions not consumed by other
 * explicit indices. At most one ellipsis is allowed per slicing expression.
 *
 * @code
 * // For a 4D tensor t, these are equivalent:
 * auto a = t(ellipsis, 3);      // = t(reserve, reserve, reserve, 3)
 * auto b = t(0, ellipsis);      // = t(0, reserve, reserve, reserve)
 * auto c = t(0, ellipsis, 3);   // = t(0, reserve, reserve, 3)
 * @endcode
 *
 * @see ellipsis
 */
struct Ellipsis {};

} // namespace details

/**
 * @brief Slicing marker: keep the current dimension as-is.
 *
 * @code
 * auto sub = t(reserve, int(2));  // Keep dim 0, select index 2 from dim 1
 * @endcode
 */
static constexpr auto reserve = details::ReserveAxis{};

/**
 * @brief Slicing marker: insert a new dimension of size 1 (default).
 *
 * When no argument is given, inserts an axis of size 1. This is useful
 * for adding batch dimensions or aligning tensor ranks.
 *
 * @code
 * auto batched = t(new_axis(), reserve, reserve);  // Insert leading dim of size 1
 * @endcode
 */
static constexpr auto new_axis() {
  return details::NewAxis<Any>{1};
}

/**
 * @brief Slicing marker: insert a new dimension with a typed repeat count.
 *
 * @tparam V  A Value type for the repeat count (preserved as compile-time info).
 * @param repeat  The size of the new dimension (must be >= 0).
 *
 * @code
 * auto batched = t(new_axis(cint<4>), reserve, reserve);  // New dim Const<4>
 * @endcode
 *
 * @note The runtime value `repeat` must be >= 0 (assertion).
 */
template <typename V, std::enable_if_t<std::is_base_of_v<Value, std::decay_t<V>>, bool> = true>
static constexpr auto new_axis(V repeat) {
  VECOPS_ASSERT(nint_t(repeat) >= 0, "Cannot repeat negative times");
  return details::NewAxis<std::decay_t<V>>{nint_t(repeat)};
}

/**
 * @brief Slicing marker: insert a new dimension with a raw integer repeat count.
 *
 * The repeat count is wrapped as `Any` (no compile-time constraints).
 *
 * @code
 * auto batched = t(new_axis(8), reserve, reserve);
 * @endcode
 */
static constexpr auto new_axis(nint_t repeat) {
  VECOPS_ASSERT(repeat >= 0, "Cannot repeat negative times");
  return details::NewAxis<Any>{repeat};
}

/**
 * @brief Slicing marker: select a sub-range of a dimension.
 *
 * Creates a `Range` with typed start, end, and step. When value types are
 * `Const<N>`, the compile-time information is preserved.
 *
 * @code
 * // Slicing with Const bounds
 * auto sub = t(range(cint<0>, cint<4>, cint<2>), reserve);
 *
 * // Slicing with raw integers (promoted to Any)
 * auto sub2 = t(range(1, 5), reserve);
 * @endcode
 *
 * @tparam S  Value type for `start`.
 * @tparam E  Value type for `end`.
 * @tparam T  Value type for `step` (default: `Any` with value 1).
 * @param start  Inclusive start index.
 * @param end    Exclusive end index.
 * @param step   Step within the slice (default: 1). Must not be 0.
 */
template <
    typename S, typename E, typename T = Any,
    std::enable_if_t<std::is_base_of_v<Value, std::decay_t<S>>, bool> = true,
    std::enable_if_t<std::is_base_of_v<Value, std::decay_t<E>>, bool> = true,
    std::enable_if_t<std::is_base_of_v<Value, std::decay_t<T>>, bool> = true
>
static constexpr auto range(S start, E end, T step = T{1}) {
  return details::Range<std::decay_t<S>, std::decay_t<E>, std::decay_t<T>>{
      nint_t(start), nint_t(end), nint_t(step)};
}

/**
 * @brief Slicing marker: range with raw integers (all promoted to `Any`).
 */
static constexpr auto range(nint_t start, nint_t end, nint_t step = 1) {
  return details::Range<Any, Any, Any>{start, end, step};
}

/**
 * @brief Slicing marker: ellipsis — fills all remaining source dimensions
 *        with `reserve`.
 *
 * Expands to the right number of `reserve` markers to cover dimensions
 * not explicitly indexed. At most one ellipsis is allowed per slicing call.
 *
 * @code
 * auto t = make_tensor<4>(data, {2, 3, 4, 5});
 * auto s1 = t(ellipsis, 3);     // shape (2, 3, 4)
 * auto s2 = t(0, ellipsis);     // shape (3, 4, 5)
 * auto s3 = t(0, ellipsis, 3);  // shape (3, 4)
 * @endcode
 */
static constexpr auto ellipsis = details::Ellipsis{};

// ======================== SlicedTraits (compile-time type computation) ========================

namespace details {

/**
 * @brief Prepend a type `S0` to the front of a variadic type pack `Meta<Ss...>`.
 *
 * Used internally by `SlicedTraitsImpl` to accumulate result dimension types.
 */
template <typename S0, typename Meta>
struct PrependMeta;

template <typename S0, template <typename...> class Meta, typename... Ss>
struct PrependMeta<S0, Meta<Ss...>> { using type = Meta<S0, Ss...>; };

/**
 * @brief Compile-time type computation for Tensor slicing results.
 *
 * `SlicedTraitsImpl<TShape, TStrides, void, TIndices...>` recursively
 * consumes the source shape and stride types together with the slicing
 * index types to compute:
 * - `NewShape`: the resulting Shape type.
 * - `NewStrides`: the resulting Strides type.
 * - `Ndim`: the rank of the resulting Tensor.
 *
 * ## How it works
 *
 * The recursion peels off one source dimension type and one index type
 * at a time. The behavior depends on the index type:
 *
 * | Index type       | Effect                                            |
 * |------------------|---------------------------------------------------|
 * | integer          | Dimension consumed, removed from result           |
 * | `ReserveAxis`    | Dimension passes through unchanged                |
 * | `NewAxis<V>`     | Dimension inserted (source dim NOT consumed)      |
 * | `Range<S,E,T>`   | Dimension consumed, size becomes Any, stride scales by T |
 *
 * @tparam TShape    Source Shape type.
 * @tparam TStrides  Source Strides type.
 * @tparam SFINAE    Internal SFINAE guard.
 * @tparam TIndices  Slicing index types (integer, ReserveAxis, NewAxis, Range).
 */
template <typename TShape, typename TStrides, typename = void, typename... TIndices>
struct SlicedTraitsImpl;

/// Base case: no more indices → result is the remaining shape/strides.
template <typename TShape, typename TStrides>
struct SlicedTraitsImpl<TShape, TStrides, void> {
  using NewShape = TShape;
  using NewStrides = TStrides;
  static constexpr int Ndim = TShape::Ndim;
};

/// Integer index: consumes one dimension.
template <typename S0, typename... Ss, typename T0, typename... Ts, typename I, typename... Is>
struct SlicedTraitsImpl<Shape<S0, Ss...>, Strides<T0, Ts...>,
    std::enable_if_t<std::is_integral_v<std::decay_t<I>>>,
    I, Is...> {
  using Next = SlicedTraitsImpl<Shape<Ss...>, Strides<Ts...>, void, Is...>;
  using NewShape = typename Next::NewShape;
  using NewStrides = typename Next::NewStrides;
  static constexpr int Ndim = Next::Ndim;
};

/// ReserveAxis: keeps the dimension unchanged.
template <typename S0, typename... Ss, typename T0, typename... Ts, typename... Is>
struct SlicedTraitsImpl<Shape<S0, Ss...>, Strides<T0, Ts...>, void, ReserveAxis, Is...> {
  using Next = SlicedTraitsImpl<Shape<Ss...>, Strides<Ts...>, void, Is...>;
  using NewShape = typename PrependMeta<S0, typename Next::NewShape>::type;
  using NewStrides = typename PrependMeta<T0, typename Next::NewStrides>::type;
  static constexpr int Ndim = NewShape::Ndim;
};

/// NewAxis<V>: inserts a new dimension (does NOT consume original dim).
template <typename V, typename... Ss, typename... Ts, typename... Is>
struct SlicedTraitsImpl<Shape<Ss...>, Strides<Ts...>, void, NewAxis<V>, Is...> {
  using Next = SlicedTraitsImpl<Shape<Ss...>, Strides<Ts...>, void, Is...>;
  using NewShape = typename PrependMeta<V, typename Next::NewShape>::type;
  using NewStrides = typename PrependMeta<Const<0>, typename Next::NewStrides>::type;
  static constexpr int Ndim = NewShape::Ndim;
};

/// Range: modifies one dimension.
template <
    typename S, typename E, typename T, typename S0, typename... Ss,
    typename T0, typename... Ts, typename... Is
>
struct SlicedTraitsImpl<Shape<S0, Ss...>, Strides<T0, Ts...>, void, Range<S, E, T>, Is...> {
  using Next = SlicedTraitsImpl<Shape<Ss...>, Strides<Ts...>, void, Is...>;

  /// Range slicing loses compile-time size information.
  using new_sz = Any;

  /// Stride is scaled by the Range step type T.
  using new_stride = decltype(std::declval<T0>() * std::declval<T>());

  using NewShape = typename PrependMeta<new_sz, typename Next::NewShape>::type;
  using NewStrides = typename PrependMeta<new_stride, typename Next::NewStrides>::type;
  static constexpr int Ndim = NewShape::Ndim;
};

// ======================== Ellipsis helpers ========================

/// True if the pack contains an Ellipsis.
template <typename... Ts>
constexpr bool has_ellipsis_v = (std::is_same_v<std::decay_t<Ts>, Ellipsis> || ...);

/// Count the number of Ellipsis markers in the pack.
template <typename... Ts>
constexpr int count_ellipsis_v = ((std::is_same_v<std::decay_t<Ts>, Ellipsis> ? 1 : 0) + ...);

/// Trait to detect Range<S, E, T> types.
template <typename T>
struct is_range : std::false_type {};
template <typename S, typename E, typename T>
struct is_range<Range<S, E, T>> : std::true_type {};
template <typename T>
constexpr bool is_range_v = is_range<std::decay_t<T>>::value;

/// True if the type consumes one source dimension (integer, ReserveAxis, or Range).
template <typename T>
constexpr bool is_dim_consuming_v =
    std::is_integral_v<std::decay_t<T>> ||
    std::is_same_v<std::decay_t<T>, ReserveAxis> ||
    is_range_v<T>;

/// Count the number of dimension-consuming types in the pack.
template <typename... Ts>
constexpr int count_dim_consuming_v = ((is_dim_consuming_v<Ts> ? 1 : 0) + ...);

/// Find the (0-based) index of Target in the pack. Returns -1 if not found.
template <int I, typename Target, typename... Ts>
struct FindIndexInPack { static constexpr int value = -1; };

template <int I, typename Target, typename T0, typename... Ts>
struct FindIndexInPack<I, Target, T0, Ts...> {
  static constexpr int value =
      std::is_same_v<std::decay_t<T0>, Target> ? I
      : FindIndexInPack<I + 1, Target, Ts...>::value;
};

} // namespace details

// ======================== Tensor ========================

/**
 * @brief Non-owning multi-dimensional tensor view with compile-time layout.
 *
 * `Tensor<T, TShape, TStrides>` is a lightweight view that pairs a data
 * pointer (`T*`, where T may itself be const) with a
 * `Layout<TShape, TStrides>`. It does **not**
 * own the underlying data — the caller must ensure the data outlives the
 * Tensor view.
 *
 * ## Template parameters
 *
 * - `T`: Element type (e.g., `float`, `int32_t`).
 * - `TShape`: A `Shape<...>` type encoding per-dimension sizes.
 * - `TStrides`: A `Strides<...>` type encoding per-dimension strides.
 *
 * Both `TShape` and `TStrides` must have the same rank.
 *
 * ## Construction
 *
 * @code
 * float data[24];
 *
 * // From raw data, typed shape, and typed strides
 * auto t = Tensor<float, Shape<Const<2>,Const<3>,Const<4>>,
 *                     Strides<Const<12>,Const<4>,Const<1>>>(data, {2,3,4}, {12,4,1});
 *
 * // From raw data + Layout
 * auto L = make_layout(make_shape(2,3,4), make_strides(12,4,1));
 * auto t2 = Tensor<float, decltype(L)::Shape, decltype(L)::Strides>(data, L);
 *
 * // Using make_tensor with initializer_list (auto-computes row-major strides)
 * auto t3 = make_tensor<3>(data, {2, 3, 4});
 *
 * // Using make_tensor with explicit shape and strides
 * auto t4 = make_tensor<3>(data, {2, 3, 4}, {12, 4, 1});
 * @endcode
 *
 * ## Slicing
 *
 * The `operator()` and `operator[]` methods support slicing with three
 * kinds of markers:
 *
 * | Marker              | Behavior                                  |
 * |---------------------|-------------------------------------------|
 * | Integer (e.g., `2`) | Selects a single index; removes the dim   |
 * | `reserve`           | Keeps the dim unchanged                   |
 * | `new_axis(n)`       | Inserts a new dim of size n               |
 * | `range(s,e,step)`   | Slices a sub-range of the dim             |
 *
 * @code
 * auto t = make_tensor<3>(data, {2, 3, 4});
 *
 * auto row = t(0, 1);           // Element access (scalar)
 * auto mat = t(0, reserve, reserve);  // Rank-3 → rank-2
 * auto sub = t(reserve, range(0, 3, 2), reserve);  // Sliced stride
 * auto b = t(new_axis(), reserve, reserve, reserve); // Insert leading dim
 * @endcode
 *
 * ## Pitfalls
 *
 * - **Non-owning**: `_data` must outlive the Tensor.
 * - **Pointer mutability**: no const-cast is used; mutability is part of the
 *   Tensor element type and is checked by output specs.
 * - **All-integer slice** returns an **element**, not a Tensor.
 * - **Mixed slice** (integer + markers) returns a new Tensor with a
 *   different Shape/Strides type.
 * - Slicing returns a Tensor whose `data()` pointer is offset by the
 *   computed element offset.
 *
 * @tparam T        Element type.
 * @tparam TShape   Shape type (rank N).
 * @tparam TStrides Strides type (rank N).
 */
template <typename T, typename TShape, typename TStrides>
class Tensor {
  static_assert(is_shape<TShape>, "TShape must be Shape<...>");
  static_assert(is_strides<TStrides>, "TStrides must be Strides<...>");
  static_assert(TShape::Ndim == TStrides::Ndim, "Shape and Strides ndim mismatch");

public:
  static constexpr int Ndim = TShape::Ndim;
  using Shape = TShape;
  using Stride = TStrides;
  using Layout = tensor::Layout<TShape, TStrides>;
  using ElementType = T;

  // -------- Construction --------

  /**
   * Construct from data pointer, Shape, and Strides.
   *
   * @param data    Pointer to the start of the data (non-owning).
   * @param shape   Shape descriptor.
   * @param strides Strides descriptor.
   */
  constexpr Tensor(T* data, TShape shape, TStrides strides)
      : _data(data), _layout(Layout{shape, strides}) {}

  /**
   * Construct from data pointer and a pre-built Layout.
   *
   * @param data   Pointer to the start of the data (non-owning).
   * @param layout Layout descriptor.
   */
  constexpr Tensor(T* data, Layout layout)
      : _data(data), _layout(layout) {}

  /**
   * Construct from raw initializer_lists for shape and strides.
   *
   * All dimension values are wrapped as `Any` (losing Const info).
   *
   * @note Asserts that both lists have `Ndim` elements.
   */
  constexpr Tensor(
      T* data,
      std::initializer_list<nint_t> shape_vals,
      std::initializer_list<nint_t> stride_vals
  )
      : _data(data),
        _layout(
            [&] {
              VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");
              VECOPS_ASSERT(stride_vals.size() == Ndim, "stride_vals.size() != Ndim");
              return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
                return Layout{
                    TShape{Any{*(shape_vals.begin() + (nint_t) Idx)}...},
                    TStrides{Any{*(stride_vals.begin() + (nint_t) Idx)}...}
                };
              }(std::make_index_sequence<Ndim>{});
            }()) {}

  /**
   * Construct from a shape initializer_list only; strides are computed
   * as row-major contiguous (last-dim stride = 1, accumulating product
   * from right to left).
   *
   * All dimension values are wrapped as `Any`.
   *
   * @note Asserts that the list has `Ndim` elements.
   */
  constexpr Tensor(
      T* data,
      std::initializer_list<nint_t> shape_vals
  )
      : _data(data),
        _layout(
            [&] {
              VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");
              return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
                std::array<nint_t, Ndim> shapes = {*(shape_vals.begin() + Idx)...};
                std::array<nint_t, Ndim> stride_vals;
                nint_t prod = 1;
                for (int d = Ndim - 1; d >= 0; --d) {
                  stride_vals[d] = prod;
                  prod *= shapes[d];
                }
                return Layout{
                    TShape{Any{shapes[Idx]}...},
                    TStrides{Any{stride_vals[Idx]}...}
                };
              }(std::make_index_sequence<Ndim>{});
            }()) {}

  // -------- Data access --------

  /// Get the raw pointer while preserving the Tensor element cv-qualification.
  constexpr T* data() const { return _data; }

  // -------- Dimension access --------

  /**
   * Get the size of dimension I (compile-time index).
   * Returns `constexpr` if the dimension is `Const<N>`.
   *
   * @tparam I  Dimension index (0 <= I < Ndim).
   */
  template <int I>
  constexpr nint_t size() const {
    return tensor::size<I>(_layout);
  }

  /**
   * Get the stride of dimension I (compile-time index).
   * Returns `constexpr` if the dimension is `Const<N>`.
   *
   * @tparam I  Dimension index (0 <= I < Ndim).
   */
  template <int I>
  constexpr nint_t stride() const {
    return tensor::stride<I>(_layout);
  }

  /// Get the size of dimension `i` (runtime index).
  nint_t size(int i) const {
    return _layout.shape()[i];
  }

  /// Get the stride of dimension `i` (runtime index).
  nint_t stride(int i) const {
    return _layout.strides()[i];
  }

  /// Get the number of dimensions.
  constexpr int ndim() const { return Ndim; }

  /// Get the full Layout.
  constexpr const Layout& layout() const { return _layout; }

  /**
   * Compute the total number of elements: prod of all dimension sizes.
   *
   * @note This is computed at runtime by iterating over all dimensions.
   */
  nint_t numel() const {
    nint_t n = 1;
    for (int i = 0; i < Ndim; ++i) n *= size(i);
    return n;
  }

  // -------- Continuity --------

  /// Compile-time check: are the last N dimensions contiguous?
  template <int N>
  static constexpr bool ct_is_last_contiguous = is_ct_last_contiguous<Layout, N>::value;

  /// Compile-time check: are all dimensions contiguous?
  static constexpr bool ct_is_contiguous = is_ct_contiguous<Layout>::value;

  /**
   * Runtime check: are the last N dimensions contiguous?
   *
   * If `ct_is_last_contiguous<N>` is true, returns `true` at compile time.
   *
   * @tparam N  Number of trailing dimensions to check (default: 1).
   */
  template <int N = 1>
  bool is_last_contiguous() const {
    return tensor::is_last_contiguous<N>(_layout);
  }

  /**
   * Runtime check: are all dimensions contiguous?
   *
   * If `ct_is_contiguous` is true, returns `true` at compile time.
   */
  bool is_contiguous() const {
    return tensor::is_contiguous(_layout);
  }

  // -------- Cross-type conversion --------

  /**
   * Reinterpret the Tensor with different Shape/Strides type parameters.
   *
   * The runtime shape and stride values are extracted from the current
   * Layout and passed to the new types' constructors. This is safe only
   * if the new types' constraints (alignment, bounds) are satisfied by
   * the actual values.
   *
   * @code
   * // From all-Any to fully-Const (if you know the values are const)
   * auto t_any = make_tensor<2>(data, {16, 32});
   * auto t_ct = t_any.as<Shape<Const<16>,Const<32>>, Strides<Const<32>,Const<1>>>();
   * @endcode
   *
   * @tparam TShape2   Target Shape type (must have same rank).
   * @tparam TStrides2 Target Strides type (must have same rank).
   * @return A new Tensor with the specified type parameters.
   *
   * @warning The new types' constraints are **asserted** at runtime
   *          (during Construction). Using mismatched types may trigger
   *          assertion failures.
   */
  template <typename TShape2, typename TStrides2>
  constexpr Tensor<T, TShape2, TStrides2> as() const {
    return make_tensor(_data, _layout.template as<TShape2, TStrides2>());
  }

  // -------- Implicit conversion --------

  /**
   * @brief Implicit conversion to a more lenient Tensor type.
   *
   * A Tensor can be implicitly converted to another Tensor with the same
   * element type and rank, where every dimension's Shape and Strides Value
   * type is "more lenient" (or equal). This means `Const<N>` can become
   * `Dynamic<A,L,H>` (if constraints satisfied) or `Any`, and `Dynamic`
   * can become `Any`, but not vice versa.
   *
   * @code
   * auto strict = make_tensor(data, make_shape(cint<4>, cint<5>),
   *                                  make_strides(cint<5>, cint<1>));
   * Array<float, 2> arr = strict;  // Implicit: Const→Any per dim
   * @endcode
   *
   * @tparam TShape2   Target Shape type (must be more-or-equal lenient).
   * @tparam TStrides2 Target Strides type (must be more-or-equal lenient).
   */
  template <typename TShape2, typename TStrides2,
      std::enable_if_t<
          !(std::is_same_v<Shape, TShape2> && std::is_same_v<Stride, TStrides2>) &&
          details::IsMoreLenientMeta<Shape, TShape2>::value &&
          details::IsMoreLenientMeta<Stride, TStrides2>::value,
      bool> = true>
  constexpr operator Tensor<T, TShape2, TStrides2>() const {
    return as<TShape2, TStrides2>();
  }

  // -------- operator() slicing --------

  /**
   * @brief Slicing via operator().
   *
   * ## Behavior depends on the index types:
   *
   * - **All integers**: computes an element offset and returns the element
   *   by reference (no Tensor).
   * - **Mixed (integers + markers)**: computes a new Layout and returns a
   *   new Tensor with offset data pointer.
   * - **Ellipsis**: expands to the right number of `reserve` markers to fill
   *   all source dimensions not explicitly indexed.
   *
   * @code
   * auto t = make_tensor<3>(data, {2, 3, 4});
   *
   * float val = t(0, 1, 2);                    // Element access
   * auto sub = t(0, reserve, range(1, 4));     // Tensor sub-view
   * auto col = t(ellipsis, 2);                 // = t(reserve, reserve, 2)
   * @endcode
   *
   * @tparam TIndices  Index types (int, ReserveAxis, NewAxis, Range, Ellipsis).
   * @param  indices   Per-dimension slicing indices.
   * @return Element reference (all-integer) or Tensor sub-view (mixed).
   */
  template <typename... TIndices>
  constexpr decltype(auto) operator()(TIndices... indices) const {
    if constexpr (details::has_ellipsis_v<TIndices...>) {
      static_assert(details::count_ellipsis_v<TIndices...> == 1,
                    "At most one ellipsis is allowed");
      constexpr int consumed = details::count_dim_consuming_v<TIndices...>;
      static_assert(consumed <= Ndim,
                    "Too many dimension-consuming indices for ellipsis");
      if constexpr (consumed <= Ndim) {
        return _slice_expand_ellipsis<Ndim - consumed>(indices...);
      }
    } else if constexpr ((std::is_integral_v<std::decay_t<TIndices>> && ...)) {
      nint_t offset = offset_at(_layout, indices...);
      return _data[offset];
    } else {
      using Traits = details::SlicedTraitsImpl<Shape, Stride, void, std::decay_t<TIndices>...>;
      using RetShape = typename Traits::NewShape;
      using RetStrides = typename Traits::NewStrides;
      constexpr int new_ndim = RetShape::Ndim;

      auto [ns, nt, offset] = _slice_make_meta<RetShape, RetStrides>(indices...);
      return make_tensor(_data + offset, ns, nt);
    }
  }

  /**
   * @brief Non-const overload: for all-integer indices, returns a mutable
   *        element reference; otherwise delegates to the const version.
   */
  template <typename... TIndices>
  constexpr decltype(auto) operator()(TIndices... indices) {
    if constexpr ((std::is_integral_v<std::decay_t<TIndices>> && ...)) {
      nint_t offset = offset_at(_layout, indices...);
      return data()[offset];
    } else {
      return static_cast<const Tensor*>(this)->operator()(indices...);
    }
  }

  /**
   * @brief Single-index slicing via operator[].
   *
   * Equivalent to `operator()(index)`.
   *
   * @note For a rank-1 Tensor, `t[0]` returns an element.
   *       For higher-rank tensors, use multiple indices: `t(0, 1)`.
   */
  template <typename TIndex>
  constexpr decltype(auto) operator[](TIndex index) const {
    return this->operator()(index);
  }

  template <typename TIndex>
  constexpr decltype(auto) operator[](TIndex index) {
    return this->operator()(index);
  }

private:
  /**
   * @brief Expand an ellipsis into the right number of `reserve` markers
   *        and recursively call operator() with the expanded pack.
   *
   * Finds the position of the Ellipsis in the index pack, splits the pack
   * into head and tail, and inserts `Nfill` reserve markers at the ellipsis
   * position. Then re-invokes `operator()` with the expanded pack, which
   * re-enters the else-if/integer-only branch correctly.
   *
   * @tparam Nfill   Number of reserve markers to insert (= Ndim - consumed).
   * @tparam TIndices  Original index types (contains exactly one Ellipsis).
   */
  template <int Nfill, typename... TIndices,
      std::enable_if_t<(Nfill >= 0), bool> = true>
  constexpr decltype(auto) _slice_expand_ellipsis(TIndices... indices) const {
    constexpr int Pos = details::FindIndexInPack<0, details::Ellipsis,
                        std::decay_t<TIndices>...>::value;
    auto tup = std::forward_as_tuple(indices...);
    return [&] <size_t... Hi, size_t... Fi, size_t... Ti>(
        std::index_sequence<Hi...>,
        std::index_sequence<Fi...>,
        std::index_sequence<Ti...>
    ) {
      return this->operator()(
          std::get<Hi>(tup)...,
          ((void)Fi, reserve)...,
          std::get<Pos + 1 + Ti>(tup)...
      );
    }(
        std::make_index_sequence<Pos>{},
        std::make_index_sequence<Nfill>{},
        std::make_index_sequence<sizeof...(TIndices) - Pos - 1>{}
    );
  }

  /**
   * Compute the linearized element offset given a pack of slicing indices.
   * This is the private implementation used during slicing.
   *
   * For each index:
   * - Integer `i`: contributes `i * stride(D)` to the offset.
   * - `ReserveAxis`: contributes 0 to offset (passes dimension through).
   * - `NewAxis<V>`: contributes 0 to offset; D is NOT incremented.
   * - `Range<S,E,T>`: contributes `start * stride(D)` to offset.
   */
  template <int D>
  nint_t _slice_index() const { return 0; }

  template <
      int D, typename I, typename... Is,
      std::enable_if_t<std::is_integral_v<std::decay_t<I>>, bool> = true
  >
  nint_t _slice_index(I i, Is... rest) const {
    nint_t idx = nint_t(i);
    VECOPS_ASSERT(0 <= idx && idx < size(D), "index out of range");
    return idx * stride(D) + _slice_index<D + 1>(rest...);
  }

  template <int D, typename... Is>
  nint_t _slice_index(details::ReserveAxis, Is... rest) const {
    return _slice_index<D + 1>(rest...);
  }

  template <int D, typename V, typename... Is>
  nint_t _slice_index(details::NewAxis<V>, Is... rest) const {
    return _slice_index<D>(rest...);
  }

  template <int D, typename S, typename E, typename TStep, typename... Is>
  nint_t _slice_index(details::Range<S, E, TStep> r, Is... rest) const {
    VECOPS_ASSERT(0 <= r.start && r.start < size(D), "range start out of range");
    VECOPS_ASSERT(0 <= r.end && r.end <= size(D), "range end out of range");
    return r.start * stride(D) + _slice_index<D + 1>(rest...);
  }

  /**
   * Tail case of `_slice_make_meta_impl`: fill remaining dimensions.
   */
  template <typename NewShape, typename NewStrides, int D, int OutD>
  auto _slice_make_meta_impl(
      std::array<nint_t, NewShape::Ndim>& ns_arr,
      std::array<nint_t, NewStrides::Ndim>& nt_arr
  ) const -> nint_t {
    for (int i = D; i < int(Ndim); ++i) {
      ns_arr[OutD + (i - D)] = size(i);
      nt_arr[OutD + (i - D)] = stride(i);
    }
    return 0;
  }

  /**
   * Integer index in `_slice_make_meta_impl`: consumes dimension D,
   * contributes `idx * stride(D)` to offset.
   */
  template <
      typename NewShape, typename NewStrides, int D, int OutD,
      typename I, typename... Is,
      std::enable_if_t<std::is_integral_v<std::decay_t<I>>, bool> = true
  >
  auto _slice_make_meta_impl(
      std::array<nint_t, NewShape::Ndim>& ns_arr,
      std::array<nint_t, NewStrides::Ndim>& nt_arr,
      I i, Is... rest
  ) const -> nint_t {
    nint_t idx = nint_t(i);
    VECOPS_ASSERT(0 <= idx && idx < size(D), "index out of range");
    return idx * stride(D) +
           _slice_make_meta_impl<NewShape, NewStrides, D + 1, OutD>(ns_arr, nt_arr, rest...);
  }

  /**
   * ReserveAxis in `_slice_make_meta_impl`: passes dimension through,
   * contributes 0 to offset.
   */
  template <typename NewShape, typename NewStrides, int D, int OutD, typename... Is>
  auto _slice_make_meta_impl(
      std::array<nint_t, NewShape::Ndim>& ns_arr,
      std::array<nint_t, NewStrides::Ndim>& nt_arr,
      details::ReserveAxis, Is... rest
  ) const -> nint_t {
    ns_arr[OutD] = size(D);
    nt_arr[OutD] = stride(D);
    return _slice_make_meta_impl<NewShape, NewStrides, D + 1, OutD + 1>(ns_arr, nt_arr, rest...);
  }

  /**
   * NewAxis in `_slice_make_meta_impl`: inserts a new dimension with
   * size `repeat` and stride 0; source dimension D is NOT consumed.
   */
  template <
      typename NewShape, typename NewStrides, int D, int OutD,
      typename V, typename... Is
  >
  auto _slice_make_meta_impl(
      std::array<nint_t, NewShape::Ndim>& ns_arr,
      std::array<nint_t, NewStrides::Ndim>& nt_arr,
      details::NewAxis<V> na, Is... rest
  ) const -> nint_t {
    ns_arr[OutD] = na.repeat;
    nt_arr[OutD] = 0;
    return _slice_make_meta_impl<NewShape, NewStrides, D, OutD + 1>(ns_arr, nt_arr, rest...);
  }

  /**
   * Range in `_slice_make_meta_impl`: computes sliced size and stride,
   * contributes `start * stride(D)` to offset.
   */
  template <
      typename NewShape, typename NewStrides, int D, int OutD,
      typename S, typename E, typename TStep, typename... Is
  >
  auto _slice_make_meta_impl(
      std::array<nint_t, NewShape::Ndim>& ns_arr,
      std::array<nint_t, NewStrides::Ndim>& nt_arr,
      details::Range<S, E, TStep> r, Is... rest
  ) const -> nint_t {
    VECOPS_ASSERT(0 <= r.start && r.start < size(D), "range start out of range");
    VECOPS_ASSERT(0 <= r.end && r.end <= size(D), "range end out of range");
    if (r.step > 0) {
      VECOPS_ASSERT(r.start < r.end, "for positive step, start must be less than end");
      ns_arr[OutD] = (r.end - r.start + r.step - 1) / r.step;
    } else if (r.step < 0) {
      VECOPS_ASSERT(r.start > r.end, "for negative step, start must be greater than end");
      ns_arr[OutD] = (r.start - r.end - r.step - 1) / (-r.step);
    } else {
      VECOPS_ASSERT(false, "step cannot be zero");
    }
    nt_arr[OutD] = stride(D) * r.step;
    return r.start * stride(D) +
           _slice_make_meta_impl<NewShape, NewStrides, D + 1, OutD + 1>(ns_arr, nt_arr, rest...);
  }

  /**
   * Entry point for building new Shape/Strides + offset during slicing.
   *
   * @return Tuple of (NewShape, NewStrides, linear_offset).
   */
  template <typename NewShape, typename NewStrides, typename... TIndices>
  auto _slice_make_meta(TIndices... indices) const
  -> std::tuple<NewShape, NewStrides, nint_t> {
    std::array<nint_t, NewShape::Ndim> ns_arr{};
    std::array<nint_t, NewStrides::Ndim> nt_arr{};
    nint_t offset = _slice_make_meta_impl<NewShape, NewStrides, 0, 0>(
        ns_arr, nt_arr, indices...
    );
    return {
        [&] <size_t... Idx>(std::index_sequence<Idx...>) {
          return NewShape{ns_arr[Idx]...};
        }(std::make_index_sequence<NewShape::Ndim>{}),
        [&] <size_t... Idx>(std::index_sequence<Idx...>) {
          return NewStrides{nt_arr[Idx]...};
        }(std::make_index_sequence<NewStrides::Ndim>{}),
        offset
    };
  }

  T* _data;
  Layout _layout;
};

namespace details {

template <typename T>
struct IsTensor : std::false_type {};
template <typename T, typename TShape, typename TStrides>
struct IsTensor<Tensor<T, TShape, TStrides>> : std::true_type {};

} // namespace details

/// Type trait: `true` if T is a Tensor.
template <typename T>
static constexpr bool is_tensor = details::IsTensor<T>::value;

// ======================== Array alias ========================

/**
 * @brief Convenience alias: a fully-dynamic Tensor of rank N.
 *
 * `Array<T, N>` is a Tensor where all N dimensions use `Any` for both
 * Shape and Strides (no compile-time `Const` information preserved).
 *
 * @code
 * Array<float, 3> t(data, {2, 3, 4});  // Tensor<float, Shape<Any,Any,Any>, Strides<Any,Any,Any>>
 * @endcode
 *
 * @tparam T  Element type.
 * @tparam N  Rank (number of dimensions).
 */
template <typename T, int64_t N>
using Array = Tensor<T,
    details::repeat_t<N, Shape, Any>,
    details::repeat_t<N, Strides, Any>>;

// ======================== make_tensor ========================

/**
 * @brief Create a Tensor from a data pointer and a Layout.
 *
 * This is the core factory; all other make_tensor overloads delegate here.
 *
 * @tparam T       Element type.
 * @tparam TLayout Layout type (Shape+Strides pair).
 * @param data     Data pointer (non-owning).
 * @param layout   Layout descriptor.
 * @return A Tensor with the given layout.
 */
template <typename T, typename TLayout,
    std::enable_if_t<is_layout<std::remove_cvref_t<TLayout>>, bool> = true>
constexpr auto make_tensor(T* data, TLayout&& layout) {
  using L = std::remove_cvref_t<TLayout>;
  return Tensor<T, typename L::Shape, typename L::Strides>(data, std::forward<TLayout>(layout));
}

/**
 * @brief Create a Tensor from typed Shape and Strides.
 *
 * @code
 * auto s = make_shape(cint<2>, cint<3>);
 * auto st = make_strides(cint<3>, cint<1>);
 * auto t = make_tensor(data, s, st);
 * @endcode
 *
 * @tparam T        Element type.
 * @tparam TShape   Shape type (deduced from argument).
 * @tparam TStrides Strides type (deduced from argument).
 * @param data      Data pointer (non-owning).
 * @param shape     Shape descriptor.
 * @param strides   Strides descriptor.
 * @return A Tensor with the given layout.
 */
template <typename T, typename TShape, typename TStrides>
constexpr auto make_tensor(T* data, TShape&& shape, TStrides&& strides) {
  return make_tensor(data, make_layout(std::forward<TShape>(shape), std::forward<TStrides>(strides)));
}

/**
 * @brief Create a Tensor from a typed Shape only.
 *
 * Strides are inferred as row-major contiguous, preserving compile-time
 * type constraints (Const remains Const, Dynamic preserves alignment, etc.).
 *
 * @code
 * auto s = make_shape(cint<2>, Any{5}, cint<4>, cint<3>);
 * auto t = make_tensor(data, s);
 * // t has inferred Strides<Dynamic<4>, Const<12>, Const<3>, Const<1>>
 * @endcode
 *
 * @tparam T      Element type.
 * @tparam TShape Shape type (deduced from argument).
 * @param data    Data pointer (non-owning).
 * @param shape   Shape descriptor.
 * @return A Tensor with inferred contiguous Strides.
 */
template <typename T, typename TShape,
    std::enable_if_t<is_shape<std::remove_cvref_t<TShape>>, bool> = true>
constexpr auto make_tensor(T* data, TShape&& shape) {
  return make_tensor(data, make_layout(std::forward<TShape>(shape)));
}

/**
 * @brief Create a Tensor of rank Ndim from initializer_lists for shape and strides.
 *
 * All dimensions are treated as `Any`.
 *
 * @tparam Ndim  Rank.
 * @tparam T     Element type.
 * @param data   Data pointer (non-owning).
 * @param shape_vals   Shape values (must have Ndim entries).
 * @param stride_vals  Stride values (must have Ndim entries).
 */
template <int Ndim, typename T>
constexpr auto make_tensor(
    T* data,
    std::initializer_list<nint_t> shape_vals,
    std::initializer_list<nint_t> stride_vals
) {
  return make_tensor(data, make_layout<Ndim>(shape_vals, stride_vals));
}

/**
 * @brief Create a Tensor of rank Ndim from a shape initializer_list only.
 *
 * Strides are auto-computed as row-major contiguous (last dim stride = 1,
 * each preceding stride = product of subsequent sizes).
 *
 * @tparam Ndim  Rank.
 * @tparam T     Element type.
 * @param data        Data pointer (non-owning).
 * @param shape_vals  Shape values (must have Ndim entries).
 */
template <int Ndim, typename T>
constexpr auto make_tensor(
    T* data,
    std::initializer_list<nint_t> shape_vals
) {
  return make_tensor(data, make_layout<Ndim>(shape_vals));
}

// ======================== Transpose ========================

/**
 * @brief Compile-time transpose: swap dimensions I and J, preserving type info.
 *
 * @tparam I       First dimension index.
 * @tparam J       Second dimension index.
 * @tparam T       Element type.
 * @tparam TShape  Shape type.
 * @tparam TStrides Strides type.
 * @param  t       The Tensor to transpose.
 * @return A new Tensor with dimensions I and J swapped.
 */
template <int I, int J, typename T, typename TShape, typename TStrides>
constexpr auto transpose(const Tensor<T, TShape, TStrides>& t) {
  auto new_layout = tensor::transpose<I, J>(t.layout());
  return make_tensor(t.data(), new_layout);
}

/**
 * @brief Runtime transpose: swap dimensions i and j.
 *
 * The return type degrades to all-`Any` Shape/Strides.
 *
 * @note Prefer `transpose<I, J>(t)` when indices are compile-time constants.
 *
 * @tparam T       Element type.
 * @tparam TShape  Shape type.
 * @tparam TStrides Strides type.
 * @param  t       The Tensor to transpose.
 * @param  i       First dimension index (runtime).
 * @param  j       Second dimension index (runtime).
 * @return A new Tensor with dimensions i and j swapped.
 */
template <typename T, typename TShape, typename TStrides>
constexpr auto transpose(const Tensor<T, TShape, TStrides>& t, int i, int j) {
  auto new_layout = tensor::transpose(t.layout(), i, j);
  return make_tensor(t.data(), new_layout);
}

// ======================== cast ========================

/**
 * @brief Cast a Tensor to a different Shape/Strides type.
 *
 * Convenience free function that forwards to `Tensor::as()`.
 * The target Shape and Strides must have the same rank as the source Tensor.
 *
 * @code
 * Array<float, 2> arr(data, {4, 5});
 * auto typed = cast<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>(arr);
 * @endcode
 *
 * @tparam TShape2   Target Shape type.
 * @tparam TStrides2 Target Strides type.
 * @param t          Source Tensor.
 * @return A new Tensor with the specified type parameters.
 */
template <typename TShape2, typename TStrides2, typename T, typename TShape, typename TStrides>
constexpr auto cast(const Tensor<T, TShape, TStrides>& t) {
  return t.template as<TShape2, TStrides2>();
}

// ======================== Continuity convenience functions ========================

/**
 * @brief Check whether the last N dimensions of a Tensor are contiguous.
 *
 * If the compile-time check already resolves to `true`, returns `true`
 * without any runtime cost.
 *
 * @tparam N       Number of trailing dimensions (default: 1).
 * @tparam T       Element type.
 * @tparam TShape  Shape type.
 * @tparam TStrides Strides type.
 * @param  t       The Tensor.
 * @return `true` if the last N dimensions are row-major contiguous.
 */
template <int N = 1, typename T, typename TShape, typename TStrides>
bool is_last_contiguous(const Tensor<T, TShape, TStrides>& t) {
  return t.template is_last_contiguous<N>();
}

/**
 * @brief Check whether all dimensions of a Tensor are contiguous.
 *
 * @tparam T       Element type.
 * @tparam TShape  Shape type.
 * @tparam TStrides Strides type.
 * @param  t       The Tensor.
 * @return `true` if the Tensor is fully row-major contiguous.
 */
template <typename T, typename TShape, typename TStrides>
bool is_contiguous(const Tensor<T, TShape, TStrides>& t) {
  return t.is_contiguous();
}

} // namespace vecops::tensor

#endif // VECOPS_TENSOR_H
