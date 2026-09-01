//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_TENSOR_LAYOUT_H
#define VECOPS_TENSOR_LAYOUT_H

#include <algorithm>
#include <array>
#include <ostream>
#include <tuple>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/Meta.h"
#include "vecops/util/TypeTraits.h"

namespace vecops::tensor {

using namespace ::vecops::meta;
namespace details {
using namespace ::vecops::meta::details;
} // namespace details

/**
 * @file Layout.h
 * @brief Logical tensor shapes and element-stride memory layouts.
 *
 * A `Layout` maps an N-dimensional logical coordinate to an element offset:
 *
 * @code
 * offset(i0, ..., iN) = i0 * stride(0) + ... + iN * stride(N)
 * @endcode
 *
 * Shape and stride entries use `meta::Const`, `meta::Dynamic`, or `meta::Any`,
 * so compile-time dimensions remain visible to optimizers while dynamic
 * layouts carry their values at runtime. Strides are measured in elements,
 * never bytes. The data pointer and element type belong to `Tensor`, not to a
 * `Layout`.
 *
 * ## Key components
 *
 * | Component      | Purpose                                                   |
 * |----------------|-----------------------------------------------------------|
 * | ArrayMeta      | Multi-dimensional typed integer arrays (base for Shape/Strides) |
 * | Shape          | Non-negative dimension sizes                              |
 * | Strides        | Stride values (may be negative)                           |
 * | Layout         | Pairs a Shape and Strides into a memory layout descriptor |
 * | Continuity     | Compile-time and run-time row-major contiguity checks     |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/tensor/Layout.h"
 * using namespace vecops::tensor;
 *
 * // Create compile-time shapes
 * auto sh = make_shape(cint<2>, cint<3>);       // Shape<Const<2>, Const<3>>
 * auto st = make_strides(cint<3>, cint<1>);     // Strides<Const<3>, Const<1>>
 *
 * // Create runtime shapes with alignment constraints
 * auto sh2 = make_shape(cint<16>, dyn<4>(128)); // Shape<Const<16>, Dynamic<4>>
 *
 * // Build a layout
 * auto layout = make_layout(sh, st);
 *
 * auto p = offset_at(layout, 1, 2);             // 1 * 3 + 2 * 1 == 5
 *
 * // Dynamic row-major layout; omitted strides are inferred.
 * auto dynamic = make_layout(make_shape(8, 128));
 *
 * // Layout transformations preserve type-level information where possible.
 * auto transposed = transpose<1, 0>(layout);
 * @endcode
 *
 * ## Logical versus physical properties
 *
 * Shape describes which coordinates are logically valid. Stride describes
 * how changing one logical coordinate changes the physical element offset.
 * Consequently:
 *
 * - stride 1 is physically contiguous along that axis;
 * - stride 0 is broadcasting/aliasing along that axis;
 * - a negative stride walks memory backwards and requires the Tensor's base
 *   pointer to already refer to the logical coordinate zero;
 * - arbitrary strides may overlap. Layout does not provide an ownership or
 *   non-aliasing guarantee.
 *
 * ## Pitfalls
 *
 * - Shape entries must be non-negative. Strides may be zero or negative.
 * - All offsets and strides are in elements; multiply by `sizeof(T)` only at a
 *   raw-byte boundary.
 * - Continuity helpers describe the layout order they name; a transpose can
 *   be dense without being row-major contiguous.
 * - Layout operations do not prove that the backing allocation covers every
 *   reachable offset, especially for negative or overlapping strides.
 * - Transformations such as `remove_dim`, `set_dim`, and `insert_dim` change
 *   the C++ type as well as the runtime values.
 * - Raw integer arguments become `meta::Any`, which may prevent compile-time
 *   continuity or tail elimination even when their runtime values are fixed.
 */

/**
 * @brief Base class for multi-dimensional typed integer arrays (Shape, Strides).
 *
 * `ArrayMeta<Is...>` is a fixed-rank container where each dimension's value
 * is represented by a Value subclass type (`Const<N>` or `Dynamic<A,L,H>`).
 * It uses `PackedStorage` internally to store only runtime values.
 *
 * ## Subclass usage
 *
 * Subclasses `Shape<Is...>` and `Strides<Is...>` enforce additional
 * semantic constraints (e.g., Shape values must be non-negative).
 *
 * ## Construction
 *
 * @code
 * // Explicit types
 * ArrayMeta<Const<2>, Dynamic<4>> m(2, 128);
 *
 * // Via make_shape
 * auto s = make_shape(cint<2>, 128);  // Shape<Const<2>, Any>
 * @endcode
 *
 * @tparam Is  Value subclass types for each dimension.
 */
template <typename... Is>
struct ArrayMeta {
  static_assert((std::is_base_of_v<Value, Is> && ...), "Is is not Value");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  constexpr ArrayMeta() = default;

  /**
   * Construct from per-dimension values. Types are deduced from the
   * parameter types via ValuePromote.
   */
  template <typename... Ints>
  constexpr explicit ArrayMeta(Ints... vs) {
    static_assert(sizeof...(Ints) == sizeof...(Is), "MatrixMeta: argument count mismatch");

    _stor = details::PackedStorage < Is...>{ Is{vs}... };
  }

  /**
   * Get the compile-time (if Const) or runtime value for dimension I.
   * @tparam I  Zero-based dimension index.
   */
  template <int I>
  VECOPS_ALWAYS_INLINE constexpr nint_t get() const {
    return _stor.template get<I>();
  }

  VECOPS_ALWAYS_INLINE constexpr nint_t operator[](int i) const {
    return _stor[i];
  }

  /**
   * Check whether dimension I is a compile-time constant.
   */
  template <int I>
  constexpr bool is_const() const {
    return !this->template is_runtime<I>();
  }

  /**
   * Check whether dimension I holds a runtime value.
   */
  template <int I>
  constexpr bool is_runtime() const {
    return decltype(_stor)::is_runtime[I];
  }

  constexpr int ndim() const {
    return Ndim;
  }

  details::PackedStorage<Is...> _stor;
}; // struct ArrayMeta

/**
 * @brief Multi-dimensional shape descriptor. All dimension values must be non-negative.
 *
 * `Shape<Is...>` inherits from `ArrayMeta<Is...>` and adds the constraint
 * that every dimension size >= 0. Construction asserts this.
 *
 * ## Usage
 *
 * @code
 * Shape<Const<2>, Const<3>> s(2, 3);          // OK
 * Shape<Const<2>> bad(-1);                     // RUNTIME ASSERTION
 *
 * auto s2 = make_shape(cint<2>, dyn<4>(128));  // Convenience factory
 * @endcode
 *
 * @tparam Is  Value types for each dimension.
 */
template <typename... Is>
struct Shape : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Shape(Ints... is) : ArrayMeta<Is...>(is...) {
    VECOPS_ASSERT(((nint_t(is) >= 0) && ...), "is must be non-negative");
  }

  constexpr Shape() = default;
}; // struct Shape

/**
 * @brief Create a Shape, automatically wrapping bare integers as Any.
 *
 * @code
 * auto s = make_shape(cint<3>, 128);  // Shape<Const<3>, Any>
 * @endcode
 */
template <typename... Ints>
constexpr auto make_shape(Ints&& ... is) -> Shape<to_value_t<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

/**
 * @brief Multi-dimensional stride descriptor. Stride values may be negative
 *        (for reversed dimensions).
 *
 * Unlike `Shape`, `Strides` does **not** enforce a sign constraint.
 *
 * @tparam Is  Value types for each dimension.
 */
template <typename... Is>
struct Strides : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Strides(Ints... is) : ArrayMeta<Is...>(is...) {
  }

  constexpr Strides() = default;
}; // class Strides

/**
 * @brief Create Strides, automatically wrapping bare integers as Any.
 */
template <typename... Ints>
constexpr auto make_strides(Ints&& ... is) -> Strides<to_value_t<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

namespace details {

// --- Type trait helpers for ArrayMeta / Shape / Strides ---

template <typename T>
struct IsArrayMeta : std::bool_constant<
    is_specialization_of_v<ArrayMeta, T> ||
    is_specialization_of_v<Shape, T> ||
    is_specialization_of_v<Strides, T>> {};


/**
 * @brief Compile-time dimension removal from an ArrayMeta type.
 *
 * Removes dimension `J` from the type `Meta<InIs...>`, producing
 * `Meta<OutIs..., InIs_without_J...>`.
 *
 * This is a compile-time O(N) operation implemented via recursive
 * template specialization with two nested Holder levels (two variadic
 * packs require double nesting).
 *
 * @tparam Meta  The target container template (e.g., Shape, Strides, ArrayMeta).
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (internal recursion counter).
 * @tparam J     Target dimension to remove.
 * @tparam InIs  Remaining input types.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InI0,
    typename... InIs
>
struct ArrayMetaRemoveDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InI0,
    typename... InIs
>
  requires (I < J)
struct ArrayMetaRemoveDim<Meta, N, I, J, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using Inner = typename ArrayMetaRemoveDim<Meta, N, I + 1, J, InIs...>
    ::template Holder<OutIs..., InI0>;
    using type = Inner::type;

    constexpr type transform(const Meta<OutIs..., InI0, InIs...>& m) {
      return Inner{}.transform(m);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InI0,
    typename... InIs
>
struct ArrayMetaRemoveDim<Meta, N, I, I, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using type = Meta<OutIs..., InIs...>;  // InI0 removed

    constexpr type transform(const Meta<OutIs..., InI0, InIs...>& m) {
      std::array<nint_t, N - 1> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      std::copy(in.data() + I + 1, in.data() + N, out.data() + I);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return type{out[Idx]...};
      }(std::make_index_sequence<N - 1>{});
    }
  };
};

template <int I, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto remove_dim(const TMeta<Is...>& m) {
  return typename details::ArrayMetaRemoveDim<TMeta, int(sizeof...(Is)), 0, I, Is...>::template Holder<>{}.transform(m);
}


/**
 * @brief Compile-time dimension replacement in an ArrayMeta type.
 *
 * Replaces dimension `J` with a new Value type `InNew`.
 *
 * @tparam Meta  The target container template.
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (recursion counter).
 * @tparam J     Target dimension to replace.
 * @tparam InNew The new Value type for dimension J.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename InI0,
    typename... InIs
>
struct ArrayMetaSetDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename InI0,
    typename... InIs
>
  requires (I < J)
struct ArrayMetaSetDim<Meta, N, I, J, InNew, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using Inner = typename ArrayMetaSetDim<Meta, N, I + 1, J, InNew, InIs...>
    ::template Holder<OutIs..., InI0>;
    using type = Inner::type;

    constexpr type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      return Inner{}.transform(m, v);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InNew,
    typename InI0,
    typename... InIs
>
struct ArrayMetaSetDim<Meta, N, I, I, InNew, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using type = Meta<OutIs..., InNew, InIs...>;  // InI0 replaced by InNew

    constexpr type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      std::array<nint_t, N> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      out[I] = nint_t(v);
      std::copy(in.data() + I + 1, in.data() + N, out.data() + I + 1);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return type{out[Idx]...};
      }(std::make_index_sequence<N>{});
    }
  };
};

template <int I, typename Inew, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto set_dim(const TMeta<Is...>& m, Inew v) {
  return typename details::ArrayMetaSetDim<TMeta, int(sizeof...(Is)), 0, I, Inew, Is...>::template Holder<>{}.transform(m, v);
}


/**
 * @brief Compile-time dimension insertion into an ArrayMeta type.
 *
 * Inserts a new Value type `InNew` at position `J` (0 <= J <= N).
 * When `J == N`, the dimension is appended at the end.
 *
 * @tparam Meta  The target container template.
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (recursion counter).
 * @tparam J     Target position for insertion.
 * @tparam InNew The Value type to insert at position J.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename... InIs
>
struct ArrayMetaInsertDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename InI0,
    typename... InIs
>
  requires (I < J)
struct ArrayMetaInsertDim<Meta, N, I, J, InNew, InI0, InIs...> {
  static_assert(0 <= I && I <= N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using Inner = typename ArrayMetaInsertDim<Meta, N, I + 1, J, InNew, InIs...>
    ::template Holder<OutIs..., InI0>;
    using type = Inner::type;

    constexpr type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      return Inner{}.transform(m, v);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InNew,
    typename... InIs
>
struct ArrayMetaInsertDim<Meta, N, I, I, InNew, InIs...> {
  static_assert(0 <= I && I <= N, "I out of range");

  template <typename... OutIs>
  struct Holder {
    using type = Meta<OutIs..., InNew, InIs...>;

    constexpr type transform(const Meta<OutIs..., InIs...>& m, InNew v) {
      std::array<nint_t, N + 1> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      out[I] = nint_t(v);
      std::copy(in.data() + I, in.data() + N, out.data() + I + 1);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return type{out[Idx]...};
      }(std::make_index_sequence<N + 1>{});
    }
  };
};

template <int I, typename Inew, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto insert_dim(const TMeta<Is...>& m, Inew v) {
  return typename details::ArrayMetaInsertDim<TMeta, int(sizeof...(Is)), 0, I, Inew, Is...>::template Holder<>{}.transform(m, v);
}

} // namespace details

/// Type trait: `true` if T is an ArrayMeta, Shape, or Strides.
template <typename T>
inline constexpr bool is_array_meta_v = details::IsArrayMeta<T>::value;
/// Concept form of is_array_meta_v.
template <typename T>
concept ArrayMetaLike = is_array_meta_v<std::remove_cvref_t<T>>;
/// Type trait: `true` if T is a Shape.
template <typename T>
inline constexpr bool is_shape_v = is_specialization_of_v<Shape, T>;
/// Type trait: `true` if T is a Strides.
template <typename T>
inline constexpr bool is_strides_v = is_specialization_of_v<Strides, T>;

/**
 * @brief Get the value of dimension I from an ArrayMeta.
 *
 * Returns `constexpr` if the dimension is a compile-time `Const<N>`.
 *
 * @tparam I     Dimension index.
 * @tparam TMeta The ArrayMeta type (Shape, Strides, or ArrayMeta itself).
 * @param  m     The metadata container.
 * @return The integer value for dimension I.
 */
template <int I, ArrayMetaLike TMeta>
VECOPS_ALWAYS_INLINE constexpr nint_t get(const TMeta& m) {
  return m.template get<I>();
}

/**
 * @brief Check whether dimension I of an ArrayMeta is a compile-time constant.
 */
template <int I, ArrayMetaLike TMeta>
constexpr bool is_const(const TMeta& m) {
  return m.template is_const<I>();
}

/**
 * @brief Check whether dimension I of an ArrayMeta is a runtime value.
 */
template <int I, ArrayMetaLike TMeta>
constexpr bool is_runtime(const TMeta& m) {
  return m.template is_runtime<I>();
}

namespace details {

// ======================== Lenient metadata matching ========================

/**
 * @brief Check whether a Shape or Strides type can be converted to another
 *        metadata type dimension by dimension.
 *
 * Returns true only when the source and destination are both `Shape<...>` or
 * both `Strides<...>`, have the same rank, and each source dimension can be
 * converted to the corresponding destination dimension according to
 * `IsMoreLenientValue`.
 */
template <typename MSrc, typename MDst>
struct IsMoreLenientMeta : std::false_type {};

template <bool SameRank, typename MSrc, typename MDst>
struct IsMoreLenientMetaImpl : std::false_type {};

template <template <typename...> class Meta, typename... Ss, typename... Ds>
struct IsMoreLenientMetaImpl<true, Meta<Ss...>, Meta<Ds...>>
    : std::bool_constant<(IsMoreLenientValue<Ss, Ds>::value && ...)> {};

template <template <typename...> class Meta, typename... Ss, typename... Ds>
struct IsMoreLenientMeta<Meta<Ss...>, Meta<Ds...>>
    : IsMoreLenientMetaImpl<sizeof...(Ss) == sizeof...(Ds), Meta<Ss...>, Meta<Ds...>> {};

/**
 * @brief Match one metadata Value against a lenient pattern.
 *
 * The `any` wildcard accepts every Value type. Non-wildcard patterns reuse
 * `IsMoreLenientValue`, so `Const<N>` can match compatible `Dynamic<A,L,H>`
 * patterns, `Dynamic` can match less constrained `Dynamic` patterns, and so on.
 */
template <typename Actual, typename Pattern>
struct IsLenientPatternValue : IsMoreLenientValue<Actual, Pattern> {};

template <typename Actual>
struct IsLenientPatternValue<Actual, any> : std::true_type {};

template <bool SameRank, typename Meta, typename... Patterns>
struct IsLenientPatternMetaImpl : std::false_type {};

template <template <typename...> class Meta, typename... Ss, typename... Patterns>
struct IsLenientPatternMetaImpl<true, Meta<Ss...>, Patterns...>
    : std::bool_constant<(IsLenientPatternValue<Ss, Patterns>::value && ...)> {};

/**
 * @brief Check whether a Shape or Strides type matches a lenient pattern list.
 *
 * Rank must match exactly. Each pattern is either the `any` wildcard or a
 * Value type accepted by `IsMoreLenientValue`.
 */
template <typename Meta, typename... Patterns>
struct IsLenientPatternMeta : std::false_type {};

template <template <typename...> class Meta, typename... Ss, typename... Patterns>
struct IsLenientPatternMeta<Meta<Ss...>, Patterns...>
    : IsLenientPatternMetaImpl<sizeof...(Ss) == sizeof...(Patterns), Meta<Ss...>, Patterns...> {};

} // namespace details

/**
 * @brief Check whether a Shape or Strides type matches a lenient pattern list.
 *
 * This is a static type predicate. It does not inspect runtime shape or stride
 * values. Use `any` as a per-dimension wildcard, and use Value types such as
 * `Const<N>`, `Dynamic<A,L,H>`, or `Any` for constrained dimensions.
 *
 * @code
 * using St = Strides<Const<64>, Const<8>, Const<2>, Const<2>>;
 * static_assert(is_lenient_v<St, any, any, Dynamic<1, -1, 3>, Const<2>>);
 * @endcode
 */
template <typename Meta, typename... Patterns>
inline constexpr bool is_lenient_v =
    details::IsLenientPatternMeta<std::remove_cvref_t<Meta>, Patterns...>::value;


/**
 * @brief Memory layout descriptor pairing a Shape and Strides.
 *
 * `Layout<TShape, TStrides>` describes the memory layout for a
 * multi-dimensional tensor: the shape specifies the size of each
 * dimension, and the strides specify the spacing (in elements) between
 * consecutive entries along each dimension. The mapping itself is fully
 * arbitrary; only the shape-only `make_layout(shape)` factory infers
 * row-major strides.
 *
 * ## Template parameters
 *
 * - `TShape`: A `Shape<...>` type encoding per-dimension sizes.
 * - `TStrides`: A `Strides<...>` type encoding per-dimension strides.
 *
 * Both must have the same rank (number of dimensions).
 *
 * ## Usage
 *
 * @code
 * auto s = make_shape(cint<2>, cint<3>);
 * auto st = make_strides(cint<3>, cint<1>);
 * auto layout = make_layout(s, st);
 *
 * // or directly
 * auto layout2 = make_layout(make_shape(2, 3), make_strides(3, 1));
 * @endcode
 *
 * @tparam TShape   Shape type.
 * @tparam TStrides Strides type.
 */
template <typename TShape, typename TStrides>
struct Layout {
  static_assert(is_shape_v<TShape>, "TShape must be Shape<_,_>");
  static_assert(is_strides_v<TStrides>, "TStride must be Stride<_,_>");
  static_assert(TShape::Ndim == TStrides::Ndim, "TShape and TStride must have the same rank");
  static constexpr int Ndim = TShape::Ndim;

  using Shape = TShape;
  using Stride = TStrides;
  using Strides = TStrides;

  VECOPS_ALWAYS_INLINE constexpr Layout(TShape shape, TStrides stride)
      : _shape(shape), _stride(stride) {
  }

  VECOPS_ALWAYS_INLINE constexpr const TShape& shape() const {
    return _shape;
  }

  VECOPS_ALWAYS_INLINE constexpr const TStrides& strides() const {
    return _stride;
  }

  VECOPS_ALWAYS_INLINE constexpr int ndim() const {
    return Ndim;
  }

  /**
   * @brief Reinterpret this Layout with different Shape and Strides metadata.
   *
   * Runtime shape and stride values are copied from the current layout into
   * the target metadata types. Target constructors enforce their own
   * constraints, so converting to incompatible `Const<N>` or
   * `Dynamic<A,L,H>` types triggers the same assertions as direct
   * construction.
   *
   * @code
   * auto layout = make_layout(make_shape(Any{4}, Any{5}),
   *                           make_strides(Any{5}, Any{1}));
   * auto typed = layout.as<Shape<Const<4>, Const<5>>,
   *                        Strides<Const<5>, Const<1>>>();
   * @endcode
   *
   * @tparam TShape2   Target Shape type with the same rank.
   * @tparam TStrides2 Target Strides type with the same rank.
   */
  template <typename TShape2, typename TStrides2>
  constexpr Layout<TShape2, TStrides2> as() const {
    static_assert(is_shape_v<TShape2>, "TShape2 must be Shape<...>");
    static_assert(is_strides_v<TStrides2>, "TStrides2 must be Strides<...>");
    static_assert(TShape2::Ndim == Ndim, "Target Shape rank must match Layout rank");
    static_assert(TStrides2::Ndim == Ndim, "Target Strides rank must match Layout rank");

    auto s_arr = _shape._stor.to_array();
    auto t_arr = _stride._stor.to_array();
    return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
      return Layout<TShape2, TStrides2>{
          TShape2{s_arr[Idx]...},
          TStrides2{t_arr[Idx]...}
      };
    }(std::make_index_sequence<Ndim>{});
  }

  /**
   * @brief Implicit conversion to a more lenient Layout type.
   *
   * A Layout can convert implicitly only when both Shape and Strides metadata
   * move toward equal-or-more-lenient Value types. Examples include
   * `Const<N>` to compatible `Dynamic<A,L,H>`, `Const<N>` to `Any`, and
   * constrained `Dynamic` to less constrained `Dynamic`. More strict
   * conversions remain explicit through `as()`.
   */
  template <typename TShape2, typename TStrides2>
  requires (!(std::same_as<TShape, TShape2> && std::same_as<TStrides, TStrides2>) && details::IsMoreLenientMeta<TShape, TShape2>::value && details::IsMoreLenientMeta<TStrides, TStrides2>::value)
  constexpr operator Layout<TShape2, TStrides2>() const {
    return as<TShape2, TStrides2>();
  }

private:
  TShape _shape;
  TStrides _stride;
}; // struct Layout

namespace details {

// ---- Repeat type helper ----

template <int N, template <typename...> class Meta, typename V, typename... Acc>
struct RepeatImpl {
  using type = typename RepeatImpl<N - 1, Meta, V, V, Acc...>::type;
};

template <template <typename...> class Meta, typename V, typename... Acc>
struct RepeatImpl<0, Meta, V, Acc...> {
  using type = Meta<Acc...>;
};

template <int N, template <typename...> class Meta, typename V>
using repeat_t = typename RepeatImpl<N, Meta, V>::type;

// ---- Inferred contiguous Strides from Shape ----

/// Compute the product type of a pack of Value types (right-to-left accumulation).
template <typename... Ts> struct Product;
template <> struct Product<> { using type = Const<1>; };
template <typename T> struct Product<T> { using type = T; };
template <typename T0, typename T1, typename... Ts>
struct Product<T0, T1, Ts...> {
  using type = decltype(std::declval<T0>() * std::declval<typename Product<T1, Ts...>::type>());
};

/// Extract types from index I to end of pack.
template <int I, typename... Ts> struct SuffixOf;
template <int I, typename T0, typename... Ts>
struct SuffixOf<I, T0, Ts...> {
  using type = std::conditional_t<(I == 0), std::tuple<T0, Ts...>,
      typename SuffixOf<I - 1, Ts...>::type>;
};
template <int I> struct SuffixOf<I> { using type = std::tuple<>; };

/// Compute the product of types in a std::tuple.
template <typename Tuple> struct TupleProduct;
template <typename... Ts>
struct TupleProduct<std::tuple<Ts...>> : Product<Ts...> {};

/// Compute the stride type for dimension I: product of sizes[I+1..N-1].
template <int I, typename TShape> struct StrideTypeForDim;
template <int I, typename... Ss>
struct StrideTypeForDim<I, Shape<Ss...>> {
  using suffix = typename SuffixOf<I + 1, Ss...>::type;
  using type = typename TupleProduct<suffix>::type;
};

/// Compute the full Strides<...> type from a Shape<...>.
template <typename TShape> struct InferredStrides;
template <typename... Ss>
struct InferredStrides<Shape<Ss...>> {
  template <size_t... Idx>
  static auto deduce(std::index_sequence<Idx...>)
      -> Strides<typename StrideTypeForDim<Idx, Shape<Ss...>>::type...>;
  using type = decltype(deduce(std::make_index_sequence<sizeof...(Ss)>{}));
};

} // namespace details

// ======================== make_layout ========================

/**
 * @brief Create a Layout, forwarding the shape and strides arguments.
 */
template <typename TShape, typename TStrides>
constexpr auto make_layout(TShape&& shape, TStrides&& stride) -> Layout<std::remove_cvref_t<TShape>, std::remove_cvref_t<TStrides>> {
  return {std::forward<TShape>(shape), std::forward<TStrides>(stride)};
}

/**
 * @brief Create a Layout from a typed Shape only; Strides are inferred as
 *        row-major contiguous, preserving compile-time type constraints.
 *
 * @code
 * auto s = make_shape(cint<2>, Any{5}, cint<4>, cint<3>);
 * auto L = make_layout(s);
 * // L.strides() has type Strides<Dynamic<4>, Const<12>, Const<3>, Const<1>>
 * @endcode
 */
template <typename TShape>
constexpr auto make_layout(TShape&& shape) -> Layout<
    std::remove_cvref_t<TShape>,
    typename details::InferredStrides<std::remove_cvref_t<TShape>>::type
> {
  using S = std::remove_cvref_t<TShape>;
  static_assert(S::Ndim > 0, "Shape must have at least 1 dimension");
  constexpr int N = S::Ndim;

  std::array<nint_t, N> stride_vals{};
  stride_vals[N - 1] = 1;
  for (int i = N - 2; i >= 0; --i)
    stride_vals[i] = stride_vals[i + 1] * nint_t(shape[i + 1]);

  using St = typename details::InferredStrides<S>::type;
  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>(
        std::forward<TShape>(shape),
        St{stride_vals[Idx]...}
    );
  }(std::make_index_sequence<N>{});
}

/**
 * @brief Create a Layout of rank Ndim from initializer_lists for shape and
 *        strides. All dimensions are treated as `Any`.
 */
template <int Ndim>
constexpr auto make_layout(
    std::initializer_list<nint_t> shape_vals,
    std::initializer_list<nint_t> stride_vals
) {
  static_assert(Ndim > 0, "Ndim must be positive");
  VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");
  VECOPS_ASSERT(stride_vals.size() == Ndim, "stride_vals.size() != Ndim");

  using S = details::repeat_t<Ndim, Shape, Any>;
  using St = details::repeat_t<Ndim, Strides, Any>;

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>{
        S{Any{*(shape_vals.begin() + (nint_t)Idx)}...},
        St{Any{*(stride_vals.begin() + (nint_t)Idx)}...}
    };
  }(std::make_index_sequence<Ndim>{});
}

namespace details {

/**
 * Strides that traverse @p shape in row-major order with @p unit_axis as the
 * fastest-varying (unit-stride) dimension; defaults to the last axis.
 */
template <std::size_t N>
constexpr std::array<nint_t, N> row_major_strides(
    const std::array<nint_t, N>& shape, int unit_axis = int(N) - 1) {
  std::array<nint_t, N> strides{};
  strides[unit_axis] = 1;
  nint_t product = shape[unit_axis];
  for (int d = N - 1; d >= 0; --d) {
    if (d == unit_axis) continue;
    strides[d] = product;
    product *= shape[d];
  }
  return strides;
}

} // namespace details

/**
 * @brief Create a Layout of rank Ndim from a shape initializer_list only.
 *        Strides are auto-computed as row-major contiguous, preserving the
 *        trailing `Const<1>` stride even though shape values are dynamic.
 */
template <int Ndim>
constexpr auto make_layout(
    std::initializer_list<nint_t> shape_vals
) {
  static_assert(Ndim > 0, "Ndim must be positive");
  VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");

  using S = details::repeat_t<Ndim, Shape, Any>;
  using St = typename details::InferredStrides<S>::type;

  std::array<nint_t, Ndim> shapes{};
  std::copy(shape_vals.begin(), shape_vals.end(), shapes.begin());

  const std::array<nint_t, Ndim> stride_vals =
      details::row_major_strides(shapes);

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>{
        S{Any{shapes[Idx]}...},
        St{stride_vals[Idx]...}
    };
  }(std::make_index_sequence<Ndim>{});
}

namespace details {

template <typename T>
struct IsLayout : std::bool_constant<is_specialization_of_v<Layout, T>> {};

template <int I, typename TMeta>
struct ArrayMetaElement;

template <int I, template <typename...> class TMeta, typename... Values>
struct ArrayMetaElement<I, TMeta<Values...>> {
  static_assert(0 <= I && I < sizeof...(Values));
  using type = std::tuple_element_t<I, std::tuple<Values...>>;
};

template <typename TShape>
struct ShapeProduct;

template <typename... Values>
struct ShapeProduct<Shape<Values...>> {
  using type = typename Product<Values...>::type;
};

} // namespace details

/// Type trait: `true` if T is a Layout.
template <typename T>
inline constexpr bool is_layout_v = details::IsLayout<T>::value;
/// Concept form of is_layout_v.
template <typename T>
concept LayoutLike = is_layout_v<std::remove_cvref_t<T>>;

/// Total element count of a layout: the product of all dimension sizes.
template <LayoutLike Layout>
VECOPS_ALWAYS_INLINE nint_t numel(const Layout& layout) {
  nint_t result = 1;
  for (int d = 0; d < Layout::Ndim; ++d) result *= layout.shape()[d];
  return result;
}

template <int I, typename TMeta>
using meta_element_t = typename details::ArrayMetaElement<
    I, std::remove_cvref_t<TMeta>>::type;

template <int I, typename TLayout>
using size_type_t = meta_element_t<
    I, typename std::remove_cvref_t<TLayout>::Shape>;

template <int I, typename TLayout>
using stride_type_t = meta_element_t<
    I, typename std::remove_cvref_t<TLayout>::Strides>;

template <typename TLayout>
using numel_type_t = typename details::ShapeProduct<
    typename std::remove_cvref_t<TLayout>::Shape>::type;

/**
 * @brief Get the size (shape value) of dimension I from a Layout.
 *
 * @tparam I       Dimension index.
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return The size of dimension I (always non-negative).
 */
template <int I, LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr size_type_t<I, TLayout> size_value(
    const TLayout& layout) {
  using Size = size_type_t<I, TLayout>;
  return Size{get<I>(layout.shape())};
}

template <int I, LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr nint_t size(const TLayout& layout) {
  return get<I>(layout.shape());
}

/**
 * @brief Get the stride value of dimension I from a Layout.
 */
template <int I, LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr stride_type_t<I, TLayout> stride_value(
    const TLayout& layout) {
  using Stride = stride_type_t<I, TLayout>;
  return Stride{get<I>(layout.strides())};
}

template <int I, LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr nint_t stride(const TLayout& layout) {
  return get<I>(layout.strides());
}

/**
 * @brief Compute the linear storage offset for a full coordinate tuple.
 *
 * Bounds are checked with VECOPS_ASSERT, so release builds do not pay for the
 * validation when assertions are disabled.
 */
template <LayoutLike TLayout, typename Coordinates>
VECOPS_ALWAYS_INLINE constexpr nint_t offset_at_coordinates(
    const TLayout& layout,
    const Coordinates& coords) {
  using LayoutT = std::remove_cvref_t<TLayout>;
  nint_t offset = 0;
  VECOPS_UNROLL
  for (int d = 0; d < LayoutT::Ndim; ++d) {
    VECOPS_ASSERT(0 <= coords[d] && coords[d] < layout.shape()[d], "index out of range");
    offset += coords[d] * layout.strides()[d];
  }
  return offset;
}

template <LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr nint_t offset_at(
    const TLayout& layout,
    const ::vecops::details::InlineArray<
        nint_t, std::remove_cvref_t<TLayout>::Ndim>& coords) {
  return offset_at_coordinates(layout, coords);
}

template <LayoutLike TLayout>
VECOPS_ALWAYS_INLINE constexpr nint_t offset_at(
    const TLayout& layout,
    const std::array<nint_t, std::remove_cvref_t<TLayout>::Ndim>& coords) {
  return offset_at_coordinates(layout, coords);
}

/**
 * @brief Compute the linear storage offset from one integer coordinate per
 *        layout dimension.
 */
template <
    LayoutLike TLayout,
    typename... Is>
VECOPS_ALWAYS_INLINE constexpr nint_t offset_at(
    const TLayout& layout, Is... is) {
  using LayoutT = std::remove_cvref_t<TLayout>;
  static_assert(sizeof...(Is) == LayoutT::Ndim, "coordinate count must match layout rank");
  static_assert((std::is_integral_v<std::decay_t<Is>> && ...), "coordinates must be integers");
  return offset_at(
      layout,
      ::vecops::details::InlineArray<nint_t, LayoutT::Ndim>{
          static_cast<nint_t>(is)...});
}

/**
 * @brief Remove dimension I from an ArrayMeta (Shape or Strides).
 *
 * Returns a new ArrayMeta of rank `Ndim-1`.
 *
 * @note This is a compile-time O(N) type transformation — the return
 *       type carries the modified type parameter pack.
 */
template <int I, ArrayMetaLike TMeta>
constexpr auto remove(const TMeta& m) { return details::remove_dim<I>(m); }

/**
 * @brief Remove dimension I from a Layout (both shape and strides).
 */
template <int I, LayoutLike TLayout>
constexpr auto remove(const TLayout& m) {
  return make_layout(remove<I>(m.shape()), remove<I>(m.strides()));
}

/** @brief Keep the first N dimensions, fixing discarded trailing axes at 0. */
template <int N, LayoutLike TLayout>
constexpr auto take_leading(const TLayout& layout) {
  static_assert(1 <= N && N <= TLayout::Ndim);
  if constexpr (N == TLayout::Ndim) return layout;
  else return take_leading<N>(remove<N>(layout));
}

/** @brief Keep the last N dimensions, fixing discarded leading axes at 0. */
template <int N, LayoutLike TLayout>
constexpr auto take_trailing(const TLayout& layout) {
  static_assert(1 <= N && N <= TLayout::Ndim);
  if constexpr (N == TLayout::Ndim) return layout;
  else return take_trailing<N>(remove<0>(layout));
}

/**
 * @brief Set dimension I in an ArrayMeta to a new value.
 *
 * @tparam I     Dimension index.
 * @tparam TMeta ArrayMeta type.
 * @tparam Inew  New Value type for this dimension.
 * @param  m     The metadata container.
 * @param  v     The new value.
 * @return A new ArrayMeta with dimension I replaced.
 */
template <int I, ArrayMetaLike TMeta, typename Inew>
constexpr auto set(const TMeta& m, Inew v) { return details::set_dim<I>(m, v); }

/**
 * @brief Set dimension I in a Layout to new shape and stride values.
 */
template <int I, LayoutLike TLayout, typename IShapeNew, typename IStridesNew>
constexpr auto set(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(set<I>(m.shape(), v_size), set<I>(m.strides(), v_stride));
}

/**
 * @brief Insert a new dimension before position I in an ArrayMeta.
 *
 * If `I == Ndim`, the new dimension is appended after the last dimension.
 *
 * @tparam I     Insertion position (0 <= I <= Ndim).
 * @tparam TMeta ArrayMeta type.
 * @tparam Inew  Value type for the new dimension.
 * @param  m     The metadata container.
 * @param  v     The value for the new dimension.
 * @return A new ArrayMeta of rank `Ndim+1`.
 */
template <int I, ArrayMetaLike TMeta, typename Inew>
constexpr auto insert(const TMeta& m, Inew v) { return details::insert_dim<I>(m, v); }

/**
 * @brief Insert a new dimension before position I in a Layout
 *        (both shape and strides).
 */
template <int I, LayoutLike TLayout, typename IShapeNew, typename IStridesNew>
constexpr auto insert(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(insert<I>(m.shape(), v_size), insert<I>(m.strides(), v_stride));
}

// ======================== ArrayMeta Swap Dim ========================

namespace details {

/**
 * @brief Compile-time dimension swapping for an ArrayMeta type.
 *
 * Swaps dimensions I and J in the type parameter pack, producing a new
 * type with the dimensions exchanged at compile time.
 *
 * @tparam Meta  The target container template.
 * @tparam I     First dimension index to swap.
 * @tparam J     Second dimension index to swap.
 * @tparam Is    Original type parameter pack.
 */
template <
    template <typename... xIs> typename Meta,
    int I,
    int J,
    typename... Is
>
struct ArrayMetaSwapDim {
private:
  static constexpr int N = sizeof...(Is);
  static_assert(0 <= I && I < N, "I out of range");
  static_assert(0 <= J && J < N, "J out of range");

  template <int Idx>
  struct select {
    using type = std::conditional_t<
        Idx == I,
        std::tuple_element_t<J, std::tuple<Is...>>,
        std::conditional_t<
            Idx == J,
            std::tuple_element_t<I, std::tuple<Is...>>,
            std::tuple_element_t<Idx, std::tuple<Is...>>
        >
    >;
  };

  template <int... Idx>
  static constexpr Meta<typename select<Idx>::type...> _make_type(std::integer_sequence<int, Idx...>);

public:
  using type = decltype(_make_type(std::make_integer_sequence<int, N>{}));

  static constexpr type transform(const Meta<Is...>& m) {
    std::array<nint_t, N> arr = m._stor.to_array();
    std::swap(arr[I], arr[J]);
    return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
      return type{arr[Idx]...};
    }(std::make_index_sequence<N>{});
  }
};

template <int I, int J, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto swap_dim(const TMeta<Is...>& m) {
  return ArrayMetaSwapDim<TMeta, I, J, Is...>::transform(m);
}

} // namespace details

/**
 * @brief Compile-time transpose: swap dimensions I and J in a Layout, preserving
 *        Const type information.
 *
 * Unlike the runtime `transpose(layout, i, j)`, this overload preserves the
 * compile-time Const/Dynamic types for each dimension, producing the exact
 * swapped type at compile time.
 *
 * @code
 * auto s = make_shape(cint<2>, cint<3>, cint<4>);
 * auto st = make_strides(cint<12>, cint<4>, cint<1>);
 * auto L = make_layout(s, st);
 * auto Lt = transpose<0, 1>(L);
 * // Lt.shape() == Shape<Const<3>, Const<2>, Const<4>>
 * @endcode
 *
 * @tparam I       First dimension index to swap.
 * @tparam J       Second dimension index to swap.
 * @tparam TLayout Layout type.
 * @param  layout  The layout to transpose.
 * @return A new Layout with dimensions I and J exchanged, preserving type info.
 */
template <int I, int J, LayoutLike TLayout>
constexpr auto transpose(const TLayout& layout) {
  static_assert(0 <= I && I < TLayout::Ndim, "I out of range");
  static_assert(0 <= J && J < TLayout::Ndim, "J out of range");
  return make_layout(
      details::swap_dim<I, J>(layout.shape()),
      details::swap_dim<I, J>(layout.strides())
  );
}

/**
 * @brief Runtime transpose: swap dimensions i and j in a Layout.
 *
 * The return type degrades to all-`Any` (all dimensions become `Any`)
 * because the swap targets are runtime values.
 *
 * @note Prefer the compile-time overload `transpose<I, J>(layout)` when
 *       the swap indices are known at compile time.
 *
 * @tparam TLayout Layout type.
 * @param  layout  The layout to transpose.
 * @param  i       First dimension index (runtime).
 * @param  j       Second dimension index (runtime).
 * @return A new Layout with all-Any Shape and Strides.
 */
template <LayoutLike TLayout>
constexpr auto transpose(const TLayout& layout, int i, int j) {
  constexpr int ndim = TLayout::Ndim;
  VECOPS_ASSERT(0 <= i && i < ndim, "i(%d) out of range [0, %d)", i, ndim);
  VECOPS_ASSERT(0 <= j && j < ndim, "j(%d) out of range [0, %d)", j, ndim);

  auto shape_arr = layout.shape()._stor.to_array();
  auto stride_arr = layout.strides()._stor.to_array();
  std::swap(shape_arr[i], shape_arr[j]);
  std::swap(stride_arr[i], stride_arr[j]);

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return make_layout(
        make_shape(Any{shape_arr[Idx]}...),
        make_strides(Any{stride_arr[Idx]}...)
    );
  }(std::make_index_sequence<ndim>{});
}

// ======================== Continuity Traits ========================

namespace details {

/**
 * @brief Compile-time check: are the last N dimensions of TLayout contiguous?
 *
 * A layout is **row-major contiguous** when each stride equals the product
 * of all subsequent dimensions' sizes:
 *   stride[D] == shape[D+1] * stride[D+1]
 * and the last stride == 1.
 *
 * This trait performs the check at compile time using only `Const` dimension
 * information. If any involved dimension is non-Const, the check returns
 * `false` (conservative).
 *
 * @tparam TLayout  The Layout type.
 * @tparam N        How many trailing dimensions to check.
 */
template <typename TLayout, int N>
struct IsCtLastContiguous;

template <typename... Ss, typename... Ts, int N>
struct IsCtLastContiguous<Layout<Shape<Ss...>, Strides<Ts...>>, N> {
private:
  static constexpr int ndim = sizeof...(Ss);

  template <int D>
  static constexpr bool _check_one() {
    if constexpr (D < 0 || D > ndim) {
      return false;
    } else if constexpr (D >= ndim) {
      return true;
    } else if constexpr (D == ndim - 1) {
      using St = std::tuple_element_t<D, std::tuple<Ts...>>;
      if constexpr (!St::is_const) {
        return false;
      } else {
        return St::value == 1;
      }
    } else {
      using Sd = std::tuple_element_t<D, std::tuple<Ts...>>;
      using Sd1 = std::tuple_element_t<D + 1, std::tuple<Ts...>>;
      using Zd1 = std::tuple_element_t<D + 1, std::tuple<Ss...>>;
      if constexpr (!Sd::is_const) {
        return false;
      } else if constexpr (!Sd1::is_const) {
        return false;
      } else if constexpr (!Zd1::is_const) {
        return false;
      } else {
        return Sd::value == Zd1::value * Sd1::value && _check_one<D + 1>();
      }
    }
  }

public:
  static constexpr bool value = (N >= 0) && (N <= ndim) && _check_one<ndim - N>();
};

} // namespace details

/**
 * @brief Compile-time check: are the last N dimensions of TLayout contiguous?
 *
 * Evaluates at compile time. When all involved dimensions are `Const`,
 * can resolve to `true` without any runtime check.
 *
 * @tparam TLayout  Layout type.
 * @tparam N        Number of trailing dimensions to check.
 */
template <typename TLayout, int N>
inline constexpr bool is_ct_last_contiguous_v =
    details::IsCtLastContiguous<std::remove_cvref_t<TLayout>, N>::value;

/**
 * @brief Compile-time check: are all dimensions of TLayout contiguous?
 *
 * Equivalent to `is_ct_last_contiguous_v<TLayout, TLayout::Ndim>`.
 */
template <typename TLayout>
inline constexpr bool is_ct_contiguous_v =
    is_ct_last_contiguous_v<TLayout, std::remove_cvref_t<TLayout>::Ndim>;

/**
 * @brief Runtime check: are the last N dimensions of `layout` contiguous?
 *
 * If the compile-time check (`is_ct_last_contiguous_v`) already returns `true`,
 * this function returns `true` at compile time with no runtime cost.
 *
 * @tparam N       Number of trailing dimensions to check.
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return `true` if the last N dimensions are row-major contiguous.
 */
template <int N, LayoutLike TLayout>
inline bool is_last_contiguous(const TLayout& layout) {
  if constexpr (is_ct_last_contiguous_v<TLayout, N>) {
    return true;
  }
  if constexpr (N <= 0) {
    return true;
  }
  int ndim = layout.ndim();
  if (N > ndim) return false;
  nint_t expected = 1;
  for (int d = ndim - 1; d >= ndim - N; --d) {
    if (layout.strides()[d] != expected) return false;
    expected *= layout.shape()[d];
  }
  return true;
}

/**
 * @brief Runtime check: are all dimensions of `layout` contiguous?
 *
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return `true` if the layout is fully row-major contiguous.
 */
template <LayoutLike TLayout>
inline bool is_contiguous(const TLayout& layout) {
  return is_last_contiguous<TLayout::Ndim>(layout);
}


// ============ I/O Support ============

/**
 * @brief Stream output for ArrayMeta (Shape / Strides).
 *
 * Prints as compact tuple form `(v0, v1, ...)`.
 *
 * Value decoration rules:
 *   - `Const<N>`             → `N!`
 *   - `Any{v}` (=Dynamic<1>) → `v`
 *   - `Dynamic<A,_,_>{v}`    → `v@A`  (when A > 1)
 *   - `Dynamic<A,Lo,Hi>{v}`  → `v@A[Lo,Hi]` (when bounds are active)
 */
template <typename... Is>
std::ostream& operator<<(std::ostream& os, const ArrayMeta<Is...>& m) {
  os << '(';
  [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    int i = 0;
    ([&] {
      if (i++) os << ", ";
      nint_t v = m.template get<Idx>();
      using DimType = std::tuple_element_t<Idx, std::tuple<Is...>>;
      os << v;
      if constexpr (DimType::is_const) {
        os << '!';
      } else {
        if constexpr (DimType::alignment > 1) {
          os << '@' << DimType::alignment;
        }
        if constexpr (DimType::has_lower && DimType::has_upper) {
          os << '[' << DimType::lo << ',' << DimType::hi << ']';
        } else if constexpr (DimType::has_lower) {
          os << '[' << DimType::lo << ",∞)";
        } else if constexpr (DimType::has_upper) {
          os << "[0," << DimType::hi << ']';
        }
      }
    }(), ...);
  }(std::index_sequence_for<Is...>{});
  os << ')';
  return os;
}

/**
 * @brief Stream output for Layout.
 *
 * Prints as `Layout(s=(...), st=(...))`.
 */
template <typename TShape, typename TStrides>
std::ostream& operator<<(std::ostream& os, const Layout<TShape, TStrides>& l) {
  os << "Layout(s=" << l.shape() << ", st=" << l.strides() << ')';
  return os;
}


} // namespace vecops::tensor

#endif // VECOPS_TENSOR_LAYOUT_H
