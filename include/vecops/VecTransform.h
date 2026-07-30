//
// Created by renyz on 2026/7/9.
//

#ifndef VECOPS_VECTRANSFORM_H
#define VECOPS_VECTRANSFORM_H

#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"

#include <tuple>
#include <type_traits>
#include <utility>

/**
 * @file VecTransform.h
 * @brief Vector-level transforms with optional logical coordinates.
 *
 * This header defines small adapter types for fused vector transforms. A
 * transform receives an output vector tag, an input vector, and zero or more
 * logical coordinates:
 *
 * @code
 * auto out = transform(out_tag, in_vec, coords...);
 * @endcode
 *
 * The coordinate pack is deliberately opaque to this layer. Its length and
 * meaning are a contract between the caller and the transform implementation.
 * For example, a tensor accessor may pass all logical coordinates while a GEMM
 * kernel may pass only tile-local row and column coordinates.
 *
 * ## Key components
 *
 * | Component                    | Purpose                                      |
 * |------------------------------|----------------------------------------------|
 * | `VecTransform`               | Base traits and vector-size limits           |
 * | `LambdaVecTransform`         | Wrap a user callable as a transform          |
 * | `ConvertedVecTransform`      | Adapt an existing transform to new dtypes    |
 * | `ZeroVecTransform`           | Built-in transform that returns zeros        |
 * | `IdentityVecTransform`       | Built-in conversion/identity transform       |
 * | `make_vec_transform()`       | Factory for coordinate-aware callables       |
 * | `make_elementwise_vec_transform()` | Factory for coordinate-free callables  |
 * | `adapt_vec_transform()`      | Wrap or convert a callable/transform         |
 *
 * ## Callable signatures
 *
 * Coordinate-aware callables are invoked as:
 *
 * @code
 * fn(out_tag, in_vec, nint_t coords...)
 * @endcode
 *
 * Elementwise callables are invoked as:
 *
 * @code
 * fn(out_tag, in_vec)
 * @endcode
 *
 * Elementwise transforms are not a separate class hierarchy. They are ordinary
 * transforms with `is_elementwise == true`; coordinate arguments are accepted
 * by the wrapper and ignored before calling the wrapped callable.
 *
 * ## Vector-size adaptation
 *
 * A wrapped callable may implement only a subset of vector POW2 tags. The
 * lambda adapter first tries the requested tag, then tries larger tags via
 * `vec::bitcast`, then recursively splits into half vectors. `ConvertedVecTransform`
 * additionally converts input and output element types through `vec::convert`.
 *
 * ## Pitfalls
 *
 * - Coordinates are vector-base coordinates, not per-lane indices.
 * - Tail masking is handled by the caller. A transform receives a full vector
 *   and should not assume that every lane is logically active.
 * - `is_elementwise` is a promise made by the transform type or factory; this
 *   layer cannot prove that a callable is truly coordinate-independent.
 * - Transform objects are called through `const operator()`. They may hold
 *   state and may read external state; any side effects, synchronization, or
 *   repeated-call semantics are the caller's responsibility.
 */

namespace vecops {
namespace details {

consteval int vec_transform_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

template <typename To, typename Ei>
static constexpr bool rebind_vec_supported_v =
#if defined(CPU_CAPABILITY_SVE)
    vec::is_scalable_tag<To> &&
    (vec::scale_power<vec::Rebind<Ei, To>> <= VEC_MAX_POW);
#else
    true;
#endif

template <typename To, typename Fn, typename Ei, typename CoordTuple, typename = void>
struct CanCallCoordinateImpl : std::false_type {};

template <typename To, typename Fn, typename Ei, typename... Coords>
struct CanCallCoordinateImpl<To, Fn, Ei, std::tuple<Coords...>, std::void_t<decltype(
    std::declval<const Fn&>()(
        std::declval<To>(),
        std::declval<vec::Vec<vec::Rebind<Ei, To>>>(),
        std::declval<Coords>()...)
)>> : std::true_type {};

template <typename To, typename Fn, typename Ei, typename = void>
struct CanCallElementwiseImpl : std::false_type {};

template <typename To, typename Fn, typename Ei>
struct CanCallElementwiseImpl<To, Fn, Ei, std::void_t<decltype(
    std::declval<const Fn&>()(
        std::declval<To>(),
        std::declval<vec::Vec<vec::Rebind<Ei, To>>>())
)>> : std::true_type {};

template <typename To, typename Fn, typename Ei, typename... Coords>
struct CanCallCoordinate : std::conditional_t<
    rebind_vec_supported_v<To, Ei>,
    CanCallCoordinateImpl<To, Fn, Ei, std::tuple<Coords...>>,
    std::false_type> {};

template <typename To, typename Fn, typename Ei>
struct CanCallElementwise : std::conditional_t<
    rebind_vec_supported_v<To, Ei>,
    CanCallElementwiseImpl<To, Fn, Ei>,
    std::false_type> {};

template <typename T, typename = void>
struct IsVecTransformLike : std::false_type {};

template <typename T>
struct IsVecTransformLike<T, std::void_t<
    typename std::remove_cvref_t<T>::TIn,
    typename std::remove_cvref_t<T>::TOut>>
    : std::true_type {};

} // namespace details

template <typename T>
static constexpr bool is_vec_transform_like_v =
    details::IsVecTransformLike<T>::value;

/**
 * @brief Base traits for vector transforms.
 *
 * `VecTransform` is an interface-by-convention base. Derived classes provide
 * `operator()(out_tag, in_vec, coords...)`; the base only defines the element
 * types, the elementwise marker, and the valid vector POW2 ranges implied by
 * the input/output element sizes.
 */
template <typename EOut, typename EIn, bool Elementwise = false>
struct VecTransform {
  static_assert(vec::Element<EIn> && vec::Element<EOut>,
                "EIn or EOut is not a supported vector element type");

  using TIn = EIn;
  using TOut = EOut;

  static constexpr bool is_elementwise = Elementwise;

  static constexpr bool is_widening = sizeof(EOut) > sizeof(EIn);
  static constexpr bool is_narrowing = sizeof(EOut) < sizeof(EIn);
  static constexpr int size_ratio =
      is_widening ? sizeof(EOut) / sizeof(EIn) : sizeof(EIn) / sizeof(EOut);
  static constexpr int pow2_shift = details::vec_transform_log2(size_ratio);

  static constexpr int max_input_pow2 = is_widening ? VEC_MAX_POW - pow2_shift : VEC_MAX_POW;
  static constexpr int min_input_pow2 = is_widening ? VEC_HW_MIN_POW - pow2_shift : VEC_HW_MIN_POW;
  static constexpr int max_output_pow2 = is_narrowing ? VEC_MAX_POW - pow2_shift : VEC_MAX_POW;
  static constexpr int min_output_pow2 = is_narrowing ? VEC_HW_MIN_POW - pow2_shift : VEC_HW_MIN_POW;
};

/**
 * @brief Wrap a callable as a vector transform.
 *
 * When `Elementwise` is true, `Fn` is called without coordinates and the
 * wrapper accepts any coordinate pack. Otherwise `Fn` is called with the
 * coordinate pack forwarded after conversion to `nint_t`.
 */
template <typename EOut, typename EIn, typename Fn, bool Elementwise = false>
struct LambdaVecTransform : public VecTransform<EOut, EIn, Elementwise> {
  using Base = VecTransform<EOut, EIn, Elementwise>;

  constexpr explicit LambdaVecTransform(Fn&& fn) : _fn(std::move(fn)) {}
  constexpr explicit LambdaVecTransform(const Fn& fn) : _fn(fn) {}

  template <vec::VectorTag To, typename... Coords>
    requires vec::is_scalable_tag<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power<To>) &&
             (vec::scale_power<To> <= Base::max_output_pow2)
  vec::Vec<To> operator()(To t, vec::Vec<vec::Rebind<EIn, To>> v_in, Coords... coords) const {
    constexpr int pow2 = vec::scale_power<To>;
    if constexpr (Elementwise) {
      ((void) coords, ...);
      return call_elementwise<To, pow2>(t, v_in);
    } else {
      return call_coordinate<To, pow2>(t, v_in, static_cast<nint_t>(coords)...);
    }
  }

private:
  Fn _fn;

  template <typename To, int Pow2>
  vec::Vec<To> call_elementwise(To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
    if constexpr (details::CanCallElementwise<To, Fn, EIn>::value) {
      return _fn(t, v_in);
    } else {
      return try_upward_elementwise<To, Pow2 + 1>(t, v_in);
    }
  }

  template <typename To, int Pow2, typename... Coords>
  vec::Vec<To> call_coordinate(To t, vec::Vec<vec::Rebind<EIn, To>> v_in, Coords... coords) const {
    if constexpr (details::CanCallCoordinate<To, Fn, EIn, Coords...>::value) {
      return _fn(t, v_in, coords...);
    } else {
      return try_upward_coordinate<To, Pow2 + 1>(t, v_in, coords...);
    }
  }

  template <typename To, int TryPow2>
  vec::Vec<To> try_upward_elementwise(To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
    if constexpr (TryPow2 > Base::max_input_pow2) {
      return try_downward_elementwise<To>(t, v_in);
    } else {
      using TryOut = vec::ScalableTag<EOut, TryPow2>;
      using TryIn = vec::Rebind<EIn, TryOut>;

      if constexpr (details::CanCallElementwise<TryOut, Fn, EIn>::value) {
        auto cast_in = vec::bitcast(TryIn{}, vec::Rebind<EIn, To>{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in);
        return vec::bitcast(To{}, TryOut{}, cast_out);
      } else {
        return try_upward_elementwise<To, TryPow2 + 1>(t, v_in);
      }
    }
  }

  template <typename To, int TryPow2, typename... Coords>
  vec::Vec<To> try_upward_coordinate(
      To t,
      vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const {
    if constexpr (TryPow2 > Base::max_input_pow2) {
      return try_downward_coordinate<To>(t, v_in, coords...);
    } else {
      using TryOut = vec::ScalableTag<EOut, TryPow2>;
      using TryIn = vec::Rebind<EIn, TryOut>;

      if constexpr (details::CanCallCoordinate<TryOut, Fn, EIn, Coords...>::value) {
        auto cast_in = vec::bitcast(TryIn{}, vec::Rebind<EIn, To>{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in, coords...);
        return vec::bitcast(To{}, TryOut{}, cast_out);
      } else {
        return try_upward_coordinate<To, TryPow2 + 1>(t, v_in, coords...);
      }
    }
  }

  template <typename To>
  vec::Vec<To> try_downward_elementwise(To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
    constexpr int pow2 = vec::scale_power<To>;

    if constexpr (details::CanCallElementwise<To, Fn, EIn>::value) {
      return _fn(t, v_in);
    } else if constexpr (pow2 > Base::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti = vec::Rebind<EIn, To>;
      auto v_lo = try_downward_elementwise<HalfTo>(HalfTo{}, vec::lower(Ti{}, v_in));
      auto v_hi = try_downward_elementwise<HalfTo>(HalfTo{}, vec::upper(Ti{}, v_in));
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > Base::min_input_pow2,
                    "LambdaVecTransform: Fn does not support any POW2 in range");
      return {};
    }
  }

  template <typename To, typename... Coords>
  vec::Vec<To> try_downward_coordinate(
      To t,
      vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const {
    constexpr int pow2 = vec::scale_power<To>;

    if constexpr (details::CanCallCoordinate<To, Fn, EIn, Coords...>::value) {
      return _fn(t, v_in, coords...);
    } else if constexpr (pow2 > Base::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti = vec::Rebind<EIn, To>;
      auto v_lo = try_downward_coordinate<HalfTo>(
          HalfTo{}, vec::lower(Ti{}, v_in), coords...);
      auto v_hi = try_downward_coordinate<HalfTo>(
          HalfTo{}, vec::upper(Ti{}, v_in), coords...);
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > Base::min_input_pow2,
                    "LambdaVecTransform: Fn does not support any POW2 in range");
      return {};
    }
  }
};

/**
 * @brief Adapt an existing transform to a new input/output element type.
 */
template <typename EOut, typename EIn, typename InnerTransform>
struct ConvertedVecTransform : public VecTransform<EOut, EIn, InnerTransform::is_elementwise> {
  using Base = VecTransform<EOut, EIn, InnerTransform::is_elementwise>;

  constexpr explicit ConvertedVecTransform(InnerTransform&& fn) : _fn(std::move(fn)) {}
  constexpr explicit ConvertedVecTransform(const InnerTransform& fn) : _fn(fn) {}

  template <vec::VectorTag To, typename... Coords>
    requires vec::is_scalable_tag<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power<To>) &&
             (vec::scale_power<To> <= Base::max_output_pow2)
  vec::Vec<To> operator()(To t, vec::Vec<vec::Rebind<EIn, To>> v_in, Coords... coords) const {
    using Ti = vec::Rebind<EIn, To>;
    using InnerIn = typename InnerTransform::TIn;
    using InnerOut = typename InnerTransform::TOut;
    constexpr int pow2_in = vec::scale_power<Ti>;

    if constexpr (pow2_in <= InnerTransform::max_input_pow2) {
      if constexpr (InnerTransform::min_input_pow2 <= pow2_in) {
        vec::Rebind<InnerIn, Ti> t_ii;
        vec::Rebind<InnerOut, Ti> t_io;
        auto inner_in = vec::convert(t_ii, Ti{}, v_in);
        auto inner_out = _fn(t_io, inner_in, static_cast<nint_t>(coords)...);
        return vec::convert(t, t_io, inner_out);
      } else {
        vec::Rebind<InnerIn, Ti> t_ii;
        vec::ScalableTag<InnerIn, InnerTransform::min_input_pow2> t_ix;
        vec::Rebind<InnerOut, decltype(t_ix)> t_ox;
        vec::Rebind<InnerOut, Ti> t_io;
        auto converted_in = vec::convert(t_ii, Ti{}, v_in);
        auto inner_in = vec::bitcast(t_ix, t_ii, converted_in);
        auto inner_out = _fn(t_ox, inner_in, static_cast<nint_t>(coords)...);
        auto resized_out = vec::bitcast(t_io, t_ox, inner_out);
        return vec::convert(t, t_io, resized_out);
      }
    } else {
      using Th = vec::Half<To>;
      Ti ti;
      auto v_lo = (*this)(Th{}, vec::lower(ti, v_in), static_cast<nint_t>(coords)...);
      auto v_hi = (*this)(Th{}, vec::upper(ti, v_in), static_cast<nint_t>(coords)...);
      return vec::concat(t, v_lo, v_hi);
    }
  }

private:
  InnerTransform _fn;
};

template <typename EOut, typename EIn = EOut>
struct ZeroVecTransform : public VecTransform<EOut, EIn, true> {
  using Base = VecTransform<EOut, EIn, true>;

  template <vec::VectorTag To, vec::VectorValue Vi, typename... Coords>
    requires vec::is_scalable_tag<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power<To>) &&
             (vec::scale_power<To> <= Base::max_output_pow2)
  vec::Vec<To> operator()(To t, Vi, Coords... coords) const {
    ((void) coords, ...);
    return vec::zeros(t);
  }
};

template <typename EOut, typename EIn = EOut>
struct IdentityVecTransform : public VecTransform<EOut, EIn, true> {
  using Base = VecTransform<EOut, EIn, true>;

  template <vec::VectorTag To, typename... Coords>
    requires vec::is_scalable_tag<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power<To>) &&
             (vec::scale_power<To> <= Base::max_output_pow2)
  vec::Vec<To> operator()(To t, vec::Vec<vec::Rebind<EIn, To>> v_in, Coords... coords) const {
    ((void) coords, ...);
    return vec::convert(t, vec::Rebind<EIn, To>{}, v_in);
  }
};

template <typename EOut, typename EIn, typename Fn>
constexpr auto make_vec_transform(Fn&& fn) {
  using Transform = LambdaVecTransform<EOut, EIn, std::remove_cvref_t<Fn>, false>;
  return Transform(std::forward<Fn>(fn));
}

template <typename EOut, typename EIn, typename Fn>
constexpr auto make_elementwise_vec_transform(Fn&& fn) {
  using Transform = LambdaVecTransform<EOut, EIn, std::remove_cvref_t<Fn>, true>;
  return Transform(std::forward<Fn>(fn));
}

template <typename EOut, typename EIn, typename Fn>
constexpr auto adapt_vec_transform(Fn&& fn) {
  using FnT = std::remove_cvref_t<Fn>;
  if constexpr (is_vec_transform_like_v<FnT>) {
    if constexpr (is_any<typename FnT::TIn, EIn> && is_any<typename FnT::TOut, EOut>) {
      return FnT(std::forward<Fn>(fn));
    } else {
      return ConvertedVecTransform<EOut, EIn, FnT>(std::forward<Fn>(fn));
    }
  } else {
    return make_vec_transform<EOut, EIn>(std::forward<Fn>(fn));
  }
}

template <typename EOut, typename EIn = EOut>
inline constexpr ZeroVecTransform<EOut, EIn> zeros_transform {};

template <typename EOut, typename EIn = EOut>
inline constexpr IdentityVecTransform<EOut, EIn> identity_transform {};

namespace details {

template <typename T>
struct IsZeroVecTransform : std::false_type {};
template <typename EOut, typename EIn>
struct IsZeroVecTransform<ZeroVecTransform<EOut, EIn>> : std::true_type {};
template <typename EOut, typename EIn, typename InnerOut, typename InnerIn>
struct IsZeroVecTransform<ConvertedVecTransform<EOut, EIn, ZeroVecTransform<InnerOut, InnerIn>>>
    : std::true_type {};

template <typename T>
struct IsIdentityVecTransform : std::false_type {};
template <typename EOut, typename EIn>
struct IsIdentityVecTransform<IdentityVecTransform<EOut, EIn>> : std::true_type {};
template <typename EOut, typename EIn, typename InnerOut, typename InnerIn>
struct IsIdentityVecTransform<ConvertedVecTransform<EOut, EIn, IdentityVecTransform<InnerOut, InnerIn>>>
    : std::true_type {};

} // namespace details

template <typename T>
static constexpr bool is_zero_vec_transform_v =
    details::IsZeroVecTransform<std::remove_cvref_t<T>>::value;

template <typename T>
static constexpr bool is_identity_vec_transform_v =
    details::IsIdentityVecTransform<std::remove_cvref_t<T>>::value;

} // namespace vecops

#endif // VECOPS_VECTRANSFORM_H
