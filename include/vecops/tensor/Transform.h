// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_TENSOR_TRANSFORM_H
#define VECOPS_TENSOR_TRANSFORM_H

#include "vecops/CoreTypes.h"
#include "vecops/util/TypeTraits.h"
#include "vecops/tensor/AccessOptions.h"
#include "vecops/vec/Vec.h"

#include <tuple>
#include <type_traits>
#include <utility>

/**
 * @file Transform.h
 * @brief Pure vector transforms with an optional TransformContext.
 *
 * This header defines small adapter types for fused vector transforms. A
 * transform receives an output vector tag, an input vector, and zero or more
 * logical tensor context:
 *
 * @code
 * auto out = transform(out_tag, in_vec, context);
 * @endcode
 *
 * TransformContext retains the original full-rank coordinate, vector axis,
 * logical lane mapping and active lanes. lane_coord(i) returns the original
 * tensor coordinate of lane i, including through sliced specs.
 *
 * ## Key components
 *
 * | Component                    | Purpose                                      |
 * |------------------------------|----------------------------------------------|
 * | `VecTransform`               | Base traits and vector-size limits           |
 * | `LambdaVecTransform`         | Wrap a user callable as a transform          |
 * | `ZeroVecTransform`           | Built-in transform that returns zeros        |
 * | `IdentityVecTransform`       | Built-in conversion/identity transform       |
 * | `make_vec_transform()`       | Factory for coordinate-aware callables       |
 * | `make_lane_local_vec_transform()` | Coordinate-aware, lane-independent factory |
 * | `make_elementwise_vec_transform()` | Factory for coordinate-free callables  |
 * | `adapt_vec_transform()`      | Wrap or convert a callable/transform         |
 *
 * ## Callable signatures
 *
 * Coordinate-aware callables are invoked as:
 *
 * @code
 * fn(out_tag, in_vec, context)
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
 * by the wrapper and ignored before calling the wrapped callable. A
 * coordinate-aware transform may instead promise `is_lane_local == true`:
 * output lane i then depends only on input lane i, but may also depend on that
 * lane's logical coordinate or external per-lane parameters.
 *
 * ## Vector-size adaptation
 *
 * A wrapped callable may implement only a subset of scalable POW2 tags. The
 * lambda adapter first tries the requested output tag, then legal larger tags
 * via `vec::bitcast`, then recursively splits into half vectors. DataAccess
 * adds an outer partitioning layer: if rebinding the caller's requested Tag to
 * `TIn` or `TOut` exceeds backend limits, it recursively partitions the logical
 * lanes and invokes the transform as many times as necessary. Thus a transform
 * is not required to accept every Tag that a kernel may request.
 *
 * For example, an SVE `int8/P2` DataAccess result passing through a
 * `double -> double` transform may require eight transform invocations even
 * though it is one logical load. Each invocation receives a corresponding
 * `context.subspan()`, so `lane_coord(0)` still refers to the right original
 * lane.
 *
 * ## Pitfalls
 *
 * - Coordinate-aware transforms use context.lane_coord(i); a split high
 *   chunk receives context.subspan() with the correct lane base.
 * - Tail masking is handled by DataAccess. A transform receives a full vector;
 *   use `context.is_active(i)` if inactive lanes affect the formula.
 * - `is_elementwise` is a promise made by the transform type or factory; this
 *   layer cannot prove that a callable is truly coordinate-independent.
 * - Transforms are deterministic, side-effect-free functions. DataAccess may
 *   call them during materialization or split one logical request into
 *   multiple legal internal vector calls.
 * - A transform lambda that may be fused into an SME streaming kernel must be
 *   declared with `VECOPS_KERNEL_LAMBDA`. This makes the callable inherit the
 *   caller's streaming mode and forces inlining; it does not start or stop
 *   streaming mode. Ordinary helper functions use
 *   `VECOPS_KERNEL_FUNCTION(complete_function_declaration)`.
 */

namespace vecops::tensor {

/** @brief Active-lane descriptor in which every logical lane is active. */
struct AllActiveLanes {
  VECOPS_ALWAYS_INLINE constexpr bool is_active(nint_t) const {
    return true;
  }

  template <vec::VectorTag Tag>
  VECOPS_ALWAYS_INLINE constexpr auto active_option(Tag, nint_t) const {
    return vec::opt::unmasked;
  }
};

/** @brief Active-lane descriptor for the half-open range `[0, count)`. */
struct FirstActiveLanes {
  nint_t count;
  VECOPS_ALWAYS_INLINE constexpr bool is_active(nint_t lane) const {
    return 0 <= lane && lane < count;
  }

  template <vec::VectorTag Tag>
  VECOPS_ALWAYS_INLINE constexpr auto active_option(Tag, nint_t lane_base) const {
    return vec::opt::first(count - lane_base);
  }
};

/**
 * @brief Active-lane descriptor backed by a mask of the caller's Tag.
 * @note The referenced mask must outlive the TransformContext invocation.
 */
template <vec::VectorTag Tag>
struct MaskedActiveLanes {
  Tag tag;
  const vec::Mask<Tag>& mask;

  VECOPS_ALWAYS_INLINE bool is_active(nint_t lane) const {
    return vec::get(tag, mask, lane);
  }
};

/**
 * @brief Arbitrary logical lane offsets backed by an index vector.
 * @note Indices are offsets along the vector axis in Tensor elements, not
 * bytes or already-scaled physical addresses.
 */
template <vec::VectorTag IndexTag>
struct IndexedLaneMapping {
  IndexTag tag;
  const vec::Vec<IndexTag>& indices;

  VECOPS_ALWAYS_INLINE nint_t offset(nint_t lane) const {
    return static_cast<nint_t>(vec::get(tag, indices, lane));
  }
};

/**
 * @brief Logical context supplied to a coordinate-aware vector transform.
 *
 * `origin` is expressed in the original unsliced Tensor coordinates.
 * `vector_axis` identifies the original axis varied by vector lanes.
 * `lane_mapping` converts a lane number into an element offset on that axis,
 * while `active` describes which caller-visible lanes are valid.
 *
 * `lane_base` is maintained internally when one DataAccess request is split
 * into several transform calls. Transform code should normally call
 * `lane_coord(i)`, `is_active(i)`, or `active_option(tag)` rather than inspect
 * `lane_base` directly. `active_option(tag)` is available for the common
 * unmasked/first-active cases and preserves the active option's compile-time
 * kind, so a transform can forward it to an auxiliary vector memory access
 * without reconstructing a tail count. Arbitrary masked activity deliberately
 * has no prefix option; use `is_active(i)` when a transform truly needs it.
 *
 * @code
 * auto bias = tensor::make_vec_transform<float, float>(
 *     [](auto tag, auto x, const auto& ctx) {
 *       auto y = x;
 *       for (nint_t i = 0; i < vec::size(tag); ++i) {
 *         if (ctx.is_active(i)) {
 *           y = vec::set(tag, y, i,
 *                        vec::get(tag, x, i) + ctx.lane_coord(i)[0]);
 *         }
 *       }
 *       return y;
 *     });
 * @endcode
 */
template <
    std::size_t Rank,
    typename LaneMapping = ContiguousLaneMapping,
    typename Active = AllActiveLanes>
struct TransformContext {
  Coord<Rank> origin{};
  int vector_axis = static_cast<int>(Rank) - 1;
  LaneMapping lane_mapping{};
  Active active{};
  nint_t lane_base = 0;

  VECOPS_ALWAYS_INLINE Coord<Rank> lane_coord(nint_t lane) const {
    auto result = origin;
    result[static_cast<std::size_t>(vector_axis)] +=
        lane_mapping.offset(lane_base + lane);
    return result;
  }

  VECOPS_ALWAYS_INLINE bool is_active(nint_t lane) const {
    return active.is_active(lane_base + lane);
  }

  /** Return the caller's prefix-active option adjusted to this subspan. */
  template <vec::VectorTag Tag>
    requires requires(const Active& value, Tag tag, nint_t base) { value.active_option(tag, base); }
  VECOPS_ALWAYS_INLINE constexpr auto active_option(Tag tag) const {
    return active.active_option(tag, lane_base);
  }
  VECOPS_ALWAYS_INLINE TransformContext subspan(nint_t begin, nint_t) const {
    auto result = *this;
    result.lane_base += begin;
    return result;
  }
};

/** @brief Structural concept implemented by transform coordinate contexts. */
template <typename T>
concept TransformContextLike = requires(
    const T& context, nint_t lane, nint_t count) {
  context.lane_coord(lane);
  context.is_active(lane);
  context.subspan(lane, count);
};

/**
 * @brief Sentinel meaning that no prologue/epilogue transform exists.
 *
 * This is intentionally distinct from `IdentityVecTransform`. DataAccess uses
 * it to fuse `MemoryType <-> ComputeType` directly into
 * `vec::load_convert/store_convert`, avoiding an artificial transform stage.
 */
struct NoTransform {
  static constexpr bool is_elementwise = true;
  static constexpr bool is_lane_local = true;
  static constexpr bool permutation_equivariant = true;
  static constexpr bool reads_input = true;
};

/** Backend-selectable relationship between transform calls and memory stores. */
enum class TransformStoreMode { automatic, narrow, wide_transform, coalesced };

template <typename Transform>
inline constexpr TransformStoreMode transform_store_mode_v = [] {
  if constexpr (requires { std::remove_cvref_t<Transform>::transform_store_mode; }) {
    return std::remove_cvref_t<Transform>::transform_store_mode;
  } else {
    return TransformStoreMode::automatic;
  }
}();

namespace details {

template <typename T>
VECOPS_ALWAYS_INLINE decltype(auto) subspan_transform_argument(T&& value, nint_t begin, nint_t count) {
  if constexpr (TransformContextLike<std::remove_cvref_t<T>>) {
    return value.subspan(begin, count);
  } else {
    return std::forward<T>(value);
  }
}

consteval int vec_transform_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

template <typename To, typename Ei>
inline constexpr bool rebind_vec_supported_v =
#if defined(CPU_CAPABILITY_SVE)
    vec::is_scalable_tag_v<To> &&
    (vec::scale_power_v<vec::Rebind<Ei, To>> <= VEC_MAX_POW);
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
inline constexpr bool is_vec_transform_like_v =
    details::IsVecTransformLike<T>::value;

/**
 * @brief Base traits for vector transforms.
 *
 * `VecTransform` is an interface-by-convention base. Derived classes provide
 * `operator()(out_tag, in_vec, context)`; the base only defines the element
 * types, the transform-planning markers, and the valid vector POW2 ranges
 * implied by the input/output element sizes.
 *
 * `TIn` and `TOut` are transform-boundary types, not Tensor memory and kernel
 * compute types. DataAccess performs conversions on both sides. The traits are
 * semantic promises used for planning:
 *
 * - `is_elementwise`: the callable is coordinate-independent and output lane
 *   i depends only on input lane i;
 * - `is_lane_local`: output lane i depends only on input lane i, while logical
 *   coordinates or external per-lane parameters may affect the result;
 * - `permutation_equivariant`: permuting input lanes equivalently permutes
 *   output lanes, which is required for unordered conversion;
 * - `reads_input`: false allows input materialization and source reads to be
 *   eliminated.
 */
template <typename EOut, typename EIn, bool Elementwise = false>
struct VecTransform {
  static_assert(vec::Element<EIn> && vec::Element<EOut>,
                "EIn or EOut is not a supported vector element type");

  using TIn = EIn;
  using TOut = EOut;

  static constexpr bool is_elementwise = Elementwise;
  static constexpr bool is_lane_local = Elementwise;
  static constexpr bool permutation_equivariant = Elementwise;
  static constexpr bool reads_input = true;

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
 * When `Elementwise` is true, `Fn` is called without context and the wrapper
 * accepts and ignores it. Otherwise the context is forwarded to `Fn`.
 * Unsupported legal vector widths are adapted as described in the file-level
 * documentation.
 */
template <typename EOut, typename EIn, typename Fn, bool Elementwise = false>
struct LambdaVecTransform : public VecTransform<EOut, EIn, Elementwise> {
  using Base = VecTransform<EOut, EIn, Elementwise>;

  constexpr explicit LambdaVecTransform(Fn&& fn) : _fn(std::move(fn)) {}
  constexpr explicit LambdaVecTransform(const Fn& fn) : _fn(fn) {}

  template <vec::VectorTag To, typename... Coords>
    requires vec::is_scalable_tag_v<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power_v<To>) &&
             (vec::scale_power_v<To> <= Base::max_output_pow2)
  VECOPS_KERNEL_FUNCTION(vec::Vec<To> operator()(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const) {
    constexpr int pow2 = vec::scale_power_v<To>;
    if constexpr (Elementwise) {
      ((void) coords, ...);
      return call_elementwise<To, pow2>(t, v_in);
    } else {
      return call_coordinate<To, pow2>(t, v_in, coords...);
    }
  }

private:
  Fn _fn;

  template <typename To, int Pow2>
  VECOPS_ALWAYS_INLINE vec::Vec<To> call_elementwise(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
    if constexpr (details::CanCallElementwise<To, Fn, EIn>::value) {
      return _fn(t, v_in);
    } else {
      return try_upward_elementwise<To, Pow2 + 1>(t, v_in);
    }
  }

  template <typename To, int Pow2, typename... Coords>
  VECOPS_ALWAYS_INLINE vec::Vec<To> call_coordinate(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const {
    if constexpr (details::CanCallCoordinate<To, Fn, EIn, Coords...>::value) {
      return _fn(t, v_in, coords...);
    } else {
      return try_upward_coordinate<To, Pow2 + 1>(t, v_in, coords...);
    }
  }

  template <typename To, int TryPow2>
  VECOPS_ALWAYS_INLINE vec::Vec<To> try_upward_elementwise(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
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
  VECOPS_ALWAYS_INLINE vec::Vec<To> try_upward_coordinate(
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
  VECOPS_ALWAYS_INLINE vec::Vec<To> try_downward_elementwise(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in) const {
    constexpr int pow2 = vec::scale_power_v<To>;

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
  VECOPS_ALWAYS_INLINE vec::Vec<To> try_downward_coordinate(
      To t,
      vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const {
    constexpr int pow2 = vec::scale_power_v<To>;

    if constexpr (details::CanCallCoordinate<To, Fn, EIn, Coords...>::value) {
      return _fn(t, v_in, coords...);
    } else if constexpr (pow2 > Base::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti = vec::Rebind<EIn, To>;
      const nint_t half_lanes = vec::size(HalfTo{});
      auto v_lo = try_downward_coordinate<HalfTo>(
          HalfTo{}, vec::lower(Ti{}, v_in),
          details::subspan_transform_argument(coords, 0, half_lanes)...);
      auto v_hi = try_downward_coordinate<HalfTo>(
          HalfTo{}, vec::upper(Ti{}, v_in),
          details::subspan_transform_argument(
              coords, half_lanes, half_lanes)...);
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > Base::min_input_pow2,
                    "LambdaVecTransform: Fn does not support any POW2 in range");
      return {};
    }
  }
};

/**
 * @brief Coordinate-aware transform whose output lanes remain independent.
 *
 * Unlike an elementwise transform, the callable still receives TransformContext
 * and is not permutation-equivariant. The lane-local promise only permits a
 * consumer to repartition naturally ordered lanes while preserving each lane's
 * logical coordinate.
 */
template <typename EOut, typename EIn, typename Fn>
struct LaneLocalLambdaVecTransform
    : public LambdaVecTransform<EOut, EIn, Fn, false> {
  using Base = LambdaVecTransform<EOut, EIn, Fn, false>;
  using Base::Base;

  static constexpr bool is_lane_local = true;
};

/**
 * @brief Pure transform that synthesizes zero without reading Tensor memory.
 *
 * `reads_input=false` lets the planner eliminate source reads and input
 * workspace. `EIn` remains part of the typed transform interface but its
 * vector value is ignored.
 */
template <typename EOut, typename EIn = EOut>
struct ZeroVecTransform : public VecTransform<EOut, EIn, true> {
  using Base = VecTransform<EOut, EIn, true>;
  static constexpr bool reads_input = false;

  template <vec::VectorTag To, vec::VectorValue Vi, typename... Coords>
    requires vec::is_scalable_tag_v<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power_v<To>) &&
             (vec::scale_power_v<To> <= Base::max_output_pow2)
  VECOPS_KERNEL_FUNCTION(vec::Vec<To> operator()(
      To t, Vi, Coords... coords) const) {
    ((void) coords, ...);
    return vec::zeros(t);
  }
};

template <typename T>
struct IsZeroVecTransform : std::false_type {};

template <typename EOut, typename EIn>
struct IsZeroVecTransform<ZeroVecTransform<EOut, EIn>> : std::true_type {};

template <typename T>
inline constexpr bool is_zero_vec_transform_v =
    IsZeroVecTransform<std::remove_cvref_t<T>>::value;

/**
 * @brief Explicit same-dtype identity transform.
 *
 * Use this only when an actual transform object is needed. Prefer
 * `NoTransform` for a normal Tensor operand, because it exposes fused memory
 * conversion opportunities and avoids a transform call.
 */
template <typename EOut, typename EIn = EOut>
struct IdentityVecTransform : public VecTransform<EOut, EIn, true> {
  static_assert(std::same_as<EOut, EIn>,
                "dtype conversion belongs to tensor::DataAccess; use NoTransform");
  using Base = VecTransform<EOut, EIn, true>;

  template <vec::VectorTag To, typename... Coords>
    requires vec::is_scalable_tag_v<To> &&
             std::same_as<vec::ElementOf<To>, EOut> &&
             (Base::min_output_pow2 <= vec::scale_power_v<To>) &&
             (vec::scale_power_v<To> <= Base::max_output_pow2)
  VECOPS_KERNEL_FUNCTION(vec::Vec<To> operator()(
      To t, vec::Vec<vec::Rebind<EIn, To>> v_in,
      Coords... coords) const) {
    ((void) coords, ...);
    return v_in;
  }
};

/**
 * Explicitly override a backend's automatic transform/store batching policy.
 * The wrapper preserves every semantic transform trait and only adds the
 * requested lowering mode; operator code does not need to mention vector
 * widths or backend tags.
 */
template <TransformStoreMode Mode, typename Inner>
  requires is_vec_transform_like_v<Inner>
class TransformStoreOverride {
public:
  using TIn = typename Inner::TIn;
  using TOut = typename Inner::TOut;
  static constexpr bool is_elementwise = Inner::is_elementwise;
  static constexpr bool is_lane_local = Inner::is_lane_local;
  static constexpr bool permutation_equivariant = Inner::permutation_equivariant;
  static constexpr bool reads_input = Inner::reads_input;
  static constexpr TransformStoreMode transform_store_mode = Mode;

  constexpr explicit TransformStoreOverride(Inner inner)
    : inner_(std::move(inner)) {
  }

  template <vec::VectorTag To, typename... Coords>
  VECOPS_KERNEL_FUNCTION(decltype(auto) operator()(To tag, vec::Vec<vec::Rebind<TIn, To>> value, Coords... coordinates)
                           const) {
    return inner_(tag, value, coordinates...);
  }

private:
  Inner inner_;
};

template <TransformStoreMode Mode, typename Inner>
struct IsZeroVecTransform<TransformStoreOverride<Mode, Inner>> : IsZeroVecTransform<Inner> {};

template <TransformStoreMode Mode, typename Transform>
  requires is_vec_transform_like_v<std::remove_cvref_t<Transform>>
constexpr auto with_transform_store_mode(Transform&& transform) {
  using Inner = std::remove_cvref_t<Transform>;
  return TransformStoreOverride<Mode, Inner>{Inner(std::forward<Transform>(transform))};
}

/**
 * @brief Wrap a coordinate-aware callable as a typed vector transform.
 * @tparam EOut Transform output element type.
 * @tparam EIn Transform input element type.
 * @param fn Callable with signature `fn(out_tag, input_vec, context)`.
 */
template <typename EOut, typename EIn, typename Fn>
constexpr auto make_vec_transform(Fn&& fn) {
  using Transform = LambdaVecTransform<EOut, EIn, std::remove_cvref_t<Fn>, false>;
  return Transform(std::forward<Fn>(fn));
}

/**
 * @brief Wrap a coordinate-aware callable whose lanes are independent.
 *
 * The callable signature is `fn(out_tag, input_vec, context)`. It may use each
 * lane's logical coordinate and external per-lane parameters, but output lane i
 * must not depend on any input lane other than i. This does not promise
 * permutation equivariance.
 */
template <typename EOut, typename EIn, typename Fn>
constexpr auto make_lane_local_vec_transform(Fn&& fn) {
  using Transform = LaneLocalLambdaVecTransform<
      EOut, EIn, std::remove_cvref_t<Fn>>;
  return Transform(std::forward<Fn>(fn));
}

/**
 * @brief Wrap a coordinate-independent callable as an elementwise transform.
 *
 * The callable signature is `fn(out_tag, input_vec)`. Marking a transform
 * elementwise also promises permutation equivariance; do not use this factory
 * for lane-position-dependent behavior even if the callable takes no context.
 */
template <typename EOut, typename EIn, typename Fn>
constexpr auto make_elementwise_vec_transform(Fn&& fn) {
  using Transform = LambdaVecTransform<EOut, EIn, std::remove_cvref_t<Fn>, true>;
  return Transform(std::forward<Fn>(fn));
}

/**
 * @brief Preserve an existing transform or wrap a raw callable.
 *
 * Existing transforms must already expose exactly compatible `TIn/TOut`.
 * This function never inserts dtype-conversion wrappers; conversions belong to
 * DataAccess so they can be fused with memory operations.
 */
template <typename EOut, typename EIn, typename Fn>
constexpr auto adapt_vec_transform(Fn&& fn) {
  using FnT = std::remove_cvref_t<Fn>;
  if constexpr (is_vec_transform_like_v<FnT>) {
    static_assert(
        is_any_v<typename FnT::TIn, EIn> &&
            is_any_v<typename FnT::TOut, EOut>,
        "transform dtype adaptation belongs to tensor::DataAccess");
    return FnT(std::forward<Fn>(fn));
  } else {
    return make_vec_transform<EOut, EIn>(std::forward<Fn>(fn));
  }
}

/** @brief Stateless zero-producing transform value. */
template <typename EOut, typename EIn = EOut>
inline constexpr ZeroVecTransform<EOut, EIn> zeros_transform {};

/** @brief Stateless same-dtype identity transform value. */
template <typename EOut, typename EIn = EOut>
inline constexpr IdentityVecTransform<EOut, EIn> identity_transform {};

} // namespace vecops::tensor

#endif // VECOPS_TENSOR_TRANSFORM_H
