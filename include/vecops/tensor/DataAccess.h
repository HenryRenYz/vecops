#ifndef VECOPS_TENSOR_DATA_ACCESS_H
#define VECOPS_TENSOR_DATA_ACCESS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/AccessPolicy.h"
#include "vecops/tensor/Tensor.h"
#include "vecops/tensor/Transform.h"
#include "vecops/util/ScalarConvert.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/Options.h"

/**
 * @file DataAccess.h
 * @brief Policy-driven vector access to logical N-dimensional Tensors.
 *
 * DataAccess is the boundary between an operator's vector computation and the
 * caller's Tensor representation. A kernel requests a vector of ComputeType at
 * a logical coordinate and axis; DataAccess applies Layout addressing, active
 * lanes, dtype conversion, an optional pure transform, materialization, and
 * hot-loop algorithm boundaries.
 *
 * ## Construction model: Spec + Policy + Session
 *
 * A caller-side **Spec** describes what the operand is:
 *
 * @code
 * auto x_spec = tensor::input<float>(x_tensor, input_transform);
 * auto y_spec = tensor::output<float>(y_tensor, output_transform);
 * @endcode
 *
 * A kernel-side **Policy** describes how the algorithm uses it. Binding both
 * creates a scoped DataAccess **session**:
 *
 * @code
 * using XPolicy = tensor::InputAccessPolicy<1, 2>;
 * using YPolicy = tensor::OutputAccessPolicy<1>;
 *
 * kernel::with_operands(
 *     workspace,
 *     tensor::operand(x_spec, XPolicy{}),
 *     tensor::operand(y_spec, YPolicy{}),
 *     [&](auto& x, auto& y) {
 *       auto v = x.load(tag, tensor::coord(row, col));
 *       y.store(tag, tensor::coord(row, col), f(v));
 *       y.commit(); // unconditional: no-op for direct output
 *     });
 * @endcode
 *
 * `with_operands` resolves dynamic Layout choices before entering the callback,
 * materializes inputs if required, supplies statically typed direct or
 * materialized sessions, verifies output commit ownership in debug builds, and
 * rewinds its workspace mark after sessions are destroyed.
 *
 * ## Coordinate vector semantics
 *
 * For `load(tag, origin, axis<Dim>, options...)`, caller lane `j` refers to:
 *
 * @code
 * logical_coord = origin + unit_axis<Dim> * lane_selector(j)
 * physical_offset = offset_at(layout, logical_coord)
 * @endcode
 *
 * The selector is `j` by default, `j * s` for `vec::strided(s)`, and
 * `indices[j]` for `vec::indexed(indices)`. These are Tensor-element offsets;
 * Layout multiplies them by the physical axis stride. A logically contiguous
 * request therefore becomes a normal load only when the resulting physical
 * stride is one; otherwise it becomes a gather/scatter or scalar fallback.
 * The vector axis defaults to the final Tensor dimension.
 *
 * ## Active lanes and population
 *
 * DataAccess reuses vec active options: `vec::opt::unmasked`,
 * `vec::opt::first(n)`, and `vec::opt::masked(mask)`. Loads additionally accept
 * `vec::opt::zero` or merge population. At most one active, addressing, and
 * population option may be supplied. The default is unmasked.
 *
 * Every active logical lane must be inside the Tensor shape. Masked-off lanes
 * may point outside the shape and are not accessed. Bounds are asserted in
 * debug builds; an out-of-range unmasked request is a caller error, not an
 * implicit tail operation.
 *
 * ## Conversion and transform pipelines
 *
 * Input:
 *
 * @code
 * MemoryType -> load_convert -> Transform::TIn -> transform
 *            -> Transform::TOut -> convert -> requested ComputeTag
 * @endcode
 *
 * Output is the reverse boundary direction. `NoTransform` removes both
 * transform boundaries and directly selects `load_convert/store_convert`.
 * Active/addressing/memory options are forwarded to the actual memory
 * operation whenever representable.
 *
 * The requested Tag is a logical result contract, not a transform-width
 * contract. If rebinding it to `TIn` or `TOut` exceeds backend POW2 limits,
 * DataAccess recursively partitions the lanes, performs several legal memory
 * and transform calls, and reassembles the requested vector. Context subspans
 * preserve original lane coordinates. Large `vec::load_convert/store_convert`
 * requests use the same recursive principle at the raw-memory layer.
 *
 * ## Cursor choice
 *
 * - `scan()` is for a one-dimensional sequence whose traversal and vector
 *   axes are identical. It owns tail accounting and supports a smaller manual
 *   TailTag.
 * - `project()` advances one axis while each operation vectors along a
 *   different axis. Each load/store supplies its own active/address options.
 *
 * Both cursors store logical coordinates, not independently incremented raw
 * pointers. Arbitrary indexed mappings are intentionally excluded from scan.
 *
 * ## Lifetime and pitfalls
 *
 * - Specs and DataAccess are non-owning with respect to Tensor memory.
 * - Output Specs require a mutable Tensor. No `const_cast` is performed.
 * - Materialized output sessions are move-only and must be explicitly
 *   `commit()`ed. Destructors do not write back; they assert on an omitted
 *   commit in debug builds. Borrowed slices never expose commit authority.
 * - Transforms must be deterministic and side-effect-free because planning may
 *   move their execution or split one request into several calls.
 * - Tensor-level `indexed` rejects explicit byte scale. Indices are logical
 *   offsets and are scaled by Layout exactly once.
 * - Unordered conversion is a kernel-wide lane-order contract, not a local
 *   speed flag. Incompatible operands are rejected at binding time.
 * - `required_workspace()` must use the same Spec and Policy as binding.
 */
namespace vecops::tensor {

namespace details {

template <typename T>
struct IsOperandFact : std::false_type {};

template <nint_t Alignment>
struct IsOperandFact<BaseAlignment<Alignment>> : std::true_type {};

template <typename T>
inline constexpr bool is_operand_fact_v =
    IsOperandFact<std::remove_cvref_t<T>>::value;

template <typename Tensor, nint_t Alignment>
VECOPS_ALWAYS_INLINE void validate_operand_fact(
    const Tensor& tensor, BaseAlignment<Alignment>) {
  VECOPS_ASSERT(
      reinterpret_cast<std::uintptr_t>(tensor.data()) % Alignment == 0,
      "tensor base pointer does not satisfy its declared alignment");
}

template <typename T, bool = T::is_const>
struct IsConstOneMeta : std::false_type {};

template <typename T>
struct IsConstOneMeta<T, true> : std::bool_constant<T::value == 1> {};

template <typename T>
inline constexpr bool is_const_one_meta_v = IsConstOneMeta<T>::value;

/** True when the meta stride type is provably the value 1 at compile time. */
template <typename T>
struct IsDefinitelyOneMeta {
  static constexpr bool value = [] {
    if constexpr (IsConstOneMeta<T>::value) {
      return true;
    } else if constexpr (T::is_runtime) {
      return T::has_lower && T::lo == 1 && T::has_upper && T::hi == 1;
    } else {
      return false;
    }
  }();
};

template <typename T>
inline constexpr bool is_definitely_one_meta_v =
    IsDefinitelyOneMeta<std::remove_cvref_t<T>>::value;

template <int I, typename T>
struct MetaElement;

template <int I, typename... Ts>
struct MetaElement<I, Strides<Ts...>> {
  using type = std::tuple_element_t<I, std::tuple<Ts...>>;
};

template <bool Load, typename... Options>
VECOPS_ALWAYS_INLINE constexpr void validate_access_options() {
  constexpr std::size_t active =
      vec::details::option_count<vec::details::IsUnmaskedOption, Options...> +
      vec::details::option_count<vec::details::IsFirstOption, Options...> +
      vec::details::option_count<vec::details::IsMaskedOption, Options...>;
  constexpr std::size_t addressing =
      vec::details::option_count<vec::details::IsStridedOption, Options...> +
      vec::details::option_count<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t population =
      vec::details::option_count<vec::details::IsZeroOption, Options...> +
      vec::details::option_count<vec::details::IsVectorMergeOption, Options...> +
      vec::details::option_count<vec::details::IsScalarMergeOption, Options...>;
  static_assert(active <= 1, "at most one tensor active option is allowed");
  static_assert(addressing <= 1,
                "at most one tensor lane-addressing option is allowed");
  static_assert(population <= 1,
                "at most one inactive-population option is allowed");
  static_assert(Load || population == 0,
                "inactive-population options are only valid for tensor loads");
}

template <typename T>
inline constexpr bool is_no_transform =
    std::same_as<std::remove_cvref_t<T>, NoTransform>;

template <typename Transform, typename Memory>
using transform_input_t = std::conditional_t<
    is_no_transform<Transform>, Memory, typename Transform::TIn>;

template <typename Transform, typename Compute>
using transform_output_t = std::conditional_t<
    is_no_transform<Transform>, Compute, typename Transform::TOut>;

template <typename Transform>
inline constexpr bool transform_permutation_equivariant = [] {
  if constexpr (is_no_transform<Transform>) return true;
  else return Transform::permutation_equivariant;
}();

template <typename Transform>
inline constexpr bool transform_reads_input = [] {
  if constexpr (is_no_transform<Transform>) return true;
  else return Transform::reads_input;
}();

template <typename Transform, typename Memory, typename Compute>
VECOPS_ALWAYS_INLINE auto normalize_transform(Transform&& transform) {
  using Clean = std::remove_cvref_t<Transform>;
  if constexpr (is_vec_transform_like_v<Clean>) {
    return Clean(std::forward<Transform>(transform));
  } else {
    return make_vec_transform<Compute, Memory>(
        std::forward<Transform>(transform));
  }
}

template <typename F, typename Tuple, std::size_t... I>
VECOPS_ALWAYS_INLINE decltype(auto) apply_inline_impl(
    F&& fn, Tuple&& tuple, std::index_sequence<I...>) {
  return std::forward<F>(fn)(
      std::get<I>(std::forward<Tuple>(tuple))...);
}

template <typename F, typename Tuple>
VECOPS_ALWAYS_INLINE decltype(auto) apply_inline(F&& fn, Tuple&& tuple) {
  return apply_inline_impl(
      std::forward<F>(fn), std::forward<Tuple>(tuple),
      std::make_index_sequence<
          std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template <typename Option>
VECOPS_ALWAYS_INLINE auto retain_non_address_option(Option&& option) {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (
      vec::details::IsIndexedOption<Clean>::value ||
      vec::details::IsStridedOption<Clean>::value) {
    return std::tuple<>{};
  } else {
    return std::forward_as_tuple(std::forward<Option>(option));
  }
}

template <typename... Options>
VECOPS_ALWAYS_INLINE auto non_address_options(Options&&... options) {
  return std::tuple_cat(
      retain_non_address_option(std::forward<Options>(options))...);
}

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE auto make_logical_mapping(
    Tag, Options&&... options) {
  constexpr std::size_t indexed_count =
      vec::details::option_count<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count<vec::details::IsStridedOption, Options...>;
  static_assert(indexed_count <= 1 && strided_count <= 1);
  static_assert(indexed_count + strided_count <= 1);

  if constexpr (indexed_count == 1) {
    auto&& indexed = vec::details::find_option<vec::details::IsIndexedOption>(
        std::forward<Options>(options)...);
    using Indexed = std::remove_cvref_t<decltype(indexed)>;
    static_assert(
        vec::details::IsIndexedOption<Indexed>::scale == 0,
        "tensor indexed addressing only accepts element-scale indices");
    using IndexTag = vec::VecToTagT<
        typename vec::details::IsIndexedOption<Indexed>::Value>;
    return IndexedLaneMapping<IndexTag>{IndexTag{}, indexed.indices};
  } else {
    nint_t stride = 1;
    if constexpr (strided_count == 1) {
      auto&& strided = vec::details::find_option<vec::details::IsStridedOption>(
          std::forward<Options>(options)...);
      stride = static_cast<nint_t>(strided.stride);
    }
    return AffineLaneMapping{stride};
  }
}

template <vec::VectorTag Tag, typename Mapping, typename F, typename... Options>
VECOPS_ALWAYS_INLINE decltype(auto) with_active_context(
    Tag tag, Mapping mapping, F&& fn, Options&&... options) {
  constexpr std::size_t first_count =
      vec::details::option_count<vec::details::IsFirstOption, Options...>;
  constexpr std::size_t masked_count =
      vec::details::option_count<vec::details::IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      vec::details::option_count<vec::details::IsUnmaskedOption, Options...>;
  static_assert(first_count + masked_count + unmasked_count <= 1);

  if constexpr (first_count == 1) {
    auto&& active = vec::details::find_option<vec::details::IsFirstOption>(
        std::forward<Options>(options)...);
    return std::forward<F>(fn)(mapping, FirstActiveLanes{active.count});
  } else if constexpr (masked_count == 1) {
    auto&& active = vec::details::find_option<vec::details::IsMaskedOption>(
        std::forward<Options>(options)...);
    using MaskType = typename vec::details::IsMaskedOption<
        std::remove_cvref_t<decltype(active)>>::Value;
    static_assert(std::same_as<MaskType, vec::Mask<Tag>>,
                  "tensor masks use the requested logical Tag");
    return std::forward<F>(fn)(
        mapping, MaskedActiveLanes<Tag>{tag, active.value});
  } else {
    return std::forward<F>(fn)(mapping, AllActiveLanes{});
  }
}

template <typename Projection>
struct ProjectionTraits;

template <std::size_t OriginalRank, std::size_t LocalRank>
struct ProjectionTraits<CoordinateProjection<OriginalRank, LocalRank>> {
  static constexpr std::size_t original_rank = OriginalRank;
  static constexpr std::size_t local_rank = LocalRank;
};

template <vec::VectorTag Tag,
          std::size_t OriginalRank,
          std::size_t LocalRank,
          typename F,
          typename... Options>
VECOPS_ALWAYS_INLINE decltype(auto) with_transform_context(
    Tag tag,
    const Coord<LocalRank>& local,
    int vector_axis_value,
    const CoordinateProjection<OriginalRank, LocalRank>& projection,
    F&& fn,
    Options&&... options) {
  auto mapping = make_logical_mapping(tag, options...);
  return with_active_context(
      tag, mapping,
      [&](auto concrete_mapping, auto active) VECOPS_INLINE_LAMBDA
          -> decltype(auto) {
        const int original_axis =
            projection.local_to_original[static_cast<std::size_t>(
                vector_axis_value)];
        TransformContext<
            OriginalRank,
            decltype(concrete_mapping),
            decltype(active)> context{
                projection.project(local),
                original_axis,
                concrete_mapping,
                active,
                0};
        return std::forward<F>(fn)(context);
      },
      options...);
}

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE vec::Vec<Tag> populate_inactive(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  constexpr std::size_t first_count =
      vec::details::option_count<vec::details::IsFirstOption, Options...>;
  constexpr std::size_t masked_count =
      vec::details::option_count<vec::details::IsMaskedOption, Options...>;
  if constexpr (first_count + masked_count == 0) {
    return value;
  } else {
    auto mask = [&] {
      if constexpr (first_count == 1) {
        auto&& first = vec::details::find_option<vec::details::IsFirstOption>(
            std::forward<Options>(options)...);
        return vec::mwhilelt(tag, 0, first.count);
      } else {
        auto&& masked = vec::details::find_option<vec::details::IsMaskedOption>(
            std::forward<Options>(options)...);
        return masked.value;
      }
    }();

    vec::Vec<Tag> inactive = vec::zeros(tag);
    constexpr std::size_t vector_merge_count =
        vec::details::option_count<vec::details::IsVectorMergeOption,
                                   Options...>;
    constexpr std::size_t scalar_merge_count =
        vec::details::option_count<vec::details::IsScalarMergeOption,
                                   Options...>;
    if constexpr (vector_merge_count == 1) {
      auto&& merge = vec::details::find_option<
          vec::details::IsVectorMergeOption>(options...);
      inactive = merge.value;
    } else if constexpr (scalar_merge_count == 1) {
      auto&& merge = vec::details::find_option<
          vec::details::IsScalarMergeOption>(options...);
      inactive = vec::fill(tag, merge.value);
    }
    return vec::blend(tag, inactive, mask, value);
  }
}

template <typename Policy, typename To, typename From>
VECOPS_ALWAYS_INLINE To scalar_policy_convert(From value) {
  if constexpr (std::same_as<typename Policy::ConversionValueOption,
                             vec::cvt::Wrap>) {
    if constexpr (requires { vecops::wrap_convert<To>(value); }) {
      return vecops::wrap_convert<To>(value);
    } else {
      return vecops::convert<To>(value);
    }
  } else {
    return vecops::convert<To>(value);
  }
}

template <typename Transform, vec::VectorTag RequestedTag>
consteval bool transform_tags_representable() {
  using InTag = vec::Rebind<typename Transform::TIn, RequestedTag>;
  using OutTag = vec::Rebind<typename Transform::TOut, RequestedTag>;
  if constexpr (
      vec::is_scalable_tag<RequestedTag> &&
      (vec::scale_power<InTag> < VEC_HW_MIN_POW ||
       vec::scale_power<InTag> > VEC_MAX_POW ||
       vec::scale_power<OutTag> < VEC_HW_MIN_POW ||
       vec::scale_power<OutTag> > VEC_MAX_POW)) {
    return false;
  } else {
    return true;
  }
}

template <typename Transform, vec::VectorTag RequestedTag, typename Context>
consteval bool can_transform_chunk() {
  if constexpr (!transform_tags_representable<Transform, RequestedTag>()) {
    return false;
  } else {
    using InTag = vec::Rebind<typename Transform::TIn, RequestedTag>;
    using OutTag = vec::Rebind<typename Transform::TOut, RequestedTag>;
    return requires(const Transform& transform, Context context,
                    vec::Vec<InTag> input) {
      { transform(OutTag{}, input, context) } ->
          std::same_as<vec::Vec<OutTag>>;
    };
  }
}

/**
 * A chunk is usable only when the converting memory access at its width is
 * representable as well: the memory-side rebind of the chunk must stay within
 * backend tag limits, otherwise the chunk keeps splitting.
 */
template <
    typename Transform, vec::VectorTag RequestedTag, typename Context,
    typename Memory>
consteval bool can_transform_chunk_with_memory() {
  if constexpr (!can_transform_chunk<Transform, RequestedTag, Context>()) {
    return false;
  } else {
    using MemoryTag = vec::Rebind<Memory, RequestedTag>;
    if constexpr (
        vec::is_scalable_tag<RequestedTag> &&
        (vec::scale_power<MemoryTag> < VEC_HW_MIN_POW ||
         vec::scale_power<MemoryTag> > VEC_MAX_POW)) {
      return false;
    } else {
      return true;
    }
  }
}


template <typename Tag>
consteval bool can_split_tag();

// ============================================================================
// Scalable-SVE transform chunk lowering (per-lane reference path).
//
// The vectorized chunk pipeline relies on tag-representation ranges that the
// scalable-SVE backend expresses differently (sizeless word groups), so the
// reference per-lane implementation below serves that target. It shares the
// public entry points; x86, scalar, and fixed-SVE builds use the vectorized
// recursion defined later in this file.
// ============================================================================

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE bool lane_is_active(
    Tag tag, nint_t lane, Options&&... options) {
  if constexpr (vec::details::option_count<
                    vec::details::IsFirstOption, Options...> == 1) {
    auto&& first = vec::details::find_option<
        vec::details::IsFirstOption>(options...);
    return 0 <= lane && lane < first.count;
  } else if constexpr (vec::details::option_count<
                           vec::details::IsMaskedOption, Options...> == 1) {
    auto&& masked = vec::details::find_option<
        vec::details::IsMaskedOption>(options...);
    return vec::get(tag, masked.value, lane);
  } else {
    return true;
  }
}

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE nint_t logical_lane_offset(
    Tag, nint_t lane, Options&&... options) {
  if constexpr (vec::details::option_count<
                    vec::details::IsIndexedOption, Options...> == 1) {
    auto&& indexed = vec::details::find_option<
        vec::details::IsIndexedOption>(options...);
    using Indexed = std::remove_cvref_t<decltype(indexed)>;
    static_assert(vec::details::IsIndexedOption<Indexed>::scale == 0,
                  "tensor indexed addressing rejects byte scales");
    using IndexTag = vec::VecToTagT<
        typename vec::details::IsIndexedOption<Indexed>::Value>;
    return static_cast<nint_t>(
        vec::get(IndexTag{}, indexed.indices, lane));
  } else if constexpr (vec::details::option_count<
                           vec::details::IsStridedOption, Options...> == 1) {
    auto&& strided = vec::details::find_option<
        vec::details::IsStridedOption>(options...);
    return lane * static_cast<nint_t>(strided.stride);
  } else {
    return lane;
  }
}

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE vec::ElementOf<Tag> inactive_scalar(
    Tag tag, nint_t lane, Options&&... options) {
  if constexpr (vec::details::option_count<
                    vec::details::IsVectorMergeOption, Options...> == 1) {
    auto&& merge = vec::details::find_option<
        vec::details::IsVectorMergeOption>(options...);
    return vec::get(tag, merge.value, lane);
  } else if constexpr (vec::details::option_count<
                           vec::details::IsScalarMergeOption,
                           Options...> == 1) {
    auto&& merge = vec::details::find_option<
        vec::details::IsScalarMergeOption>(options...);
    return vecops::convert<vec::ElementOf<Tag>>(merge.value);
  } else {
    return vec::ElementOf<Tag>{};
  }
}


template <
    typename Policy,
    typename Transform,
    vec::VectorTag RootTag,
    vec::VectorTag ChunkTag,
    typename Memory,
    typename Context,
    typename... Options>
VECOPS_ALWAYS_INLINE void load_transform_chunks_scalar(
    RootTag root_tag,
    ChunkTag,
    vec::Vec<RootTag>& result,
    const Memory* pointer,
    nint_t tensor_axis_stride,
    const Transform& transform,
    const Context& context,
    nint_t lane_begin,
    Options&&... options) {
  if constexpr (can_transform_chunk<Transform, ChunkTag, Context>()) {
    using TransformInTag = vec::Rebind<typename Transform::TIn, ChunkTag>;
    using TransformOutTag = vec::Rebind<typename Transform::TOut, ChunkTag>;
    auto transform_input = vec::zeros(TransformInTag{});
    const nint_t chunk_lanes = vec::size(TransformInTag{});

    if constexpr (transform_reads_input<Transform>) {
      for (nint_t lane = 0; lane < chunk_lanes; ++lane) {
        const nint_t root_lane = lane_begin + lane;
        if (!lane_is_active(root_tag, root_lane, options...)) continue;
        const nint_t logical = logical_lane_offset(
            root_tag, root_lane, options...);
        transform_input = vec::set(
            TransformInTag{}, transform_input, lane,
            scalar_policy_convert<Policy, typename Transform::TIn>(
                pointer[logical * tensor_axis_stride]));
      }
    }

    const auto chunk_context = context.subspan(lane_begin, chunk_lanes);
    const auto transformed = transform(
        TransformOutTag{}, transform_input, chunk_context);
    for (nint_t lane = 0; lane < chunk_lanes; ++lane) {
      const nint_t root_lane = lane_begin + lane;
      if (!lane_is_active(root_tag, root_lane, options...)) continue;
      result = vec::set(
          root_tag, result, root_lane,
          scalar_policy_convert<Policy, vec::ElementOf<RootTag>>(
              vec::get(TransformOutTag{}, transformed, lane)));
    }
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    load_transform_chunks_scalar<Policy>(
        root_tag, HalfTag{}, result, pointer, tensor_axis_stride,
        transform, context, lane_begin, options...);
    const nint_t half_lanes = vec::size(HalfTag{});
    load_transform_chunks_scalar<Policy>(
        root_tag, HalfTag{}, result, pointer, tensor_axis_stride,
        transform, context, lane_begin + half_lanes, options...);
  }
}

template <
    typename Policy,
    typename Transform,
    vec::VectorTag RootTag,
    vec::VectorTag ChunkTag,
    typename Memory,
    typename Context,
    typename... Options>
VECOPS_ALWAYS_INLINE void store_transform_chunks_scalar(
    RootTag root_tag,
    ChunkTag,
    const vec::Vec<RootTag>& value,
    Memory* pointer,
    nint_t tensor_axis_stride,
    const Transform& transform,
    const Context& context,
    nint_t lane_begin,
    Options&&... options) {
  if constexpr (can_transform_chunk<Transform, ChunkTag, Context>()) {
    using TransformInTag = vec::Rebind<typename Transform::TIn, ChunkTag>;
    using TransformOutTag = vec::Rebind<typename Transform::TOut, ChunkTag>;
    auto transform_input = vec::zeros(TransformInTag{});
    const nint_t chunk_lanes = vec::size(TransformInTag{});
    for (nint_t lane = 0; lane < chunk_lanes; ++lane) {
      const nint_t root_lane = lane_begin + lane;
      if (!lane_is_active(root_tag, root_lane, options...)) continue;
      transform_input = vec::set(
          TransformInTag{}, transform_input, lane,
          scalar_policy_convert<Policy, typename Transform::TIn>(
              vec::get(root_tag, value, root_lane)));
    }

    const auto chunk_context = context.subspan(lane_begin, chunk_lanes);
    const auto transformed = transform(
        TransformOutTag{}, transform_input, chunk_context);
    for (nint_t lane = 0; lane < chunk_lanes; ++lane) {
      const nint_t root_lane = lane_begin + lane;
      if (!lane_is_active(root_tag, root_lane, options...)) continue;
      const nint_t logical = logical_lane_offset(
          root_tag, root_lane, options...);
      pointer[logical * tensor_axis_stride] =
          scalar_policy_convert<Policy, Memory>(
              vec::get(TransformOutTag{}, transformed, lane));
    }
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    store_transform_chunks_scalar<Policy>(
        root_tag, HalfTag{}, value, pointer, tensor_axis_stride,
        transform, context, lane_begin, options...);
    const nint_t half_lanes = vec::size(HalfTag{});
    store_transform_chunks_scalar<Policy>(
        root_tag, HalfTag{}, value, pointer, tensor_axis_stride,
        transform, context, lane_begin + half_lanes, options...);
  }
}


#endif  // scalable SVE reference path

template <typename Tag>
consteval bool can_split_tag() {
  if constexpr (vec::is_fixed_tag<Tag>) {
    return vec::fixed_lanes<Tag> > 1;
  } else {
    // The largest supported element-size ratio is eight.  Once a requested
    // tag is more than three powers below the backend minimum, rebinding it
    // cannot produce a legal transform tag for any supported element type.
    return vec::scale_power<Tag> > VEC_HW_MIN_POW - 3;
  }
}

/**
 * Executes as many transform chunks as are necessary to cover RootTag.
 *
 * ChunkTag is a logical lane partition only.  In particular, it is allowed
 * to be a subword tag that cannot itself participate in conversion.  Only
 * the rebound transform input/output tags are materialized as vectors.  This
 * distinction is what makes requests such as int8/P2 -> double/P2 possible
 * on SVE: the int8 request is recursively partitioned into eight logical
 * chunks, while every actual transform call uses a legal double/P2 vector.
 */
/**
 * @brief Caller-side description of a readable Tensor operand.
 *
 * The Spec stores a Tensor view, compute-boundary type, optional transform,
 * coordinate projection, and externally verifiable facts. It deliberately
 * stores no execution plan, conversion order, or memory temporality
 * choice; those belong to `InputAccessPolicy`.
 *
 * Prefer the `input<Compute>()` factory instead of naming this type directly.
 */
/**
 * Executes the transform over RootTag lanes, using whole-vector converting
 * loads at every chunk. Active lanes ride as a chunk-domain mask, addressing
 * is one of three compile-time forms, and halves recombine with vec::concat,
 * so no per-lane scalar access appears anywhere on this path.
 *
 * AddrKind: 0 = contiguous physical (unit stride), 1 = strided with
 * `physical_stride` elements between lanes, 2 = indexed by the chunk-domain
 * logical index vector scaled by `physical_stride`.
 */
template <
    typename Policy, typename Transform,
    vec::VectorTag ChunkTag, typename Memory, typename Context,
    bool HasActive, int AddrKind, typename IndexVec, typename MaskVec>
VECOPS_ALWAYS_INLINE vec::Vec<vec::Rebind<typename Transform::TOut, ChunkTag>>
load_transform_vector(
    ChunkTag chunk_tag,
    const Memory* base_pointer,
    nint_t lane_begin,
    nint_t physical_stride,
    IndexVec logical_indices,
    MaskVec active_mask,
    const Transform& transform,
    const Context& context) {
  using TIn = typename Transform::TIn;
  using TOut = typename Transform::TOut;
  if constexpr (can_transform_chunk_with_memory<
                    Transform, ChunkTag, Context, Memory>()) {
    using TransformInTag = vec::Rebind<TIn, ChunkTag>;
    using TransformOutTag = vec::Rebind<TOut, ChunkTag>;
    using Order = typename Policy::ConversionOrderOption;
    using Value = typename Policy::ConversionValueOption;
    using Temporal = typename Policy::MemoryOptions::TemporalityOption;
    const nint_t chunk_lanes = vec::size(TransformInTag{});

    vec::Vec<TransformInTag> transform_input;
    if constexpr (transform_reads_input<Transform>) {
      constexpr vec::Active ActiveKind =
          HasActive ? vec::Active::Masked : vec::Active::Unmasked;
      constexpr vec::Addressing AddressingKind =
          AddrKind == 0   ? vec::Addressing::Contiguous
          : AddrKind == 1 ? vec::Addressing::Strided
                          : vec::Addressing::Indexed;
      vec::LoadConvertRequest<
          TransformInTag, Memory, ActiveKind, AddressingKind,
          vec::Populate::Zero, vec::mem::Unaligned, Temporal,
          0, std::remove_cvref_t<IndexVec>, Order, Value>
          request;
      if constexpr (HasActive) {
        const auto leaf_mask = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (std::same_as<ChunkTag, TransformInTag>) {
            return active_mask;
          } else {
            return vec::convert(TransformInTag{}, ChunkTag{}, active_mask);
          }
        }();
        request.mask = &leaf_mask;
      }
      const Memory* chunk_base = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (AddrKind == 0) {
          return base_pointer + lane_begin;
        } else if constexpr (AddrKind == 1) {
          return base_pointer + lane_begin * physical_stride;
        } else {
          return base_pointer;
        }
      }();
      if constexpr (AddrKind == 1) {
        request.stride = physical_stride;
      }
      if constexpr (AddrKind == 2) {
        using IndexElement = vec::ElementOf<IndexVec>;
        const auto stride_scale = vec::fill(
            vec::VecToTagT<IndexVec>{},
            static_cast<IndexElement>(physical_stride));
        const auto physical = vec::mul(logical_indices, stride_scale);
        request.indices = &physical;
      }
      transform_input =
          vec::load_convert(TransformInTag{}, chunk_base, request);
    } else {
      transform_input = vec::zeros(TransformInTag{});
    }

    const auto chunk_context = context.subspan(lane_begin, chunk_lanes);
    return transform(TransformOutTag{}, transform_input, chunk_context);
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    using HalfIndexVec = std::conditional_t<
        AddrKind == 2, vec::Vec<vec::Half<vec::VecToTagT<IndexVec>>>,
        IndexVec>;
    using HalfMaskVec = std::conditional_t<
        HasActive, vec::Mask<HalfTag>, MaskVec>;
    const nint_t half_lanes = vec::size(HalfTag{});
    const auto lower_mask = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (HasActive) return vec::lower(chunk_tag, active_mask);
      else return MaskVec{};
    }();
    const auto upper_mask = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (HasActive) return vec::upper(chunk_tag, active_mask);
      else return MaskVec{};
    }();
    const auto lower_indices = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) return vec::lower(chunk_tag, logical_indices);
      else return IndexVec{};
    }();
    const auto upper_indices = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) return vec::upper(chunk_tag, logical_indices);
      else return IndexVec{};
    }();
    const auto lower_part = load_transform_vector<
        Policy, Transform, HalfTag, Memory, Context, HasActive, AddrKind>(
        HalfTag{}, base_pointer, lane_begin, physical_stride, lower_indices,
        lower_mask, transform, context);
    const auto upper_part = load_transform_vector<
        Policy, Transform, HalfTag, Memory, Context, HasActive, AddrKind>(
        HalfTag{}, base_pointer, lane_begin + half_lanes, physical_stride,
        upper_indices, upper_mask, transform, context);
    using TransformOutTag = vec::Rebind<typename Transform::TOut, ChunkTag>;
    return vec::concat(
        TransformOutTag{}, lower_part, upper_part);
  }
}

/**
 * Entry of the vectorized transform load: materializes the kernel's active
 * predicate into a root-domain mask and lowers its addressing to one of the
 * three compile-time physical forms before recursion. Returns the converted
 * result in the root domain.
 */
template <
    typename Policy, typename Transform, vec::VectorTag RootTag,
    typename Memory, typename Context, typename StrideMeta,
    typename... Options>
VECOPS_ALWAYS_INLINE vec::Vec<RootTag> load_transform_chunks(
    RootTag root_tag,
    const Memory* pointer,
    StrideMeta tensor_axis_stride,
    const Transform& transform,
    const Context& context,
    Options&&... options) {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  vec::Vec<RootTag> result = vec::zeros(root_tag);
  load_transform_chunks_scalar<Policy>(
      root_tag, root_tag, result, pointer,
      static_cast<nint_t>(tensor_axis_stride), transform, context, 0,
      std::forward<Options>(options)...);
  return result;
#else
  const auto request = vec::details::resolve_load_request<RootTag>(
      std::forward<Options>(options)...);
  using KernelRequest = decltype(request);
  constexpr bool HasActive =
      KernelRequest::active_kind != vec::Active::Unmasked;
  const vec::Mask<RootTag> active_mask = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (KernelRequest::active_kind == vec::Active::First) {
      return vec::mwhilelt(root_tag, 0, request.first_count);
    } else if constexpr (HasActive) {
      return *request.mask;
    } else {
      return vec::Mask<RootTag>{};
    }
  }();
  constexpr bool TensorUnitStride = is_definitely_one_meta_v<StrideMeta>;

  if constexpr (
      KernelRequest::addressing_kind == vec::Addressing::Indexed) {
    static_assert(
        KernelRequest::index_scale == 0,
        "tensor indexed addressing rejects byte scales");
    const auto out = load_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 2>(
        root_tag, pointer, 0, static_cast<nint_t>(tensor_axis_stride),
        *request.indices, active_mask, transform, context);
    using TransformOutTag =
        vec::Rebind<typename Transform::TOut, RootTag>;
    return vec::convert(
        root_tag, TransformOutTag{}, out,
        typename Policy::ConversionOrderOption{},
        typename Policy::ConversionValueOption{});
  } else if constexpr (
      KernelRequest::addressing_kind == vec::Addressing::Contiguous &&
      TensorUnitStride) {
    const auto out = load_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 0>(
        root_tag, pointer, 0, 1, vec::Vec<vec::IndexTag<RootTag>>{},
        active_mask, transform,
        context);
    using TransformOutTag =
        vec::Rebind<typename Transform::TOut, RootTag>;
    return vec::convert(
        root_tag, TransformOutTag{}, out,
        typename Policy::ConversionOrderOption{},
        typename Policy::ConversionValueOption{});
  } else {
    nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Strided) {
      physical_stride *= request.stride;
    }
    const auto out = load_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 1>(
        root_tag, pointer, 0, physical_stride,
        vec::Vec<vec::IndexTag<RootTag>>{}, active_mask,
        transform, context);
    using TransformOutTag =
        vec::Rebind<typename Transform::TOut, RootTag>;
    return vec::convert(
        root_tag, TransformOutTag{}, out,
        typename Policy::ConversionOrderOption{},
        typename Policy::ConversionValueOption{});
  }
#endif
}

/**
 * Stores through the transform with whole-vector converting stores. The
 * root-domain value is split alongside the chunk tag; each leaf converts its
 * half into the transform input domain, runs the transform, and issues one
 * masked converting store.
 */
template <
    typename Policy, typename Transform,
    vec::VectorTag ChunkTag, typename Memory, typename Context,
    bool HasActive, int AddrKind, typename IndexVec, typename MaskVec,
    typename ChunkValueVec>
VECOPS_ALWAYS_INLINE void store_transform_vector(
    ChunkTag chunk_tag,
    Memory* base_pointer,
    nint_t lane_begin,
    nint_t physical_stride,
    IndexVec logical_indices,
    MaskVec active_mask,
    ChunkValueVec chunk_value,
    const Transform& transform,
    const Context& context) {
  using TIn = typename Transform::TIn;
  using TOut = typename Transform::TOut;
  if constexpr (can_transform_chunk_with_memory<
                    Transform, ChunkTag, Context, Memory>()) {
    using TransformInTag = vec::Rebind<TIn, ChunkTag>;
    using TransformOutTag = vec::Rebind<TOut, ChunkTag>;
    using Order = typename Policy::ConversionOrderOption;
    using Value = typename Policy::ConversionValueOption;
    using Temporal = typename Policy::MemoryOptions::TemporalityOption;
    using Packing = typename Policy::MemoryOptions::PackingOption;
    const nint_t chunk_lanes = vec::size(TransformInTag{});

    const auto transform_input = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<ChunkTag, TransformInTag>) {
        return chunk_value;
      } else {
        return vec::convert(
            TransformInTag{}, ChunkTag{}, chunk_value, Order{}, Value{});
      }
    }();
    const auto chunk_context = context.subspan(lane_begin, chunk_lanes);
    const auto transformed = transform(
        TransformOutTag{}, transform_input, chunk_context);

    constexpr vec::Active ActiveKind =
        HasActive ? vec::Active::Masked : vec::Active::Unmasked;
    constexpr vec::Addressing AddressingKind =
        AddrKind == 0   ? vec::Addressing::Contiguous
        : AddrKind == 1 ? vec::Addressing::Strided
                        : vec::Addressing::Indexed;
    vec::StoreConvertRequest<
        TransformOutTag, Memory, ActiveKind, AddressingKind,
        vec::mem::Unaligned, Temporal, 0,
        std::remove_cvref_t<IndexVec>, Order, Value, Packing>
        request;
    if constexpr (HasActive) {
      const auto leaf_mask = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (std::same_as<ChunkTag, TransformOutTag>) {
          return active_mask;
        } else {
          return vec::convert(TransformOutTag{}, ChunkTag{}, active_mask);
        }
      }();
      request.mask = &leaf_mask;
    }
    Memory* chunk_base = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 0) {
        return base_pointer + lane_begin;
      } else if constexpr (AddrKind == 1) {
        return base_pointer + lane_begin * physical_stride;
      } else {
        return base_pointer;
      }
    }();
    if constexpr (AddrKind == 1) {
      request.stride = physical_stride;
    }
    if constexpr (AddrKind == 2) {
      using IndexElement = vec::ElementOf<IndexVec>;
      const auto stride_scale = vec::fill(
          vec::VecToTagT<IndexVec>{},
          static_cast<IndexElement>(physical_stride));
      const auto physical = vec::mul(logical_indices, stride_scale);
      request.indices = &physical;
    }
    vec::store_convert(TransformOutTag{}, chunk_base, transformed, request);
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    using HalfIndexVec = std::conditional_t<
        AddrKind == 2, vec::Vec<vec::Half<vec::VecToTagT<IndexVec>>>,
        IndexVec>;
    using HalfMaskVec = std::conditional_t<
        HasActive, vec::Mask<HalfTag>, MaskVec>;
    using HalfValueVec = vec::Vec<HalfTag>;
    const nint_t half_lanes = vec::size(HalfTag{});
    const auto lower_mask = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (HasActive) return vec::lower(chunk_tag, active_mask);
      else return MaskVec{};
    }();
    const auto upper_mask = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (HasActive) return vec::upper(chunk_tag, active_mask);
      else return MaskVec{};
    }();
    const auto lower_indices = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) return vec::lower(chunk_tag, logical_indices);
      else return IndexVec{};
    }();
    const auto upper_indices = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) return vec::upper(chunk_tag, logical_indices);
      else return IndexVec{};
    }();
    store_transform_vector<
        Policy, Transform, HalfTag, Memory, Context, HasActive, AddrKind>(
        HalfTag{}, base_pointer, lane_begin, physical_stride, lower_indices,
        lower_mask, vec::lower(chunk_tag, chunk_value), transform, context);
    store_transform_vector<
        Policy, Transform, HalfTag, Memory, Context, HasActive, AddrKind>(
        HalfTag{}, base_pointer, lane_begin + half_lanes, physical_stride,
        upper_indices, upper_mask, vec::upper(chunk_tag, chunk_value),
        transform, context);
  }
}

/**
 * Entry of the vectorized transform store; mirrors `load_transform_chunks`.
 */
template <
    typename Policy, typename Transform, vec::VectorTag RootTag,
    typename Memory, typename Context, typename StrideMeta,
    typename... Options>
VECOPS_ALWAYS_INLINE void store_transform_chunks(
    RootTag root_tag,
    Memory* pointer,
    StrideMeta tensor_axis_stride,
    vec::Vec<RootTag> value,
    const Transform& transform,
    const Context& context,
    Options&&... options) {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  store_transform_chunks_scalar<Policy>(
      root_tag, root_tag, value, pointer,
      static_cast<nint_t>(tensor_axis_stride), transform, context, 0,
      std::forward<Options>(options)...);
#else
  const auto request = vec::details::resolve_store_request<RootTag>(
      std::forward<Options>(options)...);
  using KernelRequest = decltype(request);
  constexpr bool HasActive =
      KernelRequest::active_kind != vec::Active::Unmasked;
  const vec::Mask<RootTag> active_mask = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (KernelRequest::active_kind == vec::Active::First) {
      return vec::mwhilelt(root_tag, 0, request.first_count);
    } else if constexpr (HasActive) {
      return *request.mask;
    } else {
      return vec::Mask<RootTag>{};
    }
  }();
  constexpr bool TensorUnitStride = is_definitely_one_meta_v<StrideMeta>;

  if constexpr (
      KernelRequest::addressing_kind == vec::Addressing::Indexed) {
    static_assert(
        KernelRequest::index_scale == 0,
        "tensor indexed addressing rejects byte scales");
    store_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 2>(
        root_tag, pointer, 0, static_cast<nint_t>(tensor_axis_stride),
        *request.indices, active_mask, value, transform, context);
  } else if constexpr (
      KernelRequest::addressing_kind == vec::Addressing::Contiguous &&
      TensorUnitStride) {
    store_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 0>(
        root_tag, pointer, 0, 1, vec::Vec<vec::IndexTag<RootTag>>{},
        active_mask, value,
        transform, context);
  } else {
    nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Strided) {
      physical_stride *= request.stride;
    }
    store_transform_vector<
        Policy, Transform, RootTag, Memory, Context, HasActive, 1>(
        root_tag, pointer, 0, physical_stride,
        vec::Vec<vec::IndexTag<RootTag>>{}, active_mask,
        value, transform, context);
  }
#endif
}

/**
 * Copies the active/populate fields between requests that share the same
 * resolved kinds, used when DataAccess rewrites only the addressing axis.
 */
template <typename OutRequest, typename InRequest>
VECOPS_ALWAYS_INLINE void copy_memory_request_fields(
    OutRequest& out, const InRequest& in) {
  if constexpr (OutRequest::active_kind == vec::Active::First) {
    out.first_count = in.first_count;
  }
  if constexpr (OutRequest::active_kind == vec::Active::Masked) {
    out.mask = in.mask;
  }
  if constexpr (OutRequest::populate_kind == vec::Populate::MergeVector) {
    out.merge_vector = in.merge_vector;
  }
  if constexpr (OutRequest::populate_kind == vec::Populate::MergeScalar) {
    out.merge_scalar = in.merge_scalar;
  }
}

/**
 * Lowering for memory element pairs whose memory-side rebind exceeds the
 * representation limit: composes addressing at this layer and defers to the
 * option-pack oversized partitioner.
 */
template <vec::VectorTag Tag,
          typename Pointer,
          typename Policy,
          typename StrideMeta,
          typename... Options>
VECOPS_ALWAYS_INLINE vec::Vec<Tag> load_memory_options(
    Tag tag,
    Pointer pointer,
    StrideMeta tensor_axis_stride,
    Options&&... options) {
  constexpr std::size_t indexed_count =
      vec::details::option_count<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count<vec::details::IsStridedOption, Options...>;
  auto retained = non_address_options(std::forward<Options>(options)...);
  using Order = typename Policy::ConversionOrderOption;
  using Value = typename Policy::ConversionValueOption;
  using Temporal = typename Policy::MemoryOptions::TemporalityOption;
  using Alignment = typename Policy::MemoryOptions::AlignmentOption;

  auto invoke = [&](auto&&... retained_options) VECOPS_INLINE_LAMBDA {
    if constexpr (indexed_count == 1) {
      auto&& indexed = vec::details::find_option<vec::details::IsIndexedOption>(
          std::forward<Options>(options)...);
      using Indexed = std::remove_cvref_t<decltype(indexed)>;
      static_assert(vec::details::IsIndexedOption<Indexed>::scale == 0,
                    "tensor indexed addressing rejects byte scales");
      using IndexTag = vec::VecToTagT<
          typename vec::details::IsIndexedOption<Indexed>::Value>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{}, static_cast<Index>(tensor_axis_stride));
      const auto physical = vec::mul(indexed.indices, scale);
      return vec::load_convert(
          tag, pointer, Order{}, Value{}, Temporal{},
          std::forward<decltype(retained_options)>(retained_options)...,
          vec::indexed(physical));
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (strided_count == 1) {
        auto&& strided = vec::details::find_option<
            vec::details::IsStridedOption>(
                std::forward<Options>(options)...);
        physical_stride *= static_cast<nint_t>(strided.stride);
      }
      constexpr bool KernelUnit =
          strided_count == 0 && is_definitely_one_meta_v<StrideMeta>;
      if constexpr (KernelUnit) {
        return vec::load_convert(
            tag, pointer, Order{}, Value{}, Temporal{}, Alignment{},
            std::forward<decltype(retained_options)>(retained_options)...);
      } else {
        return vec::load_convert(
            tag, pointer, Order{}, Value{}, Temporal{},
            std::forward<decltype(retained_options)>(retained_options)...,
            vec::strided(physical_stride));
      }
    }
  };
  return apply_inline(invoke, retained);
}

template <vec::VectorTag Tag,
          typename Pointer,
          typename Policy,
          typename StrideMeta,
          typename... Options>
VECOPS_ALWAYS_INLINE void store_memory_options(
    Tag tag,
    Pointer pointer,
    vec::Vec<Tag> value,
    StrideMeta tensor_axis_stride,
    Options&&... options) {
  constexpr std::size_t indexed_count =
      vec::details::option_count<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count<vec::details::IsStridedOption, Options...>;
  auto retained = non_address_options(std::forward<Options>(options)...);
  using Order = typename Policy::ConversionOrderOption;
  using Value = typename Policy::ConversionValueOption;
  using Temporal = typename Policy::MemoryOptions::TemporalityOption;
  using Packing = typename Policy::MemoryOptions::PackingOption;
  using Alignment = typename Policy::MemoryOptions::AlignmentOption;

  auto invoke = [&](auto&&... retained_options) VECOPS_INLINE_LAMBDA {
    if constexpr (indexed_count == 1) {
      auto&& indexed = vec::details::find_option<vec::details::IsIndexedOption>(
          std::forward<Options>(options)...);
      using Indexed = std::remove_cvref_t<decltype(indexed)>;
      static_assert(vec::details::IsIndexedOption<Indexed>::scale == 0,
                    "tensor indexed addressing rejects byte scales");
      using IndexTag = vec::VecToTagT<
          typename vec::details::IsIndexedOption<Indexed>::Value>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{}, static_cast<Index>(tensor_axis_stride));
      const auto physical = vec::mul(indexed.indices, scale);
      vec::store_convert(
          tag, pointer, value, Order{}, Value{}, Temporal{}, Packing{},
          std::forward<decltype(retained_options)>(retained_options)...,
          vec::indexed(physical));
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (strided_count == 1) {
        auto&& strided = vec::details::find_option<
            vec::details::IsStridedOption>(
                std::forward<Options>(options)...);
        physical_stride *= static_cast<nint_t>(strided.stride);
      }
      constexpr bool KernelUnit =
          strided_count == 0 && is_definitely_one_meta_v<StrideMeta>;
      if constexpr (KernelUnit) {
        vec::store_convert(
            tag, pointer, value, Order{}, Value{}, Temporal{}, Packing{},
            Alignment{},
            std::forward<decltype(retained_options)>(retained_options)...);
      } else {
        vec::store_convert(
            tag, pointer, value, Order{}, Value{}, Temporal{}, Packing{},
            std::forward<decltype(retained_options)>(retained_options)...,
            vec::strided(physical_stride));
      }
    }
  };
  apply_inline(invoke, retained);
}

/**
 * Request-driven memory lowering: folds the kernel options once and hands the
 * composed addressing straight to the parse-free converting-load entry.
 */
template <vec::VectorTag Tag,
          typename Pointer,
          typename Policy,
          typename StrideMeta,
          typename... Options>
VECOPS_ALWAYS_INLINE vec::Vec<Tag> load_memory(
    Tag tag,
    Pointer pointer,
    StrideMeta tensor_axis_stride,
    Options&&... options) {
  using MemoryElement =
      std::remove_cvref_t<std::remove_pointer_t<Pointer>>;
  using Order = typename Policy::ConversionOrderOption;
  using Value = typename Policy::ConversionValueOption;
  using Temporal = typename Policy::MemoryOptions::TemporalityOption;
  using Alignment = typename Policy::MemoryOptions::AlignmentOption;
  // The layout stride's meta type decides the physical form at compile time:
  // a proven unit stride lowers to the contiguous request, everything else
  // to strided. DataAccess itself carries no runtime stride branch.
  constexpr bool TensorUnitStride = is_definitely_one_meta_v<StrideMeta>;

  if constexpr (
      std::same_as<MemoryElement, vec::ElementOf<Tag>> ||
      vec::details::memory_rebind_supported<Tag, MemoryElement>()) {
    const auto request = vec::details::resolve_load_request<Tag>(
        std::forward<Options>(options)...);
    using KernelRequest = decltype(request);

    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Indexed) {
      static_assert(
          KernelRequest::index_scale == 0,
          "tensor indexed addressing rejects byte scales");
      using IndexTag =
          vec::VecToTagT<typename KernelRequest::IndexVectorType>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{},
          static_cast<Index>(static_cast<nint_t>(tensor_axis_stride)));
      const auto physical = vec::mul(*request.indices, scale);
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Indexed, KernelRequest::populate_kind,
          vec::mem::Unaligned, Temporal, 0,
          std::remove_cvref_t<decltype(physical)>, Order, Value>
          out{};
      copy_memory_request_fields(out, request);
      out.indices = &physical;
      return vec::load_convert(tag, pointer, out);
    } else if constexpr (
        KernelRequest::addressing_kind ==
            vec::Addressing::Contiguous &&
        TensorUnitStride) {
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Contiguous, KernelRequest::populate_kind,
          Alignment, Temporal, 0>
          out{};
      copy_memory_request_fields(out, request);
      return vec::load_convert(tag, pointer, out);
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (
          KernelRequest::addressing_kind == vec::Addressing::Strided) {
        physical_stride *= request.stride;
      }
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Strided, KernelRequest::populate_kind,
          vec::mem::Unaligned, Temporal, 0>
          out{};
      copy_memory_request_fields(out, request);
      out.stride = physical_stride;
      return vec::load_convert(tag, pointer, out);
    }
  } else {
    return load_memory_options<Tag, Pointer, Policy>(
        tag, pointer, tensor_axis_stride,
        std::forward<Options>(options)...);
  }
}

/** Request-driven converting-store lowering, mirroring `load_memory`. */
template <vec::VectorTag Tag,
          typename Pointer,
          typename Policy,
          typename StrideMeta,
          typename... Options>
VECOPS_ALWAYS_INLINE void store_memory(
    Tag tag,
    Pointer pointer,
    vec::Vec<Tag> value,
    StrideMeta tensor_axis_stride,
    Options&&... options) {
  using MemoryElement =
      std::remove_cvref_t<std::remove_pointer_t<Pointer>>;
  using Order = typename Policy::ConversionOrderOption;
  using Value = typename Policy::ConversionValueOption;
  using Temporal = typename Policy::MemoryOptions::TemporalityOption;
  using Packing = typename Policy::MemoryOptions::PackingOption;
  using Alignment = typename Policy::MemoryOptions::AlignmentOption;
  constexpr bool TensorUnitStride = is_definitely_one_meta_v<StrideMeta>;

  if constexpr (
      std::same_as<MemoryElement, vec::ElementOf<Tag>> ||
      vec::details::memory_rebind_supported<Tag, MemoryElement>()) {
    const auto request = vec::details::resolve_store_request<Tag>(
        std::forward<Options>(options)...);
    using KernelRequest = decltype(request);

    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Indexed) {
      static_assert(
          KernelRequest::index_scale == 0,
          "tensor indexed addressing rejects byte scales");
      using IndexTag =
          vec::VecToTagT<typename KernelRequest::IndexVectorType>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{},
          static_cast<Index>(static_cast<nint_t>(tensor_axis_stride)));
      const auto physical = vec::mul(*request.indices, scale);
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Indexed, vec::mem::Unaligned, Temporal, 0,
          std::remove_cvref_t<decltype(physical)>, Order, Value, Packing>
          out{};
      if constexpr (out.active_kind == vec::Active::First) {
        out.first_count = request.first_count;
      }
      if constexpr (out.active_kind == vec::Active::Masked) {
        out.mask = request.mask;
      }
      out.indices = &physical;
      vec::store_convert(tag, pointer, value, out);
    } else if constexpr (
        KernelRequest::addressing_kind ==
            vec::Addressing::Contiguous &&
        TensorUnitStride) {
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Contiguous, Alignment, Temporal, 0>
          out{};
      if constexpr (out.active_kind == vec::Active::First) {
        out.first_count = request.first_count;
      }
      if constexpr (out.active_kind == vec::Active::Masked) {
        out.mask = request.mask;
      }
      vec::store_convert(tag, pointer, value, out);
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (
          KernelRequest::addressing_kind == vec::Addressing::Strided) {
        physical_stride *= request.stride;
      }
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Strided, vec::mem::Unaligned, Temporal, 0>
          out{};
      if constexpr (out.active_kind == vec::Active::First) {
        out.first_count = request.first_count;
      }
      if constexpr (out.active_kind == vec::Active::Masked) {
        out.mask = request.mask;
      }
      out.stride = physical_stride;
      vec::store_convert(tag, pointer, value, out);
    }
  } else {
    store_memory_options<Tag, Pointer, Policy>(
        tag, pointer, value, tensor_axis_stride,
        std::forward<Options>(options)...);
  }
}

template <typename Layout>
VECOPS_ALWAYS_INLINE nint_t tensor_numel(const Layout& layout) {
  nint_t result = 1;
  for (int d = 0; d < Layout::Ndim; ++d) result *= layout.shape()[d];
  return result;
}

template <int Axis, typename Layout>
VECOPS_INLINE auto auxiliary_layout(const Layout& layout) {
  constexpr int Rank = Layout::Ndim;
  std::array<nint_t, Rank> shape{};
  std::array<nint_t, Rank> strides{};
  for (int d = 0; d < Rank; ++d) shape[d] = layout.shape()[d];
  strides[Axis] = 1;
  nint_t next = shape[Axis];
  for (int d = Rank - 1; d >= 0; --d) {
    if (d == Axis) continue;
    strides[d] = next;
    next *= shape[d];
  }
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return make_layout(
        make_shape(Any{shape[I]}...),
        make_strides(Any{strides[I]}...));
  }(std::make_index_sequence<Rank>{});
}

template <int SkipDim, int Dim = 0, typename Layout, typename Fn>
VECOPS_INLINE void for_each_line(
    const Layout& layout, Coord<Layout::Ndim>& position, Fn&& fn) {
  if constexpr (Dim == Layout::Ndim) {
    std::forward<Fn>(fn)(position);
  } else if constexpr (Dim == SkipDim) {
    position[Dim] = 0;
    for_each_line<SkipDim, Dim + 1>(
        layout, position, std::forward<Fn>(fn));
  } else {
    for (position[Dim] = 0; position[Dim] < layout.shape()[Dim];
         ++position[Dim]) {
      for_each_line<SkipDim, Dim + 1>(
          layout, position, std::forward<Fn>(fn));
    }
  }
}

template <int Dim = 0, typename Layout, typename Fn>
VECOPS_INLINE void for_each_coordinate(
    const Layout& layout, Coord<Layout::Ndim>& position, Fn&& fn) {
  if constexpr (Dim == Layout::Ndim) {
    std::forward<Fn>(fn)(position);
  } else {
    for (position[Dim] = 0; position[Dim] < layout.shape()[Dim];
         ++position[Dim]) {
      for_each_coordinate<Dim + 1>(
          layout, position, std::forward<Fn>(fn));
    }
  }
}

/** True when the meta stride type can never hold the value 1. */
template <typename T>
struct MetaCannotBeOne {
  static constexpr bool value = [] {
    if constexpr (T::is_const) {
      return T::value != 1;
    } else {
      return T::has_lower && T::lo >= 2;
    }
  }();
};

template <typename T>
inline constexpr bool meta_cannot_be_one_v =
    MetaCannotBeOne<std::remove_cvref_t<T>>::value;

template <typename StridesPack, int SkipDim>
struct OtherAxisStrideScan;

template <typename... Ts, int SkipDim>
struct OtherAxisStrideScan<Strides<Ts...>, SkipDim> {
  static constexpr std::size_t kCount = sizeof...(Ts);
  static_assert(
      static_cast<std::size_t>(SkipDim) < kCount, "skip dim out of range");

  template <std::size_t... Is>
  static consteval bool any_definitely_one(std::index_sequence<Is...>) {
    return ((Is != static_cast<std::size_t>(SkipDim) &&
             is_definitely_one_meta_v<
                 typename MetaElement<Is, Strides<Ts...>>::type>) ||
            ...);
  }

  template <std::size_t... Is>
  static consteval bool all_cannot_be_one(std::index_sequence<Is...>) {
    return ((Is == static_cast<std::size_t>(SkipDim) ||
             meta_cannot_be_one_v<
                 typename MetaElement<Is, Strides<Ts...>>::type>) &&
            ...);
  }

  static constexpr bool any_unit_other =
      any_definitely_one(std::make_index_sequence<kCount>{});
  static constexpr bool no_unit_other =
      all_cannot_be_one(std::make_index_sequence<kCount>{});
};

/**
 * Resolves the automatic storage plan entirely at compile time from the
 * layout's meta stride types. A stride that cannot be proven equal to one
 * (Const<N != 1>, or a Dynamic whose bounds do not pin it to 1) is treated
 * as non-unit: dynamic layouts that happen to be contiguous at runtime use
 * the gather/scatter path, never a runtime branch here.
 */
template <typename Spec, typename Policy>
consteval AccessPlan resolve_plan() {
  if constexpr (Policy::requested_plan != AccessPlan::automatic) {
    return Policy::requested_plan;
  } else if constexpr (Spec::is_input) {
    if constexpr (!transform_reads_input<typename Spec::TransformType>) {
      return AccessPlan::direct;
    }
    using AxisStride = typename MetaElement<
        Policy::vector_axis,
        typename Spec::InputLayout::Strides>::type;
    if constexpr (Policy::read_passes == 1 ||
                  is_definitely_one_meta_v<AxisStride>) {
      return AccessPlan::direct;
    } else {
      return AccessPlan::materialize_after_transform;
    }
  } else {
    using AxisStride = typename MetaElement<
        Policy::vector_axis,
        typename Spec::OutputLayout::Strides>::type;
    using Scan = OtherAxisStrideScan<
        typename Spec::OutputLayout::Strides, Policy::vector_axis>;
    if constexpr (is_definitely_one_meta_v<AxisStride>) {
      return AccessPlan::direct;
    } else if constexpr (Scan::any_unit_other) {
      return AccessPlan::materialize_after_transform;
    } else {
      return AccessPlan::direct;
    }
  }
}

} // namespace details

template <
    typename Compute,
    typename Tensor,
    typename Transform = NoTransform,
    typename Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>,
    typename... Facts>
class InputSpec {
public:
  static constexpr bool is_input = true;
  using ComputeType = Compute;
  using InputTensor = Tensor;
  using MemoryElement = std::remove_const_t<typename Tensor::ElementType>;
  using TransformType = Transform;
  using ProjectionType = Projection;
  using ExternalFacts = std::tuple<Facts...>;
  using InputLayout = typename Tensor::Layout;

  InputSpec(
      Tensor tensor, Transform transform, Projection projection,
      Facts... facts)
      : tensor_(tensor), transform_(std::move(transform)),
        projection_(projection), facts_(std::move(facts)...) {}

  InputSpec(Tensor tensor, Transform transform)
    requires (sizeof...(Facts) == 0)
      : InputSpec(
            tensor, std::move(transform),
            identity_projection<Tensor::Ndim>()) {}

  explicit InputSpec(Tensor tensor)
    requires (std::same_as<Transform, NoTransform> &&
              sizeof...(Facts) == 0)
      : InputSpec(tensor, NoTransform{}) {}

  const Tensor& tensor() const { return tensor_; }
  const InputLayout& input_layout() const { return tensor_.layout(); }
  const Transform& transform() const { return transform_; }
  const Projection& projection() const { return projection_; }
  const ExternalFacts& facts() const { return facts_; }

private:
  Tensor tensor_;
  [[no_unique_address]] Transform transform_;
  Projection projection_;
  [[no_unique_address]] ExternalFacts facts_;
};

/**
 * @brief Caller-side description of a writable Tensor operand.
 *
 * OutputSpec has the same separation of responsibilities as InputSpec but
 * statically rejects Tensor element types qualified with `const`. Prefer the
 * `output<Compute>()` factory.
 */
template <
    typename Compute,
    typename Tensor,
    typename Transform = NoTransform,
    typename Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>,
    typename... Facts>
class OutputSpec {
public:
  static constexpr bool is_input = false;
  static_assert(!std::is_const_v<typename Tensor::ElementType>,
                "tensor::output requires a mutable Tensor");
  using ComputeType = Compute;
  using OutputTensor = Tensor;
  using MemoryElement = typename Tensor::ElementType;
  using TransformType = Transform;
  using ProjectionType = Projection;
  using ExternalFacts = std::tuple<Facts...>;
  using OutputLayout = typename Tensor::Layout;

  OutputSpec(
      Tensor tensor, Transform transform, Projection projection,
      Facts... facts)
      : tensor_(tensor), transform_(std::move(transform)),
        projection_(projection), facts_(std::move(facts)...) {}

  OutputSpec(Tensor tensor, Transform transform)
    requires (sizeof...(Facts) == 0)
      : OutputSpec(
            tensor, std::move(transform),
            identity_projection<Tensor::Ndim>()) {}

  explicit OutputSpec(Tensor tensor)
    requires (std::same_as<Transform, NoTransform> &&
              sizeof...(Facts) == 0)
      : OutputSpec(tensor, NoTransform{}) {}

  const Tensor& tensor() const { return tensor_; }
  const OutputLayout& output_layout() const { return tensor_.layout(); }
  const Transform& transform() const { return transform_; }
  const Projection& projection() const { return projection_; }
  const ExternalFacts& facts() const { return facts_; }

private:
  Tensor tensor_;
  [[no_unique_address]] Transform transform_;
  Projection projection_;
  [[no_unique_address]] ExternalFacts facts_;
};

/**
 * @brief Build an input Spec with no prologue transform.
 * @tparam Compute Element type requested by the kernel's vector Tags.
 * @param tensor Readable non-owning Tensor view.
 * @param facts Optional facts such as `assume_aligned<N>`.
 *
 * `NoTransform` enables direct fused MemoryType-to-Compute conversion.
 */
template <typename Compute, typename Tensor, typename... Facts>
  requires ((details::is_operand_fact_v<Facts>) && ...)
VECOPS_INLINE auto input(Tensor tensor, Facts... facts) {
  (details::validate_operand_fact(tensor, facts), ...);
  using Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>;
  return InputSpec<
      Compute, Tensor, NoTransform, Projection,
      std::remove_cvref_t<Facts>...>{
          tensor, NoTransform{}, identity_projection<Tensor::Ndim>(),
          std::move(facts)...};
}

/**
 * @brief Build an input Spec with a pure prologue transform.
 *
 * A raw callable is normalized as a coordinate-aware
 * `MemoryElement -> Compute` transform. A typed transform keeps its visible
 * `TIn/TOut`; DataAccess inserts boundary conversions instead of wrapping it.
 */
template <typename Compute, typename Tensor, typename Transform,
          typename... Facts>
  requires (!details::is_operand_fact_v<Transform> &&
            (details::is_operand_fact_v<Facts> && ...))
VECOPS_INLINE auto input(
    Tensor tensor, Transform&& transform, Facts... facts) {
  (details::validate_operand_fact(tensor, facts), ...);
  using Memory = std::remove_const_t<typename Tensor::ElementType>;
  auto normalized = details::normalize_transform<Transform, Memory, Compute>(
      std::forward<Transform>(transform));
  using Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>;
  return InputSpec<
      Compute, Tensor, decltype(normalized), Projection,
      std::remove_cvref_t<Facts>...>{
          tensor, std::move(normalized),
          identity_projection<Tensor::Ndim>(), std::move(facts)...};
}

/**
 * @brief Build an output Spec with no epilogue transform.
 * @note `tensor` must be writable; a Tensor created from `const T*` is rejected.
 */
template <typename Compute, typename Tensor, typename... Facts>
  requires (!std::is_const_v<typename Tensor::ElementType> &&
            (details::is_operand_fact_v<Facts> && ...))
VECOPS_INLINE auto output(Tensor tensor, Facts... facts) {
  (details::validate_operand_fact(tensor, facts), ...);
  using Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>;
  return OutputSpec<
      Compute, Tensor, NoTransform, Projection,
      std::remove_cvref_t<Facts>...>{
          tensor, NoTransform{}, identity_projection<Tensor::Ndim>(),
          std::move(facts)...};
}

/**
 * @brief Build an output Spec with a pure epilogue transform.
 *
 * The kernel stores Compute vectors. DataAccess converts them to the
 * transform's `TIn`, applies the transform, then converts `TOut` to the Tensor
 * memory element type at the actual store boundary.
 */
template <typename Compute, typename Tensor, typename Transform,
          typename... Facts>
  requires (!std::is_const_v<typename Tensor::ElementType> &&
            !details::is_operand_fact_v<Transform> &&
            (details::is_operand_fact_v<Facts> && ...))
VECOPS_INLINE auto output(
    Tensor tensor, Transform&& transform, Facts... facts) {
  (details::validate_operand_fact(tensor, facts), ...);
  using Memory = typename Tensor::ElementType;
  auto normalized = details::normalize_transform<Transform, Compute, Memory>(
      std::forward<Transform>(transform));
  using Projection = CoordinateProjection<Tensor::Ndim, Tensor::Ndim>;
  return OutputSpec<
      Compute, Tensor, decltype(normalized), Projection,
      std::remove_cvref_t<Facts>...>{
          tensor, std::move(normalized),
          identity_projection<Tensor::Ndim>(), std::move(facts)...};
}

template <typename T>
struct IsInputSpec : std::false_type {};
template <typename... Args>
struct IsInputSpec<InputSpec<Args...>> : std::true_type {};
template <typename T>
inline constexpr bool is_input_spec_v =
    IsInputSpec<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsOutputSpec : std::false_type {};
template <typename... Args>
struct IsOutputSpec<OutputSpec<Args...>> : std::true_type {};
template <typename T>
inline constexpr bool is_output_spec_v =
    IsOutputSpec<std::remove_cvref_t<T>>::value;

template <typename Spec, typename Policy>
class InputDataAccess;

template <typename Spec, typename Policy>
class OutputDataAccess;

namespace details {

template <int SlicedDim, int Dim>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) slice_argument(
    nint_t index) {
  if constexpr (SlicedDim == Dim) return index;
  else return reserve;
}

template <int SlicedDim, typename Tensor, std::size_t... I>
VECOPS_INLINE constexpr auto slice_tensor_impl(
    const Tensor& tensor, nint_t index, std::index_sequence<I...>) {
  return tensor(slice_argument<SlicedDim, static_cast<int>(I)>(index)...);
}

} // namespace details

/**
 * @brief Slice one input Spec dimension while preserving original coordinates.
 *
 * The returned Spec owns its sliced Tensor descriptor and a composed
 * CoordinateProjection, so coordinate-aware transforms still observe the
 * original full-rank coordinate.
 */
template <int Dim, typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
VECOPS_INLINE auto slice_view(
    const InputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec,
    nint_t index) {
  static_assert(0 <= Dim && Dim < Tensor::Ndim);
  auto sliced_tensor = details::slice_tensor_impl<Dim>(
      spec.tensor(), index, std::make_index_sequence<Tensor::Ndim>{});
  auto projection = spec.projection().template sliced<Dim>(index);
  return InputSpec<Compute, decltype(sliced_tensor), Transform,
                   decltype(projection)>{
      sliced_tensor, spec.transform(), projection};
}

/**
 * @brief Slice one output Spec dimension while preserving original coordinates.
 * @return A new OutputSpec with sliced Tensor and composed projection.
 */
template <int Dim, typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
VECOPS_INLINE auto slice_view(
    const OutputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec,
    nint_t index) {
  static_assert(0 <= Dim && Dim < Tensor::Ndim);
  auto sliced_tensor = details::slice_tensor_impl<Dim>(
      spec.tensor(), index, std::make_index_sequence<Tensor::Ndim>{});
  auto projection = spec.projection().template sliced<Dim>(index);
  return OutputSpec<Compute, decltype(sliced_tensor), Transform,
                    decltype(projection)>{
      sliced_tensor, spec.transform(), projection};
}

template <typename Access, typename Tag, int Dim, typename Mapping>
class ScanCursor;

template <typename Access, typename Tag, int TraverseDim, int VectorDim>
class ProjectCursor;

/**
 * @brief Direct readable DataAccess session.
 *
 * `load()` returns exactly the caller's Compute Tag regardless of Tensor memory
 * dtype or transform dtype/width. Axis defaults to `Rank - 1`. The session also
 * creates ScanCursor/ProjectCursor objects.
 *
 * Instances borrow their Spec and Tensor storage and must not outlive them.
 * Kernels should normally receive this type through `kernel::with_operands` so
 * automatic planning has already been resolved.
 */
template <typename Spec, typename Policy>
class InputDataAccess {
public:
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  using Transform = typename Spec::TransformType;
  static constexpr int Rank = Spec::InputTensor::Ndim;

  InputDataAccess(const Spec& spec, Policy policy = {})
      : spec_(&spec), data_(spec.tensor().data()), policy_(policy) {
    static_assert(Policy::vector_axis < Rank);
    static_assert(
        !details::is_unordered_policy<Policy> ||
            (Policy::permutation_safe &&
             details::transform_permutation_equivariant<Transform>),
        "unordered input conversion requires a permutation-safe kernel and transform");
  }

  /**
   * @brief Load a Compute vector along the final Tensor axis.
   * @param tag Requested result Tag; its element type must be ComputeType.
   * @param position Logical coordinate of lane zero.
   * @param options Zero or one option from each active/address/population group.
   */
  template <vec::VectorTag Tag, typename... Options>
    requires std::same_as<vec::ElementOf<Tag>, ComputeType>
  VECOPS_ALWAYS_INLINE vec::Vec<Tag> load(
      Tag tag, const Coord<Rank>& position, Options&&... options) const {
    return load(
        tag, position, axis<Rank - 1>,
        std::forward<Options>(options)...);
  }

  /**
   * @brief Load a Compute vector along compile-time axis `Dim`.
   *
   * Addressing options are interpreted in logical axis elements before Layout
   * stride lowering. `first(n)` is the preferred tail form because DataAccess
   * can derive every transform-side and memory-side mask from it.
   */
  template <vec::VectorTag Tag, int Dim, typename... Options>
    requires std::same_as<vec::ElementOf<Tag>, ComputeType>
  VECOPS_ALWAYS_INLINE vec::Vec<Tag> load(
      Tag tag,
      const Coord<Rank>& position,
      Axis<Dim>,
      Options&&... options) const {
    static_assert(0 <= Dim && Dim < Rank);
    details::validate_access_options<true, Options...>();
    const auto& tensor = spec_->tensor();
    using StrideMeta = typename details::MetaElement<
        Dim, typename Spec::InputLayout::Strides>::type;
    constexpr bool UnitRankOne =
        Rank == 1 && details::is_const_one_meta_v<StrideMeta>;
    const nint_t base = [&] {
      if constexpr (UnitRankOne) return position[0];
      else return offset_at(tensor.layout(), position);
    }();
    const StrideMeta axis_stride = [&]() -> StrideMeta {
      if constexpr (UnitRankOne) {
        return StrideMeta{nint_t{1}};
      } else {
        return StrideMeta{stride<Dim>(tensor.layout())};
      }
    }();

    if constexpr (details::is_no_transform<Transform>) {
      return details::load_memory<Tag, const MemoryElement*, Policy>(
          tag, data_ + base, axis_stride,
          std::forward<Options>(options)...);
    } else {
      return details::with_transform_context(
          tag, position, Dim, spec_->projection(),
          [&](const auto& context) VECOPS_INLINE_LAMBDA {
            vec::Vec<Tag> result;
            constexpr bool NeedsLogicalMaskLowering =
                (vec::details::option_count<
                     vec::details::IsMaskedOption, Options...> != 0 ||
                 vec::details::option_count<
                     vec::details::IsVectorMergeOption, Options...> != 0 ||
                 vec::details::option_count<
                     vec::details::IsScalarMergeOption, Options...> != 0) &&
                !std::same_as<typename Transform::TIn, ComputeType>;
            if constexpr (!details::transform_tags_representable<
                              Transform, Tag>()) {
              result = details::load_transform_chunks<Policy>(
                  tag, data_ + base, axis_stride, spec_->transform(),
                  context, std::forward<Options>(options)...);
            } else if constexpr (
                details::can_transform_chunk<
                    Transform, Tag, decltype(context)>() &&
                !NeedsLogicalMaskLowering) {
              // Keep rebound aliases inside the proven-representable branch.
              // On sizeless SVE, merely exposing Vec<Rebind<...>> in the
              // enclosing generic lambda can instantiate an over-wide
              // representation before the discarded branch is pruned.
              using TransformInTag =
                  vec::Rebind<typename Transform::TIn, Tag>;
              using TransformOutTag =
                  vec::Rebind<typename Transform::TOut, Tag>;
              auto transform_input = [&]() VECOPS_INLINE_LAMBDA {
                if constexpr (details::transform_reads_input<Transform>) {
                  return details::load_memory<
                      TransformInTag, const MemoryElement*, Policy>(
                          TransformInTag{}, data_ + base, axis_stride,
                          std::forward<Options>(options)...);
                } else {
                  return vec::zeros(TransformInTag{});
                }
              }();
              auto transformed = spec_->transform()(
                  TransformOutTag{}, transform_input, context);
              result = vec::convert(
                  tag, TransformOutTag{}, transformed,
                  typename Policy::ConversionOrderOption{},
                  typename Policy::ConversionValueOption{});
            } else {
              result = details::load_transform_chunks<Policy>(
                  tag, data_ + base, axis_stride, spec_->transform(),
                  context, std::forward<Options>(options)...);
            }
            return details::populate_inactive(
                tag, result, std::forward<Options>(options)...);
          },
          options...);
    }
  }

  /**
   * @brief Create a same-axis ScanCursor.
   * @param count Number of logical sequence elements remaining, not bytes and
   * not number of vectors.
   * @param mapping Contiguous or fixed affine lane mapping.
   *
   * An affine full advance changes the axis coordinate by
   * `vec::size(tag) * mapping.stride`. Indexed mappings are not accepted.
   */
  template <vec::VectorTag Tag, int Dim,
            typename Mapping = ContiguousLaneMapping>
    requires (std::same_as<Mapping, ContiguousLaneMapping> ||
              std::same_as<Mapping, AffineLaneMapping>)
  VECOPS_INLINE auto scan(
      Tag tag,
      Coord<Rank> origin,
      Axis<Dim>,
      nint_t count,
      Mapping mapping = {}) {
    return ScanCursor<InputDataAccess, Tag, Dim, Mapping>{
        *this, tag, origin, count, mapping};
  }

  /** @brief Convenience ScanCursor overload accepting `vec::strided(s)`. */
  template <vec::VectorTag Tag, int Dim, typename Stride>
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim> dim, nint_t count,
      vec::opt::Strided<Stride> mapping) {
    return scan(
        tag, origin, dim, count,
        AffineLaneMapping{static_cast<nint_t>(mapping.stride)});
  }

  /**
   * @brief Create a ProjectCursor with distinct traversal and vector axes.
   * @param iterations Number of cursor positions.
   * @param traversal_step Logical elements added on each `advance()`.
   *
   * Every operation may use a different Tag and access options.
   */
  template <vec::VectorTag Tag, int TraverseDim, int VectorDim>
  VECOPS_INLINE auto project(
      Tag tag,
      Coord<Rank> origin,
      TraversalAxis<TraverseDim>,
      VectorAxis<VectorDim>,
      nint_t iterations,
      nint_t traversal_step = 1) {
    static_assert(TraverseDim != VectorDim);
    return ProjectCursor<
        InputDataAccess, Tag, TraverseDim, VectorDim>{
            *this, tag, origin, iterations, traversal_step};
  }

  const Spec& spec() const { return *spec_; }
  const Policy& policy() const { return policy_; }

private:
  const Spec* spec_;
  const MemoryElement* data_;
  [[no_unique_address]] Policy policy_;
};

/**
 * @brief Direct writable DataAccess session.
 *
 * `store()` accepts ComputeType vectors and executes the output conversion and
 * epilogue pipeline at the selected logical coordinate. The type is move-only
 * to match materialized output session usage. `commit()` is a no-op marker for
 * direct storage, allowing kernels to commit unconditionally without knowing
 * the resolved plan.
 */
template <typename Spec, typename Policy>
class OutputDataAccess {
public:
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  using Transform = typename Spec::TransformType;
  static constexpr int Rank = Spec::OutputTensor::Ndim;

  OutputDataAccess(const Spec& spec, Policy policy = {})
      : spec_(&spec), data_(spec.tensor().data()), policy_(policy) {
    static_assert(Policy::vector_axis < Rank);
    static_assert(
        !details::is_unordered_policy<Policy> ||
            (Policy::permutation_safe &&
             details::transform_permutation_equivariant<Transform>),
        "unordered output conversion requires a permutation-safe kernel and transform");
  }

  OutputDataAccess(const OutputDataAccess&) = delete;
  OutputDataAccess& operator=(const OutputDataAccess&) = delete;
  OutputDataAccess(OutputDataAccess&&) = default;
  OutputDataAccess& operator=(OutputDataAccess&&) = default;

  /** @brief Store a Compute vector along the final Tensor axis. */
  template <vec::VectorTag Tag, typename... Options>
    requires std::same_as<vec::ElementOf<Tag>, ComputeType>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag,
      const Coord<Rank>& position,
      vec::Vec<Tag> value,
      Options&&... options) const {
    store(
        tag, position, axis<Rank - 1>, value,
        std::forward<Options>(options)...);
  }

  /**
   * @brief Store a Compute vector along compile-time axis `Dim`.
   *
   * Active and addressing options describe caller-visible logical lanes.
   * Inactive lanes never modify Tensor memory. Duplicate active indexed lanes
   * retain the scatter conflict semantics of the selected vec backend.
   */
  template <vec::VectorTag Tag, int Dim, typename... Options>
    requires std::same_as<vec::ElementOf<Tag>, ComputeType>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag,
      const Coord<Rank>& position,
      Axis<Dim>,
      vec::Vec<Tag> value,
      Options&&... options) const {
    static_assert(0 <= Dim && Dim < Rank);
    details::validate_access_options<false, Options...>();
    const auto& tensor = spec_->tensor();
    using StrideMeta = typename details::MetaElement<
        Dim, typename Spec::OutputLayout::Strides>::type;
    constexpr bool UnitRankOne =
        Rank == 1 && details::is_const_one_meta_v<StrideMeta>;
    const nint_t base = [&] {
      if constexpr (UnitRankOne) return position[0];
      else return offset_at(tensor.layout(), position);
    }();
    const StrideMeta axis_stride = [&]() -> StrideMeta {
      if constexpr (UnitRankOne) {
        return StrideMeta{nint_t{1}};
      } else {
        return StrideMeta{stride<Dim>(tensor.layout())};
      }
    }();
    if constexpr (details::is_no_transform<Transform>) {
      details::store_memory<Tag, MemoryElement*, Policy>(
          tag, data_ + base, value, axis_stride,
          std::forward<Options>(options)...);
    } else {
      details::with_transform_context(
          tag, position, Dim, spec_->projection(),
          [&](const auto& context) VECOPS_INLINE_LAMBDA {
            vec::Vec<Tag> result;
            constexpr bool NeedsLogicalMaskLowering =
                vec::details::option_count<
                    vec::details::IsMaskedOption, Options...> != 0 &&
                !std::same_as<typename Transform::TOut, ComputeType>;
            if constexpr (!details::transform_tags_representable<
                              Transform, Tag>()) {
              details::store_transform_chunks<Policy>(
                  tag, data_ + base, axis_stride, value,
                  spec_->transform(), context,
                  std::forward<Options>(options)...);
            } else if constexpr (
                details::can_transform_chunk<
                    Transform, Tag, decltype(context)>() &&
                !NeedsLogicalMaskLowering) {
              using TransformOutTag =
                  vec::Rebind<typename Transform::TOut, Tag>;
              using TransformInTag =
                  vec::Rebind<typename Transform::TIn, Tag>;
              auto transform_input = vec::convert(
                  TransformInTag{}, tag, value,
                  typename Policy::ConversionOrderOption{},
                  typename Policy::ConversionValueOption{});
              auto transformed = spec_->transform()(
                  TransformOutTag{}, transform_input, context);
              details::store_memory<
                  TransformOutTag, MemoryElement*, Policy>(
                      TransformOutTag{}, data_ + base, transformed,
                      axis_stride,
                      std::forward<Options>(options)...);
            } else {
              details::store_transform_chunks<Policy>(
                  tag, data_ + base, axis_stride, value,
                  spec_->transform(), context,
                  std::forward<Options>(options)...);
            }
          },
          options...);
    }
  }

  template <vec::VectorTag Tag, int Dim,
            typename Mapping = ContiguousLaneMapping>
    requires (std::same_as<Mapping, ContiguousLaneMapping> ||
              std::same_as<Mapping, AffineLaneMapping>)
  VECOPS_INLINE auto scan(
      Tag tag,
      Coord<Rank> origin,
      Axis<Dim>,
      nint_t count,
      Mapping mapping = {}) {
    return ScanCursor<OutputDataAccess, Tag, Dim, Mapping>{
        *this, tag, origin, count, mapping};
  }

  template <vec::VectorTag Tag, int Dim, typename Stride>
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim> dim, nint_t count,
      vec::opt::Strided<Stride> mapping) {
    return scan(
        tag, origin, dim, count,
        AffineLaneMapping{static_cast<nint_t>(mapping.stride)});
  }

  template <vec::VectorTag Tag, int TraverseDim, int VectorDim>
  VECOPS_INLINE auto project(
      Tag tag,
      Coord<Rank> origin,
      TraversalAxis<TraverseDim>,
      VectorAxis<VectorDim>,
      nint_t iterations,
      nint_t traversal_step = 1) {
    static_assert(TraverseDim != VectorDim);
    return ProjectCursor<
        OutputDataAccess, Tag, TraverseDim, VectorDim>{
            *this, tag, origin, iterations, traversal_step};
  }

  /**
   * @brief Mark the direct output complete.
   * @note No copy is performed. The method exists so kernels can use the same
   * unconditional commit discipline for direct and materialized sessions.
   */
  void commit() { committed_ = true; }

  /** @brief Whether this direct session has been explicitly committed. */
  bool committed() const { return committed_; }

  const Spec& spec() const { return *spec_; }
  const Policy& policy() const { return policy_; }

private:
  const Spec* spec_;
  MemoryElement* data_;
  [[no_unique_address]] Policy policy_;
  bool committed_ = false;
};

/**
 * @brief Cursor for scanning a one-dimensional logical sequence in vectors.
 *
 * Traversal axis and vector axis are both `Dim`. The cursor tracks an origin
 * coordinate plus a remaining logical element count; it never maintains an
 * independent raw pointer. Use `has_full()/load_full()/advance_full()` for
 * complete vectors and `load_tail()/advance_tail()` for the remainder.
 *
 * @code
 * auto c = x.scan(full_tag, origin, tensor::axis<1>, n);
 * while (c.has_full()) {
 *   auto v = c.load_full();
 *   c.advance_full();
 * }
 * while (!c.empty()) {
 *   auto v = c.load_tail(smaller_tail_tag); // optional different Tag
 *   c.advance_tail(smaller_tail_tag);
 * }
 * @endcode
 *
 * The default TailTag is the full Tag with `first(active)`. A manually smaller
 * TailTag consumes `min(remaining, size(tail_tag))`, enabling SVE paths whose
 *
 * @note A `load_tail()` and its `advance_tail()` must use the same TailTag.
 * @note Calling `load_full()` without `has_full()` or tail operations on an
 * empty cursor violates the cursor protocol.
 */
template <typename Access, typename Tag, int Dim, typename Mapping>
class ScanCursor {
public:
  static constexpr int Rank = Access::Rank;

  ScanCursor(
      Access& access,
      Tag tag,
      Coord<Rank> origin,
      nint_t count,
      Mapping mapping)
      : access_(&access), tag_(tag), origin_(origin), remaining_(count),
        mapping_(mapping) {}

  bool has_full() const { return remaining_ >= vec::size(tag_); }
  bool empty() const { return remaining_ <= 0; }
  nint_t remaining() const { return remaining_; }

  VECOPS_ALWAYS_INLINE auto load_full() const
    requires requires(const Access& access, Tag tag, Coord<Rank> origin) {
      access.load(tag, origin, axis<Dim>, vec::opt::unmasked);
    }
  {
    if constexpr (std::same_as<Mapping, ContiguousLaneMapping>) {
      return access_->load(tag_, origin_, axis<Dim>, vec::opt::unmasked);
    } else {
      return access_->load(
          tag_, origin_, axis<Dim>, vec::opt::unmasked,
          vec::strided(mapping_.stride));
    }
  }

  template <vec::VectorTag TailTag = Tag>
  VECOPS_ALWAYS_INLINE auto load_tail(TailTag tail_tag = {}) const
    requires requires(
        const Access& access, TailTag tag, Coord<Rank> origin) {
      access.load(tag, origin, axis<Dim>, vec::opt::first(1));
    }
  {
    const nint_t active = std::min(remaining_, vec::size(tail_tag));
    if constexpr (std::same_as<Mapping, ContiguousLaneMapping>) {
      return access_->load(
          tail_tag, origin_, axis<Dim>, vec::opt::first(active));
    } else {
      return access_->load(
          tail_tag, origin_, axis<Dim>, vec::opt::first(active),
          vec::strided(mapping_.stride));
    }
  }

  template <typename Value>
  VECOPS_ALWAYS_INLINE void store_full(Value value) const
    requires requires(const Access& access, Tag tag, Coord<Rank> origin) {
      access.store(tag, origin, axis<Dim>, value, vec::opt::unmasked);
    }
  {
    if constexpr (std::same_as<Mapping, ContiguousLaneMapping>) {
      access_->store(
          tag_, origin_, axis<Dim>, value, vec::opt::unmasked);
    } else {
      access_->store(
          tag_, origin_, axis<Dim>, value, vec::opt::unmasked,
          vec::strided(mapping_.stride));
    }
  }

  template <vec::VectorTag TailTag = Tag, typename Value>
  VECOPS_ALWAYS_INLINE void store_tail(
      TailTag tail_tag, Value value) const
    requires requires(
        const Access& access, TailTag tag, Coord<Rank> origin) {
      access.store(tag, origin, axis<Dim>, value, vec::opt::first(1));
    }
  {
    const nint_t active = std::min(remaining_, vec::size(tail_tag));
    if constexpr (std::same_as<Mapping, ContiguousLaneMapping>) {
      access_->store(
          tail_tag, origin_, axis<Dim>, value,
          vec::opt::first(active));
    } else {
      access_->store(
          tail_tag, origin_, axis<Dim>, value,
          vec::opt::first(active), vec::strided(mapping_.stride));
    }
  }

  void advance_full() {
    const nint_t lanes = vec::size(tag_);
    advance(lanes);
  }

  template <vec::VectorTag TailTag = Tag>
  void advance_tail(TailTag tail_tag = {}) {
    advance(std::min(remaining_, vec::size(tail_tag)));
  }

private:
  void advance(nint_t lanes) {
    origin_[Dim] += mapping_.offset(lanes);
    remaining_ -= lanes;
  }

  Access* access_;
  Tag tag_;
  Coord<Rank> origin_;
  nint_t remaining_;
  [[no_unique_address]] Mapping mapping_;
};

/**
 * @brief Cursor that advances one axis and vectors along another.
 *
 * Unlike ScanCursor, ProjectCursor has no full/tail policy: each operation
 * supplies its own Tag, active lanes, and lane addressing. `advance()` changes
 * only `TraverseDim` by `traversal_step`; vector width never changes traversal.
 *
 * @code
 * // Iteration i loads 32 logical elements starting at matrix coordinate (i,0).
 * auto c = x.project(tag, tensor::coord(0, 0),
 *                    tensor::traversal_axis<0>, tensor::vector_axis<1>, rows);
 * while (c.valid()) {
 *   auto row_prefix = c.load(vec::opt::first(32));
 *   c.advance();
 * }
 * @endcode
 *
 * `(i + D, 0)`, not `(i, D * vector_width)`.
 */
template <typename Access, typename Tag, int TraverseDim, int VectorDim>
class ProjectCursor {
public:
  static constexpr int Rank = Access::Rank;

  ProjectCursor(
      Access& access,
      Tag tag,
      Coord<Rank> origin,
      nint_t iterations,
      nint_t traversal_step)
      : access_(&access), tag_(tag), origin_(origin),
        remaining_iterations_(iterations), traversal_step_(traversal_step) {}

  bool valid() const { return remaining_iterations_ > 0; }

  template <typename... Options>
  VECOPS_ALWAYS_INLINE auto load(Options&&... options) const {
    return access_->load(
        tag_, origin_, axis<VectorDim>,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag OtherTag, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      OtherTag tag, Options&&... options) const {
    return access_->load(
        tag, origin_, axis<VectorDim>,
        std::forward<Options>(options)...);
  }

  template <typename Value, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Value value, Options&&... options) const
    requires requires(const Access& access, Coord<Rank> origin) {
      access.store(
          Tag{}, origin, axis<VectorDim>, value,
          std::forward<Options>(options)...);
    }
  {
    access_->store(
        tag_, origin_, axis<VectorDim>, value,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag OtherTag, typename Value, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      OtherTag tag, Value value, Options&&... options) const
    requires requires(const Access& access, Coord<Rank> origin) {
      access.store(
          tag, origin, axis<VectorDim>, value,
          std::forward<Options>(options)...);
    }
  {
    access_->store(
        tag, origin_, axis<VectorDim>, value,
        std::forward<Options>(options)...);
  }

  void advance() {
    origin_[TraverseDim] += traversal_step_;
    --remaining_iterations_;
  }

private:
  Access* access_;
  Tag tag_;
  Coord<Rank> origin_;
  nint_t remaining_iterations_;
  nint_t traversal_step_;
};

/**
 * @brief Non-owning DataAccess view produced by `slice_view`.
 *
 * A borrowed view owns only the sliced Spec descriptor and borrows the parent
 * session's Tensor/auxiliary storage. It forwards load/store/cursor operations
 * through a rebased policy, while its CoordinateProjection preserves original
 * transform coordinates.
 *
 * It intentionally has no `commit()`: slicing must not duplicate or transfer
 * write-back responsibility. The owning materialized output session must
 * outlive all borrowed views and remains the sole commit authority.
 */
template <typename Spec, typename Policy>
class BorrowedDataAccess {
public:
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  static constexpr bool IsInput = Spec::is_input;
  static constexpr int Rank = [] {
    if constexpr (IsInput) return Spec::InputTensor::Ndim;
    else return Spec::OutputTensor::Ndim;
  }();

  BorrowedDataAccess(Spec spec, Policy policy)
      : spec_(std::move(spec)), policy_(policy) {}

  template <vec::VectorTag Tag, typename... Options>
    requires IsInput
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Options&&... options) const {
    InputDataAccess<Spec, Policy> access{spec_, policy_};
    return access.load(
        tag, position, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
    requires IsInput
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      Options&&... options) const {
    InputDataAccess<Spec, Policy> access{spec_, policy_};
    return access.load(
        tag, position, dim, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, typename... Options>
    requires (!IsInput)
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, vec::Vec<Tag> value,
      Options&&... options) const {
    OutputDataAccess<Spec, Policy> access{spec_, policy_};
    access.store(
        tag, position, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
    requires (!IsInput)
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      vec::Vec<Tag> value, Options&&... options) const {
    OutputDataAccess<Spec, Policy> access{spec_, policy_};
    access.store(
        tag, position, dim, value,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim,
            typename Mapping = ContiguousLaneMapping>
    requires (std::same_as<Mapping, ContiguousLaneMapping> ||
              std::same_as<Mapping, AffineLaneMapping>)
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim>, nint_t count,
      Mapping mapping = {}) {
    return ScanCursor<BorrowedDataAccess, Tag, Dim, Mapping>{
        *this, tag, origin, count, mapping};
  }

  template <vec::VectorTag Tag, int Dim, typename Stride>
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim> dim, nint_t count,
      vec::opt::Strided<Stride> mapping) {
    return scan(
        tag, origin, dim, count,
        AffineLaneMapping{static_cast<nint_t>(mapping.stride)});
  }

  template <vec::VectorTag Tag, int TraverseDim, int VectorDim>
  VECOPS_INLINE auto project(
      Tag tag, Coord<Rank> origin, TraversalAxis<TraverseDim>,
      VectorAxis<VectorDim>, nint_t iterations,
      nint_t traversal_step = 1) {
    static_assert(TraverseDim != VectorDim);
    return ProjectCursor<
        BorrowedDataAccess, Tag, TraverseDim, VectorDim>{
            *this, tag, origin, iterations, traversal_step};
  }

  const Spec& spec() const { return spec_; }
  const Policy& policy() const { return policy_; }

private:
  Spec spec_;
  [[no_unique_address]] Policy policy_;
};

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const InputDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = SliceAccessPolicyT<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const OutputDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = SliceAccessPolicyT<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const BorrowedDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = SliceAccessPolicyT<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

/**
 * @brief Owning output session backed by a contiguous auxiliary Tensor.
 *
 * For `materialize_before_transform`, the hot loop stores ComputeType in the
 * auxiliary buffer; `commit()` applies the epilogue/conversion while copying to
 * the original Layout. For `materialize_after_transform`, the hot loop stores
 * transformed MemoryElement values; commit performs only logical Layout copy.
 *
 * The session is move-only. Moving transfers commit responsibility. `commit()`
 * is idempotent, but destruction of an uncommitted owning session triggers a
 * debug assertion and never performs implicit write-back.
 */
template <AccessPlan Plan, typename OriginalSpec, typename AuxSpec,
          typename Policy>
class MaterializedOutputDataAccess {
  static_assert(
      Plan == AccessPlan::materialize_before_transform ||
      Plan == AccessPlan::materialize_after_transform);

public:
  using ComputeType = typename OriginalSpec::ComputeType;
  using MemoryElement = typename OriginalSpec::MemoryElement;
  static constexpr int Rank = OriginalSpec::OutputTensor::Ndim;

  MaterializedOutputDataAccess(
      const OriginalSpec& original, AuxSpec auxiliary, Policy policy)
      : original_(&original), auxiliary_(std::move(auxiliary)),
        policy_(policy) {}

  MaterializedOutputDataAccess(const MaterializedOutputDataAccess&) = delete;
  MaterializedOutputDataAccess& operator=(const MaterializedOutputDataAccess&) = delete;

  MaterializedOutputDataAccess(MaterializedOutputDataAccess&& other) noexcept
      : original_(other.original_), auxiliary_(std::move(other.auxiliary_)),
        policy_(other.policy_), committed_(other.committed_), owning_(true) {
    other.owning_ = false;
  }

  MaterializedOutputDataAccess& operator=(MaterializedOutputDataAccess&&) = delete;

  ~MaterializedOutputDataAccess() {
    VECOPS_ASSERT(!owning_ || committed_,
                  "materialized output session was destroyed without commit()");
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, vec::Vec<Tag> value,
      Options&&... options) const {
    OutputDataAccess<AuxSpec, Policy> hot{auxiliary_, policy_};
    hot.store(tag, position, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      vec::Vec<Tag> value, Options&&... options) const {
    OutputDataAccess<AuxSpec, Policy> hot{auxiliary_, policy_};
    hot.store(
        tag, position, dim, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim,
            typename Mapping = ContiguousLaneMapping>
    requires (std::same_as<Mapping, ContiguousLaneMapping> ||
              std::same_as<Mapping, AffineLaneMapping>)
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim>, nint_t count,
      Mapping mapping = {}) {
    return ScanCursor<MaterializedOutputDataAccess, Tag, Dim, Mapping>{
        *this, tag, origin, count, mapping};
  }

  template <vec::VectorTag Tag, int Dim, typename Stride>
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim> dim, nint_t count,
      vec::opt::Strided<Stride> mapping) {
    return scan(
        tag, origin, dim, count,
        AffineLaneMapping{static_cast<nint_t>(mapping.stride)});
  }

  template <vec::VectorTag Tag, int TraverseDim, int VectorDim>
  VECOPS_INLINE auto project(
      Tag tag, Coord<Rank> origin, TraversalAxis<TraverseDim>,
      VectorAxis<VectorDim>, nint_t iterations,
      nint_t traversal_step = 1) {
    static_assert(TraverseDim != VectorDim);
    return ProjectCursor<MaterializedOutputDataAccess, Tag, TraverseDim,
                         VectorDim>{
        *this, tag, origin, iterations, traversal_step};
  }

  /**
   * @brief Write the complete auxiliary region back to the original Tensor.
   * @note Idempotent. Must run before `with_operands` rewinds workspace.
   */
  void commit() {
    if (committed_) return;
    if constexpr (Plan == AccessPlan::materialize_before_transform) {
      commit_before();
    } else {
      commit_after();
    }
    committed_ = true;
  }

  bool committed() const { return committed_; }

  const AuxSpec& auxiliary_spec() const { return auxiliary_; }
  const AuxSpec& spec() const { return auxiliary_; }
  const Policy& policy() const { return policy_; }

private:
  void commit_before() {
    constexpr int AxisValue = Policy::vector_axis;
    using Tag = vec::ScalableTag<ComputeType, 0>;
    using AuxInputPolicy = InputAccessPolicy<AxisValue, 1, AccessPlan::direct>;
    auto aux_input_spec = input<ComputeType>(auxiliary_.tensor());
    InputDataAccess<decltype(aux_input_spec), AuxInputPolicy> source{
        aux_input_spec};
    OutputDataAccess<OriginalSpec, Policy> destination{*original_, policy_};
    Coord<Rank> position{};
    details::for_each_line<AxisValue>(
        original_->output_layout(), position,
        [&](Coord<Rank>& line) VECOPS_INLINE_LAMBDA {
          Tag tag{};
          const nint_t count = original_->output_layout().shape()[AxisValue];
          for (nint_t offset = 0; offset < count; offset += vec::size(tag)) {
            line[AxisValue] = offset;
            const nint_t active = std::min(count - offset, vec::size(tag));
            auto value = source.load(
                tag, line, axis<AxisValue>, vec::opt::first(active));
            destination.store(
                tag, line, axis<AxisValue>, value, vec::opt::first(active));
          }
        });
    destination.commit();
  }

  void commit_after() {
    Coord<Rank> position{};
    const auto& source = auxiliary_.tensor();
    auto& destination = original_->tensor();
    details::for_each_coordinate(
        original_->output_layout(), position,
        [&](const Coord<Rank>& current) VECOPS_INLINE_LAMBDA {
          destination.data()[offset_at(destination.layout(), current)] =
              source.data()[offset_at(source.layout(), current)];
        });
  }

  const OriginalSpec* original_;
  AuxSpec auxiliary_;
  [[no_unique_address]] Policy policy_;
  bool committed_ = false;
  bool owning_ = true;
};

/**
 * @brief Slice a materialized output as a borrowed, non-committing view.
 * @note The parent session still must be committed after all slice use.
 */
template <int Dim, AccessPlan Plan, typename OriginalSpec, typename AuxSpec,
          typename Policy>
VECOPS_INLINE auto slice_view(
    const MaterializedOutputDataAccess<
        Plan, OriginalSpec, AuxSpec, Policy>& access,
    nint_t index) {
  auto sliced = slice_view<Dim>(access.auxiliary_spec(), index);
  using SlicedPolicy = SliceAccessPolicyT<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

/**
 * @brief Return workspace bytes required by the resolved Spec/Policy plan.
 *
 * The result includes element-size choice and alignment padding. It is zero for
 * direct plans and readless inputs. The same Spec, runtime Layout, and Policy
 * must subsequently be passed to binding; a layout-only estimate is not valid.
 */
template <typename Spec, typename Policy>
VECOPS_INLINE nint_t required_workspace(const Spec& spec, Policy policy) {
  constexpr AccessPlan plan =
      details::resolve_plan<Spec, Policy>();
  if constexpr (plan == AccessPlan::direct) return 0;
  if constexpr (is_input_spec_v<Spec>) {
    if constexpr (!details::transform_reads_input<typename Spec::TransformType>) {
      return 0;
    }
  }
  const nint_t element_bytes = [&] {
    if constexpr (plan == AccessPlan::materialize_before_transform) {
      if constexpr (is_input_spec_v<Spec>) {
        return static_cast<nint_t>(sizeof(typename Spec::MemoryElement));
      } else {
        return static_cast<nint_t>(sizeof(typename Spec::ComputeType));
      }
    } else {
      if constexpr (is_input_spec_v<Spec>) {
        return static_cast<nint_t>(sizeof(typename Spec::ComputeType));
      } else {
        return static_cast<nint_t>(sizeof(typename Spec::MemoryElement));
      }
    }
  }();
  const nint_t elements = [&] {
    if constexpr (is_input_spec_v<Spec>) {
      return details::tensor_numel(spec.input_layout());
    } else {
      return details::tensor_numel(spec.output_layout());
    }
  }();
  return kernel::details::workspace_round_up(
      elements * element_bytes,
      vec::DEFAULT_ALIGNMENT);
}

/**
 * @brief Bind a statically direct operand without materialization dispatch.
 *
 * This low-level helper asserts that the resolved plan is direct. General
 * kernels should use `kernel::with_operands`, which handles dynamic plans,
 * workspace lifetime, input preparation, and output ownership.
 */
template <typename Spec, typename Policy>
VECOPS_INLINE auto bind(const Spec& spec, Policy policy, kernel::WorkspaceView&) {
  VECOPS_ASSERT(
      (details::resolve_plan<Spec, Policy>() == AccessPlan::direct),
                "tensor::bind only binds direct plans; use kernel::with_operands");
  if constexpr (is_input_spec_v<Spec>) {
    return InputDataAccess<Spec, Policy>{spec, policy};
  } else {
    static_assert(is_output_spec_v<Spec>);
    return OutputDataAccess<Spec, Policy>{spec, policy};
  }
}

/** @brief Pair a caller-owned Spec reference with a kernel-owned Policy. */
template <typename Spec, typename Policy>
struct OperandBinding {
  using SpecType = Spec;
  using PolicyType = Policy;

  const Spec& spec;
  [[no_unique_address]] Policy policy;
};

template <typename T>
struct IsOperandBinding : std::false_type {};

template <typename Spec, typename Policy>
struct IsOperandBinding<OperandBinding<Spec, Policy>> : std::true_type {};

template <typename T>
inline constexpr bool is_operand_binding_v =
    IsOperandBinding<std::remove_cvref_t<T>>::value;

/**
 * @brief Create an operand binding consumed by `kernel::with_operands`.
 * @note The referenced Spec must outlive the `with_operands` call.
 */
template <typename Spec, typename Policy>
VECOPS_INLINE auto operand(const Spec& spec, Policy policy) {
  return OperandBinding<Spec, Policy>{spec, policy};
}

namespace details {

template <typename Spec, typename Policy, typename Fn>
VECOPS_INLINE decltype(auto) with_bound_input(
    const Spec& spec, Policy policy, kernel::WorkspaceView& workspace,
    Fn&& fn) {
  constexpr int AxisValue = Policy::vector_axis;
  constexpr AccessPlan plan = resolve_plan<Spec, Policy>();
  if constexpr (plan == AccessPlan::direct) {
    InputDataAccess<Spec, Policy> access{spec, policy};
    return std::forward<Fn>(fn)(access);
  } else {

  const nint_t count = tensor_numel(spec.input_layout());
  auto aux_layout = auxiliary_layout<AxisValue>(spec.input_layout());
  if constexpr (plan == AccessPlan::materialize_before_transform) {
    using Memory = typename Spec::MemoryElement;
    Memory* buffer = workspace.allocate<Memory>(count);
    auto auxiliary_tensor = make_tensor(buffer, aux_layout);
    Coord<Spec::InputTensor::Ndim> position{};
    for_each_coordinate(
        spec.input_layout(), position,
        [&](const auto& current) VECOPS_INLINE_LAMBDA {
          buffer[offset_at(aux_layout, current)] =
              spec.tensor().data()[offset_at(spec.input_layout(), current)];
        });
    using AuxTensor = decltype(auxiliary_tensor);
    using AuxSpec = InputSpec<
        typename Spec::ComputeType, AuxTensor, typename Spec::TransformType,
        typename Spec::ProjectionType>;
    AuxSpec aux_spec{
        auxiliary_tensor, spec.transform(), spec.projection()};
    InputDataAccess<AuxSpec, Policy> access{aux_spec, policy};
    return std::forward<Fn>(fn)(access);
  }

  using Compute = typename Spec::ComputeType;
  using Tag = vec::ScalableTag<Compute, 0>;
  Compute* buffer = workspace.allocate<Compute>(count);
  auto auxiliary_tensor = make_tensor(buffer, aux_layout);
  InputDataAccess<Spec, Policy> source{spec, policy};
  Coord<Spec::InputTensor::Ndim> position{};
  for_each_line<AxisValue>(
      spec.input_layout(), position,
      [&](auto& line) VECOPS_INLINE_LAMBDA {
        Tag tag{};
        const nint_t line_size = spec.input_layout().shape()[AxisValue];
        for (nint_t offset = 0; offset < line_size;
             offset += vec::size(tag)) {
          line[AxisValue] = offset;
          const nint_t active =
              std::min(line_size - offset, vec::size(tag));
          auto value = source.load(
              tag, line, axis<AxisValue>, vec::opt::first(active));
          vec::store(
              tag, buffer + offset_at(aux_layout, line), value,
              vec::opt::first(active));
        }
      });
  auto aux_spec = input<Compute>(auxiliary_tensor);
  using AuxPolicy = InputAccessPolicy<AxisValue, Policy::read_passes,
                                      AccessPlan::direct>;
  InputDataAccess<decltype(aux_spec), AuxPolicy> access{aux_spec};
  return std::forward<Fn>(fn)(access);
  }
}

template <typename Spec, typename Policy, typename Fn>
VECOPS_INLINE decltype(auto) with_bound_output(
    const Spec& spec, Policy policy, kernel::WorkspaceView& workspace,
    Fn&& fn) {
  constexpr int AxisValue = Policy::vector_axis;
  constexpr AccessPlan plan = resolve_plan<Spec, Policy>();
  if constexpr (plan == AccessPlan::direct) {
    OutputDataAccess<Spec, Policy> access{spec, policy};
    return std::forward<Fn>(fn)(access);
  } else {

  const nint_t count = tensor_numel(spec.output_layout());
  auto aux_layout = auxiliary_layout<AxisValue>(spec.output_layout());
  if constexpr (plan == AccessPlan::materialize_before_transform) {
    using Compute = typename Spec::ComputeType;
    Compute* buffer = workspace.allocate<Compute>(count);
    auto auxiliary_tensor = make_tensor(buffer, aux_layout);
    auto aux_spec = output<Compute>(auxiliary_tensor);
    using Access = MaterializedOutputDataAccess<
        AccessPlan::materialize_before_transform, Spec,
        decltype(aux_spec), Policy>;
    Access access{spec, std::move(aux_spec), policy};
    return std::forward<Fn>(fn)(access);
  }

  using Memory = typename Spec::MemoryElement;
  Memory* buffer = workspace.allocate<Memory>(count);
  auto auxiliary_tensor = make_tensor(buffer, aux_layout);
  using AuxTensor = decltype(auxiliary_tensor);
  using AuxSpec = OutputSpec<
      typename Spec::ComputeType, AuxTensor, typename Spec::TransformType,
      typename Spec::ProjectionType>;
  AuxSpec aux_spec{
      auxiliary_tensor, spec.transform(), spec.projection()};
  using Access = MaterializedOutputDataAccess<
      AccessPlan::materialize_after_transform, Spec, AuxSpec, Policy>;
  Access access{spec, std::move(aux_spec), policy};
  return std::forward<Fn>(fn)(access);
  }
}

template <typename Spec, typename Policy, typename Fn>
VECOPS_INLINE decltype(auto) with_bound_operand(
    const OperandBinding<Spec, Policy>& binding,
    kernel::WorkspaceView& workspace, Fn&& fn) {
  if constexpr (is_input_spec_v<Spec>) {
    return with_bound_input(
        binding.spec, binding.policy, workspace, std::forward<Fn>(fn));
  } else {
    return with_bound_output(
        binding.spec, binding.policy, workspace, std::forward<Fn>(fn));
  }
}

} // namespace details

} // namespace vecops::tensor

namespace vecops::kernel {

namespace details {

template <std::size_t I, typename Tuple, typename BoundTuple, typename Fn>
VECOPS_INLINE decltype(auto) bind_operands_recursive(
    WorkspaceView& workspace,
    const Tuple& bindings,
    BoundTuple&& bound,
    Fn&& fn) {
  if constexpr (I == std::tuple_size_v<Tuple>) {
    return std::apply(std::forward<Fn>(fn), std::forward<BoundTuple>(bound));
  } else {
    const auto& binding = std::get<I>(bindings);
    return tensor::details::with_bound_operand(
        binding, workspace,
        [&](auto& access) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return bind_operands_recursive<I + 1>(
              workspace, bindings,
              std::tuple_cat(
                  std::forward<BoundTuple>(bound),
                  std::forward_as_tuple(access)),
              std::forward<Fn>(fn));
        });
  }
}

} // namespace details

/**
 * @brief Resolve, materialize, bind, and scope a tuple of operands.
 *
 * The callback runs after every input preparation and receives lvalue
 * references to statically typed DataAccess sessions. Output sessions owning
 * auxiliary storage must be committed inside the callback. After callback
 * return, sessions are destroyed and the workspace is rewound to its entry
 * mark, including when the callback returns a non-void result.
 *
 * All bindings in one unordered permutation region must use compatible
 * unordered policies; mixing ordered and unordered operands is rejected at
 * compile time.
 */
template <typename... Bindings, typename Fn>
VECOPS_INLINE decltype(auto) with_operand_tuple(
    WorkspaceView& workspace,
    const std::tuple<Bindings...>& bindings,
    Fn&& fn) {
  constexpr bool any_unordered =
      (tensor::details::is_unordered_policy<
           typename Bindings::PolicyType> || ...);
  constexpr bool all_unordered =
      (tensor::details::is_unordered_policy<
           typename Bindings::PolicyType> && ...);
  static_assert(
      !any_unordered || all_unordered,
      "all operands participating in a permutation-safe region must use "
      "compatible unordered lane-order policies");
  const auto mark = workspace.mark();
  if constexpr (std::is_void_v<decltype(details::bind_operands_recursive<0>(
                    workspace, bindings, std::tuple<>{},
                    std::forward<Fn>(fn)))>) {
    details::bind_operands_recursive<0>(
        workspace, bindings, std::tuple<>{}, std::forward<Fn>(fn));
    workspace.rewind(mark);
  } else {
    auto result = details::bind_operands_recursive<0>(
        workspace, bindings, std::tuple<>{}, std::forward<Fn>(fn));
    workspace.rewind(mark);
    return result;
  }
}

/**
 * @brief Convenience overload binding one to four operands.
 *
 * Semantics and lifetime are identical to `with_operand_tuple`: plans resolve
 * outside the callback, materialized output must be committed inside it, and
 * workspace rewinds after all sessions are destroyed.
 */
template <typename Binding, typename Fn>
  requires tensor::is_operand_binding_v<Binding>
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, Binding&& binding, Fn&& fn) {
  return with_operand_tuple(
      workspace,
      std::tuple<std::remove_cvref_t<Binding>>{
          std::forward<Binding>(binding)},
      std::forward<Fn>(fn));
}

template <typename B0, typename B1, typename Fn>
  requires (tensor::is_operand_binding_v<B0> &&
            tensor::is_operand_binding_v<B1>)
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, Fn&& fn) {
  return with_operand_tuple(
      workspace,
      std::tuple<std::remove_cvref_t<B0>, std::remove_cvref_t<B1>>{
          std::forward<B0>(b0), std::forward<B1>(b1)},
      std::forward<Fn>(fn));
}

template <typename B0, typename B1, typename B2, typename Fn>
  requires (tensor::is_operand_binding_v<B0> &&
            tensor::is_operand_binding_v<B1> &&
            tensor::is_operand_binding_v<B2>)
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, B2&& b2, Fn&& fn) {
  return with_operand_tuple(
      workspace,
      std::tuple<std::remove_cvref_t<B0>, std::remove_cvref_t<B1>,
                 std::remove_cvref_t<B2>>{
          std::forward<B0>(b0), std::forward<B1>(b1),
          std::forward<B2>(b2)},
      std::forward<Fn>(fn));
}

template <typename B0, typename B1, typename B2, typename B3, typename Fn>
  requires (tensor::is_operand_binding_v<B0> &&
            tensor::is_operand_binding_v<B1> &&
            tensor::is_operand_binding_v<B2> &&
            tensor::is_operand_binding_v<B3>)
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, B2&& b2, B3&& b3,
    Fn&& fn) {
  return with_operand_tuple(
      workspace,
      std::tuple<std::remove_cvref_t<B0>, std::remove_cvref_t<B1>,
                 std::remove_cvref_t<B2>, std::remove_cvref_t<B3>>{
          std::forward<B0>(b0), std::forward<B1>(b1),
          std::forward<B2>(b2), std::forward<B3>(b3)},
      std::forward<Fn>(fn));
}

} // namespace vecops::kernel

#endif // VECOPS_TENSOR_DATA_ACCESS_H
