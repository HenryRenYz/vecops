#ifndef VECOPS_TENSOR_DATA_ACCESS_H
#define VECOPS_TENSOR_DATA_ACCESS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/Transpose2D.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/AccessPolicy.h"
#include "vecops/tensor/OptionalOperand.h"
#include "vecops/tensor/Tensor.h"
#include "vecops/tensor/Transform.h"
#include "vecops/util/ScalarConvert.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/Options.h"

/**
 * @file DataAccess.h
 * @brief Policy-driven vector and scalar access to logical N-dimensional Tensors.
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
 * auto x_pattern = tensor::unbind(x_spec); // same contract, no address
 * auto x_tile = tensor::rebind(x_pattern, actual_x_tile);
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
 * materialized sessions, and verifies output commit ownership in debug builds.
 * Groups containing a materialized operand are scoped under a workspace mark
 * that is rewound after the sessions are destroyed. An all-direct group does
 * not touch workspace state. The callback must not directly mutate the
 * `WorkspaceView` passed to `with_operands`; workspace authority belongs to
 * the bound operand sessions for the duration of the callback.
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
 * ## Scalar point access
 *
 * `load_scalar(position)` and `store_scalar(position, value)` address exactly
 * one logical element through Layout and perform direct scalar pointer traffic.
 * They apply the same saturating or wrapping dtype policy as vector conversion,
 * but accept no lane, addressing, inactive-population, or vector-memory
 * options. Scalar access is available only for `NoTransform`,
 * `IdentityVecTransform`, and `ZeroVecTransform`; arbitrary vector transforms
 * remain vector-only. Identity preserves both of its boundary conversions,
 * while a zero input does not read Tensor memory. Conversion order is accepted
 * for API uniformity but cannot permute a single value.
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
 * Conversion order/value and memory hints are access options. Stateless
 * defaults may instead be attached at `operand(spec, policy,
 * access_defaults(...))`; eager preparation and deferred commit use those
 * defaults because they execute outside a hot-loop access call.
 *
 * ## Deferred materialization and unordered regions
 *
 * `AccessPlan::automatic_deferred` follows automatic input planning but leaves
 * an after-transform Compute buffer uninitialized. Loads marked with
 * `tensor::materialize::populate` read the source and fill the canonical
 * buffer; subsequent unmarked loads reuse it. The kernel promises a complete
 * populate pass before reuse. Debug builds reject reuse-before-populate and
 * populate-after-reuse, but do not maintain an element coverage bitmap.
 *
 * `tensor::with_unordered_access(args..., fn)` means unordered conversion is
 * permitted, not guaranteed. It inspects every bound access and invokes `fn`
 * with either all-unordered or all-ordered proxies. Any canonical materialized
 * Compute buffer, incompatible storage width, or non-equivariant transform
 * makes the entire region ordered.
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
 * - Bound Specs and DataAccess are non-owning with respect to Tensor memory.
 *   Unbound Specs are address-free planning patterns. `required_workspace`
 *   accepts either; DataAccess construction accepts only rebound Specs. An
 *   unbound Spec retains a transform's type but not its value. Stateful
 *   transforms therefore rebind through an actual bound Spec.
 * - Output Specs require a mutable Tensor. No `const_cast` is performed.
 * - Materialized output sessions are move-only and must be explicitly
 *   `commit()`ed. Destructors do not write back; they assert on an omitted
 *   commit in debug builds. Borrowed slices never expose commit authority.
 * - Transforms must be deterministic and side-effect-free because planning may
 *   move their execution or split one request into several calls.
 * - Tensor-level `indexed` rejects explicit byte scale. Indices are logical
 *   offsets and are scaled by Layout exactly once.
 * - Unordered conversion is an operation-region lane-order contract, not a
 *   local speed flag. Use `with_unordered_access` for related operands.
 * - `required_workspace()` may use an unbound Spec. Execution must rebind an
 *   actual Tensor/Spec with the same typed Layout and runtime shape/stride.
 * - Layout-transpose materialization uses `kernel::transpose2d_bound` only when
 *   a non-vector axis is provably unit-stride from its meta type. A runtime
 *   stride equal to one does not activate a hidden fast-path branch.
 */
namespace vecops::tensor {

/**
 * Number of tag doublings preferred by an access in order to make its memory
 * element side occupy at least one complete native vector word.
 *
 * A kernel normally owns one native word of ComputeType.  Rebinding that tag
 * to a narrower MemoryElement preserves the logical lane count and therefore
 * describes only a subword memory access.  When the surrounding accumulator
 * or packing shape owns adjacent vectors, grouping 2^power of them restores a
 * complete memory access.  Wider memory elements already consume one or more
 * complete words for a single compute vector, so they do not request growth.
 *
 * This trait deliberately ignores a transform's declared output type: the
 * actual memory boundary is always MemoryElement.
 */
template <typename Access>
inline constexpr int preferred_memory_access_power_v = []() consteval {
  using Compute = typename std::remove_cvref_t<Access>::ComputeType;
  using Memory = typename std::remove_cvref_t<Access>::MemoryElement;
  if constexpr (sizeof(Compute) <= sizeof(Memory)) {
    return 0;
  } else {
    static_assert(sizeof(Compute) % sizeof(Memory) == 0);
    std::size_t ratio = sizeof(Compute) / sizeof(Memory);
    int power = 0;
    while (ratio > 1) {
      static_assert((sizeof(Compute) / sizeof(Memory) & (sizeof(Compute) / sizeof(Memory) - 1)) == 0,
                    "vector element-size ratio must be a power of two");
      ratio /= 2;
      ++power;
    }
    return power;
  }
}();
namespace materialize {

/**
 * Marks a load as the population pass of an automatic-deferred input.
 * Direct and eagerly materialized inputs accept the option as a no-op.
 */
struct Populate {};
inline constexpr Populate populate{};

} // namespace materialize

namespace details {

template <typename T>
struct IsMaterializePopulate : std::false_type {};
template <>
struct IsMaterializePopulate<materialize::Populate> : std::true_type {};

template <template <typename> typename Predicate, typename Default,
          typename... Options>
struct OptionTypeOr {
  using type = Default;
};

template <template <typename> typename Predicate, typename Default,
          typename First, typename... Rest>
struct OptionTypeOr<Predicate, Default, First, Rest...> {
  using Clean = std::remove_cvref_t<First>;
  using type = std::conditional_t<
      Predicate<Clean>::value, Clean,
      typename OptionTypeOr<Predicate, Default, Rest...>::type>;
};

template <template <typename> typename Predicate, typename Default,
          typename... Options>
using option_type_or_t = typename OptionTypeOr<
    Predicate, Default, Options...>::type;

template <typename T>
struct IsConversionOrderOption : std::bool_constant<
    std::same_as<T, vec::cvt::Ordered> ||
    std::same_as<T, vec::cvt::Unordered>> {};

template <typename T>
struct IsConversionValueOption : std::bool_constant<
    vec::details::IsSaturateOption<T>::value ||
    vec::details::IsWrapOption<T>::value> {};

template <typename T>
using IsTemporalityOption = vec::details::IsMemoryTemporalityOption<T>;

template <typename T>
using IsPackingOption = vec::details::IsConversionMemoryPackingOption<T>;

template <typename T>
using IsAlignmentOption = vec::details::IsMemoryAlignmentOption<T>;

template <typename... Options>
consteval bool valid_access_default_options() {
  constexpr auto allowed = []<typename T>() {
    return IsConversionOrderOption<T>::value ||
        IsConversionValueOption<T>::value ||
        IsTemporalityOption<T>::value || IsPackingOption<T>::value ||
        IsAlignmentOption<T>::value;
  };
  return (allowed.template operator()<std::remove_cvref_t<Options>>() && ...) &&
      vec::details::option_count_v<IsConversionOrderOption, Options...> <= 1 &&
      vec::details::option_count_v<IsConversionValueOption, Options...> <= 1 &&
      vec::details::option_count_v<IsTemporalityOption, Options...> <= 1 &&
      vec::details::option_count_v<IsPackingOption, Options...> <= 1 &&
      vec::details::option_count_v<IsAlignmentOption, Options...> <= 1;
}

template <typename Defaults, typename Order>
using ReorderAccessDefaults = AccessDefaults<
    Order, typename Defaults::ConversionValueOption,
    typename Defaults::TemporalityOption, typename Defaults::PackingOption,
    typename Defaults::AlignmentOption>;

template <typename... Options>
using MakeAccessDefaults = AccessDefaults<
    option_type_or_t<IsConversionOrderOption, vec::cvt::Ordered, Options...>,
    option_type_or_t<IsConversionValueOption, vec::cvt::Saturate, Options...>,
    option_type_or_t<IsTemporalityOption, vec::mem::Temporal, Options...>,
    option_type_or_t<IsPackingOption, vec::mem::Packed, Options...>,
    option_type_or_t<IsAlignmentOption, vec::mem::Unaligned, Options...>>;

template <typename Base, typename... Options>
using OverrideAccessDefaults = AccessDefaults<
    option_type_or_t<IsConversionOrderOption,
                  typename Base::ConversionOrderOption, Options...>,
    option_type_or_t<IsConversionValueOption,
                  typename Base::ConversionValueOption, Options...>,
    option_type_or_t<IsTemporalityOption,
                  typename Base::TemporalityOption, Options...>,
    option_type_or_t<IsPackingOption,
                  typename Base::PackingOption, Options...>,
    option_type_or_t<IsAlignmentOption,
                  typename Base::AlignmentOption, Options...>>;

template <typename... Options>
inline constexpr bool has_access_default_option_v =
    ((IsConversionOrderOption<std::remove_cvref_t<Options>>::value ||
      IsConversionValueOption<std::remove_cvref_t<Options>>::value ||
      IsTemporalityOption<std::remove_cvref_t<Options>>::value ||
      IsPackingOption<std::remove_cvref_t<Options>>::value ||
      IsAlignmentOption<std::remove_cvref_t<Options>>::value) || ...);

template <typename Defaults>
struct AccessMemoryDefaults {
  using TemporalityOption = typename Defaults::TemporalityOption;
  using PackingOption = typename Defaults::PackingOption;
  using AlignmentOption = typename Defaults::AlignmentOption;
};

/** Combines a lifetime policy with call-site defaults for existing lowering. */
template <typename PlanningPolicy, typename Defaults,
          typename Resources = execution::details::ResourceSet<>>
struct AccessLoweringPolicy : PlanningPolicy {
  using PlanningPolicyType = PlanningPolicy;
  using AccessDefaultsType = Defaults;
  using ConversionOrderOption = typename Defaults::ConversionOrderOption;
  using ConversionValueOption = typename Defaults::ConversionValueOption;
  using MemoryOptions = AccessMemoryDefaults<Defaults>;
  using ActiveResources = Resources;
  static constexpr bool permutation_safe = std::same_as<
      typename Defaults::ConversionOrderOption, vec::cvt::Unordered>;
};

template <typename T>
inline constexpr bool is_unordered_policy_v = std::same_as<
    typename std::remove_cvref_t<T>::ConversionOrderOption,
    vec::cvt::Unordered>;

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
  if constexpr (BoundTensorLike<Tensor>) {
    VECOPS_ASSERT(
        reinterpret_cast<std::uintptr_t>(tensor.data()) % Alignment == 0,
        "tensor base pointer does not satisfy its declared alignment");
  }
}

/** True when the meta stride type is provably the value 1 at compile time. */
template <typename T>
inline constexpr bool is_definitely_one_meta_v =
    meta::is_singleton_v<std::remove_cvref_t<T>> &&
    meta::singleton_value_v<std::remove_cvref_t<T>> == 1;

template <bool Load, typename... Options>
VECOPS_ALWAYS_INLINE constexpr void validate_access_options() {
  constexpr std::size_t active =
      vec::details::option_count_v<vec::details::IsUnmaskedOption, Options...> +
      vec::details::option_count_v<vec::details::IsFirstOption, Options...> +
      vec::details::option_count_v<vec::details::IsMaskedOption, Options...>;
  constexpr std::size_t addressing =
      vec::details::option_count_v<vec::details::IsStridedOption, Options...> +
      vec::details::option_count_v<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t population =
      vec::details::option_count_v<vec::details::IsZeroOption, Options...> +
      vec::details::option_count_v<vec::details::IsVectorMergeOption, Options...> +
      vec::details::option_count_v<vec::details::IsScalarMergeOption, Options...>;
  constexpr std::size_t resources =
      vec::details::option_count_v<vec::details::IsResourcesOption, Options...>;
  static_assert(active <= 1, "at most one tensor active option is allowed");
  static_assert(addressing <= 1,
                "at most one tensor lane-addressing option is allowed");
  static_assert(population <= 1,
                "at most one inactive-population option is allowed");
  static_assert(Load || population == 0,
                "inactive-population options are only valid for tensor loads");
  static_assert(resources == 0,
                "tensor access resources come from the execution scope");
  static_assert(([]<typename Option>() {
    using Clean = std::remove_cvref_t<Option>;
    if constexpr (vec::details::IsStridedOption<Clean>::value)
      return vec::details::IsStridedOption<Clean>::scale == 0;
    else
      return true;
  }.template operator()<Options>() && ...),
                "tensor strided addressing accepts only element strides");
}

/** Validate the deliberately small option surface of a scalar point access. */
template <bool Load, typename... Options>
VECOPS_ALWAYS_INLINE constexpr void validate_scalar_access_options() {
  constexpr auto allowed = []<typename T>() {
    return IsConversionOrderOption<T>::value ||
        IsConversionValueOption<T>::value ||
        (Load && IsMaterializePopulate<T>::value);
  };
  static_assert(
      (allowed.template operator()<std::remove_cvref_t<Options>>() && ...),
      "tensor scalar access accepts only conversion options and, for loads, "
      "materialize::populate");
  static_assert(
      vec::details::option_count_v<IsConversionOrderOption, Options...> <= 1,
      "at most one scalar conversion-order option is allowed");
  static_assert(
      vec::details::option_count_v<IsConversionValueOption, Options...> <= 1,
      "at most one scalar conversion-value option is allowed");
  static_assert(
      vec::details::option_count_v<IsMaterializePopulate, Options...> <=
          (Load ? 1 : 0),
      "materialize::populate is only valid for scalar loads");
}

template <typename T>
inline constexpr bool is_no_transform_v =
    std::same_as<std::remove_cvref_t<T>, NoTransform>;

template <typename T>
struct IsIdentityVecTransform : std::false_type {};

template <typename EOut, typename EIn>
struct IsIdentityVecTransform<IdentityVecTransform<EOut, EIn>> : std::true_type {};

template <TransformStoreMode Mode, typename Inner>
struct IsIdentityVecTransform<TransformStoreOverride<Mode, Inner>> : IsIdentityVecTransform<Inner> {};

template <typename T>
inline constexpr bool is_identity_vec_transform_v = IsIdentityVecTransform<std::remove_cvref_t<T>>::value;

/** Scalar access supports only transforms with an exact scalar definition. */
template <typename T>
inline constexpr bool scalar_access_transform_v =
    is_no_transform_v<T> || is_identity_vec_transform_v<T> ||
    is_zero_vec_transform_v<std::remove_cvref_t<T>>;

template <typename Transform>
inline constexpr bool transform_permutation_equivariant_v = [] {
  if constexpr (is_no_transform_v<Transform>) return true;
  else return Transform::permutation_equivariant;
}();

template <typename Transform>
inline constexpr bool transform_reads_input_v = [] {
  if constexpr (is_no_transform_v<Transform>) return true;
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

/** Tuple of the options that do NOT match @p Drop; order and value category
 *  are preserved. */
template <template <typename> typename Drop, typename... Options>
VECOPS_ALWAYS_INLINE auto drop_options_if(Options&&... options) {
  return std::tuple_cat(
      []<typename Option>(Option&& option) {
        using Clean = std::remove_cvref_t<Option>;
        if constexpr (Drop<Clean>::value) {
          return std::tuple<>{};
        } else {
          return std::forward_as_tuple(std::forward<Option>(option));
        }
      }(std::forward<Options>(options))...);
}

template <typename T>
struct IsAddressOption : std::bool_constant<
    vec::details::IsIndexedOption<T>::value ||
    vec::details::IsStridedOption<T>::value> {};

template <typename T>
struct IsCacheStoreStrippedOption : std::bool_constant<
    vec::details::IsZeroOption<T>::value ||
    vec::details::IsVectorMergeOption<T>::value ||
    vec::details::IsScalarMergeOption<T>::value ||
    IsConversionOrderOption<T>::value ||
    IsConversionValueOption<T>::value ||
    IsTemporalityOption<T>::value || IsPackingOption<T>::value ||
    IsAlignmentOption<T>::value> {};

template <typename T>
struct IsAccessDefaultOwnedOption : std::bool_constant<
    IsConversionOrderOption<T>::value ||
    IsConversionValueOption<T>::value ||
    IsTemporalityOption<T>::value || IsPackingOption<T>::value ||
    IsAlignmentOption<T>::value> {};

template <typename... Options>
VECOPS_ALWAYS_INLINE auto non_address_options(Options&&... options) {
  return drop_options_if<IsAddressOption>(
      std::forward<Options>(options)...);
}

template <typename... Options>
VECOPS_ALWAYS_INLINE auto non_materialize_options(Options&&... options) {
  return drop_options_if<IsMaterializePopulate>(
      std::forward<Options>(options)...);
}

template <typename... Options>
VECOPS_ALWAYS_INLINE auto cache_store_options(Options&&... options) {
  return drop_options_if<IsCacheStoreStrippedOption>(
      std::forward<Options>(options)...);
}

template <typename... Options>
VECOPS_ALWAYS_INLINE auto non_access_default_options(Options&&... options) {
  return drop_options_if<IsAccessDefaultOwnedOption>(
      std::forward<Options>(options)...);
}

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE auto make_logical_mapping(
    Tag, Options&&... options) {
  constexpr std::size_t indexed_count =
      vec::details::option_count_v<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count_v<vec::details::IsStridedOption, Options...>;
  static_assert(indexed_count <= 1 && strided_count <= 1);
  static_assert(indexed_count + strided_count <= 1);

  if constexpr (indexed_count == 1) {
    auto&& indexed = vec::details::find_option<vec::details::IsIndexedOption>(
        std::forward<Options>(options)...);
    using Indexed = std::remove_cvref_t<decltype(indexed)>;
    static_assert(
        vec::details::IsIndexedOption<Indexed>::scale == 0,
        "tensor indexed addressing only accepts element-scale indices");
    using IndexTag = vec::VecToTag<
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
      vec::details::option_count_v<vec::details::IsFirstOption, Options...>;
  constexpr std::size_t masked_count =
      vec::details::option_count_v<vec::details::IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      vec::details::option_count_v<vec::details::IsUnmaskedOption, Options...>;
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
      vec::details::option_count_v<vec::details::IsFirstOption, Options...>;
  constexpr std::size_t masked_count =
      vec::details::option_count_v<vec::details::IsMaskedOption, Options...>;
  if constexpr (first_count + masked_count == 0) {
    return value;
  } else {
    auto mask = [&]() VECOPS_INLINE_LAMBDA {
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
        vec::details::option_count_v<vec::details::IsVectorMergeOption,
                                   Options...>;
    constexpr std::size_t scalar_merge_count =
        vec::details::option_count_v<vec::details::IsScalarMergeOption,
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

template <std::size_t Rank, typename Layout>
VECOPS_ALWAYS_INLINE void validate_scalar_position(
    const Coord<Rank>& position, const Layout& layout) {
  static_assert(Rank == static_cast<std::size_t>(Layout::Ndim));
  VECOPS_UNROLL
  for (std::size_t d = 0; d < Rank; ++d) {
    VECOPS_ASSERT(
        position[d] >= 0 && position[d] < layout.shape()[d],
        "tensor scalar coordinate is out of bounds");
  }
}

template <typename Transform, vec::VectorTag RequestedTag>
consteval bool transform_tags_representable() {
  using InTag = vec::Rebind<typename Transform::TIn, RequestedTag>;
  using OutTag = vec::Rebind<typename Transform::TOut, RequestedTag>;
  if constexpr (
      vec::is_scalable_tag_v<RequestedTag> &&
      (vec::scale_power_v<InTag> < VEC_HW_MIN_POW ||
       vec::scale_power_v<InTag> > VEC_MAX_POW ||
       vec::scale_power_v<OutTag> < VEC_HW_MIN_POW ||
       vec::scale_power_v<OutTag> > VEC_MAX_POW)) {
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
        vec::is_scalable_tag_v<RequestedTag> &&
        (vec::scale_power_v<MemoryTag> < VEC_HW_MIN_POW ||
         vec::scale_power_v<MemoryTag> > VEC_MAX_POW)) {
      return false;
    } else {
      return true;
    }
  }
}


template <typename Tag>
consteval bool can_split_tag();

template <typename Request>
VECOPS_ALWAYS_INLINE vec::Vec<typename Request::TagType>
execute_resolved_load_convert(
    typename Request::TagType tag,
    const typename Request::FromElement* pointer,
    const Request& request) {
  if constexpr (std::same_as<
                    typename Request::FromElement,
                    vec::ElementOf<typename Request::TagType>>) {
    return vec::details::execute_load_request(
        vec::LoadOp{}, tag, pointer, request);
  } else {
    return vec::details::execute_load_convert_request(
        vec::LoadConvertOp{}, tag, pointer, request);
  }
}

template <typename Request>
VECOPS_ALWAYS_INLINE void execute_resolved_store_convert(
    typename Request::TagType tag,
    typename Request::ToElement* pointer,
    vec::Vec<typename Request::TagType> value,
    const Request& request) {
  if constexpr (std::same_as<
                    typename Request::ToElement,
                    vec::ElementOf<typename Request::TagType>>) {
    vec::details::execute_store_request(
        vec::StoreOp{}, tag, pointer, value, request);
  } else {
    vec::details::execute_store_convert_request(
        vec::StoreConvertOp{}, tag, pointer, value, request);
  }
}

// ============================================================================
// Scalable-SVE transform chunk lowering (per-lane reference path).
//
// BACKEND WORKAROUND: the vectorized chunk pipeline relies on sized
// tag-representation ranges. VLA SVE uses sizeless word groups, which cannot be
// placed in the temporary arrays used by that recursion. The compile-time SVE
// backend therefore selects the reference per-lane implementation below; this
// is not a runtime VL/fallback branch. x86, scalar, and fixed-SVE builds use the
// vectorized recursion defined later in this file. Replace this path only with
// an explicitly sizeless-compatible chunk representation.
// ============================================================================

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)

template <vec::VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE bool lane_is_active(
    Tag tag, nint_t lane, Options&&... options) {
  if constexpr (vec::details::option_count_v<
                    vec::details::IsFirstOption, Options...> == 1) {
    auto&& first = vec::details::find_option<
        vec::details::IsFirstOption>(options...);
    return 0 <= lane && lane < first.count;
  } else if constexpr (vec::details::option_count_v<
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
  if constexpr (vec::details::option_count_v<
                    vec::details::IsIndexedOption, Options...> == 1) {
    auto&& indexed = vec::details::find_option<
        vec::details::IsIndexedOption>(options...);
    using Indexed = std::remove_cvref_t<decltype(indexed)>;
    static_assert(vec::details::IsIndexedOption<Indexed>::scale == 0,
                  "tensor indexed addressing rejects byte scales");
    using IndexTag = vec::VecToTag<
        typename vec::details::IsIndexedOption<Indexed>::Value>;
    return static_cast<nint_t>(
        vec::get(IndexTag{}, indexed.indices, lane));
  } else if constexpr (vec::details::option_count_v<
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
  if constexpr (vec::details::option_count_v<
                    vec::details::IsVectorMergeOption, Options...> == 1) {
    auto&& merge = vec::details::find_option<
        vec::details::IsVectorMergeOption>(options...);
    return vec::get(tag, merge.value, lane);
  } else if constexpr (vec::details::option_count_v<
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

    if constexpr (transform_reads_input_v<Transform>) {
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
  if constexpr (vec::is_fixed_tag_v<Tag>) {
    return vec::fixed_lanes_v<Tag> > 1;
  } else {
    // The largest supported element-size ratio is eight.  Once a requested
    // tag is more than three powers below the backend minimum, rebinding it
    // cannot produce a legal transform tag for any supported element type.
    return vec::scale_power_v<Tag> > VEC_HW_MIN_POW - 3;
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

    // Indexed addressing multiplies the chunk's logical indices by the
    // element stride; the product must outlive `request` (referenced, not
    // stored), so it is computed at function scope.
    const IndexVec physical = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) {
        using IndexElement = vec::ElementOf<IndexVec>;
        const auto stride_scale = vec::fill(
            vec::VecToTag<IndexVec>{},
            static_cast<IndexElement>(physical_stride));
        return vec::mul(logical_indices, stride_scale);
      } else {
        return IndexVec{};
      }
    }();
    vec::Vec<TransformInTag> transform_input;
    if constexpr (transform_reads_input_v<Transform>) {
      constexpr vec::Active ActiveKind =
          HasActive ? vec::Active::Masked : vec::Active::Unmasked;
      constexpr vec::Addressing AddressingKind =
          AddrKind == 0   ? vec::Addressing::Contiguous
          : AddrKind == 1 ? vec::Addressing::Strided
                          : vec::Addressing::Indexed;
      vec::LoadConvertRequest<
          TransformInTag, Memory, ActiveKind, AddressingKind,
          vec::Populate::Zero, vec::mem::Unaligned, Temporal,
          0, std::remove_cvref_t<IndexVec>, Order, Value,
          vec::Mask<TransformInTag>, typename Policy::ActiveResources>
          request;
      // The mask must outlive `request`: requests reference, never store,
      // vector values, so the leaf mask lives at function scope in the
      // transform-input mask domain.
      using LeafMask = std::conditional_t<
          HasActive && !std::same_as<ChunkTag, TransformInTag>,
          vec::Mask<TransformInTag>, MaskVec>;
      const LeafMask leaf_mask = [&]() VECOPS_INLINE_LAMBDA -> LeafMask {
        if constexpr (HasActive) {
          if constexpr (std::same_as<ChunkTag, TransformInTag>) {
            return active_mask;
          } else {
            return vec::convert(TransformInTag{}, ChunkTag{}, active_mask);
          }
        } else {
          return LeafMask{};
        }
      }();
      if constexpr (HasActive) {
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
        request.indices = &physical;
      }
      transform_input = execute_resolved_load_convert(
          TransformInTag{}, chunk_base, request);
    } else {
      transform_input = vec::zeros(TransformInTag{});
    }

    const auto chunk_context = context.subspan(lane_begin, chunk_lanes);
    const auto transformed =
        transform(TransformOutTag{}, transform_input, chunk_context);
    return transformed;
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    using HalfIndexVec = std::conditional_t<
        AddrKind == 2, vec::Vec<vec::Half<vec::VecToTag<IndexVec>>>,
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

    const IndexVec physical = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (AddrKind == 2) {
        using IndexElement = vec::ElementOf<IndexVec>;
        const auto stride_scale = vec::fill(
            vec::VecToTag<IndexVec>{},
            static_cast<IndexElement>(physical_stride));
        return vec::mul(logical_indices, stride_scale);
      } else {
        return IndexVec{};
      }
    }();
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
        std::remove_cvref_t<IndexVec>, Order, Value, Packing,
        vec::Mask<vec::Rebind<Memory, TransformOutTag>>,
        typename Policy::ActiveResources>
        request;
    using LeafMask = std::conditional_t<
        HasActive && !std::same_as<ChunkTag, TransformOutTag>,
        vec::Mask<TransformOutTag>, MaskVec>;
    const LeafMask leaf_mask = [&]() VECOPS_INLINE_LAMBDA -> LeafMask {
      if constexpr (HasActive) {
        if constexpr (std::same_as<ChunkTag, TransformOutTag>) {
          return active_mask;
        } else {
          return vec::convert(TransformOutTag{}, ChunkTag{}, active_mask);
        }
      } else {
        return LeafMask{};
      }
    }();
    if constexpr (HasActive) {
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
      request.indices = &physical;
    }
    execute_resolved_store_convert(
        TransformOutTag{}, chunk_base, transformed, request);
  } else {
    static_assert(
        can_split_tag<ChunkTag>(),
        "DataAccess cannot find a legal vector size for this transform");
    using HalfTag = vec::Half<ChunkTag>;
    using HalfIndexVec = std::conditional_t<
        AddrKind == 2, vec::Vec<vec::Half<vec::VecToTag<IndexVec>>>,
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
  if constexpr (requires { OutRequest::populate_kind; }) {
    if constexpr (OutRequest::populate_kind == vec::Populate::MergeVector) {
      out.merge_vector = in.merge_vector;
    }
    if constexpr (OutRequest::populate_kind == vec::Populate::MergeScalar) {
      out.merge_scalar = in.merge_scalar;
    }
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
      vec::details::option_count_v<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count_v<vec::details::IsStridedOption, Options...>;
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
      using IndexTag = vec::VecToTag<
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
      vec::details::option_count_v<vec::details::IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      vec::details::option_count_v<vec::details::IsStridedOption, Options...>;
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
      using IndexTag = vec::VecToTag<
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
    const auto request = vec::details::resolve_load_request<
        Tag, typename Policy::ActiveResources>(
        std::forward<Options>(options)...);
    using KernelRequest = decltype(request);

    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Indexed) {
      static_assert(
          KernelRequest::index_scale == 0,
          "tensor indexed addressing rejects byte scales");
      using IndexTag =
          vec::VecToTag<typename KernelRequest::IndexVectorType>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{},
          static_cast<Index>(static_cast<nint_t>(tensor_axis_stride)));
      const auto physical = vec::mul(*request.indices, scale);
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Indexed, KernelRequest::populate_kind,
          vec::mem::Unaligned, Temporal, 0,
          std::remove_cvref_t<decltype(physical)>, Order, Value,
          vec::Mask<Tag>, typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      out.indices = &physical;
      return execute_resolved_load_convert(tag, pointer, out);
    } else if constexpr (
        KernelRequest::addressing_kind ==
            vec::Addressing::Contiguous &&
        TensorUnitStride) {
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Contiguous, KernelRequest::populate_kind,
          Alignment, Temporal, 0, vec::Vec<vec::IndexTag<Tag>>, Order, Value,
          vec::Mask<Tag>, typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      return execute_resolved_load_convert(tag, pointer, out);
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (
          KernelRequest::addressing_kind == vec::Addressing::Strided) {
        physical_stride *= request.stride;
      }
      vec::LoadConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Strided, KernelRequest::populate_kind,
          vec::mem::Unaligned, Temporal, 0,
          vec::Vec<vec::IndexTag<Tag>>, Order, Value,
          vec::Mask<Tag>, typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      out.stride = physical_stride;
      return execute_resolved_load_convert(tag, pointer, out);
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
    const auto request = vec::details::resolve_store_request<
        Tag, typename Policy::ActiveResources>(
        std::forward<Options>(options)...);
    using KernelRequest = decltype(request);

    if constexpr (
        KernelRequest::addressing_kind == vec::Addressing::Indexed) {
      static_assert(
          KernelRequest::index_scale == 0,
          "tensor indexed addressing rejects byte scales");
      using IndexTag =
          vec::VecToTag<typename KernelRequest::IndexVectorType>;
      using Index = vec::ElementOf<IndexTag>;
      const auto scale = vec::fill(
          IndexTag{},
          static_cast<Index>(static_cast<nint_t>(tensor_axis_stride)));
      const auto physical = vec::mul(*request.indices, scale);
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Indexed, vec::mem::Unaligned, Temporal, 0,
          std::remove_cvref_t<decltype(physical)>, Order, Value, Packing,
          vec::Mask<vec::Rebind<MemoryElement, Tag>>,
          typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      out.indices = &physical;
      execute_resolved_store_convert(tag, pointer, value, out);
    } else if constexpr (
        KernelRequest::addressing_kind ==
            vec::Addressing::Contiguous &&
        TensorUnitStride) {
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Contiguous, Alignment, Temporal, 0,
          vec::Vec<vec::IndexTag<Tag>>, Order, Value, Packing,
          vec::Mask<vec::Rebind<MemoryElement, Tag>>,
          typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      execute_resolved_store_convert(tag, pointer, value, out);
    } else {
      nint_t physical_stride = static_cast<nint_t>(tensor_axis_stride);
      if constexpr (
          KernelRequest::addressing_kind == vec::Addressing::Strided) {
        physical_stride *= request.stride;
      }
      vec::StoreConvertRequest<
          Tag, MemoryElement, KernelRequest::active_kind,
          vec::Addressing::Strided, vec::mem::Unaligned, Temporal, 0,
          vec::Vec<vec::IndexTag<Tag>>, Order, Value, Packing,
          vec::Mask<vec::Rebind<MemoryElement, Tag>>,
          typename Policy::ActiveResources>
          out{};
      copy_memory_request_fields(out, request);
      out.stride = physical_stride;
      execute_resolved_store_convert(tag, pointer, value, out);
    }
  } else {
    store_memory_options<Tag, Pointer, Policy>(
        tag, pointer, value, tensor_axis_stride,
        std::forward<Options>(options)...);
  }
}

template <int Axis, typename Layout>
/**
 * @brief Build a dense auxiliary layout whose innermost physical axis is Axis.
 * @param layout Original logical shape and compile-time stride metadata.
 * @return Layout with unchanged shape and `stride[Axis] == Const<1>`.
 *
 * Runtime strides are computed without sorting. Compile-time stride types retain
 * shape products where possible, allowing later materialization loops and tail
 * decisions to consume the original Const/constrained Dynamic information.
 */
VECOPS_INLINE auto auxiliary_layout(const Layout& layout) {
  constexpr int Rank = Layout::Ndim;
  ::vecops::details::InlineArray<nint_t, Rank> strides{};
  strides[Axis] = 1;
  nint_t next = layout.shape()[Axis];
  for (int d = Rank - 1; d >= 0; --d) {
    if (d == Axis) continue;
    strides[d] = next;
    next *= layout.shape()[d];
  }

  using Shape = typename Layout::Shape;
  // Moving Axis to the innermost physical position keeps the ordinary suffix
  // product for dimensions before Axis. Dimensions after Axis additionally
  // include shape[Axis] in their stride.
  auto make = [&]<std::size_t... I>(std::index_sequence<I...>) {
    using AuxStrides = Strides<std::conditional_t<
        static_cast<int>(I) == Axis,
        Const<1>,
        std::conditional_t<
            (static_cast<int>(I) < Axis),
            typename ::vecops::tensor::details::StrideTypeForDim<
                static_cast<int>(I), Shape>::type,
            decltype(
                std::declval<typename ::vecops::tensor::details::StrideTypeForDim<
                    static_cast<int>(I), Shape>::type>() *
                std::declval<meta_element_t<Axis, Shape>>())>>...>;
    return make_layout(
        layout.shape(), AuxStrides{strides[I]...});
  };
  return make(std::make_index_sequence<Rank>{});
}

template <typename StridesPack, int SkipDim, int Dim = 0>
/**
 * @brief Find the first non-skipped axis provably unit-stride at compile time.
 * @return Axis index, or -1 if no stride type proves the value one.
 *
 * This intentionally does not test runtime stride values. If metadata cannot
 * prove unit stride, DataAccess selects its general copy/gather lowering rather
 * than adding a runtime branch that would increase every kernel's overhead.
 */
consteval int first_unit_axis() {
  if constexpr (Dim == StridesPack::Ndim) {
    return -1;
  } else if constexpr (
      Dim != SkipDim &&
      is_definitely_one_meta_v<meta_element_t<Dim, StridesPack>>) {
    return Dim;
  } else {
    return first_unit_axis<StridesPack, SkipDim, Dim + 1>();
  }
}

template <int FirstSkip, int SecondSkip, int Dim = 0,
          typename Layout, typename Fn>
/**
 * @brief Enumerate origins of all 2-D planes spanning two selected axes.
 * @param layout N-D logical layout supplying prefix/suffix extents.
 * @param position Mutable traversal coordinate; selected axes are fixed at zero.
 * @param fn Callback receiving each complete plane origin exactly once.
 */
VECOPS_INLINE void for_each_plane(
    const Layout& layout, Coord<Layout::Ndim>& position, Fn&& fn) {
  if constexpr (Dim == Layout::Ndim) {
    std::forward<Fn>(fn)(position);
  } else if constexpr (Dim == FirstSkip || Dim == SecondSkip) {
    position[Dim] = 0;
    for_each_plane<FirstSkip, SecondSkip, Dim + 1>(
        layout, position, std::forward<Fn>(fn));
  } else {
    for (position[Dim] = 0; position[Dim] < layout.shape()[Dim];
         ++position[Dim]) {
      for_each_plane<FirstSkip, SecondSkip, Dim + 1>(
          layout, position, std::forward<Fn>(fn));
    }
  }
}

template <int SrcRow, int SrcCol, int DstRow, int DstCol,
          execution::ExecutionScope Context,
          typename Layout, typename Source, typename Destination>
/**
 * @brief Transpose every selected 2-D plane through the active execution scope.
 * @param context Scope used by Transpose2D for compile-time resource proofs.
 * @param layout Logical N-D iteration shape.
 * @param source Bound readable access.
 * @param destination Bound writable access.
 *
 * No cache tiling is performed: DataAccess materialization is already a
 * kernel-bottom operation and the complete auxiliary buffer lifetime is owned
 * by the surrounding operand scope.
 */
VECOPS_INLINE void transpose_planes(
    Context& context, const Layout& layout,
    Source& source, Destination& destination) {
  Coord<Layout::Ndim> origin{};
  using M = size_type_t<SrcRow, Layout>;
  using N = size_type_t<SrcCol, Layout>;
  const M m{get<SrcRow>(layout.shape())};
  const N n{get<SrcCol>(layout.shape())};
  for_each_plane<SrcRow, SrcCol>(
      layout, origin,
      [&](const auto& plane_origin) VECOPS_INLINE_LAMBDA {
        kernel::transpose2d_bound<SrcRow, SrcCol, DstRow, DstCol>(
            context, m, n, source, plane_origin,
            destination, plane_origin);
      });
}

template <int SkipDim, int Dim = 0, typename Layout, typename Fn>
/** Enumerate vector-line origins while holding `SkipDim` at zero. */
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

template <int SrcRow, int SrcCol, int DstRow, int DstCol,
          typename Layout, typename Source, typename Destination>
/** Compatibility overload using an empty root ExecutionSession. */
VECOPS_INLINE void transpose_planes(
    const Layout& layout, Source& source, Destination& destination) {
  execution::ExecutionSession context{};
  transpose_planes<SrcRow, SrcCol, DstRow, DstCol>(
      context, layout, source, destination);
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
             is_definitely_one_meta_v<meta_element_t<Is, Strides<Ts...>>>) ||
            ...);
  }

  template <std::size_t... Is>
  static consteval bool all_cannot_be_one(std::index_sequence<Is...>) {
    return ((Is == static_cast<std::size_t>(SkipDim) ||
             meta_cannot_be_one_v<meta_element_t<Is, Strides<Ts...>>>) &&
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
  if constexpr (Policy::requested_plan != AccessPlan::automatic &&
                Policy::requested_plan != AccessPlan::automatic_deferred) {
    return Policy::requested_plan;
  } else if constexpr (Spec::is_input) {
    if constexpr (!transform_reads_input_v<typename Spec::TransformType>) {
      return AccessPlan::direct;
    }
    using AxisStride =
        stride_type_t<Policy::vector_axis, typename Spec::InputLayout>;
    if constexpr (Policy::read_passes == 1 ||
                  is_definitely_one_meta_v<AxisStride>) {
      return AccessPlan::direct;
    } else {
      return AccessPlan::materialize_after_transform;
    }
  } else {
    using AxisStride =
        stride_type_t<Policy::vector_axis, typename Spec::OutputLayout>;
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

/** Build stateless call-site defaults for an operand binding. */
template <typename... Options>
  requires (details::valid_access_default_options<Options...>())
VECOPS_INLINE constexpr auto access_defaults(Options...) {
  return details::MakeAccessDefaults<Options...>{};
}

template <typename PlanningPolicy, typename Defaults, typename Resources,
          int SlicedDim>
struct SliceAccessPolicy<
    details::AccessLoweringPolicy<PlanningPolicy, Defaults, Resources>,
    SlicedDim> {
  using type = details::AccessLoweringPolicy<
      slice_access_policy_t<PlanningPolicy, SlicedDim>, Defaults, Resources>;
};

template <typename PlanningPolicy, typename Defaults, typename Resources,
          int I, int J>
struct TransposeAccessPolicy<
    details::AccessLoweringPolicy<PlanningPolicy, Defaults, Resources>, I, J> {
  using type = details::AccessLoweringPolicy<
      transpose_access_policy_t<PlanningPolicy, I, J>, Defaults, Resources>;
};

/**
 * @brief Opt-in contract for values retained by an unbound operand pattern.
 *
 * Planning may retain a projection or fact only when the value cannot own an
 * execution-time Tensor/address. Empty policy objects are safe by default.
 * Transforms use `TypeOnlyPatternMetadata` instead and are always stripped of
 * values. Stateful projection/fact values deliberately require an explicit
 * specialization; trivially-copyable is not sufficient because pointers and
 * bound Tensor views are trivially copyable too.
 */
template <typename T>
struct OperandPatternMetadata {
  static constexpr bool storage_independent = std::is_empty_v<T>;

  VECOPS_ALWAYS_INLINE static constexpr T make_pattern(const T& value)
    requires (storage_independent) {
    return value;
  }
};

/** Coordinate projections contain coordinates, but no storage binding. */
template <std::size_t OriginalRank, std::size_t LocalRank>
struct OperandPatternMetadata<
    CoordinateProjection<OriginalRank, LocalRank>> {
  static constexpr bool storage_independent = true;

  VECOPS_ALWAYS_INLINE static constexpr auto make_pattern(
      const CoordinateProjection<OriginalRank, LocalRank>& value) {
    return value;
  }
};

template <typename T>
inline constexpr bool is_operand_pattern_metadata_v =
    OperandPatternMetadata<std::remove_cvref_t<T>>::storage_independent;

template <typename T>
concept OperandPatternMetadataLike = is_operand_pattern_metadata_v<T>;

/** Type-only placeholder used instead of a Transform value by unbound Specs. */
template <typename Transform>
struct TypeOnlyPatternMetadata {
  using Type = Transform;
};

namespace details {

template <typename T>
VECOPS_ALWAYS_INLINE constexpr auto make_operand_pattern_metadata(
    const T& value) {
  using Clean = std::remove_cvref_t<T>;
  static_assert(
      OperandPatternMetadata<Clean>::storage_independent,
      "stateful operand metadata must specialize "
      "tensor::OperandPatternMetadata before it can be retained by an "
      "unbound planning pattern");
  return OperandPatternMetadata<Clean>::make_pattern(value);
}

template <std::size_t Axis, typename PatternLayout, typename ActualLayout>
VECOPS_ALWAYS_INLINE constexpr bool operand_shape_axis_conforms(
    const PatternLayout& pattern, const ActualLayout& actual) {
  using PatternExtent = shape_extent_type_t<
      static_cast<int>(Axis), PatternLayout>;
  const nint_t capacity = pattern.shape()[static_cast<int>(Axis)];
  const nint_t extent = actual.shape()[static_cast<int>(Axis)];
  if constexpr (meta::is_singleton_v<PatternExtent>) {
    return extent == capacity;
  } else {
    static_assert(
        PatternExtent::is_runtime,
        "operand pattern shape axes must be Const or runtime Values");
    // An unbounded Dynamic is a shape contract, not a first-call capacity
    // bucket.  Its runtime extent can legitimately change between denoising
    // steps or rank-local views; only bounded Dynamic patterns retain the
    // planning-capacity check.
    if constexpr (!PatternExtent::has_upper)
      return PatternExtent::conforms(extent);
    return PatternExtent::conforms(extent) && extent <= capacity;
  }
}

template <typename PatternLayout, typename ActualLayout, std::size_t... Axis>
VECOPS_ALWAYS_INLINE constexpr bool operand_layout_conforms_impl(
    const PatternLayout& pattern, const ActualLayout& actual,
    std::index_sequence<Axis...>) {
  return (operand_shape_axis_conforms<Axis>(pattern, actual) && ...) &&
      ((pattern.strides()[static_cast<int>(Axis)] ==
        actual.strides()[static_cast<int>(Axis)]) && ...);
}

/**
 * Validate a runtime Layout against a planning Layout contract.
 *
 * A Const shape axis is exact. A bounded Dynamic axis also respects the
 * instance capacity used by planning. An unbounded Dynamic axis accepts any
 * conforming runtime extent; it is a shape contract rather than a first-call
 * capacity bucket. Strides remain exact because changing them can alter
 * DataAccess lowering and its workspace requirement.
 */
template <typename PatternLayout, typename ActualLayout>
VECOPS_ALWAYS_INLINE constexpr bool operand_layout_conforms(
    const PatternLayout& pattern, const ActualLayout& actual) {
  static_assert(PatternLayout::Ndim == ActualLayout::Ndim);
  return operand_layout_conforms_impl(
      pattern, actual,
      std::make_index_sequence<PatternLayout::Ndim>{});
}

struct OperandLayoutContractDiagnostics {
  bool conforms = true;
  int axis = -1;
  nint_t pattern_extent = 0;
  nint_t actual_extent = 0;
  nint_t pattern_stride = 0;
  nint_t actual_stride = 0;
};

template <typename PatternLayout, typename ActualLayout, std::size_t... Axis>
VECOPS_ALWAYS_INLINE auto operand_layout_contract_diagnostics_impl(
    const PatternLayout& pattern, const ActualLayout& actual,
    std::index_sequence<Axis...>) {
  OperandLayoutContractDiagnostics result;
  auto inspect = [&]<std::size_t I>() {
    using PatternExtent = shape_extent_type_t<static_cast<int>(I), PatternLayout>;
    const nint_t pattern_extent = pattern.shape()[static_cast<int>(I)];
    const nint_t actual_extent = actual.shape()[static_cast<int>(I)];
    const nint_t pattern_stride = pattern.strides()[static_cast<int>(I)];
    const nint_t actual_stride = actual.strides()[static_cast<int>(I)];
    const bool shape_ok = [&] {
      if constexpr (meta::is_singleton_v<PatternExtent>)
        return actual_extent == pattern_extent;
      else if constexpr (!PatternExtent::has_upper)
        return PatternExtent::conforms(actual_extent);
      else
        return PatternExtent::conforms(actual_extent) &&
               actual_extent <= pattern_extent;
    }();
    if (result.conforms && (!shape_ok || pattern_stride != actual_stride)) {
      result.conforms = false;
      result.axis = static_cast<int>(I);
      result.pattern_extent = pattern_extent;
      result.actual_extent = actual_extent;
      result.pattern_stride = pattern_stride;
      result.actual_stride = actual_stride;
    }
  };
  (inspect.template operator()<Axis>(), ...);
  return result;
}

template <typename PatternLayout, typename ActualLayout>
VECOPS_ALWAYS_INLINE auto operand_layout_contract_diagnostics(
    const PatternLayout& pattern, const ActualLayout& actual) {
  static_assert(PatternLayout::Ndim == ActualLayout::Ndim);
  return operand_layout_contract_diagnostics_impl(
      pattern, actual,
      std::make_index_sequence<PatternLayout::Ndim>{});
}

template <UnboundTensorLike Pattern, BoundTensorLike Actual>
  requires (Pattern::Ndim == Actual::Ndim)
VECOPS_ALWAYS_INLINE auto bind_operand_pattern_tensor(
    const Pattern& pattern, const Actual& actual) {
  const auto diagnostics = operand_layout_contract_diagnostics(
      pattern.layout(), actual.layout());
  VECOPS_CHECK(
      diagnostics.conforms,
      "actual Tensor violates planning layout contract at axis %d: "
      "pattern extent=%td actual extent=%td pattern stride=%td "
      "actual stride=%td",
      diagnostics.axis, diagnostics.pattern_extent, diagnostics.actual_extent,
      diagnostics.pattern_stride, diagnostics.actual_stride);
  using PatternLayout = typename Pattern::Layout;
  auto execution_layout = actual.layout().template as<
      typename PatternLayout::Shape, typename PatternLayout::Strides>();
  return tensor::make_tensor(actual.data(), execution_layout);
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
  static constexpr bool is_bound = Tensor::is_bound;
  static constexpr bool is_unbound = Tensor::is_unbound;
  static constexpr bool pattern_metadata_safe =
      is_operand_pattern_metadata_v<Projection> &&
      (is_operand_pattern_metadata_v<Facts> && ...);
  static_assert(
      Tensor::is_bound || pattern_metadata_safe,
      "an unbound input Spec cannot retain projection/fact state "
      "unless OperandPatternMetadata explicitly sanitizes it");
  using ComputeType = Compute;
  using InputTensor = Tensor;
  using MemoryElement = std::remove_const_t<typename Tensor::ElementType>;
  using TransformType = Transform;
  using ProjectionType = Projection;
  using ExternalFacts = std::tuple<Facts...>;
  using InputLayout = typename Tensor::Layout;
  using TransformStorage = std::conditional_t<
      Tensor::is_bound, Transform, TypeOnlyPatternMetadata<Transform>>;

  VECOPS_ALWAYS_INLINE InputSpec(
      Tensor tensor, Transform transform, Projection projection,
      Facts... facts)
      : tensor_(tensor), transform_(make_transform_storage(std::move(transform))),
        projection_(projection), facts_(std::move(facts)...) {}

  VECOPS_ALWAYS_INLINE InputSpec(
      Tensor tensor, TypeOnlyPatternMetadata<Transform>,
      Projection projection, Facts... facts)
    requires (Tensor::is_unbound)
      : tensor_(tensor), transform_{}, projection_(projection),
        facts_(std::move(facts)...) {}

  VECOPS_ALWAYS_INLINE InputSpec(Tensor tensor, Transform transform)
    requires (sizeof...(Facts) == 0)
      : InputSpec(
            tensor, std::move(transform),
            identity_projection<Tensor::Ndim>()) {}

  VECOPS_ALWAYS_INLINE explicit InputSpec(Tensor tensor)
    requires (std::same_as<Transform, NoTransform> &&
              sizeof...(Facts) == 0)
      : InputSpec(tensor, NoTransform{}) {}

  VECOPS_ALWAYS_INLINE const Tensor& tensor() const { return tensor_; }
  VECOPS_ALWAYS_INLINE const InputLayout& input_layout() const {
    return tensor_.layout();
  }
  VECOPS_ALWAYS_INLINE const Transform& transform() const
    requires (Tensor::is_bound) {
    return transform_;
  }
  VECOPS_ALWAYS_INLINE const Projection& projection() const {
    return projection_;
  }
  VECOPS_ALWAYS_INLINE const ExternalFacts& facts() const { return facts_; }

  /** Replace only the primary Tensor while preserving access semantics. */
  template <TensorLike NewTensor>
    requires (NewTensor::Ndim == Tensor::Ndim &&
              std::same_as<
                  std::remove_const_t<typename NewTensor::ElementType>,
                  std::remove_const_t<typename Tensor::ElementType>>)
  VECOPS_ALWAYS_INLINE auto with_tensor(NewTensor tensor) const {
    return with_tensor_and_projection(tensor, projection_);
  }

  template <TensorLike NewTensor, typename NewProjection>
    requires (NewTensor::Ndim == Tensor::Ndim &&
              std::same_as<
                  std::remove_const_t<typename NewTensor::ElementType>,
                  std::remove_const_t<typename Tensor::ElementType>>)
  VECOPS_ALWAYS_INLINE auto with_tensor_and_projection(
      NewTensor tensor, NewProjection projection) const {
    return std::apply(
        [&](const auto&... facts) VECOPS_INLINE_LAMBDA {
          (details::validate_operand_fact(tensor, facts), ...);
          using Result = InputSpec<
              Compute, NewTensor, Transform, NewProjection, Facts...>;
          if constexpr (NewTensor::is_unbound) {
            return Result{
                tensor, TypeOnlyPatternMetadata<Transform>{}, projection,
                facts...};
          } else {
            static_assert(
                Tensor::is_bound ||
                    (std::is_empty_v<Transform> &&
                     std::default_initializable<Transform>),
                "a stateful transform pattern must rebind to an actual Spec, "
                "not directly to a Tensor");
            if constexpr (Tensor::is_bound) {
              return Result{tensor, transform_, projection, facts...};
            } else {
              return Result{tensor, Transform{}, projection, facts...};
            }
          }
        },
        facts_);
  }

  /** Structural view helper; pointer-relative facts are intentionally dropped. */
  template <TensorLike NewTensor, typename NewProjection>
  VECOPS_ALWAYS_INLINE auto with_view_tensor_and_projection(
      NewTensor tensor, NewProjection projection) const {
    using Result = InputSpec<
        Compute, NewTensor, Transform, NewProjection>;
    if constexpr (NewTensor::is_unbound) {
      return Result{
          tensor, TypeOnlyPatternMetadata<Transform>{}, projection};
    } else if constexpr (Tensor::is_bound) {
      return Result{tensor, transform_, projection};
    } else {
      static_assert(
          std::is_empty_v<Transform> &&
              std::default_initializable<Transform>,
          "a stateful transform pattern must rebind to an actual Spec");
      return Result{tensor, Transform{}, projection};
    }
  }

private:
  VECOPS_ALWAYS_INLINE static TransformStorage make_transform_storage(
      Transform transform) {
    if constexpr (Tensor::is_bound) return transform;
    else return {};
  }

  Tensor tensor_;
  [[no_unique_address]] TransformStorage transform_;
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
  static constexpr bool is_bound = Tensor::is_bound;
  static constexpr bool is_unbound = Tensor::is_unbound;
  static constexpr bool pattern_metadata_safe =
      is_operand_pattern_metadata_v<Projection> &&
      (is_operand_pattern_metadata_v<Facts> && ...);
  static_assert(
      Tensor::is_bound || pattern_metadata_safe,
      "an unbound output Spec cannot retain projection/fact state "
      "unless OperandPatternMetadata explicitly sanitizes it");
  static_assert(!std::is_const_v<typename Tensor::ElementType>,
                "tensor::output requires a mutable Tensor");
  using ComputeType = Compute;
  using OutputTensor = Tensor;
  using MemoryElement = typename Tensor::ElementType;
  using TransformType = Transform;
  using ProjectionType = Projection;
  using ExternalFacts = std::tuple<Facts...>;
  using OutputLayout = typename Tensor::Layout;
  using TransformStorage = std::conditional_t<
      Tensor::is_bound, Transform, TypeOnlyPatternMetadata<Transform>>;

  VECOPS_ALWAYS_INLINE OutputSpec(
      Tensor tensor, Transform transform, Projection projection,
      Facts... facts)
      : tensor_(tensor), transform_(make_transform_storage(std::move(transform))),
        projection_(projection), facts_(std::move(facts)...) {}

  VECOPS_ALWAYS_INLINE OutputSpec(
      Tensor tensor, TypeOnlyPatternMetadata<Transform>,
      Projection projection, Facts... facts)
    requires (Tensor::is_unbound)
      : tensor_(tensor), transform_{}, projection_(projection),
        facts_(std::move(facts)...) {}

  VECOPS_ALWAYS_INLINE OutputSpec(Tensor tensor, Transform transform)
    requires (sizeof...(Facts) == 0)
      : OutputSpec(
            tensor, std::move(transform),
            identity_projection<Tensor::Ndim>()) {}

  VECOPS_ALWAYS_INLINE explicit OutputSpec(Tensor tensor)
    requires (std::same_as<Transform, NoTransform> &&
              sizeof...(Facts) == 0)
      : OutputSpec(tensor, NoTransform{}) {}

  VECOPS_ALWAYS_INLINE const Tensor& tensor() const { return tensor_; }
  VECOPS_ALWAYS_INLINE const OutputLayout& output_layout() const {
    return tensor_.layout();
  }
  VECOPS_ALWAYS_INLINE const Transform& transform() const
    requires (Tensor::is_bound) {
    return transform_;
  }
  VECOPS_ALWAYS_INLINE const Projection& projection() const {
    return projection_;
  }
  VECOPS_ALWAYS_INLINE const ExternalFacts& facts() const { return facts_; }

  /** Replace only the primary Tensor while preserving access semantics. */
  template <TensorLike NewTensor>
    requires (NewTensor::Ndim == Tensor::Ndim &&
              !std::is_const_v<typename NewTensor::ElementType> &&
              std::same_as<
                  typename NewTensor::ElementType,
                  typename Tensor::ElementType>)
  VECOPS_ALWAYS_INLINE auto with_tensor(NewTensor tensor) const {
    return with_tensor_and_projection(tensor, projection_);
  }

  template <TensorLike NewTensor, typename NewProjection>
    requires (NewTensor::Ndim == Tensor::Ndim &&
              !std::is_const_v<typename NewTensor::ElementType> &&
              std::same_as<
                  typename NewTensor::ElementType,
                  typename Tensor::ElementType>)
  VECOPS_ALWAYS_INLINE auto with_tensor_and_projection(
      NewTensor tensor, NewProjection projection) const {
    return std::apply(
        [&](const auto&... facts) VECOPS_INLINE_LAMBDA {
          (details::validate_operand_fact(tensor, facts), ...);
          using Result = OutputSpec<
              Compute, NewTensor, Transform, NewProjection, Facts...>;
          if constexpr (NewTensor::is_unbound) {
            return Result{
                tensor, TypeOnlyPatternMetadata<Transform>{}, projection,
                facts...};
          } else {
            static_assert(
                Tensor::is_bound ||
                    (std::is_empty_v<Transform> &&
                     std::default_initializable<Transform>),
                "a stateful transform pattern must rebind to an actual Spec, "
                "not directly to a Tensor");
            if constexpr (Tensor::is_bound) {
              return Result{tensor, transform_, projection, facts...};
            } else {
              return Result{tensor, Transform{}, projection, facts...};
            }
          }
        },
        facts_);
  }

  /** Structural view helper; pointer-relative facts are intentionally dropped. */
  template <TensorLike NewTensor, typename NewProjection>
  VECOPS_ALWAYS_INLINE auto with_view_tensor_and_projection(
      NewTensor tensor, NewProjection projection) const {
    using Result = OutputSpec<
        Compute, NewTensor, Transform, NewProjection>;
    if constexpr (NewTensor::is_unbound) {
      return Result{
          tensor, TypeOnlyPatternMetadata<Transform>{}, projection};
    } else if constexpr (Tensor::is_bound) {
      return Result{tensor, transform_, projection};
    } else {
      static_assert(
          std::is_empty_v<Transform> &&
              std::default_initializable<Transform>,
          "a stateful transform pattern must rebind to an actual Spec");
      return Result{tensor, Transform{}, projection};
    }
  }

private:
  VECOPS_ALWAYS_INLINE static TransformStorage make_transform_storage(
      Transform transform) {
    if constexpr (Tensor::is_bound) return transform;
    else return {};
  }

  Tensor tensor_;
  [[no_unique_address]] TransformStorage transform_;
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
struct IsInputSpec : std::bool_constant<
    is_specialization_of_v<InputSpec, T>> {};
template <typename T>
inline constexpr bool is_input_spec_v =
    IsInputSpec<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsOutputSpec : std::bool_constant<
    is_specialization_of_v<OutputSpec, T>> {};
template <typename T>
inline constexpr bool is_output_spec_v =
    IsOutputSpec<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_bound_input_spec_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_input_spec_v<Clean>) return Clean::is_bound;
  else return false;
}();

template <typename T>
inline constexpr bool is_unbound_input_spec_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_input_spec_v<Clean>) return Clean::is_unbound;
  else return false;
}();

template <typename T>
inline constexpr bool is_bound_output_spec_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_output_spec_v<Clean>) return Clean::is_bound;
  else return false;
}();

template <typename T>
inline constexpr bool is_unbound_output_spec_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_output_spec_v<Clean>) return Clean::is_unbound;
  else return false;
}();

template <typename T>
concept BoundInputSpecLike = is_bound_input_spec_v<T>;

template <typename T>
concept UnboundInputSpecLike = is_unbound_input_spec_v<T>;

template <typename T>
concept BoundOutputSpecLike = is_bound_output_spec_v<T>;

template <typename T>
concept UnboundOutputSpecLike = is_unbound_output_spec_v<T>;

/** True only for storage-less Tensor/Spec planning patterns. */
template <typename T>
inline constexpr bool is_unbound_tensor_view_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_tensor_v<Clean>) return Clean::is_unbound;
  else if constexpr (is_input_spec_v<Clean> || is_output_spec_v<Clean>) {
    return Clean::is_unbound;
  } else return false;
}();

template <typename T>
concept UnboundTensorView = is_unbound_tensor_view_v<T>;

/** True only for Tensor/Spec values carrying an execution-time address. */
template <typename T>
inline constexpr bool is_bound_tensor_view_v = [] {
  using Clean = std::remove_cvref_t<T>;
  if constexpr (is_tensor_v<Clean>) return Clean::is_bound;
  else if constexpr (is_input_spec_v<Clean> || is_output_spec_v<Clean>) {
    return Clean::is_bound;
  } else return false;
}();

template <typename T>
concept BoundTensorView = is_bound_tensor_view_v<T>;

/**
 * Drop the primary storage binding from an input Spec for planning.
 *
 * Transform state is replaced by `TypeOnlyPatternMetadata<Transform>` so even
 * a transform containing auxiliary bound Tensors cannot leak execution
 * addresses into the planning graph. Projection/fact values remain subject to
 * `OperandPatternMetadata` because structural views use them during planning.
 */
template <typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
  requires (is_operand_pattern_metadata_v<Projection> &&
            (is_operand_pattern_metadata_v<Facts> && ...))
VECOPS_INLINE auto unbind(
    const InputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec) {
  auto pattern_tensor = tensor::unbind(spec.tensor());
  return std::apply(
      [&](const auto&... facts) VECOPS_INLINE_LAMBDA {
        return InputSpec<
            Compute, decltype(pattern_tensor), Transform, Projection,
            Facts...>{
                pattern_tensor,
                TypeOnlyPatternMetadata<Transform>{},
                details::make_operand_pattern_metadata(spec.projection()),
                details::make_operand_pattern_metadata(facts)...};
      },
      spec.facts());
}

/** Storage-less output counterpart of `unbind(InputSpec)`. */
template <typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
  requires (is_operand_pattern_metadata_v<Projection> &&
            (is_operand_pattern_metadata_v<Facts> && ...))
VECOPS_INLINE auto unbind(
    const OutputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec) {
  auto pattern_tensor = tensor::unbind(spec.tensor());
  return std::apply(
      [&](const auto&... facts) VECOPS_INLINE_LAMBDA {
        return OutputSpec<
            Compute, decltype(pattern_tensor), Transform, Projection,
            Facts...>{
                pattern_tensor,
                TypeOnlyPatternMetadata<Transform>{},
                details::make_operand_pattern_metadata(spec.projection()),
                details::make_operand_pattern_metadata(facts)...};
      },
      spec.facts());
}

/**
 * Rebind a safe input pattern to a corresponding bound Tensor view.
 *
 * `actual` denotes the complete view represented by the pattern, rather than
 * the base of a previously sliced Tensor. Use the pointer overload of `bind`
 * when the pattern's accumulated element offset must be applied.
 */
template <UnboundInputSpecLike Pattern, BoundTensorLike Actual>
  requires (Pattern::InputTensor::Ndim == Actual::Ndim &&
            std::same_as<
                std::remove_const_t<typename Pattern::InputTensor::ElementType>,
                std::remove_const_t<typename Actual::ElementType>> &&
            Pattern::pattern_metadata_safe &&
            std::is_empty_v<typename Pattern::TransformType> &&
            std::default_initializable<typename Pattern::TransformType>)
VECOPS_INLINE auto rebind(const Pattern& pattern, Actual actual) {
  return pattern.with_tensor(
      details::bind_operand_pattern_tensor(pattern.tensor(), actual));
}

/** Rebind a safe output pattern to a corresponding writable Tensor view. */
template <UnboundOutputSpecLike Pattern, BoundTensorLike Actual>
  requires (Pattern::OutputTensor::Ndim == Actual::Ndim &&
            std::same_as<
                typename Pattern::OutputTensor::ElementType,
                typename Actual::ElementType> &&
            !std::is_const_v<typename Actual::ElementType> &&
            Pattern::pattern_metadata_safe &&
            std::is_empty_v<typename Pattern::TransformType> &&
            std::default_initializable<typename Pattern::TransformType>)
VECOPS_INLINE auto rebind(const Pattern& pattern, Actual actual) {
  return pattern.with_tensor(
      details::bind_operand_pattern_tensor(pattern.tensor(), actual));
}

/** Bind an input pattern to the base address from which it was sliced. */
template <UnboundInputSpecLike Pattern>
  requires (Pattern::pattern_metadata_safe &&
            std::is_empty_v<typename Pattern::TransformType> &&
            std::default_initializable<typename Pattern::TransformType>)
VECOPS_INLINE auto bind(
    const Pattern& pattern,
    typename Pattern::InputTensor::ElementType* base) {
  return pattern.with_tensor(tensor::bind(pattern.tensor(), base));
}

/** Bind an output pattern to the base address from which it was sliced. */
template <UnboundOutputSpecLike Pattern>
  requires (Pattern::pattern_metadata_safe &&
            std::is_empty_v<typename Pattern::TransformType> &&
            std::default_initializable<typename Pattern::TransformType>)
VECOPS_INLINE auto bind(
    const Pattern& pattern,
    typename Pattern::OutputTensor::ElementType* base) {
  return pattern.with_tensor(tensor::bind(pattern.tensor(), base));
}

/**
 * Validate an actual input Spec against a planning pattern and retain the
 * actual Spec. This is the execution path for stateful transforms: runtime
 * transform state comes from `actual`, never from the planning pattern.
 */
template <UnboundInputSpecLike Pattern, BoundInputSpecLike Actual>
  requires (std::same_as<typename Pattern::ComputeType,
                         typename Actual::ComputeType> &&
            std::same_as<typename Pattern::MemoryElement,
                         typename Actual::MemoryElement> &&
            Pattern::InputTensor::Ndim == Actual::InputTensor::Ndim &&
            std::same_as<typename Pattern::TransformType,
                         typename Actual::TransformType> &&
            std::same_as<typename Pattern::ProjectionType,
                         typename Actual::ProjectionType> &&
            std::same_as<typename Pattern::ExternalFacts,
                         typename Actual::ExternalFacts>)
VECOPS_INLINE auto rebind(const Pattern& pattern, Actual actual) {
  auto tensor = details::bind_operand_pattern_tensor(
      pattern.tensor(), actual.tensor());
  return actual.with_tensor(tensor);
}

/** Output counterpart of `rebind(pattern, actual_input_spec)`. */
template <UnboundOutputSpecLike Pattern, BoundOutputSpecLike Actual>
  requires (std::same_as<typename Pattern::ComputeType,
                         typename Actual::ComputeType> &&
            std::same_as<typename Pattern::MemoryElement,
                         typename Actual::MemoryElement> &&
            Pattern::OutputTensor::Ndim == Actual::OutputTensor::Ndim &&
            std::same_as<typename Pattern::TransformType,
                         typename Actual::TransformType> &&
            std::same_as<typename Pattern::ProjectionType,
                         typename Actual::ProjectionType> &&
            std::same_as<typename Pattern::ExternalFacts,
                         typename Actual::ExternalFacts>)
VECOPS_INLINE auto rebind(const Pattern& pattern, Actual actual) {
  auto tensor = details::bind_operand_pattern_tensor(
      pattern.tensor(), actual.tensor());
  return actual.with_tensor(tensor);
}

template <typename T>
inline constexpr bool is_input_operand_v =
    is_tensor_v<std::remove_cvref_t<T>> || is_input_spec_v<T>;

template <typename T>
concept InputOperand = is_input_operand_v<T>;

/** Readable Tensor or InputSpec of one exact logical rank. */
template <typename Operand, int Rank>
inline constexpr bool is_input_operand_of_v = [] {
  using O = std::remove_cvref_t<Operand>;
  if constexpr (is_tensor_v<O>) {
    return O::Ndim == Rank;
  } else if constexpr (is_input_spec_v<O>) {
    return O::InputTensor::Ndim == Rank;
  } else {
    return false;
  }
}();

template <typename Operand, int Rank>
concept InputOperandOf = is_input_operand_of_v<Operand, Rank>;

/** Omitted or readable Tensor/InputSpec of any logical rank. */
template <typename Operand>
inline constexpr bool is_optional_input_operand_v =
    is_nullopt_v<Operand> || is_input_operand_v<Operand>;

template <typename Operand>
concept OptionalInputOperand = is_optional_input_operand_v<Operand>;

/** Omitted or readable Tensor/InputSpec of one exact logical rank. */
template <typename Operand, int Rank>
inline constexpr bool is_optional_input_operand_of_v =
    is_nullopt_v<Operand> || is_input_operand_of_v<Operand, Rank>;

template <typename Operand, int Rank>
concept OptionalInputOperandOf =
    is_optional_input_operand_of_v<Operand, Rank>;

template <typename T>
inline constexpr bool is_output_operand_v = [] {
  using O = std::remove_cvref_t<T>;
  if constexpr (is_output_spec_v<O>) return true;
  else if constexpr (is_tensor_v<O>) {
    return !std::is_const_v<typename O::ElementType>;
  } else return false;
}();

template <typename T>
concept OutputOperand = is_output_operand_v<T>;

/** Writable Tensor or OutputSpec of one exact logical rank. */
template <typename Operand, int Rank>
inline constexpr bool is_output_operand_of_v = [] {
  if constexpr (!is_output_operand_v<Operand>) {
    return false;
  } else {
    using O = std::remove_cvref_t<Operand>;
    if constexpr (is_tensor_v<O>) {
      return O::Ndim == Rank;
    } else {
      static_assert(is_output_spec_v<O>);
      return O::OutputTensor::Ndim == Rank;
    }
  }
}();

template <typename Operand, int Rank>
concept OutputOperandOf = is_output_operand_of_v<Operand, Rank>;

/** @brief Concept form of is_input_spec_v. */
template <typename T>
concept InputSpecLike = is_input_spec_v<T>;

/** @brief Concept form of is_output_spec_v. */
template <typename T>
concept OutputSpecLike = is_output_spec_v<T>;

/** Normalize a Tensor or an existing input Spec without discarding metadata. */
template <typename Compute, TensorLike Tensor>
VECOPS_INLINE auto as_input_spec(Tensor&& tensor) {
  return input<Compute>(std::forward<Tensor>(tensor));
}

template <typename Compute, InputSpecLike Spec>
  requires std::same_as<
      typename std::remove_cvref_t<Spec>::ComputeType, Compute>
VECOPS_INLINE auto as_input_spec(Spec&& spec) {
  return std::forward<Spec>(spec);
}

/** Preserve compile-time optionality while normalizing a readable operand. */
template <typename Compute>
VECOPS_INLINE constexpr nullopt_t as_input_spec(nullopt_t) {
  return nullopt;
}

/** Normalize a mutable Tensor or an existing output Spec. */
template <typename Compute, TensorLike Tensor>
  requires (!std::is_const_v<
      typename std::remove_cvref_t<Tensor>::ElementType>)
VECOPS_INLINE auto as_output_spec(Tensor&& tensor) {
  return output<Compute>(std::forward<Tensor>(tensor));
}

template <typename Compute, OutputSpecLike Spec>
  requires std::same_as<
      typename std::remove_cvref_t<Spec>::ComputeType, Compute>
VECOPS_INLINE auto as_output_spec(Spec&& spec) {
  return std::forward<Spec>(spec);
}

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
  return spec.with_view_tensor_and_projection(sliced_tensor, projection);
}

template <int I, int J, typename Compute, typename Tensor,
          typename Transform, typename Projection, typename... Facts>
VECOPS_INLINE auto transpose_view(
    const InputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec) {
  static_assert(0 <= I && I < Tensor::Ndim);
  static_assert(0 <= J && J < Tensor::Ndim);
  auto transposed_tensor = tensor::transpose_view<I, J>(spec.tensor());
  auto projection = spec.projection().template transposed<I, J>();
  return spec.with_view_tensor_and_projection(transposed_tensor, projection);
}

/** Preserve rank while selecting a contiguous logical interval on one axis. */
template <int Dim, typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts, meta::ValueType Extent>
VECOPS_INLINE auto narrow_view(
    const InputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec,
    nint_t offset, Extent extent) {
  static_assert(0 <= Dim && Dim < Tensor::Ndim);
  const nint_t count = static_cast<nint_t>(extent);
  VECOPS_ASSERT(offset >= 0 && count >= 0 &&
                offset + count <= spec.input_layout().shape()[Dim],
                "input narrow interval is out of bounds");
  auto layout = tensor::set<Dim>(
      spec.input_layout(), extent,
      tensor::stride<Dim>(spec.input_layout()));
  auto tensor_view = [&]() VECOPS_INLINE_LAMBDA {
    const nint_t displacement = offset * static_cast<nint_t>(
        tensor::get<Dim>(spec.input_layout().strides()));
    if constexpr (Tensor::is_bound) {
      return tensor::make_tensor(spec.tensor().data() + displacement, layout);
    } else {
      using Layout = decltype(layout);
      using Pattern = tensor::Tensor<
          typename Tensor::ElementType,
          typename Layout::Shape,
          typename Layout::Strides,
          tensor::UnboundBinding>;
      return Pattern{
          tensor::UnboundBinding{
              spec.tensor().element_offset() + displacement},
          layout};
    }
  }();
  auto projection = spec.projection().template narrowed<Dim>(offset);
  return spec.with_view_tensor_and_projection(tensor_view, projection);
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
  return spec.with_view_tensor_and_projection(sliced_tensor, projection);
}

template <int I, int J, typename Compute, typename Tensor,
          typename Transform, typename Projection, typename... Facts>
VECOPS_INLINE auto transpose_view(
    const OutputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec) {
  static_assert(0 <= I && I < Tensor::Ndim);
  static_assert(0 <= J && J < Tensor::Ndim);
  auto transposed_tensor = tensor::transpose_view<I, J>(spec.tensor());
  auto projection = spec.projection().template transposed<I, J>();
  return spec.with_view_tensor_and_projection(transposed_tensor, projection);
}

template <int Dim, typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts, meta::ValueType Extent>
VECOPS_INLINE auto narrow_view(
    const OutputSpec<Compute, Tensor, Transform, Projection, Facts...>& spec,
    nint_t offset, Extent extent) {
  static_assert(0 <= Dim && Dim < Tensor::Ndim);
  const nint_t count = static_cast<nint_t>(extent);
  VECOPS_ASSERT(offset >= 0 && count >= 0 &&
                offset + count <= spec.output_layout().shape()[Dim],
                "output narrow interval is out of bounds");
  auto layout = tensor::set<Dim>(
      spec.output_layout(), extent,
      tensor::stride<Dim>(spec.output_layout()));
  auto tensor_view = [&]() VECOPS_INLINE_LAMBDA {
    const nint_t displacement = offset * static_cast<nint_t>(
        tensor::get<Dim>(spec.output_layout().strides()));
    if constexpr (Tensor::is_bound) {
      return tensor::make_tensor(spec.tensor().data() + displacement, layout);
    } else {
      using Layout = decltype(layout);
      using Pattern = tensor::Tensor<
          typename Tensor::ElementType,
          typename Layout::Shape,
          typename Layout::Strides,
          tensor::UnboundBinding>;
      return Pattern{
          tensor::UnboundBinding{
              spec.tensor().element_offset() + displacement},
          layout};
    }
  }();
  auto projection = spec.projection().template narrowed<Dim>(offset);
  return spec.with_view_tensor_and_projection(tensor_view, projection);
}

/** @brief Keep the first N Spec dimensions at zero on trailing axes. */
template <int N, typename Spec>
  requires (InputSpecLike<Spec> || OutputSpecLike<Spec>)
VECOPS_INLINE auto take_leading(const Spec& spec) {
  constexpr int Rank = [] {
    if constexpr (is_input_spec_v<Spec>) return Spec::InputTensor::Ndim;
    else return Spec::OutputTensor::Ndim;
  }();
  static_assert(1 <= N && N <= Rank);
  if constexpr (N == Rank) return spec;
  else return take_leading<N>(slice_view<N>(spec, 0));
}

/** @brief Keep the last N Spec dimensions at zero on leading axes. */
template <int N, typename Spec>
  requires (InputSpecLike<Spec> || OutputSpecLike<Spec>)
VECOPS_INLINE auto take_trailing(const Spec& spec) {
  constexpr int Rank = [] {
    if constexpr (is_input_spec_v<Spec>) return Spec::InputTensor::Ndim;
    else return Spec::OutputTensor::Ndim;
  }();
  static_assert(1 <= N && N <= Rank);
  if constexpr (N == Rank) return spec;
  else return take_trailing<N>(slice_view<0>(spec, 0));
}

template <typename View>
  requires (is_tensor_v<std::remove_cvref_t<View>> ||
            is_input_spec_v<View> || is_output_spec_v<View> ||
            requires(const View& view) { view.spec(); })
VECOPS_INLINE constexpr decltype(auto) logical_layout(const View& view) {
  using V = std::remove_cvref_t<View>;
  if constexpr (is_tensor_v<V>) return view.layout();
  else if constexpr (is_input_spec_v<V>) return view.input_layout();
  else if constexpr (is_output_spec_v<V>) return view.output_layout();
  else {
    const auto& spec = view.spec();
    if constexpr (std::remove_cvref_t<decltype(spec)>::is_input) {
      return spec.input_layout();
    } else {
      return spec.output_layout();
    }
  }
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
  static_assert(
      is_bound_input_spec_v<Spec>,
      "InputDataAccess requires a bound input Spec; rebind the planning "
      "pattern before execution");
  using SpecType = Spec;
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  using Transform = typename Spec::TransformType;
  static constexpr int Rank = Spec::InputTensor::Ndim;

  VECOPS_ALWAYS_INLINE InputDataAccess(const Spec& spec, Policy policy = {})
      : spec_(&spec), data_(spec.tensor().data()), policy_(policy) {
    static_assert(Policy::vector_axis < Rank);
    static_assert(
        !details::is_unordered_policy_v<Policy> ||
            (Policy::permutation_safe &&
             details::transform_permutation_equivariant_v<Transform>),
        "unordered input conversion requires a permutation-safe kernel and transform");
  }

  /**
   * @brief Load one ComputeType value at an exact logical coordinate.
   *
   * Scalar access performs direct pointer traffic and supports only
   * NoTransform, IdentityVecTransform, and ZeroVecTransform. Conversion order
   * and value options may override operand defaults; lane, addressing,
   * population, and vector-memory options are intentionally unavailable.
   * `materialize::populate` remains valid for deferred input sessions and is a
   * no-op on this direct session.
   */
  template <typename... Options>
    requires details::scalar_access_transform_v<Transform>
  VECOPS_ALWAYS_INLINE ComputeType load_scalar(
      const Coord<Rank>& position, Options&&... options) const {
    details::validate_scalar_access_options<true, Options...>();
    constexpr bool HasAccessDefaults =
        details::has_access_default_option_v<Options...>;
    if constexpr (HasAccessDefaults) {
      using Planning = typename Policy::PlanningPolicyType;
      using Defaults = details::OverrideAccessDefaults<
          typename Policy::AccessDefaultsType, Options...>;
      using CallPolicy = details::AccessLoweringPolicy<
          Planning, Defaults, typename Policy::ActiveResources>;
      InputDataAccess<Spec, CallPolicy> access{
          *spec_, CallPolicy{static_cast<const Planning&>(policy_)}};
      auto retained = details::non_access_default_options(
          std::forward<Options>(options)...);
      auto invoke = [&](auto&&... retained_options)
          VECOPS_INLINE_LAMBDA -> ComputeType {
        return access.load_scalar(
            position,
            std::forward<decltype(retained_options)>(retained_options)...);
      };
      return details::apply_inline(invoke, retained);
    } else {
      details::validate_scalar_position(position, spec_->input_layout());
      if constexpr (is_zero_vec_transform_v<Transform>) {
        return ComputeType{};
      } else {
        const nint_t base = offset_at(spec_->input_layout(), position);
        if constexpr (details::is_no_transform_v<Transform>) {
          return details::scalar_policy_convert<Policy, ComputeType>(
              data_[base]);
        } else {
          using Intermediate = typename Transform::TIn;
          const Intermediate transformed =
              details::scalar_policy_convert<Policy, Intermediate>(
                  data_[base]);
          return details::scalar_policy_convert<Policy, ComputeType>(
              transformed);
        }
      }
    }
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
    constexpr bool HasAccessDefaults =
        details::has_access_default_option_v<Options...>;
    constexpr bool HasPopulate =
        vec::details::option_count_v<
            details::IsMaterializePopulate, Options...> != 0;
    if constexpr (HasAccessDefaults) {
      static_assert(
          vec::details::option_count_v<
              details::IsConversionOrderOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsConversionValueOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsTemporalityOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsAlignmentOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<details::IsPackingOption, Options...> == 0,
          "packing options are only valid for tensor stores");
      using Planning = typename Policy::PlanningPolicyType;
      using Defaults = details::OverrideAccessDefaults<
          typename Policy::AccessDefaultsType, Options...>;
      using CallPolicy = details::AccessLoweringPolicy<
          Planning, Defaults, typename Policy::ActiveResources>;
      InputDataAccess<Spec, CallPolicy> access{
          *spec_, CallPolicy{static_cast<const Planning&>(policy_)}};
      auto retained = details::non_access_default_options(
          std::forward<Options>(options)...);
      auto invoke = [&](auto&&... retained_options)
          VECOPS_INLINE_LAMBDA -> vec::Vec<Tag> {
        return access.load(
            tag, position, Axis<Dim>{},
            std::forward<decltype(retained_options)>(retained_options)...);
      };
      return details::apply_inline(invoke, retained);
    } else if constexpr (HasPopulate) {
      static_assert(
          vec::details::option_count_v<
              details::IsMaterializePopulate, Options...> == 1);
      auto retained = details::non_materialize_options(
          std::forward<Options>(options)...);
      auto invoke = [&](auto&&... retained_options)
          VECOPS_INLINE_LAMBDA -> vec::Vec<Tag> {
        return load(
            tag, position, Axis<Dim>{},
            std::forward<decltype(retained_options)>(retained_options)...);
      };
      return details::apply_inline(invoke, retained);
    } else {
    static_assert(0 <= Dim && Dim < Rank);
    details::validate_access_options<true, Options...>();
    const auto& tensor = spec_->tensor();
    using StrideMeta = stride_type_t<Dim, typename Spec::InputLayout>;
    constexpr bool UnitRankOne =
        Rank == 1 && details::is_definitely_one_meta_v<StrideMeta>;
    const nint_t base = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (UnitRankOne) return position[0];
      else return offset_at(tensor.layout(), position);
    }();
    const StrideMeta axis_stride = [&]() VECOPS_INLINE_LAMBDA -> StrideMeta {
      if constexpr (UnitRankOne) {
        return StrideMeta{nint_t{1}};
      } else {
        return StrideMeta{stride<Dim>(tensor.layout())};
      }
    }();

    if constexpr (details::is_no_transform_v<Transform>) {
      return details::load_memory<Tag, const MemoryElement*, Policy>(
          tag, data_ + base, axis_stride,
          std::forward<Options>(options)...);
    } else if constexpr (is_zero_vec_transform_v<Transform>) {
      // A zero prologue does not depend on memory, its dtype, addressing or
      // transform context.  In particular, do not send mixed-width zeros
      // through scalable-SVE transform chunking: a narrower transform input
      // can have no representable sizeless subword tag even though the output
      // Tag itself is legal.  Generate the compute-domain zero directly and
      // apply only the caller's inactive-lane population contract.
      return details::populate_inactive(
          tag, vec::zeros(tag), std::forward<Options>(options)...);
    } else {
      return details::with_transform_context(
          tag, position, Dim, spec_->projection(),
          [&](const auto& context) VECOPS_INLINE_LAMBDA {
            vec::Vec<Tag> result;
            constexpr bool NeedsLogicalMaskLowering =
                (vec::details::option_count_v<
                     vec::details::IsMaskedOption, Options...> != 0 ||
                 vec::details::option_count_v<
                     vec::details::IsVectorMergeOption, Options...> != 0 ||
                 vec::details::option_count_v<
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
                if constexpr (details::transform_reads_input_v<Transform>) {
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

  VECOPS_ALWAYS_INLINE const Spec& spec() const { return *spec_; }
  VECOPS_ALWAYS_INLINE const Policy& policy() const { return policy_; }
  VECOPS_ALWAYS_INLINE const MemoryElement* raw_data() const { return data_; }
  VECOPS_ALWAYS_INLINE Coord<Rank> raw_strides() const {
    Coord<Rank> result{};
    VECOPS_UNROLL
    for (int d = 0; d < Rank; ++d) {
      result[d] = spec_->input_layout().strides()[d];
    }
    return result;
  }

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
  static_assert(
      is_bound_output_spec_v<Spec>,
      "OutputDataAccess requires a bound output Spec; rebind the planning "
      "pattern before execution");
  using SpecType = Spec;
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  using Transform = typename Spec::TransformType;
  static constexpr int Rank = Spec::OutputTensor::Ndim;

  VECOPS_ALWAYS_INLINE OutputDataAccess(const Spec& spec, Policy policy = {})
      : spec_(&spec), data_(spec.tensor().data()), policy_(policy) {
    static_assert(Policy::vector_axis < Rank);
    static_assert(
        !details::is_unordered_policy_v<Policy> ||
            (Policy::permutation_safe &&
             details::transform_permutation_equivariant_v<Transform>),
        "unordered output conversion requires a permutation-safe kernel and transform");
  }

  OutputDataAccess(const OutputDataAccess&) = delete;
  OutputDataAccess& operator=(const OutputDataAccess&) = delete;
  OutputDataAccess(OutputDataAccess&&) = default;
  OutputDataAccess& operator=(OutputDataAccess&&) = default;

  /**
   * @brief Store one ComputeType value at an exact logical coordinate.
   *
   * Scalar access performs direct pointer traffic and supports only
   * NoTransform, IdentityVecTransform, and ZeroVecTransform. Conversion order
   * and value options may override operand defaults; vector-only access and
   * memory options are rejected.
   */
  template <typename... Options>
    requires details::scalar_access_transform_v<Transform>
  VECOPS_ALWAYS_INLINE void store_scalar(
      const Coord<Rank>& position,
      ComputeType value,
      Options&&... options) const {
    details::validate_scalar_access_options<false, Options...>();
    constexpr bool HasAccessDefaults =
        details::has_access_default_option_v<Options...>;
    if constexpr (HasAccessDefaults) {
      using Planning = typename Policy::PlanningPolicyType;
      using Defaults = details::OverrideAccessDefaults<
          typename Policy::AccessDefaultsType, Options...>;
      using CallPolicy = details::AccessLoweringPolicy<
          Planning, Defaults, typename Policy::ActiveResources>;
      OutputDataAccess<Spec, CallPolicy> access{
          *spec_, CallPolicy{static_cast<const Planning&>(policy_)}};
      auto retained = details::non_access_default_options(
          std::forward<Options>(options)...);
      auto invoke = [&](auto&&... retained_options) VECOPS_INLINE_LAMBDA {
        access.store_scalar(
            position, value,
            std::forward<decltype(retained_options)>(retained_options)...);
      };
      details::apply_inline(invoke, retained);
    } else {
      details::validate_scalar_position(position, spec_->output_layout());
      const nint_t base = offset_at(spec_->output_layout(), position);
      if constexpr (details::is_no_transform_v<Transform>) {
        data_[base] =
            details::scalar_policy_convert<Policy, MemoryElement>(value);
      } else if constexpr (is_zero_vec_transform_v<Transform>) {
        data_[base] = details::scalar_policy_convert<
            Policy, MemoryElement>(typename Transform::TOut{});
      } else {
        using Intermediate = typename Transform::TIn;
        const Intermediate transformed =
            details::scalar_policy_convert<Policy, Intermediate>(value);
        data_[base] = details::scalar_policy_convert<Policy, MemoryElement>(
            transformed);
      }
    }
  }

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
    constexpr bool HasAccessDefaults =
        details::has_access_default_option_v<Options...>;
    if constexpr (HasAccessDefaults) {
      static_assert(
          vec::details::option_count_v<
              details::IsConversionOrderOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsConversionValueOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsTemporalityOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<details::IsPackingOption, Options...> <= 1);
      static_assert(
          vec::details::option_count_v<
              details::IsAlignmentOption, Options...> <= 1);
      using Planning = typename Policy::PlanningPolicyType;
      using Defaults = details::OverrideAccessDefaults<
          typename Policy::AccessDefaultsType, Options...>;
      using CallPolicy = details::AccessLoweringPolicy<
          Planning, Defaults, typename Policy::ActiveResources>;
      OutputDataAccess<Spec, CallPolicy> access{
          *spec_, CallPolicy{static_cast<const Planning&>(policy_)}};
      auto retained = details::non_access_default_options(
          std::forward<Options>(options)...);
      auto invoke = [&](auto&&... retained_options) VECOPS_INLINE_LAMBDA {
        access.store(
            tag, position, Axis<Dim>{}, value,
            std::forward<decltype(retained_options)>(retained_options)...);
      };
      details::apply_inline(invoke, retained);
    } else {
    static_assert(0 <= Dim && Dim < Rank);
    details::validate_access_options<false, Options...>();
    const auto& tensor = spec_->tensor();
    using StrideMeta = stride_type_t<Dim, typename Spec::OutputLayout>;
    constexpr bool UnitRankOne =
        Rank == 1 && details::is_definitely_one_meta_v<StrideMeta>;
    const nint_t base = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (UnitRankOne) return position[0];
      else return offset_at(tensor.layout(), position);
    }();
    const StrideMeta axis_stride = [&]() VECOPS_INLINE_LAMBDA -> StrideMeta {
      if constexpr (UnitRankOne) {
        return StrideMeta{nint_t{1}};
      } else {
        return StrideMeta{stride<Dim>(tensor.layout())};
      }
    }();
    if constexpr (details::is_no_transform_v<Transform>) {
      details::store_memory<Tag, MemoryElement*, Policy>(
          tag, data_ + base, value, axis_stride,
          std::forward<Options>(options)...);
    } else {
      details::with_transform_context(
          tag, position, Dim, spec_->projection(),
          [&](const auto& context) VECOPS_INLINE_LAMBDA {
            vec::Vec<Tag> result;
            constexpr bool NeedsLogicalMaskLowering =
                vec::details::option_count_v<
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
  }

  /**
   * Apply a lane-local transform independently to two adjacent compute
   * vectors, then concatenate only the transformed results for one packed
   * converting store.  This separates transform width from physical memory
   * width: backends such as VLA SVE can avoid a P1 transform tuple while still
   * filling one complete BF16 memory word.
   *
   * The two logical vector intervals must be physically adjacent along a
   * compile-time unit-stride axis.  Fullness is a template property so the
   * common full/full path retains unmasked transform contexts and store code.
   */
  template <bool FirstFull, bool SecondFull, vec::VectorTag Tag, int Dim, typename FirstActive, typename SecondActive>
    requires std::same_as<vec::ElementOf<Tag>, ComputeType> && (!details::is_no_transform_v<Transform>)
  VECOPS_ALWAYS_INLINE
    void store_transform_pair_coalesced(Tag tag, const Coord<Rank>& first_position, const Coord<Rank>& second_position,
                                        Axis<Dim>, vec::Vec<Tag> first_value, vec::Vec<Tag> second_value,
                                        FirstActive first_active, SecondActive second_active) const {
    static_assert(0 <= Dim && Dim < Rank);
    using StrideMeta = stride_type_t<Dim, typename Spec::OutputLayout>;
    static_assert(details::is_definitely_one_meta_v<StrideMeta>,
                  "coalesced transform stores require a unit-stride physical axis");
    using TransformInTag = vec::Rebind<typename Transform::TIn, Tag>;
    using TransformOutTag = vec::Rebind<typename Transform::TOut, Tag>;
    using WideTransformOutTag = vec::Twice<TransformOutTag>;

    const auto apply_one = [&]<typename ActiveOption>(const Coord<Rank>& position, vec::Vec<Tag> value,
                                                      ActiveOption active) VECOPS_INLINE_LAMBDA {
      return details::with_transform_context(
        tag, position, Dim, spec_->projection(),
        [&](const auto& context) VECOPS_INLINE_LAMBDA {
          static_assert(details::can_transform_chunk<Transform, Tag, decltype(context)>());
          const auto transform_input =
            vec::convert(TransformInTag{}, tag, value, typename Policy::ConversionOrderOption{},
                         typename Policy::ConversionValueOption{});
          return spec_->transform()(TransformOutTag{}, transform_input, context);
        },
        active);
    };

    const auto first_transformed = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FirstFull) {
        return apply_one(first_position, first_value, vec::opt::unmasked);
      } else {
        return apply_one(first_position, first_value, vec::opt::first(static_cast<nint_t>(first_active)));
      }
    }();
    const auto second_transformed = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (SecondFull) {
        return apply_one(second_position, second_value, vec::opt::unmasked);
      } else {
        return apply_one(second_position, second_value, vec::opt::first(static_cast<nint_t>(second_active)));
      }
    }();
    const auto transformed = vec::concat(WideTransformOutTag{}, first_transformed, second_transformed);

    const auto& tensor = spec_->tensor();
    const nint_t first_base = offset_at(tensor.layout(), first_position);
    VECOPS_ASSERT(offset_at(tensor.layout(), second_position) - first_base == vec::size(tag),
                  "coalesced transform vectors must be physically adjacent");
    const StrideMeta axis_stride{stride<Dim>(tensor.layout())};
    if constexpr (FirstFull && SecondFull) {
      details::store_memory<WideTransformOutTag, MemoryElement*, Policy>(WideTransformOutTag{}, data_ + first_base,
                                                                         transformed, axis_stride, vec::opt::unmasked);
    } else {
      details::store_memory<WideTransformOutTag, MemoryElement*, Policy>(
        WideTransformOutTag{}, data_ + first_base, transformed, axis_stride,
        vec::opt::first(static_cast<nint_t>(first_active) + static_cast<nint_t>(second_active)));
    }
  }

  template <vec::VectorTag Tag, int Dim, typename Mapping = ContiguousLaneMapping>
    requires(std::same_as<Mapping, ContiguousLaneMapping> || std::same_as<Mapping, AffineLaneMapping>)
  VECOPS_INLINE auto scan(Tag tag, Coord<Rank> origin, Axis<Dim>, nint_t count, Mapping mapping = {}) {
    return ScanCursor<OutputDataAccess, Tag, Dim, Mapping>{*this, tag, origin, count, mapping};
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
  VECOPS_ALWAYS_INLINE void commit() { committed_ = true; }

  /** @brief Whether this direct session has been explicitly committed. */
  bool committed() const { return committed_; }

  VECOPS_ALWAYS_INLINE const Spec& spec() const { return *spec_; }
  VECOPS_ALWAYS_INLINE const Policy& policy() const { return policy_; }
  VECOPS_ALWAYS_INLINE MemoryElement* raw_data() const { return data_; }
  VECOPS_ALWAYS_INLINE Coord<Rank> raw_strides() const {
    Coord<Rank> result{};
    VECOPS_UNROLL
    for (int d = 0; d < Rank; ++d) {
      result[d] = spec_->output_layout().strides()[d];
    }
    return result;
  }

private:
  const Spec* spec_;
  MemoryElement* data_;
  [[no_unique_address]] Policy policy_;
  bool committed_ = false;
};

template <typename Resources, typename Spec,
          typename Planning, typename Defaults, typename PreviousResources>
VECOPS_ALWAYS_INLINE auto rebind_active_resources(
    const InputDataAccess<
        Spec, details::AccessLoweringPolicy<
                  Planning, Defaults, PreviousResources>>& access) {
  using Policy = details::AccessLoweringPolicy<
      Planning, Defaults, Resources>;
  return InputDataAccess<Spec, Policy>{
      access.spec(), Policy{static_cast<const Planning&>(access.policy())}};
}

template <typename Resources, typename Spec,
          typename Planning, typename Defaults, typename PreviousResources>
VECOPS_ALWAYS_INLINE auto rebind_active_resources(
    OutputDataAccess<
        Spec, details::AccessLoweringPolicy<
                  Planning, Defaults, PreviousResources>>& access) {
  using Policy = details::AccessLoweringPolicy<
      Planning, Defaults, Resources>;
  return OutputDataAccess<Spec, Policy>{
      access.spec(), Policy{static_cast<const Planning&>(access.policy())}};
}

/**
 * Input session whose canonical ComputeType materialization is populated by
 * loads carrying `tensor::materialize::populate`.
 */
template <typename OriginalSpec, typename AuxSpec, typename SourcePolicy>
class DeferredMaterializedInputDataAccess {
public:
  using ComputeType = typename OriginalSpec::ComputeType;
  using MemoryElement = typename OriginalSpec::MemoryElement;
  using Transform = typename OriginalSpec::TransformType;
  static constexpr int Rank = OriginalSpec::InputTensor::Ndim;
  static constexpr bool is_input = true;
  static constexpr bool is_canonical_compute_materialized = true;

  using CachePlanningPolicy = InputAccessPolicy<
      SourcePolicy::vector_axis, SourcePolicy::read_passes,
      AccessPlan::direct>;
  using CachePolicy = details::AccessLoweringPolicy<
      CachePlanningPolicy, DefaultAccessDefaults,
      typename SourcePolicy::ActiveResources>;

  DeferredMaterializedInputDataAccess(
      OriginalSpec original, AuxSpec auxiliary, SourcePolicy policy)
      : original_(std::move(original)), auxiliary_(std::move(auxiliary)),
        source_policy_(policy) {}

  template <typename... Options>
    requires details::scalar_access_transform_v<Transform>
  VECOPS_ALWAYS_INLINE ComputeType load_scalar(
      const Coord<Rank>& position, Options&&... options) const {
    details::validate_scalar_access_options<true, Options...>();
    constexpr bool Populate =
        vec::details::option_count_v<
            details::IsMaterializePopulate, Options...> != 0;
    auto retained = details::non_materialize_options(
        std::forward<Options>(options)...);
    auto invoke = [&](auto&&... access_options)
        VECOPS_INLINE_LAMBDA -> ComputeType {
      if constexpr (Populate) {
#if defined(VECOPS_DEBUG)
        VECOPS_ASSERT(!reuse_started_,
                      "deferred materialization populated after reuse began");
        populate_started_ = true;
#endif
        InputDataAccess<OriginalSpec, SourcePolicy> source{
            original_, source_policy_};
        const ComputeType value = source.load_scalar(
            position, access_options...);
        auto cache_output_spec = output<ComputeType>(auxiliary_.tensor());
        OutputDataAccess<decltype(cache_output_spec), CachePolicy> cache{
            cache_output_spec, CachePolicy{CachePlanningPolicy{}}};
        auto store_options = details::cache_store_options(access_options...);
        auto store = [&](auto&&... cache_options) VECOPS_INLINE_LAMBDA {
          cache.store_scalar(position, value, cache_options...);
        };
        details::apply_inline(store, store_options);
        cache.commit();
        return value;
      } else {
        static_assert(
            vec::details::option_count_v<
                details::IsConversionOrderOption,
                decltype(access_options)...> == 0 &&
            vec::details::option_count_v<
                details::IsConversionValueOption,
                decltype(access_options)...> == 0,
            "canonical Compute materialization cannot be reinterpreted");
#if defined(VECOPS_DEBUG)
        VECOPS_ASSERT(populate_started_,
                      "deferred materialization reused before population");
        reuse_started_ = true;
#endif
        InputDataAccess<AuxSpec, CachePolicy> cache{
            auxiliary_, CachePolicy{CachePlanningPolicy{}}};
        return cache.load_scalar(position, access_options...);
      }
    };
    return details::apply_inline(invoke, retained);
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE vec::Vec<Tag> load(
      Tag tag, const Coord<Rank>& position, Options&&... options) const {
    return load(
        tag, position, axis<Rank - 1>,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE vec::Vec<Tag> load(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      Options&&... options) const {
    constexpr bool Populate =
        vec::details::option_count_v<
            details::IsMaterializePopulate, Options...> != 0;
    static_assert(
        vec::details::option_count_v<
            details::IsMaterializePopulate, Options...> <= 1);
    auto retained = details::non_materialize_options(
        std::forward<Options>(options)...);
    auto invoke = [&](auto&&... access_options)
        VECOPS_INLINE_LAMBDA -> vec::Vec<Tag> {
      if constexpr (Populate) {
        static_assert(
            vec::details::option_count_v<
                vec::details::IsUnorderedOption,
                decltype(access_options)...> == 0,
            "canonical deferred materialization must be populated ordered");
#if defined(VECOPS_DEBUG)
        VECOPS_ASSERT(!reuse_started_,
                      "deferred materialization populated after reuse began");
        populate_started_ = true;
#endif
        InputDataAccess<OriginalSpec, SourcePolicy> source{
            original_, source_policy_};
        auto value = source.load(
            tag, position, dim, access_options...);
        auto cache_output_spec = output<ComputeType>(auxiliary_.tensor());
        OutputDataAccess<decltype(cache_output_spec), CachePolicy> cache{
            cache_output_spec,
            CachePolicy{CachePlanningPolicy{}}};
        auto store_options = details::cache_store_options(access_options...);
        auto store = [&](auto&&... options) VECOPS_INLINE_LAMBDA {
          cache.store(tag, position, dim, value, options...);
        };
        details::apply_inline(store, store_options);
        cache.commit();
        return value;
      } else {
        static_assert(
            vec::details::option_count_v<
                details::IsConversionOrderOption,
                decltype(access_options)...> == 0 &&
            vec::details::option_count_v<
                details::IsConversionValueOption,
                decltype(access_options)...> == 0,
            "canonical Compute materialization cannot be reinterpreted");
#if defined(VECOPS_DEBUG)
        VECOPS_ASSERT(populate_started_,
                      "deferred materialization reused before population");
        reuse_started_ = true;
#endif
        InputDataAccess<AuxSpec, CachePolicy> cache{
            auxiliary_, CachePolicy{CachePlanningPolicy{}}};
        return cache.load(tag, position, dim, access_options...);
      }
    };
    return details::apply_inline(invoke, retained);
  }

  template <vec::VectorTag Tag, int Dim,
            typename Mapping = ContiguousLaneMapping>
  VECOPS_INLINE auto scan(
      Tag tag, Coord<Rank> origin, Axis<Dim>, nint_t count,
      Mapping mapping = {}) {
    return ScanCursor<
        DeferredMaterializedInputDataAccess, Tag, Dim, Mapping>{
            *this, tag, origin, count, mapping};
  }

  template <vec::VectorTag Tag, int TraverseDim, int VectorDim>
  VECOPS_INLINE auto project(
      Tag tag, Coord<Rank> origin, TraversalAxis<TraverseDim>,
      VectorAxis<VectorDim>, nint_t iterations,
      nint_t traversal_step = 1) {
    return ProjectCursor<
        DeferredMaterializedInputDataAccess, Tag, TraverseDim, VectorDim>{
            *this, tag, origin, iterations, traversal_step};
  }

  const AuxSpec& spec() const { return auxiliary_; }
  const OriginalSpec& original_spec() const { return original_; }
  const SourcePolicy& policy() const { return source_policy_; }

private:
  OriginalSpec original_;
  AuxSpec auxiliary_;
  [[no_unique_address]] SourcePolicy source_policy_;
#if defined(VECOPS_DEBUG)
  mutable bool populate_started_ = false;
  mutable bool reuse_started_ = false;
#endif
};

/**
 * Fully prepared canonical ComputeType input materialization.
 *
 * Its auxiliary tensor is native NoTransform storage, so raw_data() and
 * raw_strides() deliberately expose it to matrix/tile instructions without a
 * second materialization pass.
 */
template <typename AuxSpec, typename CachePolicy>
class CanonicalMaterializedInputDataAccess {
public:
  using ComputeType = typename AuxSpec::ComputeType;
  using MemoryElement = typename AuxSpec::MemoryElement;
  using Transform = typename AuxSpec::TransformType;
  static constexpr int Rank = AuxSpec::InputTensor::Ndim;
  static constexpr bool is_input = true;
  static constexpr bool is_canonical_compute_materialized = true;

  VECOPS_ALWAYS_INLINE CanonicalMaterializedInputDataAccess(
      AuxSpec auxiliary, CachePolicy policy)
      : auxiliary_(std::move(auxiliary)), policy_(policy) {}

  template <typename... Options>
    requires details::scalar_access_transform_v<Transform>
  VECOPS_ALWAYS_INLINE ComputeType load_scalar(
      const Coord<Rank>& position, Options&&... options) const {
    static_assert(
        vec::details::option_count_v<
            details::IsConversionOrderOption, Options...> == 0 &&
        vec::details::option_count_v<
            details::IsConversionValueOption, Options...> == 0,
        "canonical Compute materialization cannot be reinterpreted");
    InputDataAccess<AuxSpec, CachePolicy> cache{auxiliary_, policy_};
    return cache.load_scalar(
        position, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Options&&... options) const {
    return load(
        tag, position, axis<Rank - 1>,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      Options&&... options) const {
    static_assert(
        vec::details::option_count_v<
            details::IsConversionOrderOption, Options...> == 0 &&
        vec::details::option_count_v<
            details::IsConversionValueOption, Options...> == 0,
        "canonical Compute materialization cannot be reinterpreted");
    InputDataAccess<AuxSpec, CachePolicy> cache{auxiliary_, policy_};
    return cache.load(
        tag, position, dim, std::forward<Options>(options)...);
  }

  const AuxSpec& spec() const { return auxiliary_; }
  const CachePolicy& policy() const { return policy_; }
  VECOPS_ALWAYS_INLINE const MemoryElement* raw_data() const {
    return auxiliary_.tensor().data();
  }
  VECOPS_ALWAYS_INLINE Coord<Rank> raw_strides() const {
    Coord<Rank> result{};
    VECOPS_UNROLL
    for (int d = 0; d < Rank; ++d)
      result[d] = auxiliary_.input_layout().strides()[d];
    return result;
  }

private:
  AuxSpec auxiliary_;
  [[no_unique_address]] CachePolicy policy_;
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
 * outlive all borrowed views and remains the sole commit authority. Raw
 * pointer/stride access describes the borrowed Spec itself; consumers must
 * still require NoTransform and matching memory/compute types before bypassing
 * load/store.
 */
template <typename Spec, typename Policy>
class BorrowedDataAccess {
public:
  using ComputeType = typename Spec::ComputeType;
  using MemoryElement = typename Spec::MemoryElement;
  using Transform = typename Spec::TransformType;
  static constexpr bool IsInput = Spec::is_input;
  static constexpr int Rank = [] {
    if constexpr (IsInput) return Spec::InputTensor::Ndim;
    else return Spec::OutputTensor::Ndim;
  }();

  BorrowedDataAccess(Spec spec, Policy policy)
      : spec_(std::move(spec)), policy_(policy) {}

  template <typename... Options>
    requires (IsInput && details::scalar_access_transform_v<Transform>)
  VECOPS_ALWAYS_INLINE ComputeType load_scalar(
      const Coord<Rank>& position, Options&&... options) const {
    InputDataAccess<Spec, Policy> access{spec_, policy_};
    return access.load_scalar(
        position, std::forward<Options>(options)...);
  }

  template <typename... Options>
    requires (!IsInput && details::scalar_access_transform_v<Transform>)
  VECOPS_ALWAYS_INLINE void store_scalar(
      const Coord<Rank>& position,
      ComputeType value,
      Options&&... options) const {
    OutputDataAccess<Spec, Policy> access{spec_, policy_};
    access.store_scalar(
        position, value, std::forward<Options>(options)...);
  }

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
    static_assert(
        vec::details::option_count_v<
            details::IsConversionOrderOption, Options...> == 0,
        "with_unordered_access owns the conversion-order option");
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
    static_assert(
        vec::details::option_count_v<
            details::IsConversionOrderOption, Options...> == 0,
        "with_unordered_access owns the conversion-order option");
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
  VECOPS_ALWAYS_INLINE decltype(auto) raw_data() const {
    return spec_.tensor().data();
  }
  VECOPS_ALWAYS_INLINE Coord<Rank> raw_strides() const {
    Coord<Rank> result{};
    VECOPS_UNROLL
    for (int d = 0; d < Rank; ++d) {
      if constexpr (IsInput)
        result[d] = spec_.input_layout().strides()[d];
      else
        result[d] = spec_.output_layout().strides()[d];
    }
    return result;
  }

private:
  Spec spec_;
  [[no_unique_address]] Policy policy_;
};

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const InputDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = slice_access_policy_t<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int I, int J, typename Spec, typename Policy>
VECOPS_INLINE auto transpose_view(
    const InputDataAccess<Spec, Policy>& access) {
  auto transposed = transpose_view<I, J>(access.spec());
  using TransposedPolicy = transpose_access_policy_t<Policy, I, J>;
  return BorrowedDataAccess<decltype(transposed), TransposedPolicy>{
      std::move(transposed), TransposedPolicy{}};
}

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const OutputDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = slice_access_policy_t<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int I, int J, typename Spec, typename Policy>
VECOPS_INLINE auto transpose_view(
    const OutputDataAccess<Spec, Policy>& access) {
  auto transposed = transpose_view<I, J>(access.spec());
  using TransposedPolicy = transpose_access_policy_t<Policy, I, J>;
  return BorrowedDataAccess<decltype(transposed), TransposedPolicy>{
      std::move(transposed), TransposedPolicy{}};
}

template <int Dim, typename Spec, typename Policy>
VECOPS_INLINE auto slice_view(
    const BorrowedDataAccess<Spec, Policy>& access, nint_t index) {
  auto sliced = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = slice_access_policy_t<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int I, int J, typename Spec, typename Policy>
VECOPS_INLINE auto transpose_view(
    const BorrowedDataAccess<Spec, Policy>& access) {
  auto transposed = transpose_view<I, J>(access.spec());
  using TransposedPolicy = transpose_access_policy_t<Policy, I, J>;
  return BorrowedDataAccess<decltype(transposed), TransposedPolicy>{
      std::move(transposed), TransposedPolicy{}};
}

template <int Dim, typename AuxSpec, typename CachePolicy>
VECOPS_INLINE auto slice_view(
    const CanonicalMaterializedInputDataAccess<AuxSpec, CachePolicy>& access,
    nint_t index) {
  auto auxiliary = slice_view<Dim>(access.spec(), index);
  using SlicedPolicy = slice_access_policy_t<CachePolicy, Dim>;
  return CanonicalMaterializedInputDataAccess<
      decltype(auxiliary), SlicedPolicy>{
      std::move(auxiliary), SlicedPolicy{}};
}

template <int I, int J, typename AuxSpec, typename CachePolicy>
VECOPS_INLINE auto transpose_view(
    const CanonicalMaterializedInputDataAccess<AuxSpec, CachePolicy>& access) {
  auto auxiliary = transpose_view<I, J>(access.spec());
  using TransposedPolicy = transpose_access_policy_t<CachePolicy, I, J>;
  return CanonicalMaterializedInputDataAccess<
      decltype(auxiliary), TransposedPolicy>{
      std::move(auxiliary), TransposedPolicy{}};
}

// DeferredMaterializedInputDataAccess intentionally has no structural-view
// overload. Its population/reuse phase state belongs to the complete bound
// region; create sliced or transposed Specs before binding it.

template <int N, typename Access>
  requires requires(const Access& access) {
    Access::Rank;
    access.spec();
    tensor::slice_view<N>(access, 0);
  }
VECOPS_INLINE decltype(auto) take_leading(const Access& access) {
  static_assert(1 <= N && N <= Access::Rank);
  if constexpr (N == Access::Rank) return (access);
  else return take_leading<N>(slice_view<N>(access, 0));
}

template <int N, typename Access>
  requires requires(const Access& access) {
    Access::Rank;
    access.spec();
    tensor::slice_view<0>(access, 0);
  }
VECOPS_INLINE decltype(auto) take_trailing(const Access& access) {
  static_assert(1 <= N && N <= Access::Rank);
  if constexpr (N == Access::Rank) return (access);
  else return take_trailing<N>(slice_view<0>(access, 0));
}

/** A bound session exposing its Spec and kernel-owned access policy. */
template <typename T>
inline constexpr bool is_bound_tensor_access_v = requires(
    const std::remove_cvref_t<T>& access) {
  std::remove_cvref_t<T>::Rank;
  access.spec();
  access.policy();
};

template <typename T>
concept BoundTensorAccess = is_bound_tensor_access_v<T>;

template <typename T>
inline constexpr bool is_bound_input_access_v = [] {
  if constexpr (!is_bound_tensor_access_v<T>) {
    return false;
  } else {
    using Spec = std::remove_cvref_t<decltype(
        std::declval<const std::remove_cvref_t<T>&>().spec())>;
    return Spec::is_input;
  }
}();

template <typename T>
concept BoundInputAccess = is_bound_input_access_v<T>;

template <typename T>
inline constexpr bool is_bound_output_access_v = [] {
  if constexpr (!is_bound_tensor_access_v<T>) {
    return false;
  } else {
    using Spec = std::remove_cvref_t<decltype(
        std::declval<const std::remove_cvref_t<T>&>().spec())>;
    return !Spec::is_input;
  }
}();

template <typename T>
concept BoundOutputAccess = is_bound_output_access_v<T>;

template <typename T>
inline constexpr bool is_committable_bound_output_access_v =
    is_bound_output_access_v<T> &&
    requires(std::remove_cvref_t<T>& access) {
      access.commit();
    };

template <typename T>
concept CommittableBoundOutputAccess =
    is_committable_bound_output_access_v<T>;

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
 * The raw pointer/stride surface addresses the auxiliary hot storage, never
 * the original Tensor. It is a native ComputeType/NoTransform target only for
 * `materialize_before_transform`; `commit()` still owns the original
 * epilogue/conversion.
 *
 * @tparam Scope Execution scope copied from operand binding. It allows commit
 *         to reuse an enclosing hardware mode when layout write-back invokes
 *         Transpose2D; the scope remains a non-owning lexical proof.
 */
template <AccessPlan Plan, typename OriginalSpec, typename AuxSpec,
          typename Policy, typename Scope = execution::ExecutionSession>
class MaterializedOutputDataAccess {
  static_assert(
      Plan == AccessPlan::materialize_before_transform ||
      Plan == AccessPlan::materialize_after_transform);

public:
  using ComputeType = typename OriginalSpec::ComputeType;
  // The hot session addresses auxiliary storage. For before-transform
  // materialization this is native ComputeType/NoTransform storage and may be
  // consumed directly by matrix instructions; the original conversion stays
  // owned by commit().
  using MemoryElement = typename AuxSpec::MemoryElement;
  using Transform = typename AuxSpec::TransformType;
  static constexpr int Rank = OriginalSpec::OutputTensor::Ndim;

  VECOPS_ALWAYS_INLINE MaterializedOutputDataAccess(
      const OriginalSpec& original, AuxSpec auxiliary, Policy policy,
      Scope context = {})
      : original_(&original), auxiliary_(std::move(auxiliary)),
        policy_(policy), context_(context) {}

  MaterializedOutputDataAccess(const MaterializedOutputDataAccess&) = delete;
  MaterializedOutputDataAccess& operator=(const MaterializedOutputDataAccess&) = delete;

  VECOPS_ALWAYS_INLINE MaterializedOutputDataAccess(
      MaterializedOutputDataAccess&& other) noexcept
      : original_(other.original_), auxiliary_(std::move(other.auxiliary_)),
        policy_(other.policy_), context_(other.context_),
        committed_(other.committed_), owning_(true) {
    other.owning_ = false;
  }

  MaterializedOutputDataAccess& operator=(MaterializedOutputDataAccess&&) = delete;

  ~MaterializedOutputDataAccess() {
    VECOPS_ASSERT(!owning_ || committed_,
                  "materialized output session was destroyed without commit()");
  }

  template <typename... Options>
    requires details::scalar_access_transform_v<typename AuxSpec::TransformType>
  VECOPS_ALWAYS_INLINE void store_scalar(
      const Coord<Rank>& position,
      ComputeType value,
      Options&&... options) const {
    if constexpr (Plan == AccessPlan::materialize_before_transform) {
      static_assert(
          vec::details::option_count_v<
              details::IsConversionOrderOption, Options...> == 0 &&
          vec::details::option_count_v<
              details::IsConversionValueOption, Options...> == 0,
          "before-transform output conversion is selected at operand binding");
    }
    OutputDataAccess<AuxSpec, Policy> hot{auxiliary_, policy_};
    hot.store_scalar(
        position, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, vec::Vec<Tag> value,
      Options&&... options) const {
    if constexpr (Plan == AccessPlan::materialize_before_transform) {
      static_assert(
          vec::details::option_count_v<
              details::IsConversionOrderOption, Options...> == 0 &&
          vec::details::option_count_v<
              details::IsConversionValueOption, Options...> == 0,
          "before-transform output conversion is selected at operand binding");
    }
    OutputDataAccess<AuxSpec, Policy> hot{auxiliary_, policy_};
    hot.store(tag, position, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      vec::Vec<Tag> value, Options&&... options) const {
    if constexpr (Plan == AccessPlan::materialize_before_transform) {
      static_assert(
          vec::details::option_count_v<
              details::IsConversionOrderOption, Options...> == 0 &&
          vec::details::option_count_v<
              details::IsConversionValueOption, Options...> == 0,
          "before-transform output conversion is selected at operand binding");
    }
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
   * @note Idempotent. Must run before the materialized operand scope ends.
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
  VECOPS_ALWAYS_INLINE MemoryElement* raw_data() const {
    return auxiliary_.tensor().data();
  }
  VECOPS_ALWAYS_INLINE Coord<Rank> raw_strides() const {
    Coord<Rank> result{};
    VECOPS_UNROLL
    for (int d = 0; d < Rank; ++d)
      result[d] = auxiliary_.output_layout().strides()[d];
    return result;
  }

private:
  void commit_before() {
    constexpr int AxisValue = Policy::vector_axis;
    using Tag = vec::ScalableTag<ComputeType, 0>;
    using AuxInputPolicy = InputAccessPolicy<AxisValue, 1, AccessPlan::direct>;
    using AuxLoweringPolicy = details::AccessLoweringPolicy<
        AuxInputPolicy, DefaultAccessDefaults,
        typename Policy::ActiveResources>;
    auto aux_input_spec = input<ComputeType>(auxiliary_.tensor());
    InputDataAccess<decltype(aux_input_spec), AuxLoweringPolicy> source{
        aux_input_spec, AuxLoweringPolicy{AuxInputPolicy{}}};
    OutputDataAccess<OriginalSpec, Policy> destination{*original_, policy_};
    constexpr int UnitAxis = details::first_unit_axis<
        typename OriginalSpec::OutputLayout::Strides, AxisValue>();
    if constexpr (UnitAxis >= 0) {
      details::transpose_planes<
          UnitAxis, AxisValue, AxisValue, UnitAxis>(
          context_, original_->output_layout(), source, destination);
    } else {
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
                  tag, line, axis<AxisValue>, value,
                  vec::opt::first(active));
            }
          });
    }
    destination.commit();
  }

  void commit_after() {
    const auto& source = auxiliary_.tensor();
    auto& destination = original_->tensor();
    constexpr int AxisValue = Policy::vector_axis;
    constexpr int UnitAxis = details::first_unit_axis<
        typename OriginalSpec::OutputLayout::Strides, AxisValue>();
    if constexpr (UnitAxis >= 0) {
      using Memory = typename OriginalSpec::MemoryElement;
      auto source_spec = input<Memory>(source);
      auto destination_spec = output<Memory>(destination);
      using SourcePlanning = InputAccessPolicy<
          AxisValue, 1, AccessPlan::direct>;
      using DestinationPlanning = OutputAccessPolicy<
          UnitAxis, AccessPlan::direct>;
      using SourcePolicy = details::AccessLoweringPolicy<
          SourcePlanning, DefaultAccessDefaults,
          typename Policy::ActiveResources>;
      using DestinationPolicy = details::AccessLoweringPolicy<
          DestinationPlanning, DefaultAccessDefaults,
          typename Policy::ActiveResources>;
      InputDataAccess<decltype(source_spec), SourcePolicy> source_access{
          source_spec, SourcePolicy{SourcePlanning{}}};
      OutputDataAccess<decltype(destination_spec), DestinationPolicy>
          destination_access{
              destination_spec,
              DestinationPolicy{DestinationPlanning{}}};
      details::transpose_planes<
          UnitAxis, AxisValue, AxisValue, UnitAxis>(
          context_, original_->output_layout(),
          source_access, destination_access);
      destination_access.commit();
    } else {
      Coord<Rank> position{};
      details::for_each_coordinate(
          original_->output_layout(), position,
          [&](const Coord<Rank>& current) VECOPS_INLINE_LAMBDA {
            destination.data()[offset_at(destination.layout(), current)] =
                source.data()[offset_at(source.layout(), current)];
          });
    }
  }

  const OriginalSpec* original_;
  AuxSpec auxiliary_;
  [[no_unique_address]] Policy policy_;
  [[no_unique_address]] Scope context_;
  bool committed_ = false;
  bool owning_ = true;
};

/**
 * @brief Slice a materialized output as a borrowed, non-committing view.
 * @note The parent session still must be committed after all slice use.
 */
template <int Dim, AccessPlan Plan, typename OriginalSpec, typename AuxSpec,
          typename Policy, typename Scope>
VECOPS_INLINE auto slice_view(
    const MaterializedOutputDataAccess<
        Plan, OriginalSpec, AuxSpec, Policy, Scope>& access,
    nint_t index) {
  auto sliced = slice_view<Dim>(access.auxiliary_spec(), index);
  using SlicedPolicy = slice_access_policy_t<Policy, Dim>;
  return BorrowedDataAccess<decltype(sliced), SlicedPolicy>{
      std::move(sliced), SlicedPolicy{}};
}

template <int I, int J, AccessPlan Plan, typename OriginalSpec,
          typename AuxSpec, typename Policy, typename Scope>
VECOPS_INLINE auto transpose_view(
    const MaterializedOutputDataAccess<
        Plan, OriginalSpec, AuxSpec, Policy, Scope>& access) {
  auto transposed = transpose_view<I, J>(access.auxiliary_spec());
  using TransposedPolicy = transpose_access_policy_t<Policy, I, J>;
  return BorrowedDataAccess<decltype(transposed), TransposedPolicy>{
      std::move(transposed), TransposedPolicy{}};
}

namespace details {

template <typename Order, typename Spec, typename Policy,
          vec::VectorTag Tag, int Dim, typename... Options>
VECOPS_ALWAYS_INLINE auto load_with_order(
    InputDataAccess<Spec, Policy>& access, Tag tag,
    const Coord<Spec::InputTensor::Ndim>& position, Axis<Dim> dim,
    Options&&... options) {
  using Planning = typename Policy::PlanningPolicyType;
  using Defaults = ReorderAccessDefaults<
      typename Policy::AccessDefaultsType, Order>;
  using ReorderedPolicy = AccessLoweringPolicy<
      Planning, Defaults, typename Policy::ActiveResources>;
  InputDataAccess<Spec, ReorderedPolicy> reordered{
      access.spec(), ReorderedPolicy{
          static_cast<const Planning&>(access.policy())}};
  return reordered.load(
      tag, position, dim, std::forward<Options>(options)...);
}

template <typename Order, typename OriginalSpec, typename AuxSpec,
          typename SourcePolicy, vec::VectorTag Tag, int Dim,
          typename... Options>
VECOPS_ALWAYS_INLINE auto load_with_order(
    DeferredMaterializedInputDataAccess<
        OriginalSpec, AuxSpec, SourcePolicy>& access,
    Tag tag, const Coord<OriginalSpec::InputTensor::Ndim>& position,
    Axis<Dim> dim, Options&&... options) {
  static_assert(std::same_as<Order, vec::cvt::Ordered>);
  return access.load(
      tag, position, dim, std::forward<Options>(options)...);
}

template <typename Order, typename Spec, typename Policy,
          vec::VectorTag Tag, int Dim, typename... Options>
VECOPS_ALWAYS_INLINE auto load_with_order(
    BorrowedDataAccess<Spec, Policy>& access, Tag tag,
    const Coord<BorrowedDataAccess<Spec, Policy>::Rank>& position,
    Axis<Dim> dim, Options&&... options)
  requires Spec::is_input
{
  using Planning = typename Policy::PlanningPolicyType;
  using Defaults = ReorderAccessDefaults<
      typename Policy::AccessDefaultsType, Order>;
  using ReorderedPolicy = AccessLoweringPolicy<
      Planning, Defaults, typename Policy::ActiveResources>;
  BorrowedDataAccess<Spec, ReorderedPolicy> reordered{
      access.spec(), ReorderedPolicy{
          static_cast<const Planning&>(access.policy())}};
  return reordered.load(
      tag, position, dim, std::forward<Options>(options)...);
}

template <typename Order, typename AuxSpec, typename CachePolicy,
          vec::VectorTag Tag, int Dim, typename... Options>
VECOPS_ALWAYS_INLINE auto load_with_order(
    CanonicalMaterializedInputDataAccess<AuxSpec, CachePolicy>& access,
    Tag tag, const Coord<AuxSpec::InputTensor::Ndim>& position,
    Axis<Dim> dim, Options&&... options) {
  static_assert(std::same_as<Order, vec::cvt::Ordered>);
  return access.load(
      tag, position, dim, std::forward<Options>(options)...);
}

template <typename Order, typename Spec, typename Policy,
          vec::VectorTag Tag, int Dim, typename... Options>
VECOPS_ALWAYS_INLINE void store_with_order(
    OutputDataAccess<Spec, Policy>& access, Tag tag,
    const Coord<Spec::OutputTensor::Ndim>& position, Axis<Dim> dim,
    vec::Vec<Tag> value, Options&&... options) {
  using Planning = typename Policy::PlanningPolicyType;
  using Defaults = ReorderAccessDefaults<
      typename Policy::AccessDefaultsType, Order>;
  using ReorderedPolicy = AccessLoweringPolicy<
      Planning, Defaults, typename Policy::ActiveResources>;
  OutputDataAccess<Spec, ReorderedPolicy> reordered{
      access.spec(), ReorderedPolicy{
          static_cast<const Planning&>(access.policy())}};
  reordered.store(
      tag, position, dim, value, std::forward<Options>(options)...);
  reordered.commit();
}

template <typename Order, AccessPlan Plan, typename OriginalSpec,
          typename AuxSpec, typename Policy, typename Scope,
          vec::VectorTag Tag, int Dim,
          typename... Options>
VECOPS_ALWAYS_INLINE void store_with_order(
    MaterializedOutputDataAccess<
        Plan, OriginalSpec, AuxSpec, Policy, Scope>& access,
    Tag tag, const Coord<OriginalSpec::OutputTensor::Ndim>& position,
    Axis<Dim> dim, vec::Vec<Tag> value, Options&&... options) {
  static_assert(Plan == AccessPlan::materialize_after_transform ||
                std::same_as<Order, vec::cvt::Ordered>);
  using Planning = typename Policy::PlanningPolicyType;
  using Defaults = ReorderAccessDefaults<
      typename Policy::AccessDefaultsType, Order>;
  using ReorderedPolicy = AccessLoweringPolicy<
      Planning, Defaults, typename Policy::ActiveResources>;
  OutputDataAccess<AuxSpec, ReorderedPolicy> hot{
      access.auxiliary_spec(), ReorderedPolicy{
          static_cast<const Planning&>(access.policy())}};
  hot.store(
      tag, position, dim, value, std::forward<Options>(options)...);
  hot.commit();
}

template <typename Order, typename Spec, typename Policy,
          vec::VectorTag Tag, int Dim, typename... Options>
VECOPS_ALWAYS_INLINE void store_with_order(
    BorrowedDataAccess<Spec, Policy>& access, Tag tag,
    const Coord<BorrowedDataAccess<Spec, Policy>::Rank>& position,
    Axis<Dim> dim, vec::Vec<Tag> value, Options&&... options)
  requires (!Spec::is_input)
{
  using Planning = typename Policy::PlanningPolicyType;
  using Defaults = ReorderAccessDefaults<
      typename Policy::AccessDefaultsType, Order>;
  using ReorderedPolicy = AccessLoweringPolicy<
      Planning, Defaults, typename Policy::ActiveResources>;
  BorrowedDataAccess<Spec, ReorderedPolicy> reordered{
      access.spec(), ReorderedPolicy{
          static_cast<const Planning&>(access.policy())}};
  reordered.store(
      tag, position, dim, value, std::forward<Options>(options)...);
}

template <typename Access>
struct UnorderedAccessTraits {
  static constexpr bool eligible = false;
};

template <typename Memory, typename Compute>
consteval bool unordered_conversion_preferred() {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (sizeof(Memory) == sizeof(Compute)) {
    return true;
  } else {
    return is_float16_v<Memory> && std::same_as<Compute, float32_t>;
  }
#else
  return true;
#endif
}

template <typename Spec, typename Policy>
struct UnorderedAccessTraits<InputDataAccess<Spec, Policy>> {
  using ComputeType = typename Spec::ComputeType;
  static constexpr std::size_t storage_bytes = sizeof(typename Spec::MemoryElement);
  static constexpr bool eligible =
      transform_permutation_equivariant_v<typename Spec::TransformType> &&
      unordered_conversion_preferred<
          typename Spec::MemoryElement, ComputeType>();
};

template <typename Spec, typename Policy>
struct UnorderedAccessTraits<OutputDataAccess<Spec, Policy>> {
  using ComputeType = typename Spec::ComputeType;
  static constexpr std::size_t storage_bytes = sizeof(typename Spec::MemoryElement);
  static constexpr bool eligible =
      transform_permutation_equivariant_v<typename Spec::TransformType> &&
      unordered_conversion_preferred<
          typename Spec::MemoryElement, ComputeType>();
};

template <typename Spec, typename Policy>
struct UnorderedAccessTraits<BorrowedDataAccess<Spec, Policy>> {
  using ComputeType = typename Spec::ComputeType;
  static constexpr std::size_t storage_bytes = sizeof(typename Spec::MemoryElement);
  static constexpr bool canonical_compute_output =
      !Spec::is_input &&
      Policy::requested_plan == AccessPlan::materialize_before_transform;
  static constexpr bool eligible =
      !canonical_compute_output &&
      transform_permutation_equivariant_v<typename Spec::TransformType> &&
      unordered_conversion_preferred<
          typename Spec::MemoryElement, ComputeType>();
};

template <typename OriginalSpec, typename AuxSpec, typename Policy>
struct UnorderedAccessTraits<
    DeferredMaterializedInputDataAccess<OriginalSpec, AuxSpec, Policy>> {
  using ComputeType = typename OriginalSpec::ComputeType;
  static constexpr std::size_t storage_bytes = sizeof(ComputeType);
  static constexpr bool eligible = false;
};

template <typename AuxSpec, typename Policy>
struct UnorderedAccessTraits<
    CanonicalMaterializedInputDataAccess<AuxSpec, Policy>> {
  using ComputeType = typename AuxSpec::ComputeType;
  static constexpr std::size_t storage_bytes = sizeof(ComputeType);
  static constexpr bool eligible = false;
};

template <AccessPlan Plan, typename OriginalSpec, typename AuxSpec,
          typename Policy, typename Scope>
struct UnorderedAccessTraits<
    MaterializedOutputDataAccess<
        Plan, OriginalSpec, AuxSpec, Policy, Scope>> {
  using ComputeType = typename OriginalSpec::ComputeType;
  static constexpr std::size_t storage_bytes =
      sizeof(typename OriginalSpec::MemoryElement);
  static constexpr bool eligible =
      Plan == AccessPlan::materialize_after_transform &&
      transform_permutation_equivariant_v<typename OriginalSpec::TransformType> &&
      unordered_conversion_preferred<
          typename OriginalSpec::MemoryElement, ComputeType>();
};

template <typename First, typename... Rest>
consteval bool unordered_group_compatible() {
  using FirstTraits = UnorderedAccessTraits<std::remove_cvref_t<First>>;
  if constexpr (!FirstTraits::eligible ||
                !(UnorderedAccessTraits<std::remove_cvref_t<Rest>>::eligible && ...)) {
    return false;
  } else if constexpr (sizeof...(Rest) == 0) {
    return true;
  } else {
    return ((std::same_as<
                 typename FirstTraits::ComputeType,
                 typename UnorderedAccessTraits<
                     std::remove_cvref_t<Rest>>::ComputeType> &&
             FirstTraits::storage_bytes ==
                 UnorderedAccessTraits<
                     std::remove_cvref_t<Rest>>::storage_bytes) && ...);
  }
}

} // namespace details

/**
 * Access proxy used by `with_unordered_access`.
 * `is_unordered` reports the group-wide compile-time decision.
 */
template <typename Access, bool Unordered>
class AccessOrderView {
public:
  static constexpr bool is_unordered = Unordered;
  using ComputeType = typename Access::ComputeType;
  static constexpr int Rank = Access::Rank;

  explicit AccessOrderView(Access& access) : access_(&access) {}

  template <typename... Options>
    requires requires(
        Access& access, const Coord<Rank>& position, Options&&... options) {
      access.load_scalar(
          position, std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE auto load_scalar(
      const Coord<Rank>& position, Options&&... options) const {
    // A single value has no lane permutation, so the group order does not
    // need to be injected into the underlying scalar conversion.
    return access_->load_scalar(
        position, std::forward<Options>(options)...);
  }

  template <typename... Options>
    requires requires(
        Access& access, const Coord<Rank>& position, ComputeType value,
        Options&&... options) {
      access.store_scalar(
          position, value, std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE void store_scalar(
      const Coord<Rank>& position,
      ComputeType value,
      Options&&... options) const {
    access_->store_scalar(
        position, value, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Options&&... options) const {
    return load(
        tag, position, axis<Rank - 1>,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      Options&&... options) const {
    using Order = std::conditional_t<
        Unordered, vec::cvt::Unordered, vec::cvt::Ordered>;
    return details::load_with_order<Order>(
        *access_, tag, position, dim, std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, vec::Vec<Tag> value,
      Options&&... options) const {
    store(
        tag, position, axis<Rank - 1>, value,
        std::forward<Options>(options)...);
  }

  template <vec::VectorTag Tag, int Dim, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const Coord<Rank>& position, Axis<Dim> dim,
      vec::Vec<Tag> value, Options&&... options) const {
    using Order = std::conditional_t<
        Unordered, vec::cvt::Unordered, vec::cvt::Ordered>;
    details::store_with_order<Order>(
        *access_, tag, position, dim, value,
        std::forward<Options>(options)...);
  }

  void commit() requires requires(Access& access) { access.commit(); } {
    access_->commit();
  }

private:
  Access* access_;
};

/**
 * Prefer one compatible unordered lane-order region, otherwise invoke `fn`
 * with ordered proxies for every argument. The decision always covers all
 * operands; it is never made independently per argument.
 */
template <typename... Accesses, typename Fn>
VECOPS_INLINE decltype(auto) with_unordered_access(
    std::tuple<Accesses&...> accesses, Fn&& fn) {
  constexpr bool Unordered =
      details::unordered_group_compatible<Accesses...>();
  return std::apply(
      [&](auto&... access) -> decltype(auto) {
        return std::forward<Fn>(fn)(
            AccessOrderView<std::remove_reference_t<decltype(access)>,
                            Unordered>{access}...);
      },
      accesses);
}

template <typename A0, typename Fn>
VECOPS_INLINE decltype(auto) with_unordered_access(A0& a0, Fn&& fn) {
  return with_unordered_access(
      std::tie(a0), std::forward<Fn>(fn));
}

template <typename A0, typename A1, typename Fn>
VECOPS_INLINE decltype(auto) with_unordered_access(
    A0& a0, A1& a1, Fn&& fn) {
  return with_unordered_access(
      std::tie(a0, a1), std::forward<Fn>(fn));
}

template <typename A0, typename A1, typename A2, typename Fn>
VECOPS_INLINE decltype(auto) with_unordered_access(
    A0& a0, A1& a1, A2& a2, Fn&& fn) {
  return with_unordered_access(
      std::tie(a0, a1, a2), std::forward<Fn>(fn));
}

template <typename A0, typename A1, typename A2, typename A3, typename Fn>
VECOPS_INLINE decltype(auto) with_unordered_access(
    A0& a0, A1& a1, A2& a2, A3& a3, Fn&& fn) {
  return with_unordered_access(
      std::tie(a0, a1, a2, a3), std::forward<Fn>(fn));
}

/** Invoke @p fn with unordered views for concrete optional accesses.
 *
 * `nullopt` arguments are forwarded unchanged. This retains optionality in
 * the type system while placing all concrete accesses in the same unordered
 * conversion-order region.
 */
template <typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_optional_unordered_access(Fn&& fn) {
  return std::forward<Fn>(fn)();
}

template <typename Fn, typename Access, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) with_optional_unordered_access(
    Fn&& fn, Access& access, Rest&... rest) {
  if constexpr (is_nullopt_v<Access>) {
    return with_optional_unordered_access(
        [&](auto&&... tail) -> decltype(auto) {
          return std::forward<Fn>(fn)(
              nullopt, std::forward<decltype(tail)>(tail)...);
        },
        rest...);
  } else {
    return with_unordered_access(
        access, [&](auto unordered) -> decltype(auto) {
          return with_optional_unordered_access(
              [&](auto&&... tail) -> decltype(auto) {
                return std::forward<Fn>(fn)(
                    unordered, std::forward<decltype(tail)>(tail)...);
              },
              rest...);
        });
  }
}

/**
 * @brief Return workspace bytes required by the resolved Spec/Policy plan.
 *
 * The result includes element-size choice and alignment padding. It is zero for
 * direct plans and readless inputs. An unbound Spec records planning capacity:
 * each runtime Dynamic extent may shrink while retaining its Value constraints,
 * Const extents and every stride remain exact. Rebinding enforces this contract
 * with release-mode checks so execution cannot exceed planned workspace.
 */
template <typename Spec, typename Policy>
VECOPS_INLINE nint_t required_workspace(const Spec& spec, Policy policy) {
  constexpr AccessPlan plan =
      details::resolve_plan<Spec, Policy>();
  if constexpr (plan == AccessPlan::direct) return 0;
  if constexpr (is_input_spec_v<Spec>) {
    if constexpr (!details::transform_reads_input_v<typename Spec::TransformType>) {
      return 0;
    }
  }
  const nint_t element_bytes = [&]() VECOPS_INLINE_LAMBDA {
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
  const nint_t elements = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (is_input_spec_v<Spec>) {
      return numel(spec.input_layout());
    } else {
      return numel(spec.output_layout());
    }
  }();
  return align_up(
      elements * element_bytes,
      vec::DEFAULT_ALIGNMENT);
}

/** An omitted optional operand never consumes workspace. */
template <typename Policy>
VECOPS_INLINE constexpr nint_t required_workspace(nullopt_t, Policy) {
  return 0;
}

/**
 * @brief Bind a statically direct operand without materialization dispatch.
 *
 * This low-level helper asserts that the resolved plan is direct. General
 * kernels should use `kernel::with_operands`, which handles dynamic plans,
 * workspace lifetime, input preparation, and output ownership.
 */
template <typename Spec, typename Policy,
          typename Defaults = DefaultAccessDefaults>
  requires (is_bound_input_spec_v<Spec> || is_bound_output_spec_v<Spec>)
VECOPS_INLINE auto bind(
    const Spec& spec, Policy policy, kernel::WorkspaceView&,
    Defaults defaults = {}) {
  using LoweringPolicy = details::AccessLoweringPolicy<Policy, Defaults>;
  VECOPS_ASSERT(
      (details::resolve_plan<Spec, Policy>() == AccessPlan::direct),
                "tensor::bind only binds direct plans; use kernel::with_operands");
  if constexpr (is_input_spec_v<Spec>) {
    return InputDataAccess<Spec, LoweringPolicy>{
        spec, LoweringPolicy{policy}};
  } else {
    static_assert(is_output_spec_v<Spec>);
    return OutputDataAccess<Spec, LoweringPolicy>{
        spec, LoweringPolicy{policy}};
  }
}

/** Bind an omitted optional operand without allocating or creating an access. */
template <typename Policy, typename Defaults = DefaultAccessDefaults>
VECOPS_INLINE constexpr nullopt_t bind(
    nullopt_t, Policy, kernel::WorkspaceView&, Defaults = {}) {
  return nullopt;
}

/** @brief Pair a caller-owned Spec reference with a kernel-owned Policy. */
template <typename Spec, typename Policy,
          typename Defaults = DefaultAccessDefaults>
struct OperandBinding {
  using SpecType = Spec;
  using PolicyType = Policy;
  using DefaultsType = Defaults;

  const Spec& spec;
  [[no_unique_address]] Policy policy;
  [[no_unique_address]] Defaults defaults;
};

namespace details {

template <typename T>
struct IsOperandBinding : std::bool_constant<
    is_specialization_of_v<OperandBinding, T>> {};

} // namespace details

template <typename T>
inline constexpr bool is_operand_binding_v =
    details::IsOperandBinding<std::remove_cvref_t<T>>::value;

template <typename T>
concept OperandBindingLike = is_operand_binding_v<T>;

/**
 * @brief Create an operand binding consumed by `kernel::with_operands`.
 * @note The referenced Spec must outlive the `with_operands` call.
 */
template <typename Spec, typename Policy,
          typename Defaults = DefaultAccessDefaults>
VECOPS_INLINE auto operand(
    const Spec& spec, Policy policy, Defaults defaults = {}) {
  return OperandBinding<Spec, Policy, Defaults>{spec, policy, defaults};
}

namespace details {

template <typename Binding>
inline constexpr bool binding_resolves_direct_v = [] {
  using B = std::remove_cvref_t<Binding>;
  static_assert(is_operand_binding_v<B>);
  return resolve_plan<typename B::SpecType, typename B::PolicyType>() ==
      AccessPlan::direct;
}();

template <execution::ExecutionScope Context,
          typename Spec, typename Policy, typename Defaults, typename Fn>
/**
 * @brief Resolve and bind one input under an execution context.
 * @param spec Typed input description.
 * @param policy Lifetime access policy controlling axis, passes, and plan.
 * @param defaults Call-site conversion/memory defaults.
 * @param context Execution scope supplying workspace and hardware proofs.
 * @param fn Callback receiving the bound input session.
 * @return Callback result.
 *
 * Direct plans do not touch the workspace. Materialized plans allocate from the
 * context and, when a unit axis is statically known, populate the auxiliary
 * layout through Transpose2D. Otherwise the compile-time general loop is used.
 */
VECOPS_INLINE decltype(auto) with_bound_input(
    const Spec& spec, Policy policy, Defaults defaults,
    Context& context, Fn&& fn) {
  constexpr int AxisValue = Policy::vector_axis;
  constexpr AccessPlan plan = resolve_plan<Spec, Policy>();
  using Resources = typename std::remove_cvref_t<Context>::ActiveResources;
  using LoweringPolicy = AccessLoweringPolicy<Policy, Defaults, Resources>;
  if constexpr (plan == AccessPlan::direct) {
    InputDataAccess<Spec, LoweringPolicy> access{
        spec, LoweringPolicy{policy}};
    return std::forward<Fn>(fn)(access);
  } else {

  auto& workspace = context.workspace_view();

  const nint_t count = numel(spec.input_layout());
  auto aux_layout = auxiliary_layout<AxisValue>(spec.input_layout());
  if constexpr (plan == AccessPlan::materialize_before_transform) {
    using Memory = typename Spec::MemoryElement;
    Memory* buffer = workspace.template allocate<Memory>(count);
    auto auxiliary_tensor = make_tensor(buffer, aux_layout);
    constexpr int UnitAxis = first_unit_axis<
        typename Spec::InputLayout::Strides, AxisValue>();
    if constexpr (UnitAxis >= 0) {
      auto source_spec = input<Memory>(spec.tensor());
      auto destination_spec = output<Memory>(auxiliary_tensor);
      using SourcePlanning = InputAccessPolicy<
          UnitAxis, 1, AccessPlan::direct>;
      using DestinationPlanning = OutputAccessPolicy<
          AxisValue, AccessPlan::direct>;
      using SourcePolicy = AccessLoweringPolicy<
          SourcePlanning, DefaultAccessDefaults, Resources>;
      using DestinationPolicy = AccessLoweringPolicy<
          DestinationPlanning, DefaultAccessDefaults, Resources>;
      InputDataAccess<decltype(source_spec), SourcePolicy> source{
          source_spec, SourcePolicy{SourcePlanning{}}};
      OutputDataAccess<decltype(destination_spec), DestinationPolicy>
          destination{
              destination_spec,
              DestinationPolicy{DestinationPlanning{}}};
      transpose_planes<AxisValue, UnitAxis, UnitAxis, AxisValue>(
          context, spec.input_layout(), source, destination);
      destination.commit();
    } else {
      Coord<Spec::InputTensor::Ndim> position{};
      for_each_coordinate(
          spec.input_layout(), position,
          [&](const auto& current) VECOPS_INLINE_LAMBDA {
            buffer[offset_at(aux_layout, current)] =
                spec.tensor().data()[offset_at(spec.input_layout(), current)];
          });
    }
    using AuxTensor = decltype(auxiliary_tensor);
    using AuxSpec = InputSpec<
        typename Spec::ComputeType, AuxTensor, typename Spec::TransformType,
        typename Spec::ProjectionType>;
    AuxSpec aux_spec{
        auxiliary_tensor, spec.transform(), spec.projection()};
    InputDataAccess<AuxSpec, LoweringPolicy> access{
        aux_spec, LoweringPolicy{policy}};
    return std::forward<Fn>(fn)(access);
  }

  using Compute = typename Spec::ComputeType;
  using Tag = vec::ScalableTag<Compute, 0>;
  Compute* buffer = workspace.template allocate<Compute>(count);
  auto auxiliary_tensor = make_tensor(buffer, aux_layout);
  if constexpr (Policy::requested_plan == AccessPlan::automatic_deferred) {
    auto aux_spec = input<Compute>(auxiliary_tensor);
    using OrderedDefaults = ReorderAccessDefaults<Defaults, vec::cvt::Ordered>;
    using SourcePolicy = AccessLoweringPolicy<
        Policy, OrderedDefaults, Resources>;
    using Access = DeferredMaterializedInputDataAccess<
        Spec, decltype(aux_spec), SourcePolicy>;
    Access access{
        spec, std::move(aux_spec), SourcePolicy{policy}};
    return std::forward<Fn>(fn)(access);
  } else {
  InputDataAccess<Spec, LoweringPolicy> source{
      spec, LoweringPolicy{policy}};
  constexpr int UnitAxis = first_unit_axis<
      typename Spec::InputLayout::Strides, AxisValue>();
  if constexpr (UnitAxis >= 0) {
    auto destination_spec = output<Compute>(auxiliary_tensor);
    using DestinationPlanning = OutputAccessPolicy<
        AxisValue, AccessPlan::direct>;
    using DestinationPolicy = AccessLoweringPolicy<
        DestinationPlanning, DefaultAccessDefaults, Resources>;
    OutputDataAccess<decltype(destination_spec), DestinationPolicy>
        destination{
            destination_spec,
            DestinationPolicy{DestinationPlanning{}}};
    transpose_planes<AxisValue, UnitAxis, UnitAxis, AxisValue>(
        context, spec.input_layout(), source, destination);
    destination.commit();
  } else {
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
  }
  auto aux_spec = input<Compute>(auxiliary_tensor);
  using AuxPolicy = InputAccessPolicy<AxisValue, Policy::read_passes,
                                      AccessPlan::direct>;
  using AuxLoweringPolicy = AccessLoweringPolicy<
      AuxPolicy, DefaultAccessDefaults, Resources>;
  CanonicalMaterializedInputDataAccess<
      decltype(aux_spec), AuxLoweringPolicy> access{
          std::move(aux_spec), AuxLoweringPolicy{AuxPolicy{}}};
  return std::forward<Fn>(fn)(access);
  }
  }
}

template <execution::ExecutionScope Context,
          typename Spec, typename Policy, typename Defaults, typename Fn>
/**
 * @brief Resolve and bind one output under an execution context.
 * @return Callback result from a direct or owning materialized output session.
 * @note An owning result must be committed inside `fn` before scope teardown.
 */
VECOPS_INLINE decltype(auto) with_bound_output(
    const Spec& spec, Policy policy, Defaults defaults,
    Context& context, Fn&& fn) {
  constexpr int AxisValue = Policy::vector_axis;
  constexpr AccessPlan plan = resolve_plan<Spec, Policy>();
  using Resources = typename std::remove_cvref_t<Context>::ActiveResources;
  using LoweringPolicy = AccessLoweringPolicy<Policy, Defaults, Resources>;
  if constexpr (plan == AccessPlan::direct) {
    OutputDataAccess<Spec, LoweringPolicy> access{
        spec, LoweringPolicy{policy}};
    return std::forward<Fn>(fn)(access);
  } else {

  auto& workspace = context.workspace_view();

  const nint_t count = numel(spec.output_layout());
  auto aux_layout = auxiliary_layout<AxisValue>(spec.output_layout());
  if constexpr (plan == AccessPlan::materialize_before_transform) {
    using Compute = typename Spec::ComputeType;
    Compute* buffer = workspace.template allocate<Compute>(count);

    // A materialized output commits its complete auxiliary extent even when a
    // caller performs only masked/partial stores.  Keep untouched elements at
    // the historical zero value without forcing every unrelated scratch
    // Workspace allocation to be value-initialized.
    std::fill_n(buffer, static_cast<std::size_t>(count), Compute{});
    auto auxiliary_tensor = make_tensor(buffer, aux_layout);
    auto aux_spec = output<Compute>(auxiliary_tensor);
    using Access = MaterializedOutputDataAccess<AccessPlan::materialize_before_transform, Spec, decltype(aux_spec),
                                                LoweringPolicy, Context>;
    Access access{
        spec, std::move(aux_spec), LoweringPolicy{policy}, context};
    return std::forward<Fn>(fn)(access);
  }

  using Memory = typename Spec::MemoryElement;
  Memory* buffer = workspace.template allocate<Memory>(count);
  std::fill_n(buffer, static_cast<std::size_t>(count), Memory{});
  auto auxiliary_tensor = make_tensor(buffer, aux_layout);
  using AuxTensor = decltype(auxiliary_tensor);
  using AuxSpec =
    OutputSpec<typename Spec::ComputeType, AuxTensor, typename Spec::TransformType, typename Spec::ProjectionType>;
  AuxSpec aux_spec{
      auxiliary_tensor, spec.transform(), spec.projection()};
  using Access = MaterializedOutputDataAccess<
      AccessPlan::materialize_after_transform, Spec, AuxSpec, LoweringPolicy,
      Context>;
  Access access{
      spec, std::move(aux_spec), LoweringPolicy{policy}, context};
  return std::forward<Fn>(fn)(access);
  }
}

template <execution::ExecutionScope Context,
          typename Spec, typename Policy, typename Defaults, typename Fn>
/** Dispatch one `OperandBinding` to input or output binding by Spec type. */
VECOPS_INLINE decltype(auto) with_bound_operand(
    const OperandBinding<Spec, Policy, Defaults>& binding,
    Context& context, Fn&& fn) {
  if constexpr (is_input_spec_v<Spec>) {
    return with_bound_input(
        binding.spec, binding.policy, binding.defaults, context,
        std::forward<Fn>(fn));
  } else {
    return with_bound_output(
        binding.spec, binding.policy, binding.defaults, context,
        std::forward<Fn>(fn));
  }
}

template <typename Spec, typename Policy, typename Defaults, typename Fn>
/** WorkspaceView compatibility overload creating a borrowing ExecutionSession. */
VECOPS_INLINE decltype(auto) with_bound_operand(
    const OperandBinding<Spec, Policy, Defaults>& binding,
    kernel::WorkspaceView& workspace, Fn&& fn) {
  execution::ExecutionSession context{workspace};
  return with_bound_operand(binding, context, std::forward<Fn>(fn));
}

} // namespace details

} // namespace vecops::tensor

namespace vecops::kernel {

namespace details {

template <typename Fn, typename Tuple, std::size_t... I>
VECOPS_ALWAYS_INLINE decltype(auto) apply_bound_tuple(
    Fn&& fn, Tuple&& tuple, std::index_sequence<I...>) {
  return std::forward<Fn>(fn)(
      std::get<I>(std::forward<Tuple>(tuple))...);
}

template <std::size_t I, execution::ExecutionScope Context,
          typename Tuple, typename BoundTuple, typename Fn>
VECOPS_INLINE decltype(auto) bind_operands_recursive(
    Context& context,
    const Tuple& bindings,
    BoundTuple&& bound,
    Fn&& fn) {
  if constexpr (I == std::tuple_size_v<Tuple>) {
    return apply_bound_tuple(
        std::forward<Fn>(fn), std::forward<BoundTuple>(bound),
        std::make_index_sequence<std::tuple_size_v<
            std::remove_reference_t<BoundTuple>>>{});
  } else {
    const auto& binding = std::get<I>(bindings);
    return tensor::details::with_bound_operand(
        binding, context,
        [&](auto& access) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return bind_operands_recursive<I + 1>(
              context, bindings,
              std::tuple_cat(
                  std::forward<BoundTuple>(bound),
                  std::forward_as_tuple(access)),
              std::forward<Fn>(fn));
        });
  }
}

template <std::size_t I, typename Tuple, typename BoundTuple, typename Fn>
VECOPS_INLINE decltype(auto) bind_operands_recursive(
    WorkspaceView& workspace,
    const Tuple& bindings,
    BoundTuple&& bound,
    Fn&& fn) {
  if constexpr (I == std::tuple_size_v<Tuple>) {
    return apply_bound_tuple(
        std::forward<Fn>(fn), std::forward<BoundTuple>(bound),
        std::make_index_sequence<std::tuple_size_v<
            std::remove_reference_t<BoundTuple>>>{});
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

/**
 * @brief Run one binding callback under a workspace mark that is always
 *        rewound, preserving a non-void callback result.
 */
template <typename Fn>
VECOPS_INLINE decltype(auto) with_workspace_rewind(
    WorkspaceView& workspace, Fn&& fn) {
  const auto mark = workspace.mark();
  if constexpr (std::is_void_v<decltype(std::forward<Fn>(fn)())>) {
    std::forward<Fn>(fn)();
    workspace.rewind(mark);
  } else {
    auto result = std::forward<Fn>(fn)();
    workspace.rewind(mark);
    return result;
  }
}

template <bool NeedsWorkspaceScope, typename Fn>
VECOPS_INLINE decltype(auto) with_operand_workspace_scope(
    WorkspaceView& workspace, Fn&& fn) {
  if constexpr (NeedsWorkspaceScope) {
    return with_workspace_rewind(workspace, std::forward<Fn>(fn));
  } else {
    return std::forward<Fn>(fn)();
  }
}

template <typename... Bindings>
inline constexpr bool operands_need_workspace_scope_v =
    !(tensor::details::binding_resolves_direct_v<Bindings> && ...);

} // namespace details

template <execution::ExecutionScope Context, typename... Bindings, typename Fn>
/**
 * @brief Bind a tuple of operands under an existing execution scope.
 * @param context Scope supplying resources and optional workspace.
 * @param bindings Typed operand bindings whose plans resolve at compile time.
 * @param fn Callback receiving bound lvalue sessions in tuple order.
 * @return Callback result.
 *
 * An all-direct tuple never requests `context.workspace_view()`. If any binding
 * materializes, one workspace mark covers the complete group and is rewound
 * after all bound sessions are destroyed.
 */
VECOPS_INLINE decltype(auto) with_operand_tuple(
    Context& context,
    const std::tuple<Bindings...>& bindings,
    Fn&& fn) {
  if constexpr (details::operands_need_workspace_scope_v<Bindings...>) {
    auto& workspace = context.workspace_view();
    return details::with_operand_workspace_scope<true>(
        workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return details::bind_operands_recursive<0>(
              context, bindings, std::tuple<>{}, std::forward<Fn>(fn));
        });
  } else {
    return details::bind_operands_recursive<0>(
        context, bindings, std::tuple<>{}, std::forward<Fn>(fn));
  }
}

/**
 * @brief Resolve, materialize, bind, and scope a tuple of operands.
 *
 * The callback runs after every input preparation and receives lvalue
 * references to statically typed DataAccess sessions. Output sessions owning
 * auxiliary storage must be committed inside the callback. After callback
 * return, sessions are destroyed. If any operand materializes, the workspace
 * is then rewound to its entry mark, including for a non-void callback result.
 * An all-direct group creates no mark and leaves workspace state untouched.
 * The callback must not otherwise mutate the supplied WorkspaceView.
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
  return details::with_operand_workspace_scope<
      details::operands_need_workspace_scope_v<Bindings...>>(
      workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return details::bind_operands_recursive<0>(
            workspace, bindings, std::tuple<>{}, std::forward<Fn>(fn));
      });
}

/**
 * @brief Convenience overload binding one to four operands.
 *
 * Semantics and lifetime are identical to `with_operand_tuple`: plans resolve
 * outside the callback, materialized output must be committed inside it, and
 * groups containing materialized operands rewind workspace after all sessions
 * are destroyed. Each arity expands as straight-line nested bindings: no
 * recursion, tuple packing, or std::apply stands between the callback and the
 * compiler's inliner.
 */
template <tensor::OperandBindingLike Binding, typename Fn>
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, Binding&& binding, Fn&& fn) {
  return details::with_operand_workspace_scope<
      details::operands_need_workspace_scope_v<Binding>>(
      workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return tensor::details::with_bound_operand(
            binding, workspace,
            [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
              return std::forward<Fn>(fn)(a0);
            });
      });
}

template <execution::ExecutionScope Context,
          tensor::OperandBindingLike Binding, typename Fn>
/**
 * @brief Bind one operand under an execution scope.
 * @return Callback result; direct plans introduce no workspace operations.
 */
VECOPS_INLINE decltype(auto) with_operands(
    Context& context, Binding&& binding, Fn&& fn) {
  auto invoke = [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
    return tensor::details::with_bound_operand(
        binding, context,
        [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return std::forward<Fn>(fn)(a0);
        });
  };
  if constexpr (details::operands_need_workspace_scope_v<Binding>) {
    return details::with_operand_workspace_scope<true>(
        context.workspace_view(), invoke);
  } else {
    return invoke();
  }
}

template <tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1, typename Fn>
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, Fn&& fn) {
  return details::with_operand_workspace_scope<
      details::operands_need_workspace_scope_v<B0, B1>>(
      workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return tensor::details::with_bound_operand(
            b0, workspace,
            [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
              return tensor::details::with_bound_operand(
                  b1, workspace,
                  [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                    return std::forward<Fn>(fn)(a0, a1);
                  });
            });
      });
}

template <execution::ExecutionScope Context,
          tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1, typename Fn>
/** Bind two operands under one execution/workspace lifetime. */
VECOPS_INLINE decltype(auto) with_operands(
    Context& context, B0&& b0, B1&& b1, Fn&& fn) {
  auto invoke = [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
    return tensor::details::with_bound_operand(
        b0, context,
        [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return tensor::details::with_bound_operand(
              b1, context,
              [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                return std::forward<Fn>(fn)(a0, a1);
              });
        });
  };
  if constexpr (details::operands_need_workspace_scope_v<B0, B1>) {
    return details::with_operand_workspace_scope<true>(
        context.workspace_view(), invoke);
  } else {
    return invoke();
  }
}

template <tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1,
          tensor::OperandBindingLike B2, typename Fn>
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, B2&& b2, Fn&& fn) {
  return details::with_operand_workspace_scope<
      details::operands_need_workspace_scope_v<B0, B1, B2>>(
      workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return tensor::details::with_bound_operand(
            b0, workspace,
            [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
              return tensor::details::with_bound_operand(
                  b1, workspace,
                  [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                    return tensor::details::with_bound_operand(
                        b2, workspace,
                        [&](auto& a2) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                          return std::forward<Fn>(fn)(a0, a1, a2);
                        });
                  });
            });
      });
}

template <execution::ExecutionScope Context,
          tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1,
          tensor::OperandBindingLike B2, typename Fn>
/** Bind three operands under one execution/workspace lifetime. */
VECOPS_INLINE decltype(auto) with_operands(
    Context& context, B0&& b0, B1&& b1, B2&& b2, Fn&& fn) {
  auto invoke = [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
    return tensor::details::with_bound_operand(
        b0, context,
        [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return tensor::details::with_bound_operand(
              b1, context,
              [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                return tensor::details::with_bound_operand(
                    b2, context,
                    [&](auto& a2) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                      return std::forward<Fn>(fn)(a0, a1, a2);
                    });
              });
        });
  };
  if constexpr (details::operands_need_workspace_scope_v<B0, B1, B2>) {
    return details::with_operand_workspace_scope<true>(
        context.workspace_view(), invoke);
  } else {
    return invoke();
  }
}

template <tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1,
          tensor::OperandBindingLike B2,
          tensor::OperandBindingLike B3, typename Fn>
VECOPS_INLINE decltype(auto) with_operands(
    WorkspaceView& workspace, B0&& b0, B1&& b1, B2&& b2, B3&& b3,
    Fn&& fn) {
  return details::with_operand_workspace_scope<
      details::operands_need_workspace_scope_v<B0, B1, B2, B3>>(
      workspace, [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
        return tensor::details::with_bound_operand(
            b0, workspace,
            [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
              return tensor::details::with_bound_operand(
                  b1, workspace,
                  [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                    return tensor::details::with_bound_operand(
                        b2, workspace,
                        [&](auto& a2) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                          return tensor::details::with_bound_operand(
                              b3, workspace,
                              [&](auto& a3) VECOPS_INLINE_LAMBDA
                                  -> decltype(auto) {
                                return std::forward<Fn>(fn)(a0, a1, a2, a3);
                              });
                        });
                  });
            });
      });
}

template <execution::ExecutionScope Context,
          tensor::OperandBindingLike B0,
          tensor::OperandBindingLike B1,
          tensor::OperandBindingLike B2,
          tensor::OperandBindingLike B3, typename Fn>
/** Bind four operands under one execution/workspace lifetime. */
VECOPS_INLINE decltype(auto) with_operands(
    Context& context, B0&& b0, B1&& b1, B2&& b2, B3&& b3,
    Fn&& fn) {
  auto invoke = [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
    return tensor::details::with_bound_operand(
        b0, context,
        [&](auto& a0) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return tensor::details::with_bound_operand(
              b1, context,
              [&](auto& a1) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                return tensor::details::with_bound_operand(
                    b2, context,
                    [&](auto& a2) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                      return tensor::details::with_bound_operand(
                          b3, context,
                          [&](auto& a3) VECOPS_INLINE_LAMBDA
                              -> decltype(auto) {
                            return std::forward<Fn>(fn)(a0, a1, a2, a3);
                          });
                    });
              });
        });
  };
  if constexpr (details::operands_need_workspace_scope_v<B0, B1, B2, B3>) {
    return details::with_operand_workspace_scope<true>(
        context.workspace_view(), invoke);
  } else {
    return invoke();
  }
}

} // namespace vecops::kernel

#endif // VECOPS_TENSOR_DATA_ACCESS_H
