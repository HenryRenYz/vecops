#ifndef VECOPS_VEC_DETAILS_REQUEST_H
#define VECOPS_VEC_DETAILS_REQUEST_H

#include "vecops/vec/Request.h"
#include "vecops/vec/details/Options.h"

/**
 * @file Request.h
 * @brief One-shot fold of an option pack into a resolved Request.
 *
 * `resolve_load_request` / `resolve_store_request` are the only place that
 * walks a memory option pack. The variadic entry points call them once and
 * every downstream layer receives the Request struct. Layers that construct
 * their access shape statically can build a Request directly and skip
 * resolution entirely.
 */

namespace vecops::vec::details {

/** The first option matching Predicate, or `Default` when the pack omits it. */
template <template <typename> typename Predicate,
          typename Default, typename... Options>
using option_type_or_t = std::conditional_t<
    std::is_void_v<find_option_type_t<Predicate, Options...>>,
    Default,
    find_option_type_t<Predicate, Options...>>;

namespace request_resolve {

template <typename... Options>
inline constexpr Active active_kind_of_v =
    option_count_v<IsFirstOption, Options...> == 1
        ? Active::First
        : (option_count_v<IsMaskedOption, Options...> == 1 ? Active::Masked
                                                         : Active::Unmasked);

template <typename... Options>
inline constexpr Addressing addressing_kind_of_v =
    option_count_v<IsIndexedOption, Options...> == 1
        ? Addressing::Indexed
        : (option_count_v<IsStridedOption, Options...> == 1
               ? Addressing::Strided
               : Addressing::Contiguous);

template <typename... Options>
inline constexpr Populate populate_kind_of_v =
    option_count_v<IsVectorMergeOption, Options...> == 1
        ? Populate::MergeVector
        : (option_count_v<IsScalarMergeOption, Options...> == 1
               ? Populate::MergeScalar
               : Populate::Zero);

template <typename... Options>
inline constexpr int index_scale_of_v = [] {
  using Indexed = find_option_type_t<IsIndexedOption, Options...>;
  if constexpr (!std::is_void_v<Indexed>) {
    return Indexed::scale;
  } else {
    return 0;
  }
}();

template <VectorTag Tag, typename MaybeIndexed>
struct IndexVectorOf {
  using type = Vec<IndexTag<Tag>>;
};

template <VectorTag Tag, typename V, int Scale>
struct IndexVectorOf<Tag, opt::Indexed<V, Scale>> {
  using type = V;
};

template <VectorTag Tag, typename... Options>
using index_vector_of_t = typename IndexVectorOf<
    Tag, find_option_type_t<IsIndexedOption, Options...>>::type;

} // namespace request_resolve

/**
 * Folds a validated `load` option pack into a LoadRequest. The pack must
 * already satisfy `valid_memory_options<Tag, false, Options...>()`; only the
 * fields matching the resolved kinds are populated.
 */
template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE auto resolve_load_request(Options&&... options) {
  using Request = LoadRequest<
      Tag,
      request_resolve::active_kind_of_v<Options...>,
      request_resolve::addressing_kind_of_v<Options...>,
      request_resolve::populate_kind_of_v<Options...>,
      option_type_or_t<IsMemoryAlignmentOption, mem::Unaligned, Options...>,
      option_type_or_t<IsMemoryTemporalityOption, mem::Temporal, Options...>,
      request_resolve::index_scale_of_v<Options...>,
      request_resolve::index_vector_of_t<Tag, Options...>>;
  Request request;
  if constexpr (Request::active_kind == Active::First) {
    request.first_count = find_option<IsFirstOption>(options...).count;
  }
  if constexpr (Request::active_kind == Active::Masked) {
    request.mask = &find_option<IsMaskedOption>(options...).value;
  }
  if constexpr (Request::populate_kind == Populate::MergeVector) {
    request.merge_vector = &find_option<IsVectorMergeOption>(options...).value;
  }
  if constexpr (Request::populate_kind == Populate::MergeScalar) {
    request.merge_scalar = find_option<IsScalarMergeOption>(options...).value;
  }
  if constexpr (Request::addressing_kind == Addressing::Strided) {
    request.stride = static_cast<nint_t>(
        find_option<IsStridedOption>(options...).stride);
  }
  if constexpr (Request::addressing_kind == Addressing::Indexed) {
    request.indices = &find_option<IsIndexedOption>(options...).indices;
  }
  return request;
}

/**
 * Folds a validated `store` option pack into a StoreRequest. The pack must
 * already satisfy `valid_memory_options<Tag, true, Options...>()`.
 */
template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE auto resolve_store_request(Options&&... options) {
  using Request = StoreRequest<
      Tag,
      request_resolve::active_kind_of_v<Options...>,
      request_resolve::addressing_kind_of_v<Options...>,
      option_type_or_t<IsMemoryAlignmentOption, mem::Unaligned, Options...>,
      option_type_or_t<IsMemoryTemporalityOption, mem::Temporal, Options...>,
      request_resolve::index_scale_of_v<Options...>,
      request_resolve::index_vector_of_t<Tag, Options...>>;
  Request request;
  if constexpr (Request::active_kind == Active::First) {
    request.first_count = find_option<IsFirstOption>(options...).count;
  }
  if constexpr (Request::active_kind == Active::Masked) {
    request.mask = &find_option<IsMaskedOption>(options...).value;
  }
  if constexpr (Request::addressing_kind == Addressing::Strided) {
    request.stride = static_cast<nint_t>(
        find_option<IsStridedOption>(options...).stride);
  }
  if constexpr (Request::addressing_kind == Addressing::Indexed) {
    request.indices = &find_option<IsIndexedOption>(options...).indices;
  }
  return request;
}

namespace request_resolve {

template <typename T>
struct IsAnyConversionLayoutOption : std::bool_constant<
    IsOrderedOption<T>::value || IsUnorderedOption<T>::value> {};

template <typename T>
struct IsAnyConversionValueOption : std::bool_constant<
    IsSaturateOption<T>::value || IsWrapOption<T>::value> {};

template <typename T>
struct IsAnyConversionPackingOption : std::bool_constant<
    IsPackedOption<T>::value || IsSplitOption<T>::value> {};

template <typename ToTag, typename MaybeMasked>
struct ConvertMaskVectorOf {
  using type = Mask<ToTag>;
};

template <typename ToTag, typename M>
struct ConvertMaskVectorOf<ToTag, opt::Masked<M>> {
  using type = M;
};

} // namespace request_resolve

namespace request_resolve {

template <typename... Options>
inline constexpr bool is_mask_merge_option_pack_v =
    ((IsMaskMergeOption<std::remove_cvref_t<Options>>::value) || ...);

template <typename... Options>
inline constexpr Active op_active_kind_of =
    option_count_v<IsMaskedOption, Options...> == 1
        ? Active::Masked
        : Active::Unmasked;

template <typename... Options>
inline constexpr Inactive op_inactive_kind_of =
    option_count_v<IsZeroOption, Options...> == 1
        ? Inactive::Zero
        : (option_count_v<IsVectorMergeOption, Options...> == 1
               ? Inactive::MergeVector
               : (option_count_v<IsScalarMergeOption, Options...> == 1
                      ? Inactive::MergeScalar
                      : (is_mask_merge_option_pack_v<Options...>
                             ? Inactive::MergeMask
                             : Inactive::PreserveInput)));

} // namespace request_resolve

/**
 * Folds an elementwise option pack into an OpRequest. The pack must already
 * satisfy the operation family's validation (exactly one active option, at
 * most one population option).
 */
template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE auto resolve_op_request(Options&&... options) {
  using Request = OpRequest<
      Tag,
      request_resolve::op_active_kind_of<Options...>,
      request_resolve::op_inactive_kind_of<Options...>>;
  Request request;
  if constexpr (Request::active_kind == Active::Masked) {
    request.mask = &find_option<IsMaskedOption>(options...).value;
  }
  if constexpr (Request::inactive_kind == Inactive::MergeVector) {
    request.merge_vector =
        &find_option<IsVectorMergeOption>(options...).value;
  }
  if constexpr (Request::inactive_kind == Inactive::MergeScalar) {
    request.merge_scalar =
        find_option<IsScalarMergeOption>(options...).value;
  }
  if constexpr (Request::inactive_kind == Inactive::MergeMask) {
    request.mask_merge =
        &find_option<IsMaskMergeOption>(options...).value;
  }
  return request;
}

/**
 * Folds a validated `load_convert` option pack into a LoadConvertRequest.
 * The pack must already satisfy
 * `valid_memory_conversion_options<ToTag, From, false, Options...>()`.
 */
template <VectorTag ToTag, Element From, typename... Options>
VECOPS_ALWAYS_INLINE auto resolve_load_convert_request(Options&&... options) {
  using Request = LoadConvertRequest<
      ToTag,
      From,
      request_resolve::active_kind_of_v<Options...>,
      request_resolve::addressing_kind_of_v<Options...>,
      request_resolve::populate_kind_of_v<Options...>,
      option_type_or_t<IsMemoryAlignmentOption, mem::Unaligned, Options...>,
      option_type_or_t<IsMemoryTemporalityOption, mem::Temporal, Options...>,
      request_resolve::index_scale_of_v<Options...>,
      request_resolve::index_vector_of_t<ToTag, Options...>,
      option_type_or_t<
          request_resolve::IsAnyConversionLayoutOption,
          cvt::Ordered, Options...>,
      option_type_or_t<
          request_resolve::IsAnyConversionValueOption,
          cvt::Saturate, Options...>,
      typename request_resolve::ConvertMaskVectorOf<
          ToTag,
          find_option_type_t<IsMaskedOption, Options...>>::type>;
  Request request;
  if constexpr (Request::active_kind == Active::First) {
    request.first_count = find_option<IsFirstOption>(options...).count;
  }
  if constexpr (Request::active_kind == Active::Masked) {
    request.mask = &find_option<IsMaskedOption>(options...).value;
  }
  if constexpr (Request::populate_kind == Populate::MergeVector) {
    request.merge_vector = &find_option<IsVectorMergeOption>(options...).value;
  }
  if constexpr (Request::populate_kind == Populate::MergeScalar) {
    request.merge_scalar = find_option<IsScalarMergeOption>(options...).value;
  }
  if constexpr (Request::addressing_kind == Addressing::Strided) {
    request.stride = static_cast<nint_t>(
        find_option<IsStridedOption>(options...).stride);
  }
  if constexpr (Request::addressing_kind == Addressing::Indexed) {
    request.indices = &find_option<IsIndexedOption>(options...).indices;
  }
  return request;
}

/**
 * Folds a validated `store_convert` option pack into a StoreConvertRequest.
 * The pack must already satisfy
 * `valid_memory_conversion_options<FromTag, To, true, Options...>()`.
 */
template <VectorTag FromTag, Element To, typename... Options>
VECOPS_ALWAYS_INLINE auto resolve_store_convert_request(
    Options&&... options) {
  using Request = StoreConvertRequest<
      FromTag,
      To,
      request_resolve::active_kind_of_v<Options...>,
      request_resolve::addressing_kind_of_v<Options...>,
      option_type_or_t<IsMemoryAlignmentOption, mem::Unaligned, Options...>,
      option_type_or_t<IsMemoryTemporalityOption, mem::Temporal, Options...>,
      request_resolve::index_scale_of_v<Options...>,
      request_resolve::index_vector_of_t<FromTag, Options...>,
      option_type_or_t<
          request_resolve::IsAnyConversionLayoutOption,
          cvt::Ordered, Options...>,
      option_type_or_t<
          request_resolve::IsAnyConversionValueOption,
          cvt::Saturate, Options...>,
      option_type_or_t<
          request_resolve::IsAnyConversionPackingOption,
          mem::Packed, Options...>,
      typename request_resolve::ConvertMaskVectorOf<
          FromTag,
          find_option_type_t<IsMaskedOption, Options...>>::type>;
  Request request;
  if constexpr (Request::active_kind == Active::First) {
    request.first_count = find_option<IsFirstOption>(options...).count;
  }
  if constexpr (Request::active_kind == Active::Masked) {
    request.mask = &find_option<IsMaskedOption>(options...).value;
  }
  if constexpr (Request::addressing_kind == Addressing::Strided) {
    request.stride = static_cast<nint_t>(
        find_option<IsStridedOption>(options...).stride);
  }
  if constexpr (Request::addressing_kind == Addressing::Indexed) {
    request.indices = &find_option<IsIndexedOption>(options...).indices;
  }
  return request;
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_REQUEST_H
