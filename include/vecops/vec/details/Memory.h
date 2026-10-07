#ifndef VECOPS_VEC_DETAILS_MEMORY_H
#define VECOPS_VEC_DETAILS_MEMORY_H

/**
 * @file Memory.h
 * @brief Memory operation infrastructure: option validation
 * (valid_memory_options, is_memory_option_for_v), GenericImpl for LoadOp
 * and StoreOp (contiguous, indexed, and strided access), and
 * StridedIndicesOp for building linear index sequences.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/vec/Request.h"
#include "vecops/vec/Options.h"
#include "vecops/vec/details/Request.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

/** Whether a logical strided access can represent its i32 index vector.
 * VLA SVE has bounded sizeless tuples; a valid data Tag need not have a valid
 * widened index Tag. Scalarize only that access, without instantiating the
 * impossible Vec<Rebind<int32_t, Tag>> (including in dead template code).
 */
template <VectorTag Tag, Element IndexElement = int32_t>
consteval bool strided_indices_supported() {
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
  if constexpr (is_fixed_tag_v<Tag>) {
    return fixed_lanes_v<Tag> * sizeof(IndexElement) <= 16;
  } else {
    return scale_power_v<Rebind<IndexElement, Tag>> <= VEC_MAX_POW;
  }
#else
  return true;
#endif
}

/** Architectural upper bound, not the currently selected VL or SVL.
 * Both SVE execution domains obey MAX_VEC_WIDTH. A constant ordinary stride
 * can therefore eliminate the range guard even when size(tag) must use a
 * volatile CNTB to distinguish unequal normal/streaming lengths.
 */
template <VectorTag Tag>
consteval nint_t strided_index_lane_upper_bound() {
  if constexpr (is_fixed_tag_v<Tag>) {
    return fixed_lanes_v<Tag>;
  } else if constexpr (scale_power_v<Tag> >= 0) {
    return (MAX_VEC_WIDTH / 8 / sizeof(ElementOf<Tag>)) << scale_power_v<Tag>;
  } else {
    return (MAX_VEC_WIDTH / 8 / sizeof(ElementOf<Tag>)) >> (-scale_power_v<Tag>);
  }
}

/** Whether the native i32 linear index sequence can describe this request.
 * Check the last logical lane, not just its step. SVE's indexed-memory
 * implementation also scales the indices in i32 before forming addresses.
 * Divide the limits rather than multiplying user-provided strides, so the
 * proof itself cannot overflow. An unsafe request uses legal i64 SVE indices
 * or the scalar boundary below, retaining inactive-lane semantics.
 */
template <typename MemoryElement, typename Request>
VECOPS_ALWAYS_INLINE bool strided_indices_fit(
    typename Request::TagType tag, const Request& request) {
  using Tag = typename Request::TagType;
  static_assert(Request::addressing_kind == Addressing::Strided);
  constexpr nint_t scale = std::same_as<CurrentBackend, SVEBackend>
      ? (Request::index_scale == 0
          ? static_cast<nint_t>(sizeof(MemoryElement))
          : static_cast<nint_t>(Request::index_scale))
      : nint_t{1};
  static_assert(scale > 0);
  constexpr nint_t last_upper = std::max<nint_t>(
      strided_index_lane_upper_bound<Tag>() - 1, 1);
  constexpr nint_t safe_lower = std::numeric_limits<int32_t>::min() / scale / last_upper;
  constexpr nint_t safe_upper = std::numeric_limits<int32_t>::max() / scale / last_upper;
  if (request.stride >= safe_lower && request.stride <= safe_upper)
    return true;
  const nint_t last = std::max<nint_t>(size(tag) - 1, 1);
  const nint_t lower = std::numeric_limits<int32_t>::min() / scale / last;
  const nint_t upper = std::numeric_limits<int32_t>::max() / scale / last;
  return request.stride >= lower && request.stride <= upper;
}

/** Readdress a borrowed request without changing its mask/population domain.
 * The index vector remains caller-owned and must outlive the synchronous
 * access. All conversion, packing and execution-resource policies survive.
 */
template <typename IndexedRequest, typename Request, VectorValue Indices>
VECOPS_ALWAYS_INLINE IndexedRequest copy_indexed_request_fields(
    const Request& source, const Indices& indices) {
  IndexedRequest result;
  result.indices = &indices;
  if constexpr (Request::active_kind == Active::First)
    result.first_count = source.first_count;
  else if constexpr (Request::active_kind == Active::Masked)
    result.mask = source.mask;
  if constexpr (requires { result.merge_vector = source.merge_vector; })
    result.merge_vector = source.merge_vector;
  else if constexpr (requires { result.merge_scalar = source.merge_scalar; })
    result.merge_scalar = source.merge_scalar;
  return result;
}

template <typename Request, VectorValue Indices>
VECOPS_ALWAYS_INLINE auto reindex_request(
    const Request& source, const Indices& indices) {
  using Tag = typename Request::TagType;
  constexpr Active A = Request::active_kind;
  using Alignment = typename Request::AlignmentOption;
  using Temporality = typename Request::TemporalityOption;
  using Resources = typename Request::ActiveResources;
  constexpr int Scale = Request::index_scale;
  if constexpr (requires { typename Request::FromElement; }) {
    using Result = LoadConvertRequest<Tag, typename Request::FromElement, A,
        Addressing::Indexed, Request::populate_kind, Alignment, Temporality,
        Scale, Indices, typename Request::LayoutOption,
        typename Request::ValuePolicyOption, typename Request::MaskVectorType,
        Resources>;
    return copy_indexed_request_fields<Result>(source, indices);
  } else if constexpr (requires { typename Request::ToElement; }) {
    using Result = StoreConvertRequest<Tag, typename Request::ToElement, A,
        Addressing::Indexed, Alignment, Temporality, Scale, Indices,
        typename Request::LayoutOption, typename Request::ValuePolicyOption,
        typename Request::PackingOption, typename Request::MaskVectorType,
        Resources>;
    return copy_indexed_request_fields<Result>(source, indices);
  } else if constexpr (requires { Request::populate_kind; }) {
    using Result = LoadRequest<Tag, A, Addressing::Indexed,
        Request::populate_kind, Alignment, Temporality, Scale, Indices,
        Resources>;
    return copy_indexed_request_fields<Result>(source, indices);
  } else {
    using Result = StoreRequest<Tag, A, Addressing::Indexed, Alignment,
        Temporality, Scale, Indices, Resources>;
    return copy_indexed_request_fields<Result>(source, indices);
  }
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Rebind<int64_t, Tag>> make_wide_strided_indices(
    Tag, nint_t stride) {
  // Public integer iota is modular, including each physical-word start.
  // Inactive huge offsets cannot trigger C++ signed overflow while forming
  // the sequence. Valid active byte offsets remain the caller's contract.
  return iota(Rebind<int64_t, Tag>{}, int64_t{0}, int64_t{stride});
}

template <typename Request, VectorTag Tag>
VECOPS_ALWAYS_INLINE bool request_lane_active(
    Tag tag, const Request& request, nint_t lane) {
  if constexpr (Request::active_kind == Active::First) {
    return lane < request.first_count;
  } else if constexpr (Request::active_kind == Active::Masked) {
    return get(tag, *request.mask, lane);
  } else {
    return true;
  }
}

template <typename Request, Element T, VectorTag Tag>
VECOPS_ALWAYS_INLINE const T* request_memory_address(
    Tag, const T* pointer, const Request& request, nint_t lane) {
  if constexpr (Request::addressing_kind == Addressing::Indexed) {
    using IndexTag = VecToTag<typename Request::IndexVectorType>;
    const nint_t index = static_cast<nint_t>(
        get(IndexTag{}, *request.indices, lane));
    if constexpr (Request::index_scale == 0) {
      return pointer + index;
    } else {
      const auto* bytes = reinterpret_cast<const unsigned char*>(pointer);
      return reinterpret_cast<const T*>(
          bytes + index * Request::index_scale);
    }
  } else {
    static_assert(Request::addressing_kind == Addressing::Strided);
    if constexpr (Request::index_scale == 0) {
      return pointer + lane * request.stride;
    } else {
      const auto* bytes = reinterpret_cast<const unsigned char*>(pointer);
      return reinterpret_cast<const T*>(
          bytes + lane * request.stride * Request::index_scale);
    }
  }
}

template <typename Request, Element T, VectorTag Tag>
VECOPS_ALWAYS_INLINE T* request_memory_address(
    Tag tag, T* pointer, const Request& request, nint_t lane) {
  return const_cast<T*>(request_memory_address(
      tag, static_cast<const T*>(pointer), request, lane));
}

template <typename Request>
VECOPS_ALWAYS_INLINE Vec<typename Request::TagType>
execute_scalar_load_request(
    typename Request::TagType tag,
    const ElementOf<typename Request::TagType>* pointer,
    const Request& request) {
  using Tag = typename Request::TagType;
  using T = ElementOf<Tag>;
  Vec<Tag> result = [&]() VECOPS_INLINE_LAMBDA -> Vec<Tag> {
    if constexpr (Request::populate_kind == Populate::MergeVector) {
      return *request.merge_vector;
    } else if constexpr (Request::populate_kind == Populate::MergeScalar) {
      return fill(tag, request.merge_scalar);
    } else {
      return zeros(tag);
    }
  }();
  for (nint_t lane = 0; lane < size(tag); ++lane) {
    if (!request_lane_active(tag, request, lane)) continue;
    T scalar;
    const T* address = request_memory_address(
        tag, pointer, request, lane);
    std::memcpy(&scalar, address, sizeof(T));
    result = set(tag, result, lane, scalar);
  }
  return result;
}

template <typename Request>
VECOPS_ALWAYS_INLINE void execute_scalar_store_request(
    typename Request::TagType tag,
    ElementOf<typename Request::TagType>* pointer,
    Vec<typename Request::TagType> value,
    const Request& request) {
  using T = ElementOf<typename Request::TagType>;
  for (nint_t lane = 0; lane < size(tag); ++lane) {
    if (!request_lane_active(tag, request, lane)) continue;
    const T scalar = get(tag, value, lane);
    T* address = request_memory_address(tag, pointer, request, lane);
    std::memcpy(address, &scalar, sizeof(T));
  }
}

/* **************************************************************************** */
//    Option validation for memory operations                                   //
/* **************************************************************************** */
// The IsMemory*Option group predicates live in details/Options.h.

template <VectorTag Tag, typename Option>
inline constexpr bool is_memory_addressing_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (IsIndexedOption<Clean>::value) {
    using Indices = typename IsIndexedOption<Clean>::Value;
    using IndexElement = ElementOf<VecToTag<Indices>>;
    if constexpr (std::same_as<IndexElement, int32_t>)
      return std::same_as<Indices, Vec<Rebind<int32_t, Tag>>>;
    else if constexpr (std::same_as<IndexElement, int64_t>)
      return std::same_as<Indices, Vec<Rebind<int64_t, Tag>>>;
    else
      return false;
  } else {
    return IsStridedOption<Clean>::value;
  }
}();

template <bool Aligned, bool Filtered, bool NonTemporal>
consteval bool supported_memory_access() {
  return true;
}

template <VectorTag Tag, bool IsStore, typename Option>
inline constexpr bool is_memory_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (
      IsMemoryAlignmentOption<Clean>::value ||
      IsMemoryTemporalityOption<Clean>::value ||
      IsResourcesOption<Clean>::value ||
      IsUnmaskedOption<Clean>::value ||
      IsFirstOption<Clean>::value) {
    return true;
  } else if constexpr (IsMaskedOption<Clean>::value) {
    return is_masked_option_for_v<Tag, Clean>;
  } else if constexpr (IsMemoryAddressingOption<Clean>::value) {
    return is_memory_addressing_option_for_v<Tag, Clean>;
  } else if constexpr (!IsStore && IsMemoryPopulationOption<Clean>::value) {
    return is_vector_population_option_for_v<Tag, Clean>;
  } else {
    return false;
  }
}();

template <VectorTag Tag, bool IsStore, typename... Options>
consteval bool valid_memory_options() {
  if constexpr (!(is_memory_option_for_v<Tag, IsStore, Options> && ...)) {
    return false;
  } else {
    constexpr std::size_t alignment_count =
        option_count_v<IsMemoryAlignmentOption, Options...>;
    constexpr std::size_t temporality_count =
        option_count_v<IsMemoryTemporalityOption, Options...>;
    constexpr std::size_t active_count =
        option_count_v<IsMemoryActiveOption, Options...>;
    constexpr std::size_t population_count =
        option_count_v<IsMemoryPopulationOption, Options...>;
    constexpr std::size_t indexed_count =
        option_count_v<IsIndexedOption, Options...>;
    constexpr std::size_t strided_count =
        option_count_v<IsStridedOption, Options...>;
    constexpr std::size_t resources_count =
        option_count_v<IsResourcesOption, Options...>;
    return alignment_count <= 1 && temporality_count <= 1 &&
        active_count <= 1 && population_count <= 1 &&
        indexed_count <= 1 && strided_count <= 1 && resources_count <= 1 &&
        indexed_count + strided_count <= 1 &&
        ((indexed_count + strided_count == 0) || alignment_count == 0) &&
        (!IsStore || population_count == 0) &&
        (population_count == 0 || active_count == 1);
  }
}

/* **************************************************************************** */
//    GenericImpl for LoadOp: contiguous, indexed, and strided load             //
/* **************************************************************************** */

/**
 * Generic multi-word implementation for LoadOp.
 *
 * Responsible for: contiguous word-based loads (with and without masks),
 * indexed (gather) loads with mask gating, and strided loads (decomposed
 * into index generation + indexed load).
 */
template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, LoadOp, Tag> {
 private:
  template <VectorValue Indices, int Scale>
  static VECOPS_ALWAYS_INLINE Vec<Tag> load_indexed(
      Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing,
      Mask<Tag> mask, Vec<Tag> inactive) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, Tag>;
    using IndexElement = ElementOf<IndexTag>;
    constexpr nint_t byte_scale =
        Scale == 0 ? static_cast<nint_t>(sizeof(ElementOf<Tag>)) : Scale;
    const auto* base = reinterpret_cast<const std::byte*>(pointer);
    constexpr std::size_t ScalarIndexCount = [] {
      if constexpr (std::same_as<Backend, ScalarBackend>)
        return static_cast<std::size_t>(
            RepresentationTraits<Backend, IndexTag>::logical_lanes);
      else
        return std::size_t{1};
    }();
    std::array<IndexElement, ScalarIndexCount> scalar_indices{};
    if constexpr (std::same_as<Backend, ScalarBackend>) {
      static_assert(
          sizeof(scalar_indices) <= sizeof(addressing.indices));
      std::memcpy(
          scalar_indices.data(), &addressing.indices,
          sizeof(scalar_indices));
    }
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      auto word = ::vecops::vec::get_word<Index>(tag, inactive);
      constexpr nint_t word_lanes =
          RepresentationTraits<Backend, Tag>::word_lanes;
      const nint_t begin = Index * word_lanes;
      const nint_t end = std::min(begin + word_lanes, size(tag));
      const auto mask_word = ::vecops::vec::get_word<Index>(tag, mask);
      for (nint_t lane = begin; lane < end; ++lane) {
        if (!execute_word<Index, Backend>(
                GetMaskLaneOp{}, tag, mask_word, lane - begin)) continue;
        const auto index = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (std::same_as<Backend, ScalarBackend>)
            return scalar_indices[static_cast<std::size_t>(lane)];
          else
            return get_vec_lane_at<Backend>(
                IndexTag{}, addressing.indices, lane);
        }();
        ElementOf<Tag> loaded;
        std::memcpy(
            &loaded, base + static_cast<nint_t>(index) * byte_scale,
            sizeof(loaded));
        word = execute_word<Index, Backend>(
            SetVecLaneOp{}, tag, word, lane - begin, loaded);
      }
      return word;
    });
  }

 public:
  template <typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp op, Tag tag, const ElementOf<Tag>* pointer,
      Alignment alignment, Temporality temporality) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag, pointer + Index * native_word_size(tag),
          alignment, temporality);
    });
  }

  template <typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp op, Tag tag, const ElementOf<Tag>* pointer,
      Mask<Tag> mask, Vec<Tag> inactive,
      Alignment alignment, Temporality temporality) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(
          op, tag, pointer + Index * native_word_size(tag),
          ::vecops::vec::get_word<Index>(tag, mask),
          ::vecops::vec::get_word<Index>(tag, inactive),
          alignment, temporality);
    });
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing, Temporality) {
    return load_indexed(
        tag, pointer, addressing, mfill(tag, true),
        fill(tag, ElementOf<Tag>{}));
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing,
      Mask<Tag> mask, Vec<Tag> inactive, Temporality) {
    return load_indexed(tag, pointer, addressing, mask, inactive);
  }
};

/* **************************************************************************** */
//    StridedIndicesOp: build linear index sequence from scalar stride          //
/* **************************************************************************** */

/** Builds a linear index sequence from a scalar stride: indices[i] = i * stride. */
template <typename Backend, VectorTag IndexTag>
struct GenericImpl<Backend, StridedIndicesOp, IndexTag> {
  static VECOPS_ALWAYS_INLINE Vec<IndexTag> call(
      StridedIndicesOp, IndexTag tag, nint_t stride) {
    static_assert(std::same_as<ElementOf<IndexTag>, int32_t>);
    VECOPS_ASSERT(
        size(tag) <= 1 ||
            (stride >= std::numeric_limits<int32_t>::min() /
                           (size(tag) - 1) &&
             stride <= std::numeric_limits<int32_t>::max() /
                           (size(tag) - 1)),
        "strided index sequence overflows i32");
    return construct_words<Backend>(
        IndexTag{}, [&]<nint_t Index>(IndexTag) VECOPS_INLINE_LAMBDA {
      auto word = execute_word<Index, Backend>(
          FillOp{}, IndexTag{}, int32_t{});
      constexpr nint_t word_lanes =
          RepresentationTraits<Backend, IndexTag>::word_lanes;
      const nint_t begin = Index * word_lanes;
      const nint_t end = std::min(begin + word_lanes, size(IndexTag{}));
      for (nint_t lane = begin; lane < end; ++lane) {
        word = execute_word<Index, Backend>(
            SetVecLaneOp{}, IndexTag{}, word, lane - begin,
            static_cast<int32_t>(lane * stride));
      }
      return word;
    });
  }
};

template <typename Backend, VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Rebind<int32_t, Tag>> make_strided_indices(
    Tag, nint_t stride) {
  using IndexTag = Rebind<int32_t, Tag>;
  return execute(StridedIndicesOp{}, IndexTag{}, stride);
}

/* **************************************************************************** */
//    GenericImpl for StoreOp: contiguous, indexed, and strided store           //
/* **************************************************************************** */

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, StoreOp, Tag> {
 private:
  template <VectorValue Indices, int Scale>
  static VECOPS_ALWAYS_INLINE void store_indexed(
      Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Mask<Tag> mask) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, Tag>;
    constexpr nint_t byte_scale =
        Scale == 0 ? static_cast<nint_t>(sizeof(ElementOf<Tag>)) : Scale;
    auto* base = reinterpret_cast<std::byte*>(pointer);
    for (nint_t lane = 0; lane < size(tag); ++lane) {
      if (!get_mask_lane_at<Backend>(tag, mask, lane)) continue;
      const auto index = get_vec_lane_at<Backend>(
            IndexTag{}, addressing.indices, lane);
      const auto stored = get_vec_lane_at<Backend>(tag, value, lane);
      std::memcpy(
          base + static_cast<nint_t>(index) * byte_scale, &stored,
          sizeof(stored));
    }
  }

  template <typename Alignment, typename Temporality, std::size_t... Index>
  static VECOPS_ALWAYS_INLINE void store_full(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      Alignment alignment, Temporality temporality,
      std::index_sequence<Index...>) {
    (execute_word<static_cast<nint_t>(Index), Backend>(
         op, tag,
         pointer + static_cast<nint_t>(Index) * native_word_size(tag),
         ::vecops::vec::get_word<static_cast<nint_t>(Index)>(tag, value),
         alignment, temporality), ...);
  }

  template <typename Alignment, typename Temporality, std::size_t... Index>
  static VECOPS_ALWAYS_INLINE void store_masked(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      Mask<Tag> mask, Alignment alignment, Temporality temporality,
      std::index_sequence<Index...>) {
    (execute_word<static_cast<nint_t>(Index), Backend>(
         op, tag,
         pointer + static_cast<nint_t>(Index) * native_word_size(tag),
         ::vecops::vec::get_word<static_cast<nint_t>(Index)>(tag, mask),
         ::vecops::vec::get_word<static_cast<nint_t>(Index)>(tag, value),
         alignment, temporality), ...);
  }

 public:
  template <typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      Alignment alignment, Temporality temporality) {
    store_full(
        op, tag, pointer, value, alignment, temporality,
        std::make_index_sequence<
            static_cast<std::size_t>(num_words(tag))>{});
  }

  template <typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      Mask<Tag> mask, Alignment alignment, Temporality temporality) {
    store_masked(
        op, tag, pointer, value, mask, alignment, temporality,
        std::make_index_sequence<
            static_cast<std::size_t>(num_words(tag))>{});
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Temporality) {
    store_indexed(
        tag, pointer, value, addressing,
        mfill(tag, true));
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Mask<Tag> mask,
      Temporality) {
    store_indexed(tag, pointer, value, addressing, mask);
  }
};

template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_load_options(
    LoadOp op, Tag tag, const ElementOf<Tag>* pointer,
    Options&&... options) {
  static_assert(
      valid_memory_options<Tag, false, Options...>(),
      "load received invalid, duplicate, or mismatched options");
  return execute_load_request(
      op, tag, pointer,
      resolve_load_request<Tag>(std::forward<Options>(options)...));
}

template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE void execute_store_options(
    StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Options&&... options) {
  static_assert(
      valid_memory_options<Tag, true, Options...>(),
      "store received invalid, duplicate, mismatched, or population options");
  execute_store_request(
      op, tag, pointer, value,
      resolve_store_request<Tag>(std::forward<Options>(options)...));
}

/* **************************************************************************** */
//    Request-driven execution                                                  //
/* **************************************************************************** */

/**
 * Executes a load from a resolved LoadRequest. Branching happens on the
 * request's kind template parameters; runtime values (stride, first-count,
 * mask, inactive population, indices) are read from the struct fields. The
 * emitted operations match the equivalent option-pack call exactly.
 */
template <typename Request>
VECOPS_ALWAYS_INLINE Vec<typename Request::TagType>
execute_load_request(LoadOp op, typename Request::TagType tag, const ElementOf<typename Request::TagType>* pointer,
                     const Request& request) {
  using Tag = typename Request::TagType;
  using Alignment = typename Request::AlignmentOption;
  using Temporality = typename Request::TemporalityOption;
  constexpr Active A = Request::active_kind;
  constexpr Addressing Addr = Request::addressing_kind;
  constexpr Populate P = Request::populate_kind;

  if constexpr ((Addr == Addressing::Strided && !strided_indices_supported<Tag>()) ||
                (Addr != Addressing::Contiguous &&
                 scalarize_indexed_memory_v<CurrentBackend, typename Request::ActiveResources>)) {
    return execute_scalar_load_request(tag, pointer, request);
  } else {
    if constexpr (Addr == Addressing::Strided) {
      if (!strided_indices_fit<ElementOf<Tag>>(tag, request)) [[unlikely]] {
        if constexpr (std::same_as<CurrentBackend, SVEBackend> &&
                      strided_indices_supported<Tag, int64_t>()) {
          const auto indices = make_wide_strided_indices(tag, request.stride);
          return execute_load_request(op, tag, pointer, reindex_request(request, indices));
        } else {
          return execute_scalar_load_request(tag, pointer, request);
        }
      }
    }
    if constexpr (std::same_as<Alignment, mem::Aligned>) {
      VECOPS_ASSERT(reinterpret_cast<std::uintptr_t>(pointer) % static_cast<std::uintptr_t>(memory_alignment(tag)) == 0,
                    "load pointer is not aligned to %zd bytes", memory_alignment(tag));
    }
    if constexpr (A == Active::Unmasked) {
      if constexpr (Addr == Addressing::Contiguous) {
        return execute(op, tag, pointer, Alignment{}, Temporality{});
      } else if constexpr (Addr == Addressing::Strided) {
        const auto indices = make_strided_indices<CurrentBackend>(tag, request.stride);
        if constexpr (Request::index_scale == 0) {
          return execute(op, tag, pointer, opt::indexed(indices), Temporality{});
        } else {
          return execute(op, tag, pointer, opt::indexed(indices, opt::scale<Request::index_scale>), Temporality{});
        }
      } else {
        opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{*request.indices};
        return execute(op, tag, pointer, addressing, Temporality{});
      }
    } else {
      const Mask<Tag> mask = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (A == Active::First) {
          return mwhilelt(tag, 0, request.first_count);
        } else {
          return *request.mask;
        }
      }();
      const Vec<Tag> inactive = [&]() VECOPS_INLINE_LAMBDA -> Vec<Tag> {
        if constexpr (P == Populate::MergeVector) {
          return *request.merge_vector;
        } else if constexpr (P == Populate::MergeScalar) {
          return fill(tag, request.merge_scalar);
        } else {
          return zeros(tag);
        }
      }();
      if constexpr (Addr == Addressing::Contiguous) {
        const auto loaded = execute(op, tag, pointer, mask, inactive, Alignment{}, Temporality{});
#ifdef VECOPS_DEBUG_TF
        fprintf(stderr, "[lexec] lanes=%zd mask:", (long)vec::size(tag));
        for (nint_t i = 0; i < vec::size(tag); ++i)
          fprintf(stderr, "%d", (int)vec::get(tag, mask, i));
        fprintf(stderr, " loaded:");
        for (nint_t i = 0; i < vec::size(tag); ++i)
          fprintf(stderr, " %.1f", (double)vec::get(tag, loaded, i));
        fprintf(stderr, "\n");
#endif
        return loaded;
      } else if constexpr (Addr == Addressing::Strided) {
        const auto indices = make_strided_indices<CurrentBackend>(tag, request.stride);
        if constexpr (Request::index_scale == 0) {
          return execute(op, tag, pointer, opt::indexed(indices), mask, inactive, Temporality{});
        } else {
          return execute(op, tag, pointer, opt::indexed(indices, opt::scale<Request::index_scale>), mask, inactive,
                         Temporality{});
        }
      } else {
        opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{*request.indices};
        return execute(op, tag, pointer, addressing, mask, inactive, Temporality{});
      }
    }
  }
}

/** Executes a store from a resolved StoreRequest. */
template <typename Request>
VECOPS_ALWAYS_INLINE void execute_store_request(StoreOp op, typename Request::TagType tag,
                                                ElementOf<typename Request::TagType>* pointer,
                                                Vec<typename Request::TagType> value, const Request& request) {
  using Tag = typename Request::TagType;
  using Alignment = typename Request::AlignmentOption;
  using Temporality = typename Request::TemporalityOption;
  constexpr Active A = Request::active_kind;
  constexpr Addressing Addr = Request::addressing_kind;

  if constexpr ((Addr == Addressing::Strided && !strided_indices_supported<Tag>()) ||
                (Addr != Addressing::Contiguous &&
                 scalarize_indexed_memory_v<CurrentBackend, typename Request::ActiveResources>)) {
    execute_scalar_store_request(tag, pointer, value, request);
  } else {
    if constexpr (Addr == Addressing::Strided) {
      if (!strided_indices_fit<ElementOf<Tag>>(tag, request)) [[unlikely]] {
        if constexpr (std::same_as<CurrentBackend, SVEBackend> &&
                      strided_indices_supported<Tag, int64_t>()) {
          const auto indices = make_wide_strided_indices(tag, request.stride);
          execute_store_request(op, tag, pointer, value, reindex_request(request, indices));
        } else {
          execute_scalar_store_request(tag, pointer, value, request);
        }
        return;
      }
    }
    if constexpr (std::same_as<Alignment, mem::Aligned>) {
      VECOPS_ASSERT(reinterpret_cast<std::uintptr_t>(pointer) % static_cast<std::uintptr_t>(memory_alignment(tag)) == 0,
                    "store pointer is not aligned to %zd bytes", memory_alignment(tag));
    }
    if constexpr (A == Active::Unmasked) {
      if constexpr (Addr == Addressing::Contiguous) {
        execute(op, tag, pointer, value, Alignment{}, Temporality{});
      } else if constexpr (Addr == Addressing::Strided) {
        const auto indices = make_strided_indices<CurrentBackend>(tag, request.stride);
        if constexpr (Request::index_scale == 0) {
          execute(op, tag, pointer, value, opt::indexed(indices), Temporality{});
        } else {
          execute(op, tag, pointer, value, opt::indexed(indices, opt::scale<Request::index_scale>), Temporality{});
        }
      } else {
        opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{*request.indices};
        execute(op, tag, pointer, value, addressing, Temporality{});
      }
    } else {
      const Mask<Tag> mask = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (A == Active::First) {
          return mwhilelt(tag, 0, request.first_count);
        } else {
          return *request.mask;
        }
      }();
      if constexpr (Addr == Addressing::Contiguous) {
        execute(op, tag, pointer, value, mask, Alignment{}, Temporality{});
      } else if constexpr (Addr == Addressing::Strided) {
        const auto indices = make_strided_indices<CurrentBackend>(tag, request.stride);
        if constexpr (Request::index_scale == 0) {
          execute(op, tag, pointer, value, opt::indexed(indices), mask, Temporality{});
        } else {
          execute(op, tag, pointer, value, opt::indexed(indices, opt::scale<Request::index_scale>), mask,
                  Temporality{});
        }
      } else {
        opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{*request.indices};
        execute(op, tag, pointer, value, addressing, mask, Temporality{});
      }
    }
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_MEMORY_H
