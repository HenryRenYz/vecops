#ifndef VECOPS_VEC_DETAILS_MEMORY_H
#define VECOPS_VEC_DETAILS_MEMORY_H

/**
 * @file Memory.h
 * @brief Memory operation infrastructure: option validation
 * (valid_memory_options, is_memory_option_for), GenericImpl for LoadOp
 * and StoreOp (contiguous, indexed, and strided access), and
 * StridedIndicesOp for building linear index sequences.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/vec/Request.h"
#include "vecops/vec/details/Options.h"
#include "vecops/vec/details/Request.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Option validation for memory operations                                   //
/* **************************************************************************** */
// The IsMemory*Option group predicates live in details/Options.h.

template <VectorTag Tag, typename Option>
inline constexpr bool is_memory_addressing_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (IsIndexedOption<Clean>::value) {
    using Indices = typename IsIndexedOption<Clean>::Value;
    using IndexElement = ElementOf<VecToTagT<Indices>>;
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
inline constexpr bool is_memory_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (
      IsMemoryAlignmentOption<Clean>::value ||
      IsMemoryTemporalityOption<Clean>::value ||
      IsUnmaskedOption<Clean>::value ||
      IsFirstOption<Clean>::value) {
    return true;
  } else if constexpr (IsMaskedOption<Clean>::value) {
    return is_masked_option_for<Tag, Clean>;
  } else if constexpr (IsMemoryAddressingOption<Clean>::value) {
    return is_memory_addressing_option_for<Tag, Clean>;
  } else if constexpr (!IsStore && IsMemoryPopulationOption<Clean>::value) {
    return is_vector_population_option_for<Tag, Clean>;
  } else {
    return false;
  }
}();

template <VectorTag Tag, bool IsStore, typename... Options>
consteval bool valid_memory_options() {
  if constexpr (!(is_memory_option_for<Tag, IsStore, Options> && ...)) {
    return false;
  } else {
    constexpr std::size_t alignment_count =
        option_count<IsMemoryAlignmentOption, Options...>;
    constexpr std::size_t temporality_count =
        option_count<IsMemoryTemporalityOption, Options...>;
    constexpr std::size_t active_count =
        option_count<IsMemoryActiveOption, Options...>;
    constexpr std::size_t population_count =
        option_count<IsMemoryPopulationOption, Options...>;
    constexpr std::size_t indexed_count =
        option_count<IsIndexedOption, Options...>;
    constexpr std::size_t strided_count =
        option_count<IsStridedOption, Options...>;
    return alignment_count <= 1 && temporality_count <= 1 &&
        active_count <= 1 && population_count <= 1 &&
        indexed_count <= 1 && strided_count <= 1 &&
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
  template <VectorTag IndexTag>
  static VECOPS_ALWAYS_INLINE ElementOf<IndexTag> index_lane(
      IndexTag index_tag, Vec<IndexTag> indices, nint_t lane) {
    const nint_t word_lanes = native_word_size(index_tag);
    return visit_runtime_word<Backend>(
        index_tag, lane / word_lanes, [&]<nint_t Index>() {
          return execute_word<Index, Backend>(
              GetVecLaneOp{}, index_tag,
              ::vecops::vec::get_word<Index>(index_tag, indices),
              lane % word_lanes);
        });
  }

  template <VectorValue Indices, int Scale>
  static VECOPS_ALWAYS_INLINE Vec<Tag> load_indexed(
      Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing,
      Mask<Tag> mask, Vec<Tag> inactive) {
    using IndexTag = Rebind<ElementOf<VecToTagT<Indices>>, Tag>;
    constexpr nint_t byte_scale =
        Scale == 0 ? static_cast<nint_t>(sizeof(ElementOf<Tag>)) : Scale;
    const auto* base = reinterpret_cast<const std::byte*>(pointer);
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
        const auto index = index_lane(IndexTag{}, addressing.indices, lane);
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
        tag, pointer, addressing, execute(MaskFillOp{}, tag, true),
        execute(FillOp{}, tag, ElementOf<Tag>{}));
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
  template <VectorTag IndexTag>
  static VECOPS_ALWAYS_INLINE ElementOf<IndexTag> index_lane(
      IndexTag index_tag, Vec<IndexTag> indices, nint_t lane) {
    const nint_t word_lanes = native_word_size(index_tag);
    return visit_runtime_word<Backend>(
        index_tag, lane / word_lanes, [&]<nint_t Index>() {
          return execute_word<Index, Backend>(
              GetVecLaneOp{}, index_tag,
              ::vecops::vec::get_word<Index>(index_tag, indices),
              lane % word_lanes);
        });
  }

  static VECOPS_ALWAYS_INLINE bool mask_lane(
      Tag tag, Mask<Tag> mask, nint_t lane) {
    const nint_t word_lanes = native_word_size(tag);
    return visit_runtime_word<Backend>(
        tag, lane / word_lanes, [&]<nint_t Index>() {
          return execute_word<Index, Backend>(
              GetMaskLaneOp{}, tag,
              ::vecops::vec::get_word<Index>(tag, mask),
              lane % word_lanes);
        });
  }

  static VECOPS_ALWAYS_INLINE ElementOf<Tag> value_lane(
      Tag tag, Vec<Tag> value, nint_t lane) {
    const nint_t word_lanes = native_word_size(tag);
    return visit_runtime_word<Backend>(
        tag, lane / word_lanes, [&]<nint_t Index>() {
          return execute_word<Index, Backend>(
              GetVecLaneOp{}, tag,
              ::vecops::vec::get_word<Index>(tag, value),
              lane % word_lanes);
        });
  }

  template <VectorValue Indices, int Scale>
  static VECOPS_ALWAYS_INLINE void store_indexed(
      Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Mask<Tag> mask) {
    using IndexTag = Rebind<ElementOf<VecToTagT<Indices>>, Tag>;
    constexpr nint_t byte_scale =
        Scale == 0 ? static_cast<nint_t>(sizeof(ElementOf<Tag>)) : Scale;
    auto* base = reinterpret_cast<std::byte*>(pointer);
    for (nint_t lane = 0; lane < size(tag); ++lane) {
      if (!mask_lane(tag, mask, lane)) continue;
      const auto index = index_lane(IndexTag{}, addressing.indices, lane);
      const auto stored = value_lane(tag, value, lane);
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
        execute(MaskFillOp{}, tag, true));
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
  constexpr std::size_t active_count =
      option_count<IsMemoryActiveOption, Options...>;
  constexpr std::size_t population_count =
      option_count<IsMemoryPopulationOption, Options...>;
  constexpr std::size_t alignment_count =
      option_count<IsMemoryAlignmentOption, Options...>;
  constexpr std::size_t temporality_count =
      option_count<IsMemoryTemporalityOption, Options...>;
  constexpr std::size_t addressing_count =
      option_count<IsMemoryAddressingOption, Options...>;

  auto invoke = [&](auto alignment, auto temporality) -> Vec<Tag> {
    if constexpr (std::same_as<decltype(alignment), mem::Aligned>) {
      VECOPS_ASSERT(
          reinterpret_cast<std::uintptr_t>(pointer) %
                  static_cast<std::uintptr_t>(memory_alignment(tag)) ==
              0,
          "load pointer is not aligned to %zd bytes", memory_alignment(tag));
    }
    if constexpr (
        active_count == 0 ||
        option_count<IsUnmaskedOption, Options...> == 1) {
      if constexpr (addressing_count == 0) {
        return execute(op, tag, pointer, alignment, temporality);
      } else {
        const auto addressing = find_option<IsMemoryAddressingOption>(
            std::forward<Options>(options)...);
        if constexpr (
            IsIndexedOption<std::remove_cvref_t<decltype(addressing)>>::value) {
          return execute(op, tag, pointer, addressing, temporality);
        } else {
          auto indices = make_strided_indices<CurrentBackend>(
              tag, static_cast<nint_t>(addressing.stride));
          return execute(
              op, tag, pointer, opt::indexed(indices), temporality);
        }
      }
    } else {
      Mask<Tag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        mask = mwhilelt(tag, 0, count);
      }

      Vec<Tag> inactive;
      if constexpr (population_count == 0 ||
                    option_count<IsZeroOption, Options...> == 1) {
        inactive = zeros(tag);
      } else if constexpr (
          option_count<IsVectorMergeOption, Options...> == 1) {
        inactive = find_option<IsVectorMergeOption>(
            std::forward<Options>(options)...).value;
      } else {
        inactive = fill(
            tag, find_option<IsScalarMergeOption>(
                     std::forward<Options>(options)...).value);
      }
      if constexpr (addressing_count == 0) {
        return execute(
            op, tag, pointer, mask, inactive, alignment, temporality);
      } else {
        const auto addressing = find_option<IsMemoryAddressingOption>(
            std::forward<Options>(options)...);
        if constexpr (
            IsIndexedOption<std::remove_cvref_t<decltype(addressing)>>::value) {
          return execute(
              op, tag, pointer, addressing, mask, inactive, temporality);
        } else {
          auto indices = make_strided_indices<CurrentBackend>(
              tag, static_cast<nint_t>(addressing.stride));
          return execute(
              op, tag, pointer, opt::indexed(indices), mask, inactive,
              temporality);
        }
      }
    }
  };

  if constexpr (alignment_count == 1) {
    const auto alignment = find_option<IsMemoryAlignmentOption>(
        std::forward<Options>(options)...);
    if constexpr (temporality_count == 1) {
      return invoke(
          alignment, find_option<IsMemoryTemporalityOption>(
                         std::forward<Options>(options)...));
    } else {
      return invoke(alignment, mem::temporal);
    }
  } else if constexpr (temporality_count == 1) {
    return invoke(
        mem::unaligned, find_option<IsMemoryTemporalityOption>(
                            std::forward<Options>(options)...));
  } else {
    return invoke(mem::unaligned, mem::temporal);
  }
}

template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE void execute_store_options(
    StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Options&&... options) {
  static_assert(
      valid_memory_options<Tag, true, Options...>(),
      "store received invalid, duplicate, mismatched, or population options");
  constexpr std::size_t active_count =
      option_count<IsMemoryActiveOption, Options...>;
  constexpr std::size_t alignment_count =
      option_count<IsMemoryAlignmentOption, Options...>;
  constexpr std::size_t temporality_count =
      option_count<IsMemoryTemporalityOption, Options...>;
  constexpr std::size_t addressing_count =
      option_count<IsMemoryAddressingOption, Options...>;

  auto invoke = [&](auto alignment, auto temporality) {
    if constexpr (std::same_as<decltype(alignment), mem::Aligned>) {
      VECOPS_ASSERT(
          reinterpret_cast<std::uintptr_t>(pointer) %
                  static_cast<std::uintptr_t>(memory_alignment(tag)) ==
              0,
          "store pointer is not aligned to %zd bytes", memory_alignment(tag));
    }
    if constexpr (
        active_count == 0 ||
        option_count<IsUnmaskedOption, Options...> == 1) {
      if constexpr (addressing_count == 0) {
        execute(op, tag, pointer, value, alignment, temporality);
      } else {
        const auto addressing = find_option<IsMemoryAddressingOption>(
            std::forward<Options>(options)...);
        if constexpr (
            IsIndexedOption<std::remove_cvref_t<decltype(addressing)>>::value) {
          execute(op, tag, pointer, value, addressing, temporality);
        } else {
          auto indices = make_strided_indices<CurrentBackend>(
              tag, static_cast<nint_t>(addressing.stride));
          execute(
              op, tag, pointer, value, opt::indexed(indices), temporality);
        }
      }
    } else {
      Mask<Tag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        mask = mwhilelt(tag, 0, count);
      }
      if constexpr (addressing_count == 0) {
        execute(op, tag, pointer, value, mask, alignment, temporality);
      } else {
        const auto addressing = find_option<IsMemoryAddressingOption>(
            std::forward<Options>(options)...);
        if constexpr (
            IsIndexedOption<std::remove_cvref_t<decltype(addressing)>>::value) {
          execute(
              op, tag, pointer, value, addressing, mask, temporality);
        } else {
          auto indices = make_strided_indices<CurrentBackend>(
              tag, static_cast<nint_t>(addressing.stride));
          execute(
              op, tag, pointer, value, opt::indexed(indices), mask,
              temporality);
        }
      }
    }
  };

  if constexpr (alignment_count == 1) {
    const auto alignment = find_option<IsMemoryAlignmentOption>(
        std::forward<Options>(options)...);
    if constexpr (temporality_count == 1) {
      invoke(
          alignment, find_option<IsMemoryTemporalityOption>(
                         std::forward<Options>(options)...));
    } else {
      invoke(alignment, mem::temporal);
    }
  } else if constexpr (temporality_count == 1) {
    invoke(
        mem::unaligned, find_option<IsMemoryTemporalityOption>(
                            std::forward<Options>(options)...));
  } else {
    invoke(mem::unaligned, mem::temporal);
  }
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
VECOPS_ALWAYS_INLINE Vec<typename Request::TagType> execute_load_request(
    LoadOp op, typename Request::TagType tag,
    const ElementOf<typename Request::TagType>* pointer,
    const Request& request) {
  using Tag = typename Request::TagType;
  using Alignment = typename Request::AlignmentOption;
  using Temporality = typename Request::TemporalityOption;
  constexpr Active A = Request::active_kind;
  constexpr Addressing Addr = Request::addressing_kind;
  constexpr Populate P = Request::populate_kind;

  if constexpr (std::same_as<Alignment, mem::Aligned>) {
    VECOPS_ASSERT(
        reinterpret_cast<std::uintptr_t>(pointer) %
                static_cast<std::uintptr_t>(memory_alignment(tag)) ==
            0,
        "load pointer is not aligned to %zd bytes", memory_alignment(tag));
  }
  if constexpr (A == Active::Unmasked) {
    if constexpr (Addr == Addressing::Contiguous) {
      return execute(op, tag, pointer, Alignment{}, Temporality{});
    } else if constexpr (Addr == Addressing::Strided) {
      const auto indices =
          make_strided_indices<CurrentBackend>(tag, request.stride);
      return execute(
          op, tag, pointer, opt::indexed(indices), Temporality{});
    } else {
      opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{
          *request.indices};
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
      return execute(
          op, tag, pointer, mask, inactive, Alignment{}, Temporality{});
    } else if constexpr (Addr == Addressing::Strided) {
      const auto indices =
          make_strided_indices<CurrentBackend>(tag, request.stride);
      return execute(
          op, tag, pointer, opt::indexed(indices), mask, inactive,
          Temporality{});
    } else {
      opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{
          *request.indices};
      return execute(
          op, tag, pointer, addressing, mask, inactive, Temporality{});
    }
  }
}

/** Executes a store from a resolved StoreRequest. */
template <typename Request>
VECOPS_ALWAYS_INLINE void execute_store_request(
    StoreOp op, typename Request::TagType tag,
    ElementOf<typename Request::TagType>* pointer,
    Vec<typename Request::TagType> value, const Request& request) {
  using Tag = typename Request::TagType;
  using Alignment = typename Request::AlignmentOption;
  using Temporality = typename Request::TemporalityOption;
  constexpr Active A = Request::active_kind;
  constexpr Addressing Addr = Request::addressing_kind;

  if constexpr (std::same_as<Alignment, mem::Aligned>) {
    VECOPS_ASSERT(
        reinterpret_cast<std::uintptr_t>(pointer) %
                static_cast<std::uintptr_t>(memory_alignment(tag)) ==
            0,
        "store pointer is not aligned to %zd bytes", memory_alignment(tag));
  }
  if constexpr (A == Active::Unmasked) {
    if constexpr (Addr == Addressing::Contiguous) {
      execute(op, tag, pointer, value, Alignment{}, Temporality{});
    } else if constexpr (Addr == Addressing::Strided) {
      const auto indices =
          make_strided_indices<CurrentBackend>(tag, request.stride);
      execute(op, tag, pointer, value, opt::indexed(indices), Temporality{});
    } else {
      opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{
          *request.indices};
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
      const auto indices =
          make_strided_indices<CurrentBackend>(tag, request.stride);
      execute(
          op, tag, pointer, value, opt::indexed(indices), mask,
          Temporality{});
    } else {
      opt::Indexed<typename Request::IndexVectorType, Request::index_scale> addressing{
          *request.indices};
      execute(
          op, tag, pointer, value, addressing, mask, Temporality{});
    }
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_MEMORY_H
