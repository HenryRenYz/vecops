// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_MEMORY_H
#define VECOPS_VEC_DETAILS_SVE_MEMORY_H

/**
 * @file Memory.h
 * @brief SVE backend implementations for load and store operations.
 */

#include <limits>
#include <type_traits>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/Options.h"

namespace vecops::vec::details {

template <typename Resources>
inline constexpr bool scalarize_indexed_memory_v<SVEBackend, Resources> =
#if defined(HAS_SME_FA64)
    false;
#else
    execution::details::has_resource_v<
        execution::details::arm::StreamingZA, Resources>;
#endif

// Float16/BFloat16 are two-byte wrapper types.  SVE bitwise memory leaves
// access their object representation through a may-alias scalar so GCC cannot
// assume that the intrinsic load/store is disjoint from the wrapper array.
using SVEAliasU16 = uint16_t __attribute__((__may_alias__));


/* **************************************************************************** */
//                    Load and store word implementations                     //
/* **************************************************************************** */

template <>
struct NativeWordImpl<SVEBackend, StridedIndicesOp> {
  template <nint_t Index, VectorTag IndexTag>
    requires std::same_as<ElementOf<IndexTag>, int32_t>
  static VECOPS_ALWAYS_INLINE NativeWordVec<IndexTag> call(
      StridedIndicesOp, IndexTag, nint_t stride) {
    const nint_t lanes = svcntw();
    VECOPS_ASSERT(
        stride >= std::numeric_limits<int32_t>::min() &&
            stride <= std::numeric_limits<int32_t>::max() &&
            (Index == 0 ||
             (stride >= std::numeric_limits<int32_t>::min() /
                               (Index * lanes) &&
              stride <= std::numeric_limits<int32_t>::max() /
                               (Index * lanes))),
        "strided index sequence overflows i32");
    return sve_basic_wrap_word<IndexTag>(svindex_s32(
        static_cast<int32_t>(Index * lanes * stride),
        static_cast<int32_t>(stride)));
  }
};

template <VectorTag IndexTag>
  requires std::same_as<ElementOf<IndexTag>, int32_t>
struct NativeImpl<SVEBackend, StridedIndicesOp, IndexTag> {
  static VECOPS_ALWAYS_INLINE Vec<IndexTag> call(
      StridedIndicesOp op, IndexTag tag, nint_t stride) {
    return construct_words<SVEBackend>(
        tag, [&]<nint_t Index>(IndexTag) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, SVEBackend>(op, tag, stride);
    });
  }
};

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_compact_indexed_memory_bits(
    svuint32_t bits) {
  using T = ElementOf<Tag>;
  if constexpr (sizeof(T) == 4) {
    if constexpr (std::same_as<T, float32_t>)
      return sve_basic_wrap_word<Tag>(svreinterpret_f32_u32(bits));
    else if constexpr (std::same_as<T, int32_t>)
      return sve_basic_wrap_word<Tag>(svreinterpret_s32_u32(bits));
    else
      return sve_basic_wrap_word<Tag>(bits);
  } else {
    const auto packed16 = svuzp1_u16(
        svreinterpret_u16_u32(bits), svdup_n_u16(0));
    if constexpr (sizeof(T) == 2) {
      if constexpr (std::same_as<T, int16_t>)
        return sve_basic_wrap_word<Tag>(svreinterpret_s16_u16(packed16));
      else if constexpr (std::same_as<T, float16_t>)
        return sve_basic_wrap_word<Tag>(svreinterpret_f16_u16(packed16));
      else if constexpr (std::same_as<T, bfloat16_t>)
        return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(packed16));
      else
        return sve_basic_wrap_word<Tag>(packed16);
    } else {
      const auto packed8 = svuzp1_u8(
          svreinterpret_u8_u16(packed16), svdup_n_u8(0));
      if constexpr (std::same_as<T, int8_t>)
        return sve_basic_wrap_word<Tag>(svreinterpret_s8_u8(packed8));
      else
        return sve_basic_wrap_word<Tag>(packed8);
    }
  }
}

template <VectorTag Tag>
  requires (sizeof(ElementOf<Tag>) == 8)
VECOPS_ALWAYS_INLINE Vec<Tag> sve_wrap_indexed_memory_bits(svuint64_t bits) {
  if constexpr (std::same_as<ElementOf<Tag>, float64_t>)
    return sve_basic_wrap_word<Tag>(svreinterpret_f64_u64(bits));
  else if constexpr (std::same_as<ElementOf<Tag>, int64_t>)
    return sve_basic_wrap_word<Tag>(svreinterpret_s64_u64(bits));
  else
    return sve_basic_wrap_word<Tag>(bits);
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_compact_indexed_memory_bits(
    svuint64_t bits) {
  const auto packed32 = svuzp1_u32(
      svreinterpret_u32_u64(bits), svdup_n_u32(0));
  return sve_compact_indexed_memory_bits<Tag>(packed32);
}

template <int Scale>
VECOPS_ALWAYS_INLINE svint32_t sve_scale_indexed_offsets(
    svbool_t active, svint32_t indices) {
  if constexpr (Scale == 1) return indices;
  else if constexpr (Scale == 2) return svlsl_n_s32_x(active, indices, 1);
  else if constexpr (Scale == 4) return svlsl_n_s32_x(active, indices, 2);
  else return svlsl_n_s32_x(active, indices, 3);
}

template <int Scale>
VECOPS_ALWAYS_INLINE svint64_t sve_scale_indexed_offsets(
    svbool_t active, svint64_t indices) {
  if constexpr (Scale == 1) return indices;
  else if constexpr (Scale == 2) return svlsl_n_s64_x(active, indices, 1);
  else if constexpr (Scale == 4) return svlsl_n_s64_x(active, indices, 2);
  else return svlsl_n_s64_x(active, indices, 3);
}

template <Element T, typename Temporality>
VECOPS_ALWAYS_INLINE svuint32_t sve_load_indexed_u32_bits(
    svbool_t active, const T* pointer, svint32_t offsets, Temporality) {
#if defined(HAS_SVE2)
  if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
    const auto unsigned_offsets = svreinterpret_u32_s32(offsets);
    if constexpr (sizeof(T) == 1)
      return svldnt1ub_gather_u32offset_u32(
          active, reinterpret_cast<const uint8_t*>(pointer), unsigned_offsets);
    else if constexpr (sizeof(T) == 2)
      return svldnt1uh_gather_u32offset_u32(
          active, reinterpret_cast<const uint16_t*>(pointer), unsigned_offsets);
    else if constexpr (std::same_as<T, float32_t>)
      return svreinterpret_u32_f32(svldnt1_gather_u32offset_f32(
          active, pointer, unsigned_offsets));
    else
      return svldnt1_gather_u32offset_u32(
          active, reinterpret_cast<const uint32_t*>(pointer), unsigned_offsets);
  }
#endif
  if constexpr (sizeof(T) == 1)
    return svld1ub_gather_s32offset_u32(
        active, reinterpret_cast<const uint8_t*>(pointer), offsets);
  else if constexpr (sizeof(T) == 2)
    return svld1uh_gather_s32offset_u32(
        active, reinterpret_cast<const uint16_t*>(pointer), offsets);
  else if constexpr (std::same_as<T, float32_t>)
    return svreinterpret_u32_f32(
        svld1_gather_s32offset_f32(active, pointer, offsets));
  else
    return svld1_gather_s32offset_u32(
        active, reinterpret_cast<const uint32_t*>(pointer), offsets);
}

template <Element T, typename Temporality>
VECOPS_ALWAYS_INLINE svuint64_t sve_load_indexed_u64_bits(
    svbool_t active, const T* pointer, svint64_t offsets, Temporality) {
#if defined(HAS_SVE2)
  if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
    if constexpr (sizeof(T) == 1)
      return svldnt1ub_gather_s64offset_u64(
          active, reinterpret_cast<const uint8_t*>(pointer), offsets);
    else if constexpr (sizeof(T) == 2)
      return svldnt1uh_gather_s64offset_u64(
          active, reinterpret_cast<const uint16_t*>(pointer), offsets);
    else if constexpr (sizeof(T) == 4)
      return svldnt1uw_gather_s64offset_u64(
          active, reinterpret_cast<const uint32_t*>(pointer), offsets);
    else if constexpr (std::same_as<T, float64_t>)
      return svreinterpret_u64_f64(
          svldnt1_gather_s64offset_f64(active, pointer, offsets));
    else
      return svldnt1_gather_s64offset_u64(
          active, reinterpret_cast<const uint64_t*>(pointer), offsets);
  }
#endif
  if constexpr (sizeof(T) == 1)
    return svld1ub_gather_s64offset_u64(
        active, reinterpret_cast<const uint8_t*>(pointer), offsets);
  else if constexpr (sizeof(T) == 2)
    return svld1uh_gather_s64offset_u64(
        active, reinterpret_cast<const uint16_t*>(pointer), offsets);
  else if constexpr (sizeof(T) == 4)
    return svld1uw_gather_s64offset_u64(
        active, reinterpret_cast<const uint32_t*>(pointer), offsets);
  else if constexpr (std::same_as<T, float64_t>)
    return svreinterpret_u64_f64(
        svld1_gather_s64offset_f64(active, pointer, offsets));
  else
    return svld1_gather_s64offset_u64(
        active, reinterpret_cast<const uint64_t*>(pointer), offsets);
}

template <int Scale, VectorTag Tag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_load_indexed_memory_leaf(
    Tag tag, const ElementOf<Tag>* pointer, Vec<IndexTag> indices,
    Mask<Tag> mask, Vec<Tag> inactive, Temporality temporality) {
  using T = ElementOf<Tag>;
  using I = ElementOf<IndexTag>;
  auto active = ::vecops::vec::get_word<0>(tag, mask);
  if constexpr (sizeof(T) < 2) active = svunpklo_b(active);
  if constexpr (sizeof(T) < 4) active = svunpklo_b(active);
  if constexpr (sizeof(T) < 8 && sizeof(I) == 8)
    active = svunpklo_b(active);
  const auto index_raw = sve_basic_raw_word(indices);
  Vec<Tag> loaded;
  if constexpr (sizeof(I) == 4) {
    const auto offsets32 = sve_scale_indexed_offsets<Scale>(active, index_raw);
    if constexpr (sizeof(T) <= 4) {
      loaded = sve_compact_indexed_memory_bits<Tag>(
          sve_load_indexed_u32_bits(active, pointer, offsets32, temporality));
    } else {
      const auto offsets64 = svunpklo_s64(offsets32);
      loaded = sve_wrap_indexed_memory_bits<Tag>(
          sve_load_indexed_u64_bits(active, pointer, offsets64, temporality));
    }
  } else {
    const auto offsets64 = sve_scale_indexed_offsets<Scale>(active, index_raw);
    const auto bits =
        sve_load_indexed_u64_bits(active, pointer, offsets64, temporality);
    if constexpr (sizeof(T) == 8)
      loaded = sve_wrap_indexed_memory_bits<Tag>(bits);
    else
      loaded = sve_compact_indexed_memory_bits<Tag>(bits);
  }
  return blend(tag, inactive, mask, loaded);
}

template <int Scale, VectorTag Tag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_load_indexed_memory(
    Tag tag, const ElementOf<Tag>* pointer, Vec<IndexTag> indices,
    Mask<Tag> mask, Vec<Tag> inactive, Temporality temporality) {
  if constexpr (num_words(tag) > 1 || num_words(IndexTag{}) > 1) {
    const auto lower_value = sve_load_indexed_memory<
        Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, lower(IndexTag{}, indices),
        lower(tag, mask), lower(tag, inactive),
        temporality);
    const auto upper_value = sve_load_indexed_memory<
        Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, upper(IndexTag{}, indices),
        upper(tag, mask), upper(tag, inactive),
        temporality);
    return concat(tag, lower_value, upper_value);
  } else {
    return sve_load_indexed_memory_leaf<Scale, Tag, IndexTag>(
        tag, pointer, indices, mask, inactive, temporality);
  }
}

template <VectorTag Tag>
struct NativeImpl<SVEBackend, LoadOp, Tag> {
  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp op, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    return call(
        op, tag, pointer, addressing, mfill(tag, true),
        fill(tag, ElementOf<Tag>{}), temporality);
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing,
      Mask<Tag> mask, Vec<Tag> inactive, Temporality temporality) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, Tag>;
    constexpr int scale = Scale == 0 ? sizeof(ElementOf<Tag>) : Scale;
    return sve_load_indexed_memory<scale, Tag, IndexTag>(
        tag, pointer, addressing.indices, mask, inactive, temporality);
  }
};

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE svuint32_t sve_expand_indexed_store_u32_bits(
    Vec<Tag> value) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (sizeof(T) == 4) {
    if constexpr (std::same_as<T, float32_t>)
      return svreinterpret_u32_f32(raw);
    else if constexpr (std::same_as<T, int32_t>)
      return svreinterpret_u32_s32(raw);
    else
      return raw;
  } else if constexpr (sizeof(T) == 2) {
    const auto bits = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<T, float16_t>)
        return svreinterpret_u16_f16(raw);
      else if constexpr (std::same_as<T, bfloat16_t>)
        return svreinterpret_u16_bf16(raw);
      else if constexpr (std::same_as<T, int16_t>)
        return svreinterpret_u16_s16(raw);
      else
        return raw;
    }();
    return svunpklo_u32(bits);
  } else {
    const auto bits = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<T, int8_t>)
        return svreinterpret_u8_s8(raw);
      else
        return raw;
    }();
    return svunpklo_u32(svunpklo_u16(bits));
  }
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE svuint64_t sve_expand_indexed_store_u64_bits(
    Vec<Tag> value) {
  using T = ElementOf<Tag>;
  if constexpr (sizeof(T) == 8) {
    const auto raw = sve_basic_raw_word(value);
    if constexpr (std::same_as<T, float64_t>)
      return svreinterpret_u64_f64(raw);
    else if constexpr (std::same_as<T, int64_t>)
      return svreinterpret_u64_s64(raw);
    else
      return raw;
  } else {
    return svunpklo_u64(sve_expand_indexed_store_u32_bits<Tag>(value));
  }
}

template <Element T, typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_indexed_u32_bits(
    svbool_t active, T* pointer, svint32_t offsets, svuint32_t bits,
    Temporality) {
#if defined(HAS_SVE2)
  if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
    const auto unsigned_offsets = svreinterpret_u32_s32(offsets);
    if constexpr (sizeof(T) == 1)
      svstnt1b_scatter_u32offset_u32(
          active, reinterpret_cast<uint8_t*>(pointer), unsigned_offsets, bits);
    else if constexpr (sizeof(T) == 2)
      svstnt1h_scatter_u32offset_u32(
          active, reinterpret_cast<uint16_t*>(pointer), unsigned_offsets, bits);
    else
      svstnt1_scatter_u32offset_u32(
          active, reinterpret_cast<uint32_t*>(pointer), unsigned_offsets, bits);
    return;
  }
#endif
  if constexpr (sizeof(T) == 1)
    svst1b_scatter_s32offset_u32(
        active, reinterpret_cast<uint8_t*>(pointer), offsets, bits);
  else if constexpr (sizeof(T) == 2)
    svst1h_scatter_s32offset_u32(
        active, reinterpret_cast<uint16_t*>(pointer), offsets, bits);
  else
    svst1_scatter_s32offset_u32(
        active, reinterpret_cast<uint32_t*>(pointer), offsets, bits);
}

template <Element T, typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_indexed_u64_bits(
    svbool_t active, T* pointer, svint64_t offsets, svuint64_t bits,
    Temporality) {
#if defined(HAS_SVE2)
  if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
    if constexpr (sizeof(T) == 1)
      svstnt1b_scatter_s64offset_u64(
          active, reinterpret_cast<uint8_t*>(pointer), offsets, bits);
    else if constexpr (sizeof(T) == 2)
      svstnt1h_scatter_s64offset_u64(
          active, reinterpret_cast<uint16_t*>(pointer), offsets, bits);
    else if constexpr (sizeof(T) == 4)
      svstnt1w_scatter_s64offset_u64(
          active, reinterpret_cast<uint32_t*>(pointer), offsets, bits);
    else
      svstnt1_scatter_s64offset_u64(
          active, reinterpret_cast<uint64_t*>(pointer), offsets, bits);
    return;
  }
#endif
  if constexpr (sizeof(T) == 1)
    svst1b_scatter_s64offset_u64(
        active, reinterpret_cast<uint8_t*>(pointer), offsets, bits);
  else if constexpr (sizeof(T) == 2)
    svst1h_scatter_s64offset_u64(
        active, reinterpret_cast<uint16_t*>(pointer), offsets, bits);
  else if constexpr (sizeof(T) == 4)
    svst1w_scatter_s64offset_u64(
        active, reinterpret_cast<uint32_t*>(pointer), offsets, bits);
  else
    svst1_scatter_s64offset_u64(
        active, reinterpret_cast<uint64_t*>(pointer), offsets, bits);
}

template <int Scale, VectorTag Tag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_indexed_memory_leaf(
    Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Vec<IndexTag> indices, Mask<Tag> mask, Temporality temporality) {
  using T = ElementOf<Tag>;
  using I = ElementOf<IndexTag>;
  auto active = ::vecops::vec::get_word<0>(tag, mask);
  if constexpr (sizeof(T) < 2) active = svunpklo_b(active);
  if constexpr (sizeof(T) < 4) active = svunpklo_b(active);
  if constexpr (sizeof(T) < 8 && sizeof(I) == 8)
    active = svunpklo_b(active);
  const auto index_raw = sve_basic_raw_word(indices);
  if constexpr (sizeof(I) == 4) {
    const auto offsets = sve_scale_indexed_offsets<Scale>(active, index_raw);
    if constexpr (sizeof(T) <= 4)
      sve_store_indexed_u32_bits(
          active, pointer, offsets,
          sve_expand_indexed_store_u32_bits<Tag>(value), temporality);
    else
      sve_store_indexed_u64_bits(
          active, pointer, svunpklo_s64(offsets),
          sve_expand_indexed_store_u64_bits<Tag>(value), temporality);
  } else {
    const auto offsets = sve_scale_indexed_offsets<Scale>(active, index_raw);
    sve_store_indexed_u64_bits(
        active, pointer, offsets,
        sve_expand_indexed_store_u64_bits<Tag>(value), temporality);
  }
}

template <int Scale, VectorTag Tag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_indexed_memory(
    Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Vec<IndexTag> indices, Mask<Tag> mask, Temporality temporality) {
  if constexpr (num_words(tag) > 1 || num_words(IndexTag{}) > 1) {
    sve_store_indexed_memory<Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, lower(tag, value),
        lower(IndexTag{}, indices),
        lower(tag, mask), temporality);
    sve_store_indexed_memory<Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, upper(tag, value),
        upper(IndexTag{}, indices),
        upper(tag, mask), temporality);
  } else {
    sve_store_indexed_memory_leaf<Scale, Tag, IndexTag>(
        tag, pointer, value, indices, mask, temporality);
  }
}

template <VectorTag Tag>
struct NativeImpl<SVEBackend, StoreOp, Tag> {
  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    call(
        op, tag, pointer, value, addressing,
        mfill(tag, true), temporality);
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Mask<Tag> mask,
      Temporality temporality) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, Tag>;
    constexpr int scale = Scale == 0 ? sizeof(ElementOf<Tag>) : Scale;
    sve_store_indexed_memory<scale, Tag, IndexTag>(
        tag, pointer, value, addressing.indices, mask, temporality);
  }
};

template <Element T, typename Temporality>
VECOPS_ALWAYS_INLINE auto sve_load_memory_word(
    svbool_t mask, const T* pointer, Temporality) {
  constexpr bool non_temporal =
      std::same_as<Temporality, mem::NonTemporal>;
  if constexpr (std::same_as<T, bfloat16_t>) {
    if constexpr (non_temporal)
      return svreinterpret_bf16_u16(
          svldnt1_u16(mask, reinterpret_cast<const SVEAliasU16*>(pointer)));
    else
      return svreinterpret_bf16_u16(
          svld1_u16(mask, reinterpret_cast<const SVEAliasU16*>(pointer)));
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (non_temporal)
      return svreinterpret_f16_u16(
          svldnt1_u16(mask, reinterpret_cast<const SVEAliasU16*>(pointer)));
    else
      return svreinterpret_f16_u16(
          svld1_u16(mask, reinterpret_cast<const SVEAliasU16*>(pointer)));
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (non_temporal) return svldnt1_f32(mask, pointer);
    else return svld1_f32(mask, pointer);
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (non_temporal) return svldnt1_f64(mask, pointer);
    else return svld1_f64(mask, pointer);
  } else if constexpr (std::same_as<T, int8_t>) {
    if constexpr (non_temporal) return svldnt1_s8(mask, pointer);
    else return svld1_s8(mask, pointer);
  } else if constexpr (std::same_as<T, uint8_t>) {
    if constexpr (non_temporal) return svldnt1_u8(mask, pointer);
    else return svld1_u8(mask, pointer);
  } else if constexpr (std::same_as<T, int16_t>) {
    if constexpr (non_temporal) return svldnt1_s16(mask, pointer);
    else return svld1_s16(mask, pointer);
  } else if constexpr (std::same_as<T, uint16_t>) {
    if constexpr (non_temporal) return svldnt1_u16(mask, pointer);
    else return svld1_u16(mask, pointer);
  } else if constexpr (std::same_as<T, int32_t>) {
    if constexpr (non_temporal) return svldnt1_s32(mask, pointer);
    else return svld1_s32(mask, pointer);
  } else if constexpr (std::same_as<T, uint32_t>) {
    if constexpr (non_temporal) return svldnt1_u32(mask, pointer);
    else return svld1_u32(mask, pointer);
  } else if constexpr (std::same_as<T, int64_t>) {
    if constexpr (non_temporal) return svldnt1_s64(mask, pointer);
    else return svld1_s64(mask, pointer);
  } else if constexpr (std::same_as<T, uint64_t>) {
    if constexpr (non_temporal) return svldnt1_u64(mask, pointer);
    else return svld1_u64(mask, pointer);
  } else {
    static_assert(dispatch_dependent_false<T>, "unsupported SVE load type");
  }
}

template <Element T, typename Raw, typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_memory_word(
    svbool_t mask, T* pointer, Raw value, Temporality) {
  constexpr bool non_temporal =
      std::same_as<Temporality, mem::NonTemporal>;
  if constexpr (std::same_as<T, bfloat16_t>) {
    if constexpr (non_temporal)
      svstnt1_u16(
          mask, reinterpret_cast<SVEAliasU16*>(pointer),
          svreinterpret_u16_bf16(value));
    else
      svst1_u16(
          mask, reinterpret_cast<SVEAliasU16*>(pointer),
          svreinterpret_u16_bf16(value));
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (non_temporal)
      svstnt1_u16(
          mask, reinterpret_cast<SVEAliasU16*>(pointer),
          svreinterpret_u16_f16(value));
    else
      svst1_u16(
          mask, reinterpret_cast<SVEAliasU16*>(pointer),
          svreinterpret_u16_f16(value));
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (non_temporal) svstnt1_f32(mask, pointer, value);
    else svst1_f32(mask, pointer, value);
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (non_temporal) svstnt1_f64(mask, pointer, value);
    else svst1_f64(mask, pointer, value);
  } else if constexpr (std::same_as<T, int8_t>) {
    if constexpr (non_temporal) svstnt1_s8(mask, pointer, value);
    else svst1_s8(mask, pointer, value);
  } else if constexpr (std::same_as<T, uint8_t>) {
    if constexpr (non_temporal) svstnt1_u8(mask, pointer, value);
    else svst1_u8(mask, pointer, value);
  } else if constexpr (std::same_as<T, int16_t>) {
    if constexpr (non_temporal) svstnt1_s16(mask, pointer, value);
    else svst1_s16(mask, pointer, value);
  } else if constexpr (std::same_as<T, uint16_t>) {
    if constexpr (non_temporal) svstnt1_u16(mask, pointer, value);
    else svst1_u16(mask, pointer, value);
  } else if constexpr (std::same_as<T, int32_t>) {
    if constexpr (non_temporal) svstnt1_s32(mask, pointer, value);
    else svst1_s32(mask, pointer, value);
  } else if constexpr (std::same_as<T, uint32_t>) {
    if constexpr (non_temporal) svstnt1_u32(mask, pointer, value);
    else svst1_u32(mask, pointer, value);
  } else if constexpr (std::same_as<T, int64_t>) {
    if constexpr (non_temporal) svstnt1_s64(mask, pointer, value);
    else svst1_s64(mask, pointer, value);
  } else if constexpr (std::same_as<T, uint64_t>) {
    if constexpr (non_temporal) svstnt1_u64(mask, pointer, value);
    else svst1_u64(mask, pointer, value);
  } else {
    static_assert(dispatch_dependent_false<T>, "unsupported SVE store type");
  }
}

template <>
struct NativeWordImpl<SVEBackend, LoadOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      Alignment, Temporality temporality) {
    const auto active = sve_valid_word_predicate<Index>(tag);
    return sve_basic_wrap_word<Tag>(
        sve_load_memory_word(active, pointer, temporality));
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Alignment, Temporality temporality) {
    const auto active = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sve_word_is_always_full<Tag>()) {
        return mask;
      } else {
        const auto valid = sve_valid_word_predicate<Index>(tag);
        return svand_b_z(valid, valid, mask);
      }
    }();
    const auto loaded = sve_basic_wrap_word<Tag>(
        sve_load_memory_word(active, pointer, temporality));
    return execute_word<Index, SVEBackend>(
        BlendOp{}, tag, inactive, active, loaded);
  }
};

template <>
struct NativeWordImpl<SVEBackend, StoreOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, NativeWordVec<Tag> value,
      Alignment, Temporality temporality) {
    const auto active = sve_valid_word_predicate<Index>(tag);
    sve_store_memory_word(
        active, pointer, sve_basic_raw_word(value), temporality);
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> value,
      Alignment, Temporality temporality) {
    const auto active = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sve_word_is_always_full<Tag>()) {
        return mask;
      } else {
        const auto valid = sve_valid_word_predicate<Index>(tag);
        return svand_b_z(valid, valid, mask);
      }
    }();
    sve_store_memory_word(
        active, pointer, sve_basic_raw_word(value), temporality);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MEMORY_H
