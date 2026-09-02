//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_AMX_PACK_H
#define VECOPS_MATMUL_DETAILS_PACK_AMX_PACK_H

#include <cstdint>

#include <immintrin.h>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/matmul/details/packing/generic/Pack.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_pack_details::amx {

template <nint_t KTile, vec::VectorTag Tag, bool SpatialGuaranteed,
          bool KGuaranteed = false,
          typename Source, typename T>
VECOPS_ALWAYS_INLINE void pack_a_tile(
    const Source& source, T* destination,
    nint_t spatial_base, nint_t k_base,
    nint_t logical_spatial, nint_t logical_k) {
  static_assert(std::same_as<vec::ElementOf<Tag>, T>);
  static_assert(vec::size(Tag{}) == KTile);
  const nint_t active_k = KGuaranteed
      ? KTile
      : vecops::min(logical_k - k_base, KTile);
  for (nint_t row = 0; row < 16; ++row) {
    const nint_t logical_row = spatial_base + row;
    const auto value = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (SpatialGuaranteed) {
        if constexpr (KGuaranteed) {
          return source.load(
              Tag{}, tensor::coord(logical_row, k_base), tensor::axis<1>);
        } else {
          return source.load(
              Tag{}, tensor::coord(logical_row, k_base), tensor::axis<1>,
              vec::opt::first(active_k), vec::opt::zero);
        }
      } else {
        if (logical_row >= logical_spatial) return vec::zeros(Tag{});
        if constexpr (KGuaranteed) {
          return source.load(
              Tag{}, tensor::coord(logical_row, k_base), tensor::axis<1>);
        } else {
          return source.load(
              Tag{}, tensor::coord(logical_row, k_base), tensor::axis<1>,
              vec::opt::first(active_k), vec::opt::zero);
        }
      }
    }();
    vec::store(Tag{}, destination + row * KTile, value);
  }
}

template <nint_t KTile, vec::VectorTag Tag,
          typename Source, typename T,
          meta::ValueType Spatial, meta::ValueType K>
VECOPS_NOINLINE void pack_a_access(
    const Source& source, T* destination,
    Spatial spatial, K k) {
  constexpr bool SpatialGuaranteed =
      std::remove_cvref_t<Spatial>::aligns(16);
  constexpr bool KGuaranteed =
      std::remove_cvref_t<K>::aligns(KTile);
  const nint_t spatial_value = static_cast<nint_t>(spatial);
  const nint_t k_value = static_cast<nint_t>(k);
  const nint_t panels = ceil_div(spatial_value, nint_t{16});
  const nint_t k_tiles = ceil_div(k_value, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      pack_a_tile<KTile, Tag, SpatialGuaranteed, KGuaranteed>(
          source, destination, sp * 16, kt * KTile,
          spatial_value, k_value);
      destination += 16 * KTile;
    }
  }
}

template <nint_t KPack, nint_t KTile, vec::VectorTag Tag,
          bool SpatialGuaranteed, bool KGuaranteed = false,
          typename Source, typename T>
VECOPS_ALWAYS_INLINE void pack_b_tile(
    const Source& source, T* destination,
    nint_t spatial_base, nint_t k_base,
    nint_t logical_spatial, nint_t logical_k) {
  static_assert(std::same_as<vec::ElementOf<Tag>, T>);
  static_assert(KPack == 2 || KPack == 4);
  static_assert(vec::size(Tag{}) == 16);
  for (nint_t kg = 0; kg < KTile / KPack; ++kg) {
    auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
      const nint_t kk = k_base + kg * KPack + ki;
      const nint_t active_spatial = vecops::min(
          logical_spatial - spatial_base, nint_t{16});
      if constexpr (SpatialGuaranteed) {
        if constexpr (KGuaranteed) {
          return source.load(
              Tag{}, tensor::coord(spatial_base, kk), tensor::axis<0>);
        } else {
          return kk < logical_k
              ? source.load(
                    Tag{}, tensor::coord(spatial_base, kk), tensor::axis<0>)
              : vec::zeros(Tag{});
        }
      } else {
        if (active_spatial == 0 || (!KGuaranteed && kk >= logical_k))
          return vec::zeros(Tag{});
        return source.load(
            Tag{}, tensor::coord(spatial_base, kk), tensor::axis<0>,
            vec::opt::first(active_spatial), vec::opt::zero);
      }
    };
    if constexpr (KPack == 2) {
      using PackedTag = vec::Twice<Tag>;
      const auto packed = generic::interleave_pair<Tag>(load_k(0), load_k(1));
      vec::store(PackedTag{}, destination + kg * 16 * KPack, packed);
    } else {
      using PackedTag = vec::Twice<vec::Twice<Tag>>;
      const auto packed = generic::interleave_quad<Tag>(
          load_k(0), load_k(1), load_k(2), load_k(3));
      vec::store(PackedTag{}, destination + kg * 16 * KPack, packed);
    }
  }
}

template <nint_t KPack, nint_t KTile, vec::VectorTag Tag,
          typename Source, typename T,
          meta::ValueType Spatial, meta::ValueType K>
VECOPS_NOINLINE void pack_b_access(
    const Source& source, T* destination,
    Spatial spatial, K k) {
  constexpr bool SpatialGuaranteed =
      std::remove_cvref_t<Spatial>::aligns(16);
  constexpr bool KGuaranteed =
      std::remove_cvref_t<K>::aligns(KTile);
  const nint_t spatial_value = static_cast<nint_t>(spatial);
  const nint_t k_value = static_cast<nint_t>(k);
  const nint_t panels = ceil_div(spatial_value, nint_t{16});
  const nint_t k_tiles = ceil_div(k_value, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      pack_b_tile<KPack, KTile, Tag, SpatialGuaranteed, KGuaranteed>(
          source, destination, sp * 16, kt * KTile,
          spatial_value, k_value);
      destination += 16 * KTile;
    }
  }
}

/** In-place transpose of sixteen rows containing sixteen 32-bit words. */
VECOPS_ALWAYS_INLINE void transpose_16x16_dwords(__m512i (&rows)[16]) {
  // This is the standard AVX-512 four-stage transpose network: interleave
  // 32-bit words, then 64-bit pairs, then the two levels of 128-bit lanes.
  // The same network is used by public CPU kernels such as vLLM's
  // transpose_16x16_32bit implementation.
  __m512i tmp[16];

  tmp[0] = _mm512_unpacklo_epi32(rows[0], rows[1]);
  tmp[1] = _mm512_unpackhi_epi32(rows[0], rows[1]);
  tmp[2] = _mm512_unpacklo_epi32(rows[2], rows[3]);
  tmp[3] = _mm512_unpackhi_epi32(rows[2], rows[3]);
  tmp[4] = _mm512_unpacklo_epi32(rows[4], rows[5]);
  tmp[5] = _mm512_unpackhi_epi32(rows[4], rows[5]);
  tmp[6] = _mm512_unpacklo_epi32(rows[6], rows[7]);
  tmp[7] = _mm512_unpackhi_epi32(rows[6], rows[7]);
  tmp[8] = _mm512_unpacklo_epi32(rows[8], rows[9]);
  tmp[9] = _mm512_unpackhi_epi32(rows[8], rows[9]);
  tmp[10] = _mm512_unpacklo_epi32(rows[10], rows[11]);
  tmp[11] = _mm512_unpackhi_epi32(rows[10], rows[11]);
  tmp[12] = _mm512_unpacklo_epi32(rows[12], rows[13]);
  tmp[13] = _mm512_unpackhi_epi32(rows[12], rows[13]);
  tmp[14] = _mm512_unpacklo_epi32(rows[14], rows[15]);
  tmp[15] = _mm512_unpackhi_epi32(rows[14], rows[15]);

  rows[0] = _mm512_unpacklo_epi64(tmp[0], tmp[2]);
  rows[1] = _mm512_unpackhi_epi64(tmp[0], tmp[2]);
  rows[2] = _mm512_unpacklo_epi64(tmp[1], tmp[3]);
  rows[3] = _mm512_unpackhi_epi64(tmp[1], tmp[3]);
  rows[4] = _mm512_unpacklo_epi64(tmp[4], tmp[6]);
  rows[5] = _mm512_unpackhi_epi64(tmp[4], tmp[6]);
  rows[6] = _mm512_unpacklo_epi64(tmp[5], tmp[7]);
  rows[7] = _mm512_unpackhi_epi64(tmp[5], tmp[7]);
  rows[8] = _mm512_unpacklo_epi64(tmp[8], tmp[10]);
  rows[9] = _mm512_unpackhi_epi64(tmp[8], tmp[10]);
  rows[10] = _mm512_unpacklo_epi64(tmp[9], tmp[11]);
  rows[11] = _mm512_unpackhi_epi64(tmp[9], tmp[11]);
  rows[12] = _mm512_unpacklo_epi64(tmp[12], tmp[14]);
  rows[13] = _mm512_unpackhi_epi64(tmp[12], tmp[14]);
  rows[14] = _mm512_unpacklo_epi64(tmp[13], tmp[15]);
  rows[15] = _mm512_unpackhi_epi64(tmp[13], tmp[15]);

  tmp[0] = _mm512_shuffle_i32x4(rows[0], rows[4], 0x88);
  tmp[1] = _mm512_shuffle_i32x4(rows[1], rows[5], 0x88);
  tmp[2] = _mm512_shuffle_i32x4(rows[2], rows[6], 0x88);
  tmp[3] = _mm512_shuffle_i32x4(rows[3], rows[7], 0x88);
  tmp[4] = _mm512_shuffle_i32x4(rows[0], rows[4], 0xdd);
  tmp[5] = _mm512_shuffle_i32x4(rows[1], rows[5], 0xdd);
  tmp[6] = _mm512_shuffle_i32x4(rows[2], rows[6], 0xdd);
  tmp[7] = _mm512_shuffle_i32x4(rows[3], rows[7], 0xdd);
  tmp[8] = _mm512_shuffle_i32x4(rows[8], rows[12], 0x88);
  tmp[9] = _mm512_shuffle_i32x4(rows[9], rows[13], 0x88);
  tmp[10] = _mm512_shuffle_i32x4(rows[10], rows[14], 0x88);
  tmp[11] = _mm512_shuffle_i32x4(rows[11], rows[15], 0x88);
  tmp[12] = _mm512_shuffle_i32x4(rows[8], rows[12], 0xdd);
  tmp[13] = _mm512_shuffle_i32x4(rows[9], rows[13], 0xdd);
  tmp[14] = _mm512_shuffle_i32x4(rows[10], rows[14], 0xdd);
  tmp[15] = _mm512_shuffle_i32x4(rows[11], rows[15], 0xdd);

  rows[0] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0x88);
  rows[1] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0x88);
  rows[2] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0x88);
  rows[3] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0x88);
  rows[4] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0x88);
  rows[5] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0x88);
  rows[6] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0x88);
  rows[7] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0x88);
  rows[8] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0xdd);
  rows[9] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0xdd);
  rows[10] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0xdd);
  rows[11] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0xdd);
  rows[12] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0xdd);
  rows[13] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0xdd);
  rows[14] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0xdd);
  rows[15] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0xdd);
}

VECOPS_ALWAYS_INLINE __m512i accumulate_s8_column_sums(
    __m512i sums, const __m512i (&packed_groups)[16]) {
  const auto ones = _mm512_set1_epi8(1);
  VECOPS_UNROLL
  for (nint_t group = 0; group < 16; ++group) {
    // Each dword lane contains four adjacent K values for one logical B row.
    // VPDPBUSD therefore reduces four signed bytes into the corresponding
    // int32 column accumulator while the vectors are already resident for the
    // packed store.
    sums = _mm512_dpbusd_epi32(sums, ones, packed_groups[group]);
  }
  return sums;
}

VECOPS_ALWAYS_INLINE void store_s8_column_compensation(
    __m512i sums, int32_t* destination,
    nint_t active_rows, int32_t a_zero_point) {
  VECOPS_ASSERT(active_rows >= 0 && active_rows <= 16,
                "AMX B compensation tail must be in [0, 16]");
  const auto factor = _mm512_sub_epi32(
      _mm512_setzero_si512(), _mm512_set1_epi32(a_zero_point));
  const auto correction = _mm512_mullo_epi32(sums, factor);
  const auto mask = active_rows == 16
      ? static_cast<__mmask16>(0xffffu)
      : static_cast<__mmask16>(
            (uint32_t{1} << static_cast<unsigned>(active_rows)) - 1u);
  _mm512_mask_storeu_epi32(destination, mask, correction);
}

struct NoColumnCompensation {
  static constexpr bool enabled = false;

  VECOPS_ALWAYS_INLINE void begin_panel(nint_t) {}
  VECOPS_ALWAYS_INLINE void observe(nint_t, __m512i) {}
  VECOPS_ALWAYS_INLINE void end_panel(nint_t, nint_t) {}
};

class S8ColumnCompensation {
public:
  static constexpr bool enabled = true;

  VECOPS_ALWAYS_INLINE S8ColumnCompensation(
      int32_t* destination, int32_t a_zero_point)
      : destination_(destination), a_zero_point_(a_zero_point),
        ones_(_mm512_set1_epi8(1)) {}

  VECOPS_ALWAYS_INLINE void begin_panel(nint_t) {
    sums_even_ = _mm512_setzero_si512();
    sums_odd_ = _mm512_setzero_si512();
  }

  VECOPS_ALWAYS_INLINE void observe(
      nint_t group, __m512i packed_group) {
    if (group & 1) {
      sums_odd_ = _mm512_dpbusd_epi32(
          sums_odd_, ones_, packed_group);
    } else {
      sums_even_ = _mm512_dpbusd_epi32(
          sums_even_, ones_, packed_group);
    }
  }

  VECOPS_ALWAYS_INLINE void end_panel(
      nint_t row_base, nint_t active_rows) {
    store_s8_column_compensation(
        _mm512_add_epi32(sums_even_, sums_odd_),
        destination_ + row_base, active_rows, a_zero_point_);
  }

private:
  int32_t* destination_;
  int32_t a_zero_point_;
  __m512i sums_even_;
  __m512i sums_odd_;
  __m512i ones_;
};

template <typename T, typename Observer>
VECOPS_ALWAYS_INLINE void store_packed_b_groups(
    const __m512i (&rows)[16], T* destination, Observer& observer) {
  if constexpr (Observer::enabled) static_assert(std::same_as<T, int8_t>);
  auto* output = reinterpret_cast<__m512i*>(destination);
  VECOPS_UNROLL
  for (nint_t group = 0; group < 16; ++group) {
    if constexpr (Observer::enabled) observer.observe(group, rows[group]);
    _mm512_storeu_si512(output + group, rows[group]);
  }
}

/**
 * Pack a logically row-major B access whose DataAccess transform prevents the
 * raw-pointer fast path.  Loading along K keeps conversion/transform work on
 * contiguous vectors; the existing dword transpose then produces exactly the
 * same AMX VNNI panel as pack_b_direct().
 */
template <nint_t KPack, nint_t KTile,
          typename Source, typename T, typename Observer,
          meta::ValueType Spatial, meta::ValueType K>
VECOPS_ALWAYS_INLINE void pack_b_row_major_access_impl(
    const Source& source, T* destination,
    Spatial spatial, K k, Observer& observer) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  static_assert(KTile * static_cast<nint_t>(sizeof(T)) == 64);
  using RowTag = vec::ScalableTag<T, 0>;
  static_assert(vec::size(RowTag{}) == KTile);

  alignas(64) T row_major[16 * KTile];
  constexpr bool SpatialGuaranteed =
      std::remove_cvref_t<Spatial>::aligns(16);
  constexpr bool KGuaranteed =
      std::remove_cvref_t<K>::aligns(KTile);
  const nint_t spatial_value = static_cast<nint_t>(spatial);
  const nint_t k_value = static_cast<nint_t>(k);
  const nint_t panels = ceil_div(spatial_value, nint_t{16});
  const nint_t k_tiles = ceil_div(k_value, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t row_base = sp * 16;
    observer.begin_panel(row_base);
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      pack_a_tile<KTile, RowTag, SpatialGuaranteed, KGuaranteed>(
          source, row_major, row_base, kt * KTile,
          spatial_value, k_value);

      __m512i rows[16];
      VECOPS_UNROLL
      for (nint_t row = 0; row < 16; ++row) {
        rows[row] = _mm512_load_si512(row_major + row * KTile);
      }
      transpose_16x16_dwords(rows);
      store_packed_b_groups(rows, destination, observer);
      destination += 16 * KTile;
    }
    observer.end_panel(
        row_base, vecops::min(nint_t{16}, spatial_value - row_base));
  }
}

template <nint_t KPack, nint_t KTile, typename Source, typename T,
          meta::ValueType Spatial, meta::ValueType K>
VECOPS_NOINLINE void pack_b_row_major_access(
    const Source& source, T* destination, Spatial spatial, K k) {
  NoColumnCompensation observer;
  pack_b_row_major_access_impl<KPack, KTile>(
      source, destination, spatial, k, observer);
}

/** Row-major signed-byte B pack with a fused asymmetric-quantization sidecar. */
template <nint_t KPack, nint_t KTile, typename Source,
          meta::ValueType Spatial, meta::ValueType K>
VECOPS_NOINLINE void pack_b_row_major_access_compensated(
    const Source& source, int8_t* destination,
    int32_t* compensation, Spatial spatial, K k,
    int32_t a_zero_point) {
  static_assert(KPack == 4);
  static_assert(KTile == 64);
  S8ColumnCompensation observer{compensation, a_zero_point};
  pack_b_row_major_access_impl<KPack, KTile>(
      source, destination, spatial, k, observer);
}

template <typename T>
VECOPS_ALWAYS_INLINE __m512i load_b_row_direct(
    const T* source, nint_t active_k) {
  constexpr nint_t KTile = 64 / sizeof(T);
  static_assert(sizeof(T) == 1 || sizeof(T) == 2);
  if (active_k == KTile) return _mm512_loadu_si512(source);
  if constexpr (sizeof(T) == 1) {
    const auto mask = static_cast<__mmask64>(
        (uint64_t{1} << active_k) - uint64_t{1});
    return _mm512_maskz_loadu_epi8(mask, source);
  } else {
    const auto mask = static_cast<__mmask32>(
        (uint32_t{1} << active_k) - uint32_t{1});
    return _mm512_maskz_loadu_epi16(mask, source);
  }
}

template <typename T>
VECOPS_NOINLINE void pack_a_partial_tile_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k) {
  constexpr nint_t KTile = 64 / sizeof(T);
  VECOPS_ASSERT(active_rows >= 0 && active_rows <= 16,
                "AMX A tile row tail must be in [0, 16]");
  VECOPS_ASSERT(active_k >= 0 && active_k <= KTile,
                "AMX A tile K tail is out of range");
  const auto zero = _mm512_setzero_si512();
  for (nint_t row = 0; row < 16; ++row) {
    const auto value = row < active_rows
        ? load_b_row_direct(source + row * row_stride, active_k)
        : zero;
    _mm512_storeu_si512(destination + row * KTile, value);
  }
}

template <nint_t KPack, typename T, typename Observer>
VECOPS_ALWAYS_INLINE void pack_b_full_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination,
    Observer& observer) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  __m512i rows[16];
  VECOPS_UNROLL
  for (nint_t row = 0; row < 16; ++row) {
    rows[row] = _mm512_loadu_si512(source + row * row_stride);
  }
  transpose_16x16_dwords(rows);
  store_packed_b_groups(rows, destination, observer);
}

template <nint_t KPack, typename T, typename Observer>
VECOPS_ALWAYS_INLINE void pack_b_partial_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k, Observer& observer) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  __m512i rows[16];
  VECOPS_UNROLL
  for (nint_t row = 0; row < 16; ++row) {
    rows[row] = row < active_rows
        ? load_b_row_direct(source + row * row_stride, active_k)
        : _mm512_setzero_si512();
  }
  transpose_16x16_dwords(rows);
  store_packed_b_groups(rows, destination, observer);
}

template <nint_t KPack, typename T>
VECOPS_ALWAYS_INLINE void pack_b_full_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination) {
  NoColumnCompensation observer;
  pack_b_full_panel_direct_impl<KPack>(
      source, row_stride, destination, observer);
}

template <nint_t KPack, typename T>
VECOPS_ALWAYS_INLINE void pack_b_partial_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k) {
  NoColumnCompensation observer;
  pack_b_partial_panel_direct_impl<KPack>(
      source, row_stride, destination, active_rows, active_k, observer);
}

// Matmul calls these wrappers from several generated micro-kernel families.
// Keeping one copy per dtype avoids cloning the complete transpose network into
// every caller. The offline pack loop uses the inline implementations directly.
template <nint_t KPack, typename T>
VECOPS_NOINLINE void pack_b_full_panel_direct(
    const T* source, nint_t row_stride, T* destination) {
  pack_b_full_panel_direct_impl<KPack>(
      source, row_stride, destination);
}

template <nint_t KPack, typename T>
VECOPS_NOINLINE void pack_b_partial_panel_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k) {
  pack_b_partial_panel_direct_impl<KPack>(
      source, row_stride, destination, active_rows, active_k);
}

template <bool SpatialGuaranteed, bool KGuaranteed, typename T>
VECOPS_NOINLINE void pack_a_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k) {
  constexpr nint_t Panel = 16;
  constexpr nint_t KTile = 64 / sizeof(T);
  using TileTag = vec::FixedTag<T, KTile>;
  static_assert(64 % sizeof(T) == 0);
  const nint_t panels = ceil_div(spatial, Panel);
  const nint_t k_tiles = ceil_div(k, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      const nint_t kk = kt * KTile;
      const nint_t active_k = KGuaranteed
          ? KTile
          : vecops::min(KTile, k - kk);
      for (nint_t lane = 0; lane < Panel; ++lane) {
        const nint_t row = sp * Panel + lane;
        if constexpr (SpatialGuaranteed && KGuaranteed) {
          vec::store(
              TileTag{}, destination,
              vec::load(TileTag{}, source + row * row_stride + kk));
        } else if constexpr (SpatialGuaranteed) {
          vec::store(
              TileTag{}, destination,
              vec::load(
                  TileTag{}, source + row * row_stride + kk,
                  vec::opt::first(active_k), vec::opt::zero));
        } else if constexpr (KGuaranteed) {
          const auto value = row < spatial
              ? vec::load(TileTag{}, source + row * row_stride + kk)
              : vec::zeros(TileTag{});
          vec::store(TileTag{}, destination, value);
        } else {
          // Keep the fully dynamic path in the original conditional-load
          // form. GCC otherwise turns an immediately invoked inline lambda
          // into a substantially slower control-flow shape for A-pack.
          const auto value = row < spatial
              ? vec::load(
                    TileTag{}, source + row * row_stride + kk,
                    vec::opt::first(active_k), vec::opt::zero)
              : vec::zeros(TileTag{});
          vec::store(TileTag{}, destination, value);
        }
        destination += KTile;
      }
    }
  }
}

// Keep the fully dynamic ABI separate from the Meta-specialized variants.
// Besides avoiding template fan-out, the explicit panel and K-tile arguments
// let GCC form the same profitable const-propagated clones as the original
// dynamic pack loop used by benchmark-defined shapes.
template <typename T>
VECOPS_NOINLINE void pack_a_direct_dynamic(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k, nint_t panel, nint_t k_tile) {
  using TileTag = vec::FixedTag<T, 64 / sizeof(T)>;
  static_assert(64 % sizeof(T) == 0);
  VECOPS_ASSERT(panel == 16, "AMX A-pack panel must contain 16 rows");
  VECOPS_ASSERT(64 / sizeof(T) == k_tile,
                "AMX A-pack k tile must occupy 64 bytes");
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t k_tiles = ceil_div(k, k_tile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      const nint_t kk = kt * k_tile;
      const nint_t active_k = vecops::min(k_tile, k - kk);
      for (nint_t lane = 0; lane < panel; ++lane) {
        const nint_t row = sp * panel + lane;
        const auto value = row < spatial
            ? vec::load(
                  TileTag{}, source + row * row_stride + kk,
                  vec::opt::first(active_k), vec::opt::zero)
            : vec::zeros(TileTag{});
        vec::store(TileTag{}, destination, value);
        destination += k_tile;
      }
    }
  }
}

template <nint_t KPack, bool SpatialGuaranteed, bool KGuaranteed,
          typename T, typename Observer>
VECOPS_ALWAYS_INLINE void pack_b_direct_impl(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k,
    Observer& observer) {
  constexpr nint_t Panel = 16;
  constexpr nint_t KTile = 64 / sizeof(T);
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  if constexpr (Observer::enabled) {
    static_assert(KPack == 4);
    static_assert(std::same_as<T, int8_t>);
  }
  const nint_t panels = ceil_div(spatial, Panel);
  const nint_t k_tiles = ceil_div(k, KTile);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t row_base = sp * Panel;
    const T* panel_source = k_tiles == 0
        ? source
        : source + row_base * row_stride;
    const nint_t active_rows = SpatialGuaranteed
        ? Panel
        : vecops::min_reference(Panel, spatial - row_base);
    observer.begin_panel(row_base);
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      const nint_t kk = kt * KTile;
      const nint_t active_k = KGuaranteed
          ? KTile
          : vecops::min_reference(KTile, k - kk);
      // Keep full and partial networks separate.  Trying to join after only
      // splitting the loads makes GCC spill the 16 live ZMM rows; moving the
      // observer across a noinline helper boundary spills its accumulator.
      // Both alternatives save text but cost 22% or more on multi-tile S8.
      if constexpr (SpatialGuaranteed && KGuaranteed) {
        pack_b_full_panel_direct_impl<KPack>(
            panel_source + kk, row_stride, destination, observer);
      } else {
        if (active_rows == Panel && active_k == KTile) {
          pack_b_full_panel_direct_impl<KPack>(
              panel_source + kk, row_stride, destination, observer);
        } else {
          pack_b_partial_panel_direct_impl<KPack>(
              panel_source + kk, row_stride, destination,
              active_rows, active_k, observer);
        }
      }
      destination += Panel * KTile;
    }
    observer.end_panel(row_base, active_rows);
  }
}

template <nint_t KPack, bool SpatialGuaranteed, bool KGuaranteed,
          typename T>
VECOPS_NOINLINE void pack_b_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k) {
  NoColumnCompensation observer;
  pack_b_direct_impl<KPack, SpatialGuaranteed, KGuaranteed>(
      source, row_stride, destination, spatial, k, observer);
}

template <nint_t KPack, bool SpatialGuaranteed, bool KGuaranteed>
VECOPS_NOINLINE void pack_b_direct_compensated(
    const int8_t* source, nint_t row_stride, int8_t* destination,
    int32_t* compensation, nint_t spatial, nint_t k,
    int32_t a_zero_point) {
  S8ColumnCompensation observer{compensation, a_zero_point};
  pack_b_direct_impl<KPack, SpatialGuaranteed, KGuaranteed>(
      source, row_stride, destination, spatial, k, observer);
}

VECOPS_NOINLINE inline void compensate_packed_s8_b(
    const int8_t* source, int32_t* compensation,
    nint_t spatial, nint_t k,
    int32_t a_zero_point) {
  constexpr nint_t Panel = 16;
  constexpr nint_t KTile = 64;
  const nint_t panels = ceil_div(spatial, Panel);
  const nint_t k_tiles = ceil_div(k, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    __m512i sums = _mm512_setzero_si512();
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      __m512i groups[16];
      VECOPS_UNROLL
      for (nint_t group = 0; group < 16; ++group) {
        groups[group] = _mm512_loadu_si512(source);
        source += 64;
      }
      sums = accumulate_s8_column_sums(sums, groups);
    }
    const nint_t row_base = sp * Panel;
    store_s8_column_compensation(
        sums, compensation + row_base,
        vecops::min(Panel, spatial - row_base), a_zero_point);
  }
}

} // namespace vecops::kernel::matmul_pack_details::amx

#endif // VECOPS_MATMUL_DETAILS_PACK_AMX_PACK_H
