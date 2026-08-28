//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H

#include <algorithm>
#include <cstdint>

#include <immintrin.h>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
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
      : std::clamp(logical_k - k_base, nint_t{0}, KTile);
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
          typename Source, typename T>
VECOPS_NOINLINE void pack_a_access(
    const Source& source, T* destination,
    nint_t spatial, nint_t k) {
  const nint_t panels = ceil_div(spatial, nint_t{16});
  const nint_t k_tiles = ceil_div(k, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      pack_a_tile<KTile, Tag, false>(
          source, destination, sp * 16, kt * KTile, spatial, k);
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
      const nint_t active_spatial = std::clamp(
          logical_spatial - spatial_base, nint_t{0}, nint_t{16});
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
          typename Source, typename T>
VECOPS_NOINLINE void pack_b_access(
    const Source& source, T* destination,
    nint_t spatial, nint_t k) {
  const nint_t panels = ceil_div(spatial, nint_t{16});
  const nint_t k_tiles = ceil_div(k, KTile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      pack_b_tile<KPack, KTile, Tag, false>(
          source, destination, sp * 16, kt * KTile, spatial, k);
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

template <nint_t KPack, typename T>
VECOPS_ALWAYS_INLINE void pack_b_full_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination, nint_t k_tile) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  VECOPS_ASSERT(k_tile * static_cast<nint_t>(sizeof(T)) == 64,
                "AMX B-pack k tile must occupy 64 bytes");
  __m512i rows[16];
  VECOPS_UNROLL
  for (nint_t row = 0; row < 16; ++row) {
    rows[row] = _mm512_loadu_si512(source + row * row_stride);
  }
  transpose_16x16_dwords(rows);
  auto* output = reinterpret_cast<__m512i*>(destination);
  VECOPS_UNROLL
  for (nint_t group = 0; group < 16; ++group) {
    _mm512_storeu_si512(output + group, rows[group]);
  }
}

template <nint_t KPack, typename T>
VECOPS_ALWAYS_INLINE void pack_b_partial_panel_direct_impl(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k) {
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
  auto* output = reinterpret_cast<__m512i*>(destination);
  VECOPS_UNROLL
  for (nint_t group = 0; group < 16; ++group) {
    _mm512_storeu_si512(output + group, rows[group]);
  }
}

// Matmul calls these wrappers from several generated micro-kernel families.
// Keeping one copy per dtype avoids cloning the complete transpose network into
// every caller. The offline pack loop uses the inline implementations directly.
template <nint_t KPack, typename T>
VECOPS_NOINLINE void pack_b_full_panel_direct(
    const T* source, nint_t row_stride, T* destination, nint_t k_tile) {
  pack_b_full_panel_direct_impl<KPack>(
      source, row_stride, destination, k_tile);
}

template <nint_t KPack, typename T>
VECOPS_NOINLINE void pack_b_partial_panel_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t active_rows, nint_t active_k) {
  pack_b_partial_panel_direct_impl<KPack>(
      source, row_stride, destination, active_rows, active_k);
}

template <typename T>
VECOPS_NOINLINE void pack_a_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k, nint_t panel, nint_t k_tile) {
  using TileTag = vec::FixedTag<T, 64 / sizeof(T)>;
  static_assert(64 % sizeof(T) == 0);
  VECOPS_ASSERT(64 / sizeof(T) == k_tile,
                "AMX A-pack k tile must occupy 64 bytes");
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t k_tiles = ceil_div(k, k_tile);
  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      const nint_t kk = kt * k_tile;
      const nint_t active_k = std::min(k_tile, k - kk);
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

template <nint_t KPack, typename T>
VECOPS_NOINLINE void pack_b_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k, nint_t panel, nint_t k_tile) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  VECOPS_ASSERT(panel == 16, "AMX B-pack panel must contain 16 rows");
  VECOPS_ASSERT(k_tile * static_cast<nint_t>(sizeof(T)) == 64,
                "AMX B-pack k tile must occupy 64 bytes");
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t k_tiles = ceil_div(k, k_tile);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t row_base = sp * panel;
    const T* panel_source = source + row_base * row_stride;
    const nint_t active_rows = std::min(panel, spatial - row_base);
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      const nint_t kk = kt * k_tile;
      const nint_t active_k = std::min(k_tile, k - kk);
      if (active_rows == panel && active_k == k_tile) {
        pack_b_full_panel_direct_impl<KPack>(
            panel_source + kk, row_stride, destination, k_tile);
      } else {
        pack_b_partial_panel_direct_impl<KPack>(
            panel_source + kk, row_stride, destination,
            active_rows, active_k);
      }
      destination += panel * k_tile;
    }
  }
}

} // namespace vecops::kernel::matmul_pack_details::amx

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H
