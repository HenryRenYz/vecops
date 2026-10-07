// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/**
 * @file vecops/matmul/details/packing/generic/Pack.h
 * @brief Backend-independent packing loops built on the portable vec layer.
 *
 * Two block emitters cover both block-format families without any ISA
 * intrinsics:
 *
 * - pack_blocked_rows emits the *A-style* layout ([spatial panel][k tile]
 *   [row][k], rows contiguous) by loading K runs (axis 1);
 * - pack_interleaved_panels emits the *B/SME-style* interleaved layout
 *   ([spatial panel][k group][row][k within KPack]) by loading spatial
 *   runs (axis 0) and interleaving KPack of them per group.
 *
 * They serve as the fallback packers for the AMX backend below AVX-512 and
 * as the SME format's base (Vector) implementation.  The interleave_pair /
 * interleave_quad helpers are the shared KPack primitives; the AMX
 * intrinsic packers reuse them for their byte paths.
 */

#ifndef VECOPS_MATMUL_DETAILS_PACK_GENERIC_PACK_H
#define VECOPS_MATMUL_DETAILS_PACK_GENERIC_PACK_H

#include <cstdint>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_pack_details::generic {

/// Element type an access computes in.
template <typename Access>
using ComputeOf = typename std::remove_cvref_t<Access>::ComputeType;

/// The InputSpec/OutputSpec an access was bound from.
template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

/**
 * @brief True when an access is a raw pointer over its own compute type.
 *
 * I.e. it exposes raw_data()/raw_strides(), carries no transform, and its
 * memory element equals its compute type — the precondition for the
 * intrinsic packers to reinterpret the raw pointer instead of going
 * through the generic load/store paths.
 */
template <typename Access>
inline constexpr bool is_raw_direct_access_v = requires(Access& access) {
  typename std::remove_cvref_t<Access>::MemoryElement;
  typename std::remove_cvref_t<Access>::Transform;
  { access.raw_data() };
  { access.raw_strides() };
} && std::same_as<
    typename std::remove_cvref_t<Access>::Transform, tensor::NoTransform> &&
    std::same_as<
        std::remove_cv_t<
            typename std::remove_cvref_t<Access>::MemoryElement>,
        ComputeOf<Access>>;

/// Concept form of is_raw_direct_access_v.
template <typename Access>
concept RawDirectAccess = is_raw_direct_access_v<Access>;

/**
 * @brief Emit the A-style blocked-row layout ([panel][k tile][row][k]).
 *
 * For each spatial panel and K tile, writes `panel` rows of `k_tile`
 * contiguous elements.  Loads vectorize along K (axis 1).  Tail rows
 * beyond `spatial` and tail K beyond `k` are zero-filled so every emitted
 * block is dense; the stored chunk shrinks to the vector granularity
 * (`first(chunk)`), never past k_tile.
 */
template <vec::VectorTag Tag, typename Source, typename T>
VECOPS_NOINLINE void pack_blocked_rows(
    const Source& source, T* destination,
    nint_t spatial, nint_t k, nint_t panel, nint_t k_tile) {
  static_assert(std::same_as<vec::ElementOf<Tag>, T>);
  const nint_t spatial_panels = ceil_div(spatial, panel);
  const nint_t k_tiles = ceil_div(k, k_tile);
  const nint_t vector_lanes = vec::size(Tag{});
  for (nint_t sp = 0; sp < spatial_panels; ++sp) {
    for (nint_t kt = 0; kt < k_tiles; ++kt) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        const nint_t row = sp * panel + lane;
        for (nint_t k_offset = 0; k_offset < k_tile;
             k_offset += vector_lanes) {
          const nint_t chunk = vecops::min(
              vector_lanes, k_tile - k_offset);
          const nint_t kk = kt * k_tile + k_offset;
          const nint_t active_k = vecops::clamp(
              k - kk, nint_t{0}, chunk);
          const auto value = row < spatial && active_k > 0
              ? source.load(
                    Tag{}, tensor::coord(row, kk), tensor::axis<1>,
                    vec::opt::first(active_k), vec::opt::zero)
              : vec::zeros(Tag{});
          vec::store(
              Tag{}, destination, value, vec::opt::first(chunk));
          destination += chunk;
        }
      }
    }
  }
}

template <typename Tag, typename V0, typename V1>
VECOPS_ALWAYS_INLINE auto interleave_pair(V0 v0, V1 v1) {
  using OutTag = vec::Twice<Tag>;
  return vec::interleave(OutTag{}, v0, v1);
}

template <typename Tag, typename V0, typename V1,
          typename V2, typename V3>
VECOPS_ALWAYS_INLINE auto interleave_quad(
    V0 v0, V1 v1, V2 v2, V3 v3) {
  using T = vec::ElementOf<Tag>;
  static_assert(sizeof(T) == 1);
  using PairTag = vec::Twice<Tag>;
  using PairViewTag = vec::ViewAs<uint16_t, PairTag>;
  using QuadViewTag = vec::Twice<PairViewTag>;
  using OutTag = vec::Twice<PairTag>;
  const auto pair01 = vec::interleave(PairTag{}, v0, v1);
  const auto pair23 = vec::interleave(PairTag{}, v2, v3);
  const auto pair01_view = vec::bitcast(
      PairViewTag{}, PairTag{}, pair01);
  const auto pair23_view = vec::bitcast(
      PairViewTag{}, PairTag{}, pair23);
  const auto quad_view = vec::interleave(
      QuadViewTag{}, pair01_view, pair23_view);
  return vec::bitcast(OutTag{}, QuadViewTag{}, quad_view);
}

template <nint_t KPack, vec::VectorTag Tag,
          typename Source, typename T>
VECOPS_NOINLINE void pack_interleaved_panels(
    const Source& source, T* destination,
    nint_t spatial, nint_t k, nint_t panel,
    nint_t padded_k_groups) {
  static_assert(std::same_as<vec::ElementOf<Tag>, T>);
  static_assert(KPack == 1 || KPack == 2 || KPack == 4);
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t vector_lanes = vec::size(Tag{});

  for (nint_t sp = 0; sp < panels; ++sp) {
    for (nint_t kg = 0; kg < padded_k_groups; ++kg) {
      for (nint_t lane_base = 0; lane_base < panel;
           lane_base += vector_lanes) {
        const nint_t panel_chunk = vecops::min(
            vector_lanes, panel - lane_base);
        const nint_t spatial_base = sp * panel + lane_base;
        const nint_t active = vecops::clamp(
            spatial - spatial_base, nint_t{0}, panel_chunk);
        auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
          const nint_t kk = kg * KPack + ki;
          return kk < k && active > 0
              ? source.load(
                    Tag{}, tensor::coord(spatial_base, kk), tensor::axis<0>,
                    vec::opt::first(active), vec::opt::zero)
              : vec::zeros(Tag{});
        };

        if constexpr (KPack == 1) {
          const auto packed = load_k(0);
          vec::store(
              Tag{}, destination, packed, vec::opt::first(panel_chunk));
        } else if constexpr (KPack == 2) {
          using OutTag = vec::Twice<Tag>;
          const auto packed = interleave_pair<Tag>(load_k(0), load_k(1));
          vec::store(
              OutTag{}, destination, packed,
              vec::opt::first(panel_chunk * KPack));
        } else {
          using OutTag = vec::Twice<vec::Twice<Tag>>;
          const auto packed = interleave_quad<Tag>(
              load_k(0), load_k(1), load_k(2), load_k(3));
          vec::store(
              OutTag{}, destination, packed,
              vec::opt::first(panel_chunk * KPack));
        }
        destination += panel_chunk * KPack;
      }
    }
  }
}

} // namespace vecops::kernel::matmul_pack_details::generic

#endif // VECOPS_MATMUL_DETAILS_PACK_GENERIC_PACK_H
