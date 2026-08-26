//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H

#include <algorithm>
#include <array>
#include <cstdint>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_pack_details::amx {

template <typename T>
VECOPS_NOINLINE void pack_a_direct(
    const T* source, nint_t row_stride, T* destination,
    nint_t spatial, nint_t k, nint_t panel, nint_t k_tile) {
  using TileTag = vec::FixedTag<T, 64 / sizeof(T)>;
  static_assert(64 % sizeof(T) == 0);
  VECOPS_ASSERT(64 / sizeof(T) == k_tile);
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
  static_assert(KPack == 1 || KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  VECOPS_ASSERT(panel == 16);
  using WordTag = vec::FixedTag<uint32_t, 16>;
  using IndexTag = vec::FixedTag<int32_t, 16>;
  std::array<int32_t, 16> row_offsets{};
  VECOPS_UNROLL
  for (nint_t row = 0; row < panel; ++row) {
    row_offsets[static_cast<std::size_t>(row)] =
        static_cast<int32_t>(row * row_stride);
  }
  const auto indices = vec::load(IndexTag{}, row_offsets.data());
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t groups = ceil_div(k, KPack);
  const nint_t padded_groups =
      ceil_div(k, k_tile) * (k_tile / KPack);
  auto* out = reinterpret_cast<uint32_t*>(destination);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t row_base = sp * panel;
    const T* panel_source = source + row_base * row_stride;
    const nint_t active_rows = std::min(panel, spatial - row_base);
    for (nint_t group = 0; group < groups; ++group) {
      const nint_t kk = group * KPack;
      const nint_t active_k = std::min(KPack, k - kk);
      const nint_t load_k = active_k == KPack ? kk : k - KPack;
      auto packed = vec::load(
          WordTag{},
          reinterpret_cast<const uint32_t*>(panel_source + load_k),
          vec::indexed(indices, vec::scale<sizeof(T)>),
          vec::opt::first(active_rows), vec::opt::zero);
      if (active_k != KPack) {
        const int shift = static_cast<int>(
            (KPack - active_k) * sizeof(T) * 8);
        const uint32_t mask =
            (uint32_t{1} << (active_k * sizeof(T) * 8)) - 1;
        packed = vec::bit_and(
            vec::bit_shr(packed, shift), vec::fill(WordTag{}, mask));
      }
      vec::store(WordTag{}, out, packed);
      out += panel;
    }
    const auto zero = vec::zeros(WordTag{});
    for (nint_t group = groups; group < padded_groups; ++group) {
      vec::store(WordTag{}, out, zero);
      out += panel;
    }
  }
}

} // namespace vecops::kernel::matmul_pack_details::amx

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_PACK_H
