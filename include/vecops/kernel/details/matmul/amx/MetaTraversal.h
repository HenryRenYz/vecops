//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_AMX_META_TRAVERSAL_H
#define VECOPS_KERNEL_DETAILS_MATMUL_AMX_META_TRAVERSAL_H

#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/Meta.h"
#include "vecops/kernel/Tile2D.h"

namespace vecops::kernel::matmul_details::amx {

namespace meta_traversal_details {

namespace tile = kernel::loop;

template <typename T>
inline constexpr bool fixed_nonnegative_v =
    tile::tile2d_details::fixed_value_v<T> &&
    tile::tile2d_details::fixed_value_n<T> >= 0;

template <typename T>
inline constexpr bool not_fixed_negative_v =
    !tile::tile2d_details::fixed_value_v<T> || fixed_nonnegative_v<T>;

template <typename M, typename N>
inline constexpr bool candidate_v =
#if defined(VECOPS_DISABLE_AMX_META_MAX3)
    false;
#else
    (fixed_nonnegative_v<M> && fixed_nonnegative_v<N>) ||
    (not_fixed_negative_v<M> && not_fixed_negative_v<N> &&
     std::remove_cvref_t<M>::aligns(32) &&
     std::remove_cvref_t<N>::aligns(32));
#endif

template <typename Catalog, typename M, typename N, typename Fn>
VECOPS_ALWAYS_INLINE void run_fixed(M m, N n, Fn& fn) {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  static_assert(tile::tile2d_details::fixed_value_v<MV>);
  static_assert(tile::tile2d_details::fixed_value_v<NV>);
  constexpr nint_t MExtent = tile::tile2d_details::fixed_value_n<MV>;
  constexpr nint_t NExtent = tile::tile2d_details::fixed_value_n<NV>;
  static_assert(MExtent >= 0 && NExtent >= 0);
  if constexpr (MExtent == 0 || NExtent == 0) return;

  const nint_t m_extent = static_cast<nint_t>(m);
  const nint_t n_extent = static_cast<nint_t>(n);
  constexpr nint_t MBlocks = ceil_div(MExtent, nint_t{16});
  constexpr nint_t NBlocks = ceil_div(NExtent, nint_t{16});

  if constexpr (MExtent % 32 == 0 && NExtent % 32 == 0) {
    using Family =
        tile::tile2d_details::family_for_t<Catalog, 2, 2>;
    VECOPS_NOUNROLL
    for (nint_t mi = 0; mi < m_extent; mi += 32) {
      VECOPS_NOUNROLL
      for (nint_t ni = 0; ni < n_extent; ni += 32) {
        tile::tile2d_details::invoke<
            Family,
            tile::Tile2DMaskMode::unmasked,
            tile::Tile2DMaskMode::unmasked>(
                fn, mi, ni, nint_t{32}, nint_t{32});
      }
    }
    return;
  }

  if constexpr (NBlocks == 1) {
    nint_t bm = 0;
    VECOPS_NOUNROLL
    for (; bm + 3 <= MBlocks; bm += 3) {
      tile::tile2d_details::invoke_runtime_exact_area4<3, 1, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, 0);
    }
    if constexpr (MBlocks % 3 == 2) {
      tile::tile2d_details::invoke_runtime_exact_area4<2, 1, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, 0);
    } else if constexpr (MBlocks % 3 == 1) {
      tile::tile2d_details::invoke_runtime_exact_area4<1, 1, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, 0);
    }
    return;
  }

  nint_t bm = 0;
  VECOPS_NOUNROLL
  for (; bm + 2 <= MBlocks; bm += 2) {
    nint_t bn = 0;
    VECOPS_NOUNROLL
    for (; bn + 2 <= NBlocks; bn += 2) {
      tile::tile2d_details::invoke_runtime_exact_area4<2, 2, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, bn);
    }
    if constexpr (NBlocks % 2 == 1) {
      tile::tile2d_details::invoke_runtime_exact_area4<2, 1, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, bn);
    }
  }
  if constexpr (MBlocks % 2 == 1) {
    nint_t bn = 0;
    VECOPS_NOUNROLL
    for (; bn + 3 <= NBlocks; bn += 3) {
      tile::tile2d_details::invoke_runtime_exact_area4<1, 3, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, bn);
    }
    if constexpr (NBlocks % 3 == 2) {
      tile::tile2d_details::invoke_runtime_exact_area4<1, 2, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, bn);
    } else if constexpr (NBlocks % 3 == 1) {
      tile::tile2d_details::invoke_runtime_exact_area4<1, 1, Catalog>(
          fn, m_extent, n_extent, 16, 16, bm, bn);
    }
  }
}

template <typename Catalog, typename M, typename N, typename Fn>
VECOPS_ALWAYS_INLINE void run_aligned(M m, N n, Fn& fn) {
  using Family = tile::tile2d_details::family_for_t<Catalog, 2, 2>;
  const nint_t m_extent = static_cast<nint_t>(m);
  const nint_t n_extent = static_cast<nint_t>(n);
  VECOPS_NOUNROLL
  for (nint_t mi = 0; mi < m_extent; mi += 32) {
    VECOPS_NOUNROLL
    for (nint_t ni = 0; ni < n_extent; ni += 32) {
      tile::tile2d_details::invoke<
          Family,
          tile::Tile2DMaskMode::unmasked,
          tile::Tile2DMaskMode::unmasked>(
              fn, mi, ni, nint_t{32}, nint_t{32});
    }
  }
}

} // namespace meta_traversal_details

template <typename M, typename N>
inline constexpr bool meta_max3_candidate_v =
    meta_traversal_details::candidate_v<M, N>;

template <typename Catalog, typename M, typename N, typename Fn>
VECOPS_ALWAYS_INLINE void run_meta_max3(M m, N n, Fn& fn) {
  static_assert(meta_max3_candidate_v<M, N>);
  if constexpr (
      kernel::loop::tile2d_details::fixed_value_v<M> &&
      kernel::loop::tile2d_details::fixed_value_v<N>) {
    meta_traversal_details::run_fixed<Catalog>(m, n, fn);
  } else {
    static_assert(std::remove_cvref_t<M>::aligns(32));
    static_assert(std::remove_cvref_t<N>::aligns(32));
    meta_traversal_details::run_aligned<Catalog>(m, n, fn);
  }
}

} // namespace vecops::kernel::matmul_details::amx

#endif // VECOPS_KERNEL_DETAILS_MATMUL_AMX_META_TRAVERSAL_H
