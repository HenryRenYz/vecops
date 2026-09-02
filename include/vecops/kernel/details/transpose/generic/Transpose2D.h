//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_GENERIC_TRANSPOSE2D_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_GENERIC_TRANSPOSE2D_H

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/Meta.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/transpose/Types.h"
#include "vecops/tensor/Transform.h"
#include "vecops/vec/Vec.h"

/**
 * @file Transpose2D.h
 * @brief Shared DataAccess-aware vector algorithms for platform transpose backends.
 *
 * This file contains no platform selection. Backends choose tile dimensions or
 * the gather path explicitly. The fixed path loads register rows, applies a
 * recursive interleave/concat butterfly, and stores output columns. Tail
 * regions use gather plus contiguous masked stores to reduce register pressure.
 */

namespace vecops::kernel::transpose2d_details::generic {

using namespace ::vecops::meta;
using ::vecops::kernel::loop::Tile2DKernelCatalog;
using ::vecops::kernel::loop::Tile2DKernelFamily;
using ::vecops::kernel::loop::Tile2DMaskMode;

template <typename Access>
/** Compute element type exposed by a bound DataAccess session. */
using ComputeOf = typename std::remove_cvref_t<Access>::ComputeType;

template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

template <typename Source, typename Destination>
/** Bound source/destination pair accepted by the generic transpose kernels. */
inline constexpr bool are_compatible_accesses_v = requires {
  typename ComputeOf<Source>;
  typename ComputeOf<Destination>;
} && std::same_as<ComputeOf<Source>, ComputeOf<Destination>> &&
    vec::Element<ComputeOf<Source>>;

template <typename Source, typename Destination>
concept CompatibleAccesses =
    are_compatible_accesses_v<Source, Destination>;

template <typename Access>
/** Access exposing untransformed raw storage, including converted accesses. */
inline constexpr bool is_raw_no_transform_access_v = requires(Access& access) {
  typename std::remove_cvref_t<Access>::MemoryElement;
  typename std::remove_cvref_t<Access>::Transform;
  { access.raw_data() };
  { access.raw_strides() };
  { access.spec() };
} && std::same_as<
    typename std::remove_cvref_t<Access>::Transform, tensor::NoTransform>;

template <typename Access>
/** Untransformed raw storage whose memory and compute dtypes are identical. */
inline constexpr bool is_raw_direct_access_v =
    is_raw_no_transform_access_v<Access> &&
    std::same_as<
        std::remove_cv_t<
            typename std::remove_cvref_t<Access>::MemoryElement>,
        ComputeOf<Access>>;

template <typename Access>
concept RawNoTransformAccess = is_raw_no_transform_access_v<Access>;

template <typename Access>
concept RawDirectAccess = is_raw_direct_access_v<Access>;

template <nint_t Upper, Tile2DMaskMode Mask>
VECOPS_ALWAYS_INLINE constexpr auto active_extent(nint_t value) {
  if constexpr (Mask == Tile2DMaskMode::unmasked) {
    return meta::cint<Upper>;
  } else {
    return meta::dyn<1, 1, Upper>(value);
  }
}

template <std::size_t Bytes>
using UIntOfSize = std::conditional_t<
    Bytes == 1, uint8_t,
    std::conditional_t<Bytes == 2, uint16_t,
    std::conditional_t<Bytes == 4, uint32_t, uint64_t>>>;

template <std::size_t Stage, vec::VectorTag Tag, std::size_t N>
/**
 * @brief Execute one or more butterfly stages within a 16-byte register tile.
 * @param input N row vectors representing a square 16-byte submatrix.
 * @return Row vectors after recursively doubling the logical lane width.
 */
VECOPS_ALWAYS_INLINE auto transpose_16byte_stage(
    const std::array<vec::Vec<Tag>, N>& input) {
  static_assert(vec::is_fixed_tag_v<Tag>);
  static_assert(
      static_cast<nint_t>(N * sizeof(vec::ElementOf<Tag>)) == 16);
  if constexpr ((std::size_t{1} << Stage) == N) {
    return input;
  } else {
    constexpr std::size_t Span = std::size_t{1} << Stage;
    using StageElement = UIntOfSize<sizeof(vec::ElementOf<Tag>) * Span>;
    using StageTag = vec::ViewAs<StageElement, Tag>;
    std::array<vec::Vec<Tag>, N> output{};
    VECOPS_UNROLL
    for (std::size_t block = 0; block < N; block += 2 * Span) {
      VECOPS_UNROLL
      for (std::size_t j = 0; j < Span; ++j) {
        const auto a = vec::bitcast(StageTag{}, Tag{}, input[block + j]);
        const auto b = vec::bitcast(
            StageTag{}, Tag{}, input[block + Span + j]);
        output[block + 2 * j] = vec::bitcast(
            Tag{}, StageTag{}, vec::local_interleave_lower(a, b));
        output[block + 2 * j + 1] = vec::bitcast(
            Tag{}, StageTag{}, vec::local_interleave_upper(a, b));
      }
    }
    return transpose_16byte_stage<Stage + 1, Tag>(output);
  }
}

template <vec::VectorTag Tag, std::size_t N>
/**
 * @brief Transpose an N-by-N square held in N fixed-width vectors.
 * @param rows Input matrix rows.
 * @return Vectors containing the corresponding output rows/columns.
 *
 * Larger vectors recursively split into halves; the 16-byte base case uses
 * local interleave operations. All dimensions are compile-time constants.
 */
VECOPS_ALWAYS_INLINE auto transpose_square(
    const std::array<vec::Vec<Tag>, N>& rows) {
  static_assert(vec::is_fixed_tag_v<Tag>);
  static_assert(static_cast<nint_t>(N) == vec::fixed_lanes_v<Tag>);
  if constexpr (N == 1) {
    return rows;
  } else if constexpr (N * sizeof(vec::ElementOf<Tag>) == 16) {
    return transpose_16byte_stage<0, Tag>(rows);
  } else {
    static_assert((N & (N - 1)) == 0);
    constexpr std::size_t HalfN = N / 2;
    using HalfTag = vec::Half<Tag>;
    std::array<vec::Vec<HalfTag>, HalfN> top_left{};
    std::array<vec::Vec<HalfTag>, HalfN> top_right{};
    std::array<vec::Vec<HalfTag>, HalfN> bottom_left{};
    std::array<vec::Vec<HalfTag>, HalfN> bottom_right{};
    VECOPS_UNROLL
    for (std::size_t i = 0; i < HalfN; ++i) {
      top_left[i] = vec::lower(Tag{}, rows[i]);
      top_right[i] = vec::upper(Tag{}, rows[i]);
      bottom_left[i] = vec::lower(Tag{}, rows[i + HalfN]);
      bottom_right[i] = vec::upper(Tag{}, rows[i + HalfN]);
    }
    const auto tl = transpose_square<HalfTag>(top_left);
    const auto tr = transpose_square<HalfTag>(top_right);
    const auto bl = transpose_square<HalfTag>(bottom_left);
    const auto br = transpose_square<HalfTag>(bottom_right);
    std::array<vec::Vec<Tag>, N> result{};
    VECOPS_UNROLL
    for (std::size_t i = 0; i < HalfN; ++i) {
      result[i] = vec::concat(Tag{}, tl[i], bl[i]);
      result[i + HalfN] = vec::concat(Tag{}, tr[i], br[i]);
    }
    return result;
  }
}

template <nint_t Rows, nint_t Columns,
          typename Source, typename Destination,
          int SrcRow, int SrcCol, int DstRow, int DstCol,
          Tile2DMaskMode MMask, Tile2DMaskMode NMask,
          meta::ValueType ActiveM, meta::ValueType ActiveN>
/**
 * @brief Execute one compile-time fixed register tile.
 * @param source Bound readable DataAccess.
 * @param src_origin Logical origin of the enclosing plane.
 * @param destination Bound writable DataAccess.
 * @param dst_origin Logical origin of the enclosing plane.
 * @param m_offset Row offset within the plane.
 * @param n_offset Column offset within the plane.
 * @param active_m Active rows when `MMask` is masked.
 * @param active_n Active columns when `NMask` is masked.
 */
VECOPS_ALWAYS_INLINE void fixed_kernel(
    Source& source, tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
    nint_t m_offset, nint_t n_offset,
    ActiveM active_m, ActiveN active_n) {
  using T = ComputeOf<Source>;
  using LoadTag = vec::FixedTag<T, Columns>;
  using StoreTag = vec::FixedTag<T, Rows>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  static_assert(Columns == Rows || Columns == 2 * Rows);
  std::array<vec::Vec<LoadTag>, static_cast<std::size_t>(Rows)> row_vectors{};

  VECOPS_UNROLL
  for (nint_t r = 0; r < Rows; ++r) {
    if constexpr (MMask == Tile2DMaskMode::masked) {
      if (r >= active_m_value) {
        row_vectors[static_cast<std::size_t>(r)] = vec::zeros(LoadTag{});
        continue;
      }
    }
    auto position = src_origin;
    position[SrcRow] += m_offset + r;
    position[SrcCol] += n_offset;
    if constexpr (NMask == Tile2DMaskMode::unmasked) {
      row_vectors[static_cast<std::size_t>(r)] =
          source.load(LoadTag{}, position, tensor::axis<SrcCol>);
    } else {
      row_vectors[static_cast<std::size_t>(r)] = source.load(
          LoadTag{}, position, tensor::axis<SrcCol>,
          vec::opt::first(active_n_value), vec::opt::zero);
    }
  }

  auto store_columns = [&](const auto& column_vectors,
                           nint_t column_base) VECOPS_INLINE_LAMBDA {
    VECOPS_UNROLL
    for (nint_t c = 0; c < Rows; ++c) {
      if constexpr (NMask == Tile2DMaskMode::masked) {
        if (column_base + c >= active_n_value) break;
      }
      auto position = dst_origin;
      position[DstRow] += n_offset + column_base + c;
      position[DstCol] += m_offset;
      if constexpr (MMask == Tile2DMaskMode::unmasked) {
        destination.store(
            StoreTag{}, position, tensor::axis<DstCol>,
            column_vectors[static_cast<std::size_t>(c)]);
      } else {
        destination.store(
            StoreTag{}, position, tensor::axis<DstCol>,
            column_vectors[static_cast<std::size_t>(c)],
            vec::opt::first(active_m_value));
      }
    }
  };

  if constexpr (Columns == Rows) {
    store_columns(transpose_square<LoadTag>(row_vectors), 0);
  } else {
    static_assert(std::same_as<vec::Half<LoadTag>, StoreTag>);
    {
      std::array<vec::Vec<StoreTag>, static_cast<std::size_t>(Rows)> lower{};
      VECOPS_UNROLL
      for (nint_t r = 0; r < Rows; ++r) {
        lower[static_cast<std::size_t>(r)] = vec::lower(
            LoadTag{}, row_vectors[static_cast<std::size_t>(r)]);
      }
      store_columns(transpose_square<StoreTag>(lower), 0);
    }
    {
      std::array<vec::Vec<StoreTag>, static_cast<std::size_t>(Rows)> upper{};
      VECOPS_UNROLL
      for (nint_t r = 0; r < Rows; ++r) {
        upper[static_cast<std::size_t>(r)] = vec::upper(
            LoadTag{}, row_vectors[static_cast<std::size_t>(r)]);
      }
      store_columns(transpose_square<StoreTag>(upper), Rows);
    }
  }
}

template <typename M, typename N, typename Source, typename Destination,
          int SrcRow, int SrcCol, int DstRow, int DstCol>
VECOPS_ALWAYS_INLINE void gather_transpose(
    M m, N n, Source& source,
    tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin);

template <nint_t Rows, nint_t Columns,
          typename M, typename N, typename Source, typename Destination,
          int SrcRow, int SrcCol, int DstRow, int DstCol>
/**
 * @brief Tile a logical plane and execute the fixed register network.
 * @param m Logical source rows.
 * @param n Logical source columns.
 * @param source Bound readable access.
 * @param destination Bound writable access.
 * @param src_origin Source plane origin.
 * @param dst_origin Destination plane origin.
 *
 * Full tiles use `fixed_kernel`. A raw same-dtype bottom edge keeps the fixed
 * network with masked row stores; right and corner tails use
 * `gather_transpose`. Tile2D receives meta extents unchanged, so Const and
 * constrained Dynamic information can remove empty tail regions at compile time.
 */
VECOPS_ALWAYS_INLINE void fixed_transpose(
    M m, N n, Source& source,
    tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin) {
  using Family = Tile2DKernelFamily<1, 1, 1>;
  using Catalog = Tile2DKernelCatalog<Family>;
  loop::tile2d<loop::tile2d_policy::FourRegions>(
      m, n, cint<Rows>, cint<Columns>, Catalog{},
      [&](auto kernel_case, nint_t mi, nint_t ni,
          nint_t active_m, nint_t active_n) VECOPS_INLINE_LAMBDA {
        using Case = decltype(kernel_case);
        constexpr bool FullTile =
            Case::m_mask == Tile2DMaskMode::unmasked &&
            Case::n_mask == Tile2DMaskMode::unmasked;
        constexpr bool ContiguousBottomEdge =
            Case::m_mask == Tile2DMaskMode::masked &&
            Case::n_mask == Tile2DMaskMode::unmasked &&
            is_raw_direct_access_v<Source> &&
            is_raw_direct_access_v<Destination>;
        const auto active_m_value =
            active_extent<Rows, Case::m_mask>(active_m);
        const auto active_n_value =
            active_extent<Columns, Case::n_mask>(active_n);
        if constexpr (FullTile || ContiguousBottomEdge) {
          fixed_kernel<Rows, Columns, Source, Destination,
                       SrcRow, SrcCol, DstRow, DstCol,
                       Case::m_mask, Case::n_mask>(
              source, src_origin, destination, dst_origin,
              mi, ni, active_m_value, active_n_value);
        } else {
          auto tail_source = src_origin;
          tail_source[SrcRow] += mi;
          tail_source[SrcCol] += ni;
          auto tail_destination = dst_origin;
          tail_destination[DstRow] += ni;
          tail_destination[DstCol] += mi;
          gather_transpose<
              decltype(active_m_value), decltype(active_n_value),
              Source, Destination, SrcRow, SrcCol, DstRow, DstCol>(
              active_m_value, active_n_value,
              source, tail_source, destination, tail_destination);
        }
      });
}

/**
 * @brief Gather one transposed output vector and store it contiguously.
 * @param m Logical source rows and therefore each output row's length.
 * @param n Logical source columns and therefore number of output rows.
 * @param source Bound readable access performing input conversion/transforms.
 * @param destination Bound writable access performing output transforms.
 * @param src_origin Source logical plane origin.
 * @param dst_origin Destination logical plane origin.
 *
 * The vector axis follows source rows. Loads may therefore be strided/gathered,
 * while destination stores are contiguous along the transposed column axis.
 */
template <typename M, typename N, typename Source, typename Destination,
          int SrcRow, int SrcCol, int DstRow, int DstCol>
VECOPS_ALWAYS_INLINE void gather_transpose(
    M m, N n, Source& source,
    tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin) {
  using T = ComputeOf<Source>;
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t m_extent = static_cast<nint_t>(m);
  const nint_t n_extent = static_cast<nint_t>(n);
  const nint_t lanes = vec::size(Tag{});
  for (nint_t col = 0; col < n_extent; ++col) {
    nint_t row = 0;
    for (; row + lanes <= m_extent; row += lanes) {
      auto src_position = src_origin;
      src_position[SrcRow] += row;
      src_position[SrcCol] += col;
      auto value = source.load(Tag{}, src_position, tensor::axis<SrcRow>);
      auto dst_position = dst_origin;
      dst_position[DstRow] += col;
      dst_position[DstCol] += row;
      destination.store(Tag{}, dst_position, tensor::axis<DstCol>, value);
    }
    if (row < m_extent) {
      const nint_t active = m_extent - row;
      auto src_position = src_origin;
      src_position[SrcRow] += row;
      src_position[SrcCol] += col;
      auto value = source.load(
          Tag{}, src_position, tensor::axis<SrcRow>,
          vec::opt::first(active), vec::opt::zero);
      auto dst_position = dst_origin;
      dst_position[DstRow] += col;
      dst_position[DstCol] += row;
      destination.store(
          Tag{}, dst_position, tensor::axis<DstCol>, value,
          vec::opt::first(active));
    }
  }
}

template <nint_t Rows, nint_t Columns,
          int SrcRow, int SrcCol, int DstRow, int DstCol,
          typename M, typename N, typename Source, typename Destination,
          typename Policy>
/**
 * @brief Compile-time policy dispatch shared by fixed-width vector backends.
 * @param policy `Automatic` selects the fixed network; `Gather` forces gather.
 *
 * This is an `if constexpr` dispatch and emits no runtime policy branch.
 */
VECOPS_ALWAYS_INLINE void fixed_or_gather_transpose(
    M m, N n, Source& source,
    tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
    Destination& destination,
    tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
    Policy) {
  if constexpr (std::same_as<Policy, transpose2d_policy::Gather>) {
    gather_transpose<
        M, N, Source, Destination, SrcRow, SrcCol, DstRow, DstCol>(
        m, n, source, src_origin, destination, dst_origin);
  } else {
    static_assert(std::same_as<Policy, transpose2d_policy::Automatic>);
    fixed_transpose<
        Rows, Columns, M, N, Source, Destination,
        SrcRow, SrcCol, DstRow, DstCol>(
        m, n, source, src_origin, destination, dst_origin);
  }
}

} // namespace vecops::kernel::transpose2d_details::generic

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_GENERIC_TRANSPOSE2D_H
