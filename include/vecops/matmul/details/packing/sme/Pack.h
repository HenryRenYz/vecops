//
// Copyright (c) vecops contributors.
//

/**
 * @file vecops/matmul/details/packing/sme/Pack.h
 * @brief ZA-transpose packers for the SME block format.
 *
 * Packing into the SME layout (packing/sme/Format.h) is a transpose:
 * source rows are written into ZA horizontally and K columns are read
 * back vertically.  write_hor / read_ver (plus the load_hor / store_ver
 * full-vector moves) are that vocabulary; everything else in this file
 * schedules ZA tile traffic around it.
 *
 * ZA usage conventions:
 *
 * - The spatial panel is 2 x (SVL/4) rows — two ZA32 tiles' worth, the
 *   "two-tile panel" of the format contract.  Staged at an element's own
 *   width w, a ZA tile holds SVL/w slices: >= 4-byte elements need tiles
 *   0 and 1 (half the panel each), 2-byte elements fit the whole panel
 *   in tile 0, and bytes fill half a tile.
 * - Full-panel chunks double-buffer through the two tile *pairs* 0/1 and
 *   2/3: loads of chunk N+1 into one pair overlap stores of chunk N from
 *   the other (see pack_full_word_panels).
 * - Everything here executes inside a Streaming + ZA region opened by the
 *   caller: the ZA moves are streaming instructions (the SME packing
 *   Backend wraps these entries in StreamingZARegion).
 *
 * Entry points: pack() covers every supported element width, and the
 * fp32 -> fp64 family (pack_fp32_to_fp64 / pack_fp32_to_fp64_single)
 * transposes while widening — see packing/sme/Backend.h for how the
 * variants are selected.
 */

#ifndef VECOPS_MATMUL_DETAILS_PACK_SME_PACK_H
#define VECOPS_MATMUL_DETAILS_PACK_SME_PACK_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/matmul/Packing.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sme/ZA.h"

namespace vecops::kernel::matmul_pack_details::sme {

/// Prefetch distance in K chunks used by the streaming pack loops below.
inline constexpr nint_t PrefetchChunks = 4;

/// Unsigned integer of the same width as T: ZA moves are defined on
/// integer views, so float rows are staged through ZA as bits.
template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<
        sizeof(T) == 2, uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

/// Masked load of one native word of U (zeroing masked-off lanes).
template <typename U>
VECOPS_ALWAYS_INLINE auto load_bits(
    vec::Mask<vec::ScalableTag<U, 0>> pg, const void* pointer) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::load(
      Tag{}, static_cast<const U*>(pointer),
      vec::opt::masked(pg), vec::opt::zero);
}

/// whilelt mask over the first `count` lanes at width U.
template <typename U>
VECOPS_ALWAYS_INLINE auto first(nint_t count) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::mwhilelt(Tag{}, nint_t{0}, count);
}

/// All-true mask at width U.
template <typename U>
VECOPS_ALWAYS_INLINE auto all() noexcept {
  return vec::mtrue(vec::ScalableTag<U, 0>{});
}

/// Stage one row into a ZA tile (thin re-export of the ZA move typed for
/// native words of U).
template <int Tile, typename U, typename Raw>
VECOPS_ALWAYS_INLINE void write_hor(
    uint32_t slice, vec::Mask<vec::ScalableTag<U, 0>> pg,
    Raw value) noexcept {
  vec::details::sme::write_hor<Tile>(slice, pg, value);
}

/// Read one K column out of a ZA tile — together with write_hor/load_hor,
/// the transpose vocabulary of this file.
template <int Tile, typename U>
VECOPS_ALWAYS_INLINE auto read_ver(
    uint32_t slice, vec::Mask<vec::ScalableTag<U, 0>> pg) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::details::sme::read_ver<Tile>(Tag{}, slice, pg);
}

/// Interleave two native words (the two K values of a 16-bit-element
/// K group) and store the packed group.
template <typename U, typename V0, typename V1>
VECOPS_ALWAYS_INLINE void store_pair(void* output, V0 v0, V1 v1)
    noexcept {
  using InTag = vec::ScalableTag<U, 0>;
  using OutTag = vec::ScalableTag<U, 1>;
  vec::store(OutTag{}, static_cast<U*>(output),
             vec::interleave(OutTag{},
                             static_cast<vec::Vec<InTag>>(v0),
                             static_cast<vec::Vec<InTag>>(v1)));
}

/// Interleave four native byte words (the four K values of an 8-bit-element
/// K group) into the packed [row][k(4)] group.  The tag chain mirrors
/// generic::interleave_quad: pair-interleave to 16-bit, view as uint16,
/// pair-interleave the views — two 16-bit zips equal one four-way byte
/// interleave.
template <typename U, typename V0, typename V1, typename V2, typename V3>
VECOPS_ALWAYS_INLINE void store_quad(
    void* output, vec::Mask<vec::ScalableTag<U, 0>> pg,
    V0 v0, V1 v1, V2 v2, V3 v3) noexcept {
  using InTag = vec::ScalableTag<U, 0>;
  using PairTag = vec::ScalableTag<U, 1>;
  using PairViewTag = vec::ViewAs<uint16_t, PairTag>;
  using QuadViewTag = vec::Twice<PairViewTag>;
  using OutTag = vec::ScalableTag<U, 2>;
  const auto pair01 = vec::interleave(
      PairTag{}, static_cast<vec::Vec<InTag>>(v0),
      static_cast<vec::Vec<InTag>>(v1));
  const auto pair23 = vec::interleave(
      PairTag{}, static_cast<vec::Vec<InTag>>(v2),
      static_cast<vec::Vec<InTag>>(v3));
  const auto pair01_view = vec::bitcast(PairViewTag{}, PairTag{}, pair01);
  const auto pair23_view = vec::bitcast(PairViewTag{}, PairTag{}, pair23);
  const auto quad_view = vec::interleave(
      QuadViewTag{}, pair01_view, pair23_view);
  const auto packed = vec::bitcast(OutTag{}, QuadViewTag{}, quad_view);
  (void)pg;
  // A byte K-pack contains exactly two native vector words.  Storing those
  // words directly avoids constructing a four-word predicate tuple.  Clang
  // otherwise lowers that tuple construction/access through out-of-line
  // SVE2.1-target helpers on CPUs where base SVE is selected, which would put
  // calls inside the manual Streaming+ZA region.
  auto* out = static_cast<U*>(output);
  constexpr nint_t StoredWords = 2;
  static_assert(vec::num_words(OutTag{}) >= StoredWords);
  const nint_t word_lanes = vec::size(InTag{});
  vec::store(InTag{}, out, vec::get_word<0>(OutTag{}, packed));
  vec::store(
      InTag{}, out + word_lanes, vec::get_word<1>(OutTag{}, packed));
}

/// One double-buffered step of the full-word path: load chunk data for the
/// panel into tile pair LoadBase while draining the previous chunk (held
/// in tile pair StoreBase) to memory.  Advances `output` past the chunk.
template <int LoadBase, int StoreBase, bool Prefetch,
          typename T, typename U>
VECOPS_ALWAYS_INLINE void transfer_full_word_panel(
    const T* input, nint_t panel_base, nint_t kb, nint_t row_stride,
    nint_t k_chunk, U*& output) noexcept {
  // SME outer products consume K in 32-bit groups: one FP32 value, two
  // BF16/FP16 values, or four bytes.  Treating every full source chunk as
  // one streaming vector of packed words turns packing into a native ZA32
  // transpose and
  // removes the read-ZA + ZIP + store sequence from the common path.  Separate
  // ZA tile pairs let loads for one K chunk overlap stores from the previous.
  //
  // Geometry: a full chunk of any <= 4-byte element type is exactly SVL
  // bytes wide, i.e. one ZA32 vector.  Each source row becomes one ZA32
  // horizontal slice (upper panel rows -> tile LoadBase, lower ->
  // LoadBase+1, because a tile holds only SVL/4 slices); each store_ver
  // drains one dword column as one contiguous K group of the packed
  // layout.
  using Word = uint32_t;
  using WordTag = vec::ScalableTag<Word, 0>;
  const nint_t word_lanes = vec::size(WordTag{});
  const auto pg = vec::mtrue(WordTag{});
  auto* words = reinterpret_cast<Word*>(output);

  for (nint_t row = 0; row < word_lanes; ++row) {
    const auto* first_row =
        input + (panel_base + row) * row_stride + kb;
    const auto* second_row =
        input + (panel_base + word_lanes + row) * row_stride + kb;
    if constexpr (Prefetch) {
      __builtin_prefetch(first_row + PrefetchChunks * k_chunk, 0, 3);
      __builtin_prefetch(second_row + PrefetchChunks * k_chunk, 0, 3);
    }
    vec::details::sme::load_hor<LoadBase>(
        static_cast<uint32_t>(row), pg,
        reinterpret_cast<const Word*>(first_row));
    vec::details::sme::load_hor<LoadBase + 1>(
        static_cast<uint32_t>(row), pg,
        reinterpret_cast<const Word*>(second_row));
    vec::details::sme::store_ver<StoreBase>(
        static_cast<uint32_t>(row), pg, words);
    words += word_lanes;
    vec::details::sme::store_ver<StoreBase + 1>(
        static_cast<uint32_t>(row), pg, words);
    words += word_lanes;
  }
  output = reinterpret_cast<U*>(words);
}

/// Load half of the double buffer: stage one full chunk into tile pair
/// TileBase (rows written horizontally, same geometry as
/// transfer_full_word_panel) without storing anything.
template <int TileBase, bool Prefetch, typename T>
VECOPS_ALWAYS_INLINE void load_full_word_panel(
    const T* input, nint_t panel_base, nint_t kb, nint_t row_stride,
    nint_t k_chunk) noexcept {
  using Word = uint32_t;
  using WordTag = vec::ScalableTag<Word, 0>;
  const nint_t word_lanes = vec::size(WordTag{});
  const auto pg = vec::mtrue(WordTag{});
  for (nint_t row = 0; row < word_lanes; ++row) {
    const auto* first_row =
        input + (panel_base + row) * row_stride + kb;
    const auto* second_row =
        input + (panel_base + word_lanes + row) * row_stride + kb;
    if constexpr (Prefetch) {
      __builtin_prefetch(first_row + PrefetchChunks * k_chunk, 0, 3);
      __builtin_prefetch(second_row + PrefetchChunks * k_chunk, 0, 3);
    }
    vec::details::sme::load_hor<TileBase>(
        static_cast<uint32_t>(row), pg,
        reinterpret_cast<const Word*>(first_row));
    vec::details::sme::load_hor<TileBase + 1>(
        static_cast<uint32_t>(row), pg,
        reinterpret_cast<const Word*>(second_row));
  }
}

/// Drain half of the double buffer: emit one chunk held in tile pair
/// TileBase as packed K groups — per group, the tile-0 column (upper panel
/// rows) then the tile-1 column (lower rows), which is exactly the packed
/// [row][k] order.  Advances `output` past the chunk.
template <int TileBase, typename U>
VECOPS_ALWAYS_INLINE void store_full_word_panel(U*& output) noexcept {
  using Word = uint32_t;
  using WordTag = vec::ScalableTag<Word, 0>;
  const nint_t word_lanes = vec::size(WordTag{});
  const auto pg = vec::mtrue(WordTag{});
  auto* words = reinterpret_cast<Word*>(output);
  for (nint_t group = 0; group < word_lanes; ++group) {
    vec::details::sme::store_ver<TileBase>(
        static_cast<uint32_t>(group), pg, words);
    words += word_lanes;
    vec::details::sme::store_ver<TileBase + 1>(
        static_cast<uint32_t>(group), pg, words);
    words += word_lanes;
  }
  output = reinterpret_cast<U*>(words);
}

/**
 * @brief Full-panel fast path: native ZA32 transpose with ping-pong
 *        scheduling over the two tile pairs.
 *
 * Chunk 0 is loaded into tile pair 0; each subsequent chunk is loaded into
 * the pair the concurrent store is not draining (odd chunks load pair 2 /
 * store pair 0, even chunks the mirror), so chunk N+1's loads overlap
 * chunk N's stores.  The final store drains whichever pair holds the last
 * chunk — its parity is that of `chunks - 1`, hence the `chunks & 1` test.
 * K elements beyond the last full chunk are left to the caller.  Returns
 * the number of K elements consumed.
 */
template <typename T, typename U>
VECOPS_ALWAYS_INLINE nint_t pack_full_word_panels(
    const T* input, nint_t panel_base, nint_t k, nint_t row_stride,
    nint_t k_chunk, U*& output) noexcept {
  const nint_t chunks = k / k_chunk;
  if (chunks == 0) return 0;

  if (chunks > PrefetchChunks) {
    load_full_word_panel<0, true>(
        input, panel_base, 0, row_stride, k_chunk);
  } else {
    load_full_word_panel<0, false>(
        input, panel_base, 0, row_stride, k_chunk);
  }
  for (nint_t chunk = 1; chunk < chunks; ++chunk) {
    const nint_t kb = chunk * k_chunk;
    const bool prefetch = chunk + PrefetchChunks < chunks;
    if (chunk & 1) {
      if (prefetch) {
        transfer_full_word_panel<2, 0, true>(
            input, panel_base, kb, row_stride, k_chunk, output);
      } else {
        transfer_full_word_panel<2, 0, false>(
            input, panel_base, kb, row_stride, k_chunk, output);
      }
    } else if (prefetch) {
      transfer_full_word_panel<0, 2, true>(
          input, panel_base, kb, row_stride, k_chunk, output);
    } else {
      transfer_full_word_panel<0, 2, false>(
          input, panel_base, kb, row_stride, k_chunk, output);
    }
  }
  if (chunks & 1) store_full_word_panel<0>(output);
  else store_full_word_panel<2>(output);
  return chunks * k_chunk;
}

/**
 * @brief Pack one raw, transform-free operand into the SME block format.
 *
 * Raw-pointer entry point behind the streaming SME backends; must run
 * inside a Streaming+ZA region (the caller opens it).  K is split into
 * chunks of k_chunk elements — one streaming vector wide — and each
 * spatial panel takes up to three paths:
 *
 * 1. full panels of <= 4-byte elements go through the double-buffered
 *    ZA32 word transpose (pack_full_word_panels);
 * 2. a prefetching chunk loop while enough chunks remain ahead and rows
 *    are wide enough to be worth prefetching;
 * 3. a plain chunk loop, plus a masked tail chunk when K does not divide
 *    evenly.
 *
 * SpatialGuaranteed / KGuaranteed fold to true only when the panel and
 * k_chunk Meta types are compile-time singletons (fixed SVL) and the
 * input shape aligns, which lets the tail chunk vanish entirely.
 */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType RowStride>
VECOPS_ALWAYS_INLINE void pack(
    const typename ::vecops::matmul::packing_t<Atom, Side>::Element* input,
    Spatial spatial_meta, K k_meta, RowStride row_stride_meta,
    typename ::vecops::matmul::packing_t<Atom, Side>::Element* output) noexcept {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  using U = Bits<T>;
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t row_stride = static_cast<nint_t>(row_stride_meta);
  constexpr nint_t KPack = Packing::KPack;
  const auto panel_meta = Packing::panel();
  const auto lanes_meta =
      vec::details::sme::streaming_lanes_value<U>();
  // Round the chunk length down to a whole number of KPack groups: a full
  // chunk must map whole K groups onto ZA slices so groups never straddle
  // the chunk boundary.
  const auto k_chunk_meta =
      (lanes_meta / meta::cint<KPack>) * meta::cint<KPack>;
  const nint_t panel = static_cast<nint_t>(panel_meta);
  const nint_t lanes = static_cast<nint_t>(lanes_meta);
  const nint_t k_chunk = static_cast<nint_t>(k_chunk_meta);
  using SpatialValue = std::remove_cvref_t<Spatial>;
  using KValue = std::remove_cvref_t<K>;
  using PanelValue = std::remove_cvref_t<decltype(panel_meta)>;
  using KChunkValue = std::remove_cvref_t<decltype(k_chunk_meta)>;
  constexpr bool SpatialGuaranteed = [] {
    if constexpr (meta::is_singleton_v<PanelValue>)
      return SpatialValue::aligns(meta::singleton_value_v<PanelValue>);
    else
      return false;
  }();
  constexpr bool KGuaranteed = [] {
    if constexpr (meta::is_singleton_v<KChunkValue>)
      return KValue::aligns(meta::singleton_value_v<KChunkValue>);
    else
      return false;
  }();
  const nint_t panels = ceil_div(spatial, panel);
  auto* out = reinterpret_cast<U*>(output);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const bool full_panel = SpatialGuaranteed ||
        spatial - panel_base >= panel;
    // Rows actually staged into tile 0: a tail panel loads only its active
    // rows.  The reads below mask to the same rows, and read_ver zero-fills
    // masked-off lanes — which is exactly the format's zero padding.
    const nint_t first_active = vec::details::sme::clamp_value(
        spatial - panel_base, nint_t{0}, vec::details::sme::min_value(panel, lanes));
    const auto first_pg = first<U>(first_active);

    auto pack_chunk = [&]<bool FullK, bool Prefetch>(
        nint_t kb, nint_t active_k) VECOPS_INLINE_LAMBDA_NOEXCEPT {
      const nint_t groups = FullK
          ? k_chunk / KPack
          : ceil_div(active_k, KPack);
      const auto load_pg = FullK ? all<U>() : first<U>(active_k);
      for (nint_t r = 0; r < first_active; ++r) {
        const nint_t row = panel_base + r;
        if constexpr (Prefetch) {
          const auto* row_input = input + row * row_stride + kb;
          __builtin_prefetch(
              row_input + PrefetchChunks * k_chunk, 0, 3);
        }
        vec::details::sme::load_hor<0>(
            static_cast<uint32_t>(r), load_pg,
            reinterpret_cast<const U*>(input + row * row_stride + kb));
      }

      // Store path per element width — the K index maps onto ZA slices at
      // the element's own granularity, because rows were staged
      // horizontally at that width:
      if constexpr (sizeof(U) >= 4) {
        // >= 4 bytes: the panel spans tiles 0 and 1 (a tile holds SVL/w
        // slices, half the panel here).  Group g is the single K value g
        // (KPack == 1); v0 / v1 are its upper / lower panel halves, stored
        // contiguously as the group.
        const nint_t active1 = vec::details::sme::clamp_value(
            spatial - (panel_base + lanes), nint_t{0}, lanes);
        for (nint_t r = 0; r < active1; ++r) {
          const nint_t row = panel_base + lanes + r;
          if constexpr (Prefetch) {
            const auto* row_input = input + row * row_stride + kb;
            __builtin_prefetch(
                row_input + PrefetchChunks * k_chunk, 0, 3);
          }
          vec::details::sme::load_hor<1>(
              static_cast<uint32_t>(r), load_pg,
              reinterpret_cast<const U*>(input + row * row_stride + kb));
        }
        const auto pg1 = first<U>(active1);
        for (nint_t g = 0; g < groups; ++g) {
          const auto v0 = read_ver<0, U>(static_cast<uint32_t>(g), first_pg);
          const auto v1 = read_ver<1, U>(static_cast<uint32_t>(g), pg1);
          vec::store(vec::ScalableTag<U, 0>{}, out, v0);
          out += lanes;
          vec::store(vec::ScalableTag<U, 0>{}, out, v1);
          out += lanes;
        }
      } else if constexpr (sizeof(U) == 2) {
        // 2 bytes: the whole panel fits tile 0 (SVL/2 slices == panel).
        // Group g reads the two adjacent ZA16 slices g*KPack and +1 — the
        // two K values of the group; store_pair interleaves them into the
        // packed [row][k(2)] group.
        for (nint_t g = 0; g < groups; ++g) {
          store_pair<U>(
              out,
              read_ver<0, U>(static_cast<uint32_t>(g * KPack), first_pg),
              read_ver<0, U>(
                  static_cast<uint32_t>(g * KPack + 1), first_pg));
          out += panel * KPack;
        }
      } else {
        // 1 byte: the panel fills half a tile (SVL/2 of SVL slices).
        // Group g reads the four adjacent ZA8 slices g*KPack .. +3;
        // store_quad interleaves them, masked to the panel lanes.
        const auto store_pg = first<U>(panel);
        for (nint_t g = 0; g < groups; ++g) {
          store_quad<U>(
              out, store_pg,
              read_ver<0, U>(static_cast<uint32_t>(g * KPack), first_pg),
              read_ver<0, U>(
                  static_cast<uint32_t>(g * KPack + 1), first_pg),
              read_ver<0, U>(
                  static_cast<uint32_t>(g * KPack + 2), first_pg),
              read_ver<0, U>(
                  static_cast<uint32_t>(g * KPack + 3), first_pg));
          out += panel * KPack;
        }
      }
    };

    nint_t kb = 0;
    if constexpr (sizeof(U) <= 4) {
      // The word fast path requires a full panel (it loads every panel row
      // with an all-true predicate) and <= 4-byte elements: a ZA32 word
      // must equal one whole K group (sizeof * KPack == 4); fp64's
      // single-element groups would be split across dwords.  Tail chunks
      // past the last full one fall through to the loops below.
      if (full_panel) {
        kb = pack_full_word_panels(
            input, panel_base, k, row_stride, k_chunk, out);
      }
    }
    // Prefetch only when a row spans at least 1 KiB: narrower rows are
    // typically cache-resident already, and the prefetch bookkeeping would
    // cost more than the misses it avoids.
    constexpr nint_t PrefetchMinRowBytes = 1024;
    if (k * static_cast<nint_t>(sizeof(U)) >= PrefetchMinRowBytes) {
      for (; kb + (PrefetchChunks + 1) * k_chunk <= k; kb += k_chunk) {
        pack_chunk.template operator()<true, true>(kb, k_chunk);
      }
    }
    for (; kb + k_chunk <= k; kb += k_chunk) {
      pack_chunk.template operator()<true, false>(kb, k_chunk);
    }
    if constexpr (!KGuaranteed)
      if (kb < k)
        pack_chunk.template operator()<false, false>(kb, k - kb);
  }
}

/// Stage one fp32 chunk into a ZA32 tile: the staging panel is one fp32
/// vector (SVL/4 rows), which fits a single tile exactly.  K is masked by
/// active_k; columns beyond it are never read back.
template <int Tile>
VECOPS_ALWAYS_INLINE void load_fp32_to_fp64_tile(
    const float32_t* input, nint_t panel_base, nint_t kb,
    nint_t row_stride, nint_t active_spatial,
    nint_t active_k) noexcept {
  const auto load_pg = first<uint32_t>(active_k);
  for (nint_t row = 0; row < active_spatial; ++row) {
    vec::details::sme::load_hor<Tile>(
        static_cast<uint32_t>(row), load_pg,
        reinterpret_cast<const uint32_t*>(
            input + (panel_base + row) * row_stride + kb));
  }
}

/// Store one K column of the resident chunk, widened to fp64: read the
/// ZA32 column (row-masked — masked-off lanes read as zero, which becomes
/// the format's zero padding), convert, and store one full padded panel
/// column, advancing the output by exactly that.
template <int Tile>
VECOPS_ALWAYS_INLINE void store_fp32_to_fp64_column(
    float64_t*& output, nint_t column,
    vec::Mask<vec::ScalableTag<uint32_t, 0>> store_pg) noexcept {
  using InputTag = vec::ScalableTag<float32_t, 0>;
  using BitsTag = vec::ScalableTag<uint32_t, 0>;
  using OutputTag = vec::Rebind<float64_t, InputTag>;
  const auto bits = vec::details::sme::read_ver<Tile>(
      BitsTag{}, static_cast<uint32_t>(column), store_pg);
  const auto input = vec::bitcast(InputTag{}, BitsTag{}, bits);
  vec::store(OutputTag{}, output,
             vec::convert(OutputTag{}, InputTag{}, input));
  output += vec::size(InputTag{});
}

/// Drain one whole chunk as widened K columns (store_fp32_to_fp64_column
/// over the chunk's active width).
template <int Tile>
VECOPS_ALWAYS_INLINE void store_fp32_to_fp64_tile(
    float64_t*& output, nint_t active_spatial,
    nint_t active_k) noexcept {
  const auto store_pg = first<uint32_t>(active_spatial);
  for (nint_t column = 0; column < active_k; ++column) {
    store_fp32_to_fp64_column<Tile>(output, column, store_pg);
  }
}

/// One double-buffered fp32 -> fp64 chunk step: load the *next* chunk
/// into tile LoadTile while storing the *current* chunk (resident in the
/// other tile) through widened columns.  Row count (active_spatial) and
/// the current chunk's K width (store_active_k) differ, so the interleaved
/// middle stage runs only to their minimum and the leftovers drain in two
/// dedicated loops — every index pairs an independent load with an
/// independent store while both remain.
template <int LoadTile, int StoreTile>
VECOPS_ALWAYS_INLINE void transfer_fp32_to_fp64_tile(
    const float32_t* input, nint_t panel_base, nint_t kb,
    nint_t row_stride, nint_t active_spatial, nint_t load_active_k,
    nint_t store_active_k, float64_t*& output) noexcept {
  const auto load_pg = first<uint32_t>(load_active_k);
  const auto store_pg = first<uint32_t>(active_spatial);
  const nint_t overlap =
      vec::details::sme::min_value(active_spatial, store_active_k);
  nint_t index = 0;
  for (; index < overlap; ++index) {
    vec::details::sme::load_hor<LoadTile>(
        static_cast<uint32_t>(index), load_pg,
        reinterpret_cast<const uint32_t*>(
            input + (panel_base + index) * row_stride + kb));
    store_fp32_to_fp64_column<StoreTile>(output, index, store_pg);
  }
  for (; index < active_spatial; ++index) {
    vec::details::sme::load_hor<LoadTile>(
        static_cast<uint32_t>(index), load_pg,
        reinterpret_cast<const uint32_t*>(
            input + (panel_base + index) * row_stride + kb));
  }
  for (; index < store_active_k; ++index) {
    store_fp32_to_fp64_column<StoreTile>(output, index, store_pg);
  }
}

/** Pack FP32 spatial panels with a double-buffered ZA32 transpose. */
template <meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType RowStride>
VECOPS_ALWAYS_INLINE void pack_fp32_to_fp64(
    const float32_t* input, Spatial spatial_meta, K k_meta,
    RowStride row_stride_meta, float64_t* output) noexcept {
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t row_stride = static_cast<nint_t>(row_stride_meta);
  using InputTag = vec::ScalableTag<float32_t, 0>;
  const nint_t panel = vec::size(InputTag{});
  const nint_t k_chunk = panel;
  const nint_t panels = ceil_div(spatial, panel);
  const nint_t chunks = ceil_div(k, k_chunk);

  for (nint_t sp = 0; sp < panels; ++sp) {
    if (chunks == 0) continue;
    const nint_t panel_base = sp * panel;
    const nint_t active_spatial = vec::details::sme::clamp_value(
        spatial - panel_base, nint_t{0}, panel);
    const nint_t first_active_k =
        vec::details::sme::min_value(k_chunk, k);
    load_fp32_to_fp64_tile<0>(
        input, panel_base, 0, row_stride,
        active_spatial, first_active_k);

    // Chunk c resides in tile (c & 1): the initial load uses tile 0 and
    // each transfer loads the next chunk into the tile the concurrent
    // store is not draining.
    for (nint_t chunk = 0; chunk + 1 < chunks; ++chunk) {
      const nint_t current_k = vec::details::sme::min_value(
          k_chunk, k - chunk * k_chunk);
      const nint_t next_k = vec::details::sme::min_value(
          k_chunk, k - (chunk + 1) * k_chunk);
      const nint_t next_kb = (chunk + 1) * k_chunk;
      if ((chunk & 1) == 0) {
        transfer_fp32_to_fp64_tile<1, 0>(
            input, panel_base, next_kb, row_stride,
            active_spatial, next_k, current_k, output);
      } else {
        transfer_fp32_to_fp64_tile<0, 1>(
            input, panel_base, next_kb, row_stride,
            active_spatial, next_k, current_k, output);
      }
    }
    const nint_t final_k = vec::details::sme::min_value(
        k_chunk, k - (chunks - 1) * k_chunk);
    if ((chunks & 1) != 0)
      store_fp32_to_fp64_tile<0>(output, active_spatial, final_k);
    else
      store_fp32_to_fp64_tile<1>(output, active_spatial, final_k);
  }
}

/**
 * Pack FP32 spatial panels through a compact single ZA tile.
 *
 * Same contract as pack_fp32_to_fp64 without the double buffering: load a
 * chunk, drain it, repeat.  The runtime architecture-family planner forces
 * this variant when the pack's row count is below the fp64 panel — mostly
 * zero padding, where the ping-pong schedule would just shuffle empty rows
 * (see packing/sme/Backend.h).
 */
template <meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType RowStride>
VECOPS_ALWAYS_INLINE void pack_fp32_to_fp64_single(
    const float32_t* input, Spatial spatial_meta, K k_meta,
    RowStride row_stride_meta, float64_t* output) noexcept {
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t row_stride = static_cast<nint_t>(row_stride_meta);
  const nint_t panel = vec::size(vec::ScalableTag<float32_t, 0>{});
  const nint_t panels = ceil_div(spatial, panel);
  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const nint_t active_spatial = vec::details::sme::clamp_value(
        spatial - panel_base, nint_t{0}, panel);
    for (nint_t kb = 0; kb < k; kb += panel) {
      const nint_t active_k =
          vec::details::sme::min_value(panel, k - kb);
      load_fp32_to_fp64_tile<0>(
          input, panel_base, kb, row_stride, active_spatial, active_k);
      store_fp32_to_fp64_tile<0>(output, active_spatial, active_k);
    }
  }
}

} // namespace vecops::kernel::matmul_pack_details::sme

#endif // VECOPS_MATMUL_DETAILS_PACK_SME_PACK_H
