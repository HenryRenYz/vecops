//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/Packing.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sme/ZA.h"

namespace vecops::kernel::matmul_pack_details::sme {

inline constexpr nint_t PrefetchChunks = 4;

template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<
        sizeof(T) == 2, uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

template <typename U>
VECOPS_ALWAYS_INLINE auto load_bits(
    vec::Mask<vec::ScalableTag<U, 0>> pg, const void* pointer) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::load(
      Tag{}, static_cast<const U*>(pointer),
      vec::opt::masked(pg), vec::opt::zero);
}

template <typename U>
VECOPS_ALWAYS_INLINE auto first(nint_t count) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::mwhilelt(Tag{}, nint_t{0}, count);
}

template <typename U>
VECOPS_ALWAYS_INLINE auto all() noexcept {
  return vec::mtrue(vec::ScalableTag<U, 0>{});
}

template <int Tile, typename U, typename Raw>
VECOPS_ALWAYS_INLINE void write_hor(
    uint32_t slice, vec::Mask<vec::ScalableTag<U, 0>> pg,
    Raw value) noexcept {
  vec::details::sme::write_hor<Tile>(slice, pg, value);
}

template <int Tile, typename U>
VECOPS_ALWAYS_INLINE auto read_ver(
    uint32_t slice, vec::Mask<vec::ScalableTag<U, 0>> pg) noexcept {
  using Tag = vec::ScalableTag<U, 0>;
  return vec::details::sme::read_ver<Tile>(Tag{}, slice, pg);
}

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

template <gemm::Atom Atom, gemm::Operand Side>
VECOPS_ALWAYS_INLINE void pack(
    const typename gemm::packing_t<Atom, Side>::Element* input,
    nint_t spatial, nint_t k, nint_t row_stride,
    typename gemm::packing_t<Atom, Side>::Element* output) noexcept {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  using U = Bits<T>;
  constexpr nint_t KPack = Packing::KPack;
  const nint_t panel = static_cast<nint_t>(Packing::panel());
  const nint_t lanes = vec::size(vec::ScalableTag<U, 0>{});
  const nint_t k_chunk = (lanes / KPack) * KPack;
  const nint_t panels = ceil_div(spatial, panel);
  auto* out = reinterpret_cast<U*>(output);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const bool full_panel = spatial - panel_base >= panel;
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

      if constexpr (sizeof(U) >= 4) {
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
        for (nint_t g = 0; g < groups; ++g) {
          store_pair<U>(
              out,
              read_ver<0, U>(static_cast<uint32_t>(g * KPack), first_pg),
              read_ver<0, U>(
                  static_cast<uint32_t>(g * KPack + 1), first_pg));
          out += panel * KPack;
        }
      } else {
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
      if (full_panel) {
        kb = pack_full_word_panels(
            input, panel_base, k, row_stride, k_chunk, out);
      }
    }
    constexpr nint_t PrefetchMinRowBytes = 1024;
    if (k * static_cast<nint_t>(sizeof(U)) >= PrefetchMinRowBytes) {
      for (; kb + (PrefetchChunks + 1) * k_chunk <= k; kb += k_chunk) {
        pack_chunk.template operator()<true, true>(kb, k_chunk);
      }
    }
    for (; kb + k_chunk <= k; kb += k_chunk) {
      pack_chunk.template operator()<true, false>(kb, k_chunk);
    }
    if (kb < k) {
      pack_chunk.template operator()<false, false>(kb, k - kb);
    }
  }
}

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

template <int Tile>
VECOPS_ALWAYS_INLINE void store_fp32_to_fp64_tile(
    float64_t*& output, nint_t active_spatial,
    nint_t active_k) noexcept {
  const auto store_pg = first<uint32_t>(active_spatial);
  for (nint_t column = 0; column < active_k; ++column) {
    store_fp32_to_fp64_column<Tile>(output, column, store_pg);
  }
}

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
VECOPS_ALWAYS_INLINE void pack_fp32_to_fp64(
    const float32_t* input, nint_t spatial, nint_t k,
    nint_t row_stride, float64_t* output) noexcept {
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

/** Pack FP32 spatial panels through a compact single ZA tile. */
VECOPS_ALWAYS_INLINE void pack_fp32_to_fp64_single(
    const float32_t* input, nint_t spatial, nint_t k,
    nint_t row_stride, float64_t* output) noexcept {
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

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H
