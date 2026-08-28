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

template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;

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

template <gemm::Atom Atom, gemm::Operand Side>
VECOPS_ALWAYS_INLINE void pack(
    const typename gemm::packing_t<Atom, Side>::Element* input,
    nint_t spatial, nint_t k, nint_t row_stride,
    typename gemm::packing_t<Atom, Side>::Element* output) noexcept {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  using U = Bits<T>;
  constexpr nint_t KPack = Packing::KPack;
  const nint_t panel = 2 * vec::details::sme::streaming_lanes<uint32_t>();
  const nint_t lanes = vec::size(vec::ScalableTag<U, 0>{});
  const nint_t k_chunk = (lanes / KPack) * KPack;
  const nint_t panels = ceil_div(spatial, panel);
  auto* out = reinterpret_cast<U*>(output);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const nint_t first_active = vec::details::sme::clamp_value(
        spatial - panel_base, nint_t{0}, vec::details::sme::min_value(panel, lanes));
    const auto first_pg = first<U>(first_active);
    const auto write_pg = all<U>();

    auto pack_chunk = [&]<bool FullK, bool Prefetch>(
        nint_t kb, nint_t active_k) VECOPS_INLINE_LAMBDA_NOEXCEPT {
      const nint_t groups = FullK
          ? k_chunk / KPack
          : ceil_div(active_k, KPack);
      const auto load_pg = FullK ? all<U>() : first<U>(active_k);
      constexpr nint_t PrefetchChunks = 4;
      for (nint_t r = 0; r < first_active; ++r) {
        const nint_t row = panel_base + r;
        if constexpr (Prefetch) {
          const auto* row_input = input + row * row_stride + kb;
          __builtin_prefetch(
              row_input + PrefetchChunks * k_chunk, 0, 3);
        }
        write_hor<0, U>(
            static_cast<uint32_t>(r), write_pg,
            load_bits<U>(load_pg, input + row * row_stride + kb));
      }

      if constexpr (sizeof(U) == 4) {
        const nint_t active1 = vec::details::sme::clamp_value(
            spatial - (panel_base + lanes), nint_t{0}, lanes);
        for (nint_t r = 0; r < active1; ++r) {
          const nint_t row = panel_base + lanes + r;
          if constexpr (Prefetch) {
            const auto* row_input = input + row * row_stride + kb;
            __builtin_prefetch(
                row_input + PrefetchChunks * k_chunk, 0, 3);
          }
          write_hor<1, U>(
              static_cast<uint32_t>(r), write_pg,
              load_bits<U>(load_pg, input + row * row_stride + kb));
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
    constexpr nint_t PrefetchChunks = 4;
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

} // namespace vecops::kernel::matmul_pack_details::sme

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H
