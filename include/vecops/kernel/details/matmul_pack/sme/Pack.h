//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_PACK_H

#include <arm_sme.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/Packing.h"

namespace vecops::kernel::matmul_pack_details::sme {

template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;

template <typename U>
VECOPS_ALWAYS_INLINE auto load_bits(svbool_t pg, const void* pointer)
    __arm_streaming {
  if constexpr (sizeof(U) == 1)
    return svld1_u8(pg, static_cast<const uint8_t*>(pointer));
  else if constexpr (sizeof(U) == 2)
    return svld1_u16(pg, static_cast<const uint16_t*>(pointer));
  else
    return svld1_u32(pg, static_cast<const uint32_t*>(pointer));
}

template <typename U>
VECOPS_ALWAYS_INLINE auto zero_bits() __arm_streaming {
  if constexpr (sizeof(U) == 1) return svdup_u8(0);
  else if constexpr (sizeof(U) == 2) return svdup_u16(0);
  else return svdup_u32(0);
}

template <typename U>
VECOPS_ALWAYS_INLINE svbool_t first(nint_t count) __arm_streaming {
  if constexpr (sizeof(U) == 1) return svwhilelt_b8(nint_t{0}, count);
  else if constexpr (sizeof(U) == 2)
    return svwhilelt_b16(nint_t{0}, count);
  else return svwhilelt_b32(nint_t{0}, count);
}

template <typename U>
VECOPS_ALWAYS_INLINE svbool_t all() __arm_streaming {
  if constexpr (sizeof(U) == 1) return svptrue_b8();
  else if constexpr (sizeof(U) == 2) return svptrue_b16();
  else return svptrue_b32();
}

template <int Tile, typename U, typename Raw>
VECOPS_ALWAYS_INLINE void write_hor(
    uint32_t slice, svbool_t pg, Raw value)
    __arm_streaming __arm_inout("za") {
  if constexpr (sizeof(U) == 1)
    svwrite_hor_za8_u8_m(Tile, slice, pg, value);
  else if constexpr (sizeof(U) == 2)
    svwrite_hor_za16_u16_m(Tile, slice, pg, value);
  else
    svwrite_hor_za32_u32_m(Tile, slice, pg, value);
}

template <int Tile, typename U>
VECOPS_ALWAYS_INLINE auto read_ver(uint32_t slice, svbool_t pg)
    __arm_streaming __arm_inout("za") {
  if constexpr (sizeof(U) == 1)
    return svread_ver_za8_u8_m(zero_bits<U>(), pg, Tile, slice);
  else if constexpr (sizeof(U) == 2)
    return svread_ver_za16_u16_m(zero_bits<U>(), pg, Tile, slice);
  else
    return svread_ver_za32_u32_m(zero_bits<U>(), pg, Tile, slice);
}

template <typename U, typename V0, typename V1>
VECOPS_ALWAYS_INLINE void store_pair(void* output, V0 v0, V1 v1)
    __arm_streaming {
  svst2_u16(
      svptrue_b16(), static_cast<uint16_t*>(output),
      svcreate2_u16(v0, v1));
}

template <typename U, typename V0, typename V1, typename V2, typename V3>
VECOPS_ALWAYS_INLINE void store_quad(
    void* output, svbool_t pg, V0 v0, V1 v1, V2 v2, V3 v3)
    __arm_streaming {
  svst4_u8(
      pg, static_cast<uint8_t*>(output),
      svcreate4_u8(v0, v1, v2, v3));
}

template <gemm::Atom Atom, gemm::Operand Side>
__arm_new("za") VECOPS_NOINLINE void pack(
    const typename gemm::packing_t<Atom, Side>::Element* input,
    nint_t spatial, nint_t k, nint_t row_stride,
    typename gemm::packing_t<Atom, Side>::Element* output)
    __arm_streaming {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  using U = Bits<T>;
  constexpr nint_t KPack = Packing::KPack;
  const nint_t panel = 2 * static_cast<nint_t>(svcntw());
  const nint_t lanes = []() VECOPS_INLINE_LAMBDA __arm_streaming {
    if constexpr (sizeof(U) == 1) return static_cast<nint_t>(svcntb());
    else if constexpr (sizeof(U) == 2)
      return static_cast<nint_t>(svcnth());
    else return static_cast<nint_t>(svcntw());
  }();
  const nint_t k_chunk = (lanes / KPack) * KPack;
  const nint_t panels = ceil_div(spatial, panel);
  auto* out = reinterpret_cast<U*>(output);

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const nint_t first_active = std::clamp(
        spatial - panel_base, nint_t{0}, std::min(panel, lanes));
    const auto first_pg = first<U>(first_active);
    const auto write_pg = all<U>();

    auto pack_chunk = [&]<bool FullK, bool Prefetch>(
        nint_t kb, nint_t active_k)
        VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
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
        const nint_t active1 = std::clamp(
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
          svst1_u32(svptrue_b32(), reinterpret_cast<uint32_t*>(out), v0);
          out += lanes;
          svst1_u32(svptrue_b32(), reinterpret_cast<uint32_t*>(out), v1);
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
