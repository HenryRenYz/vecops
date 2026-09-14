//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H
#define VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H

/**
 * @file vecops/matmul/details/kernel/sme/KuplMma.h
 * @brief Optional KUPL BF16 MMA provider for the SME whole-problem path.
 *
 * KUPL's KP36 BF16 leaf consumes one complete 16 x 64 output tile and its
 * own K-major A16/B64 operand layouts.  Those layouts intentionally remain
 * private because vecops' public SME packing ABI also serves specialized
 * native paths.  Raw operands are always packed online into KUPL's format.
 * The provider also owns a 16 x 64 accumulator staging tile.  It is entered
 * before vecops opens a
 * StreamingZARegion: the current KUPL Clang implementation owns SMSTART and
 * SMSTOP across each mma/store pair.
 *
 * Eligible large problems round M/N/K up to complete tiles while packing;
 * padded output lanes are staged as zero and never committed.  Small and
 * specialized GEMV/narrow/dot dispatches remain on the ordinary SME backend.
 */

#include <algorithm>
#include <limits>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/packing/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_details::sme::kupl_mma {

inline constexpr nint_t MTile = 16;
inline constexpr nint_t NTile = 64;
inline constexpr nint_t KPack = 2;
inline constexpr nint_t CTileElements = MTile * NTile;

/**
 * Isolated ABI bridge implemented in src/arch/sme/KuplMma.cpp.  Void operand
 * pointers keep KUPL's native __bf16 spelling out of vecops headers.
 */
void run_bf16_tile(const void* packed_a, const void* packed_b,
                   float32_t* c, nint_t padded_k);

VECOPS_ALWAYS_INLINE nint_t padded_m(nint_t m) {
  return ceil_div(m, MTile) * MTile;
}

VECOPS_ALWAYS_INLINE nint_t padded_n(nint_t n) {
  return ceil_div(n, NTile) * NTile;
}

VECOPS_ALWAYS_INLINE nint_t padded_k(nint_t k) {
  return ceil_div(k, KPack) * KPack;
}

/** Streaming vector length in bytes, used to guard KUPL's fixed tile ABI. */
VECOPS_ALWAYS_INLINE nint_t streaming_vector_bytes() noexcept {
  nint_t bytes;
  asm volatile("rdsvl %0, #1" : "=r"(bytes));
  return bytes;
}

/** Runtime gate for the initial BF16 provider. */
VECOPS_ALWAYS_INLINE bool profitable(nint_t m, nint_t n, nint_t k) {
  // KUPL's KP36 leaf is fixed to the 512-bit 920F tile geometry.  Checking
  // both ordinary VL (used by the private packer) and SVL (used by MMA)
  // prevents an enabled build from silently applying the fixed tile elsewhere.
  return m >= MTile && n >= NTile && k >= 32 &&
      vec::size(vec::ScalableTag<float32_t, 0>{}) == MTile &&
      streaming_vector_bytes() == 64;
}

VECOPS_ALWAYS_INLINE nint_t checked_product(nint_t a, nint_t b) {
  VECOPS_ASSERT(a >= 0 && b >= 0, "KUPL workspace extents must be non-negative");
  constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
  VECOPS_ASSERT(a == 0 || b <= Limit / a, "KUPL workspace size overflows nint_t");
  return a * b;
}

/** Workspace used by the mutually exclusive KUPL whole-problem branch. */
VECOPS_INLINE nint_t workspace_bytes(
    nint_t m, nint_t n, nint_t k, bool pack_a, bool pack_b) {
  if (!profitable(m, n, k)) return 0;
  const nint_t pk = padded_k(k);
  const nint_t a_elements = pack_a ? checked_product(padded_m(m), pk) : 0;
  const nint_t b_elements = pack_b ? checked_product(padded_n(n), pk) : 0;
  const nint_t ab_elements = a_elements + b_elements;
  VECOPS_ASSERT(ab_elements >= a_elements,
                "KUPL packed operand size overflows nint_t");
  const nint_t packed_bytes = checked_product(
      ab_elements, static_cast<nint_t>(sizeof(bfloat16_t)));
  const nint_t c_bytes = CTileElements * static_cast<nint_t>(sizeof(float32_t));
  // The accumulator and each online-packed operand are separate aligned
  // allocations.  Caller-prepared operands do not consume execution storage.
  return packed_bytes + c_bytes +
      (1 + static_cast<nint_t>(pack_a) + static_cast<nint_t>(pack_b)) * 63;
}

template <typename Source>
VECOPS_NOINLINE void pack_operand(
    const Source& source, bfloat16_t* destination,
    nint_t spatial, nint_t k, nint_t panel) {
  using HalfTag = vec::ScalableTag<bfloat16_t, -1>;
  VECOPS_ASSERT(vec::size(HalfTag{}) == MTile,
                "KUPL packing requires a 512-bit ordinary SVE vector");
  matmul_pack_details::generic::pack_interleaved_panels<KPack, HalfTag>(
      source, destination, spatial, k, panel, ceil_div(k, KPack));
}

template <typename CInput>
VECOPS_ALWAYS_INLINE void stage_c_input(
    const CInput& input, float32_t* tile, nint_t m, nint_t n,
    nint_t logical_m, nint_t logical_n) {
  using Tag = vec::ScalableTag<float32_t, 0>;
  if constexpr (tensor::is_zero_vec_transform_v<typename CInput::Transform>) {
    const auto zero = vec::zeros(Tag{});
    for (nint_t offset = 0; offset < CTileElements; offset += MTile)
      vec::store(Tag{}, tile + offset, zero);
  } else {
    for (nint_t row = 0; row < MTile; ++row) {
      for (nint_t block = 0; block < NTile / MTile; ++block) {
        const nint_t column = n + block * MTile;
        const nint_t active = m + row < logical_m
            ? vecops::clamp(logical_n - column, nint_t{0}, MTile)
            : 0;
        auto* destination = tile + row * NTile + block * MTile;
        if (active > 0) {
          const auto value = input.load(
              Tag{}, tensor::coord(m + row, column), tensor::axis<1>,
              vec::opt::first(active), vec::opt::zero);
          vec::store(Tag{}, destination, value);
        } else {
          vec::store(Tag{}, destination, vec::zeros(Tag{}));
        }
      }
    }
  }
}

template <typename COutput>
VECOPS_ALWAYS_INLINE void store_c_output(
    COutput& output, const float32_t* tile, nint_t m, nint_t n,
    nint_t logical_m, nint_t logical_n) {
  using Tag = vec::ScalableTag<float32_t, 0>;
  for (nint_t row = 0; row < MTile; ++row) {
    if (m + row >= logical_m) break;
    for (nint_t block = 0; block < NTile / MTile; ++block) {
      const nint_t column = n + block * MTile;
      const nint_t active = vecops::clamp(
          logical_n - column, nint_t{0}, MTile);
      if (active == 0) continue;
      const auto value = vec::load(
          Tag{}, tile + row * NTile + block * MTile);
      output.store(
          Tag{}, tensor::coord(m + row, column),
          tensor::axis<1>, value, vec::opt::first(active));
    }
  }
}

/** Pack as needed and execute one rank-two BF16 problem. */
template <typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE void run_bulk(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    bfloat16_t* packed_a, bfloat16_t* packed_b, float32_t* c_tile,
    nint_t m, nint_t n, nint_t k) {
  using ASpec = typename std::remove_cvref_t<A>::SpecType;
  using BSpec = typename std::remove_cvref_t<B>::SpecType;
  constexpr bool APacked = ::vecops::matmul::is_packed_layout<
      ::vecops::matmul::SME_BF16F32, ::vecops::matmul::Operand::A,
      typename ASpec::InputLayout>();
  constexpr bool BPacked = ::vecops::matmul::is_packed_layout<
      ::vecops::matmul::SME_BF16F32, ::vecops::matmul::Operand::B,
      typename BSpec::InputLayout>();
  const nint_t pm = padded_m(m);
  const nint_t pn = padded_n(n);
  const nint_t pk = padded_k(k);
  const nint_t k_groups = pk / KPack;

  const auto* a_base = [&] {
    if constexpr (APacked) {
      return reinterpret_cast<const bfloat16_t*>(a.raw_data());
    } else {
      pack_operand(a, packed_a, pm, k, MTile);
      return static_cast<const bfloat16_t*>(packed_a);
    }
  }();
  const auto* b_base = [&] {
    if constexpr (BPacked) {
      return reinterpret_cast<const bfloat16_t*>(b.raw_data());
    } else {
      pack_operand(b, packed_b, pn, k, NTile);
      return static_cast<const bfloat16_t*>(packed_b);
    }
  }();

  const nint_t a_tile_elements = MTile * pk;
  const nint_t b_tile_elements = NTile * pk;
  for (nint_t mi = 0; mi < pm; mi += MTile) {
    const auto* a_tile = a_base + (mi / MTile) * a_tile_elements;
    for (nint_t ni = 0; ni < pn; ni += NTile) {
      const auto* b_tile = b_base + (ni / NTile) * b_tile_elements;
      stage_c_input(c_input, c_tile, mi, ni, m, n);
      run_bf16_tile(a_tile, b_tile, c_tile, k_groups * KPack);
      store_c_output(c_output, c_tile, mi, ni, m, n);
    }
  }
}

} // namespace vecops::kernel::matmul_details::sme::kupl_mma

#endif // VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H
