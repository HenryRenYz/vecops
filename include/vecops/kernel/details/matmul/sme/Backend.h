//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_SME_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_SME_BACKEND_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <arm_sme.h>

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/gemm/Packing.h"
#include "vecops/gemm/details/sme/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::kernel::matmul_details::sme {

namespace generic = matmul_pack_details::generic;
namespace tile = ::vecops::kernel::loop;

template <typename T>
struct IsZeroTransform : std::false_type {};

template <typename Out, typename In>
struct IsZeroTransform<tensor::ZeroVecTransform<Out, In>> : std::true_type {};

template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

template <typename Access>
using InputLayoutOf = typename SpecOf<Access>::InputLayout;

template <typename Access>
using OutputLayoutOf = typename SpecOf<Access>::OutputLayout;

template <gemm::Atom Atom, gemm::Operand Side, typename Access>
inline constexpr bool is_packed_access_v =
    gemm::is_packed_layout<Atom, Side, InputLayoutOf<Access>>();

template <typename Access>
inline constexpr bool direct_row_major_input_v =
    generic::RawDirectAccess<Access> && Access::Rank == 2 &&
    std::same_as<
        tensor::stride_type_t<1, InputLayoutOf<Access>>, meta::Const<1>>;

template <typename Access>
inline constexpr bool direct_row_major_output_v =
    generic::RawDirectAccess<Access> && Access::Rank == 2 &&
    std::same_as<
        tensor::stride_type_t<1, OutputLayoutOf<Access>>, meta::Const<1>>;

struct KernelProvider {
  template <int A, int B, tile::Tile2DMaskMode, tile::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B <= 4) return 100 * A * B + 4 * (A + B);
    else return -1;
  }
};

using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider, tile::Tile2DSearchSpace<4, 4, 4>>;

consteval int log2_kpack(nint_t kpack) {
  return kpack == 4 ? 2 : (kpack == 2 ? 1 : 0);
}

template <gemm::Atom Atom>
VECOPS_ALWAYS_INLINE nint_t accumulator_lanes()
    __arm_streaming {
  if constexpr (sizeof(typename Atom::TAcc) == 8)
    return static_cast<nint_t>(svcntd());
  else
    return static_cast<nint_t>(svcntw());
}

template <typename T, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto as_native(V value)
    __arm_streaming {
  using Tag = vec::ScalableTag<T, 0>;
  return static_cast<vec::Vec<Tag>>(value);
}

VECOPS_ALWAYS_INLINE svuint32_t gather_words(
    svbool_t pg, const uint32_t* base, svint32_t offsets)
    __arm_streaming {
#if defined(COMPILER_CLANG)
  // WORKAROUND: BiSheng Clang 19 crashes while selecting the ACLE
  // s32-offset gather in streaming mode. Spell the same instruction directly;
  // unlike a scalar fallback this keeps online packing to one load per row.
  svuint32_t result;
  asm volatile(
      "ld1w {%0.s}, %1/z, [%2, %3.s, sxtw]"
      : "=w"(result)
      : "Upl"(pg), "r"(base), "w"(offsets)
      : "memory");
  return result;
#else
  return svld1_gather_s32offset_u32(pg, base, offsets);
#endif
}

template <gemm::Atom Atom, gemm::Operand Side, int Block, typename Source>
VECOPS_ALWAYS_INLINE auto load_operand(
    const Source& source, nint_t origin, nint_t kg,
    nint_t logical_spatial, nint_t logical_k)
    __arm_streaming {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  constexpr nint_t KP = Packing::KPack;
  using ScalarTag = vec::ScalableTag<T, -log2_kpack(KP)>;
  using NativeTag = vec::ScalableTag<T, 0>;
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t spatial = origin + static_cast<nint_t>(Block) * lanes;
  if constexpr (is_packed_access_v<Atom, Side, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed SME input must be direct and untransformed");
    const auto& layout = source.spec().input_layout();
    const nint_t panel = Packing::panel();
    const nint_t offset = tensor::offset_at(
        layout, spatial / panel, kg, spatial % panel, 0);
    return vec::load(
        NativeTag{}, reinterpret_cast<const T*>(source.raw_data()) + offset);
  } else {
    static_assert(Source::Rank == 2, "unpacked SME input must be rank two");
    const nint_t active = std::clamp(
        logical_spatial - spatial, nint_t{0}, lanes);
    if constexpr (generic::RawDirectAccess<Source> && sizeof(T) <= 4) {
      const auto strides = source.raw_strides();
      const nint_t k = kg * KP;
      const nint_t row_bytes = strides[0] * static_cast<nint_t>(sizeof(T));
      if (strides[1] == 1 && k + KP <= logical_k &&
          row_bytes >= std::numeric_limits<int32_t>::min() &&
          row_bytes <= std::numeric_limits<int32_t>::max()) {
        const svbool_t pg = svwhilelt_b32(nint_t{0}, active);
        const svint32_t offsets = svindex_s32(
            0, static_cast<int32_t>(row_bytes));
        const auto* base = reinterpret_cast<const uint32_t*>(
            source.raw_data() + spatial * strides[0] + k);
        const svuint32_t words = gather_words(pg, base, offsets);
        if constexpr (std::same_as<T, float32_t>) {
          return svreinterpret_f32_u32(words);
        } else if constexpr (std::same_as<T, float16_t>) {
          return svreinterpret_f16_u32(words);
        } else if constexpr (std::same_as<T, bfloat16_t>) {
          return svreinterpret_bf16_u32(words);
        } else if constexpr (std::same_as<T, int8_t>) {
          return svreinterpret_s8_u32(words);
        } else {
          static_assert(std::same_as<T, uint8_t>);
          return svreinterpret_u8_u32(words);
        }
      }
    }
    auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA __arm_streaming {
      const nint_t k = kg * KP + ki;
      auto result = vec::zeros(ScalarTag{});
      if (k < logical_k) {
        // Keep the online-pack path on contiguous K loads. Some production
        // SME compilers still mis-select predicated 64-bit-offset gathers in
        // streaming mode; scalar-width DataAccess calls also preserve every
        // conversion/transform contract without entering that compiler path.
        for (nint_t lane = 0; lane < active; ++lane) {
          const auto one = source.load(
              ScalarTag{}, tensor::coord(spatial + lane, k), tensor::axis<1>,
              vec::opt::first(1), vec::opt::zero);
          result = vec::set(
              ScalarTag{}, result, lane, vec::get(ScalarTag{}, one, 0));
        }
      }
      return result;
    };
    if constexpr (KP == 1) {
      return as_native<T>(load_k(0));
    } else if constexpr (KP == 2) {
      return as_native<T>(generic::interleave_pair<ScalarTag>(
          load_k(0), load_k(1)));
    } else {
      static_assert(KP == 4);
      return as_native<T>(generic::interleave_quad<ScalarTag>(
          load_k(0), load_k(1), load_k(2), load_k(3)));
    }
  }
}

template <gemm::Atom Atom, int Tile, typename VA, typename VB>
VECOPS_ALWAYS_INLINE void mopa(VA a, VB b)
    __arm_streaming __arm_inout("za") {
  const svbool_t pg = []() VECOPS_INLINE_LAMBDA __arm_streaming {
    if constexpr (sizeof(typename Atom::TA) == 1) return svptrue_b8();
    else if constexpr (sizeof(typename Atom::TA) == 2) return svptrue_b16();
    else if constexpr (sizeof(typename Atom::TA) == 4) return svptrue_b32();
    else return svptrue_b64();
  }();
  if constexpr (std::same_as<Atom, gemm::SME_F32F32>) {
    svmopa_za32_f32_m(Tile, pg, pg, a, b);
  } else if constexpr (std::same_as<Atom, gemm::SME_F16F32>) {
    svmopa_za32_f16_m(Tile, pg, pg, a, b);
  } else if constexpr (std::same_as<Atom, gemm::SME_BF16F32>) {
    svmopa_za32_bf16_m(Tile, pg, pg, a, b);
  } else if constexpr (
      std::same_as<typename Atom::TA, int8_t> &&
      std::same_as<typename Atom::TB, int8_t>) {
    svmopa_za32_s8_m(Tile, pg, pg, a, b);
  } else if constexpr (
      std::same_as<typename Atom::TA, uint8_t> &&
      std::same_as<typename Atom::TB, uint8_t>) {
    svmopa_za32_u8_m(Tile, pg, pg, a, b);
  } else if constexpr (
      std::same_as<typename Atom::TA, int8_t> &&
      std::same_as<typename Atom::TB, uint8_t>) {
    svsumopa_za32_s8_m(Tile, pg, pg, a, b);
  } else if constexpr (
      std::same_as<typename Atom::TA, uint8_t> &&
      std::same_as<typename Atom::TB, int8_t>) {
    svusmopa_za32_u8_m(Tile, pg, pg, a, b);
#if defined(HAS_SME_F64F64)
  } else if constexpr (std::same_as<Atom, gemm::SME_F64F64>) {
    svmopa_za64_f64_m(Tile, pg, pg, a, b);
#endif
  } else {
    static_assert(execution::details::dependent_false_v<Atom>,
                  "unsupported SME atom");
  }
}

template <gemm::Atom Atom, gemm::Operand Side, int Block, typename Source>
VECOPS_ALWAYS_INLINE auto packed_block_pointer(
    const Source& source, nint_t origin)
    __arm_streaming __arm_inout("za") {
  static_assert(is_packed_access_v<Atom, Side, Source>);
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t spatial = origin + static_cast<nint_t>(Block) * lanes;
  const nint_t panel = Packing::panel();
  const nint_t offset = tensor::offset_at(
      source.spec().input_layout(), spatial / panel, 0,
      spatial % panel, 0);
  return reinterpret_cast<const T*>(source.raw_data()) + offset;
}

template <gemm::Atom Atom, int NM, int NN, typename A, typename B>
VECOPS_NOINLINE void compute_packed_groups(
    const A& a, const B& b, nint_t m, nint_t n, nint_t logical_k)
    __arm_streaming __arm_inout("za") {
  static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
  static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;
  constexpr nint_t KP = gemm::packing_t<Atom, gemm::Operand::A>::KPack;
  const nint_t groups = ceil_div(logical_k, KP);
  const nint_t a_step = a.raw_strides()[1];
  const nint_t b_step = b.raw_strides()[1];

  if constexpr (NN == 1) {
    const auto* b0 = packed_block_pointer<
        Atom, gemm::Operand::B, 0>(b, n);
    const auto* a0 = packed_block_pointer<
        Atom, gemm::Operand::A, 0>(a, m);
    auto* a1 = a0;
    auto* a2 = a0;
    auto* a3 = a0;
    if constexpr (NM >= 2)
      a1 = packed_block_pointer<Atom, gemm::Operand::A, 1>(a, m);
    if constexpr (NM >= 3)
      a2 = packed_block_pointer<Atom, gemm::Operand::A, 2>(a, m);
    if constexpr (NM >= 4)
      a3 = packed_block_pointer<Atom, gemm::Operand::A, 3>(a, m);
    for (nint_t kg = 0; kg < groups; ++kg) {
      const auto bv = vec::load(BTag{}, b0 + kg * b_step);
      mopa<Atom, 0>(vec::load(ATag{}, a0 + kg * a_step), bv);
      if constexpr (NM >= 2)
        mopa<Atom, 1>(vec::load(ATag{}, a1 + kg * a_step), bv);
      if constexpr (NM >= 3)
        mopa<Atom, 2>(vec::load(ATag{}, a2 + kg * a_step), bv);
      if constexpr (NM >= 4)
        mopa<Atom, 3>(vec::load(ATag{}, a3 + kg * a_step), bv);
    }
  } else if constexpr (NM == 1) {
    const auto* a0 = packed_block_pointer<
        Atom, gemm::Operand::A, 0>(a, m);
    const auto* b0 = packed_block_pointer<
        Atom, gemm::Operand::B, 0>(b, n);
    const auto* b1 = packed_block_pointer<
        Atom, gemm::Operand::B, 1>(b, n);
    auto* b2 = b0;
    auto* b3 = b0;
    if constexpr (NN >= 3)
      b2 = packed_block_pointer<Atom, gemm::Operand::B, 2>(b, n);
    if constexpr (NN >= 4)
      b3 = packed_block_pointer<Atom, gemm::Operand::B, 3>(b, n);
    for (nint_t kg = 0; kg < groups; ++kg) {
      const auto av = vec::load(ATag{}, a0 + kg * a_step);
      mopa<Atom, 0>(av, vec::load(BTag{}, b0 + kg * b_step));
      mopa<Atom, 1>(av, vec::load(BTag{}, b1 + kg * b_step));
      if constexpr (NN >= 3)
        mopa<Atom, 2>(av, vec::load(BTag{}, b2 + kg * b_step));
      if constexpr (NN >= 4)
        mopa<Atom, 3>(av, vec::load(BTag{}, b3 + kg * b_step));
    }
  } else {
    static_assert(NM == 2 && NN == 2);
    const auto* a0 = packed_block_pointer<
        Atom, gemm::Operand::A, 0>(a, m);
    const auto* a1 = packed_block_pointer<
        Atom, gemm::Operand::A, 1>(a, m);
    const auto* b0 = packed_block_pointer<
        Atom, gemm::Operand::B, 0>(b, n);
    const auto* b1 = packed_block_pointer<
        Atom, gemm::Operand::B, 1>(b, n);
    for (nint_t kg = 0; kg < groups; ++kg) {
      const auto av0 = vec::load(ATag{}, a0 + kg * a_step);
      const auto av1 = vec::load(ATag{}, a1 + kg * a_step);
      const auto bv0 = vec::load(BTag{}, b0 + kg * b_step);
      const auto bv1 = vec::load(BTag{}, b1 + kg * b_step);
      mopa<Atom, 0>(av0, bv0);
      mopa<Atom, 1>(av0, bv1);
      mopa<Atom, 2>(av1, bv0);
      mopa<Atom, 3>(av1, bv1);
    }
  }
}

template <int Tile, typename T, vec::VectorValue V>
VECOPS_ALWAYS_INLINE void write_row(uint32_t row, svbool_t pg, V value)
    __arm_streaming __arm_inout("za") {
  if constexpr (std::same_as<T, float32_t>) {
    svwrite_hor_za32_f32_m(Tile, row, pg, value);
  } else if constexpr (std::same_as<T, float64_t>) {
    svwrite_hor_za64_f64_m(Tile, row, pg, value);
  } else {
    svwrite_hor_za32_s32_m(Tile, row, pg, value);
  }
}

template <int Tile, typename T>
VECOPS_ALWAYS_INLINE auto read_row(uint32_t row, svbool_t pg)
    __arm_streaming __arm_inout("za") {
  if constexpr (std::same_as<T, float32_t>) {
    return svread_hor_za32_f32_m(svdup_f32(0.0f), pg, Tile, row);
  } else if constexpr (std::same_as<T, float64_t>) {
    return svread_hor_za64_f64_m(svdup_f64(0.0), pg, Tile, row);
  } else {
    return svread_hor_za32_s32_m(svdup_s32(0), pg, Tile, row);
  }
}

template <int Tile, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    nint_t active_m, nint_t active_n)
    __arm_streaming __arm_inout("za") {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value) {
    return;
  }
  const svbool_t pg = [] (nint_t count)
      VECOPS_INLINE_LAMBDA __arm_streaming {
    if constexpr (sizeof(T) == 8) return svwhilelt_b64(nint_t{0}, count);
    else return svwhilelt_b32(nint_t{0}, count);
  }(active_n);
  if constexpr (direct_row_major_input_v<CInput>) {
    const auto strides = input.raw_strides();
    const auto* base = reinterpret_cast<const T*>(input.raw_data()) +
        m * strides[0] + n;
    for (nint_t row = 0; row < active_m; ++row) {
      if constexpr (sizeof(T) == 8)
        svld1_hor_za64(Tile, static_cast<uint32_t>(row), pg,
                       base + row * strides[0]);
      else
        svld1_hor_za32(Tile, static_cast<uint32_t>(row), pg,
                       base + row * strides[0]);
    }
  } else {
    using Tag = vec::ScalableTag<T, 0>;
    for (nint_t row = 0; row < active_m; ++row) {
      const auto value = input.load(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
          vec::opt::first(active_n), vec::opt::zero);
      write_row<Tile, T>(static_cast<uint32_t>(row), pg, value);
    }
  }
}

template <int Tile, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    nint_t active_m, nint_t active_n)
    __arm_streaming __arm_inout("za") {
  using T = typename COutput::ComputeType;
  const svbool_t pg = [] (nint_t count)
      VECOPS_INLINE_LAMBDA __arm_streaming {
    if constexpr (sizeof(T) == 8) return svwhilelt_b64(nint_t{0}, count);
    else return svwhilelt_b32(nint_t{0}, count);
  }(active_n);
  if constexpr (direct_row_major_output_v<COutput>) {
    const auto strides = output.raw_strides();
    auto* base = reinterpret_cast<T*>(output.raw_data()) +
        m * strides[0] + n;
    for (nint_t row = 0; row < active_m; ++row) {
      if constexpr (sizeof(T) == 8)
        svst1_hor_za64(Tile, static_cast<uint32_t>(row), pg,
                       base + row * strides[0]);
      else
        svst1_hor_za32(Tile, static_cast<uint32_t>(row), pg,
                       base + row * strides[0]);
    }
  } else {
    using Tag = vec::ScalableTag<T, 0>;
    for (nint_t row = 0; row < active_m; ++row) {
      const auto value = static_cast<vec::Vec<Tag>>(
          read_row<Tile, T>(static_cast<uint32_t>(row), pg));
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n));
    }
  }
}

template <gemm::Atom Atom, int NM, int NN,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_group(
    const A& a, const B& b, nint_t m, nint_t n, nint_t kg,
    nint_t logical_m, nint_t logical_n, nint_t logical_k)
    __arm_streaming __arm_inout("za") {
  if constexpr (NN == 1) {
    const auto bv = load_operand<Atom, gemm::Operand::B, 0>(
        b, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA
        __arm_streaming __arm_inout("za") {
      const auto av = load_operand<Atom, gemm::Operand::A, MI>(
          a, m, kg, logical_m, logical_k);
      mopa<Atom, MI>(av, bv);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>)
        VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else if constexpr (NM == 1) {
    const auto av = load_operand<Atom, gemm::Operand::A, 0>(
        a, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA
        __arm_streaming __arm_inout("za") {
      const auto bv = load_operand<Atom, gemm::Operand::B, NI>(
          b, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(av, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>)
        VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  } else {
    static_assert(NM == 2 && NN == 2);
    const auto a0 = load_operand<Atom, gemm::Operand::A, 0>(
        a, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<Atom, gemm::Operand::A, 1>(
        a, m, kg, logical_m, logical_k);
    const auto b0 = load_operand<Atom, gemm::Operand::B, 0>(
        b, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<Atom, gemm::Operand::B, 1>(
        b, n, kg, logical_n, logical_k);
    mopa<Atom, 0>(a0, b0);
    mopa<Atom, 1>(a0, b1);
    mopa<Atom, 2>(a1, b0);
    mopa<Atom, 3>(a1, b1);
  }
}

template <gemm::Atom Atom, bool FastPacked, typename Case,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n)
    __arm_streaming __arm_inout("za") {
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  const nint_t lanes = accumulator_lanes<Atom>();
  svzero_za();
  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
    (initialize_c_tile<static_cast<int>(I)>(
         c_input,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         std::clamp(active_m - static_cast<nint_t>(I / NN) * lanes,
                    nint_t{0}, lanes),
         std::clamp(active_n - static_cast<nint_t>(I % NN) * lanes,
                    nint_t{0}, lanes)), ...);
  }(std::make_index_sequence<Outputs>{});

  if constexpr (FastPacked) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    compute_packed_groups<Atom, NM, NN>(a, b, m, n, logical_k);
  } else {
    constexpr nint_t KP = gemm::packing_t<Atom, gemm::Operand::A>::KPack;
    const nint_t groups = ceil_div(logical_k, KP);
    for (nint_t kg = 0; kg < groups; ++kg) {
      compute_group<Atom, NM, NN>(
          a, b, m, n, kg, logical_m, logical_n, logical_k);
    }
  }

  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
    (store_c_tile<static_cast<int>(I)>(
         c_output,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         std::clamp(active_m - static_cast<nint_t>(I / NN) * lanes,
                    nint_t{0}, lanes),
         std::clamp(active_n - static_cast<nint_t>(I % NN) * lanes,
                    nint_t{0}, lanes)), ...);
  }(std::make_index_sequence<Outputs>{});
}

template <gemm::Atom Atom, bool FastPacked = false,
          typename A, typename B, typename CInput, typename COutput>
__arm_new("za") VECOPS_NOINLINE void run_tiles(
    nint_t m, nint_t n, nint_t k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output)
    __arm_streaming {
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t mb = ceil_div(m, lanes);
  const nint_t nb = ceil_div(n, lanes);
  auto invoke = [&]<int NM, int NN>(nint_t bm, nint_t bn)
      VECOPS_INLINE_LAMBDA __arm_streaming __arm_inout("za") {
    using Family = tile::Tile2DKernelFamily<NM, NN, 1>;
    using Case = tile::Tile2DKernelCase<
        Family, tile::Tile2DMaskMode::masked, tile::Tile2DMaskMode::masked>;
    const nint_t mi = bm * lanes;
    const nint_t ni = bn * lanes;
    microkernel<Atom, FastPacked, Case>(
        a, b, c_input, c_output, m, n, k, mi, ni,
        std::min(m - mi, static_cast<nint_t>(NM) * lanes),
        std::min(n - ni, static_cast<nint_t>(NN) * lanes));
  };

  if (nb == 1) {
    nint_t bm = 0;
    for (; bm + 4 <= mb; bm += 4) invoke.template operator()<4, 1>(bm, 0);
    if (mb - bm == 3) invoke.template operator()<3, 1>(bm, 0);
    else if (mb - bm == 2) invoke.template operator()<2, 1>(bm, 0);
    else if (mb - bm == 1) invoke.template operator()<1, 1>(bm, 0);
    return;
  }

  nint_t bm = 0;
  for (; bm + 2 <= mb; bm += 2) {
    nint_t bn = 0;
    for (; bn + 2 <= nb; bn += 2) invoke.template operator()<2, 2>(bm, bn);
    if (bn < nb) invoke.template operator()<2, 1>(bm, bn);
  }
  if (bm < mb) {
    nint_t bn = 0;
    for (; bn + 4 <= nb; bn += 4) invoke.template operator()<1, 4>(bm, bn);
    if (nb - bn == 3) invoke.template operator()<1, 3>(bm, bn);
    else if (nb - bn == 2) invoke.template operator()<1, 2>(bm, bn);
    else if (nb - bn == 1) invoke.template operator()<1, 1>(bm, bn);
  }
}

} // namespace vecops::kernel::matmul_details::sme

namespace vecops::kernel::matmul_details {

template <>
struct Backend<matmul_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::Streaming>;

  static nint_t scratch_bytes() { return 0; }

  template <gemm::Atom Atom, execution::ExecutionScope Scope,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, nint_t m, nint_t n, nint_t k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void*) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::SMEKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::arm::Streaming, Scope>);
    if constexpr (
        sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
        sme::is_packed_access_v<Atom, gemm::Operand::B, B>) {
      constexpr nint_t KP = gemm::packing_t<
          Atom, gemm::Operand::A>::KPack;
      if (ceil_div(k, KP) >= 32)
        sme::run_tiles<Atom, true>(m, n, k, a, b, c_input, c_output);
      else
        sme::run_tiles<Atom, false>(m, n, k, a, b, c_input, c_output);
    } else {
      sme::run_tiles<Atom, false>(m, n, k, a, b, c_input, c_output);
    }
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_SME_BACKEND_H
