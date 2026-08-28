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

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/gemm/Packing.h"
#include "vecops/gemm/details/sme/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/matmul/Traversal.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/details/sme/ZA.h"

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
    if constexpr (A * B <= 4) {
      constexpr int imbalance = A > B ? A - B : B - A;
      return 100 * A * B + 4 * (A + B) - 8 * imbalance;
    } else {
      return -1;
    }
  }
};

using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider, tile::Tile2DSearchSpace<4, 4, 4>>;

template <gemm::Atom Atom, meta::ValueType M, meta::ValueType N>
inline constexpr bool use_constraint_area4_v =
    tile::tile2d_details::has_max_block_count_v<
        M, decltype(Atom::M_R)> ||
    tile::tile2d_details::has_max_block_count_v<
        N, decltype(Atom::N_R)>;

template <gemm::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B>
inline constexpr bool prefer_constraint_area4_v = [] {
  if constexpr (!use_constraint_area4_v<Atom, M, N>) {
    return false;
  } else if constexpr (
      is_packed_access_v<Atom, gemm::Operand::A, A> &&
      is_packed_access_v<Atom, gemm::Operand::B, B>) {
    constexpr bool FixedSquare =
        tile::tile2d_details::fixed_value_v<M> &&
        tile::tile2d_details::fixed_value_v<N> &&
        tile::tile2d_details::fixed_value_n<M> ==
            tile::tile2d_details::fixed_value_n<N>;
    constexpr nint_t KP = gemm::packing_t<
        Atom, gemm::Operand::A>::KPack;
    constexpr bool ShortK = meta::has_upper_bound_v<K> &&
        meta::upper_bound_v<K> < 32 * KP;
    // Very short packed kernels are dominated by entry/code-layout effects,
    // while fixed square packed kernels currently receive better register
    // allocation from the compact RuntimeExact loop.  Preserve those proven
    // winners; constrained raw, rectangular, and long-K packed problems use
    // the statically pruned plan.
    return !FixedSquare && !ShortK;
  } else {
    return true;
  }
}();

consteval int log2_kpack(nint_t kpack) {
  return kpack == 4 ? 2 : (kpack == 2 ? 1 : 0);
}

template <gemm::Atom Atom>
VECOPS_ALWAYS_INLINE nint_t accumulator_lanes() {
  return static_cast<nint_t>(Atom::M_R);
}

template <typename T, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto as_native(V value) {
  using Tag = vec::ScalableTag<T, 0>;
  return static_cast<vec::Vec<Tag>>(value);
}

template <gemm::Atom Atom, gemm::Operand Side, int Block,
          bool FullK, typename Source>
VECOPS_ALWAYS_INLINE auto load_operand(
    const Source& source, nint_t origin, nint_t kg,
    nint_t logical_spatial, nint_t logical_k) {
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
    const nint_t panel = static_cast<nint_t>(Packing::panel());
    const nint_t offset = tensor::offset_at(
        layout, spatial / panel, kg, spatial % panel, 0);
    return vec::load(
        NativeTag{}, reinterpret_cast<const T*>(source.raw_data()) + offset);
  } else {
    static_assert(Source::Rank == 2, "unpacked SME input must be rank two");
    const nint_t active = vec::details::sme::clamp_value(
        logical_spatial - spatial, nint_t{0}, lanes);
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      const auto strides = source.raw_strides();
      const nint_t k = kg * KP;
      const nint_t row_bytes = strides[0] * static_cast<nint_t>(sizeof(T));
      if ((FullK || k + KP <= logical_k) &&
          row_bytes >= std::numeric_limits<int32_t>::min() &&
          row_bytes <= std::numeric_limits<int32_t>::max()) {
        using WordTag = vec::ScalableTag<uint32_t, 0>;
        using Resources = typename std::remove_cvref_t<
            decltype(source.policy())>::ActiveResources;
        const auto* base = reinterpret_cast<const uint32_t*>(
            source.raw_data() + spatial * strides[0] + k);
        const auto words = vec::load(
            WordTag{}, base, vec::opt::first(active),
            vec::strided(row_bytes, vec::scale<1>),
            vec::resources<Resources>);
        return vec::bitcast(NativeTag{}, WordTag{}, words);
      }
    }
    auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
      const nint_t k = kg * KP + ki;
      if constexpr (FullK) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
            vec::opt::first(active), vec::opt::zero);
      } else if (k < logical_k) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
            vec::opt::first(active), vec::opt::zero);
      }
      return vec::zeros(ScalarTag{});
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
VECOPS_ALWAYS_INLINE void mopa(VA a, VB b) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using TAcc = typename Atom::TAcc;
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;
  const auto pga = vec::mtrue(ATag{});
  const auto pgb = vec::mtrue(BTag{});
  vec::details::sme::mopa<Tile, TAcc, TA, TB>(
      pga, pgb, static_cast<vec::Vec<ATag>>(a),
      static_cast<vec::Vec<BTag>>(b));
}

template <gemm::Atom Atom, gemm::Operand Side, int Block, typename Source>
VECOPS_ALWAYS_INLINE auto packed_block_pointer(
    const Source& source, nint_t origin) {
  static_assert(is_packed_access_v<Atom, Side, Source>);
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t spatial = origin + static_cast<nint_t>(Block) * lanes;
  const nint_t panel = static_cast<nint_t>(Packing::panel());
  const nint_t offset = tensor::offset_at(
      source.spec().input_layout(), spatial / panel, 0,
      spatial % panel, 0);
  return reinterpret_cast<const T*>(source.raw_data()) + offset;
}

template <gemm::Atom Atom, int NM, int NN, typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_packed_groups(
    const A& a, const B& b, nint_t m, nint_t n, nint_t logical_k) {
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
    const auto* b0 = packed_block_pointer<Atom, gemm::Operand::B, 0>(b, n);
    const auto* a0 = packed_block_pointer<Atom, gemm::Operand::A, 0>(a, m);
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
    const auto* a0 = packed_block_pointer<Atom, gemm::Operand::A, 0>(a, m);
    const auto* b0 = packed_block_pointer<Atom, gemm::Operand::B, 0>(b, n);
    const auto* b1 = packed_block_pointer<Atom, gemm::Operand::B, 1>(b, n);
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
    const auto* a0 = packed_block_pointer<Atom, gemm::Operand::A, 0>(a, m);
    const auto* a1 = packed_block_pointer<Atom, gemm::Operand::A, 1>(a, m);
    const auto* b0 = packed_block_pointer<Atom, gemm::Operand::B, 0>(b, n);
    const auto* b1 = packed_block_pointer<Atom, gemm::Operand::B, 1>(b, n);
    VECOPS_LOOP_ALIGN(64) for (nint_t kg = 0; kg < groups; ++kg) {
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

template <int Tile, typename T, vec::VectorValue V, typename Mask>
VECOPS_ALWAYS_INLINE void write_row(
    uint32_t row, Mask pg, V value) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  vec::details::sme::write_hor<Tile>(
      row, static_cast<vec::Mask<Tag>>(pg),
      static_cast<vec::Vec<Tag>>(value));
}

template <int Tile, typename T, typename Mask>
VECOPS_ALWAYS_INLINE auto read_row(
    uint32_t row, Mask pg) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  return vec::details::sme::read_hor<Tile>(
      Tag{}, row, static_cast<vec::Mask<Tag>>(pg));
}

template <int Tile, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    nint_t active_m, nint_t active_n) {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value) {
    return;
  }
  using Tag = vec::ScalableTag<T, 0>;
  const auto pg = vec::mwhilelt(Tag{}, nint_t{0}, active_n);
  if constexpr (direct_row_major_input_v<CInput>) {
    const auto strides = input.raw_strides();
    const auto* base = reinterpret_cast<const T*>(input.raw_data()) + m * strides[0] + n;
    for (nint_t row = 0; row < active_m; ++row) {
      using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      using BitsTag = vec::ScalableTag<U, 0>;
      vec::details::sme::load_hor<Tile>(
          static_cast<uint32_t>(row),
          static_cast<vec::Mask<BitsTag>>(pg),
          reinterpret_cast<const U*>(base + row * strides[0]));
    }
  } else {
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
    nint_t active_m, nint_t active_n) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  const auto pg = vec::mwhilelt(Tag{}, nint_t{0}, active_n);
  if constexpr (direct_row_major_output_v<COutput>) {
    const auto strides = output.raw_strides();
    auto* base = reinterpret_cast<T*>(output.raw_data()) + m * strides[0] + n;
    for (nint_t row = 0; row < active_m; ++row) {
      using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      using BitsTag = vec::ScalableTag<U, 0>;
      vec::details::sme::store_hor<Tile>(
          static_cast<uint32_t>(row),
          static_cast<vec::Mask<BitsTag>>(pg),
          reinterpret_cast<U*>(base + row * strides[0]));
    }
  } else {
    for (nint_t row = 0; row < active_m; ++row) {
      const auto value = static_cast<vec::Vec<Tag>>(
          read_row<Tile, T>(static_cast<uint32_t>(row), pg));
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n));
    }
  }
}

template <gemm::Atom Atom, int NM, int NN, bool FullK,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_group(
    const A& a, const B& b, nint_t m, nint_t n, nint_t kg,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  if constexpr (NN == 1) {
    const auto bv = load_operand<Atom, gemm::Operand::B, 0, FullK>(
        b, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<Atom, gemm::Operand::A, MI, FullK>(
          a, m, kg, logical_m, logical_k);
      mopa<Atom, MI>(av, bv);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else if constexpr (NM == 1) {
    const auto av = load_operand<Atom, gemm::Operand::A, 0, FullK>(
        a, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<Atom, gemm::Operand::B, NI, FullK>(
          b, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(av, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  } else {
    static_assert(NM == 2 && NN == 2);
    const auto a0 = load_operand<Atom, gemm::Operand::A, 0, FullK>(
        a, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<Atom, gemm::Operand::A, 1, FullK>(
        a, m, kg, logical_m, logical_k);
    const auto b0 = load_operand<Atom, gemm::Operand::B, 0, FullK>(
        b, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<Atom, gemm::Operand::B, 1, FullK>(
        b, n, kg, logical_n, logical_k);
    mopa<Atom, 0>(a0, b0);
    mopa<Atom, 1>(a0, b1);
    mopa<Atom, 2>(a1, b0);
    mopa<Atom, 3>(a1, b1);
  }
}

template <gemm::Atom Atom, bool FastPacked, int NM, int NN,
          bool ExactBlocks,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  constexpr int Outputs = NM * NN;
  const nint_t lanes = static_cast<nint_t>(Atom::M_R);
  vec::details::sme::zero_za();
  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA {
    (initialize_c_tile<static_cast<int>(I)>(
         c_input,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         vec::details::sme::clamp_value(active_m - static_cast<nint_t>(I / NN) * lanes,
                    nint_t{0}, lanes),
         vec::details::sme::clamp_value(active_n - static_cast<nint_t>(I % NN) * lanes,
                    nint_t{0}, lanes)), ...);
  }(std::make_index_sequence<Outputs>{});

  auto compute_generic = [&]() VECOPS_INLINE_LAMBDA {
    constexpr nint_t KP = gemm::packing_t<Atom, gemm::Operand::A>::KPack;
    if constexpr (KP == 1 || std::same_as<Atom, gemm::SME_BF16F32>) {
      const nint_t full_groups = logical_k / KP;
      for (nint_t kg = 0; kg < full_groups; ++kg) {
        compute_group<Atom, NM, NN, true>(
            a, b, m, n, kg, logical_m, logical_n, logical_k);
      }
      if constexpr (KP > 1) {
        if (full_groups * KP < logical_k) {
          compute_group<Atom, NM, NN, false>(
              a, b, m, n, full_groups,
              logical_m, logical_n, logical_k);
        }
      }
    } else {
      const nint_t groups = ceil_div(logical_k, KP);
      for (nint_t kg = 0; kg < groups; ++kg) {
        compute_group<Atom, NM, NN, false>(
            a, b, m, n, kg, logical_m, logical_n, logical_k);
      }
    }
  };

  if constexpr (FastPacked && ExactBlocks) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    compute_packed_groups<Atom, NM, NN>(a, b, m, n, logical_k);
  } else if constexpr (FastPacked) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    const bool all_logical_blocks_exist =
        active_m > static_cast<nint_t>(NM - 1) * lanes &&
        active_n > static_cast<nint_t>(NN - 1) * lanes;
    if (all_logical_blocks_exist) {
      compute_packed_groups<Atom, NM, NN>(a, b, m, n, logical_k);
    } else {
      compute_generic();
    }
  } else {
    compute_generic();
  }

  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA {
    (store_c_tile<static_cast<int>(I)>(
         c_output,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         vec::details::sme::clamp_value(active_m - static_cast<nint_t>(I / NN) * lanes,
                    nint_t{0}, lanes),
         vec::details::sme::clamp_value(active_n - static_cast<nint_t>(I % NN) * lanes,
                    nint_t{0}, lanes)), ...);
  }(std::make_index_sequence<Outputs>{});
}

} // namespace vecops::kernel::matmul_details::sme

namespace vecops::kernel::matmul_details {

template <>
struct Backend<matmul_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;
  using Catalog = sme::Catalog;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 0; }

  template <gemm::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B>
  using EffectivePolicy = std::conditional_t<
      std::same_as<Policy, matmul_policy::Automatic>,
      std::conditional_t<
          sme::prefer_constraint_area4_v<Atom, M, N, K, A, B>,
          kernel::loop::tile2d_policy::FourRegions,
          kernel::loop::tile2d_policy::RuntimeExactArea4>,
      Policy>;

  template <gemm::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::SMEKernelKind>);
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (
              sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
              sme::is_packed_access_v<Atom, gemm::Operand::B, B>) {
            matmul_details::run_tiles<Backend, Atom, Policy>(
                m, n, k, a, b, c_input, c_output, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            matmul_details::run_tiles<Backend, Atom, Policy>(
                m, n, k, active_a, active_b,
                active_c_input, active_c_output, scratch);
            active_c_output.commit();
          }
        });
  }

  template <gemm::Atom Atom, typename A, typename B,
            meta::ValueType K, typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(K k, Fn&& fn) {
    auto invoke = [&]<bool FastPacked>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
      std::forward<Fn>(fn).template operator()<
          std::bool_constant<FastPacked>>();
    };
    if constexpr (
        sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
        sme::is_packed_access_v<Atom, gemm::Operand::B, B>) {
      constexpr nint_t KP = gemm::packing_t<
          Atom, gemm::Operand::A>::KPack;
      const auto k_groups = gemm::packing_details::ceil_div_value(
          k, meta::cint<KP>);
      if constexpr (decltype(k_groups)::is_const) {
        invoke.template operator()<decltype(k_groups)::value >= 32>();
      } else if constexpr (
          meta::has_upper_bound_v<decltype(k_groups)> &&
          meta::upper_bound_v<decltype(k_groups)> < 32) {
        invoke.template operator()<false>();
      } else if constexpr (
          meta::has_lower_bound_v<decltype(k_groups)> &&
          meta::lower_bound_v<decltype(k_groups)> >= 32) {
        invoke.template operator()<true>();
      } else {
        // A single direct packed-pointer loop is faster overall for an
        // unconstrained K and avoids cloning the complete tile traversal.
        invoke.template operator()<true>();
      }
    } else {
      invoke.template operator()<false>();
    }
  }

  template <gemm::Atom Atom, typename Case, typename Plan,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_case(
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void*) {
    sme::microkernel<
        Atom, Plan::value, Case::a, Case::b, Case::exact_blocks>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k,
            m, n, active_m, active_n);
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_SME_BACKEND_H
