//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_SME_BACKEND_H
#define VECOPS_MATMUL_DETAILS_SME_BACKEND_H

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/sme/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/matmul/details/Traversal.h"
#include "vecops/matmul/details/sme/RuntimeQuantInt8.h"
#include "vecops/matmul/details/pack/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"
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

template <bool ExpandedCatalog>
struct KernelProvider {
  static constexpr bool four_regions_exact_constraints = true;
  static constexpr tile::Tile2DExactGridMode exact_grid_mode = ExpandedCatalog
      ? tile::Tile2DExactGridMode::exact
      : tile::Tile2DExactGridMode::runtime;
  static constexpr nint_t exact_meta_block_limit = ExpandedCatalog
      ? std::numeric_limits<nint_t>::max() : nint_t{8};

  template <int A, int B, tile::Tile2DMaskMode, tile::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B <= 4) {
      constexpr int imbalance = A > B ? A - B : B - A;
      return 100 * A * B + 4 * (A + B) - 8 * imbalance;
    } else if constexpr (ExpandedCatalog && A * B <= 8) {
      // Only ZA64 paths expose these families to Tile2D. Their lower score
      // keeps the compact catalog family as the default for other policies;
      // ExactCover detects their presence directly from the catalog.
      constexpr int imbalance = A > B ? A - B : B - A;
      return 300 + 4 * A * B + (A + B) - 2 * imbalance;
    } else {
      return -1;
    }
  }
};

template <bool ExpandedCatalog>
using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider<ExpandedCatalog>,
    tile::Tile2DSearchSpace<4, 4, ExpandedCatalog ? 8 : 4>>;

template <gemm::Atom Atom, typename A, typename B>
inline constexpr bool use_expanded_catalog_v = [] {
#if defined(HAS_SME_F64F64)
  return std::same_as<Atom, gemm::SME_F64F64> &&
      !(is_packed_access_v<Atom, gemm::Operand::A, A> &&
        is_packed_access_v<Atom, gemm::Operand::B, B>);
#else
  return false;
#endif
}();

template <bool FastPacked, bool PrefetchLargeWorkingSet>
struct KernelPlan : std::bool_constant<FastPacked> {
  static constexpr bool prefetch_large_working_set =
      PrefetchLargeWorkingSet;
};

template <gemm::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
inline constexpr bool large_packed_prefetch_v = [] {
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (std::same_as<Atom, gemm::SME_BF16F32> &&
                meta::lower_bound_at_least_v<MV, 1> &&
                meta::lower_bound_at_least_v<NV, 1> &&
                meta::lower_bound_at_least_v<KV, 1>) {
    constexpr nint_t LogicalM = meta::lower_bound_v<MV>;
    constexpr nint_t LogicalN = meta::lower_bound_v<NV>;
    constexpr nint_t LogicalK = meta::lower_bound_v<KV>;
    constexpr uint64_t MinPackedBytes = uint64_t{2} * 1024 * 1024;
    constexpr uint64_t BytesPerK =
        static_cast<uint64_t>(LogicalM) * sizeof(typename Atom::TA) +
        static_cast<uint64_t>(LogicalN) * sizeof(typename Atom::TB);
    constexpr uint64_t MinBytesPerK =
        (MinPackedBytes + static_cast<uint64_t>(LogicalK) - 1) /
        static_cast<uint64_t>(LogicalK);
    return BytesPerK >= MinBytesPerK;
  }
#endif
  return false;
}();

template <gemm::Atom Atom, meta::ValueType M, meta::ValueType N>
inline constexpr bool has_bounded_tile_axis_v =
    tile::tile2d_details::has_max_block_count_v<
        M, decltype(Atom::M_R)> ||
    tile::tile2d_details::has_max_block_count_v<
        N, decltype(Atom::N_R)>;

template <gemm::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B>
inline constexpr bool prefer_constraint_pruning_v = [] {
  if constexpr (!has_bounded_tile_axis_v<Atom, M, N>) {
    return false;
  } else if constexpr (
      is_packed_access_v<Atom, gemm::Operand::A, A> &&
      is_packed_access_v<Atom, gemm::Operand::B, B>) {
    constexpr bool FixedSquare =
        meta::is_singleton_v<M> &&
        meta::is_singleton_v<N> &&
        meta::singleton_value_v<M> ==
            meta::singleton_value_v<N>;
    constexpr nint_t KP = gemm::packing_t<
        Atom, gemm::Operand::A>::KPack;
    constexpr bool ShortK = meta::has_upper_bound_v<K> &&
        meta::upper_bound_v<K> < 32 * KP;
    // Very short packed kernels are dominated by entry/code-layout effects,
    // while fixed square packed kernels currently receive better register
    // allocation from the compact runtime ExactCover loop. Preserve those proven
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

template <gemm::Atom Atom, gemm::Operand Side, typename Source>
struct OperandInvariants {
  nint_t row_bytes = 0;
  bool gather_offsets_fit = false;

  VECOPS_ALWAYS_INLINE explicit OperandInvariants(const Source& source) {
    using T = typename gemm::packing_t<Atom, Side>::Element;
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      row_bytes = static_cast<nint_t>(tensor::stride_value<0>(
          source.spec().input_layout())) *
          static_cast<nint_t>(sizeof(T));
      gather_offsets_fit =
          row_bytes >= std::numeric_limits<int32_t>::min() &&
          row_bytes <= std::numeric_limits<int32_t>::max();
    }
  }
};

template <typename T, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto as_native(V value) {
  using Tag = vec::ScalableTag<T, 0>;
  return static_cast<vec::Vec<Tag>>(value);
}

template <gemm::Atom Atom, gemm::Operand Side, int Block,
          bool FullSpatial, bool FullK, typename Source>
VECOPS_ALWAYS_INLINE auto load_operand(
    const Source& source,
    const OperandInvariants<Atom, Side, Source>& invariants,
    nint_t origin, nint_t kg,
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
    const nint_t active = FullSpatial
        ? lanes
        : vecops::clamp(
              logical_spatial - spatial, nint_t{0}, lanes);
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      const auto strides = source.raw_strides();
      const nint_t k = kg * KP;
      if ((FullK || k + KP <= logical_k) &&
          invariants.gather_offsets_fit) {
        using WordTag = vec::ScalableTag<uint32_t, 0>;
        using Resources = typename std::remove_cvref_t<
            decltype(source.policy())>::ActiveResources;
        const auto* base = reinterpret_cast<const uint32_t*>(
            source.raw_data() + spatial * strides[0] + k);
        const auto words = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (FullSpatial) {
            return vec::load(
                WordTag{}, base,
                vec::strided(invariants.row_bytes, vec::scale<1>),
                vec::resources<Resources>);
          } else {
            return vec::load(
                WordTag{}, base, vec::opt::first(active),
                vec::strided(invariants.row_bytes, vec::scale<1>),
                vec::resources<Resources>);
          }
        }();
        return vec::bitcast(NativeTag{}, WordTag{}, words);
      }
    }
    auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
      const nint_t k = kg * KP + ki;
      if constexpr (FullK && FullSpatial) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>);
      } else if constexpr (FullK) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
            vec::opt::first(active), vec::opt::zero);
      } else if (k < logical_k) {
        if constexpr (FullSpatial) {
          return source.load(
              ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>);
        } else {
          return source.load(
              ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
              vec::opt::first(active), vec::opt::zero);
        }
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

#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
template <typename ToTag, typename FromTag, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto skinny_convert(ToTag to, FromTag from, V value) {
  if constexpr (std::same_as<ToTag, FromTag>) return value;
  else return vec::convert(to, from, value);
}

// TODO: Generalize this into a public vec operation only after the portable
// semantics and additional backend use cases have stabilized.
template <typename InputTag>
VECOPS_ALWAYS_INLINE auto skinny_widening_dot_add(
    InputTag, vec::Vec<vec::ViewAs<float32_t, InputTag>> acc,
    vec::Vec<InputTag> a, vec::Vec<InputTag> b) {
  using AccTag = vec::ViewAs<float32_t, InputTag>;
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16)
  const auto raw_acc = vec::details::sve_basic_raw_word(acc);
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  return vec::details::sve_basic_wrap_word<AccTag>(
      svbfdot_f32(raw_acc, raw_a, raw_b));
#else
  const auto a0 = vec::convert(AccTag{}, InputTag{}, a, vec::cvt::lane<0>);
  const auto a1 = vec::convert(AccTag{}, InputTag{}, a, vec::cvt::lane<1>);
  const auto b0 = vec::convert(AccTag{}, InputTag{}, b, vec::cvt::lane<0>);
  const auto b1 = vec::convert(AccTag{}, InputTag{}, b, vec::cvt::lane<1>);
  return vec::fmadd(a1, b1, vec::fmadd(a0, b0, acc));
#endif
}

// TODO: Generalize together with skinny_widening_dot_add after the FP16
// FMLAL path has passed all performance and numerical gates.
template <typename InputTag>
VECOPS_ALWAYS_INLINE auto skinny_widening_fmadd(
    InputTag, vec::Vec<vec::Rebind<float32_t, InputTag>> acc,
    vec::Vec<InputTag> a, vec::Vec<InputTag> b) {
  using AccTag = vec::Rebind<float32_t, InputTag>;
#if defined(CPU_CAPABILITY_SVE) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  return vec::details::construct_words<vec::details::SVEBackend>(
      AccTag{}, [&]<nint_t Index>(AccTag) {
        static_assert(Index == 0 || Index == 1);
        const auto raw_acc = vec::details::sve_basic_raw_word(
            vec::get_word<Index>(AccTag{}, acc));
        if constexpr (Index == 0) {
          return vec::details::sve_basic_wrap_word<AccTag>(
              svmlalb_f32(raw_acc, raw_a, raw_b));
        } else {
          return vec::details::sve_basic_wrap_word<AccTag>(
              svmlalt_f32(raw_acc, raw_a, raw_b));
        }
      });
#else
  return vec::fmadd(
      AccTag{}, vec::convert(AccTag{}, InputTag{}, a),
      vec::convert(AccTag{}, InputTag{}, b), acc);
#endif
}

// TODO: Generalize only after signed, unsigned, and mixed-sign dot semantics
// have a stable cross-backend API. The first SME-local use covers same-sign
// raw skinny matmuls, whose contiguous K layout maps directly to SVE DOT.
// BFMMLA/I8MM MMLA need different 2x4x2/2x8x2 segment layouts and therefore
// belong in a separate tiny-kernel packing experiment, not in this raw GEMV.
template <typename InputTag>
VECOPS_ALWAYS_INLINE auto skinny_integer_dot_add(
    InputTag, vec::Vec<vec::ViewAs<int32_t, InputTag>> acc,
    vec::Vec<InputTag> a, vec::Vec<InputTag> b) {
  using T = vec::ElementOf<InputTag>;
  using AccTag = vec::ViewAs<int32_t, InputTag>;
  const auto raw_acc = vec::details::sve_basic_raw_word(acc);
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  if constexpr (std::same_as<T, int8_t>) {
    return vec::details::sve_basic_wrap_word<AccTag>(
        svdot_s32(raw_acc, raw_a, raw_b));
  } else {
    static_assert(std::same_as<T, uint8_t>);
    return vec::details::sve_basic_wrap_word<AccTag>(
        svreinterpret_s32_u32(svdot_u32(
            svreinterpret_u32_s32(raw_acc), raw_a, raw_b)));
  }
}

template <bool VaryRows, int Block,
          typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void sve_skinny_block(
    const A& a, const B& b, COutput& c_output,
    nint_t output_origin, nint_t logical_k) {
  using TA = typename A::ComputeType;
  using TB = typename B::ComputeType;
  using Acc = typename COutput::ComputeType;
  static_assert(std::same_as<TA, TB>);
  using InputTag = vec::ScalableTag<TA, 0>;
  using AccTag = std::conditional_t<
      std::same_as<TA, bfloat16_t>,
      vec::ViewAs<float32_t, InputTag>,
      std::conditional_t<
          std::is_integral_v<TA>, vec::ViewAs<int32_t, InputTag>,
          vec::Rebind<Acc, InputTag>>>;
  auto sum0 = vec::zeros(AccTag{});
  auto sum1 = vec::zeros(AccTag{});
  auto sum2 = vec::zeros(AccTag{});
  auto sum3 = vec::zeros(AccTag{});
  auto sum4 = vec::zeros(AccTag{});
  auto sum5 = vec::zeros(AccTag{});
  auto sum6 = vec::zeros(AccTag{});
  auto sum7 = vec::zeros(AccTag{});

  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const auto* a_data = reinterpret_cast<const TA*>(a.raw_data());
  const auto* b_data = reinterpret_cast<const TB*>(b.raw_data());
  nint_t kk = 0;
  nint_t remaining = logical_k;
  while (remaining > 0) {
    const nint_t active = vecops::min(vec::size(InputTag{}), remaining);
    const auto load_row = [&](const auto* data, nint_t row, nint_t stride) {
      return vec::load(
          InputTag{}, data + row * stride + kk,
          vec::opt::first(active), vec::opt::zero);
    };
    const auto madd = [&](auto lhs, auto rhs, auto sum) {
      if constexpr (std::same_as<TA, bfloat16_t>) {
        return skinny_widening_dot_add(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::same_as<TA, float16_t>) {
        return skinny_widening_fmadd(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::is_integral_v<TA>) {
        return skinny_integer_dot_add(InputTag{}, sum, lhs, rhs);
      } else {
        return vec::fmadd(
            skinny_convert(AccTag{}, InputTag{}, lhs),
            skinny_convert(AccTag{}, InputTag{}, rhs), sum);
      }
    };
    if constexpr (VaryRows) {
      const auto shared = load_row(b_data, 0, b_strides[0]);
      sum0 = madd(
          load_row(a_data, output_origin, a_strides[0]), shared, sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            load_row(a_data, output_origin + 1, a_strides[0]), shared, sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            load_row(a_data, output_origin + 2, a_strides[0]), shared, sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            load_row(a_data, output_origin + 3, a_strides[0]), shared, sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            load_row(a_data, output_origin + 4, a_strides[0]), shared, sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            load_row(a_data, output_origin + 5, a_strides[0]), shared, sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            load_row(a_data, output_origin + 6, a_strides[0]), shared, sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            load_row(a_data, output_origin + 7, a_strides[0]), shared, sum7);
    } else {
      const auto shared = load_row(a_data, 0, a_strides[0]);
      sum0 = madd(
          shared, load_row(b_data, output_origin, b_strides[0]), sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            shared, load_row(b_data, output_origin + 1, b_strides[0]), sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            shared, load_row(b_data, output_origin + 2, b_strides[0]), sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            shared, load_row(b_data, output_origin + 3, b_strides[0]), sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            shared, load_row(b_data, output_origin + 4, b_strides[0]), sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            shared, load_row(b_data, output_origin + 5, b_strides[0]), sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            shared, load_row(b_data, output_origin + 6, b_strides[0]), sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            shared, load_row(b_data, output_origin + 7, b_strides[0]), sum7);
    }
    kk += active;
    remaining -= active;
  }

  auto* output = reinterpret_cast<Acc*>(c_output.raw_data());
  const nint_t output_stride = static_cast<nint_t>(
      tensor::stride_value<0>(c_output.spec().output_layout()));
  const auto store = [&](nint_t logical, Acc value) {
    output[VaryRows ? logical * output_stride : logical] = value;
  };
  store(output_origin, vec::reduce_add(AccTag{}, sum0));
  if constexpr (Block >= 2)
    store(output_origin + 1, vec::reduce_add(AccTag{}, sum1));
  if constexpr (Block >= 3)
    store(output_origin + 2, vec::reduce_add(AccTag{}, sum2));
  if constexpr (Block >= 4)
    store(output_origin + 3, vec::reduce_add(AccTag{}, sum3));
  if constexpr (Block >= 5)
    store(output_origin + 4, vec::reduce_add(AccTag{}, sum4));
  if constexpr (Block >= 6)
    store(output_origin + 5, vec::reduce_add(AccTag{}, sum5));
  if constexpr (Block >= 7)
    store(output_origin + 6, vec::reduce_add(AccTag{}, sum6));
  if constexpr (Block >= 8)
    store(output_origin + 7, vec::reduce_add(AccTag{}, sum7));
}

template <bool VaryRows, typename A, typename B, typename COutput>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void sve_skinny_matmul(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  using TA = typename A::ComputeType;
  nint_t output = 0;
  if constexpr (std::same_as<TA, bfloat16_t> || std::is_integral_v<TA>) {
    VECOPS_LOOP_ALIGN(64) for (; output + 8 <= outputs; output += 8)
      sve_skinny_block<VaryRows, 8>(a, b, c_output, output, logical_k);
  }
  VECOPS_LOOP_ALIGN(64) for (; output + 4 <= outputs; output += 4)
    sve_skinny_block<VaryRows, 4>(a, b, c_output, output, logical_k);
  switch (outputs - output) {
    case 3:
      sve_skinny_block<VaryRows, 3>(a, b, c_output, output, logical_k);
      break;
    case 2:
      sve_skinny_block<VaryRows, 2>(a, b, c_output, output, logical_k);
      break;
    case 1:
      sve_skinny_block<VaryRows, 1>(a, b, c_output, output, logical_k);
      break;
    default:
      break;
  }
}

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc> &&
    std::same_as<typename Atom::TA, typename Atom::TB> &&
    (std::is_floating_point_v<typename Atom::TAcc> ||
     (std::is_integral_v<typename Atom::TA> &&
      std::same_as<typename Atom::TAcc, int32_t>));

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
bool try_raw_mixed_sign_skinny_s8u8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const int8_t* a, nint_t a_stride,
    const uint8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride);

bool try_raw_mixed_sign_skinny_u8s8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const uint8_t* a, nint_t a_stride,
    const int8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride);

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool mixed_sign_sve_skinny_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, int32_t> &&
    ((std::same_as<typename Atom::TA, int8_t> &&
      std::same_as<typename Atom::TB, uint8_t>) ||
     (std::same_as<typename Atom::TA, uint8_t> &&
      std::same_as<typename Atom::TB, int8_t>)) &&
    std::same_as<typename Atom::TAcc, int32_t>;

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(COMPILER_CLANG)
__attribute__((preserve_most))
#endif
__attribute__((section(".vecops_kernel_text"))) bool
try_mixed_sign_sve_skinny(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput&, COutput& c_output) {
  if constexpr (!mixed_sign_sve_skinny_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const auto a_strides = a.raw_strides();
    const auto b_strides = b.raw_strides();
    const auto output_strides = c_output.raw_strides();
    if constexpr (
        std::same_as<typename Atom::TA, int8_t> &&
        std::same_as<typename Atom::TB, uint8_t>) {
      return try_raw_mixed_sign_skinny_s8u8(
          logical_m, logical_n, logical_k,
          reinterpret_cast<const int8_t*>(a.raw_data()), a_strides[0],
          reinterpret_cast<const uint8_t*>(b.raw_data()), b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), output_strides[0]);
    } else {
      return try_raw_mixed_sign_skinny_u8s8(
          logical_m, logical_n, logical_k,
          reinterpret_cast<const uint8_t*>(a.raw_data()), a_strides[0],
          reinterpret_cast<const int8_t*>(b.raw_data()), b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), output_strides[0]);
    }
  }
}
#endif

#if !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
#if defined(HAS_SME_F64F64)
void sve_skinny_fused_compute_f64_row(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);

void sve_skinny_fused_compute_f64_col(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);
#endif

template <bool VaryRows, int Block, typename T, typename Acc>
VECOPS_ALWAYS_INLINE void sve_skinny_fused_compute_block(
    const T* a_data, nint_t a_stride,
    const T* b_data, nint_t b_stride,
    Acc* values, nint_t output_origin, nint_t logical_k) {
  static_assert(1 <= Block && Block <= 8);
  using InputTag = vec::ScalableTag<T, 0>;
  using AccTag = std::conditional_t<
      std::same_as<T, bfloat16_t>,
      vec::ViewAs<float32_t, InputTag>,
      std::conditional_t<
          std::is_integral_v<T>, vec::ViewAs<int32_t, InputTag>,
          vec::Rebind<Acc, InputTag>>>;
  auto sum0 = vec::zeros(AccTag{});
  auto sum1 = vec::zeros(AccTag{});
  auto sum2 = vec::zeros(AccTag{});
  auto sum3 = vec::zeros(AccTag{});
  auto sum4 = vec::zeros(AccTag{});
  auto sum5 = vec::zeros(AccTag{});
  auto sum6 = vec::zeros(AccTag{});
  auto sum7 = vec::zeros(AccTag{});

  nint_t kk = 0;
  nint_t remaining = logical_k;
  while (remaining > 0) {
    const nint_t active = vecops::min(vec::size(InputTag{}), remaining);
    const auto load_row = [&](
        const T* data, nint_t row, nint_t stride) VECOPS_INLINE_LAMBDA {
      return vec::load(
          InputTag{}, data + row * stride + kk,
          vec::opt::first(active), vec::opt::zero);
    };
    const auto madd = [&](auto lhs, auto rhs, auto sum)
        VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return skinny_widening_dot_add(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::same_as<T, float16_t>) {
        return skinny_widening_fmadd(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::is_integral_v<T>) {
        return skinny_integer_dot_add(InputTag{}, sum, lhs, rhs);
      } else {
        return vec::fmadd(
            skinny_convert(AccTag{}, InputTag{}, lhs),
            skinny_convert(AccTag{}, InputTag{}, rhs), sum);
      }
    };
    if constexpr (VaryRows) {
      const auto shared = load_row(b_data, 0, b_stride);
      sum0 = madd(load_row(a_data, output_origin, a_stride), shared, sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            load_row(a_data, output_origin + 1, a_stride), shared, sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            load_row(a_data, output_origin + 2, a_stride), shared, sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            load_row(a_data, output_origin + 3, a_stride), shared, sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            load_row(a_data, output_origin + 4, a_stride), shared, sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            load_row(a_data, output_origin + 5, a_stride), shared, sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            load_row(a_data, output_origin + 6, a_stride), shared, sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            load_row(a_data, output_origin + 7, a_stride), shared, sum7);
    } else {
      const auto shared = load_row(a_data, 0, a_stride);
      sum0 = madd(shared, load_row(b_data, output_origin, b_stride), sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            shared, load_row(b_data, output_origin + 1, b_stride), sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            shared, load_row(b_data, output_origin + 2, b_stride), sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            shared, load_row(b_data, output_origin + 3, b_stride), sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            shared, load_row(b_data, output_origin + 4, b_stride), sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            shared, load_row(b_data, output_origin + 5, b_stride), sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            shared, load_row(b_data, output_origin + 6, b_stride), sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            shared, load_row(b_data, output_origin + 7, b_stride), sum7);
    }
    kk += active;
    remaining -= active;
  }

  values[output_origin] = vec::reduce_add(AccTag{}, sum0);
  if constexpr (Block >= 2)
    values[output_origin + 1] = vec::reduce_add(AccTag{}, sum1);
  if constexpr (Block >= 3)
    values[output_origin + 2] = vec::reduce_add(AccTag{}, sum2);
  if constexpr (Block >= 4)
    values[output_origin + 3] = vec::reduce_add(AccTag{}, sum3);
  if constexpr (Block >= 5)
    values[output_origin + 4] = vec::reduce_add(AccTag{}, sum4);
  if constexpr (Block >= 6)
    values[output_origin + 5] = vec::reduce_add(AccTag{}, sum5);
  if constexpr (Block >= 7)
    values[output_origin + 6] = vec::reduce_add(AccTag{}, sum6);
  if constexpr (Block >= 8)
    values[output_origin + 7] = vec::reduce_add(AccTag{}, sum7);
}

template <bool VaryRows, typename T, typename Acc>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void sve_skinny_fused_compute(
    const T* a_data, nint_t a_stride,
    const T* b_data, nint_t b_stride,
    Acc* values, nint_t outputs, nint_t logical_k) {
  nint_t output = 0;
  if constexpr (std::same_as<T, bfloat16_t> || std::is_integral_v<T>) {
    VECOPS_LOOP_ALIGN(64) for (; output + 8 <= outputs; output += 8) {
      sve_skinny_fused_compute_block<VaryRows, 8>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
    }
  }
  VECOPS_LOOP_ALIGN(64) for (; output + 4 <= outputs; output += 4) {
    sve_skinny_fused_compute_block<VaryRows, 4>(
        a_data, a_stride, b_data, b_stride,
        values, output, logical_k);
  }
  switch (outputs - output) {
    case 3:
      sve_skinny_fused_compute_block<VaryRows, 3>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    case 2:
      sve_skinny_fused_compute_block<VaryRows, 2>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    case 1:
      sve_skinny_fused_compute_block<VaryRows, 1>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    default:
      break;
  }
}

template <bool VaryRows, typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void sve_skinny_fused_matmul(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  using T = typename A::ComputeType;
  using Acc = typename COutput::ComputeType;
  static_assert(std::same_as<T, typename B::ComputeType>);
  constexpr nint_t MaxOutputs =
      std::same_as<T, float16_t> ? 16 : 64;
  alignas(64) Acc values[MaxOutputs];
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  sve_skinny_fused_compute<VaryRows>(
      reinterpret_cast<const T*>(a.raw_data()), a_strides[0],
      reinterpret_cast<const T*>(b.raw_data()), b_strides[0],
      values, outputs, logical_k);

  using OutputTag = vec::ScalableTag<Acc, 0>;
  if constexpr (VaryRows) {
    for (nint_t output = 0; output < outputs; ++output) {
      c_output.store(
          OutputTag{}, tensor::coord(output, nint_t{0}), tensor::axis<1>,
          vec::fill(OutputTag{}, values[output]), vec::opt::first(1));
    }
  } else {
    const nint_t lanes = vec::size(OutputTag{});
    for (nint_t output = 0; output < outputs; output += lanes) {
      const nint_t active = vecops::min(lanes, outputs - output);
      const auto value = vec::load(
          OutputTag{}, values + output,
          vec::opt::first(active), vec::opt::zero);
      c_output.store(
          OutputTag{}, tensor::coord(nint_t{0}, output), tensor::axis<1>,
          value, vec::opt::first(active));
    }
  }
}

// Keep coordinate-aware epilogues out of the already layout-sensitive fused
// dispatch bodies. Their transform type remains open-ended, so the wrapper
// stays in the header, but one noinline COMDAT per actual transform is enough.
template <bool VaryRows, typename A, typename B, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_lane_local_matmul(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  static_assert(!COutput::Transform::is_elementwise);
  static_assert(COutput::Transform::is_lane_local);
  sve_skinny_fused_matmul<VaryRows>(
      a, b, c_output, outputs, logical_k);
}

#if defined(HAS_SME_F64F64)
template <bool VaryRows, typename A, typename B, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_matmul_f64_external(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  static_assert(std::same_as<typename A::ComputeType, float64_t>);
  static_assert(std::same_as<typename B::ComputeType, float64_t>);
  static_assert(std::same_as<typename COutput::ComputeType, float64_t>);
  alignas(64) float64_t values[64];
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  if constexpr (VaryRows) {
    sve_skinny_fused_compute_f64_col(
        reinterpret_cast<const float64_t*>(a.raw_data()), a_strides[0],
        reinterpret_cast<const float64_t*>(b.raw_data()), b_strides[0],
        values, outputs, logical_k);
  } else {
    sve_skinny_fused_compute_f64_row(
        reinterpret_cast<const float64_t*>(a.raw_data()), a_strides[0],
        reinterpret_cast<const float64_t*>(b.raw_data()), b_strides[0],
        values, outputs, logical_k);
  }

  using OutputTag = vec::ScalableTag<float64_t, 0>;
  if constexpr (VaryRows) {
    for (nint_t output = 0; output < outputs; ++output) {
      c_output.store(
          OutputTag{}, tensor::coord(output, nint_t{0}), tensor::axis<1>,
          vec::fill(OutputTag{}, values[output]), vec::opt::first(1));
    }
  } else {
    const nint_t lanes = vec::size(OutputTag{});
    for (nint_t output = 0; output < outputs; output += lanes) {
      const nint_t active = vecops::min(lanes, outputs - output);
      const auto value = vec::load(
          OutputTag{}, values + output,
          vec::opt::first(active), vec::opt::zero);
      c_output.store(
          OutputTag{}, tensor::coord(nint_t{0}, output), tensor::axis<1>,
          value, vec::opt::first(active));
    }
  }
}
#endif

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_fused_output_supported_v =
    COutput::Transform::is_elementwise ||
    COutput::Transform::is_lane_local;

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_fused_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    !direct_row_major_output_v<COutput> &&
    COutput::Rank == 2 &&
    sve_skinny_fused_output_supported_v<Atom, A, B, CInput, COutput> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc> &&
    std::same_as<typename Atom::TA, typename Atom::TB> &&
#if defined(HAS_SME_F64F64)
    (sizeof(typename Atom::TAcc) <= sizeof(float32_t) ||
     std::same_as<typename Atom::TAcc, float64_t>) &&
#else
    // TODO: Re-evaluate FP64 after its fused wrappers can live in a separate
    // cold TU. The leaf itself is faster, but adding it to the current header
    // perturbs unchanged FP64 fusion paths beyond the production regression
    // gate.
    sizeof(typename Atom::TAcc) <= sizeof(float32_t) &&
#endif
    (std::is_floating_point_v<typename Atom::TAcc> ||
     (std::is_integral_v<typename Atom::TA> &&
      std::same_as<typename Atom::TAcc, int32_t>));
#endif
#endif

#if defined(CPU_CAPABILITY_SVE)
template <typename Atom>
struct PackedMMLATraits;

#if defined(__ARM_FEATURE_SVE_BF16)
template <>
struct PackedMMLATraits<gemm::SME_BF16F32> {
  using Element = bfloat16_t;
  using Acc = float32_t;
  using InputVec = svbfloat16_t;
  using AccVec = svfloat32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svreinterpret_bf16_u16(svld1_u16(
        svptrue_b16(), reinterpret_cast<const uint16_t*>(pointer)));
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svreinterpret_bf16_u16(svdup_u16(0));
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_bf16_u32(svzip1_u32(
        svreinterpret_u32_bf16(group0),
        svreinterpret_u32_bf16(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_bf16(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_f32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svbfmmla_f32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_f32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    const auto bits = svreinterpret_u64_f32(value);
    const auto first = svreinterpret_f32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_f32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_f32(pg, row0, first);
    svst1_f32(pg, row1, second);
  }
};
#endif

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
template <>
struct PackedMMLATraits<gemm::SME_I8I32<int8_t, int8_t>> {
  using Element = int8_t;
  using Acc = int32_t;
  using InputVec = svint8_t;
  using AccVec = svint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_s8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svdup_s8(0);
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_s8_u32(svzip1_u32(
        svreinterpret_u32_s8(group0), svreinterpret_u32_s8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_s8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_s32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_s32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    const auto bits = svreinterpret_u64_s32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    svst1_s32(pg, row1, second);
  }
};

template <>
struct PackedMMLATraits<gemm::SME_I8I32<uint8_t, uint8_t>> {
  using Element = uint8_t;
  using Acc = int32_t;
  using InputVec = svuint8_t;
  using AccVec = svuint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_u8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svdup_u8(0);
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_u8_u32(svzip1_u32(
        svreinterpret_u32_u8(group0), svreinterpret_u32_u8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_u8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_u32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_u32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, svreinterpret_s32_u32(value));
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    const auto bits = svreinterpret_u64_u32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    svst1_s32(pg, row1, second);
  }
};
#endif

template <gemm::Atom Atom>
inline constexpr bool packed_mmla_supported_atom_v =
#if defined(__ARM_FEATURE_SVE_BF16)
    std::same_as<Atom, gemm::SME_BF16F32> ||
#endif
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    std::same_as<Atom, gemm::SME_I8I32<int8_t, int8_t>> ||
    std::same_as<Atom, gemm::SME_I8I32<uint8_t, uint8_t>> ||
#endif
    false;

// TODO: Keep this ordinary-SVE MMLA operation SME-backend-local until the
// segment layout, mixed-sign behavior, and VL/SVL contract have another user.
// It intentionally consumes the existing MOPA PackedAB ABI by joining two
// adjacent 32-bit K groups; no second public packing format is introduced.
template <gemm::Atom Atom>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void packed_ab_mmla(
    const typename Atom::TA* packed_a, nint_t a_group_stride,
    const typename Atom::TB* packed_b, nint_t b_group_stride,
    typename Atom::TAcc* output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  using Op = PackedMMLATraits<Atom>;
  constexpr nint_t KPack =
      gemm::packing_t<Atom, gemm::Operand::A>::KPack;
  const nint_t groups = ceil_div(logical_k, KPack);

  if (logical_m == 8) {
    auto acc = Op::zero_acc();
    VECOPS_LOOP_ALIGN(64) for (
        nint_t group = 0; group < groups; group += 2) {
      const auto a0 = Op::load(packed_a + group * a_group_stride);
      const auto b0 = Op::load(packed_b + group * b_group_stride);
      const auto a1 = group + 1 < groups
          ? Op::load(packed_a + (group + 1) * a_group_stride)
          : Op::zero_input();
      const auto b1 = group + 1 < groups
          ? Op::load(packed_b + (group + 1) * b_group_stride)
          : Op::zero_input();
      const auto a = Op::zip_groups(a0, a1);
      const auto b = Op::zip_groups(b0, b1);
      acc = Op::mmla(
          acc, a, Op::template broadcast_segment<0>(b));
    }
    Op::store_contiguous(acc, output, logical_m * logical_n);
    return;
  }

  auto acc0 = Op::zero_acc();
  auto acc1 = Op::zero_acc();
  VECOPS_LOOP_ALIGN(64) for (
      nint_t group = 0; group < groups; group += 2) {
    const auto a0 = Op::load(packed_a + group * a_group_stride);
    const auto b0 = Op::load(packed_b + group * b_group_stride);
    const auto a1 = group + 1 < groups
        ? Op::load(packed_a + (group + 1) * a_group_stride)
        : Op::zero_input();
    const auto b1 = group + 1 < groups
        ? Op::load(packed_b + (group + 1) * b_group_stride)
        : Op::zero_input();
    const auto a = Op::zip_groups(a0, a1);
    const auto b = Op::zip_groups(b0, b1);
    acc0 = Op::mmla(
        acc0, Op::template broadcast_segment<0>(a), b);
    if (logical_m == 4) {
      acc1 = Op::mmla(
          acc1, Op::template broadcast_segment<1>(a), b);
    }
  }
  Op::store_row_pair(
      acc0, output, output + logical_n, logical_n);
  if (logical_m == 4) {
    Op::store_row_pair(
        acc1, output + 2 * logical_n,
        output + 3 * logical_n, logical_n);
  }
}

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool packed_mmla_candidate_v =
    packed_mmla_supported_atom_v<Atom> &&
    is_packed_access_v<Atom, gemm::Operand::A, A> &&
    is_packed_access_v<Atom, gemm::Operand::B, B> &&
    generic::RawDirectAccess<A> && generic::RawDirectAccess<B> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    direct_row_major_output_v<COutput> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc>;

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE bool try_packed_mmla(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput&, COutput& c_output) {
  if constexpr (!packed_mmla_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const bool elongated =
        (logical_m == 2 && logical_n == 8) ||
        (logical_m == 8 && logical_n == 2);
    const bool square = logical_m == 4 && logical_n == 4;
    if ((!elongated && !square) || logical_k < 0) return false;

    const nint_t max_k = square
        ? 257
        : (std::same_as<Atom, gemm::SME_BF16F32> ? 1025 : 513);
    if (logical_k > max_k) return false;

    const nint_t a_panel = static_cast<nint_t>(
        gemm::packing_t<Atom, gemm::Operand::A>::panel());
    const nint_t b_panel = static_cast<nint_t>(
        gemm::packing_t<Atom, gemm::Operand::B>::panel());
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    // Preserve the exact X24 store sequence. Padded output rows need a
    // separate vertical store strategy and were not part of its performance
    // or correctness evidence.
    if (static_cast<nint_t>(tensor::stride_value<0>(
            c_output.spec().output_layout())) != logical_n) return false;

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        static_cast<nint_t>(tensor::stride_value<1>(
            a.spec().input_layout())),
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        static_cast<nint_t>(tensor::stride_value<1>(
            b.spec().input_layout())),
        reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data()),
        logical_m, logical_n, logical_k);
    return true;
  }
}

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE bool try_packed_mmla_tiny(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput&, COutput& c_output) {
  if constexpr (!packed_mmla_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const bool short_wide =
        logical_m == 2 && (logical_n == 2 || logical_n == 4);
    const bool tall_narrow = logical_m == 4 && logical_n == 2;
    if ((!short_wide && !tall_narrow) || logical_k < 0) return false;

    const nint_t max_k = tall_narrow
        ? 257
        : (std::same_as<Atom, gemm::SME_BF16F32> ? 1025 : 513);
    if (logical_k > max_k) return false;

    const nint_t a_panel = static_cast<nint_t>(
        gemm::packing_t<Atom, gemm::Operand::A>::panel());
    const nint_t b_panel = static_cast<nint_t>(
        gemm::packing_t<Atom, gemm::Operand::B>::panel());
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    if (static_cast<nint_t>(tensor::stride_value<0>(
            c_output.spec().output_layout())) != logical_n) return false;

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        static_cast<nint_t>(tensor::stride_value<1>(
            a.spec().input_layout())),
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        static_cast<nint_t>(tensor::stride_value<1>(
            b.spec().input_layout())),
        reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data()),
        logical_m, logical_n, logical_k);
    return true;
  }
}
#endif

enum class DispatchOwner {
  General,
  MixedSignSkinnyRow,
  MixedSignSkinnyColumn,
  RawSkinnyRow,
  RawSkinnyColumn,
  FusedSkinnyRow,
  FusedSkinnyColumn,
  RuntimeQuantINT8,
  PackedMMLAPrimary,
  PackedMMLATiny,
};

template <meta::ValueType E, nint_t Value>
inline constexpr bool extent_is_v =
    meta::range_within_v<std::remove_cvref_t<E>, Value, Value>;

template <gemm::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Scope>
consteval DispatchOwner select_dispatch_owner() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (!execution::has_resource_v<
                    execution::details::arm::StreamingZA, Scope>) {
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (mixed_sign_sve_skinny_candidate_v<
                      Atom, A, B, CInput, COutput> &&
                  meta::lower_bound_at_least_v<KV, 0> &&
                  extent_is_v<MV, 1> &&
                  meta::range_within_v<NV, 0, 64>) {
      return DispatchOwner::MixedSignSkinnyRow;
    } else if constexpr (mixed_sign_sve_skinny_candidate_v<
                             Atom, A, B, CInput, COutput> &&
                         meta::lower_bound_at_least_v<KV, 0> &&
                         extent_is_v<NV, 1> &&
                         meta::range_within_v<MV, 0, 64>) {
      return DispatchOwner::MixedSignSkinnyColumn;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
    if constexpr (sve_skinny_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, gemm::SME_F16F32> ? 16 : 64;
      if constexpr (extent_is_v<MV, 1> &&
                    meta::range_within_v<NV, 0, MaxOutputs> &&
                    meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::RawSkinnyRow;
      } else if constexpr (extent_is_v<NV, 1> &&
                           meta::range_within_v<MV, 0, MaxOutputs> &&
                           meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::RawSkinnyColumn;
      }
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
    if constexpr (sve_skinny_fused_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, gemm::SME_F16F32> ? 16 : 64;
      if constexpr (extent_is_v<MV, 1> &&
                    meta::range_within_v<NV, 0, MaxOutputs> &&
                    meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::FusedSkinnyRow;
      } else if constexpr (extent_is_v<NV, 1> &&
                           meta::range_within_v<MV, 0, MaxOutputs> &&
                           meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::FusedSkinnyColumn;
      }
    }
#endif
#if defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (fused_runtime_quant_int8_candidate_v<
                      Atom, A, B, CInput, COutput> &&
                  extent_is_v<MV, 1> &&
                  meta::lower_bound_at_least_v<NV, 1> && NV::aligns(64) &&
                  meta::lower_bound_at_least_v<KV, 0> && KV::aligns(64)) {
      return DispatchOwner::RuntimeQuantINT8;
    }
#endif
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (packed_mmla_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr bool Elongated =
          (extent_is_v<MV, 2> && extent_is_v<NV, 8>) ||
          (extent_is_v<MV, 8> && extent_is_v<NV, 2>);
      constexpr bool Square =
          extent_is_v<MV, 4> && extent_is_v<NV, 4>;
      constexpr nint_t PrimaryMaxK = Square
          ? 257
          : (std::same_as<Atom, gemm::SME_BF16F32> ? 1025 : 513);
      if constexpr ((Elongated || Square) &&
                    meta::range_within_v<KV, 0, PrimaryMaxK>) {
        return DispatchOwner::PackedMMLAPrimary;
      }
      constexpr bool ShortWide =
          extent_is_v<MV, 2> &&
          (extent_is_v<NV, 2> || extent_is_v<NV, 4>);
      constexpr bool TallNarrow =
          extent_is_v<MV, 4> && extent_is_v<NV, 2>;
      constexpr nint_t TinyMaxK = TallNarrow
          ? 257
          : (std::same_as<Atom, gemm::SME_BF16F32> ? 1025 : 513);
      if constexpr ((ShortWide || TallNarrow) &&
                    meta::range_within_v<KV, 0, TinyMaxK>) {
        return DispatchOwner::PackedMMLATiny;
      }
    }
#endif
  }
  return DispatchOwner::General;
}

template <gemm::Atom Atom, int NM, int NN,
          bool PrefetchLargeWorkingSet,
          typename A, typename B>
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
  const nint_t a_step = static_cast<nint_t>(tensor::stride_value<1>(
      a.spec().input_layout()));
  const nint_t b_step = static_cast<nint_t>(tensor::stride_value<1>(
      b.spec().input_layout()));

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
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
      constexpr nint_t PrefetchDistance = 48;
      nint_t kg = 0;
      if constexpr (PrefetchLargeWorkingSet) {
        if (groups > PrefetchDistance) {
          const nint_t prefetch_groups = groups - PrefetchDistance;
          VECOPS_LOOP_ALIGN(64) for (; kg < prefetch_groups; ++kg) {
            vec::prefetch(
                ATag{}, a0 + (kg + PrefetchDistance) * a_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                ATag{}, a1 + (kg + PrefetchDistance) * a_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                BTag{}, b0 + (kg + PrefetchDistance) * b_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                BTag{}, b1 + (kg + PrefetchDistance) * b_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
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
      VECOPS_LOOP_ALIGN(64) for (; kg < groups; ++kg) {
#else
      VECOPS_LOOP_ALIGN(64) for (nint_t kg = 0; kg < groups; ++kg) {
#endif
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

template <gemm::Atom Atom, bool Full, bool NonEmpty, int Block>
VECOPS_ALWAYS_INLINE auto tile_active_extent(nint_t active) {
  using Tile = std::remove_cvref_t<decltype(Atom::M_R)>;
  if constexpr (Full) {
    return Atom::M_R;
  } else {
    const nint_t lanes = static_cast<nint_t>(Atom::M_R);
    const nint_t value = vecops::clamp(
        active - static_cast<nint_t>(Block) * lanes,
        nint_t{0}, lanes);
    if constexpr (meta::has_upper_bound_v<Tile>) {
      constexpr nint_t Lo = NonEmpty ? 1 : 0;
      constexpr nint_t Hi = meta::upper_bound_v<Tile>;
      return meta::dyn<1, Lo, Hi>(value);
    } else {
      return meta::Any{value};
    }
  }
}

template <int Tile, bool FullM, bool FullN,
          meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n) {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value) {
    return;
  }
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto pg = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (FullN) return vec::mtrue(Tag{});
    else return vec::mwhilelt(Tag{}, nint_t{0}, active_n_value);
  }();
  if constexpr (direct_row_major_input_v<CInput>) {
    const auto strides = input.raw_strides();
    const auto* base = reinterpret_cast<const T*>(input.raw_data()) + m * strides[0] + n;
    if constexpr (
        std::is_floating_point_v<T> &&
        std::same_as<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>,
            meta::Const<0>>) {
      const auto value = vec::load(
          Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
      const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (FullM) return vec::mtrue(Tag{});
        else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
      }();
      vec::details::sme::mopa<Tile, T, T, T>(
          pg_rows, pg, vec::fill(Tag{}, T{1}), value);
    } else if constexpr (
        (std::same_as<T, int32_t> || std::same_as<T, uint32_t>) &&
        std::same_as<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>,
            meta::Const<0>>) {
      if constexpr (meta::is_singleton_v<ActiveM> &&
                    meta::singleton_value_v<ActiveM> == 1) {
        using U = std::conditional_t<std::same_as<T, int32_t>, uint32_t, T>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            0, static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base));
      } else {
        const auto value = vec::load(
            Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
        const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (FullM) return vec::mtrue(Tag{});
          else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
        }();
        vec::details::sme::addha<Tile, T>(pg_rows, pg, value);
      }
    } else {
      for (nint_t row = 0; row < active_m_value; ++row) {
        using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            static_cast<uint32_t>(row),
            static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base + row * strides[0]));
      }
    }
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) {
      const auto value = input.load(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
          vec::opt::first(active_n_value), vec::opt::zero);
      write_row<Tile, T>(static_cast<uint32_t>(row), pg, value);
    }
  }
}

template <int Tile, bool FullM, bool FullN,
          meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto pg = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (FullN) return vec::mtrue(Tag{});
    else return vec::mwhilelt(Tag{}, nint_t{0}, active_n_value);
  }();
  if constexpr (direct_row_major_output_v<COutput>) {
    const auto strides = output.raw_strides();
    auto* base = reinterpret_cast<T*>(output.raw_data()) + m * strides[0] + n;
    for (nint_t row = 0; row < active_m_value; ++row) {
      using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      using BitsTag = vec::ScalableTag<U, 0>;
      vec::details::sme::store_hor<Tile>(
          static_cast<uint32_t>(row),
          static_cast<vec::Mask<BitsTag>>(pg),
          reinterpret_cast<U*>(base + row * strides[0]));
    }
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) {
      const auto value = static_cast<vec::Vec<Tag>>(
          read_row<Tile, T>(static_cast<uint32_t>(row), pg));
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n_value));
    }
  }
}

template <gemm::Atom Atom, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks, bool FullK,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_group(
    const A& a, const OperandInvariants<Atom, gemm::Operand::A, A>& a_invariants,
    const B& b, const OperandInvariants<Atom, gemm::Operand::B, B>& b_invariants,
    nint_t m, nint_t n, nint_t kg,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  if constexpr (NN == 1) {
    const auto bv = load_operand<
        Atom, gemm::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<
          Atom, gemm::Operand::A, MI,
          FullM || (ExactBlocks && MI + 1 < NM), FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      mopa<Atom, MI>(av, bv);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else if constexpr (NM == 1) {
    const auto av = load_operand<
        Atom, gemm::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<
          Atom, gemm::Operand::B, NI,
          FullN || (ExactBlocks && NI + 1 < NN), FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(av, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  } else if constexpr (NM == 2 && NN == 2) {
    const auto a0 = load_operand<
        Atom, gemm::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<
        Atom, gemm::Operand::A, 1,
        FullM || ExactBlocks, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto b0 = load_operand<
        Atom, gemm::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<
        Atom, gemm::Operand::B, 1,
        FullN || ExactBlocks, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    mopa<Atom, 0>(a0, b0);
    mopa<Atom, 1>(a0, b1);
    mopa<Atom, 2>(a1, b0);
    mopa<Atom, 3>(a1, b1);
  } else if constexpr (NN == 2) {
    const auto b0 = load_operand<
        Atom, gemm::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<
        Atom, gemm::Operand::B, 1,
        FullN || ExactBlocks, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<
          Atom, gemm::Operand::A, MI,
          FullM || (ExactBlocks && MI + 1 < NM), FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      mopa<Atom, 2 * MI>(av, b0);
      mopa<Atom, 2 * MI + 1>(av, b1);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else {
    static_assert(NM == 2 && NN <= 4);
    const auto a0 = load_operand<
        Atom, gemm::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<
        Atom, gemm::Operand::A, 1,
        FullM || ExactBlocks, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<
          Atom, gemm::Operand::B, NI,
          FullN || (ExactBlocks && NI + 1 < NN), FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(a0, bv);
      mopa<Atom, NN + NI>(a1, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  }
}

template <gemm::Atom Atom, typename Plan, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  constexpr int Outputs = NM * NN;
  constexpr bool FastPacked = Plan::value;
  constexpr bool PrefetchLargeWorkingSet =
      Plan::prefetch_large_working_set &&
      std::same_as<Atom, gemm::SME_BF16F32> && NM == 2 && NN == 2;
  const nint_t lanes = static_cast<nint_t>(Atom::M_R);
  vec::details::sme::zero_za();
  const OperandInvariants<Atom, gemm::Operand::A, A> a_invariants(a);
  const OperandInvariants<Atom, gemm::Operand::B, B> b_invariants(b);
  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA {
    (initialize_c_tile<
         static_cast<int>(I),
         FullM || (ExactBlocks && I / NN + 1 < NM),
         FullN || (ExactBlocks && I % NN + 1 < NN)>(
         c_input,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         tile_active_extent<
             Atom,
             FullM || (ExactBlocks && I / NN + 1 < NM),
             FullM || ExactBlocks,
             static_cast<int>(I / NN)>(active_m),
         tile_active_extent<
             Atom,
             FullN || (ExactBlocks && I % NN + 1 < NN),
             FullN || ExactBlocks,
             static_cast<int>(I % NN)>(active_n)), ...);
  }(std::make_index_sequence<Outputs>{});

  auto compute_generic = [&]() VECOPS_INLINE_LAMBDA {
    constexpr nint_t KP = gemm::packing_t<Atom, gemm::Operand::A>::KPack;
    if constexpr (KP == 1 || std::same_as<Atom, gemm::SME_BF16F32> ||
                  std::same_as<Atom, gemm::SME_F16F32>) {
      const nint_t full_groups = logical_k / KP;
      for (nint_t kg = 0; kg < full_groups; ++kg) {
        compute_group<
            Atom, NM, NN, FullM, FullN, ExactBlocks, true>(
            a, a_invariants, b, b_invariants,
            m, n, kg, logical_m, logical_n, logical_k);
      }
      if constexpr (KP > 1) {
        if (full_groups * KP < logical_k) {
          compute_group<
              Atom, NM, NN, FullM, FullN, ExactBlocks, false>(
              a, a_invariants, b, b_invariants,
              m, n, full_groups,
              logical_m, logical_n, logical_k);
        }
      }
    } else {
      const nint_t groups = ceil_div(logical_k, KP);
      for (nint_t kg = 0; kg < groups; ++kg) {
        compute_group<
            Atom, NM, NN, FullM, FullN, ExactBlocks, false>(
            a, a_invariants, b, b_invariants,
            m, n, kg,
            logical_m, logical_n, logical_k);
      }
    }
  };

  if constexpr (FastPacked && ExactBlocks && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    compute_packed_groups<Atom, NM, NN, PrefetchLargeWorkingSet>(
        a, b, m, n, logical_k);
  } else if constexpr (FastPacked && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    const bool all_logical_blocks_exist =
        active_m > static_cast<nint_t>(NM - 1) * lanes &&
        active_n > static_cast<nint_t>(NN - 1) * lanes;
    if (all_logical_blocks_exist) {
      compute_packed_groups<Atom, NM, NN, PrefetchLargeWorkingSet>(
          a, b, m, n, logical_k);
    } else {
      compute_generic();
    }
  } else {
    compute_generic();
  }

  [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA {
    (store_c_tile<
         static_cast<int>(I),
         FullM || (ExactBlocks && I / NN + 1 < NM),
         FullN || (ExactBlocks && I % NN + 1 < NN)>(
         c_output,
         m + static_cast<nint_t>(I / NN) * lanes,
         n + static_cast<nint_t>(I % NN) * lanes,
         tile_active_extent<
             Atom,
             FullM || (ExactBlocks && I / NN + 1 < NM),
             FullM || ExactBlocks,
             static_cast<int>(I / NN)>(active_m),
         tile_active_extent<
             Atom,
             FullN || (ExactBlocks && I % NN + 1 < NN),
             FullN || ExactBlocks,
             static_cast<int>(I % NN)>(active_n)), ...);
  }(std::make_index_sequence<Outputs>{});
}

} // namespace vecops::kernel::matmul_details::sme

namespace vecops::kernel::matmul_details {

template <>
struct Backend<matmul_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;
  template <gemm::Atom Atom, typename, typename A, typename B>
  using Catalog = sme::Catalog<sme::use_expanded_catalog_v<Atom, A, B>>;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 0; }

  template <gemm::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B>
  using EffectivePolicy = std::conditional_t<
      std::same_as<Policy, matmul_policy::Automatic>,
      std::conditional_t<
          sme::use_expanded_catalog_v<Atom, A, B>,
          kernel::loop::tile2d_policy::ExactCover,
          std::conditional_t<
              sme::prefer_constraint_pruning_v<Atom, M, N, K, A, B>,
              kernel::loop::tile2d_policy::FourRegions,
              kernel::loop::tile2d_policy::ExactCover>>,
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
    constexpr auto Owner = sme::select_dispatch_owner<
        Atom, M, N, K, A, B, CInput, COutput, Scope>();
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (Owner == sme::DispatchOwner::MixedSignSkinnyRow ||
                  Owner == sme::DispatchOwner::MixedSignSkinnyColumn) {
      const bool handled = sme::try_mixed_sign_sve_skinny<Atom>(
          static_cast<nint_t>(m), static_cast<nint_t>(n),
          static_cast<nint_t>(k), a, b, c_input, c_output);
      VECOPS_ASSERT(handled, "compile-time mixed-sign skinny plan rejected");
      return;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
    if constexpr (Owner == sme::DispatchOwner::RawSkinnyRow) {
      sme::sve_skinny_matmul<false>(
          a, b, c_output, static_cast<nint_t>(n), static_cast<nint_t>(k));
      return;
    } else if constexpr (Owner == sme::DispatchOwner::RawSkinnyColumn) {
      sme::sve_skinny_matmul<true>(
          a, b, c_output, static_cast<nint_t>(m), static_cast<nint_t>(k));
      return;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
    if constexpr (Owner == sme::DispatchOwner::FusedSkinnyRow) {
#if defined(HAS_SME_F64F64)
      if constexpr (std::same_as<Atom, gemm::SME_F64F64>) {
        sme::sve_skinny_fused_matmul_f64_external<false>(
            a, b, c_output, static_cast<nint_t>(n), static_cast<nint_t>(k));
      } else
#endif
      {
        if constexpr (COutput::Transform::is_elementwise) {
          sme::sve_skinny_fused_matmul<false>(
              a, b, c_output, static_cast<nint_t>(n), static_cast<nint_t>(k));
        } else {
          sme::sve_skinny_fused_lane_local_matmul<false>(
              a, b, c_output, static_cast<nint_t>(n), static_cast<nint_t>(k));
        }
      }
      return;
    } else if constexpr (Owner == sme::DispatchOwner::FusedSkinnyColumn) {
#if defined(HAS_SME_F64F64)
      if constexpr (std::same_as<Atom, gemm::SME_F64F64>) {
        sme::sve_skinny_fused_matmul_f64_external<true>(
            a, b, c_output, static_cast<nint_t>(m), static_cast<nint_t>(k));
      } else
#endif
      {
        if constexpr (COutput::Transform::is_elementwise) {
          sme::sve_skinny_fused_matmul<true>(
              a, b, c_output, static_cast<nint_t>(m), static_cast<nint_t>(k));
        } else {
          sme::sve_skinny_fused_lane_local_matmul<true>(
              a, b, c_output, static_cast<nint_t>(m), static_cast<nint_t>(k));
        }
      }
      return;
    }
#endif
#if defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (Owner == sme::DispatchOwner::RuntimeQuantINT8) {
      if (sme::try_fused_runtime_quant_int8_packed_b_gemv<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    }
#endif
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (Owner == sme::DispatchOwner::PackedMMLAPrimary) {
      if (sme::try_packed_mmla<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    } else if constexpr (Owner == sme::DispatchOwner::PackedMMLATiny) {
      if (sme::try_packed_mmla_tiny<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    }
#endif
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
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(M, N, K k, Fn&& fn) {
    using MV = std::remove_cvref_t<M>;
    using NV = std::remove_cvref_t<N>;
    using KV = std::remove_cvref_t<K>;
    constexpr bool PrefetchLargeWorkingSet =
        sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
        sme::is_packed_access_v<Atom, gemm::Operand::B, B> &&
        sme::large_packed_prefetch_v<Atom, MV, NV, KV>;
    auto invoke = [&]<bool FastPacked>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
      std::forward<Fn>(fn).template operator()<
          sme::KernelPlan<FastPacked, PrefetchLargeWorkingSet>>();
    };
    if constexpr (
        sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
        sme::is_packed_access_v<Atom, gemm::Operand::B, B>) {
      constexpr nint_t KP = gemm::packing_t<
          Atom, gemm::Operand::A>::KPack;
      const auto k_groups = ceil_div(k, meta::cint<KP>);
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
        Atom, Plan, Case::a, Case::b,
        Case::m_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::n_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::exact_blocks>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k,
            m, n, active_m, active_n);
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_SME_BACKEND_H
