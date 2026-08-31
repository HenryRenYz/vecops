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
#include "vecops/kernel/details/matmul/RuntimeQuantization.h"
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
    } else if constexpr (A * B <= 8) {
      // Keep the legacy area-four family as the catalog default.  ZA64's
      // RuntimeExactArea8 policy selects these larger families explicitly;
      // ZA32 policies therefore cannot accidentally consume tile numbers
      // above three.
      constexpr int imbalance = A > B ? A - B : B - A;
      return 300 + 4 * A * B + (A + B) - 2 * imbalance;
    } else {
      return -1;
    }
  }
};

using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider, tile::Tile2DSearchSpace<4, 4, 8>>;

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

template <gemm::Atom Atom, gemm::Operand Side, typename Source>
struct OperandInvariants {
  nint_t row_bytes = 0;
  bool gather_offsets_fit = false;

  VECOPS_ALWAYS_INLINE explicit OperandInvariants(const Source& source) {
    using T = typename gemm::packing_t<Atom, Side>::Element;
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      row_bytes = source.raw_strides()[0] * static_cast<nint_t>(sizeof(T));
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
          bool FullK, typename Source>
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
    const nint_t active = vec::details::sme::clamp_value(
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
        const auto words = vec::load(
            WordTag{}, base, vec::opt::first(active),
            vec::strided(invariants.row_bytes, vec::scale<1>),
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

#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV) && \
    defined(HAS_SME_FA64)
template <typename TA, typename TB>
VECOPS_ALWAYS_INLINE auto streaming_integer_dot_add(
    vec::Vec<vec::ViewAs<int32_t, vec::ScalableTag<TA, 0>>> acc,
    vec::Vec<vec::ScalableTag<TA, 0>> a,
    vec::Vec<vec::ScalableTag<TB, 0>> b) {
  using InputTagA = vec::ScalableTag<TA, 0>;
  using AccTag = vec::ViewAs<int32_t, InputTagA>;
  const auto raw_acc = vec::details::sve_basic_raw_word(acc);
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  if constexpr (std::same_as<TA, int8_t> && std::same_as<TB, int8_t>) {
    return vec::details::sve_basic_wrap_word<AccTag>(
        svdot_s32(raw_acc, raw_a, raw_b));
  } else if constexpr (
      std::same_as<TA, uint8_t> && std::same_as<TB, uint8_t>) {
    return vec::details::sve_basic_wrap_word<AccTag>(
        svreinterpret_s32_u32(svdot_u32(
            svreinterpret_u32_s32(raw_acc), raw_a, raw_b)));
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
  } else if constexpr (
      std::same_as<TA, int8_t> && std::same_as<TB, uint8_t>) {
    return vec::details::sve_basic_wrap_word<AccTag>(
        svsudot_s32(raw_acc, raw_a, raw_b));
  } else {
    static_assert(std::same_as<TA, uint8_t> && std::same_as<TB, int8_t>);
    return vec::details::sve_basic_wrap_word<AccTag>(
        svusdot_s32(raw_acc, raw_a, raw_b));
#else
  } else {
    static_assert(
        std::same_as<TA, TB>,
        "mixed-sign streaming DOT requires SVE I8MM");
    return acc;
#endif
  }
}

template <typename T>
using ScalarBits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<
        sizeof(T) == 2, uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

template <typename T, nint_t KPack>
VECOPS_ALWAYS_INLINE auto broadcast_k_group(
    const T* input, nint_t active_k) {
  using Tag = vec::ScalableTag<T, 0>;
  if constexpr (KPack == 1) {
    return vec::fill(Tag{}, active_k == 0 ? T{} : input[0]);
  } else {
    static_assert(KPack * sizeof(T) == sizeof(uint32_t));
    uint32_t word = 0;
    for (nint_t ki = 0; ki < active_k; ++ki) {
      const auto bits = bitcast<ScalarBits<T>>(input[ki]);
      word |= static_cast<uint32_t>(bits) << (ki * 8 * sizeof(T));
    }
    using WordTag = vec::ScalableTag<uint32_t, 0>;
    return vec::bitcast(
        Tag{}, WordTag{}, vec::fill(WordTag{}, word));
  }
}

template <gemm::Atom Atom, gemm::Operand Side, typename Source>
VECOPS_ALWAYS_INLINE auto load_shared_k_group(
    const Source& source, nint_t kg, nint_t logical_k) {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  constexpr nint_t KP = Packing::KPack;
  const T* pointer = nullptr;
  nint_t active_k = KP;
  if constexpr (is_packed_access_v<Atom, Side, Source>) {
    pointer = packed_block_pointer<Atom, Side, 0>(source, 0) +
        kg * source.raw_strides()[1];
  } else {
    static_assert(direct_row_major_input_v<Source>);
    pointer = reinterpret_cast<const T*>(source.raw_data()) + kg * KP;
    active_k = vec::details::sme::clamp_value(
        logical_k - kg * KP, nint_t{0}, KP);
  }
  return broadcast_k_group<T, KP>(pointer, active_k);
}

#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_DISABLE_RAW_SHARED_UNROLL) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_LEGACY_INLINE_RAW_SHARED_UNROLL)
using PackedGemvAliasBF16 = __bf16 __attribute__((__may_alias__));

/**
 * BF16 packed-varying/raw-shared GEMV leaf.
 *
 * Keep this leaf SME-backend-local until its packed ABI and widening-DOT
 * semantics have another backend consumer.  All tensor, direction, and Block
 * adaptation stays in the inline caller; the out-of-line body sees only raw
 * pointers and runtime extents.  Keep the live accumulator count static so the
 * hot K loop has no runtime block branches, but share each leaf between row
 * and column GEMV instead of cloning it for both tensor directions.
 */
template <int Blocks>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_OUT_OF_LINE_DISPATCH)
__attribute__((section(".vecops_kernel_text")))
#endif
void packed_gemv_bf16_raw_shared_k4_leaf(
    const bfloat16_t* shared,
    const bfloat16_t* packed0, const bfloat16_t* packed1,
    const bfloat16_t* packed2, const bfloat16_t* packed3,
    float32_t* output, nint_t varying_step,
    nint_t outputs, nint_t logical_k) {
  static_assert(1 <= Blocks && Blocks <= 4);
  const nint_t lanes = static_cast<nint_t>(svcntw());
  const nint_t groups = ceil_div(logical_k, nint_t{2});
  const nint_t full_groups = logical_k / 2;
  const auto all_b16 = svptrue_b16();

  auto sum0 = svdup_f32(0.0F);
  auto sum1 = svdup_f32(0.0F);
  auto sum2 = svdup_f32(0.0F);
  auto sum3 = svdup_f32(0.0F);

  nint_t kg = 0;
  VECOPS_LOOP_ALIGN(64) for (; kg + 4 <= full_groups; kg += 4) {
    const auto shared_quad = svld1rq_bf16(
        all_b16,
        reinterpret_cast<const PackedGemvAliasBF16*>(shared + 2 * kg));
    const auto accumulate4 = [&]<typename Sum>(
        Sum sum, const bfloat16_t* packed) VECOPS_INLINE_LAMBDA {
      const auto load = [&](nint_t group) VECOPS_INLINE_LAMBDA {
        return svld1_bf16(
            all_b16, reinterpret_cast<const PackedGemvAliasBF16*>(
                packed + group * varying_step));
      };
      const auto varying0 = load(kg + 0);
      const auto varying1 = load(kg + 1);
      const auto varying2 = load(kg + 2);
      const auto varying3 = load(kg + 3);
      sum = svbfdot_lane_f32(sum, varying0, shared_quad, 0);
      sum = svbfdot_lane_f32(sum, varying1, shared_quad, 1);
      sum = svbfdot_lane_f32(sum, varying2, shared_quad, 2);
      return svbfdot_lane_f32(sum, varying3, shared_quad, 3);
    };
    sum0 = accumulate4(sum0, packed0);
    if constexpr (Blocks >= 2) sum1 = accumulate4(sum1, packed1);
    if constexpr (Blocks >= 3) sum2 = accumulate4(sum2, packed2);
    if constexpr (Blocks >= 4) sum3 = accumulate4(sum3, packed3);
  }

  // At most three complete K-pairs plus one odd BF16 element remain.
  VECOPS_LOOP_ALIGN(64) for (; kg < groups; ++kg) {
    const nint_t k = 2 * kg;
    uint32_t shared_bits = static_cast<uint32_t>(
        bitcast<uint16_t>(shared[k]));
    if (k + 1 < logical_k) {
      shared_bits |= static_cast<uint32_t>(
          bitcast<uint16_t>(shared[k + 1])) << 16;
    }
    const auto shared_group = svreinterpret_bf16_u32(
        svdup_u32(shared_bits));
    const auto accumulate = [&](auto sum, const bfloat16_t* packed)
        VECOPS_INLINE_LAMBDA {
      return svbfdot_f32(
          sum, svld1_bf16(
                   all_b16, reinterpret_cast<const PackedGemvAliasBF16*>(
                       packed + kg * varying_step)),
          shared_group);
    };
    sum0 = accumulate(sum0, packed0);
    if constexpr (Blocks >= 2) sum1 = accumulate(sum1, packed1);
    if constexpr (Blocks >= 3) sum2 = accumulate(sum2, packed2);
    if constexpr (Blocks >= 4) sum3 = accumulate(sum3, packed3);
  }

  const auto store = [&](nint_t block, auto sum) VECOPS_INLINE_LAMBDA {
    const nint_t origin = block * lanes;
    const nint_t active = vec::details::sme::min_value(
        lanes, outputs - origin);
    svst1_f32(
        svwhilelt_b32_u64(0, static_cast<uint64_t>(active)),
        output + origin, sum);
  };
  store(0, sum0);
  if constexpr (Blocks >= 2) store(1, sum1);
  if constexpr (Blocks >= 3) store(2, sum2);
  if constexpr (Blocks >= 4) store(3, sum3);
}
#endif

template <bool VaryRows, gemm::Atom Atom, int Block,
          typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void streaming_packed_gemv_block(
    const A& a, const B& b, COutput& c_output,
    nint_t output_origin, nint_t outputs, nint_t logical_k) {
  static_assert(1 <= Block && Block <= 8);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  using VaryT = std::conditional_t<VaryRows, TA, TB>;
  using VaryTag = vec::ScalableTag<VaryT, 0>;
  using AccTag = std::conditional_t<
      std::same_as<VaryT, bfloat16_t>,
      vec::ViewAs<float32_t, VaryTag>,
      std::conditional_t<
          std::is_integral_v<VaryT>,
          vec::ViewAs<int32_t, VaryTag>,
          vec::ScalableTag<Acc, 0>>>;
  constexpr gemm::Operand VarySide =
      VaryRows ? gemm::Operand::A : gemm::Operand::B;
  constexpr gemm::Operand SharedSide =
      VaryRows ? gemm::Operand::B : gemm::Operand::A;
  const auto& varying = [&]() -> const auto& {
    if constexpr (VaryRows) return a;
    else return b;
  }();
  const auto& shared = [&]() -> const auto& {
    if constexpr (VaryRows) return b;
    else return a;
  }();
  constexpr nint_t KP = gemm::packing_t<Atom, VarySide>::KPack;
  const nint_t groups = ceil_div(logical_k, KP);
  const nint_t varying_step = varying.raw_strides()[1];
  const nint_t lanes = vec::size(AccTag{});
  const auto* packed0 = packed_block_pointer<Atom, VarySide, 0>(
      varying, output_origin);
  auto* packed1 = packed0;
  auto* packed2 = packed0;
  auto* packed3 = packed0;
  auto* packed4 = packed0;
  auto* packed5 = packed0;
  auto* packed6 = packed0;
  auto* packed7 = packed0;
  if constexpr (Block >= 2)
    packed1 = packed_block_pointer<Atom, VarySide, 1>(
        varying, output_origin);
  if constexpr (Block >= 3)
    packed2 = packed_block_pointer<Atom, VarySide, 2>(
        varying, output_origin);
  if constexpr (Block >= 4)
    packed3 = packed_block_pointer<Atom, VarySide, 3>(
        varying, output_origin);
  if constexpr (Block >= 5)
    packed4 = packed_block_pointer<Atom, VarySide, 4>(
        varying, output_origin);
  if constexpr (Block >= 6)
    packed5 = packed_block_pointer<Atom, VarySide, 5>(
        varying, output_origin);
  if constexpr (Block >= 7)
    packed6 = packed_block_pointer<Atom, VarySide, 6>(
        varying, output_origin);
  if constexpr (Block >= 8)
    packed7 = packed_block_pointer<Atom, VarySide, 7>(
        varying, output_origin);

  auto sum0 = vec::zeros(AccTag{});
  auto sum1 = vec::zeros(AccTag{});
  auto sum2 = vec::zeros(AccTag{});
  auto sum3 = vec::zeros(AccTag{});
  auto sum4 = vec::zeros(AccTag{});
  auto sum5 = vec::zeros(AccTag{});
  auto sum6 = vec::zeros(AccTag{});
  auto sum7 = vec::zeros(AccTag{});

  // TODO: Generalize this only after another backend needs the same packed
  // varying-side dot abstraction.  It intentionally remains SME-local.  The
  // K-major schedule keeps several output vectors live so one shared K group
  // feeds all of them instead of being reloaded by an output-major loop.
  nint_t kg = 0;
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_DISABLE_RAW_SHARED_UNROLL) && \
    defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_LEGACY_INLINE_RAW_SHARED_UNROLL)
  if constexpr (
      std::same_as<VaryT, bfloat16_t> &&
      direct_row_major_input_v<std::remove_cvref_t<decltype(shared)>>) {
    // TODO: Keep this four-group lane-DOT schedule SME-local until another
    // backend has the same raw-shared/packed-varying ABI.  ACL's SME1 GEMV
    // uses the same idea: one 128-bit replicated shared load feeds four
    // lane-indexed BFDOT groups, reducing broadcast and loop overhead.
    const nint_t full_groups = logical_k / KP;
    VECOPS_LOOP_ALIGN(64) for (; kg + 4 <= full_groups; kg += 4) {
      const auto* shared_pointer =
          reinterpret_cast<const bfloat16_t*>(shared.raw_data()) + kg * KP;
      const auto shared_quad = svreinterpret_bf16_u16(svld1rq_u16(
          svptrue_b16(),
          reinterpret_cast<const uint16_t*>(shared_pointer)));
      const auto accumulate4 = [&]<typename Sum>(
          Sum sum, const auto* packed) VECOPS_INLINE_LAMBDA {
        const auto one = [&]<int Lane>(Sum value) VECOPS_INLINE_LAMBDA {
          const auto varying_group = vec::load(
              VaryTag{}, packed + (kg + Lane) * varying_step);
          return vec::details::sve_basic_wrap_word<AccTag>(
              svbfdot_lane_f32(
                  vec::details::sve_basic_raw_word(value),
                  vec::details::sve_basic_raw_word(varying_group),
                  shared_quad, Lane));
        };
        sum = one.template operator()<0>(sum);
        sum = one.template operator()<1>(sum);
        sum = one.template operator()<2>(sum);
        return one.template operator()<3>(sum);
      };
      sum0 = accumulate4(sum0, packed0);
      if constexpr (Block >= 2) sum1 = accumulate4(sum1, packed1);
      if constexpr (Block >= 3) sum2 = accumulate4(sum2, packed2);
      if constexpr (Block >= 4) sum3 = accumulate4(sum3, packed3);
      if constexpr (Block >= 5) sum4 = accumulate4(sum4, packed4);
      if constexpr (Block >= 6) sum5 = accumulate4(sum5, packed5);
      if constexpr (Block >= 7) sum6 = accumulate4(sum6, packed6);
      if constexpr (Block >= 8) sum7 = accumulate4(sum7, packed7);
    }
  }
#endif
  VECOPS_LOOP_ALIGN(64) for (; kg < groups; ++kg) {
    const auto shared_group = load_shared_k_group<
        Atom, SharedSide>(shared, kg, logical_k);
    const auto accumulate = [&](auto sum, const auto* packed)
        VECOPS_INLINE_LAMBDA {
      const auto varying_group = vec::load(
          VaryTag{}, packed + kg * varying_step);
      if constexpr (std::same_as<VaryT, bfloat16_t>) {
        return skinny_widening_dot_add(
            VaryTag{}, sum,
            static_cast<vec::Vec<VaryTag>>(shared_group), varying_group);
      } else if constexpr (std::is_integral_v<VaryT>) {
        if constexpr (VaryRows) {
          return streaming_integer_dot_add<TA, TB>(
              sum, varying_group, shared_group);
        } else {
          return streaming_integer_dot_add<TA, TB>(
              sum, shared_group, varying_group);
        }
      } else {
        return vec::fmadd(
            static_cast<vec::Vec<VaryTag>>(shared_group),
            varying_group, sum);
      }
    };
    sum0 = accumulate(sum0, packed0);
    if constexpr (Block >= 2) sum1 = accumulate(sum1, packed1);
    if constexpr (Block >= 3) sum2 = accumulate(sum2, packed2);
    if constexpr (Block >= 4) sum3 = accumulate(sum3, packed3);
    if constexpr (Block >= 5) sum4 = accumulate(sum4, packed4);
    if constexpr (Block >= 6) sum5 = accumulate(sum5, packed5);
    if constexpr (Block >= 7) sum6 = accumulate(sum6, packed6);
    if constexpr (Block >= 8) sum7 = accumulate(sum7, packed7);
  }

  auto* output = reinterpret_cast<Acc*>(c_output.raw_data());
  const auto store = [&](int block, auto sum) VECOPS_INLINE_LAMBDA {
    const nint_t origin =
        output_origin + static_cast<nint_t>(block) * lanes;
    const nint_t active = vec::details::sme::min_value(
        lanes, outputs - origin);
    vec::store(
        AccTag{}, output + origin, sum,
        vec::opt::first(active));
  };
  store(0, sum0);
  if constexpr (Block >= 2) store(1, sum1);
  if constexpr (Block >= 3) store(2, sum2);
  if constexpr (Block >= 4) store(3, sum3);
  if constexpr (Block >= 5) store(4, sum4);
  if constexpr (Block >= 6) store(5, sum5);
  if constexpr (Block >= 7) store(6, sum6);
  if constexpr (Block >= 8) store(7, sum7);
}

template <bool VaryRows, gemm::Atom Atom,
          typename A, typename B, typename COutput>
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) || \
    (!defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_DISABLE_RAW_SHARED_UNROLL) && \
     !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_LEGACY_INLINE_RAW_SHARED_UNROLL))
VECOPS_ALWAYS_INLINE
#else
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#endif
void streaming_packed_gemv(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  using Acc = typename Atom::TAcc;
  using AccTag = vec::ScalableTag<Acc, 0>;
  const nint_t lanes = vec::size(AccTag{});
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_DISABLE_RAW_SHARED_UNROLL) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_LEGACY_INLINE_RAW_SHARED_UNROLL)
  constexpr gemm::Operand VarySide =
      VaryRows ? gemm::Operand::A : gemm::Operand::B;
  const auto& varying = [&]() -> const auto& {
    if constexpr (VaryRows) return a;
    else return b;
  }();
  const auto& shared = [&]() -> const auto& {
    if constexpr (VaryRows) return b;
    else return a;
  }();
  using Shared = std::remove_cvref_t<decltype(shared)>;
  if constexpr (
      std::same_as<Atom, gemm::SME_BF16F32> &&
      direct_row_major_input_v<Shared>) {
    const nint_t blocks = ceil_div(outputs, lanes);
    const auto* packed0 = packed_block_pointer<Atom, VarySide, 0>(
        varying, 0);
    auto* packed1 = packed0;
    auto* packed2 = packed0;
    auto* packed3 = packed0;
    if (blocks >= 2)
      packed1 = packed_block_pointer<Atom, VarySide, 1>(varying, 0);
    if (blocks >= 3)
      packed2 = packed_block_pointer<Atom, VarySide, 2>(varying, 0);
    if (blocks >= 4)
      packed3 = packed_block_pointer<Atom, VarySide, 3>(varying, 0);
    const auto* shared_pointer =
        reinterpret_cast<const bfloat16_t*>(shared.raw_data());
    auto* output = reinterpret_cast<float32_t*>(c_output.raw_data());
    const nint_t varying_step = varying.raw_strides()[1];
    switch (blocks) {
      case 4:
        packed_gemv_bf16_raw_shared_k4_leaf<4>(
            shared_pointer, packed0, packed1, packed2, packed3,
            output, varying_step, outputs, logical_k);
        break;
      case 3:
        packed_gemv_bf16_raw_shared_k4_leaf<3>(
            shared_pointer, packed0, packed1, packed2, packed3,
            output, varying_step, outputs, logical_k);
        break;
      case 2:
        packed_gemv_bf16_raw_shared_k4_leaf<2>(
            shared_pointer, packed0, packed1, packed2, packed3,
            output, varying_step, outputs, logical_k);
        break;
      default:
        packed_gemv_bf16_raw_shared_k4_leaf<1>(
            shared_pointer, packed0, packed1, packed2, packed3,
            output, varying_step, outputs, logical_k);
        break;
    }
  } else {
#endif
    nint_t origin = 0;
    VECOPS_LOOP_ALIGN(64) for (;
         origin + 8 * lanes <= outputs; origin += 8 * lanes) {
      streaming_packed_gemv_block<VaryRows, Atom, 8>(
          a, b, c_output, origin, outputs, logical_k);
    }
    if (origin + 4 * lanes <= outputs) {
      streaming_packed_gemv_block<VaryRows, Atom, 4>(
          a, b, c_output, origin, outputs, logical_k);
      origin += 4 * lanes;
    }
    const nint_t remaining = outputs - origin;
    if (remaining <= 0) return;
    switch (ceil_div(remaining, lanes)) {
      case 4:
        streaming_packed_gemv_block<VaryRows, Atom, 4>(
            a, b, c_output, origin, outputs, logical_k);
        break;
      case 3:
        streaming_packed_gemv_block<VaryRows, Atom, 3>(
            a, b, c_output, origin, outputs, logical_k);
        break;
      case 2:
        streaming_packed_gemv_block<VaryRows, Atom, 2>(
            a, b, c_output, origin, outputs, logical_k);
        break;
      default:
        streaming_packed_gemv_block<VaryRows, Atom, 1>(
            a, b, c_output, origin, outputs, logical_k);
        break;
    }
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_DISABLE_RAW_SHARED_UNROLL) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_LEGACY_INLINE_RAW_SHARED_UNROLL)
  }
#endif
}

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool streaming_packed_gemv_candidate_v = [] {
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES)
  constexpr bool SupportedFloat =
      (std::same_as<typename Atom::TA, bfloat16_t> &&
       std::same_as<typename Atom::TB, bfloat16_t> &&
       std::same_as<typename Atom::TAcc, float32_t>) ||
      (std::same_as<typename Atom::TA, float32_t> &&
       std::same_as<typename Atom::TB, float32_t> &&
       std::same_as<typename Atom::TAcc, float32_t>) ||
      (std::same_as<typename Atom::TA, float64_t> &&
       std::same_as<typename Atom::TB, float64_t> &&
       std::same_as<typename Atom::TAcc, float64_t>);
  constexpr bool SupportedInt =
      sizeof(typename Atom::TA) == 1 && sizeof(typename Atom::TB) == 1 &&
      std::same_as<typename Atom::TAcc, int32_t>;
#if !defined(__ARM_FEATURE_SVE_MATMUL_INT8)
  constexpr bool SupportedMixedInt =
      std::same_as<typename Atom::TA, typename Atom::TB>;
#else
  constexpr bool SupportedMixedInt = true;
#endif
#else
  constexpr bool SupportedFloat =
      std::same_as<typename Atom::TA, bfloat16_t> &&
      std::same_as<typename Atom::TB, bfloat16_t> &&
      std::same_as<typename Atom::TAcc, float32_t>;
  constexpr bool SupportedInt = false;
  constexpr bool SupportedMixedInt = false;
#endif
  constexpr bool APacked =
      is_packed_access_v<Atom, gemm::Operand::A, A>;
  constexpr bool BPacked =
      is_packed_access_v<Atom, gemm::Operand::B, B>;
  constexpr bool SupportedTypes =
      SupportedFloat || (SupportedInt && SupportedMixedInt);
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES)
  constexpr bool SupportedPacking =
      (APacked && (BPacked || direct_row_major_input_v<B>)) ||
      (BPacked && (APacked || direct_row_major_input_v<A>));
#else
  // The measured no-regression production candidate is BF16 with only the
  // varying side packed. PackedAB already has an efficient MOPA path, while
  // F32/F64 and wider-output BF16 need more tuning before they can be enabled.
  constexpr bool SupportedPacking =
      (APacked && !BPacked && direct_row_major_input_v<B>) ||
      (BPacked && !APacked && direct_row_major_input_v<A>);
#endif
  return SupportedTypes &&
      IsZeroTransform<typename CInput::Transform>::value &&
      direct_row_major_output_v<COutput> &&
      std::same_as<typename COutput::ComputeType, typename Atom::TAcc> &&
      SupportedPacking;
}();

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_OUT_OF_LINE_DISPATCH)
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
__attribute__((section(".vecops_kernel_text")))
#else
VECOPS_ALWAYS_INLINE
#endif
bool try_streaming_packed_gemv(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput&, COutput& c_output) {
  if constexpr (!streaming_packed_gemv_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES)
    const nint_t packed_panel = [&] {
      if constexpr (is_packed_access_v<Atom, gemm::Operand::A, A>)
        return static_cast<nint_t>(a.spec().input_layout().shape()[2]);
      else
        return static_cast<nint_t>(b.spec().input_layout().shape()[2]);
    }();
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    if (ordinary_vl_bytes !=
        packed_panel * static_cast<nint_t>(sizeof(bfloat16_t))) {
      return false;
    }
#endif
    if constexpr (is_packed_access_v<Atom, gemm::Operand::B, B>) {
      if (logical_m == 1 && logical_n > 0
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES)
          && logical_n <= 64
#endif
      ) {
        streaming_packed_gemv<false, Atom>(
            a, b, c_output, logical_n, logical_k);
        return true;
      }
    }
    if constexpr (is_packed_access_v<Atom, gemm::Operand::A, A>) {
      if (logical_n == 1 && logical_m > 0 &&
          c_output.raw_strides()[0] == 1
#if !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES)
          && logical_m <= 64
#endif
      ) {
        streaming_packed_gemv<true, Atom>(
            a, b, c_output, logical_m, logical_k);
        return true;
      }
    }
    return false;
  }
}
#endif

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
    const nint_t active = std::min(vec::size(InputTag{}), remaining);
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
  const nint_t output_stride = c_output.raw_strides()[0];
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

#if defined(VECOPS_HAS_SME_MIXED_SIGN_SKINNY_EXTERNAL_LEAF) && \
    defined(__ARM_FEATURE_SVE_MATMUL_INT8)
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
#if defined(VECOPS_HAS_SME_FUSED_F64_EXTERNAL_LEAF) && \
    defined(HAS_SME_F64F64)
void sve_skinny_fused_compute_f64_row(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);

void sve_skinny_fused_compute_f64_col(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);
#endif

#if defined(VECOPS_HAS_SME_RUNTIME_QUANT_INT8_EXTERNAL_LEAF) && \
    defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
void fused_runtime_quant_int8_packed_b_gemv_1x4vl(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale);
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
    const nint_t active = std::min(vec::size(InputTag{}), remaining);
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
      const nint_t active = std::min(lanes, outputs - output);
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

#if defined(VECOPS_HAS_SME_FUSED_F64_EXTERNAL_LEAF) && \
    defined(HAS_SME_F64F64)
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
      const nint_t active = std::min(lanes, outputs - output);
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

#if defined(VECOPS_HAS_SME_RUNTIME_QUANT_INT8_EXTERNAL_LEAF) && \
    defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool fused_runtime_quant_int8_candidate_v =
    std::same_as<Atom, gemm::SME_I8I32<uint8_t, int8_t>> &&
    A::Rank == 2 && B::Rank == 4 &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    std::same_as<typename A::MemoryElement, float32_t> &&
    std::same_as<typename A::ComputeType, uint8_t> &&
    is_runtime_per_row_asymmetric_quantize_transform_v<
        typename A::Transform> &&
    is_packed_access_v<Atom, gemm::Operand::B, B> &&
    std::same_as<typename B::ComputeType, int8_t> &&
    std::same_as<typename CInput::MemoryElement, int32_t> &&
    std::same_as<typename CInput::ComputeType, int32_t> &&
    std::same_as<typename CInput::Transform, tensor::NoTransform> &&
    std::same_as<typename COutput::MemoryElement, float32_t> &&
    std::same_as<typename COutput::ComputeType, int32_t> &&
    is_runtime_per_row_column_dequantize_transform_v<
        typename COutput::Transform>;

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) bool
try_fused_runtime_quant_int8_packed_b_gemv(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output) {
  if constexpr (!fused_runtime_quant_int8_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    using ATag = vec::ScalableTag<uint8_t, 0>;
    using AccTag = vec::ScalableTag<int32_t, 0>;
    constexpr nint_t KChunk = 64;
    const nint_t output_block = 4 * vec::size(AccTag{});
    const nint_t ordinary_vl_bytes = vec::size(ATag{});
    const nint_t packed_panel = static_cast<nint_t>(
        b.spec().input_layout().shape()[2]);
    if (logical_m != 1 || logical_n <= 0 || logical_k < 0 ||
        logical_n % output_block != 0 || logical_k % KChunk != 0 ||
        ordinary_vl_bytes != KChunk ||
        ordinary_vl_bytes != 2 * packed_panel) {
      return false;
    }

    const auto a_strides = a.raw_strides();
    const auto b_strides = b.raw_strides();
    const auto correction_strides = c_input.raw_strides();
    const auto output_strides = c_output.raw_strides();
    if (a_strides[1] != 1 || correction_strides[1] != 1 ||
        output_strides[1] != 1) {
      return false;
    }

    const auto local_origin = tensor::coord(nint_t{0}, nint_t{0});
    const auto a_original =
        a.spec().projection().project(local_origin);
    const auto output_original =
        c_output.spec().projection().project(local_origin);
    const auto& quant = a.spec().transform().parameters();
    const auto& dequant = c_output.spec().transform().parameters();
    const nint_t quant_row = quant.row_scale_index(a_original);
    const nint_t dequant_row = dequant.row_scale_index(output_original);
    const nint_t column =
        output_original[output_original.size() - 1];

    fused_runtime_quant_int8_packed_b_gemv_1x4vl(
        reinterpret_cast<const float32_t*>(a.raw_data()),
        reinterpret_cast<const int8_t*>(b.raw_data()),
        b_strides[0], b_strides[1], b_strides[2], packed_panel,
        reinterpret_cast<const int32_t*>(c_input.raw_data()),
        reinterpret_cast<float32_t*>(c_output.raw_data()),
        dequant.column_scales + column,
        logical_n, logical_k,
        quant.multipliers[quant_row], *quant.zero_point,
        dequant.row_scales[dequant_row]);
    return true;
  }
}
#endif

template <gemm::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_fused_output_supported_v =
    COutput::Transform::is_elementwise ||
#if defined(VECOPS_EXPERIMENTAL_SME_FUSED_LANE_LOCAL) && \
    !defined(VECOPS_DISABLE_SME_FUSED_LANE_LOCAL)
    COutput::Transform::is_lane_local;
#else
    false;
#endif

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
#if defined(VECOPS_HAS_SME_FUSED_F64_EXTERNAL_LEAF) && \
    defined(HAS_SME_F64F64)
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

    const nint_t a_panel = a.spec().input_layout().shape()[2];
    const nint_t b_panel = b.spec().input_layout().shape()[2];
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    // Preserve the exact X24 store sequence. Padded output rows need a
    // separate vertical store strategy and were not part of its performance
    // or correctness evidence.
    if (c_output.raw_strides()[0] != logical_n) return false;

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        a.raw_strides()[1],
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        b.raw_strides()[1],
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

    const nint_t a_panel = a.spec().input_layout().shape()[2];
    const nint_t b_panel = b.spec().input_layout().shape()[2];
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    if (c_output.raw_strides()[0] != logical_n) return false;

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        a.raw_strides()[1],
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        b.raw_strides()[1],
        reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data()),
        logical_m, logical_n, logical_k);
    return true;
  }
}
#endif

template <gemm::Atom Atom, int NM, int NN>
VECOPS_ALWAYS_INLINE bool prefer_large_packed_prefetch(
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
  if constexpr (
      std::same_as<Atom, gemm::SME_BF16F32> && NM == 2 && NN == 2) {
    // X37/X43 found that A+B L2 prefetch at distance 48 is profitable only
    // once the complete packed operands exceed the private-cache working
    // set. Keep short-N and small-output long-K on the original loop.
    constexpr uint64_t MinPackedBytes = uint64_t{2} * 1024 * 1024;
    if (logical_m > 0 && logical_n > 0 && logical_k > 0) {
      const uint64_t bytes_per_k =
          static_cast<uint64_t>(logical_m) * sizeof(typename Atom::TA) +
          static_cast<uint64_t>(logical_n) * sizeof(typename Atom::TB);
      const uint64_t min_bytes_per_k =
          (MinPackedBytes + static_cast<uint64_t>(logical_k) - 1) /
          static_cast<uint64_t>(logical_k);
      return bytes_per_k >= min_bytes_per_k;
    }
  }
#else
  (void)logical_m;
  (void)logical_n;
  (void)logical_k;
#endif
  return false;
}

template <gemm::Atom Atom, int NM, int NN, typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_packed_groups(
    const A& a, const B& b, nint_t m, nint_t n, nint_t logical_k,
    bool prefetch_large_working_set = false) {
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
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE)
    if constexpr (std::same_as<Atom, gemm::SME_BF16F32>) {
      // Experimental two-bank schedule.  This generates the intended
      // load/MOPA interleave without loop-carried vector moves, but remains
      // opt-in: isolated square GEMM wins while the full packed suite is
      // sensitive to the larger Tile2D specialization's code layout.
      auto* pa0 = a0;
      auto* pa1 = a1;
      auto* pb0 = b0;
      auto* pb1 = b1;
      nint_t pairs = groups / 2;
      VECOPS_LOOP_ALIGN(64) for (; pairs > 0; --pairs) {
        const auto av0 = vec::load(ATag{}, pa0);
        const auto av1 = vec::load(ATag{}, pa1);
        const auto bv0 = vec::load(BTag{}, pb0);
        const auto bv1 = vec::load(BTag{}, pb1);
        const auto next_av0 = vec::load(ATag{}, pa0 + a_step);
        mopa<Atom, 0>(av0, bv0);
        const auto next_av1 = vec::load(ATag{}, pa1 + a_step);
        mopa<Atom, 1>(av0, bv1);
        const auto next_bv0 = vec::load(BTag{}, pb0 + b_step);
        mopa<Atom, 2>(av1, bv0);
        const auto next_bv1 = vec::load(BTag{}, pb1 + b_step);
        mopa<Atom, 3>(av1, bv1);
        mopa<Atom, 0>(next_av0, next_bv0);
        mopa<Atom, 1>(next_av0, next_bv1);
        mopa<Atom, 2>(next_av1, next_bv0);
        mopa<Atom, 3>(next_av1, next_bv1);
        pa0 += 2 * a_step;
        pa1 += 2 * a_step;
        pb0 += 2 * b_step;
        pb1 += 2 * b_step;
      }
      if (groups % 2 != 0) {
        const auto av0 = vec::load(ATag{}, pa0);
        const auto av1 = vec::load(ATag{}, pa1);
        const auto bv0 = vec::load(BTag{}, pb0);
        const auto bv1 = vec::load(BTag{}, pb1);
        mopa<Atom, 0>(av0, bv0);
        mopa<Atom, 1>(av0, bv1);
        mopa<Atom, 2>(av1, bv0);
        mopa<Atom, 3>(av1, bv1);
      }
    } else
#endif
    {
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
      constexpr nint_t PrefetchDistance = 48;
      nint_t kg = 0;
      if (prefetch_large_working_set && groups > PrefetchDistance) {
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
    if constexpr (
        std::is_floating_point_v<T> &&
        std::same_as<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>,
            meta::Const<0>>) {
      const auto value = vec::load(
          Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
      const auto pg_rows = vec::mwhilelt(Tag{}, nint_t{0}, active_m);
      vec::details::sme::mopa<Tile, T, T, T>(
          pg_rows, pg, vec::fill(Tag{}, T{1}), value);
    } else if constexpr (
        (std::same_as<T, int32_t> || std::same_as<T, uint32_t>) &&
        std::same_as<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>,
            meta::Const<0>>) {
      if (active_m == 1) {
        using U = std::conditional_t<std::same_as<T, int32_t>, uint32_t, T>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            0, static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base));
      } else {
        const auto value = vec::load(
            Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
        const auto pg_rows = vec::mwhilelt(Tag{}, nint_t{0}, active_m);
        vec::details::sme::addha<Tile, T>(pg_rows, pg, value);
      }
    } else {
      for (nint_t row = 0; row < active_m; ++row) {
        using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            static_cast<uint32_t>(row),
            static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base + row * strides[0]));
      }
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
    const A& a, const OperandInvariants<Atom, gemm::Operand::A, A>& a_invariants,
    const B& b, const OperandInvariants<Atom, gemm::Operand::B, B>& b_invariants,
    nint_t m, nint_t n, nint_t kg,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  if constexpr (NN == 1) {
    const auto bv = load_operand<Atom, gemm::Operand::B, 0, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<Atom, gemm::Operand::A, MI, FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      mopa<Atom, MI>(av, bv);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else if constexpr (NM == 1) {
    const auto av = load_operand<Atom, gemm::Operand::A, 0, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<Atom, gemm::Operand::B, NI, FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(av, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  } else if constexpr (NM == 2 && NN == 2) {
    const auto a0 = load_operand<Atom, gemm::Operand::A, 0, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<Atom, gemm::Operand::A, 1, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto b0 = load_operand<Atom, gemm::Operand::B, 0, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<Atom, gemm::Operand::B, 1, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    mopa<Atom, 0>(a0, b0);
    mopa<Atom, 1>(a0, b1);
    mopa<Atom, 2>(a1, b0);
    mopa<Atom, 3>(a1, b1);
  } else if constexpr (NN == 2) {
    const auto b0 = load_operand<Atom, gemm::Operand::B, 0, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<Atom, gemm::Operand::B, 1, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<Atom, gemm::Operand::A, MI, FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      mopa<Atom, 2 * MI>(av, b0);
      mopa<Atom, 2 * MI + 1>(av, b1);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else {
    static_assert(NM == 2 && NN <= 4);
    const auto a0 = load_operand<Atom, gemm::Operand::A, 0, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<Atom, gemm::Operand::A, 1, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<Atom, gemm::Operand::B, NI, FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(a0, bv);
      mopa<Atom, NN + NI>(a1, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
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
  const OperandInvariants<Atom, gemm::Operand::A, A> a_invariants(a);
  const OperandInvariants<Atom, gemm::Operand::B, B> b_invariants(b);
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
    if constexpr (KP == 1 || std::same_as<Atom, gemm::SME_BF16F32> ||
                  std::same_as<Atom, gemm::SME_F16F32>) {
      const nint_t full_groups = logical_k / KP;
      for (nint_t kg = 0; kg < full_groups; ++kg) {
        compute_group<Atom, NM, NN, true>(
            a, a_invariants, b, b_invariants,
            m, n, kg, logical_m, logical_n, logical_k);
      }
      if constexpr (KP > 1) {
        if (full_groups * KP < logical_k) {
          compute_group<Atom, NM, NN, false>(
              a, a_invariants, b, b_invariants,
              m, n, full_groups,
              logical_m, logical_n, logical_k);
        }
      }
    } else {
      const nint_t groups = ceil_div(logical_k, KP);
      for (nint_t kg = 0; kg < groups; ++kg) {
        compute_group<Atom, NM, NN, false>(
            a, a_invariants, b, b_invariants,
            m, n, kg,
            logical_m, logical_n, logical_k);
      }
    }
  };

  if constexpr (FastPacked && ExactBlocks && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    const bool prefetch_large_working_set =
        prefer_large_packed_prefetch<Atom, NM, NN>(
            logical_m, logical_n, logical_k);
    compute_packed_groups<Atom, NM, NN>(
        a, b, m, n, logical_k, prefetch_large_working_set);
  } else if constexpr (FastPacked && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, gemm::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, gemm::Operand::B, B>);
    const bool all_logical_blocks_exist =
        active_m > static_cast<nint_t>(NM - 1) * lanes &&
        active_n > static_cast<nint_t>(NN - 1) * lanes;
    if (all_logical_blocks_exist) {
      const bool prefetch_large_working_set =
          prefer_large_packed_prefetch<Atom, NM, NN>(
              logical_m, logical_n, logical_k);
      compute_packed_groups<Atom, NM, NN>(
          a, b, m, n, logical_k, prefetch_large_working_set);
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
#if defined(HAS_SME_F64F64)
      std::conditional_t<
          std::same_as<Atom, gemm::SME_F64F64> &&
              !(sme::is_packed_access_v<Atom, gemm::Operand::A, A> &&
                sme::is_packed_access_v<Atom, gemm::Operand::B, B>),
          kernel::loop::tile2d_policy::RuntimeExactArea8,
#endif
      std::conditional_t<
          sme::prefer_constraint_area4_v<Atom, M, N, K, A, B>,
          kernel::loop::tile2d_policy::FourRegions,
          kernel::loop::tile2d_policy::RuntimeExactArea4>
#if defined(HAS_SME_F64F64)
          >,
#endif
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
#if defined(VECOPS_HAS_SME_MIXED_SIGN_SKINNY_EXTERNAL_LEAF) && \
    defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::mixed_sign_sve_skinny_candidate_v<
            Atom, A, B, CInput, COutput>) {
      if (sme::try_mixed_sign_sve_skinny<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::sve_skinny_candidate_v<Atom, A, B, CInput, COutput>) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, gemm::SME_F16F32> ? 16 : 64;
      if (logical_m == 1 &&
          0 <= logical_n && logical_n <= MaxOutputs) {
        sme::sve_skinny_matmul<false>(
            a, b, c_output, logical_n, logical_k);
        return;
      }
      if (logical_n == 1 &&
          0 <= logical_m && logical_m <= MaxOutputs) {
        sme::sve_skinny_matmul<true>(
            a, b, c_output, logical_m, logical_k);
        return;
      }
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::sve_skinny_fused_candidate_v<Atom, A, B, CInput, COutput>) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, gemm::SME_F16F32> ? 16 : 64;
      if (logical_m == 1 &&
          0 <= logical_n && logical_n <= MaxOutputs) {
#if defined(VECOPS_HAS_SME_FUSED_F64_EXTERNAL_LEAF) && \
    defined(HAS_SME_F64F64)
        if constexpr (std::same_as<Atom, gemm::SME_F64F64>) {
          sme::sve_skinny_fused_matmul_f64_external<false>(
              a, b, c_output, logical_n, logical_k);
        } else
#endif
        {
          if constexpr (COutput::Transform::is_elementwise) {
            sme::sve_skinny_fused_matmul<false>(
                a, b, c_output, logical_n, logical_k);
          } else {
            sme::sve_skinny_fused_lane_local_matmul<false>(
                a, b, c_output, logical_n, logical_k);
          }
        }
        return;
      }
      if (logical_n == 1 &&
          0 <= logical_m && logical_m <= MaxOutputs) {
#if defined(VECOPS_HAS_SME_FUSED_F64_EXTERNAL_LEAF) && \
    defined(HAS_SME_F64F64)
        if constexpr (std::same_as<Atom, gemm::SME_F64F64>) {
          sme::sve_skinny_fused_matmul_f64_external<true>(
              a, b, c_output, logical_m, logical_k);
        } else
#endif
        {
          if constexpr (COutput::Transform::is_elementwise) {
            sme::sve_skinny_fused_matmul<true>(
                a, b, c_output, logical_m, logical_k);
          } else {
            sme::sve_skinny_fused_lane_local_matmul<true>(
                a, b, c_output, logical_m, logical_k);
          }
        }
        return;
      }
    }
#endif
#if defined(VECOPS_HAS_SME_RUNTIME_QUANT_INT8_EXTERNAL_LEAF) && \
    defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::fused_runtime_quant_int8_candidate_v<
            Atom, A, B, CInput, COutput>) {
      if (sme::try_fused_runtime_quant_int8_packed_b_gemv<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    }
#endif
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::packed_mmla_candidate_v<Atom, A, B, CInput, COutput>) {
      if (sme::try_packed_mmla<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
      if (sme::try_packed_mmla_tiny<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output)) {
        return;
      }
    }
#endif
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV) && \
    !defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    defined(HAS_SME_FA64)
    if constexpr (
        !execution::has_resource_v<
            execution::details::arm::StreamingZA, Scope> &&
        sme::streaming_packed_gemv_candidate_v<
            Atom, A, B, CInput, COutput>) {
      if (sme::try_streaming_packed_gemv<Atom>(
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
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    defined(HAS_SME_FA64)
            if (sme::try_streaming_packed_gemv<Atom>(
                    static_cast<nint_t>(m), static_cast<nint_t>(n),
                    static_cast<nint_t>(k),
                    a, b, c_input, c_output)) {
              return;
            }
#endif
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
#if defined(VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES) && \
    defined(HAS_SME_FA64)
            if (sme::try_streaming_packed_gemv<Atom>(
                    static_cast<nint_t>(m), static_cast<nint_t>(n),
                    static_cast<nint_t>(k), active_a, active_b,
                    active_c_input, active_c_output)) {
              active_c_output.commit();
              return;
            }
#endif
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
