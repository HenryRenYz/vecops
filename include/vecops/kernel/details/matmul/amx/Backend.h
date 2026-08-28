//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#include "vecops/execution/details/x86/Resources.h"
#include "vecops/gemm/Packing.h"
#include "vecops/gemm/details/amx/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/matmul/Traversal.h"
#include "vecops/kernel/details/matmul_pack/amx/Pack.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/details/amx/AMX.h"

namespace vecops::kernel::matmul_details::amx {

namespace generic = matmul_pack_details::generic;
namespace tile = kernel::loop;
namespace amx_intrinsics = vec::details::amx;

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

/** One TILECFG image works for every supported AMX micro-kernel shape. */
struct Configuration : execution::details::x86::TileConfiguration {
  Configuration() {
    for (int i = 0; i < 8; ++i) {
      column_bytes[i] = 64;
      rows[i] = 16;
    }
  }
};

struct KernelProvider {
  template <int A, int B, tile::Tile2DMaskMode, tile::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B + A + B <= 8) {
      return 100 * A * B + 4 * (A + B);
    } else {
      return -1;
    }
  }
};

using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider, tile::Tile2DSearchSpace<3, 3, 4>>;

template <typename T>
consteval int sixteen_lane_power() {
  nint_t bytes = 16 * static_cast<nint_t>(sizeof(T));
  nint_t native = VEC_WIDTH / 8;
  int power = 0;
  while (bytes < native) {
    bytes *= 2;
    --power;
  }
  while (bytes > native) {
    native *= 2;
    ++power;
  }
  return power;
}

template <nint_t KPack, typename T>
VECOPS_ALWAYS_INLINE void pack_b_full_panel_direct(
    const T* source, nint_t row_stride, T* destination, nint_t k_tile) {
  static_assert(KPack == 2 || KPack == 4);
  static_assert(sizeof(T) * KPack == sizeof(uint32_t));
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i offsets = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int32_t>(row_stride)));
  auto* output = reinterpret_cast<__m512i*>(destination);
  for (nint_t kg = 0; kg < k_tile / KPack; ++kg) {
    const __m512i words = _mm512_i32gather_epi32(
        offsets, source + kg * KPack, sizeof(T));
    _mm512_storeu_si512(output + kg, words);
  }
}

template <gemm::Atom Atom, bool SpatialGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE const typename Atom::TA* prepare_a(
    const Source& source, nint_t m, nint_t k, nint_t logical_m,
    nint_t logical_k, typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (is_packed_access_v<Atom, gemm::Operand::A, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed AMX A must be direct and untransformed");
    if constexpr (!SpatialGuaranteed) {
      if (m >= logical_m) {
        std::fill_n(buffer, 16 * KR, T{});
        return buffer;
      }
    }
    const auto& layout = source.spec().input_layout();
    const nint_t offset = tensor::offset_at(layout, m / 16, k / KR, 0, 0);
    return reinterpret_cast<const T*>(source.raw_data()) + offset;
  } else {
    static_assert(Source::Rank == 2, "unpacked AMX A must be rank two");
    using Tag = vec::ScalableTag<T, 0>;
    const nint_t active_k = std::clamp(logical_k - k, nint_t{0}, KR);
    for (nint_t row = 0; row < 16; ++row) {
      const nint_t logical_row = m + row;
      const auto value = [&] VECOPS_INLINE_LAMBDA {
        if constexpr (SpatialGuaranteed) {
          return source.load(
              Tag{}, tensor::coord(logical_row, k), tensor::axis<1>,
              vec::opt::first(active_k), vec::opt::zero);
        } else {
          return logical_row < logical_m
              ? source.load(
                    Tag{}, tensor::coord(logical_row, k), tensor::axis<1>,
                    vec::opt::first(active_k), vec::opt::zero)
              : vec::zeros(Tag{});
        }
      }();
      vec::store(Tag{}, buffer + row * KR, value);
    }
    return buffer;
  }
}

template <gemm::Atom Atom, bool SpatialGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE const typename Atom::TB* prepare_b(
    const Source& source, nint_t n, nint_t k, nint_t logical_n,
    nint_t logical_k, typename Atom::TB* buffer) {
  using T = typename Atom::TB;
  using Packing = gemm::packing_t<Atom, gemm::Operand::B>;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  constexpr nint_t KP = Packing::KPack;
  if constexpr (is_packed_access_v<Atom, gemm::Operand::B, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed AMX B must be direct and untransformed");
    if constexpr (!SpatialGuaranteed) {
      if (n >= logical_n) {
        std::fill_n(buffer, 16 * KR, T{});
        return buffer;
      }
    }
    const auto& layout = source.spec().input_layout();
    const nint_t offset = tensor::offset_at(
        layout, n / 16, k / KR, 0, 0, 0);
    return reinterpret_cast<const T*>(source.raw_data()) + offset;
  } else {
    static_assert(Source::Rank == 2, "unpacked AMX B must be rank two");
    if constexpr (direct_row_major_input_v<Source>) {
      const auto strides = source.raw_strides();
      const nint_t active_n = SpatialGuaranteed
          ? nint_t{16}
          : std::clamp(logical_n - n, nint_t{0}, nint_t{16});
      const nint_t active_k = std::clamp(
          logical_k - k, nint_t{0}, KR);
      const bool gather_offsets_fit =
          strides[0] >= std::numeric_limits<int32_t>::min() / 15 &&
          strides[0] <= std::numeric_limits<int32_t>::max() / 15;
      // Reuse the AMX packer's indexed word gathers for the common direct
      // row-major path.  A complete KPack is required because its tail
      // handling deliberately backs the word load up to stay in bounds.
      if (gather_offsets_fit && active_n == 16 && active_k == KR) {
        const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
            n * strides[0] + k;
        pack_b_full_panel_direct<KP>(
            pointer, strides[0], buffer, KR);
        return buffer;
      }
      if (gather_offsets_fit && active_n > 0 && active_k >= KP) {
        const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
            n * strides[0] + k;
        matmul_pack_details::amx::pack_b_direct<KP>(
            pointer, strides[0], buffer, active_n, active_k, 16, KR);
        return buffer;
      }
    }
    using Tag = vec::ScalableTag<T, sixteen_lane_power<T>()>;
    for (nint_t kg = 0; kg < KR / KP; ++kg) {
      auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
        const nint_t kk = k + kg * KP + ki;
        const nint_t active_n = std::clamp(
            logical_n - n, nint_t{0}, nint_t{16});
        if constexpr (SpatialGuaranteed) {
          return kk < logical_k
              ? source.load(
                    Tag{}, tensor::coord(n, kk), tensor::axis<0>,
                    vec::opt::first(active_n), vec::opt::zero)
              : vec::zeros(Tag{});
        } else {
          return kk < logical_k && active_n > 0
              ? source.load(
                    Tag{}, tensor::coord(n, kk), tensor::axis<0>,
                    vec::opt::first(active_n), vec::opt::zero)
              : vec::zeros(Tag{});
        }
      };
      if constexpr (KP == 2) {
        using PackedTag = vec::Twice<Tag>;
        const auto packed = generic::interleave_pair<Tag>(load_k(0), load_k(1));
        vec::store(PackedTag{}, buffer + kg * 16 * KP, packed);
      } else {
        static_assert(KP == 4);
        using PackedTag = vec::Twice<vec::Twice<Tag>>;
        const auto packed = generic::interleave_quad<Tag>(
            load_k(0), load_k(1), load_k(2), load_k(3));
        vec::store(PackedTag{}, buffer + kg * 16 * KP, packed);
      }
    }
    return buffer;
  }
}

template <int Tile, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    nint_t active_m, nint_t active_n,
    typename CInput::ComputeType* buffer) {
  using T = typename CInput::ComputeType;
  using Transform = typename CInput::Transform;
  // A resident family may be wider than a boundary region.  Such completely
  // inactive accumulator tiles still participate in the AMX dot instruction,
  // so initialize them to zero without asking DataAccess for an out-of-range
  // base coordinate.
  if (active_m == 0 || active_n == 0) {
    amx_intrinsics::zero<Tile>();
    return;
  }
  if constexpr (IsZeroTransform<Transform>::value) {
    amx_intrinsics::zero<Tile>();
  } else {
    if constexpr (direct_row_major_input_v<CInput>) {
      if (active_m == 16 && active_n == 16) {
        const auto strides = input.raw_strides();
        const auto* pointer = reinterpret_cast<const T*>(input.raw_data()) +
            m * strides[0] + n;
        amx_intrinsics::load<Tile>(
            pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
        return;
      }
    }
    using Tag = vec::ScalableTag<T, 0>;
    for (nint_t row = 0; row < 16; ++row) {
      const auto value = row < active_m
          ? input.load(
                Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
                vec::opt::first(active_n), vec::opt::zero)
          : vec::zeros(Tag{});
      vec::store(Tag{}, buffer + row * 16, value);
    }
    amx_intrinsics::load<Tile>(buffer, 16 * sizeof(T));
  }
}

template <gemm::Atom Atom, int Tile, bool SpatialGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE void load_a_tile(
    const Source& source, nint_t m, nint_t k,
    nint_t logical_m, nint_t logical_k,
    typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (direct_row_major_input_v<Source>) {
    if ((SpatialGuaranteed || m + 16 <= logical_m) &&
        k + KR <= logical_k) {
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      amx_intrinsics::load<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    }
  }
  amx_intrinsics::load<Tile>(
      prepare_a<Atom, SpatialGuaranteed>(
          source, m, k, logical_m, logical_k, buffer),
      64);
}

template <gemm::Atom Atom, int NM, int NN, std::size_t... I>
VECOPS_ALWAYS_INLINE void compute_tiles_impl(std::index_sequence<I...>) {
  constexpr int Outputs = NM * NN;
  (amx_intrinsics::dot<
       typename Atom::TAcc, typename Atom::TA, typename Atom::TB,
       static_cast<int>(I), Outputs + static_cast<int>(I / NN),
       Outputs + NM + static_cast<int>(I % NN)>(), ...);
}

template <gemm::Atom Atom, int NM, int NN>
VECOPS_ALWAYS_INLINE void compute_tiles() {
  compute_tiles_impl<Atom, NM, NN>(std::make_index_sequence<NM * NN>{});
}

template <int Tile, bool FullTile, bool NonEmpty, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    nint_t active_m, nint_t active_n,
    typename COutput::ComputeType* buffer) {
  using T = typename COutput::ComputeType;
  // Do not form a DataAccess base coordinate for an empty resident sub-tile.
  if constexpr (!NonEmpty) {
    if (active_m == 0 || active_n == 0) return;
  }
  if constexpr (direct_row_major_output_v<COutput>) {
    if constexpr (FullTile) {
      const auto strides = output.raw_strides();
      auto* pointer = reinterpret_cast<T*>(output.raw_data()) +
          m * strides[0] + n;
      amx_intrinsics::store<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    } else if (active_m == 16 && active_n == 16) {
      const auto strides = output.raw_strides();
      auto* pointer = reinterpret_cast<T*>(output.raw_data()) +
          m * strides[0] + n;
      amx_intrinsics::store<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    }
  }
  amx_intrinsics::store<Tile>(buffer, 16 * sizeof(T));
  using Tag = vec::ScalableTag<T, 0>;
  for (nint_t row = 0; row < active_m; ++row) {
    const auto value = vec::load(Tag{}, buffer + row * 16);
    output.store(
        Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
        vec::opt::first(active_n));
  }
}

template <gemm::Atom Atom, typename Case,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n,
    void* scratch) {
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  constexpr bool FullM =
      Case::m_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool FullN =
      Case::n_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool GuaranteedM = FullM || Case::exact_blocks;
  constexpr bool GuaranteedN = FullN || Case::exact_blocks;
  auto* bytes = static_cast<std::byte*>(scratch);
  auto* a_buffers = reinterpret_cast<typename Atom::TA*>(bytes);
  auto* b_buffers = reinterpret_cast<typename Atom::TB*>(bytes + NM * 1024);
  auto* c_buffers = reinterpret_cast<typename Atom::TAcc*>(
      bytes + (NM + NN) * 1024);

  [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
    (initialize_c_tile<static_cast<int>(I)>(
         c_input,
         m + static_cast<nint_t>(I / NN) * 16,
         n + static_cast<nint_t>(I % NN) * 16,
         std::clamp(active_m - static_cast<nint_t>(I / NN) * 16,
                    nint_t{0}, nint_t{16}),
         std::clamp(active_n - static_cast<nint_t>(I % NN) * 16,
                    nint_t{0}, nint_t{16}),
         c_buffers + I * 256), ...);
  }(std::make_index_sequence<Outputs>{});

  for (nint_t k = 0; k < logical_k; k += KR) {
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (load_a_tile<Atom, Outputs + static_cast<int>(I), GuaranteedM>(
           a, m + static_cast<nint_t>(I) * 16, k,
           logical_m, logical_k,
           a_buffers + I * 1024 / sizeof(typename Atom::TA)), ...);
    }(std::make_index_sequence<NM>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (amx_intrinsics::load<Outputs + NM + static_cast<int>(I)>(
           prepare_b<Atom, GuaranteedN>(
               b, n + static_cast<nint_t>(I) * 16, k,
               logical_n, logical_k,
               b_buffers + I * 1024 / sizeof(typename Atom::TB)),
           64), ...);
    }(std::make_index_sequence<NN>{});
    compute_tiles<Atom, NM, NN>();
  }

  [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
    (store_c_tile<
         static_cast<int>(I), FullM && FullN,
         (FullM && FullN) || Case::exact_blocks>(
         c_output,
         m + static_cast<nint_t>(I / NN) * 16,
         n + static_cast<nint_t>(I % NN) * 16,
         std::clamp(active_m - static_cast<nint_t>(I / NN) * 16,
                    nint_t{0}, nint_t{16}),
         std::clamp(active_n - static_cast<nint_t>(I % NN) * 16,
                    nint_t{0}, nint_t{16}),
         c_buffers + I * 256), ...);
  }(std::make_index_sequence<Outputs>{});
}

} // namespace vecops::kernel::matmul_details::amx

namespace vecops::kernel::matmul_details {

template <>
struct Backend<matmul_implementation::AMX> {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::x86::Tiles>;
  using Catalog = amx::Catalog;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 8 * 1024 + 63; }

  template <gemm::Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B>
  using EffectivePolicy = std::conditional_t<
      std::same_as<Policy, matmul_policy::Automatic>,
      kernel::loop::tile2d_policy::FourRegions, Policy>;

  template <gemm::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    amx::Configuration configuration;
    scope.with_configuration(configuration, [&](auto&)
        VECOPS_INLINE_LAMBDA_NOEXCEPT {
      matmul_details::run_tiles<Backend, Atom, Policy>(
          m, n, k, a, b, c_input, c_output, scratch);
    });
  }

  template <gemm::Atom, typename A, typename B,
            meta::ValueType K, typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(K, Fn&& fn) {
    std::forward<Fn>(fn).template operator()<std::false_type>();
  }

  template <gemm::Atom Atom, typename Case, typename Plan,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_case(
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void* scratch) {
    amx::microkernel<Atom, Case>(
        a, b, c_input, c_output,
        logical_m, logical_n, logical_k,
        m, n, active_m, active_n, scratch);
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H
