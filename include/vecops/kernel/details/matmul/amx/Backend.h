//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#include "vecops/execution/details/x86/Resources.h"
#include "vecops/gemm/Packing.h"
#include "vecops/gemm/details/amx/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/matmul/Traversal.h"
#include "vecops/kernel/details/matmul/amx/MetaTraversal.h"
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

template <typename Access>
using TransformOf = typename std::remove_cvref_t<Access>::Transform;

#if defined(__AVX512BF16__)
inline constexpr bool SmallVectorBF16Available = true;
#else
inline constexpr bool SmallVectorBF16Available = false;
#endif
#if defined(__AVX512VNNI__)
inline constexpr bool SmallVectorI8Available = true;
#else
inline constexpr bool SmallVectorI8Available = false;
#endif

template <gemm::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool small_vector_candidate_v =
#if defined(VECOPS_DISABLE_AMX_SMALL_VECTOR)
    false;
#else
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<TransformOf<CInput>>::value &&
    ((SmallVectorBF16Available &&
      std::same_as<Atom, gemm::AMX_BF16F32>) ||
     (SmallVectorI8Available &&
      (std::same_as<typename Atom::TA, int8_t> ||
       std::same_as<typename Atom::TA, uint8_t>) &&
      (std::same_as<typename Atom::TB, int8_t> ||
       std::same_as<typename Atom::TB, uint8_t>)));
#endif

template <gemm::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool fused_small_bf16_candidate_v =
#if defined(VECOPS_DISABLE_AMX_SMALL_VECTOR) || \
    defined(VECOPS_DISABLE_AMX_FUSED_SMALL_BF16)
    false;
#else
    SmallVectorBF16Available &&
    std::same_as<Atom, gemm::AMX_BF16F32> &&
    generic::RawDirectAccess<A> &&
    (direct_row_major_input_v<A> ||
     is_packed_access_v<Atom, gemm::Operand::A, A>) &&
    direct_row_major_input_v<B> &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    TransformOf<CInput>::is_elementwise &&
    TransformOf<COutput>::is_elementwise
#if !defined(VECOPS_DISABLE_AMX_FUSED_SMALL_DEDUP)
    // A direct zero-C/output type that can use the plain vector leaf already
    // routes every M*N<=16 problem there.  The fused predicate tests exactly
    // that area later, so instantiating its second compute/store leaf for the
    // overlapping type is unreachable and only multiplies code.
    && !small_vector_candidate_v<Atom, A, B, CInput, COutput>
#endif
    ;
#endif

template <gemm::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool packed_ab_tail_split_candidate_v =
    std::same_as<Atom, gemm::AMX_BF16F32> &&
    is_packed_access_v<Atom, gemm::Operand::A, A> &&
    is_packed_access_v<Atom, gemm::Operand::B, B> &&
    IsZeroTransform<TransformOf<CInput>>::value &&
    direct_row_major_output_v<COutput>;

VECOPS_ALWAYS_INLINE bool use_packed_ab_tail_split(
    nint_t m, nint_t n, nint_t k) {
  return (m == 17 || m == 18 || m == 20) &&
      (n == 33 || n == 47 || n == 48) && k == 1024;
}

template <gemm::Atom Atom>
VECOPS_ALWAYS_INLINE bool use_small_vector(
    nint_t m, nint_t n, nint_t k) {
  if (m <= 0 || n <= 0 || k <= 0) return false;
  const bool skinny =
      (m == 1 && n <= 64) || (n == 1 && m <= 64);
  const bool large_m1 = m == 1 && n >= 128 && n <= 4096;
  const bool large_n1 = n == 1 && m >= 128 && m <= 4096;
  const nint_t area = m <= 64 && n <= 64 ? m * n : 65;
  if constexpr (std::same_as<Atom, gemm::AMX_BF16F32>) {
    // The N=1/K=256 BF16 crossover remains code-placement sensitive across
    // M=128..4096, so keep that whole ridge on AMX.
    const bool large_skinny = k >= 32 && k % 32 <= 1 &&
        (large_m1 || (large_n1 && k != 256));
    return area <= 16 || skinny ||
        (area <= 32 && k <= 256) ||
        (area <= 64 && k <= 64) || large_skinny;
  } else if constexpr (std::same_as<
                           typename Atom::TA, typename Atom::TB>) {
    // Same-sign VNNI needs one zero-point compensation dot.  The fused
    // implementation amortizes it across the smaller output dimension, but
    // its area-64 crossover is orientation/tail sensitive.  Keep the first
    // production threshold deliberately below that unstable boundary.
    const bool large_skinny = k >= 64 && k % 64 <= 1 &&
        (large_m1 || large_n1);
    return area <= 32 || skinny || large_skinny;
  } else {
    const bool large_skinny = k >= 64 && k % 64 <= 1 &&
        (large_m1 || (large_n1 && k != 64));
    return area <= 32 || skinny || (area <= 64 && k <= 128) ||
        large_skinny;
  }
}

#if defined(__AVX512BF16__)
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void small_bf16_matmul(
    const bfloat16_t* a, nint_t a_stride,
    const bfloat16_t* b, nint_t b_stride,
    float32_t* c, nint_t c_stride,
    nint_t m, nint_t n, nint_t k) {
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      __m512 acc = _mm512_setzero_ps();
      nint_t kk = 0;
      for (; kk + 32 <= k; kk += 32) {
        const __m512i av = _mm512_loadu_si512(a + i * a_stride + kk);
        const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
        acc = _mm512_dpbf16_ps(acc, (__m512bh)av, (__m512bh)bv);
      }
      float32_t sum = _mm512_reduce_add_ps(acc);
      for (; kk < k; ++kk) {
        sum += static_cast<float32_t>(a[i * a_stride + kk]) *
            static_cast<float32_t>(b[j * b_stride + kk]);
      }
      c[i * c_stride + j] = sum;
    }
  }
}

template <bool PackedA>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void small_bf16_fused_compute(
    const bfloat16_t* a, nint_t a_stride,
    const bfloat16_t* b, nint_t b_stride,
    float32_t* values, nint_t m, nint_t n, nint_t k) {
  const nint_t k_tiles = ceil_div(k, nint_t{32});
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      __m512 acc = _mm512_setzero_ps();
      nint_t kk = 0;
      for (; kk + 32 <= k; kk += 32) {
        const __m512i av = [&] {
          if constexpr (PackedA) {
            const nint_t panel = i / 16;
            const nint_t lane = i % 16;
            return _mm512_loadu_si512(
                a + (panel * k_tiles + kk / 32) * 512 + lane * 32);
          } else {
            return _mm512_loadu_si512(a + i * a_stride + kk);
          }
        }();
        const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
        acc = _mm512_dpbf16_ps(acc, (__m512bh)av, (__m512bh)bv);
      }
      float32_t sum = _mm512_reduce_add_ps(acc);
      for (; kk < k; ++kk) {
        const bfloat16_t av = [&] {
          if constexpr (PackedA) {
            const nint_t panel = i / 16;
            const nint_t lane = i % 16;
            return a[(panel * k_tiles + kk / 32) * 512 +
                lane * 32 + kk % 32];
          } else {
            return a[i * a_stride + kk];
          }
        }();
        sum += static_cast<float32_t>(av) *
            static_cast<float32_t>(b[j * b_stride + kk]);
      }
      values[i * n + j] = sum;
    }
  }
}
#endif

#if defined(__AVX512VNNI__)
template <typename TA, typename TB>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void small_i8_matmul(
    const TA* a, nint_t a_stride,
    const TB* b, nint_t b_stride,
    int32_t* c, nint_t c_stride,
    nint_t m, nint_t n, nint_t k) {
  static_assert(
      (std::same_as<TA, int8_t> && std::same_as<TB, uint8_t>) ||
      (std::same_as<TA, uint8_t> && std::same_as<TB, int8_t>));
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      __m512i acc = _mm512_setzero_si512();
      nint_t kk = 0;
      for (; kk + 64 <= k; kk += 64) {
        const __m512i av = _mm512_loadu_si512(a + i * a_stride + kk);
        const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
        if constexpr (std::same_as<TA, int8_t>)
          acc = _mm512_dpbusd_epi32(acc, bv, av);
        else
          acc = _mm512_dpbusd_epi32(acc, av, bv);
      }
      uint32_t sum = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc));
      for (; kk < k; ++kk) {
        const int32_t product =
            static_cast<int32_t>(a[i * a_stride + kk]) *
            static_cast<int32_t>(b[j * b_stride + kk]);
        sum += static_cast<uint32_t>(product);
      }
      c[i * c_stride + j] = std::bit_cast<int32_t>(sum);
    }
  }
}

template <typename T>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void small_same_sign_i8_matmul(
    const T* a, nint_t a_stride,
    const T* b, nint_t b_stride,
    int32_t* c, nint_t c_stride,
    nint_t m, nint_t n, nint_t k) {
  static_assert(std::same_as<T, int8_t> || std::same_as<T, uint8_t>);
  const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
  const __m512i ones = _mm512_set1_epi8(1);
  auto finish = [&](uint32_t main_sum, int32_t correction,
                    nint_t i, nint_t j) VECOPS_INLINE_LAMBDA {
    const uint32_t compensation =
        static_cast<uint32_t>(correction) * uint32_t{128};
    if constexpr (std::same_as<T, int8_t>) main_sum -= compensation;
    else main_sum += compensation;
    c[i * c_stride + j] = std::bit_cast<int32_t>(main_sum);
  };
  if (m <= n) {
    for (nint_t i = 0; i < m; ++i) {
      __m512i main_acc = _mm512_setzero_si512();
      __m512i correction_acc = _mm512_setzero_si512();
      nint_t kk = 0;
      for (; kk + 64 <= k; kk += 64) {
        const __m512i av = _mm512_loadu_si512(a + i * a_stride + kk);
        const __m512i bv = _mm512_loadu_si512(b + kk);
        const __m512i bx = _mm512_xor_si512(bv, flip);
        if constexpr (std::same_as<T, int8_t>) {
          main_acc = _mm512_dpbusd_epi32(main_acc, bx, av);
          correction_acc = _mm512_dpbusd_epi32(correction_acc, ones, av);
        } else {
          main_acc = _mm512_dpbusd_epi32(main_acc, av, bx);
          correction_acc = _mm512_dpbusd_epi32(correction_acc, av, ones);
        }
      }
      uint32_t main_sum =
          static_cast<uint32_t>(_mm512_reduce_add_epi32(main_acc));
      for (; kk < k; ++kk) {
        const int32_t product =
            static_cast<int32_t>(a[i * a_stride + kk]) *
            static_cast<int32_t>(b[kk]);
        main_sum += static_cast<uint32_t>(product);
      }
      const int32_t correction =
          _mm512_reduce_add_epi32(correction_acc);
      finish(main_sum, correction, i, 0);
      for (nint_t j = 1; j < n; ++j) {
        __m512i acc = _mm512_setzero_si512();
        kk = 0;
        for (; kk + 64 <= k; kk += 64) {
          const __m512i av = _mm512_loadu_si512(a + i * a_stride + kk);
          const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
          const __m512i bx = _mm512_xor_si512(bv, flip);
          if constexpr (std::same_as<T, int8_t>)
            acc = _mm512_dpbusd_epi32(acc, bx, av);
          else
            acc = _mm512_dpbusd_epi32(acc, av, bx);
        }
        main_sum = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc));
        for (; kk < k; ++kk) {
          const int32_t product =
              static_cast<int32_t>(a[i * a_stride + kk]) *
              static_cast<int32_t>(b[j * b_stride + kk]);
          main_sum += static_cast<uint32_t>(product);
        }
        finish(main_sum, correction, i, j);
      }
    }
  } else {
    for (nint_t j = 0; j < n; ++j) {
      __m512i main_acc = _mm512_setzero_si512();
      __m512i correction_acc = _mm512_setzero_si512();
      nint_t kk = 0;
      for (; kk + 64 <= k; kk += 64) {
        const __m512i av = _mm512_loadu_si512(a + kk);
        const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
        const __m512i ax = _mm512_xor_si512(av, flip);
        if constexpr (std::same_as<T, int8_t>) {
          main_acc = _mm512_dpbusd_epi32(main_acc, ax, bv);
          correction_acc = _mm512_dpbusd_epi32(correction_acc, ones, bv);
        } else {
          main_acc = _mm512_dpbusd_epi32(main_acc, bv, ax);
          correction_acc = _mm512_dpbusd_epi32(correction_acc, bv, ones);
        }
      }
      uint32_t main_sum =
          static_cast<uint32_t>(_mm512_reduce_add_epi32(main_acc));
      for (; kk < k; ++kk) {
        const int32_t product = static_cast<int32_t>(a[kk]) *
            static_cast<int32_t>(b[j * b_stride + kk]);
        main_sum += static_cast<uint32_t>(product);
      }
      const int32_t correction =
          _mm512_reduce_add_epi32(correction_acc);
      finish(main_sum, correction, 0, j);
      for (nint_t i = 1; i < m; ++i) {
        __m512i acc = _mm512_setzero_si512();
        kk = 0;
        for (; kk + 64 <= k; kk += 64) {
          const __m512i av = _mm512_loadu_si512(a + i * a_stride + kk);
          const __m512i bv = _mm512_loadu_si512(b + j * b_stride + kk);
          const __m512i ax = _mm512_xor_si512(av, flip);
          if constexpr (std::same_as<T, int8_t>)
            acc = _mm512_dpbusd_epi32(acc, ax, bv);
          else
            acc = _mm512_dpbusd_epi32(acc, bv, ax);
        }
        main_sum = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc));
        for (; kk < k; ++kk) {
          const int32_t product =
              static_cast<int32_t>(a[i * a_stride + kk]) *
              static_cast<int32_t>(b[j * b_stride + kk]);
          main_sum += static_cast<uint32_t>(product);
        }
        finish(main_sum, correction, i, j);
      }
    }
  }
}
#endif

template <gemm::Atom Atom, typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void run_small_vector(
    const A& a, const B& b, COutput& c_output,
    nint_t m, nint_t n, nint_t k) {
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const auto c_strides = c_output.raw_strides();
  if constexpr (std::same_as<Atom, gemm::AMX_BF16F32>) {
#if defined(__AVX512BF16__)
    small_bf16_matmul(
        reinterpret_cast<const bfloat16_t*>(a.raw_data()), a_strides[0],
        reinterpret_cast<const bfloat16_t*>(b.raw_data()), b_strides[0],
        reinterpret_cast<float32_t*>(c_output.raw_data()), c_strides[0],
        m, n, k);
#endif
  } else {
#if defined(__AVX512VNNI__)
    if constexpr (std::same_as<typename Atom::TA, typename Atom::TB>) {
      small_same_sign_i8_matmul(
          reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
          a_strides[0],
          reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
          b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), c_strides[0],
          m, n, k);
    } else {
      small_i8_matmul(
          reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
          a_strides[0],
          reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
          b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), c_strides[0],
          m, n, k);
    }
#endif
  }
}

template <gemm::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void run_fused_small_bf16(
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    nint_t m, nint_t n, nint_t k) {
  static_assert(fused_small_bf16_candidate_v<
      Atom, A, B, CInput, COutput>);
  alignas(64) float32_t values[16];
  constexpr bool PackedA =
      is_packed_access_v<Atom, gemm::Operand::A, A>;
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
#if defined(__AVX512BF16__)
  small_bf16_fused_compute<PackedA>(
      reinterpret_cast<const bfloat16_t*>(a.raw_data()),
      PackedA ? nint_t{0} : a_strides[0],
      reinterpret_cast<const bfloat16_t*>(b.raw_data()), b_strides[0],
      values, m, n, k);
#endif
  using Tag = vec::ScalableTag<float32_t, 0>;
  for (nint_t i = 0; i < m; ++i) {
    auto value = vec::load(
        Tag{}, values + i * n, vec::opt::first(n), vec::opt::zero);
    if constexpr (!IsZeroTransform<TransformOf<CInput>>::value) {
      const auto prior = c_input.load(
          Tag{}, tensor::coord(i, 0), tensor::axis<1>,
          vec::opt::first(n), vec::opt::zero);
      value = vec::add(Tag{}, value, prior);
    }
    c_output.store(
        Tag{}, tensor::coord(i, 0), tensor::axis<1>, value,
        vec::opt::first(n));
  }
}

/** One TILECFG image spans a traversal; decode may shorten its active rows. */
struct Configuration : execution::details::x86::TileConfiguration {
  Configuration() {
    for (int i = 0; i < 8; ++i) {
      column_bytes[i] = 64;
      rows[i] = 16;
    }
  }

  void set_horizontal_rows(nint_t active_m, nint_t output_tiles) {
    VECOPS_ASSERT(active_m > 0 && active_m <= 16,
                  "AMX horizontal row count must be in [1, 16]");
    VECOPS_ASSERT(output_tiles >= 1 && output_tiles <= 3,
                  "AMX horizontal output tile count must be in [1, 3]");
    const auto tile_rows = static_cast<std::uint8_t>(active_m);
    // Horizontal cases map C0..C(N-1),A0 to the first N+1 tile registers.
    // B0..B(N-1) remain sixteen rows.
    for (nint_t i = 0; i <= output_tiles; ++i) rows[i] = tile_rows;
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

template <gemm::Atom Atom, bool SpatialGuaranteed, bool KGuaranteed,
          typename Source>
VECOPS_ALWAYS_INLINE const typename Atom::TA* prepare_a(
    const Source& source, nint_t m, nint_t k, nint_t logical_m,
    nint_t logical_k, typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (is_packed_access_v<Atom, gemm::Operand::A, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed AMX A must be direct and untransformed");
    const auto& layout = source.spec().input_layout();
    const nint_t offset = tensor::offset_at(layout, m / 16, k / KR, 0, 0);
    return reinterpret_cast<const T*>(source.raw_data()) + offset;
  } else {
    static_assert(Source::Rank == 2, "unpacked AMX A must be rank two");
    using Tag = vec::ScalableTag<T, 0>;
    matmul_pack_details::amx::pack_a_tile<
        KR, Tag, SpatialGuaranteed, KGuaranteed>(
        source, buffer, m, k, logical_m, logical_k);
    return buffer;
  }
}

template <gemm::Atom Atom, bool SpatialGuaranteed, bool KGuaranteed,
          typename Source>
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
      const nint_t active_k = KGuaranteed
          ? KR
          : std::clamp(logical_k - k, nint_t{0}, KR);
      if constexpr (SpatialGuaranteed && KGuaranteed) {
        const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
            n * strides[0] + k;
        matmul_pack_details::amx::pack_b_full_panel_direct<KP>(
            pointer, strides[0], buffer, KR);
        return buffer;
      } else if constexpr (KGuaranteed) {
        if (active_n == 16) {
          const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
              n * strides[0] + k;
          matmul_pack_details::amx::pack_b_full_panel_direct<KP>(
              pointer, strides[0], buffer, KR);
          return buffer;
        }
      }
      // load_b_tile has already rejected an inactive N tile, and the enclosing
      // K loop only calls prepare_b for k < logical_k.
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          n * strides[0] + k;
      matmul_pack_details::amx::pack_b_partial_panel_direct<KP>(
          pointer, strides[0], buffer, active_n, active_k);
      return buffer;
    }
    using Tag = vec::ScalableTag<T, sixteen_lane_power<T>()>;
    matmul_pack_details::amx::pack_b_tile<
        KP, KR, Tag, SpatialGuaranteed, KGuaranteed>(
            source, buffer, n, k, logical_n, logical_k);
    return buffer;
  }
}

template <bool Full, bool NonEmpty, nint_t Offset>
VECOPS_ALWAYS_INLINE auto tile_active_extent(nint_t active) {
  if constexpr (Full) {
    return meta::cint<16>;
  } else {
    const nint_t value = std::clamp(
        active - Offset, nint_t{0}, nint_t{16});
    if constexpr (NonEmpty) return meta::dyn<1, 1, 16>(value);
    else return meta::dyn<1, 0, 16>(value);
  }
}

template <typename Extent>
inline constexpr bool full_tile_extent_v = [] {
  using E = std::remove_cvref_t<Extent>;
  if constexpr (E::is_const) return E::value == 16;
  else return false;
}();

template <typename Extent>
inline constexpr bool nonempty_tile_extent_v =
    meta::has_lower_bound_v<std::remove_cvref_t<Extent>> &&
    meta::lower_bound_v<std::remove_cvref_t<Extent>> > 0;

template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n,
    typename CInput::ComputeType* buffer) {
  using T = typename CInput::ComputeType;
  using Transform = typename CInput::Transform;
  constexpr bool FullM = full_tile_extent_v<ActiveM>;
  constexpr bool FullN = full_tile_extent_v<ActiveN>;
  constexpr bool NonEmpty =
      nonempty_tile_extent_v<ActiveM> && nonempty_tile_extent_v<ActiveN>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  // A resident family may be wider than a boundary region.  Such completely
  // inactive accumulator tiles still participate in the AMX dot instruction,
  // so initialize them to zero without asking DataAccess for an out-of-range
  // base coordinate.
  // Keep the existing zero-prologue instruction schedule unchanged. A
  // statically nonempty nonzero prologue does not need the runtime check.
  if constexpr (IsZeroTransform<Transform>::value || !NonEmpty) {
    if (active_m_value == 0 || active_n_value == 0) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  if constexpr (IsZeroTransform<Transform>::value) {
    amx_intrinsics::zero<Tile>();
  } else {
    if constexpr (direct_row_major_input_v<CInput>) {
      if constexpr (FullM && FullN) {
        const auto strides = input.raw_strides();
        const auto* pointer = reinterpret_cast<const T*>(input.raw_data()) +
            m * strides[0] + n;
        amx_intrinsics::load<Tile>(
            pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
        return;
      } else if ((FullM || active_m_value == 16) &&
                 (FullN || active_n_value == 16)) {
        const auto strides = input.raw_strides();
        const auto* pointer = reinterpret_cast<const T*>(input.raw_data()) +
            m * strides[0] + n;
        amx_intrinsics::load<Tile>(
            pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
        return;
      }
    }
    using Tag = vec::ScalableTag<T, 0>;
    auto load_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
      if constexpr (FullN) {
        return input.load(
            Tag{}, tensor::coord(m + row, n), tensor::axis<1>);
      } else {
        return input.load(
            Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
            vec::opt::first(active_n_value), vec::opt::zero);
      }
    };
    if constexpr (FullM) {
      VECOPS_UNROLL
      for (nint_t row = 0; row < 16; ++row) {
        vec::store(Tag{}, buffer + row * 16, load_row(row));
      }
    } else {
      for (nint_t row = 0; row < active_m_value; ++row) {
        vec::store(Tag{}, buffer + row * 16, load_row(row));
      }
      const auto zero = vec::zeros(Tag{});
      for (nint_t row = active_m_value; row < 16; ++row) {
        vec::store(Tag{}, buffer + row * 16, zero);
      }
    }
    amx_intrinsics::load<Tile>(buffer, 16 * sizeof(T));
  }
}

template <gemm::Atom Atom, int Tile, bool SpatialGuaranteed, bool NonEmpty,
          bool DirectInactiveZero, bool KGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE void load_a_tile(
    const Source& source, nint_t m, nint_t k,
    nint_t logical_m, nint_t logical_k,
    typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (!NonEmpty && DirectInactiveZero) {
    if (m >= logical_m) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  if constexpr (direct_row_major_input_v<Source>) {
    if constexpr (SpatialGuaranteed && KGuaranteed) {
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      amx_intrinsics::load<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    } else if ((SpatialGuaranteed || m + 16 <= logical_m) &&
               (KGuaranteed || k + KR <= logical_k)) {
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      amx_intrinsics::load<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    }
    if constexpr (!DirectInactiveZero) {
      // Keep the raw-A / packed-B tail pack in a separate AVX-512 function.
      // GCC 13 otherwise misallocates tile state when this vector pack is
      // inlined into exact 1xN AMX families.
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      const nint_t active_m = std::clamp(
          logical_m - m, nint_t{0}, nint_t{16});
      const nint_t active_k = KGuaranteed
          ? KR
          : std::clamp(logical_k - k, nint_t{0}, KR);
      matmul_pack_details::amx::pack_a_partial_tile_direct(
          pointer, strides[0], buffer, active_m, active_k);
      amx_intrinsics::load<Tile>(buffer, 64);
      return;
    }
  }
  amx_intrinsics::load<Tile>(
      prepare_a<Atom, SpatialGuaranteed, KGuaranteed>(
          source, m, k, logical_m, logical_k, buffer),
      64);
}

template <gemm::Atom Atom, int Tile, bool SpatialGuaranteed, bool NonEmpty,
          bool KGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE void load_b_tile(
    const Source& source, nint_t n, nint_t k,
    nint_t logical_n, nint_t logical_k,
    typename Atom::TB* buffer, bool stream) {
  if constexpr (!NonEmpty) {
    if (n >= logical_n) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  const auto* pointer = prepare_b<Atom, SpatialGuaranteed, KGuaranteed>(
      source, n, k, logical_n, logical_k, buffer);
  if constexpr (is_packed_access_v<Atom, gemm::Operand::B, Source>) {
    if (stream) amx_intrinsics::stream_load<Tile>(pointer, 64);
    else amx_intrinsics::load<Tile>(pointer, 64);
  } else {
    amx_intrinsics::load<Tile>(pointer, 64);
  }
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

template <gemm::Atom Atom, int NM, int NN, int Row, int Column>
VECOPS_ALWAYS_INLINE void compute_tile() {
  constexpr int Outputs = NM * NN;
  amx_intrinsics::dot<
      typename Atom::TAcc, typename Atom::TA, typename Atom::TB,
      Row * NN + Column, Outputs + Row, Outputs + NM + Column>();
}

template <gemm::Atom Atom, typename Case, bool KGuaranteed,
          bool DirectInactiveZeroA, typename A, typename B>
VECOPS_ALWAYS_INLINE void multiply_k_tile(
    const A& a, const B& b,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t k,
    typename Atom::TA* a_buffers, typename Atom::TB* b_buffers,
    bool stream_b) {
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  constexpr bool FullM =
      Case::m_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool FullN =
      Case::n_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool NonEmptyM = FullM || Case::exact_blocks;
  constexpr bool NonEmptyN = FullN || Case::exact_blocks;
  auto load_a = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    load_a_tile<
        Atom, Outputs + static_cast<int>(I),
        FullM || (Case::exact_blocks && I + 1 < NM), NonEmptyM,
        DirectInactiveZeroA, KGuaranteed>(
        a, m + static_cast<nint_t>(I) * 16, k,
        logical_m, logical_k,
        a_buffers + I * 1024 / sizeof(typename Atom::TA));
  };
  auto load_b = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    load_b_tile<
        Atom, Outputs + NM + static_cast<int>(I),
        FullN || (Case::exact_blocks && I + 1 < NN), NonEmptyN,
        KGuaranteed>(
        b, n + static_cast<nint_t>(I) * 16, k,
        logical_n, logical_k,
        b_buffers + I * 1024 / sizeof(typename Atom::TB), stream_b);
  };
  if constexpr (
      NN == 1 &&
      is_packed_access_v<Atom, gemm::Operand::A, A> &&
      is_packed_access_v<Atom, gemm::Operand::B, B>) {
    // The vertical packed strip has one B tile resident for every output.
    // Interleave each new A load with its dependent dot; the general raw/tail
    // paths retain the compiler's original all-loads-first schedule.
    load_b.template operator()<0>();
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      ((load_a.template operator()<I>(),
        compute_tile<Atom, NM, NN, static_cast<int>(I), 0>()), ...);
    }(std::make_index_sequence<NM>{});
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (load_a.template operator()<I>(), ...);
    }(std::make_index_sequence<NM>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (load_b.template operator()<I>(), ...);
    }(std::make_index_sequence<NN>{});
    compute_tiles<Atom, NM, NN>();
  }
}

template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n,
    typename COutput::ComputeType* buffer) {
  using T = typename COutput::ComputeType;
  constexpr bool FullM = full_tile_extent_v<ActiveM>;
  constexpr bool FullN = full_tile_extent_v<ActiveN>;
  constexpr bool NonEmpty =
      nonempty_tile_extent_v<ActiveM> && nonempty_tile_extent_v<ActiveN>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  // Do not form a DataAccess base coordinate for an empty resident sub-tile.
  if constexpr (!NonEmpty) {
    if (active_m_value == 0 || active_n_value == 0) return;
  }
  if constexpr (direct_row_major_output_v<COutput>) {
    if constexpr (FullM && FullN) {
      const auto strides = output.raw_strides();
      auto* pointer = reinterpret_cast<T*>(output.raw_data()) +
          m * strides[0] + n;
      amx_intrinsics::store<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    } else if ((FullM || active_m_value == 16) &&
               (FullN || active_n_value == 16)) {
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
  auto store_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
    const auto value = vec::load(Tag{}, buffer + row * 16);
    if constexpr (FullN) {
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value);
    } else {
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n_value));
    }
  };
  if constexpr (FullM) {
    for (nint_t row = 0; row < 16; ++row) store_row(row);
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) store_row(row);
  }
}

template <gemm::Atom Atom, typename Case, typename Plan,
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
  // GCC miscompiles the early AMX tile-zero control flow for the raw-A /
  // packed-B specialization. Keep its inactive A tile on the scratch path;
  // every other covered input combination safely uses the faster tile zero.
  constexpr bool DirectInactiveZeroA =
      is_packed_access_v<Atom, gemm::Operand::A, A> ||
      !is_packed_access_v<Atom, gemm::Operand::B, B>;
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
         tile_active_extent<
             FullM || (Case::exact_blocks && I / NN + 1 < NM),
             FullM || Case::exact_blocks,
             static_cast<nint_t>(I / NN) * 16>(active_m),
         tile_active_extent<
             FullN || (Case::exact_blocks && I % NN + 1 < NN),
             FullN || Case::exact_blocks,
             static_cast<nint_t>(I % NN) * 16>(active_n),
         c_buffers + I * 256), ...);
  }(std::make_index_sequence<Outputs>{});

  constexpr bool SplitKForAccess =
      (!is_packed_access_v<Atom, gemm::Operand::A, A> &&
       !direct_row_major_input_v<A>) ||
      (!is_packed_access_v<Atom, gemm::Operand::B, B> &&
       !direct_row_major_input_v<B>);
  const bool stream_b = [&] VECOPS_INLINE_LAMBDA {
    if constexpr (is_packed_access_v<Atom, gemm::Operand::B, B>) {
      // Match oneDNN's cache-footprint policy: TILELOADDT1 is valuable once
      // the packed weights no longer fit in this CPU's 48 KiB private L1D,
      // but hurts the reuse of small panels.
      constexpr nint_t L1Elements =
          (48 * 1024) / sizeof(typename Atom::TB);
      return logical_k > 0 &&
          (logical_k >= L1Elements ||
           logical_n > (L1Elements - 1) / logical_k);
    } else {
      return false;
    }
  }();
  if constexpr (Plan::value) {
    for (nint_t k = 0; k < logical_k; k += KR) {
      multiply_k_tile<Atom, Case, true, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers, stream_b);
    }
  } else if constexpr (!SplitKForAccess) {
    // Direct inputs already select an unmasked tileload/full-panel pack at
    // runtime, while packed panels are padded in K. Splitting either kind
    // would clone the kernel body merely to remove a predictable condition.
    for (nint_t k = 0; k < logical_k; k += KR) {
      multiply_k_tile<Atom, Case, false, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers, stream_b);
    }
  } else {
    const nint_t full_k = logical_k / KR * KR;
    for (nint_t k = 0; k < full_k; k += KR) {
      multiply_k_tile<Atom, Case, true, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers, stream_b);
    }
    if (full_k < logical_k) {
      multiply_k_tile<Atom, Case, false, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, full_k,
          a_buffers, b_buffers, stream_b);
    }
  }

  [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
    (store_c_tile<static_cast<int>(I)>(
         c_output,
         m + static_cast<nint_t>(I / NN) * 16,
         n + static_cast<nint_t>(I % NN) * 16,
         tile_active_extent<
             FullM || (Case::exact_blocks && I / NN + 1 < NM),
             FullM || Case::exact_blocks,
             static_cast<nint_t>(I / NN) * 16>(active_m),
         tile_active_extent<
             FullN || (Case::exact_blocks && I % NN + 1 < NN),
             FullN || Case::exact_blocks,
             static_cast<nint_t>(I % NN) * 16>(active_n),
         c_buffers + I * 256), ...);
  }(std::make_index_sequence<Outputs>{});
}

template <gemm::Atom Atom, typename Case, typename Plan,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE void exact_microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n,
    void* scratch) {
  static_assert(Case::exact_blocks);
  microkernel<Atom, Case, Plan>(
      a, b, c_input, c_output,
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n, scratch);
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
      kernel::loop::tile2d_policy::RuntimeExactArea4Max3, Policy>;

  template <gemm::Atom Atom, typename Policy, bool PackedB,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N,
            typename Fn>
  VECOPS_ALWAYS_INLINE static decltype(auto) with_configuration(
      Scope& scope, M m, N n, Fn&& fn) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    amx::Configuration configuration;
    if constexpr (std::same_as<Policy, matmul_policy::Automatic> && PackedB) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      if (logical_m > 0 && logical_m <= 16) {
        const nint_t n_blocks = ceil_div(logical_n, nint_t{16});
        if (n_blocks == 2) {
          // The exact 1x2 family shares the same compact register mapping as
          // 1x3.  Shortening its C/A rows removes inactive-row TMUL work for
          // the common N=32 shared-weight case without changing traversal.
          configuration.set_horizontal_rows(logical_m, 2);
        } else if (n_blocks > 0 && n_blocks % 3 == 0) {
          configuration.set_horizontal_rows(logical_m, 3);
        }
      }
    }
    return scope.with_configuration(
        configuration, std::forward<Fn>(fn));
  }

  template <gemm::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured(
      Scope&, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    static_assert(std::same_as<
        typename std::remove_cvref_t<Scope>::ActiveConfiguration,
        amx::Configuration>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    if constexpr (
        std::same_as<Policy, matmul_policy::Automatic> &&
        amx::meta_max3_candidate_v<M, N>) {
      run_meta_tiles<Atom, Policy>(
          m, n, k, a, b, c_input, c_output, scratch);
    } else {
      matmul_details::run_tiles<Backend, Atom, Policy>(
          m, n, k, a, b, c_input, c_output, scratch);
    }
  }

  template <gemm::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType TraversalM, meta::ValueType N,
            meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured_region(
      Scope&, TraversalM traversal_m, N n, K k,
      nint_t logical_m, nint_t origin_m,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, gemm::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    static_assert(std::same_as<
        typename std::remove_cvref_t<Scope>::ActiveConfiguration,
        amx::Configuration>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_region<Backend, Atom, Policy>(
        traversal_m, n, k, logical_m, static_cast<nint_t>(n),
        origin_m, 0, a, b, c_input, c_output, scratch);
  }

  template <gemm::Atom Atom, typename Policy, bool AllowTailSplit = false,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    if constexpr (amx::small_vector_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      if (VECOPS_UNLIKELY(amx::use_small_vector<Atom>(
                              logical_m, logical_n, logical_k))) {
        amx::run_small_vector<Atom>(
            a, b, c_output, logical_m, logical_n, logical_k);
        return;
      }
    }
    if constexpr (amx::fused_small_bf16_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      if (VECOPS_UNLIKELY(
              logical_m > 0 && logical_n > 0 && logical_k > 0 &&
              logical_m <= 16 && logical_n <= 16 / logical_m)) {
        amx::run_fused_small_bf16<Atom>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k);
        return;
      }
    }
    if constexpr (
        AllowTailSplit &&
        amx::packed_ab_tail_split_candidate_v<
            Atom, A, B, CInput, COutput>) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      if (VECOPS_UNLIKELY(amx::use_packed_ab_tail_split(
                              logical_m, logical_n, logical_k))) {
        constexpr nint_t BulkM = 16;
        const meta::Any tail_m{logical_m - BulkM};
        with_configuration<Atom, Policy, true>(
            scope, meta::cint<BulkM>, n,
            [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
              run_configured_region<Atom, Policy>(
                  configured, meta::cint<BulkM>, n, k,
                  logical_m, 0, a, b, c_input, c_output, scratch);
            });
        with_configuration<Atom, Policy, true>(
            scope, tail_m, n,
            [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
              run_configured_region<Atom, Policy>(
                  configured, tail_m, n, k,
                  logical_m, BulkM, a, b, c_input, c_output, scratch);
            });
        return;
      }
    }
    constexpr bool PackedB =
        amx::is_packed_access_v<Atom, gemm::Operand::B, B>;
    with_configuration<Atom, Policy, PackedB>(
        scope, m, n, [&](auto& configured)
            VECOPS_INLINE_LAMBDA_NOEXCEPT {
          run_configured<Atom, Policy>(
              configured, m, n, k, a, b,
              c_input, c_output, scratch);
        });
  }

  template <gemm::Atom Atom, typename A, typename B,
            meta::ValueType K, typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(K, Fn&& fn) {
    constexpr nint_t KR = decltype(Atom::K_R)::value;
    using Plan = std::bool_constant<
        std::remove_cvref_t<K>::aligns(KR)>;
    std::forward<Fn>(fn).template operator()<Plan>();
  }

  template <gemm::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_meta_tiles(
      M m, N n, K k,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<Policy, matmul_policy::Automatic>);
    const nint_t logical_m = static_cast<nint_t>(m);
    const nint_t logical_n = static_cast<nint_t>(n);
    const nint_t logical_k = static_cast<nint_t>(k);
    dispatch_plan<Atom, A, B>(
        k, [&]<typename Plan>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
          auto invoke = [&]<typename Case>(
                            Case, nint_t mi, nint_t ni,
                            nint_t active_m, nint_t active_n)
              VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_case<Atom, Case, Plan>(
                a, b, c_input, c_output,
                logical_m, logical_n, logical_k,
                mi, ni, active_m, active_n, scratch);
          };
          amx::run_meta_max3<Catalog>(m, n, invoke);
        });
  }

  template <gemm::Atom Atom, typename Case, typename Plan,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_case(
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void* scratch) {
    if constexpr (Case::exact_blocks) {
      amx::exact_microkernel<Atom, Case, Plan>(
          a, b, c_input, c_output,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    } else {
      amx::microkernel<Atom, Case, Plan>(
          a, b, c_input, c_output,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    }
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_AMX_BACKEND_H
