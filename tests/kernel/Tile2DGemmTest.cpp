#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/kernel/Tile2D.h"
#include "vecops/vec/Vec.h"

namespace {

using vecops::nint_t;
using namespace vecops::meta;
namespace hop = vecops::kernel::loop;
namespace vec = vecops::vec;

using F11 = hop::Tile2DKernelFamily<1, 1, 1, 1, 1, 1>;
using F12 = hop::Tile2DKernelFamily<1, 2, 2, 2, 2, 2>;
using F21 = hop::Tile2DKernelFamily<2, 1, 2, 2, 2, 2>;
using F22 = hop::Tile2DKernelFamily<2, 2, 5, 4, 4, 3>;
using GemmCatalog = hop::Tile2DKernelCatalog<F11, F12, F21, F22>;

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
using GemmBaseTag = vec::ScalableTag<float>;
#else
// 16 float lanes are exactly 64 bytes on x86 and fixed-width SVE builds.
using GemmBaseTag = vec::FixedTag<float, 16>;
#endif

template <vec::VectorTag Tag, int Words>
struct WordsTag;

template <vec::VectorTag Tag>
struct WordsTag<Tag, 1> { using type = Tag; };

template <vec::VectorTag Tag>
struct WordsTag<Tag, 2> { using type = vec::Twice<Tag>; };

template <vec::VectorTag BaseTag, typename Case>
using CaseTag = typename WordsTag<BaseTag, Case::b>::type;

template <vec::VectorTag Tag, typename Case>
VECOPS_ALWAYS_INLINE auto load_b(
    Tag tag, const float* pointer, nint_t active_n) {
  if constexpr (Case::n_mask == hop::Tile2DMaskMode::unmasked) {
    return vec::load(tag, pointer);
  } else {
    return vec::load(
        tag, pointer, vec::opt::first(active_n), vec::opt::zero);
  }
}

template <vec::VectorTag Tag, typename Case>
VECOPS_ALWAYS_INLINE void store_c(
    Tag tag, float* pointer, vec::Vec<Tag> value, nint_t active_n) {
  if constexpr (Case::n_mask == hop::Tile2DMaskMode::unmasked) {
    vec::store(tag, pointer, value);
  } else {
    vec::store(tag, pointer, value, vec::opt::first(active_n));
  }
}

template <vec::VectorTag BaseTag, typename Case>
VECOPS_ALWAYS_INLINE void gemm_tile(
    BaseTag,
    const float* A, const float* B, float* C,
    nint_t M, nint_t N, nint_t K,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  using Tag = CaseTag<BaseTag, Case>;
  Tag tag{};

  auto one_row = [&](nint_t row) {
    auto acc = vec::zeros(tag);
    for (nint_t k = 0; k < K; ++k) {
      const auto bv = load_b<Tag, Case>(tag, B + k * N + n, active_n);
      acc = vec::fmadd(vec::fill(tag, A[row * K + k]), bv, acc);
    }
    store_c<Tag, Case>(tag, C + row * N + n, acc, active_n);
  };

  if constexpr (Case::a == 1) {
    one_row(m);
  } else {
    static_assert(Case::a == 2, "test GEMM implements one or two rows");
    if constexpr (Case::m_mask == hop::Tile2DMaskMode::unmasked) {
      auto acc0 = vec::zeros(tag);
      auto acc1 = vec::zeros(tag);
      for (nint_t k = 0; k < K; ++k) {
        const auto bv = load_b<Tag, Case>(tag, B + k * N + n, active_n);
        acc0 = vec::fmadd(vec::fill(tag, A[m * K + k]), bv, acc0);
        acc1 = vec::fmadd(vec::fill(tag, A[(m + 1) * K + k]), bv, acc1);
      }
      store_c<Tag, Case>(tag, C + m * N + n, acc0, active_n);
      store_c<Tag, Case>(tag, C + (m + 1) * N + n, acc1, active_n);
    } else {
      if (active_m == 2) {
        auto acc0 = vec::zeros(tag);
        auto acc1 = vec::zeros(tag);
        for (nint_t k = 0; k < K; ++k) {
          const auto bv = load_b<Tag, Case>(tag, B + k * N + n, active_n);
          acc0 = vec::fmadd(vec::fill(tag, A[m * K + k]), bv, acc0);
          acc1 = vec::fmadd(vec::fill(tag, A[(m + 1) * K + k]), bv, acc1);
        }
        store_c<Tag, Case>(tag, C + m * N + n, acc0, active_n);
        store_c<Tag, Case>(tag, C + (m + 1) * N + n, acc1, active_n);
      } else {
        one_row(m);
      }
    }
  }
  (void)M;
}

template <typename Policy, vec::VectorTag BaseTag,
          typename M, typename N, typename TN>
VECOPS_ALWAYS_INLINE void vector_gemm(
    BaseTag tag,
    const float* A, const float* B, float* C,
    M M_value, N N_value, nint_t K, TN tn) {
  const nint_t M_int = static_cast<nint_t>(M_value);
  const nint_t N_int = static_cast<nint_t>(N_value);
  hop::tile2d<Policy>(M_value, N_value, cint<1>, tn, GemmCatalog{},
      [&]<typename Case>(Case,
                         nint_t m, nint_t n,
                         nint_t active_m, nint_t active_n) {
        gemm_tile<BaseTag, Case>(
            tag, A, B, C, M_int, N_int, K,
            m, n, active_m, active_n);
      });
}

void reference_gemm(
    const float* A, const float* B, float* C,
    nint_t M, nint_t N, nint_t K) {
  for (nint_t m = 0; m < M; ++m) {
    for (nint_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (nint_t k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
}

template <typename Policy>
void check_vector_gemm(nint_t M, nint_t N, nint_t K) {
  GemmBaseTag tag{};
  const nint_t lanes = vec::size(tag);
  std::mt19937 rng(1234 + static_cast<unsigned>(M * 31 + N * 7 + K));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> A(static_cast<std::size_t>(M * K));
  std::vector<float> B(static_cast<std::size_t>(K * N));
  std::vector<float> actual(static_cast<std::size_t>(M * N), -17.0f);
  std::vector<float> expected(static_cast<std::size_t>(M * N), 0.0f);
  std::generate(A.begin(), A.end(), [&] { return dist(rng); });
  std::generate(B.begin(), B.end(), [&] { return dist(rng); });

  vector_gemm<Policy>(
      tag, A.data(), B.data(), actual.data(),
      Any{M}, Any{N}, K, dyn<1>(lanes));
  reference_gemm(A.data(), B.data(), expected.data(), M, N, K);

  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 2.0e-5f) << "at linear index " << i;
  }
}

TEST(Tile2DGemmTest, RowMajorHandlesFullAndTailTiles) {
  const nint_t lanes = vec::size(GemmBaseTag{});
  check_vector_gemm<hop::tile2d_policy::RowMajor>(5, 3 * lanes + 3, 11);
}

TEST(Tile2DGemmTest, FourRegionsHandlesFullAndTailTiles) {
  const nint_t lanes = vec::size(GemmBaseTag{});
  check_vector_gemm<hop::tile2d_policy::FourRegions>(5, 3 * lanes + 3, 11);
}

TEST(Tile2DGemmTest, UniformHandlesFullAndTailTiles) {
  const nint_t lanes = vec::size(GemmBaseTag{});
  check_vector_gemm<hop::tile2d_policy::Uniform>(5, 3 * lanes + 3, 11);
}

TEST(Tile2DGemmTest, RuntimeVectorLengthExactMultipleIsCorrect) {
  const nint_t lanes = vec::size(GemmBaseTag{});
  check_vector_gemm<hop::tile2d_policy::RowMajor>(4, 4 * lanes, 13);
}

} // namespace

// These two externally visible functions are intentionally kept simple so the
// optimized test binary can be inspected with objdump.  The fixed probe proves
// compile-time tail removal.  On scalable SVE, the second probe checks whether
// the compiler also eliminates the conservative tail after inlining the value
// relationship N == 2 * svcntw().
#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
extern "C" VECOPS_NOINLINE void tile2d_gemm_fixed64_probe(
    const float* A, const float* B, float* C, nint_t K) {
  using Tag = vec::FixedTag<float, 16>;
  vector_gemm<hop::tile2d_policy::RowMajor>(
      Tag{}, A, B, C, cint<2>, cint<32>, K, cint<16>);
}
#endif

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
extern "C" VECOPS_NOINLINE void tile2d_gemm_scalable_probe(
    const float* A, const float* B, float* C, nint_t K) {
  using Tag = vec::ScalableTag<float>;
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  vector_gemm<hop::tile2d_policy::RowMajor>(
      tag, A, B, C, cint<2>, Any{2 * lanes}, K, Any{lanes});
}
#endif
