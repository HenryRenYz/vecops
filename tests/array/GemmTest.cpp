//
// Created by renyz on 2026/5/9.
//
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <memory>
#include <random>
#include <cmath>

#include "vecops/array/Array.h"
#include "vecops/array/Gemm.h"
#include "vecops/CoreDefs.h"

using namespace vecops;
using namespace vecops::array;

// ============================================================================
// Reference GEMM
// ============================================================================

template <typename TA, typename TB, typename TC>
void ref_gemm(int M, int N, int K,
              const TA* A, int lda, int lda_k,
              const TB* B, int ldb, int ldb_k,
              TC* C, int ldc, int ldc_n)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<float>(A[m * lda + k * lda_k]) *
               static_cast<float>(B[n * ldb + k * ldb_k]);
      }
      C[m * ldc + n * ldc_n] = static_cast<TC>(acc);
    }
  }
}

// ============================================================================
// Scalar Kernels
// ============================================================================

struct ScalarAMXBf16Kernel {
  static constexpr int Mtile = 16;
  static constexpr int Ntile = 16;
  static constexpr int Ktile = 32;
  using TAccumulator = float;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <bool Accumulate,
            typename TA, typename ALayout,
            typename TB, typename BLayout>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TAccumulator* acc, int acc_ld)
      const
  {
    int tile_m = A_layout.shape().template get<0>();
    int K = A_layout.shape().template get<1>();
    int tile_n = B_layout.shape().template get<0>();

    int lda = A_layout.stride().template get<0>();
    int ldk_a = A_layout.stride().template get<1>();
    int ldb = B_layout.stride().template get<0>();
    int ldk_b = B_layout.stride().template get<1>();

    for (int m = 0; m < tile_m; ++m) {
      for (int n = 0; n < tile_n; ++n) {
        float val = Accumulate ? static_cast<float>(acc[m * acc_ld + n])
                               : 0.0f;
        for (int k = 0; k < K; ++k) {
          float av = static_cast<float>(A[m * lda + k * ldk_a]);
          float bv = static_cast<float>(B[n * ldb + k * ldk_b]);
          val += av * bv;
        }
        acc[m * acc_ld + n] = static_cast<TAccumulator>(val);
      }
    }
  }
};

struct ScalarSMEFp32Kernel {
  static constexpr int Mtile = 16;
  static constexpr int Ntile = 16;
  static constexpr int Ktile = 16;
  using TAccumulator = float;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <bool Accumulate,
            typename TA, typename ALayout,
            typename TB, typename BLayout>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TAccumulator* acc, int acc_ld)
      const
  {
    int tile_m = A_layout.shape().template get<0>();
    int K = A_layout.shape().template get<1>();
    int tile_n = B_layout.shape().template get<0>();

    int lda = A_layout.stride().template get<0>();
    int ldk_a = A_layout.stride().template get<1>();
    int ldb = B_layout.stride().template get<0>();
    int ldk_b = B_layout.stride().template get<1>();

    for (int m = 0; m < tile_m; ++m) {
      for (int n = 0; n < tile_n; ++n) {
        float val = Accumulate ? static_cast<float>(acc[m * acc_ld + n])
                               : 0.0f;
        for (int k = 0; k < K; ++k) {
          val += static_cast<float>(A[m * lda + k * ldk_a]) *
                 static_cast<float>(B[n * ldb + k * ldk_b]);
        }
        acc[m * acc_ld + n] = static_cast<TAccumulator>(val);
      }
    }
  }
};

// ============================================================================
// Kernel type traits (needed because TA/TB/TC are not deducible from Layout)
// ============================================================================

template <typename Kernel> struct KernelTraits;

template <> struct KernelTraits<ScalarAMXBf16Kernel> {
  using TA = bfloat16_t; using TB = bfloat16_t; using TC = float;
};
template <> struct KernelTraits<ScalarSMEFp32Kernel> {
  using TA = float; using TB = float; using TC = float;
};

// ============================================================================
// Shared test helpers
// ============================================================================

static auto identity_epilog = [](int, int, float val) -> float {
  return val;
};

// Runtime layout factories (Any-based)
namespace lt {
  inline auto A_row(int M, int K) { return make_layout(make_shape(M, K), make_stride(K, 1)); }
  inline auto A_col(int M, int K) { return make_layout(make_shape(M, K), make_stride(1, M)); }
  inline auto B_col(int N, int K) { return make_layout(make_shape(N, K), make_stride(1, N)); }
  inline auto B_row(int N, int K) { return make_layout(make_shape(N, K), make_stride(K, 1)); }
  inline auto C_row(int M, int N) { return make_layout(make_shape(M, N), make_stride(N, 1)); }
  inline auto C_col(int M, int N) { return make_layout(make_shape(M, N), make_stride(1, M)); }
}

// ============================================================================
// Test Fixture
// ============================================================================

class GemmTest : public ::testing::Test {
protected:
  std::mt19937 rng{42};

  void fill_random(std::vector<float>& v, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
  }

  template <typename T>
  void fill_random_bf16(std::vector<T>& v, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = static_cast<T>(dist(rng));
  }

  template <typename Kernel, typename Scheduler,
            typename ALayout, typename BLayout,
            typename CLayout, typename TShape,
            typename EpilogFn>
  void run_gemm_test(
      int M, int N, int K,
      int lda, int lda_k, int ldb, int ldb_k, int ldc, int ldc_n,
      const ALayout& a_layout, const BLayout& b_layout,
      const CLayout& c_layout, const TShape& tiles_shape,
      const EpilogFn& epilog, float tol,
      bool c_is_col_major = false, bool apply_relu_ref = false)
  {
    using TA = typename KernelTraits<Kernel>::TA;
    using TB = typename KernelTraits<Kernel>::TB;
    using TC = typename KernelTraits<Kernel>::TC;
    std::vector<TA> A_data(static_cast<size_t>(M) * K);
    std::vector<TB> B_data(static_cast<size_t>(N) * K);
    std::vector<TC> C_data(static_cast<size_t>(M) * N, TC{0});
    std::vector<TC> C_ref(static_cast<size_t>(M) * N, TC{0});

    if constexpr (std::is_same_v<TA, float>)
      fill_random(A_data, -1.0f, 1.0f);
    else
      fill_random_bf16(A_data, -1.0f, 1.0f);

    if constexpr (std::is_same_v<TB, float>)
      fill_random(B_data, -1.0f, 1.0f);
    else
      fill_random_bf16(B_data, -1.0f, 1.0f);

    ref_gemm(M, N, K,
             A_data.data(), lda, lda_k,
             B_data.data(), ldb, ldb_k,
             C_ref.data(), ldc, ldc_n);

    if (apply_relu_ref) {
      for (auto& x : C_ref) x = std::max(TC{0}, x);
    }

    auto shape_mnk = make_shape(M, N, K);
    Kernel kernel;
    Scheduler scheduler{};
    gemm(shape_mnk,
         A_data.data(), a_layout,
         B_data.data(), b_layout,
         C_data.data(), c_layout,
         tiles_shape, kernel, scheduler, epilog);

    if (c_is_col_major) {
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          size_t off = static_cast<size_t>(m) + static_cast<size_t>(n) * static_cast<size_t>(M);
          EXPECT_NEAR(float(C_data[off]), float(C_ref[off]), tol)
              << "M=" << M << " N=" << N << " K=" << K
              << "  at (" << m << "," << n << ")";
        }
      }
    } else {
      size_t total = static_cast<size_t>(M) * N;
      for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(float(C_data[i]), float(C_ref[i]), tol)
            << "M=" << M << " N=" << N << " K=" << K
            << "  at index " << i;
      }
    }
  }
};

// ============================================================================
// Size vectors used across tests
// ============================================================================

// Full list of varied (M,N,K) — odd, boundary, prime, tiny, large
static const std::tuple<int,int,int> g_all_sizes[] = {
  {335, 34, 605}, {14, 36, 16}, {16, 32, 16}, {512, 512, 512},
  {256, 32, 256}, {32, 32, 64}, {1024, 32, 1024}, {32, 29, 32},
  {16, 16, 16}, {64, 16, 64}, {3, 5, 7}, {65, 32, 64},
  {64, 32, 70}, {64, 32, 29}, {31, 62, 31}, {15, 62, 15},
  {13, 127, 63}, {63, 127, 13}, {32767, 95, 17}, {9, 192, 152},
  {416, 128, 40}, {10, 27, 64}, {32, 1, 32}, {354, 445, 156},
  {325, 173, 776}, {5, 45, 8}, {13, 32, 16},
};

// Sorted ascending by work (M*N*K) for bf16 / limited-run tests
static const std::tuple<int,int,int> g_small_sizes[] = {
  {3, 5, 7}, {5, 45, 8}, {10, 27, 64}, {13, 32, 16},
  {14, 36, 16}, {16, 16, 16}, {16, 32, 16}, {32, 1, 32},
  {32, 29, 32}, {32, 32, 64}, {64, 16, 64}, {65, 32, 64},
  {64, 32, 29}, {64, 32, 70}, {256, 32, 256},
};

// Subset for SchedulerConsistency and ReLU
static const std::tuple<int,int,int> g_quick_sizes[] = {
  {3, 5, 7}, {16, 32, 16}, {32, 29, 32}, {64, 32, 70},
  {13, 127, 63}, {256, 32, 256}, {512, 512, 512}, {325, 173, 776},
};

// Tileshape must be a multiple of 16 (Mtile, Ntile)
static constexpr int kMTile = 16;
static constexpr int kNTile = 16;

inline int round_up_mtile(int x) { return ((x + kMTile - 1) / kMTile) * kMTile; }
inline int round_up_ntile(int x) { return ((x + kNTile - 1) / kNTile) * kNTile; }

#define TILE_2D(M, N) make_shape(M, N)
#define TILE_3D(M, N, Kt) make_shape(M, N, Kt)
#define BASIC_TILE(M, N) make_shape(round_up_mtile(M), round_up_ntile(N))

// ============================================================================
// Basic correctness — varied sizes, default row-major, no tiling
// ============================================================================

TEST_F(GemmTest, SMEFp32_Basic) {
  for (auto [M,N,K] : g_all_sizes) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-5f);
  }
}

TEST_F(GemmTest, AMXBf16_Basic) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-2f);
  }
}

// ============================================================================
// Tiling — 2D outer tiles (M, N only)
// ============================================================================

TEST_F(GemmTest, SMEFp32_Tiling2D) {
  struct { int M,N,K, Mt,Nt; } t[] = {
    {335, 34, 605, 320, 32}, {512, 512, 512, 256, 256},
    {256, 32, 256, 128, 32}, {32, 32, 64, 32, 32},
    {16, 16, 16, 16, 16},   {64, 16, 64, 64, 16},
    {354, 445, 156, 336, 432}, {32, 29, 32, 32, 16},
    {416, 128, 40, 256, 128},
  };
  for (auto [M,N,K, Mt,Nt] : t) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        TILE_2D(Mt,Nt), identity_epilog, 1e-5f);
  }
}

// ============================================================================
// Tiling — 3D outer tiles (M, N, K) with K accumulation
// ============================================================================

TEST_F(GemmTest, SMEFp32_Tiling3D) {
  struct { int M,N,K, Mt,Nt,Kt; } t[] = {
    {512, 512, 512, 256, 256, 128},
    {256, 32, 256, 128, 32, 128},
    {32, 32, 64, 32, 32, 32},
    {64, 16, 64, 64, 16, 32},
    {325, 173, 776, 320, 160, 256},
    {416, 128, 40, 256, 128, 32},
    {32, 29, 32, 32, 16, 16},
  };
  for (auto [M,N,K, Mt,Nt,Kt] : t) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        TILE_3D(Mt,Nt,Kt), identity_epilog, 1e-5f);
  }
}

// ============================================================================
// SchedulerConsistency — MaxCases == MinCases
// ============================================================================

TEST_F(GemmTest, SchedulerConsistency) {
  for (auto [M,N,K] : g_quick_sizes) {
    std::vector<float> A_data(static_cast<size_t>(M) * K);
    std::vector<float> B_data(static_cast<size_t>(N) * K);
    std::vector<float> C_max(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> C_min(static_cast<size_t>(M) * N, 0.0f);
    fill_random(A_data);
    fill_random(B_data);

    auto a_l = lt::A_row(M,K);
    auto b_l = lt::B_col(N,K);
    auto c_l = lt::C_row(M,N);
    auto shape_mnk = make_shape(M, N, K);
    auto ts = BASIC_TILE(M,N);
    ScalarSMEFp32Kernel kernel;

    SchedulerMaxCases maxS;
    SchedulerMinCases minS;
    gemm(shape_mnk, A_data.data(), a_l, B_data.data(), b_l, C_max.data(), c_l,
         ts, kernel, maxS, identity_epilog);
    gemm(shape_mnk, A_data.data(), a_l, B_data.data(), b_l, C_min.data(), c_l,
         ts, kernel, minS, identity_epilog);

    for (int i = 0; i < M * N; ++i)
      EXPECT_FLOAT_EQ(C_max[i], C_min[i])
          << "M=" << M << " N=" << N << " K=" << K << " idx=" << i;
  }
}

// ============================================================================
// Layout variation — B row-major (stored as N×K, inner-stride=1 along K)
// ============================================================================

TEST_F(GemmTest, SMEFp32_BRowMajor) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, K,1, N,1,
        lt::A_row(M,K), lt::B_row(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-5f);
  }
}

// ============================================================================
// Layout variation — A column-major (runtime gather path)
// ============================================================================

TEST_F(GemmTest, SMEFp32_AColMajor) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, 1,N, N,1,
        lt::A_col(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-5f);
  }
}

// ============================================================================
// Layout variation — C column-major (non-contiguous output)
// ============================================================================

TEST_F(GemmTest, SMEFp32_NonContigC) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, 1,M,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_col(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-5f,
        /*c_is_col_major=*/true);
  }
}

// ============================================================================
// Layout variation — all column-major (A col, B rowK, C col) + K-tiling
// ============================================================================

TEST_F(GemmTest, SMEFp32_AllColMajor) {
  struct { int M,N,K, Mt,Nt,Kt; } t[] = {
    {64, 16, 64, 64, 16, 32},
    {32, 32, 64, 32, 32, 32},
    {65, 32, 64, 64, 32, 32},
    {256, 32, 256, 256, 32, 128},
    {512, 512, 512, 256, 256, 128},
  };
  for (auto [M,N,K, Mt,Nt,Kt] : t) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, K,1, 1,M,
        lt::A_col(M,K), lt::B_row(N,K), lt::C_col(M,N),
        TILE_3D(Mt,Nt,Kt), identity_epilog, 1e-5f,
        /*c_is_col_major=*/true);
  }
}

// ============================================================================
// Compile-time layouts — all strides are Int<N>
// ============================================================================

TEST_F(GemmTest, SMEFp32_CT_RowMajor) {
  auto run = [this]<int M, int N, int K>() {
    constexpr int Mt = ((M + 15) / 16) * 16;
    constexpr int Nt = ((N + 15) / 16) * 16;
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        Layout(Shape<Int<M>,Int<K>>{}, Stride<Int<K>,Int<1>>{}),
        Layout(Shape<Int<N>,Int<K>>{}, Stride<Int<1>,Int<N>>{}),
        Layout(Shape<Int<M>,Int<N>>{}, Stride<Int<N>,Int<1>>{}),
        Shape<Int<Mt>,Int<Nt>>{},
        identity_epilog, 1e-5f);
  };
  run.template operator()<16, 32, 16>();
  run.template operator()<3, 5, 7>();
  run.template operator()<32, 29, 32>();
  run.template operator()<64, 16, 64>();
  run.template operator()<31, 62, 31>();
  run.template operator()<13, 32, 16>();
  run.template operator()<5, 45, 8>();
  run.template operator()<10, 27, 64>();
}

TEST_F(GemmTest, SMEFp32_CT_ColMixed) {
  auto run = [this]<int M, int N, int K>() {
    constexpr int Mt = ((M + 15) / 16) * 16;
    constexpr int Nt = ((N + 15) / 16) * 16;
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, K,1, 1,M,
        Layout(Shape<Int<M>,Int<K>>{}, Stride<Int<1>,Int<M>>{}),
        Layout(Shape<Int<N>,Int<K>>{}, Stride<Int<K>,Int<1>>{}),
        Layout(Shape<Int<M>,Int<N>>{}, Stride<Int<1>,Int<M>>{}),
        Shape<Int<Mt>,Int<Nt>,Int<32>>{},
        identity_epilog, 1e-5f,
        /*c_is_col_major=*/true);
  };
  run.template operator()<16, 32, 16>();
  run.template operator()<32, 32, 64>();
  run.template operator()<64, 16, 64>();
  run.template operator()<13, 127, 63>();
  run.template operator()<14, 36, 16>();
}

TEST_F(GemmTest, AMXBf16_CT) {
  auto run = [this]<int M, int N, int K>() {
    constexpr int Mt = ((M + 15) / 16) * 16;
    constexpr int Nt = ((N + 15) / 16) * 16;
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        Layout(Shape<Int<M>,Int<K>>{}, Stride<Int<K>,Int<1>>{}),
        Layout(Shape<Int<N>,Int<K>>{}, Stride<Int<1>,Int<N>>{}),
        Layout(Shape<Int<M>,Int<N>>{}, Stride<Int<N>,Int<1>>{}),
        Shape<Int<Mt>,Int<Nt>>{},
        identity_epilog, 1e-2f);
  };
  run.template operator()<16, 32, 16>();
  run.template operator()<3, 5, 7>();
  run.template operator()<32, 32, 64>();
  run.template operator()<64, 16, 64>();
  run.template operator()<10, 27, 64>();
}

// ============================================================================
// Epilog — ReLU
// ============================================================================

TEST_F(GemmTest, SMEFp32_ReLU) {
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  for (auto [M,N,K] : g_quick_sizes) {
    run_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), relu, 1e-5f,
        /*c_col=*/false, /*apply_relu_ref=*/true);
  }
}

TEST_F(GemmTest, AMXBf16_ReLU) {
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  for (auto [M,N,K] : g_quick_sizes) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), relu, 1e-2f,
        /*c_col=*/false, /*apply_relu_ref=*/true);
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
