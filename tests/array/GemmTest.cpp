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
// Reference GEMM Implementation (uses per-dimension strides)
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
// Scalar Kernel 1: AMX BF16 -> FP32 simulator
// ============================================================================

struct ScalarAMXBf16Kernel {
  static constexpr int Mtile = 32;
  static constexpr int Ntile = 32;
  static constexpr int Ktile = 32;
  using TAccumulator = float;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <typename TA, typename ALayout,
            typename TB, typename BLayout,
            typename TC, typename CLayout,
            typename EpilogFn>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TC* C, CLayout C_layout,
      const EpilogFn& fn,
      bool accumulate = false,
      TAccumulator* acc_buffer = nullptr,
      int acc_buf_ld = 0,
      bool output_to_c = true)
  {
    int tile_m = A_layout.shape().template get<0>();
    int K = A_layout.shape().template get<1>();
    int tile_n = B_layout.shape().template get<0>();

    int lda = A_layout.stride().template get<0>();
    int ldk_a = A_layout.stride().template get<1>();
    int ldb = B_layout.stride().template get<0>();
    int ldk_b = B_layout.stride().template get<1>();
    int ldc = C_layout.stride().template get<0>();
    int ldc_n = C_layout.stride().template get<1>();

    int buf_ld = acc_buf_ld > 0 ? acc_buf_ld : tile_n;
    bool write_buf = acc_buffer && !output_to_c;

    for (int m = 0; m < tile_m; ++m) {
      for (int n = 0; n < tile_n; ++n) {
        float acc = (accumulate && acc_buffer) ? acc_buffer[m * buf_ld + n]
                   : (accumulate && !acc_buffer) ? static_cast<float>(C[m * ldc + n * ldc_n])
                   : 0.0f;
        for (int k = 0; k < K; ++k) {
          float av = static_cast<float>(A[m * lda + k * ldk_a]);
          float bv = static_cast<float>(B[n * ldb + k * ldk_b]);
          acc += av * bv;
        }
        if (write_buf)
          acc_buffer[m * buf_ld + n] = static_cast<TAccumulator>(acc);
        else
          C[m * ldc + n * ldc_n] = static_cast<TC>(fn(m, n, acc));
      }
    }
  }
};

// ============================================================================
// Scalar Kernel 2: ARM SME FP32 -> FP32 simulator
// ============================================================================

struct ScalarSMEFp32Kernel {
  static constexpr int Mtile = 16;
  static constexpr int Ntile = 16;
  static constexpr int Ktile = 16;
  using TAccumulator = float;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <typename TA, typename ALayout,
            typename TB, typename BLayout,
            typename TC, typename CLayout,
            typename EpilogFn>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TC* C, CLayout C_layout,
      const EpilogFn& fn,
      bool accumulate = false,
      TAccumulator* acc_buffer = nullptr,
      int acc_buf_ld = 0,
      bool output_to_c = true)
  {
    int tile_m = A_layout.shape().template get<0>();
    int K = A_layout.shape().template get<1>();
    int tile_n = B_layout.shape().template get<0>();

    int lda = A_layout.stride().template get<0>();
    int ldk_a = A_layout.stride().template get<1>();
    int ldb = B_layout.stride().template get<0>();
    int ldk_b = B_layout.stride().template get<1>();
    int ldc = C_layout.stride().template get<0>();
    int ldc_n = C_layout.stride().template get<1>();

    int buf_ld = acc_buf_ld > 0 ? acc_buf_ld : tile_n;
    bool write_buf = acc_buffer && !output_to_c;

    for (int m = 0; m < tile_m; ++m) {
      for (int n = 0; n < tile_n; ++n) {
        float acc = (accumulate && acc_buffer) ? acc_buffer[m * buf_ld + n]
                   : (accumulate && !acc_buffer) ? static_cast<float>(C[m * ldc + n * ldc_n])
                   : 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += static_cast<float>(A[m * lda + k * ldk_a]) *
                 static_cast<float>(B[n * ldb + k * ldk_b]);
        }
        if (write_buf)
          acc_buffer[m * buf_ld + n] = static_cast<TAccumulator>(acc);
        else
          C[m * ldc + n * ldc_n] = static_cast<TC>(fn(m, n, acc));
      }
    }
  }
};

// ============================================================================
// Helper: Identity Epilog
// ============================================================================

static auto identity_epilog = [](int, int, float val) -> float {
  return val;
};

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

  template <typename Kernel, typename Scheduler, typename TA, typename TB, typename TC>
  void run_gemm(int M, int N, int K,
                const std::vector<TA>& A_data,
                const std::vector<TB>& B_data,
                std::vector<TC>& C_data,
                const auto& epilog,
                int Mt = 0, int Nt = 0, int Kt = 0)
  {
    auto shape_mnk = make_shape(M, N, K);
    auto a_layout = make_layout(make_shape(M, K), make_stride(K, 1));
    auto b_layout = make_layout(make_shape(N, K), make_stride(1, N));
    auto c_layout = make_layout(make_shape(M, N), make_stride(N, 1));

    Kernel kernel;
    Scheduler scheduler{};
    if (Mt > 0 && Kt > 0) {
      auto ts = make_shape(Mt, Nt, Kt);
      gemm(shape_mnk, A_data.data(), a_layout, B_data.data(), b_layout,
           C_data.data(), c_layout, ts, kernel, scheduler, epilog);
    } else if (Mt > 0) {
      auto ts = make_shape(Mt, Nt);
      gemm(shape_mnk, A_data.data(), a_layout, B_data.data(), b_layout,
           C_data.data(), c_layout, ts, kernel, scheduler, epilog);
    } else {
      auto ts = make_shape(M, N);
      gemm(shape_mnk, A_data.data(), a_layout, B_data.data(), b_layout,
           C_data.data(), c_layout, ts, kernel, scheduler, epilog);
    }
  }
};

// ============================================================================
// AMX BF16 — SchedulerMaxCases Tests
// ============================================================================

TEST_F(GemmTest, AMXBf16_Max_ExactTile) {
  constexpr int M = 64, N = 64, K = 64;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Max_WithTiling) {
  constexpr int M = 100, N = 120, K = 128;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Max_Large) {
  constexpr int M = 200, N = 250, K = 300;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Max_ReLU) {
  constexpr int M = 100, N = 80, K = 128;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data, -2.f, 2.f); fill_random_bf16(B_data, -2.f, 2.f);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  for (auto& x : C_ref) x = std::max(0.0f, x);
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  run_gemm<ScalarAMXBf16Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, relu);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

// ============================================================================
// AMX BF16 — SchedulerMinCases Tests
// ============================================================================

TEST_F(GemmTest, AMXBf16_Min_ExactTile) {
  constexpr int M = 64, N = 64, K = 64;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Min_WithTiling) {
  constexpr int M = 100, N = 120, K = 128;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Min_Large) {
  constexpr int M = 200, N = 250, K = 300;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data); fill_random_bf16(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarAMXBf16Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, AMXBf16_Min_ReLU) {
  constexpr int M = 100, N = 80, K = 128;
  std::vector<bfloat16_t> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random_bf16(A_data, -2.f, 2.f); fill_random_bf16(B_data, -2.f, 2.f);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  for (auto& x : C_ref) x = std::max(0.0f, x);
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  run_gemm<ScalarAMXBf16Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, relu);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-2f) << "Mismatch at index " << i;
}

// ============================================================================
// SME FP32 — SchedulerMaxCases Tests
// ============================================================================

TEST_F(GemmTest, SMEFp32_Max_ExactTile) {
  constexpr int M = 32, N = 32, K = 32;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Max_WithTiling) {
  constexpr int M = 80, N = 56, K = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Max_Large) {
  constexpr int M = 200, N = 180, K = 256;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Max_ReLU) {
  constexpr int M = 80, N = 64, K = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data, -2.f, 2.f); fill_random(B_data, -2.f, 2.f);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  for (auto& x : C_ref) x = std::max(0.0f, x);
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, relu);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

// ============================================================================
// SME FP32 — SchedulerMinCases Tests
// ============================================================================

TEST_F(GemmTest, SMEFp32_Min_ExactTile) {
  constexpr int M = 32, N = 32, K = 32;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Min_WithTiling) {
  constexpr int M = 80, N = 56, K = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Min_Large) {
  constexpr int M = 200, N = 180, K = 256;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Min_ReLU) {
  constexpr int M = 80, N = 64, K = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data, -2.f, 2.f); fill_random(B_data, -2.f, 2.f);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  for (auto& x : C_ref) x = std::max(0.0f, x);
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, relu);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

// ============================================================================
// Cross-validation: same data, different schedulers must match
// ============================================================================

TEST_F(GemmTest, SchedulerConsistency) {
  constexpr int M = 100, N = 120, K = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_max(M * N, 0.0f), C_min(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);

  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_max, identity_epilog);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_min, identity_epilog);

  for (int i = 0; i < M * N; ++i)
    EXPECT_FLOAT_EQ(C_max[i], C_min[i]) << "Scheduler mismatch at index " << i;
}

// ============================================================================
// TilesShape Tests — 2D L2 tiling (M, N only)
// ============================================================================

TEST_F(GemmTest, SMEFp32_Tiles2D) {
  constexpr int M = 120, N = 100, K = 128;
  constexpr int Mt = 64, Nt = 64;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog, Mt, Nt);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Max_Tiles2D) {
  constexpr int M = 200, N = 180, K = 256;
  constexpr int Mt = 128, Nt = 64;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog, Mt, Nt);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

// ============================================================================
// TilesShape Tests — 3D L2 tiling (M, N, K)
// ============================================================================

TEST_F(GemmTest, SMEFp32_Tiles3D_NoBuffer) {
  constexpr int M = 64, N = 64, K = 128;
  constexpr int Mt = 32, Nt = 64, Kt = 64;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMinCases>(M, N, K, A_data, B_data, C_data, identity_epilog, Mt, Nt, Kt);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

TEST_F(GemmTest, SMEFp32_Max_Tiles3D) {
  constexpr int M = 128, N = 128, K = 256;
  constexpr int Mt = 128, Nt = 64, Kt = 128;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);
  run_gemm<ScalarSMEFp32Kernel, SchedulerMaxCases>(M, N, K, A_data, B_data, C_data, identity_epilog, Mt, Nt, Kt);
  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(C_data[i], C_ref[i], 1e-5f) << "Mismatch at index " << i;
}

// ============================================================================
// TilesShape Tests — 3D with non-contiguous C (buffer path)
// ============================================================================

TEST_F(GemmTest, SMEFp32_Tiles3D_NonContigC) {
  constexpr int M = 64, N = 64, K = 128;
  constexpr int Mt = 64, Nt = 32, Kt = 64;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);

  auto shape_mnk = make_shape(M, N, K);
  auto a_layout = make_layout(make_shape(M, K), make_stride(K, 1));
  auto b_layout = make_layout(make_shape(N, K), make_stride(1, N));
  auto c_layout = make_layout(make_shape(M, N), make_stride(1, M));
  auto ts = make_shape(Mt, Nt, Kt);
  ScalarSMEFp32Kernel kernel;
  SchedulerMinCases scheduler{};
  gemm(shape_mnk, A_data.data(), a_layout, B_data.data(), b_layout,
       C_data.data(), c_layout, ts, kernel, scheduler, identity_epilog);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      EXPECT_NEAR(C_data[m + n * M], C_ref[m * N + n], 1e-5f)
          << "Mismatch at (" << m << "," << n << ")";
}

TEST_F(GemmTest, SMEFp32_Tiles3D_NonContigC_MaxCases) {
  constexpr int M = 80, N = 64, K = 128;
  constexpr int Mt = 64, Nt = 64, Kt = 64;
  std::vector<float> A_data(M * K), B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f), C_ref(M * N, 0.0f);
  fill_random(A_data); fill_random(B_data);
  ref_gemm(M, N, K, A_data.data(), K, 1, B_data.data(), 1, N, C_ref.data(), N, 1);

  auto shape_mnk = make_shape(M, N, K);
  auto a_layout = make_layout(make_shape(M, K), make_stride(K, 1));
  auto b_layout = make_layout(make_shape(N, K), make_stride(1, N));
  auto c_layout = make_layout(make_shape(M, N), make_stride(1, M));
  auto ts = make_shape(Mt, Nt, Kt);
  ScalarSMEFp32Kernel kernel;
  SchedulerMaxCases scheduler{};
  gemm(shape_mnk, A_data.data(), a_layout, B_data.data(), b_layout,
       C_data.data(), c_layout, ts, kernel, scheduler, identity_epilog);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      EXPECT_NEAR(C_data[m + n * M], C_ref[m * N + n], 1e-5f)
          << "Mismatch at (" << m << "," << n << ")";
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
