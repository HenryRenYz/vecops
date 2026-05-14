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
// Helper: initialize accumulator based on AccumulateT type
//   bool → runtime ternary; bool_constant<false> → compile-time DCE
// ============================================================================

template <typename AccumulateT, typename TAcc>
VECOPS_ALWAYS_INLINE float init_acc(const TAcc* acc, int idx, AccumulateT acc_flag) {
    if constexpr (std::is_same_v<std::decay_t<AccumulateT>, std::bool_constant<false>>)
        return 0.0f;
    else
        return acc_flag ? static_cast<float>(acc[idx]) : 0.0f;
}

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
  static constexpr int Wd = 2;
  using TAccumulator = float;

  static constexpr bool pack_A_M_on_dim2 = true;
  static constexpr bool pack_B_N_on_dim2 = false;
  static constexpr int A_block_dim2 = Mtile;
  static constexpr int A_block_dim3 = Ktile;
  static constexpr int B_block_dim2 = Ktile / Wd;
  static constexpr int B_block_dim3 = Ntile * Wd;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <typename AccumulateT,
            typename TA, typename ALayout,
            typename TB, typename BLayout,
            typename TC, typename CLayout,
            typename EpilogFn>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TAccumulator* acc, int acc_ld,
      TC* C, CLayout C_layout,
      int offM, int offN,
      const EpilogFn& fn,
      int validM, int validN,
      AccumulateT accumulate)
      const
  {
    constexpr bool a_packed = (ALayout::Ndim >= 4);
    constexpr bool b_packed = (BLayout::Ndim >= 4);
    constexpr int Mt = Mtile;
    constexpr int Nt = Ntile;
    constexpr int Kt = Ktile;
    constexpr int Wd_ = Wd;

    const int nM = A_layout.shape().template get<0>();
    const int nN = B_layout.shape().template get<0>();

    int a_ts = A_layout.stride().template get<0>();
    int b_ts = B_layout.stride().template get<0>();
    int a_ks = a_packed ? A_layout.stride().template get<1>()
                        : (Kt * A_layout.stride().template get<2>());
    int b_ks = b_packed ? B_layout.stride().template get<1>()
                        : (Kt * B_layout.stride().template get<2>());
    int k_a = a_packed ? (A_layout.shape().template get<1>() * Kt)
                       : A_layout.shape().template get<2>();
    int k_b = b_packed ? (B_layout.shape().template get<1>() * Kt)
                       : B_layout.shape().template get<2>();

    int c_s = C_layout.stride().template get<0>();
    int M = C_layout.shape().template get<0>();
    int N = C_layout.shape().template get<1>();

    auto load_a = [&](const TA* a, float* tile, int Mr, int Kr) {
      if constexpr (a_packed) {
        for (int m = 0; m < Mt; ++m)
          for (int k = 0; k < Kt; ++k)
            tile[m * Kt + k] = float(a[m * Kt + k]);
      } else {
        int a_ms = A_layout.stride().template get<1>();
        int a_ks_r = A_layout.stride().template get<2>();
        for (int m = 0; m < Mt; ++m)
          for (int k = 0; k < Kt; ++k)
            tile[m * Kt + k] = (m < Mr && k < Kr)
                ? float(a[m * a_ms + k * a_ks_r]) : 0.0f;
      }
    };
    auto load_b = [&](const TB* b, float* tile, int Nr, int Kr) {
      if constexpr (b_packed) {
        for (int n = 0; n < Nt; ++n)
          for (int k = 0; k < Kt; ++k)
            tile[n * Kt + k] = float(b[n * Kt + k]);
      } else {
        int b_ns = B_layout.stride().template get<1>();
        int b_ks_r = B_layout.stride().template get<2>();
        for (int n = 0; n < Nt; ++n)
          for (int k = 0; k < Kt; ++k) {
            int vnni_idx = (k / Wd_) * Kt + Wd_ * n + (k % Wd_);
            tile[vnni_idx] = (n < Nr && k < Kr)
                ? float(b[n * b_ns + k * b_ks_r]) : 0.0f;
          }
      }
    };

    alignas(64) float a_tiles[4][16][32];
    alignas(64) float b_tiles[4][16][32];
    alignas(64) float accum_tiles[4][4][16][16];

    auto tdpbf16ps = [&](float* dst, const float* a, const float* b) {
      for (int m = 0; m < Mt; ++m)
        for (int n = 0; n < Nt; ++n) {
          float tmp = dst[m * Nt + n];
          for (int k = 0; k < Kt / Wd_; ++k) {
            tmp += a[m * Kt + (2 * k + 0)] * b[k * Kt + (2 * n + 0)];
            tmp += a[m * Kt + (2 * k + 1)] * b[k * Kt + (2 * n + 1)];
          }
          dst[m * Nt + n] = tmp;
        }
    };

    for (int nm = 0; nm < nM; ++nm)
      for (int nn = 0; nn < nN; ++nn)
        for (int mt = 0; mt < Mt; ++mt)
          for (int nt = 0; nt < Nt; ++nt) {
            int m_off = nm * Mt + mt, n_off = nn * Nt + nt;
            accum_tiles[nm][nn][mt][nt] = (accumulate && m_off < M && n_off < N)
                ? float(acc[m_off * acc_ld + n_off]) : 0.0f;
          }

    int K_lim = std::min(k_a, k_b);
    for (int k = 0; k < K_lim; k += Kt) {
      int kr = std::min(K_lim - k, Kt);
      for (int nm = 0; nm < nM; ++nm) {
        int mr = std::min(M - nm * Mt, Mt);
        load_a(A + (k / Kt) * a_ks + nm * a_ts, &a_tiles[nm][0][0], mr, kr);
      }
      for (int nn = 0; nn < nN; ++nn) {
        int nr = std::min(N - nn * Nt, Nt);
        load_b(B + (k / Kt) * b_ks + nn * b_ts, &b_tiles[nn][0][0], nr, kr);
      }
      for (int nm = 0; nm < nM; ++nm)
        for (int nn = 0; nn < nN; ++nn)
          tdpbf16ps(&accum_tiles[nm][nn][0][0], &a_tiles[nm][0][0], &b_tiles[nn][0][0]);
    }

    for (int nm = 0; nm < nM; ++nm)
      for (int nn = 0; nn < nN; ++nn)
        for (int mt = 0; mt < Mt; ++mt)
          for (int nt = 0; nt < Nt; ++nt) {
            int m_off = nm * Mt + mt, n_off = nn * Nt + nt;
            if (m_off >= M || n_off >= N) continue;
            float v = fn(m_off + offM, n_off + offN,
                         accum_tiles[nm][nn][mt][nt]);
            if (acc)
              acc[m_off * acc_ld + n_off] = static_cast<TAccumulator>(v);
            else
              C[m_off * c_s + n_off] = static_cast<TC>(v);
          }
  }

  template <typename TL>
  using PackedALayout = Layout<
      Shape<Any, Any, Int<Mtile>, Int<Ktile>>,
      Stride<Any, Any, Int<Ktile>, Int<1>>
  >;

  template <typename TL>
  using PackedBLayout = Layout<
      Shape<Any, Any, Int<Ktile / Wd>, Int<Ntile * Wd>>,
      Stride<Any, Any, Int<Ntile * Wd>, Int<1>>
  >;

  template <typename T, typename InputLayout, typename PackedLayout = PackedALayout<InputLayout>>
  void pack_A(const T* A, InputLayout in_layout, T* A_packed, PackedLayout) const {
    constexpr int kMt = Mtile;
    constexpr int kKt = Ktile;
    int M = in_layout.shape().template get<0>();
    int K = in_layout.shape().template get<1>();
    int sa_m = int(in_layout.stride().template get<0>());
    int sa_k = int(in_layout.stride().template get<1>());

    int M_pad = ((M + kMt - 1) / kMt) * kMt;
    int K_pad = ((K + kKt - 1) / kKt) * kKt;
    int Mtiled = M_pad / kMt;
    int Ktiled = K_pad / kKt;

    for (int im = 0; im < Mtiled; ++im) {
      for (int ik = 0; ik < Ktiled; ++ik) {
        for (int m = 0; m < kMt; ++m) {
          for (int k = 0; k < kKt; ++k) {
            int m_global = im * kMt + m;
            int k_global = ik * kKt + k;
            T val = static_cast<T>(0);
            if (m_global < M && k_global < K)
              val = A[m_global * sa_m + k_global * sa_k];
            size_t out_off = static_cast<size_t>(im) * size_t(Ktiled) * size_t(kMt) * size_t(kKt)
                           + static_cast<size_t>(ik) * size_t(kMt) * size_t(kKt)
                           + static_cast<size_t>(m) * size_t(kKt)
                           + static_cast<size_t>(k);
            A_packed[out_off] = val;
          }
        }
      }
    }
  }

  template <typename T, typename InputLayout, typename PackedLayout = PackedBLayout<InputLayout>>
  void pack_B(const T* B, InputLayout in_layout, T* B_packed, PackedLayout) const {
    constexpr int kNt = Ntile;
    constexpr int kKt = Ktile;
    constexpr int kWd = Wd;
    int N = in_layout.shape().template get<0>();
    int K = in_layout.shape().template get<1>();
    int sb_n = int(in_layout.stride().template get<0>());
    int sb_k = int(in_layout.stride().template get<1>());

    int N_pad = ((N + kNt - 1) / kNt) * kNt;
    int K_pad = ((K + kKt - 1) / kKt) * kKt;
    int Ntiled = N_pad / kNt;
    int Ktiled = K_pad / kKt;

    constexpr int kWt = kKt / kWd;
    constexpr int kNw = kNt * kWd;

    for (int in = 0; in < Ntiled; ++in) {
      for (int ik = 0; ik < Ktiled; ++ik) {
        for (int kw = 0; kw < kWt; ++kw) {
          for (int x = 0; x < kNw; ++x) {
            int n = x / kWd;
            int ko = x % kWd;
            int n_global = in * kNt + n;
            int k_global = ik * kKt + kw * kWd + ko;
            T val = static_cast<T>(0);
            if (n_global < N && k_global < K)
              val = B[n_global * sb_n + k_global * sb_k];
            size_t out_off = static_cast<size_t>(in) * size_t(Ktiled) * size_t(kWt) * size_t(kNw)
                           + static_cast<size_t>(ik) * size_t(kWt) * size_t(kNw)
                           + static_cast<size_t>(kw) * size_t(kNw)
                           + static_cast<size_t>(x);
            B_packed[out_off] = val;
          }
        }
      }
    }
  }
};

struct ScalarSMEFp32Kernel {
  static constexpr int Mtile = 16;
  static constexpr int Ntile = 16;
  static constexpr int Ktile = 1;
  using TAccumulator = float;

  static constexpr bool pack_A_M_on_dim2 = false;
  static constexpr bool pack_B_N_on_dim2 = false;
  static constexpr int A_block_dim2 = Ktile;
  static constexpr int A_block_dim3 = Mtile;
  static constexpr int B_block_dim2 = Ktile;
  static constexpr int B_block_dim3 = Ntile;

  static constexpr Shape<Int<2>, Int<2>> shape_upper_left {};
  static constexpr Shape<Int<4>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<4>> shape_lower_left {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  template <typename AccumulateT,
            typename TA, typename ALayout,
            typename TB, typename BLayout,
            typename TC, typename CLayout,
            typename EpilogFn>
  VECOPS_ALWAYS_INLINE void run(
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TAccumulator* acc, int acc_ld,
      TC* C, CLayout C_layout,
      int offM, int offN,
      const EpilogFn& fn,
      int validM, int validN,
      AccumulateT accumulate)
      const
  {
    constexpr bool a_packed = (ALayout::Ndim >= 4);
    constexpr bool b_packed = (BLayout::Ndim >= 4);
    constexpr int Mt = Mtile;
    constexpr int Nt = Ntile;

    const int nM = A_layout.shape().template get<0>();
    const int nN = B_layout.shape().template get<0>();

    int a_ts = A_layout.stride().template get<0>();
    int b_ts = B_layout.stride().template get<0>();
    int a_ks = a_packed ? A_layout.stride().template get<1>()
                        : A_layout.stride().template get<2>();
    int b_ks = b_packed ? B_layout.stride().template get<1>()
                        : B_layout.stride().template get<2>();
    int K = a_packed ? A_layout.shape().template get<1>()
                     : A_layout.shape().template get<2>();

    int c_s = C_layout.stride().template get<0>();
    int M = C_layout.shape().template get<0>();
    int N = C_layout.shape().template get<1>();

    auto load_a = [&](const TA* a, float* vec, int Mr) {
      if constexpr (a_packed) {
        for (int m = 0; m < Mt; ++m) vec[m] = float(a[m]);
      } else {
        int a_ms = A_layout.stride().template get<1>();
        for (int m = 0; m < Mt; ++m)
          vec[m] = (m < Mr) ? float(a[m * a_ms]) : 0.0f;
      }
    };
    auto load_b = [&](const TB* b, float* vec, int Nr) {
      if constexpr (b_packed) {
        for (int n = 0; n < Nt; ++n) vec[n] = float(b[n]);
      } else {
        int b_ns = B_layout.stride().template get<1>();
        for (int n = 0; n < Nt; ++n)
          vec[n] = (n < Nr) ? float(b[n * b_ns]) : 0.0f;
      }
    };

    alignas(64) float a_vecs[4][16];
    alignas(64) float b_vecs[4][16];
    alignas(64) float accum_tiles[4][4][16][16];

    auto fmopa = [&](float* dst, const float* a, const float* b) {
      for (int m = 0; m < Mt; ++m) {
        float va = a[m];
        for (int n = 0; n < Nt; ++n)
          dst[m * Nt + n] += va * b[n];
      }
    };

    for (int nm = 0; nm < nM; ++nm)
      for (int nn = 0; nn < nN; ++nn)
        for (int mt = 0; mt < Mt; ++mt)
          for (int nt = 0; nt < Nt; ++nt) {
            int m_off = nm * Mt + mt, n_off = nn * Nt + nt;
            accum_tiles[nm][nn][mt][nt] = (accumulate && m_off < M && n_off < N)
                ? float(acc[m_off * acc_ld + n_off]) : 0.0f;
          }

    for (int k = 0; k < K; ++k) {
      for (int nm = 0; nm < nM; ++nm) {
        int mr = std::min(M - nm * Mt, Mt);
        load_a(A + k * a_ks + nm * a_ts, &a_vecs[nm][0], mr);
      }
      for (int nn = 0; nn < nN; ++nn) {
        int nr = std::min(N - nn * Nt, Nt);
        load_b(B + k * b_ks + nn * b_ts, &b_vecs[nn][0], nr);
      }
      for (int nm = 0; nm < nM; ++nm)
        for (int nn = 0; nn < nN; ++nn)
          fmopa(&accum_tiles[nm][nn][0][0], &a_vecs[nm][0], &b_vecs[nn][0]);
    }

    for (int nm = 0; nm < nM; ++nm)
      for (int nn = 0; nn < nN; ++nn)
        for (int mt = 0; mt < Mt; ++mt)
          for (int nt = 0; nt < Nt; ++nt) {
            int m_off = nm * Mt + mt, n_off = nn * Nt + nt;
            if (m_off >= M || n_off >= N) continue;
            float v = fn(m_off + offM, n_off + offN,
                         accum_tiles[nm][nn][mt][nt]);
            if (acc)
              acc[m_off * acc_ld + n_off] = static_cast<TAccumulator>(v);
            else
              C[m_off * c_s + n_off] = static_cast<TC>(v);
          }
  }

  template <typename TL>
  using PackedALayout = Layout<
      Shape<Any, Any, Int<Ktile>, Int<Mtile>>,
      Stride<Any, Any, Int<Mtile>, Int<1>>
  >;

  template <typename TL>
  using PackedBLayout = Layout<
      Shape<Any, Any, Int<Ktile>, Int<Ntile>>,
      Stride<Any, Any, Int<Ntile>, Int<1>>
  >;

  template <typename T, typename InputLayout, typename PackedLayout = PackedALayout<InputLayout>>
  void pack_A(const T* A, InputLayout in_layout, T* A_packed, PackedLayout) const {
    constexpr int kMt = Mtile;
    constexpr int kKt = Ktile;
    int M = in_layout.shape().template get<0>();
    int K = in_layout.shape().template get<1>();
    int sa_m = int(in_layout.stride().template get<0>());
    int sa_k = int(in_layout.stride().template get<1>());

    int M_pad = ((M + kMt - 1) / kMt) * kMt;
    int K_pad = ((K + kKt - 1) / kKt) * kKt;
    int Mtiled = M_pad / kMt;
    int Ktiled = K_pad / kKt;

    for (int im = 0; im < Mtiled; ++im) {
      for (int ik = 0; ik < Ktiled; ++ik) {
        for (int k = 0; k < kKt; ++k) {
          for (int m = 0; m < kMt; ++m) {
            int m_global = im * kMt + m;
            int k_global = ik * kKt + k;
            T val = static_cast<T>(0);
            if (m_global < M && k_global < K)
              val = A[m_global * sa_m + k_global * sa_k];
            size_t out_off = static_cast<size_t>(im) * size_t(Ktiled) * size_t(kKt) * size_t(kMt)
                           + static_cast<size_t>(ik) * size_t(kKt) * size_t(kMt)
                           + static_cast<size_t>(k) * size_t(kMt)
                           + static_cast<size_t>(m);
            A_packed[out_off] = val;
          }
        }
      }
    }
  }

  template <typename T, typename InputLayout, typename PackedLayout = PackedBLayout<InputLayout>>
  void pack_B(const T* B, InputLayout in_layout, T* B_packed, PackedLayout) const {
    constexpr int kNt = Ntile;
    constexpr int kKt = Ktile;
    int N = in_layout.shape().template get<0>();
    int K = in_layout.shape().template get<1>();
    int sb_n = int(in_layout.stride().template get<0>());
    int sb_k = int(in_layout.stride().template get<1>());

    int N_pad = ((N + kNt - 1) / kNt) * kNt;
    int K_pad = ((K + kKt - 1) / kKt) * kKt;
    int Ntiled = N_pad / kNt;
    int Ktiled = K_pad / kKt;

    for (int in = 0; in < Ntiled; ++in) {
      for (int ik = 0; ik < Ktiled; ++ik) {
        for (int k = 0; k < kKt; ++k) {
          for (int n = 0; n < kNt; ++n) {
            int n_global = in * kNt + n;
            int k_global = ik * kKt + k;
            T val = static_cast<T>(0);
            if (n_global < N && k_global < K)
              val = B[n_global * sb_n + k_global * sb_k];
            size_t out_off = static_cast<size_t>(in) * size_t(Ktiled) * size_t(kKt) * size_t(kNt)
                           + static_cast<size_t>(ik) * size_t(kKt) * size_t(kNt)
                           + static_cast<size_t>(k) * size_t(kNt)
                           + static_cast<size_t>(n);
            B_packed[out_off] = val;
          }
        }
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

  // Packed GEMM test — packs A,B then runs gemm via packed path
  template <typename Kernel, typename Scheduler,
            typename ALayout, typename BLayout,
            typename CLayout, typename EpilogFn>
  void run_packed_gemm_test(
      int M, int N, int K,
      const ALayout& a_in_layout, const BLayout& b_in_layout,
      const CLayout& c_layout,
      const EpilogFn& epilog, float tol,
      bool c_is_col_major = false)
  {
    using TA = typename KernelTraits<Kernel>::TA;
    using TB = typename KernelTraits<Kernel>::TB;
    using TC = typename KernelTraits<Kernel>::TC;
    constexpr int kMt = Kernel::Mtile;
    constexpr int kNt = Kernel::Ntile;
    constexpr int kKt = Kernel::Ktile;

    int M_pad = ((M + kMt - 1) / kMt) * kMt;
    int N_pad = ((N + kNt - 1) / kNt) * kNt;
    int K_pad = ((K + kKt - 1) / kKt) * kKt;
    int Mtiled = M_pad / kMt;
    int Ntiled = N_pad / kNt;
    int Ktiled = K_pad / kKt;

    std::vector<TA> A_in(static_cast<size_t>(M) * K);
    std::vector<TB> B_in(static_cast<size_t>(N) * K);
    std::vector<TC> C_out(static_cast<size_t>(M) * N, TC{0});
    std::vector<TC> C_ref(static_cast<size_t>(M) * N, TC{0});

    if constexpr (std::is_same_v<TA, float>)
      fill_random(A_in, -1.0f, 1.0f);
    else
      fill_random_bf16(A_in, -1.0f, 1.0f);

    if constexpr (std::is_same_v<TB, float>)
      fill_random(B_in, -1.0f, 1.0f);
    else
      fill_random_bf16(B_in, -1.0f, 1.0f);

    int lda = int(a_in_layout.stride().template get<0>());
    int ldk_a = int(a_in_layout.stride().template get<1>());
    int ldb = int(b_in_layout.stride().template get<0>());
    int ldk_b = int(b_in_layout.stride().template get<1>());
    int ldc = int(c_layout.stride().template get<0>());
    int ldc_n = int(c_layout.stride().template get<1>());

    ref_gemm(M, N, K,
             A_in.data(), lda, ldk_a,
             B_in.data(), ldb, ldk_b,
             C_ref.data(), ldc, ldc_n);

    auto a_packed_shape = make_shape(Mtiled, Ktiled, Int<kKt>{}, Int<kMt>{});
    auto a_packed_stride = make_stride(Ktiled * kKt * kMt,
                                       kKt * kMt,
                                       Int<Kernel::pack_A_M_on_dim2 ? kKt : kMt>{},
                                       Int<1>{});
    auto a_packed_layout = make_layout(a_packed_shape, a_packed_stride);

    auto b_packed_shape = make_shape(Ntiled, Ktiled, Int<kKt>{}, Int<kNt>{});
    constexpr int b_s2 = Kernel::pack_B_N_on_dim2 ? kKt : kNt;
    auto b_packed_stride = make_stride(Ktiled * kKt * kNt,
                                       kKt * kNt,
                                       Int<b_s2>{}, Int<1>{});
    auto b_packed_layout = make_layout(b_packed_shape, b_packed_stride);

    size_t a_packed_elems = static_cast<size_t>(Mtiled) * Ktiled * kMt * kKt;
    size_t b_packed_elems = static_cast<size_t>(Ntiled) * Ktiled * kNt * kKt;
    std::vector<TA> A_packed(a_packed_elems);
    std::vector<TB> B_packed(b_packed_elems);

    Kernel kernel;
    kernel.pack_A(A_in.data(), a_in_layout, A_packed.data(), a_packed_layout);
    kernel.pack_B(B_in.data(), b_in_layout, B_packed.data(), b_packed_layout);

    auto shape_mnk = make_shape(M, N, K);
    auto tiles = make_shape(kMt, kNt, kKt);
    Scheduler scheduler{};
    gemm(shape_mnk,
         A_packed.data(), a_packed_layout,
         B_packed.data(), b_packed_layout,
         C_out.data(), c_layout,
         tiles, kernel, scheduler, epilog);

    if (c_is_col_major) {
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          size_t off = static_cast<size_t>(m) + static_cast<size_t>(n) * static_cast<size_t>(M);
          EXPECT_NEAR(float(C_out[off]), float(C_ref[off]), tol)
              << "packed M=" << M << " N=" << N << " K=" << K
              << " at (" << m << "," << n << ")";
        }
    } else {
      size_t total = static_cast<size_t>(M) * N;
      for (size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), tol)
            << "packed M=" << M << " N=" << N << " K=" << K
            << " at index " << i;
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

TEST_F(GemmTest, AMXBf16_Tiling2D) {
  struct { int M,N,K, Mt,Nt; } t[] = {
    {335, 34, 605, 320, 32}, {512, 512, 512, 256, 256},
    {256, 32, 256, 128, 32}, {32, 32, 64, 32, 32},
    {16, 16, 16, 16, 16},   {64, 16, 64, 64, 16},
    {354, 445, 156, 336, 432}, {32, 29, 32, 32, 16},
    {416, 128, 40, 256, 128},
  };
  for (auto [M,N,K, Mt,Nt] : t) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        TILE_2D(Mt,Nt), identity_epilog, 1e-2f);
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

TEST_F(GemmTest, AMXBf16_Tiling3D) {
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
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, N,1,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        TILE_3D(Mt,Nt,Kt), identity_epilog, 1e-2f);
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

TEST_F(GemmTest, AMXBf16_BRowMajor) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, K,1, N,1,
        lt::A_row(M,K), lt::B_row(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-2f);
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

TEST_F(GemmTest, AMXBf16_AColMajor) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, 1,N, N,1,
        lt::A_col(M,K), lt::B_col(N,K), lt::C_row(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-2f);
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

TEST_F(GemmTest, AMXBf16_NonContigC) {
  for (auto [M,N,K] : g_small_sizes) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, K,1, 1,N, 1,M,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_col(M,N),
        BASIC_TILE(M,N), identity_epilog, 1e-2f,
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

TEST_F(GemmTest, AMXBf16_AllColMajor) {
  struct { int M,N,K, Mt,Nt,Kt; } t[] = {
    {64, 16, 64, 64, 16, 32},
    {32, 32, 64, 32, 32, 32},
    {65, 32, 64, 64, 32, 32},
    {256, 32, 256, 256, 32, 128},
    {512, 512, 512, 256, 256, 128},
  };
  for (auto [M,N,K, Mt,Nt,Kt] : t) {
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, K,1, 1,M,
        lt::A_col(M,K), lt::B_row(N,K), lt::C_col(M,N),
        TILE_3D(Mt,Nt,Kt), identity_epilog, 1e-2f,
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

TEST_F(GemmTest, AMXBf16_CT_ColMixed) {
  auto run = [this]<int M, int N, int K>() {
    constexpr int Mt = ((M + 15) / 16) * 16;
    constexpr int Nt = ((N + 15) / 16) * 16;
    run_gemm_test<ScalarAMXBf16Kernel, SchedulerMaxCases>(
        M, N, K, 1,M, K,1, 1,M,
        Layout(Shape<Int<M>,Int<K>>{}, Stride<Int<1>,Int<M>>{}),
        Layout(Shape<Int<N>,Int<K>>{}, Stride<Int<K>,Int<1>>{}),
        Layout(Shape<Int<M>,Int<N>>{}, Stride<Int<1>,Int<M>>{}),
        Shape<Int<Mt>,Int<Nt>,Int<32>>{},
        identity_epilog, 1e-2f,
        /*c_is_col_major=*/true);
  };
  run.template operator()<16, 32, 16>();
  run.template operator()<32, 32, 64>();
  run.template operator()<64, 16, 64>();
  run.template operator()<13, 127, 63>();
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
  run.template operator()<32, 29, 32>();
  run.template operator()<32, 32, 64>();
  run.template operator()<64, 16, 64>();
  run.template operator()<31, 62, 31>();
  run.template operator()<13, 32, 16>();
  run.template operator()<5, 45, 8>();
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
// Pack tests — pack_A / pack_B + gemm with pre-packed A/B
// ============================================================================

TEST_F(GemmTest, SMEFp32_PackGemm_Row) {
  for (auto [M,N,K] : g_all_sizes) {
    run_packed_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K,
        lt::A_row(M,K), lt::B_col(N,K), lt::C_row(M,N),
        identity_epilog, 1e-5f);
  }
}

TEST_F(GemmTest, SMEFp32_PackGemm_AMixed) {
  for (auto [M,N,K] : g_small_sizes) {
    run_packed_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K,
        lt::A_col(M,K), lt::B_col(N,K), lt::C_row(M,N),
        identity_epilog, 1e-5f);
  }
}

TEST_F(GemmTest, SMEFp32_PackGemm_BRow) {
  for (auto [M,N,K] : g_small_sizes) {
    run_packed_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K,
        lt::A_row(M,K), lt::B_row(N,K), lt::C_row(M,N),
        identity_epilog, 1e-5f);
  }
}

TEST_F(GemmTest, SMEFp32_PackGemm_AllCol) {
  for (auto [M,N,K] : g_small_sizes) {
    run_packed_gemm_test<ScalarSMEFp32Kernel, SchedulerMaxCases>(
        M, N, K,
        lt::A_col(M,K), lt::B_row(N,K), lt::C_col(M,N),
        identity_epilog, 1e-5f, /*c_is_col_major=*/true);
  }
}

TEST_F(GemmTest, SMEFp32_PackGemm_ReLU) {
  auto relu = [](int, int, float v) -> float { return std::max(0.0f, v); };
  for (auto [M,N,K] : g_quick_sizes) {
    using TA = float; using TB = float; using TC = float;
    constexpr int kMt = ScalarSMEFp32Kernel::Mtile;
    constexpr int kNt = ScalarSMEFp32Kernel::Ntile;
    constexpr int kKt = ScalarSMEFp32Kernel::Ktile;
    int M_pad = ((M+kMt-1)/kMt)*kMt, K_pad = ((K+kKt-1)/kKt)*kKt, N_pad = ((N+kNt-1)/kNt)*kNt;
    int Mtiled=M_pad/kMt, Ktiled=K_pad/kKt, Ntiled=N_pad/kNt;

    std::vector<TA> A_in(M*K); fill_random(A_in);
    std::vector<TB> B_in(N*K); fill_random(B_in);
    std::vector<TC> C_out(M*N, TC{0});
    std::vector<TC> C_ref(M*N, TC{0});

    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);
    for (auto& x : C_ref) x = std::max(TC{0}, x);

    constexpr int sme_a_s2 = ScalarSMEFp32Kernel::pack_A_M_on_dim2 ? kKt : kMt;
    auto a_ps = make_shape(Mtiled,Ktiled,Int<kKt>{},Int<kMt>{});
    auto a_pstr = make_stride(Ktiled*kKt*kMt, kKt*kMt, Int<sme_a_s2>{},Int<1>{});
    auto a_pl = make_layout(a_ps, a_pstr);
    auto b_ps = make_shape(Ntiled,Ktiled,Int<kKt>{},Int<kNt>{});
    constexpr int sme_b_s2 = ScalarSMEFp32Kernel::pack_B_N_on_dim2 ? kKt : kNt;
    auto b_pstr = make_stride(Ktiled*kKt*kNt, kKt*kNt, Int<sme_b_s2>{},Int<1>{});
    auto b_pl = make_layout(b_ps, b_pstr);

    std::vector<TA> A_p(Mtiled*Ktiled*kMt*kKt);
    std::vector<TB> B_p(Ntiled*Ktiled*kNt*kKt);
    ScalarSMEFp32Kernel kernel;
    kernel.pack_A(A_in.data(), lt::A_row(M,K), A_p.data(), a_pl);
    kernel.pack_B(B_in.data(), lt::B_col(N,K), B_p.data(), b_pl);
    gemm(make_shape(M,N,K), A_p.data(), a_pl, B_p.data(), b_pl, C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, relu);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-5f)
          << "packed+relu M=" << M << " N=" << N << " K=" << K;
  }
}

TEST_F(GemmTest, AMXBf16_PackGemm_Row) {
  // AMX B packed layout uses Wd interleaving incompatible with simple 2D
  // kernel stride — only pack A here, keep B in runtime layout
  for (auto [M,N,K] : g_small_sizes) {
    using TA = bfloat16_t; using TB = bfloat16_t; using TC = float;
    constexpr int kMt = ScalarAMXBf16Kernel::Mtile;
    constexpr int kNt = ScalarAMXBf16Kernel::Ntile;
    constexpr int kKt = ScalarAMXBf16Kernel::Ktile;
    int M_pad = ((M+kMt-1)/kMt)*kMt, K_pad = ((K+kKt-1)/kKt)*kKt;
    int Mtiled=M_pad/kMt, Ktiled=K_pad/kKt;

    std::vector<TA> A_in(M*K); fill_random_bf16(A_in);
    std::vector<TB> B_in(N*K); fill_random_bf16(B_in);
    std::vector<TC> C_out(M*N, TC{0});
    std::vector<TC> C_ref(M*N, TC{0});

    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);

    auto a_ps = make_shape(Mtiled,Ktiled,Int<kKt>{},Int<kMt>{});
    constexpr int amx_a_s2 = ScalarAMXBf16Kernel::pack_A_M_on_dim2 ? kKt : kMt;
    auto a_pstr = make_stride(Ktiled*kKt*kMt, kKt*kMt, Int<amx_a_s2>{},Int<1>{});
    auto a_pl = make_layout(a_ps, a_pstr);
    std::vector<TA> A_p(Mtiled*Ktiled*kMt*kKt);
    ScalarAMXBf16Kernel kernel;
    kernel.pack_A(A_in.data(), lt::A_row(M,K), A_p.data(), a_pl);

    gemm(make_shape(M,N,K), A_p.data(), a_pl,
         B_in.data(), lt::B_col(N,K),
         C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, identity_epilog);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-2f)
          << "AMX packA M=" << M << " N=" << N << " K=" << K;
  }
}

// ============================================================================
// Pack tests — single-dimension packing coverage for SME and AMX
// ============================================================================

TEST_F(GemmTest, SMEFp32_PackGemm_AOnly) {
  for (auto [M,N,K] : g_all_sizes) {
    using TA = float; using TB = float; using TC = float;
    constexpr int kMt = ScalarSMEFp32Kernel::Mtile;
    constexpr int kNt = ScalarSMEFp32Kernel::Ntile;
    constexpr int kKt = ScalarSMEFp32Kernel::Ktile;
    int M_pad = ((M+kMt-1)/kMt)*kMt, K_pad = ((K+kKt-1)/kKt)*kKt;
    int Mtiled=M_pad/kMt, Ktiled=K_pad/kKt;

    std::vector<TA> A_in(M*K); fill_random(A_in);
    std::vector<TB> B_in(N*K); fill_random(B_in);
    std::vector<TC> C_out(M*N, TC{0}), C_ref(M*N, TC{0});
    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);

    constexpr int a_s2 = ScalarSMEFp32Kernel::pack_A_M_on_dim2 ? kKt : kMt;
    auto a_ps = make_shape(Mtiled,Ktiled,Int<kKt>{},Int<kMt>{});
    auto a_pstr = make_stride(Ktiled*kKt*kMt, kKt*kMt, Int<a_s2>{},Int<1>{});
    auto a_pl = make_layout(a_ps, a_pstr);
    std::vector<TA> A_p(Mtiled*Ktiled*kMt*kKt);
    ScalarSMEFp32Kernel kernel;
    kernel.pack_A(A_in.data(), lt::A_row(M,K), A_p.data(), a_pl);
    gemm(make_shape(M,N,K), A_p.data(), a_pl,
         B_in.data(), lt::B_col(N,K), C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, identity_epilog);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-5f)
          << "SME Aonly M=" << M << " N=" << N << " K=" << K;
  }
}

TEST_F(GemmTest, SMEFp32_PackGemm_BOnly) {
  for (auto [M,N,K] : g_small_sizes) {
    using TA = float; using TB = float; using TC = float;
    constexpr int kMt = ScalarSMEFp32Kernel::Mtile;
    constexpr int kNt = ScalarSMEFp32Kernel::Ntile;
    constexpr int kKt = ScalarSMEFp32Kernel::Ktile;
    int N_pad = ((N+kNt-1)/kNt)*kNt, K_pad = ((K+kKt-1)/kKt)*kKt;
    int Ntiled=N_pad/kNt, Ktiled=K_pad/kKt;

    std::vector<TA> A_in(M*K); fill_random(A_in);
    std::vector<TB> B_in(N*K); fill_random(B_in);
    std::vector<TC> C_out(M*N, TC{0}), C_ref(M*N, TC{0});
    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);

    constexpr int b_s2 = ScalarSMEFp32Kernel::pack_B_N_on_dim2 ? kKt : kNt;
    auto b_ps = make_shape(Ntiled,Ktiled,Int<kKt>{},Int<kNt>{});
    auto b_pstr = make_stride(Ktiled*kKt*kNt, kKt*kNt, Int<b_s2>{},Int<1>{});
    auto b_pl = make_layout(b_ps, b_pstr);
    std::vector<TB> B_p(Ntiled*Ktiled*kNt*kKt);
    ScalarSMEFp32Kernel kernel;
    kernel.pack_B(B_in.data(), lt::B_col(N,K), B_p.data(), b_pl);
    gemm(make_shape(M,N,K), A_in.data(), lt::A_row(M,K),
         B_p.data(), b_pl, C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, identity_epilog);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-5f)
          << "SME Bonly M=" << M << " N=" << N << " K=" << K;
  }
}

TEST_F(GemmTest, AMXBf16_PackGemm_BOnly) {
  for (auto [M,N,K] : g_small_sizes) {
    using TA = bfloat16_t; using TB = bfloat16_t; using TC = float;
    constexpr int kMt = ScalarAMXBf16Kernel::Mtile;
    constexpr int kNt = ScalarAMXBf16Kernel::Ntile;
    constexpr int kKt = ScalarAMXBf16Kernel::Ktile;
    constexpr int kWd = ScalarAMXBf16Kernel::Wd;
    int N_pad = ((N+kNt-1)/kNt)*kNt, K_pad = ((K+kKt-1)/kKt)*kKt;
    int Ntiled=N_pad/kNt, Ktiled=K_pad/kKt;

    std::vector<TA> A_in(M*K); fill_random_bf16(A_in);
    std::vector<TB> B_in(N*K); fill_random_bf16(B_in);
    std::vector<TC> C_out(M*N, TC{0}), C_ref(M*N, TC{0});
    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);

    // AMX B VNNI: A_block_dim2=kKt/kWd, A_block_dim3=kNt*kWd
    constexpr int b_d2 = kKt / kWd, b_d3 = kNt * kWd;
    constexpr int b_s2 = b_d3;  // stride dim2 = Ntile*Wd
    auto b_ps = make_shape(Ntiled,Ktiled,Int<b_d2>{},Int<b_d3>{});
    auto b_pstr = make_stride(Ktiled * b_d2 * b_d3, b_d2 * b_d3, Int<b_s2>{},Int<1>{});
    auto b_pl = make_layout(b_ps, b_pstr);
    std::vector<TB> B_p(Ntiled*Ktiled*b_d2*b_d3);
    ScalarAMXBf16Kernel kernel;
    kernel.pack_B(B_in.data(), lt::B_col(N,K), B_p.data(), b_pl);
    gemm(make_shape(M,N,K), A_in.data(), lt::A_row(M,K),
         B_p.data(), b_pl, C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, identity_epilog);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-2f)
          << "AMX Bonly M=" << M << " N=" << N << " K=" << K;
  }
}

TEST_F(GemmTest, AMXBf16_PackGemm_Both) {
  for (auto [M,N,K] : g_small_sizes) {
    using TA = bfloat16_t; using TB = bfloat16_t; using TC = float;
    constexpr int kMt = ScalarAMXBf16Kernel::Mtile;
    constexpr int kNt = ScalarAMXBf16Kernel::Ntile;
    constexpr int kKt = ScalarAMXBf16Kernel::Ktile;
    constexpr int kWd = ScalarAMXBf16Kernel::Wd;
    int M_pad = ((M+kMt-1)/kMt)*kMt, N_pad = ((N+kNt-1)/kNt)*kNt, K_pad = ((K+kKt-1)/kKt)*kKt;
    int Mtiled=M_pad/kMt, Ntiled=N_pad/kNt, Ktiled=K_pad/kKt;

    std::vector<TA> A_in(M*K); fill_random_bf16(A_in);
    std::vector<TB> B_in(N*K); fill_random_bf16(B_in);
    std::vector<TC> C_out(M*N, TC{0}), C_ref(M*N, TC{0});
    ref_gemm(M,N,K, A_in.data(),K,1, B_in.data(),1,N, C_ref.data(),N,1);

    constexpr int a_s2 = ScalarAMXBf16Kernel::pack_A_M_on_dim2 ? kKt : kMt;
    auto a_ps = make_shape(Mtiled,Ktiled,Int<kKt>{},Int<kMt>{});
    auto a_pstr = make_stride(Ktiled*kKt*kMt, kKt*kMt, Int<a_s2>{},Int<1>{});
    auto a_pl = make_layout(a_ps, a_pstr);

    constexpr int b_d2 = kKt / kWd, b_d3 = kNt * kWd;
    constexpr int b_s2 = b_d3;
    auto b_ps = make_shape(Ntiled,Ktiled,Int<b_d2>{},Int<b_d3>{});
    auto b_pstr = make_stride(Ktiled * b_d2 * b_d3, b_d2 * b_d3, Int<b_s2>{},Int<1>{});
    auto b_pl = make_layout(b_ps, b_pstr);

    std::vector<TA> A_p(Mtiled*Ktiled*kMt*kKt);
    std::vector<TB> B_p(Ntiled*Ktiled*b_d2*b_d3);
    ScalarAMXBf16Kernel kernel;
    kernel.pack_A(A_in.data(), lt::A_row(M,K), A_p.data(), a_pl);
    kernel.pack_B(B_in.data(), lt::B_col(N,K), B_p.data(), b_pl);
    gemm(make_shape(M,N,K), A_p.data(), a_pl, B_p.data(), b_pl,
         C_out.data(), lt::C_row(M,N),
         make_shape(kMt,kNt,kKt), kernel, SchedulerMaxCases{}, identity_epilog);
    size_t total = size_t(M)*N;
    for (size_t i = 0; i < total; ++i)
      EXPECT_NEAR(float(C_out[i]), float(C_ref[i]), 1e-2f)
          << "AMX both M=" << M << " N=" << N << " K=" << K;
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
