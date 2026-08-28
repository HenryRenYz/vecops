#include <gtest/gtest.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <vector>

#include "vecops/ops/Matmul.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

bool enable_amx() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}

TEST(MatmulBatchAMXTest, TraversesBatchAndExposesTilePolicy) {
  ASSERT_TRUE(enable_amx());
  using Atom = gemm::AMX_BF16F32;
  using Policy = kernel::loop::tile2d_policy::Natural;
  constexpr nint_t Batch = 3;
  constexpr nint_t M = 19;
  constexpr nint_t N = 21;
  constexpr nint_t K = 35;
  std::vector<bfloat16_t> a(Batch * M * K);
  std::vector<bfloat16_t> b(Batch * N * K);
  std::vector<float32_t> c(Batch * M * N);
  for (nint_t i = 0; i < Batch * M * K; ++i)
    a[i] = bfloat16_t(static_cast<float>(i % 13 - 6) / 13.0f);
  for (nint_t i = 0; i < Batch * N * K; ++i)
    b[i] = bfloat16_t(static_cast<float>(i % 11 - 5) / 11.0f);

  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));
  auto operation = ops::make_matmul<Atom, Policy>(M, N, K, at, bt, ct);
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t batch = 0; batch < Batch; ++batch) {
    for (nint_t m = 0; m < M; ++m) {
      for (nint_t n = 0; n < N; ++n) {
        float expected = 0;
        for (nint_t k = 0; k < K; ++k) {
          expected += static_cast<float>(a[(batch * M + m) * K + k]) *
              static_cast<float>(b[(batch * N + n) * K + k]);
        }
        EXPECT_NEAR(c[(batch * M + m) * N + n], expected, 2.0e-4f)
            << "batch=" << batch << " m=" << m << " n=" << n;
      }
    }
  }
}

} // namespace
