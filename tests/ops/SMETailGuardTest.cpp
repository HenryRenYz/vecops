#include <gtest/gtest.h>

#include <cstddef>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "vecops/ops/Attention.h"

namespace {

using namespace vecops;
using BF16 = vecops::bfloat16_t;

TEST(SMETailGuardTest, MaterializedSDPADoesNotReadPastBF16ValueTail) {
  constexpr nint_t Lq = 32;
  constexpr nint_t Lkv = 64;
  constexpr nint_t D = 24;
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  constexpr std::size_t value_bytes = Lkv * D * sizeof(BF16);
  ASSERT_LT(value_bytes, static_cast<std::size_t>(page_size));
  void* mapping = mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  auto* v = reinterpret_cast<BF16*>(
      static_cast<char*>(mapping) + page_size - value_bytes);
  for (nint_t i = 0; i < Lkv * D; ++i)
    v[i] = BF16(float(i % 13 - 6) / 16);
  ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page_size,
                     page_size, PROT_NONE), 0);

  std::vector<BF16> q(Lq * D, BF16(0.125f));
  std::vector<BF16> k(Lkv * D, BF16(0.25f));
  std::vector<float> out(Lq * D, -99.0f);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto vt = tensor::make_tensor(v, tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  using Config = ops::AttentionConfig<
      ops::MatmulConfig<matmul::SME_BF16F32>, 32, 32,
      ops::AttentionCausalMode::none, ops::SDPAStrategy::materialized>;
  auto op = ops::scaled_dot_product_attention(Config{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, ot, 0.125f);
  for (nint_t row = 0; row < Lq; ++row) {
    for (nint_t col = 0; col < D; ++col) {
      float expected = 0;
      for (nint_t kv = 0; kv < Lkv; ++kv)
        expected += static_cast<float>(v[kv * D + col]) / Lkv;
      EXPECT_NEAR(out[row * D + col], expected, 2.5e-2f);
    }
  }
  EXPECT_EQ(munmap(mapping, 2 * page_size), 0);
}

}  // namespace
