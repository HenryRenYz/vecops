#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "vecops/platform/Features.h"
#include "vecops/ops/Attention.h"

#include "MatmulTestArch.h"

namespace {

using namespace vecops;
using bfloat16_t = vecops::bfloat16_t;
using float16_t = vecops::float16_t;
using float32_t = vecops::float32_t;

#if defined(ARCH_X86_FAMILY)
using TestAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using TestAtom = matmul::SME_BF16F32;
#endif

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)

using Config = ops::AttentionConfig<ops::MatmulConfig<TestAtom>, 4, 4>;
using StrictConfig = ops::AttentionConfig<
    ops::MatmulConfig<TestAtom>, 4, 4,
    ops::AttentionCausalMode::none, ops::AttentionStrategy::automatic,
    vec::Accuracy::Strict>;
using ModelConfig = ops::AttentionConfig<ops::MatmulConfig<TestAtom>, 32, 32>;
using StreamingModelConfig = ops::AttentionConfig<
    ops::MatmulConfig<TestAtom>, 32, 32,
    ops::AttentionCausalMode::none, ops::AttentionStrategy::streaming>;
using MaterializedModelConfig = ops::AttentionConfig<
    ops::MatmulConfig<TestAtom>, 32, 32,
    ops::AttentionCausalMode::none, ops::AttentionStrategy::materialized>;
using CausalConfig = ops::AttentionConfig<
    ops::MatmulConfig<TestAtom>, 4, 4,
    ops::AttentionCausalMode::bottom_right>;
using TopLeftCausalConfig = ops::AttentionConfig<
    ops::MatmulConfig<TestAtom>, 4, 4,
    ops::AttentionCausalMode::top_left>;

using DynamicMapTensor = decltype(tensor::make_tensor(
    static_cast<int32_t*>(nullptr),
    tensor::make_shape(meta::cint<2>, meta::cint<3>)));
using ConstDynamicMapTensor = decltype(tensor::make_tensor(
    static_cast<const int32_t*>(nullptr),
    tensor::make_shape(meta::cint<2>, meta::cint<3>)));
using DynamicMapSpec = decltype(tensor::input<int32_t>(
    std::declval<DynamicMapTensor>()));
static_assert(ops::attention_details::AttentionBufferTensor<
              DynamicMapTensor, int32_t, 2>);
static_assert(ops::attention_details::WritableAttentionBufferTensor<
              DynamicMapTensor, int32_t, 2>);
static_assert(!ops::attention_details::WritableAttentionBufferTensor<
              ConstDynamicMapTensor, int32_t, 2>);
static_assert(!ops::attention_details::AttentionBufferTensor<
              DynamicMapSpec, int32_t, 2>);

template <typename TQK, typename TV>
std::vector<float> reference_attention(
    const std::vector<TQK>& q, const std::vector<TQK>& k,
    const std::vector<TV>& v, nint_t lq, nint_t lkv,
    nint_t dqk, nint_t dv, float scale,
    const std::vector<bool>* query_mask = nullptr,
    const std::vector<bool>* key_mask = nullptr,
    const std::vector<bool>* attention_mask = nullptr,
    const std::vector<float>* bias = nullptr) {
  std::vector<float> out(static_cast<std::size_t>(lq * dv));
  std::vector<float> scores(static_cast<std::size_t>(lkv));
  for (nint_t i = 0; i < lq; ++i) {
    float maximum = -std::numeric_limits<float>::infinity();
    bool valid = false;
    for (nint_t j = 0; j < lkv; ++j) {
      bool keep = (!query_mask || (*query_mask)[static_cast<std::size_t>(i)]) &&
          (!key_mask || (*key_mask)[static_cast<std::size_t>(j)]) &&
          (!attention_mask ||
           (*attention_mask)[static_cast<std::size_t>(i * lkv + j)]);
      if (!keep) {
        scores[static_cast<std::size_t>(j)] =
            -std::numeric_limits<float>::infinity();
        continue;
      }
      float dot = 0.0f;
      for (nint_t d = 0; d < dqk; ++d) {
        dot += static_cast<float>(q[static_cast<std::size_t>(i * dqk + d)]) *
            static_cast<float>(k[static_cast<std::size_t>(j * dqk + d)]);
      }
      float score = dot * scale;
      if (bias) score += (*bias)[static_cast<std::size_t>(i * lkv + j)];
      scores[static_cast<std::size_t>(j)] = score;
      maximum = std::max(maximum, score);
      valid = true;
    }
    if (!valid) continue;
    float sum = 0.0f;
    for (nint_t j = 0; j < lkv; ++j) {
      float& score = scores[static_cast<std::size_t>(j)];
      score = std::exp(score - maximum);
      sum += score;
    }
    for (nint_t j = 0; j < lkv; ++j) {
      const float probability = scores[static_cast<std::size_t>(j)] / sum;
      for (nint_t d = 0; d < dv; ++d) {
        out[static_cast<std::size_t>(i * dv + d)] += probability *
            static_cast<float>(v[static_cast<std::size_t>(j * dv + d)]);
      }
    }
  }
  return out;
}

template <typename QK, typename Value, typename Output,
          typename LqExtent, typename LkvExtent>
void run_dense_dtype_meta_case(
    LqExtent lq_extent, LkvExtent lkv_extent,
    nint_t dqk, nint_t dv) {
  const nint_t lq = static_cast<nint_t>(lq_extent);
  const nint_t lkv = static_cast<nint_t>(lkv_extent);
  std::vector<QK> q(static_cast<std::size_t>(lq * dqk));
  std::vector<QK> k(static_cast<std::size_t>(lkv * dqk));
  std::vector<Value> v(static_cast<std::size_t>(lkv * dv));
  std::vector<Output> out(static_cast<std::size_t>(lq * dv));
  for (nint_t i = 0; i < lq * dqk; ++i)
    q[static_cast<std::size_t>(i)] =
        static_cast<QK>(float((i * 3) % 17 - 8) / 32);
  for (nint_t i = 0; i < lkv * dqk; ++i)
    k[static_cast<std::size_t>(i)] =
        static_cast<QK>(float((i * 5) % 19 - 9) / 32);
  for (nint_t i = 0; i < lkv * dv; ++i)
    v[static_cast<std::size_t>(i)] =
        static_cast<Value>(float((i * 7) % 23 - 11) / 32);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      lq_extent, meta::Any{dqk}));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      lkv_extent, meta::Any{dqk}));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      lkv_extent, meta::Any{dv}));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      lq_extent, meta::Any{dv}));
  auto op = ops::dense_attention(ModelConfig{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, ot, 0.125f);
  const auto expected = reference_attention(
      q, k, v, lq, lkv, dqk, dv, 0.125f);
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(static_cast<float>(out[i]), expected[i], 5.0e-2f)
        << "index=" << i << " shape=" << lq << 'x' << lkv
        << " dqk=" << dqk << " dv=" << dv;
  }
}

TEST(AttentionTest, DenseUnmaskedHandlesQueryTailAndDistinctValueWidth) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 5;
  constexpr nint_t Lkv = 7;
  constexpr nint_t Dqk = 8;
  constexpr nint_t Dv = 6;
  std::vector<bfloat16_t> q(Lq * Dqk);
  std::vector<bfloat16_t> k(Lkv * Dqk);
  std::vector<bfloat16_t> v(Lkv * Dv);
  std::vector<float> out(Lq * Dv, -99.0f);
  for (nint_t i = 0; i < Lq * Dqk; ++i)
    q[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 3) % 11 - 5) / 8);
  for (nint_t i = 0; i < Lkv * Dqk; ++i)
    k[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 5) % 13 - 6) / 8);
  for (nint_t i = 0; i < Lkv * Dv; ++i)
    v[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 7) % 17 - 8) / 16);

  auto qt = tensor::make_tensor(
      q.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{Dqk}));
  auto kt = tensor::make_tensor(
      k.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{Dqk}));
  auto vt = tensor::make_tensor(
      v.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{Dv}));
  auto ot = tensor::make_tensor(
      out.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{Dv}));
  const float scale = 0.375f;
  auto op = ops::dense_attention(StrictConfig{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, ot, scale);

  const auto expected = reference_attention(
      q, k, v, Lq, Lkv, Dqk, Dv, scale);
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 2.5e-2f) << "index=" << i;
  }
}

TEST(AttentionTest, DenseCompileTimeOptionalMasksBiasAndAllMaskedRow) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 3;
  constexpr nint_t Lkv = 5;
  constexpr nint_t D = 8;
  std::vector<bfloat16_t> q(Lq * D, bfloat16_t(0.25f));
  std::vector<bfloat16_t> k(Lkv * D, bfloat16_t(0.5f));
  std::vector<bfloat16_t> v(Lkv * D);
  std::array<bool, Lq> qm{true, false, true};
  std::array<bool, Lkv> km{true, true, false, true, true};
  std::array<bool, Lq * Lkv> am{};
  am.fill(true);
  std::vector<float> bias(Lq * Lkv);
  std::vector<float> out(Lq * D, -99.0f);
  for (nint_t i = 0; i < Lkv * D; ++i)
    v[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 9 - 4) / 8);
  am[static_cast<std::size_t>(2 * Lkv + 3)] = 0;
  for (nint_t i = 0; i < Lq * Lkv; ++i)
    bias[static_cast<std::size_t>(i)] = float(i % Lkv) * 0.125f;

  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto qmt = tensor::make_tensor(qm.data(), tensor::make_shape(meta::cint<Lq>));
  auto kmt = tensor::make_tensor(km.data(), tensor::make_shape(meta::cint<Lkv>));
  auto amt = tensor::make_tensor(am.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<Lkv>));
  auto bt = tensor::make_tensor(bias.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<Lkv>));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));

  auto op = ops::dense_attention(Config{});
  kernel::Workspace storage(op.required_workspace(
      qt, kt, vt, qmt, kmt, amt, bt, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, qmt, kmt, amt, bt, ot, 0.5f);

  std::vector<bool> ref_qm{true, false, true};
  std::vector<bool> ref_km{true, true, false, true, true};
  std::vector<bool> ref_am(Lq * Lkv, true);
  ref_am[static_cast<std::size_t>(2 * Lkv + 3)] = false;
  const auto expected = reference_attention(
      q, k, v, Lq, Lkv, D, D, 0.5f,
      &ref_qm, &ref_km, &ref_am, &bias);
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 2.5e-2f) << "index=" << i;
  }
  for (nint_t d = 0; d < D; ++d)
    EXPECT_EQ(out[static_cast<std::size_t>(D + d)], 0.0f);

  std::array<int32_t, 2> all_blocks{0, 1};
  std::vector<float> sparse_out(Lq * D, -99.0f);
  auto it = tensor::make_tensor(
      all_blocks.data(), tensor::make_shape(meta::cint<1>, meta::cint<2>));
  auto sot = tensor::make_tensor(sparse_out.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto sparse = ops::sparse_flash_attention(Config{});
  kernel::Workspace sparse_storage(sparse.required_workspace(
      qt, kt, vt, it, qmt, kmt, amt, bt, sot));
  auto sparse_workspace = sparse_storage.view();
  sparse(
      sparse_workspace, qt, kt, vt, it,
      qmt, kmt, amt, bt, sot, 0.5f);
  for (std::size_t i = 0; i < sparse_out.size(); ++i)
    EXPECT_NEAR(sparse_out[i], expected[i], 2.5e-2f) << "index=" << i;
}

TEST(AttentionTest, SparseAttentionSupportsDifferentBlocksAndTailBlocks) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 5;
  constexpr nint_t Lkv = 9;
  constexpr nint_t Dqk = 8;
  constexpr nint_t Dv = 6;
  constexpr nint_t S = 3;
  std::vector<bfloat16_t> q(Lq * Dqk);
  std::vector<bfloat16_t> k(Lkv * Dqk);
  std::vector<bfloat16_t> v(Lkv * Dv);
  std::vector<int32_t> index{0, 2, -1, 1, 2, -1};
  std::vector<float> out(Lq * Dv, -99.0f);
  for (nint_t i = 0; i < Lq * Dqk; ++i)
    q[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 3) % 13 - 6) / 8);
  for (nint_t i = 0; i < Lkv * Dqk; ++i)
    k[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 5) % 17 - 8) / 8);
  for (nint_t i = 0; i < Lkv * Dv; ++i)
    v[static_cast<std::size_t>(i)] = bfloat16_t(float((i * 7) % 19 - 9) / 16);

  auto qt = tensor::make_tensor(
      q.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{Dqk}));
  auto kt = tensor::make_tensor(
      k.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{Dqk}));
  auto vt = tensor::make_tensor(
      v.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{Dv}));
  auto it = tensor::make_tensor(
      index.data(), tensor::make_shape(meta::Any{2}, meta::Any{S}));
  auto ot = tensor::make_tensor(
      out.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{Dv}));
  auto op = ops::sparse_attention(Config{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, it, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, it, ot, 0.25f);

  std::vector<bool> selected(Lq * Lkv, false);
  for (nint_t qi = 0; qi < Lq; ++qi) {
    const bool first_query_block = qi < 4;
    for (nint_t kj = 0; kj < Lkv; ++kj) {
      const nint_t block = kj / 4;
      selected[static_cast<std::size_t>(qi * Lkv + kj)] =
          first_query_block ? (block == 0 || block == 2)
                            : (block == 1 || block == 2);
    }
  }
  const auto expected = reference_attention(
      q, k, v, Lq, Lkv, Dqk, Dv, 0.25f,
      nullptr, nullptr, &selected, nullptr);
  for (std::size_t i = 0; i < out.size(); ++i)
    EXPECT_NEAR(out[i], expected[i], 3.0e-2f) << "index=" << i;
}

TEST(AttentionTest, MaterializedSparsePreservesDuplicateBlockMultiplicity) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 4;
  constexpr nint_t Lkv = 8;
  constexpr nint_t D = 8;
  constexpr nint_t PackedLkv = 12;
  std::vector<bfloat16_t> q(Lq * D);
  std::vector<bfloat16_t> k(Lkv * D);
  std::vector<bfloat16_t> v(Lkv * D);
  std::vector<int32_t> index{1, 0, 1};
  std::vector<float> out(Lq * D, -99.0f);
  for (nint_t i = 0; i < Lq * D; ++i)
    q[static_cast<std::size_t>(i)] =
        bfloat16_t(float((i * 3) % 11 - 5) / 8);
  for (nint_t i = 0; i < Lkv * D; ++i) {
    k[static_cast<std::size_t>(i)] =
        bfloat16_t(float((i * 5) % 17 - 8) / 8);
    v[static_cast<std::size_t>(i)] =
        bfloat16_t(float((i * 7) % 19 - 9) / 16);
  }

  auto qt = tensor::make_tensor(
      q.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{D}));
  auto kt = tensor::make_tensor(
      k.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{D}));
  auto vt = tensor::make_tensor(
      v.data(), tensor::make_shape(meta::Any{Lkv}, meta::Any{D}));
  auto it = tensor::make_tensor(
      index.data(), tensor::make_shape(meta::cint<1>, meta::cint<3>));
  auto ot = tensor::make_tensor(
      out.data(), tensor::make_shape(meta::Any{Lq}, meta::Any{D}));
  auto op = ops::sparse_attention(Config{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, it, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, it, ot, 0.25f);

  std::vector<bfloat16_t> packed_k(PackedLkv * D);
  std::vector<bfloat16_t> packed_v(PackedLkv * D);
  constexpr std::array<nint_t, 3> blocks{1, 0, 1};
  for (nint_t packed_block = 0; packed_block < 3; ++packed_block) {
    const nint_t source = blocks[static_cast<std::size_t>(packed_block)] * 4;
    std::copy_n(
        k.data() + source * D, 4 * D,
        packed_k.data() + packed_block * 4 * D);
    std::copy_n(
        v.data() + source * D, 4 * D,
        packed_v.data() + packed_block * 4 * D);
  }
  const auto expected = reference_attention(
      q, packed_k, packed_v, Lq, PackedLkv, D, D, 0.25f,
      nullptr, nullptr, nullptr, nullptr);
  for (std::size_t i = 0; i < out.size(); ++i)
    EXPECT_NEAR(out[i], expected[i], 3.0e-2f) << "index=" << i;
}

TEST(AttentionTest, DynamicMaskRanksTailBlocksAndUsesStableTieBreak) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 5;
  constexpr nint_t Lkv = 9;
  constexpr nint_t D = 8;
  constexpr nint_t Tq = 2;
  constexpr nint_t S = 3;
  std::vector<bfloat16_t> q(Lq * D, bfloat16_t(1.0f));
  std::vector<bfloat16_t> k(Lkv * D);
  for (nint_t row = 0; row < Lkv; ++row) {
    const float value = float(row / 4);
    for (nint_t d = 0; d < D; ++d)
      k[static_cast<std::size_t>(row * D + d)] = bfloat16_t(value);
  }
  std::vector<int32_t> index(Tq * S, -7);
  std::vector<float> weight(Tq * S, -7.0f);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::dyn<1, 1, 64>(Lq), meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::dyn<1, 1, 64>(Lkv), meta::cint<D>));
  auto it = tensor::make_tensor(index.data(), tensor::make_shape(
      meta::cint<Tq>, meta::cint<S>));
  auto wt = tensor::make_tensor(weight.data(), tensor::make_shape(
      meta::cint<Tq>, meta::cint<S>));
  Config config{};
  config.static_probability = 1.0f;
  config.random_probability = 1.0f;
  auto op = ops::dynamic_attention_mask(config);
  kernel::Workspace storage(op.required_workspace(qt, kt, it, wt));
  auto workspace = storage.view();
  op(workspace, qt, kt, it, wt, 1.0f);

  for (nint_t row = 0; row < Tq; ++row) {
    EXPECT_EQ(index[static_cast<std::size_t>(row * S)], 2);
    EXPECT_EQ(index[static_cast<std::size_t>(row * S + 1)], 1);
    EXPECT_EQ(index[static_cast<std::size_t>(row * S + 2)], 0);
    EXPECT_EQ(weight[static_cast<std::size_t>(row * S)], -1.0f);
    EXPECT_EQ(weight[static_cast<std::size_t>(row * S + 1)], -1.0f);
    EXPECT_EQ(weight[static_cast<std::size_t>(row * S + 2)], -1.0f);
  }
}

#ifdef VECOPS_DEBUG
TEST(AttentionTest, DynamicMapBuffersRejectNonContiguousTensor) {
  std::array<int32_t, 8> storage{};
  auto map = tensor::make_tensor(
      storage.data(), tensor::make_shape(meta::cint<2>, meta::cint<3>),
      tensor::make_strides(meta::cint<4>, meta::cint<1>));
  ASSERT_FALSE(map.is_contiguous());
  EXPECT_DEATH(
      (ops::attention_details::validate_contiguous_buffer<int32_t, 2>(
          map, "dynamic map must be contiguous")),
      "contiguous");
}
#endif

TEST(AttentionTest, SampledSparseIsReproducibleForFixedRngSeed) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 4;
  constexpr nint_t Lkv = 12;
  constexpr nint_t D = 8;
  std::vector<bfloat16_t> q(Lq * D, bfloat16_t(0.25f));
  std::vector<bfloat16_t> k(Lkv * D);
  std::vector<bfloat16_t> v(Lkv * D);
  for (nint_t i = 0; i < Lkv * D; ++i) {
    k[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 7 - 3) / 8);
    v[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 11 - 5) / 8);
  }
  std::vector<int32_t> index{0, 1, 2};
  std::vector<float> weight{-1.0f, 0.3f, 0.2f};
  std::vector<float> out_a(Lq * D);
  std::vector<float> out_b(Lq * D);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto it = tensor::make_tensor(index.data(), tensor::make_shape(
      meta::cint<1>, meta::cint<3>));
  auto wt = tensor::make_tensor(weight.data(), tensor::make_shape(
      meta::cint<1>, meta::cint<3>));
  auto out_at = tensor::make_tensor(out_a.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto out_bt = tensor::make_tensor(out_b.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  Config config{};
  config.random_blocks = 1;
  config.static_probability = 0.5f;
  config.random_probability = 1.0f;
  auto op = ops::sparse_flash_attention(config);
  const nint_t bytes = op.required_workspace(
      qt, kt, vt, it, wt, tensor::nullopt, tensor::nullopt,
      tensor::nullopt, tensor::nullopt, out_at);
  kernel::Workspace storage_a(bytes);
  kernel::Workspace storage_b(bytes);
  auto workspace_a = storage_a.view();
  auto workspace_b = storage_b.view();
  std::mt19937 rng_a(17);
  std::mt19937 rng_b(17);
  op(workspace_a, qt, kt, vt, it, wt,
     tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt,
     out_at, 0.5f, rng_a);
  op(workspace_b, qt, kt, vt, it, wt,
     tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt,
     out_bt, 0.5f, rng_b);
  EXPECT_EQ(out_a, out_b);
  EXPECT_TRUE(std::any_of(out_a.begin(), out_a.end(), [](float value) {
    return value != 0.0f;
  }));
}

TEST(AttentionTest, DynamicCompositionMatchesDenseWhenAllBlocksSelected) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 5;
  constexpr nint_t Lkv = 9;
  constexpr nint_t D = 8;
  std::vector<bfloat16_t> q(Lq * D);
  std::vector<bfloat16_t> k(Lkv * D);
  std::vector<bfloat16_t> v(Lkv * D);
  std::vector<float> dense_out(Lq * D);
  std::vector<float> dynamic_out(Lq * D);
  for (nint_t i = 0; i < Lq * D; ++i)
    q[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 7 - 3) / 8);
  for (nint_t i = 0; i < Lkv * D; ++i) {
    k[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 11 - 5) / 8);
    v[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 13 - 6) / 8);
  }
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::Any{Lq}, meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::Any{Lkv}, meta::cint<D>));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::Any{Lkv}, meta::cint<D>));
  auto dense_ot = tensor::make_tensor(dense_out.data(), tensor::make_shape(
      meta::Any{Lq}, meta::cint<D>));
  auto dynamic_ot = tensor::make_tensor(dynamic_out.data(), tensor::make_shape(
      meta::Any{Lq}, meta::cint<D>));
  Config config{};
  config.selected_blocks = 3;
  config.static_probability = 1.0f;
  config.random_probability = 1.0f;
  auto dense = ops::dense_attention(config);
  kernel::Workspace dense_storage(
      dense.required_workspace(qt, kt, vt, dense_ot));
  auto dense_workspace = dense_storage.view();
  dense(dense_workspace, qt, kt, vt, dense_ot, 0.5f);

  auto dynamic = ops::dynamic_attention(config);
  kernel::Workspace dynamic_storage(
      dynamic.required_workspace(qt, kt, vt, dynamic_ot));
  auto dynamic_workspace = dynamic_storage.view();
  std::mt19937 rng(31);
  dynamic(dynamic_workspace, qt, kt, vt, dynamic_ot, 0.5f, rng);
  for (std::size_t i = 0; i < dense_out.size(); ++i)
    EXPECT_NEAR(dynamic_out[i], dense_out[i], 3.0e-2f) << "index=" << i;
}

TEST(AttentionTest, CoversFixedAlignedAndArbitrarySequenceMetadataAndDtypes) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  run_dense_dtype_meta_case<float32_t, bfloat16_t, float32_t>(
      meta::cint<32>, meta::cint<64>, 128, 128);
  run_dense_dtype_meta_case<float16_t, float32_t, float16_t>(
      meta::dyn<32, 32, 128>(32), meta::dyn<32, 32, 128>(96), 48, 32);
  run_dense_dtype_meta_case<bfloat16_t, float16_t, bfloat16_t>(
      meta::Any{33}, meta::Any{65}, 24, 48);
}

TEST(AttentionTest, OnlineDenseMatchesMaterializedDense) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 33;
  constexpr nint_t Lkv = 65;
  constexpr nint_t D = 24;
  std::vector<bfloat16_t> q(Lq * D);
  std::vector<bfloat16_t> k(Lkv * D);
  std::vector<bfloat16_t> v(Lkv * D);
  std::vector<float> online_out(Lq * D);
  std::vector<float> materialized_out(Lq * D);
  for (nint_t i = 0; i < Lq * D; ++i)
    q[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 11 - 5) / 16);
  for (nint_t i = 0; i < Lkv * D; ++i) {
    k[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 13 - 6) / 16);
    v[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 17 - 8) / 16);
  }
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::Any{Lq}, meta::Any{D}));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::Any{Lkv}, meta::Any{D}));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::Any{Lkv}, meta::Any{D}));
  auto oot = tensor::make_tensor(online_out.data(), tensor::make_shape(
      meta::Any{Lq}, meta::Any{D}));
  auto mot = tensor::make_tensor(materialized_out.data(), tensor::make_shape(
      meta::Any{Lq}, meta::Any{D}));
  auto online = ops::dense_attention(StreamingModelConfig{});
  auto materialized = ops::dense_materialized_attention(ModelConfig{});
  kernel::Workspace online_storage(
      online.required_workspace(qt, kt, vt, oot));
  kernel::Workspace materialized_storage(
      materialized.required_workspace(qt, kt, vt, mot));
  auto online_workspace = online_storage.view();
  auto materialized_workspace = materialized_storage.view();
  online(online_workspace, qt, kt, vt, oot, 0.25f);
  materialized(materialized_workspace, qt, kt, vt, mot, 0.25f);
  for (std::size_t i = 0; i < online_out.size(); ++i)
    EXPECT_NEAR(online_out[i], materialized_out[i], 3.0e-2f)
        << "index=" << i;
}

TEST(AttentionTest, AutomaticStrategyUsesShapeAppropriateWorkspace) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  auto verify = [](nint_t lq, nint_t lkv, nint_t d,
                   bool expect_materialized) {
    std::vector<bfloat16_t> q(static_cast<std::size_t>(lq * d));
    std::vector<bfloat16_t> k(static_cast<std::size_t>(lkv * d));
    std::vector<bfloat16_t> v(static_cast<std::size_t>(lkv * d));
    std::vector<float> out(static_cast<std::size_t>(lq * d));
    auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
        meta::Any{lq}, meta::Any{d}));
    auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
        meta::Any{lkv}, meta::Any{d}));
    auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
        meta::Any{lkv}, meta::Any{d}));
    auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
        meta::Any{lq}, meta::Any{d}));
    auto automatic = ops::dense_attention(ModelConfig{});
    const nint_t automatic_bytes =
        automatic.required_workspace(qt, kt, vt, ot);
    const nint_t expected_bytes = expect_materialized
        ? ops::dense_attention(MaterializedModelConfig{})
              .required_workspace(qt, kt, vt, ot)
        : ops::dense_attention(StreamingModelConfig{})
              .required_workspace(qt, kt, vt, ot);
    EXPECT_EQ(automatic_bytes, expected_bytes);
  };
#if defined(ARCH_X86_FAMILY)
  verify(1, 128, 128, true);
  verify(1, 2048, 128, true);
  verify(128, 512, 32, true);
  verify(128, 4096, 32, true);
  verify(128, 8193, 32, false);
#else
  verify(1, 128, 128, false);
  verify(1, 2048, 128, true);
  verify(128, 512, 32, true);
  verify(128, 4096, 32, false);
#endif
}

TEST(AttentionTest, BottomRightCausalModeSupportsKvCacheAlignment) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Lq = 3;
  constexpr nint_t Lkv = 5;
  constexpr nint_t D = 8;
  std::vector<bfloat16_t> q(Lq * D, bfloat16_t(0.25f));
  std::vector<bfloat16_t> k(Lkv * D, bfloat16_t(0.5f));
  std::vector<bfloat16_t> v(Lkv * D);
  std::vector<float> out(Lq * D);
  for (nint_t i = 0; i < Lkv * D; ++i)
    v[static_cast<std::size_t>(i)] = bfloat16_t(float(i % 13 - 6) / 8);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::cint<Lkv>, meta::cint<D>));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      meta::cint<Lq>, meta::cint<D>));
  auto op = ops::dense_attention(CausalConfig{});
  kernel::Workspace storage(op.required_workspace(qt, kt, vt, ot));
  auto workspace = storage.view();
  op(workspace, qt, kt, vt, ot, 0.5f);
  std::vector<bool> causal(Lq * Lkv, false);
  for (nint_t i = 0; i < Lq; ++i)
    for (nint_t j = 0; j < Lkv; ++j)
      causal[static_cast<std::size_t>(i * Lkv + j)] =
          j <= i + (Lkv - Lq);
  const auto expected = reference_attention(
      q, k, v, Lq, Lkv, D, D, 0.5f,
      nullptr, nullptr, &causal, nullptr);
  for (std::size_t i = 0; i < out.size(); ++i)
    EXPECT_NEAR(out[i], expected[i], 2.5e-2f) << "index=" << i;

  std::array<int32_t, 2> all_blocks{0, 1};
  std::vector<float> sparse_out(Lq * D);
  auto index = tensor::make_tensor(
      all_blocks.data(), tensor::make_shape(meta::cint<1>, meta::cint<2>));
  auto sparse_ot = tensor::make_tensor(
      sparse_out.data(), tensor::make_shape(meta::cint<Lq>, meta::cint<D>));
  auto sparse = ops::sparse_attention(CausalConfig{});
  kernel::Workspace sparse_storage(
      sparse.required_workspace(qt, kt, vt, index, sparse_ot));
  auto sparse_workspace = sparse_storage.view();
  sparse(sparse_workspace, qt, kt, vt, index, sparse_ot, 0.5f);
  for (std::size_t i = 0; i < sparse_out.size(); ++i)
    EXPECT_NEAR(sparse_out[i], expected[i], 2.5e-2f) << "index=" << i;

  std::vector<float> top_left_out(Lq * D);
  auto top_left_ot = tensor::make_tensor(
      top_left_out.data(), tensor::make_shape(meta::cint<Lq>, meta::cint<D>));
  auto top_left = ops::dense_attention(TopLeftCausalConfig{});
  kernel::Workspace top_left_storage(
      top_left.required_workspace(qt, kt, vt, top_left_ot));
  auto top_left_workspace = top_left_storage.view();
  top_left(top_left_workspace, qt, kt, vt, top_left_ot, 0.5f);
  for (nint_t i = 0; i < Lq; ++i)
    for (nint_t j = 0; j < Lkv; ++j)
      causal[static_cast<std::size_t>(i * Lkv + j)] = j <= i;
  const auto top_left_expected = reference_attention(
      q, k, v, Lq, Lkv, D, D, 0.5f,
      nullptr, nullptr, &causal, nullptr);
  for (std::size_t i = 0; i < top_left_out.size(); ++i)
    EXPECT_NEAR(top_left_out[i], top_left_expected[i], 2.5e-2f)
        << "index=" << i;
}

#endif

} // namespace
