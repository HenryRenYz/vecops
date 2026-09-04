#ifndef VECOPS_OPS_DETAILS_ATTENTION_DYNAMIC_H
#define VECOPS_OPS_DETAILS_ATTENTION_DYNAMIC_H

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/ops/details/attention/Sparse.h"

namespace vecops::ops {

/**
 * @brief Build the coarse block index and probability maps used by dynamic
 * block-sparse attention.
 *
 * Q/K rows are averaged in `Bq`/`Bkv` blocks, respecting query/key masks.
 * Coarse scores are normalized over every K block before selecting the top-S
 * entries. Entries contributing to `static_probability` receive weight -1;
 * the following probability band up to `random_probability` keeps its true
 * probability for optional sampling; the remaining slots are `(-1, 0)`.
 * Index and weight outputs must be writable, contiguous rank-two Tensor
 * buffers. They bypass DataAccess because the paired sparse kernel consumes
 * them as algorithm-private row-major storage.
 */
template <typename Config>
class DynamicAttentionMask {
  using MatmulOp = Matmul<typename Config::MatmulConfig>;
  using Atom = typename Config::Atom;
  using Mean = typename Atom::TA;
  using Score = typename Config::ScoreType;
  using VectorPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct>;
  using MatrixPolicy = tensor::InputAccessPolicy<
      1, 1, tensor::AccessPlan::direct>;
#if defined(ARCH_X86_FAMILY)
  static constexpr int traversal_tail_factor = 2;
#else
  static constexpr int traversal_tail_factor = 1;
#endif

public:
  using ResourceRequirements = typename MatmulOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit DynamicAttentionMask(Config cfg)
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out) const {
    auto q_spec = tensor::as_input_spec<Score>(q);
    auto k_spec = tensor::as_input_spec<Score>(k);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = attention_details::as_optional_input<Score>(bias);
    return required_workspace_specs(
        q_spec, k_spec, qm_spec, km_spec, am_spec, bias_spec,
        index_out, weight_out);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k,
      const IndexOutput& index_out, const WeightOutput& weight_out) const {
    return required_workspace(
        q, k, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, index_out, weight_out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out,
      Score scale) const {
    auto q_spec = tensor::as_input_spec<Score>(q);
    auto k_spec = tensor::as_input_spec<Score>(k);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = attention_details::as_optional_input<Score>(bias);
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          execute_specs(
              active, q_spec, k_spec, qm_spec, km_spec, am_spec,
              bias_spec, index_out, weight_out, scale);
        });
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k,
      const IndexOutput& index_out, const WeightOutput& weight_out,
      Score scale) const {
    (*this)(scope, q, k, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt,
            index_out, weight_out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const Q& q, const K& k,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out,
      Score scale) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, query_mask, key_mask,
            attention_mask, bias, index_out, weight_out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const Q& q, const K& k,
      const IndexOutput& index_out, const WeightOutput& weight_out,
      Score scale) const {
    (*this)(workspace, q, k, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt,
            index_out, weight_out, scale);
  }

private:
  template <typename QSpec, typename KSpec, typename QMSpec,
            typename KMSpec, typename AMSpec, typename BiasSpec,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void validate_specs(
      const QSpec& q, const KSpec& k,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out) const {
    static_assert(QSpec::InputTensor::Ndim == 2);
    static_assert(KSpec::InputTensor::Ndim == 2);
    attention_details::validate_contiguous_buffer<int32_t, 2>(
        index_out, "dynamic-attention index output must be contiguous");
    attention_details::validate_contiguous_buffer<Score, 2>(
        weight_out, "dynamic-attention weight output must be contiguous");
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    VECOPS_ASSERT(lq > 0 && lkv > 0 && q.input_layout().shape()[1] > 0,
                  "dynamic-attention dimensions must be non-empty");
    VECOPS_ASSERT(q.input_layout().shape()[1] ==
                  k.input_layout().shape()[1],
                  "dynamic-attention Q/K head dimensions differ");
    const nint_t tq = ceil_div(
        lq, static_cast<nint_t>(Config::query_block));
    VECOPS_ASSERT(index_out.layout().shape()[0] == tq &&
                  index_out.layout().shape()[1] > 0,
                  "dynamic-attention index output shape mismatch");
    VECOPS_ASSERT(weight_out.layout().shape()[0] == tq &&
                  weight_out.layout().shape()[1] ==
                      index_out.layout().shape()[1],
                  "dynamic-attention weight output shape mismatch");
    VECOPS_ASSERT(config.static_probability >= Score{} &&
                  config.static_probability <= Score{1} &&
                  config.random_probability >= config.static_probability &&
                  config.random_probability <= Score{1} &&
                  config.minimum_probability >= Score{} &&
                  config.minimum_probability <= Score{1},
                  "dynamic-attention probability thresholds are invalid");
    attention_details::validate_vector_optional(
        query_mask, lq, "attention query-mask shape mismatch");
    attention_details::validate_vector_optional(
        key_mask, lkv, "attention key-mask shape mismatch");
    attention_details::validate_matrix_optional(
        attention_mask, lq, lkv,
        "attention keep-mask shape mismatch");
    attention_details::validate_matrix_optional(
        bias, lq, lkv, "attention bias shape mismatch");
  }

  template <typename QSpec, typename KSpec, typename QMSpec,
            typename KMSpec, typename AMSpec, typename BiasSpec,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE nint_t required_workspace_specs(
      const QSpec& q, const KSpec& k,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out) const {
    validate_specs(
        q, k, query_mask, key_mask, attention_mask, bias,
        index_out, weight_out);
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t d = q.input_layout().shape()[1];
    const nint_t tq = ceil_div(lq, static_cast<nint_t>(Config::query_block));
    const nint_t tkv = ceil_div(
        lkv, static_cast<nint_t>(Config::key_value_block));
    const nint_t selected = index_out.layout().shape()[1];
    auto tq_extent = meta::Any{tq};
    auto tkv_extent = meta::Any{tkv};
    auto d_extent = tensor::size_value<1>(q.input_layout());
    auto q_mean_tensor = tensor::make_tensor(
        static_cast<Mean*>(nullptr),
        tensor::make_shape(tq_extent, d_extent));
    auto k_mean_tensor = tensor::make_tensor(
        static_cast<Mean*>(nullptr),
        tensor::make_shape(tkv_extent, d_extent));
    auto score_tensor = tensor::make_tensor(
        static_cast<Score*>(nullptr),
        tensor::make_shape(tq_extent, tkv_extent));
    MatmulOp matmul_op{config.matmul};
    const nint_t matmul_bytes = matmul_op.required_workspace(
        tq_extent, tkv_extent, d_extent,
        q_mean_tensor, k_mean_tensor, score_tensor);
    const nint_t persistent =
        attention_details::aligned_bytes<Mean>(tq * d) +
        attention_details::aligned_bytes<Mean>(tkv * d) +
        attention_details::aligned_bytes<Score>(tq * tkv) +
        attention_details::aligned_bytes<Score>(selected) +
        attention_details::aligned_bytes<int32_t>(selected) +
        tensor::required_workspace(q, MatrixPolicy{}) +
        tensor::required_workspace(k, MatrixPolicy{}) +
        attention_details::optional_required_workspace(
            query_mask, VectorPolicy{}) +
        attention_details::optional_required_workspace(
            key_mask, VectorPolicy{}) +
        attention_details::optional_required_workspace(
            attention_mask, MatrixPolicy{}) +
        attention_details::optional_required_workspace(
            bias, MatrixPolicy{});
    return persistent + matmul_bytes;
  }

  template <execution::ExecutionScope Scope,
            typename QSpec, typename KSpec, typename QMSpec,
            typename KMSpec, typename AMSpec, typename BiasSpec,
            attention_details::WritableAttentionBufferTensor<int32_t, 2>
                IndexOutput,
            attention_details::WritableAttentionBufferTensor<Score, 2>
                WeightOutput>
  VECOPS_INLINE void execute_specs(
      Scope& scope, const QSpec& q, const KSpec& k,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const IndexOutput& index_out, const WeightOutput& weight_out,
      Score scale) const {
    validate_specs(
        q, k, query_mask, key_mask, attention_mask, bias,
        index_out, weight_out);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t d = q.input_layout().shape()[1];
    const nint_t tq = ceil_div(lq, static_cast<nint_t>(Config::query_block));
    const nint_t tkv = ceil_div(
        lkv, static_cast<nint_t>(Config::key_value_block));
    const nint_t selected = index_out.layout().shape()[1];
    Mean* q_mean = attention_details::allocate_aligned<Mean>(
        workspace, tq * d);
    Mean* k_mean = attention_details::allocate_aligned<Mean>(
        workspace, tkv * d);
    Score* coarse = attention_details::allocate_aligned<Score>(
        workspace, tq * tkv);
    Score* top_probability = attention_details::allocate_aligned<Score>(
        workspace, selected);
    int32_t* top_index = attention_details::allocate_aligned<int32_t>(
        workspace, selected);

    auto q_access = tensor::bind(q, MatrixPolicy{}, workspace);
    auto k_access = tensor::bind(k, MatrixPolicy{}, workspace);
    auto qm = attention_details::bind_optional_input(
        query_mask, VectorPolicy{}, workspace);
    auto km = attention_details::bind_optional_input(
        key_mask, VectorPolicy{}, workspace);
    auto am = attention_details::bind_optional_input(
        attention_mask, MatrixPolicy{}, workspace);
    auto b = attention_details::bind_optional_input(
        bias, MatrixPolicy{}, workspace);
    int32_t* index_data = index_out.data();
    Score* weight_data = weight_out.data();

    auto compute_block_means = [&]<nint_t BlockSize>(
                                   auto& access, auto& mask,
                                   nint_t blocks, nint_t length,
                                   Mean* destination) {
      using Tag = vec::ScalableTag<Score, 0>;
      Tag tag{};
      tensor::with_unordered_access(access, [&](auto unordered) {
        attention_details::with_optional_access(
            [&](auto umask) {
              for (nint_t block = 0; block < blocks; ++block) {
                const nint_t begin = block * BlockSize;
                const nint_t rows = std::min<nint_t>(
                    BlockSize, length - begin);
                std::array<bool, static_cast<std::size_t>(BlockSize)>
                    row_active{};
                nint_t count = rows;
                if constexpr (!tensor::is_nullopt_v<decltype(umask)>) {
                  count = 0;
                  for (nint_t row = 0; row < rows; ++row) {
                    const bool keep =
                        umask.load_scalar(tensor::coord(begin + row)) !=
                        Score{};
                    row_active[static_cast<std::size_t>(row)] = keep;
                    count += static_cast<nint_t>(keep);
                  }
                }
                const Score reciprocal = count > 0
                    ? Score{1} / static_cast<Score>(count)
                    : Score{};
                kernel::loop::fold<4, traversal_tail_factor>(
                    tag, d,
                    [&](auto block_tag, nint_t feature, auto active,
                        const auto& reciprocal_v) VECOPS_INLINE_LAMBDA {
                      auto sum = vec::zeros(block_tag);
                      for (nint_t row = 0; row < rows; ++row) {
                        if constexpr (!tensor::is_nullopt_v<decltype(umask)>) {
                          if (!row_active[static_cast<std::size_t>(row)])
                            continue;
                        }
                        auto value = unordered.load(
                            block_tag,
                            tensor::coord(begin + row, feature),
                            tensor::axis<1>, active, vec::opt::zero);
                        sum = vec::add(block_tag, sum, value);
                      }
                      sum = vec::mul(block_tag, sum, reciprocal_v);
                      vec::store_convert(
                          block_tag,
                          destination + block * d + feature, sum, active);
                    },
                    kernel::loop::invariant(reciprocal));
              }
            },
            mask);
      });
    };
    compute_block_means.template operator()<Config::query_block>(
        q_access, qm, tq, lq, q_mean);
    compute_block_means.template operator()<Config::key_value_block>(
        k_access, km, tkv, lkv, k_mean);

    auto tq_extent = meta::Any{tq};
    auto tkv_extent = meta::Any{tkv};
    auto d_extent = tensor::size_value<1>(q.input_layout());
    auto q_mean_tensor = tensor::make_tensor(
        q_mean, tensor::make_shape(tq_extent, d_extent));
    auto k_mean_tensor = tensor::make_tensor(
        k_mean, tensor::make_shape(tkv_extent, d_extent));
    auto coarse_tensor = tensor::make_tensor(
        coarse, tensor::make_shape(tq_extent, tkv_extent));
    MatmulOp matmul_op{config.matmul};
    matmul_op(
        scope, tq_extent, tkv_extent, d_extent,
        q_mean_tensor, k_mean_tensor, coarse_tensor);

    attention_details::with_optional_access(
        [&](auto uqm, auto ukm, auto uam, auto ubias) {
          for (nint_t qi = 0; qi < tq; ++qi) {
                Score maximum = -std::numeric_limits<Score>::infinity();
                bool any_valid = false;
                const nint_t q_begin = qi * Config::query_block;
                const nint_t q_rows = std::min<nint_t>(
                    Config::query_block, lq - q_begin);
                std::array<
                    bool,
                    static_cast<std::size_t>(Config::query_block)>
                    query_active{};
                for (nint_t qr = 0; qr < q_rows; ++qr) {
                  bool active = true;
                  if constexpr (!tensor::is_nullopt_v<decltype(uqm)>) {
                    active = uqm.load_scalar(
                        tensor::coord(q_begin + qr)) != Score{};
                  }
                  query_active[static_cast<std::size_t>(qr)] = active;
                }
                for (nint_t kj = 0; kj < tkv; ++kj) {
                  const nint_t k_begin = kj * Config::key_value_block;
                  const nint_t k_rows = std::min<nint_t>(
                      Config::key_value_block, lkv - k_begin);
                  Score bias_sum{};
                  nint_t pair_count = 0;
                  for (nint_t qr = 0; qr < q_rows; ++qr) {
                    const nint_t q_index = q_begin + qr;
                    if (!query_active[static_cast<std::size_t>(qr)]) continue;
                    using Tag = vec::ScalableTag<Score, 0>;
                    Tag tag{};
                    Score row_bias_sum{};
                    nint_t row_pair_count = 0;
                    kernel::loop::fold<4, traversal_tail_factor>(
                        tag, k_rows,
                        [&](auto block_tag, nint_t kr, auto active,
                            auto& bias_sum_v) VECOPS_INLINE_LAMBDA {
                          const nint_t active_count = std::min(
                              vec::size(block_tag), k_rows - kr);
                          auto keep = attention_details::attention_keep_mask<
                              Config::causal_mode>(
                              block_tag, active, active_count, true,
                              q_index, k_begin, kr, lq, lkv, ukm, uam);
                          row_pair_count += vec::mask_count(block_tag, keep);
                          if constexpr (
                              !tensor::is_nullopt_v<decltype(ubias)>) {
                            auto bias_v = ubias.load(
                                block_tag,
                                tensor::coord(q_index, k_begin + kr),
                                tensor::axis<1>, active, vec::opt::zero);
                            bias_sum_v = vec::add(
                                bias_sum_v,
                                vec::blend(
                                    block_tag, vec::zeros(block_tag), keep,
                                    bias_v));
                          }
                        },
                        kernel::loop::reduce_add(row_bias_sum));
                    bias_sum += row_bias_sum;
                    pair_count += row_pair_count;
                  }
                  Score value = -std::numeric_limits<Score>::infinity();
                  if (pair_count > 0) {
                    value = coarse[qi * tkv + kj] * scale;
                    if constexpr (!tensor::is_nullopt_v<decltype(ubias)>) {
                      value += bias_sum / static_cast<Score>(pair_count);
                    }
                    maximum = std::max(maximum, value);
                    any_valid = true;
                  }
                  coarse[qi * tkv + kj] = value;
                }

                Score normalizer{};
                if (any_valid) {
                  using Tag = vec::ScalableTag<Score, 0>;
                  Tag tag{};
                  kernel::loop::fold<4, 1>(
                      tag, tkv,
                      [&](auto block_tag, nint_t kj, auto active,
                          auto& normalizer_v, const auto& maximum_v,
                          const auto& negative_infinity)
                          VECOPS_INLINE_LAMBDA {
                        auto value = vec::load(
                            block_tag, coarse + qi * tkv + kj, active,
                            vec::opt::merge(negative_infinity));
                        auto probability = vec::exp_neg(
                            block_tag,
                            vec::sub(block_tag, value, maximum_v),
                            vec::opt::math::accuracy<Config::exp_accuracy>,
                            active, vec::opt::zero);
                        normalizer_v = vec::add(
                            normalizer_v, probability);
                        vec::store(
                            block_tag, coarse + qi * tkv + kj,
                            probability, active);
                      },
                      kernel::loop::reduce_add(normalizer),
                      kernel::loop::invariant(maximum),
                      kernel::loop::invariant(
                          -std::numeric_limits<Score>::infinity()));
                  const Score reciprocal = Score{1} / normalizer;
                  kernel::loop::fold<4, 1>(
                      tag, tkv,
                      [&](auto block_tag, nint_t kj, auto active,
                          const auto& reciprocal_v)
                          VECOPS_INLINE_LAMBDA {
                        auto probability = vec::load(
                            block_tag, coarse + qi * tkv + kj,
                            active, vec::opt::zero);
                        probability = vec::mul(
                            block_tag, probability, reciprocal_v);
                        vec::store(
                            block_tag, coarse + qi * tkv + kj,
                            probability, active);
                      },
                      kernel::loop::invariant(reciprocal));
                }
                std::fill(
                    top_probability, top_probability + selected, Score{-1});
                std::fill(top_index, top_index + selected, int32_t{-1});
                if (normalizer > Score{}) {
                  for (nint_t kj = 0; kj < tkv; ++kj) {
                    const Score probability = coarse[qi * tkv + kj];
                    nint_t position = 0;
                    while (position < selected &&
                           (top_probability[position] > probability ||
                            (top_probability[position] == probability &&
                             top_index[position] >= 0 &&
                             top_index[position] < kj))) {
                      ++position;
                    }
                    if (position == selected) continue;
                    for (nint_t move = selected - 1;
                         move > position; --move) {
                      top_probability[move] = top_probability[move - 1];
                      top_index[move] = top_index[move - 1];
                    }
                    top_probability[position] = probability;
                    top_index[position] = static_cast<int32_t>(kj);
                  }
                }

                Score cdf{};
                nint_t slot = 0;
                for (; slot < selected; ++slot) {
                  if (top_index[slot] < 0 ||
                      cdf >= config.static_probability ||
                      top_probability[slot] < config.minimum_probability)
                    break;
                  cdf += top_probability[slot];
                  index_data[qi * selected + slot] = top_index[slot];
                  weight_data[qi * selected + slot] = Score{-1};
                }
                for (; slot < selected; ++slot) {
                  if (top_index[slot] < 0 ||
                      cdf >= config.random_probability ||
                      top_probability[slot] < config.minimum_probability)
                    break;
                  cdf += top_probability[slot];
                  index_data[qi * selected + slot] = top_index[slot];
                  weight_data[qi * selected + slot] = top_probability[slot];
                }
                for (; slot < selected; ++slot) {
                  index_data[qi * selected + slot] = int32_t{-1};
                  weight_data[qi * selected + slot] = Score{};
                }
          }
        },
        qm, km, am, b);
    workspace.rewind(mark);
  }
};

template <typename Config>
using DynAttentionMask = DynamicAttentionMask<Config>;

/**
 * @brief Compose dynamic block selection with sampled sparse attention.
 *
 * Index and weight are a compile-time pair: both are contiguous rank-two
 * Tensor buffers, or both are `tensor::nullopt`. In the omitted form the maps
 * are created in caller workspace with `config.selected_blocks` columns and
 * consumed before that workspace region is released.
 */
template <typename Config>
class DynamicAttention {
  using Score = typename Config::ScoreType;
  using MaskBuilder = DynamicAttentionMask<Config>;
  using SparseOp = SparseFlashAttention<Config>;

public:
  using ResourceRequirements = typename SparseOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit DynamicAttention(Config cfg)
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalAttentionBuffer<int32_t, 2> Index,
            attention_details::OptionalAttentionBuffer<Score, 2> Weight,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v,
      const Index& index, const Weight& weight,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out) const {
    constexpr bool HasIndex = !tensor::is_nullopt_v<Index>;
    constexpr bool HasWeight = !tensor::is_nullopt_v<Weight>;
    static_assert(HasIndex == HasWeight,
                  "dynamic attention index and weight must be both present or both omitted");
    SparseOp sparse{config};
    if constexpr (HasIndex) {
      return sparse.required_workspace(
          q, k, v, index, weight, query_mask, key_mask,
          attention_mask, bias, out);
    } else {
      auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const nint_t lq = q_spec.input_layout().shape()[0];
      const nint_t tq = ceil_div(
          lq, static_cast<nint_t>(Config::query_block));
      VECOPS_ASSERT(config.selected_blocks > 0,
                    "dynamic attention selected block count must be positive");
      auto map_shape = tensor::make_shape(
          meta::Any{tq}, meta::Any{config.selected_blocks});
      auto index_tensor = tensor::make_tensor(
          static_cast<int32_t*>(nullptr), map_shape);
      auto weight_tensor = tensor::make_tensor(
          static_cast<Score*>(nullptr), map_shape);
      MaskBuilder builder{config};
      const nint_t map_bytes =
          attention_details::aligned_bytes<int32_t>(
              tq * config.selected_blocks) +
          attention_details::aligned_bytes<Score>(
              tq * config.selected_blocks);
      const nint_t builder_bytes = builder.required_workspace(
          q, k, query_mask, key_mask, attention_mask, bias,
          index_tensor, weight_tensor);
      const nint_t sparse_bytes = sparse.required_workspace(
          q, k, v, index_tensor, weight_tensor,
          query_mask, key_mask, attention_mask, bias, out);
      return map_bytes + std::max(builder_bytes, sparse_bytes);
    }
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Output& out) const {
    return required_workspace(
        q, k, v, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalAttentionBuffer<int32_t, 2> Index,
            attention_details::OptionalAttentionBuffer<Score, 2> Weight,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output, typename RNG>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Index& index, const Weight& weight,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale, RNG& rng) const {
    constexpr bool HasIndex = !tensor::is_nullopt_v<Index>;
    constexpr bool HasWeight = !tensor::is_nullopt_v<Weight>;
    static_assert(HasIndex == HasWeight,
                  "dynamic attention index and weight must be both present or both omitted");
    SparseOp sparse{config};
    if constexpr (HasIndex) {
      sparse(
          scope, q, k, v, index, weight,
          query_mask, key_mask, attention_mask, bias, out, scale, rng);
    } else {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const nint_t tq = ceil_div(
          q_spec.input_layout().shape()[0],
          static_cast<nint_t>(Config::query_block));
      VECOPS_ASSERT(config.selected_blocks > 0,
                    "dynamic attention selected block count must be positive");
      int32_t* index_data = attention_details::allocate_aligned<int32_t>(
          workspace, tq * config.selected_blocks);
      Score* weight_data = attention_details::allocate_aligned<Score>(
          workspace, tq * config.selected_blocks);
      auto map_shape = tensor::make_shape(
          meta::Any{tq}, meta::Any{config.selected_blocks});
      auto index_tensor = tensor::make_tensor(index_data, map_shape);
      auto weight_tensor = tensor::make_tensor(weight_data, map_shape);
      MaskBuilder builder{config};
      builder(
          scope, q, k, query_mask, key_mask, attention_mask, bias,
          index_tensor, weight_tensor, scale);
      sparse(
          scope, q, k, v, index_tensor, weight_tensor,
          query_mask, key_mask, attention_mask, bias, out, scale, rng);
      workspace.rewind(mark);
    }
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalAttentionBuffer<int32_t, 2> Index,
            attention_details::OptionalAttentionBuffer<Score, 2> Weight,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output, typename RNG>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const Index& index, const Weight& weight,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale, RNG& rng) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, v, index, weight,
            query_mask, key_mask, attention_mask, bias, out, scale, rng);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output,
            typename RNG>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const Output& out, Score scale, RNG& rng) const {
    (*this)(workspace, q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, out, scale, rng);
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto dynamic_attention_mask(Config config) {
  return DynamicAttentionMask<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto dynamic_attention(Config config) {
  return DynamicAttention<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_DYNAMIC_H
