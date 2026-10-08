// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_DETAILS_ATTENTION_IBS_ATTENTION_H
#define VECOPS_OPS_DETAILS_ATTENTION_IBS_ATTENTION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "vecops/ops/details/attention/SparseFlashAttention.h"

namespace vecops::ops {

/**
 * @brief Compute the reusable base index and weight maps for IBS Attention.
 *
 * Q/K rows are averaged in `Bq`/`Bkv` blocks, respecting query/key masks.
 * Coarse scores are normalized over every K block before selecting the top-S
 * entries. Entries contributing to `static_probability` receive weight -1;
 * the following probability band up to `random_probability` keeps its true
 * probability for optional sampling; the remaining slots are `(-1, 0)`.
 * Index and weight outputs must be writable, contiguous rank-two Tensor
 * buffers. They bypass DataAccess because IBS Attention consumes them as
 * algorithm-private row-major storage. This implements the paper's base
 * index-computation stage; weighted sampling remains part of IBSAttention.
 */
template <typename Config>
class IBSAttentionIndexer {
  using MatmulOp = Matmul<typename Config::MatmulConfig>;
  using Atom = typename Config::Atom;
  using Mean = typename Atom::TA;
  using Score = typename Config::ScoreType;
  using VectorPolicy = tensor::InputAccessPolicy<0, 1, tensor::AccessPlan::direct>;
  using MatrixPolicy = tensor::InputAccessPolicy<1, 1, tensor::AccessPlan::direct>;
#if defined(ARCH_X86_FAMILY)
  static constexpr int traversal_tail_factor = 2;
#else
  static constexpr int traversal_tail_factor = 1;
#endif

public:
  using ResourceRequirements = typename MatmulOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit IBSAttentionIndexer(Config cfg)
    : config(std::move(cfg)) {
  }

  template <nint_t Parallelism, typename WorkerScratch>
  class PatternPrepared {
  public:
    using ResourceRequirements = typename IBSAttentionIndexer::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(Config config, WorkerScratch scratch, nint_t workspace_bytes)
      : config_(std::move(config))
      , scratch_(std::move(scratch))
      , workspace_bytes_(workspace_bytes) {
    }

    template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::OptionalInputOperandOf<1> QueryMask,
              tensor::OptionalInputOperandOf<1> KeyMask, tensor::OptionalInputOperandOf<2> AttentionMask,
              tensor::OptionalInputOperandOf<2> Bias, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
              tensor::WritableTensorOf<Score, 2> WeightOutput>
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, const Q& q, const K& k,
                                  const QueryMask& query_mask, const KeyMask& key_mask,
                                  const AttentionMask& attention_mask, const Bias& bias, const IndexOutput& index_out,
                                  const WeightOutput& weight_out, Score scale) const {
      IBSAttentionIndexer operation{config_};
      VECOPS_CHECK(operation.required_workspace(q, k, query_mask, key_mask, attention_mask, bias, index_out,
                                                weight_out) <= workspace_bytes_,
                   "active IBS indexer scratch exceeds its planning-pattern capacity");
      auto scratch = task.local(scratch_);
      kernel::WorkspaceView view{scratch.data(), workspace_bytes_};
      ExecutionSession execution{view};
      operation(execution, q, k, query_mask, key_mask, attention_mask, bias, index_out, weight_out, scale);
    }

  private:
    Config config_;
    WorkerScratch scratch_;
    nint_t workspace_bytes_ = 0;
  };

  template <nint_t Parallelism, typename WorkspaceAuthority, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::UnboundTensorLike IndexOutput, tensor::UnboundTensorLike WeightOutput>
    requires(tensor::is_unbound_tensor_view_v<Q> && tensor::is_unbound_tensor_view_v<K> &&
             (tensor::is_nullopt_v<QueryMask> || tensor::is_unbound_tensor_view_v<QueryMask>) &&
             (tensor::is_nullopt_v<KeyMask> || tensor::is_unbound_tensor_view_v<KeyMask>) &&
             (tensor::is_nullopt_v<AttentionMask> || tensor::is_unbound_tensor_view_v<AttentionMask>) &&
             (tensor::is_nullopt_v<Bias> || tensor::is_unbound_tensor_view_v<Bias>) && IndexOutput::Ndim == 2 &&
             WeightOutput::Ndim == 2 && std::same_as<typename IndexOutput::ElementType, int32_t> &&
             std::same_as<typename WeightOutput::ElementType, Score>)
  VECOPS_INLINE auto prepare(WorkspaceAuthority& workspace, std::string_view site_name, Q&& q, K&& k,
                             QueryMask&& query_mask, KeyMask&& key_mask, AttentionMask&& attention_mask, Bias&& bias,
                             IndexOutput&& index_out, WeightOutput&& weight_out) const {
    auto bound_index = tensor::bind(index_out, static_cast<int32_t*>(nullptr));
    auto bound_weight = tensor::bind(weight_out, static_cast<Score*>(nullptr));
    const nint_t bytes =
      required_workspace(q, k, query_mask, key_mask, attention_mask, bias, bound_index, bound_weight);
    auto scratch =
      workspace.template worker_tensor<std::byte, Parallelism>(site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<Parallelism, decltype(scratch)>{config, std::move(scratch), bytes};
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask, tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
            tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE nint_t required_workspace(const Q& q, const K& k, const QueryMask& query_mask, const KeyMask& key_mask,
                                          const AttentionMask& attention_mask, const Bias& bias,
                                          const IndexOutput& index_out, const WeightOutput& weight_out) const {
    auto q_spec = tensor::as_input_spec<Score>(q);
    auto k_spec = tensor::as_input_spec<Score>(k);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = tensor::as_input_spec<Score>(bias);
    return required_workspace_specs(q_spec, k_spec, qm_spec, km_spec, am_spec, bias_spec, index_out, weight_out);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
            tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE nint_t required_workspace(const Q& q, const K& k, const IndexOutput& index_out,
                                          const WeightOutput& weight_out) const {
    return required_workspace(q, k, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt, index_out,
                              weight_out);
  }

  template <execution::ExecutionScope Scope, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::WritableTensorOf<int32_t, 2> IndexOutput, tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void operator()(Scope& scope, const Q& q, const K& k, const QueryMask& query_mask,
                                const KeyMask& key_mask, const AttentionMask& attention_mask, const Bias& bias,
                                const IndexOutput& index_out, const WeightOutput& weight_out, Score scale) const {
    auto q_spec = tensor::as_input_spec<Score>(q);
    auto k_spec = tensor::as_input_spec<Score>(k);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = tensor::as_input_spec<Score>(bias);
    scope.with_resources(*this, [&](auto& active) VECOPS_INLINE_LAMBDA {
      execute_specs(active, q_spec, k_spec, qm_spec, km_spec, am_spec, bias_spec, index_out, weight_out, scale);
    });
  }

  template <execution::ExecutionScope Scope, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::WritableTensorOf<int32_t, 2> IndexOutput, tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void operator()(Scope& scope, const Q& q, const K& k, const IndexOutput& index_out,
                                const WeightOutput& weight_out, Score scale) const {
    (*this)(scope, q, k, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt, index_out, weight_out,
            scale);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask, tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
            tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const Q& q, const K& k, const QueryMask& query_mask,
                                const KeyMask& key_mask, const AttentionMask& attention_mask, const Bias& bias,
                                const IndexOutput& index_out, const WeightOutput& weight_out, Score scale) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, query_mask, key_mask, attention_mask, bias, index_out, weight_out, scale);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
            tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const Q& q, const K& k, const IndexOutput& index_out,
                                const WeightOutput& weight_out, Score scale) const {
    (*this)(workspace, q, k, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt, index_out, weight_out,
            scale);
  }

private:
  template <typename QSpec, typename KSpec, typename QMSpec, typename KMSpec, typename AMSpec, typename BiasSpec,
            tensor::WritableTensorOf<int32_t, 2> IndexOutput, tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void validate_specs(const QSpec& q, const KSpec& k, const QMSpec& query_mask, const KMSpec& key_mask,
                                    const AMSpec& attention_mask, const BiasSpec& bias, const IndexOutput& index_out,
                                    const WeightOutput& weight_out) const {
    static_assert(QSpec::InputTensor::Ndim == 2);
    static_assert(KSpec::InputTensor::Ndim == 2);
    attention_details::validate_contiguous_buffer<int32_t, 2>(index_out,
                                                              "dynamic-attention index output must be contiguous");
    attention_details::validate_contiguous_buffer<Score, 2>(weight_out,
                                                            "dynamic-attention weight output must be contiguous");
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    VECOPS_ASSERT(lq > 0 && lkv > 0 && tensor::size<1>(q.input_layout()) > 0,
                  "dynamic-attention dimensions must be non-empty");
    VECOPS_ASSERT(tensor::size<1>(q.input_layout()) == tensor::size<1>(k.input_layout()),
                  "dynamic-attention Q/K head dimensions differ");
    const auto tq = ceil_div(lq, Config::query_block_extent);
    VECOPS_ASSERT(tensor::size<0>(index_out.layout()) == tq && tensor::size<1>(index_out.layout()) > 0,
                  "dynamic-attention index output shape mismatch");
    VECOPS_ASSERT(tensor::size<0>(weight_out.layout()) == tq &&
                    tensor::size<1>(weight_out.layout()) == tensor::size<1>(index_out.layout()),
                  "dynamic-attention weight output shape mismatch");
    VECOPS_ASSERT(config.static_probability >= Score{} && config.static_probability <= Score{1} &&
                    config.random_probability >= config.static_probability && config.random_probability <= Score{1} &&
                    config.minimum_probability >= Score{} && config.minimum_probability <= Score{1},
                  "dynamic-attention probability thresholds are invalid");
    attention_details::validate_vector_optional(query_mask, lq, "attention query-mask shape mismatch");
    attention_details::validate_vector_optional(key_mask, lkv, "attention key-mask shape mismatch");
    attention_details::validate_matrix_optional(attention_mask, lq, lkv, "attention keep-mask shape mismatch");
    attention_details::validate_matrix_optional(bias, lq, lkv, "attention bias shape mismatch");
  }

  template <typename QSpec, typename KSpec, typename QMSpec, typename KMSpec, typename AMSpec, typename BiasSpec,
            tensor::WritableTensorOf<int32_t, 2> IndexOutput, tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE nint_t required_workspace_specs(const QSpec& q, const KSpec& k, const QMSpec& query_mask,
                                                const KMSpec& key_mask, const AMSpec& attention_mask,
                                                const BiasSpec& bias, const IndexOutput& index_out,
                                                const WeightOutput& weight_out) const {
    validate_specs(q, k, query_mask, key_mask, attention_mask, bias, index_out, weight_out);
    const auto lq_extent = tensor::size<0>(q.input_layout());
    const auto lkv_extent = tensor::size<0>(k.input_layout());
    const auto d_extent = tensor::size<1>(q.input_layout());
    const auto tq_extent = ceil_div(lq_extent, Config::query_block_extent);
    const auto tkv_extent = ceil_div(lkv_extent, Config::key_value_block_extent);
    const auto selected_extent = tensor::size<1>(index_out.layout());
    auto q_mean_tensor = tensor::make_tensor(static_cast<Mean*>(nullptr), tensor::make_shape(tq_extent, d_extent));
    auto k_mean_tensor = tensor::make_tensor(static_cast<Mean*>(nullptr), tensor::make_shape(tkv_extent, d_extent));
    auto score_tensor = tensor::make_tensor(static_cast<Score*>(nullptr), tensor::make_shape(tq_extent, tkv_extent));
    MatmulOp matmul_op{config.matmul};
    const nint_t matmul_bytes =
      matmul_op.required_workspace(tq_extent, tkv_extent, d_extent, q_mean_tensor, k_mean_tensor, score_tensor);
    const nint_t persistent =
      kernel::WorkspaceView::allocation_bytes<Mean>(tq_extent * d_extent) +
      kernel::WorkspaceView::allocation_bytes<Mean>(tkv_extent * d_extent) +
      kernel::WorkspaceView::allocation_bytes<Score>(tq_extent * tkv_extent) +
      kernel::WorkspaceView::allocation_bytes<Score>(selected_extent) +
      kernel::WorkspaceView::allocation_bytes<int32_t>(selected_extent) +
      tensor::required_workspace(q, MatrixPolicy{}) + tensor::required_workspace(k, MatrixPolicy{}) +
      tensor::required_workspace(query_mask, VectorPolicy{}) + tensor::required_workspace(key_mask, VectorPolicy{}) +
      tensor::required_workspace(attention_mask, MatrixPolicy{}) + tensor::required_workspace(bias, MatrixPolicy{});
    return persistent + matmul_bytes;
  }

  template <execution::ExecutionScope Scope, typename QSpec, typename KSpec, typename QMSpec, typename KMSpec,
            typename AMSpec, typename BiasSpec, tensor::WritableTensorOf<int32_t, 2> IndexOutput,
            tensor::WritableTensorOf<Score, 2> WeightOutput>
  VECOPS_INLINE void execute_specs(Scope& scope, const QSpec& q, const KSpec& k, const QMSpec& query_mask,
                                   const KMSpec& key_mask, const AMSpec& attention_mask, const BiasSpec& bias,
                                   const IndexOutput& index_out, const WeightOutput& weight_out, Score scale) const {
    validate_specs(q, k, query_mask, key_mask, attention_mask, bias, index_out, weight_out);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const auto lq_extent = tensor::size<0>(q.input_layout());
    const auto lkv_extent = tensor::size<0>(k.input_layout());
    const auto d_extent = tensor::size<1>(q.input_layout());
    const auto tq_extent = ceil_div(lq_extent, Config::query_block_extent);
    const auto tkv_extent = ceil_div(lkv_extent, Config::key_value_block_extent);
    const auto selected_extent = tensor::size<1>(index_out.layout());
    const auto lq = lq_extent;
    const auto lkv = lkv_extent;
    const auto d = d_extent;
    const auto tq = tq_extent;
    const auto tkv = tkv_extent;
    const auto selected = selected_extent;
    auto q_mean_buffer = workspace.template allocate_tensor<Mean>(tensor::make_shape(tq_extent, d_extent));
    auto k_mean_buffer = workspace.template allocate_tensor<Mean>(tensor::make_shape(tkv_extent, d_extent));
    auto coarse_buffer = workspace.template allocate_tensor<Score>(tensor::make_shape(tq_extent, tkv_extent));
    auto top_probability_buffer = workspace.template allocate_tensor<Score>(tensor::make_shape(selected_extent));
    auto top_index_buffer = workspace.template allocate_tensor<int32_t>(tensor::make_shape(selected_extent));
    Mean* q_mean = q_mean_buffer.data();
    Mean* k_mean = k_mean_buffer.data();
    Score* coarse = coarse_buffer.data();
    Score* top_probability = top_probability_buffer.data();
    int32_t* top_index = top_index_buffer.data();

    auto q_access = tensor::bind(q, MatrixPolicy{}, workspace);
    auto k_access = tensor::bind(k, MatrixPolicy{}, workspace);
    auto qm = tensor::bind(query_mask, VectorPolicy{}, workspace);
    auto km = tensor::bind(key_mask, VectorPolicy{}, workspace);
    auto am = tensor::bind(attention_mask, MatrixPolicy{}, workspace);
    auto b = tensor::bind(bias, MatrixPolicy{}, workspace);
    int32_t* index_data = index_out.data();
    Score* weight_data = weight_out.data();

    auto compute_block_means = [&]<nint_t BlockSize>(auto& access, auto& mask, nint_t blocks, nint_t length,
                                                     Mean* destination) {
      using Tag = vec::ScalableTag<Score, 0>;
      Tag tag{};
      tensor::with_unordered_access(access, [&](auto unordered) {
        tensor::with_optional_unordered_access(
          [&](auto umask) {
            for (nint_t block = 0; block < blocks; ++block) {
              const nint_t begin = block * BlockSize;
              const nint_t rows = std::min<nint_t>(BlockSize, length - begin);
              std::array<Score, static_cast<std::size_t>(BlockSize)> row_active{};
              const nint_t count =
                attention_details::load_activity_segment<Score>(umask, begin, rows, row_active.data());
              const Score reciprocal = count > 0 ? Score{1} / static_cast<Score>(count) : Score{};
              kernel::loop::fold<4, traversal_tail_factor>(
                tag, d_extent,
                [&](auto block_tag, nint_t feature, auto active, const auto& reciprocal_v) VECOPS_INLINE_LAMBDA {
                  auto sum0 = vec::zeros(block_tag);
                  auto sum1 = vec::zeros(block_tag);
                  auto sum2 = vec::zeros(block_tag);
                  auto sum3 = vec::zeros(block_tag);
                  auto accumulate_row = [&](auto accumulator, nint_t row) VECOPS_INLINE_LAMBDA {
                    if constexpr (!tensor::is_nullopt_v<decltype(umask)>) {
                      if (row_active[static_cast<std::size_t>(row)] == Score{})
                        return accumulator;
                    }
                    auto value = unordered.load(block_tag, tensor::coord(begin + row, feature), tensor::axis<1>, active,
                                                vec::opt::zero);
                    return vec::add(block_tag, accumulator, value);
                  };
                  nint_t row = 0;
                  for (; row + 4 <= rows; row += 4) {
                    sum0 = accumulate_row(sum0, row);
                    sum1 = accumulate_row(sum1, row + 1);
                    sum2 = accumulate_row(sum2, row + 2);
                    sum3 = accumulate_row(sum3, row + 3);
                  }
                  auto sum = vec::add(block_tag, vec::add(block_tag, sum0, sum1), vec::add(block_tag, sum2, sum3));
                  for (; row < rows; ++row)
                    sum = accumulate_row(sum, row);
                  sum = vec::mul(block_tag, sum, reciprocal_v);
                  vec::store_convert(block_tag, destination + block * d_extent + feature, sum, active);
                },
                kernel::loop::invariant(reciprocal));
            }
          },
          mask);
      });
    };
    compute_block_means.template operator()<Config::query_block>(q_access, qm, tq_extent, lq_extent, q_mean);
    compute_block_means.template operator()<Config::key_value_block>(k_access, km, tkv_extent, lkv_extent, k_mean);
    auto q_mean_tensor = tensor::make_tensor(q_mean, tensor::make_shape(tq_extent, d_extent));
    auto k_mean_tensor = tensor::make_tensor(k_mean, tensor::make_shape(tkv_extent, d_extent));
    auto coarse_tensor = tensor::make_tensor(coarse, tensor::make_shape(tq_extent, tkv_extent));
    MatmulOp matmul_op{config.matmul};
    matmul_op(scope, tq_extent, tkv_extent, d_extent, q_mean_tensor, k_mean_tensor, coarse_tensor);

    tensor::with_optional_unordered_access(
      [&](auto uqm, auto ukm, auto uam, auto ubias) {
        for (nint_t qi = 0; qi < tq; ++qi) {
          Score maximum = -std::numeric_limits<Score>::infinity();
          bool any_valid = false;
          const nint_t q_begin = qi * Config::query_block;
          const nint_t q_rows = std::min<nint_t>(Config::query_block, lq - q_begin);
          std::array<Score, static_cast<std::size_t>(Config::query_block)> query_active{};
          const nint_t active_queries =
            attention_details::load_activity_segment<Score>(uqm, q_begin, q_rows, query_active.data());
          if constexpr (Config::causal_mode == AttentionCausalMode::none && tensor::is_nullopt_v<decltype(ukm)> &&
                        tensor::is_nullopt_v<decltype(uam)> && tensor::is_nullopt_v<decltype(ubias)>) {
            if (active_queries > 0) {
              using Tag = vec::ScalableTag<Score, 0>;
              Tag tag{};
              kernel::loop::fold<4, 1>(
                tag, tkv,
                [&](auto block_tag, nint_t kj, auto active, auto& maximum_v, const auto& scale_v,
                    const auto& negative_infinity) VECOPS_INLINE_LAMBDA {
                  const nint_t active_count = std::min<nint_t>(vec::size(block_tag), tkv - kj);
                  auto value = vec::load(block_tag, coarse + qi * tkv + kj, active, vec::opt::zero);
                  value = vec::mul(block_tag, value, scale_v);
                  value =
                    vec::blend(block_tag, negative_infinity, vec::mwhilelt(block_tag, nint_t{0}, active_count), value);
                  vec::store(block_tag, coarse + qi * tkv + kj, value, active);
                  maximum_v = vec::max(maximum_v, value);
                },
                kernel::loop::reduce_max(maximum), kernel::loop::invariant(scale),
                kernel::loop::invariant(-std::numeric_limits<Score>::infinity()));
              any_valid = true;
            } else {
              std::fill(coarse + qi * tkv, coarse + (qi + 1) * tkv, -std::numeric_limits<Score>::infinity());
            }
          } else {
            for (nint_t kj = 0; kj < tkv; ++kj) {
              const nint_t k_begin = kj * Config::key_value_block;
              const nint_t k_rows = std::min<nint_t>(Config::key_value_block, lkv - k_begin);
              Score bias_sum{};
              nint_t pair_count = 0;
              for (nint_t qr = 0; qr < q_rows; ++qr) {
                const nint_t q_index = q_begin + qr;
                if constexpr (!tensor::is_nullopt_v<decltype(uqm)>) {
                  if (query_active[static_cast<std::size_t>(qr)] == Score{})
                    continue;
                }
                using Tag = vec::ScalableTag<Score, 0>;
                Tag tag{};
                Score row_bias_sum{};
                nint_t row_pair_count = 0;
                kernel::loop::fold<4, traversal_tail_factor>(
                  tag, k_rows,
                  [&](auto block_tag, nint_t kr, auto active, auto& bias_sum_v) VECOPS_INLINE_LAMBDA {
                    const nint_t active_count = std::min(vec::size(block_tag), k_rows - kr);
                    auto keep = attention_details::attention_keep_mask<Config::causal_mode>(
                      block_tag, active, active_count, true, q_index, k_begin, kr, lq, lkv, ukm, uam);
                    row_pair_count += vec::mask_count(block_tag, keep);
                    if constexpr (!tensor::is_nullopt_v<decltype(ubias)>) {
                      auto bias_v = ubias.load(block_tag, tensor::coord(q_index, k_begin + kr), tensor::axis<1>, active,
                                               vec::opt::zero);
                      bias_sum_v = vec::add(bias_sum_v, vec::blend(block_tag, vec::zeros(block_tag), keep, bias_v));
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
          }

          Score normalizer{};
          if (any_valid) {
            using Tag = vec::ScalableTag<Score, 0>;
            Tag tag{};
            kernel::loop::fold<4, 1>(
              tag, tkv,
              [&](auto block_tag, nint_t kj, auto active, auto& normalizer_v, const auto& maximum_v,
                  const auto& negative_infinity) VECOPS_INLINE_LAMBDA {
                auto value = vec::load(block_tag, coarse + qi * tkv + kj, active, vec::opt::merge(negative_infinity));
                auto probability = softmax_details::exp_neg_estimate_safe<
                    Score, Config::exp_accuracy>(
                    block_tag, vec::sub(block_tag, value, maximum_v), active);
                normalizer_v = vec::add(normalizer_v, probability);
                vec::store(block_tag, coarse + qi * tkv + kj, probability, active);
              },
              kernel::loop::reduce_add(normalizer), kernel::loop::invariant(maximum),
              kernel::loop::invariant(-std::numeric_limits<Score>::infinity()));
            const Score reciprocal = Score{1} / normalizer;
            kernel::loop::fold<4, 1>(
              tag, tkv,
              [&](auto block_tag, nint_t kj, auto active, const auto& reciprocal_v) VECOPS_INLINE_LAMBDA {
                auto probability = vec::load(block_tag, coarse + qi * tkv + kj, active, vec::opt::zero);
                probability = vec::mul(block_tag, probability, reciprocal_v);
                vec::store(block_tag, coarse + qi * tkv + kj, probability, active);
              },
              kernel::loop::invariant(reciprocal));
          }
          std::fill(top_probability, top_probability + selected, Score{-1});
          std::fill(top_index, top_index + selected, int32_t{-1});
          if (normalizer > Score{}) {
            for (nint_t kj = 0; kj < tkv; ++kj) {
              const Score probability = coarse[qi * tkv + kj];
              nint_t position = 0;
              while (position < selected && (top_probability[position] > probability ||
                                             (top_probability[position] == probability && top_index[position] >= 0 &&
                                              top_index[position] < kj))) {
                ++position;
              }
              if (position == selected)
                continue;
              for (nint_t move = selected - 1; move > position; --move) {
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
            if (top_index[slot] < 0 || cdf >= config.static_probability ||
                top_probability[slot] < config.minimum_probability)
              break;
            cdf += top_probability[slot];
            index_data[qi * selected + slot] = top_index[slot];
            weight_data[qi * selected + slot] = Score{-1};
          }
          for (; slot < selected; ++slot) {
            if (top_index[slot] < 0 || cdf >= config.random_probability ||
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

/**
 * @brief Index-based Block Sparse (IBS) Attention.
 *
 * Index and weight are a compile-time pair: both are contiguous rank-two
 * Tensor buffers, or both are `tensor::nullopt`. In the omitted form the maps
 * are created by IBSAttentionIndexer in caller workspace. Each invocation
 * samples the dynamic candidates into a final selected-index sequence and
 * delegates its attention-computation stage to SparseFlashAttention. Keeping
 * base maps separate enables the index-reuse scheme described by IBS
 * Attention while resampling dynamic dependencies on every invocation.
 */
template <typename Config>
class IBSAttention {
  using Score = typename Config::ScoreType;
  using Indexer = IBSAttentionIndexer<Config>;
  using SparseOp = SparseFlashAttention<Config>;

public:
  using ResourceRequirements = typename SparseOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit IBSAttention(Config cfg)
    : config(std::move(cfg)) {
  }

  template <nint_t Parallelism, typename WorkerScratch>
  class PatternPrepared {
  public:
    using ResourceRequirements = typename IBSAttention::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(Config config, WorkerScratch scratch, nint_t workspace_bytes)
      : config_(std::move(config))
      , scratch_(std::move(scratch))
      , workspace_bytes_(workspace_bytes) {
    }

    template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
              tensor::OptionalTensorOf<int32_t, 2> Index, tensor::OptionalTensorOf<Score, 2> Weight,
              tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
              tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
              tensor::OutputOperandOf<2> Output, typename RNG>
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, const Q& q, const K& k, const V& v,
                                  const Index& index, const Weight& weight, const QueryMask& query_mask,
                                  const KeyMask& key_mask, const AttentionMask& attention_mask, const Bias& bias,
                                  const Output& out, Score scale, RNG& rng) const {
      IBSAttention operation{config_};
      VECOPS_CHECK(operation.required_workspace(q, k, v, index, weight, query_mask, key_mask, attention_mask, bias,
                                                out) <= workspace_bytes_,
                   "active IBS attention scratch exceeds its planning-pattern capacity");
      auto scratch = task.local(scratch_);
      kernel::WorkspaceView view{scratch.data(), workspace_bytes_};
      ExecutionSession execution{view};
      operation(execution, q, k, v, index, weight, query_mask, key_mask, attention_mask, bias, out, scale, rng);
    }

  private:
    Config config_;
    WorkerScratch scratch_;
    nint_t workspace_bytes_ = 0;
  };

  template <nint_t Parallelism, typename WorkspaceAuthority, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OptionalTensorOf<int32_t, 2> Index,
            tensor::OptionalTensorOf<Score, 2> Weight, tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask, tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias, tensor::OutputOperandOf<2> Output>
    requires(tensor::is_unbound_tensor_view_v<Q> && tensor::is_unbound_tensor_view_v<K> &&
             tensor::is_unbound_tensor_view_v<V> &&
             (tensor::is_nullopt_v<Index> || tensor::is_unbound_tensor_view_v<Index>) &&
             (tensor::is_nullopt_v<Weight> || tensor::is_unbound_tensor_view_v<Weight>) &&
             (tensor::is_nullopt_v<QueryMask> || tensor::is_unbound_tensor_view_v<QueryMask>) &&
             (tensor::is_nullopt_v<KeyMask> || tensor::is_unbound_tensor_view_v<KeyMask>) &&
             (tensor::is_nullopt_v<AttentionMask> || tensor::is_unbound_tensor_view_v<AttentionMask>) &&
             (tensor::is_nullopt_v<Bias> || tensor::is_unbound_tensor_view_v<Bias>) &&
             tensor::is_unbound_tensor_view_v<Output>)
  VECOPS_INLINE auto prepare(WorkspaceAuthority& workspace, std::string_view site_name, Q&& q, K&& k, V&& v,
                             Index&& index, Weight&& weight, QueryMask&& query_mask, KeyMask&& key_mask,
                             AttentionMask&& attention_mask, Bias&& bias, Output&& out) const {
    const nint_t bytes = required_workspace(q, k, v, index, weight, query_mask, key_mask, attention_mask, bias, out);
    auto scratch =
      workspace.template worker_tensor<std::byte, Parallelism>(site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<Parallelism, decltype(scratch)>{config, std::move(scratch), bytes};
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
            tensor::OptionalTensorOf<int32_t, 2> Index, tensor::OptionalTensorOf<Score, 2> Weight,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace(const Q& q, const K& k, const V& v, const Index& index, const Weight& weight,
                                          const QueryMask& query_mask, const KeyMask& key_mask,
                                          const AttentionMask& attention_mask, const Bias& bias,
                                          const Output& out) const {
    constexpr bool HasIndex = !tensor::is_nullopt_v<Index>;
    constexpr bool HasWeight = !tensor::is_nullopt_v<Weight>;
    static_assert(HasIndex == HasWeight, "IBS Attention index and weight must be both present or both omitted");
    if constexpr (HasIndex) {
      return required_sampled_workspace(q, k, v, index, weight, query_mask, key_mask, attention_mask, bias, out);
    } else {
      auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const auto lq_extent = tensor::size<0>(q_spec.input_layout());
      const auto tq_extent = ceil_div(lq_extent, Config::query_block_extent);
      const auto selected_blocks = selected_blocks_extent();
      auto map_shape = tensor::make_shape(tq_extent, selected_blocks);
      auto index_tensor = tensor::make_tensor(static_cast<int32_t*>(nullptr), map_shape);
      auto weight_tensor = tensor::make_tensor(static_cast<Score*>(nullptr), map_shape);
      Indexer indexer{config};
      const nint_t map_bytes = kernel::WorkspaceView::allocation_bytes<int32_t>(tq_extent * selected_blocks) +
                               kernel::WorkspaceView::allocation_bytes<Score>(tq_extent * selected_blocks);
      const nint_t indexer_bytes =
        indexer.required_workspace(q, k, query_mask, key_mask, attention_mask, bias, index_tensor, weight_tensor);
      const nint_t sampled_bytes = required_sampled_workspace(q, k, v, index_tensor, weight_tensor, query_mask,
                                                              key_mask, attention_mask, bias, out);
      return map_bytes + std::max(indexer_bytes, sampled_bytes);
    }
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace(const Q& q, const K& k, const V& v, const Output& out) const {
    return required_workspace(q, k, v, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt,
                              tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OptionalTensorOf<int32_t, 2> Index,
            tensor::OptionalTensorOf<Score, 2> Weight, tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask, tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias, tensor::OutputOperandOf<2> Output, typename RNG>
  VECOPS_INLINE void operator()(Scope& scope, const Q& q, const K& k, const V& v, const Index& index,
                                const Weight& weight, const QueryMask& query_mask, const KeyMask& key_mask,
                                const AttentionMask& attention_mask, const Bias& bias, const Output& out, Score scale,
                                RNG& rng) const {
    constexpr bool HasIndex = !tensor::is_nullopt_v<Index>;
    constexpr bool HasWeight = !tensor::is_nullopt_v<Weight>;
    static_assert(HasIndex == HasWeight, "IBS Attention index and weight must be both present or both omitted");
    if constexpr (HasIndex) {
      execute_sampled(scope, q, k, v, index, weight, query_mask, key_mask, attention_mask, bias, out, scale, rng);
    } else {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const auto tq_extent = ceil_div(tensor::size<0>(q_spec.input_layout()), Config::query_block_extent);
      const auto selected_blocks = selected_blocks_extent();
      auto map_shape = tensor::make_shape(tq_extent, selected_blocks);
      auto index_tensor = workspace.template allocate_tensor<int32_t>(map_shape);
      auto weight_tensor = workspace.template allocate_tensor<Score>(map_shape);
      Indexer indexer{config};
      indexer(scope, q, k, query_mask, key_mask, attention_mask, bias, index_tensor, weight_tensor, scale);
      execute_sampled(scope, q, k, v, index_tensor, weight_tensor, query_mask, key_mask, attention_mask, bias, out,
                      scale, rng);
      workspace.rewind(mark);
    }
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
            tensor::OptionalTensorOf<int32_t, 2> Index, tensor::OptionalTensorOf<Score, 2> Weight,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output, typename RNG>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const Q& q, const K& k, const V& v,
                                const Index& index, const Weight& weight, const QueryMask& query_mask,
                                const KeyMask& key_mask, const AttentionMask& attention_mask, const Bias& bias,
                                const Output& out, Score scale, RNG& rng) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, v, index, weight, query_mask, key_mask, attention_mask, bias, out, scale, rng);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
            tensor::OutputOperandOf<2> Output, typename RNG>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const Q& q, const K& k, const V& v, const Output& out,
                                Score scale, RNG& rng) const {
    (*this)(workspace, q, k, v, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, out, scale, rng);
  }

private:
  VECOPS_INLINE meta::Dynamic<1, 1> selected_blocks_extent() const {
    VECOPS_ASSERT(config.selected_blocks > 0, "IBS Attention selected block count must be positive");
    return meta::Dynamic<1, 1>{config.selected_blocks};
  }

  template <tensor::TensorOf<int32_t, 2> Index, tensor::TensorOf<Score, 2> Weight>
  VECOPS_INLINE void validate_base_maps(const Index& index, const Weight& weight) const {
    attention_details::validate_contiguous_buffer<int32_t, 2>(index, "IBS Attention index map must be contiguous");
    attention_details::validate_contiguous_buffer<Score, 2>(weight, "IBS Attention weight map must be contiguous");
    VECOPS_ASSERT(tensor::size<0>(index.layout()) == tensor::size<0>(weight.layout()) &&
                    tensor::size<1>(index.layout()) == tensor::size<1>(weight.layout()),
                  "IBS Attention index/weight shapes differ");
    VECOPS_ASSERT(config.random_blocks >= 0 && config.random_blocks <= tensor::size<1>(index.layout()),
                  "IBS Attention random block count is invalid");
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K, tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index, tensor::TensorOf<Score, 2> Weight,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_sampled_workspace(const Q& q, const K& k, const V& v, const Index& index,
                                                  const Weight& weight, const QueryMask& query_mask,
                                                  const KeyMask& key_mask, const AttentionMask& attention_mask,
                                                  const Bias& bias, const Output& out) const {
    validate_base_maps(index, weight);
    const nint_t elements = tensor::numel(index.layout());
    const auto capacity = tensor::size<1>(index.layout());
    const nint_t sampling_bytes = kernel::WorkspaceView::allocation_bytes<int32_t>(elements) +
                                  kernel::WorkspaceView::allocation_bytes<Score>(capacity) +
                                  kernel::WorkspaceView::allocation_bytes<int32_t>(capacity);
    auto selected_tensor =
      tensor::make_tensor(static_cast<int32_t*>(nullptr),
                          tensor::make_shape(tensor::size<0>(index.layout()), tensor::size<1>(index.layout())));
    return sampling_bytes + SparseOp{config}.required_workspace(q, k, v, selected_tensor, query_mask, key_mask,
                                                                attention_mask, bias, out);
  }

  template <execution::ExecutionScope Scope, tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::TensorOf<int32_t, 2> Index, tensor::TensorOf<Score, 2> Weight,
            tensor::OptionalInputOperandOf<1> QueryMask, tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask, tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output, typename RNG>
  VECOPS_INLINE void execute_sampled(Scope& scope, const Q& q, const K& k, const V& v, const Index& index,
                                     const Weight& weight, const QueryMask& query_mask, const KeyMask& key_mask,
                                     const AttentionMask& attention_mask, const Bias& bias, const Output& out,
                                     Score scale, RNG& rng) const {
    validate_base_maps(index, weight);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const auto rows = tensor::size<0>(index.layout());
    const auto capacity = tensor::size<1>(index.layout());
    const nint_t rows_count = rows;
    const nint_t capacity_count = capacity;
    auto selected_buffer = workspace.template allocate_tensor<int32_t>(tensor::make_shape(rows, capacity));
    auto dynamic_weight_buffer = workspace.template allocate_tensor<Score>(tensor::make_shape(capacity));
    auto dynamic_index_buffer = workspace.template allocate_tensor<int32_t>(tensor::make_shape(capacity));
    int32_t* selected = selected_buffer.data();
    Score* dynamic_weight = dynamic_weight_buffer.data();
    int32_t* dynamic_index = dynamic_index_buffer.data();
    const int32_t* index_data = index.data();
    const Score* weight_data = weight.data();
    for (nint_t row = 0; row < rows_count; ++row) {
      nint_t out_count = 0;
      nint_t dynamic_count = 0;
      Score dynamic_mass{};
      for (nint_t slot = 0; slot < capacity_count; ++slot) {
        const int32_t candidate = index_data[row * capacity_count + slot];
        const Score probability = weight_data[row * capacity_count + slot];
        if (candidate < 0)
          break;
        if (probability < Score{}) {
          selected[row * capacity_count + out_count++] = candidate;
        } else if (probability > Score{}) {
          dynamic_index[dynamic_count] = candidate;
          dynamic_weight[dynamic_count] = probability;
          dynamic_mass += probability;
          ++dynamic_count;
        }
      }
      const Score band =
        std::max(config.random_probability - config.static_probability, std::numeric_limits<Score>::epsilon());
      nint_t random_count = static_cast<nint_t>(
        std::ceil(static_cast<Score>(config.random_blocks) * std::min(dynamic_mass / band, Score{1})));
      random_count = std::min({random_count, dynamic_count, capacity_count - out_count});
      for (nint_t draw = 0; draw < random_count; ++draw) {
        Score remaining{};
        for (nint_t i = 0; i < dynamic_count; ++i)
          remaining += dynamic_weight[i];
        if (!(remaining > Score{}))
          break;
        std::uniform_real_distribution<Score> distribution(Score{}, remaining);
        const Score target = distribution(rng);
        Score cdf{};
        nint_t chosen = dynamic_count - 1;
        for (nint_t i = 0; i < dynamic_count; ++i) {
          cdf += dynamic_weight[i];
          if (target < cdf) {
            chosen = i;
            break;
          }
        }
        selected[row * capacity_count + out_count++] = dynamic_index[chosen];
        dynamic_weight[chosen] = Score{};
      }
      std::fill(selected + row * capacity_count + out_count, selected + (row + 1) * capacity_count, int32_t{-1});
    }
    auto selected_tensor = tensor::make_tensor(
      selected, tensor::make_shape(tensor::size<0>(index.layout()), tensor::size<1>(index.layout())));
    SparseOp{config}(scope, q, k, v, selected_tensor, query_mask, key_mask, attention_mask, bias, out, scale);
    workspace.rewind(mark);
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto ibs_attention_indexer(Config config) {
  return IBSAttentionIndexer<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto ibs_attention(Config config) {
  return IBSAttention<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_IBS_ATTENTION_H
