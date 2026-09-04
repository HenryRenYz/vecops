#ifndef VECOPS_OPS_DETAILS_ATTENTION_SPARSE_H
#define VECOPS_OPS_DETAILS_ATTENTION_SPARSE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "vecops/ops/details/attention/Operation.h"

namespace vecops::ops {

/**
 * @brief Online block-sparse scaled dot-product attention for one head.
 *
 * The index map has shape `[ceil_div(Lq, Bq), S]`; every nonnegative entry
 * selects one `Bkv`-wide K/V block and `-1` terminates the row. The final K/V
 * block and final query block may be partial, so arbitrary sequence lengths
 * are supported. Selected block order affects only floating-point rounding.
 * The running row maximum/sum and output-rescaling recurrence follows the
 * exact tiled online-softmax algorithm in FlashAttention, adapted here to
 * CPU Matmul tiles and caller-owned workspace.
 * The index map (and optional sampling-weight map) is an internal buffer:
 * callers pass a contiguous rank-two `tensor::Tensor` directly. InputSpec
 * transforms and DataAccess materialization are intentionally not accepted.
 * @see https://arxiv.org/abs/2205.14135
 */
template <typename Config>
class SparseFlashAttention {
  using MatmulOp = Matmul<typename Config::MatmulConfig>;
  using Atom = typename Config::Atom;
  using Score = typename Config::ScoreType;
  using Probability = typename Config::ProbabilityType;
  using MaskPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct>;
  using MatrixPolicy = tensor::InputAccessPolicy<
      1, 1, tensor::AccessPlan::direct>;
  using OutputPolicy = tensor::OutputAccessPolicy<
      1, tensor::AccessPlan::direct>;

public:
  using ResourceRequirements = typename MatmulOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit SparseFlashAttention(Config cfg)
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Index& index,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out) const {
    auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
    auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
    auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = attention_details::as_optional_input<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    return required_workspace_specs(
        q_spec, k_spec, v_spec, index, qm_spec, km_spec,
        am_spec, bias_spec, out_spec);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::AttentionBufferTensor<Score, 2> Weight,
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
    validate_sampling_layouts(index, weight);
    const nint_t elements = tensor::numel(index.layout());
    const nint_t selected_bytes =
        attention_details::aligned_bytes<int32_t>(elements) +
        attention_details::aligned_bytes<Score>(
            index.layout().shape()[1]) +
        attention_details::aligned_bytes<int32_t>(
            index.layout().shape()[1]);
    auto selected_tensor = tensor::make_tensor(
        static_cast<int32_t*>(nullptr), tensor::make_shape(
            tensor::size_value<0>(index.layout()),
            tensor::size_value<1>(index.layout())));
    return selected_bytes + required_workspace(
        q, k, v, selected_tensor, query_mask, key_mask,
        attention_mask, bias, out);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Index& index,
      const Output& out) const {
    return required_workspace(
        q, k, v, index, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Index& index, const QueryMask& query_mask,
      const KeyMask& key_mask, const AttentionMask& attention_mask,
      const Bias& bias, const Output& out, Score scale) const {
    auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
    auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
    auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = attention_details::as_optional_input<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          execute_specs(
              active, q_spec, k_spec, v_spec, index,
              qm_spec, km_spec, am_spec, bias_spec, out_spec, scale);
        });
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::AttentionBufferTensor<Score, 2> Weight,
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
    validate_sampling_layouts(index, weight);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const nint_t rows = index.layout().shape()[0];
    const nint_t capacity = index.layout().shape()[1];
    int32_t* selected = attention_details::allocate_aligned<int32_t>(
        workspace, rows * capacity);
    Score* dynamic_weight = attention_details::allocate_aligned<Score>(
        workspace, capacity);
    int32_t* dynamic_index = attention_details::allocate_aligned<int32_t>(
        workspace, capacity);
    const int32_t* index_data = index.data();
    const Score* weight_data = weight.data();
    for (nint_t row = 0; row < rows; ++row) {
      nint_t out_count = 0;
      nint_t dynamic_count = 0;
      Score dynamic_mass{};
      for (nint_t slot = 0; slot < capacity; ++slot) {
        const int32_t candidate = index_data[row * capacity + slot];
        const Score probability = weight_data[row * capacity + slot];
        if (candidate < 0) break;
        if (probability < Score{}) {
          selected[row * capacity + out_count++] = candidate;
        } else if (probability > Score{}) {
          dynamic_index[dynamic_count] = candidate;
          dynamic_weight[dynamic_count] = probability;
          dynamic_mass += probability;
          ++dynamic_count;
        }
      }
      const Score band = std::max(
          config.random_probability - config.static_probability,
          std::numeric_limits<Score>::epsilon());
      nint_t random_count = static_cast<nint_t>(std::ceil(
          static_cast<Score>(config.random_blocks) *
          std::min(dynamic_mass / band, Score{1})));
      random_count = std::min(
          {random_count, dynamic_count, capacity - out_count});
      for (nint_t draw = 0; draw < random_count; ++draw) {
        Score remaining{};
        for (nint_t i = 0; i < dynamic_count; ++i)
          remaining += dynamic_weight[i];
        if (!(remaining > Score{})) break;
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
        selected[row * capacity + out_count++] = dynamic_index[chosen];
        dynamic_weight[chosen] = Score{};
      }
      std::fill(
          selected + row * capacity + out_count,
          selected + (row + 1) * capacity, int32_t{-1});
    }
    auto selected_tensor = tensor::make_tensor(
        selected, tensor::make_shape(
            tensor::size_value<0>(index.layout()),
            tensor::size_value<1>(index.layout())));
    (*this)(scope, q, k, v, selected_tensor,
            query_mask, key_mask, attention_mask, bias, out, scale);
    workspace.rewind(mark);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::AttentionBufferTensor<Score, 2> Weight,
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

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Index& index, const Output& out, Score scale) const {
    (*this)(scope, q, k, v, index, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v, const Index& index,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, v, index, query_mask, key_mask,
            attention_mask, bias, out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v, const Index& index,
      const Output& out, Score scale) const {
    (*this)(workspace, q, k, v, index,
            tensor::nullopt, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, out, scale);
  }

private:
  template <attention_details::AttentionBufferTensor<int32_t, 2> Index,
            attention_details::AttentionBufferTensor<Score, 2> Weight>
  VECOPS_INLINE void validate_sampling_layouts(
      const Index& index, const Weight& weight) const {
    attention_details::validate_contiguous_buffer<int32_t, 2>(
        index, "sparse-attention index map must be contiguous");
    attention_details::validate_contiguous_buffer<Score, 2>(
        weight, "sparse-attention weight map must be contiguous");
    VECOPS_ASSERT(index.layout().shape()[0] ==
                      weight.layout().shape()[0] &&
                  index.layout().shape()[1] ==
                      weight.layout().shape()[1],
                  "sparse-attention index/weight shapes differ");
    VECOPS_ASSERT(config.random_blocks >= 0 &&
                  config.random_blocks <= index.layout().shape()[1],
                  "sparse-attention random block count is invalid");
  }

  template <typename QSpec, typename KSpec, typename VSpec,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            typename QMSpec, typename KMSpec,
            typename AMSpec, typename BiasSpec, typename OutSpec>
  VECOPS_INLINE void validate_specs(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const Index& index, const QMSpec& query_mask,
      const KMSpec& key_mask, const AMSpec& attention_mask,
      const BiasSpec& bias, const OutSpec& out) const {
    attention_details::validate_dense_layouts(
        q.input_layout(), k.input_layout(), v.input_layout(),
        out.output_layout());
    attention_details::validate_contiguous_buffer<int32_t, 2>(
        index, "sparse-attention index map must be contiguous");
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    VECOPS_ASSERT(
        index.layout().shape()[0] ==
            ceil_div(lq, static_cast<nint_t>(Config::query_block)) &&
        index.layout().shape()[1] > 0,
        "sparse-attention index-map shape mismatch");
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

  template <typename QSpec, typename KSpec, typename VSpec,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            typename QMSpec, typename KMSpec,
            typename AMSpec, typename BiasSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_specs(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const Index& index, const QMSpec& query_mask,
      const KMSpec& key_mask, const AMSpec& attention_mask,
      const BiasSpec& bias, const OutSpec& out) const {
    validate_specs(
        q, k, v, index, query_mask, key_mask, attention_mask, bias, out);
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t dv = v.input_layout().shape()[1];
    const nint_t rows = std::min<nint_t>(Config::query_block, lq);
    const nint_t columns = std::min<nint_t>(Config::key_value_block, lkv);
    const auto row_extent = meta::dyn<1, 1, Config::query_block>(rows);
    const auto column_extent =
        meta::dyn<1, 1, Config::key_value_block>(columns);
    const auto dqk_extent = tensor::size_value<1>(q.input_layout());
    const auto dv_extent = tensor::size_value<1>(v.input_layout());
    auto q_block = tensor::narrow_view<0>(q, 0, row_extent);
    auto k_block = tensor::narrow_view<0>(k, 0, column_extent);
    auto v_block = tensor::narrow_view<0>(v, 0, column_extent);
    auto v_transposed = tensor::transpose_view<0, 1>(v_block);
    auto score_tensor = tensor::make_tensor(
        static_cast<Score*>(nullptr),
        tensor::make_shape(row_extent, column_extent));
    auto probability_tensor = tensor::make_tensor(
        static_cast<Probability*>(nullptr),
        tensor::make_shape(row_extent, column_extent));
    auto output_tensor = tensor::make_tensor(
        static_cast<Score*>(nullptr),
        tensor::make_shape(row_extent, dv_extent));
    MatmulOp matmul_op{config.matmul};
    const nint_t qk_bytes = matmul_op.required_workspace(
        row_extent, column_extent, dqk_extent,
        q_block, k_block, score_tensor);
    const nint_t pv_bytes = matmul_op.required_workspace(
        row_extent, dv_extent, column_extent,
        probability_tensor, v_transposed,
        tensor::input<Score>(output_tensor),
        tensor::output<Score>(output_tensor));
    const nint_t persistent =
        attention_details::aligned_bytes<Score>(
            Config::query_block * Config::key_value_block) +
        attention_details::aligned_bytes<Probability>(
            Config::query_block * Config::key_value_block) +
        attention_details::aligned_bytes<Score>(
            Config::query_block * dv) +
        3 * attention_details::aligned_bytes<Score>(Config::query_block) +
        attention_details::optional_required_workspace(
            query_mask, MaskPolicy{}) +
        attention_details::optional_required_workspace(
            key_mask, MaskPolicy{}) +
        attention_details::optional_required_workspace(
            attention_mask, MatrixPolicy{}) +
        attention_details::optional_required_workspace(
            bias, MatrixPolicy{}) +
        tensor::required_workspace(out, OutputPolicy{});
    return persistent + std::max(qk_bytes, pv_bytes);
  }

  template <execution::ExecutionScope Scope,
            typename QSpec, typename KSpec, typename VSpec,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            typename QMSpec, typename KMSpec,
            typename AMSpec, typename BiasSpec, typename OutSpec>
  VECOPS_INLINE void execute_specs(
      Scope& scope, const QSpec& q, const KSpec& k, const VSpec& v,
      const Index& index, const QMSpec& query_mask,
      const KMSpec& key_mask, const AMSpec& attention_mask,
      const BiasSpec& bias, const OutSpec& out, Score scale) const {
    validate_specs(
        q, k, v, index, query_mask, key_mask, attention_mask, bias, out);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t dqk = q.input_layout().shape()[1];
    const nint_t dv = v.input_layout().shape()[1];
    const nint_t selected_capacity = index.layout().shape()[1];
    Score* scores = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block * Config::key_value_block);
    Probability* probabilities =
        attention_details::allocate_aligned<Probability>(
            workspace, Config::query_block * Config::key_value_block);
    Score* output_acc = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block * dv);
    Score* row_max = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block);
    Score* row_sum = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block);
    Score* row_scale = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block);

    auto qm = attention_details::bind_optional_input(
        query_mask, MaskPolicy{}, workspace);
    auto km = attention_details::bind_optional_input(
        key_mask, MaskPolicy{}, workspace);
    auto am = attention_details::bind_optional_input(
        attention_mask, MatrixPolicy{}, workspace);
    auto b = attention_details::bind_optional_input(
        bias, MatrixPolicy{}, workspace);
    auto output_access = tensor::bind(out, OutputPolicy{}, workspace);

    const int32_t* index_data = index.data();
    attention_details::with_optional_access(
          [&](auto uqm, auto ukm, auto uam, auto ubias) {
              tensor::with_unordered_access(
                  output_access, [&](auto uout) {
                MatmulOp matmul_op{config.matmul};
                for (nint_t block_row = 0,
                     begin = 0; begin < lq;
                     ++block_row, begin += Config::query_block) {
                  const nint_t rows = std::min<nint_t>(
                      Config::query_block, lq - begin);
                  const auto row_extent =
                      meta::dyn<1, 1, Config::query_block>(rows);
                  auto q_block = tensor::narrow_view<0>(
                      q, begin, row_extent);
                  std::fill(output_acc, output_acc + rows * dv, Score{});
                  std::fill(row_max, row_max + rows,
                            -std::numeric_limits<Score>::infinity());
                  std::fill(row_sum, row_sum + rows, Score{});
                  bool accumulated = false;
                  std::array<
                      bool,
                      static_cast<std::size_t>(Config::query_block)>
                      query_active{};
                  for (nint_t row = 0; row < rows; ++row) {
                    bool active = true;
                    if constexpr (!tensor::is_nullopt_v<decltype(uqm)>) {
                    active = uqm.load_scalar(
                        tensor::coord(begin + row)) != Score{};
                    }
                    query_active[static_cast<std::size_t>(row)] = active;
                  }

                  for (nint_t slot = 0; slot < selected_capacity; ++slot) {
                    const int32_t block_index =
                        index_data[block_row * selected_capacity + slot];
                    if (block_index < 0) break;
                    const nint_t key_begin =
                        static_cast<nint_t>(block_index) *
                        Config::key_value_block;
                    VECOPS_ASSERT(key_begin < lkv,
                                  "sparse-attention block index out of range");
                    const nint_t columns = std::min<nint_t>(
                        Config::key_value_block, lkv - key_begin);
                    const auto column_extent =
                        meta::dyn<1, 1, Config::key_value_block>(columns);
                    auto k_block = tensor::narrow_view<0>(
                        k, key_begin, column_extent);
                    auto v_block = tensor::narrow_view<0>(
                        v, key_begin, column_extent);
                    auto v_transposed = tensor::transpose_view<0, 1>(v_block);
                    auto score_tensor = tensor::make_tensor(
                        scores,
                        tensor::make_shape(row_extent, column_extent));
                    matmul_op(
                        scope, row_extent, column_extent, meta::Any{dqk},
                        q_block, k_block, score_tensor);

                    for (nint_t row = 0; row < rows; ++row) {
                      const nint_t q_index = begin + row;
                      const auto stats =
                          attention_details::decorate_score_row<
                              Config::causal_mode>(
                              scores + row * columns, columns,
                              query_active[static_cast<std::size_t>(row)],
                              q_index, key_begin,
                              lq, lkv, scale, ukm, uam, ubias);
                      const Score block_max = stats.maximum;
                      const bool block_valid = stats.any;
                      const Score next_max = std::max(row_max[row], block_max);
                      if (!block_valid ||
                          next_max == -std::numeric_limits<Score>::infinity()) {
                        row_scale[row] = Score{1};
                        std::fill(
                            probabilities + row * columns,
                            probabilities + (row + 1) * columns,
                            Probability{});
                        continue;
                      }
                      Score alpha = Score{1};
                      if (row_max[row] ==
                          -std::numeric_limits<Score>::infinity()) {
                        alpha = Score{};
                      } else if (block_max > row_max[row]) {
                        alpha = std::exp(row_max[row] - block_max);
                      }
                      row_scale[row] = alpha;
                      Score block_sum{};
                      using ScoreTag = vec::ScalableTag<Score, 0>;
                      ScoreTag score_tag{};
                      kernel::loop::fold<4, 1>(
                          score_tag, columns,
                          [&](auto block_tag, nint_t column, auto active,
                              auto& block_sum_v, const auto& next_max_v,
                              const auto& negative_infinity)
                              VECOPS_INLINE_LAMBDA {
                            auto score_v = vec::load(
                                block_tag,
                                scores + row * columns + column, active,
                                vec::opt::merge(negative_infinity));
                            auto probability_v = vec::exp_neg(
                                block_tag,
                                vec::sub(block_tag, score_v, next_max_v),
                                vec::opt::math::accuracy<
                                    Config::exp_accuracy>,
                                active, vec::opt::zero);
                            block_sum_v = vec::add(
                                block_sum_v, probability_v);
                            vec::store_convert(
                                block_tag,
                                probabilities + row * columns + column,
                                probability_v, active);
                          },
                          kernel::loop::reduce_add(block_sum),
                          kernel::loop::invariant(next_max),
                          kernel::loop::invariant(
                              -std::numeric_limits<Score>::infinity()));
                      row_sum[row] = alpha * row_sum[row] + block_sum;
                      row_max[row] = next_max;
                    }

                    if (accumulated) {
                      for (nint_t row = 0; row < rows; ++row) {
                        const Score alpha = row_scale[row];
                        if (alpha == Score{1} || row_sum[row] == Score{})
                          continue;
                        using Tag = vec::ScalableTag<Score, 0>;
                        Tag tag{};
                        kernel::loop::fold<4, 1>(
                            tag, dv,
                            [&](auto block_tag, nint_t d, auto active,
                                const auto& alpha_v) VECOPS_INLINE_LAMBDA {
                              auto value = vec::load(
                                  block_tag, output_acc + row * dv + d,
                                  active, vec::opt::zero);
                              value = vec::mul(block_tag, value, alpha_v);
                              vec::store(
                                  block_tag, output_acc + row * dv + d,
                                  value, active);
                            },
                            kernel::loop::invariant(alpha));
                      }
                    }
                    auto probability_tensor = tensor::make_tensor(
                        probabilities,
                        tensor::make_shape(row_extent, column_extent));
                    auto output_tensor = tensor::make_tensor(
                        output_acc,
                        tensor::make_shape(
                            row_extent,
                            tensor::size_value<1>(v.input_layout())));
                    if (accumulated) {
                      matmul_op(
                          scope, row_extent,
                          tensor::size_value<1>(v.input_layout()),
                          column_extent, probability_tensor, v_transposed,
                          tensor::input<Score>(output_tensor),
                          tensor::output<Score>(output_tensor));
                    } else {
                      matmul_op(
                          scope, row_extent,
                          tensor::size_value<1>(v.input_layout()),
                          column_extent, probability_tensor, v_transposed,
                          output_tensor);
                    }
                    accumulated = true;
                  }

                  using Tag = vec::ScalableTag<Score, 0>;
                  Tag tag{};
                  for (nint_t row = 0; row < rows; ++row) {
                    const Score reciprocal =
                        accumulated && row_sum[row] > Score{}
                        ? Score{1} / row_sum[row]
                        : Score{};
                    kernel::loop::fold<4, 1>(
                        tag, dv,
                        [&](auto block_tag, nint_t d, auto active,
                            const auto& reciprocal_v)
                            VECOPS_INLINE_LAMBDA {
                          auto value = vec::load(
                              block_tag, output_acc + row * dv + d,
                              active, vec::opt::zero);
                          value = vec::mul(
                              block_tag, value, reciprocal_v);
                          uout.store(
                              block_tag, tensor::coord(begin + row, d),
                              value, active);
                        },
                        kernel::loop::invariant(reciprocal));
                  }
                }
                uout.commit();
              });
          },
          qm, km, am, b);
    workspace.rewind(mark);
  }
};

/**
 * @brief Materialized block-sparse attention with an online fallback.
 *
 * For direct contiguous K/V Tensors, selected blocks are packed into compact
 * contiguous panels and evaluated by one materialized attention call per
 * query block. Duplicate indices remain duplicate rows in the packed panel,
 * preserving their probability multiplicity. General Specs, non-contiguous
 * tensors, optional masks/bias, and sampled maps retain the fully general
 * online implementation inherited from SparseFlashAttention.
 */
template <typename Config>
class SparseAttention : public SparseFlashAttention<Config> {
  using Base = SparseFlashAttention<Config>;
  using Atom = typename Config::Atom;
  using Score = typename Config::ScoreType;
  using DenseOp = DenseMaterializedAttention<Config>;

public:
  using ResourceRequirements = typename Base::ResourceRequirements;
  using Base::operator();
  using Base::required_workspace;

  const Config config;

  VECOPS_INLINE constexpr explicit SparseAttention(Config cfg)
      : Base(cfg), config(std::move(cfg)) {}

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Index& index,
      const Output& out) const {
    const nint_t online_bytes =
        Base::required_workspace(q, k, v, index, out);
    if constexpr (!direct_packable<K, V>()) {
      return online_bytes;
    } else {
      if (!k.is_contiguous() || !v.is_contiguous()) return online_bytes;
      using KElement = std::remove_const_t<
          typename std::remove_cvref_t<K>::ElementType>;
      using VElement = std::remove_const_t<
          typename std::remove_cvref_t<V>::ElementType>;
      auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
      auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
      auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
      auto out_spec = tensor::as_output_spec<Score>(out);
      const nint_t lq = q_spec.input_layout().shape()[0];
      const nint_t lkv = k_spec.input_layout().shape()[0];
      validate_pack_layouts(q_spec, k_spec, v_spec, index, out_spec);
      const auto extents = selected_extents(index, lkv);
      if (extents.minimum == 0) return online_bytes;
      const nint_t rows = std::min<nint_t>(Config::query_block, lq);
      const auto row_extent =
          meta::dyn<1, 1, Config::query_block>(rows);
      const auto selected_extent = meta::Any{extents.maximum};
      auto q_block = tensor::narrow_view<0>(q_spec, 0, row_extent);
      auto out_block = tensor::narrow_view<0>(out_spec, 0, row_extent);
      auto packed_k = tensor::make_tensor(
          static_cast<KElement*>(nullptr), tensor::make_shape(
              selected_extent, tensor::size_value<1>(k_spec.input_layout())));
      auto packed_v = tensor::make_tensor(
          static_cast<VElement*>(nullptr), tensor::make_shape(
              selected_extent, tensor::size_value<1>(v_spec.input_layout())));
      const nint_t pack_bytes =
          attention_details::aligned_bytes<KElement>(
              extents.maximum * k_spec.input_layout().shape()[1]) +
          attention_details::aligned_bytes<VElement>(
              extents.maximum * v_spec.input_layout().shape()[1]);
      DenseOp dense{config};
      return std::max(
          online_bytes,
          pack_bytes + dense.required_workspace(
              q_block, packed_k, packed_v, out_block));
    }
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Index& index, const Output& out, Score scale) const {
    if constexpr (!direct_packable<K, V>()) {
      Base::operator()(scope, q, k, v, index, out, scale);
    } else {
      if (!k.is_contiguous() || !v.is_contiguous()) {
        Base::operator()(scope, q, k, v, index, out, scale);
        return;
      }
      using KElement = std::remove_const_t<
          typename std::remove_cvref_t<K>::ElementType>;
      using VElement = std::remove_const_t<
          typename std::remove_cvref_t<V>::ElementType>;
      const auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
      const auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
      const auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
      const auto out_spec = tensor::as_output_spec<Score>(out);
      const nint_t lq = q_spec.input_layout().shape()[0];
      const nint_t lkv = k_spec.input_layout().shape()[0];
      validate_pack_layouts(q_spec, k_spec, v_spec, index, out_spec);
      const nint_t dqk = k_spec.input_layout().shape()[1];
      const nint_t dv = v_spec.input_layout().shape()[1];
      const nint_t capacity = index.layout().shape()[1];
      const auto extents = selected_extents(index, lkv);
      if (extents.minimum == 0) {
        Base::operator()(scope, q, k, v, index, out, scale);
        return;
      }
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      KElement* packed_k = attention_details::allocate_aligned<KElement>(
          workspace, extents.maximum * dqk);
      VElement* packed_v = attention_details::allocate_aligned<VElement>(
          workspace, extents.maximum * dv);
      const int32_t* index_data = index.data();
      DenseOp dense{config};
      for (nint_t block_row = 0, q_begin = 0; q_begin < lq;
           ++block_row, q_begin += Config::query_block) {
        nint_t selected_rows = 0;
        for (nint_t slot = 0; slot < capacity; ++slot) {
          const int32_t block_index =
              index_data[block_row * capacity + slot];
          if (block_index < 0) break;
          const nint_t k_begin =
              static_cast<nint_t>(block_index) * Config::key_value_block;
          VECOPS_ASSERT(
              k_begin < lkv,
              "sparse-attention block index out of range");
          const nint_t rows = std::min<nint_t>(
              Config::key_value_block, lkv - k_begin);
          std::memcpy(
              packed_k + selected_rows * dqk,
              k.data() + k_begin * dqk,
              static_cast<std::size_t>(rows * dqk) * sizeof(KElement));
          std::memcpy(
              packed_v + selected_rows * dv,
              v.data() + k_begin * dv,
              static_cast<std::size_t>(rows * dv) * sizeof(VElement));
          selected_rows += rows;
        }
        const nint_t q_rows = std::min<nint_t>(
            Config::query_block, lq - q_begin);
        const auto row_extent =
            meta::dyn<1, 1, Config::query_block>(q_rows);
        const auto selected_extent = meta::Any{selected_rows};
        auto q_block = tensor::narrow_view<0>(
            q_spec, q_begin, row_extent);
        auto out_block = tensor::narrow_view<0>(
            out_spec, q_begin, row_extent);
        auto packed_k_tensor = tensor::make_tensor(
            packed_k, tensor::make_shape(
                selected_extent,
                tensor::size_value<1>(k_spec.input_layout())));
        auto packed_v_tensor = tensor::make_tensor(
            packed_v, tensor::make_shape(
                selected_extent,
                tensor::size_value<1>(v_spec.input_layout())));
        dense(
            scope, q_block, packed_k_tensor, packed_v_tensor,
            out_block, scale);
      }
      workspace.rewind(mark);
    }
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const Index& index, const Output& out, Score scale) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, v, index, out, scale);
  }

private:
  struct SelectedExtents {
    nint_t minimum;
    nint_t maximum;
  };

  template <typename K, typename V>
  static consteval bool direct_packable() {
    if constexpr (Config::causal_mode != AttentionCausalMode::none)
      return false;
    using KType = std::remove_cvref_t<K>;
    using VType = std::remove_cvref_t<V>;
    if constexpr (!tensor::is_tensor_v<KType> ||
                  !tensor::is_tensor_v<VType>) {
      return false;
    } else {
      return std::same_as<
          std::remove_const_t<typename KType::ElementType>,
          typename Atom::TB>;
    }
  }

  template <typename QSpec, typename KSpec, typename VSpec,
            attention_details::AttentionBufferTensor<int32_t, 2> Index,
            typename OutSpec>
  VECOPS_INLINE static void validate_pack_layouts(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const Index& index, const OutSpec& out) {
    attention_details::validate_dense_layouts(
        q.input_layout(), k.input_layout(), v.input_layout(),
        out.output_layout());
    attention_details::validate_contiguous_buffer<int32_t, 2>(
        index, "sparse-attention index map must be contiguous");
    VECOPS_ASSERT(
        index.layout().shape()[0] == ceil_div(
            q.input_layout().shape()[0],
            static_cast<nint_t>(Config::query_block)) &&
        index.layout().shape()[1] > 0,
        "sparse-attention index-map shape mismatch");
  }

  template <attention_details::AttentionBufferTensor<int32_t, 2> Index>
  VECOPS_INLINE static SelectedExtents selected_extents(
      const Index& index, nint_t lkv) {
    attention_details::validate_contiguous_buffer<int32_t, 2>(
        index, "sparse-attention index map must be contiguous");
    const nint_t map_rows = index.layout().shape()[0];
    const nint_t capacity = index.layout().shape()[1];
    const int32_t* data = index.data();
    nint_t minimum = std::numeric_limits<nint_t>::max();
    nint_t maximum = 0;
    for (nint_t row = 0; row < map_rows; ++row) {
      nint_t selected = 0;
      for (nint_t slot = 0; slot < capacity; ++slot) {
        const int32_t block_index = data[row * capacity + slot];
        if (block_index < 0) break;
        const nint_t begin =
            static_cast<nint_t>(block_index) * Config::key_value_block;
        if (begin >= lkv) return SelectedExtents{0, 0};
        selected += std::min<nint_t>(
            Config::key_value_block, lkv - begin);
      }
      minimum = std::min(minimum, selected);
      maximum = std::max(maximum, selected);
    }
    return SelectedExtents{
        minimum == std::numeric_limits<nint_t>::max() ? 0 : minimum,
        maximum};
  }
};

/**
 * @brief Exact dense attention with compile-time strategy selection.
 *
 * `streaming` traverses every K/V block online with `O(Bq*Bkv)` score storage;
 * `materialized` stores `Bq*Lkv` scores and normally wins for decode and short
 * rows. `automatic` chooses once per invocation from tensor dimensions, so no
 * strategy branch appears in the score loops. Optional operands remain
 * compile-time choices in all strategies.
 */
template <typename Config>
class DenseAttention {
  using SparseOp = SparseFlashAttention<Config>;
  using MaterializedOp = DenseMaterializedAttention<Config>;
  using Score = typename Config::ScoreType;

public:
  using ResourceRequirements = typename SparseOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit DenseAttention(Config cfg)
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out) const {
    if (use_materialized(q, k, v)) {
      return MaterializedOp{config}.required_workspace(
          q, k, v, query_mask, key_mask, attention_mask, bias, out);
    }
    const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
    const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
    const nint_t tq = ceil_div(
        q_spec.input_layout().shape()[0],
        static_cast<nint_t>(Config::query_block));
    const nint_t tkv = ceil_div(
        k_spec.input_layout().shape()[0],
        static_cast<nint_t>(Config::key_value_block));
    auto index_tensor = make_index_tensor(nullptr, tq, tkv);
    SparseOp sparse{config};
    return attention_details::aligned_bytes<int32_t>(tq * tkv) +
        sparse.required_workspace(
            q, k, v, index_tensor, query_mask, key_mask,
            attention_mask, bias, out);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Output& out) const {
    return required_workspace(
        q, k, v, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    if (use_materialized(q, k, v)) {
      MaterializedOp{config}(
          scope, q, k, v, query_mask, key_mask,
          attention_mask, bias, out, scale);
      return;
    }
    const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
    const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
    const nint_t tq = ceil_div(
        q_spec.input_layout().shape()[0],
        static_cast<nint_t>(Config::query_block));
    const nint_t tkv = ceil_div(
        k_spec.input_layout().shape()[0],
        static_cast<nint_t>(Config::key_value_block));
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    int32_t* index = attention_details::allocate_aligned<int32_t>(
        workspace, tq * tkv);
    for (nint_t row = 0; row < tq; ++row)
      for (nint_t i = 0; i < tkv; ++i)
        index[row * tkv + i] = static_cast<int32_t>(i);
    auto index_tensor = make_index_tensor(index, tq, tkv);
    SparseOp sparse{config};
    sparse(
        scope, q, k, v, index_tensor, query_mask, key_mask,
        attention_mask, bias, out, scale);
    workspace.rewind(mark);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Output& out, Score scale) const {
    (*this)(scope, q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    ExecutionSession execution{workspace};
    (*this)(execution, q, k, v, query_mask, key_mask,
            attention_mask, bias, out, scale);
  }

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const Output& out, Score scale) const {
    (*this)(workspace, q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

private:
  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V>
  VECOPS_INLINE static bool use_materialized(
      const Q& q, const K& k, const V& v) {
    if constexpr (Config::strategy == AttentionStrategy::materialized) {
      return true;
    } else if constexpr (Config::strategy == AttentionStrategy::streaming) {
      return false;
    } else {
      const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
      const nint_t lkv = k_spec.input_layout().shape()[0];
      const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const nint_t lq = q_spec.input_layout().shape()[0];
      (void)v;
      // Materialization is per query panel, not for the whole score matrix.
      // It avoids one QK and one PV Matmul dispatch per K/V block and wins for
      // the measured AF3/LLM range. The crossover is backend-specific: AMX's
      // large cache and efficient wide panels favor materialization through
      // 8192 keys, while SME favors streaming above 2048 keys and for single-
      // query sub-512 decode (where panel setup dominates).
#if defined(ARCH_X86_FAMILY)
      (void)lq;
      return lkv <= 8192;
#elif defined(CPU_CAPABILITY_SVE)
      if (lq == 1 && lkv <= 512) return false;
      return lkv <= 2048;
#else
      (void)lq;
      return lkv <= 2048;
#endif
    }
  }

  VECOPS_INLINE static auto make_index_tensor(
      int32_t* data, nint_t tq, nint_t tkv) {
    return tensor::make_tensor(
        data, tensor::make_shape(meta::Any{tq}, meta::Any{tkv}));
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto sparse_flash_attention(Config config) {
  return SparseFlashAttention<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto sparse_attention(Config config) {
  return SparseAttention<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto dense_attention(Config config) {
  return DenseAttention<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_SPARSE_H
