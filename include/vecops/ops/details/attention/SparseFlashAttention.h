// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_DETAILS_ATTENTION_SPARSE_FLASH_ATTENTION_H
#define VECOPS_OPS_DETAILS_ATTENTION_SPARSE_FLASH_ATTENTION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include "vecops/ops/details/attention/SDPA.h"
#include "vecops/util/Bitcast.h"

namespace vecops::ops {

// The online-softmax recurrence has one scalar rescale per query row.  Keep
// that unavoidable recurrence, but do not route its exponential through 920F
// libm: rare inputs can spend seconds in expf.  The reduced remainder is in
// [-ln(2)/2, ln(2)/2], where a short minimax-like Taylor polynomial is safe.
template <typename T>
VECOPS_INLINE T sparse_attention_exp(T value) {
  using Work = std::conditional_t<sizeof(T) <= sizeof(float), float, double>;
  const Work x = static_cast<Work>(value);
  constexpr Work Log2E = static_cast<Work>(1.4426950408889634073599L);
  constexpr Work Ln2 = static_cast<Work>(0.6931471805599453094172L);
  constexpr Work MinInput = sizeof(T) <= sizeof(float)
      ? static_cast<Work>(-87.336544750553032L)
      // The bit-built scale below intentionally flushes subnormals.  Keep
      // the exponent in the normal range rather than shifting a negative
      // unsigned exponent (which would be undefined for double).
      : static_cast<Work>(-708.3964185322641062L);
  if (!(x > MinInput)) return static_cast<T>(0);

  const Work scaled = x * Log2E;
  const int exponent = static_cast<int>(
      scaled + (scaled >= Work(0) ? Work(0.5) : Work(-0.5)));
  const Work remainder = x - static_cast<Work>(exponent) * Ln2;
  // Degree 10 keeps the double path within a few ulps on the reduced interval
  // while retaining the cheap degree-6 float path after Work narrowing.
  Work polynomial = Work(1.0 / 3628800.0);
  polynomial = Work(1.0 / 362880.0) + remainder * polynomial;
  polynomial = Work(1.0 / 40320.0) + remainder * polynomial;
  polynomial = Work(1.0 / 5040.0) + remainder * polynomial;
  polynomial = Work(1.0 / 720.0) + remainder * polynomial;
  polynomial = Work(1.0 / 120.0) + remainder * polynomial;
  polynomial = Work(1.0 / 24.0) + remainder * polynomial;
  polynomial = Work(1.0 / 6.0) + remainder * polynomial;
  polynomial = Work(0.5) + remainder * polynomial;
  polynomial = Work(1) + remainder * polynomial;
  polynomial = Work(1) + remainder * polynomial;
  if constexpr (sizeof(T) <= sizeof(float)) {
    const auto power_bits = static_cast<std::uint32_t>(exponent + 127) << 23;
    return static_cast<T>(polynomial * bitcast<float>(power_bits));
  } else {
    const auto power_bits = static_cast<std::uint64_t>(exponent + 1023) << 52;
    return static_cast<T>(polynomial * bitcast<double>(power_bits));
  }
}

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
 * The final selected-index map is an internal buffer: callers pass a
 * contiguous rank-two `tensor::Tensor` directly. InputSpec transforms and
 * DataAccess materialization are intentionally not accepted. IBS Attention
 * owns base index/weight maps and sampling; this primitive only consumes the
 * resulting selected block sequence.
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

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
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
    auto bias_spec = tensor::as_input_spec<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    return required_workspace_specs(
        q_spec, k_spec, v_spec, index, qm_spec, km_spec,
        am_spec, bias_spec, out_spec);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Index& index,
      const Output& out) const {
    return required_workspace(
        q, k, v, index, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
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
    auto bias_spec = tensor::as_input_spec<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          execute_specs(
              active, q_spec, k_spec, v_spec, index,
              qm_spec, km_spec, am_spec, bias_spec, out_spec, scale);
        });
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Index& index, const Output& out, Score scale) const {
    (*this)(scope, q, k, v, index, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
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

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::TensorOf<int32_t, 2> Index,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v, const Index& index,
      const Output& out, Score scale) const {
    (*this)(workspace, q, k, v, index,
            tensor::nullopt, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, out, scale);
  }

private:
  template <typename QSpec, typename KSpec, typename VSpec,
            tensor::TensorOf<int32_t, 2> Index,
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
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    VECOPS_ASSERT(
        tensor::size<0>(index.layout()) ==
            ceil_div(lq, static_cast<nint_t>(Config::query_block)) &&
        tensor::size<1>(index.layout()) > 0,
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
            tensor::TensorOf<int32_t, 2> Index,
            typename QMSpec, typename KMSpec,
            typename AMSpec, typename BiasSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_specs(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const Index& index, const QMSpec& query_mask,
      const KMSpec& key_mask, const AMSpec& attention_mask,
      const BiasSpec& bias, const OutSpec& out) const {
    validate_specs(
        q, k, v, index, query_mask, key_mask, attention_mask, bias, out);
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    const auto dv = tensor::size<1>(v.input_layout());
    const nint_t rows = std::min<nint_t>(Config::query_block, lq);
    const nint_t columns = std::min<nint_t>(Config::key_value_block, lkv);
    const auto row_extent = meta::dyn<1, 1, Config::query_block>(rows);
    const auto column_extent =
        meta::dyn<1, 1, Config::key_value_block>(columns);
    const auto dqk_extent = tensor::size<1>(q.input_layout());
    const auto dv_extent = tensor::size<1>(v.input_layout());
    auto q_block = tensor::narrow_view<0>(q, 0, row_extent);
    auto k_block = tensor::narrow_view<0>(k, 0, column_extent);
    auto v_block = tensor::narrow_view<0>(v, 0, column_extent);
    auto v_transposed = tensor::transpose_view<0, 1>(v_block);
    auto score_tensor = tensor::make_unbound_tensor<Score>(
        tensor::make_shape(row_extent, column_extent));
    auto probability_tensor = tensor::make_unbound_tensor<Probability>(
        tensor::make_shape(row_extent, column_extent));
    auto output_tensor = tensor::make_unbound_tensor<Score>(
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
        kernel::WorkspaceView::allocation_bytes<Score>(
            Config::query_block * Config::key_value_block) +
        kernel::WorkspaceView::allocation_bytes<Probability>(
            Config::query_block * Config::key_value_block) +
        kernel::WorkspaceView::allocation_bytes<Score>(
            Config::query_block * dv) +
        3 * kernel::WorkspaceView::allocation_bytes<Score>(Config::query_block) +
        tensor::required_workspace(
            query_mask, MaskPolicy{}) +
        tensor::required_workspace(
            key_mask, MaskPolicy{}) +
        tensor::required_workspace(
            attention_mask, MatrixPolicy{}) +
        tensor::required_workspace(
            bias, MatrixPolicy{}) +
        tensor::required_workspace(out, OutputPolicy{});
    return persistent + std::max(qk_bytes, pv_bytes);
  }

  template <execution::ExecutionScope Scope,
            typename QSpec, typename KSpec, typename VSpec,
            tensor::TensorOf<int32_t, 2> Index,
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
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    const auto dqk = tensor::size<1>(q.input_layout());
    const auto dv = tensor::size<1>(v.input_layout());
    const auto selected_capacity = tensor::size<1>(index.layout());
    auto score_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent,
                           Config::key_value_block_extent));
    auto probability_buffer = workspace.template allocate_tensor<Probability>(
        tensor::make_shape(Config::query_block_extent,
                           Config::key_value_block_extent));
    auto output_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent,
                           tensor::size<1>(v.input_layout())));
    auto row_max_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent));
    auto row_sum_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent));
    auto row_scale_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent));
    Score* scores = score_buffer.data();
    Probability* probabilities = probability_buffer.data();
    Score* output_acc = output_buffer.data();
    Score* row_max = row_max_buffer.data();
    Score* row_sum = row_sum_buffer.data();
    Score* row_scale = row_scale_buffer.data();

    auto qm = tensor::bind(
        query_mask, MaskPolicy{}, workspace);
    auto km = tensor::bind(
        key_mask, MaskPolicy{}, workspace);
    auto am = tensor::bind(
        attention_mask, MatrixPolicy{}, workspace);
    auto b = tensor::bind(
        bias, MatrixPolicy{}, workspace);
    auto output_access = tensor::bind(out, OutputPolicy{}, workspace);

    const int32_t* index_data = index.data();
    tensor::with_optional_unordered_access(
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
                      Score,
                      static_cast<std::size_t>(Config::query_block)>
                      query_active{};
                  attention_details::load_activity_segment<Score>(
                      uqm, begin, rows, query_active.data());

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
                        scope, row_extent, column_extent, dqk,
                        q_block, k_block, score_tensor);

                    for (nint_t row = 0; row < rows; ++row) {
                      const nint_t q_index = begin + row;
                      const auto stats =
                          attention_details::decorate_score_row<
                              Config::causal_mode>(
                              scores + row * columns, columns,
                              tensor::is_nullopt_v<decltype(uqm)> ||
                                  query_active[static_cast<std::size_t>(row)] !=
                                      Score{},
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
                        alpha = sparse_attention_exp(
                            row_max[row] - block_max);
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
                            auto probability_v =
                                softmax_details::exp_neg_estimate_safe<
                                    Score, Config::exp_accuracy>(
                                    block_tag,
                                    vec::sub(block_tag, score_v, next_max_v),
                                    active);
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
                            tensor::size<1>(v.input_layout())));
                    if (accumulated) {
                      matmul_op(
                          scope, row_extent,
                          tensor::size<1>(v.input_layout()),
                          column_extent, probability_tensor, v_transposed,
                          tensor::input<Score>(output_tensor),
                          tensor::output<Score>(output_tensor));
                    } else {
                      matmul_op(
                          scope, row_extent,
                          tensor::size<1>(v.input_layout()),
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
 * @brief Exact scaled dot-product attention with compile-time strategy selection.
 *
 * `streaming` traverses every K/V block online with `O(Bq*Bkv)` score storage;
 * `materialized` stores `Bq*Lkv` scores and normally wins for decode and short
 * rows. `automatic` chooses once per invocation from tensor dimensions, so no
 * strategy branch appears in the score loops. Optional operands remain
 * compile-time choices in all strategies.
 */
template <typename Config>
class ScaledDotProductAttention {
  using SparseOp = SparseFlashAttention<Config>;
  using MaterializedOp = attention_details::MaterializedSDPA<Config>;
  using Score = typename Config::ScoreType;

public:
  using ResourceRequirements = typename SparseOp::ResourceRequirements;
  const Config config;

  VECOPS_INLINE constexpr explicit ScaledDotProductAttention(Config cfg)
      : config(std::move(cfg)) {}

  /**
   * Address-independent SDPA plan with one private scratch replica per lane.
   * Operand patterns retain layout/access types only. Runtime calls rebind
   * their actual Specs, so stateful transforms never leak into a cached plan.
   */
  template <nint_t Parallelism, typename QPattern, typename KPattern,
            typename VPattern, typename QueryMaskPattern,
            typename KeyMaskPattern, typename AttentionMaskPattern,
            typename BiasPattern, typename OutputPattern,
            typename WorkerScratch>
  class PatternPrepared {
  public:
    static_assert(Parallelism > 0,
                  "prepared attention parallelism must be positive");
    using ResourceRequirements = typename ScaledDotProductAttention::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(
        Config config, QPattern q, KPattern k, VPattern v,
        QueryMaskPattern query_mask, KeyMaskPattern key_mask,
        AttentionMaskPattern attention_mask, BiasPattern bias,
        OutputPattern out, WorkerScratch worker_scratch,
        nint_t workspace_bytes, bool materialized)
      : config_(std::move(config))
      , q_(std::move(q))
      , k_(std::move(k))
      , v_(std::move(v))
      , query_mask_(std::move(query_mask))
      , key_mask_(std::move(key_mask))
      , attention_mask_(std::move(attention_mask))
      , bias_(std::move(bias))
      , out_(std::move(out))
      , worker_scratch_(std::move(worker_scratch))
      , workspace_bytes_(workspace_bytes)
      , materialized_(materialized) {}

    template <tensor::InputOperandOf<2> Q,
              tensor::InputOperandOf<2> K,
              tensor::InputOperandOf<2> V,
              tensor::OptionalInputOperandOf<1> QueryMask,
              tensor::OptionalInputOperandOf<1> KeyMask,
              tensor::OptionalInputOperandOf<2> AttentionMask,
              tensor::OptionalInputOperandOf<2> Bias,
              tensor::OutputOperandOf<2> Output>
      requires (tensor::is_bound_tensor_view_v<Q> &&
                tensor::is_bound_tensor_view_v<K> &&
                tensor::is_bound_tensor_view_v<V> &&
                (tensor::is_nullopt_v<QueryMask> ||
                 tensor::is_bound_tensor_view_v<QueryMask>) &&
                (tensor::is_nullopt_v<KeyMask> ||
                 tensor::is_bound_tensor_view_v<KeyMask>) &&
                (tensor::is_nullopt_v<AttentionMask> ||
                 tensor::is_bound_tensor_view_v<AttentionMask>) &&
                (tensor::is_nullopt_v<Bias> ||
                 tensor::is_bound_tensor_view_v<Bias>) &&
                tensor::is_bound_tensor_view_v<Output>)
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        Q&& q, K&& k, V&& v,
        QueryMask&& query_mask, KeyMask&& key_mask,
        AttentionMask&& attention_mask, Bias&& bias,
        Output&& out, Score scale) const {
      auto active_q = tensor::as_input_spec<typename Config::Atom::TA>(
          std::forward<Q>(q));
      auto active_k = tensor::as_input_spec<typename Config::Atom::TB>(
          std::forward<K>(k));
      auto active_v = tensor::as_input_spec<typename Config::Atom::TB>(
          std::forward<V>(v));
      auto active_query_mask = attention_details::as_mask_input<Score>(
          std::forward<QueryMask>(query_mask));
      auto active_key_mask = attention_details::as_mask_input<Score>(
          std::forward<KeyMask>(key_mask));
      auto active_attention_mask = attention_details::as_mask_input<Score>(
          std::forward<AttentionMask>(attention_mask));
      auto active_bias = tensor::as_input_spec<Score>(
          std::forward<Bias>(bias));
      auto active_out = tensor::as_output_spec<Score>(
          std::forward<Output>(out));

      auto rebound_q = tensor::rebind(q_, std::move(active_q));
      auto rebound_k = tensor::rebind(k_, std::move(active_k));
      auto rebound_v = tensor::rebind(v_, std::move(active_v));
      auto rebound_query_mask = rebind_optional(
          query_mask_, std::move(active_query_mask));
      auto rebound_key_mask = rebind_optional(
          key_mask_, std::move(active_key_mask));
      auto rebound_attention_mask = rebind_optional(
          attention_mask_, std::move(active_attention_mask));
      auto rebound_bias = rebind_optional(
          bias_, std::move(active_bias));
      auto rebound_out = tensor::rebind(out_, std::move(active_out));

      ScaledDotProductAttention operation{config_};
      VECOPS_CHECK(
          operation.required_workspace_with_strategy(
              materialized_,
              rebound_q, rebound_k, rebound_v,
              rebound_query_mask, rebound_key_mask,
              rebound_attention_mask, rebound_bias, rebound_out) <=
              workspace_bytes_,
          "active attention scratch exceeds its planning-pattern capacity");
      auto scratch = task.local(worker_scratch_);
      kernel::WorkspaceView workspace{scratch.data(), workspace_bytes_};
      ExecutionSession execution{workspace};
      operation.execute_with_strategy(
          execution, materialized_, rebound_q, rebound_k, rebound_v,
          rebound_query_mask, rebound_key_mask,
          rebound_attention_mask, rebound_bias, rebound_out, scale);
    }

    template <tensor::InputOperandOf<2> Q,
              tensor::InputOperandOf<2> K,
              tensor::InputOperandOf<2> V,
              tensor::OutputOperandOf<2> Output>
      requires (tensor::is_bound_tensor_view_v<Q> &&
                tensor::is_bound_tensor_view_v<K> &&
                tensor::is_bound_tensor_view_v<V> &&
                tensor::is_bound_tensor_view_v<Output>)
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        Q&& q, K&& k, V&& v, Output&& out, Score scale) const {
      (*this)(task, std::forward<Q>(q), std::forward<K>(k),
              std::forward<V>(v), tensor::nullopt, tensor::nullopt,
              tensor::nullopt, tensor::nullopt,
              std::forward<Output>(out), scale);
    }

  private:
    template <typename Pattern, typename Actual>
    VECOPS_INLINE static auto rebind_optional(
        const Pattern& pattern, Actual&& actual) {
      if constexpr (tensor::is_nullopt_v<Pattern>) {
        static_assert(tensor::is_nullopt_v<Actual>,
                      "an omitted attention pattern requires an omitted runtime operand");
        (void)pattern;
        (void)actual;
        return tensor::nullopt;
      } else {
        static_assert(!tensor::is_nullopt_v<Actual>,
                      "a present attention pattern requires a runtime operand");
        return tensor::rebind(pattern, std::forward<Actual>(actual));
      }
    }

    Config config_;
    QPattern q_;
    KPattern k_;
    VPattern v_;
    QueryMaskPattern query_mask_;
    KeyMaskPattern key_mask_;
    AttentionMaskPattern attention_mask_;
    BiasPattern bias_;
    OutputPattern out_;
    WorkerScratch worker_scratch_;
    nint_t workspace_bytes_ = 0;
    bool materialized_ = true;
  };

  /** Prepare the complete optional-operand SDPA contract from patterns. */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::InputOperandOf<2> Q,
            tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
    requires (tensor::is_unbound_tensor_view_v<Q> &&
              tensor::is_unbound_tensor_view_v<K> &&
              tensor::is_unbound_tensor_view_v<V> &&
              (tensor::is_nullopt_v<QueryMask> ||
               tensor::is_unbound_tensor_view_v<QueryMask>) &&
              (tensor::is_nullopt_v<KeyMask> ||
               tensor::is_unbound_tensor_view_v<KeyMask>) &&
              (tensor::is_nullopt_v<AttentionMask> ||
               tensor::is_unbound_tensor_view_v<AttentionMask>) &&
              (tensor::is_nullopt_v<Bias> ||
               tensor::is_unbound_tensor_view_v<Bias>) &&
              tensor::is_unbound_tensor_view_v<Output> &&
              requires(WorkspaceAuthority& authority, nint_t bytes) {
                authority.template worker_tensor<std::byte, Parallelism>(
                    std::string_view{},
                    tensor::make_shape(meta::Any{bytes}));
              })
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& workspace, std::string_view site_name,
      Q&& q, K&& k, V&& v,
      QueryMask&& query_mask, KeyMask&& key_mask,
      AttentionMask&& attention_mask, Bias&& bias, Output&& out) const {
    auto q_pattern = tensor::as_input_spec<typename Config::Atom::TA>(
        std::forward<Q>(q));
    auto k_pattern = tensor::as_input_spec<typename Config::Atom::TB>(
        std::forward<K>(k));
    auto v_pattern = tensor::as_input_spec<typename Config::Atom::TB>(
        std::forward<V>(v));
    auto query_mask_pattern = attention_details::as_mask_input<Score>(
        std::forward<QueryMask>(query_mask));
    auto key_mask_pattern = attention_details::as_mask_input<Score>(
        std::forward<KeyMask>(key_mask));
    auto attention_mask_pattern = attention_details::as_mask_input<Score>(
        std::forward<AttentionMask>(attention_mask));
    auto bias_pattern = tensor::as_input_spec<Score>(
        std::forward<Bias>(bias));
    auto out_pattern = tensor::as_output_spec<Score>(
        std::forward<Output>(out));
    const bool materialized = use_materialized(
        q_pattern, k_pattern, v_pattern);
    const nint_t bytes = required_workspace_with_strategy(
        materialized,
        q_pattern, k_pattern, v_pattern,
        query_mask_pattern, key_mask_pattern,
        attention_mask_pattern, bias_pattern, out_pattern);
    auto scratch = workspace.template worker_tensor<std::byte, Parallelism>(
        site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<
        Parallelism, decltype(q_pattern), decltype(k_pattern),
        decltype(v_pattern), decltype(query_mask_pattern),
        decltype(key_mask_pattern), decltype(attention_mask_pattern),
        decltype(bias_pattern), decltype(out_pattern), decltype(scratch)>{
          config, std::move(q_pattern), std::move(k_pattern),
          std::move(v_pattern), std::move(query_mask_pattern),
          std::move(key_mask_pattern), std::move(attention_mask_pattern),
          std::move(bias_pattern), std::move(out_pattern),
          std::move(scratch), bytes, materialized};
  }

  /** Prepare SDPA with all optional mask and bias operands omitted. */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::InputOperandOf<2> Q,
            tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OutputOperandOf<2> Output>
    requires (tensor::is_unbound_tensor_view_v<Q> &&
              tensor::is_unbound_tensor_view_v<K> &&
              tensor::is_unbound_tensor_view_v<V> &&
              tensor::is_unbound_tensor_view_v<Output>)
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& workspace, std::string_view site_name,
      Q&& q, K&& k, V&& v, Output&& out) const {
    return this->template prepare<Parallelism>(
        workspace, site_name,
        std::forward<Q>(q), std::forward<K>(k), std::forward<V>(v),
        tensor::nullopt, tensor::nullopt, tensor::nullopt, tensor::nullopt,
        std::forward<Output>(out));
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out) const {
    return required_workspace_with_strategy(
        use_materialized(q, k, v), q, k, v, query_mask, key_mask,
        attention_mask, bias, out);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace(
      const Q& q, const K& k, const V& v, const Output& out) const {
    return required_workspace(
        q, k, v, tensor::nullopt, tensor::nullopt,
        tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    execute_with_strategy(
        scope, use_materialized(q, k, v), q, k, v,
        query_mask, key_mask, attention_mask, bias, out, scale);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      Scope& scope, const Q& q, const K& k, const V& v,
      const Output& out, Score scale) const {
    (*this)(scope, q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
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

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      const Q& q, const K& k, const V& v,
      const Output& out, Score scale) const {
    (*this)(workspace, q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

private:
  template <tensor::InputOperandOf<2> Q,
            tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE nint_t required_workspace_with_strategy(
      bool materialized, const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out) const {
    if (materialized) {
      return MaterializedOp{config}.required_workspace(
          q, k, v, query_mask, key_mask, attention_mask, bias, out);
    }
    const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
    const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
    const auto tq = ceil_div(
        tensor::size<0>(q_spec.input_layout()),
        Config::query_block_extent);
    const auto tkv = ceil_div(
        tensor::size<0>(k_spec.input_layout()),
        Config::key_value_block_extent);
    auto index_tensor = tensor::make_unbound_tensor<int32_t>(
        tensor::make_shape(tq, tkv));
    SparseOp sparse{config};
    return kernel::WorkspaceView::allocation_bytes<int32_t>(
               static_cast<nint_t>(tq * tkv)) +
        sparse.required_workspace(
            q, k, v, index_tensor, query_mask, key_mask,
            attention_mask, bias, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperandOf<2> Q,
            tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void execute_with_strategy(
      Scope& scope, bool materialized,
      const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    if (materialized) {
      MaterializedOp{config}(
          scope, q, k, v, query_mask, key_mask,
          attention_mask, bias, out, scale);
      return;
    }
    const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
    const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
    const auto tq = ceil_div(
        tensor::size<0>(q_spec.input_layout()),
        Config::query_block_extent);
    const auto tkv = ceil_div(
        tensor::size<0>(k_spec.input_layout()),
        Config::key_value_block_extent);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    auto index_tensor = workspace.template allocate_tensor<int32_t>(
        tensor::make_shape(tq, tkv));
    int32_t* index = index_tensor.data();
    const auto tq_count = tq;
    const auto tkv_count = tkv;
    for (nint_t row = 0; row < tq_count; ++row) {
      for (nint_t i = 0; i < tkv_count; ++i)
        index[row * tkv_count + i] = static_cast<int32_t>(i);
    }
    SparseOp sparse{config};
    sparse(
        scope, q, k, v, index_tensor, query_mask, key_mask,
        attention_mask, bias, out, scale);
    workspace.rewind(mark);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V>
  VECOPS_INLINE static bool use_materialized(
      const Q& q, const K& k, const V& v) {
    if constexpr (Config::strategy == SDPAStrategy::materialized) {
      return true;
    } else if constexpr (Config::strategy == SDPAStrategy::streaming) {
      return false;
    } else {
      const auto k_spec = tensor::as_input_spec<typename Config::Atom::TB>(k);
      const auto lkv = tensor::size<0>(k_spec.input_layout());
      const auto q_spec = tensor::as_input_spec<typename Config::Atom::TA>(q);
      const auto lq = tensor::size<0>(q_spec.input_layout());
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

  template <meta::ValueInput Tq, meta::ValueInput Tkv>
  VECOPS_INLINE static auto make_index_tensor(
      int32_t* data, Tq tq, Tkv tkv) {
    return tensor::make_tensor(
        data, tensor::make_shape(tq, tkv));
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto sparse_flash_attention(Config config) {
  return SparseFlashAttention<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto scaled_dot_product_attention(Config config) {
  return ScaledDotProductAttention<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_SPARSE_FLASH_ATTENTION_H
