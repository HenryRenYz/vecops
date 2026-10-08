// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_DETAILS_ATTENTION_SDPA_H
#define VECOPS_OPS_DETAILS_ATTENTION_SDPA_H

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Atom.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/details/attention/Common.h"

namespace vecops::ops {

using AttentionCausalMode = attention_details::CausalMode;

enum class SDPAStrategy {
  automatic,
  materialized,
  streaming,
};

/**
 * @brief Configuration shared by SDPA, Sparse FlashAttention, and IBS Attention.
 *
 * The Matmul configuration fixes the hardware atom and its input/accumulator
 * types. Tensor memory element types remain independent and are converted by
 * DataAccess. QueryBlock and KeyValueBlock are semantic sparse-block extents;
 * materialized SDPA also uses QueryBlock as its score-panel height.
 */
template <typename MatmulConfigT, int QueryBlock = 32,
          int KeyValueBlock = 32,
          AttentionCausalMode Causal = AttentionCausalMode::none,
          SDPAStrategy Strategy = SDPAStrategy::automatic,
          vec::Accuracy ExpAccuracy = vec::Accuracy::Fast>
struct AttentionConfig {
  static_assert(QueryBlock > 0 && KeyValueBlock > 0);
  using MatmulConfig = MatmulConfigT;
  using Atom = typename MatmulConfig::Atom;
  using ScoreType = typename Atom::TAcc;
  using ProbabilityType = typename Atom::TA;
  static_assert(matmul::Atom<Atom>);
  static_assert(std::is_floating_point_v<ScoreType>);
  static_assert(std::same_as<typename Atom::TA, typename Atom::TB>,
                "attention currently requires a symmetric floating Matmul atom");

  static constexpr int query_block = QueryBlock;
  static constexpr int key_value_block = KeyValueBlock;
  using QueryBlockExtent = meta::Const<QueryBlock>;
  using KeyValueBlockExtent = meta::Const<KeyValueBlock>;
  inline static constexpr QueryBlockExtent query_block_extent{};
  inline static constexpr KeyValueBlockExtent key_value_block_extent{};
  static constexpr AttentionCausalMode causal_mode = Causal;
  static constexpr SDPAStrategy strategy = Strategy;
  static constexpr vec::Accuracy exp_accuracy = ExpAccuracy;

  [[no_unique_address]] MatmulConfig matmul{};
  nint_t selected_blocks = 1;
  nint_t random_blocks = 0;
  ScoreType static_probability = ScoreType{1};
  ScoreType random_probability = ScoreType{1};
  ScoreType minimum_probability = ScoreType{};
};

namespace attention_details {

/**
 * @brief Exact scaled dot-product attention for one logical head.
 *
 * Q is `[Lq,Dqk]`, K is `[Lkv,Dqk]`, V is `[Lkv,Dv]`, and output is
 * `[Lq,Dv]`. Optional query/key keep masks and the optional two-dimensional
 * keep mask use nonzero as active. Bias is added after score scaling. Every
 * optional operand is either a Tensor/Spec or `tensor::nullopt`; omission is
 * therefore a compile-time property and introduces no hot-loop branch.
 *
 * This materialized implementation is the correctness baseline and the small
 * problem path. It materializes at most `QueryBlock * Lkv` scores rather than
 * the full `Lq * Lkv` matrix. The streaming implementation shares this public
 * contract and is selected internally by ScaledDotProductAttention.
 */
template <typename Config>
class MaterializedSDPA {
  using MatmulOp = Matmul<typename Config::MatmulConfig>;
  using Atom = typename Config::Atom;
  using Score = typename Config::ScoreType;
  using Probability = typename Config::ProbabilityType;
  using MaskPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct>;
  using MatrixPolicy = tensor::InputAccessPolicy<
      1, 1, tensor::AccessPlan::direct>;

public:
  using ResourceRequirements = typename MatmulOp::ResourceRequirements;

  const Config config;

  VECOPS_INLINE constexpr explicit MaterializedSDPA(Config cfg)
      : config(std::move(cfg)) {}

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
    auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
    auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
    auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = tensor::as_input_spec<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    return required_workspace_specs(
        q_spec, k_spec, v_spec, qm_spec, km_spec, am_spec, bias_spec,
        out_spec);
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
              active, q_spec, k_spec, v_spec, qm_spec, km_spec,
              am_spec, bias_spec, out_spec, scale);
        });
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

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V,
            tensor::OptionalInputOperandOf<1> QueryMask,
            tensor::OptionalInputOperandOf<1> KeyMask,
            tensor::OptionalInputOperandOf<2> AttentionMask,
            tensor::OptionalInputOperandOf<2> Bias,
            tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      const Q& q, const K& k, const V& v,
      const QueryMask& query_mask, const KeyMask& key_mask,
      const AttentionMask& attention_mask, const Bias& bias,
      const Output& out, Score scale) const {
    kernel::Workspace storage(required_workspace(
        q, k, v, query_mask, key_mask, attention_mask, bias, out));
    auto workspace = storage.view();
    (*this)(workspace, q, k, v, query_mask, key_mask,
            attention_mask, bias, out, scale);
  }

  template <tensor::InputOperandOf<2> Q, tensor::InputOperandOf<2> K,
            tensor::InputOperandOf<2> V, tensor::OutputOperandOf<2> Output>
  VECOPS_INLINE void operator()(
      const Q& q, const K& k, const V& v,
      const Output& out, Score scale) const {
    (*this)(q, k, v, tensor::nullopt, tensor::nullopt,
            tensor::nullopt, tensor::nullopt, out, scale);
  }

private:
  template <typename QSpec, typename KSpec, typename VSpec,
            typename QMSpec, typename KMSpec, typename AMSpec,
            typename BiasSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_specs(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const OutSpec& out) const {
    validate_specs(
        q, k, v, query_mask, key_mask, attention_mask, bias, out);
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    const nint_t block = std::min<nint_t>(Config::query_block, lq);
    const auto block_extent = meta::dyn<1, 1, Config::query_block>(block);
    const auto lkv_extent = tensor::size<0>(k.input_layout());
    const auto dqk_extent = tensor::size<1>(q.input_layout());
    const auto dv_extent = tensor::size<1>(v.input_layout());
    auto q_block = tensor::narrow_view<0>(q, 0, block_extent);
    auto out_block = tensor::narrow_view<0>(out, 0, block_extent);
    auto score_tensor = tensor::make_unbound_tensor<Score>(
        tensor::make_shape(block_extent, lkv_extent));
    auto probability_tensor = tensor::make_unbound_tensor<Probability>(
        tensor::make_shape(block_extent, lkv_extent));
    auto v_transposed = tensor::transpose_view<0, 1>(v);
    MatmulOp matmul_op{config.matmul};
    const nint_t qk_bytes = matmul_op.required_workspace(
        block_extent, lkv_extent, dqk_extent,
        q_block, k, score_tensor);
    const nint_t pv_bytes = matmul_op.required_workspace(
        block_extent, dv_extent, lkv_extent,
        probability_tensor, v_transposed, out_block);
    const nint_t persistent =
        kernel::WorkspaceView::allocation_bytes<Score>(
            Config::query_block * lkv) +
        kernel::WorkspaceView::allocation_bytes<Probability>(
            Config::query_block * lkv) +
        tensor::required_workspace(
            query_mask, MaskPolicy{}) +
        tensor::required_workspace(
            key_mask, MaskPolicy{}) +
        tensor::required_workspace(
            attention_mask, MatrixPolicy{}) +
        tensor::required_workspace(
            bias, MatrixPolicy{});
    return persistent + std::max(qk_bytes, pv_bytes);
  }

  template <typename QSpec, typename KSpec, typename VSpec,
            typename QMSpec, typename KMSpec, typename AMSpec,
            typename BiasSpec, typename OutSpec>
  VECOPS_INLINE void validate_specs(
      const QSpec& q, const KSpec& k, const VSpec& v,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const OutSpec& out) const {
    attention_details::validate_dense_layouts(
        q.input_layout(), k.input_layout(), v.input_layout(),
        out.output_layout());
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
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

  template <execution::ExecutionScope Scope,
            typename QSpec, typename KSpec, typename VSpec,
            typename QMSpec, typename KMSpec, typename AMSpec,
            typename BiasSpec, typename OutSpec>
  VECOPS_INLINE void execute_specs(
      Scope& scope, const QSpec& q, const KSpec& k, const VSpec& v,
      const QMSpec& query_mask, const KMSpec& key_mask,
      const AMSpec& attention_mask, const BiasSpec& bias,
      const OutSpec& out, Score scale) const {
    validate_specs(
        q, k, v, query_mask, key_mask, attention_mask, bias, out);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const auto lq = tensor::size<0>(q.input_layout());
    const auto lkv = tensor::size<0>(k.input_layout());
    const auto dv = tensor::size<1>(v.input_layout());
    const auto lkv_extent = tensor::size<0>(k.input_layout());
    const auto dqk_extent = tensor::size<1>(q.input_layout());
    const auto dv_extent = tensor::size<1>(v.input_layout());
    auto score_buffer = workspace.template allocate_tensor<Score>(
        tensor::make_shape(Config::query_block_extent, lkv_extent));
    auto probability_buffer = workspace.template allocate_tensor<Probability>(
        tensor::make_shape(Config::query_block_extent, lkv_extent));
    Score* scores = score_buffer.data();
    Probability* probabilities = probability_buffer.data();

    auto qm = tensor::bind(
        query_mask, MaskPolicy{}, workspace);
    auto km = tensor::bind(
        key_mask, MaskPolicy{}, workspace);
    auto am = tensor::bind(
        attention_mask, MatrixPolicy{}, workspace);
    auto b = tensor::bind(
        bias, MatrixPolicy{}, workspace);

    tensor::with_optional_unordered_access(
        [&](auto uqm, auto ukm, auto uam, auto ubias) {
            MatmulOp matmul_op{config.matmul};
            auto v_transposed = tensor::transpose_view<0, 1>(v);
            for (nint_t begin = 0; begin < lq;
                 begin += Config::query_block) {
              const nint_t rows = std::min<nint_t>(
                  Config::query_block, lq - begin);
              const auto row_extent =
                  meta::dyn<1, 1, Config::query_block>(rows);
              auto q_block = tensor::narrow_view<0>(
                  q, begin, row_extent);
              auto out_block = tensor::narrow_view<0>(
                  out, begin, row_extent);
              auto score_tensor = tensor::make_tensor(
                  scores, tensor::make_shape(row_extent, lkv_extent));
              auto probability_tensor = tensor::make_tensor(
                  probabilities,
                  tensor::make_shape(row_extent, lkv_extent));
              matmul_op(
                  scope, row_extent, lkv_extent, dqk_extent,
                  q_block, k, score_tensor);

              std::array<
                  Score,
                  static_cast<std::size_t>(Config::query_block)>
                  query_activity{};
              attention_details::load_activity_segment<Score>(
                  uqm, begin, rows, query_activity.data());
              for (nint_t row = 0; row < rows; ++row) {
                const nint_t q_index = begin + row;
                const bool query_active =
                    tensor::is_nullopt_v<decltype(uqm)> ||
                    query_activity[static_cast<std::size_t>(row)] != Score{};
                const auto stats = attention_details::decorate_score_row<
                    Config::causal_mode>(
                    scores + row * lkv, lkv, query_active,
                    q_index, 0, lq, lkv, scale, ukm, uam, ubias);
                attention_details::softmax_row<
                    Score, Probability, Config::exp_accuracy>(
                    scores + row * lkv,
                    probabilities + row * lkv, lkv,
                    stats.any, stats.maximum);
              }

              matmul_op(
                  scope, row_extent, dv_extent, lkv_extent,
                  probability_tensor, v_transposed, out_block);
            }
        },
        qm, km, am, b);
    workspace.rewind(mark);
    (void)dv;
  }
};

} // namespace attention_details

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_SDPA_H
