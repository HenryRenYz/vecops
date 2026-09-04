#ifndef VECOPS_OPS_DETAILS_ATTENTION_OPERATION_H
#define VECOPS_OPS_DETAILS_ATTENTION_OPERATION_H

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Atom.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/details/attention/Common.h"

namespace vecops::ops {

using AttentionCausalMode = attention_details::CausalMode;

enum class AttentionStrategy {
  automatic,
  materialized,
  streaming,
};

/**
 * @brief Configuration shared by the dense and block-sparse attention family.
 *
 * The Matmul configuration fixes the hardware atom and its input/accumulator
 * types. Tensor memory element types remain independent and are converted by
 * DataAccess. QueryBlock and KeyValueBlock are semantic sparse-block extents;
 * the dense materialized path also uses QueryBlock as its score-panel height.
 */
template <typename MatmulConfigT, int QueryBlock = 32,
          int KeyValueBlock = 32,
          AttentionCausalMode Causal = AttentionCausalMode::none,
          AttentionStrategy Strategy = AttentionStrategy::automatic,
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
  static constexpr AttentionCausalMode causal_mode = Causal;
  static constexpr AttentionStrategy strategy = Strategy;
  static constexpr vec::Accuracy exp_accuracy = ExpAccuracy;

  [[no_unique_address]] MatmulConfig matmul{};
  nint_t selected_blocks = 1;
  nint_t random_blocks = 0;
  ScoreType static_probability = ScoreType{1};
  ScoreType random_probability = ScoreType{1};
  ScoreType minimum_probability = ScoreType{};
};

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
 * contract and is selected by the higher-level sparse/automatic operators.
 */
template <typename Config>
class DenseMaterializedAttention {
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

  VECOPS_INLINE constexpr explicit DenseMaterializedAttention(Config cfg)
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
    auto q_spec = tensor::as_input_spec<typename Atom::TA>(q);
    auto k_spec = tensor::as_input_spec<typename Atom::TB>(k);
    auto v_spec = tensor::as_input_spec<typename Atom::TB>(v);
    auto qm_spec = attention_details::as_mask_input<Score>(query_mask);
    auto km_spec = attention_details::as_mask_input<Score>(key_mask);
    auto am_spec = attention_details::as_mask_input<Score>(attention_mask);
    auto bias_spec = attention_details::as_optional_input<Score>(bias);
    auto out_spec = tensor::as_output_spec<Score>(out);
    return required_workspace_specs(
        q_spec, k_spec, v_spec, qm_spec, km_spec, am_spec, bias_spec,
        out_spec);
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
              active, q_spec, k_spec, v_spec, qm_spec, km_spec,
              am_spec, bias_spec, out_spec, scale);
        });
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

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V,
            attention_details::OptionalInputOperand<1> QueryMask,
            attention_details::OptionalInputOperand<1> KeyMask,
            attention_details::OptionalInputOperand<2> AttentionMask,
            attention_details::OptionalInputOperand<2> Bias,
            tensor::OutputOperand Output>
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

  template <tensor::InputOperand Q, tensor::InputOperand K,
            tensor::InputOperand V, tensor::OutputOperand Output>
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
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t block = std::min<nint_t>(Config::query_block, lq);
    const auto block_extent = meta::dyn<1, 1, Config::query_block>(block);
    const auto lkv_extent = tensor::size_value<0>(k.input_layout());
    const auto dqk_extent = tensor::size_value<1>(q.input_layout());
    const auto dv_extent = tensor::size_value<1>(v.input_layout());
    auto q_block = tensor::narrow_view<0>(q, 0, block_extent);
    auto out_block = tensor::narrow_view<0>(out, 0, block_extent);
    auto* score_ptr = static_cast<Score*>(nullptr);
    auto* probability_ptr = static_cast<Probability*>(nullptr);
    auto score_tensor = tensor::make_tensor(
        score_ptr, tensor::make_shape(block_extent, lkv_extent));
    auto probability_tensor = tensor::make_tensor(
        probability_ptr, tensor::make_shape(block_extent, lkv_extent));
    auto v_transposed = tensor::transpose_view<0, 1>(v);
    MatmulOp matmul_op{config.matmul};
    const nint_t qk_bytes = matmul_op.required_workspace(
        block_extent, lkv_extent, dqk_extent,
        q_block, k, score_tensor);
    const nint_t pv_bytes = matmul_op.required_workspace(
        block_extent, dv_extent, lkv_extent,
        probability_tensor, v_transposed, out_block);
    const nint_t persistent =
        attention_details::aligned_bytes<Score>(
            Config::query_block * lkv) +
        attention_details::aligned_bytes<Probability>(
            Config::query_block * lkv) +
        attention_details::optional_required_workspace(
            query_mask, MaskPolicy{}) +
        attention_details::optional_required_workspace(
            key_mask, MaskPolicy{}) +
        attention_details::optional_required_workspace(
            attention_mask, MatrixPolicy{}) +
        attention_details::optional_required_workspace(
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
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
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
    const nint_t lq = q.input_layout().shape()[0];
    const nint_t lkv = k.input_layout().shape()[0];
    const nint_t dv = v.input_layout().shape()[1];
    const auto lkv_extent = tensor::size_value<0>(k.input_layout());
    const auto dqk_extent = tensor::size_value<1>(q.input_layout());
    const auto dv_extent = tensor::size_value<1>(v.input_layout());
    Score* scores = attention_details::allocate_aligned<Score>(
        workspace, Config::query_block * lkv);
    Probability* probabilities =
        attention_details::allocate_aligned<Probability>(
            workspace, Config::query_block * lkv);

    auto qm = attention_details::bind_optional_input(
        query_mask, MaskPolicy{}, workspace);
    auto km = attention_details::bind_optional_input(
        key_mask, MaskPolicy{}, workspace);
    auto am = attention_details::bind_optional_input(
        attention_mask, MatrixPolicy{}, workspace);
    auto b = attention_details::bind_optional_input(
        bias, MatrixPolicy{}, workspace);

    attention_details::with_optional_access(
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

              for (nint_t row = 0; row < rows; ++row) {
                const nint_t q_index = begin + row;
                bool query_active = true;
                if constexpr (!tensor::is_nullopt_v<decltype(uqm)>) {
                  query_active =
                      uqm.load_scalar(tensor::coord(q_index)) != Score{};
                }
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

template <typename Config>
VECOPS_INLINE constexpr auto dense_materialized_attention(Config config) {
  return DenseMaterializedAttention<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_ATTENTION_OPERATION_H
