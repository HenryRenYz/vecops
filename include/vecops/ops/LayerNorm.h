#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

namespace vecops::ops {

template <typename ComputeT = float32_t>
struct LayerNormConfig {
  using ComputeType = ComputeT;
  ComputeT eps = ComputeT(1e-5f);
};

namespace details {

template <typename InLayout, typename ScaleLayout, typename BiasLayout,
          typename OutLayout>
VECOPS_INLINE void validate_layernorm_layouts(
    const InLayout& in, const ScaleLayout& scale, const BiasLayout& bias,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1);
  static_assert(OutLayout::Ndim == InLayout::Ndim);
  static_assert(ScaleLayout::Ndim == 1);
  static_assert(BiasLayout::Ndim == 1);
  for (int d = 0; d < InLayout::Ndim; ++d) {
    VECOPS_ASSERT(in.shape()[d] == out.shape()[d],
                  "LayerNorm input/output shape mismatch");
  }
  const nint_t n = in.shape()[InLayout::Ndim - 1];
  VECOPS_ASSERT(n > 0, "LayerNorm normalized dimension must be non-empty");
  VECOPS_ASSERT(scale.shape()[0] == n && bias.shape()[0] == n,
                "LayerNorm scale/bias size mismatch");
}

/** @brief Per-row loop tuning. */
template <int Full, int Tail, bool Fused,
          kernel::loop::TailCarryPolicy Carry>
struct RowRecipe {
  static constexpr int FullFactor = Full;
  static constexpr int TailFactor = Tail;
  static constexpr bool FusedShift = Fused;
  static constexpr kernel::loop::TailCarryPolicy TailPolicy = Carry;
};

using GenericRowRecipe = RowRecipe<
    4, 1, false, kernel::loop::TailCarryPolicy::independent>;

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
using SVE16RowRecipe = RowRecipe<
    4, 2, true, kernel::loop::TailCarryPolicy::reuse_prefix>;
#endif

} // namespace details

template <typename Config = LayerNormConfig<>>
class LayerNorm {
  using XPolicy = tensor::InputAccessPolicy<
      0, 2, tensor::AccessPlan::automatic_deferred>;
  using ParamPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct>;
  using YPolicy = tensor::OutputAccessPolicy<
      0, tensor::AccessPlan::direct>;

public:
  using ComputeType = typename Config::ComputeType;
  using Tag = vec::ScalableTag<ComputeType, 0>;

  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <tensor::InputSpecLike InSpec, tensor::InputSpecLike ScaleSpec,
            tensor::InputSpecLike BiasSpec, tensor::OutputSpecLike OutSpec>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    details::validate_layernorm_layouts(
        in.input_layout(), scale.input_layout(), bias.input_layout(),
        out.output_layout());
    return tensor::required_workspace(
        tensor::take_trailing<1>(in), XPolicy{});
  }

  template <tensor::InputSpecLike InSpec, tensor::InputSpecLike ScaleSpec,
            tensor::InputSpecLike BiasSpec, tensor::OutputSpecLike OutSpec>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    details::validate_layernorm_layouts(
        in.input_layout(), scale.input_layout(), bias.input_layout(),
        out.output_layout());

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        (is_float16_v<InElement> || is_bfloat16_v<InElement>);
    using RowRecipe = std::conditional_t<
        UseSVE16, details::SVE16RowRecipe,
        details::GenericRowRecipe>;
#else
    using RowRecipe = details::GenericRowRecipe;
#endif

    run_rows<RowRecipe>(workspace, in, scale, bias, out);
  }

  template <tensor::InputSpecLike InSpec, tensor::InputSpecLike ScaleSpec,
            tensor::InputSpecLike BiasSpec, tensor::OutputSpecLike OutSpec>
  VECOPS_INLINE void operator()(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    kernel::Workspace storage(required_workspace(in, scale, bias, out));
    auto workspace = storage.view();
    (*this)(workspace, in, scale, bias, out);
  }

private:
  // Measured compiler tuning: x86 keeps the row nest outlined; SVE inlines it.
  template <typename Recipe, typename InSpec, typename ScaleSpec,
            typename BiasSpec, typename OutSpec>
#if defined(ARCH_X86_FAMILY)
  VECOPS_NOINLINE
#else
  VECOPS_ALWAYS_INLINE
#endif
  void run_rows(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    auto gamma = tensor::bind(scale, ParamPolicy{}, workspace);
    auto beta = tensor::bind(bias, ParamPolicy{}, workspace);
    kernel::loop::for_each_dims<PrefixRank>(
        [this, &workspace, &gamma, &beta](const auto& in_row,
                                          const auto& out_row)
            VECOPS_INLINE_LAMBDA {
          bind_row<Recipe>(workspace, in_row, gamma, beta, out_row);
        },
        in, out);
  }

  template <typename Recipe, typename InSpec, typename ScaleAccess,
            typename BiasAccess, typename OutSpec>
  VECOPS_ALWAYS_INLINE void bind_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      ScaleAccess& gamma, BiasAccess& beta, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    kernel::with_operands(
        workspace, tensor::operand(in, XPolicy{}),
        tensor::operand(out, YPolicy{}),
        [this, &in, &gamma, &beta](auto& x, auto& y)
            VECOPS_INLINE_LAMBDA {
          run_row<Recipe>(in, x, gamma, beta, y);
        });
  }

  template <typename Recipe, typename InSpec, typename X, typename Gamma,
            typename Beta, typename Y>
  VECOPS_ALWAYS_INLINE void run_row(
      const InSpec& in, X& x, Gamma& gamma, Beta& beta, Y& y) const {
    static_assert(InSpec::InputTensor::Ndim == 1);

    Tag base_tag{};
    const nint_t n = in.input_layout().shape()[0];

    ComputeType sum_value{};
    ComputeType sum_sq_value{};
    tensor::with_unordered_access(x, [&](auto ux) VECOPS_INLINE_LAMBDA {
      kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor, Recipe::TailPolicy>(
          base_tag, n,
          [&, n](auto block_tag, nint_t col, auto active,
                 auto& sum, auto& sum_sq) VECOPS_INLINE_LAMBDA {
            auto value = ux.load(
                block_tag, tensor::coord(col), active,
                tensor::materialize::populate);
            sum = vec::add(sum, value);
            sum_sq = vec::fmadd(value, value, sum_sq);
          },
          kernel::loop::reduce_add(sum_value),
          kernel::loop::reduce_add(sum_sq_value));
    });

    const ComputeType inv_n =
        ComputeType(1) / static_cast<ComputeType>(n);
    const ComputeType mean = sum_value * inv_n;
    const ComputeType variance =
        std::max(sum_sq_value * inv_n - mean * mean, ComputeType(0));
    const ComputeType rstd =
        ComputeType(1) / std::sqrt(variance + config.eps);
    const ComputeType center = Recipe::FusedShift ? -mean * rstd : mean;

    tensor::with_unordered_access(
        x, gamma, beta, y,
        [&](auto ux, auto ugamma, auto ubeta, auto uy) VECOPS_INLINE_LAMBDA {
          kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor>(
              base_tag, n,
              [&, n](auto block_tag, nint_t col, auto active,
                     const auto& center_v,
                     const auto& rstd_v) VECOPS_INLINE_LAMBDA {
                const auto position = tensor::coord(col);
                auto value = ux.load(block_tag, position, active);
                auto scale_value = ugamma.load(block_tag, position, active);
                auto bias_value = ubeta.load(block_tag, position, active);
                if constexpr (Recipe::FusedShift) {
                  value = vec::fmadd(value, rstd_v, center_v);
                } else {
                  value = vec::mul(vec::sub(value, center_v), rstd_v);
                }
                uy.store(
                    block_tag, position,
                    vec::fmadd(value, scale_value, bias_value), active);
              },
              kernel::loop::invariant(center),
              kernel::loop::invariant(rstd));
          uy.commit();
        });
  }
};

template <typename Config = LayerNormConfig<>>
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
