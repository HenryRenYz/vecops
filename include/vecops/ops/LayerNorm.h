//
// Created by renyz on 2026/7/9.
//

#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/gemm/HOP.h"
#include "vecops/gemm/Tensor.h"
#include "vecops/gemm/Workspace.h"
#include "vecops/vec/Vec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
#include <arm_sve.h>
#endif

namespace vecops::ops {

template <
    typename ComputeT = float32_t,
    typename VecTag = vec::ScalableTag<ComputeT, 0>>
struct LayerNormConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;

  ComputeT eps = ComputeT(1e-5f);
};

namespace details {

template <typename Config>
using layernorm_compute_t = typename std::remove_cvref_t<Config>::ComputeType;

template <typename Config>
using layernorm_tag_t = typename std::remove_cvref_t<Config>::Tag;

// TODO sooner or later I will take this down
template <typename Tag>
VECOPS_INLINE auto vector_step_value(Tag tag) {
  using TagT = std::remove_cvref_t<Tag>;
  if constexpr (TagT::is_runtime_size) {
    return gemm::Any{vec::size(tag)};
  } else {
    return gemm::Const<vec::size(TagT{})>{};
  }
}

template <typename InLayout, typename ScaleLayout, typename BiasLayout, typename OutLayout>
VECOPS_INLINE void validate_layernorm_layouts(
    const InLayout& in,
    const ScaleLayout& scale,
    const BiasLayout& bias,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1, "LayerNorm input rank must be at least 1");
  static_assert(OutLayout::Ndim == InLayout::Ndim, "LayerNorm output rank must match input rank");
  static_assert(ScaleLayout::Ndim == 1, "LayerNorm scale rank must be 1");
  static_assert(BiasLayout::Ndim == 1, "LayerNorm bias rank must be 1");

  constexpr int rank = InLayout::Ndim;
  for (int d = 0; d < rank; ++d) {
    VECOPS_ASSERT(
        in.shape()[d] == out.shape()[d],
        "LayerNorm input/output shape mismatch at dim %d", d);
  }
  const nint_t normalized_size = in.shape()[rank - 1];
  VECOPS_ASSERT(normalized_size > 0, "LayerNorm normalized dimension must be non-empty");
  VECOPS_ASSERT(
      scale.shape()[0] == normalized_size,
      "LayerNorm scale size mismatch: %td != %td", scale.shape()[0], normalized_size);
  VECOPS_ASSERT(
      bias.shape()[0] == normalized_size,
      "LayerNorm bias size mismatch: %td != %td", bias.shape()[0], normalized_size);
}

} // namespace details

template <typename Config = LayerNormConfig<>>
struct LayerNorm {
  using ComputeType = details::layernorm_compute_t<Config>;
  using Tag = details::layernorm_tag_t<Config>;

public:
  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec,
      std::enable_if_t<
          gemm::is_input_spec_v<InSpec> &&
          gemm::is_input_spec_v<ScaleSpec> &&
          gemm::is_input_spec_v<BiasSpec> &&
          gemm::is_output_spec_v<OutSpec>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(gemm::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(gemm::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");
    details::validate_layernorm_layouts(
        in.input_layout(),
        scale.input_layout(),
        bias.input_layout(),
        out.output_layout());
    return required_workspace_impl(in, scale, bias, out);
  }

  template <
      typename InLayout,
      typename ScaleLayout,
      typename BiasLayout,
      typename OutLayout,
      std::enable_if_t<
          gemm::is_layout<std::remove_cvref_t<InLayout>> &&
          gemm::is_layout<std::remove_cvref_t<ScaleLayout>> &&
          gemm::is_layout<std::remove_cvref_t<BiasLayout>> &&
          gemm::is_layout<std::remove_cvref_t<OutLayout>>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InLayout& in,
      const ScaleLayout& scale,
      const BiasLayout& bias,
      const OutLayout& out) const {
    return required_workspace(
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<InLayout>>(in),
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<ScaleLayout>>(scale),
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<BiasLayout>>(bias),
        gemm::OutputSpec<ComputeType, ComputeType, std::remove_cvref_t<OutLayout>>(out));
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  VECOPS_INLINE void operator()(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(gemm::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(gemm::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");

    const auto& in_layout = in.input_layout();
    const auto& scale_layout = scale.input_layout();
    const auto& bias_layout = bias.input_layout();
    const auto& out_layout = out.output_layout();
    details::validate_layernorm_layouts(in_layout, scale_layout, bias_layout, out_layout);

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InSpecT = std::remove_cvref_t<InSpec>;
    using ScaleSpecT = std::remove_cvref_t<ScaleSpec>;
    using BiasSpecT = std::remove_cvref_t<BiasSpec>;
    using OutSpecT = std::remove_cvref_t<OutSpec>;
    using InElement = typename InSpecT::InputTensor::ElementType;
    using ScaleElement = typename ScaleSpecT::InputTensor::ElementType;
    using BiasElement = typename BiasSpecT::InputTensor::ElementType;
    using OutElement = typename OutSpecT::OutputTensor::ElementType;
#if defined(__ARM_FEATURE_SVE_BF16)
    constexpr bool supported_sve_16bit_element =
        IsFloat16V<InElement> || IsBfloat16V<InElement>;
#else
    constexpr bool supported_sve_16bit_element = IsFloat16V<InElement>;
#endif
    constexpr bool use_sve_16bit_path =
        std::is_same_v<ComputeType, float32_t> &&
        std::is_same_v<Tag, vec::ScalableTag<float32_t, 0>> &&
        supported_sve_16bit_element &&
        std::is_same_v<ScaleElement, InElement> &&
        std::is_same_v<BiasElement, InElement> &&
        std::is_same_v<OutElement, InElement> &&
        std::is_same_v<typename InSpecT::Transform,
                       IdentityVecTransform<float32_t, InElement>> &&
        std::is_same_v<typename ScaleSpecT::Transform,
                       IdentityVecTransform<float32_t, InElement>> &&
        std::is_same_v<typename BiasSpecT::Transform,
                       IdentityVecTransform<float32_t, InElement>> &&
        std::is_same_v<typename OutSpecT::Transform,
                       IdentityVecTransform<InElement, float32_t>> &&
        gemm::details::IsLastDimContiguous<
            typename InSpecT::InputLayout>::value &&
        gemm::details::IsLastDimContiguous<
            typename ScaleSpecT::InputLayout>::value &&
        gemm::details::IsLastDimContiguous<
            typename BiasSpecT::InputLayout>::value &&
        gemm::details::IsLastDimContiguous<
            typename OutSpecT::OutputLayout>::value;

    if constexpr (use_sve_16bit_path) {
      constexpr int prefix_rank = InSpecT::InputLayout::Ndim - 1;
      gemm::hop::for_each_dims<prefix_rank>(
          [this, &scale, &bias](const auto &in_row, const auto &out_row) {
            this->run_row_sve_16bit(in_row, scale, bias, out_row);
          },
          in, out);
      return;
    }
#endif

    auto mark = workspace.mark();
    {
      auto gamma = scale.bind(workspace);
      auto beta = bias.bind(workspace);

      constexpr int prefix_rank =
          std::remove_cvref_t<decltype(in_layout)>::Ndim - 1;
      using InElement =
          typename std::remove_cvref_t<InSpec>::InputTensor::ElementType;
      bool use_three_way_output = false;
#if defined(ARCH_X86_FAMILY)
      if constexpr (std::is_same_v<std::remove_cv_t<InElement>, float32_t>) {
        nint_t row_count = 1;
        for (int d = 0; d < prefix_rank; ++d)
          row_count *= in_layout.shape()[d];
        const nint_t normalized_size = in_layout.shape()[prefix_rank];
        // Three output chains avoid the sustained-AVX512 regression observed
        // for large batches of short fp32 rows on Sapphire Rapids.
        use_three_way_output = row_count >= 4096 && normalized_size <= 768;
      }
#endif
#if defined(__aarch64__)
      if constexpr (IsFloat16V<std::remove_cv_t<InElement>>) {
        const nint_t normalized_size = in_layout.shape()[prefix_rank];
        // Three explicitly scheduled output chains improve long-row SVE IPC,
        // while HOP remains faster for short rows and for the reduction pass.
        if (normalized_size >= 1000) {
          gemm::hop::for_each_dims<prefix_rank>(
              [this, &workspace, &gamma, &beta](const auto &in_row,
                                                 const auto &out_row)
                  __attribute__((aligned(64))) {
                    this->run_row<3>(workspace, in_row, gamma, beta, out_row);
                  },
              in, out);
        } else {
          gemm::hop::for_each_dims<prefix_rank>(
              [this, &workspace, &gamma, &beta](const auto &in_row,
                                                 const auto &out_row) {
                this->run_row_hop(workspace, in_row, gamma, beta, out_row);
              },
              in, out);
        }
      } else if constexpr (sizeof(InElement) <= 2) {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace, &gamma, &beta](const auto &in_row,
                                               const auto &out_row) {
              this->run_row_hop(workspace, in_row, gamma, beta, out_row);
            },
            in, out);
      } else {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace, &gamma, &beta](const auto &in_row,
                                               const auto &out_row) {
              this->run_row(workspace, in_row, gamma, beta, out_row);
            },
            in, out);
      }
#else
      gemm::hop::for_each_dims<prefix_rank>(
          [this, &workspace, &gamma, &beta,
           use_three_way_output](const auto &in_row, const auto &out_row) {
#if defined(ARCH_X86_FAMILY)
            if constexpr (std::is_same_v<
                              std::remove_cv_t<InElement>,
                              float32_t>) {
              if (use_three_way_output) {
                this->run_row<3>(workspace, in_row, gamma, beta, out_row);
              } else {
                this->run_row(workspace, in_row, gamma, beta, out_row);
              }
            } else {
              this->run_row(workspace, in_row, gamma, beta, out_row);
            }
#else
            (void)use_three_way_output;
            this->run_row(workspace, in_row, gamma, beta, out_row);
#endif
          },
          in, out);
#endif
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void operator()(const InSpec &in, const ScaleSpec &scale,
                                const BiasSpec &bias,
                                const OutSpec &out) const {
    gemm::Workspace workspace(required_workspace(in, scale, bias, out));
    auto view = workspace.view();
    (*this)(view, in, scale, bias, out);
  }

private:
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void run_row_sve_16bit(const InSpec &in,
                                       const ScaleSpec &scale,
                                       const BiasSpec &bias,
                                       const OutSpec &out) const {
    using Element =
        typename std::remove_cvref_t<InSpec>::InputTensor::ElementType;
    static_assert(IsFloat16V<Element> || IsBfloat16V<Element>);

    const Element *x = in.tensor().data();
    const Element *gamma = scale.tensor().data();
    const Element *beta = bias.tensor().data();
    Element *y = const_cast<Element *>(out.tensor().data());
    const nint_t n = in.input_layout().shape()[0];
    const nint_t half_lanes = static_cast<nint_t>(svcnth());
    const nint_t word_lanes = static_cast<nint_t>(svcntw());
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();

    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);
    svfloat32_t sum_sq0 = svdup_n_f32(0.0f);
    svfloat32_t sum_sq1 = svdup_n_f32(0.0f);
    svfloat32_t sum_sq2 = svdup_n_f32(0.0f);
    svfloat32_t sum_sq3 = svdup_n_f32(0.0f);

    svuint32_t mask_hi = svdup_u32(0xffff0000);

    nint_t col = 0;
    const nint_t reduction_step = 2 * half_lanes;
    for (; col + reduction_step <= n; col += reduction_step) {
      svfloat32_t x0_lo;
      svfloat32_t x0_hi;
      svfloat32_t x1_lo;
      svfloat32_t x1_hi;
      if constexpr (IsFloat16V<Element>) {
//        const auto h01 = svld2_f16(pg16, reinterpret_cast<const __fp16 *>(x + col));
//        x0_lo = svcvt_f32_f16_x(pg32, svget2(h01, 0));
//        x0_hi = svcvtlt_f32_f16_x(pg32, svget2(h01, 0));
//        x1_lo = svcvt_f32_f16_x(pg32, svget2(h01, 1));
//        x1_hi = svcvtlt_f32_f16_x(pg32, svget2(h01, 1));
        const auto h0 = svld1_f16(
            pg16, reinterpret_cast<const __fp16 *>(x + col));
        const auto h1 = svld1_f16(
            pg16,
            reinterpret_cast<const __fp16 *>(x + col + half_lanes));
        x0_lo = svcvt_f32_f16_x(pg32, h0);
        x0_hi = svcvtlt_f32_f16_x(pg32, h0);
        x1_lo = svcvt_f32_f16_x(pg32, h1);
        x1_hi = svcvtlt_f32_f16_x(pg32, h1);
      } else {
#if defined(__ARM_FEATURE_SVE_BF16)
        x0_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    pg32, reinterpret_cast<const uint16_t *>(x + col)),
                16));
        x0_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    pg32,
                    reinterpret_cast<const uint16_t *>(x + col + word_lanes)),
                16));
        x1_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    pg32,
                    reinterpret_cast<const uint16_t *>(x + col + half_lanes)),
                16));
        x1_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    pg32, reinterpret_cast<const uint16_t *>(
                              x + col + half_lanes + word_lanes)),
                16));
#endif
      }

      sum0 = svadd_f32_x(pg32, sum0, x0_lo);
      sum1 = svadd_f32_x(pg32, sum1, x0_hi);
      sum2 = svadd_f32_x(pg32, sum2, x1_lo);
      sum3 = svadd_f32_x(pg32, sum3, x1_hi);
      sum_sq0 = svmla_f32_x(pg32, sum_sq0, x0_lo, x0_lo);
      sum_sq1 = svmla_f32_x(pg32, sum_sq1, x0_hi, x0_hi);
      sum_sq2 = svmla_f32_x(pg32, sum_sq2, x1_lo, x1_lo);
      sum_sq3 = svmla_f32_x(pg32, sum_sq3, x1_hi, x1_hi);
    }

    for (; col < n; col += half_lanes) {
      const svbool_t tail = svwhilelt_b16(
          static_cast<uint64_t>(col), static_cast<uint64_t>(n));
      svfloat32_t x_lo;
      svfloat32_t x_hi;
      if constexpr (IsFloat16V<Element>) {
        const auto h = svld1_f16(
            tail, reinterpret_cast<const __fp16 *>(x + col));
        x_lo = svcvt_f32_f16_x(pg32, h);
        x_hi = svcvtlt_f32_f16_x(pg32, h);
      } else {
#if defined(__ARM_FEATURE_SVE_BF16)
        const svbool_t tail_lo = svwhilelt_b32(
            static_cast<uint64_t>(col), static_cast<uint64_t>(n));
        const svbool_t tail_hi = svwhilelt_b32(
            static_cast<uint64_t>(col + word_lanes),
            static_cast<uint64_t>(n));
        x_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_lo, reinterpret_cast<const uint16_t *>(x + col)),
                16));
        x_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_hi,
                    reinterpret_cast<const uint16_t *>(x + col + word_lanes)),
                16));
#endif
      }
      sum0 = svadd_f32_x(pg32, sum0, x_lo);
      sum1 = svadd_f32_x(pg32, sum1, x_hi);
      sum_sq0 = svmla_f32_x(pg32, sum_sq0, x_lo, x_lo);
      sum_sq1 = svmla_f32_x(pg32, sum_sq1, x_hi, x_hi);
    }

    sum0 = svadd_f32_x(pg32, sum0, sum1);
    sum2 = svadd_f32_x(pg32, sum2, sum3);
    sum_sq0 = svadd_f32_x(pg32, sum_sq0, sum_sq1);
    sum_sq2 = svadd_f32_x(pg32, sum_sq2, sum_sq3);
    sum0 = svadd_f32_x(pg32, sum0, sum2);
    sum_sq0 = svadd_f32_x(pg32, sum_sq0, sum_sq2);

    const float sum = svaddv_f32(pg32, sum0);
    const float sum_sq = svaddv_f32(pg32, sum_sq0);
    const float inv_n = 1.0f / static_cast<float>(n);
    const float mean = sum * inv_n;
    const float variance =
        std::max(sum_sq * inv_n - mean * mean, 0.0f);
    const float rstd = 1.0f / std::sqrt(variance + config.eps);
    const svfloat32_t mean_v = svdup_n_f32(mean);
    const svfloat32_t rstd_v = svdup_n_f32(rstd);

    for (col = 0; col < n; col += half_lanes) {
      const svbool_t tail = svwhilelt_b16(
          static_cast<uint64_t>(col), static_cast<uint64_t>(n));
      svfloat32_t x_lo;
      svfloat32_t x_hi;
      svfloat32_t gamma_lo;
      svfloat32_t gamma_hi;
      svfloat32_t beta_lo;
      svfloat32_t beta_hi;

      if constexpr (IsFloat16V<Element>) {
        const auto xh = svld1_f16(
            tail, reinterpret_cast<const __fp16 *>(x + col));
        const auto gh = svld1_f16(
            tail, reinterpret_cast<const __fp16 *>(gamma + col));
        const auto bh = svld1_f16(
            tail, reinterpret_cast<const __fp16 *>(beta + col));
        x_lo = svcvt_f32_f16_x(pg32, xh);
        x_hi = svcvtlt_f32_f16_x(pg32, xh);
        gamma_lo = svcvt_f32_f16_x(pg32, gh);
        gamma_hi = svcvtlt_f32_f16_x(pg32, gh);
        beta_lo = svcvt_f32_f16_x(pg32, bh);
        beta_hi = svcvtlt_f32_f16_x(pg32, bh);
      } else {
#if defined(__ARM_FEATURE_SVE_BF16)
        const svbool_t tail_lo = svwhilelt_b32(
            static_cast<uint64_t>(col), static_cast<uint64_t>(n));
        const svbool_t tail_hi = svwhilelt_b32(
            static_cast<uint64_t>(col + word_lanes),
            static_cast<uint64_t>(n));
        x_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_lo, reinterpret_cast<const uint16_t *>(x + col)),
                16));
        x_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_hi,
                    reinterpret_cast<const uint16_t *>(x + col + word_lanes)),
                16));
        gamma_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_lo,
                    reinterpret_cast<const uint16_t *>(gamma + col)),
                16));
        gamma_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_hi, reinterpret_cast<const uint16_t *>(
                                 gamma + col + word_lanes)),
                16));
        beta_lo = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_lo, reinterpret_cast<const uint16_t *>(beta + col)),
                16));
        beta_hi = svreinterpret_f32_u32(
            svlsl_n_u32_x(
                pg32,
                svld1uh_u32(
                    tail_hi, reinterpret_cast<const uint16_t *>(
                                beta + col + word_lanes)),
                16));
#endif
      }

      x_lo = svmul_f32_x(
          pg32, svsub_f32_x(pg32, x_lo, mean_v), rstd_v);
      x_hi = svmul_f32_x(
          pg32, svsub_f32_x(pg32, x_hi, mean_v), rstd_v);
      const auto out_lo = svmla_f32_x(pg32, beta_lo, x_lo, gamma_lo);
      const auto out_hi = svmla_f32_x(pg32, beta_hi, x_hi, gamma_hi);

      if constexpr (IsFloat16V<Element>) {
        auto packed = svcvt_f16_f32_z(pg32, out_lo);
        packed = svcvtnt_f16_f32_m(packed, pg32, out_hi);
        svst1_f16(tail, reinterpret_cast<__fp16 *>(y + col), packed);
      } else {
#if defined(__ARM_FEATURE_SVE_BF16)
        const auto lo_bf16 = svcvt_bf16_f32_z(pg32, out_lo);
        const auto hi_bf16 = svcvt_bf16_f32_z(pg32, out_hi);
        const auto packed = svuzp1_bf16(lo_bf16, hi_bf16);
        svst1_bf16(tail, reinterpret_cast<__bf16 *>(y + col), packed);
#endif
      }
    }
  }
#endif

  template <typename InSpec, typename ScaleAccessor, typename BiasAccessor,
            typename OutSpec>
  VECOPS_INLINE void run_row_hop(gemm::WorkspaceView &workspace,
                                 const InSpec &in,
                                 const ScaleAccessor &gamma,
                                 const BiasAccessor &beta,
                                 const OutSpec &out) const {
    const auto &in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "LayerNorm row spec must be rank 1");

    Tag t;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const auto step = details::vector_step_value(t);

      const auto [sum, sum_sq] = gemm::hop::scan<4>(
          normalized_count,
          step,
          [&](auto &&use) VECOPS_INLINE_LAMBDA -> decltype(auto) {
            auto v_sum = vec::zeros(t);
            auto v_sum_sq = vec::zeros(t);
            return use(v_sum, v_sum_sq);
          },
          [&](nint_t col, auto &&count, auto &v_sum, auto &v_sum_sq)
              VECOPS_INLINE_LAMBDA {
                auto xv = x(t, count, col);
                v_sum = vec::add(v_sum, xv);
                v_sum_sq = vec::fmadd(xv, xv, v_sum_sq);
              },
          [&](auto &dst_sum, auto &dst_sum_sq, auto &src_sum, auto &src_sum_sq)
              VECOPS_INLINE_LAMBDA {
                dst_sum = vec::add(dst_sum, src_sum);
                dst_sum_sq = vec::add(dst_sum_sq, src_sum_sq);
              },
          [&](auto &v_sum, auto &v_sum_sq) {
            return std::make_pair(vec::reduce_add(t, v_sum),
                                  vec::reduce_add(t, v_sum_sq));
          });

      const auto inv_n =
          ComputeType(1) / static_cast<ComputeType>(normalized_count);
      const auto mean = sum * inv_n;
      const auto variance =
          std::max(sum_sq * inv_n - mean * mean, ComputeType(0));
      const auto rstd = ComputeType(1) / std::sqrt(variance + config.eps);

      const auto mean_v = vec::fill(t, mean);
      const auto rstd_v = vec::fill(t, rstd);

      gemm::hop::map<4>(normalized_count, step,
                        [&](nint_t col, auto &&count) {
                          auto xv = x(t, count, col);
                          auto gamma_v = gamma(t, count, col);
                          auto beta_v = beta(t, count, col);
                          auto centered = vec::sub(xv, mean_v);
                          auto normalized = vec::mul(centered, rstd_v);
                          auto affine =
                              vec::fmadd(normalized, gamma_v, beta_v);
                          y(t, affine, count, col);
                        });
    }
    workspace.rewind(mark);
  }

  template <int OutputUnrollOverride = 0, typename InSpec,
            typename ScaleAccessor, typename BiasAccessor, typename OutSpec>
  VECOPS_INLINE void run_row(gemm::WorkspaceView &workspace, const InSpec &in,
                             const ScaleAccessor &gamma,
                             const BiasAccessor &beta,
                             const OutSpec &out) const {
    const auto &in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "LayerNorm row spec must be rank 1");

    Tag t;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const auto step = details::vector_step_value(t);
      const nint_t normalized_count_int = static_cast<nint_t>(normalized_count);
      const nint_t step_int = static_cast<nint_t>(step);
      using InElement =
          typename std::remove_cvref_t<InSpec>::InputTensor::ElementType;
      constexpr int default_output_unroll = 4;
      constexpr int output_unroll = OutputUnrollOverride == 0
                                        ? default_output_unroll
                                        : OutputUnrollOverride;
      static_assert(output_unroll >= 2 && output_unroll <= 4);
      const nint_t reduction_group_step = 4 * step_int;
      const nint_t output_group_step = output_unroll * step_int;
#if defined(__aarch64__)
      // fp16 uses the HOP reduction but a manually scheduled output pass;
      // wider types use manual scheduling for both passes.
      constexpr bool use_grouped_reduction = sizeof(InElement) > 2;
      constexpr bool use_grouped_output = true;
#else
      constexpr bool use_grouped_reduction = true;
      constexpr bool use_grouped_output = true;
#endif

      ComputeType sum{};
      ComputeType sum_sq{};
      if constexpr (use_grouped_reduction) {
        // Keep loads, adds, and FMAs in separate phases. Compiler/HOP unrolling
        // otherwise keeps each lane's dependent operations adjacent.
        auto v_sum0 = vec::zeros(t);
        auto v_sum1 = vec::zeros(t);
        auto v_sum2 = vec::zeros(t);
        auto v_sum3 = vec::zeros(t);
        auto v_sum_sq0 = vec::zeros(t);
        auto v_sum_sq1 = vec::zeros(t);
        auto v_sum_sq2 = vec::zeros(t);
        auto v_sum_sq3 = vec::zeros(t);
        nint_t scan_col = 0;

        for (; scan_col + reduction_group_step <= normalized_count_int;
             scan_col += reduction_group_step) {
          auto x0 = x(t, step, scan_col);
          auto x1 = x(t, step, scan_col + step_int);
          auto x2 = x(t, step, scan_col + 2 * step_int);
          auto x3 = x(t, step, scan_col + 3 * step_int);

          v_sum0 = vec::add(v_sum0, x0);
          v_sum1 = vec::add(v_sum1, x1);
          v_sum2 = vec::add(v_sum2, x2);
          v_sum3 = vec::add(v_sum3, x3);

          v_sum_sq0 = vec::fmadd(x0, x0, v_sum_sq0);
          v_sum_sq1 = vec::fmadd(x1, x1, v_sum_sq1);
          v_sum_sq2 = vec::fmadd(x2, x2, v_sum_sq2);
          v_sum_sq3 = vec::fmadd(x3, x3, v_sum_sq3);
        }

        if (scan_col + step_int <= normalized_count_int) {
          auto xv = x(t, step, scan_col);
          v_sum0 = vec::add(v_sum0, xv);
          v_sum_sq0 = vec::fmadd(xv, xv, v_sum_sq0);
          scan_col += step_int;
        }
        if (scan_col + step_int <= normalized_count_int) {
          auto xv = x(t, step, scan_col);
          v_sum1 = vec::add(v_sum1, xv);
          v_sum_sq1 = vec::fmadd(xv, xv, v_sum_sq1);
          scan_col += step_int;
        }
        if (scan_col + step_int <= normalized_count_int) {
          auto xv = x(t, step, scan_col);
          v_sum2 = vec::add(v_sum2, xv);
          v_sum_sq2 = vec::fmadd(xv, xv, v_sum_sq2);
          scan_col += step_int;
        }
        if (scan_col < normalized_count_int) {
          const auto count = gemm::Any{normalized_count_int - scan_col};
          auto xv = x(t, count, scan_col);
          v_sum0 = vec::add(v_sum0, xv);
          v_sum_sq0 = vec::fmadd(xv, xv, v_sum_sq0);
        }

        v_sum0 = vec::add(v_sum0, v_sum1);
        v_sum0 = vec::add(v_sum0, v_sum2);
        v_sum0 = vec::add(v_sum0, v_sum3);
        v_sum_sq0 = vec::add(v_sum_sq0, v_sum_sq1);
        v_sum_sq0 = vec::add(v_sum_sq0, v_sum_sq2);
        v_sum_sq0 = vec::add(v_sum_sq0, v_sum_sq3);

        sum = vec::reduce_add(t, v_sum0);
        sum_sq = vec::reduce_add(t, v_sum_sq0);
      } else {
        std::tie(sum, sum_sq) = gemm::hop::scan<4>(
            normalized_count, step,
            [&](auto &&use) VECOPS_INLINE_LAMBDA -> decltype(auto) {
              auto v_sum = vec::zeros(t);
              auto v_sum_sq = vec::zeros(t);
              return use(v_sum, v_sum_sq);
            },
            [&](nint_t col, auto &&count, auto &v_sum, auto &v_sum_sq)
                VECOPS_INLINE_LAMBDA {
                  auto xv = x(t, count, col);
                  v_sum = vec::add(v_sum, xv);
                  v_sum_sq = vec::fmadd(xv, xv, v_sum_sq);
                },
            [&](auto &dst_sum, auto &dst_sum_sq, auto &src_sum,
                auto &src_sum_sq) VECOPS_INLINE_LAMBDA {
              dst_sum = vec::add(dst_sum, src_sum);
              dst_sum_sq = vec::add(dst_sum_sq, src_sum_sq);
            },
            [&](auto &v_sum, auto &v_sum_sq) {
              return std::make_pair(vec::reduce_add(t, v_sum),
                                    vec::reduce_add(t, v_sum_sq));
            });
      }

      const auto inv_n =
          ComputeType(1) / static_cast<ComputeType>(normalized_count);
      const auto mean = sum * inv_n;
      const auto variance =
          std::max(sum_sq * inv_n - mean * mean, ComputeType(0));
      const auto rstd = ComputeType(1) / std::sqrt(variance + config.eps);

      const auto mean_v = vec::fill(t, mean);
      const auto rstd_v = vec::fill(t, rstd);

      if constexpr (use_grouped_output) {
        nint_t col = 0;

        if constexpr (output_unroll == 4) {
          for (; col + output_group_step <= normalized_count_int;
               col += output_group_step) {
            auto x0 = x(t, step, col);
            auto x1 = x(t, step, col + step_int);
            auto x2 = x(t, step, col + 2 * step_int);
            auto x3 = x(t, step, col + 3 * step_int);

            auto gamma0 = gamma(t, step, col);
            auto gamma1 = gamma(t, step, col + step_int);
            auto gamma2 = gamma(t, step, col + 2 * step_int);
            auto gamma3 = gamma(t, step, col + 3 * step_int);

            auto beta0 = beta(t, step, col);
            auto beta1 = beta(t, step, col + step_int);
            auto beta2 = beta(t, step, col + 2 * step_int);
            auto beta3 = beta(t, step, col + 3 * step_int);

            auto centered0 = vec::sub(x0, mean_v);
            auto centered1 = vec::sub(x1, mean_v);
            auto centered2 = vec::sub(x2, mean_v);
            auto centered3 = vec::sub(x3, mean_v);

            auto normalized0 = vec::mul(centered0, rstd_v);
            auto normalized1 = vec::mul(centered1, rstd_v);
            auto normalized2 = vec::mul(centered2, rstd_v);
            auto normalized3 = vec::mul(centered3, rstd_v);

            auto affine0 = vec::fmadd(normalized0, gamma0, beta0);
            auto affine1 = vec::fmadd(normalized1, gamma1, beta1);
            auto affine2 = vec::fmadd(normalized2, gamma2, beta2);
            auto affine3 = vec::fmadd(normalized3, gamma3, beta3);

            y(t, affine0, step, col);
            y(t, affine1, step, col + step_int);
            y(t, affine2, step, col + 2 * step_int);
            y(t, affine3, step, col + 3 * step_int);
          }
        } else if constexpr (output_unroll == 3) {
          for (; col + output_group_step <= normalized_count_int;
               col += output_group_step) {
            auto x0 = x(t, step, col);
            auto x1 = x(t, step, col + step_int);
            auto x2 = x(t, step, col + 2 * step_int);
            auto gamma0 = gamma(t, step, col);
            auto gamma1 = gamma(t, step, col + step_int);
            auto gamma2 = gamma(t, step, col + 2 * step_int);
            auto beta0 = beta(t, step, col);
            auto beta1 = beta(t, step, col + step_int);
            auto beta2 = beta(t, step, col + 2 * step_int);
            auto centered0 = vec::sub(x0, mean_v);
            auto centered1 = vec::sub(x1, mean_v);
            auto centered2 = vec::sub(x2, mean_v);
            auto normalized0 = vec::mul(centered0, rstd_v);
            auto normalized1 = vec::mul(centered1, rstd_v);
            auto normalized2 = vec::mul(centered2, rstd_v);
            auto affine0 = vec::fmadd(normalized0, gamma0, beta0);
            auto affine1 = vec::fmadd(normalized1, gamma1, beta1);
            auto affine2 = vec::fmadd(normalized2, gamma2, beta2);
            y(t, affine0, step, col);
            y(t, affine1, step, col + step_int);
            y(t, affine2, step, col + 2 * step_int);
          }
        } else {
          static_assert(output_unroll == 2);
          for (; col + output_group_step <= normalized_count_int;
               col += output_group_step) {
            auto x0 = x(t, step, col);
            auto x1 = x(t, step, col + step_int);
            auto gamma0 = gamma(t, step, col);
            auto gamma1 = gamma(t, step, col + step_int);
            auto beta0 = beta(t, step, col);
            auto beta1 = beta(t, step, col + step_int);
            auto centered0 = vec::sub(x0, mean_v);
            auto centered1 = vec::sub(x1, mean_v);
            auto normalized0 = vec::mul(centered0, rstd_v);
            auto normalized1 = vec::mul(centered1, rstd_v);
            auto affine0 = vec::fmadd(normalized0, gamma0, beta0);
            auto affine1 = vec::fmadd(normalized1, gamma1, beta1);
            y(t, affine0, step, col);
            y(t, affine1, step, col + step_int);
          }
        }

        for (; col + step_int <= normalized_count_int; col += step_int) {
          auto xv = x(t, step, col);
          auto gamma_v = gamma(t, step, col);
          auto beta_v = beta(t, step, col);
          auto centered = vec::sub(xv, mean_v);
          auto normalized = vec::mul(centered, rstd_v);
          auto affine = vec::fmadd(normalized, gamma_v, beta_v);
          y(t, affine, step, col);
        }

        if (col < normalized_count_int) {
          const auto count = gemm::Any{normalized_count_int - col};
          auto xv = x(t, count, col);
          auto gamma_v = gamma(t, count, col);
          auto beta_v = beta(t, count, col);
          auto centered = vec::sub(xv, mean_v);
          auto normalized = vec::mul(centered, rstd_v);
          auto affine = vec::fmadd(normalized, gamma_v, beta_v);
          y(t, affine, count, col);
        }
      } else {
        gemm::hop::map<4>(
            normalized_count, step, [&](nint_t col, auto &&count) {
              auto xv = x(t, count, col);
              auto gamma_v = gamma(t, count, col);
              auto beta_v = beta(t, count, col);
              auto centered = vec::sub(xv, mean_v);
              auto normalized = vec::mul(centered, rstd_v);
              auto affine = vec::fmadd(normalized, gamma_v, beta_v);
              y(t, affine, count, col);
            });
      }
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_impl(const InSpec &in,
                                               const ScaleSpec &scale,
                                               const BiasSpec &bias,
                                               const OutSpec &out) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int prefix_rank = InLayout::Ndim - 1;
    const auto in_row = row_input_spec<prefix_rank>(in);
    const auto out_row = row_output_spec<prefix_rank>(out);
    const nint_t row_workspace = gemm::required_workspace(in_row, out_row);
    return gemm::details::workspace_round_up(
        gemm::required_workspace(scale, bias) + row_workspace,
        vec::DEFAULT_ALIGNMENT);
  }

  template <int PrefixRank, typename Layout>
  VECOPS_INLINE static auto row_layout(const Layout& layout) {
    if constexpr (PrefixRank == 0) {
      return layout;
    } else {
      return row_layout<PrefixRank - 1>(gemm::remove<0>(layout));
    }
  }

  template <int PrefixRank, typename Spec>
  VECOPS_INLINE static auto row_input_spec(const Spec& spec) {
    using SpecT = std::remove_cvref_t<Spec>;
    auto layout = row_layout<PrefixRank>(spec.input_layout());
    using RowLayout = std::remove_cvref_t<decltype(layout)>;
    return gemm::InputSpec<
        typename SpecT::OutputElement,
        typename SpecT::InputTensor::ElementType,
        RowLayout,
        typename SpecT::Transform>(layout, spec.transform());
  }

  template <int PrefixRank, typename Spec>
  VECOPS_INLINE static auto row_output_spec(const Spec& spec) {
    using SpecT = std::remove_cvref_t<Spec>;
    auto layout = row_layout<PrefixRank>(spec.output_layout());
    using RowLayout = std::remove_cvref_t<decltype(layout)>;
    using TOut = typename SpecT::OutputTensor::ElementType;
    return gemm::OutputSpec<
        typename SpecT::InputElement,
        TOut,
        RowLayout,
        typename SpecT::Transform>(layout, spec.transform());
  }
};

template <typename Config = LayerNormConfig<>>
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
