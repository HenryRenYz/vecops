//
// Created by renyz on 2026/7/10.
//

#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/gemm/HOP.h"
#include "vecops/gemm/Tensor.h"
#include "vecops/gemm/Workspace.h"
#include "vecops/vec/Vec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace vecops::ops {

template <
    typename ComputeT = float32_t,
    typename VecTag = vec::ScalableTag<ComputeT, 0>,
    vec::Accuracy ExpAccuracy = vec::Accuracy::Strict>
struct SoftmaxConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;

  /** Accuracy selected for the exponential stage of softmax. */
  static constexpr vec::Accuracy exp_accuracy = ExpAccuracy;

  // Allow the operator to select an online normalizer when it is expected to
  // be faster.  False always preserves the regular implementation.
  bool allow_online = true;
};

namespace details {

template <typename Config>
using softmax_compute_t = typename std::remove_cvref_t<Config>::ComputeType;

template <typename Config>
using softmax_tag_t = typename std::remove_cvref_t<Config>::Tag;

template <typename Config>
VECOPS_INLINE constexpr bool softmax_online_allowed(const Config& config) {
  if constexpr (requires { config.allow_online; }) {
    return static_cast<bool>(config.allow_online);
  } else {
    // Keep existing custom Config types source-compatible.  Like the default
    // config, a legacy config permits automatic online selection.
    return true;
  }
}

template <typename Tag>
VECOPS_INLINE auto softmax_vector_step_value(Tag tag) {
  using TagT = std::remove_cvref_t<Tag>;
  if constexpr (vec::is_runtime_size<TagT>) {
    return gemm::Any{vec::size(tag)};
  } else {
    return gemm::Const<vec::size(TagT{})>{};
  }
}

template <typename InLayout, typename OutLayout>
VECOPS_INLINE void validate_softmax_layouts(
    const InLayout& in,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1, "Softmax input rank must be at least 1");
  static_assert(
      OutLayout::Ndim == InLayout::Ndim,
      "Softmax output rank must match input rank");

  constexpr int rank = InLayout::Ndim;
  for (int d = 0; d < rank; ++d) {
    VECOPS_ASSERT(
        in.shape()[d] == out.shape()[d],
        "Softmax input/output shape mismatch at dim %d", d);
  }
  VECOPS_ASSERT(
      in.shape()[rank - 1] > 0,
      "Softmax normalized dimension must be non-empty");
}

} // namespace details

template <typename Config = SoftmaxConfig<>>
struct Softmax {
  using ConfigType = std::remove_cvref_t<Config>;
  using ComputeType = details::softmax_compute_t<Config>;
  using Tag = details::softmax_tag_t<Config>;

  static constexpr vec::Accuracy ExpAccuracy = ConfigType::exp_accuracy;

public:
  const Config config;

  VECOPS_INLINE constexpr explicit Softmax(Config cfg = {}) : config(cfg) {}

  template <
      typename InSpec,
      typename OutSpec,
      std::enable_if_t<
          gemm::is_input_spec_v<InSpec> && gemm::is_output_spec_v<OutSpec>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "Softmax input must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "Softmax output must be an OutputSpec");
    details::validate_softmax_layouts(in.input_layout(), out.output_layout());
    return required_workspace_impl(in, out);
  }

  template <
      typename InLayout,
      typename OutLayout,
      std::enable_if_t<
          gemm::is_layout<std::remove_cvref_t<InLayout>> &&
          gemm::is_layout<std::remove_cvref_t<OutLayout>>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InLayout& in,
      const OutLayout& out) const {
    return required_workspace(
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<InLayout>>(in),
        gemm::OutputSpec<ComputeType, ComputeType, std::remove_cvref_t<OutLayout>>(out));
  }

  template <typename InSpec, typename OutSpec>
  void operator()(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "Softmax input must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "Softmax output must be an OutputSpec");

    const auto& in_layout = in.input_layout();
    const auto& out_layout = out.output_layout();
    details::validate_softmax_layouts(in_layout, out_layout);

    auto mark = workspace.mark();
    {
      constexpr int prefix_rank = std::remove_cvref_t<decltype(in_layout)>::Ndim - 1;
#if defined(CPU_CAPABILITY_SVE)
      const bool use_online = should_use_online(in);
      const auto run_full_rows = [&]<bool PackedBf16Output>() VECOPS_INLINE_LAMBDA {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace](const auto& in_row, const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              this->template run_row_full<PackedBf16Output>(
                  workspace, in_row, out_row);
            },
            in,
            out);
      };
      const auto run_online_rows = [&]<bool PackedBf16Output>() VECOPS_INLINE_LAMBDA {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace](const auto& in_row, const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              this->template run_row_online<PackedBf16Output>(
                  workspace, in_row, out_row);
            },
            in,
            out);
      };
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
      using OutSpecT = std::remove_cvref_t<OutSpec>;
      using OutElement = typename OutSpecT::OutputTensor::ElementType;
      constexpr bool tensor_backed =
          requires(const OutSpecT& spec) { spec.tensor(); };
      constexpr bool can_pack_bf16_output =
          tensor_backed &&
          std::is_same_v<ComputeType, float32_t> &&
          std::is_same_v<Tag, vec::ScalableTag<float32_t, 0>> &&
          std::is_same_v<OutElement, bfloat16_t> &&
          std::is_same_v<
              typename OutSpecT::Transform,
              IdentityVecTransform<bfloat16_t, float32_t>> &&
          gemm::details::IsLastDimContiguous<
              typename OutSpecT::OutputLayout>::value;
      if constexpr (can_pack_bf16_output) {
        nint_t element_count = 1;
        for (int d = 0; d <= prefix_rank; ++d) {
          element_count *= in_layout.shape()[d];
        }
        // Cache-resident SVE BF16 prefers two narrow stores.  Once the whole
        // tensor leaves cache, pack two FP32 vectors into one full-width store.
        if (element_count > 64 * 1024) {
          if (use_online) {
            run_online_rows.template operator()<true>();
          } else {
            run_full_rows.template operator()<true>();
          }
        } else {
          if (use_online) {
            run_online_rows.template operator()<false>();
          } else {
            run_full_rows.template operator()<false>();
          }
        }
      } else {
        if (use_online) {
          run_online_rows.template operator()<false>();
        } else {
          run_full_rows.template operator()<false>();
        }
      }
#else
      if (use_online) {
        run_online_rows.template operator()<false>();
      } else {
        run_full_rows.template operator()<false>();
      }
#endif
#elif defined(CPU_CAPABILITY_AVX512)
      const nint_t normalized_n = in_layout.shape()[prefix_rank];
      const nint_t step = static_cast<nint_t>(
          details::softmax_vector_step_value(Tag{}));
      if (should_use_online(in)) {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace](const auto& in_row, const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              this->run_row_online(workspace, in_row, out_row);
            },
            in,
            out);
      } else if (normalized_n >= 16 * step && normalized_n <= 1024) {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace](const auto& in_row, const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              this->run_row_unrolled(workspace, in_row, out_row);
            },
            in,
            out);
      } else {
        gemm::hop::for_each_dims<prefix_rank>(
            [this, &workspace](const auto& in_row, const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              this->run_row_hop(workspace, in_row, out_row);
            },
            in,
            out);
      }
#else
      gemm::hop::for_each_dims<prefix_rank>(
          [this, &workspace](const auto& in_row, const auto& out_row) VECOPS_INLINE_LAMBDA {
            this->run_row_hop(workspace, in_row, out_row);
          },
          in,
          out);
#endif
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void operator()(const InSpec& in, const OutSpec& out) const {
    gemm::Workspace workspace(required_workspace(in, out));
    auto view = workspace.view();
    (*this)(view, in, out);
  }

private:
  template <typename InSpec>
  VECOPS_INLINE bool should_use_online(const InSpec& in) const {
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
    if (!details::softmax_online_allowed(config)) return false;
    using InElement =
        typename std::remove_cvref_t<InSpec>::InputTensor::ElementType;
    const auto& layout = in.input_layout();
    constexpr int rank =
        std::remove_cvref_t<decltype(layout)>::Ndim;
    const nint_t normalized_n = layout.shape()[rank - 1];
    nint_t element_count = 1;
    for (int d = 0; d < rank; ++d) {
      element_count *= layout.shape()[d];
    }
#if defined(CPU_CAPABILITY_SVE)
    // The block-online path wins in the middle working-set range on 920f.
    // Longer rows put the exp cache and metadata beyond the profitable cache
    // regime, while shorter rows do not amortize the metadata reduction.
    if constexpr (
        std::is_same_v<ComputeType, float32_t> &&
        std::is_same_v<InElement, float32_t> &&
        ExpAccuracy != vec::Accuracy::Estimate) {
      return element_count >= 512 * 1024 &&
          element_count <= 8 * 1024 * 1024 && normalized_n >= 8192 &&
          normalized_n <= 16384;
    } else {
      return false;
    }
#else
    if constexpr (
        std::is_same_v<ComputeType, float32_t> &&
        std::is_same_v<InElement, float32_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Strict) {
        return element_count >= 512 * 1024 &&
            normalized_n >= 2048 && normalized_n <= 4096;
      } else {
        return element_count >= 512 * 1024 &&
            normalized_n >= 2048 && normalized_n <= 32768;
      }
    } else if constexpr (
        std::is_same_v<ComputeType, float64_t> &&
        std::is_same_v<InElement, float64_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return element_count >= 512 * 1024 && normalized_n >= 2048;
      } else {
        return element_count >= 16 * 1024 * 1024 &&
            normalized_n >= 2048;
      }
    } else if constexpr (
        std::is_same_v<ComputeType, float32_t> &&
        std::is_same_v<InElement, float16_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return element_count >= 16 * 1024 * 1024 &&
            normalized_n == 2048;
      } else {
        return false;
      }
    } else {
      return false;
    }
#endif
#else
    (void)in;
    return false;
#endif
  }

  template <typename TagT, typename V, typename M>
  VECOPS_INLINE static V apply_exp(TagT tag, V v, M mask, V default_v) {
    (void)tag;
    return vec::exp_neg(
        v,
        vec::opt::math::accuracy<ExpAccuracy>,
        vec::opt::masked(mask),
        vec::opt::merge(default_v));
  }

#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
  template <typename V>
  VECOPS_INLINE static V apply_exp_unmasked(V v) {
    return vec::exp_neg(v, vec::opt::math::accuracy<ExpAccuracy>);
  }
#endif

#if defined(CPU_CAPABILITY_SVE)
  template <bool PackedBf16Output = false, typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row_full(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    const auto& in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "Softmax row spec must be rank 1");

    Tag t;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const nint_t normalized_n = normalized_count;
      ComputeType* exp_cache = workspace.allocate<ComputeType>(normalized_n);
      const auto step = details::softmax_vector_step_value(t);
      const nint_t step_int = static_cast<nint_t>(step);
      const nint_t full_end = normalized_n - step_int;
      constexpr gemm::FullVectorCount full_vector_count{};
      const auto negative_infinity =
          static_cast<ComputeType>(-std::numeric_limits<double>::infinity());

      auto v_max = vec::fill(t, negative_infinity);
      nint_t col = 0;
      for (; col <= full_end; col += step_int) {
        auto xv = x(t, full_vector_count, col);
        v_max = vec::max(v_max, xv);
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto xv = x(t, gemm::Any{tail_count}, col);
        auto mask = vec::mwhilelt(t, 0, tail_count);
        v_max = vec::max(v_max, xv, vec::opt::masked(mask));
      }
      const auto max_value = vec::reduce_max(t, v_max);
      const auto max_v = vec::fill(t, max_value);

      auto v_sum = vec::zeros(t);
      col = 0;
      for (; col <= full_end; col += step_int) {
        auto xv = x(t, full_vector_count, col);
        auto exp_v = apply_exp_unmasked(vec::sub(xv, max_v));
        vec::store(t, exp_cache + col, exp_v);
        v_sum = vec::add(v_sum, exp_v);
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto xv = x(t, gemm::Any{tail_count}, col);
        auto mask = vec::mwhilelt(t, 0, tail_count);
        auto exp_v = apply_exp(
            t, vec::sub(xv, max_v), mask, vec::zeros(t));
        vec::store(t, exp_cache + col, exp_v, vec::opt::masked(mask));
        v_sum = vec::add(v_sum, exp_v);
      }
      const auto inv_sum = ComputeType(1) / vec::reduce_add(t, v_sum);
      const auto inv_sum_v = vec::fill(t, inv_sum);

      col = 0;
      const nint_t output_group_step = 2 * step_int;
      const nint_t output_group_end = normalized_n - output_group_step;
      if constexpr (PackedBf16Output) {
        using OutElement =
            typename std::remove_cvref_t<OutSpec>::OutputTensor::ElementType;
        static_assert(std::is_same_v<OutElement, bfloat16_t>);
        using PairTag = vec::Twice<Tag>;
        constexpr PairTag pair_tag{};
        OutElement* out_pointer =
            const_cast<OutElement*>(out.tensor().data());
        for (; col <= output_group_end;
             col += output_group_step) {
          const auto out0 =
              vec::mul(vec::load(t, exp_cache + col), inv_sum_v);
          const auto out1 = vec::mul(
              vec::load(t, exp_cache + col + step_int), inv_sum_v);
          vec::store_convert(
              pair_tag,
              out_pointer + col,
              vec::concat(pair_tag, out0, out1));
        }
        if (col <= full_end) {
          vec::store_convert(
              t,
              out_pointer + col,
              vec::mul(vec::load(t, exp_cache + col), inv_sum_v));
          col += step_int;
        }
      } else {
        for (; col <= output_group_end;
             col += output_group_step) {
          y(t,
            vec::mul(vec::load(t, exp_cache + col), inv_sum_v),
            full_vector_count,
            col);
          y(t,
            vec::mul(vec::load(t, exp_cache + col + step_int), inv_sum_v),
            full_vector_count,
            col + step_int);
        }
        if (col <= full_end) {
          auto exp_v = vec::load(t, exp_cache + col);
          y(t, vec::mul(exp_v, inv_sum_v), full_vector_count, col);
          col += step_int;
        }
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto mask = vec::mwhilelt(t, 0, tail_count);
        auto exp_v = vec::load(
            t, exp_cache + col, vec::opt::masked(mask));
        y(t, vec::mul(exp_v, inv_sum_v), gemm::Any{tail_count}, col);
      }
    }
    workspace.rewind(mark);
  }
#endif

#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
  template <
      bool PackedBf16Output = false,
      typename InSpec,
      typename OutSpec>
  VECOPS_NOINLINE void run_row_online(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    const auto& in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "Softmax row spec must be rank 1");

    Tag t;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const nint_t normalized_n = gemm::size<0>(in_layout);
      const nint_t step = static_cast<nint_t>(
          details::softmax_vector_step_value(t));
      const nint_t tile_step = 4 * step;
      const nint_t tile_count =
          (normalized_n + tile_step - 1) / tile_step;
      constexpr gemm::FullVectorCount full_vector_count{};
      ComputeType* exp_cache = workspace.allocate<ComputeType>(normalized_n);
      ComputeType* tile_max_cache =
          workspace.allocate<ComputeType>(tile_count);
      ComputeType* tile_sum_cache =
          workspace.allocate<ComputeType>(tile_count);
      const ComputeType negative_infinity = static_cast<ComputeType>(
          -std::numeric_limits<double>::infinity());

      nint_t col = 0;
      nint_t tile = 0;
      for (; col + tile_step <= normalized_n;
           col += tile_step, ++tile) {
        const auto x0 = x(t, full_vector_count, col);
        const auto x1 = x(t, full_vector_count, col + step);
        const auto x2 = x(t, full_vector_count, col + 2 * step);
        const auto x3 = x(t, full_vector_count, col + 3 * step);
        const auto tile_max_v = vec::max(
            vec::max(x0, x1), vec::max(x2, x3));
        const ComputeType tile_max = vec::reduce_max(t, tile_max_v);
        tile_max_cache[tile] = tile_max;
        const auto max_v = vec::fill(t, tile_max);
        const auto exp0 = apply_exp_unmasked(vec::sub(x0, max_v));
        const auto exp1 = apply_exp_unmasked(vec::sub(x1, max_v));
        const auto exp2 = apply_exp_unmasked(vec::sub(x2, max_v));
        const auto exp3 = apply_exp_unmasked(vec::sub(x3, max_v));
        vec::store(t, exp_cache + col, exp0);
        vec::store(t, exp_cache + col + step, exp1);
        vec::store(t, exp_cache + col + 2 * step, exp2);
        vec::store(t, exp_cache + col + 3 * step, exp3);
        tile_sum_cache[tile] = vec::reduce_add(
            t,
            vec::add(
                vec::add(exp0, exp1),
                vec::add(exp2, exp3)));
      }

      if (col < normalized_n) {
        const nint_t remaining = normalized_n - col;
        auto tile_max_v = vec::fill(t, negative_infinity);
        nint_t local = 0;
        for (; local + step <= remaining; local += step) {
          tile_max_v = vec::max(
              tile_max_v,
              x(t, full_vector_count, col + local));
        }
        if (local < remaining) {
          const nint_t tail_count = remaining - local;
          const auto mask = vec::mwhilelt(t, 0, tail_count);
          tile_max_v = vec::max(
              tile_max_v,
              x(t, gemm::Any{tail_count}, col + local),
              vec::opt::masked(mask));
        }

        const ComputeType tile_max = vec::reduce_max(t, tile_max_v);
        tile_max_cache[tile] = tile_max;
        auto tile_sum_v = vec::zeros(t);
        local = 0;
        const auto max_v = vec::fill(t, tile_max);
        for (; local + step <= remaining; local += step) {
          const auto exp_v = apply_exp_unmasked(vec::sub(
              x(t, full_vector_count, col + local),
              max_v));
          vec::store(t, exp_cache + col + local, exp_v);
          tile_sum_v = vec::add(tile_sum_v, exp_v);
        }
        if (local < remaining) {
          const nint_t tail_count = remaining - local;
          const auto mask = vec::mwhilelt(t, 0, tail_count);
          const auto exp_v = apply_exp(
              t,
              vec::sub(
                  x(t, gemm::Any{tail_count}, col + local),
                  max_v),
              mask,
              vec::zeros(t));
          vec::store(
              t,
              exp_cache + col + local,
              exp_v,
              vec::opt::masked(mask));
          tile_sum_v = vec::add(tile_sum_v, exp_v);
        }
        tile_sum_cache[tile] = vec::reduce_add(t, tile_sum_v);
        ++tile;
      }

      auto metadata_max_v = vec::fill(t, negative_infinity);
      tile = 0;
      for (; tile + step <= tile_count; tile += step) {
        metadata_max_v = vec::max(
            metadata_max_v,
            vec::load(t, tile_max_cache + tile));
      }
      if (tile < tile_count) {
        const nint_t tail_count = tile_count - tile;
        const auto mask = vec::mwhilelt(t, 0, tail_count);
        metadata_max_v = vec::max(
            metadata_max_v,
            vec::load(
                t,
                tile_max_cache + tile,
                vec::opt::masked(mask)),
            vec::opt::masked(mask));
      }
      const ComputeType global_max = vec::reduce_max(t, metadata_max_v);

      const auto run_regular_fallback = [&]() VECOPS_INLINE_LAMBDA {
        workspace.rewind(mark);
#if defined(CPU_CAPABILITY_SVE)
        this->template run_row_full<PackedBf16Output>(
            workspace, in, out);
#else
        if (normalized_n >= 16 * step && normalized_n <= 1024) {
          this->run_row_unrolled(workspace, in, out);
        } else {
          this->run_row_hop(workspace, in, out);
        }
#endif
      };
      if (!std::isfinite(static_cast<double>(global_max))) {
        run_regular_fallback();
        return;
      }

      const auto global_max_v = vec::fill(t, global_max);
      auto metadata_sum_v = vec::zeros(t);
      tile = 0;
      for (; tile + step <= tile_count; tile += step) {
        const auto scale_v = apply_exp_unmasked(vec::sub(
            vec::load(t, tile_max_cache + tile),
            global_max_v));
        vec::store(t, tile_max_cache + tile, scale_v);
        metadata_sum_v = vec::fmadd(
            scale_v,
            vec::load(t, tile_sum_cache + tile),
            metadata_sum_v);
      }
      if (tile < tile_count) {
        const nint_t tail_count = tile_count - tile;
        const auto mask = vec::mwhilelt(t, 0, tail_count);
        const auto scale_v = apply_exp(
            t,
            vec::sub(
                vec::load(
                    t,
                    tile_max_cache + tile,
                    vec::opt::masked(mask)),
                global_max_v),
            mask,
            vec::zeros(t));
        vec::store(
            t,
            tile_max_cache + tile,
            scale_v,
            vec::opt::masked(mask));
        metadata_sum_v = vec::fmadd(
            scale_v,
            vec::load(
                t,
                tile_sum_cache + tile,
                vec::opt::masked(mask)),
            metadata_sum_v);
      }
      const ComputeType global_sum = vec::reduce_add(t, metadata_sum_v);
      if (!std::isfinite(static_cast<double>(global_sum)) ||
          !(global_sum > ComputeType(0))) {
        run_regular_fallback();
        return;
      }
      const ComputeType inv_sum = ComputeType(1) / global_sum;

      col = 0;
      for (tile = 0; tile < tile_count; ++tile) {
        const nint_t count = std::min(tile_step, normalized_n - col);
        const auto scale_v = vec::fill(
            t, tile_max_cache[tile] * inv_sum);
        nint_t local = 0;
#if defined(CPU_CAPABILITY_SVE)
        if constexpr (PackedBf16Output) {
          using OutElement =
              typename std::remove_cvref_t<OutSpec>::OutputTensor::ElementType;
          static_assert(std::is_same_v<OutElement, bfloat16_t>);
          using PairTag = vec::Twice<Tag>;
          constexpr PairTag pair_tag{};
          OutElement* out_pointer =
              const_cast<OutElement*>(out.tensor().data());
          for (; local + 2 * step <= count; local += 2 * step) {
            const auto out0 = vec::mul(
                vec::load(t, exp_cache + col + local),
                scale_v);
            const auto out1 = vec::mul(
                vec::load(t, exp_cache + col + local + step),
                scale_v);
            vec::store_convert(
                pair_tag,
                out_pointer + col + local,
                vec::concat(pair_tag, out0, out1));
          }
          if (local + step <= count) {
            vec::store_convert(
                t,
                out_pointer + col + local,
                vec::mul(
                    vec::load(t, exp_cache + col + local),
                    scale_v));
            local += step;
          }
        } else
#endif
        {
          for (; local + step <= count; local += step) {
            y(t,
              vec::mul(
                  vec::load(t, exp_cache + col + local),
                  scale_v),
              full_vector_count,
              col + local);
          }
        }
        if (local < count) {
          const nint_t tail_count = count - local;
          const auto mask = vec::mwhilelt(t, 0, tail_count);
          y(t,
            vec::mul(
                vec::load(
                    t,
                    exp_cache + col + local,
                    vec::opt::masked(mask)),
                scale_v),
            gemm::Any{tail_count},
            col + local);
        }
        col += count;
      }
    }
    workspace.rewind(mark);
  }
#endif

#if defined(CPU_CAPABILITY_AVX512)
  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row_unrolled(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    const auto& in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "Softmax row spec must be rank 1");

    Tag t;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const nint_t normalized_n = gemm::size<0>(in_layout);
      ComputeType* exp_cache = workspace.allocate<ComputeType>(normalized_n);
      const nint_t step = static_cast<nint_t>(
          details::softmax_vector_step_value(t));
      constexpr gemm::FullVectorCount full_vector_count{};
      const auto negative_infinity =
          static_cast<ComputeType>(-std::numeric_limits<double>::infinity());

      auto v_max0 = vec::fill(t, negative_infinity);
      auto v_max1 = v_max0;
      auto v_max2 = v_max0;
      auto v_max3 = v_max0;
      nint_t col = 0;
      for (; col + 4 * step <= normalized_n; col += 4 * step) {
        v_max0 = vec::max(v_max0, x(t, full_vector_count, col));
        v_max1 = vec::max(v_max1, x(t, full_vector_count, col + step));
        v_max2 = vec::max(v_max2, x(t, full_vector_count, col + 2 * step));
        v_max3 = vec::max(v_max3, x(t, full_vector_count, col + 3 * step));
      }
      for (; col + step <= normalized_n; col += step) {
        v_max0 = vec::max(v_max0, x(t, full_vector_count, col));
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto mask = vec::mwhilelt(t, 0, tail_count);
        v_max0 = vec::max(
            v_max0,
            x(t, gemm::Any{tail_count}, col),
            vec::opt::masked(mask));
      }
      const auto v_max = vec::max(
          vec::max(v_max0, v_max1), vec::max(v_max2, v_max3));
      const auto max_value = vec::reduce_max(t, v_max);
      const auto max_v = vec::fill(t, max_value);

      auto v_sum0 = vec::zeros(t);
      auto v_sum1 = v_sum0;
      auto v_sum2 = v_sum0;
      auto v_sum3 = v_sum0;
      col = 0;
      for (; col + 4 * step <= normalized_n; col += 4 * step) {
        auto exp0 = apply_exp_unmasked(
            vec::sub(x(t, full_vector_count, col), max_v));
        auto exp1 = apply_exp_unmasked(
            vec::sub(x(t, full_vector_count, col + step), max_v));
        auto exp2 = apply_exp_unmasked(
            vec::sub(x(t, full_vector_count, col + 2 * step), max_v));
        auto exp3 = apply_exp_unmasked(
            vec::sub(x(t, full_vector_count, col + 3 * step), max_v));
        vec::store(t, exp_cache + col, exp0);
        vec::store(t, exp_cache + col + step, exp1);
        vec::store(t, exp_cache + col + 2 * step, exp2);
        vec::store(t, exp_cache + col + 3 * step, exp3);
        v_sum0 = vec::add(v_sum0, exp0);
        v_sum1 = vec::add(v_sum1, exp1);
        v_sum2 = vec::add(v_sum2, exp2);
        v_sum3 = vec::add(v_sum3, exp3);
      }
      for (; col + step <= normalized_n; col += step) {
        auto exp_v = apply_exp_unmasked(
            vec::sub(x(t, full_vector_count, col), max_v));
        vec::store(t, exp_cache + col, exp_v);
        v_sum0 = vec::add(v_sum0, exp_v);
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto mask = vec::mwhilelt(t, 0, tail_count);
        auto exp_v = apply_exp(
            t,
            vec::sub(x(t, gemm::Any{tail_count}, col), max_v),
            mask,
            vec::zeros(t));
        vec::store(t, exp_cache + col, exp_v, vec::opt::masked(mask));
        v_sum0 = vec::add(v_sum0, exp_v);
      }
      const auto v_sum = vec::add(
          vec::add(v_sum0, v_sum1), vec::add(v_sum2, v_sum3));
      const auto inv_sum = ComputeType(1) / vec::reduce_add(t, v_sum);
      const auto inv_sum_v = vec::fill(t, inv_sum);

      col = 0;
      for (; col + 4 * step <= normalized_n; col += 4 * step) {
        y(t,
          vec::mul(vec::load(t, exp_cache + col), inv_sum_v),
          full_vector_count,
          col);
        y(t,
          vec::mul(vec::load(t, exp_cache + col + step), inv_sum_v),
          full_vector_count,
          col + step);
        y(t,
          vec::mul(vec::load(t, exp_cache + col + 2 * step), inv_sum_v),
          full_vector_count,
          col + 2 * step);
        y(t,
          vec::mul(vec::load(t, exp_cache + col + 3 * step), inv_sum_v),
          full_vector_count,
          col + 3 * step);
      }
      for (; col + step <= normalized_n; col += step) {
        y(t,
          vec::mul(vec::load(t, exp_cache + col), inv_sum_v),
          full_vector_count,
          col);
      }
      if (col < normalized_n) {
        const nint_t tail_count = normalized_n - col;
        auto mask = vec::mwhilelt(t, 0, tail_count);
        y(t,
          vec::mul(
              vec::load(t, exp_cache + col, vec::opt::masked(mask)),
              inv_sum_v),
          gemm::Any{tail_count},
          col);
      }
    }
    workspace.rewind(mark);
  }
#endif

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row_hop(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    const auto& in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "Softmax row spec must be rank 1");

    Tag t;
    using VecT = vec::Vec<Tag>;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const nint_t normalized_n = normalized_count;
      ComputeType* exp_cache = workspace.allocate<ComputeType>(normalized_n);
      const auto step = details::softmax_vector_step_value(t);
      const auto negative_infinity =
          static_cast<ComputeType>(-std::numeric_limits<double>::infinity());

      const auto v_max = gemm::hop::scan<1>(
          vec::fill(t, negative_infinity),
          normalized_count,
          step,
          [&](VecT acc, nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto xv = x(t, count, col);
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            return vec::max(acc, xv, vec::opt::masked(mask));
          });
      const auto max_value = vec::reduce_max(t, v_max);
      const auto max_v = vec::fill(t, max_value);

      const auto v_sum = gemm::hop::scan<1>(
          vec::zeros(t),
          normalized_count,
          step,
          [&](VecT acc, nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto xv = x(t, count, col);
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            auto shifted = vec::sub(xv, max_v);
            auto zero_v = vec::zeros(t);
            auto exp_v = apply_exp(t, shifted, mask, zero_v);
            vec::store(t, exp_cache + col, exp_v, vec::opt::masked(mask));
            return vec::add(acc, exp_v);
          });
      const auto inv_sum = ComputeType(1) / vec::reduce_add(t, v_sum);
      const auto inv_sum_v = vec::fill(t, inv_sum);

      // Loop unrolling 2
      gemm::hop::map<2>(
          normalized_count,
          step,
          [&](nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            auto exp_v = vec::load(
                t, exp_cache + col, vec::opt::masked(mask));
            y(t, vec::mul(exp_v, inv_sum_v), count, col);
          });
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_impl(
      const InSpec& in,
      const OutSpec& out) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int prefix_rank = InLayout::Ndim - 1;
    const auto in_row = row_input_spec<prefix_rank>(in);
    const auto out_row = row_output_spec<prefix_rank>(out);
    const nint_t row_workspace = gemm::required_workspace(in_row, out_row);
    const nint_t normalized_count = gemm::size<0>(in_row.input_layout());
    const nint_t exp_cache_bytes =
        normalized_count * static_cast<nint_t>(sizeof(ComputeType));
    nint_t online_metadata_bytes = 0;
    if (should_use_online(in)) {
      const nint_t step = vec::size(Tag{});
      const nint_t tile_step = 4 * step;
      const nint_t tile_count =
          (normalized_count + tile_step - 1) / tile_step;
      online_metadata_bytes =
          2 * tile_count * static_cast<nint_t>(sizeof(ComputeType));
    }
    return gemm::details::workspace_round_up(
        row_workspace + exp_cache_bytes + online_metadata_bytes,
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

template <typename Config = SoftmaxConfig<>>
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_SOFTMAX_H
