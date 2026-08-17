#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

namespace vecops::ops {

template <typename ComputeT = float32_t,
          typename VecTag = vec::ScalableTag<ComputeT, 0>,
          vec::Accuracy ExpAccuracy = vec::Accuracy::Strict>
struct SoftmaxConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;
  static constexpr vec::Accuracy exp_accuracy = ExpAccuracy;
  bool allow_online = true;
};

namespace details {

template <typename Config>
VECOPS_INLINE constexpr bool softmax_online_allowed(const Config& config) {
  if constexpr (requires { config.allow_online; }) {
    return static_cast<bool>(config.allow_online);
  } else {
    return true;
  }
}

template <typename InLayout, typename OutLayout>
VECOPS_INLINE void validate_softmax_layouts(
    const InLayout& in, const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1);
  static_assert(OutLayout::Ndim == InLayout::Ndim);
  for (int d = 0; d < InLayout::Ndim; ++d) {
    VECOPS_ASSERT(in.shape()[d] == out.shape()[d],
                  "Softmax input/output shape mismatch");
  }
  VECOPS_ASSERT(in.shape()[InLayout::Ndim - 1] > 0,
                "Softmax normalized dimension must be non-empty");
}

template <int Count, typename Spec>
VECOPS_INLINE auto softmax_row_spec(const Spec& spec) {
  if constexpr (Count == 0) return spec;
  else return softmax_row_spec<Count - 1>(tensor::slice_view<0>(spec, 0));
}

} // namespace details

template <typename Config = SoftmaxConfig<>>
class Softmax {
public:
  using ConfigType = std::remove_cvref_t<Config>;
  using ComputeType = typename ConfigType::ComputeType;
  using Tag = typename ConfigType::Tag;
  static constexpr vec::Accuracy ExpAccuracy = ConfigType::exp_accuracy;

  const Config config;

  VECOPS_INLINE constexpr explicit Softmax(Config cfg = {}) : config(cfg) {}

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in, const OutSpec& out) const {
    static_assert(tensor::is_input_spec_v<InSpec>);
    static_assert(tensor::is_output_spec_v<OutSpec>);
    details::validate_softmax_layouts(
        in.input_layout(), out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    const auto in_row = details::softmax_row_spec<PrefixRank>(in);
    const auto out_row = details::softmax_row_spec<PrefixRank>(out);
    using InPolicy = tensor::InputAccessPolicy<0, 2>;
    using OutPolicy = tensor::OutputAccessPolicy<0>;
    const nint_t n = in.input_layout().shape()[PrefixRank];
    const nint_t cache_bytes = kernel::details::workspace_round_up(
        n * static_cast<nint_t>(sizeof(ComputeType)),
        vec::DEFAULT_ALIGNMENT);
    return cache_bytes + tensor::required_workspace(in_row, InPolicy{}) +
        tensor::required_workspace(out_row, OutPolicy{});
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(tensor::is_input_spec_v<InSpec>);
    static_assert(tensor::is_output_spec_v<OutSpec>);
    details::validate_softmax_layouts(
        in.input_layout(), out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
    constexpr bool CanPackBf16Output =
        std::same_as<ComputeType, float32_t> &&
        std::same_as<Tag, vec::ScalableTag<float32_t, 0>> &&
        std::same_as<typename OutSpec::MemoryElement, bfloat16_t> &&
        std::same_as<typename OutSpec::TransformType, tensor::NoTransform> &&
        tensor::is_ct_last_contiguous<
            typename OutSpec::OutputLayout, 1>::value;
    if constexpr (CanPackBf16Output) {
      nint_t element_count = 1;
      for (int d = 0; d <= PrefixRank; ++d) {
        element_count *= in.input_layout().shape()[d];
      }
      if (element_count > 64 * 1024) {
        kernel::loop::for_each_dims<PrefixRank>(
            [this, &workspace](const auto& in_row, const auto& out_row) {
              run_row<true>(workspace, in_row, out_row);
            },
            in, out);
        return;
      }
    }
#endif
    kernel::loop::for_each_dims<PrefixRank>(
        [this, &workspace](const auto& in_row, const auto& out_row) {
          run_row<false>(workspace, in_row, out_row);
        },
        in, out);
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void operator()(const InSpec& in, const OutSpec& out) const {
    kernel::Workspace storage(required_workspace(in, out));
    auto workspace = storage.view();
    (*this)(workspace, in, out);
  }

private:
  template <bool PackedBf16Output, typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = tensor::InputAccessPolicy<0, 2>;
    using OutPolicy = tensor::OutputAccessPolicy<0>;
    const nint_t n = in.input_layout().shape()[0];
    const auto mark = workspace.mark();
    ComputeType* exp_cache = workspace.allocate<ComputeType>(n);
    kernel::with_operands(
        workspace,
        tensor::operand(in, InPolicy{}),
        tensor::operand(out, OutPolicy{}),
        [this, n, exp_cache](auto& x, auto& y) {
          Tag tag{};
          const nint_t lanes = vec::size(tag);
          const ComputeType negative_infinity =
              -std::numeric_limits<ComputeType>::infinity();
          auto maximum0 = vec::fill(tag, negative_infinity);
          auto maximum1 = maximum0;
          auto maximum2 = maximum0;
          auto maximum3 = maximum0;
          nint_t col = 0;
#if defined(CPU_CAPABILITY_AVX512)
          for (; col + 4 * lanes <= n; col += 4 * lanes) {
            maximum0 = vec::max(
                maximum0,
                x.load(tag, tensor::coord(col), vec::opt::unmasked));
            maximum1 = vec::max(
                maximum1, x.load(
                    tag, tensor::coord(col + lanes), vec::opt::unmasked));
            maximum2 = vec::max(
                maximum2, x.load(
                    tag, tensor::coord(col + 2 * lanes),
                    vec::opt::unmasked));
            maximum3 = vec::max(
                maximum3, x.load(
                    tag, tensor::coord(col + 3 * lanes),
                    vec::opt::unmasked));
          }
#endif
          for (; col + lanes <= n; col += lanes) {
            maximum0 = vec::max(
                maximum0,
                x.load(tag, tensor::coord(col), vec::opt::unmasked));
          }
          if (col < n) {
            maximum0 = vec::max(
                maximum0,
                x.load(tag, tensor::coord(col), vec::opt::first(n - col),
                       vec::opt::merge(negative_infinity)));
          }
          const auto maximum = vec::max(
              vec::max(maximum0, maximum1),
              vec::max(maximum2, maximum3));
          const ComputeType max_value = vec::reduce_max(tag, maximum);
          const auto max_vector = vec::fill(tag, max_value);

          auto sum0 = vec::zeros(tag);
          auto sum1 = sum0;
          auto sum2 = sum0;
          auto sum3 = sum0;
          col = 0;
#if defined(CPU_CAPABILITY_AVX512)
          for (; col + 4 * lanes <= n; col += 4 * lanes) {
            auto value0 = x.load(
                tag, tensor::coord(col), vec::opt::unmasked);
            auto value1 = x.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto value2 = x.load(
                tag, tensor::coord(col + 2 * lanes), vec::opt::unmasked);
            auto value3 = x.load(
                tag, tensor::coord(col + 3 * lanes), vec::opt::unmasked);
            auto exp0 = vec::exp_neg(
                vec::sub(value0, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp1 = vec::exp_neg(
                vec::sub(value1, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp2 = vec::exp_neg(
                vec::sub(value2, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp3 = vec::exp_neg(
                vec::sub(value3, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            vec::store(tag, exp_cache + col, exp0);
            vec::store(tag, exp_cache + col + lanes, exp1);
            vec::store(tag, exp_cache + col + 2 * lanes, exp2);
            vec::store(tag, exp_cache + col + 3 * lanes, exp3);
            sum0 = vec::add(sum0, exp0);
            sum1 = vec::add(sum1, exp1);
            sum2 = vec::add(sum2, exp2);
            sum3 = vec::add(sum3, exp3);
          }
#endif
          for (; col + lanes <= n; col += lanes) {
            auto value = x.load(
                tag, tensor::coord(col), vec::opt::unmasked);
            auto exponential = vec::exp_neg(
                vec::sub(value, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            vec::store(tag, exp_cache + col, exponential);
            sum0 = vec::add(sum0, exponential);
          }
          if (col < n) {
            const nint_t active = n - col;
            const auto mask = vec::mwhilelt(tag, 0, active);
            const auto zero = vec::zeros(tag);
            auto value = x.load(
                tag, tensor::coord(col), vec::opt::first(active));
            auto exponential = vec::exp_neg(
                vec::sub(value, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>,
                vec::opt::masked(mask),
                vec::opt::merge(zero));
            vec::store(
                tag, exp_cache + col, exponential, vec::opt::first(active));
            sum0 = vec::add(sum0, exponential);
          }
          const auto sum_vector = vec::add(
              vec::add(sum0, sum1), vec::add(sum2, sum3));
          const ComputeType inverse_sum =
              ComputeType(1) / vec::reduce_add(tag, sum_vector);
          const auto inverse = vec::fill(tag, inverse_sum);

          col = 0;
          if constexpr (PackedBf16Output) {
            using PairTag = vec::Twice<Tag>;
            PairTag pair_tag{};
            for (; col + 2 * lanes <= n; col += 2 * lanes) {
              auto value0 = vec::load(tag, exp_cache + col);
              auto value1 = vec::load(tag, exp_cache + col + lanes);
              y.store(
                  pair_tag, tensor::coord(col),
                  vec::concat(
                      pair_tag, vec::mul(value0, inverse),
                      vec::mul(value1, inverse)),
                  vec::opt::unmasked);
            }
          }
#if defined(CPU_CAPABILITY_AVX512)
          for (; col + 4 * lanes <= n; col += 4 * lanes) {
            auto value0 = vec::load(tag, exp_cache + col);
            auto value1 = vec::load(tag, exp_cache + col + lanes);
            auto value2 = vec::load(tag, exp_cache + col + 2 * lanes);
            auto value3 = vec::load(tag, exp_cache + col + 3 * lanes);
            y.store(tag, tensor::coord(col), vec::mul(value0, inverse),
                    vec::opt::unmasked);
            y.store(tag, tensor::coord(col + lanes),
                    vec::mul(value1, inverse), vec::opt::unmasked);
            y.store(tag, tensor::coord(col + 2 * lanes),
                    vec::mul(value2, inverse), vec::opt::unmasked);
            y.store(tag, tensor::coord(col + 3 * lanes),
                    vec::mul(value3, inverse), vec::opt::unmasked);
          }
#else
          for (; col + 2 * lanes <= n; col += 2 * lanes) {
            auto value0 = vec::load(tag, exp_cache + col);
            auto value1 = vec::load(tag, exp_cache + col + lanes);
            y.store(tag, tensor::coord(col), vec::mul(value0, inverse),
                    vec::opt::unmasked);
            y.store(tag, tensor::coord(col + lanes),
                    vec::mul(value1, inverse), vec::opt::unmasked);
          }
#endif
          for (; col + lanes <= n; col += lanes) {
            auto value = vec::load(tag, exp_cache + col);
            y.store(
                tag, tensor::coord(col), vec::mul(value, inverse),
                vec::opt::unmasked);
          }
          if (col < n) {
            const nint_t active = n - col;
            auto value = vec::load(
                tag, exp_cache + col, vec::opt::first(active));
            y.store(tag, tensor::coord(col), vec::mul(value, inverse),
                    vec::opt::first(active));
          }
          y.commit();
        });
    workspace.rewind(mark);
  }
};

template <typename Config = SoftmaxConfig<>>
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_SOFTMAX_H
