#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

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

template <typename Memory>
class ContiguousSoftmaxInput {
public:
  explicit ContiguousSoftmaxInput(const Memory* data) : data_(data) {}

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE auto load(
      Tag tag, const tensor::Coord<1>& position,
      Options&&... options) const {
    return vec::load_convert(
        tag, data_ + position[0],
        std::forward<Options>(options)...);
  }

private:
  const Memory* data_;
};

template <typename Memory>
class ContiguousSoftmaxOutput {
public:
  explicit ContiguousSoftmaxOutput(Memory* data) : data_(data) {}

  template <vec::VectorTag Tag, typename... Options>
  VECOPS_ALWAYS_INLINE void store(
      Tag tag, const tensor::Coord<1>& position, vec::Vec<Tag> value,
      Options&&... options) const {
    vec::store_convert(
        tag, data_ + position[0], value,
        std::forward<Options>(options)...);
  }

  VECOPS_ALWAYS_INLINE void commit() const {}

private:
  Memory* data_;
};

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
    nint_t cache_elements = n;
    if (should_use_online(in, out)) {
      const nint_t tile_step = 4 * vec::size(Tag{});
      const nint_t tile_count = (n + tile_step - 1) / tile_step;
      cache_elements += 2 * tile_count;
    }
    const nint_t cache_bytes = kernel::details::workspace_round_up(
        cache_elements * static_cast<nint_t>(sizeof(ComputeType)),
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
    const bool use_online = should_use_online(in, out);
    const auto run_rows =
        [&]<bool Online, bool PackedBf16Output, bool Unrolled = false>() {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &workspace](const auto& in_row, const auto& out_row) {
            if constexpr (Online) {
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
              run_row_online<PackedBf16Output>(
                  workspace, in_row, out_row);
#endif
            } else {
              run_row<PackedBf16Output, Unrolled>(
                  workspace, in_row, out_row);
            }
          },
          in, out);
    };
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
        if (use_online) run_rows.template operator()<true, true>();
        else run_rows.template operator()<false, true>();
        return;
      }
    }
#endif
    if (use_online) {
      run_rows.template operator()<true, false>();
    } else {
#if defined(CPU_CAPABILITY_AVX512)
      const nint_t normalized_n =
          in.input_layout().shape()[PrefixRank];
      const nint_t lanes = vec::size(Tag{});
      if (normalized_n >= 16 * lanes && normalized_n <= 1024) {
        run_rows.template operator()<false, false, true>();
      } else {
        run_rows.template operator()<false, false, false>();
      }
#else
      run_rows.template operator()<false, false, false>();
#endif
    }
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void operator()(const InSpec& in, const OutSpec& out) const {
    kernel::Workspace storage(required_workspace(in, out));
    auto workspace = storage.view();
    (*this)(workspace, in, out);
  }

private:
  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE bool should_use_online(
      const InSpec& in, const OutSpec& out) const {
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
    if (!details::softmax_online_allowed(config)) return false;
    constexpr int Rank = InSpec::InputTensor::Ndim;
    const auto& in_layout = in.input_layout();
    const auto& out_layout = out.output_layout();
    if constexpr (!tensor::is_ct_last_contiguous<
                      typename InSpec::InputLayout, 1>::value) {
      if (in_layout.strides()[Rank - 1] != 1) return false;
    }
    if constexpr (!tensor::is_ct_last_contiguous<
                      typename OutSpec::OutputLayout, 1>::value) {
      if (out_layout.strides()[Rank - 1] != 1) return false;
    }

    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    const nint_t normalized_n = in_layout.shape()[Rank - 1];
    const auto element_count = [&] {
      nint_t count = 1;
      for (int d = 0; d < Rank; ++d) count *= in_layout.shape()[d];
      return count;
    };
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float32_t> &&
        ExpAccuracy != vec::Accuracy::Estimate) {
      if (normalized_n < 8192 || normalized_n > 16384) return false;
      const nint_t count = element_count();
      return count >= 512 * 1024 && count <= 8 * 1024 * 1024;
    } else {
      return false;
    }
#else
    if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float32_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Strict) {
        if (normalized_n < 2048 || normalized_n > 4096) return false;
        return element_count() >= 512 * 1024;
      } else {
        if (normalized_n < 2048 || normalized_n > 32768) return false;
        return element_count() >= 512 * 1024;
      }
    } else if constexpr (
        std::same_as<ComputeType, float64_t> &&
        std::same_as<InElement, float64_t>) {
      if (normalized_n < 2048) return false;
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return element_count() >= 512 * 1024;
      } else {
        return element_count() >= 16 * 1024 * 1024;
      }
    } else if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float16_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return normalized_n == 2048 &&
            element_count() >= 16 * 1024 * 1024;
      } else {
        return false;
      }
    } else {
      return false;
    }
#endif
#else
    (void)in;
    (void)out;
    return false;
#endif
  }

#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
  template <bool PackedBf16Output, typename InSpec, typename OutSpec>
  VECOPS_NOINLINE void run_row_online(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = tensor::InputAccessPolicy<0, 2>;
    using OutPolicy = tensor::OutputAccessPolicy<0>;
    Tag tag{};
    const nint_t n = in.input_layout().shape()[0];
    const nint_t lanes = vec::size(tag);
    const nint_t tile_step = 4 * lanes;
    const nint_t tile_count = (n + tile_step - 1) / tile_step;
    const auto mark = workspace.mark();
    ComputeType* exp_cache = static_cast<ComputeType*>(workspace.allocate(
        n * static_cast<nint_t>(sizeof(ComputeType)),
        vec::DEFAULT_ALIGNMENT));
    ComputeType* tile_max_cache =
        workspace.allocate<ComputeType>(tile_count);
    ComputeType* tile_sum_cache =
        workspace.allocate<ComputeType>(tile_count);
    bool fallback = false;

    kernel::with_operands(
        workspace, tensor::operand(in, InPolicy{}),
        tensor::operand(out, OutPolicy{}),
        [this, n, lanes, tile_step, tile_count, exp_cache,
         tile_max_cache, tile_sum_cache, &fallback](auto& x, auto& y) {
          Tag tag{};
          const ComputeType negative_infinity =
              -std::numeric_limits<ComputeType>::infinity();
          const auto zero = vec::zeros(tag);
          nint_t col = 0;
          nint_t tile = 0;
          for (; col + tile_step <= n; col += tile_step, ++tile) {
            auto x0 = x.load(
                tag, tensor::coord(col), vec::opt::unmasked);
            auto x1 = x.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto x2 = x.load(
                tag, tensor::coord(col + 2 * lanes),
                vec::opt::unmasked);
            auto x3 = x.load(
                tag, tensor::coord(col + 3 * lanes),
                vec::opt::unmasked);
            const auto tile_max_vector = vec::max(
                vec::max(x0, x1), vec::max(x2, x3));
            const ComputeType tile_max =
                vec::reduce_max(tag, tile_max_vector);
            tile_max_cache[tile] = tile_max;
            const auto max_vector = vec::fill(tag, tile_max);
            auto exp0 = vec::exp_neg(
                vec::sub(x0, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp1 = vec::exp_neg(
                vec::sub(x1, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp2 = vec::exp_neg(
                vec::sub(x2, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            auto exp3 = vec::exp_neg(
                vec::sub(x3, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            vec::store(tag, exp_cache + col, exp0);
            vec::store(tag, exp_cache + col + lanes, exp1);
            vec::store(tag, exp_cache + col + 2 * lanes, exp2);
            vec::store(tag, exp_cache + col + 3 * lanes, exp3);
            tile_sum_cache[tile] = vec::reduce_add(
                tag, vec::add(
                         vec::add(exp0, exp1), vec::add(exp2, exp3)));
          }

          if (col < n) {
            const nint_t remaining = n - col;
            auto tile_max_vector =
                vec::fill(tag, negative_infinity);
            nint_t local = 0;
            for (; local + lanes <= remaining; local += lanes) {
              tile_max_vector = vec::max(
                  tile_max_vector,
                  x.load(
                      tag, tensor::coord(col + local),
                      vec::opt::unmasked));
            }
            if (local < remaining) {
              const nint_t active = remaining - local;
              tile_max_vector = vec::max(
                  tile_max_vector,
                  x.load(
                      tag, tensor::coord(col + local),
                      vec::opt::first(active),
                      vec::opt::merge(negative_infinity)));
            }

            const ComputeType tile_max =
                vec::reduce_max(tag, tile_max_vector);
            tile_max_cache[tile] = tile_max;
            const auto max_vector = vec::fill(tag, tile_max);
            auto tile_sum_vector = vec::zeros(tag);
            local = 0;
            for (; local + lanes <= remaining; local += lanes) {
              auto exponential = vec::exp_neg(
                  vec::sub(
                      x.load(
                          tag, tensor::coord(col + local),
                          vec::opt::unmasked),
                      max_vector),
                  vec::opt::math::accuracy<ExpAccuracy>);
              vec::store(tag, exp_cache + col + local, exponential);
              tile_sum_vector = vec::add(tile_sum_vector, exponential);
            }
            if (local < remaining) {
              const nint_t active = remaining - local;
              const auto mask = vec::mwhilelt(tag, 0, active);
              auto exponential = vec::exp_neg(
                  vec::sub(
                      x.load(
                          tag, tensor::coord(col + local),
                          vec::opt::first(active)),
                      max_vector),
                  vec::opt::math::accuracy<ExpAccuracy>,
                  vec::opt::masked(mask),
                  vec::opt::merge(zero));
              vec::store(
                  tag, exp_cache + col + local, exponential,
                  vec::opt::first(active));
              tile_sum_vector = vec::add(tile_sum_vector, exponential);
            }
            tile_sum_cache[tile] =
                vec::reduce_add(tag, tile_sum_vector);
            ++tile;
          }

          auto metadata_max = vec::fill(tag, negative_infinity);
          tile = 0;
          for (; tile + lanes <= tile_count; tile += lanes) {
            metadata_max = vec::max(
                metadata_max, vec::load(tag, tile_max_cache + tile));
          }
          if (tile < tile_count) {
            metadata_max = vec::max(
                metadata_max,
                vec::load(
                    tag, tile_max_cache + tile,
                    vec::opt::first(tile_count - tile),
                    vec::opt::merge(negative_infinity)));
          }
          const ComputeType global_max =
              vec::reduce_max(tag, metadata_max);
          if (!std::isfinite(static_cast<double>(global_max))) {
            fallback = true;
            return;
          }

          const auto global_max_vector = vec::fill(tag, global_max);
          auto metadata_sum = vec::zeros(tag);
          tile = 0;
          for (; tile + lanes <= tile_count; tile += lanes) {
            auto scale = vec::exp_neg(
                vec::sub(
                    vec::load(tag, tile_max_cache + tile),
                    global_max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            vec::store(tag, tile_max_cache + tile, scale);
            metadata_sum = vec::fmadd(
                scale, vec::load(tag, tile_sum_cache + tile),
                metadata_sum);
          }
          if (tile < tile_count) {
            const nint_t active = tile_count - tile;
            const auto mask = vec::mwhilelt(tag, 0, active);
            auto scale = vec::exp_neg(
                vec::sub(
                    vec::load(
                        tag, tile_max_cache + tile,
                        vec::opt::first(active)),
                    global_max_vector),
                vec::opt::math::accuracy<ExpAccuracy>,
                vec::opt::masked(mask),
                vec::opt::merge(zero));
            vec::store(
                tag, tile_max_cache + tile, scale,
                vec::opt::first(active));
            metadata_sum = vec::fmadd(
                scale,
                vec::load(
                    tag, tile_sum_cache + tile,
                    vec::opt::first(active)),
                metadata_sum);
          }
          const ComputeType global_sum =
              vec::reduce_add(tag, metadata_sum);
          if (!std::isfinite(static_cast<double>(global_sum)) ||
              !(global_sum > ComputeType(0))) {
            fallback = true;
            return;
          }
          const ComputeType inverse_sum = ComputeType(1) / global_sum;

          col = 0;
          for (tile = 0; tile < tile_count; ++tile) {
            const nint_t count = std::min(tile_step, n - col);
            const auto scale = vec::fill(
                tag, tile_max_cache[tile] * inverse_sum);
            nint_t local = 0;
            if constexpr (PackedBf16Output) {
              using PairTag = vec::Twice<Tag>;
              PairTag pair_tag{};
              for (; local + 2 * lanes <= count; local += 2 * lanes) {
                auto out0 = vec::mul(
                    vec::load(tag, exp_cache + col + local), scale);
                auto out1 = vec::mul(
                    vec::load(
                        tag, exp_cache + col + local + lanes),
                    scale);
                y.store(
                    pair_tag, tensor::coord(col + local),
                    vec::concat(pair_tag, out0, out1),
                    vec::opt::unmasked);
              }
            }
            for (; local + lanes <= count; local += lanes) {
              y.store(
                  tag, tensor::coord(col + local),
                  vec::mul(
                      vec::load(tag, exp_cache + col + local), scale),
                  vec::opt::unmasked);
            }
            if (local < count) {
              const nint_t active = count - local;
              y.store(
                  tag, tensor::coord(col + local),
                  vec::mul(
                      vec::load(
                          tag, exp_cache + col + local,
                          vec::opt::first(active)),
                      scale),
                  vec::opt::first(active));
            }
            col += count;
          }
          y.commit();
        });
    workspace.rewind(mark);
    if (fallback) {
      run_row<PackedBf16Output>(workspace, in, out);
    }
  }
#endif

  template <bool PackedBf16Output, bool Unrolled = false,
            typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = tensor::InputAccessPolicy<0, 2>;
    using OutPolicy = tensor::OutputAccessPolicy<0>;
    const nint_t n = in.input_layout().shape()[0];
    const auto mark = workspace.mark();
    ComputeType* exp_cache = static_cast<ComputeType*>(workspace.allocate(
        n * static_cast<nint_t>(sizeof(ComputeType)),
        vec::DEFAULT_ALIGNMENT));
    auto compute = [this, n, exp_cache](auto& x, auto& y) {
          Tag tag{};
          const nint_t lanes = vec::size(tag);
          const ComputeType negative_infinity =
              -std::numeric_limits<ComputeType>::infinity();
          auto maximum = vec::fill(tag, negative_infinity);
          nint_t col = 0;
          if constexpr (Unrolled) {
            auto maximum1 = maximum;
            auto maximum2 = maximum;
            auto maximum3 = maximum;
            for (; col + 4 * lanes <= n; col += 4 * lanes) {
              maximum = vec::max(
                  maximum,
                  x.load(tag, tensor::coord(col), vec::opt::unmasked));
              maximum1 = vec::max(
                  maximum1, x.load(
                      tag, tensor::coord(col + lanes),
                      vec::opt::unmasked));
              maximum2 = vec::max(
                  maximum2, x.load(
                      tag, tensor::coord(col + 2 * lanes),
                      vec::opt::unmasked));
              maximum3 = vec::max(
                  maximum3, x.load(
                      tag, tensor::coord(col + 3 * lanes),
                      vec::opt::unmasked));
            }
            maximum = vec::max(
                vec::max(maximum, maximum1),
                vec::max(maximum2, maximum3));
          }
          for (; col + lanes <= n; col += lanes) {
            maximum = vec::max(
                maximum,
                x.load(tag, tensor::coord(col), vec::opt::unmasked));
          }
          if (col < n) {
            maximum = vec::max(
                maximum,
                x.load(tag, tensor::coord(col), vec::opt::first(n - col),
                       vec::opt::merge(negative_infinity)));
          }
          const ComputeType max_value = vec::reduce_max(tag, maximum);
          const auto max_vector = vec::fill(tag, max_value);

          auto sum = vec::zeros(tag);
          col = 0;
          if constexpr (Unrolled) {
            auto sum1 = sum;
            auto sum2 = sum;
            auto sum3 = sum;
            for (; col + 4 * lanes <= n; col += 4 * lanes) {
              auto value0 = x.load(
                  tag, tensor::coord(col), vec::opt::unmasked);
              auto value1 = x.load(
                  tag, tensor::coord(col + lanes), vec::opt::unmasked);
              auto value2 = x.load(
                  tag, tensor::coord(col + 2 * lanes),
                  vec::opt::unmasked);
              auto value3 = x.load(
                  tag, tensor::coord(col + 3 * lanes),
                  vec::opt::unmasked);
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
              sum = vec::add(sum, exp0);
              sum1 = vec::add(sum1, exp1);
              sum2 = vec::add(sum2, exp2);
              sum3 = vec::add(sum3, exp3);
            }
            sum = vec::add(
                vec::add(sum, sum1), vec::add(sum2, sum3));
          }
          for (; col + lanes <= n; col += lanes) {
            auto value = x.load(
                tag, tensor::coord(col), vec::opt::unmasked);
            auto exponential = vec::exp_neg(
                vec::sub(value, max_vector),
                vec::opt::math::accuracy<ExpAccuracy>);
            vec::store(tag, exp_cache + col, exponential);
            sum = vec::add(sum, exponential);
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
            sum = vec::add(sum, exponential);
          }
          const ComputeType inverse_sum =
              ComputeType(1) / vec::reduce_add(tag, sum);
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
          if constexpr (Unrolled) {
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
          } else {
            for (; col + 2 * lanes <= n; col += 2 * lanes) {
              auto value0 = vec::load(tag, exp_cache + col);
              auto value1 = vec::load(tag, exp_cache + col + lanes);
              y.store(tag, tensor::coord(col), vec::mul(value0, inverse),
                      vec::opt::unmasked);
              y.store(tag, tensor::coord(col + lanes),
                      vec::mul(value1, inverse), vec::opt::unmasked);
            }
          }
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
    };
    constexpr bool StaticDirect =
        tensor::is_ct_last_contiguous<typename InSpec::InputLayout, 1>::value &&
        tensor::is_ct_last_contiguous<typename OutSpec::OutputLayout, 1>::value;
#if defined(ARCH_X86_FAMILY)
    constexpr bool RawContiguous = StaticDirect &&
        std::same_as<typename InSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename OutSpec::TransformType, tensor::NoTransform>;
#else
    constexpr bool RawContiguous = false;
#endif
    if constexpr (RawContiguous) {
      using InputMemory =
          std::remove_const_t<typename InSpec::MemoryElement>;
      using OutputMemory = typename OutSpec::MemoryElement;
      details::ContiguousSoftmaxInput<InputMemory> x{in.tensor().data()};
      details::ContiguousSoftmaxOutput<OutputMemory> y{out.tensor().data()};
      compute(x, y);
    } else if constexpr (StaticDirect) {
      using DirectInPolicy = tensor::InputAccessPolicy<
          0, 2, tensor::AccessPlan::direct>;
      using DirectOutPolicy = tensor::OutputAccessPolicy<
          0, tensor::AccessPlan::direct>;
      auto x = tensor::bind(in, DirectInPolicy{}, workspace);
      auto y = tensor::bind(out, DirectOutPolicy{}, workspace);
      compute(x, y);
    } else {
      kernel::with_operands(
          workspace, tensor::operand(in, InPolicy{}),
          tensor::operand(out, OutPolicy{}), compute);
    }
    workspace.rewind(mark);
  }
};

template <typename Config = SoftmaxConfig<>>
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_SOFTMAX_H
