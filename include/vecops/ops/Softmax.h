#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

/**
 * @file Softmax.h
 * @brief Last-dimension Softmax with DataAccess and ExecutionSession support.
 *
 * The operator normalizes every row independently, supports Tensor or Spec
 * operands, conversion transforms, cached and selected online algorithms, and
 * caller-owned or internally allocated workspace. Algorithm selection uses
 * compile-time layout/shape ranges; no runtime ISA fallback is introduced.
 */

namespace vecops::ops {

template <typename ComputeT = float32_t,
          vec::Accuracy ExpAccuracy = vec::Accuracy::Strict,
          bool AllowOnline = true>
/**
 * @brief Compile-time Softmax numerical and algorithm configuration.
 * @tparam ComputeT Accumulation and exponential element type.
 * @tparam ExpAccuracy Accuracy contract forwarded to `vec::exp_neg`.
 * @tparam AllowOnline Whether eligible typed shapes may use online tiling.
 */
struct SoftmaxConfig {
  using ComputeType = ComputeT;
  static constexpr vec::Accuracy exp_accuracy = ExpAccuracy;
  static constexpr bool allow_online = AllowOnline;
};

namespace details {

template <typename Config>
consteval bool softmax_online_allowed() {
  if constexpr (requires { Config::allow_online; }) {
    return Config::allow_online;
  } else {
    return true;
  }
}

template <typename Config>
consteval bool softmax_online_allowed(const Config&) {
  return softmax_online_allowed<Config>();
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

template <int Fold, int Store>
struct SoftmaxRowRecipe {
  static constexpr int FoldFactor = Fold;
  static constexpr int StoreFactor = Store;
};

using GenericSoftmaxRecipe = SoftmaxRowRecipe<1, 2>;
using PackedSoftmaxRecipe = SoftmaxRowRecipe<1, 2>;
using UnrolledSoftmaxRecipe = SoftmaxRowRecipe<4, 4>;
using SingleStoreSoftmaxRecipe = SoftmaxRowRecipe<1, 1>;

} // namespace details

template <typename Config = SoftmaxConfig<>>
/**
 * @brief Softmax operator normalizing the final dimension of an N-D operand.
 * @tparam Config `SoftmaxConfig`-compatible type.
 *
 * Input and output shapes must match and the last dimension must be non-empty.
 * `ResourceRequirements` makes ordinary ARM instantiations non-streaming-only;
 * calling through a scope enforces that contract at compile time.
 */
class Softmax {
  using XPolicy = tensor::InputAccessPolicy<
      0, 2, tensor::AccessPlan::automatic_deferred>;
  using YPolicy = tensor::OutputAccessPolicy<0>;

public:
  using ComputeType = typename Config::ComputeType;
  using Tag = vec::ScalableTag<ComputeType, 0>;
  /** Compile-time hardware-mode contract consumed by ExecutionSession. */
  using ResourceRequirements =
      typename execution::details::CurrentBackend::DefaultRequirements;
  static constexpr vec::Accuracy ExpAccuracy = Config::exp_accuracy;

  const Config config;

  VECOPS_INLINE constexpr explicit Softmax(Config cfg = {}) : config(cfg) {}

  template <typename InSpec, typename OutSpec>
    requires (tensor::is_input_spec_v<InSpec> &&
              tensor::is_output_spec_v<OutSpec>)
  /**
   * @brief Compute workspace bytes for normalized input/output Specs.
   * @return Cache, DataAccess materialization, and alignment bytes required by
   *         the exact typed plan.
   * @note The same Specs and runtime layouts must be used for execution.
   */
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in, const OutSpec& out) const {
    details::validate_softmax_layouts(
        in.input_layout(), out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    const auto in_row = tensor::take_trailing<1>(in);
    const auto out_row = tensor::take_trailing<1>(out);
    const nint_t n = tensor::size<PrefixRank>(in.input_layout());
    nint_t cache_elements = n;
    if constexpr (should_use_online<InSpec, OutSpec>()) {
      const nint_t tile_step = 4 * vec::size(Tag{});
      const nint_t tile_count = ceil_div(n, tile_step);
      cache_elements += 2 * tile_count;
    }
    const nint_t cache_bytes = align_up(
        cache_elements * static_cast<nint_t>(sizeof(ComputeType)),
        vec::DEFAULT_ALIGNMENT);
    return cache_bytes + tensor::required_workspace(in_row, XPolicy{}) +
        tensor::required_workspace(out_row, YPolicy{});
  }

  template <typename InOperand, typename OutOperand>
    requires (tensor::InputOperand<InOperand> &&
              tensor::OutputOperand<OutOperand> &&
              !(tensor::is_input_spec_v<InOperand> &&
                tensor::is_output_spec_v<OutOperand>))
  /** Normalize Tensor operands to Specs and return exact workspace bytes. */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const OutOperand& out) const {
    auto in_spec = tensor::as_input_spec<ComputeType>(in);
    auto out_spec = tensor::as_output_spec<ComputeType>(out);
    return required_workspace(in_spec, out_spec);
  }

  template <typename InSpec, typename OutSpec>
    requires (tensor::is_input_spec_v<InSpec> &&
              tensor::is_output_spec_v<OutSpec>)
  /**
   * @brief Execute with caller-owned workspace and normalized Specs.
   * @param workspace Scratch storage at least `required_workspace()` bytes.
   * @param in Readable input Spec.
   * @param out Writable output Spec with identical shape.
   */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    details::validate_softmax_layouts(
        in.input_layout(), out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    const auto run_rows = [&]<bool Online, typename Recipe>() {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &workspace](const auto& in_row, const auto& out_row) {
            if constexpr (Online) {
              run_row_online<Recipe>(workspace, in_row, out_row);
            } else {
              run_row<Recipe>(workspace, in_row, out_row);
            }
          },
          in, out);
    };
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
    constexpr bool CanPackBf16Output =
        std::same_as<ComputeType, float32_t> &&
        std::same_as<typename OutSpec::MemoryElement, bfloat16_t> &&
        std::same_as<typename OutSpec::TransformType, tensor::NoTransform> &&
        tensor::is_ct_last_contiguous<
            typename OutSpec::OutputLayout, 1>::value;
    using ElementCount = tensor::numel_type_t<
        typename InSpec::InputLayout>;
    constexpr bool UsePackedBf16Output = CanPackBf16Output &&
        meta::lower_bound_at_least_v<ElementCount, 64 * 1024 + 1>;
    if constexpr (UsePackedBf16Output) {
      if constexpr (should_use_online<InSpec, OutSpec>()) {
        run_rows.template operator()<true, details::PackedSoftmaxRecipe>();
      } else {
        run_rows.template operator()<false, details::PackedSoftmaxRecipe>();
      }
      return;
    }
#endif
    if constexpr (should_use_online<InSpec, OutSpec>()) {
      run_rows.template operator()<true, details::SingleStoreSoftmaxRecipe>();
    } else {
#if defined(ARCH_X86_FAMILY) && defined(__GNUC__) && !defined(__clang__)
      // WORKAROUND/TUNING: GCC's paired BF16 conversion/store path increases
      // register pressure and emits a slower dependency chain on AVX-512.
      // Retain one-vector stores for BF16 only; other dtypes use the generic
      // recipe. Validate generated assembly before removing this split.
      using GenericRecipe = std::conditional_t<
          std::same_as<typename OutSpec::MemoryElement, bfloat16_t>,
          details::SingleStoreSoftmaxRecipe,
          details::GenericSoftmaxRecipe>;
#elif defined(CPU_CAPABILITY_SVE) && defined(__GNUC__) && !defined(__clang__)
      // WORKAROUND/TUNING: GCC 15 expands paired FP16 conversion/store into a
      // longer dependency chain than the single-vector SVE form. Keep this
      // compiler-specific recipe until generated assembly no longer shows the
      // extra chain; Clang benefits from the generic two-vector store recipe.
      using GenericRecipe = std::conditional_t<
          std::same_as<typename OutSpec::MemoryElement, float16_t>,
          details::SingleStoreSoftmaxRecipe,
          details::GenericSoftmaxRecipe>;
#else
      using GenericRecipe = details::GenericSoftmaxRecipe;
#endif
#if defined(CPU_CAPABILITY_AVX512)
      using NormalizedSize = tensor::size_type_t<
          PrefixRank, typename InSpec::InputLayout>;
      constexpr nint_t lanes = vec::size(Tag{});
      constexpr nint_t max_unrolled_n =
          sizeof(typename InSpec::MemoryElement) < sizeof(ComputeType)
          ? 4096
          : 1024;
      // For an unconstrained caller, prefer the throughput recipe used by
      // common neural-network widths. Explicit bounds outside the tuned range
      // select the smaller generic recipe without a runtime branch.
      constexpr bool UseUnrolled = meta::range_within_v<
          NormalizedSize, 16 * lanes, max_unrolled_n> ||
          !meta::is_bounded_v<NormalizedSize>;
      if constexpr (UseUnrolled) {
        run_rows.template operator()<false, details::UnrolledSoftmaxRecipe>();
      } else {
        run_rows.template operator()<false, GenericRecipe>();
      }
#else
      run_rows.template operator()<false, GenericRecipe>();
#endif
    }
  }

  template <typename InOperand, typename OutOperand>
    requires (tensor::InputOperand<InOperand> &&
              tensor::OutputOperand<OutOperand> &&
              !(tensor::is_input_spec_v<InOperand> &&
                tensor::is_output_spec_v<OutOperand>))
  /** Execute Tensor operands using caller-owned workspace. */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const OutOperand& out) const {
    auto in_spec = tensor::as_input_spec<ComputeType>(in);
    auto out_spec = tensor::as_output_spec<ComputeType>(out);
    (*this)(workspace, in_spec, out_spec);
  }

  template <execution::ExecutionScope Scope,
            typename InOperand, typename OutOperand>
    requires (tensor::InputOperand<InOperand> &&
              tensor::OutputOperand<OutOperand>)
  /**
   * @brief Execute through an ExecutionSession or active enclosing scope.
   * @param scope Scope providing resource transitions and worker workspace.
   * @param in Input Tensor/Spec operand.
   * @param out Output Tensor/Spec operand with identical shape.
   *
   * `with_resources` is nested deliberately: if an enclosing region already
   * satisfies the operator, all transition logic is removed by `if constexpr`.
   */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const OutOperand& out) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          (*this)(active.workspace_view(), in, out);
        });
  }

  template <typename InOperand, typename OutOperand>
    requires (tensor::InputOperand<InOperand> &&
              tensor::OutputOperand<OutOperand>)
  /** Allocate exact temporary workspace, create a session, and execute once. */
  VECOPS_INLINE void operator()(
      const InOperand& in, const OutOperand& out) const {
    kernel::Workspace storage(required_workspace(in, out));
    auto workspace = storage.view();
    ExecutionSession execution{workspace};
    (*this)(execution, in, out);
  }

  /**
   * @brief Execute one already-bound row using the regular stable Softmax path.
   * @param exp_cache Storage for at least one Compute value per logical lane.
   * @param x Bound rank-one readable access.
   * @param y Bound rank-one committable writable access.
   * `exp_cache` points to at least logical_layout(x).shape()[0] Compute values
   * allocated before entering the operand-binding scope.
   */
  template <typename X, typename Y>
    requires (tensor::BoundInputAccess<X> &&
              tensor::CommittableBoundOutputAccess<Y>)
  VECOPS_INLINE void run_bound(
      ComputeType* exp_cache, X& x, Y& y) const {
    static_assert(X::Rank == 1 && Y::Rank == 1);
    const nint_t n = tensor::logical_layout(x).shape()[0];
    VECOPS_ASSERT(exp_cache != nullptr, "Softmax bound cache is null");
    run_bound_cached<details::GenericSoftmaxRecipe, false>(
        n, exp_cache, x, y);
  }

private:
  template <typename InSpec, typename OutSpec>
  static consteval bool should_use_online() {
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
    if constexpr (!details::softmax_online_allowed<Config>()) return false;
    constexpr int Rank = InSpec::InputTensor::Ndim;
    if constexpr (
        !tensor::is_ct_last_contiguous<
            typename InSpec::InputLayout, 1>::value ||
        !tensor::is_ct_last_contiguous<
            typename OutSpec::OutputLayout, 1>::value) return false;
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    using NormalizedSize = tensor::size_type_t<
        Rank - 1, typename InSpec::InputLayout>;
    using ElementCount = tensor::numel_type_t<
        typename InSpec::InputLayout>;
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float32_t> &&
        ExpAccuracy != vec::Accuracy::Estimate) {
      return meta::range_within_v<NormalizedSize, 8192, 16384> &&
          meta::range_within_v<
              ElementCount, 512 * 1024, 8 * 1024 * 1024>;
    } else {
      return false;
    }
#else
    if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float32_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Strict) {
        return meta::range_within_v<NormalizedSize, 2048, 4096> &&
            meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      } else {
        return meta::range_within_v<NormalizedSize, 2048, 32768> &&
            meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      }
    } else if constexpr (
        std::same_as<ComputeType, float64_t> &&
        std::same_as<InElement, float64_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return meta::lower_bound_at_least_v<NormalizedSize, 2048> &&
            meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      } else {
        return meta::lower_bound_at_least_v<NormalizedSize, 2048> &&
            meta::lower_bound_at_least_v<ElementCount, 16 * 1024 * 1024>;
      }
    } else if constexpr (
        std::same_as<ComputeType, float32_t> &&
        std::same_as<InElement, float16_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return meta::range_within_v<NormalizedSize, 2048, 2048> &&
            meta::lower_bound_at_least_v<ElementCount, 16 * 1024 * 1024>;
      } else {
        return false;
      }
    } else {
      return false;
    }
#endif
#else
    return false;
#endif
  }

  template <typename Recipe, typename InSpec, typename OutSpec>
  VECOPS_NOINLINE void run_row_online(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    Tag tag{};
    const nint_t n = tensor::size<0>(in.input_layout());
    const nint_t tile_step = 4 * vec::size(tag);
    const nint_t tile_count = ceil_div(n, tile_step);
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
        workspace, tensor::operand(in, XPolicy{}),
        tensor::operand(out, YPolicy{}),
        [this, n, tile_step, tile_count, exp_cache,
         tile_max_cache, tile_sum_cache, &fallback](auto& x, auto& y) {
          Tag tag{};
          const ComputeType negative_infinity =
              -std::numeric_limits<ComputeType>::infinity();
          nint_t tile = 0;
          kernel::loop::fold<4, 4>(
              tag, n,
              [&](auto block_tag, nint_t col, auto active,
                  const auto& negative_infinity_v) VECOPS_INLINE_LAMBDA {
                auto value = x.load(
                    block_tag, tensor::coord(col), active,
                    vec::opt::merge(negative_infinity_v),
                    tensor::materialize::populate);
                const ComputeType tile_max =
                    vec::reduce_max(block_tag, value);
                const auto maximum = vec::fill(block_tag, tile_max);
                auto centered = vec::sub(value, maximum);
                const auto zero = vec::zeros(block_tag);
                auto exponential = vec::exp_neg(
                    block_tag, centered,
                    vec::opt::math::accuracy<ExpAccuracy>, active,
                    vec::opt::merge(zero));
                vec::store(block_tag, exp_cache + col, exponential, active);
                tile_max_cache[tile] = tile_max;
                tile_sum_cache[tile] = vec::reduce_add(block_tag, exponential);
                ++tile;
              },
              kernel::loop::invariant(negative_infinity));

          ComputeType global_max{};
          kernel::loop::fold(
              tag, tile_count,
              [&](auto block_tag, nint_t i, auto active, auto& maximum) VECOPS_INLINE_LAMBDA {
                auto tile_max = vec::load(
                    block_tag, tile_max_cache + i, active,
                    vec::opt::merge(negative_infinity));
                maximum = vec::max(maximum, tile_max);
              },
              kernel::loop::reduce_max(global_max));
          if (!std::isfinite(static_cast<double>(global_max))) {
            fallback = true;
            return;
          }

          ComputeType global_sum{};
          kernel::loop::fold(
              tag, tile_count,
              [&](auto block_tag, nint_t i, auto active, auto& sum) VECOPS_INLINE_LAMBDA {
                auto tile_max = vec::load(
                    block_tag, tile_max_cache + i, active,
                    vec::opt::merge(global_max));
                const auto maximum = vec::fill(block_tag, global_max);
                auto centered = vec::sub(tile_max, maximum);
                const auto zero = vec::zeros(block_tag);
                auto scale = vec::exp_neg(
                    block_tag, centered,
                    vec::opt::math::accuracy<ExpAccuracy>, active,
                    vec::opt::merge(zero));
                vec::store(block_tag, tile_max_cache + i, scale, active);
                auto tile_sum = vec::load(
                    block_tag, tile_sum_cache + i, active,
                    vec::opt::merge(ComputeType(0)));
                sum = vec::fmadd(scale, tile_sum, sum);
              },
              kernel::loop::reduce_add(global_sum));
          if (!std::isfinite(static_cast<double>(global_sum)) ||
              !(global_sum > ComputeType(0))) {
            fallback = true;
            return;
          }

          nint_t col = 0;
          const ComputeType inverse_sum = ComputeType(1) / global_sum;
          for (tile = 0; tile < tile_count; ++tile) {
            const nint_t count = std::min(tile_step, n - col);
            const ComputeType scale = tile_max_cache[tile] * inverse_sum;
            kernel::loop::fold<Recipe::StoreFactor>(
                tag, count,
                [&](auto block_tag, nint_t local, auto active,
                    const auto& scale_v) VECOPS_INLINE_LAMBDA {
                  auto exponential = vec::load(
                      block_tag, exp_cache + col + local, active);
                  auto normalized = vec::mul(exponential, scale_v);
                  y.store(
                      block_tag, tensor::coord(col + local),
                      normalized, active);
                },
                kernel::loop::invariant(scale));
            col += count;
          }
          y.commit();
        });
    workspace.rewind(mark);
    if (fallback) {
      run_row<Recipe>(workspace, in, out);
    }
  }

  template <typename Recipe, bool CopyDirectInput, typename X, typename Y>
  VECOPS_ALWAYS_INLINE void run_bound_cached(
      nint_t n, ComputeType* exp_cache, X& x, Y& y) const {
    Tag tag{};
    const ComputeType negative_infinity =
        -std::numeric_limits<ComputeType>::infinity();
    ComputeType max_value{};
    auto fold_max = [&](auto&& ux) VECOPS_INLINE_LAMBDA {
      kernel::loop::fold<Recipe::FoldFactor>(
          tag, n,
          [&](auto block_tag, nint_t col, auto active,
              auto& maximum) VECOPS_INLINE_LAMBDA {
            auto value = ux.load(
                block_tag, tensor::coord(col), active,
                vec::opt::merge(negative_infinity),
                tensor::materialize::populate);
            maximum = vec::max(maximum, value);
          },
          kernel::loop::reduce_max(max_value));
    };
    if constexpr (sizeof(typename X::MemoryElement) == sizeof(ComputeType)) {
      if constexpr (CopyDirectInput) {
        auto direct_x = x;
        fold_max(direct_x);
      } else {
        fold_max(x);
      }
    } else {
      tensor::with_unordered_access(x, fold_max);
    }

    ComputeType sum{};
    kernel::loop::fold<Recipe::FoldFactor>(
        tag, n,
        [&](auto block_tag, nint_t col, auto active, auto& sum_vector)
            VECOPS_INLINE_LAMBDA {
          auto value = x.load(
              block_tag, tensor::coord(col), active,
              vec::opt::merge(negative_infinity));
          const auto maximum = vec::fill(block_tag, max_value);
          auto centered = vec::sub(value, maximum);
          const auto zero = vec::zeros(block_tag);
          auto exponential = vec::exp_neg(
              block_tag, centered,
              vec::opt::math::accuracy<ExpAccuracy>, active,
              vec::opt::merge(zero));
          vec::store(block_tag, exp_cache + col, exponential, active);
          sum_vector = vec::add(sum_vector, exponential);
        },
        kernel::loop::reduce_add(sum));

    const ComputeType inverse_sum = ComputeType(1) / sum;
    kernel::loop::fold<Recipe::StoreFactor>(
        tag, n,
        [&](auto block_tag, nint_t col, auto active,
            const auto& inverse) VECOPS_INLINE_LAMBDA {
          auto exponential = vec::load(
              block_tag, exp_cache + col, active);
          auto normalized = vec::mul(exponential, inverse);
          y.store(block_tag, tensor::coord(col), normalized, active);
        },
        kernel::loop::invariant(inverse_sum));
    y.commit();
  }

  template <typename Recipe, typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    nint_t n = tensor::size<0>(in.input_layout());
#if defined(CPU_CAPABILITY_SVE) && defined(__GNUC__) && !defined(__clang__)
    if constexpr (std::same_as<typename InSpec::MemoryElement, float16_t>) {
      // WORKAROUND: GCC 15 otherwise propagates large Const-shaped FP16 row
      // counts into the fold and aggressively unrolls them, causing excessive
      // code size and compile time. The empty asm keeps only the trip count
      // value runtime-visible; access plans and loop recipes remain compile-time.
      asm volatile("" : "+r"(n));
    }
#endif
    const auto mark = workspace.mark();
    ComputeType* exp_cache = static_cast<ComputeType*>(workspace.allocate(
        n * static_cast<nint_t>(sizeof(ComputeType)),
        vec::DEFAULT_ALIGNMENT));
    auto compute =
        [this, n, exp_cache](auto& x, auto& y) VECOPS_INLINE_LAMBDA {
          constexpr bool CopyDirectInput =
              sizeof(typename InSpec::MemoryElement) == sizeof(ComputeType) &&
              tensor::details::resolve_plan<InSpec, XPolicy>() ==
                  tensor::AccessPlan::direct;
          run_bound_cached<Recipe, CopyDirectInput>(n, exp_cache, x, y);
        };
    kernel::with_operands(
        workspace, tensor::operand(in, XPolicy{}),
        tensor::operand(out, YPolicy{}), compute);
    workspace.rewind(mark);
  }
};

template <typename Config = SoftmaxConfig<>>
/** Construct a Softmax operator from its compile-time configuration type. */
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_SOFTMAX_H
