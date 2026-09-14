#ifndef VECOPS_OPS_DETAILS_SOFTMAX_OPERATION_H
#define VECOPS_OPS_DETAILS_SOFTMAX_OPERATION_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/Parallel.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

/**
 * @file vecops/ops/details/softmax/Operation.h
 * @brief Last-dimension Softmax implementation (DataAccess + ExecutionSession).
 *
 * Every prefix coordinate defines an independently normalized row. Tensor and
 * Spec operands share the same typed DataAccess plan, while algorithm and
 * resource selection use compile-time layout/shape information only.
 *
 * ## Row algorithms
 *
 * Two row strategies exist behind one numerical contract:
 *
 * - **Offline (three passes over x)**: global max pass, exp+sum pass with the
 *   exponentials cached, then a normalized store pass reading the cache.
 *   Always applicable; used by `run_bound` and as the online fallback.
 * - **Online (tiled, single pass over x)**: rows are processed in
 *   `4 * vector_lanes` tiles; each tile's max and (locally centered) exp-sum
 *   are computed in one read pass, then reconciled to the global max, and the
 *   store pass re-uses the cached exponentials. Wins when the row is long
 *   enough that the second x read misses cache but the tile caches fit.
 *
 * `should_use_online<InSpec, OutSpec>()` gates online selection on measured
 * shape windows (see its comment); non-finite intermediate results (all
 * -inf/NaN rows, underflowed sums) fall back to the offline path at runtime.
 *
 * ## Recipes
 *
 * Row loops are `kernel::loop::fold` instances parameterized by
 * `SoftmaxRowRecipe<Fold, Store>`: the fold unroll factor and the store-pass
 * vector blocking. See the recipe aliases below for the tuned combinations
 * and the platform-specific selection logic in `operator()`.
 */

namespace vecops::ops {

template <typename ComputeT = float32_t, vec::Accuracy ExpAccuracy = vec::Accuracy::Strict, bool AllowOnline = true>
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

namespace softmax_details {

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
VECOPS_INLINE void validate_softmax_layouts(const InLayout& in, const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1);
  static_assert(OutLayout::Ndim == InLayout::Ndim);
  for (int d = 0; d < InLayout::Ndim; ++d) {
    VECOPS_ASSERT(in.shape()[d] == out.shape()[d], "Softmax input/output shape mismatch");
  }
  VECOPS_ASSERT(in.shape()[InLayout::Ndim - 1] > 0, "Softmax normalized dimension must be non-empty");
}

template <int Fold, int Store>
/**
 * @brief Fold unrolling and store blocking for one softmax row loop family.
 * @tparam Fold  `kernel::loop::fold` unroll factor for the reduction passes.
 * @tparam Store Vector blocking of the normalized store pass; also controls
 *               whether the store re-loads exponentials one or two vectors at
 *               a time (paired stores benefit some conversion widths).
 */
struct SoftmaxRowRecipe {
  static constexpr int FoldFactor = Fold;
  static constexpr int StoreFactor = Store;
};

/// Default recipe: no unrolling, two-vector store blocks.
using GenericSoftmaxRecipe = SoftmaxRowRecipe<1, 2>;
/// SVE packed-BF16 output recipe (see the packed-output path in operator()).
using PackedSoftmaxRecipe = SoftmaxRowRecipe<1, 2>;
/// Throughput recipe for large AVX-512 rows: 4x-unrolled folds and stores.
using UnrolledSoftmaxRecipe = SoftmaxRowRecipe<4, 4>;
/// Single-vector stores; avoids GCC's slower paired-conversion lowering.
using SingleStoreSoftmaxRecipe = SoftmaxRowRecipe<1, 1>;

} // namespace softmax_details

template <typename Config = SoftmaxConfig<>>
/**
 * @brief Softmax operator normalizing the final dimension of an N-D operand.
 * @tparam Config `SoftmaxConfig`-compatible type.
 *
 * Input and output shapes must match and the last dimension must be non-empty.
 * `ResourceRequirements` lets ExecutionSession reject incompatible hardware
 * modes at compile time without introducing runtime capability tests.
 */
class Softmax {
  // Row-level access planning: vector axis 0 is the normalized dimension.
  // XPolicy declares two read passes (max pass, then exp/sum pass) with a
  // deferred-materialization plan: the max pass populates the materialized
  // buffer, so the second pass re-reads the already-converted values.
  using XPolicy = tensor::InputAccessPolicy<0, 2, tensor::AccessPlan::automatic_deferred>;
  // One streaming write pass along the normalized dimension.
  using YPolicy = tensor::OutputAccessPolicy<0>;

public:
  using ComputeType = typename Config::ComputeType;
  using Tag = vec::ScalableTag<ComputeType, 0>;
  /** Compile-time hardware-mode contract consumed by ExecutionSession. */
  using ResourceRequirements = typename execution::details::current_backend_t::DefaultRequirements;
  static constexpr vec::Accuracy ExpAccuracy = Config::exp_accuracy;

  const Config config;

  VECOPS_INLINE constexpr explicit Softmax(Config cfg = {})
    : config(cfg) {
  }

  /**
   * Address-independent Softmax plan with one private scratch replica per
   * logical lane. The planning Specs carry layout capacity only; every call
   * rebinds them to an actual Tensor/Spec and validates that dynamic extents
   * do not exceed that capacity.
   */
  template <nint_t Parallelism, typename InPattern, typename OutPattern,
            typename WorkerScratch>
  class PatternPrepared {
  public:
    static_assert(Parallelism > 0,
                  "prepared Softmax parallelism must be positive");
    using ResourceRequirements = typename Softmax::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(
        Config config, InPattern in, OutPattern out,
        WorkerScratch worker_scratch, nint_t workspace_bytes)
      : config_(std::move(config))
      , in_(std::move(in))
      , out_(std::move(out))
      , worker_scratch_(std::move(worker_scratch))
      , workspace_bytes_(workspace_bytes) {
    }

    template <tensor::InputOperand In, tensor::OutputOperand Out>
      requires (tensor::is_bound_tensor_view_v<In> &&
                tensor::is_bound_tensor_view_v<Out>)
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        In&& in, Out&& out) const {
      auto active_in = tensor::rebind(in_, std::forward<In>(in));
      auto active_out = tensor::rebind(out_, std::forward<Out>(out));
      const nint_t active_bytes = Softmax{config_}.required_workspace(
          active_in, active_out);
      VECOPS_CHECK(
          active_bytes <= workspace_bytes_,
          "active Softmax scratch exceeds its planning-pattern capacity");
      auto scratch = task.local(worker_scratch_);
      kernel::WorkspaceView workspace{scratch.data(), workspace_bytes_};
      ExecutionSession execution{workspace};
      Softmax{config_}(execution, active_in, active_out);
    }

  private:
    Config config_;
    InPattern in_;
    OutPattern out_;
    WorkerScratch worker_scratch_;
    nint_t workspace_bytes_ = 0;
  };

  /** Prepare Softmax from storage-less operand patterns. */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::UnboundInputSpecLike InPattern,
            tensor::UnboundOutputSpecLike OutPattern>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) {
      authority.template worker_tensor<std::byte, Parallelism>(
          std::string_view{}, tensor::make_shape(meta::Any{bytes}));
    }
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& parent, std::string_view site_name,
      InPattern&& in, OutPattern&& out) const {
    using In = std::remove_cvref_t<InPattern>;
    using Out = std::remove_cvref_t<OutPattern>;
    static_assert(std::same_as<typename In::ComputeType, ComputeType>,
                  "Softmax input pattern compute type must match Config");
    static_assert(std::same_as<typename Out::ComputeType, ComputeType>,
                  "Softmax output pattern compute type must match Config");
    auto in_pattern = std::forward<InPattern>(in);
    auto out_pattern = std::forward<OutPattern>(out);
    const nint_t bytes = required_workspace(in_pattern, out_pattern);
    auto scratch = parent.template worker_tensor<std::byte, Parallelism>(
        site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<
        Parallelism, decltype(in_pattern), decltype(out_pattern),
        decltype(scratch)>{
          config, std::move(in_pattern), std::move(out_pattern),
          std::move(scratch), bytes};
  }

  /** Centers an already-loaded block, exponentiates it, and caches it.
   *  Shared leaf of the online and offline row paths so the numerical
   *  sequence (subtract max, exp_neg, masked store) has one definition. */
  template <typename BlockTag, typename Active>
  VECOPS_ALWAYS_INLINE static auto exp_cache_block(BlockTag block_tag, nint_t col, Active active,
                                                   vec::Vec<BlockTag> centered, ComputeType* exp_cache) {
    const auto zero = vec::zeros(block_tag);
    auto exponential =
      vec::exp_neg(block_tag, centered, vec::opt::math::accuracy<ExpAccuracy>, active, vec::opt::merge(zero));
    vec::store(block_tag, exp_cache + col, exponential, active);
    return exponential;
  }

  /** Loads a cached block, scales it, and stores it to the output.
   *  @p col is the row-local lane offset of the block. */
  template <typename Y, typename BlockTag, typename Active, typename ScaleV>
  VECOPS_ALWAYS_INLINE static void store_normalized_block(Y& y, BlockTag block_tag, nint_t col, Active active,
                                                          ScaleV scale_v, const ComputeType* exp_cache) {
    auto exponential = vec::load(block_tag, exp_cache + col, active);
    y.store(block_tag, tensor::coord(col), vec::mul(exponential, scale_v), active);
  }

  template <tensor::InputSpecLike InSpec, tensor::OutputSpecLike OutSpec>
  /**
   * @brief Compute exact workspace bytes for normalized input/output Specs.
   * @param in Readable input Spec.
   * @param out Writable output Spec with the same shape.
   * @return Cache, DataAccess materialization, and alignment bytes.
   */
  VECOPS_INLINE nint_t required_workspace(const InSpec& in, const OutSpec& out) const {
    softmax_details::validate_softmax_layouts(in.input_layout(), out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    const auto in_row = tensor::take_trailing<1>(in);
    const auto out_row = tensor::take_trailing<1>(out);
    const nint_t n = static_cast<nint_t>(tensor::size<PrefixRank>(in.input_layout()));
    nint_t cache_bytes = kernel::WorkspaceView::allocation_bytes<ComputeType>(n);
    if constexpr (should_use_online<InSpec, OutSpec>()) {
      const nint_t tile_step = 4 * vec::size(Tag{});
      const nint_t tile_count = ceil_div(n, tile_step);
      // These are three separate aligned bump allocations at execution time.
      // Budget each allocation independently: rounding their combined payload
      // only once can undercount the two inter-allocation alignment gaps.
      cache_bytes += 2 * kernel::WorkspaceView::allocation_bytes<ComputeType>(tile_count);
    }
    return cache_bytes + tensor::required_workspace(in_row, XPolicy{}) + tensor::required_workspace(out_row, YPolicy{});
  }

  template <tensor::InputOperand InOperand, tensor::OutputOperand OutOperand>
    requires(!tensor::InputSpecLike<InOperand> || !tensor::OutputSpecLike<OutOperand>)
  /** Normalize Tensor operands to Specs and return exact workspace bytes. */
  VECOPS_INLINE nint_t required_workspace(const InOperand& in, const OutOperand& out) const {
    auto in_spec = tensor::as_input_spec<ComputeType>(in);
    auto out_spec = tensor::as_output_spec<ComputeType>(out);
    return required_workspace(in_spec, out_spec);
  }

  template <tensor::InputSpecLike InSpec, tensor::OutputSpecLike OutSpec>
  /**
   * @brief Execute normalized Specs with caller-owned workspace.
   * @param workspace Scratch storage at least `required_workspace()` bytes.
   * @param in Readable input Spec.
   * @param out Writable output Spec with identical shape.
   */
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const InSpec& in, const OutSpec& out) const {
    softmax_details::validate_softmax_layouts(in.input_layout(), out.output_layout());
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
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
    // SVE-only fast path: keep output in packed BF16 when the operator is
    // computing in FP32 anyway. Packing exponentials into BF16 registers
    // halves the store bandwidth, but only pays off once the problem is
    // large enough to be bandwidth-bound: element counts of 64 KiB and
    // below stay on the converting store path. (lower_bound_at_least_v with
    // 64*1024 + 1 expresses "> 64 KiB" — meta bounds are inclusive.)
    constexpr bool CanPackBf16Output = std::same_as<ComputeType, float32_t> &&
                                       std::same_as<typename OutSpec::MemoryElement, bfloat16_t> &&
                                       std::same_as<typename OutSpec::TransformType, tensor::NoTransform> &&
                                       tensor::is_ct_last_contiguous_v<typename OutSpec::OutputLayout, 1>;
    using ElementCount = tensor::numel_type_t<typename InSpec::InputLayout>;
    constexpr bool UsePackedBf16Output = CanPackBf16Output && meta::lower_bound_at_least_v<ElementCount, 64 * 1024 + 1>;
    if constexpr (UsePackedBf16Output) {
      if constexpr (should_use_online<InSpec, OutSpec>()) {
        run_rows.template operator()<true, softmax_details::PackedSoftmaxRecipe>();
      } else {
        run_rows.template operator()<false, softmax_details::PackedSoftmaxRecipe>();
      }
      return;
    }
#endif
    if constexpr (should_use_online<InSpec, OutSpec>()) {
      run_rows.template operator()<true, softmax_details::SingleStoreSoftmaxRecipe>();
    } else {
#if defined(ARCH_X86_FAMILY) && defined(__GNUC__) && !defined(__clang__)
      // WORKAROUND/TUNING: GCC's paired BF16 conversion/store path increases
      // register pressure and emits a slower dependency chain on AVX-512.
      // Validate generated assembly before removing this compiler split.
      using GenericRecipe =
        std::conditional_t<std::same_as<typename OutSpec::MemoryElement, bfloat16_t>,
                           softmax_details::SingleStoreSoftmaxRecipe, softmax_details::GenericSoftmaxRecipe>;
#elif defined(CPU_CAPABILITY_SVE) && defined(__GNUC__) && !defined(__clang__)
      // WORKAROUND/TUNING: GCC 15 expands paired FP16 conversion/store into a
      // longer dependency chain than the single-vector SVE form. Clang still
      // benefits from the generic two-vector recipe.
      using GenericRecipe =
        std::conditional_t<std::same_as<typename OutSpec::MemoryElement, float16_t>,
                           softmax_details::SingleStoreSoftmaxRecipe, softmax_details::GenericSoftmaxRecipe>;
#else
      using GenericRecipe = softmax_details::GenericSoftmaxRecipe;
#endif
#if defined(CPU_CAPABILITY_AVX512)
      using NormalizedSize = tensor::size_type_t<PrefixRank, typename InSpec::InputLayout>;
      constexpr nint_t lanes = vec::size(Tag{});
      // Rows wider than these bounds keep the generic recipe: the unrolled
      // variant's extra live registers outweigh the unrolling once stores
      // dominate. Narrow (sub-compute-width) inputs allow a larger window
      // because their loads are proportionally cheaper.
      constexpr nint_t max_unrolled_n = sizeof(typename InSpec::MemoryElement) < sizeof(ComputeType) ? 4096 : 1024;
      // For an unconstrained caller, prefer the throughput recipe used by
      // common neural-network widths. Explicit bounds outside the tuned range
      // select the smaller generic recipe without a runtime branch.
      constexpr bool UseUnrolled =
        meta::range_within_v<NormalizedSize, 16 * lanes, max_unrolled_n> || !meta::is_bounded_v<NormalizedSize>;
      if constexpr (UseUnrolled) {
        run_rows.template operator()<false, softmax_details::UnrolledSoftmaxRecipe>();
      } else {
        run_rows.template operator()<false, GenericRecipe>();
      }
#else
      run_rows.template operator()<false, GenericRecipe>();
#endif
    }
  }

  template <tensor::InputOperand InOperand, tensor::OutputOperand OutOperand>
    requires(!tensor::InputSpecLike<InOperand> || !tensor::OutputSpecLike<OutOperand>)
  /** Execute Tensor operands using caller-owned workspace. */
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, const InOperand& in, const OutOperand& out) const {
    auto in_spec = tensor::as_input_spec<ComputeType>(in);
    auto out_spec = tensor::as_output_spec<ComputeType>(out);
    (*this)(workspace, in_spec, out_spec);
  }

  template <execution::ExecutionScope Scope, tensor::InputOperand InOperand, tensor::OutputOperand OutOperand>
  /**
   * @brief Execute through an ExecutionSession or active enclosing scope.
   * @param scope Scope providing resource transitions and worker workspace.
   * @param in Readable Tensor or Spec operand.
   * @param out Writable Tensor or Spec operand with identical shape.
   *
   * A nested call under an already-compatible region compiles to no additional
   * hardware transition because resource membership is part of `Scope`'s type.
   */
  VECOPS_INLINE void operator()(Scope& scope, const InOperand& in, const OutOperand& out) const {
    scope.with_resources(*this, [&](auto& active) VECOPS_INLINE_LAMBDA { (*this)(active.workspace_view(), in, out); });
  }

  template <tensor::InputOperand InOperand, tensor::OutputOperand OutOperand>
  /** Allocate exact workspace, create an ExecutionSession, and execute once. */
  VECOPS_INLINE void operator()(const InOperand& in, const OutOperand& out) const {
    kernel::Workspace storage(required_workspace(in, out));
    auto workspace = storage.view();
    ExecutionSession execution{workspace};
    (*this)(execution, in, out);
  }

  template <tensor::BoundInputAccess X, tensor::CommittableBoundOutputAccess Y>
  /**
   * @brief Execute one already-bound row without replanning or rebinding.
   * @param exp_cache Storage for one `ComputeType` value per logical lane.
   * @param x Bound rank-one readable access.
   * @param y Bound rank-one committable writable access.
   */
  VECOPS_INLINE void run_bound(ComputeType* exp_cache, X& x, Y& y) const {
    static_assert(X::Rank == 1 && Y::Rank == 1);
    const nint_t n = tensor::logical_layout(x).shape()[0];
    VECOPS_ASSERT(exp_cache != nullptr, "Softmax bound cache is null");
    run_bound_cached<softmax_details::GenericSoftmaxRecipe, false>(n, exp_cache, x, y);
  }

private:
  /**
   * @brief Compile-time gate for the online (single-pass) row algorithm.
   *
   * Online tiling only wins in measured windows: the row must be long enough
   * that the offline path's second read of x no longer hits cache, while the
   * total element count must be large enough for the difference to matter
   * (the store pass re-reads cached exponentials either way). Outside the
   * tuned windows — or without compile-time last-dim contiguity, which the
   * tile loop's addressing relies on — the offline three-pass path is used.
   * Unknown shape metadata conservatively disables online selection.
   *
   * Decision table (all bounds inclusive; tuned in SoftmaxAB.md):
   *
   * | Backend | Compute/Input | Accuracy | Normalized size | Element count |
   * |---|---|---|---|---|
   * | SVE | f32/f32 | != Estimate | [8192, 16384] | [512K, 8M] |
   * | AVX-512 | f32/f32 | Strict | [2048, 4096] | >= 512K |
   * | AVX-512 | f32/f32 | other | [2048, 32768] | >= 512K |
   * | AVX-512 | f64/f64 | Estimate | >= 2048 | >= 512K |
   * | AVX-512 | f64/f64 | other | >= 2048 | >= 16M |
   * | AVX-512 | f32/f16 | Estimate | exactly 2048 | >= 16M |
   *
   * Every other combination (including all other backends) returns false.
   */
  template <tensor::InputSpecLike InSpec, tensor::OutputSpecLike OutSpec>
  static consteval bool should_use_online() {
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
    if constexpr (!softmax_details::softmax_online_allowed<Config>())
      return false;
    constexpr int Rank = InSpec::InputTensor::Ndim;
    if constexpr (!tensor::is_ct_last_contiguous_v<typename InSpec::InputLayout, 1> ||
                  !tensor::is_ct_last_contiguous_v<typename OutSpec::OutputLayout, 1>)
      return false;
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    using NormalizedSize = tensor::size_type_t<Rank - 1, typename InSpec::InputLayout>;
    using ElementCount = tensor::numel_type_t<typename InSpec::InputLayout>;
#  if defined(CPU_CAPABILITY_SVE)
    if constexpr (std::same_as<ComputeType, float32_t> && std::same_as<InElement, float32_t> &&
                  ExpAccuracy != vec::Accuracy::Estimate) {
      return meta::range_within_v<NormalizedSize, 8192, 16384> &&
             meta::range_within_v<ElementCount, 512 * 1024, 8 * 1024 * 1024>;
    } else {
      return false;
    }
#  else
    if constexpr (std::same_as<ComputeType, float32_t> && std::same_as<InElement, float32_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Strict) {
        return meta::range_within_v<NormalizedSize, 2048, 4096> &&
               meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      } else {
        return meta::range_within_v<NormalizedSize, 2048, 32768> &&
               meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      }
    } else if constexpr (std::same_as<ComputeType, float64_t> && std::same_as<InElement, float64_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return meta::lower_bound_at_least_v<NormalizedSize, 2048> &&
               meta::lower_bound_at_least_v<ElementCount, 512 * 1024>;
      } else {
        return meta::lower_bound_at_least_v<NormalizedSize, 2048> &&
               meta::lower_bound_at_least_v<ElementCount, 16 * 1024 * 1024>;
      }
    } else if constexpr (std::same_as<ComputeType, float32_t> && std::same_as<InElement, float16_t>) {
      if constexpr (ExpAccuracy == vec::Accuracy::Estimate) {
        return meta::range_within_v<NormalizedSize, 2048, 2048> &&
               meta::lower_bound_at_least_v<ElementCount, 16 * 1024 * 1024>;
      } else {
        return false;
      }
    } else {
      return false;
    }
#  endif
#else
    return false;
#endif
  }

  /**
   * @brief Online (tiled, single-read-pass) softmax for one row.
   *
   * Three phases, reading x exactly once:
   * 1. **Tile pass**: per `4 * vector_lanes` tile, compute the tile max,
   *    cache the exponentials of `x - tile_max`, and record both the tile
   *    max and the tile's exp-sum.
   * 2. **Reconciliation**: reduce tile maxima to `global_max`, then convert
   *    every tile's sum to the global baseline (see the rescaling comment
   *    below) and accumulate `global_sum`.
   * 3. **Store pass**: for each tile, multiply the cached exponentials by
   *    `rescale(tile) / global_sum` and store.
   *
   * Numerical fallback: a non-finite `global_max` (a row that is entirely
   * -inf/NaN) or a non-finite/non-positive `global_sum` (underflow) cannot
   * be normalized this way. The already-bound operands and `exp_cache` are
   * reused by the offline row kernel, so normal and fallback data follow the
   * same workspace-allocation path and an owning output remains alive through
   * its fallback commit.
   */
  template <typename Recipe, typename InSpec, typename OutSpec>
  VECOPS_NOINLINE void run_row_online(kernel::WorkspaceView& workspace, const InSpec& in, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    Tag tag{};
    const nint_t n = static_cast<nint_t>(tensor::size<0>(in.input_layout()));
    const nint_t tile_step = 4 * vec::size(tag);
    const nint_t tile_count = ceil_div(n, tile_step);
    const auto mark = workspace.mark();
    ComputeType* exp_cache = static_cast<ComputeType*>(
      workspace.allocate(n * static_cast<nint_t>(sizeof(ComputeType)), vec::DEFAULT_ALIGNMENT));
    ComputeType* tile_max_cache = workspace.allocate<ComputeType>(tile_count);
    ComputeType* tile_sum_cache = workspace.allocate<ComputeType>(tile_count);
    kernel::with_operands(
      workspace, tensor::operand(in, XPolicy{}), tensor::operand(out, YPolicy{}),
      [this, n, tile_step, tile_count, exp_cache, tile_max_cache, tile_sum_cache](auto& x, auto& y) {
        Tag tag{};
        const ComputeType negative_infinity = -std::numeric_limits<ComputeType>::infinity();
        constexpr bool CopyDirectInput = sizeof(typename InSpec::MemoryElement) == sizeof(ComputeType) &&
                                         tensor::details::resolve_plan<InSpec, XPolicy>() == tensor::AccessPlan::direct;
        nint_t tile = 0;
        // Phase 1. Loads merge -inf into inactive lanes: the tile max is
        // unaffected (no real element can be smaller), and the exp cache
        // stores exp(-inf)=0 there, keeping the tile sum untouched.
        kernel::loop::fold<4, 4>(
          tag, n,
          [&](auto block_tag, nint_t col, auto active, const auto& negative_infinity_v) VECOPS_INLINE_LAMBDA {
            auto value = x.load(block_tag, tensor::coord(col), active, vec::opt::merge(negative_infinity_v),
                                tensor::materialize::populate);
            const ComputeType tile_max = vec::reduce_max(block_tag, value);
            const auto maximum = vec::fill(block_tag, tile_max);
            auto exponential = exp_cache_block(block_tag, col, active, vec::sub(value, maximum), exp_cache);
            tile_max_cache[tile] = tile_max;
            tile_sum_cache[tile] = vec::reduce_add(block_tag, exponential);
            ++tile;
          },
          kernel::loop::invariant(negative_infinity));

        ComputeType global_max{};
        kernel::loop::fold(
          tag, tile_count,
          [&](auto block_tag, nint_t i, auto active, auto& maximum) VECOPS_INLINE_LAMBDA {
            auto tile_max = vec::load(block_tag, tile_max_cache + i, active, vec::opt::merge(negative_infinity));
            maximum = vec::max(maximum, tile_max);
          },
          kernel::loop::reduce_max(global_max));
        // An all--inf/NaN row has no finite maximum: the offline path (via
        // its own isfinite-free arithmetic) produces the reference output
        // for such rows, so bail out before dividing by a broken baseline.
        if (!std::isfinite(static_cast<double>(global_max))) {
          run_bound_fallback<Recipe, CopyDirectInput>(n, exp_cache, x, y);
          return;
        }

        ComputeType global_sum{};
        // Phase 2. Rescale tile sums to the global baseline and reuse
        // tile_max_cache in a second role: each slot is overwritten in
        // place with exp(tile_max - global_max), the per-tile correction
        // factor the store pass will need. This avoids a fourth array.
        kernel::loop::fold(
          tag, tile_count,
          [&](auto block_tag, nint_t i, auto active, auto& sum) VECOPS_INLINE_LAMBDA {
            auto tile_max = vec::load(block_tag, tile_max_cache + i, active, vec::opt::merge(global_max));
            const auto maximum = vec::fill(block_tag, global_max);
            auto centered = vec::sub(tile_max, maximum);
            const auto zero = vec::zeros(block_tag);
            auto scale =
              vec::exp_neg(block_tag, centered, vec::opt::math::accuracy<ExpAccuracy>, active, vec::opt::merge(zero));
            vec::store(block_tag, tile_max_cache + i, scale, active);
            auto tile_sum = vec::load(block_tag, tile_sum_cache + i, active, vec::opt::merge(ComputeType(0)));
            // sum += exp(tile_max - global_max) * tile_sum: each tile's
            // sum was centered on its own tile_max, so it must be
            // rebased onto global_max before the tiles can be added.
            sum = vec::fmadd(scale, tile_sum, sum);
          },
          kernel::loop::reduce_add(global_sum));
        // A denormal/underflowed sum (or a NaN produced by the rescale)
        // would divide the row by garbage; fall back instead. The check
        // `> 0` also rejects a zero sum, whose rows are constant -inf.
        if (!std::isfinite(static_cast<double>(global_sum)) || !(global_sum > ComputeType(0))) {
          run_bound_fallback<Recipe, CopyDirectInput>(n, exp_cache, x, y);
          return;
        }

        nint_t col = 0;
        const ComputeType inverse_sum = ComputeType(1) / global_sum;
        // Phase 3. tile_max_cache[tile] now holds exp(tile_max -
        // global_max), so the store scale folds both the tile rebase and
        // the row normalization into one multiply per tile.
        for (tile = 0; tile < tile_count; ++tile) {
          const nint_t count = std::min(tile_step, n - col);
          const ComputeType scale = tile_max_cache[tile] * inverse_sum;
          kernel::loop::fold<Recipe::StoreFactor>(
            tag, count,
            [&](auto block_tag, nint_t local, auto active, const auto& scale_v)
              VECOPS_INLINE_LAMBDA { store_normalized_block(y, block_tag, col + local, active, scale_v, exp_cache); },
            kernel::loop::invariant(scale));
          col += count;
        }
        y.commit();
      });
    workspace.rewind(mark);
  }

  /** Keep the data-exception path out of the normal online row body while
   *  reusing the online row's already-bound operands and cache allocation. */
  template <typename Recipe, bool CopyDirectInput, tensor::BoundInputAccess X, tensor::CommittableBoundOutputAccess Y>
  VECOPS_NOINLINE void run_bound_fallback(nint_t n, ComputeType* exp_cache, X& x, Y& y) const {
    run_bound_cached<Recipe, CopyDirectInput>(n, exp_cache, x, y);
  }

  template <typename Recipe, bool CopyDirectInput, tensor::BoundInputAccess X, tensor::CommittableBoundOutputAccess Y>
  VECOPS_ALWAYS_INLINE void run_bound_cached(nint_t n, ComputeType* exp_cache, X& x, Y& y) const {
    Tag tag{};
    const ComputeType negative_infinity = -std::numeric_limits<ComputeType>::infinity();
    ComputeType max_value{};
    auto fold_max = [&](auto&& unordered_x) VECOPS_INLINE_LAMBDA {
      kernel::loop::fold<Recipe::FoldFactor>(
        tag, n,
        [&](auto block_tag, nint_t col, auto active, auto& maximum) VECOPS_INLINE_LAMBDA {
          auto value = unordered_x.load(block_tag, tensor::coord(col), active, vec::opt::merge(negative_infinity),
                                        tensor::materialize::populate);
          maximum = vec::max(maximum, value);
        },
        kernel::loop::reduce_max(max_value));
    };
    if constexpr (sizeof(typename X::MemoryElement) == sizeof(ComputeType)) {
      // Equal-width elements need no conversion, so the max reduction can
      // read x directly (with_unordered_access would only add overhead);
      // lane provenance stays ordered. The by-value copy of the direct
      // access session is a codegen aid kept for GCC (see SoftmaxAB.md);
      // it is only needed when the resolved plan is `direct`.
      if constexpr (CopyDirectInput) {
        auto direct_x = x;
        fold_max(direct_x);
      } else {
        fold_max(x);
      }
    } else {
      // Narrow inputs (e.g. fp16 into fp32 compute) reduce through an
      // unordered (gathered-lane) access session for lane-provenance
      // correctness during the conversion.
      tensor::with_unordered_access(x, fold_max);
    }

    ComputeType sum{};
    kernel::loop::fold<Recipe::FoldFactor>(
      tag, n,
      [&](auto block_tag, nint_t col, auto active, auto& sum_vector) VECOPS_INLINE_LAMBDA {
        auto value = x.load(block_tag, tensor::coord(col), active, vec::opt::merge(negative_infinity));
        const auto maximum = vec::fill(block_tag, max_value);
        auto exponential = exp_cache_block(block_tag, col, active, vec::sub(value, maximum), exp_cache);
        sum_vector = vec::add(sum_vector, exponential);
      },
      kernel::loop::reduce_add(sum));

    const ComputeType inverse_sum = ComputeType(1) / sum;
    kernel::loop::fold<Recipe::StoreFactor>(
      tag, n,
      [&](auto block_tag, nint_t col, auto active, const auto& inverse)
        VECOPS_INLINE_LAMBDA { store_normalized_block(y, block_tag, col, active, inverse, exp_cache); },
      kernel::loop::invariant(inverse_sum));
    y.commit();
  }

  template <typename Recipe, typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row(kernel::WorkspaceView& workspace, const InSpec& in, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    nint_t n = static_cast<nint_t>(tensor::size<0>(in.input_layout()));
#if defined(CPU_CAPABILITY_SVE) && defined(__GNUC__) && !defined(__clang__)
    if constexpr (std::same_as<typename InSpec::MemoryElement, float16_t>) {
      // WORKAROUND: GCC 15 otherwise propagates large Const-shaped FP16 row
      // counts into the fold and aggressively unrolls them, causing excessive
      // code size and compile time. Only the trip-count value is obscured;
      // access plans and recipes remain compile-time choices.
      asm volatile("" : "+r"(n));
    }
#endif
    const auto mark = workspace.mark();
    ComputeType* exp_cache = static_cast<ComputeType*>(
      workspace.allocate(n * static_cast<nint_t>(sizeof(ComputeType)), vec::DEFAULT_ALIGNMENT));
    auto compute = [this, n, exp_cache](auto& x, auto& y) VECOPS_INLINE_LAMBDA {
      constexpr bool CopyDirectInput = sizeof(typename InSpec::MemoryElement) == sizeof(ComputeType) &&
                                       tensor::details::resolve_plan<InSpec, XPolicy>() == tensor::AccessPlan::direct;
      run_bound_cached<Recipe, CopyDirectInput>(n, exp_cache, x, y);
    };
    kernel::with_operands(workspace, tensor::operand(in, XPolicy{}), tensor::operand(out, YPolicy{}), compute);
    workspace.rewind(mark);
  }
};

template <typename Config = SoftmaxConfig<>>
/** Construct a Softmax operator from its compile-time configuration type. */
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_DETAILS_SOFTMAX_OPERATION_H
