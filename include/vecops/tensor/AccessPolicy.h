#ifndef VECOPS_TENSOR_ACCESS_POLICY_H
#define VECOPS_TENSOR_ACCESS_POLICY_H

#include <type_traits>

#include "vecops/tensor/AccessOptions.h"

/**
 * @file AccessPolicy.h
 * @brief Kernel-owned compile-time policies for Tensor DataAccess lowering.
 *
 * Operand Specs describe data supplied by a caller: Tensor, compute boundary
 * type, optional transform, and verifiable facts. Access policies instead
 * describe how one kernel uses that operand. Keeping the two separate prevents
 * callers from selecting unsafe lane reordering, materialization, or prefetch
 * strategies that only the kernel can validate.
 *
 * @code
 * using XPolicy = tensor::InputAccessPolicy<
 *     1,                              // vector axis
 *     2,                              // kernel reads each value twice
 *     tensor::AccessPlan::automatic,
 *     vec::cvt::Ordered>;
 * using YPolicy = tensor::OutputAccessPolicy<1>;
 *
 * kernel::with_operands(
 *     workspace,
 *     tensor::operand(x_spec, XPolicy{}),
 *     tensor::operand(y_spec, YPolicy{}),
 *     [&](auto& x, auto& y) {
 *       // hot loop
 *       y.commit();
 *     });
 * @endcode
 *
 * ## Automatic planning
 *
 * - Readless input transforms stay direct and do not read source memory.
 * - Unit physical stride on the vector axis is direct.
 * - A non-contiguous, single-pass input uses direct gather.
 * - A non-contiguous, multi-pass input materializes after transform.
 * - A non-contiguous output materializes only when another unit-stride axis
 *   can commit a complete rectangular region; otherwise it scatters directly.
 *
 * Dynamic choices are resolved outside the hot loop by `with_operands`, which
 * invokes a statically typed branch.
 *
 * ## Pitfalls
 *
 * - `VectorAxis` is a logical Tensor dimension, not a byte stride.
 * - `ReadPasses` must reflect the kernel algorithm; a wrong value changes the
 *   planner's cost decision but cannot be inferred from actual calls.
 * - `Unordered` conversion requires `PermutationSafe=true` and a
 *   permutation-equivariant transform. All related operands must use a
 *   compatible lane order; DataAccess cannot validate an entire algorithm.
 * - Forced materialization requires a bounded rectangular Tensor region and
 *   sufficient workspace.
 */
namespace vecops::tensor {

/**
 * @brief Requested storage path for one operand.
 *
 * For input, `before_transform` caches MemoryElement and
 * `after_transform` caches ComputeType. For output, `before_transform` caches
 * ComputeType and applies the epilogue during commit, while `after_transform`
 * caches MemoryElement after the epilogue and commit only remaps Layout.
 * `automatic` uses the fixed planner rules documented above.
 */
enum class AccessPlan {
  automatic,
  direct,
  materialize_before_transform,
  materialize_after_transform,
};

/**
 * @brief Low-level memory hints forwarded to the final memory operation.
 *
 * These types are hints/encodings understood by vec load/store conversion;
 * they do not change Tensor-level coordinates or active-lane semantics.
 */
template <
    typename Temporality = vec::mem::Temporal,
    typename Packing = vec::mem::Packed,
    typename Alignment = vec::mem::Unaligned>
struct MemoryPolicy {
  using TemporalityOption = Temporality;
  using PackingOption = Packing;
  using AlignmentOption = Alignment;
};

/** @brief Default temporal, packed, unaligned memory policy. */
using DefaultMemoryPolicy = MemoryPolicy<>;

/**
 * @brief Compile-time input access policy selected by a kernel.
 *
 * @tparam VectorAxis Default logical axis represented by vector lanes.
 * @tparam ReadPasses Algorithmic passes over each input region.
 * @tparam Plan Requested direct/materialized plan.
 * @tparam ConversionOrder Ordered or explicitly permitted unordered conversion.
 * @tparam ConversionValue Saturating/wrapping conversion behavior.
 * @tparam Memory Final memory-operation hints.
 * @tparam PermutationSafe Whether the kernel tolerates this lane permutation.
 * @tparam Prefetch Cursor prefetch policy.
 */
template <
    int VectorAxis,
    int ReadPasses = 1,
    AccessPlan Plan = AccessPlan::automatic,
    typename ConversionOrder = vec::cvt::Ordered,
    typename ConversionValue = vec::cvt::Saturate,
    typename Memory = DefaultMemoryPolicy,
    bool PermutationSafe = false,
    typename Prefetch = NoPrefetch>
struct InputAccessPolicy {
  static_assert(VectorAxis >= 0, "vector axis must be non-negative");
  static_assert(ReadPasses > 0, "input read pass count must be positive");
  static constexpr int vector_axis = VectorAxis;
  static constexpr int read_passes = ReadPasses;
  static constexpr AccessPlan requested_plan = Plan;
  static constexpr bool permutation_safe = PermutationSafe;
  using ConversionOrderOption = ConversionOrder;
  using ConversionValueOption = ConversionValue;
  using MemoryOptions = Memory;
  using PrefetchOptions = Prefetch;
};

/**
 * @brief Compile-time output access policy selected by a kernel.
 *
 * Output policies mirror input conversion, memory, permutation, and prefetch
 * controls but omit ReadPasses because output planning is governed by commit
 * layout rather than repeated reads.
 */
template <
    int VectorAxis,
    AccessPlan Plan = AccessPlan::automatic,
    typename ConversionOrder = vec::cvt::Ordered,
    typename ConversionValue = vec::cvt::Saturate,
    typename Memory = DefaultMemoryPolicy,
    bool PermutationSafe = false,
    typename Prefetch = NoPrefetch>
struct OutputAccessPolicy {
  static_assert(VectorAxis >= 0, "vector axis must be non-negative");
  static constexpr int vector_axis = VectorAxis;
  static constexpr AccessPlan requested_plan = Plan;
  static constexpr bool permutation_safe = PermutationSafe;
  using ConversionOrderOption = ConversionOrder;
  using ConversionValueOption = ConversionValue;
  using MemoryOptions = Memory;
  using PrefetchOptions = Prefetch;
};

/**
 * @brief Rebase a policy after slicing away one Tensor dimension.
 *
 * Slicing a dimension below the vector axis decrements the axis number.
 * Slicing the vector axis itself is rejected because the resulting scalar
 * projection has no equivalent vector-view policy.
 */
template <typename Policy, int SlicedDim>
struct SliceAccessPolicy;

template <
    int VectorAxis, int ReadPasses, AccessPlan Plan,
    typename ConversionOrder, typename ConversionValue, typename Memory,
    bool PermutationSafe, typename Prefetch, int SlicedDim>
struct SliceAccessPolicy<
    InputAccessPolicy<
        VectorAxis, ReadPasses, Plan, ConversionOrder, ConversionValue,
        Memory, PermutationSafe, Prefetch>,
    SlicedDim> {
  static_assert(
      SlicedDim != VectorAxis,
      "slicing away a DataAccess vector axis has no vector-view semantics");
  using type = InputAccessPolicy<
      VectorAxis - (SlicedDim < VectorAxis ? 1 : 0), ReadPasses, Plan,
      ConversionOrder, ConversionValue, Memory, PermutationSafe, Prefetch>;
};

template <
    int VectorAxis, AccessPlan Plan, typename ConversionOrder,
    typename ConversionValue, typename Memory, bool PermutationSafe,
    typename Prefetch, int SlicedDim>
struct SliceAccessPolicy<
    OutputAccessPolicy<
        VectorAxis, Plan, ConversionOrder, ConversionValue, Memory,
        PermutationSafe, Prefetch>,
    SlicedDim> {
  static_assert(
      SlicedDim != VectorAxis,
      "slicing away a DataAccess vector axis has no vector-view semantics");
  using type = OutputAccessPolicy<
      VectorAxis - (SlicedDim < VectorAxis ? 1 : 0), Plan,
      ConversionOrder, ConversionValue, Memory, PermutationSafe, Prefetch>;
};

/** @brief Result type of `SliceAccessPolicy`. */
template <typename Policy, int SlicedDim>
using SliceAccessPolicyT = typename SliceAccessPolicy<
    std::remove_cvref_t<Policy>, SlicedDim>::type;

namespace details {

template <typename T>
inline constexpr bool is_unordered_policy = std::same_as<
    typename std::remove_cvref_t<T>::ConversionOrderOption,
    vec::cvt::Unordered>;

} // namespace details

} // namespace vecops::tensor

#endif // VECOPS_TENSOR_ACCESS_POLICY_H
