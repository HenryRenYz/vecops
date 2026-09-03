//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_QUANTIZATION_H
#define VECOPS_MATMUL_QUANTIZATION_H

#include <cstdint>
#include <type_traits>

#include "vecops/tensor/Transform.h"

/**
 * @file vecops/matmul/Quantization.h
 * @brief Runtime activation quantization / output dequantization transforms
 *        for integer matmul paths.
 *
 * Two named `tensor::VecTransform` types cover the FP32 bookkeeping around
 * an integer matmul:
 *
 * - `RuntimePerRowAsymmetricQuantizeTransform` — pre-scales the FP32
 *   activation operand A row by row and adds a zero point
 *   (`q = x * multiplier[row] + zero_point`), feeding an INT8/UINT8 weight
 *   operand;
 * - `RuntimePerRowColumnDequantizeTransform` — the output epilogue
 *   (`c = acc * row_scale[row] * column_scale[col]`).
 *
 * Both are *runtime* transforms: scales arrive as pointers, not template
 * parameters. Beyond the generic execution path, the named types let fused
 * backend leaves (e.g. the SME runtime-quant INT8 GEMV) recognize a
 * transform through the `is_..._transform_v` traits and read its scale
 * arrays directly via the stable `Parameters` view, without depending on
 * lambda object layout.
 *
 * ## Pitfalls
 *
 * - The transforms compute in the FP32 domain; narrowing into the integer
 *   operand element happens where the operand is consumed.
 * - Scale arrays are indexed by logical coordinates (row = second-to-last
 *   axis); rank-3 inputs additionally offset by `batch * batch_stride`.
 * - `is_lane_local = true` is a performance promise to the execution layer;
 *   keep it true only while output lane i depends solely on input lane i.
 */
namespace vecops::matmul {

/**
 * @brief Stable parameter view for runtime per-row asymmetric A quantization.
 *
 * @param multipliers   Per-row FP32 scale factors, `multipliers[row]`.
 * @param zero_point    Zero point added after scaling; read as a single
 *                      value shared by the whole matrix (as both the
 *                      transform and the fused kernels do).
 * @param batch_stride  Rows per batch: for rank-3 operands the scale index
 *                      steps by `batch_stride` per leading batch
 *                      coordinate; ignored for rank-2 operands.
 */
struct RuntimePerRowAsymmetricQuantizeParameters {
  const float32_t* multipliers = nullptr;
  const int32_t* zero_point = nullptr;
  nint_t batch_stride = 0;

  /// Map a logical coordinate to its index into `multipliers`. The last
  /// axis is K (columns), so the row is the second-to-last coordinate; a
  /// rank-3 coordinate adds the batch offset in front.
  template <typename Coord>
  VECOPS_ALWAYS_INLINE constexpr nint_t row_scale_index(
      const Coord& coordinate) const {
    const auto rank = coordinate.size();
    const nint_t row = coordinate[rank - 2];
    const nint_t batch = rank == 3 ? coordinate[0] : 0;
    return batch * batch_stride + row;
  }
};

/**
 * @brief Coordinate-aware FP32 transform shared by every matmul backend.  The named
 * type also exposes a stable parameter view to architecture-specific fused
 * leaves without relying on lambda object layout.
 *
 * Computes `q = x * multipliers[row] + float(*zero_point)` per element, in
 * the FP32 domain, on the operand lanes' own logical coordinates. The
 * result feeds an INT8/UINT8 quantized operand; the integer cast happens
 * where the operand is consumed, not here.
 *
 * @note `is_lane_local = true` promises the execution layer that output
 *       lane i depends only on input lane i (see `tensor/Transform.h`),
 *       keeping full-vector traversal legal wherever no spatial packing
 *       mixes rows inside one vector.
 */
struct RuntimePerRowAsymmetricQuantizeTransform
    : tensor::VecTransform<float32_t, float32_t, false> {
  static constexpr bool is_lane_local = true;
  using Parameters = RuntimePerRowAsymmetricQuantizeParameters;

  constexpr explicit RuntimePerRowAsymmetricQuantizeTransform(
      Parameters parameters) : parameters_(parameters) {}

  /// Stable access to the scale arrays; fused leaves use this instead of
  /// capturing the transform by its lambda-free layout.
  VECOPS_ALWAYS_INLINE constexpr const Parameters& parameters() const {
    return parameters_;
  }

  /// Applies the per-row affine quantization map. Only instantiated for
  /// scalable FP32 vector tags.
  template <vec::VectorTag Tag, typename Context>
    requires vec::is_scalable_tag_v<Tag> &&
             std::same_as<vec::ElementOf<Tag>, float32_t>
  VECOPS_KERNEL_FUNCTION(vec::Vec<Tag> operator()(
      Tag tag, vec::Vec<Tag> value, const Context& context) const) {
    const auto coordinate0 = context.lane_coord(0);
    const auto rank = coordinate0.size();
    // The int32->float32 zero-point conversion is identical for every lane
    // and every row, so hoist it out of the per-lane loop below.
    const float32_t zero_point =
        static_cast<float32_t>(*parameters_.zero_point);

    // Normal K traversal keeps every vector lane on one logical row. Spatial
    // packing may vary row or batch and therefore needs per-lane coordinates.
    if (context.vector_axis == static_cast<int>(rank) - 1) {
      const nint_t scale_index = parameters_.row_scale_index(coordinate0);
      return vec::add(
          tag,
          vec::mul(
              tag, value,
              vec::fill(tag, parameters_.multipliers[scale_index])),
          vec::fill(tag, zero_point));
    }

    auto result = value;
    for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
      if (!context.is_active(lane)) continue;
      const auto coordinate = context.lane_coord(lane);
      const nint_t scale_index = parameters_.row_scale_index(coordinate);
      result = vec::set(
          tag, result, lane,
          vec::get(tag, value, lane) * parameters_.multipliers[scale_index] +
              zero_point);
    }
    return result;
  }

private:
  Parameters parameters_;
};

/// Factory shorthand: wraps `parameters` into the named transform type so
/// it can be attached to a tensor input spec.
VECOPS_ALWAYS_INLINE constexpr auto
make_runtime_per_row_asymmetric_quantize_transform(
    RuntimePerRowAsymmetricQuantizeParameters parameters) {
  return RuntimePerRowAsymmetricQuantizeTransform{parameters};
}

/**
 * @brief Stable parameter view for row-by-column FP32 dequantization.
 *
 * @param row_scales    Per-row (per-output-row) scale factors, indexed like
 *                      the quantization multipliers.
 * @param column_scales Per-column scale factors, `column_scales[column]`;
 *                      assumed contiguous in the column coordinate.
 * @param batch_stride  Rows per batch; offsets the row scale index for
 *                      rank-3 operands, ignored for rank-2.
 */
struct RuntimePerRowColumnDequantizeParameters {
  const float32_t* row_scales = nullptr;
  const float32_t* column_scales = nullptr;
  nint_t batch_stride = 0;

  /// Map a logical coordinate to its index into `row_scales`. The last
  /// axis is K (columns), so the row is the second-to-last coordinate; a
  /// rank-3 coordinate adds the batch offset in front.
  template <typename Coord>
  VECOPS_ALWAYS_INLINE constexpr nint_t row_scale_index(
      const Coord& coordinate) const {
    const auto rank = coordinate.size();
    const nint_t row = coordinate[rank - 2];
    const nint_t batch = rank == 3 ? coordinate[0] : 0;
    return batch * batch_stride + row;
  }
};

/**
 * @brief Named epilogue shared by generic execution and optimized backend leaves.
 *
 * Computes `c = value * row_scales[row] * column_scales[column]` per output
 * element — the row scale undoes the activation quantization, the column
 * scale the weight quantization. Like the quantize transform, it exposes
 * its `Parameters` so fused leaves can read the scale arrays directly.
 *
 * @note `is_lane_local = true` promises the execution layer that output
 *       lane i depends only on input lane i (see `tensor/Transform.h`).
 */
struct RuntimePerRowColumnDequantizeTransform
    : tensor::VecTransform<float32_t, float32_t, false> {
  static constexpr bool is_lane_local = true;
  using Parameters = RuntimePerRowColumnDequantizeParameters;

  constexpr explicit RuntimePerRowColumnDequantizeTransform(
      Parameters parameters) : parameters_(parameters) {}

  /// Stable access to the scale arrays; fused leaves use this instead of
  /// capturing the transform by its lambda-free layout.
  VECOPS_ALWAYS_INLINE constexpr const Parameters& parameters() const {
    return parameters_;
  }

  /// Applies the row-by-column dequantization map. Only instantiated for
  /// scalable FP32 vector tags.
  template <vec::VectorTag Tag, typename Context>
    requires vec::is_scalable_tag_v<Tag> &&
             std::same_as<vec::ElementOf<Tag>, float32_t>
  VECOPS_KERNEL_FUNCTION(vec::Vec<Tag> operator()(
      Tag tag, vec::Vec<Tag> value, const Context& context) const) {
    const auto coordinate0 = context.lane_coord(0);
    const auto rank = coordinate0.size();
    // Normal K traversal keeps every vector lane on one logical row, so
    // only the column scale varies across lanes. Spatial packing may vary
    // row or batch as well and therefore needs per-lane coordinates.
    if (context.vector_axis == static_cast<int>(rank) - 1) {
      const nint_t scale_index = parameters_.row_scale_index(coordinate0);
      const nint_t column = coordinate0[rank - 1];
      return vec::mul(
          tag,
          vec::mul(
              tag, value,
              vec::fill(tag, parameters_.row_scales[scale_index])),
          // Contiguous gather-free load: the column scale vector is a
          // direct load at the lane coordinates' own columns.
          vec::load(tag, parameters_.column_scales + column));
    }

    auto result = value;
    for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
      if (!context.is_active(lane)) continue;
      const auto coordinate = context.lane_coord(lane);
      const nint_t scale_index = parameters_.row_scale_index(coordinate);
      const nint_t column = coordinate[rank - 1];
      result = vec::set(
          tag, result, lane,
          vec::get(tag, value, lane) * parameters_.row_scales[scale_index] *
              parameters_.column_scales[column]);
    }
    return result;
  }

private:
  Parameters parameters_;
};

/// Factory shorthand: wraps `parameters` into the named transform type so
/// it can be attached to a tensor output spec.
VECOPS_ALWAYS_INLINE constexpr auto
make_runtime_per_row_column_dequantize_transform(
    RuntimePerRowColumnDequantizeParameters parameters) {
  return RuntimePerRowColumnDequantizeTransform{parameters};
}

/// Exact-type recognition hook for the activation-quantize transform:
/// fused backend leaves test this trait to decide whether they can replace
/// the generic "transform then matmul" route with a fused kernel.
template <typename T>
inline constexpr bool is_runtime_per_row_asymmetric_quantize_transform_v =
    std::same_as<
        std::remove_cvref_t<T>,
        RuntimePerRowAsymmetricQuantizeTransform>;

/// Exact-type recognition hook for the dequantize epilogue transform, used
/// by fused leaves the same way as the quantize trait above.
template <typename T>
inline constexpr bool is_runtime_per_row_column_dequantize_transform_v =
    std::same_as<
        std::remove_cvref_t<T>,
        RuntimePerRowColumnDequantizeTransform>;

} // namespace vecops::matmul

namespace vecops::kernel::matmul_details {

/// Compatibility aliases for the original internal namespace.
using ::vecops::matmul::RuntimePerRowAsymmetricQuantizeParameters;
using ::vecops::matmul::RuntimePerRowAsymmetricQuantizeTransform;
using ::vecops::matmul::RuntimePerRowColumnDequantizeParameters;
using ::vecops::matmul::RuntimePerRowColumnDequantizeTransform;
using ::vecops::matmul::is_runtime_per_row_asymmetric_quantize_transform_v;
using ::vecops::matmul::is_runtime_per_row_column_dequantize_transform_v;
using ::vecops::matmul::make_runtime_per_row_asymmetric_quantize_transform;
using ::vecops::matmul::make_runtime_per_row_column_dequantize_transform;

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_QUANTIZATION_H
