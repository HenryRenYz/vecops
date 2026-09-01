//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_QUANTIZATION_H
#define VECOPS_MATMUL_QUANTIZATION_H

#include <cstdint>
#include <type_traits>

#include "vecops/tensor/Transform.h"

namespace vecops::matmul {

/** Stable parameter view for runtime per-row asymmetric A quantization. */
struct RuntimePerRowAsymmetricQuantizeParameters {
  const float32_t* multipliers = nullptr;
  const int32_t* zero_point = nullptr;
  nint_t batch_stride = 0;

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
 * Coordinate-aware FP32 transform shared by every matmul backend.  The named
 * type also exposes a stable parameter view to architecture-specific fused
 * leaves without relying on lambda object layout.
 */
struct RuntimePerRowAsymmetricQuantizeTransform
    : tensor::VecTransform<float32_t, float32_t, false> {
  static constexpr bool is_lane_local = true;
  using Parameters = RuntimePerRowAsymmetricQuantizeParameters;

  constexpr explicit RuntimePerRowAsymmetricQuantizeTransform(
      Parameters parameters) : parameters_(parameters) {}

  VECOPS_ALWAYS_INLINE constexpr const Parameters& parameters() const {
    return parameters_;
  }

  template <vec::VectorTag Tag, typename Context>
    requires vec::is_scalable_tag_v<Tag> &&
             std::same_as<vec::ElementOf<Tag>, float32_t>
  VECOPS_KERNEL_FUNCTION(vec::Vec<Tag> operator()(
      Tag tag, vec::Vec<Tag> value, const Context& context) const) {
    const auto coordinate0 = context.lane_coord(0);
    const auto rank = coordinate0.size();
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

VECOPS_ALWAYS_INLINE constexpr auto
make_runtime_per_row_asymmetric_quantize_transform(
    RuntimePerRowAsymmetricQuantizeParameters parameters) {
  return RuntimePerRowAsymmetricQuantizeTransform{parameters};
}

/** Stable parameter view for row-by-column FP32 dequantization. */
struct RuntimePerRowColumnDequantizeParameters {
  const float32_t* row_scales = nullptr;
  const float32_t* column_scales = nullptr;
  nint_t batch_stride = 0;

  template <typename Coord>
  VECOPS_ALWAYS_INLINE constexpr nint_t row_scale_index(
      const Coord& coordinate) const {
    const auto rank = coordinate.size();
    const nint_t row = coordinate[rank - 2];
    const nint_t batch = rank == 3 ? coordinate[0] : 0;
    return batch * batch_stride + row;
  }
};

/** Named epilogue shared by generic execution and optimized backend leaves. */
struct RuntimePerRowColumnDequantizeTransform
    : tensor::VecTransform<float32_t, float32_t, false> {
  static constexpr bool is_lane_local = true;
  using Parameters = RuntimePerRowColumnDequantizeParameters;

  constexpr explicit RuntimePerRowColumnDequantizeTransform(
      Parameters parameters) : parameters_(parameters) {}

  VECOPS_ALWAYS_INLINE constexpr const Parameters& parameters() const {
    return parameters_;
  }

  template <vec::VectorTag Tag, typename Context>
    requires vec::is_scalable_tag_v<Tag> &&
             std::same_as<vec::ElementOf<Tag>, float32_t>
  VECOPS_KERNEL_FUNCTION(vec::Vec<Tag> operator()(
      Tag tag, vec::Vec<Tag> value, const Context& context) const) {
    const auto coordinate0 = context.lane_coord(0);
    const auto rank = coordinate0.size();
    if (context.vector_axis == static_cast<int>(rank) - 1) {
      const nint_t scale_index = parameters_.row_scale_index(coordinate0);
      const nint_t column = coordinate0[rank - 1];
      return vec::mul(
          tag,
          vec::mul(
              tag, value,
              vec::fill(tag, parameters_.row_scales[scale_index])),
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

VECOPS_ALWAYS_INLINE constexpr auto
make_runtime_per_row_column_dequantize_transform(
    RuntimePerRowColumnDequantizeParameters parameters) {
  return RuntimePerRowColumnDequantizeTransform{parameters};
}

template <typename T>
inline constexpr bool is_runtime_per_row_asymmetric_quantize_transform_v =
    std::same_as<
        std::remove_cvref_t<T>,
        RuntimePerRowAsymmetricQuantizeTransform>;

template <typename T>
inline constexpr bool is_runtime_per_row_column_dequantize_transform_v =
    std::same_as<
        std::remove_cvref_t<T>,
        RuntimePerRowColumnDequantizeTransform>;

} // namespace vecops::matmul

namespace vecops::kernel::matmul_details {

// Compatibility aliases for the original internal namespace.
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
