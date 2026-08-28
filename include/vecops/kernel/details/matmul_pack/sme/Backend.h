//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_BACKEND_H

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/gemm/details/sme/Atoms.h"
#include "vecops/gemm/details/sme/Packing.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/kernel/details/matmul_pack/sme/Pack.h"
#include "vecops/kernel/details/matmul_pack/sme/TransformPack.h"

namespace vecops::kernel::matmul_pack_details {

template <>
struct Backend<
    gemm::details::sme::Format,
    matmul_pack_implementation::Vector> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    using Packing = gemm::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    using Tag = vec::ScalableTag<T, 0>;
    const auto& layout = source.spec().input_layout();
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    const nint_t panel = destination.spec().output_layout().shape()[2];
    generic::pack_interleaved_panels<Packing::KPack, Tag>(
        source, reinterpret_cast<T*>(destination.raw_data()),
        spatial, k, panel, ceil_div(k, Packing::KPack));
  }
};

template <>
struct Backend<
    gemm::details::sme::Format,
    matmul_pack_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sizeof(typename InputSpec::ComputeType) <= 4 &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement,
                   typename InputSpec::ComputeType> &&
      std::same_as<typename OutputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename OutputSpec::MemoryElement,
                   typename OutputSpec::ComputeType> &&
      std::same_as<
          tensor::stride_type_t<1, typename InputSpec::InputLayout>,
          meta::Const<1>>;

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = gemm::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    const auto& layout = source.spec().input_layout();
    static_assert(generic::RawDirectAccess<Source>);
    const auto* input = reinterpret_cast<const T*>(source.raw_data());
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    const nint_t row_stride = source.raw_strides()[0];
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack<Atom, Side>(
              input, spatial, k, row_stride, output);
        });
  }
};

template <typename InputSpec, typename OutputSpec>
inline constexpr bool sme_common_eligible_v =
    std::same_as<typename OutputSpec::TransformType, tensor::NoTransform> &&
    std::same_as<typename OutputSpec::MemoryElement,
                 typename OutputSpec::ComputeType> &&
    std::same_as<
        tensor::stride_type_t<1, typename InputSpec::InputLayout>,
        meta::Const<1>>;

template <>
struct Backend<
    gemm::details::sme::Format,
    matmul_pack_implementation::SMEPostprocess> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sme_common_eligible_v<InputSpec, OutputSpec> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement, float32_t> &&
      std::same_as<typename InputSpec::ComputeType, bfloat16_t>;

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    using T = typename gemm::packing_t<Atom, Side>::Element;
    static_assert(std::same_as<T, bfloat16_t>);
    const auto& layout = source.spec().input_layout();
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    const nint_t row_stride = source.raw_strides()[0];
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack_postprocess<Atom, Side>(
              source, spatial, k, row_stride, output);
        });
  }
};

template <>
struct Backend<
    gemm::details::sme::Format,
    matmul_pack_implementation::SMEStagedTransform> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = [] {
    using Transform = typename InputSpec::TransformType;
    if constexpr (std::same_as<Transform, tensor::NoTransform>) {
      return false;
    } else {
      return sme_common_eligible_v<InputSpec, OutputSpec> &&
          std::same_as<typename InputSpec::MemoryElement,
                       typename InputSpec::ComputeType> &&
          Transform::is_elementwise &&
          Transform::permutation_equivariant &&
          std::same_as<typename Transform::TIn,
                       typename InputSpec::ComputeType> &&
          std::same_as<typename Transform::TOut,
                       typename InputSpec::ComputeType>;
    }
  }();

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(!execution::has_resource_v<
        execution::details::arm::StreamingZA, Scope>);
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = gemm::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    using Transform = typename Source::Transform;
    static_assert(std::same_as<typename Source::MemoryElement, T>);
    static_assert(!std::same_as<Transform, tensor::NoTransform>);
    static_assert(Transform::is_elementwise);
    static_assert(std::same_as<typename Transform::TIn, T>);
    static_assert(std::same_as<typename Transform::TOut, T>);
    const auto& layout = source.spec().input_layout();
    const auto* input = reinterpret_cast<const T*>(source.raw_data());
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    const nint_t row_stride = source.raw_strides()[0];
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack<Atom, Side>(
              input, spatial, k, row_stride, output);
        });
    const nint_t panel = destination.spec().output_layout().shape()[2];
    sme::transform_packed_inplace(
        output, layout.shape()[0], layout.shape()[1],
        panel, Packing::KPack, source.spec().transform());
  }
};

template <>
struct Backend<
    gemm::details::sme::Format,
    matmul_pack_implementation::SMEStagedFP16ToFP32> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sme_common_eligible_v<InputSpec, OutputSpec> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement, float16_t> &&
      std::same_as<typename InputSpec::ComputeType, float32_t>;

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(!execution::has_resource_v<
        execution::details::arm::StreamingZA, Scope>);
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = gemm::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    static_assert(std::same_as<T, float32_t>);
    static_assert(std::same_as<typename Source::MemoryElement, float16_t>);
    static_assert(std::same_as<typename Source::Transform, tensor::NoTransform>);
    const auto& layout = source.spec().input_layout();
    const auto* input = source.raw_data();
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    const nint_t row_stride = source.raw_strides()[0];
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    auto* temporary = reinterpret_cast<float16_t*>(output);
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack<gemm::SME_F16F32, Side>(
              input, spatial, k, row_stride, temporary);
        });
    const nint_t panel = destination.spec().output_layout().shape()[2];
    sme::expand_fp16_packed_to_fp32(
        temporary, output, layout.shape()[0], layout.shape()[1], panel);
  }
};

} // namespace vecops::kernel::matmul_pack_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_SME_BACKEND_H
