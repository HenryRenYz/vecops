//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_SME_BACKEND_H
#define VECOPS_MATMUL_DETAILS_PACK_SME_BACKEND_H

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/details/sme/Atoms.h"
#include "vecops/matmul/details/sme/Packing.h"
#include "vecops/matmul/details/pack/generic/Pack.h"
#include "vecops/matmul/details/pack/sme/Pack.h"
#include "vecops/matmul/details/pack/sme/TransformPack.h"

namespace vecops::kernel::matmul_pack_details {

template <>
struct Backend<
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::Vector> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    using Tag = vec::ScalableTag<T, 0>;
    const auto& layout = source.spec().input_layout();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto panel = tensor::size_value<2>(
        destination.spec().output_layout());
    generic::pack_interleaved_panels<Packing::KPack, Tag>(
        source, reinterpret_cast<T*>(destination.raw_data()),
        static_cast<nint_t>(spatial), static_cast<nint_t>(k),
        static_cast<nint_t>(panel),
        ceil_div(static_cast<nint_t>(k), Packing::KPack));
  }
};

template <>
struct Backend<
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sizeof(typename InputSpec::ComputeType) <= 8 &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement,
                   typename InputSpec::ComputeType> &&
      std::same_as<typename OutputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename OutputSpec::MemoryElement,
                   typename OutputSpec::ComputeType> &&
      std::same_as<
          tensor::stride_type_t<1, typename InputSpec::InputLayout>,
          meta::Const<1>>;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    const auto& layout = source.spec().input_layout();
    static_assert(generic::RawDirectAccess<Source>);
    const auto* input = reinterpret_cast<const T*>(source.raw_data());
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
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
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::SMEPostprocess> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = [] {
    using Transform = typename InputSpec::TransformType;
    using Memory = typename InputSpec::MemoryElement;
    using Compute = typename InputSpec::ComputeType;
    if constexpr (!sme_common_eligible_v<InputSpec, OutputSpec>) {
      return false;
    } else if constexpr (std::same_as<Transform, tensor::NoTransform>) {
      return
          (std::same_as<Memory, float32_t> &&
           (std::same_as<Compute, bfloat16_t> ||
            std::same_as<Compute, float16_t>)) ||
          ((std::same_as<Memory, bfloat16_t> ||
            std::same_as<Memory, float16_t>) &&
           std::same_as<Compute, float32_t>);
    } else {
      return std::same_as<Memory, float32_t> &&
          (std::same_as<Compute, int8_t> ||
              std::same_as<Compute, uint8_t>) &&
          Transform::is_elementwise &&
          Transform::permutation_equivariant &&
          std::same_as<typename Transform::TIn, float32_t> &&
          std::same_as<typename Transform::TOut, float32_t>;
    }
  }();

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    using T = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
    static_assert(
        std::same_as<T, float32_t> || std::same_as<T, float16_t> ||
        std::same_as<T, bfloat16_t> || std::same_as<T, int8_t> ||
        std::same_as<T, uint8_t>);
    const auto& layout = source.spec().input_layout();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
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
    ::vecops::matmul::details::sme::Format,
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

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(!execution::has_resource_v<
        execution::details::arm::StreamingZA, Scope>);
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    using Transform = typename Source::Transform;
    static_assert(std::same_as<typename Source::MemoryElement, T>);
    static_assert(!std::same_as<Transform, tensor::NoTransform>);
    static_assert(Transform::is_elementwise);
    static_assert(std::same_as<typename Transform::TIn, T>);
    static_assert(std::same_as<typename Transform::TOut, T>);
    const auto& layout = source.spec().input_layout();
    const auto* input = reinterpret_cast<const T*>(source.raw_data());
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack<Atom, Side>(
              input, spatial, k, row_stride, output);
        });
    const auto panel = tensor::size_value<2>(
        destination.spec().output_layout());
    sme::transform_packed_inplace<Packing::KPack>(
        output, spatial, k, panel,
        source.spec().transform());
  }
};

template <>
struct Backend<
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::SMEStagedFP16ToFP32> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sme_common_eligible_v<InputSpec, OutputSpec> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement, float16_t> &&
      std::same_as<typename InputSpec::ComputeType, float32_t>;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(!execution::has_resource_v<
        execution::details::arm::StreamingZA, Scope>);
    static_assert(generic::RawDirectAccess<Destination>);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    static_assert(std::same_as<T, float32_t>);
    static_assert(std::same_as<typename Source::MemoryElement, float16_t>);
    static_assert(std::same_as<typename Source::Transform, tensor::NoTransform>);
    const auto& layout = source.spec().input_layout();
    const auto* input = source.raw_data();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    auto* temporary = reinterpret_cast<float16_t*>(output);
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack<::vecops::matmul::SME_F16F32, Side>(
              input, spatial, k, row_stride, temporary);
        });
    const auto panel = tensor::size_value<2>(
        destination.spec().output_layout());
    sme::expand_fp16_packed_to_fp32(
        temporary, output, spatial, k, panel);
  }
};

template <>
struct Backend<
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::SMEFP32ToFP64> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible =
      sme_common_eligible_v<InputSpec, OutputSpec> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement, float32_t> &&
      std::same_as<typename InputSpec::ComputeType, float64_t>;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    static_assert(std::same_as<typename Source::MemoryElement, float32_t>);
    static_assert(std::same_as<typename Source::ComputeType, float64_t>);
    static_assert(std::same_as<typename Source::Transform, tensor::NoTransform>);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    static_assert(std::same_as<typename Packing::Element, float64_t>);
    const auto& layout = source.spec().input_layout();
    const auto* input = source.raw_data();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
    auto* output = reinterpret_cast<float64_t*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack_fp32_to_fp64(
              input, spatial, k, row_stride, output);
        });
  }
};

template <>
struct Backend<
    ::vecops::matmul::details::sme::Format,
    matmul_pack_implementation::SMEFP32ToFP64Single> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = Backend<
      ::vecops::matmul::details::sme::Format,
      matmul_pack_implementation::SMEFP32ToFP64>::template eligible<
          InputSpec, OutputSpec>;

  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, const Source& source, Destination& destination) {
    static_assert(generic::RawDirectAccess<Destination>);
    static_assert(std::same_as<typename Source::MemoryElement, float32_t>);
    static_assert(std::same_as<typename Source::ComputeType, float64_t>);
    static_assert(std::same_as<typename Source::Transform, tensor::NoTransform>);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    static_assert(std::same_as<typename Packing::Element, float64_t>);
    const auto& layout = source.spec().input_layout();
    const auto* input = source.raw_data();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    const auto row_stride = tensor::stride_value<0>(layout);
    auto* output = reinterpret_cast<float64_t*>(destination.raw_data());
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          sme::pack_fp32_to_fp64_single(
              input, spatial, k, row_stride, output);
        });
  }
};

} // namespace vecops::kernel::matmul_pack_details

#endif // VECOPS_MATMUL_DETAILS_PACK_SME_BACKEND_H
