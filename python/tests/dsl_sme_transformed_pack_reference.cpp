// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Keep the public MatmulPack dispatch/TransformPack optimizations intact.
// The auxiliary tensors express the same FP32 pointwise graph as the DSL;
// the test oracle separately verifies logical coordinates and padded zeros.
#include "vecops/Kernel.h"
#include "vecops/ops/MatmulPack.h"

template <typename Native>
constexpr auto sme_transformed_pack_atom() {
  if constexpr (std::same_as<Native, vecops::bfloat16_t>)
    return std::type_identity<vecops::matmul::SME_BF16F32>{};
  else if constexpr (std::same_as<Native, vecops::float16_t>)
    return std::type_identity<vecops::matmul::SME_F16F32>{};
  else
    return std::type_identity<vecops::matmul::SME_I8I32<Native, Native>>{};
}

void __kernel__(vecops::tensor::TensorLike auto X,
                vecops::tensor::TensorLike auto Mean,
                vecops::tensor::TensorLike auto Rstd,
                vecops::tensor::TensorLike auto Weight,
                vecops::tensor::TensorLike auto Bias,
                vecops::tensor::TensorLike auto P) {
  using Native = vecops::spec::Native;
  using Atom = typename decltype(sme_transformed_pack_atom<Native>())::type;
  constexpr auto side = vecops::spec::SideB ? vecops::matmul::Operand::B
                                           : vecops::matmul::Operand::A;
  auto orient = [](const auto& input) VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::SideB)
      return vecops::tensor::transpose<0, 1>(input);
    else
      return input;
  };
  const auto source = orient(X);
  const auto mean = orient(Mean);
  const auto rstd = orient(Rstd);
  const auto weight = orient(Weight);
  const auto bias = orient(Bias);
  auto transform = vecops::tensor::make_lane_local_vec_transform<float, float>(
      [&](auto tag, auto value, const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const auto row = coordinate[0], column = coordinate[1];
        const auto active = context.active_option(tag);
        // TransformPack can select either logical vector axis. Keep the
        // vector-axis and scalar broadcast cases explicit and equally fast.
        auto rows = [&](const auto& tensor) VECOPS_KERNEL_LAMBDA {
          if (context.vector_axis == 0)
            return vecops::vec::load(tag, tensor.data() + row * tensor.stride(0),
                                    active, vecops::vec::opt::strided(tensor.stride(0)));
          return vecops::vec::fill(tag, tensor.data()[row * tensor.stride(0)]);
        };
        auto columns = [&](const auto& tensor) VECOPS_KERNEL_LAMBDA {
          if (context.vector_axis == 1)
            return vecops::vec::load(tag, tensor.data() + column * tensor.stride(1),
                                    active, vecops::vec::opt::strided(tensor.stride(1)));
          return vecops::vec::fill(tag, tensor.data()[column * tensor.stride(1)]);
        };
        if constexpr (vecops::spec::Pipeline == 1)
          return vecops::vec::add(tag, vecops::vec::mul(tag, value, vecops::vec::fill(tag, 0.5f)),
                                  vecops::vec::fill(tag, 7.0f));
        else if constexpr (vecops::spec::Pipeline >= 2) {
          const auto normalized = vecops::vec::mul(tag, vecops::vec::sub(tag, value, rows(mean)), rows(rstd));
          if constexpr (vecops::spec::Pipeline == 3)
            return vecops::vec::add(tag, vecops::vec::mul(tag, normalized, columns(weight)), columns(bias));
          else
            return vecops::vec::add(tag, normalized, vecops::vec::fill(tag, 7.0f));
        } else
          return value;
      });
  const auto input = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::Pipeline == 0)
      return vecops::tensor::input<Native>(source);
    else if constexpr (vecops::spec::Pipeline == 1) {
      const auto elementwise = vecops::tensor::make_elementwise_vec_transform<float, float>(
          [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
            return vecops::vec::add(tag, vecops::vec::mul(tag, value, vecops::vec::fill(tag, 0.5f)),
                                    vecops::vec::fill(tag, 7.0f));
          });
      return vecops::tensor::input<Native>(source, elementwise);
    } else
      return vecops::tensor::input<Native>(source, transform);
  }();
  const auto destination = vecops::tensor::make_tensor(
      P.data(), vecops::matmul::packed_layout<Atom, side>(source.layout()));
  const auto operation = vecops::ops::matmul_pack(vecops::ops::MatmulPackConfig<Atom, side>{});
  vecops::execution::ExecutionSession execution;
  operation(execution, input, vecops::tensor::output<Native>(destination));
}
