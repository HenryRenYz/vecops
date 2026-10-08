// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Exercise the maintained paired output transform through real DataAccess.
// Swapping an access must preserve the original coordinate/vector-axis map.
#define __kernel__ unused_column_scale_matmul_reference
#include "../../benchmarks/dsl_matmul_fusion_reference.cpp"
#undef __kernel__

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto C,
                vecops::tensor::TensorLike auto Scale, vecops::tensor::TensorLike auto Zero) {
  using namespace vecops;
  constexpr int axis = decltype(spec::Axis)::value;
  constexpr auto pipeline = static_cast<fusion_benchmark::Output>(decltype(spec::OutputPipeline)::value);
  const auto input = tensor::input<float32_t>(A);
  const auto output = fusion_benchmark::output<float32_t, pipeline>(C, Scale, Scale, Scale, Scale, Scale, Zero, 1);
  auto oriented_input = [&](const auto& value) VECOPS_INLINE_LAMBDA {
    if constexpr (spec::SwapCoordinates)
      return tensor::transpose_view<0, 1>(value);
    else
      return value;
  }(input);
  auto oriented_output = [&](const auto& value) VECOPS_INLINE_LAMBDA {
    if constexpr (spec::SwapCoordinates)
      return tensor::transpose_view<0, 1>(value);
    else
      return value;
  }(output);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto source =
    tensor::bind(oriented_input, tensor::InputAccessPolicy<axis, 1, tensor::AccessPlan::direct>{}, workspace);
  auto destination =
    tensor::bind(oriented_output, tensor::OutputAccessPolicy<axis, tensor::AccessPlan::direct>{}, workspace);
  using Tag = vec::ScalableTag<float32_t>;
  const Tag tag{};
  const auto shape = tensor::logical_layout(source).shape();
  const nint_t lanes = vec::size(tag);
  for (nint_t line = 0; line < static_cast<nint_t>(shape[1 - axis]); ++line) {
    for (nint_t begin = 0; begin < static_cast<nint_t>(shape[axis]); begin += lanes) {
      const auto coordinate = axis == 0 ? tensor::coord(begin, line) : tensor::coord(line, begin);
      const auto active = vec::opt::first(std::min(lanes, static_cast<nint_t>(shape[axis]) - begin));
      const auto value = source.load(tag, coordinate, tensor::axis<axis>, active);
      destination.store(tag, coordinate, tensor::axis<axis>, value, active);
    }
  }
}
