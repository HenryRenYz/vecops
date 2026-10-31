// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file _Pointwise.cpp
 * Static broadcast indexing and SIMD floating point operations for replay.
 * Each grid writes exactly one original operation's result, preserving BF16
 * rounding. Outer strides and singleton broadcasts are resolved in prepare.
 */
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
#include "vecops/vec/Vec.h"
#include <algorithm>
#include <cstring>
#include <type_traits>

using namespace vecops;
constexpr int Operation = spec::Operation;
constexpr nint_t Parallelism = spec::Parallelism;
using Tag = vec::ScalableTag<float32_t, 0>;

template <typename Input, typename Output>
auto outer_offset(Input input, Output output, nint_t row) {
  nint_t offset = 0;
  for (int axis = Output::Ndim - 2; axis >= 0; --axis) {
    const auto coordinate = row % output.size(axis);
    row /= output.size(axis);
    const int input_axis = axis - (Output::Ndim - Input::Ndim);
    if (input_axis >= 0 && input.size(input_axis) != 1)
      offset += coordinate * input.stride(input_axis);
  }
  return offset;
}

template <typename Input>
auto load(Input input, nint_t offset, nint_t column, auto active) {
  if constexpr (Input::Ndim == 0)
    return vec::fill(Tag{}, static_cast<float32_t>(*input.data()));
  else if constexpr (std::is_same_v<std::remove_cv_t<typename Input::ElementType>, bool>) {
    // Boolean masks broadcast one scalar across the SIMD feature dimension.
    // Do not instantiate an unsupported vector load of bool, or replace
    // multiplication by masking: 0*NaN and the sign of zero must survive.
    static_assert(decltype(input.template size<Input::Ndim - 1>())::value == 1);
    return vec::fill(Tag{}, static_cast<float32_t>(input.data()[offset]));
  }
  else {
    if (input.size(Input::Ndim - 1) == 1)
      return vec::fill(Tag{}, static_cast<float32_t>(input.data()[offset]));
    return vec::load_convert(Tag{}, input.data() + offset + column, active, vec::opt::zero);
  }
}

void __kernel__(execution::KernelBuilder& builder, auto a, auto b, auto output, auto alpha) {
  constexpr nint_t Block = 1024;
  nint_t elements = 1;
  for (int axis = 0; axis < decltype(output)::Ndim; ++axis)
    elements *= output.size(axis);
  static_assert(decltype(output)::Ndim > 0);
  const auto columns = output.size(decltype(output)::Ndim - 1);
  const nint_t rows = columns ? elements / columns : 0;
  const auto blocks = ceil_div(columns, cint<Block>);
  builder.grid<Parallelism>("pointwise", nint_t{0}, rows * blocks, cint<1>,
      execution::kernel_bind(a, b, output, alpha, columns, blocks),
      [](execution::TaskContext<Parallelism>, const auto& work, auto a, auto b, auto output,
         auto alpha, auto columns, auto blocks) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const nint_t row = work.begin() / blocks;
        const nint_t first = (work.begin() % blocks) * Block;
        const nint_t last = std::min(first + Block, columns);
        const auto ao = outer_offset(a, output, row);
        const auto bo = outer_offset(b, output, row);
        const auto destination_offset = [&] {
          if constexpr (decltype(spec::OutputContiguous)::value)
            return row * columns;
          else
            return outer_offset(output, output, row);
        }();
        if constexpr (Operation == 0 &&
                      std::is_same_v<std::remove_cv_t<typename decltype(a)::ElementType>,
                                     std::remove_cv_t<typename decltype(output)::ElementType>>) {
          // Readonly inputs have const ElementType. A same-dtype copy must
          // preserve raw bits, including signaling NaNs; widening/narrowing
          // through the float compute tag can quiet/canonicalize them.
          if (a.size(decltype(a)::Ndim - 1) == columns) {
            auto* destination = output.data() + destination_offset + first;
            const auto* source = a.data() + ao + first;
            if (destination != source)
              std::memcpy(destination, source, (last - first) * sizeof(typename decltype(output)::ElementType));
            return;
          }
          if (a.size(decltype(a)::Ndim - 1) == 1) {
            const auto* source = a.data() + ao;
            for (nint_t column = first; column < last; ++column)
              std::memcpy(output.data() + destination_offset + column, source,
                          sizeof(typename decltype(output)::ElementType));
            return;
          }
        }
        const auto lanes = vec::size(Tag{});
        for (nint_t column = first; column < last; column += lanes) {
          const auto active = vec::opt::first(std::min(lanes, last - column));
          auto x = load(a, ao, column, active);
          if constexpr (Operation == 5)
            x = vec::blend(Tag{}, x, vec::cmplt(Tag{}, x, vec::fill(Tag{}, 0.f)), vec::fill(Tag{}, 0.f));
          else if constexpr (Operation == 6) x = vec::neg(Tag{}, x);
          else if constexpr (Operation != 0) {
            auto y = [&] {
              if constexpr (Operation >= 10) return vec::fill(Tag{}, static_cast<float32_t>(alpha));
              else return load(b, bo, column, active);
            }();
            if constexpr (Operation == 1 || Operation == 2)
              y = vec::mul(Tag{}, y, vec::fill(Tag{}, static_cast<float32_t>(alpha)));
            if constexpr (Operation == 1 || Operation == 11) x = vec::add(Tag{}, x, y);
            else if constexpr (Operation == 2 || Operation == 12) x = vec::sub(Tag{}, x, y);
            else if constexpr (Operation == 3 || Operation == 13) x = vec::mul(Tag{}, x, y);
            else if constexpr (Operation == 4 || Operation == 14) x = vec::div(Tag{}, x, y);
          }
          vec::store_convert(Tag{}, output.data() + destination_offset + column, x, active);
        }
      });
}
