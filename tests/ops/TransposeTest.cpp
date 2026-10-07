// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "vecops/execution/WorkspaceContext.h"
#include "vecops/ops/Transpose.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

using AuxiliarySourceLayout = Layout<
    Shape<Const<2>, Const<3>, Const<5>>,
    Strides<Const<15>, Const<5>, Const<1>>>;
using AxisOneAuxiliaryLayout = decltype(
    vecops::tensor::details::auxiliary_layout<1>(
        std::declval<const AuxiliarySourceLayout&>()));
static_assert(std::same_as<
    typename AxisOneAuxiliaryLayout::Shape,
    typename AuxiliarySourceLayout::Shape>);
static_assert(std::same_as<
    typename AxisOneAuxiliaryLayout::Strides,
    Strides<Const<15>, Const<1>, Const<3>>>);

template <typename T>
T test_value(nint_t row, nint_t col) {
  return static_cast<T>(row * 37 + col * 3 - 11);
}

template <typename T, typename M, typename N,
          typename Policy = kernel::transpose2d_policy::Automatic>
void run_transpose(M m, N n, Policy policy = {}) {
  const nint_t mi = static_cast<nint_t>(m);
  const nint_t ni = static_cast<nint_t>(n);
  std::vector<T> input(static_cast<std::size_t>(mi * ni));
  std::vector<T> output(static_cast<std::size_t>(mi * ni), T{});
  for (nint_t i = 0; i < mi; ++i) {
    for (nint_t j = 0; j < ni; ++j) {
      input[static_cast<std::size_t>(i * ni + j)] = test_value<T>(i, j);
    }
  }
  auto input_layout = make_layout(make_shape(m, n));
  auto output_layout = make_layout(make_shape(n, m));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto output_tensor = make_tensor(output.data(), output_layout);
  // A direct transpose needs no workspace. Keeping this session empty makes
  // accidental materialization or a runtime fallback fail immediately.
  ExecutionSession execution{};
  auto operation = ops::transpose(ops::TransposeConfig<T, Policy>{policy});
  operation(execution, input_tensor, output_tensor);
  for (nint_t i = 0; i < mi; ++i) {
    for (nint_t j = 0; j < ni; ++j) {
      EXPECT_EQ(
          output[static_cast<std::size_t>(j * mi + i)],
          test_value<T>(i, j))
          << "i=" << i << " j=" << j;
    }
  }
}

TEST(TransposeTest, ConstFullTilesAndTailShapes) {
  run_transpose<int32_t>(cint<16>, cint<16>);
  run_transpose<int32_t>(cint<17>, cint<19>);
  run_transpose<int64_t>(cint<9>, cint<7>);
}

TEST(TransposeTest, DynamicShapes) {
  run_transpose<float32_t>(Any{23}, Any{14});
  run_transpose<int16_t>(Dynamic<8>{24}, Dynamic<4>{20});
  run_transpose<int32_t>(Dynamic<16>{32}, Dynamic<16>{48});
}

TEST(TransposeTest, ExhaustiveSmallDynamicShapesAndElementWidths) {
  auto run_grid = []<typename T>() {
    for (nint_t m = 0; m <= 9; ++m) {
      for (nint_t n = 0; n <= 9; ++n) {
        run_transpose<T>(Any{m}, Any{n});
      }
    }
  };
  run_grid.template operator()<int8_t>();
  run_grid.template operator()<uint16_t>();
  run_grid.template operator()<int32_t>();
  run_grid.template operator()<uint64_t>();
  run_grid.template operator()<float32_t>();
  run_grid.template operator()<float64_t>();
}

TEST(TransposeTest, ForcedGatherPolicyHandlesFullAndTailVectors) {
  run_transpose<int32_t>(
      Any{19}, Any{23}, kernel::transpose2d_policy::Gather{});
  run_transpose<int8_t>(
      Any{7}, Any{41}, kernel::transpose2d_policy::Gather{});
}

TEST(TransposeTest, ZeroExtent) {
  run_transpose<int32_t>(cint<0>, cint<7>);
  run_transpose<int32_t>(cint<5>, cint<0>);
}

TEST(TransposeTest, ConvertsAtDataAccessBoundary) {
  constexpr nint_t M = 7;
  constexpr nint_t N = 11;
  std::array<int16_t, M * N> input{};
  std::array<float32_t, M * N> output{};
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      input[static_cast<std::size_t>(i * N + j)] =
          static_cast<int16_t>(i * 19 + j - 20);
    }
  }
  auto input_tensor = make_tensor(
      input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto output_tensor = make_tensor(
      output.data(), make_layout(make_shape(cint<N>, cint<M>)));
  auto operation = ops::transpose(ops::TransposeConfig<float32_t>{});
  ExecutionSession execution{};
  operation(execution, input_tensor, output_tensor);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                static_cast<float32_t>(input[static_cast<std::size_t>(i * N + j)]));
    }
  }
}

TEST(TransposeTest, ConfigOnlyOperatorCanBindDifferentOperands) {
  std::array<int32_t, 6> first{1, 2, 3, 4, 5, 6};
  std::array<int32_t, 6> second{7, 8, 9, 10, 11, 12};
  std::array<int32_t, 6> first_output{};
  std::array<int32_t, 6> second_output{};
  auto layout = make_layout(make_shape(cint<2>, cint<3>));
  auto transposed = make_layout(make_shape(cint<3>, cint<2>));
  auto operation = ops::transpose(ops::TransposeConfig<int32_t>{});
  ExecutionSession execution{};
  operation(execution, make_tensor(first.data(), layout),
            make_tensor(first_output.data(), transposed));
  operation(execution, make_tensor(second.data(), layout),
            make_tensor(second_output.data(), transposed));
  EXPECT_EQ(first_output, (std::array<int32_t, 6>{1, 4, 2, 5, 3, 6}));
  EXPECT_EQ(second_output, (std::array<int32_t, 6>{7, 10, 8, 11, 9, 12}));
}

TEST(TransposeTest, PatternPreparedBindsRuntimeStorage) {
  constexpr nint_t M = 5;
  constexpr nint_t N = 7;
  std::array<int32_t, M * N> first{};
  std::array<int32_t, M * N> second{};
  std::array<int32_t, M * N> first_output{};
  std::array<int32_t, M * N> second_output{};
  for (nint_t i = 0; i < M * N; ++i) {
    first[static_cast<std::size_t>(i)] = static_cast<int32_t>(i + 1);
    second[static_cast<std::size_t>(i)] = static_cast<int32_t>(101 + i);
  }
  const auto input_layout = make_layout(make_shape(cint<M>, cint<N>));
  const auto output_layout = make_layout(make_shape(cint<N>, cint<M>));
  auto first_input = make_tensor(first.data(), input_layout);
  auto first_out = make_tensor(first_output.data(), output_layout);
  auto second_input = make_tensor(second.data(), input_layout);
  auto second_out = make_tensor(second_output.data(), output_layout);

  execution::WorkspaceContext workspace{"prepared_transpose"};
  auto prepared = ops::transpose(ops::TransposeConfig<int32_t>{})
                      .template prepare<1>(
                          workspace, "scratch", unbind(first_input),
                          unbind(first_out));
  prepared(execution::TaskContext<1>{0}, first_input, first_out);
  prepared(execution::TaskContext<1>{0}, second_input, second_out);

  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(first_output[static_cast<std::size_t>(j * M + i)],
                first[static_cast<std::size_t>(i * N + j)]);
      EXPECT_EQ(second_output[static_cast<std::size_t>(j * M + i)],
                second[static_cast<std::size_t>(i * N + j)]);
    }
  }
}

extern "C" VECOPS_NOINLINE void transpose_i32_const_probe(
    const int32_t* input, int32_t* output) {
  auto input_tensor = make_tensor(
      input, make_layout(make_shape(cint<16>, cint<16>)));
  auto output_tensor = make_tensor(
      output, make_layout(make_shape(cint<16>, cint<16>)));
  ExecutionSession execution{};
  ops::transpose(ops::TransposeConfig<int32_t>{})(
      execution, input_tensor, output_tensor);
}

extern "C" VECOPS_NOINLINE void transpose_i32_aligned_probe(
    const int32_t* input, int32_t* output, nint_t m, nint_t n) {
  auto input_tensor = make_tensor(
      input, make_layout(make_shape(Dynamic<16>{m}, Dynamic<16>{n})));
  auto output_tensor = make_tensor(
      output, make_layout(make_shape(Dynamic<16>{n}, Dynamic<16>{m})));
  ExecutionSession execution{};
  ops::transpose(ops::TransposeConfig<int32_t>{})(
      execution, input_tensor, output_tensor);
}

#if defined(HAS_SME_FA64)

TEST(TransposeTest, ConfigOnlyOperatorUsesSMEPath) {
  constexpr nint_t M = 19;
  constexpr nint_t N = 23;
  std::array<int32_t, M * N> input{};
  std::array<int32_t, M * N> output{};
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      input[static_cast<std::size_t>(i * N + j)] = test_value<int32_t>(i, j);
    }
  }
  auto input_tensor = make_tensor(
      input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto output_tensor = make_tensor(
      output.data(), make_layout(make_shape(cint<N>, cint<M>)));
  auto operation = ops::transpose(ops::TransposeConfig<int32_t>{});
  ExecutionSession execution{};
  operation(execution, input_tensor, output_tensor);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                test_value<int32_t>(i, j));
    }
  }
}

TEST(TransposeTest, ConfigOnlyFloatOperatorUsesSMEPath) {
  constexpr nint_t M = 17;
  constexpr nint_t N = 18;
  std::array<float32_t, M * N> input{};
  std::array<float32_t, M * N> output{};
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      input[static_cast<std::size_t>(i * N + j)] =
          test_value<float32_t>(i, j);
    }
  }
  auto input_tensor = make_tensor(
      input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto output_tensor = make_tensor(
      output.data(), make_layout(make_shape(cint<N>, cint<M>)));
  auto operation = ops::transpose(ops::TransposeConfig<float32_t>{});
  ExecutionSession execution{};
  operation(execution, input_tensor, output_tensor);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                test_value<float32_t>(i, j));
    }
  }
}

TEST(TransposeTest, ConfigOnlyConversionOperatorUsesSMEPath) {
  constexpr nint_t M = 63;
  constexpr nint_t N = 129;
  std::array<int8_t, M * N> input{};
  std::array<float32_t, M * N> output{};
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      input[static_cast<std::size_t>(i * N + j)] =
          static_cast<int8_t>((i * 19 + j) % 127 - 63);
    }
  }
  auto input_tensor = make_tensor(
      input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto output_tensor = make_tensor(
      output.data(), make_layout(make_shape(cint<N>, cint<M>)));
  auto operation = ops::transpose(ops::TransposeConfig<float32_t>{});
  ExecutionSession execution{};
  operation(execution, input_tensor, output_tensor);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(
          output[static_cast<std::size_t>(j * M + i)],
          static_cast<float32_t>(
              input[static_cast<std::size_t>(i * N + j)]));
    }
  }
}

extern "C" VECOPS_NOINLINE void transpose_i32_streaming_probe(
    const int32_t* input, int32_t* output) {
  auto input_tensor = make_tensor(
      input, make_layout(make_shape(cint<16>, cint<16>)));
  auto output_tensor = make_tensor(
      output, make_layout(make_shape(cint<16>, cint<16>)));
  auto operation = ops::transpose(ops::TransposeConfig<int32_t>{});
  ExecutionSession execution{};
  operation(execution, input_tensor, output_tensor);
}

#endif

} // namespace
