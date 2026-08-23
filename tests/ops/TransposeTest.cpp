#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

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
  ops::transpose<T>(execution, input_tensor, output_tensor, policy);
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
  auto operation = ops::make_transpose<float32_t>(
      input_tensor, output_tensor);
  static_assert(std::same_as<
      typename decltype(operation)::ResourceRequirements,
      typename execution::details::CurrentBackend::DefaultRequirements>);
  ExecutionSession execution{};
  execution.with_region(
      operation, [&](auto& region) VECOPS_INLINE_LAMBDA {
        operation(region);
      });
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                static_cast<float32_t>(input[static_cast<std::size_t>(i * N + j)]));
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
  ops::transpose<int32_t>(execution, input_tensor, output_tensor);
}

extern "C" VECOPS_NOINLINE void transpose_i32_aligned_probe(
    const int32_t* input, int32_t* output, nint_t m, nint_t n) {
  auto input_tensor = make_tensor(
      input, make_layout(make_shape(Dynamic<16>{m}, Dynamic<16>{n})));
  auto output_tensor = make_tensor(
      output, make_layout(make_shape(Dynamic<16>{n}, Dynamic<16>{m})));
  ExecutionSession execution{};
  ops::transpose<int32_t>(execution, input_tensor, output_tensor);
}

#if defined(HAS_ARM_LOCALLY_STREAMING)

TEST(TransposeTest, PreparedOperatorDeclaresAndUsesSMEPath) {
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
  auto operation = ops::make_transpose<int32_t>(input_tensor, output_tensor);
  static_assert(execution::details::has_resource_v<
      execution::details::arm::Streaming,
      typename decltype(operation)::ResourceRequirements>);
  ExecutionSession execution{};
  execution.with_region(
      operation, [&](auto& region) VECOPS_INLINE_LAMBDA {
        operation(region);
      });
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                test_value<int32_t>(i, j));
    }
  }
}

TEST(TransposeTest, PreparedFloatOperatorUsesSMEPath) {
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
  auto operation = ops::make_transpose<float32_t>(
      input_tensor, output_tensor);
  ExecutionSession execution{};
  execution.with_region(
      operation, [&](auto& region) VECOPS_INLINE_LAMBDA {
        operation(region);
      });
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      EXPECT_EQ(output[static_cast<std::size_t>(j * M + i)],
                test_value<float32_t>(i, j));
    }
  }
}

extern "C" VECOPS_NOINLINE void transpose_i32_streaming_probe(
    const int32_t* input, int32_t* output) {
  auto input_tensor = make_tensor(
      input, make_layout(make_shape(cint<16>, cint<16>)));
  auto output_tensor = make_tensor(
      output, make_layout(make_shape(cint<16>, cint<16>)));
  auto operation = ops::make_transpose<int32_t>(input_tensor, output_tensor);
  ExecutionSession execution{};
  execution.with_region(
      operation, [&](auto& region) VECOPS_INLINE_LAMBDA {
        operation(region);
      });
}

#endif

} // namespace
