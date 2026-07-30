#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/gemm/DataAccess.h"
#include "vecops/ops/Softmax.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::ops;

namespace {

template <typename... Ts>
struct TypeList {};

using SoftmaxFloatTypes = TypeList<
    vecops::float16_t,
    vecops::float32_t,
    vecops::float64_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, vecops::float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, vecops::float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, vecops::float64_t>) return "fp64";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return "bf16";
  return typeid(T).name();
}

template <SoftmaxExpMode Mode, typename T>
double tolerance() {
  if constexpr (Mode == SoftmaxExpMode::Estimate) {
    if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 2e-2;
    if constexpr (std::is_same_v<T, vecops::float16_t>) return 1e-2;
    return 8e-3;
  }
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 8e-3;
  if constexpr (std::is_same_v<T, vecops::float16_t>) return 2e-3;
  if constexpr (std::is_same_v<T, vecops::float64_t>) return 2e-12;
  return 3e-5;
}

void reference_softmax_rows(
    const std::vector<double>& x,
    nint_t rows,
    nint_t n,
    std::vector<double>& ref) {
  for (nint_t row = 0; row < rows; ++row) {
    const nint_t base = row * n;
    double max_value = -std::numeric_limits<double>::infinity();
    for (nint_t col = 0; col < n; ++col) {
      max_value = std::max(max_value, x[static_cast<size_t>(base + col)]);
    }
    double sum = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      const double value = std::exp(x[static_cast<size_t>(base + col)] - max_value);
      ref[static_cast<size_t>(base + col)] = value;
      sum += value;
    }
    for (nint_t col = 0; col < n; ++col) {
      ref[static_cast<size_t>(base + col)] /= sum;
    }
  }
}

template <SoftmaxExpMode Mode, typename TX, typename TY>
void run_dtype_combo() {
  constexpr nint_t n = 9;
  using ComputeT = std::conditional_t<
      std::is_same_v<TX, vecops::float64_t> || std::is_same_v<TY, vecops::float64_t>,
      vecops::float64_t,
      vecops::float32_t>;
  using Config = SoftmaxConfig<ComputeT, vec::ScalableTag<ComputeT, 0>, Mode>;

  std::vector<TX> x(static_cast<size_t>(n));
  std::vector<TY> out(static_cast<size_t>(n), TY{});
  std::vector<double> x_ref(static_cast<size_t>(n));
  std::vector<double> ref(static_cast<size_t>(n));
  for (nint_t i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = static_cast<TX>(-3.0 + 0.7 * double(i) + 0.1 * double(i % 3));
    x_ref[static_cast<size_t>(i)] = static_cast<double>(x[static_cast<size_t>(i)]);
  }
  reference_softmax_rows(x_ref, 1, n, ref);

  auto x_t = make_tensor(x.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto y_t = make_tensor(out.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto op = softmax(Config{});
  auto x_spec = input<ComputeT>(x_t);
  auto y_spec = output<ComputeT>(y_t);
  const nint_t workspace_bytes = op.required_workspace(x_spec, y_spec);
  EXPECT_EQ(
      workspace_bytes,
      gemm::details::workspace_round_up(
          n * static_cast<nint_t>(sizeof(ComputeT)),
          vec::DEFAULT_ALIGNMENT));
  Workspace workspace(workspace_bytes);
  auto view = workspace.view();
  op(view, x_spec, y_spec);

  const double tol = tolerance<Mode, TY>();
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(
        ref[static_cast<size_t>(i)],
        static_cast<double>(out[static_cast<size_t>(i)]),
        tol)
        << "mode=" << static_cast<int>(Mode)
        << " input=" << dtype_name<TX>()
        << " output=" << dtype_name<TY>()
        << " i=" << i;
  }
}

template <SoftmaxExpMode Mode, typename TX, typename... TYs>
void run_out_combos(TypeList<TYs...>) {
  (run_dtype_combo<Mode, TX, TYs>(), ...);
}

template <SoftmaxExpMode Mode, typename... TXs>
void run_all_dtype_combos(TypeList<TXs...>) {
  (run_out_combos<Mode, TXs>(SoftmaxFloatTypes{}), ...);
}

template <int Rank>
void run_contiguous_rank_case() {
  constexpr nint_t n = 7;
  std::array<nint_t, Rank> shape{};
  if constexpr (Rank == 1) shape = {n};
  if constexpr (Rank == 2) shape = {3, n};
  if constexpr (Rank == 3) shape = {2, 3, n};
  if constexpr (Rank == 4) shape = {2, 2, 3, n};

  nint_t rows = 1;
  for (int d = 0; d < Rank - 1; ++d) rows *= shape[static_cast<size_t>(d)];
  const nint_t total = rows * n;
  std::vector<float> x(static_cast<size_t>(total));
  std::vector<float> out(static_cast<size_t>(total), -1.0f);
  std::vector<double> x_ref(static_cast<size_t>(total));
  std::vector<double> ref(static_cast<size_t>(total));
  for (nint_t i = 0; i < total; ++i) {
    x[static_cast<size_t>(i)] = -4.0f + float((i * 5) % 13) * 0.3f;
    x_ref[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
  }
  reference_softmax_rows(x_ref, rows, n, ref);

  auto op = softmax();
  if constexpr (Rank == 1) {
    auto x_t = make_tensor<1>(x.data(), {shape[0]});
    auto y_t = make_tensor<1>(out.data(), {shape[0]});
    op(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  } else if constexpr (Rank == 2) {
    auto x_t = make_tensor<2>(x.data(), {shape[0], shape[1]});
    auto y_t = make_tensor<2>(out.data(), {shape[0], shape[1]});
    op(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  } else if constexpr (Rank == 3) {
    auto x_t = make_tensor<3>(x.data(), {shape[0], shape[1], shape[2]});
    auto y_t = make_tensor<3>(out.data(), {shape[0], shape[1], shape[2]});
    op(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  } else {
    auto x_t = make_tensor<4>(x.data(), {shape[0], shape[1], shape[2], shape[3]});
    auto y_t = make_tensor<4>(out.data(), {shape[0], shape[1], shape[2], shape[3]});
    op(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  }

  for (nint_t i = 0; i < total; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 3e-5)
        << "rank=" << Rank << " i=" << i;
  }
}

struct DoubleTransform : VecTransform<vecops::float32_t, vecops::float32_t> {
  template <vec::VectorTag To, typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::mul(tag, v, vec::fill(tag, 2.0f));
  }
};

struct HalfTransform : VecTransform<vecops::float32_t, vecops::float32_t> {
  template <vec::VectorTag To, typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::mul(tag, v, vec::fill(tag, 0.5f));
  }
};

template <SoftmaxExpMode Mode>
void run_shift_stability_case() {
  using Config = SoftmaxConfig<
      vecops::float32_t,
      vec::ScalableTag<vecops::float32_t, 0>,
      Mode>;
  constexpr nint_t n = 11;
  std::vector<float> x(n);
  std::vector<float> shifted(n);
  std::vector<float> out(n);
  std::vector<float> shifted_out(n);
  for (nint_t i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = -8.0f + float(i) * 0.9f;
    shifted[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] + 10000.0f;
  }
  auto x_t = make_tensor<1>(x.data(), {n});
  auto shifted_t = make_tensor<1>(shifted.data(), {n});
  auto y_t = make_tensor<1>(out.data(), {n});
  auto shifted_y_t = make_tensor<1>(shifted_out.data(), {n});
  auto op = softmax(Config{});
  op(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  op(input<vecops::float32_t>(shifted_t), output<vecops::float32_t>(shifted_y_t));

  double sum = 0.0;
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(out[static_cast<size_t>(i)], shifted_out[static_cast<size_t>(i)], 8e-3);
    sum += shifted_out[static_cast<size_t>(i)];
  }
  EXPECT_NEAR(sum, 1.0, 8e-3);
}

} // namespace

TEST(SoftmaxDTypeTest, CoversAllFloatInputOutputCombinationsAndModes) {
  run_all_dtype_combos<SoftmaxExpMode::Strict>(SoftmaxFloatTypes{});
  run_all_dtype_combos<SoftmaxExpMode::Fast>(SoftmaxFloatTypes{});
  run_all_dtype_combos<SoftmaxExpMode::Estimate>(SoftmaxFloatTypes{});
}

TEST(SoftmaxRankTest, CoversRanksOneThroughFour) {
  run_contiguous_rank_case<1>();
  run_contiguous_rank_case<2>();
  run_contiguous_rank_case<3>();
  run_contiguous_rank_case<4>();
}

TEST(SoftmaxNumericsTest, HandlesAllNegativeTailWithoutInactiveLaneContamination) {
  constexpr nint_t n = 7;
  std::vector<float> x{-9.0f, -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f};
  std::vector<float> out(n);
  std::vector<double> x_ref(x.begin(), x.end());
  std::vector<double> ref(n);
  reference_softmax_rows(x_ref, 1, n, ref);
  auto x_t = make_tensor<1>(x.data(), {n});
  auto y_t = make_tensor<1>(out.data(), {n});
  softmax()(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 3e-5);
  }
}

TEST(SoftmaxNumericsTest, IsStableUnderLargeConstantShiftForEveryMode) {
  run_shift_stability_case<SoftmaxExpMode::Strict>();
  run_shift_stability_case<SoftmaxExpMode::Fast>();
  run_shift_stability_case<SoftmaxExpMode::Estimate>();
}

TEST(SoftmaxNumericsTest, HandlesNegativeInfinity) {
  std::vector<float> x{0.0f, -std::numeric_limits<float>::infinity(), -1.0f};
  std::vector<float> out(3);
  auto x_t = make_tensor<1>(x.data(), {3});
  auto y_t = make_tensor<1>(out.data(), {3});
  softmax()(input<vecops::float32_t>(x_t), output<vecops::float32_t>(y_t));
  EXPECT_FLOAT_EQ(out[1], 0.0f);
  EXPECT_NEAR(out[0] + out[2], 1.0f, 3e-5f);
}

TEST(SoftmaxWorkspaceTest, LayoutAndSpecWorkspaceMatchForStridedRows) {
  using InLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<17>, Const<2>>>;
  using OutLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<1>, Const<2>>>;
  InLayout in_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<17>, Const<2>>{}};
  OutLayout out_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<1>, Const<2>>{}};
  std::vector<float> x(30, 0.0f);
  std::vector<float> out(14, -1.0f);
  std::vector<double> dense_x(14);
  std::vector<double> ref(14);
  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      const float value = -3.0f + float(row * 7 + col) * 0.2f;
      x[static_cast<size_t>(row * 17 + col * 2)] = value;
      dense_x[static_cast<size_t>(row * 7 + col)] = value;
    }
  }
  reference_softmax_rows(dense_x, 2, 7, ref);

  auto x_t = make_tensor(x.data(), in_layout);
  auto y_t = make_tensor(out.data(), out_layout);
  auto x_spec = input<vecops::float32_t>(x_t);
  auto y_spec = output<vecops::float32_t>(y_t);
  auto op = softmax();
  const nint_t spec_bytes = op.required_workspace(x_spec, y_spec);
  const nint_t layout_bytes = op.required_workspace(in_layout, out_layout);
  EXPECT_EQ(spec_bytes, layout_bytes);
  EXPECT_GT(spec_bytes, 0);
  Workspace workspace(spec_bytes);
  auto view = workspace.view();
  op(view, x_spec, y_spec);

  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      EXPECT_NEAR(
          ref[static_cast<size_t>(row * 7 + col)],
          out[static_cast<size_t>(row + col * 2)],
          3e-5);
    }
  }
}

TEST(SoftmaxFusionTest, AppliesInputPrologueAndOutputEpilogue) {
  constexpr nint_t n = 7;
  std::vector<float> x{-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  std::vector<float> out(n);
  std::vector<double> transformed(n);
  std::vector<double> ref(n);
  for (nint_t i = 0; i < n; ++i) transformed[static_cast<size_t>(i)] = 2.0 * x[static_cast<size_t>(i)];
  reference_softmax_rows(transformed, 1, n, ref);
  auto x_t = make_tensor<1>(x.data(), {n});
  auto y_t = make_tensor<1>(out.data(), {n});
  softmax()(
      input<vecops::float32_t>(x_t, DoubleTransform{}),
      output<vecops::float32_t>(y_t, HalfTransform{}));
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)] * 0.5, out[static_cast<size_t>(i)], 3e-5);
  }
}

TEST(SoftmaxAliasingTest, SupportsContiguousInPlaceExecution) {
  constexpr nint_t n = 9;
  std::vector<float> values{-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<double> x_ref(values.begin(), values.end());
  std::vector<double> ref(n);
  reference_softmax_rows(x_ref, 1, n, ref);
  auto tensor = make_tensor<1>(values.data(), {n});
  softmax()(input<vecops::float32_t>(tensor), output<vecops::float32_t>(tensor));
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], values[static_cast<size_t>(i)], 3e-5);
  }
}
