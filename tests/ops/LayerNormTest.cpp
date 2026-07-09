#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/ops/LayerNorm.h"
#include "vecops/gemm/DataAccess.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::ops;

namespace {

template <typename... Ts>
struct TypeList {};

using LayerNormFloatTypes = TypeList<
    float16_t,
    float32_t,
    float64_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , bfloat16_t
#endif
>;

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, float64_t>) return "fp64";
  if constexpr (std::is_same_v<T, bfloat16_t>) return "bf16";
  return typeid(T).name();
}

template <typename T>
double as_double(T value) {
  return static_cast<double>(value);
}

template <typename T>
double tolerance() {
  if constexpr (std::is_same_v<T, float64_t>) return 2e-5;
  if constexpr (std::is_same_v<T, float32_t>) return 2e-5;
  if constexpr (std::is_same_v<T, float16_t>) return 2e-2;
  if constexpr (std::is_same_v<T, bfloat16_t>) return 4e-2;
  return 1e-5;
}

template <typename TX, typename TS, typename TB, typename TY>
void run_dtype_combo() {
  constexpr nint_t n = 9;
  std::vector<TX> x(static_cast<size_t>(n));
  std::vector<TS> scale(static_cast<size_t>(n));
  std::vector<TB> bias(static_cast<size_t>(n));
  std::vector<TY> out(static_cast<size_t>(n), TY{});

  for (nint_t i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = static_cast<TX>(float((i % 5) - 2) * 0.75f + float(i) * 0.05f);
    scale[static_cast<size_t>(i)] = static_cast<TS>(0.8f + 0.1f * float(i % 4));
    bias[static_cast<size_t>(i)] = static_cast<TB>(-0.3f + 0.07f * float(i % 5));
  }

  auto x_t = make_tensor(x.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto s_t = make_tensor(scale.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto b_t = make_tensor(bias.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto y_t = make_tensor(out.data(), make_shape(Any{n}), make_strides(cint<1>));

  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  auto x_spec = input<float32_t>(x_t);
  auto s_spec = input<float32_t>(s_t);
  auto b_spec = input<float32_t>(b_t);
  auto y_spec = output<float32_t>(y_t);
  Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);

  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (nint_t i = 0; i < n; ++i) {
    const float v = static_cast<float>(x[static_cast<size_t>(i)]);
    sum += v;
    sum_sq += v * v;
  }
  const float mean = sum / float(n);
  const float var = std::max(sum_sq / float(n) - mean * mean, 0.0f);
  const float rstd = 1.0f / std::sqrt(var + 1e-5f);
  for (nint_t i = 0; i < n; ++i) {
    const float xv = static_cast<float>(x[static_cast<size_t>(i)]);
    const float sv = static_cast<float>(scale[static_cast<size_t>(i)]);
    const float bv = static_cast<float>(bias[static_cast<size_t>(i)]);
    const TY expected = static_cast<TY>((xv - mean) * rstd * sv + bv);
    EXPECT_NEAR(
        as_double(expected),
        as_double(out[static_cast<size_t>(i)]),
        tolerance<TY>())
        << "combo x=" << dtype_name<TX>()
        << " scale=" << dtype_name<TS>()
        << " bias=" << dtype_name<TB>()
        << " out=" << dtype_name<TY>()
        << " i=" << i;
  }
}

template <typename TX, typename TS, typename TB, typename... TYs>
void run_out_combos(TypeList<TYs...>) {
  (run_dtype_combo<TX, TS, TB, TYs>(), ...);
}

template <typename TX, typename TS, typename... TBs>
void run_bias_combos(TypeList<TBs...>) {
  (run_out_combos<TX, TS, TBs>(LayerNormFloatTypes{}), ...);
}

template <typename TX, typename... TSs>
void run_scale_combos(TypeList<TSs...>) {
  (run_bias_combos<TX, TSs>(LayerNormFloatTypes{}), ...);
}

template <typename... TXs>
void run_all_dtype_combos(TypeList<TXs...>) {
  (run_scale_combos<TXs>(LayerNormFloatTypes{}), ...);
}

void reference_layernorm_rows(
    const std::vector<float>& x,
    const std::vector<float>& scale,
    const std::vector<float>& bias,
    nint_t rows,
    nint_t n,
    std::vector<float>& ref,
    float eps,
    float post_scale = 1.0f) {
  for (nint_t row = 0; row < rows; ++row) {
    const nint_t base = row * n;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (nint_t col = 0; col < n; ++col) {
      const float v = x[static_cast<size_t>(base + col)];
      sum += v;
      sum_sq += v * v;
    }
    const float mean = sum / float(n);
    const float var = std::max(sum_sq / float(n) - mean * mean, 0.0f);
    const float rstd = 1.0f / std::sqrt(var + eps);
    for (nint_t col = 0; col < n; ++col) {
      ref[static_cast<size_t>(base + col)] =
          ((x[static_cast<size_t>(base + col)] - mean) * rstd *
           scale[static_cast<size_t>(col)] +
           bias[static_cast<size_t>(col)]) *
          post_scale;
    }
  }
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
  for (int i = 0; i < Rank - 1; ++i) rows *= shape[static_cast<size_t>(i)];
  const nint_t total = rows * n;

  std::vector<float> x(static_cast<size_t>(total));
  std::vector<float> scale(static_cast<size_t>(n));
  std::vector<float> bias(static_cast<size_t>(n));
  std::vector<float> out(static_cast<size_t>(total), -99.0f);
  std::vector<float> ref(static_cast<size_t>(total));

  for (nint_t i = 0; i < total; ++i) x[static_cast<size_t>(i)] = float((i % 11) - 5) * 0.2f;
  for (nint_t i = 0; i < n; ++i) {
    scale[static_cast<size_t>(i)] = 0.7f + 0.05f * float(i);
    bias[static_cast<size_t>(i)] = -0.2f + 0.03f * float(i);
  }
  reference_layernorm_rows(x, scale, bias, rows, n, ref, 1e-5f);

  auto s_t = make_tensor(scale.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto b_t = make_tensor(bias.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});

  if constexpr (Rank == 1) {
    auto x_t = make_tensor(x.data(), make_shape(cint<7>), make_strides(cint<1>));
    auto y_t = make_tensor(out.data(), make_shape(cint<7>), make_strides(cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t), input<float32_t>(b_t), output<float32_t>(y_t));
  } else if constexpr (Rank == 2) {
    auto x_t = make_tensor(
        x.data(), make_shape(cint<3>, cint<7>), make_strides(cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(), make_shape(cint<3>, cint<7>), make_strides(cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t), input<float32_t>(b_t), output<float32_t>(y_t));
  } else if constexpr (Rank == 3) {
    auto x_t = make_tensor(
        x.data(),
        make_shape(cint<2>, cint<3>, cint<7>),
        make_strides(cint<21>, cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(),
        make_shape(cint<2>, cint<3>, cint<7>),
        make_strides(cint<21>, cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t), input<float32_t>(b_t), output<float32_t>(y_t));
  } else {
    auto x_t = make_tensor(
        x.data(),
        make_shape(cint<2>, cint<2>, cint<3>, cint<7>),
        make_strides(cint<42>, cint<21>, cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(),
        make_shape(cint<2>, cint<2>, cint<3>, cint<7>),
        make_strides(cint<42>, cint<21>, cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t), input<float32_t>(b_t), output<float32_t>(y_t));
  }

  for (nint_t i = 0; i < total; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 2e-5f)
        << "rank=" << Rank << " i=" << i;
  }
}

struct AddOneTransform : VecTransform<float32_t, float32_t> {
  template <TLV_DECL_TAG(To), typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::add(v, vec::fill(tag, 1.0f));
  }
};

struct HalfTransform : VecTransform<float32_t, float32_t> {
  template <TLV_DECL_TAG(To), typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::mul(v, vec::fill(tag, 0.5f));
  }
};

} // namespace

TEST(LayerNormDTypeTest, CoversAllFloatInputScaleBiasOutputCombinations) {
  run_all_dtype_combos(LayerNormFloatTypes{});
}

TEST(LayerNormRankTest, CoversRanksOneThroughFour) {
  run_contiguous_rank_case<1>();
  run_contiguous_rank_case<2>();
  run_contiguous_rank_case<3>();
  run_contiguous_rank_case<4>();
}

TEST(LayerNormRankTest, HandlesRuntimeRankFourTensorLayout) {
  constexpr nint_t d0 = 2;
  constexpr nint_t d1 = 2;
  constexpr nint_t d2 = 2;
  constexpr nint_t n = 5;
  constexpr nint_t rows = d0 * d1 * d2;
  std::vector<float> x(rows * n);
  std::vector<float> scale(n);
  std::vector<float> bias(n);
  std::vector<float> out(rows * n, -99.0f);
  std::vector<float> ref(rows * n);

  for (nint_t i = 0; i < rows * n; ++i) x[static_cast<size_t>(i)] = float((i % 13) - 6) * 0.17f;
  for (nint_t i = 0; i < n; ++i) {
    scale[static_cast<size_t>(i)] = 0.75f + 0.04f * float(i);
    bias[static_cast<size_t>(i)] = 0.1f - 0.03f * float(i);
  }
  reference_layernorm_rows(x, scale, bias, rows, n, ref, 1e-5f);

  auto x_t = make_tensor<4>(x.data(), {d0, d1, d2, n});
  auto s_t = make_tensor<1>(scale.data(), {n});
  auto b_t = make_tensor<1>(bias.data(), {n});
  auto y_t = make_tensor<4>(out.data(), {d0, d1, d2, n});
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  auto x_spec = input<float32_t>(x_t);
  auto s_spec = input<float32_t>(s_t);
  auto b_spec = input<float32_t>(b_t);
  auto y_spec = output<float32_t>(y_t);
  Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);

  for (nint_t i = 0; i < rows * n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 2e-5f)
        << "i=" << i;
  }
}

TEST(LayerNormWorkspaceTest, LayoutAndSpecWorkspaceMatchContiguous) {
  using InLayout = Layout<Shape<Const<3>, Const<7>>, Strides<Const<7>, Const<1>>>;
  using VecLayout = Layout<Shape<Const<7>>, Strides<Const<1>>>;
  using OutLayout = Layout<Shape<Const<3>, Const<7>>, Strides<Const<7>, Const<1>>>;

  std::vector<float> x(21);
  std::vector<float> scale(7);
  std::vector<float> bias(7);
  std::vector<float> out(21);

  InLayout in_layout{Shape<Const<3>, Const<7>>{}, Strides<Const<7>, Const<1>>{}};
  VecLayout vec_layout{Shape<Const<7>>{}, Strides<Const<1>>{}};
  OutLayout out_layout{Shape<Const<3>, Const<7>>{}, Strides<Const<7>, Const<1>>{}};
  auto x_t = make_tensor(x.data(), in_layout);
  auto s_t = make_tensor(scale.data(), vec_layout);
  auto b_t = make_tensor(bias.data(), vec_layout);
  auto y_t = make_tensor(out.data(), out_layout);

  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  const nint_t spec_bytes = op.required_workspace(
      input<float32_t>(x_t),
      input<float32_t>(s_t),
      input<float32_t>(b_t),
      output<float32_t>(y_t));
  const nint_t layout_bytes = op.required_workspace(in_layout, vec_layout, vec_layout, out_layout);

  EXPECT_EQ(spec_bytes, layout_bytes);
}

TEST(LayerNormWorkspaceTest, LegacyLayoutSpecsUseRowWorkspace) {
  using InLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<17>, Const<2>>>;
  using VecLayout = Layout<Shape<Const<7>>, Strides<Const<1>>>;
  using OutLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<1>, Const<2>>>;

  InLayout in_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<17>, Const<2>>{}};
  VecLayout vec_layout{Shape<Const<7>>{}, Strides<Const<1>>{}};
  OutLayout out_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<1>, Const<2>>{}};
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});

  auto x_spec = InputSpec<float32_t, float32_t, InLayout>(in_layout);
  auto s_spec = InputSpec<float32_t, float32_t, VecLayout>(vec_layout);
  auto b_spec = InputSpec<float32_t, float32_t, VecLayout>(vec_layout);
  auto y_spec = OutputSpec<float32_t, float32_t, OutLayout>(out_layout);

  const nint_t spec_bytes = op.required_workspace(x_spec, s_spec, b_spec, y_spec);
  const nint_t layout_bytes = op.required_workspace(in_layout, vec_layout, vec_layout, out_layout);

  EXPECT_EQ(spec_bytes, layout_bytes);
  EXPECT_GT(spec_bytes, 0);
}

TEST(LayerNormWorkspaceTest, HandlesStridedInputAndSecondLastContiguousOutput) {
  using InLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<17>, Const<2>>>;
  using OutLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<1>, Const<2>>>;

  std::vector<float> x(40, -100.0f);
  std::vector<float> scale(7);
  std::vector<float> bias(7);
  std::vector<float> out(24, -99.0f);
  std::vector<float> dense_x(14);
  std::vector<float> dense_ref(14);

  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      const float v = float(row * 7 + col - 5) * 0.25f;
      x[static_cast<size_t>(row * 17 + col * 2)] = v;
      dense_x[static_cast<size_t>(row * 7 + col)] = v;
    }
  }
  for (nint_t i = 0; i < 7; ++i) {
    scale[static_cast<size_t>(i)] = 0.5f + 0.1f * float(i);
    bias[static_cast<size_t>(i)] = -0.1f + 0.02f * float(i);
  }
  reference_layernorm_rows(dense_x, scale, bias, 2, 7, dense_ref, 1e-5f);

  InLayout in_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<17>, Const<2>>{}};
  OutLayout out_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<1>, Const<2>>{}};
  auto x_t = make_tensor(x.data(), in_layout);
  auto s_t = make_tensor(scale.data(), make_shape(cint<7>), make_strides(cint<1>));
  auto b_t = make_tensor(bias.data(), make_shape(cint<7>), make_strides(cint<1>));
  auto y_t = make_tensor(out.data(), out_layout);

  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  auto x_spec = input<float32_t>(x_t);
  auto s_spec = input<float32_t>(s_t);
  auto b_spec = input<float32_t>(b_t);
  auto y_spec = output<float32_t>(y_t);
  const nint_t bytes = op.required_workspace(x_spec, s_spec, b_spec, y_spec);
  const nint_t layout_bytes = op.required_workspace(
      in_layout,
      s_t.layout(),
      b_t.layout(),
      out_layout);
  EXPECT_EQ(bytes, layout_bytes);
  EXPECT_GT(bytes, 0);

  Workspace workspace(layout_bytes);
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);

  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      EXPECT_NEAR(
          dense_ref[static_cast<size_t>(row * 7 + col)],
          out[static_cast<size_t>(row + col * 2)],
          2e-5f)
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(LayerNormFusionTest, AppliesInputPrologueAndOutputEpilogue) {
  constexpr nint_t rows = 3;
  constexpr nint_t n = 6;
  std::vector<float> x(rows * n);
  std::vector<float> scale(n);
  std::vector<float> bias(n);
  std::vector<float> out(rows * n, -99.0f);
  std::vector<float> shifted(rows * n);
  std::vector<float> ref(rows * n);

  for (nint_t i = 0; i < rows * n; ++i) {
    x[static_cast<size_t>(i)] = float((i % 9) - 4) * 0.33f;
    shifted[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] + 1.0f;
  }
  for (nint_t i = 0; i < n; ++i) {
    scale[static_cast<size_t>(i)] = 0.9f + 0.02f * float(i);
    bias[static_cast<size_t>(i)] = -0.4f + 0.04f * float(i);
  }
  reference_layernorm_rows(shifted, scale, bias, rows, n, ref, 1e-5f, 0.5f);

  auto x_t = make_tensor<2>(x.data(), {rows, n});
  auto s_t = make_tensor(scale.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto b_t = make_tensor(bias.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto y_t = make_tensor<2>(out.data(), {rows, n});

  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  op(input<float32_t>(x_t, AddOneTransform{}),
     input<float32_t>(s_t),
     input<float32_t>(b_t),
     output<float32_t>(y_t, HalfTransform{}));

  for (nint_t i = 0; i < rows * n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 2e-5f)
        << "i=" << i;
  }
}
