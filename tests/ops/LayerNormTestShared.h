#ifndef VECOPS_TESTS_OPS_LAYERNORM_TEST_SHARED_H
#define VECOPS_TESTS_OPS_LAYERNORM_TEST_SHARED_H

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/ops/LayerNorm.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
using namespace vecops::kernel;
using namespace vecops::ops;

namespace {

template <typename... Ts>
struct TypeList {};

using LayerNormFloatTypes = TypeList<
    vecops::float16_t,
    float32_t,
    float64_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    , vecops::bfloat16_t
#endif
>;

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, vecops::float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, float64_t>) return "fp64";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return "bf16";
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
  if constexpr (std::is_same_v<T, vecops::float16_t>) return 2e-2;
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 4e-2;
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
  const nint_t workspace_bytes = op.required_workspace(x_spec, s_spec, b_spec, y_spec);
  EXPECT_EQ(workspace_bytes, 0);
  Workspace workspace(workspace_bytes);
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
  template <vec::VectorTag To, typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::add(tag, v, vec::fill(tag, 1.0f));
  }
};

struct HalfTransform : VecTransform<float32_t, float32_t> {
  template <vec::VectorTag To, typename... Coords>
  vec::Vec<To> operator()(To tag, vec::Vec<To> v, Coords...) const {
    return vec::mul(tag, v, vec::fill(tag, 0.5f));
  }
};

} // namespace

#endif // VECOPS_TESTS_OPS_LAYERNORM_TEST_SHARED_H
