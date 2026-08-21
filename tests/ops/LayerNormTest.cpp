// @vecops-test-shards: 4

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
#include "TestShard.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
using namespace vecops::kernel;
using namespace vecops::ops;

template <typename T>
void run_layer_norm_dtype_test();

namespace {

template <typename... Ts>
struct TypeList {};

template <typename Op, typename In, typename Gamma, typename Beta,
          typename Out>
concept CanQueryLayerNormWorkspace = requires(
    const Op& op, const In& in, const Gamma& gamma,
    const Beta& beta, const Out& out) {
  op.required_workspace(in, gamma, beta, out);
};

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
    x[static_cast<size_t>(i)] =
        static_cast<TX>(float((i % 5) - 2) * 0.75f + float(i) * 0.05f);
    scale[static_cast<size_t>(i)] =
        static_cast<TS>(0.8f + 0.1f * float(i % 4));
    bias[static_cast<size_t>(i)] =
        static_cast<TB>(-0.3f + 0.07f * float(i % 5));
  }

  auto x_t = make_tensor(x.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto s_t =
      make_tensor(scale.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto b_t =
      make_tensor(bias.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto y_t =
      make_tensor(out.data(), make_shape(Any{n}), make_strides(cint<1>));

  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  auto x_spec = input<float32_t>(x_t);
  auto s_spec = input<float32_t>(s_t);
  auto b_spec = input<float32_t>(b_t);
  auto y_spec = output<float32_t>(y_t);
  const nint_t workspace_bytes =
      op.required_workspace(x_spec, s_spec, b_spec, y_spec);
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

template <bool HasGamma, bool HasBeta>
void run_optional_parameter_case(bool strided) {
  constexpr nint_t rows = 3;
  constexpr nint_t n = 9;
  std::vector<float> x_storage(
      static_cast<size_t>(strided ? rows * 23 : rows * n), -99.0f);
  std::vector<float> gamma(static_cast<size_t>(n));
  std::vector<float> beta(static_cast<size_t>(n));
  std::vector<float> out(static_cast<size_t>(rows * n), -77.0f);
  std::vector<float> dense_x(static_cast<size_t>(rows * n));

  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      const float value = float((row * 11 + col * 7) % 23 - 11) * 0.13f;
      dense_x[static_cast<size_t>(row * n + col)] = value;
      const nint_t offset = strided ? row * 23 + col * 2 : row * n + col;
      x_storage[static_cast<size_t>(offset)] = value;
    }
  }
  for (nint_t col = 0; col < n; ++col) {
    gamma[static_cast<size_t>(col)] = 0.65f + float(col) * 0.04f;
    beta[static_cast<size_t>(col)] = -0.2f + float(col) * 0.03f;
  }

  const auto x_t = strided
      ? make_tensor(
            x_storage.data(), make_shape(cint<rows>, cint<n>),
            make_strides(Any{23}, Any{2}))
      : make_tensor(
            x_storage.data(), make_shape(cint<rows>, cint<n>),
            make_strides(Any{n}, Any{1}));
  auto g_t = make_tensor(
      gamma.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto b_t = make_tensor(
      beta.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto y_t = make_tensor(
      out.data(), make_shape(cint<rows>, cint<n>),
      make_strides(cint<n>, cint<1>));

  auto x_spec = input<float32_t>(x_t);
  auto g_spec = input<float32_t>(g_t);
  auto b_spec = input<float32_t>(b_t);
  auto y_spec = output<float32_t>(y_t);
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});

  const nint_t bytes = [&] {
    if constexpr (HasGamma && HasBeta) {
      return op.required_workspace(x_spec, g_spec, b_spec, y_spec);
    } else if constexpr (HasGamma) {
      return op.required_workspace(x_spec, g_spec, y_spec);
    } else if constexpr (HasBeta) {
      return op.required_workspace(
          x_spec, tensor::nullopt, b_spec, y_spec);
    } else {
      return op.required_workspace(x_spec, y_spec);
    }
  }();
  if (strided) EXPECT_GT(bytes, 0);

  Workspace storage(bytes);
  auto workspace = storage.view();
  if constexpr (HasGamma && HasBeta) {
    op(workspace, x_spec, g_spec, b_spec, y_spec);
  } else if constexpr (HasGamma) {
    op(workspace, x_spec, g_spec, y_spec);
  } else if constexpr (HasBeta) {
    op(workspace, x_spec, tensor::nullopt, b_spec, y_spec);
  } else {
    op(workspace, x_spec, y_spec);
  }

  for (nint_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (nint_t col = 0; col < n; ++col) {
      const float value = dense_x[static_cast<size_t>(row * n + col)];
      sum += value;
      sum_sq += value * value;
    }
    const float mean = sum / float(n);
    const float variance =
        std::max(sum_sq / float(n) - mean * mean, 0.0f);
    const float rstd = 1.0f / std::sqrt(variance + 1e-5f);
    for (nint_t col = 0; col < n; ++col) {
      float expected =
          (dense_x[static_cast<size_t>(row * n + col)] - mean) * rstd;
      if constexpr (HasGamma) expected *= gamma[static_cast<size_t>(col)];
      if constexpr (HasBeta) expected += beta[static_cast<size_t>(col)];
      EXPECT_NEAR(
          expected, out[static_cast<size_t>(row * n + col)], 2e-5f)
          << "row=" << row << " col=" << col
          << " gamma=" << HasGamma << " beta=" << HasBeta
          << " strided=" << strided;
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
  for (int i = 0; i < Rank - 1; ++i)
    rows *= shape[static_cast<size_t>(i)];
  const nint_t total = rows * n;

  std::vector<float> x(static_cast<size_t>(total));
  std::vector<float> scale(static_cast<size_t>(n));
  std::vector<float> bias(static_cast<size_t>(n));
  std::vector<float> out(static_cast<size_t>(total), -99.0f);
  std::vector<float> ref(static_cast<size_t>(total));

  for (nint_t i = 0; i < total; ++i)
    x[static_cast<size_t>(i)] = float((i % 11) - 5) * 0.2f;
  for (nint_t i = 0; i < n; ++i) {
    scale[static_cast<size_t>(i)] = 0.7f + 0.05f * float(i);
    bias[static_cast<size_t>(i)] = -0.2f + 0.03f * float(i);
  }
  reference_layernorm_rows(x, scale, bias, rows, n, ref, 1e-5f);

  auto s_t =
      make_tensor(scale.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto b_t =
      make_tensor(bias.data(), make_shape(Any{n}), make_strides(cint<1>));
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});

  if constexpr (Rank == 1) {
    auto x_t =
        make_tensor(x.data(), make_shape(cint<7>), make_strides(cint<1>));
    auto y_t =
        make_tensor(out.data(), make_shape(cint<7>), make_strides(cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t),
       input<float32_t>(b_t), output<float32_t>(y_t));
  } else if constexpr (Rank == 2) {
    auto x_t = make_tensor(
        x.data(), make_shape(cint<3>, cint<7>),
        make_strides(cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(), make_shape(cint<3>, cint<7>),
        make_strides(cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t),
       input<float32_t>(b_t), output<float32_t>(y_t));
  } else if constexpr (Rank == 3) {
    auto x_t = make_tensor(
        x.data(),
        make_shape(cint<2>, cint<3>, cint<7>),
        make_strides(cint<21>, cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(),
        make_shape(cint<2>, cint<3>, cint<7>),
        make_strides(cint<21>, cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t),
       input<float32_t>(b_t), output<float32_t>(y_t));
  } else {
    auto x_t = make_tensor(
        x.data(),
        make_shape(cint<2>, cint<2>, cint<3>, cint<7>),
        make_strides(cint<42>, cint<21>, cint<7>, cint<1>));
    auto y_t = make_tensor(
        out.data(),
        make_shape(cint<2>, cint<2>, cint<3>, cint<7>),
        make_strides(cint<42>, cint<21>, cint<7>, cint<1>));
    op(input<float32_t>(x_t), input<float32_t>(s_t),
       input<float32_t>(b_t), output<float32_t>(y_t));
  }

  for (nint_t i = 0; i < total; ++i) {
    EXPECT_NEAR(
        ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 2e-5f)
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

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == 4);

template <typename T>
void run_layer_norm_dtype_test() {
  run_all_dtype_combos(TypeList<T>{});
}

using LayerNormShardTypes =
    std::tuple<vecops::float16_t, float32_t, float64_t>;

#if VECOPS_TEST_SHARD_INDEX < 3
using LayerNormShardType =
    std::tuple_element_t<VECOPS_TEST_SHARD_INDEX, LayerNormShardTypes>;
template void run_layer_norm_dtype_test<LayerNormShardType>();
#elif defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
template void run_layer_norm_dtype_test<vecops::bfloat16_t>();
#endif

#else

TEST(LayerNormDTypeTest, CoversAllCombinationsWithFp16Input) {
  run_layer_norm_dtype_test<vecops::float16_t>();
}

TEST(LayerNormDTypeTest, CoversAllCombinationsWithFp32Input) {
  run_layer_norm_dtype_test<float32_t>();
}

TEST(LayerNormDTypeTest, CoversAllCombinationsWithFp64Input) {
  run_layer_norm_dtype_test<float64_t>();
}

#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
TEST(LayerNormDTypeTest, CoversAllCombinationsWithBf16Input) {
  run_layer_norm_dtype_test<vecops::bfloat16_t>();
}
#endif

TEST(LayerNormRankTest, CoversRanksOneThroughFour) {
  run_contiguous_rank_case<1>();
  run_contiguous_rank_case<2>();
  run_contiguous_rank_case<3>();
  run_contiguous_rank_case<4>();
}

TEST(LayerNormOptionalParameterTest, CoversAllCompileTimeAffineModes) {
  run_optional_parameter_case<true, true>(false);
  run_optional_parameter_case<true, false>(false);
  run_optional_parameter_case<false, true>(false);
  run_optional_parameter_case<false, false>(false);
}

TEST(LayerNormOptionalParameterTest, CoversDeferredStridedInput) {
  run_optional_parameter_case<true, false>(true);
  run_optional_parameter_case<false, true>(true);
  run_optional_parameter_case<false, false>(true);
}

TEST(LayerNormOptionalParameterTest, ConvenienceOverloadsMatchPlaceholders) {
  constexpr nint_t n = 7;
  std::array<float, n> x{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 1.5f, 0.25f};
  std::array<float, n> gamma{0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f};
  std::array<float, n> y_short{};
  std::array<float, n> y_full{};

  auto x_t = make_tensor(
      x.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto g_t = make_tensor(
      gamma.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto ys_t = make_tensor(
      y_short.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto yf_t = make_tensor(
      y_full.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  auto x_spec = input<float32_t>(x_t);
  auto g_spec = input<float32_t>(g_t);

  op(x_spec, g_spec, output<float32_t>(ys_t));
  op(x_spec, g_spec, tensor::nullopt, output<float32_t>(yf_t));
  EXPECT_EQ(y_short, y_full);

  y_short.fill(0.0f);
  y_full.fill(0.0f);
  op(x_spec, output<float32_t>(ys_t));
  op(x_spec, tensor::nullopt, tensor::nullopt, output<float32_t>(yf_t));
  EXPECT_EQ(y_short, y_full);
}

TEST(LayerNormCompositionTest, AcceptsRawTensorsAndBoundRowAccesses) {
  constexpr nint_t n = 9;
  std::array<float, n> x{2.0f, -1.0f, 0.5f, 3.0f, -2.0f,
                         1.5f, 0.25f, -0.75f, 4.0f};
  std::array<float, n> gamma{0.7f, 0.8f, 0.9f, 1.0f, 1.1f,
                             1.2f, 1.3f, 1.4f, 1.5f};
  std::array<float, n> tensor_out{};
  std::array<float, n> bound_out{};
  auto x_t = make_tensor(
      x.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto g_t = make_tensor(
      gamma.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto tensor_y_t = make_tensor(
      tensor_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto bound_y_t = make_tensor(
      bound_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});

  EXPECT_EQ(op.required_workspace(x_t, g_t, tensor_y_t), 0);
  op(x_t, g_t, tensor_y_t);

  auto x_spec = input<float32_t>(x_t);
  auto g_spec = input<float32_t>(g_t);
  auto y_spec = output<float32_t>(bound_y_t);
  Workspace storage(0);
  auto workspace = storage.view();
  auto x_access = bind(
      x_spec, InputAccessPolicy<0, 2, AccessPlan::direct>{}, workspace);
  auto gamma_access = bind(
      g_spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, workspace);
  auto y_access = bind(
      y_spec, OutputAccessPolicy<0, AccessPlan::direct>{}, workspace);
  auto no_beta = tensor::nullopt;
  op.run_bound(x_access, gamma_access, no_beta, y_access);
  EXPECT_EQ(tensor_out, bound_out);
}

TEST(LayerNormOptionalParameterTest, RejectsRankTwoParameterAtInterface) {
  using InTensor = decltype(make_tensor(
      static_cast<float*>(nullptr), make_shape(cint<2>, cint<7>),
      make_strides(cint<7>, cint<1>)));
  using ParamTensor = decltype(make_tensor(
      static_cast<float*>(nullptr), make_shape(cint<7>),
      make_strides(cint<1>)));
  using OutTensor = InTensor;
  using BadParamTensor = InTensor;
  using InSpec = decltype(input<float32_t>(std::declval<InTensor>()));
  using ParamSpec = decltype(input<float32_t>(std::declval<ParamTensor>()));
  using BadParamSpec = decltype(
      input<float32_t>(std::declval<BadParamTensor>()));
  using OutSpec = decltype(output<float32_t>(std::declval<OutTensor>()));
  using Op = LayerNorm<LayerNormConfig<float32_t>>;
  static_assert(CanQueryLayerNormWorkspace<
                Op, InSpec, ParamSpec, ParamSpec, OutSpec>);
  static_assert(!CanQueryLayerNormWorkspace<
                Op, InSpec, BadParamSpec, ParamSpec, OutSpec>);
}

#if defined(VECOPS_DEBUG)
TEST(LayerNormOptionalParameterDeathTest, RejectsPresentParameterSizeMismatch) {
  std::array<float, 14> x{};
  std::array<float, 6> gamma{};
  std::array<float, 14> y{};
  auto x_t = make_tensor(
      x.data(), make_shape(cint<2>, cint<7>),
      make_strides(cint<7>, cint<1>));
  auto g_t = make_tensor(
      gamma.data(), make_shape(cint<6>), make_strides(cint<1>));
  auto y_t = make_tensor(
      y.data(), make_shape(cint<2>, cint<7>),
      make_strides(cint<7>, cint<1>));
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  EXPECT_DEATH_IF_SUPPORTED(
      (void)op.required_workspace(
          input<float32_t>(x_t), input<float32_t>(g_t),
          tensor::nullopt, output<float32_t>(y_t)),
      "LayerNorm parameter size mismatch");
}
#endif

TEST(LayerNormVectorBoundaryTest, CoversGroupedLoopAndTailBoundaries) {
  using Tag = LayerNorm<LayerNormConfig<float32_t>>::Tag;
  const nint_t lanes = vec::size(Tag{});
  ASSERT_GT(lanes, 0);

  const std::array<nint_t, 8> sizes = {
      1,
      std::max<nint_t>(1, lanes - 1),
      lanes,
      lanes + 1,
      4 * lanes - 1,
      4 * lanes,
      4 * lanes + 1,
      5 * lanes + 3};

  for (const nint_t n : sizes) {
    SCOPED_TRACE("normalized_size=" + std::to_string(n));
    constexpr nint_t rows = 3;
    std::vector<float> x(static_cast<size_t>(rows * n));
    std::vector<float> scale(static_cast<size_t>(n));
    std::vector<float> bias(static_cast<size_t>(n));
    std::vector<float> out(static_cast<size_t>(rows * n), -99.0f);
    std::vector<float> ref(static_cast<size_t>(rows * n));

    for (nint_t i = 0; i < rows * n; ++i) {
      x[static_cast<size_t>(i)] = float((i * 7) % 29 - 14) * 0.11f;
    }
    for (nint_t i = 0; i < n; ++i) {
      scale[static_cast<size_t>(i)] = 0.65f + float(i % 17) * 0.013f;
      bias[static_cast<size_t>(i)] = -0.18f + float(i % 13) * 0.009f;
    }
    reference_layernorm_rows(x, scale, bias, rows, n, ref, 1e-5f);

    auto x_t = make_tensor<2>(x.data(), {rows, n});
    auto s_t = make_tensor<1>(scale.data(), {n});
    auto b_t = make_tensor<1>(bias.data(), {n});
    auto y_t = make_tensor<2>(out.data(), {rows, n});
    auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
    op(
        input<float32_t>(x_t),
        input<float32_t>(s_t),
        input<float32_t>(b_t),
        output<float32_t>(y_t));

    for (nint_t i = 0; i < rows * n; ++i) {
      EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 3e-5f)
          << "i=" << i;
    }
  }
}

template <typename T, bool HasGamma = true, bool HasBeta = true>
void run_sve_16bit_boundary_cases() {
  using Tag = LayerNorm<LayerNormConfig<float32_t>>::Tag;
  const nint_t half_lanes = 2 * vec::size(Tag{});
  const std::array<nint_t, 8> sizes = {
      1,
      std::max<nint_t>(1, half_lanes - 1),
      half_lanes,
      half_lanes + 1,
      2 * half_lanes - 1,
      2 * half_lanes,
      2 * half_lanes + 1,
      3 * half_lanes + 5};

  for (const nint_t n : sizes) {
    SCOPED_TRACE(std::string(dtype_name<T>()) + " normalized_size=" +
                 std::to_string(n));
    constexpr nint_t rows = 3;
    std::vector<T> x(static_cast<size_t>(rows * n));
    std::vector<T> scale(static_cast<size_t>(n));
    std::vector<T> bias(static_cast<size_t>(n));
    std::vector<T> out(static_cast<size_t>(rows * n), T{});

    for (nint_t i = 0; i < rows * n; ++i)
      x[static_cast<size_t>(i)] =
          static_cast<T>(float((i * 7) % 29 - 14) * 0.11f);
    for (nint_t i = 0; i < n; ++i) {
      scale[static_cast<size_t>(i)] =
          static_cast<T>(0.65f + float(i % 17) * 0.013f);
      bias[static_cast<size_t>(i)] =
          static_cast<T>(-0.18f + float(i % 13) * 0.009f);
    }

    auto x_t = make_tensor<2>(x.data(), {rows, n});
    auto s_t = make_tensor<1>(scale.data(), {n});
    auto b_t = make_tensor<1>(bias.data(), {n});
    auto y_t = make_tensor<2>(out.data(), {rows, n});
    auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
    if constexpr (HasGamma && HasBeta) {
      op(input<float32_t>(x_t), input<float32_t>(s_t),
         input<float32_t>(b_t), output<float32_t>(y_t));
    } else if constexpr (HasGamma) {
      op(input<float32_t>(x_t), input<float32_t>(s_t),
         output<float32_t>(y_t));
    } else if constexpr (HasBeta) {
      op(input<float32_t>(x_t), tensor::nullopt,
         input<float32_t>(b_t), output<float32_t>(y_t));
    } else {
      op(input<float32_t>(x_t), output<float32_t>(y_t));
    }

    for (nint_t row = 0; row < rows; ++row) {
      float sum = 0.0f;
      float sum_sq = 0.0f;
      for (nint_t col = 0; col < n; ++col) {
        const float v = static_cast<float>(
            x[static_cast<size_t>(row * n + col)]);
        sum += v;
        sum_sq += v * v;
      }
      const float mean = sum / float(n);
      const float variance =
          std::max(sum_sq / float(n) - mean * mean, 0.0f);
      const float rstd = 1.0f / std::sqrt(variance + 1e-5f);
      for (nint_t col = 0; col < n; ++col) {
        const size_t index = static_cast<size_t>(row * n + col);
        float expected_value =
            (static_cast<float>(x[index]) - mean) * rstd;
        if constexpr (HasGamma) {
          expected_value *=
              static_cast<float>(scale[static_cast<size_t>(col)]);
        }
        if constexpr (HasBeta) {
          expected_value +=
              static_cast<float>(bias[static_cast<size_t>(col)]);
        }
        const T expected = static_cast<T>(expected_value);
        EXPECT_NEAR(as_double(expected), as_double(out[index]), tolerance<T>())
            << "row=" << row << " col=" << col;
      }
    }
  }
}

TEST(LayerNormVectorBoundaryTest, CoversSVE16BitFullLoadsAndTails) {
  run_sve_16bit_boundary_cases<vecops::float16_t>();
  run_sve_16bit_boundary_cases<vecops::float16_t, true, false>();
  run_sve_16bit_boundary_cases<vecops::float16_t, false, true>();
  run_sve_16bit_boundary_cases<vecops::float16_t, false, false>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  run_sve_16bit_boundary_cases<vecops::bfloat16_t>();
  run_sve_16bit_boundary_cases<vecops::bfloat16_t, true, false>();
  run_sve_16bit_boundary_cases<vecops::bfloat16_t, false, true>();
  run_sve_16bit_boundary_cases<vecops::bfloat16_t, false, false>();
#endif
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
  const nint_t workspace_bytes = op.required_workspace(x_spec, s_spec, b_spec, y_spec);
  EXPECT_EQ(workspace_bytes, 0);
  Workspace workspace(workspace_bytes);
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);

  for (nint_t i = 0; i < rows * n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 2e-5f)
        << "i=" << i;
  }
}

TEST(LayerNormWorkspaceTest, ContiguousSpecsNeedNoWorkspace) {
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
  EXPECT_EQ(spec_bytes, 0);
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
  EXPECT_GT(bytes, 0);

  Workspace workspace(bytes);
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);
  EXPECT_LE(view.high_watermark(), bytes + vec::DEFAULT_ALIGNMENT);

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

// A compile-time non-unit row stride forces the strided row path: the input
// materializes before the two-pass reduction and the output scatters or
// stores through the converting path. Half-precision storage additionally
// exercises the 16-bit row recipe on SVE2, including its strided lowering.
template <typename T>
void run_strided_16bit_row_case() {
  using InLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<17>, Const<2>>>;
  using OutLayout = Layout<Shape<Const<2>, Const<7>>, Strides<Const<7>, Const<1>>>;

  std::vector<T> x(40, T{});
  std::vector<T> scale(7);
  std::vector<T> bias(7);
  std::vector<T> out(14, T{});
  std::vector<float> dense_x(14);
  std::vector<float> dense_scale(7);
  std::vector<float> dense_bias(7);
  std::vector<float> dense_ref(14);

  // Fill storage buffers with contiguous stores and derive the strided views
  // from index arithmetic: BiSheng 5.1 fails to select a scalar FCVT/BFCVT
  // feeding a constant-offset pre-increment store, which fully unrolled
  // strided fills of half-precision vectors would produce. `off` is zero at
  // runtime and keeps the converted values off the constant-folding path.
  volatile float opaque_zero = 0.0f;
  const float off = opaque_zero;
  auto value_at = [off](nint_t j) {
    return float((j * 7) % 29 - 14) * 0.11f + off;
  };

  for (nint_t j = 0; j < 40; ++j) {
    x[static_cast<size_t>(j)] = static_cast<T>(value_at(j));
  }
  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      dense_x[static_cast<size_t>(row * 7 + col)] =
          value_at(row * 17 + col * 2);
    }
  }
  for (nint_t i = 0; i < 7; ++i) {
    dense_scale[static_cast<size_t>(i)] = 0.5f + 0.1f * float(i) + off;
    dense_bias[static_cast<size_t>(i)] = -0.1f + 0.02f * float(i) + off;
    scale[static_cast<size_t>(i)] = static_cast<T>(dense_scale[static_cast<size_t>(i)]);
    bias[static_cast<size_t>(i)] = static_cast<T>(dense_bias[static_cast<size_t>(i)]);
  }

  InLayout in_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<17>, Const<2>>{}};
  OutLayout out_layout{Shape<Const<2>, Const<7>>{}, Strides<Const<7>, Const<1>>{}};
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
  EXPECT_GT(bytes, 0);

  Workspace workspace(bytes);
  auto view = workspace.view();
  op(view, x_spec, s_spec, b_spec, y_spec);

  reference_layernorm_rows(dense_x, dense_scale, dense_bias, 2, 7, dense_ref, 1e-5f);
  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      const T expected =
          static_cast<T>(dense_ref[static_cast<size_t>(row * 7 + col)]);
      EXPECT_NEAR(
          as_double(expected),
          as_double(out[static_cast<size_t>(row * 7 + col)]),
          tolerance<T>())
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(LayerNormWorkspaceTest, HandlesStrided16BitRows) {
  run_strided_16bit_row_case<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  run_strided_16bit_row_case<vecops::bfloat16_t>();
#endif
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

TEST(LayerNormFusionTest, AppliesTransformsWithoutAffineParameters) {
  constexpr nint_t rows = 2;
  constexpr nint_t n = 7;
  std::array<float, rows * n> x{};
  std::array<float, rows * n> out{};
  for (nint_t i = 0; i < rows * n; ++i) {
    x[static_cast<size_t>(i)] = float((i * 5) % 17 - 8) * 0.21f;
  }
  auto x_t = make_tensor<2>(x.data(), {rows, n});
  auto y_t = make_tensor<2>(out.data(), {rows, n});
  auto op = layer_norm(LayerNormConfig<float32_t>{.eps = 1e-5f});
  op(input<float32_t>(x_t, AddOneTransform{}),
     output<float32_t>(y_t, HalfTransform{}));

  for (nint_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (nint_t col = 0; col < n; ++col) {
      const float value = x[static_cast<size_t>(row * n + col)] + 1.0f;
      sum += value;
      sum_sq += value * value;
    }
    const float mean = sum / float(n);
    const float variance =
        std::max(sum_sq / float(n) - mean * mean, 0.0f);
    const float rstd = 1.0f / std::sqrt(variance + 1e-5f);
    for (nint_t col = 0; col < n; ++col) {
      const float value = x[static_cast<size_t>(row * n + col)] + 1.0f;
      EXPECT_NEAR(out[static_cast<size_t>(row * n + col)],
                  (value - mean) * rstd * 0.5f, 2e-5f);
    }
  }
}

#endif
