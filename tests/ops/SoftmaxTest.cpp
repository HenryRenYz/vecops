#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/tensor/DataAccess.h"
#include "vecops/ops/Softmax.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
using namespace vecops::kernel;
using namespace vecops::ops;

namespace {

template <typename... Ts>
struct TypeList {};

using SoftmaxFloatTypes = TypeList<vecops::float16_t, vecops::float32_t, vecops::float64_t
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
                                   ,
                                   vecops::bfloat16_t
#endif
                                   >;

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, vecops::float16_t>)
    return "fp16";
  if constexpr (std::is_same_v<T, vecops::float32_t>)
    return "fp32";
  if constexpr (std::is_same_v<T, vecops::float64_t>)
    return "fp64";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>)
    return "bf16";
  return typeid(T).name();
}

template <vec::Accuracy Mode, typename T>
double tolerance() {
  if constexpr (Mode == vec::Accuracy::Estimate) {
    if constexpr (std::is_same_v<T, vecops::bfloat16_t>)
      return 2e-2;
    if constexpr (std::is_same_v<T, vecops::float16_t>)
      return 1e-2;
    return 8e-3;
  }
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>)
    return 8e-3;
  if constexpr (std::is_same_v<T, vecops::float16_t>)
    return 2e-3;
  if constexpr (std::is_same_v<T, vecops::float64_t>)
    return 2e-12;
  return 3e-5;
}

void reference_softmax_rows(const std::vector<double>& x, nint_t rows, nint_t n, std::vector<double>& ref) {
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

template <vec::Accuracy Mode, typename TX, typename TY>
void run_dtype_combo() {
  constexpr nint_t n = 9;
  using ComputeT = std::conditional_t<std::is_same_v<TX, vecops::float64_t> || std::is_same_v<TY, vecops::float64_t>,
                                      vecops::float64_t, vecops::float32_t>;
  using Config = SoftmaxConfig<ComputeT, Mode>;

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
  EXPECT_EQ(workspace_bytes, align_up(n * static_cast<nint_t>(sizeof(ComputeT)), vec::DEFAULT_ALIGNMENT));
  Workspace workspace(workspace_bytes);
  auto view = workspace.view();
  op(view, x_spec, y_spec);

  const double tol = tolerance<Mode, TY>();
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(ref[static_cast<size_t>(i)], static_cast<double>(out[static_cast<size_t>(i)]), tol)
      << "mode=" << static_cast<int>(Mode) << " input=" << dtype_name<TX>() << " output=" << dtype_name<TY>()
      << " i=" << i;
  }
}

template <vec::Accuracy Mode, typename TX, typename... TYs>
void run_out_combos(TypeList<TYs...>) {
  (run_dtype_combo<Mode, TX, TYs>(), ...);
}

template <vec::Accuracy Mode, typename... TXs>
void run_all_dtype_combos(TypeList<TXs...>) {
  (run_out_combos<Mode, TXs>(SoftmaxFloatTypes{}), ...);
}

template <vec::Accuracy Mode, typename T>
void run_vector_boundary_cases() {
  using ComputeT = std::conditional_t<std::is_same_v<T, vecops::float64_t>, vecops::float64_t, vecops::float32_t>;
  using Tag = vec::ScalableTag<ComputeT, 0>;
  using Config = SoftmaxConfig<ComputeT, Mode>;
  const nint_t vl = vec::size(Tag{});
  const std::array<nint_t, 15> counts{1,       vl - 1,      vl,     vl + 1,     2 * vl - 1,
                                      2 * vl,  2 * vl + 1,  4 * vl, 4 * vl + 1, 16 * vl - 1,
                                      16 * vl, 16 * vl + 1, 1023,   1024,       1025};

  for (const nint_t n : counts) {
    if (n <= 0)
      continue;
    std::vector<T> x(static_cast<size_t>(n));
    std::vector<T> out(static_cast<size_t>(n + 2), static_cast<T>(-17.0));
    std::vector<double> x_ref(static_cast<size_t>(n));
    std::vector<double> ref(static_cast<size_t>(n));
    for (nint_t i = 0; i < n; ++i) {
      x[static_cast<size_t>(i)] = static_cast<T>(-8.0 - 0.125 * double((i * 7) % 19));
    }
    x.back() = static_cast<T>(-0.25);
    if (n > 2)
      x[1] = static_cast<T>(-std::numeric_limits<double>::infinity());
    for (nint_t i = 0; i < n; ++i) {
      x_ref[static_cast<size_t>(i)] = static_cast<double>(x[static_cast<size_t>(i)]);
    }
    reference_softmax_rows(x_ref, 1, n, ref);

    auto x_t = make_tensor<1>(x.data(), {n});
    auto y_t = make_tensor<1>(out.data() + 1, {n});
    softmax(Config{})(input<ComputeT>(x_t), output<ComputeT>(y_t));

    EXPECT_EQ(static_cast<double>(out.front()), -17.0) << "n=" << n;
    EXPECT_EQ(static_cast<double>(out.back()), -17.0) << "n=" << n;
    const double tol = tolerance<Mode, T>();
    for (nint_t i = 0; i < n; ++i) {
      EXPECT_NEAR(ref[static_cast<size_t>(i)], static_cast<double>(out[static_cast<size_t>(i + 1)]), tol)
        << "mode=" << static_cast<int>(Mode) << " dtype=" << dtype_name<T>() << " n=" << n << " i=" << i;
    }
  }
}

template <vec::Accuracy Mode, typename... Ts>
void run_all_vector_boundary_cases(TypeList<Ts...>) {
  (run_vector_boundary_cases<Mode, Ts>(), ...);
}

enum class LargeRowPattern {
  Increasing,
  Decreasing,
  Alternating,
  MixedNegativeInfinity,
  FullTileNegativeInfinity,
};

const char* large_row_pattern_name(LargeRowPattern pattern) {
  switch (pattern) {
  case LargeRowPattern::Increasing:
    return "increasing";
  case LargeRowPattern::Decreasing:
    return "decreasing";
  case LargeRowPattern::Alternating:
    return "alternating";
  case LargeRowPattern::MixedNegativeInfinity:
    return "mixed-negative-infinity";
  case LargeRowPattern::FullTileNegativeInfinity:
    return "full-tile-negative-infinity";
  }
  return "unknown";
}

template <typename T>
auto make_online_test_tensor(T* data, nint_t rows, nint_t n) {
#if defined(CPU_CAPABILITY_SVE)
  return make_tensor(data, make_shape(dyn<1, 64, 256>(rows), dyn<1, 8192, 16384>(n)));
#elif defined(CPU_CAPABILITY_AVX512)
  return make_tensor(data, make_shape(dyn<1, 256, 4096>(rows), dyn<1, 2048, 4096>(n)));
#else
  return make_tensor<2>(data, {rows, n});
#endif
}

template <typename Config>
void run_large_row_reference_case(Config config, bool in_place, const char* config_name) {
  using ComputeT = typename Config::ComputeType;
  using Tag = vec::ScalableTag<ComputeT, 0>;
  static_assert(std::is_same_v<ComputeT, vecops::float32_t>);

  constexpr std::array patterns{LargeRowPattern::Increasing, LargeRowPattern::Decreasing, LargeRowPattern::Alternating,
                                LargeRowPattern::MixedNegativeInfinity, LargeRowPattern::FullTileNegativeInfinity};
  const nint_t vl = vec::size(Tag{});
#if defined(CPU_CAPABILITY_SVE)
  const nint_t n = 512 * vl + 3;
#else
  const nint_t n = 256 * vl - 1;
#endif
  const nint_t rows = ceil_div(nint_t{1024 * 1024}, n);
  const nint_t total = rows * n;
  const float negative_infinity = -std::numeric_limits<float>::infinity();

  std::vector<float> input_values(static_cast<size_t>(total));
  std::vector<double> input_reference(static_cast<size_t>(total));
  std::vector<double> reference(static_cast<size_t>(total));
  for (nint_t row = 0; row < rows; ++row) {
    const LargeRowPattern pattern = patterns[static_cast<size_t>(row) % patterns.size()];
    for (nint_t col = 0; col < n; ++col) {
      float value = 0.0f;
      switch (pattern) {
      case LargeRowPattern::Increasing:
        value = -8.0f + 16.0f * static_cast<float>(col) / static_cast<float>(n - 1);
        break;
      case LargeRowPattern::Decreasing:
        value = 8.0f - 16.0f * static_cast<float>(col) / static_cast<float>(n - 1);
        break;
      case LargeRowPattern::Alternating: {
        const float magnitude = 0.125f * static_cast<float>((col * 13) % 61);
        value = col % 2 == 0 ? magnitude : -magnitude;
        break;
      }
      case LargeRowPattern::MixedNegativeInfinity:
        value = col % 11 == 0 ? negative_infinity : -6.0f + 0.25f * static_cast<float>((col * 17 + 5) % 47);
        break;
      case LargeRowPattern::FullTileNegativeInfinity:
        value =
          col >= 4 * vl && col < 8 * vl ? negative_infinity : -5.0f + 0.2f * static_cast<float>((col * 7 + 3) % 43);
        break;
      }
      const nint_t index = row * n + col;
      input_values[static_cast<size_t>(index)] = value;
      input_reference[static_cast<size_t>(index)] = value;
    }
  }
  reference_softmax_rows(input_reference, rows, n, reference);

  std::vector<float> output_values(static_cast<size_t>(total + 2), -17.0f);
  auto x_t = make_online_test_tensor(input_values.data(), rows, n);
  if (in_place) {
    softmax(config)(input<ComputeT>(x_t), output<ComputeT>(x_t));
  } else {
    auto y_t = make_online_test_tensor(output_values.data() + 1, rows, n);
    softmax(config)(input<ComputeT>(x_t), output<ComputeT>(y_t));
    EXPECT_FLOAT_EQ(output_values.front(), -17.0f);
    EXPECT_FLOAT_EQ(output_values.back(), -17.0f);
  }

  const auto& actual_values = in_place ? input_values : output_values;
  const size_t actual_offset = in_place ? 0 : 1;
  for (nint_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      const nint_t index = row * n + col;
      const double actual = static_cast<double>(actual_values[actual_offset + static_cast<size_t>(index)]);
      EXPECT_NEAR(reference[static_cast<size_t>(index)], actual, 5e-5)
        << "config=" << config_name << " in_place=" << in_place
        << " pattern=" << large_row_pattern_name(patterns[static_cast<size_t>(row) % patterns.size()]) << " n=" << n
        << " col=" << col;
      sum += actual;
    }
    EXPECT_NEAR(sum, 1.0, 5e-5) << "config=" << config_name << " in_place=" << in_place << " pattern="
                                << large_row_pattern_name(patterns[static_cast<size_t>(row) % patterns.size()])
                                << " n=" << n;
  }
}

void run_online_nonfinite_equivalence_case(bool in_place) {
  using Config = SoftmaxConfig<>;
  using Tag = vec::ScalableTag<typename Config::ComputeType, 0>;
  const nint_t vl = vec::size(Tag{});
#if defined(CPU_CAPABILITY_SVE)
  const nint_t n = 512 * vl + 3;
#else
  const nint_t n = 256 * vl - 1;
#endif
  const nint_t rows = ceil_div(nint_t{1024 * 1024}, n);
  const nint_t total = rows * n;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> input_values(static_cast<size_t>(total));
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      input_values[static_cast<size_t>(row * n + col)] = -5.0f + 0.125f * static_cast<float>((col * 17 + row * 3) % 73);
    }
    if (row % 4 == 0) {
      input_values[static_cast<size_t>(row * n + row % n)] = nan;
    } else if (row % 4 == 1) {
      input_values[static_cast<size_t>(row * n + (3 * row) % n)] = inf;
    } else if (row % 4 == 2) {
      std::fill_n(input_values.begin() + static_cast<size_t>(row * n), static_cast<size_t>(n), -inf);
    } else {
      const nint_t tile_begin = 4 * vl;
      std::fill_n(input_values.begin() + static_cast<size_t>(row * n + tile_begin), static_cast<size_t>(4 * vl), -inf);
    }
  }

  struct ExecutionResult {
    std::vector<float> values;
    nint_t requested_workspace;
    nint_t high_watermark;
  };

  const auto execute = [&]<typename RunConfig>() -> ExecutionResult {
    std::vector<float> input_buffer = input_values;
    std::vector<float> output_buffer(static_cast<size_t>(total));
    auto x_t = make_online_test_tensor(input_buffer.data(), rows, n);
    auto op = softmax(RunConfig{});
    if (in_place) {
      auto x_spec = input<float>(x_t);
      auto y_spec = output<float>(x_t);
      const nint_t bytes = op.required_workspace(x_spec, y_spec);
      Workspace storage(bytes);
      auto workspace = storage.view();
      op(workspace, x_spec, y_spec);
      EXPECT_EQ(workspace.used(), 0);
      EXPECT_LE(workspace.high_watermark(), storage.storage_size());
      return {std::move(input_buffer), bytes, workspace.high_watermark()};
    }
    auto y_t = make_online_test_tensor(output_buffer.data(), rows, n);
    auto x_spec = input<float>(x_t);
    auto y_spec = output<float>(y_t);
    const nint_t bytes = op.required_workspace(x_spec, y_spec);
    Workspace storage(bytes);
    auto workspace = storage.view();
    op(workspace, x_spec, y_spec);
    EXPECT_EQ(workspace.used(), 0);
    EXPECT_LE(workspace.high_watermark(), storage.storage_size());
    return {std::move(output_buffer), bytes, workspace.high_watermark()};
  };

  using DisabledConfig = SoftmaxConfig<typename Config::ComputeType, Config::exp_accuracy, false>;
  const auto regular = execute.template operator()<DisabledConfig>();
  const auto automatic = execute.template operator()<Config>();
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
  EXPECT_GT(automatic.requested_workspace, regular.requested_workspace);
  const nint_t tile_count = ceil_div(n, 4 * vl);
  const nint_t expected_online_workspace =
    WorkspaceView::allocation_bytes<float>(n) + 2 * WorkspaceView::allocation_bytes<float>(tile_count);
  EXPECT_EQ(automatic.requested_workspace, expected_online_workspace);
#endif
  EXPECT_GT(automatic.high_watermark, 0);
  for (nint_t i = 0; i < total; ++i) {
    const float expected = regular.values[static_cast<size_t>(i)];
    const float actual = automatic.values[static_cast<size_t>(i)];
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "in_place=" << in_place << " i=" << i;
    } else if (std::isinf(expected)) {
      EXPECT_EQ(expected, actual) << "in_place=" << in_place << " i=" << i;
    } else {
      EXPECT_NEAR(expected, actual, 5e-5f) << "in_place=" << in_place << " i=" << i;
    }
  }
}

struct LegacySoftmaxConfig {
  using ComputeType = vecops::float32_t;
  static constexpr vec::Accuracy exp_accuracy = vec::Accuracy::Strict;
};

template <int Rank>
void run_contiguous_rank_case() {
  constexpr nint_t n = 7;
  std::array<nint_t, Rank> shape{};
  if constexpr (Rank == 1)
    shape = {n};
  if constexpr (Rank == 2)
    shape = {3, n};
  if constexpr (Rank == 3)
    shape = {2, 3, n};
  if constexpr (Rank == 4)
    shape = {2, 2, 3, n};

  nint_t rows = 1;
  for (int d = 0; d < Rank - 1; ++d)
    rows *= shape[static_cast<size_t>(d)];
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
    EXPECT_NEAR(ref[static_cast<size_t>(i)], out[static_cast<size_t>(i)], 3e-5) << "rank=" << Rank << " i=" << i;
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

template <vec::Accuracy Mode>
void run_shift_stability_case() {
  using Config = SoftmaxConfig<vecops::float32_t, Mode>;
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
  run_all_dtype_combos<vec::Accuracy::Strict>(SoftmaxFloatTypes{});
  run_all_dtype_combos<vec::Accuracy::Fast>(SoftmaxFloatTypes{});
  run_all_dtype_combos<vec::Accuracy::Estimate>(SoftmaxFloatTypes{});
}

TEST(SoftmaxVectorBoundaryTest, CoversFullVectorsAndTailsForEveryModeAndType) {
  run_all_vector_boundary_cases<vec::Accuracy::Strict>(SoftmaxFloatTypes{});
  run_all_vector_boundary_cases<vec::Accuracy::Fast>(SoftmaxFloatTypes{});
  run_all_vector_boundary_cases<vec::Accuracy::Estimate>(SoftmaxFloatTypes{});
}

TEST(SoftmaxOnlineOptionTest, DefaultsToAllowedAndCanBeDisabled) {
  using Config = SoftmaxConfig<>;
  using DisabledConfig = SoftmaxConfig<vecops::float32_t, vec::Accuracy::Strict, false>;
  constexpr Config default_config{};
  constexpr DisabledConfig disabled_config{};
  static_assert(Config::allow_online);
  static_assert(!DisabledConfig::allow_online);
  EXPECT_TRUE(Config::allow_online);
  EXPECT_FALSE(DisabledConfig::allow_online);

  run_large_row_reference_case(default_config, false, "default-allowed");
  run_large_row_reference_case(default_config, true, "default-allowed");
  run_large_row_reference_case(disabled_config, false, "explicit-disabled");
  run_large_row_reference_case(disabled_config, true, "explicit-disabled");
}

TEST(SoftmaxOnlineOptionTest, MetadataSelectsOnlineConservatively) {
  using EnabledConfig = SoftmaxConfig<>;
  using DisabledConfig = SoftmaxConfig<vecops::float32_t, vec::Accuracy::Strict, false>;
  float value = 0.0f;
#if defined(CPU_CAPABILITY_SVE)
  constexpr nint_t rows = 64;
  constexpr nint_t n = 8192;
#elif defined(CPU_CAPABILITY_AVX512)
  constexpr nint_t rows = 256;
  constexpr nint_t n = 2048;
#else
  constexpr nint_t rows = 1;
  constexpr nint_t n = 8;
#endif
  auto constrained = make_online_test_tensor(&value, rows, n);
  auto constrained_in = input<float>(constrained);
  auto constrained_out = output<float>(constrained);
  const auto enabled = softmax(EnabledConfig{}).required_workspace(constrained_in, constrained_out);
  const auto disabled = softmax(DisabledConfig{}).required_workspace(constrained_in, constrained_out);
#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_AVX512)
  EXPECT_GT(enabled, disabled);
#else
  EXPECT_EQ(enabled, disabled);
#endif

  auto unknown = make_tensor<2>(&value, {rows, n});
  auto unknown_in = input<float>(unknown);
  auto unknown_out = output<float>(unknown);
  EXPECT_EQ(softmax(EnabledConfig{}).required_workspace(unknown_in, unknown_out),
            softmax(DisabledConfig{}).required_workspace(unknown_in, unknown_out));
}

TEST(SoftmaxOnlineOptionTest, LegacyConfigWithoutOptionRemainsCompatible) {
  constexpr LegacySoftmaxConfig config{};
  static_assert(vecops::ops::softmax_details::softmax_online_allowed(config));
  run_large_row_reference_case(config, false, "legacy-default-allowed");
}

TEST(SoftmaxOnlineOptionTest, PreservesRegularNonFiniteSemantics) {
  run_online_nonfinite_equivalence_case(false);
  run_online_nonfinite_equivalence_case(true);
}

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
TEST(SoftmaxSveBf16PackedOutputTest, CoversLargeMultiRowTensorWithTailAndGuards) {
  using ComputeT = vecops::float32_t;
  using Tag = vec::ScalableTag<ComputeT, 0>;
  using Config = SoftmaxConfig<ComputeT, vec::Accuracy::Strict>;

  const nint_t vl = vec::size(Tag{});
  const nint_t n = 512 * vl + 3;
  const nint_t rows = ceil_div(nint_t{512 * 1024}, n);
  const nint_t total = rows * n;
  ASSERT_GE(total, 512 * 1024);
  ASSERT_LE(total, 8 * 1024 * 1024);
  ASSERT_GE(n, 8192);
  ASSERT_LE(n, 16384);
  ASSERT_NE(n % vl, 0);

  std::vector<float> x(static_cast<size_t>(total));
  std::vector<vecops::bfloat16_t> out(static_cast<size_t>(total + 2), static_cast<vecops::bfloat16_t>(-17.0f));
  std::vector<double> x_ref(static_cast<size_t>(total));
  std::vector<double> ref(static_cast<size_t>(total));
  for (nint_t row = 0; row < rows; ++row) {
    const nint_t multiplier = row % 13 + 1;
    for (nint_t col = 0; col < n; ++col) {
      const nint_t index = row * n + col;
      const float value = -7.0f + 0.25f * float((col * multiplier + 3 * row) % 29);
      x[static_cast<size_t>(index)] = value;
      x_ref[static_cast<size_t>(index)] = value;
    }
  }
  reference_softmax_rows(x_ref, rows, n, ref);

  auto x_t = make_online_test_tensor(x.data(), rows, n);
  auto y_t = make_online_test_tensor(out.data() + 1, rows, n);
  softmax(Config{})(input<ComputeT>(x_t), output<ComputeT>(y_t));

  EXPECT_EQ(static_cast<float>(out.front()), -17.0f);
  EXPECT_EQ(static_cast<float>(out.back()), -17.0f);
  const double tol = tolerance<vec::Accuracy::Strict, vecops::bfloat16_t>();
  for (nint_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      const nint_t index = row * n + col;
      const double actual = static_cast<double>(out[static_cast<size_t>(index + 1)]);
      EXPECT_NEAR(ref[static_cast<size_t>(index)], actual, tol) << "row=" << row << " col=" << col << " n=" << n;
      sum += actual;
    }
    EXPECT_NEAR(sum, 1.0, 1e-2) << "row=" << row << " n=" << n;
  }
}
#endif

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
  run_shift_stability_case<vec::Accuracy::Strict>();
  run_shift_stability_case<vec::Accuracy::Fast>();
  run_shift_stability_case<vec::Accuracy::Estimate>();
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

TEST(SoftmaxWorkspaceTest, SpecWorkspaceHandlesStridedRows) {
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
  EXPECT_GT(spec_bytes, 0);
  Workspace workspace(spec_bytes);
  auto view = workspace.view();
  op(view, x_spec, y_spec);

  for (nint_t row = 0; row < 2; ++row) {
    for (nint_t col = 0; col < 7; ++col) {
      EXPECT_NEAR(ref[static_cast<size_t>(row * 7 + col)], out[static_cast<size_t>(row + col * 2)], 3e-5);
    }
  }
}

TEST(SoftmaxWorkspaceTest, MixedStorageUsesComputeCacheSize) {
  using ComputeT = vecops::float64_t;
  using Config = SoftmaxConfig<ComputeT, vec::Accuracy::Strict>;
  const nint_t vl = vec::size(vec::ScalableTag<ComputeT, 0>{});
  const nint_t n = 1024 * vl + 3;
  const nint_t rows = ceil_div(nint_t{1024 * 1024}, n);
  std::vector<float> x(static_cast<size_t>(rows * n));
  std::vector<float> out(static_cast<size_t>(rows * n));
  auto x_t = make_tensor<2>(x.data(), {rows, n});
  auto y_t = make_tensor<2>(out.data(), {rows, n});
  auto x_spec = input<ComputeT>(x_t);
  auto y_spec = output<ComputeT>(y_t);
  auto op = softmax(Config{});
  EXPECT_GE(op.required_workspace(x_spec, y_spec), n * static_cast<nint_t>(sizeof(ComputeT)));
}

TEST(SoftmaxFusionTest, AppliesInputPrologueAndOutputEpilogue) {
  constexpr nint_t n = 7;
  std::vector<float> x{-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  std::vector<float> out(n);
  std::vector<double> transformed(n);
  std::vector<double> ref(n);
  for (nint_t i = 0; i < n; ++i)
    transformed[static_cast<size_t>(i)] = 2.0 * x[static_cast<size_t>(i)];
  reference_softmax_rows(transformed, 1, n, ref);
  auto x_t = make_tensor<1>(x.data(), {n});
  auto y_t = make_tensor<1>(out.data(), {n});
  softmax()(input<vecops::float32_t>(x_t, DoubleTransform{}), output<vecops::float32_t>(y_t, HalfTransform{}));
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

TEST(SoftmaxCompositionTest, AcceptsRawTensorsAndBoundRowAccesses) {
  constexpr nint_t n = 9;
  std::array<float, n> x{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 1.5f, 0.25f, -0.75f, 4.0f};
  std::array<float, n> tensor_out{};
  std::array<float, n> bound_out{};
  auto x_t = make_tensor(x.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto tensor_y_t = make_tensor(tensor_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto bound_y_t = make_tensor(bound_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto op = softmax(SoftmaxConfig<float32_t>{});

  EXPECT_GT(op.required_workspace(x_t, tensor_y_t), 0);
  op(x_t, tensor_y_t);

  auto x_spec = input<float32_t>(x_t);
  auto y_spec = output<float32_t>(bound_y_t);
  kernel::Workspace access_storage(n * static_cast<nint_t>(sizeof(float32_t)) + vec::DEFAULT_ALIGNMENT);
  auto workspace = access_storage.view();
  auto* exp_cache = workspace.allocate<float32_t>(n);
  auto x_access = bind(x_spec, InputAccessPolicy<0, 2, AccessPlan::direct>{}, workspace);
  auto y_access = bind(y_spec, OutputAccessPolicy<0, AccessPlan::direct>{}, workspace);
  op.run_bound(exp_cache, x_access, y_access);

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(tensor_out[static_cast<size_t>(i)], bound_out[static_cast<size_t>(i)], 2e-6f);
  }
}

TEST(SoftmaxCompositionTest, ExecutesThroughExecutionSessionAndActiveRegion) {
  constexpr nint_t n = 7;
  std::array<float, n> x{-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
  std::array<float, n> direct_out{};
  std::array<float, n> region_out{};
  auto x_t = make_tensor(x.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto direct_t = make_tensor(direct_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto region_t = make_tensor(region_out.data(), make_shape(cint<n>), make_strides(cint<1>));
  auto op = softmax();
  static_assert(requires { typename decltype(op)::ResourceRequirements; });

  const nint_t bytes = op.required_workspace(x_t, direct_t);
  Workspace storage(bytes);
  auto workspace = storage.view();
  ExecutionSession execution{workspace};
  op(execution, x_t, direct_t);
  execution.with_region(op, [&](auto& region) VECOPS_INLINE_LAMBDA { op(region, x_t, region_t); });

  for (nint_t i = 0; i < n; ++i) {
    EXPECT_NEAR(direct_out[static_cast<size_t>(i)], region_out[static_cast<size_t>(i)], 2e-6f);
  }
}
