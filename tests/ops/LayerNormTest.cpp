#include "LayerNormTestShared.h"

TEST(LayerNormRankTest, CoversRanksOneThroughFour) {
  run_contiguous_rank_case<1>();
  run_contiguous_rank_case<2>();
  run_contiguous_rank_case<3>();
  run_contiguous_rank_case<4>();
}

TEST(LayerNormVectorBoundaryTest, CoversGroupedLoopAndTailBoundaries) {
  using Tag = LayerNormConfig<float32_t>::Tag;
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

template <typename T>
void run_sve_16bit_boundary_cases() {
  using Tag = LayerNormConfig<float32_t>::Tag;
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
    op(input<float32_t>(x_t), input<float32_t>(s_t),
       input<float32_t>(b_t), output<float32_t>(y_t));

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
        const T expected = static_cast<T>(
            (static_cast<float>(x[index]) - mean) * rstd *
                static_cast<float>(scale[static_cast<size_t>(col)]) +
            static_cast<float>(bias[static_cast<size_t>(col)]));
        EXPECT_NEAR(as_double(expected), as_double(out[index]), tolerance<T>())
            << "row=" << row << " col=" << col;
      }
    }
  }
}

TEST(LayerNormVectorBoundaryTest, CoversSVE16BitFullLoadsAndTails) {
  run_sve_16bit_boundary_cases<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  run_sve_16bit_boundary_cases<vecops::bfloat16_t>();
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
