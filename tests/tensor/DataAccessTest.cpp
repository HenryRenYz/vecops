#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "vecops/kernel/Loop.h"
#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

namespace {

using Tag = vec::ScalableTag<float32_t, 0>;
using InPolicy = InputAccessPolicy<1, 1, AccessPlan::direct>;
using OutPolicy = OutputAccessPolicy<1, AccessPlan::direct>;

struct DoubleP2CoordinateTransform {
  using TIn = float64_t;
  using TOut = float64_t;
  static constexpr bool is_elementwise = false;
  static constexpr bool permutation_equivariant = false;
  static constexpr bool reads_input = true;

  template <vec::VectorTag TransformTag, TransformContextLike Context>
    requires vec::is_scalable_tag_v<TransformTag> &&
             (vec::scale_power_v<TransformTag> == 2)
  vec::Vec<TransformTag> operator()(
      TransformTag tag,
      vec::Vec<TransformTag> input,
      const Context& context) const {
    auto result = vec::zeros(tag);
    for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
      result = vec::set(
          tag, result, lane,
          vec::get(tag, input, lane) +
              static_cast<float64_t>(context.lane_coord(lane)[0] % 100));
    }
    return result;
  }
};

#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
static_assert(!tensor::details::can_transform_chunk<
              DoubleP2CoordinateTransform,
              vec::ScalableTag<int8_t, 2>, TransformContext<1>>());
#endif

template <typename Tensor>
concept CanMakeOutputSpec = requires(Tensor tensor) {
  output<float32_t>(tensor);
};

template <typename Access>
concept HasCommit = requires(Access& access) { access.commit(); };

template <typename T>
concept CanNormalizeInput = requires(T value) {
  as_input_spec<float32_t>(value);
};

template <typename T>
concept CanTransposeView = requires(const T& value) {
  transpose_view<0, 1>(value);
};

static_assert(CanMakeOutputSpec<decltype(make_tensor<1>(
    static_cast<float*>(nullptr), {1}))>);
static_assert(!CanMakeOutputSpec<decltype(make_tensor<1>(
    static_cast<const float*>(nullptr), {1}))>);

TEST(TensorDataAccessTest, OperandSpecsRetainValidatedExternalFacts) {
  alignas(64) std::array<float, 64> values{};
  auto tensor = make_tensor<1>(values.data(), {64});
  auto in = input<float32_t>(tensor, assume_aligned<64>);
  auto out = output<float32_t>(tensor, assume_aligned<64>);
  static_assert(std::tuple_size_v<typename decltype(in)::ExternalFacts> == 1);
  static_assert(std::tuple_size_v<typename decltype(out)::ExternalFacts> == 1);
  EXPECT_EQ(std::tuple_size_v<typename decltype(in)::ExternalFacts>, 1u);
}

TEST(TensorDataAccessTest, NormalizesTensorsAndSpecsWithoutAcceptingAccess) {
  std::array<float, 12> values{};
  auto tensor = make_tensor<2>(values.data(), {3, 4});
  auto from_tensor = as_input_spec<float32_t>(tensor);
  auto from_spec = as_input_spec<float32_t>(from_tensor);
  static_assert(is_input_spec_v<decltype(from_tensor)>);
  static_assert(is_input_spec_v<decltype(from_spec)>);
  EXPECT_EQ(logical_layout(tensor).shape()[0], 3);
  EXPECT_EQ(logical_layout(from_spec).shape()[1], 4);

  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(from_spec, InPolicy{}, workspace);
  static_assert(!CanNormalizeInput<decltype(access)>);
  EXPECT_EQ(logical_layout(access).shape()[0], 3);
}

TEST(TensorDataAccessTest, LoadsAlongAnyLogicalAxis) {
  std::array<float, 4 * 64> values{};
  for (nint_t i = 0; i < static_cast<nint_t>(values.size()); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  auto tensor = make_tensor<2>(values.data(), {4, 64});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace workspace(0);
  auto view = workspace.view();
  auto access = bind(spec, InPolicy{}, view);

  Tag tag;
  const nint_t lanes = vec::size(tag);
  auto row = access.load(
      tag, coord(2, 3), axis<1>, vec::opt::first(lanes));
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, row, lane),
        values[static_cast<std::size_t>(2 * 64 + 3 + lane)]);
  }

  const nint_t active = std::min<nint_t>(4, lanes);
  auto column = access.load(
      tag, coord(0, 3), axis<0>, vec::opt::first(active));
  for (nint_t lane = 0; lane < active; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, column, lane),
        values[static_cast<std::size_t>(lane * 64 + 3)]);
  }
}

TEST(TensorDataAccessTest, LogicalStridedAddressingComposesWithLayout) {
  std::array<float, 128> values{};
  for (nint_t i = 0; i < static_cast<nint_t>(values.size()); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  auto tensor = make_tensor<2>(values.data(), {2, 64});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace workspace(0);
  auto view = workspace.view();
  auto access = bind(spec, InPolicy{}, view);

  Tag tag;
  const nint_t active = std::min<nint_t>(16, vec::size(tag));
  auto result = access.load(
      tag, coord(1, 1), axis<1>, vec::opt::first(active),
      vec::strided(2));
  for (nint_t lane = 0; lane < active; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, result, lane),
        values[static_cast<std::size_t>(64 + 1 + lane * 2)]);
  }
}

TEST(TensorDataAccessTest, IndexedMaskedLoadUsesLogicalTensorOffsets) {
  Tag tag{};
  using IndexTag = vec::Rebind<int32_t, Tag>;
  const nint_t lanes = vec::size(tag);
  const nint_t columns = 2 * lanes + 1;
  std::vector<float> values(static_cast<std::size_t>(4 * columns));
  for (nint_t column = 0; column < columns; ++column) {
    for (nint_t row = 0; row < 4; ++row) {
      values[static_cast<std::size_t>(row + column * 4)] =
          static_cast<float>(1000 * row + column);
    }
  }
  auto indices = vec::zeros(IndexTag{});
  auto mask = vec::mfalse(tag);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    indices = vec::set(
        IndexTag{}, indices, lane, static_cast<int32_t>(2 * lane));
    mask = vec::set(tag, mask, lane, lane % 2 == 0);
  }
  auto tensor = make_tensor<2>(
      values.data(), {4, columns}, {1, 4});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(spec, InPolicy{}, workspace);
  auto result = access.load(
      tag, coord(2, 0), axis<1>, vec::opt::masked(mask),
      vec::opt::merge(99.0f), vec::indexed(indices));
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, result, lane),
        lane % 2 == 0 ? static_cast<float>(2000 + 2 * lane) : 99.0f);
  }
}

TEST(TensorDataAccessTest, StoreHonorsTailAndRequiresMutableTensor) {
  std::array<float, 64> values{};
  auto tensor = make_tensor<1>(values.data(), {64});
  auto spec = output<float32_t>(tensor);
  kernel::Workspace workspace(0);
  auto view = workspace.view();
  auto access = bind(spec, OutputAccessPolicy<0, AccessPlan::direct>{}, view);

  Tag tag;
  auto value = vec::fill(tag, 7.0f);
  const nint_t active = std::min<nint_t>(5, vec::size(tag));
  access.store(tag, coord(3), value, vec::opt::first(active));
  access.commit();
  for (nint_t i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(values[static_cast<std::size_t>(i)],
                    3 <= i && i < 3 + active ? 7.0f : 0.0f);
  }
}

TEST(TensorDataAccessTest, ScanCursorSupportsSmallerTailTag) {
  std::array<float, 128> values{};
  for (nint_t i = 0; i < static_cast<nint_t>(values.size()); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
  }
  auto tensor = make_tensor<1>(values.data(), {128});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace workspace(0);
  auto view = workspace.view();
  auto access = bind(
      spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, view);

  Tag tag;
  using TailTag = vec::Half<Tag>;
  auto cursor = access.scan(tag, coord(0), axis<0>, vec::size(tag) + 3);
  ASSERT_TRUE(cursor.has_full());
  auto full = cursor.load_full();
  EXPECT_FLOAT_EQ(vec::get(tag, full, 0), 1.0f);
  cursor.advance_full();
  auto tail = cursor.load_tail(TailTag{});
  EXPECT_FLOAT_EQ(
      vec::get(TailTag{}, tail, 0),
      static_cast<float>(vec::size(tag) + 1));
  cursor.advance_tail(TailTag{});
  while (!cursor.empty()) {
    (void)cursor.load_tail(TailTag{});
    cursor.advance_tail(TailTag{});
  }
  EXPECT_TRUE(cursor.empty());
}

TEST(TensorDataAccessTest, ScanCursorAcceptsLogicalStridedOption) {
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  std::vector<float> values(static_cast<std::size_t>(4 * lanes));
  for (nint_t i = 0; i < 4 * lanes; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  auto tensor = make_tensor<1>(values.data(), {4 * lanes});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(
      spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, workspace);
  auto cursor = access.scan(
      tag, coord(0), axis<0>, 2 * lanes, vec::strided(2));
  auto first = cursor.load_full();
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(vec::get(tag, first, lane), static_cast<float>(2 * lane));
  }
  cursor.advance_full();
  auto second = cursor.load_full();
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_FLOAT_EQ(
        vec::get(tag, second, lane),
        static_cast<float>(2 * (lanes + lane)));
  }
}

TEST(TensorDataAccessTest, ProjectCursorAdvancesTraversalAxis) {
  std::array<float, 8 * 64> values{};
  for (nint_t i = 0; i < static_cast<nint_t>(values.size()); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  auto tensor = make_tensor<2>(values.data(), {8, 64});
  auto spec = input<float32_t>(tensor);
  kernel::Workspace workspace(0);
  auto view = workspace.view();
  auto access = bind(spec, InPolicy{}, view);

  Tag tag;
  auto cursor = access.project(
      tag, coord(2, 0), traversal_axis<0>, vector_axis<1>, 3);
  for (nint_t row = 2; row < 5; ++row) {
    ASSERT_TRUE(cursor.valid());
    auto value = cursor.load(vec::opt::first(1));
    EXPECT_FLOAT_EQ(vec::get(tag, value, 0), static_cast<float>(row * 64));
    cursor.advance();
  }
  EXPECT_FALSE(cursor.valid());
}

TEST(TensorDataAccessTest, SlicedDataAccessIsBorrowedAndKeepsCoordinates) {
  std::array<float, 4 * 64> input_values{};
  std::array<float, 4 * 64> output_values{};
  for (nint_t i = 0; i < static_cast<nint_t>(input_values.size()); ++i) {
    input_values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  auto in_spec = input<float32_t>(
      make_tensor<2>(input_values.data(), {4, 64}));
  auto out_spec = output<float32_t>(
      make_tensor<2>(output_values.data(), {4, 64}));
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto x = bind(in_spec, InPolicy{}, workspace);
  auto y = bind(out_spec, OutPolicy{}, workspace);
  auto x_row = slice_view<0>(x, 2);
  auto y_row = slice_view<0>(y, 2);
  static_assert(!HasCommit<decltype(y_row)>);
  Tag tag{};
  const nint_t active = std::min<nint_t>(7, vec::size(tag));
  auto value = x_row.load(
      tag, coord(3), vec::opt::first(active));
  y_row.store(tag, coord(3), value, vec::opt::first(active));
  y.commit();
  for (nint_t lane = 0; lane < active; ++lane) {
    EXPECT_FLOAT_EQ(
        output_values[static_cast<std::size_t>(2 * 64 + 3 + lane)],
        input_values[static_cast<std::size_t>(2 * 64 + 3 + lane)]);
  }
}

TEST(TensorDataAccessTest, TakeSpecsKeepZeroedOriginalCoordinates) {
  std::array<float64_t, 2 * 3 * 64> input_values{};
  std::array<float64_t, 2 * 3 * 64> output_values{};
  auto transform = [](auto tag, auto value, const auto& context) {
    const auto logical = context.lane_coord(0);
    return vec::add(
        value, vec::fill(
                   tag, logical[0] * 100 + logical[1] * 10 +
                       logical[2]));
  };
  auto in = input<float64_t>(
      make_tensor<3>(input_values.data(), {2, 3, 64}), transform);
  auto out = output<float64_t>(
      make_tensor<3>(output_values.data(), {2, 3, 64}), transform);
  auto in_trailing = take_trailing<1>(in);
  auto out_leading = take_leading<2>(out);
  using InTakePolicy = InputAccessPolicy<0, 1, AccessPlan::direct>;
  using OutTakePolicy = OutputAccessPolicy<1, AccessPlan::direct>;
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto x = bind(in_trailing, InTakePolicy{}, workspace);
  auto y = bind(out_leading, OutTakePolicy{}, workspace);
  vec::ScalableTag<float64_t, 0> tag{};

  auto loaded = x.load(tag, coord(7), vec::opt::first(1));
  EXPECT_DOUBLE_EQ(vec::get(tag, loaded, 0), 7.0);
  y.store(tag, coord(1, 2), vec::fill(tag, 3.0), vec::opt::first(1));
  EXPECT_DOUBLE_EQ(output_values[static_cast<std::size_t>(1 * 3 * 64 + 2 * 64)],
                   123.0);
}

TEST(TensorDataAccessTest, TransposeViewsPreserveCoordinatesAndPolicyAxis) {
  constexpr nint_t rows = 3;
  constexpr nint_t columns = 64;
  std::array<float, rows * columns> values{};
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t column = 0; column < columns; ++column) {
      values[static_cast<size_t>(row * columns + column)] =
          static_cast<float>(row * columns + column);
    }
  }
  auto transform = [](auto tag, auto value, const auto& context) {
    const auto original = context.lane_coord(0);
    return vec::add(
        value, vec::fill(tag, static_cast<float>(
                              original[0] * 100 + original[1])));
  };
  auto spec = input<float32_t>(
      make_tensor<2>(values.data(), {rows, columns}), transform);
  auto transposed_spec = transpose_view<0, 1>(spec);
  static_assert(decltype(transposed_spec)::InputTensor::Ndim == 2);
  EXPECT_EQ(transposed_spec.input_layout().shape()[0], columns);
  EXPECT_EQ(transposed_spec.input_layout().shape()[1], rows);

  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto original_access = bind(spec, InPolicy{}, workspace);
  auto transposed_access = transpose_view<0, 1>(original_access);
  static_assert(decltype(transposed_access)::Rank == 2);
  static_assert(
      std::remove_cvref_t<decltype(transposed_access.policy())>::vector_axis ==
      0);
  EXPECT_EQ(logical_layout(transposed_access).shape()[0], columns);

  Tag tag{};
  const nint_t active = std::min<nint_t>(7, vec::size(tag));
  auto loaded = transposed_access.load(
      tag, coord(0, 1), axis<0>, vec::opt::first(active));
  for (nint_t lane = 0; lane < active; ++lane) {
    const float source = static_cast<float>(columns + lane);
    const float transform_value = 100.0f;
    EXPECT_FLOAT_EQ(
        vec::get(tag, loaded, lane), source + transform_value);
  }
}

TEST(TensorDataAccessTest, AutomaticInputMaterializesAfterTransformForMultiplePasses) {
  std::array<float, 3 * 16> values{};
  for (nint_t row = 0; row < 3; ++row) {
    for (nint_t column = 0; column < 16; ++column) {
      values[static_cast<std::size_t>(row + column * 3)] =
          static_cast<float>(100 * row + column);
    }
  }
  auto tensor = make_tensor<2>(values.data(), {3, 16}, {1, 3});
  auto spec = input<float32_t>(tensor);
  using Policy = InputAccessPolicy<1, 2>;
  const nint_t bytes = required_workspace(spec, Policy{});
  EXPECT_GT(bytes, 0);
  kernel::Workspace storage(bytes);
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& access) {
        Tag tag{};
        const nint_t active = std::min<nint_t>(7, vec::size(tag));
        auto value = access.load(
            tag, coord(2, 1), axis<1>, vec::opt::first(active));
        for (nint_t lane = 0; lane < active; ++lane) {
          EXPECT_FLOAT_EQ(vec::get(tag, value, lane),
                          static_cast<float>(201 + lane));
        }
      });
}

TEST(TensorDataAccessTest, ForcedBeforeTransformPlansPreserveBoundaryPipeline) {
  std::array<float64_t, 3 * 16> input_values{};
  std::array<float64_t, 3 * 16> output_values{};
  for (nint_t row = 0; row < 3; ++row) {
    for (nint_t column = 0; column < 16; ++column) {
      input_values[static_cast<std::size_t>(row + column * 3)] =
          static_cast<float64_t>(10 * row + column);
    }
  }
  auto transform = make_elementwise_vec_transform<float64_t, float64_t>(
      []<vec::VectorTag TransformTag>(
          TransformTag tag, vec::Vec<TransformTag> value) {
        return vec::add(value, vec::fill(tag, 1.0));
      });
  auto in_spec = input<float32_t>(
      make_tensor<2>(input_values.data(), {3, 16}, {1, 3}), transform);
  auto out_spec = output<float32_t>(
      make_tensor<2>(output_values.data(), {3, 16}, {1, 3}), transform);
  using InBefore = InputAccessPolicy<
      1, 2, AccessPlan::materialize_before_transform>;
  using OutBefore = OutputAccessPolicy<
      1, AccessPlan::materialize_before_transform>;
  kernel::Workspace storage(
      required_workspace(in_spec, InBefore{}) +
      required_workspace(out_spec, OutBefore{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(in_spec, InBefore{}),
      operand(out_spec, OutBefore{}), [&](auto& x, auto& y) {
        Tag tag{};
        const nint_t active = std::min<nint_t>(11, vec::size(tag));
        auto value = x.load(
            tag, coord(2, 1), axis<1>, vec::opt::first(active));
        y.store(
            tag, coord(1, 2), axis<1>, value,
            vec::opt::first(active));
        y.commit();
      });
  const nint_t active = std::min<nint_t>(11, vec::size(Tag{}));
  for (nint_t lane = 0; lane < active; ++lane) {
    EXPECT_DOUBLE_EQ(
        output_values[static_cast<std::size_t>(1 + (2 + lane) * 3)],
        static_cast<float64_t>(23 + lane));
  }
}

TEST(TensorDataAccessTest, ReadlessTransformNeedsNoSourceOrWorkspace) {
  auto source = make_tensor<1>(static_cast<const float*>(nullptr), {64});
  auto spec = input<float32_t>(
      source, ZeroVecTransform<float32_t, float32_t>{});
  using Policy = InputAccessPolicy<0, 4>;
  EXPECT_EQ(required_workspace(spec, Policy{}), 0);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [](auto& x) {
        Tag tag{};
        auto value = x.load(
            tag, coord(0), vec::opt::first(vec::size(tag)));
        for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
          EXPECT_FLOAT_EQ(vec::get(tag, value, lane), 0.0f);
        }
      });
}

TEST(TensorDataAccessTest, OperandScopeIsElidedOnlyWhenEveryBindingIsDirect) {
  std::array<float, 128> values{};
  values[0] = 7.0f;
  auto direct_spec = input<float32_t>(
      make_tensor<1>(values.data(), {64}));
  auto materialized_spec = input<float32_t>(
      make_tensor<1>(values.data(), {64}, {2}));
  using Policy = InputAccessPolicy<0, 2, AccessPlan::automatic>;
  auto direct_binding = operand(direct_spec, Policy{});
  auto materialized_binding = operand(materialized_spec, Policy{});
  static_assert(!kernel::details::operands_need_workspace_scope_v<
                decltype(direct_binding)>);
  static_assert(kernel::details::operands_need_workspace_scope_v<
                decltype(materialized_binding)>);

  kernel::Workspace direct_storage(0);
  auto direct_workspace = direct_storage.view();
  const auto direct_result = kernel::with_operand_tuple(
      direct_workspace, std::tuple{direct_binding}, [](auto& access) {
        const auto value = access.load(
            Tag{}, coord(0), vec::opt::first(1));
        return vec::get(Tag{}, value, 0);
      });
  EXPECT_FLOAT_EQ(direct_result, 7.0f);
  EXPECT_EQ(direct_workspace.used(), 0);
  EXPECT_EQ(direct_workspace.high_watermark(), 0);

  const nint_t bytes = required_workspace(materialized_spec, Policy{});
  kernel::Workspace materialized_storage(bytes + vec::DEFAULT_ALIGNMENT);
  auto materialized_workspace = materialized_storage.view();
  (void)materialized_workspace.allocate<std::byte>(1);
  const nint_t entry_offset = materialized_workspace.used();
  const auto materialized_result = kernel::with_operands(
      materialized_workspace, materialized_binding, [](auto& access) {
        const auto value = access.load(
            Tag{}, coord(0), vec::opt::first(1));
        return vec::get(Tag{}, value, 0);
      });
  EXPECT_FLOAT_EQ(materialized_result, 7.0f);
  EXPECT_EQ(materialized_workspace.used(), entry_offset);
  EXPECT_GT(materialized_workspace.high_watermark(), entry_offset);
}

TEST(TensorDataAccessTest, MaterializedOutputRequiresExplicitCommitAndWritesLayout) {
  std::array<float, 3 * 16> values{};
  // The commit axis must promise unit stride at compile time: plans are
  // resolved from meta stride types, and a runtime-only unit axis is treated
  // as non-unit (direct scatter).
  auto tensor = make_tensor(
      values.data(), make_shape(3, 16), make_strides(cint<1>, cint<3>));
  auto spec = output<float32_t>(tensor);
  using Policy = OutputAccessPolicy<1>;
  const nint_t bytes = required_workspace(spec, Policy{});
  EXPECT_GT(bytes, 0);
  kernel::Workspace storage(bytes);
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& access) {
        Tag tag{};
        const nint_t active = std::min<nint_t>(9, vec::size(tag));
        access.store(
            tag, coord(1, 2), axis<1>, vec::fill(tag, 13.0f),
            vec::opt::first(active));
        access.commit();
        access.commit();
      });
  const nint_t active = std::min<nint_t>(9, vec::size(Tag{}));
  for (nint_t column = 0; column < 16; ++column) {
    EXPECT_FLOAT_EQ(values[static_cast<std::size_t>(1 + column * 3)],
                    2 <= column && column < 2 + active ? 13.0f : 0.0f);
  }
}

TEST(TensorDataAccessTest, LoopSlicesMaterializedOutputAsBorrowedViews) {
  std::array<float, 3 * 16> values{};
  auto spec = output<float32_t>(
      make_tensor<2>(values.data(), {3, 16}, {1, 3}));
  using Policy = OutputAccessPolicy<1>;
  kernel::Workspace storage(required_workspace(spec, Policy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& y) {
        kernel::loop::for_each_dims_with_index<1>(
            [](nint_t row, auto y_row) {
              static_assert(!HasCommit<decltype(y_row)>);
              Tag tag{};
              for (nint_t column = 0; column < 16;
                   column += vec::size(tag)) {
                const nint_t active =
                    std::min<nint_t>(16 - column, vec::size(tag));
                y_row.store(
                    tag, coord(column),
                    vec::fill(tag, static_cast<float>(row + 1)),
                    vec::opt::first(active));
              }
            },
            y);
        y.commit();
      });
  for (nint_t row = 0; row < 3; ++row) {
    for (nint_t column = 0; column < 16; ++column) {
      EXPECT_FLOAT_EQ(
          values[static_cast<std::size_t>(row + column * 3)],
          static_cast<float>(row + 1));
    }
  }
}

TEST(TensorDataAccessTest, TransposesCanonicalInputAndMaterializedOutputViews) {
  constexpr nint_t rows = 3;
  constexpr nint_t columns = 16;
  std::array<float, rows * columns> input_values{};
  std::array<float, rows * columns> output_values{};
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t column = 0; column < columns; ++column) {
      input_values[static_cast<size_t>(row + column * rows)] =
          static_cast<float>(100 * row + column);
    }
  }
  auto in_spec = input<float32_t>(make_tensor(
      input_values.data(), make_shape(cint<rows>, cint<columns>),
      make_strides(cint<1>, cint<rows>)));
  auto out_spec = output<float32_t>(make_tensor(
      output_values.data(), make_shape(cint<rows>, cint<columns>),
      make_strides(cint<1>, cint<rows>)));
  using InMaterialize = InputAccessPolicy<1, 2, AccessPlan::automatic>;
  using OutMaterialize = OutputAccessPolicy<1, AccessPlan::automatic>;
  kernel::Workspace storage(
      required_workspace(in_spec, InMaterialize{}) +
      required_workspace(out_spec, OutMaterialize{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(in_spec, InMaterialize{}),
      operand(out_spec, OutMaterialize{}), [&](auto& x, auto& y) {
        auto tx = transpose_view<0, 1>(x);
        auto ty = transpose_view<0, 1>(y);
        static_assert(!HasCommit<decltype(ty)>);
        static_assert(
            std::remove_cvref_t<decltype(tx.policy())>::vector_axis == 0);
        Tag tag{};
        const nint_t active = std::min<nint_t>(7, vec::size(tag));
        auto value = tx.load(
            tag, coord(0, 2), axis<0>, vec::opt::first(active));
        ty.store(tag, coord(0, 2), axis<0>, value,
                 vec::opt::first(active));
        y.commit();
      });
  const nint_t active = std::min<nint_t>(7, vec::size(Tag{}));
  for (nint_t column = 0; column < columns; ++column) {
    EXPECT_FLOAT_EQ(
        output_values[static_cast<size_t>(2 + column * rows)],
        column < active ? static_cast<float>(200 + column) : 0.0f);
  }
}

#ifdef VECOPS_DEBUG
TEST(TensorDataAccessDeathTest, MaterializedOutputMustBeCommitted) {
  EXPECT_DEATH(
      ([] {
        std::array<float, 3 * 16> values{};
        auto spec = output<float32_t>(
            make_tensor<2>(values.data(), {3, 16}, {1, 3}));
        using Policy = OutputAccessPolicy<
            1, AccessPlan::materialize_after_transform>;
        kernel::Workspace storage(required_workspace(spec, Policy{}));
        auto workspace = storage.view();
        kernel::with_operands(
            workspace, operand(spec, Policy{}), [](auto&) {});
      }()),
      "without commit");
}
#endif

TEST(TensorDataAccessTest, InputSplitsTransformBeyondInternalMaximumTag) {
  using RequestedTag = vec::ScalableTag<int8_t, 2>;
  RequestedTag tag{};
  const nint_t lanes = vec::size(tag);
  std::vector<float64_t> values(static_cast<std::size_t>(lanes), 1.0);
  auto tensor = make_tensor<1>(values.data(), {lanes});
  auto spec = input<int8_t>(tensor, DoubleP2CoordinateTransform{});
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(
      spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, workspace);

  auto result = access.load(tag, coord(0), vec::opt::unmasked);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(
        vec::get(tag, result, lane),
        static_cast<int8_t>(1 + lane % 100));
  }
}

TEST(TensorDataAccessTest, OutputSplitsTransformBeyondInternalMaximumTag) {
  using RequestedTag = vec::ScalableTag<int8_t, 2>;
  RequestedTag tag{};
  const nint_t lanes = vec::size(tag);
  std::vector<float64_t> values(static_cast<std::size_t>(lanes), -1.0);
  auto tensor = make_tensor<1>(values.data(), {lanes});
  auto spec = output<int8_t>(tensor, DoubleP2CoordinateTransform{});
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(
      spec, OutputAccessPolicy<0, AccessPlan::direct>{}, workspace);

  access.store(
      tag, coord(0), vec::fill(tag, static_cast<int8_t>(2)),
      vec::opt::unmasked);
  access.commit();
  for (nint_t lane = 0; lane < lanes; ++lane) {
    EXPECT_DOUBLE_EQ(
        values[static_cast<std::size_t>(lane)],
        static_cast<float64_t>(2 + lane % 100));
  }
}

TEST(TensorDataAccessTest, TransformRemapsRequestedMaskAcrossDtypes) {
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  std::vector<float64_t> input_values(static_cast<std::size_t>(lanes));
  std::vector<float64_t> output_values(
      static_cast<std::size_t>(lanes), -1.0);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    input_values[static_cast<std::size_t>(lane)] = lane + 0.5;
  }
  auto transform = make_elementwise_vec_transform<float64_t, float64_t>(
      []<vec::VectorTag TransformTag>(
          TransformTag transform_tag, vec::Vec<TransformTag> value) {
        return vec::add(value, vec::fill(transform_tag, 1.0));
      });
  auto in_spec = input<float32_t>(
      make_tensor<1>(input_values.data(), {lanes}), transform);
  auto out_spec = output<float32_t>(
      make_tensor<1>(output_values.data(), {lanes}), transform);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto x = bind(
      in_spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, workspace);
  auto y = bind(
      out_spec, OutputAccessPolicy<0, AccessPlan::direct>{}, workspace);
  auto mask = vec::mfalse(tag);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    mask = vec::set(tag, mask, lane, lane % 2 == 0);
  }

  auto loaded = x.load(
      tag, coord(0), vec::opt::masked(mask), vec::opt::merge(77.0f));
  y.store(tag, coord(0), loaded, vec::opt::masked(mask));
  y.commit();
  for (nint_t lane = 0; lane < lanes; ++lane) {
    if (lane % 2 == 0) {
      EXPECT_FLOAT_EQ(
          vec::get(tag, loaded, lane), static_cast<float>(lane + 1.5));
      EXPECT_DOUBLE_EQ(
          output_values[static_cast<std::size_t>(lane)], lane + 2.5);
    } else {
      EXPECT_FLOAT_EQ(vec::get(tag, loaded, lane), 77.0f);
      EXPECT_DOUBLE_EQ(
          output_values[static_cast<std::size_t>(lane)], -1.0);
    }
  }
}

TEST(TensorDataAccessTest, AutomaticDeferredPopulatesAndReusesCanonicalValues) {
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  const nint_t rows = 2;
  const nint_t columns = 2 * lanes + 3;
  const nint_t row_stride = 2 * columns + 5;
  std::vector<float> values(
      static_cast<std::size_t>(rows * row_stride), -1.0f);
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t column = 0; column < columns; ++column) {
      values[static_cast<std::size_t>(row * row_stride + 2 * column)] =
          static_cast<float>(100 * row + column);
    }
  }

  auto spec = input<float32_t>(make_tensor<2>(
      values.data(), {rows, columns}, {row_stride, 2}));
  using Policy = InputAccessPolicy<
      1, 2, AccessPlan::automatic_deferred>;
  kernel::Workspace storage(required_workspace(spec, Policy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& access) {
        static_assert(!CanTransposeView<std::remove_cvref_t<decltype(access)>>);
        with_unordered_access(access, [&](auto deferred) {
          static_assert(!decltype(deferred)::is_unordered);
          for (nint_t row = 0; row < rows; ++row) {
            for (nint_t column = 0; column < columns; column += lanes) {
              const nint_t active = std::min(columns - column, lanes);
              auto value = deferred.load(
                  tag, coord(row, column), axis<1>, vec::opt::first(active),
                  materialize::populate);
              for (nint_t lane = 0; lane < active; ++lane) {
                EXPECT_FLOAT_EQ(
                    vec::get(tag, value, lane),
                    static_cast<float>(100 * row + column + lane));
              }
            }
          }
        });

        std::fill(values.begin(), values.end(), -99.0f);
        for (nint_t row = 0; row < rows; ++row) {
          for (nint_t column = 0; column < columns; column += lanes) {
            const nint_t active = std::min(columns - column, lanes);
            auto value = access.load(
                tag, coord(row, column), axis<1>, vec::opt::first(active));
            for (nint_t lane = 0; lane < active; ++lane) {
              EXPECT_FLOAT_EQ(
                  vec::get(tag, value, lane),
                  static_cast<float>(100 * row + column + lane));
            }
          }
        }
      });
}

#if defined(VECOPS_DEBUG)
TEST(TensorDataAccessDeathTest, DeferredMaterializationEnforcesPhaseOrder) {
  std::array<float, 64> values{};
  auto spec = input<float32_t>(make_tensor<1>(values.data(), {32}, {2}));
  using Policy = InputAccessPolicy<
      0, 2, AccessPlan::automatic_deferred>;
  kernel::Workspace storage(required_workspace(spec, Policy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& access) {
        EXPECT_DEATH(
            (void)access.load(Tag{}, coord(0), vec::opt::first(1)),
            "reused before population");
        (void)access.load(
            Tag{}, coord(0), vec::opt::first(1), materialize::populate);
        (void)access.load(Tag{}, coord(0), vec::opt::first(1));
        EXPECT_DEATH(
            (void)access.load(
                Tag{}, coord(0), vec::opt::first(1),
                materialize::populate),
            "populated after reuse");
      });
}
#endif

TEST(TensorDataAccessTest, DeferredDirectPopulateIsNoOpAndAllowsUnordered) {
  std::array<vecops::float16_t, 64> values{};
  auto tensor = make_tensor(
      values.data(), make_shape(cint<64>), make_strides(cint<1>));
  auto spec = input<float32_t>(tensor);
  using Policy = InputAccessPolicy<
      0, 2, AccessPlan::automatic_deferred>;
  EXPECT_EQ(required_workspace(spec, Policy{}), 0);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(spec, Policy{}), [&](auto& access) {
        with_unordered_access(access, [&](auto unordered) {
          static_assert(decltype(unordered)::is_unordered);
          (void)unordered.load(
              vec::ScalableTag<float32_t, 0>{}, coord(0),
              vec::opt::first(1), materialize::populate);
        });
      });
}

TEST(TensorDataAccessTest, UnorderedRegionChecksEveryArgument) {
  std::array<vecops::float16_t, 32> direct_values{};
  auto direct_tensor = make_tensor(
      direct_values.data(), make_shape(cint<32>), make_strides(cint<1>));
  auto direct_spec = input<float32_t>(direct_tensor);

  std::array<vecops::float16_t, 64> strided_values{};
  auto strided_spec = input<float32_t>(
      make_tensor<1>(strided_values.data(), {32}, {2}));
  using DirectPolicy = InputAccessPolicy<0, 1, AccessPlan::direct>;
  using EagerPolicy = InputAccessPolicy<0, 2, AccessPlan::automatic>;
  kernel::Workspace storage(required_workspace(strided_spec, EagerPolicy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace, operand(direct_spec, DirectPolicy{}),
      operand(strided_spec, EagerPolicy{}), [&](auto& first, auto& second) {
        with_unordered_access(first, second, [&](auto a, auto b) {
          static_assert(!decltype(a)::is_unordered);
          static_assert(!decltype(b)::is_unordered);
        });
      });
}

TEST(TensorDataAccessTest, UnorderedRegionAcceptsSameWidthMixedStorage) {
  std::array<vecops::float16_t, 32> x_values{};
  std::array<uint16_t, 32> scale_values{};
  std::array<int16_t, 32> out_values{};
  auto layout = Layout{
      make_shape(cint<32>), make_strides(cint<1>)};
  auto x_spec = input<float32_t>(make_tensor(x_values.data(), layout));
  auto scale_spec = input<float32_t>(make_tensor(scale_values.data(), layout));
  auto out_spec = output<float32_t>(make_tensor(out_values.data(), layout));
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto x = bind(x_spec, InputAccessPolicy<0>{}, workspace);
  auto scale = bind(scale_spec, InputAccessPolicy<0>{}, workspace);
  auto out = bind(out_spec, OutputAccessPolicy<0>{}, workspace);
  with_unordered_access(x, scale, out, [&](auto a, auto b, auto c) {
#if defined(CPU_CAPABILITY_SVE)
    // The provenance is compatible, but SVE currently selects unordered only
    // for the measured-profitable fp16<->fp32 pair.
    static_assert(!decltype(a)::is_unordered);
    static_assert(!decltype(b)::is_unordered);
    static_assert(!decltype(c)::is_unordered);
#else
    static_assert(decltype(a)::is_unordered);
    static_assert(decltype(b)::is_unordered);
    static_assert(decltype(c)::is_unordered);
#endif
  });
  out.commit();
}

TEST(TensorDataAccessTest, CallConversionOptionsOverrideOperandDefaults) {
  std::array<int16_t, 64> values{};
  values[0] = 300;
  auto tensor = make_tensor(
      values.data(), make_shape(cint<64>), make_strides(cint<1>));
  auto spec = input<int8_t>(tensor);
  kernel::Workspace storage(0);
  auto workspace = storage.view();
  auto access = bind(spec, InputAccessPolicy<0>{}, workspace);
  using I8Tag = vec::ScalableTag<int8_t, 0>;
  const auto saturated = access.load(
      I8Tag{}, coord(0), vec::opt::first(1));
  const auto wrapped = access.load(
      I8Tag{}, coord(0), vec::opt::first(1), vec::cvt::wrap);
  EXPECT_EQ(vec::get(I8Tag{}, saturated, 0), std::numeric_limits<int8_t>::max());
  EXPECT_EQ(vec::get(I8Tag{}, wrapped, 0), static_cast<int8_t>(300));
}

TEST(TensorDataAccessTest, OperandDefaultsConfigureEagerPreparation) {
  std::array<int16_t, 128> values{};
  values[0] = 300;
  auto spec = input<int8_t>(make_tensor<1>(values.data(), {64}, {2}));
  using Policy = InputAccessPolicy<0, 2, AccessPlan::automatic>;
  kernel::Workspace storage(required_workspace(spec, Policy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace,
      operand(spec, Policy{}, access_defaults(vec::cvt::wrap)),
      [&](auto& access) {
        using I8Tag = vec::ScalableTag<int8_t, 0>;
        const auto value = access.load(
            I8Tag{}, coord(0), vec::opt::first(1));
        EXPECT_EQ(vec::get(I8Tag{}, value, 0), static_cast<int8_t>(300));
      });
}

TEST(TensorDataAccessTest, OperandDefaultsConfigureMaterializedCommit) {
  std::array<int8_t, 64> values{};
  auto tensor = make_tensor(
      values.data(), make_shape(cint<64>), make_strides(cint<1>));
  auto spec = output<int16_t>(tensor);
  using Policy = OutputAccessPolicy<
      0, AccessPlan::materialize_before_transform>;
  kernel::Workspace storage(required_workspace(spec, Policy{}));
  auto workspace = storage.view();
  kernel::with_operands(
      workspace,
      operand(spec, Policy{}, access_defaults(vec::cvt::wrap)),
      [&](auto& access) {
        using I16Tag = vec::ScalableTag<int16_t, 0>;
        access.store(
            I16Tag{}, coord(0), vec::fill(I16Tag{}, int16_t{300}),
            vec::opt::first(1));
        access.commit();
      });
  EXPECT_EQ(values[0], static_cast<int8_t>(300));
}

} // namespace
