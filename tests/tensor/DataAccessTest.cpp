#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
    requires vec::is_scalable_tag<TransformTag> &&
             (vec::scale_power<TransformTag> == 2)
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

#ifdef VECOPS_DEBUG
TEST(TensorDataAccessDeathTest, MaterializedOutputMustBeCommitted) {
  EXPECT_DEATH(
      ([] {
        std::array<float, 3 * 16> values{};
        auto spec = output<float32_t>(
            make_tensor<2>(values.data(), {3, 16}, {1, 3}));
        using Policy = OutputAccessPolicy<1>;
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

} // namespace
