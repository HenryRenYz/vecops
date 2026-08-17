#include <gtest/gtest.h>

#include "vecops/tensor/Transform.h"

using namespace vecops;
using namespace vecops::tensor;

namespace {

using FullTag = vec::ScalableTag<float32_t, 1>;
using HalfTag = vec::Half<FullTag>;

TEST(TensorTransformTest, NoTransformExposesPlannerTraits) {
  static_assert(NoTransform::is_elementwise);
  static_assert(NoTransform::permutation_equivariant);
  static_assert(NoTransform::reads_input);
}

TEST(TensorTransformTest, ElementwiseTransformIgnoresContext) {
  auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
      []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value) {
        return vec::add(value, vec::fill(tag, 2.0f));
      });
  FullTag tag{};
  TransformContext<2> context{coord(3, 7), 1};
  auto result = transform(tag, vec::fill(tag, 4.0f), context);
  for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_FLOAT_EQ(vec::get(tag, result, lane), 6.0f);
  }
  static_assert(decltype(transform)::is_elementwise);
  static_assert(decltype(transform)::permutation_equivariant);
}

struct HalfOnlyCoordinateTransform {
  template <vec::VectorTag Tag, TransformContextLike Context>
    requires (vec::scale_power<Tag> == vec::scale_power<HalfTag>)
  vec::Vec<Tag> operator()(
      Tag tag, vec::Vec<Tag>, const Context& context) const {
    auto result = vec::zeros(tag);
    for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
      result = vec::set(
          tag, result, lane,
          static_cast<float>(context.lane_coord(lane)[1]));
    }
    return result;
  }
};

TEST(TensorTransformTest, SplitUsesSubspanForHighHalf) {
  LambdaVecTransform<float32_t, float32_t,
                     HalfOnlyCoordinateTransform> transform{
      HalfOnlyCoordinateTransform{}};
  FullTag tag{};
  TransformContext<2> context{coord(4, 11), 1};
  auto result = transform(tag, vec::zeros(tag), context);
  for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_FLOAT_EQ(vec::get(tag, result, lane),
                    static_cast<float>(11 + lane));
  }
}

TEST(TensorTransformTest, LaneMappingAndActiveAreElementSemantic) {
  TransformContext<2, AffineLaneMapping, FirstActiveLanes> context{
      coord(5, 9), 0, AffineLaneMapping{3}, FirstActiveLanes{2}, 0};
  EXPECT_EQ(context.lane_coord(0), coord(5, 9));
  EXPECT_EQ(context.lane_coord(1), coord(8, 9));
  EXPECT_TRUE(context.is_active(1));
  EXPECT_FALSE(context.is_active(2));
}

TEST(TensorTransformTest, ZeroTransformDoesNotReadInput) {
  using Transform = ZeroVecTransform<float32_t>;
  static_assert(!Transform::reads_input);
  static_assert(Transform::permutation_equivariant);
  FullTag tag{};
  auto result = Transform{}(tag, vec::fill(tag, 9.0f));
  for (nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_FLOAT_EQ(vec::get(tag, result, lane), 0.0f);
  }
}

TEST(TensorTransformTest, AdaptPreservesVisibleBoundaryTypes) {
  auto original = make_elementwise_vec_transform<float32_t, float32_t>(
      []<vec::VectorTag Tag>(Tag, vec::Vec<Tag> value) { return value; });
  auto adapted = adapt_vec_transform<float32_t, float32_t>(original);
  static_assert(std::same_as<typename decltype(adapted)::TIn, float32_t>);
  static_assert(std::same_as<typename decltype(adapted)::TOut, float32_t>);
}

} // namespace
