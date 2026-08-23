#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "vecops/vec/Vec.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

template <vec::VectorTag Tag>
void expect_same(Tag tag, vec::Vec<Tag> expected, vec::Vec<Tag> actual) {
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, expected, lane), vec::get(tag, actual, lane)))
        << "lane=" << lane;
  }
}

template <vec::VectorTag Tag>
void expect_same(Tag tag, vec::Mask<Tag> expected, vec::Mask<Tag> actual) {
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    EXPECT_EQ(vec::get(tag, expected, lane), vec::get(tag, actual, lane))
        << "lane=" << lane;
}

template <typename Mask>
concept HasTaglessMaskAnd = requires(Mask mask) {
  vec::mask_and(mask, mask);
};

template <int ScalePower>
void verify_float_tagless() {
  using SourceTag = vec::ScalableTag<float, ScalePower>;
  using V = vec::Vec<SourceTag>;
  using Tag = vec::VecToTag<V>;
  static_assert(vec::VectorValue<V>);
  static_assert(std::same_as<V, vec::Vec<Tag>>);
  static_assert(std::same_as<Tag, vec::InferredTagOf<V>>);

  constexpr Tag tag{};
  auto a = vec::fill(tag, 1.25F);
  auto b = vec::fill(tag, 2.0F);
  auto c = vec::fill(tag, 0.5F);
  const auto mask = vec::mwhilelt(tag, 0, vec::size(tag) / 2);

  expect_same(tag, vec::add(tag, a, b), vec::add(a, b));
  expect_same(tag, vec::sub(tag, a, b), vec::sub(a, b));
  expect_same(tag, vec::mul(tag, a, b), vec::mul(a, b));
  expect_same(tag, vec::div(tag, a, b), vec::div(a, b));
  expect_same(tag, vec::min(tag, a, b), vec::min(a, b));
  expect_same(tag, vec::max(tag, a, b), vec::max(a, b));
  expect_same(
      tag,
      vec::add(tag, a, b, vec::opt::masked(mask), vec::opt::zero),
      vec::add(a, b, vec::opt::masked(mask), vec::opt::zero));

  expect_same(tag, vec::fmadd(tag, a, b, c), vec::fmadd(a, b, c));
  expect_same(tag, vec::fmsub(tag, a, b, c), vec::fmsub(a, b, c));
  expect_same(tag, vec::fnmadd(tag, a, b, c), vec::fnmadd(a, b, c));
  expect_same(tag, vec::fnmsub(tag, a, b, c), vec::fnmsub(a, b, c));
  expect_same(tag, vec::neg(tag, a), vec::neg(a));
  expect_same(tag, vec::abs(tag, vec::neg(tag, a)), vec::abs(vec::neg(tag, a)));
  expect_same(tag, vec::sqrt(tag, b), vec::sqrt(b));
  expect_same(tag, vec::rcp(tag, b), vec::rcp(b));
  expect_same(tag, vec::rsqrt(tag, b), vec::rsqrt(b));

  expect_same(tag, vec::cmpeq(tag, a, b), vec::cmpeq(a, b));
  expect_same(tag, vec::cmpne(tag, a, b), vec::cmpne(a, b));
  expect_same(tag, vec::cmplt(tag, a, b), vec::cmplt(a, b));
  expect_same(tag, vec::cmpgt(tag, a, b), vec::cmpgt(a, b));
  expect_same(tag, vec::cmple(tag, a, b), vec::cmple(a, b));
  expect_same(tag, vec::cmpge(tag, a, b), vec::cmpge(a, b));
  expect_same(tag, vec::isnan(tag, a), vec::isnan(a));
  expect_same(tag, vec::isposinf(tag, a), vec::isposinf(a));
  expect_same(tag, vec::isneginf(tag, a), vec::isneginf(a));
  expect_same(tag, vec::isinf(tag, a), vec::isinf(a));

  const auto negative = vec::fill(tag, -0.5F);
  expect_same(tag, vec::exp(tag, a), vec::exp(a));
  expect_same(
      tag,
      vec::exp(tag, a, vec::opt::math::fast),
      vec::exp(a, vec::opt::math::fast));
  expect_same(tag, vec::exp_strict(tag, a), vec::exp_strict(a));
  expect_same(tag, vec::exp_fast(tag, a), vec::exp_fast(a));
  expect_same(tag, vec::exp_est(tag, a), vec::exp_est(a));
  expect_same(tag, vec::exp_neg(tag, negative), vec::exp_neg(negative));
  expect_same(
      tag,
      vec::exp_neg(tag, negative, vec::opt::math::estimate),
      vec::exp_neg(negative, vec::opt::math::estimate));
  expect_same(
      tag,
      vec::exp_neg_strict(tag, negative),
      vec::exp_neg_strict(negative));
  expect_same(
      tag, vec::exp_neg_fast(tag, negative), vec::exp_neg_fast(negative));
  expect_same(
      tag, vec::exp_neg_est(tag, negative), vec::exp_neg_est(negative));

  expect_same(tag, vec::blend(tag, a, mask, b), vec::blend(a, mask, b));
  expect_same(
      tag, vec::interleave_even(tag, a, b), vec::interleave_even(a, b));
  expect_same(
      tag, vec::interleave_odd(tag, a, b), vec::interleave_odd(a, b));
  expect_same(
      tag,
      vec::local_interleave_lower(tag, a, b),
      vec::local_interleave_lower(a, b));
  expect_same(
      tag,
      vec::local_interleave_upper(tag, a, b),
      vec::local_interleave_upper(a, b));

  using IndexTag = vec::IndexTag<Tag>;
  auto indices = vec::zeros(IndexTag{});
  auto local_indices = vec::zeros(IndexTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    indices = vec::set(
        IndexTag{}, indices, lane, lane % vec::native_word_size(tag));
  constexpr vecops::nint_t block_lanes = 16 / sizeof(float);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    local_indices = vec::set(
        IndexTag{}, local_indices, lane, lane % block_lanes);
  expect_same(tag, vec::shuf(tag, a, indices), vec::shuf(a, indices));
  expect_same(
      tag,
      vec::local_shuf(tag, a, local_indices),
      vec::local_shuf(a, local_indices));

  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, a, 0), vec::get(a, 0)));
  expect_same(
      tag,
      vec::set(tag, a, 0, 9.0F),
      vec::set(a, 0, 9.0F));

  static_assert(!HasTaglessMaskAnd<vec::Mask<Tag>>);
}

template <int ScalePower>
void verify_integer_tagless() {
  using SourceTag = vec::ScalableTag<std::int32_t, ScalePower>;
  using V = vec::Vec<SourceTag>;
  using Tag = vec::VecToTag<V>;
  constexpr Tag tag{};
  auto a = vec::fill(tag, std::int32_t{0x12345678});
  auto b = vec::fill(tag, std::int32_t{0x0f0f0f0f});
  auto counts = vec::fill(tag, std::int32_t{3});
  const auto mask = vec::mwhilelt(tag, 0, vec::size(tag) / 2);

  expect_same(tag, vec::bit_and(tag, a, b), vec::bit_and(a, b));
  expect_same(tag, vec::bit_or(tag, a, b), vec::bit_or(a, b));
  expect_same(tag, vec::bit_xor(tag, a, b), vec::bit_xor(a, b));
  expect_same(tag, vec::bit_andnot(tag, a, b), vec::bit_andnot(a, b));
  expect_same(tag, vec::bit_not(tag, a), vec::bit_not(a));
  expect_same(tag, vec::bit_shl(tag, a, 3), vec::bit_shl(a, 3));
  expect_same(tag, vec::bit_shr(tag, a, 3), vec::bit_shr(a, 3));
  expect_same(
      tag, vec::bit_shl(tag, a, vecops::meta::cint<3>),
      vec::bit_shl(a, vecops::meta::cint<3>));
  expect_same(
      tag, vec::bit_shr(tag, a, vecops::meta::dyn<1>(3)),
      vec::bit_shr(a, vecops::meta::dyn<1>(3)));
  expect_same(tag, vec::bit_shl(tag, a, counts), vec::bit_shl(a, counts));
  expect_same(tag, vec::bit_shr(tag, a, counts), vec::bit_shr(a, counts));
  expect_same(
      tag,
      vec::bit_xor(tag, a, b, vec::opt::masked(mask), vec::opt::zero),
      vec::bit_xor(a, b, vec::opt::masked(mask), vec::opt::zero));
}

TEST(VecTaglessTest, CanonicalPhysicalTagAndForwarding) {
  verify_float_tagless<0>();
  verify_float_tagless<1>();
  verify_integer_tagless<0>();
  verify_integer_tagless<1>();
}

using SubwordTag = vec::ScalableTag<float, -1>;
using SubwordInferredTag = vec::VecToTag<vec::Vec<SubwordTag>>;
static_assert(!std::same_as<SubwordTag, SubwordInferredTag>);

} // namespace
