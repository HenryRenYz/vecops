#include <gtest/gtest.h>

#include <arm_sve.h>

#include <utility>

#include "vecops/kernel/Loop.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
namespace hop = vecops::kernel::loop;

template <int Unroll>
void expect_sizeless_carries() {
  const auto result = hop::scan<Unroll>(
      cint<16>,
      cint<4>,
      [](auto&& use) -> decltype(auto) {
        auto position_sum = svdup_n_f32(0.0f);
        auto count_sum = svdup_n_f32(0.0f);
        return use(position_sum, count_sum);
      },
      [](nint_t i, auto&& count, auto& position_sum, auto& count_sum) {
        const auto pg = svptrue_b32();
        position_sum = svadd_n_f32_x(pg, position_sum, static_cast<float>(i));
        count_sum = svadd_n_f32_x(
            pg, count_sum, static_cast<float>(static_cast<nint_t>(count)));
      },
      [](auto& dst_position, auto& dst_count, auto& src_position, auto& src_count) {
        const auto pg = svptrue_b32();
        dst_position = svadd_f32_x(pg, dst_position, src_position);
        dst_count = svadd_f32_x(pg, dst_count, src_count);
      },
      [](auto& position_sum, auto& count_sum) {
        const auto pg = svptrue_b32();
        return std::pair{
            svaddv_f32(pg, position_sum),
            svaddv_f32(pg, count_sum)};
      });

  const auto lanes = static_cast<float>(svcntw());
  EXPECT_FLOAT_EQ(result.first, 24.0f * lanes);
  EXPECT_FLOAT_EQ(result.second, 16.0f * lanes);

  const auto single_carry = hop::scan<Unroll>(
      svdup_n_f32(0.0f),
      cint<16>,
      cint<4>,
      [](svfloat32_t acc, nint_t, auto&& count) {
        return svadd_n_f32_x(
            svptrue_b32(), acc,
            static_cast<float>(static_cast<nint_t>(count)));
      },
      [](svfloat32_t lhs, svfloat32_t rhs) {
        return svadd_f32_x(svptrue_b32(), lhs, rhs);
      });
  EXPECT_FLOAT_EQ(
      svaddv_f32(svptrue_b32(), single_carry),
      16.0f * lanes);
}

TEST(HOPSizelessTest, VariadicSVECarrySupportsUnrollOneThroughFour) {
  expect_sizeless_carries<1>();
  expect_sizeless_carries<2>();
  expect_sizeless_carries<3>();
  expect_sizeless_carries<4>();
}
