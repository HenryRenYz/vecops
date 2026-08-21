#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "vecops/util/Math.h"

using namespace vecops;

static_assert(floor_div(9, 4) == 2);
static_assert(ceil_div(9, 4) == 3);
static_assert(floor_div(-9, 4) == -3);
static_assert(ceil_div(-9, 4) == -2);
static_assert(ceil_div(8, 4) == 2);
static_assert(align_down(17, 8) == 16);
static_assert(align_up(17, 8) == 24);
static_assert(cdiv(9, 4) == ceil_div(9, 4));

TEST(UtilMathTest, IntegerDivisionRoundsInNamedDirection) {
  EXPECT_EQ(floor_div(0, 7), 0);
  EXPECT_EQ(floor_div(15, 4), 3);
  EXPECT_EQ(ceil_div(0, 7), 0);
  EXPECT_EQ(ceil_div(15, 4), 4);
  EXPECT_EQ(ceil_div(16, 4), 4);
}

TEST(UtilMathTest, AlignmentRoundsToMultiples) {
  EXPECT_EQ(align_down(0, 16), 0);
  EXPECT_EQ(align_down(31, 16), 16);
  EXPECT_EQ(align_up(0, 16), 0);
  EXPECT_EQ(align_up(31, 16), 32);
  EXPECT_EQ(align_up(32, 16), 32);
}

TEST(UtilMathTest, CeilDivisionAvoidsAdditionOverflow) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(ceil_div(maximum, std::uint64_t{2}), maximum / 2 + 1);
}
