#include <cmath>
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

static_assert(min(3, 7) == 3);
static_assert(min(7, 3) == 3);
static_assert(min(5, 5) == 5);
static_assert(min(-3, 3) == -3);
static_assert(max(3, 7) == 7);
static_assert(max(7, 3) == 7);
static_assert(max(5, 5) == 5);
static_assert(max(-3, 3) == 3);
static_assert(clamp(2, 0, 8) == 2);
static_assert(clamp(-1, 0, 8) == 0);
static_assert(clamp(9, 0, 8) == 8);
static_assert(clamp(0, 0, 8) == 0);
static_assert(clamp(8, 0, 8) == 8);
static_assert(min({3, 1, 4, 1, 5}) == 1);
static_assert(max({3, 1, 4, 1, 5}) == 5);

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

TEST(UtilMathTest, MinMaxClampMatchStdSemantics) {
  EXPECT_EQ(min(3, 7), std::min(3, 7));
  EXPECT_EQ(max(3.5, -2.0), std::max(3.5, -2.0));
  EXPECT_EQ(clamp(9, 0, 8), std::clamp(9, 0, 8));

  // NaN behaviour matches std:: exactly: the comparison is false, so the
  // first operand wins in every case (std::min(a,b) is (b<a)?b:a, etc).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(min(nan, 1.0)));
  EXPECT_EQ(min(1.0, nan), 1.0);
  EXPECT_TRUE(std::isnan(max(nan, 1.0)));
  EXPECT_EQ(max(1.0, nan), 1.0);
  EXPECT_TRUE(std::isnan(clamp(nan, 0.0, 1.0)));
  // Non-NaN cross-checks against the standard implementations.
  EXPECT_EQ(std::min(1.0, 2.0), min(1.0, 2.0));
  EXPECT_EQ(std::max(1.0, 2.0), max(1.0, 2.0));
  EXPECT_EQ(std::clamp(0.5, 0.0, 1.0), clamp(0.5, 0.0, 1.0));

  EXPECT_EQ(min({7, 3, 9, 3}), std::min({7, 3, 9, 3}));
  EXPECT_EQ(max({7, 3, 9, 3}), std::max({7, 3, 9, 3}));
}
