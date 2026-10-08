// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 12

#include <gtest/gtest.h>

#include "TestHelpers.h"
#include "TestShard.h"
#include "vecops/vec/Basic.h"

namespace vec = vecops::vec;

template <typename T>
void run_mask_query_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size);

template <vec::VectorTag Tag>
void verify_mask_queries(Tag tag) {
  const vecops::nint_t lanes = vec::size(tag);
  const auto none = vec::mfalse(tag);
  const auto all = vec::mtrue(tag);

  EXPECT_FALSE(vec::mask_all(tag, none));
  EXPECT_FALSE(vec::mask_any(tag, none));
  EXPECT_TRUE(vec::mask_none(tag, none));
  EXPECT_EQ(0, vec::mask_count(tag, none));
  EXPECT_EQ(-1, vec::mask_first(tag, none));
  EXPECT_EQ(-1, vec::mask_last(tag, none));

  EXPECT_TRUE(vec::mask_all(tag, all));
  EXPECT_TRUE(vec::mask_any(tag, all));
  EXPECT_FALSE(vec::mask_none(tag, all));
  EXPECT_EQ(lanes, vec::mask_count(tag, all));
  EXPECT_EQ(0, vec::mask_first(tag, all));
  EXPECT_EQ(lanes - 1, vec::mask_last(tag, all));

  auto sparse = vec::mfalse(tag);
  sparse = vec::set(tag, sparse, 0, true);
  sparse = vec::set(tag, sparse, lanes / 2, true);
  sparse = vec::set(tag, sparse, lanes - 1, true);
  const vecops::nint_t expected_count = lanes == 1 ? 1 : lanes == 2 ? 2 : 3;
  EXPECT_EQ(expected_count == lanes, vec::mask_all(tag, sparse));
  EXPECT_TRUE(vec::mask_any(tag, sparse));
  EXPECT_FALSE(vec::mask_none(tag, sparse));
  EXPECT_EQ(expected_count, vec::mask_count(tag, sparse));
  EXPECT_EQ(0, vec::mask_first(tag, sparse));
  EXPECT_EQ(lanes - 1, vec::mask_last(tag, sparse));

  if (lanes > 1) {
    auto interior = vec::mfalse(tag);
    const vecops::nint_t first = lanes / 3;
    const vecops::nint_t last = lanes - 1 - lanes / 4;
    interior = vec::set(tag, interior, first, true);
    interior = vec::set(tag, interior, last, true);
    EXPECT_EQ(first, vec::mask_first(tag, interior));
    EXPECT_EQ(last, vec::mask_last(tag, interior));
    EXPECT_EQ(first == last ? 1 : 2, vec::mask_count(tag, interior));
  }
}

template <typename T>
void run_mask_query_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_mask_queries(Tag{});
  });
}

using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_mask_query_test<ShardType>();

#else

template <typename T>
class VecMaskQueryTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecMaskQueryTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecMaskQueryTest, EveryLegalElementAndScalableShape) {
  run_mask_query_test<TypeParam>();
}

#endif
