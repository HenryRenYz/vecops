#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vecops/gemm/HOP.h"

using namespace vecops;
using namespace vecops::gemm;

TEST(HOPForEachTest, OneDimTensorYieldsScalars) {
  std::vector<int64_t> data{1, 2, 3, 4};
  auto t = make_tensor(data.data(), make_shape(cint<4>), make_strides(cint<1>));

  std::vector<int64_t> seen;
  hop::for_each<0>([&](auto&& x) {
    seen.push_back(static_cast<int64_t>(x));
  }, t);

  EXPECT_EQ(seen, (std::vector<int64_t>{1, 2, 3, 4}));
}

TEST(HOPForEachTest, TraversingOuterDimYieldsSubTensors) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> first_values;
  hop::for_each<0>([&](auto&& row) {
    EXPECT_EQ(row.ndim(), 1);
    EXPECT_EQ(row.size(0), 3);
    first_values.push_back(row(0));
  }, t);

  EXPECT_EQ(first_values, (std::vector<int64_t>{0, 3}));
}

TEST(HOPForEachTest, TraversingBothDimsYieldsScalars) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each<0, 1>([&](auto&& x) {
    seen.push_back(static_cast<int64_t>(x));
  }, t);

  EXPECT_EQ(seen, data);
}

TEST(HOPForEachTest, MultiTensorSameShape) {
  std::vector<int64_t> a{0, 1, 2, 3, 4, 5};
  std::vector<int64_t> b{10, 11, 12, 13, 14, 15};
  auto ta = make_tensor(a.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tb = make_tensor(b.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));

  std::vector<int64_t> sums;
  hop::for_each_dims<2>([&](auto&& x, auto&& y) {
    sums.push_back(static_cast<int64_t>(x) + static_cast<int64_t>(y));
  }, ta, tb);

  EXPECT_EQ(sums, (std::vector<int64_t>{10, 12, 14, 16, 18, 20}));
}

TEST(HOPForEachTest, ForEachDimsUsesTrailingAlignment) {
  std::vector<int64_t> matrix{0, 1, 2, 3, 4, 5};
  std::vector<int64_t> vector{10, 20, 30};
  auto tm = make_tensor(matrix.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tv = make_tensor(vector.data(), make_shape(cint<3>), make_strides(cint<1>));

  std::vector<int64_t> seen;
  hop::for_each_dims<2>([&](auto&& x, auto&& y) {
    seen.push_back(static_cast<int64_t>(x) + static_cast<int64_t>(y));
  }, tm, tv);

  EXPECT_EQ(seen, (std::vector<int64_t>{10, 21, 32, 13, 24, 35}));
}

TEST(HOPForEachTest, ConstOneDimBroadcasts) {
  std::vector<int64_t> row{10, 20, 30};
  std::vector<int64_t> matrix{0, 1, 2, 3, 4, 5};
  auto tr = make_tensor(row.data(), make_shape(cint<1>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tm = make_tensor(matrix.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each_dims<2>([&](auto&& x, auto&& y) {
    seen.push_back(static_cast<int64_t>(x) + static_cast<int64_t>(y));
  }, tr, tm);

  EXPECT_EQ(seen, (std::vector<int64_t>{10, 21, 32, 13, 24, 35}));
}

TEST(HOPForEachTest, SubTensorShapesMayDiffer) {
  std::vector<int64_t> lhs(2 * 3 * 4);
  std::vector<int64_t> rhs(2 * 4);
  auto tl = make_tensor(lhs.data(), make_shape(cint<2>, cint<3>, cint<4>),
                        make_strides(cint<12>, cint<4>, cint<1>));
  auto tr = make_tensor(rhs.data(), make_shape(cint<2>, cint<4>),
                        make_strides(cint<4>, cint<1>));

  int calls = 0;
  hop::for_each<0>([&](auto&& l, auto&& r) {
    ++calls;
    EXPECT_EQ(l.ndim(), 2);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(l.size(0), 3);
    EXPECT_EQ(l.size(1), 4);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 4);
  }, tl, tr);

  EXPECT_EQ(calls, 2);
}

TEST(HOPForEachTest, NonTensorInputsAreForwardedWithoutAffectingTraversal) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));
  int scale = 7;

  int calls = 0;
  int64_t sum = 0;
  hop::for_each_dims<2>([&](auto&& s, auto&& x) {
    ++calls;
    EXPECT_EQ(s, scale);
    sum += static_cast<int64_t>(s) * static_cast<int64_t>(x);
  }, scale, t);

  EXPECT_EQ(calls, 6);
  EXPECT_EQ(sum, 7 * (0 + 1 + 2 + 3 + 4 + 5));
}

TEST(HOPForEachTest, ForEachDimsCanWriteScalarElements) {
  std::vector<int64_t> input{0, 1, 2, 3, 4, 5};
  std::vector<int64_t> output(6, -1);
  auto ti = make_tensor(input.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto to = make_tensor(output.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  int factor = 3;

  hop::for_each_dims<2>([](auto&& dst, auto&& src, auto&& scale) {
    dst = static_cast<int64_t>(src) * static_cast<int64_t>(scale);
  }, to, ti, factor);

  EXPECT_EQ(output, (std::vector<int64_t>{0, 3, 6, 9, 12, 15}));
}

TEST(HOPForEachTest, ForEachCanWriteSubTensorViews) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  hop::for_each<0>([](auto&& row) {
    row(0) = 10 + row(0);
    row(2) = 100 + row(2);
  }, t);

  EXPECT_EQ(data, (std::vector<int64_t>{10, 1, 102, 13, 4, 105}));
}

TEST(HOPForEachTest, ForEachCanTraverseNonLeadingDim) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each<1>([&](auto&& col) {
    EXPECT_EQ(col.ndim(), 1);
    EXPECT_EQ(col.size(0), 2);
    seen.push_back(col(0));
    seen.push_back(col(1));
  }, t);

  EXPECT_EQ(seen, (std::vector<int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(HOPForEachTest, ForEachHonorsTraversalOrder) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each<1, 0>([&](auto&& x) {
    seen.push_back(static_cast<int64_t>(x));
  }, t);

  EXPECT_EQ(seen, (std::vector<int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(HOPForEachTest, ForEachWithoutDimsCallsOnceWithOriginalInputs) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));
  int value = 11;

  int calls = 0;
  hop::for_each<>([&](auto&& whole, auto&& x) {
    ++calls;
    EXPECT_EQ(whole.ndim(), 2);
    EXPECT_EQ(whole.size(0), 2);
    EXPECT_EQ(whole.size(1), 3);
    EXPECT_EQ(whole(1, 2), 5);
    EXPECT_EQ(x, value);
  }, t, value);

  EXPECT_EQ(calls, 1);
}

#ifdef VECOPS_DEBUG
TEST(HOPForEachDeathTest, NonBroadcastExtentMismatch) {
  std::vector<int64_t> a(2 * 3);
  std::vector<int64_t> b(2 * 4);
  auto ta = make_tensor(a.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tb = make_tensor(b.data(), make_shape(cint<2>, cint<4>),
                        make_strides(cint<4>, cint<1>));

  EXPECT_DEATH((hop::for_each_dims<2>([](auto&&, auto&&) {}, ta, tb)),
               "broadcast extent mismatch");
}

TEST(HOPForEachDeathTest, DynamicSizeOneDoesNotBroadcast) {
  std::vector<int64_t> a(1 * 3);
  std::vector<int64_t> b(2 * 3);
  auto ta = make_tensor(a.data(), make_shape(Any{1}, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tb = make_tensor(b.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));

  EXPECT_DEATH((hop::for_each_dims<2>([](auto&&, auto&&) {}, ta, tb)),
               "broadcast extent mismatch");
}
#endif
