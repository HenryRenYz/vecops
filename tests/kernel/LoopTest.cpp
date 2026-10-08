// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include "vecops/kernel/Loop.h"
#include "vecops/tensor/OptionalOperand.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
namespace hop = vecops::kernel::loop;

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

TEST(HOPForEachTest, SingletonDynamicOneDimBroadcasts) {
  std::vector<int64_t> row{10, 20, 30};
  std::vector<int64_t> matrix{0, 1, 2, 3, 4, 5};
  auto tr = make_tensor(row.data(),
                        make_shape(Dynamic<1, 1, 1>{1}, cint<3>),
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

TEST(HOPForEachTest, NulloptIsForwardedAsRankZeroOperand) {
  static_assert(hop::details::slice_rank_v<tensor::nullopt_t> == 0);
  static_assert(!hop::details::is_sliceable_v<tensor::nullopt_t>);
  std::array<int64_t, 6> data{0, 1, 2, 3, 4, 5};
  auto tensor = make_tensor(
      data.data(), make_shape(cint<2>, cint<3>),
      make_strides(cint<3>, cint<1>));
  int calls = 0;
  hop::for_each_dims<2>(
      [&](auto missing, auto&& value) {
        static_assert(tensor::is_nullopt_v<decltype(missing)>);
        ++calls;
        EXPECT_GE(static_cast<int64_t>(value), 0);
      },
      tensor::nullopt, tensor);
  EXPECT_EQ(calls, 6);
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

TEST(HOPForEachTest, ForEachDimsCoversRanksOneThroughFour) {
  {
    std::vector<int64_t> data{0, 1, 2};
    auto t = make_tensor(data.data(), make_shape(cint<3>), make_strides(cint<1>));

    std::vector<int64_t> seen;
    hop::for_each_dims<1>([&](auto&& x) {
      seen.push_back(static_cast<int64_t>(x));
    }, t);

    EXPECT_EQ(seen, data);
  }

  {
    std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                         make_strides(cint<3>, cint<1>));

    std::vector<int64_t> seen;
    hop::for_each_dims<2>([&](auto&& x) {
      seen.push_back(static_cast<int64_t>(x));
    }, t);

    EXPECT_EQ(seen, data);
  }

  {
    std::vector<int64_t> data(2 * 3 * 4);
    for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
      data[static_cast<size_t>(i)] = i;
    }
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<4>),
                         make_strides(cint<12>, cint<4>, cint<1>));

    std::vector<int64_t> seen;
    hop::for_each_dims<3>([&](auto&& x) {
      seen.push_back(static_cast<int64_t>(x));
    }, t);

    EXPECT_EQ(seen, data);
  }

  {
    std::vector<int64_t> data(2 * 3 * 2 * 4);
    for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
      data[static_cast<size_t>(i)] = i;
    }
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<2>, cint<4>),
                         make_strides(cint<24>, cint<8>, cint<4>, cint<1>));

    std::vector<int64_t> seen;
    hop::for_each_dims<4>([&](auto&& x) {
      seen.push_back(static_cast<int64_t>(x));
    }, t);

    EXPECT_EQ(seen, data);
  }
}

TEST(HOPForEachTest, ForEachDimsCanTraversePrefixOfHigherRankTensor) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> row_first_values;
  hop::for_each_dims<1>([&](auto&& row) {
    EXPECT_EQ(row.ndim(), 1);
    EXPECT_EQ(row.size(0), 3);
    row_first_values.push_back(row(0));
  }, t);

  EXPECT_EQ(row_first_values, (std::vector<int64_t>{0, 3}));
}

TEST(HOPForEachTest, ForEachDimsWithIndexCoversRanksOneThroughFour) {
  {
    std::vector<int64_t> data{0, 1, 2};
    auto t = make_tensor(data.data(), make_shape(cint<3>), make_strides(cint<1>));

    int calls = 0;
    hop::for_each_dims_with_index<1>([&](nint_t i0, auto&& x) {
      ++calls;
      EXPECT_EQ(static_cast<int64_t>(x), i0);
    }, t);

    EXPECT_EQ(calls, 3);
  }

  {
    std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                         make_strides(cint<3>, cint<1>));

    int calls = 0;
    hop::for_each_dims_with_index<2>([&](nint_t i0, nint_t i1, auto&& x) {
      ++calls;
      EXPECT_EQ(static_cast<int64_t>(x), i0 * 3 + i1);
    }, t);

    EXPECT_EQ(calls, 6);
  }

  {
    std::vector<int64_t> data(2 * 3 * 4);
    for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
      data[static_cast<size_t>(i)] = i;
    }
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<4>),
                         make_strides(cint<12>, cint<4>, cint<1>));

    int calls = 0;
    hop::for_each_dims_with_index<3>(
        [&](nint_t i0, nint_t i1, nint_t i2, auto&& x) {
          ++calls;
          EXPECT_EQ(static_cast<int64_t>(x), i0 * 12 + i1 * 4 + i2);
        },
        t);

    EXPECT_EQ(calls, 24);
  }

  {
    std::vector<int64_t> data(2 * 3 * 2 * 4);
    for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
      data[static_cast<size_t>(i)] = i;
    }
    auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<2>, cint<4>),
                         make_strides(cint<24>, cint<8>, cint<4>, cint<1>));

    int calls = 0;
    hop::for_each_dims_with_index<4>(
        [&](nint_t i0, nint_t i1, nint_t i2, nint_t i3, auto&& x) {
          ++calls;
          EXPECT_EQ(static_cast<int64_t>(x), i0 * 24 + i1 * 8 + i2 * 4 + i3);
        },
        t);

    EXPECT_EQ(calls, 48);
  }
}

TEST(HOPForEachTest, ForEachDimsWithIndexCanTraversePrefixOfHigherRankTensor) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each_dims_with_index<1>([&](nint_t i0, auto&& row) {
    EXPECT_EQ(row.ndim(), 1);
    EXPECT_EQ(row.size(0), 3);
    seen.push_back(i0 * 10 + row(0));
  }, t);

  EXPECT_EQ(seen, (std::vector<int64_t>{0, 13}));
}

TEST(HOPForEachTest, ForEachDimsWithIndexTupleWithoutDimsCallsOnce) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  int calls = 0;
  hop::for_each_dims_with_index_tuple<0>(
      [&](const auto& indices, auto&& whole) {
        ++calls;
        EXPECT_EQ(std::tuple_size_v<std::remove_cvref_t<decltype(indices)>>, 0);
        EXPECT_EQ(whole.ndim(), 2);
        EXPECT_EQ(whole(1, 2), 5);
      },
      t);

  EXPECT_EQ(calls, 1);
}

TEST(HOPForEachTest, ForEachDimsWithIndexTupleCanTraversePrefixOfHigherRankTensor) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));

  std::vector<int64_t> seen;
  hop::for_each_dims_with_index_tuple<1>(
      [&](const auto& indices, auto&& row) {
        EXPECT_EQ(row.ndim(), 1);
        EXPECT_EQ(row.size(0), 3);
        seen.push_back(std::get<0>(indices) * 10 + row(0));
      },
      t);

  EXPECT_EQ(seen, (std::vector<int64_t>{0, 13}));
}

TEST(HOPForEachTest, ForEachWithIndexTupleHonorsRequestedIndexOrder) {
  std::vector<int64_t> data(2 * 3 * 4);
  for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
    data[static_cast<size_t>(i)] = i;
  }
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<4>),
                       make_strides(cint<12>, cint<4>, cint<1>));

  std::vector<int64_t> seen;
  std::vector<int64_t> expected;
  for (nint_t i1 = 0; i1 < 3; ++i1) {
    for (nint_t i2 = 0; i2 < 4; ++i2) {
      for (nint_t i0 = 0; i0 < 2; ++i0) {
        expected.push_back(i1 * 100 + i2 * 10 + i0);
      }
    }
  }

  hop::for_each_with_index_tuple<1, 2, 0>(
      [&](const auto& indices, auto&& x) {
        const nint_t i1 = std::get<0>(indices);
        const nint_t i2 = std::get<1>(indices);
        const nint_t i0 = std::get<2>(indices);
        EXPECT_EQ(static_cast<int64_t>(x), i0 * 12 + i1 * 4 + i2);
        seen.push_back(i1 * 100 + i2 * 10 + i0);
      },
      t);

  EXPECT_EQ(seen, expected);
}

TEST(HOPForEachTest, ForEachDimsWithIndexTuplePreservesBroadcastAndForwarding) {
  std::vector<int64_t> row{10, 20, 30};
  std::vector<int64_t> matrix{0, 1, 2, 3, 4, 5};
  auto tr = make_tensor(row.data(), make_shape(cint<1>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tm = make_tensor(matrix.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  int scale = 2;

  std::vector<int64_t> seen;
  hop::for_each_dims_with_index_tuple<2>(
      [&](const auto& indices, auto&& x, auto&& y, auto&& s) {
        EXPECT_EQ(s, scale);
        seen.push_back(
            std::get<0>(indices) * 100 +
            std::get<1>(indices) * 10 +
            static_cast<int64_t>(x + y) * s);
      },
      tr,
      tm,
      scale);

  EXPECT_EQ(seen, (std::vector<int64_t>{20, 52, 84, 126, 158, 190}));
}

TEST(HOPForEachTest, ForEachWithIndexHonorsRequestedIndexOrder) {
  std::vector<int64_t> data(2 * 3 * 4);
  for (nint_t i = 0; i < static_cast<nint_t>(data.size()); ++i) {
    data[static_cast<size_t>(i)] = i;
  }
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>, cint<4>),
                       make_strides(cint<12>, cint<4>, cint<1>));

  std::vector<int64_t> seen;
  std::vector<int64_t> expected;
  for (nint_t i1 = 0; i1 < 3; ++i1) {
    for (nint_t i2 = 0; i2 < 4; ++i2) {
      for (nint_t i0 = 0; i0 < 2; ++i0) {
        expected.push_back(i1 * 100 + i2 * 10 + i0);
      }
    }
  }

  hop::for_each_with_index<1, 2, 0>(
      [&](nint_t i1, nint_t i2, nint_t i0, auto&& x) {
        EXPECT_EQ(static_cast<int64_t>(x), i0 * 12 + i1 * 4 + i2);
        seen.push_back(i1 * 100 + i2 * 10 + i0);
      },
      t);

  EXPECT_EQ(seen, expected);
}

TEST(HOPForEachTest, ForEachWithIndexWithoutDimsCallsOnceWithOriginalInputs) {
  std::vector<int64_t> data{0, 1, 2, 3, 4, 5};
  auto t = make_tensor(data.data(), make_shape(cint<2>, cint<3>),
                       make_strides(cint<3>, cint<1>));
  int value = 13;

  int calls = 0;
  hop::for_each_with_index<>([&](auto&& whole, auto&& x) {
    ++calls;
    EXPECT_EQ(whole.ndim(), 2);
    EXPECT_EQ(whole(1, 2), 5);
    EXPECT_EQ(x, value);
  }, t, value);

  EXPECT_EQ(calls, 1);
}

TEST(HOPForEachTest, ForEachDimsWithIndexPreservesBroadcastAndForwarding) {
  std::vector<int64_t> row{10, 20, 30};
  std::vector<int64_t> matrix{0, 1, 2, 3, 4, 5};
  auto tr = make_tensor(row.data(), make_shape(cint<1>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  auto tm = make_tensor(matrix.data(), make_shape(cint<2>, cint<3>),
                        make_strides(cint<3>, cint<1>));
  int scale = 2;

  std::vector<int64_t> seen;
  hop::for_each_dims_with_index<2>(
      [&](nint_t i0, nint_t i1, auto&& x, auto&& y, auto&& s) {
        EXPECT_EQ(s, scale);
        seen.push_back(i0 * 100 + i1 * 10 + static_cast<int64_t>(x + y) * s);
      },
      tr,
      tm,
      scale);

  EXPECT_EQ(seen, (std::vector<int64_t>{20, 52, 84, 126, 158, 190}));
}

template <hop::TailCarryPolicy Policy>
void expect_vector_fold_reductions_and_invariants() {
  using Tag = vec::ScalableTag<int32_t>;
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  const nint_t n = 6 * lanes + 1;
  std::vector<int32_t> values(static_cast<std::size_t>(n));
  for (nint_t i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<int32_t>(i + 1);
  }

  int32_t sum = -1;
  int32_t maximum = -1;
  int32_t minimum = -1;
  const int32_t scalar_invariant = 2;
  auto base_vector = vec::fill(tag, int32_t{7});
  int calls = 0;
  hop::fold<4, 1, Policy>(
      tag, n,
      [&](auto block_tag, nint_t i, auto active,
          auto& sum_carry, auto& max_carry, auto& min_carry,
          auto& scalar_value, auto& vector_value)
          VECOPS_INLINE_LAMBDA {
        static_assert(!std::is_const_v<
            std::remove_reference_t<decltype(sum_carry)>>);
        static_assert(std::is_const_v<
            std::remove_reference_t<decltype(scalar_value)>>);
        static_assert(std::is_const_v<
            std::remove_reference_t<decltype(vector_value)>>);
        EXPECT_EQ(vec::reduce_min(block_tag, scalar_value), 2);
        EXPECT_EQ(vec::reduce_min(block_tag, vector_value), 7);

        if constexpr (std::same_as<
                          std::remove_cvref_t<decltype(active)>,
                          vec::opt::Unmasked>) {
          auto value = vec::load(block_tag, values.data() + i);
          sum_carry = vec::add(sum_carry, value);
          max_carry = vec::max(max_carry, value);
          min_carry = vec::min(min_carry, value);
        } else {
          auto sum_value = vec::load(
              block_tag, values.data() + i, active,
              vec::opt::merge(int32_t{0}));
          auto max_value = vec::load(
              block_tag, values.data() + i, active,
              vec::opt::merge(std::numeric_limits<int32_t>::lowest()));
          auto min_value = vec::load(
              block_tag, values.data() + i, active,
              vec::opt::merge(std::numeric_limits<int32_t>::max()));
          sum_carry = vec::add(sum_carry, sum_value);
          max_carry = vec::max(max_carry, max_value);
          min_carry = vec::min(min_carry, min_value);
        }
        ++calls;
      },
      hop::reduce_add(sum),
      hop::reduce_max(maximum),
      hop::reduce_min(minimum),
      hop::invariant(scalar_invariant),
      hop::invariant(base_vector));

  EXPECT_EQ(calls, 4);
  EXPECT_EQ(sum, static_cast<int32_t>(n * (n + 1) / 2));
  EXPECT_EQ(maximum, static_cast<int32_t>(n));
  EXPECT_EQ(minimum, 1);
}

TEST(HOPVectorFoldTest, IndependentTailSupportsVariadicCarries) {
  expect_vector_fold_reductions_and_invariants<
      hop::TailCarryPolicy::independent>();
}

TEST(HOPVectorFoldTest, ReusedPrefixSupportsVariadicCarries) {
  expect_vector_fold_reductions_and_invariants<
      hop::TailCarryPolicy::reuse_prefix>();
}

TEST(HOPVectorFoldTest, ZeroLengthReturnsReductionIdentity) {
  using Tag = vec::ScalableTag<int32_t>;
  int32_t sum = 123;
  int calls = 0;
  hop::fold(
      Tag{}, nint_t{0},
      [&](auto, nint_t, auto, auto&) VECOPS_INLINE_LAMBDA {
        ++calls;
      },
      hop::reduce_add(sum));
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(sum, 0);
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

TEST(HOPForEachDeathTest, UnconstrainedDynamicSizeOneDoesNotBroadcast) {
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
