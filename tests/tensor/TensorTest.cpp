//
// Created by renyz on 2026/6/11.
// Comprehensive tests for Tensor.h - compile-time shape/stride tensor
//

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <memory>

#include "vecops/tensor/Tensor.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

// ============================================================================
// Test Fixtures
// ============================================================================

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_1d_.resize(10);
        std::iota(data_1d_.begin(), data_1d_.end(), 0);

        data_2d_.resize(20);
        std::iota(data_2d_.begin(), data_2d_.end(), 0);

        data_3d_.resize(60);
        std::iota(data_3d_.begin(), data_3d_.end(), 0);

        data_4d_.resize(120);
        std::iota(data_4d_.begin(), data_4d_.end(), 0);
    }

    std::vector<int64_t> data_1d_;
    std::vector<int64_t> data_2d_;
    std::vector<int64_t> data_3d_;
    std::vector<int64_t> data_4d_;
};

// ============================================================================
// 1. Helper/Marker Tests
// ============================================================================

class TensorMarkerTest : public ::testing::Test {};
  
TEST_F(TensorMarkerTest, NewAxisDefaultRepeat) {
    auto na = new_axis();
    EXPECT_EQ(na.repeat, 1);
    EXPECT_TRUE((std::is_same_v<typename decltype(na)::value_type, Const<1>>));
}

TEST_F(TensorMarkerTest, NewAxisIntRepeat) {
    auto na = new_axis(5);
    EXPECT_EQ(na.repeat, 5);
}

TEST_F(TensorMarkerTest, NewAxisConstRepeat) {
    auto na = new_axis(cint<3>);
    EXPECT_EQ(na.repeat, 3);
    using ExpectedType = tensor::details::NewAxis<Const<3>>;
    EXPECT_TRUE((std::is_same_v<decltype(na), ExpectedType>));
}

TEST_F(TensorMarkerTest, NewAxisZeroRepeat) {
    auto na = new_axis(0);
    EXPECT_EQ(na.repeat, 0);
}

TEST_F(TensorMarkerTest, NewAxisLargeRepeat) {
    auto na = new_axis(1000000);
    EXPECT_EQ(na.repeat, 1000000);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorMarkerTest, NewAxisNegativeRepeatDeath) {
    EXPECT_DEATH(new_axis(-1), "repeat negative");
}
#endif

TEST_F(TensorMarkerTest, RangeBasic) {
    auto r = range(0, 10);
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 10);
    EXPECT_EQ(r.step, 1);
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::start_type, Any>));
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::end_type, Any>));
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::step_type, Const<1>>));
}

TEST_F(TensorMarkerTest, RangeWithStep) {
    auto r = range(0, 10, 2);
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 10);
    EXPECT_EQ(r.step, 2);
}

TEST_F(TensorMarkerTest, RangeConstArgs) {
    auto r = range(cint<0>, cint<10>, cint<2>);
    using StartT = typename decltype(r)::start_type;
    using EndT = typename decltype(r)::end_type;
    using StepT = typename decltype(r)::step_type;
    EXPECT_TRUE((std::is_same_v<StartT, Const<0>>));
    EXPECT_TRUE((std::is_same_v<EndT, Const<10>>));
    EXPECT_TRUE((std::is_same_v<StepT, Const<2>>));
}

TEST_F(TensorMarkerTest, RangeConstSlicePreservesShapeAndStride) {
    std::array<int64_t, 10> data{};
    auto tensor = make_tensor(data.data(), make_shape(cint<10>),
                              make_strides(cint<1>));
    auto sliced = tensor(range(cint<0>, cint<10>, cint<2>));
    static_assert(std::same_as<size_type_t<0, typename decltype(sliced)::Layout>,
                               Const<5>>);
    static_assert(std::same_as<stride_type_t<0, typename decltype(sliced)::Layout>,
                               Const<2>>);
    EXPECT_EQ(sliced.size(0), 5);
}

TEST_F(TensorMarkerTest, RangeDynamicSlicePreservesBounds) {
    std::array<int64_t, 12> data{};
    auto tensor = make_tensor(data.data(), make_shape(cint<12>),
                              make_strides(cint<1>));
    auto sliced = tensor(range(Dynamic<1, 0, 2>{2}, Dynamic<1, 8, 10>{10},
                               cint<2>));
    using Size = size_type_t<0, typename decltype(sliced)::Layout>;
    static_assert(std::same_as<Size, Dynamic<1, 3, 5>>);
    EXPECT_EQ(sliced.size(0), 4);
}

TEST_F(TensorMarkerTest, RangeSingletonDynamicStepKeepsSizeBounds) {
    std::array<int64_t, 12> data{};
    auto tensor = make_tensor(data.data(), make_shape(cint<12>),
                              make_strides(cint<1>));
    auto sliced = tensor(range(Dynamic<1, 0, 2>{2}, Dynamic<1, 8, 10>{10},
                               Dynamic<1, 1, 1>{1}));
    using Size = size_type_t<0, typename decltype(sliced)::Layout>;
    static_assert(std::same_as<Size, Dynamic<1, 6, 10>>);
    EXPECT_EQ(sliced.size(0), 8);
}

TEST_F(TensorMarkerTest, RangeTypedBoundsKeepDefaultUnitStep) {
    auto r = range(Dynamic<4, 0, 32>{4}, Dynamic<4, 4, 64>{20});
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::start_type,
                                Dynamic<4, 0, 32>>));
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::end_type,
                                Dynamic<4, 4, 64>>));
    EXPECT_TRUE((std::is_same_v<typename decltype(r)::step_type, Const<1>>));
}

TEST_F(TensorMarkerTest, RangeNegativeStep) {
    auto r = range(10, 0, -1);
    EXPECT_EQ(r.step, -1);
}

TEST_F(TensorMarkerTest, RangeLargeStep) {
    auto r = range(0, 100, 25);
    EXPECT_EQ(r.step, 25);
}

TEST_F(TensorMarkerTest, ReserveCompiles) {
    auto r = reserve;
    (void)r;
}

// ============================================================================
// 2. Construction Tests
// ============================================================================

class TensorConstructionTest : public TensorTest {};

TEST_F(TensorConstructionTest, ConstructTyped) {
    auto s = make_shape(Any{4}, Any{5});
    auto st = make_strides(Any{5}, Any{1});
    Tensor<int64_t, decltype(s), decltype(st)> t(data_2d_.data(), s, st);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.size(1), 5);
    EXPECT_EQ(t.stride(0), 5);
    EXPECT_EQ(t.stride(1), 1);
}

TEST_F(TensorConstructionTest, ConstructLayout) {
    auto layout = make_layout(make_shape(4, 5), make_strides(5, 1));
    Tensor<int64_t, Shape<Any, Any>, Strides<Any, Any>> t(data_2d_.data(), layout);
    EXPECT_EQ(t.ndim(), 2);
}

TEST_F(TensorConstructionTest, ConstructInitListBoth) {
    Tensor<int64_t, Shape<Any, Any>, Strides<Any, Any>> t(data_2d_.data(), {4, 5}, {5, 1});
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.size(1), 5);
    EXPECT_EQ(t.stride(0), 5);
    EXPECT_EQ(t.stride(1), 1);
}

TEST_F(TensorConstructionTest, ConstructInitListSizesOnly) {
    Tensor<int64_t, Shape<Any, Any>, Strides<Any, Any>> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.size(1), 5);
    EXPECT_EQ(t.stride(0), 5);
    EXPECT_EQ(t.stride(1), 1);
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(TensorConstructionTest, ArrayAlias_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 4);
    EXPECT_EQ(arr.size(1), 5);
    EXPECT_EQ(arr.stride(0), 5);
    EXPECT_EQ(arr.stride(1), 1);
    EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(TensorConstructionTest, ArrayAlias_1D) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_EQ(arr.ndim(), 1);
    EXPECT_EQ(arr.numel(), 10);
}

TEST_F(TensorConstructionTest, Construct_3D) {
    Tensor<int64_t, Shape<Any, Any, Any>, Strides<Any, Any, Any>> t(
        data_3d_.data(), {3, 4, 5});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.numel(), 60);
}

TEST_F(TensorConstructionTest, Construct_4D) {
    Tensor<int64_t, Shape<Any, Any, Any, Any>, Strides<Any, Any, Any, Any>> t(
        data_4d_.data(), {2, 3, 4, 5});
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.numel(), 120);
}

TEST_F(TensorConstructionTest, ConstructWithConstShape) {
    auto s = make_shape(cint<4>, cint<5>);
    auto st = make_strides(cint<5>, cint<1>);
    Tensor<int64_t, decltype(s), decltype(st)> t(data_2d_.data(), s, st);
    static_assert(std::same_as<decltype(t.numel_value()), meta::Const<20>>);
    EXPECT_EQ(t.size<0>(), 4);
    EXPECT_EQ(t.numel_value(), 20);
    EXPECT_TRUE((t.ct_is_contiguous));
}

#ifdef VECOPS_DEBUG
TEST_F(TensorConstructionTest, ConstructInitListWrongSizeDeath) {
    EXPECT_DEATH(
        (Tensor<int64_t, Shape<Any, Any>, Strides<Any, Any>>(data_2d_.data(), {4, 5, 6})),
        "shape_vals.size()");
}
#endif

// ============================================================================
// 3. make_tensor Tests
// ============================================================================

class MakeTensorTest : public TensorTest {};

TEST_F(MakeTensorTest, MakeTensorTyped) {
    auto t = make_tensor(data_2d_.data(), make_shape(4, 5), make_strides(5, 1));
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(0), 4);
}

TEST_F(MakeTensorTest, MakeTensorTypedConst) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                         make_strides(cint<5>, cint<1>));
    EXPECT_TRUE((t.ct_is_contiguous));
}

TEST_F(MakeTensorTest, MakeTensorTypedMixed) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, Any{5}),
                         make_strides(cint<5>, cint<1>));
    EXPECT_EQ(t.size<0>(), 4);
    EXPECT_FALSE((t.ct_is_contiguous));
}

TEST_F(MakeTensorTest, MakeTensorInitListBoth) {
    auto t = make_tensor<2>(data_2d_.data(), {4, 5}, {5, 1});
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.stride(1), 1);
}

TEST_F(MakeTensorTest, MakeTensorInitListSizesOnly) {
    auto t = make_tensor<2>(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.stride(0), 5);
    EXPECT_TRUE((std::is_same_v<typename decltype(t)::Stride, Strides<Any, Const<1>>>));
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(MakeTensorTest, MakeTensorInitList_1D) {
    auto t = make_tensor<1>(data_1d_.data(), {10});
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.stride(0), 1);
    EXPECT_TRUE((std::is_same_v<typename decltype(t)::Stride, Strides<Const<1>>>));
}

TEST_F(MakeTensorTest, MakeTensorInitList_3D) {
    auto t = make_tensor<3>(data_3d_.data(), {3, 4, 5});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.numel(), 60);
    EXPECT_TRUE((std::is_same_v<typename decltype(t)::Stride, Strides<Any, Any, Const<1>>>));
}

TEST_F(MakeTensorTest, MakeTensorInitList_4D) {
    auto t = make_tensor<4>(data_4d_.data(), {2, 3, 4, 5});
    EXPECT_EQ(t.ndim(), 4);
}

// ============================================================================
// 4. Dimension Access Tests
// ============================================================================

class TensorDimAccessTest : public TensorTest {};

TEST_F(TensorDimAccessTest, Size_CT_Access) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                         make_strides(cint<5>, cint<1>));
    EXPECT_EQ(t.size<0>(), 4);
    EXPECT_EQ(t.size<1>(), 5);
}

TEST_F(TensorDimAccessTest, Size_RT_Access) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.size(0), 4);
    EXPECT_EQ(t.size(1), 5);
}

TEST_F(TensorDimAccessTest, Stride_CT_Access) {
    auto t = make_tensor(data_2d_.data(), make_shape(4, 5),
                         make_strides(cint<5>, cint<1>));
    EXPECT_EQ(t.stride<0>(), 5);
    EXPECT_EQ(t.stride<1>(), 1);
}

TEST_F(TensorDimAccessTest, Stride_RT_Access) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.stride(0), 5);
    EXPECT_EQ(t.stride(1), 1);
}

TEST_F(TensorDimAccessTest, Ndim) {
    EXPECT_EQ((Array<int64_t, 1>(data_1d_.data(), {10}).ndim()), 1);
    EXPECT_EQ((Array<int64_t, 3>(data_3d_.data(), {3, 4, 5}).ndim()), 3);
}

TEST_F(TensorDimAccessTest, Numel_1D) {
    Array<int64_t, 1> t(data_1d_.data(), {10});
    EXPECT_EQ(t.numel(), 10);
}

TEST_F(TensorDimAccessTest, Numel_2D) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.numel(), 20);
}

TEST_F(TensorDimAccessTest, Numel_3D) {
    Array<int64_t, 3> t(data_3d_.data(), {3, 4, 5});
    EXPECT_EQ(t.numel(), 60);
}

TEST_F(TensorDimAccessTest, Numel_4D) {
    Array<int64_t, 4> t(data_4d_.data(), {2, 3, 4, 5});
    EXPECT_EQ(t.numel(), 120);
}

TEST_F(TensorDimAccessTest, NumelWithSizeOne) {
    Array<int64_t, 3> t(data_3d_.data(), {1, 1, 60});
    EXPECT_EQ(t.numel(), 60);
}

TEST_F(TensorDimAccessTest, DataPointer) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.data(), data_2d_.data());
}

TEST_F(TensorDimAccessTest, DataPointerConst) {
    const Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_EQ(t.data(), data_2d_.data());
}

TEST_F(TensorDimAccessTest, DataMutable) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    int64_t* ptr = t.data();
    ptr[0] = 999;
    EXPECT_EQ(data_2d_[0], 999);
}

TEST_F(TensorDimAccessTest, LayoutAccess) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    auto& layout = t.layout();
    EXPECT_EQ(layout.ndim(), 2);
    EXPECT_EQ(size<0>(layout), 4);
}

// ============================================================================
// 5. Contiguity Tests
// ============================================================================

class TensorContiguityTest : public TensorTest {};

TEST_F(TensorContiguityTest, CT_Contiguous) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                         make_strides(cint<6>, cint<1>));
    EXPECT_TRUE((t.ct_is_contiguous));
    EXPECT_TRUE((t.ct_is_last_contiguous<2>));
    EXPECT_TRUE((t.ct_is_last_contiguous<1>));
}

TEST_F(TensorContiguityTest, SingletonDynamicIsCompileTimeContiguous) {
    auto t = make_tensor(
        data_2d_.data(),
        make_shape(Dynamic<1, 4, 4>{4}, Dynamic<1, 5, 5>{5}),
        make_strides(Dynamic<1, 5, 5>{5}, Dynamic<1, 1, 1>{1}));
    static_assert(decltype(t)::ct_is_contiguous);
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(TensorContiguityTest, CT_LastContiguous) {
    auto t = make_tensor(data_2d_.data(), make_shape(4, 6),
                         make_strides(Any{6}, cint<1>));
    EXPECT_TRUE((t.ct_is_last_contiguous<1>));
    EXPECT_FALSE((t.ct_is_contiguous));
}

TEST_F(TensorContiguityTest, CT_NotContiguous_ShapeNotConst) {
    auto t = make_tensor(data_2d_.data(), make_shape(Any{4}, Any{6}),
                         make_strides(cint<6>, cint<1>));
    EXPECT_FALSE((t.ct_is_last_contiguous<2>));
}

TEST_F(TensorContiguityTest, CT_NotContiguous_StrideNotUnit) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                         make_strides(cint<6>, cint<2>));
    EXPECT_FALSE((t.ct_is_last_contiguous<1>));
}

TEST_F(TensorContiguityTest, RT_Contiguous) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 6});
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_TRUE(t.is_last_contiguous<1>());
    EXPECT_TRUE(t.is_last_contiguous<2>());
}

TEST_F(TensorContiguityTest, RT_NonContiguous) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5}, {10, 2});
    EXPECT_FALSE(t.is_contiguous());
    EXPECT_FALSE(t.is_last_contiguous<1>());
}

TEST_F(TensorContiguityTest, RT_3D) {
    Array<int64_t, 3> t(data_3d_.data(), {3, 4, 5});
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(TensorContiguityTest, FreeFunction_IsContiguous) {
    Array<int64_t, 2> t(data_2d_.data(), {4, 5});
    EXPECT_TRUE(is_contiguous(t));
}

TEST_F(TensorContiguityTest, FreeFunction_IsLastContiguous) {
    Array<int64_t, 3> t(data_3d_.data(), {3, 4, 5});
    EXPECT_TRUE(is_last_contiguous<2>(t));
    EXPECT_TRUE(is_last_contiguous<1>(t));
}

// ============================================================================
// 6. operator() — Integer Index
// ============================================================================

class TensorIntIndexTest : public TensorTest {};

TEST_F(TensorIntIndexTest, Index_1D_Valid) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    for (int i = 0; i < 10; ++i) EXPECT_EQ(arr(i), i);
}

TEST_F(TensorIntIndexTest, Index_2D_Valid) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    int64_t expected = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_EQ(arr(i, j), expected++);
}

TEST_F(TensorIntIndexTest, Index_3D_Valid) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    int64_t expected = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 5; ++k)
                EXPECT_EQ(arr(i, j, k), expected++);
}

TEST_F(TensorIntIndexTest, Index_3D_Modify) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    arr(0, 0, 0) = 999;
    arr(0, 1, 2) = 888;
    arr(1, 2, 3) = 777;
    arr(2, 3, 4) = 666;
    EXPECT_EQ(arr(0, 0, 0), 999);
    EXPECT_EQ(arr(0, 1, 2), 888);
    EXPECT_EQ(arr(1, 2, 3), 777);
    EXPECT_EQ(arr(2, 3, 4), 666);
    EXPECT_EQ(arr(0, 0, 1), 1);
    EXPECT_EQ(arr(2, 2, 2), 52);
}

TEST_F(TensorIntIndexTest, Index_4D_Valid) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    int64_t expected = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < 5; ++l)
                    EXPECT_EQ(arr(i, j, k, l), expected++);
}

TEST_F(TensorIntIndexTest, Index_ReturnsConstRef) {
    const Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    const int64_t& val = arr(0, 0);
    EXPECT_EQ(val, 0);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorIntIndexTest, Index_1D_OOB_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(-1), "out of range");
    EXPECT_DEATH(arr(10), "out of range");
}

TEST_F(TensorIntIndexTest, Index_2D_OOB_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH(arr(-1, 0), "out of range");
    EXPECT_DEATH(arr(4, 0), "out of range");
    EXPECT_DEATH(arr(0, -1), "out of range");
    EXPECT_DEATH(arr(0, 5), "out of range");
}

TEST_F(TensorIntIndexTest, Index_3D_OOB_Death) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    EXPECT_DEATH(arr(3, 0, 0), "out of range");
}

TEST_F(TensorIntIndexTest, Index_4D_OOB_Death) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    EXPECT_DEATH(arr(2, 0, 0, 0), "out of range");
}
#endif

// ============================================================================
// 7. operator() — Reserve
// ============================================================================

class TensorReserveSliceTest : public TensorTest {};

TEST_F(TensorReserveSliceTest, Reserve_1D) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(reserve);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 10);
}

TEST_F(TensorReserveSliceTest, Reserve_2D_Both) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(reserve, reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
}

TEST_F(TensorReserveSliceTest, Reserve_Partial) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(reserve, 2, reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 5);
}

TEST_F(TensorReserveSliceTest, Reserve_3D_All) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(reserve, reserve, reserve);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 4);
    EXPECT_EQ(s.size(2), 5);
}

TEST_F(TensorReserveSliceTest, Reserve_4D_All) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(reserve, reserve, reserve, reserve);
    EXPECT_EQ(s.ndim(), 4);
}

TEST_F(TensorReserveSliceTest, Reserve_4D_Partial) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(1, reserve, 2, reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 5);
}

TEST_F(TensorReserveSliceTest, Reserve_ModifyThroughSlice) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto slice = arr(2, reserve);
    slice.data()[0] = 100;
    EXPECT_EQ(arr(2, 0), 100);
}

// ============================================================================
// 8. operator() — NewAxis
// ============================================================================

class TensorNewAxisTest : public TensorTest {};

TEST_F(TensorNewAxisTest, NewAxis_Default) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(new_axis(), reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s.size(1), 10);
    EXPECT_EQ(s.stride(0), 0);
}

TEST_F(TensorNewAxisTest, NewAxis_CustomRepeat) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(new_axis(3), reserve, reserve);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.stride(0), 0);
}

TEST_F(TensorNewAxisTest, NewAxis_ZeroRepeat) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(new_axis(0), reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 0);
}

TEST_F(TensorNewAxisTest, NewAxis_InMiddle) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(reserve, new_axis(), reserve);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 1);
    EXPECT_EQ(s.size(2), 5);
}

TEST_F(TensorNewAxisTest, NewAxis_AtEnd) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(reserve, reserve, new_axis());
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
    EXPECT_EQ(s.size(2), 1);
}

TEST_F(TensorNewAxisTest, NewAxis_MultipleConsecutive) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(new_axis(2), new_axis(3), reserve);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 2);
    EXPECT_EQ(s.size(1), 3);
    EXPECT_EQ(s.size(2), 10);
}

TEST_F(TensorNewAxisTest, NewAxis_ConstRepeat) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(new_axis(cint<2>), reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 2);
}

// ============================================================================
// 9. operator() — Range
// ============================================================================

class TensorRangeSliceTest : public TensorTest {};

TEST_F(TensorRangeSliceTest, Range_1D_Basic) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(2, 5));
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s(0), 2);
    EXPECT_EQ(s(2), 4);
}

TEST_F(TensorRangeSliceTest, Range_1D_WithStep) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(0, 10, 2));
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 5);
    EXPECT_EQ(s(0), 0);
    EXPECT_EQ(s(1), 2);
    EXPECT_EQ(s(4), 8);
}

TEST_F(TensorRangeSliceTest, Range_1D_NegativeStep) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(9, 0, -1));
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 9);
    EXPECT_EQ(s(0), 9);
    EXPECT_EQ(s(8), 1);
}

TEST_F(TensorRangeSliceTest, Range_1D_NegativeStep_2) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(9, 0, -2));
    EXPECT_EQ(s.size(0), 5);
    EXPECT_EQ(s(0), 9);
    EXPECT_EQ(s(1), 7);
}

TEST_F(TensorRangeSliceTest, Range_1D_SingleElement) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(5, 6));
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s(0), 5);
}

TEST_F(TensorRangeSliceTest, Range_1D_LargeStep) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto s = arr(range(0, 10, 100));
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s(0), 0);
}

TEST_F(TensorRangeSliceTest, Range_2D_OneAxis) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(range(1, 3), reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 2);
    EXPECT_EQ(s.size(1), 5);
}

TEST_F(TensorRangeSliceTest, Range_2D_BothAxes) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(range(1, 3), range(2, 4));
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 2);
    EXPECT_EQ(s.size(1), 2);
    EXPECT_EQ(s(0, 0), 7);   // arr(1,2)
    EXPECT_EQ(s(0, 1), 8);   // arr(1,3)
    EXPECT_EQ(s(1, 0), 12);  // arr(2,2)
    EXPECT_EQ(s(1, 1), 13);  // arr(2,3)

    auto typed = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                             make_strides(cint<5>, cint<1>));
    auto typed_slice = typed(range(1, 3), range(2, 4));
    static_assert(std::same_as<
        stride_type_t<1, typename decltype(typed_slice)::Layout>, Const<1>>);
}

TEST_F(TensorRangeSliceTest, Range_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(reserve, range(1, 3), reserve);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 2);
    EXPECT_EQ(s.size(2), 5);
}

TEST_F(TensorRangeSliceTest, Range_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(reserve, range(1, 3), reserve, range(0, 3));
    EXPECT_EQ(s.ndim(), 4);
    EXPECT_EQ(s.size(0), 2);
    EXPECT_EQ(s.size(1), 2);
    EXPECT_EQ(s.size(2), 4);
    EXPECT_EQ(s.size(3), 3);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorRangeSliceTest, Range_InvalidFrom_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(range(-1, 5)), "range start");
}

TEST_F(TensorRangeSliceTest, Range_InvalidTo_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(range(0, 11)), "range end");
}

TEST_F(TensorRangeSliceTest, Range_ZeroStep_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(range(0, 10, 0)), "step cannot be zero");
}

TEST_F(TensorRangeSliceTest, Range_PosStep_FromGteTo_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(range(5, 5, 1)), "less than");
}

TEST_F(TensorRangeSliceTest, Range_NegStep_FromLteTo_Death) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    EXPECT_DEATH(arr(range(5, 5, -1)), "greater than");
}
#endif

// ============================================================================
// 10. operator() — Mixed Indices
// ============================================================================

class TensorMixedSliceTest : public TensorTest {};

TEST_F(TensorMixedSliceTest, Mixed_IndexAndReserve) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(1, reserve, reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
    EXPECT_EQ(s(0, 0), 20);
}

TEST_F(TensorMixedSliceTest, Mixed_IndexAndRange) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(reserve, range(1, 3), reserve);
    EXPECT_EQ(s.ndim(), 3);
}

TEST_F(TensorMixedSliceTest, Mixed_AllTypes) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(new_axis(), reserve, range(1, 3), 2);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s.size(1), 3);
    EXPECT_EQ(s.size(2), 2);
}

TEST_F(TensorMixedSliceTest, Mixed_ReduceToScalar) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    const int64_t& val = arr(2, 3);
    EXPECT_EQ(val, 13);
}

TEST_F(TensorMixedSliceTest, Mixed_4D_AllTypes) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(new_axis(), reserve, range(1, 3), reserve, 2);
    EXPECT_EQ(s.ndim(), 4);
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s.size(1), 2);
    EXPECT_EQ(s.size(2), 2);
    EXPECT_EQ(s.size(3), 4);
}

TEST_F(TensorMixedSliceTest, Mixed_ModifyThroughSlice) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto slice = arr(1, range(1, 3), reserve);
    slice.data()[0] = 999;
    EXPECT_EQ(arr(1, 1, 0), 999);
}

// ============================================================================
// 11. Transpose Tests
// ============================================================================

class TensorTransposeTest : public TensorTest {};

TEST_F(TensorTransposeTest, TransposeCT_2D_Size) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 6});
    auto t = transpose<0, 1>(arr);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(0), 6);
    EXPECT_EQ(t.size(1), 4);
    EXPECT_EQ(t.stride(0), 1);
    EXPECT_EQ(t.stride(1), 6);
}

TEST_F(TensorTransposeTest, TransposeCT_2D_Values) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 6});
    auto t = transpose<0, 1>(arr);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(arr(i, j), t(j, i));
}

TEST_F(TensorTransposeTest, TransposeCT_3D_Size) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto t = transpose<0, 2>(arr);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.size(0), 5);
    EXPECT_EQ(t.size(1), 4);
    EXPECT_EQ(t.size(2), 3);
}

TEST_F(TensorTransposeTest, TransposeCT_3D_Values) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto t = transpose<1, 2>(arr);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 5; ++k)
                EXPECT_EQ(arr(i, j, k), t(i, k, j));
}

TEST_F(TensorTransposeTest, TransposeCT_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto t = transpose<0, 3>(arr);
    EXPECT_EQ(t.size(0), 5);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.size(2), 4);
    EXPECT_EQ(t.size(3), 2);
}

TEST_F(TensorTransposeTest, TransposeCT_ConstPreserved) {
    auto t1 = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                          make_strides(cint<6>, cint<1>));
    auto t2 = transpose<0, 1>(t1);
    static_assert(decltype(t2)::ct_is_last_contiguous<1> == false);
    EXPECT_TRUE((t2.ct_is_last_contiguous<1>) == false);
    EXPECT_EQ(t2.stride<1>(), 6);
}

TEST_F(TensorTransposeTest, TransposeCT_SameAxis) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto t = transpose<1, 1>(arr);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 4);
    EXPECT_EQ(t.size(2), 5);
}

TEST_F(TensorTransposeTest, TransposeRT_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 6});
    auto t = transpose(arr, 0, 1);
    EXPECT_EQ(t.size(0), 6);
    EXPECT_EQ(t.size(1), 4);
}

TEST_F(TensorTransposeTest, TransposeRT_Values) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 6});
    auto t = transpose(arr, 0, 1);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(arr(i, j), t(j, i));
}

TEST_F(TensorTransposeTest, TransposeRT_SameAxis) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto t = transpose(arr, 1, 1);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 4);
    EXPECT_EQ(t.size(2), 5);
}

TEST_F(TensorTransposeTest, Transpose_Chained) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto t1 = transpose<0, 1>(arr);  // (4,3,5)
    auto t2 = transpose<1, 2>(t1);   // (4,5,3)
    EXPECT_EQ(t2.size(0), 4);
    EXPECT_EQ(t2.size(1), 5);
    EXPECT_EQ(t2.size(2), 3);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorTransposeTest, TransposeRT_InvalidIndex_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH(transpose(arr, -1, 0), "out of range");
    EXPECT_DEATH(transpose(arr, 0, 2), "out of range");
}
#endif

// ============================================================================
// 12. Cross-Type Conversion Tests (as)
// ============================================================================

class TensorAsTest : public TensorTest {};

TEST_F(TensorAsTest, As_ToMatchingConst) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = arr.as<Shape<Const<4>, Any>, Strides<Any, Const<1>>>();
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_TRUE((typed.ct_is_last_contiguous<1>));
}

#ifdef VECOPS_DEBUG
TEST_F(TensorAsTest, As_ToMismatchingConst_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH((arr.as<Shape<Const<8>, Any>, Strides<Any, Any>>()), "!= 8");
}
#endif

TEST_F(TensorAsTest, As_ToAllRuntime) {
    auto t1 = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                          make_strides(cint<5>, cint<1>));
    auto t2 = t1.as<Shape<Any, Any>, Strides<Any, Any>>();
    EXPECT_EQ(t2.ndim(), 2);
    EXPECT_EQ(t2.size(0), 4);
}

TEST_F(TensorAsTest, As_ContiguityAfter) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = arr.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorAsTest, As_AnyToConst_Matching) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = arr.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_EQ(typed.size<1>(), 5);
    EXPECT_EQ(typed.stride<0>(), 5);
    EXPECT_EQ(typed.stride<1>(), 1);
}

TEST_F(TensorAsTest, As_AnyToDynamic) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    using S = Shape<Dynamic<4>, Dynamic<1>>;
    using ST = Strides<Dynamic<1>, Dynamic<1>>;
    auto typed = arr.as<S, ST>();
    EXPECT_EQ(typed.ndim(), 2);
    EXPECT_EQ(typed.size(0), 4);
    EXPECT_EQ(typed.size(1), 5);
}

TEST_F(TensorAsTest, As_DynamicToAny) {
    auto d = make_tensor(data_2d_.data(), make_shape(dyn<4>(4), dyn<1>(5)),
                         make_strides(dyn<1>(5), dyn<1>(1)));
    auto relaxed = d.as<Shape<Any, Any>, Strides<Any, Any>>();
    EXPECT_EQ(relaxed.ndim(), 2);
    EXPECT_EQ(relaxed.size(0), 4);
    EXPECT_EQ(relaxed.size(1), 5);
}

TEST_F(TensorAsTest, As_ConstToDynamic_Conforming) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    using S = Shape<Dynamic<4>, Dynamic<1>>;
    using ST = Strides<Dynamic<1>, Dynamic<1>>;
    auto typed = strict.as<S, ST>();
    EXPECT_EQ(typed.ndim(), 2);
    EXPECT_EQ(typed.size(0), 4);
}

TEST_F(TensorAsTest, As_DynamicToDynamic_Relaxed) {
    auto d = make_tensor(data_2d_.data(), make_shape(dyn<8>(64), dyn<4>(128)),
                         make_strides(dyn<2>(128), dyn<2>(2)));
    using S = Shape<Dynamic<4>, Dynamic<2>>;
    using ST = Strides<Dynamic<1>, Dynamic<1>>;
    auto relaxed = d.as<S, ST>();
    EXPECT_EQ(relaxed.ndim(), 2);
    EXPECT_EQ(relaxed.size(0), 64);
    EXPECT_EQ(relaxed.size(1), 128);
}

TEST_F(TensorAsTest, As_StridesOnly) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    using ST = Strides<Const<5>, Const<1>>;
    auto typed = arr.as<Shape<Any, Any>, ST>();
    EXPECT_EQ(typed.stride<0>(), 5);
    EXPECT_EQ(typed.stride<1>(), 1);
}

TEST_F(TensorAsTest, As_ShapeOnly) {
    auto t = make_tensor(data_2d_.data(), make_shape(Any{4}, Any{5}),
                         make_strides(cint<5>, cint<1>));
    using S = Shape<Const<4>, Const<5>>;
    auto typed = t.as<S, Strides<Any, Any>>();
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_EQ(typed.size<1>(), 5);
}

TEST_F(TensorAsTest, As_SameType_Identity) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto same = arr.as<Shape<Any, Any>, Strides<Any, Any>>();
    EXPECT_EQ(same.size(0), 4);
    EXPECT_EQ(same.size(1), 5);
    EXPECT_EQ(same.stride(0), 5);
    EXPECT_EQ(same.stride(1), 1);
}

TEST_F(TensorAsTest, As_1D) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto typed = arr.as<Shape<Const<10>>, Strides<Const<1>>>();
    EXPECT_EQ(typed.ndim(), 1);
    EXPECT_EQ(typed.size<0>(), 10);
    EXPECT_EQ(typed.stride<0>(), 1);
}

TEST_F(TensorAsTest, As_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    using S = Shape<Const<2>, Const<3>, Const<4>, Const<5>>;
    using ST = Strides<Const<60>, Const<20>, Const<5>, Const<1>>;
    auto typed = arr.as<S, ST>();
    EXPECT_EQ(typed.ndim(), 4);
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorAsTest, As_Chained) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s1 = arr.as<Shape<Const<4>, Any>, Strides<Any, Const<1>>>();
    auto s2 = s1.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();
    EXPECT_TRUE((s2.ct_is_contiguous));
    EXPECT_EQ(s2(0, 0), 0);
}

TEST_F(TensorAsTest, As_VerifyDataPointer) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = arr.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();
    EXPECT_EQ(typed.data(), data_2d_.data());
}

TEST_F(TensorAsTest, As_PartialConst_Mixed) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, Any{5}),
                         make_strides(Any{5}, cint<1>));
    auto typed = t.as<Shape<Const<4>, Any>, Strides<Any, Const<1>>>();
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_EQ(typed.stride<1>(), 1);
}

TEST_F(TensorAsTest, As_3D_ToAllConst) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    using S = Shape<Const<3>, Const<4>, Const<5>>;
    using ST = Strides<Const<20>, Const<5>, Const<1>>;
    auto typed = arr.as<S, ST>();
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorAsTest, As_ValuePreservation) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = arr.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_EQ(typed(i, j), arr(i, j));
}

#ifdef VECOPS_DEBUG
TEST_F(TensorAsTest, As_ConstMismatchSize_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH((arr.as<Shape<Const<4>, Const<9>>, Strides<Any, Any>>()), "!= 9");
}

TEST_F(TensorAsTest, As_ConstMismatchStride_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH((arr.as<Shape<Any, Any>, Strides<Const<3>, Const<1>>>()), "!= 3");
}
#endif

// ============================================================================
// 13. cast() Free Function Tests
// ============================================================================

class TensorCastTest : public TensorTest {};

TEST_F(TensorCastTest, Cast_ConstToAny_2D) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    auto relaxed = cast<Shape<Any, Any>, Strides<Any, Any>>(strict);
    EXPECT_EQ(relaxed.ndim(), 2);
    EXPECT_EQ(relaxed.size(0), 4);
    EXPECT_EQ(relaxed.size(1), 5);
    EXPECT_EQ(relaxed(0, 0), 0);
}

TEST_F(TensorCastTest, Cast_AnyToConst_Matching) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = cast<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>(arr);
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_EQ(typed.size<1>(), 5);
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorCastTest, Cast_ConstToDynamic) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    using S = Shape<Dynamic<4>, Dynamic<1>>;
    using ST = Strides<Dynamic<1>, Dynamic<1>>;
    auto typed = cast<S, ST>(strict);
    EXPECT_EQ(typed.ndim(), 2);
    EXPECT_EQ(typed.size(0), 4);
}

TEST_F(TensorCastTest, Cast_DynamicToAny) {
    auto d = make_tensor(data_1d_.data(), make_shape(dyn<2>(10)),
                         make_strides(cint<1>));
    auto relaxed = cast<Shape<Any>, Strides<Any>>(d);
    EXPECT_EQ(relaxed.ndim(), 1);
    EXPECT_EQ(relaxed.size(0), 10);
}

TEST_F(TensorCastTest, Cast_1D) {
    Array<int64_t, 1> arr(data_1d_.data(), {10});
    auto typed = cast<Shape<Const<10>>, Strides<Const<1>>>(arr);
    EXPECT_EQ(typed.ndim(), 1);
    EXPECT_EQ(typed.size<0>(), 10);
    EXPECT_EQ(typed(5), 5);
}

TEST_F(TensorCastTest, Cast_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto typed = cast<Shape<Const<3>, Const<4>, Const<5>>,
                      Strides<Const<20>, Const<5>, Const<1>>>(arr);
    EXPECT_EQ(typed.ndim(), 3);
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorCastTest, Cast_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    using S = Shape<Const<2>, Const<3>, Const<4>, Const<5>>;
    using ST = Strides<Const<60>, Const<20>, Const<5>, Const<1>>;
    auto typed = cast<S, ST>(arr);
    EXPECT_EQ(typed.ndim(), 4);
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorCastTest, Cast_StridesOnly) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                         make_strides(Any{5}, Any{1}));
    auto typed = cast<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>(t);
    EXPECT_EQ(typed.stride<0>(), 5);
    EXPECT_EQ(typed.stride<1>(), 1);
    EXPECT_TRUE((typed.ct_is_contiguous));
}

TEST_F(TensorCastTest, Cast_EquivalenceWithAs) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    using S = Shape<Const<4>, Const<5>>;
    using ST = Strides<Const<5>, Const<1>>;
    auto r1 = cast<S, ST>(arr);
    auto r2 = arr.as<S, ST>();
    EXPECT_EQ(r1.size<0>(), r2.size<0>());
    EXPECT_EQ(r1.size<1>(), r2.size<1>());
    EXPECT_EQ(r1(0, 0), r2(0, 0));
}

TEST_F(TensorCastTest, Cast_Chained) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    auto r1 = cast<Shape<Any, Any>, Strides<Any, Any>>(strict);
    auto r2 = cast<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>(r1);
    EXPECT_TRUE((r2.ct_is_contiguous));
}

TEST_F(TensorCastTest, Cast_VerifyData) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto typed = cast<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>(arr);
    EXPECT_EQ(typed.data(), data_2d_.data());
}

TEST_F(TensorCastTest, Cast_PartialConst) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, Any{5}),
                         make_strides(cint<5>, cint<1>));
    auto typed = cast<Shape<Const<4>, Any>, Strides<Any, Const<1>>>(t);
    EXPECT_EQ(typed.size<0>(), 4);
    EXPECT_EQ(typed.stride<1>(), 1);
}

TEST_F(TensorCastTest, Cast_DynamicToDynamic_Relaxed) {
    auto d = make_tensor(data_2d_.data(), make_shape(dyn<8>(64), dyn<4>(128)),
                         make_strides(dyn<2>(128), cint<1>));
    using S = Shape<Dynamic<4>, Dynamic<2>>;
    using ST = Strides<Dynamic<1>, Dynamic<1>>;
    auto relaxed = cast<S, ST>(d);
    EXPECT_EQ(relaxed.ndim(), 2);
    EXPECT_EQ(relaxed.size(0), 64);
    EXPECT_EQ(relaxed.size(1), 128);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorCastTest, Cast_MismatchConst_Death) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_DEATH(
        (cast<Shape<Const<8>, Any>, Strides<Any, Any>>(arr)),
        "!= 8");
}
#endif

// ============================================================================
// 13. Data Type Tests
// ============================================================================

class TensorDataTypeTest : public ::testing::Test {};

TEST_F(TensorDataTypeTest, DataType_Float) {
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);
    Array<float, 2> arr(data.data(), {4, 5});
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_FLOAT_EQ(arr(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(arr(1, 2), 7.0f);
}

TEST_F(TensorDataTypeTest, DataType_Double) {
    std::vector<double> data(20);
    std::iota(data.begin(), data.end(), 0.0);
    Array<double, 2> arr(data.data(), {4, 5});
    EXPECT_DOUBLE_EQ(arr(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(arr(2, 3), 13.0);
}

TEST_F(TensorDataTypeTest, DataType_Int32) {
    std::vector<int32_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    Array<int32_t, 2> arr(data.data(), {4, 5});
    EXPECT_EQ(arr(3, 4), 19);
}

TEST_F(TensorDataTypeTest, DataType_Int8) {
    std::vector<int8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    Array<int8_t, 2> arr(data.data(), {4, 5});
    EXPECT_EQ(arr(1, 4), 9);
}

// ============================================================================
// 14. High-Dimension Tests (5D-8D)
// ============================================================================

class TensorHighDimTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_5d_.resize(720);
        std::iota(data_5d_.begin(), data_5d_.end(), 0);
    }
    std::vector<int64_t> data_5d_;
};

TEST_F(TensorHighDimTest, Construct_5D) {
    Array<int64_t, 5> arr(data_5d_.data(), {2, 3, 4, 5, 6});
    EXPECT_EQ(arr.ndim(), 5);
    EXPECT_EQ(arr.numel(), 720);
    EXPECT_EQ(arr.size(0), 2);
    EXPECT_EQ(arr.size(4), 6);
    EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(TensorHighDimTest, Construct_6D) {
    Array<int64_t, 6> arr(data_5d_.data(), {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(arr.ndim(), 6);
    EXPECT_EQ(arr.numel(), 720);
}

TEST_F(TensorHighDimTest, Construct_7D) {
    Array<int64_t, 7> arr(data_5d_.data(), {1, 1, 2, 3, 4, 5, 6});
    EXPECT_EQ(arr.ndim(), 7);
    EXPECT_EQ(arr.numel(), 720);
}

TEST_F(TensorHighDimTest, Construct_8D) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    EXPECT_EQ(arr.ndim(), 8);
    EXPECT_EQ(arr.numel(), 720);
}

TEST_F(TensorHighDimTest, Size_Access_8D) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    EXPECT_EQ(arr.size(3), 2);
    EXPECT_EQ(arr.size(4), 3);
    EXPECT_EQ(arr.size(5), 4);
    EXPECT_EQ(arr.size(6), 5);
    EXPECT_EQ(arr.size(7), 6);
}

TEST_F(TensorHighDimTest, Stride_Access_8D) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    EXPECT_EQ(arr.stride(3), 360);
    EXPECT_EQ(arr.stride(4), 120);
    EXPECT_EQ(arr.stride(5), 30);
    EXPECT_EQ(arr.stride(6), 6);
    EXPECT_EQ(arr.stride(7), 1);
}

TEST_F(TensorHighDimTest, Index_8D_Valid) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    const int64_t& v = arr(0, 0, 0, 1, 2, 3, 4, 5);
    EXPECT_EQ(v, 1 * 360 + 2 * 120 + 3 * 30 + 4 * 6 + 5);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorHighDimTest, Index_8D_OOB_Death) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    EXPECT_DEATH(arr(0, 0, 0, 2, 0, 0, 0, 0), "out of range");
}
#endif

TEST_F(TensorHighDimTest, Reserve_8D_All) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    auto s = arr(reserve, reserve, reserve, reserve, reserve, reserve, reserve, reserve);
    EXPECT_EQ(s.ndim(), 8);
}

TEST_F(TensorHighDimTest, Reserve_8D_Partial) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    auto s = arr(0, 0, reserve, reserve, reserve, 0, reserve, 0);
    EXPECT_EQ(s.ndim(), 4);
}

TEST_F(TensorHighDimTest, TransposeCT_8D) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    auto t = transpose<3, 6>(arr);
    EXPECT_EQ(t.size(3), 5);
    EXPECT_EQ(t.size(6), 2);
}

TEST_F(TensorHighDimTest, As_8D_Conversion) {
    Array<int64_t, 8> arr(data_5d_.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    using S = Shape<Any, Any, Any, Any, Any, Any, Any, Any>;
    using T = Strides<Any, Any, Any, Any, Any, Any, Any, Any>;
    auto c = arr.as<S, T>();
    EXPECT_EQ(c.ndim(), 8);
}

// ============================================================================
// 15. constexpr Tests
// ============================================================================

#ifndef VECOPS_DEBUG

class TensorConstexprTest : public ::testing::Test {};

TEST_F(TensorConstexprTest, Constexpr_MakeTensor) {
    static float d[24]{};
    constexpr auto t = make_tensor(d, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<6>, cint<1>));
    EXPECT_EQ(t.ndim(), 2);
}

TEST_F(TensorConstexprTest, Constexpr_Size) {
    static float d[24]{};
    constexpr auto t = make_tensor(d, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<6>, cint<1>));
    constexpr auto sz0 = t.size<0>();
    constexpr auto sz1 = t.size<1>();
    static_assert(std::same_as<decltype(sz0), const meta::Const<4>>);
    static_assert(std::same_as<decltype(sz1), const meta::Const<6>>);
    EXPECT_EQ(sz0, 4);
    EXPECT_EQ(sz1, 6);
}

TEST_F(TensorConstexprTest, Constexpr_Stride) {
    static float d[24]{};
    constexpr auto t = make_tensor(d, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<6>, cint<1>));
    constexpr auto st0 = t.stride<0>();
    constexpr auto st1 = t.stride<1>();
    static_assert(std::same_as<decltype(st0), const meta::Const<6>>);
    static_assert(std::same_as<decltype(st1), const meta::Const<1>>);
    EXPECT_EQ(st0, 6);
    EXPECT_EQ(st1, 1);
}

TEST_F(TensorConstexprTest, Constexpr_Ndim) {
    static float d[24]{};
    constexpr auto t = make_tensor(d, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<6>, cint<1>));
    constexpr int nd = t.ndim();
    EXPECT_EQ(nd, 2);
}

TEST_F(TensorConstexprTest, StaticAssert_CtIsContiguous) {
    constexpr auto t = make_tensor((float*)nullptr, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<6>, cint<1>));
    static_assert(t.ct_is_contiguous, "");
    static_assert(t.ct_is_last_contiguous<2>, "");
}

TEST_F(TensorConstexprTest, StaticAssert_NotContiguous) {
    constexpr auto t = make_tensor((float*)nullptr, make_shape(cint<4>, cint<6>),
                                   make_strides(cint<8>, cint<2>));
    static_assert(!t.ct_is_last_contiguous<1>, "");
}

TEST_F(TensorConstexprTest, Constexpr_TransposeCT) {
    constexpr auto t1 = make_tensor((float*)nullptr, make_shape(cint<4>, cint<6>),
                                    make_strides(cint<6>, cint<1>));
    constexpr auto t2 = transpose<0, 1>(t1);
    constexpr auto sz0 = t2.size<0>();
    constexpr auto sz1 = t2.size<1>();
    EXPECT_EQ(sz0, 6);
    EXPECT_EQ(sz1, 4);
}

TEST_F(TensorConstexprTest, Constexpr_Layout_Size) {
    constexpr auto layout = make_layout(make_shape(cint<4>, cint<6>),
                                        make_strides(cint<6>, cint<1>));
    constexpr auto s0 = size<0>(layout);
    constexpr auto s1 = size<1>(layout);
    static_assert(std::same_as<decltype(s0), const meta::Const<4>>);
    static_assert(std::same_as<decltype(s1), const meta::Const<6>>);
    EXPECT_EQ(s0, 4);
    EXPECT_EQ(s1, 6);
}

TEST_F(TensorConstexprTest, Constexpr_Layout_IsCtContiguous) {
    constexpr auto layout = make_layout(make_shape(cint<4>, cint<6>),
                                        make_strides(cint<6>, cint<1>));
    constexpr bool v = is_ct_last_contiguous_v<decltype(layout), 2>;
    EXPECT_TRUE(v);
}

#endif  // VECOPS_DEBUG

// ============================================================================
// 16. Continuity Propagation Tests
// ============================================================================

class TensorContiguityPropagationTest : public TensorTest {};

TEST_F(TensorContiguityPropagationTest, Reserve_Preserves_Contiguity) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                         make_strides(cint<6>, cint<1>));
    auto s = t(reserve, reserve);
    EXPECT_TRUE((s.ct_is_contiguous));
}

TEST_F(TensorContiguityPropagationTest, Reserve_Preserves_StrideConst) {
    auto t = make_tensor(data_2d_.data(), make_shape(4, 6),
                         make_strides(Any{6}, cint<1>));
    auto s = t(reserve, reserve);
    EXPECT_TRUE((s.ct_is_last_contiguous<1>));
}

TEST_F(TensorContiguityPropagationTest, Index_Breaks_Contiguity) {
    auto t = make_tensor(data_3d_.data(), make_shape(cint<3>, cint<4>, cint<5>),
                         make_strides(cint<20>, cint<5>, cint<1>));
    auto s = t(1, reserve, reserve);
    EXPECT_TRUE((s.ct_is_contiguous));
}

TEST_F(TensorContiguityPropagationTest, ChainedReserve_KeepsConst) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                         make_strides(cint<6>, cint<1>));
    auto s1 = t(reserve, reserve);
    auto s2 = s1(reserve, reserve);
    EXPECT_TRUE((s2.ct_is_contiguous));
}

TEST_F(TensorContiguityPropagationTest, Range_Breaks_Contiguity) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(range(1, 3), range(1, 4, 2));
    EXPECT_FALSE(s.is_contiguous());
}

TEST_F(TensorContiguityPropagationTest, NewAxis_Breaks_Contiguity) {
    auto t = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<6>),
                         make_strides(cint<6>, cint<1>));
    auto s = t(new_axis(), reserve, reserve);
    EXPECT_FALSE((s.ct_is_contiguous));
}

TEST_F(TensorContiguityPropagationTest, SliceThenTranspose) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(0, reserve, reserve);
    auto t = transpose<0, 1>(s);
    EXPECT_EQ(t.size(0), 5);
}

// ============================================================================
// 17. Method Boundary Tests
// ============================================================================

class TensorBoundaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_2d_.resize(20);
        std::iota(data_2d_.begin(), data_2d_.end(), 0);
        data_8d_.resize(256);
        std::iota(data_8d_.begin(), data_8d_.end(), 0);
    }
    std::vector<int64_t> data_2d_;
    std::vector<int64_t> data_8d_;
};

TEST_F(TensorBoundaryTest, Numel_AllSizeOne) {
    int64_t data[1] = {42};
    Array<int64_t, 4> arr(data, {1, 1, 1, 1});
    EXPECT_EQ(arr.numel(), 1);
}

TEST_F(TensorBoundaryTest, Numel_MixedSizeOne) {
    int64_t data[100];
    Array<int64_t, 4> arr(data, {1, 100, 1, 1});
    EXPECT_EQ(arr.numel(), 100);
}

TEST_F(TensorBoundaryTest, Numel_Large) {
    std::vector<int64_t> data(1000000);
    Array<int64_t, 2> arr(data.data(), {1000, 1000});
    EXPECT_EQ(arr.numel(), 1000000);
}

TEST_F(TensorBoundaryTest, Numel_ZeroSize) {
    Array<int64_t, 2> arr(data_2d_.data(), {0, 5});
    EXPECT_EQ(arr.numel(), 0);
}

TEST_F(TensorBoundaryTest, Numel_4D_Product) {
    int64_t data[360];
    Array<int64_t, 4> arr(data, {3, 4, 5, 6});
    EXPECT_EQ(arr.numel(), 360);
}

TEST_F(TensorBoundaryTest, Numel_8D_Product) {
    Array<int64_t, 8> arr(data_8d_.data(), {2, 2, 2, 2, 2, 2, 2, 2});
    EXPECT_EQ(arr.numel(), 256);
}

TEST_F(TensorBoundaryTest, Ndim_8D) {
    Array<int64_t, 8> arr(data_8d_.data(), {2, 2, 2, 2, 2, 2, 2, 2});
    EXPECT_EQ(arr.ndim(), 8);
}

TEST_F(TensorBoundaryTest, Data_Ptr_Unchanged_Reserve) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(reserve, reserve);
    EXPECT_EQ(s.data(), arr.data());
}

TEST_F(TensorBoundaryTest, Data_Ptr_Offset_Index) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(1, reserve);
    EXPECT_EQ(s.data(), arr.data() + 5);
}

TEST_F(TensorBoundaryTest, Data_Ptr_Offset_Range) {
    Array<int64_t, 1> arr(data_2d_.data(), {10});
    auto s = arr(range(3, 8));
    EXPECT_EQ(s.data(), arr.data() + 3);
}

TEST_F(TensorBoundaryTest, BracketOp_1D) {
    Array<int64_t, 1> arr(data_2d_.data(), {10});
    EXPECT_EQ(arr[3], arr(3));
}

TEST_F(TensorBoundaryTest, BracketOp_2D_Chained) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto sub = arr(1, reserve);
    EXPECT_EQ(sub[2], arr(1, 2));
}

TEST_F(TensorBoundaryTest, Size_Access_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    EXPECT_EQ(arr.size(0), 4);
    EXPECT_EQ(arr.size(1), 5);
}

TEST_F(TensorBoundaryTest, Size_Access_8D) {
    Array<int64_t, 8> arr(data_8d_.data(), {2, 2, 2, 2, 2, 2, 2, 2});
    for (int i = 0; i < 8; ++i) EXPECT_EQ(arr.size(i), 2);
}

TEST_F(TensorBoundaryTest, Stride_Access_8D) {
    Array<int64_t, 8> arr(data_8d_.data(), {2, 2, 2, 2, 2, 2, 2, 2});
    EXPECT_EQ(arr.stride(7), 1);
    EXPECT_EQ(arr.stride(6), 2);
    EXPECT_EQ(arr.stride(5), 4);
    EXPECT_EQ(arr.stride(4), 8);
    EXPECT_EQ(arr.stride(3), 16);
    EXPECT_EQ(arr.stride(2), 32);
    EXPECT_EQ(arr.stride(1), 64);
    EXPECT_EQ(arr.stride(0), 128);
}

TEST_F(TensorBoundaryTest, Edge_NegativeStride) {
    int64_t data[10];
    std::iota(data, data + 10, 0);
    Array<int64_t, 1> arr(data + 5, {6}, {-1});
    EXPECT_EQ(arr(0), 5);
    EXPECT_EQ(arr(1), 4);
    EXPECT_EQ(arr(5), 0);
}

TEST_F(TensorBoundaryTest, Edge_BroadcastStride) {
    int64_t data[5];
    std::iota(data, data + 5, 0);
    Array<int64_t, 2> arr(data, {3, 5}, {0, 1});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_EQ(arr(i, j), j);
}

TEST_F(TensorBoundaryTest, Edge_StridedView) {
    Array<int64_t, 1> arr(data_2d_.data(), {5}, {2});
    EXPECT_EQ(arr(0), 0);
    EXPECT_EQ(arr(1), 2);
    EXPECT_EQ(arr(4), 8);
}

TEST_F(TensorBoundaryTest, Edge_LargeDim_1D) {
    std::vector<int64_t> data(1000000);
    std::iota(data.begin(), data.end(), 0);
    Array<int64_t, 1> arr(data.data(), {1000000});
    EXPECT_EQ(arr(999999), 999999);
}

// ============================================================================
// 18. Implicit Conversion Tests
// ============================================================================

class TensorImplicitConversionTest : public TensorTest {};

TEST_F(TensorImplicitConversionTest, ConstToAny_AllDims) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    Array<int64_t, 2> arr = strict;
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 4);
    EXPECT_EQ(arr.size(1), 5);
    EXPECT_EQ(arr.stride(0), 5);
    EXPECT_EQ(arr.stride(1), 1);
    EXPECT_EQ(arr(0, 0), 0);
}

TEST_F(TensorImplicitConversionTest, ConstToAny_Partial) {
    auto mixed = make_tensor(data_2d_.data(), make_shape(cint<4>, Any{5}),
                             make_strides(cint<5>, cint<1>));
    Tensor<int64_t, Shape<Any, Any>, Strides<Any, Any>> arr = mixed;
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 4);
    EXPECT_EQ(arr.size(1), 5);
}

TEST_F(TensorImplicitConversionTest, ConstToDynamic_Conforming) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    Tensor<int64_t, Shape<Dynamic<4>, Dynamic<1>>,
           Strides<Dynamic<1>, Dynamic<1>>> arr = strict;
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 4);
}

TEST_F(TensorImplicitConversionTest, ConstToConst_DirectAssignment) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    auto same = strict;
    EXPECT_EQ(same.size<0>(), 4);
    EXPECT_TRUE((same.ct_is_contiguous));
}

TEST_F(TensorImplicitConversionTest, DynamicToAny) {
    auto d = make_tensor(data_2d_.data(), make_shape(dyn<4>(4), dyn<1>(5)),
                         make_strides(dyn<1>(5), dyn<1>(1)));
    Array<int64_t, 2> arr = d;
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 4);
}

TEST_F(TensorImplicitConversionTest, FullyConstToArray_3D) {
    auto strict = make_tensor(data_3d_.data(),
                              make_shape(cint<3>, cint<4>, cint<5>),
                              make_strides(cint<20>, cint<5>, cint<1>));
    Array<int64_t, 3> arr = strict;
    EXPECT_EQ(arr.ndim(), 3);
    EXPECT_EQ(arr.numel(), 60);
    EXPECT_EQ(arr(2, 3, 4), 2 * 20 + 3 * 5 + 4);
}

TEST_F(TensorImplicitConversionTest, ConstToDynamic_SameAlignment) {
    auto strict = make_tensor(data_1d_.data(), make_shape(cint<10>),
                              make_strides(cint<1>));
    Tensor<int64_t, Shape<Dynamic<2>>, Strides<Dynamic<1>>> arr = strict;
    EXPECT_EQ(arr.size(0), 10);
}

TEST_F(TensorImplicitConversionTest, DynamicToMoreLenientDynamic) {
    auto d = make_tensor(data_2d_.data(), make_shape(dyn<8>(64), dyn<4>(128)),
                         make_strides(dyn<2>(128), cint<1>));
    Tensor<int64_t, Shape<Dynamic<4>, Dynamic<2>>,
           Strides<Dynamic<1>, Dynamic<1>>> arr = d;
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(0), 64);
    EXPECT_EQ(arr.size(1), 128);
}

TEST_F(TensorImplicitConversionTest, FunctionParameter_Implicit) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    auto fn = [](Array<int64_t, 2> arr) {
        return arr.size(0);
    };
    EXPECT_EQ(fn(strict), 4);
}

TEST_F(TensorImplicitConversionTest, FullConstToDynamic_Bounded) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    using S = Shape<Dynamic<2, 0, 16>, Dynamic<1, 0, 16>>;
    using T = Strides<Dynamic<1, 0, 16>, Dynamic<1, 0, 16>>;
    Tensor<int64_t, S, T> arr = strict;
    EXPECT_EQ(arr.ndim(), 2);
}

TEST_F(TensorImplicitConversionTest, AnyToAny_DirectSameType) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    Array<int64_t, 2> arr2 = arr;
    EXPECT_EQ(arr2.size(0), 4);
}

TEST_F(TensorImplicitConversionTest, ConstToAny_4D) {
    auto strict = make_tensor(data_4d_.data(),
                              make_shape(cint<2>, cint<3>, cint<4>, cint<5>),
                              make_strides(cint<60>, cint<20>, cint<5>, cint<1>));
    Array<int64_t, 4> arr = strict;
    EXPECT_EQ(arr.ndim(), 4);
    EXPECT_EQ(arr.numel(), 120);
    EXPECT_EQ(arr(1, 2, 3, 4), 1 * 60 + 2 * 20 + 3 * 5 + 4);
}

#ifdef VECOPS_DEBUG
TEST_F(TensorImplicitConversionTest, As_NonConforming_Death) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    using BadShape = Shape<Dynamic<8>, Any>;
    using BadStrides = Strides<Any, Any>;
    EXPECT_DEATH((strict.template as<BadShape, BadStrides>()), "constraints");
}
#endif

// ============================================================================
// 19. Ellipsis Slicing Tests
// ============================================================================

class TensorEllipsisTest : public TensorTest {};

TEST_F(TensorEllipsisTest, Marker_Exists) {
    auto e = ellipsis;
    (void)e;
}

TEST_F(TensorEllipsisTest, EllipsisOnly_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(ellipsis);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
}

TEST_F(TensorEllipsisTest, EllipsisOnly_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(ellipsis);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 4);
    EXPECT_EQ(s.size(2), 5);
}

TEST_F(TensorEllipsisTest, EllipsisOnly_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(ellipsis);
    EXPECT_EQ(s.ndim(), 4);
    EXPECT_EQ(s.numel(), 120);
}

TEST_F(TensorEllipsisTest, EllipsisTail_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(ellipsis, 2);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s(0), arr(0, 2));
    EXPECT_EQ(s(3), arr(3, 2));
}

TEST_F(TensorEllipsisTest, EllipsisHead_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(1, ellipsis);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 5);
    EXPECT_EQ(s(2), arr(1, 2));
}

TEST_F(TensorEllipsisTest, EllipsisMiddle_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(1, ellipsis, 3);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s(2), arr(1, 2, 3));
}

TEST_F(TensorEllipsisTest, EllipsisMiddle_4D) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s = arr(0, ellipsis, 3);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 4);
    EXPECT_EQ(s(2, 1), arr(0, 2, 1, 3));
}

TEST_F(TensorEllipsisTest, EllipsisWithRange) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(ellipsis, range(1, 4));
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 4);
    EXPECT_EQ(s.size(2), 3);
    EXPECT_EQ(s(0, 0, 0), arr(0, 0, 1));
    EXPECT_EQ(s(0, 3, 2), arr(0, 3, 3));
}

TEST_F(TensorEllipsisTest, EllipsisWithNewAxis) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(new_axis(), ellipsis);
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s.size(1), 4);
    EXPECT_EQ(s.size(2), 5);
    EXPECT_EQ(s(0, 1, 2), arr(1, 2));
}

TEST_F(TensorEllipsisTest, EllipsisTailWithNewAxis) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(ellipsis, new_axis());
    EXPECT_EQ(s.ndim(), 3);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
    EXPECT_EQ(s.size(2), 1);
    EXPECT_EQ(s(1, 2, 0), arr(1, 2));
}

TEST_F(TensorEllipsisTest, EllipsisAllIntegers_ResultScalar_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    const auto& val = arr(1, ellipsis, 2, 3);
    EXPECT_EQ(val, arr(1, 2, 3));
    EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(val)>, int64_t>));
}

TEST_F(TensorEllipsisTest, EllipsisAllIntegers_ResultScalar_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    const auto& val = arr(1, ellipsis, 3);
    EXPECT_EQ(val, arr(1, 3));
}

TEST_F(TensorEllipsisTest, EllipsisWithSingleInteger_ReturnsTensor_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(1, ellipsis, 2);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s(0), arr(1, 0, 2));
    EXPECT_EQ(s(3), arr(1, 3, 2));
}

TEST_F(TensorEllipsisTest, EllipsisNoGap_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(ellipsis, reserve);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
    EXPECT_EQ(s(2, 4), arr(2, 4));
}

TEST_F(TensorEllipsisTest, EllipsisNoGapHead_2D) {
    Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
    auto s = arr(reserve, ellipsis);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.size(0), 4);
    EXPECT_EQ(s.size(1), 5);
    EXPECT_EQ(s(0, 3), arr(0, 3));
}

TEST_F(TensorEllipsisTest, EllipsisWithZeroFill_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(2, 3, ellipsis);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 5);
    EXPECT_EQ(s(4), arr(2, 3, 4));
}

TEST_F(TensorEllipsisTest, EllipsisHeadZeroFill_3D) {
    Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
    auto s = arr(ellipsis, 3, 4);
    EXPECT_EQ(s.ndim(), 1);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s(2), arr(2, 3, 4));
}

TEST_F(TensorEllipsisTest, EllipsisChainedSlice) {
    Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
    auto s1 = arr(0, ellipsis);
    EXPECT_EQ(s1.ndim(), 3);
    auto s2 = s1(ellipsis, 2);
    EXPECT_EQ(s2.ndim(), 2);
    EXPECT_EQ(s2.size(0), 3);
    EXPECT_EQ(s2.size(1), 4);
    EXPECT_EQ(s2(1, 3), arr(0, 1, 3, 2));
}

TEST_F(TensorEllipsisTest, EllipsisPreservesConst) {
    auto strict = make_tensor(data_2d_.data(), make_shape(cint<4>, cint<5>),
                              make_strides(cint<5>, cint<1>));
    auto s = strict(ellipsis);
    EXPECT_EQ(s.size<0>(), 4);
    EXPECT_EQ(s.size<1>(), 5);
    EXPECT_TRUE((s.ct_is_contiguous));
}

TEST_F(TensorEllipsisTest, EllipsisPreservesContiguity) {
    auto t = make_tensor(data_3d_.data(),
                         make_shape(cint<3>, cint<4>, cint<5>),
                         make_strides(cint<20>, cint<5>, cint<1>));
    auto s = t(1, ellipsis);
    EXPECT_TRUE((s.ct_is_contiguous));
}

TEST_F(TensorEllipsisTest, EllipsisWithConstShape_8D) {
    std::vector<int64_t> data(720);
    std::iota(data.begin(), data.end(), 0);
    Array<int64_t, 8> arr(data.data(), {1, 1, 1, 2, 3, 4, 5, 6});
    auto s = arr(0, 0, ellipsis, 5);
    EXPECT_EQ(s.ndim(), 5);
    EXPECT_EQ(s.size(0), 1);
    EXPECT_EQ(s.size(1), 2);
    EXPECT_EQ(s.size(2), 3);
    EXPECT_EQ(s.size(3), 4);
    EXPECT_EQ(s.size(4), 5);
}

TEST_F(TensorEllipsisTest, TakeLeadingAndTrailingUseZeroDiscardedAxes) {
    Array<int64_t, 3> tensor(data_3d_.data(), {3, 4, 5});
    auto leading = take_leading<2>(tensor);
    auto trailing = take_trailing<2>(tensor);

    static_assert(decltype(leading)::Ndim == 2);
    static_assert(decltype(trailing)::Ndim == 2);
    EXPECT_EQ(leading.data(), tensor.data());
    EXPECT_EQ(trailing.data(), tensor.data());
    EXPECT_EQ(leading(2, 3), tensor(2, 3, 0));
    EXPECT_EQ(trailing(2, 4), tensor(0, 2, 4));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
