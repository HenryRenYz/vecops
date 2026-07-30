#include <array>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/util/ScalarConvert.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::vec;

namespace {

template <typename T>
struct CoordShiftFn : VecTransform<T, T> {
  int* calls{};
  std::vector<nint_t>* last_coords{};

  CoordShiftFn(int* calls_, std::vector<nint_t>* last_coords_)
      : calls(calls_), last_coords(last_coords_) {}

  template <VectorTag To, typename... Coords>
    requires is_scalable_tag<To> &&
             std::same_as<ElementOf<To>, T> &&
             (CoordShiftFn::min_output_pow2 <= scale_power<To>) &&
             (scale_power<To> <= CoordShiftFn::max_output_pow2)
  Vec<To> operator()(To t, Vec<To> v, Coords... coords) const {
    if (calls != nullptr) ++*calls;
    nint_t encoded = 0;
    ((encoded = encoded * 100 + static_cast<nint_t>(coords)), ...);
    if (last_coords != nullptr) {
      *last_coords = {static_cast<nint_t>(coords)...};
    }
    return add(t, v, fill(t, static_cast<T>(encoded)));
  }
};

template <typename T, typename... Coords>
T coord_shift_expected(T value, Coords... coords) {
  nint_t encoded = 0;
  ((encoded = encoded * 100 + static_cast<nint_t>(coords)), ...);
  return static_cast<T>(value + static_cast<T>(encoded));
}

template <typename T>
std::vector<T> make_source(nint_t n) {
  std::vector<T> data(static_cast<size_t>(n));
  for (nint_t i = 0; i < n; ++i) {
    data[static_cast<size_t>(i)] = static_cast<T>((i % 97) + 1);
  }
  return data;
}

template <typename T>
std::vector<std::byte> make_aux(nint_t bytes) {
  std::vector<std::byte> aux(static_cast<size_t>(bytes + DEFAULT_ALIGNMENT));
  std::fill(aux.begin(), aux.end(), std::byte{0});
  return aux;
}

template <typename T>
T read_vec_lane(Vec<ScalableTag<T, 0>> v, nint_t lane) {
  return vec::get(ScalableTag<T, 0>{}, v, lane);
}

template <typename TIn_, typename TOut_, int PowIn_, int PowOut_>
struct DataAccessConversionCase {
  using TIn = TIn_;
  using TOut = TOut_;
  using InTag = ScalableTag<TIn, PowIn_>;
  using OutTag = ScalableTag<TOut, PowOut_>;
  static constexpr const char* Kind = "C";
  static constexpr int InPow = PowIn_;
  static constexpr int OutPow = PowOut_;
};

template <typename T1_, typename T2_, int PowIn_>
struct DataAccessPromoteCase {
  using TIn = T1_;
  using TOut = T2_;
  using InTag = ScalableTag<TIn, PowIn_>;
  using OutTag = Rebind<TOut, InTag>;
  static constexpr const char* Kind = "P";
  static constexpr int InPow = PowIn_;
  static constexpr int OutPow = scale_power<OutTag>;
};

template <typename T1_, typename T2_, int PowOut_>
struct DataAccessDemoteCase {
  using TIn = T1_;
  using TOut = T2_;
  using OutTag = ScalableTag<TOut, PowOut_>;
  using InTag = Rebind<TIn, OutTag>;
  static constexpr const char* Kind = "D";
  static constexpr int InPow = scale_power<InTag>;
  static constexpr int OutPow = PowOut_;
};

template <typename T>
struct TypeLabel;
template <> struct TypeLabel<int8_t> { static constexpr const char* value = "i8"; };
template <> struct TypeLabel<uint8_t> { static constexpr const char* value = "u8"; };
template <> struct TypeLabel<int16_t> { static constexpr const char* value = "i16"; };
template <> struct TypeLabel<uint16_t> { static constexpr const char* value = "u16"; };
template <> struct TypeLabel<vecops::float16_t> { static constexpr const char* value = "f16"; };
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
template <> struct TypeLabel<vecops::bfloat16_t> { static constexpr const char* value = "bf16"; };
#endif
template <> struct TypeLabel<int32_t> { static constexpr const char* value = "i32"; };
template <> struct TypeLabel<uint32_t> { static constexpr const char* value = "u32"; };
template <> struct TypeLabel<float32_t> { static constexpr const char* value = "f32"; };
template <> struct TypeLabel<int64_t> { static constexpr const char* value = "i64"; };
template <> struct TypeLabel<uint64_t> { static constexpr const char* value = "u64"; };
template <> struct TypeLabel<float64_t> { static constexpr const char* value = "f64"; };

inline void append_pow(std::string& name, int pow) {
  name += pow < 0 ? "m" : "p";
  name += std::to_string(pow < 0 ? -pow : pow);
}

struct DataAccessConversionCaseNames {
  template <typename TCase>
  static std::string GetName(int) {
    std::string name = TCase::Kind;
    name += "_";
    name += TypeLabel<typename TCase::TIn>::value;
    name += "_";
    name += TypeLabel<typename TCase::TOut>::value;
    name += "_i";
    append_pow(name, TCase::InPow);
    name += "_o";
    append_pow(name, TCase::OutPow);
    return name;
  }
};

#define DATA_ACCESS_PROMOTE_SHIFT1(T1, T2) \
  DataAccessPromoteCase<T1, T2, 0>, DataAccessPromoteCase<T1, T2, -1>
#define DATA_ACCESS_DEMOTE_SHIFT1(T1, T2) \
  DataAccessDemoteCase<T1, T2, 0>, DataAccessDemoteCase<T1, T2, -1>

#define DATA_ACCESS_PROMOTE_SHIFT2(T1, T2) \
  DataAccessPromoteCase<T1, T2, 0>, DataAccessPromoteCase<T1, T2, -1>, \
  DataAccessPromoteCase<T1, T2, -2>
#define DATA_ACCESS_DEMOTE_SHIFT2(T1, T2) \
  DataAccessDemoteCase<T1, T2, 0>, DataAccessDemoteCase<T1, T2, -1>, \
  DataAccessDemoteCase<T1, T2, -2>

#if VEC_MAX_POW >= 3
#define DATA_ACCESS_PROMOTE_SHIFT3(T1, T2) \
  DataAccessPromoteCase<T1, T2, 0>, DataAccessPromoteCase<T1, T2, -1>, \
  DataAccessPromoteCase<T1, T2, -2>, DataAccessPromoteCase<T1, T2, -3>
#define DATA_ACCESS_DEMOTE_SHIFT3(T1, T2) \
  DataAccessDemoteCase<T1, T2, 0>, DataAccessDemoteCase<T1, T2, -1>, \
  DataAccessDemoteCase<T1, T2, -2>, DataAccessDemoteCase<T1, T2, -3>
#else
#define DATA_ACCESS_PROMOTE_SHIFT3(T1, T2) \
  DataAccessPromoteCase<T1, T2, -1>, DataAccessPromoteCase<T1, T2, -2>, \
  DataAccessPromoteCase<T1, T2, -3>
#define DATA_ACCESS_DEMOTE_SHIFT3(T1, T2) \
  DataAccessDemoteCase<T1, T2, -1>, DataAccessDemoteCase<T1, T2, -2>, \
  DataAccessDemoteCase<T1, T2, -3>
#endif

#define DATA_ACCESS_CONVERT_POWS(T1, T2) \
  DataAccessConversionCase<T1, T2, 0, 0>, DataAccessConversionCase<T1, T2, 1, 1>, \
  DataAccessConversionCase<T1, T2, 2, 2>

template <typename T>
class DataAccessTypedTest : public ::testing::Test {};

using DataAccessTypes = test_utils::AllVecDataTypes;
TYPED_TEST_SUITE(DataAccessTypedTest, DataAccessTypes);

TYPED_TEST(DataAccessTypedTest, LastContiguousInputDefaultIdentityLoadsTail) {
  using T = TypeParam;
  using L = Layout<Shape<Const<3>, Const<64>>, Strides<Const<64>, Const<1>>>;
  using Spec = InputSpec<T, T, L>;

  auto src = make_source<T>(256);
  L layout{Shape<Const<3>, Const<64>>{}, Strides<Const<64>, Const<1>>{}};
  Spec spec(layout);
  auto input = spec.make_input(src.data(), nullptr);

  ScalableTag<T, 0> t;
  const nint_t count = std::min<nint_t>(std::max<nint_t>(1, size(t) - 1), 62);
  auto v = input(t, count, nint_t{1}, nint_t{2});

  for (nint_t i = 0; i < count; ++i) {
    EXPECT_TRUE(test_utils::values_equal(src[66 + i], read_vec_lane<T>(v, i)));
  }
  if (count < size(t)) {
    EXPECT_TRUE(test_utils::values_equal(T{}, read_vec_lane<T>(v, count)));
  }
}

TYPED_TEST(DataAccessTypedTest, StridedInputUsesAuxAndPadding) {
  using T = TypeParam;
  using L = Layout<Shape<Const<3>, Const<63>>, Strides<Const<257>, Const<3>>>;
  using Spec = InputSpec<T, T, L>;

  auto src = make_source<T>(800);
  L layout{Shape<Const<3>, Const<63>>{}, Strides<Const<257>, Const<3>>{}};
  auto aux = make_aux<T>(Spec::required_workspace(layout));
  Spec spec(layout);
  auto input = spec.make_input(src.data(), aux.data());

  ScalableTag<T, 0> t;
  const nint_t count = std::min<nint_t>(std::max<nint_t>(1, size(t) - 1), 62);
  auto v = input(t, count, nint_t{2}, nint_t{1});

  for (nint_t i = 0; i < count; ++i) {
    EXPECT_TRUE(test_utils::values_equal(src[517 + i * 3], read_vec_lane<T>(v, i)));
  }
  EXPECT_GT(Spec::required_workspace(layout), 3 * 63 * static_cast<nint_t>(sizeof(T)));
}

TYPED_TEST(DataAccessTypedTest, StridedOutputScattersTailOnly) {
  using T = TypeParam;
  using L = Layout<Shape<Const<3>, Const<63>>, Strides<Const<257>, Const<3>>>;
  using Spec = OutputSpec<T, T, L>;

  std::vector<T> dst(800, static_cast<T>(-7));
  L layout{Shape<Const<3>, Const<63>>{}, Strides<Const<257>, Const<3>>{}};
  Spec spec(layout);
  auto output = spec.make_output(dst.data(), nullptr);

  ScalableTag<T, 0> t;
  std::array<T, 128> values{};
  for (nint_t i = 0; i < size(t); ++i) {
    values[static_cast<size_t>(i)] = static_cast<T>(i + 11);
  }
  auto v = load(t, values.data());
  const nint_t count = std::min<nint_t>(std::max<nint_t>(1, size(t) - 1), 62);
  output(t, v, count, nint_t{1}, nint_t{1});

  for (nint_t i = 0; i < count; ++i) {
    EXPECT_TRUE(test_utils::values_equal(static_cast<T>(i + 11), dst[260 + i * 3]));
  }
  EXPECT_TRUE(test_utils::values_equal(static_cast<T>(-7), dst[260 + count * 3]));
}

TEST(DataAccessTransformTest, CoordinatesAreForwardedForLastContiguousInput) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;

  auto src = make_source<T>(32);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<5>, Const<1>>{}};
  int calls = 0;
  std::vector<nint_t> coords;
  CoordShiftFn<T> fn(&calls, &coords);
  DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> input(
      src.data(), layout, fn, nullptr);

  ScalableTag<T, 0> t;
  auto v = input(t, nint_t{3}, nint_t{1}, nint_t{2});

  EXPECT_GE(calls, 1);
  EXPECT_EQ(coords, (std::vector<nint_t>{1, 2}));
  EXPECT_EQ(read_vec_lane<T>(v, 0), coord_shift_expected(src[7], 1, 2));
  EXPECT_EQ(read_vec_lane<T>(v, 1), coord_shift_expected(src[8], 1, 2));
  if (3 < size(t)) {
    EXPECT_EQ(read_vec_lane<T>(v, 3), 0);
  }
}

TEST(DataAccessTransformTest, SecondLastContiguousOutputFlushesInDestructor) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<1>, Const<3>>>;
  using Spec = OutputSpec<T, T, L, CoordShiftFn<T>>;

  std::vector<T> dst(32, -9);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<1>, Const<3>>{}};
  auto aux = make_aux<T>(Spec::required_workspace(layout));
  int calls = 0;
  std::vector<nint_t> coords;
  CoordShiftFn<T> fn(&calls, &coords);

  {
    Spec spec(layout, fn);
    auto output = spec.make_output(dst.data(), aux.data());
    ScalableTag<T, 0> t;
    std::array<T, 64> values{};
    for (nint_t i = 0; i < size(t); ++i) {
      values[static_cast<size_t>(i)] = static_cast<T>(i + 1);
    }
    auto v = load(t, values.data());
    output(t, v, nint_t{4}, nint_t{1}, nint_t{1});
    EXPECT_EQ(dst[4], -9);
  }

  EXPECT_GE(calls, 1);
  EXPECT_EQ(coords, (std::vector<nint_t>{1, 1}));
  EXPECT_EQ(dst[4], coord_shift_expected(1, 1, 1));
  EXPECT_EQ(dst[7], coord_shift_expected(2, 1, 1));
  EXPECT_EQ(dst[10], coord_shift_expected(3, 1, 1));
  EXPECT_EQ(dst[13], coord_shift_expected(4, 1, 1));
}

TEST(DataAccessSpecWrapperTest, InputHelperBindsTensorWithWorkspace) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<17>, Const<3>>>;

  auto src = make_source<T>(80);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<17>, Const<3>>{}};
  auto tensor = make_tensor(src.data(), layout);
  auto spec = input<T>(tensor);

  EXPECT_EQ(spec.required_workspace(), (InputSpec<T, T, L>::required_workspace(layout)));

  Workspace workspace(spec.required_workspace());
  auto view = workspace.view();
  auto accessor = spec.bind(view);

  ScalableTag<T, 0> t;
  auto v = accessor(t, nint_t{3}, nint_t{2}, nint_t{1});
  EXPECT_EQ(read_vec_lane<T>(v, 0), src[37]);
  EXPECT_EQ(read_vec_lane<T>(v, 1), src[40]);
  EXPECT_EQ(read_vec_lane<T>(v, 2), src[43]);
}

TEST(DataAccessSpecWrapperTest, ShapeOnlyInitializerTensorUsesLastContiguousInputPath) {
  using T = int32_t;

  auto src = make_source<T>(32);
  auto tensor = make_tensor<2>(src.data(), {3, 5});
  auto spec = input<T>(tensor);

  using Accessor = typename decltype(spec)::InputAccessor;
  EXPECT_TRUE((std::is_same_v<
      typename Accessor::AccessKind,
      vecops::gemm::details::AccessKindLastContiguous>));
  EXPECT_EQ(spec.required_workspace(), 0);
}

TEST(DataAccessSpecWrapperTest, OutputHelperBindsTensorWithWorkspace) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;

  std::vector<T> dst(32, -1);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<5>, Const<1>>{}};
  auto tensor = make_tensor(dst.data(), layout);
  auto spec = output<T>(tensor);

  EXPECT_EQ(spec.required_workspace(), 0);
  EXPECT_EQ(required_workspace(spec), 0);

  Workspace workspace(required_workspace(spec));
  auto view = workspace.view();
  auto accessor = spec.bind(view);

  ScalableTag<T, 0> t;
  std::array<T, 64> values{};
  for (nint_t i = 0; i < size(t); ++i) {
    values[static_cast<size_t>(i)] = static_cast<T>(100 + i);
  }
  auto v = load(t, values.data());
  accessor(t, v, nint_t{3}, nint_t{1}, nint_t{2});

  EXPECT_EQ(dst[7], 100);
  EXPECT_EQ(dst[8], 101);
  EXPECT_EQ(dst[9], 102);
}

TEST(DataAccessSpecWrapperTest, ShapeOnlyInitializerTensorUsesLastContiguousOutputPath) {
  using T = int32_t;

  std::vector<T> dst(32, -1);
  auto tensor = make_tensor<2>(dst.data(), {3, 5});
  auto spec = output<T>(tensor);

  using Accessor = typename decltype(spec)::OutputAccessor;
  EXPECT_TRUE((std::is_same_v<
      typename Accessor::AccessKind,
      vecops::gemm::details::AccessKindLastContiguous>));
  EXPECT_EQ(spec.required_workspace(), 0);
}

TEST(DataAccessHOPSliceTest, SlicedInputAccessorUsesRowLayoutAndFullTransformCoords) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;

  auto src = make_source<T>(32);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<5>, Const<1>>{}};
  int calls = 0;
  std::vector<nint_t> coords;
  CoordShiftFn<T> fn(&calls, &coords);
  DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> accessor(
      src.data(), layout, fn, nullptr);

  ScalableTag<T, 0> tag;
  std::vector<T> seen;
  hop::for_each<0>(
      [&](const auto& row) {
        auto v = row(tag, nint_t{2}, nint_t{1});
        seen.push_back(read_vec_lane<T>(v, 0));
        seen.push_back(read_vec_lane<T>(v, 1));
      },
      accessor);

  EXPECT_GE(calls, 3);
  EXPECT_EQ(coords, (std::vector<nint_t>{2, 1}));
  EXPECT_EQ(seen[0], coord_shift_expected(src[1], 0, 1));
  EXPECT_EQ(seen[1], coord_shift_expected(src[2], 0, 1));
  EXPECT_EQ(seen[4], coord_shift_expected(src[11], 2, 1));
  EXPECT_EQ(seen[5], coord_shift_expected(src[12], 2, 1));
}

TEST(DataAccessHOPSliceTest, SlicedInputSpecBindsWithFullTransformCoords) {
  using T = int32_t;
  using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;

  auto src = make_source<T>(32);
  L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<5>, Const<1>>{}};
  auto tensor = make_tensor(src.data(), layout);
  int calls = 0;
  std::vector<nint_t> coords;
  auto spec = input<T>(tensor, CoordShiftFn<T>(&calls, &coords));

  Workspace workspace(0);
  auto view = workspace.view();
  ScalableTag<T, 0> tag;
  std::vector<T> seen;
  hop::for_each<0>(
      [&](const auto& row_spec) {
        auto row = row_spec.bind(view);
        auto v = row(tag, nint_t{2}, nint_t{2});
        seen.push_back(read_vec_lane<T>(v, 0));
      },
      spec);

  EXPECT_GE(calls, 3);
  EXPECT_EQ(coords, (std::vector<nint_t>{2, 2}));
  EXPECT_EQ(seen[0], coord_shift_expected(src[2], 0, 2));
  EXPECT_EQ(seen[1], coord_shift_expected(src[7], 1, 2));
  EXPECT_EQ(seen[2], coord_shift_expected(src[12], 2, 2));
}

TEST(DataAccessHOPSliceTest, SlicedSecondLastOutputAccessorFlushesOnlyFromParent) {
  using T = int32_t;
  using L = Layout<Shape<Const<2>, Const<5>>, Strides<Const<1>, Const<2>>>;
  using Spec = OutputSpec<T, T, L, CoordShiftFn<T>>;

  std::vector<T> dst(16, -9);
  L layout{Shape<Const<2>, Const<5>>{}, Strides<Const<1>, Const<2>>{}};
  auto aux = make_aux<T>(Spec::required_workspace(layout));
  int calls = 0;
  std::vector<nint_t> coords;

  {
    Spec spec(layout, CoordShiftFn<T>(&calls, &coords));
    auto output = spec.make_output(dst.data(), aux.data());
    ScalableTag<T, 0> tag;
    std::array<T, 64> values{};
    for (nint_t i = 0; i < size(tag); ++i) {
      values[static_cast<size_t>(i)] = static_cast<T>(10 + i);
    }
    auto v = load(tag, values.data());

    hop::for_each<0>(
        [&](const auto& row) {
          row(tag, v, nint_t{2}, nint_t{1});
          EXPECT_EQ(dst[2], -9);
          EXPECT_EQ(dst[3], -9);
        },
        output);
  }

  EXPECT_GE(calls, 2);
  EXPECT_EQ(coords, (std::vector<nint_t>{1, 1}));
  EXPECT_EQ(dst[2], coord_shift_expected(static_cast<T>(10), 0, 1));
  EXPECT_EQ(dst[4], coord_shift_expected(static_cast<T>(11), 0, 1));
  EXPECT_EQ(dst[3], coord_shift_expected(static_cast<T>(10), 1, 1));
  EXPECT_EQ(dst[5], coord_shift_expected(static_cast<T>(11), 1, 1));
}

TEST(WorkspaceTest, ParallelWorkspaceProvidesIndependentThreadViews) {
  ParallelWorkspace workspace(2, 64);
  auto left = workspace.thread_view(0);
  auto right = workspace.thread_view(1);

  auto* a = left.allocate<int32_t>(4);
  auto* b = right.allocate<int32_t>(4);
  a[0] = 11;
  b[0] = 22;

  EXPECT_EQ(a[0], 11);
  EXPECT_EQ(b[0], 22);
  EXPECT_NE(a, b);
}

TEST(DataAccessTransformTest, FullCoordinatesAreForwardedForInputRanksOneThroughFour) {
  using T = int32_t;

  {
    using L = Layout<Shape<Const<7>>, Strides<Const<1>>>;
    auto src = make_source<T>(16);
    L layout{Shape<Const<7>>{}, Strides<Const<1>>{}};
    int calls = 0;
    std::vector<nint_t> coords;
    CoordShiftFn<T> fn(&calls, &coords);
    DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> input(
        src.data(), layout, fn, nullptr);

    ScalableTag<T, 0> t;
    auto v = input(t, nint_t{1}, nint_t{4});
    EXPECT_GE(calls, 1);
    EXPECT_EQ(coords, (std::vector<nint_t>{4}));
    EXPECT_EQ(read_vec_lane<T>(v, 0), coord_shift_expected(src[4], 4));
  }

  {
    using L = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;
    auto src = make_source<T>(32);
    L layout{Shape<Const<3>, Const<5>>{}, Strides<Const<5>, Const<1>>{}};
    int calls = 0;
    std::vector<nint_t> coords;
    CoordShiftFn<T> fn(&calls, &coords);
    DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> input(
        src.data(), layout, fn, nullptr);

    ScalableTag<T, 0> t;
    auto v = input(t, nint_t{1}, nint_t{2}, nint_t{3});
    EXPECT_GE(calls, 1);
    EXPECT_EQ(coords, (std::vector<nint_t>{2, 3}));
    EXPECT_EQ(read_vec_lane<T>(v, 0), coord_shift_expected(src[13], 2, 3));
  }

  {
    using L = Layout<Shape<Const<2>, Const<3>, Const<5>>, Strides<Const<15>, Const<5>, Const<1>>>;
    auto src = make_source<T>(64);
    L layout{Shape<Const<2>, Const<3>, Const<5>>{}, Strides<Const<15>, Const<5>, Const<1>>{}};
    int calls = 0;
    std::vector<nint_t> coords;
    CoordShiftFn<T> fn(&calls, &coords);
    DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> input(
        src.data(), layout, fn, nullptr);

    ScalableTag<T, 0> t;
    auto v = input(t, nint_t{1}, nint_t{1}, nint_t{2}, nint_t{4});
    EXPECT_GE(calls, 1);
    EXPECT_EQ(coords, (std::vector<nint_t>{1, 2, 4}));
    EXPECT_EQ(read_vec_lane<T>(v, 0), coord_shift_expected(src[29], 1, 2, 4));
  }

  {
    using L = Layout<
        Shape<Const<2>, Const<3>, Const<4>, Const<5>>,
        Strides<Const<60>, Const<20>, Const<5>, Const<1>>>;
    auto src = make_source<T>(128);
    L layout{
        Shape<Const<2>, Const<3>, Const<4>, Const<5>>{},
        Strides<Const<60>, Const<20>, Const<5>, Const<1>>{}};
    int calls = 0;
    std::vector<nint_t> coords;
    CoordShiftFn<T> fn(&calls, &coords);
    DataInput<T, Tensor<T, typename L::Shape, typename L::Strides>, CoordShiftFn<T>> input(
        src.data(), layout, fn, nullptr);

    ScalableTag<T, 0> t;
    auto v = input(t, nint_t{1}, nint_t{1}, nint_t{2}, nint_t{3}, nint_t{4});
    EXPECT_GE(calls, 1);
    EXPECT_EQ(coords, (std::vector<nint_t>{1, 2, 3, 4}));
    EXPECT_EQ(read_vec_lane<T>(v, 0), coord_shift_expected(src[119], 1, 2, 3, 4));
  }
}

template <typename Tag>
void run_last_contiguous_input_for_tag() {
  using T = ElementOf<Tag>;
  using L = Layout<Shape<Const<2>, Const<128>>, Strides<Const<128>, Const<1>>>;
  using Spec = InputSpec<T, T, L>;

  auto src = make_source<T>(256);
  L layout{Shape<Const<2>, Const<128>>{}, Strides<Const<128>, Const<1>>{}};
  Spec spec(layout);
  auto input = spec.make_input(src.data(), nullptr);

  Tag t;
  const nint_t count = std::min<nint_t>(std::max<nint_t>(1, size(t) - 1), 125);
  auto v = input(t, count, nint_t{1}, nint_t{3});
  for (nint_t i = 0; i < count; ++i) {
    EXPECT_TRUE(test_utils::values_equal(src[131 + i], vec::get(t, v, i)));
  }
  if (count < size(t)) {
    EXPECT_TRUE(test_utils::values_equal(T{}, vec::get(t, v, count)));
  }
}

TEST(DataAccessVectorLengthTest, LastContiguousInputSupportsSmallAndLargeTags) {
  run_last_contiguous_input_for_tag<ScalableTag<int32_t, 0>>();
  run_last_contiguous_input_for_tag<ScalableTag<int32_t, 1>>();
  run_last_contiguous_input_for_tag<ScalableTag<int8_t, 0>>();
  run_last_contiguous_input_for_tag<ScalableTag<int8_t, 1>>();
}

template <typename TCase>
class DataAccessConversionTest : public ::testing::Test {};

using DataAccessConversionCases = ::testing::Types<
    DATA_ACCESS_PROMOTE_SHIFT1(int8_t, int16_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int8_t, uint16_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int8_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_PROMOTE_SHIFT1(int8_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_PROMOTE_SHIFT1(uint8_t, uint16_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint8_t, int16_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint8_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_PROMOTE_SHIFT1(uint8_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_PROMOTE_SHIFT2(int8_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT2(int8_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT2(int8_t, float32_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint8_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint8_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint8_t, float32_t),
    DATA_ACCESS_PROMOTE_SHIFT3(int8_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT3(int8_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT3(uint8_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT3(uint8_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT3(uint8_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int16_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int16_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int16_t, float32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint16_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint16_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint16_t, float32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::float16_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::float16_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::float16_t, float32_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::bfloat16_t, int32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::bfloat16_t, uint32_t),
    DATA_ACCESS_PROMOTE_SHIFT1(vecops::bfloat16_t, float32_t),
#endif
    DATA_ACCESS_PROMOTE_SHIFT2(int16_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(int16_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(int16_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint16_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint16_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(uint16_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::float16_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::float16_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::float16_t, float64_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::bfloat16_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::bfloat16_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT2(vecops::bfloat16_t, float64_t),
#endif
    DATA_ACCESS_PROMOTE_SHIFT1(int32_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int32_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(int32_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint32_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint32_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(uint32_t, float64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(float32_t, uint64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(float32_t, int64_t),
    DATA_ACCESS_PROMOTE_SHIFT1(float32_t, float64_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int16_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int16_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint16_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint16_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(vecops::float16_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(vecops::float16_t, uint8_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT1(vecops::bfloat16_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(vecops::bfloat16_t, uint8_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT1(int32_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int32_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT1(int32_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT1(uint32_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint32_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT1(uint32_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT1(float32_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(float32_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT1(float32_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT1(float32_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT2(int32_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT2(int32_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT2(uint32_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT2(uint32_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT2(float32_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT2(float32_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int64_t, int32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int64_t, uint32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(int64_t, float32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint64_t, int32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint64_t, uint32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(uint64_t, float32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(float64_t, int32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(float64_t, uint32_t),
    DATA_ACCESS_DEMOTE_SHIFT1(float64_t, float32_t),
    DATA_ACCESS_DEMOTE_SHIFT2(int64_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(int64_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(int64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT2(int64_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT2(uint64_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(uint64_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(uint64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT2(uint64_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT2(float64_t, int16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(float64_t, uint16_t),
    DATA_ACCESS_DEMOTE_SHIFT2(float64_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_DEMOTE_SHIFT2(float64_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_DEMOTE_SHIFT3(int64_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT3(int64_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT3(uint64_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT3(uint64_t, uint8_t),
    DATA_ACCESS_DEMOTE_SHIFT3(float64_t, int8_t),
    DATA_ACCESS_DEMOTE_SHIFT3(float64_t, uint8_t),
    DATA_ACCESS_CONVERT_POWS(int8_t, uint8_t),
    DATA_ACCESS_CONVERT_POWS(uint8_t, int8_t),
    DATA_ACCESS_CONVERT_POWS(int16_t, uint16_t),
    DATA_ACCESS_CONVERT_POWS(int16_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_CONVERT_POWS(int16_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_CONVERT_POWS(uint16_t, int16_t),
    DATA_ACCESS_CONVERT_POWS(uint16_t, vecops::float16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_CONVERT_POWS(uint16_t, vecops::bfloat16_t),
#endif
    DATA_ACCESS_CONVERT_POWS(vecops::float16_t, int16_t),
    DATA_ACCESS_CONVERT_POWS(vecops::float16_t, uint16_t),
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
    DATA_ACCESS_CONVERT_POWS(vecops::float16_t, vecops::bfloat16_t),
    DATA_ACCESS_CONVERT_POWS(vecops::bfloat16_t, int16_t),
    DATA_ACCESS_CONVERT_POWS(vecops::bfloat16_t, uint16_t),
    DATA_ACCESS_CONVERT_POWS(vecops::bfloat16_t, vecops::float16_t),
#endif
    DATA_ACCESS_CONVERT_POWS(int32_t, uint32_t),
    DATA_ACCESS_CONVERT_POWS(int32_t, float32_t),
    DATA_ACCESS_CONVERT_POWS(uint32_t, int32_t),
    DATA_ACCESS_CONVERT_POWS(uint32_t, float32_t),
    DATA_ACCESS_CONVERT_POWS(float32_t, int32_t),
    DATA_ACCESS_CONVERT_POWS(float32_t, uint32_t),
    DATA_ACCESS_CONVERT_POWS(int64_t, uint64_t),
    DATA_ACCESS_CONVERT_POWS(int64_t, float64_t),
    DATA_ACCESS_CONVERT_POWS(uint64_t, int64_t),
    DATA_ACCESS_CONVERT_POWS(uint64_t, float64_t),
    DATA_ACCESS_CONVERT_POWS(float64_t, int64_t),
    DATA_ACCESS_CONVERT_POWS(float64_t, uint64_t)
>;

TYPED_TEST_SUITE(DataAccessConversionTest, DataAccessConversionCases, DataAccessConversionCaseNames);

TYPED_TEST(DataAccessConversionTest, InputSpecCoversTypeAndVectorLengthCombinations) {
  using TIn = typename TypeParam::TIn;
  using TOut = typename TypeParam::TOut;
  using OutTag = typename TypeParam::OutTag;
  using L = Layout<Shape<Const<2>, Const<128>>, Strides<Const<257>, Const<3>>>;
  using Spec = InputSpec<TOut, TIn, L>;

  auto src = make_source<TIn>(600);
  L layout{Shape<Const<2>, Const<128>>{}, Strides<Const<257>, Const<3>>{}};
  auto aux = make_aux<TOut>(Spec::required_workspace(layout));
  Spec spec(layout);
  auto input = spec.make_input(src.data(), aux.data());

  OutTag t;
  // A scalable Tag has no architecture-independent compile-time lane count.
  // One lane keeps this a genuinely static-count API test on every VL.
  constexpr nint_t StaticN = 1;
  auto v = input(t, Const<StaticN>{}, nint_t{1}, nint_t{2});
  const nint_t lanes = std::min<nint_t>(size(t), StaticN);
  for (nint_t i = 0; i < lanes; ++i) {
    TOut expected = vecops::convert<TOut>(src[263 + i * 3]);
    EXPECT_TRUE(test_utils::values_near(expected, vec::get(t, v, i))) << "i=" << i;
  }
}

TYPED_TEST(DataAccessConversionTest, OutputSpecCoversTypeAndVectorLengthCombinations) {
  using TIn = typename TypeParam::TIn;
  using TOut = typename TypeParam::TOut;
  using InTag = typename TypeParam::InTag;
  using L = Layout<Shape<Const<2>, Const<128>>, Strides<Const<257>, Const<3>>>;
  using Spec = OutputSpec<TIn, TOut, L>;

  std::vector<TOut> dst(600, TOut{});
  L layout{Shape<Const<2>, Const<128>>{}, Strides<Const<257>, Const<3>>{}};
  Spec spec(layout);
  auto output = spec.make_output(dst.data(), nullptr);

  InTag t;
  auto v = fill(t, static_cast<TIn>(7));
  constexpr nint_t StaticN = 1;
  output(t, v, Const<StaticN>{}, nint_t{1}, nint_t{2});
  const nint_t lanes = std::min<nint_t>(size(t), StaticN);
  for (nint_t i = 0; i < lanes; ++i) {
    TOut expected = vecops::convert<TOut>(static_cast<TIn>(7));
    EXPECT_TRUE(test_utils::values_near(expected, dst[263 + i * 3])) << "i=" << i;
  }
}

TEST(DataAccessTraitTest, SpecHelpersClassifyTransformsAndContinuity) {
  using Last = Layout<Shape<Const<3>, Const<5>>, Strides<Const<5>, Const<1>>>;
  using Second = Layout<Shape<Const<3>, Const<5>>, Strides<Const<1>, Const<3>>>;
  using Strided = Layout<Shape<Const<3>, Const<5>>, Strides<Const<17>, Const<3>>>;

  using LastIn = InputSpec<int32_t, int32_t, Last>;
  using SecondIn = InputSpec<int32_t, int32_t, Second>;
  using StridedIn = InputSpec<int32_t, int32_t, Strided>;
  using ZeroIn = InputSpec<int32_t, int32_t, Last, ZeroVecTransform<int32_t>>;
  using LastOut = OutputSpec<int32_t, int32_t, Last>;

  EXPECT_TRUE(is_input_spec_v<LastIn>);
  EXPECT_TRUE(is_input_spec_v<const LastIn&>);
  EXPECT_FALSE(is_input_spec_v<LastOut>);
  EXPECT_TRUE(is_output_spec_v<LastOut>);
  EXPECT_TRUE(is_output_spec_v<const LastOut&>);
  EXPECT_FALSE(is_output_spec_v<LastIn>);
  EXPECT_TRUE(is_identity_last_contiguous_input_spec_v<LastIn>);
  EXPECT_FALSE(is_identity_last_contiguous_input_spec_v<SecondIn>);
  EXPECT_TRUE(is_identity_second_last_contiguous_input_spec_v<SecondIn>);
  EXPECT_FALSE(is_identity_second_last_contiguous_input_spec_v<StridedIn>);
  EXPECT_TRUE(is_zero_input_spec_v<ZeroIn>);
}

} // namespace
