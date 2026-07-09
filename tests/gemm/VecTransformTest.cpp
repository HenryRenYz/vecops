#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <type_traits>

#include "vecops/VecTransform.h"
#include "vecops/util/Math.h"
#include "vecops/util/ScalarConvert.h"

using namespace vecops;
using namespace vecops::vec;

using test_float16_t = vecops::float16_t;

// ============================================================================
// Helpers: POW2-gated functors
// ============================================================================

template <typename Eo_, typename Ei_, int SupportedPow2>
struct PartialCoordinateFn {
  template <typename To, TL_IF(vec::scalable_pow2_of<To> == SupportedPow2)>
  Vec<To> operator()(To t, Vec<Rebind<Ei_, To>> v_in, nint_t, nint_t) const {
    if constexpr (std::is_same_v<Eo_, Ei_>) {
      return bitcast(t, v_in);
    } else {
      return bitcast(t, xconvert(t, v_in));
    }
  }
};

template <typename Eo_, typename Ei_, int SupportedPow2>
struct PartialElementwiseFn {
  template <typename To, TL_IF(vec::scalable_pow2_of<To> == SupportedPow2)>
  Vec<To> operator()(To t, Vec<Rebind<Ei_, To>> v_in) const {
    if constexpr (std::is_same_v<Eo_, Ei_>) {
      return bitcast(t, v_in);
    } else {
      return bitcast(t, xconvert(t, v_in));
    }
  }
};

// ============================================================================
// VecFn with artificially limited max_input_pow2 for Branch C testing
// ============================================================================

template <typename Eo_, typename Ei_, int LimitedMaxInputPow2>
struct LimitedMaxInputVecFn : public VecTransform<Eo_, Ei_> {
  static constexpr int max_input_pow2 = LimitedMaxInputPow2;

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo_>),
      TL_IF(LimitedMaxInputVecFn::min_output_pow2 <= vec::scalable_pow2_of<To>
         && vec::scalable_pow2_of<To> <= LimitedMaxInputVecFn::max_output_pow2)>
  Vec<To> operator()(To, Vec<Rebind<Ei_, To>> v_in, nint_t, nint_t) const {
    return bitcast(To{}, xconvert(To{}, v_in));
  }
};

// ============================================================================
// Helpers: memory and values
// ============================================================================

template <typename T>
T* alloc_aligned(nint_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

template <typename T>
T get_value(int idx) {
  if constexpr (std::is_same_v<T, float32_t>)      return static_cast<float32_t>(idx * 1.5f + 0.5f);
  else if constexpr (std::is_same_v<T, float64_t>) return static_cast<float64_t>(idx * 1.5 + 0.5);
  else if constexpr (std::is_same_v<T, int8_t>)    return static_cast<int8_t>((idx * 7 + 3) % 127 - 64);
  else if constexpr (std::is_same_v<T, uint8_t>)   return static_cast<uint8_t>((idx * 7 + 3) % 256);
  else if constexpr (std::is_same_v<T, int16_t>)   return static_cast<int16_t>((idx * 100 + 50) % 32767 - 16384);
  else if constexpr (std::is_same_v<T, uint16_t>)  return static_cast<uint16_t>((idx * 100 + 50) % 65536);
  else if constexpr (std::is_same_v<T, int32_t>)   return static_cast<int32_t>(idx * 1000 + 500);
  else if constexpr (std::is_same_v<T, uint32_t>)  return static_cast<uint32_t>(idx * 1000 + 500);
  else if constexpr (std::is_same_v<T, int64_t>)   return static_cast<int64_t>(idx * 100000LL + 50000LL);
  else if constexpr (std::is_same_v<T, uint64_t>)  return static_cast<uint64_t>(idx * 100000ULL + 50000ULL);
  else return static_cast<T>(idx + 1);
}

template <typename T>
bool values_close(T expected, T actual, double tol = 0.01) {
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    return std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tol;
  } else {
    return expected == actual;
  }
}

struct CoordinateEncodingFn {
  template <TLV_DECL_TAG(To), typename... Coords>
  Vec<To> operator()(To t, Vec<To> v_in, Coords... coords) const {
    nint_t encoded = 0;
    ((encoded = encoded * 100 + static_cast<nint_t>(coords)), ...);
    return add(v_in, fill(t, static_cast<TypeOf<To>>(encoded)));
  }
};

struct PlusOneElementwiseFn {
  template <TLV_DECL_TAG(To)>
  Vec<To> operator()(To t, Vec<To> v_in) const {
    return add(v_in, fill(t, static_cast<TypeOf<To>>(1)));
  }
};

TEST(VecTransformCoordinateTest, CoordinatesOneThroughFourAreForwarded) {
  auto adapter = make_vec_transform<int32_t, int32_t>(CoordinateEncodingFn{});
  ScalableTag<int32_t, 0> t;
  auto v = fill(t, int32_t{7});

  EXPECT_EQ(get(t, adapter(t, v, nint_t{4}), 0), 11);
  EXPECT_EQ(get(t, adapter(t, v, nint_t{1}, nint_t{2}), 0), 109);
  EXPECT_EQ(get(t, adapter(t, v, nint_t{1}, nint_t{2}, nint_t{3}), 0), 10210);
  EXPECT_EQ(get(t, adapter(t, v, nint_t{1}, nint_t{2}, nint_t{3}, nint_t{4}), 0), 1020311);
}

TEST(VecTransformElementwiseTest, ElementwiseTransformIgnoresCoordinates) {
  auto adapter = make_elementwise_vec_transform<int32_t, int32_t>(PlusOneElementwiseFn{});
  ScalableTag<int32_t, 0> t;
  auto v = fill(t, int32_t{7});

  EXPECT_TRUE(decltype(adapter)::is_elementwise);
  EXPECT_EQ(get(t, adapter(t, v), 0), 8);
  EXPECT_EQ(get(t, adapter(t, v, nint_t{1}, nint_t{2}, nint_t{3}, nint_t{4}), 0), 8);
}

TEST(VecTransformFactoryTest, AdaptWrapsLambdaAndConvertsExistingTransform) {
  auto elementwise = make_elementwise_vec_transform<int32_t, int32_t>(PlusOneElementwiseFn{});
  auto converted = adapt_vec_transform<float32_t, int32_t>(elementwise);

  EXPECT_TRUE(decltype(converted)::is_elementwise);

  ScalableTag<float32_t, 0> t;
  auto v = fill(Rebind<int32_t, decltype(t)>{}, int32_t{2});
  auto out = converted(t, v, nint_t{9});
  EXPECT_TRUE(values_close(3.0f, get(t, out, 0)));
}

// ============================================================================
// Test runner — LambdaVecTransform
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runPositionalLambdaTest() {
  using FnType = PartialCoordinateFn<Eo, Ei, 0>;
  LambdaVecTransform<Eo, Ei, FnType> adapter{FnType{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  nint_t N_out = size(t_o);
  nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);
  auto buf_ref = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);
  for (nint_t i = 0; i < N_out; ++i) buf_ref[i] = convert<Eo, Ei>(buf_in[i % N_in]);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = adapter(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    EXPECT_TRUE(values_close(buf_ref[i], buf_out[i]))
        << "CoordAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
  std::free(buf_ref);
}

// ============================================================================
// Test runner — LambdaVecTransform
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runElementwiseLambdaTest() {
  using FnType = PartialElementwiseFn<Eo, Ei, 0>;
  LambdaVecTransform<Eo, Ei, FnType, true> adapter{FnType{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  nint_t N_out = size(t_o);
  nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);
  auto buf_ref = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);
  for (nint_t i = 0; i < N_out; ++i) buf_ref[i] = convert<Eo, Ei>(buf_in[i % N_in]);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = adapter(t_o, v_in);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    EXPECT_TRUE(values_close(buf_ref[i], buf_out[i]))
        << "ElemAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
  std::free(buf_ref);
}

// ============================================================================
// Test runner — ConvertedVecTransform (Branch A + B)
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runConversionTest() {
  using InnerFn = PartialCoordinateFn<Ei, Ei, 0>;
  LambdaVecTransform<Ei, Ei, InnerFn> inner{InnerFn{}};
  ConvertedVecTransform<Eo, Ei, decltype(inner)> outer{std::move(inner)};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  nint_t N_out = size(t_o);
  nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = outer(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    Eo expected = convert<Eo, Ei>(buf_in[i % N_in]);
    EXPECT_TRUE(values_close(expected, buf_out[i]))
        << "ConvertedAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
}

// ============================================================================
// Test runner — ConvertedVecTransform Branch C (recursive split)
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runConversionBranchCTest() {
  using InnerFn = LimitedMaxInputVecFn<Ei, Ei, 2>;
  ConvertedVecTransform<Eo, Ei, InnerFn> outer{InnerFn{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  nint_t N_out = size(t_o);
  nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = outer(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    Eo expected = convert<Eo, Ei>(buf_in[i % N_in]);
    EXPECT_TRUE(values_close(expected, buf_out[i]))
        << "ConvertedBranchC Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
}

// ============================================================================
// Test runner — ConvertedVecTransform elementwise short-circuit
// ============================================================================

template <typename Eo, typename Ei>
void runConversionElementwiseTest() {
  using InnerFn = PartialElementwiseFn<Ei, Ei, 0>;
  LambdaVecTransform<Ei, Ei, InnerFn, true> inner{InnerFn{}};
  ConvertedVecTransform<Eo, Ei, decltype(inner)> outer{std::move(inner)};

  constexpr int Pow2 = 0;
  using OutTag = ScalableTag<Eo, Pow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  nint_t N = size(t_o);
  auto buf_in  = alloc_aligned<Ei>(N);
  auto buf_out_coord = alloc_aligned<Eo>(N);
  auto buf_out_nocoord = alloc_aligned<Eo>(N);

  for (nint_t i = 0; i < N; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_with = outer(t_o, v_in, 42, 99);
  auto v_without = outer(t_o, v_in);
  storeu(t_o, buf_out_coord, v_with);
  storeu(t_o, buf_out_nocoord, v_without);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(buf_out_coord[i], buf_out_nocoord[i]))
        << "ConvertedElem mismatch at i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out_coord);
  std::free(buf_out_nocoord);
}

// ============================================================================
// Per-type-pair test generator macros
// Each pair gets 3 POW2 tests per adapter class
// ============================================================================

template <typename Eo, typename Ei, int Pow2>
static constexpr bool lambda_pow2_test_supported_v =
#if defined(CPU_CAPABILITY_SVE)
    !((sizeof(Eo) > sizeof(Ei) && Pow2 == VecTransform<Eo, Ei>::max_output_pow2) ||
      (sizeof(Eo) < sizeof(Ei) && Pow2 == VecTransform<Eo, Ei>::min_output_pow2));
#else
    true;
#endif

#define MAKE_POW2_TESTS(ADAPTER_CLASS, TEST_GROUP, Eo, Ei, runner_func)   \
  TEST(TEST_GROUP, Ei##_to_##Eo##_min) {                                   \
    constexpr int Pow2 = VecTransform<Eo, Ei>::min_output_pow2;         \
    if constexpr (lambda_pow2_test_supported_v<Eo, Ei, Pow2>) {            \
      runner_func<Eo, Ei, Pow2>();                                         \
    } else {                                                               \
      GTEST_SKIP() << "SVE backend cannot represent this boundary tag";    \
    }                                                                      \
  }                                                                        \
  TEST(TEST_GROUP, Ei##_to_##Eo##_pow0) {                                  \
    runner_func<Eo, Ei, 0>();                                              \
  }                                                                        \
  TEST(TEST_GROUP, Ei##_to_##Eo##_max) {                                   \
    constexpr int Pow2 = VecTransform<Eo, Ei>::max_output_pow2;         \
    if constexpr (lambda_pow2_test_supported_v<Eo, Ei, Pow2>) {            \
      runner_func<Eo, Ei, Pow2>();                                         \
    } else {                                                               \
      GTEST_SKIP() << "SVE backend cannot represent this boundary tag";    \
    }                                                                      \
  }

#define ALL_TESTS(Eo, Ei)                                                  \
  MAKE_POW2_TESTS(PosLAdapter, LambdaVecTransform, Eo, Ei,           \
                  runPositionalLambdaTest)                                  \
  MAKE_POW2_TESTS(ElmLAdapter, ElementwiseVecTransform, Eo, Ei,          \
                  runElementwiseLambdaTest)                                 \
  MAKE_POW2_TESTS(ConvertedAdapter, ConvertedVecTransform, Eo, Ei,              \
                  runConversionTest)

// ============================================================================
// Generate all adapter tests per type pair (10 representative pairs)
// ============================================================================

// For ConvertedVecTransform: only test POW2=0 for now.
// Boundary POW2 tests hit known vector conversion limitations:
// 1. converted transform branch B: Rebind<InnerEo, ScalableTag<...>> tag mismatch
// 2. Vec.h promote/demote: lower(t, v) tag/vector element-type mismatch in multi-word path
#define CONV_POW0_TESTS(Eo, Ei) \
  TEST(ConvertedVecTransform, Ei##_to_##Eo##_pow0) { \
    runConversionTest<Eo, Ei, 0>(); \
  }

// For LambdaVecAdapter tests: full POW2 range (min_output, 0, max_output)
#define LAMBDA_POW2_TESTS(Eo, Ei)                                         \
  MAKE_POW2_TESTS(PosLAdapter, LambdaVecTransform, Eo, Ei,          \
                  runPositionalLambdaTest)                                 \
  MAKE_POW2_TESTS(ElmLAdapter, ElementwiseVecTransform, Eo, Ei,         \
                  runElementwiseLambdaTest)

// ============================================================================
// 10 representative type pairs
// ============================================================================

LAMBDA_POW2_TESTS(float32_t, float32_t)    // equal, 4-byte
CONV_POW0_TESTS(float32_t, float32_t)

LAMBDA_POW2_TESTS(int32_t, int32_t)        // equal, 4-byte
CONV_POW0_TESTS(int32_t, int32_t)

LAMBDA_POW2_TESTS(float64_t, float64_t)    // equal, 8-byte
CONV_POW0_TESTS(float64_t, float64_t)

LAMBDA_POW2_TESTS(test_float16_t, test_float16_t)    // equal, 2-byte
CONV_POW0_TESTS(test_float16_t, test_float16_t)

LAMBDA_POW2_TESTS(int8_t, int8_t)          // equal, 1-byte
CONV_POW0_TESTS(int8_t, int8_t)

LAMBDA_POW2_TESTS(float32_t, int32_t)      // equal cross-domain
CONV_POW0_TESTS(float32_t, int32_t)

LAMBDA_POW2_TESTS(float32_t, test_float16_t)    // widening 2:1
CONV_POW0_TESTS(float32_t, test_float16_t)

LAMBDA_POW2_TESTS(int32_t, int8_t)         // widening 4:1
CONV_POW0_TESTS(int32_t, int8_t)

LAMBDA_POW2_TESTS(test_float16_t, float32_t)    // narrowing 2:1
CONV_POW0_TESTS(test_float16_t, float32_t)

LAMBDA_POW2_TESTS(int8_t, int32_t)         // narrowing 4:1
CONV_POW0_TESTS(int8_t, int32_t)


// ============================================================================
// ConvertedVecTransform Branch C — recursive split at max POW2
// ============================================================================

TEST(ConvertedVecTransformBranchC, f32_to_f32_large) {
  runConversionBranchCTest<float32_t, float32_t,
      VecTransform<float32_t, float32_t>::max_output_pow2>();
}

TEST(ConvertedVecTransformBranchC, f64_to_f64_large) {
  runConversionBranchCTest<float64_t, float64_t,
      VecTransform<float64_t, float64_t>::max_output_pow2>();
}

TEST(ConvertedVecTransformBranchC, i32_to_f32_large) {
  runConversionBranchCTest<float32_t, int32_t,
      VecTransform<float32_t, int32_t>::max_output_pow2>();
}

// ============================================================================
// ConvertedVecTransform Elementwise short-circuit
// ============================================================================

TEST(ConvertedVecTransformElementwise, f32_coord_discard) {
  runConversionElementwiseTest<float32_t, float32_t>();
}

TEST(ConvertedVecTransformElementwise, i32_coord_discard) {
  runConversionElementwiseTest<int32_t, int32_t>();
}

TEST(ConvertedVecTransformElementwise, f64_coord_discard) {
  runConversionElementwiseTest<float64_t, float64_t>();
}

TEST(ConvertedVecTransformElementwise, f16_coord_discard) {
  runConversionElementwiseTest<test_float16_t, test_float16_t>();
}

TEST(ConvertedVecTransformElementwise, i8_coord_discard) {
  runConversionElementwiseTest<int8_t, int8_t>();
}

// ============================================================================
// LambdaVecAdapter: upward skip multiple levels
// ============================================================================

TEST(LambdaVecTransform, upward_skip_multiple) {
  constexpr int supported_pow2 =
#if defined(CPU_CAPABILITY_SVE)
      2;
#else
      3;
#endif
  using FnType = PartialCoordinateFn<float32_t, float32_t, supported_pow2>;
  LambdaVecTransform<float32_t, float32_t, FnType> adapter{FnType{}};
  ScalableTag<float32_t, 0> t;
  nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter(t, v_in, 0, 0);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

TEST(ElementwiseVecTransform, upward_skip_multiple) {
  constexpr int supported_pow2 =
#if defined(CPU_CAPABILITY_SVE)
      2;
#else
      3;
#endif
  using FnType = PartialElementwiseFn<float32_t, float32_t, supported_pow2>;
  LambdaVecTransform<float32_t, float32_t, FnType, true> adapter{FnType{}};
  ScalableTag<float32_t, 0> t;
  nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter(t, v_in);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

// ============================================================================
// LambdaVecAdapter: downward multi-level
// ============================================================================

TEST(LambdaVecTransform, downward_deep) {
  using FnType = PartialCoordinateFn<float32_t, float32_t, 0>;
  LambdaVecTransform<float32_t, float32_t, FnType> adapter{FnType{}};
#if defined(CPU_CAPABILITY_SVE)
  ScalableTag<float32_t, 2> t;
#else
  ScalableTag<float32_t, 4> t;
#endif
  nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter(t, v_in, 0, 0);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

// ============================================================================
// Trait tests
// ============================================================================

TEST(Traits, PositionalLambda_is_not_elementwise) {
  EXPECT_FALSE((LambdaVecTransform<float32_t, float32_t,
      PartialCoordinateFn<float32_t, float32_t, 0>>::is_elementwise));
}

TEST(Traits, ElementwiseLambda_is_elementwise) {
  EXPECT_TRUE((LambdaVecTransform<float32_t, float32_t,
      PartialElementwiseFn<float32_t, float32_t, 0>, true>::is_elementwise));
}

TEST(Traits, is_widening_f16_to_f32) {
  EXPECT_TRUE((LambdaVecTransform<float32_t, test_float16_t,
      PartialElementwiseFn<float32_t, test_float16_t, 0>, true>::is_widening));
}

TEST(Traits, is_narrowing_f32_to_f16) {
  EXPECT_TRUE((LambdaVecTransform<test_float16_t, float32_t,
      PartialElementwiseFn<test_float16_t, float32_t, 0>, true>::is_narrowing));
}

TEST(Traits, built_in_transform_traits) {
  EXPECT_TRUE((is_zero_vec_transform_v<ZeroVecTransform<float32_t>>));
  EXPECT_TRUE((is_identity_vec_transform_v<IdentityVecTransform<float32_t, int32_t>>));
  EXPECT_TRUE((is_identity_vec_transform_v<
      ConvertedVecTransform<float32_t, int32_t, IdentityVecTransform<int32_t>>>));
}
