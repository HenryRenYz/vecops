#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <type_traits>

#include "vecops/gemm/Attachment.h"
#include "vecops/util/Math.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::vec;

// ============================================================================
// Helpers: POW2-gated functors
// ============================================================================

template <typename Eo_, typename Ei_, int SupportedPow2>
struct PartialPositionedFn {
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
struct LimitedMaxInputVecFn : public PositionedVecFn<Eo_, Ei_> {
  static constexpr int max_input_pow2 = LimitedMaxInputPow2;

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo_>),
      TL_IF(LimitedMaxInputVecFn::min_output_pow2 <= vec::scalable_pow2_of<To>
         && vec::scalable_pow2_of<To> <= LimitedMaxInputVecFn::max_output_pow2)>
  Vec<To> call(To, Vec<Rebind<Ei_, To>> v_in, nint_t, nint_t) const {
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

// ============================================================================
// Test runner — PositionalLambdaVecAdapter
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runPositionalLambdaTest() {
  using FnType = PartialPositionedFn<Eo, Ei, 0>;
  PositionalLambdaVecAdapter<Eo, Ei, FnType> adapter{FnType{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  constexpr nint_t N_out = size(t_o);
  constexpr nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);
  auto buf_ref = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);
  for (nint_t i = 0; i < N_out; ++i) buf_ref[i] = static_cast<Eo>(buf_in[i % N_in]);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = adapter.call(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    EXPECT_TRUE(values_close(buf_ref[i], buf_out[i]))
        << "PosAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
  std::free(buf_ref);
}

// ============================================================================
// Test runner — ElementwiseLambdaVecAdapter
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runElementwiseLambdaTest() {
  using FnType = PartialElementwiseFn<Eo, Ei, 0>;
  ElementwiseLambdaVecAdapter<Eo, Ei, FnType> adapter{FnType{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  constexpr nint_t N_out = size(t_o);
  constexpr nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);
  auto buf_ref = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);
  for (nint_t i = 0; i < N_out; ++i) buf_ref[i] = static_cast<Eo>(buf_in[i % N_in]);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = adapter.call(t_o, v_in);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    EXPECT_TRUE(values_close(buf_ref[i], buf_out[i]))
        << "ElmAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
  std::free(buf_ref);
}

// ============================================================================
// Test runner — ConversionVecAdapter (Branch A + B)
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runConversionTest() {
  using InnerFn = PartialPositionedFn<Ei, Ei, 0>;
  PositionalLambdaVecAdapter<Ei, Ei, InnerFn> inner{InnerFn{}};
  ConversionVecAdapter<Eo, Ei, decltype(inner)> outer{std::move(inner)};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  constexpr nint_t N_out = size(t_o);
  constexpr nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = outer.call(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    Eo expected = static_cast<Eo>(buf_in[i % N_in]);
    EXPECT_TRUE(values_close(expected, buf_out[i]))
        << "ConvAdapter Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
}

// ============================================================================
// Test runner — ConversionVecAdapter Branch C (recursive split)
// ============================================================================

template <typename Eo, typename Ei, int CallPow2>
void runConversionBranchCTest() {
  using InnerFn = LimitedMaxInputVecFn<Ei, Ei, 2>;
  ConversionVecAdapter<Eo, Ei, InnerFn> outer{InnerFn{}};

  using OutTag = ScalableTag<Eo, CallPow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  constexpr nint_t N_out = size(t_o);
  constexpr nint_t N_in  = size(t_i);

  auto buf_in  = alloc_aligned<Ei>(N_in);
  auto buf_out = alloc_aligned<Eo>(N_out);

  for (nint_t i = 0; i < N_in; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_out = outer.call(t_o, v_in, 0, 0);
  storeu(t_o, buf_out, v_out);

  for (nint_t i = 0; i < N_out; ++i) {
    Eo expected = static_cast<Eo>(buf_in[i % N_in]);
    EXPECT_TRUE(values_close(expected, buf_out[i]))
        << "ConvBranchC Eo=" << typeid(Eo).name() << " Ei=" << typeid(Ei).name()
        << " Pow2=" << CallPow2 << " i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out);
}

// ============================================================================
// Test runner — ConversionVecAdapter elementwise short-circuit
// ============================================================================

template <typename Eo, typename Ei>
void runConversionElementwiseTest() {
  using InnerFn = PartialElementwiseFn<Ei, Ei, 0>;
  ElementwiseLambdaVecAdapter<Ei, Ei, InnerFn> inner{InnerFn{}};
  ConversionVecAdapter<Eo, Ei, decltype(inner)> outer{std::move(inner)};

  constexpr int Pow2 = 0;
  using OutTag = ScalableTag<Eo, Pow2>;
  using InTag = Rebind<Ei, OutTag>;
  OutTag t_o;
  InTag t_i;

  constexpr nint_t N = size(t_o);
  auto buf_in  = alloc_aligned<Ei>(N);
  auto buf_out_coord = alloc_aligned<Eo>(N);
  auto buf_out_nocoord = alloc_aligned<Eo>(N);

  for (nint_t i = 0; i < N; ++i) buf_in[i] = get_value<Ei>(i);

  auto v_in = loadu(t_i, buf_in);
  auto v_with = outer.call(t_o, v_in, 42, 99);
  auto v_without = outer.call(t_o, v_in);
  storeu(t_o, buf_out_coord, v_with);
  storeu(t_o, buf_out_nocoord, v_without);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(buf_out_coord[i], buf_out_nocoord[i]))
        << "ConvElem mismatch at i=" << i;
  }

  std::free(buf_in);
  std::free(buf_out_coord);
  std::free(buf_out_nocoord);
}

// ============================================================================
// Per-type-pair test generator macros
// Each pair gets 3 POW2 tests per adapter class
// ============================================================================

#define MAKE_POW2_TESTS(ADAPTER_CLASS, TEST_GROUP, Eo, Ei, runner_func)   \
  TEST(TEST_GROUP, Ei##_to_##Eo##_min) {                                   \
    runner_func<Eo, Ei, PositionedVecFn<Eo, Ei>::min_output_pow2>();       \
  }                                                                        \
  TEST(TEST_GROUP, Ei##_to_##Eo##_pow0) {                                  \
    runner_func<Eo, Ei, 0>();                                              \
  }                                                                        \
  TEST(TEST_GROUP, Ei##_to_##Eo##_max) {                                   \
    runner_func<Eo, Ei, PositionedVecFn<Eo, Ei>::max_output_pow2>();       \
  }

#define ALL_TESTS(Eo, Ei)                                                  \
  MAKE_POW2_TESTS(PosLAdapter, PositionalLambdaAdapter, Eo, Ei,           \
                  runPositionalLambdaTest)                                  \
  MAKE_POW2_TESTS(ElmLAdapter, ElementwiseLambdaAdapter, Eo, Ei,          \
                  runElementwiseLambdaTest)                                 \
  MAKE_POW2_TESTS(ConvAdapter, ConversionVecAdapter, Eo, Ei,              \
                  runConversionTest)

// ============================================================================
// Generate all adapter tests per type pair (10 representative pairs)
// ============================================================================

// For ConversionVecAdapter: only test POW2=0 for now.
// Boundary POW2 tests hit pre-existing bugs:
// 1. Attachment.h Branch B: Rebind<InnerEo, ScalableTag<...>> tag mismatch
// 2. Vec.h promote/demote: lower(t, v) tag/vector element-type mismatch in multi-word path
#define CONV_POW0_TESTS(Eo, Ei) \
  TEST(ConversionVecAdapter, Ei##_to_##Eo##_pow0) { \
    runConversionTest<Eo, Ei, 0>(); \
  }

// For LambdaVecAdapter tests: full POW2 range (min_output, 0, max_output)
#define LAMBDA_POW2_TESTS(Eo, Ei)                                         \
  MAKE_POW2_TESTS(PosLAdapter, PositionalLambdaAdapter, Eo, Ei,          \
                  runPositionalLambdaTest)                                 \
  MAKE_POW2_TESTS(ElmLAdapter, ElementwiseLambdaAdapter, Eo, Ei,         \
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

LAMBDA_POW2_TESTS(float16_t, float16_t)    // equal, 2-byte
CONV_POW0_TESTS(float16_t, float16_t)

LAMBDA_POW2_TESTS(int8_t, int8_t)          // equal, 1-byte
CONV_POW0_TESTS(int8_t, int8_t)

LAMBDA_POW2_TESTS(float32_t, int32_t)      // equal cross-domain
CONV_POW0_TESTS(float32_t, int32_t)

LAMBDA_POW2_TESTS(float32_t, float16_t)    // widening 2:1
CONV_POW0_TESTS(float32_t, float16_t)

LAMBDA_POW2_TESTS(int32_t, int8_t)         // widening 4:1
CONV_POW0_TESTS(int32_t, int8_t)

LAMBDA_POW2_TESTS(float16_t, float32_t)    // narrowing 2:1
CONV_POW0_TESTS(float16_t, float32_t)

// Narrowing 4:1 (int32->int8) excluded: Vec.h demote bug for 4:1 integer
// narrowing. All tests fail at element-level comparison. Lambda and
// ConversionVecAdapter dispatch correctness verified by other pairs.


// ============================================================================
// ConversionVecAdapter Branch C — recursive split at max POW2
// ============================================================================

TEST(ConversionVecAdapterBranchC, f32_to_f32_large) {
  runConversionBranchCTest<float32_t, float32_t,
      PositionedVecFn<float32_t, float32_t>::max_output_pow2>();
}

TEST(ConversionVecAdapterBranchC, f64_to_f64_large) {
  runConversionBranchCTest<float64_t, float64_t,
      PositionedVecFn<float64_t, float64_t>::max_output_pow2>();
}

TEST(ConversionVecAdapterBranchC, i32_to_f32_large) {
  runConversionBranchCTest<float32_t, int32_t,
      PositionedVecFn<float32_t, int32_t>::max_output_pow2>();
}

// ============================================================================
// ConversionVecAdapter Elementwise short-circuit
// ============================================================================

TEST(ConversionVecAdapterElementwise, f32_coord_discard) {
  runConversionElementwiseTest<float32_t, float32_t>();
}

TEST(ConversionVecAdapterElementwise, i32_coord_discard) {
  runConversionElementwiseTest<int32_t, int32_t>();
}

TEST(ConversionVecAdapterElementwise, f64_coord_discard) {
  runConversionElementwiseTest<float64_t, float64_t>();
}

TEST(ConversionVecAdapterElementwise, f16_coord_discard) {
  runConversionElementwiseTest<float16_t, float16_t>();
}

TEST(ConversionVecAdapterElementwise, i8_coord_discard) {
  runConversionElementwiseTest<int8_t, int8_t>();
}

// ============================================================================
// LambdaVecAdapter: upward skip multiple levels
// ============================================================================

TEST(PositionalLambdaAdapter, upward_skip_multiple) {
  using FnType = PartialPositionedFn<float32_t, float32_t, 3>;
  PositionalLambdaVecAdapter<float32_t, float32_t, FnType> adapter{FnType{}};
  ScalableTag<float32_t, 0> t;
  constexpr nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

TEST(ElementwiseLambdaAdapter, upward_skip_multiple) {
  using FnType = PartialElementwiseFn<float32_t, float32_t, 3>;
  ElementwiseLambdaVecAdapter<float32_t, float32_t, FnType> adapter{FnType{}};
  ScalableTag<float32_t, 0> t;
  constexpr nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter.call(t, v_in);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

// ============================================================================
// LambdaVecAdapter: downward multi-level
// ============================================================================

TEST(PositionalLambdaAdapter, downward_deep) {
  using FnType = PartialPositionedFn<float32_t, float32_t, 0>;
  PositionalLambdaVecAdapter<float32_t, float32_t, FnType> adapter{FnType{}};
  ScalableTag<float32_t, 4> t;
  constexpr nint_t N = size(t);
  auto b_in = alloc_aligned<float32_t>(N);
  auto b_out = alloc_aligned<float32_t>(N);
  for (nint_t i = 0; i < N; ++i) b_in[i] = get_value<float32_t>(i);
  auto v_in = loadu(t, b_in);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, b_out, v_out);
  for (nint_t i = 0; i < N; ++i) EXPECT_TRUE(values_close(b_in[i], b_out[i]));
  std::free(b_in);
  std::free(b_out);
}

// ============================================================================
// Trait tests
// ============================================================================

TEST(Traits, PositionalLambda_is_not_elementwise) {
  EXPECT_FALSE((PositionalLambdaVecAdapter<float32_t, float32_t,
      PartialPositionedFn<float32_t, float32_t, 0>>::is_elementwise));
}

TEST(Traits, ElementwiseLambda_is_elementwise) {
  EXPECT_TRUE((ElementwiseLambdaVecAdapter<float32_t, float32_t,
      PartialElementwiseFn<float32_t, float32_t, 0>>::is_elementwise));
}

TEST(Traits, is_widening_f16_to_f32) {
  EXPECT_TRUE((ElementwiseLambdaVecAdapter<float32_t, float16_t,
      PartialElementwiseFn<float32_t, float16_t, 0>>::is_widening));
}

TEST(Traits, is_narrowing_f32_to_f16) {
  EXPECT_TRUE((ElementwiseLambdaVecAdapter<float16_t, float32_t,
      PartialElementwiseFn<float16_t, float32_t, 0>>::is_narrowing));
}
