#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "vecops/gemm/Attachment.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::vec;

// ============================================================================
// Helper: POW2-gated functor that implements Vec<T> op only at a specific POW2
// ============================================================================

template <typename Eo_, typename Ei_, int SupportedPow2>
struct PartialPositionedFn {
  template <typename To, TL_IF(vec::scalable_pow2_of<To> == SupportedPow2)>
  Vec<To> operator()(To, Vec<Rebind<Ei_, To>> v_in, nint_t, nint_t) const {
    return bitcast(To{}, xconvert(To{}, v_in));
  }
};

template <typename Eo_, typename Ei_, int SupportedPow2>
struct PartialElementwiseFn {
  template <typename To, TL_IF(vec::scalable_pow2_of<To> == SupportedPow2)>
  Vec<To> operator()(To, Vec<Rebind<Ei_, To>> v_in) const {
    return bitcast(To{}, xconvert(To{}, v_in));
  }
};

// ============================================================================
// Helper: allocate aligned memory and fill
// ============================================================================

template <typename T>
T* alloc_aligned(nint_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

template <typename T>
T get_value(int idx) {
  if constexpr (std::is_same_v<T, float32_t>) return static_cast<float32_t>(idx * 1.5f + 0.5f);
  else if constexpr (std::is_same_v<T, float64_t>) return static_cast<float64_t>(idx * 1.5 + 0.5);
  else if constexpr (std::is_same_v<T, int32_t>) return static_cast<int32_t>(idx * 3 + 1);
  else if constexpr (std::is_same_v<T, uint32_t>) return static_cast<uint32_t>(idx * 3 + 1);
  else return static_cast<T>(idx * 3 + 1);
}

template <typename T>
bool values_close(T expected, T actual, double tol = 0.01) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tol;
  } else {
    return expected == actual;
  }
}

// ============================================================================
// PositionalLambdaVecAdapter Tests
// ============================================================================

using Elem = float32_t;

TEST(PositionalLambdaVecAdapterTest, DirectMatch) {
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(PositionalLambdaVecAdapterTest, UpwardDispatch) {
  using FnType = PartialPositionedFn<Elem, Elem, 2>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(PositionalLambdaVecAdapterTest, DownwardDispatch) {
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 2> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(PositionalLambdaVecAdapterTest, MultiLevelDownward) {
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 3> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(PositionalLambdaVecAdapterTest, TypeConversion) {
  using FnType = PartialPositionedFn<float32_t, int32_t, 0>;
  PositionalLambdaVecAdapter<float32_t, int32_t, FnType> adapter{FnType{}};

  ScalableTag<float32_t, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf  = alloc_aligned<int32_t>(N);
  auto out_ref = alloc_aligned<float32_t>(N);
  auto out_buf = alloc_aligned<float32_t>(N);

  for (nint_t i = 0; i < N; ++i) {
    in_buf[i] = static_cast<int32_t>(i * 3 + 1);
    out_ref[i] = static_cast<float32_t>(in_buf[i]);
  }

  using InTag = ScalableTag<int32_t, 0>;
  InTag t_in;
  auto v_in = loadu(t_in, in_buf);
  auto v_out = adapter.call(t, bitcast(Rebind<int32_t, decltype(t)>{}, v_in), 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// ElementwiseLambdaVecAdapter Tests
// ============================================================================

TEST(ElementwiseLambdaVecAdapterTest, DirectMatch) {
  using FnType = PartialElementwiseFn<Elem, Elem, 0>;
  ElementwiseLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(ElementwiseLambdaVecAdapterTest, CoordinateDiscarding) {
  using FnType = PartialElementwiseFn<Elem, Elem, 0>;
  ElementwiseLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_ref = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_ref = adapter.call(t, v_in);
  auto v_out = adapter.call(t, v_in, 42, 99);
  storeu(t, out_ref, v_ref);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

TEST(ElementwiseLambdaVecAdapterTest, UpwardDispatch) {
  using FnType = PartialElementwiseFn<Elem, Elem, 2>;
  ElementwiseLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(ElementwiseLambdaVecAdapterTest, DownwardDispatch) {
  using FnType = PartialElementwiseFn<Elem, Elem, 0>;
  ElementwiseLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 2> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) in_buf[i] = get_value<Elem>(i);

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i])) << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_buf);
}

TEST(ElementwiseLambdaVecAdapterTest, IsElementwiseTrait) {
  EXPECT_TRUE((ElementwiseLambdaVecAdapter<Elem, Elem, PartialElementwiseFn<Elem, Elem, 0>>
      ::is_elementwise));
}
