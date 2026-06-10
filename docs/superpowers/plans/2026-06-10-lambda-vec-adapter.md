# Lambda Vec Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PositionalLambdaVecAdapter` and `ElementwiseLambdaVecAdapter` to `Attachment.h` and test them.

**Architecture:** Two template classes wrapping a lambda/functor `Fn` as a `PositionedVecFn` facade. At call time, auto-dispatch to the nearest supported POW2 via upward bitcast or downward split. SFINAE traits detect which POW2 levels `Fn` supports.

**Tech Stack:** C++17 (`if constexpr`, `std::void_t`), vecops SIMD vectors, gtest with `add_multiarch_execute(ARCH Native)`

---

### File Structure

- **Modify:** `include/vecops/gemm/Attachment.h` — Add SFINAE traits + two adapter classes
- **Create:** `tests/gemm/AttachmentTest.cpp` — Tests
- **Modify:** `tests/CMakeLists.txt` — Register new test

---

### Task 1: Add SFINAE detection traits to `Attachment.h`

**Files:**
- Modify: `include/vecops/gemm/Attachment.h`

- [ ] **Step 1: Add `#include <type_traits>` and `namespace details` with SFINAE traits**

After the existing includes and before `namespace vecops::gemm`, add the SFINAE detection helpers.

```cpp
#include <type_traits>

namespace vecops::gemm {
namespace details {

template <typename To, typename Fn, typename Ei, typename = void>
struct can_call_positional : std::false_type {};

template <typename To, typename Fn, typename Ei>
struct can_call_positional<To, Fn, Ei, std::void_t<decltype(
    std::declval<Fn>()(std::declval<To>(),
                       std::declval<vec::Vec<vec::Rebind<Ei, To>>>(),
                       std::declval<nint_t>(),
                       std::declval<nint_t>())
)>> : std::true_type {};

template <typename To, typename Fn, typename Ei, typename = void>
struct can_call_elementwise : std::false_type {};

template <typename To, typename Fn, typename Ei>
struct can_call_elementwise<To, Fn, Ei, std::void_t<decltype(
    std::declval<Fn>()(std::declval<To>(),
                       std::declval<vec::Vec<vec::Rebind<Ei, To>>>())
)>> : std::true_type {};

} // namespace details
```

- [ ] **Step 2: Build to verify no compile errors**

Run: `cmake --build build --target vecops`
Expected: Compiles successfully.

- [ ] **Step 3: Commit**

```bash
git add include/vecops/gemm/Attachment.h
git commit -m "feat: add SFINAE detection traits for lambda adapters"
```

---

### Task 2: Add `PositionalLambdaVecAdapter` class

**Files:**
- Modify: `include/vecops/gemm/Attachment.h`

- [ ] **Step 1: Add the class after `ConversionVecAdapter` (before closing `}` of namespace)**

```cpp
/**
 * Lambda适配器，将仿函数Fn包装为PositionedVecFn外观。
 * Fn可能只实现了部分POW2级别的转换，本适配器通过向上bitcast和向下拆分自动找到可用的POW2级别。
 */
template <typename Eo, typename Ei, typename Fn>
struct PositionalLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {

  PositionalLambdaVecAdapter(Fn&& fn) : _fn(std::move(fn)) { }
  PositionalLambdaVecAdapter(const Fn& fn) : _fn(fn) { }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(PositionalLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= PositionalLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_positional<To, Fn, Ei>::value) {
      return _fn(t, v_in, x, y);
    } else {
      return _try_upward<To, pow2 + 1>(t, v_in, x, y);
    }
  }

private:
  Fn _fn;

  template <typename To, int TryPow2>
  vec::Vec<To> _try_upward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    if constexpr (TryPow2 > PositionalLambdaVecAdapter::max_input_pow2) {
      return _try_downward<To>(t, v_in, x, y);
    } else {
      using TryOut = vec::ScalableTag<Eo, TryPow2>;
      using TryIn  = vec::Rebind<Ei, TryOut>;

      if constexpr (details::can_call_positional<TryOut, Fn, Ei>::value) {
        auto cast_in  = vec::bitcast(TryIn{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in, x, y);
        return vec::bitcast(To{}, cast_out);
      } else {
        return _try_upward<To, TryPow2 + 1>(t, v_in, x, y);
      }
    }
  }

  template <typename To>
  vec::Vec<To> _try_downward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_positional<To, Fn, Ei>::value) {
      return _fn(t, v_in, x, y);
    } else if constexpr (pow2 > PositionalLambdaVecAdapter::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti     = vec::Rebind<Ei, To>;
      HalfTo t_h;
      auto v_lo = _try_downward<HalfTo>(
          t_h, vec::lower(Ti{}, v_in), x, y);
      auto v_hi = _try_downward<HalfTo>(
          t_h, vec::upper(Ti{}, v_in), x, y);
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > PositionalLambdaVecAdapter::min_input_pow2,
          "PositionalLambdaVecAdapter: Fn does not support any POW2 in range");
      return {};
    }
  }
};
```

- [ ] **Step 2: Build to verify no compile errors**

Run: `cmake --build build --target vecops`
Expected: Compiles successfully.

- [ ] **Step 3: Commit**

```bash
git add include/vecops/gemm/Attachment.h
git commit -m "feat: add PositionalLambdaVecAdapter"
```

---

### Task 3: Add `ElementwiseLambdaVecAdapter` class

**Files:**
- Modify: `include/vecops/gemm/Attachment.h`

- [ ] **Step 1: Add the class after `PositionalLambdaVecAdapter`**

```cpp
/**
 * 点对点Lambda适配器，将仿函数Fn包装为ElementwiseVecFn外观。
 */
template <typename Eo, typename Ei, typename Fn>
struct ElementwiseLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {
  static constexpr bool is_elementwise = true;

  ElementwiseLambdaVecAdapter(Fn&& fn) : _fn(std::move(fn)) { }
  ElementwiseLambdaVecAdapter(const Fn& fn) : _fn(fn) { }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_elementwise<To, Fn, Ei>::value) {
      return _fn(t, v_in);
    } else {
      return _try_upward<To, pow2 + 1>(t, v_in);
    }
  }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    ((void) x);
    ((void) y);
    return this->call(t, v_in);
  }

private:
  Fn _fn;

  template <typename To, int TryPow2>
  vec::Vec<To> _try_upward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    if constexpr (TryPow2 > ElementwiseLambdaVecAdapter::max_input_pow2) {
      return _try_downward<To>(t, v_in);
    } else {
      using TryOut = vec::ScalableTag<Eo, TryPow2>;
      using TryIn  = vec::Rebind<Ei, TryOut>;

      if constexpr (details::can_call_elementwise<TryOut, Fn, Ei>::value) {
        auto cast_in  = vec::bitcast(TryIn{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in);
        return vec::bitcast(To{}, cast_out);
      } else {
        return _try_upward<To, TryPow2 + 1>(t, v_in);
      }
    }
  }

  template <typename To>
  vec::Vec<To> _try_downward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_elementwise<To, Fn, Ei>::value) {
      return _fn(t, v_in);
    } else if constexpr (pow2 > ElementwiseLambdaVecAdapter::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti     = vec::Rebind<Ei, To>;
      HalfTo t_h;
      auto v_lo = _try_downward<HalfTo>(
          t_h, vec::lower(Ti{}, v_in));
      auto v_hi = _try_downward<HalfTo>(
          t_h, vec::upper(Ti{}, v_in));
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > ElementwiseLambdaVecAdapter::min_input_pow2,
          "ElementwiseLambdaVecAdapter: Fn does not support any POW2 in range");
      return {};
    }
  }
};
```

- [ ] **Step 2: Build to verify no compile errors**

Run: `cmake --build build --target vecops`
Expected: Compiles successfully.

- [ ] **Step 3: Commit**

```bash
git add include/vecops/gemm/Attachment.h
git commit -m "feat: add ElementwiseLambdaVecAdapter"
```

---

### Task 4: Write tests for `PositionalLambdaVecAdapter`

**Files:**
- Create: `tests/gemm/AttachmentTest.cpp`

- [ ] **Step 1: Create test file**

```cpp
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
  using TIn  = Ei_;
  using TOut = Eo_;

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
// Test: PositionalLambdaVecAdapter — Direct Match
// ============================================================================

using Elem = float32_t;

TEST(PositionalLambdaVecAdapterTest, DirectMatch) {
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_ref = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) {
    in_buf[i] = get_value<Elem>(i);
    out_ref[i] = in_buf[i];  // identity reference (same-size convert)
  }

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]))
        << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// Test: PositionalLambdaVecAdapter — Upward Dispatch
// ============================================================================

TEST(PositionalLambdaVecAdapterTest, UpwardDispatch) {
  // Fn only supports POW2=2, test calling with POW2=0 → bitcast up to 2
  using FnType = PartialPositionedFn<Elem, Elem, 2>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 0> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_ref = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) {
    in_buf[i] = get_value<Elem>(i);
    out_ref[i] = in_buf[i];
  }

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]))
        << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// Test: PositionalLambdaVecAdapter — Downward Dispatch
// ============================================================================

TEST(PositionalLambdaVecAdapterTest, DownwardDispatch) {
  // Fn only supports POW2=0, test calling with POW2=2 → split to 0
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 2> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_ref = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) {
    in_buf[i] = get_value<Elem>(i);
    out_ref[i] = in_buf[i];
  }

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]))
        << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// Test: PositionalLambdaVecAdapter — Multi-level Downward
// ============================================================================

TEST(PositionalLambdaVecAdapterTest, MultiLevelDownward) {
  // Fn only supports POW2=0, test calling with POW2=3 → split twice
  using FnType = PartialPositionedFn<Elem, Elem, 0>;
  PositionalLambdaVecAdapter<Elem, Elem, FnType> adapter{FnType{}};

  ScalableTag<Elem, 3> t;
  constexpr nint_t N = size(t);
  auto in_buf = alloc_aligned<Elem>(N);
  auto out_ref = alloc_aligned<Elem>(N);
  auto out_buf = alloc_aligned<Elem>(N);

  for (nint_t i = 0; i < N; ++i) {
    in_buf[i] = get_value<Elem>(i);
    out_ref[i] = in_buf[i];
  }

  auto v_in = loadu(t, in_buf);
  auto v_out = adapter.call(t, v_in, 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]))
        << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// Test: PositionalLambdaVecAdapter — Type Conversion (Ei != Eo)
// ============================================================================

TEST(PositionalLambdaVecAdapterTest, TypeConversion) {
  // Fn: int32_t → float32_t at POW2=0
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

  ScalableTag<int32_t, 0> t_in;
  auto v_in = loadu(t_in, in_buf);
  auto v_out = adapter.call(t, bitcast(Rebind<int32_t, decltype(t)>{}, v_in), 0, 0);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]))
        << "Mismatch at index " << i;
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}
```

- [ ] **Step 2: Build the test**

Run: `cmake --build build --target AttachmentTest-Native`
Expected: Compiles successfully.

- [ ] **Step 3: Run the test**

Run: `./build/tests/AttachmentTest-Native`
Expected: All 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/gemm/AttachmentTest.cpp tests/CMakeLists.txt
git commit -m "test: add PositionalLambdaVecAdapter tests"
```

---

### Task 5: Write tests for `ElementwiseLambdaVecAdapter`

**Files:**
- Modify: `tests/gemm/AttachmentTest.cpp`

- [ ] **Step 1: Append tests to the test file**

```cpp
// ============================================================================
// Test: ElementwiseLambdaVecAdapter — Direct Match
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
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i]));
  }

  std::free(in_buf);
  std::free(out_buf);
}

// ============================================================================
// Test: ElementwiseLambdaVecAdapter — Coordinate-Discarding Adapter
// ============================================================================

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
  auto v_ref = adapter.call(t, v_in);            // coord-free
  auto v_out = adapter.call(t, v_in, 42, 99);    // with coords (should ignore)
  storeu(t, out_ref, v_ref);
  storeu(t, out_buf, v_out);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(values_close(out_ref[i], out_buf[i]));
  }

  std::free(in_buf);
  std::free(out_ref);
  std::free(out_buf);
}

// ============================================================================
// Test: ElementwiseLambdaVecAdapter — Upward Dispatch
// ============================================================================

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
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i]));
  }

  std::free(in_buf);
  std::free(out_buf);
}

// ============================================================================
// Test: ElementwiseLambdaVecAdapter — Downward Dispatch
// ============================================================================

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
    EXPECT_TRUE(values_close(in_buf[i], out_buf[i]));
  }

  std::free(in_buf);
  std::free(out_buf);
}

// ============================================================================
// Test: ElementwiseLambdaVecAdapter — is_elementwise trait
// ============================================================================

TEST(ElementwiseLambdaVecAdapterTest, IsElementwiseTrait) {
  EXPECT_TRUE((ElementwiseLambdaVecAdapter<Elem, Elem, PartialElementwiseFn<Elem, Elem, 0>>::is_elementwise));
}
```

- [ ] **Step 2: Build and run the test**

Run: `cmake --build build --target AttachmentTest-Native && ./build/tests/AttachmentTest-Native`
Expected: All tests PASS (5 positioned + 6 elementwise = 11 total).

- [ ] **Step 3: Commit**

```bash
git add tests/gemm/AttachmentTest.cpp
git commit -m "test: add ElementwiseLambdaVecAdapter tests"
```

---

### Task 6: Register test in CMakeLists.txt

**Files:**
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add the test registration**

After the existing test registrations, add:

```cmake
add_multiarch_executable(AttachmentTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/gemm/AttachmentTest.cpp
    ARCH Native
)
```

- [ ] **Step 2: Reconfigure and build**

Run: `cmake -B build && cmake --build build --target AttachmentTest-Native`
Expected: Configures and builds successfully.

- [ ] **Step 3: Run all tests**

Run: `./build/tests/AttachmentTest-Native`
Expected: All 11 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "build: register AttachmentTest in CMake"
```

---

### Task 7: Fix any bugs in existing Attachment.h found during testing

**Files:**
- Modify: `include/vecops/gemm/Attachment.h` (if bugs found)

- [ ] **Step 1: Investigate any compilation or test failures**

If compilation or test failures reveal bugs in the existing `Attachment.h` code (e.g., `ConversionVecAdapter` split logic), report them first before fixing. Only fix bugs that are:
- In `Attachment.h` (not external dependencies)
- Not structural/architectural (fixable locally)

- [ ] **Step 2: Apply fixes if applicable**

Fix bugs, rebuild, re-run tests.

- [ ] **Step 3: Commit fixes**

```bash
git add include/vecops/gemm/Attachment.h
git commit -m "fix: correct bug in AttachMent.h found during testing"
```
