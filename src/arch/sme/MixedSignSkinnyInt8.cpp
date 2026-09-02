//
// Copyright (c) vecops contributors.
//
// Ordinary-SVE mixed-sign INT8 skinny leaves for SME-capable multiarch
// targets. This source is intentionally excluded from generic libvecops.a and
// compiled in the ISA-specific SME matmul archive.

#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sve/Basic.h"

namespace vecops::kernel::matmul_details::sme {

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)

// TODO: Keep mixed-sign widening DOT SME-backend-local until signedness,
// operand ordering, and accumulator semantics have a second backend consumer.
// This uses only ordinary SVE data instructions; SME state remains owned by
// the existing vec SME resource wrapper.
template <typename TA, typename TB>
VECOPS_ALWAYS_INLINE svint32_t mixed_sign_dot_add(
    svint32_t accumulator, svint8_t signed_value,
    svuint8_t unsigned_value) {
  static_assert(
      (std::same_as<TA, int8_t> && std::same_as<TB, uint8_t>) ||
      (std::same_as<TA, uint8_t> && std::same_as<TB, int8_t>));
  if constexpr (std::same_as<TA, int8_t>) {
    return svsudot_s32(accumulator, signed_value, unsigned_value);
  } else {
    return svusdot_s32(accumulator, unsigned_value, signed_value);
  }
}

template <typename T>
VECOPS_ALWAYS_INLINE auto load_mixed_sign_input(
    const T* pointer, nint_t active) {
  using Tag = vec::ScalableTag<T, 0>;
  return vec::load(
      Tag{}, pointer, vec::opt::first(active), vec::opt::zero);
}

template <typename TA, typename TB>
VECOPS_ALWAYS_INLINE svint32_t mixed_sign_dot_values(
    svint32_t accumulator, vec::Vec<vec::ScalableTag<TA, 0>> a,
    vec::Vec<vec::ScalableTag<TB, 0>> b) {
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  if constexpr (std::same_as<TA, int8_t>) {
    return mixed_sign_dot_add<TA, TB>(accumulator, raw_a, raw_b);
  } else {
    return mixed_sign_dot_add<TA, TB>(accumulator, raw_b, raw_a);
  }
}

template <bool VaryRows, int Block, typename TA, typename TB>
VECOPS_ALWAYS_INLINE void mixed_sign_skinny_block(
    const TA* a, nint_t a_stride, const TB* b, nint_t b_stride,
    int32_t* output, nint_t output_stride, nint_t output_origin,
    nint_t logical_k) {
  static_assert(1 <= Block && Block <= 8);
  using ATag = vec::ScalableTag<TA, 0>;
  using AccTag = vec::ScalableTag<int32_t, 0>;

  auto sum0 = svdup_s32(0);
  auto sum1 = svdup_s32(0);
  auto sum2 = svdup_s32(0);
  auto sum3 = svdup_s32(0);
  auto sum4 = svdup_s32(0);
  auto sum5 = svdup_s32(0);
  auto sum6 = svdup_s32(0);
  auto sum7 = svdup_s32(0);

  nint_t kk = 0;
  nint_t remaining = logical_k;
  VECOPS_LOOP_ALIGN(64) while (remaining > 0) {
    const nint_t active = vecops::min(vec::size(ATag{}), remaining);
    const auto load_a = [&](nint_t row) VECOPS_INLINE_LAMBDA {
      return load_mixed_sign_input(a + row * a_stride + kk, active);
    };
    const auto load_b = [&](nint_t row) VECOPS_INLINE_LAMBDA {
      return load_mixed_sign_input(b + row * b_stride + kk, active);
    };
    if constexpr (VaryRows) {
      const auto shared = load_b(0);
      sum0 = mixed_sign_dot_values<TA, TB>(
          sum0, load_a(output_origin), shared);
      if constexpr (Block >= 2)
        sum1 = mixed_sign_dot_values<TA, TB>(
            sum1, load_a(output_origin + 1), shared);
      if constexpr (Block >= 3)
        sum2 = mixed_sign_dot_values<TA, TB>(
            sum2, load_a(output_origin + 2), shared);
      if constexpr (Block >= 4)
        sum3 = mixed_sign_dot_values<TA, TB>(
            sum3, load_a(output_origin + 3), shared);
      if constexpr (Block >= 5)
        sum4 = mixed_sign_dot_values<TA, TB>(
            sum4, load_a(output_origin + 4), shared);
      if constexpr (Block >= 6)
        sum5 = mixed_sign_dot_values<TA, TB>(
            sum5, load_a(output_origin + 5), shared);
      if constexpr (Block >= 7)
        sum6 = mixed_sign_dot_values<TA, TB>(
            sum6, load_a(output_origin + 6), shared);
      if constexpr (Block >= 8)
        sum7 = mixed_sign_dot_values<TA, TB>(
            sum7, load_a(output_origin + 7), shared);
    } else {
      const auto shared = load_a(0);
      sum0 = mixed_sign_dot_values<TA, TB>(
          sum0, shared, load_b(output_origin));
      if constexpr (Block >= 2)
        sum1 = mixed_sign_dot_values<TA, TB>(
            sum1, shared, load_b(output_origin + 1));
      if constexpr (Block >= 3)
        sum2 = mixed_sign_dot_values<TA, TB>(
            sum2, shared, load_b(output_origin + 2));
      if constexpr (Block >= 4)
        sum3 = mixed_sign_dot_values<TA, TB>(
            sum3, shared, load_b(output_origin + 3));
      if constexpr (Block >= 5)
        sum4 = mixed_sign_dot_values<TA, TB>(
            sum4, shared, load_b(output_origin + 4));
      if constexpr (Block >= 6)
        sum5 = mixed_sign_dot_values<TA, TB>(
            sum5, shared, load_b(output_origin + 5));
      if constexpr (Block >= 7)
        sum6 = mixed_sign_dot_values<TA, TB>(
            sum6, shared, load_b(output_origin + 6));
      if constexpr (Block >= 8)
        sum7 = mixed_sign_dot_values<TA, TB>(
            sum7, shared, load_b(output_origin + 7));
    }
    kk += active;
    remaining -= active;
  }

  const auto reduce = [&](svint32_t value) VECOPS_INLINE_LAMBDA {
    return vec::reduce_add(
        AccTag{}, vec::details::sve_basic_wrap_word<AccTag>(value));
  };
  const auto store = [&](nint_t logical, int32_t value)
      VECOPS_INLINE_LAMBDA {
    output[VaryRows ? logical * output_stride : logical] = value;
  };
  store(output_origin, reduce(sum0));
  if constexpr (Block >= 2) store(output_origin + 1, reduce(sum1));
  if constexpr (Block >= 3) store(output_origin + 2, reduce(sum2));
  if constexpr (Block >= 4) store(output_origin + 3, reduce(sum3));
  if constexpr (Block >= 5) store(output_origin + 4, reduce(sum4));
  if constexpr (Block >= 6) store(output_origin + 5, reduce(sum5));
  if constexpr (Block >= 7) store(output_origin + 6, reduce(sum6));
  if constexpr (Block >= 8) store(output_origin + 7, reduce(sum7));
}

template <bool VaryRows, typename TA, typename TB>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void mixed_sign_skinny_compute(
    const TA* a, nint_t a_stride, const TB* b, nint_t b_stride,
    int32_t* output, nint_t output_stride, nint_t outputs,
    nint_t logical_k) {
  nint_t origin = 0;
  VECOPS_LOOP_ALIGN(64) for (; origin + 8 <= outputs; origin += 8) {
    mixed_sign_skinny_block<VaryRows, 8>(
        a, a_stride, b, b_stride, output, output_stride,
        origin, logical_k);
  }
  VECOPS_LOOP_ALIGN(64) for (; origin + 4 <= outputs; origin += 4) {
    mixed_sign_skinny_block<VaryRows, 4>(
        a, a_stride, b, b_stride, output, output_stride,
        origin, logical_k);
  }
  switch (outputs - origin) {
    case 3:
      mixed_sign_skinny_block<VaryRows, 3>(
          a, a_stride, b, b_stride, output, output_stride,
          origin, logical_k);
      break;
    case 2:
      mixed_sign_skinny_block<VaryRows, 2>(
          a, a_stride, b, b_stride, output, output_stride,
          origin, logical_k);
      break;
    case 1:
      mixed_sign_skinny_block<VaryRows, 1>(
          a, a_stride, b, b_stride, output, output_stride,
          origin, logical_k);
      break;
    default:
      break;
  }
}

template <typename TA, typename TB>
VECOPS_ALWAYS_INLINE bool try_mixed_sign_skinny(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const TA* a, nint_t a_stride, const TB* b, nint_t b_stride,
    int32_t* output, nint_t output_stride) {
  if (logical_k < 0) return false;
  if (logical_m == 1 && 0 <= logical_n && logical_n <= 64) {
    mixed_sign_skinny_compute<false>(
        a, a_stride, b, b_stride, output, output_stride,
        logical_n, logical_k);
    return true;
  }
  if (logical_n == 1 && 0 <= logical_m && logical_m <= 64) {
    mixed_sign_skinny_compute<true>(
        a, a_stride, b, b_stride, output, output_stride,
        logical_m, logical_k);
    return true;
  }
  return false;
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) bool
try_raw_mixed_sign_skinny_s8u8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const int8_t* a, nint_t a_stride,
    const uint8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride) {
  return try_mixed_sign_skinny(
      logical_m, logical_n, logical_k,
      a, a_stride, b, b_stride, output, output_stride);
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) bool
try_raw_mixed_sign_skinny_u8s8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const uint8_t* a, nint_t a_stride,
    const int8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride) {
  return try_mixed_sign_skinny(
      logical_m, logical_n, logical_k,
      a, a_stride, b, b_stride, output, output_stride);
}

#endif

} // namespace vecops::kernel::matmul_details::sme
