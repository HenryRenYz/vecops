#ifndef VECOPS_VEC_DETAILS_X86_MATH_H
#define VECOPS_VEC_DETAILS_X86_MATH_H

/**
 * @file Math.h
 * @brief x86 backend implementations for math operations (exp family).
 */

#include <cmath>
#include <cstring>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                Exponential helpers and word implementations                //
/* **************************************************************************** */

template <nint_t Index, FloatingTag Tag>
struct X86ExpContext {
  using T = ElementOf<Tag>;
  using Word = NativeWordVec<Tag>;
  using MaskWord = NativeWordMask<Tag>;
  using ITag = IndexTag<Tag>;
  using IWord = NativeWordVec<ITag>;

  static VECOPS_ALWAYS_INLINE Word fill(Tag tag, T value) {
    return execute_word<Index, X86Backend>(FillOp{}, tag, value);
  }
  static VECOPS_ALWAYS_INLINE Word add(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(AddOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE Word sub(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(SubOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE Word mul(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(MulOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE Word fmadd(
      Tag tag, Word a, Word b, Word c) {
    return execute_word<Index, X86Backend>(FmaddOp{}, tag, a, b, c);
  }
  static VECOPS_ALWAYS_INLINE Word min(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(MinOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE Word max(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(MaxOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE MaskWord lt(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(CmpLtOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE MaskWord gt(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(CmpGtOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE MaskWord ne(Tag tag, Word a, Word b) {
    return execute_word<Index, X86Backend>(CmpNeOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE Word select(
      Tag tag, Word fallback, MaskWord mask, Word selected) {
    return execute_word<Index, X86Backend>(
        BlendOp{}, tag, fallback, mask, selected);
  }
  static VECOPS_ALWAYS_INLINE IWord iadd(ITag tag, IWord a, IWord b) {
    return execute_word<Index, X86Backend>(AddOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE IWord isub(ITag tag, IWord a, IWord b) {
    return execute_word<Index, X86Backend>(SubOp{}, tag, a, b);
  }
  static VECOPS_ALWAYS_INLINE IWord ishr(ITag tag, IWord a, int count) {
    return execute_word<Index, X86Backend>(
        BitShiftRightOp{}, tag, a, count);
  }
  static VECOPS_ALWAYS_INLINE IWord ishl(ITag tag, IWord a, int count) {
    return execute_word<Index, X86Backend>(
        BitShiftLeftOp{}, tag, a, count);
  }
  static VECOPS_ALWAYS_INLINE IWord ifill(ITag tag, IndexElement<T> value) {
    return execute_word<Index, X86Backend>(FillOp{}, tag, value);
  }
  static VECOPS_ALWAYS_INLINE Word from_bits(Tag tag, IWord value) {
    return execute_word<Index, X86Backend>(BitCastOp{}, tag, value);
  }
};

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_round_nearest(
    NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  using Raw = decltype(value.value);
  constexpr int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_castph_si128(_mm_roundscale_ph(_mm_castsi128_ph(value.value), mode))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_castph_si256(_mm256_roundscale_ph(
            _mm256_castsi256_ph(value.value), mode))};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_castph_si512(_mm512_roundscale_ph(
        _mm512_castsi512_ph(value.value), mode))};
#endif
#endif
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_round_ps(value.value, mode)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_round_ps(value.value, mode)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_roundscale_ps(value.value, mode)};
#endif
  } else {
    if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{
        _mm_round_pd(value.value, mode)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{
        _mm256_round_pd(value.value, mode)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_roundscale_pd(value.value, mode)};
#endif
  }
}

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<IndexTag<Tag>> x86_exp_to_index(
    Tag, NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  using Raw = decltype(value.value);
  using Result = NativeWordVec<IndexTag<Tag>>;
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) return Result{
        _mm_cvttps_epi32(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return Result{
        _mm256_cvttps_epi32(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Result{_mm512_cvttps_epi32(value.value)};
#endif
  } else {
#if defined(HAS_AVX512DQ)
    if constexpr (sizeof(Raw) == 16) return Result{
        _mm_cvttpd_epi64(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return Result{
        _mm256_cvttpd_epi64(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Result{_mm512_cvttpd_epi64(value.value)};
#endif
#else
    constexpr std::size_t lanes = sizeof(Raw) / sizeof(double);
    alignas(64) double input[lanes];
    alignas(64) int64_t output[lanes];
    std::memcpy(input, &value.value, sizeof(Raw));
    for (std::size_t lane = 0; lane < lanes; ++lane)
      output[lane] = static_cast<int64_t>(input[lane]);
    decltype(Result{}.value) raw;
    std::memcpy(&raw, output, sizeof(Raw));
    return Result{raw};
#endif
  }
}

template <ExpTier Tier, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_poly(
    Tag tag, NativeWordVec<Tag> r) {
  using T = ElementOf<Tag>;
  using C = X86ExpContext<Index, Tag>;
  const auto r2 = C::mul(tag, r, r);
  if constexpr (Tier == ExpTier::Estimate) {
    if constexpr (std::same_as<T, float16_t>) {
      auto p = C::fill(tag, T(0.1666259765625));
      p = C::fmadd(tag, p, r, C::fill(tag, T(0.5)));
      p = C::fmadd(tag, p, r, C::fill(tag, T(1)));
      return C::fmadd(tag, p, r, C::fill(tag, T(1)));
    }
    const auto c1 = C::fill(tag, T(1.0214147557098976));
    const auto c2 = C::fill(tag, T(0.5160640924984561));
    return C::fmadd(
        tag, c2, r2,
        C::fmadd(tag, c1, r, C::fill(tag, T(1))));
  } else if constexpr (std::same_as<T, float16_t>) {
    auto p = C::fill(tag, T(0.041656494140625));
    p = C::fmadd(tag, p, r, C::fill(tag, T(0.1666259765625)));
    p = C::fmadd(tag, p, r, C::fill(tag, T(0.5)));
    p = C::fmadd(tag, p, r, C::fill(tag, T(1)));
    return C::fmadd(tag, p, r, C::fill(tag, T(1)));
  } else if constexpr (Tier == ExpTier::Fast && std::same_as<T, float32_t>) {
    auto p = C::fill(tag, T(0.0083691484928131103515625));
    p = C::fmadd(tag, p, r, C::fill(tag, T(0.0419175066053867340087891)));
    p = C::fmadd(tag, p, r, C::fill(tag, T(0.166665047407150268554688)));
    p = C::fmadd(tag, p, r, C::fill(tag, T(0.499988704919815063476562)));
    p = C::fmadd(tag, p, r, C::fill(tag, T(1)));
    return C::fmadd(tag, p, r, C::fill(tag, T(1)));
  } else if constexpr (std::same_as<T, float32_t>) {
    const auto p0 = C::fmadd(
        tag, C::fill(tag, T(0.166666671633720397949219)), r,
        C::fill(tag, T(0.5)));
    const auto p1 = C::fmadd(
        tag, C::fill(tag, T(0.00833336077630519866943359)), r,
        C::fill(tag, T(0.0416664853692054748535156)));
    const auto p2 = C::fmadd(
        tag, C::fill(tag, T(0.000198527617612853646278381)), r,
        C::fill(tag, T(0.00139304355252534151077271)));
    const auto q = C::fmadd(
        tag, C::fmadd(tag, p2, r2, p1), r2, p0);
    return C::add(
        tag, C::fmadd(tag, q, r2, r), C::fill(tag, T(1)));
  } else {
    constexpr double coefficients[] = {
        0.5, 0.166666666666666851703837,
        0.0416666666666665047591422, 0.00833333333331652721664984,
        0.00138888888889774492207962, 0.000198412698960509205564975,
        2.4801587159235472998791e-5, 2.75572362911928827629423e-6,
        2.75573911234900471893338e-7, 2.51112930892876518610661e-8,
        2.08860621107283687536341e-9};
    const auto pair = [&](int low) {
      return C::fmadd(
          tag, C::fill(tag, T(coefficients[low + 1])), r,
          C::fill(tag, T(coefficients[low])));
    };
    const auto p0 = pair(0);
    const auto p1 = pair(2);
    const auto p2 = pair(4);
    const auto p3 = pair(6);
    const auto p4 = pair(8);
    const auto p5 = C::fill(tag, T(coefficients[10]));
    const auto r4 = C::mul(tag, r2, r2);
    const auto r8 = C::mul(tag, r4, r4);
    const auto q0 = C::fmadd(tag, p1, r2, p0);
    const auto q1 = C::fmadd(tag, p3, r2, p2);
    const auto q2 = C::fmadd(tag, p5, r2, p4);
    const auto q = C::fmadd(
        tag, q2, r8, C::fmadd(tag, q1, r4, q0));
    return C::add(
        tag, C::fmadd(tag, q, r2, r), C::fill(tag, T(1)));
  }
}

template <ExpTier Tier, bool NegativeOnly, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_standard(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using C = X86ExpContext<Index, Tag>;
  constexpr bool strict = Tier == ExpTier::Strict;
  constexpr bool estimate = Tier == ExpTier::Estimate;
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = strict;
#else
  constexpr bool gradual = false;
#endif
#if defined(CPU_CAPABILITY_AVX512) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  constexpr bool natural_underflow = true;
#else
  constexpr bool natural_underflow = false;
#endif
  const T overflow = std::same_as<T, float16_t> ? T(11.0859375)
      : std::same_as<T, float32_t>
          ? T(88.72283905206835) : T(709.782712893384);
  const T zero_limit = std::same_as<T, float16_t> ? T(-17.328679513999)
      : std::same_as<T, float32_t>
          ? T(-103.972084045410) : T(-745.1332191019411);
  const T normal_limit = std::same_as<T, float16_t> ? T(-9.704060527839)
      : std::same_as<T, float32_t>
          ? T(-87.3365447505531) : T(-708.3964185322641);
  const T zero_sentinel = std::same_as<T, float16_t> ? T(-18.0)
      : std::same_as<T, float32_t> ? T(-104.0) : T(-746.0);
  const T lower = natural_underflow
      ? zero_sentinel : (gradual ? zero_limit : normal_limit);

  auto xc = C::max(tag, x, C::fill(tag, lower));
  if constexpr (!NegativeOnly)
    xc = C::min(tag, xc, C::fill(tag, overflow));
  const auto qf = x86_exp_round_nearest<Tag>(C::mul(
      tag, xc, C::fill(tag, T(1.442695040888963407359924681))));
  auto r = C::fmadd(
      tag, qf, C::fill(tag, estimate
          ? T(-0.693147180559945309417232121458)
          : std::same_as<T, float16_t> ? T(-0.693359375)
              : std::same_as<T, float32_t> ? T(-0.693145751953125F)
              : T(-0.6931471805596629565116018)), xc);
  if constexpr (!estimate) {
    r = C::fmadd(
        tag, qf, C::fill(tag, std::same_as<T, float16_t>
            ? T(0.000212192535400390625)
            : std::same_as<T, float32_t> ? T(-1.428606765330187045e-6F)
            : T(-0.28235290563031577122588448175e-12)), r);
  }
  const auto poly = x86_exp_poly<Tier, Index>(tag, r);
  NativeWordVec<Tag> y;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_castph_si128(_mm_scalef_ph(
          _mm_castsi128_ph(poly.value), _mm_castsi128_ph(qf.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_castph_si256(_mm256_scalef_ph(
          _mm256_castsi256_ph(poly.value), _mm256_castsi256_ph(qf.value)))};
#endif
#if VEC_WIDTH >= 512
    else y = NativeWordVec<Tag>{_mm512_castph_si512(_mm512_scalef_ph(
        _mm512_castsi512_ph(poly.value), _mm512_castsi512_ph(qf.value)))};
#endif
#endif
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_scalef_ps(poly.value, qf.value)};
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_scalef_ps(poly.value, qf.value)};
    else y = NativeWordVec<Tag>{_mm512_scalef_ps(poly.value, qf.value)};
  } else {
    if constexpr (sizeof(Raw) == 16)
      y = NativeWordVec<Tag>{_mm_scalef_pd(poly.value, qf.value)};
    else if constexpr (sizeof(Raw) == 32)
      y = NativeWordVec<Tag>{_mm256_scalef_pd(poly.value, qf.value)};
    else y = NativeWordVec<Tag>{_mm512_scalef_pd(poly.value, qf.value)};
  }
#else
  constexpr typename C::ITag index_tag{};
  const auto qi = x86_exp_to_index<Tag>(tag, qf);
  const auto half = C::ishr(index_tag, qi, 1);
  const auto rest = C::isub(index_tag, qi, half);
  const auto bias = C::ifill(
      index_tag, std::same_as<T, float32_t> ? 127 : 1023);
  const int shift = std::same_as<T, float32_t> ? 23 : 52;
  if constexpr (NegativeOnly && !gradual) {
    const auto scale = C::from_bits(
        tag, C::ishl(index_tag, C::iadd(index_tag, qi, bias), shift));
    y = C::mul(tag, poly, scale);
  } else {
    const auto a = C::from_bits(
        tag, C::ishl(index_tag, C::iadd(index_tag, half, bias), shift));
    const auto b = C::from_bits(
        tag, C::ishl(index_tag, C::iadd(index_tag, rest, bias), shift));
    y = C::mul(tag, C::mul(tag, poly, a), b);
  }
#endif
  if constexpr (!NegativeOnly) {
    y = C::select(
        tag, y, C::gt(tag, x, C::fill(tag, overflow)),
        C::fill(tag, std::numeric_limits<T>::infinity()));
  }
  if constexpr (!natural_underflow) {
    y = C::select(
        tag, y,
        C::lt(tag, x, C::fill(tag, gradual ? zero_limit : normal_limit)),
        C::fill(tag, T(0)));
  }
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  if constexpr (!NegativeOnly) {
    y = C::select(
        tag, y, C::ne(tag, x, x), C::add(tag, x, x));
  }
#endif
  return y;
}

template <ExpTier Tier, bool NegativeOnly, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_exp_low_precision(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t,
      static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
  constexpr ExpTier compute_tier =
      std::same_as<T, bfloat16_t> && Tier == ExpTier::Estimate
          ? ExpTier::Fast : Tier;
  if constexpr (sizeof(Raw) == 16) {
    __m128 low;
    __m128 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256 low;
    __m256 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
#if VEC_WIDTH >= 512
  else {
    __m512 low;
    __m512 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_exp_standard<
        compute_tier, NegativeOnly, 0>(
            float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
}

/* **************************************************************************** */
//                    Exponential word implementation                        //
/* **************************************************************************** */

template <typename Op>
struct X86ExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    constexpr ExpTier tier = std::same_as<Op, ExpOp> ||
            std::same_as<Op, ExpNegOp>
        ? ExpTier::Strict
        : (std::same_as<Op, ExpFastOp> || std::same_as<Op, ExpNegFastOp>
            ? ExpTier::Fast : ExpTier::Estimate);
    constexpr bool negative_only =
        std::same_as<Op, ExpNegOp> ||
        std::same_as<Op, ExpNegFastOp> ||
        std::same_as<Op, ExpNegEstOp>;
    if constexpr (
        std::same_as<ElementOf<Tag>, float32_t> ||
        std::same_as<ElementOf<Tag>, float64_t>
#if defined(HAS_AVX512_FP16)
        || std::same_as<ElementOf<Tag>, float16_t>
#endif
        )
      return x86_exp_standard<tier, negative_only, Index>(tag, value);
    else
      return x86_exp_low_precision<tier, negative_only, Index>(tag, value);
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto zero = X86ExpContext<Index, Tag>::fill(tag, ElementOf<Tag>{});
    const auto safe = X86ExpContext<Index, Tag>::select(
        tag, zero, mask, value);
    return X86ExpContext<Index, Tag>::select(
        tag, inactive, mask, call<Index>(op, tag, safe));
  }
};

#define VECOPS_VEC_DEFINE_X86_EXP(OpType)                              \
  template <>                                                          \
  struct NativeWordImpl<X86Backend, OpType>                            \
      : X86ExpWordImpl<OpType> {}

VECOPS_VEC_DEFINE_X86_EXP(ExpOp);
VECOPS_VEC_DEFINE_X86_EXP(ExpFastOp);
VECOPS_VEC_DEFINE_X86_EXP(ExpEstOp);
VECOPS_VEC_DEFINE_X86_EXP(ExpNegOp);
VECOPS_VEC_DEFINE_X86_EXP(ExpNegFastOp);
VECOPS_VEC_DEFINE_X86_EXP(ExpNegEstOp);

#undef VECOPS_VEC_DEFINE_X86_EXP

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_H
