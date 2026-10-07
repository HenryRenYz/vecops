// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// The project copyright above applies to project-owned portions only.
// SPDX-FileCopyrightText: 2019-2024 Arm Limited
// Arm-derived portions retain MIT OR Apache-2.0 WITH LLVM-exception;
// the conjunction below also requires MIT for project-owned portions.
// Reference-specific original notices and scope: THIRD_PARTY_NOTICES.md.
// Complete upstream terms: LICENSES/Arm-optimized-routines.txt.
// SPDX-License-Identifier: MIT AND (MIT OR Apache-2.0 WITH LLVM-exception)
#ifndef VECOPS_VEC_DETAILS_X86_MATH_LOG_H
#define VECOPS_VEC_DETAILS_X86_MATH_LOG_H

/**
 * @file Log.h
 * @brief x86 backend implementations for the log family (log, log2, log10).
 *
 * The kernels mirror the SVE backend on the shared constants in
 * vec/details/Math.h: a branch-free inline core covers the normal domain
 * and one NOINLINE special routine owns every tail (subnormals, zero,
 * negatives, +inf, NaN). The f64 Strict kernel, the special-case structure,
 * and all minimax coefficients and 128-entry invc/logc tables are derived
 * from the Arm optimized-routines project
 * (https://github.com/ARM-software/optimized-routines), files
 * math/aarch64/sve/{log,log2,log10}.c and
 * math/aarch64/v_log{,2,10}_data.c (dual-licensed under "MIT OR Apache-2.0
 * WITH LLVM-exception"); the same reduction also underlies Intel SVML and
 * glibc's libmvec. The table-free Fast/Estimate polynomials follow the SVE
 * header's shapes.
 *
 * Reduction: x = 2^k * z with z inside a fixed window (table kernel:
 * [0x3fe6900900000000, doubled) via invc/logc gathers; polynomial kernels:
 * [2/3, 4/3) via a float subtraction r = z - 1 that is exact by Sterbenz).
 * The gather index is computed as ((ix - OFF) >> 44) & 0xfe, which stays
 * inside the interleaved double-element table for every input bit pattern,
 * so garbage or inactive lanes can never form wild addresses.
 *
 * Tiers: f64 Strict is the upstream table algorithm (measured 2.64/2.58/
 * 2.46 ULP for log/log2/log10, within the documented 4-ULP contract);
 * AVX2/AVX-512 words gather invc/logc with hardware gathers while 128-bit
 * words load their two entries through memory (one 16-byte load per lane).
 * f32 Strict is the native degree-13 minimax kernel of the shared
 * LogF32StrictFamily (same skeleton and coefficients as the SVE backend;
 * coefficients and the bit-exact simulation live in scripts/
 * log-native-design/): the symmetric [1/sqrt2, sqrt2) window, Estrin
 * FMAs, and the Cody-Waite assembly whose kf*hi product stays exact
 * inside the final FMA. Fast keeps six Taylor terms (relative error
 * ~2^-13.9), Estimate three (~2^-10.3); non-e bases scale the natural-log
 * result once at the end, which preserves the relative-error tier bound.
 * Every base runs the identical instruction sequence, so their throughputs
 * match.
 *
 * Special routines mirror the SVE structure: f64 renormalizes subnormal
 * lanes by 2^52 and adds the logarithmic correction, while f32 folds the
 * correction into the extracted exponent as an integer bias on the
 * renormalized lanes (the cores take the bias through a masked integer
 * subtract), so subnormal lanes need no additive term. f16 and bf16 widen
 * through the f32 pipeline: every f16/bf16 input is normal in f32 and the
 * f32 special routine owns all narrow-format tails exactly.
 */

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Math.h"
#include "vecops/vec/details/x86/Arithmetic.h"
#include "vecops/vec/details/x86/Basic.h"
#include "vecops/vec/details/x86/Types.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    index conversion and table gather helpers                                 //
/* **************************************************************************** */

/**
 * Converts an index-typed word to its floating twin. i32 -> f32 uses the
 * plain convert everywhere; i64 -> f64 uses the AVX-512DQ convert and falls
 * back to the exponent-bias trick on targets without it: bias k into the
 * mantissa of 2^52 + 2^51 + k (the half-mantissa offset keeps every
 * reachable log exponent inside one binade, so negative k cannot borrow
 * into the exponent field), then subtract 2^52 + 2^51.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_index_to_float(
    Tag tag, NativeWordVec<IndexTag<Tag>> value) {
  constexpr IndexTag<Tag> itag{};
  if constexpr (std::same_as<ElementOf<Tag>, float32_t>) {
    if constexpr (sizeof(value.value) == 16)
      return NativeWordVec<Tag>{_mm_cvtepi32_ps(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(value.value) == 32)
      return NativeWordVec<Tag>{_mm256_cvtepi32_ps(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_cvtepi32_ps(value.value)};
#endif
  } else {
#if defined(HAS_AVX512DQ)
    if constexpr (sizeof(value.value) == 16)
      return NativeWordVec<Tag>{_mm_cvtepi64_pd(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(value.value) == 32)
      return NativeWordVec<Tag>{_mm256_cvtepi64_pd(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return NativeWordVec<Tag>{_mm512_cvtepi64_pd(value.value)};
#endif
#else
    const auto biased = add(
        itag, value,
        fill_word(itag, std::int64_t{0x4338000000000000}));
    return sub(
        tag, bitcast(tag, biased),
        fill_word(tag, float64_t(6755399441055744.0)));
#endif
  }
}

/**
 * Gathers the invc/logc pair for every lane of one f64 word from the
 * interleaved 128-entry table. The index is a double-element index (always
 * even and below 256), so hardware gathers address both fields with the
 * same index register; 128-bit words load each entry with one unaligned
 * 16-byte load and unpack the halves.
 */
template <const LogF64Family& C, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_log_gather_f64(
    Tag, NativeWordVec<IndexTag<Tag>> index, NativeWordVec<Tag>& invc,
    NativeWordVec<Tag>& logc) {
  using Raw = decltype(index.value);
  if constexpr (sizeof(Raw) == 64) {
    const auto i32 = _mm512_cvtepi64_epi32(index.value);
    invc = NativeWordVec<Tag>{_mm512_i32gather_pd(i32, &C.table[0].invc, 8)};
    logc = NativeWordVec<Tag>{_mm512_i32gather_pd(i32, &C.table[0].logc, 8)};
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    invc = NativeWordVec<Tag>{
        _mm256_i64gather_pd(&C.table[0].invc, index.value, 8)};
    logc = NativeWordVec<Tag>{
        _mm256_i64gather_pd(&C.table[0].logc, index.value, 8)};
  }
#endif
  else {
    alignas(16) std::int64_t indices[2];
    std::memcpy(indices, &index.value, sizeof(indices));
    const auto entry0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
        &C.table[static_cast<std::size_t>(indices[0]) >> 1]));
    const auto entry1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
        &C.table[static_cast<std::size_t>(indices[1]) >> 1]));
    invc = NativeWordVec<Tag>{
        _mm_castsi128_pd(_mm_unpacklo_epi64(entry0, entry1))};
    logc = NativeWordVec<Tag>{
        _mm_castsi128_pd(_mm_unpackhi_epi64(entry0, entry1))};
  }
}

/* **************************************************************************** */
//    f64 kernels                                                               //
/* **************************************************************************** */

/**
 * f64 table kernel: log(x) = (logc + r*lead + k*kd_scale) +
 * r^2*(c0 + r*c1 + r^2*(c2 + r*c3 + r^2*c4)), r = invc*z - 1.
 *
 * The bit-twiddling runs on the signed index word: the k extraction needs
 * the arithmetic shift, and the cell index keeps only bits 44..51 of the
 * same shifted value, where the signed and unsigned shifts agree.
 */
template <const LogF64Family& C, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f64_table_core(
    Tag tag, NativeWordVec<Tag> x) {
  constexpr IndexTag<Tag> itag{};
  constexpr std::int64_t kOff = std::bit_cast<std::int64_t>(
      std::uint64_t{0x3fe6900900000000});
  constexpr std::int64_t kMantissaLow = 0xFFFFFFFFFFFFF;
  constexpr std::int64_t kExponentBits = ~kMantissaLow;
  const auto bits = bitcast(itag, x);
  const auto tmp = sub(itag, bits, fill_word(itag, kOff));
  const auto k = shr(itag, tmp, 52);
  const auto i = bit_and(
      itag, shr(itag, tmp, 44), fill_word(itag, std::int64_t{0xfe}));
  const auto z = bitcast(
      tag,
      sub(itag, bits, bit_and(itag, tmp, fill_word(itag, kExponentBits))));
  NativeWordVec<Tag> invc{};
  NativeWordVec<Tag> logc{};
  x86_log_gather_f64<C>(tag, i, invc, logc);
  const auto kd = x86_log_index_to_float(tag, k);
  const auto r = fmsub(tag, invc, z, fill_word(tag, float64_t(1)));
  const auto w = fmadd(tag, r, fill_word(tag, float64_t(C.lead)), logc);
  const auto hi = fmadd(tag, kd, fill_word(tag, float64_t(C.kd_scale)), w);
  const auto r2 = mul(tag, r, r);
  auto y = fmadd(
      tag, r, fill_word(tag, float64_t(C.c3)), fill_word(tag, float64_t(C.c2)));
  y = fmadd(tag, r2, fill_word(tag, float64_t(C.c4)), y);
  const auto p = fmadd(
      tag, r, fill_word(tag, float64_t(C.c1)), fill_word(tag, float64_t(C.c0)));
  y = fmadd(tag, r2, y, p);
  return fmadd(tag, r2, y, hi);
}

/**
 * f64 table-free kernel for the Fast and Estimate tiers: r = z - 1 on the
 * [2/3, 4/3) window (exact by Sterbenz), six Taylor terms for Fast and
 * three for Estimate, and one final base scale for non-e bases.
 */
template <const LogF64PolyFamily& C, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f64_poly_core(
    Tag tag, NativeWordVec<Tag> x) {
  constexpr IndexTag<Tag> itag{};
  constexpr std::int64_t kOff = std::bit_cast<std::int64_t>(
      std::uint64_t{0x3fe5555555555555});
  constexpr std::int64_t kMantissaLow = 0xFFFFFFFFFFFFF;
  constexpr std::int64_t kExponentBits = ~kMantissaLow;
  const auto bits = bitcast(itag, x);
  const auto tmp = sub(itag, bits, fill_word(itag, kOff));
  const auto n = x86_log_index_to_float(tag, shr(itag, tmp, 52));
  const auto z = bitcast(
      tag,
      sub(itag, bits, bit_and(itag, tmp, fill_word(itag, kExponentBits))));
  const auto r = sub(tag, z, fill_word(tag, float64_t(1)));
  const auto r2 = mul(tag, r, r);
  NativeWordVec<Tag> y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = fmadd(
        tag, r, fill_word(tag, float64_t(C.c1)), fill_word(tag, float64_t(C.c0)));
    const auto b = fmadd(
        tag, r, fill_word(tag, float64_t(C.c3)), fill_word(tag, float64_t(C.c2)));
    const auto d = fmadd(
        tag, r, fill_word(tag, float64_t(C.c5)), fill_word(tag, float64_t(C.c4)));
    y = fmadd(tag, r2, d, b);
    y = fmadd(tag, r2, y, a);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = fmadd(
        tag, r, fill_word(tag, float64_t(C.c2)), fill_word(tag, float64_t(C.c1)));
    y = fmadd(tag, r, y, fill_word(tag, float64_t(C.c0)));
  }
  const auto hi = fmadd(tag, n, fill_word(tag, float64_t(C.ln2)), r);
  auto result = fmadd(tag, r2, y, hi);
  if constexpr (C.scale != 1.0)
    result = mul(tag, result, fill_word(tag, float64_t(C.scale)));
  return result;
}

/**
 * Marks lanes outside the plain kernel domain: x below the smallest normal
 * (zero, subnormals, all negatives), x above the largest finite (+inf), and
 * NaN (which fails both ordered tests and is caught by the self-inequality).
 * Float-domain compares avoid emulating the unsigned bit compare on
 * pre-AVX-512 targets.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordMask<Tag> x86_log_special_mask(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  const auto tiny = cmplt(tag, x, fill_word(tag, std::numeric_limits<T>::min()));
  const auto huge = cmpgt(tag, x, fill_word(tag, std::numeric_limits<T>::max()));
  const auto odd = cmpne(tag, x, x);
  return x86_mask_word_or<Tag>(
      tiny, x86_mask_word_or<Tag>(huge, odd));
}

/** Tails: subnormals renormalized by 2^52, constants for 0/inf/NaN. */
template <const LogF64Family& C, FloatingTag Tag>
VECOPS_NOINLINE NativeWordVec<Tag> x86_log_f64_special(
    Tag tag, NativeWordVec<Tag> x, NativeWordMask<Tag> special) {
  using T = ElementOf<Tag>;
  const auto inf = std::numeric_limits<T>::infinity();
  const auto is_sub = cmpgt(tag, x, fill_word(tag, T(0)));
  const auto is_zero = cmpeq(tag, x, fill_word(tag, T(0)));
  const auto is_inf = cmpeq(tag, x, fill_word(tag, inf));
  // Subnormal lanes (and only they among the finite special lanes) need the
  // 2^52 renormalization; every other special lane discards the scaled
  // result through the final blend, so one masked multiply covers all.
  const auto scaled = blend(
      tag, x, special, mul(tag, x, fill_word(tag, T(0x1p52))));
  auto y = x86_log_f64_table_core<C>(tag, scaled);
  auto corr = blend(
      tag, fill_word(tag, std::numeric_limits<T>::quiet_NaN()), is_sub,
      fill_word(tag, T(C.sub)));
  corr = blend(
      tag, corr, is_zero, fill_word(tag, T(-inf)));
  corr = blend(tag, corr, is_inf, fill_word(tag, T(inf)));
  return blend(tag, y, special, add(tag, y, corr));
}

template <const LogF64Family& C, const LogF64PolyFamily& C2, Accuracy Tier,
          FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f64(
    Tag tag, NativeWordVec<Tag> x) {
  const auto special = x86_log_special_mask(tag, x);
  if (x86_mask_word_any<Tag>(special))
    return x86_log_f64_special<C>(tag, x, special);
  if constexpr (Tier == Accuracy::Strict)
    return x86_log_f64_table_core<C>(tag, x);
  else
    return x86_log_f64_poly_core<C2, Tier>(tag, x);
}

/* **************************************************************************** */
//    f32 kernels                                                               //
/* **************************************************************************** */

/**
 * f32 table-free kernel on the [2/3, 4/3) window; r = z - 1 is exact by
 * Sterbenz. Fast keeps six Taylor terms, Estimate three. KBias subtracts
 * from the extracted exponent on the bias lanes (the special path
 * renormalizes subnormals by 2^23); pass 0 on the normal path.
 */
template <const LogF32FastFamily& C, Accuracy Tier, int KBias,
          FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f32_poly_core(
    Tag tag, NativeWordVec<Tag> x, NativeWordMask<Tag> bias_lanes) {
  constexpr IndexTag<Tag> itag{};
  constexpr std::int32_t kOff = std::int32_t{0x3f2aaaab};
  const auto bits = bitcast(itag, x);
  const auto u_off = sub(itag, bits, fill_word(itag, kOff));
  const auto k_int = [&] {
    const auto shifted = shr(itag, u_off, 23);
    if constexpr (KBias == 0)
      return shifted;
    else
      return sub(
          itag, shifted,
          blend(itag, fill_word(itag, std::int32_t{}), bias_lanes,
                fill_word(itag, std::int32_t(KBias))));
  }();
  const auto n = x86_log_index_to_float(tag, k_int);
  const auto z = bitcast(
      tag,
      add(itag, bit_and(itag, u_off, fill_word(itag, std::int32_t{0x007fffff})),
          fill_word(itag, kOff)));
  const auto r = sub(tag, z, fill_word(tag, float32_t(1)));
  const auto r2 = mul(tag, r, r);
  NativeWordVec<Tag> y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = fmadd(
        tag, r, fill_word(tag, float32_t(C.c1)), fill_word(tag, float32_t(C.c0)));
    const auto b = fmadd(
        tag, r, fill_word(tag, float32_t(C.c3)), fill_word(tag, float32_t(C.c2)));
    const auto d = fmadd(
        tag, r, fill_word(tag, float32_t(C.c5)), fill_word(tag, float32_t(C.c4)));
    y = fmadd(tag, r2, d, b);
    y = fmadd(tag, r2, y, a);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = fmadd(
        tag, r, fill_word(tag, float32_t(C.c2)), fill_word(tag, float32_t(C.c1)));
    y = fmadd(tag, r, y, fill_word(tag, float32_t(C.c0)));
  }
  const auto hi = fmadd(tag, n, fill_word(tag, float32_t(C.ln2)), r);
  auto result = fmadd(tag, r2, y, hi);
  if constexpr (C.scale != 1.0f)
    result = mul(tag, result, fill_word(tag, float32_t(C.scale)));
  return result;
}

/**
 * Native f32 Strict kernel on the [1/sqrt2, sqrt2) window: r = z - 1 is
 * exact by Sterbenz, Q is the Estrin evaluation of the degree-13 minimax
 * in C.d, and the tail keeps every rounding at or below the result binade
 * (kflo = kf*lo and pf = kflo + r^2*Q round in tiny binades, t = alpha*r
 * + pf rounds at |t| <= |log_b(z)| <= |y| on every k band, and the final
 * FMA rounds once at the result scale with kf*hi exact). KBias serves the
 * special path exactly as in x86_log_f32_poly_core.
 */
template <const LogF32StrictFamily& C, int KBias, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f32_strict_core(
    Tag tag, NativeWordVec<Tag> x, NativeWordMask<Tag> bias_lanes) {
  constexpr IndexTag<Tag> itag{};
  constexpr std::int32_t kOff = std::int32_t{0x3f3504f3}; // fl32(1/sqrt2)
  const auto bits = bitcast(itag, x);
  const auto u_off = sub(itag, bits, fill_word(itag, kOff));
  const auto k_int = [&] {
    const auto shifted = shr(itag, u_off, 23);
    if constexpr (KBias == 0)
      return shifted;
    else
      return sub(
          itag, shifted,
          blend(itag, fill_word(itag, std::int32_t{}), bias_lanes,
                fill_word(itag, std::int32_t(KBias))));
  }();
  const auto kf = x86_log_index_to_float(tag, k_int);
  const auto z = bitcast(
      tag,
      add(itag, bit_and(itag, u_off, fill_word(itag, std::int32_t{0x007fffff})),
          fill_word(itag, kOff)));
  const auto r = sub(tag, z, fill_word(tag, float32_t(1)));
  const auto r2 = mul(tag, r, r);
  const auto qa = fmadd(tag, r, fill_word(tag, float32_t(C.d[1])), fill_word(tag, float32_t(C.d[0])));
  const auto qb = fmadd(tag, r, fill_word(tag, float32_t(C.d[3])), fill_word(tag, float32_t(C.d[2])));
  const auto qc = fmadd(tag, r, fill_word(tag, float32_t(C.d[5])), fill_word(tag, float32_t(C.d[4])));
  const auto qd = fmadd(tag, r, fill_word(tag, float32_t(C.d[7])), fill_word(tag, float32_t(C.d[6])));
  const auto qe = fmadd(tag, r, fill_word(tag, float32_t(C.d[9])), fill_word(tag, float32_t(C.d[8])));
  const auto qf = fmadd(tag, r, fill_word(tag, float32_t(C.d[11])), fill_word(tag, float32_t(C.d[10])));
  const auto qg = fmadd(tag, r, fill_word(tag, float32_t(C.d[13])), fill_word(tag, float32_t(C.d[12])));
  const auto qab = fmadd(tag, r2, qb, qa);
  const auto qcd = fmadd(tag, r2, qd, qc);
  const auto qef = fmadd(tag, r2, qf, qe);
  const auto r4 = mul(tag, r2, r2);
  const auto qabcd = fmadd(tag, r4, qcd, qab);
  const auto qefg = fmadd(tag, r4, qg, qef);
  const auto r8 = mul(tag, r4, r4);
  const auto q = fmadd(tag, r8, qefg, qabcd);
  const auto kflo = mul(tag, kf, fill_word(tag, float32_t(C.lo)));
  const auto pf = fmadd(tag, r2, q, kflo);
  const auto t = fmadd(tag, r, fill_word(tag, float32_t(C.alpha)), pf);
  return fmadd(tag, kf, fill_word(tag, float32_t(C.hi)), t);
}

/**
 * f32 special routine: renormalizes subnormal lanes by 2^23, reruns the
 * exact-tier core with the -23 exponent bias folded into the extracted
 * exponent (so subnormal lanes need no additive correction and their
 * select value is +0), and selects the IEEE constants for zero/inf/NaN
 * lanes through the final blend.
 */
template <LogBase Base, Accuracy Tier, FloatingTag Tag>
VECOPS_NOINLINE NativeWordVec<Tag> x86_log_f32_special(
    Tag tag, NativeWordVec<Tag> x, NativeWordMask<Tag> special) {
  using T = ElementOf<Tag>;
  const auto inf = std::numeric_limits<T>::infinity();
  const auto is_sub = cmpgt(tag, x, fill_word(tag, T(0)));
  const auto is_zero = cmpeq(tag, x, fill_word(tag, T(0)));
  const auto is_inf = cmpeq(tag, x, fill_word(tag, inf));
  const auto scaled = blend(
      tag, x, special, mul(tag, x, fill_word(tag, T(0x1p23))));
  const auto y = [&] {
    if constexpr (Tier == Accuracy::Strict)
      return x86_log_f32_strict_core<log_f32_strict_family<Base>(), 23>(
          tag, scaled, special);
    else
      return x86_log_f32_poly_core<log_f32_fast_family<Base>(), Tier, 23>(
          tag, scaled, special);
  }();
  auto corr = blend(
      tag, fill_word(tag, std::numeric_limits<T>::quiet_NaN()), is_sub,
      fill_word(tag, T(0)));
  corr = blend(
      tag, corr, is_zero, fill_word(tag, T(-inf)));
  corr = blend(tag, corr, is_inf, fill_word(tag, T(inf)));
  return blend(tag, y, special, add(tag, y, corr));
}

template <LogBase Base, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_f32(
    Tag tag, NativeWordVec<Tag> x) {
  const auto special = x86_log_special_mask(tag, x);
  if (x86_mask_word_any<Tag>(special))
    return x86_log_f32_special<Base, Tier>(tag, x, special);
  if constexpr (Tier == Accuracy::Strict)
    return x86_log_f32_strict_core<log_f32_strict_family<Base>(), 0>(
        tag, x, special);
  else
    return x86_log_f32_poly_core<log_f32_fast_family<Base>(), Tier, 0>(
        tag, x, special);
}

/* **************************************************************************** */
//    f16 / bf16 widening route                                                 //
/* **************************************************************************** */

/**
 * Widens a narrow word to two f32 words, evaluates the f32 pipeline on
 * both halves, and narrows back. Every f16/bf16 value widens exactly, and
 * the f32 pipeline's own tails own every narrow-format special input.
 */
template <LogBase Base, Accuracy Tier, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_log_low_precision(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t, static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
  if constexpr (sizeof(Raw) == 16) {
    __m128 low;
    __m128 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_log_f32<Base, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_log_f32<Base, Tier>(
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
    const auto result_low = x86_log_f32<Base, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_log_f32<Base, Tier>(
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
    const auto result_low = x86_log_f32<Base, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_log_f32<Base, Tier>(
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
//    word implementation                                                       //
/* **************************************************************************** */

/**
 * Shared word implementation for the log family. Wide element formats are
 * elementwise, so an explicit mask only sanitizes inactive lanes and
 * blends the caller's inactive value back in. The sanitize value is 1,
 * whose logarithm is exactly +0: masked calls keep taking the fast inline
 * path instead of the special routine (zero would mark every masked call
 * special). Narrow formats widen through whole-word f32 conversions after
 * the same sanitization, so the widening cannot observe stale lanes.
 */
template <LogBase Base, Accuracy A>
struct X86LogWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A>, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    if constexpr (std::same_as<T, float64_t>)
      return x86_log_f64<log_f64_family<Base>(), log_f64_poly_family<Base>(),
                         A>(tag, value);
    else if constexpr (std::same_as<T, float32_t>)
      return x86_log_f32<Base, A>(tag, value);
    else
      return x86_log_low_precision<Base, A, Index>(tag, value);
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto safe = blend(
        tag, fill_word(tag, ElementOf<Tag>(1)), mask, value);
    return blend(tag, inactive, mask, call<Index>(op, tag, safe));
  }
};

template <LogBase Base, Accuracy A>
struct NativeWordImpl<X86Backend, LogOp<Base, A>> : X86LogWordImpl<Base, A> {
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_LOG_H
