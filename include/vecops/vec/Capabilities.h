#ifndef VECOPS_VEC_CAPABILITIES_H
#define VECOPS_VEC_CAPABILITIES_H

#include "vecops/Features.h"

/**
 * @file Capabilities.h
 * @brief Compile-time SIMD capability detection and platform constants.
 *
 * The active CPU capability is auto-detected from compiler feature macros
 * (__AVX512F__, __AVX2__, __AVX__, __ARM_FEATURE_SVE, __ARM_NEON) in
 * descending priority order: AVX512 → AVX2 → AVX → SVE → NEON → GENERIC.
 * The detected capability selects:
 *
 * - The active backend (scalar, x86, or SVE) in details/Backend.h.
 * - VEC_WIDTH: the native SIMD register width in bits, or -1 for sizeless SVE.
 * - VEC_MAX_POW: the maximum base-2 exponent of native words a Tag may span.
 * - VEC_HW_MIN_POW: the minimum exponent describing a subword fraction.
 */

#if defined(CPU_CAPABILITY_GENERIC) && !defined(CPU_CAPABILITY)
#define CPU_CAPABILITY GENERIC
#endif

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#if defined(HAS_AVX512F) && defined(HAS_AVX512CD) && defined(HAS_AVX512BW) && defined(HAS_AVX512DQ)
#define HAS_CPU_CAPABILITY_AVX512 1
#endif
#if defined(HAS_AVX2)
#define HAS_CPU_CAPABILITY_AVX2 1
#endif
#if defined(HAS_AVX)
#define HAS_CPU_CAPABILITY_AVX 1
#endif
/** Maximum number of native words a Tag can span on x86 (2^5 = 32 words). */
#define VEC_MAX_POW (5)
/** Minimum subword fraction exponent on x86 (2^-2 = 1/4 word). */
#define VEC_HW_MIN_POW (-2)
#endif

#if defined(ARCH_ARM_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#if defined(HAS_SVE)
#define HAS_CPU_CAPABILITY_SVE 1
/** Maximum number of native words a Tag can span on SVE (2^2 = 4 words). */
#define VEC_MAX_POW (2)
/** SVE VLS mode does not support subword fractions. */
#define VEC_HW_MIN_POW (0)
#endif
#if defined(HAS_NEON)
#define HAS_CPU_CAPABILITY_NEON 1
/** Maximum number of native words on NEON (2^2 = 4 words for larger vectors). */
#define VEC_MAX_POW (2)
/** NEON minimum fraction (full word). */
#define VEC_HW_MIN_POW (0)
#endif
#endif

#if defined(CPU_CAPABILITY_GENERIC)
/** Scalar backend SIMD register width (128 bits, 16 bytes). */
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
/** Scalar backend supports up to 32 words for large synthetic vectors. */
#define VEC_MAX_POW (5)
/** Scalar backend minimum fraction: full word. */
#define VEC_HW_MIN_POW (0)
#endif

#if defined(CPU_CAPABILITY_AVX512) || (defined(HAS_CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX512)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX512
#define CPU_CAPABILITY_AVX512 1
/** AVX-512 native register width: 512 bits (64 bytes). */
#define VEC_WIDTH 512
#define MAX_VEC_WIDTH VEC_WIDTH
#endif

#if defined(CPU_CAPABILITY_AVX2) || (defined(HAS_CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX2)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX2
#define CPU_CAPABILITY_AVX2 1
/** AVX2 native register width: 256 bits (32 bytes). */
#define VEC_WIDTH 256
#define MAX_VEC_WIDTH VEC_WIDTH
#endif

#if defined(CPU_CAPABILITY_AVX) || (defined(HAS_CPU_CAPABILITY_AVX) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX
#define CPU_CAPABILITY_AVX 1
/** AVX/SSE native register width: 128 bits (16 bytes). */
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#endif

#if defined(CPU_CAPABILITY_SVE) || (defined(HAS_CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_SVE)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY SVE
#define CPU_CAPABILITY_SVE 1
/** SVE sizeless mode marker: width is unknown at compile time. */
#define VEC_WIDTH (-1)
/** Maximum practical SVE width: 2048 bits. */
#define MAX_VEC_WIDTH 2048
#endif

#if defined(CPU_CAPABILITY_NEON) || (defined(HAS_CPU_CAPABILITY_NEON) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_NEON)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY NEON
#define CPU_CAPABILITY_NEON 1
/** NEON native register width: 128 bits (16 bytes). */
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#endif

/** Fallback: no SIMD capability detected → scalar backend. */
#if !defined(CPU_CAPABILITY)
#define CPU_CAPABILITY GENERIC
#define CPU_CAPABILITY_GENERIC 1
/** Scalar fallback register width: 128 bits. */
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (5)
#undef VEC_HW_MIN_POW
/** Scalar fallback minimum fraction: full word. */
#define VEC_HW_MIN_POW (0)
#endif

#endif // VECOPS_VEC_CAPABILITIES_H
