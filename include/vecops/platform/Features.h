//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_PLATFORM_FEATURES_H
#define VECOPS_PLATFORM_FEATURES_H

/**
 * @file Features.h
 * @brief Compile-time ISA and CPU-feature detection macros.
 *
 * This header maps the predefined macros of the target compiler (GCC/Clang
 * target-feature macros, ACLE macros on ARM, _M_* macros on MSVC) onto a
 * uniform HAS_* / ARCH_* vocabulary.  Everything here reflects the
 * translation-time target only; there is deliberately no runtime CPUID
 * probing.  Runtime dispatch is achieved by compiling one translation unit
 * per capability instead.  The rest of the platform layer builds on this
 * file:
 *
 *  - platform/Capabilities.h derives the CPU_CAPABILITY selection from these
 *    macros;
 *  - CMake (VecopsMultiarchTargets.cmake) propagates target guarantees that
 *    compilers have no feature macro for (VECOPS_TARGET_FIXED_STREAMING_SVE_BITS,
 *    VECOPS_TARGET_SME_FA64);
 *  - platform/Target.h maps the architecture family to a C++ target tag.
 *
 * The HAS_* / ARCH_* names are intentionally not prefixed with VECOPS_; they
 * are the historical project-wide vocabulary.  Keep new entries consistent
 * with the existing spelling style.
 */

/**
 * Detect compiler
 */
#if defined(_MSC_VER)
  #define COMPILER_MSVC 1
#elif defined(__clang__)
  #define COMPILER_CLANG 1
#elif defined(__GNUC__)
  #define COMPILER_GCC 1
#else
  #error "Unsupported compiler"
#endif

/**
 * Detect CPU architecture
 */
// x86_64 / AMD64
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
  #define ARCH_X86_64 1
  #define ARCH_X86_FAMILY 1
// x86 (32-bit)
#elif defined(_M_IX86) || defined(__i386__) || defined(__i686__)
  #define ARCH_X86 1
  #define ARCH_X86_FAMILY 1
// ARM64 / AArch64
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__)
  #define ARCH_ARM64 1
  #define ARCH_ARM_FAMILY 1
// ARM (32-bit)
#elif defined(_M_ARM) || defined(__arm__) || defined(__ARM_ARCH)
  #define ARCH_ARM 1
  #define ARCH_ARM_FAMILY 1
#endif

// ============================================================================
//                    x86/x86_64 ISA detection
// ============================================================================

#if defined(ARCH_X86_FAMILY)

// ==================== SSE family ====================

// SSE (x86-64 always has it; on 32-bit x86 MSVC reports it via _M_IX86_FP)
#if defined(COMPILER_MSVC)
  #if defined(_M_IX86_FP) && _M_IX86_FP >= 1
    #define HAS_SSE 1
  #elif defined(ARCH_X86_64)
    #define HAS_SSE 1
  #endif
#elif defined(__SSE__)
  #define HAS_SSE 1
#endif

// SSE2
#if defined(COMPILER_MSVC)
  #if defined(_M_IX86_FP) && _M_IX86_FP >= 2
    #define HAS_SSE2 1
  #elif defined(ARCH_X86_64)
    #define HAS_SSE2 1
  #endif
#elif defined(__SSE2__)
  #define HAS_SSE2 1
#endif

// SSE3 (and beyond: MSVC exposes no command-line switch for these on x86,
// so they are GCC/Clang-only)
#if defined(__SSE3__)
  #define HAS_SSE3 1
#endif

// SSSE3
#if defined(__SSSE3__)
  #define HAS_SSSE3 1
#endif

// SSE4.1
#if defined(__SSE4_1__)
  #define HAS_SSE4_1 1
#endif

// SSE4.2
#if defined(__SSE4_2__)
  #define HAS_SSE4_2 1
#endif

// F16C (half-precision <-> single-precision conversion)
#if defined(__F16C__)
  #define HAS_F16C 1
#endif

// ==================== AVX family ====================

// AVX
#if defined(__AVX__)
  #define HAS_AVX 1
#endif

// AVX2
#if defined(__AVX2__)
  #define HAS_AVX2 1
#endif

// FMA (fused multiply-add)
#if defined(__FMA__)
  #define HAS_FMA 1
#endif

// FMA4 (AMD legacy)
#if defined(__FMA4__)
  #define HAS_FMA4 1
#endif

// XOP (AMD legacy)
#if defined(__XOP__)
  #define HAS_XOP 1
#endif

// AVX-VNNI (128/256-bit VEX-encoding VNNI forms; Alder Lake, Zen 3)
#if defined(__AVXVNNI__)
  #define HAS_AVX_VNNI 1
#endif

// AVX-VNNI-INT8 (VEX 128/256-bit int8 dot products; Granite Rapids, Zen 5;
// GCC 14+/clang 20+ emit the macro)
#if defined(__AVXVNNIINT8__)
  #define HAS_AVX_VNNI_INT8 1
#endif

// AVX-VNNI-INT16 (VEX 128/256-bit bf16 dot products; Granite Rapids, Zen 5)
#if defined(__AVXVNNIINT16__)
  #define HAS_AVX_VNNI_INT16 1
#endif

// ==================== AVX-512 family ====================

// AVX512F (foundation, mandatory for everything below)
#if defined(__AVX512F__)
  #define HAS_AVX512F 1
#endif

// AVX512BW (byte/word operations)
#if defined(__AVX512BW__)
  #define HAS_AVX512BW 1
#endif

// AVX512CD (conflict detection)
#if defined(__AVX512CD__)
  #define HAS_AVX512CD 1
#endif

// AVX512DQ (doubleword/quadword)
#if defined(__AVX512DQ__)
  #define HAS_AVX512DQ 1
#endif

// AVX512VL (vector length: 128/256-bit forms of AVX-512)
#if defined(__AVX512VL__)
  #define HAS_AVX512VL 1
#endif

// AVX512IFMA (integer fused multiply add)
#if defined(__AVX512IFMA__)
  #define HAS_AVX512IFMA 1
#endif

// AVX512VBMI (vector byte manipulation)
#if defined(__AVX512VBMI__)
  #define HAS_AVX512VBMI 1
#endif

// AVX512VBMI2
#if defined(__AVX512VBMI2__)
  #define HAS_AVX512VBMI2 1
#endif

// AVX512VNNI (vector neural network instructions)
#if defined(__AVX512VNNI__)
  #define HAS_AVX512VNNI 1
#endif

// AVX512BF16 (bfloat16)
#if defined(__AVX512BF16__)
  #define HAS_AVX512_BF16 1
#endif

// AVX512FP16 (fp16 arithmetic)
#if defined(__AVX512FP16__)
  #define HAS_AVX512_FP16 1
#endif

// AVX512VPOPCNTDQ (population count)
#if defined(__AVX512VPOPCNTDQ__)
  #define HAS_AVX512VPOPCNTDQ 1
#endif

// AVX512BITALG (bit algorithms)
#if defined(__AVX512BITALG__)
  #define HAS_AVX512BITALG 1
#endif

// AVX512_4FMAPS (fused multiply-add permutation single-precision, Xeon Phi)
#if defined(__AVX5124FMAPS__)
  #define HAS_AVX512_4FMAPS 1
#endif

// AVX512_4VNNIW (vector neural network instructions word variable, Xeon Phi)
#if defined(__AVX5124VNNIW__)
  #define HAS_AVX512_4VNNIW 1
#endif

// AVX512VP2INTERSECT (Tiger Lake, reintroduced on Granite Rapids)
#if defined(__AVX512VP2INTERSECT__)
  #define HAS_AVX512_VP2INTERSECT 1
#endif

// ==================== AMX family ====================

// AMX TILE (tile architecture state and load/store)
#if defined(__AMX_TILE__)
  #define HAS_AMX_TILE 1
#endif

// AMX INT8 (int8 tile dot products)
#if defined(__AMX_INT8__)
  #define HAS_AMX_INT8 1
#endif

// AMX BF16 (bf16 tile dot products)
#if defined(__AMX_BF16__)
  #define HAS_AMX_BF16 1
#endif

// AMX FP16 (fp16 tile dot products; clang spells the macro __AMX_FP16__,
// GCC __AMXFP16__)
#if defined(__AMX_FP16__) || defined(__AMXFP16__)
  #define HAS_AMX_FP16 1
#endif

// AMX COMPLEX (complex-number tile dot products)
#if defined(__AMX_COMPLEX__)
  #define HAS_AMX_COMPLEX 1
#endif

// AMX TF32 (tf32 tile dot products; Granite Rapids era, GCC 15+/clang 20+)
#if defined(__AMX_TF32__)
  #define HAS_AMX_TF32 1
#endif

// AMX FP8 (fp8 tile dot products; Granite Rapids era)
#if defined(__AMX_FP8__)
  #define HAS_AMX_FP8 1
#endif

// AMX TRANSPOSE (tile transpose operations)
#if defined(__AMX_TRANSPOSE__)
  #define HAS_AMX_TRANSPOSE 1
#endif

// AMX AVX512 (tile <-> vector loads/stores)
#if defined(__AMX_AVX512__)
  #define HAS_AMX_AVX512 1
#endif

// ==================== AVX10 ====================

// GCC reports the AVX10 version through __AVX10_VER__; clang instead emits
// the spellings __AVX10_<ver>__ / __AVX10_<ver>_<bits>__.  Accept both.
#if defined(__AVX10_VER__)
  #define HAS_AVX10 1
  #define AVX10_VERSION __AVX10_VER__
#elif defined(__AVX10_2__) || defined(__AVX10_2_256__) || defined(__AVX10_2_512__)
  #define HAS_AVX10 1
  #define AVX10_VERSION 2
#elif defined(__AVX10_1__) || defined(__AVX10_1_256__) || defined(__AVX10_1_512__)
  #define HAS_AVX10 1
  #define AVX10_VERSION 1
#endif

// ==================== Other ISA extensions ====================

// POPCNT
#if defined(__POPCNT__)
  #define HAS_POPCNT 1
#endif

// BMI (bit manipulation instructions; includes TZCNT — there is no
// separate __TZCNT__ compiler macro)
#if defined(__BMI__)
  #define HAS_BMI 1
#endif

// BMI2
#if defined(__BMI2__)
  #define HAS_BMI2 1
#endif

// LZCNT
#if defined(__LZCNT__)
  #define HAS_LZCNT 1
#endif

// AES-NI
#if defined(__AES__)
  #define HAS_AES_NI 1
#endif

// SHA (SHA-1/SHA-256 message scheduling)
#if defined(__SHA__)
  #define HAS_SHA 1
#endif

// SHA-512 (Granite Rapids era, GCC 15+/clang 20+)
#if defined(__SHA512__)
  #define HAS_SHA512 1
#endif

// SM3 (Granite Rapids era)
#if defined(__SM3__)
  #define HAS_SM3 1
#endif

// SM4 (Granite Rapids era)
#if defined(__SM4__)
  #define HAS_SM4 1
#endif

// PCLMULQDQ (carry-less multiply)
#if defined(__PCLMUL__)
  #define HAS_PCLMULQDQ 1
#endif

// GFNI (Galois field instructions)
#if defined(__GFNI__)
  #define HAS_GFNI 1
#endif

// VAES (256/512-bit AES rounds)
#if defined(__VAES__)
  #define HAS_VAES 1
#endif

// VPCLMULQDQ (256/512-bit carry-less multiply)
#if defined(__VPCLMULQDQ__)
  #define HAS_VPCLMULQDQ 1
#endif

// RDRAND
#if defined(__RDRND__)
  #define HAS_RDRAND 1
#endif

// RDSEED
#if defined(__RDSEED__)
  #define HAS_RDSEED 1
#endif

// ADX (multi-precision add-carry)
#if defined(__ADX__)
  #define HAS_ADX 1
#endif

#endif // ARCH_X86_FAMILY


// ============================================================================
//                    ARM ISA detection (ACLE macros)
// ============================================================================

#if defined(ARCH_ARM_FAMILY)

// ==================== NEON (Advanced SIMD) ====================

// NEON support (ACLE standard macro)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define HAS_NEON 1
#endif

// NEON floating-point support (ACLE value macro; a bitset where 0x2 marks
// half-precision support)
#if defined(__ARM_NEON_FP)
  #define HAS_NEON_FP __ARM_NEON_FP
#endif

// NEON FP16 arithmetic (FEAT_FP16 scalar/vector arithmetic forms)
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  #define HAS_NEON_FP16_ARITH 1
#endif

// NEON FP16 fused multiply-add (FEAT_FP16FML, fmlal/fmlsl families)
#if defined(__ARM_FEATURE_FP16_FML)
  #define HAS_NEON_FP16_FML 1
#endif

// 32-bit SIMD (ARMv6 SIMD32, A32/T32)
#if defined(__ARM_FEATURE_SIMD32)
  #define HAS_ARM_SIMD32 1
#endif

// ==================== SVE (Scalable Vector Extension) ====================

// SVE base support (__ARM_FEATURE_SVE is defined to 1; it is a flag and
// does not carry the vector length)
#if defined(__ARM_FEATURE_SVE)
  #define HAS_SVE 1
#endif

// Fixed SVE vector length (VLS mode: only defined when the compiler commits
// to a specific width, e.g. -msve-vector-bits=512)
#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
  #define HAS_FIXED_SVE_BITS 1
  #define FIXED_SVE_BITS __ARM_FEATURE_SVE_BITS
#endif

// SME streaming vector length can differ from the ordinary SVE VL.  Until
// compilers standardize a predefined macro for a fixed SVL, CMake propagates
// the target guarantee explicitly after validating/detecting the requested
// architectural width.
#if defined(VECOPS_TARGET_FIXED_STREAMING_SVE_BITS)
  #if VECOPS_TARGET_FIXED_STREAMING_SVE_BITS < 128 || \
      VECOPS_TARGET_FIXED_STREAMING_SVE_BITS > 2048 || \
      (VECOPS_TARGET_FIXED_STREAMING_SVE_BITS % 128) != 0
    #error "fixed streaming SVE bits must be a multiple of 128 in [128, 2048]"
  #endif
  #define HAS_FIXED_STREAMING_SVE_BITS 1
  #define FIXED_STREAMING_SVE_BITS VECOPS_TARGET_FIXED_STREAMING_SVE_BITS
#endif

// Predicate tuples (svboolx2_t/svboolx4_t and svcreate/get/set) are exposed
// by Clang and by GCC 13 or newer.  Older GCC can still use SVE in VLS mode,
// where masks are represented as ordinary arrays and do not need these types.
#if defined(HAS_SVE) && \
    (defined(COMPILER_CLANG) || \
     (defined(COMPILER_GCC) && __GNUC__ >= 13))
  #define HAS_SVE_PREDICATE_TUPLES 1
#endif

// SVE vector operators (ACLE value macro, 2 on AArch64)
#if defined(__ARM_FEATURE_SVE_VECTOR_OPERATORS)
  #define HAS_SVE_VECTOR_OPERATORS 1
#endif

// SVE bf16 support (FEAT_BF16 SVE view: BFMMLA lane forms and bf16 FMLA;
// distinct from the NEON __ARM_FEATURE_BF16 below)
#if defined(__ARM_FEATURE_SVE_BF16)
  #define HAS_SVE_BF16 1
#endif

// SVE matrix multiply (FMMLA): i8mm / f32mm / f64mm optional extensions
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
  #define HAS_SVE_MATMUL_INT8 1
#endif

#if defined(__ARM_FEATURE_SVE_MATMUL_FP32)
  #define HAS_SVE_MATMUL_FP32 1
#endif

#if defined(__ARM_FEATURE_SVE_MATMUL_FP64)
  #define HAS_SVE_MATMUL_FP64 1
#endif

// B16B16 arithmetic format on the SVE side (FEAT_B16B16; SVE2.1/SME2.1 treat
// bf16 with full arithmetic semantics, required for bf16 FMA reductions)
#if defined(__ARM_FEATURE_SVE_B16B16)
  #define HAS_SVE_B16B16 1
#endif

// ==================== SVE2 ====================

// SVE2 base support
#if defined(__ARM_FEATURE_SVE2)
  #define HAS_SVE2 1
#endif

// SVE2 AES extension
#if defined(__ARM_FEATURE_SVE2_AES)
  #define HAS_SVE2_AES 1
#endif

// SVE2 Bitperm extension
#if defined(__ARM_FEATURE_SVE2_BITPERM)
  #define HAS_SVE2_BITPERM 1
#endif

// SVE2 SHA3 extension
#if defined(__ARM_FEATURE_SVE2_SHA3)
  #define HAS_SVE2_SHA3 1
#endif

// SVE2 SM4 extension
#if defined(__ARM_FEATURE_SVE2_SM4)
  #define HAS_SVE2_SM4 1
#endif

// SVE2.1
#if defined(__ARM_FEATURE_SVE2p1)
  #define HAS_SVE2P1 1
#endif

// ==================== SME (Scalable Matrix Extension) ====================

// SME base support.  The non-wide f32/bf16/i16i32 outer products are part of
// base FEAT_SME and have no separate feature macro; only the optional
// variants below do.
#if defined(__ARM_FEATURE_SME)
  #define HAS_SME 1
#endif

// SME Full A64 support in Streaming SVE mode. ACLE does not currently define
// a portable feature-test macro for FEAT_SME_FA64, so native CMake targets
// propagate the accepted -march=...+sme-fa64 guarantee explicitly.
#if defined(HAS_SME) && \
    (defined(__ARM_FEATURE_SME_FA64) || defined(VECOPS_TARGET_SME_FA64))
  #define HAS_SME_FA64 1
#endif

// __arm_locally_streaming and streaming-compatible codegen attributes
// (needed for hand-written SME kernels that enter streaming mode)
#if defined(__ARM_FEATURE_LOCALLY_STREAMING)
  #define HAS_LOCALLY_STREAMING 1
#endif

// SME F64F64 (f64 outer product, FEAT_SME_F64F64)
#if defined(__ARM_FEATURE_SME_F64F64)
  #define HAS_SME_F64F64 1
#endif

// SME F16F16 (fp16 outer product, FEAT_SME_F16F16)
#if defined(__ARM_FEATURE_SME_F16F16)
  #define HAS_SME_F16F16 1
#endif

// SME F8F16 (fp8 -> fp16 outer products, FEAT_SME_F8F16; Armv9.6-A era.
// ACLE documents the macros and GCC 15 emits them; clang 19 accepts the
// -march extensions but does not yet emit the macros)
#if defined(__ARM_FEATURE_SME_F8F16)
  #define HAS_SME_F8F16 1
#endif

// SME F8F32 (fp8 -> fp32 outer products, FEAT_SME_F8F32)
#if defined(__ARM_FEATURE_SME_F8F32)
  #define HAS_SME_F8F32 1
#endif

// SME I16I64 (int16 -> int64 wide outer product, FEAT_SME_I16I64)
#if defined(__ARM_FEATURE_SME_I16I64)
  #define HAS_SME_I16I64 1
#endif

// SME B16B16 arithmetic format (FEAT_SME_B16B16; bf16 multiply-accumulate
// in streaming mode)
#if defined(__ARM_FEATURE_SME_B16B16)
  #define HAS_SME_B16B16 1
#endif

// SME2
#if defined(__ARM_FEATURE_SME2)
  #define HAS_SME2 1
#endif

// SME2.1
#if defined(__ARM_FEATURE_SME2p1)
  #define HAS_SME2P1 1
#endif

// ==================== MVE (M-profile Vector Extension) ====================

// MVE (Cortex-M SIMD; ACLE value macro: 1 integer-only, 3 integer+float)
#if defined(__ARM_FEATURE_MVE)
  #define HAS_MVE 1
#endif

// MVE floating point (clang reports it as __ARM_FEATURE_MVE >= 3; the
// separate __ARM_FEATURE_MVE_FP spelling is kept for older compilers)
#if defined(__ARM_FEATURE_MVE_FP) || \
    (defined(__ARM_FEATURE_MVE) && __ARM_FEATURE_MVE >= 3)
  #define HAS_MVE_FP 1
#endif

// ==================== Cryptographic extensions ====================

// CRC32
#if defined(__ARM_FEATURE_CRC32)
  #define HAS_CRC32 1
#endif

// AES
#if defined(__ARM_FEATURE_AES)
  #define HAS_AES 1
#endif

// SHA2 (SHA-256)
#if defined(__ARM_FEATURE_SHA2)
  #define HAS_SHA2 1
#endif

// SHA3
#if defined(__ARM_FEATURE_SHA3)
  #define HAS_SHA3 1
#endif

// SHA512
#if defined(__ARM_FEATURE_SHA512)
  #define HAS_SHA512 1
#endif

// SM3 (Chinese national cryptography)
#if defined(__ARM_FEATURE_SM3)
  #define HAS_SM3 1
#endif

// SM4 (Chinese national cryptography)
#if defined(__ARM_FEATURE_SM4)
  #define HAS_SM4 1
#endif

// PMULL (64-bit polynomial multiply; 32-bit ARM macro — on AArch64 the
// PMULL(64) forms are covered by __ARM_FEATURE_AES)
#if defined(__ARM_FEATURE_PMULL)
  #define HAS_PMULL 1
#endif

// ==================== Compute extensions ====================

// Dot Product (UDOT/SDOT)
#if defined(__ARM_FEATURE_DOTPROD)
  #define HAS_DOTPROD 1
#endif

// BFloat16 (NEON bf16 scalar/vector forms incl. BFMMLA; the SVE view is
// __ARM_FEATURE_SVE_BF16 above)
#if defined(__ARM_FEATURE_BF16)
  #define HAS_BF16 1
#endif

// INT8 matrix multiply (SMMLA/UMMLA/USMMLA)
#if defined(__ARM_FEATURE_MATMUL_INT8)
  #define HAS_MATMUL_INT8 1
#endif

// I8MM (int8 matrix multiply accumulate, USDOT)
#if defined(__ARM_FEATURE_I8MM)
  #define HAS_I8MM 1
#endif

// ==================== Floating-point extensions ====================

// Hardware floating point (ACLE value macro, bitset of 16/32/64-bit support)
#if defined(__ARM_FP)
  #define HAS_VFP __ARM_FP
#endif

// FMA (fused multiply-add)
#if defined(__ARM_FEATURE_FMA)
  #define HAS_ARM_FMA 1
#endif

// ==================== Other features ====================

// Hardware integer divide
#if defined(__ARM_FEATURE_IDIV)
  #define HAS_IDIV 1
#endif

// Atomics (LSE - Large System Extensions)
#if defined(__ARM_FEATURE_ATOMICS)
  #define HAS_LSE 1
#endif

// RCPC (Release Consistent Processor Consistent)
#if defined(__ARM_FEATURE_RCPC)
  #define HAS_RCPC 1
#endif

// RCPC3 (FEAT_RCPC3: compilers report it as __ARM_FEATURE_RCPC >= 3; the
// separate __ARM_FEATURE_RCPC3 spelling is kept for compilers using it)
#if defined(__ARM_FEATURE_RCPC3) || \
    (defined(__ARM_FEATURE_RCPC) && __ARM_FEATURE_RCPC >= 3)
  #define HAS_RCPC3 1
#endif

// MOPS (memory operations: memset/memcpy/decmp)
#if defined(__ARM_FEATURE_MOPS)
  #define HAS_MOPS 1
#endif

// Branch Target Identification
#if defined(__ARM_FEATURE_BTI)
  #define HAS_BTI 1
#endif

// Pointer Authentication
#if defined(__ARM_FEATURE_PAUTH)
  #define HAS_PAUTH 1
#endif

// Memory Tagging Extension
#if defined(__ARM_FEATURE_MEMORY_TAGGING)
  #define HAS_MTE 1
#endif

// Guarded Control Stack
#if defined(__ARM_FEATURE_GCS)
  #define HAS_GCS 1
#endif

// JSCONV (JavaScript conversion, FJCVTZS)
#if defined(__ARM_FEATURE_JCVT)
  #define HAS_JSCONV 1
#endif

// FRINT (floating-point round to integer with rounding-mode variants)
#if defined(__ARM_FEATURE_FRINT)
  #define HAS_FRINT 1
#endif

// Directed rounding FRINT32/FRINT64 (to-int 32/64-bit forms with X/Z
// variants, FEAT_FRINTTS)
#if defined(__ARM_FEATURE_DIRECTED_ROUNDING)
  #define HAS_DIRECTED_ROUNDING 1
#endif

// Numeric FMINNM/FMAXNM forms
#if defined(__ARM_FEATURE_NUMERIC_MAXMIN)
  #define HAS_NUMERIC_MAXMIN 1
#endif

// RDM (rounding doubling multiply-add, SQRDMLAH family;
// __ARM_FEATURE_QRDMX is the AArch64 spelling, __ARM_FEATURE_RDM the
// 32-bit spelling)
#if defined(__ARM_FEATURE_RDM) || defined(__ARM_FEATURE_QRDMX)
  #define HAS_RDM 1
#endif

// Complex Number (FCMLA/FCADD)
#if defined(__ARM_FEATURE_COMPLEX)
  #define HAS_COMPLEX 1
#endif

// Q (saturation) Flag (32-bit ARM)
#if defined(__ARM_FEATURE_QBIT)
  #define HAS_QBIT 1
#endif

// DSP instructions (32-bit ARM)
#if defined(__ARM_FEATURE_DSP)
  #define HAS_DSP 1
#endif

// Unaligned access
#if defined(__ARM_FEATURE_UNALIGNED)
  #define HAS_UNALIGNED 1
#endif

// CLZ (count leading zeros)
#if defined(__ARM_FEATURE_CLZ)
  #define HAS_CLZ 1
#endif

// CB/CZB (compare and branch, 32-bit ARM)
#if defined(__ARM_FEATURE_CB)
  #define HAS_CB 1
#endif

// SEL (select, 32-bit ARM)
#if defined(__ARM_FEATURE_SEL)
  #define HAS_SEL 1
#endif

// LDREX/STREX (exclusive load/store)
#if defined(__ARM_FEATURE_LDREX)
  #define HAS_LDREX 1
#endif

// ==================== MSVC specifics ====================

#if defined(COMPILER_MSVC) && defined(ARCH_ARM64)
  // MSVC for ARM64 always provides NEON
  #if defined(_M_ARM64)
    #define HAS_NEON 1
    #define HAS_VFP 1
  #endif

  // MSVC ARM64 crypto.  _M_ARM64CRYPTO is not part of the documented MSVC
  // predefined-macro set; kept for compatibility, do not rely on it.
  #if defined(_M_ARM64CRYPTO)
    #define HAS_AES 1
    #define HAS_SHA2 1
    #define HAS_PMULL 1
  #endif
#endif

#endif // ARCH_ARM_FAMILY


// ============================================================================
//                           Convenience macros
// ============================================================================

// SIMD width level for the current target (applies to both architectures).
// A negative value denotes a scalable vector length (SVE); 0 means scalar.
// SVE must be probed before the fixed 128-bit ISAs: every AArch64 SVE target
// also defines NEON.
#if defined(HAS_AVX512F)
  #define SIMD_WIDTH 512
#elif defined(HAS_SVE)
  #define SIMD_WIDTH (-1)
#elif defined(HAS_AVX)
  #define SIMD_WIDTH 256
#elif defined(HAS_SSE) || defined(HAS_NEON)
  #define SIMD_WIDTH 128
#else
  #define SIMD_WIDTH 0  // Scalar
#endif

// Any SIMD support
#if defined(HAS_SSE) || defined(HAS_NEON) || defined(HAS_SVE) || defined(HAS_MVE)
  #define HAS_SIMD 1
#endif

// Any matrix engine
#if defined(HAS_AMX_TILE) || defined(HAS_SME)
  #define HAS_MATRIX_EXTENSION 1
#endif

// BFloat16 support. On ARM with SVE, bf16 is available via software
// emulation (reinterpret + uint16 ops) even when __ARM_FEATURE_BF16 is not
// defined by the compiler.
#if defined(HAS_AVX512_BF16) || defined(HAS_BF16) || defined(HAS_SVE)
  #define HAS_BFLOAT16 1
#endif

// FP16 (half-precision) support
#if defined(HAS_AVX512_FP16) || defined(HAS_NEON_FP16_ARITH) || defined(HAS_SVE)
  #define HAS_HALF_PRECISION 1
#endif


// ============================================================================
//                           Helper macros
// ============================================================================

// Architecture name string
#if defined(ARCH_X86_64)
  #define ARCH_NAME "x86_64"
#elif defined(ARCH_X86)
  #define ARCH_NAME "x86"
#elif defined(ARCH_ARM64)
  #define ARCH_NAME "ARM64"
#elif defined(ARCH_ARM)
  #define ARCH_NAME "ARM"
#else
  #define ARCH_NAME "Unknown"
#endif

// Strongest enabled SIMD ISA name
#if defined(HAS_AVX512F)
  #define SIMD_NAME "AVX-512"
#elif defined(HAS_AVX2)
  #define SIMD_NAME "AVX2"
#elif defined(HAS_AVX)
  #define SIMD_NAME "AVX"
#elif defined(HAS_SVE2)
  #define SIMD_NAME "SVE2"
#elif defined(HAS_SVE)
  #define SIMD_NAME "SVE"
#elif defined(HAS_SSE4_2)
  #define SIMD_NAME "SSE4.2"
#elif defined(HAS_SSE4_1)
  #define SIMD_NAME "SSE4.1"
#elif defined(HAS_SSSE3)
  #define SIMD_NAME "SSSE3"
#elif defined(HAS_SSE3)
  #define SIMD_NAME "SSE3"
#elif defined(HAS_SSE2)
  #define SIMD_NAME "SSE2"
#elif defined(HAS_SSE)
  #define SIMD_NAME "SSE"
#elif defined(HAS_NEON)
  #define SIMD_NAME "NEON"
#elif defined(HAS_MVE)
  #define SIMD_NAME "MVE"
#else
  #define SIMD_NAME "Scalar"
#endif

#endif // VECOPS_PLATFORM_FEATURES_H
