//
// Created by renyz on 2026/3/21.
//

#ifndef VECOPS_CAPABILITIES_H
#define VECOPS_CAPABILITIES_H

#include "vecops/Features.h"

#if defined(ARCH_X86_FAMILY)
  #if defined(HAS_AVX512F) && defined(HAS_AVX512CD) && defined(HAS_AVX512BW) && defined(HAS_AVX512DQ)
    #define HAS_CPU_CAPABILITY_AVX512 1
  #endif
  #if defined(HAS_AVX2)
    #define HAS_CPU_CAPABILITY_AVX2 1
  #endif
  #if defined(HAS_AVX)
    #define HAS_CPU_CAPABILITY_AVX 1
  #endif
  #define VEC_MAX_POW (5)
  #define VEC_HW_MIN_POW (-2) // min POW2 in ScalableTag that Vec can be of a hardware vector rather than software emulated
#endif // ARCH_X86_FAMILY
#if defined(ARCH_ARM_FAMILY)
  #if defined(HAS_SVE)
    #define HAS_CPU_CAPABILITY_SVE 1
    #define VEC_MAX_POW (2) // only up to svdtypex4_t is supported till SVE2p1
    #define VEC_HW_MIN_POW (0) // min POW2 in ScalableTag that Vec can be of a hardware vector rather than software emulated
  #endif
  #if defined(HAS_NEON)
    #define HAS_CPU_CAPABILITY_NEON 1
    #define VEC_MAX_POW (2) // TODO temp
    #define VEC_HW_MIN_POW (0) // TODO temp
  #endif
#endif // ARCH_ARM_FAMILY

#if defined(CPU_CAPABILITY_AVX512) || (defined(HAS_CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY))
  #if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX512)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX512
  #define CPU_CAPABILITY_AVX512 1
  #define VEC_WIDTH 512
#endif // CPU_CAPABILITY_AVX512

#if defined(CPU_CAPABILITY_AVX2) || (defined(HAS_CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX2)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX2
  #define CPU_CAPABILITY_AVX2 1
  #define VEC_WIDTH 256
#endif // CPU_CAPABILITY_AVX2

#if defined(CPU_CAPABILITY_AVX) || (defined(HAS_CPU_CAPABILITY_AVX) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY AVX
  #define CPU_CAPABILITY_AVX
  #define VEC_WIDTH 128
#endif // CPU_CAPABILITY_AVX

#if defined(CPU_CAPABILITY_SVE) || (defined(HAS_CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_SVE)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY SVE
  #define CPU_CAPABILITY_SVE 1
  #define VEC_WIDTH (-1) // scalable
#endif // CPU_CAPABILITY_SVE

#if defined(CPU_CAPABILITY_NEON) || (defined(HAS_CPU_CAPABILITY_NEON) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_NEON)
    #error "CPU capability redefined or does not supported by compiler option"
  #endif
  #define CPU_CAPABILITY NEON
  #define CPU_CAPABILITY_NEON 1
  #define VEC_WIDTH 128
#endif // CPU_CAPABILITY_NEON

#if !defined(CPU_CAPABILITY)
  #define CPU_CAPABILITY GENERIC
  #define CPU_CAPABILITY_GENERIC 1
  #define VEC_WIDTH 128 // default width for scalar vector implementation
  #define VEC_MAX_POW (5) // default max pow2 for scalar implementation
  #define VEC_HW_MIN_POW (0)
#endif // CPU_CAPABILITY

#endif //VECOPS_CAPABILITIES_H
