//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_PLATFORM_CAPABILITIES_H
#define VECOPS_PLATFORM_CAPABILITIES_H

#include "vecops/Features.h"

/**
 * @file Capabilities.h
 * @brief Select the translation unit's CPU capability from compiler features.
 *
 * Selection is entirely compile-time. An explicitly requested
 * `CPU_CAPABILITY_*` must be supported by the compiler command line; otherwise
 * preprocessing fails instead of adding a runtime dispatch or fallback. When
 * no capability is requested, the strongest enabled implementation is chosen.
 *
 * This header owns target/capability selection only. Vector widths and logical
 * Tag limits remain in `vec/Capabilities.h`, while execution-state policy is
 * implemented by the platform files under `execution/details`.
 */

#if defined(CPU_CAPABILITY_GENERIC) && !defined(CPU_CAPABILITY)
#define CPU_CAPABILITY GENERIC
#endif

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#if defined(HAS_AVX512F) && defined(HAS_AVX512CD) && \
    defined(HAS_AVX512BW) && defined(HAS_AVX512DQ)
#define HAS_CPU_CAPABILITY_AVX512 1
#endif
#if defined(HAS_AVX2)
#define HAS_CPU_CAPABILITY_AVX2 1
#endif
#if defined(HAS_AVX)
#define HAS_CPU_CAPABILITY_AVX 1
#endif
#endif

#if defined(ARCH_ARM_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#if defined(HAS_SVE)
#define HAS_CPU_CAPABILITY_SVE 1
#endif
#if defined(HAS_NEON)
#define HAS_CPU_CAPABILITY_NEON 1
#endif
#endif

#if defined(CPU_CAPABILITY_AVX512) || \
    (defined(HAS_CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX512)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX512
#define CPU_CAPABILITY_AVX512 1
#endif

#if defined(CPU_CAPABILITY_AVX2) || \
    (defined(HAS_CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX2)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX2
#define CPU_CAPABILITY_AVX2 1
#endif

#if defined(CPU_CAPABILITY_AVX) || \
    (defined(HAS_CPU_CAPABILITY_AVX) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_AVX)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY AVX
#define CPU_CAPABILITY_AVX 1
#endif

#if defined(CPU_CAPABILITY_SVE) || \
    (defined(HAS_CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_SVE)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY SVE
#define CPU_CAPABILITY_SVE 1
#endif

#if defined(CPU_CAPABILITY_NEON) || \
    (defined(HAS_CPU_CAPABILITY_NEON) && !defined(CPU_CAPABILITY))
#if defined(CPU_CAPABILITY) || !defined(HAS_CPU_CAPABILITY_NEON)
#error "CPU capability redefined or not supported by compiler options"
#endif
#define CPU_CAPABILITY NEON
#define CPU_CAPABILITY_NEON 1
#endif

#if !defined(CPU_CAPABILITY)
#define CPU_CAPABILITY GENERIC
#define CPU_CAPABILITY_GENERIC 1
#endif

#endif // VECOPS_PLATFORM_CAPABILITIES_H
