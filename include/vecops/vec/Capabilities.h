//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_VEC_CAPABILITIES_H
#define VECOPS_VEC_CAPABILITIES_H

#include "vecops/platform/Capabilities.h"

/**
 * @file Capabilities.h
 * @brief Define vec representation limits for the selected CPU capability.
 *
 * `VEC_WIDTH` is the native fixed width in bits, or -1 for VLA SVE.
 * `MAX_VEC_WIDTH` bounds fixed storage. `VEC_MAX_POW` and `VEC_HW_MIN_POW`
 * constrain logical Tag scaling. CPU selection belongs to
 * `platform/Capabilities.h`, so non-vector modules need not depend on vec.
 */

#if defined(CPU_CAPABILITY_AVX512)
#define VEC_WIDTH 512
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (5)
#define VEC_HW_MIN_POW (-2)
#elif defined(CPU_CAPABILITY_AVX2)
#define VEC_WIDTH 256
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (5)
#define VEC_HW_MIN_POW (-2)
#elif defined(CPU_CAPABILITY_AVX)
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (5)
#define VEC_HW_MIN_POW (-2)
#elif defined(CPU_CAPABILITY_SVE)
#define VEC_WIDTH (-1)
#define MAX_VEC_WIDTH 2048
#define VEC_MAX_POW (2)
#define VEC_HW_MIN_POW (0)
#elif defined(CPU_CAPABILITY_NEON)
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (2)
#define VEC_HW_MIN_POW (0)
#else
#define VEC_WIDTH 128
#define MAX_VEC_WIDTH VEC_WIDTH
#define VEC_MAX_POW (5)
#define VEC_HW_MIN_POW (0)
#endif

#endif // VECOPS_VEC_CAPABILITIES_H
