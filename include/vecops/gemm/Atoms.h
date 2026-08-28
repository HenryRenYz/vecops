//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_ATOMS_H
#define VECOPS_GEMM_ATOMS_H

#include "vecops/Features.h"
#include "vecops/gemm/KernelBase.h"

#if defined(ARCH_X86_FAMILY)
#include "vecops/gemm/details/amx/Atoms.h"
#endif

#if defined(HAS_SME_FA64)
#include "vecops/gemm/details/sme/Atoms.h"
#endif

#endif // VECOPS_GEMM_ATOMS_H
