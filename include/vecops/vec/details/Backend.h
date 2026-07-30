#ifndef VECOPS_VEC_DETAILS_BACKEND_H
#define VECOPS_VEC_DETAILS_BACKEND_H

#include "vecops/vec/Capabilities.h"

namespace vecops::vec::details {

struct ScalarBackend {};
struct X86Backend {};
struct SVEBackend {};

#if defined(CPU_CAPABILITY_SVE)
using CurrentBackend = SVEBackend;
#elif defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
using CurrentBackend = X86Backend;
#else
using CurrentBackend = ScalarBackend;
#endif

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_BACKEND_H
