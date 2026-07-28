#ifndef VECOPS_NVEC_DETAILS_BACKEND_H
#define VECOPS_NVEC_DETAILS_BACKEND_H

#include "vecops/vec/Capabilities.h"

namespace vecops::nvec::details {

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

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_BACKEND_H
