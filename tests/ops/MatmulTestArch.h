//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_TESTS_OPS_MATMUL_TEST_ARCH_H
#define VECOPS_TESTS_OPS_MATMUL_TEST_ARCH_H

#include "vecops/platform/Features.h"

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace vecops::test::matmul {

struct MatmulTestArchTraits {
#if defined(ARCH_X86_FAMILY)
  static constexpr const char* name = "AMX";

  static bool enable() {
    constexpr long ArchRequestXcompPerm = 0x1023;
    constexpr long XfeatureTileData = 18;
    return syscall(
        SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
  }
#else
  static constexpr const char* name = "SME";

  static bool enable() {
    return true;
  }
#endif
};

} // namespace vecops::test::matmul

#endif // VECOPS_TESTS_OPS_MATMUL_TEST_ARCH_H
