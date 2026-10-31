# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
include(CheckCSourceCompiles)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "VECOPS_BUILD_NUMA_LOADER requires Linux/glibc")
endif()
check_c_source_compiles("#define _GNU_SOURCE\n#include <features.h>\n#include <link.h>\n#ifndef __GLIBC__\n#error glibc required\n#endif\nint main(void) { return LAV_CURRENT + LA_SER_ORIG; }" VECOPS_HAVE_GLIBC_AUDIT)
if(NOT VECOPS_HAVE_GLIBC_AUDIT)
  message(FATAL_ERROR "VECOPS_BUILD_NUMA_LOADER requires the glibc audit interface")
endif()
add_library(vecops_numa_loader SHARED NumaLoader.c)
set_target_properties(vecops_numa_loader PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_compile_options(vecops_numa_loader PRIVATE -Wall -Wextra -Werror -fvisibility=hidden)
install(TARGETS vecops_numa_loader LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT numa-loader)
if(VECOPS_BUILD_PYTHON)
  install(TARGETS vecops_numa_loader LIBRARY DESTINATION vecops/_sdk/lib COMPONENT python)
endif()
install(FILES "${VECOPS_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/vecops-numa-loader"
  COMPONENT numa-loader)
