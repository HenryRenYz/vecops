# `_C` is the intentionally low-level native module. Python code installs its
# ergonomic public facade separately under `vecops`, while the wheel SDK copies
# the headers/core archive required by its lazy source-kernel compiler.
pybind11_add_module(_C MODULE module.cpp)
target_compile_features(_C PRIVATE cxx_std_20)
if(NOT TARGET vecops::compiler)
    message(FATAL_ERROR "VECOPS_BUILD_PYTHON requires VECOPS_BUILD_COMPILER=ON")
endif()
target_link_libraries(_C PRIVATE vecops::vecops vecops::compiler)
if(TARGET vecops::memory)
    target_link_libraries(_C PRIVATE vecops::memory)
    target_compile_definitions(_C PRIVATE VECOPS_PYTHON_HAS_MEMORY=1)
endif()
target_include_directories(_C PRIVATE "${VECOPS_SOURCE_DIR}/include")

install(TARGETS _C LIBRARY DESTINATION vecops COMPONENT python)
