
file(GLOB_RECURSE _VECOPS_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
# Architecture-specialized objects are compiled with the target ISA by
# VecopsMultiarchTargets.cmake.  They must not enter the generic libvecops.a.
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/arch/")

add_library(vecops ${_VECOPS_SRCS})
target_compile_features(vecops PUBLIC cxx_std_20)
target_include_directories(vecops
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
    PUBLIC "${CMAKE_SOURCE_DIR}/include/vecops"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(vecops PUBLIC "$<$<CONFIG:Debug>:VECOPS_DEBUG>")
if(VECOPS_PRESERVE_SUBNORMALS)
    target_compile_definitions(vecops PUBLIC VECOPS_PRESERVE_SUBNORMALS=1)
endif()
target_compile_options(vecops PUBLIC "-Wno-ignored-attributes")
