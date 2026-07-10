
file(GLOB_RECURSE _VECOPS_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(vecops ${_VECOPS_SRCS})
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
