include("${CMAKE_CURRENT_LIST_DIR}/cmake/VecopsTestTargets.cmake")

vecops_add_test(NAME ArrayTest FILES array/ArrayTest.cpp FOLDER array/array)
vecops_add_test(NAME TensorTest FILES tensor/TensorTest.cpp FOLDER tensor/tensor)
vecops_add_test(NAME GemmTest FILES array/GemmTest.cpp FOLDER array/gemm)
vecops_add_test(NAME LayoutTest FILES tensor/LayoutTest.cpp FOLDER tensor/layout)
vecops_add_test(NAME KernelLoopTest FILES kernel/LoopTest.cpp FOLDER kernel/loop)
vecops_add_test(NAME UtilMathTest FILES util/MathTest.cpp FOLDER util/math)

if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    vecops_add_test(
        NAME KernelLoopSizelessTest
        FILES kernel/LoopSizelessTest.cpp
        FOLDER kernel/loop
        ARCH SVE)
endif()

if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    set(VECOPS_TENSOR_KERNEL_ARCH Scalar AVX512 Native)
else()
    set(VECOPS_TENSOR_KERNEL_ARCH Scalar SVE SVE2 Native)
endif()

vecops_add_test(
    NAME TensorDataAccessTest
    FILES tensor/DataAccessTest.cpp TestUtils.cpp
    FOLDER tensor/data_access
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH})

set(VEC_BASE_ARCH ${VECOPS_ARCH_LIST})
if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    if(VECOPS_NATIVE_FIXED_SVE_BITS)
        list(APPEND VEC_BASE_ARCH NativeFixedSVE)
    endif()
endif()

vecops_add_test(NAME VecBaseTest FILES vec/BaseTest.cpp
    FOLDER vec/base ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecOptionsTest FILES vec/OptionsTest.cpp
    FOLDER vec/options ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecArithmeticTest FILES vec/ArithmeticTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecUnaryArithmeticTest FILES vec/UnaryArithmeticTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecFloatingUnaryTest FILES vec/FloatingUnaryTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecMathTest FILES vec/MathTest.cpp
    FOLDER vec/math ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecMemoryTest FILES vec/MemoryTest.cpp
    FOLDER vec/memory ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecPrefetchTest FILES vec/PrefetchTest.cpp
    FOLDER vec/memory ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecMaskConversionTest FILES vec/MaskConversionTest.cpp
    FOLDER vec/conversion ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion)
vecops_add_test(NAME VecConversionTest
    FILES vec/ConversionTest.cpp vec/ConversionExhaustiveTest.cpp
    FOLDER vec/conversion ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion)
vecops_add_test(NAME VecConversionMemoryTest
    FILES vec/ConversionMemoryTest.cpp vec/ConversionMemoryExhaustiveTest.cpp
    FOLDER vec/conversion ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion)
vecops_add_test(NAME VecReductionTest FILES vec/ReductionTest.cpp
    FOLDER vec/reduction ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecBitTest FILES vec/BitTest.cpp
    FOLDER vec/bit ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecComparisonTest FILES vec/ComparisonTest.cpp
    FOLDER vec/comparison ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecBasicTest FILES vec/BasicTest.cpp
    FOLDER vec/basic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecShuffleTest FILES vec/ShuffleTest.cpp
    FOLDER vec/shuffle ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecTaglessTest FILES vec/TaglessTest.cpp
    FOLDER vec/tagless ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecOverloadContractTest FILES vec/OverloadContractTest.cpp
    FOLDER vec/overload ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(
    NAME TensorTransformTest
    FILES tensor/TransformTest.cpp
    FOLDER tensor/transform
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH})
vecops_add_test(
    NAME LayerNormTest
    FILES
        ops/LayerNormTest.cpp
        TestUtils.cpp
    FOLDER ops/layer_norm
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH})
vecops_add_test(
    NAME SoftmaxTest
    FILES ops/SoftmaxTest.cpp TestUtils.cpp
    FOLDER ops/softmax
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH})
vecops_add_test(NAME VecRequestTest FILES vec/RequestTest.cpp
    FOLDER vec/request ARCH ${VEC_BASE_ARCH} LABELS vec)
