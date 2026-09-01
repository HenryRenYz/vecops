include("${CMAKE_CURRENT_LIST_DIR}/cmake/VecopsTestTargets.cmake")

vecops_add_test(NAME TensorTest FILES tensor/TensorTest.cpp FOLDER tensor/tensor)
vecops_add_test(NAME LayoutTest FILES tensor/LayoutTest.cpp FOLDER tensor/layout)
vecops_add_test(NAME KernelLoopTest FILES kernel/LoopTest.cpp FOLDER kernel/loop)
vecops_add_test(NAME KernelTile2DTest FILES kernel/Tile2DTest.cpp FOLDER kernel/tile2d)
vecops_add_test(
    NAME ExecutionSessionTest
    FILES execution/ExecutionSessionTest.cpp
    FOLDER execution/session
    ARCH Scalar Native)

if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    set(VECOPS_TILE2D_GEMM_ARCH Scalar AVX512 Native)
else()
    set(VECOPS_TILE2D_GEMM_ARCH Scalar SVE Native)
    if(VECOPS_NATIVE_FIXED_SVE_BITS)
        list(APPEND VECOPS_TILE2D_GEMM_ARCH NativeFixedSVE)
    endif()
endif()
vecops_add_test(
    NAME KernelTile2DGemmTest
    FILES kernel/Tile2DGemmTest.cpp
    FOLDER kernel/tile2d
    ARCH ${VECOPS_TILE2D_GEMM_ARCH})
vecops_add_test(NAME UtilMathTest FILES util/MathTest.cpp FOLDER util/math)

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
vecops_add_test(NAME VecClampTest FILES vec/ClampTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecWideningDotTest FILES vec/WideningDotTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecUnaryArithmeticTest FILES vec/UnaryArithmeticTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecFloatingUnaryTest FILES vec/FloatingUnaryTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecRoundingTest FILES vec/RoundingTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecMathTest FILES vec/MathTest.cpp
    FOLDER vec/math ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecReciprocalTest FILES vec/ReciprocalTest.cpp
    FOLDER vec/math ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecLogTest FILES vec/LogTest.cpp
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
vecops_add_test(NAME VecMaskQueryTest FILES vec/MaskQueryTest.cpp
    FOLDER vec/basic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecSequenceTest FILES vec/SequenceTest.cpp
    FOLDER vec/basic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecCopySignTest FILES vec/CopySignTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
vecops_add_test(NAME VecSaturatingArithmeticTest
    FILES vec/SaturatingArithmeticTest.cpp
    FOLDER vec/arithmetic ARCH ${VEC_BASE_ARCH} LABELS vec)
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
set(VECOPS_TRANSPOSE_ARCH ${VECOPS_TENSOR_KERNEL_ARCH})
if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND VECOPS_NATIVE_FIXED_SVE_BITS)
    list(APPEND VECOPS_TRANSPOSE_ARCH NativeFixedSVE)
endif()
vecops_add_test(
    NAME TransposeTest
    FILES ops/TransposeTest.cpp
    FOLDER ops/transpose
    ARCH ${VECOPS_TRANSPOSE_ARCH})
if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    vecops_add_test(
        NAME MatmulPackTest
        FILES ops/MatmulPackTest.cpp TestUtils.cpp
        FOLDER ops/matmul_pack
        ARCH Scalar AVX512 Native)
    vecops_add_test(
        NAME MatmulTest
        FILES
            ops/MatmulTest.cpp
            ops/MatmulMixedPackingTest.cpp
            TestUtils.cpp
        FOLDER ops/matmul
        ARCH Native)
    vecops_add_test(
        NAME MatmulFusionTest
        FILES ops/MatmulFusionTest.cpp
        FOLDER ops/matmul
        ARCH Native)
    vecops_add_test(
        NAME MatmulBatchTest
        FILES ops/MatmulBatchTest.cpp
        FOLDER ops/matmul
        ARCH Native)
elseif(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    set(VECOPS_SME_TEST_ARCH Native)
    if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
        list(APPEND VECOPS_SME_TEST_ARCH NativeFixedStreamingSVE)
    endif()
    if(VECOPS_NATIVE_HAS_SME)
        vecops_add_test(
            NAME VecSMETest
            FILES vec/SMETest.cpp
            FOLDER vec/sme
            ARCH ${VECOPS_SME_TEST_ARCH}
            LABELS vec sme)
        if(CMAKE_OBJDUMP)
            set(_VECOPS_SME_VEC_TARGETS VecSMETest-Native)
            if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
                list(APPEND _VECOPS_SME_VEC_TARGETS
                    VecSMETest-NativeFixedStreamingSVE)
            endif()
            foreach(_TARGET IN LISTS _VECOPS_SME_VEC_TARGETS)
                add_test(
                    NAME ${_TARGET}-Disassembly
                    COMMAND "${CMAKE_COMMAND}"
                        "-DOBJDUMP=${CMAKE_OBJDUMP}"
                        "-DBINARY=$<TARGET_FILE:${_TARGET}>"
                        -P "${CMAKE_CURRENT_LIST_DIR}/cmake/VerifyManualSMEDisassembly.cmake")
                set_tests_properties(
                    ${_TARGET}-Disassembly
                    PROPERTIES LABELS "vec;sme;disassembly")
            endforeach()
        endif()
    endif()
    vecops_add_test(
        NAME MatmulPackTest
        FILES ops/MatmulPackTest.cpp TestUtils.cpp
        FOLDER ops/matmul_pack
        ARCH ${VECOPS_SME_TEST_ARCH})
    vecops_add_test(
        NAME MatmulTest
        FILES
            ops/MatmulTest.cpp
            ops/MatmulMixedPackingTest.cpp
            TestUtils.cpp
        FOLDER ops/matmul
        ARCH ${VECOPS_SME_TEST_ARCH})
    vecops_add_test(
        NAME MatmulFusionTest
        FILES ops/MatmulFusionTest.cpp
        FOLDER ops/matmul
        ARCH ${VECOPS_SME_TEST_ARCH})
    vecops_add_test(
        NAME MatmulBatchTest
        FILES ops/MatmulBatchTest.cpp
        FOLDER ops/matmul
        ARCH ${VECOPS_SME_TEST_ARCH})
    if(VECOPS_NATIVE_HAS_SME AND CMAKE_OBJDUMP)
        # BiSheng/Clang 19 keeps predicate-tuple accessors behind an SVE2.1
        # target boundary.  The helpers only regroup predicate registers, but
        # cannot be inlined into the base-SVE caller.  Allow exactly these
        # compiler boundary calls while rejecting every other call in an SME
        # interval.
        set(_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX
            "(make_word_groupEu10__SVBool_tS2_|get_wordILl[01]EEEu10__SVBool_t10svboolx2_t)")
        set(_VECOPS_SME_PACK_TARGETS MatmulPackTest-Native)
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            list(APPEND _VECOPS_SME_PACK_TARGETS
                MatmulPackTest-NativeFixedStreamingSVE)
        endif()
        foreach(_TARGET IN LISTS _VECOPS_SME_PACK_TARGETS)
            add_test(
                NAME ${_TARGET}-Disassembly
                COMMAND "${CMAKE_COMMAND}"
                    "-DOBJDUMP=${CMAKE_OBJDUMP}"
                    "-DBINARY=$<TARGET_FILE:${_TARGET}>"
                    -DREQUIRE_MOPA=OFF
                    "-DALLOWED_CALL_REGEX=${_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX}"
                    -DREQUIRE_ALLOWED_CALL=ON
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/VerifyManualSMEDisassembly.cmake")
            set_tests_properties(
                ${_TARGET}-Disassembly
                PROPERTIES LABELS "matmul_pack;sme;disassembly")
        endforeach()
        set(_VECOPS_SME_MATMUL_TARGETS
            MatmulTest-Native MatmulBatchTest-Native)
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            list(APPEND _VECOPS_SME_MATMUL_TARGETS
                MatmulTest-NativeFixedStreamingSVE
                MatmulBatchTest-NativeFixedStreamingSVE)
        endif()
        foreach(_TARGET IN LISTS _VECOPS_SME_MATMUL_TARGETS)
            set(_VECOPS_SME_MATMUL_CALL_ARGS "")
            if(_TARGET MATCHES "^MatmulTest-")
                list(APPEND _VECOPS_SME_MATMUL_CALL_ARGS
                    "-DALLOWED_CALL_REGEX=${_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX}"
                    -DREQUIRE_ALLOWED_CALL=ON)
            endif()
            add_test(
                NAME ${_TARGET}-Disassembly
                COMMAND "${CMAKE_COMMAND}"
                    "-DOBJDUMP=${CMAKE_OBJDUMP}"
                    "-DBINARY=$<TARGET_FILE:${_TARGET}>"
                    ${_VECOPS_SME_MATMUL_CALL_ARGS}
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/VerifyManualSMEDisassembly.cmake")
            set_tests_properties(
                ${_TARGET}-Disassembly PROPERTIES LABELS "matmul;sme;disassembly")
        endforeach()
        # Clang emits out-of-line numeric_limits min/max/infinity calls in the
        # runtime-VL gather and activation paths.  Keep these exact compiler
        # artifacts visible together with the predicate-tuple boundary calls.
        set(_VECOPS_SME_FUSION_TARGETS MatmulFusionTest-Native)
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            list(APPEND _VECOPS_SME_FUSION_TARGETS
                MatmulFusionTest-NativeFixedStreamingSVE)
        endif()
        set(_VECOPS_SME_FUSION_CALL_REGEX
            "(${_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX}|numeric_limitsI(i|f)E(3(min|max)|8infinity)Ev)")
        foreach(_TARGET IN LISTS _VECOPS_SME_FUSION_TARGETS)
            add_test(
                NAME ${_TARGET}-Disassembly
                COMMAND "${CMAKE_COMMAND}"
                    "-DOBJDUMP=${CMAKE_OBJDUMP}"
                    "-DBINARY=$<TARGET_FILE:${_TARGET}>"
                    "-DALLOWED_CALL_REGEX=${_VECOPS_SME_FUSION_CALL_REGEX}"
                    -DREQUIRE_ALLOWED_CALL=ON
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/VerifyManualSMEDisassembly.cmake")
            set_tests_properties(
                ${_TARGET}-Disassembly
                PROPERTIES LABELS "matmul;sme;disassembly")
        endforeach()
    endif()
endif()
vecops_add_test(NAME VecRequestTest FILES vec/RequestTest.cpp
    FOLDER vec/request ARCH ${VEC_BASE_ARCH} LABELS vec)
