include("${CMAKE_CURRENT_LIST_DIR}/cmake/VecopsTestTargets.cmake")

# This is an explicitly-invoked correctness/performance CLI rather than a
# GoogleTest binary, so it is intentionally not registered with CTest.
if(TARGET vecops::compiler)
    vecops_add_multiarch_executable(
        NAME NormOperatorCli
        FILES ops/NormOperatorCli.cpp
        FOLDER ops/norm_cli
        ARCH Native
        LIBRARIES vecops::compiler vecops
        DEFINITIONS
            "VECOPS_NORM_SDK_SOURCE_DIR=\"${VECOPS_SOURCE_DIR}\""
            "VECOPS_NORM_KERNEL_SOURCE=\"${VECOPS_SOURCE_DIR}/tests/runtime/NormOperatorKernel.cpp\"")
else()
    message(STATUS
        "Skipping NormOperatorCli because VECOPS_BUILD_COMPILER is disabled")
endif()

if(TARGET vecops::compiler)
    add_library(VecopsWorkspaceContextFixture MODULE
        runtime/WorkspaceContextFixture.cpp)
    target_compile_features(VecopsWorkspaceContextFixture PRIVATE cxx_std_20)
    target_include_directories(VecopsWorkspaceContextFixture PRIVATE
        "${VECOPS_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/runtime")
    set_target_properties(VecopsWorkspaceContextFixture PROPERTIES
        CXX_EXTENSIONS OFF
        FOLDER runtime)

    add_executable(VecopsRuntimeTest runtime/RuntimeTest.cpp)
    target_compile_features(VecopsRuntimeTest PRIVATE cxx_std_20)
    target_compile_definitions(VecopsRuntimeTest PRIVATE
        "VECOPS_TEST_KERNEL=\"${VECOPS_SOURCE_DIR}/tests/runtime/TestKernel.cpp\""
        "VECOPS_SOURCE_ROOT=\"${VECOPS_SOURCE_DIR}\""
        "VECOPS_WORKSPACE_CONTEXT_FIXTURE=\"$<TARGET_FILE:VecopsWorkspaceContextFixture>\"")
    target_link_libraries(VecopsRuntimeTest PRIVATE vecops::compiler)
    add_dependencies(VecopsRuntimeTest VecopsWorkspaceContextFixture)
    set_target_properties(VecopsRuntimeTest PROPERTIES CXX_EXTENSIONS OFF FOLDER runtime)
    add_test(NAME VecopsRuntimeTest
        COMMAND VecopsRuntimeTest "${CMAKE_CURRENT_BINARY_DIR}/runtime-work")
else()
    message(STATUS "Skipping VecopsRuntimeTest because VECOPS_BUILD_COMPILER is disabled")
endif()

vecops_add_test(NAME TensorTest FILES tensor/TensorTest.cpp FOLDER tensor/tensor)
vecops_add_test(NAME LayoutTest FILES tensor/LayoutTest.cpp FOLDER tensor/layout)
# GCC 13.3 miscompiles LayoutTest at -O2/-O3: the TestBody of
# ValueTest.CeilDivDynConstKeepsDivisibility gets a corrupted prologue and
# segfaults before its first statement (verified with gdb; clang 19 passes
# at -O1/-O2/-O3 and GCC passes at -O1, so this is a GCC codegen bug, not a
# library defect). Keep this test TU at -O1 on GCC until the bug is fixed
# upstream; the -O1 cap only affects test-binary performance.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    target_compile_options(LayoutTest PRIVATE -O1)
endif()
vecops_add_test(NAME KernelLoopTest FILES kernel/LoopTest.cpp FOLDER kernel/loop)
vecops_add_test(NAME KernelTile2DTest FILES kernel/Tile2DTest.cpp FOLDER kernel/tile2d)
vecops_add_test(NAME WorkspaceTest FILES kernel/WorkspaceTest.cpp
    FOLDER kernel/workspace ARCH Scalar)
vecops_add_test(NAME WorkspacePlanTest FILES execution/WorkspacePlanTest.cpp
    FOLDER execution/workspace_plan ARCH Scalar)
vecops_add_test(NAME WorkspaceContextTest FILES execution/WorkspaceContextTest.cpp
    FOLDER execution/workspace_context ARCH Scalar)
vecops_add_test(NAME MatmulTilerTest FILES matmul/TilerTest.cpp FOLDER matmul/tiler)
vecops_add_test(
    NAME ExecutionSessionTest
    FILES execution/ExecutionSessionTest.cpp
    FOLDER execution/session
    ARCH Scalar Native)
vecops_add_test(
    NAME ParallelTest
    FILES execution/ParallelTest.cpp
    FOLDER execution/parallel
    ARCH Scalar)

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
vecops_add_test(NAME VecTrigTest FILES
    vec/TrigSinTest.cpp
    vec/TrigCosTest.cpp
    vec/TrigTanTest.cpp
    vec/TrigSinPiTest.cpp
    vec/TrigCosPiTest.cpp
    vec/TrigTanPiTest.cpp
    vec/TrigSinCosTest.cpp
    vec/TrigSinCosPiTest.cpp
    vec/TrigSpecialTest.cpp
    vec/TrigF16ExhaustiveTest.cpp
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
vecops_add_test(
    NAME AttentionTest
    FILES ops/AttentionTest.cpp TestUtils.cpp
    FOLDER ops/attention
    ARCH Native)
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
        NAME MatmulPanelLifetimeTest
        FILES ops/MatmulPanelLifetimeTest.cpp
        FOLDER ops/matmul
        ARCH Native)
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
    vecops_add_test(
        NAME MatmulOrientationTest
        FILES ops/MatmulOrientationTest.cpp
        FOLDER ops/matmul
        ARCH Native)
elseif(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    set(VECOPS_SME_TEST_ARCH Native)
    if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
        list(APPEND VECOPS_SME_TEST_ARCH NativeFixedStreamingSVE)
    endif()
    if(VECOPS_NATIVE_HAS_SME)
        vecops_add_test(
            NAME MatmulPanelLifetimeTest
            FILES ops/MatmulPanelLifetimeTest.cpp
            FOLDER ops/matmul
            ARCH Native)
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
    vecops_add_test(
        NAME MatmulOrientationTest
        FILES ops/MatmulOrientationTest.cpp
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
        # A real split-K loop intentionally outlines one routed tile traversal
        # so first/middle/last share the same matrix-compute body.  The call is
        # made while StreamingZA is active and the callee contains the SME
        # leaves; keep this exact symbol alongside the predicate-tuple helpers
        # rather than weakening the interval-wide call audit.
        set(_VECOPS_SME_MATMUL_CALL_REGEX
            "(${_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX}|run_tiles_phased_shared)")
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            list(APPEND _VECOPS_SME_MATMUL_TARGETS
                MatmulTest-NativeFixedStreamingSVE
                MatmulBatchTest-NativeFixedStreamingSVE)
        endif()
        foreach(_TARGET IN LISTS _VECOPS_SME_MATMUL_TARGETS)
            set(_VECOPS_SME_MATMUL_CALL_ARGS "")
            if(_TARGET MATCHES "^MatmulTest-")
                list(APPEND _VECOPS_SME_MATMUL_CALL_ARGS
                    "-DALLOWED_CALL_REGEX=${_VECOPS_SME_MATMUL_CALL_REGEX}")
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
        # runtime-VL gather path and may share the common phased tile walker.
        # The fused sigmoid/SiLU epilogues deliberately call the shared SVE exp
        # leaf instead of cloning that large polynomial into every SME
        # microkernel. Allow these exact calls when present, while accepting a
        # fully inlined result as well.
        set(_VECOPS_SME_FUSION_TARGETS MatmulFusionTest-Native)
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            list(APPEND _VECOPS_SME_FUSION_TARGETS
                MatmulFusionTest-NativeFixedStreamingSVE)
        endif()
        set(_VECOPS_SME_FUSION_CALL_REGEX
            "(${_VECOPS_SME_PREDICATE_TUPLE_CALL_REGEX}|run_tiles_phased_shared(_n_major)?|numeric_limitsI(i|f)E(3(min|max)|8infinity)Ev|sve_exp_f32_(special|core))")
        foreach(_TARGET IN LISTS _VECOPS_SME_FUSION_TARGETS)
            add_test(
                NAME ${_TARGET}-Disassembly
                COMMAND "${CMAKE_COMMAND}"
                    "-DOBJDUMP=${CMAKE_OBJDUMP}"
                    "-DBINARY=$<TARGET_FILE:${_TARGET}>"
                    "-DALLOWED_CALL_REGEX=${_VECOPS_SME_FUSION_CALL_REGEX}"
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/VerifyManualSMEDisassembly.cmake")
            set_tests_properties(
                ${_TARGET}-Disassembly
                PROPERTIES LABELS "matmul;sme;disassembly")
        endforeach()
    endif()
endif()

vecops_add_test(
    NAME StatefulMatmulTest
    FILES ops/StatefulMatmulTest.cpp
    FOLDER ops/matmul
    ARCH Native)
vecops_add_test(NAME VecRequestTest FILES vec/RequestTest.cpp
    FOLDER vec/request ARCH ${VEC_BASE_ARCH} LABELS vec)
