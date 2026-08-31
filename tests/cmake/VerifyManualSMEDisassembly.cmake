if(NOT DEFINED OBJDUMP OR NOT DEFINED BINARY)
    message(FATAL_ERROR "OBJDUMP and BINARY are required")
endif()
if(NOT DEFINED REQUIRE_MOPA)
    set(REQUIRE_MOPA ON)
endif()

execute_process(
    COMMAND "${OBJDUMP}" -d "${BINARY}"
    RESULT_VARIABLE _OBJDUMP_RESULT
    OUTPUT_VARIABLE _DISASSEMBLY
    ERROR_VARIABLE _OBJDUMP_ERROR)
if(NOT _OBJDUMP_RESULT EQUAL 0)
    message(FATAL_ERROR "objdump failed: ${_OBJDUMP_ERROR}")
endif()

string(REPLACE "\n" ";" _LINES "${_DISASSEMBLY}")
set(_INSIDE FALSE)
set(_STARTS 0)
set(_STOPS 0)
set(_CALLS "")
set(_ALLOWED_CALLS "")
foreach(_LINE IN LISTS _LINES)
    if(_LINE MATCHES "[ \t]smstart([ \t]|$)")
        if(_INSIDE)
            message(FATAL_ERROR "nested smstart in ${BINARY}: ${_LINE}")
        endif()
        set(_INSIDE TRUE)
        math(EXPR _STARTS "${_STARTS} + 1")
    endif()
    if(_INSIDE AND _LINE MATCHES "[ \t]bl(r)?[ \t]")
        if(DEFINED ALLOWED_CALL_REGEX AND
           NOT ALLOWED_CALL_REGEX STREQUAL "" AND
           _LINE MATCHES "${ALLOWED_CALL_REGEX}")
            string(APPEND _ALLOWED_CALLS "\n${_LINE}")
        else()
            string(APPEND _CALLS "\n${_LINE}")
        endif()
    endif()
    if(_LINE MATCHES "[ \t]smstop([ \t]|$)")
        if(NOT _INSIDE)
            message(FATAL_ERROR "smstop without active smstart in ${BINARY}: ${_LINE}")
        endif()
        set(_INSIDE FALSE)
        math(EXPR _STOPS "${_STOPS} + 1")
    endif()
endforeach()

if(_INSIDE)
    message(FATAL_ERROR "unterminated SME interval in ${BINARY}")
endif()
if(NOT _STARTS EQUAL _STOPS OR _STARTS EQUAL 0)
    message(FATAL_ERROR
        "invalid SME boundaries in ${BINARY}: starts=${_STARTS}, stops=${_STOPS}")
endif()
if(_CALLS)
    message(FATAL_ERROR "calls inside SME interval in ${BINARY}:${_CALLS}")
endif()
if(REQUIRE_ALLOWED_CALL AND NOT _ALLOWED_CALLS)
    message(FATAL_ERROR
        "expected an allowlisted call matching '${ALLOWED_CALL_REGEX}' "
        "inside an SME interval in ${BINARY}")
endif()
if(_DISASSEMBLY MATCHES "__arm_tpidr2_save|__arm_za_disable|__arm_sme_state")
    message(FATAL_ERROR "ACLE ZA runtime helper remains in ${BINARY}")
endif()
if(REQUIRE_MOPA AND
   NOT _DISASSEMBLY MATCHES "(fmopa|bfmopa|smopa|umopa|sumopa|usmopa)")
    message(FATAL_ERROR "expected MOPA instruction is absent from ${BINARY}")
endif()

set(_ALLOWED_CALL_COUNT 0)
if(_ALLOWED_CALLS)
    string(REGEX MATCHALL "\n" _ALLOWED_CALL_NEWLINES "${_ALLOWED_CALLS}")
    list(LENGTH _ALLOWED_CALL_NEWLINES _ALLOWED_CALL_COUNT)
endif()
message(STATUS
    "manual SME disassembly verified: regions=${_STARTS}, "
    "unexpected_calls_inside=0, allowed_calls_inside=${_ALLOWED_CALL_COUNT}")
