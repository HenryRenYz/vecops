cmake_minimum_required(VERSION 3.22)
project(vecops)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Address sanitizer

# ---------------------------------- Options -----------------------------------
set(VECOPS_GITHUB_ENDPOINT "https://github.com" CACHE STRING "Set endpoint address for GitHub repositories")

# -------------------------------- Dependencies --------------------------------
include(FetchContent)

# ------------------------------- Source Config --------------------------------
add_subdirectory(src)

# -------------------------------- Test Config ---------------------------------
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/googletest.git"
            GIT_TAG        v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(tests)
endif()
