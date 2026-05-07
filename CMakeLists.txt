cmake_minimum_required(VERSION 3.28)
project(vecops)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# ---------------------------------- Options -----------------------------------
set(VECOPS_GITHUB_ENDPOINT "https://github.com" CACHE STRING "Set endpoint address for GitHub repositories")

# -------------------------------- Dependencies --------------------------------
include(FetchContent)
enable_testing()
FetchContent_Declare(
        googletest
        GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/googletest.git"
        GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# ------------------------------- Source Config --------------------------------
add_subdirectory(src)

# -------------------------------- Test Config ---------------------------------
add_subdirectory(tests)
