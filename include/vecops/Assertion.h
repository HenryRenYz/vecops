//
// Created by renyz on 2026/3/13.
//

#ifndef VECOPS_ASSERTION_H
#define VECOPS_ASSERTION_H

#include "vecops/CoreTypes.h"

#define VECOPS_INTERNAL_RUN_WHEN_FALSE(cond, ...)\
  do {                \
    if (!(cond)) {    \
       __VA_ARGS__;   \
    }                 \
  } while (0)

/**
 * Assertion, panic when cond is false, only check in debug mode.
 * usage: VECOPS_ASSERT(cond, message[, arg0[, arg1[, ...]]])
 *      where message and args are a format same to printf
 */
#ifdef VECOPS_DEBUG
  #define VECOPS_ASSERT(cond, ...) VECOPS_INTERNAL_RUN_WHEN_FALSE(cond, ::vecops::details::assertion_failed(__FILE__, __LINE__, VECOPS_FUNC_NAME, __VA_ARGS__))
#else
  #define VECOPS_ASSERT(cond ...) ((void) 0)
#endif

/**
 * Runtime checks, always exists, throw std::runtime_error when check failed.
 * usage: VECOPS_CHECK(cond, message[, arg0[, arg1[, ...]]])
 *      where message and args are a format same to printf
 */
#define VECOPS_CHECK(cond, ...) VECOPS_INTERNAL_RUN_WHEN_FALSE(cond, ::vecops::details::check_failed(__FILE__, __LINE__, VECOPS_FUNC_NAME, __VA_ARGS__))

namespace vecops::details {

[[noreturn]] VECOPS_NOINLINE
void assertion_failed(const char* file, int line, const char* fn_name, const char* message, ...);

[[noreturn]] VECOPS_NOINLINE
void check_failed(const char* file, int line, const char* fn_name, const char* message, ...);

} // namespace vecops::details


#endif //VECOPS_ASSERTION_H
