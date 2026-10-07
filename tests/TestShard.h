// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_TESTS_TESTSHARD_H
#define VECOPS_TESTS_TESTSHARD_H

#ifndef VECOPS_TEST_SHARD_INDEX
#define VECOPS_TEST_SHARD_INDEX 0
#endif

#ifndef VECOPS_TEST_SHARD_COUNT
#define VECOPS_TEST_SHARD_COUNT 1
#endif

#if defined(VECOPS_TEST_SHARD_ACTIVE)
static_assert(VECOPS_TEST_SHARD_COUNT > 0);
static_assert(VECOPS_TEST_SHARD_INDEX >= 0);
static_assert(VECOPS_TEST_SHARD_INDEX < VECOPS_TEST_SHARD_COUNT);
#endif

#endif // VECOPS_TESTS_TESTSHARD_H
