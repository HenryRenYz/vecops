// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_TESTS_RUNTIME_WORKSPACE_CONTEXT_FIXTURE_H
#define VECOPS_TESTS_RUNTIME_WORKSPACE_CONTEXT_FIXTURE_H

#include <cstdint>

#include "vecops/runtime/CallAbi.h"

namespace vecops::test {

inline constexpr std::uint64_t WorkspaceContextFixtureBytes = 64;

struct WorkspaceContextObservation {
  std::uint64_t workspace_query_calls = 0;
  std::uint64_t run_calls = 0;
  const VecopsExecutionContext* query_context = nullptr;
  const VecopsExecutionContext* run_context = nullptr;
  void* run_workspace = nullptr;
  std::uint64_t run_workspace_size = 0;
  std::uint32_t requested_threads = 0;
  void* stream = nullptr;
  std::uint64_t context_flags = 0;
};

} // namespace vecops::test

#endif // VECOPS_TESTS_RUNTIME_WORKSPACE_CONTEXT_FIXTURE_H
