// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/LinuxRuntime.h"

namespace vecops::memory::model {
/** memory-runtime/0.1 application policy, separate from hardware topology.
 * Each allocatable AllocationDomain with this opaque rule gets one arena.
 * Fraction denominators are unique physical capacity groups in that domain;
 * preallocation fractions are relative to its resolved hard limit.
 */
struct RuntimeArenaPolicy {
  std::string domain_rule;
  Limit hard_limit;
  Preallocate initial;
  std::uint64_t grow_by_bytes;
  std::uint64_t retain_idle_bytes;
  std::uint64_t page_bytes;
  BackingMode backing;
  bool allow_growth;
};
struct RuntimePolicy {
  RuntimeArenaPolicy arena;
};
/** Strict TOML schema; rejects unknown fields, ambiguous byte/fraction pairs,
 * nonfinite fractions, invalid enums and implicit size units. */
Result<RuntimePolicy> parse_runtime_policy(std::string_view);
/** Resolve the policy against process allocation grants. Installs shared domain
 * constraints into options before make_linux_runtime constructs its ledger.
 * Returns concrete arena options. Failure leaves options unchanged. This does
 * not create a launcher namespace. Supply options.capacity_directory to enforce
 * the resulting limits jointly across same-host participants.
 */
Result<std::vector<ArenaOptions>> apply_runtime_policy(const LinuxDiscovery&, LinuxRuntimeOptions&,
                                                       const RuntimePolicy&);
/** Construct the ledger with policy limits, then materialize every initial arena
 * before returning. Any arena failure destroys the partially built runtime. */
Result<std::unique_ptr<MemoryRuntime>> make_policy_linux_runtime(LinuxDiscovery, const RuntimePolicy&,
                                                                 LinuxRuntimeOptions = {});
} // namespace vecops::memory::model
