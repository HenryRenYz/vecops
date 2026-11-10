<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# Memory resource migration

Maintained projects: `~/project/vecops-neo` and `~/project/af3-torchneo`.
C++ implements memory resources, capacity, backing, mappings, eager pools,
transfers and workspace binding. Python implements graph placement, lifetimes,
communication preparation and integer-offset allocation programs.

## Current implementation

- Agents, pools, capacity groups, services and directed connections use opaque
  IDs. TOML/discovery binds hardware; generic planning has no socket/die rules.
- Process views preserve physical shared limits and separate execution,
  allocation and access. LX2 DDR controller service is shared across each die.
- Eager arenas grow, cache and trim aligned blocks. Tensor aliases and unfinished
  asynchronous users retain their physical owners.
- Graph temporaries and operator workspace share lifetime and placement rules.
  Resident model state remains separate and may be placed during preparation.
- Offset-only programs separate address reservation, capacity admission,
  acquisition and binding. Shared roots support export/import/composite maps;
  eligible local whole roots use private movable backing.
- Graph, eager, framework and communication demands share capacity authority.
  Estimates expose exact group/pool/constraint shortfalls.
- Distributed preparation holds successful pending grants while peers replan.
  A stopped collective finalizer withdraws grants and retains offset proposals;
  communication growth invalidates proposals. Fatal failures return grants.
- Cold preparation can learn observed physical-free loss while accounting is
  unchanged. It learns observed declines across untruncated groups, preserves
  reusable cache and independent limits, and leaves native admission unchanged.
  Compact fallback releases failed preferred-layout metadata before recompiling.
- Resource costs retain shared-resource accounting and optional measured path
  curves. Query caches are scoped to immutable compilation attempts.

The previous allocator/session and planner-factory APIs and active callers are
removed. References: [Memory](../../../docs/Memory.md),
[Workspace planning](../../../docs/WorkspacePlanning.md),
[Python](../../../docs/Python.md), and the
[design document](../../../docs/memory-resource-model/index.html).

## Qualification and measurements

R48 corrects shared backing names for independently linked C++ SDK consumers.
Fresh builds passed274 native tests and377 affected frontend tests on each
architecture, plus10 real ARM SDMA tests. The new regression loads two ordinary
SDK consumers in one process and verifies unique backing names and full cleanup.
AF3 graph, numerical and communication source remains identical to R47.

R47 completed full ARM inference for both inputs at8r and16r. All13 reference
fields and both cross-world13-field comparisons are bitwise equal; finite
values, sampled NUMA placement, communication and capacity refund pass. For
T1044 the numerical reference is the earlier migrated implementation; an
original full output is still pending. R48 local740aa matches all13 original
reference fields bitwise.

Measurements use zero warmups and one timed inference per attempt, S5/N200/R10,
seed1 for740aa and seed42 for T1044. Local740aa uses56 workers on CPUs0–55 at
1.8GHz maximum. ARM uses36 workers per rank on exclusive HBM Flat nodes; seconds
are the maximum across ranks. Preparation and post-inference checks are outside
the clock. No further local x86 T1044 full runs are required.

| Target | Input | Original s | Candidate s | Change | Status |
|---|---|---:|---:|---:|---|
| x86 / R48 |740aa|188.575228|189.275693|+0.371451%|Pass within1%|
| ARM8r / R47 / single socket |740aa|68.429607|68.330300|−0.145123%|Pass|
| ARM16r / R47 / double socket |740aa|86.112268|73.407798|−14.753380%|Pass|
| ARM8r / R47 / single socket |T1044|Unavailable|875.701956|Unknown|Accepted: candidate time and correctness|
| ARM16r / R47 / double socket |T1044|Unavailable|807.857483|Unknown|Accepted: candidate time and correctness|

R48 local740aa first measured190.842901s (+1.202530%). The authorized noise
reassessment measured189.275693s (+0.371451%); both records are retained. The
ARM740aa original/candidate nodes differ:cn11208/cn11211 for8r and
cn11207/cn11210 for16r. The8r workers bind to one socket; memory allocation
permission covers the whole node. Slowdown tolerance is1%.

Original ARM T1044 did not produce a valid complete baseline; several shortened
capacity diagnostics failed with OOM. Relative T1044 performance remains unknown.
The user accepted the measured candidate8r/16r timings and correctness with this
limitation explicitly recorded. No additional original full run is required.
The latest shortened original call and the independent page-lifecycle probe are
diagnostic evidence only and are not timing denominators.

## Deployment

R48 ordinary local and ARM editable installations are complete. Both passed
canonical imports, full SDK/legal and RECORD checks, eager escaped aliases,
retained graph output and complete capacity refund, the installed C++ SDK
consumer, and ordinary2r/3r communication with complete refunds and no window
growth. ARM installation job2084963 succeeded on cn11210. Its canonical projects
are `/home/share/youxin/vecops-neo` and `/home/share/youxin/af3-torchneo`, using the
`af3` environment. The remote source transaction preserves seven independent
edits; the five independent local memory-benchmark edits are retained.

Five CPU profiles recommend2048KiB graph-buffer alignment. The accepted
740aa/8r measurement explicitly uses64KiB; it does not change generic API or
profile defaults.

Actual requests, frozen source manifests, numerical outputs, qualification
logs, timing comparisons and reversible deployment backups remain under
`build/memory-migration/`. Current root reviews include
`r47-all-full-results-root-reviewed.json`,
`r48-local-final-performance-root-reviewed.json`,
`r48-local-ordinary-install-root-reviewed.json` and
`r48-arm-ordinary-install-root-reviewed.json`. Historical release proofs remain
immutable; current installation receipts identify the fresh R48 native payload.

Historical cleanup reclaimed7.791195GiB: unrelated old builds, obsolete object
files and downloaded package archives. Byte-identical historical output files
were hard-linked with their paths and hashes retained. Sources, environments,
current validation evidence and deployment rollback assets are preserved; see
`r47-cleanup-root-combined.json`.
