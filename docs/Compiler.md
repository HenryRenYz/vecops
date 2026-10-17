<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Native source-kernel compiler

Installed CMake SDKs keep the project license, third-party notices, and full
upstream texts under `share/licenses/vecops`; preserve this directory when
redistributing the installed headers or archives. Python SDK copies carry the
same materials within their `_sdk` tree.

`vecops::compiler::Compiler` turns one bound C++ file defining `__kernel__`
into an artifact DSO. It is optional: runtime-only deployments build without
`vecops_compiler` and can use only cache/AOT providers.

## Inputs and boundaries

`KernelCompilerConfig` is reusable compiler policy: SDK location, CMake and
native tools, a disposable work root, target architecture, and user compile/
link additions. `KernelBuildRequest` describes one independently-addressable
target. `KernelBuildBatchRequest` combines targets under one top-level CMake
project while retaining a generated fragment, binary subdirectory, and
artifact directory per task. The compiler uses no shell and does not mutate
the parent's environment. Configure/build output is streamed to the caller's
stderr as it arrives, so Ninja's native `[completed/total]` target progress is
visible during cold precompilation. The same bytes remain in batch-wide command
logs and are returned with the ordered per-task `BuildResult` values.

When `VECOPS_BUILD_CPU_LIST` is set to a Linux CPU-list expression, configure
and build subprocess trees run through `taskset --cpu-list`. This restores the
full intended build mask when an OpenMP-bound caller's master thread is pinned
to one CPU; it does not alter the caller or model-worker affinity.

`compile_batch()` validates the common SDK/toolchain contract, imports Vecops
once, adds every task fragment with `add_subdirectory`, and runs one configure
and one build command. CMake's generated build system therefore sees all
translation units and enforces one global parallel-job limit. Outputs are
written directly to task artifact directories; no shared install step is
required. `compile()` delegates to a one-task batch, so single and batch builds
cannot diverge. Both entry points remain deliberately cache-independent.

`compile_kernel(kernel_file, definition, call)` is the source-kernel adapter:

1. it binds the `KernelDef` and call metadata;
2. it generates `vecops_spec.h`, giving every named value a declaration in
   `vecops::spec`;
3. it generates an adapter that converts `VecopsCall` into typed Tensor views
   and scalar arguments, invokes `__kernel__`, and exports `KernelAbi.h` v1;
4. it builds and loads the resulting artifact as an `Executable`.

`compile_kernels()` performs the same operation for a span of already-bound
source recipes. It generates one task per specialization and submits the whole
set through `compile_batch()`.
Python's native operator batch planner first matches/binds every request, then
shares exact source-artifact identities within one compiler configuration. The
representative recipe retains every independent cold-build validator owner.
Bindings of the same source recipe share that owner, checked once per batch
before building and freshly again before publication. Different source recipes
retain independent checks, even when their artifacts share a build. No check
result is reused across the two boundaries or between batches. The loaded
result is adopted separately by every requesting provider. This does not replace call metadata or Dynamic
relationship validation, and the cache-independent C++ compiler entry points
continue to preserve their caller-supplied task list.

The compiler gets no persistent cache policy. `ArtifactExecutableProvider`
owns cache lookup/publication and invokes a compiler callback only when its
configured mode allows a build.

## Kernel source contract

Generated CPU-grid or explicit resident-team sources define `VECOPS_KERNEL_EXECUTION_CONTEXT` and take
`const VecopsExecutionContext*` as the leading `__kernel__` argument. The source
adapter forwards the size-versioned call context directly; it does not create
a workspace replay session solely to obtain an executor. This is an explicit
opt-in so unconstrained legacy workspace templates cannot accidentally bind
an execution-context pointer. Sources without this macro retain the existing
workspace-aware/ordinary invocation selection.
When DSL workspace sites (or worker capacity queries) are present, generated
sources instead take `execution::WorkspaceContext&`. The existing source
adapter selects its allocation/replay path; `WorkspaceContext::thread_pool()`
provides the borrowed call executor to grid/team lowering. Allocation sites
remain serial and metadata-derived, so runtime-parallelism replay guards
remain valid. Executor-only kernels keep their lightweight direct-context path.
Pair it with `KernelDef.runtime_parallelism=true` for a runtime grid scheduler;
otherwise execution parallelism retains legacy compile-time specialization.

Primitive lowering is prepared before C++ emission. Value-transform folding,
indexed/mask vector regions and guarded CPU software-pipeline planning use
the same frozen operation/cost/pass environment as existing kernels. The
emitter runs no compiler passes. Ordered custom reducers/scans expand to
structured IR; arbitrary scans do not currently imply SIMD prefix lowering.
CPU pipelining means prefetch, explicit double-buffer storage and opt-in
two-bank scheduling under an invariant-extent/no-alias proof, not arbitrary
alias-dependent code motion or an asynchronous event model.

Logical iteration contracts are shared by explicit local loops, folds, axis
folds, grids and backend ScheduleLoops. Contracts retain domain, carried state,
reducer ordering and scheduling separately from ISA representation. Builtins
and recognized independent associative custom reducers lower through FoldLoop
to the existing vector accumulation/finalization regions; ordered or coupled
custom folds do not silently change association. Generic fusion requires
memory/state proofs and treats collective/barrier/helper exits as boundaries.
CPU memory planning now emits a MatmulSchedule before source printing, with
cache/native domains, explicit pack/load/MAC actions and pipeline prime/drain
phases. Target bindings supply VL, native widths and runtime cache budgets;
optimization IR contains no C++ identifier strings. `LoweredArtifact.backend_ir`
exposes the prepared target IR, also available as `CompiledKernel.backend_ir`.
The printer consumes these domains and does not run a scheduling pass.

The source file contains a function named `__kernel__`. It can use values
declared by the generated header:

```cpp
using namespace vecops;
constexpr auto B = spec::B;
using ComputeType = spec::ComputeType;

void __kernel__(TensorLike auto input, TensorLike auto output, float64_t epsilon) {
  Workspace workspace;
  // output is caller-allocated; workspace allocation belongs to this kernel.
}
```

Named specialization values become either `inline constexpr meta::cint<N>`
(`ConstInt`) or a type alias (`DType`). A named Dynamic is emitted as its
constrained Meta type, for example
`using B = meta::Dynamic<1, 0, meta::kHiInf>`; construct it from tensor
metadata inside `__kernel__`. Its runtime value does not specialize the
artifact. Anonymous Const/dtype choices still participate in the artifact
specialization key but intentionally have no `spec` name. The C++ signature
must match the `KernelDef` positional order and mutability. All outputs are
out-style caller allocations; no external workspace is requested for
generated kernels.

Generated Tensor views construct every Dynamic coordinate from the current
`VecopsCall` sizes/strides, never from the metadata that happened to trigger
compilation. The adapter also checks equality relations for repeated named
Dynamic coordinates before entering `__kernel__`.

## SDK layouts

`SdkLayout::SourceTree` uses `add_subdirectory()` on a checkout.
`SdkLayout::Package` uses the installed `VecopsConfig.cmake`; `Auto` detects
which form the path provides. The Python wheel ships a relocatable SDK under
`vecops/_sdk` (headers, library, and CMake package), while editable Python
installs resolve the checkout SDK and keep built shared/static libraries out of
the Python source package.

Package SDK configuration preserves the native ARM feature contract detected
when vecops was built. In particular, `Native` JIT targets reuse the expanded
`-march` mapping and feature facts such as SME FA64 rather than relying on a
consumer compiler's interpretation of bare `-march=native`. This matters for
compiler drivers that support SME but do not enable every optional extension
from `native`. The exported static-library target also carries OpenMP as a
transitive dependency when the SDK was built with OpenMP support.

## Build observability and publication

`ToolchainSpec::output_mode` selects `quiet`, `progress` (default), or `verbose`.
Commands always retain complete configure/build logs in their attempt directory.
Progress output brackets each named stage, reports duration and points failures
to their logs. Quiet suppresses successful chatter; failures retain labelled
diagnostics and bounded log tails. Verbose adds raw tool output between stage
messages. `ToolchainSpec::observer` receives `BuildEvent` phase/state/log-path,
duration and exit status; Python exposes these via `Compiler.build_events` and
kernel `get_build_report()`.

Torch bridge compilation and native kernel compilation remain two separately
batched CMake stages. They have distinct Torch ABI/link policies and may use
different OpenMP/compiler choices. Within each compatible toolchain group,
all pending tasks share a single configure/build. Keeping the two policies
separate avoids coupling runtime-only kernels to a Torch installation. Progress
labels distinguish `Torch bridges` from `kernels`, so the stages are visible
instead of two unexplained CMake dumps. This change does not claim one combined
configure across both stages.

Python source recipes attest source/header inventories, implementation and
toolchain contents, plus compile/link policy. Before and after a cold build,
`SourceKernelRecipe::validate_inputs` rejects concurrent edits. Artifact cache
publication uses staging and atomic rename, with a full identity manifest
checked by the provider. Generated Torch bridges use the same staged/validated
publication discipline. Prepared executables remain immutable snapshots; their
execution path does not re-hash source. Cache-only readers consume existing
attestations without running native tools or creating a build tree.

`compile_commands.json` is retained for read-only adapter, specialization and
object inspection. Source/IR getters never invoke a build to synthesize missing
evidence, and optional disassembly tools report explicit unavailability.

### ARM thread-local geometry and toolchains

Python compilation captures the calling thread's independent SVE VL and SME
SVL. Both lengths participate in factory, lowering, artifact and batch identities;
hardware detection or a cost-profile override is not a vector-length snapshot.
Generated C++ uses the captured lengths for constant geometry while retaining
sizeless SVE representations. It does not require compiler support for a fixed
streaming-vector-bits flag. Actual code-generation architecture is recorded
separately from the optimization cost profile.

A batch may contain kernels captured under different VL/SVL pairs: the build
thread need not use those lengths. Entry adapters reject execution with a
different current pair; callers obtain the appropriate specialization through
`T.jit` or `T.compile` again. Worker regions propagate the caller's lengths and
restore the persistent workers' original settings before returning.

Compiler attestation includes effective `CXXFLAGS` when resolving implicit
headers and runtime libraries. A project-private GCC standard library selected
by clang is therefore a distinct content-addressed toolchain. On the ARM
cluster, `benchmarks/arm_toolchain_setup.sh` prepares this library under the
project's `build-toolchain-gcc14` directory, without changing af3 or global
configuration; source its generated `job-env.sh` inside scheduled jobs before
running Python/CMake. Both online and offline package caches remain inside the
selected private storage. Only C++ headers/development files and runtimes are
installed: clang remains the compiler and the host's C library is reused.
The generated task-local envfile also selects that same Clang pair for the
Torch bridge, overriding any older GNU bridge setting only in the current
job. Mixing an older GCC executable with the selected GCC14 headers is not a
supported configuration; no af3 activation hook or global setting is edited.
An offline, checksummed package bundle can be passed with
`--offline-package-dir` if compute nodes cannot access the login-node proxy.
When explicitly permitted for a cluster run, `--scratch-root` instead selects
an existing task-private, non-symlink `/tmp/vecops-*` or `/dev/shm/vecops-*`
directory owned by the current UID with mode `0700`. Headers and runtimes stay
inside that directory, not the shared temporary root; its `job-env.sh` applies
only to that compute-node job. Preserve logs/results separately before node-local
scratch is reclaimed. Changing the selected toolchain path/content changes its
attestation rather than reusing an artifact linked to an unavailable old runtime.

For quota-constrained validation, the verified deployment can instead retain
only the selected C++ headers, GCC development/start files and a checksummed
compressed bundle in the project. Each scheduled job unpacks those files into
its private scratch directory; shared runtime libraries remain in a durable
project-private directory selected by explicit link flags and RPATH. The
job-local envfile must locate the unpacked header tree relative to itself,
verify its payload and the durable runtime contents, and select clang before
any Python import. Runtime ELF dependencies must not point to reclaimed
scratch directories. This deployment does not install a second GCC compiler,
change af3, or require a global shell modification; retain probe and actual
JIT evidence separately from merely successful extraction.

`benchmarks/arm_job_env.sh` is the sourceable, self-contained entry point for
this deployment. Activate af3 in a scheduled ARM compute job, then set the
job-local `VECOPS_GCC14_BUNDLE`/`VECOPS_GCC14_BUNDLE_SHA256`,
`VECOPS_DURABLE_LIBGCC`/`VECOPS_DURABLE_LIBGCC_SHA256` and
`VECOPS_DURABLE_LIBSTDCXX`/`VECOPS_DURABLE_LIBSTDCXX_SHA256` pairs before sourcing
it. Its adjacent standard-library-only Python helper verifies the archive,
runtime libraries, active interpreter and project print header; extracts the
headers into a fresh UID-owned mode-0700 directory; and performs Clang syntax,
link and execution probes for format, floating `to_chars`, atomic wait/notify
and the actual print facility. It records the exact commands, compiler and
probe identities in `VECOPS_ARM_JOB_ENV_MANIFEST`.

Bundle extraction rejects hard links, special files, path escapes and arbitrary
symbolic links. The deployment's four `installation/lib/libgcc_s.so{,.1}` and
`installation/lib/libstdc++.so{,.6}` links are the sole exception: all four
must exist and name exactly the explicitly selected, hash-verified durable
runtime files. Link chains and archive members beneath those links are
rejected; extraction never follows a link to write runtime contents.

Source this shell before importing NumPy, Torch or vecops: it preloads both
durable runtimes for future Python processes and selects the same explicit
Clang drivers for CMake and the Torch bridge. The C++ driver invocation keeps
its `clang++` spelling even if it is a symlink to `clang-19`; resolved binary
contents are attested separately. A successful probe alone does not establish
JIT correctness, numerical accuracy, performance or editable installation.

BiSheng Clang 19.1.7 on the ARM validation cluster may omit its AArch64 UBSan
runtime archives even though `-fsanitize=undefined` is enabled. The
task-private `benchmarks/arm_sanitizer_setup.sh` installs only the matching
`libclang_rt.ubsan_standalone-aarch64.a` and
`libclang_rt.ubsan_standalone_cxx-aarch64.a` from the checksummed conda-forge
`compiler-rt_linux-aarch64-19.1.7` package into the project-local
`build-toolchain-compiler-rt19/clang19-resource` resource directory. It symlinks
the existing Clang builtin headers and verifies both ordinary execution and a
deliberate signed-overflow diagnostic using explicit Clang 19.1.7. This is a
task-local resource-dir selection, not a global compiler/runtime installation:
it does not modify af3, shell startup files, BiSheng's installation, or the
UBSan flags, and it never falls back to GCC. Keep the package SHA, private
archive hashes, and probe evidence with the validation run.

Inaccessible unrelated system-header paths are represented by explicit error
identities rather than crashing discovery or being silently omitted. Error
identities are never memoized, so restoring access causes contents to be
re-attested.

Relative include paths are anchored when a Python `Compiler` is created. Empty
components and `.` in `CPATH`, `CPLUS_INCLUDE_PATH`, and `C_INCLUDE_PATH` mean
that captured directory and are preserved as absolute entries; relative `-I`,
`-iquote`, `-isystem`, and `-idirafter` options in `CXXFLAGS` or compiler flags
are anchored there as well. The normalized environment is shared by the
compiler identity probe and native CMake build. An implicit working-directory
root from an empty path-list component is validated through each source's
resolved include closure instead of recursively inventorying unrelated
workspace outputs. Explicit include directories and non-empty include roots
retain directory inventory checks for shadowing headers.

For diagnosing an unexpected cache invalidation, set
`VECOPS_BUILD_INPUT_TRACE` to an absolute JSONL file path whose parent directory
already exists. Read/write identity checks append component digests (tool paths
and content digests, system-header roots, linked runtimes, hardware digest,
generator, and configuration key); failed publication validation appends the
source-input comparison and a bounded path-level diff. Environment values are
not included. This trace is observational: it does not affect identity hashes
or relax validation. Cache-only mode neither discovers tools nor writes trace
records, including when the variable is set.

Python compiler identities read complete file contents and rebuild sorted
directory inventories on every check. They do not memoize content from
filesystem notifications: writes through an open shared mapping or an external
hard link can escape those notifications. Metadata probes retain explicit
permission and missing-input diagnostics, and never replace content reads.
Cold compilation validates input identities before and after building;
individual reads and scans are not an atomic filesystem snapshot.

### Scalable-vector compiler compatibility

ARM builds using SVE probe the selected C++ compiler for a reproducible
SimplifyCFG common-store sinking crash (a scalable `TypeSize` is requested as
a fixed integer). Only when that exact diagnostic occurs and a second probe
succeeds does configuration add `-mllvm=-simplifycfg-sink-common=false`.
Other optimization passes and `-O3` remain enabled. The probe covers both
source C++ and generated DSL kernels; its bounded execution, compiler identity,
result and selected flag are retained for build/cache attribution. CMake
applies the same test for its ARM target configuration.
