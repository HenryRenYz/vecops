# Python JIT API

The public Python API is implemented above the low-level `vecops._C` pybind11
module. Applications use `vecops.language` for kernel templates and the
classes and functions exported by `vecops` for runtime services; `_C` is an
expert/debugging escape hatch and mirrors C++ types
without Python convenience behavior.

Importing `vecops` does not import NumPy or Torch. Dtype conversion recognizes
those frameworks only when the application has already loaded them.

## Floating-point subnormal mode

`vecops.set_flush_subnormals(True)` enables hardware flush-to-zero for the
calling thread and publishes `VECOPS_FLUSH_SUBNORMALS=1` for vecops worker
tasks. On AArch64 this sets FPCR.FZ and FPCR.FZ16; on x86 it sets MXCSR.FTZ
and MXCSR.DAZ. `False` clears those bits. Other FP control bits are preserved.
Use `vecops.current_thread_flush_subnormals()` to inspect the calling thread.

Call this before importing a framework that creates its own CPU worker pool:
new threads inherit the caller's mode, while an already-running framework
pool must be configured on its own threads. Existing vecops workers apply the
process setting when entering a logical task. The mode changes treatment of
tiny floating-point values and can change results near zero; applications
should opt in after checking their numerical tolerance.

## Native thread-pool regions

`vecops.native_thread_pool_region(threads)` keeps the process-local native
worker team active across a bounded series of vecops submissions:

```python
with vecops.native_thread_pool_region(torch.get_num_threads()):
  state = native_attention(state)
  state = native_transition(state)
```

The context manager is nestable and always releases the active region on an
exception. Use it only when every parallel operation inside the region uses
the native pool; an interleaved Torch/OpenMP operation would otherwise compete
with the active workers for the same CPU set. Pass `enabled=False` to make the
region a no-op when selecting the executor at runtime.

## Compiler and directories

`vecops.Compiler()` discovers CMake and matching C/C++ compilers lazily. The
priority is an explicit argument, the corresponding environment variable, and
finally a supported executable on `PATH`. Resolved absolute executable paths
are passed to CMake. The generator defaults to `CMAKE_GENERATOR`, then Ninja
when its executable is available, otherwise CMake's platform default.

Persistent artifacts default to the platform user cache:

- `$VECOPS_CACHE_DIR`, when set;
- `$XDG_CACHE_HOME/vecops` or `~/.cache/vecops` on Linux;
- `~/Library/Caches/vecops` on macOS;
- `%LOCALAPPDATA%/vecops/Cache` on Windows.

The default build root is `$VECOPS_BUILD_DIR` or the `builds` directory below
the same user cache. Use a local scratch/NVMe path explicitly when appropriate.
`cache-only` does not discover build tools and creates neither cache nor build
directories.

The default cache namespace contains an explicit Python SDK generation. It is
bumped whenever generated-adapter semantics or the packaged SDK ABI changes,
so a new compiler cannot silently reuse an artifact emitted by an older
adapter generator. An application-supplied `cache_namespace` takes ownership
of that compatibility boundary.

The native module also exports a build-API version. Batch compilation checks
that version and its required symbols before touching CMake. This catches an
editable installation whose Python files come from a newer checkout while
`vecops._C` still comes from an older build, and reports the exact editable
reinstall command instead of failing later with a missing pybind attribute.

```python
compiler = vecops.Compiler(
  target="native",
  cache_mode="read-write",  # cache-only | read-write | compile-only
  cache_dir=None,
  build_dir=None,
  cc=None,
  cxx=None,
  cmake=None,
  generator=None,
  jobs=None,
)
```

When `target="native"`, `VECOPS_TARGET_ARCH` may select a more specific
process-wide default without changing every recipe declaration. For example,
`VECOPS_TARGET_ARCH=NativeFixedSVE` keeps the native ISA feature set while
compiling ordinary SVE code for the width detected by the source SDK (or the
explicit `VECOPS_FIXED_SVE_BITS` CMake setting). An explicitly non-native
`target=` argument is never overridden. Fixed-width artifacts require the
runtime threads to use the matching SVE vector length; use `Native` when that
contract is not controlled by the embedding.

A wheel contains a relocatable compiler SDK below `vecops/_sdk`, including
headers, `libvecops`, and the CMake package. An editable installation maps
Python modules and headers to the checkout while keeping `_C`, libraries, and
generated CMake state in its isolated build/install directory; it does not
place `.so` or `.a` files in `python/vecops`. The packaged CMake target also
exports vecops' transitive runtime dependencies, including OpenMP when enabled;
JIT modules that link the static archive therefore do not need to rediscover
or spell those dependencies themselves.

## Expression-based language frontend

`vecops.language` (`import vecops.language as T`) is the Python
metaprogramming frontend. Source kernels use `T.jit`, `T.template`, and Python
dimension expressions. The old signature-derived decorator and string tensor
annotation modules have been removed. `T.template(source=...)` forwards to the
source file's `__kernel__`; omitting `source` parses the function body as a
typed Python kernel DSL and emits CPU C++ using public vector APIs.

```python
import vecops
import vecops.language as T

@T.jit(out_idx=[1])
def scale(B, D, stride_D, ComputeType):
  # These outer annotations, if present, are ordinary user hints. T.jit does
  # not inspect them; the actual call values determine specialization kinds.
  @T.template(source="kernels/scale.cpp")
  def kernel(
    input: T.Tensor((B, D), (stride_D, 1), vecops.float32),
    output: T.Tensor((B, D), dtype=vecops.float32),
    factor: float = 2.0,
  ):
    ComputeType  # capture a value used only by C++

  return kernel
```

`T.jit` normalizes the actual bound call, including `*args`, `**kwargs`, and
recursive tuple elements. Integers and booleans become named `T.Const`
values, framework/native dtypes become `T.DType`, and `None` remains `None`.
Lists, dictionaries passed as values, sets, bytearrays, and other mutable
arguments are rejected. The normalized call is both the elaboration input and
the decorator cache key. Forwarding a normalized value into another `T.jit`
function rebinds its specialization name to the callee parameter.

`T.Const` arithmetic builds exact integer expressions. `T.dynamic()` creates
a bounded, aligned runtime value which does not specialize the artifact;
expressions combining named dynamic and Const leaves validate shape/stride
relations at runtime. A comparison involving an unresolved dynamic value is
not proven and is false in Python control flow, while `T.is_value(x)` reports
whether `x` is compile-time known. `T.any` is the unconstrained dynamic value.

`T.Tensor(shape, strides, dtype)` accepts tuples or lists in the annotation.
Omitted strides mean dense row-major for the current source backend; the IR
retains the omission so a future backend may select a padded non-overlapping
layout. Explicit strides may describe padding, overlap, zero strides, or
negative NumPy strides. Automatic output allocation computes the complete
reachable storage interval rather than multiplying shape extents, so aliases
inside such layouts remain in bounds. Torch does not expose negative-stride
tensors and rejects that automatic-allocation case.

`out_idx` uses Python indices (including negative indices) over the template's
positional parameters; keyword-only runtime scalars are excluded. It controls
output order. A caller may provide those tensors explicitly or omit them. Omitted
outputs are allocated in the framework/device of the first tensor argument;
one output is returned directly, multiple outputs as a tuple, and no outputs
as `None`. Dtype is taken from the annotation or inferred from the first tensor
when the annotation omits it.

Template annotations and external values referenced in the template body are
captured as C++ specialization values. A bare reference in the ignored body is
therefore the explicit way to capture a value which does not occur in an
annotation. The body is never executed for a source template. `T.compile`
caches the lowered operator by template IR, source/dependency content,
`out_idx`, and compiler configuration. `T.jit` adds a higher-level elaboration
cache and only calls `T.compile` when the elaborated result is a
`TemplateFunction`; ordinary Python callables and forwarding wrappers are
cached and returned unchanged.

Calls made inside `vecops.collect_compile_requests()` allocate output metadata
or framework tensors, record the source-kernel request, and skip execution.
The resulting requests use the existing `vecops.compile_batch()` shared CMake
build graph, so distinct captured configurations remain distinct targets while
duplicates are removed. This is the supported batch-compilation boundary for
the first migration stage.

### Source-less vector kernels

```python
@T.jit(out_idx=[1])
def softmax(M, N, dtype):
  @T.template
  def kernel(A: T.Tensor((M, N), dtype=dtype),
             C: T.Tensor((M, N), dtype=dtype)):
    for row in T.serial(M):
      x = T.load(A[row, :], dtype=T.float32)
      T.store(C[row, :], T.softmax(x))
  return kernel
```

The outer factory runs ordinary Python. The inner body is **parsed, not
executed**: dynamic `if`/`while` become runtime control flow; constants are
partially evaluated. `range`/`T.serial`, nested `T.grid`, constant `T.unroll`,
scalar loop-carried variables, `break`/`continue`, assertions and source-located
errors are supported. Branch-carried scalar variables must be initialized
before a dynamic branch; tile/view rebinding across dynamic branches or loops
requires explicit `T.alloc` storage. An unrolled region cannot use
`break`/`continue`. Kernels return through output tensors.

`T.func` helpers are source-parsed and inlined, including their immutable
captures in cache identity. Arbitrary host calls, mutable captures, exception
handling, generators and optional DSL tensor operands are rejected. A source
template's body remains ignored as before. Source-less definitions require
inspectable, file-backed Python source; they are not bytecode interpreters.
DSL tensor annotations require an explicit dtype. Assertions remain checked in
release builds; scalar indexing still requires valid caller-provided bounds.

`T.tile(A)` and positive-step slices describe memory regions; negative physical
strides are allowed. `T.transpose(A)`/`A.T` change address mappings without
copying. A store of a last-two-axis permutation automatically selects dense
loads, an explicit named register butterfly and dense stores when legal,
including leading batch planes and fused elementwise expressions. Masks,
broadcasts and unusual layouts retain gather/scatter. `T.transpose_copy(A, C,
compute_dtype=...)` is the rank-2 visible `T.func` algorithm, with a linear-copy
specialization for dense single-row/column shapes. Conversions fuse into
load/store boundaries; a sole widening conversion may move after the
permutation on x86 without changing lane values or rounding.
`T.reshape` requires a dense view and preserves element count. Shapes broadcast
statically where possible; unknown relations are checked at runtime. A source
extent must equal the destination extent or be one; incompatible extents error.

### Matrix products and packing

`T.dot((A, B), output, acc=None, acc_dtype=None, implementation="auto")`
computes the rank-two tile product. A is `[M,K]`, B is **`[K,N]`** (unlike the
C++ Matmul operator's `[N,K]` convention). Inputs may be views, elementwise
tile DAGs or explicitly packed values. The result is also returned as a tile
expression for a fused final store. `T.dot(A, B, acc=None)` is the pure
returning form. Dot does not introduce cache blocking or persistent packing;
instruction-format staging is still necessary for raw AMX/SME inputs.

```python
a = T.cast(T.library.layernorm(T.load(A)) * scale + bias, "bfloat16")
b = T.load(B, dtype="bfloat16")
value = T.library.matmul((a, b), C, acc=T.load(D), BK=256)
T.store(C, T.max(value * 0.5, 0))
```

`T.library.matmul` adds cache traversal and automatic packing, with optional
positive `BM`, `BN`, `BK`, `packing="auto"/"none"/"A"/"B"/"AB"` and `swap=True`.
Unspecified blocks use target-side L1/L2 budgets and the discovered last-level
cache, never the Python build host's cache. Known small working sets and GEMV
may bypass the cache traversal. Block sizes are aligned to the instruction
family. Column-contiguous output can swap/transposed A/B and the **whole fused
epilogue**, not just the product. Already packed operands are not silently
reinterpreted by this transformation.

Row/column GEMV use vector widening-dot/FMA leaves; matrix leaves use public
AMX or SME instructions when their dtype/target contract permits, otherwise
vector FMA. Accumulation defaults to int32 for 8-bit integer inputs, float64
if either input is float64, otherwise float32. Native accumulator loads can
initialize hardware accumulators directly. K-block continuation uses native
accumulator storage; the original acc/bias is applied only on the first block
and the user epilogue only on the last. Intermediate output conversions are
not inserted between blocks.

`T.pack(source, output=None, side="A", format="auto", dtype=None)` creates
or writes a padded physical buffer and returns a packed logical tile. Formats
are `dense`, `amx`, `sme`, or target-selected `auto`. `T.packed(buffer,
logical_shape, side=..., format=...)` wraps an external packed buffer. Its
physical storage must be dense, large enough and independent of the output;
shape/capacity/alias validation precedes hardware-state entry. A/B formats
are distinct: AMX A is row/K-blocked, B is dword-group/VNNI interleaved; SME
uses spatial panels and K groups parameterized by SVL. Dense B stores `[N,K]`.
Explicit AMX packing supports bf16/fp16/int8/uint8 and requires an AMX target.
Full AMX spatial/K panels use unmasked load/convert and a prepared dword
permutation; only the final spatial panel or K block clears transformed padding.
Prepacked cache panels retain the original physical layout and logical
origins, including K-block continuation; block sizes do not reinterpret the
buffer. Dense `T.pack` and runtime dense cache panels use the automatic
transpose/vector memory planner. External packed buffers are tied to the
target ISA and (for SME) SVL ABI; shape reinterpretation is not a format
conversion.

`T.func` is inlined **before** fusion analysis, including nested
`T.library.*` calls in frozen capture identity. Rank-two layernorm/softmax
hoist row statistics once; their final affine/exp tail stays in the operand
DAG. The CPU fusion policy estimates operation cost and reloads to choose
recomputation versus materialization, or puts the transform in packing loads.
Dynamic row/column GEMV selects the non-packing reuse model when it bypasses
packing; a tiled API alone does not imply that B is produced only once.
`pass_options={"cpu_matmul_fusion": "auto"/"always"/"never"}` provides an
explicit diagnostic override; `auto` is a heuristic, not an optimality proof.
`cpu_gemv_columns` (integer 1–8, default 4) controls independent GEMV result
columns per register group for CPU tuning; it is not a tile shape restriction.
`cpu_gemv_k_unroll` (1, 2 or 4) overrides independent K accumulator banks.
The target cost policy defaults to one bank on x86 and two on other CPU
targets: four independent output columns already hide short x86 dot
dependencies. Known K >= 512 retains two banks according to the
`gemv_k_unroll_long` / `gemv_k_unroll_long_min` policies; unknown K also retains
the conservative two-bank schedule. Simple epilogues use two-column then
single-column remainders with at least four independent accumulators; known one/two
column groups narrow to their real column count and retain at least four
independent accumulators. Mathematical epilogues retain vector-grouped output
evaluation. An explicit K-bank override bypasses the long-K policy.
Automatic AMX cache panels align to the selected register family, not a
universal 32-column boundary. A proven single M group does not automatically
pack B a second time; explicit packing/block sizes remain authoritative.
The x86 `amx_cache_b_panel_divisor` (4) bounds automatic B panels by a quarter
of the target's runtime L2, rounded to the selected register-family quantum.
This reserves residency for A and the B producer/packed-copy footprint, rather
than budgeting packed B alone. `amx_cache_b_panel_max_m` (256) limits the policy
to shallower M reuse; larger M keeps the original budget. Dynamic M selects
the budget once outside panel loops. These are measured scheduling heuristics,
not guarantees about cache occupancy. Explicit BN always wins; a zero divisor
disables the cap, and a zero max-M removes its M gate. Other target profiles
leave the cap disabled. Both settings can be overridden with a cost profile.
On x86, a native, untransformed, K-contiguous B also skips automatic cache-panel
packing when the original constant M is below the `amx_native_b_pack_min_m`
policy (128) and both spatial extents fill complete register groups. Unknown
M and ragged groups remain conservative; conversions, transforms,
non-K-contiguous B and explicitly requested packing retain their own paths.
The original problem's M, not a shortened cache panel, determines this gate.
Small unblocked multi-M-group products retain their resident packed buffer;
repeating the raw transpose is not necessarily cheaper than that small copy.
Explicit dynamic cache phases share one packing wrapper with runtime family
alignment, rather than cloning transformed panel producers for every family;
shortened leaves still select a suitable register family.
Known instruction-aligned extents survive aligned cache-panel slicing. Full
AMX K loops hoist compatible raw-A addresses/strides and packed-A/B offsets
into individual pointer cursors; only unproven spatial/K tails retain staging
and padding checks. A padded K walk is legal for AMX packing, not dense packing.
The x86 `amx_m_major_n_max` (128) and `amx_m_major_ratio` (4) policies select
M-major instruction traversal for tall small-N strips. Dynamic selection uses
one coordinate body rather than duplicating the entire matrix kernel; other
CPU profiles leave this AMX-specific policy disabled by default.
Known-size library products retain the cache driver's N-major order: a tall
whole problem does not prove that shortened MC/NC panels remain tall. Direct
dot retains whole-tile M-major; dynamic traversal uses its separately measured
coordinate-selection path. Panel-local profitability remains a tuning task.
The cost policies `matmul_unblocked_bytes` (512 KiB for constant shapes) and
`matmul_runtime_unblocked_bytes` (128 KiB for dynamic shapes) control the
automatic small-footprint bypass. Dynamic matrices above the latter cutoff
retain cache-panel packing even when an equal-sized constant kernel can be
simplified enough to profit from an unblocked schedule.
Alias snapshots preserve a statically known unit-stride axis while removing
padding and overlap; compatible unmasked loads select an address/layout before
one load/convert instead of selecting between two complete load expressions.
Nested products, sibling products and product-to-pack/reduction boundaries
materialize only the necessary intermediate products. A trailing rank-one
bias/scale follows the matrix orientation and cache-panel origin.
GEMV mathematical epilogues group completed horizontal sums into a legal
vector before evaluating exp/log/etc, including strided output and K-block
continuation. SME source generation keeps vector-only math exceptional
paths inline, preserving their numerical contract and the no-call gate.
No generated kernel calls `vecops::ops::Matmul` or private kernel helpers.

`T.Vector(dtype, N=...)` or `T.Vector(dtype, pow=...)` describes a tagged virtual
vector. It is not limited by the C++ Vec maximum power. `T.load(A,
vector=descriptor)` clips its logical extent to the region; `T.lanes(descriptor)`
reports the target-relative extent. Large extents strip-mine into native
legalized ScalableTag groups. Negative powers must yield an integral, positive lane count.
Conversions retain the logical lane extent and its original scalable dtype
basis, rather than recomputing the extent from the destination dtype.

Loads and stores independently choose computation/storage dtype and fuse to
`vec::load_convert`/`vec::store_convert`; fp16, bf16, fp32 and fp64 storage are
supported. Use fp32/fp64 loads when wider accumulation is needed. `T.load(view,
mask=predicate, other=0)` never reads masked-off lanes; `T.store(view, value,
mask=predicate)` preserves masked-off output locations. Tail masks always
filter memory accesses. Transpose/noncontiguous views do not force full buffers.
Intermediate conversion rounding is preserved, even when the final dtype
matches the input dtype.

Arithmetic, comparison, predicate logic, `T.where`, `T.fma`, vector math and
`T.reduce_add`, `T.reduce_min`, `T.reduce_max` reductions (all-region or one
constant axis) pass through explicit Vector IR before lowering to public vec
operations. Exponential, logarithm, reciprocal and reciprocal-square-root math
accept `accuracy='strict'|'fast'|'estimate'` (not `sqrt`/`abs`); `T.exp_neg` and
`T.exp2_neg` require nonpositive input (NaNs retain backend behavior). Reductions
associate by native vector chunks and horizontal reductions, not Python scalar
order. Results need not be bit-identical across targets.
`T.min(a, b)` and `T.max(a, b)` are strictly elementwise; unary forms and the old
`T.sum` spelling are not supported. Reductions and `T.serial` accept a positive,
compile-time `unroll=` keyword (default 2, 1 disables expansion). `T.layernorm`
and `T.softmax` forward their optional `unroll=` to their reductions.

Unrolling copies each operation across iterations, not the complete body per
iteration. Seven adjacent vector chunks group as 4+2+1, with powers P+2/P+1/P.
Target `VEC_MAX_POW` and lane-preserving dtype rebind requirements limit merging;
the C++ preprocessing alternatives keep generated sources portable to SVE.
Oversized virtual vectors remain streamable, not forced into one native value.
Nonadjacent reduction-axis copies retain distinct lane domains. Loop-carried
scalar dependencies and break/continue retain ordered loops; alias-dependent
serial iterations have a runtime ordered fallback. Tails are always guarded.

Adjacent compatible, independent reductions with the same unroll count can
share a traversal (currently up to two reductions). An independent elementwise
side output can fuse into a reduction. A consumer needing the completed global
reduction remains a later phase. Adjacent elementwise stores can share a vector
traversal with runtime alias guards and a sequential fallback.

`T.layernorm` and `T.softmax` are visible rank-one/rank-two DSL algorithms:
centered variance and maximum-subtracted exponentials respectively. Rank two
normalizes each row, with statistics separated from the fusible tail. Affine/post-ops can
be expressed directly before `T.store`. `T.layernorm(x, variance='moments')`
explicitly selects the existing C++ operator's sum/sum-of-squares algorithm,
allowing a fused reduction pass; it can suffer cancellation for large means and
small variance. The default `variance='centered'` retains the more stable extra
pass. Cheap expressions and nonescaping
full-buffer scratch are forwarded; expensive expressions reused after a
reduction may be cached based on native size. Inputs are not implicitly noalias:
overlapping remaps and tile values retained across writes use conditional
snapshots, while ordinary disjoint calls do not allocate snapshots. Writable
region strides must pass a conservative non-overlap proof; repeated scatter
locations are rejected, unlike source kernels which may define their own policy.

The initial backend uses standard-library AST/IR and the existing compiler SDK,
with no MLIR/TVM dependency. Generated sources are content-addressed below
`cache_dir.parent / 'dsl-sources'`. `compiled.generated_source`,
`compiled.kernel_ir`, and `compiled.source` expose diagnostics/artifacts.
`cache-only` requires those pre-generated sources alongside the binary cache
and never creates missing directories or discovers build tools.

The architecture and explicitly deferred features are described in
[new-dsl-design.md](internal/new-dsl-design.md). Reproducible single-thread norm
comparisons are in `benchmarks/dsl_norm.py`.

### Compiler passes

`T.passes` exposes a named pass registry and `PassManager`. Kernel, vector and
CPU memory-planning stages have separate pipelines, ordered by `before`/`after`.
Managers snapshot the registry; analyses and traces are local to a run.

```python
@T.passes.register_pass("my-analysis", before="fuse-reductions")
def my_analysis(ir, context):
  # context.analysis(name, compute, ir) caches a run-local analysis.
  return T.passes.PassResult(ir, changed=False)

compiled = T.compile(template, pass_options={"max_fused_reductions": 4})
# Optional explicit selection: T.compile(template, pipeline=[...pass names...]).
print(compiled.kernel_ir.pass_trace)
print(compiled.vector_ir.regions)  # typed operations, copies, groupings, traces
```

The built-in kernel passes canonicalize converting memory, perform expression
CSE, plan computation reuse, fuse reductions/reduction-side-outputs/elementwise
stores, and expand independent serial operations. Vector passes create
operation-major copies, coalesce groups and legalize dtype lane domains.
Separate CPU passes split full chunks from masked tails, select wider automatic
tile chunks from accumulated region cost and a conservative live-vector budget, and legalize multiword math into native-word
calculations while preserving coalesced memory requests. Explicit Vector tags
and logical unroll counts do not change. `pass_options` can override
`cpu_full_chunks`/`cpu_wordwise_math` (bool), `cpu_math_chunk_pow` and
`cpu_reduction_chunk_pow` (0/1/2, default 1; 0 disables the corresponding
wider automatic chunk). Single-reduction regions can use a wider base to
expose more accumulator chains; fused reductions already supply independent
chains. These are tuning policies, not performance guarantees.
Register plugins before factory elaboration, or call the existing factory
`cache_clear()` when changing a pipeline after `T.jit` has cached a kernel.
These passes do not reinterpret source-backed C++ kernels.

Operation semantics are registered centrally under `_compiler/operations`,
separately from CPU implementation capabilities. Recompute decisions count the
whole fused DAG after helper inlining/CSE, not membership in an expensive-op
list. Shared producers count once; reductions, matrix products and materialized
values are boundaries. `compiled.vector_ir.regions[*].cost` reports estimated
compute throughput, memory throughput, critical path and live-vector hints.
The hint is not a physical register allocator or an exact spill prediction.

Cost profiles are immutable and layered: default, CPU, ISA, processor, explicit
overrides. Cycles are estimates per native word (scalar queries per scalar
operation); latency is separate from reciprocal throughput. Current profiles
retain low-confidence heuristic values, not measured processor calibrations.
Unknown operations receive a nonzero conservative fallback and are not implicitly
CSE/transpose eligible. `Native` may inspect the local CPU; explicit SVE/SME or
AVX targets never inherit the build host's ISA. `cpu_cost_target` and
`cpu_cost_processor` override cost selection only, never compiler ISA legality.
For custom compiler flags/cross toolchains, specify the cost target explicitly
if it differs from `Compiler.target`.

```python
profile = T.cost.CostProfile(
    "my-calibration", version="1",
    rules=(T.cost.CostRule(
        T.cost.OperationCost(throughput_cycles=4, latency_cycles=8,
                             provenance="local microbenchmark"),
        match=(("operation", "exp"), ("dtype", "float32")),
    ),),
    policies=(("reuse_compute_budget", 6),),
)
compiled = T.compile(template, pass_options={"cost_profiles": (profile,)})
# Alternatively T.cost.register_profile(profile), then use its name.
```

Later profiles override earlier matching rules; within a profile priority and
selector count determine precedence, with ambiguous ties rejected. Policy
budgets for recomputation, temporary traffic and vector registers live alongside
profiles, separately from operation costs. Matmul uses a continuous DAG-work
versus temporary-traffic budget, retaining dynamic packing/GEMV predicates;
this is a coarse model, not an exact cache/port simulator. All compiler stages
share one frozen semantic/implementation/cost environment, whose fingerprint
participates in generated-source identity. Registry changes also invalidate
decorator caches, including profiles selected by a returned template's target.
Pass-registry changes still require `cache_clear()` as described above.

Development formatting is pinned in the `dev` extra. Run
`python -m ruff format python/vecops/language` or append `--check` in CI;
the project uses two-space indentation and a 120-column target.

The `cpu-memory` stage runs `cpu-plan-transpose` and `cpu-plan-matmul` on the prepared emission
catalog. All three managers are frozen before user passes execute. This pass
recognizes layout conversions, proves dense destinations, selects dtype/ISA
networks, and prepares SME state-owning alternatives. C++ printing runs no
passes. `cpu_transpose_cache_block` (multiple of 16, default 128) and
`cpu_transpose_loop_order` (`"input"`/`"output"`, default `"input"`) are CPU
tuning overrides. Generic streaming multi-output fusion currently yields to
layout-aware transpose plans rather than bypassing their permutation DAGs.
`cpu_transpose_defer_conversion` (bool) overrides boundary-conversion motion;
the default cost policy keeps byte→FP64 in the compute domain for unknown
strides/extents and permutes narrow storage bits for constant layouts.
Matrix planning records input fusion/materialization decisions and prepares
packing permutations and dense-pack layout conversions. Empty matrix outputs
skip hardware resource/configuration entry; zero K still applies acc/epilogue.

### Architecture-specific state

`T.store`/`T.copy`/`T.transpose_copy` accept `implementation="auto"`, `"vector"`,
`"gather"`, or `"sme"`. Automatic SME selection requires SME-FA64, a simple
unmasked load (possibly converted), legal strides, and nonoverlapping operands.
Otherwise vector/gather paths preserve semantics, including alias snapshots.
Explicit SME is a compile-time target requirement, not a CPU runtime probe.

```python
with T.sme.region():
    T.sme.zero_za()
    T.transpose_copy(A, C, compute_dtype="float32", implementation="sme")

with T.amx.region(((16, 64),)):  # (rows, column bytes) per tile, immutable tuple
    T.amx.load(0, A)
    T.amx.store(0, C)
```

These are compiled lexical regions, not Python host context managers. Currently
SME bodies admit simple whole-view rank-2 transposes and `zero_za`; AMX bodies
admit tile zero/load/stream_load/store. Bare Python MMA instructions are not
exposed yet; `T.dot` owns its typed instruction schedule and resource region. Views are
validated before entry; allocation, throwing work, nested explicit regions and
values escaping a region are rejected. Explicit SME rejects overlapping operands
instead of allocating inside SM. AMX obtains per-thread Linux XTILEDATA permission
and owns configuration/release through `ExecutionSession`. Generated state-owner
callbacks always use `VECOPS_INLINE_LAMBDA_NOEXCEPT`.
Owned forwarded scratch is materialized before raw hardware memory operations;
pre-existing loads used after an AMX write receive the ordinary alias snapshot.
Multiple transposes in an explicit SME region must have independent input/output
storage; cross-operation alias dependencies are rejected before state entry.

SVE vectors/predicates do not cross SM boundaries: VL and SVL may differ. Sized
SVE Vec types can enter the SME path only with matching fixed VL/SVL guarantees;
automatic selection otherwise retains the vector path. VLA uses named sizeless
locals, never Vec/Mask arrays. Wide virtual vectors likewise emit individual
legalized variables. The public C++ surfaces are `vecops/vec/{SME,AMX}.h` and
`vecops/execution/Resources.h`; DSL sources include no backend `details` headers.

NumPy `ml_dtypes.bfloat16` storage is accepted by the native ABI using ndarray
metadata (it has no PEP-3118 buffer format). Endianness and writable output checks
are unchanged. Contiguous empty NumPy arrays retain canonical dense ABI strides.
See `benchmarks/dsl_transpose.py` for the exact 399-case TransposeBench catalog
comparison and `benchmarks/dsl_transpose_cross.py` for ARM object/disassembly
checks. Cross compilation does not establish ARM correctness or performance.

### Internal organization

Public imports remain under `vecops.language`: `compile.py` and `passes.py`
are thin façades, while `jit.py`, `template.py`, `expr.py`, `types.py`,
`dsl.py` and the small `library.py` retain their public roles.

Compiler implementation lives in `language/_compiler/`: `frontend/` owns
immutable AST/helper capture and parsing; `ir/` owns types, kernel/vector IR
and traversal; `passes/` owns the registry and generic transforms;
`lowering/` converts tile consumers to vector regions. `pipeline.py` is the
single builtin-registration entry point and assembles registry snapshots.
`driver.py` orchestrates compilation, with core-cache identity in `cache.py`.

`backends/cpu/` owns CPU policy, schema adaptation, emission planning and
`cpp/` source printing. Alias fallback, snapshot and axis-tail vector regions
are prepared before emission: printing C++ never runs lowering or compiler
passes. Unused alternatives are not included in the emitted-region diagnostics.
Other backends may choose different IR/codegen/build adapters; they are not
required to use the CPU vector-to-C++ pipeline. Only CPU is implemented today.
`language/_runtime/` owns runtime binding, strided output allocation and lazy
Torch bridge calls. These private package paths are not stable public APIs.

The CPU emitter lowers reductions to result locals and ordinary lexical blocks,
not capturing lambdas. This keeps invariant metadata/scalars visible to the C++
optimizer. Scalar conditionals and short-circuit predicates keep reductions
inside the selected branch; a reducing `while` condition is recomputed at the
loop head, including after `continue`. Vector tails and conditional cache reads
use explicitly typed Vec/Mask locals and `if`, including on sizeless SVE; common
normalization kernels need no closures. An expression-only context such as a
statement-producing `for` condition may still require a closure; every generated
closure uses `VECOPS_INLINE_LAMBDA` from `CoreDefs.h`.


## Calls and precompilation

Calling a compiled template binds metadata, resolves its specialization,
loads or compiles a cache miss, and invokes the source kernel:

```python
kernel = scale(32, 1024, 1024, vecops.float32)
kernel(input, output, 3.0)
```

Use `TensorMeta` to compile without allocating tensor storage:

```python
kernel.compile_for(
  vecops.TensorMeta((32, 1024), dtype=vecops.float32),
  vecops.TensorMeta(
    (32, 1024),
    dtype=vecops.float32,
    access=vecops.TensorAccess.output,
  ),
)
```

Storage is deliberately not required during binding or `compile_for`; it is
required and checked when an executable is actually invoked.

For a Torch model containing several registered vecops operators, use the
model-level collector instead of warming each operator serially:

```python
result = vecops.precompile(
  model,
  inputs,
  parallelism=8,
  use_real_tensors=False,
)
print(result.collected, result.elapsed_seconds)
```

The default trace runs inside `FakeTensorMode` and temporarily forces nested
`torch.compile` callables to eager mode. Model parameters, inputs, and tensors
created by Torch factory operations therefore carry metadata without backing
storage. Every registered vecops wrapper records its tensor shape, element
stride, dtype, optional presence, and explicit specialization values, then
returns the caller-allocated output arguments without invoking native code.
Exact duplicate requests are removed; runtime scalar payloads are deliberately
not part of the deduplication key because they do not specialize artifacts.
After tracing, `compile_batch()` performs cache lookup and binding first, then
places all misses sharing a toolchain into one generated CMake project. Each
specialization remains a separate target/subdirectory, but one Ninja or Make
jobserver schedules every translation unit. No Python compilation thread pool
is used. `parallelism` is the job limit passed to that shared build graph; the
default honors the process CPU affinity and is capped conservatively. Distinct
`Compiler` objects with identical SDK, toolchain, work-root, target, and
compile/link policy share a batch; incompatible configurations are submitted
as separate build graphs.

Set `use_real_tensors=True` when fake execution is unsupported. Ordinary Torch
operations then consume the supplied values, but vecops outputs remain
uninitialized because native kernels are still skipped. Data-dependent model
logic may consequently observe meaningless values. Models may query
`vecops.is_precompiling()` to bypass such validation or stateful paths.

`collect_compile_requests()` exposes the lower-level process-global collection
context, and its collector's `requests` may be passed directly to
`compile_batch()`. Nested or concurrent model collection is intentionally
rejected; compilation itself is concurrent and starts only after collection
has ended. A model that stores intermediate tensors in application-level
caches should clear or restore those caches after the tracing pass.

### Framework-neutral execution graph capture

Pass `capture_graph=True` to reuse the same FakeTensor or real-tensor model
trace for memory analysis:

```python
result = vecops.precompile(
  model,
  inputs,
  capture_graph=True,
  graph_name="encoder-bucket-512",
)
execution_graph = result.graph
execution_graph.write("encoder-bucket-512.json")
```

`ExecutionGraph` contains only plain Python dataclasses and JSON-compatible
values; it has no Torch dependency. Its ordered events record operator names,
stable logical scopes, formal argument access (`read`, `write`, or
`read_write`), and storage versions. A `TensorValue` represents one logical
view while `StorageValue` groups every alias of the same allocation. Finalized
logical lifetimes end at the last semantic access; storage lifetimes are the
union of all alias lifetimes and extend to graph exit for persistent or escaped
values.

The optional Torch adapter observes dispatcher operations, so allocations
implied by result-producing operations such as `add`, `mm`, `contiguous`, and
custom operators are visible even when there is no explicit `aten::empty`.
Real and FakeTensor storage identities provide the primary alias relation;
operator schemas provide a fallback for declared view returns. Consequently,
conditional cases such as `contiguous()` and `to()` are resolved against the
concrete precompile shape, stride, dtype, and device: a no-op result retains the
input identity, while an actual conversion records a new allocation. Graph input
and output lists preserve pytree leaf order and repeated slots, even when several
formal arguments or results refer to the same logical tensor. Named module
parameters and buffers are registered as persistent state before execution; tied
state retains every qualified name without duplicating its tensor or storage.
Vecops calls short-circuited by precompilation are emitted directly with their
KernelDef `In`/`Out`/`InOut` contracts.

Frameworks and model libraries may add stable hierarchy and information that
an operator dispatcher cannot infer:

```python
from vecops import graph

@graph.wrap_region(name="transformer.block")
def block(...):
  ...

loop = graph.declare_loop("transformer.layers", 48)
with loop.iteration(0, role="prologue"):
  ...
loop.close()

graph.mark_alias(view, base, kind="model-declared")
graph.mark_tensor(weight, persistent=True, read_only=True)
graph.mark_retained(cached_value, reason="decoder KV cache")
```

All annotations are no-ops outside an active capture. Loop declarations retain
the full trip count separately from observed iterations, allowing a model's
precompile path to sample cold and steady-state iterations without pretending
that the shortened trace is the complete execution. The capture layer does not
choose memory tiers or allocate physical arenas; those are consumers of the
serialized IR.

`mark_retained` is intended for Python-side caches that outlive a tensor's last
observed operator access. It extends the tensor and its complete alias storage
group to graph exit without classifying the value as a model output.

### Offline HBM/DDR planning

`vecops.planner` consumes a concrete `ExecutionGraph` without importing Torch.
The production entry point accepts an explicit machine profile and returns a
pointer-free, serializable plan:

```python
from vecops import planner

machine = planner.profile_from_system(
  memory_system,
  fast_capacity=3 << 30,
  slow_capacity=64 << 30,
  fast_bandwidth_mib_s=295_578,
  slow_bandwidth_mib_s=153_324,
  transfer_bandwidth_mib_s=148_797,
)
plan = planner.plan_memory(execution_graph, machine)
planner.validate_plan(execution_graph, plan)
plan.write("model.memory-plan.json")
```

The planner collapses tensor aliases to storage groups, scales access benefit
through sampled loop trip counts, selects HBM intervals under a hard capacity,
colors non-overlapping lifetimes into reusable arena offsets, and emits ordered
allocate/copy/free actions. Read-only external state has a DDR home and an HBM
replica; graph outputs remain in DDR. A plan embeds the exact graph digest and
is rejected if replayed with another trace.

The equivalent CLI is:

```sh
python -m vecops.planner model.graph.json \
  --output model.plan.json \
  --report model.plan-summary.json \
  --discover-system \
  --fast-capacity 3GiB \
  --slow-capacity 64GiB \
  --fast-bandwidth-mib-s 295578 \
  --slow-bandwidth-mib-s 153324 \
  --transfer-bandwidth-mib-s 148797
```

The explicit bandwidth values above are an example of measured target data.
When omitted, `profile_from_system()` retains topology-provided values. Use
measured values for production scoring because rated topology bandwidth can
substantially overstate the fast/slow ratio seen by the intended thread count.

`planner.PlanSession` binds the plan to `vecops.memory` arenas. It can
materialize strided Torch views with `tensor_view()` and temporarily replace
selected read-only module parameters/buffers with HBM replicas through
`bind_module_state()`. Session exit restores the original module state and
releases both arenas, including exceptional exits.

Framework wrappers may call `planner.empty()` and `planner.empty_like()` as
drop-in factory functions. With factory replay enabled, `PlanSession` allocates
the complete planned HBM peak once and returns prebuilt views at fixed offsets;
without a session they directly forward to Torch. Sampled steady-state loops
are replayed by runtime epochs, and temporaries whose aliases escape an
iteration are excluded by `PlannerConfig(require_replayable_temporaries=True)`.

`planner.audit_plan(graph, plan)` reports coverage for parameters, buffers, and
packed buffers, including accessed bytes and DDR-to-HBM copy actions. Its
`movement` section also reports duplicate copies, low-reuse copies, and
HBM-to-DDR copy-backs. Read-only state remains a clean DDR-backed replica and
never receives an unnecessary HBM-to-DDR writeback.

## Framework registration

Compiled templates select NumPy execution or a generated Torch bridge from
the runtime tensor arguments. Torch bridge tasks are content-addressed and
compiled on the first real call or at the end of `vecops.precompile()`.
Use `T.compile_pending(parallelism=N)` to materialize all declared bridges
in a shared build graph before execution.

The generated Torch schema marks output/inout tensors mutable and returns
`()`. The Python callable returns the original output objects. CPU and Meta
implementations support fake tracing and model compilation.
`CompiledKernel.explicit_torch_function()`
exposes an explicit-output callable with positional specialization values for
hot paths. After an invocation or precompile has prepared its bridge,
`explicit_torch_binding()` returns the raw Torch operator and immutable bound
specialization tuple for graph capture.

At load time vecops supplies the process-local operator handle and current
`_C` ABI entry-point address. The bridge has no link-time dependency on a
particular `_C` path. `VECOPS_TORCH_BRIDGE_DIR` may place its registration cache
separately from the kernel artifact root selected by `VECOPS_CACHE_DIR`.

Generated bridges are ordinary shared libraries rather than Python extension
modules. They expose a small C setter for the process-local operator handle, so
compilation does not include `torch/extension.h`, pybind11, Python headers, or
`torch_python`. CMake obtains Torch include/library paths and the libstdc++ ABI
mode from the active Torch installation. Loading the DSO runs its
`TORCH_LIBRARY_FRAGMENT` initializers before vecops supplies the handle. A
cache-root file lock coalesces bridge writers across processes; the cache is
rechecked after acquiring that lock.

For workloads dominated by many short vecops parallel regions, set
`VECOPS_THREAD_POOL=native` to route generated bridges through vecops' single
process-local persistent pool instead of `at::parallel_for`. Its size defaults
to `torch.get_num_threads()` and may be fixed with `VECOPS_NUM_THREADS`.
On Linux, `VECOPS_THREAD_POOL_CPU_LIST` may provide an explicit comma/range CPU
list for worker pinning; this is useful when the submitting OpenMP thread is
already pinned to one CPU but the native pool should span the rank's full mask.
`VECOPS_THREAD_POOL_SPIN_COUNT` sets the bounded worker spin before parking.
`VECOPS_NATIVE_THREAD_POOL_OPS` optionally limits native-pool injection to
comma-separated qualified operator globs. The default remains
`VECOPS_THREAD_POOL=torch`.

Generated Torch bridges build their `VecopsValue` and specialization arrays in
fixed-size stack storage. Tensor size and stride pointers refer directly to
the synchronous ATen arguments instead of being copied into temporary vectors.
The native extension also caches exact, successfully resolved invocations per
thread, so steady-state calls bypass handle-registry locking, schema binding,
provider lookup, and workspace revalidation. Calls with different metadata or
explicit specializations automatically take the slow path and populate a new
entry. `VECOPS_TORCH_PREPARED_CALL=0` disables this optimization for A/B tests.

NumPy arrays use the compiled template directly, with caller-owned or
automatically allocated outputs according to `out_idx`.

## Heterogeneous CPU memory

When the optional `vecops::memory` component is built, Python exports
`vecops.memory.System`, owning buffers, NumPy factories, topology/tiers/stats,
and a bounded workspace session. The memory module is absent when
`VECOPS_BUILD_MEMORY=OFF`; importing ordinary vecops APIs remains unchanged.

```python
system = vecops.memory.System(
  target_overrides=[
    {"os_numa_id": 16, "kind": "hbm", "max_managed_bytes": 3 << 30},
  ],
)

with system.workspace_session(fast_capacity=3 << 30, slow_capacity=8 << 30):
  result = model(inputs)
```

The single-use session affects registered Python operators and framework
bridges on the calling thread. It freezes the CPU domain, owns one shared arena
per tier, and releases both on exit. It does not replace the process allocator.
See `docs/Memory.md` for placement and lifetime semantics.
