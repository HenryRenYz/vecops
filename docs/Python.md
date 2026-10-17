<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Python JIT API

The public Python API is implemented above the low-level `vecops._C` pybind11
module. Applications use `vecops.language` for kernel templates and the
classes and functions exported by `vecops` for runtime services; `_C` is an
expert/debugging escape hatch and mirrors C++ types
without Python convenience behavior.

Importing `vecops` does not import NumPy or Torch. Dtype conversion recognizes
those frameworks only when the application has already loaded them. Canonical
string names (e.g. `"float32"`, `"bfloat16"`, `"bool"`/`"bool_"`) are recognized
without either framework and use the same normalization in compilation, JIT
arguments and automatic output allocation. In particular, a BF16 string
annotation does not require a NumPy BF16 dtype registration. Torch allocation is
supported directly; NumPy automatic BF16 output allocation requires an extension
dtype already registered by the application (e.g. `import ml_dtypes`). No optional
extension is imported implicitly. `None` remains an unspecified schema dtype and
is rejected as a concrete dtype, rather than being interpreted as NumPy's FP64
default. Non-native-endian NumPy dtypes are rejected consistently by metadata
normalization and native tensor views.

Python dtype names, byte widths and allocation adapters are maintained centrally
in `vecops._dtype`; DSL dtype wrappers use `__vecops_dtype__` to expose their native
enum without a dependency from runtime normalization to the DSL frontend.
Compilation, JIT argument keys, automatic outputs and debug metadata all use this
normalization. Torch unsigned 16/32/64-bit tensors are supported when provided by
the installed Torch version, including the generated Torch-to-runtime ABI bridge.
C++ type spelling and ABI mappings remain backend/boundary-specific.
The memory NumPy allocator and memory-plan Torch adapter use the same allocation
adapter, explicitly allowing framework-only storage dtypes such as complex.
Allocating that storage does not extend the supported kernel ABI dtype set.

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

`T.jit(..., pipeline=..., pass_options=...)` forwards the compilation pipeline
and pass policy to `T.compile` for returned templates, including scheduling
cost-profile overrides. Plain Python return values/forwarders remain uncompiled.
The option mapping is copied when the decorator is created; option values and
pipeline objects must remain immutable for that decorator's lifetime.

`T.Const` arithmetic builds exact integer expressions. `T.dynamic()` creates
a bounded, aligned runtime value which does not specialize the artifact;
expressions combining named dynamic and Const leaves validate shape/stride
relations at runtime. A comparison involving an unresolved dynamic value is
not proven and is false in Python control flow, while `T.is_value(x)` reports
whether `x` is compile-time known. `T.any` is the unconstrained dynamic value.
Anonymous dynamic and dtype leaves passed through a `TensorType`-only helper
factory receive deterministic internal names before IR/schema capture. Distinct
dynamic objects remain independent; repeated references to one object enforce
the same runtime dimension, including derived dense-stride expressions. These
internal names are not a public API and do not depend on allocation addresses.

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
exports every referenced name when multiple `T.Const`/`T.DType` variables alias
the same value (for example, `NativeA = NativeB = T.DType("float16")`). These
names still reference the original value: shared annotation identities and
delayed batch captures are preserved rather than copied or rebound. Source-less
DSL templates keep their existing Python capture environment. `T.compile`
caches the lowered operator by template IR, source/dependency content,
`out_idx`, and compiler configuration. `T.jit` adds a higher-level elaboration
cache and only calls `T.compile` when the elaborated result is a
`TemplateFunction`; ordinary Python callables and forwarding wrappers are
cached and returned unchanged.

`Optional[T.Tensor(...)]` and `T.Tensor(...) | None` both mark an optional
input. Tensor descriptors have a stable hash for Python's typing caches,
including older supported Python versions; binding symbolic names does not
invalidate that hash. The hash is not artifact identity: template/compilation
cache keys still use canonical dimensions, dtype, access and capture values.

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
typed scalar/tile/vector loop-carried variables, `break`/`continue`, assertions
and source-located errors are supported. New scalar/tile values (including
nested tuples) defined on both dynamic branches are joined automatically;
names defined on only one branch cannot be used afterward. Tile/vector state
keeps its dtype, predicate kind, logical shape and explicit vector tag.
Dynamic shape equality is checked at runtime. Views cannot be rebound across
loops; use explicit `T.alloc` storage instead. An unrolled region cannot use
`break`/`continue`. Kernels return through output tensors.

Literal scalar locals and pure constant `T.func` results retain compile-time
identity, so they can feed `T.static_assert` and specialization branches.
Immutable captured Python `bool`/`int`/`float` values, tuple elements and
helper defaults use the same scalar IR literals when consumed as expressions;
ordinary constants such as a captured `BM = 32` do not require `T.Const(32)`.
If a runtime loop or branch updates such a local, it is promoted to mutable
scalar state before that region. Immutable tuples support compile-time
slicing and starred tuple expansion; runtime tuple slice bounds and mutable
iterable unpacking are not supported.

Weak-literal metadata controls promotion; it is not a distinct storage dtype.
Boolean literal `flush` values (including the omitted default) remain valid
for printing. Loop-returning helpers use strong scalar result slots and allow
weak literals and explicitly typed values of the same dtype on different
return paths; different dtypes still require an explicit cast.

Python numeric literals in DSL expressions are **weak**: they do not widen an
existing floating tile/vector. Thus `float32_tile * 0.5` computes in FP32,
including intermediate rounding, rather than implicitly computing in FP64.
An integer or boolean value combined with a weak Python float computes in FP32
when the float is zero, NaN, infinity, or its finite nonzero magnitude lies in
`[2**-149, (2-2**-23)*2**127]`; otherwise it computes in FP64. This is a
magnitude-range rule, not an exact-representation guarantee for every Python
float. In particular, `int32_tile * 0.5 + 0.125` no longer truncates the
constants to integers before arithmetic. Output stores still convert to the
destination dtype after the floating computation.

Explicit `T.cast`, typed runtime scalar parameters and compiler-created typed
literals are **strong** operands. Their floating dtype participates in the
common computation type: FP64 dominates, FP32 dominates small-floats, and
mixed FP16/BF16 computes in FP32. Same-type half tile/vector arithmetic retains
that dtype; scalar FP16/BF16 arithmetic retains the existing FP32 computation
policy, with explicit casts/stores providing the rounding boundaries. A
literal promoted to mutable control-flow state becomes strong. Scalar-only
expressions containing only untyped Python literals keep Python's existing
double-precision constant-folding behavior. Constant folding preserves strong
types and rounds typed literals; small-float literals follow the public scalar
constructors' FP64-to-FP32-to-FP16/BF16 chain.

The shared rules also apply to `T.where`, `T.fma`, elementwise `min`/`max`, and
comparisons. A predicate retains its numeric lane basis, not a boolean vector
tag. This does not change integer/integer width or signedness selection,
integer division semantics, or the distinction between separate `mul`/`add`
and explicitly fused `T.fma`.

Tile values retain snapshot semantics: `old = x; x = x + 1` leaves `old`
unchanged, and `x, y = y, x` reads both old values before updating either.
The CPU state-placement pass keeps small fixed tiles and bounded scalable
vectors in independent Vec/Mask locals (also in sizeless SVE); large/dynamic
tiles use private scalar scratch with staged simultaneous updates. No arrays
of Vec/Mask objects are generated. This is a conservative placement policy,
not a global register allocator or a universal no-spill guarantee.

`T.func` helpers are source-parsed and inline by default, including their immutable
captures and inline policy in cache identity. `@T.func(inline=False)` retains
a private non-inlined call for each kernel specialization and call site. No
public symbol or cross-kernel helper cache is created. Scalar/tile/tuple results
use caller-owned slots; tile states crossing the call use scalar storage, so
sizeless SVE values never become closure members. Locally declared states can
still use registers. Calls may contain hardware regions, but cannot cross an
already-active AMX/SME instruction region. Views and compile-time descriptors
are not return values of private calls: use output arguments, or retain the
default inline helper for descriptor construction.
The CPU backend plans private-call captures after storage and hardware planning.
Entry tensor data-pointer snapshots are materialized only when these plans
need them for value capture; ordinary inline accesses retain `.data()`.
Eligible invocation-invariant tensor pointer/metadata snapshots and lexically
available, singly defined scalar bindings are copied by value. Pointer copies
retain the original pointee access and alias semantics. Mutable return slots,
loop induction variables, tile states, resizable storage owners and unknown
bindings remain by reference. The plan does not change private-call boundaries,
evaluate expressions earlier, or copy scratch pointers before a resize.
For each referenced ABI tensor root, the plan includes its existing size/stride
snapshots as well as logical-view metadata: alias guards may inspect the whole
physical root. Historical-read identities also select any already-established
snapshot condition for value capture after region protection. These conditions
are immutable booleans; snapshot Buffer owners and their live data pointers
retain their existing reference and lifetime semantics.
The tested BiSheng Clang 19 toolchain crashes while optimizing the large
outlined SME cache schedule, so that library schedule remains inline. Private
calls are currently used by the smaller ordinary-SVE PackedDot leaf; moving an
entire hardware schedule behind a call requires separate target validation.
Conditional scalar/tile/tuple returns and early
returns from nested loops are supported. A helper return exits only the
helper, not its caller kernel. Inline loop returns lower to lexical exit regions;
private calls keep the exit continuation inside their own closure. A value helper that reaches its end without returning
raises a runtime error. Result shapes must be available before the result
slot is declared. Arbitrary host calls, mutable captures, exception
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
`T.reshape` requires a dense memory view; for computation values it performs a
logical reshape and preserves element count. Dense read/elementwise DAGs fold
back to dense views; general value reshapes have a register-only lane fallback.
`T.transpose(value, axes=...)` and `value.T` transform computation values too.
`T.expand_dims(view_or_value, axis=-1)` inserts a singleton axis;
`T.squeeze(view_or_value, axis=-1)` removes one, checking its extent at runtime
when necessary. Memory transforms remain zero-copy and value transforms retain
their logical coordinates when broadcast into higher-rank consumers.
Compile-time tuple parameters such as transpose axes and reshape extents may be
immutable tuples captured from the surrounding Python factory or written as
literals in the kernel. Their scalar leaves are normalized at the DSL call
boundary; mutable lists remain unsupported.
Shapes broadcast
statically where possible; unknown relations are checked at runtime. A source
extent must equal the destination extent or be one; incompatible extents error.

### Register primitives and explicit indexing

`T.shuf(value, indices)` follows the public vec API's **native-word-local**
shuffle contract, not a global permutation of an arbitrarily large virtual
vector. Indices must address lanes in their native word. `T.local_shuf(value,
(...))` applies a compile-time lane order independently in each 16-byte block.
The order contains one in-range entry per block lane; repetitions are legal.
Neither primitive generates Vec/Mask arrays. `T.element_at(value, (i,...))`
extracts a scalar using valid caller-provided logical coordinates; the current
word-local shuffle DAGs are evaluated in their original layout before a scalar
or differently laid-out consumer accesses them.

`T.gather(view, indices, dtype=None, mask=None, other=0)` loads a rank-one
memory view using an int32/int64 index tile (which may have multiple axes).
For a higher-rank memory view, pass a tuple containing one logical coordinate
per memory axis. Coordinates can be scalar or broadcastable index tiles;
at least one must be a tile. `T.indices(shape, axis=-1)` produces an int64
coordinate grid along one compile-time axis, without storing a dense grid:

```python
rows = T.expand_dims(T.load(Idx), 1)
columns = T.indices((Idx.shape[0], A.shape[1]), axis=1)
selected = T.gather(A, (rows, columns), dtype="float32", mask=rows >= 0)
```

Indices are in **logical elements**, including for negative/padded strides.
Active indices are bounds checked; inactive indices may be invalid and never
form load addresses. Conversion is fused into `load_convert`. Gather values
retain the usual alias/snapshot protection.

`T.scatter(view, indices, value, mask=None, unique=False)` writes through the
same indexing convention to a rank-one view (coordinate-tuple scatter is not
yet implemented). Default duplicate indices have ordered logical
last-writer-wins semantics; `unique=True` is a caller promise permitting
native indexed `store_convert`. Both paths use the same default saturating
conversion as `T.store`: float-to-integer truncates toward zero, clamps at
both endpoints, and maps NaN to zero. A failed bounds check may leave earlier
chunks written; neither operation promises transactional error handling.
Arbitrary indexed grid writes cannot currently prove program ownership:
automatic grids retain serial order and `parallel="require"` rejects them.

`T.mask_all/any/none/count/first/last(predicate, unroll=2)` query all logical
lanes, excluding physical tail padding. First/last are flattened logical
indices, or -1 if absent; empty all/none are true, any is false, count is zero.
These use native mask operations; numeric reductions keep separate semantics.

### Local loops, general folds and resident collectives

`T.parallel(*extents, layout=T.ExecutionLayout(...), unroll=2)` describes
tile-local iteration, **not** a thread-pool grid. `order` is an outer-to-inner
axis permutation. An optional `vector_axis` must be innermost and yields a
tagged index vector. The current CPU fallback orders scalar axes and emits
explicit SIMD for the vector axis. `T.simd(stop, vector=T.Vector(...))` or
`T.simd(start, stop, ...)` yields clipped index vectors, including the tail:

```python
for index in T.simd(N, vector=T.Vector("float32", N=16)):
  value = T.gather(A, index, dtype="float32")
  T.scatter(C, index, value * scale + bias, unique=True)
```

Local parallel/SIMD loops do not support break/continue. `ExecutionLayout`
does not alter memory strides, cache blocking or worker/warp placement.

`T.reduce(value, reducer="add"/"min"/"max", axis=None, unroll=2)` retains
the builtin vector reduction paths. A custom `@T.reducer(identity=...,
associative=False)` is a pure binary scalar/tuple combiner, accepting two
matching values and returning a matching scalar/tuple. Inputs may be tuples
of equally shaped tiles, for example `(value, logical_index)` for argmax.
Identities are cast to the corresponding input dtype. Ordered custom folds
preserve left-fold order. With `associative=True`, a pure combiner consisting
of independent add/min/max components and their exact neutral identities
uses the same logical FoldLoop and SIMD lowering as builtin reductions.
Coupled tuple state (e.g. argmax), non-neutral initial values and unrecognized
combiners retain ordered scalar loops. `T.scan(value,
reducer="add", axis=-1, exclusive=False, reverse=False)` supports builtin or
custom scalar/tuple combiners and uses ordered CPU loops plus private scalar
output storage. It is not yet a specialized SIMD prefix-scan kernel.

Builtin reductions, including reductions nested in scalar arithmetic, are
normalized to explicit logical fold regions **without scalarizing SIMD**.
Independent equal-domain folds and axis folds share source computation and
iteration (`max_fused_reductions=2` by default). Dependent normalization phases
remain separate; scalar epilogues can be safely deferred but not recomputed
across writes. Adjacent local loops use a shared domain/state/effects analysis:
only pointwise or independent accesses may fuse. Runtime root-span disjointness
and physical non-overlap guards preserve the original program for aliases.
Unknown effects, mutable bound dependencies, helper exits and synchronization
remain barriers. Prefix consumers may fuse at the scan's actual write site,
including exclusive scans, eliminating private intermediate storage on the
non-alias path. `pass_options={"loop_fusion": False}` disables these generic
loop/consumer transforms.

`T.simd(..., unroll=2)` supports fixed and scalable extents. Proven clipped
arange accesses become normal converting memory operations; arbitrary indices
keep gather/scatter and their bounds checks. Full operation-major SIMD groups
use unmasked constant-width chunks and a separately clipped tail. Lowering
does not create another strip-mining loop around a proven single scalable
chunk. Live vectors remain individually named, never C++ arrays.

```python
shared = T.shared((workers,), dtype="float32")
with T.team(workers=workers) as worker:
  shared[worker] = local_value
  T.barrier()
  total = T.team_reduce(T.load(shared[worker]), op="add")
  C[worker] = total
```

`T.team` invokes one callback per **resident** worker, respecting executor
capacity and execution budgets. The requested count is a maximum: size shared
storage for it, and partition work using `T.worker_count()`/`T.worker_id()`.
`workers=0` selects the execution budget. Missing resident-team support errors
for multiple workers; a single worker is synchronous. Teams cannot nest inside
loops/grids/other teams. Barrier and team_reduce currently require team-top-level
control (including inline helpers at that level); unproven branch/loop
uniformity is rejected. Team bodies cannot return/break/continue or mutate an
enclosing scalar/tile state. Ordinary owned allocations inside a team are
worker-private; declare shared storage before entry. Overlapping non-atomic
writes remain the caller's responsibility. Provider cancellation releases
barrier waiters when a worker fails.

`T.team_reduce(value, op="add"/"min"/"max")` returns the same scalar to every
worker, in worker-id fold order. Shared scalar slots are protected by two
barriers, not arrays of vector objects. `T.atomic_add/min/max/exchange` and
`T.atomic_compare_exchange(view, expected, desired)` operate on writable scalar
views of 32/64-bit integers or float32/float64 and return the observed old
value. `order` is relaxed (default), acq_rel or seq_cst. Required alignment is
checked; min/max use ordinary comparisons, preserving a stored NaN and
ignoring a NaN candidate. Concurrent accesses to an atomic location must all
be atomic or separated by a team barrier.

### Replayable workspace and composable attention/math

`T.workspace(shape, dtype="float32")` declares an owned, dense scratch site
before conditional/loop/grid/team/ISA regions. The generated kernel takes the
public C++ `WorkspaceContext&`; the existing source adapter plans and replays
its allocation. Repeated calls can reuse allocation, **not tensor contents**.
Initialize every element before reading it. Unlike `T.alloc`, workspace is
not automatically forwarded into a delayed computation value.

Workspace extents may use constants, tensor metadata and
`T.worker_capacity()` (the call's effective executor/budget capacity). Arbitrary
runtime scalar payloads or tensor data are rejected as extents: they are not
part of the existing replay key. Allocation sites must be unconditional and
serial. Metadata-derived locals must remain immutable; updates, including
updates inside preceding branches/loops, invalidate the extent proof.
Immutable shape tuples support compile-time slicing and starred tuple
expansion (not arbitrary runtime iterable unpacking).
Shape/stride specialization and effective parallelism changes select
a different plan; no ABI version or new compiler dependency is needed.

For manual persistent worker scratch, allocate
`T.workspace((T.worker_capacity(), *tile_shape), dtype=...)` before `T.team`,
then slice its leading axis by the resident worker id. This is dense storage;
pad a worker's row explicitly if cache-line separation is needed. It does not
automatically add the C++ prepared-operator scratch packing, phase-lifetime
coloring or cache-line padding. Owned `T.alloc` inside a grid remains private
per worker/invocation, not automatically persistent across calls.

Visible `T.library` helpers now include:

- `stable_sigmoid(x, accuracy="fast", algorithm="exp")` and `silu`: clamp
  magnitude to 80 and call native `T.exp_neg` at the requested accuracy.
  `algorithm="polynomial"` remains a compatibility alias for the same native
  path; no polynomial or squaring schedule lives in the DSL. Small-float inputs
  compute in float32. Accuracy values are strict, fast or estimate.
- `rmsnorm(x, eps=1e-5, axis=-1, unroll=2)`: float32 accumulation for fp16/bf16.
- `masked_softmax(x, mask=None, axis=-1, accuracy="fast", unroll=2)`:
  nonzero numeric/predicate masks; finite valid logits; fully masked/empty
  rows give zero. Rank-two last-axis RMSNorm and masked softmax hoist row
  statistics and preserve their elementwise tails for dot/packing/store
  fusion rather than recomputing them inside microtiles.
- `scaled_dot_product_attention(Q, K, V, output, block_k=64, ...)`: rank-two
  memory views with K `[keys,head_dim]`, V `[keys,value_dim]`, optional scalar
  `scale`, matrix `bias`, query/key masks and causal offsets. It streams K/V,
  updates max/denominator/accumulator as typed loop state, stores output and
  returns its value for a subsequent epilogue. The caller chooses query tiles,
  heads/batch grids and scratch layouts.

Attention uses finite valid logits and returns zero for fully masked or
zero-key rows. Optional tensors are selected as `None` in the outer factory,
not nullable runtime DSL parameters. `acc_dtype` defaults to float32;
Q/K/V/output storage must be floating-point; the accumulator must be float32
or float64. Default scaling requires a positive head dimension.
`probability_dtype` defaults to V.dtype, allowing BF16/FP16 dot backends but
rounding probabilities. Explicit float32 avoids that rounding (and can force
a vector rather than matrix-instruction family). `accuracy` and
`implementation` are forwarded to visible math/dot operations. One streamed
QK block is materialized with scale/bias fused into its epilogue, not a full
Q-by-K matrix. This is a portable composition implementation; it does not
promise parity with C++ Attention's prepared packing/workspace/strategy
scheduler, indexed/sparse fast paths or every possible state-fusion schedule.

### CPU software pipeline and architecture operands

`T.prefetch(view, level="L1"/"L2", intent="read"/"write", temporal=True)`
is a synchronous CPU hint for the first address of a nonempty view, not an
asynchronous copy. `T.pipelined(..., stages=2, distance=1, unroll=2)` uses
serial range syntax and a registered planning pass: eligible straight-line
reads generate guarded lookahead prefetches. Control-dependent reads are not
speculated; with buffering disabled, runtime data is not moved across stores.
The hint does not change numerical iteration order. The current pipeline
keeps an ordered loop instead of operation-major unrolling.

Opt-in `buffering=True` (with distance=1) automatically primes and rotates two
banks for eligible straight-line, invariant-extent tile loads. A root-level
runtime no-alias guard selects this schedule; aliased inputs/outputs retain
the original ordered loop. Control/state/reduction-dependent loops and
varying-width tails fall back to prefetch-only scheduling, with diagnostics.
Buffering is off by default because extra scalar-storage traffic can cost
more than it saves on inexpensive kernels.

`T.double_buffer(shape, dtype=...)` allocates two scalar-storage banks, selected
as `banks[i % 2, ...]`. Shape accepts a scalar extent or a tuple, including
captured tuples and parsed helper results, just like `T.alloc`/`T.shared`;
shape expressions are evaluated once during frontend expansion. Prime the
first bank and explicitly load the next bank
before consuming the current bank. The compiler does not yet infer arbitrary
double-buffer schedules or asynchronous events; GPU/NPU async semantics are
not advertised by these CPU operations.

Inside `T.amx.region`, zero/load/stream_load return typed, region-local tile
handles. `T.amx.tdp(acc, A, B)` validates dtype, distinct registers and tile
configuration, then emits the public AMX dot instruction primitive. AMX
memory inputs must already have the required packed instruction layout.
Inside `T.sme.region`, `T.sme.load(view)` creates a clipped single streaming
word, `T.sme.mopa(tile, A, B)` selects the instruction by operand dtype, and
`T.sme.store_tile(tile, C)` stores one clipped accumulator tile. All checks
occur before streaming entry; handles cannot escape the region. Inputs need
unit stride and must fit SVL; low-dtype grouped-K packing is caller-managed.
Architecture feature/state ownership remains in the architecture packages,
not in portable T.dot or the top-level execution layout.

Matmul uses a backend schedule IR before C++ emission: NC/KC/MC, native spatial
loops, K recurrence, pack-A/pack-B and accumulator init/finalization are visible
in `compiled.backend_ir.matmuls[id].schedule`. This inspection interface is
internal and may evolve. It retains GEMV unrolling, traversal selection,
packing provenance and first/last-KC fusion semantics.
`pass_options={"cpu_matmul_pipeline": "off"/"prefetch"/"staged"/"auto"}`
selects the CPU schedule. Prefetch guards future addresses before forming them;
distance and hints per AMX operand are controlled by
`cpu_matmul_prefetch_distance` (1..16) and `cpu_matmul_prefetch_lines` (1..16).
Staged vector/SME kernels prime, rotate and drain two **register** operand banks
without changing MAC order. AMX's common 2x2 kernel occupies all eight tile
registers, so its staged path uses panel/cursor prefetch instead of a ninth
tile or a new scratch copy. CPU staging is not asynchronous packing. `auto`
currently selects `off` via the centralized target cost policy; small measured
gains do not justify enabling it on every shape/processor.
The option also adds bounded hints to the existing GEMV K-bank traversal;
it does not change GEMV's numerical order or allocate additional AMX tiles.
Logical custom fold/scan loops retain their reducer descriptors and carried
state; an associative promise alone never removes scan's sequential ordering.

### CPU grids and execution budgets

`for i, j in T.grid(M, N, schedule="auto", chunk=1, parallel="auto")`
preserves logical programs in IR. The CPU ownership pass proves separable,
non-overlapping output partitions, retains root coordinates through ordinary
slices/transposes, and checks actual writable layouts and parameter aliases
at runtime. Unknown ownership, shared scalar state and explicit hardware effects
retain ordered execution. `parallel="require"` rejects missing proofs or failed
runtime guards; it requires safe parallel eligibility, not a minimum worker
count. `parallel="serial"` retains lexicographic program order.

The runtime selects active workers from task count, execution budget and the
central target cost model. Scalar reduction epilogues are charged once per
program, vector producers by native words, and dots include their MAC work.
Small grids may remain serial. This is a heuristic, not an optimal scheduler
or a guarantee that every bandwidth-bound shape scales. `schedule="auto"`
currently uses contiguous, balanced static assignment. `"dynamic"` claims
positive `chunk`-sized program batches from an overflow-safe atomic cursor;
it helps uneven work but is not a work-stealing deque or an indivisible-task
tail eliminator. Temporary allocations/materializations belong to the worker
and are reused across its programs. Nested grids do not open nested teams.

`pass_options={"cpu_grid_regions": ...}` accepts `"individual"`, `"adjacent"`,
`"template"` and `"auto"` (currently the template-safe policy). The planner
shares adjacent eligible grids, optionally spanning short pure scalar
assignments executed by worker zero. Grid metadata must remain available at
region entry. RAW/WAR/WAW root effects and runtime span aliases decide phase
barriers; proven independent phases omit them. Same-root dependencies remain
conservative, even when static same-owner reasoning could remove a barrier.
Unknown effects/control boundaries end the region. V1 logical-task-only
executors use individual synchronous submissions instead of team barriers.

```python
with vecops.cpu_thread_limit(8):  # direct CPU/NumPy calls; does not change Torch
  kernel(a, out)
```

Use process affinity or explicit budgets for multiple independent processes,
with multiprocessing spawn/forkserver. `T.grid` schedules programs on CPU
threads; it does not spawn processes or coordinate their CPU ownership.

### Matrix products and packing

Generated Python DSL kernels disable implicit floating-point contraction at
native compilation: distinct `mul` and `add` operations retain their separate
rounding, including when fused into packing or a Matmul input producer.
Use `T.fma(a, b, c)` for explicitly fused rounding. This target-only policy is
part of artifact identity; it does not mutate the user's `Compiler`, global
`CXXFLAGS`, or the compilation policy of `T.template(source=...)` C++ kernels.

`T.dot((A, B), output, acc=None, acc_dtype=None, implementation="auto")`
computes the rank-two tile product. A is `[M,K]`, B is **`[K,N]`** (unlike the
C++ Matmul operator's `[N,K]` convention). Inputs may be views, elementwise
tile DAGs or explicitly packed values. The result is also returned as a tile
expression for a fused final store. `T.dot(A, B, acc=None)` is the pure
returning form. Dot does not introduce cache blocking or persistent packing;
instruction-format staging is still necessary for raw AMX/SME inputs. Native
floating products follow the selected instruction's fused accumulation and
rounding semantics; bitwise equality with the vector widening path is not
promised. Use an explicit implementation when that distinction matters.

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

`T.widening_dot(a, b, dtype="float32")` groups rank-one computation values
with equal logical lengths. Statically incompatible lengths are rejected;
otherwise an unresolved equality is checked before evaluating the result.
BF16 inputs produce one FP32 result per
two input elements; signed/unsigned byte inputs accept `dtype="int32"` or
`"uint32"` and produce one result per four elements. The result length is the
ceiling of the input length divided by that group size. Missing elements of
the last group contribute zero, including after a fused input transform.
Byte products use modulo-2^32 output semantics. BF16 grouping follows the
selected native widening-dot instruction's rounding semantics; it need not
match a separately rounded FP32 multiply/reduction bit for bit. The CPU
compiler emits the public typed `vec::widening_dot` primitive, allowing the
C++ ISA backend to select BF16 dot/VNNI instructions or its portable fallback.
For a complete output vector with a constant input length divisible by the
group size, input evaluation retains the full-vector proof and omits redundant
load masks and zero blends. Partial groups still sanitize the final transformed
producer value before the dot instruction.

K-contiguous BF16 and byte inner-product leaves use `T.widening_dot` followed
by the ordinary DSL reduction. Automatic x86 AMX owners with unresolved
spatial sizes retain their original cast/multiply/reduce vector fallback:
expanding a different fallback changed GCC optimization of the entire owner
and regressed measured native tail paths. Static small GEMV, explicit vector
implementations and ordinary ARM inputs retain grouped widening. Dynamic
SME owners whose byte operands include wider tile producers use the compact
cast/multiply/reduce fallback too: expanding the converted groups caused
severe LLVM inlining time growth. This does not change explicit
`T.widening_dot` or pure BF16/byte GEMV. Row/column GEMV otherwise use vector FMA or
conversion/reduction leaves; matrix leaves use public
AMX or SME instructions when their dtype/target contract permits, otherwise
vector FMA. Accumulation defaults to int32 for 8-bit integer inputs, int64 for
16-bit integer inputs, float64
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
`T.pack` creates zero physical padding, matching public C++ MatmulPack output.
Native DSL matrix recipes ignore external slots beyond logical K by masking
or privately repairing the final instruction group; wrapping storage itself
does not repair padding. This is a stronger input guarantee than a C++ leaf
that requires packer-generated zero padding. Shape/capacity/alias validation precedes hardware-state
entry. A/B formats
are distinct: AMX A is row/K-blocked, B is dword-group/VNNI interleaved; SME
uses spatial panels and K groups parameterized by SVL. Dense B stores `[N,K]`.
Packing/layout algorithms are source-parsed `T.func` recipes. Dense B stores
`[N,K]` and uses ordinary transpose/vector memory planning. Interleaved formats
zero their physical padding before writing logical elements; padded positions
never evaluate an input DAG. Capacity and layout checks precede every access.
The physical packing ABI decoder remains a compiler primitive, not a hidden
packing loop. External packed buffers remain tied to their target ISA and
process SVL; changing a logical shape is not a format conversion.
Fresh packed operands created after online packing retain their native physical
panel binding. Borrowing raw storage is permitted only when no potentially
aliasing write follows its lexical capture; cross-region and historical
ordinary loads retain their logical value path.

`T.library.pack_b_compensated(source, destination, correction,
zero_point=..., format="amx"/"sme")` packs an already converted `[K,N]`
int8/uint8 value and writes an `[N]` int32 sidecar. The correction is
`-zero_point * sum(source, axis=0)` modulo 2**32. It preserves the exact
public packed ABI and zero padding, leaves excess destination capacity
untouched, and requires disjoint packed/sidecar outputs. Disjoint source
storage streams through 256-by-one-panel byte scratch; overlapping sources
are captured before either output is written. Consumers broadcast the
N-element sidecar as required; no full K-by-N int32 buffer is needed.


`T.dot` and `T.library.matmul` expand into ordinary DSL loops, mutable
accumulator state, conversions and stores before vector optimization.
Default `T.func` helpers inline before fusion analysis, including nested library
calls; `inline=False` retains a call region whose body is optimized separately
from its caller's adjacent statements.
Rank-two layernorm/softmax compute row statistics once; their final affine/exp
tail can remain in the operand DAG. Fusion estimates the complete DAG's cost,
not membership in an expensive-op list.
`pass_options={"cpu_matmul_fusion": "auto"/"always"/"never"}` overrides this
heuristic. Expensive producers materialize before cache traversal; conditional
caches are allocated and read only on their selected branch.

The library recipe owns BM/BN/BK cache loops, optional dense packing, one
initial accumulator seed per panel and one final epilogue. Explicit blocks
and optional dense-cache packing choices remain authoritative; required native
grouping/padding is independent of that choice. Automatic blocks query actual
L1/L2 capacity and may bypass packing/blocking when A, B and C fit conservative
private-L2 shares. Empty M/N is checked before cache queries and traversal.
L3 and LLC queries do not invent absent cache levels. Dot alone does not add
cache blocking; matrix-instruction leaves do perform their required physical
grouping and zero-padding. Column-contiguous library outputs may
transpose the complete fused DAG by swapping A/B.

`cpu_gemv_k_unroll` accepts 1, 2 or 4 (default 2) and controls source K-loop
unrolling. `cpu_matmul_pipeline` accepts auto/off/prefetch/staged; vector source
recipes use the generic `T.pipelined` planner. Matrix-instruction recipes use
their fixed register reuse and K-loop unroll schedules. A staged request that cannot
legally double-buffer a recurrence uses its diagnosed prefetch fallback.
There is no private matrix pipeline printer.

Matrix instruction recipes live in `_library/amx_matmul.py` and
`_library/sme_matmul.py`. AMX covers BF16/F16-to-F32 and all four I8-to-I32
signedness pairs. SME additionally covers F32/F64 and I16-to-I64/I32.
I16-to-I64 supports all sign pairs and requires SME_I16I64; I16-to-I32
requires matching signedness and SME2, and AMX F16 requires
AMX_FP16. Automatic selection retains a source vector fallback when the
compiled target lacks an instruction; explicit implementation requests reject
an unavailable family. Instruction support does not imply hardware availability. Automatic native
selection requires M and N at least 16. A known smaller extent prunes the
native recipe before helper expansion, even when the other extent is dynamic.
For an unconfigured vector matmul (`packing="auto"`, no BM/BN/BK), positive
constant dimensions with at most 256 output elements, K at most 256 and at
most 32 KiB of logical inputs go directly to the vector leaf. This avoids
expanding a cache traversal and issuing cache-size queries for that small
working set. Explicit blocks and packing choices retain their cache recipe;
explicit native requests retain native fringe support.

The compiler memoizes metadata membership queries within each shared DAG
traversal, including the definition-cycle context. Queries do not share facts
across changing definitions or loop scopes. Automatic recipe-level affine
simplification retains its constant-shape admission: extending it to Dynamic
native recipes reduced clipped expressions but regressed measured ARM raw-dot
paths. Runtime extents and residual checks therefore retain the existing
policy. Captured reads and checked reshape ownership keep their identity.

Prologue expressions are evaluated over logical inputs before physical padding;
accumulator/broadcast initialization precedes hardware compute, and output
conversion/epilogues follow it. Aligned tiles reuse operands in 2x2 register
families. AMX B packing uses the historical 16x16 dword register-transpose
schedule, preserving grouped-K payload bits. Legal AMX raw A and complete-K external packed B can be read directly;
same-type row-contiguous output can be written directly. Raw first-sink F32/F64
SME sources also retain signed-stride direct reads, including with an accumulator
and epilogue. Historical reads, aliasing, capacity and layout checks precede
the lexical instruction owner. Whole-catalog performance parity remains a
separate measured property, rather than an instruction-selection guarantee.

The old matrix/cache/packing C++ loop generators, specialized schedule IR,
SME automatic-packing policy objects and their experiment switches are gone.
`cpu_sme_packed_ab_cache_bypass`, `cpu_sme_narrow_singleton_slices`,
`cpu_sve_runtime_quant` and `cpu_gemv_columns` no longer select an algorithm.
Use the general source catalog and cache-block benchmarks for current
comparisons. Public C++ operators remain supported and provide unchanged
reference implementations; no generated DSL kernel calls `vecops::ops::Matmul`.

Alias snapshots preserve historical load identity, including writes crossing
helper boundaries, and retain a proven contiguous axis without retaining
padding/overlap. Shared, nested, masked and non-pointwise dot consumers receive
ordinary private intermediate values. Scalar tails and physical packing
conversions use the public saturating scalar conversion contract; same-dtype
copies preserve storage bits instead of routing FP16/BF16 through FP32.

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
`T.softmax(x, accuracy="strict" | "fast" | "estimate")` uses the requested
exponential accuracy; strict normalization computes dtype-typed `1 / sum`
with ordinary IEEE division, matching C++ Softmax's scalar inverse algorithm.
Fast/Estimate normalization retains the corresponding native reciprocal
tier. The default remains `"strict"` for both rank-one and rank-two tiles;
their scalar inverse is defined by the same visible DSL helper, including
rank-two masked normalization's safe denominator. This strengthens strict
inverse precision and does not change the explicit `T.rcp` primitive,
reduction order, or the default `unroll=2`. Every accuracy/compute dtype
uses native `vec::exp_neg`, including masked softmax. There is no separate
FP32 Estimate polynomial or -80 cutoff: the vec backend owns approximation,
NaN propagation and the documented storage-format underflow policy.
Scalar exponential, logarithm and reciprocal primitives use the same native
vec math dispatcher and requested accuracy as vector values. Scalar row
statistics broadcast into one native word and extract lane zero; no libm
fallback silently discards their accuracy option.

Unrolling copies each operation across iterations, not the complete body per
iteration. Seven adjacent vector chunks group as 4+2+1, with powers P+2/P+1/P.
Target `VEC_MAX_POW` and lane-preserving dtype rebind requirements limit merging;
the C++ preprocessing alternatives keep generated sources portable to SVE.
Oversized virtual vectors remain streamable, not forced into one native value.
Nonadjacent reduction-axis copies retain distinct lane domains. Loop-carried
scalar dependencies and break/continue retain ordered loops; alias-dependent
serial iterations have a runtime ordered fallback. Tails are always guarded.
Ordinary SVE/SME and x86 AVX traversals may split only their masked remainder into native-word
register fragments, accounting for wider load/store dtypes. Full-loop grouping
and the requested unroll remain unchanged. Reduction banks are reconstructed
before their original horizontal reduction, preserving their lane/order
contract; explicit SIMD and loop-carried vector extents retain their existing
lowering. Empty physical words then avoid grouped zero-mask operations.
This internal policy does not relax masks or conversion rounding.

Adjacent compatible, independent reductions with the same unroll count can
share a traversal (currently up to two reductions). An independent elementwise
side output can fuse into a reduction. A consumer needing the completed global
reduction remains a later phase. Adjacent elementwise stores can share a vector
traversal with runtime alias guards and a sequential fallback.

`T.layernorm` and `T.softmax` are visible rank-one/rank-two DSL algorithms:
centered variance and maximum-subtracted exponentials respectively. Rank two
normalizes each row. A sole row-separable `T.store` consumer can fuse its
row statistics and elementwise tail into one row loop, keeping mean/rstd
in scalar locals instead of writing and rereading two statistics buffers.
This compiler transformation requires full-region read/write non-alias
proofs and a non-overlapping full output layout. Overlapping/in-place
calls retain the complete original schedule and its conditional snapshots;
only checking aliases within the current row is not sufficient for shifted
matrix views. The `row_map_streaming=False` pass option disables
this fast path independently. Affine/post-ops can
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
The two-axis proof uses one compare/swap rather than a general-purpose sort;
negative strides, empty/singleton axes and span-overflow diagnostics retain
the same sorted-axis semantics.

The algorithms above are ordinary `T.func` row programs: rank one calls the
program directly, and rank two uses `T.map_rows(program, x, **options)`.
`map_rows` accepts a source-parsed, pure straight-line helper returning one
same-width numeric or predicate tile. Scalar options are specialized or
captured before the region. The generic compiler may stream a sole consumer
or save only row-varying scalar intermediates; it does not dispatch on helper
names or reconstruct LayerNorm/Softmax formulae. Stateful callbacks,
allocation and runtime control inside a row callback are currently rejected.
The historical `row_normalization_streaming` and
`row_normalization_release_captures` options remain aliases for
`row_map_streaming` and `row_map_release_captures`, respectively.

CPU lowering captures each ABI tensor parameter's dimensions and strides into
invocation-local const scalars at kernel entry. Data pointers are materialized
only when an existing private-call capture plan needs them by value; ordinary
inline accesses continue to use the tensor descriptor's `.data()`. Those
descriptors are invariant, but the pointed-to tensor contents are not:
serial iterations, aliases and atomics still observe the original memory
effects. This does not hoist dynamic loop state, view offsets, or addresses
of scratch buffers that can resize. The independent diagnostic pass option
`cpu_parameter_metadata_snapshot=False` retains direct descriptor access.
Dynamic axes use compile-time indexed size/stride access and convert the existing
`meta::Value` to an integer. This preserves its validated range/alignment contract
for the native compiler after IR constant propagation removes redundant guards.
Const axes keep their original access, which already folds at C++ compilation.
Only existing validated contracts are propagated; input validation is unchanged.

Independent elementwise stores can version their vector traversal once on the
current store's alias guards. On the disjoint fast branch, only conditional
snapshots freshly created for that exact store are released; a load captured
before an earlier write still reads its historical snapshot. The cold branch
retains the original traversal and every snapshot. Branch-local materialization
and vector caches cannot escape into subsequent stores. Reduction, state,
register-transform and opaque-effect boundaries retain their existing lowering.
`store_snapshot_regions=False` disables this internal versioning policy for
diagnostics without changing capture or alias semantics.

Ordinary SVE can give a closed, complete F32 pointwise output region a private
native lane permutation when all vector inputs and its output use compatible
F16/BF16 memory conversions. The compiler applies the same permutation to
both loads and stores; logical tensor order, individual-element rounding and
strict arithmetic remain unchanged. Uniform row statistics stay in their
original loop scope. Reduction banks, masked/tail accesses, indexed or
register-transform operations, historical/materialized values and explicit
SME hardware scopes retain their ordered paths. This is a CPU memory-planning
optimization, not permission to change public ordered conversion semantics.
The proof accepts only VectorRegion plans; hardware matmul plans in the shared
debug inventory stay outside it, including when that inventory is re-prepared.
`T.compile(..., pass_options={"cpu_pointwise_register_layout": True})` enables
the broad experimental policy. Actual same-node testing
found FP16 affine improvements but regressions for simpler FP16 and BF16
regions, so the legality proof alone does not authorize default selection.
`cpu_pointwise_register_layout_policy="auto"` is the narrow default
profitability policy. It uses centrally profiled ordered/packed conversion
costs and a conservative compute/register-pressure budget, and currently only
considers FP16 regions with at least three independent full-width paired loads.
Shared producers and scalar broadcasts do not count as independent pairs;
BF16, mixed storage and one/two-load regions retain their ordered paths.
`cpu_pointwise_register_layout_policy="off"` restores ordered lowering. An
explicit boolean `cpu_pointwise_register_layout` takes priority over the policy:
`False` disables it, while `True` forces the broad legal experiment. Costs are
heuristic relative cycles with provenance, not measured instruction latencies.
Selection follows the real code-generation ISA, not a
cost-profile override; see the [register-layout design](internal/arm-pointwise-register-layout.md).

Casting a captured load preserves that exact read's identity, including
through helper returns, branches and loop-carried values. A later cast must
not reread storage changed by an intervening aliased write; a distinct
`T.load` after that write still observes the new contents. Ordered intermediate
rounding is retained. The CPU emitter may fuse a single current-region
consumer into `load_convert` while retaining its snapshot selection; shared
or materialized reads instead reuse their already computed value. This is a
local fusion rule, not a promise of register residency across separate loops.

LayerNorm explicitly shares one reciprocal of the normalized-axis length
between its statistics; rank-two normalization computes it before its row
loop. This does not permit the compiler to reassociate arbitrary strict
division expressions or to change the reduction ordering. Rank-three/four
direct LayerNorm/Softmax calls are not yet supported and receive a located
`static_assert` diagnostic. `benchmarks/dsl_norm_library.py` measures the
actual rank-two library call against both a row-wise DSL wrapper and C++, with
separate `library_over_cpp` and `library_over_row` ratios; the row catalog
alone is not evidence for the direct library path's performance.

Multiple consumers, layout/index transforms, dot/packing consumers, and
effects between a tile capture and its store keep materialized statistics
as a fusion boundary. In particular, the row-store optimization does not
replace normalization-to-dot/packing schedules or recompute their reductions
inside a matrix microkernel. Statistics construction and the streaming
schedule share the same reduction/accuracy IR, not an opaque native operator.

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
The reusable `kernel-cleanup` and `kernel-loop` stages accept extension passes
with the same ordering rules. One registry snapshot is shared by all stages of
a CPU compilation, including the nested cleanup/loop groups.
These constraints order the default registry snapshot. Explicit pipelines are
ordered programs: they preserve the supplied order, permit repeated names and
do not implicitly restore omitted passes. `PassInvocation(name, options=...,
label=..., stage=...)` gives a call independent options and a diagnostic label;
`PassGroup(name, pipeline, stage=None, max_iterations=1)` nests a sequence and
can repeat it to a bounded fixed point. Every callback must report
`PassResult(..., changed=False)` for a round to converge; legacy callbacks are
conservatively treated as changed. Registry implementations remain uniquely
named and every nested callback is snapshotted before compilation begins.

The default CPU pipeline runs reusable cleanup after normalization, source
recipe expansion, loop optimization, serial unrolling and software-pipeline
materialization. Cleanup combines typed scalar/branch simplification, logical
view folding, memory normalization, CSE, private dead-store elimination and conservative DCE. It preserves
observable checks, historical load identities, explicit rounding and floating
reduction order. `cleanup_iterations` controls the maximum rounds (default 4,
range 1..16). Pure integer metadata can be hoisted from loops; a bounded
single-branch loop can be unswitched when its condition and bounds are safe to
speculate. `loop_unswitch_budget` bounds that transformation's owned-node count
(default 32; zero disables it).
Signed arithmetic moves onto previously empty paths only when its full typed
interval proves that it cannot overflow.
Index generators retain their coordinate maps through slicing, transpose and
reshape; row epilogues explicitly map their axes and origins. They cannot be
retagged like coordinate-independent elementwise arithmetic.
Reshapes that only insert or remove unit axes normalize to squeeze/expand-dims
projections. This preserves the original read identity, including historical
captures and noncontiguous inputs, while allowing ordinary vector memory access.
Private dead-store elimination requires an exact adjacent overwrite, harmless
initial evaluation/address metadata and no observation of the old private
contents. Public stores, masks, captures, fences and unknown effects retain
their ordering. `forward-private-stores` additionally removes an adjacent,
single-consumer private store/load round trip when its complete dense tile,
element type and pointwise evaluation are identical. The replacement retains
the original input read objects and therefore their normal output-alias
protection. Conversion, masks, earlier historical captures, metadata uses,
other consumers and intervening control/effects prevent forwarding.
`private_store_forwarding=False` disables this cleanup.
CSE treats mutable scalar versions and induction variables as
identity-sensitive dependencies, in addition to historical loads/tile state.

Representation verifiers run at kernel, vector and prepared-CPU boundaries.
Generic kernel/vector verifiers live with the IR vocabulary; CPU storage and
capture-plan verification lives in the CPU backend. The IR layer does not import
backend capabilities or execute lowering passes.
Terminal checks reject executable opaque dot/packing operations. Lazy
`BindTile` descriptors retain historical load identities and can describe an
original product after its oriented consumer has been lowered; they do not
evaluate that product. Their structural/type invariants are still checked.
`verify_ir=False` disables these diagnostic checks, not frontend semantics or
hardware preflight. Unknown operations conservatively may trap and cannot be
speculated. An operation contract must explicitly declare `may_trap=False`
and `speculatable=True` before motion can rely on that promise; effects='none'
alone is insufficient. Pass reports record each invocation, nested group round
and reported change separately, including repeated callbacks.

Shared integer facts combine validated extent bounds/alignment with typed DAG
intervals and congruences. Arithmetic facts are discarded when the complete
interval can overflow its result dtype. Proven predicates or remainders are
folded only when their evaluation is independently safe to discard; an actual
load or shape check cannot disappear merely because its numerical result is
known. Positive constant integer division/remainder has a separately checked
nontrapping contract. No floating reassociation is introduced.
Sequential scalar assignments and proven native-loop induction ranges refine
these facts within their lexical scope. Loop-carried writes, mutable bounds,
unknown effects and control transfers invalidate affected facts; loop and
branch facts do not escape their region. Rewrites retain historical read and
state identities. Single-definition lexical `int64`/boolean locals can also
retain a bounded symbolic equality when every operation is total and effect
free at both definition and use. Versioned transitive dependencies invalidate
recomputation after a write; a captured value's independent interval remains
valid. Missing equality lookups participate in memo invalidation. Expansion
shares expression DAGs and stops at 64 unique nodes or 64 version dependencies;
over-budget definitions retain their original bindings. Branch-local definitions
do not escape, and mutable values, hardware queries, reads, owned metadata,
unmodeled integer promotions and floating values are excluded.
Unknown `auto` locals do not inherit their nominal IR dtype's range: existing
C++ integer promotions can make their executable type wider. Unsigned scalar
rewrites and unmodeled signed/unsigned conversions are retained intact,
including their original operand bindings. This preserves the existing C++
behavior rather than introducing a new integer promotion policy. Shared effect
proofs include expression-owned type/data dependencies; loads, traps and cycles
in those dependencies cannot become discardable through a pure wrapper.
Conditional operand-cache selections follow the same lexical facts as their
allocation guards. A safely proven false selector reuses the original producer,
including its historical read/state identities. Named allocation predicates stay
protected until a subsequent cleanup round confirms that every selector use
has disappeared. A single cleanup round remains correct and can retain unused
private declarations. True or unknown selectors keep
their conditional storage read. When an opaque consumer or owned metadata prevents
that rewrite, dependent allocation predicates remain intact; unrelated boolean
facts can still fold. Unknown metadata ownership conservatively preserves the
control structure instead of inspecting extension objects or adding allocations.
At emission, an exact effect-free boolean literal selects only its executable
cache-read or original-producer arm. An unreachable cache arm is not printed
after cleanup removes its owner. Nonliteral, malformed and extension-owned
selectors retain the lazy conditional path.
Fusion dependency checks include the selected cache-storage read as well as
ordinary producer reads. An unresolved cache choice preserves separate stores
instead of evaluating an inactive cache address for an alias predicate;
historical load identity queries retain their existing contract.
Shape compatibility compares representable integer literal dimensions by value,
including equal weak and explicitly typed literals. Scalar expression identity
and arithmetic promotion continue to distinguish their types.

The late `cpu-plan-storage` pass computes branch declaration scopes and
conservative lexical live intervals before emission. `cpu_scratch_reuse`
(bool, default True) permits compatible private buffers with nonoverlapping
lifetimes to share a capacity slot; logical storage/read identities remain
distinct. Historical reads and metadata references extend intervals. Worker,
hardware, implicit snapshot, state and workspace owners are not merged by this
planner. Intervals currently cover entire top-level regions, conservatively
overlapping lifetimes inside a shared nested loop; this is not hierarchical
loop/branch liveness. C++ printing consumes the frozen placement/slot map and does not
rediscover branch ownership. Runtime allocation traces describe physical slots.
Resource inspection lists each logical buffer's `physical_slot` and the
`memory_lowering.scratch_slots` count. Source dot recipes have already become
ordinary executable IR, so there is no separate empty matmul-plan catalog.
Managers snapshot the registry; analyses and traces are local to a run.

The late `cpu-plan-memory-proofs` pass records disjoint destination regions
for fused register programs after scalar facts and loop motion have finished.
It combines bounded signed integer aliases with lexical induction domains,
checks actual offsets/strides against a dense private allocation, and proves
separation on at least one root axis. Clipping on another axis does not erase
a valid separation proof. Every lexical occurrence of a shared group must
satisfy the certificate. Writes to allocation metadata, optional unvalidated
operands, unknown effects, unsupported geometry and possible arithmetic overflow
retain the existing alias guards and independent-store fallback. The prepared
CPU verifier checks certificates against the final IR; C++ printing only looks
up their group/view identities and does not run affine or induction analysis.
For two destinations based on the same required tensor argument, a separate
certificate can instead compare their actual offset/shape/stride address
intervals. It checks dtype identity and bounded nonnegative signed arithmetic;
holes remain inside a conservative bounding span. This proves only nonoverlap,
not buffer capacity or view bounds, including when a logical reshape has a
different rank from the ABI buffer. Original bounds/preflight checks remain.
Optional operands, storage shadowing, unknown geometry and overlapping spans
retain runtime versioning. Distinct tensor arguments receive no nonalias
promise from this analysis.

The same pass and lexical traversal also use bounded domains
to prove store source and destination extents equal, or the source extent one.
Captured scalar extents can simplify while historical tile and view descriptors
retain their original expressions. Compatibility certificates remove a redundant
broadcast check without rewriting those descriptors or reader identities.
Equality certificates additionally identify equivalent extents within exactly
one store or elementwise group, so vector coordinates, load broadcasting and
register-state consumers can reuse matching geometry. A source proved to have
extent one is compatible but does not identify it with a wider destination.
Nested store emission replaces the surrounding geometry scope and isolates
its vector cache; both are restored on exit, including exceptions.
Prepared equality classes also select an original scalar representative for
shape printing. A dominating scalar with one declaring assignment and no
other definition takes precedence over expanding its historical clamp;
remaining candidates use DAG cost and stable IR order. The planner intersects
availability across shared occurrences and rejects cycles between selected
representatives. Printing substitutes only exact certified expression objects
in the current store/group, preserving all historical load/View identities.
Hardware-owner scalar bindings retain their existing spelling.
The `ScalarGeometry` certificate also covers proved equal view offsets and
strides, using the same total, bounded signed-integer analysis. Alias snapshot
preparation consumes the exact store's certificate before printing its guards;
it still compares actual pointers and runtime address spans. Equality never
establishes nonaliasing or moves a metadata read, and unknown, effectful,
trapping or potentially overflowing expressions retain their original form.
Nested synthetic snapshot stores and sequential fusion fallbacks install their
own certificate scope and restore the outer aliases and vector cache on exit.
Group certificates additionally compare each store's historical source and
destination extents with the actual prepared vector traversal shape. They
intersect equality across every lexical occurrence; singleton compatibility
never supplies a cross-store equality. These extra relations apply only to the
exact group and do not pass into its sequential-store fallback.
Every occurrence of a shared store must satisfy the proof; unknown bindings,
scope effects, unsupported integer arithmetic and possible intermediate overflow
preserve the runtime check. The verifier recomputes certificates against the
final IR and its captured operation catalog. C++ printing consumes only the
exact store/extent identities and performs no range analysis.

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
On x86, complete-vector loops compute their invariant endpoint once using
widened arithmetic, keeping overflow checks out of the induction test. VLA
backends retain the comparison form to avoid runtime division.
Separate CPU passes split full chunks from masked tails, select wider automatic
tile chunks from accumulated region cost and a conservative live-vector budget, and legalize multiword math into native-word
calculations while preserving coalesced memory requests. Explicit Vector tags
and logical unroll counts do not change. `pass_options` can override
`cpu_full_chunks`/`cpu_wordwise_math` (bool), `cpu_math_chunk_pow` and
`cpu_reduction_chunk_pow` (0/1/2, default 1; 0 disables the corresponding
wider automatic chunk). `cpu_state_register_budget` (integer 0–64, default
12) limits standalone register pieces **per tile state**; 0 forces private
scratch for comparison/debugging. It is not a whole-kernel liveness budget.
State placement uses the actual code-generation ISA, independently of cost
profile overrides. On SVE, a captured runtime VL permits single-word scalable
register pieces; it does **not** enlarge the VLA `FixedTag` limit. Without a
captured VL, fixed pieces remain within the architectural 128-bit minimum.
Finite multi-row pieces preserve row coordinates and mask the final word.
Widening intermediates reserve lane headroom, and connected snapshots use the
same geometry, avoiding multiword register arrays and repeated scalar repacking.
Single-reduction regions can use a wider base to
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
The environment separately records the actual code-generation target. An ARM
matmul omits unreachable x86/AMX families, and vice versa; generic lowering
retains both target-conditional paths. Overriding `cpu_cost_target` only changes
planning costs, never instruction availability or architecture pruning.
Pass-registry changes still require `cache_clear()` as described above.

SME conversion packing may version an eligible pure FP32 pointwise DAG at
the packing boundary when producer-cache and historical READ-snapshot guards
are all false. The guard is outside the packing loops: K-contiguous input
then uses ZA staging and vertical vector drains before narrowing/interleaving.
Active caches or snapshots, unknown/unconditional materializations, masks,
reductions/state, nonunit K strides and intermediate narrowing retain the
original snapshot-aware vector fallback. This changes only code-generation
eligibility, not cache ownership, numerical precision or packing lifetime.
Emitter maps, expression caches and hardware scope are restored before the
fallback is emitted, including on an emission exception. Geometry comes from
the captured target vector state; it is not an assumed64-byte SME length.

Development formatting is pinned in the `dev` extra. Run
`python -m ruff format python/vecops/language` or append `--check` in CI;
the project uses two-space indentation and a 120-column target.

The `cpu-memory` stage plans transpose, state, grids, register layouts/programs,
hardware preflight and scratch storage on the prepared emission catalog. All
stage managers are frozen before user passes execute. Matrix algorithms have
already expanded into ordinary source recipes; there is no separate opaque
matmul schedule catalog. Transpose planning recognizes layout conversions,
proves dense destinations, selects dtype/ISA networks, and prepares SME
state-owning alternatives. C++ printing runs no passes.
`cpu_transpose_cache_block` (multiple of 16, default 128) and
`cpu_transpose_loop_order` (`"input"`/`"output"`, default `"input"`) are CPU
tuning overrides. Generic streaming multi-output fusion currently yields to
layout-aware transpose plans rather than bypassing their permutation DAGs.
`cpu_transpose_defer_conversion` (bool) overrides boundary-conversion motion;
the default cost policy keeps byte→FP64 in the compute domain for unknown
strides/extents and permutes narrow storage bits for constant layouts.
With a captured ordinary SVE VL wider than its guaranteed 128-bit prefix,
simple constant rank-2 conversions rejected by the shared automatic SME
selection may instead use a single-word native gather. This includes
conversions of at most 256 elements and eight-byte inputs of at most 8192
elements; it does not expand the automatic SME rejection thresholds.
For these automatic native gathers with both constant axes at most 16,
the independent row loops remain rolled, including their alias-snapshot
alternatives. This bounds native address-register pressure without changing
the inner Vector IR unroll factor, conversion rounding or mask/tail handling.
Medium gathers and user-selected implementations retain their row schedules.
Explicit `vector`/`sme`/`gather` choices remain unchanged. SME transposes copy
their current origins, dimensions and signed row pitches inside the streaming
owner, including fresh scratch pointers after resizing; ZA slice instructions
do not repeatedly reload reference-captured tensor metadata.
`cpu_transpose_sme_dynamic_unroll` (bool, default `False`) is an experimental
A/B tuning option: it requests two-way native loop unrolling only for SME
slice transfers with a dynamic row pitch. Constant pitches retain the native
compiler strategy. It can reduce address-multiple spills but is not a
performance guarantee; measure both settings on the same target and VL/SVL.
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

These are compiled lexical regions, not Python host context managers. SME
bodies admit simple whole-view rank-2 transposes, `zero_za`, typed load/mopa and
direct ZA slice moves; AMX admits typed tile zero/load/stream_load/store/tdp.
`T.sme.load(view, strided=True)` explicitly enables signed-stride numeric
vector loads (including gathers) under the same pre-owner root/word bounds.
The default, raw ZA memory moves and `load_bits` still require unit stride.
Bounded `T.serial` instruction loops preserve one owner. Pure integer/boolean
metadata branches must be invariant in enclosing instruction loops. Views are
validated before entry; allocation, throwing work, nested explicit regions and
values escaping a region are rejected. Explicit SME rejects overlapping operands
instead of allocating inside SM. AMX obtains per-thread Linux XTILEDATA permission
and owns configuration/release through `ExecutionSession`. Generated state-owner
callbacks always use `VECOPS_INLINE_LAMBDA_NOEXCEPT`.
The CPU preflight pass proves loop-wide view bounds before ownership rather
than validating/replaying each iteration. Unknown nonlinear index intervals,
lost root provenance, loop-varying store layouts, memory-dependent control,
mutable scalar/tile state and general SIMD computation are rejected;
`T.sme.apply` admits the bounded pure selection/conversion subset described
below. Region
assertions are checked before entry and must be loop-invariant.
`T.sme.load_hor/ver(tile, slice_index, rank_one_view)` and `store_hor/ver`
transfer storage bits without conversion; the view and index must fit the
actual streaming word and memory element stride must be one. Tile indices are
constant and legal for the selected word width. An optional `word_dtype`
reinterprets memory elements as complete unsigned ZA words; a fractional
terminal word is untouched rather than overread. For zero-padded partial
words, `T.sme.load_bits(view, dtype="uint32")` masks in storage-element units,
then bitcasts to a region-local unsigned word vector. `T.sme.store_bits(v,
view)` stores the requested storage prefix without numerical conversion,
including zero inactive lanes from a masked ZA read. Its destination may be
larger than the active prefix, but cannot exceed one physical streaming word.
Numeric conversion remains `T.sme.load` / `store`, not the bit-memory path.
`T.sme.transpose_tile(A, C)` is an
inspectable DSL helper used inside an active region: all input rows load
before any output column is written. It supports same-dtype word-sized
matrices (including partial tiles and overlapping operands), not arbitrary
large matrices. `T.transpose_copy` retains its existing selection.
An optional `compute_dtype` enables numeric boundary conversion via
`T.sme.load(view, dtype=...)`, `read_hor/ver(tile, index, active, dtype=...)`,
`write_hor/ver(tile, index, vector)` and `T.sme.store(vector, view)`.
Region-local vectors carry dtype/active extent independently of memory views;
their shape and every ZA slice index are checked before ownership. Conversions
fuse into the memory load/store; there are no temporary tensor allocations.
`T.sme.apply(pure_helper, vector, ...)` applies a source-parsed pointwise
selection/conversion expression to region-local vectors with identical active
prefixes. Scalar arguments and immutable string options (for example a dtype)
may accompany the vectors. The result is one numeric streaming vector.
Supported expressions are casts, min/max, comparisons, bitwise logic and
`where`; intermediates must fit the input compute word. Memory/state reads,
arithmetic, allocation and shape changes are rejected. Scalar numeric casts
retain the public conversion contract. This bounded vocabulary lets matrix
recipes consume ZA rows and directly write ReLU/converted results while keeping
floating arithmetic and exceptional math on their existing ordinary path.
Immutable scalar metadata declarations are legal within an instruction region;
preflight expands those definitions before hoisting checks. They are not
loop-carried mutable state.
`T.prefetch` is legal in bounded instruction loops. Its address receives the
same pre-owner root-bound validation even though the hint consumes no ZA
capacity. Loop steps and integer divisors may use proven positive native
extents/products; unknown runtime divisors remain rejected.
`T.sme.transpose_matrix(A, C, compute_dtype=...)` composes native-length serial
loops, clipped views and `transpose_tile` in one owner for larger matrices.
Its operands must be disjoint. `T.view_disjoint(A, B)` is a conservative
bounding-address-span predicate (not an exact arbitrary-stride alias oracle);
it can be asserted before ownership, not used as throwing work within SM.
Nested clipped views retain exact root coordinates separately from requested
affine partitions. Losing an affine ownership proof still prevents grid/loop
fusion, while exact bounds can remain usable for safe instruction traversal.
Automatic SME transposes always stage the source-parsed driver during CPU
memory planning, including paired/static-conversion cases. K-contiguous raw
SME packing likewise uses its single DSL recipe (panel traversal, ping-pong
ZA pairs, zero-padded K/spatial fringes). Terminal printing only binds checked
memory boundaries; it never invokes the frontend. The former
`cpu_transpose_sme_dsl` / `cpu_sme_packing_dsl` migration switches and their
duplicate native loop generators have been removed. Performance tuning now
applies to these recipes rather than selecting an obsolete implementation.
Likewise `cpu_amx_shared_cache_dispatch=True` enables the experimental shared
dynamic cache driver; its smaller source is not itself performance acceptance.
The default retains the original family-dependent automatic packing policy.

Logical tile expressions can be indexed with unit-step slices, including a
scalar axis when at least one sliced axis remains: `value[mi:mi+BM, ki:ki+BK]`
or `value[row, :]`. Windows retain the original producer/read identity and
capture mutable starts when assigned; they do not silently reload an input
after a later write. Slice bounds clip as for views. Scalar axes are checked;
use `T.element_at` for entirely scalar extraction and `T.gather` for explicit
stepped indices. An arbitrary window is a logical tile, not a new physical
register geometry; use `lower_half`/`upper_half` for register splits. This is
a foundation for source-level operator schedules, not a replacement of the
current matmul driver.
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
are unchanged. Empty C-contiguous NumPy arrays with an all-zero stride tuple
retain the legacy canonical dense ABI strides. Explicit nonzero signed strides
are preserved on empty/singleton axes: NumPy's C/F contiguity flags alone do
not specify the layout of an empty tensor.
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
After tracing, `compile_batch()` matches, binds, and looks up every call first,
including the relationships between named Dynamic dimensions. Cold misses
sharing a toolchain are then deduplicated by their bound source-artifact
identity: different runtime Dynamic observations can share one compiled target,
while Const/dtype specializations, source/interface identities, and logical
parallelism remain distinct. Every alias retains its build-input validation
before compilation and before publication, and the resulting executable is
adopted by every requesting provider/cache namespace. `PrecompileResult.prepared`
counts prepared requests, not unique compiler targets.

All unique misses sharing a toolchain enter one generated CMake project. Each
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

## Kernel inspection, diagnostics and profiling

`T.template`, `T.compile` and `T.jit` objects expose `get_kernel_source()`,
`get_host_source()`, `get_ir(stage)`, `get_pass_report()`, `explain(topic)`,
`get_cache_info()`, `get_build_report()`, `get_assembly(specialization)` and
`export_debug(directory)`. `T.lower(template, target="AVX512")` elaborates and
optimizes without native compilation. Template getters describe the latest
lowering/compilation; factory getters describe its latest returned callable.
Python forwarding functions retain their identity and expose child reports
when they allow attributes. IR text and diagnostic schemas are visualization
interfaces, **not cross-version-stable serialization**. Missing IR/artifacts
return `T.debug.Unavailable(reason)` rather than silently substituting another
kernel. Source-backed C++ templates do not acquire fictitious DSL IR.

```python
lowered = T.lower(kernel, debug=T.DebugOptions(trace="changed"))
print(lowered.get_ir("vector"))
compiled = T.compile(kernel, out_idx=-1, compiler=compiler,
                     debug=T.DebugOptions(memory=True, timeline=True))
compiled.compile_for(A)              # prepare; no execution
print(compiled.get_kernel_source())
print(compiled.analyze(A))           # metadata only; no output allocation
print(compiled.get_layout(A))
timings = compiled.profile(A, mode="native", warmup=10, repeat=100,
                           trace=True, memory=True)
timings.export_trace("trace.json")   # Chrome/Perfetto JSON
compiled.export_debug("empty-debug-directory")
```

`DebugOptions(trace="off"|"changed"|"all", passes=(...))` controls immutable
before/after pass snapshots and textual diffs. Pass durations exclude snapshot
formatting. Source positions refer to captured Python expressions/statements;
rewritten nodes may not retain an exact original location. Failed lowering or
compilation attaches available evidence as `exception.debug_info`.

`analyze()` reports affinity/quota/executor capacity, grid worker coverage and
equal-cost wave efficiency, scratch/layout plans and IR vector/mask liveness.
These are **CPU estimates, not GPU-style hardware occupancy**. Runtime guards,
external/nested executors, data-dependent bounds, address registers and native
compiler temporaries can change the answer. `analyze_assembly()` counts whole
object call/stack sites, not exact hot-loop spills; optional `T.debug.llvm_mca`
accepts caller-selected assembly and reports unavailability without that tool.

Profiling explicitly executes and mutates outputs. Compilation/preparation is
outside timed samples; `mode="python"` includes binding/output allocation and
`mode="native"` prebinds NumPy ABI arguments (Torch uses its native bridge).
Both retain provider dispatch, and neither claims pure instruction timing.

Native compiler regression benchmarks warm their compared callbacks in round-robin
order for at least 100 ms after establishing CPU affinity, first-touch buffers and
the grid thread budget. Results record the warmup duration and invocation rounds.
This lets an idle powersave CPU reach its working performance state without
changing the system governor. Calibration and timed samples follow the warmup;
source, numerical and allocation checks remain outside those samples.
Reports include frequency/governor/affinity without changing them. Memory
instrumentation records DSL scratch allocation/release plus existing workspace
sites; timeline instrumentation records kernel/grid-program regions. IDs are
anonymous, contents and raw allocation addresses are not exported. This does
not instrument all process allocations, packing phases or barrier waits.
Instrumentation is opt-in at compilation, with no observer calls inside normal
un-instrumented vector loops. Traces must not be interpreted as uninstrumented
performance measurements.

`compiled.interpret(...)` provides a checked sequential NumPy reference for
the supported frontend subset. Unsupported hardware/atomic/collective
operations raise `UnsupportedInterpretation`; this is not an ISA emulator or
a promise of bitwise identity. `T.debug.compare(actual, expected)` reports
absolute/relative errors, same-dtype floating ULPs, NaN/Inf counts and the first
mismatch. `T.debug.minimize(candidates, fails, max_attempts=...)` performs bounded
subsequence delta debugging using an explicit deterministic failure oracle;
budget exhaustion never claims a proven minimal result. Neither helper rewrites
source or runs arbitrary kernels without the caller requesting execution.
Debug export requires an empty directory, omits full command arguments unless
`include_commands=True`, and never dumps tensors or the complete environment.

### Compile-time and runtime print

`T.static_print` executes during elaboration. `T.print` emits a runtime effect;
an unshadowed Python builtin `print` inside a template is its alias. Both accept
`sep=" "`, `end="\n"`, `file=None`, `flush=False`, `buffer_size=1024` and
`truncation=" ... (trunc)"`. `sep/end=None` select Python defaults, and `file`
must be `None`, `T.stdout` or `T.stderr`, not a captured Python file object.
Text/format specifications, buffer size and truncation marker are compile-time
values; runtime `flush` may be a boolean scalar. F-strings accept scalar fields
with constant C++20 format specifications and default/`!s` conversion; `!r` and
`!a` are rejected. To keep the path allocation-free, precision is capped at 64,
locale/debug-escape formats are rejected, and numeric fill characters must be
ASCII. Float formatting uses a bounded `to_chars`-based custom formatter within
the same `format_to` pipeline, avoiding libstdc++'s large-fixed-value temporary
string fallback. Immutable tuples, scalars, masks, tiles and vectors can be
printed. Tile/vector displays are bounded by `DebugOptions(max_print_elements=16)`.

```python
T.print(f"sum={total:.3f}", file=T.stderr, flush=True,
        buffer_size=1024, truncation="[cut]")
```

Capacity is measured in **UTF-8 bytes**. Marker bytes are always reserved:
`buffer_size=1024, truncation="[cut]"` permits 1019 body bytes, including
separators and `end`. No terminating zero is required. If the body exceeds that
capacity, a complete UTF-8 prefix followed by the marker is emitted; the final
newline may therefore be truncated too. An empty marker reserves zero bytes;
a marker equal to buffer capacity leaves no body room. `buffer_size` must be
an integer in `[1, 1048576]`, and the marker must fit it.

Runtime lowering always uses `std::format_to` directly into a fixed stack byte
buffer, followed by one synchronized C-stream sink write. It never constructs
an intermediate `std::string` and does not select a separate `<print>` path.
The stream lock is shared across threads and generated DSOs. Each invocation
is atomic relative to other users of that C stream, not across processes or
separate file descriptors. Separate `end=""` invocations are separate records;
there is no deterministic worker ordering or computation-memory barrier.
`DebugOptions(prints=False)` strips runtime prints before optimization, but does
not suppress explicitly requested compile-time `static_print` effects.

#### ARM toolchains with an older C++ standard library

The CPU print implementation requires C++20 `std::format_to`; selecting a
recent Clang alone does not provide a recent C++ standard library. For example,
the target cluster's BiSheng Clang can otherwise select GCC 10 headers without
`<format>`. Do not replace the bounded-buffer print path with an allocating or
`printf` fallback to work around that toolchain mismatch.

For scheduled ARM validation, `benchmarks/arm_job_env.sh` verifies a pinned,
project-private GCC 14 header bundle and durable runtime libraries, runs actual
Clang syntax/link/execution probes, and exports a job-scoped environment. Source
it after activating `af3`, **before starting Python**: NumPy or Torch can select
an older `libgcc_s`/`libstdc++` before vecops is imported, and changing library
search paths after that load does not replace those mapped libraries. The
helper does not modify the conda environment, shell startup files or global
compiler configuration. Its inputs and resulting runtime identities are
recorded in the job manifest and participate in compiler/cache identity.

LLVM 19 can also reject a scalar-conditioned scalable BF16 selection with
`Cannot select ... nxv8bf16 ... select_cc`. Upstream
[LLVM #128881](https://github.com/llvm/llvm-project/pull/128881) adds the missing
selection legalization. For the affected ARM transpose validation, both frozen
DSL versions and the C++ reference use the same LLVM 19 `-O3` configuration
with `-mllvm=-two-entry-phi-node-folding-threshold=0` and
`-mllvm=-phi-node-folding-threshold=0`; these are recorded compiler-workaround
options, not a global optimization default or a general cure for BF16 errors.
Keep the actual compiler, wrapper contents and options in the native identity.
LLVM 21 is not a drop-in replacement for the current base-SVE predicate-tuple
bridge: [LLVM #108008](https://github.com/llvm/llvm-project/pull/108008) changed
tuple IR representation, while that bridge binds the LLVM 19 vector intrinsics
directly. Its compatibility must be established separately before upgrading.

The helper requires the explicit bundle/runtime paths and SHA-256 values listed
in its usage error. Temporary headers are extracted under an owner-only `0700`
job directory; runtime DSOs use durable project paths so compiled artifacts do
not depend on a deleted temporary runtime. This is a validation/deployment
environment helper, not an automatic Python-import-time toolchain upgrade.

`benchmarks/dsl_arm_validate.py` scopes implicit compiler, build and Torch
bridge caches to a fresh `*-cache` sibling of its pytest basetemp. It overrides
inherited `VECOPS_CACHE_DIR`, `VECOPS_BUILD_DIR` and `VECOPS_TORCH_BRIDGE_DIR`
for that invocation and restores them on exit, including failure. This keeps
an application's home cache out of a private `/tmp`/`/dev/shm` validation run.
Cache contents are retained for debugging; an existing sibling is rejected
rather than reused or overwritten. Explicit per-test `Compiler` paths are
unchanged. The selected paths are recorded in validation provenance.

### Content-addressed cache invalidation

Artifact keys now include source/include/dependency content, generated C++,
Python/DSL/SDK/native-extension implementation, registered pass/fold callbacks,
target/cost environment, compiler tools, standard-library headers/runtime
libraries and compile/link policy. Source-backed kernels use the same mechanism;
manually incrementing `-v2/-v6` recipe labels is unnecessary. Explicit
`T.template(..., dependencies=(...))` / `Operator(..., dependencies=(...))`
cover external macro-generated includes outside declared include directories.
Custom opaque native/plugin state must provide `__vecops_cache_token__` or its
identity is deliberately non-cacheable. Directory inventories also detect new
shadowing headers. Python identities read file contents and sorted directory
inventories on every check. Filesystem notifications and size/mtime do not
authorize content reuse: shared mappings, external hard links and ancestor
rebinding can escape directory notifications. Cold compilation validates its
inputs before and after the build. These reads do not provide an atomic
snapshot of a concurrently changing filesystem. Toolchain attestations are
atomically published under writer locks.

Relative include paths and the include environment are anchored at the
`Compiler` creation directory, including nonempty path lists' empty components
such as a trailing `CPATH` separator. Incidental outputs under this implicit
cwd root do not become toolchain dependencies: resolved include contents and
shadow selection remain tracked. Explicit include-directory inventories retain
their conservative behavior. Quoted includes search the including file's
directory first; angle includes do not. Macro-generated includes should use
declared dependencies or explicit include-directory inventories.

On Linux AArch64, the **current calling thread's** SVE VL and SME SVL are
independent compilation inputs, queried through the OS rather than assumed
from a CPU model or maximum supported width. Factory, lowering, artifact and
batch keys include both lengths. Generated kernels use captured constant lane
counts while retaining sizeless vector representations, including when VL and
SVL differ. Changing either length requires obtaining a kernel again through
`T.jit`/`T.compile`; invoking an older compiled object reports a length mismatch
instead of executing with incompatible geometry. `T.grid`/resident-team workers
temporarily match the caller's lengths and restore their original state afterward.
`Compiler.snapshot()` freezes this state and compiler policy for pending
source/DSL compilations on every platform; its environment dictionary is copied.
The original `Compiler` remains reusable by threads with different settings.
Factory lookup separately checks that original compiler's current policy, so
changing flags or build policy produces a new specialization without changing
an already returned kernel. Snapshot calls are idempotent; build-event storage
is shared for inspection. This prepared-kernel contract does not make an
already obtained raw native compiler/backend follow later policy mutations.
Standalone dry/cross lowering with an explicit non-Native target does not probe
the build host for vector geometry. Pass an explicit `_vector_state` snapshot
for geometry-specific inspection; the normal compile driver supplies one.

Source-parsed hardware library recipes can specialize integer loop metadata
using that captured streaming length. Full-tile branch facts refine only their
own induction domains; tails and sibling/successor blocks retain their bounds.
Integer interval and clamp proofs memoize shared expression DAGs only within one
query. Cache keys retain expression and endpoint identities, lexical bound order,
ordered-simplification mode, and operation-catalog identity; nested catalog
changes use separate caches. Exceptions release the query state and parallel
lowerings have isolated contexts. Facts do not survive into another query.
Constant-shape packing binds shape metadata before instruction printing, while
signed runtime pitches, alias snapshots, capacity checks and storage-bit
semantics remain intact. Unknown SVL retains native length queries. These
optimizations do not change the current experimental status of the complete
SME packing driver or promise universal performance parity.

SME's centralized `sme_const_half_transpose` policy selects a bounded paired
input/output schedule for simple FP32-memory/FP32-compute to FP16 transposes
with known nonzero signed pitches. It derives the row/column count from the
captured SVL; at most sixteen compute lanes bound code size and liveness,
not the hardware's vector width. Actual complete-tile guards precede unmasked
loads. Tails, overlap snapshots, unsupported strides, converting inputs and
other dtype families retain their existing paths. A cost profile cannot supply
missing SVL or enable SME on another code-generation target.

The policy can be disabled with a cost-profile override of integer `0` (only
exact integers `0`/`1` are accepted). Diagnostic pass options
`cpu_transpose_sme_const_half_output` and
`cpu_transpose_sme_const_half_full_input` are strict booleans and override the
policy independently. Setting only the output option to `False` also disables
the implicit full-input schedule; explicit input `True` can still isolate that
schedule for inspection. These compiler-tuning interfaces are not portable
DSL semantics or a guarantee of performance parity for all transposes.

Decorator lookup invalidates stale dependencies; **already returned/prepared
kernel objects are immutable snapshots**, so hot execution does not read files
or unexpectedly rebuild. Cold builds validate inputs before and after building
and refuse publication if inputs changed concurrently. Disk artifacts validate
the complete identity manifest, not merely a short directory hash. Torch bridge
keys additionally attest Torch headers/libraries and publish staged artifacts
atomically after validation. Cache-only mode uses an existing unambiguous
attestation, executes no compiler/tool discovery and creates no build directory;
absence/ambiguity is a diagnostic miss rather than loading an incompatible file.

### Source-parsed register algorithms

`T.bitcast(value, dtype)` reinterprets equal logical bytes, unlike numeric
`T.cast`. Equal element widths use an elementwise bit reinterpretation,
retaining scalar/tile shape and any clipped Vector prefix; this does not
require a complete register geometry. Different widths change logical lane
geometry and retain the complete-register constraints below. Numeric casts
may saturate and are not a substitute for signed/unsigned bit reinterpretation.
`T.lower_half`/`T.upper_half` select halves of a logical vector and
`T.concat` joins two equal-width vectors. Their names do not replace the
existing debugging function `T.lower`. `T.local_interleave_lower` and
`T.local_interleave_upper` interleave within complete 16-byte blocks;
`T.block_shuffle(a, b, order)` selects whole 16-byte blocks from concatenated
inputs. Fixed geometry requires a positive compile-time rank-one extent
(even for half selection), not necessarily a power of two. Complete scalable
register programs also preserve their original byte domain and power through
bitcast, half/concat and `T.interleave(a, b)`. Interleave returns alternating
elements from two complete same-dtype/same-extent registers, doubling length.
Arbitrary clipped/dynamic geometry remains diagnosed, not silently retagged;
block-shuffle orders still require fixed geometry.

`T.itemsize(dtype)` and `T.static_log2(power_of_two)` provide compile-time
geometry; the latter rejects nonconstant/nonpositive/non-power-of-two values.
`T.library.transpose_registers(tuple_of_vectors)` is an inspectable `T.func`
butterfly, returning one vector per output column. It accepts power-of-two
square or double-width geometry and numeric storage types, with no floating
point arithmetic or conversion hidden in its bit-level permutation.
`T.library.pack_interleave(tuple_of_vectors)` forms grouped-K dwords from
two 16-bit or four 8-bit streams using ordinary interleave/bitcast primitives.
Both fixed and scalable inputs are supported; no native length is guessed.

Automatic transpose and matrix-packing plans stage the same library helpers.
The backend shares one typed instruction printer; small closed register DAGs
use native instructions, while unsupported/large virtual geometries retain
an exact scalar-lane fallback. Fixed SVE placement uses its guaranteed prefix,
not a hardcoded 64-byte vector length. Every physical vector is individually
named. Register-transform stores preserve aliased inputs before permuting;
this also covers historical reads and in-place logical-vector fallbacks.

Structured control can carry a fixed-arity immutable tuple of scalar,
tile or vector values, including nested tuples. Assignment captures all
source values before updating any destination slot, so bank rotation and
swapping retain simultaneous semantics. Runtime arity/dtype/tag changes
are errors; view rebinding still requires explicit storage. This is a DSL
grouping only: native Vec/Mask values remain individually named variables,
never C++ arrays or tuple aggregates.

`T.sme.apply(helper, vector, ...)` evaluates a pure `T.func` selection or
conversion on region-local streaming vectors with identical active prefixes.
Casts, min/max, comparisons, bitwise logic and `where` are supported when
all vector intermediates fit the input compute word. Arithmetic, captured
memory/state reads, allocation, shape changes and widening remain outside
the owner. Scalar casts retain the ordinary public conversion contract;
chained selections retain the narrowed streaming tag geometry. The resulting
vector cannot escape its owner.

Raw SME memory/ZA slice moves accept `word_dtype=...` to select the ZA word
width independently of memory element type. For example, a contiguous byte
view can be transferred as packed `uint32` words without numeric conversion.
Only the complete-word prefix is transferred; a fractional last word is
untouched. Slice-index/capacity checks use the selected word geometry, and
never round a tail up into an out-of-bounds access. This supports library
grouped-K packing without embedding its algorithm in the instruction emitter.

`T.sme.mask(count, dtype="uint32")` constructs a region-local streaming
prefix predicate. `T.sme.read_hor/ver(tile, index, predicate, dtype=...)`
accepts it in place of the scalar active count; the optional result dtype
must have the same lane width. A `T.func` may return this immutable handle
inside its owner. The predicate cannot escape or be reused by another owner,
and its `[0, streaming_lanes(dtype)]` capacity is checked before entry.
This lets a source-level schedule construct spatial predicates once before
its K-group loop, without relying on native loop-invariant code motion.
Known complete prefixes lower to `mtrue` only using the captured **SVL** or
an explicit streaming-length expression, never the ordinary VL or CPU model.

Experimental CPU recipe tuning: `cpu_vector_dot_dsl=True` expands eligible
explicit unblocked vector dots into the source-parsed FMA/state library leaf.
The current scope is known small output geometry without packed operands;
all other dots retain their existing path. This is not a portable DSL promise
or completion of matmul/cache-driver migration. It defaults to false and
sets defaults for `cpu_state_vector_reuse` and `cpu_ordered_loop_unroll` to
true unless explicitly overridden. All three options require exact booleans.
Projection reuse is local to proven complete register pieces; ordered native
unrolling preserves each iteration's full state update when operation-major
unrolling cannot legally duplicate it. These diagnostics will be removed or
consolidated after the complete migration is validated.

Scalar extraction through a logical window/cast may use a direct memory read
instead of a one-lane vector fill/get. It retains the original capture's
snapshot and every explicit conversion boundary. Masked or nonlocal values
keep normal vector legalization; geometry folding never authorizes a later
read of a previously captured and overwritten input.

`T.zeros(T.Vector(dtype, N=... or pow=...))` and
`T.full(T.Vector(...), scalar)` create explicit register values without a
dummy memory load. Their dtype comes from the descriptor; a conflicting
explicit dtype is diagnosed. Numeric fixed/scalable geometry is supported.
`active=integer` optionally restricts the logical value to a checked prefix
without changing its physical descriptor, allowing zero-size/tail values and
loop-carried scalable accumulators. The prefix must lie in `[0, lanes]` and
remain shape-compatible across state updates. Ordinary tuple-shaped fills
keep their tile semantics and do not accept `active`. Virtual powers exceeding
the register budget retain normal strip-mining/private-scalar placement, not
C++ arrays of Vec/Mask. No memory access is implied by vector initialization.

`T.as_vector(rank_one_value_or_view, T.Vector(...))` clips to the descriptor's
logical extent, converts to its numeric dtype and retains explicit vector
geometry. It is a value conversion, not a packing/allocation operation or an
invocation of a native operator; memory masks and historical captures remain
attached to the producer. Mixed Tile/Vector arithmetic preserves a compatible
explicit vector tag irrespective of which operand appears first. Broadcasting
a short vector to a larger unrelated tile does not make that tile a vector.
CPU state placement splits at the widest vector intermediate/consumer, and
canonicalizes equivalent physical rebinds, avoiding multiword state objects
and scalar get/set repacking across a widening/narrowing boundary.
Closed fixed-register programs are planned before state placement. On x86
with a known native register width, an explicitly typed rank-one fixed state
may retain that fixed representation when every declaration and update has
a complete, matching register program whose intermediates fit the native
register and whose bit reinterpretations preserve byte counts. Shape alone
does not establish this proof. Masks, partial or unknown descriptors, matrix
state, private-call crossings, other targets and exhausted register budgets
retain the existing placement rules. Prepared verification rechecks the
original writes and program subjects before code emission.

`T.view_disjoint(a, b)` also accepts captured scalar/tile/vector producer
values. It expands to pairwise runtime disjointness checks for their memory
read views, without evaluating arithmetic or creating another tensor. Values
with no external reads need no checks; opaque/effectful producers that cannot
expose their memory views are diagnosed. An enclosing true branch contributes
the same bounded-view proof as the existing two-view form. This is a runtime
guard, not an unchecked noalias annotation, and historical captures are kept.

## Distribution license material

Python wheels retain the root MIT license, `THIRD_PARTY_NOTICES.md`, and all
referenced upstream texts from `LICENSES/` in wheel metadata and under
`vecops/_sdk/share/licenses/vecops`. The latter stays with the bundled headers
and native SDK when it is copied independently. Project SPDX fields cover only
project-owned support code; user kernel logic and third-party portions retain
their original terms. Generated DSL C++ identifies this distinction explicitly.

### Register-prefix and SIMD store legalization

Generic vector traversal and carried registers share canonical physical tag
geometry, including dtype rebinds and generated C++ tag aliases. A bounded
one-word prefix is reused directly at its zero origin. On x86, a fully active
fixed vector can read the low bytes of one native scalable state word through
the public fixed-to-fixed bitcast; emitted type assertions verify the physical
representation bridge. A wider scalable consumer can similarly resize one
native state word when its runtime source origin is zero and its logical
extent is not one. A predicate clears every inactive destination lane, and
the public resize zeros bytes beyond the physical source word. This path
requires a known x86 native width, equal element types, rank one, and no
hardware-owned register region. Empty, shifted, singleton, masked and ARM
sizeless reads retain their existing projection paths. Singleton broadcasting
replicates source lane zero rather than advancing through inactive lanes.

A state whose complete typed register programs prove a physical `FixedTag<T,N>`
keeps that exact representation when consumed. Generated register aliases
record the descriptor's physical geometry; logical shape alone never supplies
it. Equal fixed consumers reuse the register directly. A wider rank-one x86
consumer may zero-extend a single closed fixed piece at zero origin, using
public `concat` operations for exact power-of-two widths and public resize
otherwise. Its predicate is clipped to both active lanes and the logical
source length: small fixed vectors can have nonzero physical padding bytes.
Unknown, shifted, singleton, masked and hardware-owned cases retain the
projection fallback. The scalable-word and closed-fixed proofs stay separate
while sharing the checked bridge and isolated lazy-fallback printer.

An exact unit-step `T.simd` scatter into a known nonzero-stride rank-one view
can become a bounded vector store without requiring a `unique=True` hint.
Arbitrary/permuted indices and unknown or zero strides retain their indexed
semantics and checks. Unblocked numeric vector dots, including packed inputs, use
source-parsed register or streaming helpers by default, including K-vectorized
inner products for column GEMV/K-contiguous B. Cache-blocked products now use
the same source leaves inside ordinary DSL panel/K loops. Their epilogues are projected
from the captured caller DAG into the current vector, not implemented as host
callbacks or native closures. Cost-based operand caches use lazy reads: an
inactive cache branch never loads uninitialized private memory or evaluates
both arithmetic alternatives.
Integer vector recipes perform modulo-32/64-bit multiply/accumulate using unsigned
arithmetic plus bit reinterpretation, not numeric saturating conversion.
Packed ABI capacity, density and output-disjointness checks precede decoding.
AMX and SME instruction families now stage source matrix recipes, including
accumulators and pointwise producer/epilogue DAGs. Packing and fringes remain
source algorithms; the terminal instruction printer owns no matrix/cache loop.
Raw first-sink FP32/FP64 SME products use masked direct ZA stores. Complete
FP32 blocks reuse four tiles in a 2x2 schedule; eligible FP64 blocks use all
eight ZA tiles in a 2x4 schedule to reuse each A load across more columns.
Other complete/fringe shapes retain their bounded four-/one-tile schedules.
Optional reuse-driven online packing belongs to `matmul`, not `dot`.
Raw FP32/FP64 SME dots retain strided reads, including repeated and fused
calls; they do not query cache capacity to choose a private input pack.
For automatic matmul, moderate inputs spanning several register tiles and
at least 64 K elements can be privately packed within a two-L2 input budget;
smaller/nonresident inputs retain strided reads. Const and Dynamic FP64 use
the same cache-budgeted choice. Explicit `packing="none"` disables that
optional choice; A/B/AB select their explicit dense-cache policy.
Instruction-required layouts, bounded tail padding and producer/alias
snapshots remain valid for both APIs. In particular, accepting raw B for AMX
still requires its architectural grouped layout. Online packing is included
in operation timing, and aliased producers are captured before writes.
SME native operands use the public panel-major packed layout, constructed by
ZA storage-word transposes from logical producers. Complete spatial panels with K-contiguous raw views skip the intermediate
dense copy; transformed, spatial-fringe or other-stride producers first
materialize their logical domain. Private storage rounds K to complete
streaming chunks for bounded access proofs; MOPA consumes only the logical
K groups, and no producer is evaluated at a padded index. Checked external
SME packed A/B and AMX packed A/B are read directly when K contains complete instruction groups.
For fractional K, SME borrows the complete-group prefix and privately copies
only one terminal group per spatial panel. That repaired group is consumed
once after the prefix, inside the same ZA owner and in the original K order.
The one-sided packed path uses the other operand's corresponding terminal
group. AMX retains its physical-panel copy and terminal-tile repair.
Neither path decodes or retransposes the whole logical operand. Capacity and
packed-input/output nonaliasing checks remain in force. External K padding is
not a numerical promise; discarded NaN/Inf payloads never participate in the
matrix arithmetic. SME selects zero with byte group offsets, without a wide
int64 index mask or per-dword scalar loop. This repair is required for external
padding correctness even when a benchmark packer happens to supply zeros.
SME I16/I64 retains its separate four-halfword layout:
the public I16 packed ABI uses two halfwords per dword and is not compatible.
The default resident library Matmul leaf preserves these direct-input paths;
explicit cache phases use logical slices and preserve the initializer/epilogue
once across all K phases. Closed SME selection/conversion epilogues fitting
one accumulator word are evaluated from ZA rows and stored directly. Raw
gather operands, wider intermediates, extra memory reads, arithmetic and
unsupported output strides retain buffered output. Initializers are completely
snapshotted before direct stores, including full matrices and row broadcasts
that alias C. Buffered epilogues project over a whole logical row, letting
ordinary vector legalization share layout checks and batch vector words
instead of rebuilding a one-word scatter traversal for every word.
SME row/scalar broadcast initializers stage one private row and initialize ZA
with bit-preserving row writes rather than a full matrix memory round trip.
This preserves a bias alias of the output and negative zero/NaN payloads;
arithmetic initialization with an outer product of ones is not equivalent.
Owned-buffer forwarding may replace an initializer only for the same view.
A zero-offset partial store first materializes a different previously
forwarded view, preserving its surrounding elements and matrix padding.
Alias snapshots follow the original read-capture order, with stable storage
ordering at multi-output boundaries. Object addresses identify retained reads
but never select their emission order: otherwise equivalent JIT lowerings can
change the native compiler's spills and layout without an algorithm change.
AMX and SME callbacks snapshot invariant pointers and integer metadata into local SSA
values after preflight, so instruction memory clobbers do not reload captured
descriptors inside K loops. Buffers may resize before ownership; their addresses
are never cached across region invocations. Default cache panels reuse multiple
register tiles within an L2 budget; explicit BM/BN/BK values remain exact.
Single-definition immutable streaming-length aliases and bounded literal
multiplication/division chains are reconstructed inside
the owner, preserving the adapter-validated SVL facts across the outlined
trampoline. Ordinary VL, mutable aliases and runtime tensor metadata are not
substituted. Static SME recipes also specialize their integer interval proofs
with captured SVL; Dynamic recipes retain scoped runtime bounds. Private packed
SME views of instruction-private panels expose each panel's two complete words
as separate axes, avoiding repeated clipped slices without changing the public
packing ABI or K repair. BF16 private panels and externally packed inputs keep
their existing slices: additional reshape/preflight work regressed their tiny
resident leaves. Fully borrowed physical panels also retain their original
scalar snapshots, while pointers keep region-local SSA values; poisoned K
padding still receives private repair. These measured scope limits affect
lowering alone, not input validation or numerical semantics.
For coordinate-preserving AMX B inputs, full panels use direct source
registers with either K-contiguous transpose or spatial-word interleave.
Direct dots share the downstream A/accumulator/output consumer across layout
branches. Cache Matmul uses this choice for exact full-panel metadata.
Unconfigured automatic raw BF16 matmul with a native FP32 result, no
initializer and no epilogue also retains B layout evidence in its resident
leaf. This avoids the dense intermediate for small and medium raw products.
Other unbounded Dynamic cases retain the materialized fallback: broader
layout branches regressed small x86 BiasRelu kernels, including under a
1.8 GHz cap. The original producer and all cast boundaries remain intact. Proven-layout fringes stage one physical B tile; unknown layouts
retain their full logical-domain staging. Only padding is zero filled and
producers never see padding coordinates. K-contiguous fringe rows use an
explicit one-word vector descriptor, avoiding unused wide tail aggregates.
Multi-output register permutations share one typed DAG, including input loads
and conversions, and drain each output when ready to limit live vectors.
Bare permutations of unbounded, already materialized private reads keep
independent output banks. Exact tile caches, original producers and numeric
conversions remain eligible for shared evaluation.
Automatic grouped SME matmul additionally reuses one full-K private A pack
across bounded full-K B panels when padded A fits 4 MiB and a B panel fits
1 MiB. B panels cover at most 256 columns and align to two accumulator
streaming words. Native raw BF16 inputs with at least 128 rows and contiguous
K strides may instead prepare the complete B when its physical, padded size
fits 32 MiB, avoiding repeated panel-owner entry. Other layouts retain the
bounded panel schedule. Each leaf initializes its accumulator once and consumes all
K, avoiding cache-K partial reloads and a separate full-result zero fill.
The complete private result remains live until all input panels have been
consumed, preserving input/output overlap and captured epilogue reads.
Private A/B views retain their actual K-chunk strides; they are never treated
as flattened external packed descriptors. Oversized A/K, explicit block sizes,
explicit packing policies and external native packed descriptors retain their previous
paths. This scheduling is implemented in the DSL recipe; dot does not acquire
automatic optional online packing.
CPU emission places an explicitly allocated scratch Buffer descriptor inside
its common conditional branch when all storage references remain in that
branch. Unselected matrix/cache paths therefore do not construct or destroy
those descriptors. Serial-loop capacity reuse and worker isolation retain
their existing lifetimes. The compiler
option `cpu_branch_local_buffers` defaults to `True`; setting it to `False`
restores entry-level declarations for controlled code-generation comparisons.
This is a compiler lifetime optimization, not a change to T.func inlining or
the private function-call ABI.
The existing CPU storage plan also scopes an implicit snapshot owner to its
single pointwise Store or ElementwiseGroup when every use of its exact
historical read belongs to that consumer. BindTile remains a read definition;
shared reads, metadata uses, outlined captures, hardware/worker regions and
opaque ownership retain the previous scope. The verifier recomputes this
proof from final IR. Emission additionally refuses a local scope when it must
protect an active read for a future consumer. Alias checks and snapshot copies
are unchanged, and outer historical bindings are restored after local emission.
For a single ordinary read with no existing snapshot/materialization, emission
computes its original alias predicate once at the original point. Only the
alias branch constructs the Buffer and executes the original guarded copy;
the nonalias branch has no implicit owner or destructor. Multiple reads and
existing historical owners keep their previous protection order, including
potentially throwing span/layout checks.
`cpu_snapshot_lifetime_scopes=False` retains the previous implicit-owner scopes
for controlled comparisons. This optional proof tracks direct ownership edges
and each node’s first consumer or escaped status, rather than transitive read
sets. It has no global limit on independent reads. Cyclic or opaque ownership
graphs use the existing path.
Private disjoint storage skips runtime alias checks; unknown overlap retains
sequential stores. An empty output prefix skips all input computation. Checked
rank-one reshapes retain element counts, rounding and historical captures.
Pure integer metadata min/max and vector-tail clamps evaluate each operand once through an inline
value helper; floating-point min/max semantics are unchanged. On ARM, signed scalar
floor division, remainder and ceil division by a positive constant use small
nonthrowing specializations. They evaluate the numerator once and preserve
negative-numerator semantics, including `INT64_MIN`; unknown or nonpositive
divisors retain the checked general helpers. ARM known-rank shape checks pass
dimensions by value and preserve ordered negative-size/product-overflow checks,
including checks after a zero axis. This avoids outlined initializer-list
iteration in large generated kernels. x86 keeps its measured original emission:
applying both rewrites to large GCC AMX functions regressed small fused dots.
These choices follow the actual codegen architecture, not a cost-profile
override. Divisors derived from SVL or other SSA aliases retain their runtime
form: broader specialization regressed the measured smallest packed dot.
Recipe views, packing selection, loop/branch interval analysis, runtime shape,
stride, capacity and alias contracts retain their original behavior.
Same-dtype SME
loads use public masked `load`, whose inactive lanes are zero, and known unit
strides use contiguous addressing. Converting, unknown-stride and fringe
loads retain their checked conversion/mask paths.
JIT streaming geometry uses the adapter-validated captured SVL, independently
of ordinary VL. This makes panel widths and their division/remainder constant;
standalone sources without that contract retain the public runtime query.
Unsigned casts of unit-step index sequences generate modulo-width indices
directly, preserving wraparound while avoiding unnecessarily wide vectors.
Moderate raw FP64 problems whose outputs fit half of L2 and whose inputs fit
twice L2 use one resident leaf instead of repeated cache-panel packing. The
leaf may pack its inputs once according to the same raw-input reuse policy.
Pointwise epilogue loads retain their capture at the original whole-matrix
store, including inline broadcast reads that alias an output row; row-wise
stores cannot substitute values already overwritten by an earlier row.
Availability guards retain vector fallbacks for automatic selection and reject
unavailable explicit instruction families. Low-level `T.amx`/`T.sme` APIs remain
available; public C++ operators are unchanged. Performance parity is measured
separately from correctness and instruction-family coverage.

In the native recipe, an automatic matmul with both operands carrying matching
physical packed descriptors and no explicit BM/BN/BK uses a whole-problem
leaf. This is a compile-time choice even when M/N/K are runtime values: the
native path and its automatic vector fallback share the same DSL policy,
so unused cache traversal and full-matrix partial buffers are not expanded.
Explicit block sizes and packing policies retain cache traversal. Arbitrary
logical slices of a packed operand do not inherit a physical-panel shortcut.
For native cache traversal with explicit B/AB dense packing, each N/K B panel is
built once outside the M loop and reused by all its M blocks. Instruction
packing uses a separate private interface: automatic AMX and grouped SME
cache traversal prepares a native B panel once per N/K phase and passes its
actual physical View to every M leaf. Private SME K-chunk padding and strides
are retained; these panels are not reinterpreted as flat external packed
storage or repaired as untrusted external padding. I16/I64 and SME group-one
FP32/FP64 retain their existing preparation policy. Explicit B/AB still uses
the dense cache policy above.

The automatic AMX raw BF16 path also selects bounded full-K B panels for large
K-contiguous B working sets, following the C++ WholeProblem reuse thresholds.
It requires raw A/B, aligned K, no explicit cache blocks, and a physical B
panel no larger than 1 MiB. Each panel spans 64--256 columns and is consumed
across all M rows before advancing N. Cache packers retain original layout
evidence without widening the resident AMX leaf's raw-layout dispatch. Both
schedules accumulate into private output and finish once, preserving captured
inputs, broadcast initializers, output aliases and whole-matrix epilogues.

Cache traversal, private accumulator initialization, K-phase accumulation,
optional dense packing and final epilogue application live in
`_library/matmul.py` and `_library/matrix_recipe.py`. `T.cache_bytes("L1"/"L2"/"L3"/"LLC")` queries the execution
CPU; an absent L3/LLC returns zero. Block sizes use positive bounded DSL
arithmetic rather than a target-side native scheduler. Generic packed-ABI
construction is in `_library/packing.py`, including zero padding and source
transforms; this conservative implementation can be slower than the retired
specialized packers. Packed offset/size decoding remains a physical ABI
primitive, not an opaque packing algorithm. Automatic register-transpose
traversal is also source DSL (`_library/layout.py`); hardware-unavailable,
unsupported-stride and alias cases use ordinary gather/scatter lowering.
The historical descriptions of native-family tuning and raw-packing selection
earlier in this guide are superseded by this section.

`broadcast_to` also accepts scalar expressions, filling the requested logical
shape with the scalar's dtype. This covers scalar initial accumulators without
inventing a synthetic rank-one tensor. Vector hints/prefix geometry are
retained when a staged value helper binds its caller's captured DAG.
Transposing a captured value preserves that exact read across later writes,
including transposes expanded inside a `T.func` or staged matrix recipe.
Memory-layout folding waits until full-kernel capture provenance is available.

### Matrix library: native panels, batch and reusable storage

`T.pack` retains the public packed ABI and exact required capacity. Native
AMX/SME targets now use source-level panel schedules, including transformed
producer values and fringe padding. Extra destination capacity is untouched.
A raw SME packing shortcut requires unit stride along the logical K dimension
(`A.strides[1] == 1` or `B.strides[0] == 1`); other layouts use the existing
producer staging path.
It also requires a disjointness proof against all relevant
kernel writes; otherwise it stages the captured producer before packing. This
preserves `old = T.load(X); T.store(X, ...); T.pack(old, P)` semantics.
SME public packing traverses complete K instruction chunks with constant
group counts and handles the terminal remainder separately. This preserves
exact public group capacity for K tails without combining unrelated loop
extrema in hardware bounds checks, including converted and bounded staging.

`pass_options={"cpu_pack_bounded": True}` opts converted SME public packing
into cache-block staging. It evaluates the original producer outside the SME
owner, then packs several instruction chunks inside one owner per cache block.
The staging allocation is at most 64 KiB for architectural streaming vector
lengths, with K blocks no larger than 1024 elements. The public group-padded
K pitch remains unchanged; private whole-chunk padding is not exposed. A
runtime disjointness proof covers all producer read roots and the packed
destination. Overlapping inputs retain complete capture before the first
write, preserving aliases and historical loads. The default is `False`;
native performance remains a criterion for enabling this policy.

Native packed AMX K tails keep complete external tiles in place and privately
repair only the terminal tile. SME keeps its equivalent terminal-group repair.
External padding has no numerical promise: masked lanes are selected to zero
before arithmetic, and caller-owned padding is never modified.

Logical rank-two slices of packed inputs may borrow native physical panels
when their spatial origin is panel-aligned and their K origin and length cover
complete instruction groups. The compiler must prove these facts; dynamic or
unaligned windows retain the general logical decoder. Borrowed views preserve
the original allocation's physical K pitch, even when the slice has a much
smaller K. Root density, capacity and output nonalias checks still precede all
native access. An aligned operand can be combined with a raw operand. Explicit
cache block or packing policies retain their requested traversal.

Automatic native matrix selection additionally admits fresh AMX/SME packed
small operands with K >= 32 and M*N*K >= 512. Every consumer must retain a
valid current physical-read proof; historical or cross-region captures use the
vector path. Raw byte inputs use native kernels only for M >= 4, N >= 16,
K >= 32 and M*N*K >= 16384 on x86. Raw SME products retain the existing
M >= 16 and N >= 16 native gate: small raw products can lose substantially
to vector execution once online packing is included with the source SDK.
Raw BF16/FP16 skinny shapes keep the existing
vector policy. A statically satisfied native shape condition omits the second
shape fallback while retaining hardware-feature fallback.

Private batched-result stores can fuse with an immediate epilogue when no
intervening or fresh aliased read can observe the removed store. External
loads captured after that store retain its ordering, even when their view is
not syntactically C but aliases C at runtime.

`T.library.batched_matmul((A, B), C, ...)` accepts rank-two shared inputs or
rank-three inputs with an ordinary batch axis (an extent of one broadcasts).
`C` is rank three. `strategy="serial"` executes independent products;
`"rows"` flattens batch×M for shared B; `"columns"` flattens batch×N for shared A.
`"auto"` uses optional `raw_inputs=(A_view, B_view)` layout evidence to choose
these folds. Layout hints do not replace the original producer expressions;
a logical reshape is not a promise of zero-copy execution. Complete private
result storage protects input/output aliases across different batches.

`parallel="serial"` is the default. `"auto"` requests a batch grid and retains
serial execution when ownership cannot be proved; `"require"` rejects a missing
proof. Parallel traversal requires `strategy="auto"` or `"serial"` and does not
apply the row/column folds. The private result is copied to `C` after the grid
joins, and each worker owns its internal scratch and hardware register state.
`shared_packing=False` is the default; `True` packs rank-two shared inputs once
before traversal. This is reuse within one invocation, not a persistent cache.
Native grid legalization accepts only known memory effects inside a lexical
hardware owner. Unrecognized effects, shared mutable owner state, or clipped
grid-dependent output partitions still reject required parallel execution.
On ARM SVE targets, compiler snapshots probe the selected compiler for the
scalable-vector common-store sinking crash. Only a reproduced failure followed
by a successful probe with the compatibility flag enables
`-mllvm=-simplifycfg-sink-common=false`. The same policy applies to C++ and DSL
builds, including serial kernels, and participates in artifact/cache identity.
Cache-only execution replays the attested choice without discovering or
executing compiler tools; missing or ambiguous policy records require an
online build with an unambiguous cache namespace.

Both `T.library.matmul` and `T.library.batched_matmul` accept `workspace=` with
exact result shape and accumulation dtype. Declare `T.workspace` at kernel top
level and pass its view; disjointness from inputs, initializer and output is
checked before use. Every invocation overwrites this storage. This reuses
capacity, not cached matrix values: changing input data still recomputes the
product. Explicit external prepacking remains the way to reuse packed contents.

`pass_options={"cpu_workspace_scratch": True}` additionally promotes eligible
top-level internal `T.alloc` sites to the existing replay workspace. Extents
must depend only on immutable tensor metadata and supported replay-keyed
properties. Loop-local, branch-local, payload-dependent and hardware-owner
allocations retain their previous lifetime. The default is disabled. Runtime
workspace planning/replay continues to control capacity, invalidation and
concurrent invocation ownership; a JIT compilation cache alone is not a
workspace cache.

Experimental schedule controls are kept separate from arithmetic semantics:

- `cpu_matmul_orientation="output"` retains the established column-output
  transpose policy. `"target"` uses proven AMX A/B layouts and shape costs,
  and lets eligible SME products use direct vertical output. `swap=False`
  remains authoritative.
- `cpu_packed_dot=True` enables the ordinary-SVE MMLA candidate for public
  SME-packed BF16 or same-sign byte AB inputs. It is guarded by ISA support,
  matching captured VL/SVL, small shape and K limits; unsupported combinations
  retain the previous leaf. It does not pack raw inputs or quantize them.
  Native BF16, I8 and U8 leaves, register-geometry composition, and automatic
  dispatch with its K-limit fallback have been validated on ARM. The default
  remains disabled: measured packed I8 2×2×65 and 4×4×257 are faster than
  the generic leaf, while BF16 2×2×65 is slower. Performance depends on the
  exact dtype, shape and K; this option is an explicit candidate.

Explicit associative tuple `T.reduce` can retain four independent accumulators
in one ordered traversal. The matrix GEMV recipe uses this to share a common
K vector across four output dots; unrelated automatic reduction fusion keeps
its existing register-pressure limit. AMX defaults and the explicit SME
`paired-fringe` schedule retain multi-tile interiors when a right or bottom
fringe is present, with fringe work inside the same instruction-owner interval.
The SME default retains its measured modulo-aligned dispatch.

`T.library.quantize_per_row_u8(value, multipliers, zero_point)` maps FP32 values
through separate FP32 multiply/add operations, replaces NaN with zero,
clamps to `[0, 255]`, and truncates into U8. Supply row multipliers with shape
`(M, 1)` and a shared scalar zero point; ordinary DSL broadcasting applies.
Positive and negative infinities clamp to 255 and zero respectively.
`T.library.dequantize_per_row_column(value, row_scale, column_scale)` requires
I32 input and performs I32-to-FP32 conversion, FP32 row-scale multiplication,
then FP32 column-scale multiplication, in that order. Row and column scales
can use shapes `(M, 1)` and `(1, N)`. These composable producer helpers do not
change generic cast semantics or automatically subtract integer zero-point
corrections; callers supply any correction in their integer accumulation.

`cpu_runtime_quant_int8=True` enables an experimental exact-pattern route for
`cast_u8(A_fp32 * row_multiplier + shared_i32_zero_point)` with native
SME-packed I8 B and an I32 accumulation domain. The route is selected when
M or N is below the existing native matrix threshold of 16; larger shapes
retain SME packing/MOPA, which outperforms row-wise grouped GEMV there.
The original conversion DAG is evaluated in bounded four-element K groups;
the last partial group quantizes only valid inputs. A private function boundary
keeps the grouped leaf independent of the larger native matrix fallback's
inlining and register allocation. It is not reassociated into FMA. Integer
correction remains modulo 2**32, and a fused dequantization DAG preserves its
ordered FP32 multiplications. Explicit FMA, unmatched transforms, packed
origins, and explicit cache/packing policies retain the original recipe. This
option never introduces quantization into an ordinary floating-point product.
It also recognizes the exact NaN/clamp wrapper of
`T.library.quantize_per_row_u8`, preserving those operations. The option
defaults to `False`; using either public helper does not enable it implicitly.

`cpu_matmul_native_schedule` selects experimental native schedules:
`"default"`, SME `"shared-owner"` or `"prefetch"`, AMX `"stream-b"`, and
AMX/SME `"paired-fringe"`.
The shared-owner leaf uses preallocated, typed packing inside the same lexical
resource owner as matrix compute when dtype/layout/alias proofs allow it;
other inputs retain the established path. These alternatives remain explicit
until target measurements justify a default. Instruction hints do not relax
logical bounds, padding, or alias checks. The paired-fringe candidate shares
one input across a 1×2 or 2×1 tail of an existing 2×2 tile body. It remains
explicit: enabling it globally regressed the dynamic 32×32×17 x86 raw-dot
benchmark by about 12% against the P3 baseline, despite that case having no
spatial tail. AMX default leaves retain their previous 1×1 fringe traversal. SME defaults
retain the original modulo-aligned 2×2/2×4 dispatch and use the one-tile
schedule for a nonaligned problem; the SME paired candidate additionally
enables a multi-tile body on nonaligned problems.

`cpu_outline_vector_fallback` is a boolean option, defaulting to `False` on
AMX targets and `True` on SME targets.
It outlines only the ordinary vector fallback in automatic native dispatch;
native resident and cache drivers retain their existing inlining. This can
change native code size and adds a private call when the vector fallback is
selected. Isolating a large recipe can reduce compiler work and register pressure
in packing/matmul owners. The fallback is not inherently cold: runtime shape
dispatch may select it frequently, and a disabled instruction feature can make
it the only active implementation. Profitability depends on the target, caller
and selected path; the default does not promise a speedup for every kernel.
Dynamic AMX fallbacks retain the compact contraction across either outline
setting; expanding grouped dot can regress both inline and private owners.
Static small products and explicit vector recipes retain the library's grouped
BF16/byte contraction. The SME guard for wide producers converted to bytes
remains independent of outlining. These choices do not alter the native tile
schedule or generic FP32/FP64 reduction rules.
An explicit boolean override remains available for target-specific experiments.

`T.sve.iota`, `join_k_groups`, `broadcast_segment`, `row_pair`, and `mmla` expose pure
ordinary-SVE register operations. They require complete scalable `pow=0`
registers and run outside SME owners. `iota(dtype)` returns the zero-based lane
indices in one native unsigned-integer register (`uint8/16/32/64`). Segment geometry is fixed at 128 bits;
MMLA supports BF16→FP32 and same-sign byte→I32 combinations. These are typed
instruction building blocks; the packed-dot traversal and output scheduling
remain ordinary DSL source.

### ARM scalable-store compiler compatibility

A compiler snapshot on an SVE host runs a bounded compiler probe once per
compiler identity and flag configuration. A reproduced scalable-size failure
selects only `-mllvm=-simplifycfg-sink-common=false`, after verifying that it
fixes the probe. Source C++ and generated DSL use the same selected flag,
which participates in artifact identity. Cache-only snapshots replay the
matching policy attestation from `cache_dir/identities`; they do not invoke
a toolchain. Missing or ambiguous attestations are explicit cache errors.
