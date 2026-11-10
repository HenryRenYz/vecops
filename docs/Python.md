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

CPU vector scheduling emits a single-use load/cast pair as one converting
load. Its mask, inactive values, and original read capture remain dependencies;
the unused pre-conversion memory read is omitted, including in masked tails.

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

Software-pipeline lookahead guards and double-buffer bank ordinals use
unsigned distances between signed loop endpoints. They do not introduce
signed overflow near int64 limits; an unrepresentable lookahead is omitted.

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
load masks and zero blends. The same proof applies to an incomplete final
source group when the exact output domain and a guaranteed SIMD lane-count
multiple establish that this group can only occur in the masked remainder.
Partial groups still sanitize the final transformed producer value before
the dot instruction; scalar projections retain their own bounds.

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
CPU recipe staging can specialize captured helper call boundaries within one
expansion, preserving the original shared helper and its capture identity.
After native AMX owners are exposed, online B-packing uses private helper
calls to bound compute-frame scratch lifetimes; independent packing kernels
retain their inline register transforms. Hardware regions stay in their
original caller, and private calls cannot cross an active hardware region.
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
Ordinary SVE/SME traversals split only their masked remainder into native-word
register fragments, accounting for wider load/store dtypes. AVX traversals retain
grouped masked remainders unless a constant remainder contains a complete physical
word, whose full predicate can then fold away. Large AVX-512 private-helper owners
(more than 64 control/storage IR nodes) and owners with hardware regions also split dynamic
tails to reduce register pressure across fallback paths. Large register-layout
owners with narrow floating arithmetic also split; their arithmetic widens inside
a physical word. Small maps and copy/conversion-only packers retain grouped
dynamic tails, avoiding duplicated fragment setup.
The internal `cpu_native_word_tails` option can
explicitly enable or disable fragment planning for target experiments. The default
uses the code generation target, independently of cost-model overrides. Full-loop grouping
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

On x86, repeated serial protection of the same retained read can reuse its
existing private snapshot owner when dtype, shape, strides and offset match.
An already captured value is kept; a previously uncaptured value is copied
only when the next original alias guard requires it. Alias checks keep their
original control flow and exception behavior. Parallel workers retain separate
owners. Other CPU backends retain their existing snapshot construction policy.

The CPU memory stage's `cpu-plan-entry-geometry` analysis identifies geometry
derived solely from immutable tensor arguments. On x86, serial loops may reuse
such overlap predicates within one invocation. The first check remains at its
original execution site: empty loops and untaken branches perform no new span
evaluation, including checks that can throw. Snapshot contents are refreshed
normally; only the predicate is reused. Mutable coordinates, scratch storage,
parallel workers, hardware owners and conditional geometry substitutions stay
outside this policy. At most sixteen predicates receive invocation-local state.

Allocation liveness is collected once in the CPU storage plan over the final
executable value DAG, including view metadata and historical read provenance.
Unused declarations alone do not keep a scratch allocation alive. Source emission consumes
this verified set instead of rewalking the entire kernel at every allocation.

The CPU memory stage's `cpu-plan-value-projections` analysis recognizes reshape
suffixes that preserve immutable entry-derived dimensions, including dimensions
whose spelling changes through scalar captures. On x86, a vector wholly inside
such a suffix keeps its original producer and historical snapshots, and maps
only the changed prefix coordinates. Empty or inactive traversals perform no
new prefix division; inactive lanes remain zero. Changed suffixes, opaque or
mutable shape metadata and explicit physical vectors retain ordinary lowering.
The verifier recomputes positive projection certificates against the final IR.

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
Analyses declared preserved remain cached across callbacks and nested groups
within the same stage. Entering or leaving a different stage clears the cache;
per-invocation option overrides also isolate their analyses.

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
Operation definitions can supply a pure `fold_guard` over typed literal IR
operands. The constructor invokes the host folding callback only when that
guard approves, and frozen catalogs include the guard in their identity.
Constant signed-int64 `ceildiv` uses this contract to expose exact grouped-dot
extents without an overflowing numerator-plus-divisor calculation. Unsigned,
narrow, floating-point, zero-divisor and signed-overflow cases retain their
original operation; no integer promotion or trapping behavior is inferred
from host Python arithmetic.
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
Opacity checks stop the structural walk at expression roots and delegate their
arguments and owned metadata to the memoized semantic query. Shared DAG tails
are not expanded again for each enclosing statement. Mutable control bodies
are still inspected after rewriting, so unknown effects continue to invalidate
facts at the correct program point.
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
Positive constant ceiling division participates in the shared signed integer
range analysis, including negative numerators and integer-limit endpoints.
Scheduling uses these facts for constant-shaped grouped vectors even when
their original type metadata retains a `ceildiv` expression. Zero, negative,
unknown, unsigned and narrowing-sensitive division stays conservative.

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

`cpu-plan-memory-proofs` also intersects total, stable store geometry over
lexical occurrences. `cpu-plan-store-alignment` uses those certificates for
pointwise stores on AVX-512: sufficiently wide, unit-stride rows may
start with a masked prefix, continue at native-word-aligned addresses, and end
with the existing masked remainder. Pointer arithmetic stays inside the
original nonempty row and all coordinates retain their logical offsets.
Historical reads and alias snapshots retain their original evaluation owners.
Explicit vector contracts, reductions, register transforms, masks, converting
stores, worker/hardware scopes and unproved geometry keep their schedules.
The final verifier rechecks both the lexical proof and its exact vector-region
owners; printing consumes the prepared schedule. Fused outputs share the prefix
only when their runtime row addresses have the same native-word alignment
remainder; incompatible addresses use the existing unaligned traversal.
Grouped native words retain their original register geometry.
`cpu_align_stores=False`
disables this policy. The prime-width cases in `dsl_loop_facts.py` and
`dsl_compiler_optimizations.py` exercise split rows with fixed buffer layouts.
The same `cpu-plan-memory-proofs` lexical traversal prepares root-alias checks for bounded, ordinary
serial loops with short constant pointwise tiles on AVX-512 and SVE/SME. At least two
iterations are required before checking roots. A successful
input/output root check supplies scoped disjoint facts to the original loop;
its fallback preserves the original snapshots, checks and iteration order.
Output/output aliases are not assumed independent. Root-span predicates are
nonthrowing: invalid or overflowing descriptor arithmetic selects fallback,
so a zero-trip loop does not acquire an earlier metadata exception. The
planner admits ordinary serial loop nests with bounded headers and excludes
worker/hardware, software-pipeline and operation-major unroll scopes,
allocations, state, explicit vectors and unproved body geometry. An enclosing
version supplies facts to its entire nest; nested versions are suppressed to
bound code size. Captured integer-to-float metadata stays in its original
loop body. Guards and facts use the same logical view-root
shape, strides and offset; shared loops with different normalized origins
retain the original schedule. The final verifier rechecks the
exact loop, stores, required external root origins and every lexical occurrence.
SVE/SME profitability uses the architectural minimum of 16 bytes, independently
of captured ordinary or streaming vector lengths. Hardware and private early-exit
owners retain their original schedules on these targets. Other targets and
wide streaming tiles retain their existing schedules; `cpu_loop_versioning=False`
disables this policy. The short reverse-range and invariant-branch cases in
`dsl_loop_facts.py` and `dsl_compiler_optimizations.py` exercise these checks.

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

Qualified application discovery providers may return a separate immutable
`result.planning_graph` and defer decoding the full IR until `result.graph` is
first accessed. The latter still denotes the complete trace. Frozen decoding
and `ExecutionGraph.planning_packet()` / `from_planning_packet()` preserve a
canonical digest while validating each newly decoded graph. Successful
validation is reused only for recursively immutable snapshots, never mutable
graphs or serialized validation flags.

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

`graph.event_execution_weights(capture_or_graph)` reads closed loop summaries
and returns exact `fractions.Fraction` weights keyed by stable event ID. The
first observed sample and explicitly marked prologue/epilogue samples each
count once; other samples share the remaining declared iterations, with a
minimum weight of one. Nested sample weights multiply. For example, a declared
seven-iteration loop with a prologue and one steady sample gives their events
weights `1` and `6`; nesting a five-iteration loop gives steady-body weights
`4` and `24` in those two outer samples. Symbolic/nonpositive trip counts are
left unexpanded. This read-only helper accepts `GraphCapture` after its loops
close, ordinary graphs and immutable snapshots; it does not execute a loop or
establish that two samples have interchangeable effects. Memory traffic queries
use the same helper, so adapters can carry logical call counts from the shared
graph facts without repeating the planning algorithm.

`mark_retained` is intended for Python-side caches that outlive a tensor's last
observed operator access. It extends the tensor and its complete alias storage
group to graph exit without classifying the value as a model output.

### Native CPU replay from the same GraphCapture

`GraphCapture` also records a real eager invocation into an executable tensor
program. This extends the existing capture mechanism; precompile/manual graph
captures retain their metadata-only behavior.

```python
capture = vecops.graph.GraphCapture("denoiser-step")
captured_output = capture.capture_torch(model, positions, noise)
replay = capture.compile_torch()
next_output = replay(next_positions, next_noise)
```

Capture executes the function once, including its input/state mutations. CPU
RNG state is restored after capture by default, including capture failure;
`preserve_rng_state=False` opts into consuming capture's random draws. Replay
executes recorded standard ATen CPU random kernels against the current default
CPU Generator, advancing its state normally. `torch.manual_seed` therefore
reproduces the same replay sequence. Explicit Generator arguments are rejected
before capture executes their operator.

The executable uses Torch's C++ `Code`/`InterpreterState`, with direct,
token-checked CPU `SafeKernelFunction` bindings. There is no Python node loop,
model callback, profiling executor, fusion pass or dead-code elimination.
Void native calls and writes to externally supplied output buffers remain in
recorded order. ATen kernels must have native packaged registration evidence;
materialized vecops registrations and their modules remain strongly owned.
Other native custom operators require `native_operators={qualified_name: owner}`.
Python CPU registrations, tensor subclasses, active Python dispatch/function
modes, hidden functional/pre-dispatch transformation contexts and outer CPU
autocast contexts are rejected explicitly. Dispatcher registrations, including
operators called internally by an accepted native kernel, must remain stable
for the executable's lifetime. Dynamic registration replacement is unsupported.
For each recorded node, an additional saved native binding either remains valid
or throws its invalidation error if its own CPU registration is replaced; it
cannot redirect that node's binding into Python. This does not claim to detect
arbitrary changes to an unrecorded nested dispatcher call.

Pytree structure and non-Tensor arguments are static. Native guards validate
CPU strided layout, shape, strides, dtype, storage offset/capacity, supported
dispatch keys, StorageImpl alias groups and byte-overlap geometry (including
separate `frombuffer` storages). Guards deliberately reject some otherwise
compatible standalone copies of views if their storage capacities differ.
Data-dependent output shapes, Python scalar extraction, NumPy/list data escapes, resizing/rebinding
storage, asynchronous operations and indirect Python calls are unsupported.
An unsupported capture raises, clears its capture context and cannot compile;
replay never silently runs an eager fallback. A kernel error clears replay's
execution locks, but earlier in-place writes/random draws are not rolled back.
Failed compilation restores the capture's ownership state and releases temporary
arena allocations. To avoid retaining those allocations through a saved exception,
inactive compiler-frame locals are cleared; error text and traceback locations
remain available. The same successful capture may be compiled again.

With `native_pointwise=True`, floating results of multiplication can also
consume a boolean scalar or a mask whose final dimension is statically one.
The boolean is converted to 0/1 and broadcast over SIMD feature lanes. This
preserves multiplication semantics (including `0 * NaN` and signed zero),
rather than replacing it with a mask-select operation. Plain bool-to-floating
copies/conversions with that layout are supported too. General boolean inner
vectors and boolean output operations keep their native ATen implementations.
The restriction is checked before kernel compilation; no vector load of the
SDK's bool element type is required.

An enabled `native_thread_pool_region` is captured as native begin/end events in
the same tape. Compilation retains the actual provider DSO and validates its V1
factory and activation C ABI; a provider without that evidence raises explicitly.
Replay invokes the C ABI directly and unwinds outstanding regions on a kernel
error, without calling Python region wrappers. This preserves an explicitly
enabled region's scheduling hint; it does not enable regions disabled by a model.

Per-node float32 matmul precision is captured, including temporary precision
changes inside a Python module. Native execution restores the caller's precision
on success and error, and does not set it when already correct. Graph execution
rejects reentry/concurrent calls, including capture versus capture/replay and
calls through a different graph in the same process, to protect Torch's
process-global math policy and capture's default-RNG fork. Concurrent
external mutation of Torch global configuration, dispatcher registrations, the
default Generator during capture, or live tensor contents is unsupported.
A factory's omitted dtype is normalized
to its observed result dtype so a later default-dtype change cannot change its
output type.

Only graph inputs and first-used external bindings retain tensor owners during
capture. Produced temporaries follow eager lifetimes; capture retains their IR
metadata rather than all intermediate payloads. Capture/recorder ownership avoids
reference cycles, so releasing an executable does not require cyclic GC to
return its owned physical roots. External inputs and module state retain their
explicit storage owners. In-place updates remain visible; rebinding a module
attribute or `.data` requires binding a new graph.

`capture.prepare_torch(...)` performs pointwise lowering, donation and native
workspace description before allocating storage. It returns one `TorchGraphPlan`
whose `memory_requirements()` describes the final optimized storage identities.
`plan.compile(memory_program=prepared, memory_call=...)` binds that exact plan;
it cannot repeat graph rewrites or silently add a second tensor/workspace arena.
A plan binds once; prepare another plan for another executable. Ordinary
`capture.compile_torch(...)` runs the same preparation/binding sequence.

Graph-owned factory and supported pointwise outputs use the common resource
program. Small tensors share page slabs with aligned byte offsets. Absolute
storage offsets on relocated graph buffers are rejected; a relative
`as_strided(..., storage_offset=None)` must remain inside its proven source span.
Borrowed outputs can be overwritten by the next call; clone values that need
independent contents. Escaped Tensor storages retain the actual native mapping
and backing owner after explicit program closure.

Functional ATen results without explicit output buffers remain framework-owned.
They appear in `plan.framework_requirements`, separate from managed requirements.
`plan.framework_budget_recipe(pool, participant=...)` creates a backing-only
pending-budget recipe for a caller-selected concrete pool. Reserve it without
acquiring backing; it reserves capacity and does not claim to place Torch malloc
pages. Actual framework allocations already reflected in OS free samples can
therefore be conservatively counted again while this lease remains live.

Cold distributed preparation must also close transport capacity before graph
admission. `PreparedProgram.planned_placement(...)` inspects the offset-only
proposal without claiming resident pages; `placement(...)` describes materialized
backing. `TorchGraphPlan.configure_communication_placements(...)` selects routes
from the proposal, and `vecops.graph.provision_communication_capacity(plans)`
provisions each distinct stopped transport provider collectively. A no-growth
fixed point is retained with `finalize_communication_contract()` before final
compilation. `GraphMemoryCompiler.prepare_model` accepts `finalize_resources` and
`preparation_consensus` callbacks to coordinate this ordering and all-participant
admission retries. Successful grants stay held while peers replan; a collective
finalizer runs only after all proposals are ready and all graph grants are
withdrawn at a barrier. It retains the offset plans and reserved VA unless
resource growth requires recompilation. Fatal failure refunds the local grant;
see `docs/WorkspacePlanning.md` for the complete sequence.

Each compilation freezes its budget scalars and reuses only exact placement and
cost-query results within that attempt. A fresh admission snapshot rebuilds
those caches. Final runtime estimates report capacity shortfalls by group, pool
or shared constraint, including exact requested and available byte counts; the
planner retains these scalar facts in a capacity error for auditing.
Group and constraint shortfalls compare newly required backing after cache reuse;
pool shortfalls compare total requested bytes. The available count is the one
used by that estimate, rather than a later snapshot.

During `prepare_model`, a typed group shortfall can also teach a temporary
fresh-allocation margin equal to the observed physical sample loss. Learning
requires identical accounting and untruncated pool allowances. A new grant,
constraint or independent pool limit clears the margin. Idle cache remains
available, public snapshots and runtime limits stay unchanged, and all temporary
policy state is restored on success or failure. Once a typed physical rejection
establishes that observation, every untruncated group's measured decline across
the same-accounting snapshots can contribute its own maximum margin. No margin
is summed across samples or copied between groups. One compiler instance
prepares one model at a time. Compact fallback releases the preferred failure's
traceback and layout metadata before compiling its replacement.

A known unaccepted shared physical-free reserve failure carries primitive
evidence from its last actual snapshot. `prepare_model` can learn its sample
loss after a local-grant rollback advances epoch, provided compilation, reserve
and current snapshots have identical complete nonphysical accounting. This
exception does not waive changed peer balances, alias-pool quotas or overlapping
constraints. The helper preserves its `RuntimeError` category and message,
retains no native snapshot or original exception context, and does not itself
retry a capacity rejection.

The thin Torch-only executor is compiled at the first executable capture against the installed Torch
version and C++ ABI. It does not link a second OpenMP runtime or an SDK library.
`VECOPS_GRAPH_REPLAY_CXX` selects an explicit compiler and
`VECOPS_GRAPH_REPLAY_CACHE` selects the private binary cache. Frozen C++ source,
compiler/version, Torch/ABI and include/library paths form the cache identity;
interprocess locking protects cache construction. First-use native module build,
graph capture, graph lowering and steady replay costs should be measured
separately.

### Resource-graph memory planning

`vecops.planner` consumes framework-neutral storage requirements and the public
C++ memory query interface. Agents, pools, capacity groups and transfer methods
have opaque identities. The same algorithm supports any number of memory pools;
no CPU hierarchy or fixed tier count appears in its policy data structures.

```python
from vecops import memory, planner

runtime = memory.default_runtime()
view = runtime.view()
consumers = tuple(agent.id for agent in view.graph.agents
                  if agent.id in view.executable_agents
                  and 'host_load_store' in agent.capabilities)
plan = capture.prepare_torch(native_pointwise=True, borrow_outputs=True)
requirements = plan.memory_requirements(consumers=consumers)
compiler = planner.GraphMemoryCompiler(runtime, view, runtime.snapshot(view))
program = compiler.compile(requirements)
prepared = program.prepare(runtime, view)        # virtual addresses only
prepared.materialize(external=external_tensors)  # admit roots and map pages
replay = plan.compile(runtime=runtime, memory_program=prepared)
```

Materialization uses `planner.reserve_recipe_latest(runtime, view, recipe)`:
each admission attempt takes a fresh capacity snapshot. Only an explicitly
rejected stale epoch is retried, up to 64 attempts; capacity, permission and
validation errors propagate immediately. Callers reserving framework-only
budget recipes can use the same helper without acquiring physical backing.

For a framework allocator whose actual policy permits several pools, use
`lease, recipe = planner.reserve_budget_latest(runtime, view, byte_count,
pool_order, native_module=None, attempts=64)`. The exact uint64 byte count is
reserved without physical backing or virtual mappings. Each attempt rebuilds
the recipe from a fresh snapshot, greedily using the caller's ordered opaque
pool IDs and rounding each share up to its pool page quantum. The returned
recipe records the full charge, which may exceed `byte_count`. Alias pools
share one capacity-group balance; physical free, hard/shared headroom and every
overlapping constraint remain binding. Managed idle cache is not malloc capacity.
Only explicit unaccepted stale/capacity rejections allow bounded replanning;
permission, invalid-recipe and uncertain-commit errors propagate. Insufficient
ordered packing fails without a reservation and does not establish global
infeasibility under other pool orders. This helper neither selects a hardware
domain nor forces framework malloc placement: callers must prove that all
listed pools belong to the allocator's actual permitted domain and retain the
lease for the budget's lifetime. The static `reserve_recipe_latest` contract is
unchanged.

Requirements collapse tensor aliases into storage identities. Placement uses
runtime-authorized mapping capabilities and measured/reference cost facts;
unknown costs retain an explicit conservative flag. Pool capacity, unique
physical capacity groups and overlapping domain caps all constrain admission.
Integer-offset packing reuses only compatible placements with nonoverlapping
lifetimes. Weighted page patterns retain their stripe phase; exported and
independently mapped buffers receive exclusive page envelopes.

`compile_model(programs, calls, serialized=True)` jointly compiles multiple graph
calls. `ProgramCall.bindings` links local storage identities to model identities,
including producer outputs, consumer inputs and retained state. Transient peaks
share roots across proven serialized calls. Retained native workspace remains
live throughout the model program. The caller specifies consumers and output
retention; these are scheduling facts rather than memory-model hierarchy rules.

`planner.audit_plan(program)` reports offset-only bindings, roots, initialization
copies and group charges. `program.dumps()` emits canonical TOML without process
addresses or handles. The command-line entry point accepts a captured graph,
resource hardware/policy files and optional opaque pool/agent selections:

```sh
python -m vecops.planner model.graph.json --hardware machine.toml \
  --policy memory-policy.toml --output model.memory.toml
```

Use ordinary `torch.empty`, `torch.empty_like` and `tensor.contiguous()` in
captured code. Their ATen dispatcher operations and storage lifetimes remain
visible to the graph and are bound by the prepared resource program. Eager
resource-backed allocation uses `memory.Pool`.

## Framework registration

Compiled templates select NumPy execution or a generated Torch bridge from
the runtime tensor arguments. Torch bridge tasks are content-addressed and
compiled on the first real call or at the end of `vecops.precompile()`.
Registrations sharing a bridge artifact resolve to the backend-published
filename, including its platform-specific library prefix.
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
It does not configure barriers inside a joined native team. Those actively
spin by default; `VECOPS_THREAD_POOL_TEAM_SPIN_COUNT=2048` restores short
spinning followed by parking (`0` parks immediately, `4294967295` is the
active default). Set it before pool creation. Leaving the joined team returns
workers to the bounded idle-wait policy; their OS threads remain persistent.
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

## Memory resources and eager pools

When `VECOPS_BUILD_MEMORY` is enabled, `vecops.memory.Runtime` owns the C++
resource graph, capacity authority, arenas, mappings and completion service.
`Runtime.open(hardware_file, policy_file=...)` binds a machine-independent TOML
template to current backend inventory and process grants. `Runtime.discover()`
provides a generic Linux CPU topology when no template is selected.

```python
from vecops import memory
runtime = memory.Runtime.open('machine.toml', policy_file='memory-policy.toml')
pool = memory.Pool.create(runtime, 'working-domain', allowed_pools=['chosen-pool'],
    hard_limit=0.8, grow_by_bytes=16 << 20)
with memory.use_runtime(runtime):
    tensor = pool.empty((1024, 1024), dtype=torch.float32)
```

Domain and pool names come from the configured resource graph. Arena caches
retain already charged roots, and suballocations do not reenter global capacity
admission. Tensor storage and aliases keep their native allocation alive.
`Runtime.contract()` returns immutable resource/grant/policy facts for prepared
artifact guards, excluding addresses, current free capacity and epochs.
See [Memory](Memory.md) for C++ ownership, sharing and mapping contracts.

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
`benchmarks/dsl_compiler_lowering.py` separately measures fixed dot/matmul source
lowering, including outlined SIMD fallbacks. It starts each sample from a fresh
template IR, records elapsed time and generated-source hashes, and excludes
native compilation/execution. Use the same target, fallback option and pinned
CPU when comparing snapshots; it does not select an implementation.
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
rebinding can escape directory notifications. A tree scan validates each
directory before scanning its entries, without repeating the same ancestor
resolution for every header; each leaf is still checked and read afresh.
Cold compilation validates its
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
native resident and cache drivers retain their existing inlining. When disabled,
the proven-disjoint SIMD branch stays inline and only the general alias-preserving
branch uses a private dispatcher. Explicit vector recipes retain inline dispatch.
Whole-fallback outlining keeps its single boundary without a nested alias call. This can
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

Generated translation units that emit an AMX hardware owner request inline
scalar prefix-mask normalization from the x86 backend. This avoids outlined
metadata calls and their vector spills in large packing/matmul owners. The
emitter selects this policy from emitted hardware regions; ordinary SIMD
recipes and SME kernels retain their existing normalization. Empty, negative,
full-width and oversized counts preserve the same saturation semantics.

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
For Clang, the deliberately failing probe disables crash reproducer files while
retaining stderr in its attestation. This diagnostic option is confined to the
probe; selected kernel flags and existing cache policy attestations are unchanged.


### Checked reshape span provenance

A dense, dtype-preserving `T.reshape` retains its checked source span as well
as a new logical coordinate system. Bounded slices and transposes preserve
that enclosing physical span. A dominating `T.view_disjoint` can therefore
remove redundant conditional snapshots inside a reshaped view. This does not
establish affine grid independence or change load capture timing.

The proof checks both guarded windows, their element byte widths and physical
root origins. A guard for a short prefix does not cover an entire parameter;
two windows of one storage must match the two distinct guard roles. Only
immutable ABI metadata and scalar definitions with immutable dependencies can
be equated through captures. Mutable or unknown coordinates, different origins
and unproved spans retain conservative overlap checks and snapshot ownership.

Validated, nonoptional tensor parameters share their exact `T.Const` int64
shape and stride facts with integer analysis and guarded span proofs. Thus an
ABI descriptor's `size(axis)` or `stride(axis)` can match the same static
logical root, avoiding redundant snapshots in fused SIMD loops. Dynamic
observations and optional descriptor metadata do not establish these facts;
the dominating guard still has to cover both actual physical windows.

### Metadata analysis cost

Integer definition expansion preserves untouched expression identities and
shares rewritten argument DAGs within one query. Its cache includes the set
of excluded definitions, retains its expression inputs, and expires before
returning; definitions and branch facts never carry across queries. Shape and
provenance payloads keep their existing substitution rules. Affine membership
walks owned metadata only when a payload can contain IR nodes.
Canonical integer term keys intern child numbers within that same traversal,
so shared nonlinear metadata is neither expanded into recursive tuple keys nor
repeatedly hashed as a tree. Named affine symbols retain their namespaces;
opaque and effectful terms retain identity, and each traversal owns its catalog
and strong expression references.

`benchmarks/dsl_compile_latency.py` measures fixed matrix-catalog groups through
public `T.compile` and `generated_source`, without requesting native kernel
builds or execution. It reports wall and process CPU time, source hashes,
compiler input hashes, and CPU frequency policy. Imports, template construction
and compiler policy initialization are outside each sample. Repeats must emit
identical C++; compare snapshots using the same plan, CPU and frequency epoch.

Dead scalar cleanup removes an unused pure reassignment together with its
scalar declaration, including updates inside structured control flow. A
trapping or effectful reassignment retains the declaration it requires even
when its resulting value is unread; cleanup preserves its original execution
path and exception timing.

#### Resource-program tensor ownership

Captured external inputs and module state retain their explicit storage owners.
Graph-owned storage is compiled to integer byte offsets in a resource program;
relative tensor aliases preserve their canonical storage identity and strides.
Returned tensor storage pins both its virtual reservation and actual physical
binding, including aliases made after replay. Closing a prepared program cannot
free physical pages still owned by an escaped tensor.

Readonly state relocation is explicit: mark its `BufferRequirement` with
`may_redirect=True` and a selected `PlacementPlan`, then pass the canonical
source tensor to `PreparedProgram.materialize(external=...)`. Initialization
copies happen once during materialization. Mutable state needs an explicit
commit policy and cannot use this readonly relocation path.

Storage identity and data escapes remain rejected during capture. Replay has
no Python profiling callback or hidden allocator session.


### Prepared C++ source nodes in Torch replay

Sources defining `VECOPS_RECORDABLE_KERNEL` are lowered to persistent grid
plans during `GraphCapture.compile_torch()`. Check
`replay.allocator_policy['recorded_cpp_plan_nodes']` for the number selected.
`replay._executor.native_islands()` is a diagnostic count of multi-operator
resident islands. Set `VECOPS_THREAD_POOL=native` before source registration
to select a resident provider; the ordinary Torch provider may expose logical
tasks only and therefore use synchronous individual grids. Replay keeps native
plan leases, not Python source callbacks. Unconverted sources continue using
the existing native boxed path.

`capture.compile_torch(native_pointwise=True)` optionally lowers supported
floating `to.dtype`/`_to_copy`/clone, add/sub/mul/div, negation and ReLU intermediates into
recordable SIMD C++ grids. Broadcasting has fixed captured metadata; input
addresses and Tensor scalar values remain dynamic. Unsupported dtype/layout or
overload combinations keep their native ATen implementation. Each original
operation still stores its result in the captured dtype, including BF16 rounding
boundaries. Escaping functional results continue using ATen allocation.
Supported in-place arithmetic and `copy_` retain their target storage and alias
semantics. Same-dtype copies, including singleton-feature broadcasting, preserve
raw element bits; readonly `const` inputs do not introduce an FP32 conversion
that would quiet signaling NaNs or alter their payloads. Kernel artifacts are prepared once for each metadata/VL/SVL/worker
specialization; each executable still owns independent scratch and plan frames.

With this option, internal nonpersistent, nonescaping factories and lowered
results share a private static scratch arena. Slot reuse uses the complete
storage lifetime, including aliases, and never reuses an input's last-use slot
at the event producing a new result. Immutable bound views are constructed at
compile time. Views of dynamic inputs are evaluated before computation and
follow rebound addresses. Planner input/binding overlap checks use actual live
byte spans; disjoint views of one arena owner require no staging copy.
Integer-argument `narrow` views follow the same guarded geometry rule.
An empty view without an addressed anchor retains its runtime operation.
An internal storage declared as zero bytes uses one metadata-only empty
Storage per storage identity. Its empty aliases preserve their strides and
original storage offsets without requesting backing pages or planner capacity.
A nonempty view of a zero-byte storage is rejected. Empty slices of a nonempty
storage continue to use that storage's planned binding.
Tensor-valued `narrow` starts remain runtime operations because their offsets
depend on data rather than input metadata.
Recorded `to`/`contiguous` calls that returned the exact input tensor are
eliminated under these static metadata guards, so they do not split an island.
The C++ executor also moves `as_strided`/relative static views whose tensor
roots are graph inputs and whose metadata is literal into the submission
prologue. They still follow rebound input addresses on every call; their list
construction and view binding occur before all compute, allowing the native
team to advance across the original view locations. Data operations and mutable
calls retain their original order. `executable._executor.prologue_views()`
reports this additional preparation count. Scalar views retain typed empty
shape/stride lists during capture, including after input address rebinding.

Allocator diagnostics include `native_pointwise_nodes`, `hoisted_static_views`,
`eliminated_static_noops`, `static_factory_count`, `static_scratch_arena_bytes`
and `native_island_sizes`.
The latter lists all resident submissions, including singleton plans. The older
`native_islands()` count includes multi-plan islands only. Torch CPU profiling
also exposes `vecops::kernel_plan`, `vecops::plan_bind`,
`vecops::native_island` and `vecops::resident_team` scopes; their nested times
must not be summed as independent wall contributions. Compilation, source
attestation and scratch allocation are setup work, outside steady replay.


### Optional NUMA-group dynamic-library loader

The independent Linux/glibc `vecops_numa_loader` CMake target is enabled with
`VECOPS_BUILD_NUMA_LOADER=ON` (default OFF). Set `LD_AUDIT` to its installed DSO
and configure `VECOPS_NUMA_PLACEMENT` before exec. It selects independent,
content-verified library files for each configured NUMA group; members share
one inode. It never changes page policy/protection and does not intercept
kernel launch/replay. See [the loader guide](NumaLoader.md) for configuration,
path compatibility, diagnostics, and deployment.

The former `prepare_code_locality` COW/page-migration API has been removed.
File isolation replaces that deployment path, but does not guarantee physical
page placement: verify matched model timings before adopting a sharing group.
Torch GOMP provenance checks retain a NOLOAD handle to the actual mapped DSO;
a loader copy must prove both its source association and exact content, and all
required symbol addresses must belong to that same object.


## Independent CPU communication

`vecops.distributed` provides an optional Linux coherent-shared-memory c10d
backend. Its C++ transport (`vecops::comm`) has no Torch, Kutacc or KUPL
link dependency. The Torch adapter is built lazily against the installed Torch
headers and ABI; an available C++20 compiler is required. Importing configuration
types alone does not import Torch. Install the `communication` extra for Torch
and, on Python <3.11, TOML parsing.

Initialization passes the active runtime through `runtime.native.export_bridge()`.
Its named capsule retains a native owner and validates the runtime interface and
C++ standard-library ABI, independently of each extension's pybind registry.
The backend retains that owner after initialization; no raw runtime integer or
Python type-registry cast is used. Capsules are process-local and cannot be
saved as graph or configuration data.

```python
import torch.distributed as dist
from vecops import distributed as vd

vd.register_backend()
dist.init_process_group("vecops", pg_options=vd.Options.from_file("communication.toml"))
group = vd.group()
full = group.all_gather(local, dim=0, hints=vd.Hints(reuse="once"))
other_shard = group.reshard(pair, source_axis=0, target_axis=1)
```

Run this under `torchrun` or another launcher that sets the usual rendezvous and
rank environment. Bind each process and install its `memory.Runtime` before
initialization. All ranks on one host must use the same private job capacity
directory; independent or different capacity authorities are rejected.
`VECOPS_MEMORY_CAPACITY_DIRECTORY` configures the default runtime. Source
snapshots, required relay replicas and PageExchange storage all allocate real
roots through that runtime. Their imported aliases create no second charge.

`VECOPS_COMM_POOL` optionally selects an opaque allocatable pool ID. Otherwise
public host-agent access costs choose among pools with shared page mapping
capabilities; incomplete costs retain deterministic resource order. Runtime
backend bindings determine physical placement and the page quantum. A separate
portable communication policy (`VECOPS_COMM_CONFIG`) controls thresholds and
search; it contains no machine hierarchy or memory-kind definitions.

The optional adapter links `libvecops_memory.a`. By default it first checks the
selected SDK's `lib`, then `_sdk/lib` beside the active `vecops._C` extension.
Thus an editable installation can use checkout headers and the installed runtime
archive without an environment override. A source extension in CMake's
`build/python/bindings` layout can also use that same build's `src` archive; no
other ancestor directories are searched. Installed wheels carry the archive in
`vecops/_sdk/lib`.

For a separately copied source/frozen extension, `VECOPS_MEMORY_LIBRARY` explicitly
selects its matching archive. An invalid or empty override raises an error instead
of falling back to another build. The selected archive's content hash remains
part of the communication artifact identity.

Potential relay windows reserve address space but allocate only segments required
by registered plans. Growth retains old root receipts until all ranks acknowledge
the replacement generation. Stable replay uses the existing shared control
protocol without capacity-file transactions. Normal process-group shutdown
drains local work, unmaps imported aliases and waits for all-rank stopped
acknowledgements before refunding roots. An abort without that proof retains its
receipts conservatively; dead owners remain visible to the authority.

The `communication-policy/0.1` TOML is independent of the machine template:

```toml
schema = "communication-policy/0.1"
[communication]
algorithm = "auto"
relay_threshold_bytes = 1048576
minimum_gain = 0.05
sync_warmup = 3
sync_rounds = 7
# calibration = "routes.json"
```

`CommunicationCompiler` exchanges public rank-agent grants, payload pool IDs,
actual launch worker counts and resource connections at group initialization.
It derives nested partitions from shared directed fabric cuts. Arbitrary pool,
agent and resource names work; no hierarchy labels or memory kinds are parsed.
Unequal group sizes and unequal directed rates are supported. A resource local
to one branch can be nontransitive across the initial full group: defer that
resource while another proven cut establishes its enclosing branch, then
recheck it within the refined groups. If no proven refinement can advance and
an unresolved resource remains, discard the hierarchy conservatively. Crossing
valid cuts and unprovable routes retain the flat candidate and report
`missing_facts`; local deferral does not manufacture their intersections.

For a known materialized AllGather layout, cold graph preparation obtains final
placement from its materialized `PreparedProgram`, or observes fixed external
CPU pages through `Runtime.observe_external`. Dynamic invocation inputs remain
unknown; metadata examples and framework budget declarations are not placement
proof. Costs include source publication, replica reads and writes, and actual
final output writes. Each enabled copy stage assigns remote sources in rank
order to the destination member with least accumulated bytes, then synchronizes.
The native lowering uses these same partitions and assignments.

Bandwidth loads are evaluated by the public memory cost model. A bounded cold
native-barrier sample supplies stage overhead (max across ranks, then median),
or `stage_us` supplies an explicit microsecond value. An unknown publisher input
read may be cancelled when it is exactly common to all routes; total cost still
reports `quality="incomplete"`, while `comparison_complete` can be true. Unknown
candidate-specific writes or routes force flat selection. `statistics()["routing"]`
contains derived partitions, overhead samples, missing facts, candidates and
selected modes. Capture performs the search once; hot native replay has no
Python planning, residency queries or capacity-file transactions.

`algorithm="flat"` disables automatic replication. `Hints(copy_levels=(0,1))`
provides an explicit route for the generated coarse-to-fine partitions. The
relay threshold remains independent; to measure small copy candidates use
`relay_threshold_bytes=0` in both tuning and deployment policy.

Empirical calibration may replace the reference cost comparison.

The memory runtime's optional TOML overlay is separate from communication's
consumer-specific calibration. `native.open_runtime(hardware_file,
calibration_file=..., options=...)` loads the supported
`memory-calibration/0.4` subset: access read/write curves, copy transfer
curves/cases, and copy write coefficients. Unsupported compute/triad/fallback
models are errors rather than ignored fields. The complete historical design
examples contain additional models; they are not interchangeable with this
executable subset. See `docs/MemoryResourceMigration.md` for the schema and
provenance rules.

For a known workload, create `native.CalibrationControl()`, assign its actual
`cpu_cores` and opaque `implementation`, then assign
`options.calibration_controls = {agent_id: control}` on a
`native.LinuxRuntimeOptions` instance before opening the runtime. Bound STL
mappings are Python copies; editing a dictionary returned by a property does
not update its C++ owner. This is a construction-time workload contract, not an
inference from CPU grants or execution slots. Missing or mismatched controls retain hardware
reference estimates. `runtime.performance()` returns a detached dictionary with
concrete target bindings, profiles and resolved route cases; editing it cannot
change the runtime. A measured curve does not promote a mixed estimate involving
reference physical resources to calibrated quality.
`tuning.tune_all_gather` measures actual materialization. Borrowed consumers
require **their own** profile: `tuning.tune_consumer` runs the real native
consumer, activates its described workspace through a memory Program and
checks every route against flat numeric results. It accepts an actual output
placement (pool-to-byte weights); otherwise it requires complete observed pages.
It restores initial output contents before each timed block, including consumers
that update existing values. Profiles include exact operator/DSO hash, integer
specializations, source/output layouts, observed output placement and all-rank
runtime context. Ordinary copy calibration never selects a borrowed-consumer
route. Without a matching consumer profile or an explicit hint, fused compute
and memory overlap remain an auditable missing fact and that consumer stays flat.

Both tuners use the first candidate as baseline, rotate candidate order and
retain all max-rank round timings. The consumer tuner requires a flat baseline. The initial median-time winner is fixed before confirmation.
With at least five rounds, unanimous paired gains above `minimum_gain` may select
it only when both median(paired gain)-2*MAD and the all-sample mean-time gain
also exceed that minimum. This margin is a heuristic, not a confidence interval.
An otherwise plausible uncertain winner receives at most one additional batch
of five baseline/winner rounds, alternating order and preserving per-block
consumer numeric checks and output restoration. All five new rounds must beat
the minimum; at least 90% of all paired rounds must win, and combined robust and
mean-time margins must pass. No other candidate is retried or substituted after
confirmation; inconclusive cases retain the baseline. Three-round tuning therefore
requires confirmation before selecting a non-flat route. A case whose maximum
possible confirmed win ratio cannot reach 90% remains flat without extra work.
`noise_fraction` keeps its historical full-range/median diagnostic definition;
it no longer vetoes a route by itself. `selection` records the reason, fixed
winner, initial/confirmation/combined statistics and actual per-candidate round
counts. All raw samples, including adverse tails, remain in the profile. `write_profile` writes rank-zero
JSON evidence atomically and installs the same table on all ranks. Profiles
use schema 2 and match stable CPU/architecture identity (including available
microcode and firmware facts), kernel, frequency policy, ordered rank CPU grants,
worker counts/binding configuration, resource graph, policy and transport binary.
The consumer key separately retains the exact executed DSO and tensor contract.
Job-local capacity lock identity and boot UUID are excluded only from this
performance projection. The complete `backend.tuning_context`, live same-host
and shared-capacity checks, and captured group/alias authority remain unchanged;
the profile also records its original full context as `runtime_context` for audit.
`tuning.profile_context(backend)` returns a detached performance match context
for `load_profile`; it is not a mapping permission. Explicit thread-pool CPU
lists or runtime-discovered grants take precedence over a caller mask narrowed
by OpenMP. An authoritative nonempty CPU grant may have fewer CPUs than workers
because native workers may share granted CPUs; only an undersized caller-only
fallback mask is rejected as incomplete. Missing stable CPU evidence or a complete team CPU grant prevents
profile creation/application. Old schema-1 profiles are rejected and need fresh
calibration. Stale profiles fall back collectively; unreadable or inconsistent files
fail collectively. `VECOPS_COMM_TUNING_TAG` invalidates manual clock changes not
reflected in sysfs. These measurements cover the measured operations; complete
inference still needs its own performance check.

An upper compiler can call `tuning.consumer_requests(captures)` after native
compilation and before executable binding. It returns deduplicated opaque
operator/specification and source/output geometry requests in observed order,
with their existing control groups. The query creates no plans, payloads,
workspaces or mappings. `tuning.consumer_calibration_options(group)` supplies
detached policy facts and candidates from the generated resource cuts. The
consumer owner must explicitly provide valid bounded synthetic inputs;
arbitrary opaque consumers are never automatically executed by this query.
`tune_consumer` collectively admits its workspace before starting measurement
and records every rank's offset-only workspace recipe alongside its results,
so output placement evidence does not hide a different workspace memory tier.

`configs/communication/default.toml` is the portable default. For example:

```sh
torchrun --nproc-per-node=8 benchmarks/comm/TorchCollectives.py \
  --backend vecops --config configs/communication/default.toml \
  --cpus rank-cpus.json --tokens 768 --batch-dim \
  --iterations 20 --tune-profile routes-768.json --output timing-768.json
```

`benchmarks/comm/ResidentGranularity.py` compares materialized AllGather and
AllToAll graph replay across chunk sizes, using the configured memory pool,
rotated order and maximum-rank medians. It retains the same physical native
pool for every policy; allocation, capture and warmup are outside the clocks.
Bind each child **before Python/library startup** when studying code/data NUMA
placement. This probe measures collective replay, not complete inference or
pure copy instructions, and its winning chunk still needs model validation.

The profile only covers measured layouts. `SharedExchange.py` separately
compares the joint producer + AllToAll + consumer cost for private, source-placed
and destination-placed outputs under native graph replay, including reuse=1/4.
A synchronization-only microbenchmark cannot establish the benefit of aliases.

Additional policy keys are `initial_capacity_bytes`, `chunk_bytes`, `timeout_ms`,
`copy_schedule` and `early_snapshot`. Unknown keys, including old hardware
hierarchy sections, are rejected. Actual page placement and permissions belong
to `MemoryRuntime` and its Linux backend. `Options.threads` selects participating
lanes in the existing vecops native pool, not a second communication pool.

Standard c10d operations include CPU strided AllGather/AllToAll, contiguous
single-tensor variants, broadcast, barrier, ragged gather through
`all_gatherv_into_tensor`, and FP32 SUM all-reduce. Local strided outputs cannot
overlap internally, and distinct output regions must be disjoint. Wire dtype,
counts and sequence are validated across ranks. Inputs may alias outputs because
publication completes before materialization. All outputs remain private,
materialized tensors: hints never authorize shared aliases.
`Hints(copy_levels=(0,1,0))` explicitly overrides a value AllGather's per-level
route; other backends ignore it. The number/order matches the generated resource partitions. Other c10d backends ignore these
performance hints; the value facade uses native functional collectives during
vecops graph capture, including its Gloo fallback.

Eager `async_op=True` is cooperative: submission retains tensors and queues work;
`wait()` progresses submissions in order. It does not promise background overlap.
A graph communication node drains any older queued work before execution.
Use the facade, or `vd.launch(...)` from a graph adapter, to record the mutable
native `vecops_comm::run` operation. Capture builds immutable layouts, copy offsets
and an owned plan handle. Replay rebinds addresses and executes that plan without
Python/c10d Work construction or layout planning. Preserve the owner mapping
returned by `vd.native_operators()` when compiling an explicit graph. A plain
unadapted c10d call is not a graph-capture interface.

For a native execution provider, graph lowering prepares `run` and eligible
`consume` effects as `VecopsPreparedKernelV1` plans. Their static TensorLists
are flattened cold; each graph node retains its own pointer bindings even when
several nodes reuse the same communication descriptor. Consecutive compute,
communication and compute plans with the **same physical pool** share one
resident team. Copy tasks execute directly on its lanes; one lane publishes and
polls cross-process epochs while the remaining lanes wait at a local cancellable
barrier. No nested pool submission or Python/c10d dispatch is needed inside the
island. This removes launch/join overhead, not collective dependencies or reader
lifetime constraints. All participating readers finish before publishing `done`.

A borrowed consumer joins this team only when it exposes a compatible recorded
prepared kernel. Its internal grids execute on the existing lanes, with an
owned pointer table that is valid only until all readers join. Opaque consumers,
complex-wire collectives and nonresident providers retain the ordinary native
operation. ATen work still ends a resident island. Set
`VECOPS_GRAPH_RESIDENT_COMM=0` **before graph preparation** to compare the opaque
path; changing this variable does not rewrite an already prepared graph.
Profilers expose `vecops_comm::resident_run`, `resident_consume` and
`resident_consumer_kernel` scopes on the protocol lane. These overlap the parent
native island and must not be added to it as independent wall time.

After all plans have been exercised to provision windows, `vd.seal()` freezes
both window growth and new plan registration. Do this only when no further model
shapes or collectives need preparation. `vd.statistics()` reports calls, plans,
capacity growths and copied bytes. The initial implementation limits each group
to 256 ranks and 1024 distinct cached plans. Groups are serialized; concurrent
execution of one prepared instance is rejected. Errors poison the communicator
and wake waiting peers; process loss is detected by the group timeout. Per-Work
finite timeout overrides, multi-host transport and background progress are not
implemented. Cold initialization uses a Torch Store; the hot payload and control
protocol use shared mappings and acquire/release atomics, without Store traffic.

### Explicit borrowing and page aliases

`consume_all_gather(source, output, operator_name, specs, owners=...)` is one
native effect: publish, acquire source snapshots/domain replicas, invoke a
registered CPU consumer and retire its lease. The consumer signature is
`(Tensor addresses, Tensor(a!) output, int...) -> ()`; DSL optional `int?`
specialization parameters are also accepted when every concrete value is supplied.
The int64 address table exists only inside C++. All readers must join before it returns. Addresses
cannot escape into Python, caches, a later collective or another graph effect.
The output is independently owned; this can consume directly into a packed
matrix layout without a complete gathered intermediate. The graph keeps every
consumer compiler/DSO owner alive. This is an explicit contract, not something
`reuse="once"` authorizes automatically.

Borrowed resident consumers preserve the registered kernel's destination access
when preparing graph plans: an `inout` destination remains read-modify-write,
while an output-only destination remains write-only. A synchronous consumer may
pack an initialized owned destination before overwriting it, provided all reads
finish before returning and it retains no peer address. This does not change the
snapshot lease, publish/acquire/retire ordering, or read-only peer storage.

`PageExchange(block_shape, dtype, slots=1, placement="destination")` collectively
prepares fixed, shared AllToAll storage before capture. Each rank sees a
peer-major producer and a source-major consumer. Aliases are mapped once;
page tables are prefaulted cold. The first stride is page padded, including for
empty/small blocks, so kernels must honor strides rather than assume a dense
Tensor. `placement="source"` charges remote reads to consumers;
`"destination"` charges remote writes to producers. Physical pages follow the
owning rank's configured memory target. Consumer mappings are OS read-only;
writing them is a contract violation. The normal protocol is:

```python
exchange = vd.PageExchange((rows, channels), torch.bfloat16, slots=2)
# Construct before capture; capture/replay the following tensor program.
exchange.begin(slot)                         # waits for previous readers
producer_kernel(..., out=exchange.producer(slot))
exchange.publish(slot)                       # acquire every producer
consumer_kernel(exchange.consumer(slot), out=owned_result)
exchange.retire(slot)                        # after ALL readers have joined
```

These calls capture mutable `vecops_comm::exchange` effects with both aliased
storages as arguments. They cannot be DCE'd as pure view operations. Both slots
may be live together; reuse waits for each previous reader. Phase locks are per
slot, so a wait on one slot cannot prevent another slot from retiring. The 1–64 slots must
cover maximum simultaneous versions. There is no hot mmap, layout allocation,
payload copy or page placement. Copying an existing private Tensor into the
producer is allowed, but then that copy remains part of the workload; only a
producer writing this layout directly removes it. Page alias storage is not
silently substituted for standard writable c10d outputs. HugeTLB aliases and
automatic graph producer-layout rewriting are not implemented. A consumer clone
is needed if its value must remain writable/live after retirement. Call
`exchange.abort()` if a producer/consumer fails outside a transport effect;
`vd.backend().abort()` cancels the whole communicator and its exchange regions.
Prepared exchange storage is group-owned until the process group is destroyed;
include all live regions/slots in the model's preparation memory budget.

For C++-only use, configure `VECOPS_BUILD_COMMUNICATION=ON` and consume
`find_package(Vecops REQUIRED COMPONENTS comm)` / `vecops::comm`. Pass a cold
bootstrap exchange callback and optionally an existing `VecopsThreadPoolV1` to
`Communicator::create`. Its `Options` must also hold a strong `memory_runtime`
and an opaque `payload_pool`; same-host ranks share the capacity authority.
Prepare `Specification` instances and execute their pointer arrays, then call
collective `close()` after all readers stop to release shared roots. PyTorch is
not required for this C++ interface.

Regular peer-interleaved AllGather outputs use destination-row ownership:
one worker writes all peer fragments of a row, avoiding source-dependent work
imbalance and split cache-line ownership. Recognition (equal run/row geometry)
is cold-plan work; pointer rebinding is checked at execution. Ragged, irregular,
large-run and independent-output layouts retain generic peer scheduling.
The portable communication policy accepts the `[communication]` scheduling policy:

```toml
copy_schedule = "auto"      # default; "peer" selects peer scheduling
```

Every explicit payload copy uses the public vec module: contiguous snapshots,
strided pack/unpack, hierarchical replicas and destination-row transport. Long
runs use four independent vector loads before their stores; predication handles
unaligned short fragments and tails without reading beyond the source. No libc
copy backend or instruction-selection setting exists. ISA selection occurs once
in baseline code, before worker callbacks: SVE, AVX512, AVX2 or the portable vec
implementation. The same source implements all variants; older CPUs never enter
unsupported instructions. Python wheels include both copy and cold-dispatch
sources and freeze the complete header set in the communication build identity.
`backend.statistics()` exposes `row_copy_calls` and `row_copy_bytes`. Byte counters
count explicit payload, never actual interconnect traffic or consumer-kernel
reads. Layout/ISA selection does not alter peer synchronization, snapshot
lifetime, graph ordering or numerical data.

For protocol audits, `benchmarks/comm/instrument_protocol.py --source-checkout
<checkout> --output <new-private-directory>` creates an independent communication
source tree with opt-in C diagnostic hooks. `comm_diag_start()` reserves records
before inference; `comm_diag_stop(path)` disables capture and writes JSON after
inference. Use a single calling thread per process and match identical
communicator epochs across ranks. The records distinguish caller-only and
pool copy submissions, blocked waits by publication phase, descriptor cache
misses, capacity growth, capacity-equal mapping checks, and actual payload mmap
calls. Consumer execution remains a separate nested interval. These hooks are
absent from production builds. Copy submission wall includes worker dispatch
and completion; it is not a measurement of pure vector instruction time.

`benchmarks/comm/ScatterCopyLatency.cpp` and its Python driver isolate the
18 MiB / 72-task / 36-worker contiguous AllToAll payload on 920f-4. Link the
probe to the same communication copy dispatcher/ISA objects as the backend.
The default mode uses private destination slices explicitly allocated on each
peer's HBM, with no communicator or shared mapping. `--shared-buffer` instead
requires an isolated `instrument_protocol.py --snapshot-addresses` build and
an explicit local-HBM `VECOPS_COMM_POOL` and `--pool-map`
JSON mapping from OS NUMA numbers to runtime pool IDs: it borrows the exact prepared
AllToAll source/destination addresses and runs the same payload without the
transport protocol. Additional blocks precede each copy with an untimed full
collective. All group/plan/input owners must remain alive through these probes;
the raw-address diagnostic is not a production API.

Select `--socket 0|1`; this benchmark intentionally uses 920f-4 OS numbering.
`--exec-workers` binds before exec, making `LD_AUDIT` group selection effective;
ordinary multiprocessing import/binding order cannot establish that guarantee.
`--placement` queries every source/destination page before/after each block,
records actual copy-code pages and `/proc/self/numa_maps`, without migrating
pages. `--pmu` adds per-thread user cycles/instructions; check nonzero PMU
running/enabled deltas before interpreting them. It also records per-thread
user/system CPU and voluntary/involuntary switch deltas. CPU accounting is not
an instruction-retirement clock: neither it nor user-only PMU counters alone
can distinguish every kernel/firmware/platform pause. PMU/accounting reads, page queries,
admission, allocation and full-output checks are outside the callback clocks.
Thread CPU clocks and per-lane record stores remain diagnostic overhead.
Executions may receive different physical pages: a reduction in rare tails
between fresh runs alone does not prove that a loader/placement change caused
it. The independently exec'd workers' file-backed atomic admission is solely
benchmark control, not the production collective barrier algorithm.

### Torch OpenMP resident teams (opt in)

Set `VECOPS_THREAD_POOL=torch` and `VECOPS_TORCH_RESIDENT_TEAM=1` before
registering kernels or creating a communicator to use one canonical provider
for source kernels, graph islands, and collectives. Its workers are Torch's
OpenMP team; it does not activate the native pool. A matched direct OpenMP
compiler adapter is required (Clang/libomp or GNU/libgomp). The packaged-GOMP
function-table fallback does not provide this extension. Prepare a new graph
when changing providers, thread budgets or worker affinity.
Compiler wrappers with nonstandard executable names are identified through
`--version`; a BiSheng `bisheng-cxx` wrapper is recognized as Clang. Runtime
and compiler binary identities participate in the provider cache key. The
provider honors the selected toolchain environment, `CXXFLAGS` and `LDFLAGS`;
these flags and include/library search variables are also keyed. Loading a
second OpenMP runtime is rejected. An explicit GNU compiler remains incompatible with
Torch/libomp: choose a matching Clang wrapper through the bridge compiler
environment settings.

Resident execution requests exactly its prepared worker count. If OpenMP
dynamic teams or thread limits reduce that count, it rejects the call before
entering any worker callback rather than deadlocking a barrier. Worker errors
cancel all local barriers and are rethrown after every OpenMP worker joins.
The provider can be reused after failure; a failed communicator remains poisoned.
Single-worker teams also count as nested callbacks, through the C10 guard;
logical tasks submitted inside them execute serially on the calling worker.
`OMP_WAIT_POLICY`, `KMP_BLOCKTIME`, `OMP_PROC_BIND` and `OMP_PLACES` must be
selected before runtime initialization and benchmarked with the full model;
active waiting may compete with native/ATen work when providers are mixed.

The communication policy `communication.early_snapshot=true` opts AllGather into
early local publication: after previous readers retire, a rank copies its
source while other ranks may still compute. The ready acquire precedes
descriptor validation and every output/borrowed read. No collective uses
background copy workers; AllToAll retains validation before remote writes.
This can hide publication under peer arrival skew; it does not remove that
skew or promise less waiting in every individual phase counter. Evaluate
whole-model and last-arrival-to-completion latency, not just arrival wait.

Replica ingress targets are planned once from the actual partition membership
and per-source payload bytes. Each remote source is assigned to the least-loaded
local member, with global-rank order breaking ties. Interleaved rank numbering
and unequal group sizes do not concentrate traffic through `source % members`.
Every member independently derives the same immutable routing table; replay
only reads it. This balances modeled payload bytes, not measured fabric load.

Resident collectives join the local team at copy-preparation and actual copy
completion boundaries. Leader-only ready/relay publication flows directly into
the following preparation join. Barrier-only and borrowed-output phases omit
copies using immutable operation properties. A dynamic zero-task check cannot
skip a join: a fast leader could overwrite the next phase's task metadata while
a delayed worker still reads it.
Relay branch decisions are immutable prepared-plan properties, so workers never
read an unpublished leader decision. Cross-process publication and reader
retirement ordering are unchanged.

Prepared pointwise lowering also supports guarded scalar square/sqrt and
scalar fills without reading uninitialized output. Functional scalar outputs
retain their captured zero-dimensional Tensor metadata; only the native ABI
uses a one-element logical view. Short `sum(dim=-2)` reductions of up to 32
floating values and contiguous feature panels divisible by 64 use one
row/panel grid with float accumulation and the original output dtype. Other
axes, integer masks, long reductions and unsupported layouts retain ATen.
Reduction order is explicit and must be numerically validated; no bitwise
equivalence across all Torch implementations or ISA widths is promised.


Graph control values with a captured one-element `int64` Tensor and a Python
integer offset can lower `add` to one native grid. The grid preserves full-width
integer wrap and executes in tape order before consumers read the value; it
does not convert random seeds through floating point. Other integer operations
retain ATen. `VECOPS_GRAPH_INTEGER_SCALARS=0` disables this lowering.

### Flattened Tensor replay

`replay.run_tensors(tensors)` accepts an already flattened sequence in capture
leaf order when every captured input leaf is a Tensor. Static non-Tensor leaves
require the ordinary `replay(*args, **kwargs)` entry. Outputs retain their
captured container structure and ownership/alias policy.

This entry avoids Python pytree construction and duplicate Python Tensor
metadata queries. It still rejects Tensor subclasses, checks inference/context
and reentrancy, and runs the same C++ input/state guards for dispatch keys,
names, metadata, storage sizes, storage aliases and overlapping byte ranges
before any graph operation. Dynamic input addresses are rebound on each call;
this is not an unchecked pointer-only entry or an invariant-content promise.

### Resource-backed graph memory compilation

`vecops.planner.BufferRequirement` represents both tensor storage and native
operator workspace. `GraphMemoryCompiler` consumes the runtime's immutable
resource view, current budget snapshot, placement/mapping capabilities and cost
queries. Pool and agent names are opaque; the compiler has no tier or CPU
hierarchy types. Missing consumer/cost information remains unknown.

```python
from vecops import memory, planner

runtime = memory.default_runtime()
view = runtime.view()
consumers = tuple(agent.id for agent in view.graph.agents
    if agent.id in view.executable_agents and "host_load_store" in agent.capabilities)
lowered = capture.prepare_torch(borrow_outputs=True)
requirements = lowered.memory_requirements(consumers=consumers)

# framework_pool_id is an allocatable pool ID chosen by the application.
# The recipe helper takes its concrete MemoryPool description.
# This reserves capacity, not physical placement of ordinary ATen allocations.
framework_lease = None
if lowered.framework_requirements:
    framework_pool = next(pool for pool in view.graph.pools
        if pool.id == framework_pool_id and pool.id in view.allocatable_pools)
    recipe = lowered.framework_budget_recipe(framework_pool,
        participant=runtime.participant)
    framework_lease = planner.reserve_recipe_latest(runtime.native, view, recipe)

compiler = planner.GraphMemoryCompiler(runtime, view, runtime.snapshot(view))
program = compiler.compile(requirements)
prepared = program.prepare(runtime, view)       # virtual addresses only
prepared.admit()                               # pending physical capacity
prepared.materialize(external=external_tensors) # roots + mappings + state copies
replay = lowered.compile(runtime=runtime, view=view, memory_program=prepared)
```

Describe and bind the same lowered plan: lowering again with different
`borrow_outputs`, pointwise or donation options can change its storage contract.
Keep `framework_lease` until all framework users stop; it is independent of the
managed Program backing. For a joint model, budget framework peaks over the
combined schedule rather than summing independent graph reservations.

The caller supplies actual executable consumers for meaningful bandwidth costs.
The external map uses storage IDs for borrowed inputs/state; these buffers are
not charged again as owned roots. Native workspace descriptions use IDs
`kernel:<instruction ordinal>/<workspace name>` and are prepared without physical
allocation. Native workspace symbols include a stable slot ordinal as well as
the optional debug label, so two equally named buffers remain distinct.
Metadata-only kernel plans activate with addresses selected by this same
program. By default a provider's retained workspace remains live for the whole
model program; only proven transient workspace can reuse storage across calls.

For multiple serialized graphs, call `compile_model(programs, calls,
serialized=True)` with a requirement tuple per graph and `ProgramCall` objects.
`ProgramCall.bindings` links local storage IDs to shared model identities, for
example an initializer output and a steady-graph input. Unlinked transient
storage can reuse offsets across calls; linked producer/consumer storage stays
live through the intervening gap. Pass `memory_call=call.id` when compiling each
capture against the shared prepared program. Without the serialized contract,
all calls conservatively overlap, even if their local event numbers differ.
Repeated calls use explicit repeat counts and weight access traffic by that
count. For serialized calls with identical bindings and no interleaved calls,
strictly internal lifetimes fold onto one iteration. External input/output/state
ranges cover the entire repeated call domain, preserving tail-write/next-head-read
backedges. An internal buffer cannot be used after its call returns; escaping or
loop-carried state must be external or retained.

`call_time_domains(programs, calls, serialized=True)` exposes those integer time
domains. `compile_model(..., scoped_requirements=...)` also accepts requirements
with explicit model-global time offsets. `ScopedNativeLease` is the lifecycle
adapter for native state with a bounded activation span:

```python
scope = planner.ScopedNativeLease(lease, prefix="phase-workspace",
    lifetime=(phase_begin, phase_end), allowed_pools=working_pools,
    consumers=actual_consumers, execution_count=logical_calls)
# Include scope.requirements() as scoped_requirements of the joint Program.
# After the Program is materialized, at the real phase boundaries:
scope.activate(prepared)  # constructs/initializes this native instance now
# Execute every user of the instance, including its synchronous consumers.
scope.suspend()          # stops/destroys the state before its pages are reused
```

Here native `retained` still protects state throughout the instance's active
span. Delaying initialization and enforcing the stop boundary make physical
reuse safe; changing the retained flag without these lifecycle operations does
not. A signature used in separated phases stays live through the intervening
phases. Unscoped instances keep whole-program retention. The AF3 adapter applies
this protocol to uncaptured eager native workspaces; captured graph-native
retained workspaces currently keep whole-program retention.

Both `native_workspace_requirements` and `ScopedNativeLease` accept an exact
positive integer `execution_count` (default 1). It multiplies the per-call
workspace traffic estimate, including the one-pass fallback when native traffic
is unspecified. It leaves bytes, replicas, alignment and liveness unchanged.
The AF3 adapter passes the metadata discovery's full logical eager call count,
including enclosing recurrence weights, exactly once. This gives eager and
captured calls the same traffic accounting without treating repetitions as
additional physical allocations.

Planning caches only exact placement/access queries within one compilation:
size, mode, page pattern, complete endpoint segments, consumer, payload, view
epoch and execution-slot context remain part of query identity. It does not
extrapolate another payload's cost. Capacity admission uses a fresh ledger
snapshot. Retained spatial ranges are indexed separately from transient event
intervals without changing traffic-based placement priority.

`tools/memory/migration/ReplayPlanner.py` replans a trusted local
`programs/calls/serialized` pickle without numerical inference or physical
allocation. It writes a JSON timing/capacity report and the canonical TOML recipe;
the migration source/native overlay can be used to select the implementation.

`program.dumps()` emits canonical TOML with integer offsets and no addresses or
handles. `planner.audit_plan(program)` prints logical bindings, offsets, backing
charges and explicit state initialization copies. Small same-placement buffers
share page slabs; independent mappings and exports get exclusive page envelopes.

`GraphMemoryCompiler(..., preferred_alignment_bytes=None)` keeps the R33 default:
for allocated buffers of at least 1 MiB, packing prefers an offset aligned to
the placement's logical page quantum as well as the requested alignment. An
explicit hint must be a positive uint64 power of two; it replaces that soft
quantum preference with `max(page_bytes, preferred_alignment_bytes)` only for
those large allocated buffers. Stronger requested alignment and interleaved
placement periods still apply. The hint is not a hard buffer requirement. This is a
soft preparation heuristic: if its padded demand does not fit, the same
placement is tried with compact packing before considering another pool. A
planning-capacity failure of the complete preferred layout, including its final
physical estimate, retries once from empty slabs with compact packing throughout.
This retry uses the same planning inputs and allocates no backing; provider,
permission and validation errors are not retried. Requested alignment remains a
minimum. Concrete offsets, automatic pool choices and root extents may change;
no unchanged capacity total or performance improvement is promised. Small
subpage buffers keep compact packing. No work is added to numerical replay.

`BufferRequirement.alignment` is a hard minimum on the absolute allocated buffer
address. Each slab's `VirtualRegion.base_alignment_bytes` carries the maximum
requested alignment (and at least its page quantum) to VA reservation. Zero on a
manually constructed region means the existing page-aligned default. The base
minimum is independent of `page_bytes`, mapping lengths and physical-capacity
charges: a 64-byte buffer may have a 4 MiB aligned VA in one ordinary logical
page. A nonzero base minimum must be a power of two. Existing large Linux
reservations retain their THP alignment preference even with the default.
Preparing a stronger recipe against an older, weaker base guarantee is rejected;
a stronger prepared base may satisfy a weaker request. No live-address-dependent
offset patch changes the compiled recipe.

Output-cache copies accept explicit `target_program` and `target_bindings` from
the model placement policy. Source input aliases must be redirected together;
consumer writes, physical aliases with other operands and opaque asynchronous
uses prevent relocation. No destination capacity is allocated implicitly.
Preparation requires writable destination receipts and proves that distinct
cache sources have disjoint physical destinations. It also checks all supplied
source outputs, producer state and other live consumer operands. Different
virtual addresses do not prove independence: remapped aliases are compared by
their backend-certified backing identity and offset. Incomplete physical
receipts are rejected. These checks add no query to replay.

Graph templates persist address-free IR, schemas and symbolic model/input
bindings. Restoring them reacquires current owners, then runs ordinary guards.
Opaque ATen allocation interiors remain owned by the framework.

## Shared preparation policy

```python
with vecops.BuildSession(parallelism=16, report_path="build-session.json"):
    # Bootstrap/parameter packing, native request discovery, then real graphs.
    result = vecops.precompile(model, example, parallelism=16)
    # GraphCapture.compile_torch and output-transfer preparation inherit jobs.
```

BuildSession is separate from `collect_compile_requests`: real eager capture
executes bootstrap kernels, whereas the request collector short-circuits them.
The session stores content identities, not tensor addresses or bound owners.
Do not mutate build inputs during the context; start a new session after edits.
One independent context may be active per process; its worker contexts share
one actual compiler/linker budget. Ready kernel/bridge builds may overlap, and
an integration can prewarm the graph executor during request discovery. Compiler
objects remain reusable across sessions through a process-local launcher route.
Content and include-resolution guards are checked before publication. The
session report separates capture, lowering/binding, source costs, and individual
validation leaves. Graph preparation still honors initializer-to-steady
cache binding dependencies. Compilation may prepare a shared SDK package when
a native compiler is first resolved within a session.

Repeated callback lookups reuse only immutable code-object digests. Defaults,
closure/global state and source contents retain their ordinary fresh identity
checks. Code replacement and source-location changes are observed; weak cache
entries disappear when their code objects are unloaded. Mutable code constants or metadata
use the uncached path.

A model integration can call `vecops.graph.discover_pointwise_requests` around
its prepared region while `vecops.is_discovering_graph_artifacts()` is true.
The function returns the ordinary fake result while appending storage-free
artifact requests. `borrow_outputs` must match real graph lowering. Source
kernel compilation and Torch ABI bridges remain separate compatible groups;
pointwise jobs no longer need to be discovered first during real capture.

### Metadata executable capture

`GraphCapture.capture_torch_metadata(function, *fake_args)` requires the strict
`vecops.graph._metadata.metadata_mode()` context (a FakeTensorMode with CPU
fallback and tensor constant folding disabled). It records the same static dispatcher tape and semantic IR without
CPU numerical execution. Data-dependent escapes are rejected. The metadata
capture retains FakeTensor source owners until binding, so weak conversion-cache
collection cannot recycle a storage token and associate it with another weight.
Produced storage identities use one weak receipt per graph storage; historical
addresses alone never prove a cross-graph alias. Traffic normalization visits
each event once and counts aliased reads/writes by their maximum logical span.
The metadata
binding helper `vecops.graph._metadata.bind_metadata(capture, real_args)` binds
current real input/module storage before `compile_torch`; it never executes the
program to infer values. Unknown external tensors fail explicitly. The resulting
executable exposes `metadata_outputs`, uninitialized geometry/alias prototypes
for linking an initializer to a steady executable. These prototypes must never
be interpreted as computed results. Only actual replay initializes numeric
outputs. Runtime replay continues to reject FakeTensors and active tensor modes.

Binding and compilation may run after the inference scope that created the
source tensors has ended. Projected state and static bound views retain their
source storage, strides and offsets. Read-only state redirection preserves the
source Tensor object, its inference status and its version-counter contract,
including when the destination storage was prepared in another mode. Ordinary
Parameter autograd metadata remains intact.

`torch.empty` and `torch.empty_like` calls inside metadata executable capture
record dispatcher factory nodes. The resource compiler owns their temporary
storage and lifetimes when metadata is bound and lowered. Outside a prepared
graph these remain framework allocations; eager applications request managed
storage explicitly through `vecops.memory.Pool`.

### Immutable planning snapshots

`planning_graph = graph.freeze()` returns a detached, recursively read-only
`ExecutionGraph`: nested dataclasses reject assignment, mappings reject writes
and sequences become tuples. Edit the original graph, then freeze a new
snapshot after any metadata/storage binding. Capture/replay graphs are not
frozen implicitly. `planning_graph.content_digest()` (also used by
`planner.graph_digest`) encodes format-v1 canonical JSON once and reuses the
digest for subsequent plan checks. Mutable graphs always recompute their
digest. `to_dict()`/`dumps()` retain their existing JSON representation and
return detached ordinary values; serializers no longer deep-copy dataclasses
before walking them again. Existing plan digests remain compatible.

Within a BuildSession, SVE compatibility checks reuse each complete compiler
policy and hashed effective environment. The first lookup still invokes the
selected explicit driver; checkpoints re-read wrapper and reported installed
Clang bytes, including a previously missing installed binary. New sessions
probe again, and offline cache-only operation continues to replay attestations
without finding or executing tools. Reports include `toolchain-probe` events
with wall time and version query counts. User loader/instrumentation variables
are retained in the effective environment, not silently stripped.

AF3's qualified frontend continuation uses the private precompile
`_torch_call_observer(registration, arguments)` hook to copy storage-free Torch
call descriptors while the request collector short-circuits the dispatcher.
The hook also runs before a cold bridge is materialized. It does not submit
operators, change request deduplication, or alter semantic graph recording.
Runtime scalar arguments are observable even when they do not specialize the
artifact; integrations must not infer workspace requirements from the deduped
artifact set alone. Observer exceptions propagate and unwind collection.

### Shared preparation and validated discovery reuse

Torch bridge v8 identities store content digests of the common Torch input
inventories. The original file/tree bytes remain BuildSession validation leaves;
compact manifests do not replace publication/seal checkpoints with timestamps.
Equivalent registrations check one artifact per batch. Hot bridge loading does
not take the build-writer lock; a miss retries and rechecks under that lock.

The optional `precompile(..., prepare_exchange=...)` callback returns JSON-safe
records from all ranks in rank order. Python-declared operator recipes with the
same host, schema, compiler/source/cache identity and execution parallelism share
an owner; uneven request geometries are unioned on that owner. Unknown expert
operators keep their originating owner. Each process still creates its native
provider and Torch dispatcher bindings. Build failures are exchanged before the
next phase. Without the callback the previous rank-turn protocol is retained.
No process-private native handles or Tensor payloads enter the exchanged records.

`VECOPS_PRECOMPILE_CPU_LIST` optionally names an explicitly reserved compiler CPU
pool, independent of each model rank's affinity. BuildSession's parallelism is
the shared compile/link limit; native execution parallelism remains the model's
thread count. Compiler RSS and available memory must bound a larger build pool.
This scheduling override does not change artifact semantics.

`ExecutionGraph.state_access_summary()` is a projection for state-only planning.
It retains external/persistent aliases, all loop-carried storage, original event
indexes/scopes and exact payload/metadata classification. Temporary factory
replay is outside this projection. Freeze the summary separately and derive
explicit external state requirements; the full graph remains available for diagnostics.

The private discovery recipe adapter serializes JSON schemas, source/compiler
attestations and storage-free arguments. Its application must separately attest
model code/structure, inputs and control configuration and agree on hits across
all participating ranks. Restoring a recipe revalidates compiler, vector state
and source dependencies. Address-free metadata tapes can also be restored under
strict FakeTensorMode. Input-dependent initializer values still execute at real
replay; post-lowering output slot geometry is cached, never output payloads.
Identical complete compiler policies share one frozen snapshot only within a
single declaration restoration. Native owner verification checks each distinct
schema once per verification call. Pointwise tapes can carry address-free
lowering recipes, guarded by tape/schema/logical geometry, implementation and
registry identity, execution threads, vector state and borrowing/donation
policy. Restored recipes require current published native owners; missing or
changed contracts fall back to normal selection. Workspace, tensors and native
addresses are always rebound in the current process.


### Logical communication in metadata templates

The private template adapter can relocate layout-only `vecops_comm::run` and
`vecops_comm::consume` plans. It writes logical group membership/platform,
operation/splits/root/copy hints and tracked consumer symbols, never a live
handle. Binding prepares new local plans from metadata on an existing group,
retains the native registration and communicator, and submits no collective.
Ambiguous/missing groups, platform changes, unknown owners and PageExchange
external mappings remain unsupported. Pointwise recipe contracts normalize
communication through these logical descriptions as well.

Metadata restoration binds current explicit owners. Final storage/lifetime
requirements come from the optimized graph plan, not a capture-time allocator
session or a state-only graph summary.

`implementation_identity()` still attests the complete Python frontend for
model/tape caches and in-process elaboration guards. Native build attestation
uses a separate identity excluding a small audited list of runtime-only modules;
all Python paths remain in its inventory and unknown modules remain binary
dependencies. Generators, schema bridges, headers, native libraries, tools and
compile/link policy still invalidate native artifacts. This is dependency
separation, not an option to ignore arbitrary SDK edits.

Donation lowering detaches only the tensor/storage records it mutates and the
storage-membership lists; execution events, metadata and annotations remain
read-only. This avoids recursively copying the complete IR. Binding retains
explicit physical owners and validates storage aliases and byte overlaps before
execution.

Collective template restoration validates relocation ordinals before preparing
plans and follows numerical instruction order. Canonical JSON sorts object keys
lexicographically, which must not reorder native descriptor creation or allocation
requests at digit boundaries (for example instruction 10 before instruction 2).
Duplicate/noncanonical indices and changed instruction placeholders reject the
restoration; executable communication order remains the captured tape order.

Metadata region projections can attach an executable tape to the existing
semantic capture. They install no nested `GraphCapture`: a recorder above an
application router suppresses lower recorders for the same operation, while
source requests emitted by that router still enter the compilation union.
A regional snapshot detaches its event/tensor/storage records before lowering;
parent planning events and liveness remain independent. Logical owner bindings
are resolved again after parameter relocation, including nonzero view offsets.
Unmaterialized source schemas are resolved only after publication. A borrowed
consumer discovered in this mode records a logical relocation and its source
kernel request, not a transport submission or an executable pointer table.
Missing steady variants are recorded outside the model planning event stream.

Graph lowering builds the input-position index once before binding pointwise
scratch outputs. It must not enumerate and search the complete pybind input
list for every output: that introduces a product of input count and pointwise
node count into preparation. Alias inputs are appended after this lookup phase;
the existing input order and tensor bindings are preserved.
