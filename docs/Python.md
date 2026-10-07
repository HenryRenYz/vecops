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

`T.func` helpers are source-parsed and inlined, including their immutable
captures in cache identity. Conditional scalar/tile/tuple returns and early
returns from nested loops are supported. A helper return exits only the
helper, not its caller kernel. Loop returns lower to lexical exit regions
without C++ closures; a value helper that reaches its end without returning
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
native indexed `store_convert`. A failed bounds check may leave earlier
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
  magnitude to 80; use reduced-argument exp plus squaring, or the explicit
  `algorithm="polynomial"` AF3 SME evaluation schedule. Small-float inputs
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
all physical slots beyond logical K must be zero, matching the public C++
MatmulPack format contract. Wrapping external storage does not create or
repair this padding. Shape/capacity/alias validation precedes hardware-state
entry. A/B formats
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

Ordinary-SVE BF16 PackedDot leaves load full registers for complete K groups
only when the actual vector width fits inside each operand's current physical
spatial panel. Shifted windows near a panel boundary keep masked loads.
MMLA's row/column segments do not mix discarded spatial-padding results into
valid outputs; the final partial K group explicitly clears its nonlogical
halfword before MAC, rather than relying on an external buffer's zero padding.
This local robustness does not change `T.packed`'s general zero-K-padding
contract: other packed consumers and fallback families still require it.
FP16 inner-product leaves keep FP16 input vectors and use the public grouped
widening-dot primitive with FP32 accumulators (FMLALB/FMLALT on SVE2).
The input DAG's FP16 conversion boundary and strict multiply/add policy remain
in place, and an explicit `T.fma` remains fused. These ordinary-SVE operations
use VL independently of SVL and do not enter streaming mode.

Dynamic SME matrix geometry is selected before entering streaming mode. Each
selected family has its own small forced-inline callback and owns one SM/ZA
interval; unrelated family bodies do not share an oversized streaming owner.
Scalar lane bounds use the captured SVL independently of ordinary VL, and
clipped fringe extents use public forced-inline scalar helpers. This is an
inlining/code-generation policy, not a change to packing or family thresholds;
target disassembly and paired execution still determine its effectiveness.

On an SME-FA64 target, same-storage-type operands with unit K stride can
pack through public ZA horizontal loads and vertical drains. Full panels
of elements up to 32 bits ping-pong between two ZA tile pairs; FP64 and
spatial/K fringes use element-width staging with exact zero padding. Raw
packing preserves storage bits (including NaN payloads and signed zeros).
Negative spatial strides are legal; non-unit K strides retain the vector
gather/conversion path. An unmasked rank-two FP32 storage leaf can also stage
through ZA32 and feed its transposed slices to the normal pointwise Vector
DAG before narrowing to BF16/FP16/int8/uint8. FP32 row-statistic and K-axis
parameter broadcasts keep their own logical coordinates; final mul/add
remain separate unless the DSL explicitly requests FMA. The route does not
cross reduction, materialization or register/state boundaries, and 16-bit
auxiliary loads retain the ordinary path. A folded load-convert's narrow
alias snapshot also uses the ordinary path rather than pretending it is
FP32 input storage. Absent spatial lanes are zeroed after the DAG and before
conversion; padded K slots skip the entire DAG, including parameter reads.
Conversion uses the public ordered/saturating conversion contract, not a
bitwise guarantee for converted NaNs. Capacity and alias checks, including
any snapshot, run before streaming entry. Geometry
uses the captured calling-thread SVL, not a build-host or fixed 64-byte
assumption, and no vector arrays are generated.

`T.func` is inlined **before** fusion analysis, including nested
`T.library.*` calls in frozen capture identity. Rank-two layernorm/softmax
hoist row statistics once; their final affine/exp tail stays in the operand
DAG. The CPU fusion policy estimates operation cost and reloads to choose
recomputation versus materialization, or puts the transform in packing loads.
Dynamic row/column GEMV selects the non-packing reuse model when it bypasses
packing; a tiled API alone does not imply that B is produced only once.
`pass_options={"cpu_matmul_fusion": "auto"/"always"/"never"}` provides an
explicit diagnostic override; `auto` is a heuristic, not an optimality proof.
`cpu_sme_packed_ab_cache_bypass` is a strict boolean override. The SME cost
profile enables the narrow policy by default; `False` retains the blocked
control path. It may skip automatic cache panels for short, wide BF16 products
whose A and B are already SME-packed. Geometry follows captured SVL, while a
quotient-based runtime test requires the padded two-by-two full-K micro-panel
to fit private L2 (the default budget uses its full capacity, not an
arbitrary half-cache reservation). A target cost profile can select a stricter
`sme_packed_ab_private_cache_divisor`; this is a performance budget, not an
instruction-legality rule or universal profitability guarantee. Tuning thresholds live in the immutable SME cost
profile (`sme_packed_ab_*`), not scattered opcode or exact-shape tables.
Explicit BM/BN/BK, other dtypes/formats, ordinary vector requests and raw
operands keep their existing paths. The full-dot leaf preserves prior
accumulation, fused epilogue and alias checks. A processor cost profile can
disable the default with `sme_packed_ab_cache_bypass=0` (the policy accepts only
integer 0 or 1); the explicit boolean override takes precedence. The default
is qualified by the complete 44-key packed-AB geometry cohort on 920f-lx,
not a universal performance-parity claim or a fixed processor vector length.
`benchmarks/dsl_matmul_packed_ab_ab.py` compares the explicitly blocked control, this
policy and the original C++ reference on the complete source-registered
geometry cohort. It retains GEMV/no-op dispatches, policy/fusion variants
and private-cache budget rejects rather than selecting only profitable keys.
Listing requires an explicit `--svl-bytes`; execution queries actual process
SVL inside a scheduled ARM job and verifies any supplied value. Deterministic
`--shard-index`/`--shard-count` partitions permit parallel validation without
dropping or duplicating original keys. Every six rounds balance all three
labels' positions and directed neighbors; all clocks, repeats and samples
are retained. Offline packing remains outside timing, as in the source
registration; original packing ABI and independent numerical checks precede
timing. These measurements qualify this cohort, not the entire DSL catalog.
`cpu_sme_narrow_singleton_slices` is a strict boolean experiment, default
`False`. For a statically one-element ZA slice, memory loads, conversions and
fused elementwise work use `FixedTag<T,1>`; ZA instructions keep their native
word tag and masks. The proof is orientation-specific: a vertical slice of an
Mx1 output still contains M elements and is not narrowed. Dynamic or merely
short/tail slices keep their original route. This avoids instantiating wide
index/predicate tuples for scalar-width memory work without scalarizing the
matrix instruction. Cross-codegen checks pass; Native correctness and paired
performance acceptance remain separate prerequisites for enabling a default.
For an unpacked BF16 operand whose spatial axis is statically one, the same
experiment computes one value per K slot and interleaves those values before
MOPA. Already packed words, other operand dtypes and unknown spatial extents
retain their original operand path. This is not a smaller hardware ZA tile.

`cpu_sve_runtime_quant` is a strict boolean experimental option, default
`False`. When enabled, a lane-local FP32-to-U8 operand DAG and pure signed-I8
SME-packed B can use an ordinary-SVE widening-dot row driver without ZA.
Matching uses inlined IR semantics, not helper names. Four quantized K bytes
are broadcast per dot instruction; incomplete K groups are zeroed **after**
conversion, preserving a nonzero zero-point. Strict multiply/add remains
separate; an explicit `T.fma` remains fused. Ordinary VL sizes the arithmetic
vectors while the independently captured SVL bounds packed-panel loads.
Cache blocking, phased I32 prior accumulation and the final epilogue remain
in place; unsupported DAGs, execution domains or layouts retain the general
path. This experiment fuses A production unless `cpu_matmul_fusion="never"`
explicitly requests materialization. It does not yet assert a profitability
crossover: compare its target timings with both the general MOPA path and
the public C++ RuntimeQuant family before choosing a default or M threshold.
Its intermediate tile values are limited to FP32/I32/bool, followed by the
final U8 conversion. FP64 intermediates and staged FP16/BF16 conversions
retain the general lowering, including their original rounding boundaries.
SME cache leaves retain a constant K extent only when it fits the actual
rounded BK: the K quantum is 1 for FP32/FP64, 2 for FP16/BF16 and 4 for byte
operands. For example, `K=5, BK=3` still needs two phases, including its
shortened final panel; the generic vector family's quantum of 32 is not a
valid bound for this proof. Accumulator input applies only to the first
phase, and output transforms only to the final phase.
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
For SME, automatic packing can also pack native, K-contiguous A when its
spatial gather loads are sufficiently expensive. The decision includes padded
storage and reuse within the actual NC panel, rather than assuming reuse
across all N panels. Small/GEMV, broadcast and spatially contiguous inputs
retain their existing route. A rejected decision neither allocates nor packs
A; a selected decision dispatches once before each ZA block's K traversal.
This does not add automatic packing to `T.dot`, or change explicit `packing=`
requests. The centralized SME cost profile controls this internal policy.
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
`T.softmax(x, accuracy="strict" | "fast" | "estimate")` uses the requested
exponential accuracy; strict normalization computes dtype-typed `1 / sum`
with ordinary IEEE division, matching C++ Softmax's scalar inverse algorithm.
Fast/Estimate normalization retains the corresponding native reciprocal
tier. The default remains `"strict"` for both rank-one and rank-two tiles;
their scalar inverse is defined by the same visible DSL helper, including
rank-two masked normalization's safe denominator. This strengthens strict
inverse precision and does not change the explicit `T.rcp` primitive,
reduction order, or the default `unroll=2`. Like native C++ Softmax, FP32 `"estimate"`
uses a visible degree-nine polynomial with seven squarings instead of FEXPA,
which can be exceptionally slow for some ARM input values. Centered scores
below -80 become zero and NaNs remain NaNs. Strict/Fast and non-FP32 compute
dtypes keep the native exponential. Masked softmax uses the same policy.
Scalar `T.rsqrt`/`T.rcp` on FP32/FP64 row statistics use the same native vec
math dispatcher and accuracy tier as vector values, including SVE estimate
and refinement instructions; they are not silently replaced by scalar libm.

Unrolling copies each operation across iterations, not the complete body per
iteration. Seven adjacent vector chunks group as 4+2+1, with powers P+2/P+1/P.
Target `VEC_MAX_POW` and lane-preserving dtype rebind requirements limit merging;
the C++ preprocessing alternatives keep generated sources portable to SVE.
Oversized virtual vectors remain streamable, not forced into one native value.
Nonadjacent reduction-axis copies retain distinct lane domains. Loop-carried
scalar dependencies and break/continue retain ordered loops; alias-dependent
serial iterations have a runtime ordered fallback. Tails are always guarded.
Ordinary SVE traversals may split only their masked remainder into native-word
register fragments, accounting for wider load/store dtypes. Full-loop grouping
and the requested unroll remain unchanged. Reduction banks are reconstructed
before their original horizontal reduction, preserving their lane/order
contract; explicit SIMD and loop-carried vector extents retain their existing
lowering. This internal policy does not relax masks or conversion rounding.

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
matrix views. The `row_normalization_streaming=False` pass option disables
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

CPU lowering captures each ABI tensor parameter's data pointer, dimensions
and strides into invocation-local const scalars at kernel entry. Those
descriptors are invariant, but the pointed-to tensor contents are not:
serial iterations, aliases and atomics still observe the original memory
effects. This does not hoist dynamic loop state, view offsets, or addresses
of scratch buffers that can resize. The independent diagnostic pass option
`cpu_parameter_metadata_snapshot=False` retains direct descriptor access.

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
shadowing headers. Linux mutation notifications accelerate content hashing;
the dnotify fallback uses an unused process signal when inotify quota is full,
and unavailable/overflowed watches fall back to rehashing, never size/mtime-only
identity. Toolchain attestations are atomically published under writer locks.

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
