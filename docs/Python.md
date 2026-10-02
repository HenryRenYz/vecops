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
annotation modules have been removed. A future
Python DSL backend can consume the same captured `TemplateFunction`; the
current implementation requires `T.template(source=...)` and forwards to the
source file's `__kernel__`.

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
