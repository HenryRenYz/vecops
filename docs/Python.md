# Python JIT API

The public Python API is implemented above the low-level `vecops._C` pybind11
module. Applications should use the classes and functions exported by
`vecops`; `_C` is an expert/debugging escape hatch and mirrors C++ types
without Python convenience behavior.

Importing `vecops` does not import NumPy or Torch. Dtype conversion recognizes
those frameworks only when the application has already loaded them.

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

A wheel contains a relocatable compiler SDK below `vecops/_sdk`, including
headers, `libvecops`, and the CMake package. An editable installation maps
Python modules and headers to the checkout while keeping `_C`, libraries, and
generated CMake state in its isolated build/install directory; it does not
place `.so` or `.a` files in `python/vecops`. The packaged CMake target also
exports vecops' transitive runtime dependencies, including OpenMP when enabled;
JIT modules that link the static archive therefore do not need to rediscover
or spell those dependencies themselves.

## Signature-derived recipes

`@vecops.jit` reads a declaration signature and returns a lazy `JitKernel`.
Decoration validates the schema but does not run a compiler. A relative source
path is resolved against the Python file defining the decorated function;
interactive definitions must use an absolute path or `source_root`.

```python
import vecops
import vecops.typing as vt

@vecops.jit(
  source="kernels/scale.cpp",
  compiler=compiler,
  symbols={"B": vt.Dynamic(lower=0)},
)
def scale(
  input: vt.In(vecops.float32)["B D", "stride_D 1"],
  output: vt.Out()["B D", "Dynamic<1,D,1048576> 1"],
  factor: float = 2.0,
  *,
  ComputeType: vt.DType = vecops.float32,
) -> None:
  ...
```

Positional and positional-or-keyword parameters become ordered kernel
parameters. `bool`, `int`, and `float` scalar annotations map to ABI bool,
int64, and float64. Keyword-only parameters must use `vt.Const` or `vt.DType`
and become specialization values instead of runtime scalar parameters.

Tensor access is explicit:

- `vt.In(dtype)` requires read access and gives `__kernel__` a const tensor;
- `vt.Out(dtype)` requires write access and is caller-allocated;
- `vt.InOut(dtype)` requires read/write access and is mutable;
- omitting `dtype` creates an anonymous dtype specialization;
- a string dtype names a dtype specialization symbol.

Concrete dtype inputs accept `vecops.float32`, `np.float32`,
`np.dtype("float32")`, and `torch.float32` without making either framework a
base dependency.

## Dimension strings

Annotations accept a shape string and an optional stride string. Omitting
strides produces one anonymous `Const` stride per axis.

```text
"B N"                              named Const dimensions by default
"32 N"                             fixed and named dimensions
"Const 1"                          anonymous Const and fixed stride
"Dynamic N"                        unconstrained runtime dimension
"Dynamic<8,1,MaxB> N"              aligned/bounded runtime dimension
"Dynamic<1,N,1048576> 1"           symbolic lower bound
```

Every symbol referenced by a shape, stride, dtype, or Dynamic constraint is
automatically added to the `KernelDef` with the correct kind. Dynamic
alignment and bound symbols must be Const values. Arithmetic expressions are
not part of the grammar.

The optional `symbols` mapping overrides the default interpretation of a
dimension identifier. Mapping `"B"` to `vt.Dynamic(lower=0)` makes every `B`
occurrence a named runtime dimension. Repeated occurrences must agree, but the
observed B does not enter the specialization key. Generated C++ exposes
`spec::B` as a Meta type alias rather than a constexpr value.

## Runtime tensor signatures

`vt.Tensor` reuses the dtype and dimension parser for framework-neutral eager
validation. It accepts loaded Torch tensors, NumPy arrays, and
`vecops.TensorMeta` without importing either framework. Omitting dtype or
strides leaves that property unconstrained.

```python
@vt.check_tensors(symbols={"B": vt.Dynamic(lower=0)})
def transition(
  act: vt.Tensor(vecops.bfloat16)["*B N C"],
  first_weight: vt.Tensor(vecops.bfloat16)["C 2*M"],
  second_weight: vt.Tensor(vecops.bfloat16)["M C"],
):
  ...
```

The runtime-only `*B` token binds B to the product of zero or more leading
dimensions, and `2*M`/`M*2` checks a positive integer multiple of an already
inferable symbol. These forms deliberately do not lower into a JIT KernelDef;
JIT annotations retain the non-arithmetic grammar above. Tensor validation is
skipped while Torch Dynamo traces; the registered operator still validates
its canonical KernelDef metadata when the graph executes.

## Calls and precompilation

Calling a `JitKernel` binds metadata, resolves the ordered recipe, loads or
compiles a cache miss, and invokes the caller-provided outputs:

```python
scale(input, output, 3.0, ComputeType=vecops.float32)
```

Use `TensorMeta` to compile without allocating tensor storage:

```python
scale.compile_for(
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

## Framework registration

A qualified framework name and ordered recipe sequence create one logical
operator. Candidate matching and invocation occur in C++, not through a Python
dispatch callback.

```python
run = vecops.ops.torch.register(
  "my_library::scale",
  [float32_recipe, float64_recipe],
  compiler=compiler,
)
```

Registration only declares a content-addressed bridge task. Call
`vecops.ops.torch.compile_pending(parallelism=N)` to build all pending bridges
explicitly; otherwise the first real wrapper call materializes the current
pending set. `vecops.precompile()` also materializes bridges after its
storage-free model trace, before returning to ordinary execution.

Torch's internal schema marks `Out`/`InOut` tensors mutable and returns `()` by
default. `register(..., return_outputs=True)` additionally gives the internal
op unannotated Tensor returns for older TorchInductor custom-op wrappers that
reject void results. Mutation remains declared on the caller-owned arguments;
the internal return is an implementation signal and the public wrapper always
returns the original output tensor objects. Generated bridges register CPU and
Meta implementations, allowing the mutable wrapper to participate in
`torch.compile`; the internal `torch.ops` name is not the public vecops calling
convention. At load time vecops supplies both the process-local operator handle
and the current `_C` ABI entry-point address. The bridge therefore has no
link-time dependency on a particular `_C` path and can be reused by compatible
installations while still entering the correct process-local registry.
`VECOPS_TORCH_BRIDGE_DIR` may place this stable registration cache separately
from the kernel artifact root selected by `VECOPS_CACHE_DIR`.

Generated bridges are ordinary shared libraries rather than Python extension
modules. They expose a small C setter for the process-local operator handle, so
compilation does not include `torch/extension.h`, pybind11, Python headers, or
`torch_python`. CMake obtains Torch include/library paths and the libstdc++ ABI
mode from the active Torch installation. Loading the DSO runs its
`TORCH_LIBRARY_FRAGMENT` initializers before vecops supplies the handle. A
cache-root file lock coalesces bridge writers across processes; the cache is
rechecked after acquiring that lock.

Generated Torch bridges build their `VecopsValue` and specialization arrays in
fixed-size stack storage. Tensor size and stride pointers refer directly to
the synchronous ATen arguments instead of being copied into temporary vectors.
The native extension also caches exact, successfully resolved invocations per
thread, so steady-state calls bypass handle-registry locking, schema binding,
provider lookup, and workspace revalidation. Calls with different metadata or
explicit specializations automatically take the slow path and populate a new
entry. `VECOPS_TORCH_PREPARED_CALL=0` disables this optimization for A/B tests.

NumPy uses the same ordered registration model:

```python
run = vecops.ops.numpy.register(
  "my_library::scale",
  [float32_recipe, float64_recipe],
  compiler=compiler,
)
```
