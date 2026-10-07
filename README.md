# vecops
CPU operator extension using native SIMD &amp; matrix extension.

## Development status and branches

Vecops is under active development (currently 0.1.0). APIs, kernel ABI,
build options and performance characteristics may change before a stable
release. Check the documentation for host and ISA requirements.

`main` is the primary development branch. `feat/kupl-mma` contains the
experimental KUPL-derived SME BF16 provider; it has its own licensing
notices and is not implicitly included in `main`.

## Documentation

- [Code organization](docs/Architecture.md)
- [Runtime and artifact ABI](docs/Runtime.md)
- [Native source-kernel compiler](docs/Compiler.md)
- [Python JIT API](docs/Python.md)

## CMake SDK

The default build produces the C++ core and the optional C++ kernel compiler;
tests and benchmarks are opt-in:

```sh
cmake -S . -B build \
  -DVECOPS_BUILD_TESTING=OFF \
  -DVECOPS_BUILD_BENCHMARKS=OFF
cmake --build build
cmake --install build --prefix /path/to/vecops-sdk
```

Installed consumers use `find_package(Vecops CONFIG REQUIRED)` and the
exported `vecops::vecops`, `vecops::compiler`, and
`vecops::optimize_for_kernels` targets. Loadable kernels can be declared with:

```cmake
vecops_add_kernel_library(
  NAME my_kernel
  SOURCES kernel.cpp
  TARGET_ARCH Native
)
```

`TARGET_ARCH` accepts the architecture names supported by the host family,
including `Scalar`, `AVX512`, `Native`, `SVE`, and `SVE2`.

## C++ kernel compiler

`vecops::compiler::Compiler` generates CMake projects and launches CMake
directly as an argv subprocess (never through a shell). A
`KernelBuildRequest` supplies the source files, optional include/link policy,
explicit build and artifact directories, target ISA, SDK location, and a
`ToolchainSpec`. Explicit `c_compiler`, `cxx_compiler`, and `toolchain_file`
values are passed to CMake; when omitted, CMake uses its normal environment and
platform selection. `compile_batch()` gives every request its own generated
fragment, binary subdirectory, and artifact directory beneath one top-level
project. Vecops is imported once and one CMake-generated scheduler sees all
translation units. `compile()` is implemented as a one-task batch.

The SDK can be either a Vecops source checkout (`SdkLayout::SourceTree`) or an
installed package (`SdkLayout::Package`). Compilation performs no artifact
lookup or caching. The artifact directory must be empty, and reuse of an
existing generated project or build tree must be explicitly enabled.

### Declarative source kernels

The high-level compiler accepts a C++ file defining one global `__kernel__`
function. `compile_kernel()` generates `vecops::spec`, reconstructs typed
Vecops tensors, and emits the versioned ABI descriptor/run adapter around it.

```cpp
#include "vecops/Kernel.h"
using namespace vecops;

constexpr auto B = spec::B;
constexpr auto D = spec::D;
constexpr auto stride_D = spec::stride_D;
using ComputeType = spec::ComputeType;

void __kernel__(TensorLike auto input, TensorLike auto output,
                float64_t factor, bool add_one) {
  assert_shape(input, {B, D}, {stride_D, cint<1>});
  assert_shape(output, {B, D}, {any, cint<1>});
  // Kernel-owned temporary allocation is permitted. Generated artifacts
  // deliberately request zero external runtime workspace.
}
```

`KernelDef` describes tensor shapes, strides, dtypes, optionality, out-style
mutation, scalar defaults, and named specialization values. Dimension entries
accept fixed integers, `ConstInt` symbol names, anonymous
`DimensionDef::constant()` specializations, or runtime
`DimensionDef::dynamic(alignment, lower, upper)` declarations. Every Dynamic
argument may be a fixed integer or a `ConstInt` symbol. Binding first infers
the complete symbol environment and then validates symbolic bounds.

```cpp
using namespace vecops::runtime;

KernelDef definition(
  "example::scale",
  {
    TensorDef{"input", {"B", "D"}, {"stride_D", 1}, TensorDTypeDef("IOType")},
    TensorDef{"output", {"B", "D"},
              {DimensionDef::dynamic(1, "D", 1048576), 1},
              TensorDTypeDef("IOType"), false, TensorAccess::Output},
    ValueDef::typed<double>("factor", 2.0),
  },
  {
    {"B", SpecializationType::ConstInt},
    {"D", SpecializationType::ConstInt},
    {"stride_D", SpecializationType::ConstInt},
    {"IOType", SpecializationType::DType},
    {"ComputeType", SpecializationType::DType},
  });

KernelCall call(make_arguments(input, output, 3.0),
                {{"ComputeType", DType::Float32}});
auto executable = compiler.compile_kernel("kernel.cpp", definition, call);
```

Explicit inferable values are checked against tensor metadata. Anonymous
Const dimensions, anonymous dtypes, and optional-tensor presence participate
in the specialization key without becoming names in `vecops::spec`.

## Runtime model

The framework-neutral runtime is split into four layers:

1. `OperatorSchema` validates the logical operator call, including argument
   kinds, tensor rank, dtype, device, shape/stride constraints, access mode,
   and cross-argument relations.
2. `Operator` asks a `DispatchPolicy` for candidate `KernelRecipe` objects.
   Recipe matching and binding are side-effect free and may not invoke a build
   system.
3. An application-supplied `ExecutableProvider` resolves a bound recipe from
   an AOT store, cache, or JIT compiler. Disabling JIT is therefore a provider
   choice; the core runtime does not implicitly touch CMake.
4. `Executable` owns a reference-counted loaded DSO and exposes guarded
   workspace-query and invocation operations. It remains usable after its
   originating `Operator` has been destroyed.

`ArtifactExecutableProvider` supplies the standard provider policy. Its modes
are explicit: `CacheOnly` performs no build callback and creates no files or
directories; `ReadWrite` compiles misses and atomically publishes them;
`CompileOnly` builds without consulting or modifying persistent cache.
`Operator(KernelDef, ...)` normalizes a `KernelCall`, dispatches a
`SourceKernelRecipe`, resolves its artifact, and invokes caller-owned outputs.

Artifacts export the versioned C entry point `vecops_kernel_query_v1`. Its
descriptor contains the logical operator name, specialization key, ordered
parameter constraints, workspace callback, and invocation callback. This ABI
deliberately has no Python, NumPy, or Torch types. Framework adapters remain
optional: the NumPy binding converts arrays to runtime arguments, while the
Torch adapter registers logical operators directly with the C++ dispatcher so
the hot path does not call back into Python.

The current artifact format is one loadable library with its manifest embedded
in the ABI descriptor. A portable archive containing dependency DSOs, hashes,
and explicit host/ISA capability requirements is a separate next milestone;
it is not yet part of this implementation.

## LayerNorm/Softmax JIT CLI

With `VECOPS_BUILD_TESTING=ON` and `VECOPS_BUILD_COMPILER=ON`, build the ordinary
(non-GTest) `NormOperatorCli-Native` executable. For example:

```sh
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DVECOPS_BUILD_TESTING=ON
cmake --build build --target NormOperatorCli-Native

./build/tests/NormOperatorCli-Native \
  --op layernorm \
  --input-dtype bfloat16 \
  --output-dtype float32 \
  --accumulation-dtype float32 \
  --shape 2,3,1024 \
  --warmup 5 \
  --iterations 50 \
  --cache-mode read-write \
  --build-work-dir ./jit-work \
  --cache-dir ./jit-cache \
  --cmake /usr/bin/cmake \
  --c-compiler /usr/bin/gcc \
  --cxx-compiler /usr/bin/g++ \
  --target-arch Native
```

The CLI constructs a `KernelDef`, `SourceKernelRecipe`, artifact provider, and
`Operator`; the kernel source itself contains only `__kernel__`. It prints
first-call latency (including compilation/loading on a miss), timed mean and
population standard deviation, numerical error, and a final PASS/FAIL banner.
`--cache-mode` accepts `cache-only`, `read-write`, or `compile-only`;
`cache-only` never invokes the compiler and never writes either cache or build
state. LayerNorm and Softmax operate over the last dimension of an
arbitrary-rank dense-contiguous tensor. Supported input/output dtypes are FP16,
BF16, FP32, and FP64; accumulation is FP32 or FP64. Run `--help` for all
options.

The compiler and dynamic loader currently target POSIX hosts. The rest of the
runtime ABI is intentionally independent of CMake and Python frameworks.

## Python, NumPy, and Torch

See [`docs/Python.md`](docs/Python.md) for the complete annotation grammar,
cache/build directory policy, wheel/editable SDK layout, and registration API.

The low-level pybind11 API lives under `vecops._C`. The public layer provides
Python schema objects, automatic compiler/SDK discovery, dtype normalization,
and signature-driven JIT recipes. Importing vecops does not import NumPy or
Torch; their dtype values are recognized lazily when those frameworks are
already loaded.

```python
import vecops
import vecops.typing as vt

@vecops.jit(source="kernels/scale.cpp")
def scale(
    input: vt.In(vecops.float32)["B D", "stride_D 1"],
    output: vt.Out()["B D", "Dynamic<1,D,1048576> 1"],
    factor: float = 2.0,
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None:
    ...
```

Bare identifiers in shape/stride strings become named Const values. `Const`
means an anonymous compile-time value; `Dynamic` and
`Dynamic<alignment,lower,upper>` retain runtime metadata. Omitting the stride
string creates one anonymous Const stride per axis. Fixed dtypes accept vecops,
NumPy, or Torch dtype values; an omitted dtype is anonymous and a string names
a dtype symbol. Keyword-only `vt.Const`/`vt.DType` parameters are specialization
values rather than runtime scalar arguments.

`vecops.Compiler` defaults to a user cache directory, discovers explicit
`CC`/`CXX` and CMake tools lazily, and uses the SDK bundled under
`vecops/_sdk`. `cache-only` mode performs none of that discovery and creates no
filesystem state. `TensorMeta` plus `compile_for()` precompiles without tensor
storage.

`vecops.precompile(model, *inputs, parallelism=N)` performs a storage-free
FakeTensor trace by default, collects and deduplicates registered JIT calls,
and then prepares the resulting artifacts through one CMake build graph.
During collection wrappers return their caller-owned output tensors without
executing native code. `vecops.is_precompiling()` lets model code avoid
data-dependent checks or persistent caching during that trace;
`use_real_tensors=True` is available for models that cannot run with fake
tensors.

`vecops.ops.torch.register()` declares one C++ dispatcher bridge for the
logical operator. `vecops.ops.torch.compile_pending()` compiles all pending
bridges as separate targets in one CMake graph; the first real call also
provides a lazy fallback. Bridges are ordinary DSOs with a C handle setter and
do not require `torch/extension.h`, pybind11, or `torch_python`. Their internal
schema marks outputs as distinct mutable
`Tensor(a!)` arguments and returns `()` by default. `return_outputs=True` adds
internal Tensor returns for TorchInductor versions that reject void custom
operators. The public `vecops.ops.torch.<library>.<name>` wrapper returns the
caller's original output tensors in both modes. Per-shape kernel DSOs never
use `TORCH_LIBRARY`, avoiding duplicate registration. Named specialization
values are accepted as kwargs and omitted values are inferred by the same C++
binder.

```sh
python -m pip install '.[numpy,torch]'
```

NumPy arrays are adapted to non-owning CPU `TensorView` objects while retaining
their Python owners. Inputs may use positive element-aligned strides; writable
outputs must be writable and C-contiguous. `Executable.invoke()` releases the
GIL during the synchronous C++ execution path.

Torch operators are registered from an `Operator` and its `KernelDef`; output
storage is always supplied by the caller:

```python
import torch
import vecops

run = vecops.ops.torch.register("example::scale", [scale])
x = torch.randn(4, 1024)
out = torch.empty_like(x)
run(x, out, ComputeType=torch.float32)
```

An ordered recipe list is dispatched inside `_C`, without a Python callback on
the Torch execution path. Registration compiles one schema-specific CPU/Meta
adapter against the installed Torch version. The internal `torch.ops` symbol is
an implementation detail; public calls use the returned function or
`vecops.ops.torch.<library>.<name>`.
