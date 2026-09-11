# Native source-kernel compiler

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
the parent's environment; subprocess output is returned as batch-wide command
logs plus ordered per-task `BuildResult` values.

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

The compiler gets no persistent cache policy. `ArtifactExecutableProvider`
owns cache lookup/publication and invokes a compiler callback only when its
configured mode allows a build.

## Kernel source contract

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
