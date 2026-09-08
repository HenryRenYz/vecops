# Code organization

Vecops is a header-heavy operator library. Directory boundaries therefore
describe ownership and dependency direction rather than the usual split
between declarations in headers and implementations in source files.

## Directory roles

| Directory | Role |
| --- | --- |
| `include/vecops/ops` | Stable public entry points for complete operators. |
| `include/vecops/ops/details/<feature>` | Operator-owned planning, selection, and implementation details. |
| `include/vecops/kernel` | Reusable bound primitives and scheduling utilities shared by multiple subsystems. |
| `include/vecops/matmul` | Public Matmul domain vocabulary and its feature-owned implementation tree. |
| `include/vecops/tensor` | Tensor descriptions, layouts, access planning, and bound access sessions. |
| `include/vecops/execution` | Execution scopes and hardware-resource lifetime management. |
| `include/vecops/vec` | SIMD abstraction and architecture backends. |
| `include/vecops/runtime` | Framework-neutral logical schemas, recipe dispatch, artifact ABI, DSO loading, and guarded execution. |
| `include/vecops/compiler` | Optional native CMake compiler API; it creates artifacts but owns neither dispatch nor caching. |

## Ownership rules

1. Users include complete operators through `vecops/ops/*.h`. Public operator
   symbols live in `vecops::ops`.
2. An implementation used by only one operator remains under that feature's
   `details` tree. Performing vector computation alone does not make a file a
   shared kernel.
3. `vecops/kernel` contains allocation-free primitives that operate on already
   bound accesses and have more than one real consumer, plus generic loop and
   tiling machinery.
4. A kernel must not depend on a complete operator. Operators may use Tensor,
   Execution, and shared Kernel facilities; Tensor materialization may also use
   a shared bound kernel. None of these lower layers may include `ops/*.h`.
5. `details` headers are private implementation headers. Tests may exercise
   them when validating backend contracts, but application code must not rely
   on them as stable API.

## Runtime and compilation boundaries

The C ABI is split by protocol:

- `CallAbi.h` owns the tensor, scalar, call-frame, error, dtype, device, and
  status records shared by every ABI boundary.
- `KernelAbi.h` owns the independently-built kernel DSO descriptor and
  `vecops_kernel_query_v1`; it has its own kernel ABI version.
- `OperatorBridgeAbi.h` owns the process-local handle protocol used by
  generated framework adapters to call an `Operator`; it is versioned
  independently from kernel artifacts.

There is deliberately no umbrella ABI header.  Code must include the header
for the protocol it implements, which keeps an artifact DSO from accidentally
depending on the process-local operator bridge.

The runtime represents two distinct things instead of registering every
specialization as a separate operator:

- `Operator` is the stable logical operation. Its `OperatorSchema` may accept
  dynamic rank, dtype, shape, and stride metadata.
- `Executable` is one concrete specialization. Its DSO descriptor records the
  exact subset of calls that it accepts and every invocation is checked
  against that descriptor.

Between them, a `DispatchPolicy` orders `KernelRecipe` candidates and each
recipe binds invocation metadata to a specialization key. Matching and binding
must be free of filesystem, compiler, and build-system side effects. An
`ExecutableProvider` is the only policy boundary allowed to decide whether to
load AOT code, consult an application cache, or invoke the optional compiler.
Consequently a runtime-only deployment can omit `vecops::compiler` and cannot
accidentally start CMake.

Compiler-managed sources add a `KernelDef`/`KernelCall` binding step before
recipe resolution. A binding owns normalized positional arguments, the fully
inferred symbol environment, anonymous compile-time choices, and one
specialization key. Dynamic dimension constraints may use fixed integers or
named `ConstInt` values for alignment and bounds; all symbols are inferred
before those constraints are checked. Validation, generated `vecops::spec`,
descriptor constraints, and dispatch keys therefore consume one binding.

The standard `ArtifactExecutableProvider` has `CacheOnly`, `ReadWrite`, and
`CompileOnly` modes. Cache-only resolution never calls a compiler callback and
never creates locks, directories, staging files, or indexes. Read-write
publication uses a staging directory followed by rename. Its key includes the
recipe/source identity, canonical KernelDef, specialization, and an
application-supplied target/toolchain/SDK namespace.

The ownership chain uses shared lifetime at the load boundary:

```text
Operator -> Schema + Recipes + DispatchPolicy + ExecutableProvider
                                      |
                                      v
                              shared Executable
                                      |
                                      v
                                LoadedModule (DSO)
```

Destroying an `Operator` releases what it owns, but an explicitly retained
`shared_ptr<Executable>` remains callable and keeps its DSO loaded. This is the
intended fast path for callers that have already resolved and cached a concrete
specialization.

Framework integrations belong above this boundary. A Torch adapter can own
operator registration and `at::Tensor` conversion, while a NumPy adapter can
own Python-buffer conversion; neither type appears in the core ABI. Generated
adapter C++ may still be compiled into a framework-specific extension DSO so
that execution does not round-trip through Python.

Generated Torch bridges register each logical operator exactly once and call a
narrow framework C ABI exported by the Python runtime module. Per-shape kernel
DSOs continue to export only `vecops_kernel_query_v1`. Output tensors are
caller-allocated mutable arguments; internal Torch schemas use distinct alias
sets and return no tensor aliases, while the public Python wrapper returns the
original outputs for ergonomics.

## Current operator layout

```text
include/vecops/
|-- ops/
|   |-- LayerNorm.h
|   |-- Softmax.h
|   |-- Transpose.h
|   |-- Matmul.h
|   |-- MatmulPack.h
|   `-- details/
|       |-- layernorm/Operation.h
|       |-- softmax/Operation.h
|       `-- transpose/Selection.h
|-- matmul/
|   |-- Atom.h
|   |-- Config.h
|   |-- Family.h
|   |-- Packing.h
|   |-- Quantization.h
|   `-- details/
|       |-- planning/
|       |   |-- FamilyPlan.h
|       |   |-- FamilySelector.h
|       |   |-- Implementation.h
|       |   `-- families/
|       |-- tiled/
|       |   |-- LoopNest.h
|       |   |-- OperandController.h
|       |   `-- Tiler.h
|       |-- kernel/
|       |   |-- Kernel.h
|       |   |-- amx/
|       |   `-- sme/
|       `-- packing/
|           |-- Plan.h
|           |-- Kernel.h
|           |-- amx/
|           |-- sme/
|           `-- generic/
`-- kernel/
    |-- Loop.h
    |-- Tile2D.h
    |-- Transpose2D.h
    `-- details/transpose/
```

Matmul is large enough to own a domain subtree containing atom, packing, and
quantization contracts. Its complete operator still has the same public entry
point convention as every other operator: `vecops/ops/Matmul.h`. Packing is
exposed through `vecops/ops/MatmulPack.h`; the `matmul/details` tree contains
only private implementation machinery. `planning/` owns family selection and
whole-operation invocation construction, `tiled/` owns generic cache blocking,
`kernel/` owns bound matrix-multiply leaves, and `packing/` owns packed-format
contracts plus the plans and kernels that produce them. Packing plans are
stateless typed policies: operand-dependent implementation and resource
selection remain compile-time decisions without storing a prepared operand
invocation.

Transpose intentionally has both an operator and a shared kernel. The operator
normalizes operands, chooses resources, and binds DataAccess sessions.
`kernel::transpose2d_bound` performs only a bound two-dimensional transpose and
is also used by Tensor materialization. It therefore belongs to the shared
kernel layer rather than to the private Transpose implementation.
