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
|   |-- Packing.h
|   |-- Quantization.h
|   `-- details/
|       |-- Operation.h
|       |-- PackOperation.h
|       |-- amx/
|       |-- sme/
|       `-- pack/
`-- kernel/
    |-- Loop.h
    |-- Tile2D.h
    |-- Transpose2D.h
    `-- details/transpose/
```

Matmul is large enough to own a domain subtree containing atom, packing, and
quantization contracts. Its complete operator still has the same public entry
point convention as every other operator: `vecops/ops/Matmul.h`.
`vecops/matmul/Matmul.h` and `vecops/matmul/MatmulPack.h` remain compatibility
includes for code written during the earlier directory migration.

Transpose intentionally has both an operator and a shared kernel. The operator
normalizes operands, chooses resources, and binds DataAccess sessions.
`kernel::transpose2d_bound` performs only a bound two-dimensional transpose and
is also used by Tensor materialization. It therefore belongs to the shared
kernel layer rather than to the private Transpose implementation.
