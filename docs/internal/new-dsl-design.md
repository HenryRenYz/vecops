<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# Python kernel DSL design

## Status and implementation boundary

This document specifies the new source-less `vecops.language.template` frontend.
The source-backed frontend, ABI, specialization rules, batch compilation, Torch
bridge, and execution graph remain supported. Outer `T.jit` annotations remain
ordinary Python hints; actual values determine immutable normalized keys.

The first implementation covers structured control flow, tensor/tile views,
tagged virtual vectors, arithmetic/comparison/math/reductions, converting memory
accesses, and CPU C++ code generation. Layer normalization and softmax are written
in this DSL, not lowered to `vecops::ops`. Subsequent implementation sections
below extend this initial boundary with specialized transpose, hardware matrix
accumulators, matrix products/packing, dependency-aware CPU grid scheduling
and typed loop/branch state with lexical helper exits;
they supersede the corresponding phase-one TODOs. Instruction scheduling and an
exact register allocator remain later milestones.

### Common iterations and CPU matmul scheduling

The implementation now uses common IterationDomain/IterationContract,
ReductionSpec and LoopSchedule descriptors. Explicit For/Grid, logical
FoldLoop/AxisFold and target ScheduleLoop retain their different legality
contracts: prefix recurrences stay ordered, resident teams/barriers are
synchronization boundaries, and independent domains are not evidence of
non-aliasing. Shared effect/access analysis performs alpha-renaming, capture
resolution and runtime root-span/physical-layout versioning for pointwise
loop fusion. A generic loop-consumer transform removes private intermediates
at the original write site, so inclusive/exclusive scan ordering is preserved.

Builtin and independent neutral associative custom reductions share FoldLoop
and the established SIMD prologue/accumulate/horizontal-finalizer lowering.
Unrecognized coupled reducers remain ordered instead of requiring an unsafe
blanket reassociation. Independent axis reductions also share one loop and
source DAG. Dependent layernorm/softmax phases are not collapsed into a single
one-pass loop without an algorithmic transformation.

CPU matmul scheduling has moved before the C++ printer. The prepared IR owns
cache NC/KC/MC and vector/GEMV/AMX/SME domains, pack producer actions, K state,
init/store phases, and optional operand-pipeline prime/steady/drain. VL,
register-family widths and runtime cache budgets are target-bound parameters.
Vector/SME staging rotates named register banks; AMX respects its eight-tile
budget and uses bounded cursor/panel hints. The target cost policy keeps auto
off until a shape/processor policy is justified by same-round measurements.
This is load/MAC overlap, not parallel or asynchronous panel packing.

Validation includes `test_dsl_iteration.py`, `test_dsl_matmul_schedule.py`,
`benchmarks/dsl_iteration.py` with an independently written force-inlined C++
reference, paired MatmulBench metadata, and SVE/SME cross-compilation. Inspect
both generated source and object code: outlined validation helpers can spill
every live Vec even when there is no explicit DSL scratch allocation. Proven
SIMD chunk bounds eliminate redundant rank-one size checks; hot required
validation helpers are force-inlined. Full SIMD unroll groups replace clipped
widths with their proven full extents, while aliases/tails retain safe fallbacks.

#### 2026-10-04 validation and performance policy

All five implementation parts are present: common domain/state/effect/reducer
contracts; native fold normalization; guarded loop/consumer/axis-fold fusion;
pre-emission CPU matmul schedules; bounded panel hints and operand staging.
An associative custom scan still carries a sequential recurrence contract.
This is not permission to fuse across worker collectives or to parallelize a
prefix recurrence. General coupled custom reducers retain their ordered body.

The local CPU is frequency-limited to 1.8 GHz. Use same-round native timings,
not historical measurements at a different governor setting. With compilation
excluded and a reserved timing core, `build-dsl-iteration/final.json` records:

| workload | N=257 | N=4096 | N=65536 |
|---|---:|---:|---:|
| SIMD two-loop producer/consumer, DSL / hand-written C++ | 0.880 | 0.980 | 1.012 |
| associative tuple moments, DSL / hand-written C++ | 1.184 | 1.055 | 0.996 |
| ordered scan + affine consumer, DSL / scalar C++ | 0.537 | 0.448 | 0.445 |

The small moments case retains about 4.5 ns overhead; it is not described as
universally faster. The SIMD assembly contains two full chunks per iteration,
unmasked loads/stores and no hot helper call or vector spill. Moments share
the input loads and use named vector accumulators, with no hot helper call.
Artifacts: `build-dsl-iteration/{simd-final,reducer-final}.asm`.

FP32 row-B dot, three MatmulBench shapes (8x128x128, 64x256x256,
32x1024x256), plain and bias-ReLU, reaches 0.935--1.005 of the hand-written
four-row vector reference in `build-dsl-matmul/vector-final-off.csv`. x86 public
Matmul has no FP32 matrix atom, so that FP32 reference is explicitly a Vec
leaf, not the public AMX operator. BF16 comparisons use the public Matmul.
One-lane Vec loads now use a scalar load/fill, allowing scalar broadcast to
fold into memory FMA instead of masked-load/extract/broadcast sequences.

Staged FP32 is 1.014--1.038 of off in the paired final sweep. AMX panel hints
have small or shape-dependent gains (including regressions around 1%). Keep
`auto=off`; calibrated target profiles or explicit pass options can opt in.
No asynchronous or concurrent panel packing is claimed. SVE/SME validation
is local cross-compilation only: 42 primitive cases, four tuple-fold/scan cases,
and 24 staged fused matmul cases, including VLA/fixed VL and streaming-state
inspection. No ARM hardware execution or ARM performance claim is made.

Final local regression groups (conda `af3`; fresh build/cache directories):

| test group | result |
|---|---:|
| iteration, primitives, collectives/pipeline, grid | 63 passed |
| loops/reducers, state, passes, cost, attention codegen | 86 passed |
| iteration legality, matmul codegen, matmul policy | 27 passed |
| dynamic FP32/BF16 off/prefetch/staged and phased cache fusion | 8 passed (native Clang 19) |
| explicit AMX fused K=0/1/3/31/32/33/65 tails | 2 passed (GCC 13) |
| fixed one-lane converting/snapshot/masked loads | 3 passed (GCC 13) |

After retaining explicit reducer descriptors, ordered/coupled tuple reducer
and scan cases were also rerun (3 passed), along with IR/attention/workspace
rules (13 passed). Required generated lambdas, including the constexpr packing
selector, use the inline macro. Cache fusion was recompiled and executed with
GCC after this final source-only macro adjustment (2 passed). Test groups
overlap; these counts are not summed into a distinct-test total. Dynamic BF16
auto-family artifacts also passed directed GCC checks, but the complete
eight-mode matrix group above uses Clang because those large dynamic fused
GCC translation units had excessive compile times. No runtime fallback or
different numerical tolerance was used to make that group pass.

## Compiler dependency decision

Phase one uses a small typed, structured Python IR and the existing C++ compiler
SDK. It adds **no MLIR/LLVM/TVM dependency**. Python's standard `ast` module is the
source parser; the existing native toolchain builds generated C++.

IR operations, effects, views and backend legalization are kept separate from
the AST and C++ printer. A future MLIR implementation may carry these semantics
using `arith`/`scf` plus custom vector, memory and resource dialects. It would
require a version-pinned LLVM/MLIR development toolchain, native bindings, build
and wheel integration, and custom passes/codegen. It is not necessary to make
the initial CPU kernels fast and is not silently required by runtime users.

## Programming model

There are two execution stages. The outer JIT factory runs Python and selects
specializations. Its symbolic predicate truth means "proven true". The inner
source-less template is parsed, never executed as arbitrary Python; unknown
predicates become runtime branches, not false. Immutable compile-time values
are partially evaluated. Unsupported syntax or calls produce source-located
errors rather than a silent fallback.

`@T.func` declares a DSL helper eligible for specialization and inlining. Calls
to arbitrary host functions, mutable captures, generators, exceptions, runtime
Python objects and reflection are not part of the kernel language. Dependencies
include helper bodies and captured immutable constants. Source-backed bodies
continue to be ignored.

## Types and values

* Scalar: explicit dtype, SSA expression, optional compile-time value.
* Vector: dtype plus `Fixed(N)` or `Scalable(P)` extent and a backend vector
  domain. A value retains its complete tag, including subword information.
* Tile value: shaped logical computation; it is not automatically a buffer or
  a promise that an entire cache-sized tile stays in registers.
* Tensor/tile view: storage identity, shape, element strides, offset, bounds,
  dtype and read/write effects. Slice/transpose are address transformations.
* Buffer: owned scratch storage with lifetime; nonescaping single-definition
  temporaries can be promoted to computation values.
* Mask: logical lane domain and predicate, separate from its physical encoding.
* Future backend types: AMX tile, SME ZA/slice, distributed GPU fragments,
  operand descriptors, asynchronous tokens and lexical resource proofs.

Fixed logical extents need not be powers of two. Scalable powers are not limited
by the C++ representation. Legalization may split a value into several Vecs or
strip-mine it into vector loops. Arbitrarily large values do not imply arbitrary
code expansion. Conversion preserves logical lanes, not byte size; bitcast has
a different contract. Negative scalable powers must still have a meaningful,
integral lane extent on every selected target.
The vector tag records its scalable dtype basis as well as the current value
dtype: a conversion changes element representation, not the original extent.

## Public vocabulary

`T.Vector(dtype, N=..., pow=...)` describes a virtual vector; N and scalable pow
are alternative extent forms. The default is scalable power zero.

`T.tile(tensor)`, tensor slicing, `T.transpose`, and `T.reshape` construct
views where legal. `T.load`, `T.store`, `T.copy`, `T.cast`, arithmetic, comparison,
`T.where`, `T.exp`, `T.log`, `T.sqrt`, `T.rsqrt`, `T.reduce_add`,
`T.reduce_max`, `T.reduce_min`, and elementwise `T.max(a,b)`/`T.min(a,b)` express
computation. Reductions specify their numerical association contract. Dtypes
include fp16, bf16, fp32 and fp64; computation dtype can differ from storage dtype.

`T.serial`, `T.grid`, `T.unroll` and Python if/while express structured control
flow. Logical grids are not synonymous with OS threads. The CPU grid scheduler
needs explicit task/effect verification and backend scheduling.

High-level operations such as dot, norm and softmax are DSL library functions
whose implementations remain visible to the compiler. They are not opaque calls
to the existing C++ operator library. Preserve structured loop and region
information when expanding helpers; do not eagerly flatten every loop.

## Native MMA extension (out of phase-one scope)

Use backend packages (`T.sme.mopa`, `T.amx.tdp`, future GPU operations) rather
than a misleading one-size-fits-all primitive. Infer opcode from operand and
accumulator types, signedness, encoding, shape, predication and target capability.
An unsupported combination errors; it does not silently cast or emulate.
Native instructions require native fragments. Large virtual dot regions are
explicitly decomposed by the selected DSL implementation.

AMX/SME/SIMD register banks are distinct. Moving a matrix accumulator to a vector
representation may cost ZA transfers or memory traffic. Express this conversion
in IR, including effects; never assume it is free. Future asynchronous matrix
operations require completion tokens before reading their accumulators.

## Blocking and execution

Keep task blocks TM/TN, cache/packing blocks MC/NC/KC, microkernel accumulator
blocks MR/NR, and native instruction extents IM/IN/IK separate. They may coincide
but are not required to. A dot may consume cache-sized regions while keeping
only a microkernel-sized accumulator live.

Logical program, logical scratch lane and physical worker identities differ.
Future CPU scheduling must account for packing replication, task granularity,
worker reuse and nested parallelism. Cross-program dependencies are phases or
execution-graph edges, not an assumed device-wide barrier inside a kernel.

## IR and lowering stages

1. TemplateIR: signature, annotations, source or AST body, frozen captures.
2. KernelIR: typed expressions, structured control, views and memory effects.
3. Region optimization: helper expansion, scalar/constant folding, view
   composition, scratch promotion, forwarding, fusion and dead-store removal.
4. Backend legalization: virtual extents to native vector chunks, masks,
   converting accesses, and eventually backend resource/operand representations.
5. Native optimization: scheduling/unrolling and register-pressure estimates.
6. CPU emitter: public vec APIs and structured C++ control flow.
7. Existing ABI adapter, compiler/batch builder, executable and framework bridge.

Phase one retains loops for large/dynamic tiles instead of mandatory vector
expansion. Arithmetic chains and converting accesses are fused at the chunk
consumer. Typed loop-carried state now expands bounded small/fixed vectors to
standalone locals and uses private scratch for general dynamic/large state.
Cost-driven scratch/recompute choices remain independent of language semantics.

## Data access and safety

Logical vector traversal and physical tensor strides are independent. The
compiler selects contiguous load/store when possible and strided/indexed
gather/scatter otherwise. Layout transpose is a coordinate permutation, not a
mandatory temporary. TODO: choose specialized transpose packing kernels when
the reuse/cost model makes them profitable.

Storage dtype and compute dtype are independently typed. Loads use
`vec::load_convert`; stores use `vec::store_convert` where a conversion can be
fused without altering rounding or intermediate arithmetic. Tail masks filter
memory accesses: no overread or overwrite, even for negative strides.

Buffer promotion requires nonescape, matching accesses and no intervening
possibly-aliasing effects. Inputs are not implicitly noalias. Copies across
overlapping regions require a snapshot or a safe traversal; overlap is not a
license for parallel scatter races. Broadcast reads are legal; repeated output
locations require an explicit policy or rejection.

For a nonempty ordinary strided allocation, lo=sum(min(0,(n-1)*s)),
hi=sum(max(0,(n-1)*s)), span=hi-lo+1 and logical origin=-lo. Empty extents,
integer overflow, alignment and specialized packed spans are separate checks.
External input/output strides are ABI contracts and cannot be silently padded.

## Optimization and numerical contracts

Eliminate virtual scratch where provable, but preserve required packing,
representation transfers, escaped buffers and cross-phase data. Softmax and
layernorm implementations may trade recomputation for scratch; eliminating all
temporary traffic is not always the fastest strategy. Avoid using a raw
sum-of-squares formula where cancellation damages layernorm accuracy.

Record compute dtype, conversion rounding, FMA contraction, reduction order,
math accuracy tier, NaN/Inf and subnormal behavior. Do not remove or reorder a
rounding conversion just because dtypes eventually match. Tail identities differ
between sum/min/max; inactive lanes cannot contaminate reductions.

Register-pressure estimates are made after native chunk legalization, by bank,
and include conversion/shuffle temporaries. C++ APIs such as exp may expand into
many instructions. Before direct instruction lowering, estimates are guidance,
not a proof of zero spill. Validate assembly and benchmark generated code.

## Backends, artifacts and caching

Separate TargetBackend (types/capabilities/passes/codegen), DSL OperatorLibrary
(visible implementations), and ExecutionBackend (build/load/launch). CPU C++ is
one artifact format, not a universal requirement. Future GPU/NPU backends may
use different representations or execution runtimes without changing the AST.

Cache identity includes DSL source, frozen captures, transitive helper bodies,
IR/codegen version, specialization, numerical/schedule options, target, SDK and
toolchain. Batch requests use independent environments, never global current
kernel state. Tensor pointers are bound at execution, not captured in artifacts.

## Validation gates

* Source-less control flow: dynamic if/while, nested loops, unrolling, scalar
  loop-carried state, helper specialization and located diagnostics.
* Access: tails, zero extents, transposed/strided/negative-stride inputs and
  outputs, tile/layout mismatch, dtype conversion, aliasing, scratch forwarding.
* Types: fp16/bf16/fp32/fp64, fixed and scalable vectors, masks and reductions.
* Compilation: stable cache identity, batch isolation, old source-backed tests.
* Layernorm/softmax: stable reference comparisons, generated vec/conversion
  operations, no calls to ops::LayerNorm/Softmax, no unnecessary scratch traffic.
* CPU performance: warmed representative widths/layouts, generated assembly,
  tails and conversion paths; distinguish measured results from guarantees.

Local x86 tests are the execution gate. ARM source syntax is inspected without
claiming hardware validation. MMA and dedicated transpose kernels are not
implemented in this milestone.

## 概念对应与有意保留的差异

下表是本项目的设计对应关系，不是各系统实现的一对一等价声明。
TileLang 的 tile/布局和结构化循环可参考其
[语言基础](https://tilelang.com/programming_guides/language_basics.html)与
[控制流文档](https://tilelang.com/programming_guides/control_flow.html)。
Triton 的逻辑 tensor、program、访存与归约词汇见其
[官方语言 API](https://triton-lang.org/main/python-api/triton.language.html)。

| 本设计 | TileLang / Triton 的相关概念 | vecops / CPU / 后续硬件映射 |
|---|---|---|
| 外层 `T.jit` 工厂 | specialization / JIT elaboration | 普通 Python 调度与独立缓存；不是机器内核 |
| Tensor / View | buffer、descriptor、pointer tensor | Tensor ABI + shape/stride/offset；不强制某个物理存储层 |
| Tile value | tile、逻辑 tensor block | 计算区域；可大于寄存器容量，不能等同于寄存器 fragment |
| Tagged vector | fragment / 分布式逻辑向量的相关抽象 | 本阶段对应 `vec::Vec<Tag>`；一个逻辑值可以对应多个机器值 |
| Mask | predicate / masked memory | AVX-512 mask、SVE predicate；掩码编码由后端决定 |
| `serial/grid` | 结构化循环、逻辑 program 网格 | serial 保持顺序；grid 经 CPU ownership/pass 与运行时别名检查后调度，不等于线程数 |
| 后续执行域 | block / warp / program | CPU logical task、OS worker、GPU block/warp、NPU core 分开建模 |
| 后续 native matrix value | native MMA operand/accumulator | AMX tile、SME ZA、GPU fragments；不伪装成普通 SIMD Vec |

缓存分块、向量分块、累加器分块和 native MMA 尺寸必须分别表示。
CPU 优先策略可以有 CPU 专用实现；通用层不要求 GPU/NPU 使用同样的分块、
同样的寄存器模型或 C++ 产物。跨平台的目标是共享逻辑语义与可替换的模块，
不是用一份无架构信息的实现保证所有平台都达到最优。

## 当前实现落点

| 文件 | 职责 |
|---|---|
| `language/template.py` | 捕获 source 或 AST；保持 source-backed 兼容 |
| `language/dsl.py` | 不可在 host 执行的词汇、`Vector` 描述、`T.func` |
| `language/_compiler/frontend/{capture,parser,diagnostics}.py` | AST/helper 捕获冻结、部分求值、静态检查、源码诊断 |
| `language/_compiler/ir/{types,kernel,vector,traversal}.py` | 无前端/后端/运行时依赖的类型、表达式 DAG、控制流、Vector IR 与遍历 |
| `language/passes.py` | 公共注册/pass manager façade，保持导入路径不变 |
| `language/_compiler/passes/manager.py` | 注册表、确定性顺序、分析失效、快照与每次运行上下文 |
| `language/_compiler/passes/kernel/{canonicalize,reuse,fusion,unroll}.py` | 按优化家族组织 kernel passes，不为每个小 pass 建一个文件 |
| `language/_compiler/passes/vector.py` | 通用 vector 展开、分组、类型合法化 |
| `language/_compiler/lowering/tile_to_vector.py` | tile consumer 转显式 Vector IR |
| `language/_compiler/pipeline.py` | 内置 pass 的唯一加载入口与 pipeline 装配 |
| `language/_compiler/backends/base.py` | 后端降级/产物边界，不要求所有后端经过 CPU Vector IR/C++ |
| `language/_compiler/backends/cpu/{backend,capabilities,passes,source,planning}.py` | CPU 编译驱动、类型映射、策略、ABI/schema 适配与发射前计划准备 |
| `language/_compiler/backends/cpu/cpp/{emitter,expressions,vectors,memory,control,support}.py` | C++ 发射状态、表达式/向量/访存/控制流与支持代码 |
| `language/library.py` | 纯 DSL layernorm / softmax，供优化器观察的算法实现 |
| `language/compile.py` | 公共编译入口 façade，保持导入路径不变 |
| `language/_compiler/{driver,cache,dtypes}.py` | 编译编排、产物持久化、core cache/identity、共享 dtype 规范化 |
| `language/_runtime/{kernel,binding,allocation}.py` | 执行/批量桥接适配、参数绑定、符号求值、带 stride/重叠的输出分配 |

仅私有包名带下划线，包内按职责命名。公共类型及 DSL 词汇不搬到私有包，
避免捕获身份和用户导入变化。IR 拥有显式 Node 标记；遍历/展开不能根据
dataclass 所在模块名判断是否属于 IR，否则类型拆分后会漏遍历。

CPU 发射分成 prepare 与 print：prepare 使用本次编译的 pass snapshot，
先降级 axis reduction 的主循环/尾部、别名退化路径、snapshot 拷贝；
print 只取计划、绑定 scratch 标识符和输出代码，不再次调用 pass/lowering。
未采用的备选计划不混入 emitted region 诊断。框架暂时仅暴露 source artifact
给原有 CPU build adapter；新增 GPU/NPU 时需提供对应 build/runtime adapter，
而不是让它们强行复用 CPU 的 tag、向量策略或 C++ 发射器。

前端不通过执行任意 Python 来 tracing：动态分支和 while 必须保留，所以
采用标准库 AST，而不是仅记录一次执行路径。字节码解析会引入 Python 版本
相关的栈与跳转恢复问题，当前没有必要。本阶段要求可 inspect 的 Python 源文件。

表达式按 DAG 而不是展开成树处理。标量捕获的 memo 仅在单次赋值的
capture 边界内有效，不跨越后续标量更新；变换后保留共享参数边。
`walk` / `loads` 保留重复使用次数，成本和启发式计数不能静默去重；
`walk_unique` / `loads_unique` 则用于状态存在性、依赖与别名集合，
避免多项式/helper 中的共享子图导致指数级编译时间。

后续后端模块应接收 KernelIR 与目标能力，返回其产物/执行适配器；不能把
SME streaming 状态、AMX tilecfg 或 GPU warp layout 写进 AST 解析器。
当前编译/执行适配器只有 CPU C++ 实现；GPU/NPU 的 codegen、resource/effect
验证、异步操作、执行器和产物格式仍需相应模块实现，并非本次已支持的后端。

### 目录重组验证（2026-10-03）

重组不修改 C++ 算子层、不新增 MLIR 或其他依赖，也不改变 DSL VERSION、
公共导入/序列化身份、pass 名字/顺序、默认策略及源码内容缓存键。
本机 x86、conda `af3`、gcc/g++ 验证：

- DSL/pass 全量 63 项、目录/边界/发射分离/快照隔离 7 项、
  运行时 23 项、language/graph/memory/planner 32 项，合计 125 项通过。
- 测试生成的 35 + 17 份源码与重组前逐字节一致；另有 16 组静态/动态
  LayerNorm/Softmax（含转换输出）源码、pass trace、Vector grouping 完全一致。
- 本机 Clang 19.1.3 对全部 54 个生成 ABI adapter 做 SVE VLA 与 512-bit
  两轮语法检查，108 次均通过；没有进行 ARM 运行/性能测试。
- 用相同已生成 C++、相同输入和输出缓冲区，对照重组前保留的 Python
  runtime 与重组后的 runtime：CPU 20、单线程、64 行 fp32，128/512/1024/4096
  列，长行另含 output +16B；11 轮随机交错、每轮 1200 次调用。
  10 组 after/before 为 0.995–1.007，未观察到明显调用性能退化。
  此项含 Python 参数绑定/native dispatcher，并非纯 microkernel 吞吐测试。

回归命令：

```bash
conda run --no-capture-output -n af3 python -m pytest \
  python/tests/test_dsl.py python/tests/test_dsl_passes.py \
  python/tests/test_dsl_architecture.py -q
conda run --no-capture-output -n af3 python -m pytest python/tests/test_runtime.py -q
conda run --no-capture-output -n af3 python -m pytest \
  python/tests/test_language.py python/tests/test_graph.py \
  python/tests/test_memory.py python/tests/test_planner.py -q
```

### 本阶段实际写法

```python
import vecops.language as T

@T.jit(out_idx=[-1])
def normalization(M, N, dtype, mode="layernorm"):
    @T.template
    def kernel(A: T.Tensor((M, N), dtype=dtype),
               C: T.Tensor((M, N), dtype=dtype)):
        for row in T.serial(M):
            x = T.load(A[row, :], dtype=T.float32)
            if mode == "layernorm":       # 外层捕获常量，编译期选择
                y = T.layernorm(x)
            else:
                y = T.softmax(x)
            T.store(C[row, :], y)          # 尾部掩码与存储转换自动生成
    return kernel

# 普通连续张量：k = normalization(*a.shape, a.dtype); c = k(a)
# 非连续张量：在 Tensor annotation 中显式声明对应 stride / Dynamic。
```

逐行运算并不把整个行固化成一组寄存器：消费者按 native vector chunk
降级。需要明确逻辑向量时可用 `v = T.Vector(T.float32, N=37)` 或
`v = T.Vector(T.float32, pow=9)`，再 `T.load(A[:T.lanes(v)], vector=v)`。
dtype 转换保留逻辑 lane 数，超大 pow 不生成非法 C++ Tag。

### 已实现的优化与安全条件

1. 形状/地址视图合成和常量折叠；正步长 region 切片裁剪，负物理 stride 保留。
2. 消费端逐 chunk 融合与公共子表达式复用；tile 不默认写入内存。
3. 完整、匹配映射的非逃逸 owned buffer 可转发。部分更新、不同映射读取和
   动态控制边界需要物化；转换到 buffer 的舍入不能删除。
4. cast/load 与最终 cast/store 仅在不会丢失中间舍入时融合。
   `fp32 -> fp16 -> fp32` 必须保留 fp16 舍入，不能直接变成 identity。
5. 同一昂贵表达式被归约及后续消费者复用时，规划可选计算缓存。
   CPU 当前门槛为四个 native vector 的元素量；更短区域继续重算。
   缓存写入融合进归约 pass，缓存按需分配并跨行复用，fp32/fp64 不额外清零。
   这是初始成本规则，不是对所有缓存层/向量宽度的最终最优性证明。
6. 同映射且不再读取旧值的原地逐元素操作不需要快照。错位重叠、原地转置，
   或跨写入保留的 tile 值进行运行时 span 判断并按需快照；span 判断保守覆盖空洞。
7. region scatter 的非重叠性使用按绝对 stride 排序的充分条件证明。
   零 stride/重叠写区域拒绝；复杂但实际无重叠的特殊布局可能被保守拒绝。
   两个轴时排序降为一次比较交换，避免小矩阵 guard 实例化通用 introsort；
   单例/空轴、负 stride、地址跨度和溢出检查的语义保持不变。
8. 动态 stride 生成连续/strided 两条访问路径；layout transpose 当前直接
   gather/scatter，生成源码和诊断明确标记 TODO 专用 transpose 内核。
9. 标量归约使用展开后各合法向量组的独立累加器及最终合并。
   相邻、同 shape/dtype、无依赖的归约最多两个合并为一次遍历，复用输入 load；
   不跨过内存副作用移动归约。当前上限是保守控制寄存器压力，不是精确分配。
10. 静态广播在编译期验证；动态关系生成运行时检查与 singleton 地址选择，
    不假定动态维度一定相等。reshape 的非负形状、连续性、元素量同样运行时检查。

load/store 显式 mask 与 tail mask 合并，masked-off 地址不访问。
广播读取允许重复位置；顺序独立的 region scatter 不能依赖重复位置写入次序。
当前 gather/scatter 是由规则 strided view 自动生成；任意 index-tensor 访存、
packed/swizzled layout、异步搬运和架构专用打包是后续扩展点。

### 数值与控制流边界

标量和 typed tile/vector 可跨迭代携带状态，动态 if 支持双方均定义的
scalar/tile/tuple 结果合并。tile 的赋值保持旧值快照语义，不等于 view alias；
循环状态保持 dtype、predicate kind、logical shape、显式 tag，动态 shape
关系需运行时验证。CPU 后端选择独立寄存器变量或私有 scalar scratch，
尚不是完整的全局物理寄存器分配/驻留证明。view 跨循环 rebinding 仍需
显式 storage。非法 rebinding 必须报错，不能悄悄采用最后一次编译遍历
的表达式。详见后续 typed-state 补全记录。`T.unroll` 仅支持常量界限，
有 1024 次展开预算，暂不支持其中的 break/continue。

归约按 native chunks 和水平归约结合，不保证与 Python 顺序或另一 ISA
位级一致。math accuracy 显式选择 strict/fast/estimate，不能无声明地降精度。
layernorm 默认使用中心化方差，避免 `E[x²]-E[x]²` 的大均值抵消问题。
显式 `variance='moments'` 可选择原 C++ 算子的算法并合并 sum/sum-of-squares
遍历，但调用方须接受其数值抵消风险；不会自动为性能切换默认数值算法。softmax
先减最大值、使用 nonpositive-exp 路径，再乘标量倒数。两者接受 rank-one
计算值；存储可用 fp16/bf16/fp32/fp64，用户通过 load dtype 选择累加精度。
复杂 NaN/min/max 选择与 subnormal 行为沿用对应 vec 后端，不能宣称跨 ISA
完全相同。空行不写数据；尾部归约使用各自的 identity，不能让 exp 的非活动
lane 值污染 sum。
DSL 的 assert 在 release 构建仍有效；低层标量索引要求调用方保证合法边界，
region 切片则裁剪到视图。kernel tensor annotation 当前要求显式 dtype，
不会将源码绑定支持的 dtype 推断或 optional tensor 能力误称为已经支持。

### 缓存与部署

AST/helper 的传递依赖及不可变捕获参与模板 identity；生成源码内容寻址，
保存于 `cache_dir.parent / 'dsl-sources' / hash / 'kernel.cpp'`。编译器配置
包括显式工具链与环境；实际 native SDK/编译产物继续由原有缓存系统管理。
批量请求使用各自的冻结环境，没有全局 current-kernel 捕获槽。
`cache-only` 只读既有源码和二进制产物，缺失时明确失败，不创建目录。

`generated_source` / `kernel_ir` / `source` 可用于检查实际降级，而不仅看
Python 算法文本。性能验收必须查看机器代码并排除首次 JIT 构建时间。

### 后续 MLIR 接入方式

若后续资源/布局/异步 passes 的复杂度使 MLIR 有收益，可将上述结构化 IR
转换为 `scf`、`arith` 与自定义 memory/resource dialect，再逐级降至 native
vectors/fragments。MLIR 本身也区分 virtual vector 与 hardware vector，见
[Vector dialect 文档](https://mlir.llvm.org/docs/Dialects/Vector/)。不应把
整个 cache-sized tile 直接硬编码成一个 LLVM/native vector。

届时需新增固定版本的 LLVM/MLIR SDK（headers、libraries、TableGen/pass
工具）、本项目 native IR bindings、构建/发布与 wheel 集成；Python bindings
是否使用上游包应明确固定版本。当前不新增这些依赖，也没有要求用户安装
TileLang、Triton、TVM 或 MLIR。

## 本阶段验收记录

下面 41 项 DSL 测试、37 个 ARM adapter 与性能表为第一阶段的验收快照。
后续 pass/Vector IR 改造的验收单独记录于本文末尾，不将旧性能数据冒充新结果。

执行环境：本机 Xeon w9-3495X、AVX-512、conda `af3`、系统 gcc/g++。
没有在 ARM、GPU、NPU 实机运行，也没有实现 MMA、专用 transpose、
多线程调度、指令级调度或精确寄存器分配。本次不修改 AF3 仓库。

### 正确性与兼容性

```bash
conda run --no-capture-output -n af3 pytest python/tests/test_dsl.py -q --tb=short
conda run --no-capture-output -n af3 pytest python/tests/test_runtime.py -q --tb=short
conda run --no-capture-output -n af3 pytest python/tests/test_language.py python/tests/test_graph.py python/tests/test_memory.py python/tests/test_planner.py -q --tb=short
```

DSL 41 项通过；原有回归 55 项通过（runtime 23 项，其余 32 项）。
runtime 测试须在独立进程执行：其中有现有的“不预先导入 torch”断言，不能
接在已导入 torch 的测试后运行。测试覆盖 Native，以及 Scalar/AVX2 目标，
不是只检查生成源码文本。

覆盖零长度、尾部、动态/分块切片、动态广播、转置、正负 stride、四种浮点存储、
fp64 计算、转换中间舍入、masked load/store、全域与单轴归约、控制流、虚拟
向量超大 pow、别名快照、owned buffer 转发/物化、捕获冻结、批量编译隔离、
只读 cache-only、release assert 与错误诊断。layernorm 与 softmax 使用
NumPy/Torch 数值参考；没有使用现有 C++ 算子作为 DSL 的实现。

使用本机 Clang 19.1.3 和隔离的 Ubuntu AArch64 sysroot 对测试生成的 37 个
ABI adapter（包含实际 DSL 源码）进行两轮语法检查：SVE 512-bit、SVE VLA
各 37 个通过。sysroot 下载于 `build-dsl-cross`，不安装到系统，也不是
DSL 用户的新依赖。核心参数如下（另传 vecops include 与 libstdc++ sysroot）：

```text
clang++ --target=aarch64-linux-gnu --sysroot=<sysroot>
  -march=armv9-a+sve2+bf16 [-msve-vector-bits=512]
  -std=c++20 -O3 -fsyntax-only <vecops_adapter.cpp>
```

这是跨目标语法检查，不是 ARM 链接、运行或性能验证，不涉及 SME/MMA。

### 性能观察与复现

本节下面的旧表使用不同的 DSL/C++ 输出分配地址，保留作历史记录，
不能作为稳定 speedup 结论。新版脚本共用同一个输入/输出缓冲区，
默认 page-aligned；可用 `--input-offset 16 --output-offset 16` 检查偏移。
计时改为 7 轮随机交错顺序，不含首次构建。短行仍受 native ABI/参数个数影响，
不是纯计算 microkernel 对照。目录重构不改变 kernel 算法或生成源码。

```bash
VECOPS_NUM_THREADS=1 conda run --no-capture-output -n af3 taskset -c 20 \
  python benchmarks/dsl_norm.py --iterations 500
```

64 行、fp32、连续布局、单线程、预分配输出，排除首次 JIT，20 次预热后取
5 组各 500 次的中位数。DSL 与原 C++ 都走 native ABI dispatcher；C++ 路径
仍包含其现有 one-shot workspace/session 成本。构建产物在 `build-dsl-bench`。
下表是一次实际测量，单位 µs；不是纯指令级 microkernel 比较，也不代表
多线程或所有 dtype/layout。moments 选择与现有 C++ LayerNorm 相同的算法。

| 算法 | 列数 | DSL | 原 C++ | Torch | DSL / C++ |
|---|---:|---:|---:|---:|---:|
| layernorm-moments | 128 | 13.309 | 20.600 | 8.404 | 0.646 |
| layernorm-moments | 512 | 16.303 | 22.859 | 18.010 | 0.713 |
| layernorm-moments | 1024 | 20.314 | 29.002 | 30.544 | 0.700 |
| layernorm-moments | 4096 | 65.428 | 89.694 | 110.919 | 0.729 |
| softmax | 128 | 14.927 | 22.606 | 5.090 | 0.660 |
| softmax | 512 | 22.467 | 28.112 | 14.651 | 0.799 |
| softmax | 1024 | 34.985 | 36.847 | 29.856 | 0.949 |
| softmax | 4096 | 117.627 | 128.894 | 115.866 | 0.913 |

默认 centered 方差有额外遍历：另一轮 4096 列测量为 DSL 78.382 µs、
C++ moments 71.230 µs（1.100 倍）。其数值更稳，不能将不同算法的差异
掩盖为无条件“零退化”。4096 列 C++ 在不同轮次约 71–90 µs，也说明需
多轮测量而不能把单次比值作为架构保证；Torch 在短行上仍可更快。
另一轮 4096 列 softmax 为 DSL 117.902 µs、C++ 117.974 µs，基本持平。

生成代码和机器码可见 AVX-512 向量算术/FMA、masked converting accesses、
连续与 gather/scatter 路径，没有调用 `ops::LayerNorm`/`ops::Softmax`。
当前编译器仍可能为复杂分支/转换产生寄存器 spill；没有宣称零 spill。
后续性能门槛须按目标 CPU、算法数值契约、布局与 dtype 分开建立。

## Pass 与 Vector IR 改造

### 注册与 pipeline

`language/passes.py` 提供 `register_pass`、`PassManager`、`PassContext`、
`PassResult`。名字唯一，支持 `before`/`after` 拓扑排序、循环约束诊断、
显式 pass 名单与 per-run options。分析缓存属于本次运行，变换默认使分析
失效；`PassResult` 可声明不变或保留部分分析。没有 current-kernel 全局状态。
一次编译冻结 kernel、vector 与 CPU memory pipeline 的注册表快照，批量请求相互独立。
改变注册表后，已存在的 manager 不变化；已有 `T.jit` 工厂须 `cache_clear()`
才能重新 elaboration。缓存继续使用生成源码内容 identity。

默认 kernel pipeline：

1. `canonicalize-memory`：合法的 cast/load 融合，保留中间舍入。
2. `expression-cse`：复用相同 operand DAG 的纯表达式；不按地址合并不同 load。
3. `plan-computation-reuse`：昂贵表达式的跨归约复用规划。
4. `fuse-reductions`：同域、同 dtype、同 unroll、无结果依赖的相邻归约合并。
5. `fuse-reduction-elementwise`：独立 side output 融合进归约遍历。
6. `fuse-elementwise`：相邻同域 stores 共用向量遍历，带跨 store 的运行时别名保护。
7. `unroll-serial`：独立操作的 per-operation 复制、局部变量重命名及尾部保序。
8. `refresh-computation-reuse`：为副本重新规划生命周期，不能混用两行的 exp 缓存。

`T.compile(..., pipeline=[...], pass_options={...})` 可选择 kernel passes。
默认融合归约上限为 2，可设置 `max_fused_reductions`；该上限是压力控制策略，
不是语义限制。source-backed C++ 不通过这些 DSL passes。

### 明确的中间层

现在是 **KernelIR → VectorKernel / VectorRegion → VectorOp / OperationCopy /
VectorGroup → C++**。每个 VectorOp 有向量类型、原始表达式和显式 operand
依赖。Vector IR 不包含 C++ 类型字符串，`compiled.vector_ir` 可以直接检查。
用户侧 `T.Vector` 的 dtype/N/pow/tag 与算术、转换、访存及归约继续可用；
tile 只是逻辑计算域，CPU 消费端先建立 Vector IR，不再直接打印整个 tile DAG。

Vector IR 还包含显式 `VectorEffect`：初始化累加器、逐 chunk 累加、
store_convert、横向归约/向量累加器合并。归约并非藏在 C++ 打印器中的
一个不透明算子，插件在运行时可以检查这些前/后操作与 effects。

通用 vector pipeline：`unroll-vector-operations` → `coalesce-vector-groups`
→ `legalize-vector-types`。前两者区分逻辑副本与合并组，最后计算 dtype rebind
所需的 POW headroom；发射器用当前目标的 `VEC_MAX_POW` 选择合法分组。
同一份生成源码既可供 x86 编译，也可供较小 MAX_POW 的 SVE 编译。

CPU 专用 pass 单独注册：`cpu-plan-full-chunks` 将完整组与 masked tail
分离；`cpu-choose-vector-chunk` 在自动 tile、unroll=2、较长行的昂贵数学
遍历或单累加器归约中选 P=1 的 base chunk；`cpu-legalize-math` 将复杂数学的多 word
请求拆成独立 native-word 计算后重组，访存仍保留合并后的请求。
不修改显式 Vector tag 或用户的逻辑展开次数。这些策略可分别通过
`cpu_full_chunks`、`cpu_math_chunk_pow` / `cpu_reduction_chunk_pow`（0 禁用）、`cpu_wordwise_math`
覆盖；不是所有 CPU/算法/布局的最优性保证。
完整 chunk 直接打印 VectorOp；尾部的空组保护使用内联 lambda，避免将
访存/向量运算变成循环内的 C++ ABI 调用与结构体暂存。

### 展开与合并语义

`T.serial(..., unroll=2)`、`T.reduce_add/min/max(x, axis=..., unroll=2)` 的
默认次数为 2。1 禁用展开；当前显式复制有 1024 的代码量预算。
`T.layernorm` / `T.softmax` 同样接受并转发该参数。
旧 `T.sum` 与 unary min/max 删除，`T.min(a,b)` / `T.max(a,b)` 仅逐元素。

例如 load → mul → add → store，展开 7 次是先 load0…load6，再 mul0…mul6，
再 add0…add6，最后 stores；不是先跑完整 body0 再跑 body1。
连续 lane 域中，7 个 P 向量合并为 4 个的 P+2、2 个的 P+1、1 个的 P。
P+2 超过 MAX 时再分成较小组，不能先生成非法 native Tag。
dtype 转换按 logical lanes 重绑定，必须把较宽计算/内存类型的 POW 需求一起算入。
expand load 与 narrow store 因此可接收合并后的宽请求，由现有公开转换访存
API 合并内存侧操作。尾部组为零时不构造越界地址。

公开 Scalable Vector 的原始 POW 与 lane basis 保留在 IR；合法时可作为
chunk base power。超过当前 target 能力的值使用可流式的小向量 pieces，
不创建指数级列表，也不把整个大 tile 强制物化。`VectorGroup.repeat` 表示
大向量的合法重复 piece 数；CPU 可继续选较小 native chunks 控制压力。

非 lane 连续的 reduction-axis 副本保留独立 lane 域，不能伪装成一次普通
连续 load；它们仍按操作展开，有独立累加器与最终合并。serial 的循环携带
依赖、control/effect 边界保留有序循环；未知输入/输出重叠走原顺序 fallback。
目前 parallel 仍是后续功能，不能把相同展开规则误写成已经实现并行执行。

### 融合边界与仍未迁出的工作

归约前的 elementwise DAG 直接进入 vector traversal；无依赖的 side output
可共用这一遍历。依赖整行 mean/max/denominator 的消费者不能直接前移：
softmax 与 centered layernorm 仍有必要的多阶段计算和可选缓存。
elementwise stores 合并必须验证跨 store 的读写和写写关系，否则执行原顺序。
别名快照、owned buffer 转发/物化与舍入规则不能因为 fusion 消失。

本次没有声称所有优化都已迁成 pass：owned buffer 前端转发以及后端的
尾部/别名保护仍在原模块，后续可通过更显式的 memory/effect IR 继续迁移。
寄存器分配、parallel、MMA、任意 indexed-memory 仍未实现。
专用 transpose 已在下述新阶段实现；上文旧阶段的验证记录保留历史语境。

### 本轮验证：pass、展开与显式 Vector IR

本机 x86、conda `af3`、系统 gcc/g++；没有远程实机测试。

```bash
conda run --no-capture-output -n af3 pytest python/tests/test_dsl.py python/tests/test_dsl_passes.py -x -q --tb=short
conda run --no-capture-output -n af3 pytest python/tests/test_dsl_passes.py::test_cse_keeps_shared_edges_through_parent_operations -q --tb=short
```

联合 60 项通过，新增 CSE 回归另行 1 项通过；原有 runtime 23 项与
language/graph/memory/planner 32 项通过，共 116 项。CSE 回归验证结构相等
但 identity 被替换的 operand 能穿过父节点，最终 Vector DAG 只计算一次。

新测试覆盖注册表快照/排序/分析失效、pipeline 插件与显式名单、
operation-major copies、7=4+2+1 与 MAX_POW 分解、显式 Vector tag、
unroll=1/2/7、正负 serial step 与尾部、三类融合、跨 iteration/store 的
别名保序、expand load/narrow store、CPU 策略开关，以及 fp32/fp64
空 tile 和各向量边界附近的数学归约/side-output。原有 Scalar/AVX2、
动态布局、转换舍入与捕获/批量编译测试继续通过。

使用前文相同的本机 Clang 19.1.3 / 隔离 AArch64 sysroot，对上述测试
生成的 54 个完整 adapter 进行 SVE VLA、512-bit 两轮语法检查：各 54 个
通过，包含实际 DSL 源码、math-word legalization 与尾部路径。这仍不是
ARM 运行/性能验证；无需连接 920f-X，也没有新增用户依赖。

本轮末次默认参数性能测量（64 行、fp32、单线程、taskset CPU 20、预分配
输出、排除 JIT，5×1000 次取中位数，单位 µs）：

| 算法 | 列数 | DSL | 原 C++ | Torch | DSL / C++ |
|---|---:|---:|---:|---:|---:|
| layernorm-moments | 128 | 13.608 | 20.885 | 8.335 | 0.652 |
| layernorm-moments | 512 | 16.431 | 23.824 | 18.064 | 0.690 |
| layernorm-moments | 1024 | 21.376 | 29.165 | 30.610 | 0.733 |
| layernorm-moments | 4096 | 68.296 | 89.754 | 111.553 | 0.761 |
| softmax | 128 | 14.968 | 22.382 | 5.132 | 0.669 |
| softmax | 512 | 23.213 | 29.008 | 15.040 | 0.800 |
| softmax | 1024 | 31.563 | 36.813 | 29.910 | 0.857 |
| softmax | 4096 | 141.717 | 117.457 | 124.012 | 1.207 |

默认展开次数 2 是 API 契约，不等于所有层级都应使用相同的最佳配置。
本轮 CPU 策略改善了最初较长 Softmax 的回退，但**默认 4096 列性能仍未
达到零退化目标**，不能用第一阶段的旧性能表替代这一结果。同一最终实现
另测 `--serial-unroll 1 --widths 4096 --iterations 1000`：Softmax 为
129.464 / 125.516 µs（1.031 倍），layernorm-moments 为
69.808 / 84.207 µs（0.829 倍）。行级展开改变缓存工作集与代码体积，
仍须按层级调参；单轮接近不等于长期无退化保证。后续需继续做 cache-aware
调度、寄存器压力分析和更细数学 DAG legalization。本轮没有修改 AF3
或已有 C++ 算子，也没有提交工作区改动。

## 性能诊断与无闭包归约发射

上表默认 4096 列 Softmax 回退不是 Python 执行开销或算法差异：双方都走
预分配输出的 native ABI，严格 exp，且该尺寸的现有 C++ 使用 offline
三遍算法；DSL 同样缓存 exp，没有额外重算整行 exp。

同进程、CPU 20、9 轮随机交错、每轮每配置 800 次的进一步消融中，
原 C++ / 默认行展开 2 / 行展开 1 分别为 120.804 / 146.207 / 139.580 µs。
单独关闭数学加宽、归约加宽或 native-word 数学后，行展开 1 仍约
139.6–140.8 µs；因此不能将前面的单轮 1.031 倍解释为行展开已解决回退。

### 两个已定位的原因

1. **归约发射的捕获边界隐藏了不变量。** 原打印器用未强制内联的
   `[&]` IIFE 包住完整归约，GCC 在 unroll=2/4 时将其 outline。
   exp 热循环机器码中每个 chunk 都通过 capture 再读 tensor/row offset，
   重载并广播 max，测试固定的 cache-enabled 标志。该热循环未发现 ZMM
   栈 spill，不能把此回退归因于寄存器溢出。诊断进程仅改为标量/metadata
   按值捕获、buffer 按引用捕获，保持其余计划不变；另一轮同轮对照中
   默认配置由 143.631 降至 119.318 µs，C++ 为 130.278 µs，正确性通过。
   说明关键是 C++ 优化可见性，而非 unroll=2 本身不合适。
2. **行级展开增加工作集，却没有提供细粒度跨行调度。** 当前展开的是
   每个 IR statement；一个归约仍是原子 statement，因此形成
   max0/max1/exp0/exp1/store0/store1。一个 fp32 4096 元素行为 16 KiB，
   两行同时存活的输入/exp/output 区域对本机 48 KiB L1 更不友好。
   相同 40000 次调用的进程级 PMU 对照中，行展开 2/1 的 L1 replacement
   约 3.313/1.804 billion，retired L1 misses 约 834/304 million。
   计数包含 Python 启动，不能当作纯 kernel 精确计数；但支持缓存压力
   差异。它是与捕获问题独立的次要因素，仍需 cache-aware 调度。

Buffer 对齐、冗余 alias guards、合法性优先的 grouping/成本模型也有
优化空间，但本轮没有量化它们的独立贡献，不将它们列为已证实的主因。

### 本轮发射约束

归约结果声明为 scalar/tuple local，归约主体放在普通词法代码块中，
不再生成捕获闭包。条件归约、scalar where/masked load 与 and/or 的
右操作数只在对应分支发射；包含归约的 while 条件放在循环头每次重算，
continue 保持正确语义。向量尾部/缓存选择进一步改为显式 Vec/Mask local
和 if；SVE sizeless local 也支持这种写法，无须捕获闭包。缓存分支中
的临时表达式不能泄露到分支外的 CSE 表。常规 LayerNorm/Softmax 的
生成源码不含闭包。需要保持重复求值的 for 条件这种 expression-only
位置仍可使用局部闭包，但所有生成闭包必须使用 CoreDefs.h 的
`VECOPS_INLINE_LAMBDA`，不依靠 C++ 编译器的内联启发式。

本轮只修改该发射策略，没有调整默认 unroll、grouping、数值算法、
AF3 或 C++ 算子，也没有增加依赖。

只去除归约闭包后的同进程交错复测（本机 gcc/g++ 13.3、CPU 20、64×4096 fp32、
9 轮随机顺序×800 次、排除 JIT、预分配输出）：

| Softmax 配置 | 时间 µs | 相对 C++ |
|---|---:|---:|
| 原 C++ | 129.804 | 1.000 |
| 旧默认 DSL（保留原生成源码重编译） | 143.761 | 1.108 |
| 新默认 DSL（行/归约展开均为 2） | 116.692 | 0.899 |
| 新 DSL（行展开 1，归约展开 2） | 108.492 | 0.836 |

默认配置改善约 18.8%，本次相对 C++ 快约 10.1%；对应同轮配对中位数
为 0.899。所有配置的结果与 C++ 比较通过（atol=5e-6、rtol=5e-5）。
机器码没有 outline 的归约 lambda；行展开 1 的 exp 循环可见 max 广播
已在循环前、固定 cache-enabled 分支已消除。这仍只是该尺寸/类型/布局
的测量，不是跨 CPU、跨 dtype/layout 的无退化承诺。

### 最终版本验证（归约、向量尾部和缓存读取均无闭包）

```bash
conda run --no-capture-output -n af3 pytest python/tests/test_dsl.py python/tests/test_dsl_passes.py -q --tb=short
conda run --no-capture-output -n af3 pytest python/tests/test_runtime.py -q --tb=short
conda run --no-capture-output -n af3 pytest python/tests/test_language.py python/tests/test_graph.py python/tests/test_memory.py python/tests/test_planner.py -q --tb=short
```

DSL/pass 63 项通过（438.57 s），原有 runtime 23 项、其他回归 32 项
通过，共 118 项。新增测试覆盖条件归约、scalar where/masked load、
and/or 短路、包含归约的 while/continue 条件重复求值，以及 expression-only
闭包的强制内联约束。8 个 norm benchmark case 的源码另行确认无闭包。
compileall、tracked/untracked whitespace 检查通过。

对 `/tmp/pytest-of-renyz/pytest-242` 本轮产生的全部 56 个完整 adapter，
用前述本机 Clang 19.1.3 / 隔离 AArch64 sysroot 检查 SVE VLA 和
`-msve-vector-bits=512`：各 56 个通过，包含显式 Vec/Mask local、tuple
归约和保留的 forced-inline for-condition 闭包；不是 ARM 实机验证。

最终同进程随机交错复测，测量约定与上表一致；默认行/归约展开均为 2，
LayerNorm 显式使用与 C++ 一致的 moments 算法，单位 µs：

| 算法 | 列数 | DSL | C++ | 同轮配对 DSL/C++ 中位数 |
|---|---:|---:|---:|---:|
| layernorm-moments | 128 | 13.446 | 20.782 | 0.647 |
| layernorm-moments | 512 | 16.249 | 23.773 | 0.684 |
| layernorm-moments | 1024 | 22.905 | 26.606 | 0.857 |
| layernorm-moments | 4096 | 65.287 | 85.181 | 0.768 |
| softmax | 128 | 15.097 | 22.312 | 0.677 |
| softmax | 512 | 22.366 | 27.885 | 0.802 |
| softmax | 1024 | 31.107 | 36.126 | 0.861 |
| softmax | 4096 | 115.810 | 128.662 | 0.900 |

该最终轮还加载旧默认 Softmax 的原生成源码：同轮 143.590 µs，新版
改善约 19.3%，相对 C++ 快约 10.0%。所有结果与 C++ 比较通过
（atol=5e-6、rtol=5e-5）。此处不将 centered 方差与 moments 的算法
差异混入性能比较，也不宣称所有平台/布局已达最佳性能。

## 专用 transpose 与架构状态（2026-10）

`T.transpose` 仍是零拷贝 view；`T.func` 的 `T.transpose_copy` 提供 rank-2
可见算法，编译期选择稠密单行/单列线性 copy，其余交给布局 lowering。
对自动 `T.store(T.transpose(...))` 也适用，并支持前导 batch 轴和
转置后的 elementwise DAG；不需要调用已有 C++ `ops::Transpose`。

新增 `cpu-memory` 注册 pass 阶段（默认 `cpu-plan-transpose`），在 C++
打印之前冻结并执行。该 pass 识别 memory permutation、证明连续目的布局，
根据 dtype/ISA/常量 shape 选择寄存器网络和 SME alternatives，生成
`TransposePlan`/`RegisterTranspose`/`PermutationOp`。通用 lowering 先构建
split/interleave/concat 的显式 vector DAG，CPU emitter 仅打印已准备好的
操作。数组只可用于标量 scratch/shape/configuration；Vec/Mask 操作均为
独立命名变量，VLA/VLS SVE 都不使用向量数组。

x86 采用 128-bit 蝶形基元及递归 split/concat；更宽 tile 和矩形 tile
保留完整 load、控制 GCC 寄存器压力。VLS SVE 采用相同网络；VLA SVE
利用不超过架构最小 VL（16B）的固定逻辑前缀构建 named sizeless network。
窄类型 rebind 到 i32 索引超出表示上限时，仅在该访存边界 scalar fallback，
不实例化非法索引 Vec。列尾部使用较小 FixedTag gather，而非无必要的大
ScalableTag 索引组。完整 source row 明确无掩码，转换由 load/store 融合。
仅有一个边界转换时，x86 可以先置换窄存储类型，再转换到目的类型；这是
转换与 permutation 的合法交换，不会省略存在中间舍入的转换链。

缺省采用 128-square cache block 和 input-major microtile 遍历，可通过
`cpu_transpose_cache_block`、`cpu_transpose_loop_order` 调整。小单寄存器
tile 有直接路径；已有输入输出重叠保护提升为区域级分流，alias fallback
读取快照，而不是把每次 vector load 都混入 snapshot ternary。generic
多 store fusion 不能绕过这些布局计划，暂时保留分别置换。
`cpu_transpose_defer_conversion` 可覆盖边界转换移动；缺省 cost policy 对
未知 shape/stride 的 byte→FP64 保留 compute-domain network，避免过多
宽 store/address chains。常量布局仍保留 narrow-bit permutation。

SME 简单加载转置在无需数值转换的边界直接使用 ZA horizontal load /
vertical store，不经过临时 Z 寄存器；需要转换的边界保留 load_convert +
horizontal write / vertical read + store_convert。浮点和整数用等宽无符号
位表示搬运，保留 NaN payload 和有符号零。输入/输出两个边界独立选路，
尾块直接使用 ZA 谓词。沿用原 C++ 已测量的常量 shape/dtype SME 选择及双行/双列流水阈值；
没有 ARM 实机，不能将这些阈值声称为本轮的新性能优化结果。显式 SME
要求目标 SME-FA64，automatic 在不满足条件时走 vector/gather。普通固定
VL 和固定 SVL 不匹配时，automatic 不进入 SME；显式路径编译诊断拒绝。

公开 C++ `vecops/vec/{SME,AMX}.h` 为原 details 中的指令封装提供稳定入口，
`vecops/execution/Resources.h` 提供词法 state/configuration requirements。
SME manual owner 内部禁止异常、分配和任意未内联计算调用；强制内联 noexcept
callback，state entry 之前做全部验证。AMX 是 tile state/memory API，
并非转置指令：Python `T.amx.region` 支持 configuration/zero/load/
stream_load/store，Linux XTILEDATA permission 按线程取得，失败抛出明确异常，
退出时 release。没有在本阶段加入 Python MMA。
raw hardware memory effect 边界之前 materialize forwarded owned scratch，
AMX store 对之后仍要使用的既有 load 做普通别名快照；显式 SME region
中的多个 transpose 必须输入输出独立，跨操作依赖在 entry 前拒绝。

测试/复现：`benchmarks/dsl_transpose.py` 自动读取
`TransposeBenchCommon.h` 的 133 组 shape/输入/输出/compute dtype，分别
Const/Any/Aligned，共 399 case；显式保持 C++ 等价 shape/stride 元数据，
复用页对齐输入输出、native repeat + compiler fence、随机交错同轮计时，
不计 JIT。本机 CPU 由用户限频，结果仅在该限频状态下解释，不能与旧轮
绝对耗时直接比较。ARM 只做 Clang 19 AArch64 object 跨编译和手动 SME
反汇编检查，不连接远程机器，也不声称实机正确性/性能已验证。
最终配对测试：本机 Xeon w9-3495X、CPU 2 affinity、gcc/g++ 13、conda
`af3`；intel_pstate/powersave，频率上下限 800/1800 MHz（未修改用户设置）。
缓存单独放在 `build-dsl-transpose-final`，完整 399/399 case 与同轮 C++
和 NumPy 参考逐元素一致。最终 DSL/C++ 耗时比几何平均 **0.8297**，
即耗时平均降低约 **17.0%**。不是所有 shape 均无退化：26 个 case
慢于 C++ 超过 10%，其中 10 个超过 20%；最大慢点是动态/Aligned
17×19 FP64（0.456 vs 0.344 µs，约 +32.5%），其余显著慢点主要是
3×5/32×64 小动态矩阵的边界和派发开销。对此保留后续优化空间，
不以总体平均掩盖逐 case 退化。

| Const case | 最终 DSL/C++ |
|---|---:|
| 1024² FP32→FP32 | 0.704 |
| 4096² INT8→INT8 | 0.543 |
| 4096² FP16→FP16 | 0.695 |
| 256² FP16→FP64 | 0.595 |
| 1024² FP16→FP64 | 0.826 |
| 1024² INT8→FP64 | 0.700 |

动态/Aligned 1024² INT8→FP64 的 compute-domain 策略在最终轮为
0.885/0.887；全部逐 case 数据为 `build-dsl-transpose-final/final.csv`。
基线使用本轮当前 C++，包括同样的 FP16/BF16→FP64 寄存器转换改进。

```bash
conda run --no-capture-output -n af3 python benchmarks/dsl_transpose.py \
  --build-dir build-dsl-transpose-final --jobs 6 --rounds 13 \
  --output build-dsl-transpose-final/final.csv
conda run --no-capture-output -n af3 python benchmarks/dsl_transpose_cross.py \
  --clang <clang19>/bin/clang++ --objdump <clang19>/bin/llvm-objdump \
  --sysroot build-dsl-cross/sysroot/usr/aarch64-linux-gnu --jobs 4
```

## 矩阵乘与跨 helper 融合（实现阶段）

公开入口为 `T.dot((A, B), output, acc=None)`；A 的逻辑形状是 M×K，B 是
K×N。另提供无输出参数的纯表达式 `T.dot(A, B)`，方便连接下一个乘法。
`T.library.matmul` 在 dot 上增加缓存分块、自动 packing、输出朝向派发；
dot 自身只负责 tile 内核和必要的指令格式 staging，不自动进行缓存 packing。
原有 C++ 算子没有成为 DSL 的黑盒实现，生成代码使用公开 vec/AMX/SME 原语。

融合观察的是内联后的 DAG，而不是 Python helper 的名字或调用边界。
rank-two layernorm/softmax 先拆成一次行统计和可重算的元素级 tail；例如
`T.library.layernorm(act)` 的统计不进入 MMA 的重复加载循环，而其最终仿射
可以进入 A 加载。CPU planning 记录输入操作成本、预计重用、物化条件和原因。
少重用或廉价算术保留融合；昂贵重复计算则先物化；packing 路径把变换放在
packing 的 load/convert 上。`cpu_matmul_fusion` 的 always/never 开关用于实测
成本模型，不承诺一个固定经验阈值在所有 CPU 和频率条件下最优。
动态 GEMV 会绕过 packing，因此按实际 M/N 选择 raw 或 packed 的重用
模型，不因使用 library 包装就假定 B 只生产一次。转换型 load 同样计入
操作成本，物化 buffer 在单次调用的循环中复用容量，而不是跨调用共享。

K 分块的路由显式区分 first/middle/last：原始 acc/bias 只在 first 载入，
middle 保持 native acc dtype，last 执行整个 epilogue（包括内联 helper），
因此不会重复 bias 或在每一块重复量化。NC/KC/MC 循环限定 packed panel
大小；B panel 在 MC 循环之外共享。可直接读取的 A 不做无意义的复制。
Column-major 输出允许交换 A/B 并转置整个前后 DAG；架构 packed ABI 不允许
仅改 shape 就解释成另一侧。

CPU 内核包含 widening-dot/FMA GEMV、小 M/N AMX register families，以及
一般 AMX/SME outer-product families。AMX B 的 K-contiguous 输入先按整行
load/convert，再把 32-bit K groups 通过显式寄存器 permutation 转成 VNNI；
不逐个 byte/half 做 gather。所有 vec/Mask 都是独立变量，staging/partial
buffer 则是标量内存。输入元素变换之后再次清零 K/spatial padding，避免
`0 * scale + bias` 在无效 lane 产生非零乘积。epilogue 转换使用实际结果
tag，宽转换按命名 fragment 拆分，避免 SVE 多字 Vec 聚合。
完整 AMX packing 面板与 K 块单独生成无 mask 的 load/convert，空间与 K
尾部独立处理，避免每个完整块都付出边界检查。动态空输出在资源入口前
短路，避免小矩阵 exact TILECFG 的 rows=0/cols!=0 非法组合；K=0 但
输出非空仍执行 acc 与 epilogue。

AVX-512 B packing 使用四阶段 dword 网络：完整 512-bit local unpack，
然后两级 128-bit lane shuffle。每一个输入、stage、结果都是独立变量；
SVE 和其他 x86 宽度仍使用公共 split/interleave/concat 网络。GCC 13
AMX load intrinsic 的 asm 缺少内存读取声明，公开 load/stream_load 封装
在其两侧加入编译器屏障，避免 freshly packed stores 被删除/重排；无
硬件 fence，不改变线程同步语义。另保留生成器临时 Expr 的强引用和
panel/ISA 分支的局部映射，避免 Python id 复用挂接错误的 C++ 局部地址。

显式预打包输入在缓存切片时保存原始 ABI shape 和逻辑 origins，MC/NC/KC
只改变计算区域，不重算物理 strides；acc 及 epilogue 的阶段语义与 raw
输入相同。公开 dense pack 也通过注册 memory pass 生成自动 transpose
计划，缓存分块的运行时 dense panel 绑定同一个预先准备的寄存器网络，
printer 不重新运行优化或 lowering。SVE VLA 的窄 prefix 根据 memory、
计算和输出三者的最大 lane 宽度限制，避免 f64→f32 等转换越过 128-bit
最小 VL；不是只检查结果 dtype。rank-one bias/scale/arange 随矩阵朝向
转置，并保留全局 panel 坐标。外部 AMX/SME packed buffer 绑定目标 ISA
及 SVL ABI，不能通过仅改 shape 或跨平台重解释来转换格式。
多个 sibling dot 和 dot→pack/reduction 通过产品边界物化，保留最终 dot
的 epilogue 融合，不将多个独立 MMA 偷偷合并成一个累加器。

寄存器 permutation 在 CPU-memory pass 准备，输入融合政策也是可注册
`cpu-plan-matmul`；前端不把 ARM/x86 的 packing 格式写死到通用 Tensor。
当前 instruction schedule 仍由 CPU C++ emitter 展开，不等同于完整的
架构无关 MMA dialect；新平台应扩展自己的 planning/lowering，而不是复用
CPU schedule。无需新增 MLIR 依赖。

基准入口 `benchmarks/dsl_matmul.py` 从 MatmulBenchCommon/Scenario catalog
读取 shape，独立覆盖 Const/Dynamic、bf16/浮点转换/整型、acc、bias+relu/clamp、
scale、silu、输入仿射。同轮公共 C++ Matmul、native repeat、随机交错顺序，
JIT 不进入计时。完整性能验收尚以实际报告为准，不能以正确性测试代替。
独立 packing 基准 `benchmarks/dsl_matmul_pack.py` 读取 MatmulPackBench 的
面板形状，覆盖 A/B、dtype 转换及输入仿射，逐字节检查 padding 与公开
C++ packed ABI。`benchmarks/dsl_matmul_packed.py` 把预打包放在计时之外，
区分叶内核代码生成差距与在线 packing/缓存策略差距。矩阵基准通过文件
锁串行占用同一计时核心，并将自身编译子进程排除于该核心及 SMT sibling，
不修改用户的频率限制或其他进程。
`benchmarks/dsl_matmul_cross.py` 对 SVE VLA/VLS、SME VLA/SVL 生成对象并做
SMSTART/SMSTOP no-call/mopa gate；此检查不等于离线 ARM 主机上的性能验证。

缓存信息扩展 `llc_bytes/llc_level/l3_present`，区分没有物理 L3 与旧 API 的
fallback `l3_bytes` 预算。SME f64 目标显式传递 `VECOPS_TARGET_SME_F64F64`，
以兼容接受 `+sme-f64f64` 但不定义对应 ACLE 宏的 Clang 19。
含数学 epilogue 的 GEMV 将数个完成的 horizontal sum 装入一个合法
向量，再执行 exp 等操作，不逐个结果调用 scalar libm；行/列输出、宽
seed/output、K 分块的阶段语义保持一致。简单 affine/relu 仍可走标量
结果路径，避免没有收益的额外组装。
GEMV 完整 K 边界在循环前计算，完整组与单个 masked K 尾块分开；
完整输出列组也在 K 循环外判断，避免依赖 GCC 自动 loop unswitching。
CPU planning 的 `cpu_gemv_columns` 调优参数控制独立列数（1–8，默认 4），
不改变顶层 tile/API 的形状限制。
实际生成 SME 状态 owner 的 DSL 编译单元在第一个头文件之前选择
`VECOPS_SVE_INLINE_MATH_SPECIALS=1`，强制内联 vector-only exp/log/rcp/rsqrt
特殊值路径，保持精度与特殊值语义；即使目标支持 SME，完全没有 streaming
owner 的普通 SVE 内核仍保留 cold outlining。状态 owner 通过统一 emitter
入口声明这一需求，不依赖生成字符串中的指令名扫描。
没有解除 SMSTART/SMSTOP 的 no-call 验证，也没有将 scalar libm 标记为安全。

### 矩阵阶段验证与性能记录（本机限频）

本机 Xeon w9-3495X，conda `af3`，GCC 13，Native/AVX-512/AMX。
用户设置的 800–1800 MHz 限频保持不变。CPU 2 上 native repeat，编译
进程排除 CPU 2 及 SMT sibling 58；同一核心的计时通过文件锁串行执行。
数字只比较同轮 DSL 与当前公共 C++ Matmul/MatmulPack，JIT、跨 shape
编译时间不进入计时，不能和未限频的历史绝对时间直接比较。

| 已完成范围 | case 数 | DSL/C++ 耗时几何均值 | 最慢比例 |
|---|---:|---:|---:|
| 完整 BF16 raw matmul catalog（86 shape，Const/Dynamic） | 172 | 0.9434 | 1.3080 |
| BF16 预打包 AB，dot/library、plain/bias+ReLU | 32 | 0.8894 | 1.2095 |
| 独立 A/B packing，BF16/FP32→BF16、plain/affine | 64 | 1.0258 | 1.6896 |
| INT8 与 U8×S8，256³ 的七种前后融合 pipeline | 28 | 0.6478 | 1.0990 |
| INT8 与 U8×S8，8×128×128、plain/bias+ReLU | 8 | 0.9124 | 1.0562 |
| INT8 与 U8×S8，行/列 GEMV 和 small-M（完整 K/N 组优化后） | 24 | 0.8353 | 1.2645 |
| 最新 BF16/FP32→BF16 行 GEMV，七种融合 pipeline | 28 | 0.5618 | 1.2453 |
| 融合 metadata 合并对照（最新 GEMV 替换较早对应 case） | 112 | 0.6411 | 1.2453 |

前三项分别为 `build-dsl-matmul/{catalog-latest,packed-latest,pack-latest}.csv`；
整型报告为 `fusion-integer-final.csv` 与 `integer-slim-final.csv`。
额外行/列整型报告为 `integer-groups-final.csv`，三个 shape 为
1×257×257、131×1×1025、16×48×256，Const/Dynamic、plain/bias+ReLU。
最新浮点 GEMV 为 `gemv-policy-final.csv`，shape 1×257×129；112-case 合并
是 `fusion-float-final.csv` 的 84 case 加 `fusion-integer-final.csv` 的
28 case，再以最新 GEMV 28 case 按完整 metadata key 替换，不是一次性
重新运行全部 112 case。完整 catalog/packing 是相应阶段的记录，后续
GEMV 微优化另有单独复测，不把历史记录冒充全部最新源码的一轮计时。
融合均值包含输入仿射 pipeline 的大幅收益，不能理解为普通矩阵乘普遍
快 35%；合并报告仍有 11 个 case 超过 1.1×、4 个超过 1.2×，动态
FP32→BF16+SiLU GEMV 为 1.2453×，动态 BF16 plain/accumulate 也有差距。
所有 case 先做 NumPy/当前 C++ 正确性检查；packing 比较整个物理 buffer，
包括 padding。不能声称所有形状无性能退化：完整 catalog 有 17 个 case
超过 1.1×、8 个超过 1.2×，最慢为 Const 64×1024×4096；预打包最大的
Const 128×4096×4096 library case 为 1.2095×。packing 最慢是动态
17×65 FP32→BF16 B，0.7442µs 对 0.4404µs；小尾部的检查/派发开销仍存在。
缓存/packing 选择与叶指令调度都可能影响结果，不能仅凭整体耗时将差距
全部归为算法，或据平均值宣称完成逐 case 的无退化验收。
该最大预打包 shape 的 plain dot 为 0.9914×、bias+ReLU dot 为 1.0588×，
而 library 分别为 1.2095×、1.1900×：此 case 的额外差距主要出现在缓存
wrapper/切片调度，不能据 library 数字直接判定底层 MMA 指令整体慢 21%。

矩阵正确性套件分批覆盖原矩阵/edge/layout/dispatch 44、空 shape 1、
运行时 dense panel 1、GEMV 数学 epilogue 8、融合策略 10。
额外动态 packing-bypass 数值测试 1 项覆盖昂贵的 A/B `T.func` tail、行/列
GEMV、真实 packing 和显式 K 分块，合计 65 个独立矩阵 case 分批通过。
最后一轮重新检查 GEMV/基本布局 12 项，以及 dense panel 与强制/禁止/自动
LayerNorm tail 融合；另外布局/转置/架构 36 项、优化 pass 22 项通过。
C++ ExecutionSession 3 项、MatmulTiler 16 项通过。
ARM 离线检查共 156 个矩阵对象（132 基础/融合/packing/normalization/SiLU
对象 + 24 exp2/log/rcp/rsqrt 对象），覆盖 FP16/BF16/FP32/FP64/S8/U8×S8，
四种 SVE/SME VL/SVL 编译配置；另重跑 88 个转置对象。编译与 SME no-call
gate 全部通过。这些不是 ARM 实机数值正确性或性能测试。
最后一轮输入融合政策变更后的日志为
`build-dsl-transpose/{pytest-matmul-policy,pytest-dynamic-fusion-final,matmul-cross-final-policy}.log`；
GEMV 完整组的 17 项数值/政策回归为 `pytest-gemv-groups-final.log`。
source-only checks：`git diff --check` 与 Python compileall 通过。
当前动态 kernel 会生成多个 ISA/family/alias alternatives，冷编译可达
数分钟；这仍是后续编译时长优化点，并未以剔除冷编译来声称它已经解决。
这一阶段无需新增 MLIR 或其他编译框架依赖。

### 集中操作语义与目标成本模型

操作契约统一放在 `_compiler/operations`，保留字符串 opcode 和现有
Scalar/Tile/Vector 类型；traits 与类型正交，不为每条操作引入类层级。
目录集中维护 effects、规约种类、普通 intrinsic 参数/arity、常量折叠、
CSE 合法性和 transpose/vector lowering 边界。未知操作默认 opaque，
不能通过“未出现在排除列表”意外获得优化合法性。CPU implementation
descriptor 单独维护 scalar/vector spelling、lane fallback 和 wordwise math
能力；这些能力不能由 cost 或昂贵程度推断。

`_compiler/cost` 定义 profile/rule/query 和融合 DAG 分析，CPU 的 ISA/处理器
覆盖位于 `backends/cpu/cost`。覆盖顺序为 default → CPU → ISA → processor
→ 显式用户 profile；规则按 priority、selector 数匹配，歧义报错。
当前默认 compute 估计继承 1/4/16 的启发式基线；区分 reciprocal throughput
和 dependency latency，保留 confidence/provenance。x86、SVE、SME 和 SPR
表只是未校准估计，SME 当前继承 SVE，不宣称 ARM 实机周期数据。
query 可区分 dtype、源 dtype、accuracy、scalar/vector、物理 word 数、mask
和访问方式。未知操作非零估价；融合后的 load_convert 只收一次 conversion。

region analysis 在 helper 内联/CSE 后按执行 DAG 累计，而不是检查是否出现
exp/log。共享节点只计一次，已物化/规约/dot 是边界；scalar 条件按实际分支
或最坏分支估计，vector select 两臂都计。输出包含 compute/memory work、
critical path 和 postorder live-vector hint；后者只是调度预算，不是完整物理
寄存器分配或端口/缓存模拟。CPU chunk 同时使用累计成本与 register budget，
显式 Vector tag 不变，legalization 后重新估算物理 word group。

重算预算与 temporary traffic 等策略集中在 profile policies，不再保留
散落的 expensive 列表。matmul 重算阈值随整个 producer DAG 成本连续变化，
仍保留 B packing 外置、A N-panel 重用、动态 GEMV 绕过 packing 的条件。
目前 raw 访问的重用次数是粗粒度 output-span proxy，packing 使用 panel
次数；精确到每个 kernel-family loop 的缓存/端口模型是后续校准工作。
GEMV scalar/vector epilogue 由 planning pass 对比成本，emitter 只读取结果；
切片后的临时 IR 继承原 store 的已选计划，不在 emitter 重新运行分析。

每次编译开始冻结 operation/implementation/profile，以及全部 pass 阶段；
ContextVar 保证 frontend、IR 构造和 pass 读取同一语义快照。快照带入生成
source identity，registry 指纹带入 T.jit 装饰器 key。显式跨编译 target 不
探测本机；只有 Native 允许本机探测，自定义 flags/toolchain 应明确指定
`cpu_cost_target`。模板/源码批量捕获机制不变，编译中途注册仅影响后续编译。

整个 `python/vecops/language` 使用项目 Ruff formatter（两空格、120 列）；
`dev` extra 固定 Ruff 0.16.10，避免格式随工具升级漂移。不增加 MLIR 依赖。

#### 本阶段验证（2026-10-03）

本机 conda `af3`：cost/architecture/matmul policy 33 项、optimization pass
22 项、GEMV/dynamic fusion/transpose native 36 项全部通过，合计 91 项
相关测试。新测试覆盖累计 DAG 成本、共享节点/边界、精度与 dtype 覆盖、
未知操作、快照/缓存失效和动态转置的不可变 shape 捕获。全量 Python
套件因逐项冷编译耗时中断，不将其计为全量通过。

ARM 仅离线检查：矩阵 156/156、转置 88/88 object 与 SME gate 通过；
显式 SVE/SME target 选择自身 profile，不继承本机 x86 成本。使用 Clang
19.1.3 与 `build-dsl-cross/sysroot/usr/aarch64-linux-gnu`；日志为
`build-dsl-transpose/{matmul,transpose}-cross-cost-final.log`，没有 ARM
实机正确性/性能结论。

保留用户当前 x86 限频设置，paired 正确性/性能抽查：64 行、宽度
512/4096 的 LayerNorm DSL/C++ 耗时比 0.664/0.901，Softmax 为
0.725/0.904。bf16、256³ 和 1×257×129、Const/Dynamic、plain/SiLU
组合的 8 个 matmul case 几何平均比 0.963；Dynamic plain GEMV 仍为
1.215，保留先前该路径的性能差距，不宣称所有 case 与 C++ 持平，也
不将这次抽查等同于严格同轮修改前后无退化证明。日志分别为
`build-dsl-transpose/norm-cost-smoke.log`、
`build-dsl-transpose/matmul-cost-smoke-final.log`，matmul CSV 位于
`build-dsl-matmul/cost-model-smoke-final.csv`。

复现完整 catalog 与独立 packing（保留上述隔离和限频条件）：

```bash
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul.py \
  --kind bf16 --level matmul --pipeline plain --max-flops 0 \
  --rounds 9 --jobs 4 --output build-dsl-matmul/catalog.csv
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul_pack.py \
  --shape 17x65 64x256 513x1000 1025x1027 --kind bf16 convert \
  --rounds 9 --jobs 6 --output build-dsl-matmul/pack.csv
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul_cross.py \
  --clang <clang19>/bin/clang++ --objdump <clang19>/bin/llvm-objdump \
  --sysroot build-dsl-cross/sysroot/usr/aarch64-linux-gnu --jobs 8
```

### 前一阶段转置验证记录

ARM 最终 **88/88 object + SME gate** 通过：四个目标 SVE VLA、512-bit
VLS、SME VLA、固定 512-bit SVL；17 个 dtype 三元组的完整 source
instantiation、动态/aligned 窄转换、自动/显式区域与双行/列流水。
另对公开 API 迁移后的 C++ SME Matmul umbrella 做完整语法检查通过。
仍没有 ARM 链接运行或性能数据。C++ x86 原有转置 8 项、公开 AMX
状态/搬运 2 项、访存转换 65 项，以及当前源码的 focused 转换 29 项
分别通过（focused 与原测试部分重叠，未累加为独立测试数量）。
Python 新转置/架构边界 29 项、补充注册/NumPy/特殊值等测试继续通过；
AMX scratch materialization 与写入前捕获 load 的保值回归也通过。
现有 DSL/优化 pass 回归 **63/63** 通过，runtime 回归 **23/23** 通过；
相关日志位于 `build-dsl-transpose/pytest-regression.log` 和
`build-dsl-transpose/pytest-runtime.log`。这些 Python 测试均使用本机 conda
`af3`，没有增加 MLIR 或其他编译框架依赖。

### 矩阵性能审计：02a2edf 之后

先以 `02a2edf` 提交此前全部修改，再针对生成代码和当前公共 C++ Matmul
做三路同轮对照：旧 DSL 源码、新 DSL 源码、C++。旧源通过 benchmark 的
`--source-manifest` 保存；`--baseline-manifest` 使用相同 schema、真实输入、
编译器和 specialization values 重放，不更改 C++ baseline 的算法。
动态 shape 复用同一个 Operator；所有构建在计时之前完成，每个 case 都
先检查 NumPy/C++/旧 DSL 数值，再随机交错 native repeat。仍保持 CPU 2、
SMT sibling 58 隔离，以及用户的 800–1800 MHz 限频。

确认的生成/调度问题及修改：

1. **别名保护破坏 K 连续性。** B 的原始 `[K,N]` 视图可能沿 K 连续，
   旧快照却固定 row-major。每个 dot load 保留“连续原始加载 / gather
   快照加载”两个完整表达式。快照现在保留一个已证明 unit-stride 的轴，
   重新密集化，去除 padding、重叠、负 stride。兼容的无 mask 加载先选
   地址/stride，再执行一个 load/convert。真正别名时仍复制数据，不能
   将运行时 non-alias 当作未经证明的前提。
2. **缓存边界先于 register family 决策。** 通用 32 列边界让动态 small-M
   的 1×3 AMX 内核频繁遇到不完整的 48 列组。自动路径先选 family，再
   按 16×48 / 48×16 / 32×32 等实际跨度分块。
3. **无 M 复用却再次打包 B。** 证明只有一组 M instruction tile 时，
   自动 B packing 省掉整面板临时 buffer，直接使用叶内核必需的 staging。
   显式 packing、已有 packed 输入不被覆盖。
4. **小动态问题总走缓存 wrapper。** 自动 AMX 路径按运行时 footprint
   绕过缓存规划与面板循环；乘积比较以预算除法避免整数溢出。Const
   cutoff 仍为 512 KiB；Dynamic 为 128 KiB，因为 256³ 的动态 unblocked
   实测反而比保留 packing 更慢。阈值放在 cost policies 中，不能将
   Const 的 C++ 常量传播收益直接假定给 Dynamic。
5. **循环内重复边界处理。** raw K-contiguous B 的完整 N 组判断提到 K
   循环外；完整 K 与一次补零尾部独立。已验证非负、有界的 packing/K
   迭代改用 `extent-index >= width`，不在每一步重复 signed-128 加法。
6. **GEMV 分组不是越宽越好。** x86 短且编译期已知 K 默认一个 bank；
   长 K（默认 >=512）和未知 K 保留两个 bank。数学 epilogue 仍组向量
   求值；简单 epilogue 的剩余输出采用双列组 + 单列组，不为两个剩余
   输出分别重载 A。已知窄 N 收窄真实列数，并保持至少四个独立累加器。
   显式 `cpu_gemv_k_unroll` 覆盖长 K 策略。
7. **显式 cache phase 的代码膨胀。** 动态 BM/BN/BK 路径将 family
   alignment 表示为运行时 quanta，共用一个 packing wrapper；缩短的
   panel 在叶内选择 family。否则带 transform 的面板生产会被复制五次，
   GCC 冷编译时间显著增加。此修改不把整个 dot 换成 opaque C++ 算子。

这些修改没有新增 vector 数组，也没有直接调用 `ops::Matmul`。AMX 的
staging 数组是矩阵数据内存，不是 Vec 寄存器容器；寄存器仍为单独变量。
必要的资源管理闭包继续带 CoreDefs 中的强制内联宏。

复现完整三路对照（baseline manifest 中的旧文件需要保留）：

```bash
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul.py \
  --max-flops 0 --kind bf16 --mode Const Dynamic --level matmul \
  --pipeline plain --rounds 17 --jobs 8 \
  --baseline-manifest build-dsl-matmul/baseline-catalog-02a2edf.json \
  --output build-dsl-matmul/perf-catalog-validated-paired.csv
```

`--max-flops 0` 很重要；默认过滤会跳过 64×1024×4096 和大 prefill，
不能把过滤后的 focused 结果称作完整 catalog。基准额外支持显式 `--cc`
和 `--cxx`；GCC 性能对照与 Clang 数值补充检查分开记录。

#### 当前源码的验证与性能结果

本机 GCC 13、conda `af3`，完整 BF16 catalog 共 86 个 shape ×
Const/Dynamic = **172 个 case**，17 轮同轮三路计时，数值检查全部通过。
以下比值均为耗时比，越小越好；不是跨轮选择的最佳结果。

| Shape / metadata | 新 DSL / 旧 DSL | 新 DSL / C++ |
|---|---:|---:|
| 4×4096×4096 / Const | 0.670 | 0.931 |
| 4×4096×4096 / Dynamic | 0.608 | 0.918 |
| 1×1152×896 / Const | 0.776 | 0.984 |
| 16×48×128 / Dynamic | 0.755 | 0.958 |
| 172 case 几何均值 | 0.952 | 0.915 |

旧 DSL / C++ 超过 1.1 的 case 从 27 个降为 10 个，超过 1.2 从
19 个降为 4 个。不过相对旧 DSL，仍有 9 个 case 超过 1.1、最差
1.181；相对 C++ 最差为 Const 128×4096×4096 的 1.302。
其中几组微小 Dynamic case 的绝对差值只有几十 ns，但 Dynamic
256³ 和 3136×64×576 等并非只能归因于微小计时噪声，仍需继续优化。
大型 case 在不同轮次有明显波动，不能混用不同源码阶段/轮次的比值。

当前源码的独立验证：

- native codegen、真实 A/C 和 B/C 别名、部分常量 AMX family、已有
  edges 测试 **16/16** 通过；包含零 K、packed、融合、类型转换等边界。
- codegen 快速检查、cost、matmul policy 和 architecture 测试
  **38/38** 通过（与上述 native cohort 有 5 项重叠，不累加成独立总数）。
- ARM SVE/SME 的 VLA / 固定 VL/SVL **156/156 object + SME gate**
  通过；仅跨编译，不包含离线 ARM 机器上的链接运行和性能测试。
- 另对当前源码完成 **32/32 Const 融合 benchmark case**：四种输入
  dtype 组合，GEMV 尾部和 AMX 1×3 shape，四种前后处理。每个 case
  先比较 NumPy 与 C++，9 轮 native 计时中最差 DSL / C++ 为 1.059。
  此补充没有旧 DSL manifest，不用来证明相对旧 DSL 的融合性能不退化。
- `vecops.language`、此 benchmark 和新测试的 Ruff 格式检查，以及
  未定义名称/语法检查通过；没有宣称整个 Python 测试集或全量 lint 通过。

报告与日志保留在 `build-dsl-matmul/`：
`perf-catalog-validated-paired.csv`、`pytest-native-validated.log`、
`pytest-policy-validated.log`、`matmul-cross-perf-validated.log`、
`perf-fusion-const-validated.csv`。Const 融合补充复现命令：

```bash
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul.py \
  --shape 1x257x129 1x257x257 16x48x128 16x48x256 \
  --kind bf16 convert i8 mixed \
  --pipeline accumulate bias_relu silu affine_input \
  --mode Const --level matmul --rounds 9 --jobs 6 \
  --output build-dsl-matmul/perf-fusion-const-validated.csv
```

还不能据总体平均值宣称所有 shape 不退化。大型宽 prefill 的剩余差距
需要隔离 cache wrapper / packing / traversal 与 leaf codegen 才能归因；
动态 GEMV 仍保留 metadata/别名检查和调度。native repeat 基准不包含
Python 调用，因此这些差距不是 Python 调用耗时。后续应单独比较
prepacked dot 与 library、full/tail instruction 路径及配置区间，逐步验证
traversal 和缓存面板选择，不能仅凭整体耗时归因于某一条 MMA 指令。

**冷编译限制：** BF16、float32→BF16、int8、uint8×int8 与
accumulate / bias-ReLU / SiLU / affine-input 的大组合补充验证各包含
112 case。部分 Dynamic transform 对象的 GCC / Clang 优化阶段分别
超过 50 / 30 分钟，单进程内存达数 GiB，因此停止了这两组任务，
保留构建产物和日志，不能将它们记为通过。GCC 那组使用较早的调度
源码，Clang 那组使用当前源码；两者也不合并成当前性能报告。
共享显式 cache packing wrapper 减少了一处复制，但没有解决所有
动态 family/full-tail/producer 分支造成的编译膨胀。后续应在保持
强制内联和精简热路径的前提下，进一步提取/共用生成区域，并为
动态融合冷编译时间建立独立回归预算。

### 生成结果深审与第二轮优化（基线 4b3d738）

这轮分开检查预打包 `T.dot`、预打包 library 和 raw library，避免将
packing/cache wrapper 的差距全部归因于 AMX 指令调度。环境仍为本机
Xeon w9-3495X、GCC 13、conda `af3`；保持用户的 800–1800 MHz 频率限制，
计时锁定 CPU 2，编译排除该 CPU 及 SMT sibling 58。只比较同轮配对
耗时；不同阶段的绝对耗时有显著波动，不能拼接成旧/新的加速比。

#### 源码、汇编与采样证据

对 Const 128×4096×4096 的 native repeat 做 `perf record`，再沿采样
热符号定位 K 循环，而不是挑共享对象中最短的任意 AMX 分支。实际热
路径的 C++ family 是 balanced 2×2，不能未经验证地假定它使用 1×3。
该 case 的 raw A stride 在 C++ 汇编中是 8192 字节，不能将差距解释为
“C++ 一定打包了 A”。

旧 DSL cache-panel shape 变为普通 raw scalar，丢掉原 problem 的
instruction-aligned 属性，循环内部因此还保留 A staging 判断、B 边界
判断、packed offset 计算及重复 metadata 读取。当前完整路径对应的
循环为 4 次 tile load、4 次 tdpbf16ps、4 次指针推进和一个 K 回边。

| 采样对应的完整 2×2 K 循环 | 静态指令 | tile load | dot | 条件分支 | 栈引用 |
|---|---:|---:|---:|---:|---:|
| 当前 DSL，0x4000–0x4048 | 15 | 4 | 4 | 1 | 0 |
| C++，0x3300–0x3370 | 25 | 4 | 4 | 1 | 0 |

这些是静态区域计数，不是各指令的延迟测量。采样存在 skid，某条
`tileloadd` 上的样本比例不能直接解释为它自身的耗时。也不能由 15/25
直接推导端到端加速：packing、缓存复用和实际内存流量仍占相当比例。
首次采样含冷编译，进程总样本比例不作为 kernel 独占耗时占比。

可复查的对象分别是
`build-dsl-matmul/cache/2d7b3e0ddb30cb7e/fcf06c5e2d7bab16/kernel.so`
和 `.../06ee6e668466fc7a/kernel.so`；修改前 DSL 是
`.../8227016173e49d26/kernel.so`。这些构建产物需保留；对象地址不是
跨编译版本稳定的 API。新增 `benchmarks/dsl_matmul_inspect.py` 支持
数值检查后 profile、打印本任务加载的 artifact 路径，以及静态 AMX
回边审计；`asm --limit` 返回的短区域未必是采样热路径。

```bash
perf record -F 999 -o build-dsl-matmul/profile.data -- \
  conda run --no-capture-output -n af3 python benchmarks/dsl_matmul_inspect.py \
    profile --shape 128x4096x4096 --mode Const --kernel dsl --iterations 1000
python benchmarks/dsl_matmul_inspect.py asm path/to/kernel.so --limit 5
perf annotate --stdio -i build-dsl-matmul/profile.data --dsos kernel.so
```

#### 本轮实现

1. **保留 full M/N/K 证明。** 按同一 register family 对齐的 cache panel
   与其 remainder 继续满足原尺寸的整除关系，传递给叶内核。显式动态
   family wrapper 不擅自传递不匹配跨度的证明；真正 partial 路径不消失。
2. **循环不变量提取。** AMX packed A/B 与已证明完整空间组的 native
   K-contiguous A 建立单独的 K cursor。地址、alias 选择和 stride 在
   K 外准备；inactive packed tile 从有效 panel-zero 地址开始，不能
   先形成 allocation 外指针。尾部仍使用合法的补零/staging 路径。
3. **packed ABI 不混淆。** 只有 AMX/target-auto AMX packed A、B 均有
   padded K 时才直接走完整指令跨度；dense packed 不保证 padding，必须
   保留最后一个 K tail。新增 K=33 的 dense AB native 回归。
4. **高瘦矩阵遍历。** x86 policy 在 N<=128 且 M>=4N 时使用 M-major，
   尽量让小 B 留驻、减少大 A 的跨 N 组重访。Dynamic 只选择坐标顺序，
   不复制整份 full/tail/fused kernel。其他 CPU policy 默认关闭。
   直接 dot 保留整 tile 策略；已知尺寸的 library 保留 cache-driver
   N-major。复测确认整问题高瘦不代表缩短后的 MC/NC panel 高瘦，将
   原问题的静态 order 强加给 cache leaf 缺少 panel-level 盈利依据。
   但改回 N-major 后 1024×64×4096 仍相对旧版慢约 12%，不能据此
   认定 traversal 是该退化的根因。Dynamic 仍保留单独测量的坐标
   选择路径；panel-local cost 还不是自动最优证明。
5. **native B packing 复用门槛。** 已知原 M<128、完整空间 register group、
   未转换/变换且 K 连续的 B 在 cache-panel 路径不自动增加面板副本。显式 packing、已有 packed、转换和变换
   不受该门槛覆盖；未知 M 仍保留原保守策略。阈值集中在 cost policy，
   不散落在 emitter 的 shape 常数判断里。
   初次只按 M 阈值的全量对照在 33×35×8193 / 35×53×257 上分别
   相对旧版退化 21.7% / 12.9%，因此加入完整空间组条件；不能将
   64×1024×4096 的收益推广到 ragged tile 的反复 masked staging。
   Small unblocked 多 M 组仍保留原 resident packed buffer：64×256×256
   的实验说明，去掉这类小副本会增加重复 transpose 计算。已有单 M
   组 bypass 不变；不能将 cache-panel 的收益推广到全部 footprint。
6. **零 K 不形成无效 row cursor。** 已知 K=0 不创建 cursor；原 K
   未知时，native A cursor 在 K>0 才形成 row-offset 地址，否则保留
   storage base。这个判断在 K 外，不污染热循环。新增动态 K 的
   0/33/64、非连续 A stride 的 native 回归；空 NumPy tensor 的
   dense-stride ABI 归一化与 DSL 的独立 stride 绑定保持一致。

未改变 MMA register family，没有转发到 opaque `ops::Matmul`，没有
Vec 数组。资源管理闭包仍使用强制内联宏。宽 prefill 的差距仍需要在
packing producer、cache panel 大小和遍历之间继续隔离，不能宣布全部
达到 C++ 性能。Dynamic native B 的更多运行时 packing 派发尚未加入，
避免为一个阈值再克隆大段编译区域。

最终版本的完整三路性能对照命令（旧 manifest 来自独立 4b3d738
worktree；生成时校验 package 来源与既有 focused source 的字节一致性）：

```bash
conda run --no-capture-output -n af3 python benchmarks/dsl_matmul.py \
  --max-flops 0 --kind bf16 --mode Const Dynamic --level matmul \
  --pipeline plain --rounds 21 --jobs 6 \
  --baseline-manifest build-dsl-matmul/baseline-catalog-4b3d738.json \
  --source-manifest build-dsl-matmul/audit-final-catalog-sources.json \
  --output build-dsl-matmul/audit-final-catalog-paired.csv
```

#### 最终验证与保留的差距

完整 BF16 raw catalog **172/172** 数值通过，21 轮三路同轮计时：

| Shape / metadata | 当前 / 4b3d738 | 当前 / C++ |
|---|---:|---:|
| 64×1024×4096 / Const | 0.6701 | 0.8740 |
| 256³ / Dynamic | 0.9125 | 0.9393 |
| 3136×64×576 / Dynamic | 0.9525 | 0.9709 |
| 128×4096×4096 / Const | 0.9907 | 1.2497 |
| 128×4096×4096 / Dynamic | 0.9572 | 1.2725 |
| 172 case 几何均值 | 0.9938 | 0.9113 |

相对 C++ 超过 1.2 的 case 从同轮旧版的 5 个变为 3 个；超过 1.1
从 12 个变为 10 个。相对旧版仍有一个超过 1.1：Const
1024×64×4096 为 **1.1196**，另 Const 3136×64×576 为 1.0908。
因此本轮是特定 shape 和生成路径的改进，不是全部 shape 无退化。
同一源码的复测也出现该静态高瘦回归；不能将它全部解释为几十 ns
噪声，不能宣称切换 cache order 已修复。宽 prefill 的 packing/cache
策略和这两个静态 case 的地址/访存调度仍需后续定位。

最终预打包隔离基准 **40/40** 数值通过（17 轮）：几何均值
当前 / C++ 为 0.9429，最差 1.1924。Const 128×4096×4096 的 dot
packed B / AB 为 1.0045 / 1.0089，但同 shape 的 library 为
1.1518 / 1.1759；说明即使排除 online packing，cache wrapper /
局部遍历仍有约 15%–18% 差距，不能仅归咎于 packing producer。
Const 3136×64×576 的直接 dot packed B / AB 为 1.0472 / 1.0251；
旧轮次为 1.3683 / 1.2509，但旧/新来自不同轮次，不能据这两组
比值计算同轮旧/新的加速比。Dynamic ragged dot packed B 仍有
约 19% 差距，全部预打包路径也没有宣称无差距。

其他验证：

- native codegen/别名/packed/融合/零 K 边界 cohort **20/20** 通过；
  追加动态 K 与非连续 A stride cursor 回归 **1/1** 通过。
- 最新 codegen 快速检查、cost、matmul policy、architecture **41/41**
  通过；与 native cohort 有重叠，不累加为独立测试数量。
- optimization pass 测试 **22/22** 通过。
- Const 融合补充 **32/32** 数值通过（四类输入 dtype，accumulate /
  bias-ReLU / SiLU / affine-input），9 轮最差当前 / C++ 为 1.0674。
  这些 shape 的最终 cache-order/crossover 修订不改变其生成内核路径。
- ARM **156/156 object + SME disassembly gate** 通过；最终 generator
  重生成的 156 份源码逐字节匹配已验证对象的 source，包含 SVE/SME
  VLA / 固定 VL/SVL。没有离线 ARM 机器上的运行或性能测试。
- Ruff format（87 个文件）、未定义名称/语法检查、`git diff --check`
  通过；没有宣称整个 Python 测试集或全量 lint 通过。

当前报告：`build-dsl-matmul/audit-final-catalog-paired.csv`、
`build-dsl-matmul-packed/audit-final.csv`。日志：
`pytest-final-native.log`、`pytest-verified-zero-cursors.log`、
`pytest-audit-final-fast.log`、`pytest-final-passes.log`、
`matmul-cross-final-cursors.log`、`final-fusion-const.log`。
中间实验的 `final-*` CSV 明确属于各自源码阶段，不与最终报告混用。

### 2026-10-03：解除限频后的大矩阵缓存驻留审计

先将上一轮变更提交为 `75795c5`（`perf(dsl): hoist AMX loop
invariants and refine packing schedules`），再重新测量。用户已解除
CPU 限频，本机 w9-3495X 的 `scaling_max_freq` 为 4.8 GHz、governor
仍为 powersave；本轮没有改动任何频率/电源设置。下述结果不与上一节
限频时的绝对耗时混用。

#### 证据与定位

- 原始 metadata / BF16 / plain / library 基线，25 轮：
  Const / Dynamic 的 128×4096×4096 为 C++ 的 1.2110 / 1.1674，
  128×11008×4096 为 1.1620 / 1.1191。问题依然存在，并非解除限频
  后自动消失。
- 排除在线 packing 的原始隔离测试，32 case：Const 的
  128×4096×4096，dot packed B / AB 为 1.0399 / 0.9988，library
  则为 1.0937 / 1.1581。叶内核已经接近；不能只优化 packing producer，
  也不能把全部差距归咎于 AMX 指令数。
- 对相同生成物重新采样：DSL 热 K 区域 `4000–4048` 仍是 15 条
  指令、4 tile loads、4 dots、1 个条件跳转、0 栈引用。C++ 实际
  采样到的是 balanced 2×2 family，其完整热 K 区域为 25 条指令；
  A 是原始 stride=8192B 的直接加载，不是预打包 A。更短的控制/地址
  序列不等于更好的操作数驻留。
- C++ 的 `execute_whole_b_panel` 对 M=128 使用 64 列 B panel；
  DSL 原来的目标 cache 公式给出 MC=64、NC=128、KC=4096。
  原公式只把 packed B 计入预算，未给生产该 panel 的源 B footprint
  留余量。128 列 packed B 已是 1 MiB，源 panel 也是 1 MiB，
  之后还要访问 A / C；这形成私有 2 MiB L2 的驻留压力。
  该容量解释是模型，不能把采样落在 tileloadd 上直接解释成该指令
  自身延迟；panel 扫描是支持该模型的独立性能证据。
- 另采样 `l2_rqsts.demand_data_rd_miss`（固定 period=1000，
  前后各 1500 次 native 重复），仅统计选定 DSL `.so` 的样本，
  排除 Python / 编译器 / C++ 校验调用。估计需求数据读 L2 miss
  从 13,971,000 降为 11,188,000，约减少 19.9%。无 lost samples。
  这是 DSO 归属采样估计，不是所有 tile 内存事务的精确计数；
  进程总事件数还包含启动/编译，不能直接用于比较两个内核。
- `dsl_matmul_schedule.py` 在同一轮交错比较在线 packing / 各种
  BM、BN、BK / raw dot / whole packed dot / 全尺寸 panel，记录
  每种生成源码。128×4096×4096：默认 4.4746 ms，BM=64、BN=64、
  BK=4096 为 3.6840 ms，C++ 3.7224 ms；BN=256 明显变慢。
  128×11008×4096 的对应三者为 12.5481 / 10.2479 / 10.3064 ms。
  把整个 B 先打包再 dot 并不好：分别是 C++ 的 1.666 / 2.079，
  因而未采用 whole-operand packing 作为通用修复。

#### 实现与适用边界

在集中 x86 cost/policy profile 增加：

- `amx_cache_b_panel_divisor=4`：自动 NC 同时受目标运行时 L2/4
  限制，按当前 family 的 N quantum 向下对齐，至少一个 quantum。
  不是把本机的 NC=64 写死在架构无关层。
- `amx_cache_b_panel_max_m=256`：只约束较浅的 M 复用。
  常量 M 在生成时折叠，动态 M 在 panel 循环外选择 divisor。
  零 divisor 关闭上限；零 max-M 表示不加 M gate。

原默认 profile 都为 0，SVE/SME/vector cache 路径传 divisor=0；
显式 BN 覆盖自动预算。公开 DSL 签名、packed ABI、C++ Matmul
实现和 AMX 指令 family 均未改变。没有新增向量数组或闭包。

不能无条件缩小 panel：初版统一限制使 512×512×4096 从
1.5786 ms 变为 1.8310 ms，1024×1024×4096 也回退。
扩大 M 复用可摊销更宽的 B panel。加入 M gate 后的同轮复测为：

| Shape | 新策略 / 原预算策略 | 新策略 / C++ |
|---|---:|---:|
| 256×4096×4096 | 0.9727 | 1.0065 |
| 512×512×4096 | 0.9972 | 1.0280 |
| 1024³ | 1.0021 | 1.1029 |
| 1024×1024×4096 | 0.9973 | 0.6384 |

这里的原预算策略是在同一个 generator 中关闭新 policy，不冒充
逐字节的历史源码。完整 catalog 的 before/after 则使用保存的
`75795c5` 源码与原 schema/compiler，是真正的三路同轮比较。

#### 最终性能与验证

最终带 M gate 的 BF16 plain / library catalog **172/172**
数值通过，25 轮，CPU2 固定亲和性、native repeat、同 buffer、随机
交错新 DSL / C++ / 保存的旧 DSL；编译不计时。各 benchmark 的 CPU2
lease 串行化计时，编译排除 CPU2 及其 SMT sibling。

| Shape / metadata | 新 DSL / 75795c5 | 新 DSL / C++ |
|---|---:|---:|
| 128×4096×4096 / Const | 0.7981 | 0.9950 |
| 128×4096×4096 / Dynamic | 0.8322 | 0.9850 |
| 128×11008×4096 / Const | 0.8278 | 0.9880 |
| 128×11008×4096 / Dynamic | 0.8478 | 0.9586 |
| 172 case 几何均值 | 0.9923 | 0.9089 |

本轮没有相对历史源码超过 5% 的 case 回退；这是这组基准的观察，
不是所有输入/机器/程序无回退的保证。几何均值按 case 等权，也不是
AF3 应用整体加速比。绝对时间会随动态频率、package 状态变化，
应使用同轮比值。

仍保留的差距：1024³ 为 C++ 的约 1.103；对它进行更大 MC/NC 的
扫描未获得收益，直接 raw/whole-pack dot 也明显更慢，未合入这些
尝试。最终 Const prepacked 8 case 数值通过：128×4096×4096
library packed B / AB 为 1.098 / 1.026；128×11008×4096 为
1.053 / 1.032。预打包 B-only 仍需进一步审计局部访存顺序；不能
宣称所有大矩阵或 packed 路径均已追平 C++。

其他验证：

- codegen / cost / architecture / matmul policy 快速 cohort **45/45**
  通过（另 2 个 native codegen case 不在此 cohort）。
- native 边界 cohort **11/11** 通过，包括动态 K=0/33/64 与
  非连续 A stride、dense-packed 非对齐 K、别名与融合/尾部路径。
  最终分为 edges 9 case 与 codegen 2 case 复测；日志分别为
  `unrestricted-guarded-tests-native.log`（edges 的 9 项通过）与
  `unrestricted-guarded-tests-cursors.log`（2 项通过）。前一批读取
  codegen 测试时恰逢该文件行号变动，inspect 抓取了旧行号的源片段；
  两项在停止修改源文件后重新通过，未修改 frontend 来掩盖该错误。
- Const 大 prefill / ragged 融合补充 **40/40** 数值通过：bf16、
  float32→bf16、s8、u8×s8，accumulate / bias-ReLU / SiLU /
  affine-input；21 轮最差新 DSL / C++ 为 1.0795。
  M gate 修订不改变这些 M≤128 常量内核的可执行源码。
- ARM SVE/SME VLA / 固定 VL/SVL **156/156** 跨编译 + SME
  disassembly gate 通过。最终 generator 再生成的 156 份源码与
  这些对象对应的源码相同，仅 compiler-environment 指纹注释变化；
  未进行离线 ARM 主机上的执行或性能测试。
- Ruff format、F821/F822/F823/E9 检查与 `git diff --check` 通过；
  没有宣称全部 Python 测试或全量 lint 通过。

可复现报告位于 `build-dsl-matmul/`：
`unrestricted-guarded-catalog-paired.csv`（最终三路）、
`unrestricted-guarded-catalog-sources.json`、
`unrestricted-schedule-sweep.csv`（BN 因果扫描）、
`unrestricted-guarded-large-extra.csv`（M gate）、
`unrestricted-square-sweep.csv`（未采用的方阵调参）、
`unrestricted-fusion-large.csv`。预打包报告是
`build-dsl-matmul-packed/unrestricted-guarded.csv`。
L2 事件采样文件为 `unrestricted-l2-before.data` /
`unrestricted-l2-after.data`；`dsl_matmul_inspect.py profile` 的
`--baseline-source` 可以在同 schema/buffer 下重放保存的生成源码。
中间 `unrestricted-candidate-*` / `unrestricted-final-catalog-*`
属于未加 M gate 的实验阶段，不作为最终报告。

## CPU grid 调度实现（2026-10-03）

### 接口与编译器边界

```python
@T.template
def rows(A: T.Tensor((M, N), dtype="float32"),
         C: T.Tensor((M, N), dtype="float32", access="output")):
    for row in T.grid(M, schedule="auto", chunk=1, parallel="auto"):
        T.store(C[row, :], T.softmax(T.load(A[row, :])))

with vecops.cpu_thread_limit(8):
    kernel(a, out)
```

`grid` 仍然表示逻辑 program 集合，不表示线程数，更不创建进程。
`auto/static` 使用连续、均衡的静态分配；`dynamic` 使用原子游标领取
`chunk` 个 program。chunk 为编译期正整数。program 内部不被调度器
拆分，尾波不能凭调度消失。`parallel="serial"` 保持字典序，
`"require"` 要求并行安全证明/运行时 guard 成立，不要求至少两个线程；
小任务和嵌套调用仍可以串行。

| 模块 | 职责 |
| --- | --- |
| `frontend/parser.py`、`ir/kernel.py` | 独立 Grid IR；ViewRegion 保存逻辑根坐标，不改变地址生成 |
| `backends/cpu/grid.py`、`grid_pass.py` | 注册的 ownership、cost、region planning；与 C++ 打印分离 |
| `backends/cpu/cpp/grid.py` | 已规划 grid、worker-private scratch、运行时 guard 和条件屏障的生成 |
| `execution/Grid.h` | 线程数选择、static/dynamic 分发、resident-team 与 logical-lane 接口 |
| `runtime/CallAbi.h`、`NativeThreadPool.cpp` | 可选 team ABI、同步调用、异常取消、预算和持久 worker |

调度代价来自集中的默认/处理器 cost profile，而不是在 printer 中新增
expensive 操作列表。按 native word 估计向量吞吐、规约生产者/累加工作，
标量 epilogue 每 program 只计一次，dot 计入 MAC 工作。选择 P 时比较
`ceil(Q/P) * program_cost + P * worker_overhead`，并受 Q、调用预算、
环境容量限制。依赖链启动成本不乘整个向量长度。该模型是启发式，
尚不包含精确带宽饱和、NUMA、所有内层循环迭代数或动态分支概率。

### 安全性、共享并行域和同步

ownership pass 要求写入沿 program 坐标形成可分离、不重叠的分区；
同一根 storage 的读必须限制在本 program 的分区中。普通 slice、
transpose 和整数索引保留根坐标；无法表示的视图变换、共享可变标量、
未知副作用或硬件裸指令导致串行回退。重复切割已裁剪的轴也保守丢弃
证明。临时分配、packing 和规约 materialization 为 worker 私有，
跨本 worker 的 programs 复用容量。

不能凭不同参数名推断 no-alias：运行时检查根张量布局无重叠、跨参数
span 不相交以及分区边界。转置/负 stride 输出可以通过布局检查。
guard 失败回到有序路径，require 模式报错。ownership 的 base/width
不能依赖 program 内存读取、规约或内层 loop 状态，以免 guard 在
region 外引用未定义变量/旧数据。

`cpu_grid_regions` 支持：

- `individual`：每个 grid 独立同步提交，作为基线/调试模式。
- `adjacent`：连续的安全 grid 共用一个 resident team。
- `template`：还可跨越短的纯标量声明，由 worker 0 执行并发布。
- `auto`：目前采用 template-safe 策略；不是无条件将整个内核并行化。

无法安全搬移的控制流/副作用结束 region。后续 grid 的调度元数据不能
依赖中间或更早 single-worker phase 新定义的标量，也不能提前读取内存
或计算规约结果。尤其不能把后续 grid 中读取前一个 grid 输出的任务数
提前到 region 入口。此类情况拆回独立 regions。

region 内按 RAW/WAR/WAW 根效果和运行时跨参数 alias 决定屏障：独立
阶段不插入屏障；确定发生的屏障清空已完成阶段的 pending effects；
条件屏障仍保留未必已完成的 effects。scalar phase 前同步、后发布。
同根但同 static owner 的依赖暂未消除屏障，以保留保守正确性。

### C++ ABI 和运行时

V1 minor 1 在 `VecopsThreadPoolV1` 尾部追加 size-gated `team_executor`。
resident team 提供稳定 dense worker id、实际 worker count 和可取消的
happens-before barrier。logical task 不能冒充 resident worker，否则
多个 logical tasks 共用一个物理 worker 时可能屏障死锁。旧 V1 和当前
Torch/OpenMP logical-only provider 保持有效，自动退回逐 grid 同步提交。
native provider 提供真正的 resident team。

生成的 grid 内核显式定义 `VECOPS_KERNEL_EXECUTION_CONTEXT`，adapter
才转发 `const VecopsExecutionContext*`；不通过宽泛的 requires 探测来
误匹配已有 WorkspaceContext 模板。KernelDef 的 `runtime_parallelism`
策略将执行预算从 DSL artifact/hot-call key 中移出，预算/亲和性变化
不会重编译相同 DSL。现有 source-backed kernel 默认仍按 Parallelism
编译期特化。仅为拿 executor 不需要 workspace plan/replay。

直接 Python/NumPy 调用获得稳定的 adaptive CPU provider；容量读取当前
调用线程的 affinity、物理核心、cgroup v2 quota 和显式预算。quota 最多
缓存一秒，affinity 不缓存。默认先使用不同物理核心；显式预算可以
使用 SMT，但不突破允许 CPU/quota。预算 API 是调用线程局部、可嵌套
和异常恢复的，不改变 Torch team 或旧固定 native pool。

native worker 有独立 mailbox，仅通知本次参与者，不等待整个池中闲置
worker 的确认。池按几何容量/CPU 集合复用，避免任务数变化创建 O(P²)
个休眠线程；一次只执行选中的前缀。barrier 短暂自旋后 atomic wait，
异常取消唤醒全部屏障、等待所有参与者结束，随后池仍可复用。提交时
caller 临时 affinity 的修改会恢复，旧 native active-scope API 继续有效。
嵌套 region 在 native 或报告 in_parallel_region 的 embedding 中串行。

目前没有 work-stealing deque、跨进程共享任务队列或自动 NUMA 放置。
多进程使用 spawn/forkserver、独立 CPU affinity/预算；不支持 fork
继承已启动的 native workers。dynamic chunk 是 uneven-work 的可选方式，
不作为所有任务的默认策略。本实现不增加 MLIR/LLVM 依赖，不生成 Vec
数组，生成的调度闭包全部带 `VECOPS_INLINE_LAMBDA`。

### 验证与性能

本机 Intel Xeon w9-3495X / GCC 13.3，conda `af3`；不改动 CPU
频率配置。最终验证记录：

- `test_dsl_grid.py` **26/26**：动态/零维/tail、static/dynamic、私有
  scratch、softmax、跨 phase RAW/WAR/WAW 和实际 alias、scalar phase、
  错误取消/恢复、三种 region 策略、转置/负 stride 输出、线程预算和
  affinity、预算变化不生成新 artifact、BF16/FP32 batched dot/packing、
  Torch logical/native 两种 provider，以及 metadata 不安全 hoist。
  shifted-alias 的有序语义补强后单项再次通过（使用会发生连锁读取的
  输入，不能靠一次整张量 snapshot 得到相同结果）。
- `ParallelTest-Scalar` **23/23**、`TaskPartitionTest-Scalar` **13/13**；
  `VecopsRuntimeTest` PASS，包括 legacy/static 与 runtime-parallelism
  artifact key 的区别；serial fast paths 的 process 浮点模式也有独立回归。
- language/pass/cost **48/48**；runtime Python 独立进程 **23/23**。
  最终 installed runtime 的 grid + 既有 DSL 快速 cohort **28/28**，
  包含 layernorm moments、inline reduction/alias 和批量编译 capture。
  本阶段另执行 norm/transpose/matmul 回归 **87/87**；这组完整回归在
  runtime-parallelism 缓存策略变更之前，之后以 runtime 与上述 cohort
  验证该策略，没有将其冒充又一次全量运行。
- 生成的 FP32/BF16、Const/Dynamic、SVE VLA/VLS、SME VLA/fixed-SVL
  **16/16** 跨编译；更新后的 native pool 也通过 AArch64 编译，C ABI
  header 通过 C11 语法检查。未连接或运行离线 ARM 主机。
- Ruff format（91 文件）、F821/F822/F823/E9 和 `git diff --check`
  通过；不是宣称全量 lint/所有项目测试通过。

可复现的本机命令：

```bash
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=8 conda run --no-capture-output -n af3 \
  python -m pip install -e . --no-build-isolation --no-deps
cmake --build cmake-build-debug \
  --target ParallelTest-Scalar TaskPartitionTest-Scalar VecopsRuntimeTest -j6
conda run --no-capture-output -n af3 pytest -q python/tests/test_dsl_grid.py
conda run --no-capture-output -n af3 python benchmarks/dsl_grid.py \
  --threads 1 2 4 8 --rows 9 37 256 \
  --operation softmax layernorm multigrid imbalance --repeat 80 --rounds 13 \
  --cpus 2 3 4 5 6 7 8 9 \
  --output build-dsl-grid/scheduling-budget-checked-final.json
```

CMake 配置使用 `/usr/bin/gcc`、`/usr/bin/g++`。ARM 跨编译脚本
`benchmarks/dsl_grid_cross.py` 指定 clang 19.1.3 和
`build-dsl-cross/sysroot/usr/aarch64-linux-gnu`；clang 路径为
`/opt/spack/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/llvm-19.1.3-2tjty3xfpnxpvl3l2ghtirmts4fyekim/bin/clang++`。

benchmark 使用同一 aligned buffers、native repeat、交错随机顺序的
13 轮中位数，编译不计时且编译 affinity 排除 CPU2–9；每个线程预算
下再次检查输出，包括 multigrid 的 B/C。**180/180** 配置通过。
下表是最终一次测量，8 为预算上限，不承诺每个 shape 都启动 8 个
worker；参考路径是显式 serial 的同一个 DSL，**不是 C++ 算子基线**。

| 256×4096，8-worker budget | Serial μs | Static μs | Dynamic μs |
| --- | ---: | ---: | ---: |
| softmax | 384.73 | 80.20 | 88.95 |
| layernorm moments | 221.85 | 53.50 | 61.81 |
| 三阶段 multigrid + scalar | 775.72 | 124.99 | 261.62 |
| 前 1/4 行 softmax，其余行 add | 263.03 | 77.56 | 63.40 |

softmax / layernorm 分别约 **4.80× / 4.15×**。multigrid 的
individual / adjacent / template 为 **144.64 / 131.68 / 123.95 μs**，
template 相比 individual 少约 14.3% 时间；auto 与 template 的小差别
属于同轮采样波动。uneven case dynamic 比 static 少约 **18.3%** 时间。
这不意味着 dynamic 普遍更快：均匀的 multigrid 中小 chunk 原子领取
明显更贵；小 shape 的启动/同步开销也可能超过并行收益。
9×4096 layernorm serial / auto 为 3.54 / 3.66 μs（约 +3.6%）。
启发式保留小任务串行，但运行时预算/guard 等入口仍有开销；没有
宣称所有 shape 无退化。绝对时间和比值会随 package/频率
状态波动，应使用同轮比较。

两个 spawn 进程各占独立 4 CPU（10–13、14–17），256×4096 softmax
均数值通过。记录为
`build-dsl-grid/multiprocess-complete-final.json`；验证的是独立进程的
CPU 隔离与调用能力，不是进程间 work stealing，也不据此宣称线性扩展。
验证日志保存在 `cmake-build-debug/grid-*-final.log`，完整计时结果保存在
`build-dsl-grid/scheduling-budget-checked-final.json`。

## 语言能力补全：typed control-flow state（2026-10-03）

补全顺序保持为以下五层。以下是第一层交付时的状态；后续第二至第五层
的实现与剩余限制见文末「后续原语交付」，不将历史记录当成当前状态：

1. **已实现**：typed tile/vector loop-carried state、动态分支结果合并、
   helper 的条件/循环内提前返回。
2. **待实现**：通用寄存器数据变换、任意 index 的 gather/scatter、
   predicate reductions。已有 view transpose/reshape 和规则 stride
   gather/scatter 不等于这些完整的值级原语。
3. **待实现**：tile-local 独立循环、SIMD 循环和 execution layout，
   与 task-level `T.grid` 保持不同层次。
4. **待实现**：通用 reducer、scan、跨 worker collective/atomic，
   必须明确数值顺序、同步域和共享存储契约。
5. **待实现**：软件流水和架构专属裸指令扩展；沿用 CPU-first 后端模块，
   不把 AMX/SME 的约束写进通用语言层。

### 语义和前端

`x = T.load(A); for ...: x = x + 1` 不再要求手工创建中间 buffer。
IR 使用 `TileState` / `DeclareState` / `UpdateState` / `state_ref`
描述逻辑值，不携带具体寄存器、Vec 数组或硬件 tile 数量。scalar
继续使用 typed mutable locals。state 的 dtype、predicate kind、rank、
logical extent 和显式 Vector tag 保持固定；静态不一致编译报错，
尚不能证明相等的动态 extent 在 update 前检查。tile 的 shape 和地址
表达式在值创建时捕获，不能随随后修改的标量悄悄变化。

`old = x; x = x + 1` 保持 old 的快照；tuple 同时赋值先捕获各 RHS，
因此 `x, y = y, x` 不会变成顺序覆盖。dynamic if 的两个分支均定义
的 scalar/tile/nested tuple 通过 typed result slots 合并；只在一个
路径定义的名字不导出。结果 shape 必须能在 if 前获得，不能将
branch-local 的未知大小直接 hoist。view rebinding 仍需显式 storage，
循环内才出现的新名字也不导出为循环后的结果。

`T.func` 的 acyclic 条件 return 使用 continuation normalization。
循环内 return 使用 `ExitRegion` / `Exit`，结果 slots 在 region 外声明，
typed tuple 逐项写入；C++ 降级为向外跳转的 lexical label，不生成闭包，
不跳入初始化作用域。value helper 的动态路径漏 return 会运行时报错，
不是返回未初始化值或默认零；void return 仅退出 helper，不退出 kernel。
递归深度、源文件定位、不可变 captures 的 cache identity 沿用已有规则。

前端模块分工：`frontend/parser.py` 驱动 AST；`state.py` 管 typed
状态和 branch joins；`helpers.py` 管参数绑定和 helper exit/result；
`control.py` 仅管 AST store-name 和 continuation 变换。它们不决定
CPU 存储布局。

### CPU lowering 与优化边界

注册的 `cpu-plan-state` pass 在 grid ownership 前执行，选择：

- 小固定 tile：末轴拆成 native-register 范围内的 pieces，
  高维 row 坐标显式枚举；每个 piece 独立 C++ Vec/Mask local。
  x86 使用 FixedTag；SVE 有捕获的 VL 时使用单 word ScalableTag，
  不把运行时 VL 误当成 VLA FixedTag 的合法宽度。未知 VL 时的固定片
  保持在 128-bit architectural minimum 内。placement 使用真实
  codegen ISA，而非用户可覆盖的 cost profile；widening 给每片预留
  lane headroom，相关 snapshots 共用 geometry，避免循环中的 scalar
  get/set 重组。末片保留 active count，不读取 tile 以外的内存。
- Scalable Vector：在预算内拆成多个 P=0 的 ScalableTag local，
  runtime lane count 只决定每片 active count；不依赖 SVE VLS，
  VLA sizeless 类型同样不进入数组/struct/tuple 存储。
- 超出预算、一般动态 tile、矩阵累加的内存路径：private scalar Buffer，
  update 使用独立 staging buffer 完成后 swap，保留同时更新语义。
  暂存容量按 invocation/worker 私有复用，不共享到其他调用。

`pass_options={"cpu_state_register_budget": n}` 支持 0–64，默认 12；
这是 **每个 state** 的 piece 上限，不是全内核 liveness 或精确 spill
预测。0 强制 scratch，便于验证两种降级语义。未实现全局物理寄存器
分配，也不保证大量同时存活的 states 都不 spill。

declare/update 和 state consumers 仍先生成 Vector IR 并运行已有
vector passes，发射器读取已准备的 placement/regions，不自行跑 pass。
同 extent/piece 的 consumer 和 reduction 直接遍历这些寄存器；不能
匹配的轴/广播/布局通过显式 get/set 做正确性 fallback，未来由第二层
的寄存器变换替代，不能称为已优化 shuffle。固定 piece 的完整访问
省略额外 first mask；动态/scalable active counts 与显式 mask 保留。

state reads 在中心 operation registry 中标为 read effect 和 state
reference。CSE/reuse 不能跨 update 复用含 mutable state read 的 DAG；
update 后清除发射阶段的物化缓存。worker-private state 可以在并行
grid 内做独立递推，共享 mutable state 则阻止并行 ownership 证明，
保守回退；require 模式报错。

`T.dot(..., acc=acc)` 可以跨循环正确累加；当前不声称 AMX/SME
矩阵寄存器能跨一般控制流持续驻留。保持正确性的内存路径与第五层
ISA 状态/流水整合是不同工作。本阶段不增加 MLIR/LLVM 依赖。

### 验证方式

```bash
conda run --no-capture-output -n af3 pytest -q \
  python/tests/test_dsl_state.py python/tests/test_dsl_architecture.py
conda run --no-capture-output -n af3 pytest -q \
  python/tests/test_dsl.py python/tests/test_dsl_passes.py \
  python/tests/test_dsl_grid.py python/tests/test_dsl_cost.py \
  python/tests/test_language.py
conda run --no-capture-output -n af3 python benchmarks/dsl_state.py \
  --width 7 17 37 64 --count 32 --repeat 50000 --rounds 21 \
  --output build-dsl-state/recurrence-final.json
```

`benchmarks/dsl_state_cross.py` 显式传入前文的 clang 19.1.3 路径和
AArch64 sysroot，覆盖 fixed/dynamic tiles、scalable vector、loop/branch
state 和 helper early exit，分别编译 SVE VLA/VLS、SME VLA/fixed SVL。
这不是 ARM 运行或性能验证；没有连接离线主机。

`benchmarks/dsl_state_reference.cpp` 的显式 C++ 对照同样将所有向量
piece 保留在一个 recurrence loop 中，避免用逐片完整循环的较差
调度夸大 DSL 的优势。计时共用 aligned buffers、native repeat、随机
交错的轮次，编译不计时，CPU affinity 在结束时恢复。对照的是小
tile state microbenchmark，不是 layernorm/matmul 全面性能结论；
一般动态 tile 的 scratch 仍可能显著更慢，不能据此宣称无条件零开销。

本机 GCC 13.3 / conda `af3` 验证记录：

- 最终 `test_dsl_state.py` + `test_dsl_architecture.py` **37/37**（状态
  29 项、既有架构约束 8 项）；覆盖 static/dynamic/zero/tail、快照和
  simultaneous tuple update、typed branch/tuple joins、predicate state、
  FP16/BF16/FP32/FP64 存储转换、Scalable Vector、while/reduction/
  break/continue、helper 的嵌套循环 return/void return/missing return、
  dot 累加、input-write snapshot、private/shared grid state 和非法变型。
- 既有 DSL/architecture/pass/grid/cost/language cohort **123/123**。
  此组执行在最后的模块拆分和 fixed-piece unmasked 优化之前；最终
  版本由上述 37 项及 ARM 跨编译再次验证，不将前一组称为又一次
  最终全量测试。
- SVE/SME，VLA/VLS/fixed-SVL，fixed/dynamic/scalable **16/16** 跨编译。
  没有 ARM 运行/性能数据。Ruff format 检查 language 和新增测试/脚本
  **97 文件**，language 的 F821/F822/F823/E9 和 `git diff --check` 通过。

最终 21 轮中位数（单次完整 32-step recurrence，单位 μs），同 CPU2、
同缓冲区、同一次测量中的 C++ 对照：

| Width | DSL register | 显式 C++ 同调度 | DSL forced scratch |
| ---: | ---: | ---: | ---: |
| 7 | 0.03166 | 0.03124 | 0.39216 |
| 17 | 0.03205 | 0.03227 | 0.39892 |
| 37 | 0.05017 | 0.04969 | 0.40964 |
| 64 | 0.05029 | 0.05039 | 0.25464 |

寄存器路径相对同调度 C++ 约 **−0.7% 至 +1.4%**，未观察到明显退化；
forced scratch 约慢 **5–12.5×**。CPU 频率不设限时绝对时间在复测间
有波动，因此不以不同轮次的最快值作比较。最终 JSON 保存在
`build-dsl-state/recurrence-final.json`；反汇编检查 width=37 的 register
版本，count 循环只有四组 add/FMA 和循环控制，没有 stack spill 或
tile scratch 读写。它不是对其他 shape、更多存活 states 或其他 ISA
的性能保证。所有修改仍在现有 `feat/new-dsl` 工作区，未自动提交。

## 后续原语交付：第二至第五层

第一层的 typed loop/branch/helper 状态仍然保留。本次按原顺序补齐
后面四层的 CPU 基础实现，没有增加 MLIR/LLVM Python 依赖，也没有
把 GPU/NPU 的 warp、异步事件或专用内存硬编码进顶层接口。

| 层次 | 公共接口 | 当前 CPU 路径 |
| --- | --- | --- |
| 2：值变换与索引 | `transpose/reshape/broadcast_to`、`shuf/local_shuf/element_at`、`gather/scatter`、`mask_all/any/none/count/first/last` | 布局折叠、原有转置 fast path、native vec 原语和显式 lane fallback |
| 3：局部执行 | `parallel`、`simd`、`ExecutionLayout` | ordered scalar axes + tagged SIMD induction，独立于任务级 `grid` |
| 4：fold 与跨 worker 通信 | `reducer/reduce/scan`、`team/shared/barrier/team_reduce`、`atomic_*` | builtin vector reduction，custom ordered fold/scan，resident-team ABI，`atomic_ref` |
| 5：流水与 ISA 包 | `prefetch/pipelined/double_buffer`、`amx.tdp`、`sme.load/mopa/store_tile` | 注册流水 pass、可选 guarded two-bank schedule、公开 AMX/SME 指令封装 |

### 模块与降级职责

- `frontend/{primitives,loops,reducers,collectives,pipeline,hardware}.py`
  处理源语言合法性、dtype/shape 检查和结构化 IR。通用 reducer 在
  这里展开为 typed scalar/tuple fold，保留 `kind/associative` 信息；
  layout 描述逻辑遍历顺序，不代表内存 strides，也不代表 CPU worker。
- `ir/{collective,pipeline}.py` 保留显式 team/barrier/collective 与
  软件流水请求。索引存储是 `IndexedStore` effect，不能被当成普通
  elementwise expression 重排。
- `passes/kernel/transforms.py` 注册 `fold-value-transforms`，对可证明的
  dense reshape 和 elementwise transpose 做表达式/布局折叠。一般
  computation-value 的变换仍有正确性 lane fallback。
- `passes/kernel/{pipeline,buffering}.py` 注册软件流水计划，在 Vector
  lowering 前完成 lookahead/缓冲变换；C++ 发射器不运行优化 pass。
- `backends/cpu/cpp/{primitives,collectives,hardware_sme}.py` 消费准备好的
  Vector plans 和控制 IR。所有 Vec/Mask pieces 均是独立变量；标量
  scratch/shared/buffer slots 可以使用 Buffer，不是向量数组。必要的
  resident callback 与 ISA region callback 仍带强制内联宏。
- 新增 C++ 公共 `execution/Atomic.h`，不从 DSL 引用 vec 的私有 details。
  team 沿用 resident execution ABI；CPU `team`/atomic 是线程语义，
  不承诺 Python 多进程之间的共享内存原子操作。

op family、effects、traits 和基础 cost 仍集中在 operation/cost registry，
本次新增的 indexed memory、register transform、mask reduction、atomic
不在各优化函数里另造 expensive 列表。

### 语义与保守边界

`shuf` 是 public Vec API 的 native-word-local shuffle，不是任意大 tile
上的全局置换；`local_shuf` 在每个 16B block 中应用常量 LaneOrder。
当后续消费者抽取标量或改变布局时，先在原布局求值，避免因更换 tag
改变 word-local 语义。尾部始终保留 logical active count。非 dense
reshape 暂时使用逐 lane 读取/写入寄存器的 fallback，不宣称所有值变换
都具有专用指令级实现。

indexed gather/scatter 接受 rank-one memory view 和任意 rank 的 int32/
int64 index tile；index 是逻辑元素下标，支持负物理 stride，active
indices 做边界检查。gather 使用 load_convert；unique scatter 可以
使用 native indexed store_convert。默认 duplicate scatter 保留逻辑顺序
的 last-writer-wins；unique=True 是用户保证。任意 indexed grid writes
尚无 ownership 证明，因此 auto 串行回退，require 报错。

内建 add/min/max reduction 继续使用现有 Vector reduction 快路径；
custom reducer 可以返回耦合 tuple（例如 value/index argmax），目前
保持 left-fold 顺序，即使标注 associative=True 也不自动并行重关联。
scan 支持 inclusive/exclusive、reverse、指定 axis 和 tuple reducer，
但当前是 ordered CPU loop + private scalar output，尚无专门的 SIMD
prefix scan。不能将这条正确性路径描述成与手写 SIMD scan 性能等价。

`team` 每个 resident worker 执行一次，worker 数同时受到 provider
capacity、调用预算和显式 requested workers 约束。shared 在进入 team
前初始化；内部普通 allocation 是 worker-private。barrier/collective
要求 team 顶层控制路径；目前不对一般条件/循环做 uniformity 推导。
enclosing scalar/tile state mutation 和非本地退出被拒绝。collective
通过共享 scalar slots 和两次 barrier 保证同步、避免提前复用；异常
由 provider cancellation 释放等待者。普通共享写的 race 仍由调用者
负责，编译器没有宣称提供完整的 race checker。

`pipelined` 目前只支持 two-stage CPU 软件流水。默认插入有边界保护
的预取 hint，不改变数值迭代顺序。`buffering=True` 仅适用于直线、
固定逻辑 tile extent 的纯读取/存储循环：首轮 preload，next bank
lookahead，current bank compute/store。输入与输出 root span 在运行时
做不相交检查；别名、控制依赖、state/reduction 依赖或 varying-width
tail 都保留 ordered fallback。它默认关闭，多出的标量 scratch 访存
不一定划算。`double_buffer` 还可由用户显式排布两个 bank；尚不提供
任意 stage 软件流水、GPU/NPU async copy/event 或自动依赖图调度。

AMX tdp / SME mopa 根据 typed operands 选择指令，handle 只能在相应
region 内使用。AMX 输入 packing、SME low-dtype grouped-K packing 是
调用者责任，不能等同于 portable `T.dot` 的 scheduler/packing。
所有可抛异常的尺寸/stride/feature 检查先于 ISA 状态切换；后端对
不支持的组合报错。本轮 ARM 只做跨编译，不提供真实运行或性能数据。

### 验证与性能记录

CPU 再次由用户限频，当前 CPU2 governor 为 powersave，max 为 1.8GHz。
不修改频率设置；性能只比较同一次运行中、同 buffer/affinity 的
DSL 与 C++，不和上一节解除限频时的绝对计时作前后对比。结果仅用于
已存在的 recurrence 路径回归，不能替代新 scan/流水/collective 的
专项性能调优。

本机 GCC 13.3、conda `af3` 的验证记录：

- 既有 `test_dsl.py/state/grid/cost/passes/language` cohort **144/144**。
  此轮发生在最后的 word-local state/two-bank 与接口边界修订之前，
  不把它称为最后代码的再次全量运行。
- 新增 primitives/local-loop/reducer/collective/pipeline 与既有架构
  cohort **45/45**。随后修订的 positional execution layout、tuple
  scan、empty fold/scan、非法输入、scalable SIMD、team 预算/初始化、
  word-local state、indexed conversion/negative stride 重点复测
  **15/15**；scalable widened-state 新增测试 **1/1**。
- 最后的 scalable consumer 尾部保护后，既有 state + architecture
  **37/37**；再补了前述 widened-state 测试。更宽 dtype 保持原 lane
  extent，同时按 descriptor basis 修正 native pieces 的数量。
- 状态 ARM 跨编译 **16/16**；新原语 ARM 跨编译 **30/30**，分别覆盖
  SVE VLA/VLS、SME VLA/fixed SVL。后者包括 indexed conversion/masks、
  local SIMD、scan/prefetch、自动双缓冲、widened state、team/atomic；
  SME raw mopa/store 是 FP32 编译覆盖，不宣称所有 low-dtype 指令都已
  实机验证。没有连接 ARM 主机或进行 ARM 性能测试。
- Ruff format check **113 文件**，language F821/F822/F823/E9、新增
  原语测试/跨编译脚本的 Ruff check 与 `git diff --check` 通过。生成
  C++ 检查未出现 Vec/Mask arrays，必要 callbacks 均保留 inline 宏。

复测命令：

```bash
conda run --no-capture-output -n af3 pytest -q \
  python/tests/test_dsl_primitives.py python/tests/test_dsl_loops_reducers.py \
  python/tests/test_dsl_collectives_pipeline.py python/tests/test_dsl_architecture.py
conda run --no-capture-output -n af3 pytest -q \
  python/tests/test_dsl_state.py python/tests/test_dsl_architecture.py
conda run --no-capture-output -n af3 python benchmarks/dsl_state.py \
  --width 7 37 --rounds 21 --repeat 50000 --cpu 2 \
  --output build-dsl-state/recurrence-primitives-limited.json
```

`benchmarks/dsl_primitives_cross.py` 与已有 state cross 脚本一样，需要
显式指定 clang 路径及 sysroot；本轮使用前文相同的 clang 19.1.3，
产物在 `build-dsl-primitives-cross`，只做 compile-only 检查。

当前限频下 21 轮中位数（单次完整 32-step recurrence，μs）：

| Width | DSL register | 同调度 C++ | Forced scratch | Register 相对 C++ |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.07417 | 0.07318 | 0.87388 | +1.35% |
| 37 | 0.11217 | 0.11110 | 0.91214 | +0.96% |

只比较同一行的同轮结果，寄存器路径没有明显退化，scalar scratch
仍显著更慢；不可将它外推成所有新 primitive、任意 dtype/shape 或
多 worker 调度均与手写 C++ 等速。原始记录在
`build-dsl-state/recurrence-primitives-limited.json`。

## 能力复核与 AF3 C++ 迁移准备

本轮复核针对当前代码，而不是沿用早期 TODO：原先五层的 CPU 基础
路径已经具备，但“有接口/正确性 fallback”和“所有算法均有最佳专用
降级”不能混为一谈。本轮不修改 AF3 仓库，也不迁移其生产调用；先补
会阻碍组合算法表达的能力，并保留源 C++ 路径作为之后的逐算子基线。

### 五层的剩余边界

| 层次 | 已有能力 | 尚未承诺的能力 |
| --- | --- | --- |
| 1：控制状态 | typed tile/vector loop state、if join、tuple/helper 返回、while/break/continue | 循环携带 memory view 的任意重绑定；一般控制流 uniformity 证明 |
| 2：值与地址 | 寄存器变换、显式索引、mask 查询；本轮补 singleton 变换、多维坐标 gather | 全局大 vector 任意 shuffle 的专用指令网络；多维坐标 scatter；一般 reshape 的最优网络 |
| 3：局部与任务执行 | ordered local axes、显式 SIMD/tag、grid static/dynamic/team-region | GPU fragment/thread/warp 布局；任意索引写入的 ownership 证明 |
| 4：reducer/scan/collective | builtin SIMD reductions、ordered custom scalar/tuple fold/scan、CPU team/atomic | custom associative 并行重关联、SIMD prefix scan、分支同步证明、跨进程 collective |
| 5：流水与 ISA | CPU prefetch、显式/可选自动双缓冲、AMX/SME operands | 任意 stage 依赖调度、GPU/NPU async copy/event/TMA、所有 ISA 的精细 cost/scheduling |

第二至第五层剩余项多数是性能特化或后端能力，不能通过增加同名空 API
算作完成。现有 CPU 双缓冲不会被标成 GPU 异步语义，CPU team/atomic
也不被标成多进程/MPI 语义。

### 与 TileLang / Triton 的对应

对照官方 [TileLang language API](https://www.tilelang.com/autoapi/tilelang/language/index.html)、
[loop API](https://www.tilelang.com/autoapi/tilelang/language/loop/index.html)、
[reduction API](https://www.tilelang.com/autoapi/tilelang/language/reduce_op/index.html)
以及 [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)：

| 组件 | 当前 DSL 对应/本轮补充 | 处理原则 |
| --- | --- | --- |
| shaped values / memory views | Tile/Vector、strides、load/store_convert、transpose、expand_dims/squeeze、indices、coordinate gather | 顶层保持逻辑值/布局语义，ISA 表示由后端合法化 |
| fragment / execution layouts | ExecutionLayout + Vector tags；CPU grid/team 与 local SIMD 分层 | 不把 CPU scratch 冒充 GPU shared/fragment；未来布局属于 backend 模块 |
| Parallel / serial / Pipelined | local loops、grid、two-stage CPU pipeline | 当前无 GPU warp layout、arbitrary-stage async 语义 |
| reductions / scans / atomics | 已有基础实现；未重关联的 custom fold 保留顺序 | SIMD prefix、GPU warp/cluster collective 仍是专用 lowering 工作 |
| allocation / composed algorithms | 本轮新增 replayable workspace、稳定 gate 数学、masked softmax/RMSNorm、在线分块 SDPA | 不通过 opaque ops::Attention/Matmul 调用隐藏算法 |
| 其他通用算法原语 | sort/top-k/histogram、随机数、通用 bitcast、外部 intrinsic 扩展协议尚未全面覆盖 | 本次 AF3 C++ 加速内核清单不依赖它们，不为 API 数量盲目扩张 |

这是一份针对 CPU/AF3 的取舍，不是 TileLang/Triton 全 API parity 声明。
后端独立的 shape/address/状态语义可以继续扩展；异步资源、fragment、
硬件精度/指令包应通过后端类型和模块加入，而非顶层硬编码 GPU 概念。
不增加 MLIR/TVM/LLVM Python 依赖。

### AF3 实际内核需求

审查路径：
`~/project/af3-torchneo/src/alphafold3/torch/acceleration/vecops/kernels/resources`。
重点读取 indexed cross-attention、single/grid/MSA attention、transition、
OuterProductMean、TriangleMultiplication 和 `StableMath.h`；没有更改这些
文件。由源代码需求得到以下迁移分组：

| 内核族 | DSL 所需组件 | 当前状态/下一步验收 |
| --- | --- | --- |
| Linear/Matmul/PackLinearB/PackMatmulB | cache-blocked matmul、packed operands、bias/residual epilogue、mixed dtype | 已有基础；迁移时保持 C++ packed ABI 和每种 shape/fusion benchmark |
| Transition / DiffusionTransition / Conditioning | layernorm、adaptive affine、stable sigmoid/SiLU、连续 GEMM、private/replayed scratch | 本轮补可组合稳定数学与 workspace；校验 rounding、gate 算法和中间精度 |
| Self/Cross/Grid/MSA attention、SingleActUpdate | 多头/批次 query grids、bias/mask、online max/sum/acc、QK/PV dots、gate epilogue | 本轮补 visible SDPA；功能表达具备，但 prepared C++ Attention 的策略/packing/acc 状态调度尚未性能等价 |
| CrossAttentionIndexed / IBS / Indexer | 先 unique-atom projection，再按 row/feature 多维坐标 gather、局部 key/value tiles | 本轮补坐标 gather；sparse/indexed scheduler、窗口特化和 packing 复用仍需迁移期专项优化 |
| OuterProductMean / TriangleMultiplication | 轴变换、transpose、mask/归约、分块 contraction、workspace 多阶段复用 | 基础原语可组合；具体调度与 scratch 生命周期/coloring 需要逐内核迁移及测量 |
| Distributed* projections/updates | 本地计算 + 现有分布式通信边界 | 本地 DSL 可替换计算；MPI/跨 rank 通信保留框架/provider 层，不用 CPU team 模拟 |

“表达所需基础能力齐备”不代表所有 AF3 C++ 内核已经转换，也不代表
107aa TM-score/RMSD 和前后速度已验证。生产迁移仍需独立分组验收：先
Linear/pack/transition，再 attention/indexed，最后 contraction/distributed
本地算子；每组清空编译缓存、同轮 C++ 对照，并最终运行 107aa 和
`compare_structures`。本轮没有 AF3 生产绑定修改，所以不声称这些验收
已经完成。

### 本轮实现与性能边界

1. 新 `frontend/shapes.py` 提供 `indices/expand_dims/squeeze`；值变换在
   广播到更高 rank 时正确转换 consumer axis，singleton 保持广播语义。
   `indices` 降到 iota/fill，不生成坐标网格数组。
2. `gather(view,(i0,...,in),...)` 对每个逻辑 axis 做 active bounds check，
   合成物理 stride offset，然后 native indexed load_convert；支持负
   stride、padding、mask 和 alias snapshot。scatter 仍限 rank-one view。
3. 新 `ir/workspace.py` / `frontend/workspace.py` 保留 serial allocation
   sites。C++ emitter 选 `WorkspaceContext&`，复用既有 replay adapter。
   新公开 accessor `WorkspaceContext::thread_pool()` 借用 executor，不
   赋予 worker 并发分配权限。元数据和当前执行预算确定 plan key；任意
   scalar/data-dependent extent 被拒绝，避免错误 replay；被更新过的
   metadata local（含前置 branch/loop）不能继续作为尺寸证明。不可变
   shape tuple 还支持 constexpr slice 和 starred tuple expansion。
4. `T.workspace((T.worker_capacity(),*tile_shape))` + `T.team` 可手工
   复制持久 worker scratch，但默认 dense、无自动 cache-line padding。
   这不等同于 AF3 C++ `worker_tensor` 的完整 prepared/lifetime 规划；
   普通 grid 内 T.alloc 仍仅是 invocation/worker-private reuse。
5. `_library/math.py` 提供 AF3 schedule 的 stable sigmoid/SiLU、RMSNorm、
   finite-logit masked softmax。SME polynomial 显式选择，不把目标 cost
   profile 误当成实际硬件 capability。rank-two last-axis 的后两者扩展
   registered normalization pass，hoist row statistics，避免 consumer
   微块中逐元素反复规约；其尾部仍能透明融合到 dot/packing/store。
6. `_library/attention.py` 使用 visible dot、reduction、loop-state 实现
   streamed SDPA，支持 bias、numeric masks、causal offsets、zero/all-masked
   rows 和 epilogue。每块 QK scale/bias 在 product store 中融合，只存
   当前 query-tile × key-block 分数，未创建完整 scores matrix。
   概率默认按 V.dtype 舍入以便矩阵指令，float32 可显式关闭此舍入。
7. 组合验证修正无 acc 的 dot 状态分析、state sink 的 dependent-dot
   materialization，以及矩阵融合的 shape validation：只验证合法的
   elementwise 边，不能把 reshape/expand/reduction 的子值误当作根
   输出的直接广播。row slicing 保留 predicate 类型与变换坐标。
8. literal locals / 纯常量 helper 返回保留 compile-time identity，支持
   `static_assert(helper(...))`；进入会修改它们的 runtime loop/branch
   才创建 mutable scalar slot。连续 grid 中的纯常量赋值无需 worker
   publication/barrier；原调度测试改用真正的 runtime scalar 输入验证
   single-worker phase，而非要求编译器同步一个无需运行时计算的常量。

在线 SDPA 当前是功能与编译器组合基线。QK scores、部分 matrix state
和 probability/统计仍可能走标量 scratch，packing/ISA resource setup
未完全 hoist；与 C++ prepared Attention 的性能等价还不能承诺。这些
是后续应深入生成结果优化的具体位置，而非声称所有 fusion 已完成。
本轮继续遵守 Vec/Mask 拆成独立变量、必要 callback 强制内联；不更改
用户的 CPU 限频，也不混用不同频率轮次的计时。

### 本轮验证记录

执行环境为本机 x86（conda `af3`），保留用户当前 CPU 限频设置。
没有连接 ARM 主机；ARM 仅作 clang 19.1.3 跨编译检查。

| 测试组 | 结果 | 覆盖范围 |
| --- | --- | --- |
| attention_library + memory_extensions + attention_edges | 23 passed | 多维负/padded stride gather、workspace replay/执行预算、worker replicas、masked softmax/RMSNorm/gate、在线 attention 的 causal/all-masked/empty/dynamic cases、BF16 概率和显式 AMX |
| workspace_rules + attention_codegen | 9 passed | replay 尺寸不可混入 runtime scalar、tuple shape 语法、row-stat hoisting、visible dot、独立 Vec/Mask locals、inline callback、constexpr helper |
| grid + loops_reducers + collectives_pipeline | 54 passed | 调度、循环/reducer/scan、team/atomic 与 CPU pipeline 回归 |
| dsl + state + passes + cost + language | 133 passed | 既有基础 DSL、状态、pass、cost 和语言回归 |
| matmul_codegen + matmul（下述筛选） | 25 passed, 16 deselected | dot/matmul 代码生成及针对性执行回归，不是完整 MatmulBench 性能扫测 |
| state + dsl（constexpr 修改后的下述筛选） | 15 passed, 56 deselected | while/helper/tuple/control/状态 snapshot/reduction 条件 |
| ARM primitive cross | 42/42 compile-only | SVE VLA/VLS 与 SME VLA/fixed SVL；包含新 coordinates/workspace/attention 组合 |

133 项基础回归和 25 项矩阵回归先于最终 constexpr-local/helper 与
tuple 语法调整运行；最后的 15 项控制流、9 项纯代码生成/规则和
54 项调度/collective 回归验证了该调整。测试组存在重复，不能把表中
数字相加作为独立测试总数。调度 fixture 最终改用 runtime `count_input`
后，又运行相关筛选回归，5 passed、21 deselected。本轮不报告新旧 CPU
频率间的性能比例，
也没有运行 C++ prepared Attention 的等价性能验收或 AF3 107aa。

主要复测命令：

```bash
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl_attention_library.py \
  python/tests/test_dsl_memory_extensions.py \
  python/tests/test_dsl_attention_edges.py
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl_workspace_rules.py \
  python/tests/test_dsl_attention_codegen.py
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl_grid.py python/tests/test_dsl_loops_reducers.py \
  python/tests/test_dsl_collectives_pipeline.py
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl.py python/tests/test_dsl_state.py \
  python/tests/test_dsl_passes.py python/tests/test_dsl_cost.py \
  python/tests/test_language.py
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl_matmul_codegen.py python/tests/test_dsl_matmul.py \
  -k 'not product_layouts and not tails_packed and not panel_phases and not packed_abi'
conda run --no-capture-output -n af3 python -m pytest -q \
  python/tests/test_dsl_state.py python/tests/test_dsl.py \
  -k 'while or helper or tuple or predicate or control_flow or aliasing_snapshots or reduction_condition or axis_reductions'
conda run --no-capture-output -n af3 python benchmarks/dsl_primitives_cross.py \
  --clang /opt/spack/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/llvm-19.1.3-2tjty3xfpnxpvl3l2ghtirmts4fyekim/bin/clang++ \
  --sysroot build-dsl-cross/sysroot/usr/aarch64-linux-gnu
```

跨编译产物位于 `build-dsl-primitives-cross`。Ruff format check（121 个
文件）、language 的 F821/F822/F823/E9 检查和 `git diff --check` 通过。
自动 worker scratch padding/lifetime coloring、SIMD prefix scan、prepared
attention 的 packing/resource hoisting，以及 GPU/NPU async pipeline
仍按前述边界保留为明确的后续工作，不以功能 fallback 冒充已完成的
专用性能路径。

## DSL 调试与内存可观测性（2026-10）

调试逻辑独立放在 `language/_debug`：session/options 管编译期快照，
printer/report 管可读 IR 和导出，resources/evaluate 管纯 metadata 估算，
machine/profile 管实际产物与显式计时，interpreter/minimize 管参考执行与
有界缩例。各编译阶段仅接入 session；不会从调试模块反向修改编译策略。
展示接口不承诺跨版本稳定，缺失信息明确返回 Unavailable。

`T.lower` 不触发 native build；Template/CompiledKernel/JIT factory 可取
源码、前端/kernel/vector/backend IR、pass trace/diff、cache/build 信息和
实际 object 的反汇编。失败保留当时 debug_info；Python forwarding 函数
不改变调用身份，调试报告聚合其可发现子 kernel。CPU analyze 的占用率
定义为 worker coverage/wave efficiency 加寄存器/工作区估算，不冒充
GPU occupancy 或精确 spill 计数。profile 分开 cold preparation 与 hot
samples，记录当时 affinity/quota/frequency，不修改用户机器频率。

打印统一为 `T.static_print` / `T.print`；runtime 使用 format_to→定长
栈缓冲区→共享 C stream 锁→单次 fwrite。默认 1024 字节和
`" ... (trunc)"`，新增独立 `truncation` 参数，标记无条件预留容量，
sep/end 也算正文，溢出保留完整 UTF-8 前缀并附加标记。不生成中间
std::string，不创建 Vec 数组，不另做 `<print>` 路径。线程/DSO 共用
libc FILE 锁，原子性只覆盖一次调用，不覆盖多个 end="" 调用。
浮点 formatter 内部使用定长 to_chars 缓冲，避免标准库在极大固定小数
或高精度字段上的 std::string fallback；精度上限 64，locale/debug-escape
格式拒绝，数值 fill 为 ASCII。这仍是同一 format_to pipeline，不新增
独立打印路径。

内存 ABI 只追加 observer 的可选 tail，经 struct_size 检查，execution
context 布局不变。编译插桩可记录 scratch 分配/释放和 kernel/grid-program
区间；默认无插桩。ID 匿名、collector 有上限、丢事件显式报告。当前不
宣称捕获所有 malloc、packing 子阶段或 barrier wait。参考 interpreter
对未支持的原语明确拒绝，缩例必须由调用者提供确定性的 failure oracle。

缓存从手工版本 label 转为完整 content/build identity，包括 SDK/DSL、
callback、C++ source/dependency、toolchain、Torch bridge 依赖；编译前后
检查和原子发布防止并发编辑污染。热执行使用 immutable prepared snapshot。
输出默认简洁 progress，另有 quiet/verbose；Torch bridge/native kernels
仍保留不同 ABI/toolchain 的两个批量 CMake 阶段，不为了减少一个 configure
将 Torch 依赖引入普通内核。完整 API、范围和限制见 docs/Python.md。

### 调试系统验证记录

验证在本机 x86、conda `af3` 下执行，保留用户的 1.8 GHz 限频。
未连接 ARM 主机。最新打印路径通过 SVE-VLA、SVE-VLS、SME-VLA 和
SME-fixed-SVL 四种 clang 19.1.3 跨编译检查，不代表 ARM 实机执行通过。

- 调试/打印/缓存/单元素 lowering 回归：25 passed；覆盖自定义标记预留、UTF-8
  边界、空标记、标记占满缓冲、NUL、浮点格式、零 `operator new` 分配、
  多 worker 打印完整记录、scratch 生命周期/时间线、参考执行和导出。
- runtime + language 回归：34 passed，覆盖 NumPy/Torch、自动输出和重叠
  存储、批量编译与原有运行时 ABI。
- 扩展 DSL 回归初次运行有 133 passed，另有 6 次并发编辑触发的输入
  变更拒绝、1 处公共 constexpr helper 被误计为 reduction 闭包的断言、
  1 次主动取消的矩阵编译。没有把这次运行计为全部通过。断言限定到
  `__kernel__` 后，6 个非矩阵相关用例在冻结源码下复测通过。
- 矩阵调度测试用本机 LLVM 19 Clang 再执行全部 8 个用例：FP32/BF16
  的 off/prefetch/staged、动态零维和尾部、prologue/epilogue、缓存分块
  phased fusion 均通过。Torch bridge 保持其 GNU OpenMP ABI 的 GCC
  编译器；内核使用 Clang，未冒称整个矩阵组均在 GCC 下重新跑完。
- 额外回归保护 Torch bridge 复制 toolchain 时继承 quiet/progress/verbose
  和 build observer，避免只在 native kernels 阶段生效。

源码 hash 的 fork 回归确认子进程不会消费父进程的通知队列；子进程
重新建立 watcher，不能建立可靠 watcher 时回退为内容重算而非 mtime
判定。没有把 immutable prepared execution 变成每次调用的目录扫描。

同机、同频率、CPU2/单 worker 的新旧生成 DSL 配对测量（11 轮交替，
每轮 10000 次 prepared native 调用）中，当前/旧版中位耗时比为 SIMD
1.0141、moments/reducer 1.0003、scan 0.9999。scan 曾因单元素 load 被
替换成 fill 出现额外广播，已恢复 x86 原生 load_convert；SVE 保留
安全降级。这仅是这三个 4096 元素样例的无显著退化检查，不是完整
算子性能或跨频率比较。原始结果在忽略的
`build-dsl-debug-performance/paired.json`；未修改 CPU 频率。

最终 25 项回归在输出模式/observer 转发和无效 file 参数校验修正后运行。
Ruff format check（146 个文件）、新增调试文件的 Ruff check、language 的
F821/F822/F823/E9、compileall 和 `git diff --check` 均通过。

## ARM SME matrix validation: runtime panels and raw ZA packing

Runtime-small SME library products now use the same overflow-safe working-set
budget predicate as the AMX path before querying cache sizes. Automatic
blocking may be bypassed below the target runtime budget; explicit BM/BN/BK
still force the requested panel policy. The unblocked leaf preserves packing,
input transforms, the initial accumulator, and a single final epilogue. Large
products retain NC/KC/MC continuation in native accumulator storage.

Direct ZA accumulator seed/store loops precompute slice base pointers and
signed strides; matrix-family bounds precompute immutable dimensions. This
is necessary because public ZA instructions carry an assembly memory clobber,
which otherwise prevents CSE of tensor metadata inside the instruction loop.
On the same 920f-lx target, the paired Dynamic BF16 16x16x8 packed-AB source
case changed from DSL/C++ 2.28386 to 1.07927 after this stage. The new paired
times were 0.35330225/0.32735425 microseconds. This is one measured case, not
a claim of full matrix-shape parity.

Raw SME packing is a separate backend emitter in `cpp/sme_packing.py`. It
uses only public `vec::sme::{load_hor,read_ver,store_ver}` and vector
primitives, never an opaque MatmulPack call. A same-dtype unit-K-stride
guard (using actual alias-snapshot strides) chooses it. Spatial strides may
be negative. Full <=32-bit panels use ZA32 pair 0/1 and pair 2/3 to pipeline
load/drain of K chunks; FP64 uses ZA64. Fringes stage at the storage element
width, zero masked K lanes, mask absent rows on read, and assemble K groups
as individually named native words. Whole-word ZIP halves are required:
the public `local_interleave_*` operations are 16-byte-block-local and are
not a substitute at SVL >16 bytes. Values are extracted directly from the
public interleave expression; no multiword Vec variable or Vec array is
declared. Callback-entry scalar copies prevent reference-capture reloads
from reappearing around inline assembly clobbers.

The raw-packing stage initially leaves conversion/elementwise packing on the
existing vector path, and does not merge packing and multiply streaming
owners. The new ARM
execution tests use arbitrary storage bits, full/tail panels, zero axes,
negative spatial strides, non-unit-K fallback, aliases and output sentinels;
their oracle derives panel geometry from the current-thread SVL. Local
proof includes native byte/halfword ZIP ordering at every active-row count
and 48 true-FA64 ARM assembly instantiations over six dtypes, A/B, and
VL/SVL pairs 16/64, 64/16, 32/64 and 64/64. Each assembly gate checks ZA
input, streaming ownership, native word ZIP where required, and absence of
matrix-multiply instructions in packing-only wrappers. Cross-compilation
does not establish ARM runtime correctness or speed; those results must be
recorded separately from the scheduled target run.

The next conversion-packing stage admits a conservative FP32 pointwise DAG:
an unmasked full-shape FP32 storage leaf is loaded into two ZA32 tiles, read
as spatial FP32 vectors, and substituted into the ordinary Vector emitter.
FP32 row statistics and column parameters retain their separate broadcast
coordinates. Each resulting half-panel is narrowed independently and its
K=2/4 slots are interleaved into one named native word, keeping every live
ScalableTag at POW2 <= 0. Neither Vec arrays nor opaque MatmulPack calls are
introduced. Distinct mul/add nodes remain distinct; only explicit FMA has
fused arithmetic semantics. Missing spatial lanes are zeroed after any
non-zero-preserving affine expression and before narrowing. A padded K slot
does not evaluate its expression or access mean, scale, weight or bias.

Selection does not traverse reduction, materialized, state or register
transform boundaries, and widening 16-bit auxiliary captures stays in the
ordinary domain. A final conversion folded into a load uses a separate FP32
computation leaf without discarding an intermediate FP16 rounding. Alias
snapshots keep the original load identity; if such a snapshot is already
narrow storage, that alias takes the original vector fallback instead of an
invalid narrow/FP32 pointer selection. The conversion route does not promise
raw NaN-payload preservation: it inherits ordered/saturating public vector
conversion semantics. Its paired target tests use the real public C++
MatmulPack, with identity, elementwise or lane-local transforms as appropriate,
and a separate finite-value oracle for broadcast coordinates, saturating
integer conversion, full/tail panels, non-unit-K fallback and output sentinels.
Packing/multiply ownership merging and staged BF16/FP16-to-FP32 input expansion
remain subsequent work rather than being hidden in this specialization.

### SME exact-block leaves and capture lifetime

The streaming resource trampoline can outline a reference-capturing callback.
Even a const extent captured from its caller becomes address-exposed to the
public ZA instructions' memory clobbers. The dot callback therefore copies
M/N/K into callback-owned values before issuing ZA instructions. The actual
packed four-MOPA K loop is checked for the absence of scalar extent reloads,
rather than assuming an outer const declaration guarantees CSE.

Small runtime domains select exact 1×1, 1×2, 1×3, 2×1, 3×1 and 2×2 ZA leaves
before K. Their non-last spatial bands have a full compile-time slice extent;
last bands still mask raw inputs and output tails. All block counts derive
from the current-thread captured SVL and accumulator dtype, not a 64-byte
assumption. Larger domains retain the existing rectangular families. These
leaves increase generated-source volume, so cross-compiled hot-loop proof is
not performance acceptance: target correctness, paired C++ timings and cold
build/code-size costs must be assessed together before extending the catalog.

### SME column drains and mixed packed spatial proofs

A same-dtype, unmasked output with a proven unit row stride and one active
column drains ZA through one vertical memory slice. Dynamic output strides
are checked outside the K loop; nonunit/negative strides, transformed output
and wider tiles retain their original orientation and vector/store path.
Initial accumulators use the corresponding direct ZA axis only when their
layout independently proves it safe. This does not bypass prologue/epilogue
or alias ordering.

A whole native packed input may supply full physical spatial lanes for a
logical fringe, independently of the other raw operand's full-lane proof.
This relaxes only the spatial full-family dispatch, never the raw K bound.
Packed windows, including an explicit zero-origin window, retain their
conservative condition until their panel origins are independently proven.
Odd raw K still uses its masked tail; padding is consumed only when both
input formats guarantee it. All tile geometry comes from captured SVL.

### SME transpose origins and small VLA conversions

Actual stage-6 ARM artifacts showed that constant 16×16 conversions could
select a 128-bit-prefix register permutation even at ordinary VL=64 bytes.
For FP16→FP32 that path measured 0.3520/0.1035 microseconds (DSL/C++) and
used four-lane shuffles/stores, whereas the C++ VLA path used native gathers.
The CPU memory planner now admits only automatic, simple rank-2 conversions
with a constant positive element count rejected by the existing SME preference
policy and a captured ordinary SVE VL wider than the safe prefix. That single
policy rejects conversions with ≤256 elements and eight-byte inputs with
≤8192 elements. Actual strict stage-18b ARM 63×129 FP64→FP32/Int8 artifacts
confirmed the same mismatch: the C++ hot loop used eight-lane `LD1D` gathers,
while DSL used two-lane `LD1D` loads and `ZIP1` prefix permutations. The planner
therefore reuses the original SME rejection decision, not a second threshold
table, to select the existing gather/store-convert Vector IR with unroll=1.
Raw values, larger/dynamic shapes, compound producers,
unknown VL and explicit implementation choices retain their original policy.
This preserves conversion boundaries, snapshots and signed-stride fallbacks;
it does not infer VL from SVL or from a cost-profile override.

Stage-49 isolated the remaining tiny-constant gap using all 12 original
16×16 mixed-dtype triples, four LLVM row-unroll choices, exact numerical and
overlap/canary checks, and 60 actual AArch64 modules. Two/four-row hints
regressed Int8→FP64 by about 10%, while disabled outer auto-unrolling had
median 0.8681× the unchanged duration and worst 1.0012× in that paired period.
The planner therefore keeps only tiny automatic conversion-gather rows
rolled (both constant axes ≤16), including their cold snapshot alternatives.
`VectorRegion.outer_unroll` is an independent schedule hint; the explicit
inner vector grouping/unroll, ordered conversions, snapshots and tails do
not change. Other targets, explicit implementations and medium gathers keep
the native compiler's existing row-loop strategy. The isolated measurement
is diagnostic evidence, not a claim that the combined full catalog passed.

Stage-8 worker transpose disassembly independently confirmed that outer
parameter snapshots captured by reference were still reloaded after ZA memory
clobbers. SME transpose now copies this invocation's source/target origins,
dimensions and signed row pitches into local SSA inside its actual owner.
Origins incorporate view offsets and leading batch coordinates once. Empty
views do not form offset pointers, and owned scratch data is read after its
latest resize rather than being globally cached. The contiguous axes are
already guarded before entry and their unit-stride proof is restated locally.
Raw loads/stores remain direct LD1/ST1 ZA slices; conversion paths retain their
existing native vector precision and predicates.

Cross-codegen distinguishes remaining stack spills from escaped-closure
reloads. LLVM may fully unroll a bounded eight-slice loop and keep many
dynamic-pitch address multiples live. The experimental pass option
`cpu_transpose_sme_dynamic_unroll=True` requests two-way Clang/GCC unrolling
only for those dynamic pitches; it defaults to `False` until target paired
measurements decide the tradeoff. A lack of scalar loads is a code-generation
check, not by itself a speed result. ARM correctness and timing results for
these new paths must be reported separately from cross compilation.

## Library/region boundary refactor (2026-10-07)

Algorithm formulae belong in source-parsed DSL helpers, not operator-name
branches in compiler passes. `T.map_rows(helper, input, **scalar_options)`
stages an ordinary rank-one program over a rank-two logical tile. The generic
region scheduler either fuses a sole row-separable consumer under whole-view
alias/nonoverlap guards, or exports only row-varying scalar intermediates for
later consumers. It preserves historical captures and helper producer DAG
identity. LayerNorm, RMSNorm and Softmax use this mechanism; there is no
`_normalize`/`row_normalize` algorithm in the compiler. Stateful, allocating,
runtime-control and external-tensor-option callbacks are initially rejected.

The remaining migration order is executable hardware regions and accumulator
state, packing/register-transpose algorithms, vector and architecture matrix
leaves, then cache traversal and automatic packing. A mere relocation of a
C++ string emitter does not count as DSL migration. Preserve explicit FMA and
conversion boundaries and named register pieces, and measure before/after
using frozen source snapshots on the same x86 and scheduled ARM machines.
Checkpoint `6f57d0a` is the pre-refactor baseline, not a claim of prior universal
ARM performance parity. Native/benchmark execution must not import a live
editable tree that is being changed during its build.

### Register algorithms (second migration slice)

Portable transpose butterflies and the 512-bit dword packing permutation
are now ordinary `_library/registers.py` `T.func` programs. Static branches,
tuple construction, split/concat, bitcast and local/block interleave compose
their logic. `RegisterProgram` contains ordinary typed Expr nodes rather
than a separate transpose-specific instruction vocabulary. The same terminal
printer serves transpose, AMX packing and explicit logical register stores.
The compiler adapter checks pure closed geometry; it no longer implements
the butterfly. Oversized/non-power-of-two logical vectors use individual
native words/scalar lane indexing, never a C++ array of Vecs. Primitive
geometry admits complete fixed and scalable register extents; arbitrary
clipped/dynamic bit geometry remains unproved. Alias preservation is mandatory even when source and
destination have identical whole-view mappings: a permutation is not an
elementwise in-place update.

This is not completion of the whole library migration. SME ZA traversal,
interleaved pack drivers, cache blocking and matrix-family algorithms still
need executable DSL/hardware-region support before their C++ emitters can be
removed. Adding reusable frontend/legalization mechanisms initially increases
total production size; the final code-reduction requirement is not satisfied
by moving files or by this intermediate slice alone.

Validation records: fixed phase-one snapshot has 52 x86 native test passes
and 109 non-native passes; sixteen paired LayerNorm cases over 32 balanced
rounds have median new/old 0.999887 (range 0.989898–1.013308). These are scoped
checks, not complete catalog or ARM parity claims. ARM job 1816914 validates
the same phase-one source/SDK inputs privately on 920f-lx using clang;
it completed with 52 native passes and no skips. Its 28 paired LayerNorm/
Softmax cases over 32 balanced rounds had median new/old 1.000378, range
0.989179–1.005086. The phase-two x86 frozen recursive-register snapshot passed
17 selected native tests; 36 TransposeBench cases over 24 balanced rounds had
median approximately 1.0 and range 0.991783–1.005128. Earlier register-order
variants regressed and are not accepted performance evidence.

### Structured hardware regions and grouped-K streams

Explicit SME/AMX regions now admit bounded serial instruction loops and
loop-invariant metadata branches. A dedicated CPU preflight pass uses the
shared affine/interval analysis to check indexed views and tile capacities
before ownership, without replaying instruction loops. Allocation, mutable
state, ordinary SIMD computation, memory-dependent control and unbounded
loops remain illegal. Unknown nonlinear intervals or lost root provenance
are diagnosed conservatively. This is a deliberately limited executable
foundation, not a claim that arbitrary DSL control can run in streaming mode.

Raw `T.sme.load_hor/ver(tile, index, view)` and `store_hor/ver` expose direct
bit-preserving memory/ZA slice instructions. `T.sme.transpose_tile` implements
the word-sized same-dtype algorithm in DSL: load all rows, then drain columns.
This public helper is not yet the default large/converted transpose driver.
It retains one owner and derives capacity from the actual streaming length.
Optional compute dtype now composes converting loads, ZA vector writes/reads
and converting stores; those are small region-local primitives, not a second
opaque transpose algorithm. Immutable metadata declarations are expanded by
preflight; ordinary throwing/SIMD/stateful work remains disallowed. Invariant
whole-view owned/reshaped operands retain their existing compatibility path,
while loop-dependent accesses require root provenance. Allocations now retain
their own logical root coordinates as workspace declarations already do.

Grouped-K interleaving is also a source-parsed library function, not a C++
algorithm in `interleave_group`. Two streams use one `T.interleave`; four byte
streams combine two pairs through an explicit 16-bit reinterpretation. The
same staged Expr DAG serves AMX fixed columns and scalable SME/vector packing.
Bitcast retains the original scalable byte-domain/tag, never infers VL from
SVL, and never treats a clipped load as a complete scalable register. These
new paths require their own frozen native, ARM and performance validation;
the previous records do not cover them.

The next slice adds exact clamped view coordinates, correlated nested-loop
intervals and positive native-length serial steps. These are general control/
memory mechanisms, not a transpose-name pattern. A source-parsed larger
`sme.transpose_matrix` composes one owner and clipped word tiles. Default CPU
transpose emission has not yet switched to that driver. Fixed-register
placement can use a captured ordinary VL with backend-planned scalable tags;
unknown VL still uses the guaranteed prefix. SVL cannot establish that proof.
Parallel/fusion requested-affine partitions remain conservative after nested
clipping even when exact hardware access coordinates are retained.
The experimental CPU option `cpu_transpose_sme_dsl` prepares the executable
library body, including paired traffic and bounded full-tile conversion
branches, and binds it to invocation-local SSA at emission. The old driver
remains the default pending paired ARM measurements. Constant pitches remain
constant in the staged body; fringes keep their typed predication. This option
must be removed when the complete measured algorithm is migrated, not kept
as a permanent second implementation.

CPU transpose policies now live with the CPU planner, not in portable IR.
The shared `RegisterProgram` lives in vector IR and is also used by packing;
the old transpose-specific IR module has been removed. Nested immutable
tuples can carry scalar/tile/vector slots through structured control. All
sources are captured before any slot update, and each register is emitted as
an individual variable, including sizeless SVE. Tuple arity/types cannot
change at runtime. This enables library register banks without C++ arrays.

ARM phase-four validation exposed a conservative interval bug for clipped
matrix tails. Preflight must simplify induction-domain clamps before bounding
their differences, preserving the correlation between an origin and its
remaining length. It must also retain access/capacity checks and write effects
when a rank-one store needs no separate layout check. Pure metadata tests cover
multiple streaming lengths, empty dimensions and multiword matrix extents.
The phase-five x86 runner reported 171 passes with 15 SME platform skips, but
that local invocation selected the frozen package without rebasing relative
pytest paths. It is not an all-inputs-frozen acceptance gate. The runner now
executes and records its working directory inside the snapshot. A corrected
phase-six x86 run passed all 65 state/matmul/attention tests from that directory.
The updated ARM snapshot and paired driver measurements are still pending.
The phase-four ARM matrix/attention suite passed 22 tests with 10 platform skips;
the same job's transpose failures are not accepted validation evidence.

ARM phase-five passed the structured matrix transpose and MOPA tests, but
exposed two small-register compilation failures. Captured VL can fit a
scalable descriptor while C++'s implicit Half<Tag> has no representation at
the architectural minimum. CPU implementation contracts record those implicit
half domains centrally; small fragments use fixed prefixes and unsupported
combinations retain exact generic legalization. The fixed/scalable bitcast
boundary reinterprets one physical SVE word without changing logical bytes.
This does not alter the separately measured automatic-transpose register
programs, which retain their own placement policy.

The dynamic AMX cache decision is shared before family dispatch, reducing one
transformed-input generated source from 23.46 MB to 14.07 MB. This is a code-size
result, not accepted performance evidence. It is currently gated by
`cpu_amx_shared_cache_dispatch=True`; the default retains the original policy.
In particular, automatic B-packing
and partial-panel family selection still need large and slim-matrix paired
measurements. Native FP16 Matmul's public C++ reference requires AMX-FP16;
Sapphire Rapids supplies only the DSL vector fallback for that dtype. Paired
benchmarks fail early for unsupported reference kinds rather than substitute
BF16 or silently report an incomplete catalog.

The next SME packing slice stages `_library/sme_packing.py` before terminal
emission. Its full-word schedule explicitly primes a ZA pair, overlaps loads
with stores through two alternating pairs, and drains the final pair. An
optional `word_dtype` on raw slice moves reinterprets storage rather than
performing numerical conversion. `T.prefetch` is legal in bounded hardware
instruction regions; its address still receives root-coordinate preflight
checks before ownership, even though it consumes no tile/vector capacity.
The temporary `cpu_sme_packing_dsl` option remains off by default while native
correctness, instruction order and balanced MatmulPackBench measurements are
being checked. The experimental fringe schedule now uses storage-masked
`sme.load_bits` / `store_bits` and word-sized ZA reads/writes, including
fractional K zero padding. These are small region-local memory primitives,
not operator algorithms. Panel traversal is now also part of the experimental
`pack_raw` DSL recipe, not the terminal printer. The default raw driver and
transformed packing remain unchanged pending native/performance validation;
this is not a completed packing migration.

Paired packing validation derives an independent zero-padded
`[spatial-panel, K-group, spatial-lane, K-lane]` oracle from the actual SME
vector length. All old/new/public-C++ variants must match it before and after
timing. The source snapshot, SDK/native input hashes and compiler environment
are pinned; failed or mixed live/snapshot runs are not acceptance evidence.

Instruction metadata uses lexical SSA before terminal printing. Multiple
independent clamps are not distributed into a Cartesian product of arms;
equivalent pure metadata expressions are shared within their defining scope,
never across a branch or a loop-index shadow. This reduced one whole packing
wrapper from approximately 1.8 MB to 147 KB and its real cross-generated
assembly from 172 KB to 68 KB. The corresponding fixed-input check finished
in 29 seconds instead of 230 seconds. These are code-generation/build results,
not runtime-performance claims. The bit-move model covers six storage dtypes,
three SVLs, empty shapes, both tails and padded pitches; native ARM validation
is still needed for these new packing paths.

ARM phase-seven job 1818543 on cn00002 completed its hardware/register/state
suite with 168 passes and one AMX platform skip. The frozen working directory,
source/SDK/native hashes and compiler environment were recorded. Shared-home
quota prevented the stage marker and result archive from being created, so the
paired transpose rank could not advance to timing. The task is not accepted as
a performance gate; its private outputs were retained under
`/tmp/vecops-dsl-library-phase7.BITbt2uF` pending recovery.

Logical tile windows now preserve the producer argument edge and carry their
starts as scalar argument edges, so normal capture freezing applies. They
reuse view clipping, retain kept-axis coordinates, and lower through ordinary
vector memory/conversion operations rather than an operator-specific emitter.
They currently admit unit-step slices and mixed scalar/sliced axes; entirely
scalar extraction remains `element_at` and stepped indices remain `gather`.
This enables DSL algorithms to name a region of an elementwise/helper result
without requiring an intermediate tensor, while retaining historical-load
and overlap handling. It does not yet remove `_matmul` or its cache driver.

The phase-fourteen fixed x86 source/test/SDK snapshot completed with 252 passes
and 28 architecture skips, including the real ARM cross-codegen checks.
The separate phase-fifteen window snapshot completed all eight native/reference
tests, including changed scalar indices, historical reads and aliased outputs.
The checkpoint matmul A/B run completed 48 supported BF16/I8 cases over 24
balanced rounds (3456 unfiltered samples); median new/old was 1.000176, range
0.985058–1.016215. These small-problem cases do not exercise the experimental
large dynamic cache branch. Source extraction into the indexing frontend is
mechanical and keeps the same view/window semantics.

### Vector matrix leaves through ordinary control/state IR

`_library/matmul.py::vector_tile_dot` expresses an outer-product microtile
using logical input windows, explicit casts, an ordered serial K loop, FMA
and carried tile state. It does not emit C++, call `ops::Matmul`, or create
an opaque `dot` node. Input and output helper/elementwise DAGs stay visible.
The CPU recipe adapter only chooses legal geometry and expands this function
before the ordinary loop, fusion, vector and state passes. The experimental
`cpu_vector_dot_dsl` option remains off by default; it currently selects only
explicit unblocked vector dots with known small output geometry and no packed
inputs. Unknown/large shapes, matrix-instruction families and cache traversal
retain their existing drivers. An unmeasured path is not enabled merely to
make the old implementation disappear.

Two generic lowering improvements support these algorithms:

- Projection reuse shares a read/pure vector operation across complete
  register pieces with identical projected coordinates, tag and active count.
  The proof is local to one operation and traversal; it does not cross state
  updates, writes or partial pieces. Full-piece proof also reaches the final
  memory effects, so complete scalable register stores need no tail predicate.
- A scalar projection through logical geometry and explicit cast boundaries
  resolves the original load's historical snapshot directly. It avoids a
  one-lane fill/get round trip without replacing vector math with scalar libm,
  changing conversion rounding, or speculatively reading masked memory.
  Layout folding must not create a new read for a captured load. Windows made
  after a write still observe the earlier value; mutable starts remain frozen.

The diagnostic options `cpu_state_vector_reuse` and
`cpu_ordered_loop_unroll` are strict booleans, initially off. The former scopes
complete-piece and projection proofs to register state; the latter requests
native Clang/GCC unrolling when a state dependency prevents operation-major
DSL unrolling, retaining whole-iteration order. Enabling the dot recipe sets
their defaults to true without overriding explicit user settings.

Fixed phase-twenty x86 tests passed all 34 primitive/window/vector checks,
including a window created after overwriting its captured source. Phase
twenty-one real SVE cross-codegen at VL 16/64/128 has one B vector load per K,
four A broadcasts, register FMA, no call/spill in the hot loop, and no remaining
CNTB query in the full microtile wrapper. This is not native ARM timing.
The frozen x86 paired recipe/legacy-vector benchmark passed independent
correctness checks before and after all 24 balanced rounds for 24 microtile
cases (FP32/FP16, plain/seed/affine, several slim/full/fringe geometries).
Median recipe/legacy time was 0.811766, range 0.471989–1.017553. These results
do not compare hardware MMA or cache-sized matrix drivers with public C++
Matmul; the small FP16 fringe case is not selected by the initial adapter.

Explicit vector initialization accepts a `T.Vector` descriptor in `zeros`
and `full`. It derives dtype/geometry from that descriptor, with a checked
optional active logical prefix. This removes the need for dummy loads or
implicit `lanes`-shaped tiles when a library algorithm needs register banks.
Scalable carried values retain their tag basis/power at tails, while oversized
virtual geometry still follows the ordinary placement/legalization budget.
Initialization does not perform memory reads; tile-shaped fills are unchanged.

The next experimental leaf, `vector_dot_into`, expresses four-row/one-word
streaming traversal in DSL rather than printing its loops in C++. It supports
dynamic M/N/K, both spatial tails, broadcast seeds, typed input transforms,
and a source-parsed epilogue callback. It is not the default dispatcher and
does not yet replace packed/GEMV/matrix-instruction or cache drivers.
`as_vector` supplies a numeric clip/convert/retag boundary for logical rank-one
producer values; unlike a dummy load, it composes with captured helper DAGs.
Compatible mixed Tile/Vector arithmetic retains explicit geometry independent
of operand order. Scalable carried-state pieces now account for the widest
intermediate/consumer, and physical tag rebinds are canonical, so a widening
conversion cannot silently produce a multiword state object or trigger a
scalar get/set reconstruction solely because equivalent tag spellings differ.

The streaming experiment uses the existing disjointness proof mechanism to
version its whole producer/output region before entering nested loops.
`view_disjoint` can expand captured producer values to their readable memory
views; unknown/effectful memory producers remain rejected. The true branch
can omit repeated loop-level snapshot checks, while overlap retains ordinary
historical copies. The phase-thirty streaming timing run overlapped another
sparsefold workload pinned to CPU 0 and is explicitly excluded as performance
evidence; its unfiltered observations and independent correctness checks were
retained. The new versioned driver needs an uncontested performance rerun.

Fixed phase-twenty-nine tests passed all 28 streaming/vector-fill checks,
including dynamic dimensions, empty shapes, broadcast seeds and source-level
epilogues. Phase thirty passed 17 selected native/real-SVE-cross checks for
tag retention, widening pieces, historical clip/conversion and instruction
shape. The versioned phase-thirty-one driver passed its two pure-region tests
and native A/C-overlap test on one low-priority CPU. The broader phase-twenty-
six run had 125 passes and one outdated exact-pipeline expectation; that test
was updated for the registered recipe pass and independently passed afterward.
These scoped gates do not satisfy full catalog coverage, ARM native parity,
or the final production-LOC decrease requirement. Old family/cache/packing
drivers remain until those remaining acceptance gates are actually passed.

### Public-main integration and resumed standalone ARM gates

Public main `f8500b9` was merged with its MIT/SPDX and third-party provenance
materials, without restoring the removed legacy Python binding. Reviewed
project-owned DSL additions carry SPDX notices; full notices are installed
with both CMake and Python SDKs and Python wheel metadata. The eight license
materials in the built wheel were checked byte-for-byte against their sources.
These notice edits do not relicense imported portions or user kernel logic.

The fixed post-merge x86 snapshot passed 221 native tests. Its eight paired
normalization cases had new/old ratios 0.995173–1.002301, and twenty paired
transpose cases 0.998060–1.002169 over 24 balanced rounds. Generated sources
for those cases and 32 MatmulBench metadata combinations differed only by
the added license prefix. This is scoped merge validation, not whole-catalog
performance acceptance for the experimental migration paths.

ARM work moved to authorized `920f-4` in isolated directories, preserving its
dirty pre-existing checkout and af3 installation. Its current process reports
ordinary VL 64B and streaming VL 32B; both remain part of environment/cache
identity, not a common-width assumption. A host clock at year 2000 initially
caused Ninja regeneration failures. The user corrected the clock; both source
SDK Makefiles and subsequent Ninja smoke tests passed two native state tests.
Broader source/installed SDK suites and packing/transpose performance gates
are still in progress; no default SME recipe switch follows from a smoke.

Streaming-vector measurements exposed unnecessary lane reconstruction when
tag aliases and dynamic logical prefixes met generic tile writeback. Physical
tag geometry is now shared across state and ordinary vector traversal aliases.
Singleton projections do not advance their source lane. Exact unit-step SIMD
stores with known nonzero stride are proved injective without an unchecked
user hint; arbitrary indices and unknown/overlapping strides retain fallback.
The library uses those ordinary SIMD primitives, not a special Matmul printer.
Some fused/fringe cases still regress, so its production dispatch remains off.
A tuple-bank row specialization worsened fused-input cases and was reverted;
its exploratory outputs are explicitly excluded as acceptance evidence.

Integer metadata canonicalization retains unchanged DAG nodes and memoizes
within one traversal, holding original nodes strongly to prevent recycled-id
collisions. Representative packing C++ stayed byte-identical and the 22
model/bit-preservation checks passed. One profiled lowering changed from
approximately 4.1s to 3.0s; that measures compile-time work, not kernel latency.

The resumed ARM suite exposed unbounded directory descriptors in the native
dnotify cache fallback. Watches now consume at most 128 descriptors and at
most one eighth of the current soft FD limit; unwatched paths still hash
their actual contents. The loaded extension passed the same 64-FD,
200-directory mutation/hash regression on both local x86 and `920f-4`.
Failure-log I/O errors also preserve the original diagnostic on stderr.

Nonmatrix kernels no longer stage unrelated matrix packing permutations.
Products share the prepared networks instead of retaining an unused duplicate
in every family plan. The scoped planner change left generated C++ for sixteen
MatmulBench metadata combinations and one SME transpose case byte-identical;
433 register/source-catalog checks and 24 normalization/vector-driver checks
passed. This removes compile-time work, not a matrix algorithm or a native
performance gate. Whole-driver SME packing and three-way ARM timings remain
the acceptance gates for deleting their old terminal algorithm printers.

Frozen-source transpose/matmul/packing benchmarks separate template declaration
from compilation. Replay compiles the supplied C++ variants and the public C++
reference, without lowering a third, unused DSL implementation. Source emission
lowers only the selected DSL variant. Existing benchmark `kernels` wrappers
retain their interface; their declaration factories are also tested without
calling `T.compile`. The same sixteen matrix and one SME transpose source
comparisons stayed byte-identical after this benchmark-only change.

Later fresh `920f-4` processes reported ordinary/streaming lengths 64B/64B,
while the earlier still-running process retained 64B/32B. Register-refactor
source catalogs now record the compiler's actual vector state and replay
rejects missing/mismatched geometry before compilation. Previously emitted
catalogs without this record are not performance-acceptance evidence for a
new process; regenerate them rather than infer a width from the host model.

The rebuilt extension's scoped installed-SDK ARM run at VL/SVL 64B/32B
finished with 94 passes and 9 optional cross-codegen skips, without failures.
It includes six raw packing storage types on both sides, exact empty/full/tail
bits and output canaries, vector fills/driver and iteration legality. Extended
negative-spatial-stride and aliased-output packing checks are a separate native
gate still in progress at 64B/64B; do not count its partial run as complete.

SME owner binding now preserves exact IR integer constants inside a possibly
outlined callback rather than reloading them through reference captures.
Packing strides are restored only when every possible alias snapshot has the
same stride, or when that owner proves the differently typed snapshot inactive.
Runtime pitches and alias-dependent negative-to-dense pitches keep their
callback-local runtime values. Pure hardware/transpose checks passed (82 plus
three owner-binding checks); new native timing/coverage remains required.

The earlier merged installed-SDK ARM snapshot completed 317 tests with 22
optional skips; the extended reversed/aliased raw packing suite completed all
12 dtype/side combinations at VL/SVL 64B/64B. These validate those frozen
snapshots, not every subsequent codegen experiment.

Matched-geometry three-way timing found a real experimental raw-packing gap:
64x256 BF16 A packing took about 2.14x its historical DSL checkpoint (24
balanced rounds, independent layout oracle). Its assembly retained repeated
output clamps/predicates. The prepared library boundary now uses the required
packed extent, after validating the caller's full physical output capacity.
Known geometry reconstructs that checked extent inside the owner using
quotient/remainder ceilings; dynamic geometry retains the validated captured
value. A later paired run for the same metadata measured about 1.40x the
checkpoint, still too slow to enable or delete the old driver. These are
separate paired runs, not a same-round estimate of the two experiments.

Empty raw-packing domains skip panel traversal, including an enormous M with
K=0. Group ceilings avoid pre-division signed overflow. Model checks include
zero storage with the other extent at int64.max and retain output canaries;
the 25 model/structure/bit checks pass. Broader native/performance gates for
this latest logical-bound change remain outstanding. No experimental packing
or transpose recipe became the production default.
