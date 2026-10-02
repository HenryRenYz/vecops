# Python kernel DSL design

## Status and implementation boundary

This document specifies the new source-less `vecops.language.template` frontend.
The source-backed frontend, ABI, specialization rules, batch compilation, Torch
bridge, and execution graph remain supported. Outer `T.jit` annotations remain
ordinary Python hints; actual values determine immutable normalized keys.

The first implementation covers structured control flow, tensor/tile views,
tagged virtual vectors, arithmetic/comparison/math/reductions, converting memory
accesses, and CPU C++ code generation. Layer normalization and softmax are written
in this DSL, not lowered to `vecops::ops`. MMA, hardware matrix accumulators,
parallel scheduling, instruction scheduling and an exact register allocator are
later milestones. Transposed/noncontiguous accesses use gather/scatter first;
specialized transpose-kernel selection is explicitly TODO.

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
flow. Logical grids are not synonymous with OS threads. A later `T.parallel`
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
consumer. Scalar loop-carried reductions are explicit. Dedicated fixed small
vector expansion and cost-driven scratch/recompute choices can be added without
changing language semantics.

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
| `serial/grid` | 结构化循环、逻辑 program 网格 | 本阶段是串行循环；不是线程数，也不引入 GPU barrier 假设 |
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

当前标量有可变局部变量；tile/向量是不可变计算值。动态 if 的跨分支标量须
提前初始化，动态循环跨迭代 tile/向量状态须写入显式 `T.alloc` buffer。
尚未实现 native vector 的循环携带 SSA/phi 与寄存器驻留证明。非法 rebinding
必须报错，不能悄悄采用最后一次编译遍历的表达式。`T.unroll` 仅支持常量界限，
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
一次编译冻结 kernel 与 vector pipeline 的注册表快照，批量请求相互独立。
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
寄存器分配、parallel、MMA、任意 indexed-memory、专用 transpose 仍未实现。

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
