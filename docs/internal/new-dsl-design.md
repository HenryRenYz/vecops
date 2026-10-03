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

SME 简单加载转置使用 ZA horizontal write / vertical read，再执行边界
转换。沿用原 C++ 已测量的常量 shape/dtype SME 选择及双行/双列流水阈值；
没有 ARM 实机，不能将这些阈值声称为本轮的新性能优化结果。显式 SME
要求目标 SME-FA64，automatic 在不满足条件时走 vector/gather。普通固定
VL 和固定 SVL 不匹配时，automatic 不进入 SME；显式路径编译诊断拒绝。

公开 C++ `vecops/vec/{SME,AMX}.h` 为原 details 中的指令封装提供稳定入口，
`vecops/execution/Resources.h` 提供词法 state/configuration requirements。
SME manual owner 内部禁止异常、分配和未内联调用；强制内联 noexcept
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
SME DSL 编译单元自动选择
`VECOPS_SVE_INLINE_MATH_SPECIALS=1`，强制内联 vector-only exp/log/rcp/rsqrt
特殊值路径，保持精度与特殊值语义；普通 SVE 编译仍保留 cold outlining。
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
