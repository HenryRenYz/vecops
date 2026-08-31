# Matmul 优化与实验记录

本文是 AMX/SME Matmul、MatmulPack 及融合 DataAccess 路径的长期实验账本。
它不是只描述最终实现：成功、失败、噪声来源、编译器行为和待验证假设都必须记录，
以免后续重复试错或只对少数 benchmark 过拟合。

## 记录规则

每次性能相关修改至少记录：

1. 实验编号、日期、代码差异或 commit；
2. 主机、CPU、SVL、编译器及完整关键编译选项；
3. benchmark 二进制、filter、shape、dtype、packing/fusion mode；
4. ABBA 运行顺序、重复次数、单次最短测量时间和 CPU 绑定；
5. wall time，以及可用时的 cycles、instructions、IPC、frontend/backend stall、
   L1D/L2D refill 和 branch miss；
6. 关键热循环的反汇编变化、符号大小和整个 `.text` 大小；
7. 结论：保留、拒绝或继续实验，并写明原因。

默认性能判定协议：固定 CPU，先 warmup，基线/候选采用 ABBA 交错，至少 9 次重复，
每次目标 case 尽量测量 1 秒以上。单项超过 2% 的稳定回退默认拒绝；短 tail case
必须结合 cycles 和反汇编判断，不能只看几十纳秒级 wall time。

构建注意：同一个 CMake build tree 有待自动 reconfigure 时，不要并发启动两个
`cmake --build`。实测并发 configure 会让其中一个进程看到暂时不完整的 Makefile，
并可能使自动探测的 fixed-SVL cache 变空。远端固定 SVL 验证统一显式传
`-DVECOPS_FIXED_STREAMING_SVE_BITS=512`，然后在同一 build invocation 中构建目标。

## 测试环境

### ARM SME

- 主机：`920f-4`
- 工程：`/home/ryz/vecops-neo`
- 构建目录：`cmake-build-sme-pack-arm-clang-release`
- CPU：many-core ARM，SME/SME FA64，SVL=512 bit
- 编译器：BiSheng Clang 19.1.7
- 关键选项：`-O3 -march=native+bf16+sme+sme-fa64+sme-f64f64`
- kernel benchmark 额外选项：`-falign-functions=64 -falign-loops=64`

### x86 AMX

- 主机：本地 4th Gen Intel Xeon
- ISA：AVX-512、AMX BF16/INT8，工具链允许时包含 AMX FP16
- GCC 构建目录按实验分别使用 `cmake-build-matmul-*-gcc-*`

## 覆盖目标

正确性测试最终应完整覆盖，性能 benchmark 则采用正交代表集以控制代码体积：

- 所有原生 Atom：BF16、FP16、FP32、条件可用的 FP64、四种 INT8 signedness；
- memory→compute→memory：至少包括
  - FP32→BF16 kernel→FP32；
  - FP32→FP64 kernel→FP32；
  - BF16→FP32 kernel→BF16；
  - FP16→FP32 kernel→FP16；
- zero/accumulate C prologue；
- output convert、scalar scale、ReLU、clamp、sigmoid、bias、residual；
- INT8 对称量化、反量化、ReLU+requant，以及后续非对称 zero-point correction；
- RawDirect、RawAutoPack、PackedA、PackedB、PackedAB、online packing 和摊销权重 packing；
- full/tail M/N/K、KPack 边界、auto-pack 阈值、cache working-set、decode、prefill、
  MLP、attention、卷积 lowering、极端 tall/wide/skinny-K 和 batch。

## 已保留实现

### MATMUL-SME-001：ZA32 转置 packing 与双 tile 交替

- 文件：`include/vecops/kernel/details/matmul_pack/sme/Pack.h`
- 方法：full word panel 通过 ZA horizontal load / vertical store 完成转置；
  ZA0/1 与 ZA2/3 交替 charge/drain，并对远端 K chunk 做预取。
- 特性：避免 gather、read-ZA+ZIP+store；BF16/FP16/INT8/FP32 都按 32-bit
  MOPA group 统一处理。
- 状态：保留。后续仍需测试将 parity/prefetch 分支改成固定双 chunk 主循环。

### MATMUL-SME-002：packed 2×2 K loop 64B 对齐

- 文件：`include/vecops/kernel/details/matmul/sme/Backend.h`
- 最终热循环：2 个地址乘法、4 个 `ld1b`、4 个 `bfmopa`、4 个 ZA tile。
- 循环大小接近一个 I-cache line；显式 64B loop alignment 避免跨两行。
- 状态：保留。

### MATMUL-SME-003：K full group 与单独 tail

- BF16 的完整 K groups 使用 `FullK=true`，最多最后一个 group 使用 tail path；
  `KPack==1` 在编译期完全删除 tail 分支。
- 状态：保留。FP16/INT8 尚需推广并做汇编门禁。

### MATMUL-SME-004：大 raw Matmul 自动复用 MatmulPack

- 文件：`include/vecops/ops/Matmul.h`
- 双 raw 阈值：`M*N*K >= 64K`；单 raw 阈值：`>=128K`；同时要求
  `M*N>=128`。
- 仅用于 rank-2、inner stride=1、NoTransform、memory=compute=native element、
  元素不超过 4 bytes 的 SME 输入。
- 选择在 operation 构造时缓存；执行时只保留布尔分支。
- on-the-fly 与 offline packing 共用 `matmul_pack`，没有复制打包逻辑。

920f-4，BF16，包含 on-the-fly packing 成本：

| Shape | Raw 基线→最终 | Raw 提升 | PackedB 基线→最终 | PackedB 提升 |
|---|---:|---:|---:|---:|
| 1×1152×896 | 464.4→262.6 us | 43.5% | 186.5→107.1 us | 42.6% |
| 128×1024×768 | 1807.6→554.6 us | 69.3% | 932.2→443.8 us | 52.4% |
| 256×256×256 | 233.8→72.5 us | 69.0% | 147.2→58.9 us | 60.0% |
| 127×1025×769 | 2558.7→552.7 us | 78.4% | 979.3→439.1 us | 55.2% |

调阈值期间 FP16、FP32、INT8 large raw 分别约提升 53%、59%、56%。
最终 benchmark `.text` 从 569,364 增至 576,084 bytes（+1.18%）。

## 已拒绝或未合入实验

### MATMUL-SME-X01：packed 2×2 C++ 四路 K 展开

- `.text` 增加 1,920 bytes。
- packed-ab 大多持平，但 square 回退约 1.33%。
- 结论：拒绝；寄存器压力/代码体积增加，没有形成稳定的跨迭代流水。

### MATMUL-SME-X02：单 group 显式 preload 软件流水

- `.text` 减少约 640 bytes。
- batch packed-ab 改善约 2.43%，但 square 回退约 8.53%。
- 结论：拒绝；单 group preload 不等价于 KleidiAI/AtomGit 的双 register-bank
  覆盖式流水。

### MATMUL-SME-X03：byte-offset induction 替换 K loop 的两个乘法

- `.text` 减少约 128 bytes。
- batch/ragged packed-ab 回退约 10%，square 回退约 3%。
- 结论：拒绝；地址依赖链比独立乘法更不利。

### MATMUL-SME-X04：旧路径看似回退的代码布局噪声

- 增加临时 benchmark 注册后，未改动的 packed-ab tail 一度显示 8%–17% 回退。
- 移除实验注册后差异缩小到约 ±2%；基线/候选热函数大小均为 `0x201c`，
  机器指令逐条相同，仅页内地址不同。
- square packed-ab 聚焦复测：41.1066→41.1054 us；`perf stat -r 5`
  cycles 209.08M→208.28M。
- 结论：大型模板 benchmark 必须拆 binary/shard；不能把地址相位或 DVFS 波动
  误判为算法回退。

## 公开实现与可迁移技巧

### AtomGit Kunpeng HPC PR

- PR 7/8：线程级 `smstart/smstop`、K×2 展开、FP32→FP64 转换与 MOPA 交错；
- PR 10/12：32×16 tile、K×4 展开、256×64 macro blocking、interior/edge 分离；
- PR 11：`kernel_edge<N_RB,N_CG>` 编译期删除无效 MOPA；
- PR 14：首次计算时转换并缓存 B panel，后续 tile 复用；
- PR 15/16：最清晰的双 register-bank SME1 流水，load/convert 下一组与当前组
  8 个 MOPA 交错；
- PR 18：按 768KiB/core L2 设计 blocking、prefetch 和 non-temporal store；
  其中依赖 benchmark 分配模式提前处理未来输入的投机逻辑严禁迁移。

入口：<https://atomgit.com/kunpengcompute/HPC-competitions/pulls>

### OpenBLAS / KleidiAI / ACL / oneDNN

- OpenBLAS direct FP32 SME：2VL×2VL，循环外 predicate，K loop 双缓冲，
  `addvl` 地址递增；packed kernel 反而保持简单 4-load/4-MOPA。
- KleidiAI SME1：precharge 4 K groups，再把下一轮 load 均匀穿插到 16 个 MOPA；
  K main/tail/oddments 分离；packing 使用 ZA charge/drain。
- Arm Compute Library：M<=VL 或极宽 N 选 1VL×4VL，N<=VL 选 4VL×1VL，
  其余 2VL×2VL，M==1 使用独立 GEMV；这是当前最值得优先引入的 shape-aware
  dispatcher。
- oneDNN SME BRGEMM：ZA0 缓存/转置 raw A，ZA1–ZA3 同时计算最多三个 N tile。
- vLLM Arm CPU 没有独立 SME 微内核，主要复用 oneDNN/ACL；可借鉴 static weight
  packing、NUMA 和 decode/prefill 分流。

## 2026-08-29：覆盖扩展阶段

### MATMUL-COV-001：shape catalog 扩展

- 状态：实现中，尚未完成双架构验证。
- 新增 tile/K 边界、M/N=1、skinny/long K、4K QKV decode/prefill、
  11008×4096 MLP up/down、attention、conv lowering、rank expansion/reduction。
- shape 使用 runtime extent，避免每个 shape 新增一份 kernel 模板实例。

### MATMUL-COV-002：分片 MatmulScenario benchmark

- 状态：实现中，尚未完成编译与正确性验证。
- 独立于原生 Matmul benchmark，12 个编译 shard，防止转换/融合实例改变 packed
  性能 binary 的代码布局。
- 计划/初始实现覆盖：
  - 原生 BF16/FP16/FP32/FP64 和四种 INT8 signedness；
  - FP32 memory→BF16 compute→FP32 output；
  - FP32 memory→FP64 compute→FP32 output；
  - BF16/FP16 memory→FP32 compute→同型 output；
  - accumulate、ReLU、sigmoid、scalar scale、accumulate+ReLU+scale；
  - FP32 input quantize→S8/U8 MOPA→FP32 dequant；
  - INT32 accumulator→ReLU→INT8 requant。

### MATMUL-COV-003：mixed-width zero prologue 的 scalable-SVE 编译失败

- 触发场景：BF16/FP16 memory→FP32 kernel→同型 output，以及
  FP32 memory→FP64 kernel→FP32 output。
- 症状：`make_matmul` 为 C 构造 `ZeroVecTransform<Acc, Memory>`；DataAccess
  把这个完全不读取输入的 transform 送入 scalable-SVE chunk recursion，随后不断
  尝试 `P-3/P-4/...` tag，最终以“subword may contain no lanes”静态断言失败。
- 根因：当 transform input 比 compute output 窄时，相同逻辑 lane 数可能没有可表示的
  sizeless输入 subword；继续二分 output tag 只会让 input tag 更窄。零 prologue 本来就
  不需要 materialize input，进入这条路径没有意义。
- 修复：在 `InputDataAccess::load` 中识别 `ZeroVecTransform`，直接在 compute Tag
  生成零向量，仅应用 inactive-lane population，不做 memory/address/context/convert。
- 状态：已验证。SME Native 的 12-shard scenario binary 完整编译；所有 mixed-width
  tail scenario 通过 reference sampling。本地新增 DataAccess 定向测试也通过。该行为
  同时减少 zero-C prologue 的无效模板实例和潜在运行时工作。

### MATMUL-COV-004：第一批 AMX scenario 基线

- 本地 GCC release 已成功构建 12-shard `MatmulScenarioBench-Native`。
- 注册 45 个 benchmark；所有 tail scenario 通过 benchmark 内置 reference sampling。
- 19×23×257：原生 BF16 约 0.57 us；FP32 memory→BF16 compute 约 10.6 us；
  FP32 quantize→S8/U8 MOPA→FP32 dequant 约 13.4 us。
- 特性：conversion/quantize 当前明显落在 raw per-tile DataAccess 路径，暴露出将
  conversion MatmulPack 接入自动打包的高优先级机会。

### MATMUL-COV-005：第一批 SME scenario 基线

- 920f-4 / BiSheng Clang 19.1.7 / SVL=512，Native scenario binary 注册 57 个
  benchmark；19 个 tail pipeline 均通过 reference sampling。
- tail 中位数（不同 Atom 的 KPack 对应 K 分别为 9/17/33，不能横向当作同工作量）：
  - FP32 native，19×23×9：0.430 us；
  - BF16 native，19×23×17：0.438 us；FP16 native：0.443 us；
  - FP64 native，19×23×9：0.673 us；FP32 memory→FP64→FP32：1.033 us；
  - 四种 INT8 signedness，19×23×33：0.448–0.460 us；
  - FP32 memory→BF16→FP32，19×23×17：0.622 us；
  - BF16/FP16 memory→FP32→同型 output，19×23×9：均约 0.551 us；
  - accumulate 0.539 us，ReLU 0.420 us，scale 0.428 us，sigmoid 1.049 us，
    accumulate+ReLU+scale 0.530 us；
  - FP32 quantize→S8/U8→FP32 dequant，19×23×33：4.050 us；
    ReLU+INT8 requant：4.511 us。
- 初步特性：简单 ReLU/scale 已被融合到 store 路径，成本接近测量噪声；sigmoid 受
  指数近似支配；在线量化的 per-tile transform 是当前最明显的慢路径。

### MATMUL-COV-006：真实 batch、全 mixed packing、bias 与非对称量化

- 状态：完成。
- 纠正旧覆盖：原名为 `scenario/batch/shape:64x256x256` 的 case 只是二维
  `M=64`，并不含 batch 维。新增真正 rank-3 `[B,M,K] x [B,N,K]`：
  - `independent_tail`：B=4，各 batch 的 A/B 均独立，且 M/N/K 都含 tail；
  - `shared_weight_decode`：B=8、M=1，B 使用 batch-stride=0，模拟多个请求共享
    同一权重矩阵。
- 类型/融合：把上述两种真实 batch 注册到所有原生 atom、memory→compute 转换、
  accumulate/ReLU/scale/sigmoid、量化/反量化和输出转换场景；每个 benchmark 在
  计时前验证跨首/中/末 batch 的 6 个位置。
- bias：C prologue 使用 shape `[M,N]`、stride `[0,1]` 的 N 元素 bias；rank-3
  使用 `[B,M,N]`、stride `[0,0,1]`，覆盖 bias 与 bias+ReLU，实际走 Matmul
  prologue/epilogue DataAccess，而非 benchmark 外部相加。
- 非对称量化：使用 U8×S8 atom，A 为 `qA=x*4+3`，B 为 `qB=x*4`；通过按 N
  广播的预计算 correction `-3*sum(qB)` 初始化 C，使内核计算等价于
  `sum((qA-3)*qB)`，epilogue 乘 1/16。该场景明确代表“离线/packing 阶段已有
  B column sum”的现实路径，不宣称覆盖动态 column-sum 生成成本。
- packing：dtype probe 从 raw/packedAB 两端扩为 raw/packedA/packedB/packedAB；
  测试辅助函数也泛化到所有原生浮点 atom 和四种 INT8 signedness。
- 本地 Intel AMX：scenario 45→90 个，其中真实 batch 30 个、bias 10 个、非对称
  量化 5 个，全部通过内置 reference sampling；拆分后的主测试 11/11、batch 2/2、
  mixed packing 1/1。
- 初次把所有新增组合直接放进 `MatmulSMETest.cpp` 后，BiSheng Clang 单 TU 编译约
  8 分钟、峰值 RSS 约 11.7 GiB。功能正确但工程结构不可接受，因此调整为：
  - 原主测试只保留 bias/非对称融合；
  - 现有 `MatmulBatch{AMX,SME}Test` 使用 7/9 个编译 shard，按 atom 分片；
  - 新增 `MatmulMixedPackingTest`，使用 8 个 shard，按 atom 分片，并加入 SME
    disassembly gate。
  本地分片后的 AMX batch 2/2、mixed packing 1/1；各 shard 可并行编译且不再形成
  单个超大 AST/优化单元。
- 注册坑点：仅修改 `register_probe_cases` 不够，`MatmulCase::modes` 仍会二次过滤；
  `dtype_probe_cases` 必须同时从 `kRawAndPackedAB` 改成 `kAllInputModes`，否则
  packed-A/B 看似注册、实际被静默跳过。
- 920f-4 Native：scenario 注册 110 个，其中真实 batch 44、bias 10、非对称量化
  5 个，新增覆盖执行时 reference error=0；主测试 16/16、batch 2/2、mixed packing
  1/1。主 SME benchmark 的 `dtype_coverage` 最终扩为 48 个 case：FP16/FP32/
  代表 INT8/FP64 四个 atom × tail/decode/batch 三种 shape × raw、packed-A、
  packed-B、packed-AB 四种模式，单边 packed 共 24 个。本机不支持 AMX-FP16，
  AMX 主 benchmark 的代表 INT8 dtype probe 为 12 个（3 shape × 4 mode）。
- fixed streaming-SVL=512 主测试 16/16；Native/fixed 主测试 manual-SME
  disassembly gate 均通过。最终全量回归中 batch/mixed 的 Native 与 fixed 功能测试、
  反汇编门禁也全部通过，详见 `MATMUL-REG-001`。

### MATMUL-SME-005：conversion-aware 自动打包

- 状态：保留。第二版通过性能、Native/fixed-SVL 正确性与汇编门禁。
- 假设：large Matmul 中每个 output tile 重复执行 memory→compute 转换远贵于一次
  全矩阵 conversion pack；现有 MatmulPack 已有 FP32→BF16 `SMEPostprocess` 和
  FP16→FP32 `SMEStagedFP16ToFP32`，BF16→FP32 可先使用 Vector pack。
- 修改：把自动打包 eligibility 从 native memory 扩到：
  - FP32 memory→BF16 operand；
  - FP16/BF16 memory→FP32 operand。
- 保持限制：rank-2、inner stride=1、NoTransform、元素不超过 4 bytes；F64、任意
  transform、dynamic inner stride 仍不自动打包。
- 基线 binary：920f-4 `~/MatmulScenarioBench-before-conversion-autopack`，SHA256
  `e9c447f9a8a9cd4b36abb27887ea8b91e2d4337671f6ae902d8f5cee99da0259`。
- 需要验证：tail 不触发阈值；decode/batch 分别比较三种 conversion pipeline；
  native 和 fused output case 不回退；检查实际 selector 命中预期 MatmulPack backend。

第一轮 ABBA（4 组，CPU 8）：

| Pipeline | Shape | 基线→候选 | 变化 |
|---|---:|---:|---:|
| FP32→BF16→FP32 | 1×1024×1024 | 1344.2→781.4 us | -41.87% |
| FP32→BF16→FP32 | 64×256×256 | 160.4→40.5 us | -74.76% |
| FP16→FP32→FP16 | 1×1024×1024 | 1341.9→578.6 us | -56.88% |
| FP16→FP32→FP16 | 64×256×256 | 138.1→39.1 us | -71.66% |
| BF16→FP32→BF16 | 1×1024×1024 | 1357.5→1472.1 us | **+8.45%** |
| BF16→FP32→BF16 | 64×256×256 | 141.2→93.2 us | -34.00% |

BF16→FP32 decode 回退原因：该路径暂时只有通用 Vector pack，B 从 2 bytes 扩到
4 bytes 后只被 M=1 使用一次，额外 packing 和 packed-B 带宽超过 compute 收益；
batch 中每个 B row 被 64 个 M row 复用，因此仍有 34% 收益。第二版为每个
BF16-widen operand 增加对侧空间复用门槛 4：A 要求 N>=4，B 要求 M>=4。
当前 auto-pack plan 仍是 A/B 全开或全关，所以 M=1 时回退到完整 direct path；后续
改成 per-side plan 后可以只 pack 高复用的 A。

加入复用门槛后的第二轮 ABBA：

| Pipeline | Shape | 基线→候选 | 变化 |
|---|---:|---:|---:|
| FP32→BF16→FP32 | 1×1024×1024 | 1355.3→781.2 us | -42.36% |
| FP32→BF16→FP32 | 64×256×256 | 160.6→41.4 us | -74.19% |
| FP16→FP32→FP16 | 1×1024×1024 | 1339.3→575.5 us | -57.03% |
| FP16→FP32→FP16 | 64×256×256 | 138.2→40.9 us | -70.39% |
| BF16→FP32→BF16 | 1×1024×1024 | 1360.2→1336.0 us | -1.78%（未打包） |
| BF16→FP32→BF16 | 64×256×256 | 141.3→95.5 us | -32.45% |

19 个 tail pipeline 的配对变化中位数为 -0.14%，未触发 auto-pack 的普通路径最差
为 +1.94%，处于当前短测噪声门槛内；没有发现系统性回退。量化 transform 不满足
NoTransform eligibility，观测到的较大正负变化属于代码布局，不能归因于本候选。
scenario binary `.text` 从 1,172,732 增至 1,202,428 bytes（+29,696，+2.53%）；
新增体积来自三类 conversion auto-pack orchestration/packing 实例，因此继续保持独立
scenario binary，避免扰动原生 packed 性能 benchmark。

最终验证：

- 920f-4 Native `MatmulSMETest`：14/14；
- 920f-4 fixed streaming-SVL=512 `MatmulSMETest`：14/14；
- Native/fixed 两项 manual-SME disassembly gate：2/2；
- 本地 GCC `MatmulAMXTest`：9/9；
- 本地/920f-4 `TensorDataAccessTest`：32/32（含 death tests）和 30/30；
- large conversion test 明确断言 `required_workspace()>0`，覆盖 FP32→BF16、
  FP16→FP32 和 BF16→FP32 的实际 on-the-fly packing 路径。

### MATMUL-SME-006：FP16/INT8 完整 K group 与唯一 tail 分离

- 状态：第一版拒绝；FP16-only 第二版保留。
- 假设：旧实现仅为 `KP==1` 和 BF16 atom 实例化 `FullK=true` 主循环；FP16 与
  四种 INT8 signedness 的所有 K group 均走 `FullK=false`，使 predicate/bounds
  处理留在每次迭代中。应统一执行 `floor(K/KPack)` 个编译期完整 group，再至多执行
  一个编译期非完整 group。
- 修改：`compute_generic` 对所有 atom 使用相同的 full-groups + optional-tail 结构；
  `KPack==1` 的 FP32/FP64 由 `if constexpr` 消除 tail 代码。
- 基线：920f-4，CPU 8，BiSheng Clang 19.1.7，Native SVL=512；binary
  `~/MatmulScenarioBench-before-full-k-all-atoms`，SHA256
  `ef104dc65549beafa23c2f4270fd6b3f0f971d3edab943bcda6a583471d739c6`。
- 测量范围：FP16 与四种原生 INT8 signedness 的 tail/decode/batch，并观察两个
  quantized fused pipeline；先完成基线 A，候选构建后做 B-A-B-A 配对。

第一版 ABBA 结果（3 个 baseline run 与 2 个 candidate run 的中位数）：

| 场景 | 变化 |
|---|---:|
| FP16 tail 19×23×17 | -5.50% |
| FP16 decode 1×1024×1024 | +0.39% |
| FP16 batch 64×256×256 | -0.02% |
| 四种原生 INT8 tail 19×23×33 | **+60.69%～+62.81%** |
| 四种原生 INT8 decode | -1.00%～+1.62% |
| 四种原生 INT8 batch | -0.76%～+0.24% |

拒绝原因不是 MOPA 本身：每个 INT8 raw/direct operation 同时实例化
`FullK=true` 主循环和 `FullK=false` tail 后，编译器无法在 Tile2D catalog 的多个
tile plan 间共享 direct fast path 与 generic fallback。四个 native INT8 operation 的
对应 streaming-ZA 符号分别从 25,984 bytes 增至 77,236 bytes（各 +51,252，约
2.97 倍）；另两个量化 operation 分别增加 28,896/28,708 bytes。scenario `.text`
从 1,326,031 增至 1,593,831 bytes（+267,800，+20.20%）。小矩阵只执行一个
空间 tile，新增前端/控制流代价无法摊销，因而出现稳定 61% 回退；大 K 中旧版的
runtime `k+KPack<=K` 本就会进入 direct word-gather fast path，省掉该分支没有可测
收益。量化融合路径观测到 -3%～-16% 不能和原生 INT8 的相反结果自洽，判定为代码
布局变化，不作为保留依据。

第二版仅让 FP16 使用 full-groups + unique-tail；INT8 恢复原有单实例循环。FP16
对应符号第一版只增加 3,700 bytes，且 tail 有 5.5% 初步收益，值得单独复测。

第二版 ABBA（A-B-A-B-A-B，CPU 8，三组各取中位数）：

| FP16 场景 | 基线→候选 | 变化 |
|---|---:|---:|
| tail 19×23×17 | 0.4534→0.4321 us | **-4.71%** |
| decode 1×1024×1024 | 280.7342→281.4094 us | +0.24% |
| batch 64×256×256 | 16.8420→16.8421 us | +0.00% |

decode/batch 会触发自动打包并进入 `compute_packed_groups`，因此不受
`compute_generic` 改动影响，+0.24% 属于测量噪声；raw tail 是目标路径，4.71%
收益跨三次交错测量稳定。最终 `.text` 仅从 1,326,031 增至 1,329,743 bytes
（+3,712，+0.28%），candidate SHA256 为
`4a834d118ee6ad237549136ab1cb31cdb5ef1f1235b5fe3b33b098aa101f51cc`。

最终门禁（920f-4，`~/vecops-neo/cmake-build-sme-pack-arm-clang-release`）：

- `MatmulSMETest-Native`：14/14；
- `MatmulSMETest-NativeFixedStreamingSVE`：14/14；
- Native/fixed-SVL manual-SME disassembly gate：2/2（14.33 s / 14.75 s）。

结论：保留 FP16-only 分组；明确不把同一模板策略推广到 INT8。INT8 后续若要消除
完整 K group 判断，应在 direct loader 内用单份循环/运行时 fast-path hoist 实现，不能
再复制整个 Tile2D catalog。

### MATMUL-SME-007：hoist direct gather 的 stride 不变量

- 状态：保留。
- 基线反汇编：920f-4 Native 原生 S8×S8 raw operation 位于 `0x4bac0`，大小
  25,984 bytes；其中一个 K loop 的回边为 `0x4bffc -> 0x4bc5c`。精查后需要修正
  初步判断：Clang 已把 `tst row_bytes` eligibility 检查提升到每个 tile plan 的循环
  外，但因为 stride/accessor 的别名关系，仍在 loop 内装载 stride 字段并构造 gather
  index/offset，且 direct/fallback 控制流较分散；不是所有不变量都未提升。
- 假设：`row_bytes = stride[0] * sizeof(T)` 与 `gather_offsets_fit` 对每个 operand、
  每个 microkernel 都是循环不变量；预计算后传给 `load_operand`，可以减少 K loop
  内的标量乘法和范围比较，同时不复制 Tile2D catalog。
- 修改：增加轻量 `OperandInvariants<Atom, Side, Source>`，在 microkernel 入口分别为
  A/B 计算 `row_bytes` 与 int32 offset eligibility；所有 `compute_group` 复用该状态。
  未改变 K-tail、空间 mask、DataAccess fallback 或 packed layout 的语义。
- 基线 binary：`~/MatmulSMEBench-before-loader-invariants`，SHA256
  `850e7df565a5a7fee5925e461305bb5e8992d1ee32dc140e58693746bfbb3fae`；
  scenario 基线为 `~/MatmulScenarioBench-before-loader-invariants`，SHA256
  `4a834d118ee6ad237549136ab1cb31cdb5ef1f1235b5fe3b33b098aa101f51cc`。
- 测量范围：主 BF16 benchmark 的全部 tail/shape_coverage raw case，以及 scenario
  中 FP32/BF16/FP16/F64/四种 INT8 和 transform/fused tail；检查汇编、符号大小与
  Native/fixed-SVL 正确性。

主 BF16 raw ABBA（3A/2B 中位数，11 个 case）：

- 全部 case 变化中位数 -1.08%；
- `one_by_one` -4.17%，`single_col_tall` -2.39%；
- m/n/mn tail 为 -4.10%～-4.40%，k-tail -1.08%；
- `below_tile` -1.07%，`long_k` +0.00%，`single_row_wide` +0.17%；
- 首轮 `skinny_k` +1.09%，随后用 0.20 s、A-B-A-B-A-B-A-B 单独复测为
  10.7390→10.7633 us（+0.23%），归入噪声而非稳定回退。

全 scenario tail ABBA（19 个 pipeline）：变化中位数 -1.44%。实际使用 direct
invariants 的原生路径均为正向或噪声：FP32 -8.09%，四种原生 INT8 下降
2.44%～4.79%，BF16 convert -2.43%，FP16 -0.50%，FP64 -0.39%。不满足
`direct_row_major_input_v` 的 conversion/quantize transform 理论上不受影响；其中
量化 dequant +2.65%、FP16-memory→FP32-compute +1.78% 与相邻 transform case 的
反向变化归为链接布局噪声，不拿来评价本优化。

候选反汇编中，首个 raw INT8 tile plan 已在 K loop 之前生成 `index z0.s` 并装载
A/B stride，基线对应工作位于 loop 内；S8×S8 raw operation 从 25,984 缩至
25,524 bytes（-460）。scenario `.text` 从 1,329,743 缩至 1,328,079 bytes
（-1,664，-0.13%），SHA256
`fa31aa53180cf707a20296b8a0f8af081d7869dc0c26db89d26334db281f167a`。
主 benchmark SHA256 为
`65702dfe564670cd8a5219afad6cf48fb1a981294b3407170130b5b878b42d9e`；其总 `.text`
对比还包含前一项 FP16-only K-tail 改动，不能把净 +2,880 bytes 全归到本候选。

最终门禁（920f-4）：Native 14/14、fixed-SVL=512 14/14；两项 manual-SME
disassembly gate 2/2（11.42 s / 10.85 s）。结论：保留。这个实验也说明仅看源码中的
runtime `if` 不足以判断循环内成本：必须同时检查编译器 loop versioning 后的汇编；
本次收益来自让剩余 gather index/stride 状态成为显式、无别名的不变量。

### MATMUL-SME-008：融合 transform 热区的跨函数调用消除

- 状态：保留；Native/fixed-SVL 功能、反汇编和性能门禁通过。
- 触发：新增 U8×S8 非对称在线量化测试后，`MatmulSMETest-Native-Disassembly`
  首次发现 Streaming+ZA 区间内存在普通 `bl`。旧覆盖没有实例化这条复杂 transform
  路径，因此不是新增代码引入了原有路径的回退，而是门禁现在观察到了过去未覆盖的
  codegen 缺陷。
- 修复前 binary：920f-4 `~/MatmulSMETest-before-inline-transform-fix`；性能基线：
  `~/MatmulScenarioBench-before-inline-transform-fix` 与
  `~/matmul-transform-inline-before.json`。
- 修复前代表 CPU median：对称量化 FP32 output 的 tail/decode/64×256×256 为
  6.37 us / 4069 us / 688 us；INT8 requant 为 4.49 us / 2937 us / 466 us；
  非对称量化为 6.52 us / 4022 us / 691 us。真实 batch 的 independent-tail/shared-
  weight-decode：对称量化 16.0/967 us，requant 18.0/1138 us，非对称 18.8/1145 us。
- 第一轮门禁列出的调用包括：transform DataAccess 内部回调、request storage 隐式
  构造、`std::clamp<long>`、SVE Interleave builder 和输出 transform store 回调。
  对应修改：
  - `Load/StoreConvertRequest`、Op/Reduce request 与所有 storage field 补显式
    `always_inline` 默认构造；
  - `Wordwise::valid_word_lanes` 和 SVE `set_mask` 改为内联 min/max，避免标准库
    `std::clamp` 被单独 outline；
  - DataAccess 的 axis-stride IILE、SVE Concat/Interleave `construct_words` builder
    补 lambda 级 `always_inline`。
- 第二轮门禁中上述调用全部消失，只剩 `tensor::stride<0/1>(layout)` 被 BiSheng
  Clang outline 数十次。根因是外层 axis-stride lambda 虽已强制内联，但 Layout 的
  自由 `get/size/stride` accessor 本身只有普通 `constexpr`。已给这三个 accessor
  补 `VECOPS_ALWAYS_INLINE`。
- 第三轮结果：Native 16/16，manual-SME disassembly gate 通过，Streaming+ZA
  区间 `bl=0`；fixed streaming-SVL=512 同样 16/16 且门禁通过（13.34 s）。测试
  binary `.text` 884,664→857,560 bytes（-27,104，-3.06%），SHA256
  `fa354627c70b3d3d4ab9984817eac271026414ccf38f9e08b32a710ba1c75d99`。

固定 CPU 8、三轮 A-B 交错、每个 case 内部 repeats=3 后取跨轮 median：

| pipeline | shape | baseline→candidate | 变化 |
|---|---|---:|---:|
| 对称 dequant | tail | 4.731→1.313 us | **-72.24%** |
| 对称 dequant | decode 1×1024×1024 | 2942→1505 us | **-48.86%** |
| 对称 dequant | 64×256×256 | 465.3→179.3 us | **-61.46%** |
| 对称 dequant | real batch independent-tail | 17.905→5.279 us | **-70.52%** |
| 对称 dequant | real batch shared-weight | 1120→435 us | **-61.17%** |
| ReLU+requant | 五种 shape | — | -50.65%～-65.24% |
| 非对称 dequant | 五种 shape | — | -49.22%～-69.20% |

15 个量化 case 全部提升 48.86%～72.24%，变化中位数 -61.46%。原因与汇编证据
一致：修复前 accessor/transform/request helper 位于 tile/K-chunk 重复路径，而不是
每次 Matmul 只调用一次。scenario `.text` 2,391,932→2,234,476 bytes
（-157,456，-6.58%），candidate SHA256
`96f7110fe83f345f1d5cf799965ea7c9565eafb6a91e9205b1be28fb3632c0a3`。

非目标回归：22 个 tail pipeline 的变化中位数 -0.60%；12 个纯 convert/convert
case 为 -1.97%～+3.14%。对首轮正向噪声最大的 BF16 bias/native 和 FP16 output
做七轮 A-B 复测后分别为 -1.39%、-0.82%、+0.47%，最后一项归入噪声。更强的
证据是独立原生 `MatmulSMEBench-Native` 重建前后 SHA256 均为
`77528ad64016a4694d09bd2b39da263a83afced35181fcd7c35a915c1b9e7e1a`，
`.text` 同为 788,628 bytes，即原生 benchmark 的机器码逐字节未改变。

结论：保留。这次也形成一条新的门禁规则：任何新增 SME transform/fusion 测试必须
纳入 manual-SME disassembly target，否则“普通路径无调用”不能外推到复杂模板实例。

### MATMUL-SME-009：native auto-pack 与 Matmul 合并为单 SME region

- 状态：第三版保留；第一版拒绝，第二版收窄。
- 假设：streaming-compatible 的 native pack 和 FP32→BF16 postprocess pack 与
  Matmul 原本分别打开 Streaming+ZA。`ExecutionSession` 已能在类型系统中识别已持有
  的 resource 并消除嵌套进入，因此可在 `execute_auto_packed` 外层打开一次 region，
  让 pack A、pack B、Matmul 连续执行。FP16/BF16→FP32 pack 仍含 ordinary-SVE
  postprocess，明确排除，不能为了减少边界改变其执行模式。
- 基线：`~/MatmulScenarioBench-before-single-sme-region`，SHA256
  `96f7110fe83f345f1d5cf799965ea7c9565eafb6a91e9205b1be28fb3632c0a3`；
  主 BF16 基线 `~/MatmulSMEBench-before-inline-transform-fix`，SHA256
  `77528ad64016a4694d09bd2b39da263a83afced35181fcd7c35a915c1b9e7e1a`。

第一版对任意至少一侧 auto-pack 的兼容路径启用单 region。scenario 静态 `smstart`
点 78→63，plain decode/batch 的 24 个 case 变化中位数 -1.11%；代表收益：

| 路径 | decode | 64×256×256 |
|---|---:|---:|
| FP32→BF16 compute | -1.33% | **-13.30%** |
| native FP32 | -0.64% | **-13.80%** |
| native BF16 | -1.02% | -1.57% |
| native FP16 | -0.55% | -1.67% |
| 四种 native INT8 | -0.88%～+0.46% | -2.15%～-3.26% |

但第一版在主 benchmark 中把 raw、packed-A/raw-B、raw-A/packed-B 三类 operation
都实例化成单 region；`.text` 788,628→917,268 bytes（+128,640，+16.31%），
静态 `smstart` 反而 28→40。原因不是 resource framework 失效：为了满足“SME 区间
无普通函数调用”门禁，pack 和 traversal 必须全部内联到同一个 region；基线每个
`execute_auto_packed` 只有约 0.5–0.8 KiB 并调用独立 SME region，而合并后每个
`with_streaming_za` specialization 约 10–13 KiB。单边 packing 只减少一个边界，
不值得复制整份 traversal，故拒绝第一版。

第一版另观测到不满足 auto-pack 的 FP64 decode +6.92%。对应 F64 streaming kernel
符号大小基线/候选完全相同（0x2034 bytes），但链接地址从 `0x61300` 移至
`0x60400`；这是前序 shard 代码增长引发的 I-cache/page-offset 布局敏感性，而不是
F64 算法路径变化。它再次说明 64-byte function/loop alignment 不能消除所有链接布局
噪声；评价候选必须结合目标符号是否改变，并以独立主 benchmark 复核。

第二版只在 A/B 两侧都需要 auto-pack 时合并，一侧已预打包的 mixed operation 保持
原实现。主 benchmark `.text` 增幅降为 788,628→837,588 bytes（+48,960，+6.21%），
静态 `smstart` 为 32；三轮 A-B 交错的 raw 结果：decode -2.41%、batch -1.44%、
ragged -0.58%、square -4.76%、long-K -3.20%、skinny-K -2.93%。不触发 auto-pack
的小 shape/tail 为 -1.25%～+2.47%。Native 16/16 且 disassembly gate 通过，但
主测试单 TU 峰值 RSS 从约 6.6 增至 8.7 GiB，说明全 atom 默认特化的编译成本也过高。

第三版进一步只保留 `SME_BF16F32` 与 `SME_F32F32`：这两类在复用型 shape 上有
4%～14% 的观察收益；FP16 最大仅 1.7%，四种 INT8 signedness 各自复制完整 traversal
而收益仅约 2%～3%。主 benchmark `.text` 降为 813,204 bytes，相对基线仅
+24,576（+3.12%），静态 `smstart` 为 30（基线 28）；scenario `.text`
2,234,476→2,266,540 bytes（+32,064，+1.43%），静态 `smstart` 78→73。

为获得不受 scenario 大量无关实例链接布局影响的严格对照，dtype coverage 从单一
tail 扩为 tail/decode/batch 三种 runtime shape；四个代表 atom × 四种 packing mode
共 48 个 case。新增注册只使主 binary `.text` 增加 512 bytes，没有增加 kernel
模板种类。随后用完全相同 48-case binary 分别关闭/打开第三版单 region：

| atom | tail | decode | batch |
|---|---:|---:|---:|
| F16（未启用） | +0.08% | +0.24% | +0.01% |
| F32（启用） | -3.68% | +0.42% | **-3.99%** |
| INT8（未启用） | -0.02% | -0.79% | +0.18% |
| F64（未启用） | +1.57% | -0.21% | -3.62% |

未启用 atom 的正负变化是链接布局噪声；F32 batch 是干净的目标收益，decode 的
+0.42% 属于噪声。主 BF16 严格对照中，auto-pack/reuse 型 square -2.07%、long-K
-3.08%、skinny-K -2.80%、batch projection -0.83%；decode +0.36%。七轮 K-tail
复测从三轮的 +2.8% 收敛为 +0.60%。最终 Native/fixed streaming-SVL=512 均
16/16，manual-SME disassembly gate 均通过（14.20/15.66 s）；Native test `.text`
857,560→882,688 bytes（+25,128，+2.93%）。结论：保留，并把“无普通调用的单
region 必然以一定模板代码量换取状态边界减少”作为显式设计取舍。

### MATMUL-SME-010：packed 2×2 K-loop 软件流水 / 双 bank

- 状态：默认拒绝；第二版保留为 `VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE`
  编译期实验分支。
- 基线：第三版 single-region + expanded dtype 的
  `~/MatmulSMEBench-v3-expanded-dtype`，`.text` 813,716 bytes。
- perf：packed-AB square 以 CPU 8、50 repeats、`perf record -F 999` 采样，94.08%
  cycles 位于一个 BF16 streaming kernel；原 2×2 K-loop 为连续四条 `ld1b` 后四条
  `bfmopa`，采样主要归因到四条 load（单条最高 62.5%），Clang 没有跨迭代软件
  流水。`perf stat` 整体（含 benchmark 控制）约 1.62 IPC。
- 第一版：预载当前四向量，每轮把下一组四次 load 与当前四次 MOPA 交错。Clang
  确实生成目标交错，但为 loop phi 每轮增加四条 `mov z?.d`；binary `.text` -64 bytes。
  目标子集 square -1.12%、2×2 microkernel -0.34%，其余约 ±0.6%。
- 第二版：显式一次处理两个 K group；第一组四次 load 后，将第二组 load 与第一组
  MOPA 交错，再直接发射第二组四条 MOPA。反汇编是 8 load / 8 MOPA、无 vector
  move、每两组一个回边；`.text` +320 bytes。只跑目标 case 的七轮 A-B 中：square
  **-8.50%**、2×2 microkernel -0.92%、long-K -0.60%、batch +0.22%。
- 拒绝原因：同一进程运行完整 23 个 packed-AB case 时，第一版 square +10.43%，
  第二版 square +4.91%；两版还使未修改的 1×1/1×2/one-by-one case 随代码块布局和
  前序 case cache 状态出现最高约 +9% 的回退。目标指令序列本身已生成，问题不是
  C++ 表达力，而是它被内联进包含完整 Tile2D catalog 的巨大函数后改变了其他 case
  的前端布局；单 case 收益不能外推为全套收益。
- 保留方式：默认仍编译原单-group loop；定义
  `VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE` 可启用第二版双-bank C++ 序列。后续应
  先把 2×2 packed microkernel 隔离成不扰动其他 catalog block 的 streaming-call ABI
  或独立汇编实验，再评估能否在保持“SME 区间无普通调用”约束下默认启用。关闭
  宏后重建的 benchmark 与实验前 SHA256 都为
  `38840e19d0f1890dfc5a46128c80dfdd87fd3b3d61a8ca42250d6b77cf47ecc2`，
  证明实验分支对默认机器码为零影响（之后仅因新增 square packed-A 注册改变 hash）。

### MATMUL-SME-011：mixed packing 的定向 4×1 / 1×4 tile policy

- 状态：拒绝；仅保留新增 benchmark 覆盖，不保留 policy 代码。
- 假设：PackedA+RawB 使用 4×1 可将昂贵 raw-B loader 从每四输出两次降为一次；
  RawA+PackedB 对称使用 1×4。原 `RuntimeExactArea4` 对双轴大于一的内部区域固定
  2×2，只按 shape aspect ratio 调度，不感知哪一侧已打包。
- 第一版：为 Tile2D 增加 vertical/horizontal exact strip policy，mixed-A/B 自动选择
  对应方向。`.text` 813,716→733,140 bytes（-9.90%），因为每种 mixed operation
  只实例化一个 strip family，确实大幅剪枝 catalog；但 36 个 mixed case 中 tail
  回退最高 25.94%，F64 batch packed-A/B 回退 36.22%/45.91%。说明减少 raw load
  次数不能替代对 atom、轴长度和 ZA/MOPA 发射平衡的建模。
- 第二版：仅对 BF16 启用，且 M/N 任一不足四个 base tile 时回退原
  `RuntimeExactArea4`；同时把 representative square 从 weight-reuse modes 扩成
  `kAllInputModes`，补上之前缺失的大 square packed-A 对称覆盖。候选 `.text`
  +11,648 bytes（+1.43%）。已有 packed-B 严格对比：square +4.38%、ragged +2.17%、
  mnk-tail +5.23%，只有 skinny-K -11.42%，目标大 square 本身即回退。
- 结论：2×2 的均衡 ZA/input 复用在当前实现整体优于仅按 raw/packed 侧定向；撤回
  两个新 Tile2D policy 和 SME selector。保留 square `kAllInputModes`，使以后任何
  mixed 策略必须同时通过 packed-A 与 packed-B 的大矩阵门禁。

### MATMUL-REG-001：SME-005～011 与覆盖扩展的最终回归快照

- 日期：2026-08-29。
- ARM 主机/路径：`920f-4:/home/ryz/vecops-neo`；构建目录
  `cmake-build-sme-pack-arm-clang-release`；BiSheng Clang 19.1.7，
  `-O3 -march=native+bf16+sme+sme-fa64+sme-f64f64`，fixed streaming 显式
  `VECOPS_TARGET_FIXED_STREAMING_SVE_BITS=512`。
- 最终构建目标：`MatmulSMEBench-Native`、`MatmulScenarioBench-Native`，以及
  `MatmulSMETest`、`MatmulBatchSMETest`、`MatmulMixedPackingTest` 的 Native 与
  NativeFixedStreamingSVE 两套目标；全部成功构建。主 benchmark `.text` 为
  813,716 bytes；该数值包含最终新增的 square 四种 packing mode 注册。
- benchmark 注册/内置校验：SME scenario 110 个，dtype coverage 48 个，代表
  square 的 raw/packed-A/packed-B/packed-AB 四种模式均存在；主 benchmark 与
  scenario benchmark 的 JSON 均无 error。这里记录的是覆盖与正确性快照，不把一次
  全套运行的混合 wall time 当作性能结论；各候选的 ABBA 结果分别记录在 SME-005～011。
- ARM 功能测试：Native 与 fixed 各自为主测试 16/16、真实 batch 2/2、mixed
  packing 1/1，合计 38/38 通过。
- ARM manual-SME 反汇编门禁：以下 6/6 通过，总 real time 61.76 s：
  `MatmulSMETest` Native/fixed 14.48/12.47 s，`MatmulBatchSMETest` 7.70/7.20 s，
  `MatmulMixedPackingTest` 10.17/9.70 s。门禁确认 Streaming+ZA 区间内没有因新增
  transform、batch 或 mixed packing 模板重新引入普通函数调用。
- x86 本地主机：`/home/renyz/project/vecops-neo`，GCC release build
  `cmake-build-matmul-bench-gcc-release`。AMX scenario 90 个、代表 INT8 dtype
  coverage 12 个，两个 benchmark JSON 均无 error；主测试 11/11、batch 2/2、
  mixed packing 1/1，合计 14/14 通过。该 CPU/工具链未提供 AMX-FP16，因此 FP16
  原生覆盖由 SME 端承担，不伪造不可执行的 AMX case。
- 额外通用回归：本地 `TensorDataAccessTest-Native` 32/32；`git diff --check` 和
  所有本轮修改/新增 C++ 文件的 `clang-format --dry-run --Werror` 通过。
- 结论：本轮保留项在两种 SME streaming 编译模型和 AMX 侧均没有功能/代码生成
  回退；软件流水与定向 strip 的默认启用仍按实测拒绝，不能因 isolated case 有收益
  而绕过全套回归。

### MATMUL-SME-012：广播 bias/correction 单次 load 后重复写 ZA 行

- 状态：拒绝，源码已恢复基线。
- 假设：bias 和 asymmetric zero-point correction 的 C prologue 都具有编译期
  row-stride=0。原 `initialize_c_tile` 对每个 ZA 行发射一次指向同一地址的
  `ld1w {za?h.s[...]}`；候选先 predicated load 到一个 Z 寄存器，再把同一向量写入
  所有 active ZA 行，希望减少重复 L1 load。
- 基线特性：同一 binary 的三次运行中，简单 ReLU/scale 相对 convert 在 tail/decode
  基本为噪声；bias 在 tail 为 +16.95%，decode 为 +8.10%，bias+ReLU 分别为
  +11.13%/+7.90%。这证明广播 prologue 值得检查，但跨 specialization 的 batch 数字
  受链接布局影响，不能直接归因于 transform 指令。
- 二进制：基线 SHA256
  `098c541d6d2622e9b56825598f1434723570c1fce416290e35c93a01efb7d2e4`，`.text`
  2,266,540 bytes；候选 SHA256
  `d1b536c74735ba4d119f5126d504fa91daf0ced149611b983ce3a2a2d29e491b`，`.text`
  2,269,356 bytes（+2,816，+0.12%）。
- 测量：CPU 8，两个固定二进制按 A-B-B-A-A-B-B-A 交错；每次每 case 内含三次
  repetition。覆盖 bias、bias+ReLU、asymmetric correction 的 tail/decode/batch 与
  两种真实 batch，共 15 个 case。变化中位数 -0.23%，范围 -17.97%～+4.24%；目标
  case 并不一致：bias tail +0.22%、bias+ReLU tail +2.14%、bias batch +4.24%，但
  bias+ReLU batch -3.81%。asymmetric 五项为 -0.39%～+0.07%，等同噪声。
- 伪收益识别：shared-weight decode 的 bias/bias+ReLU 显示 -17.78%/-17.97%，但其
  M=1，每 tile 只有一个 active ZA 行，候选在语义上没有任何重复 load 可消除。这是
  新增代码使大 streaming kernel 改变地址/page offset 后的前端布局收益，不能作为
  保留依据。
- 反汇编：全 binary 中 baseline 有 760 条 ZA horizontal `ld1w`，候选为 552；减少
  的 208 条并没有消失，而是变成额外 208 条普通 Z-vector `ld1w`，同时 ZA
  `mov ..., z...` 从 4 增至 212。也就是每个受影响 ZA 行从一条 memory-to-ZA 指令
  变成一条 load 加一条 move，动态指令更多，并解释了代码尺寸增长和目标 case 回退。
- 结论：SME 的 `ld1w {za?h}` 已经是适合 bias 初始化的融合 memory-to-ZA 形式；
  “先缓存一个 Z 向量再广播”在当前硬件/编译器上不划算。后续 bias 优化应研究 ZA
  tile clear 后的向量/外积式广播、跨 output tile 复用或改变 prologue 表示，不能再
  重复这次纯 load-hoist 方案。

### MATMUL-SME-013：用 FP32/FP64 FMOPA 初始化广播 bias ZA tile

- 状态：保留。
- 公开依据：[Arm SME introduction part 2](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction-p2)
  说明一条 `FMOPA` 会更新整个 predicated ZA tile；[Arm ACLE SME intrinsics](https://arm-software.github.io/acle/main/acle.html#sme-instruction-intrinsics)
  给出 non-widening FP32/FP64 `svmopa` 及整数 `ADDHA` 的标准接口。`ADDHA` 只有
  s32/u32（以及条件可用的 s64/u64），不能直接处理 FP32 bias，因此浮点路径采用
  `ones × bias` 的 outer product。
- 修改：`initialize_c_tile` 在 C input 为 direct row-major、row stride 的类型是
  `Const<0>`、accumulator 为 FP32/FP64 时，先 predicated load 一次 bias，生成全 1
  向量，再用已有 `vec::details::sme::mopa<Tile,T,T,T>` 一次初始化 active M×N。
  普通 accumulate、动态/非零 row stride、整数 correction 和非 direct DataAccess
  保持逐行 ZA load/fallback，不增加运行时类型分支。
- 与 SME-012 的本质区别：SME-012 将每行一条 memory-to-ZA 变成每行 `load+mov`；
  本方案每个 output tile 只有一次 Z load 和一次 outer product，完整 16 行不再循环。
  反汇编在 bias kernel 中明确生成 `fmov z0.s,#1.0`、predicated `ld1w z1.s`、
  `fmopa za?.s,...,z0.s,z1.s`，随后直接进入 BF16 `bfmopa` K loop。
- 二进制：scenario 基线 SHA256
  `098c541d6d2622e9b56825598f1434723570c1fce416290e35c93a01efb7d2e4`，`.text`
  2,266,540 bytes；候选 SHA256
  `e49a4f8a2a7afff80fffc39888f1ed7cd1b9bcc254c3fe124c6715a08e6b070a`，`.text`
  2,259,884 bytes（-6,656，-0.29%）。全 binary 静态 FP32 `fmopa` 670→826，ZA
  horizontal `ld1w` 760→604；正好 156 个实例从逐行 loop 收敛为 outer product。
- 目标 ABBA：CPU 8，两个固定二进制按 A-B-B-A-A-B-B-A，单次每 case 三次
  repetition。bias：tail -14.14%、decode -1.29%、M=64 batch -3.54%、真实
  independent-tail -14.47%；bias+ReLU：tail -13.62%、decode -0.04%、batch
  -5.03%、真实 independent-tail -14.47%。shared-weight decode 首轮约 -16%，其中
  M=1 不足以单独证明广播收益，故不把它计入主要结论。
- 全 110-case ABBA：10 个 bias 目标变化中位数 -13.21%，8/10 至少提升 2%；全套
  内 bias+ReLU decode 曾为 +6.38%，将它与 bias decode 及一个未触发分支的可疑 case
  独立做 12 次 A-B-B-A 后分别为 -6.49%、-5.26%，证明全套顺序中的正值是前序
  cache/layout 噪声。100 个未触发分支变化中位数 -0.05%。
- 回退核查：全套中未修改的 F32/INT8 case 曾出现最高 +18% 的正负布局摆动；把最差
  相关的 36 个 decode/batch case 单独复测后中位数 -0.03%，仅 BF16-memory→FP32
  kernel batch 为 +2.90%，再独立 12 次为 +3.85%。该 case 的 streaming 函数地址、
  大小和完整 0x2674 bytes 哈希在基线/候选完全相同；不包含 bias 实例的主
  `MatmulSMEBench-Native` 重建前后 SHA256 也都为
  `75f6d262f6cba5a0e59827613bfe97b1cd3f1d5416d5b4b1856c3e1c937f818a`，`.text`
  均为 813,716 bytes。因此该 +3.85% 来自 scenario `.text` 缩短后只读数据/注册布局
  改变，而不是普通算子机器码回退；保留并显式记录，不能把它误写成 kernel 变化。
- `perf stat -r 10`（完整 benchmark 进程，包含启动开销）：tail bias cycles
  209,537,996→206,325,532（-1.53%），instructions
  298,493,387→290,228,878（-2.77%），branch misses
  1,552,287→944,322（-39.16%）。wall-time 收益大于全进程 counter 变化，是因为
  目标 benchmark 仅占带启动/注册过程的一部分；指令和分支方向仍与删除逐行 loop
  一致。
- 正确性与门禁：scenario 全 110 case 无 error；920f-4 Native/fixed-SVL=512 主测试
  各 16/16；两项 manual-SME disassembly gate 2/2（12.78/12.02 s）。
- 结论：保留。它同时证明针对 ZA 的算法级指令选择比把普通 load hoist 到 Z 寄存器
  更重要。下一步对 int32 broadcast correction 单独评估 `ADDHA`，但不能从浮点
  `FMOPA` 的结果直接外推。

### MATMUL-COV-007：预量化非对称 U8×S8 correction 场景

- 状态：完成，AMX/SME 共用。
- 原覆盖缺口：已有 `asymmetric_quantize_x4_zp3` 在计时区内将 FP32 A/B 在线量化，
  decode 约 1.5 ms，int32 column-sum correction 的 ZA 初始化成本几乎不可见。现实中
  权重通常已经量化，activation 也可能由前序算子产出量化 tensor，因此需要独立的
  prequantized 路径。
- 新增 `InputPipeline::PrequantizedAsymmetricZp3`：memory/compute 都是 U8×S8，A
  的 zero point 为 3；C prologue 按 N 广播预计算的 `-3*sum(B)`，使结果等价于
  `sum((qA-3)*qB)`，output 乘 1/16 转 FP32。输入没有 DataAccess quantize transform，
  correction 和 dequantize 仍在 Matmul 内融合。
- shape：tail、decode、M=64 batch，以及真实 independent-tail B=4、共享权重 decode
  B=8，共 5 个；每项继续使用首/中/末采样 reference 校验。
- 920f-4 无 ADDHA 基线：tail 0.4897 us、decode 140.284 us、M=64 batch 9.468 us、
  real independent-tail 2.075 us、shared decode 70.306 us，5 项无 error。SME scenario
  110→115。
- 本地 AMX：tail（K=257）0.421 us、decode 27.268 us、M=64 batch 5.770 us、real
  independent-tail 2.019 us、shared decode 14.613 us；5 项无 error，scenario
  90→95。AMX 数字仅是覆盖 sanity，主机启用了 frequency scaling，不作为跨架构
  性能结论。
- 工程代价：新 direct U8×S8 + broadcast-int32 C + FP32 output-transform kernel 使
  scenario `.text` 从 SME-013 的 2,259,884 增至 2,371,244 bytes；这是新增可执行
  场景的模板实例成本，不归因于后续 ADDHA 优化。

### MATMUL-SME-014：用 ADDHA 初始化 int32 broadcast correction

- 状态：第二版保留；第一版作为对照记录，不保留第二条实现分支。
- 指令语义：Arm ACLE 的 `svaddha_za32_s32/u32` 将一个 32-bit vector 加到每个 active
  ZA horizontal slice。microkernel 已先执行 `zero {za}`，所以一次 predicated Z
  load 加一次 `ADDHA` 即可把 correction 填入整个 active tile。
- 实现：在 `vec/details/sme/ZA.h` 增加类型安全的 s32/u32 `addha<Tile,T>` wrapper；
  Backend 仅在 direct row-major、编译期 row-stride=`Const<0>`、accumulator 为
  int32/uint32 时使用。反汇编确认候选有 130 个静态 `addha za?.s` 实例；普通
  zero-C/native INT8 不满足 C layout 条件。
- 第一版：所有 active M 直接 ADDHA。scenario 基线 SHA256
  `b3270b956ae54b1c72865f7055dc13ac87cb37a2f6b891be8709e41185f2256f`，`.text`
  2,371,244 bytes；v1 SHA256
  `ce18fff31fb8b2535de99f4cb11a0b442b5437e4cbd95f28ede3df99bc4c33ef`，`.text`
  2,367,084 bytes（-4,160）。预量化 tail -11.02%、M=64 batch -6.38%、real
  independent-tail -12.24%，但 M=1 decode +1.24%、shared decode +0.25%。在线量化
  tail/real-tail -4.77%/-5.15%，其余被 transform 主导。
- 第二版：`active_m==1` 保留原单条 direct `ld1w {za?h.s[...]}`，多行才走
  `ld1w z + addha`。这在 correction kernel 内增加一次运行时判断并复制小段初始化
  block；最终 SHA256
  `4fa5d8fc4fb11b137c72bf61f4b903869dcb6988eebfb05e9d68edc864732c47`，`.text`
  2,372,908 bytes，相对无 ADDHA 基线 +1,664（+0.07%），相对 v1 +5,824。
- v2 固定二进制 ABBA：预量化 tail -11.64%、decode +0.00%、M=64 batch -9.89%、
  real independent-tail -10.93%、shared decode -0.25%；在线量化 tail -4.40%、
  decode +1.29%、batch -0.62%、real independent-tail -4.48%、shared decode
  -1.02%。M=1 的目标回退被消除，故选择 v2 而非代码更小的 v1。
- 全 115-case A-B-B-A：10 个 correction 目标中位数 -2.43%，没有 >2% 回退；105 个
  未触发分支中位数 -0.20%。全套仍出现最高 +19% 的已知前序 cache/链接布局摆动；
  不把它归因于 ADDHA，因为不包含 correction 的 `MatmulSMEBench-Native` 重建后与
  SME-013 基线 SHA256 均为
  `75f6d262f6cba5a0e59827613bfe97b1cd3f1d5416d5b4b1856c3e1c937f818a`，`.text`
  均为 813,716 bytes，即普通主路径机器码逐字节相同。
- `perf stat -r 10`（完整、自动校准 iteration 的 benchmark 进程）预量化 tail：cycles
  209,045,099→207,852,457（-0.57%），branch misses 806,650→92,783；instructions
  345,087,276→345,990,529。由于更快版本会由 Google Benchmark 选择不同 iteration
  数，后三项不能直接换算成单次动态指令；只把 counters 作为方向性辅助，性能结论
  仍以固定二进制 ABBA 的 per-iteration time 为准。
- 正确性/代码生成：SME 115 个 scenario 无 error；本地 AMX 新增 5 项无 error；
  920f-4 Native/fixed-SVL=512 主测试各 16/16，manual-SME gate 2/2
  （12.74/11.92 s）。
- 结论：保留 v2。浮点 bias 的 FMOPA 与整数 correction 的 ADDHA 现已共享同一个
  编译期 broadcast-C 判定框架，但保留各自适合的 ZA 指令；M=1 仍使用融合 ZA load。

### MATMUL-COV-008：F64 MatmulPack 覆盖

- 状态：完成。仅在工具链定义 `HAS_SME_F64F64` 时注册，不增加不支持 SME F64
  平台的实例。
- benchmark 新增 FP64 A/B、Const/Dyn extent，shape 为 17×65 tail、64×256 small、
  513×1000 medium-tail、1025×1027 large-tail，共 16 项；测试新增 A=19×17、
  B=21×9 两个非整 panel direct-pack case。
- 新覆盖首先暴露的不是 F64 Matmul kernel，而是 pack backend 的 element-size gate：
  `sizeof(ComputeType)<=4` 使 F64 静默落到通用 Vector backend。后续 SME-015 将该
  gate 扩到 8 bytes，并用 ZA64 实现真实 SME 路径。

### MATMUL-SME-015：ZA64 transpose packing 与 F64 自动打包

- 状态：保留 ZA64 pack 和 F64 auto-pack；拒绝把 F64 auto-pack/Matmul 合并成单个
  streaming region。
- ZA64 packing：`Bits<T>` 增加 `uint64_t`，panel 改由 `Packing::panel()` 决定，
  8-byte 元素使用 ZA64 horizontal-load/vertical-store 转置。ZA32 的双 bank full-panel
  流水严格限制为 4-byte 元素，避免把 ZA64 错套进 ZA0/1 与 ZA2/3 的四-tile 调度。
- 920f-4、SVL=512、BiSheng Clang 19.1.7，16 个 F64 A/B、Const/Dyn case，各固定
  二进制运行 4 次、每次三 repetition。Vector→ZA64 的变化范围为 -68.55%～-39.25%，
  中位数 -55.67%；A/B 结果对称：

| Shape | Const | Dyn |
|---|---:|---:|
| 17×65 | 1.493→0.677 us（-54.62%） | 2.155→0.678 us（-68.55%） |
| 64×256 | 15.140→7.305 us（-51.75%） | 21.906→7.292 us（-66.71%） |
| 513×1000 | 424.151→257.682 us（-39.25%） | 624.367→257.427 us（-58.77%） |
| 1025×1027 | 1040.744→622.564 us（-40.18%） | 1440.691→622.755 us（-56.77%） |

  表中列 A，B 的对应差异不超过约 0.2%。pack benchmark SHA256 从
  `3965286959147119e4cd565c47d098b30af8dc4af1c43dbb8005b4ac00f7c51d`
  变为 `0e0577e3e8fc9ca1a0bbd4b9a62ed120e9f6db30a959f4f213edd40cb02fba4f`；
  `.text` 1,136,624→1,081,048 bytes（-55,576），即性能和代码体积同时改善。
- F64 auto-pack：将原生、rank-2、inner-stride=1、NoTransform 的允许元素大小扩到
  8 bytes，仍复用同一个 `matmul_pack` 算子与 workspace，不复制 packing 逻辑。
  相对未启用 auto-pack 的固定二进制，decode raw/packed-A/packed-B 分别
  -27.21%/-7.60%/-38.70%，batch raw/packed-A/packed-B 分别
  -53.37%/-31.39%/-46.93%；tail raw +1.19%，因为阈值未触发，属于短函数布局噪声。
  `.text` 813,716→806,680 bytes，候选 SHA256
  `5b13768f4cc12582242738215f54596ac40f2474edb27e4f9aaf314ccb8d0937`。
- 单 streaming region 实验：把 F64 加入 SME-009 的合并区域后 `.text` 增加 9,728
  bytes；针对真正触发该分支的 raw case 隔离复测，decode +2.26%，batch -4.94%。
  decode 稳定回退且收益只在大 batch，故拒绝；F64 继续使用各自独立的 pack/Matmul
  streaming region。非触发的 packed case 曾随链接布局摆动到约 +9%，不作为该实验
  的因果证据。
- 正确性：ZA64 版本在 Native/fixed-SVL=512 的 MatmulPack 测试各 6/6；F64
  auto-pack/Matmul 在两种构建的主测试各 16/16。公共 Tile2D/packing 改动在本地 GCC
  AMX 主测试 11/11、batch 2/2、mixed packing 1/1 也全部通过。

### MATMUL-SME-016：F64 balanced area-8 ZA 调度

- 状态：保留 balanced 版本。仅 `SME_F64F64` 的 Automatic policy 可选 area-8；
  ZA32 atom 继续保持 area-4，双 packed fast path 也限制为最多 4 个输出 tile。
- 动机：ZA64 每个 tile 只占 ZA 的 1/8；旧 catalog 仍按 ZA32 的 4-tile 上限调度，
  浪费一半 accumulator 容量。新增 `RuntimeExactArea8` 和 3×2、4×2、2×3、2×4
  microkernel；M/N 只有一个 block 时使用 1×4/4×1，而不是盲目拉到 1×8/8×1。
- 第一版 full-area8：允许所有面积不超过 8 的 family，并让 packed 路径走 generic
  loader。主 benchmark `.text` 增加 133,568 bytes；decode/batch 的 packed/auto-packed
  case 回退 24%～138%。原因不是 MOPA 数量，而是面积大于 4 后绕开已有 direct packed
  pointer K loop，重新进入通用 DataAccess/load 路径。因此不能只扩大 catalog 而不
  同步约束 fast path。
- 第二版 raw-only：双 packed 保持 area-4，`.text` 仍增加 101,824 bytes；大 shape
  基本恢复，但 1×8 在 FP32-memory→F64-compute decode 上回退 +45.71%。极端单边 tile
  增加 live ZA/Z 状态和展开体，在 M=1 的转换加载瓶颈下得不偿失。
- 最终 balanced 版：单边最多 4，二维只选 3×2、4×2、2×3、2×4。严格对照版本
  同时保留 SME-015 的 F64 auto-pack，只把 EffectivePolicy 在 area-4/area-8 间切换。
  最终在 CPU 500 按 A-B-B-A 固定二进制交错，每个进程内三 repetition；batch 因首轮
  20ms 短测出现相反方向，又追加 A-B-B-A-A-B-B-A，表中 batch 使用追加复测中位数：

| Case | Raw | Packed-A | Packed-B | Packed-AB |
|---|---:|---:|---:|---:|
| tail 11×13×9 | +1.64% | -8.56% | -12.07% | +0.11% |
| decode 1×1024×1024 | -0.66% | -0.29% | -0.43% | -0.09% |
| batch 64×256×256 | -0.53% | -1.43% | +0.82% | -10.52% |

- 同一固定 CPU 的 scenario 严格 area-4+auto-pack→balanced-area8：原生 F64
  tail/decode/batch/real-tail/shared-decode 为
  -12.96%/+0.06%/-1.58%/-12.24%/-15.35%；FP32-memory→F64-compute 为
  -12.56%/-7.80%/-3.96%/-14.26%/-0.62%。10 项变化中位数 -10.02%，所有目标
  无超过 2% 的回退，full-area8 的 conversion decode 问题被消除。
- 代价：主 benchmark `.text` 806,680→861,336 bytes（+54,656，+6.78%）；
  area-4 对照 SHA256 为
  `5b13768f4cc12582242738215f54596ac40f2474edb27e4f9aaf314ccb8d0937`，balanced
  候选为 `3411fb0bea751d6aca4ca4b9f835b4adaef4e04f5b815848a90d867a8d3506f1`。
  scenario `.text` 2,350,188→2,422,636 bytes（+72,448），候选 SHA256
  `265158c2b8be7c42cede4e137e3867f1fa38d2c1d1a469d91d301161a51d1378`。
  代码体积主要来自新增的 6/8-tile template family，而非运行时 scheduler 本身。
  后续若要继续压体积，应将 ZA64 大 family 拆到独立 streaming-call specialization，
  不能退回 full catalog。
- 噪声记录：首轮固定 CPU 的完整主 benchmark 只有每版本两个进程样本，batch raw/
  packed-A 曾显示 +3.03%/+3.30%；追加四个进程样本后变为 -0.53%/-1.43%，单进程
  样本跨度约 8%～12%。20ms benchmark 的两个进程中位数不足以判定 2% 级回退，
  后续必须至少使用四个交错进程样本或延长单次测量。
- 无效 fast-path 实验：曾为 3×2、4×2、2×3、2×4 实现 direct packed-pointer K
  loop，并移除 `Outputs<=4` gate；候选与基线 SHA256、`.text` 完全相同，说明这些
  specialization 全部在编译期不可达。原因是双 packed 输入已由 EffectivePolicy
  强制 area-4，单边 packed 又不满足 `FastPacked`。固定 CPU 八进程对比仍出现
  -2.87%～+6.05% 的 wall-time 摆动，进一步证明同一二进制也会产生数个百分点的
  假差。源码已回退；以后若要给 area-8 做 packed fast path，必须先有意改变该 policy，
  不能只扩展 `compute_packed_groups`。
- 反汇编/门禁：最终主 binary 静态包含 78 条写 ZA4～ZA7 的 F64 `FMOPA`，证明
  area-8 family 实际生成；pack binary 有 60 条 ZA64 horizontal load 和 60 条
  vertical ZA→Z read。Native/fixed-SVL Matmul 各 16/16、MatmulPack 各 6/6；两项
  manual-SME disassembly gate 2/2（13.17/13.29 s）。

### MATMUL-COV-009：secondary dtype Pack 全 signedness/大 shape 覆盖

- 状态：保留，AMX/SME 共用。原 `DTypeProbeCases` 只有 17×65，无法判断
  full-panel/K-chunk 优化对 FP16/INT8 的效果；将公共 secondary catalog 扩为
  17×65、64×256、513×1000、1025×1027，继续覆盖 A/B 与 Const/Dyn。
- SME 由 117 增至 201 个注册：FP16、FP32、F64 与 s8×s8、s8×u8、u8×s8、
  u8×u8 四种 INT8 Atom 均运行上述 16 项；BF16 仍运行完整 Qwen/DeepSeek catalog。
  AMX 同步扩至 137 项，包含 FP16 和四种 INT8 signedness。
- 测试补齐：SME direct 与 full-panel multi-K-chunk 对四种 INT8 的 A/B 两侧全部
  验证，BF16/FP16 也补齐缺失侧；AMX 四种 INT8 从“每种只抽一侧”扩为 A/B 两侧。
  本地 GCC `MatmulPackAMXTest-Native` 7/7；920f-4 Native/fixed-SVL 的 SME Pack
  各 6/6。
- 工程代价：仅增加注册和模板覆盖时，SME Pack benchmark `.text`
  1,081,048→1,728,024 bytes（+646,976）；这是 benchmark 可执行文件成本，不是库
  二进制成本。后续若继续扩类型×shape，应该按 dtype shard，避免单 binary 的链接
  布局污染短 case。

### MATMUL-SME-017：恢复 BF16/FP16/INT8 的 ZA32 full-panel 快路

- 状态：保留。SME-015 为排除 ZA64 时把 full-panel gate 从所有 ZA32 类型误改为
  `sizeof(U)==4`，使 BF16/FP16/INT8 绕过 SME-001 的 ZA32 tile0/1↔tile2/3
  charge/drain，退回 `load ZA → read vertical to Z → interleave/store`。正确条件是
  `sizeof(U)<=4`；F64/uint64 继续走 SME-015 的 ZA64 generic transpose。
- 测量：920f-4、CPU 500，扩展覆盖后的完全相同二进制按
  A-B-B-A-A-B-B-A，每进程每 case 三 repetition。独立 Pack 结果：

| Atom | 全套中位数 | Small 64×256 | Medium 513×1000 | Large 1025×1027 |
|---|---:|---:|---:|---:|
| BF16 | -8.71% | -17.83% | -8.58% | -20.63% |
| FP16 | -12.70% | -16.13% | -8.74% | -20.17% |
| INT8 s8×s8 | -26.17% | -38.86% | -31.30% | -20.46% |
| INT8 s8×u8 | -27.20% | -38.98% | -33.94% | -21.01% |
| INT8 u8×s8 | -27.31% | -38.93% | -33.47% | -20.91% |
| INT8 u8×u8 | -27.67% | -38.68% | -35.71% | -21.07% |

- 控制组：FP32 16 项中位数 -0.03%、范围 -0.81%～+0.52%；F64 16 项中位数
  -0.00%、范围 -0.93%～+0.26%。反汇编中 ZA32 horizontal `ld1w` 252→1308，
  ZA32 vertical `st1w` 168→1224；F64 的 ZA64 `ld1d` 与 vertical read 都保持 60，
  证明 uint64 路径没有被错误并入 ZA32 fast path。
- 短 tail 噪声：BF16 tiny/tile 最高显示 +10.44%/+2.62%，INT8 17×65 为
  +3.03%～+4.36%，但这些 shape 的 spatial 小于 panel，运行时不会进入 full-panel
  分支；INT8 tail 的 `run_case` 符号前后同为 0x770 bytes。它们来自候选新增大块使
  page/只读数据位置变化，不是执行路径回退；扩展 dtype shard 是后续消除此类测量
  污染的正确办法。
- 端到端 Matmul：主 benchmark 候选 `.text` 861,336→873,176（+11,840）。35 个
  `workspace_bytes>0`、确实触发 auto-pack 的 BF16 case 中位数 -2.08%，范围
  -17.54%～+0.86%；square/long-K/decode-QKV/decode-MLP/rank-expansion raw 分别
  -15.96%/-15.86%/-16.45%/-16.28%/-14.81%。FP16 decode raw/packed-A
  -16.85%/-16.97%，batch raw -9.59%；INT8 decode raw/packed-A
  -17.15%/-17.14%，batch raw -20.05%。不触发 packing 的 packed-AB case 的正负
  波动不归因于本修改。
- 二进制：扩展 Pack baseline SHA256
  `8bd00087da33703885834b5d4197a86bebde4b90b91e465023bc26ecc7d3f654`，`.text`
  1,728,024；修复候选
  `86af8e717bee97c7168119e2ffbca699b652970b906940b66d7db73ef15c9e82`，`.text`
  1,857,584（+129,560，+7.50%）。主 Matmul 候选 SHA256
  `f6ba927206644f7e2dda6b72f7ee61b99eb1996b7cd602c779de7b33a739205c`。
- 被拒绝的配套实验：曾把 full-word loop 改成固定 odd/even 两 chunk，并拆开预取/
  非预取阶段；FP32 四项仅 -0.09%～-0.37%，`.text` 却增加 3,008 bytes，故恢复原
  parity/prefetch loop。说明当前主要问题是 gate 错误，不是每 chunk 两个小分支。
- 构建坑：`rsync -a` 保留 mtime；快速来回切换同一 header 条件时，远端 Make 可能
  认为对象已更新而跳过重编译。严格 A/B 必须核对 SHA/`.text`，必要时显式
  `cmake -E touch <header>`，否则会得到伪基线。
- 正确性：SME Native/fixed-SVL Pack 各 6/6；本地 AMX Pack 7/7。最终保留的是一行
  gate 修复和覆盖扩展，没有保留 paired-loop 重排。

### MATMUL-COV-010：转换/融合 × 七种 packing mode 独立 benchmark

- 状态：保留。新增六 shard 的 `MatmulPackedScenarioBench`，与原 115-case scenario
  分离，避免扩展 packing 模板再次改变主性能 binary 的函数布局。
- 七种模式：Raw；offline PackedA/PackedB/PackedAB（计时前 pack）；以及
  OnlinePackedA/OnlinePackedB/OnlinePackedAB（计时区内每次 pack）。名字显式包含
  `input_mode`，同时记录 `packed_elements`、`packing_included` 和 operation workspace；
  online byte accounting 包含源输入读与 packed buffer 写。
- 覆盖交叉：原生 BF16+bias+ReLU；FP32-memory→BF16-compute+ReLU；BF16/FP16
  memory→FP32-compute→narrow output+bias+ReLU；FP32 动态量化→INT8 dequant；
  FP32 动态量化→ReLU+int8 requant；预量化非对称 U8×S8 correction+dequant。
  每种继续覆盖 tail、M=1 decode、M=64 batch。
- 构建/正确性：本地 AMX 127 个注册、126 个实际 case；920f-4 SME 148 个注册、
  147 个实际 case；两边完整 JSON 都无 error。SME Native 基线 `.text` 1,776,600，
  SHA256 `6d7358cd6e33c62e12f6f8baba215c8585d06f472ecc2b68459fa9635e56d520`。
- 第一批 cost-model 结论：offline PackedB 在 decode 上通常比 Raw 快 54%～89%，
  PackedAB 快 56%～96%；native/conversion 的 online pack 基本等于 Raw，说明已有
  auto-pack 已覆盖它们。FP32→INT8 的 online PackedB 却比 Raw 慢 43%～79%，虽然
  offline PackedB 快约 89%；这个矛盾直接定位了 SME-018 的 transform-pack backend
  缺口。

### MATMUL-SME-018：FP32→I8/U8 融合量化 packing 与自动打包

- 状态：保留。`SMEPostprocess` eligibility 从仅 NoTransform FP32→BF16 扩为：
  memory=FP32、compute=S8/U8、elementwise 且 permutation-equivalent、transform
  `TIn/TOut=FP32`。复用现有 `pack_postprocess`：ZA32 transpose 后执行 transform，
  `FCVTZS/FCVTZU`、narrow/interleave，再直接形成 KPack=4 packed byte；padding 在
  transform 后清零，不复制 packing 布局逻辑。
- 仅替换 MatmulPack selector 的固定 CPU A-B-B-A-A-B-B-A：

| 模式 | 六个 tail/decode/batch×两种输出的中位数 | 代表收益 |
|---|---:|---:|
| OnlinePackedA | -23.67% | tail -51%～-54%，batch -23%～-24% |
| OnlinePackedB | -52.90% | decode -66.5%，batch -49%～-51% |
| OnlinePackedAB | -69.85% | decode -69.8%，batch -73%～-76% |
| Raw/offline packed 控制 | 约 -0.8%～+0.2% | 不执行计时区 transform pack |

- selector 候选 `.text` 1,776,600→1,774,168（-2,432），SHA256
  `94a444a22536f566af343b06b7ed796e8979f8a164f1c166b70b71714f3529ab`；性能与
  代码体积同时改善。online PackedAB 相对同一候选 Raw：decode 快约 41%～44%，
  batch 快约 73%～76%，但 tail 慢约 5%～10%，因此不能无条件打包。
- AutoPackOperand 随后允许相同的 FP32 elementwise quantization transform；沿用现有
  `M*N*K`、`M*N` 阈值，tail workspace=0，decode/batch 自动 pack。相对“已有 SME
  quant pack、尚未 auto-pack”的严格 A/B：dequant Raw tail/decode/batch 为
  +0.89%/-41.39%/-75.67%；ReLU+requant 为 -0.32%/-44.48%/-71.38%。候选 `.text`
  1,774,168→1,782,360（+8,192），SHA256
  `834f091edcaa30fcf2b02b0f67221840a50e03082a604716b020ae88028bab55`。
- 单侧显式 packed 输入也会自动 pack 剩余 raw transform 侧，因此 packed-A/B 的
  decode/batch 同样提升 37%～84%；PackedAB 不触发新分支，个别 decode 显示
  +4%～+6% 是新模板改变链接布局后的非目标波动，不归因于算法。
- 反汇编：S8/U8 selector 测试的两个 Streaming+ZA owner 各 0x494 bytes、普通
  `BL=0`；每个静态实例有 8 条 `FMUL`，S8 为 8 条 `FCVTZS`，U8 为 8 条
  `FCVTZU`，随后 `UZP` narrow/interleave 并 packed store。说明量化没有标量化或
  退回通用 gather。
- 正确性/门禁：FP32→S8/U8 Pack 编译期静态断言选择 SMEPostprocess，并逐元素验证
  A/B、K tail 和 padding；新增大 asymmetric transform Matmul 断言 workspace>0。
  SME Pack Native/fixed 各 6/6，Matmul Native/fixed 各 17/17，manual-SME
  disassembly gate 2/2（16.77/14.09 s）；最终 packed scenario Native/fixed-SVL
  各 147 个 case 无 error，本地 AMX packed scenario 126 个 case 无 error。

### MATMUL-SME-019：transform pack 的满面板/满 K chunk 编译期特化

- 状态：保留。`pack_postprocess` 原先对每个 panel、K chunk 都重复执行
  spatial 和 K tail 防护：构造 partial predicate，每个 slice 判断
  `slice >= active_k`，并对已转换结果执行 `zero_inactive_spatial`。将
  chunk 拆成 `FullPanel`×`FullK` 四个编译期版本：完整 K chunk 使用
  all-true predicate 和固定 group 数，完整 panel 使用 all-true read predicate
  并移除 spatial zeroing；每个 panel 只对最后一个不完整 chunk/panel 进入 tail
  版本。这是将用户提出的 K-tail 运行时判断变为编译期判断的最终落地。
- 测量协议：920f-4，BiSheng Clang 19.1.7，SVL=512，CPU 500；独立
  benchmark 固定 CPU，8 个独立进程按 A-B-B-A-A-B-B-A 运行，每进程每
  case 三次 repetition，取严格配对中位数。
- 直接 FP32→BF16 SME postprocess pack：A 128×3584 Const
  270.527→251.873 us（-6.90%），Dyn 同为 -6.90%；B 4608×3584 Const
  8512.962→8068.550 us（-5.22%），Dyn -5.09%；四项中位数 -6.06%。
- 端到端 FP32→I8/U8 quant packed scenario：batch Raw dequant/requant
  -5.71%/-3.74%，OnlinePackedA -4.54%/-3.60%，OnlinePackedB
  -3.17%/-2.52%，OnlinePackedAB -5.04%/-3.58%。decode 大多数为
  -0.1%～-1.1%，tail 最大 +0.98%。七种 mode 的全 case 中位数分别为
  Raw -0.67%、PackedA -0.35%、PackedB +0.22%、PackedAB -0.77%、
  OnlinePackedA -0.62%、OnlinePackedB -0.57%、OnlinePackedAB -1.55%。
  PackedB 的最大 +5.92% 来自不执行该 pack 路径的已打包布局 case，不作为
  本修改的回退归因。
- perf 解释：B 4608×3584 Const 五次中位数，cycles
  368,327,631→350,503,690（-4.84%），instructions
  444,010,841→377,200,310（-15.05%），IPC 1.21→1.08，branch misses
  105,777→110,263。收益来自满 chunk 移除冗余 predicate/tail/zeroing 指令；
  IPC 下降是指令数大幅减少后的结果，不代表吞吐回退。
- 代码体积：Pack baseline SHA256
  `86af8e717bee97c7168119e2ffbca699b652970b906940b66d7db73ef15c9e82`，
  `.text` 1,857,584；候选
  `dbf64c88e4672f1fbc78639555874e9055325761664698fb8e042000fc84c64b`，
  `.text` 1,863,216（+5,632，+0.30%）。packed scenario baseline
  `834f091edcaa30fcf2b02b0f67221840a50e03082a604716b020ae88028bab55`，
  `.text` 1,782,360；候选
  `866e261179e78f57c8a61bcc4a543826429c1e90a2d5ce51d51812f14f6c9b5b`，
  `.text` 1,792,344（+9,984，+0.56%）。FP32→BF16 owner 由四个
  0x3f8-byte 实例变为四个 0x968-byte 实例，每个 +1,392 bytes；这是四路
  compile-time specialization 的显式代价。对应 5%～7% 直接 pack 收益与
  3%～6% batch 收益，当前性价比可接受；后续若扩展大量 transform，需要
  监控模板实例膨胀。
- 正确性/门禁：920f-4 Native/fixed-SVL MatmulPack 各 6/6，Matmul 各
  17/17；manual-SME disassembly gate 2/2（14.40/14.51 s）。packed scenario
  Native/fixed-SVL 各 147 个注册 case，每 case 四次输出（JSON 各 588 条），
  全部 0 error。

### MATMUL-SME-020：rank-3 shared-B 跨 batch 打包一次并复用

- 状态：保留。原 auto-pack 把 `COutput rank != 2` 全部排除，真实
  `B=8, M=1, N=K=256` shared-weight decode 因此对同一 B 连续执行 8 次
  raw loader。新路径只对 rank-3 且 B batch stride 在布局类型中为
  `Const<0>` 的 operation 实例化：batch 外分配一对 rank-2 packed buffer，B
  只在首个 batch pack 一次，A 每 batch 更新，然后所有 batch 复用同一 B
  并进入 packed kernel。公开 packed-layout ABI 仍是 rank-2，没有引入带 batch
  维的新格式。
- selector 坑点：第一版虽允许 rank-3，却仍硬编码检查
  `stride_type_t<1>`；rank-3 的 K 轴是 `Rank-1`，所有 case 的
  `workspace_bytes` 仍为 0。该伪候选 SHA256
  `a4237c8f90513bd3733f0758f6cbe561b7ae2caf11ae2f573a6ac3539e325c9f`，
  `.text` 2,504,556；虽显示 FP16 -20.68%、F64 +10.18%，但目标路径没执行，
  全部判定为代码布局噪声，不计收益。
- 运行时 gate 实验：修正 K-stride 后，若对所有 rank-3 布局都静态
  实例化 auto-pack，shared-B 已有 21 项中位数 -83.32%，但不打包的
  independent-tail 23 项中位数 **+6.32%**，范围 +0.61%～+7.73%。
  原因是每个小 operation 类型仍携带巨大的 typed auto-pack fallback 和运行时
  分支；候选 `.text` 2,836,140，SHA256
  `5d3f95291a2e269184e2aa36af26b1de8d1eec309e5610f5b57db80eec25aa85`。
  该版拒绝，改为编译期 `Const<0>` shared-B specialization；独立 tail
  最终中位数 -0.69%，范围 -1.80%～+0.04%，回退消失。
- 严格性能：920f-4、CPU 500、A-B-B-A-A-B-B-A，四个进程中位数配对。
  22 个 `workspace_bytes>0` shared-B case 全部正向，中位数 -82.68%，
  范围 -88.11%～-60.84%：

| 类型/融合 | baseline→candidate | 变化 |
|---|---:|---:|
| FP32 native convert | 465.807→55.364 us | -88.11% |
| BF16 native convert | 174.737→26.691 us | -84.72% |
| BF16 bias / bias+ReLU | 145.722→26.653 / 144.994→26.651 us | -81.71% / -81.62% |
| FP16 native | 175.009→28.127 us | -83.93% |
| F64 native | 499.478→195.614 us | -60.84% |
| 四种 native INT8 | 约 69.8→13.6 us | -80.45%～-80.51% |
| quant dequant / requant / asymmetric | 约 433～441→61～63 us | -85.68%～-85.86% |
| BF16 memory→FP32 compute widening | 396.617→130.601 us | -67.07% |

- widening cost model 修正：最后一项初版仍用单 batch `M=1` 判断 B 复用
  不足，未执行目标路径并显示 +5.96%。shared B 的真实复用是
  `batch*M`；阈值纳入 batch=8 后实际打包，变为 -67.07%。唯一未启用的
  FP32-memory→F64-compute case 为 +1.17%，它本来就不是 auto-pack 可用
  conversion，不执行新分支。
- perf 解释：BF16 native convert 固定 1000 iteration、外层三次 perf 重复，
  cycles 1,079,320,969→191,748,998（-82.23%），instructions
  1,425,767,109→369,406,290（-74.09%），branch misses
  309,958→176,776（-42.97%），IPC 1.32→1.93。收益与 wall time 一致，
  来自消除 8 次 raw traversal 并提高 packed kernel 利用率。
- 代码体积：基线 SHA256
  `265158c2b8be7c42cede4e137e3867f1fa38d2c1d1a469d91d301161a51d1378`，
  `.text` 2,422,636；最终候选
  `42f8b7d75c28247c6d30dd89e9a3571bf509a46c4b6ae06791209e64403a951d`，
  `.text` 3,066,072（+643,436，+26.56%）。这个总量主要是 benchmark
  为每条 pipeline 同时实例化 shared/independent 两种布局，不是单个
  应用 operation 的代价。代表 BF16 native `run_batched_scenario`：旧通用
  实例 0x1a40 bytes，新 independent 0x1944（-252 B），新 shared
  0x1d30（相对旧实例 +752 B）。
- 测试与覆盖：batch benchmark 现在以编译期 `BroadcastB` 分开注册，
  防止两种 layout type 再次合并。SME Native/fixed-SVL scenario 各 115 个
  case，0 error，各有 22 个 shared-B case 确认 workspace>0；SME batch 测试
  Native/fixed 各 2/2，其中 BF16 shared case 扩为 B=8、M=1、N=K=256 并断言
  workspace>0。本地 AMX batch 2/2，scenario 95 case 全无 error；AMX 不启用
  SME-only batch auto-pack，仅受益于更准确的 benchmark layout 类型。主 Matmul
  Native/fixed 各 17/17；manual-SME disassembly gate 2/2（15.01/14.81 s）。

### MATMUL-SME-021：shared-B 多 dtype 单 Streaming+ZA region

- 状态：保留。SME-020 在 B=8 时只 pack B 一次，但除 BF16/F32 外的
  atom 仍会经历 B pack 一次、A pack 八次、Matmul 八次的 Streaming+ZA
  区间边界。将 SME-009 的单 region 扩展严格限定为 SME-020 独立的
  `Rank3SharedB` specialization：原生 FP16/F64/INT8，以及 SME-018 已证明
  全程 Streaming+ZA 的 FP32→I8/U8 融合量化 pack，在 batch 外只打开一次
  region。BF16/FP16→FP32 staged pack 因仍含 ordinary-SVE postprocess，继续排除。
- 范围边界：普通 rank-2 继续只启用 SME-009 已保留的 BF16/F32；这很
  重要，因为 SME-015 曾测得 rank-2 F64 单 region decode +2.26%。本次不改
  rank-2、independent batch、mixed packed/raw 或未达 auto-pack 阈值的路径。
- 严格 A-B-B-A-A-B-B-A（920f-4、CPU 500，四进程中位数）：

| 新命中路径 | SME-020→SME-021 | 变化 |
|---|---:|---:|
| FP16 native | 28.085→26.600 us | -5.29% |
| F64 native | 190.890→174.729 us | -8.47% |
| native s8×s8 | 13.681→11.851 us | -13.37% |
| 其他三种 native INT8 | 约 13.7～14.1→11.9～12.3 us | -13.20%～-13.52% |
| quant dequant | 61.152→59.297 us | -3.03% |
| quant requant | 62.125→59.869 us | -3.63% |
| asymmetric quant | 63.175→60.962 us | -3.50% |

- 控制组：已经是单 region 的 BF16 七项为 -0.08%～+0.40%，FP32
  native convert -1.22%。FP32 sigmoid/accumulate+ReLU+scale 在大 scenario binary
  中显示 +2.62%/+2.39%，但两个 `run_batched_scenario` 符号大小前后严格
  相同（0x1958/0x1918），仅因新实例使链接地址同时前移 0x1bc0。
  independent-tail 23 项中位数 +0.06%，范围 -0.89%～+1.33%；没有运行时
  新分支或系统性回退。
- perf/汇编：整个 scenario 静态 `SMSTART/SMSTOP` 各 127→112，减少 15 对
  region owner。固定 iteration perf 中，F64 cycles
  1,125,207,468→1,073,164,477（-4.62%），instructions
  1,399,146,038→1,373,181,005（-1.86%），IPC 1.24→1.28；INT8 s8×s8
  cycles 846,738,166→733,125,344（-13.42%），instructions
  1,946,693,371→1,874,175,998（-3.73%），IPC 2.30→2.56。说明 INT8 主要
  收益来自区间边界/流水恢复与更高发射率，不是大幅删除计算指令。
- 代码体积：SME-020 基线 SHA256
  `42f8b7d75c28247c6d30dd89e9a3571bf509a46c4b6ae06791209e64403a951d`，
  `.text` 3,066,072；候选 SHA256
  `f07bf14ec4c685c8f9bc6a46eb1c20f62f8e80cdfe40caf30cd6d522c29a4125`，
  `.text` 3,076,120（+10,048，+0.33%）。这个体积增长限定在显式 shared-B
  benchmark specialization，没有重复 SME-009 首版对所有 rank-2 catalog 复制
  traversal 的 +16.31% 问题。
- 正确性/门禁：Native/fixed-SVL scenario 各 115 case，0 error。batch 测试
  Native/fixed 各 2/2，其中 shared-B 大 case 已扩为 FP32/BF16/FP16/F64 和
  s8×s8、s8×u8、u8×s8、u8×u8 全部原生 atom，每项断言 workspace>0 并
  逐元素校验。batch manual-SME disassembly gate 2/2（8.66/9.49 s），新单
  region 内 `BL/BLR=0`。rank-2 编译期条件未改变，继承 SME-020 主 Matmul
  Native/fixed 各 17/17 与主 disassembly 2/2 门禁。本地 GCC 重建 AMX
  scenario 后 95 case 全部 0 error，证明 SME-only trait 扩展没有污染 AMX 选择。

### MATMUL-COV-011：shared-weight batch 阈值边界与 Qwen 实用 shape

- 状态：保留，SME/AMX 共用 scenario 目录。原真实 batch 只有
  independent-tail 和 B=8、M=1、N=K=256 两个 shape，无法定位 cost
  threshold，也无法看到真实 LLM 权重规模。新增七个 shared-B shape：

| 名称 | B×M×N×K | 目的 |
|---|---:|---|
| tiny | 8×1×32×64 | 保持不打包的下界控制 |
| small | 8×1×64×128 | 总 work 刚好达 64 Ki 阈值 |
| small-long-K | 8×1×64×256 | 小 N、长 K |
| wide-short-K | 8×1×128×64 | 大 N、短 K |
| MLP projection | 8×1×1024×4096 | 中等规模 MLP 投影 |
| Qwen O projection | 8×1×3584×3584 | 真实 hidden projection |
| Qwen QKV | 8×1×4608×3584 | 真实 QKV 投影 |

- 每个 shape 继续交叉 23 条 pipeline：全原生 atom、FP32/BF16/FP16/F64
  conversion、bias/ReLU/sigmoid/scale/accumulate，对称/非对称量化、dequant/requant 和
  prequantized correction。Native scenario 从 115 增至 276 个实际 case；扩展只增加
  runtime shape，不新增 Matmul operation 类型。
- saturating requant 坑点：首次全量运行有 9 条失败，全部来自 K=3584/4096
  的 ReLU+INT8 requant。kernel 经 DataAccess `AccessDefaults` 执行 saturating store，
  benchmark reference 却对超出 int8 范围的 float 使用 `static_cast<int8_t>`，两者
  语义不同；小 K 时数值未越界，因而一直未暴露。reference 现在先 clamp
  到 `numeric_limits<Memory>::lowest()/max()` 再窄化，明确建模现实的
  saturating requant；276 case 最终全部 0 error。
- coverage-only 基线 SHA256
  `e5192a15c525c5b66130dfa5225b849c7c5ed9bafd07a29b48899afe66d48bef`，
  `.text` 3,080,792。当时 tiny/small/small-long-K/wide-short-K 各 23 条全部
  workspace=0；decode/MLP/Qwen 每个 shape 各 22/23 条 workspace>0，唯一例外是
  不支持 auto-pack 的 FP32-memory→F64-compute。

### MATMUL-SME-022：shared-B batch-aware auto-pack cost model

- 状态：保留。旧通用阈值只使用单 batch 的 M×N×K；对 B batch
  stride=`Const<0>` 的 operation，一次调用实际计算 batch×M 行，且 B 只
  pack 一次。新建 `effective_m = saturating(batch*M)`，只在 `Rank3SharedB`
  specialization 中用于 reuse/work 和 BF16-widening 阈值；rank-2、AMX 和
  independent batch 仍使用原 M。
- 选择结果：tiny 的总 work 仅 16,384，仍保持 workspace=0；small、
  small-long-K、wide-short-K 总 work 分别为 65,536、131,072、65,536，
  各有 22/23 条开启 auto-pack。唯一不启用的仍是 FP32-memory→F64-compute。
- 920f-4、CPU 500、A-B-B-A-A-B-B-A 严格配对：69 条全 case 中位数
  -73.76%，范围 -84.62%～+0.32%；排除三条未执行新路径的 F64-conversion
  控制后，66 条目标 case 中位数 -74.10%，范围 -84.62%～-28.73%。

| shape | 全 23 条中位数 | BF16 native | s8×s8 native |
|---|---:|---:|---:|
| 8×1×64×128 | -69.78% | 18.121→4.911 us（-72.90%） | 9.546→2.907 us（-69.55%） |
| 8×1×64×256 | -74.31% | 37.686→9.217 us（-75.54%） | 17.305→5.226 us（-69.80%） |
| 8×1×128×64 | -75.24% | 18.388→4.503 us（-75.51%） | 9.777→2.518 us（-74.24%） |

- 回退核查：未打包 tiny 与原已打包 decode/MLP/Qwen 共 115 条控制中位数
  +0.04%，范围 -3.55%～+2.54%；分 shape 中位数为 -0.03%～+0.11%，没有
  系统性回退。新计算只发生在 operation 构造/cost selection，不在计时的
  Matmul 热循环内。
- perf：BF16 small-long-K 固定 5000 iteration，cycles
  1,294,475,027→301,639,321（-76.70%），instructions
  1,813,862,067→691,996,927（-61.85%），IPC 1.40→2.29，branch misses
  352,833→319,804（-9.36%）。这与 wall time 收益一致，证明不是单纯
  的 threshold 分支布局差异。
- 代码体积：基线 `.text` 3,080,792；候选 SHA256
  `4a0f1d6c976fe34f753d9f4b540028e9cb26afbeb85c2d10dcb79cd8c8a367fe`，
  `.text` 3,081,368（+576，+0.019%）。增长只是溢出安全的 `batch*M` cost
  计算，没有新增 kernel/packing specialization。
- 正确性/门禁：Native/fixed-SVL 各 276 case，0 error；batch Native/fixed 各
  2/2，batch manual-SME disassembly 2/2（8.74/8.83 s）。本地 GCC AMX 重建
  扩展目录后为 228 case，全部 0 error；AMX 仍使用 rank-2 原 M cost model。

### MATMUL-COV-012：rank-3 shared weight 的离线/在线 packed-B 覆盖

- 状态：保留。此前 packed scenario 只验证 rank-2，无法回答“模型权重在
  operation 外预打包一次、在 B 个 decode token 间复用”是否真的优于每次
  auto-pack。现在 packed 输入允许用于逻辑 rank>=2；packed format 本身仍只
  保存二维矩阵，所以 rank-2 packed A/B 在任意 batch prefix 上的语义都是
  broadcast，不表示每个 batch 拥有一份独立 packed matrix。
- 遍历坑点：不能把 packed tensor 直接交给 `for_each_dims<PrefixRank>`。
  packed layout 的 panel 维度是物理格式，不是 batch 维；直接切片会把第一个
  panel 当 batch，既可能越界也会静默算错。`BroadcastPackedOperand` 让循环只
  遍历 raw/C operand，进入单个 GEMM 前再取回完整 packed spec；auto-pack
  已经打包了某一侧、另一侧原本就是 packed 的分支也使用同一包装。
- 新 benchmark mode：`raw`、计时区外一次完成的 `packed_b`、每次 iteration
  都执行 pack 的 `online_packed_b`。共享 B 使用真正的 rank-2 layout，而 A/C
  保持 rank-3；离线模式把权重打包成本排除在 steady-state Matmul 外，在线
  模式则明确以 `packing_included=1` 计入。新增三个代表 pipeline：原生
  BF16+bias+ReLU、FP32->BF16+ReLU、FP32->I8/U8+dequant，并覆盖 COV-011
  的八个 shared-weight shape。Packed Scenario 的 SME case 从 147 增至 219，
  AMX 从 126 增至 198。
- 初始 coverage binary：SHA256
  `1c180b8e6a0b6bdac622b7c45dbca35d70038df4c31b7aa1a5440b14334161d0`，
  `.text` 2,184,656。SME 219/219、AMX 198/198 均 0 error。batch 单测还对
  SME 的 F32/BF16/F16/F64/四种 INT8 atom 和 AMX 的 BF16/四种 INT8 atom
  逐元素验证离线 packed B；Native/fixed-SVL SME 与 AMX 各 2/2。
- 覆盖本身暴露了一个 cost-model 缺口：大 shape 的离线 packed B 已有明显
  收益，但两个小 shape 的原生 BF16 一度出现约 +148%/+176% 回退。原因不是
  B pack 实现错误，而是“B 已 packed、A 仍 raw”只按单侧 128 Ki work threshold
  判断，未把两侧最终会组成完整 packed pair 作为更便宜的路径处理。后续
  SME-024～026 分三步修复，并为量化 A 单独保留更严格边界。

### MATMUL-SME-023：shared weight 离线 packed-B steady-state 路径

- 状态：保留。目标不是用 operation 内 auto-pack 替代离线打包，而是允许调用者
  对长期复用的权重只调用一次 `MatmulPack<B>`，随后把同一个 rank-2 packed B
  广播给整个 rank-3 Matmul。最终 SME-026 binary 内，离线 pack 相对 raw 的
  steady-state 结果如下；所有数值是 920f-4、CPU 500、3 repeats median，单位 us：

| B×M×N×K | BF16+bias+ReLU | FP32->BF16+ReLU | FP32->I8/U8+dequant |
|---|---:|---:|---:|
| 8×1×32×64 | 6.405→3.809（-40.53%） | 12.026→5.405（-55.06%） | 14.572→5.964（-59.07%） |
| 8×1×64×128 | 4.859→4.229（-12.97%） | 12.209→9.868（-19.18%） | 16.375→11.972（-26.89%） |
| 8×1×64×256 | 9.134→7.891（-13.61%） | 23.705→19.249（-18.80%） | 32.050→25.393（-20.77%） |
| 8×1×128×64 | 4.476→3.873（-13.47%） | 9.016→6.602（-26.77%） | 10.843→7.470（-31.11%） |
| 8×1×256×256 | 26.811→21.694（-19.09%） | 52.361→32.865（-37.23%） | 59.103→30.043（-49.17%） |
| 8×1×1024×4096 | 4312.859→3391.255（-21.37%） | 6088.584→3555.621（-41.60%） | 4926.978→2043.973（-58.51%） |
| 8×1×3584×3584 | 12219.835→10546.917（-13.69%） | 16874.295→10580.620（-37.30%） | 13631.845→5464.568（-59.91%） |
| 8×1×4608×3584 | 15648.510→13465.490（-13.95%） | 21748.390→13606.930（-37.43%） | 17490.820→6980.953（-60.09%） |

- 解释：原生 BF16 主要省掉每次调用的 B traversal/pack；conversion/quantization
  还能把 FP32->BF16 或 FP32->I8/U8 transform 一并摊到模型加载/权重更新阶段，
  因而大权重收益分别扩大到约 37%～42% 和 58%～60%。这正是 offline pack
  与 on-the-fly pack 不应被混为一谈之处：两者可以共用同一个 MatmulPack
  底层实现，但成本归属和复用次数不同。
- `online_packed_b` 把 B pack 放回每次 iteration 后，除 tiny 外基本回到 raw：
  small 及以上 BF16 为 -0.60%～+3.80%，FP32->BF16 为 -0.39%～+0.73%，
  quant 为 -0.34%～+2.07%。tiny 仍有 -37%～-52%，说明显式完整 pack 在该
  shape 偶尔能胜过 raw traversal，但不能据此把大权重的在线复制视作免费；
  Qwen O 的 BF16 在线模式已回退 +3.80%。

### MATMUL-SME-024：已 packed 一侧的 complete-pair cost threshold

- 状态：保留。若 rank-3 shared B 已离线 packed，auto-pack A 后得到与“两侧
  同时 auto-pack”相同的完整 packed pair；因此这一特例使用 64 Ki work
  threshold，普通单侧 pack 仍保持 128 Ki。变化只作用于 `Rank3SharedB &&
  ((AutoPackA && PackedBInput) || (AutoPackB && PackedAInput))`，不改变 raw、
  rank-2 或 independent-batch 的选择。
- 对 COV-012 binary 做严格 A/B：24 条 raw 控制中位数 +0.046%，范围
  -0.71%～+1.28%；`packed_b` 的 small 中位数 -18.39%，wide-short-K 中位数
  -41.96%，证明原先的小 shape 回退来自 A 未补齐打包，不是 packed B layout。
- 第一版仍有一个反例：FP32->I8/U8 的 8×1×64×128 `packed_b` 回退
  +20.81%。B 是 shared/offline，但 A 每个 batch 都不同；强制 pack A 会重复
  八次量化 transform，64 Ki 总 work 并不足以摊薄它。这个现象要求按 transform
  类型建模，不能只依赖“完整 packed pair”这一结构条件。
- binary SHA256
  `d669fb496162fdc5ac848095b80b75e0ffd46c2ac063ff7b6435c42bf119c552`，
  `.text` 2,184,656，与 COV-012 相同；修改只是 operation 构造期选择逻辑。

### MATMUL-SME-025：complete packed pair 单 Streaming+ZA region

- 状态：保留。SME-024 仍分别打开 A pack 和 Matmul 的 SME region。对 rank-3
  shared-B 且一侧已经 packed 的 operation，把剩余 A pack 与整个 batch Matmul
  放进一个手工 Streaming+ZA region；只接受 native、FP32->BF16 和已经确认
  全程 streaming-compatible 的 FP32->I8/U8 transform pack，BF16/FP16->FP32
  ordinary-SVE postprocess 仍排除。
- 严格 A-B-B-A-A-B-B-A，相对 SME-024：24 条 raw 中位数 +0.08%，范围
  -0.49%～+0.81%；24 条 `packed_b` 中位数 -2.83%，范围 -28.75%～+1.20%；
  online mode 中位数 -1.95%，范围 -25.75%～+1.85%。按 shape，packed-B 的
  small -12.72%、small-long-K -8.50%、wide-short-K -19.04%、decode -5.36%；
  大 Qwen case 的小幅波动没有形成一致回退。
- 静态代码体积反而下降：SHA256
  `f7b094f3d0412590c67b41685b82374cfc707235467d2a2dde3ad77bc632223d`，
  `.text` 2,181,968，相对 SME-024 减少 2,688 bytes。batch Native/fixed-SVL
  manual-SME disassembly 均确认 region 内 `BL/BLR=0`。

### MATMUL-SME-026：量化 A 的小 shape 边界

- 状态：保留，作为本轮最终候选。对 `Rank3CompletesPackedPair && AutoPackA`
  且 A 是 FP32->I8/U8 transform pack 的情况，只有 `N>=128 || K>=256` 才补齐
  A；否则保留 raw A + packed B。A 的 batch stride 非零，不能用 shared B 的
  batch 复用去虚构 A pack 复用。
- 与 SME-025 对量化全 shape 做严格 A/B：raw 中位数 -0.077%，范围
  -0.217%～+0.179%；`packed_b` 中位数 -0.100%，其中 8×1×64×128 为
  -7.39%；online 中位数 -0.348%，该 small shape 为 -5.62%。最终全量 run
  的 small 为 raw 16.375 us、offline 11.972 us（-26.89%）、online
  15.637 us（-4.51%）；small-long-K 因 K=256 继续 pack A，offline
  -20.77%；wide-short-K 因 N=128 继续 pack A，offline -31.11%。
- 最终 Native binary SHA256
  `51dd2b6280f0f3eb2446b4cb348eac98d841b90eb8374c6994d2c6104b5165ed`，
  `.text` 2,181,968；fixed-SVL SHA256
  `42dee04ab510aad297bbbd856c7e5fccc7073d76b4c6da96097bace870421d2d`，
  `.text` 2,102,940。量化边界没有增加静态代码体积。
- 最终验证：920f-4 `/home/ryz/vecops-neo`、build
  `cmake-build-sme-pack-arm-clang-release`、BiSheng Clang 19.1.7、
  `-O3 -march=native+bf16+sme+sme-fa64+sme-f64f64`、SVL=512。
  SME Native/fixed-SVL Packed Scenario 各 219 median case、0 error；batch
  各 2/2；两套 batch disassembly gate 各通过。local Intel AMX（system GCC）
  Packed Scenario 198 median case、0 error，batch 2/2。`git diff --check`
  通过。

### MATMUL-COV-013：权重生命周期与 calls-per-pack break-even

- 状态：保留。新增独立 `MatmulWeightReuseBench`，不继续扩张主 Packed Scenario
  binary。每个计时 iteration 中，raw strategy 连续调用同一 Matmul `R` 次；
  `pack_once_then_reuse` 在计时区内先调用一次 `MatmulPack<B>`，再调用 packed-B
  Matmul `R` 次。这样补上旧 `packed_b`（不计首次 pack、等价无限复用）与
  `online_packed_b`（每次调用都重新 pack）之间缺失的 2/4/8/16 次真实生命周期。
- 目录共 47 个 runtime case：decode 对 batch=1/2/4/8/16 × R=1/2/4/8/16；
  small 取相同五种 batch × R=1/4/16；MLP projection 取 R=1/2/4/8；
  Qwen O projection 取 R=1/2/4。每个 case 交叉 raw/pack-once 两种 strategy 和
  BF16+bias+ReLU、FP32->BF16+ReLU、FP32->I8/U8+dequant 三条 pipeline，合计
  282 个 median case。
- counter 修正：新增 `calls_per_pack` 和 `lifecycle_pack_bytes`；FLOP/items 与
  Matmul 访存乘以 R，首次 pack 的 raw-read+packed-write 只加一次。顺带修复旧
  packed-B byte counter 仍按 raw `MemoryB` 大小计数的问题；现在 packed Matmul
  使用实际 packed buffer bytes，online/lifecycle 再单独加入 pack 流量。这个修改
  只影响统计 counter，不改变计时代码路径。
- SME-026 初始 break-even：除 tiny/small 的特殊阈值外，R=1 基本为 ±2%；R=2
  已在所有 decode/MLP/Qwen pipeline 稳定转正。代表 batch=8 decode 在 R=2/4/16：
  BF16 -8.9%/-13.7%/-17.2%，FP32->BF16 -18.8%/-28.1%/-35.1%，量化
  -25.0%/-37.2%/-46.4%。MLP/Qwen 在 R=4 时分别约 BF16 -14%/-12%、
  FP32->BF16 -30%/-28%、量化 -45%/-45%。结论是长期权重只要至少复用两次，
  offline pack 通常已经值得；R=1 继续交给 operation 内 cost model。
- 最终覆盖：920f-4 Native/fixed-SVL 各 282 case、0 error；本地 AMX 282 case、
  0 error。最终 Native SHA256
  `febe103c67756c2e27306bfe3e2da54eb58478e3d1c134d6cca3c40b3c2ce228`，
  `.text` 594,040；fixed-SVL SHA256
  `d07be3fd70d7b666b716abeb7506b3c0c245fc6deb15dcb28d3c93eebc3f7311`，
  `.text` 568,820。

### MATMUL-COV-014：shared-weight K-tail 与最终场景计数

- 状态：保留。batch catalog 增加 `shared_weight_tail`：B=8、M=1、N=47、
  K=`2*k_step+1`，同时覆盖 shared-B、非整 panel N 和唯一 K tail，防止小工作量
  阈值优化只在 KPack 整除时正确。主 Scenario 每条 pipeline 因此从九个 batch
  case 增至十个，Native 最终为 299 median case；Packed Scenario 的三个 rank-3
  pipeline × 三种 mode 各增加一个 tail，共 249 median case。两套 Native 全量
  JSON 均为 0 error。

### MATMUL-SME-X05：在 lifecycle binary 重测双-bank packed K-loop

- 状态：拒绝，`VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE` 继续只作实验开关。
  小 lifecycle binary 相对基线 `.text` 593,464→594,744（+1,280），但所有 case
  都是 M=1，实际使用 1×N family，并不执行目标 2×2 loop。严格 A-B-B-A 中，
  BF16/量化控制中位数仅 +0.08%/+0.18%，FP32->BF16 却因新代码改变后续热块
  布局而为 +7.14%、最差 +29.18%。这再次证明“减小 benchmark 总类型数”仍不
  等于隔离 2×2 microkernel；必须有真正独立的 streaming-call/汇编边界。
- ABI 探针：BiSheng Clang 19.1.7 支持把 `__arm_streaming __arm_inout("za")`
  写在函数 declarator 之后，也能生成 64B 对齐的无状态切换 callee；但普通函数
  调用 shared-ZA callee 会直接报错“caller must have ZA state”。当前手工 runtime
  `SMSTART` scope 在 C++ 类型系统中不是 ZA-sharing caller，不能直接用标准函数调用
  隔离 microkernel。强行函数指针转换或裸 `BL` 会绕过 PCS 类型检查，暂不采用；
  后续若做独立汇编，必须同时扩展 disassembly gate 验证被调函数确为 streaming-
  compatible/inout-ZA，不能简单允许任意 `BL`。

### MATMUL-SME-X06：小 shared-B 的 B-only mixed specialization

- 状态：拒绝并从最终源码移除。lifecycle 数据显示 1×64×128 在 B=2/4 时只 pack
  shared B 已可获得巨大收益；第一版为 native BF16、FP32->BF16 和量化同时增加
  `SharedBOnly` mode，目标分别约 -36%/-43%、-47%/-58%、-45%/-62%，但小
  binary `.text` 增加 56,000 bytes，未命中 FP32->BF16 decode 因布局改变最高
  回退约 22%。
- 第二版只保留量化并把 enum state 限定到命中 specialization，`.text` 仍增加
  20,288 bytes；目标 B=1/2/4 为约 -17%/-47%/-63%，但隔离 sigmoid 控制在
  八进程复测后仍约 +3.6%，超过默认回退门槛。
- 尝试通过 SME 原本为 null 的 scratch 传递 packed-B pointer，让 raw/raw backend
  复用同一模板；正确性通过，但 `NM×NN × full/tail` prepared-B loader 被 Tile2D
  catalog 放大，`.text` 反而增加 76,096 bytes，立即拒绝。坑点是“复用 source
  type”并不自动复用生成代码，只要给每个 tile family 再实例化一套 K loop，体积
  仍会爆炸。

### MATMUL-SME-027：量化 shared-B 的低工作量 full-pack threshold

- 状态：保留。最终不增加 mixed specialization，而是复用已经存在的 A+B full-pack
  路径：仅当 SME、rank-3 shared B、A/B 都是 FP32->I8/U8 transform pack 时，
  把 reuse/work threshold 从 128 dots/64 Ki 降至 64 dots/8 Ki。A 仍按 batch
  逐个 pack，B 仍只 pack 一次；大 shape、非量化、rank-2、AMX 和已有 packed
  operand 的选择完全不变。
- 干净基线通过 `VECOPS_DISABLE_SME_RANK3_QUANT_FULL_PACKING` 构建，benchmark
  源码、counter 和链接目录相同。WeightReuse candidate/baseline `.text` 都是
  594,040，证明只改变 operation 构造期布尔选择，没有新增 kernel/catalog。
  A-B-B-A 严格配对结果：

| shape | baseline→SME-027 | 变化 |
|---|---:|---:|
| B=1, 1×64×128 | 5.714→5.381 us | -5.82% |
| B=2, 1×64×128 | 11.449→6.984 us | -39.00% |
| B=4, 1×64×128 | 22.870→10.216 us | -55.33% |
| B=8, 1×64×128 | 16.623→16.631 us | +0.05% |
| B=16, 1×64×128 | 29.483→29.474 us | -0.03% |

- 后两项原本已达到 64 Ki full-pack 阈值，因此是未改变控制。排除九条 B=1/2/4
  ×R=1/4/16 目标后，其余 273 条 lifecycle 控制中位数 -0.017%，范围
  -1.30%～+2.02%；BF16 与 FP32->BF16 的 raw/packed strategy 中位数都在
  -0.07%～0%，没有系统性回退。目标九条中位数 -39.00%，范围
  -55.33%～-5.79%。
- 正确性：新增 batch 单测用 B=4、1×64×128 对称量化，逐元素验证 dequant
  结果，并断言 workspace 精确等于一份 rank-2 packed A + 一份 packed B（各自
  含 63B alignment slack），证明走的是现有 full-pack 而非被拒绝的 B-only 路径。
  Native/fixed-SVL batch 各 3/3；manual-SME disassembly 2/2（14.42/14.87 s），
  region 内仍为 `BL/BLR=0`。最终 Native/fixed WeightReuse 各 282、主 Scenario
  Native 299、Packed Scenario Native 249，全部 0 error。

### MATMUL-COV-015：pipeline 隔离与固定工作量 perf 协议

- 日期：2026-08-29。状态：保留。这项工作不改 kernel，而是修复后续优化的测量
  基础设施。原 `MatmulWeightReuseBench` 同时实例化 BF16、FP32->BF16 和
  FP32->I8/U8 三条 pipeline，共 282 个 case；新增三个各 94 case 的独立目标：
  `MatmulWeightReuseBF16Bench`、`MatmulWeightReuseConvertBench` 和
  `MatmulWeightReuseQuantBench`。原组合目标继续保留，独立目标通过编译期 shard
  选择，不会把另外两条 pipeline 的 kernel/catalog 链进来。
- `VECOPS_BENCH_ITERATIONS=<正整数>` 可令 lifecycle、普通/real-batch scenario
  和 packed scenario 跳过 Google Benchmark 的 `MinTime` 自适应校准，直接执行
  固定 iteration；`VECOPS_BENCH_REPETITIONS=<正整数>` 独立控制重复次数。未设置时
  仍保持原来的 0.01/0.02 s、3 repetitions。非法、零、负数、尾随字符和超过
  `int` 范围的 repetition 都会明确报错。这样 perf A/B 不再因为候选更快而自动
  执行更多 iteration。
- 主机 `920f-4`，远端目录 `/home/ryz/vecops-neo`，BiSheng Clang 19.1.7，
  SVL=512，`-O3 -march=native+bf16+sme+sme-fa64+sme-f64f64
  -falign-functions=64 -falign-loops=64 -fno-math-errno`。SME-027 基线只额外定义
  `VECOPS_DISABLE_SME_RANK3_QUANT_FULL_PACKING`。perf 测量版的独立 Quant
  基线/候选 `.text` 均为 367,544 bytes；相对三 pipeline 组合目标的 594,040
  bytes 少 226,496 bytes（-38.13%），而基线/候选之间没有代码体积变量。最后增加
  repetition 超 `int` 范围诊断后，两侧同步变为 367,672 bytes（各 +128 bytes），
  kernel 与两侧相等关系不变。
- 固定工作量命中点为 `small/B=4/1x64x128/R=16/raw/quantize_x4`，每个进程
  10,000 iterations、1 benchmark repetition。JSON wall time 是完整 R=16 生命周期
  每 iteration 的时间：365.778 -> 163.373 us（-55.34%）；除以 16 后为
  22.861 -> 10.211 us/call，与 COV-013/SME-027 的自适应结果吻合。
- perf 使用 `taskset -c 0`，事件为 `cycles,instructions,branch-misses`，协议为
  A1-B1-B2-A2，每一项再由 `perf stat -r 3` 求均值。把两轮 A、两轮 B 再平均：

| counter | baseline | SME-027 | 变化 |
|---|---:|---:|---:|
| cycles | 7,314,729,686 | 3,280,945,294 | -55.15% |
| instructions | 11,828,731,119 | 5,151,252,489 | -56.45% |
| branch-misses | 908,428 | 1,421,957 | +56.53% |

- 结论：收益来自少执行约 56% 指令，不是频率、MinTime 校准或不同迭代数的假象。
  branch miss 绝对数反而增加约 51.4 万，说明 full-pack 路径的边界/调度控制仍有
  优化空间，但它只占约 142 misses/iteration，远不足以抵消少掉的 66.77 亿条
  指令；本机 `branches` 通用事件不受支持，因此不能从这组数据构造可靠 miss rate。
- 最终 367,672-byte 二进制再做一次 10,000-iteration 单轮复核：baseline/candidate
  为 370.450/165.110 us（-55.43%）、7.339/3.279 billion cycles（-55.31%）、
  11.829/5.151 billion instructions（-56.45%），与上表多轮协议一致。
- 门禁：本地 AMX 三个独立目标各 94/94、共 282 个结果全部 0 error；`920f-4`
  上 SME BF16/Convert/Quant Native 和 Quant fixed-SVL 各 94/94、共 376 个结果
  全部 0 error，且固定工作量结果均精确记录请求的 iteration 数。固定迭代开关
  扩展到全部 scenario 入口后，本地 AMX 主 Scenario 247/247、Packed Scenario
  228/228 也全部 0 error；最终 ARM Clang 主 Scenario 299/299、Packed Scenario
  249/249 同样全部 0 error。最终 ARM 此轮共 924 个结果。
- 测量坑点：旧的自适应 perf 会让更快的候选在同一 `MinTime` 内运行更多次，进程级
  cycles/instructions 因而不可直接比较；必须固定 iteration，或者至少按实际
  iteration 归一化。只按 wall time 看不出这个错误，后续硬件计数器 A/B 一律优先
  使用本节协议。

### MATMUL-COV-016：动态/按列 epilogue 与双 zero-point 覆盖

- 日期：2026-08-29。状态：覆盖与测量基础设施保留；三种为消除 Native 编译器
  `BL` 的 SME 候选全部因性能回退拒绝，生产 microkernel 保持原快路径。
- 新增独立 6-shard `MatmulFusionScenarioBench`，避免继续放大主 Scenario。rank-2
  shape 为 tail 19x23、GEMV tail 1x257、small 8x128、medium 64x256、wide
  32x1024；rank-3 为 independent batch tail、shared small/decode/prefill。AMX
  最终 108 case，SME Native/fixed-SVL 各 135 case，全部 fixed-one iteration、
  0 error。AMX fusion 单测 2/2，SME Native/fixed-SVL 各 2/2。
- 新增 output pipeline：运行时 scalar scale、按 N 列 scale、clamp、bias+clamp、
  U8 requantize（scale=0.125、zero-point=7）。按列 scale 使用 coordinate-aware
  `TransformContext` 从最后一个逻辑轴定位参数，再做一条连续向量 load；参数数组为
  tail 额外 padding，避免 inactive lane 越界读取。benchmark counters 单独记录
  `epilogue_parameter_bytes`。同时保留同 shape 的 Convert 与固定 Scale 控制组，
  防止把 layout/注册变化误当成动态参数成本。
- 新增双非对称 FP32->U8×U8 场景，A/B zero-point 分别为 3/5。它没有复用只适用于
  U8×S8 的单侧列补偿，而是构造完整 C prologue：
  `-Za*sum(B)-Zb*sum(A)+K*Za*Zb`，随后与原始 U8 dot 累加并乘 0.0625。
  shared-B 时 `sum(B)` 可复用，但 `sum(A)` 仍按 batch/row 生成；当前校正生成在
  benchmark 计时区外，测量 fused Matmul 本体，生命周期成本后续单列。
- 公开实现交叉检查：[KleidiAI](https://github.com/ARM-software/kleidiai) 的 SME/SME2
  matmul API 普遍把 clamp min/max 作为运行时参数，量化 family 包含 bias、scale 与
  clamp；[Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) 的
  GEMMLowp 双 offset 路径同时使用 row/column sum，并把 per-channel multiplier、
  shift 和输出上下界纳入 output stage；[oneDNN](https://github.com/oneapi-src/oneDNN)
  也把 runtime scales/zero-points/post-ops 作为 primitive 属性建模。这些公开接口
  支持本轮动态参数、按列 scale 与双校正覆盖方向。
- AtomGit [HPC-competitions](https://atomgit.com/kunpengcompute/HPC-competitions) 的
  MR refs 可通过 Git 获取。审阅 starwing `73300e3`、Overclock `5701844`、
  team10-MakerFirst `237a0ae` 等 Tensor Contraction 提交后，提取的候选包括：单次
  `smstart/smstop` 包住多 tile、缓存 packed B、thread-local aligned A、A/B prefetch、
  2-way K unroll、8 ZA tile cluster、full/edge 分离、partial tile 跳过无用 B load/
  FMOPA、ZA tile/row loop 编译期展开。竞赛代码依赖固定 shape/SVL，不能原样进入
  通用后端；后续只按独立 A/B 实验逐项迁移。

Native fusion 单测的反汇编暴露了 BiSheng Clang 19.1.7 的代码生成问题：运行时 VL
版本在 `OperandInvariants::gather_offsets_fit` 内，把本应可常量折叠的
`std::numeric_limits<int32_t>::min()/max()` 生成为 SME interval 内的普通 `BL`；
fixed-SVL 没有这个问题。尝试了三种消除方式：

| 候选 | `.text`（当时观测） | instructions | cycles | 结论 |
|---|---:|---:|---:|---|
| v1：乘法前的安全 stride bound | 2,118,464 | +2.757% | +1.723% | 拒绝 |
| v2：局部 constexpr bound | 2,289,716 | +1.285% | +1.091% | 拒绝 |
| v3：通过 scratch 向外提升 invariants | 2,300,520 | +4.069% | +7.177% | 拒绝 |

perf 目标固定为 BF16 1x257x9 bias+clamp、1,000,000 iterations、CPU0，协议是
A-B-B-A，每项 `perf stat -r3`。v1 增加范围检查/控制指令；v2 虽避免调用，但常量
形态改变了大批模板的生成；v3 又增加 scratch 状态传播和寄存器压力。三者都不是
“免费消除分支”。最终选择保留更快的原实现，并扩展 disassembly gate：Native 只
精确允许 mangled symbol 中 `numeric_limits<int>::min/max`，而且要求至少命中一次；
任何其他 `BL/BLR` 仍失败。fixed-SVL 继续严格要求 SME region 内零调用。

代码体积分析同时纠正了一个错误归因：v2/v3 测量期间通用 batch 注册器新增
shared-A 分支，导致每条 fusion pipeline 都实例化从未注册的 `BroadcastA=true`
kernel。最终 Native `.text` 一度为 2,291,764，`nm` 比 clean baseline 多 92 个
符号；为注册器增加编译期 `EnableBroadcastA` 并让 fusion 明确关闭后，降为
2,122,100，消除 169,664 bytes，符号数恢复与 baseline 相同。普通 Scenario 仍保留
shared-A case。因而 v2/v3 表中的绝对 `.text` 不能全归因于 gather 修改；表中的固定
工作量 instructions/cycles 仍可用于拒绝运行时路径。

恢复原 SME 快路径后，与 clean baseline `2,116,160-byte .text` 再做同一固定工作量
A-B-B-A：cycles 2,056,010,964.5 -> 2,052,546,779（-0.1685%），instructions
3,736,739,759.5 -> 3,736,742,350（+0.0001%），branch-misses 1,103,824.5 ->
1,116,513.5（+1.15%，绝对约 +12.7k/百万次）。结论是最终路径无性能回退；剩余
5,940 bytes 来自 clean baseline 之后保留的 shared-A/flatten 生产逻辑和模板签名，
不是失败的 gather 实验。

构建/测试坑点：

- `rsync -a` 保留源文件 mtime；当远端对象比“恢复后的本地 header”更新时，Make 会
  错误复用旧对象，曾表现为 Native 注册 135、fixed-SVL 仍只有 117。实验切换后必须
  `cmake -E touch` 变更头文件，或使用 checksum 同步并显式强制重编。
- 把 fusion 测试直接塞进原 `MatmulSMETest` 曾让单翻译单元超过 8 分钟、峰值约
  12.8 GiB RSS；拆为 `MatmulFusionSMETest` 后约 3--4 分钟，基础测试不再承受后续
  pipeline 类型增长。
- 首次 AMX fusion 单测崩溃是 helper 使用空 `ExecutionSession`、没有按
  `required_workspace()` 分配 AMX scratch；修正 workspace 后 2/2 通过，不是 kernel
  数值错误。
- GCC 交叉编译发现 batch-row packing 状态漏了实际成员，且类模板中后定义成员调用
  需要 `this->` 才能稳定通过两阶段查找；补充独立 empty state 后不会给不适用的
  operation 增加对象大小。ARM 随后又发现通用头直接点名 x86-only
  `AMX_BF16F32`；改为在 AMX implementation 条件下检查 `Atom::TA`，保持语义并恢复
  跨架构可编译性。
- 最终环境：`920f-4:/home/ryz/vecops-neo`，build
  `cmake-build-sme-pack-arm-clang-release`，BiSheng Clang 19.1.7，SVL=512；Native/
  fixed-SVL benchmark SHA256 分别为
  `57928ff579b1b1c55c8abab9186ba63f425cfe15806932f822cd7cebc81ddd93` / 
  `b5e37c92e4f246c2bbfc25fefe51946dc449eccf1f64674a2b280c3c3b898305`，`.text`
  2,122,100 / 2,016,308。两者各 135/135、0 error；单测各 2/2。manual SME gate
  各检查 16 个 region：Native 仅命中 38 个精确 allowlisted min/max 调用、unexpected
  calls=0；fixed-SVL allowed/unexpected calls 都为 0。本地 system GCC/AMX 最终
  benchmark 108/108、0 error，fusion 单测 2/2，`clang-format --dry-run --Werror`
  与 `git diff --check` 通过。

### MATMUL-COV-017：主 Matmul benchmark 固定工作量协议

- 日期：2026-08-29。状态：保留。COV-015 的 `VECOPS_BENCH_ITERATIONS` /
  `VECOPS_BENCH_REPETITIONS` 原先只覆盖 scenario/lifecycle；本轮把同一
  `configure_registered_benchmark()` 接到主 Matmul benchmark 的 runtime mode、
  fixed mode 和 fixed-policy mode 三个注册入口。未设置环境变量时仍为 0.02 s、
  3 repetitions；设置后跳过 MinTime 校准并记录精确 iteration 数。
- 本地 system GCC/AMX 用 `VECOPS_BENCH_ITERATIONS=5`、repetitions=1 验证 JSON
  精确为 5；920f-4 的后续 X07 perf 对每个 case 使用按工作量选择的固定 iteration，
  `perf stat -r3` 的 cycles/instructions 因而可以直接 A-B 比较，不再受候选执行次数
  变化影响。最终 AMX 主 benchmark fixed-one 94/94、0 error，SME 主 benchmark
  fixed-one 132/132、0 error。这个补充只改变 benchmark 配置，不改 Matmul 生产代码。

### MATMUL-SME-X07：Fusion 覆盖下重做 packed 2×2 双-bank 软件流水

- 日期：2026-08-29。状态：v2 改进保留在
  `VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE` 实验宏内，默认继续关闭；v1 与选择性
  v3 拒绝。本轮使用 COV-016 的真实 Fusion medium/wide 和主 benchmark offline
  `packed_ab`，修复 X05 只有 M=1、未命中 2×2 loop 的测量缺陷。
- 命中证据：Fusion medium 64×256×256 的 workspace 为 163,966 bytes，精确等于
  packed A+B 加两份 63B alignment slack；M/N 均跨至少两个 ZA block。全 binary
  回边扫描中，默认版 60 个短 MOPA loop 全为 4 `bfmopa`；实验版变为 40 个
  4-MOPA + 20 个 8-MOPA loop，证明目标代码实际生成。初版 O(n²) 反汇编脚本因对
  所有大跨度回边切片而长时间占用单核；最终限制回边到 320 bytes，使分析成为线性
  扫描加固定小窗口。
- v1 沿用 SME-010 的 `(kg, kg+1)` 下标表达。基线单 group loop 为 15 条指令，两个
  group 共 30；v1 虽生成 8 load/8 MOPA 且没有 vector move，但 ORR/MUL/LSL/MOV
  等第二组地址计算使 pair loop 仍恰好 30 条。主 binary `.text`
  876,248→876,568（+320）。固定 perf 中 square cycles +2.03%、instructions
  +0.057%；long-K +0.16%/+0.15%；batch projection +0.50%/+0.017%。它只减少回边，
  没减少总指令，拒绝。
- v2 把 A0/A1/B0/B1 改成四个 pointer induction variable，每 pair 递增两组 stride。
  BiSheng Clang 生成 25 条的 8-load/8-MOPA loop，相对两个基线 group 少 5 条；仍有
  两条 invariant stride `lsl` 因寄存器压力被留在 loop 内。主 binary `.text`
  876,248→876,440（+192）。offline packed 固定 perf：

| case | iterations | instructions | cycles | wall |
|---|---:|---:|---:|---:|
| acc_2x2, K=8 | 2,000,000 | -1.45% | -1.13% | -2.11% |
| square 256³ | 10,000 | **-17.84%** | **-4.53%** | **-8.42%** |
| long-K 33×35×513 | 200,000 | -7.44% | +0.13% | +0.63% |
| batch projection 128×1024×768 | 1,000 | **-20.40%** | -0.65% | -0.37% |
| skinny-K 256×256×4 | 50,000 | -2.27% | -5.55% | -3.42% |
| rank expansion 64×1024×4096 | 500 | **-20.95%** | +0.39% | +2.01% |

  长 K 已受 MOPA/访存吞吐限制：减少约 21% retired instructions 仍不保证周期下降。
  v2 主 benchmark fixed-one 132/132、0 error。
- Fusion v2 `.text` 2,122,100→2,125,748（+3,648），fixed-one 135/135、0 error。
  每项 20,000 iterations、A-B-B-A、`perf stat -r3`：

| Fusion case | instructions | cycles | wall |
|---|---:|---:|---:|
| medium BF16 convert | -7.52% | -0.34% | -2.94% |
| medium BF16 bias+clamp | -7.14% | -0.06% | -1.13% |
| medium FP32→BF16 dynamic scale | -3.73% | -3.91% | -1.11% |
| medium FP32→BF16 per-column | -3.52% | **+4.29%** | +1.54% |
| wide BF16 convert | -6.24% | **+5.60%** | +6.20% |
| wide BF16 bias+clamp | -5.97% | **+6.16%** | +5.51% |
| wide FP32→BF16 per-column | -2.53% | -0.17% | +0.99% |

  medium 的补充 epilogue 中，fixed scale/dynamic scale/narrow BF16/narrow FP16 为
  -0.22%～+0.01% cycles，per-column +0.15%，但 clamp 仍 +1.90%。同一 compute
  loop 会因 surrounding pack/epilogue specialization 的寄存器分配、代码布局和
  load/MOPA 时序出现不同结果，不能用 square 单点收益默认推广。
- v3 尝试只让 NoTransform、N≤256、K-groups≤128 使用 v2，其余走原 loop。它在
  每个 tile 内携带两套 loop 并做 runtime selection，使主 `.text` 增至 879,128
  （相对默认 +2,880）。square 仍为 -2.60% cycles，但本应排除的 long-K、batch
  projection、rank expansion 分别回退 +0.42%、+0.59%、+1.12%；拒绝并删除。
- 结论：v2 证明 pointer induction + 双-bank 能显著减少标量指令，并在 square 获得
  4.5% cycles 收益；但 wide/部分 epilogue 有 4%～6% 明确回退，无法满足默认门禁。
  最终保留改进后的 v2 作为实验宏，默认仍使用单-group loop。要安全默认化，选择
  必须提升到 traversal 外且不能复制完整 Tile2D catalog；tile 内 runtime 双 loop
  已由 v3 证明不可行。
- 最终 v2 实验 binary：主 SHA256
  `3a81aa1595022670ef6095ce263b3b40354f105944085897cef7d6217c22854e`，Fusion
  `e12505a202f6664d61d5c0fd68583b7934572362a6e84e23ba113d5bc5df43c2`。关闭宏后按
  最终源码重建，主/Fusion `.text` 精确恢复为 876,248 / 2,122,100，SHA256 精确
  恢复为 `be52edb9577cc672e620e349d71f70f0ad2b768c61d0dd51310c25982b377e68` /
  `57928ff579b1b1c55c8abab9186ba63f425cfe15806932f822cd7cebc81ddd93`，证明默认机器码
  零变化。最终 SME 主/Fusion fixed-one 为 132/132、135/135，均 0 error；Fusion
  Native/fixed-SVL 单测各 2/2，两个 manual-SME disassembly gate 均通过。

### MATMUL-SME-X08：单-group packed 2×2 pointer induction

- 日期：2026-08-29。状态：拒绝并删除。X07 v2 说明减少乘法寻址有效，但双-group
  load/MOPA 重排会让 wide/部分 epilogue 回退；因此本实验保持原 4 load→4 MOPA
  顺序，只把 `base + kg*stride` 改成 A0/A1/B0/B1 四路 pointer induction。
- 反汇编中候选 loop 仍为 15 条：四次 MUL/LSL/index 更新被四次 pointer add 和
  counter 更新替代，没有减少 retired instruction；pointer add 被排进 MOPA 之间。
  主 binary `.text` 876,248→876,056（-192），SHA256
  `88f0a713caa045c0252d7ee7c0186146704ed7a4f83f01f564d63875a926f5ca`。
- offline packed 固定 perf：acc_2x2 cycles -1.98%/instructions +0.48%，square
  -4.82%/+0.002%，long-K -0.42%/+0.18%，batch projection -0.19%/+0.001%；但
  skinny-K 已回退 +2.06% cycles，rank expansion +0.14%。显式 packed square 的收益
  来自 add/MOPA 调度，不是指令数量减少。
- Fusion candidate `.text` 2,122,100→2,122,804（+704），SHA256
  `b69c1074c621cc8ea106ef03685b372d549258d84cfc296c9f73eb09588570ce`，fixed-one
  135/135、0 error。相同 20,000-iteration A-B-B-A perf：

| Fusion case | instructions | cycles | wall |
|---|---:|---:|---:|
| medium BF16 convert | +6.25% | -0.07% | +2.21% |
| medium BF16 bias+clamp | +4.85% | +0.41% | +1.34% |
| medium FP32→BF16 dynamic scale | +3.03% | -10.38% | -10.37% |
| medium FP32→BF16 per-column | +2.86% | +1.83% | +3.88% |
| wide BF16 convert | +5.00% | **+7.60%** | +6.85% |
| wide BF16 bias+clamp | +3.81% | **+10.08%** | +10.26% |
| wide FP32→BF16 per-column | +2.05% | -1.79% | +0.39% |

- 结论：显式 packed 主 benchmark 与 operation 内 auto-pack 的生成结果明显不同；
  同一个源码 loop 在 Fusion specialization 中反而增加 2%～6% 指令。wide 回退远超
  门槛，不能只凭 square -4.8% 默认化。实验宏和代码均删除，默认仍用编译器现有的
  `kg*stride` induction；后续若继续优化地址生成，必须直接以 auto-pack Fusion
  反汇编为目标，不能用 offline packed 代理。

### MATMUL-COV-018：MatmulPack 固定工作量与 wide 生命周期覆盖

- 日期：2026-08-29。状态：保留。把 COV-015 的固定工作量协议接到 SME/AMX
  MatmulPack 的 fused registration 和 AMX compensation registration；普通 A/B pack
  注册也统一经 `configure_registered_benchmark()`。未设置环境变量时仍使用原有
  min-time/repetitions，设置 `VECOPS_BENCH_ITERATIONS` 后可直接做 perf/ABBA。
- `RepresentativeCases` 新增 1024×256 BF16 wide pack；Scenario runtime shape 新增
  32×1024×256 wide。后者同时进入 raw、offline packed A/B/AB 和 online packed
  A/B/AB 生命周期，而不增加新的 kernel 模板类型。这样能单独测出 X07/X08 中 wide
  Fusion 回退最敏感的 B 打包成本。
- 本轮源码变更后，本地 system GCC 的 `MatmulPackAMXBench-Native` 已成功链接；SME
  Native 的 wide A/B、Const/Dyn 四个 pack case 均可精确注册为 5000 iterations。
  全量 case 数和数值门禁在本轮候选结束后统一复核。

### MATMUL-SME-X09：auto-pack 热点拆解与 SME Pack 预取距离

- 日期：2026-08-29。状态：完成；默认 4-chunk 保留，关闭 auto-pack、2/8-chunk
  预取和 pair-unroll 候选均拒绝。
  目标是先回答 wide Fusion 的时间究竟耗在 packing 还是 MOPA，再只修改被证实的
  Pack 热点，避免继续用显式 packed square 代理 operation 内 auto-pack。
- `perf record -F999 cycles:u` 固定采样 BF16 bias+ReLU：wide 32×1024×256 的单个
  `execute_auto_packed` symbol 中，setup/pack 约占 49.21%，compute/store 约占
  50.51%；medium 64×256×256 分别约 39.85%/58.97%。wide 热循环的一条 pack 回边
  单独承接约 10.68% samples。默认 packed 2×2 compute loop 为 15 条：6 条标量
  地址/计数、4 `ld1b`、4 `bfmopa`、1 回边。
- 临时关闭 SME rank-2 auto-pack 后，固定工作量 A-B-B-A 明确回退：medium wall
  +193.2%、cycles +182.2%、instructions +127.8%、branch-misses +53.0%；wide wall
  +47.6%、cycles +46.2%、instructions +81.4%、branch-misses +60.4%。因此自动打包的
  cost model 不是误判；即使包含 pack 成本，连续 packed MOPA 仍远快于 raw gather。
  诊断宏已删除，不能把“省掉 packing”当作优化。
- BF16 bias+ReLU 生命周期分解（单位 µs）：

| shape | raw/auto | packed A | packed B | packed AB | online A | online B | online AB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64×256×256 | 17.068 | 16.235 | 12.571 | 11.242 | 17.569 | 17.659 | 17.609 |
| 32×1024×256 | 84.805 | 80.886 | 31.328 | 29.352 | 85.409 | 85.261 | 84.987 |

  medium 的纯 packed compute 约 11.24 µs，总 pack 约 5.8--6.4 µs；wide 的纯 compute
  约 29.35 µs，总 pack 约 55.6 µs，其中 B pack 约 51.5 µs、A pack 约 2.0 µs。
  与关闭 auto-pack 的 wide 约 117.0 µs 相比，默认仍净省约 32 µs。真正的首要热点是
  wide B pack，而不是 MOPA 本身或 A pack。
- 把 full-word/generic path 的硬编码 4-chunk 预取距离提为实验常量，默认仍为 4。
  1024×256 BF16 A/B、Const/Dyn 各固定 5000 iterations，CPU0，A-B-B-A，
  `perf stat -r3`。8-chunk 候选结果：

| pack case | CPU time | cycles | instructions | branch-misses | cache-misses |
|---|---:|---:|---:|---:|---:|
| A Const | +34.90% | +22.67% | -11.42% | +111.25% | -30.78% |
| A Dyn | +54.79% | +46.93% | -11.42% | +99.17% | -6.23% |
| B Const | +57.27% | +36.00% | -11.42% | +109.83% | -2.29% |
| B Dyn | +54.09% | +39.00% | -11.42% | +109.38% | -32.22% |

  两版 `.text` 都是 2,116,637 bytes，说明不是模板体积差异。更远且更稀疏的预取虽
  少执行 11.4% 指令、cache-miss 也没有增加，却使 IPC/分支预测和 load-use 重叠恶化，
  cycles 增加 23%--47%；“cache-miss 更少”不能推出 pack 更快。8-chunk 明确拒绝。
  2-chunk 候选让 `.text` 减少 896 bytes，但四组 instructions 都增加 5.56%、
  branch-misses 增加 9.93%--18.83%；cycles 分别为 A-Const -0.81%、A-Dyn
  +8.35%、B-Const +7.78%、B-Dyn -1.33%。收益随 specialization 翻转且两个 case
  明确回退，同样拒绝。当前保留经过实测更稳的默认 4，并把该机器上的预取敏感性作为
  后续 pack 重排门禁；接下来改测 odd/even chunk 成对展开，以直接消除热循环的奇偶
  运行时分支，而不是继续盲扫距离。
- COV-018 的本地 system GCC 门禁：AMX MatmulPack 216/216、PackedScenario
  277/277，全部固定 1 iteration、0 error。
- odd/even chunk 成对展开把每对固定成 `<ZA2,ZA0>`、`<ZA0,ZA2>` transfer，避免
  每个 chunk 的奇偶运行时选择。候选 `.text` 增加 10,880 bytes；5000-iteration
  初测 instructions -2.80%、branch-misses -28%--41%，但 B-Const cycles +2.27%。
  放大到 20,000 iterations 后，instructions 稳定为 -2.93%、branch-misses
  -37.73%--53.90%，cycles 为 A-Const -10.49%、A-Dyn +4.25%、B-Const +0.38%、
  B-Dyn +2.97%。它证明控制分支可消除，但双份 transfer body 的 I-cache/缓存副作用
  抵消了收益；而 wide auto-pack 的主热点正是 B，故拒绝并删除实现。后续若重做，
  必须用单份 out-of-line/汇编循环控制代码体积，不能继续在 header 内复制展开体。
- 920f-4 最终默认门禁：SME MatmulPack 204/204、Fusion 135/135、PackedScenario
  305/305，全部固定 1 iteration、0 error。Pack benchmark SHA256
  `d13d1d8e80ef7ace92caa68b9102cda9080881e97bab6412f00bb61a19972898`；Fusion
  SHA256 精确恢复为 X07/X08 前的
  `57928ff579b1b1c55c8abab9186ba63f425cfe15806932f822cd7cebc81ddd93`，证明所有被
  拒绝候选关闭/删除后默认 Fusion 机器码零变化。PackedScenario SHA256
  `11b9c8faa51a7ac122aef48987b29f1565b9633b4a747e7ee5cbd865d2a67277`。
- 用于扫描 2/8-chunk 的外部实验宏最终删除，只保留单一
  `inline constexpr PrefetchChunks = 4`，避免 rejected tuning knob 长期进入公共构建面。

### MATMUL-SME-X10：共享 ZA 的 out-of-line Pack 边界

- 日期：2026-08-29。状态：拒绝并删除。X09 的 pair unroll 证明减少分支有效，但复制
  transfer body 会放大 I-cache；因此本轮尝试每次完整 pack 只做一次 noinline 调用，
  让多个 Fusion specialization 共享单份 Pack 热循环。
- [Arm ACLE](https://arm-software.github.io/acle/main/acle.html) 规定
  `__arm_streaming_compatible` 不切换 streaming mode，`__arm_inout("za")` 让 callee
  与 caller 共享 ZA；概念上正适合此处。但 vecops 当前 owner 是普通 C++ 函数内手写
  `smstart/smstop`，编译器类型系统不知道它拥有 ZA。BiSheng Clang 19.1.7 的最小
  实验会直接报错：调用 shared-ZA 函数要求 caller 有 ZA state。
- 把 owner 改为 `__arm_locally_streaming __arm_new("za")` 虽能合法调用，最小函数已
  生成 8 个 D 寄存器 spill/reload、`TPIDR2_EL0` 检查、可能的
  `__arm_tpidr2_save`、额外 `zero {za}` 与 SM/ZA 分别开关。它破坏当前“一次轻量
  手工 region”的核心假设，不能仅为得到合法 call 边界直接替换。
- 架构约束：SME 指令必须继续走 `vec::details::sme` 封装；不引入 `<arm_sme.h>`、
  ACLE 原生 SME intrinsic 或 ACLE 强制状态管理。上面的 attribute 实验只用于确认
  ABI 不兼容原因，不是实现候选。
- 下一候选仅在实验宏下使用普通 noinline 函数，延续项目现有 raw-asm SME 编译器
  契约，函数体内部仍只调用 vec SME 封装；必须同时满足：caller region 只出现精确
  allowlist 的 pack call，callee 内无
  普通调用、无 `smstart/smstop`、无意外 ZA save/restore，且 Pack/Fusion perf 不回退。
  任一条件失败就删除，不能把非标准 ABI 假设默认化。
- Pack benchmark 候选总 text 2,116,637→1,873,533（-11.49%），SHA256
  `c871c1885f8d67cf65a5ebdabdd79244e1c45947d540eb67ad174941a96c027e`。反汇编得到
  16 个按 atom/side 共享的 pack helper；每个内部 calls=0、state ops=0，BF16 helper
  只保存 x19--x30 等标量 callee-saved 寄存器，没有 D/Z/P spill、SM 切换或 ZA
  save/restore。fixed-one 204/204 可执行、0 error。
- wide BF16 Pack 20,000-iteration A-B-B-A 中，instructions 四组都仅 +0.01%，即一次
  call/prologue 的固定成本；但 cycles 为 A-Const -3.64%、A-Dyn +16.86%、B-Const
  +1.59%、B-Dyn -11.97%，cache-misses 也随 specialization 翻转。这再次说明独立
  Pack benchmark 对 code placement 很敏感，不能用它单独决定默认化；正在以真实
  auto-pack Fusion medium/wide 作为最终性能门禁。
- Fusion 总 text 2,280,517→2,183,845（-4.24%），SHA256
  `ab6b53f1839712adf922774eb1f4c99b357facc74e3b812c675069fcc0180541`；只生成
  BF16/F32 A/B 四个共享 helper，内部同样 calls=0、state ops=0，fixed-one
  135/135、0 error。20,000-iteration、A-B-B-A、`perf stat -r3` 的六次 wall/CPU
  样本均值：

| Fusion case | wall | CPU | cycles | instructions |
|---|---:|---:|---:|---:|
| medium BF16 convert | -0.39% | -0.40% | -1.00% | +0.01% |
| medium BF16 bias+clamp | -2.93% | -3.59% | -3.90% | +0.08% |
| medium FP32 dynamic scale | **+6.80%** | **+6.69%** | **+6.42%** | 0.00% |
| medium FP32 per-column | -3.76% | -3.56% | -3.64% | 0.00% |
| wide BF16 convert | -2.88% | -2.66% | -2.74% | -0.01% |
| wide BF16 bias+clamp | -1.77% | -1.79% | -1.85% | +0.08% |
| wide FP32 per-column | -1.01% | -1.11% | -1.05% | 0.00% |

- medium FP32 dynamic scale 在 CPU0 的额外 5 轮 ABBA 一度收敛到 wall +0.94%/
  cycles +0.76%，但该核单次 37--57 µs、噪声过大。换到 CPU100 后再做 5 轮：wall
  +5.84%、CPU +5.82%、cycles +5.61%、branch-misses +11.31%、instructions 仍为
  0.00%，确认是代码放置/前端回退而非新增工作量。这个 transform-pack 路径并不直接
  使用新 helper，却会因整 binary 收缩改变布局；说明 out-of-line 不能只看目标调用者。
- 结论：4.2% Fusion 体积收敛和多数 case 的 1%--4% 收益不足以抵消一个稳定 5%--6%
  回退；普通 noinline 边界还会放宽当前 manual region 的“全内联、零调用”契约。
  实验宏与代码均删除，维持 vec SME 封装 + 完全内联状态模型。若以后重试代码共享，
  必须同时控制 shard/link placement，并先解决不相关 transform specialization 的布局
  扰动，不能用 ACLE 强制状态管理绕开当前架构。

### MATMUL-SME-X11：auto-pack owner 256-byte entry alignment

- 日期：2026-08-29。状态：编译期拒绝并删除。X08--X10 多次出现“instructions 基本不变，但
  specialization 因全 binary 布局变化而翻转 5%--10% cycles”的现象；当前 benchmark
  目标虽统一带 `-falign-functions=64`，`execute_auto_packed` 自身只依赖该 64B 下限。
- 候选仅在实验宏 `VECOPS_EXPERIMENTAL_MATMUL_AUTO_PACK_ALIGN_256` 下给这个 noinline
  owner 增加 `VECOPS_FUNCTION_ALIGN(256)`；不改 Pack/Matmul 指令、vec SME 封装、
  `StreamingZARegion` 或函数调用图。验证内容包括所有 owner symbol 的实际地址模 256、
  总 text padding，以及与 X10 相同 7 个 Fusion medium/wide 固定工作量门禁。
- BiSheng Clang 19.1.7 在实例化阶段失败：原本正常的
  `matmul_bound<Atom,TilePolicy,bool>(scope,...)` 被误报为显式 `Scope` 模板实参无效。
  预处理输出精确为
  `__attribute__((noinline,aligned(256))) void execute_auto_packed(...)`，没有宏泄漏；把
  `aligned(256)` 从声明前缀改到 `const` 后缀仍得到同一组错误。删除 attribute 后同一
  attribute 后错误仍会在强制重编中复现；因此不能只归因于 aligned，而是该巨大 TU
  中 constrained function-template overload viability 的前端脆弱性被本实验暴露。
- 恢复办法是把 `matmul_bound` 的 `execution::ExecutionScope Scope` 形参改为普通
  `typename Scope`，并在函数体首行做同一 `static_assert(ExecutionScope<Scope>)`。
  约束语义不变，但避开 overload viability 阶段；BiSheng 随即恢复 6-shard 构建。
  最终 Fusion text 仍为 2,280,517，SHA256 精确恢复
  `57928ff579b1b1c55c8abab9186ba63f425cfe15806932f822cd7cebc81ddd93`，证明 workaround
  不改变机器码。远端 Clang fixed-one 135/135、本地 system GCC/AMX fixed-one
  108/108，均 0 error。
- 结论：对齐候选无法安全进入性能测量，实验宏和代码删除。该编译器上函数模板的定点对齐目前
  只能依赖目标级 `-falign-functions=64`；不要把 `VECOPS_FUNCTION_ALIGN` 直接加到
  `execute_auto_packed`。若未来编译器修复，再按本条的 7-case 门禁复测。

### MATMUL-SME-X12：拆分 full-word pack 的预取与尾部 loop

- 日期：2026-08-29。状态：拒绝并删除。默认 full-word loop 每个 chunk 先判断奇偶 ZA
  bank，再判断 `chunk+4<chunks` 选择带/不带 prefetch 的 transfer。X09 pair-unroll
  同时消除两层分支但复制整个 pair body，导致 I-cache 回退；本候选仅拆为“prefetch
  prefix”和“no-prefetch tail”两个 loop，各自仍保留单份奇偶选择。
- Pack benchmark 总 text 2,116,637→2,086,733（-1.41%），SHA256
  `19a02b4832cb571a99adf9de32d6fb70ebc19b2c2d34cc9527ae849ca7eb2a9e`。
  wide BF16、CPU100、20,000 iterations、A-B-B-A、`perf stat -r3`：四组
  instructions 均 -3.01%；A-Const/A-Dyn/B-Dyn 的 CPU 分别 -13.12%/-9.43%/
  -11.19%，cycles -12.91%/-9.10%/-10.87%。B-Const 初测 CPU +2.39%，但独立 5 轮
  ABBA 复核为 CPU -3.20%、cycles -3.18%、branch-misses -28.55%，确认无稳定回退。
- 与 pair-unroll 不同，本候选同时缩小代码、减少 retired instructions 和分支错误，
  说明“按预取阶段拆 loop”比“按 ZA bank 成对展开”更适合当前编译器。正在构建真实
  Fusion，最终仍以 medium/wide transform/epilogue 交叉门禁决定是否默认化。
- Fusion text 2,280,517→2,268,869（-0.51%），SHA256
  `e4502b1b89e6340c94f6e1f99031f708f1da7d4f929586574bef056efd096d50`，fixed-one
  135/135、0 error。CPU100、20,000-iteration A-B-B-A：

| Fusion case | CPU | cycles | instructions | branch-misses |
|---|---:|---:|---:|---:|
| medium BF16 convert | +1.87% | +1.90% | -1.23% | +10.52% |
| medium BF16 bias+clamp | **+2.98%** | **+2.87%** | -1.08% | +16.49% |
| medium FP32 dynamic scale | +0.75% | +0.76% | 0.00% | +1.03% |
| medium FP32 per-column | -2.79% | -2.74% | 0.00% | +11.30% |
| wide BF16 convert | -3.05% | -3.04% | -1.66% | +11.50% |
| wide BF16 bias+clamp | -1.50% | -1.50% | -1.44% | +1.69% |
| wide FP32 per-column | -1.09% | -0.86% | 0.00% | +1.50% |

- 最差的 medium BF16 bias+clamp 另做 5 轮 ABBA：CPU +2.25%、cycles +2.20%、
  instructions -1.08%、branch-misses +5.40%。因此不是首轮单点噪声；split loop 虽在
  wide 与独立 Pack 上有效，却让 medium 的分支预测/前端尾延迟恶化。不能仅按
  `spatial>=512` 增加双路径选择：X07 v3 已证明同一 owner 内携带两份 loop 会使不相关
  specialization 受布局影响。实验代码删除，默认继续使用单 loop。

### MATMUL-COV-019：F16/F64 与窄转换 Fusion 交叉覆盖

- 日期：2026-08-29。状态：本批完成。权威注册审计发现主 Scenario 已有 SME F16/F64、
  FP32→BF16/F64-kernel、BF16/FP16→F32-kernel，但专门的 Fusion target 仍只覆盖
  BF16/F32、两类整数和 AMX F16 原生输入；测试也没有 F16/F64 动态/per-column
  scale 与 bias+clamp。
- Fusion target 从 6 拆为 10 shard。新增 SME：F16 原生输入的 clamp/dynamic scale/
  per-column scale；F64 原生 FP64 和 FP32 memory conversion 的同三类 epilogue；
  F32 kernel 的 BF16→BF16、FP16→FP16 bias+clamp。AMX 在硬件支持 FP16 时新增
  FP32 memory→F16 kernel 的三类 epilogue；不支持时新增 shard 为空。
- `check_runtime_and_per_column_scale` 从只接受 FP32 accumulator 泛化为任意浮点 Acc，
  参数/输出/scales 都使用 Acc，因此同一数值 oracle 可验证 F64。SME Fusion test 新增
  F16 runtime/per-column/bias-clamp 和条件 F64 对应检查；AMX test 条件新增 F16。
- 本地 system GCC/AMX 已构建 10-shard benchmark 与测试：fixed-one 108/108、0 error，
  单测 2/2。920f-4 最终 fixed-one 234/234、0 error，单测 2/2；按 atom 为
  BF16 90、F16 27、F32 45、F64 54、两类 I8 各 9。benchmark text 3,700,349，
  SHA256 `777b29b1764160e1f12a346e9f3ede29fe5f83763281172b8024df9813b4fb73`。
- 扩展过程中发现 dynamic/per-column 参数被硬编码为 FP32；已改为
  `ScenarioOutputParameters<Acc>`，运行时参数、列数组、reference oracle 和 bytes
  counter 均跟随 accumulator 类型。否则 double tag 会尝试 load float pointer，F64
  Fusion 无法实例化。
- 独立 MatmulPackSMEBench 新增 FP32→FP64 的 A/B、Const/Dyn 四组 Vector 基线，明确
  只走通用 vec/DataAccess conversion，不错误注册尚不支持该 widening conversion 的
  ZA postprocess。920f-4、固定 1000 iterations：A 64×256 Const/Dyn 为
  20.1/23.5 µs，B 256×256 为 77.7/93.4 µs；吞吐分别约 9.10/7.81 和
  9.42/7.84 GiB/s。这四组将作为下一步 FP32→FP64 pack 优化的直接门禁。

### MATMUL-SME-X13：FP32→F64 conversion auto-pack

- 日期：2026-08-29。状态：保留。COV-019 固定 1000-iteration 基线中，F64 kernel
  原生 FP64 medium/wide clamp 分别约 184/427 µs，而 FP32 memory→FP32 output 为
  446/986 µs，且后者 `workspace_bytes=0`。这证明每次 MOPA traversal 都在直接转换/
  gather，没有走 packing。
- 候选把 SME `AutoPackOperand::SupportedConversion` 扩展到
  `Memory=float32_t, Element=float64_t`。打包实现复用现有 Vector MatmulPack：DataAccess
  用 double tag 做 load-convert，写入 F64 packed layout；随后现有 F64 packed kernel
  连续消费。没有 ACLE SME intrinsic 或状态管理变更。正在构建 10-shard Fusion 并与
  SHA256 `777b29...fb73` 的扩展覆盖基线做 A-B 性能比较。
- 第一版 fixed-one 234/234、0 error，目标 workspace 从 0 变为 medium 655,486B、
  wide 2,162,814B。CPU100、1000-iteration A-B-B-A：medium clamp/dynamic/per-column
  分别 -38.90%/-35.25%/-40.62% CPU，wide -21.39%/-19.38%/-21.45%；instructions
  下降 18%--40%，cache-misses 下降 43%--66%。BF16 medium/wide 5 轮控制为
  +0.23%/-0.66%，native F64 的目标 owner 地址与大小在两版中精确相同。
- 初版 rank2 small 8×128×128 回退 38.5%，说明通用双边 pack 的 64K work threshold
  对 FP32→F64 过低；但 shared-B batch 的相同 small、decode、prefill 分别提升
  64.2%/55.5%/59.3%，因为 B conversion 可跨 batch 复用。第二版仅把 rank2
  F64-conversion pair 的 threshold 提到 256K，rank3 shared-B 继续使用原阈值；正在
  重建并复核 small workspace/性能和 medium/wide 收益。
- 第二版 fixed-one 中 tail/gemv/small/independent-batch workspace 均恢复为 0；medium/
  wide 为 655,486/2,162,814B，shared-small/decode/prefill 仍打包。5 轮 ABBA 复核：
  rank2 small CPU 0.00%、instructions 0.00%；medium clamp CPU -37.33%、instructions
  -40.88%、cache-misses -67.16%；shared-small CPU -64.11%。因此阈值同时消除了已知
  回退并保留主要收益。
- 非目标控制：BF16 medium/wide 5 轮 CPU +0.23%/-0.66%；native F64 +1.39%，但
  baseline/candidate 中 native F64 owner 的地址、大小与机器指令不变，instructions
  -0.00%，属于运行频率噪声而非路径改变。目标六 case 的收益远高于噪声带。
- 新增显式 Vector Pack A/B oracle，验证 FP32 load-convert 到 F64 packed layout；新增
  large 64×256×256 end-to-end conversion 必须申请 workspace 的数值检查。为避免基础
  `MatmulSMETest` 回到 9GiB/6min 单 TU，该大类型放进专门的 Fusion test。最终
  MatmulPackSMETest 6/6、MatmulFusionSMETest 3/3，manual SME disassembly gate 通过；
  全部实现继续使用 DataAccess/vec SME 封装，无 ACLE intrinsic。
- 最终扩展 Fusion text 3,792,869，SHA256
  `ace49623da3a4e9e71ae8ff6316d6cbddf62a93f59245bdaa720ecf1490ed4d5`；fixed-one
  234/234、0 error。

### MATMUL-SME-X14：统一 SME wrapper 边界

- 日期：2026-08-29。状态：保留。按项目状态管理约束审计 SME 代码，kernel/pack/test
  中未发现 `<arm_sme.h>`、ACLE SME intrinsic 或 ACLE streaming/ZA attributes；唯一
  遗留是 `MatmulSMEBench.cpp` 为读取 SVL 直接包含 `<arm_sme.h>` 并调用
  `svcntsw/svcntsd`。
- benchmark 已改用 `vec::details::sme::streaming_lanes<float32_t/float64_t>()`。该封装
  通过项目统一的 `rdsvl` 路径读取 streaming vector length，不把 ACLE 的强制状态
  语义带进调用者。后续 SME kernel、pack、benchmark 与 test 一律以 vec SME wrapper
  为边界；需要的新指令先补 wrapper，再由算子调用，禁止直接使用 ACLE SME intrinsic
  或 streaming/ZA ownership attribute。
- 920f-4/BiSheng Clang 19.1.7 重建成功；MatmulSMEBench fixed-one 132/132、0 error，
  `.text=1,000,065`，SHA256
  `4b4b34e155ba695e23ae09ec5b5926ae1f7011f349d8f68d950fae23630c36d2`。
  `nm -u` 没有 `__arm*`、TPIDR2 或 SME runtime symbol；反汇编的 SVL 查询是直接
  `rdsvl`。源码审计也不再发现 `<arm_sme.h>`、原生 SME intrinsic 或 ACLE SME
  attribute（历史 Markdown 中对被拒实验的说明不计入可执行代码）。

### MATMUL-SME-X15：FP32→FP64 的 ZA32 transpose-convert pack

- 日期：2026-08-29。状态：保留。X13 让
  FP32 memory→F64 kernel 进入 auto-pack 后，真正的 pack 仍是通用 Vector/DataAccess：
  它按空间轴读取 row-major 输入，920f-4 反汇编为每个 K/panel 两次 indexed
  `ld1w [base,z*.s,sxtw]` gather，再转换和写出。A 64×256 只有约 7.8--9.2 GiB/s，
  B 256×256 为 8.0--9.5 GiB/s，成为新的主要瓶颈。
- 新实现 `SMEFP32ToFP64` 每次用一个 ZA32 tile 做 16×16 blocked transpose：沿每个
  source row 连续 `load_hor<uint32_t>`，按 column `read_ver<uint32_t>`，bitcast 为
  FP32 后用 `vec::convert` 扩成完整 16-lane FP64 panel 并连续 store。空间/K tail
  分别由 read/load predicate 处理，inactive spatial lane 由 vec wrapper 的 zero+merge
  语义写成 +0。整个实现只调用 vec SME/向量封装，继续由现有
  `StreamingZARegion` 持有状态；没有 ACLE SME intrinsic/attribute。
- selector 只对 NoTransform、FP32 memory、FP64 compute、unit inner stride 选择新路径；
  padded row stride 允许，inner stride≠1 保留 Vector fallback。独立 packed-layout
  oracle 首轮 A/B、19×17 tail 通过；正在扩成 full/spatial-tail/K-tail/both-tail ×
  Const/Dyn，以及 padded-row/inner-stride fallback 矩阵。
- 同一 candidate binary 的 fixed-5000 初测：A Const/Dyn 20.0/23.3→6.69/6.73 µs，
  B Const/Dyn 77.2/92.0→36.9/26.5 µs。随后固定 CPU100、10,000-iteration、
  baseline-candidate-candidate-baseline：A Const/Dyn 分别 -63.72%/-74.29%，B
  Const/Dyn -63.18%/-67.49%。
- `perf stat -r3`、20,000 iterations：A Const cycles/instructions/cache-misses
  -66.87%/-69.64%/-19.14%；B Dyn -65.53%/-73.54%/-2.72%。branch-misses 虽分别
  增加约 63%/121%，但 retired work 和 cycles 的下降远大于该代价。
- 目标 helper 反汇编为连续 `ld1w {za0h.s}`、vertical ZA read、两条有序 FP64
  `fcvt` 和两条连续 `st1d`；helper 内 calls=0、indexed gather=0，只在现有 resource
  边界各一次 `smstart/smstop`。下一门禁是 X13 的 rank2 small/medium/wide、rank3
  shared-B/independent case 与原生 F64/BF16 控制，确认 pack 本体收益能转化为完整
  Matmul 收益且不会因代码布局回退。
- 单 ZA 版的完整 Fusion ABBA 已通过：rank2 medium 三种 epilogue 平均 -33.36%
  （-37.09%~-30.98%），wide 平均 -43.58%；rank3 shared-small/decode/prefill 分别
  -52.96%/-38.72%/-30.85%。不打包的 tail/gemv/small/independent-batch 分组均值
  -0.03%/+0.60%/-0.07%/+0.58%，最差单项 +1.76%。原生 BF16 medium 的隔离 8-run
  control 六种 epilogue 在 -0.87%~+0.16%；原生 F64 shared-decode 的短测有明显双峰，
  20,000-iteration perf ABBA 中两版 instructions 只差约 2e-6%，cycles 快慢随运行
  顺序翻转，因此没有新增工作量证据。
- 按既定软件流水方向继续做 ZA0/ZA1 双缓冲：先 charge ZA0；drain 当前 tile 的每个
  vertical column/FCVT/store 时，同时向另一 tile 装下一 K block 的对应 source row，
  最后 drain。forced simple/pipelined、CPU100、20,000-iteration ABBA 的八项全部加速：
  35×17 tail -3.64%~-8.78%，A64×256 -23.81%/-19.01%，B256×256
  -6.30%/-18.33%。`perf stat -r3` 中 A-Const cycles/instructions/branch-misses/
  cache-misses -12.83%/-9.65%/-18.81%/-69.09%，B-Dyn
  -7.94%/-10.18%/-24.86%/-65.56%。因此流水版替换单 ZA 版，实验 implementation
  tag 和重复 backend 已合并删除，只保留统一 `SMEFP32ToFP64` 生产抽象。
- 扩展正确性门禁：Native 与 FixedStreamingSVE 各 6/6，覆盖 A/B × Const/Dyn ×
  full/spatial-tail/K-tail/both-tail、padded row stride，以及 inner stride=2 必须选择
  Vector fallback。单 ZA 阶段的全量 Pack/Fusion/MatmulSME 分别 220/234/203 case、
  全部 0 error；最终生产重建的 Native/FixedStreamingSVE Pack 仍各 6/6，并且两套
  manual SME disassembly gate 都通过：各 37 个 SME region、region 内 unexpected call=0。
- 全量双 ZA Fusion 发现 shared-decode 相对单 ZA 回退约 25%，而 medium/wide/
  shared-prefill 有 3%--13% 额外收益。根因是每 batch 的 A 只有一个 active spatial
  row：只能交叠一条下一块 load，却仍执行完整双 bank transfer 控制。把 partial loop
  直接并入同一 owner 虽让 shared-decode 相对全双 ZA 恢复 14.85%，却因代码体积/
  寄存器分配扰动让 full medium 回退 9.44%；拆成两个 resource owner 又让 wide 回退
  7.34%，两版均拒绝。
- 最终分层不改变公开 Pack selector：普通 `SMEFP32ToFP64` 继续双 ZA；只在 Matmul
  rank3 shared-B、FP32→F64、A 的 `M<panel` 时，由 orchestration 内部强制紧凑单 ZA
  A-pack，shared B 和所有 rank2 仍走双 ZA。相对全双 ZA，shared-decode -12.71%；
  相对原单 ZA，shared-small/decode/prefill -2.80%/-13.86%/-3.32%，medium/wide
  -2.86%/-3.68%。tail/gemv/small/independent 分组都在 ±0.2% 左右。该内部 tag 不进入
  用户选择面，只解决已证实的 skinny crossover。

### MATMUL-COV-020：tiny/GEMV/shape-aware dispatch 的 AMX-SME 对等网格

- 日期：2026-08-29。状态：第一批保留。对照 `matmul-x86.md` 和 SME backend 后确认：
  两端 primary BF16 已共享 representative/workload catalog，Scenario 也已有 shared-B
  tail/tiny/small/decode/大权重与 shared-A scalar/tail/boundary；但无 batch 的 tiny/
  GEMV 只有少数孤立 shape，无法系统测出 AMX 的 AVX-512 crossover 与 SME 始终 MOPA
  的差异。
- 新增架构公共 `dispatch_shape_cases`，AMX_BF16F32 与 SME_BF16F32 同时注册完全相同
  的 23 个数学 shape、71 个 input-mode case：tiny 1×8/8×1/2×4/4×2/4×4/8×8/
  16×16；M=1 与 N=1 两个方向的长边 16--1024、K=65--4097；以及 tile+短尾配
  K=1024/1025。包含 Raw、PackedB、PackedA、PackedAB 对照，所有尺寸均为 runtime，
  因而增加 case 而不复制 Matmul kernel template catalog。
- 本地 system GCC/AMX 与 920f-4/BiSheng Clang/SME 新网格 fixed-one 均为
  71/71、0 error；SME 主 Matmul 全量由 132 增至 203/203。第一项 tiny
  `1×8×64` 的单次观察中 raw 约 9.71 µs、packed-AB 1.97 µs，已明确暴露进入/退出
  SME 与 raw loader 对极小输出的固定成本；正式 SVE crossover 仍需固定工作量网格，
  不能用该单次值决定阈值。该网格将作为后续 SVE-vs-SME tiny/GEMV dispatch 的生产
  门禁。
- 实现审计还确认两个 P0 调度缺口：公共 facade 的 shared-B batch×M flatten 和
  shared-A batch×N flatten 都被编译期限定为 AMX；SME 当前虽能把 shared B 只 pack
  一次，却仍逐 batch 运行部分空闲的 ZA problem。下一阶段会先用现有 Scenario
  shared catalog 做 SME baseline/crossover，再分别放开两种 flatten，不能直接照搬
  AMX 阈值。

### MATMUL-SME-X16：shared operand 的 batch 维合并

- 日期：2026-08-29。状态：实验与门禁进行中。覆盖审计确认 Scenario 虽已有 shared-B
  的 tail/tiny/small/decode/大权重和 shared-A 的 scalar/tail/boundary，但 SME facade
  一直逐 batch 启动一个只占少数 ZA 行/列的问题；x86 已能把 dense shared-B 的
  `[batch,M,K]` 合成 `[batch*M,K]`，并在 `M=1` 的 shared-A 情况把 B/C 合成
  `batch*N` 列。
- 第一候选只放开语义可证明的保守子集：shared-B 要求 raw dense A、dense C、
  elementwise A/C transform、`M<=16` 且 `batch*M<=64`；shared-A 要求 shared raw/
  PackedA、raw mergeable B、dense C、elementwise transform、`M=1`、`batch>=4` 且
  `batch*N<=64`。AMX 原有门禁与配置复用保持不变，SME 合并后直接进入普通 backend，
  仍只通过 vec SME wrapper 持有 streaming/ZA 状态。
- shared-A 的 `{0,0,1}` per-N bias 不能用 affine layout 表示合并后的周期寻址，因此在
  workspace 中物化至多 64 个 accumulator。审计发现实验初版在 SME 路径没有 rewind
  该 bump allocation；已在 flattened-column 调用内部加 mark/rewind。测试同步改为同一
  workspace 连续调用两次，每次断言 `used()==0`，并断言精确需求为
  `batch*N*sizeof(Acc)+63`；同时补 PackedA+bias。
- 性能基线保存在 920f-4 `/home/ryz/tmp/x17-scenario-no-sme-flatten-baseline`，SHA256
  `036565e6fc023ac7d8df06f253d0877b6ab37da5c7a8bf721bb775862be1d441`，text
  3,676,999。固定 CPU、2000 iterations 的 shared catalog 共 207 case、0 error；
  正在用修正后的候选做 A-B-B-A，未通过 correctness、workspace 和分组回退门禁前不
  移除实验宏。
- v1 fixed-one 为 207/207、0 error/0 skip，BatchSME 4/4；CPU100、2000-iteration
  B-C-C-B 的 207 case 总几何平均耗时下降 58.66%。shared-A scalar/tail/boundary 分别
  -93.60%/-78.93%/-40.92%；未触发 full auto-pack 的 shared-B tail/tiny 分别
  -73.65%/-77.26%。shared-B small/small-long-K/wide/decode 的分组均值仍有
  6.6%--8.3% 收益，主要来自 FP32→F64 conversion，但 native BF16/F16/F32 多数只在
  ±2% 内，wide-short-K 六个 BF16 pipeline 反而稳定回退 2.62%--4.50%。
- 根因不是 flatten 本身：这些 native shape 的既有 cost model 同时 auto-pack A/B；v1
  的 `AutoPackA&&AutoPackB` 分支仍复用一个 panel-sized A buffer逐 batch运行，所以没有
  合并 problem，新增 fallback 机器码只扰动原路径布局。v2 在 cost model 已决定双边
  packing 时，改为一次 pack 完整 `[batch*M,K]` A 和共享 B，再执行一个 packed problem；
  cost-model 拒绝 packing 的 SME tail/conversion 则保留 direct flattened leaf。v1 binary
  保存为 `/home/ryz/tmp/x17-candidate-v1`，SHA256 `726206e25b9ffaa5d90b30d489534cb1c16821310ef50242e7959c5bf8daeb77`；
  Berkeley text 4,525,875，较 baseline 3,676,999 增加 848,876B。v2 正在重建并会重新
  做 207-case ABBA；若 native regressions 不消失则按 shape/cost gate 收窄。
- v2 fixed-one 仍为 207/207、0 error/0 skip；相对无 flatten baseline 的 207-case
  几何平均耗时下降 78.02%，九组的最弱单项仍下降 19.94%，没有运行时回退。shared-B
  tail/tiny/small/small-long-K/wide-short-K/decode 的分组分别 -85.56%/-86.66%/
  -72.90%/-73.23%/-68.84%/-64.98%；shared-A 三组为 -93.60%/-79.01%/-41.58%。
  相对 v1，双边 full-pack 修复让四个 shared-B 中大组再下降 61.8%--71.1%，原 native
  wide 2.6%--4.5% 回退完全消失。
- v2 binary 保存为 `/home/ryz/tmp/x17-candidate-v2`，SHA256
  `dad0e680309990091c68286f6b0be45e1a1feab51bad3db943584eaebd0846e6`，Berkeley text
  4,971,051；相对 baseline +1,294,052B（35.2%），相对 v1 +445,176B。这个 catalog
  同时实例化 23 种 dtype/pipeline，每个真实模型类型只承担自己的 specialization，但
  峰值编译约 11GiB 仍是明确代价。算法按零回退门禁保留，后续结构优化应把 full-pack
  orchestration 拆成 noinline owner并重新核对 `.text`，不得回退到逐 batch problem。
- 生产源已移除 `VECOPS_EXPERIMENTAL_SME_BATCH_FLATTEN`：SME 与 AMX 共享同一静态语义
  门禁，差异只在 AMX 配置/pack-A cost model和 SME 的 packed-flat owner。最终测试新增
  shared-B M=16/17、flat-M=64/>64、zero-batch raw/PackedB；shared-A batch=3/4、
  N=16/17、M=1/2，以及 Raw/PackedA periodic-bias 连续调用和 workspace 精确回收。
- 最终 920f-4 生产门禁：BatchSME 5/5、FusionSME 3/3、Pack Native/Fixed 各 6/6；
  MatmulSMEBench 221/221、0 error/0 skip。SHA256：Batch
  `982031371a40972b0c80ca64311cbc90b7457335b1c5e360d397604768a73ea3`，Fusion
  `2b06bc0ba4608a02d71d6dfecf46ff4ba831a95923ccf6a1f057c917fbf87941`，Pack Native
  `80107420b3af92b706032c874787ddafb47ff8611399c2943cca0c46c0c510ab`，Fixed
  `7a453cfbb9f2f52693fa85591eed6180ad6e9127d03a74351ba1ab4e0555d7c7`。

### MATMUL-SME-X17：tiny/GEMV 的普通 SVE fallback 设计审计

- 日期：2026-08-29。状态：设计完成、尚未改生产路径。COV-020 的 71-case 对等网格
  证明 SME 的 raw/PackedA/PackedB/PackedAB tiny、M=1、N=1 覆盖已与 x86 对齐，但
  当前全部进入 StreamingZA/MOPA；`1x8x64` 的 raw 单次约 9.71 us，而 PackedAB 仍约
  1.97 us，固定状态成本清晰可见。
- 可行的第一阶段是在获取 `StreamingZARegion` 之前分派 raw BF16/F32/F64 的
  `M=1 || N=1`：predicated contiguous load、扩宽、FMA、`reduce_add`，再通过
  DataAccess 完成 C input transform、累加和 C output transform。已有 vec API 没有
  BF16 widening-dot 或 INT8 SDOT/UDOT 抽象，所以 BF16 只能生成 BF16→FP32 扩宽加
  FP32 FMA，INT8 第一版预期性价比较低；不得用 ACLE intrinsic 绕过 wrapper。
- packed operand 不能当连续 K 行读取：SME BF16 格式是 `[panel32,kg,panel,2]`，固定
  output row 跨 kg 的步长为 64 个 BF16。第二阶段应反过来沿 packed 的变化 output 维
  一次计算多项，并正确处理普通 VL 与 SVL 不等；在 raw crossover 证据出来之前不实现
  该复杂路径。初始阈值只测 output<=64，Block1 与 Block4 都保留 forced benchmark，
  由 71-case 网格决定最终 dispatch，而不是照搬 AMX 阈值。
- 实验 leaf 已按上述边界落地为 64B-aligned noinline compute：每次 K traversal 复用
  singleton operand，并按 4 个输出分组；K tail 只用 predicated zero load，没有标量
  oddment。它只在调用 scope 尚未持有 StreamingZA 时允许命中，避免 auto-pack 已经打开
  streaming region 后又切回普通 SVE。除 BF16 的 71-case 网格外，F16/F32/F64 各新增
  `1x8x64`、`8x1x64`、双向 16x65 和双向 64x257 共 18 个 raw probe；实验宏只在
  MatmulSMEBench 定义，完成 A-B 后再决定生产门槛和是否扩展 DataAccess fusion。
- 920f-4、CPU100、10,000-iteration、candidate-candidate 与独立同源 disable binary
  baseline 各两轮：BF16 双向 8/16/32/64 输出分别约 -71%、-31%、-30%、-14%~-20%；
  F32 六点为 -79.39%~-91.05%，F64 为 -72.04%~-82.21%。F16 的 8/16 输出为
  -68%/-22%~-25%，64 输出只剩 -4.01%/-8.03%，因此生产阈值设为 F16<=16、BF16/
  F32/F64<=64。BF16 >=128 的 control 中有明显运行顺序双峰，候选本来就不命中，继续
  保留 SME/MOPA。
- candidate 与 disable baseline 的扩展 catalog 都是 221/221、0 error/0 skip。
  candidate SHA256 `d39eded94476436accb8cbc22318b862c0ccff1c673a98f4d3fb4ccd15dd6ed8`，
  Berkeley text 1,015,097；disable baseline SHA256
  `0ffd49e58fe6cf55ad21182314236c7aa68a0acca883bccf34db72175f1bfbbe`，text
  1,002,889，增量 12,208B（含 8 个方向/dtype leaf 与 dispatch）。生产默认启用，并保留
  `VECOPS_DISABLE_SME_SVE_SKINNY` 严格 A/B 门禁。
- BF16 row leaf 反汇编大小 0x654：`smstart/smstop/mopa/zero za/call` 均为 0；只含
  predicated `ld1h`、BF16->FP32 widening、FP32 FMA、末端 `faddv` 与 store。该阶段
  尚未使用 BFDOT；后续实验按当前架构约定先放在 SME backend 局部 helper并标 TODO，
  不提前扩大公共 vec API。
- 最终生产 MatmulSMEBench SHA256
  `c7875548731dd490367ac3006897b4d0ed5d61bd26c28d613f3b460485cbf21f`，Berkeley text
  1,015,097；221/221、0 error/0 skip。源码审计在 include/benchmarks/tests 中没有
  `<arm_sme.h>`、ACLE streaming/ZA attribute、原生 MOPA/SMSTART，也没有遗留实验宏。
  本地 x86 最终回归同时通过：BatchAMX 4/4、dispatch 71/71、Fusion 108/108，均 0
  error；MatmulAMX SHA256 `033036fbe45bd51ffb2f02932ce93d8296f5307c59c8394724f699c10e2375d1`。

### MATMUL-COV-021：Scenario 类型/流水线覆盖审计与公共层拆分

- 日期：2026-08-30。状态：实现与双架构门禁进行中。精确审计得到主 Scenario 当前
  SME 23 个 tuple、391 case，AMX 21 个 tuple、357 case（无 AMX-FP16 时 323）；
  Fusion 为 SME 234、AMX 108（有 FP16 时 162），现有 16 种 output pipeline 虽都
  至少出现一次，但 Atom×输入类型×输出类型的交叉仍不完整。
- 发现两个 P0 “伪覆盖”：公共 runner 原先用一个 `InputPipeline` 同时处理 A/B，无法
  表示最常见的在线量化 activation A + 预量化/离线 packed weight B；`c_input` 又与
  `c_output` 共用 `MemoryC`，导致 BF16/F16 输出时 bias/residual 也被迫变窄。公共模板
  已扩展为独立 `InPipelineB` 与 `MemoryCInput`，旧调用保留默认值和原 benchmark 名；
  不同配置的名字明确写出 A/B pipeline 与 C-input dtype。非对称双 zero-point 的独立
  A/B pipeline 尚未实现，暂用 compile-time assertion 禁止误测。
- Scenario 从 12 shard 扩到 16 shard。SME 新增 FP32→F16 compute→FP32、BF16/FP16
  memory→F32 compute→FP32、在线 Quantize4 FP32 A + direct INT8 B，以及 FP32 bias→
  BF16/F16 BiasRelu output；AMX 同步增加现实量化、FP32 C-input narrow output和两侧
  mixed-memory BF16。rank-2 packed bytes counter 同时修正：offline packed 现在只计
  实际 packed operand，online lifecycle 再计 raw read + packed write，不再重复计 raw。
- 按每 tuple 4个 rank2+13个 batch case计算，SME 主 Scenario将从391增至493：三种
  conversion补51、独立 A/B量化流水线17、两种 FP32 C-input narrow output补34；
  最终 ARM list-tests必须与493精确一致。
- 920f-4 最终 list-tests确认为493，独立 A/B pipeline 17、FP32 C-input 34；新增
  conversion/pipeline/C-input 的102 case（408 aggregate records）运行均为0 error/skip。
  Scenario SHA256 `9f3a58344a66138d8929de82010301bd14ccf7bbcb1e7ee077cc5c72cad9b97c`。
- MixedPacking correctness 原先 SME 八个原生 Atom 都缺 PackedAB，AMX BF16/F16也缺
  明确 PackedAB。现已让双 packed helper 跨架构复用，并在原 8 个 shard 内给所有原生
  Atom 加门禁，不额外扩大单 TU。
- 双 packed helper原先假定总能构造“不对齐 KR”的 bounded K；F32/F64 的 KR=1使该
  static assertion无解。现仅 KR>1保留 non-aligned bounded extent，KR=1改用 runtime
  `Any`。最终 920f-4 的 EverySMEAtomSupportsEither/BothPackedOperands 两项均通过；
  test SHA256 `377c25e82b98280359a622856095e3eb535feb0c33fc4dc5e8224568257fc612`。
- 16 shard 的 main registry最初仍只调用0--11；本地 list-tests因此暴露新 object虽已
  编译却未注册。现已改为从统一 `MatmulScenarioShardCount` 生成 index-sequence并注册
  全部 shard，避免下次再手工漏项。system GCC/AMX 对新现实量化、FP32 C-input
  narrow-output和 mixed-memory过滤共68 case（272 aggregate records）为0 error/skip；
  无 AMX-FP16 的本地总注册数由323增至391，其中独立 A/B pipeline 17、FP32 C-input
  17、双向 mixed-memory 34，和预期逐项一致；
  MixedPacking 两个 AMX test也通过。该问题说明 shard count、显式 registry调用和
  list-tests计数必须作为三重门禁，不能只以链接成功判断覆盖已生效。
- 未完成缺口仍包括：SME asymmetric compensation lifecycle、shared-A/independent-batch
  packing benchmark、native F16/F32/F64/INT8 的完整 7-mode packed 性能矩阵、Fusion
  每个浮点 Atom 的统一七类 pipeline，以及 batch conversion/pipeline 独立单测。后续
  必须继续分 shard，不能把这些实例塞回已有重 TU。
- Fusion和WeightReuse的注册 wrapper现也已透传 `InPipelineB/MemoryCInput`，并把
  registry改为由统一 shard count生成；旧配置默认值和名字不变。本地 GCC/AMX完整编译
  通过，list-tests仍精确为Fusion 108、WeightReuse 282。Fusion暂保持
  `EnableBroadcastA=false`，因为现有4个 batch shape没有shared-A；提前设true会为每个
  tuple实例化未使用的shared-A模板并显著增加编译成本，待专项shape加入时再按shard开启。

### MATMUL-SME-X18：公开实现复核与 FP64 area-8 基线

- 日期：2026-08-30。状态：候选识别完成。AtomGit Tensor Contraction PR 4/9/11/14/
  16/18 与 oneDNN、ACL、KleidiAI 的共同模式不是重写普通 pack，而是 raw A 用 ZA
  片内 transpose、packed K-loop 做 load-next/MOPA-current、full/tail owner 分离，以及
  GEMV 使用普通 SVE dot。PR11 的 16×32 FP64 kernel按 K=2 软件流水，PR18 使用八个
  ZA64 tile、单侧预扩宽和大输出 non-temporal store；这些提交缺统一公开基线，因此
  只作为指令序列 oracle，不能直接把作者注释里的收益当作本项目结论。
- 当前 vecops FP64 packed fast group只覆盖 Outputs<=4，area-8 的 `64×256×256` packed
  A/B 会退回 generic `load_operand`。920f-4/CPU100/BiSheng Clang19、固定 2000 次的
  生产基线两轮为 116.587/118.865 us；perf 为 965,488,966 cycles、1,385,154,693
  instructions，IPC 1.43。下一步候选是隔离的 area-8 owner和 K2/K4 pipeline，必须同时
  测 square、wide、tail、Fusion及 `.text`，不能再次把实验 loop复制进 Tile2D catalog。

### MATMUL-SME-X19：SME-local BF16 widening-dot 与 skinny leaf

- 日期：2026-08-30。状态：ARM A/B 通过，保留。X17 的 BF16 skinny leaf仍把每个
  BF16 vector扩成两组 FP32 word后 FMA，每个输出占两份 accumulator。现在仅在 SME
  backend 内实现局部 `skinny_widening_dot_add`：语义为相邻两对 BF16 product累加到
  一个 FP32 lane；SVE-BF16 路径发 BFDOT，fallback 使用两相 widening conversion +
  FMADD。按架构约束不把尚未稳定的 widening-dot 操作加入公共 vec API；实现处带 TODO，
  待其他后端需求与跨架构语义稳定后再考虑泛化。该普通 SVE 数据指令
  不带 ACLE SME streaming/ZA 状态属性，也没有原生 SME state intrinsic。
- skinny BF16 accumulator因此从两个 Z word降为一个。920f-4/CPU100/BiSheng
  Clang19、固定 10,000 iterations 的 baseline-candidate-candidate-baseline 八点结果：
  row/col 的 8 输出 -51.21%/-52.85%，16 输出 -63.16%/-63.08%，32 输出
  -72.49%/-71.72%，64 输出 -70.76%/-70.28%，八点几何平均 -65.27%。baseline 是
  X17 生产 binary `/home/ryz/tmp/x17-sve-candidate`，candidate 为
  `/home/ryz/tmp/x19-bfdot-candidate`，原始 JSON 为 `x19-bfdot-{b1,c1,c2,b2}.json`。
- MatmulSME 扩展 catalog全量 221 case完成，0 error/skip。row/col BF16 leaf大小分别
  1116/1020B，各含 11 条 `bfdot`，`call/smstart/smstop/mopa/fma` 均为 0；反汇编保存
  为 `/home/ryz/tmp/x19-bfdot-{row,col}.dis`。candidate SHA256
  `e192b8d8ae8924005746907295e978583528c52f13b9d560394ab4e89886a8a2`；Berkeley text
  1,013,897，相对 baseline 1,015,097 还减少 1,200B。该方案同时通过性能、正确性、
  状态管理和代码体积门禁，保留为生产实现。
- BFDOT 让每个 output只占一个 accumulator后，又把 BF16 leaf的输出 block从4扩到8；
  仍以 block4+1--3 tail处理非8倍数。相对 block4 的同机 10k B-C-C-B，8/16/32/64
  双向八点分别再下降 5.62%--13.43%，几何平均 -9.61%，所有点均改善；221 case再次
  全量通过。block8 binary `/home/ryz/tmp/x19-bfdot-block8`，SHA256
  `350fe90cac7ab3e22ad8cb9a8bc791af9313f5d98491208640ef0b6872e3ff93`；Berkeley text
  1,015,233，相对 block4 +1,336B、相对 X17 原生产仅 +136B。该 block size保留。
- 920f-4 另有 `svei8mm/i8mm`，但没有 `f32mm/f64mm`。native CMake feature拼接已扩展
  到 `i8mm/f32mm/f64mm`：INT8 可继续测 SDOT/UDOT 与 SMMLA/UMMLA；FP32/FP64
  FMMLA 必须由各自 feature gate决定，不能从 `SME_FA64` 推断。`SME_FA64`只关系到
  streaming mode内普通 A64/SVE 指令可用性，不提供 F32MM/F64MM 计算单元。

### MATMUL-SME-X20：FP16 skinny 的 widening FMLAL

- 日期：2026-08-30。状态：保留。反汇编审计发现 X17 FP16 leaf为扩宽生成大量
  zip/splice/fcvt，再用两组 FP32 FMA。SME backend 新增局部 `skinny_widening_fmadd`
  helper，用 SVE `FMLALB/FMLALT`把 FP16→FP32 widening 与乘加合成；与 BFDOT helper
  一样只在本 backend 内实现并标 TODO，不进入公共 vec API，也不涉及 SME状态 intrinsic。
- 920f-4/CPU100/10k B-C-C-B，相对 X19 block8：实际命中阈值的 row/col 8 输出
  -53.50%/-54.35%，16 输出 -66.09%/-64.01%。64 输出超过生产阈值16，仍走 SME，
  为 +0.80%/-0.73% 的噪声。四个 active点均明显改善；candidate SHA256
  `5b4300f490c2bef7cbbb90e9f19a634e9d0c3aa7768c75b4b8b695bfed7740e7`，Berkeley
  text 1,014,177，相对 X19 block8再减少 1,056B。
- 扩展 MatmulSME 221 case全量仍为 0 error/skip。row/col F16 leaf大小 1172/1268B，
  各含 10 条 `fmlalb`+10条 `fmlalt`，`fcvt/fma/call/smstart/smstop/mopa`均为0；
  反汇编保存为 `/home/ryz/tmp/x20-fmlal-{a,b}.dis`。因此 FMLAL 同时通过性能、数值、
  状态管理、指令序列和代码体积门禁。

### MATMUL-SME-X21：同符号 INT8 skinny 的 SVE DOT

- 日期：2026-08-30。状态：保留。新增独立 `dispatch_integer_skinny` raw catalog，
  S8×S8/U8×U8各覆盖双向8×64、16×65、64×257，共12 case。SME backend局部
  `skinny_integer_dot_add` 让每四个 byte product累加到一个 I32 lane，signed/unsigned
  分别生成 SDOT/UDOT；混合符号 Atom由编译期 TA==TB门禁排除。实现带 TODO，不进入
  公共 vec API；MMLA所需2×8·8×2布局另做 tiny专用 pack实验。
- 920f-4/CPU100/10k B-C-C-B，相对带同样 benchmark注册但仍走 MOPA 的 baseline：
  S8×S8 六点 -66.35%~-81.17%，U8×U8 六点 -66.25%~-81.32%，12点几何平均
  -75.35%，没有回退。baseline `/home/ryz/tmp/x21-int-dot-baseline` SHA256
  `5b50a6e7d55bcd13a21eed080182d51483ad67a98664766d0f6535d76571524c`；candidate
  `/home/ryz/tmp/x21-int-dot-candidate` SHA256
  `b5c3d5b7edbf0d807eb6f2c28e5adea409f4d3c4c58c5d8dc7ce436e96630272`。
- 扩展 MatmulSME 从221增至233 case，全量0 error/skip。四个方向/signedness leaf大小
  1564/1564/1604/1604B，分别含18条 SDOT或18条 UDOT，`call/smstart/smstop/mopa`
  都为0，反汇编保存为 `/home/ryz/tmp/x21-int-{a,b,c,d}.dis`。两个新 Atom使 benchmark
  Berkeley text从1,280,073增至1,287,313（candidate相对同 catalog baseline +7,240B），
  编译峰值约5.9GiB；后续 mixed-sign与packed矩阵应放到独立 shard/target，不能继续
  膨胀这个单 TU。
- 最终 native feature重建已显式加入 `+i8mm`并定义 `__ARM_FEATURE_SVE_MATMUL_INT8`；
  产物 `/home/ryz/tmp/x22-final-i8mm` 与上述 candidate SHA256逐字节相同，text/data/bss
  也完全一致。因此 feature补全没有通过代码放置或编译器重选路径引入性能回退。

### MATMUL-SME-X22：conversion-pack 门禁扩展

- 日期：2026-08-30。状态：保留。`SMEPostprocess` 的 ZA transpose、向量 conversion和
  按 KPack store原本已能表达 BF16→FP32 与 FP32→FP16，但 eligibility和 run内
  static assertion只允许 FP32→BF16/I8，导致这两种现实转换错误退回普通 SVE pack。
  现只扩展 SME pack backend 的类型门禁，没有新增公共 vec API或新的状态管理路径。
- 920f-4、CPU100、BiSheng Clang19、每项3次中位数：BF16→FP32 的 A Const为
  388→142 us（-63.4%）、B Const为696→198 us（-71.6%）；FP32→FP16 的 A Const为
  702→254 us（-63.8%）、B Const为788→311 us（-60.5%），Dyn对应为760→254 us
  （-66.6%）和899→310 us（-65.5%）。原始 JSON：
  `/home/ryz/tmp/x23-pack-conversions.json`。
- `MatmulPackSMETest-Native` 新增两侧 BF16→FP32、FP32→FP16 tail conversion校验后
  6/6 test通过；AMX侧同时补 FP32→FP16两侧检查，8/8通过。这个结果说明回退原因是
  后端选择门禁漏项，而不是 ZA conversion pack 算法不适用。

### MATMUL-SME-X23：MMLA/tiny-pack 与 packed GeMV 布局探针

- 日期：2026-08-30。状态：布局验证完成，第一版 packed GeMV拒绝。920f-4确认每个128-bit
  segment中 BFMMLA/SMMLA/UMMLA 输出 lane均为 `[C00,C01,C10,C11]`。现有 MOPA
  pack相邻两个 K-group只需 U32 ZIP1/ZIP2即可组成 BF16 K4或INT8 K8；行对广播用
  DUPQ时 Clang生成单条 `mov z?.q,z?.q[index]`，TBL替代约慢1%，不采用。
- 动态16 K-block内层探针中，专用 tiny pack相对“现 pack + inline ZIP/DUPQ”分别使
  BF16 2x8 53.30→41.98 ns（-21.2%）、BF16 8x8 128.19→105.70 ns（-17.5%）、
  INT8 2x8 48.63→35.64 ns（-26.7%）。但 raw uint64 gather相对专用 pack慢55.4%/
  80.9%，所以不应把 gather当 tiny首选。第一阶段应先用受限 PackedAB MMLA fast path
  做端到端 A/B，不能仅凭 inner-loop数字替换公共 Pack ABI。
- 当前 packed GeMV仍浪费 ZA tile：BF16 `1x64x257` raw/PackedB/PackedAB为
  1.028/2.893/0.679 us；`1x1024x1024` 的 F32为624.0/218.7/214.1 us，S8U8为
  146.2/68.0/65.7 us，F64为1242.6/434.5/429.5 us。最小候选是保持当前 pack格式，
  在既有 StreamingZARegion内用普通 streaming-SVE BFDOT/FMA/DOT沿 packed变化维计算；
  widening/dot helper必须只放 SME backend并标 TODO，待跨后端语义稳定后才考虑泛化。
- 第一版按“一个输出向量完整遍历K，再处理下一个输出向量”实现并完成 B-C-C-B。它在
  `1x1024x1024` 上使 PackedB/PackedAB 的 F32回退69.9%/66.1%、F64回退69.7%/
  66.0%、S8U8回退56.0%/52.4%；BF16 `1x4096x4096`回退63.6%/61.8%。短K
  `131x1x33 PackedA`虽改善3.8%，不足以抵消主工作负载回退。根因是 loop order让每个
  输出块重复扫描 shared K并放弃现有 MOPA跨输出复用，不是状态切换或指令可用性问题。
  候选只保留在 `VECOPS_EXPERIMENTAL_SME_PACKED_GEMV` 默认关闭宏下，生产路径恢复；
  下一版必须至少4/8个输出向量并行、K外层广播一次再更新多 accumulator。原始 JSON：
  `/home/ryz/tmp/x24-packed-gemv-{b1,c1,c2,b2}.json`，候选 SHA256
  `7301c17f44883ae8dd866af878154a12d47937f040e27a5287a1a845f06ccfd2`。
- 默认关闭候选后的最终 `MatmulSMEBench-Native` 与 X22基线逐字节一致：SHA256均为
  `b5c3d5b7edbf0d807eb6f2c28e5adea409f4d3c4c58c5d8dc7ce436e96630272`，
  Berkeley text/data/bss均为1,287,313/7,480/1,024；因此生产性能没有代码放置回退。

### MATMUL-SME-X24：现有 PackedAB ABI 的 SVE MMLA 端到端实验

- 日期：2026-08-30。状态：隔离 benchmark完成，严格门禁的backend-local dispatch已
  通过生产门禁并默认启用。新增默认关闭的
  `VECOPS_ENABLE_SME_MMLA_EXPERIMENT_BENCHMARK`，只在打开时构建
  `MatmulSMEMMLAExperimentBench-Native`。实验继续调用公开 MatmulPack生成现有 MOPA
  PackedAB，基线调用公开 Matmul；候选只在 benchmark内部以普通 SVE
  ZIP/DUPQ/BFMMLA/SMMLA/UMMLA消费相同 packed ABI，没有修改 Backend、公共 Pack ABI
  或 SME状态管理。
- 覆盖 BF16、S8×S8、U8×U8 的 `2x8/8x2/4x4`，K为
  `64/65/256/257/512/513/1024/1025`，共72个候选 case。所有 case均在计时前对 raw
  reference校验，包含 BF16 KPack2和INT8 KPack4的奇数组尾部；72/72无 error/skip。
  仅支持普通 VL等于 SVL时运行，不满足时明确 skip，不能默默按错误宽度解释 packed
  panel。
- 920f-4、CPU100、BiSheng Clang19、固定10,000次、MMLA-MOPA按 B-C-C-B 四进程
  几何合并。短中K的36点整体几何平均 -52.61%；扩到长K后72点几何平均仍为
  -32.57%，但长K揭示必须做 shape/type门禁：
  - BF16 2x8/8x2在 K=512/513仍为 -22.65%~-28.80%，K=1024/1025仍为
    -14.92%~-19.79%；4x4在 K=257仅 -3.45%，K=512/513开始回退
    +5.14%/+6.58%，K=1024/1025回退 +13.39%~+17.68%。
  - S8×S8/U8×U8 2x8/8x2在 K=512/513仍改善 -15.18%~-17.74%，到
    K=1024/1025已在 -0.49%~+1.63%内打平；4x4在 K=257仍改善
    -9.83%/-11.12%，K=512/513回退 +9.75%~+12.11%，K=1024/1025回退
    +31.90%~+35.02%。
- 由实测得到的零回退初始门禁是：BF16细长2x8/8x2允许到 K<=1025，整数细长允许到
  K<=513，三种类型的4x4只允许到 K<=257。该门禁内52点几何平均 -44.24%，各类型
  分别为 BF16 -38.34%、S8×S8 -47.39%、U8×U8 -47.87%；最弱点是 BF16
  4x4x257的 -3.45%，没有运行时回退。进入生产前仍应补2x2/2x4/4x2及真实 epilogue，
  并优先把阈值写成可单独关闭的 backend-local dispatch。
- 三个 noinline leaf大小均为0x24c；各自有4个静态路径 MMLA、6个 ZIP、2个
  UZP1/UZP2，`mopa/smstart/smstop/call`均为0。最终实验二进制
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release/benchmarks/`
  `MatmulSMEMMLAExperimentBench-Native` SHA256
  `50555e4d67af8bb7ad233859c84f181675365015fd10e68da8af97fa0c9bcaf3`；反汇编保存在
  `/home/ryz/tmp/mmla-{bf16,s8s8,u8u8}-leaf.dis`，原始 JSON 为
  `/home/ryz/tmp/mmla-{e2e,mid,long}-{b1,c1,c2,b2}.json`。
- 已把同一算法收敛为 SME backend-local dispatch。编译期严格要求双方都是现有
  PackedAB、无 input transform、Zero C、native direct row-major output，且调用 scope
  尚未持有 StreamingZA；运行时再检查普通 VL==SVL、输出无 row padding、shape精确为
  2x8/8x2/4x4，并套用上述 dtype/shape K阈值。helper继续带泛化 TODO，只使用普通 SVE
  MMLA intrinsic，不使用 ACLE SME状态函数或新增 vec API。
- 生产模板在新增36个PackedAB probe上完成B-C-C-B：26个实际命中点几何平均
  -41.45%，范围 -72.70%~-7.17%，0回退；10个超过阈值而回落MOPA的control几何平均
  -0.09%，范围 -0.86%~+0.49%。36/36均通过benchmark内部raw reference，0
  error/skip。开启dispatch的二进制SHA256为
  `847cbd0adb32a42213f1ff3a4bc143a6165e5fa6d88083f8eee9f10c7bd738f8`，
  text 1,294,257，相对同probe但dispatch关闭版增加4,844B。
- 三个生产leaf各自反汇编均有4条对应的`BFMMLA/SMMLA/UMMLA`，且leaf内
  `SMSTART/SMSTOP/MOPA/FMOPA/BL`均为0；状态区间仍完全交给现有vec SME wrapper。
  原始数据为 `/home/ryz/tmp/x26-mmla-{b1,c1,c2,b2}.json`，leaf反汇编为
  `/home/ryz/tmp/x26-mmla-{bf16,s8,u8}.asm`。基于严格命中点全胜和fallback稳定，已去掉
  实验宏、默认启用该dispatch；无宏的最终CMake生产二进制与候选逐字节相同，SHA256
  仍为上述值。未覆盖的shape/epilogue仍明确回退原SME路径。

### MATMUL-SME-X25：K-major 多 accumulator packed GeMV

- 日期：2026-08-30。状态：全类型实验完成；保守BF16候选因未命中路径无法满足零回退
  门禁，保持默认关闭。X23第一版按输出块
  逐个扫描K，长K回退52%--70%；现改为每个K-group只加载/广播一次shared operand，
  同时更新最多8个输出向量accumulator，随后用Block4和Block1--4处理尾部。实现仍只在
  SME backend内并由默认关闭的 `VECOPS_EXPERIMENTAL_SME_PACKED_GEMV`控制，不进入
  公共vec API。
- BF16反汇编确认核心loop为一次 `LD1RW` 后连续8条BFDOT，手工Streaming+ZA区间内
  call为0。全类型候选二进制SHA256
  `517b3826a6bf99e108d9ed98d965d376c849490f7ec97938360f8d8562167821`，
  Berkeley text 1,390,545，相对生产基线增加103,232B，因此最终门禁还必须同时收敛
  模板实例和代码体积。
- 920f-4/CPU100/B-C-C-B的34点网格包含BF16双向16--1024输出、K65--4097，以及
  F32/F64/S8U8 `1x1024x1024` PackedB/PackedAB。全网格几何平均 -12.19%；BF16
  28点 -13.12%，S8U8两点 -19.95%，F32/F64仅 -1.21%/-0.46%。相对X23已彻底修正
  loop-order灾难，但仍不是可全开的零回退路径。
- 明确回退集中在BF16大输出长K和若干PackedAB：512/1024输出、K4096/4097回退
  2%--8%；PackedAB的32列向、64双向也回退1%--11%。另一方面，仅变化侧packed的
  BF16 16/32/64输出在两个方向全部改善8.7%--65.8%。因此下一候选收窄为BF16、恰好
  一侧packed且该侧为变化维、outputs<=64；PackedAB及F32/F64/INT8保留在额外的
  `VECOPS_EXPERIMENTAL_SME_PACKED_GEMV_ALL_TYPES`下。原始数据：
  `/home/ryz/tmp/x25-packed-gemv-{b1,c1,c2,b2}.json`。
- 保守门禁的命中6点（BF16、恰好一侧packed、outputs<=64）均改善，首版几何平均
  -43.89%，改为64B对齐的noinline leaf并在进入StreamingZA前调用后为 -51.08%，
  单点范围 -34.45%--67.60%。但同一二进制中未命中的28个control几何平均仍为
  +1.30%，其中128/256输出control反复出现明显双峰；三轮隔离B-C-C-B的同一
  n256 control分别为 +0.47%、+26.41%、-3.31%。这说明命中路径本身的收益真实，
  但当前巨大模板catalog的代码放置对短case仍不稳定，不能据此承诺生产零回退。
- 尝试把leaf放入 `.text.zz_vecops_sme_packed_gemv` 输入section也没有改变最终布局：
  默认链接脚本将其重新合并到 `.text`，section版与普通noinline版逐字节相同，SHA256
  都是 `00745bcce5d33991e68d1adc55142f63e71f12cb031a7cc4badc3c0b4077c2d3`，
  Berkeley text 1,297,073（比生产基线增加9,760B）。该无效属性已移除；候选代码只由
  默认关闭宏保留，待catalog/冷块真正拆TU或具备链接脚本级函数放置能力后再启用。

### MATMUL-COV-022：Fusion/Packed/WeightReuse 第二批类型与融合覆盖

- 日期：2026-08-30。状态：本地 AMX及ARM Fusion/Packed/WeightReuse/SME test门禁
  全部通过。Fusion从10扩到17
  shard，新增 FP32 bias→BF16/FP16、BF16 Relu/Sigmoid、在线量化A+原生B，以及
  S8S8/S8U8/U8S8/U8U8原生输入→INT32 accumulate。每个新增shard最多两个tuple，
  没有把所有类型重新塞入已有重 TU。
- 本地无AMX-FP16环境的 Fusion注册数由108增至189：FP32 C-input 18、Relu/Sigmoid
  18、独立A/B input pipeline 9、四种原生INT accumulate 36；新增81 case对应324条
  aggregate record，0 error/skip。带AMX-FP16时连同原有FP16配置预期总数261；SME含
  F16/F32 narrow组合时在920f-4精确注册351。ARM新增过滤集命中117 case、产生468条
  aggregate record，全部通过内部reference，0 error/skip。
- PackedScenario从9扩到19 shard；新类型只用 tail和GeMV-tail两个probe，但仍覆盖
  Raw/PackedA/PackedB/PackedAB及三种online pack，共7种mode。新增BF16窄输出+FP32
  C-input、F16/F32/F64、四种原生INT→I32及量化A+原生B。本地无AMX-FP16精确361；
  920f-4上的SME+F64精确445（预期值吻合），SME无F64预期431，AMX+FP16预期417。
  ARM选取14个raw/PackedAB tail case覆盖BF16/F16窄输出、F32、混合符号INT、独立量化
  pipeline等组合，全部通过benchmark内部reference，0 error/skip。
- WeightReuse从3扩到12 shard，新增8点轻量lifecycle网格；统一target本地精确378，
  920f-4 SME+F64精确426，SME无F64预期410，AMX+FP16预期394。新增tuple由
  `!VECOPS_WEIGHT_REUSE_PIPELINE`门禁隔离，三个专用BF16/Convert/Quant target仍各94，
  没有重复注册。ARM选取16个覆盖六类pipeline的raw/pack-once小batch case，全部
  0 error/skip。
- Fusion correctness公共helper新增 mixed-memory bias+ReLU、独立量化A+direct B、四种
  原生INT accumulate、Relu/Sigmoid。AMX Fusion test由2增至4项并4/4通过；benchmark
  本地最终重编译后注册数仍为Fusion 189、Packed 361、WeightReuse 378，AMX test仍
  4/4通过。920f-4上的SME Fusion test由3增至5项并5/5通过。
- 构建成本方面发现新的模板TU问题：BiSheng Clang 19编译未分片的
  `MatmulFusionSMETest.cpp`最终约7分钟、峰值观测约13.56GiB，远高于benchmark分片。
  本轮机器有足够内存可继续，但后续新增cross-product前应把Fusion correctness按
  float/mixed-int拆TU，避免测试覆盖扩展反过来拖慢日常门禁。

### MATMUL-COV-023：SME MatmulPack FP32→INT8 融合覆盖

- 日期：2026-08-30。状态：完成。`MatmulPackSMEBench-Native`新增 FP32→S8 与
  FP32→U8 的A/B两侧、Const/Dyn shape，并同时注册Vector基线和SMEPostprocess，
  合计16个新case；量化transform为乘4，保持与correctness test相同语义。总注册数
  现为252，新过滤集16/16运行成功，0 error/skip。
- correctness不依赖benchmark：`MatmulPackSMETest`已有
  `check_fp32_to_int8_postprocess`，用S8×U8 atom分别从A/B侧覆盖有符号和无符号目标，
  且逐个检查packed panel、KPack与空间/K尾部补零；因此本轮只补齐缺失的性能覆盖，
  没有为benchmark另造一套打包实现。

### MATMUL-SME-X26：PackedAB MMLA 扩展到2x2/2x4/4x2

- 日期：2026-08-30。状态：生产门禁通过，保留。隔离实验把现有MOPA PackedAB上的
  ordinary-SVE MMLA扩到`2x2/2x4/4x2`，三种类型×8个K共72点全部正确。隔离门禁
  BF16 K<=1025、INT8 K<=513的60点几何平均 -44.67%，最弱仍 -14.06%；三个leaf
  仍各0x24c，对应4条BFMMLA/SMMLA/UMMLA，`MOPA/SMSTART/SMSTOP/call`均为0。
  原始隔离数据为`/home/ryz/tmp/mmla-tiny-{b1,c1,c2,b2}.json`。
- 真实Matmul dispatch揭示orientation差异：`4x2`的BF16 K513/1025回退
  +7.63%/+14.68%，INT8 K513约+14.2%；故最终把`4x2`收紧到所有类型K<=257，
  `2x2/2x4`仍保留BF16<=1025、INT8<=513。不得把isolated leaf阈值直接当作生产阈值。
- 为避免扩展predicate扰动已验证的旧shape，原`try_packed_mmla`保持原代码，tiny只在
  其返回false后进入独立dispatch。最终72点B-C-C-B：新增26个命中点几何平均
  -40.40%，范围 -69.98%~-4.70%；旧26个命中点 -0.68%，最差+0.44%；20个fallback
  -0.10%，范围 -0.81%~+0.33%，无>1%回退。候选SHA256
  `2512a67465d90c606711ca964a075748c73d38bc6e775a9851b2b1cbb36c8856`，
  text 1,298,201；相对同72注册的旧门禁增加2,104B。
- F32/F64负面门禁：`svmmla_f32/f64`要求普通SVE `f32mm/f64mm`，920f-4没有；
  `SME_FA64/SME_F64F64`不能替代，FP16也没有widening MMLA，继续使用FMLAL。

### MATMUL-SME-X27：ACL/KleidiAI/oneDNN/OpenBLAS/AtomGit 再审计

- 日期：2026-08-30。状态：完成源码审计，两个P0候选进入实验。ACL SME1把
  M=1、batch=1、无indirect/accumulate单独分流到8VL GEMV；BF16版本一次
  `LD1RQW`取8个连续shared值并用4个lane-indexed BFDOT消费4个K-group，F32版本
  类似地一次取4个shared值。这比当前实验packed GEMV逐K-group构造broadcast更省
  load/loop开销，但只适用于shared侧raw contiguous、变化侧packed：
  [BF16 GEMV](https://github.com/ARM-software/ComputeLibrary/blob/2a8ca9840a3112d12d6cc68e49e6c833f9759fd6/src/core/NEON/kernels/arm_gemm/kernels/sme_gemv_fp32bf16fp32_dot_8VL/generic.cpp#L94-L149)、
  [F32 GEMV](https://github.com/ARM-software/ComputeLibrary/blob/2a8ca9840a3112d12d6cc68e49e6c833f9759fd6/src/core/NEON/kernels/arm_gemm/kernels/sme_gemv_fp32_mla_8VL/generic.cpp#L95-L133)。
  第一版已按相同方法在SME Backend局部加入BF16 K×4实验，仍由默认关闭宏控制。
- ACL同时保留1VL×4VL、4VL×1VL、2VL×2VL三个MOPA family，并按N>=8VL或M<=VL、
  N<=VL、其余方形选择；这证明aspect ratio本身应进入shape benchmark/dispatch，而不是
  只依赖面积：[SME注册与规则](https://github.com/ARM-software/ComputeLibrary/blob/2a8ca9840a3112d12d6cc68e49e6c833f9759fd6/src/core/NEON/kernels/arm_gemm/gemm_fp32.cpp#L192-L224)。
- AtomGit MR11的F64 contraction把K展开2，并交错FP32→FP64转换、B load和8个FMOPA；
  `kernel_edge<N_RB,N_CG>`在编译期删除无效B load/FMOPA：
  [kernel](https://atomgit.com/kunpengcompute/HPC-competitions/blob/5701844773d447745547c5b7a36913857c773bde/01_official/Kunpeng_HPC_Global_Challenge_S1/submissions/Overclock_Overclock_tensor-contract/code/tc_kernel.h#L17-L223)。
  MR11选择B预widen，MR18选择A预widen，说明conversion pack应根据复用侧和footprint选择，
  不能固定两侧都扩宽。MR18对256MiB write-only C报告STNT1收益，但只可作为巨大、无后续
  读取输出的门禁实验，不能默认启用。
- KleidiAI BF16 MOPA一次预载8对A/B vector，并把下一组load穿插在16个BFMOPA之间：
  [汇编](https://github.com/ARM-software/kleidiai/blob/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa_asm.S#L84-L157)。
  它把bias放在packed RHS前并用ones+FMOPA初始化ZA；vecops的CInput语义更一般，因此只能
  考虑“离线固定权重+固定bias”的专用wrapper，不能污染公共packing ABI。
- oneDNN SME JIT选择raw-B不pack，并用ZA0横载/纵读转置A、ZA1..3并行三个N tile：
  [JIT kernel](https://github.com/uxlfoundation/oneDNN/blob/a244fe249f408a8f6ae61ad800e28fa64a8fb074/src/cpu/aarch64/brgemm/jit_brgemm_sme_kernel.cpp#L47-L144)。
  更直接可复用的是batch规则：shared B且A/C plain时合并batch→M，M=1 transpose-A视作
  plain避免copy；其“只copy K-tail”实验反而比copy完整A差，尤其大K，故不得照搬。
- OpenBLAS SVE transposed GEMV每次做3个输出、每个输出4条独立K累加链，适合作为
  outputs<=3且长K的raw skinny候选；BF16版本也用`LD1RQ`+lane widening FMLAL。
  OpenBLAS SME direct SGEMM只预处理A、B保持raw并使用双bank K流水，但封装每次
  malloc/free，vecops若实验必须使用已有workspace。vLLM当前没有SVE/SME CPU kernel；
  其NEON BFMMLA使用专用K/4×row-pair×4布局，再次说明MMLA的理想pack ABI与MOPA不同。
- ISA门禁复核：`SME_FA64/SME_F64F64`不能替代普通SVE `F32MM/F64MM`。920f-4没有
  `svef32mm/svef64mm`，Clang对`svmmla_f32/f64`分别报缺`f32mm/f64mm`；因此不注册
  不可执行的F32/F64 MMLA，FP16继续使用widening FMLAL。

### MATMUL-SME-X28：raw-shared `LD1RQ`+lane-BFDOT K×4

- 日期：2026-08-30。状态：算法保留在默认关闭的packed GEMV实验宏，未进入生产。
  针对恰好一侧packed、另一侧raw contiguous的BF16 GEMV，一次`SVLD1RQ_U16`读取8个
  shared BF16，并以4个`SVBFDOT_LANE_F32`消费4个KPack2 group；尾部仍走原逐group
  路径。实现与泛化TODO只在SME Backend，不新增vec API；额外disable宏仅用于A/B
  二进制隔离。
- 920f-4/CPU100/B-C-C-B中，真正命中的`packed_b` row和`packed_a` col共6点，相对
  原K-major候选几何平均再改善 -61.92%，范围 -68.14%~-54.24%，6/6正确。说明ACL
  的indexed-K展开对当前packing ABI同样有效。
- 但候选text由1,307,961增至1,311,801（+3,840B），原因是Block1/2/4与两个方向
  各自复制展开loop；同一49点二进制中的43个未命中control出现 -42%~+26%的代码布局
  波动，无法证明生产零回退。下一步必须把BF16 leaf提成单份out-of-line实现或独立TU，
  再复测完整suite；当前不能仅因6点大幅获益而默认打开GeMV宏。原始数据为
  `/home/ryz/tmp/x28-gemv-{b1,c1,c2,b2}.json`。

### MATMUL-COV-024：单边非对称量化与tiny/skinny fusion shape

- 日期：2026-08-30。状态：本地AMX与ARM统一门禁通过。新增现实常见的
  “FP32 activation在线量化为U8(zp=3)+原生S8权重”单边非对称组合；公共判定只放开
  这一有reference/correction语义的组合，其他不支持的pipeline pair仍静态拒绝。
  Scenario/Fusion/PackedScenario/WeightReuse各用独立shard注册；Packed覆盖7种packing
  mode，WeightReuse覆盖8个lifecycle点及raw/pack-once两种策略。
- Fusion所有tuple新增`2x4x(2*KStep+1)`、`1x8x4097`和`8x1x4097`三个runtime
  shape，分别覆盖AMX area<=16 fused-small，以及SME/AMX双向长K+tail skinny
  fusion。correctness在AMX/SME各增加`1x8x129`与`8x1x129` Relu/Sigmoid，并让
  非对称helper显式验证direct-S8 B。
- 本地无AMX-FP16实际注册数：Scenario 391→408、Fusion 189→264、Packed 361→375、
  WeightReuse 378→394；四target均重编译通过，AMX Fusion 4/4通过。抽测Scenario单边
  非对称、Fusion tiny/skinny、Packed online-packed-B以及WeightReuse pack-once共6组，
  全部0 error/skip。920f-4 SME+F64最终精确注册Scenario 510、Fusion 480、Packed 459、
  WeightReuse 442；新增tiny/双向skinny共120 case全部通过内部reference、0
  error/skip。单边/双边非对称过滤实际运行的mean case分别为Scenario 34、Fusion 22、
  Packed 42、WeightReuse 16，全部0 error。SME Fusion correctness 5/5通过。
- 下一批尚缺：runtime per-row activation scale×per-column weight scale/zero-point、GELU/
  SiLU/SwiGLU、INT32 bias+per-channel requantize、Fusion shared-A batch、Packed shared-A/
  higher-rank广播、mixed A/B memory type、WeightReuse M>1/K-tail/independent batch，以及
  scenario级strided/transposed性能矩阵。

### MATMUL-SME-X29：零C逐元素epilogue的双向skinny融合

- 日期：2026-08-30。状态：BF16/FP16/FP32生产门禁通过并默认启用；FP64暂缓。旧路径对
  非direct `COutput`直接退回StreamingZA，即使`M=1`或`N=1`也要承担ZA区域和tile
  traversal成本。新路径只接受raw row-major A/B、同类型原生compute、Zero C、rank-2
  且`is_elementwise`的输出transform；bias、累加、输入转换、量化补偿和坐标感知
  `per_column_scale`均保持旧路径。计算先落到最多64个栈上accumulator，再通过原
  `COutput.store`执行clamp/scale/dynamic scale/Relu/Sigmoid/narrow，不复制epilogue
  语义。
- 第一版把`COutput`直接带进每个skinny计算实例，correctness虽5/5通过，但测试binary
  text达1,394,425B且Clang峰值约14.8GiB。最终把计算提成与transform无关的单份
  `noinline`、64B对齐leaf，测试text降到1,374,793B（比第一版少19,632B）；完整480
  case候选相对同注册baseline的text为7,120,283 vs 7,071,639B，增加48,644B
  （0.69%）。baseline/candidate SHA256分别为
  `d4698d1991d24148fe2d398b33e3a22d1581d248bd2e007763bb936d5981bc46`和
  `077ad0228a8e1ae1d897aefa1cb74892b3013abf33ee024b54ce53f3f6dfd2aa`；主项目真实
  源码路径下的最终生产binary text同为7,120,283B，SHA256为
  `de7af57cd4cb8997c122b0feb34fc59e1d4435428d4ddd61375b6f376dec61d2`，480个注册中
  `tiny_area`与双向skinny共120 case再次全部reference通过、0 error。
- 920f-4/CPU100/BiSheng Clang19、`1x8x4097`与`8x1x4097`、baseline-candidate-
  candidate-baseline共80 case：双向各11个生产命中点的几何平均分别为-92.95%与
  -92.97%，范围分别为-93.32%~-92.59%和-93.53%~-92.14%。row 29个control几何
  平均-0.04%、范围-0.46%~+0.57%；col 29个control除一个FP16 per-column点外均
  <=+0.22%，该点首轮+1.68%、反向顺序复测+0.06%、合并为+0.87%，通过1%门禁。
  原始JSON为`/home/ryz/tmp/x29-final-{b1,c1,c2,b2}.json`，专项复测为
  `x29-final-{b3,c3,b4,c4}.json`。
- FP64 leaf本身同样快80%~93%，但启用时四个未命中FP64 control首轮出现
  +1.26%~+2.13%，反向复测虽降到-0.10%~+0.77%，合并仍有三个约
  +1.01%~+1.34%。根因是header-only wrapper与FP64旧路径共处重模板分片造成代码
  布局扰动，不是计算算法慢；生产predicate因此先限制accumulator<=32bit，并在实现处
  留TODO，待FP64 wrapper可拆独立冷TU后复测。
- 最终六个计算leaf均无`SMSTART/SMSTOP/MOPA/FMOPA/BL`：BF16双向各0x778、18条
  BFDOT；FP16双向各0x4c0、20条FMLALB/FMLALT；FP32为0x4b0/0x470。它们只使用
  ordinary-SVE数据指令，SME状态仍完全由现有vec SME wrapper管理。widening-dot、
  FMLAL和MMLA helper继续只放SME Backend并标TODO，不进入公共vec API。
- 远端最后一次同步还暴露出增量构建陷阱：`rsync -a`保留的源码mtime可能早于刚在远端
  生成的旧object，即使文件内容已变，Make仍会误判`Nothing to be done`。本轮对
  `MatmulFusionSMETest.cpp`显式刷新mtime后重编；今后同步后若远端构建与本地修改时间
  交错，必须用依赖hash/强制touch或全新build目录确认重模板TU确实重编，不能只看target
  成功消息。强制重编约8分半，最终test SHA256为
  `f2641b304a332aec215a00b73289a6c0b477152a0f2d1b3ce845e5382a1c57c0`，加入双向
  Relu/Sigmoid调用后的套件仍5/5通过。

### MATMUL-SME-X30：coordinate-aware lane-local与per-column skinny

- 日期：2026-08-30。状态：lane-local语义保留；fused lane-local最终528-case门禁失败，
  改由`VECOPS_EXPERIMENTAL_SME_FUSED_LANE_LOCAL`默认关闭。Tensor
  transform新增`is_lane_local`与`make_lane_local_vec_transform`：它承诺输出lane i只
  依赖输入lane i，但仍允许读取逻辑坐标和外部per-lane参数；与`is_elementwise`不同，
  它不承诺coordinate-independent或permutation-equivariant，因此DataAccess不会据此
  做unordered重排。原per-column scale benchmark/correctness改用该factory；本地与ARM
  TensorTransform分别7/7通过，AMX Fusion仍4/4通过。
- 第一轮直接拿旧transform类型binary作baseline时，candidate text增加26,184B并出现
  大量布局波动，数据不可用于生产判断。随后加入只用于A/B的
  `VECOPS_DISABLE_SME_FUSED_LANE_LOCAL`，以完全相同lane-local类型、480注册和源码路径
  重建严格baseline/candidate。inline wrapper的text为7,138,107→7,146,467B（+8,360B）；
  新per-column双向6点约-87.6%，但旧active/control可回退约2倍，否决inline版本。
- 最终仅把非-elementwise的lane-local COutput wrapper变为noinline、64B对齐COMDAT，旧
  elementwise wrapper和六个compute leaf保持原样。candidate text降为7,143,331B，
  相对严格baseline只增加5,224B；可见12个wrapper合计2,640B。CPU100 B-C-C-B中，
  BF16/FP16/FP32 native per-column双向各3点几何平均-93.09%/-93.02%，范围
  -93.38%~-92.32%。row旧11 active几何-0.12%、26 control -0.03%，均无>1%回退。
  col首轮两个FP16旧active和一个量化control约+2.4%，反向轮均在±0.15%，第三轮亦在
  ±0.20%；六次合并分别为+0.90%、+0.75%、+0.82%，通过1%门禁。严格baseline/candidate
  SHA256为`5932518679ae931d05ddf91b434221c78fc76a3a6fe4f95e5d0b80366481735e`和
  `e5b48883ea1af50880ea6f88fcf30d87cd362ab0583431fdee4d89168c1adf43`；原始数据为
  `/home/ryz/tmp/x30-noinline-{b1,c1,c2,b2}.json`及专项`{b3..b6,c3..c6}`。
- 扩展到最终528注册后重新做同源码严格A/B：baseline/candidate SHA256为
  `e25fbf4e4b693c7059e0506ec35c433f17ecdcb098dcbc9e7a8d582534a71af9`和
  `d6dea2736d6c1db659844ded5bb5ffe645ab9ab963ea041ac3396b07dc3b3c7a`，text仍只差
  5,224B。新增per-column双向仍约-93.25%，但row旧active/control最差+4.73%/+3.97%，
  col旧active几何+3.55%、最差+12.09%，col control最差+2.90%。这证明通过480注册并
  不能外推到更完整链接布局；noinline header wrapper仍不足以隔离开放COutput类型。
  生产因此恢复只接受`is_elementwise`，lane-local factory仍用于正确表达语义和coverage；
  后续需结合arch-object、type-erased epilogue或链接末段TU再实验。最终JSON为
  `/home/ryz/tmp/x33-final528-{b1,c1,c2,b2}.json`。
- 默认关闭实验宏后的主项目生产binary与同528注册严格baseline逐字节一致：SHA256均为
  `e25fbf4e4b693c7059e0506ec35c433f17ecdcb098dcbc9e7a8d582534a71af9`，
  text/data/bss均为7,931,139/13,648/1,120B，且不存在任何
  `sve_skinny_fused_lane_local_matmul`符号。因此实验代码和trait保留不会造成生产布局回退。

### MATMUL-SME-X31：FP64 fused compute独立arch leaf

- 日期：2026-08-30。状态：最终542-case门禁通过，已生产默认ON；可用CMake选项
  `VECOPS_ENABLE_SME_FUSED_F64_EXTERNAL_LEAF=OFF`紧急回退。与transform无关的FP64
  row/col compute显式实例化到`src/arch/sme/FusedSkinnyF64.cpp`，开放COutput epilogue
  wrapper留在header但为小型noinline符号。`src/CMakeLists.txt`从通用source GLOB排除
  `src/arch/`；multiarch层按完整`-march`的SHA创建/复用静态归档并只链接到匹配SME
  F64目标，因此公共`libvecops.a`不携带SVE，未使用项目multiarch helper的header用户
  也不会得到宏或undefined，而是安全回退原header/Streaming-ZA路径。
- 920f-4、BiSheng Clang 19.1.7、最终COV026修后源码：ON/OFF均542注册，各2168条
  mean/median/stddev/cv record，reference均0 error/skip；只有FP64所在shard7对象不同。
  ELF `.text` 8,109,848→8,115,632B（+5,784B），`.data/.data.rel.ro/.bss`尺寸完全相同；
  OFF/ON SHA256分别为
  `5cd875237844ce5adea2e538533432a032c10103b53698c5f0d85ccebd25dd21`和
  `ee39593cbf4f37cfcd7c864630c7faa2c49cff463215cb62abb7affebd1b1041`。非Fusion
  `ExecutionSessionTest-Native` ON/OFF逐字节相同且leaf符号0，证明静态归档未被无关目标抽取。
- CPU100 B-C-C-B覆盖`1×8×4097`/`8×1×4097`、native FP64与FP32-memory/F64-compute、
  clamp/dynamic/per-column：4个active几何-81.49%，范围-82.30%~-80.88%；8个control
  几何+0.01%，范围-0.17%~+0.41%。注册自身固定20ms/repeats3覆盖了命令行10000x，故按
  实际MinTime协议报告；原始JSON为`/home/ryz/tmp/f64-production-542-{b1,c1,c2,b2}.json`。
- arch归档10,930B，SHA256
  `a26bdff544d9f6d49808907f0c484be78fc6d1076c20593a33c364c01d229ceb`；Native与
  NativeFixedStreamingSVE复用同一归档，leaf本身不带fixed-SVL flag。两个compute为
  0x4b0/0x470、合计20条普通SVE FP64 FMAD和22条FADDV，两个入口各4B；
  `SMSTART/SMSTOP/MOPA/FMOPA`均为0，状态仍完全由vec SME wrapper管理。SME Fusion
  correctness 5/5，binary SHA256为
  `3f1fc99442ecc79090595557174c316bbee0e324080e6d395c972286b83f200f`；x86显式GCC
  configure/build也确认不创建SME leaf target。

### MATMUL-COV-025：SiLU与INT32 bias+per-column requant

- 日期：2026-08-30。状态：本地AMX与ARM统一门禁通过。Fusion新增
  精确SiLU `x/(1+exp(-x))`与Bias+SiLU，各自独立shard；另新增在线量化FP32 A+原生S8
  B、INT32 per-column bias、runtime per-column FP32 scale、FP32 dequant，以及带runtime
  output zero-point的U8 requant，dequant/requant各自独立shard。流量计数包含scale tensor
  和runtime zero-point。
- 本地GCC Fusion注册264→312，AMX Fusion correctness仍4/4；tiny-area SiLU、
  skinny Bias+SiLU、INT32 bias+per-column dequant、runtime-zp U8 requant四个代表case
  均reference通过。920f-4 SME+F64实际注册480→528，新增48项产生192条aggregate
  record并在实验启用/生产默认关闭两种binary上全部reference通过、0 error；最终生产
  binary text/data/bss为7,931,139/13,648/1,120B，SHA256
  `e25fbf4e4b693c7059e0506ec35c433f17ecdcb098dcbc9e7a8d582534a71af9`。TensorTransform
  7/7、SME Fusion 5/5通过，后者最终SHA256为
  `4472bc7c2cee9f11ac2cbd22feaf637c540f76c28a309d27673f7f53a339f41a`。新增实例使
  该未分片correctness TU的Clang O3单次编译约10分钟；现已用通用target-shard机制把
  SME拆为浮点、整数、F64三个TU，AMX拆为浮点/整数两个TU，原始source TU为空且不重复
  注册。F64 runtime断言和F32→F64转换集中在SME shard2；本地AMX全量wall约1:56，4/4通过。
- GELU未用近似公式硬凑：当前vec API没有erf/tanh，exp拼tanh近似会改变语义和溢出边界。
  仍缺runtime per-row activation scale×per-column weight scale、runtime input zero-point，
  以及它们与batch广播/packing lifecycle的明确计时归属。

### MATMUL-SME-X32：packed GEMV Kx4展开的单份noinline leaf

- 日期：2026-08-30。状态：保留在默认关闭packed-GEMV实验宏；命中与短control通过，
  宽control仍有双峰布局噪声，不能生产启用。X28的BF16 raw-shared
  `LD1RQ`+4个lane-BFDOT在
  Block1/2/4和两个方向中复制，旧两个方向符号各约0x1950并造成control布局失真。最终
  按Block1--4生成四个裸指针noinline leaf，方向由调用点提供连续shared/packed地址；
  使用SME-local `may_alias` BF16视图，不新增公共vec API。
- 四leaf尺寸分别0x154/0x1b4/0x218/0x280，合计0x7a0；实验binary text从
  1,311,505降到1,302,073B，减少9,432B。920f-4/CPU100 B-C-C-B中6个真正命中case
  全部改善，几何平均-17.68%、范围-36.41%~-7.84%；15个短非命中control几何
  -0.04%、范围-1.57%~+0.58%。四轮35-case benchmark均reference通过，最终9-case
  smoke 0 error/skip，MatmulSMETest 17/17通过。may_alias最终candidate SHA256为
  `d49b1a779a73ad1d85c1746a3c0cc1669d7193c928464d9c1fadb57f907ce375`。另14个
  128/256输出宽control仍呈双峰，几何+2.67%、范围-16.59%~+24.21%；因此本轮只解决
  leaf复制和短case回退，未解决整个实验target的链接布局敏感性。原始JSON为
  `/home/ryz/tmp/packed-gemv-v3-{sb1,sc1,sc2,sb2}.json`，反汇编为
  `/home/ryz/tmp/packed-gemv-v3-final-leaf-*.asm`。
- 曾尝试单一runtime-block leaf进一步去重，但Block2回退1%~2%、Block4回退9%~10%；
  分支和动态block索引抵消了代码体积收益，因此否决。编译期Block1--4四实例是本机当前
  性能/体积折中，仍只存在于SME Backend实验区并带泛化TODO。

### MATMUL-SME-X33：Kunpeng公开PR源码复核与PMU可用性

- 日期：2026-08-30。状态：源码审计完成，候选待严格A/B。直接抓取
  `kunpengcompute/HPC-competitions`的AtomGit merge-request refs并逐份阅读，而不是只依据
  README：重点包括MR7/starwing（`6001df0`）、MR10/pny（`ec30f5e`）和
  MR18/MakerFirst（`237a0ae`）。这些实现共同使用连续panel、在外层保持streaming/ZA
  状态、多个ZA tile打散同一K链，以及full tile直接写回/tail单独处理；这些原则当前
  vecops的pack、vec SME状态wrapper、编译期`NM/NN`多tile和full/tail traversal已经覆盖，
  不能作为新的优化重复移植。
- 尚值得独立门禁的差异有两项。第一，MR18的FP32-memory/FP64-compute按复用比只把A预
  扩宽为FP64，B保持FP32 footprint并在kernel中`zip/cvt`；vecops已有FP32→FP64 pack与
  双侧原生ZA64路径，但还应在shared-A/shared-B和batch广播下比较“只扩宽高复用侧”。
  第二，MR18在大工作集K loop使用约48次迭代前瞻并对不会立即读取的巨大C使用
  non-temporal store；vecops的packed MOPA主循环目前没有显式prefetch，输出也默认普通
  store。这两项必须分别测量，不能把比赛中固定shape的参数直接变成通用默认值。
- 同步复核KleidiAI `main@4bc7bd4`的SME1 INT8 1×N DOT实现：其主要优势不是单独
  `SDOT`，而是一次保持四个输出向量、每16B K块用一次`LD1RQB`供16组packed-weight
  load复用，并在同一leaf尾部完成zero-point compensation、整数bias、scale、FP32
  accumulate、clamp和tail store。vecops已有SME-local整数DOT及多accumulator实验，但
  当前默认路径尚未把这些量化epilogue并入同一个leaf；待runtime input zero-point和
  per-row/per-column scale coverage落地后，应以这一完整融合段而非裸DOT作为优化单元。
- 在920f-4尝试用`perf stat -C 100`采集cycles/instructions/cache/branch事件时，内核返回
  `perf_event_paranoid=2`且当前用户没有`CAP_PERFMON`。按共享主机规则没有修改sysctl；
  现阶段改用CPU100冻结binary、多轮B-C-C-B、反汇编和符号尺寸门禁。若后续管理员开放
  PMU，再补L1D/LLC refill、stall和MOPA利用率，尤其用于prefetch与non-temporal store
  的最终判定。

### MATMUL-COV-026：runtime per-row A量化与双向scale

- 日期：2026-08-30。状态：AMX与ARM最终542注册门禁通过。新增现实推理
  pipeline：FP32 A按runtime `(batch,row)` multiplier在线量化为U8，runtime input
  zero-point为7，原生S8 weight配`-zp*sum(B)`补偿；INT32累加再乘runtime per-row
  dequant scale与per-column weight scale输出FP32。输入transform用lane-local逻辑坐标，
  rank-2取`coord[-2]`，rank-3独立A的batch stride为M、共享A为0，避免把物理广播地址
  错当逻辑batch坐标。
- Fusion shard由22→23，新增14注册：8个rank-2 shape、4个常规batch、1个shared-A raw、
  1个shared-B `OnlinePackedB`。AMX总注册312→326，SME预计528→542。benchmark新增
  `input_parameter_bytes`、`epilogue_parameter_bytes`和`runtime_input_zero_point=7`
  counter；online pack生命周期还计入读取zero-point的4B，补偿是预计算还是随pack计时
  继续由架构/模式counter明确表达。
- correctness同时覆盖独立A+共享B、共享A+独立B，并全量核对`batch=3,M=5`的所有输出，
  因此非首batch/row不是只靠抽样通过。本地GCC Debug AMX Fusion 4/4、Release Fusion
  23 shards全量构建，修正后14项各56条aggregate record且0 error。初版只在K轴取
  `lane_coord(0)`并广播scale，AMX没有暴露但SME spatial-axis packing会跨row；最终实现
  在K轴保留fill fast path，其他axis逐lane读取逻辑坐标。ARM ON/OFF 542全量reference
  各2168条record、0 error/skip，SME Fusion 5/5；rank-2、independent batch、shared-A和
  OnlinePackedB均覆盖。此批没有用近似公式伪造GELU。

### MATMUL-SME-X34：packed GEMV独立section与剩余布局回退

- 日期：2026-08-30。状态：拒绝默认启用，保留OOL实验方向。为隔离X32候选，把四个
  K×4 leaf及noinline dispatch放入独立`.vecops_kernel_text`；同源码exact baseline/OOL
  的主`.text`只增加128B，独立section为0xc7c，data/bss地址完全一致。冻结二进制后在
  920f-4 CPU100、ASLR off做4轮B-C-C-B：6个真正命中case几何-84.624%，范围
  -87.127%~-82.694%；15个short control几何+0.182%，范围-0.848%~+1.406%，其中
  `m64 raw`的+1.406%和`n32 raw`的+0.976%跨轮稳定，仍不满足完整控制无回退门禁。
- 128/256 wide组合过滤仍有双峰，同一baseline前后可漂移+103%，其表面几何+3.589%
  不可归因。拆成单case perf后，`n128 packed_b`候选cycles仅+0.0244%，instructions
  稳定+0.0465%（每次恰多12.99条，是OOL runtime dispatch/reject成本）；`n256 raw`
  instructions 0变化，cycles范围-1.154%~+1.874%、几何+0.695%，属于布局/缓存波动。
- 静态比较钉死根因：两个PackedB/PackedA `run_operation`各增长约68B并跨过64B对齐边界，
  后续12个SVE skinny leaf有10个整体平移128B；12/12函数机器码逐字节完全相同。raw
  short控制的算术内核未变，perf中raw L1I miss几何+0.737%与I-cache/BTB set映射变化
  一致。因此独立leaf section解决了大部分复制和主体布局问题，但还没有隔离header callsite；
  在能把整个类型化dispatch specialization移到链接末段前，总开关继续默认关闭。
- 严格JSON为
  `/home/ryz/tmp/packed-gemv-v3-exact-ool-strict-{short,wide}-r{1..4}-{b1,c1,c2,b2}.json`；
  单case计数为
  `/home/ryz/tmp/packed-gemv-v3-exact-ool-perf-{packed128,raw256}-r{1..4}-{b1,c1,c2,b2}.{json,perf}`，
  冻结binary、完整反汇编和符号表同在`/home/ryz/tmp/packed-gemv-v3-exact-*`。

### MATMUL-SME-X35：runtime量化INT8 1×N融合DOT原型

- 日期：2026-08-30。状态：隔离原型性能/正确性通过，因开放lambda参数ABI与40.8KiB
  wrapper重复尚未默认合入。参考KleidiAI `main@4bc7bd4`的SME1 1×4VL DOT schedule，
  在现有packed-B ABI上实现：每K64只在线量化一次FP32 A为U8，拆成四个128-bit quarter
  广播，64个packed-B load与64个`SUDOT`交错到四个accumulator；INT32 compensation
  作为初值，尾部`SCVTF × row-scale × column-scale`写FP32。DOT helper仍只在SME
  Backend且带TODO，leaf内`smstart/smstop`为0，状态由既有vec SME wrapper管理。
- 首次ARM覆盖运行反而发现COV026语义bug：transform只取`lane_coord(0)`并把row scale
  广播到全向量；AMX主要沿K vectorize所以没有暴露，SME pack沿spatial axis时不同lane
  属于不同行。修复为K轴保留scalar-fill fast path，其他axis逐lane使用
  `lane_coord(lane)`/`is_active(lane)`；修后baseline/candidate的14个注册全部reference通过，
  各56条aggregate record、0 error。
- 920f-4 CPU100、冻结其余22个shard、只替换shard22、B-C-C-B：batch=8 shared-B
  `1×128×128`为69.7028→4.5744us（-93.437%，15.24×），`1×256×256`为
  275.9639→14.3042us（-94.817%，19.29×），active几何-94.168%。12个control几何
  -0.709%，最坏正回退+0.526%；两个M=5表面-3.82%来自B1/B2漂移，不计作收益。
- 原型text 8,443,195→8,483,991B（+40,796B）；rank-2和projected-batch两个开放
  transform wrapper约0x4d50/0x4f40。`LaneLocalLambdaVecTransform`不暴露capture，单一
  raw leaf无法稳定取得row quant multiplier/input zp/row dequant/column scale。生产化优先
  引入内部named transform参数view并复用SME arch object；次选把一行A先量化到scratch再
  调共享DOT leaf，compute-only leaf最难保留完整收益。隔离报告和可应用patch分别为
  `/home/renyz/tmp/vecops-int8-fused-v1/REPORT.md`与
  `/home/renyz/tmp/vecops-int8-fused-v1/sme-int8-fused-gemv.patch`，远端完整产物在
  `/home/ryz/vecops-neo-int8-fused-v1/`。

### MATMUL-SME-X36：named-transform ABI与INT8融合DOT生产化

- 日期：2026-08-30。状态：最终门禁通过，ARM默认ON；可用
  `VECOPS_ENABLE_SME_RUNTIME_QUANT_INT8_EXTERNAL_LEAF=OFF`回退。新增内部
  `RuntimeQuantization.h`，用两个named transform稳定暴露row multiplier/input zp及
  row/column dequant参数，同时保留K-axis fill fast path与其他vector axis逐active-lane
  坐标语义。SME Backend据此识别精确pipeline，不再依赖匿名lambda capture布局。
- Kleidi式1×4VL leaf位于`src/arch/sme/RuntimeQuantInt8.cpp`，与F64 leaf作为不同object
  合入同一个按`-march+feature-set`生成的SME arch archive，成员顺序固定F64在前、INT8
  在后；静态链接器只对真正引用者抽取。量化/widening/SUDOT helper全部留在SME文件并带
  TODO，没有新增vec API或ACLE SME状态调用。leaf约0x6c0，对象text约1,916B，含64条
  SUDOT；`SMSTART/SMSTOP/USMOPA`均为0。
- 初版inline类型适配虽active约-95%，但未命中的shared-prefill稳定回退1.3%~2.1%；把
  精确pipeline adapter改为64B对齐noinline后，CPU100 B-C-C-B的两个active分别
  -95.087%/-95.697%，几何-95.402%。12个control几何+0.286%，prefill降至+0.412%；
  一次gemv-tail组合outlier独立B-C-C-B复核为+0.224%，没有稳定>1%回退。M=5 online、
  M=5 batch、K4097和N257均由命中计数确认回退原路径。
- combined archive对FP64完全隔离：4 active/8 control几何均约0，最差+0.066%，ON/OFF
  的F64 object SHA逐字节相同。nonFusion `MatmulSMEBench` ON/OFF也逐字节相同，SHA256
  `0212c9b04ae176df17f6e8f36c6a065ce3c5b73a29ed38521f7f9602ebc16bcc`，INT8 leaf
  符号0；x86配置不创建该选项或SME target。
- 最终默认ON Fusion `.text` 8,478,547→8,481,263B，仅+2,716B，相比X35开放lambda
  方案的+40,796B减少93.3%；binary SHA256
  `000b0a10e75f8223357e0d9907edb1e366b8455fcc18c04bbc8e69533b8e6f55`。
  SME Fusion分片测试6/6，默认ON全542 case/2168 aggregate record reference 0 error；
  本地GCC AMX Fusion 4/4。最终INT8严格数据为
  `/home/ryz/vecops-neo-int8-production-v1/int8-noinline-{b5,c5,c6,b6}.json`，tail复核为
  `int8-gemvtail-{b7,c7,c8,b8}.json`，F64隔离为`f64-int8-isolation-{b1,c1,c2,b2}.json`。

### MATMUL-SME-X37：packed MOPA显式prefetch与non-temporal store

- 日期：2026-08-30。状态：隔离实验完成；发现大工作集高收益prefetch候选，但因小输出
  long-K回退约20%尚未默认合入；NT收益过小，仅保留极大输出窄候选。参考AtomGit MR18
  的distance48/locality3，在现有packed MOPA loop分别测试A/B/AB、distance8/16/32/48、
  L1/L2共24候选；只用`vec::prefetch`，无SME状态ACLE。256²×4096 pilot筛选后7个
  shortlist对7类shape做严格B-C-C-B。
- 大工作集结果明确：AB/L2/d48在256²×4096、prefill、rank-expansion分别
  -11.62%/-12.23%/-12.50%，AB/L2/d32为-10.26%/-10.88%/-10.18%；AB/L1/d32
  仅-3.77%，A-only约-1.6%~-2.1%，B-only最佳接近持平。相反同一AB/L2候选在packed
  工作集仅几十KiB的long-K与short-N回退+18.8%~+21%，四条PRF及分支开销占主导。
  因此生产化必须同时要求PackedAB、L2、distance48/32，并按packed operand bytes或
  spatial tile工作集设大阈值；不能只看K长。
- NT路径用vec SME `read_hor`读ZA后`vec::store(non_temporal)`，静态含1005条STNT1，
  base为0。原最大2MiB C的no-consume/consume几何+0.251%/-0.441%，无明确价值；专项
  64MiB/256MiB/约134MiB-tail C在不消费时分别-0.681%/-0.802%/-1.379%，几何-0.954%，
  立即完整读取后仍仅-0.202%/-0.262%/-0.441%，几何-0.302%。收益真实但远小于参考固定
  workload，且普通输出因额外read_hor没有优势，不替换默认store_hor。
- 隔离树为`/home/ryz/vecops-neo-x33-mopa`，pilot/strict为
  `/home/ryz/tmp/x33-mopa-{pilot,strict-prefetch,strict-nt,strict-nt-large}`，汇编为
  `/home/ryz/tmp/x33-mopa-asm`。一次与clang重叠的strict整目录单独移到带
  `invalid-clang-overlap-20260830-114226`后缀且完全未计入结论；本实验最终没有采集
  PMU事件，不能用静态PRF/STNT数冒充cache miss证据。

### MATMUL-SME-X38：单边FP32-footprint的FP32→FP64 widening

- 日期：2026-08-30。状态：v5布局与端到端实验通过，识别出可生产化的online auto-pack
  门禁，但本阶段不改公共packed ABI、也不默认合入；下一轮应先把候选接入真实
  `execute_auto_packed_*`并做Fusion全量A/B。参考AtomGit MR18“只预widen高复用侧”的
  思路，目标是在FP32-memory/FP64-compute时只把一侧pack成FP64，另一侧保持FP32
  footprint，在packed K-loop中用连续`LD1W+ZIP+FCVT`扩宽；没有gather，widening helper
  仍只在SME实现内并带TODO。
- v1/v2因改变packed类型后丢失`FastPacked`、回到generic owner而全面大幅回退。v3把
  ZIP+FCVT并入现有`compute_packed_groups`后保留fast path：offline 16/18提升、几何
  -23.93%，但online因临时scalar triple-loop pack而几何+41.8%。v4首次复用ZA32 transpose
  时误把F64 packed panel当8 lane，只写半个实际16-element panel，fixed-one 36/36
  mismatch后立即停止，未采性能；这是“物理load宽度”不能替代packing panel ABI的明确坑点。
- v5复用生产ZA32 vertical transpose并完整连续写16个FP32 lane，再由kernel分成两个8-lane
  F64 block转换。baseline/candidate fixed-one均36/36、0 error/skip。CPU100、1000次、
  B-C-C-B初始结果：offline 15/18提升、几何-20.95%；online 18/18提升、几何-16.83%，
  范围-41.49%~-1.67%。candidate text增加8,184B（+2.18%），data/bss不变。
- 定点7 repetitions B-C-C-B确认强命中：wide/preA offline/online -49.46%/-46.76%，
  tall/preB -31.27%/-23.68%；online shared-B-medium两方向仍为-25.17%/-10.01%。明确
  不应命中的offline wide/preB与tall/preA仅-1.62%（高CV）/-0.39%，shared-B-medium
  两方向则稳定回退+17.02%/+18.66%。因此不能把半footprint变成新的公共MatmulPack默认
  ABI，也不能对所有FP32→FP64 packed operand启用。
- 生命周期可以在operation构造期安全区分：显式`PackedAInput/PackedBInput`继续使用现有
  双FP64公共ABI；只在raw→`AutoPackA/B`的内部workspace考虑v5。低风险候选门禁为
  `SMEF64ConversionPair && K>=256 && max(M,N)>=4*min(M,N)`；`N>=4M`预widen A，反向
  预widen B，square和ratio=2的shared-B-medium保持旧路径。该策略不需要在tile内复制
  catalog，但当前benchmark是显式pack+compute并用计时范围模拟online，尚未验证真实
  auto-pack workspace、Fusion specialization和新增构造期分支的布局影响，故本轮只记录
  候选，不把实验宏带入主工作区。
- 冻结产物位于`/home/ryz/vecops-single-widen-v1/frozen-v5`；严格原始JSON为
  `/home/ryz/tmp/single-widen-v5-*.json`与`single-widen-v5-target-*.json`。下一步若实现
  生产接线，必须同时测online active、显式offline control、ratio=2/square control、
  F64 native control、Fusion全量reference与text/函数地址，不能只复用本实验总体几何均值。

### MATMUL-SME-X39：named per-column skinny epilogue与benchmark分片

- 日期：2026-08-30。状态：v1计算收益确认，但header内实现因text与control回退拒绝；
  下一版必须把compute+epilogue一起移入arch object。新增内部稳定
  `PerColumnScaleTransform<T>`，避免architecture dispatch依赖开放lambda对象布局；只在
  candidate宏下允许该named lane-local transform走现有fused skinny compute。helper仍是
  matmul内部类型，没有加入公共vec API，也没有ACLE SME状态函数。
- 隔离benchmark同时覆盖BF16/FP16/FP32/FP64，per-column为active，dynamic-scale及所有
  非skinny shape/batch为control，共96个median。最初四种atom放在一个TU：baseline与
  candidate各需约8--9分钟，单进程峰值约15GiB。按`@vecops-target-shards: 4`以atom拆分
  后，baseline+candidate两份完整构建总wall为313.29秒，峰值RSS 4.38GiB；注册数仍为
  96，没有通过减少覆盖换取编译速度。后续组合型benchmark/test默认应按dtype/pipeline
  分片，而不是等单TU再次膨胀后处理。
- 分片版baseline/candidate均96 medians/384 aggregate records、0 error。CPU100、ASLR
  off、完整96-case B-C-C-B：8个双向skinny active几何-91.89%，范围-94.13%~-81.77%；
  BF16/FP16/FP32约-93.5%~-94.1%，FP64约-81.8%~-82.1%，证明per-column本身适合融合。
- 但candidate text从1,729,001增至1,820,461B（+91,460B）。88个control几何-1.39%看似
  偏快，范围却为-19.95%~+12.77%；FP32 medium per-column稳定表面+12.77%、BF16
  shared-small dynamic-scale +9.19%、FP32 medium dynamic-scale +7.35%，再次出现巨大
  header specialization改变函数地址/I-cache/BTB的布局污染。因此不能把“named type”
  本身当作隔离边界，也不能默认启用v1。
- 严格JSON位于`/home/ryz/vecops-lane-local-v1/build-clang/named-strict-{b1,c1,c2,b2}.json`；
  分片baseline/candidate SHA256分别为
  `13193c8947f6a6529c2ebc3126b8e90d1e05b6783686a00049c050d9f5eec153`和
  `798082e304d83196d9aa015d92122b5e48c6b87190aa07139846d18f921d93d7`。下一版应让header
  只做pointer/stride/scale抽取，BF16/FP16/FP32/FP64双方向compute+scale+store全部位于
  独立对象；只有该边界能同时保留约82%--94% active收益并避免91KiB前端膨胀。

### MATMUL-SME-X40：mixed-sign raw INT8 skinny外部DOT leaf

- 日期：2026-08-30。状态：最终门禁通过，ARM默认ON；可用
  `VECOPS_ENABLE_SME_MIXED_SIGN_SKINNY_EXTERNAL_LEAF=OFF`回退。精确门禁仅覆盖raw、
  Zero-C、S8xU8/U8xS8、M=1或N=1且输出数不超过64；其他shape、预打包输入和epilogue
  均保持原dispatch。实现位于`src/arch/sme/MixedSignSkinnyInt8.cpp`，使用普通SVE DOT，
  不进入streaming/ZA状态；widening-dot helper留在该SME文件并带后续泛化TODO。
- v1已确认16个active几何约-83.35%，但直接增加header adapter使旧same-sign leaf平移约
  320B，短K control出现无法归因的布局波动。v2把adapter放入独立64B对齐
  `.vecops_kernel_text`并让Clang使用`preserve_most`，同时将MatmulSMEBench按dtype拆成
  7个shard、把新增mixed-sign shard固定在最后；四个旧same-sign leaf地址、机器码和
  shard object SHA均逐字节不变。
- CPU100严格B-C-C-B：16个mixed-sign active几何-83.150%，范围
  -95.624%~-64.602%；16个same-sign control几何-0.152%，长K四点均在+-0.16%。唯一
  U8U8 M8N1K64初轮+3.66%的点，在对象/hash/address完全相同的前提下用C-B-B-C、
  2,000,000次复核为+0.905%，低于1%门槛，判定为时序漂移而非代码回退。
- OFF/ON的`--benchmark_list_tests`均输出338行，其中337行是实际case、1行是结果文件提示；
  mixed筛选40个唯一case/160 aggregate record均0 error。
  leaf object text/data/bss为7,276/8/1B，四个compute leaf为两个0x678和两个0x61c；
  反汇编含72条USDOT，SUDOT/SMSTART/SMSTOP/MOPA/USMOPA均为0。编译器通过交换可交换
  乘数把源级SUDOT规范化为USDOT。完整原始数据与报告位于
  `/home/renyz/tmp/vecops-mixed-sign-skinny-v2/`。

### MATMUL-SME-X41：BF16单边预打包GEMV全dispatch外置

- 日期：2026-08-30。状态：v4 active收益确认，但control仍有稳定超过1%的回退，拒绝
  默认合入。整个类型化BF16 packed-GEMV dispatch与compute均已移入arch object，header
  两个`run_operation`调用点仅增加约68B；12个旧skinny函数机器码逐字节相同，但链接
  地址仍平移128--256B。
- CPU100严格A/B的6个active全部提升，几何-84.482%，范围-87.596%~-80.289%；short
  controls几何仅-0.495%，但三个点分别稳定+0.373%/+0.505%/+1.030%，wide control仍有
  双峰。leaf反汇编含50条BFDOT、4条LD1RQH，且没有SME状态指令或MOPA。
- 结论不是packing/GEMV计算实现慢，而是当前巨型header binary对新增callsite和静态链接
  放置极敏感；仅外置compute尚不能固定fallback前端布局。补丁保留在
  `/home/ryz/tmp/packed-gemv-prod-v1.patch`，严格结果在
  `/home/ryz/tmp/packed-gemv-prod-v1-strict-{short,wide}`；默认继续OFF。下一次需要把
  dispatch owner也移到可稳定排序的独立section/object，并复测同一组short/wide control。

### MATMUL-SME-X42：真实auto-pack单边FP32-footprint widening v6

- 日期：2026-08-30。状态：真实`execute_auto_packed_*`接线与正确性通过，但header owner
  体积和未命中control风险不满足零回退门禁，默认OFF、不合入主工程。精确门禁为raw
  FP32xFP32->FP64、K>=256且有效输出长宽比至少4:1；宽shape只把A预扩宽，tall只把B
  预扩宽，公共`MatmulPack`仍输出标准双FP64 ABI。compact operand以SME-local
  `FP32FootprintForFP64Packed<Side>`事实标记，不把FP32内存误认成普通F64 packed layout。
- fixed-one baseline/candidate均26/26、0 error/skip；八个raw active全部提升，几何
  -28.56%，范围-44.17%~-10.97%。wide/tall为-44.17%/-24.12%，skinny K-tail为
  -42.69%/-43.82%，shared-B decode/tail为-23.73%/-12.56%。显式公共packed controls
  工作区逐字节不变，12点几何-1.34%、最差+0.28%。
- 但四个rank/batch x A/B mixed owner各约9.2KiB，binary text增加44,864B（+6.61%）。
  六个raw inactive controls几何+1.10%，ratio2、K255、shared-B ratio2初轮分别
  +1.86%/+5.45%/+1.13%；加长定点复核又出现正负翻转，说明是函数放置/相位敏感，但
  仍无法证明所需的零回退。机器码使用连续LD1W+ZIP1+FCVT、无gather和BL/BLR，问题
  不在compact load本身。完整报告位于
  `/home/renyz/tmp/vecops-auto-single-widen-v6/AUTO_SINGLE_WIDEN_REPORT.md`；下一版必须
  把四个owner移入共享SME arch object后再重复active/inactive/offline/tail/batch门禁。

### MATMUL-SME-X43：大工作集PackedAB MOPA L2 prefetch生产化

- 日期：2026-08-30。状态：严格门禁通过，默认ON；可定义
  `VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH`回退。X37已经证明A+B/L2/distance48在
  256²×4096、prefill和rank-expansion改善约11.6%--12.5%，但无门禁版本使packed
  footprint仅几十KiB的long-K/short-N回退约19%--21%。本轮只让BF16 PackedAB 2×2
  MOPA在完整A+B估算footprint至少2MiB时进入prefetch prefix；尾部48个K-group以及所有
  其他atom/tile继续执行原loop。门禁比较每K字节数与`ceil(2MiB/K)`，避免尺寸乘法溢出，
  也不会把“长K”本身误当成“大工作集”。
- 旧7-case严格集之外增加现有`conv_1x1_lowering 3136×64×576`命中点，以及门槛下方
  `batch_projection 128×1024×768`、`ragged_projection 127×1025×769`控制。CPU100、
  ASLR关闭、B-C-C-B、active固定50次×5：rank-expansion、square、prefill、conv分别
  -14.252%/-12.136%/-12.049%/-10.342%，几何-12.206%，两个独立pair全部同方向提升。
- false-gate controls按延迟分两组加长：4个small固定100,000次×3，变化
  +0.030%/+0.072%/+0.104%/+0.508%；两个约430us medium固定1,000次×3，变化
  +0.176%/+0.184%。初始仅10次的control曾显示-0.87%--+4.19%的虚假波动，加长后全部
  小于1%，因此不能用短采样判断未命中路径布局回退。
- OFF/ON `MatmulSMETest-Native`各17/17。ON text仅增加448B；7个MatmulSMEBench shard
  中只有BF16 shard0 object变化，其余6个逐字节相同。反汇编新增两份4×PRFH
  `PLDL2KEEP` prefix，合计8条静态PRF；没有新增状态调用，仍由vec SME wrapper持有
  StreamingZA。隔离树和原始数据位于`/home/ryz/vecops-prefetch-prod-v1`及本地
  `/home/renyz/tmp/vecops-prefetch-prod-v1/{strict-v1,strict-v1-controls,strict-v2-active}`。

### MATMUL-SME-X44：named per-column skinny arch leaf v2

- 日期：2026-08-30。状态：active收益保留，但runtime shape dispatch定点回退超过1%，
  默认OFF、不合入主工程。v2把BF16/FP16/FP32/FP64双方向compute+per-column scale/store
  全部移入`NamedPerColumnSkinny.cpp` arch object；adapter使用独立section和direct BL，
  避免函数指针产生8个PIE相对重定位。修复后OFF/ON `.rela.dyn`同为0x3888，66个dynamic
  executable符号名称、地址和机器码逐项完全相同，前4个dynamic shard object SHA也一致；
  没有ACLE SME状态函数，helper仍只在SME source并带TODO。
- CPU101 OFF/ON各96 unique/384 aggregate、0 error。CPU100完整96-case B-C-C-B的8个
  active几何-91.757%，范围-94.007%~-80.240%，证明外部leaf计算本身有效。binary text
  增加17,288B，data/bss不变；旧`.text`增量只来自4个per-column callsite约0x45c，named
  compute+adapter约0x39f0全部在末段独立section。
- 完整control短采样受相位影响很大：dynamic地址/机器码相同仍出现约-6%--+12%，所以
  不能把单点波动归因于candidate。最终用2,000,000次×3、C-B-B-C定点同shape对照：
  BF16/FP16/FP32/FP64 tiny per-column false-gate分别+1.994%/+2.372%/+1.604%/-0.589%，
  对应dynamic哨兵为+0.047%/-0.360%/+1.018%/+0.454%。前三类相对哨兵仍稳定超过1%，
  判定为inline runtime shape compare/dispatch的真实成本，而不是leaf或链接地址变化。
- 结论：下一版必须在operation构造期决定skinny owner，或让固定外部entry同时拥有fast与
  fallback；继续在每次`Backend::run`中添加shape gate不满足零回退要求。完整patch/report
  位于`/home/renyz/tmp/vecops-lane-local-v2-artifacts/`，patch SHA256为
  `75903d96d7ef5c94b5b3d1b3cca999ceeef17e3a7349f67f89aec35e8e948d0d`，基于当前X43
  主树apply-check通过；默认必须保持OFF。

### MATMUL-SME-X45：FP32-footprint/FP64-compute arch owner v7

- 日期：2026-08-30。状态：arch owner方向有效，但shared-B fallback稳定回退，默认OFF。
  v7把v6的rank2/batch×A/B四份约9.2KiB header owner收敛成A/B两个arch owner；只覆盖
  Zero-C、FP32 output Convert contract，其他epilogue回退标准双FP64 auto-pack。text增量
  从v6的44,864B（+6.61%）降到约21,956B（+3.35%）：arch leaf约19,124B，独立
  adapter/policy约1.8KiB，普通`.text`净增约320B。baseline/candidate各26/26正确性通过，
  仅8个active workspace变化，6 inactive与12 public packed逐字节不变。
- `SmallFloat.h`不必要包含`<iostream>`使arch TU产生本地`std::__ioinit`、额外
  `.init_array`和PIE relocation；改为语义足够的`<ostream>`后，OFF/ON `.rela.dyn`均577、
  `.init_array`均0xc8，data/bss为5,904/880B且地址完全相同。fallback owner归一化branch
  target/立即数后指令差异为0，但最终地址仍因新增代码平移约0x200--0x440。
- 8个active仍全部提升，几何-28.38%，范围-42.49%~-11.76%。controls存在强phase效应，
  因此对raw square、raw shared-B-ratio2、PackedAB skinny-row做逐case
  B-C-C-B-C-B-B-C、每轮至少0.5秒、5次复核：square总体-0.26%但pair正负翻转；
  skinny-row几何-0.90%通过；shared-B-ratio2四pair为-0.16%/+13.56%/+4.95%/+4.96%，
  pooled +4.94%、几何+5.72%，3/4稳定超过1%。默认ON门禁失败。
- 结论：mixed load/compute值得保留为实验patch，但还需把构造期dispatch与shared-B fallback
  放入稳定owner，不能仅依赖独立section。`<ostream>`头文件卫生修复可作为独立小patch
  合入，不应与v7性能路径绑定。v7 patch/report/results位于
  `/home/renyz/tmp/vecops-auto-single-widen-v7/`；完整patch SHA256
  `e9e5f95c547d4adaeda3cde977629454680d453421cc32989060e52aa252f69a`，SmallFloat最小
  patch SHA256 `b4d5bb7e1eed3c46fec02d8d2badc5d15b98e0756f7637bc8982af5011af004e`。

### MATMUL-SME-X46：packed-GEMV caller ABI下限与构造期descriptor原型

- 日期：2026-08-30。状态：v5/v6默认OFF；完成下一代operation ABI的本地原型。v4外部
  BF16 leaf已有active -84.482%，但两个caller各+68B、short control最差+1.030%。v5尝试
  direct raw ABI后caller各+124B；恢复typed minimal ABI仍各+128B，10/12旧skinny leaf
  统一平移256B。旧leaf机器码始终12/12逐字节一致；50 BFDOT/4 LD1RQH且无SME状态/MOPA。
  这证明膨胀下限来自AArch64 caller准备runtime m/n/k、A/B/C引用、call和status branch，
  不是DataAccess类型或leaf实现。
- 默认OFF最小patch为
  `/home/renyz/tmp/packed-gemv-prod-v6-current-default-off.patch`，报告为
  `/home/renyz/tmp/packed-gemv-prod-v6-report.md`；基于含X43/mixed-sign的当前主树
  apply-check通过。继续在全内联`Backend::run`缩小adapter不会生产化。
- 构造期descriptor v1在operation中保存16B描述符，保持`Backend::run`不变，通过固定
  `matmul_bound_dispatched -> run_packed_gemv_dispatched` entry在arch object内选择fast或
  同步fallback callback；DataAccess不逃逸，scratch、commit、online-pack复核和
  StreamingZA绕过均显式处理。其他backend由`if constexpr`+`no_unique_address`消除，
  本地AMX operation大小仍208B，GCC syntax/ABI smoke/CMake构建通过。
- 原型位于`/home/renyz/tmp/vecops-packed-gemv-descriptor-v1`，patch SHA256
  `5ec458e32d4d872eaabeb60819938a0f1fe6dd4cf76b4a995a2957c82bc4e997`。主要阻点是固定
  dispatcher必须对fallback-only ARM消费者也可链接，而当前SME arch archive仅由内部
  multiarch helper按引用抽取；下一步需建立公开ARM dispatch target，再做ARM compile和
  以fallback-only descriptor build为新baseline的间接调用门禁。

### MATMUL-SME-X47：SmallFloat头文件移除iostream全局初始化

- 日期：2026-08-30。状态：保留并合入。`SmallFloat.h`只声明/定义`std::ostream&`
  `operator<<`，却包含`<iostream>`；每个SME arch leaf TU因此生成本地`std::__ioinit`、
  `_GLOBAL__sub_I`、`.init_array`和额外PIE relocation。改为语义足够的`<ostream>`，不改
  SmallFloat类定义、operator ABI或数值行为。
- 最终统一SME archive中三个object的GNU text/data/bss分别从
  2,912/40/65、1,916/8/1、7,276/8/1B降到
  2,812/32/64、1,816/0/0、7,176/0/0B；每个object恰好减少100B text和8B data，
  `std::__ioinit`与`_GLOBAL__sub_I`符号均为0。archive SHA256更新为
  `1fdbab9d4a7927c7f38c45c182f3de18434f854f9c3e04f5fd979dc0059f8499`。
- 本地GCC重新构建AMX Matmul/Fusion和40-shard ConversionMemory；AMX 17/17、Fusion
  4/4、Conversion 38/38、ConversionMemory 63/63均通过。该修复与v7 auto-widen默认
  OFF结论独立，不会启用任何新SME路径。

### MATMUL-COV-027：Matmul大型benchmark/test编译单元分片

- 日期：2026-08-30。状态：保留。`MatmulSMEBench`按BF16、FP16、FP32、S8S8、U8U8、
  FP64、mixed-sign拆成7 shard；`MatmulAMXBench`按BF16、FP16、INT8拆成3 shard；
  `MatmulSMETest`和`MatmulAMXTest`分别按microkernel/meta、float/conversion、
  integer/packing、epilogue/quant拆成4 shard。`MatmulFusionSMETest`原有3 shard在最终
  编译中仍观察到单进程约9.8GiB峰值，进一步改为每个GTest case一个shard，共6 shard；
  最重单进程约5.1GiB，降低约48%。注册与测试仍由一个最终binary汇总，没有减少case覆盖。
- named-epilogue隔离实验中，四dtype单TU的每个binary约8--9分钟、峰值约15GiB；4 shard
  后baseline+candidate两份总wall 313.29秒、峰值4.38GiB。主工程SME benchmark
  native+fixed两套在加入mixed-sign前总wall 117.00秒、峰值1.27GiB；SME test两套总wall
  254.05秒、峰值3.39GiB且各17/17通过。本地AMX benchmark/test单次全新分片构建分别
  20.04秒/501MiB和42.62秒/911MiB，注册166项且17/17通过。
- 关键坑点：在已有build目录里仅给源码添加`@vecops-target-shards`并不保证CMake自动
  重新扫描注释；第一次增量构建曾只链接空壳原TU并显示0 tests。必须显式执行
  `cmake -S <src> -B <build>`后再构建，确认`generated/shards/.../shard_N.cpp`确实出现。
  后续新增大型组合benchmark/test时应从开始就按dtype或pipeline分片。

## 2026-08-30：SME 微内核与调度阶段审计

本节只讨论 cache tiling、K tiling 之外的微内核、epilogue 融合、函数放置和调度问题。
当前快照来自 920f-4/CPU100/BiSheng Clang 19.1.7、默认生产选项开启后的 542 个
Fusion 注册；原始文件为
`/home/ryz/vecops-neo-int8-production-v1/final-default-on-reference-542.json`，2168 条
aggregate record 全部 reference 通过。由于该共享主机的 `perf_event_paranoid=2` 且
当前用户没有 `CAP_PERFMON`，本阶段没有可靠的 MOPA 利用率或 stall breakdown；以下
“空间”来自冻结机器码的严格 A/B 和已命中路径，不把静态指令数冒充硬件峰值利用率。

### 当前性能画像

| 代表 pipeline | 当前 median | 观察 |
|---|---:|---|
| BF16 medium `64x256x256`，convert | 16.58 us / 506 GFLOP/s | 通用 dense 路径已进入较稳健区间；bias+clamp 约16.96 us，per-column scale 20.10 us（约慢21%）。 |
| BF16 wide `32x1024x256`，convert | 82.86 us / 202 GFLOP/s | wide 对 load/MOPA 顺序和代码布局敏感；双-bank实验在这类 Fusion 上曾回退约5%--6%。 |
| BF16 shared-B prefill `B=4,16x512x512` | 106.04 us / 316 GFLOP/s | batch flatten、shared-weight pack-once 与单 SME region 已有效。 |
| INT8 native medium `64x256x256` | 9.45 us / 888 GOP/s | 四种 signedness 的通用 MOPA 主体较强；tiny/skinny仍应优先 DOT/MMLA。 |
| BF16 raw skinny `1x8x4097` | 1.76 us / 37.2 GFLOP/s | BFDOT+Block8已生效；elementwise clamp/scale基本不增加成本。 |
| 同一 BF16 skinny，bias/per-column | 26.54/27.74 us | lane-local专用路径默认关闭后回落到通用路径，延迟约15--16倍，是当前最明显的调度洞。 |
| FP64 skinny clamp/dynamic scale | 6.57/6.59 us / 10.0 GFLOP/s | 外置F64 leaf有效；per-column未命中时为33.32 us，仍约5倍差距。 |
| runtime-quant shared-B decode `B=8,1x256x256` | 11.86 us / 88.4 GOP/s | named-transform SUDOT leaf命中；相对旧路径严格A/B几何改善95.40%。 |
| runtime-quant medium `64x256x256` raw | 771.19 us / 10.9 GOP/s | 不满足M=1/PackedB/full-block精确门禁，仍走通用transform+MOPA；说明数量级缺口在pipeline dispatch而非裸INT8算力。 |

不能据此宣称相对硬件峰值的百分比：当前没有可靠PMU权限，且Fusion数据包含pack、
transform和epilogue。可以确认的是，默认代码已经不再是单一通用MOPA实现，而是按顺序
选择普通SVE skinny、融合skinny、runtime-quant INT8 arch leaf、PackedAB MMLA，最后才
进入Streaming-ZA/MOPA。Pack本身已有ZA32/ZA64转置、双tile charge/drain、full-panel与
full-K特化和多种conversion postprocess；继续重写普通转置的低风险普适收益已经很小。

### 微内核层面的剩余空间

- 通用 packed 2x2 MOPA当前每K group为4 load+4 MOPA、loop约15条指令且64B对齐。
  双-bank软件流水已把两个group从30条降到25条，square cycles改善4.53%、wall改善
  8.42%，但wide/部分epilogue回退4%--6%。因此近期可兑现的通用dense核心空间约
  5%--10%，前提是把流水化owner独立放置；继续在Tile2D模板内复制两套loop不成立。
- 大工作集PackedAB的A+B/L2/d48预取已在三类shape改善11.62%--12.50%，这是另一个
  约10%的高可信空间；但小输出long-K回退18.8%--21%，必须按packed bytes和空间tile数
  在traversal外门禁，不能按K长单独判断。
- raw/fused skinny是目前最成熟部分：BF16 BFDOT相对旧widening-FMA几何改善65.27%，
  Block8再改善9.61%；FP16 FMLAL改善53%--66%，同符号INT8 DOT改善75.35%。对已命中的
  elementwise pipeline，继续优化空间大致为0%--15%；对mixed-sign INT8、lane-local
  epilogue和one-side-packed输入则仍可能是数量级，但必须逐pipeline实测。
- PackedAB MMLA生产门禁的两批active分别几何改善41.45%和40.40%，fallback最差约
  +0.49%。专用MMLA pack在inner probe中比现有MOPA-pack+ZIP还快BF16 17%--21%、INT8
  26.7%，但这是inner-loop上限，只有离线高复用权重才能摊薄第二种pack ABI的成本。
- FP32/F64极少输出长K仍是一条accumulator依赖链；可以尝试每输出2--4条独立K链再归约，
  但当前没有A/B数据。F32MM/F64MM不可用，`SME_FA64`也不等价于这些矩阵扩展，因此不能
  把MMLA作为FP32/F64通用答案。
- 巨大write-only C的non-temporal store只改善约0.95%，若随后消费仅约0.30%；它是
  低优先级窄门禁，不是微内核主线。

### 当前 experimental/可回退路径总表

| 开关/路径 | 默认 | 有提升的情况 | 回撤或限制 | 结论 |
|---|---:|---|---|---|
| `VECOPS_EXPERIMENTAL_SME_PACKED_PIPELINE` | OFF | BF16 PackedAB 2x2，square cycles -4.53%、wall -8.42% | Fusion wide/部分epilogue +4%--+6%；模板内双loop还会增大text | 保留实验，等待独立owner。 |
| named per-column skinny arch leaf v2 | OFF | 8 active几何-91.757% | tiny BF16/FP16/FP32 false-gate +1.994%/+2.372%/+1.604% | leaf有效；runtime shape dispatch未生产化。 |
| `VECOPS_EXPERIMENTAL_SME_PACKED_GEMV` | OFF | BF16单边packed/shared-raw，X34 exact active几何-84.624% | short control稳定最差+1.406%，wide仍受I-cache/BTB布局影响 | 计算已验证，需把整个类型化dispatch外置。 |
| `...PACKED_GEMV_ALL_TYPES` | OFF | 全网格几何-12.19%，BF16 -13.12%，S8U8 -19.95% | F32/F64仅-1.21%/-0.46%；部分PackedAB/大输出long-K回退2%--11%，text曾+103KiB | 拒绝宽泛门禁。 |
| `...DISABLE_RAW_SHARED_UNROLL` | 未定义 | 仅用于隔离BF16 Kx4 leaf贡献 | 没有独立生产收益 | 诊断开关。 |
| `...LEGACY_INLINE_RAW_SHARED_UNROLL` | 未定义 | 6个命中相对旧K-major几何再改善61.92% | 复制leaf，43个control出现-42%--+26%布局波动 | 已被单份noinline leaf替代，不应启用。 |
| `...OUT_OF_LINE_DISPATCH` | 未定义 | 配合保守Packed GEMV时保留-84.624% active | header callsite仍增长约68B并使后续skinny leaf平移128B | 方向正确但隔离边界还不够外。 |
| F64 external leaf CMake option | ON | 4 active几何-81.49% | 8 control几何+0.01%，最差+0.41% | 已生产化，可OFF紧急回退。 |
| runtime-quant INT8 external leaf CMake option | ON | 两active -95.09%/-95.70% | 12 control几何+0.286%，tail复核+0.224%；门禁仅M=1/PackedB/full block | 已生产化，应扩named pipeline而非放宽匿名ABI。 |
| mixed-sign INT8 skinny external leaf CMake option | ON | 16个S8xU8/U8xS8 raw active几何-83.150% | same-sign control几何-0.152%，最差复核+0.905%；仅Zero-C、M/N=1、输出<=64 | 已生产化，可OFF紧急回退。 |
| BF16 PackedAB大工作集L2/d48 prefetch | ON | 4个>=2MiB active几何-12.206% | 6个false-gate control最差+0.508%；仅2×2 MOPA | 已生产化，可用disable宏回退。 |
| FP32-footprint/FP64-compute auto-pack v7 | OFF，隔离补丁 | 8 active几何-28.38%；text增量减半 | shared-B-ratio2逐case pooled +4.94% | arch owner有效，构造期/shared-B dispatch仍需外置。 |
| MMLA experiment benchmark | OFF，仅构建选项 | 隔离72点几何-32.57%，生产门禁内约-40% | 只增加诊断target，不影响生产binary | 保留作阈值/布局实验。 |

### 下一批优化候选

优先级按已测收益、可生产化概率和回退风险排序：

1. 继续packed-GEMV构造期descriptor v1：先建立fallback-only消费者也能链接的公开ARM
   dispatch target，再在920f-4验证固定entry/callback ABI；旧header status branch路线已到
   +128B caller下限，不再继续微调。
2. 把named per-column skinny的shape/owner选择移到operation构造期；v2 leaf active
   -91.757%且布局隔离已完成，但每次run的inline gate仍使三种tiny control回退1.6%--2.4%。
3. 把FP32-footprint/FP64-compute选择与shared-B fallback一起移入构造期稳定owner；v7
   active -28.38%、text增量已减半，但shared-B-ratio2 pooled仍+4.94%。
4. 大工作集`PackedAB + L2 + distance48`预取已按2MiB footprint门禁生产化并改善
   12.206%；后续只需用真实workload校准阈值，不得退回仅按K长度判断。
5. 把packed 2x2双-bank MOPA变成traversal外的单份owner/汇编边界，目标约5%--10%；
   不能使用与当前vec SME状态模型冲突的ACLE强制状态管理。
6. 只对离线高复用权重评估专用MMLA pack ABI，并把pack-once成本纳入端到端计时；inner
   probe上限为BF16 17%--21%、INT8约27%。
7. 扩展runtime-quant named leaf到N/K tail及少量高价值M>1形状；当前exact GEMV已快95%，
   而未命中medium仍只有约10.9 GOP/s，收益空间很大但实现复杂度也最高。
8. mixed-sign raw INT8 skinny已经生产化并改善83.15%；后续只在有实际workload证据时扩展
   packed输入、非Zero-C或更大输出，不能放宽现有精确门禁。
9. INT8 direct loader的full-K versioning只能做单份loop；旧复制catalog方案使符号近3倍、
   tail回退约61%，不得重复。
10. FP32/F64 skinny尝试2--4条独立K依赖链，以及F64 packed area-8独立owner；两者当前均
   未量化，优先级低于已有严格A/B证据的候选。
11. non-temporal store仅面向极大且短期不复用的C，预期小于1.5%，最后考虑。

总体判断：Pack约处于成熟阶段，raw/elementwise skinny最强，MMLA覆盖窄但门禁可靠；
通用MOPA仍有约5%--12%的调度空间。最大的剩余收益来自“已有专用微内核如何在不扰动
巨大header catalog布局的前提下生产化”，而不是继续增加通用tile family、无条件展开或
泛化runtime分支。

### MATMUL-REG-002：本阶段最终主工程门禁

- 日期：2026-08-30。主机920f-4，源码同步到`/home/ryz/vecops-neo`，构建目录
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release`。显式配置
  `CMAKE_C_COMPILER=$(which clang)`、`CMAKE_CXX_COMPILER=$(which clang++)`、Release、
  benchmarks/tests ON、fixed streaming SVE=512；实际编译器为BiSheng Clang 19.1.7，
  feature set含BF16/I8MM/SME/SME_FA64/SME_F64F64，kernel目标实际带
  `-falign-functions=64 -falign-loops=64 -fno-math-errno`。
- 最终运行门禁全部通过：SME Fusion 6/6，Matmul native 17/17，fixed-SVL 17/17，
  MatmulPack 6/6，TensorTransform 7/7，ExecutionSession 1/1。本地x86 GCC Release
  同时完成AMX Fusion/主Scenario/主Matmul等目标构建；AMX Fusion 4/4、ExecutionSession
  1/1，14个runtime-per-row注册产生56条aggregate record且0 error。
- CPU100运行最终`MatmulFusionScenarioBench-Native`全量：542 medians/2168 aggregate
  records、0 error，14个runtime-per-row场景全部在内。原始文件为
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release/final-fusion-542.json`；
  最终Fusion SHA256为
  `9651ffc84fa985f214e5a78d3048eb911f5d102ac7d60d09ed5d5aca1ef2b2a5`，GNU size
  text/data/bss为8,481,247/14,216/1,120B。
- SME arch归档SHA256为
  `cf4ca0f7ff5461ae9a5f74f95c4e8751bce699d6fe4e3f94d9bee66a3715c8a4`；F64/INT8对象
  GNU text分别2,912/1,916B。归档反汇编含64条SUDOT，`SMSTART/SMSTOP/MOPA`均为0。
  非Fusion`MatmulSMEBench-Native`和`ExecutionSessionTest-Native`的两个external leaf符号
  计数均为0，证明静态归档没有被无关目标抽取。两个binary SHA256分别为
  `1c65a6b95ecca13bf94d49884db23fddb6539a082423d32dca0d60721bff2c4a`和
  `36c864ce8e3fd4c4dc31d288a292fb5afc1136f9ea7448606d107efa2354bb57`。
- 本阶段所有widening-dot/FMLAL/DOT/MMLA helper仍只位于SME backend或SME arch source，
  均带待泛化TODO；公共vec模块没有新增这些API。状态管理继续由vec SME wrapper负责，
  external leaf与实验候选均未引入ACLE SME状态函数。

### MATMUL-REG-003：mixed-sign leaf与编译分片最终集成门禁

- 日期：2026-08-30。主机920f-4，源码`/home/ryz/vecops-neo`，构建目录
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release`；显式使用BiSheng
  Clang 19.1.7并重新执行CMake配置，feature set仍为BF16/I8MM/SME/SME_FA64/SME_F64F64，
  fixed streaming SVE=512。最终Fusion test从3拆到6 shard后，干净单线程重编最重shard0
  wall 3:02.61、最大RSS 5,380,816KiB；原3-shard同内容峰值约9.8GiB，下降约48%。
- 23-shard Fusion benchmark、6-shard Fusion test及ExecutionSession的最终重编wall
  10:50.11，`time -v`最大单进程RSS 7,181,016KiB。曾中断的旧clang子进程被精确终止后，
  对可能重叠写入的shard0显式touch并单线程完整重编、重链接；最终门禁不使用争用期间的
  object。SME Matmul native/fixed各17/17，Fusion 6/6，ExecutionSession 1/1；本地GCC
  AMX Matmul 17/17、Fusion 4/4、ExecutionSession 1/1。
- CPU100、ASLR关闭的最终native主benchmark为337 medians/1,348 aggregate records，
  wall 52.77秒、无skip/error且进程正常退出；JSON为
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release/final-matmul-337.json`，SHA256
  `f278e1bee7e6847b8498fc09fa4f64da3367f4517f28d9219cde991011666265`。Fusion仍为
  542 medians/2,168 aggregate records，wall 51.99秒、无skip/error；JSON
  `final-fusion-542-mixed.json`，SHA256
  `5dc9006e52744ee4e8f0fec832e27639c8020beb437ceba11cabaa365d590b68`。
- 最终`MatmulSMEBench-Native` SHA256
  `f76f444dad6079786a5ea6071670294452844c9bc99ad025cd031f6b836d71fe`，text/data/bss
  1,445,801/12,424/1,112B，并且只抽取两个mixed-sign dispatcher。Fusion benchmark SHA
  仍为`9651ffc84fa985f214e5a78d3048eb911f5d102ac7d60d09ed5d5aca1ef2b2a5`，与REG-002
  逐字节相同且mixed符号数0；ExecutionSession SHA仍为
  `36c864ce8e3fd4c4dc31d288a292fb5afc1136f9ea7448606d107efa2354bb57`、mixed符号数0，
  证明新增默认ON leaf不会被无关pipeline抽取。
- SME arch archive SHA256为
  `14b3e57c90bc87b571ed2791c6769b21285cebe1514e183cdaf3a3e62d744045`，成员固定为
  `FusedSkinnyF64.cpp.o`、`RuntimeQuantInt8.cpp.o`、`MixedSignSkinnyInt8.cpp.o`，GNU text
  分别2,912/1,916/7,276B。归档反汇编合计72条USDOT、64条SUDOT，
  `SMSTART/SMSTOP/MOPA`均为0；mixed-sign helper继续只存在SME arch source中，没有加入
  公共vec模块，也没有引入ACLE SME状态管理。

### MATMUL-REG-004：X43/X47最终收尾门禁

- 日期：2026-08-30。状态：通过并冻结；按用户要求停止本轮目标，不再启动新优化实验。
  主机920f-4，源码`/home/ryz/vecops-neo`，构建目录
  `/home/ryz/vecops-neo/cmake-build-sme-pack-arm-clang-release`；显式配置
  `CMAKE_C_COMPILER=$(which clang)`、`CMAKE_CXX_COMPILER=$(which clang++)`、Release、
  tests/benchmarks ON、fixed streaming SVE=512，实际为BiSheng Clang 19.1.7。
- 最终重编目标包括MatmulSMEBench native/fixed、MatmulSMETest native/fixed、6-shard
  Fusion test、23-shard Fusion benchmark和ExecutionSession；wall 17:42.57，最大单进程
  RSS 7,181,448KiB，exit 0。SME Matmul native/fixed新增footprint gate单测后各18/18，
  Fusion 6/6、ExecutionSession 1/1。本地GCC重编AMX/Conversion相关目标wall 4:26.12，
  最大RSS 1,777,332KiB；AMX Matmul 17/17、Fusion 4/4、Conversion 38/38、
  ConversionMemory 63/63全部通过。
- CPU100、ASLR关闭的最终主benchmark为337 medians/1,348 aggregate records，wall
  53.62秒、0 error/skip；JSON `x43-final-matmul-337.json` SHA256
  `c8612cdcac56f3b3d11a681ce836e5ca0f5eb101c6c8336ed1b4ce3936e59df7`。Fusion为
  542 medians/2,168 records，wall 52.14秒、0 error/skip；JSON
  `x43-final-fusion-542.json` SHA256
  `b2fd6d7a682bd44deff307fba76107bb7b064f71397c678deef0e6055d6ade73`。
- 最终`MatmulSMEBench-Native` SHA256
  `ba3601a429c9448ede3638bd5eb5b31fcb1345054096cd67e215675d1fb310a7`，text/data/bss
  1,446,177/12,416/1,112B；反汇编静态PRF为142条，相比X42基线134条恰增加两份4×
  `PRFH PLDL2KEEP` prefix。Fusion binary SHA256
  `3ddff7b99a0a64c1b05f0158907873ceb7567c85373aefd6da92fe6f87eb6d96`，text/data/bss
  8,494,159/14,200/1,120B。
- 最终SME arch archive SHA256
  `1fdbab9d4a7927c7f38c45c182f3de18434f854f9c3e04f5fd979dc0059f8499`；三个object
  text/data/bss为2,812/32/64、1,816/0/0、7,176/0/0B，`std::__ioinit`为0。归档反汇编
  仍为72 USDOT、64 SUDOT，`SMSTART/SMSTOP/MOPA`均为0。ExecutionSession SHA仍为
  `36c864ce8e3fd4c4dc31d288a292fb5afc1136f9ea7448606d107efa2354bb57`，说明X43/X47
  没有污染无关执行路径。
- 本轮最终默认ON新增项只有X43大工作集prefetch与X47头文件初始化修复。X44 lane-local、
  X45 auto-widen、X46 packed-GEMV均因可重复>1% control回退保持默认OFF；补丁、报告和
  原始数据已冻结，后续只有实际profile证明这些pipeline重要时才启动operation构造期
  dispatch ABI重构。通用dense/packing预计只剩约3%--8%低风险空间，进入收益递减阶段。

## 2026-08-30：测试与 benchmark 统一 catalog

### MATMUL-COV-028：架构中性目标、Dynamic/Const 配对与编译分片

- 日期：2026-08-30。状态：完成。测试产物统一为 `MatmulTest`、
  `MatmulBatchTest`、`MatmulFusionTest`、`MatmulPackTest`，MixedPacking 并入 Core；
  benchmark 统一为 `MatmulBench`、`MatmulPackBench`、`MatmulFusionBench`、
  `MatmulPackedBench`、`MatmulPolicyBench` 和 `MatmulMMLAExperimentBench`。
  `MatmulScenarioBench` 与 `MatmulWeightReuse*Bench` 继续独立，旧 AMX/SME/Fixed 目标
  不提供别名。历史目标到新目标的完整映射见
  `benchmarks/ops/MatmulCoverage.md`。
- 物理源文件同步完成合并：benchmark 只保留 `MatmulBench.cpp`、
  `MatmulPackBench.cpp`；测试只保留 `MatmulTest.cpp`、`MatmulBatchTest.cpp`、
  `MatmulFusionTest.cpp`、`MatmulPackTest.cpp`。旧 AMX/SME 命名源已删除，GTest suite
  也改为架构中性名称。AMX permission 集中在 `MatmulTestArch.h`，其余 Atom/SVL/
  F64 等差异留在同一源内部的 traits/能力分支。
- 所有普通正向 shape 由同一 catalog 自动注册 `/extent:Dynamic` 与
  `/extent:Const`。Dynamic 使用 `Any`，Const 使用 `cint<>`；二者共享输入、stride、
  packing、pipeline 与 reference oracle。bounded Dynamic、混合 extent、动态 stride、
  zero、death test、runtime-SVL 和反汇编检查仍作为专项覆盖。
- 大型源全部改用 `@vecops-target-shards`：Core/Scenario/Fusion/Packed/WeightReuse
  分别为 176/306/396/480/260 shard。Fusion/F64/量化等重 tuple 按 case 继续切分；
  Core 与 WeightReuse 的 pair 放在同一 TU 并引入显式 `ShardTag`，避免宏依赖模板体被
  ODR 合并后产生重复注册。ARM 统一源最终重建最大 RSS 5,964,692 KiB，本地整轮构建最大 RSS
  1,601,592 KiB，分别低于 6 GiB/2 GiB 门限。
- shard marker 新增按架构选择的通用形式，同一个源可声明 x86/ARM 两套数量。Core
  benchmark 为5/176，Pack benchmark 为8/10；Core/Batch/Fusion/Pack测试为
  5/5、7/9、2/8、9/6，避免物理合并后在x86生成大量空SME shard。
- 统一源最终在920f-4以`-j16`重建12个Native/Fixed目标，wall 36:10.21、峰值
  5,964,692 KiB。Core/Pack配对审计仍为337/674与126/252；四个对应benchmark
  fixed-one共1,852 variant、0 error/skip，8个MOPA/call反汇编门禁全部通过。
- `audit_matmul_extent_pairs.py` 同时检查 malformed、孤立 pair 和完全同名重复注册；
  `compare_matmul_extents.py` 输出 Dynamic/Const median、双方 CV 与 Const/Dynamic 比例。
  本地 9 个二进制共 2,060 个逻辑 case、4,120 个注册，全部 0 malformed/unpaired/
  duplicate；ARM Native/Fixed 以及可选 Policy/MMLA 全部通过相同审计。
- 本地 Release 36/36、Debug 39/39；920f-4 Native 与 Fixed 各 40/40，Fixed 运行时
  SVL=512。ARM 20 个正式 benchmark fixed-one 总计 11,440 variant，0 error/skip；
  Native/Fixed 八个 MOPA/call 反汇编门禁全部通过。基线 tuple 归一化后 Core、Pack、
  Scenario、Fusion、Packed、WeightReuse 的缺失数均为0。
- 性能代表集固定本地 CPU40、ARM CPU100 串行执行。x86 五项 Const/Dynamic 为
  0.9668/0.9684/0.9894/0.9670/0.9762；ARM Fixed 五项为
  0.8186/0.7898/0.8061/0.7749/0.7901。ARM Native packedAB tail 单项为1.6386，按约定
  只报告、不阻断。反汇编中 packedAB microkernel、raw auto-pack、Fusion 分别剪除
  208B/52指令/2分支、108B/27指令/2分支、420B/105指令/7分支。完整数据与复现命令见
  `benchmarks/ops/MatmulExtentAB.md`。
- 最终环境：本地 Release/Debug 构建目录为
  `/home/renyz/project/vecops-neo/cmake-build-matmul-unified-gcc-{release,debug}`；ARM
  为 920f-4 的 `/home/ryz/vecops-neo/cmake-build-matmul-unified-clang-release`，使用
  BiSheng Clang 19.1.7、显式 clang/clang++、Release、tests/benchmarks ON 和
  `VECOPS_FIXED_STREAMING_SVE_BITS=512`。

## 实验记录模板

```text
### MATMUL-<ARCH>-<NNN>: <标题>

- 日期/版本：
- 假设：
- 修改：
- 主机/编译器/flags：
- benchmark/filter：
- 测量协议：
- wall time：
- perf counters：
- 汇编/符号大小：
- 正确性与门禁：
- 结论：保留 / 拒绝 / 继续
- 后续：
```
