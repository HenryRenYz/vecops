# Transpose benchmark、静态分析与优化记录

## 1. Benchmark 覆盖

`TransposeBench-Native` 当前注册 399 个 case。每个逻辑 case 都分别生成：

- `shape_meta:Const`：两个维度均为 `Const<M/N>`；
- `shape_meta:Dynamic<MxN>`：两个维度均为带最大可证明 2 的幂对齐的
  `Dynamic<Alignment>`；
- `shape_meta:Any`：两个维度均为无约束的 `Any`（即 `Dynamic<1>`）。

三类元数据各 133 个 case。这里的 `Dynamic<16x32>` 表示行维度为
`Dynamic<16>`、列维度为 `Dynamic<32>`，不是一个二维类型。

同类型覆盖 `int8`、`fp16`、`bf16`、`fp32`、`fp64`，并在全部 17 个
shape 上运行。跨类型覆盖 1/2/4/8 字节代表类型之间的全部 12 个非对角
组合，并放在 `16x16`、`63x129`、`256x256`、`1024x1024` 四个小/尾/中/大
代表 shape 上。跨类型的 `compute_dtype` 选为能够表达输入和输出转换的
`fp32` 或 `fp64`。

shape 分为：

- 通用小/中/大矩阵：`16x16`、`32x64`、`256x768`、`768x256`、
  `1024x1024`、`4096x4096`；
- 非整块和极小矩阵：`3x5`、`17x19`、`63x129`；
- decode：`1x32`、`128x128`；
- attention layout：`2048x28`、`4096x64`、`8192x128`；
- Q/K head plane：`2048x128`、`4096x192`、`4096x256`。

这些维度来自模型发布方的配置：Qwen2.5-7B 使用 28 个 attention heads；
Llama 3.1 8B/70B 使用 32/64 个 heads，405B 使用 128 个 heads；
DeepSeek-V3 使用 128 个 heads，Q/K 的非 RoPE 与 RoPE 维度为 128+64；
Gemma 2 的典型 `head_dim` 为 256。序列长度同时覆盖 decode、2K、4K、8K
prefill。参考：

- <https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json>
- <https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py>
- <https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_v3.1.json>
- <https://huggingface.co/docs/transformers/model_doc/gemma2>

## 2. 构建组织

最初把所有模板实例放在一个翻译单元中，GCC 编译超过 21 分钟且峰值常驻
内存达到约 9.3 GiB。现在同类型各自独立，12 个跨类型 pair 又按输入宽度
拆成 4 个翻译单元。固定 SVE 构建中，原 conversion 单进程约 16 GiB，拆分
后约 4 GiB/输入宽度，并可并行和增量构建。该拆分不改变 benchmark 名称
和被测代码。

ARM 毕昇 Clang 的 SME benchmark 需要关闭 SME attribute 的兼容性告警，并
显式使用 `compiler-rt` 与 `libgcc` unwind。该选项只在 ARM、Clang 且目标
含 SME 时启用。

## 3. 静态分析和机器码

### x86（Intel Sapphire Rapids，GCC，AVX-512）

- 原始同类型主块走 fixed register network：连续 load，随后使用
  `vpunpck*`/concat 递归网络转置，再连续 store。
- GCC 对 32/64 位元素限制寄存器行数，以避免 16x16 AVX-512 网络的寄存器
  溢出；int8/fp16/bf16 使用 16 行，fp32 使用 8x16，fp64 使用 8x8。
- 跨类型和原尾块走 indexed gather。例如 int8 到 fp32 可见
  `vpmovsxbd`、`vcvtdq2ps`，尾部可见 `vpgatherdd`。
- 本轮保留的底边路径仍使用连续 load 和 `vpunpck*` 网络；不足的行通过
  k-mask `vmovdqu16`/对应元素宽度的 masked store 写出。`vpgatherdd` 只保留
  在右边和角落窄尾块中。

### ARM（920f-4，毕昇 Clang，SVE/SVE2/SME，VL=SVL=64 B）

- 原始同类型选择 SME。机器码包含 `smstart za`、编译器生成的 `zero {za}`、
  向 ZA 水平 slice 写入、从垂直 slice 读取以及 `smstop za`。
- `__arm_new("za")` 会要求编译器建立新 ZA 状态，因此 `zero {za}` 不能只靠
  删除源码中的显式清零来消除。
- 跨类型现在也可选择 SME：以 compute dtype 作为 ZA 元素宽度，在 ZA 写入前
  `load_convert`，从 ZA 读出后 `store_convert`。`int8->fp32` 的最终主循环可见
  `ld1b`、`scvtf`、`mov za0h.s`、`mov za0v.s`、`st1w`，tile 循环内没有
  `smstop/smstart`。
- benchmark 把 streaming region 放在计时循环外；计时中的主要固定成本是
  ZA 建立，而不是每次迭代重复切换整个 streaming mode。

### ARM SME、SVE-VLA 与 SVE-VLS 对比

新增精简的 `TransposeCompareBench`，只保留 `63x129`、`128x128`、
`1024x1024`、`4096x4096`，分别构建：

- Native：`-march=native+bf16+sme`，同类型自动选择 SME；
- SVE2-VLA：`-march=armv8-a+sve2`，sizeless SVE 自动走 gather；
- SVE2-VLS：额外使用 `-msve-vector-bits=512`，自动走 fixed butterfly。

固定 CPU 0，5 轮正反序运行，fp32 Const 平均中位数如下：

| shape | SME | SVE-VLA | SVE-VLS | SME/VLA 加速 | VLS/VLA 加速 |
|---|---:|---:|---:|---:|---:|
| 63x129 | 1.76 µs | 2.38 µs | 20.02 µs | 1.35x | 0.12x |
| 128x128 | 2.76 µs | 17.90 µs | 23.32 µs | 6.49x | 0.77x |
| 1024x1024 | 476.4 µs | 1777.2 µs | 1556.4 µs | 3.73x | 1.14x |
| 4096x4096 | 6.256 ms | 142.721 ms | 24.391 ms | 22.82x | 5.85x |

4096 平方的 Const 结果按类型为：

| dtype | SME | SVE-VLA | SVE-VLS | SME/VLA 加速 |
|---|---:|---:|---:|---:|
| int8 | 2.220 ms | 87.632 ms | 69.135 ms | 39.48x |
| fp16 | 3.759 ms | 128.245 ms | 18.354 ms | 34.12x |
| bf16 | 3.780 ms | 121.357 ms | 18.382 ms | 32.10x |
| fp32 | 6.247 ms | 139.341 ms | 24.357 ms | 22.30x |
| fp64 | 10.972 ms | 135.816 ms | 38.133 ms | 12.38x |

表中 fp32 VLS 已包含完整-word predicate 优化；其余 VLS dtype 来自优化前
二进制，因此是保守值。该优化只移除 predicate 构造，不改变 fixed network，
不会改变大/小矩阵的趋势判断。

VLS 只在大矩阵上明显优于 VLA；小矩阵和尾块反而更慢。因此 fixed-SVE 的
workaround 必须是按规模和尾块比例选择 gather/fixed，而不能全局切换。
`VECOPS_FIXED_SVE_BITS` 只应用于精简的 compare target，不应用于全量
Transpose benchmark，避免默认构建重复实例化全部 fixed network。
`Const`、对齐 `Dynamic`、`Any` 在 VLA 中几乎完全相同，因为三者都受
sizeless vector 不能放入 `std::array` 的限制而走 gather；metadata 不能解决
这个类型系统约束。VLS 中三者也没有稳定、可复现的排序，后端算法选择才是
主导因素。

### SME helper 去重

SME 中按 1/2/4/8 字节重复分发的 zero、lane count、predicate、load/store
已替换为通用 `vec::zeros/size/mwhilelt/load/store/bitcast`。ZA horizontal
write 和 vertical read 没有通用 `vec` 等价物，继续保留为 SME bridge。
在 fixed-SVE+SME 同时启用时，`Vec/Mask` 是 sized wrapper，而 ZA intrinsic
使用原生 SVE word/predicate；该边界通过现有 `sve_basic_wrap_word`、
`sve_basic_raw_word` 和单 word mask bridge 处理，不恢复 dtype 分发。

初版替换暴露了通用 SVE masked memory 对完整 scalable word 仍生成冗余
full predicate 和 predicate-and。修复通用 SVE memory backend 后，fp16
1024 平方 SME leaf 恢复到与旧 helper 相同的 0x168 字节；指令流仅有非负
计数下等价的 `whilelt`/`whilelo` 形式差异。7 轮 old/new 性能均在 2% 噪声
阈值内，因此保留通用实现。

### SME 跨类型转换

SME 后端现接受 rank-2、`NoTransform`、源/目标内层 stride 为 1 的 raw access，
不再要求 memory dtype 等于 compute dtype。ZA 始终使用 compute dtype，三条
路径由类型在编译期选择：

1. source == compute：`load + transpose + store_convert`；
2. destination == compute：`load_convert + transpose + store`；
3. 两端都不等于 compute：`load_convert + transpose + store_convert`。

这没有绕过 DataAccess 的转换语义；尤其第三条仍保留
`Source -> Compute -> Destination` 两次转换。完整 tile 使用无 mask 的通用
`vec::load_convert/store_convert`，真正尾块才使用 masked 版本。

首版机器码发现通用 SVE 转换内部的无参数 lambda 被毕昇 Clang outline 成
non-streaming helper，导致每一行宽化时插入 `smstop -> call -> smstart za`；
int8 到 fp32 甚至每行切换两次。为避免在 SME 内复制转换实现，现给通用
SVE conversion/conversion-memory 内部 lambda 补齐 `always_inline`。最终
`int8->fp32` 1024 平方 SME leaf 只有入口 `smstart za` 和出口 `smstop za`，
中间直接生成 `ld1b + scvtf + ZA move + st1w`。

固定 CPU 500、三次重复的 Const 中位数如下。SVE2 是最终通用 conversion
代码的 VLA 构建，Native 是最终自动选择结果；Const 16 平方自动保留 vector。

| 路径/代表 pair | shape | SVE2-VLA | Native 自动 | 加速 |
|---|---:|---:|---:|---:|
| load-convert + transpose + store (`int8->fp32`) | 16x16 | 0.099 µs | 0.089 µs (vector) | 1.11x |
| 同上 | 63x129 | 3.76 µs | 1.89 µs (SME) | 1.99x |
| 同上 | 256x256 | 70.74 µs | 12.49 µs (SME) | 5.66x |
| 同上 | 1024x1024 | 1201.68 µs | 300.96 µs (SME) | 3.99x |
| load + transpose + store-convert (`fp32->fp16`) | 16x16 | 0.072 µs | 0.071 µs (vector) | 1.01x |
| 同上 | 63x129 | 3.20 µs | 1.87 µs (SME) | 1.71x |
| 同上 | 256x256 | 69.54 µs | 13.13 µs (SME) | 5.30x |
| 同上 | 1024x1024 | 2234.30 µs | 593.50 µs (SME) | 3.76x |
| 双端转换 (`int8->fp16`, compute fp32) | 16x16 | 0.104 µs | 0.102 µs (vector) | 1.02x |
| 同上 | 63x129 | 4.16 µs | 2.16 µs (SME) | 1.92x |
| 同上 | 256x256 | 70.04 µs | 14.88 µs (SME) | 4.71x |
| 同上 | 1024x1024 | 1237.78 µs | 408.06 µs (SME) | 3.03x |

256 平方的全部 12 组转换均提升 1.37--5.97x；1024 平方全部提升
2.30--3.99x。63x129 中 fp64 source 的三组 SME 候选慢 18--26%，所以 Const 且
元素数不超过 8192 时保留 vector；所有 Const 16 平方转换也保留 vector。
Dynamic/Any 无法在构造 operation 时知道实际元素数，继续保持 SME eligibility，
从而不牺牲大动态矩阵的 2--6x 收益。16 平方 Dynamic/Any 的额外绝对开销在
本机约 8--76 ns。

SME ZA slice 及 streaming/ZA 状态属性以 Arm ACLE 为准：
<https://arm-software.github.io/acle/main/acle.html>。

### SME 选择性成对软件流水

原循环每次执行一条 load/convert 后立即写一行 ZA，读侧也逐列执行
`ZA read -> convert/store`，机器码中的依赖链几乎没有可供乱序窗口利用的
独立操作。最终微内核把两个相邻行/列作为一组：

- 写侧先产生 `value0/value1`，再依次写 ZA；
- 读侧先读取 `raw0/raw1`，再依次 store/store-convert；
- 奇数尾部仍执行原单行路径。

第一版跨迭代携带向量的传统 software pipeline 会额外生成每轮 `mov z`，168
个代表 case 的几何平均只有基线的 92.5%，最差回归 25%，已撤回。第二版
成对展开消除了跨迭代 move，但全类型启用会增加 conversion 的 live vector
压力；fp64 Compute 的窄化最差回归 19%，部分长矩形也有明显回归，因此同样
没有全局启用。

最终只对七轮正反序 A/B 全部不回归的 Const 实例启用：

1. 同类型 int8/fp32，元素数至少 1024 平方；
2. 同类型 fp64、方阵且元素数至少 1024 平方；
3. fp32 到 2-byte destination 的 store-convert，元素数至少 8000。

Dynamic/Any、load-convert、双端转换、同类型 16-bit、fp64 conversion 和未入选
几何形状都在 `if constexpr` 中落回原循环。抽查这些 leaf 的 symbol size 与
基线完全一致；例如 Dynamic fp32、Const fp16、int8->fp32 和 fp64->fp32 均
保持原大小，避免用平均收益掩盖单 case 回归。

固定 CPU 500、七轮正反序中位数的保留实例结果如下：

| 路径 | shape | 加速 |
|---|---:|---:|
| int8 同类型 | 1024x1024 / 4096x4096 | 1.012x / 1.031x |
| int8 同类型 | 8192x128 / 4096x256 | 1.093x / 1.118x |
| fp32 同类型 | 1024x1024 / 4096x4096 | 1.189x / 1.138x |
| fp32 同类型 | 8192x128 / 4096x256 | 1.532x / 1.091x |
| fp64 同类型 | 1024x1024 / 4096x4096 | 1.033x / 1.003x |
| fp32->fp16 | 63x129 / 256x256 / 1024x1024 | 1.042x / 1.036x / 1.037x |

13 个启用实例的几何平均为 1.097x，最小为 1.003x，没有负收益实例。最终
机器码在 fp32 写侧形成两个独立 `ld1w` 后的两次 horizontal ZA move，读侧
形成两次 vertical ZA move 后的两次 store；不再出现第一版的循环携带 move。

其他候选暂不进入默认路径：

- ACLE 规定 `__arm_new("za")` 的新 ZA 必须零初始化；删除 `zero {za}` 需要把
  ZA 所有权提升到 execution region，并用 `__arm_out/__arm_inout` 共享状态。
  当前资源模型只有 Streaming、没有 ZA lifetime resource，单改 leaf 会改变
  ABI 状态契约。
- 非临时 store 对大冷输出可能有价值，但当前 SVE store-convert lowering 会
  消费 temporality option 后仍使用 temporal store；需要先补齐通用后端并建立
  cold-output benchmark，不能直接在 Transpose 中绕过通用 vec。
- 软件预取会为每个 ZA tile 增加多条 `prfm`，而当前连续 source row 已形成
  多路顺序流；没有跨全部长宽比的无回归证据前不启用。
- 多 ZA tile 双缓冲只适用于 ZA16/32/64，且会增加代码副本和 ZA 端口竞争；
  应先做独立 microbenchmark。更低风险的下一候选是拆分 full/tail tile loop，
  让完整 tile 消除每 tile 的 active 比较、`whilelt` 和 full/masked 分支。

## 4. 已验证的优化

### 保留：同类型 raw bottom edge 使用 masked fixed network

旧实现把所有非完整 tile 都转到逐列 gather。对于 `63x129` 和 16x16 tile，
底部 `15x128` 区域实际上拥有 8 个完整列块，只有行数不足；逐列 gather
浪费了绝大多数连续访问机会。

新路径只在以下条件同时成立时启用：

1. 行方向 masked、列方向 unmasked；
2. 输入、输出均为无 transform 的 raw direct access；
3. 内存元素类型等于 compute 类型。

因此跨类型、右边尾块和角落尾块保持原 gather 语义，模板实例数量也受到
控制。

在固定 CPU 0 的 x86 上，`63x129` 中位数如下（单位 µs）：

| dtype | metadata | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|---:|
| fp16 | Const | 6.40 | 1.36 | -78.8% |
| fp16 | Dynamic/Any | 14.6 | 1.54 | -89.5% |
| bf16 | Const | 6.30 | 1.34 | -78.7% |
| bf16 | Dynamic/Any | 14.6 | 1.52 | -89.6% |

优化后同一 shape 的 int8 为 1.26/1.52 µs（Const/Dynamic），fp32 为
3.91/3.95 µs，fp64 为 5.89/5.93 µs。`4096x4096` 没有尾块，生成路径未变；
实测差异作为频率和系统噪声处理，不把它计为本优化收益。

### 撤回：无条件扩大 x86 列 tile

尝试把窄类型从 16x16 扩为 16x32。4096 平方的 int8 基本持平，fp16 约快
1.7%，但 bf16 约慢 2.4%，且 fp16 `63x129` Const 从约 6.4 µs 退化到
8.0 µs。收益不稳定且尾块回归明显，因此没有保留。

### 撤回：四种 mask case 全部实例化 fixed kernel

该方案会让 GCC 单翻译单元编译超过 21 分钟并占用约 9.3 GiB，模板成本
不可接受。最终方案只增加 bottom-edge 一种、且只限 raw 同类型访问。

## 5. 验证结果

- x86：`TransposeTest-Native` 6/6 通过；399 个 benchmark case 全量执行，
  0 错误。
- ARM `920f-4`：源码位于 `~/vecops-neo-ws2`，VLA/SME 构建目录为
  `~/vecops-neo-ws2/cmake-build-release-transpose`；使用毕昇 Clang 显式配置
  编译器。Native/SME 与 SVE2-VLA 均为 399 个逻辑 case、1596 条 JSON
  aggregate 记录（mean/median/stddev/cv）、0 错误。SVE512 精简对比目标为
  60 个 case、120 条聚合结果、0 错误。
- `TransposeTest-NativeFixedSVE` 9/9 通过，新增跨类型 SME prepared operator，
  并覆盖 fixed-SVE 与 SME 同时启用时的 raw/sized vector 和 predicate bridge。
- 通用 SVE conversion 内联改动另由 fixed-SVE 的 `VecConversionTest` 39/39、
  `VecConversionMemoryTest` 63/63 通过。
- ARM 抽样中，4096 平方 int8/fp32/fp64 的中位数约为 2.15/6.27/11.03 ms；
  63x129 int8/fp32/fp64 约为 0.438/1.79/3.50 µs。同类型均标记为 SME。

## 6. 后续优化计划

按收益、风险和可验证性排序：

1. **x86 右边窄尾块微内核**：仅为 active columns 为 1/2/4/8 的常见值生成
   小型 packing/transpose 内核，替代右边区域逐列 gather；角落先保持 gather。
   目标是改善 `3x5`、`17x19`、`63x129`，同时把新增模板内存限制在每个
   dtype 翻译单元 1 GiB 内。
2. **ARM SME ZA 生命周期**：把 ZA owning 放到 execution region，只让叶内核
   使用共享/`inout` ZA；前提是证明所有被读取 slice 在本次调用内都被完整
   predicated 覆盖。目标是消除小矩阵每次调用的 `zero {za}`，大矩阵不应
   回归。该项需要同时审计 ACLE 属性和异常展开边界。
3. **大矩阵 cache blocking**：在 fixed/SME tile 外增加按元素宽度选择的
   L1/L2 macro tile，候选从 64x64、128x64、128x128 开始。重点看
   1024/4096 平方和长矩形，使用 LLC miss、DTLB miss、写回带宽共同选型。
4. **SVE-VLS hybrid dispatch**：小矩阵、尾块或 `m*n` 低于实测阈值时使用
   VLA-style gather，大矩阵才进入预编译的 VLS fixed kernel；同一 dtype/VL
   只生成少量运行时 shape 内核，避免每个 Const shape 重新内联 network。
5. **数据驱动 tile 表**：分别为 GCC/Clang、AVX2/AVX-512 建立离线候选表，
   不再全局扩大 tile。只有在全部 metadata 模式的几何平均提升超过 3%、
   且单 case 回归不超过 2% 时才进入默认配置。
6. **构建成本门槛**：conversion 按输入宽度的拆分已经完成；后续在 CI 记录
   编译耗时和峰值 RSS，禁止单进程重新超过 8 GiB。

每个阶段都必须运行全部 399 case 的内置逐元素校验、x86 单测、ARM 全量
benchmark sanity，并分别保存 Const、Dynamic、Any 的 A/B 中位数。性能测试
固定 CPU、记录频率与负载；小于 2% 的差异默认视为噪声。
