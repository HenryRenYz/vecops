# CPU SIMD Exp 实现调研与 Vec.h 设计建议

> 文件名沿用现有任务名。本报告实际主题是 CPU SIMD `exp/expf`，重点面向
> x86 AVX2/AVX-512、Arm SVE 和 `Vec.h` 抽象层。

## 1. 结论摘要

1. **需要一个可移植、严格语义的基线时，Highway `Exp` 是三组现有代码中最可靠的参考。**
   它采用最近整数范围缩减、Cody-Waite 两段 `ln(2)`、Estrin 多项式和拆分指数重构。
   Highway 声明 FP32/FP64 最大误差为 1 ULP；本次 FP32 测试在约 600 万个重点样本上也只观察到 1 ULP。

2. **Highway `FastExp` 不是“接近 1 ULP”的函数。**
   它的四次多项式本身最大相对误差为
   `3.7044659365801727e-6`，本次端到端 FP32 最大观察误差为
   `3.900146388854e-6`、48 ULP。用于激活或容忍约 `4e-6`
   相对误差的 Softmax 通常没有问题，但不应把它标成准确 `expf`。

3. **古老 `exp.h` 的默认五次多项式并不等价于 1 ULP 实现。**
   仅多项式的实数最大相对误差是 `2.3053725469423e-7`，但
   `exp(x)` 先做一次 `x * log2(e)`，范围缩减误差随 `|x|` 放大。
   本次在 `[-87, 88]` 上测得最大 `4.008126957667e-6`、66 ULP；
   六次多项式仍有 `3.862591859790e-6`、64 ULP，说明继续加次数已经基本无效。
   此实现还会把所有很小输入钳到约 `1.17631e-38`，把正溢出钳到有限数，
   并且 scalar FP64 路径的位构造参数写死为 FP32，不能直接复用。

4. **KuTACC 当前启用的 FEXPA 路径非常快，但它只是 1/64 个 octave 的台阶近似。**
   实测最大相对误差约 `0.5435%`，约 9 万 FP32 ULP；它还会让 NaN 变成 0、
   提前下溢，并在 `x=88.7174225` 时提前返回 `+Inf`。
   这不适合作为通用 `expf`，也不建议作为默认深度学习 Exp。

5. **SVE 上最值得采用的是“FEXPA 查表 + 小残差二次多项式”，而不是 KuTACC 当前的裸 FEXPA。**
   按 Arm 官方算法实现后，本次在 920f-hpc 上：

   | SVE FP32 实现 | `[-20,0]` 最大观察误差 | `[-87,88]` 最大观察误差 | 1M 数组 ns/元素 |
      |---|---:|---:|---:|
   | KuTACC 裸 FEXPA | `5.4328538e-3`, 90651 ULP | `5.4350085e-3`, 90689 ULP | 0.3032 |
   | FEXPA + 二次残差 | `1.1754584e-7`, 1 ULP | `1.1809026e-7`, 1 ULP | 0.3341 |

   只多约 10% 时间，就把误差从约 `0.54%` 降到本次测试的 1 ULP。
   这是 Arm SVE 后端最接近“硬件性能极限且精度可靠”的方案。

6. **推荐最终提供两个明确契约，而不是一个含糊的 `fast_exp`：**

    - `exp`: 严格特殊值、渐进下溢、目标 FP32 全域不超过 1 ULP。
      可先移植 Highway `Exp`。
    - `exp_fast` 或 Softmax 内部专用 `exp_minus_or_zero`: 输入已知 `x <= 0`，
      可把小于 `log(FLT_MIN)` 的结果刷成 0。通用 x86 使用 Cody-Waite +
      Estrin 五次多项式；SVE 使用 FEXPA + 二次残差。

## 2. 调研对象与版本

| 对象 | 版本 | 关键代码 |
|---|---|---|
| Highway 准确数学 | `9973fb4a12c0b4876ad0304d226373a99b853c83` | [math-inl.h](/home/renyz/project/highway/hwy/contrib/math/math-inl.h:809) |
| Highway 快速数学 | 同上 | [fast_math-inl.h](/home/renyz/project/highway/hwy/contrib/math/fast_math-inl.h:1151) |
| 古老 vecops | `eaf5ffeb30ed5a6a54fd4a4f1df7e2068894f447` | [exp.h](/home/renyz/project/af3-torchneo/csrc/alphafold3_native/ext/vecops/src/_C/vecops/exp.h:44) |
| KuTACC | `5c2af3434893734676828f9a8c1f53968e66eb65` | [AtomGit fast_exp.h](https://atomgit.com/kunpengcompute/kutacc/blob/main/src/math/fast_exp.h) |
| Arm FEXPA 参考 | 2026-06-17 页面版本 | [Arm FEXPA Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/fexpa/fexpa/) |

Highway 为 Apache-2.0；KuTACC 文件声明木兰宽松许可证第 2 版。若复制代码或系数，
应保留并核对相应许可证和版权要求。

## 3. 评价口径

### 3.1 数值指标

- **ULP 误差**：结果与正确舍入 FP32 参考值之间的 bit 距离。
- **相对误差**：`abs(y - exp(x)) / exp(x)`。接近下溢时相对误差会被量化噪声放大。
- **多项式误差**：假定实数精确运算，只评价缩减区间上的多项式。
- **端到端误差**：包括输入乘法、取整、范围缩减、FMA 舍入、指数构造和最终乘法。

不能用注释里的 MSE 代替最大误差。MSE 很小并不排除区间端点有更大的最坏误差。

### 3.2 测量方法

多项式最坏误差使用 Python `mpmath`、100 位十进制精度计算。对
`E(x)=P(x)/b^x-1`，求解 `P'(x)-ln(b)P(x)=0` 的全部实根，并连同区间端点
一起比较。因此第 6.1 节的“实数多项式误差”不是均匀采样估计。

FP32 端到端测试使用实际 Highway SIMD 代码或逐操作 FP32/FMA 的等价实现：

- Highway/古老实现每个主区间约 600 万点：
  500 万均匀点、100 万固定种子随机点，以及每个范围缩减边界附近的相邻浮点数。
- FEXPA 每个区间约 22 万或 42 万点：
  覆盖每个 1/64 表格 cell 的中心和舍入边界两侧若干相邻 FP32，再加 20 万随机点。
- 参考值为对精确 FP32 输入计算的长双精度 `exp`，再舍入到 FP32。

这些端到端结果是很强的边界定向测试，但不是对全部 `2^32` 个 FP32 bit pattern
的形式化穷举。文中严格区分“上游声明”和“本次最大观察值”。

### 3.3 深度学习区间

重点给出两个区间：

- `[-20, 0]`：代表减去最大值后的常见 Softmax 有效区间。
- `[-87, 88]`：FP32 normal 输出的主要通用区间。

Softmax 中输入先减最大值，所以指数输入必定 `<= 0`，至少有一个输入精确为 0。
因此 `exp(0)=1`、单调性、正值性以及 NaN 是否传播都很重要。

## 4. 各实现的算法分析

### 4.1 Highway `Exp`: 准确基线

FP32 路径可概括为：

1. `q = round(x / ln(2))`，通过正负 `0.5` 加偏置后向零转换实现。
2. 用两段常数缩减：

   ```text
   r = fma(q, -0.693145751953125, x)
   r = fma(q, -1.428606765330187045e-6, r)
   ```

   得到 `r in [-ln(2)/2, ln(2)/2]`。第二个 FMA 避免 `q * ln(2)`
   在大 `|q|` 时丢失低位。

3. 计算

   ```text
   exp(r) ~= 1 + r + r^2 * P5(r)
   ```

   总次数为 7，`P5` 用 Estrin 树求值，不是五级串行 Horner。

4. 构造两个 `2^(q/2)` 因子后相乘。拆成两半使 FP32 subnormal 区间也能正确生成，
   避免直接构造非法指数 bit pattern。
5. `x < -104` 返回 0；上溢由最终浮点乘法自然得到 `+Inf`。

**优点**

- Highway 声明最大 1 ULP；FP32 本次也观察到最大 1 ULP。
- Cody-Waite 缩减消除了古老实现中随 `|x|` 增长的主要误差源。
- Estrin 缩短多项式依赖链，适合双 FMA 管线的 x86 和多发射 Arm 核。
- `-Inf/+Inf/NaN/-0` 本次均得到符合 `expf` 的分类和结果。
- 支持 FP32 和 FP64，并处理 subnormal。

**代价和注意点**

- FP32 AVX-512 编译结果每向量约 31 条核心向量指令，寄存器压力较高。
- 两次整数/浮点转换、两段缩减和拆分重构位于关键路径。
- FP64 多项式有 11 个系数，成本明显高于 FP32。
- 当前 Highway 测试虽写了 1 ULP 门限，但只按 bit 范围抽样约 1000 点/段，
  不是穷举。此 commit 的 FP64 `Exp` 测试上界还写成了 `+104`，
  小于头文件声明的 `+706`，属于覆盖缺口。

### 4.2 Highway `FastExp`

范围缩减仍是 `q = round(x/ln(2))`，但只用一个 FP 常数：

```text
r = fma(q, -0.69314718056f, x)
```

缩减后的 `exp(r)` 使用 Caratheodory-Fejer 四次多项式和 Estrin：

```text
x2    = r*r
term0 = fma(c1, r, c0)
term1 = fma(c3, r, c2)
term2 = fma(c4, x2, term1)
poly  = fma(term2, x2, term0)
```

有三个值得区分的入口：

| 入口 | subnormal | 边界工作 | 适用场景 |
|---|---|---|---|
| `FastExp<true>` | 保留 | 拆分指数 + 下界 mask | 通用快速函数 |
| `FastExp<false>` | 刷零 | 下界 clamp，直接 `2^q` | 已允许 FTZ 的 normal-only 负载 |
| `FastExpMinusOrZero` | 刷零 | 假设 `x<=0`，固定 `-0.5` | Softmax/Sigmoid 内部 |

所有路径都没有按 lane 的控制流分支，只有比较、mask 或 predication。
`FastExpMinusOrZero` 省掉符号提取和最终 mask，是三者中最快的。

**精度**

- 源码注释声称 normal 区间最大相对误差 `0.0007%`，测试实际门限为 `8e-6`。
- 本次 FP32 最大观察值为 `3.9001464e-6`、48 ULP。
- `c0=1.000000151...` 舍入后使 `FastExp(0)=0x3f800001`，
  即 1 ULP 高于 1。它是有意识的 minimax 误差分配，不满足 `exp(0)` 精确锚点。
- `FastExp<false>` 必须保持其 normal-only 前置条件。本次给它输入 `104`
  得到了负数 bit pattern，说明不能把它当通用入口。

### 4.3 古老 `exp.h`

该实现先把自然指数改写为 `exp2`：

```text
t  = clamp(x * log2(e), range_min, range_max)
n  = floor(t)
f  = t - n                 # f in [0, 1)
p  = Horner(P_N(f))
2n = bitcast((n + bias) << mantissa_bits)
y  = 2n * p
```

`Nprec=1..6` 是多项式次数，不是 Newton 迭代。每增加一次会增加一个串行 FMA。

**性能特点**

- 默认五次路径的 AVX-512 主循环约 15 条核心向量指令，少于 Highway `Exp`。
- 五个 FMA 是 Horner 串行链，最低关键路径至少包含五个连续 FMA latency；
  指令少不代表延迟最低。实测它反而慢于 Estrin 的 Highway `FastExp<false>`。
- 循环有固定模板上界并有 unroll pragma，优化后没有数据相关分支。

**主要精度问题**

1. 系数优化目标是积分平方误差，不是最大误差。
2. `x * log2(e)` 先舍入，再从它取整数和小数。即使 `P6` 很准，
   原始 `x` 的低位已经丢失，误差随 `|x|` 放大。
3. 由此出现明显的收益饱和：在 `[-87,88]` 上，五次为 66 ULP，
   六次仍为 64 ULP。
4. 奇数次数 1、3、5 在每个 `exp2` 整数边界存在向下跳变。
   对默认五次，精确实数多项式的
   `2*P(0)-P(1)=-2.179165439458487e-7`，因此不具备严格单调性保证。

**正确性问题**

- FP32 下界钳到 `-125.999`，所以任何更小输入都返回约
  `1.17631e-38`，而不是 subnormal 或 0；`exp(-Inf)` 也不是 0。
- 上界钳到 `127.999`，`exp(+Inf)`、`exp(89)` 等返回约
  `3.40047e38` 的有限数，而不是 `+Inf`。
- 默认五次 `exp(0)` 为 `0x3f7ffffc`，比 1 低 4 ULP。
- scalar 模板的指数构造写死为 `ExpBias<fp32>` 和
  `MantissaBits<fp32>`，所以 scalar FP64 路径是错误的。
- scalar `reinterpret_cast<const TFp&>(xi_i)` 违反严格别名规则，
  应使用 `bit_cast`。
- FP16、BF16 和 FP32 的下界都只保留 normal，不保留 subnormal。
  FP16/BF16 原生多项式计算的舍入也会很快压过高阶系数收益。

因此，这份实现可以贡献“次数可调”和系数研究，但不应原样迁移。

### 4.4 KuTACC 当前硬件路径

文件无条件定义了 `USE_HARDWARE_EXP`，实际编译的 FP32 路径是：

```text
u = uint32(round(min(x, ln(FLT_MAX)) * (64/ln2) + 127*64))
y = FEXPA(u)
```

920f-hpc 上生成的主计算恰好是 5 条 SVE 指令：

```text
fmin -> fmad -> frinta -> fcvtzu -> fexpa
```

`FEXPA` 使用输入低 6 bit 选择 `2^(j/64)` 表项，并把其余 bit 形成指数。
当前代码没有计算表格点与真实输入之间的残差，所以输出在 log 域每
`ln(2)/64` 更新一次。

仅由半个表格间隔即可得到理想量化误差：

```text
exp(ln(2)/128) - 1 = 0.0054299011128028213513839559348
```

实际常数舍入和 FEXPA 表项使本次最大观察值略增到约 `0.005435`。

**正确性实测**

| 输入/区域 | KuTACC 当前结果 | 正确结果 |
|---|---:|---:|
| NaN | 0 | NaN |
| `-Inf` | 0 | 0 |
| `-100` | 0 | `3.78e-44` |
| `-88.0242844` 附近 | 已为 0 | 仍为非零 subnormal |
| `88.7174225` | `+Inf` | `3.38444e38` 有限 |
| `+Inf` | `+Inf` | `+Inf` |

对于 Softmax，`0.54%` 的单项相对误差未必导致 `0.54%` 的最终概率误差，
但不同 logit 落在不同台阶方向，会破坏平滑性，并影响小差值、梯度和可重复性。
在几乎不损失性能的替代方案已经存在时，没有理由把它设为默认。

### 4.5 KuTACC 未启用的软件路径

`#else` 中有 FP32/FP16 五次多项式实现。其 FP32 多项式在
`[-ln(2)/2, ln(2)/2]` 上的精确实数最大相对误差只有
`9.211870692088485e-8`；在 `[-20,0]` 上，本次实际 SVE 最大观察误差
`2.0312135e-7`、2 ULP，精度很好。

但该代码的指数重构先构造 `2^(n-1)` 再乘 2。当 `n=-126` 时，
构造出来的指数域为 0，bitcast 是 0 而不是 `2^-127`。因此
`x=-86.9954147` 附近会直接返回 0，而正确值约 `1.65e-38`。
它在 normal 下界形成一个整段零洞，导致本次 `[-87,88]` 最大相对误差为 100%。

另外，由于 `USE_HARDWARE_EXP` 在头文件内无条件定义，FP16 overload 当前根本不会编译。

## 5. FEXPA 的正确用法

Arm 官方方案把范围缩减细化到 1/64：

```text
k     = round(x / ln2 * 64) / 64
r     = x - k*ln2_hi - k*ln2_lo
scale = FEXPA(encoded(k))
poly  = r * (c0 + c1*r)
y     = scale + scale*poly
```

此时 `r in [-ln(2)/128, ln(2)/128]`，区间宽度仅约 `0.01083`，
二次多项式已经足够覆盖 FP32 精度。

在 920f-hpc 上，主循环除 load/store/loop 外生成 11 条动态 SVE 指令，
其中 3 条是 `movprfx`，语义计算为：

```text
fmla, fsub, fexpa, fmls, fmls, fmla, fmul, fmad
```

`FEXPA(scale)` 与残差多项式存在较多并行机会，所以虽然比裸 FEXPA 多了
两段范围缩减和二次多项式，端到端数组吞吐只慢约 10%。

本次原型还没有加完整特殊值和 subnormal 处理。产品实现应：

- 对通用 `exp` 显式处理 NaN、`+Inf`、`-Inf`、上溢和下溢。
- normal-only Softmax 路径可把 `x < log(FLT_MIN)` 刷成 0。
- 严格路径若要保留 subnormal，需要像 Highway 一样拆分 scale，或提供单独慢边界修正。
- 不要用 `min(x, upper)` 处理 NaN，因为不同 ISA 的 min/max NaN 语义可能不同；
  应先保留 NaN mask，最后恢复 NaN。

## 6. 精度结果

### 6.1 精确实数多项式误差

下表不含任何 FP 舍入或范围缩减误差：

| 多项式 | 缩减区间 | 最大相对误差 | 最坏点 |
|---|---|---:|---:|
| 古老 N=1 | `[0,1]`, 逼近 `2^x` | `5.334644107806047e-2` | 0 |
| 古老 N=2 | 同上 | `3.762902276381367e-3` | 0 |
| 古老 N=3 | 同上 | `1.880920710445534e-4` | 0 |
| 古老 N=4 | 同上 | `7.2868379401323e-6` | 0 |
| 古老 N=5 | 同上 | `2.3053725469423e-7` | 0 |
| 古老 N=6 | 同上 | `6.1646238251e-9` | 0 |
| Highway `Exp` FP32 | `[-ln2/2,ln2/2]` | `1.1302394270962321e-9` | `-ln2/2` |
| Highway `FastExp` | 同上 | `3.7044659365801727e-6` | `-ln2/2` |
| KuTACC 软件五次 | 同上 | `9.211870692088485e-8` | `+ln2/2` |

古老实现从 N=5 到 N=6 的多项式误差改善约 37 倍，但端到端只从
`4.008e-6` 改到 `3.863e-6`，直接证明瓶颈已转移到范围缩减。

### 6.2 FP32 端到端最大观察误差

| 实现 | `[-20,0]` 相对误差 / ULP | `[-87,88]` 相对误差 / ULP |
|---|---:|---:|
| Highway `Exp` | `8.6169e-8` / 1 | `8.3771e-8` / 1 |
| Highway `FastExp` | `3.7249e-6` / 45 | `3.9001e-6` / 48 |
| 古老 N=1 | `5.3347e-2` / 895009 | `5.3350e-2` / 895035 |
| 古老 N=2 | `3.7637e-3` / 34851 | `3.7665e-3` / 34904 |
| 古老 N=3 | `1.8856e-4` / 3160 | `1.9167e-4` / 3186 |
| 古老 N=4 | `8.1577e-6` / 73 | `1.0955e-5` / 128 |
| 古老 N=5 | `1.0549e-6` / 16 | `4.0081e-6` / 66 |
| 古老 N=6 | `9.6974e-7` / 16 | `3.8626e-6` / 64 |
| KuTACC 裸 FEXPA | `5.4329e-3` / 90651 | `5.4350e-3` / 90689 |
| KuTACC 软件五次 | `2.0312e-7` / 2 | 100% / 11863238，因下界零洞 |
| FEXPA + 二次残差原型 | `1.1755e-7` / 1 | `1.1809e-7` / 1 |

对于 subnormal，ULP 比相对误差更合理。Highway `Exp` 在
`[-104,-87]` 的重点样本仍最多 1 ULP；接近舍入为 0 的位置，相对误差自然可达 100%。

### 6.3 FP64 和低精度类型

- Highway `Exp<double>` 声明 1 ULP，使用更高次多项式和两段 `ln(2)`；
  `FastExp<double>` 仍是约 `8e-6` 相对误差等级，而不是 double ULP 等级。
- 古老 N=6 的实数误差 `6.16e-9` 对 FP32 很小，但相对 FP64 epsilon
  仍大约有 2780 万倍，不能称为 FP64 精确实现；scalar FP64 还有写死 FP32 位域的功能错误。
- FP16/BF16 输入建议提升到 FP32 做范围缩减、多项式和 Softmax 累加，最后再 demote。
  这既避免低精度原生 FMA 的系数量化，也让 x86 和 SVE 使用同一精度契约。

## 7. 性能、指令数和延迟

### 7.1 编译后主循环指令

计数来自本次 GCC/BiSheng Clang 的实际反汇编。常量均已移到循环外，
下表排除 load/store、地址更新和循环分支，`~` 表示寄存器 move/prefix
会随编译器分配略变。

| 实现 | ISA | 每向量核心动态指令 | 数据相关分支 | 主要关键路径 |
|---|---|---:|---|---|
| Highway `Exp` FP32 | AVX-512 | ~31 | 无 | 转换、2 FMA 缩减、Estrin、2 级 scale 乘法 |
| Highway `FastExp<true>` | AVX-512 | ~22 | 无 | 转换、1 FMA 缩减、Estrin、拆分 scale |
| Highway `FastExp<false>` | AVX-512 | ~16 | 无 | 转换、1 FMA 缩减、Estrin、1 次 scale |
| Highway `FastExpMinusOrZero` | AVX-512 | ~15 | 无 | 同上，省符号和最终 mask |
| 古老 N=5 | AVX-512 | 15 | 无 | 5 个串行 Horner FMA |
| KuTACC 裸 FEXPA | SVE | 5 | 无 | `fmad -> frinta -> fcvtzu -> fexpa` |
| KuTACC 软件五次 | SVE | ~22 | 无 | 5 个串行 FMA + 指数构造 |
| FEXPA + 二次残差 | SVE | 8 条语义计算，11 条含 `movprfx` | 无 | FEXPA 与残差链可部分并行 |

每种实现都只有数组循环和 tail 控制分支，不按 lane 跳转。x86 使用 mask，
SVE 使用 predicate，因此混合输入不会引发传统 branch misprediction。

### 7.2 延迟与吞吐判断

- **Horner 的问题是依赖深度。** 古老 N=5 和 KuTACC 软件路径的第
  `i+1` 个 FMA 必须等第 `i` 个完成，至少五层 FP 依赖。
- **Estrin 用更多寄存器换较短延迟。** Highway 四次 Fast 多项式有两个初始 FMA
  可并行；准确多项式也把偶/奇项拆树计算。
- **整数转换是不可忽略的瓶颈。** x86 的 FP32 到 INT32、再回 FP32
  位于范围缩减关键路径。魔数舍入可以减少显式转换，但依赖默认 round-to-nearest。
- **subnormal 支持有真实成本。** Highway Fast 的拆分指数与 mask 使它明显慢于
  normal-only 入口。Softmax 若接受 FTZ，应使用专门入口，而不是让编译器猜。
- **FEXPA 的优势不只是少一张软件表。** 它把指数位构造和 64 项查表合并成一条指令，
  并把残差缩小 64 倍，使二次多项式已经足够。
- **AVX-512 的实际吞吐还受频率影响。** ZMM 指令数更少不一定线性变快，
  应同时测量 AVX2 双向量版本；部分服务器上 AVX2 可能以更高频率运行。

### 7.3 实测吞吐

#### x86 本机

- CPU：Intel Xeon w9-3495X，AVX-512。
- GCC `-O3 -march=native`，单核绑定。
- 1024 个 FP32，数据驻留 L1，输入 `[-10,0]`，重复 100 万次。

| 实现 | ns/元素 | 相对 Highway `Exp` |
|---|---:|---:|
| Highway `Exp` | 0.2373 | 1.00x |
| Highway `FastExp<true>` | 0.1543 | 1.54x |
| Highway `FastExp<false>` | 0.1199 | 1.98x |
| Highway `FastExpMinusOrZero` | 0.1105 | 2.15x |
| 古老 N=5 的 AVX-512 等价实现 | 0.1393 | 1.70x |

古老 N=5 虽比 FastNormal 少/相近指令，却因五级 Horner 链慢于 FastNormal，
与依赖分析一致。

#### Arm 920f-hpc

- 共享主机运行前 load average 为 `0.00, 0.32, 1.86`，没有高负载任务。
- 64-byte SVE vector，2.0 GHz，BiSheng Clang 19.1.7。
- 1M FP32 数组，输入 `[-20,0]`，重复 200 或 500 次。

| 实现 | ns/元素 |
|---|---:|
| KuTACC 裸 FEXPA | 0.3032 |
| FEXPA + 二次残差原型 | 0.3341 |
| Highway `FastExpMinusOrZero` SVE | 0.4239 |
| Highway `FastExp<false>` SVE | 0.5186 |
| KuTACC 软件五次 | 0.6955 |
| Highway `FastExp<true>` SVE | 0.7030 |
| Highway `Exp` SVE | 0.8888 |

按固定 2.0 GHz 和 16 lane/VL 换算，这些是稳态吞吐而非单调用 latency：
裸 FEXPA 约 9.70 cycles/vector，FEXPA + 二次残差约 10.69 cycles/vector，
Highway Fast minus-or-zero 约 13.56 cycles/vector，Highway `Exp` 约
28.44 cycles/vector。真实单依赖链 latency 会更高，需用第 9.3 节的专门基准测量。

当前 Highway commit 将 Clang < 22 的可伸缩 SVE/SVE2 标为 broken。
为研究算法，本次显式覆盖了该 blocklist；因此这些 Highway SVE 数值是探索性结果，
不能当作 Highway 对 BiSheng 19 的正式支持声明。未覆盖时它默认生成 128-bit NEON。

## 8. 面向 Vec.h 的推荐设计

当前 [Vec.h](/home/renyz/project/vecops-neo/include/vecops/vec/Vec.h:1040)
已有 `fmadd`、min/max、mask blend、整数转换、bitcast 和 shift，足以实现
通用多项式路径；但尚缺统一的 floor/round-to-nearest API，也没有 SVE FEXPA 原语。

### 8.1 API 契约

建议至少区分：

```cpp
// 严格通用入口：NaN/Inf/subnormal 正确，FP32 目标 <= 1 ULP。
Vec<T> exp(T tag, Vec<T> x);

// 显式近似入口：误差契约写入文档，不默认为 strict exp。
Vec<T> exp_fast(T tag, Vec<T> x);

// 内部融合入口：前置条件 x <= 0，允许小结果 FTZ。
Vec<T> exp_minus_or_zero(T tag, Vec<T> x);
```

不要用一个模板次数参数把所有语义暴露给调用方。次数并不能单独决定误差，
因为范围缩减、边界和 scale 重构同样重要。

### 8.2 通用 x86/Vec 路径

严格路径第一版直接采用 Highway FP32 算法：

1. 最近整数 `q`。
2. 两段 Cody-Waite 缩减。
3. Highway 次数 7 的 Estrin 多项式。
4. 拆分 `q` 构造 scale，保留 subnormal。
5. 显式恢复 NaN/Inf 和 underflow/overflow。

Fast 路径建议重新拟合/验证一个 **常数项精确为 1 的五次 minimax 多项式**，
并使用 Estrin，而不是复用古老实现的 `exp2` Horner：

```text
r2  = r*r
p01 = fma(c1, r, c0)
p23 = fma(c3, r, c2)
p45 = fma(c5, r, c4)
p   = fma(r2, fma(r2, p45, p23), p01)
```

配合两段 Cody-Waite，可避免古老实现的 `|x|` 误差平台。若最终全域测试达不到
2 ULP，再调整系数，而不是盲目增加 Horner 次数。

Softmax 内部则固定 `x<=0`：

- 取整偏置固定为 `-0.5`。
- clamp 到 normal 下界，直接构造一个 `2^q`。
- 不生成 subnormal，不做最后 underflow mask。
- `x=0` 必须精确输出 1。

### 8.3 SVE 专用路径

在 `word::` 后端增加受 capability 保护的 FEXPA 原语，只让高层 Exp 实现调用。
FEXPA 属于 SVE，不需要 SVE2，但仍应以编译器 intrinsic 可用性为准。

FP32 normal 主路径采用第 5 节的 FEXPA + 二次残差。严格通用入口在其外层补：

- NaN/Inf 分类。
- `log(FLT_MAX)` 和下溢边界。
- subnormal 修正或拆分 scale。

不要把 KuTACC 当前的裸 FEXPA 暴露成 `exp_fast`。若确实有模型愿意接受
`0.55%`，可以作为名字明确的内部 `exp_ultrafast_0p55pct` 实验开关，
且默认关闭。

### 8.4 FP16/BF16

默认策略应为：

```text
load fp16/bf16 -> promote FP32 -> FP32 exp -> FP32 reduce/accumulate -> demote if needed
```

Softmax 的分母本来就应该用 FP32 累加。直接做 FP16 Exp 不仅动态范围小，
还会使多项式阶数和系数优化的收益被半精度舍入吞掉。

## 9. 正确性与验收计划

### 9.1 必测属性

1. `exp(+0)=exp(-0)=1`，bit-exact。
2. `exp(-Inf)=+0`，`exp(+Inf)=+Inf`，NaN 输入输出 NaN。
3. 所有有限输入结果非负；normal 有效区间结果严格为正。
4. 对相邻 FP32 输入检查单调不减，特别是每个 `q` 或 FEXPA table 边界。
5. 检查 `log(FLT_MAX)`、`log(FLT_MIN)`、`log(min_subnormal)`
   两侧的 `nextafter`。
6. 检查所有范围缩减半整数边界两侧至少 8 个 ULP。
7. 在不同 rounding mode 和 FTZ/DAZ 设置下验证契约；若只支持默认模式，应明确断言或文档化。

### 9.2 精度门限

建议门限：

| 入口 | 区间 | 门限 |
|---|---|---|
| `exp<float>` | 全有限 FP32 | <= 1 ULP，特殊值正确 |
| `exp_fast<float>` | normal 输出区间 | 暂定 <= 2 ULP；若选择 Highway Fast 系数则明确改为 <= `8e-6` 相对误差 |
| `exp_minus_or_zero<float>` | `[-87,0]` | <= 2 ULP，`exp(0)=1` |
| FTZ 区域 | `x < log(FLT_MIN)` | 用绝对误差和 Softmax 总遗漏质量评价 |

FEXPA + 二次残差已经在重点样本达到 1 ULP，但在写入正式契约前仍应做全 bit
穷举。FP32 总共约 42.9 亿 bit pattern，可在 920f-hpc 多核分片；由于它是共享主机，
执行这种重任务前应再次检查负载并协调资源。

### 9.3 性能基准

至少分开测：

- 单依赖链 latency。
- 2/4/8 条独立向量的稳态 throughput。
- L1 常驻数组、L2/L3 数组、真实 Softmax 融合 kernel。
- AVX2、AVX-512、SVE 不同 VL。
- normal-only 与完整 subnormal/special-value 路径。
- 编译后的动态指令、端口压力、寄存器 spill 和 AVX-512 降频。

只测单个 `exp` 函数调用不够。最终决策应以融合 Softmax/激活中的 cycles/element
和端到端模型偏差为准。

## 10. 远程验证记录

主机：`920f-hpc`  
远程临时目录：`~/tmp/vecops-exp-study`  
Highway 源码目录：`~/highway`  
编译器：`/home/HPC/HPC032/BiShengCompiler-5.2.0-aarch64-linux/bin/clang++`，
BiSheng Enterprise 5.2.0 / Clang 19.1.7。  
硬件：64-byte SVE/SVE2/SME，608 CPU，2.0 GHz。

核心命令如下：

```bash
source ~/.bashrc
cd ~/tmp/vecops-exp-study

clang++ -O3 -DNDEBUG -std=c++20 -march=armv9-a+sve2 \
  kutacc_probe.cc -o kutacc_probe
./kutacc_probe

clang++ -O3 -DNDEBUG -std=c++20 -march=armv9-a+sve2 \
  -DHWY_COMPILE_ONLY_STATIC \
  -DHWY_BROKEN_SVE=0 -DHWY_BROKEN_SVE2=0 -DHWY_BROKEN_SVE2_128=0 \
  -DHWY_BASELINE_TARGETS=HWY_SVE2 \
  -I$HOME/highway hwy_exp_bench.cc -o hwy_exp_bench_sve2
./hwy_exp_bench_sve2
```

结果摘要：

- KuTACC 裸 FEXPA：`0.3032 ns/element`，最大观察相对误差约 `0.5435%`。
- FEXPA + 二次残差：`0.3341 ns/element`，两个主区间均最大观察 1 ULP。
- Highway SVE：准确 `0.8888`、Fast normal `0.5186`、
  Fast minus-or-zero `0.4239 ns/element`。
- 远程工作均放在用户 home 下，没有使用共享 `/tmp`。

## 11. 最终建议排序

| 优先级 | 工作项 | 理由 |
|---:|---|---|
| P0 | 在 Vec.h 实现 Highway 风格 1 ULP FP32 基线 | 先建立正确性标杆和 fallback |
| P0 | SVE 实现 FEXPA + 二次残差，并补特殊值/边界 | 目前最佳精度/吞吐组合 |
| P0 | 为 Softmax 增加 `x<=0`、FTZ 的内部专用入口 | 可省掉通用边界成本 |
| P1 | x86 设计 Cody-Waite + Estrin 五次 fast 路径 | 目标接近旧实现成本但达到约 1-2 ULP |
| P1 | 增加 round/floor 语义和 SVE FEXPA capability | 让实现能干净落在 Vec 抽象层 |
| P1 | 全 FP32 bit pattern 精度/单调性验证 | 把“观察到 1 ULP”升级为工程保证 |
| P2 | FP64 严格路径和 FEXPA FP64 专门拟合 | 与 DL FP32 主目标分开推进 |
| 不建议 | 原样迁移古老 `exp.h` | 范围缩减平台、边界和 scalar FP64 错误 |
| 不建议 | 把 KuTACC 裸 FEXPA 作为默认 Exp | 0.54% 误差、NaN/边界错误，收益仅约 10% |
