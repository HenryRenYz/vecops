# Matmul Dynamic/Const 性能与剪枝报告

日期：2026-08-30

## 测量方法

同一个 catalog shape 自动注册 Dynamic 与 Const，两者共享数值、stride、packing、
pipeline 和 reference oracle。正式性能运行串行执行，固定本地 x86 CPU 40、920f-4
CPU 100，`min_time=20ms`、3 次重复；表中的 CV 为 Google Benchmark aggregate CV，
比例为 `Const median / Dynamic median`。比例只报告，不作为正确性合入门禁。

原始数据：

- x86：`cmake-build-matmul-unified-gcc-release/matmul-validation/x86-core-cpu40.json`
- ARM Native：`/home/ryz/vecops-neo/cmake-build-matmul-unified-clang-release/matmul-validation/MatmulBench-Native.cpu100.json`
- ARM Fixed：`/home/ryz/vecops-neo/cmake-build-matmul-unified-clang-release/matmul-validation/MatmulBench-NativeFixedStreamingSVE.cpu100.json`

## x86 AMX（CPU 40）

| case | Dynamic us | Const us | Dynamic/Const CV % | Const/Dynamic |
|---|---:|---:|---:|---:|
| packedAB microkernel 16x16x128 | 0.105618 | 0.102110 | 0.1242 / 0.0329 | 0.966793 |
| packedA tail 35x53x257 | 1.840323 | 1.782141 | 0.0304 / 0.1052 | 0.968385 |
| packedAB tail 35x53x257 | 0.900948 | 0.891392 | 0.0098 / 0.0371 | 0.989393 |
| packedB tail 35x53x257 | 1.375748 | 1.330362 | 0.0543 / 0.0208 | 0.967010 |
| raw/auto-pack tail 35x53x257 | 2.078164 | 2.028803 | 0.0202 / 0.0885 | 0.976248 |

五个代表项 Const 均更快，改善约 1.1%--3.3%。

## ARM SME Native（CPU 100）

| case | Dynamic us | Const us | Dynamic/Const CV % | Const/Dynamic |
|---|---:|---:|---:|---:|
| packedAB microkernel 16x16x8 | 0.129027 | 0.121462 | 0.0191 / 0.1475 | 0.941367 |
| packedA tail 35x53x17 | 0.990190 | 0.923550 | 0.7168 / 1.2604 | 0.932700 |
| packedAB tail 35x53x17 | 0.565809 | 0.927160 | 0.4199 / 0.3225 | 1.638644 |
| packedB tail 35x53x17 | 0.917669 | 0.883789 | 2.0475 / 0.8348 | 0.963080 |
| raw/auto-pack tail 35x53x17 | 1.236055 | 1.108003 | 2.9142 / 0.4596 | 0.896403 |

Native 的 packedAB tail 出现 63.9% 回退，其余四项改善约 3.7%--10.4%。该异常按计划仅
记录，不阻断合入；FixedStreamingSVE 的同一 shape 没有复现。

## ARM SME NativeFixedStreamingSVE（CPU 100，SVL=512）

| case | Dynamic us | Const us | Dynamic/Const CV % | Const/Dynamic |
|---|---:|---:|---:|---:|
| packedAB microkernel 16x16x8 | 0.120050 | 0.098270 | 0.0193 / 0.0003 | 0.818577 |
| packedA tail 35x53x17 | 0.864513 | 0.682784 | 0.5456 / 1.2287 | 0.789790 |
| packedAB tail 35x53x17 | 0.511796 | 0.412582 | 0.3011 / 0.1915 | 0.806146 |
| packedB tail 35x53x17 | 0.729155 | 0.564986 | 1.9227 / 1.2800 | 0.774850 |
| raw/auto-pack tail 35x53x17 | 1.113743 | 0.879940 | 1.9308 / 1.2094 | 0.790074 |

五个代表项均改善约 18.1%--22.5%，说明固定 streaming SVL 与 shape 常量能共同消除
运行时区域选择和边界计算。

## 反汇编剪枝

在 920f-4 的 `MatmulBench-NativeFixedStreamingSVE` / `MatmulFusionBench-...` 上用
BiSheng `llvm-objdump` 对对应模板符号做静态计数：

| 代表项 | Dynamic bytes/instructions/条件分支 | Const bytes/instructions/条件分支 | 剪除 |
|---|---:|---:|---:|
| packedAB microkernel 16x16x8 | 1368 / 342 / 23 | 1160 / 290 / 21 | 208 B、52 指令、2 分支 |
| raw auto-pack 64x256x256 | 1156 / 289 / 23 | 1048 / 262 / 21 | 108 B、27 指令、2 分支 |
| Fusion F16 64x256x256 | 9228 / 2307 / 143 | 8808 / 2202 / 136 | 420 B、105 指令、7 分支 |

Const specialization 明确缩短了 wrapper/operation 的运行时 extent 检查、边界计算、
tile/packing 区域选择和 Fusion pipeline 分支；MOPA/call 门禁同时通过，证明剪枝没有把
微内核替换为回退调用。反汇编文本保存在 920f-4 的 `~/tmp/*.extent-disasm.txt`。

可用下列命令重新生成比例报告：

```bash
python3 benchmarks/compare_matmul_extents.py RESULT.json
```

