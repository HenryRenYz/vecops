# Matmul 测试与 benchmark 统一覆盖报告

日期：2026-08-30

## 结论

Matmul 测试与 benchmark 已统一为架构中性目标。以重构前的未提交工作树二进制清单为
基线，按 Atom、shape、packing/batch、conversion、fusion pipeline 等语义字段归一化后，
新 catalog 的缺失 tuple 为 0；所有普通正向 shape 均有且仅有一份 Dynamic 和 Const
注册。零尺寸、death test、元数据和反汇编检查保留为专项覆盖，不纳入 pair 计数。

## 目标迁移

| 旧目标 | 新目标 |
|---|---|
| `MatmulAMXTest` / `MatmulSMETest` / `MatmulMixedPackingTest` | `MatmulTest` |
| `MatmulBatchAMXTest` / `MatmulBatchSMETest` | `MatmulBatchTest` |
| `MatmulFusionAMXTest` / `MatmulFusionSMETest` | `MatmulFusionTest` |
| `MatmulPackAMXTest` / `MatmulPackSMETest` | `MatmulPackTest` |
| `MatmulAMXBench` / `MatmulSMEBench` / Fixed Bench | `MatmulBench` |
| `MatmulPackAMXBench` / `MatmulPackSMEBench` | `MatmulPackBench` |
| `MatmulFusionScenarioBench` | `MatmulFusionBench` |
| `MatmulPackedScenarioBench` | `MatmulPackedBench` |
| `MatmulSMEFixedPolicyBench` | `MatmulPolicyBench` |
| `MatmulSMEMMLAExperimentBench` | `MatmulMMLAExperimentBench` |

`MatmulScenarioBench` 与 `MatmulWeightReuse*Bench` 继续保持独立。旧目标没有兼容别名。

源文件也已经架构中性化：Core/Pack benchmark 分别只保留 `MatmulBench.cpp` 与
`MatmulPackBench.cpp`；测试只保留 `MatmulTest.cpp`、`MatmulBatchTest.cpp`、
`MatmulFusionTest.cpp`、`MatmulPackTest.cpp`。旧 AMX/SME 命名源文件均已删除，GTest
suite 名也统一为目标名。AMX 权限初始化集中在 `MatmulTestArch.h`；Atom、能力宏与
SME 专项 case 作为同一源文件内的架构分支，不再由 CMake 选择两套测试主体。

## 覆盖对照

本地基线来自
`cmake-build-matmul-bench-gcc-release/benchmarks`。比较时删除 benchmark 目标前缀、
case 展示名、arch/计时字段和 extent/shape_meta；Fixed 类别映射回相同普通类别；
Packed catalog 中省略的 `input_mode` 视为 `raw`。允许新条目增加显式元数据（例如
`b_mode:shared`），但基线的全部语义字段必须仍被新条目包含。

| x86 catalog | 基线语义 tuple | 新语义 tuple | 基线缺失 | 新注册数 |
|---|---:|---:|---:|---:|
| Core（含旧 Fixed） | 165 | 165 | 0 | 330 |
| Pack | 110 | 110 | 0 | 220 |
| Scenario | 408 | 408 | 0 | 816 |
| Fusion | 326 | 326 | 0 | 652 |
| Packed | 375 | 375 | 0 | 750 |
| WeightReuse main | 394 | 394 | 0 | 788 |
| WeightReuse BF16 / Convert / Quant | 94 / 94 / 94 | 94 / 94 / 94 | 0 / 0 / 0 | 188 / 188 / 188 |

920f-4 上 `audit_matmul_extent_pairs.py` 的最终结果如下。Native 与
NativeFixedStreamingSVE 的相同目标计数一致。

| ARM catalog | 逻辑 case | 注册数 | malformed / unpaired / duplicate |
|---|---:|---:|---:|
| Core | 337 | 674 | 0 / 0 / 0 |
| Pack | 126 | 252 | 0 / 0 / 0 |
| Scenario | 510 | 1020 | 0 / 0 / 0 |
| Fusion | 542 | 1084 | 0 / 0 / 0 |
| Packed | 459 | 918 | 0 / 0 / 0 |
| WeightReuse main | 442 | 884 | 0 / 0 / 0 |
| WeightReuse BF16 / Convert / Quant | 94 / 94 / 94 | 188 / 188 / 188 | 0 / 0 / 0 |
| Policy（Fixed） | 36 | 72 | 0 / 0 / 0 |
| MMLA experiment（Native） | 288 | 576 | 0 / 0 / 0 |

## 正确性门禁

| 主机/配置 | 结果 |
|---|---|
| local, GCC Release | `MatmulTest` 20/20，Batch 4/4，Fusion 4/4，Pack 8/8 |
| local, GCC Debug | `MatmulTest` 20/20，Batch 4/4，Fusion 4/4，Pack 11/11（含 death test） |
| 920f-4, Native | `MatmulTest` 21/21，Batch 5/5，Fusion 8/8，Pack 6/6 |
| 920f-4, NativeFixedStreamingSVE | 21/21，5/5，8/8，6/6；运行时 SVL=512 |

本地 9 个最终 benchmark 二进制也在最终重建后串行完成 fixed-one：4,120/4,120 个
variant，0 error、0 skip。

ARM 的 20 个正式 benchmark 二进制均以 fixed-one 模式串行运行；总计 11,440 个
Dynamic/Const variant（每个 JSON 另含 mean/median/stddev/CV 四类 aggregate record），
全部 reference 检查为 0 error、0 skip。最终 Core 重建后也再次单独验证为 Native 与
Fixed 各 674 variant、0 duplicate。

Native 与 Fixed 的 SME MOPA/call 反汇编门禁各四项全部通过。允许列表仅包含 BiSheng
产生的精确编译器辅助符号（predicate tuple 的 `make_word_group/get_word`、Fusion 的
`numeric_limits` min/max/infinity）；仍禁止其他 call，MOPA 数量限制未放宽。

## 编译分片与资源

| 大型源 | shard 数 | ARM 代表/最终峰值 RSS | 本地峰值 |
|---|---:|---:|---:|
| Core | 176 | 2,751,164 KiB | — |
| Scenario | 306 | 2,664,252 KiB | — |
| Fusion | 396 | 4,034,180 KiB | — |
| Packed | 480 | 2,108,164 KiB | — |
| WeightReuse | 260 | 5,345,644 KiB（代表单 case 4,755,524 KiB） | — |
| 统一源最终全量重建 | 按架构选择 | 5,964,692 KiB / 36:10.21 | — |
| 本地全套受影响目标 | 按上表生成 | — | 1,601,592 KiB |

ARM 最终峰值低于 6 GiB，本地低于 2 GiB。Fusion/F64/量化等重 tuple 进一步按 case
拆分；Core 与 WeightReuse 将 Dynamic/Const pair 放入同一 TU，并使用显式 `ShardTag`
避免宏依赖模板体发生 ODR 合并和重复注册。每次 shard 调整后均重新运行 CMake 并检查
`generated/shards`。

通用 shard marker 支持为同一源分别声明 x86/ARM 数量。Core benchmark 使用 5/176，
Pack benchmark 使用 8/10；Core/Batch/Fusion/Pack 测试分别使用
5/5、7/9、2/8、9/6。因此合并源文件不会在 x86 生成 SME 的空 shard。

统一源改名后的 ARM 最终重建覆盖 12 个 Native/Fixed benchmark/test 目标，wall
36:10.21、最大 RSS 5,964,692 KiB；Core/Pack 配对审计保持 337/674 与 126/252，
四个对应 benchmark fixed-one 共 1,852 variant、0 error/skip，8 个反汇编门禁全过。

## 构建与运行环境

本地路径：`/home/renyz/project/vecops-neo/cmake-build-matmul-unified-gcc-release` 与
`/home/renyz/project/vecops-neo/cmake-build-matmul-unified-gcc-debug`。配置命令：

```bash
cmake -S . -B cmake-build-matmul-unified-gcc-release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON
cmake -S . -B cmake-build-matmul-unified-gcc-debug \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=OFF
```

ARM 主机为 `920f-4`，同步路径 `/home/ryz/vecops-neo`，构建路径
`/home/ryz/vecops-neo/cmake-build-matmul-unified-clang-release`，编译器为
`/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux/bin/clang{,++}`（Clang 19.1.7）。
配置为 Release、tests/benchmarks ON、`VECOPS_FIXED_STREAMING_SVE_BITS=512`。项目同步使用
不带 `--delete` 的 rsync；所有远端命令均先 `source ~/.bashrc`。验证前检查到主机为
608 CPU、约 1 TiB 内存且负载可用；先以 `-j1` 测量重 shard，再把全量构建提高到
`-j64`，未干扰现有高负载任务。

远端同步与配置命令：

```bash
rsync -az \
  --exclude='cmake-build-*' --exclude='build/' --exclude='.cache/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  ./ 920f-4:~/vecops-neo/

ssh 920f-4 << 'EOF'
source ~/.bashrc
cd ~/vecops-neo
cmake -S . -B cmake-build-matmul-unified-clang-release \
  -DCMAKE_C_COMPILER="$(which clang)" \
  -DCMAKE_CXX_COMPILER="$(which clang++)" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON \
  -DVECOPS_FIXED_STREAMING_SVE_BITS=512
cmake --build cmake-build-matmul-unified-clang-release -j64
EOF
```

配对审计命令：

```bash
python3 benchmarks/audit_matmul_extent_pairs.py \
  cmake-build-matmul-unified-gcc-release/benchmarks/MatmulBench-Native \
  cmake-build-matmul-unified-gcc-release/benchmarks/MatmulPackBench-Native \
  cmake-build-matmul-unified-gcc-release/benchmarks/MatmulScenarioBench-Native \
  cmake-build-matmul-unified-gcc-release/benchmarks/MatmulFusionBench-Native \
  cmake-build-matmul-unified-gcc-release/benchmarks/MatmulPackedBench-Native
```
