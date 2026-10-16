<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Matmul 分级性能回归集（20261015）

这套固定清单来自本地 x86 与 ARM 920f-4 的完整已通过目录，用于后续 DSL、编译器和 C++ 后端调整。主目录是 x86 2,622、ARM 3,856 个**执行配置**，包括 M/N/K、dtype、输入/输出 pipeline、packing、Const/Dynamic、batch、family/policy；不是数千个独立 M/N/K。加上独立 packing、WeightReuse 和 Stateful Legacy 后，可比较注册分别为 3,241 / 4,555。主目录 packing 阶段是附加测量，不重复计作 shape；每平台 Prepared 的 5 个 API 缺口与默认关闭的 ARM MMLA 实验单列。

## 循环优化回归集

`dsl_loop_facts.py` covers forward-loop congruences, complete-panel bounds and
reverse-loop ranges independently of matrix scheduling. Its separate build and
time phases use the fixed-buffer protocol shared with
`dsl_compiler_optimizations.py`; both driver hashes participate in artifact
identity. The default twelve configurations include a separate captured-panel case
(`end = min(column + 16, Width); span = end - column`) for symbolic scalar
relations. They validate both modes for row counts
0, 1, 3, 4, 5 and 33, poisoning outputs and checking input/padding bytes before
timing. Timing reports wall nanoseconds per repeat and verifies outputs again.
Compare baseline and candidate within the same CPU-frequency epoch.

## 固定档位

所有档位逐级包含：`smoke ⊂ standard ⊂ extended ⊂ full`。同一架构上复测必须沿用相同版本的清单，不要每次按候选结果重新选“好看”的样例。两平台按各自合法目录选择，不能混用清单。

| 平台 | 档位 | 主计算 | 补充注册 | 合计 | 主packing附加阶段 | 主编译组 | 按25ms估算的正式轮次时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| x86 | smoke | 64 | 16 | 80 | 33 | 63 | 1.1 min |
| x86 | standard | 256 | 64 | 320 | 105 | 196 | 4.2 min |
| x86 | extended | 1024 | 256 | 1280 | 368 | 445 | 16.5 min |
| x86 | full | 2622 | 619 | 3241 | 842 | 1511 | 40.8 min |
| arm | smoke | 64 | 16 | 80 | 31 | 64 | 1.1 min |
| arm | standard | 256 | 64 | 320 | 151 | 197 | 4.7 min |
| arm | extended | 1024 | 256 | 1280 | 382 | 642 | 16.6 min |
| arm | full | 3856 | 699 | 4555 | 1276 | 2266 | 58.3 min |

表中时间仅按 `(主计算+补充注册+主packing阶段)×2实现×12轮×25ms`估算，实际轮次耗时可浮动；**不是测得的完整耗时或用时承诺**。它不含校准、加载、输入生成、oracle、预后校验和编译。编译器调整通常使产物失效，需要重新 lowering/native compile；看“主编译组”比只看 shape 数更有意义，补充编译成本另计。

- `smoke`（约80项）：每次局部修改后，快速发现主要路径和已知哨兵回退。
- `standard`（约320项）：一般优化提交前，增加 dtype/pipeline、尾部、Const/Dynamic、packing/batch 组合覆盖。
- `extended`（约1,280项）：编译器 pass、分派、融合、布局传播等跨路径调整后；ARM 包含三个已观察到的超大编译组代表项。
- `full`：发布、合并广泛编译器变更或更新正式基线。覆盖清单内全部可比较配置。

ARM `smoke/standard` 排除已记录的三个约9–13MB通用TU对应配置，防止最小集被长编译拖住；`extended/full` 再纳入。其准确ID与证据在 `arm/sentinels.json`，不能据此声称其他动态模块一定编译很快。

## 选择依据与边界

生成器按真实 catalog 字段做确定性加权覆盖：主路径、dtype/内存类型、pipeline、Const/Dynamic、geometry、M/N/K对4/16/32/64的边界、packing、batch/broadcast、family/policy以及组合；轻微偏好复用已选编译组。额外固定本轮恢复失败族、量化packing回退、online packing热点和编译压力哨兵。历史哨兵用到了既有结果，**不是独立留出集上的检出率验证**。

每平台 `coverage.json` 给出计数、缺失特征和已覆盖风险标签；100%已定义特征覆盖也不等于100%编译器分支或所有shape覆盖。较小集合可能漏掉未选配置的回退，不能以子集均值估计全量均值。完整逐项 metadata 留在 `build-matmul-opt-20261015/regression-subsets/<arch>/<tier>/workloads.json`。

每档文件：

- `main.json`：主计算配置，运行时同时测其适用的packing阶段。
- `main-packing.json`：仅上述集合中适用packing的ID，供packing阶段对比；不含`not_applicable`。
- `lifecycle.json`：WeightReuse + Stateful Legacy，保留真实R次计算。
- `packing.json`：独立MatmulPack，保留转换、补偿和强制selector。

## 主目录执行示例

先为待测版本建立**新的完整源码快照**，包含本轮新增的benchmark脚本、匹配原生扩展和SDK；不要修改旧正式快照或通过PYTHONPATH猜测实际导入。运行脚本本身也应来自该快照。修改Python compiler不等于可以继续使用旧编译产物：使用独立run目录，让源、SDK、toolchain和选项身份正常参与构建缓存。

下面以本地x86的smoke为例；`mm_snapshot`替换为待测版本实际快照：

```bash
mm_python=/opt/anaconda3/envs/af3/bin/python
mm_snapshot="$PWD/build-my-change/source"
mm_run="$PWD/build-my-change/x86-smoke"
mm_sets="$PWD/benchmarks/matmul_regression_sets/20261015/x86/smoke"
mm_cmd=("$mm_python" "$mm_snapshot/benchmarks/dsl_matmul_full_suite.py"
  --snapshot "$mm_snapshot" --suite-dir "$mm_run"
  --architecture x86 --feature HAS_AMX_BF16 --feature HAS_AMX_INT8
  --complete-registration --execution-ids-file "$mm_sets/main.json"
  --cpu 54 --rounds 12 --stages compute packing --timing-batch-size 16 --build-timeout 7200
  --cc /usr/bin/gcc --cxx /usr/bin/g++)
"${mm_cmd[@]}" --plan-only
"${mm_cmd[@]}" --phase build --build-workers 4 --jobs 4 --build-cpus 64-79
# 构建全部结束，检查负载、计时核心及同核线程空闲后：
"${mm_cmd[@]}" --phase time
"${mm_cmd[@]}" --phase report
```

ARM在`ssh 920f-4`后先`source ~/.bashrc`和`conda activate af3`，项目为`/home/ryz/vecops-neo`。使用arm清单、`--architecture arm --cpu 181`；省略`--feature`会选择清单所需的`HAS_SME`与`HAS_SME_F64F64`，也可以明确传入这两个宏名。`sme_f64`不是注册器识别的宏名，会遗漏FP64配置。显式编译器为`/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux/bin/clang`与项目内`build-math-unify/clangxx-gcc15`。构建核心需根据共享主机当前负载选择；不能照搬x86核心集合。正式计时保持静默、同主机、同CPU、同轮数，所有本任务构建/guard退出后执行。

## 补充目录执行

使用快照内`dsl_matmul_supplemental_accelerated.py`原流程，分别加：

```text
--suite lifecycle --registration-ids-file <tier>/lifecycle.json
--suite packing   --registration-ids-file <tier>/packing.json
```

其余必需参数为`--root <snapshot> --work-dir <independent-dir> --architecture x86|arm --features ... --cc ... --cxx ...`。可先`--plan-only`核对注册数。随后`--lower-only`、`--native-only`，使用相同worker配置和选择文件。按实际`native/batch-*`逐批执行`dsl_matmul_supplemental_validate.py --batch-index i`，aggregate validation，再执行`dsl_matmul_supplemental_timing.py --phase pre --batch-index i`。所有worker退出且资源检查通过后，用绑定全部pre文件SHA的quiet-barrier进行`--phase measure --cpu ... --rounds 12`，再逐批`--phase post --batch-index i`，最后aggregate timing。

从构建、pre、measure到post应保持`CC/CXX/PATH`等被记录的外部环境一致；仅显式传入`--cc/--cxx`并不能消除环境身份差异。本轮ARM曾因计时控制器与构建环境不同而在post拒绝准入，按checkpoint恢复完整环境后才继续校验，不能通过修改key或删guard解决。

这些校验不可删去。完整已验证的prepare/finish流程见本轮artifact根目录中的`supplemental-prepare-{x86,arm}-packed-checked.sh`与`finish_platform.py`；复用流程时须换独立目录、精确选择文件及所选注册数，不能原样覆盖全量结果或沿用全量硬编码计数。缺失或失败项不能静默排除。

## 新旧对比与回退判定

当前全量正式结果可直接作为所有子集的起始基线，无需为每档重测一次baseline。未来比较应在对应主机保留完整证据路径；ARM搬运的摘要可用于阅读，supplemental严格对比还需要原始checkpoint、source和binary可读。

```bash
python benchmarks/dsl_matmul_regression_compare.py \
  --kind main --architecture x86 --selection "$mm_sets/main.json" \
  --baseline build-matmul-opt-20261015/x86-packed-checked \
  --candidate "$mm_run" --output "$mm_run/compare-compute.json"
# packing阶段改用 main-packing.json，并加 --stage packing。
# supplemental使用 --kind supplemental，选择lifecycle.json或packing.json；
# baseline/candidate为相应目录的timing-results.json。
```

比较器拒绝缺失、重复、未通过、架构/metadata变化、非有限耗时及准入证据无效的结果，输出绑定输入SHA。每项同时报告DSL新/旧、C++新/旧、相对DSL/C++变化和绝对ns；总体geo/median不覆盖逐项回退。默认单实现耗时增长同时**超过10%且超过50ns**即要求review，C++单独回退也会被列出。编译器细调可用`--relative 0.05 --absolute-ns 0`，对很小的耗时变化保持敏感，但须处理测量噪声。

主机频率或运行环境变化后，不要把历史耗时直接当作当前基线。
`dsl_matmul_paired_regression.py`可以复用两份已构建suite中的保留二进制，
在当前环境重新交替测量baseline、candidate与C++，同时保留独立数值与packing
ABI检查。它验证精确配置、源码清单、build证据和二进制SHA，不修改旧suite，
不读取旧耗时作为比较值；输出还记录可读的CPU频率信息。所有构建及验证worker
结束后执行，例如：

```bash
python benchmarks/dsl_matmul_paired_regression.py \
  --root "$mm_snapshot" --baseline-suite build-matmul-opt-20261015/x86-packed-checked \
  --candidate-suite "$mm_run" --selection "$mm_sets/main.json" \
  --architecture x86 --cpu 54 --rounds 12 --output build-my-change/paired-current-frequency
```

疑似回退在新的输出目录独立复测；不能通过复用旧结果或覆写原结果消除回退。

如果仅前端继续修正，而希望复用先前候选的本机编译产物，必须先对全部选中编译组
使用`dsl_matmul_source_equivalence.py --phase emit`分别重新生成两版程序，再以
`--phase compare --baseline <old-emission> --candidate <new-emission>
--native-suite <old-suite> --output <proof.json>`核对。证明绑定完整C++源码、ABI、
捕获参数、有效编译选项、SDK、原生扩展与工具链，并逐项检查实际构建归档内的源码、
adapter和specialization；任一程序输入变化，都必须重新编译该组。只有全部一致，
才可给paired工具传入`--source-equivalence <proof.json>`，把旧二进制作为最终前端
输出的精确C++回放。报告应保留两个快照身份，不能把回放称作最终快照的新编译。

补充目录可用`dsl_matmul_supplemental_paired.py`在独立输出目录重测：提供
`--root`、两份`--baseline-suite/--candidate-suite`、原始构建快照
`--baseline-root/--candidate-root`、精确`--selection`、
`--architecture`、
`--cpu`和`--rounds 12`。它成对校验并交错测量四个实现（旧/新DSL与旧/新C++），
重新执行完整SDK/工具链pre/post guard，guard记录写入新输出目录，保留原生构建
checkpoint不变。完整guard在各自原始快照的独立进程中执行，恢复checkpoint中
记录的外部编译环境。如果`--root`就是`--candidate-root`，直接验证该原始前端；
否则必须提供最终前端与候选原生源码的`--source-equivalence`证明，不能替换旧guard的
Python实现身份。packing采用双poison逐字节oracle，lifecycle保留原数值oracle。
它的结果格式独立于原来的supplemental timing chain，不能直接冒充旧格式交给summary。

已有`dsl_state.py`、`dsl_iteration.py`、`dsl_grid.py`、`dsl_attention_catalog.py`
也提供`--prepare-only`（构建并验证，不计时）与`--cache-only`（缺失原生产物即失败）。
两阶段使用相同驱动、参数、构建目录和环境；grid目前只在单进程驱动支持此模式。

通用优化使用固定的`dsl_compiler_optimizations.py`：5 类程序分别覆盖循环不变量、
不变分支、scratch 生命周期、融合压力和私有死写，宽度为17/257/4096。
`--phase build`记录源码、构建成本并验证数值；`--phase time`只允许已有缓存。
两版使用同一驱动、参数及各自独立的`--work-dir`，静默后按旧/新/新/旧顺序测量。

该驱动的`compileropt-fixed-buffers-v1`协议在任何`T.compile`和程序factory之前，
先reserve并绑定目标CPU，为全部选中程序分配、首次写入输入及输出，再恢复编译核心
亲和性。每个程序的A/C/D来自同一个owning byte allocation的互不重叠视图：
base按4096字节对齐，各块容量为payload向4096上取整后再留4096字节padding，
A/C/D相对各块起点的固定偏移为0/64/128字节。4096是地址布局协议粒度，
不是对操作系统页大小的假设；选择这组固定偏移是为了控制布局，没有搜索更快布局。
结果逐项记录真实指针、`ptr%4096`和`ptr%64`，配置和manifest包含协议版本。
旧manifest因此不能用于新协议的`--phase time`，应使用新的独立目录重新构建两版。

原有两mode数值oracle保留，time完成后再对每个程序完整复验两mode。
主度量仍为原生ABI重复循环的wall ns/repeat，不做CPU时间归一化，也未新增paired API。
固定64字节对齐及页内偏移会改变评测条件，绝对地址、物理页和cache放置仍可能不同；
这组结果应与同协议基线比较，不能替换或改写此前独立布局的R1/R2结果，
也不能仅据新结果推断原差异来自NUMA、分配器或某个pass。

`dsl_scratch_lifetimes.py`另外覆盖晚期存储规划：两个顶层循环使用互不重叠的
私有buffer，宽度17/257/4096、行数1/32共6项。原有优化benchmark的重复循环
会使保守顶层区间重叠，不能据此验证slot复用；这个补充程序将重复放在计时驱动中。
同样先`--phase build`再`--phase time`，必需参数为`--work-dir`、`--output`、
`--cpu`、`--cc`、`--cxx`，默认11轮。准备阶段检查负stride输入和两个输出，
并记录instrumented scratch峰值与物理slot数；计时阶段重新核对全部身份和数值。
峰值指DSL scratch，不是进程RSS；耗时包含Python循环和原生ABI调用开销，
不能与仅测内核循环的benchmark直接比较。以上均为固定评测，不搜索编译参数。

有疑似回退时，在独立run目录、相同负载条件下复测，使用`--confirmation <independent-run>`；重复超阈值才标`repeated_regression`。退出码0为本集合未触发阈值，1为疑似回退需要review，2为证据或配置错误。0不表示所有未选shape无回退；12轮也不自动提供统计独立性或环境无漂移保证。C++漂移须单独观察，不能仅用相对比值除掉DSL真实变慢。

## 编译器正确性与编译成本

性能集合之外保留正确性门槛，尤其是别名、负stride、历史读取和异常值：这些并未在benchmark metadata中完整表达，生成器不会伪造其覆盖。对应现有专项包括`test_dsl_affine_metadata.py`、`test_dsl_small_native_policy.py`、`test_dsl_online_packed_binding.py`、`test_dsl_transpose_capture.py`、`test_dsl_scatter_conversion.py`、`test_dsl_compensated_packing.py`；ARM另加`test_dsl_sme_pack_raw_stride.py`与`test_dsl_sme_pack_payload_stride.py`。使用`dsl_snapshot_entry.py`在匹配快照上运行，注意测试所需平台及源SDK；它们与性能注册不重复累计。

编译器变化还应分别记录lowering时间、native编译时间、生成源码/DSO体积、峰值内存和运行性能。冷构建与缓存命中分开，不能用仅执行旧产物的性能测试证明新编译器生成速度。ARM超大TU代表项已纳入extended；完整旧/新输入与编译预算需保存。

## 清单工具初次验证

本节记录建立清单工具时的验证，不作为后续编译器改动的性能证据；频率或编译策略
变化后，仍需按前述方式重新成对测量新旧版本。

相关单元测试61项通过；实际x86主目录和补充目录的plan-only筛选通过，包括相对选择路径；两平台主结果和两平台补充packing的严格self-compare通过，ARM补充比较在920f-4读取原始证据。x86历史结果已按四档共同通过项回放。所有选中配置均来自本轮完整正式测量，没有额外重复构建四档，也没有测得各档端到端耗时。

## 清单维护

`dsl_matmul_regression_subsets.py`可从完整已准入的plan/ledger和supplemental summary重新生成下一版；传入明确sentinel文件。只在新增catalog、变更覆盖策略或确认新回退哨兵时创建新版本目录，记录差异，切勿在同一候选评测中漂移清单。已有source-backed ID变更会明确报错，不能自动丢弃。
