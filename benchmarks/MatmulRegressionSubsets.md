<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Matmul 分级性能回归集（20261015）

这套固定清单来自本地 x86 与 ARM 920f-4 的完整已通过目录，用于后续 DSL、编译器和 C++ 后端调整。主目录是 x86 2,622、ARM 3,856 个**执行配置**，包括 M/N/K、dtype、输入/输出 pipeline、packing、Const/Dynamic、batch、family/policy；不是数千个独立 M/N/K。加上独立 packing、WeightReuse 和 Stateful Legacy 后，可比较注册分别为 3,241 / 4,555。主目录 packing 阶段是附加测量，不重复计作 shape；每平台 Prepared 的 5 个 API 缺口与默认关闭的 ARM MMLA 实验单列。

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

ARM在`ssh 920f-4`后先`source ~/.bashrc`和`conda activate af3`，项目为`/home/ryz/vecops-neo`。使用arm清单、`--architecture arm --feature sme_f64 --cpu 181`；显式编译器为`/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux/bin/clang`与项目内`build-math-unify/clangxx-gcc15`。构建核心需根据共享主机当前负载选择；不能照搬x86核心集合。正式计时保持静默、同主机、同CPU、同轮数，所有本任务构建/guard退出后执行。

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

有疑似回退时，在独立run目录、相同负载条件下复测，使用`--confirmation <independent-run>`；重复超阈值才标`repeated_regression`。退出码0为本集合未触发阈值，1为疑似回退需要review，2为证据或配置错误。0不表示所有未选shape无回退；12轮也不自动提供统计独立性或环境无漂移保证。C++漂移须单独观察，不能仅用相对比值除掉DSL真实变慢。

## 编译器正确性与编译成本

性能集合之外保留正确性门槛，尤其是别名、负stride、历史读取和异常值：这些并未在benchmark metadata中完整表达，生成器不会伪造其覆盖。对应现有专项包括`test_dsl_affine_metadata.py`、`test_dsl_small_native_policy.py`、`test_dsl_online_packed_binding.py`、`test_dsl_transpose_capture.py`、`test_dsl_scatter_conversion.py`、`test_dsl_compensated_packing.py`；ARM另加`test_dsl_sme_pack_raw_stride.py`与`test_dsl_sme_pack_payload_stride.py`。使用`dsl_snapshot_entry.py`在匹配快照上运行，注意测试所需平台及源SDK；它们与性能注册不重复累计。

编译器变化还应分别记录lowering时间、native编译时间、生成源码/DSO体积、峰值内存和运行性能。冷构建与缓存命中分开，不能用仅执行旧产物的性能测试证明新编译器生成速度。ARM超大TU代表项已纳入extended；完整旧/新输入与编译预算需保存。

## 本次工具验证

相关单元测试61项通过；实际x86主目录和补充目录的plan-only筛选通过，包括相对选择路径；两平台主结果和两平台补充packing的严格self-compare通过，ARM补充比较在920f-4读取原始证据。x86历史结果已按四档共同通过项回放。所有选中配置均来自本轮完整正式测量，没有额外重复构建四档，也没有测得各档端到端耗时。

## 清单维护

`dsl_matmul_regression_subsets.py`可从完整已准入的plan/ledger和supplemental summary重新生成下一版；传入明确sentinel文件。只在新增catalog、变更覆盖策略或确认新回退哨兵时创建新版本目录，记录差异，切勿在同一候选评测中漂移清单。已有source-backed ID变更会明确报错，不能自动丢弃。
