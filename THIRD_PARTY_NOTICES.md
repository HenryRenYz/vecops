<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Third-party notices

The root [LICENSE](LICENSE) covers project-owned contributions. It does not
replace the terms, copyrights, or disclaimers of third-party material below.
Retain this document and the relevant `LICENSES/` files when redistributing
source or binaries containing that material, including installed headers.
The SPDX header above covers the original prose in this document; quoted
upstream notices retain their original terms.

These are source references, not claims that the complete upstream libraries
are vendored. A **comparison reference** is a fixed version used to verify
the cited implementation and notices. Where the original import revision was
not recorded, the reference does not establish that revision or a complete
historical chain of ownership. Those gaps remain to be reviewed before release.

## Half-precision conversion: PyTorch and FP16

- Local material: `include/vecops/util/Float16.h`, the fallback
  `fp16_to_fp32` and `fp16_from_fp32` implementations and their comments;
  `include/vecops/vec/details/x86/Arithmetic.h`, their vector adaptations.
- The scalar file explicitly attributes its copied code to PyTorch.
  Comparison reference: PyTorch `v2.9.0`, commit
  `0fabc3ba44823f257e70ce397d989c8de5e362c1`,
  [torch/headeronly/util/Half.h](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/torch/headeronly/util/Half.h).
  The original copied revision is not recorded. Local changes include type
  and bit-cast integration, platform conditionals, and vector adaptation.
- The implementations also share substantial conversion comments and
  expressions with FP16. Comparison reference: commit
  `782eea126dc5c755827be751a099eb01826175cf`,
  [include/fp16/fp16.h](https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h).
  Similarity alone does not prove the original import chain. Both sets of
  upstream notices are retained pending that review; they are not presented
  as alternative permissions to discard either set.
- PyTorch's original BSD-style terms and full contributor copyright list are
  reproduced unchanged in [LICENSES/PyTorch-BSD.txt](LICENSES/PyTorch-BSD.txt),
  from that fixed commit's `LICENSE`. This includes its source/binary notice
  conditions and non-endorsement condition.
- FP16's original MIT terms and these original notices are reproduced
  unchanged in [LICENSES/FP16-MIT.txt](LICENSES/FP16-MIT.txt):

  ```text
  Copyright (c) 2017 Facebook Inc.
  Copyright (c) 2017 Georgia Institute of Technology
  Copyright 2019 Google LLC
  ```

## Arm optimized-routines

Local material:

| Path under `include/vecops/vec/details/` | Material and local adaptation |
|---|---|
| `Math.h` | Shared logarithm tables and minimax coefficients; tables reorganized into vecops constants |
| `sve/math/Exp.h` | Exponential coefficients and kernel structure; templated bases, accuracy tiers, and local special-case handling |
| `sve/math/Log.h` | Logarithm coefficients, tables and reduction structure; vecops integration |
| `sve/math/Trig.h` | Trigonometric reduction, SVE instruction choices and pi-scaled coefficients; templated families and local special cases |
| `sve/math/Reciprocal.h` | Reciprocal-square-root refinement design reference; source comments describe vecops accuracy tiers, not a verbatim upstream file |
| `x86/math/Log.h` | Arm-derived coefficients/tables adapted to x86 evaluation |
| `x86/math/Exp.h` | Arm-derived f32 coefficient sets alongside the separately attributed SLEEF sets |

Fixed reference: commit `67126040cf80f956676fbf473c2d9bebdb475283` of
[ARM-software/optimized-routines](https://github.com/ARM-software/optimized-routines/tree/67126040cf80f956676fbf473c2d9bebdb475283).
`sve/math/Trig.h` already records this source commit. For the other families
it is a comparison reference; their original import revision is not recorded.

The referenced sources carry `MIT OR Apache-2.0 WITH LLVM-exception`.
Their complete original dual-license text, including the Arm root copyright,
is retained unchanged in
[LICENSES/Arm-optimized-routines.txt](LICENSES/Arm-optimized-routines.txt).
Arm-derived material remains available under those upstream terms; the
project-owned MIT notice does not substitute its owner or years.

Original per-file notices from the fixed reference are retained below.
Paths are relative to its `math/aarch64/` directory.

| Original notice | Referenced source files |
|---|---|
| Copyright (c) 2019-2026, Arm Limited. | `sve/cos.c`, `sve/cosf.c`, `sve/expf.c`, `sve/logf.c`, `sve/sin.c`, `sve/sinf.c` |
| Copyright (c) 2023-2025, Arm Limited. | `sve/cospi.c`, `sve/cospif.c`, `sve/sinpi.c`, `sve/sinpif.c`, `sve/tan.c` |
| Copyright (c) 2023-2026, Arm Limited. | `sve/exp.c`, `sve/exp10.c`, `sve/exp10f.c`, `sve/exp2.c`, `sve/exp2f.c`, `sve/sincos.c`, `sve/sincosf.c` |
| Copyright (c) 2020-2026, Arm Limited. | `sve/log.c`, `sve/tanf.c` |
| Copyright (c) 2022-2026, Arm Limited. | `sve/log10.c`, `sve/log10f.c`, `sve/log2.c`, `sve/log2f.c` |
| Copyright (c) 2025, Arm Limited. | `sve/rsqrt.c`, `sve/rsqrtf.c` |
| Copyright (c) 2024-2025, Arm Limited. | `sve/sincospi.c`, `sve/sincospif.c`, `sve/tanpi.c`, `sve/tanpif.c` |
| Copyright (c) 2022-2024, Arm Limited. | `v_log10_data.c`, `v_log2_data.c` |
| Copyright (c) 2019-2024, Arm Limited. | `v_log_data.c` |

## SLEEF

- Local material: `include/vecops/vec/details/x86/math/Exp.h`, the
  SLEEF-derived `xexp2`/`xexp10` minimax sets; and `x86/math/Trig.h`, the
  Strict f32/f64 unary u35 sets and cited reduction design.
  Other Taylor coefficients in the latter file are separately identified
  by its existing comments; they are not all attributed to SLEEF.
- Comparison reference: commit `7623d6cfa2712462880fa63a4d0f0b5f775d1a83`,
  [src/libm/sleefsimddp.c](https://github.com/shibatch/sleef/blob/7623d6cfa2712462880fa63a4d0f0b5f775d1a83/src/libm/sleefsimddp.c)
  and [src/libm/sleefsimdsp.c](https://github.com/shibatch/sleef/blob/7623d6cfa2712462880fa63a4d0f0b5f775d1a83/src/libm/sleefsimdsp.c).
  The original import revision is not recorded. Local changes include
  extracting coefficient sets and expressing them in vecops templates.
- Original source copyright:

  ```text
  Copyright Naoki Shibata and contributors 2010 - 2025.
  ```

- Original Boost Software License 1.0 (`BSL-1.0`) text is retained unchanged
  in [LICENSES/SLEEF-BSL-1.0.txt](LICENSES/SLEEF-BSL-1.0.txt).

## LLVM libc design reference

`include/vecops/vec/details/x86/math/Trig.h` cites LLVM libc's table-free
small-range design, alongside SLEEF. This does not assert that complete LLVM
functions or all local coefficients were copied. Comparison reference:
commit `5215171a44273c2b24e6d6266f39ee8f5b04c30e`,
[libc/src/__support/math](https://github.com/llvm/llvm-project/tree/5215171a44273c2b24e6d6266f39ee8f5b04c30e/libc/src/__support/math),
including `sin.h`, `sinf.h`, `cos.h`, `cosf.h`, `tan.h`, and `tanf.h`.
The original design-reference revision is not recorded.

The referenced files carry this original notice and identifier:

```text
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

The original `libc/LICENSE.TXT` is retained unchanged in
[LICENSES/LLVM-libc.txt](LICENSES/LLVM-libc.txt), including the LLVM exception
and historical license appendix. This is a conservative reference notice,
not a relicensing of unrelated project code.

## KUPL

Fixed dependency revision, recorded in
`benchmarks/ops/matmul-otherlibs/dependencies.json`:
`bd7059eae348f1d77f73a9fbb1cadfe7a6405fb8` of
`https://atomgit.com/kunpengcompute/kupl.git`.
The same revision is accessible from the
[GitHub upstream mirror](https://github.com/kunpengcompute/kupl/tree/bd7059eae348f1d77f73a9fbb1cadfe7a6405fb8),
including `src/kupl_mma.h`.

In addition to the external optional benchmark dependency via
`benchmarks/ops/matmul-otherlibs/KUPLBench.cpp`, this branch adapts the
`KP36_16x64x2_BF16BF16F32` schedule in:

- `include/vecops/matmul/details/kernel/sme/KuplMma.h` (provider catalog;
  its existing Huawei copyright and Mulan notice are preserved);
- `include/vecops/matmul/details/kernel/sme/Backend.h` (the conditional
  `VECOPS_HAS_KUPL_MMA` packed compute schedule);
- `include/vecops/matmul/details/packing/sme/Format.h` (the matching
  conditional A16/B64 packing selection).

The local adaptation integrates the fixed tile and packing choices with
vecops-owned initialization, epilogues, tails, scheduling and ZA lifetime.
The KUPL-derived portions retain Huawei copyright and MulanPSL-2.0 terms;
they are not covered solely by the project-owned MIT notice.

KUPL is licensed under `MulanPSL-2.0`. Its complete original bilingual
license is reproduced unchanged in
[LICENSES/KUPL-MulanPSL-2.0.txt](LICENSES/KUPL-MulanPSL-2.0.txt).
The original source statement is:

```text
Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

KUPL is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
       http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
```

## External dependencies and benchmark inputs

These dependencies are fetched or supplied separately, rather than made
project-owned by the root MIT license. Follow each dependency's actual
version and any component-specific notices when distributing it or material
embedded from it. The table records existing configuration; it does not lock
currently unbounded Python requirements or claim a build was performed.

| Dependency or reference | Existing configured version | License material |
|---|---|---|
| GoogleTest (optional tests) | `v1.17.0` tag | [Upstream LICENSE](https://github.com/google/googletest/blob/v1.17.0/LICENSE) |
| Google Benchmark (optional benchmarks) | `v1.8.3` tag | [Upstream LICENSE](https://github.com/google/benchmark/blob/v1.8.3/LICENSE) |
| pybind11 / scikit-build-core (Python build) | `>=2.13` / `>=0.11` | Record the actual installed versions and their notices for each distribution |
| NumPy / PyTorch (optional framework adapters) | `>=1.23` / `>=2.9` | Record actual framework versions and component notices; these adapters do not relicense those frameworks |
| OpenBLAS | `e0166008be8e466242aa76b2ff75ce3f0fbf574a` | [LICENSE](https://github.com/OpenMathLib/OpenBLAS/blob/e0166008be8e466242aa76b2ff75ce3f0fbf574a/LICENSE) |
| oneDNN | `0e2a5bfeef1bfbffc3137464606540233086ce9b` | [LICENSE](https://github.com/uxlfoundation/oneDNN/blob/0e2a5bfeef1bfbffc3137464606540233086ce9b/LICENSE) |
| LIBXSMM | `3668d249637792dcc988414eb3d2a374033743df` | [LICENSE.md](https://github.com/libxsmm/libxsmm/blob/3668d249637792dcc988414eb3d2a374033743df/LICENSE.md) |
| Arm Compute Library | `2a8ca9840a3112d12d6cc68e49e6c833f9759fd6` | [LICENSES/MIT.txt](https://github.com/ARM-software/ComputeLibrary/blob/2a8ca9840a3112d12d6cc68e49e6c833f9759fd6/LICENSES/MIT.txt); inspect the actual linked components |
| KUPL (optional benchmark provider) | `bd7059eae348f1d77f73a9fbb1cadfe7a6405fb8` | See KUPL above |
| AlphaFold 3 (shape reference) | `c0f97eda2f1f482fd94d3a38bece18c7069b4a5c` | [Source LICENSE](https://github.com/google-deepmind/alphafold3/blob/c0f97eda2f1f482fd94d3a38bece18c7069b4a5c/LICENSE) |

`benchmarks/ops/matmul-otherlibs/Cases.h` records representative dimensions
from AlphaFold 3 `GridSelfAttention` defaults; the source reference is
[modules.py](https://github.com/google-deepmind/alphafold3/blob/c0f97eda2f1f482fd94d3a38bece18c7069b4a5c/src/alphafold3/model/network/modules.py),
whose original notice is `Copyright 2024 DeepMind Technologies Limited`.
This checkout contains benchmark shape descriptions, not AlphaFold model
weights or a licensed copy of its databases. Source-code permission does not
grant rights to separately supplied weights, datasets, or service outputs.
Benchmark results describe their recorded revisions and environments; they
do not establish performance for a different release revision.

Before distributing an SDK, sdist, wheel, or linked benchmark executable,
check the actual artifact contains the applicable license texts and notices.
Adding these source-tree materials alone does not change installation or
packaging rules, or prove that artifact-level requirements have been met.
