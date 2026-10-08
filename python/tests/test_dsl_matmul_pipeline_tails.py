# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Direct AMX pipeline tail/zero-K validation with fused input and epilogue."""

import pytest
import torch
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-amx-pipeline-tail")
  return make_compiler(
    target="Native", jobs=4, build_dir=root / "build", cache_dir=root / "cache"
  )


@pytest.mark.parametrize("mode", ["prefetch", "staged"])
@pytest.mark.skipif(not amx_available_for_test(), reason="explicit AMX pipeline requires an x86 AMX target")
def test_fused_amx_k_tails(compiler, mode):
  k = T.dynamic(lower=0)

  @T.func
  def affine(x):
    return x * 2 + 1

  @T.template
  def kernel(
    A: T.Tensor((17, k), dtype="bfloat16"),
    B: T.Tensor((k, 33), dtype="bfloat16"),
    D: T.Tensor((1, 33), dtype="float32"),
    C: T.Tensor((17, 33), dtype="float32", access="output"),
  ):
    a = T.cast(affine(T.load(A, dtype="float32")), "bfloat16")
    result = T.dot((a, T.load(B)), C, acc=T.load(D), implementation="amx")
    T.store(C, T.max(result * 0.5 + 1, 0))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_pipeline": mode})
  rng = torch.Generator().manual_seed(47)
  for K in (0, 1, 3, 31, 32, 33, 65):
    a = torch.empty_strided((17, K), (K, 1), dtype=torch.bfloat16)
    b = torch.empty_strided((K, 33), (33, 1), dtype=torch.bfloat16)
    a.copy_(torch.randn((17, K), generator=rng))
    b.copy_(torch.randn((K, 33), generator=rng))
    d = torch.randn((1, 33), generator=rng)
    c = torch.empty((17, 33))
    compiled(a, b, d, c)
    expected = (((a.float() * 2 + 1).bfloat16().float() @ b.float() + d) * 0.5 + 1).clamp_min(0)
    torch.testing.assert_close(c, expected, rtol=3e-5, atol=3e-5)
