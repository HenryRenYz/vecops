# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A spatial fringe must not downgrade the complete native tile body."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import children
from vecops.language._library.amx_matmul import _tiles
from vecops.language._library.sme_matmul import _grouped_tiles


def _nodes(block):
  for statement in block.statements:
    yield statement
    for child in children(statement):
      yield from _nodes(child)


@pytest.mark.parametrize("m,n", [(32, 48), (48, 32), (48, 48), (64, 48)])
def test_amx_fringe_retains_four_accumulator_body_and_one_owner(m, n):
  @T.template
  def kernel(
    A: T.Tensor((m // 16, 2, 16, 32), dtype="bfloat16"),
    B: T.Tensor((n // 16, 2, 16, 32), dtype="bfloat16"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    _tiles(A, B, C, initial=False, kr=32, paired_fringe=True)

  nodes = list(_nodes(lower(kernel.ir).body))
  owners = [node for node in nodes if isinstance(node, I.HardwareRegion)]
  assert len(owners) == 1 and len(owners[0].configuration) == 8
  stores = [node for node in nodes if isinstance(node, I.ArchitectureCall) and node.operation == "amx.store"]
  assert {node.tile for node in stores} >= {0, 1, 2, 3}


@pytest.mark.parametrize("m,n,dtype", [(33, 49, "float32"), (49, 33, "float32"), (17, 35, "float64")])
def test_sme_fringe_retains_multiple_tiles_and_one_owner(m, n, dtype):
  width = 8 if dtype == "float64" else 16

  @T.template
  def kernel(
    A: T.Tensor((2, (m + width - 1) // width, 2 * width), dtype=dtype),
    B: T.Tensor((2, (n + width - 1) // width, 2 * width), dtype=dtype),
    C: T.Tensor((m, n), dtype=dtype, access="output"),
  ):
    _grouped_tiles(A, B, C, initial=False, group=1, groups=2, acc_dtype=dtype, paired_fringe=True)

  kernel = lower(kernel.ir)
  nodes = list(_nodes(kernel.body))
  # Width is runtime SME metadata here: each selected branch owns exactly
  # one region, and no fringe introduces a nested ownership interval.
  owners = [node for node in nodes if isinstance(node, I.HardwareRegion)]
  assert owners
  assert all(not any(isinstance(child, I.HardwareRegion) for child in _nodes(owner.body)) for owner in owners)
  calls = [node for node in nodes if isinstance(node, I.ArchitectureCall) and node.operation == "sme.mopa"]
  assert any(node.tile == (7 if dtype == "float64" else 3) for node in calls)


@T.func
def _epilogue(value, row, column):
  return value * 2 + T.cast(row + column, value.dtype)


def template(family, dtype="bf16"):
  ad, bd, cd = ("bfloat16", "bfloat16", "float32") if dtype == "bf16" else ("uint8", "int8", "int32")
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), ad),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), bd),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), cd),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), cd, access="output"),
  ):
    product = T.dot((A, B), C, acc=T.load(Prior), implementation=family)
    if dtype == "bf16":
      T.store(C, product * 2 + T.indices((m, n), axis=0) + T.indices((m, n), axis=1))
    else:
      T.store(C, product * 2 + 1)

  return kernel


@pytest.fixture(scope="module", params=["bf16", "byte"])
def native(request, tmp_path_factory):
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  root = tmp_path_factory.mktemp("native-fringe")
  return T.compile(
    template(family, request.param),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
    pass_options={"cpu_matmul_native_schedule": "paired-fringe"},
  ), request.param


@pytest.mark.parametrize("m,n,k", [(17, 33, 17), (33, 49, 33), (48, 48, 65), (65, 35, 3), (33, 49, 0)])
def test_native_fringe_odd_k_seed_alias_and_epilogue(native, m, n, k):
  native, dtype = native
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  ad, bd, cd = (bf16, bf16, np.float32) if dtype == "bf16" else (np.uint8, np.int8, np.int32)
  a = np.zeros((m, 2 * k), dtype=ad)[:, ::-2]
  b = np.zeros((2 * k, n), dtype=bd)[::-2]
  a[:] = (np.arange(m * k).reshape(m, k) % 11) / (8 if dtype == "bf16" else 1)
  b[:] = (np.arange(k * n).reshape(k, n) % 13 - 6) / (8 if dtype == "bf16" else 1)
  poison = np.nan if dtype == "bf16" else 12345
  backing = np.full((m + 2, n + 2), poison, dtype=cd)
  c = backing[1:-1, 1:-1]
  prior = c[:1]
  prior[:] = np.arange(n) / 8 if dtype == "bf16" else 2147483500 + np.arange(n)
  compute = np.float64 if dtype == "bf16" else np.int64
  oracle = (a.astype(compute) @ b.astype(compute) + prior.copy()) * 2
  oracle += np.arange(m)[:, None] + np.arange(n)[None, :] if dtype == "bf16" else 1
  native.operator.native(a, b, prior, c, **native._core.specialization_values)
  np.testing.assert_array_equal(c, oracle.astype(cd))
  np.testing.assert_array_equal(backing[[0, -1]], poison)
  np.testing.assert_array_equal(backing[:, [0, -1]], poison)


@pytest.mark.parametrize("family,rows", [("amx", True), ("amx", False), ("sme", True), ("sme", False)])
def test_dual_fringe_shares_one_operand_load_per_k_group(family, rows):
  from vecops.language import amx, sme
  from vecops.language._library.amx_matmul import _tile_pair
  from vecops.language._library.sme_matmul import _grouped_tile_pair

  if family == "amx":

    @T.template
    def kernel(
      A: T.Tensor((2, 2, 16, 32), dtype="bfloat16"),
      B: T.Tensor((2, 2, 16, 32), dtype="bfloat16"),
      C: T.Tensor((32, 32), dtype="float32", access="output"),
    ):
      with amx.region(((16, 64),) * 8):
        _tile_pair(A, B, C, 0, 0, rows=rows, initial=False, kr=32)
  else:

    @T.template
    def kernel(
      A: T.Tensor((2, 2, 32), dtype="float32"),
      B: T.Tensor((2, 2, 32), dtype="float32"),
      C: T.Tensor((32, 32), dtype="float32", access="output"),
    ):
      with sme.region():
        _grouped_tile_pair(A, B, C, 0, 0, 16, rows=rows, initial=False, group=1, groups=2, raw_a=False, raw_b=False)

  nodes = list(_nodes(lower(kernel.ir).body))
  calls = [n for n in nodes if isinstance(n, I.ArchitectureCall)]
  op = "amx.tdp" if family == "amx" else "sme.mopa"
  assert sum(c.operation == op for c in calls) == 2
  loads = [c for c in calls if c.operation in {"amx.load", "sme.load"}]
  assert len(loads) == 3


@pytest.mark.parametrize("family", ["amx", "sme"])
@pytest.mark.parametrize("paired,expected", [(False, 6), (True, 9)])
def test_default_bulk_keeps_previous_architecture_expansion(family, paired, expected):
  from vecops.language._library.amx_matmul import _tiles_2x2
  from vecops.language._library.sme_matmul import _grouped_tiles_2x2

  if family == "amx":

    @T.template
    def kernel(
      A: T.Tensor((2, 2, 16, 32), dtype="bfloat16"),
      B: T.Tensor((2, 2, 16, 32), dtype="bfloat16"),
      C: T.Tensor((32, 32), dtype="float32", access="output"),
    ):
      _tiles_2x2(A, B, C, initial=False, kr=32, paired_fringe=paired)
  else:

    @T.template
    def kernel(
      A: T.Tensor((2, 2, 32), dtype="float32"),
      B: T.Tensor((2, 2, 32), dtype="float32"),
      C: T.Tensor((32, 32), dtype="float32", access="output"),
    ):
      _grouped_tiles_2x2(
        A, B, C, initial=False, group=1, groups=2, raw_a=False, raw_b=False, acc_dtype="float32", paired_fringe=paired
      )

  calls = [n for n in _nodes(lower(kernel.ir).body) if isinstance(n, I.ArchitectureCall)]
  assert sum(c.operation == ("amx.tdp" if family == "amx" else "sme.mopa") for c in calls) == (
    4 if family == "sme" and not paired else expected
  )


@pytest.mark.parametrize("family", ["amx", "sme"])
def test_explicit_paired_fringe_recipe_routes_to_native_leaf(family):
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  artifact = CPUBackend().lower(
    template(family, "bf16").ir,
    (),
    target="AVX512" if family == "amx" else "SME",
    options={"cpu_matmul_native_schedule": "paired-fringe"},
  )
  calls = [n for n in _nodes(artifact.kernel.body) if isinstance(n, I.ArchitectureCall)]
  assert any(c.operation == ("amx.tdp" if family == "amx" else "sme.mopa") for c in calls)
