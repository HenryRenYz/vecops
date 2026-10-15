# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME column stores and independent group-one native packing policies."""

import platform

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import children
from vecops.language._library.sme_matmul import dot_into


def _nodes(block):
  for statement in block.statements:
    yield statement
    for child in children(statement):
      yield from _nodes(child)


def static_template(
  dtype, a_natural=False, b_natural=False, *, pack=True, owner=False, prefetch=False, k=65, paired=False
):
  m, n = 64, 48
  acc_dtype = "float32" if dtype == "bfloat16" else dtype

  @T.template
  def kernel(
    A: T.Tensor((m, k), (1, m) if a_natural else (k, 1), dtype),
    B: T.Tensor((k, n), (n, 1) if b_natural else (1, k), dtype),
    C: T.Tensor((m, n), (1, m), acc_dtype, access="output"),
  ):
    dot_into(
      (T.load(A), T.load(B)),
      C,
      acc_dtype=acc_dtype,
      raw_a=A,
      raw_b=B,
      pack_raw_inputs=pack,
      share_pack_owner=owner,
      prefetch_inputs=prefetch,
      paired_fringe=paired,
    )

  return kernel


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
  "a_natural,b_natural,expected", [(True, True, 0), (True, False, 1), (False, True, 1), (False, False, 2)]
)
def test_group_one_natural_side_is_not_repacked(dtype, a_natural, b_natural, expected):
  artifact = CPUBackend().lower(static_template(dtype, a_natural, b_natural).ir, (), target="SME")
  packed = [node for node in _nodes(artifact.kernel.body) if isinstance(node, I.Allocate) and len(node.view.shape) == 3]
  assert len(packed) == expected


def test_disabled_optional_packing_retains_raw_group_one_path():
  artifact = CPUBackend().lower(static_template("float32", pack=False).ir, (), target="SME")
  assert not any(isinstance(node, I.Allocate) and len(node.view.shape) == 3 for node in _nodes(artifact.kernel.body))


def test_column_output_uses_vertical_za_stores_without_full_result_buffer():
  artifact = CPUBackend().lower(static_template("bfloat16").ir, (), target="SME")
  nodes = list(_nodes(artifact.kernel.body))
  assert any(isinstance(node, I.ArchitectureCall) and node.operation == "sme.store_ver" for node in nodes)
  assert not any(
    isinstance(node, I.Allocate)
    and node.view.dtype == "float32"
    and tuple(I.constant(x) for x in node.view.shape) == (64, 48)
    for node in nodes
  )


def dynamic_template(dtype, *, owner=False, prefetch=False, paired=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  acc_dtype = "float32" if dtype == "bfloat16" else dtype

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), dtype),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), acc_dtype),
    C: T.Tensor((m, n), (1, T.dynamic()), acc_dtype, access="output"),
  ):
    dot_into(
      (T.load(A), T.load(B)),
      C,
      acc=T.load(Prior),
      acc_dtype=acc_dtype,
      raw_a=A,
      raw_b=B,
      broadcast_acc=True,
      pack_raw_inputs=True,
      share_pack_owner=owner,
      prefetch_inputs=prefetch,
      paired_fringe=paired,
    )

  return kernel


def _collect_dynamic_request(compiled, dtype):
  acc_dtype = "float32" if dtype == "bfloat16" else dtype
  arguments = (
    vecops.TensorMeta((65, 65), strides=(65, 1), dtype=dtype),
    vecops.TensorMeta((65, 35), strides=(1, 65), dtype=dtype),
    vecops.TensorMeta((1, 35), strides=(35, 1), dtype=acc_dtype),
    vecops.TensorMeta((65, 35), strides=(1, 67), dtype=acc_dtype, access=vecops.TensorAccess.output),
  )
  with vecops.collect_compile_requests() as collector:
    compiled.operator(*arguments, **compiled._core.specialization_values)
  assert len(collector.requests) == 1
  return collector.requests


def _prepare_dynamic(compiled, dtype):
  # Metadata-only preparation preserves the Dynamic shape/stride contract.
  # All following native calls share that artifact, including alias layouts.
  vecops.compile_batch(_collect_dynamic_request(compiled, dtype), parallelism=2, execution_parallelism=1)
  return compiled


@pytest.fixture(
  scope="module",
  params=[("bfloat16", False), ("float32", False), ("float64", False), ("bfloat16", True), ("float32", True)],
)
def native(request, tmp_path_factory):
  if platform.machine().lower() not in {"aarch64", "arm64"}:
    pytest.skip("SME native execution requires an ARM SME host")
  dtype, owner = request.param
  root = tmp_path_factory.mktemp("sme-layout-policy")
  compiled = T.compile(
    dynamic_template(dtype, owner=owner, paired=True),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
  )
  return _prepare_dynamic(compiled, dtype), dtype


@pytest.mark.parametrize(
  "a_natural,b_natural,k",
  [(False, False, 65), (True, False, 65), (False, True, 65), (True, True, 65), (False, False, 1), (True, True, 0)],
)
def test_native_column_fringe_seed_alias_and_one_sided_pack(native, a_natural, b_natural, k):
  compiled, dtype = native
  dt = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
  acc = np.float32 if dtype == "bfloat16" else np.dtype(dtype)
  m, n = 65, 35
  a = np.empty((m, k), dtype=dt, order="F" if a_natural else "C")
  b = np.empty((k, n), dtype=dt, order="C" if b_natural else "F")
  a[:] = (np.arange(m * k).reshape(m, k) % 11) / 8
  b[:] = (np.arange(k * n).reshape(k, n) % 13 - 6) / 8
  backing = np.full((n + 2, m + 2), np.nan, dtype=acc).T
  output = backing[1:-1, 1:-1]
  prior = output[:1]
  prior[:] = np.arange(n) / 8
  oracle = a.astype(np.float64) @ b.astype(np.float64) + prior.copy()
  compiled.operator.native(a, b, prior, output, **compiled._core.specialization_values)
  np.testing.assert_allclose(output, oracle.astype(acc), rtol=2e-6, atol=1e-7)
  assert np.isnan(backing[[0, -1]]).all() and np.isnan(backing[:, [0, -1]]).all()


@pytest.mark.parametrize("m,n,k", [(0, 65, 1), (65, 0, 1), (1, 65, 1), (65, 1, 65)])
def test_native_shared_owner_empty_and_single_spatial_axes(native, m, n, k):
  compiled, dtype = native
  dt = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
  acc = np.float32 if dtype == "bfloat16" else np.dtype(dtype)
  a = np.ones((m, k), dtype=dt)
  b = np.ones((k, n), dtype=dt, order="F")
  backing = np.full((n + 2, m + 2), np.nan, dtype=acc).T
  output = backing[1 : 1 + m, 1 : 1 + n]
  prior = np.full((1, n), 0.25, dtype=acc)
  compiled.operator.native(a, b, prior, output, **compiled._core.specialization_values)
  np.testing.assert_allclose(output, np.full((m, n), k + 0.25, dtype=acc))
  assert np.isnan(backing[[0, -1]]).all() and np.isnan(backing[:, [0, -1]]).all()


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_optional_shared_pack_and_compute_has_one_owner_and_no_internal_allocations(dtype):
  artifact = CPUBackend().lower(static_template(dtype, owner=True).ir, (), target="SME")
  nodes = list(_nodes(artifact.kernel.body))
  owners = [node for node in nodes if isinstance(node, I.HardwareRegion)]
  shared = [
    owner
    for owner in owners
    if {"sme.store_bits", "sme.mopa"}.issubset(
      {node.operation for node in _nodes(owner.body) if isinstance(node, I.ArchitectureCall)}
    )
  ]
  # Alias/layout rejection retains separately owned ordinary fallback code.
  assert len(shared) == 1
  owner_nodes = list(_nodes(shared[0].body))
  assert not any(isinstance(node, (I.Allocate, I.HardwareRegion)) for node in owner_nodes)
  assert any(isinstance(node, I.ArchitectureCall) and node.operation == "sme.store_bits" for node in owner_nodes)
  assert any(isinstance(node, I.ArchitectureCall) and node.operation == "sme.mopa" for node in owner_nodes)


@pytest.mark.parametrize("dtype,owner", [("bfloat16", True), ("float32", True), ("float64", False)])
def test_dynamic_layout_and_shared_owner_templates_lower(dtype, owner):
  CPUBackend().lower(dynamic_template(dtype, owner=owner).ir, (), target="SME")


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("proof", ["none", "A", "B", "packed"])
def test_shared_policy_keeps_missing_raw_proofs_and_external_packed_compatible(owner, proof):
  has_a, has_b, packed = proof == "A", proof == "B", proof == "packed"

  @T.template
  def kernel(
    A: T.Tensor((16, 17), dtype="bfloat16"),
    B: T.Tensor((17, 16), dtype="bfloat16"),
    C: T.Tensor((16, 16), dtype="float32", access="output"),
  ):
    pa = T.alloc((576,), dtype="bfloat16") if packed else None
    pb = T.alloc((576,), dtype="bfloat16") if packed else None
    dot_into(
      (T.load(A), T.load(B)),
      C,
      raw_a=A if has_a else None,
      raw_b=B if has_b else None,
      packed_a=pa,
      packed_b=pb,
      share_pack_owner=owner,
    )

  CPUBackend().lower(kernel.ir, (), target="SME")


@pytest.mark.parametrize("owner", [False, True])
def test_optional_prefetch_stays_in_complete_k_prefix(owner):
  artifact = CPUBackend().lower(
    static_template("bfloat16", owner=owner, prefetch=True, k=193, paired=True).ir, (), target="SME"
  )
  hints = [node for node in _nodes(artifact.kernel.body) if isinstance(node, I.Prefetch)]
  assert hints and all(hint.level == "L2" and hint.intent == "read" for hint in hints)

  def hint_bounds(block, loops=()):
    for node in block.statements:
      if isinstance(node, I.Prefetch):
        yield I.constant(loops[-1].stop)
      for child in children(node):
        yield from hint_bounds(child, loops + ((node,) if isinstance(node, I.For) else ()))

  # ceil(193/2) - lookahead48: only groups 0..48 may issue a hint.
  assert set(hint_bounds(artifact.kernel.body)) == {49}


@pytest.fixture(scope="module", params=[False, True])
def prefetched_native(request, tmp_path_factory):
  if platform.machine().lower() not in {"aarch64", "arm64"}:
    pytest.skip("SME native execution requires an ARM SME host")
  root = tmp_path_factory.mktemp("sme-prefetch-policy")
  compiled = T.compile(
    dynamic_template("bfloat16", owner=request.param, prefetch=True, paired=True),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
  )
  return _prepare_dynamic(compiled, "bfloat16")


@pytest.mark.parametrize("k", [1, 97, 193])
def test_native_prefetch_prefix_and_no_hint_tail(prefetched_native, k):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  m, n = 65, 35
  a = np.full((m, k + 1), np.nan, dtype=bf16)[:, :k]
  b = np.full((n, k + 1), np.nan, dtype=bf16).T[:k, :]
  a[:] = (np.arange(m * k).reshape(m, k) % 11) / 8
  b[:] = (np.arange(k * n).reshape(k, n) % 13 - 6) / 8
  output = np.empty((m, n), dtype=np.float32, order="F")
  prior = output[:1]
  prior[:] = np.arange(n) / 8
  oracle = a.astype(np.float64) @ b.astype(np.float64) + prior.copy()
  prefetched_native.operator.native(a, b, prior, output, **prefetched_native._core.specialization_values)
  np.testing.assert_array_equal(output, oracle.astype(np.float32))


@pytest.mark.parametrize("m,n,k", [(0, 65, 1), (65, 0, 1), (1, 65, 1), (65, 1, 65), (65, 35, 1), (65, 35, 65)])
def test_shared_raw_fringe_hardware_preflight_accepts_actual_logical_bounds(m, n, k):
  from vecops.language._compiler.backends.cpu.hardware import preflight
  from vecops.language._compiler.dtypes import ELEMENT_BYTES
  from vecops.language._debug.interpreter import Interpreter

  artifact = CPUBackend().lower(dynamic_template("bfloat16", owner=True, paired=True).ir, (), target="SME")

  class MetadataInterpreter(Interpreter):
    def dtype(self, name):
      return bf16 if name == "bfloat16" else super().dtype(name)

    def value(self, expression):
      if isinstance(expression, I.Expr):
        if expression.op == "streaming_lanes":
          return 64 // ELEMENT_BYTES[expression.data]
        if expression.op == "lanes":
          dtype, power = expression.data
          return (64 // ELEMENT_BYTES[dtype]) * 2**power
        if expression.op == "cache_bytes":
          return 2 * 1024 * 1024
      return super().value(expression)

    def block(self, block):
      for statement in block.statements:
        if isinstance(statement, I.HardwareRegion):
          for condition, message in preflight(statement).checks:
            assert self.value(condition), message
        elif isinstance(statement, (I.Store, I.BindTile)):
          # Data movement is irrelevant to metadata preflight. Allocation
          # and scalar/control staging still use their real implementation.
          continue
        else:
          super().block(I.Block([statement]))

  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  a = np.empty((m, k), dtype=bf16)
  b = np.empty((k, n), dtype=bf16, order="F")
  c = np.empty((m, n), dtype=np.float32, order="F")
  prior = np.empty((1, n), dtype=np.float32)
  MetadataInterpreter(artifact.kernel, {"A": a, "B": b, "Prior": prior, "C": c}).block(artifact.kernel.body)


@pytest.mark.parametrize("owner", [False, True])
def test_dynamic_native_request_capture_keeps_one_artifact(owner, tmp_path):
  compiled = T.compile(
    dynamic_template("bfloat16", owner=owner),
    compiler=make_compiler(target="SME", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
  )
  _collect_dynamic_request(compiled, "bfloat16")
