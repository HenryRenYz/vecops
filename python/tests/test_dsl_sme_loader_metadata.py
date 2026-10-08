# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Selected SME owners keep producer addressing in callback-owned SSA.

Source/cross tests do not claim ARM speed. The Native fixture also runs on
x86 to protect the unchanged DAG's strict arithmetic and historical aliases;
the same fixture on ARM exercises the selected streaming owners.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_exact_blocks import _body, _cross_wrapper
from test_dsl_sme_family_owners import _OWNER, _assert_inlined_owners, _lower, _strided_lower, _strided_wrapper
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.cpp.matmul import MatmulEmitter
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter


def _owners(source):
  return [_body(source, match[0]) for match in re.finditer(_OWNER, source)]


def _k_bodies(owner):
  return [
    _body(owner[match.start() :], match[0])
    for match in re.finditer(r"/\* (?:full K groups: no per-element tail tests|one partial K group) \*/", owner)
  ]


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
@pytest.mark.parametrize("packed", (False, True))
def test_sme_owner_reads_only_callback_local_loader_metadata_in_k(dtype, packed):
  output = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  source = _lower(dtype=dtype, output=output, packed=packed, pipeline="affine")
  for owner in _owners(source):
    prefix, body = owner.split("/* SME ", 1)
    copies = re.findall(r"(?:auto\* const|const vecops::nint_t|const bool) (\w+_sme_loader_\w+)=(\w+);", prefix)
    assert copies
    for local, original in copies:
      assert not re.search(rf"\b{original}\b", body), f"escaped capture {original} remains in K"
      # A local may be optimized away when its consumer is a literal stride.
      assert prefix.count(local) == 1
    assert "selected SME owner: immutable producer addressing" in prefix
    k_bodies = _k_bodies(owner)
    assert k_bodies
    assert all(".data()" not in body for body in k_bodies)
  if packed:
    assert "sme_packed_k_cursor" in source
  else:
    assert "opt::strided" in source
  assert_no_vec_mask_arrays(source)


def test_owner_metadata_scope_preserves_strict_transformed_and_negative_stride_loads():
  source = _strided_lower("float16", "float16", transformed=True, vl=16, svl=64)
  for owner in _owners(source):
    prefix, body = owner.split("/* SME ", 1)
    assert "sme_loader_enabled" in prefix
    assert "vecops::vec::load_convert" in body and "opt::strided" in body
    assert "vecops::vec::mul" in body and "vecops::vec::add" in body
    assert "vecops::vec::fmadd" not in body
    assert all(".data()" not in body for body in _k_bodies(owner))
  assert_no_vec_mask_arrays(source)


class _ScopeEmitter(MatmulEmitter, MemoryEmitter):
  """Use the real owner scope with minimal unrelated emitter plumbing."""

  def __init__(self):
    self.parameter_metadata = {"A.size(0)": "entry_m", "A.stride(0)": "entry_stride"}
    self.parameter_pointers = {"A": "entry_a"}
    self.snapshots = {}
    self.materialized = {}
    self.sme_optional_packs = {}
    self.chunk_cache = {("previous",): "previous_value"}
    self.ir = type("Bindings", (), {"bindings": {"M": I.Expr(I.ScalarType("int64"), "raw", data="A.size(0)")}})()
    self.lines = []
    self.counter = 0

  def unique(self, prefix):
    self.counter += 1
    return f"local_{prefix}_{self.counter}"

  def line(self, value):
    self.lines.append(value)

  def scalar(self, value):
    if value.op == "symbol":
      return self.scalar(self.ir.bindings[value.data])
    if value.op == "raw":
      return self.parameter_metadata.get(value.data, value.data)
    if value.op == "literal":
      return str(value.data)
    if value.op in {"local", "frozen"}:
      return value.data
    raise AssertionError(f"unexpected fixture expression {value.op}")


def _scope_fixture():
  emitter = _ScopeEmitter()
  m = I.Expr(I.ScalarType("int64"), "symbol", data="M")
  stride = I.Expr(I.ScalarType("int64"), "raw", data="A.stride(0)")
  view = I.View("A", "float32", (m, I.literal(7)), (stride, I.literal(1)), I.literal(0))
  load = I.Expr(I.TileType("float32", view.shape), "load", data=view)
  saved = I.View("saved", "float32", view.shape, (stride, I.literal(1)), I.literal(0), owned=True)
  # Null Buffer data() is legal, even when historical_alias is false. An
  # optional accessor or pointer + offset must not be evaluated at entry.
  guarded = I.View(
    "optional.value()",
    "float32",
    view.shape,
    (I.Expr(I.ScalarType("int64"), "raw", data="optional.value().stride(0)"), I.literal(1)),
    I.Expr(I.ScalarType("int64"), "raw", data="optional.value().offset()"),
  )
  emitter.snapshots[id(load)] = (saved, "historical_alias")
  emitter.materialized[id(load)] = (guarded, "materialization_enabled")
  return emitter, load


def test_scope_freezes_only_pure_bases_and_abi_leaves_without_pointer_arithmetic():
  emitter, load = _scope_fixture()
  saved = emitter.snapshots[id(load)][0]
  with emitter.sme_loader_metadata((load, load)):
    prefix = "\n".join(emitter.lines)
    assert prefix.count("=entry_a;") == prefix.count("=entry_m;") == prefix.count("=entry_stride;") == 1
    assert prefix.count("=saved.data();") == 1
    assert "optional.value()" not in prefix and "+" not in prefix
    assert "historical_alias" in prefix and "materialization_enabled" in prefix
    assert emitter.snapshots[id(load)][0] is saved
    assert "sme_loader_base" in emitter.pointer(load.data)
    assert emitter.parameter_metadata["A.size(0)"] != "entry_m"
    assert "optional.value()" in emitter.pointer(emitter.materialized[id(load)][0])


def test_scope_keeps_mutable_state_and_unknown_raw_metadata_at_original_consumption():
  emitter, load = _scope_fixture()
  unknown = I.Expr(I.ScalarType("int64"), "raw", data="mutable_induction")
  view = I.View("A", "float32", load.data.shape, (unknown, I.literal(1)), unknown)
  original = I.Expr(load.type, "load", data=view)
  with emitter.sme_loader_metadata((original,)):
    assert "mutable_induction" not in "\n".join(emitter.lines)
    assert emitter.scalar(unknown) == "mutable_induction"


def test_scope_restores_all_maps_and_read_identity_after_nested_owner_and_exception():
  emitter, load = _scope_fixture()
  names = ("parameter_metadata", "parameter_pointers", "snapshots", "materialized", "sme_optional_packs", "chunk_cache")
  originals = {name: getattr(emitter, name) for name in names}
  original_snapshot = emitter.snapshots[id(load)]
  with pytest.raises(RuntimeError, match="deliberate"), emitter.sme_loader_metadata((load,)):
    outer = {name: getattr(emitter, name) for name in names}
    with emitter.sme_loader_metadata((load,)):
      assert emitter.snapshots[id(load)][0] is original_snapshot[0]
    assert all(getattr(emitter, name) is value for name, value in outer.items())
    raise RuntimeError("deliberate")
  assert all(getattr(emitter, name) is value for name, value in originals.items())
  assert emitter.snapshots[id(load)] is original_snapshot


@pytest.mark.parametrize(
  "dtype,output,packed,vl,svl",
  (("float16", "float32", False, 16, 64), ("bfloat16", "float32", False, 64, 16), ("float64", "float64", True, 32, 64)),
)
def test_owner_loader_ssa_real_arm_cross_retains_mopa_and_inline_state(tmp_path, dtype, output, packed, vl, svl):
  source = _lower(dtype=dtype, output=output, packed=packed, pipeline="affine", vl=vl, svl=svl)
  assembly = _cross(tmp_path, source, lambda source: _cross_wrapper(source, dtype, output, False), vl=vl, svl=svl)
  _assert_inlined_owners(assembly, 9, vl=vl, svl=svl)


def test_owner_loader_ssa_real_arm_cross_retains_strict_transformed_dag(tmp_path):
  source = _strided_lower("float16", "float16", transformed=True, vl=16, svl=64)
  assembly = _cross(
    tmp_path,
    source,
    lambda source: _strided_wrapper(source, "float16", "float16", transformed=True),
    vl=16,
    svl=64,
  )
  _assert_inlined_owners(assembly, 9, vl=16, svl=64)


def _native_template(dtype):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  A0, A1, B0, B1, C0, C1 = (T.dynamic() for _ in range(6))
  acc = "float64" if dtype == "float64" else "float32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), dtype),
    B: T.Tensor((K, N), (B0, B1), dtype),
    C: T.Tensor((M, N), (C0, C1), acc, access="inout"),
  ):
    captured = T.load(A)
    # A=C aliases exercise historical read protection before owner entry;
    # the following matmul must not reload this captured read after the write.
    T.store(C, T.cast(captured, acc))
    a = T.cast(T.cast(captured, acc) * 0.5 + 0.125, dtype)
    value = T.dot((a, T.load(B)), C, acc=T.load(C))
    T.store(C, T.max(value * 0.5 + 0.125, 0))

  return kernel


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_owner_loader_metadata_strict_and_historical_alias(dtype, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(
    _native_template(dtype),
    compiler=compiler,
    pass_options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"},
  )
  with vecops.collect_compile_requests() as collector:
    kernel.operator(
      vecops.TensorMeta((3, 17), strides=(37, 2), dtype=dtype),
      vecops.TensorMeta((17, 17), strides=(37, 2), dtype=dtype),
      vecops.TensorMeta((3, 17), strides=(37, 2), dtype=dtype),
      **kernel._core.specialization_values,
    )
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(531)
  for m, n in ((3, 17), (64, 1), (1, 64), (19, 23)):
    for negative in (False, True):
      for alias in (False, True):
        a = rng.integers(-4, 5, size=(m, 2 * n + 3)).astype(dtype)[:, : 2 * n : 2]
        b = rng.integers(-4, 5, size=(n, 2 * n + 3)).astype(dtype)[:, : 2 * n : 2]
        a[:] /= np.dtype(dtype).type(16)
        b[:] /= np.dtype(dtype).type(16)
        if negative:
          a, b = a[::-1, ::-1], b[::-1, ::-1]
        c = a if alias else np.empty_like(a)
        original = a.copy()
        transformed = (original * np.dtype(dtype).type(0.5) + np.dtype(dtype).type(0.125)).astype(dtype)
        expected = np.maximum((transformed @ b + original) * np.dtype(dtype).type(0.5) + np.dtype(dtype).type(0.125), 0)
        kernel.operator.native(a, b, c, **kernel._core.specialization_values)
        np.testing.assert_array_equal(c, expected, err_msg=f"{dtype} {(m, n)} negative={negative} alias={alias}")
