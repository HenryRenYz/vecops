# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Alignment peeling owns only proven pointwise rows and preserves all bytes."""

from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.memory_proofs import collect_memory_proofs
from vecops.language._compiler.backends.cpu.verification import verify_prepared
from vecops.language._compiler.ir.traversal import nodes


def declaration(dtype="float32", width=257, negative=False):
  N = T.dynamic(lower=0, upper=64)
  AS = T.dynamic(lower=-8192, upper=8192)
  CS = T.dynamic(lower=-8192, upper=-width) if negative else T.dynamic(lower=width, upper=width + 17)
  Width = T.Const(width)
  DType = dtype

  @T.template
  def kernel(
    A: T.Tensor((N, Width), (AS, 1), DType),
    C: T.Tensor((N, Width), (CS, 1), DType, access="output"),
  ):
    T.store(C, T.load(A))

  return kernel


def test_plan_and_renderer_use_the_same_exact_store_owner():
  artifact = CPUBackend().lower(declaration().ir, (), target="AVX512")
  plans = artifact.backend_ir.aligned_stores
  assert plans and "alignment prologue" in artifact.source and "aligned_full_end" in artifact.source
  for identity, plan in plans.items():
    assert identity == id(plan.region)
    for effect, proof in zip(plan.region.effects, plan.proofs):
      assert effect.store is proof.store and proof.destination is proof.store.destination
      assert proof.minimum_columns == 257 and proof.occurrences == 1
  identity, plan = next(iter(plans.items()))
  bad = replace(plan, region=replace(plan.region))
  with pytest.raises(ValueError, match="stale aligned store"):
    verify_prepared(replace(artifact.backend_ir, aligned_stores={identity: bad}))
  proof_id, proof = next(iter(artifact.backend_ir.store_alignment_proofs.items()))
  with pytest.raises(ValueError, match="geometry or lexical"):
    verify_prepared(replace(artifact.backend_ir, store_alignment_proofs={proof_id: replace(proof, minimum_columns=999)}))


@pytest.mark.parametrize("target", ("Scalar", "SVE", "SME"))
def test_other_architectures_keep_existing_source(target):
  artifact = CPUBackend().lower(declaration().ir, (), target=target)
  assert not artifact.backend_ir.aligned_stores and "alignment prologue" not in artifact.source


@pytest.mark.parametrize("kind", ("small", "negative", "disabled"))
def test_unprofitable_or_unproved_rows_keep_existing_schedule(kind):
  artifact = CPUBackend().lower(
    declaration(width=17 if kind == "small" else 257, negative=kind == "negative").ir, (), target="AVX512",
    options={"cpu_align_stores": kind != "disabled"},
  )
  assert not artifact.backend_ir.aligned_stores and "alignment prologue" not in artifact.source


def test_shared_store_cannot_inherit_a_proof_from_another_occurrence():
  artifact = CPUBackend().lower(declaration().ir, (), target="AVX512")
  kernel = artifact.kernel
  store = next(node for node in nodes(kernel.body) if type(node) is I.Store)
  opaque = I.Assign("unknown", I.Expr(I.ScalarType("int64"), "opaque_scalar"), declare=True)
  kernel = replace(kernel, body=I.Block([store, opaque, store]))
  assert id(store) not in collect_memory_proofs(kernel, align_stores=True).alignment
  for kind in (I.Grid(("worker",), (I.literal(1),), I.Block([store])), I.HardwareRegion("amx", I.Block([store]))):
    nested = replace(kernel, body=I.Block([kind]))
    assert id(store) not in collect_memory_proofs(nested, align_stores=True).alignment


@pytest.mark.parametrize("dtype,width", (("float32", 257), ("float64", 257), ("uint8", 1025), ("int16", 257)))
@pytest.mark.parametrize("negative", (False, True))
def test_native_offsets_padding_reverse_input_empty_rows_and_aliases(tmp_path, dtype, width, negative):
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(declaration(dtype, width, negative), compiler=compiler)
  target = compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target
  if target.isa == "avx512":
    assert bool(compiled.backend_ir.aligned_stores) == (not negative)
  element = np.dtype(dtype).itemsize
  for rows in (0, 1, 3, 4, 17, 33):
    pitch = width + 7
    block = pitch * (rows + 1) + 64
    for offset in range(64 // element):
      for overlap in (False, True):
        owner = (np.arange(3 * block + 128, dtype=np.int64) % 101).astype(dtype)
        expected = owner.copy()
        a_offset = offset
        c_offset = offset + 1 if overlap else block + offset
        a_stride = -pitch if not overlap else pitch
        c_stride = -pitch if negative else pitch
        if a_stride < 0:
          a_offset += max(0, rows - 1) * pitch
        if c_stride < 0:
          c_offset += max(0, rows - 1) * pitch
        A = np.ndarray((rows, width), dtype, buffer=owner, offset=a_offset * element,
                       strides=(a_stride * element, element))
        C = np.ndarray((rows, width), dtype, buffer=owner, offset=c_offset * element,
                       strides=(c_stride * element, element))
        output = np.ndarray((rows, width), dtype, buffer=expected, offset=c_offset * element,
                            strides=(c_stride * element, element))
        output[:] = A.copy()
        compiled(A, C)
        np.testing.assert_array_equal(owner, expected)



def test_byte_unaligned_dtype_pointers_keep_the_original_unaligned_path(tmp_path):
  width, rows, pitch = 257, 3, 264
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(declaration(width=width), compiler=compiler)
  block = (pitch * (rows + 1) + 64) * 4
  owner = np.zeros(3 * block, dtype=np.uint8)
  A = np.ndarray((rows, width), "float32", buffer=owner, offset=1, strides=(pitch * 4, 4))
  C = np.ndarray((rows, width), "float32", buffer=owner, offset=block + 3, strides=(pitch * 4, 4))
  A[:] = np.arange(rows * width, dtype=np.float32).reshape(rows, width) / 100
  expected = owner.copy()
  output = np.ndarray(C.shape, C.dtype, buffer=expected, offset=block + 3, strides=C.strides)
  output[:] = A.copy()
  compiled(A, C)
  np.testing.assert_array_equal(owner, expected)


def multi_declaration(dtype="float32"):
  N = T.dynamic(lower=0, upper=64)
  AS = T.dynamic(lower=-8192, upper=8192)
  CS = T.dynamic(lower=257, upper=274)
  DS = T.dynamic(lower=257, upper=274)
  Width = T.Const(257)
  DType = dtype

  @T.template
  def kernel(
    A: T.Tensor((N, Width), (AS, 1), DType),
    C: T.Tensor((N, Width), (CS, 1), DType, access="output"),
    D: T.Tensor((N, Width), (DS, 1), DType, access="output"),
  ):
    x = T.load(A)
    p = x * x + 1
    q = p * x + 2
    T.store(C, q + p)
    T.store(D, q - p)

  return kernel


def test_fused_alignment_covers_every_exact_store_and_grouped_word():
  artifact = CPUBackend().lower(multi_declaration().ir, (), target="AVX512")
  plans = {k: p for k, p in artifact.backend_ir.aligned_stores.items() if len(p.proofs) == 2}
  assert plans and "&&" in artifact.source and "alignment prologue" in artifact.source
  identity, plan = next(iter(plans.items()))
  assert plan.region.base_power > 0
  assert tuple(effect.store for effect in plan.region.effects) == tuple(proof.store for proof in plan.proofs)
  for proofs in (plan.proofs[:1], plan.proofs[::-1]):
    with pytest.raises(ValueError, match="stale aligned store"):
      verify_prepared(replace(artifact.backend_ir, aligned_stores={identity: replace(plan, proofs=proofs)}))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_fused_alignment_phases_and_sequential_overlapping_outputs(tmp_path, dtype):
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(multi_declaration(dtype), compiler=compiler)
  target = compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target
  if target.isa == "avx512":
    assert any(len(plan.proofs) == 2 for plan in compiled.backend_ir.aligned_stores.values())
  element = np.dtype(dtype).itemsize
  for rows in (0, 1, 3, 4, 17, 33):
    pitch = 264
    block = pitch * (rows + 1) + 128
    for phase in range(64 // element):
      for delta in (0, 1, 3):
        for kind in ("disjoint", "outputs_overlap", "input_overlap"):
          owner = (np.arange(4 * block, dtype=np.int64) % 31).astype(dtype) / 16
          expected = owner.copy()
          a_offset = phase + max(0, rows - 1) * pitch
          c_offset = block + phase
          d_offset = 2 * block + phase + delta
          if kind == "outputs_overlap":
            d_offset = c_offset + delta
          elif kind == "input_overlap":
            c_offset = phase + 1
            d_offset = c_offset + delta
          A = np.ndarray((rows, 257), dtype, buffer=owner, offset=a_offset * element,
                         strides=(-pitch * element, element))
          C = np.ndarray((rows, 257), dtype, buffer=owner, offset=c_offset * element,
                         strides=(pitch * element, element))
          D = np.ndarray((rows, 257), dtype, buffer=owner, offset=d_offset * element,
                         strides=(pitch * element, element))
          ec = np.ndarray(C.shape, dtype, buffer=expected, offset=c_offset * element, strides=C.strides)
          ed = np.ndarray(D.shape, dtype, buffer=expected, offset=d_offset * element, strides=D.strides)
          x = A.copy()
          p = x * x + 1
          q = p * x + 2
          ec[:] = q + p
          ed[:] = q - p
          compiled(A, C, D)
          np.testing.assert_array_equal(owner, expected)
