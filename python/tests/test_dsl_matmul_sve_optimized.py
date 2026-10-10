# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Safe full-register PackedDot loads and FP16 widening-vector dot lowering.

Pure/cross tests inspect real lowering without claiming target performance.
Native tests separately exercise poisoned physical padding and logical tails.
"""

import ctypes
import mmap
import os
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import (
  arm_cross_cxx_include_flags,
  make_compiler,
)
from test_dsl_sve_packed_mmla import _lower, _sme_available, _template
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes, rewrite_expressions


def _fp16_template(*, shape=None, output="float32", transformed=False, epilogue="affine", library=False):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.dynamic(lower=0) for _ in range(3))
  A0, A1, D0, D1, C0, C1 = (T.dynamic() for _ in range(6))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), "float16"),
    B: T.Tensor((K, N), (1, K), "float16"),
    D: T.Tensor((M, N), (D0, D1), "float32"),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
    Bias: T.Tensor((1, N), dtype="float32"),
  ):
    a = T.cast(T.load(A, dtype="float32") * 0.5 + 0.125, "float16") if transformed else T.load(A)
    value = (
      T.library.matmul((a, T.load(B)), C, acc=T.load(D), swap=False, BK=33, implementation="vector")
      if library
      else T.dot((a, T.load(B)), C, acc=T.load(D), implementation="vector")
    )
    if epilogue == "affine":
      T.store(C, T.cast((value + T.load(Bias)) * 0.5 + 0.125, output))
    else:
      T.store(C, T.cast(T.fma(value + T.load(Bias), 0.5, 0.125), output))

  return kernel


def _fp16_lower(*, vl=64, svl=64, target="SME", **kwargs):
  return (
    CPUBackend()
    .lower(
      _fp16_template(**kwargs).ir,
      (),
      target=target,
      options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always", "_vector_state": VectorState(vl, svl)},
    )
    .source
  )


def test_fp16_widening_preserves_explicit_fma_epilogue():
  source = _fp16_lower(transformed=True, epilogue="fma")
  assert "dsl_recipe_dot_" in source
  assert "vecops::vec::fmadd(" in source
  assert "vecops::vec::mul(" in source and "vecops::vec::add(" in source


def _cross(tmp_path, source, wrapper, *, vl, svl):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real ARM codegen")
  root = Path(__file__).resolve().parents[2]
  cpp = tmp_path / "optimized.cpp"
  cpp.write_text(wrapper(source), encoding="utf-8")
  assembly = cpp.with_suffix(".s")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-S",
    "-ffp-contract=off",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-6000:]
  return assembly.read_text(encoding="utf-8")


def _fp16_wrapper(source):
  # The ordinary-SVE input vectors use VL, independent of the captured SVL.
  definitions = (
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    'extern "C" void fp16_dot(void* ap,void* bp,void* dp,void* cp,void* biasp,vecops::nint_t m,vecops::nint_t n,vecops::nint_t k) {',
    "auto shape=vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n});",
    "auto strides=vecops::tensor::make_strides(vecops::meta::Any{n},vecops::meta::Any{1});",
    "auto A=vecops::tensor::make_tensor(static_cast<vecops::float16_t*>(ap),vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{k}),vecops::tensor::make_strides(vecops::meta::Any{k},vecops::meta::Any{1}));",
    "auto B=vecops::tensor::make_tensor(static_cast<vecops::float16_t*>(bp),vecops::tensor::make_shape(vecops::meta::Any{k},vecops::meta::Any{n}),vecops::tensor::make_strides(vecops::meta::cint<1>,vecops::meta::Any{k}));",
    "auto D=vecops::tensor::make_tensor(static_cast<vecops::float32_t*>(dp),shape,strides);",
    "auto C=vecops::tensor::make_tensor(static_cast<vecops::float32_t*>(cp),shape,strides);",
    "auto Bias=vecops::tensor::make_tensor(static_cast<vecops::float32_t*>(biasp),vecops::tensor::make_shape(vecops::meta::cint<1>,vecops::meta::Any{n}),vecops::tensor::make_strides(vecops::meta::Any{n},vecops::meta::cint<1>));",
    "__kernel__(A,B,D,C,Bias);",
    "}",
  )
  return "\n".join(definitions)


def _strided_output(m, n, dtype, layout):
  strides = (n, 1) if layout == "row" else (1, m) if layout == "column" else (2 * n + 3, 2)
  backing = np.empty(max(1, (m - 1) * strides[0] + (n - 1) * strides[1] + 1), dtype=dtype)
  result = np.ndarray((m, n), dtype=dtype, buffer=backing, strides=tuple(s * np.dtype(dtype).itemsize for s in strides))
  return result[::-1, ::-1] if layout == "negative" else result


@contextmanager
def _guarded_copy(values):
  """End packed allocation exactly at an inaccessible page, not allocator slack."""
  page = mmap.PAGESIZE
  boundary = ((values.nbytes + page - 1) // page) * page
  allocation = mmap.mmap(-1, boundary + page)
  address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
  # from_address does not export the mmap buffer: the context can close the
  # allocation after use even while the caller still has the ndarray name.
  buffer = (ctypes.c_ubyte * values.nbytes).from_address(address + boundary - values.nbytes)
  result = np.ndarray(values.shape, dtype=values.dtype, buffer=buffer)
  result[:] = values
  libc = ctypes.CDLL(None, use_errno=True)
  libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
  libc.mprotect.restype = ctypes.c_int
  if libc.mprotect(address + boundary, page, 0) != 0:
    allocation.close()
    raise OSError(ctypes.get_errno(), "mprotect failed")
  try:
    yield result
  finally:
    allocation.close()


def test_guarded_packed_fixture_preserves_exact_extent():
  values = np.arange(37, dtype=np.uint16)
  with _guarded_copy(values) as result:
    np.testing.assert_array_equal(result, values)
    assert (result.ctypes.data + result.nbytes) % mmap.PAGESIZE == 0


def _mmla_poison_k_allowed(m, n, k, vl, svl, *, blocked=False):
  """Fixture oracle: only the ordinary-SVE MMLA consumer clears poisoned K.

  This is independent of the emitter's source strings. The external SME
  packed contract still requires zero K padding for every other consumer.
  BM/BN=2 round up to the SME spatial alignment (SVL/2), not to 2x2 leaves.
  BK=3 rounds to four BF16 lanes. Only when every resulting spatial leaf is
  an eligible family can the whole physical input's final K group be poison.
  """
  if vl is None or svl is None or vl != svl:
    return False
  if blocked:
    panel = svl // 2
    if m > panel or n > panel:
      return False
    k = min(k, 4)
  limits = {
    (2, 2): (16, 1025),
    (2, 4): (32, 1025),
    (2, 8): (64, 1025),
    (4, 2): (32, 257),
    (4, 4): (32, 257),
    (8, 2): (64, 1025),
  }
  minimum, maximum = limits.get((m, n), (None, None))
  return minimum is not None and vl >= minimum and 0 <= k <= maximum


@pytest.mark.parametrize(
  "shape,vl,svl,blocked,expected",
  (
    ((2, 2, 1025), 16, 16, False, True),
    ((2, 4, 1025), 32, 32, False, True),
    ((2, 8, 1025), 64, 64, False, True),
    ((4, 2, 257), 32, 32, False, True),
    ((4, 4, 257), 32, 32, False, True),
    ((8, 2, 1025), 64, 64, False, True),
    ((4, 2, 1025), 64, 64, False, False),
    ((4, 4, 258), 64, 64, False, False),
    ((2, 2, 1026), 64, 64, False, False),
    ((2, 8, 5), 32, 32, False, False),
    ((2, 4, 5), 16, 16, False, False),
    ((2, 2, 5), 16, 64, False, False),
    ((2, 2, 5), 64, 16, False, False),
    ((1, 2, 5), 64, 64, False, False),
    ((32, 32, 1025), 64, 64, True, False),
    ((8, 8, 1025), 16, 16, True, False),
    ((2, 2, 1025), 16, 16, True, True),
    ((4, 2, 1025), 32, 32, True, True),
    ((8, 8, 1025), 16, 64, True, False),
    ((3, 2, 5), 64, 64, True, False),
    ((2, 2, 5), None, 64, False, False),
  ),
)
def test_poison_fixture_respects_mmla_family_capacity_and_vector_domains(shape, vl, svl, blocked, expected):
  assert _mmla_poison_k_allowed(*shape, vl, svl, blocked=blocked) is expected


def _poison_pack(array, panel, *, poison_k):
  spatial, k = array.shape
  physical = np.full(((spatial + panel - 1) // panel, (k + 1) // 2, panel, 2), np.nan, dtype=array.dtype)
  for row in range(spatial):
    for col in range(k):
      physical[row // panel, col // 2, row % panel, col % 2] = array[row, col]
  if not poison_k and k % 2:
    physical[:, -1, :, 1] = 0
  return physical.reshape(-1)


def test_poison_fixture_keeps_spatial_nan_but_zeros_fallback_k_tail():
  pytest.importorskip("ml_dtypes")
  source = np.arange(6, dtype=np.float32).reshape(2, 3).astype("bfloat16")
  for poison_k in (False, True):
    packed = _poison_pack(source, 8, poison_k=poison_k).reshape(1, 2, 8, 2)
    assert np.isnan(packed[0, 0, 2:]).all()
    assert np.isnan(packed[0, 1, 2:, 0]).all()
    assert np.isnan(packed[0, 1, :, 1]).all() if poison_k else (packed[0, 1, :, 1] == 0).all()
    np.testing.assert_array_equal(packed[0, 0, :2], source[:, :2])
    np.testing.assert_array_equal(packed[0, 1, :2, 0], source[:, 2])


def _packed_window_template(panel, *, k_value=None):
  K = T.dynamic(lower=0) if k_value is None else T.Const(k_value)
  PSA, PSB = (T.dynamic(lower=0) for _ in range(2))
  D0, D1, C0, C1 = (T.dynamic() for _ in range(4))

  @T.template
  def packed_mmla_window(
    A: T.Tensor((2, K), dtype="bfloat16"),
    AP: T.Tensor((PSA,), dtype="bfloat16"),
    BP: T.Tensor((PSB,), dtype="bfloat16"),
    D: T.Tensor((2, 2), (D0, D1), "float32"),
    C: T.Tensor((2, 2), (C0, C1), "float32", access="output"),
    Bias: T.Tensor((1, 2), dtype="float32"),
    OriginA: int,
    OriginB: int,
    OriginK: int,
  ):
    assert OriginA >= 0 and OriginA + 2 <= panel
    assert OriginB >= 0 and OriginB + 2 <= panel
    assert OriginK >= 0 and OriginK % 2 == 0
    a = T.packed(AP, (2, K), side="A", format="sme")
    b = T.packed(BP, (K, 2), side="B", format="sme")
    value = T.dot((a, b), C, acc=T.load(D))
    T.store(C, T.max(value + T.load(Bias), 0))

  return packed_mmla_window


@contextmanager
def _packed_window_pass(panel):
  """Express the internal packed window IR, without mocking kernel dispatch.

  Public frontend slicing of a packed Expr is not implemented. This test
  plugin uses the same original-layout/origin representation as cache leaves,
  while retaining the 2xK/Kx2 logical types. It only touches its named kernel.
  """
  name = "test-packed-mmla-window"
  kernel_name = (
    f"{_packed_window_template.__module__}.{_packed_window_template.__qualname__}.<locals>.packed_mmla_window"
  )

  def window(ir, context):
    if ir.name != kernel_name:
      return T.passes.PassResult(ir, changed=False)
    k_origin = I.Expr(I.ScalarType("int64"), "raw", data="OriginK")

    def rewrite(value):
      if value.op != "packed_load":
        return value
      packed = value.data
      k = value.type.shape[1 if packed.side == "A" else 0]
      full_k = I.binary("add", k, k_origin)
      spatial = I.literal(panel)
      origin = I.Expr(I.ScalarType("int64"), "raw", data="OriginA" if packed.side == "A" else "OriginB")
      shape = (spatial, full_k) if packed.side == "A" else (full_k, spatial)
      origins = (origin, k_origin) if packed.side == "A" else (k_origin, origin)
      return replace(value, data=replace(packed, shape=shape, origins=origins))

    return rewrite_expressions(ir, rewrite)

  T.passes.register_pass(name, before="canonicalize-memory")(window)
  try:
    yield
  finally:
    T.passes.unregister_pass(name)


def test_packed_window_ir_retains_original_capacity_and_runtime_boundary_proof():
  with _packed_window_pass(32):
    template = _packed_window_template(32)
    result = CPUBackend().lower(
      template.ir, (), target="SME", options={"cpu_matmul_pipeline": "off", "_vector_state": VectorState(64, 64)}
    )
    packed = [value for value in nodes(result.kernel.body) if isinstance(value, I.Expr) and value.op == "packed_load"]
    assert packed and {value.data.side for value in packed} == {"A", "B"}
    for value in packed:
      spatial_axis, k_axis = (0, 1) if value.data.side == "A" else (1, 0)
      assert I.constant(value.type.shape[spatial_axis]) == 2
      assert I.constant(value.data.shape[spatial_axis]) == 32
      assert value.data.shape[k_axis].op == "add"
      assert value.data.shape[k_axis].args[1].data == "OriginK"
      assert value.data.origins[k_axis].data == "OriginK"
    source = result.source
    assert "OriginA" in source and "OriginB" in source and "OriginK" in source
    assert "packed input buffer too small" in source
    assert "vecops_dsl::pack_size<vecops::bfloat16_t, 2>" in source
  assert "test-packed-mmla-window" not in {p.name for p in T.passes.registered_passes()}


def test_packed_window_plugin_unregistered_on_exception_and_skips_other_kernels():
  with pytest.raises(RuntimeError, match="deliberate fixture exception"), _packed_window_pass(32):
    source = _lower(shape=(2, 2, 5), vl=64, svl=64)
    assert "OriginA" not in source and "OriginK" not in source
    raise RuntimeError("deliberate fixture exception")
  assert "test-packed-mmla-window" not in {p.name for p in T.passes.registered_passes()}


@pytest.mark.skipif(not _sme_available(), reason="actual ARM ordinary-SVE/SME packing required")
def test_bf16_arm_native_packed_window_poison_tail_and_guard_page(tmp_path):
  pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  assert state.sve_bytes is not None and state.sme_bytes is not None
  panel = state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  rng = np.random.default_rng(2022)
  with _packed_window_pass(panel):
    kernel = T.compile(_packed_window_template(panel), compiler=compiler, pass_options={"cpu_matmul_pipeline": "off"})
    assert "packed input buffer too small" in kernel.generated_source
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 257, 1025):
      for origin_k in (0, 2):
        a = (rng.integers(-3, 4, size=(panel, k + origin_k)) / 8).astype("bfloat16")
        b = (rng.integers(-3, 4, size=(k + origin_k, panel)) / 8).astype("bfloat16")
        poison_k = _mmla_poison_k_allowed(2, 2, k, state.sve_bytes, state.sme_bytes)
        ap, bp = (_poison_pack(array, panel, poison_k=poison_k) for array in (a, b.T))
        with _guarded_copy(ap) as guarded_a, _guarded_copy(bp) as guarded_b:
          for oa, ob in ((0, 0), (panel // 2, 0), (0, panel // 2), (panel - 2, panel - 2)):
            # The last origin's full register would cross the final physical
            # group into PROT_NONE; only the masked branch is legal there.
            av, bv = a[oa : oa + 2, origin_k:], b[origin_k:, ob : ob + 2]
            anchor = np.ascontiguousarray(av)
            bias = rng.integers(-2, 3, size=(1, 2)).astype(np.float32)
            for layout in ("row", "column", "negative"):
              c = _strided_output(2, 2, np.float32, layout)
              c[:] = rng.integers(-2, 3, size=(2, 2))
              d = np.broadcast_to(c[:1], (2, 2))
              expected = np.maximum(av.astype(np.float32) @ bv.astype(np.float32) + d.copy() + bias, 0)
              kernel.operator.native(
                anchor, guarded_a, guarded_b, d, c, bias, oa, ob, origin_k, **kernel._core.specialization_values
              )
              np.testing.assert_array_equal(c, expected, err_msg=f"{k=}, {oa=}, {ob=}, {origin_k=}, {layout=}")


@pytest.mark.skipif(not _sme_available(), reason="actual ARM ordinary-SVE/SME packing required")
@pytest.mark.parametrize("blocked", (False, True))
def test_bf16_arm_native_poison_padding_boundary_and_alias(blocked, tmp_path):
  pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  assert state.sve_bytes is not None and state.sme_bytes is not None
  panel = state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(
    _template(bm=2 if blocked else None, bn=2 if blocked else None, bk=3 if blocked else None), compiler=compiler
  )
  rng = np.random.default_rng(2020)

  shapes = ((panel, panel),) if blocked else ((2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2))
  for m, n in shapes:
    for k in (1, 2, 3, 4, 5, 6, 7, 9, 257, 1025):
      a = (rng.integers(-3, 4, size=(m, k)) / 8).astype("bfloat16")
      b = (rng.integers(-3, 4, size=(k, n)) / 8).astype("bfloat16")
      poison_k = _mmla_poison_k_allowed(m, n, k, state.sve_bytes, state.sme_bytes, blocked=blocked)
      packed_a, packed_b = (_poison_pack(array, panel, poison_k=poison_k) for array in (a, b.T))
      bias = rng.integers(-2, 3, size=(1, n)).astype(np.float32)
      context = _guarded_copy if blocked else nullcontext
      with context(packed_a) as ap, context(packed_b) as bp:
        for layout in ("row", "column", "negative"):
          c = _strided_output(m, n, np.float32, layout)
          c[:] = rng.integers(-2, 3, size=(m, n))
          for broadcast in (False, True):
            d = np.broadcast_to(c[:1], (m, n)) if broadcast else c
            expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + d.copy() + bias, 0)
            kernel.operator.native(a, ap, bp, d, c, bias, **kernel._core.specialization_values)
            np.testing.assert_array_equal(
              c, expected, err_msg=f"{blocked=}, {poison_k=}, {(m, n, k)}, {layout=}, {broadcast=}"
            )


@pytest.mark.parametrize("output", ("float32", "float16"))
@pytest.mark.parametrize("transformed,library", ((False, False), (True, False), (True, True)))
def test_fp16_native_widening_tails_fusion_alias_and_strides(output, transformed, library, tmp_path):
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(_fp16_template(output=output, transformed=transformed, library=library), compiler=compiler)
  rng = np.random.default_rng(2021)
  for m, n in ((1, 16), (16, 1), (2, 5)):
    for k in (0, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 67):
      a = (rng.integers(-3, 4, size=(m, k)) / 8).astype(np.float16)[:, ::-1]
      # Explicit backing.T preserves the promised column K stride, including
      # the N=1 case where asfortranarray may legally keep a row-major stride.
      backing = (rng.integers(-3, 4, size=(n, max(k, 1))) / 8).astype(np.float16)
      b = np.ndarray((k, n), dtype=np.float16, buffer=backing, strides=(2, max(k, 1) * 2))
      if k == 0:
        # The annotation's B pitch is K, even for an empty matrix.
        b = np.ndarray((k, n), dtype=np.float16, buffer=backing, strides=(2, 0))
      bias = rng.integers(-2, 3, size=(1, n)).astype(np.float32)
      for layout in ("row", "column", "negative"):
        c = _strided_output(m, n, output, layout)
        c[:] = rng.integers(-2, 3, size=(m, n))
        d = (
          np.broadcast_to(c[:1], (m, n))
          if output == "float32"
          else np.broadcast_to(np.ones((1, n), np.float32), (m, n))
        )
        av = (a.astype(np.float32) * 0.5 + 0.125).astype(np.float16) if transformed else a
        expected = ((av.astype(np.float32) @ b.astype(np.float32) + d.copy() + bias) * 0.5 + 0.125).astype(output)
        kernel.operator.native(a, b, d, c, bias, **kernel._core.specialization_values)
        np.testing.assert_array_equal(
          c, expected, err_msg=f"{output=}, {transformed=}, {library=}, {(m, n, k)}, {layout=}"
        )
