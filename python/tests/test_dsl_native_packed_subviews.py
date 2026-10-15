# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Aligned windows keep root panel pitch; unaligned windows keep the decoder."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(family, *, shift=32, tiled=False, mixed=False):
  ca, cb = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype="bfloat16"), BP: T.Tensor((cb,), dtype="bfloat16"),
             B: T.Tensor((32, 32), dtype="bfloat16"),
             C: T.Tensor((32, 32), dtype="float32", access="output")):
    a = T.packed(AP, (96, 160), side="A", format=family)[shift:shift + 32, 32:64]
    b = T.load(B) if mixed else T.packed(BP, (160, 96), side="B", format=family)[32:64, shift:shift + 32]
    if tiled:
      T.library.matmul((a, b), C, implementation=family)
    else:
      T.dot((a, b), C, implementation=family)

  return kernel


@pytest.mark.parametrize("family", ["amx", "sme"])
@pytest.mark.parametrize("tiled,mixed", [(False, False), (True, False), (False, True)])
def test_aligned_native_subviews_borrow_root_stride(family, tiled, mixed):
  lowered = CPUBackend().lower(template(family, tiled=tiled, mixed=mixed).ir, (),
                              target="AVX512" if family == "amx" else "SME",
                              options={"_vector_state": VectorState(64, 64)})
  assert "borrowed aligned native packed subview with original panel pitch" in lowered.kernel.diagnostics
  body = lowered.source.partition("void __kernel__")[2]
  assert "packed input buffer too small" in body
  assert "packed input and matrix output may not overlap" in body
  assert "pack_offset" not in body


@pytest.mark.parametrize("family", ["amx", "sme"])
def test_unaligned_native_subviews_retain_decoder(family):
  lowered = CPUBackend().lower(template(family, shift=3).ir, (),
                              target="AVX512" if family == "amx" else "SME",
                              options={"_vector_state": VectorState(64, 64)})
  assert "borrowed aligned native packed subview with original panel pitch" not in lowered.kernel.diagnostics
  assert "pack_offset" in lowered.source.partition("void __kernel__")[2]


def pack(value, side, family, span):
  spatial, reduction = value.shape if side == "A" else value.shape[::-1]
  logical = value if side == "A" else value.T
  group, kr = 2, 32
  if family == "amx":
    out = np.full(((spatial + 15) // 16, (reduction + kr - 1) // kr, 16, kr), np.nan, dtype=value.dtype)
    for s in range(spatial):
      for k in range(reduction):
        if side == "A":
          out[s // 16, k // kr, s % 16, k % kr] = logical[s, k]
        else:
          out[s // 16, k // kr, k % kr // group, s % 16 * group + k % group] = logical[s, k]
  else:
    out = np.full(((spatial + span - 1) // span, (reduction + group - 1) // group, span * group), np.nan, dtype=value.dtype)
    for s in range(spatial):
      for k in range(reduction):
        out[s // span, k // group, s % span * group + k % group] = logical[s, k]
  return out.reshape(-1)


@pytest.mark.parametrize("tiled,mixed", [(False, False), (True, False), (False, True)])
def test_native_subview_numerics_original_pitch_and_alias_rejection(tiled, mixed, tmp_path):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  span = 2 * (get_vector_state().sme_bytes or 64) // 4
  if family == "sme" and span != 32:
    pytest.skip("catalog uses a 32-row physical panel")
  a = ((np.arange(96 * 160).reshape(96, 160) % 13 - 6) / 8).astype(bf16)
  b = ((np.arange(160 * 96).reshape(160, 96) % 11 - 5) / 8).astype(bf16)
  ap, bp = pack(a, "A", family, span), pack(b, "B", family, span)
  right = b[32:64, 32:64].copy()
  out = np.empty((32, 32), dtype=np.float32)
  compiled = T.compile(template(family, tiled=tiled, mixed=mixed),
                       compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  def call(c):
    compiled.operator.native(ap, bp, right, c, **compiled._core.specialization_values)
  call(out)
  np.testing.assert_array_equal(out, a[32:64, 32:64].astype(np.float32) @ right.astype(np.float32))
  before = ap.view(np.uint8).copy()
  with pytest.raises(Exception, match="may not overlap"):
    call(ap.view(np.float32)[:1024].reshape(32, 32))
  np.testing.assert_array_equal(ap.view(np.uint8), before)
