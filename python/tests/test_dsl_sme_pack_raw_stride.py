# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME raw packing stages non-unit logical K strides before ZA loads."""
import platform
import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state


def template(side, shape=None, strides=None):
  if shape is None:
    shape = (T.dynamic(lower=0), T.dynamic(lower=0))
    strides = (T.dynamic(), T.dynamic())
  capacity = T.dynamic(lower=0)
  @T.template
  def kernel(X: T.Tensor(shape, strides, dtype="int8"),
             P: T.Tensor((capacity,), dtype="int8", access="output")):
    T.pack(T.load(X), P, side=side, format="sme")
  return kernel


def oracle(x, side, span):
  logical = x if side == "A" else x.T
  s,k = logical.shape
  p = np.zeros(((s+span-1)//span, (k+3)//4, span, 4), dtype=np.int8)
  for i in range(s):
    for j in range(k): p[i//span,j//4,i%span,j%4] = logical[i,j]
  return p.reshape(-1)


@pytest.mark.skipif(platform.machine().lower() not in {"aarch64","arm64"}, reason="requires native SME")
@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.parametrize("static", [False, True])
def test_native_sme_raw_pack_stride_padding_alias(side, static, tmp_path):
  span = get_vector_state().sme_bytes // 2
  shape = (17,19) if side == "A" else (19,17)
  values = (np.arange(np.prod(shape)).reshape(shape)%31-15).astype(np.int8)
  layouts = [np.array(values, order="C"), np.array(values, order="F"), values[:,::-1]]
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path/"cache", build_dir=tmp_path/"build")
  dynamic = None if static else T.compile(template(side), compiler=compiler)
  for x in layouts:
    expected = oracle(x,side,span)
    compiled = (T.compile(template(side,x.shape,tuple(v//x.itemsize for v in x.strides)),compiler=compiler)
                if static else dynamic)
    out = np.full(expected.size+13, 71, dtype=np.int8)
    compiled.operator.native(x,out,**compiled._core.specialization_values)
    np.testing.assert_array_equal(out[:expected.size],expected)
    np.testing.assert_array_equal(out[expected.size:],71)
  # Overlap requires historical capture before writing the packed prefix.
  x = layouts[0]
  expected = oracle(x,side,span)
  backing = np.full(expected.size+13, 71, dtype=np.int8)
  alias = backing[:x.size].reshape(x.shape)
  alias[:] = x
  compiled = (T.compile(template(side,alias.shape,tuple(v//alias.itemsize for v in alias.strides)),compiler=compiler)
              if static else dynamic)
  compiled.operator.native(alias,backing,**compiled._core.specialization_values)
  np.testing.assert_array_equal(backing[:expected.size],expected)
  np.testing.assert_array_equal(backing[expected.size:],71)
