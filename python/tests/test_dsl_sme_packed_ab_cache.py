# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Whole-problem prepacked traversal: bounded, profile-selected and SVL-relative."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_sme_packed_ab_col import _OPTIONS, _count, _geometry, _output, _pack_template, _template
from test_dsl_sve_packed_mmla import _sme_available
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend

_OPTION = "cpu_sme_packed_ab_cache_bypass"
_MARKER = "/* SME packed-AB whole-problem geometry */"


def _lower(*, enabled=None, vl=64, svl=64, cost_profiles=None, **kwargs):
  options = {**_OPTIONS, "_vector_state": VectorState(vl, svl)}
  if enabled is not None:
    options[_OPTION] = enabled
  if cost_profiles is not None:
    options["cost_profiles"] = cost_profiles
  return CPUBackend().lower(_template(**kwargs).ir, (), target="SME", options=options).source


@pytest.mark.skipif(not _sme_available(), reason="actual ARM SME required")
def test_native_prepacked_cache_bypass_full_tail_fusion_alias_and_fallback(tmp_path):
  """Independent exact dyadic oracle, including zero-K and boundary rejects."""
  pytest.importorskip("ml_dtypes")
  lanes, panel, group = _geometry()
  shape = (2 * lanes, 8 * lanes + 1, 513)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  pack = T.compile(_pack_template(), compiler=compiler, pass_options=_OPTIONS)
  kernels = {
    "control": T.compile(_template(level="dot"), compiler=compiler, pass_options=_OPTIONS),
    "dynamic": T.compile(_template(level="matmul"), compiler=compiler, pass_options=_OPTIONS),
    "constant": T.compile(_template(shape=shape, level="matmul"), compiler=compiler, pass_options=_OPTIONS),
  }
  m, n, k = shape
  psa, psb = _count(m, k, panel, group), _count(n, k, panel, group)
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((m, k), dtype="float32"),
      vecops.TensorMeta((k, n), dtype="bfloat16"),
      vecops.TensorMeta((psa,), dtype="bfloat16"),
      vecops.TensorMeta((psb,), dtype="bfloat16"),
      **pack._core.specialization_values,
    )
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((m, k), dtype="float32"),
        vecops.TensorMeta((psa,), dtype="bfloat16"),
        vecops.TensorMeta((psb,), dtype="bfloat16"),
        vecops.TensorMeta((m, n), dtype="float32"),
        vecops.TensorMeta((m, n), dtype="float32"),
        vecops.TensorMeta((1, n), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 4
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  assert "dsl_recipe_dot_" in kernels["constant"].generated_source
  rng = np.random.default_rng(629)
  shapes = (shape, (8 * lanes + 1, n, k), (lanes, 8 * lanes - 1, k), (m, n, 511), (m, n, 0), (0, n, k), (m, 0, k))
  calls = 0
  for m, n, k in shapes:
    for negative in (False, True):
      a = rng.integers(-4, 5, size=(m, k)).astype("float32") / np.float32(16)
      b = (rng.integers(-3, 4, size=(k, n)) / 8).astype("bfloat16")
      if negative:
        a, b = a[:, ::-1], b[::-1]
      ap = np.full(_count(m, k, panel, group), 7, dtype="bfloat16")
      bp = np.full(_count(n, k, panel, group), 7, dtype="bfloat16")
      pack.operator.native(a, b, ap, bp, **pack._core.specialization_values)
      transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype("bfloat16").astype("float32")
      product = transformed @ b.astype("float32")
      for prior_mode in ("disjoint", "alias", "broadcast"):
        prior = _output(1 if prior_mode == "broadcast" else m, n, negative=negative)
        prior[:] = rng.integers(-2, 3, size=prior.shape) / 8
        bias = np.full((1, n), np.float32(0.125))
        expected = np.maximum(((product + prior.copy()) + bias) * np.float32(0.5) + np.float32(0.125), 0)
        for label, kernel in kernels.items():
          if label == "constant" and (m, n, k) != shape:
            continue
          d = prior.copy() if prior_mode != "alias" else _output(m, n, negative=negative)
          d[:] = prior
          c = d if prior_mode == "alias" else _output(m, n, negative=negative)
          physical = c.base.copy()
          kernel.operator.native(a, ap, bp, d, c, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(c, expected, err_msg=f"{label}/{(m, n, k)}/{negative}/{prior_mode}")
          active = np.zeros(c.base.size, dtype=bool)
          active[[i * (2 * n + 3) + 2 * j for i in range(m) for j in range(n)]] = True
          np.testing.assert_array_equal(c.base[~active], physical[~active])
          calls += 1
  assert calls == 90
