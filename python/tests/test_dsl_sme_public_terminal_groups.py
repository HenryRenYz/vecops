# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Public group padding must retain exact bounds across instruction K tails."""

import os
from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.hardware import preflight
from vecops.language._compiler.dtypes import ELEMENT_BYTES
from vecops.language._compiler.frontend.helpers import expand_helper
from vecops.language._compiler.operations import snapshot
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.packing import _pack_sme_view


@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "uint8"])
def test_public_terminal_group_preflight_preserves_correlated_bounds(dtype):
  size = ELEMENT_BYTES[dtype]
  group, span, chunk = 4 // size, 32, 64 // size
  for m in (0, 2, 4, 33, 65):
    for k in (0, chunk - 1, chunk + 1, 2 * chunk - 1, 2 * chunk + 1):
      groups = (k + group - 1) // group
      dense = I.View("input", dtype, tuple(map(I.literal, (m, k))), tuple(map(I.literal, (k, 1))), I.literal(0))
      shape = ((m + span - 1) // span, groups, span * group)
      panels = I.View("output", dtype, tuple(map(I.literal, shape)),
                      tuple(map(I.literal, (groups * span * group, span * group, 1))), I.literal(0))
      def rooted(view):
        rank = len(view.shape)
        return replace(view, region=I.ViewRegion(view.shape, view.strides,
          (I.literal(0),) * rank, view.shape, (1,) * rank, tuple(range(rank)), (False,) * rank))
      dense, panels = rooted(dense), rooted(panels)
      _, body = expand_helper(_pack_sme_view, (dense, panels, "uint32", group), catalog=snapshot(), prefix="public_tail")
      reference = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
      scalar = reference.value
      reference.value = lambda value, scalar=scalar: 64 // ELEMENT_BYTES[value.data] if isinstance(value, I.Expr) and value.op == "streaming_lanes" else scalar(value)
      for statement in body:
        if isinstance(statement, I.Assign):
          reference.environment[statement.target] = reference.value(statement.value)
        elif isinstance(statement, I.HardwareRegion):
          for condition, message in preflight(statement).checks:
            assert reference.value(condition), (dtype, m, k, message)
        else:
          raise TypeError(type(statement))


def packing_template(dtype, side):
  m, k, capacity = (T.dynamic(lower=0) for _ in range(3))
  ss, ks = T.dynamic(), T.dynamic()
  shape, strides = ((m, k), (ss, ks)) if side == "A" else ((k, m), (ks, ss))

  @T.template
  def kernel(X: T.Tensor(shape, strides, dtype), P: T.Tensor((capacity,), dtype=dtype, access="output")):
    T.pack(X, P, side=side, format="sme")

  return kernel


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native SME")
@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "uint8"])
@pytest.mark.parametrize("side", ["A", "B"])
def test_native_public_terminal_groups_raw_capacity_alias(dtype, side, tmp_path):
  pytest.importorskip("ml_dtypes")
  from vecops._arm_vector_state import get_vector_state
  dt = np.dtype(dtype)
  bits = np.dtype("uint16" if dt.itemsize == 2 else "uint8")
  span, group, chunk = get_vector_state().sme_bytes // 2, 4 // dt.itemsize, get_vector_state().sme_bytes // dt.itemsize
  compiled = T.compile(packing_template(dtype, side), compiler=make_compiler(target="Native", jobs=2,
    cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for m in (2, span + 1):
    for k in (0, chunk - 1, chunk + 1, 2 * chunk - 1, 2 * chunk + 1):
      original = (np.arange(m * k, dtype=bits) * 13 + 7).reshape(m, k)
      expected = np.zeros(((m + span - 1) // span, (k + group - 1) // group, span, group), dtype=bits)
      for row in range(m):
        for col in range(k):
          expected[row // span, col // group, row % span, col % group] = original[row, col]
      for alias in (False, True):
        memory = np.full(max(m * (k + 3), expected.size) + 5, 19, dtype=bits)
        source = np.ndarray((m, k), dtype=dt, buffer=memory, strides=((k + 3) * dt.itemsize, dt.itemsize))
        source.view(bits)[:] = original
        output_bits = memory if alias else np.full(expected.size + 5, 19, dtype=bits)
        output = output_bits[:expected.size + 5].view(dt)
        untouched = output_bits[expected.size:].copy()
        compiled.operator.native(source if side == "A" else source.T, output, **compiled._core.specialization_values)
        np.testing.assert_array_equal(output_bits[:expected.size], expected.reshape(-1))
        np.testing.assert_array_equal(output_bits[expected.size:], untouched)
