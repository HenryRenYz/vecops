# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Unit-axis reshapes preserve captured reads and native vector access."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.kernel.transforms import fold_value_transforms
from vecops.language._compiler.passes.manager import PassContext


@pytest.mark.parametrize(
  "before,after",
  (
    ((1, 1, 17), (1, 17)),
    ((2, 1, 3), (1, 2, 3, 1)),
    ((1, 0, 3), (0, 1, 3)),
    ((1, 1), (1,)),
  ),
)
def test_unit_axis_canonicalization_keeps_the_original_historical_read(before, after):
  shape = tuple(I.literal(x) for x in before)
  view = I.View("A", "float32", shape, tuple(I.literal(1) for _ in shape), I.literal(0))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  target = tuple(I.literal(x) for x in after)
  reshaped = I.Expr(I.TileType("float32", target), "value_transform", (load,), ("reshape", ()))
  output = I.View("C", "float32", target, tuple(I.literal(1) for _ in target), I.literal(0))
  kernel = I.KernelIR(I.Block([I.BindTile(load), I.Store(output, reshaped)]), {}, {})
  assert fold_value_transforms(kernel, PassContext("kernel")).changed
  result = kernel.body.statements[-1].value
  assert tuple(I.loads_unique(result)) == (load,)
  assert not any(n.op == "value_transform" and n.data[0] == "reshape" for n in nodes(result) if type(n) is I.Expr)


def test_equal_looking_distinct_read_dimensions_are_not_equated():
  scalar = I.View("metadata", "int64", (), (), I.literal(0))
  first = I.Expr(I.ScalarType("int64"), "load", data=scalar)
  second = I.Expr(first.type, "load", data=scalar)
  shape = (I.literal(1), first)
  view = I.View("A", "float32", shape, (first, I.literal(1)), I.literal(0))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  reshape = I.Expr(I.TileType("float32", (second,)), "value_transform", (load,), ("reshape", ()))
  output = I.View("C", "float32", (second,), (I.literal(1),), I.literal(0))
  kernel = I.KernelIR(I.Block([I.BindTile(load), I.Store(output, reshape)]), {}, {})
  fold_value_transforms(kernel, PassContext("kernel"))
  assert kernel.body.statements[-1].value.op == "value_transform"
  assert kernel.body.statements[-1].value.data[0] == "reshape"


@pytest.mark.parametrize("strided", (False, True))
def test_native_singleton_reshape_preserves_alias_history_and_conversion(tmp_path, strided):
  N = T.dynamic(lower=0, upper=257)
  stride = 2 if strided else 1

  @T.template
  def program(
    A: T.Tensor((1, 1, N), (257 * stride, 257 * stride, stride), dtype="float32"),
    C: T.Tensor((1, N), dtype="float32", access="output"),
    D: T.Tensor((1, N), dtype="float32", access="output"),
  ):
    old = T.load(A)
    flat = T.reshape(old, (1, N))
    T.store(C, flat + 10)
    T.store(D, T.cast(T.cast(flat, "float16"), "float32"))

  compiler = make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build", jobs=2)
  compiled = T.compile(program, compiler=compiler)
  assert "reshape_lane" not in compiled.generated_source
  for width in (0, 1, 17, 257):
    for shift in (0, 1):
      for poison in (90, 195):
        owner = np.empty((257 * stride + 50) * 4, dtype=np.uint8)
        owner.fill(poison)
        a = np.ndarray(
          (1, 1, width),
          dtype=np.float32,
          buffer=owner,
          offset=16 * 4,
          strides=(257 * stride * 4, 257 * stride * 4, stride * 4),
        )
        c = np.ndarray((1, width), dtype=np.float32, buffer=owner, offset=(16 + shift) * 4, strides=(width * 4, 4))
        d_owner = np.full((width + 16) * 4, poison, dtype=np.uint8)
        d = np.ndarray((1, width), dtype=np.float32, buffer=d_owner, offset=8 * 4, strides=(width * 4, 4))
        d.fill(np.nan)
        a[:] = np.linspace(-1, 1, width, dtype=np.float32)
        original = a.copy().reshape(1, width)
        before = owner.copy()
        d_before = d_owner.copy()
        compiled(a, c, d)
        np.testing.assert_array_equal(c, original + 10)
        np.testing.assert_array_equal(d, original.astype(np.float16).astype(np.float32))
        mask = np.zeros(owner.size, dtype=bool)
        mask[(16 + shift) * 4 : (16 + shift) * 4 + c.nbytes] = True
        np.testing.assert_array_equal(owner[~mask], before[~mask])
        np.testing.assert_array_equal(d_owner[: 8 * 4], d_before[: 8 * 4])
        np.testing.assert_array_equal(d_owner[8 * 4 + d.nbytes :], d_before[8 * 4 + d.nbytes :])
