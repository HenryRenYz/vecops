# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Capture-preserving vector reshape and entry-extent proof boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.value_projections import EntryExtents, collect_reshape_suffixes
from vecops.language._compiler.backends.cpu.verification import verify_prepared
from vecops.language._compiler.environment import freeze


def scalar(op, data=None, args=()):
  return I.Expr(I.ScalarType("int64"), op, args, data)


def geometry(statements=()):
  raw = scalar("raw", "A.size(2)")
  kernel = I.KernelIR(
    I.Block(list(statements)), {"A": T.Tensor((2, 3, T.dynamic(lower=0)), dtype="float32")}, {"K": raw}
  )
  return kernel, freeze(target="Scalar").operations


def test_entry_extent_aliases_compare_without_mutating_or_evaluating_ir():
  symbol = scalar("symbol", "K")
  definition = I.binary("add", symbol, I.literal(3))
  kernel, catalog = geometry((I.Assign("saved", definition, declare=True),))
  plan = EntryExtents(kernel, catalog)
  capture = scalar("frozen", "saved")
  assert plan.key(capture) == plan.key(definition)
  assert plan.key(symbol) == plan.key(kernel.bindings["K"])
  assert kernel.body.statements[0].value is definition
  assert capture.op == "frozen"


@pytest.mark.parametrize("kind", ("mutable", "iteration", "opaque", "memory", "unbound", "cycle", "typed"))
def test_unsafe_or_differently_typed_geometry_does_not_become_an_entry_extent(kind):
  symbol = scalar("symbol", "K")
  statements = [I.Assign("saved", symbol, declare=True)]
  if kind == "mutable":
    statements.append(I.Assign("saved", I.literal(7)))
  elif kind == "iteration":
    statements[0] = I.Assign("saved", scalar("local", "i"), declare=True)
  elif kind == "opaque":
    statements[0] = I.Assign("saved", scalar("raw", "callback()"), declare=True)
  elif kind == "memory":
    statements[0] = I.Assign("saved", scalar("load"), declare=True)
  elif kind == "unbound":
    statements.clear()
  elif kind == "cycle":
    statements[0] = I.Assign("saved", scalar("frozen", "saved"), declare=True)
  else:
    statements[0] = I.Assign("saved", I.Expr(I.ScalarType("int32"), "literal", data=3), declare=True)
  kernel, catalog = geometry(statements)
  plan = EntryExtents(kernel, catalog)
  assert plan.key(scalar("frozen", "saved")) is None
  assert plan.key(symbol) is not None


def test_unknown_geometry_nodes_and_recursive_operands_remain_ineligible():
  kernel, catalog = geometry()
  plan = EntryExtents(kernel, catalog)

  class Hidden(I.Expr):
    def __getattribute__(self, name):
      raise AssertionError("unknown geometry inspected")

  assert plan.key(Hidden(I.ScalarType("int64"), "literal", data=1)) is None
  cyclic = scalar("add", args=(I.literal(1),))
  object.__setattr__(cyclic, "args", (cyclic, I.literal(1)))
  assert plan.key(cyclic) is None
  assert plan.key(I.literal(2)) is not None


def projected_template(kind):
  batch, rows, columns = T.dynamic(lower=0), T.dynamic(lower=0), T.dynamic(lower=0)
  s0, s1, s2 = T.dynamic(), T.dynamic(), T.dynamic()
  o0, o1 = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((batch, rows, columns), (s0, s1, s2), "float32", access="inout"),
    O: T.Tensor((batch * rows, columns), (o0, o1), dtype="float32", access="output"),
  ):
    original = T.load(A)
    if kind == "mask":
      value = original > 0
    else:
      value = original * 2 + 1
    # Force historical reads: a fresh read through a rewritten memory view
    # would observe -7, including when O shares the original owner's storage.
    T.store(A, -7)
    result = T.reshape(value, (batch * rows, columns))
    if kind == "mask":
      T.store(O, T.where(result, 1.0, -1.0))
    else:
      T.store(O, result)

  return kernel


def test_projection_is_a_verified_memory_plan_and_keeps_original_subjects():
  artifact = CPUBackend().lower(projected_template("value").ir, (), target="Native")
  prepared = artifact.backend_ir
  plans = prepared.reshape_suffixes
  assert plans
  assert all(proof.subject.args[0] is not None and proof.dimensions >= 1 for proof in plans.values())
  trace = prepared.pass_trace
  assert (
    trace.index("cpu-plan-entry-geometry") < trace.index("cpu-plan-value-projections") < trace.index("verify-cpu-plan")
  )
  identity, proof = next(iter(plans.items()))
  with pytest.raises(ValueError, match="invalid reshape suffix dimensions"):
    verify_prepared(replace(prepared, reshape_suffixes={identity: replace(proof, dimensions=proof.dimensions + 1)}))
  with pytest.raises(ValueError, match="invalid reshape suffix subject"):
    verify_prepared(replace(prepared, reshape_suffixes={identity: object()}))
  with pytest.raises(ValueError, match="stale reshape suffix subject"):
    other = replace(proof.subject)
    verify_prepared(replace(prepared, reshape_suffixes={id(other): replace(proof, subject=other)}))


def test_no_projection_certificate_for_physical_vectors_or_changed_suffix():
  kernel, catalog = geometry()
  source = I.Expr(I.TileType("float32", (I.literal(2), I.literal(3), I.literal(5))), "fill", (I.literal(1.0),))
  for typ in (
    I.TileType("float32", (I.literal(5), I.literal(6))),
    I.VectorType("float32", (I.literal(6), I.literal(5)), tag="physical"),
  ):
    value = I.Expr(typ, "value_transform", (source,), ("reshape", ()))
    candidate = replace(kernel, body=I.Block([I.Assign("v", value, declare=True)]))
    assert not collect_reshape_suffixes(candidate, catalog)


@pytest.fixture(scope="module")
def native_reshapes(tmp_path_factory):
  directory = tmp_path_factory.mktemp("reshape-suffix")
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  return {kind: T.compile(projected_template(kind), compiler=compiler) for kind in ("value", "mask")}


@pytest.mark.parametrize("kind", ("value", "mask"))
def test_native_suffix_history_strides_tails_empty_and_output_alias(native_reshapes, kind):
  compiled = native_reshapes[kind]
  for batch, rows, columns in ((2, 3, 1), (2, 3, 17), (1, 1, 65), (0, 3, 17), (2, 0, 17), (2, 3, 0)):
    for reverse in (False, True):
      for alias in (False, True):
        owner = np.linspace(-3, 4, max(1, batch * rows * (columns + 3)), dtype=np.float32)
        a = owner[: batch * rows * (columns + 3)].reshape(batch, rows, columns + 3)[..., :columns]
        if reverse:
          a = a[::-1, :, ::-1]
        original = a.copy()
        expected = np.where(original > 0, 1, -1).astype(np.float32) if kind == "mask" else original * 2 + 1
        # A padded/negative-stride input still has a logically dense output
        # alias only in the empty case; otherwise select a dense owner view.
        output = (
          owner[: batch * rows * columns].reshape(batch * rows, columns)
          if alias
          else np.full((batch * rows, columns), -91, np.float32)
        )
        before = owner.copy()
        compiled.operator.native(a, output, **compiled._core.specialization_values)
        np.testing.assert_array_equal(output, expected.reshape(batch * rows, columns))
        if not alias:
          np.testing.assert_array_equal(a, -7)
        touched = np.zeros(owner.size, dtype=bool)
        if a.size:
          base = owner.ctypes.data
          for index in np.ndindex(a.shape):
            touched[(a.ctypes.data - base + sum(i * s for i, s in zip(index, a.strides))) // 4] = True
        if alias:
          touched[: output.size] = True
        np.testing.assert_array_equal(owner[~touched], before[~touched])
  if freeze(target="Native").codegen_target.architecture == "x86":
    assert "preserved reshape suffix" in compiled.generated_source
    assert "reshape_lane" not in compiled.generated_source


def test_vector_axis_crossing_the_changed_prefix_keeps_scalar_legalization():
  batch, rows, columns = T.dynamic(lower=0), T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((batch, rows, columns), dtype="float32", access="inout"),
    O: T.Tensor((columns, batch * rows), dtype="float32", access="output"),
  ):
    old = T.load(A)
    T.store(A, -7)
    T.store(O, T.transpose(T.reshape(old, (batch * rows, columns)), (1, 0)))

  artifact = CPUBackend().lower(kernel.ir, (), target="Native")
  assert artifact.backend_ir.reshape_suffixes
  assert "reshape_lane" in artifact.source


@pytest.mark.parametrize("binding", ("loop", "grid", "collective"))
def test_coordinate_and_collective_bindings_cannot_reuse_an_entry_alias(binding):
  statements = [I.Assign("saved", scalar("symbol", "K"), declare=True)]
  if binding == "loop":
    statements.append(I.For("saved", I.literal(0), I.literal(3), I.literal(1), I.Block()))
  elif binding == "grid":
    statements.append(I.Grid(("saved",), (I.literal(3),), I.Block()))
  else:
    statements.append(I.TeamReduce("team", "saved", I.literal(1), "sum"))
  kernel, catalog = geometry(statements)
  assert EntryExtents(kernel, catalog).key(scalar("frozen", "saved")) is None


def test_native_split_prefix_keeps_negative_strides_and_historical_values(tmp_path):
  batch, rows, columns = T.dynamic(lower=0), T.dynamic(lower=0), T.dynamic(lower=0)
  s0, s1, o0, o1, o2 = (T.dynamic() for _ in range(5))

  @T.template
  def kernel(
    A: T.Tensor((batch * rows, columns), (s0, s1), "float32", access="inout"),
    O: T.Tensor((batch, rows, columns), (o0, o1, o2), "float32", access="output"),
  ):
    original = T.load(A) + 3
    T.store(A, -7)
    T.store(O, T.reshape(original, (batch, rows, columns)))

  compiler = make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  for shape in ((2, 3, 17), (2, 3, 0), (0, 3, 17), (2, 0, 17)):
    b, m, k = shape
    owner = np.arange(max(1, b * m * (k + 3)), dtype=np.float32)
    a = owner[: b * m * (k + 3)].reshape(b * m, k + 3)[:, :k][::-1, ::-1]
    original = a.copy()
    output = np.full(shape, -91, dtype=np.float32)
    compiled.operator.native(a, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, (original + 3).reshape(shape))
    np.testing.assert_array_equal(a, -7)
  if freeze(target="Native").codegen_target.architecture == "x86":
    assert "preserved reshape suffix" in compiled.generated_source
    assert "reshape_lane" not in compiled.generated_source
