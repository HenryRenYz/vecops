# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Entry geometry proofs and native first-use/iteration snapshot semantics."""

import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.backends.cpu.invariants import EntryGeometry
from vecops.language._compiler.backends.cpu.verification import verify_prepared
from vecops.language._compiler.environment import freeze


def geometry():
  catalog = freeze(target="Scalar").operations
  raw = I.Expr(I.ScalarType("int64"), "raw", data="A.size(0)")
  kernel = I.KernelIR(I.Block(), {"A": T.Tensor((T.dynamic(lower=0),), dtype="float32"), "limit": T.int64}, {"N": raw})
  shape = (I.Expr(I.ScalarType("int64"), "symbol", data="N"),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  return kernel, catalog, EntryGeometry(kernel, catalog), view


def test_only_abi_geometry_and_pure_integer_derivations_are_invariant():
  _, _, plan, view = geometry()
  assert plan.view(view)
  assert plan.scalar(I.binary("add", view.shape[0], I.literal(3)))
  for value in (
    I.Expr(I.ScalarType("int64"), "raw", data="limit"),
    I.Expr(I.ScalarType("int64"), "raw", data="arbitrary_cpp()"),
    I.Expr(I.ScalarType("int64"), "local", data="iteration"),
    I.Expr(I.ScalarType("int64"), "frozen", data="captured_iteration"),
    I.Expr(I.ScalarType("int64"), "load", data=view),
    I.literal(1.5),
  ):
    assert not plan.scalar(value)
  assert not plan.view(replace(view, storage="scratch", owned=True))
  assert not plan.view(replace(view, owned=True))
  assert not plan.view(replace(view, storage="limit"))


def test_unknown_metadata_is_not_executed_and_cycles_remain_ineligible():
  _, _, plan, view = geometry()

  class HiddenExpr(I.Expr):
    def __getattribute__(self, name):
      raise AssertionError("unknown expression was inspected")

  class HiddenView(I.View):
    def __getattribute__(self, name):
      raise AssertionError("unknown view was inspected")

  assert not plan.scalar(HiddenExpr(I.ScalarType("int64"), "literal", data=1))
  assert not plan.view(HiddenView(view.storage, view.dtype, view.shape, view.strides, view.offset))
  cycle = I.Expr(I.ScalarType("int64"), "add", ())
  object.__setattr__(cycle, "args", (cycle, I.literal(1)))
  assert not plan.scalar(cycle)
  value = I.literal(41)
  reference = weakref.ref(value)
  assert plan.scalar(value)
  del value
  assert reference() is not None


def emitter():
  kernel, catalog, plan, view = geometry()
  result = MemoryEmitter()
  result.ir = kernel
  result.prepared = SimpleNamespace(entry_geometry=plan)
  result.environment = SimpleNamespace(operations=catalog, codegen_target=SimpleNamespace(architecture="x86"))
  result.serial_guard_depth = 1
  result.parallel_depth = 0
  result.hardware_scope = None
  result.hardware_scalar_bindings = {}
  result.active_scalar_aliases = {}
  result.invariant_alias_predicates = {}
  result.unique = lambda prefix: f"{prefix}_{len(result.invariant_alias_predicates)}"
  return result, view


def test_replaced_bindings_cannot_reuse_an_old_or_already_cached_proof():
  result, view = emitter()
  assert "_ready" in result.invariant_alias_predicate(view, view, "predicate")
  result.ir.bindings["N"] = I.Expr(I.ScalarType("int64"), "local", data="iteration")
  assert not result.prepared.entry_geometry.matches(result.ir, result.environment.operations)
  assert result.invariant_alias_predicate(view, view, "predicate") == "predicate"


@pytest.mark.parametrize("boundary", ("outside", "worker", "hardware", "hardware_capture", "conditional_geometry", "arm"))
def test_owner_and_conditional_geometry_boundaries_do_not_share_state(boundary):
  result, view = emitter()
  if boundary == "outside":
    result.serial_guard_depth = 0
  elif boundary == "worker":
    result.parallel_depth = 1
  elif boundary == "hardware":
    result.hardware_scope = ("amx", "owner")
  elif boundary == "hardware_capture":
    result.hardware_scalar_bindings = {("raw", "A.size(0)"): "owned"}
  elif boundary == "conditional_geometry":
    result.active_scalar_aliases = {0: (view.shape[0], I.literal(1))}
  else:
    result.environment.codegen_target.architecture = "arm"
  assert result.invariant_alias_predicate(view, view, "predicate") == "predicate"
  assert not result.invariant_alias_predicates


def test_many_constant_subviews_have_a_bounded_invocation_state():
  result, view = emitter()
  for index in range(64):
    result.invariant_alias_predicate(replace(view, offset=I.literal(index)), view, f"predicate_{index}")
  assert len(result.invariant_alias_predicates) == 16
  assert "_ready" in result.invariant_alias_predicate(view, view, "predicate_0")
  assert result.invariant_alias_predicate(view, view, "predicate_63") == "predicate_63"


def test_memory_stage_constructs_and_verifies_the_actual_entry_proof():
  N = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((N,), dtype="float32"), O: T.Tensor((N,), dtype="float32", access="output")):
    for iteration in T.serial(2, unroll=1):
      T.store(O, T.load(A) + 1)

  prepared = CPUBackend().lower(kernel.ir, (), target="Scalar").backend_ir
  trace = prepared.pass_trace
  assert trace.count("cpu-plan-entry-geometry") == 1
  assert trace.index("cpu-plan-private-captures") < trace.index("cpu-plan-entry-geometry") < trace.index("verify-cpu-plan")
  with pytest.raises(ValueError, match="unknown entry geometry proof"):
    verify_prepared(replace(prepared, entry_geometry=object()))
  original = prepared.vectors.kernel
  different = replace(original)
  catalog = prepared.vectors.manager.options["_compiler_environment"].operations
  with pytest.raises(ValueError, match="stale entry geometry proof"):
    verify_prepared(replace(prepared, entry_geometry=EntryGeometry(different, catalog)))


@pytest.fixture(scope="module")
def native_predicate_kernels(tmp_path_factory):
  directory = tmp_path_factory.mktemp("invariant-predicates")
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  kernels = {}

  def template(kind):
    M, N, S = T.dynamic(lower=0), T.dynamic(lower=0), T.dynamic(lower=0)

    @T.template
    def kernel(
      A: T.Tensor((M, N), (S, 1), dtype="float32"),
      O: T.Tensor((M, N), dtype="float32", access="output"),
      repeat: T.int64,
      enabled: T.int64,
    ):
      if kind == "for":
        for iteration in T.serial(repeat, unroll=1):
          if enabled:
            old = T.load(A)
            T.store(O, old + 1)
          T.compiler_fence()
      else:
        iteration = 0
        while iteration < repeat:
          if enabled:
            old = T.load(A)
            T.store(O, old + 1)
          T.compiler_fence()
          iteration += 1

    return kernel

  with vecops.collect_compile_requests() as batch:
    for kind in ("for", "while"):
      compiled = T.compile(template(kind), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((4, 1), strides=(1, 1), dtype="float32"),
        vecops.TensorMeta((4, 1), dtype="float32"), 1, 1, **compiled._core.specialization_values,
      )
      if freeze(target=compiler.target).codegen_target.architecture == "x86":
        assert "_ready = false" in compiled.source.read_text()
      else:
        assert "_ready = false" not in compiled.source.read_text()
      kernels[kind] = compiled
  vecops.compile_batch(batch.requests, parallelism=2)
  return kernels


@pytest.mark.parametrize("kind", ("for", "while"))
@pytest.mark.parametrize("repeat,enabled,raises", ((0, 1, False), (3, 0, False), (1, 1, True)))
def test_span_overflow_occurs_only_at_the_first_original_use(native_predicate_kernels, kind, repeat, enabled, raises):
  compiled = native_predicate_kernels[kind]
  backing = np.zeros(1, dtype=np.float32)
  # No load from this view is legal. Its original guard must reject the byte
  # span overflow, but an empty loop or untaken branch must never evaluate it.
  source = np.lib.stride_tricks.as_strided(backing, shape=(4, 1), strides=(1 << 62, 4))
  output = np.full((4, 1), -193, dtype=np.float32)
  if raises:
    with pytest.raises((RuntimeError, OverflowError), match="span"):
      compiled.operator.native(source, output, repeat, enabled, **compiled._core.specialization_values)
  else:
    compiled.operator.native(source, output, repeat, enabled, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, np.full((4, 1), -193, dtype=np.float32))


@pytest.mark.parametrize("kind", ("for", "while"))
def test_contents_refresh_each_iteration_and_each_invocation_has_its_own_predicates(native_predicate_kernels, kind):
  compiled = native_predicate_kernels[kind]
  for shift in (8, 1, -1, 0, 8, 1):
    owner = np.arange(24, dtype=np.float32) / 4
    source = owner[5:9].reshape(4, 1)
    output = owner[5 + shift : 9 + shift].reshape(4, 1)
    expected = owner.copy()
    for _ in range(3):
      prior = expected[5:9].copy()
      expected[5 + shift : 9 + shift] = prior + 1
    compiled.operator.native(source, output, 3, 1, **compiled._core.specialization_values)
    np.testing.assert_array_equal(owner, expected)
