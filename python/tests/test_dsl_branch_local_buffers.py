# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Branch-private scratch descriptors retain loop and capture lifetimes."""

import re

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.storage import branch_buffer_scopes


def view():
  return I.View("private", "float32", (I.literal(17),), (I.literal(1),), I.literal(0), owned=True)


def read(storage):
  return I.Expr(I.TileType(storage.dtype, storage.shape), "load", data=storage)


def condition():
  return I.Expr(I.ScalarType("bool"), "local", data="condition")


def use(storage):
  return I.Assign("consumer", read(storage))


def branch(body, otherwise=()):
  return I.If(condition(), I.Block(list(body)), I.Block(list(otherwise)))


def loop(body):
  return I.For("iteration", I.literal(0), I.literal(3), I.literal(1), I.Block(list(body)))


def test_scope_is_common_branch_of_all_storage_occurrences():
  storage = view()
  inner = branch([I.Allocate(storage), use(storage)])
  outer = branch([inner, use(storage)])
  assert branch_buffer_scopes(I.Block([outer])) == {"private": id(outer.body)}


def test_shared_view_identity_outside_branch_prevents_placement():
  storage = view()
  body = branch([I.Allocate(storage), use(storage)])
  assert branch_buffer_scopes(I.Block([body, use(storage)])) == {}


def test_metadata_payload_outside_branch_prevents_placement():
  storage = view()
  body = branch([I.Allocate(storage), use(storage)])
  proof = I.Expr(I.ScalarType("bool"), "view_nonoverlap", data=storage)
  assert branch_buffer_scopes(I.Block([body, I.Assert(proof, "layout")])) == {}


def test_loop_inner_branch_keeps_invocation_capacity_reuse():
  storage = view()
  inner = branch([I.Allocate(storage), use(storage)])
  assert branch_buffer_scopes(I.Block([loop([inner])])) == {}
  outer = branch([loop([inner])])
  assert branch_buffer_scopes(I.Block([outer])) == {"private": id(outer.body)}


@pytest.mark.parametrize("worker", ("grid", "team"))
def test_worker_private_declaration_policy_is_unchanged(worker):
  storage = view()
  body = I.Block([branch([I.Allocate(storage), use(storage)])])
  region = I.Grid(("program",), (I.literal(4),), body) if worker == "grid" else I.Team("team", I.literal(2), body)
  assert branch_buffer_scopes(I.Block([branch([region])])) == {}


def template():
  n = T.dynamic(lower=1)

  @T.template
  def kernel(A: T.Tensor((n,), dtype="float32"), C: T.Tensor((n,), dtype="float32", access="output"), flag: int):
    old = T.load(A)
    if flag > 0:
      scratch = T.alloc((n,), dtype="float32")
      for _ in T.serial(3, unroll=1):
        T.store(scratch, old + 1)
      T.store(C, T.load(scratch))
    else:
      T.store(C, old)
    # An overlapping C write must not make old observe the updated input.
    T.store(C, T.load(C) + old)

  return kernel


@pytest.mark.parametrize("enabled", (False, True))
def test_explicit_buffer_declaration_is_branch_local_but_outside_serial_loop(enabled):
  artifact = CPUBackend().lower(template().ir, (), target="AVX512", options={"cpu_branch_local_buffers": enabled})
  declarations = list(re.finditer(r"^( +)vecops_dsl::Buffer<float> (\w+);", artifact.source, re.MULTILINE))
  explicit = next(item for item in declarations if "scratch" in item.group(2))
  assert len(explicit.group(1)) == (4 if enabled else 2)
  assert explicit.start() < artifact.source.index("for (vecops::nint_t")
  snapshots = [item for item in declarations if "snapshot" in item.group(2)]
  assert snapshots and len(snapshots[0].group(1)) == 2


def test_native_branch_buffers_preserve_history_for_aliasing_inputs(tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(), compiler=compiler)
  for flag in (1, 0, 1):
    storage = np.arange(20, dtype=np.float32)
    a, c = storage[:17], storage[1:18]
    old = a.copy()
    compiled.operator.native(a, c, flag, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, old * 2 + int(flag > 0))
    np.testing.assert_array_equal(storage[18:], [18, 19])


def test_branch_buffer_option_rejects_non_boolean():
  with pytest.raises(TypeError, match="cpu_branch_local_buffers must be bool"):
    CPUBackend().lower(template().ir, (), target="AVX512", options={"cpu_branch_local_buffers": "on"})


@pytest.mark.parametrize("payload", ("dict", "frozenset"))
def test_container_metadata_outside_branch_prevents_placement(payload):
  storage = view()
  body = branch([I.Allocate(storage), use(storage)])
  data = {"owner": storage} if payload == "dict" else frozenset((storage,))
  metadata = I.Expr(I.ScalarType("bool"), "ownership_metadata", data=data)
  assert branch_buffer_scopes(I.Block([body, I.Assign("proof", metadata)])) == {}
