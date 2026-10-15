# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Batch folds retain producer, initializer and cross-batch alias semantics."""

import numpy as np
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.batched_matmul import batched_matmul


@pytest.mark.parametrize("read_output", [False, True])
def test_captured_batch_result_eliminates_redundant_output_store(read_output):
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.passes.kernel.matmul import fuse_dot_epilogue
  from vecops.language._compiler import ir as I

  @T.template
  def kernel(A: T.Tensor((2, 3, 7), dtype="float32"), B: T.Tensor((7, 5), dtype="float32"),
             C: T.Tensor((2, 3, 5), dtype="float32", access="output")):
    value = T.library.batched_matmul((A, B), C, strategy="rows", implementation="vector")
    if read_output:
      T.store(C, value * 2 + T.load(C))
    else:
      T.store(C, value * 2 + 1)

  ir = lower(kernel.ir)
  before = [s for s in ir.body.statements if isinstance(s, I.Store) and s.destination.storage == "C"]
  fuse_dot_epilogue(ir, None)
  after = [s for s in ir.body.statements if isinstance(s, I.Store) and s.destination.storage == "C"]
  assert len(before) == 2
  assert len(after) == (2 if read_output else 1)


def test_external_captured_load_keeps_first_output_store():
  from vecops.language._compiler.passes.kernel.matmul import fuse_dot_epilogue

  @T.template
  def kernel(B: T.Tensor((2, 3, 5), dtype="float32"),
             C: T.Tensor((2, 3, 5), dtype="float32", access="output")):
    value = T.load(B)
    T.store(C, value)
    T.store(C, value * 2 + 1)

  ir = lower(kernel.ir)
  fuse_dot_epilogue(ir, None)
  # External B can alias C at runtime. Only private/workspace captured loads
  # carry the disjoint storage promise used by the batched optimization.
  assert len([s for s in ir.body.statements if isinstance(s, I.Store)]) == 2


def test_fresh_external_epilogue_read_keeps_batch_output_store():
  from vecops.language._compiler.passes.kernel.matmul import fuse_dot_epilogue

  @T.template
  def kernel(A: T.Tensor((2, 3, 7), dtype="float32"), B: T.Tensor((7, 5), dtype="float32"),
             Bias: T.Tensor((2, 3, 5), dtype="float32"),
             C: T.Tensor((2, 3, 5), dtype="float32", access="output")):
    value = T.library.batched_matmul((A, B), C, strategy="rows", implementation="vector")
    T.store(C, value + T.load(Bias))

  ir = lower(kernel.ir)
  fuse_dot_epilogue(ir, None)
  # Bias can alias C, so this fresh read observes the helper's output write.
  assert len([s for s in ir.body.statements if isinstance(s, I.Store) and s.destination.storage == "C"]) == 2


def test_batch_fresh_alias_epilogue_observes_helper_store(tmp_path):
  from dsl_test_utils import make_compiler

  @T.template
  def kernel(A: T.Tensor((2, 3, 4), dtype="float32"), B: T.Tensor((4, 5), dtype="float32"),
             Bias: T.Tensor((2, 3, 5), dtype="float32"),
             C: T.Tensor((2, 3, 5), dtype="float32", access="output")):
    value = T.library.batched_matmul((A, B), C, strategy="rows", implementation="vector")
    T.store(C, value + T.load(Bias))

  compiled = T.compile(kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  a = (np.arange(24).reshape(2, 3, 4) % 7).astype(np.float32)
  b = (np.arange(20).reshape(4, 5) % 5).astype(np.float32)
  product = a @ b
  for alias in (False, True):
    c = np.full((2, 3, 5), 19, dtype=np.float32)
    bias = c if alias else np.full(c.shape, 3, dtype=np.float32)
    compiled(a, b, bias, c)
    np.testing.assert_array_equal(c, product * 2 if alias else product + 3)


def template(shape_a, shape_b, shape_c, strategy, hints, seed=True, return_ir=False):
  @T.template
  def kernel(
    A: T.Tensor(shape_a, tuple(T.dynamic() for _ in shape_a), "float32"),
    B: T.Tensor(shape_b, tuple(T.dynamic() for _ in shape_b), "float32"),
    Prior: T.Tensor((1, 1, shape_c[2]), dtype="float32"),
    C: T.Tensor(shape_c, tuple(T.dynamic() for _ in shape_c), "float32", access="output"),
  ):
    result = batched_matmul(
      (T.load(A) * 2 + 1, T.load(B) - 1),
      C,
      acc=T.load(Prior) if seed else None,
      strategy=strategy,
      raw_inputs=(A, B) if hints else None,
      implementation="vector",
      BM=2,
      BN=3,
      BK=4,
    )
    T.store(C, T.max(result * 2 - 1, 0))

  return kernel.ir if return_ir else lower(kernel.ir)


class BatchInterpreter(Interpreter):
  def __init__(self, program, arguments):
    self.products = []
    super().__init__(program, arguments)

  def value(self, expression):
    if isinstance(expression, I.Expr):
      if expression.op == "dot":
        self.products.append(tuple(tuple(int(self.value(n)) for n in a.type.shape) for a in expression.args[:2]))
      if expression.op == "broadcast_dim":
        a, b = (self.value(v) for v in expression.args)
        if a != b and a != 1 and b != 1:
          raise ValueError("incompatible broadcast")
        return b if a == 1 else a
      if expression.op == "value_transform":
        if expression.data[0] == "broadcast_to":
          return np.broadcast_to(
            self.value(expression.args[0]), tuple(int(self.value(n)) for n in expression.type.shape)
          )
        if expression.data[0] == "reshape":
          return np.reshape(self.value(expression.args[0]), tuple(int(self.value(n)) for n in expression.type.shape))
        if expression.data[0] == "transpose":
          return np.transpose(self.value(expression.args[0]), expression.data[1])
    return super().value(expression)


@pytest.mark.parametrize("mode", ["shared_b", "shared_a", "both", "broadcast_a", "broadcast_b"])
@pytest.mark.parametrize("strategy", ["serial", "auto", "explicit"])
@pytest.mark.parametrize("hints,reverse", [(False, False), (True, False), (True, True)])
def test_batch_strategies_keep_conversion_seed_epilogue_and_strides(mode, strategy, hints, reverse):
  batch, m, n, k = 4, 1 if mode in {"shared_a", "broadcast_a"} else 3, 5, 7
  shape_a = (m, k) if mode == "shared_a" else (1 if mode == "broadcast_a" else batch, m, k)
  shape_b = (k, n) if mode == "shared_b" else (1 if mode == "broadcast_b" else batch, k, n)
  if strategy == "explicit":
    if mode not in {"shared_a", "shared_b", "broadcast_a", "broadcast_b"}:
      pytest.skip("explicit folds require a shared rank-two input")
    strategy = "rows" if mode in {"shared_b", "broadcast_b"} else "columns"
  a = (np.arange(np.prod(shape_a)).reshape(shape_a) % 5 - 2).astype(np.float32)
  b = (np.arange(np.prod(shape_b)).reshape(shape_b) % 3 - 1).astype(np.float32)
  if mode in {"shared_a", "broadcast_a"}:
    b = np.ascontiguousarray(b.transpose(1, 0, 2)).transpose(1, 0, 2)
  if reverse:
    a, b = a[..., ::-1], b[..., ::-1, :]
  prior = np.arange(n, dtype=np.float32).reshape(1, 1, n)
  backing = np.full((batch, m, 2 * n + 3), -111, np.float32)
  c = backing[:, :, 1 : 1 + 2 * n : 2]
  expected = np.maximum(((a.astype(np.int64) * 2 + 1) @ (b.astype(np.int64) - 1) + prior) * 2 - 1, 0)
  program = template(shape_a, shape_b, (batch, m, n), strategy, hints)
  runner = BatchInterpreter(program, {"A": a, "B": b, "Prior": prior, "C": c})
  runner.block(program.body)
  np.testing.assert_array_equal(c, expected.astype(np.float32))
  assert np.all(backing[:, :, 0] == -111) and np.all(backing[:, :, 2::2] == -111)
  expected_count = (
    1
    if strategy in {"rows", "columns"}
    or strategy == "auto"
    and hints
    and not reverse
    and mode in {"shared_a", "shared_b", "broadcast_a", "broadcast_b"}
    else batch
  )
  assert len(runner.products) == expected_count


@pytest.mark.parametrize(
  "shared,strategy", [("B", "serial"), ("B", "rows"), ("B", "auto"), ("A", "serial"), ("A", "columns"), ("A", "auto")]
)
@pytest.mark.parametrize("alias", ["A", "B"])
def test_cross_batch_input_and_broadcast_prior_alias(shared, strategy, alias):
  batch, m, n, k = 4, 1 if shared == "A" else 3, 3, 3
  output = np.arange(batch * m * n, dtype=np.float32).reshape(batch, m, n) % 5
  a_shape = (m, k) if shared == "A" else (batch, m, k)
  b_shape = (k, n) if shared == "B" else (batch, k, n)
  a = output[0] if alias == "A" and shared == "A" else output if alias == "A" else np.ones(a_shape, np.float32)
  if alias == "B":
    b = output[0] if shared == "B" else np.broadcast_to(output, b_shape)
  else:
    b = np.ones(b_shape, np.float32)
  prior = output[:1, :1, :]
  expected = np.maximum(
    ((a.copy().astype(np.int64) * 2 + 1) @ (b.copy().astype(np.int64) - 1) + prior.copy()) * 2 - 1, 0
  )
  program = template(a_shape, b_shape, output.shape, strategy, True)
  runner = BatchInterpreter(program, {"A": a, "B": b, "Prior": prior, "C": output})
  runner.block(program.body)
  np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("batch,m,n,k", [(0, 3, 5, 7), (4, 0, 5, 7), (4, 3, 0, 7), (4, 3, 5, 0)])
@pytest.mark.parametrize("strategy", ["serial", "rows", "auto"])
@pytest.mark.parametrize("seed", [False, True])
def test_batch_empty_domains_and_zero_k(batch, m, n, k, strategy, seed):
  a, b = np.zeros((batch, m, k), np.float32), np.zeros((k, n), np.float32)
  prior = np.arange(n, dtype=np.float32).reshape(1, 1, n)
  output = np.full((batch, m, n), -111, np.float32)
  program = template(a.shape, b.shape, output.shape, strategy, True, seed)
  runner = BatchInterpreter(program, {"A": a, "B": b, "Prior": prior, "C": output})
  runner.block(program.body)
  expected = np.broadcast_to(np.maximum((prior if seed else np.zeros_like(prior)) * 2 - 1, 0), output.shape)
  np.testing.assert_array_equal(output, expected)


def workspace_template(overlap):
  read_workspace = overlap == "producer"

  @T.template
  def kernel(
    A: T.Tensor((4, 3, 3), dtype="float32"),
    B: T.Tensor((3, 3), dtype="float32"),
    Prior: T.Tensor((1, 1, 3), dtype="float32"),
    Scratch: T.Tensor((4, 3, 3), dtype="float32", access="output"),
    C: T.Tensor((4, 3, 3), dtype="float32", access="output"),
  ):
    result = T.library.batched_matmul(
      (T.load(A) * 2 + T.load(Scratch) if read_workspace else T.load(A), B),
      C,
      acc=T.load(Prior),
      workspace=Scratch,
      raw_inputs=(A, B),
    )
    T.store(C, result + 1)

  return lower(kernel.ir)


@pytest.mark.parametrize("overlap", [None, "A", "B", "C", "Prior", "producer"])
def test_explicit_workspace_reuse_checks_all_read_dependencies_before_writes(overlap):
  a = np.arange(36, dtype=np.float32).reshape(4, 3, 3) % 5
  b = np.arange(9, dtype=np.float32).reshape(3, 3) % 3
  prior = np.arange(3, dtype=np.float32).reshape(1, 1, 3)
  scratch = np.full((4, 3, 3), -111, np.float32)
  c = np.full_like(scratch, -222)
  if overlap == "A":
    scratch = a
  elif overlap == "B":
    b = scratch[0]
  elif overlap == "C":
    c = scratch
  elif overlap == "Prior":
    prior = scratch[:1, :1]
  before_scratch, before_c = scratch.copy(), c.copy()
  program = workspace_template(overlap)
  runner = BatchInterpreter(program, {"A": a, "B": b, "Prior": prior, "Scratch": scratch, "C": c})
  if overlap is not None:
    with pytest.raises(AssertionError, match="workspace overlaps"):
      runner.block(program.body)
    np.testing.assert_array_equal(scratch, before_scratch)
    np.testing.assert_array_equal(c, before_c)
  else:
    runner.block(program.body)
    np.testing.assert_array_equal(c, a @ b + prior + 1)
    np.testing.assert_array_equal(scratch, a @ b + prior)
    # Reuse the same capacity with changed payload, proving there is no
    # implicit packed-content cache keyed only on shapes or pointers.
    a[:] = a + 2
    BatchInterpreter(program, {"A": a, "B": b, "Prior": prior, "Scratch": scratch, "C": c}).block(program.body)
    np.testing.assert_array_equal(c, a @ b + prior + 1)


def test_public_batch_workspace_uses_existing_runtime_workspace_authority():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  @T.template
  def kernel(
    A: T.Tensor((4, 3, 7), dtype="float32"),
    B: T.Tensor((7, 5), dtype="float32"),
    C: T.Tensor((4, 3, 5), dtype="float32", access="output"),
  ):
    scratch = T.workspace(C.shape, dtype="float32")
    for repeat in T.serial(2):
      T.library.batched_matmul((A, B), C, raw_inputs=(A, B), workspace=scratch, implementation="vector")

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert "vecops::execution::WorkspaceContext& dsl_workspace" in source
  assert "dsl_workspace.tensor<float>" in source


def test_native_public_batch_workspace_folds_and_replays_changed_payload(tmp_path):
  from dsl_test_utils import make_compiler

  batch, m, n, k = (T.dynamic(lower=0) for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((batch, m, k), (T.dynamic(), T.dynamic(), T.dynamic()), "float32"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "float32"),
    Prior: T.Tensor((1, 1, n), (T.dynamic(), T.dynamic(), T.dynamic()), "float32"),
    C: T.Tensor((batch, m, n), (T.dynamic(), T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    scratch = T.workspace(C.shape, dtype="float32")
    value = T.library.batched_matmul(
      (T.load(A) * 2 + 1, T.load(B) - 1),
      C,
      acc=T.load(Prior),
      raw_inputs=(A, B),
      workspace=scratch,
      implementation="vector",
    )
    T.store(C, T.max(value * 2 - 1, 0))

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for reverse, alias in [(False, None), (True, None), (False, "A"), (True, "B")]:
    c = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3) % 5
    a = c if alias == "A" else np.arange(36, dtype=np.float32).reshape(4, 3, 3) % 5 - 2
    b = c[0] if alias == "B" else np.arange(9, dtype=np.float32).reshape(3, 3) % 3 - 1
    if reverse:
      a, b, c = a[..., ::-1], b[::-1, ::-1], c[..., ::-1]
    prior = c[:1, :1, :]
    expected = np.maximum(
      ((a.copy().astype(np.int64) * 2 + 1) @ (b.copy().astype(np.int64) - 1) + prior.copy()) * 2 - 1, 0
    )
    compiled(a, b, prior, c)
    np.testing.assert_array_equal(c, expected)
  # An unchanged shape must still consume new input payload on replay.
  a, b = np.full((4, 3, 3), 2, np.float32), np.full((3, 3), 3, np.float32)
  prior, c = np.ones((1, 1, 3), np.float32), np.zeros((4, 3, 3), np.float32)
  compiled(a, b, prior, c)
  np.testing.assert_array_equal(c, np.full(c.shape, 61, np.float32))


@pytest.mark.parametrize("shared", ["A", "B"])
def test_shared_batch_auto_strategy_lowers_to_native_vector_source(shared):
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  shape_a = (1, 7) if shared == "A" else (4, 3, 7)
  shape_b = (4, 7, 5) if shared == "A" else (7, 5)
  source = (
    CPUBackend()
    .lower(template(shape_a, shape_b, (4, shape_a[-2], 5), "auto", True, return_ir=True), (), target="AVX512")
    .source
  )
  assert "__kernel" in source


def parallel_template(shared="B", *, packed=False, native=False):
  batch, m, n, k = 8, 16, 16, 65 if native else 64
  shape_a = (m, k) if shared == "A" else (batch, m, k)
  shape_b = (k, n) if shared == "B" else (batch, k, n)

  @T.template
  def kernel(
    A: T.Tensor(shape_a, tuple(T.dynamic() for _ in shape_a), "float32"),
    B: T.Tensor(shape_b, tuple(T.dynamic() for _ in shape_b), "float32"),
    Prior: T.Tensor((1, 1, n), (T.dynamic(), T.dynamic(), T.dynamic()), "float32"),
    C: T.Tensor((batch, m, n), (T.dynamic(), T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    value = T.library.batched_matmul(
      (T.load(A) * 2 + 1, T.load(B) - 1),
      C,
      acc=T.load(Prior),
      parallel="require",
      shared_packing=packed,
      implementation="vector",
    )
    T.store(C, value * 2 + 1)

  return kernel


@pytest.mark.parametrize("shared", ["A", "B"])
@pytest.mark.parametrize("packed", [False, True])
def test_explicit_parallel_grid_and_shared_pack_have_proven_ownership(shared, packed):
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._compiler.ir.traversal import nodes

  kernel = parallel_template(shared, packed=packed)
  program = lower(kernel.ir)
  assert sum(isinstance(s, I.Grid) for s in nodes(program.body)) == 1
  packs = [s for s in program.body.statements if isinstance(s, I.Pack)]
  assert len(packs) == int(packed)
  assert all(s.destination.side == shared for s in packs)
  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert "verified grid ownership" in source
  assert "grid_lanes" in source


def test_native_explicit_parallel_batch_uses_two_and_four_workers_and_shared_pack(tmp_path):
  import vecops
  from dsl_test_utils import make_compiler
  from vecops.workspace_observer import drain_memory, tracking

  compiled = T.compile(
    parallel_template(packed=True),
    compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
    debug=T.DebugOptions(memory=True, timeline=True),
  )
  a = np.arange(8 * 16 * 64, dtype=np.float32).reshape(8, 16, 64) % 5 - 2
  b = np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 3 - 1
  prior = np.arange(16, dtype=np.float32).reshape(1, 1, 16)
  c = np.empty((8, 16, 16), np.float32)
  expected = ((a.astype(np.int64) * 2 + 1) @ (b.astype(np.int64) - 1) + prior) * 2 + 1
  for workers in (2, 4):
    with vecops.cpu_thread_limit(workers), tracking():
      drain_memory()
      compiled(a, b, prior, c)
      events = drain_memory()
    np.testing.assert_array_equal(c, expected)
    active = {e["worker"] for e in events if e["site"] == "grid-program" and e["action"] == 2}
    assert len(active) == workers, events
    for reverse, alias in [(True, None), (False, "A"), (True, "B")]:
      backing = np.full((8, 16, 64), -111, np.float32)
      target = backing[:, :, :16]
      left = backing if alias == "A" else np.arange(8 * 16 * 64, dtype=np.float32).reshape(8, 16, 64) % 5 - 2
      right = (
        backing.reshape(-1, 64)[:64, :16]
        if alias == "B"
        else np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 3 - 1
      )
      left[:] = np.arange(left.size).reshape(left.shape) % 5 - 2
      right[:] = np.arange(right.size).reshape(right.shape) % 3 - 1
      if reverse:
        left, right, target = left[..., ::-1], right[::-1, ::-1], target[..., ::-1]
      initial = target[:1, :1, :]
      initial[:] = np.arange(16) % 4
      expected_alias = (
        (left.copy().astype(np.int64) * 2 + 1) @ (right.copy().astype(np.int64) - 1) + initial.copy()
      ) * 2 + 1
      before = backing.copy()
      outside = np.ones(backing.shape, bool)
      outside[:, :, :16] = False
      with vecops.cpu_thread_limit(workers):
        compiled(left, right, initial, target)
      np.testing.assert_array_equal(target, expected_alias)
      np.testing.assert_array_equal(backing[outside], before[outside])
