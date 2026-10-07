"""Structured tile state, value snapshots, branch joins and helper exits."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-state")
  return make_compiler(target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("shape", [(37,), (3, 17)])
def test_register_recurrence_and_snapshot(compiler, shape):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor(shape, dtype=T.float32), C: T.Tensor(shape, dtype=T.float32), *, count: int = 3):
      x = T.load(A)
      for i in T.serial(count):
        old = x
        x = x + 1
        x = x + old * 0.25
      T.store(C, x)

    return kernel

  kernel = factory()
  a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16
  for count in (0, 1, 3):
    expected = a.copy()
    for _ in range(count):
      expected = expected * 1.25 + 1
    np.testing.assert_allclose(kernel(a, count=count), expected)
  source = kernel.generated_source
  assert re.search(r"dsl_f_state_\d+_r0", source)
  assert not re.search(r"vecops_dsl::Buffer<[^>]+> dsl_f_state_", source)
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", source)


def test_while_break_continue_and_branch_state(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((17,), dtype=T.float32), C: T.Tensor((17,), dtype=T.float32), *, count: int = 4):
      x = T.load(A)
      i = 0
      while i < count:
        i += 1
        if i == 2:
          continue
        if i % 2 == 0:
          x = x * 2
        else:
          x = x + 1
        if i == 4:
          break
      T.store(C, x)

    return kernel

  a = np.arange(17, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), (a + 2) * 2)


def test_branch_new_values_and_helper_early_return(compiler):
  @T.func
  def helper(x, choice):
    if choice < 0:
      return x * -2
    if choice == 0:
      return x + 3
    return x * 4

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((19,), dtype=T.float32), C: T.Tensor((19,), dtype=T.float32), *, choice: int = 0):
      if choice > 0:
        scale = 2
        y = T.load(A) + 1
      else:
        scale = 3
        y = T.load(A) - 1
      T.store(C, helper(y, choice) * scale)

    return kernel

  a, kernel = np.arange(19, dtype=np.float32), factory()
  for choice in (-1, 0, 1):
    y, scale = (a + 1, 2) if choice > 0 else (a - 1, 3)
    expected = y * -2 if choice < 0 else y + 3 if choice == 0 else y * 4
    np.testing.assert_array_equal(kernel(a, choice=choice), expected * scale)
  assert "VECOPS_INLINE_LAMBDA" not in kernel.generated_source.split("void __kernel__", 1)[1]


def test_tuple_simultaneous_state_swap(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32)):
      x = T.load(A)
      y = T.load(A) + 10
      for i in T.serial(3):
        x, y = y, x
      T.store(C, x * 2 + y)

    return kernel

  a = np.arange(7, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), (a + 10) * 2 + a)


def tuple_recurrence_template():
  @T.template
  def kernel(A: T.Tensor((7,), dtype="float32"), C: T.Tensor((7,), dtype="float32", access="output"), count: int = 3):
    values = (T.load(A), (T.load(A) + 10, 0))
    for i in T.serial(count):
      values = (values[1][0], (values[0] + values[1][1], values[1][1] + 1))
    T.store(C, values[0] * 2 + values[1][0])

  return kernel


def test_nested_tuple_carried_state_reference_and_codegen():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._debug.interpreter import Interpreter

  artifact = CPUBackend().lower(tuple_recurrence_template().ir, (), target="AVX512")
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", artifact.source)
  for count in (0, 1, 3, 4):
    a, c = np.arange(7, dtype=np.float32), np.empty(7, dtype=np.float32)
    Interpreter(artifact.lowered.kernel, {"A": a, "C": c, "count": count}).block(artifact.lowered.kernel.body)
    x, y, scalar = a.copy(), a + 10, 0
    for _ in range(count):
      x, y, scalar = y, x + scalar, scalar + 1
    np.testing.assert_array_equal(c, x * 2 + y)


def test_native_nested_tuple_carried_state(compiler):
  compiled = T.compile(tuple_recurrence_template(), compiler=compiler)
  for count in (0, 1, 3, 4):
    a, c = np.arange(7, dtype=np.float32), np.empty(7, dtype=np.float32)
    compiled.operator.native(a, c, count)
    x, y, scalar = a.copy(), a + 10, 0
    for _ in range(count):
      x, y, scalar = y, x + scalar, scalar + 1
    np.testing.assert_array_equal(c, x * 2 + y)


def test_helper_conditional_nested_tuple_result(compiler):
  @T.func
  def helper(x, choice):
    if choice:
      return x + 1, (2, x > 3)
    return x - 1, (3, x < 3)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32), *, choice: bool = True):
      y, pair = helper(T.load(A), choice)
      factor, mask = pair
      T.store(C, T.where(mask, y * factor, y))

    return kernel

  a, kernel = np.arange(7, dtype=np.float32), factory()
  for choice in (False, True):
    y, factor, mask = (a + 1, 2, a > 3) if choice else (a - 1, 3, a < 3)
    np.testing.assert_array_equal(kernel(a, choice=choice), np.where(mask, y * factor, y))


@pytest.mark.parametrize("budget", [0, 12])
def test_predicate_carried_state(compiler, budget):
  @T.template
  def kernel(A: T.Tensor((19,), dtype=T.float32), C: T.Tensor((19,), dtype=T.float32, access="output")):
    x = T.load(A)
    predicate = x > 3
    for i in T.serial(3):
      predicate = predicate & (x < 15 - i)
    T.store(C, T.where(predicate, x, 0))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_state_register_budget": budget})
  a, c = np.arange(19, dtype=np.float32), np.empty(19, dtype=np.float32)
  compiled(a, c)
  np.testing.assert_array_equal(c, np.where((a > 3) & (a < 13), a, 0))


def test_dynamic_scratch_recurrence(compiler):
  n = T.dynamic(lower=0)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((n,), dtype=T.float32), C: T.Tensor((n,), dtype=T.float32), *, count: int = 3):
      x = T.load(A)
      for i in T.serial(count):
        x = x * 2 + 1
      T.store(C, x)

    return kernel

  kernel = factory()
  for size in (0, 1, 19, 257):
    a = np.arange(size, dtype=np.float32)
    np.testing.assert_array_equal(kernel(a), a * 8 + 7)
  assert "simultaneous tile state update" in kernel.generated_source


def test_state_reduction_condition(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5,), dtype=T.float32), C: T.Tensor((5,), dtype=T.float32)):
      x = T.load(A)
      while T.reduce_add(x) < 25:
        x = x + 1
      T.store(C, x)

    return kernel

  a = np.arange(5, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), a + 3)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_mixed_precision_state(compiler, dtype):
  numpy_dtype = np.dtype(dtype) if dtype != "bfloat16" else np.dtype(pytest.importorskip("ml_dtypes").bfloat16)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((17,), dtype=dtype), C: T.Tensor((17,), dtype=dtype)):
      x = T.load(A, dtype=T.float32)
      for i in T.serial(3):
        x = x * 2 + 1
      T.store(C, x)

    return kernel

  a = np.arange(17, dtype=np.float32).astype(numpy_dtype)
  expected = (a.astype(np.float32) * 8 + 7).astype(numpy_dtype)
  c = np.empty_like(a)
  factory()(a, c)
  np.testing.assert_array_equal(c.astype(np.float32), expected.astype(np.float32))


def test_scalable_vector_register_state(compiler):
  n = T.dynamic(lower=0)
  descriptor = T.Vector(T.float32, pow=1)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((n,), dtype=T.float32), C: T.Tensor((n,), dtype=T.float32)):
      x = T.load(A, vector=descriptor)
      for i in T.serial(3):
        x = x * 2 + 1
      T.store(C[: T.lanes(descriptor)], x)

    return kernel

  kernel = factory()
  # Native AVX-512: two 16-lane f32 registers. The generated schedule itself
  # remains scalable and the ARM cross suite checks both VLA and VLS modes.
  for size in (0, 1, 17, 32, 37):
    a, c = np.arange(size, dtype=np.float32), np.full(size, -123, dtype=np.float32)
    kernel(a, c)
    np.testing.assert_array_equal(c[:32], a[:32] * 8 + 7)
    assert np.all(c[32:] == -123)
  assert not re.search(r"vecops_dsl::Buffer<[^>]+> dsl_f_state_", kernel.generated_source)


def test_scalable_vector_widened_state(compiler):
  n = T.dynamic(lower=0)
  descriptor = T.Vector(T.float32)

  @T.template
  def kernel(A: T.Tensor((n,), dtype=T.float32), C: T.Tensor((n,), dtype="int64", access="output")):
    x = T.cast(T.load(A, vector=descriptor), "int64")
    for i in T.serial(3):
      x = x + 1
    T.store(C[: T.lanes(descriptor)], x)

  compiled = T.compile(kernel, compiler=compiler)
  for size in (0, 1, 9, 16, 17, 37):
    a, c = np.arange(size, dtype=np.float32), np.full(size, -123, dtype=np.int64)
    compiled(a, c)
    np.testing.assert_array_equal(c[:16], a[:16].astype(np.int64) + 3)
    assert np.all(c[16:] == -123)


def test_helper_nested_loop_early_return(compiler):
  @T.func
  def helper(x, limit):
    for i in T.serial(8):
      for j in T.serial(4):
        if i + j >= limit:
          return x * (i + 1), j
    return x * 10, 4

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32), *, limit: int = 2):
      y, offset = helper(T.load(A), limit)
      T.store(C, y + offset + 1)

    return kernel

  a, kernel = np.arange(7, dtype=np.float32), factory()
  for limit, factor, offset in ((0, 1, 0), (2, 1, 2), (4, 2, 3), (20, 10, 4)):
    np.testing.assert_array_equal(kernel(a, limit=limit), a * factor + offset + 1)
  assert "inlined helper exit region" in kernel.generated_source


def test_dot_carried_state(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((3, 5), dtype=T.float32), B: T.Tensor((5, 7), dtype=T.float32), C: T.Tensor((3, 7), dtype=T.float32)
    ):
      acc = T.zeros((3, 7), dtype=T.float32)
      for i in T.serial(3):
        acc = T.dot(A, B, acc=acc)
      T.store(C, acc)

    return kernel

  a, b = np.arange(15, dtype=np.float32).reshape(3, 5), np.arange(35, dtype=np.float32).reshape(5, 7)
  np.testing.assert_allclose(factory()(a, b), (a @ b) * 3)


def test_void_helper_returns_only_from_helper(compiler):
  @T.func
  def helper(a, output, limit):
    for i in T.serial(4):
      if i == limit:
        return
      T.store(output, T.load(a) + i)
    T.store(output, T.load(a) + 10)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32), *, limit: int = 2):
      helper(A, C, limit)
      T.store(C, T.load(C) * 2)

    return kernel

  a, kernel = np.arange(7, dtype=np.float32), factory()
  np.testing.assert_array_equal(kernel(a, limit=2), (a + 1) * 2)
  np.testing.assert_array_equal(kernel(a, limit=10), (a + 10) * 2)


def test_helper_loop_missing_return_is_not_a_zero_result(compiler):
  @T.func
  def helper(x, count):
    for i in T.serial(count):
      return x + 1

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32), *, count: int = 1):
      T.store(C, helper(T.load(A), count))

    return kernel

  a, kernel = np.arange(7, dtype=np.float32), factory()
  np.testing.assert_array_equal(kernel(a), a + 1)
  with pytest.raises(RuntimeError, match="without returning"):
    kernel(a, count=0)


@pytest.mark.parametrize("budget", [0, 12])
def test_carried_input_value_survives_memory_write(compiler, budget):
  @T.template
  def kernel(A: T.Tensor((19,), dtype=T.float32, access="inout"), C: T.Tensor((19,), dtype=T.float32, access="output")):
    x = T.load(A)
    T.store(A, T.zeros((19,), dtype=T.float32))
    for i in T.serial(3):
      x = x + 1
    T.store(C, x)

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_state_register_budget": budget})
  a, c = np.arange(19, dtype=np.float32), np.empty(19, dtype=np.float32)
  expected = a + 3
  compiled(a, c)
  np.testing.assert_array_equal(c, expected)
  np.testing.assert_array_equal(a, np.zeros_like(a))


def test_dynamic_state_extent_is_captured(compiler):
  @T.template
  def kernel(A: T.Tensor((7,), dtype=T.float32), C: T.Tensor((7,), dtype=T.float32, access="output")):
    width = 3
    x = T.zeros((width,), dtype=T.float32)
    for i in T.serial(2):
      width += 1
      x = T.load(A[:width])
    T.store(C[:3], x)

  compiled = T.compile(kernel, compiler=compiler)
  with pytest.raises((ValueError, RuntimeError), match="state shape changes"):
    compiled(np.arange(7, dtype=np.float32), np.empty(7, dtype=np.float32))


def test_worker_private_state(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((64, 17), dtype=T.float32), C: T.Tensor((64, 17), dtype=T.float32)):
      for row in T.grid(64, parallel="require"):
        x = T.load(A[row, :])
        for i in T.serial(3):
          x = x * 2 + 1
        T.store(C[row, :], x)

    return kernel

  a = np.arange(64 * 17, dtype=np.float32).reshape(64, 17)
  with vecops.cpu_thread_limit(4):
    np.testing.assert_array_equal(factory()(a), a * 8 + 7)


@pytest.mark.parametrize("kind", ["dtype", "shape", "view", "tag"])
def test_invalid_state_changes(compiler, kind):
  @T.template
  def kernel(A: T.Tensor((7,), dtype=T.float32)):
    x = T.load(A)
    view = A
    for i in T.serial(2):
      if kind == "dtype":
        x = T.cast(x, T.float64)
      elif kind == "shape":
        x = T.zeros((3,), dtype=T.float32)
      elif kind == "tag":
        x = T.load(A, vector=T.Vector(T.float32, N=7))
      else:
        view = A[1:]
    T.load(view)

  with pytest.raises(TypeError, match="state dtype|state logical shape|view rebinding|state tag"):
    T.compile(kernel, compiler=compiler)


@pytest.mark.parametrize("policy", ["auto", "require"])
def test_shared_carried_state_cannot_run_concurrently(compiler, policy):
  @T.template
  def kernel(C: T.Tensor((7,), dtype=T.float32, access="output")):
    x = T.zeros((7,), dtype=T.float32)
    for i in T.grid(3, parallel=policy):
      x = x + i + 1
    T.store(C, x)

  if policy == "require":
    with pytest.raises(TypeError, match="shared mutable tile state"):
      T.compile(kernel, compiler=compiler)
  else:
    compiled = T.compile(kernel, compiler=compiler)
    c = np.empty(7, dtype=np.float32)
    with vecops.cpu_thread_limit(4):
      compiled(c)
    np.testing.assert_array_equal(c, np.full(7, 6, dtype=np.float32))
