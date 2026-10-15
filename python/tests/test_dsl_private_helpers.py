# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private helper calls preserve captures, typed exits and per-kernel linkage."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.frontend.capture import capture


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("private-helpers")
  return make_compiler(target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build")


def test_inline_option_identity_and_validation():
  def helper(x):
    return x + 1

  default = T.func(helper)
  private = T.func(inline=False)(helper)
  assert default.inline is True and private.inline is False
  assert capture(default.function, inline=default.inline).identity != capture(private.function, inline=private.inline).identity
  for invalid in (0, 1, None, "false"):
    with pytest.raises(TypeError, match="inline must be a bool"):
      T.func(inline=invalid)


def test_scalar_tuple_early_exit_and_nested_calls(compiler):
  @T.func(inline=False)
  def inner(x):
    return x + 1

  @T.func(inline=False)
  def helper(x, count):
    for i in T.serial(count):
      if i == 2:
        return inner(x), x * 3
    return inner(x), x * 2

  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(C: T.Tensor((2,), dtype=T.int64, access="output"), *, x: int, count: int):
      left, right = helper(x, count)
      T.store(C[0], left)
      T.store(C[1], right)
    return kernel

  kernel = factory()
  for count in (0, 2, 5):
    result = np.empty(2, dtype=np.int64)
    kernel(result, x=7, count=count)
    np.testing.assert_array_equal(result, [8, 21 if count > 2 else 14])
  assert "vecops_dsl::private_call([&]()" in kernel.generated_source
  assert "static VECOPS_NOINLINE void private_call" in kernel.generated_source


def test_tile_result_and_input_output_capture(compiler):
  @T.func(inline=False)
  def update(old, output, choice):
    T.store(output, old + 10)
    if choice > 0:
      return old * 2
    return old - 3

  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((19,), dtype=T.float32, access="inout"),
               C: T.Tensor((19,), dtype=T.float32, access="output"), *, choice: int):
      before = T.load(A)
      result = update(before, A, choice)
      T.store(C, result + before)
    return kernel

  kernel = factory()
  initial = np.arange(19, dtype=np.float32)
  for choice in (-1, 1):
    a, c = initial.copy(), np.empty_like(initial)
    kernel(a, c, choice=choice)
    np.testing.assert_array_equal(a, initial + 10)
    np.testing.assert_array_equal(c, initial * 3 if choice > 0 else initial * 2 - 3)


def test_void_private_scratch_loop_and_exception(compiler):
  @T.func(inline=False)
  def fill(a, c, flag):
    assert flag >= 0, "private helper flag"
    scratch = T.alloc(c.shape, dtype=c.dtype)
    T.store(scratch, a + 2)
    T.store(c, T.load(scratch))

  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float32), C: T.Tensor((37,), dtype=T.float32, access="output"), *, flag: int):
      for _ in T.serial(3):
        fill(T.load(A), C, flag)
    return kernel

  kernel = factory()
  a, c = np.arange(37, dtype=np.float32), np.empty(37, dtype=np.float32)
  kernel(a, c, flag=0)
  np.testing.assert_array_equal(c, a + 2)
  with pytest.raises(Exception, match="private helper flag"):
    kernel(a, c, flag=-1)
