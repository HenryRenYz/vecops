"""Pure frontend regressions for boolean values captured by T.func helpers."""

from __future__ import annotations

import pytest
import vecops.language as T
from vecops.language._compiler.frontend.parser import lower


def _captured_matmul_template(swap):
  @T.func
  def multiply(a, b, c):
    return T.library.matmul((a, b), c, swap=swap)

  @T.template
  def kernel(
    A: T.Tensor((4, 4), dtype="float32"),
    B: T.Tensor((4, 4), dtype="float32"),
    C: T.Tensor((4, 4), dtype="float32", access="output"),
  ):
    result = multiply(T.load(A), T.load(B), C)
    T.store(C, result)

  return kernel


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("as_const", [False, True])
def test_captured_swap_bool_lowers_like_t_const(value, as_const):
  captured = T.Const(value) if as_const else value
  template = _captured_matmul_template(captured)

  # Frontend lower only: this exercises the helper-capture and forwarding
  # boundary without compiler discovery, native code generation, or execution.
  lowered = lower(template.ir)
  matmuls = [
    statement.value
    for statement in lowered.body.statements
    if getattr(statement, "value", None) is not None and getattr(statement.value, "op", None) == "dot"
  ]
  assert matmuls
  assert matmuls[0].data.swap is value
