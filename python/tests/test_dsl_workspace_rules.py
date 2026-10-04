"""Mutable locals cannot smuggle runtime scalar payloads into replay shapes."""

import pytest
import vecops.language as T
from vecops.language._compiler.frontend.parser import lower


@pytest.mark.parametrize("conditional", [False, True])
def test_workspace_mutated_metadata_local_is_not_a_replay_key(conditional):
  @T.template
  def kernel(A: T.Tensor((7,), dtype="float32"), extra: int):
    extent = A.shape[0]
    if conditional:
      if extra > 0:
        extent = extent + extra
    else:
      extent = extent + extra
    T.workspace((extent,))

  with pytest.raises(TypeError, match="tensor metadata"):
    lower(kernel.ir)


def test_workspace_immutable_metadata_expression_remains_legal():
  rows = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((rows, 7), dtype="float32")):
    extent = A.shape[0] + 2
    T.workspace((T.worker_capacity(), extent, *A.shape[1:]))

  lower(kernel.ir)


def test_tuple_slice_rejects_runtime_bound():
  @T.template
  def kernel(A: T.Tensor((7, 9), dtype="float32"), begin: int):
    T.alloc(A.shape[begin:])

  with pytest.raises(TypeError, match="tuple slice bounds"):
    lower(kernel.ir)
