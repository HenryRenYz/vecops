"""Attention composition through indexing, strided views, grids and AMX."""

import numpy as np
import pytest
import vecops
import vecops.language as T


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-attention-edges")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


def test_indexed_attention_in_grid(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(
    Q: T.Tensor((5, 8), dtype="float32"),
    K: T.Tensor((7, 8), (16, 2), "float32"),
    V: T.Tensor((7, 5), (10, 2), "float32"),
    Idx: T.Tensor((9,), dtype="int64"),
    C: T.Tensor((5, 5), dtype="float32", access="output"),
  ):
    selected_k = T.workspace((9, 8))
    selected_v = T.workspace((9, 5))
    row = T.expand_dims(T.load(Idx), 1)
    T.store(selected_k, T.gather(K, (row, T.indices((9, 8), axis=1))))
    T.store(selected_v, T.gather(V, (row, T.indices((9, 5), axis=1))))
    for tile in T.grid(3, parallel="require"):
      T.library.scaled_dot_product_attention(
        Q[tile * 2 : tile * 2 + 2, :],
        selected_k,
        selected_v,
        C[tile * 2 : tile * 2 + 2, :],
        block_k=4,
        accuracy="strict",
        implementation="vector",
      )

  rng = np.random.default_rng(15)
  q = rng.normal(size=(5, 8)).astype(np.float32)
  k = rng.normal(size=(7, 16)).astype(np.float32)[:, ::2]
  v = rng.normal(size=(7, 10)).astype(np.float32)[:, ::2]
  idx = np.array([3, 2, 6, 4, 0, 3, 1, 5, 3], dtype=np.int64)
  c = np.empty((5, 5), dtype=np.float32)
  compiled = T.compile(kernel, compiler=compiler)
  logits = q.astype(np.float64) @ k[idx].astype(np.float64).T / np.sqrt(8)
  p = np.exp(logits - logits.max(axis=1, keepdims=True))
  expected = (p / p.sum(axis=1, keepdims=True)) @ v[idx]
  for budget in (4, 1, 4):
    with vecops.cpu_thread_limit(budget):
      compiled(q, k, v, idx, c)
    np.testing.assert_allclose(c, expected, rtol=7e-6, atol=1e-6)


def test_attention_default_bfloat16_probabilities(compiler):
  torch = pytest.importorskip("torch")

  @T.template
  def kernel(
    Q: T.Tensor((2, 16), dtype="bfloat16"),
    K: T.Tensor((9, 16), dtype="bfloat16"),
    V: T.Tensor((9, 7), dtype="bfloat16"),
    C: T.Tensor((2, 7), dtype="float32", access="output"),
  ):
    T.library.scaled_dot_product_attention(Q, K, V, C, block_k=8, accuracy="strict", implementation="amx")

  torch.manual_seed(16)
  q = torch.randn(2, 16).bfloat16()
  k = torch.randn(9, 16).bfloat16()
  v = torch.randn(9, 7).bfloat16()
  c = torch.empty(2, 7)
  T.compile(kernel, compiler=compiler)(q, k, v, c)
  expected = torch.softmax(q.float() @ k.float().T / 4, dim=-1) @ v.float()
  np.testing.assert_allclose(c.numpy(), expected.numpy(), rtol=0.008, atol=0.003)


def test_attention_dynamic_empty_axes(compiler):
  rows, keys = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    Q: T.Tensor((rows, 8), dtype="float32"),
    K: T.Tensor((keys, 8), dtype="float32"),
    V: T.Tensor((keys, 5), dtype="float32"),
    C: T.Tensor((rows, 5), dtype="float32", access="output"),
  ):
    T.library.scaled_dot_product_attention(Q, K, V, C, block_k=4, accuracy="strict", implementation="vector")

  compiled = T.compile(kernel, compiler=compiler)
  for m, n in ((0, 0), (0, 9), (3, 0)):
    q, k, v = np.zeros((m, 8), np.float32), np.zeros((n, 8), np.float32), np.zeros((n, 5), np.float32)
    c = np.full((m, 5), -7, np.float32)
    compiled(q, k, v, c)
    np.testing.assert_array_equal(c, np.zeros_like(c))
