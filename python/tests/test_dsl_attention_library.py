"""Composable AF3 math and streamed attention, including empty/masked tiles."""

import numpy as np
import pytest
import vecops
import vecops.language as T


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-attention-library")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


@pytest.mark.parametrize("algorithm", ["exp", "polynomial"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_stable_sigmoid(compiler, algorithm, dtype):
  @T.template
  def kernel(A: T.Tensor((37,), dtype=dtype), C: T.Tensor((37,), dtype="float32", access="output")):
    T.store(C, T.library.stable_sigmoid(A, accuracy="strict", algorithm=algorithm))

  a = np.linspace(-80, 80, 37).astype(dtype)
  c = np.empty(37, dtype=np.float32)
  T.compile(kernel, compiler=compiler)(a, c)
  expected = 1 / (1 + np.exp(-a.astype(np.float64)))
  np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-7)


def test_rmsnorm_silu(compiler):
  @T.template
  def kernel(A: T.Tensor((3, 19), dtype="float32"), C: T.Tensor((3, 19), dtype="float32", access="output")):
    x = T.library.rmsnorm(A, eps=0.0001)
    T.store(C, T.library.silu(x, accuracy="strict"))

  a = np.random.default_rng(11).normal(size=(3, 19)).astype(np.float32)
  c = np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, c)
  x = a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 0.0001)
  np.testing.assert_allclose(c, x / (1 + np.exp(-x)), rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("width", [0, 19])
def test_masked_softmax(compiler, width):
  @T.template
  def kernel(
    A: T.Tensor((3, width), dtype="float32"),
    Mask: T.Tensor((3, width), dtype="int32"),
    C: T.Tensor((3, width), dtype="float32", access="output"),
  ):
    T.store(C, T.library.masked_softmax(A, Mask, accuracy="strict"))

  rng = np.random.default_rng(12)
  a = rng.normal(size=(3, width)).astype(np.float32)
  mask = rng.integers(0, 2, size=(3, width), dtype=np.int32)
  mask[1] = 0
  c = np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, mask, c)
  expected = np.zeros_like(a)
  for row in range(3):
    valid = mask[row] != 0
    if valid.any():
      p = np.exp(a[row, valid].astype(np.float64) - a[row, valid].max())
      expected[row, valid] = p / p.sum()
  np.testing.assert_allclose(c, expected, rtol=3e-6, atol=1e-7)


def attention_reference(q, k, v, bias, qm, km, causal, query_offset=0):
  scores = q.astype(np.float64) @ k.astype(np.float64).T / np.sqrt(q.shape[1]) + bias
  valid = (qm[:, None] != 0) & (km[None, :] != 0)
  if causal:
    valid &= np.arange(q.shape[0])[:, None] + query_offset >= np.arange(k.shape[0])[None, :]
  out = np.zeros((q.shape[0], v.shape[1]), dtype=np.float64)
  for row in range(q.shape[0]):
    keep = valid[row]
    if keep.any():
      p = np.exp(scores[row, keep] - scores[row, keep].max())
      out[row] = p @ v[keep].astype(np.float64) / p.sum()
  return out


def test_masked_softmax_row_predicate(compiler):
  @T.template
  def kernel(
    A: T.Tensor((3, 7), dtype="float32"),
    Mask: T.Tensor((3,), dtype="int32"),
    C: T.Tensor((3, 7), dtype="float32", access="output"),
  ):
    predicate = T.expand_dims(T.load(Mask) != 0, 1)
    T.store(C, T.library.masked_softmax(A, predicate, accuracy="strict"))

  a = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
  mask = np.array([1, 0, 1], dtype=np.int32)
  c = np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, mask, c)
  p = np.exp(a - a.max(axis=1, keepdims=True))
  expected = p / p.sum(axis=1, keepdims=True)
  expected[1] = 0
  np.testing.assert_allclose(c, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("keys,causal", [(0, False), (13, False), (13, True)])
def test_online_attention_masks_epilogue(compiler, keys, causal):
  @T.template
  def kernel(
    Q: T.Tensor((3, 8), dtype="float32"),
    K: T.Tensor((keys, 8), dtype="float32"),
    V: T.Tensor((keys, 5), dtype="float32"),
    Bias: T.Tensor((3, keys), dtype="float32"),
    QM: T.Tensor((3,), dtype="int32"),
    KM: T.Tensor((keys,), dtype="int32"),
    C: T.Tensor((3, 5), dtype="float32", access="output"),
  ):
    scratch = T.alloc((3, 5), dtype="float32")
    result = T.library.scaled_dot_product_attention(
      Q,
      K,
      V,
      scratch,
      bias=Bias,
      query_mask=QM,
      key_mask=KM,
      causal=causal,
      query_offset=7,
      block_k=4,
      accuracy="strict",
      implementation="vector",
    )
    T.store(C, result * 0.5 + 1)

  rng = np.random.default_rng(13)
  q = rng.normal(size=(3, 8)).astype(np.float32)
  k = rng.normal(size=(keys, 8)).astype(np.float32)
  v = rng.normal(size=(keys, 5)).astype(np.float32)
  bias = rng.normal(size=(3, keys)).astype(np.float32)
  qm = np.array([1, 0, 1], dtype=np.int32)
  km = np.ones(keys, dtype=np.int32)
  km[:4] = 0  # First block has no valid keys; later blocks must recover.
  c = np.empty((3, 5), dtype=np.float32)
  compiled = T.compile(kernel, compiler=compiler)
  compiled(q, k, v, bias, qm, km, c)
  expected = attention_reference(q, k, v, bias, qm, km, causal, query_offset=7) * 0.5 + 1
  np.testing.assert_allclose(c, expected, rtol=5e-6, atol=5e-7)
  assert "ops::Attention" not in compiled.generated_source


def test_online_attention_bfloat16(compiler):
  torch = pytest.importorskip("torch")

  @T.template
  def kernel(
    Q: T.Tensor((2, 16), dtype="bfloat16"),
    K: T.Tensor((9, 16), dtype="bfloat16"),
    V: T.Tensor((9, 7), dtype="bfloat16"),
    C: T.Tensor((2, 7), dtype="float32", access="output"),
  ):
    T.library.scaled_dot_product_attention(Q, K, V, C, block_k=8, accuracy="strict", probability_dtype="float32")

  torch.manual_seed(14)
  q = torch.randn(2, 16).bfloat16()
  k = torch.randn(9, 16).bfloat16()
  v = torch.randn(9, 7).bfloat16()
  c = torch.empty(2, 7)
  T.compile(kernel, compiler=compiler)(q, k, v, c)
  expected = torch.softmax(q.float() @ k.float().T / 4, dim=-1) @ v.float()
  np.testing.assert_allclose(c.numpy(), expected.numpy(), rtol=5e-6, atol=1e-6)
