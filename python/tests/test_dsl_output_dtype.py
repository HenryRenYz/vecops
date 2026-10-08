# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""String BF16 annotations work for automatic outputs, not just explicit ones."""

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.mark.parametrize("strides", [None, (41, 1)])
def test_automatic_bfloat16_output_from_string_annotation(tmp_path, strides):
  torch = pytest.importorskip("torch")
  compiler = make_compiler(
    target="AVX512",
    jobs=2,
    output="quiet",
    build_dir=tmp_path / "build",
    cache_dir=tmp_path / "cache",
  )

  @T.jit(out_idx=-1, compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((2, 37), dtype="bfloat16"),
      C: T.Tensor((2, 37), strides, dtype="bfloat16"),
    ):
      T.store(C, T.load(A, dtype="float32") + 1)

    return kernel

  a = (torch.arange(74, dtype=torch.float32).reshape(2, 37) / 16).to(torch.bfloat16)
  compiled = factory()
  output = compiled(a)
  assert output.dtype == torch.bfloat16
  assert output.shape == a.shape
  assert output.stride() == (strides or (37, 1))
  torch.testing.assert_close(output, (a.float() + 1).to(torch.bfloat16), rtol=0, atol=0)
  explicit = torch.empty_strided(output.shape, output.stride(), dtype=output.dtype)
  assert compiled(a, explicit) is explicit
  torch.testing.assert_close(explicit, output, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["uint16", "uint32", "uint64"])
def test_torch_unsigned_dtype_through_complete_bridge(tmp_path, name):
  torch = pytest.importorskip("torch")
  dtype = getattr(torch, name, None)
  if dtype is None:
    pytest.skip(f"Torch does not expose {name}")
  compiler = make_compiler(
    target="AVX512",
    jobs=2,
    output="quiet",
    build_dir=tmp_path / "build",
    cache_dir=tmp_path / "cache",
  )

  @T.jit(out_idx=-1, compiler=compiler)
  def factory(dtype):
    @T.template
    def kernel(A: T.Tensor((5,), dtype=dtype), C: T.Tensor((5,), dtype=dtype)):
      T.store(C, T.load(A))

    return kernel

  bits = int(name.removeprefix("uint"))
  a = torch.tensor([0, 1, 7, 1 << (bits - 1), (1 << bits) - 1], dtype=dtype)
  compiled = factory(dtype)
  assert factory(name) is compiled
  layout = compiled.get_layout(a)
  assert layout["C"]["dtype"] == name
  assert layout["C"]["automatically_allocated"]
  output = compiled(a)
  assert output.dtype == dtype
  assert torch.equal(output, a)
  explicit = torch.empty_like(a)
  assert compiled(a, explicit) is explicit
  assert torch.equal(explicit, a)


def test_numpy_bfloat16_extension_automatic_output(tmp_path):
  np = pytest.importorskip("numpy")
  ml_dtypes = pytest.importorskip("ml_dtypes")
  compiler = make_compiler(
    target="AVX512",
    jobs=2,
    output="quiet",
    build_dir=tmp_path / "build",
    cache_dir=tmp_path / "cache",
  )

  @T.jit(out_idx=-1, compiler=compiler)
  def factory(dtype):
    @T.template
    def kernel(A: T.Tensor((37,), dtype=dtype), C: T.Tensor((37,), (2,), dtype=dtype)):
      T.store(C, T.load(A, dtype="float32") + 1)

    return kernel

  a = np.asarray(np.arange(37) / 16, dtype=ml_dtypes.bfloat16)
  compiled = factory(ml_dtypes.bfloat16)
  assert factory("bfloat16") is compiled
  output = compiled(a)
  assert output.dtype == a.dtype
  assert output.strides == (4,)
  np.testing.assert_array_equal(output, (a.astype(np.float32) + 1).astype(a.dtype))


@pytest.mark.parametrize("name", ["float32", "bfloat16", "uint64"])
def test_debug_metadata_unwraps_captured_dtype(name):
  from vecops._dtype import dtype_itemsize

  dtype = T.DType(name, name="dtype")

  @T.template
  def kernel(A: T.Tensor((5,), dtype=dtype), C: T.Tensor((5,), (2,), dtype=dtype)):
    T.store(C, T.load(A))

  lowered = T.lower(kernel, out_idx=-1, target="Scalar")
  metadata = vecops.TensorMeta((5,), dtype=dtype)
  layout = lowered.get_layout(metadata)
  assert layout["A"]["dtype"] == name
  assert layout["C"]["dtype"] == name
  assert layout["C"]["automatically_allocated"]
  assert layout["C"]["logical_bytes"] == 5 * dtype_itemsize(dtype)
  assert layout["C"]["storage_span_bytes"] == 9 * dtype_itemsize(dtype)
  assert lowered.analyze(metadata)["layouts"] == layout
