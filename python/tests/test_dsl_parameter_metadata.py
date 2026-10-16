# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Entry snapshots cover immutable ABI descriptors, never mutable DSL state."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def metadata_template(*, grid=False, scratch=False):
  N, AS, CS = T.dynamic(lower=0), T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), T.float32),
    C: T.Tensor((N,), (CS,), T.float32, access="output"),
    repeat: int,
  ):
    if grid:
      for index in T.grid(N, parallel="require"):
        C[index] = T.load(A[index]) + 1
    else:
      offset = 0
      for iteration in T.serial(repeat, unroll=1):
        if scratch:
          temporary = T.alloc((N + iteration,), dtype=T.float32)
          T.store(temporary[:N], T.load(A))
          T.store(temporary[0:1], offset)
          T.store(C, T.load(temporary[:N]))
        else:
          T.store(C, T.load(A) + offset)
        offset += 1  # noqa: SIM113 -- exercise explicit loop-carried DSL state
        T.compiler_fence()

  return kernel


def snapshot_map(source):
  return dict(re.findall(r"const auto (dsl_c_parameter_\w+) = ([^;\n]+);", source))


@pytest.mark.parametrize("target", ("SVE", "AVX512"))
@pytest.mark.parametrize("grid", (False, True))
def test_entry_parameter_metadata_reads_once_and_uses_only_exact_raw_mapping(target, grid):
  source = CPUBackend().lower(metadata_template(grid=grid).ir, (), target=target).source
  snapshots = snapshot_map(source)
  assert set(snapshots.values()) == {
    "A.data()",
    "C.data()",
    *(
      f"static_cast<std::int64_t>({name}.template {operation}<0>())"
      for name in ("A", "C")
      for operation in ("size", "stride")
    ),
  }
  assert all(source.count(expression) == 1 for expression in snapshots.values())
  # Runtime scalar arguments and induction/carried locals are not tensor
  # descriptor metadata. Their evaluation must remain in the loop domain.
  assert not any("repeat" in expression or "offset" in expression for expression in snapshots.values())
  assert_no_vec_mask_arrays(source)
  if grid:
    assert "grid_lanes" in source
    assert "VECOPS_INLINE_LAMBDA" in source
  else:
    loop = source[source.index("for (vecops::nint_t") :]
    assert "dsl_f_offset_" in loop
    assert re.search(r"dsl_f_offset_\d+ =", loop)


def test_resizable_scratch_keeps_live_pointer_and_dynamic_extent_reads():
  source = CPUBackend().lower(metadata_template(scratch=True).ir, (), target="SVE").source
  snapshots = snapshot_map(source)
  assert all(re.search(r"\b(?:A|C)\.", expression) for expression in snapshots.values())
  storage = re.search(r"vecops_dsl::Buffer<float> (dsl_f_\w+);", source).group(1)
  assert f"{storage}.data()" in source
  loop = source[source.index("for (vecops::nint_t") :]
  assert f"{storage}.resize(" in loop
  assert "dsl_f_iteration_" in loop
  assert "invocation-invariant" in source


@pytest.mark.parametrize("shape", ((), (0,), (0, 3), (2, 0, 3)))
def test_scalar_and_zero_extent_descriptor_snapshots_do_not_read_tensor_contents(shape):
  @T.template
  def kernel(A: T.Tensor(shape, dtype=T.float32), C: T.Tensor(shape, dtype=T.float32, access="output")):
    T.store(C, T.load(A) + 1)

  source = CPUBackend().lower(kernel.ir, (), target="SVE").source
  snapshots = snapshot_map(source)
  assert len(snapshots) == 2 + 4 * len(shape)
  assert all("load_convert" not in expression and "[" not in expression for expression in snapshots.values())
  assert all(source.count(expression) == 1 for expression in snapshots.values())


def test_snapshot_policy_is_independently_disabled_and_type_checked():
  source = (
    CPUBackend()
    .lower(metadata_template().ir, (), target="SVE", options={"cpu_parameter_metadata_snapshot": False})
    .source
  )
  assert not snapshot_map(source)
  assert "A.data()" in source
  assert "A.size(0)" in source
  for invalid in (0, 1, None, "true"):
    with pytest.raises(TypeError, match="cpu_parameter_metadata_snapshot must be bool"):
      CPUBackend().lower(metadata_template().ir, (), target="SVE", options={"cpu_parameter_metadata_snapshot": invalid})


def test_dynamic_axis_snapshots_preserve_existing_meta_value_constraints():
  N = T.dynamic(alignment=4, lower=4, upper=128)

  @T.template
  def kernel(A: T.Tensor((N, 8), dtype=T.float32), C: T.Tensor((N, 8), dtype=T.float32, access="output")):
    T.store(C, T.load(A) + 1)

  source = CPUBackend().lower(kernel.ir, (), target="Scalar").source
  snapshots = snapshot_map(source)
  assert "static_cast<std::int64_t>(A.template size<0>())" in snapshots.values()
  assert "static_cast<std::int64_t>(C.template size<0>())" in snapshots.values()
  # Static axes already preserve Const metadata through the original accessor.
  assert "A.size(1)" in snapshots.values()
  assert "A.stride(0)" in snapshots.values()
  assert "A.stride(1)" in snapshots.values()
  assert "VECOPS_ASSUME" not in source[source.index("void __kernel__(") :]


@pytest.fixture(scope="module")
def native_metadata_kernels(tmp_path_factory):
  path = tmp_path_factory.mktemp("parameter-metadata")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for scratch in (False, True):
      for policy in (False, True):
        compiled = T.compile(
          metadata_template(scratch=scratch),
          compiler=compiler,
          pass_options={"cpu_parameter_metadata_snapshot": policy},
        )
        meta = vecops.TensorMeta((17,), dtype="float32")
        compiled.operator(meta, meta, 3, **compiled._core.specialization_values)
        kernels[scratch, policy] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("scratch", (False, True))
@pytest.mark.parametrize("policy", (False, True))
def test_native_entry_snapshots_preserve_aliases_strides_resizes_and_loop_state(
  native_metadata_kernels, scratch, policy
):
  compiled = native_metadata_kernels[scratch, policy]
  for length in (1, 17, 65):
    for layout in ("dense", "strided", "negative", "alias"):
      storage = np.linspace(-1, 1, length * 2 + 1, dtype=np.float32)
      inputs = storage[:length] if layout != "strided" else storage[: length * 2 : 2]
      if layout == "negative":
        inputs = inputs[::-1]
      output = storage[1 : length + 1] if layout == "alias" else np.empty_like(inputs)
      # Shifted aliasing intentionally changes input memory each iteration;
      # metadata snapshots may not snapshot or freeze tensor contents.
      expected_storage = storage.copy()
      expected_input = expected_storage[:length] if layout != "strided" else expected_storage[: length * 2 : 2]
      if layout == "negative":
        expected_input = expected_input[::-1]
      expected_output = expected_storage[1 : length + 1] if layout == "alias" else np.empty_like(inputs)
      for offset in range(3):
        value = expected_input.copy()
        if scratch:
          value[0] = offset
        else:
          value += offset
        expected_output[:] = value
      compiled(inputs, output, 3)
      np.testing.assert_array_equal(output, expected_output, err_msg=f"{scratch}/{policy}/{length}/{layout}")


def test_native_typed_metadata_preserves_required_dimension_validation(tmp_path):
  N = T.dynamic(alignment=4, lower=4, upper=128)

  @T.template
  def kernel(A: T.Tensor((N, 8), dtype=T.float32), C: T.Tensor((N, 8), dtype=T.float32, access="output")):
    T.store(C, T.load(A) + 1)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for length in (4, 8, 128):
    a = np.arange(length * 8, dtype=np.float32).reshape(length, 8)
    c = np.empty_like(a)
    compiled(a, c)
    np.testing.assert_array_equal(c, a + 1)
  for length in (0, 1, 5, 132):
    a = np.ones((length, 8), dtype=np.float32)
    c = np.full_like(a, -123)
    with pytest.raises(ValueError, match=r"shape\[0\] violates Dynamic<4, 4, 128>"):
      compiled(a, c)
    np.testing.assert_array_equal(c, -123)


def test_native_typed_metadata_preserves_required_stride_validation(tmp_path):
  S = T.dynamic(alignment=16, lower=16, upper=128)

  @T.template
  def kernel(A: T.Tensor((4, 8), (S, 1), T.float32), C: T.Tensor((4, 8), dtype=T.float32, access="output")):
    T.store(C, T.load(A) + 1)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for stride in (16, 32, 128, 0, 8, 24, 144):
    owner = np.arange(4 * 144 + 8, dtype=np.float32)
    a = np.ndarray((4, 8), dtype=np.float32, buffer=owner, strides=(stride * owner.itemsize, owner.itemsize))
    c = np.full((4, 8), -123, dtype=np.float32)
    if stride in (16, 32, 128):
      compiled(a, c)
      np.testing.assert_array_equal(c, a + 1)
    else:
      with pytest.raises(ValueError, match=r"stride\[0\] violates Dynamic<16, 16, 128>"):
        compiled(a, c)
      np.testing.assert_array_equal(c, -123)
