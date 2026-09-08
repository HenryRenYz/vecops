"""End-to-end ordered dispatch through a generated mutable Torch bridge."""

from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.typing as vt

_DYNAMIC_OUTPUT_STRIDES = "Dynamic<1,D,1048576> 1"


@vecops.jit(source="../runtime/TestKernel.cpp")
def float32_recipe(
  input: vt.In(torch.float32)["B D", "stride_D 1"],
  output: vt.Out(torch.float32)["B D", _DYNAMIC_OUTPUT_STRIDES],
  factor: float = 2.0,
  add_one: bool = False,
  *,
  ComputeType: vt.DType = torch.float32,
) -> None: ...


@vecops.jit(source="../runtime/TestKernel.cpp")
def float64_recipe(
  input: vt.In(np.float64)["B D", "stride_D 1"],
  output: vt.Out(np.float64)["B D", _DYNAMIC_OUTPUT_STRIDES],
  factor: float = 2.0,
  add_one: bool = False,
  *,
  ComputeType: vt.DType = np.float64,
) -> None: ...


def test_generated_mutable_out_operator_and_ordered_dispatch(tmp_path: Path) -> None:
  compiler = vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "kernels",
    target="Scalar",
    cc="/usr/bin/gcc",
    cxx="/usr/bin/g++",
    jobs=2,
  )
  function = vecops.ops.torch.register(
    "vecops_generated_test::run",
    [float32_recipe, float64_recipe],
    compiler=compiler,
    build_directory=tmp_path / "bridge",
  )

  input = torch.arange(1, 9, dtype=torch.float64).reshape(2, 4)
  output = torch.zeros_like(input)
  result = function(input, output, 3.0, True)

  assert result is output
  torch.testing.assert_close(output, input * 3 + 1)
  assert vecops.ops.torch.vecops_generated_test.run is function
  schema = torch._C._dispatch_find_schema_or_throw("vecops_generated_test::run", "").schema()
  assert "Tensor(a!) output" in str(schema)
  assert "-> ()" in str(schema)
  meta_input = torch.empty((2, 4), dtype=torch.float64, device="meta")
  meta_output = torch.empty_like(meta_input)
  assert function(meta_input, meta_output, 3.0, True) is meta_output
  assert torch._C._dispatch_has_kernel_for_dispatch_key("vecops_generated_test::run", "Meta")

  def compiled_call(value, destination):
    return function(value, destination, 3.0, True)

  compiled = torch.compile(compiled_call, backend="eager", fullgraph=True)
  compiled_output = torch.zeros_like(input)
  assert compiled(input, compiled_output) is compiled_output
  torch.testing.assert_close(compiled_output, input * 3 + 1)
