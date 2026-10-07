"""Inspection is metadata-only; print records are bounded and effectful."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import cpp_compile_command, make_compiler


@T.template
def printed(
  A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32, access="output"), *, flag: bool = True
):
  T.static_print("elaborated", sep=None)
  T.print("value", None, flag, f"{T.reduce_add(T.load(A)):.2f}", sep="|", end="!\n", flush=True)
  T.print("x" * 40, buffer_size=16, truncation="[cut]", end="")
  T.print("\n", end="", flush=True)
  print(T.load(A), file=T.stderr)
  T.store(C, T.load(A) + 1)


def test_lower_inspection_and_print_options(capsys):
  lowered = T.lower(printed, debug=T.DebugOptions(trace="changed"))
  assert "elaborated" in capsys.readouterr().out
  source = lowered.get_kernel_source()
  assert "PrintRecord<16>" in source
  assert "std::string text" not in source
  assert "[cut]" in source
  assert "Print(" in lowered.get_ir("frontend")
  assert lowered.get_pass_report()
  assert "print" in printed.get_kernel_source()
  disabled = T.lower(printed, debug=T.DebugOptions(prints=False))
  assert "PrintRecord" not in disabled.get_kernel_source()
  assert "execution/Print.h" not in disabled.get_kernel_source()


def test_native_print_and_debug(tmp_path, capfd):
  compiler = make_compiler(
    target="AVX512",
    output="quiet",
    jobs=4,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )
  kernel = T.compile(printed, out_idx=-1, compiler=compiler)
  a = np.arange(4, dtype=np.float32)
  np.testing.assert_array_equal(kernel(a), a + 1)
  output = capfd.readouterr()
  assert "value|None|True|6.00!\n" in output.out
  assert "x" * 11 + "[cut]\n" in output.out
  assert "float32[0, 1, 2, 3]" in output.err
  assert "[vecops]" not in output.err
  assert kernel.get_ir("vector")
  assert kernel.get_build_report()
  assert kernel.debug.artifacts()
  assert isinstance(kernel.get_host_source(), str)
  assert isinstance(kernel.get_assembly(), str)
  report = kernel.analyze(a)
  assert report["kind"] == "cpu-resource-estimate"
  assert kernel.get_layout(a)["A"]["shape"] == (4,)
  np.testing.assert_array_equal(kernel.interpret(a), a + 1)
  report = kernel.profile(a, repeat=3, warmup=1, mode="native", trace=True)
  assert report.median_ns > 0
  assert report.p95_ns >= report.median_ns
  assert report.export_trace(tmp_path / "trace.json").is_file()
  assert kernel.export_debug(tmp_path / "debug").joinpath("report.json").is_file()


@pytest.mark.parametrize(
  "kwargs",
  [{"buffer_size": 0}, {"buffer_size": 4, "truncation": "12345"}, {"truncation": 3}, {"file": 0}, {"file": ""}],
)
def test_invalid_print_settings(kwargs):
  @T.template
  def kernel(A: T.Tensor((1,), dtype=T.float32)):
    T.print(T.load(A))

  import ast

  body = kernel.ir.body
  call = body.tree.body[0].value
  call.keywords = [ast.keyword(arg=k, value=ast.copy_location(ast.Constant(value=v), call)) for k, v in kwargs.items()]
  with pytest.raises(TypeError, match="print"):
    T.lower(kernel)


@T.template
def scratch_kernel(A: T.Tensor((31,), dtype=T.float32), C: T.Tensor((31,), dtype=T.float32, access="output")):
  temporary = T.alloc((31,), dtype=T.float32)
  for i in T.serial(2):
    T.store(temporary, T.load(A) + i)
    T.store(C, T.load(temporary) * 2)


def test_memory_lifetimes_and_interpreter(tmp_path):
  compiler = make_compiler(
    target="AVX512",
    output="quiet",
    jobs=4,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )
  kernel = T.compile(scratch_kernel, out_idx=-1, compiler=compiler, debug=T.DebugOptions(memory=True, timeline=True))
  a = np.arange(31, dtype=np.float32)
  np.testing.assert_array_equal(kernel(a), (a + 1) * 2)
  np.testing.assert_array_equal(kernel.interpret(a), (a + 1) * 2)
  result = kernel.profile(a, repeat=2, warmup=1, memory=True, trace=True, mode="native")
  assert result.memory_events
  assert result.memory_summary["instrumented_peak_bytes"] >= 31 * 4
  assert result.memory_summary["live_bytes_at_end"] == 0
  assert any(e["cat"] == "native" for e in result.events)
  assert not result.memory_summary["dropped"]


def test_multithreaded_print_records(tmp_path, capfd):
  @T.template
  def kernel(C: T.Tensor((8,), dtype=T.float32, access="output")):
    with T.team(8) as worker:
      for step in T.serial(4):
        T.print(f"worker={worker},step={step}", flush=True)
      C[worker] = float(worker)

  compiler = make_compiler(output="quiet", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler, out_idx=-1)
  output = np.empty(8, dtype=np.float32)
  compiled(output)
  np.testing.assert_array_equal(output, np.arange(8))
  lines = capfd.readouterr().out.splitlines()
  assert sorted(lines) == sorted(f"worker={w},step={s}" for w in range(8) for s in range(4))


def test_metadata_only_and_correctness_utilities():
  from vecops.language.debug import compare, minimize

  report = T.lower(scratch_kernel).analyze(np.arange(31, dtype=np.float32))
  assert report["layouts"]["C"]["automatically_allocated"]
  assert compare(np.array([np.nan, np.inf, 1.0]), np.array([np.nan, np.inf, 1.0]))["ok"]
  a = np.array([1], dtype=np.float32)
  assert compare(a, np.nextafter(a, np.float32(2)))["max_ulp"] == 1
  assert not compare(np.array([np.inf]), np.array([-np.inf]))["ok"]
  result = minimize((1, 2, 3, 4), lambda values: 3 in values)
  assert result["minimal"] and result["case"] == (3,)
  assert not minimize((1, 2, 3, 4), lambda values: bool(values), max_attempts=3)["minimal"]


def test_fixed_buffer_formatting_has_no_heap_allocations(tmp_path):
  import subprocess
  from pathlib import Path

  root = Path(__file__).resolve().parents[2]
  executable = tmp_path / "print_reference"
  subprocess.run(
    cpp_compile_command(
      Path(__file__).with_name("print_reference.cpp"),
      executable,
      flags=[
        "-std=c++20",
        "-O3",
        "-I",
        str(root / "include"),
      ],
    ),
    check=True,
  )
  result = subprocess.run([str(executable)], capture_output=True, check=True)
  expected = (
    "42 1.250 text\n中文中[cut]abcdeENDa\0b"
    + f"\n{1e300:.3f}|{0.5:+010.2f}|{10.0:#.4g}|{np.finfo(np.float64).max:.0f}\n"
  )
  assert result.stdout == expected.encode()


def test_print_utf8_masks_and_float16(tmp_path, capfd):
  @T.template
  def kernel(A: T.Tensor((4,), dtype=T.float16)):
    T.static_print("中文中文", buffer_size=16, truncation="[cut]", end="")
    T.print("中文中文", buffer_size=16, truncation="[cut]", end="")
    T.print(T.load(A) > 1, T.load(A), sep="|", flush=True)
    T.print(buffer_size=4, truncation="", flush=True)

  compiled = T.compile(
    kernel,
    compiler=make_compiler(output="quiet", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
  )
  compiled(np.arange(4, dtype=np.float16))
  assert capfd.readouterr().out == "中文中[cut]中文中[cut]bool[False, False, True, True]|float16[0, 1, 2, 3]\n\n"


def test_print_view_memory_effects_are_visible_to_grid_planning():
  from vecops.language._compiler.ir.traversal import nodes

  @T.template
  def kernel(A: T.Tensor((8, 128), dtype=T.float32, access="inout")):
    for i in T.grid(8):
      T.store(A[i, :], T.load(A[i, :]) + 1)
    for i in T.grid(8):
      T.print(A[i, :])

  lowered = T.lower(kernel)
  grids = [n for n in nodes(lowered.kernel_ir.body) if type(n).__name__ == "Grid"]
  assert lowered.backend_ir.grids[id(grids[1])].reads == frozenset({"A"})


def test_unbounded_float_format_is_rejected_before_native_build():
  @T.template
  def kernel(A: T.Tensor((1,), dtype=T.float64)):
    T.print(f"{T.load(A):.1000f}")

  with pytest.raises(TypeError, match="precision.*64"):
    T.lower(kernel)


def test_grid_resource_estimate_uses_target_lanes_and_thread_limit():
  @T.template
  def kernel(A: T.Tensor((8, 4096), dtype=T.float32), C: T.Tensor((8, 4096), dtype=T.float32)):
    for i in T.grid(8):
      T.store(C[i, :], T.load(A[i, :]) * 2 + 1)

  lowered = T.lower(kernel, target="AVX512", out_idx=-1)
  with vecops.cpu_thread_limit(1):
    report = lowered.analyze(np.zeros((8, 4096), dtype=np.float32))
  assert report["phases"][0]["selected_workers"] == 1
  assert report["phases"][0]["programs"] == 8
  assert report["layouts"]["C"]["access"] == "output"


def test_deterministic_ir_and_failed_pass_report():
  from vecops.language import passes

  @passes.register_pass("debug-failure-test", stage="kernel")
  def fail(kernel, context):
    raise RuntimeError("intentional pass failure")

  try:
    with pytest.raises(RuntimeError, match="intentional") as error:
      T.lower(scratch_kernel, pipeline=["debug-failure-test"], debug=T.DebugOptions(trace="all"))
    assert error.value.debug_info.get_pass_report()[0]["status"] == "failed"
  finally:
    passes.unregister_pass("debug-failure-test")
  first = T.lower(scratch_kernel)
  second = T.lower(scratch_kernel)
  for stage in ("frontend", "kernel", "vector", "backend"):
    assert first.get_ir(stage) == second.get_ir(stage)


def test_torch_bridge_inherits_quiet_output_and_build_observer(tmp_path, capfd, monkeypatch):
  import torch

  # Earlier Torch tests may have populated the process-wide bridge cache with
  # the same schema-shaped DSO. Keep this observer assertion about a cold
  # build deterministic instead of requiring a "Torch bridges" event on a
  # legitimate cache hit.
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_DIR", str(tmp_path / "torch-bridges"))

  @T.template
  def kernel(A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32)):
    T.store(C, T.load(A) + 2)

  compiler = make_compiler(
    target="AVX512",
    output="quiet",
    jobs=4,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )
  compiled = T.compile(kernel, out_idx=-1, compiler=compiler)
  a = torch.arange(4, dtype=torch.float32)
  torch.testing.assert_close(compiled(a), a + 2)
  output = capfd.readouterr()
  assert not output.out and not output.err
  phases = {event["phase"].split(":", 1)[0] for event in compiler.build_events}
  assert phases == {"Torch bridges", "kernels"}
