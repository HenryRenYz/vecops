# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exercise the generated Clang/GOMP adapter's nested and error paths."""

from __future__ import annotations

import ctypes
import shlex
import subprocess
from pathlib import Path

import pytest
import vecops

_PROBE = r"""
namespace {
std::atomic<int> probe_gomp_calls{0};
GompParallel probe_real_gomp = nullptr;
void counted_gomp(void (*fn)(void*), void* data, unsigned int threads, unsigned int flags) {
  probe_gomp_calls.fetch_add(1, std::memory_order_relaxed);
  probe_real_gomp(fn, data, threads, flags);
}
struct ProbeContext {
  State* state;
  bool throw_from_inner;
  std::atomic<int> outer{0};
  std::atomic<int> inner{0};
  std::atomic<int> bad{0};
  std::atomic<unsigned> worker_mask{0};
};
void inner_task(void* opaque, uint32_t index, uint32_t total) {
  auto& context = *static_cast<ProbeContext*>(opaque);
  context.inner.fetch_add(1, std::memory_order_relaxed);
  const int configured_team = context.state->get_max();
  if (total != 3 || at::get_thread_num() != 0 || !at::in_parallel_region() ||
      static_cast<int>(max_threads(context.state)) != configured_team || at::get_num_threads() != configured_team)
    context.bad.fetch_add(1, std::memory_order_relaxed);
  if (context.throw_from_inner && index == 1)
    throw std::runtime_error("nested callback failure");
}
void outer_task(void* opaque, uint32_t index, uint32_t total) {
  auto& context = *static_cast<ProbeContext*>(opaque);
  auto* state = context.state;
  const int thread_id = state->get_id();
  context.outer.fetch_add(1, std::memory_order_relaxed);
  const int configured_team = state->get_max();
  if (total != 4 || thread_id < 0 || thread_id >= 32 || at::get_thread_num() != thread_id ||
      !at::in_parallel_region() || static_cast<int>(max_threads(state)) != configured_team ||
      at::get_num_threads() != configured_team || state->get_num() != configured_team)
    context.bad.fetch_add(1, std::memory_order_relaxed);
  if (thread_id >= 0 && thread_id < 32)
    context.worker_mask.fetch_or(1u << thread_id, std::memory_order_relaxed);
  VecopsError nested_error{};
  std::array<char, 256> nested_message{};
  nested_error.struct_size = sizeof(nested_error);
  nested_error.message = nested_message.data();
  nested_error.message_capacity = nested_message.size();
  const auto status = parallel_for(state, 3, &context, &inner_task, &nested_error);
  if (at::get_thread_num() != thread_id)
    context.bad.fetch_add(1, std::memory_order_relaxed);
  if (status != VECOPS_STATUS_OK)
    throw std::runtime_error("nested callback failure propagated to outer task");
  (void)index;
}
} // namespace

extern "C" VECOPS_RUNTIME_EXPORT int32_t vecops_test_gomp_nested(
    void* gomp, void* get_max, void* get_num, void* get_id, void* is_nested, int throw_from_inner,
    int* parallel_calls, int* outer_tasks, int* inner_tasks, int* bad_checks, unsigned* worker_mask,
    char* error_message, size_t error_capacity) {
  probe_gomp_calls.store(0, std::memory_order_relaxed);
  probe_real_gomp = reinterpret_cast<GompParallel>(gomp);
  State state{0, nullptr, "nested_probe", {}, nullptr, nullptr, &counted_gomp,
              reinterpret_cast<OmpInt>(get_max), reinterpret_cast<OmpInt>(get_num),
              reinterpret_cast<OmpInt>(get_id), reinterpret_cast<OmpInt>(is_nested)};
  ProbeContext context{&state, throw_from_inner != 0};
  VecopsError error{};
  error.struct_size = sizeof(error);
  error.message = error_message;
  error.message_capacity = error_capacity;
  const auto status = parallel_for(&state, 4, &context, &outer_task, &error);
  *parallel_calls = probe_gomp_calls.load(std::memory_order_relaxed);
  *outer_tasks = context.outer.load(std::memory_order_relaxed);
  *inner_tasks = context.inner.load(std::memory_order_relaxed);
  *bad_checks = context.bad.load(std::memory_order_relaxed);
  *worker_mask = context.worker_mask.load(std::memory_order_relaxed);
  return status;
}
"""


def _compile_probe(
  tmp_path: Path, torch, include_root: Path, torch_includes: tuple[Path, ...], torch_libraries: tuple[Path, ...], policy
):
  from vecops import _schema_bridge

  definition = vecops.KernelDef(
    "probe::nested",
    [vecops.TensorDef("input", [1], [1], dtype=vecops.float32)],
  )._native()
  source = _schema_bridge._generate_torch_source("probe", "nested", definition, parallel_backend="gomp") + _PROBE
  source_path = tmp_path / "torch_gomp_nested_probe.cpp"
  library_path = tmp_path / "torch_gomp_nested_probe.so"
  source_path.write_text(source)
  toolchain = policy.toolchain
  env = toolchain.environment
  command = [
    str(toolchain.cxx_compiler),
    *shlex.split(env.get("CXXFLAGS", "")),
    "-std=c++20",
    "-O2",
    "-fPIC",
    "-shared",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    f"-I{include_root}",
  ]
  for include in torch_includes:
    command.extend(("-isystem", str(include)))
  command.extend((str(source_path), "-o", str(library_path)))
  command.extend(shlex.split(env.get("LDFLAGS", "")))
  command.extend(policy.link_options)
  for library_dir in torch_libraries:
    command.extend((f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"))
  command.extend(("-ltorch_cpu", "-ltorch", "-lc10"))
  command.extend(policy.link_libraries)
  completed = subprocess.run(command, capture_output=True, text=True, check=False)
  assert completed.returncode == 0, "\n".join((" ".join(command), completed.stdout, completed.stderr))
  return library_path


def test_generated_gomp_adapter_serializes_nested_parallelism_and_propagates_errors(tmp_path: Path, request):
  torch = pytest.importorskip("torch")
  from vecops import _schema_bridge

  torch, include_root, torch_includes, torch_libraries, parallel_info, _native_library, _identity = (
    _schema_bridge._torch_bridge_inputs()
  )
  if _schema_bridge._torch_openmp_runtime(torch_libraries, parallel_info) != "gomp":
    pytest.skip("Torch does not use the packaged GNU OpenMP adapter")
  compiler = vecops.Compiler(target="Native")
  config = compiler._native_config()
  policy = _schema_bridge._torch_bridge_build_policy(config, torch_libraries, parallel_info)
  if policy.parallel_backend != "gomp":
    pytest.skip("the selected Clang toolchain does not use the direct packaged-GOMP adapter")

  runtime = policy.runtime_path
  assert runtime is not None
  previous_threads = torch.get_num_threads()
  _schema_bridge._verify_gomp_runtime(runtime)
  handle, symbols = _schema_bridge._verified_gomp_handles[str(runtime.resolve())]
  assert handle is not None
  omp_get_dynamic = handle.omp_get_dynamic
  omp_get_dynamic.argtypes = []
  omp_get_dynamic.restype = ctypes.c_int
  omp_set_dynamic = handle.omp_set_dynamic
  omp_set_dynamic.argtypes = [ctypes.c_int]
  omp_set_dynamic.restype = None

  previous_dynamic = int(omp_get_dynamic())

  def restore_runtime_settings():
    try:
      omp_set_dynamic(previous_dynamic)
    finally:
      torch.set_num_threads(previous_threads)

  request.addfinalizer(restore_runtime_settings)
  torch.set_num_threads(4)
  omp_set_dynamic(0)

  library_path = _compile_probe(tmp_path, torch, include_root, torch_includes, torch_libraries, policy)
  probe_library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
  probe = probe_library.vecops_test_gomp_nested
  probe.argtypes = [
    *([ctypes.c_void_p] * 5),
    ctypes.c_int,
    *([ctypes.POINTER(ctypes.c_int)] * 4),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_char_p,
    ctypes.c_size_t,
  ]
  probe.restype = ctypes.c_int32
  function_table = [
    ctypes.c_void_p(symbols[name])
    for name in (
      "GOMP_parallel",
      "omp_get_max_threads",
      "omp_get_num_threads",
      "omp_get_thread_num",
      "omp_in_parallel",
    )
  ]

  def invoke(throw_from_inner: bool):
    parallel_calls = ctypes.c_int()
    outer_tasks = ctypes.c_int()
    inner_tasks = ctypes.c_int()
    bad_checks = ctypes.c_int()
    worker_mask = ctypes.c_uint()
    error = ctypes.create_string_buffer(256)
    status = probe(
      *function_table,
      int(throw_from_inner),
      ctypes.byref(parallel_calls),
      ctypes.byref(outer_tasks),
      ctypes.byref(inner_tasks),
      ctypes.byref(bad_checks),
      ctypes.byref(worker_mask),
      error,
      len(error),
    )
    return (
      status,
      parallel_calls.value,
      outer_tasks.value,
      inner_tasks.value,
      bad_checks.value,
      worker_mask.value,
      error.value.decode(),
    )

  status, calls, outer, inner, bad, worker_mask, message = invoke(False)
  assert status == 0, message
  assert calls == 1
  assert outer == 4 and inner == 12
  assert bad == 0
  assert worker_mask.bit_count() == 4

  status, calls, outer, inner, bad, worker_mask, message = invoke(True)
  assert status != 0
  assert calls == 1, "nested failure must not start another GOMP region"
  assert outer == 4
  assert inner == 8
  assert bad == 0
  assert worker_mask.bit_count() == 4
  assert "nested callback failure propagated" in message
