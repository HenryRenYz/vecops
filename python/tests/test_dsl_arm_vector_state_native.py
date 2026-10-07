"""Native mixed SVE/SME correctness and JIT cache isolation across VL changes.

Run only on a Linux AArch64 SME compute allocation. No vector or ZA value is
returned to Python: each ordinary-PCS kernel call finishes its streaming owner
before the next prctl, and SVCR is checked immediately before state changes.
The original per-thread length/inheritance flags are restored in finally.
"""

import ctypes
import errno
import os
import platform
import subprocess
import sys

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import PR_SME_GET_VL, PR_SVE_GET_VL, VL_LEN_MASK, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.cost.model import CostProfile

native_arm = pytest.mark.skipif(
  platform.machine().lower() not in {"aarch64", "arm64"} or sys.platform != "linux",
  reason="native Linux AArch64 vector-state test",
)
PR_SVE_SET_VL = 50
PR_SME_SET_VL = 63


class ArmVectorControl:
  """Change only a caller with no live vector/ZA state; preserve raw flags."""

  def __init__(self, probe):
    self.probe = probe
    self.libc = ctypes.CDLL(None, use_errno=True)
    self.libc.prctl.restype = ctypes.c_int
    self.original_sve = self.raw(PR_SVE_GET_VL)
    self.original_sme = self.raw(PR_SME_GET_VL)

  @property
  def original_pair(self):
    return self.original_sve & VL_LEN_MASK, self.original_sme & VL_LEN_MASK

  def raw(self, option, value=0):
    result = self.libc.prctl(
      ctypes.c_int(option), ctypes.c_ulong(value), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)
    )
    if result < 0:
      error = ctypes.get_errno()
      raise OSError(error, os.strerror(error))
    return result

  def assert_no_streaming_owner(self):
    assert self.probe.read_svcr() & 3 == 0, "changing VL/SVL with a live SM/ZA region would be unsafe"

  def set_length(self, option, value):
    self.assert_no_streaming_owner()
    return self.raw(option, value)

  def supported(self, option):
    lengths = set()
    for requested in (16, 32, 64, 128, 256):
      try:
        lengths.add(self.set_length(option, requested) & VL_LEN_MASK)
      except OSError as error:
        if error.errno != errno.EINVAL:
          raise
    return sorted(lengths)

  def select(self, pair):
    assert self.set_length(PR_SVE_SET_VL, pair[0]) & VL_LEN_MASK == pair[0]
    assert self.set_length(PR_SME_SET_VL, pair[1]) & VL_LEN_MASK == pair[1]
    self.assert_pair(pair)

  def assert_pair(self, pair):
    self.assert_no_streaming_owner()
    current = get_vector_state()
    assert (current.sve_bytes, current.sme_bytes) == pair

  def restore(self):
    self.set_length(PR_SVE_SET_VL, self.original_sve)
    self.set_length(PR_SME_SET_VL, self.original_sme)
    assert self.raw(PR_SVE_GET_VL) == self.original_sve
    assert self.raw(PR_SME_GET_VL) == self.original_sme


@pytest.fixture
def arm_vector_control(tmp_path):
  state = get_vector_state()
  if state.sve_bytes is None or state.sme_bytes is None:
    pytest.skip("current CPU/OS must support SVE and SME")
  # This read-only scalar probe contains no Vec/ZA values and no SMSTART.
  # It guards every SET operation, including exceptional-path restoration.
  source = tmp_path / "svcr.c"
  source.write_text(
    '#include <stdint.h>\nuint64_t read_svcr(void) { uint64_t v; __asm__ __volatile__("mrs %0, svcr" : "=r"(v)); return v; }\n',
    encoding="utf-8",
  )
  probe_path = tmp_path / "svcr.so"
  subprocess.run(
    [
      os.environ.get("CC", "clang"),
      "-march=armv9-a+sve2+sme",
      "-O2",
      "-shared",
      "-fPIC",
      str(source),
      "-o",
      str(probe_path),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  probe = ctypes.CDLL(str(probe_path))
  probe.read_svcr.restype = ctypes.c_uint64
  control = ArmVectorControl(probe)
  try:
    yield control
  finally:
    control.restore()


@native_arm
def test_native_jit_distinguishes_vl_svl_and_restores_original_state(tmp_path, arm_vector_control):
  control = arm_vector_control
  probe = control.probe
  select = control.select
  sve, sme = control.supported(PR_SVE_SET_VL), control.supported(PR_SME_SET_VL)
  pairs = [(a, b) for a in sve for b in sme]
  unequal = [pair for pair in pairs if pair[0] != pair[1]]
  if not unequal or len(pairs) < 2:
    pytest.skip("CPU/OS exposes no distinct supported VL/SVL pair")
  first_pair = unequal[0]
  second_pair = (
    next(pair for pair in reversed(unequal or pairs) if pair != first_pair)
    if len(unequal) > 1
    else next(pair for pair in pairs if pair != first_pair)
  )
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")

  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((23, 37), dtype="float32"),
      V: T.Tensor((851,), dtype="float32", access="output"),
      C: T.Tensor((37, 23), dtype="float32", access="output"),
    ):
      T.copy(T.reshape(A, (851,)), V)
      with T.sme.region():
        T.transpose_copy(A, C, implementation="sme")

    return kernel

  inputs = np.arange(851, dtype=np.float32).reshape(23, 37) - np.float32(200)
  copied, transposed = np.empty(851, dtype=np.float32), np.empty((37, 23), dtype=np.float32)

  def execute():
    selected = factory()
    selected(inputs, copied, transposed)
    assert probe.read_svcr() & 3 == 0, "DSL kernel leaked a streaming/ZA owner"
    np.testing.assert_array_equal(copied, inputs.reshape(-1))
    np.testing.assert_array_equal(transposed, inputs.T)
    return selected

  call_metadata = (
    vecops.TensorMeta((23, 37), dtype="float32"),
    vecops.TensorMeta((851,), dtype="float32"),
    vecops.TensorMeta((37, 23), dtype="float32"),
  )
  with vecops.collect_compile_requests() as collector:
    select(first_pair)
    first_snapshot = factory()
    first_snapshot.operator(*call_metadata, **first_snapshot._core.specialization_values)
    select(second_pair)
    second_snapshot = factory()
    second_snapshot.operator(*call_metadata, **second_snapshot._core.specialization_values)
  assert len(collector) == 2
  assert (get_vector_state().sve_bytes, get_vector_state().sme_bytes) == second_pair
  # Cold requests retain independent captured policies. Preparing first's
  # snapshot while the caller is at second must not reject that snapshot or
  # silently compile it with second's geometry; invocation is guarded later.
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  assert probe.read_svcr() & 3 == 0
  select(first_pair)
  first = execute()
  select(second_pair)
  with pytest.raises(RuntimeError, match="SVE VL/SME SVL.*calling thread"):
    first(inputs, copied, transposed)
  assert probe.read_svcr() & 3 == 0, "rejected stale kernel changed SM/ZA state"
  second = execute()
  assert first.get_cache_info()["key"] != second.get_cache_info()["key"]
  assert first.operator.compiler.vector_state != second.operator.compiler.vector_state
  select(first_pair)
  restored = execute()
  assert restored is first
  assert restored.get_cache_info()["decorator_hit"]


@T.func
def mixed_worker_program(A, V, C, W, Stats, task):
  # Bounds guards for slicing are throwing ordinary-PCS preparation, not
  # instructions admitted inside the explicit no-throw SME interval.
  input_tile = A[task, :, :]
  output_tile = C[task, :, :]
  # Arithmetic prevents the ordinary-PCS sweeps becoming an opaque memcpy.
  Stats[task, 0] = T.lanes(T.Vector("float32"))
  Stats[task, 1] = T.sme.streaming_lanes("float32")
  T.store(V[task, :], T.load(T.reshape(input_tile, (851,))) + 1)
  transposed_value = T.load(T.transpose(input_tile))
  with T.sme.region():
    T.store(output_tile, transposed_value, implementation="sme")
  Stats[task, 2] = T.lanes(T.Vector("float32"))
  T.store(W[task, :], T.load(T.reshape(input_tile, (851,))) - 1)


def mixed_worker_template(mode, workers):
  tasks = 2 * workers

  @T.template
  def kernel(
    A: T.Tensor((tasks, 23, 37), dtype="float32"),
    V: T.Tensor((tasks, 851), dtype="float32", access="output"),
    C: T.Tensor((tasks, 37, 23), dtype="float32", access="output"),
    W: T.Tensor((tasks, 851), dtype="float32", access="output"),
    Stats: T.Tensor((tasks, 3), dtype="int64", access="output"),
    Team: T.Tensor((workers, 2), dtype="int64", access="output"),
  ):
    if mode == "grid":
      T.store(Team, T.full((workers, 2), 0, dtype="int64"))
      for task in T.grid(tasks, parallel="require", schedule="static"):
        mixed_worker_program(A, V, C, W, Stats, task)
    else:
      with T.team(workers=workers) as worker:
        Team[worker, 0] = T.worker_count()
        Team[worker, 1] = worker
        for task in T.serial(worker, tasks, workers, unroll=1):
          mixed_worker_program(A, V, C, W, Stats, task)
        T.barrier()

  return kernel


def worker_pass_options():
  # This is a vector-state regression, not a scheduling crossover benchmark.
  # Force the cost model to actually launch the allowed worker team, even
  # when a small 23x37 numerical tile would normally prefer serial execution.
  return {"cost_profiles": (CostProfile("test-vector-state-workers", policies=(("grid_worker_cycles", 0),)),)}


@pytest.mark.parametrize("mode", ["grid", "team"])
def test_worker_vector_state_codegen_has_real_vectors_and_inline_callbacks(mode):
  source = CPUBackend().lower(mixed_worker_template(mode, 4).ir, (), target="SME", options=worker_pass_options()).source
  assert ("grid_lanes" if mode == "grid" else "grid_team") in source
  assert "VECOPS_INLINE_LAMBDA" in source
  assert "vecops::vec::load_convert" in source
  assert "vecops::vec::sme::load_hor" in source
  assert "vecops::vec::sme::store_ver" in source
  assert_no_vec_mask_arrays(source)


@native_arm
def test_native_workers_reuse_pool_across_vector_pairs(tmp_path, arm_vector_control, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  workers = min(4, vecops.cpu_thread_pool_capacity())
  if workers < 2:
    pytest.skip("at least two physical CPU workers are required")
  tasks = 2 * workers
  control = arm_vector_control
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")

  def make_factory(mode):
    @T.jit(compiler=compiler, pass_options=worker_pass_options())
    def factory():
      return mixed_worker_template(mode, workers)

    return factory

  factories = {mode: make_factory(mode) for mode in ("grid", "team")}
  inputs = np.arange(tasks * 851, dtype=np.float32).reshape(tasks, 23, 37) - np.float32(200)
  before = np.empty((tasks, 851), dtype=np.float32)
  transposed = np.empty((tasks, 37, 23), dtype=np.float32)
  after = np.empty_like(before)
  stats = np.empty((tasks, 3), dtype=np.int64)
  team = np.empty((workers, 2), dtype=np.int64)

  def execute(mode, pair):
    control.assert_pair(pair)
    configs = control.raw(PR_SVE_GET_VL), control.raw(PR_SME_GET_VL)
    for output in (before, transposed, after, stats, team):
      output.fill(-999)
    kernel = factories[mode]()
    kernel(inputs, before, transposed, after, stats, team)
    control.assert_pair(pair)
    assert (control.raw(PR_SVE_GET_VL), control.raw(PR_SME_GET_VL)) == configs
    np.testing.assert_array_equal(before, inputs.reshape(tasks, -1) + 1)
    np.testing.assert_array_equal(transposed, inputs.transpose(0, 2, 1))
    np.testing.assert_array_equal(after, inputs.reshape(tasks, -1) - 1)
    np.testing.assert_array_equal(stats, np.tile((pair[0] // 4, pair[1] // 4, pair[0] // 4), (tasks, 1)))
    if mode == "team":
      np.testing.assert_array_equal(team[:, 0], np.full(workers, workers))
      np.testing.assert_array_equal(team[:, 1], np.arange(workers))
    else:
      np.testing.assert_array_equal(team, 0)
    assert_no_vec_mask_arrays(kernel.generated_source)
    return kernel

  with vecops.cpu_thread_limit(workers):
    # Warm both facades before any SET_VL. The adaptive pool key is capacity
    # plus unchanged CPU affinity, not VL/SVL, so later calls reuse workers
    # originally born with this pair rather than creating convenient new ones.
    original = {mode: execute(mode, control.original_pair) for mode in factories}
    sve, sme = control.supported(PR_SVE_SET_VL), control.supported(PR_SME_SET_VL)
    pairs = [(a, b) for a in sve for b in sme]
    unequal = [pair for pair in pairs if pair[0] != pair[1]]
    if len(pairs) < 2:
      pytest.skip("CPU/OS exposes only one supported VL/SVL pair")
    first_pair = (unequal or pairs)[0]
    remaining = [pair for pair in unequal if pair != first_pair]
    second_pair = (remaining or [pair for pair in pairs if pair != first_pair])[-1]
    snapshots = {}
    for pair in (first_pair, second_pair, first_pair):
      control.select(pair)
      for mode in factories:
        kernel = execute(mode, pair)
        if (mode, pair) in snapshots:
          assert kernel is snapshots[mode, pair]
        snapshots[mode, pair] = kernel
    for mode in factories:
      assert snapshots[mode, first_pair].get_cache_info()["key"] != snapshots[mode, second_pair].get_cache_info()["key"]
    control.restore()
    for mode in factories:
      assert execute(mode, control.original_pair) is original[mode]
