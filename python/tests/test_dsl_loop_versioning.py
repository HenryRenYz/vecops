# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Root checks preserve serial alias history and fail closed on invalid spans."""

import subprocess
from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import cpp_compile_command, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.support import _LOOP_VERSION_SUPPORT
from vecops.language._compiler.backends.cpu.loop_versioning import collect_loop_versions
from vecops.language._compiler.backends.cpu.verification import verify_prepared


def declaration(reverse=True, width=17):
  N = T.dynamic(lower=0, upper=64)
  Width = T.Const(width)
  Reverse = reverse

  @T.template
  def kernel(
    A: T.Tensor((N, Width), dtype=T.float32),
    C: T.Tensor((N, Width), dtype=T.float32, access="output"),
    D: T.Tensor((N, Width), dtype=T.float32, access="output"),
    mode: int,
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      if Reverse:
        for row in T.serial(N - 1, -1, -1, unroll=1):
          T.store(C[row, :], T.load(A[row, :]) + mode)
          T.store(D[row, :], T.load(A[row, :]) * 3)
      else:
        for row in T.serial(N, unroll=1):
          if mode > 0:
            T.store(C[row, :], T.load(A[row, :]) + 1)
            T.store(D[row, :], T.load(A[row, :]) * 2)
          else:
            T.store(C[row, :], T.load(A[row, :]) - 1)
            T.store(D[row, :], T.load(A[row, :]) * 3)

  return kernel


@pytest.mark.parametrize("reverse", (False, True))
def test_plan_binds_loop_owners_and_serial_effects(reverse):
  artifact = CPUBackend().lower(declaration(reverse).ir, (), target="AVX512")
  plans = artifact.backend_ir.loop_versions
  assert len(plans) == 1 and "loop-invariant root alias version" in artifact.source
  identity, plan = next(iter(plans.items()))
  assert identity == id(plan.loop) and plan.occurrences == 1
  assert plan.pairs == (("A", "C"), ("A", "D"))
  assert {root.storage for root in plan.roots} == {"A", "C", "D"}
  assert all(not root.owned for root in plan.roots)
  assert "retain original loop and first-use checks" in artifact.source
  fast = artifact.source.split("/* loop-invariant root alias version */", 1)[1].split(
    "else /* retain original loop", 1
  )[0]
  assert "const bool dsl_c_alias" not in fast
  for bad in [
    replace(plan, loop=replace(plan.loop)),
    replace(plan, occurrences=2),
    replace(plan, minimum_iterations=0),
    replace(plan, stores=plan.stores[::-1]),
    replace(plan, roots=(replace(plan.roots[0], offset=I.literal(1)), *plan.roots[1:])),
  ]:
    with pytest.raises(ValueError, match="loop version"):
      verify_prepared(replace(artifact.backend_ir, loop_versions={identity: bad}))


@pytest.mark.parametrize("target", ("Scalar", "SVE", "SME"))
def test_other_targets_are_byte_identical(target):
  a = CPUBackend().lower(declaration().ir, (), target=target)
  b = CPUBackend().lower(declaration().ir, (), target=target, options={"cpu_loop_versioning": False})
  assert not a.backend_ir.loop_versions and a.source == b.source


def test_wide_tiles_disabled_and_invalid_options():
  artifact = CPUBackend().lower(declaration(width=257).ir, (), target="AVX512")
  assert not artifact.backend_ir.loop_versions and "try_root_span" not in artifact.source
  artifact = CPUBackend().lower(declaration().ir, (), target="AVX512", options={"cpu_loop_versioning": False})
  assert not artifact.backend_ir.loop_versions
  with pytest.raises(TypeError, match="cpu_loop_versioning must be bool"):
    CPUBackend().lower(declaration().ir, (), target="AVX512", options={"cpu_loop_versioning": 1})


def test_shared_occurrences_opaque_and_execution_scopes():
  artifact = CPUBackend().lower(declaration().ir, (), target="AVX512")
  plan = next(iter(artifact.backend_ir.loop_versions.values()))
  kernel = artifact.kernel
  environment = artifact.backend_ir.vectors.manager.options["_compiler_environment"]
  loop = plan.loop
  opaque = I.Assign("opaque", I.Expr(I.ScalarType("int64"), "unknown_scalar"))
  outer = I.For("outer", I.literal(0), I.literal(2), I.literal(1), I.Block([loop]), unroll=1)
  for body in [
    I.Block([loop, opaque, loop]),
    I.Block([I.Grid(("worker",), (I.literal(1),), I.Block([loop]))]),
    I.Block([I.HardwareRegion("amx", I.Block([loop]))]),
    I.Block([loop, I.PipelineFor(outer)]),
    I.Block([loop, I.UnrolledFor(outer, I.Block([loop]))]),
  ]:
    assert not collect_loop_versions(replace(kernel, body=body), environment.operations, environment.codegen_target)


@pytest.mark.parametrize("reverse", (False, True))
def test_native_cross_row_overlap_repeat_and_empty_loops(tmp_path, reverse):
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(declaration(reverse), compiler=compiler)
  target = compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target
  if target.isa == "avx512":
    assert compiled.backend_ir.loop_versions
  for rows in (0, 1, 3, 5, 33):
    block = rows * 17 + 64
    for kind in ("disjoint", "inplace", "plus", "minus", "all", "outputs", "plus_row", "minus_row"):
      offsets = [0, block, 2 * block]
      if kind == "inplace":
        offsets[1] = 0
      elif kind == "plus":
        offsets[1] = 1
      elif kind == "minus":
        offsets[0], offsets[1] = 1, 0
      elif kind == "all":
        offsets = [0, 1, 2]
      elif kind == "outputs":
        offsets[2] = offsets[1] + 1
      elif kind == "plus_row":
        offsets[1] = 17
      elif kind == "minus_row":
        offsets[0], offsets[1] = 17, 0
      for mode in (-1, 1):
        for repeat in (0, 1, 3):
          owner = np.linspace(0, 0.125, 3 * block + 128, dtype=np.float32)
          expected = owner.copy()
          arrays = [np.ndarray((rows, 17), np.float32, buffer=owner, offset=p * 4) for p in offsets]
          reference = [np.ndarray((rows, 17), np.float32, buffer=expected, offset=p * 4) for p in offsets]
          with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(repeat):
              for row in reversed(range(rows)) if reverse else range(rows):
                reference[1][row] = reference[0][row].copy() + np.float32(mode if reverse else 1 if mode > 0 else -1)
                reference[2][row] = reference[0][row].copy() * np.float32(3 if reverse or mode < 0 else 2)
          compiled(*arrays, mode, repeat)
          np.testing.assert_allclose(owner, expected, rtol=2e-6, atol=2e-6)


def test_root_span_predicate_never_speculates_an_exception(tmp_path):
  source = tmp_path / "root_guard.cpp"
  source.write_text(
    """// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#define VECOPS_ALWAYS_INLINE inline __attribute__((always_inline))
namespace vecops_dsl {
struct Span { std::uintptr_t first, last; };
template<class T> T min_value(T a, T b) { return std::min(a,b); }
template<class T> T max_value(T a, T b) { return std::max(a,b); }
bool overlap(Span a, Span b) { return a.first < b.last && b.first < a.last && a.first != a.last && b.first != b.last; }
}
"""
    + _LOOP_VERSION_SUPPORT
    + """
int main() {
  using namespace vecops_dsl;
  auto ptr = [](std::uintptr_t p) { return reinterpret_cast<const void*>(p); };
  Span span;
  assert(try_root_span(nullptr, {0, -1}, {1, 1}, 4, span));
  assert(span.first == 0 && span.last == 0);
  assert(!try_root_span(nullptr, {1}, {1}, 4, span));
  assert(!try_root_span(ptr(256), {-1}, {1}, 4, span));
  assert(!try_root_span(ptr(256), {3}, {INT64_MAX}, 4, span));
  assert(!try_root_span(ptr(256), {2}, {INT64_MAX/2}, 8, span));
  assert(!try_root_span(ptr(16), {2}, {-64}, 4, span));
  assert(!try_root_span(ptr(UINTPTR_MAX-16), {8}, {1}, 4, span));
  assert(try_root_span(ptr(256), {3}, {-2}, 4, span));
  assert(span.first == 240 && span.last == 260);
  assert(try_root_span(ptr(256), {3}, {1}, 4, span, -4));
  assert(span.first == 240 && span.last == 252);
  assert(!try_root_span(ptr(256), {2}, {-1}, 4, span, INT64_MIN));
  assert(try_disjoint_roots(ptr(256), {3}, {-2}, 4, ptr(1024), {4}, {1}, 4));
  assert(!try_disjoint_roots(ptr(256), {3}, {-2}, 4, ptr(248), {4}, {1}, 4));
  assert(!try_disjoint_roots(ptr(256), {3}, {INT64_MAX}, 4, ptr(1024), {4}, {1}, 4));
}
"""
  )
  executable = tmp_path / "root_guard"
  subprocess.run(
    cpp_compile_command(source, executable, flags=("-std=c++20", "-O2")), check=True, capture_output=True, text=True
  )
  subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_native_bounded_signed_row_strides_and_repeated_destinations(tmp_path):
  N = T.dynamic(lower=0, upper=64)
  AS = T.dynamic(lower=-64, upper=64)
  CS = T.dynamic(lower=-64, upper=64)
  DS = T.dynamic(lower=-64, upper=64)

  @T.template
  def kernel(
    A: T.Tensor((N, 17), (AS, 1), T.float32),
    C: T.Tensor((N, 17), (CS, 1), T.float32, access="output"),
    D: T.Tensor((N, 17), (DS, 1), T.float32, access="output"),
    mode: int,
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      for row in T.serial(N - 1, -1, -1, unroll=1):
        T.store(C[row, :], T.load(A[row, :]) + mode)
        T.store(D[row, :], T.load(A[row, :]) * 3)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  if compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target.isa == "avx512":
    assert compiled.backend_ir.loop_versions
  block = 8192
  for rows in (0, 1, 3, 17, 33):
    for strides in ((-23, 31, -29), (23, -31, 29), (23, 0, 31), (-23, 31, 0)):
      for alias in (False, True):
        offsets = [block, 4 * block, 7 * block]
        actual_strides = list(strides)
        if alias:
          offsets[1] = offsets[0] + 1
          actual_strides[1] = actual_strides[0]
        for mode in (-1, 1):
          owner = np.linspace(0, 0.125, 9 * block, dtype=np.float32)
          expected = owner.copy()
          arrays = [
            np.ndarray((rows, 17), np.float32, buffer=owner, offset=p * 4, strides=(s * 4, 4))
            for p, s in zip(offsets, actual_strides)
          ]
          reference = [
            np.ndarray((rows, 17), np.float32, buffer=expected, offset=p * 4, strides=(s * 4, 4))
            for p, s in zip(offsets, actual_strides)
          ]
          for _ in range(3):
            for row in reversed(range(rows)):
              reference[1][row] = reference[0][row].copy() + np.float32(mode)
              reference[2][row] = reference[0][row].copy() * np.float32(3)
          compiled(*arrays, mode, 3)
          np.testing.assert_allclose(owner, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("repeat_loop", (False, True))
def test_native_historical_reader_crossing_versioned_loop(tmp_path, repeat_loop):
  N = T.dynamic(lower=0, upper=64)
  Repeated = repeat_loop

  @T.template
  def kernel(
    A: T.Tensor((N, 17), dtype=T.float32),
    C: T.Tensor((N, 17), dtype=T.float32, access="output"),
    D: T.Tensor((N, 17), dtype=T.float32, access="output"),
    repeat: int,
  ):
    before = T.load(A)
    if Repeated:
      for iteration in T.serial(repeat, unroll=1):
        for row in T.serial(N, unroll=1):
          T.store(C[row, :], T.load(A[row, :]) + 1)
    else:
      for row in T.serial(N, unroll=1):
        T.store(C[row, :], T.load(A[row, :]) + 1)
    T.store(D, before)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  if compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target.isa == "avx512":
    assert compiled.backend_ir.loop_versions
  for rows in (0, 1, 3, 33):
    block = rows * 17 + 64
    for c_offset, d_offset in ((block, 2 * block), (1, 2 * block), (block, 0), (1, 2)):
      for repeat in (0, 1, 3):
        owner = np.linspace(0, 0.125, 3 * block + 64, dtype=np.float32)
        expected = owner.copy()
        offsets = (0, c_offset, d_offset)
        arrays = [np.ndarray((rows, 17), np.float32, buffer=owner, offset=p * 4) for p in offsets]
        reference = [np.ndarray((rows, 17), np.float32, buffer=expected, offset=p * 4) for p in offsets]
        history = reference[0].copy()
        for _ in range(repeat if repeat_loop else 1):
          for row in range(rows):
            reference[1][row] = reference[0][row].copy() + np.float32(1)
        reference[2][:] = history
        compiled(*arrays, repeat)
        np.testing.assert_allclose(owner, expected, rtol=2e-6, atol=2e-6)


def test_native_zero_trip_and_safe_first_row_with_overflowing_root(tmp_path):
  K = T.dynamic(lower=0, upper=2)
  Stride = T.dynamic(lower=10**18, upper=10**18)

  @T.template
  def kernel(
    A: T.Tensor((64, 17), (Stride, 1), T.float32),
    C: T.Tensor((64, 17), (Stride, 1), T.float32, access="output"),
    Bound: T.Tensor((K, 1), dtype=T.float32),
  ):
    for row in T.serial(K, unroll=1):
      T.store(C[0, :], T.load(A[0, :]) + 1)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  if compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target.isa == "avx512":
    assert compiled.backend_ir.loop_versions
  for trips in (0, 1, 2):
    # Only the first row is addressable and only that row may execute. Root
    # metadata itself overflows a signed span, which must select fallback.
    a = np.linspace(0, 1, 17, dtype=np.float32)
    c = np.full(17, -99, dtype=np.float32)
    A = np.lib.stride_tricks.as_strided(a, shape=(64, 17), strides=(4 * 10**18, 4))
    C = np.lib.stride_tricks.as_strided(c, shape=(64, 17), strides=(4 * 10**18, 4))
    compiled(A, C, np.empty((trips, 1), dtype=np.float32))
    np.testing.assert_array_equal(c, a + np.float32(1) if trips else np.full(17, -99, dtype=np.float32))


@pytest.mark.parametrize("dtype", ("float64", "int16", "uint8"))
def test_native_element_widths_keep_wrapping_and_alias_order(tmp_path, dtype):
  N = T.dynamic(lower=0, upper=64)
  DType = dtype

  @T.template
  def kernel(
    A: T.Tensor((N, 17), dtype=DType),
    C: T.Tensor((N, 17), dtype=DType, access="output"),
    D: T.Tensor((N, 17), dtype=DType, access="output"),
  ):
    for row in T.serial(N, unroll=1):
      T.store(C[row, :], T.load(A[row, :]) + 1)
      T.store(D[row, :], T.load(A[row, :]) * 3)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  if compiled.backend_ir.vectors.manager.options["_compiler_environment"].codegen_target.isa == "avx512":
    assert compiled.backend_ir.loop_versions
  for rows in (0, 1, 3, 33):
    block = rows * 17 + 64
    for alias in (False, True):
      owner = np.arange(3 * block + 128, dtype=np.int64).astype(dtype)
      expected = owner.copy()
      offsets = (0, 1 if alias else block, 2 * block)
      width = owner.dtype.itemsize
      arrays = [np.ndarray((rows, 17), dtype, buffer=owner, offset=p * width) for p in offsets]
      reference = [np.ndarray((rows, 17), dtype, buffer=expected, offset=p * width) for p in offsets]
      for row in range(rows):
        reference[1][row] = reference[0][row].copy() + np.asarray(1, dtype=dtype)
        reference[2][row] = reference[0][row].copy() * np.asarray(3, dtype=dtype)
      compiled(*arrays)
      np.testing.assert_array_equal(owner, expected)


def test_shared_loop_intersects_normalized_root_origins():
  from dataclasses import fields

  artifact = CPUBackend().lower(declaration().ir, (), target="AVX512")
  plan = next(iter(artifact.backend_ir.loop_versions.values()))
  environment = artifact.backend_ir.vectors.manager.options["_compiler_environment"]
  origin = I.Expr(I.ScalarType("int64"), "local", data="root_base")

  def rewrite(value):
    if type(value) is I.View:
      assert value.region is not None
      return replace(
        value, offset=I.binary("add", value.offset, origin), region=replace(value.region, root_offset=origin)
      )
    if isinstance(value, I.Node):
      return replace(value, **{field.name: rewrite(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
      return tuple(rewrite(x) for x in value)
    if isinstance(value, list):
      return [rewrite(x) for x in value]
    return value

  loop = rewrite(plan.loop)

  def collect(statements):
    return collect_loop_versions(
      replace(artifact.kernel, body=I.Block(statements)), environment.operations, environment.codegen_target
    )

  first, second = I.Assign("root_base", I.literal(0)), I.Assign("root_base", I.literal(17))
  assert len(collect([first, loop])) == len(collect([second, loop])) == 1
  assert collect([first, loop, loop])[id(loop)].occurrences == 2
  assert not collect([first, loop, second, loop])


def test_bounded_numeric_metadata_capture_stays_inside_loop(tmp_path):
  N = T.dynamic(lower=0, upper=64)

  @T.template
  def kernel(A: T.Tensor((N, 17), dtype=T.float32), C: T.Tensor((N, 17), dtype=T.float32, access="output")):
    for row in T.serial(N, unroll=1):
      metadata = N * 3 + 17
      T.store(C[row, :], T.load(A[row, :]) + T.cast(metadata, "float32"))

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert len(artifact.backend_ir.loop_versions) == 1
  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for rows in (0, 1, 3, 33):
    for alias in (False, True):
      block = rows * 17 + 64
      owner = np.linspace(0, 0.125, 3 * block, dtype=np.float32)
      expected = owner.copy()
      offset = 1 if alias else block
      a = np.ndarray((rows, 17), np.float32, buffer=owner)
      c = np.ndarray((rows, 17), np.float32, buffer=owner, offset=offset * 4)
      reference_a = np.ndarray((rows, 17), np.float32, buffer=expected)
      reference_c = np.ndarray((rows, 17), np.float32, buffer=expected, offset=offset * 4)
      for row in range(rows):
        reference_c[row] = reference_a[row].copy() + np.float32(rows * 3 + 17)
      compiled(a, c)
      np.testing.assert_array_equal(owner, expected)


@pytest.mark.parametrize("width", (17, 33))
def test_native_nested_loops_keep_chunk_and_cross_row_alias_order(tmp_path, width):
  N = T.dynamic(lower=0, upper=64)
  Width = T.Const(width)

  @T.template
  def kernel(
    A: T.Tensor((N, Width), dtype=T.float32),
    C: T.Tensor((N, Width), dtype=T.float32, access="output"),
    D: T.Tensor((N, Width), dtype=T.float32, access="output"),
  ):
    for row in T.serial(N, unroll=1):
      for column in T.serial(0, Width - Width % 16, 16, unroll=1):
        T.store(C[row, column : column + 16], T.load(A[row, column : column + 16]) + 1)
        T.store(D[row, column : column + 16], T.load(A[row, column : column + 16]) * 2)
      T.store(C[row, Width - Width % 16 :], T.load(A[row, Width - Width % 16 :]) + 1)
      T.store(D[row, Width - Width % 16 :], T.load(A[row, Width - Width % 16 :]) * 2)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert artifact.backend_ir.loop_versions
  assert artifact.source.count("/* loop-invariant root alias version */") == 1
  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for rows in (0, 1, 3, 33):
    block = rows * width + 64
    for offsets in (
      (0, block, 2 * block),
      (0, 1, 2 * block),
      (0, width, 2 * block),
      (width, 0, 2 * block),
      (0, block, block + 1),
      (0, 1, 2),
    ):
      owner = np.linspace(0, 0.125, 3 * block + 128, dtype=np.float32)
      expected = owner.copy()
      arrays = [np.ndarray((rows, width), np.float32, buffer=owner, offset=p * 4) for p in offsets]
      reference = [np.ndarray((rows, width), np.float32, buffer=expected, offset=p * 4) for p in offsets]
      for row in range(rows):
        for begin in range(0, width, 16):
          end = min(begin + 16, width)
          reference[1][row, begin:end] = reference[0][row, begin:end].copy() + np.float32(1)
          reference[2][row, begin:end] = reference[0][row, begin:end].copy() * np.float32(2)
      compiled(*arrays)
      np.testing.assert_array_equal(owner, expected)


@pytest.mark.parametrize("step", (-2, 2))
def test_native_nonunit_steps_preserve_unvisited_rows(tmp_path, step):
  N = T.dynamic(lower=0, upper=64)
  Step = T.Const(step)

  @T.template
  def kernel(A: T.Tensor((N, 17), dtype=T.float32), C: T.Tensor((N, 17), dtype=T.float32, access="output")):
    if Step > 0:
      for row in T.serial(0, N, Step, unroll=1):
        T.store(C[row, :], T.load(A[row, :]) + 1)
    else:
      for row in T.serial(N - 1, -1, Step, unroll=1):
        T.store(C[row, :], T.load(A[row, :]) + 1)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert artifact.backend_ir.loop_versions
  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for rows in (0, 1, 2, 3, 33):
    for alias in (False, True):
      block = rows * 17 + 64
      owner = np.linspace(0, 0.125, 3 * block, dtype=np.float32)
      expected = owner.copy()
      offset = 1 if alias else block
      a = np.ndarray((rows, 17), np.float32, buffer=owner)
      c = np.ndarray((rows, 17), np.float32, buffer=owner, offset=offset * 4)
      reference_a = np.ndarray((rows, 17), np.float32, buffer=expected)
      reference_c = np.ndarray((rows, 17), np.float32, buffer=expected, offset=offset * 4)
      for row in range(0, rows, step) if step > 0 else range(rows - 1, -1, step):
        reference_c[row] = reference_a[row].copy() + np.float32(1)
      compiled(a, c)
      np.testing.assert_array_equal(owner, expected)
