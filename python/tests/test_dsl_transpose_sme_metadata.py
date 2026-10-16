# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""ZA slice loops consume owner-local geometry, not escaped metadata refs."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import (
  arm_cross_cxx_include_flags,
  assert_no_vec_mask_arrays,
  make_compiler,
  parameter_snapshot_name,
)
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

TYPES = (
  ("float32", "float32", "float32"),
  ("float64", "float64", "float64"),
  ("float16", "float32", "float32"),
  ("float32", "float16", "float32"),
)


def transpose_template(source="float32", output="float32", compute="float32", *, batch=False, scope=False):
  m, n, b = (T.dynamic(lower=0) for _ in range(3))
  rank = 3 if batch else 2
  source_shape = (b, m, n) if batch else (m, n)
  output_shape = (b, n, m) if batch else (n, m)
  source_strides = tuple(T.dynamic() for _ in range(rank))
  output_strides = tuple(T.dynamic() for _ in range(rank))

  @T.template
  def kernel(
    A: T.Tensor(source_shape, source_strides, source),
    C: T.Tensor(output_shape, output_strides, output, access="output"),
  ):
    if scope:
      with T.sme.region():
        T.store(C, T.load(T.transpose(A), dtype=compute), implementation="sme")
    elif batch:
      T.store(C, T.load(T.transpose(A, axes=(0, 2, 1)), dtype=compute))
    else:
      T.store(C, T.load(T.transpose(A), dtype=compute))

  return kernel


def owned_template():
  m, n = T.dynamic(lower=1), T.dynamic(lower=1)

  @T.template
  def kernel(
    A: T.Tensor((m, n), dtype="float32"),
    C: T.Tensor((n, m), dtype="float32", access="output"),
  ):
    for iteration in T.serial(2, unroll=1):
      temporary = T.alloc((m * n + iteration,), dtype="float32")
      T.store(temporary, -7)
      temporary[0] = 0  # Force real scratch, not promoted producer forwarding.
      matrix = T.reshape(temporary[: m * n], (m, n))
      T.store(matrix, T.load(A))
      T.transpose_copy(matrix, C)

  return kernel


def lower(template, *, vl=64, svl=64, metadata=True, dynamic_unroll=False):
  return CPUBackend().lower(
    template.ir,
    (),
    target="SME",
    options={
      "_vector_state": VectorState(vl, svl),
      "cpu_parameter_metadata_snapshot": metadata,
      "cpu_transpose_sme_dynamic_unroll": dynamic_unroll,
    },
  )


def sme_body(source):
  # There is one prepared SME alternative. Stop before its generic/register
  # fallback, whose descriptor/snapshot accesses are intentionally unchanged.
  start = source.index("const vecops::nint_t dsl_c_transpose_extent_m_")
  return source[start : source.index("});", start)]


@pytest.mark.parametrize("source,output,compute", TYPES)
@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("metadata", (False, True))
def test_transpose_owner_snapshots_exact_geometry_before_za_loops(source, output, compute, vl, svl, metadata):
  artifact = lower(transpose_template(source, output, compute), vl=vl, svl=svl, metadata=metadata)
  body = sme_body(artifact.source)
  declarations = re.findall(r"const (?:auto|vecops::nint_t) (dsl_c_transpose_\w+_\d+)=(.+);", body)
  assert len(declarations) == len({name for name, _ in declarations}) == 8
  assert all(body.count(f"{name}=") == 1 for name, _ in declarations)
  strides = {name: value for name, value in declarations if "_stride_" in name}
  assert len(strides) == 4
  for name, value in strides.items():
    if "source_n" in name or "target_m" in name:
      assert value == "std::int64_t{1}"
    else:
      storage = "A" if "source_m" in name else "C"
      descriptor = f"{storage}.stride(0)"
      if metadata:
        descriptor = parameter_snapshot_name(artifact.source, f"{storage}.stride(0)")
      assert value == descriptor
  origins = {name: value for name, value in declarations if "_base_" in name}
  assert len(origins) == 2
  for name, value in origins.items():
    assert re.search(r"transpose_extent_m_\d+>0 && dsl_c_transpose_extent_n_\d+>0 \?", value)
    storage = "A" if "source" in name else "C"
    descriptor = f"{storage}.data()"
    if metadata:
      # Data pointers are captured only when an outlined helper needs them.
      # The SME owner still binds its exact input/output origin before loops.
      captures = re.findall(rf"const auto (\w+) = {storage}\.data\(\);", artifact.source)
      assert len(captures) <= 1
      if captures:
        descriptor = captures[0]
    assert descriptor in value
  loops = body[body.index("for (vecops::nint_t") :]
  assert "dsl_c_parameter_" not in loops
  assert not re.search(r"\b[AC]\.(?:data|size|stride)\(", loops)
  reads = [line for line in loops.splitlines() if "load_hor<" in line or "load_convert(" in line]
  writes = [line for line in loops.splitlines() if "store_ver<" in line or "store_convert(" in line]
  assert reads and writes
  input_pointer = re.search(r"auto\* const (\w+) = dsl_c_transpose_source_base_\d+;", body)[1]
  output_pointer = re.search(r"auto\* const (\w+) = dsl_c_transpose_target_base_\d+;", body)[1]
  assert all(input_pointer in line for line in reads)
  assert all(output_pointer in line for line in writes)
  assert f"std::int64_t{{{svl // (8 if compute == 'float64' else 4)}}}" in body
  assert "vecops_dsl::vector_size" not in body  # Never substitute ordinary VL.
  assert "/* overlapping operands */" in artifact.source
  assert "/* unsupported strides: preserve gather/scatter */" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def test_existing_explicit_sme_scope_reuses_owner_without_new_closure():
  artifact = lower(transpose_template(scope=True), vl=16, svl=64)
  assert artifact.source.count(".with_resources(") == 1
  body = sme_body(artifact.source)
  assert "transpose_source_base_" in body and "transpose_target_base_" in body
  assert "dsl_c_parameter_" not in body[body.index("for (vecops::nint_t") :]
  assert "explicit SME region requires unit column strides" in artifact.source


def test_batch_origins_include_offset_and_leading_coordinate_once():
  artifact = lower(transpose_template(batch=True))
  body = sme_body(artifact.source)
  origins = re.findall(r"const auto (dsl_c_transpose_\w+_base_\d+)=(.+);", body)
  assert len(origins) == 2
  for name, value in origins:
    storage = "A" if "source" in name else "C"
    stride = parameter_snapshot_name(artifact.source, f"{storage}.stride(0)")
    assert "transpose_batch_" in value and stride in value
  loops = body[body.index("for (vecops::nint_t") :]
  assert "transpose_batch_" not in loops
  assert_no_vec_mask_arrays(artifact.source)


def test_owned_storage_origin_is_read_fresh_after_each_resize():
  source = lower(owned_template()).source
  storage = re.search(r"vecops_dsl::Buffer<float> (dsl_f_\w+);", source)[1]
  body = sme_body(source)
  origin = re.search(r"const auto (dsl_c_transpose_source_base_\d+)=(.+);", body)
  assert storage + ".data()" in origin[2]
  assert source.index(storage + ".resize(") < source.index(origin[0])
  assert storage + ".data()" not in body[body.index("for (vecops::nint_t") :]
  assert "dsl_f_iteration_" in source
  assert_no_vec_mask_arrays(source)


def test_dynamic_pitch_unroll_hint_is_opt_in_and_does_not_change_constant_pitches():
  plain = lower(transpose_template()).source
  hinted = lower(transpose_template(), dynamic_unroll=True).source
  assert "#pragma clang loop unroll_count(2)" not in plain
  assert hinted.count("#pragma clang loop unroll_count(2)") == 2
  assert hinted.count("#pragma GCC unroll 2") == 2

  @T.template
  def constant(A: T.Tensor((17, 19), dtype="float64"), C: T.Tensor((19, 17), dtype="float64", access="output")):
    T.transpose_copy(A, C, implementation="sme")

  assert "#pragma clang loop unroll_count(2)" not in lower(constant, dynamic_unroll=True).source
  for invalid in (None, 0, 2, "true"):
    with pytest.raises(TypeError, match="cpu_transpose_sme_dynamic_unroll must be bool"):
      lower(transpose_template(), dynamic_unroll=invalid)


def cross_wrapper(source, source_dtype, output_dtype):
  return (
    "#include <vecops/Kernel.h>\n"
    "namespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }\n"
    + source
    + '\nextern "C" void transpose_ssa(void* ap,void* cp,vecops::nint_t m,vecops::nint_t n,'
    "vecops::nint_t as0,vecops::nint_t as1,vecops::nint_t cs0,vecops::nint_t cs1) {\n"
    + f"auto A=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[source_dtype]}*>(ap),"
    "vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n}),"
    "vecops::tensor::make_strides(vecops::meta::Any{as0},vecops::meta::Any{as1}));\n"
    + f"auto C=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[output_dtype]}*>(cp),"
    "vecops::tensor::make_shape(vecops::meta::Any{n},vecops::meta::Any{m}),"
    "vecops::tensor::make_strides(vecops::meta::Any{cs0},vecops::meta::Any{cs1}));\n"
    "__kernel__(A,C);\n}\n"
  )


def _assembly_functions(assembly):
  starts = list(re.finditer(r"^\s*\.type\s+[^,\n]+,\s*@function\b", assembly, re.MULTILINE))
  for index, start in enumerate(starts):
    end = starts[index + 1].start() if index + 1 < len(starts) else len(assembly)
    yield assembly[start.start() : end]


def _natural_za_loops(function):
  """Recover natural ZA loops from branch CFG dominance, not text direction."""
  instructions, labels = [], {}
  for raw in function.splitlines():
    label = re.match(r"^\s*(\.L[\w.]+):", raw)
    if label:
      labels[label[1]] = len(instructions)
      continue
    line = raw.split("//", 1)[0].strip()
    if line and not line.startswith("."):
      instructions.append(line)
  if not instructions:
    return []

  successors = [set() for _ in instructions]
  for index, line in enumerate(instructions):
    parts = line.split(None, 1)
    operation = parts[0]
    operands = parts[1] if len(parts) == 2 else ""
    target = operands.split(",")[-1].strip()
    destination = labels.get(target)
    if operation == "b":
      if destination is not None:
        successors[index].add(destination)
    elif operation.startswith("b.") or operation in {"cbz", "cbnz", "tbz", "tbnz"}:
      if destination is not None:
        successors[index].add(destination)
      if index + 1 < len(instructions):
        successors[index].add(index + 1)
    elif operation not in {"ret", "br", "brk", "eret"} and index + 1 < len(instructions):
      successors[index].add(index + 1)

  predecessors = [set() for _ in instructions]
  reachable, pending = {0}, [0]
  while pending:
    source = pending.pop()
    for target in successors[source]:
      predecessors[target].add(source)
      if target not in reachable:
        reachable.add(target)
        pending.append(target)

  dominators = {node: ({node} if node == 0 else set(reachable)) for node in reachable}
  changed = True
  while changed:
    changed = False
    for node in reachable - {0}:
      incoming = [dominators[pred] for pred in predecessors[node] if pred in reachable]
      common = set.intersection(*incoming) if incoming else set()
      updated = {node} | common
      if updated != dominators[node]:
        dominators[node] = updated
        changed = True

  natural = {}
  for latch in reachable:
    for header in successors[latch]:
      if header not in dominators[latch]:
        continue
      loop_nodes, pending = {header, latch}, [latch]
      while pending:
        node = pending.pop()
        for pred in predecessors[node]:
          if pred in reachable and pred not in loop_nodes:
            loop_nodes.add(pred)
            if pred != header:
              pending.append(pred)
      if any(re.search(r"\bza\d+[hv]\.", instructions[node]) for node in loop_nodes):
        natural[(header, latch)] = loop_nodes
  return [
    {
      "header": header,
      "latch": latch,
      "nodes": nodes,
      "body": [instructions[node] for node in sorted(nodes)],
    }
    for (header, latch), nodes in natural.items()
  ]


def _sme_owner_loops(assembly):
  owners = [
    function
    for function in _assembly_functions(assembly)
    if re.search(r"^\s*smstart\b", function, re.MULTILINE)
    and re.search(r"^\s*smstop\b", function, re.MULTILINE)
    and re.search(r"\bza\d+[hv]\.", function)
  ]
  assert len(owners) == 1, f"expected one streamed ZA owner, got {len(owners)}"
  return owners[0], _natural_za_loops(owners[0])


def _innermost_za_loops(loops):
  return [loop for loop in loops if not any(loop["nodes"] > inner["nodes"] for inner in loops)]


def test_natural_za_loop_analysis_excludes_non_dominating_backedges_and_scopes_functions():
  assembly = """
  .type owner, @function
  .LBB0_0:
    b .LBB0_1
  .LBB0_1:
    ldr x9, [sp, #8]
  .LBB0_2:
    ld1w {za0h.s[w12, 0]}, p0/z, [x1]
    b.ne .LBB0_2
  .LBB0_3:
    b.gt .LBB0_1
    ret
  .type alternate, @function
  .LBB0_0:
    cmp x0, #0
    b.eq .LBB0_2
  .LBB0_1:
    b .LBB0_2
  .LBB0_2:
    ld1w {za0h.s[w12, 0]}, p0/z, [x1]
    b.gt .LBB0_1
    ret
  .type row_spill, @function
  .LBB0_0:
    b .LBB0_1
  .LBB0_1:
    ldr x10, [sp, #24]
    ld1w {za0h.s[w12, 0]}, p0/z, [x1]
    b.ne .LBB0_1
    ret
  .type other, @function
  .LBB0_0:
    ret
  """
  functions = list(_assembly_functions(assembly))
  assert len(functions) == 4
  owner_loops = _natural_za_loops(functions[0])
  alternate_loops = _natural_za_loops(functions[1])
  row_spill_loops = _natural_za_loops(functions[2])
  assert len(owner_loops) == 2  # One outer tile and its nested ZA row loop.
  inner_loops = _innermost_za_loops(owner_loops)
  assert len(inner_loops) == 1
  assert not any(re.match(r"\s*(?:ldr|ldp)\s+[xw]", line) for line in inner_loops[0]["body"])
  assert any(
    re.match(r"\s*(?:ldr|ldp)\s+[xw]", line) for loop in owner_loops for line in loop["body"]
  )  # The outer tile may spill its advancement geometry.
  assert not alternate_loops  # The backward edge does not have a dominating header.
  assert len(_innermost_za_loops(row_spill_loops)) == 1
  assert any(
    re.match(r"\s*(?:ldr|ldp)\s+[xw]", line) for line in _innermost_za_loops(row_spill_loops)[0]["body"]
  )  # A genuine inner-row stack reload must remain detectable.
  assert not _natural_za_loops(functions[3])  # Reused .LBB names cannot leak across functions.


@pytest.mark.parametrize("source,output,compute", TYPES)
@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("dynamic_unroll", (False, True))
def test_cross_za_hot_loops_do_not_reload_scalar_metadata(tmp_path, source, output, compute, vl, svl, dynamic_unroll):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real FA64 cross-codegen checks")
  root = Path(__file__).resolve().parents[2]
  cpp = tmp_path / "transpose.cpp"
  cpp.write_text(
    cross_wrapper(
      lower(transpose_template(source, output, compute), vl=vl, svl=svl, dynamic_unroll=dynamic_unroll).source,
      source,
      output,
    )
  )
  assembly = cpp.with_suffix(".s")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-msve-vector-bits=scalable",
    "-std=c++20",
    "-O3",
    "-S",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr[-6000:]
  text = assembly.read_text()
  assert re.search(r"^\s*smstart\b", text, re.MULTILINE)
  assert re.search(r"^\s*smstop\b", text, re.MULTILINE)
  function, loops = _sme_owner_loops(text)
  assert loops, "no real ZA transfer back edge found"

  def scalar_loads(block):
    return [line for line in block.splitlines() if re.match(r"\s*(?:ldr|ldp)\s+[xw]", line)]

  scalar_reads = [line for loop in loops for line in loop["body"] if scalar_loads(line)]
  # A stack spill from LLVM's native auto-unroll is a register-pressure issue,
  # not reloading the reference-captured closure after each ZA memory clobber.
  assert all(re.search(r"\[sp(?:,|\])", line) for line in scalar_reads), "ZA loop still reloads closure/ABI metadata"
  if dynamic_unroll:
    # The hint bounds the slice loops, not both outer tile loops. LLVM may
    # spill the latter's advancement geometry at high register pressure, but
    # the innermost LD1/ST1 slice back edges must keep their pitches in GPRs.
    # Do not misclassify an outer [sp] reload as an escaped closure reference.
    inner_loops = _innermost_za_loops(loops)
    assert inner_loops
    assert not any(scalar_loads(line) for loop in inner_loops for line in loop["body"]), (
      "bounded natural ZA slice loop reloads scalar geometry"
    )
  if source == compute:
    assert re.search(r"\bld1\w*\s+\{za\d+h\.", function)
  if output == compute:
    assert re.search(r"\bst1\w*\s+\{za\d+v\.", function)


@pytest.fixture(scope="module")
def native_transposes(tmp_path_factory):
  directory = tmp_path_factory.mktemp("sme-transpose-metadata")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for types in TYPES:
      for batch in (False, True):
        compiled = T.compile(transpose_template(*types, batch=batch), compiler=compiler)
        a_shape = (3, 17, 19) if batch else (17, 19)
        c_shape = (3, 19, 17) if batch else (19, 17)
        compiled.operator(
          vecops.TensorMeta(a_shape, dtype=types[0]),
          vecops.TensorMeta(c_shape, dtype=types[1]),
          **compiled._core.specialization_values,
        )
        kernels[types, batch] = compiled
    owned = T.compile(owned_template(), compiler=compiler)
    owned.operator(
      vecops.TensorMeta((17, 19), dtype="float32"),
      vecops.TensorMeta((19, 17), dtype="float32"),
      **owned._core.specialization_values,
    )
    kernels["owned"] = owned
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("source,output,compute", TYPES)
@pytest.mark.parametrize("batch", (False, True))
def test_native_transpose_offsets_tails_aliases_and_signed_layouts(native_transposes, source, output, compute, batch):
  compiled = native_transposes[(source, output, compute), batch]
  for m, n in ((0, 7), (7, 0), (1, 1), (3, 5), (17, 19), (37, 23)):
    shape = (3, m, n) if batch else (m, n)
    output_shape = (3, n, m) if batch else (n, m)
    axes = (0, 2, 1) if batch else (1, 0)
    count = int(np.prod(shape))
    original = ((np.arange(count).reshape(shape) % 127 - 63) / 16).astype(source)
    for layout in ("offset", "negative-input-rows", "negative-output-rows", "strided-output", "shared-storage"):
      # Nonzero offsets in the caller's view and distinct batch strides are
      # included in each fresh region origin, not in later slice addresses.
      backing = np.empty((*shape[:-2], m + 2, n + 4), dtype=source)
      a = backing[..., 1 : m + 1, 2 : n + 2]
      a[...] = original
      if layout == "negative-input-rows":
        a = a[..., ::-1, :]
      c_backing = np.empty((*output_shape[:-2], n + 2, m * 2 + 4), dtype=output)
      c = c_backing[..., 1 : n + 1, 2 : m * 2 + 2 : 2]
      if layout != "strided-output":
        c = c_backing[..., 1 : n + 1, 1 : m + 1]
      if layout == "negative-output-rows":
        c = c[..., ::-1, :]
      elif layout == "shared-storage":
        storage = np.empty(max(1, count * max(np.dtype(source).itemsize, np.dtype(output).itemsize)), dtype=np.uint8)
        a = np.ndarray(shape, dtype=source, buffer=storage)
        a[...] = original
        c = np.ndarray(output_shape, dtype=output, buffer=storage)
      expected = a.astype(compute).transpose(axes).astype(output)
      compiled.operator.native(a, c, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
  assert_no_vec_mask_arrays(compiled.generated_source)


def test_native_owned_source_pointer_tracks_resize_each_invocation(native_transposes):
  compiled = native_transposes["owned"]
  for shape in ((3, 5), (17, 19), (2, 1), (37, 23)):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    c = np.empty(shape[::-1], dtype=np.float32)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a.T)
