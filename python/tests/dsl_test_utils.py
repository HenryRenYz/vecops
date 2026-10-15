# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared compiler selection for executable DSL tests.

The defaults match the historical test configuration. ARM test jobs can set
``CC``, ``CXX`` and ``VECOPS_TEST_TARGET`` to select their native toolchain.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from pathlib import Path

import vecops


def make_compiler(*, target="Native", **kwargs):
  """Build a test compiler, honoring explicit CI toolchain/target settings."""
  requested_target = os.environ.get("VECOPS_TEST_TARGET")
  if requested_target and target in {"Native", "AVX512"}:
    target = requested_target
  cc = os.environ.get("CC", kwargs.pop("cc", "gcc"))
  cxx = os.environ.get("CXX", kwargs.pop("cxx", "g++"))
  return vecops.Compiler(target=target, cc=cc, cxx=cxx, **kwargs)


def cpp_compile_command(source, output, *, cxx=None, flags=(), link_flags=()):
  """Build argv for standalone test C++ compilation using the active toolchain."""
  compiler = cxx or os.environ.get("CXX", "g++")
  cxxflags = shlex.split(os.environ.get("CXXFLAGS", ""))
  ldflags = shlex.split(os.environ.get("LDFLAGS", ""))
  return [compiler, *cxxflags, *flags, str(source), *ldflags, "-o", str(output)]


def arm_cross_cxx_include_flags(sysroot):
  """Select one explicit C++ header family for real ARM codegen tests.

  The historical GCC13 cross sysroot remains the default. Allocated ARM
  jobs may instead provide the private GCC14 generic/target directories
  through ``VECOPS_TEST_ARM_CROSS_CXX_INCLUDE`` and
  ``VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE``. Missing or partial overrides
  fail before compilation, rather than silently mixing system headers.
  """
  generic = os.environ.get("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE")
  target = os.environ.get("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE")
  if (generic is None) != (target is None):
    raise ValueError("ARM cross C++ include overrides must specify both generic and target directories")
  generic = Path(generic) if generic is not None else Path(sysroot) / "include/c++/13"
  target = Path(target) if target is not None else generic / "aarch64-linux-gnu"
  for path in (generic, target):
    if not path.is_dir():
      raise ValueError(f"ARM cross C++ include directory is missing: {path}")
  flags = ["-nostdinc++", "-isystem", str(generic), "-isystem", str(target)]
  backward = generic / "backward"
  if backward.is_dir():
    flags.extend(("-isystem", str(backward)))
  return flags


def arm_objdump_command(object_file):
  """Disassemble ARM objects on cross hosts and native ARM test machines."""
  tool = (
    os.environ.get("VECOPS_TEST_ARM_OBJDUMP")
    or shutil.which("aarch64-linux-gnu-objdump")
    or shutil.which("llvm-objdump")
  )
  if tool is None:
    raise FileNotFoundError("set VECOPS_TEST_ARM_OBJDUMP to an ARM-capable objdump")
  features = ["--mattr=+sve,+sme,+sme-fa64,+sme-f64f64,+bf16"] if "llvm-objdump" in Path(tool).name else []
  return [tool, *features, "-Cd", str(object_file)]


def amx_available_for_test() -> bool:
  """Whether the selected test target is the x86 AMX-capable path."""
  if not amx_tile_available_for_test():
    return False
  try:
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8")
  except OSError:
    return False
  return "amx_bf16" in features or "amx_int8" in features


def amx_tile_available_for_test() -> bool:
  """Whether explicit AMX-TILE tests match the selected x86 machine/target."""
  selected = os.environ.get("VECOPS_TEST_TARGET")
  if selected and selected.lower() not in {"native", "avx512", "amx"}:
    return False
  if os.uname().machine.lower() not in {"x86_64", "amd64"}:
    return False
  try:
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8")
  except OSError:
    return False
  return "amx_tile" in features


def x86_target_available_for_test() -> bool:
  """Whether x86-specific test targets match the selected test machine."""
  if os.uname().machine.lower() not in {"x86_64", "amd64"}:
    return False
  selected = os.environ.get("VECOPS_TEST_TARGET")
  if selected:
    return selected.lower() in {"native", "avx512", "avx2", "amx"}
  return True


def assert_no_vec_mask_arrays(source: str) -> None:
  """Enforce the DSL invariant against std::array/vector of register values."""
  assert not re.search(r"std::(?:array|vector)\s*<\s*vecops::vec::(?:Vec|Mask)", source)


def assert_source_algorithms_lowered(prepared) -> None:
  """Require terminal executable IR rather than obsolete empty planner fields."""
  import copy
  from dataclasses import replace

  from vecops.language._compiler.backends.cpu.transpose import TransposePlan
  from vecops.language._compiler.ir.vector import VectorRegion
  from vecops.language._compiler.ir.verification import verify_prepared

  # The emitter appends the CPU transpose schedules it actually used to the
  # debug region list. The phase-boundary verifier runs before that emission.
  # Reconstruct its vector-only view while checking the debug additions are
  # known CPU schedules, rather than treating them as portable vector IR.
  vectors = copy.copy(prepared.vectors)
  assert all(isinstance(region, (VectorRegion, TransposePlan)) for region in vectors.regions)
  vectors.regions = [region for region in vectors.regions if isinstance(region, VectorRegion)]

  verify_prepared(replace(prepared, vectors=vectors))


def parameter_snapshot_name(source: str, expression: str) -> str:
  """Resolve an ABI entry local without hardcoding generated suffixes."""
  matches = re.findall(rf"const auto (dsl_c_parameter_\w+) = {re.escape(expression)};", source)
  assert len(matches) == 1, f"expected one entry snapshot for {expression!r}"
  assert source.count(expression) == 1, f"ABI metadata {expression!r} must be read once"
  return matches[0]


def assert_full_sve_memory_words(body: str, *, kind: str, words: int, bits="scalable") -> None:
  """Check complete physical words independently of LD1H/LD1B spelling.

  Clang may express a bit-preserving full-word access using bytes or words.
  Count actual accesses, and check the reaching predicate definition at each
  access: its granularity and explicit VL must still cover one whole word.
  A full finer-grained predicate also covers every coarser-grained lane
  (PTRUE .B can govern LD1H/LD1W); the reverse is not true.
  A later predicate overwrite must not retroactively validate an earlier load.
  This checks access width, not pointer arithmetic or numerical lane order;
  callers retain their conversion, guard-page and numerical assertions.
  """
  assert kind in {"ld1", "st1"}
  predicates = {}
  accesses = 0
  widths = {"b": 8, "h": 16, "s": 32, "d": 64}
  for raw in body.splitlines():
    line = raw.split("//", 1)[0].strip()
    writer = re.match(r"(\w+)\s+(p\d+)(?:\.([bhsd]))?\b(.*)", line)
    if writer:
      operation, predicate, element, suffix = writer.groups()
      predicates[predicate] = (element, suffix.strip()) if operation == "ptrue" else None
    if not re.match(rf"{kind}[bhwd]\b", line):
      continue
    match = re.fullmatch(rf"{kind}([bhwd])\s+\{{\s*z\d+\.([bhsd])\s*\}},\s*(p\d+)(/z)?,\s*\[[^\]]+\]", line)
    assert match, line
    memory, element, predicate, zero = match.groups()
    assert element == {"b": "b", "h": "h", "w": "s", "d": "d"}[memory], line
    assert (kind == "ld1") == (zero == "/z"), line
    definition = predicates.get(predicate)
    assert definition is not None and definition[0] in widths, (line, definition)
    assert widths[definition[0]] <= widths[element], (line, definition)
    suffix = definition[1]
    expected = "" if bits == "scalable" else f", vl{int(bits) // widths[definition[0]]}"
    assert suffix == "" or bits != "scalable" and suffix == expected, (line, definition, bits)
    accesses += 1
  assert accesses == words, (kind, accesses, words, body)
