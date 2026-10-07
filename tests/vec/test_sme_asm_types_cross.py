"""Real SME asm operands stay sizeless across the public VLS boundary.

This standalone cross test deliberately does not import vecops/_C. It compiles
the public-header probe and inspects complete instruction-owner functions. VLS
cases guarantee equal ordinary/streaming lengths; unequal cases are VLA only.
No target execution or performance is implied by these checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from arm_sme_asm_utils import assert_sme_owners


def _sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _toolchain():
  cxx = os.environ.get("VECOPS_TEST_ARM_CXX")
  sysroot = os.environ.get("VECOPS_TEST_ARM_SYSROOT")
  if cxx is None or sysroot is None:
    pytest.skip("explicit ARM cross compiler and sysroot are required")
  cxx, sysroot = Path(cxx), Path(sysroot)
  assert cxx.is_file() and sysroot.is_dir(), (cxx, sysroot)
  generic = os.environ.get("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE")
  target = os.environ.get("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE")
  assert (generic is None) == (target is None), "partial C++ header override"
  generic = Path(generic) if generic is not None else sysroot / "include/c++/13"
  target = Path(target) if target is not None else generic / "aarch64-linux-gnu"
  assert generic.is_dir() and target.is_dir(), (generic, target)
  includes = ["-nostdinc++", "-isystem", str(generic), "-isystem", str(target)]
  if (generic / "backward").is_dir():
    includes.extend(("-isystem", str(generic / "backward")))
  return cxx, sysroot, includes


def _assert_owners(assembly, expected, *, equal_domains):
  owners = assert_sme_owners(assembly, expected, equal_domains=equal_domains)
  for name, body in owners.items():
    # VLS sized predicate bridges may use stack temporaries in current LLVM.
    # Compilation/inlining is the contract checked here, not spill-free VLS.
    assert "za" in body and re.search(r"^\s*(?:zero|mov|mova|\w*mopa|addha|ld1\w|st1\w)\b", body, re.MULTILINE), name
  return owners


@pytest.mark.parametrize("include_kind", ("source", "sdk"))
@pytest.mark.parametrize(
  "bits,vl,svl",
  (
    (128, 16, 16),
    (512, 64, 64),
    ("scalable", 16, 16),
    ("scalable", 64, 64),
    ("scalable", 16, 64),
    ("scalable", 64, 16),
  ),
)
def test_sme_public_asm_types_real_cross(tmp_path, include_kind, bits, vl, svl):
  cxx, sysroot, cxx_includes = _toolchain()
  root = Path(__file__).resolve().parents[2]
  if include_kind == "source":
    include = root / "include"
  else:
    sdk = os.environ.get("VECOPS_TEST_SME_ASM_SDK")
    if sdk is None:
      pytest.skip("private matching SDK must be explicitly selected")
    include = Path(sdk) / "include"
  header = include / "vecops/vec/details/sme/ZA.h"
  assert header.is_file()
  assert header.read_bytes() == (root / "include/vecops/vec/details/sme/ZA.h").read_bytes()
  probe = Path(__file__).with_name("SMEAsmTypesProbe.cpp")
  copied = tmp_path / "probe.cpp"
  copied.write_bytes(probe.read_bytes())
  assembly, elf = tmp_path / "probe.s", tmp_path / "probe.o"
  flags = [
    str(cxx),
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    f"-msve-vector-bits={bits}",
    "-std=c++20",
    "-O3",
    "-DNDEBUG",
    "-ffp-contract=off",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(include),
    "-I",
    str(include / "vecops"),
    *cxx_includes,
  ]
  if bits != "scalable":
    assert vl == svl == bits // 8
    flags.append(f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={bits}")
  commands = [flags + ["-S", str(copied), "-o", str(assembly)], flags + ["-c", str(copied), "-o", str(elf)]]
  evidence = {
    "commands": commands,
    "include_kind": include_kind,
    "bits": bits,
    "vl": vl,
    "svl": svl,
    "compiler_sha256": _sha(cxx),
    "probe_sha256": _sha(copied),
    "header_sha256": _sha(header),
  }
  (tmp_path / "commands.json").write_text(json.dumps(evidence, indent=2) + "\n")
  logs = []
  for command in commands:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    logs.append({"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    (tmp_path / "compiler.json").write_text(json.dumps(logs, indent=2) + "\n")
    assert result.returncode == 0, result.stderr
  assert elf.read_bytes()[:4] == b"\x7fELF"
  owners = _assert_owners(assembly.read_text(), 21, equal_domains=vl == svl)
  instructions = assembly.read_text()
  for mnemonic in ("fmopa", "bfmopa", "smopa", "umopa", "sumopa", "usmopa", "addha"):
    assert re.search(rf"^\s*{mnemonic}\b", instructions, re.MULTILINE), mnemonic
  objdump = cxx.with_name("llvm-objdump")
  result = subprocess.run([str(objdump), "-d", str(elf)], capture_output=True, text=True, check=False)
  (tmp_path / "probe.dump").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr
  evidence.update({"elf_sha256": _sha(elf), "assembly_sha256": _sha(assembly), "owners": tuple(owners)})
  (tmp_path / "identity.json").write_text(json.dumps(evidence, indent=2) + "\n")


def test_owner_check_rejects_active_calls_or_missing_state():
  prefix = ".type with_streaming_za_probe,@function\nsmstart\nzero {za}\n"
  suffix = "smstop\nret\n.size with_streaming_za_probe,0\n"
  _assert_owners(prefix + suffix, 1, equal_domains=True)
  for bad in ("bl helper\n", "blr x0\n"):
    with pytest.raises(AssertionError):
      _assert_owners(prefix + bad + suffix, 1, equal_domains=True)
  with pytest.raises(AssertionError):
    _assert_owners(prefix + suffix.replace("smstop\n", ""), 1, equal_domains=True)
  # Explicitly permit mandatory ABI preservation outside the SM region.
  _assert_owners(prefix.replace("smstart", "str z1, [sp]\nsmstart") + suffix, 1, equal_domains=True)
