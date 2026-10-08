# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-tree and relocatable SDK SME leaf-library construction contracts."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _cmake_compilers():
  """Prefer the explicitly selected test toolchain over PATH defaults."""
  c_compiler = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
  cxx_compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
  return c_compiler, cxx_compiler


def _cmake_flags():
  """Forward private standard-library and linker flags to the CMake probe."""
  cxxflags = os.environ.get("CXXFLAGS", "")
  ldflags = os.environ.get("LDFLAGS", "")
  return [f"-DCMAKE_CXX_FLAGS={cxxflags}", f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}"]


def test_source_adapter_checks_sve_and_sme_independently():
  generator = (ROOT / "src/compiler/KernelCompiler.cpp").read_text()
  assert "#if defined(VECOPS_KERNEL_SVE_BYTES) || defined(VECOPS_KERNEL_SME_BYTES)" in generator
  assert "{kernel_sve_bytes, kernel_sme_bytes}" in generator
  assert "constexpr auto kernel_sve_bytes = 0;" in generator
  assert "constexpr auto kernel_sme_bytes = 0;" in generator


@pytest.mark.parametrize("relocated", (False, True))
def test_sme_source_kernel_links_shared_pic_leaves(tmp_path, relocated):
  cmake = shutil.which("cmake")
  c_compiler, cxx_compiler = _cmake_compilers()
  if cmake is None or cxx_compiler is None:
    pytest.skip("CMake and a C++ compiler are required")
  module_directory = ROOT / "cmake"
  if relocated:
    module_directory = tmp_path / "sdk" / "lib" / "cmake" / "Vecops"
    source_directory = module_directory / "sme"
    source_directory.mkdir(parents=True)
    shutil.copy2(ROOT / "cmake" / "VecopsSMEMatmulLeaves.cmake", module_directory)
    for name in ("FusedSkinnyF64.cpp", "RuntimeQuantInt8.cpp", "MixedSignSkinnyInt8.cpp"):
      shutil.copy2(ROOT / "src" / "arch" / "sme" / name, source_directory)
  subprocess.run(
    [
      cmake,
      "-S",
      str(ROOT / "tests" / "cmake" / "SMEMatmulLeaves"),
      "-B",
      str(tmp_path / "build"),
      f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
      *_cmake_flags(),
      *([f"-DCMAKE_C_COMPILER={c_compiler}"] if c_compiler else []),
      f"-DVECOPS_MODULE_DIR={module_directory}",
    ],
    check=True,
    capture_output=True,
    text=True,
  )


def test_cmake_compilers_honor_cc_and_cxx(monkeypatch):
  monkeypatch.setenv("CC", "/opt/arm/bin/clang")
  monkeypatch.setenv("CXX", "/opt/arm/bin/clang++")
  monkeypatch.setattr(shutil, "which", lambda _name: None)
  assert _cmake_compilers() == ("/opt/arm/bin/clang", "/opt/arm/bin/clang++")


def test_cmake_compilers_use_explicit_toolchain_defaults(monkeypatch):
  monkeypatch.delenv("CC", raising=False)
  monkeypatch.delenv("CXX", raising=False)
  monkeypatch.setattr(shutil, "which", lambda name: f"/toolchain/{name}")
  assert _cmake_compilers() == ("/toolchain/gcc", "/toolchain/g++")


def test_cmake_flags_forward_private_toolchain_flags(monkeypatch):
  monkeypatch.setenv("CXXFLAGS", "-isystem /toolchain/include -D_GLIBCXX_USE_CXX11_ABI=1")
  monkeypatch.setenv("LDFLAGS", "-L /toolchain/lib -Wl,--as-needed")
  assert _cmake_flags() == [
    "-DCMAKE_CXX_FLAGS=-isystem /toolchain/include -D_GLIBCXX_USE_CXX11_ABI=1",
    "-DCMAKE_EXE_LINKER_FLAGS=-L /toolchain/lib -Wl,--as-needed",
  ]
