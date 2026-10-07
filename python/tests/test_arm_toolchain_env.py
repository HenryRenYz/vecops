"""Private standard-library envfiles keep kernels and Torch bridges on Clang."""

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/arm_toolchain_setup.sh"


@pytest.mark.parametrize("suffix", ("plain", "directory with spaces", "a'quoted;directory"))
def test_private_toolchain_env_uses_same_quoted_clang_pair_for_bridge(suffix, tmp_path):
  source = SCRIPT.read_text()
  # Execute the real emission block, not a copied implementation, without
  # installing packages or requiring an ARM host/compute allocation.
  match = re.search(
    r'(\{\n  echo \'# Generated for this project only\..*?\n\}) >"\$\{task_envfile\}"', source, re.DOTALL
  )
  assert match is not None
  envfile = tmp_path / "job-env.sh"
  environment = dict(
    os.environ,
    task_cc=f"/project/{suffix}/clang",
    task_cxx=f"/project/{suffix}/clang++",
    task_private_cxxflags=shlex.join([f"--gcc-install-dir=/project/{suffix}/gcc/14", "-nostdinc++"]),
    task_envfile=str(envfile),
    LD_LIBRARY_PATH=f"/project/{suffix}/runtime",
    VECOPS_TORCH_BRIDGE_CC="/old/gcc",
    VECOPS_TORCH_BRIDGE_CXX="/old/g++",
  )
  block = (
    match[0]
    + '\nsource "${task_envfile}"\nprintf "%s\\0" "$CC" "$CXX" "$VECOPS_TORCH_BRIDGE_CC" "$VECOPS_TORCH_BRIDGE_CXX" "$CXXFLAGS" "$LD_LIBRARY_PATH"\n'
  )
  result = subprocess.run(["bash", "-c", block], env=environment, capture_output=True, check=True)
  assert result.stdout.decode().split("\0") == [
    environment["task_cc"],
    environment["task_cxx"],
    environment["task_cc"],
    environment["task_cxx"],
    environment["task_private_cxxflags"],
    environment["LD_LIBRARY_PATH"],
    "",
  ]
