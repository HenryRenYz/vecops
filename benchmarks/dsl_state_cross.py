# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cross-compile typed recurrence/branch/helper states in SVE and SME modes."""

import argparse
import subprocess
from pathlib import Path

import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ROOT = Path(__file__).resolve().parents[1]


def source(dynamic, vector):
  N = T.dynamic(lower=0) if dynamic else 17
  descriptor = T.Vector(T.float32, pow=1)

  @T.func
  def transform(x, count):
    for i in T.serial(count):
      if i > 2:
        return x + 3
    return x + 1

  @T.template
  def kernel(A: T.Tensor((N,), dtype="float32"), C: T.Tensor((N,), dtype="float32", access="output"), count: int):
    if vector:
      x = T.load(A, vector=descriptor)
    else:
      x = T.load(A)
    for i in T.serial(count):
      old = x
      if i % 2 == 0:
        x = x + 1
      else:
        x = x * 2
      x = x + old * 0.25
    if vector:
      T.store(C[: T.lanes(descriptor)], transform(x, count))
    else:
      T.store(C, transform(x, count))

  sizes = "meta::Any{17}" if dynamic else "meta::cint<17>"
  return (
    CPUBackend().lower(kernel.ir, (), target="SVE").source
    + f"""
extern "C" void check(float* a, float* c, vecops::nint_t count) {{
  using namespace vecops;
  const auto layout=tensor::make_layout(tensor::make_shape({sizes}));
  __kernel__(tensor::make_tensor(a,layout),tensor::make_tensor(c,layout),count);
}}
"""
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", required=True)
  parser.add_argument("--sysroot", required=True, type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-state-cross")
  args = parser.parse_args()
  args.build_dir.mkdir(parents=True, exist_ok=True)
  for target in ("sve-vla", "sve-vls", "sme-vla", "sme-svl"):
    for dynamic, vector in ((d, v) for d in (False, True) for v in (False, True)):
      path = args.build_dir / f"{target}-dynamic{dynamic}-vector{vector}.cpp"
      path.write_text(source(dynamic, vector))
      command = [
        args.clang,
        "--target=aarch64-linux-gnu",
        f"--sysroot={args.sysroot}",
        "-march=armv9-a+sve2+bf16" + ("+sme+sme-fa64" if target.startswith("sme") else ""),
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Wno-parentheses-equality",
        "-ferror-limit=3",
        "-I",
        str(ROOT / "include"),
        "-I",
        str(ROOT / "include/vecops"),
        "-isystem",
        str(args.sysroot / "include/c++/13"),
        "-isystem",
        str(args.sysroot / "include/c++/13/aarch64-linux-gnu"),
      ]
      if target == "sve-vls":
        command.append("-msve-vector-bits=512")
      if target == "sme-svl":
        command.append("-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS=512")
      if target.startswith("sme"):
        command.append("-DVECOPS_TARGET_SME_FA64=1")
      subprocess.run(command + ["-c", str(path), "-o", str(path.with_suffix(".o"))], check=True)
      print(path.stem, "passed", flush=True)


if __name__ == "__main__":
  main()
