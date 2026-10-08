# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compile instantiated transpose kernels for SVE VLA/VLS and SME; no execution.

Needs an AArch64 sysroot with GCC 13 C++ headers and Clang supporting SME.
SME objects additionally pass the project's manual-state disassembly gate:
no calls may remain between SMSTART/SMSTOP. This is not ARM performance QA.
"""
import argparse
import concurrent.futures
import subprocess
from pathlib import Path

import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from dsl_transpose import catalog

ROOT = Path(__file__).resolve().parents[1]
CPP = {"int8": "std::int8_t", "float16": "vecops::float16_t", "bfloat16": "vecops::bfloat16_t",
       "float32": "float", "float64": "double"}


def source(ad, cd, compute, implementation, *, mode="Const", explicit=False, paired=False):
  m, n = (1024, 1024) if paired else (17, 19)
  M, N = ((T.dynamic(), T.dynamic()) if mode == "Any" else
          (T.dynamic(alignment=16), T.dynamic(alignment=32)) if mode == "Aligned" else (m, n))
  if mode == "Aligned":
    m, n = 32, 64
  @T.template
  def kernel(A: T.Tensor((M, N), (N, 1), dtype=ad),
             C: T.Tensor((N, M), (M, 1), dtype=cd, access="output")):
    if explicit:
      with T.sme.region():
        T.sme.zero_za()
        T.transpose_copy(A, C, compute_dtype=compute, implementation="sme")
    else:
      T.transpose_copy(A, C, compute_dtype=compute, implementation=implementation)
  artifact = CPUBackend().lower(kernel.ir, (), target="SME" if explicit or implementation == "sme" else "SVE")
  cpp_m, cpp_n = (f"meta::Any{{{m}}}", f"meta::Any{{{n}}}") if mode == "Any" else (
      f"meta::dyn<16>({m})", f"meta::dyn<32>({n})") if mode == "Aligned" else (
      f"meta::cint<{m}>", f"meta::cint<{n}>")
  return artifact.source + f'''
extern "C" void check({CPP[ad]}* input, {CPP[cd]}* output) {{
  using namespace vecops;
  auto a = tensor::make_tensor(input, tensor::make_layout(tensor::make_shape({cpp_m}, {cpp_n})));
  auto c = tensor::make_tensor(output, tensor::make_layout(tensor::make_shape({cpp_n}, {cpp_m})));
  __kernel__(a, c);
}}
'''


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", required=True)
  parser.add_argument("--objdump", required=True)
  parser.add_argument("--sysroot", required=True, type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-transpose-cross")
  parser.add_argument("--jobs", type=int, default=4)
  args = parser.parse_args()
  args.build_dir.mkdir(parents=True, exist_ok=True)
  pairs = tuple(dict.fromkeys((a, c, compute) for _, _, a, c, compute in catalog()))
  tasks = []
  for target in ("sve-vla", "sve-vls", "sme-vla", "sme-svl"):
    for ad, cd, compute in pairs:
      stem = f"{target}-{ad}-{cd}-{compute}"
      path = args.build_dir / (stem + ".cpp")
      # These are generated compiler artifacts, never editable source files.
      path.write_text(source(ad, cd, compute, "sme" if target.startswith("sme") else "vector"))
      tasks.append((target, path))
    for mode in ("Any", "Aligned"):
      for ad, cd, compute in (("int8", "int8", "int8"), ("float16", "float64", "float64")):
        path = args.build_dir / f"{target}-{mode}-{ad}-{cd}.cpp"
        path.write_text(source(ad, cd, compute, "auto", mode=mode))
        tasks.append((target, path))
    if target.startswith("sme"):
      for explicit in (False, True):
        path = args.build_dir / f"{target}-paired-explicit{explicit}.cpp"
        path.write_text(source("float32", "float32", "float32", "sme", explicit=explicit, paired=True))
        tasks.append((target, path))
  def compile_one(task):
    target, path = task
    command = [args.clang, "--target=aarch64-linux-gnu", f"--sysroot={args.sysroot}",
               "-march=armv9-a+sve2+bf16" + ("+sme+sme-fa64" if target.startswith("sme") else ""),
               "-std=c++20", "-O3", "-DNDEBUG", "-Wno-parentheses-equality", "-ferror-limit=3",
               "-I", str(ROOT / "include"), "-I", str(ROOT / "include/vecops"),
               "-isystem", str(args.sysroot / "include/c++/13"),
               "-isystem", str(args.sysroot / "include/c++/13/aarch64-linux-gnu")]
    if target == "sve-vls":
      command += ["-msve-vector-bits=512"]
    if target.startswith("sme"):
      command += ["-DVECOPS_TARGET_SME_FA64=1"]
    if target == "sme-svl":
      command += ["-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS=512"]
    object_ = path.with_suffix(".o")
    result = subprocess.run(command + ["-c", str(path), "-o", str(object_)], text=True, capture_output=True)
    if result.returncode:
      return f"FAIL {path.name}\n{result.stderr}"
    if target.startswith("sme"):
      result = subprocess.run(["cmake", f"-DOBJDUMP={args.objdump}", f"-DBINARY={object_}",
                               "-DREQUIRE_MOPA=OFF", "-P", str(ROOT / "tests/cmake/VerifyManualSMEDisassembly.cmake")],
                              text=True, capture_output=True)
      if result.returncode:
        return f"FAIL gate {path.name}\n{result.stdout}{result.stderr}"
    return f"PASS {path.name}"
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    results = list(pool.map(compile_one, tasks))
  for result in results:
    print(result, flush=True)
  failures = sum(result.startswith("FAIL") for result in results)
  print(f"{len(results)-failures}/{len(results)} passed")
  raise SystemExit(bool(failures))


if __name__ == "__main__":
  main()
