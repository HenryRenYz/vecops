"""Compile generated CPU grid kernels for SVE/SME; never run on ARM hosts."""

import argparse
import subprocess
from pathlib import Path

import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ROOT = Path(__file__).resolve().parents[1]


def source(dtype, dynamic):
  M, N = (T.dynamic(lower=0), T.dynamic(lower=0)) if dynamic else (37, 257)

  @T.template
  def kernel(
    A: T.Tensor((M, N), dtype=dtype),
    B: T.Tensor((M, N), dtype=dtype, access="inout"),
    C: T.Tensor((M, N), dtype=dtype, access="output"),
  ):
    for i in T.grid(M, schedule="dynamic", chunk=3):
      T.store(B[i, :], T.softmax(T.load(A[i, :], dtype="float32")))
    scale = 2.0
    for i in T.grid(M):
      T.store(C[i, :], T.load(B[i, :], dtype="float32") * scale)

  cpp = {"float32": "float", "bfloat16": "vecops::bfloat16_t"}[dtype]
  sizes = "meta::Any{37},meta::Any{257}" if dynamic else "meta::cint<37>,meta::cint<257>"
  return (
    CPUBackend().lower(kernel.ir, (), target="SVE").source
    + f"""
extern "C" void check(const VecopsExecutionContext* context,{cpp}* a,{cpp}* b,{cpp}* c) {{
  using namespace vecops;
  const auto layout=tensor::make_layout(tensor::make_shape({sizes}));
  __kernel__(context,tensor::make_tensor(a,layout),tensor::make_tensor(b,layout),tensor::make_tensor(c,layout));
}}
"""
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", required=True)
  parser.add_argument("--sysroot", required=True, type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-grid-cross")
  args = parser.parse_args()
  args.build_dir.mkdir(parents=True, exist_ok=True)
  for target in ("sve-vla", "sve-vls", "sme-vla", "sme-svl"):
    for dtype in ("float32", "bfloat16"):
      for dynamic in (False, True):
        path = args.build_dir / f"{target}-{dtype}-dynamic{dynamic}.cpp"
        path.write_text(source(dtype, dynamic))
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
