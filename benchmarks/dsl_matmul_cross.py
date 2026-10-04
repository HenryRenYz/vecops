"""Instantiate SVE/SME matrix kernels and verify manual SM regions; no execution."""

import argparse
import concurrent.futures
import subprocess
from pathlib import Path
import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ROOT = Path(__file__).resolve().parents[1]
CPP = {
  "float16": "vecops::float16_t",
  "bfloat16": "vecops::bfloat16_t",
  "float32": "float",
  "float64": "double",
  "int8": "std::int8_t",
  "uint8": "std::uint8_t",
  "int32": "std::int32_t",
}


def source(
  ad,
  bd,
  acc,
  *,
  library=False,
  fused=False,
  packed=False,
  vector=False,
  normalized=False,
  math=False,
  special_math=False,
  target="generic",
  schedule_pipeline="off",
):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), ad),
    B: T.Tensor((K, N), (N, 1), bd),
    D: T.Tensor((1, N), dtype=acc),
    C: T.Tensor((M, N), dtype=acc, access="output"),
  ):
    a = T.load(A)
    b = T.load(B)
    if fused:
      a = T.cast(T.load(A, dtype="float32") * 2 + 1, ad)
      b = T.cast(T.load(B, dtype="float32") * 2 + 1, bd)
    if normalized:
      a = T.cast(T.library.layernorm(T.load(A, dtype="float32")) * 2 + 1, ad)
    if packed:
      a = T.pack(a, side="A", format="sme")
      b = T.pack(b, side="B", format="sme")
    if library:
      value = T.library.matmul((a, b), C, acc=T.load(D) if fused else None)
    elif vector:
      value = T.dot((a, b), C, acc=T.load(D) if fused else None, implementation="vector")
    else:
      value = T.dot((a, b), C, acc=T.load(D) if fused else None)
    if special_math:
      result = T.cast(value, "float64" if acc == "float64" else "float32")
      positive = T.abs(result) + 1
      T.store(C, T.log(positive) + T.rcp(positive) + T.rsqrt(positive) + T.exp2(-positive))
    elif math:
      result = T.cast(value, "float32")
      T.store(C, result / (1 + T.exp(-result)))
    elif fused:
      T.store(C, T.max(value, 0))

  artifact = CPUBackend().lower(kernel.ir, (), target=target, options={"cpu_matmul_pipeline": schedule_pipeline})
  return (
    artifact.source
    + f"""
extern "C" void check({CPP[ad]}* ap,{CPP[bd]}* bp,{CPP[acc]}* dp,{CPP[acc]}* cp,
                       std::int64_t m,std::int64_t n,std::int64_t k) {{
  using namespace vecops;
  auto a=tensor::make_tensor(ap,tensor::make_layout(tensor::make_shape(meta::Any{{m}},meta::Any{{k}})));
  auto b=tensor::make_tensor(bp,tensor::make_layout(tensor::make_shape(meta::Any{{k}},meta::Any{{n}})));
  auto d=tensor::make_tensor(dp,tensor::make_layout(tensor::make_shape(meta::cint<1>,meta::Any{{n}})));
  auto c=tensor::make_tensor(cp,tensor::make_layout(tensor::make_shape(meta::Any{{m}},meta::Any{{n}})));
  __kernel__(a,b,d,c);
}}
"""
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", required=True)
  parser.add_argument("--objdump", required=True)
  parser.add_argument("--sysroot", required=True, type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-cross")
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--schedule-pipeline", choices=["off", "prefetch", "staged"], default="off")
  parser.add_argument(
    "--variant",
    nargs="+",
    default=["plain", "fused", "library", "packed", "normalized", "math", "special_math"],
    choices=["plain", "fused", "library", "packed", "normalized", "math", "special_math"],
  )
  args = parser.parse_args()
  args.build_dir.mkdir(parents=True, exist_ok=True)
  tasks = []
  for target in ("sve-vla", "sve-vls", "sme-vla", "sme-svl"):
    for ad, bd, acc in (
      ("float16", "float16", "float32"),
      ("bfloat16", "bfloat16", "float32"),
      ("float32", "float32", "float32"),
      ("float64", "float64", "float64"),
      ("int8", "int8", "int32"),
      ("uint8", "int8", "int32"),
    ):
      for variant in args.variant:
        if variant == "packed" and not target.startswith("sme"):
          continue
        path = args.build_dir / f"{target}-{ad}-{bd}-{variant}.cpp"
        path.write_text(
          source(
            ad,
            bd,
            acc,
            library=variant in {"library", "normalized", "math", "special_math"},
            fused=variant in {"fused", "library", "normalized", "math", "special_math"},
            packed=variant == "packed",
            vector=target.startswith("sve"),
            normalized=variant == "normalized",
            math=variant == "math",
            special_math=variant == "special_math",
            target="SME" if target.startswith("sme") else "SVE",
            schedule_pipeline=args.schedule_pipeline,
          )
        )
        tasks.append((target, path))

  def compile_one(task):
    target, path = task
    command = [
      args.clang,
      "--target=aarch64-linux-gnu",
      f"--sysroot={args.sysroot}",
      "-march=armv9-a+sve2+bf16" + ("+sme+sme-fa64+sme-f64f64" if target.startswith("sme") else ""),
      "-std=c++20",
      "-O3",
      "-DNDEBUG",
      "-ferror-limit=2",
      "-Wno-parentheses-equality",
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
      command += ["-msve-vector-bits=512"]
    if target.startswith("sme"):
      command += ["-DVECOPS_TARGET_SME_FA64=1", "-DVECOPS_TARGET_SME_F64F64=1"]
    if target == "sme-svl":
      command += ["-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS=512"]
    object_ = path.with_suffix(".o")
    result = subprocess.run(command + ["-c", str(path), "-o", str(object_)], text=True, capture_output=True)
    if result.returncode:
      return f"FAIL {path.name}\n{result.stderr}"
    if target.startswith("sme"):
      result = subprocess.run(
        [
          "cmake",
          f"-DOBJDUMP={args.objdump}",
          f"-DBINARY={object_}",
          "-DREQUIRE_MOPA=ON",
          "-P",
          str(ROOT / "tests/cmake/VerifyManualSMEDisassembly.cmake"),
        ],
        text=True,
        capture_output=True,
      )
      if result.returncode:
        return f"FAIL gate {path.name}\n{result.stdout}{result.stderr}"
    return f"PASS {path.name}"

  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    results = list(pool.map(compile_one, tasks))
  for result in results:
    print(result, flush=True)
  failures = sum(result.startswith("FAIL") for result in results)
  print(f"{len(results) - failures}/{len(results)} passed")
  raise SystemExit(bool(failures))


if __name__ == "__main__":
  main()
