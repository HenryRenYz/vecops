"""Compile new primitive, local-loop, team and SME paths; no ARM execution."""

import argparse
import subprocess
from pathlib import Path

import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ROOT = Path(__file__).resolve().parents[1]


def source(kind):
  if kind == "indexed":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      idx = T.load(Idx)
      value = T.gather(A, idx, mask=idx >= 0, dtype="float32")
      T.scatter(C, T.arange(37), value + 1, unique=True)

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "control":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      for i in T.simd(37, vector=T.Vector("float32", N=16)):
        x = T.gather(A, i, dtype="float32")
        T.scatter(C, i, T.local_shuf(x, (3, 2, 1, 0)), unique=True)
      count = T.mask_count(T.load(A) > 3)
      C[0] = float(count)

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "scan":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      T.store(C, T.scan(T.load(A, dtype="float32")))
      for i in T.pipelined(37):
        C[i] = T.load(A[i]) + 1

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "scalable":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      for i in T.simd(37):
        T.scatter(C, i, T.gather(A, i, dtype="float32") + 1, unique=True)
      x = T.local_shuf(T.load(A, dtype="float32"), (3, 2, 1, 0))
      T.store(C, x + 2)

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "buffering":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      for i in T.pipelined(9, buffering=True):
        T.store(C[i * 4 : i * 4 + 4], T.load(A[i * 4 : i * 4 + 4], dtype="float32") + 1)

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "widened":
    descriptor = T.Vector("float32")

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      x = T.cast(T.load(A, vector=descriptor), "int64")
      for i in T.serial(3):
        x = x + 1
      T.store(C[: T.lanes(descriptor)], x)

    wrapper = "__kernel__(tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "team":

    @T.template
    def kernel(
      A: T.Tensor((37,), dtype="float16"),
      Idx: T.Tensor((37,), dtype="int64"),
      C: T.Tensor((37,), dtype="float32", access="output"),
    ):
      with T.team(4) as worker:
        sum_ = T.team_reduce(float(worker), op="add")
        T.atomic_add(C[0], sum_)
        T.barrier()
        C[worker + 1] = sum_

    wrapper = "__kernel__(nullptr,tensor::make_tensor(a,l37),tensor::make_tensor(idx,l37),tensor::make_tensor(c,l37));"
  elif kind == "coordinates":

    @T.template
    def kernel(
      A: T.Tensor((6, 5), (10, 2), "float16"),
      Idx: T.Tensor((7,), dtype="int64"),
      C: T.Tensor((7, 5), dtype="float32", access="output"),
    ):
      rows = T.expand_dims(T.load(Idx), 1)
      columns = T.indices((7, 5), axis=1)
      x = T.gather(A, (rows, columns), mask=rows >= 0, dtype="float32")
      T.store(C, T.library.masked_softmax(x, rows >= 0))

    wrapper = """__kernel__(
      tensor::make_tensor(a,tensor::make_layout(tensor::make_shape(meta::cint<6>,meta::cint<5>),tensor::make_strides(meta::cint<10>,meta::cint<2>))),
      tensor::make_tensor(idx,tensor::make_layout(tensor::make_shape(meta::cint<7>))),
      tensor::make_tensor(c,tensor::make_layout(tensor::make_shape(meta::cint<7>,meta::cint<5>))));"""
  elif kind == "workspace":

    @T.template
    def kernel(A: T.Tensor((37,), dtype="float16"), C: T.Tensor((37,), dtype="float32", access="output")):
      scratch = T.workspace((T.worker_capacity(), 37), dtype="float32")
      with T.team(4) as worker:
        T.store(scratch[worker, :], T.library.stable_sigmoid(A, algorithm="polynomial"))
        T.barrier()
        if worker == 0:
          T.store(C, T.library.rmsnorm(scratch[worker, :]))

    wrapper = "__kernel__(context,tensor::make_tensor(a,l37),tensor::make_tensor(c,l37));"
  elif kind == "attention":

    @T.template
    def kernel(
      Q: T.Tensor((3, 8), dtype="float32"),
      K: T.Tensor((13, 8), dtype="float32"),
      V: T.Tensor((13, 5), dtype="float32"),
      C: T.Tensor((3, 5), dtype="float32", access="output"),
    ):
      T.library.scaled_dot_product_attention(Q, K, V, C, causal=True, block_k=4, implementation="vector")

    wrapper = """__kernel__(
      tensor::make_tensor(c,tensor::make_layout(tensor::make_shape(meta::cint<3>,meta::cint<8>))),
      tensor::make_tensor(c,tensor::make_layout(tensor::make_shape(meta::cint<13>,meta::cint<8>))),
      tensor::make_tensor(c,tensor::make_layout(tensor::make_shape(meta::cint<13>,meta::cint<5>))),
      tensor::make_tensor(c,tensor::make_layout(tensor::make_shape(meta::cint<3>,meta::cint<5>))));"""
  else:

    @T.template
    def kernel(
      A: T.Tensor((4,), dtype="float32"),
      B: T.Tensor((4,), dtype="float32"),
      C: T.Tensor((4, 4), dtype="float32", access="output"),
    ):
      with T.sme.region():
        T.sme.zero_za()
        a = T.sme.load(A)
        b = T.sme.load(B)
        T.sme.mopa(0, a, b)
        T.sme.store_tile(0, C)

    wrapper = "__kernel__(tensor::make_tensor(c,l4),tensor::make_tensor(c,l4),tensor::make_tensor(c,l44));"
  return (
    CPUBackend().lower(kernel.ir, (), target="SVE").source
    + """
#include "vecops/execution/WorkspaceContext.h"
extern "C" void check(vecops::float16_t* a,std::int64_t* idx,float* c,vecops::execution::WorkspaceContext& context) {
  using namespace vecops;
  const auto l37=tensor::make_layout(tensor::make_shape(meta::cint<37>));
  const auto l4=tensor::make_layout(tensor::make_shape(meta::cint<4>));
  const auto l44=tensor::make_layout(tensor::make_shape(meta::cint<4>,meta::cint<4>));
  """
    + wrapper
    + "\n}\n"
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", required=True)
  parser.add_argument("--sysroot", required=True, type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-primitives-cross")
  parser.add_argument(
    "--kind",
    nargs="+",
    default=[
      "indexed",
      "control",
      "scan",
      "scalable",
      "buffering",
      "widened",
      "team",
      "coordinates",
      "workspace",
      "attention",
      "sme",
    ],
  )
  parser.add_argument(
    "--target",
    nargs="+",
    choices=("sve-vla", "sve-vls", "sme-vla", "sme-svl"),
    default=["sve-vla", "sve-vls", "sme-vla", "sme-svl"],
  )
  args = parser.parse_args()
  args.build_dir.mkdir(parents=True, exist_ok=True)
  for target in args.target:
    for kind in args.kind:
      if kind == "sme" and not target.startswith("sme"):
        continue
      path = args.build_dir / f"{target}-{kind}.cpp"
      path.write_text(source(kind))
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
