# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Profile one native matrix schedule or audit AMX loops in its shared object.

The profile path checks the DSL/C++ pair before sampling repeats, keeps the
existing benchmark buffers/layouts, and prints loaded artifact paths. The asm
path reports static backward-branch regions, not execution frequencies or
proven register spills; use perf samples to identify the relevant hot region.
"""

import argparse
import json
import os
import re
import subprocess
from copy import copy
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(path, objdump, limit):
  assembly = subprocess.check_output([objdump, "-d", "--no-show-raw-insn", str(path)], text=True)
  instructions = []
  for line in assembly.splitlines():
    match = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", line)
    if match:
      address, op, operands = match.groups()
      instructions.append((int(address, 16), op, operands.split(" <", 1)[0]))
  indices = {address: i for i, (address, _, _) in enumerate(instructions)}
  regions = []
  for end, (address, op, operands) in enumerate(instructions):
    target = re.match(r"([0-9a-f]+)\b", operands)
    if not op.startswith("j") or target is None:
      continue
    start = indices.get(int(target[1], 16))
    if start is None or start >= end:
      continue
    body = instructions[start : end + 1]
    if any(inst.startswith("tdp") for _, inst, _ in body):
      regions.append((start, end, body))
  # Exclude spatial/repeat regions enclosing a smaller AMX K region.
  regions = [
    region for region in regions if not any(region[0] <= other[0] and other[1] < region[1] for other in regions)
  ]
  for _, _, body in sorted(regions, key=lambda r: len(r[2]))[:limit]:
    print(
      json.dumps(
        {
          "region": f"{body[0][0]:x}-{body[-1][0]:x}",
          "instructions": len(body),
          "dots": sum(op.startswith("tdp") for _, op, _ in body),
          "tile_loads": sum(op.startswith("tileload") for _, op, _ in body),
          "conditional_branches": sum(op.startswith("j") and op != "jmp" for _, op, _ in body),
          "calls": sum(op.startswith("call") for _, op, _ in body),
          "stack_references": sum("(%rsp" in args or "(%rbp" in args for _, _, args in body),
        }
      )
    )
    for address, op, operands in body:
      print(f"  {address:x}: {op:14} {operands}")


def profile(args):
  import numpy as np
  import torch
  import vecops
  import vecops.language as T
  from dsl_matmul import kernels
  from dsl_timing import isolate_compilation, reserve_cpu
  from dsl_transpose import aligned_empty

  m, n, k = map(int, args.shape.split("x"))
  if min(m, n, k) <= 0 or args.iterations <= 0:
    raise ValueError("profile extents and iterations must be positive")
  isolate_compilation(args.cpu)
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(
    target="Native",
    cc="gcc",
    cxx="g++",
    jobs=4,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  pair, types = kernels(
    compiler,
    m,
    n,
    k,
    args.mode,
    args.kind,
    args.pipeline,
    args.level,
    {"cpu_matmul_fusion": "auto", "cpu_gemv_k_unroll": 2},
  )
  if args.baseline_source:
    baseline = copy(pair[0].template)
    baseline.ir = replace(baseline.ir, source=args.baseline_source.resolve())
    pair = (T.compile(baseline, compiler=compiler), pair[1])
  ad, bd, _, _, acc, out, prologue = types
  a, b = aligned_empty((m, k), ad), aligned_empty((n, k), bd).T
  c = aligned_empty((m, n), out)
  d = aligned_empty((1 if prologue == 2 else m, n), acc)
  rng = np.random.default_rng(17)
  a[:] = rng.integers(0 if ad == "uint8" else -3, 4, size=a.shape) / (8 if "float" in ad else 1)
  b[:] = rng.integers(-3, 4, size=b.shape) / (8 if "float" in bd else 1)
  d[:] = rng.integers(-3, 4, size=d.shape)
  for index in (1, 0):
    kernel = pair[index]
    kernel.operator.native(a, b, c, d, 1, **kernel._core.specialization_values)
    if index == 1:
      expected = c.copy()
    else:
      np.testing.assert_allclose(c, expected, rtol=5e-3, atol=3e-4)
  # Only disclose this task's kernel modules, not unrelated process mappings.
  paths = sorted(
    {
      line.split()[-1]
      for line in Path("/proc/self/maps").read_text().splitlines()
      if str(args.build_dir.resolve()) in line and line.rstrip().endswith(".so")
    }
  )
  print(json.dumps({"dsl_source": str(pair[0].source), "artifacts": paths, "selected": args.kernel}), flush=True)
  kernel = pair[args.kernel == "cpp"]
  with reserve_cpu(args.cpu):
    os.sched_setaffinity(0, {args.cpu})
    kernel.operator.native(a, b, c, d, args.iterations, **kernel._core.specialization_values)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  asm = commands.add_parser("asm")
  asm.add_argument("object", type=Path)
  asm.add_argument("--objdump", default="objdump")
  asm.add_argument("--limit", type=int, default=3)
  sample = commands.add_parser("profile")
  sample.add_argument("--shape", required=True)
  sample.add_argument("--mode", choices=("Const", "Dynamic"), default="Const")
  sample.add_argument("--kind", choices=("bf16", "convert", "i8", "mixed"), default="bf16")
  sample.add_argument(
    "--pipeline",
    choices=("plain", "accumulate", "bias_relu", "bias_clamp", "scale", "silu", "affine_input"),
    default="plain",
  )
  sample.add_argument("--level", choices=("dot", "matmul"), default="matmul")
  sample.add_argument("--kernel", choices=("dsl", "cpp"), default="dsl")
  sample.add_argument("--iterations", type=int, default=500)
  sample.add_argument("--baseline-source", type=Path, help="Profile saved DSL source with the same signature/buffers")
  sample.add_argument("--cpu", type=int, default=2)
  sample.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul")
  args = parser.parse_args()
  if args.command == "asm":
    audit(args.object, args.objdump, args.limit)
  else:
    profile(args)


if __name__ == "__main__":
  main()
