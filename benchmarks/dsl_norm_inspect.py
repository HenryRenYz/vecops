"""Profile one normalization pair or inspect its native vector loops.

The profile path checks correctness before sampling and excludes compilation.
Use perf around this command in a scheduled compute allocation. The asm path
reports static loops, not execution counts; stack references alone do not
prove hot-path spills. Both AArch64 and x86 disassembly syntax are supported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def backward_loops(assembly):
  """Group addresses per function before resolving backwards branch targets."""
  functions = []
  current = ("<unknown>", [])
  for line in assembly.splitlines():
    header = re.match(r"^\s*[0-9a-f]+ <(.+)>:$", line)
    if header:
      if current[1]:
        functions.append(current)
      current = (header[1], [])
      continue
    instruction = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", line)
    if instruction:
      address, op, operands = instruction.groups()
      current[1].append((int(address, 16), op, operands.split(" <", 1)[0]))
  if current[1]:
    functions.append(current)
  regions = []
  for function, instructions in functions:
    indices = {address: index for index, (address, _, _) in enumerate(instructions)}
    for end, (address, op, operands) in enumerate(instructions):
      if not (op.startswith(("j", "b.")) or op in {"b", "cbz", "cbnz", "tbz", "tbnz"}):
        continue
      token = operands.rsplit(",", 1)[-1].strip().split()[0].removeprefix("0x")
      if not re.fullmatch(r"[0-9a-f]+", token):
        continue
      start = indices.get(int(token, 16))
      if start is None or start >= end:
        continue
      body = instructions[start : end + 1]
      if any(re.search(r"\bz\d+\b|%[xyz]mm\d+", args) for _, _, args in body):
        regions.append((function, start, end, body))
  # Smaller row-vector loops, not their enclosing row or benchmark repeat nest.
  return tuple(
    region
    for region in regions
    if not any(region[0] == other[0] and region[1] <= other[1] and other[2] < region[2] for other in regions)
  )


def audit(args):
  result = subprocess.run(
    [args.objdump, "-d", "-C", "--no-show-raw-insn", str(args.object)],
    capture_output=True,
    text=True,
    check=True,
  )
  loops = backward_loops(result.stdout)
  if args.symbol:
    loops = tuple(region for region in loops if args.symbol in region[0])
  for function, _, _, body in sorted(loops, key=lambda region: len(region[3]))[: args.limit]:
    counts = Counter(op for _, op, _ in body)
    print(
      json.dumps(
        {
          "function": function,
          "region": f"{body[0][0]:x}-{body[-1][0]:x}",
          "instructions": len(body),
          "op_counts": counts,
          "calls": sum(op in {"bl", "blr", "call", "callq"} for _, op, _ in body),
          "stack_references": sum(bool(re.search(r"\[sp[\],]|\(%[er](?:sp|bp)", operands)) for _, _, operands in body),
        }
      )
    )
    for address, op, operands in body:
      print(f"  {address:x}: {op:14} {operands}")


def profile(args):
  import vecops
  from dsl_norm_catalog import Case, kernels, metadata, validate
  from vecops.language._debug.machine import objects

  shape = tuple(map(int, args.shape.split("x")))
  if not shape or min(shape) <= 0 or args.repeat <= 0:
    raise ValueError("shape extents and repeat must be positive")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    raise ValueError(f"timing CPU {cpu} is not in the allowed affinity {sorted(allowed)}")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  case = Case(
    args.operation,
    shape,
    args.dtype,
    args.meta,
    args.affine,
    args.accuracy,
    args.variance if args.operation == "layernorm" else "n/a",
  )
  with vecops.collect_compile_requests() as collector:
    pair = kernels(compiler, case, unroll=args.unroll)
    for kernel in pair:
      kernel.operator(*metadata(case), 1, **kernel._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  os.sched_setaffinity(0, {cpu})
  operands = validate(case, pair)
  report = {"case": case.key, "timing_cpu": cpu, "selected": args.kernel, "kernels": {}}
  for name, kernel in zip(("dsl", "cpp"), pair):
    report["kernels"][name] = {
      "source": str(kernel.source),
      "native_objects": objects(kernel.debug),
      "cache": kernel.get_cache_info(),
    }
    if args.export:
      kernel.export_debug(args.export / name, include_commands=True)
  print(json.dumps(report, default=str), flush=True)
  chosen = pair[args.kernel == "cpp"]
  chosen.operator.native(*operands, args.repeat, **chosen._core.specialization_values)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  asm = commands.add_parser("asm")
  asm.add_argument("object", type=Path)
  asm.add_argument("--objdump", default="objdump")
  asm.add_argument("--symbol", help="Filter native function names before selecting loops")
  asm.add_argument("--limit", type=int, default=12)
  sample = commands.add_parser("profile")
  sample.add_argument("--operation", required=True, choices=("layernorm", "softmax"))
  sample.add_argument("--shape", required=True)
  sample.add_argument("--dtype", default="float32", choices=("float16", "bfloat16", "float32", "float64"))
  sample.add_argument("--meta", default="Any", choices=("Const", "Any"))
  sample.add_argument("--affine", default="none", choices=("gamma_beta", "gamma", "beta", "none"))
  sample.add_argument("--accuracy", default="strict", choices=("strict", "fast", "estimate"))
  sample.add_argument("--variance", default="moments", choices=("centered", "moments"))
  sample.add_argument("--kernel", default="dsl", choices=("dsl", "cpp"))
  sample.add_argument("--repeat", type=int, default=10000)
  sample.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  sample.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  sample.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  sample.add_argument("--unroll", type=int, default=2)
  sample.add_argument("--cpu", type=int)
  sample.add_argument("--jobs", type=int, default=4)
  sample.add_argument("--export", type=Path, help="Export IR/pass/source bundles into empty dsl/cpp subdirectories")
  sample.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-norm-catalog")
  args = parser.parse_args()
  audit(args) if args.command == "asm" else profile(args)


if __name__ == "__main__":
  main()
