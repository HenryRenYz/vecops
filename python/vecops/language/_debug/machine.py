"""Inspect actual native objects; never rebuild as a side effect of a getter."""

import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .report import Unavailable


def objects(info):
  results = []
  for artifact in info.artifacts():
    command = artifact["command"]
    args = command.get("arguments") or shlex.split(command.get("command", ""))
    output = command.get("output")
    if output is None and "-o" in args:
      output = args[args.index("-o") + 1]
    if output:
      path = Path(output)
      path = path if path.is_absolute() else Path(artifact["build_dir"]) / path
      if not path.is_file():
        path = Path(command["directory"]) / output
      if path.is_file():
        spec = Path(command["file"]).parent / "vecops_spec.h"
        results.append({"object": str(path), "specialization": spec.read_text() if spec.is_file() else "unknown"})
  return tuple({entry["object"]: entry for entry in results}.values())


def assembly(info, specialization=None):
  candidates = objects(info)
  if specialization is not None:
    candidates = tuple(
      c for c in candidates if str(specialization) in c["specialization"] or str(specialization) in c["object"]
    )
  if not candidates:
    return Unavailable("native object not present; prepare a specialization and retain its build directory")
  if len(candidates) != 1:
    return Unavailable("multiple native specializations; select one of: " + ", ".join(c["object"] for c in candidates))
  tool = shutil.which("llvm-objdump") or shutil.which("objdump")
  if not tool:
    return Unavailable("install objdump or llvm-objdump to inspect assembly")
  result = subprocess.run([tool, "-d", "-C", candidates[0]["object"]], capture_output=True, text=True, check=False)
  if result.returncode:
    return Unavailable(result.stderr.strip())
  return result.stdout


def machine_report(info, specialization=None):
  text = assembly(info, specialization)
  if isinstance(text, Unavailable):
    return {"unavailable": text.reason}
  return {
    "stack_access_sites": len(re.findall(r"\(%[er]sp\)|\[sp[,\]]", text)),
    "call_sites": len(re.findall(r"\bcall[q]?\b|\bbl\b", text)),
    "basis": "whole object, includes ABI adapters/exception paths; stack accesses are NOT an exact hot-loop spill count",
  }


def llvm_mca(source, *, target=None, cpu=None, iterations=100):
  """Analyze caller-selected assembly (not all disassembly including cold paths)."""
  tool = shutil.which("llvm-mca")
  if not tool:
    return Unavailable("llvm-mca is optional and not installed")
  arguments = [tool, f"-iterations={int(iterations)}"]
  if target:
    arguments.append(f"-mtriple={target}")
  if cpu:
    arguments.append(f"-mcpu={cpu}")
  result = subprocess.run(arguments, input=source, capture_output=True, text=True, check=False)
  return result.stdout if result.returncode == 0 else Unavailable(result.stderr.strip())
