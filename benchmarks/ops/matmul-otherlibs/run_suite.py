#!/usr/bin/env python3
"""Run a pinned single-thread benchmark suite and produce its merged CSV."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


BINARIES = {
    "vecops": [
        "MatmulOtherlibsVecopsGeneralBench-Native",
        "MatmulOtherlibsVecopsLLMBench-Native",
        "MatmulOtherlibsVecopsAF3ProjectionBench-Native",
        "MatmulOtherlibsVecopsAF3AttentionQKLargeBench-Native",
        "MatmulOtherlibsVecopsAF3AttentionShallowBench-Native",
        "MatmulOtherlibsVecopsAF3AttentionPVBench-Native",
        "MatmulOtherlibsVecopsAF3TriangleBench-Native",
        "MatmulOtherlibsVecopsBiasBench-Native",
        "MatmulOtherlibsVecopsBiasReluBench-Native",
        "MatmulOtherlibsVecopsBiasSiluBench-Native",
    ],
    "openblas": ["MatmulOtherlibsOpenBLASBench-Native"],
    "onednn": ["MatmulOtherlibsOneDNNBench-Native"],
    "libxsmm": ["MatmulOtherlibsLIBXSMMBench-Native"],
    "acl": ["MatmulOtherlibsACLBench-Native"],
    "kupl": ["MatmulOtherlibsKUPLBench-Native"],
}


def output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--providers", nargs="+", choices=BINARIES,
                        default=["vecops", "openblas", "onednn", "libxsmm"])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--filter", default=".*")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--compiler", default="")
    parser.add_argument("--backend-isa", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--disable-aslr", action="store_true",
        help="run each benchmark through setarch -R for stable code/data addresses")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse an existing non-empty benchmark JSON for this run ID")
    args = parser.parse_args()

    args.build_dir = args.build_dir.resolve()
    binary_dir = args.build_dir / "benchmarks/ops/matmul-otherlibs"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    json_files: list[Path] = []
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    for provider in args.providers:
        for binary_name in BINARIES[provider]:
            executable = binary_dir / binary_name
            if not executable.is_file():
                raise SystemExit(f"missing benchmark executable: {executable}")
            stem = binary_name.removesuffix("-Native")
            destination = args.output_dir / f"{args.host}_{stem}_{run_id}.json"
            if args.resume and destination.is_file():
                try:
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                    reusable = bool(previous.get("benchmarks"))
                except (OSError, json.JSONDecodeError, AttributeError):
                    reusable = False
                if reusable:
                    print(f"REUSE {binary_name}", flush=True)
                    json_files.append(destination)
                    continue
            command = [str(executable), f"--benchmark_filter={args.filter}",
                       f"--benchmark_out={destination}",
                       "--benchmark_out_format=json", "--benchmark_color=false"]
            if args.cpu is not None:
                if shutil.which("taskset") is None:
                    raise SystemExit("--cpu requires taskset")
                command = ["taskset", "-c", str(args.cpu), *command]
            if args.disable_aslr:
                if shutil.which("setarch") is None:
                    raise SystemExit("--disable-aslr requires setarch")
                command = ["setarch", platform.machine(), "-R", *command]
            print("RUN", " ".join(command), flush=True)
            if not args.dry_run:
                log_path = destination.with_suffix(".log")
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, env=environment, text=True,
                        stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    tail = log_path.read_text(
                        encoding="utf-8", errors="replace").splitlines()[-30:]
                    print("\n".join(tail))
                    raise SystemExit(
                        f"benchmark failed ({result.returncode}); see {log_path}")
                print(f"DONE {binary_name}", flush=True)
                json_files.append(destination)

    if args.dry_run:
        return 0

    source_root = Path(__file__).resolve().parents[3]
    dependency_file = Path(__file__).with_name("dependencies.json")
    dependencies = json.loads(dependency_file.read_text(encoding="utf-8"))
    vecops_revision = output(["git", "rev-parse", "HEAD"], source_root)
    if output([
            "git", "status", "--porcelain", "--untracked-files=no"
    ], source_root):
        vecops_revision += "+dirty"
    version_map = {
        "version_openblas": dependencies["libraries"]["OpenBLAS"]["revision"],
        "version_onednn": dependencies["libraries"]["oneDNN"]["revision"],
        "version_libxsmm": dependencies["libraries"]["LIBXSMM"]["revision"],
        "version_acl": dependencies["libraries"]["Arm Compute Library"]["revision"],
        "version_kupl_mma": dependencies["libraries"]["KUPL"]["revision"],
        "version_vecops": vecops_revision,
    }
    cpu = output(["bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n1"])
    metadata = {
        "common": {
            "run_id": run_id, "host": args.host, "cpu": cpu,
            "compiler": args.compiler, "backend_isa": args.backend_isa,
            **version_map,
        },
        "platform": platform.platform(),
        "command_filter": args.filter,
        "aslr_disabled": args.disable_aslr,
    }
    metadata_path = args.output_dir / f"{args.host}_{run_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / f"{args.host}_{run_id}.csv"
    merge = Path(__file__).with_name("merge_results.py")
    subprocess.run([
        "python3", str(merge), *(str(path) for path in json_files),
        "--output", str(csv_path), "--metadata", str(metadata_path),
    ], check=True)
    print(f"CSV {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
