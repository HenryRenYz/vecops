#!/usr/bin/env python3
"""Run a pinned single-thread benchmark suite and produce its merged CSV."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
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


def reusable_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return bool(document.get("benchmarks"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def error_document(name: str, message: str) -> dict[str, object]:
    return {
        "context": {},
        "benchmarks": [{
            "name": name,
            "run_name": name,
            "run_type": "iteration",
            "repetitions": 1,
            "repetition_index": 0,
            "threads": 1,
            "iterations": 0,
            "real_time": 0.0,
            "cpu_time": 0.0,
            "time_unit": "ns",
            "error_occurred": True,
            "error_message": message,
        }],
    }


def wrapped_command(
    executable: Path, benchmark_filter: str, destination: Path,
    cpu: int | None, disable_aslr: bool,
) -> list[str]:
    command = [str(executable), f"--benchmark_filter={benchmark_filter}",
               f"--benchmark_out={destination}",
               "--benchmark_out_format=json", "--benchmark_color=false"]
    if cpu is not None:
        if shutil.which("taskset") is None:
            raise SystemExit("--cpu requires taskset")
        command = ["taskset", "-c", str(cpu), *command]
    if disable_aslr:
        if shutil.which("setarch") is None:
            raise SystemExit("--disable-aslr requires setarch")
        command = ["setarch", platform.machine(), "-R", *command]
    return command


def regex_literal(text: str) -> str:
    # Google Benchmark uses std::regex. Python's re.escape() also escapes '-'
    # and several punctuation characters which are invalid escapes in that
    # grammar outside a character class.
    return re.sub(r"([.\\^$*+?{}\[\]|()])", r"\\\1", text)


def run_isolated_cases(
    executable: Path, destination: Path, benchmark_filter: str,
    environment: dict[str, str], cpu: int | None, disable_aslr: bool,
    timeout_seconds: float, resume: bool,
) -> None:
    listed = subprocess.run(
        [str(executable), "--benchmark_list_tests"], env=environment,
        check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT).stdout.splitlines()
    selector = re.compile(benchmark_filter)
    names = [name.strip() for name in listed
             if name.strip().startswith("MatmulOtherlibs/")
             and selector.search(name.strip())]
    if not names:
        raise SystemExit(
            f"filter {benchmark_filter!r} selects no cases in {executable}")

    case_dir = destination.parent / f"{destination.stem}.cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, object]] = []
    for index, name in enumerate(names):
        case_json = case_dir / f"{index:04d}.json"
        case_log = case_dir / f"{index:04d}.log"
        if resume and reusable_json(case_json):
            print(f"REUSE {executable.name} case {index + 1}/{len(names)}",
                  flush=True)
        else:
            command = wrapped_command(
                executable, f"^{regex_literal(name)}$", case_json,
                cpu, disable_aslr)
            print(f"RUN {executable.name} case {index + 1}/{len(names)}",
                  flush=True)
            with case_log.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        command, env=environment, text=True, stdout=log,
                        stderr=subprocess.STDOUT, timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    message = f"timeout after {timeout_seconds:g} seconds"
                    log.write(f"\n{message}\n")
                    case_json.write_text(
                        json.dumps(error_document(name, message), indent=2) +
                        "\n", encoding="utf-8")
                    print(f"TIMEOUT {executable.name} case "
                          f"{index + 1}/{len(names)}", flush=True)
                else:
                    if result.returncode != 0 or not reusable_json(case_json):
                        message = (f"benchmark process failed with exit code "
                                   f"{result.returncode}")
                        case_json.write_text(
                            json.dumps(error_document(name, message), indent=2) +
                            "\n", encoding="utf-8")
                        print(f"ERROR {executable.name} case "
                              f"{index + 1}/{len(names)}: {message}",
                              flush=True)
                    else:
                        print(f"DONE {executable.name} case "
                              f"{index + 1}/{len(names)}", flush=True)
        documents.append(json.loads(case_json.read_text(encoding="utf-8")))

    context = next((document.get("context", {}) for document in documents
                    if document.get("context")), {})
    benchmarks = [entry for document in documents
                  for entry in document.get("benchmarks", [])]
    destination.write_text(json.dumps({
        "context": context,
        "benchmarks": benchmarks,
    }, indent=2) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--case-timeout-seconds", type=float, default=0.0,
        help="isolate each registered row and terminate it after this many "
             "wall-clock seconds; zero disables isolation")
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
            if args.resume and reusable_json(destination):
                print(f"REUSE {binary_name}", flush=True)
                json_files.append(destination)
                continue
            if args.dry_run:
                command = wrapped_command(
                    executable, args.filter, destination,
                    args.cpu, args.disable_aslr)
                print("RUN", " ".join(command), flush=True)
            elif args.case_timeout_seconds > 0:
                run_isolated_cases(
                    executable, destination, args.filter, environment,
                    args.cpu, args.disable_aslr,
                    args.case_timeout_seconds, args.resume)
                json_files.append(destination)
            else:
                command = wrapped_command(
                    executable, args.filter, destination,
                    args.cpu, args.disable_aslr)
                print("RUN", " ".join(command), flush=True)
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
