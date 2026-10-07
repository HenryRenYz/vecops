#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Run one instruction benchmark case per perf-stat process and summarize it."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any


DEFAULT_EVENTS = "cycles:u,instructions:u,branches:u,branch-misses:u,task-clock"
MEMORY_EVENTS = (
    "cycles:u,instructions:u,L1-dcache-load-misses,"
    "LLC-load-misses,cache-misses,task-clock"
)


def command_output(command: list[str], cwd: pathlib.Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def cpu_numa_node(cpu: int) -> int:
    cpu_dir = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}")
    nodes = sorted(cpu_dir.glob("node[0-9]*"))
    if not nodes:
        return 0
    return int(nodes[0].name.removeprefix("node"))


def parse_number(text: str) -> float | None:
    value = text.strip()
    if not value or value.startswith("<not"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_perf_stat(path: pathlib.Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        count = parse_number(fields[0])
        event = fields[2].strip()
        if not event:
            continue
        runtime = parse_number(fields[3]) if len(fields) > 3 else None
        running_pct = parse_number(fields[4]) if len(fields) > 4 else None
        events.append(
            {
                "event": event,
                "count": count,
                "unit": fields[1].strip(),
                "event_runtime": runtime,
                "running_pct": running_pct,
            }
        )
    return events


def event_count(events: list[dict[str, Any]], requested: str) -> float | None:
    requested_base = requested.removesuffix(":u")
    for event in events:
        actual = str(event["event"])
        actual_base = actual.removesuffix(":u")
        if actual == requested or actual_base == requested_base:
            return event["count"]
    return None


def benchmark_time_ns(row: dict[str, Any]) -> float:
    value = float(row["real_time"])
    unit = str(row.get("time_unit", "ns"))
    factors = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
    return value * factors.get(unit, 1.0)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def describe_context(binary: pathlib.Path, cpu: int, node: int, command: list[str]) -> dict[str, Any]:
    repo = pathlib.Path(__file__).resolve().parents[2]
    sibling_path = pathlib.Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )
    governor_path = pathlib.Path(
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    )
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "hostname": command_output(["hostname"]),
        "uname": command_output(["uname", "-a"]),
        "lscpu": command_output(["lscpu"]),
        "compiler": command_output(["c++", "--version"]),
        "perf": command_output(["perf", "--version"]),
        "numactl": command_output(["numactl", "--show"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], repo),
        "git_status": command_output(["git", "status", "--short"], repo),
        "binary": str(binary),
        "cpu": cpu,
        "numa_node": node,
        "thread_siblings": sibling_path.read_text().strip() if sibling_path.exists() else "unknown",
        "scaling_governor": governor_path.read_text().strip() if governor_path.exists() else "unknown",
        "invocation": command,
    }


def summarize(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    mean = statistics.fmean(values)
    mad = statistics.median(abs(value - median) for value in values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "min": min(values),
        "median": median,
        "mean": mean,
        "mad": mad,
        "cv_percent": 100.0 * stdev / mean if mean else math.nan,
    }


def write_markdown(
    path: pathlib.Path,
    arch: str,
    cpu: int,
    node: int,
    loops: int,
    repetitions: int,
    summaries: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    lines = [
        f"# Instruction benchmark: {arch}",
        "",
        f"CPU `{cpu}`, NUMA node `{node}`, inner loops `{loops}`, repetitions `{repetitions}`.",
        "",
        "| instruction or sequence | cycle interpretation | ISA | working set | raw cycles/sequence | baseline-adjusted cycles/sequence | raw bytes/cycle | adjusted bytes/cycle | GB/s | raw elements/cycle | adjusted elements/cycle | Gelem/s | CV |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        net = row.get("net_cycles_per_sequence_median")
        net_text = f"{net:.4f}" if isinstance(net, float) and math.isfinite(net) else "N/A"
        bytes_per_cycle = row.get("bytes_per_cycle_median")
        bytes_text = (
            f"{bytes_per_cycle:.4f}"
            if isinstance(bytes_per_cycle, float) and math.isfinite(bytes_per_cycle)
            else "N/A"
        )
        net_bytes_per_cycle = row.get("net_bytes_per_cycle_median")
        net_bytes_text = (
            f"{net_bytes_per_cycle:.4f}"
            if isinstance(net_bytes_per_cycle, float) and math.isfinite(net_bytes_per_cycle)
            else "N/A"
        )
        bandwidth = row.get("bandwidth_gb_s_median")
        bandwidth_text = (
            f"{bandwidth:.3f}"
            if isinstance(bandwidth, float) and math.isfinite(bandwidth)
            else "N/A"
        )
        working_set = int(row.get("working_set_bytes", 0))
        working_set_text = str(working_set) if working_set else "-"
        elements_per_cycle = row.get("elements_per_cycle_median")
        elements_text = (
            f"{elements_per_cycle:.4f}"
            if isinstance(elements_per_cycle, float) and math.isfinite(elements_per_cycle)
            else "N/A"
        )
        net_elements_per_cycle = row.get("net_elements_per_cycle_median")
        net_elements_text = (
            f"{net_elements_per_cycle:.4f}"
            if isinstance(net_elements_per_cycle, float)
            and math.isfinite(net_elements_per_cycle)
            else "N/A"
        )
        conversion_rate = row.get("conversion_gigaelements_s_median")
        conversion_text = (
            f"{conversion_rate:.3f}"
            if isinstance(conversion_rate, float) and math.isfinite(conversion_rate)
            else "N/A"
        )
        lines.append(
            f"| `{row['name']}` | {row['cycle_interpretation']} | {row['isa']} | "
            f"{working_set_text} | "
            f"{row['cycles_per_sequence_median']:.4f} | "
            f"{net_text} | {bytes_text} | {net_bytes_text} | {bandwidth_text} | "
            f"{elements_text} | {net_elements_text} | {conversion_text} | "
            f"{row['cycles_per_sequence_cv_percent']:.2f}% |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--numa-node", type=int)
    parser.add_argument("--loops", type=int, default=10_000_000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--filter", default=".*", help="Python regex matched against full benchmark names")
    parser.add_argument(
        "--events",
        help="Comma-separated perf events for every case; default selects core/memory presets",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--dump-asm", action="store_true")
    args = parser.parse_args()

    if args.cpu < 0 or args.loops <= 0 or args.repetitions <= 0:
        parser.error("--cpu must be non-negative; --loops and --repetitions must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"benchmark binary does not exist: {binary}")
    for tool in ("perf", "numactl"):
        if shutil.which(tool) is None:
            parser.error(f"required tool is not available: {tool}")

    node = args.numa_node if args.numa_node is not None else cpu_numa_node(args.cpu)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        output = repo / "benchmarks" / "results" / "inst" / timestamp
    else:
        output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_dir = output / "runs"
    run_dir.mkdir()

    list_result = subprocess.run(
        [
            "numactl",
            f"--physcpubind={args.cpu}",
            f"--membind={node}",
            str(binary),
            "--inst_list_json",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    listing = json.loads(list_result.stdout)
    pattern = re.compile(args.filter)
    all_cases = listing["cases"]
    selected = [case for case in all_cases if pattern.search(case["benchmark_name"])]
    cases_by_name = {case["name"]: case for case in all_cases}
    empty_case = cases_by_name.get("empty_loop")
    if not selected:
        raise SystemExit(f"no cases matched --filter={args.filter!r}")
    required_baselines: list[dict[str, Any]] = []
    for case in list(selected):
        baseline_name = str(case.get("baseline_name", "empty_loop"))
        baseline = cases_by_name.get(baseline_name)
        if baseline is not None and baseline not in selected and baseline not in required_baselines:
            required_baselines.append(baseline)
    if empty_case is not None and empty_case not in selected and empty_case not in required_baselines:
        required_baselines.insert(0, empty_case)
    selected = required_baselines + selected

    explicit_events = (
        [event.strip() for event in args.events.split(",") if event.strip()]
        if args.events
        else None
    )
    raw_rows: list[dict[str, Any]] = []
    measurements: dict[str, list[dict[str, Any]]] = {
        case["benchmark_name"]: [] for case in selected
    }
    warnings: list[str] = []

    invocation = [str(value) for value in sys.argv]
    context = describe_context(binary, args.cpu, node, invocation)
    context.update(
        {
            "arch": listing["arch"],
            "loops": args.loops,
            "repetitions": args.repetitions,
            "events": explicit_events if explicit_events is not None else {
                "core": DEFAULT_EVENTS.split(","),
                "memory": MEMORY_EVENTS.split(","),
            },
            "case_count": len(selected),
        }
    )
    (output / "context.json").write_text(
        json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for case_index, case in enumerate(selected):
        events = explicit_events or [
            event.strip()
            for event in (
                MEMORY_EVENTS if case.get("category") == "memory" else DEFAULT_EVENTS
            ).split(",")
            if event.strip()
        ]
        case_tag = f"{case_index:03d}_{safe_name(case['benchmark_name'])}"
        for repetition in range(args.repetitions):
            stem = run_dir / f"{case_tag}_r{repetition + 1:02d}"
            perf_output = stem.with_suffix(".perf.csv")
            benchmark_output = stem.with_suffix(".benchmark.json")
            log_output = stem.with_suffix(".log")
            exact_filter = "^" + re.escape(case["benchmark_name"]) + "$"
            command = [
                "numactl",
                f"--physcpubind={args.cpu}",
                f"--membind={node}",
                "perf",
                "stat",
                "--no-big-num",
                "-x",
                ";",
            ]
            for event in events:
                command.extend(["-e", event])
            command.extend(
                [
                    "-o",
                    str(perf_output),
                    "--",
                    str(binary),
                    f"--benchmark_filter={exact_filter}",
                    "--benchmark_min_time=1x",
                    "--benchmark_repetitions=1",
                    "--benchmark_color=false",
                    f"--benchmark_out={benchmark_output}",
                    "--benchmark_out_format=json",
                    f"--inst_loops={args.loops}",
                ]
            )
            environment = os.environ.copy()
            environment["LC_ALL"] = "C"
            result = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            log_output.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"case {case['benchmark_name']} repetition {repetition + 1} failed; "
                    f"see {log_output} and {perf_output}"
                )

            perf_events = parse_perf_stat(perf_output)
            bench_document = json.loads(benchmark_output.read_text(encoding="utf-8"))
            bench_rows = [
                row for row in bench_document.get("benchmarks", [])
                if row.get("run_type", "iteration") == "iteration"
            ]
            if len(bench_rows) != 1:
                raise RuntimeError(f"expected one benchmark row in {benchmark_output}")
            outer_iterations = int(bench_rows[0]["iterations"])
            elapsed_ns = benchmark_time_ns(bench_rows[0])
            sequences = (
                outer_iterations * args.loops * int(case["sequences_per_loop"])
            )
            cycles = event_count(perf_events, "cycles:u")
            instructions = event_count(perf_events, "instructions:u")
            if cycles is None or cycles <= 0:
                raise RuntimeError(f"cycles counter unavailable in {perf_output}")
            measurement = {
                "cycles": cycles,
                "instructions": instructions,
                "outer_iterations": outer_iterations,
                "sequences": sequences,
                "cycles_per_loop": cycles / (outer_iterations * args.loops),
                "cycles_per_sequence": cycles / sequences,
                "sequences_per_cycle": sequences / cycles,
                "elapsed_ns": elapsed_ns,
            }
            bytes_per_sequence = int(case.get("bytes_per_sequence", 0))
            logical_bytes = sequences * bytes_per_sequence
            measurement["logical_bytes"] = logical_bytes
            measurement["bytes_per_cycle"] = logical_bytes / cycles if logical_bytes else math.nan
            measurement["bandwidth_gb_s"] = (
                logical_bytes / elapsed_ns if logical_bytes and elapsed_ns > 0 else math.nan
            )
            elements_per_sequence = int(case.get("elements_per_sequence", 0))
            converted_elements = sequences * elements_per_sequence
            measurement["converted_elements"] = converted_elements
            measurement["elements_per_cycle"] = (
                converted_elements / cycles if converted_elements else math.nan
            )
            measurement["conversion_gigaelements_s"] = (
                converted_elements / elapsed_ns
                if converted_elements and elapsed_ns > 0 else math.nan
            )
            measurements[case["benchmark_name"]].append(measurement)

            for event in perf_events:
                raw_rows.append(
                    {
                        "benchmark_name": case["benchmark_name"],
                        "name": case["name"],
                        "mode": case["mode"],
                        "repetition": repetition + 1,
                        "event": event["event"],
                        "count": event["count"],
                        "unit": event["unit"],
                        "event_runtime": event["event_runtime"],
                        "running_pct": event["running_pct"],
                        "outer_iterations": outer_iterations,
                        "inner_loops": args.loops,
                        "sequences": sequences,
                    }
                )
                running_pct = event["running_pct"]
                if running_pct is not None and running_pct < 95.0:
                    warnings.append(
                        f"{case['benchmark_name']} run {repetition + 1}: "
                        f"{event['event']} counter ran only {running_pct:.2f}% of enabled time"
                    )

            print(
                f"[{case_index + 1}/{len(selected)} r{repetition + 1}/{args.repetitions}] "
                f"{case['benchmark_name']}: {measurement['cycles_per_sequence']:.4f} cycles/sequence",
                flush=True,
            )

    with (output / "raw_perf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    baseline_cycles_per_loop: dict[str, float] = {}
    for case in selected:
        if case["mode"] != "control":
            continue
        baseline_cycles_per_loop[case["name"]] = statistics.median(
            value["cycles_per_loop"] for value in measurements[case["benchmark_name"]]
        )

    # perf counts process startup and Google Benchmark bookkeeping.  Compare
    # retired instructions against the matching empty-loop run so that the
    # validation covers only the incremental instructions in the asm body.
    if empty_case is not None and empty_case["benchmark_name"] in measurements:
        baseline_measurements = measurements[empty_case["benchmark_name"]]
        for case in selected:
            if case["benchmark_name"] == empty_case["benchmark_name"]:
                continue
            expected_increment = (
                args.loops
                * int(case["sequences_per_loop"])
                * int(case["instructions_per_sequence"])
            )
            if expected_increment == 0 or case.get("category") == "memory":
                continue
            for repetition, (measurement, baseline) in enumerate(
                zip(measurements[case["benchmark_name"]], baseline_measurements), start=1
            ):
                if measurement["instructions"] is None or baseline["instructions"] is None:
                    continue
                observed_increment = measurement["instructions"] - baseline["instructions"]
                relative_error = abs(observed_increment - expected_increment) / expected_increment
                if relative_error > 0.02:
                    warnings.append(
                        f"{case['benchmark_name']} run {repetition}: baseline-adjusted retired "
                        f"instructions differ from kernel expectation by {relative_error * 100.0:.2f}%"
                    )

    summaries: list[dict[str, Any]] = []
    for case in selected:
        case_measurements = measurements[case["benchmark_name"]]
        cycle_stats = summarize([value["cycles_per_sequence"] for value in case_measurements])
        throughput_stats = summarize([value["sequences_per_cycle"] for value in case_measurements])
        bytes_per_cycle_values = [
            value["bytes_per_cycle"] for value in case_measurements
            if math.isfinite(value["bytes_per_cycle"])
        ]
        bandwidth_values = [
            value["bandwidth_gb_s"] for value in case_measurements
            if math.isfinite(value["bandwidth_gb_s"])
        ]
        elements_per_cycle_values = [
            value["elements_per_cycle"] for value in case_measurements
            if math.isfinite(value["elements_per_cycle"])
        ]
        conversion_rate_values = [
            value["conversion_gigaelements_s"] for value in case_measurements
            if math.isfinite(value["conversion_gigaelements_s"])
        ]
        instruction_values = [
            value["instructions"] for value in case_measurements
            if value["instructions"] is not None
        ]
        row: dict[str, Any] = {
            "benchmark_name": case["benchmark_name"],
            "name": case["name"],
            "mode": case["mode"],
            "cycle_interpretation": {
                "throughput": "throughput_cycles_per_sequence",
                "latency": "dependent_latency_cycles_per_sequence",
                "control": "control_cycles_per_sequence",
            }.get(case["mode"], "unknown"),
            "isa": case["isa"],
            "description": case["description"],
            "category": case.get("category", "unknown"),
            "vector_bits": case["vector_bits"],
            "memory_pattern": case.get("memory_pattern", "none"),
            "working_set_level": case.get("working_set_level", "none"),
            "working_set_bytes": case.get("working_set_bytes", 0),
            "bytes_per_sequence": case.get("bytes_per_sequence", 0),
            "conversion_kind": case.get("conversion_kind", "none"),
            "source_element_bits": case.get("source_element_bits", 0),
            "destination_element_bits": case.get("destination_element_bits", 0),
            "elements_per_sequence": case.get("elements_per_sequence", 0),
            "sequences_per_loop": case["sequences_per_loop"],
            "instructions_per_sequence": case["instructions_per_sequence"],
            "dependency_chains": case["dependency_chains"],
            "cycles_per_sequence_min": cycle_stats["min"],
            "cycles_per_sequence_median": cycle_stats["median"],
            "cycles_per_sequence_mean": cycle_stats["mean"],
            "cycles_per_sequence_mad": cycle_stats["mad"],
            "cycles_per_sequence_cv_percent": cycle_stats["cv_percent"],
            "sequences_per_cycle_median": throughput_stats["median"],
            "net_sequences_per_cycle_median": math.nan,
            "retired_instructions_median": (
                statistics.median(instruction_values) if instruction_values else math.nan
            ),
            "net_cycles_per_sequence_median": math.nan,
            "bytes_per_cycle_median": (
                statistics.median(bytes_per_cycle_values)
                if bytes_per_cycle_values else math.nan
            ),
            "net_bytes_per_cycle_median": math.nan,
            "bandwidth_gb_s_median": (
                statistics.median(bandwidth_values) if bandwidth_values else math.nan
            ),
            "elements_per_cycle_median": (
                statistics.median(elements_per_cycle_values)
                if elements_per_cycle_values else math.nan
            ),
            "net_elements_per_cycle_median": math.nan,
            "conversion_gigaelements_s_median": (
                statistics.median(conversion_rate_values)
                if conversion_rate_values else math.nan
            ),
        }
        baseline_name = str(case.get("baseline_name", "empty_loop"))
        baseline = baseline_cycles_per_loop.get(baseline_name)
        if baseline is not None and case["mode"] != "control":
            case_cycles_per_loop = statistics.median(
                value["cycles_per_loop"] for value in case_measurements
            )
            row["net_cycles_per_sequence_median"] = (
                case_cycles_per_loop - baseline
            ) / int(case["sequences_per_loop"])
            net_cycles = row["net_cycles_per_sequence_median"]
            if net_cycles > 0:
                row["net_sequences_per_cycle_median"] = 1.0 / net_cycles
                bytes_per_sequence = int(case.get("bytes_per_sequence", 0))
                if bytes_per_sequence:
                    row["net_bytes_per_cycle_median"] = (
                        bytes_per_sequence / net_cycles
                    )
                elements_per_sequence = int(case.get("elements_per_sequence", 0))
                if elements_per_sequence:
                    row["net_elements_per_cycle_median"] = (
                        elements_per_sequence / net_cycles
                    )
        if cycle_stats["cv_percent"] > 5.0:
            warnings.append(
                f"{case['benchmark_name']}: cycles/sequence CV is "
                f"{cycle_stats['cv_percent']:.2f}% (> 5%)"
            )
        summaries.append(row)

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    write_markdown(
        output / "summary.md",
        listing["arch"],
        args.cpu,
        node,
        args.loops,
        args.repetitions,
        summaries,
        list(dict.fromkeys(warnings)),
    )
    if args.dump_asm:
        asm = command_output(
            ["objdump", "-Cd", "--demangle", "--no-show-raw-insn", str(binary)]
        )
        (output / "instruction_kernels.asm").write_text(asm + "\n", encoding="utf-8")

    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
