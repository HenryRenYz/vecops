#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Merge Google Benchmark JSON files into one comparison CSV.

Benchmark names deliberately carry all semantic dimensions.  This script does
not guess tensor layouts from executable names; it only adds host/toolchain
metadata supplied on the command line or present in Google's JSON context.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = [
    "run_id", "host", "arch", "cpu", "compiler", "library", "version",
    "backend_isa", "implementation", "case", "model", "subgraph", "shape", "batch",
    "tokens", "chunk", "op", "dtype", "extent", "phase", "epilogue",
    "orientation", "tuning", "mc", "nc", "kc", "native_batch",
    "packing_included", "weights_prepacked", "call_multiplicity",
    "weight_reorder_required", "prepared_weight_bytes",
    "selected_kc", "selected_outer_block", "padding_ratio", "isa_fallback",
    "median_us", "p10_us", "p90_us", "cv", "logical_gflops",
    "openblas_raw_e2e_us", "speedup_vs_openblas",
    "speedup_vs_openblas_raw_e2e", "vecops_dynamic_extent",
    "vecops_dynamic_us", "vecops_const_us",
    "vecops_dynamic_speedup_vs_row", "vecops_const_speedup_vs_row",
    "status", "error", "source_json",
]


def parse_name(name: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for component in name.split("/")[1:]:
        if ":" in component:
            key, value = component.split(":", 1)
            fields[key] = value
    return fields


def to_microseconds(value: float, unit: str) -> float:
    scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    if unit not in scale:
        raise ValueError(f"unsupported benchmark time unit: {unit}")
    return value * scale[unit]


def metadata_for(path: Path, args: argparse.Namespace) -> dict[str, str]:
    result = {
        "run_id": args.run_id,
        "host": args.host,
        "cpu": args.cpu,
        "compiler": args.compiler,
        "backend_isa": args.backend_isa,
    }
    result.update(args.version_map)
    if args.metadata:
        document = json.loads(args.metadata.read_text(encoding="utf-8"))
        common = document.get("common", {})
        result.update({key: str(value) for key, value in common.items()
                       if str(value)})
        per_file = document.get("files", {}).get(path.name, {})
        result.update({key: str(value) for key, value in per_file.items()
                       if str(value)})
    for key in ("run_id", "host", "cpu", "compiler", "backend_isa"):
        explicit = getattr(args, key)
        if explicit:
            result[key] = explicit
    if args.vecops_version:
        result["version_vecops"] = args.vecops_version
    return result


def load_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    context = document.get("context", {})
    metadata = metadata_for(path, args)
    metadata["host"] = metadata["host"] or str(context.get("host_name", ""))
    metadata["cpu"] = metadata["cpu"] or str(context.get("cpu_info", ""))

    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for entry in document.get("benchmarks", []):
        base_name = str(entry.get("run_name") or entry.get("name", ""))
        aggregate = str(entry.get("aggregate_name", "sample"))
        groups[base_name][aggregate] = entry

    rows: list[dict[str, Any]] = []
    for base_name, aggregates in groups.items():
        sample = aggregates.get("median") or next(iter(aggregates.values()))
        dimensions = parse_name(base_name)
        row: dict[str, Any] = {field: "" for field in FIELDS}
        row.update({key: value for key, value in metadata.items()
                    if key in row})
        row.update({key: dimensions.get(key, "") for key in (
            "arch", "case", "model", "subgraph", "shape", "batch", "tokens",
            "chunk", "op", "dtype", "extent", "phase", "epilogue",
            "orientation", "tuning", "mc", "nc", "kc",
        )})
        row["library"] = dimensions.get("provider", "")
        row["version"] = metadata.get(
            f"version_{row['library'].lower().replace('-', '_')}", "")
        row["source_json"] = str(path)
        row["implementation"] = str(sample.get("label", ""))
        row["error"] = str(sample.get("error_message", ""))
        if row["error"].startswith("timeout after "):
            row["status"] = "timeout"
        else:
            row["status"] = (
                "skipped" if sample.get("error_occurred") else "ok")
        for counter in (
            "native_batch", "packing_included", "weights_prepacked",
            "call_multiplicity", "weight_reorder_required",
            "prepared_weight_bytes", "selected_kc", "selected_outer_block",
            "padding_ratio", "isa_fallback",
        ):
            if counter in sample:
                row[counter] = sample[counter]

        for aggregate, column in (
            ("median", "median_us"), ("p10", "p10_us"), ("p90", "p90_us"),
        ):
            entry = aggregates.get(aggregate)
            if entry and not entry.get("error_occurred"):
                row[column] = to_microseconds(
                    float(entry["real_time"]), str(entry["time_unit"]))
        cv = aggregates.get("cv")
        if cv and not cv.get("error_occurred"):
            row["cv"] = float(cv["real_time"])
        if row["median_us"]:
            m, n, k = (int(value) for value in dimensions["shape"].split("x"))
            flops = 2.0 * int(dimensions["batch"]) * m * n * k
            row["logical_gflops"] = flops / (float(row["median_us"]) * 1e3)
        rows.append(row)
    return rows


def comparison_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in (
        "host", "arch", "case", "op", "dtype"))


def add_reference_speedups(rows: list[dict[str, Any]]) -> None:
    same_phase_openblas: dict[tuple[str, ...], float] = {}
    raw_openblas: dict[tuple[str, ...], float] = {}
    vecops_dynamic: dict[tuple[str, ...], tuple[float, str]] = {}
    vecops_const: dict[tuple[str, ...], float] = {}
    for row in rows:
        if (row["library"] == "OpenBLAS" and row["status"] == "ok" and
                row["tuning"] in ("", "default") and row["median_us"]):
            key = (*comparison_key(row), str(row["phase"]))
            same_phase_openblas[key] = float(row["median_us"])
            if row["phase"] == "raw_e2e":
                raw_openblas[comparison_key(row)] = float(row["median_us"])
        if (row["library"] == "vecops" and row["status"] == "ok" and
                row["phase"] == "raw_e2e" and
                row["tuning"] in ("", "default") and row["median_us"]):
            key = comparison_key(row)
            if row["extent"] in ("AllDynamic", "TokenDynamic"):
                vecops_dynamic[key] = (
                    float(row["median_us"]), str(row["extent"]))
            elif row["extent"] == "AllConst":
                vecops_const[key] = float(row["median_us"])
    for row in rows:
        if not row["median_us"]:
            continue
        row_time = float(row["median_us"])
        key = comparison_key(row)
        same_phase = same_phase_openblas.get((*key, str(row["phase"])))
        if same_phase is not None:
            row["speedup_vs_openblas"] = same_phase / row_time
        raw_baseline = raw_openblas.get(key)
        if raw_baseline is not None:
            row["openblas_raw_e2e_us"] = raw_baseline
            row["speedup_vs_openblas_raw_e2e"] = raw_baseline / row_time
        dynamic = vecops_dynamic.get(key)
        if dynamic is not None:
            dynamic_time, dynamic_extent = dynamic
            row["vecops_dynamic_extent"] = dynamic_extent
            row["vecops_dynamic_us"] = dynamic_time
            row["vecops_dynamic_speedup_vs_row"] = row_time / dynamic_time
        const_time = vecops_const.get(key)
        if const_time is not None:
            row["vecops_const_us"] = const_time
            row["vecops_const_speedup_vs_row"] = row_time / const_time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--cpu", default="")
    parser.add_argument("--compiler", default="")
    parser.add_argument("--backend-isa", default="")
    parser.add_argument("--dependencies", type=Path,
                        default=Path(__file__).with_name("dependencies.json"))
    parser.add_argument("--vecops-version", default="")
    parser.add_argument(
        "--exclude-case", action="append", default=[],
        help="omit this case on every host (repeatable)")
    parser.add_argument(
        "--exclude-host-case", action="append", default=[], metavar="HOST:CASE",
        help="omit one host/case pair (repeatable)")
    args = parser.parse_args()

    dependency_document = json.loads(
        args.dependencies.read_text(encoding="utf-8"))
    args.version_map = {
        "version_openblas": dependency_document["libraries"]["OpenBLAS"]["revision"],
        "version_onednn": dependency_document["libraries"]["oneDNN"]["revision"],
        "version_libxsmm": dependency_document["libraries"]["LIBXSMM"]["revision"],
        "version_acl": dependency_document["libraries"]["Arm Compute Library"]["revision"],
        "version_kupl_mma": dependency_document["libraries"]["KUPL"]["revision"],
        "version_vecops": args.vecops_version,
    }

    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(load_rows(path, args))
    # Later inputs are deliberate reruns and replace the same semantic row.
    deduplicated: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in (
            "host", "arch", "library", "case", "op", "dtype", "extent",
            "phase", "tuning", "mc", "nc", "kc"))
        deduplicated[key] = row
    rows = list(deduplicated.values())
    excluded_host_cases: set[tuple[str, str]] = set()
    for value in args.exclude_host_case:
        if ":" not in value:
            parser.error(f"--exclude-host-case expects HOST:CASE, got {value!r}")
        excluded_host_cases.add(tuple(value.split(":", 1)))
    excluded_cases = set(args.exclude_case)
    rows = [
        row for row in rows
        if row["case"] not in excluded_cases
        and (str(row["host"]), str(row["case"])) not in excluded_host_cases
    ]
    add_reference_speedups(rows)
    rows.sort(key=lambda row: tuple(str(row[field]) for field in (
        "host", "case", "op", "phase", "library", "extent", "tuning")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
