#!/usr/bin/env python3
"""Compare median Matmul timings from two Google Benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


RUN_OPTIONS_RE = re.compile(r"/min_time:[^/]+/repeats:[^/]+$")
TIME_TO_NS = {"s": 1.0e9, "ms": 1.0e6, "us": 1.0e3, "ns": 1.0}


def load_medians(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    medians: dict[str, dict[str, object]] = {}
    for entry in document.get("benchmarks", []):
        if entry.get("aggregate_name") != "median":
            continue
        name = str(entry.get("run_name", entry.get("name", "")))
        key = RUN_OPTIONS_RE.sub("", name)
        unit = str(entry.get("time_unit", "ns"))
        medians[key] = {
            "cpu_ns": float(entry["cpu_time"]) * TIME_TO_NS[unit],
            "fields": parse_fields(key),
        }
    return medians


def merge_medians(paths: list[Path]) -> dict[str, dict[str, object]]:
    runs = [load_medians(path) for path in paths]
    keys = runs[0].keys()
    if any(run.keys() != keys for run in runs[1:]):
        raise SystemExit("repeated benchmark files contain different case sets")
    return {
        key: {
            "cpu_ns": geometric_mean([
                float(run[key]["cpu_ns"]) for run in runs
            ]),
            "fields": runs[0][key]["fields"],
        }
        for key in keys
    }


def parse_fields(name: str) -> dict[str, str]:
    parts = name.split("/")
    fields = {"family": parts[1] if len(parts) > 1 else "unknown"}
    for part in parts[2:]:
        if ":" in part:
            key, value = part.split(":", 1)
            fields[key] = value
    return fields


def geometric_mean(values: list[float]) -> float:
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--before-extra", type=Path, action="append", default=[])
    parser.add_argument("--after-extra", type=Path, action="append", default=[])
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--detail", type=Path)
    args = parser.parse_args()

    before = merge_medians([args.before, *args.before_extra])
    after = merge_medians([args.after, *args.after_extra])
    if before.keys() != after.keys():
        missing_after = sorted(before.keys() - after.keys())
        missing_before = sorted(after.keys() - before.keys())
        raise SystemExit(
            f"benchmark sets differ: missing after={len(missing_after)}, "
            f"missing before={len(missing_before)}"
        )

    details: list[dict[str, object]] = []
    groups: dict[
        tuple[str, str, str], list[tuple[float, float, float]]
    ] = defaultdict(list)
    for name in sorted(before):
        fields = before[name]["fields"]
        assert isinstance(fields, dict)
        before_ns = float(before[name]["cpu_ns"])
        after_ns = float(after[name]["cpu_ns"])
        speedup = before_ns / after_ns
        group = (
            str(fields.get("dtype", "unknown")),
            str(fields.get("family", "unknown")),
            str(fields.get("extent", "unknown")),
        )
        groups[group].append((before_ns, after_ns, speedup))
        details.append({
            "backend": args.backend,
            "dtype": group[0],
            "family": group[1],
            "case": fields.get("case", ""),
            "shape": fields.get("shape", ""),
            "input": fields.get("input", ""),
            "extent": group[2],
            "before_ns": before_ns,
            "after_ns": after_ns,
            "speedup": speedup,
            "name": name,
        })

    print(
        "| 后端 | dtype | shape family | metadata | case数 | "
        "before 几何ns | after 几何ns | 几何加速比 | 最差 | 最好 |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (dtype, family, extent), samples in sorted(groups.items()):
        before_values = [sample[0] for sample in samples]
        after_values = [sample[1] for sample in samples]
        speedups = [sample[2] for sample in samples]
        print(
            f"| {args.backend} | {dtype} | {family} | {extent} | "
            f"{len(samples)} | {geometric_mean(before_values):.3f} | "
            f"{geometric_mean(after_values):.3f} | "
            f"{geometric_mean(speedups):.4f}x | "
            f"{min(speedups):.4f}x | {max(speedups):.4f}x |"
        )

    totals: dict[tuple[str, str], list[tuple[float, float, float]]] = (
        defaultdict(list)
    )
    for (dtype, _family, extent), samples in groups.items():
        totals[(dtype, extent)].extend(samples)
        totals[("ALL", extent)].extend(samples)
    print("\n| 后端 | dtype | metadata | case数 | before 几何ns | after 几何ns | 几何加速比 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (dtype, extent), samples in sorted(totals.items()):
        before_values = [sample[0] for sample in samples]
        after_values = [sample[1] for sample in samples]
        speedups = [sample[2] for sample in samples]
        print(
            f"| {args.backend} | {dtype} | {extent} | {len(samples)} | "
            f"{geometric_mean(before_values):.3f} | "
            f"{geometric_mean(after_values):.3f} | "
            f"{geometric_mean(speedups):.4f}x |"
        )

    if args.detail is not None:
        args.detail.parent.mkdir(parents=True, exist_ok=True)
        with args.detail.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=details[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
