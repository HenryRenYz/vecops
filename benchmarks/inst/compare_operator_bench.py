#!/usr/bin/env python3
"""Compare ours and baseline Google Benchmark CSV output.

Both inputs must be produced by the same benchmark target.
Rows are joined by operator, shape, dtype, and (when present) approximation
mode.

``ours_speedup_x`` is ``baseline_time / ours_time``; values greater than one
mean that ours is faster. ``ours_time_reduction_pct`` is positive when ours
takes less time than baseline.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import pathlib
import re
import sys
from dataclasses import dataclass


TIME_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}
AGGREGATE_PRIORITY = {"median": 3, "mean": 2, "iteration": 1}
AGGREGATE_RE = re.compile(r"_(mean|median|stddev|cv)$")


@dataclass(frozen=True)
class CaseKey:
    operator: str
    group: str
    rank: int
    shape: str
    dtype: str
    domain: str


@dataclass
class Measurement:
    key: CaseKey
    mode: str
    arch: str
    aggregate: str
    time_ns: float
    bytes_per_second: float
    items_per_second: float


def benchmark_rows(path: pathlib.Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines)
            if line.startswith("name,")
        )
    except StopIteration as exc:
        raise ValueError(f"{path}: Google Benchmark CSV header not found") from exc
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))


def finite_number(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        result = float(value)
    except ValueError:
        return math.nan
    return result if math.isfinite(result) else math.nan


def parse_measurement(row: dict[str, str], metric: str) -> Measurement | None:
    if row.get("error_occurred", "").lower() == "true":
        return None
    name = row.get("name", "")
    aggregate_match = AGGREGATE_RE.search(name)
    aggregate = aggregate_match.group(1) if aggregate_match else "iteration"
    if aggregate in {"stddev", "cv"}:
        return None

    segments = name.split("/")
    if len(segments) < 2:
        return None
    fields: dict[str, str] = {}
    for segment in segments[2:]:
        if ":" in segment:
            key, value = segment.split(":", 1)
            fields[key] = value
    required = ("rank", "shape", "dtype")
    if any(field not in fields for field in required):
        return None

    time_value = finite_number(row.get(metric))
    unit = row.get("time_unit", "")
    if not math.isfinite(time_value) or unit not in TIME_TO_NS:
        return None
    key = CaseKey(
        operator=segments[0],
        group=segments[1],
        rank=int(fields["rank"]),
        shape=fields["shape"],
        dtype=fields["dtype"],
        domain=fields.get("domain", ""),
    )
    return Measurement(
        key=key,
        mode=fields.get("mode", ""),
        arch=fields.get("arch", ""),
        aggregate=aggregate,
        time_ns=time_value * TIME_TO_NS[unit],
        bytes_per_second=finite_number(row.get("bytes_per_second")),
        items_per_second=finite_number(row.get("items_per_second")),
    )


def load_measurements(
    path: pathlib.Path, metric: str, distinguish_mode: bool
) -> dict[tuple[CaseKey, str], Measurement]:
    selected: dict[tuple[CaseKey, str], Measurement] = {}
    for row in benchmark_rows(path):
        measurement = parse_measurement(row, metric)
        if measurement is None:
            continue
        map_key = (
            measurement.key,
            measurement.mode if distinguish_mode else "",
        )
        old = selected.get(map_key)
        if (
            old is None
            or AGGREGATE_PRIORITY[measurement.aggregate]
            > AGGREGATE_PRIORITY[old.aggregate]
        ):
            selected[map_key] = measurement
    return selected


def format_number(value: float) -> str:
    return f"{value:.9g}" if math.isfinite(value) else ""


def compare(
    ours_path: pathlib.Path,
    baseline_path: pathlib.Path,
    output_path: pathlib.Path,
    metric: str,
) -> tuple[int, int, int]:
    ours = load_measurements(ours_path, metric, distinguish_mode=True)
    baseline = load_measurements(
        baseline_path, metric, distinguish_mode=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operator",
        "group",
        "rank",
        "shape",
        "dtype",
        "domain",
        "mode",
        "ours_arch",
        "baseline_arch",
        "aggregate",
        f"ours_{metric}_ns",
        f"baseline_{metric}_ns",
        "ours_speedup_x",
        "ours_time_reduction_pct",
        "ours_bytes_per_second",
        "baseline_bytes_per_second",
        "ours_items_per_second",
        "baseline_items_per_second",
    ]
    matched = 0
    ours_keys: set[CaseKey] = set()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (key, mode), ours_value in sorted(
            ours.items(),
            key=lambda item: (
                item[0][0].operator,
                item[0][0].group,
                item[0][0].rank,
                item[0][0].shape,
                item[0][0].dtype,
                item[0][0].domain,
                item[0][1],
            ),
        ):
            ours_keys.add(key)
            baseline_value = baseline.get((key, mode))
            if baseline_value is None:
                continue
            speedup = baseline_value.time_ns / ours_value.time_ns
            time_reduction = (
                1.0 - (ours_value.time_ns / baseline_value.time_ns)
            ) * 100.0
            writer.writerow(
                {
                    "operator": key.operator,
                    "group": key.group,
                    "rank": key.rank,
                    "shape": key.shape,
                    "dtype": key.dtype,
                    "domain": key.domain,
                    "mode": mode,
                    "ours_arch": ours_value.arch,
                    "baseline_arch": baseline_value.arch,
                    "aggregate": (
                        f"{ours_value.aggregate}/"
                        f"{baseline_value.aggregate}"
                    ),
                    f"ours_{metric}_ns": format_number(
                        ours_value.time_ns
                    ),
                    f"baseline_{metric}_ns": format_number(
                        baseline_value.time_ns
                    ),
                    "ours_speedup_x": format_number(speedup),
                    "ours_time_reduction_pct": format_number(
                        time_reduction
                    ),
                    "ours_bytes_per_second": format_number(
                        ours_value.bytes_per_second
                    ),
                    "baseline_bytes_per_second": format_number(
                        baseline_value.bytes_per_second
                    ),
                    "ours_items_per_second": format_number(
                        ours_value.items_per_second
                    ),
                    "baseline_items_per_second": format_number(
                        baseline_value.items_per_second
                    ),
                }
            )
            matched += 1

    baseline_keys = {key for key, _ in baseline}
    return (
        matched,
        len(ours_keys - baseline_keys),
        len(baseline_keys - ours_keys),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join ours and baseline Google Benchmark CSV rows by operator, "
            "input shape, rank, and dtype."
        )
    )
    parser.add_argument("ours_csv", type=pathlib.Path)
    parser.add_argument("baseline_csv", type=pathlib.Path)
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--metric",
        choices=("cpu_time", "real_time"),
        default="cpu_time",
        help="time field used for the comparison (default: cpu_time)",
    )
    args = parser.parse_args()

    for path in (args.ours_csv, args.baseline_csv):
        if not path.is_file():
            parser.error(f"input CSV does not exist: {path}")

    try:
        matched, ours_only, baseline_only = compare(
            args.ours_csv,
            args.baseline_csv,
            args.output,
            args.metric,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if matched == 0:
        raise SystemExit("no comparable benchmark rows were found")
    print(
        f"Wrote {matched} comparison rows to {args.output} "
        f"(ours-only cases: {ours_only}, baseline-only cases: "
        f"{baseline_only})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
