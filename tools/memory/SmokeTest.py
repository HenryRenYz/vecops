# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exercise real kernels, synchronization, placement, and exported accounting.

Uses only the Python standard library. Disturbed samples may return 1; this
checks correctness/accounting, not an expected performance score. Exit 2,
verification failures, missing measurements and malformed exports always fail.
All generated files live under --output-dir, never a shared temporary directory.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--peer-cpu", type=int, default=1)
    parser.add_argument("--second-node", type=int)
    parser.add_argument("--memory-only-node", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    common = ["--numa", str(args.node), "--cpu", str(args.cpu),
              "--seconds", ".02", "--warmup", ".01", "--samples", "1",
              "--retries", "1", "--max-skew", ".02"]
    checked = 0

    def run(name, command, extra=()):
        nonlocal checked
        csv_path = args.output_dir / (name + ".csv")
        json_path = args.output_dir / (name + ".jsonl")
        result = subprocess.run([binary, command, *common, *extra,
                                 "--csv", str(csv_path), "--json", str(json_path)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120)
        (args.output_dir / (name + ".log")).write_text(result.stdout)
        assert result.returncode in (0, 1), result.stdout
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        json_rows = [json.loads(line) for line in json_path.read_text().splitlines()]
        assert rows and rows == json_rows, "CSV and JSON Lines differ"
        for row in rows:
            if row["status"] == "unsupported":
                continue
            assert row["status"] in {"ok", "retry", "skew", "interrupted", "page-fault"}, row
            assert row["placement"] == "all-pages-verified", row
            duration = float(row["elapsed_seconds"])
            assert duration > 0, row
            begin, end = int(row["start_ns"]), int(row["end_ns"])
            assert math.isclose(duration, (end - begin) / 1e9, rel_tol=1e-10), row
            if row["kind"] == "bandwidth":
                read, write = int(row["read_bytes"]), int(row["write_bytes"])
                assert read + write > 0, row
                assert math.isclose(float(row["total_GB_s"]), (read + write) / duration / 1e9,
                                    rel_tol=1e-10), row
                op = row["operation"]
                if op == "read": assert write == 0
                if op == "write": assert read == 0
                if op == "copy": assert read == write
                if op == "triad": assert read == 2 * write
            else:
                assert int(row["accesses"]) > 0, row
                assert math.isclose(float(row["ns_per_access"]),
                                    (end - begin) / int(row["accesses"]), rel_tol=1e-10), row
        # Aggregate each attempt independently. Never sum independently timed rates.
        for total in (r for r in rows if r["scope"] == "total"):
            threads = [r for r in rows if r["scope"].startswith("thread:")
                       and r["kind"] == "bandwidth"
                       and all(r[k] == total[k] for k in ("config", "sample", "attempt"))]
            assert int(total["read_bytes"]) == sum(int(r["read_bytes"]) for r in threads)
            assert int(total["write_bytes"]) == sum(int(r["write_bytes"]) for r in threads)
            assert int(total["start_ns"]) == min(int(r["start_ns"]) for r in threads)
            assert int(total["end_ns"]) == max(int(r["end_ns"]) for r in threads)
        checked += len(rows)
        print(f"PASS {name}: {len(rows)} raw rows (exit {result.returncode})")
        return rows

    rows = run("matrix", "bandwidth", ["--op", "read,write,copy,triad", "--kernel",
               "scalar16,scalar32,scalar64,vec1,vec2,vec4", "--access", "temporal,stream",
               "--size", "16KiB,1MiB"])
    measured = {(r["operation"], r["kernel"], r["requested_access"], r["requested_size_bytes"])
                for r in rows if r["scope"] == "total"}
    unsupported = {(r["operation"], r["kernel"], r["requested_access"], r["requested_size_bytes"])
                   for r in rows if r["status"] == "unsupported"}
    assert len(measured | unsupported) == 96
    run("separate-multiply-add", "bandwidth", ["--op", "triad", "--kernel", "scalar64,vec4",
        "--fma", "off", "--size", "16KiB"])
    run("group-size", "bandwidth", ["--op", "copy", "--kernel", "vec4", "--size", "32KiB",
        "--size-scope", "group-array", "--cpu", f"{args.cpu},{args.peer_cpu}"])
    run("chase", "latency", ["--size", "16KiB,1MiB", "--seed", "17"])
    run("loaded", "latency", ["--loaded", "--latency-cpu", str(args.cpu), "--cpu", str(args.peer_cpu),
        "--op", "triad", "--kernel", "vec4", "--size", "1MiB"])
    run("pingpong", "pingpong", ["--peer-cpu", str(args.peer_cpu)])
    if args.second_node is not None:
        run("cross-node", "bandwidth", ["--op", "copy,triad", "--kernel", "vec4",
            "--numa", f"{args.node}:{args.second_node}", "--size", "1MiB"])
        run("aggregate", "bandwidth", ["--op", "triad", "--kernel", "vec4",
            "--numa", f"{args.node},{args.second_node}", "--size", "1MiB"])
        run("cross-pingpong", "pingpong", ["--numa", f"{args.node}:{args.second_node}",
            "--exec-node", "src", "--peer-cpu", str(args.cpu)])
    if args.memory_only_node is not None:
        run("memory-only", "bandwidth", ["--op", "read,write", "--kernel", "vec1",
            "--numa", str(args.memory_only_node), "--exec-node", str(args.node), "--size", "1MiB"])
    invalid = [
        ["--cpu", "0,,1"], ["--cpu", "3-1"], ["--cpu", "0,0"],
        ["--cpu", "1000000"], ["--size", "0"], ["--size", "18446744073709551615GiB"],
        ["--seconds", "nan"], ["--samples", "0"], ["--op", "bogus"],
        ["--op", "read", "--numa", f"{args.node}:{args.node}"],
        ["--size", "3B"], ["--size", "1B", "--size-scope", "group-array", "--cpu", f"{args.cpu},{args.peer_cpu}"],
    ]
    for extra in invalid:
        result = subprocess.run([binary, "bandwidth", *common, "--dry-run", *extra],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        assert result.returncode == 2, (extra, result.stdout.decode())
    bad = subprocess.run([binary, "latency", *common, "--loaded", "--latency-cpu", str(args.cpu),
                          "--dry-run"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert bad.returncode == 2, bad.stdout.decode()
    print(f"PASS invalid arguments: {len(invalid) + 1} cases")
    print(f"PASS all checks: {checked} measured/exported rows")


if __name__ == "__main__":
    main()
