# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Audit emitted bandwidth kernels, including NT instruction preservation.

Run on the build host, or pass --arch and a matching objdump explicitly.
Only the specialized run() functions are inspected, not CLI/string handling.
This intentionally fails on new unexpected compiler output for manual review.
"""
import argparse
from pathlib import Path
import platform
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--arch", choices=("x86_64", "aarch64"), default=platform.machine())
    parser.add_argument("--objdump", default="objdump")
    args = parser.parse_args()
    assert args.arch in ("x86_64", "aarch64"), args.arch
    messages = []
    for family in ("Scalar", "Vector"):
        obj = args.build_dir / "tools/memory/CMakeFiles/vecops-memory-bench.dir" / (family + "Kernels.cpp.o")
        text = subprocess.check_output([args.objdump, "-d", "-C", "--no-show-raw-insn", str(obj)], text=True)
        (args.build_dir / (family.lower() + ".asm")).write_text(text)
        functions = [f for f in re.split(r"(?m)^[0-9a-f]+ <", text)[1:]
                     if "::run<" in f.splitlines()[0]]
        assert functions, f"No {family} kernel symbols found"
        for function in functions:
            title, _, body = function.partition("\n")
            match = re.search(r"Operation\)(\d), (\d+)(?:ul)?, (true|false), (true|false), (true|false)", title)
            assert match, title
            op, extent, load_nt, store_nt, _ = match.groups()
            op, extent = int(op), int(extent)
            words = 1 << extent if family == "Vector" else extent
            if args.arch == "x86_64":
                assert not re.search(r"\bcall\b|\(%rsp\)|\(%rbp\)", body), title
                data = "\n".join(line for line in body.splitlines() if "%rip" not in line)
                if family == "Scalar":
                    assert not re.search(r"%(?:ymm|zmm)|\b(?:v?mov[au]ps|v?mov[au]pd|v?movdqu|v?addps|v?mulps)\b", body), title
                    loads = len(re.findall(r"vmovss\s+[^\n]*\([^\n]*,%xmm", data))
                    expected = 0 if op == 1 else words * (2 if op == 3 else 1)
                    assert loads == expected, (title, "scalar loads", loads, expected)
                    stores = len(re.findall(r"\bmovnti\b", data))
                    assert stores == (words if store_nt == "true" and op != 0 else 0), title
                else:
                    loads = len(re.findall(r"\bvmovntdqa\b", data))
                    stores = len(re.findall(r"\bvmovnt(?:dq|ps|pd)\b", data))
                    assert loads == (words * (2 if op == 3 else 1) if load_nt == "true" and op != 1 else 0), title
                    assert stores == (words if store_nt == "true" and op != 0 else 0), title
                    if op == 0:
                        all_loads = len(re.findall(r"\bvmov\w+\s+[^\n]*\([^\n]*,%(?:xmm|ymm|zmm)", data))
                        assert all_loads == words, (title, all_loads, words)
            else:
                assert not re.search(r"\bbl\b|\bblr\b|\[sp[,\]]|\[x29[,\]]", body), title
                if family == "Scalar":
                    assert not re.search(r"\bz\d+\b|\bq\d+\b|\bv\d+\.", body), title
                    loads = len(re.findall(r"\bldr\s+s\d+", body)) + 2 * len(re.findall(r"\bldnp\s+s\d+", body))
                    expected = 0 if op == 1 else words * (2 if op == 3 else 1)
                    assert loads == expected, (title, loads, expected)
                    stores = 2 * len(re.findall(r"\bstnp\s+s\d+", body))
                    assert stores == (words if store_nt == "true" and op != 0 else 0), title
                else:
                    loads = len(re.findall(r"\bldnt1w\b", body))
                    stores = len(re.findall(r"\bstnt1w\b", body))
                    assert loads == (words * (2 if op == 3 else 1) if load_nt == "true" and op != 1 else 0), title
                    assert stores == (words if store_nt == "true" and op != 0 else 0), title
                    if op == 0:
                        all_loads = len(re.findall(r"\b(?:ld1[bwhd]|ldnt1w)\b", body))
                        assert all_loads == words, (title, all_loads, words)
        messages.append(f"PASS {args.arch} {family}: {len(functions)} kernels, exact load/NT counts, no calls or stack traffic")
    report = "\n".join(messages) + "\n"
    (args.build_dir / "assembly-check.log").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
