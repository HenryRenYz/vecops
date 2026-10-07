#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Summarize matmul kernel text size and matrix instructions in binaries.

The report is deliberately independent of a particular benchmark target so it
can compare an optimization-cycle binary with a validation-copy binary. Symbol
sizes come from ``nm --print-size --size-sort``. Instruction counts are static
occurrences in the matched kernel symbol ranges, not retired instructions.

For the split-K probe, the route columns heuristically classify the canonical
rank-2 FP32 accumulator by its dynamic inner stride. An original row-major C
endpoint has a compile-time unit inner stride. An unrecognized endpoint remains
``unknown`` rather than being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass


KERNEL_MARKERS = (
    "matmul_details::Backend<vecops::kernel::matmul_implementation::SME>::run_tiles_phased_shared",
    "matmul_details::sme::microkernel_phased_shared",
    "matmul_details::amx::exact_microkernel",
    "matmul_details::amx::microkernel",
    "matmul_details::sme::microkernel_body",
    "matmul_details::sme::microkernel_shared",
    "matmul_details::sme::microkernel",
)

ROUTES = (
    "unsplit", "split_shared", "first", "middle", "last", "unknown")

AMX_DOT_RE = re.compile(
    r"\b(?:tdpbf16ps|tdpfp16ps|tdpbssd|tdpbsud|tdpbusd|tdpbuud)\b")
SME_MOPA_RE = re.compile(
    r"\b(?:bfmopa|fmopa|smopa|umopa|sumopa|usmopa)\b")
TILE_LOAD_RE = re.compile(r"\btileloadd(?:t1)?\b")
TILE_STORE_RE = re.compile(r"\btilestored\b")
INSTRUCTION_ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):")


@dataclass(frozen=True)
class KernelSymbol:
    address: int
    size: int
    name: str
    route: str


@dataclass
class RouteCounts:
    symbols: int = 0
    symbol_bytes: int = 0
    matrix_static: int = 0


@dataclass
class Summary:
    binary: str
    text_bytes: int
    kernel_symbols: int
    kernel_symbol_bytes: int
    amx_dot_static: int
    sme_mopa_static: int
    tile_load_static: int
    tile_store_static: int
    unsplit_symbols: int
    unsplit_symbol_bytes: int
    unsplit_matrix_static: int
    split_shared_symbols: int
    split_shared_symbol_bytes: int
    split_shared_matrix_static: int
    first_symbols: int
    first_symbol_bytes: int
    first_matrix_static: int
    middle_symbols: int
    middle_symbol_bytes: int
    middle_matrix_static: int
    last_symbols: int
    last_symbol_bytes: int
    last_matrix_static: int
    unknown_symbols: int
    unknown_symbol_bytes: int
    unknown_matrix_static: int


def run(*args: str) -> str:
    try:
        return subprocess.run(
            args, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError(
            f"required binary-analysis tool not found: {args[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"command failed ({' '.join(args)}){suffix}") from error


def analysis_tool(llvm_name: str, generic_name: str) -> str:
    """Prefer LLVM tools: GNU binutils cannot demangle Clang constraint ABI."""
    return shutil.which(llvm_name) or shutil.which(generic_name) or generic_name


def split_top_level(arguments: str) -> list[str]:
    """Split a demangled C++ template argument list at top-level commas."""
    depth = 0
    start = 0
    result: list[str] = []
    for index, character in enumerate(arguments):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def template_arguments(name: str, marker: str) -> list[str] | None:
    start = name.find(marker)
    if start < 0:
        return None
    start += len(marker)
    if start >= len(name) or name[start] != "<":
        return None
    depth = 0
    for end in range(start, len(name)):
        if name[end] == "<":
            depth += 1
        elif name[end] == ">":
            depth -= 1
            if depth == 0:
                return split_top_level(name[start + 1:end])
    return None


def nested_template_arguments(value: str, template: str) -> list[str] | None:
    start = value.find(template + "<")
    if start < 0:
        return None
    start += len(template)
    depth = 0
    for end in range(start, len(value)):
        if value[end] == "<":
            depth += 1
        elif value[end] == ">":
            depth -= 1
            if depth == 0:
                return split_top_level(value[start + 1:end])
    return None


def unit_inner_stride(endpoint: str) -> bool | None:
    strides = nested_template_arguments(endpoint, "vecops::tensor::Strides")
    if strides is None or len(strides) < 2:
        return None
    return "vecops::meta::Const<1" in strides[-1]


def classify_route(name: str) -> str:
    """Classify the current probe's original-C/accumulator endpoint pair."""
    # Clang can omit a constrained parameter type from the demangled spelling,
    # so recognize both the shared entry name and its explicit route type.
    if ("run_tiles_phased_shared<" in name or
            "microkernel_phased_shared<" in name or
            "DynamicAccumulatorRoute" in name):
        return "split_shared"
    arguments = None
    for marker in KERNEL_MARKERS:
        arguments = template_arguments(name, marker)
        if arguments is not None:
            break
    if arguments is None:
        return "unknown"
    inputs = [
        argument for argument in arguments
        if "vecops::tensor::InputDataAccess<" in argument
    ]
    outputs = [
        argument for argument in arguments
        if "vecops::tensor::OutputDataAccess<" in argument
    ]
    if len(inputs) < 3 or not outputs:
        return "unknown"
    c_input = inputs[-1]
    c_output = outputs[-1]
    input_unit = unit_inner_stride(c_input)
    output_unit = unit_inner_stride(c_output)
    if input_unit is None or output_unit is None:
        return "unknown"

    input_is_original = (
        "vecops::tensor::ZeroVecTransform<" in c_input or input_unit)
    input_is_accumulator = (
        "vecops::tensor::NoTransform" in c_input and not input_unit)
    if input_is_original and output_unit:
        return "unsplit"
    if input_is_original and not output_unit:
        return "first"
    if input_is_accumulator and not output_unit:
        return "middle"
    if input_is_accumulator and output_unit:
        return "last"
    return "unknown"


def parsed_symbols(
        nm_output: str, *, kernel_only: bool) -> list[KernelSymbol]:
    symbol_re = re.compile(
        r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\S\s+(.*)$")
    by_range: dict[tuple[int, int], KernelSymbol] = {}
    ambiguous: set[tuple[int, int]] = set()
    for line in nm_output.splitlines():
        match = symbol_re.match(line)
        if not match:
            continue
        symbol_type = line.split(None, 3)[2]
        if symbol_type.lower() not in ("t", "w"):
            continue
        if kernel_only and not any(
                marker in match.group(3) for marker in KERNEL_MARKERS):
            continue
        address = int(match.group(1), 16)
        size = int(match.group(2), 16)
        if size == 0:
            continue
        symbol = KernelSymbol(
            address, size, match.group(3), classify_route(match.group(3)))
        key = (address, size)
        previous = by_range.get(key)
        if previous is not None and previous.route != symbol.route:
            ambiguous.add(key)
        else:
            by_range.setdefault(key, symbol)
    for key in ambiguous:
        symbol = by_range[key]
        by_range[key] = KernelSymbol(
            symbol.address, symbol.size, symbol.name, "unknown")
    return sorted(by_range.values(), key=lambda symbol: symbol.address)


def matrix_instruction_addresses(disassembly: str) -> list[int]:
    result = []
    for line in disassembly.splitlines():
        address_match = INSTRUCTION_ADDRESS_RE.match(line)
        if address_match is None:
            continue
        instruction = line.lower()
        if AMX_DOT_RE.search(instruction) or SME_MOPA_RE.search(instruction):
            result.append(int(address_match.group(1), 16))
    return result


def kernel_symbols(nm_output: str, disassembly: str) -> list[KernelSymbol]:
    """Find explicit leaf symbols and owners of force-inlined matrix ops."""
    explicit = parsed_symbols(nm_output, kernel_only=True)
    all_functions = parsed_symbols(nm_output, kernel_only=False)
    by_range = {(symbol.address, symbol.size): symbol for symbol in explicit}
    for address in matrix_instruction_addresses(disassembly):
        if owning_route(address, list(by_range.values())) is not None:
            continue
        owners = [
            symbol for symbol in all_functions
            if symbol.address <= address < symbol.address + symbol.size
        ]
        if not owners:
            continue
        # Prefer the innermost/most specific sized symbol when aliases or
        # compiler-generated enclosing regions overlap.
        owner = min(owners, key=lambda symbol: symbol.size)
        by_range[(owner.address, owner.size)] = owner
    return sorted(by_range.values(), key=lambda symbol: symbol.address)


def owning_route(address: int, symbols: list[KernelSymbol]) -> str | None:
    owners = [
        symbol.route for symbol in symbols
        if symbol.address <= address < symbol.address + symbol.size
    ]
    if not owners:
        return None
    return owners[0] if all(route == owners[0] for route in owners) \
        else "unknown"


def summarize(binary: pathlib.Path) -> Summary:
    size_tool = analysis_tool("llvm-size", "size")
    nm_tool = analysis_tool("llvm-nm", "nm")
    objdump_tool = analysis_tool("llvm-objdump", "objdump")
    size_output = run(size_tool, "-A", str(binary))
    text_bytes = 0
    for line in size_output.splitlines():
        fields = line.split()
        if fields and fields[0] == ".text":
            text_bytes = int(fields[1], 0)
            break

    if pathlib.Path(nm_tool).name.startswith("llvm-"):
        nm_output = run(
            nm_tool, "--demangle", "--print-size", "--size-sort", str(binary))
    else:
        # Deep DataAccess endpoint types exceed binutils' default demangler
        # recursion limit on the x86 probe.
        nm_output = run(
            nm_tool, "-C", "--no-recurse-limit", "-S", "--size-sort",
            str(binary))
    if pathlib.Path(objdump_tool).name.startswith("llvm-"):
        disassembly = run(objdump_tool, "-d", "--demangle", str(binary))
    else:
        disassembly = run(
            objdump_tool, "-d", "-C", "--no-recurse-limit", str(binary))
    symbols = kernel_symbols(nm_output, disassembly)
    routes = {route: RouteCounts() for route in ROUTES}
    for symbol in symbols:
        routes[symbol.route].symbols += 1
        routes[symbol.route].symbol_bytes += symbol.size

    amx_dot_static = 0
    sme_mopa_static = 0
    tile_load_static = 0
    tile_store_static = 0
    for line in disassembly.splitlines():
        address_match = INSTRUCTION_ADDRESS_RE.match(line)
        if address_match is None:
            continue
        route = owning_route(int(address_match.group(1), 16), symbols)
        if route is None:
            continue
        instruction = line.lower()
        amx = bool(AMX_DOT_RE.search(instruction))
        sme = bool(SME_MOPA_RE.search(instruction))
        amx_dot_static += amx
        sme_mopa_static += sme
        tile_load_static += bool(TILE_LOAD_RE.search(instruction))
        tile_store_static += bool(TILE_STORE_RE.search(instruction))
        routes[route].matrix_static += amx or sme

    values: list[int | str] = [
        str(binary.resolve()),
        text_bytes,
        len(symbols),
        sum(symbol.size for symbol in symbols),
        amx_dot_static,
        sme_mopa_static,
        tile_load_static,
        tile_store_static,
    ]
    for route in ROUTES:
        values.extend((
            routes[route].symbols,
            routes[route].symbol_bytes,
            routes[route].matrix_static,
        ))
    return Summary(*values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binaries", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    missing_tools = [
        generic for llvm, generic in (
            ("llvm-size", "size"),
            ("llvm-nm", "nm"),
            ("llvm-objdump", "objdump"),
        ) if not shutil.which(llvm) and not shutil.which(generic)
    ]
    if missing_tools:
        parser.error(
            "missing required binary-analysis tool(s): " +
            ", ".join(missing_tools))
    missing_binaries = [
        str(path) for path in args.binaries if not path.is_file()
    ]
    if missing_binaries:
        parser.error("binary not found: " + ", ".join(missing_binaries))
    try:
        rows = [summarize(path) for path in args.binaries]
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    fieldnames = list(Summary.__dataclass_fields__)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", newline="")
    else:
        import sys
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
