#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-backed manifest for the C++ matmul benchmark catalogs.

This module deliberately keeps declaration-level entries (including entries
which the DSL runner cannot represent) instead of pretending that the small
DSL suite is a complete mirror of the C++ benchmark suite.  It is importable
by runners and also has a JSON/text CLI:

    python benchmarks/dsl_matmul_catalog.py --list
    python benchmarks/dsl_matmul_catalog.py --json
    python benchmarks/dsl_matmul_catalog.py --json --architecture x86 --feature HAS_AMX_INT8

The default ARM view remains compatible with prior source-backed execution
IDs. The x86 view selects source ISA gates before expansion (AMX_FP16 is not
enabled implicitly), retaining original source line numbers. Standalone
packing/stateful/weight-reuse contracts use a separate supplemental inventory;
they must not be represented as ordinary stateless matmul rows.
Use ``complete_registration=True`` (CLI ``--complete-registration``) for the
full compute inventory: this additionally includes ScenarioBench and the
previously omitted Const batched packed registrations. OrientationBench is an
explicit optional CMake target. PolicyBench exists only on ARM with fixed SVL;
the x86 inventory retains an excluded-build record instead of a fake policy.

The parser is intentionally conservative: explicit numeric registrations are
decoded, while symbolic template registrations are retained as unsupported
source records with their source line and expression.  Thus additions to the
C++ catalog cannot silently disappear from the audit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
  "MatmulBench.cpp",
  "MatmulBenchCommon.h",
  "MatmulScenarioBenchCommon.h",
  "MatmulFusionBench.cpp",
  "MatmulOrientationBench.cpp",
  "MatmulFamilyBench.cpp",
  "MatmulPolicyBench.cpp",
  "MatmulPackedBench.cpp",
  "MatmulPackBenchCommon.h",
  "MatmulScenarioBench.cpp",
)
OPS = ROOT / "benchmarks" / "ops"


@dataclasses.dataclass(frozen=True)
class CatalogCase:
  """One source-declared case or a preserved non-equivalent declaration."""

  catalog: str
  source: str
  line: int
  name: str
  m: int | None = None
  n: int | None = None
  k: int | None = None
  dtype_a: str | None = None
  dtype_b: str | None = None
  accumulator: str | None = None
  native_a: str | None = None
  native_b: str | None = None
  memory_a: str | None = None
  memory_b: str | None = None
  memory_c: str | None = None
  memory_c_input: str | None = None
  input_modes: tuple[str, ...] = ()
  input_pipeline_a: str | None = None
  input_pipeline_b: str | None = None
  output_pipeline: str | None = None
  metadata: tuple[str, ...] = ()
  batch: str | None = None
  broadcast: str | None = None
  orientation: str | None = None
  configuration: str | None = None
  arm_condition: str = "ARM/SVE-SME source branch; SME F64 cases require HAS_SME_F64F64"
  dsl_supported: bool = False
  unsupported_reason: str | None = None
  declaration: str = ""
  svl_bytes: int | None = None
  lane: int | None = None

  def as_dict(self) -> dict:
    result = dataclasses.asdict(self)
    for key in ("input_modes", "metadata"):
      result[key] = list(result[key])
    return result


def _read_sources(root: Path) -> dict[str, str]:
  return {name: (root / "benchmarks" / "ops" / name).read_text() for name in SOURCE_FILES}


def _architecture_source(source: str, features: frozenset[str]) -> str:
  """Select concrete preprocessor branches without changing source line numbers.

  Target-shard conditionals remain enabled so every compiled shard contributes
  its registration. This is inventory selection, not a C++ preprocessor.
  """
  symbols = set(features) | {"VECOPS_TARGET_SHARD_ACTIVE", "VECOPS_WEIGHT_REUSE_TARGET_SHARD_ACTIVE"}
  stack = []
  active = True
  output = []
  lines = source.splitlines(keepends=True)
  index = 0
  while index < len(lines):
    line = lines[index]
    directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
    if not directive:
      output.append(line if active else "\n" if line.endswith("\n") else "")
      index += 1
      continue
    command, expression = directive.groups()
    count = 1
    while expression.rstrip().endswith("\\"):
      expression = expression.rstrip()[:-1] + lines[index + count].strip()
      count += 1

    def condition(text):
      text = re.sub(r"defined\s*\(?\s*(\w+)\s*\)?", lambda m: str(int(m[1] in symbols)), text)
      text = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: str(int(m[0] in symbols)), text)
      text = text.replace("&&", " and ").replace("||", " or ")
      text = re.sub(r"!(?!=)", " not ", text)
      if re.search(r"[^0-9\s()<>=!&|+*/.\-andort]", text):
        raise ValueError(f"unsupported inventory preprocessor condition: {text}")
      return bool(eval(text.strip(), {"__builtins__": {}}, {}))

    if command in {"if", "ifdef", "ifndef"}:
      selected = condition(expression) if command == "if" else expression.strip() in symbols
      if command == "ifndef":
        selected = not selected
      stack.append((active, selected))
      active = active and selected
    elif command == "elif":
      parent, taken = stack[-1]
      selected = not taken and condition(expression)
      stack[-1] = (parent, taken or selected)
      active = parent and selected
    elif command == "else":
      parent, taken = stack[-1]
      active = parent and not taken
      stack[-1] = (parent, True)
    else:
      active, _ = stack.pop()
    output.extend("\n" if x.endswith("\n") else "" for x in lines[index : index + count])
    index += count
  if stack:
    raise ValueError("unbalanced inventory preprocessor condition")
  return "".join(output)

def _line(source: str, offset: int) -> int:
  return source.count("\n", 0, offset) + 1


def _balanced_calls(source: str, function: str) -> Iterable[tuple[int, str]]:
  """Yield function-call bodies, balancing nested templates and parentheses."""
  needle = function
  start = 0
  while True:
    pos = source.find(needle, start)
    if pos < 0:
      return
    # Calls may have a template argument list between name and paren.
    i = pos + len(needle)
    while i < len(source) and source[i].isspace():
      i += 1
    if i < len(source) and source[i] == "<":
      depth = 0
      while i < len(source):
        if source[i] == "<":
          depth += 1
        elif source[i] == ">":
          depth -= 1
          if depth == 0:
            i += 1
            break
        i += 1
    while i < len(source) and source[i].isspace():
      i += 1
    if i >= len(source) or source[i] != "(":
      start = pos + len(needle)
      continue
    begin = i + 1
    depth = 1
    i += 1
    while i < len(source) and depth:
      if source[i] == "(":
        depth += 1
      elif source[i] == ")":
        depth -= 1
      i += 1
    yield pos, source[begin : i - 1]
    start = i


def _split_top(text: str, separator: str = ",") -> list[str]:
  parts, begin, paren, angle, brace = [], 0, 0, 0, 0
  for i, ch in enumerate(text):
    if ch == "(":
      paren += 1
    elif ch == ")":
      paren -= 1
    elif ch == "<":
      angle += 1
    elif ch == ">":
      angle = max(0, angle - 1)
    elif ch == "{":
      brace += 1
    elif ch == "}":
      brace -= 1
    elif ch == separator and not (paren or angle or brace):
      parts.append(text[begin:i].strip())
      begin = i + 1
  parts.append(text[begin:].strip())
  return parts


def _template_args(source: str, pos: int, func: str) -> list[str]:
  i = pos + len(func)
  while i < len(source) and source[i].isspace():
    i += 1
  if i >= len(source) or source[i] != "<":
    return []
  begin, depth = i + 1, 1
  i += 1
  while i < len(source) and depth:
    if source[i] == "<":
      depth += 1
    elif source[i] == ">":
      depth -= 1
      if depth == 0:
        break
    i += 1
  template = source[begin:i]
  # These catalogs put architecture selection inside the template argument
  # list. The manifest is specifically the ARM view, so choose the #else arm
  # of ARCH_X86_FAMILY while keeping non-architecture selector directives
  # visible to the caller.
  template = re.sub(
    r"#if\s+defined\(ARCH_X86_FAMILY\)\s*(.*?)\s*#else\s*(.*?)\s*#endif",
    lambda match: match.group(2),
    template,
    flags=re.DOTALL,
  )
  return _split_top(template)


def _eval_int(expr: str, env: dict[str, int] | None = None) -> int | None:
  expr = re.sub(r"\b(?:nint_t|cint)\b", "", expr).strip()
  env = env or {}
  if not re.fullmatch(r"[\w\s+*()/-]+", expr):
    return None
  names = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
  if names - set(env):
    return None
  try:
    return int(eval(expr, {"__builtins__": {}}, env))
  except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
    return None


def _normalize_dtype(value: str) -> str | None:
  """Map a complete C++ dtype token, never a substring of another token."""
  mapping = {
    "bfloat16_t": "bf16",
    "float16_t": "fp16",
    "float32_t": "fp32",
    "float64_t": "fp64",
    "int8_t": "int8",
    "uint8_t": "uint8",
    "int32_t": "int32",
    "float32": "fp32",
    "int8": "int8",
  }
  # Template expressions may wrap a type in aliases/qualifiers. Prefer an
  # exact identifier match; uint8_t must not be mistaken for int8_t.
  identifiers = re.findall(r"[A-Za-z_]\w*", value.strip())
  for token in identifiers:
    if token in mapping:
      return mapping[token]
  return None


def _extent_cases(filename: str, source: str) -> list[CatalogCase]:
  result: list[CatalogCase] = []
  # Shape declarations are templated as <Atom, M, N, K, ...>. For unresolved
  # Tile/KStep expressions, retain the expression instead of omitting it.
  for function in (
    "register_extent_pair",
    "register_policy_extent_pair",
    "register_required_family_extent_pair",
    "register_scenario_extent_pair",
  ):
    for pos, body in _balanced_calls(source, function):
      args = _template_args(source, pos, function)
      if len(args) < 4:
        continue
      shape_args = (
        args[-3:]
        if function in {"register_policy_extent_pair", "register_required_family_extent_pair"}
        else (args[-4:-1] if function == "register_scenario_extent_pair" else args[1:4])
      )
      m, n, k = (_eval_int(x) for x in shape_args)
      quoted = re.findall(r'"([^"]+)"', body)
      name = quoted[-1] if quoted else f"{function}@{_line(source, pos)}"
      expression = f"{function}<{', '.join(args)}>({body.strip()})"
      modes = tuple(x.rsplit("::", 1)[-1].lower() for x in args if "InputMode::" in x)
      dsl_supported = m is not None and n is not None and k is not None
      result.append(
        CatalogCase(
          catalog=filename.removesuffix(".cpp").removesuffix(".h"),
          source=f"benchmarks/ops/{filename}",
          line=_line(source, pos),
          name=name,
          m=m,
          n=n,
          k=k,
          input_modes=modes,
          metadata=("Const", "Dynamic") if function.endswith("extent_pair") else (),
          input_pipeline_a=next((x.rsplit("::", 1)[-1] for x in args if "InputPipeline::" in x), None),
          output_pipeline=next((x.rsplit("::", 1)[-1] for x in args if "OutputPipeline::" in x), None),
          configuration="; ".join(x for x in args if "Policy" in x or "Family" in x) or None,
          dsl_supported=dsl_supported,
          unsupported_reason=None
          if dsl_supported
          else "C++ template shape uses symbolic Tile/KStep or dependent type; preserve source declaration and resolve per ARM atom using the selected runtime vector width before DSL execution",
          declaration=expression,
        )
      )
  return result


def _scenario_shape_cases(source: str) -> list[CatalogCase]:
  """Decode rank-2 and batched shape tables in ScenarioBenchCommon.h."""
  result: list[CatalogCase] = []
  # These declarations are C++ aggregate entries: name,m,n,k,mode or
  # name,batch,m,n,k,broadcast_b[,broadcast_a]. KStep expressions are kept
  # explicitly unresolved because the instantiated atom determines K_R.
  for fn in (
    "make_scenario_cases",
    "make_fusion_scenario_cases",
    "make_batched_scenario_cases",
    "make_batched_fusion_scenario_cases",
  ):
    match = re.search(r"inline std::vector<[^>]+>\s+" + fn + r"\([^)]*\)\s*\{(.*?)\n\}", source, re.DOTALL)
    if not match:
      continue
    block = match.group(1)
    pattern = re.compile(r'\{"([^\"]+)"\s*,\s*([^{}]+?)\}')
    for row in pattern.finditer(block):
      name, rest = row.groups()
      values = _split_top(rest)
      batched = "batched" in fn
      # The first string is the name for batched entries and the group
      # for rank-2 MatmulCase entries; in the latter, values[0] is name.
      group = name if not batched else None
      if not batched and values and values[0].startswith('"'):
        name = f"{group}/{values[0].strip(chr(34))}"
      offset = 1
      batch = _eval_int(values[0]) if batched and values else None
      m_expr, n_expr, k_expr = values[offset : offset + 3] if len(values) >= offset + 3 else ("", "", "")
      m, n, k = _eval_int(m_expr), _eval_int(n_expr), _eval_int(k_expr)
      flags = values[offset + 3 :] if len(values) > offset + 3 else []
      modes = tuple(v.rsplit("::", 1)[-1].lower() for v in values if "InputMode::" in v)
      result.append(
        CatalogCase(
          catalog="ScenarioShape",
          source="benchmarks/ops/MatmulScenarioBenchCommon.h",
          line=_line(source, match.start() + row.start()),
          name=f"{fn}/{name}",
          m=m,
          n=n,
          k=k,
          input_modes=modes,
          metadata=("Dynamic",),
          batch=str(batch) if batched else None,
          broadcast=(
            f"A={'shared' if len(flags) > 1 and 'true' in flags[1] else 'independent'}, B={'shared' if flags and 'true' in flags[0] else 'independent'}"
            if batched
            else None
          ),
          dsl_supported=False,
          unsupported_reason=(
            "K is parameterized by atom K_R and this declaration is a scenario template; DSL parity requires atom/pipeline/mode expansion"
            if k is None
            else "Batched scenario semantics (shared A/B broadcast or independent batches) are not equivalent to rank-2 DSL matmul"
            if batched
            else "Raw scenario shape is declared, but input/output pipeline and memory/compute dtype configuration are registered elsewhere and must be joined before calling equivalent"
          ),
          declaration=row.group(0),
        )
      )
  return result


def _matmul_case_tables(source: str, *, lane: int | None = None, f64_lane: int | None = None) -> list[CatalogCase]:
  """Read all explicit MatmulCase vectors from MatmulBenchCommon.h."""
  result: list[CatalogCase] = []
  signature = re.compile(r"(?:inline\s+)?std::vector<MatmulCase>\s+(\w+)\([^)]*\)\s*\{")
  for match in signature.finditer(source):
    name = match.group(1)
    begin = match.end()
    depth = 1
    i = begin
    while i < len(source) and depth:
      if source[i] == "{":
        depth += 1
      elif source[i] == "}":
        depth -= 1
      i += 1
    body = source[begin : i - 1]
    for row in re.finditer(r'\{\s*"([^\"]+)"\s*,\s*"([^\"]+)"\s*,([^{}]+)\}', body):
      group, case_name, rest = row.groups()
      values = _split_top(rest)
      if len(values) < 3:
        continue
      # KStep belongs to the Atom registration in MatmulBench.cpp, not
      # the shape table. Leaving it unresolved is safer than inventing
      # the BF16 value (2) for FP32/F64 (1) or I8 (4).
      env = {"Tile": lane, "tile": lane, "F64Tile": f64_lane}
      env = {key: value for key, value in env.items() if value is not None}
      m, n, k = (_eval_int(x, env) for x in values[:3])
      modes = tuple(x.rsplit("::", 1)[-1].lower() for x in values[3:] if "InputMode::" in x)
      result.append(
        CatalogCase(
          catalog="MatmulShape",
          source="benchmarks/ops/MatmulBenchCommon.h",
          line=_line(source, match.start() + row.start()),
          name=f"{name}/{group}/{case_name}",
          m=m,
          n=n,
          k=k,
          input_modes=modes,
          metadata=("Dynamic",),
          dsl_supported=m is not None and n is not None and k is not None,
          unsupported_reason=None
          if m is not None and n is not None and k is not None
          else "Shape uses Tile/KStep/MaxStrip symbolic parameters; resolve using selected ARM Atom tile and K_R from MatmulBench.cpp registration",
          declaration=row.group(0),
        )
      )
  return result


def _orientation_cases(source: str) -> list[CatalogCase]:
  result = []
  pattern = re.compile(
    r'register_pair\s*<\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>\s*\("([^"]+)"\)'
  )
  for match in pattern.finditer(source):
    at, bt, ct, m, n, k, name = match.groups()
    for swap in (True, False):
      result.append(
        CatalogCase(
          "MatmulOrientation",
          "benchmarks/ops/MatmulOrientationBench.cpp",
          _line(source, match.start()),
          f"{name}/swap_{'on' if swap else 'off'}",
          int(m),
          int(n),
          int(k),
          native_a="bf16",
          native_b="bf16",
          dtype_a="bf16",
          dtype_b="bf16",
          accumulator="fp32",
          orientation=f"A={'T' if at == 'true' else 'R'},B={'T' if bt == 'true' else 'R'},C={'T' if ct == 'true' else 'R'}",
          memory_a="bf16",
          memory_b="bf16",
          memory_c="fp32",
          memory_c_input="fp32",
          input_pipeline_a="Convert",
          input_pipeline_b="Convert",
          output_pipeline="Convert",
          input_modes=("raw",),
          metadata=("Const",),
          configuration=f"EnableSwapAB={str(swap).lower()}",
          dsl_supported=True,
          declaration=match.group(0),
        )
      )
  conversion = re.compile(r'register_conversion_pair\s*<\s*(.*?)\s*>\s*\(\s*"([^"]+)"\s*\)', re.DOTALL)
  for match in conversion.finditer(source):
    args = _split_top(match.group(1))
    if len(args) != 5 or not all(_eval_int(x) is not None for x in args[2:]):
      continue
    ma_expr, mb_expr = args[:2]
    m, n, k = (_eval_int(x) for x in args[2:])
    name = match.group(2)
    dtype_a = _normalize_dtype(ma_expr) or (
      "fp32" if "float32_t" in ma_expr else "bf16" if "PrimaryAtom::TA" in ma_expr else None
    )
    dtype_b = _normalize_dtype(mb_expr) or (
      "fp32" if "float32_t" in mb_expr else "bf16" if "PrimaryAtom::TB" in mb_expr else None
    )
    for swap in (True, False):
      result.append(
        CatalogCase(
          "MatmulOrientation",
          "benchmarks/ops/MatmulOrientationBench.cpp",
          _line(source, match.start()),
          f"{name}/swap_{'on' if swap else 'off'}",
          int(m),
          int(n),
          int(k),
          dtype_a=dtype_a,
          dtype_b=dtype_b,
          native_a="bf16",
          native_b="bf16",
          accumulator="fp32",
          orientation="A=T,B=T,C=T",
          memory_a=dtype_a,
          memory_b=dtype_b,
          memory_c="fp32",
          memory_c_input="fp32",
          input_pipeline_a="Convert",
          input_pipeline_b="Convert",
          output_pipeline="Convert",
          input_modes=("raw",),
          metadata=("Const",),
          configuration=f"EnableSwapAB={str(swap).lower()}",
          dsl_supported=True,
          declaration=match.group(0),
        )
      )
  return result


def _pack_cases(source: str) -> list[CatalogCase]:
  result = []
  pattern = re.compile(
    r"PackCase<\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(true|false)\s*,\s*(true|false))?\s*>"
  )
  names = dict(re.findall(r'VECOPS_PACK_CASE_NAME\((\w+)\s*,\s*"([^"]+)"\)', source))
  for match in pattern.finditer(source):
    group, name_type, spatial, k, pack_a, pack_b = match.groups()
    name = names.get(name_type, name_type)
    result.append(
      CatalogCase(
        "MatmulPack",
        "benchmarks/ops/MatmulPackBenchCommon.h",
        _line(source, match.start()),
        name,
        int(spatial),
        1,
        int(k),
        input_modes=("pack_a" if pack_a != "false" else "", "pack_b" if pack_b != "false" else ""),
        metadata=("Const", "Dynamic"),
        configuration=f"group={group}; pack_a={pack_a or 'true'}; pack_b={pack_b or 'true'}",
        dsl_supported=False,
        unsupported_reason="DSL matmul benchmark does not model standalone MatmulPack layout/padding output and its element/byte counters",
        declaration=match.group(0),
      )
    )
  return result


def _vocab_cases(filename: str, source: str) -> list[CatalogCase]:
  """Preserve complete scenario vocabularies and batch/broadcast registrars."""
  result = []
  for enum in ("InputPipeline", "OutputPipeline", "InputMode"):
    match = re.search(r"enum class " + enum + r"\s*:[^{]+\{([^}]+)\}", source)
    if not match:
      continue
    for value in re.findall(r"\b([A-Za-z]\w*)\b(?:\s*=\s*[^,]+)?\s*,?", match.group(1)):
      if value in {"uint32_t", "uint8_t"}:
        continue
      result.append(
        CatalogCase(
          catalog="ScenarioVocabulary",
          source=f"benchmarks/ops/{filename}",
          line=_line(source, match.start()),
          name=f"{enum}.{value}",
          input_modes=(value,) if enum == "InputMode" else (),
          input_pipeline_a=value if enum == "InputPipeline" else None,
          output_pipeline=value if enum == "OutputPipeline" else None,
          dsl_supported=False,
          unsupported_reason="Vocabulary entry is not itself an executable shape; concrete scenario registrations are separately retained, and unsupported fusion semantics must not be inferred from the enum alone",
          declaration=match.group(0),
        )
      )
  # Explicitly inventory registrar families, especially batch/broadcast paths
  # whose shape semantics are not encoded in the ordinary rank-2 DSL runner.
  for token in (
    "register_scenario",
    "register_batched_scenario",
    "register_batched_scenario_extent_pair",
    "register_broadcast_scenario",
    "broadcast_a",
    "broadcast_b",
  ):
    for match in re.finditer(r"\b" + token + r"\b", source):
      result.append(
        CatalogCase(
          catalog="ScenarioRegistration",
          source=f"benchmarks/ops/{filename}",
          line=_line(source, match.start()),
          name=token,
          batch="batched" if "batched" in token else None,
          broadcast="declared in scenario case; see source declaration" if "broadcast" in token else None,
          dsl_supported=False,
          unsupported_reason="Scenario template expansion couples dtype, input/output pipeline, input mode and batch/broadcast operands; no DSL-equivalent case has been established",
          declaration=source[match.start() : match.start() + 180].split("\n", 1)[0].strip(),
        )
      )
  return result


def _registration_cases(filename: str, source: str) -> list[CatalogCase]:
  """Keep every concrete registration invocation and its full template args."""
  result: list[CatalogCase] = []
  names = (
    "register_rank2_and_batched_fusion",
    "register_rank2_fusion",
    "register_batched_fusion",
    "register_runtime_quant_pair",
    "register_runtime_quant_extent",
    "register_all_packing_modes",
    "register_all_batched_packed_b_modes",
    "register_all_batched_modes",
    "register_packed_dtype_mode",
    "register_required",
    "register_required_family_extent_pair",
    "register_extent_pair_mode",
    "register_policy",
    "register_conversion_pair",
  )
  for function in names:
    for pos, body in _balanced_calls(source, function):
      args = _template_args(source, pos, function)
      # Exclude declarations/definitions (these have no instantiated
      # args or are followed by a function body rather than a call).
      if not args:
        continue
      # Call-site expressions are joined with the explicit invocation
      # args, so the manifest retains atom, memory, pipeline, mode,
      # KStep and policy/family selectors even when names are aliases.
      declaration = f"{function}<{', '.join(args)}>({body.strip()})"
      pipeline_args = [x.rsplit("::", 1)[-1] for x in args if "InputPipeline::" in x or "OutputPipeline::" in x]
      mode_args = [x.rsplit("::", 1)[-1] for x in args if "InputMode::" in x]
      dtype_args = tuple(filter(None, (_normalize_dtype(x) for x in args)))
      quoted = re.findall(r'"([^"]+)"', body)
      result.append(
        CatalogCase(
          catalog="ConcreteRegistration",
          source=f"benchmarks/ops/{filename}",
          line=_line(source, pos),
          name=f"{function}/" + (quoted[-1] if quoted else f"line_{_line(source, pos)}"),
          dtype_a=dtype_args[0] if dtype_args else None,
          dtype_b=dtype_args[1] if len(dtype_args) > 1 else None,
          accumulator=dtype_args[2] if len(dtype_args) > 2 else None,
          input_modes=tuple(mode_args),
          input_pipeline_a=pipeline_args[0] if pipeline_args else None,
          input_pipeline_b=pipeline_args[1] if len(pipeline_args) > 1 else None,
          output_pipeline=next((x.rsplit("::", 1)[-1] for x in args if "OutputPipeline::" in x), None),
          configuration="; ".join(x for x in args if any(k in x for k in ("Policy", "Family", "KStep", "Atom")))
          or None,
          dsl_supported=False,
          unsupported_reason="Concrete C++ registration selector is preserved, but its shape catalog is supplied by a separate templated case catalog; pipeline/packing/batch tuple must be joined against that catalog before claiming DSL equivalence",
          declaration=declaration,
        )
      )
  return result


def _base_matmul_registrations(source: str, *, architecture: str = "arm") -> list[CatalogCase]:
  """Retain each ARM MatmulBench shard-to-template expansion site."""
  result = []
  functions = (
    "register_representative_extent_pairs",
    "register_workload_extent_pairs",
    "register_dispatch_extent_pairs",
    "register_packed_dot_extent_pairs",
    "register_dtype_probe_extent_pairs",
    "register_skinny_extent_pairs",
    "register_extended_integer_skinny_extent_pairs",
  )
  for function in functions:
    for pos, body in _balanced_calls(source, function):
      # The same NativeI8 aliases are used by the mutually-exclusive x86
      # registrations above the ARM branch.  Keep only calls below the ARM
      # catalog block, rather than treating an alias name as proof of ARM.
      arm_catalog = source.find("constexpr int CatalogShard")
      if architecture == "arm" and (arm_catalog < 0 or pos < arm_catalog):
        continue
      args = _template_args(source, pos, function)
      # The ARM branch uses SME aliases; discard the mutually exclusive
      # x86 registrations without inferring that they run on ARM.
      if not any("SME_" in arg or "AMX_" in arg or "NativeI8" in arg for arg in args):
        continue
      result.append(
        CatalogCase(
          catalog="MatmulBaseRegistration",
          source="benchmarks/ops/MatmulBench.cpp",
          line=_line(source, pos),
          name=f"{function}/line_{_line(source, pos)}",
          native_a=next((arg.rsplit("::", 1)[-1] for arg in args if "SME_" in arg or "AMX_" in arg or "NativeI8" in arg), None),
          input_modes=tuple(x.rsplit("::", 1)[-1].lower() for x in args if "InputMode::" in x),
          metadata=("Const", "Dynamic"),
          dsl_supported=False,
          unsupported_reason=(
            "ARM base registration selects shape helper, Atom, K_R and "
            "mode mask; its constexpr helper expansion is preserved "
            "here but not yet joined to individual executable DSL cases"
          ),
          declaration=f"{function}<{', '.join(args)}>({body.strip()})",
        )
      )
  return result


def _base_matmul_expanded(source: str, common: str, *, architecture: str = "arm") -> list[CatalogCase]:
  """Expand concrete ARM helper registrations through their shape templates."""
  atoms = {
    "SME_BF16F32": ("bf16", "bf16", "fp32", 2),
    "SME_F16F32": ("fp16", "fp16", "fp32", 2),
    "SME_F32F32": ("fp32", "fp32", "fp32", 1),
    "SME_F64F64": ("fp64", "fp64", "fp64", 1),
    "NativeI8S8S8": ("int8", "int8", "int32", 4),
    "NativeI8S8U8": ("int8", "uint8", "int32", 4),
    "NativeI8U8S8": ("uint8", "int8", "int32", 4),
    "NativeI8U8U8": ("uint8", "uint8", "int32", 4),
  }
  if architecture == "x86":
    atoms.update({"AMX_BF16F32": ("bf16", "bf16", "fp32", 32), "AMX_F16F32": ("fp16", "fp16", "fp32", 32)})
  packed_shapes = [
    (int(m), int(n), int(k), f"m{m}_n{n}_k{k}", ("packedab",))
    for m, n, k in re.findall(r"VECOPS_REGISTER_PACKED_DOT_PAIR\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,", common)
  ]
  result: list[CatalogCase] = []
  functions = (
    "register_representative_extent_pairs",
    "register_workload_extent_pairs",
    "register_dispatch_extent_pairs",
    "register_packed_dot_extent_pairs",
    "register_dtype_probe_extent_pairs",
    "register_skinny_extent_pairs",
    "register_extended_integer_skinny_extent_pairs",
  )
  for function in functions:
    for pos, body in _balanced_calls(source, function):
      arm_catalog = source.find("constexpr int CatalogShard")
      if architecture == "arm" and (arm_catalog < 0 or pos < arm_catalog):
        continue
      args = _template_args(source, pos, function)
      joined = " ".join(args)
      atom = next((name for name in atoms if name in joined), None)
      if atom is None:
        continue
      native_a, native_b, accumulator, atom_kstep = atoms[atom]
      tile = (
        _eval_int(args[1])
        if len(args) > 1
        and function
        in {
          "register_representative_extent_pairs",
          "register_workload_extent_pairs",
          "register_dispatch_extent_pairs",
          "register_dtype_probe_extent_pairs",
        }
        else None
      )
      kstep = (
        _eval_int(args[2])
        if len(args) > 2
        and function
        in {
          "register_representative_extent_pairs",
          "register_workload_extent_pairs",
          "register_dtype_probe_extent_pairs",
        }
        else atom_kstep
      )
      max_strip = _eval_int(args[3]) if function == "register_representative_extent_pairs" and len(args) > 3 else 6
      group_name = next(iter(re.findall(r'"([^"]+)"', body)), None)
      declarations: list[tuple[int, int, int, str, tuple[str, ...]]] = []
      if function == "register_representative_extent_pairs" and tile is not None and kstep is not None:
        declarations = [
          (tile, tile, 4 * kstep, "acc_1x1", ("packedab",)),
          (tile, 2 * tile, 4 * kstep, "acc_1x2", ("packedab",)),
          (tile, 3 * tile, 4 * kstep, "acc_1x3", ("packedab",)),
          (2 * tile, tile, 4 * kstep, "acc_2x1", ("packedab",)),
          (3 * tile, tile, 4 * kstep, "acc_3x1", ("packedab",)),
          (2 * tile, 2 * tile, 4 * kstep, "acc_2x2", ("packedab",)),
        ]
        if max_strip >= 4:
          declarations += [
            (tile, 4 * tile, 4 * kstep, "acc_1x4", ("packedab",)),
            (4 * tile, tile, 4 * kstep, "acc_4x1", ("packedab",)),
          ]
        if max_strip >= 5:
          declarations += [
            (tile, 5 * tile, 4 * kstep, "acc_1x5", ("packedab",)),
            (5 * tile, tile, 4 * kstep, "acc_5x1", ("packedab",)),
          ]
        if max_strip >= 6:
          declarations += [
            (tile, 6 * tile, 4 * kstep, "acc_1x6", ("packedab",)),
            (6 * tile, tile, 4 * kstep, "acc_6x1", ("packedab",)),
          ]
        declarations += [
          (2 * tile + 1, 2 * tile, 8 * kstep, "m_tail", ("raw", "packedab")),
          (2 * tile, 2 * tile + 1, 8 * kstep, "n_tail", ("raw", "packedab")),
          (2 * tile + 3, 3 * tile + 5, 8 * kstep, "mn_tail", ("raw", "packedab")),
          (2 * tile, 2 * tile, 8 * kstep + 1, "k_tail", ("raw", "packedab")),
          (2 * tile + 3, 3 * tile + 5, 8 * kstep + 1, "mnk_tail", ("raw", "packeda", "packedb", "packedab")),
          (1, 1152, 896, "decode_projection", ("raw", "packedb", "packedab")),
          (128, 1024, 768, "batch_projection", ("raw", "packedb", "packedab")),
          (256, 256, 256, "square", ("raw", "packeda", "packedb", "packedab")),
          (127, 1025, 769, "ragged_projection", ("raw", "packedb", "packedab")),
        ]
      elif function == "register_workload_extent_pairs" and tile is not None and kstep is not None:
        declarations = [
          (tile - 1, tile - 1, 4 * kstep - 1, "below_tile", ("raw", "packedab")),
          (1, 1, 8 * kstep + 1, "one_by_one", ("raw", "packedab")),
          (1, 8 * tile + 3, 16 * kstep + 1, "single_row_wide", ("raw", "packeda", "packedb", "packedab")),
          (8 * tile + 3, 1, 16 * kstep + 1, "single_col_tall", ("raw", "packeda", "packedb", "packedab")),
          (16 * tile, 16 * tile, 2 * kstep, "skinny_k", ("raw", "packedb", "packedab")),
          (2 * tile + 1, 2 * tile + 3, 256 * kstep + 1, "long_k", ("raw", "packedb", "packedab")),
          (1, 4096, 4096, "decode_qkv_4k", ("raw", "packedb", "packedab")),
          (4, 4096, 4096, "decode_qkv_batch4", ("raw", "packedb", "packedab")),
          (128, 4096, 4096, "prefill_qkv_4k", ("raw", "packedb", "packedab")),
          (1, 11008, 4096, "decode_mlp_up", ("raw", "packedb", "packedab")),
          (1, 4096, 11008, "decode_mlp_down", ("raw", "packedb", "packedab")),
          (128, 11008, 4096, "prefill_mlp_up", ("raw", "packedb", "packedab")),
          (128, 128, 128, "attention_score", ("raw", "packedb", "packedab")),
          (3136, 64, 576, "conv_1x1_lowering", ("raw", "packedb", "packedab")),
          (256, 256, 4096, "batched_square_k4k", ("raw", "packedb", "packedab")),
          (1024, 64, 4096, "rank_reduction", ("raw", "packedb", "packedab")),
          (64, 1024, 4096, "rank_expansion", ("raw", "packedb", "packedab")),
        ]
      elif function == "register_dispatch_extent_pairs" and tile is not None:
        declarations = [
          (1, 8, 64, "m1_n8_k64", ("raw", "packedab")),
          (8, 1, 64, "m8_n1_k64", ("raw", "packedab")),
          (2, 4, 65, "m2_n4_k65", ("raw", "packedab")),
          (4, 2, 65, "m4_n2_k65", ("raw", "packedab")),
          (4, 4, 256, "m4_n4_k256", ("raw", "packedab")),
          (8, 8, 257, "m8_n8_k257", ("raw", "packedab")),
          (16, 16, 1024, "m16_n16_k1024", ("raw", "packedab")),
        ]
        declarations += [
          (1, n, k, f"n{n}_k{k}", ("raw", "packedb", "packedab"))
          for n, k in ((16, 65), (32, 256), (64, 257), (128, 1024), (256, 1025), (512, 4096), (1024, 4097))
        ]
        declarations += [
          (m, 1, k, f"m{m}_k{k}", ("raw", "packeda", "packedb", "packedab"))
          for m, k in ((16, 65), (32, 256), (64, 257), (128, 1024), (256, 1025), (512, 4096), (1024, 4097))
        ]
        declarations += [
          (tile + 1, 2 * tile + 1, 1024, "short_m_long_k", ("raw", "packeda", "packedb", "packedab")),
          (2 * tile + 1, tile + 1, 1025, "short_n_long_k", ("raw", "packeda", "packedb", "packedab")),
        ]
      elif function == "register_packed_dot_extent_pairs":
        declarations = packed_shapes
      elif function == "register_dtype_probe_extent_pairs" and tile is not None and kstep is not None:
        declarations = [
          (tile + 3, tile + 5, 8 * kstep + 1, "tail_probe", ("raw", "packeda", "packedb", "packedab")),
          (1, 1024, 1024, "decode_probe", ("raw", "packeda", "packedb", "packedab")),
          (64, 256, 256, "batch_probe", ("raw", "packeda", "packedb", "packedab")),
        ]
      elif function == "register_skinny_extent_pairs":
        declarations = [
          (1, 8, 64, "m1_n8_k64", ("raw",)),
          (8, 1, 64, "m8_n1_k64", ("raw",)),
          (1, 16, 65, "m1_n16_k65", ("raw",)),
          (16, 1, 65, "m16_n1_k65", ("raw",)),
          (1, 64, 257, "m1_n64_k257", ("raw",)),
          (64, 1, 257, "m64_n1_k257", ("raw",)),
        ]
      elif function == "register_extended_integer_skinny_extent_pairs":
        declarations = [
          (1, 8, 64, "m1_n8_k64", ("raw",)),
          (8, 1, 64, "m8_n1_k64", ("raw",)),
          (1, 16, 65, "m1_n16_k65", ("raw",)),
          (16, 1, 65, "m16_n1_k65", ("raw",)),
          (1, 64, 257, "m1_n64_k257", ("raw",)),
          (64, 1, 257, "m64_n1_k257", ("raw",)),
          (1, 8, 4097, "m1_n8_k4097", ("raw",)),
          (8, 1, 4097, "m8_n1_k4097", ("raw",)),
        ]
      for m, n, k, name, modes in declarations:
        if group_name is not None:
          group = group_name
        elif function == "register_representative_extent_pairs":
          group = "microkernel" if name.startswith("acc_") else "tail" if name.endswith("tail") else "representative"
        elif function == "register_workload_extent_pairs":
          group = (
            "shape_coverage"
            if name in {"below_tile", "one_by_one", "single_row_wide", "single_col_tall", "skinny_k", "long_k"}
            else "transformer"
            if name
            in {
              "decode_qkv_4k",
              "decode_qkv_batch4",
              "prefill_qkv_4k",
              "decode_mlp_up",
              "decode_mlp_down",
              "prefill_mlp_up",
              "attention_score",
            }
            else "workload"
          )
        elif function == "register_dispatch_extent_pairs":
          group = (
            "dispatch_tiny"
            if name in {"m1_n8_k64", "m8_n1_k64", "m2_n4_k65", "m4_n2_k65", "m4_n4_k256", "m8_n8_k257", "m16_n16_k1024"}
            else "dispatch_gemv_row"
            if name.startswith("n")
            else "dispatch_gemv_col"
            if name.startswith("m")
            else "dispatch_tail"
          )
        elif function == "register_packed_dot_extent_pairs":
          group = "dispatch_packed_dot"
        elif function == "register_dtype_probe_extent_pairs":
          group = "dtype_coverage"
        elif function == "register_extended_integer_skinny_extent_pairs":
          group = "dispatch_integer_skinny"
        else:
          group = function
        result.append(
          CatalogCase(
            catalog="MatmulBaseExpanded",
            source="benchmarks/ops/MatmulBench.cpp",
            line=_line(source, pos),
            name=f"{group}/{name}",
            m=m,
            n=n,
            k=k,
            dtype_a=native_a,
            dtype_b=native_b,
            accumulator=accumulator,
            native_a=native_a,
            native_b=native_b,
            memory_a=native_a,
            memory_b=native_b,
            memory_c=accumulator,
            input_modes=modes,
            metadata=("Const", "Dynamic"),
            configuration=f"Atom={atom}; Tile={tile}; KStep={kstep}; helper={function}",
            arm_condition="ARM SME registration; SME_F64 requires HAS_SME_F64F64",
            dsl_supported=True,
            declaration=f"{function}<{', '.join(args)}>({body.strip()}) -> {name}",
          )
        )
  return result


def _arm_family_cases(source: str, *, architecture: str = "arm") -> list[CatalogCase]:
  """Expand explicit ARM family/policy-only shapes and retain their limits."""
  start = source.find("#else\n  using PackedDot")
  end = source.find("\n#endif\n}", start)
  if architecture == "x86":
    start = source.find("void register_family_benchmarks()")
    end = len(source)
  if start < 0 or end < 0:
    return []
  arm = source[start:end]
  result = []
  atom_types = {
    "SME_BF16F32": ("bf16", "bf16", "fp32", "bf16"),
    "SME_F16F32": ("fp16", "fp16", "fp32", "fp16"),
    "SME_F32F32": ("fp32", "fp32", "fp32", "fp32"),
    "SME_F64F64": ("fp64", "fp64", "fp64", "fp64"),
    "FamilyI8S8S8": ("int8", "int8", "int32", "int8"),
    "FamilyI8S8U8": ("int8", "uint8", "int32", "int8"),
    "FamilyI8U8S8": ("uint8", "int8", "int32", "uint8"),
    "FamilyI8U8U8": ("uint8", "uint8", "int32", "uint8"),
  }
  if architecture == "x86":
    atom_types["AMX_BF16F32"] = ("bf16", "bf16", "fp32", "bf16")
  family_aliases = {
    "Small": "kernel_family::SmallVector",
    "PackedDot": "kernel_family::PackedDot",
    "Residual": "kernel_family::ResidualSplit",
    "Automatic": "family_selection::Automatic",
    "RequireRuntimeQuant": "family_selection::Require<kernel_family::RuntimeQuantInt8>",
  }
  for function in ("register_required", "register_runtime_quant_pair", "register_extent_pair_mode"):
    for pos, body in _balanced_calls(arm, function):
      args = _template_args(arm, pos, function)
      if function == "register_extent_pair_mode":
        if len(args) < 5:
          continue
        shape_args = args[-3:]
        atom = next((key for key in atom_types if key in args[0]), None)
        mode_arg = args[1].rsplit("::", 1)[-1].lower()
        family_arg = "family_selection::Automatic"
      elif function == "register_required":
        if len(args) < 6:
          continue
        shape_args = args[-3:]
        atom = next((key for key in atom_types if key in args[0]), None)
        mode_arg = args[1].rsplit("::", 1)[-1].lower()
        selector = args[2]
        family_arg = family_aliases.get(selector, selector)
      else:
        if len(args) < 4:
          continue
        shape_args = args[-3:]
        atom = "FamilyI8U8S8"
        # register_runtime_quant_extent labels its reference input packed_b;
        # run_runtime_quant pre-packs B outside the timed matmul operation.
        mode_arg = "packedb"
        family_arg = family_aliases.get(args[0], args[0])
      runtime_quant = function == "register_runtime_quant_pair"
      m, n, k = (_eval_int(value) for value in shape_args)
      dtypes = atom_types.get(atom or "", (None, None, None, None))
      strings = re.findall(r'"([^"]+)"', body)
      case_name = strings[-1] if strings else f"line_{_line(arm, pos)}"
      result.append(
        CatalogCase(
          catalog="MatmulFamilyExpanded",
          source="benchmarks/ops/MatmulFamilyBench.cpp",
          line=_line(source, start + pos),
          name=case_name,
          m=m,
          n=n,
          k=k,
          dtype_a=dtypes[0],
          dtype_b=dtypes[1],
          accumulator=dtypes[2],
          native_a=dtypes[0],
          native_b=dtypes[1],
          memory_a="fp32" if runtime_quant else dtypes[0],
          memory_b=dtypes[1],
          memory_c="fp32" if runtime_quant else dtypes[2],
          memory_c_input=dtypes[2],
          input_modes=(mode_arg,),
          input_pipeline_a="RuntimePerRowAsymmetricQuantize" if runtime_quant else "Convert",
          input_pipeline_b="Identity" if runtime_quant else "Convert",
          output_pipeline="RuntimePerRowColumnDequantize" if runtime_quant else "Convert",
          metadata=("Const", "Dynamic"),
          configuration=family_arg,
          dsl_supported=m is not None and n is not None and k is not None,
          unsupported_reason=None
          if m is not None and n is not None and k is not None
          else "Template shape could not be resolved; family selector is reference-only while DSL executes automatic dispatch",
          declaration=f"{function}<{', '.join(args)}>({body.strip()})",
        )
      )
  return result


def _arm_policy_cases(source: str) -> list[CatalogCase]:
  """Expand policy helper shapes across the three concrete ARM policies."""
  policy_names = dict(re.findall(r'register_policy<([^>]+)>\s*\(\s*"([^"]+)"\s*\)', source))
  policies = list(policy_names.items())
  helper = re.search(r"void register_policy\(const char\* policy\)\s*\{(.*?)\n\}", source, re.DOTALL)
  if helper is None:
    return []
  body = helper.group(1)
  result = []
  for pos, call_body in _balanced_calls(body, "register_policy_extent_pair"):
    args = _template_args(body, pos, "register_policy_extent_pair")
    if len(args) != 6:
      continue
    _, mode, _policy, m_expr, n_expr, k_expr = args
    m, n, k = (_eval_int(value) for value in (m_expr, n_expr, k_expr))
    strings = re.findall(r'"([^"]+)"', call_body)
    name = "/".join(strings) if strings else f"line_{_line(body, pos)}"
    for policy_type, policy_name in policies:
      result.append(
        CatalogCase(
          catalog="MatmulPolicyExpanded",
          source="benchmarks/ops/MatmulPolicyBench.cpp",
          line=_line(source, helper.start(1) + pos),
          name=f"{policy_name}/{name}",
          m=m,
          n=n,
          k=k,
          dtype_a="bf16",
          dtype_b="bf16",
          accumulator="fp32",
          native_a="bf16",
          native_b="bf16",
          input_modes=(mode.rsplit("::", 1)[-1].lower(),),
          metadata=("Const", "Dynamic"),
          configuration=policy_type,
          dsl_supported=m is not None and n is not None and k is not None,
          unsupported_reason=None
          if m is not None and n is not None and k is not None
          else "Template shape could not be resolved; policy selector is reference-only while DSL executes automatic dispatch",
          declaration=f"register_policy_extent_pair<{', '.join(args)}>({call_body.strip()})",
        )
      )
  return result


def _packed_scenario_cases(source: str, *, architecture: str = "arm", complete_registration: bool = False) -> list[CatalogCase]:
  """Join ARM packed registrations with their concrete scenario shape tables."""
  atoms = {
    "SME_BF16F32": ("bf16", "bf16", "fp32", 2),
    "SME_F16F32": ("fp16", "fp16", "fp32", 2),
    "SME_F32F32": ("fp32", "fp32", "fp32", 1),
    "SME_F64F64": ("fp64", "fp64", "fp64", 1),
    "PackedI8S8S8": ("int8", "int8", "int32", 4),
    "PackedI8S8U8": ("int8", "uint8", "int32", 4),
    "PackedI8U8S8": ("uint8", "int8", "int32", 4),
    "PackedI8U8U8": ("uint8", "uint8", "int32", 4),
  }
  if architecture == "x86":
    atoms.update({"AMX_BF16F32": ("bf16", "bf16", "fp32", 32), "AMX_F16F32": ("fp16", "fp16", "fp32", 32)})
  result: list[CatalogCase] = []
  registrations = (
    ("register_all_packing_modes", "rank2"),
    ("register_all_batched_packed_b_modes", "batched"),
  )
  for function, rank in registrations:
    for pos, body in _balanced_calls(source, function):
      args = _template_args(source, pos, function)
      if not args:
        continue
      joined_args = " ".join(args)
      atom = next((name for name in atoms if name in joined_args), None)
      # Mutually exclusive AMX-only invocations are not ARM catalog entries.
      if atom is None and "SME_" not in joined_args and "PackedI8" not in joined_args:
        continue
      declaration = f"{function}<{', '.join(args)}>({body.strip()})"
      call_line = _line(source, pos)
      atom_types = atoms.get(atom or "")
      kstep_match = re.search(
        r"(?:scenario_cases|batched_scenario_cases|packed_dtype_probe_cases)\s*\(\s*(BF16KStep|I8KStep|1)\s*\)",
        body,
      )
      if len(args) < 4 or atom_types is None or kstep_match is None:
        result.append(
          CatalogCase(
            catalog="MatmulPackedUnresolved",
            source="benchmarks/ops/MatmulPackedBench.cpp",
            line=call_line,
            name=f"{function}/line_{call_line}",
            declaration=declaration,
            dsl_supported=False,
            unsupported_reason="ARM registration is visible, but Atom/memory/KStep/case-table selector could not be resolved; see exact declaration",
          )
        )
        continue
      native_a, native_b, accumulator, _atom_kstep = atom_types
      kstep_token = kstep_match.group(1)
      kstep = 1 if kstep_token == "1" else (32 if architecture == "x86" else 2) if kstep_token == "BF16KStep" else (64 if architecture == "x86" else 4)
      # The registration templates take Atom, MemoryA/B/C, input pipeline,
      # output pipeline, then optional B pipeline and C-input memory type.
      memory_a, memory_b, memory_c = (_normalize_dtype(arg) for arg in args[1:4])
      pipelines = [
        (index, arg.rsplit("::", 1)[-1])
        for index, arg in enumerate(args)
        if "InputPipeline::" in arg or "OutputPipeline::" in arg
      ]
      input_pipelines = [value for _, value in pipelines if "InputPipeline::" in args[_]]
      output_pipeline = next((value for _, value in pipelines if "OutputPipeline::" in args[_]), "Convert")
      input_pipeline_a = input_pipelines[0] if input_pipelines else "Convert"
      input_pipeline_b = input_pipelines[1] if len(input_pipelines) > 1 else input_pipeline_a
      enum_indexes = {index for index, _ in pipelines}
      c_input_candidates = [
        dtype for index, arg in enumerate(args[4:], 4) if index not in enum_indexes and (dtype := _normalize_dtype(arg))
      ]
      memory_c_input = c_input_candidates[0] if c_input_candidates else memory_c
      if rank == "rank2":
        probe = "packed_dtype_probe_cases" in body
        if probe:
          shapes = [
            ("tail", 19, 23, 8 * kstep + 1),
            ("gemv_tail", 1, 257, 4 * kstep + 1),
          ]
          # register_all_packing_modes registers all seven InputMode values;
          # probe shapes are still a genuine Cartesian registration set.
          modes = (
            "raw",
            "packeda",
            "packedb",
            "packedab",
            "onlinepackeda",
            "onlinepackedb",
            "onlinepackedab",
          )
        else:
          shapes = [
            ("tail", 19, 23, 8 * kstep + 1),
            ("decode", 1, 1024, 1024),
            ("batch", 64, 256, 256),
            ("wide", 32, 1024, 256),
          ]
          modes = ("raw", "packeda", "packedb", "packedab", "onlinepackeda", "onlinepackedb", "onlinepackedab")
      else:
        probe = False
        shapes = [
          ("shared_weight_tail", 8, 1, 47, 2 * kstep + 1, "B"),
          ("shared_weight_tiny", 8, 1, 32, 64, "B"),
          ("shared_weight_small", 8, 1, 64, 128, "B"),
          ("shared_weight_small_long_k", 8, 1, 64, 256, "B"),
          ("shared_weight_wide_short_k", 8, 1, 128, 64, "B"),
          ("shared_weight_decode", 8, 1, 256, 256, "B"),
          ("shared_weight_mlp_projection", 8, 1, 1024, 4096, "B"),
          ("shared_weight_qwen_o_proj", 8, 1, 3584, 3584, "B"),
          ("shared_weight_qwen_qkv", 8, 1, 4608, 3584, "B"),
        ]
        modes = ("raw", "packedb", "onlinepackedb")
      for shape in shapes:
        if rank == "rank2":
          name, m, n, k = shape
          batch, broadcast = None, None
        else:
          name, batch, m, n, k, broadcast = shape
        for mode in modes:
          result.append(
            CatalogCase(
              catalog="MatmulPackedExpanded",
              source="benchmarks/ops/MatmulPackedBench.cpp",
              line=call_line,
              name=f"{function}/{name}",
              m=m,
              n=n,
              k=k,
              dtype_a=native_a,
              dtype_b=native_b,
              accumulator=accumulator,
              native_a=native_a,
              native_b=native_b,
              memory_a=memory_a,
              memory_b=memory_b,
              memory_c=memory_c,
              memory_c_input=memory_c_input,
              input_modes=(mode,),
              input_pipeline_a=input_pipeline_a,
              input_pipeline_b=input_pipeline_b,
              output_pipeline=output_pipeline,
              metadata=("Dynamic",) if rank == "batched" and architecture == "arm" and not complete_registration else ("Const", "Dynamic"),
              batch=str(batch) if batch is not None else None,
              broadcast=broadcast,
              configuration=f"Atom={atom}; KStep={kstep_token}; source_case_table={'packed_dtype_probe_cases' if probe else 'batched_scenario_cases' if rank == 'batched' else 'scenario_cases'}",
              arm_condition="ARM branch; BF16KStep=2, I8KStep=4, explicit scenario(1)=1; SME_F64 branch requires HAS_SME_F64F64",
              # Rank-3 batched matmul is expressible through the runner's
              # rank-3 public DSL wrapper, including packed shared-B input.
              dsl_supported=True,
              declaration=declaration,
            )
          )
  return result


def _fusion_cases(source: str, *, architecture: str = "arm") -> list[CatalogCase]:
  """Expand ARM fusion registrations over their declared shape tables."""
  ranks = {
    "register_rank2_and_batched_fusion": ("fusion", "batched_fusion"),
    "register_rank2_fusion": ("fusion",),
    "register_batched_fusion": ("batched_fusion",),
  }
  atoms = {
    "FusionFloatAtom": ("bf16", "bf16", "fp32", 2),
    "::vecops::matmul::SME_BF16F32": ("bf16", "bf16", "fp32", 2),
    "::vecops::matmul::SME_F16F32": ("fp16", "fp16", "fp32", 2),
    "::vecops::matmul::SME_F32F32": ("fp32", "fp32", "fp32", 1),
    "::vecops::matmul::SME_F64F64": ("fp64", "fp64", "fp64", 1),
    "FusionI8S8S8": ("int8", "int8", "int32", 4),
    "FusionI8S8U8": ("int8", "uint8", "int32", 4),
    "FusionI8U8S8": ("uint8", "int8", "int32", 4),
    "FusionI8U8U8": ("uint8", "uint8", "int32", 4),
  }
  if architecture == "x86":
    atoms["FusionFloatAtom"] = ("bf16", "bf16", "fp32", 32)
    atoms["::vecops::matmul::AMX_F16F32"] = ("fp16", "fp16", "fp32", 32)
    for key in ("FusionI8S8S8", "FusionI8S8U8", "FusionI8U8S8", "FusionI8U8U8"):
      a, b, c, _ = atoms[key]
      atoms[key] = (a, b, c, 64)
  shape_tables = {
    "fusion": [
      ("tail", 19, 23, None),
      ("tiny_area", 2, 4, None),
      ("skinny_long_k_tail", 1, 8, 4097),
      ("skinny_long_k_tail_col", 8, 1, 4097),
      ("gemv_tail", 1, 257, None),
      ("small", 8, 128, 128),
      ("medium", 64, 256, 256),
      ("wide", 32, 1024, 256),
    ],
    "batched_fusion": [
      ("fusion_independent_tail", 19, 23, None, 4, "independent"),
      ("fusion_shared_small", 1, 128, 128, 8, "B"),
      ("fusion_shared_decode", 1, 256, 256, 8, "B"),
      ("fusion_shared_prefill", 16, 512, 512, 4, "B"),
    ],
  }
  result = []
  for function, tables in ranks.items():
    for pos, body in _balanced_calls(source, function):
      args = _template_args(source, pos, function)
      if len(args) < 6:
        continue
      atom = next((a for a in atoms if a in args[0]), None)
      if atom is None:
        continue
      # This file also contains AMX registrations in the mutually
      # exclusive x86 branch. Keep ARM SME/Fusion aliases only.
      native_a, native_b, acc, kstep = atoms[atom]
      ma, mb, mc = (_normalize_dtype(args[i]) for i in (1, 2, 3))
      pipa = next((a.rsplit("::", 1)[-1] for a in args if "InputPipeline::" in a), None)
      pipb = next(
        (
          a.rsplit("::", 1)[-1]
          for a in args[args.index(next(x for x in args if "InputPipeline::" in x)) + 1 :]
          if "InputPipeline::" in a
        ),
        pipa,
      )
      out = next((a.rsplit("::", 1)[-1] for a in args if "OutputPipeline::" in a), None)
      c_input = next((_normalize_dtype(a) for a in args[6:] if _normalize_dtype(a)), mc)
      for table in tables:
        for shape in shape_tables[table]:
          if table == "fusion":
            name, m, n, kval = shape
            k = (
              kval
              if kval is not None
              else (
                8 * kstep + 1
                if name == "tail"
                else 2 * kstep + 1
                if name == "tiny_area"
                else 4 * kstep + 1
                if name == "gemv_tail"
                else None
              )
            )
            batch, broadcast = None, None
          else:
            name, m, n, kval, batch, broadcast = shape
            k = kval if kval is not None else 8 * kstep + 1
          result.append(
            CatalogCase(
              catalog="MatmulFusionExpanded",
              source="benchmarks/ops/MatmulFusionBench.cpp",
              line=_line(source, pos),
              name=f"{function}/{name}",
              m=m,
              n=n,
              k=k,
              dtype_a=native_a,
              dtype_b=native_b,
              accumulator=acc,
              native_a=native_a,
              native_b=native_b,
              memory_a=ma,
              memory_b=mb,
              memory_c=mc,
              memory_c_input=c_input,
              input_modes=("raw",),
              input_pipeline_a=pipa,
              input_pipeline_b=pipb,
              output_pipeline=out,
              metadata=("Const", "Dynamic"),
              batch=str(batch) if batch is not None else None,
              broadcast=broadcast,
              arm_condition="ARM registration; SME_F64 requires HAS_SME_F64F64; SME BF16/F16 K_R=2, F32/F64 K_R=1, I8 K_R=4; compile-time vector widths come from current thread runtime state",
              dsl_supported=True,
              declaration=f"{function}<{', '.join(args)}>({body.strip()}) x {table}/{name}",
            )
          )
  return result


def _scenario_registrations(source: str, common: str) -> list[CatalogCase]:
  """Join the default scenario target to its rank-two/rank-three tables."""
  atoms = {
    "AMX_BF16F32": ("bf16", "bf16", "fp32"),
    "AMX_F16F32": ("fp16", "fp16", "fp32"),
    "SME_BF16F32": ("bf16", "bf16", "fp32"),
    "SME_F16F32": ("fp16", "fp16", "fp32"),
    "SME_F32F32": ("fp32", "fp32", "fp32"),
    "SME_F64F64": ("fp64", "fp64", "fp64"),
    "I8S8S8": ("int8", "int8", "int32"),
    "I8S8U8": ("int8", "uint8", "int32"),
    "I8U8S8": ("uint8", "int8", "int32"),
    "I8U8U8": ("uint8", "uint8", "int32"),
  }
  environment = {name: int(value) for name, value in re.findall(r"constexpr nint_t (\w+)\s*=\s*(\d+)", source)}
  variables = dict(re.findall(r"const auto (\w+)\s*=\s*([\w_]+\([^;]+\));", source))
  result = []
  for function in ("register_scenario", "register_batched_scenario"):
    for pos, body in _balanced_calls(source, function):
      args = _template_args(source, pos, function)
      if len(args) < 4:
        continue
      atom = next((key for key in atoms if key in args[0]), None)
      if atom is None:
        raise ValueError(f"unresolved scenario Atom at line {_line(source, pos)}: {args[0]}")
      selector = variables.get(body.strip(), body.strip())
      match = re.fullmatch(r"(batched_)?scenario_cases\(([^)]+)\)", selector)
      if match is None:
        raise ValueError(f"unresolved scenario table at line {_line(source, pos)}: {selector}")
      kstep = _eval_int(match[2], environment)
      if kstep is None:
        raise ValueError(f"unresolved scenario KStep: {match[2]}")
      batched = function == "register_batched_scenario"
      table = "make_batched_scenario_cases/" if batched else "make_scenario_cases/"
      native_a, native_b, accumulator = atoms[atom]
      tail = args[4:]
      pipelines = [arg.rsplit("::", 1)[-1] for arg in tail if "InputPipeline::" in arg]
      pipa = pipelines[0] if pipelines else "Convert"
      pipb = pipelines[1] if len(pipelines) > 1 else pipa
      out = next((arg.rsplit("::", 1)[-1] for arg in tail if "OutputPipeline::" in arg), "Convert")
      mc = _normalize_dtype(args[3])
      prior = next((_normalize_dtype(arg) for arg in tail if _normalize_dtype(arg)), mc)
      for shape in _scenario_shape_cases(common.replace("k_step", str(kstep))):
        if not shape.name.startswith(table):
          continue
        broadcast = shape.broadcast
        if batched and "false" in tail and broadcast and "A=shared" in broadcast:
          broadcast = broadcast.replace("A=shared", "A=independent")
        if broadcast:
          broadcast = "A" if "A=shared" in broadcast else "B" if "B=shared" in broadcast else "independent"
        result.append(
          dataclasses.replace(
            shape,
            catalog="MatmulScenarioExpanded",
            source="benchmarks/ops/MatmulScenarioBench.cpp",
            line=_line(source, pos),
            name=function + "/" + shape.name.removeprefix(table),
            dtype_a=native_a,
            dtype_b=native_b,
            accumulator=accumulator,
            native_a=native_a,
            native_b=native_b,
            memory_a=_normalize_dtype(args[1]),
            memory_b=_normalize_dtype(args[2]),
            memory_c=mc,
            memory_c_input=prior,
            input_modes=("raw",),
            metadata=("Const", "Dynamic"),
            broadcast=broadcast,
            input_pipeline_a=pipa,
            input_pipeline_b=pipb,
            output_pipeline=out,
            configuration=f"Atom={atom}; KStep={kstep}",
            dsl_supported=True,
            unsupported_reason=None,
            declaration=f"{function}<{', '.join(args)}>({body})",
          )
        )
  return result

def load_cases(
  root: Path | str = ROOT, *, svl_bytes: int | None = None, lane: int | None = None,
  architecture: str = "arm", features: Iterable[str] | None = None,
  complete_registration: bool = False,
) -> tuple[CatalogCase, ...]:
  root = Path(root)
  if architecture not in {"arm", "x86"}:
    raise ValueError("architecture must be 'arm' or 'x86'")
  if architecture == "x86":
    svl_bytes, lane = None, 16
  if svl_bytes is None and architecture == "arm":
    try:
      from vecops._arm_vector_state import get_vector_state

      svl_bytes = get_vector_state().sme_bytes
    except (ImportError, OSError):
      svl_bytes = None
  if svl_bytes is not None and (svl_bytes < 16 or svl_bytes > 256 or svl_bytes & (svl_bytes - 1)):
    raise ValueError("svl_bytes must be a power-of-two byte length in [16, 256]")
  if lane is None and svl_bytes is not None:
    lane = svl_bytes // 4
  sources = _read_sources(root)
  if architecture == "x86":
    selected_features = frozenset(features if features is not None else {"HAS_AMX_BF16", "HAS_AMX_INT8"})
    sources = {name: _architecture_source(source, selected_features | {"ARCH_X86_FAMILY"}) for name, source in sources.items()}
  cases: list[CatalogCase] = []
  for filename, source in sources.items():
    if filename == "MatmulPolicyBench.cpp" and architecture == "x86":
      cases.append(CatalogCase("ArchitectureExcluded", "benchmarks/CMakeLists.txt", 168,
        "MatmulPolicyBench", dsl_supported=False,
        unsupported_reason="CMake registers MatmulPolicyBench only for ARM NativeFixedStreamingSVE; its Atom is SME_BF16F32, so no x86 registration exists"))
      continue
    if filename == "MatmulBench.cpp":
      cases.extend(_base_matmul_registrations(source, architecture=architecture))
      cases.extend(_base_matmul_expanded(source, sources["MatmulBenchCommon.h"], architecture=architecture))
    if filename == "MatmulScenarioBench.cpp" and complete_registration:
      selected = source if architecture == "x86" else _architecture_source(source, frozenset(features or {"HAS_SME", "HAS_SME_F64F64"}))
      cases.extend(_scenario_registrations(selected, sources["MatmulScenarioBenchCommon.h"]))
    if filename == "MatmulFamilyBench.cpp":
      cases.extend(_arm_family_cases(source, architecture=architecture))
    if filename == "MatmulPolicyBench.cpp":
      cases.extend(_arm_policy_cases(source))
    if filename == "MatmulPackedBench.cpp":
      cases.extend(_packed_scenario_cases(source, architecture=architecture, complete_registration=complete_registration))
    if filename in {"MatmulBenchCommon.h", "MatmulFamilyBench.cpp", "MatmulPolicyBench.cpp", "MatmulPackedBench.cpp"}:
      cases.extend(_extent_cases(filename, source))
    if filename == "MatmulBenchCommon.h":
      cases.extend(_matmul_case_tables(source, lane=lane, f64_lane=svl_bytes // 8 if svl_bytes else None))
    if filename in {
      "MatmulFusionBench.cpp",
      "MatmulPackedBench.cpp",
      "MatmulFamilyBench.cpp",
      "MatmulPolicyBench.cpp",
      "MatmulOrientationBench.cpp",
    }:
      cases.extend(_registration_cases(filename, source))
    if filename == "MatmulFusionBench.cpp":
      cases.extend(_fusion_cases(source, architecture=architecture))
    if filename == "MatmulOrientationBench.cpp":
      cases.extend(_orientation_cases(source))
    if filename == "MatmulPackBenchCommon.h":
      cases.extend(_pack_cases(source))
    if filename == "MatmulScenarioBenchCommon.h":
      cases.extend(_vocab_cases(filename, source))
      cases.extend(_scenario_shape_cases(source))
  return tuple(dataclasses.replace(case, svl_bytes=svl_bytes, lane=lane, **({"arm_condition": "x86 source registration; selected ISA features: " + ",".join(sorted(selected_features))} if architecture == "x86" else {})) for case in cases)


def cases_by_catalog(cases: Iterable[CatalogCase] | None = None) -> dict[str, tuple[CatalogCase, ...]]:
  grouped: dict[str, list[CatalogCase]] = {}
  for case in cases if cases is not None else load_cases():
    grouped.setdefault(case.catalog, []).append(case)
  return {key: tuple(value) for key, value in grouped.items()}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  output = parser.add_mutually_exclusive_group()
  output.add_argument("--list", action="store_true", help="print concise case listing")
  output.add_argument("--json", action="store_true", help="print full JSON manifest")
  parser.add_argument(
    "--svl-bytes", type=int, help="streaming vector length in bytes; defaults to current calling thread on ARM Linux"
  )
  parser.add_argument("--architecture", choices=("arm", "x86"), default="arm")
  parser.add_argument("--feature", action="append", help="explicit target HAS_* feature; repeat for each enabled feature")
  parser.add_argument("--complete-registration", action="store_true", help="include ScenarioBench and complete batched packed metadata coverage")
  args = parser.parse_args()
  cases = load_cases(svl_bytes=args.svl_bytes, architecture=args.architecture, features=args.feature, complete_registration=args.complete_registration)
  if args.json:
    print(json.dumps([case.as_dict() for case in cases], indent=2))
  else:
    for case in cases:
      shape = "?" if None in (case.m, case.n, case.k) else f"{case.m}x{case.n}x{case.k}"
      status = "DSL" if case.dsl_supported else "UNSUPPORTED"
      detail = f" -- {case.unsupported_reason}" if case.unsupported_reason else ""
      print(f"{status:11} {case.catalog:24} {case.source}:{case.line} {case.name} [{shape}]{detail}")
    vector_state = cases[0] if cases else None
    if vector_state is not None:
      print(f"\nARM vector state: SVL={vector_state.svl_bytes!r} bytes, tile lane={vector_state.lane!r}")
    print(
      f"\nTotal declarations: {len(cases)}; DSL-equivalent: {sum(c.dsl_supported for c in cases)}; unsupported/metadata: {sum(not c.dsl_supported for c in cases)}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
