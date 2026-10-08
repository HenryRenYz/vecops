# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Strict, interprocedural checks for compiler-emitted manual SME owners.

This module deliberately has no vecops/native-extension dependency. Equal
captured vector domains retain one no-call owner. Other domains require a
fixed scalar entry, exactly one resolved private call, and a same-SVL compute
body. These are code-generation checks, not target execution or ABI evidence.
"""

from __future__ import annotations

import re


def assembly_functions(assembly: str) -> dict[str, str]:
  """Read complete function ranges, including internal and cold labels."""
  result = {}
  for declaration in re.finditer(r"^\s*\.type\s+([^,\s]+),\s*@function\s*$", assembly, re.MULTILINE):
    name = declaration[1]
    end = re.search(rf"^\s*\.size\s+{re.escape(name)},", assembly[declaration.end() :], re.MULTILINE)
    assert end is not None, f"missing function boundary: {name}"
    assert name not in result, f"duplicate function: {name}"
    result[name] = assembly[declaration.end() : declaration.end() + end.start()]
  return result


def _instructions(body: str):
  instructions, labels = [], {}
  for line in body.splitlines():
    line = line.split("//", 1)[0].strip()
    if not line:
      continue
    label = re.fullmatch(r"([^\s:]+):", line)
    if label:
      assert label[1] not in labels, label[1]
      labels[label[1]] = len(instructions)
    elif not line.startswith("."):
      parts = line.split(None, 1)
      instructions.append((parts[0], parts[1].strip() if len(parts) == 2 else ""))
  assert instructions, "empty owner"
  return instructions, labels


def _check_cfg(name: str, body: str, *, kind: str, callee: str | None = None):
  instructions, labels = _instructions(body)
  assert sum(op == "smstart" for op, _ in instructions) == 1, (name, "entry count")
  # LLVM may duplicate the close into mutually exclusive return blocks.
  # Require one close per execution path, not one spelling in the function.
  assert any(op == "smstop" for op, _ in instructions), (name, "missing exit")
  for op, operands in instructions:
    assert op not in {"blr", "br"}, (name, "indirect call/branch", op, operands)
    if kind != "entry":
      assert op != "bl", (name, "outlined compute call", operands)
  # Single: starts with SM/ZA off. Entry: same, but changes SM only.
  # Body: called with SM on and ZA off; must preserve SM and close its own ZA.
  initial_sm = kind == "body"
  pending = [(0, initial_sm, False, 0, 0, 0)]
  visited, returns = set(), set()
  while pending:
    state = pending.pop()
    if state in visited:
      continue
    visited.add(state)
    index, sm, za, starts, calls, stops = state
    assert 0 <= index < len(instructions), (name, "control flow escapes function", state)
    op, operands = instructions[index]
    if op == "bl":
      assert kind == "entry" and sm and not za and calls == 0 and operands == callee, (
        name,
        "unexpected call",
        operands,
      )
      calls = 1
    elif op == "smstart":
      assert starts == 0, (name, "repeated state entry")
      if kind == "single":
        assert not sm and not za and operands == "", (name, op, operands)
        sm, za = True, True
      elif kind == "entry":
        assert not sm and not za and operands == "sm", (name, op, operands)
        sm = True
      else:
        assert sm and not za and operands == "za", (name, op, operands)
        za = True
      starts = 1
    elif op == "smstop":
      assert stops == 0, (name, "repeated state exit")
      if kind == "single":
        assert sm and za and operands == "", (name, op, operands)
        sm, za = False, False
      elif kind == "entry":
        assert sm and not za and calls == 1 and operands == "sm", (name, op, operands)
        sm = False
      else:
        assert sm and za and operands == "za", (name, op, operands)
        za = False
      stops = 1
    elif op == "ret":
      assert (sm, za, starts, calls, stops) == (initial_sm, False, 1, int(kind == "entry"), 1), (
        name,
        "unbalanced return",
        state,
      )
      returns.add(index)
      continue
    if op not in {"smstart", "smstop"} and (re.search(r"\bza(?:\d|\b)", operands) or op.endswith("mopa")):
      assert sm and za, (name, "ZA operation outside active interval", op, operands)
    successors = [index + 1]
    if op == "b" or op.startswith("b.") or op in {"cbz", "cbnz", "tbz", "tbnz"}:
      target = operands.rsplit(",", 1)[-1].strip()
      assert target in labels, (name, "out-of-function branch", op, operands)
      successors = [labels[target]] if op == "b" else [labels[target], index + 1]
    pending.extend((successor, sm, za, starts, calls, stops) for successor in successors)
  assert returns, f"no validated return: {name}"
  reachable = {state[0] for state in visited}
  assert all(index in reachable for index, (op, _) in enumerate(instructions) if op in {"smstart", "smstop"}), (
    name,
    "unreachable state boundary",
  )
  return instructions


def assert_sme_owners(assembly: str, expected: int, *, equal_domains: bool):
  """Validate every owner and return its actual compute-body assembly.

  ``equal_domains`` must come from fixed or invocation-checked captured build
  metadata, not from an assumption about a machine's usual vector length.
  """
  functions = assembly_functions(assembly)
  owners = {name: body for name, body in functions.items() if "with_streaming_za" in name}
  assert len(owners) == expected, tuple(owners)
  bodies, seen = {}, set()
  for name, outer in owners.items():
    if equal_domains:
      _check_cfg(name, outer, kind="single")
      bodies[name] = outer
      continue
    instructions, _ = _instructions(outer)
    calls = [operands for op, operands in instructions if op == "bl"]
    assert len(calls) == 1, (name, "entry must have exactly one private call", calls)
    callee = calls[0]
    assert "streaming_za_body" in callee and callee in functions and callee not in seen, (name, callee)
    seen.add(callee)
    for op, operands in instructions:
      assert op not in {"addvl", "addpl", "rdvl", "rdsvl", "addsvl", "addspl"}, (name, op)
      assert not re.search(r"\b(?:z|p)\d+(?:\b|\.)", operands), (name, "scalable outer state", op, operands)
      assert "mul vl" not in operands and "mul svl" not in operands, (name, operands)
    _check_cfg(name, outer, kind="entry", callee=callee)
    _check_cfg(callee, functions[callee], kind="body")
    bodies[name] = functions[callee]
  private = {name for name in functions if "streaming_za_body" in name}
  assert private == seen, ("unbound or unexpected private compute body", private - seen)
  return bodies
