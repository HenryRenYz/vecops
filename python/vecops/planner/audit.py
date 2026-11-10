# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Offset-only audit records for graph memory placement and sharing."""
from dataclasses import asdict
from .validate import plan_summary


def audit_plan(program):
  return {'summary':plan_summary(program),
          'buffers':[asdict(binding) for _,binding in sorted(program.bindings.items())],
          'initializations':[asdict(item) for item in program.initializations],
          'calls':{key:dict(value) for key,value in program.call_bindings.items()},
          'memory_toml':program.dumps()}
