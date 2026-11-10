#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Replan a trusted local programs/calls capture without tracing or allocating.

Use the migration sitecustomize overlay when testing a candidate native module.
The input is a local pickle produced by the model's planner-input capture; its
programs, calls and serialized fields are passed to the public graph compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('inputs',type=Path)
  parser.add_argument('--output',type=Path,required=True,help='report JSON; recipe uses the same stem')
  parser.add_argument('--hardware',type=Path)
  parser.add_argument('--policy',type=Path)
  options=parser.parse_args()
  if options.policy and not options.hardware:
    parser.error('--policy requires --hardware')
  from vecops.memory import Runtime
  from vecops.planner.compiler import GraphMemoryCompiler
  with options.inputs.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
  with options.inputs.open('rb') as stream:
    inputs=pickle.load(stream)
  if (not isinstance(inputs,dict) or not {'programs','calls','serialized'}<=set(inputs)
      or set(inputs)-{'programs','calls','serialized','scoped_requirements'}):
    raise ValueError('planner capture needs programs, calls, serialized and optional scoped_requirements')
  runtime=Runtime.open(options.hardware,policy_file=options.policy) if options.hardware else Runtime.discover()
  view=runtime.view()
  compiler=GraphMemoryCompiler(runtime,view,runtime.snapshot(view))
  started=time.perf_counter()
  program=compiler.compile_model(**inputs)
  elapsed=time.perf_counter()-started
  report={'inputs_sha256':digest,'planning_seconds':elapsed,
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'calls':len(inputs['calls']),'programs':len(inputs['programs']),
    'repeat_counts':[call.repeat_count for call in inputs['calls']],
    'requirements':len(program.requirements),
    'lifetime_intervals':sum(len(requirement.lifetimes) for requirement in program.requirements),
    'retained_requirements':sum(requirement.retained for requirement in program.requirements),
    'native_queries':dict(compiler.query_counts),
    'planned_capacity_groups':dict(program.peak_bytes_by_capacity_group),
    'planned_backing_bytes':sum(backing.bytes for backing in program.memory.backings),
    'planned_virtual_bytes':sum(region.bytes for region in program.memory.addresses.regions),
    'actually_committed_bytes':sum(group.committed_bytes for group in runtime.snapshot(view).groups),
    'runtime':runtime.contract()}
  options.output.parent.mkdir(parents=True,exist_ok=True)
  options.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  options.output.with_suffix('.toml').write_text(program.dumps())
  print(json.dumps({key:value for key,value in report.items() if key!='runtime'},indent=2))


if __name__=='__main__':
  main()
