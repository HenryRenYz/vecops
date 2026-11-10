# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compile a captured execution graph with a selected resource runtime."""
import argparse
import json
from pathlib import Path
from vecops.graph import ExecutionGraph
from vecops.memory import Runtime
from . import plan_memory,audit_plan


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('graph',type=Path)
  parser.add_argument('--hardware')
  parser.add_argument('--policy')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--agent',action='append',default=[])
  parser.add_argument('--pool',action='append',default=[])
  arguments=parser.parse_args()
  graph=ExecutionGraph.from_dict(json.loads(arguments.graph.read_text()))
  runtime=(Runtime.open(arguments.hardware,policy_file=arguments.policy) if arguments.hardware
           else Runtime.discover())
  view=runtime.view()
  program=plan_memory(graph,runtime,view,runtime.snapshot(view),
      allowed_pools=arguments.pool,consumers=arguments.agent)
  arguments.output.write_text(program.dumps())
  print(json.dumps(audit_plan(program),indent=2))


if __name__=='__main__':
  main()
