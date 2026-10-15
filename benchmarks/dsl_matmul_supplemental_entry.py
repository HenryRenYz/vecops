# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Select frozen runtime before loading a supplemental benchmark frontend."""

import argparse
import os
import runpy
import sys
from pathlib import Path

from dsl_snapshot_entry import select_snapshot


def main():
  arguments=sys.argv[1:]
  split=arguments.index('--') if '--' in arguments else len(arguments)
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--runner',choices=('pack','lifecycle'),required=True)
  args=parser.parse_args(arguments[:split])
  root=select_snapshot(args.root)
  # Match the main builder's definition cwd; the imported runtime remains frozen.
  os.chdir(root.parents[1])
  path=root/'benchmarks'/f'dsl_matmul_supplemental_{args.runner}.py'
  sys.argv=[str(path),*arguments[split+1:]]
  print(f'Frozen supplemental runtime: {root}',flush=True)
  runpy.run_path(str(path),run_name='__main__')


if __name__=='__main__':
  main()
