#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Submit through installed lx-run with an explicit Flat-only Donau node filter.

Run on the login host after bashrc initialization. This process-local adapter
adds a verified Donau argument; it never modifies the installed shared helper.
The normal helper request directory records all recovery/job metadata.
"""
import argparse
import json
from pathlib import Path
import runpy


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('config', type=Path)
  args = parser.parse_args()
  config = json.loads(args.config.read_text())
  if not config.get('exclusive'):
    raise ValueError('CPU builds and workloads require an exclusive allocation')
  node_list = 'cn[11205-11328]'
  config['node_list'] = node_list
  helper = runpy.run_path(str(Path.home() / 'bin/lx-run'), run_name='vecops_lx_helper')
  scheduler = helper['scheduler']

  def select_flat(argv):
    if argv[0] == 'dsub':
      argv = [*argv, '--node-list', node_list]
    return scheduler(argv)

  helper['submit'].__globals__['scheduler'] = select_flat
  print(json.dumps(helper['submit'](config)))


if __name__ == '__main__':
  main()
