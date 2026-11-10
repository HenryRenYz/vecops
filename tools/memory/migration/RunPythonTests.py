#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Test source Python with an explicitly selected candidate extension.

This overrides editable-install routing only within this process, asserts the
loaded paths, and leaves the installed environment and active extension intact.
"""
import argparse
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--extension', required=True, type=Path)
  parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[3]
  package = root / 'python/vecops'
  extension = args.extension.resolve(strict=True)

  class Candidate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
      if fullname == 'vecops':
        return importlib.util.spec_from_file_location(fullname, package / '__init__.py',
          submodule_search_locations=[str(package)])
      if fullname == 'vecops._C':
        return importlib.util.spec_from_file_location(fullname, extension)
      if fullname.startswith('vecops.'):
        result = importlib.machinery.PathFinder.find_spec(fullname, path)
        if result is None:
          raise ModuleNotFoundError(f'{fullname} is absent from the candidate source')
        return result
      return None

  sys.meta_path.insert(0, Candidate())
  import vecops
  assert Path(vecops.__file__).resolve() == package / '__init__.py'
  assert Path(vecops._C.__file__).resolve() == extension
  print(f'candidate Python: {vecops.__file__}', flush=True)
  print(f'candidate native: {vecops._C.__file__}', flush=True)
  import pytest
  return pytest.main(args.pytest_args[1:] if args.pytest_args[:1] == ['--'] else args.pytest_args)


if __name__ == '__main__':
  raise SystemExit(main())
