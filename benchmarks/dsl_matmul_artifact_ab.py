# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Calibrated, balanced Matmul A/B on immutable, source-catalog artifacts.

Index in the corresponding source snapshot (no compilation), then compare
indexes in one process. Generated source hashes identify DSL semantics exactly;
C++ tensor/scalar contracts, named dimension relations and captures identify
the public reference; can_invoke alone cannot distinguish Const from Dynamic.
Missing artifacts remain explicit rather than silently becoming timing samples.
Offline packers are checked byte-for-byte, and every compute implementation
passes the catalog's independent oracle before timing. Each label calibrates
its native repeat to at least 25 ms. Twelve rounds balance all six orders;
raw clock readings and the exact source metadata/artifact paths are retained.
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path


def adapter_contract(text):
  """Extract the generated ABI, including bounds and dimension equalities.

  Unknown adapter formats fail closed. Expressions retain the ABI's dynamic
  constraints, rather than becoming constants from one observed invocation.
  """
  def integer(value):
    return int(value.strip().replace('INT64_MAX', str(2**63-1)).replace('INT64_MIN', str(-2**63)))

  arrays = {}
  for name, body in re.findall(r'static constexpr VecopsDimensionConstraint (\w+)\[\] = \{(.*?)\n\};', text, re.DOTALL):
    arrays[name] = tuple((kind, *(integer(v) for v in values)) for kind, *values in re.findall(
      r'\{sizeof\(VecopsDimensionConstraint\), (VECOPS_DIM_\w+), ([^,]+), ([^,]+), ([^,]+), ([^}]+)\}', body))
  parameters = []
  for kind, name, body in re.findall(
      r'\{sizeof\(VecopsParameterDescriptor\), VECOPS_VALUE_(TENSOR|SCALAR), "([^"]+)", \{\.(?:tensor|scalar)=\{sizeof\(Vecops\w+Parameter\), ([^}]+)\}\}\}', text):
    fields = [v.strip() for v in body.split(',')]
    mask = int(fields[0].removesuffix('ULL'))
    if kind == 'SCALAR':
      parameters.append((kind, name, mask))
    else:
      device, index, rank, optional = map(int, fields[1:5])
      access = int('VECOPS_TENSOR_READ' in fields[8]) | (int('VECOPS_TENSOR_WRITE' in fields[8]) << 1)
      parameters.append((kind, name, mask, device, index, rank, optional, access,
                         arrays.get(fields[6], ()), arrays.get(fields[7], ())))
  coordinate = r'call->values\[\d+\]\.value\.tensor\.(?:sizes|strides)\[\d+\]'
  relations = tuple(re.findall(r'if \((' + coordinate + r') != (' + coordinate + r')\)', text))
  if not parameters:
    raise ValueError('unrecognized artifact parameter descriptor')
  return tuple(parameters), relations


def template_contract(template):
  import vecops
  from vecops.language._compiler.backends.cpu.source import _build_definition

  definition, captures, _ = _build_definition(template.ir, ())
  parameters, relations, observed = [], [], {}
  for index, parameter in enumerate(definition._native().inputs):
    if isinstance(parameter, vecops._C.ValueDef):
      parameters.append(('SCALAR', parameter.name, 1 << (int(parameter.dtype)-1)))
      continue
    if parameter.optional:
      raise ValueError('artifact matching requires a bound optional tensor contract')
    dimensions = []
    for member, values in (('sizes', parameter.shape), ('strides', parameter.strides)):
      constraints = []
      for axis, value in enumerate(values):
        if value.kind == vecops._C.DimensionKind.fixed:
          constraints.append(('VECOPS_DIM_CONST', value.fixed_value, 0, 0, 1))
        elif value.kind in (vecops._C.DimensionKind.dynamic, vecops._C.DimensionKind.expression):
          bounds = (value.lower_bound, value.upper_bound, value.alignment)
          if any(v.is_symbol for v in bounds):
            raise ValueError('symbolic bounds need specialization before artifact matching')
          constraints.append(('VECOPS_DIM_DYNAMIC', 0, *(v.fixed_value for v in bounds)))
          if value.kind == vecops._C.DimensionKind.dynamic and value.symbol:
            coordinate = f'call->values[{index}].value.tensor.{member}[{axis}]'
            if value.symbol in observed:
              relations.append((coordinate, observed[value.symbol]))
            else:
              observed[value.symbol] = coordinate
        else:
          raise ValueError(f'unresolved artifact dimension: {value.kind}')
      dimensions.append(tuple(constraints))
    dtype = captures.get(parameter.dtype.symbol, parameter.dtype.dtype)
    access = {vecops._C.TensorAccess.input: 1, vecops._C.TensorAccess.output: 2,
              vecops._C.TensorAccess.inout: 3}[parameter.access]
    parameters.append(('TENSOR', parameter.name, 1 << (int(dtype)-1), parameter.device_type,
                       parameter.device_index, len(parameter.shape), 0, access, *dimensions))
  from vecops.language._compiler.dtypes import dtype_name
  specialization = {name: dtype_name(value) + '_t'
                    if isinstance(value, vecops._C.DType) else value for name, value in captures.items()}
  return (tuple(parameters), tuple(relations)), specialization


def arguments(values, repeat):
  import vecops

  return vecops._C.ArgumentMetadata([
    vecops._C.TensorView.from_numpy(v, writable=i in (2, 11, 12)) for i, v in enumerate(values)
  ] + [repeat])


def source_scenarios(document):
  from dsl_matmul_catalog import CatalogCase
  from dsl_matmul_source_catalog import SourceScenario

  fields = {field.name for field in dataclasses.fields(CatalogCase)}
  chosen = set(document['selected_execution_ids'])
  return tuple(SourceScenario(CatalogCase(**{k: v for k, v in row.items() if k in fields}),
                              row['meta'], row['level'], 'auto', row['packing'])
               for row in document['executions'] if row['execution_id'] in chosen)


def index_artifacts(args):
  import vecops
  import vecops.language as T
  from dsl_matmul_source_catalog import FAMILIES, INPUTS, OUTPUTS, POLICIES, source_arguments, source_templates

  manifest = json.loads(args.manifest.read_text())
  inventory = []
  adapters = {adapter for directory in args.build_dir
              for adapter in directory.glob('build/kernel-batch-*/tasks/*/input/vecops_adapter.cpp')}
  for adapter in sorted(adapters):
    text = adapter.read_text()
    source = re.search(r'#include "([^"\n]+\.cpp)"', text)
    if not source:
      continue
    batch, task = adapter.parents[3], adapter.parents[1]
    for binary in (batch / 'artifacts' / task.name).glob('*.so'):
      spec = adapter.with_name('vecops_spec.h').read_text()
      selectors = {name: int(value) for name, value in re.findall(
        r'inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;', spec)}
      aliases = dict(re.findall(r'using (\w+) = ::vecops::(\w+);', spec))
      inventory.append((Path(source.group(1)), {**selectors, **aliases}, adapter_contract(text),
                        binary.resolve(), vecops._C.Executable.load(str(binary)), adapter.resolve()))
  output, cache, audit = {}, {}, {}
  for scenario in source_scenarios(manifest):
    key = scenario.compilation_key
    if key not in cache:
      pairs = source_templates(scenario)
      paths = {}
      values, _ = source_arguments(scenario)
      probe = arguments(values, 1)
      for packing, pair in enumerate(pairs):
        if pair is None:
          continue
        for role, template in zip(('dsl', 'cpp'), pair):
          if role == 'dsl':
            recorded = manifest.get('artifact_sources', {}).get(scenario.execution_id, {}).get(('pack_' if packing else '') + role)
            if recorded:
              digest = hashlib.sha256(Path(recorded).read_bytes()).hexdigest()
              if Path(recorded).parent.name != digest:
                raise ValueError('recorded DSL source changed after compilation')
            else:
              source = T.lower(template, target=args.target, pass_options=manifest['pass_options']).artifact.source
              digest = hashlib.sha256(source.encode()).hexdigest()
            matches = [(path, exe, adapter) for file, _, _, path, exe, adapter in inventory
                       if file.parent.name == digest and exe.can_invoke(probe)]
          else:
            contract, captures = template_contract(template)
            # The source-catalog compiler always uses execution_parallelism=1.
            wanted = {'Parallelism': 1, 'PackA': int(scenario.pack_a), 'PackB': int(scenario.pack_b),
                          'OnlinePack': int(scenario.online), 'OnlyPack': packing,
                          'HasPrior': int(scenario.has_prior), 'BroadcastPrior': int(scenario.broadcast_prior),
                          'SharedB': int(scenario.shared_b), 'SwapAB': int(scenario.swap),
                          'InputPipelineA': INPUTS.index(scenario.case.input_pipeline_a),
                          'InputPipelineB': INPUTS.index(scenario.case.input_pipeline_b),
                          'OutputPipeline': OUTPUTS.index(scenario.case.output_pipeline),
                          'ReferenceFamily': FAMILIES.index(scenario.reference[0]),
                          'ReferencePolicy': POLICIES.index(scenario.reference[1])}
            matches = [(path, exe, adapter) for file, selectors, actual_contract, path, exe, adapter in inventory
                       if file.name == 'dsl_matmul_source_reference.cpp'
                       and actual_contract == contract
                       and all(selectors.get(k) == v for k, v in {**wanted, **captures}.items()) and exe.can_invoke(probe)]
          label = ('pack_' if packing else '') + role
          chosen = max(matches, key=lambda p: p[0].stat().st_mtime) if matches else None
          paths[label] = str(chosen[0]) if chosen else None
          audit.setdefault(scenario.execution_id, {})[label] = {
            'matching_artifacts': [str(p[0]) for p in matches], 'adapter': str(chosen[2]) if chosen else None,
            'expected_contract': contract if role == 'cpp' else None,
            'captures': captures if role == 'cpp' else None}
          if chosen:
            adapter = chosen[2]
            commands = adapter.parents[3] / 'build/compile_commands.json'
            command = next((entry for entry in json.loads(commands.read_text())
                            if Path(entry['file']).resolve() == adapter), None) if commands.exists() else None
            audit[scenario.execution_id][label].update(
              compile_command=command,
              specialization_sha256=hashlib.sha256(adapter.with_name('vecops_spec.h').read_bytes()).hexdigest())
      cache[key] = paths, audit[scenario.execution_id]
    output[scenario.execution_id], audit[scenario.execution_id] = cache[key]
  args.index_output.write_text(json.dumps({'manifest': str(args.manifest.resolve()), 'snapshot': str(args.snapshot),
                                          'artifacts': output, 'schema_audit': audit}, indent=2) + '\n')
  print(f'Indexed {len(output)} rows; {sum(any(p is None for p in row.values()) for row in output.values())} with missing artifacts')


def compare(args):
  import numpy as np
  import vecops
  from dsl_matmul_source_catalog import result_metadata, source_arguments, validate_kernel_pair
  from dsl_timing import reserve_cpu

  reference_only = args.baseline_index is None
  indexes = [json.loads(path.read_text()) for path in (args.baseline_index or args.candidate_index, args.candidate_index)]
  documents = [json.loads(Path(index['manifest']).read_text()) for index in indexes]
  if documents[0]['source_hashes'] != documents[1]['source_hashes']:
    raise ValueError('C++ catalog source changed between snapshots')
  old_ids, new_ids = (set(index['artifacts']) for index in indexes)
  if old_ids != new_ids and not (args.allow_baseline_subset and old_ids <= new_ids):
    raise ValueError('A/B indexes must cover the same exact execution IDs; explicitly allow a baseline subset if needed')
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows, missing, candidate_only = [], [], []
  executables = {}
  def executable(path):
    if path not in executables:
      executables[path] = vecops._C.Executable.load(path)
    return executables[path]
  orders = tuple(itertools.permutations(('candidate', 'cpp') if reference_only else ('baseline', 'candidate', 'cpp')))
  with reserve_cpu(args.cpu), (args.output_dir / 'raw.jsonl').open('w') as stream:
    os.sched_setaffinity(0, {args.cpu})
    for scenario in source_scenarios(documents[1]):
      if scenario.execution_id not in old_ids:
        candidate_only.append(result_metadata(scenario))
        continue
      old, new = (index['artifacts'][scenario.execution_id] for index in indexes)
      if any(path is None for row in (old, new) for path in row.values()):
        missing.append(dict(result_metadata(scenario), baseline=old, candidate=new))
        continue
      values, expected = source_arguments(scenario)
      # The same offline packed ABI must satisfy both generations and C++.
      if 'pack_cpp' in new:
        reference = None
        for path in (old['pack_dsl'], new['pack_dsl'], new['pack_cpp']):
          executable(path).invoke(arguments(values, 1))
          packed = tuple(values[i].copy() for i in (11, 12))
          if reference is None:
            reference = packed
          for active, actual, gold in zip((scenario.pack_a, scenario.pack_b), packed, reference):
            if active:
              np.testing.assert_array_equal(actual, gold)
      paths = {'baseline': old['dsl'], 'candidate': new['dsl'], 'cpp': new['cpp']}
      if reference_only:
        paths.pop('baseline')
      probe = arguments(values, 1)
      calls = {label: lambda exe=executable(path): exe.invoke(probe) for label, path in paths.items()}
      tolerance = .02 if values[2].dtype.itemsize <= 2 else 2e-10 if values[2].dtype.itemsize == 8 else 8e-4
      for label in paths.keys() - {'cpp'}:
        validate_kernel_pair(calls[label], calls['cpp'], values[2], expected, tolerance=tolerance)
      repeats = {}
      for label, path in paths.items():
        repeat = 1
        while True:
          call_arguments = arguments(values, repeat)
          begin = time.perf_counter_ns()
          executable(path).invoke(call_arguments)
          elapsed = time.perf_counter_ns() - begin
          if elapsed >= 25_000_000 or repeat >= 1_048_576:
            break
          repeat *= 2
        repeats[label] = repeat
        calls[label] = lambda exe=executable(path), call_arguments=call_arguments: exe.invoke(call_arguments)
      samples = {label: [] for label in paths}
      for round_ in range(args.rounds):
        for position, label in enumerate(orders[round_ % len(orders)]):
          begin = time.perf_counter_ns()
          calls[label]()
          end = time.perf_counter_ns()
          ns = (end - begin) / repeats[label]
          samples[label].append(ns)
          stream.write(json.dumps({'execution_id': scenario.execution_id, 'round': round_, 'position': position,
                                       'label': label, 'repeat': repeats[label], 'begin_ns': begin, 'end_ns': end, 'ns': ns}) + '\n')
      stream.flush()
      medians = {label: statistics.median(samples[label]) for label in paths}
      row = dict(result_metadata(scenario), median_ns=medians, speedup=None if reference_only else medians['baseline']/medians['candidate'],
                 candidate_over_cpp=medians['candidate']/medians['cpp'], artifacts=paths)
      rows.append(row)
      print(scenario.execution_id, row['speedup'], row['candidate_over_cpp'], flush=True)
  (args.output_dir / 'results.json').write_text(json.dumps({'cpu': args.cpu, 'rounds': args.rounds,
    'indexes': [str(args.baseline_index), str(args.candidate_index)], 'reference_only': reference_only,
    'rows': rows, 'missing_artifacts': missing, 'candidate_only': candidate_only}, indent=2)+'\n')
  if missing:
    print(f'{len(missing)} rows have missing artifacts; see the explicit ledger')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--snapshot', type=Path, required=True)
  parser.add_argument('--target', default='Native')
  parser.add_argument('--manifest', type=Path)
  parser.add_argument('--build-dir', type=Path, action='append', help='artifact build root; repeat for multiple cohorts')
  parser.add_argument('--index-output', type=Path)
  parser.add_argument('--baseline-index', type=Path)
  parser.add_argument('--candidate-index', type=Path)
  parser.add_argument('--allow-baseline-subset', action='store_true', help='retain candidate-only rows in the unpaired ledger')
  parser.add_argument('--output-dir', type=Path)
  parser.add_argument('--cpu', type=int)
  parser.add_argument('--rounds', type=int, default=12)
  args = parser.parse_args()
  sys.path.insert(0, str(args.snapshot.resolve() / 'benchmarks'))
  from dsl_snapshot_entry import select_snapshot
  select_snapshot(args.snapshot.resolve())
  if args.index_output:
    if not args.manifest or not args.build_dir:
      parser.error('indexing requires --manifest and --build-dir')
    index_artifacts(args)
  else:
    if not args.candidate_index or not args.output_dir or args.cpu is None:
      parser.error('timing requires --candidate-index, --output-dir and --cpu; add --baseline-index for A/B')
    if args.rounds <= 0 or args.rounds % 6:
      parser.error('rounds must be positive and divisible by six')
    compare(args)


if __name__ == '__main__':
  main()
