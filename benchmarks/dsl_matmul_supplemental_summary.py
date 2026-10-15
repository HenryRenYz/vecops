# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Strict supplemental coverage and performance summary from retained evidence.

Built, oracle-validated and timed are distinct milestones. Timeouts are retained
as scheduling failures, never capability exclusions. Prepared plans are the
Stateful public-API gap; optional MMLA experiments remain a separate category.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from dsl_matmul_supplemental_catalog import manifest
from dsl_matmul_supplemental_timing import verify_timing_chains

FIELDS=('architecture','suite','category','name','shape','batch','calls_per_pack','broadcast','strategy',
        'metadata','atom','native_a','native_b','memory_a','memory_b','memory_c','memory_c_input',
        'input_pipeline_a','input_pipeline_b','output_pipeline','spatial','k','side','memory','native',
        'pipeline','selector','compensation_mode','zero_point','engine')


def key(row):
  return json.dumps({k:row[k] for k in FIELDS if k in row},sort_keys=True)


def metrics(rows):
  ratios=[r['median_ns']['dsl']/r['median_ns']['cpp'] for r in rows]
  if not ratios:return {'timed_rows':0}
  return {'timed_rows':len(rows),'dsl_over_cpp_geomean':math.exp(statistics.mean(map(math.log,ratios))),
    'dsl_over_cpp_median':statistics.median(ratios),'dsl_slower_than_2x_count':sum(r>2 for r in ratios),
    'dsl_ns_geomean':math.exp(statistics.mean(math.log(r['median_ns']['dsl']) for r in rows)),
    'cpp_ns_geomean':math.exp(statistics.mean(math.log(r['median_ns']['cpp']) for r in rows)),
    'dsl_ns_median':statistics.median(r['median_ns']['dsl'] for r in rows),
    'cpp_ns_median':statistics.median(r['median_ns']['cpp'] for r in rows)}


def admitted_checkpoint(path, expected):
  """Check the retained, post-guarded source provenance before using results."""
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
    raise ValueError(f'attestation checkpoint changed: {path}')
  document=json.loads(path.read_text())
  if document.get('invalid_reason') or not document.get('policies'):
    raise ValueError(f'attestation checkpoint is invalid: {path}')
  if not any('post_build' in event.get('phase','') for event in document['events']):
    raise ValueError(f'attestation checkpoint has no post-build admission: {path}')
  for record in document['sources']:
    for name,digest in record['inputs'][0]:
      source=Path(name)
      if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise ValueError(f'admitted source changed: {source}')
  return document


def summarize(architecture,features,directories):
  coverage=manifest(architecture,features)
  states={key(row):{'registration':row,'built':False,'validated':False,'timed':False,'failures':[]}
          for row in coverage['registrations']}
  ids={row['registration_id']:key(row) for row in coverage['registrations']}
  results={}
  for directory in directories:
    for path in directory.rglob('coverage.json'):
      try:
        for row in json.loads(path.read_text()).get('registrations',[]):
          ids[row['registration_id']]=key(row)
      except (OSError,ValueError,KeyError):continue
  for directory in directories:
    for path in directory.rglob('manifest.json'):
      try:doc=json.loads(path.read_text())
      except (OSError,ValueError):continue
      artifacts=doc.get('artifacts',[])
      mapping={a['registration_id']:key(a['row']) for a in artifacts if 'row' in a}
      parent=path.parent
      admission=parent/'build-status.json'
      built=[]
      if admission.exists():
        record=json.loads(admission.read_text())
        if record.get('admitted'):
          admitted_checkpoint(Path(record['attestation_checkpoint']),record['attestation_checkpoint_sha256'])
          built=record.get('registrations',[])
      for id_ in built:
        identity=mapping.get(id_,ids.get(id_))
        if identity in states:states[identity]['built']=True
      for failure in doc.get('failures',[]):
        identity=ids.get(failure['registration_id'])
        if identity in states:states[identity]['failures'].append({'evidence':str(path),**failure})
    result_paths=set(directory.rglob('validation-results.json'))
    result_paths.update(directory.rglob('timing-results.json'))
    result_paths=sorted(result_paths)
    for path in result_paths:
      try:doc=json.loads(path.read_text())
      except (OSError,ValueError):continue
      if doc.get('timing_chains'):
        verify_timing_chains(doc)
      for chain in doc.get('timing_chains',[]):
        for name,expected in [(chain['pre'],chain['pre_sha256']),(chain['pending'],chain['pending_sha256']),*chain['inputs'].items()]:
          if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'timing proof chain changed: {name}')
      admitted=set()
      for record in doc.get('batch_admissions',[]):
        admitted_checkpoint(Path(record['checkpoint']),record['checkpoint_sha256'])
        manifest_path=Path(record['directory'])/'manifest.json'
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=record['manifest_sha256']:
          raise ValueError(f'validated batch manifest changed: {manifest_path}')
        admitted.update(record['registrations'])
      for row in doc.get('rows',[]):
        identity=key(row['registration'])
        if identity not in states:continue
        if row['registration']['registration_id'] not in admitted or not states[identity]['built']:
          raise ValueError(f'result lacks an admitted build/validation batch: {path}')
        if row.get('error'):
          states[identity]['failures'].append({'evidence':str(path),'phase':row.get('phase'),'error':row['error']})
          continue
        if set(row.get('artifacts',{}))!={'dsl','cpp'}:
          raise ValueError(f'result lacks exact DSL/CPP artifact provenance: {path}')
        for artifact in row['artifacts'].values():
          if hashlib.sha256(Path(artifact['binary']).read_bytes()).hexdigest()!=artifact['binary_sha256']:
            raise ValueError(f'validated binary changed: {artifact["binary"]}')
        if row.get('validated'):states[identity]['validated']=True
        if row.get('timed',bool(row.get('median_ns'))) and row.get('median_ns'):
          states[identity]['timed']=True;results[identity]=row
  categories={}
  for suite in ('MatmulPack','MatmulWeightReuse','StatefulMatmul','MatmulMMLAExperiment'):
    rows=[v for v in states.values() if v['registration']['suite']==suite]
    if not rows:continue
    timing_failed=lambda row:any(failure.get('phase')=='measurement_oracle_or_timing' for failure in row['failures']) and not row['timed']
    categories[suite]={'registered':len(rows),'comparable':sum(r['registration']['dsl_equivalent'] for r in rows),
      'public_api_unsupported':sum(r['registration'].get('comparison_status')=='unsupported_public_api' for r in rows),
      'built':sum(r['built'] for r in rows),'validated':sum(r['validated'] for r in rows),'timed':sum(r['timed'] for r in rows),
      'failed_unrecovered':sum(bool(r['failures']) and (not r['validated'] or timing_failed(r)) for r in rows),
      'timing_failed_unrecovered':sum(timing_failed(r) for r in rows),
      **metrics([results[k] for k,v in states.items() if v['registration']['suite']==suite and k in results])}
  reuse={}
  for calls in (1,2,4,8,16):
    selected=[r for r in results.values() if r['registration']['suite']=='MatmulWeightReuse' and r['registration']['calls_per_pack']==calls]
    reuse[str(calls)]=metrics(selected)
  return {'architecture':architecture,'categories':categories,'weight_reuse_calls_per_pack':reuse,
    'states':list(states.values()),'notes':['Coverage totals come from actual source registrations, not result-directory counts.',
      'Failed scheduling attempts remain evidence after recovery; unrecovered failures are distinct from unsupported APIs.',
      'Packing, weight-reuse lifecycle and Stateful Legacy performance are not pooled.',
      'Each Prepared registration is a persistent prepared-plan API gap, not an unsupported matrix datatype.']}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--features',nargs='*',default=[])
  parser.add_argument('--evidence',type=Path,nargs='+',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();result=summarize(args.architecture,args.features,args.evidence)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result['categories'],indent=2))


if __name__=='__main__':
  main()
