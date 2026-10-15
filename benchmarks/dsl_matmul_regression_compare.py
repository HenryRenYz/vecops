# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compare an exact admitted regression selection; every suspect needs review.

main inputs are suite directories containing plan.json and ledger.json; --stage
chooses one exact stage. supplemental inputs are formal timing-results.json.
All retained supplemental provenance paths must remain readable. --confirmation
is an independent candidate rerun against the same baseline and exact selection.
Exit 1 means suspect regression/review, exit 2 means invalid evidence. Twelve
rounds alone do not establish confidence intervals or absence of host drift.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from dsl_benchmark_selection import selection_identity


def read(path):
  return json.loads(Path(path).read_text())


def indexed(rows, key):
  result={}
  for row in rows:
    identity=key(row)
    if identity in result:raise ValueError(f'duplicate configuration: {identity}')
    result[identity]=row
  return result


def positive(value):
  if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
    raise ValueError('timing must be finite and positive')
  return value


def load_main(path, identity, stage):
  path=Path(path);plan=read(path/'plan.json');ledger=read(path/'ledger.json')
  arch=identity['architecture']
  if plan.get('architecture')!=arch or ledger.get('architecture')!=arch:raise ValueError('architecture mismatch')
  metadata=indexed(plan['executions'],lambda r:r['execution_id'])
  records=indexed(ledger['rows'],lambda r:r['execution_id']);result={}
  selected=plan.get('selected_execution_ids',list(metadata))
  if not isinstance(selected,list) or len(selected)!=len(set(selected)) or set(selected)!=set(metadata) or set(records)!=set(metadata):
    raise ValueError('main plan/executions/ledger scope is not exact and unique')
  for id_ in identity['execution_ids']:
    if id_ not in metadata or id_ not in records or id_ not in plan.get('selected_execution_ids',metadata):raise ValueError(f'unknown/missing execution: {id_}')
    row=records[id_];timing=row.get('stages',{}).get(stage,{})
    if row.get('build')!='built' or timing.get('status')!='passed':raise ValueError(f'unpassed configuration: {id_}/{stage}')
    values=timing.get('median_ns',{})
    result[id_]={'metadata':metadata[id_],'dsl':positive(values.get('candidate')),'cpp':positive(values.get('cpp'))}
  return result


def load_supplemental(path, identity):
  from dsl_matmul_supplemental_timing import verify_timing_chains
  from dsl_matmul_supplemental_summary import admitted_checkpoint
  doc=read(path)
  if not doc.get('completed') or not doc.get('timing_chains') or not doc.get('batch_admissions'):
    raise ValueError('supplemental results lack completed formal admissions/chains')
  verify_timing_chains(doc)
  admitted=set()
  for batch in doc['batch_admissions']:
    admitted_checkpoint(Path(batch['checkpoint']),batch['checkpoint_sha256'])
    manifest=Path(batch['directory'])/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=batch['manifest_sha256']:raise ValueError('batch manifest changed')
    if admitted.intersection(batch['registrations']):raise ValueError('duplicate admitted registrations')
    admitted.update(batch['registrations'])
  records=indexed(doc['rows'],lambda r:r['registration']['registration_id']);result={}
  for id_ in identity['registration_ids']:
    if id_ not in records or id_ not in admitted:raise ValueError(f'unknown/missing admitted registration: {id_}')
    row=records[id_];metadata=row['registration']
    if metadata.get('architecture')!=identity['architecture']:raise ValueError('architecture mismatch')
    if row.get('error') or not row.get('validated') or not row.get('timed'):raise ValueError(f'unpassed configuration: {id_}')
    if set(row.get('artifacts',{}))!={'dsl','cpp'}:raise ValueError('missing DSL/CPP artifact provenance')
    for artifact in row['artifacts'].values():
      if hashlib.sha256(Path(artifact['binary']).read_bytes()).hexdigest()!=artifact['binary_sha256']:raise ValueError('admitted binary changed')
    values=row.get('median_ns',{})
    result[id_]={'metadata':metadata,'dsl':positive(values.get('dsl')),'cpp':positive(values.get('cpp'))}
  return result


def compare(baseline, candidate, *, confirmation=None, relative=.10, absolute_ns=50):
  if not baseline or baseline.keys()!=candidate.keys() or (confirmation is not None and baseline.keys()!=confirmation.keys()):
    raise ValueError('comparison requires the same nonempty exact configuration scope')
  rows=[]
  for id_,old in baseline.items():
    new=candidate[id_]
    if old['metadata']!=new['metadata'] or (confirmation is not None and old['metadata']!=confirmation[id_]['metadata']):
      raise ValueError(f'configuration metadata changed: {id_}')
    for record in (old,new,*([confirmation[id_]] if confirmation is not None else [])):
      positive(record['dsl']);positive(record['cpp'])
    dsl=new['dsl']/old['dsl'];cpp=new['cpp']/old['cpp'];suspect=dsl>1+relative and new['dsl']-old['dsl']>absolute_ns
    regressed_roles=[role for role in ('dsl','cpp') if new[role]/old[role]>1+relative and new[role]-old[role]>absolute_ns]
    improved_roles=[role for role in ('dsl','cpp') if new[role]/old[role]<1-relative and old[role]-new[role]>absolute_ns]
    repeated_roles=[role for role in regressed_roles if confirmation is not None and confirmation[id_][role]/old[role]>1+relative and confirmation[id_][role]-old[role]>absolute_ns]
    repeat=confirmation[id_]['dsl'] if confirmation is not None else None
    rows.append({'id':id_,'baseline_ns':{'dsl':old['dsl'],'cpp':old['cpp']},
      'candidate_ns':{'dsl':new['dsl'],'cpp':new['cpp']},'dsl_new_over_old':dsl,'cpp_new_over_old':cpp,
      'relative_dsl_cpp_new_over_old':dsl/cpp,'dsl_delta_ns':new['dsl']-old['dsl'],
      'suspected_regression':bool(regressed_roles),'regressed_roles':regressed_roles,'improvement':bool(improved_roles),'improved_roles':improved_roles,'repeated_regressed_roles':repeated_roles,
      'confirmation_dsl_ns':repeat,'repeated_regression':bool(repeated_roles)})
  metrics={}
  for key in ('dsl_new_over_old','cpp_new_over_old','relative_dsl_cpp_new_over_old'):
    values=[r[key] for r in rows];metrics[key]={'geomean':math.exp(statistics.mean(map(math.log,values))),'median':statistics.median(values)}
  suspects=[r['id'] for r in rows if r['suspected_regression']]
  return {'count':len(rows),'thresholds':{'relative':relative,'absolute_ns':absolute_ns,'rule':'both strict greater-than'},
    'metrics':metrics,'suspected_regressions':suspects,'improvements':[r['id'] for r in rows if r['improvement']],
    'repeated_regressions':[r['id'] for r in rows if r['repeated_regression']], 'review_required':bool(suspects),
    'note':'Per-configuration findings are retained. Confirmation is an independent rerun, not a claim of confidence intervals or absence of environment drift.','rows':rows}


def main(argv=None):
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--kind',choices=('main','supplemental'),required=True)
  p.add_argument('--selection',type=Path,required=True);p.add_argument('--architecture',choices=('arm','x86'),required=True)
  p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--confirmation',type=Path);p.add_argument('--stage',choices=('compute','packing'),default='compute')
  p.add_argument('--relative',type=float,default=.10);p.add_argument('--absolute-ns',type=float,default=50)
  p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
  if not math.isfinite(args.relative) or not math.isfinite(args.absolute_ns) or min(args.relative,args.absolute_ns)<0:p.error('thresholds must be finite and nonnegative')
  field='execution_ids' if args.kind=='main' else 'registration_ids'
  try:identity=selection_identity(args.selection,args.architecture,field)
  except (ValueError,OSError,TypeError) as error:p.exit(2,f'invalid selection: {error}\n')
  if args.confirmation and args.confirmation.resolve() in (args.baseline.resolve(),args.candidate.resolve()):p.error('confirmation must be a separate run path')
  loader=(lambda path:load_main(path,identity,args.stage)) if args.kind=='main' else (lambda path:load_supplemental(path,identity))
  def inputs(path):
    paths=[path/'plan.json',path/'ledger.json'] if args.kind=='main' else [path]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
  try:
    provenance={role:inputs(path) for role,path in [('baseline',args.baseline),('candidate',args.candidate),*([('confirmation',args.confirmation)] if args.confirmation else [])]}
    report=compare(loader(args.baseline),loader(args.candidate),confirmation=loader(args.confirmation) if args.confirmation else None,
      relative=args.relative,absolute_ns=args.absolute_ns)
    if any(inputs(Path(path) if args.kind!='main' else Path(path).parent).get(path)!=expected for documents in provenance.values() for path,expected in documents.items()):
      raise ValueError('comparison input changed during validation')
  except (ValueError,KeyError,OSError,TypeError) as error:p.exit(2,f'invalid comparison evidence: {error}\n')
  report.update(input_sha256=provenance,selection=identity,kind=args.kind,stage=args.stage if args.kind=='main' else None,
    baseline=str(args.baseline.resolve()),candidate=str(args.candidate.resolve()),confirmation=str(args.confirmation.resolve()) if args.confirmation else None)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  return 1 if report['review_required'] else 0


if __name__=='__main__':
  raise SystemExit(main())
