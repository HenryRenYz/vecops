# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Summarize retained campaign observations without hiding scheduler flags.

Main observations require verified placement, no timed page faults/migration,
and the original start/end skew bound. Context-switched observations retain
their elapsed envelope and are counted separately from strict clean samples.
All accepted observations, including interrupted ones, enter the same median;
there is no per-case selection of a faster clean-only statistic.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def usable(row):
    if row.get('placement')!='all-pages-verified':return False
    if row['status'] in ('unsupported','verification-failed','migration','missed-window','page-fault','retry'):return False
    if int(row['minor_faults']) or int(row['major_faults']):return False
    duration=float(row['target_seconds'])
    tolerance=duration*1e9*float(row['max_skew_fraction'])
    if int(row['start_skew_ns'])>tolerance or int(row['end_skew_ns'])>tolerance:return False
    if row['kind']=='bandwidth' and int(row['common_ns'])<duration*1e9*(1-2*float(row['max_skew_fraction'])):return False
    if row['status']=='skew':return False
    key='total_GB_s' if row['kind']=='bandwidth' else 'ns_per_access'
    return math.isfinite(float(row[key])) and float(row[key])>0


def summarize(root):
    summaries=[];observations=[];states=[];sources=set();metadata={};cases_out=[]
    for run in sorted(root.iterdir()):
        if not run.is_dir() or run.name in ('pilot','probe'):continue
        if not (run/'manifest.json').exists() or not (run/'states.jsonl').exists():continue
        manifest=json.loads((run/'manifest.json').read_text())
        cases={c['id']:c for c in manifest['cases']}
        case_states={}
        for line in (run/'states.jsonl').read_text().splitlines():
            try:state=json.loads(line)
            except json.JSONDecodeError:continue
            case_states[state['id']]=state
        for case_id,state in case_states.items():
            case=cases[case_id];state=dict(state,run=run.name);states.append(state)
            path=run/state['file']
            if not path.exists():continue
            with path.open(newline='') as stream:
                rows=[r for r in csv.DictReader(stream) if r.get('scope') and r.get('kind')]
            if not rows:continue
            accepted={id(r):usable(r) for r in rows}
            if case.get('loaded'):
                attempts=collections.defaultdict(list)
                for row in rows:attempts[(row['sample'],row['attempt'])].append(row)
                for participants in attempts.values():
                    traffic=next((r for r in participants if r['scope']=='total'),None)
                    chase=next((r for r in participants if r['scope']=='chase:g0'),None)
                    joint=bool(traffic and chase and usable(traffic) and usable(chase))
                    if joint:
                        duration=float(traffic['target_seconds'])*1e9
                        fraction=float(traffic['max_skew_fraction'])
                        earliest_start=min(int(chase['start_ns']),int(traffic['start_ns']))
                        latest_start=max(int(chase['start_ns']),int(traffic['start_ns'])+int(traffic['start_skew_ns']))
                        earliest_end=min(int(chase['end_ns']),int(traffic['end_ns'])-int(traffic['end_skew_ns']))
                        latest_end=max(int(chase['end_ns']),int(traffic['end_ns']))
                        joint=(latest_start-earliest_start<=duration*fraction and
                               latest_end-earliest_end<=duration*fraction and
                               earliest_end-latest_start>=duration*(1-2*fraction))
                    for row in participants:
                        if not row['scope'].startswith('thread:'):accepted[id(row)]=joint
            sources.add(rows[0]['tool_source_sha256'])
            vector_bytes=max(int(metadata.get('vector_bytes','0')),int(rows[0]['vector_bytes']))
            metadata={k:rows[0].get(k,'') for k in ('host','cpu_model','compiler','build_flags','revision','tool_source_sha256','vector_bytes','base_page_bytes')}
            metadata['vector_bytes']=str(vector_bytes)
            by_scope=collections.defaultdict(list)
            for row in rows:
                if row['scope'].startswith('thread:'):continue
                by_scope[row['scope']].append(row)
            case_info=dict(case,run=run.name,csv=str(path.relative_to(root)),default_memory_bind=manifest.get('default_memory_bind',''))
            cases_out.append(case_info)
            for scope,samples in by_scope.items():
                good=[r for r in samples if accepted[id(r)]]
                clean=[r for r in good if r['status']=='ok' and int(r['context_switches'])==0]
                first=samples[0]
                key='total_GB_s' if first['kind']=='bandwidth' else 'ns_per_access'
                values=[float(r[key]) for r in good]
                median=statistics.median(values) if values else None
                groups=first.get('requested_group_cpus','').split('|')
                counts=[len(x.split(',')) for x in groups if x]
                record=dict(case_id=case_id,case_name=case['name'],family=case['family'],run=run.name,
                            default_memory_bind=manifest.get('default_memory_bind',''),
                            tool_source_sha256=first.get('tool_source_sha256',''),
                            memory_policy=first.get('memory_policy','bind'),
                            source_memory_nodes=first.get('source_memory_nodes',''),
                            destination_memory_nodes=first.get('destination_memory_nodes',''),
                            source_page_counts=first.get('source_page_counts',''),
                            destination_page_counts=first.get('destination_page_counts',''),
                            interleave_bytes=int(first.get('interleave_bytes',0)),
                            kind=first['kind'],scope=scope,operation=first['operation'],
                            background_operation=case['operation'] if case.get('loaded') else '',
                            kernel=first['kernel'],access=first['requested_access'],pages=case['pages'],
                            targets=case['targets'],execution=case['execution'],threads=int(first['threads']),
                            cpus=first['cpus'],
                            group_cpu_offsets=first.get('requested_group_cpus',''),
                            source_node=int(first['source_node']),destination_node=int(first['destination_node']),execution_node=int(first['execution_node']),
                            group_threads=counts,requested_threads_per_group=case['threads_per_group'],
                            per_thread_array_bytes=int(first['per_thread_array_bytes']),
                            working_set_bytes=int(first['working_set_bytes']),requested_size=case['size'],
                            target_seconds=float(first['target_seconds']),
                            samples_recorded=len(samples),samples_usable=len(good),samples_strict=len(clean),
                            median=median,minimum=min(values) if values else None,
                            maximum=max(values) if values else None,
                            spread_pct=(max(values)-min(values))/median*100 if values and median else None,
                            stdev=statistics.stdev(values) if len(values)>1 else None,
                            copy_payload_median=statistics.median(float(r['payload_GB_s']) for r in good) if good and first['operation']=='copy' else None,
                            interrupted_usable=sum(int(r['context_switches'])>0 or r['status']=='interrupted' for r in good),
                            max_start_skew_us=max(int(r['start_skew_ns']) for r in samples)/1000,
                            max_end_skew_us=max(int(r['end_skew_ns']) for r in samples)/1000,
                            rejected_statuses=dict(collections.Counter(r['status'] for r in samples if not accepted[id(r)])),
                            all_statuses=dict(collections.Counter(r['status'] for r in samples)),
                            source_csv=str(path.relative_to(root)),raw_exit=state['exit'])
                summaries.append(record)
                for row in samples:
                    observations.append(dict(case_id=case_id,run=run.name,scope=scope,kind=row['kind'],
                                             operation=row['operation'],sample=int(row['sample']),attempt=int(row['attempt']),
                                             value=float(row[key]),usable=accepted[id(row)],strict=accepted[id(row)] and row['status']=='ok',
                                             status=row['status'],context_switches=int(row['context_switches']),
                                             minor_faults=int(row['minor_faults']),start_skew_ns=int(row['start_skew_ns']),
                                             end_skew_ns=int(row['end_skew_ns']),cpus=row['cpus']))
    if len(sources)>1:raise RuntimeError('Campaign source hashes differ; separate the runs before comparing')
    result=dict(metadata=metadata,source_hashes=sorted(sources),cases=cases_out,states=states,
                summaries=summaries,observations=observations)
    result['selected_summaries']=select_and_pool(result)
    return result


def select_and_pool(data):
    """Select the latest prescribed conditions, pooling identical repeats.

    Different windows, default memory policies or actual CPU maps never mix.
    All qualifying observations in an identical-condition cohort contribute,
    regardless of their speed, scheduler status or original process exit code.
    """
    preferred={};observations=collections.defaultdict(list)
    for row in data['summaries']:
        key=(row['case_id'],row['scope'])
        rank=(row['run'].startswith('repair'),row['run']=='scaling',row['run'])
        if key not in preferred or rank>preferred[key][0]:preferred[key]=(rank,row)
    for obs in data['observations']:observations[(obs['run'],obs['case_id'],obs['scope'])].append(obs)
    def condition(row):
        return tuple(str(row.get(k,'')) for k in ('case_id','scope','target_seconds','default_memory_bind','cpus','group_cpu_offsets','per_thread_array_bytes','pages','operation','kernel','access','tool_source_sha256','memory_policy','source_memory_nodes','destination_memory_nodes','interleave_bytes'))
    cohorts=collections.defaultdict(list)
    for row in data['summaries']:cohorts[condition(row)].append(row)
    selected=[]
    for _,latest in preferred.values():
        members=cohorts[condition(latest)];row=dict(latest)
        obs=[o for member in members for o in observations[(member['run'],member['case_id'],member['scope'])]]
        good=[o for o in obs if o['usable']];values=[o['value'] for o in good]
        row.update(samples_recorded=len(obs),samples_usable=len(good),samples_strict=sum(o['strict'] for o in good),
                   median=statistics.median(values) if values else None,
                   minimum=min(values) if values else None,maximum=max(values) if values else None,
                   stdev=statistics.stdev(values) if len(values)>1 else None,
                   interrupted_usable=sum(o['context_switches']>0 or o['status']=='interrupted' for o in good),
                   pooled_runs=[m['run'] for m in members],source_csvs=[m['source_csv'] for m in members])
        row['spread_pct']=(row['maximum']-row['minimum'])/row['median']*100 if row['median'] else None
        row['copy_payload_median']=row['median']/2 if row['operation']=='copy' and row['median'] else None
        row['qualified_median']=row['median'] if row['samples_usable']>=3 else None
        row['all_statuses']=dict(collections.Counter(o['status'] for o in obs))
        row['rejected_statuses']=dict(collections.Counter(o['status'] for o in obs if not o['usable']))
        row['max_start_skew_us']=max(o['start_skew_ns'] for o in obs)/1000
        row['max_end_skew_us']=max(o['end_skew_ns'] for o in obs)/1000
        selected.append(row)
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=summarize(args.root)
    (args.root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    rows=result['summaries']
    if rows:
        fields=list(rows[0])
        with (args.root/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    selected=result['selected_summaries']
    if selected:
        with (args.root/'selected-summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader()
            for row in selected:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    metric=[r for r in rows if r['scope'] in ('total','chase:g0','round-trip')]
    poor=sorted({r['case_id'] for r in selected if r['scope'] in ('total','chase:g0','round-trip') and r['samples_usable']<3})
    (args.root/'repair-ids.json').write_text(json.dumps(poor,indent=2))
    print(json.dumps(dict(completed_cases=len(result['cases']),metric_rows=len(metric),
                          cases_with_fewer_than_three_usable=len(poor),
                          strict_clean=sum(r['samples_strict'] for r in metric),
                          usable=sum(r['samples_usable'] for r in metric)),indent=2))


if __name__=='__main__':main()
