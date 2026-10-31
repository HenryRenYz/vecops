# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Audit retained socket-1 CPU placement and byte/time accounting."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path


def cpu_list(value):
    cpus=set()
    for part in value.split(','):
        if not part:continue
        bounds=list(map(int,part.split('-')))
        cpus.update(range(bounds[0],bounds[-1]+1))
    return cpus


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    files=0;rows=0;threads=0;errors=[];hashes=set();exits=collections.Counter();switches=0
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir() or not (directory/'manifest.json').exists():continue
        if directory.name in ('probe','pilot'):continue
        manifest=json.loads((directory/'manifest.json').read_text());excluded=set(manifest['excluded_cpus'])
        states={}
        if not (directory/'states.jsonl').exists():continue
        for line in (directory/'states.jsonl').read_text().splitlines():
            state=json.loads(line);states[state['id']]=state
        for case_id,state in states.items():
            exits[state['exit']]+=1
            if state['exit'] not in (0,1):errors.append(f'{directory.name}/{case_id}: exit {state["exit"]}')
            path=directory/state['file']
            with path.open(newline='') as stream:data=list(csv.DictReader(stream))
            files+=1;rows+=len(data)
            by_sample=collections.defaultdict(list)
            for row in data:
                hashes.add(row['tool_source_sha256'])
                if row['status']=='unsupported':errors.append(f'{path.name}: unexpected unsupported kernel');continue
                used=cpu_list(row['cpus'])
                if not used or not used.issubset(set(range(304,608))):errors.append(f'{path.name}: CPU outside socket 1: {used}')
                if used&excluded:errors.append(f'{path.name}: excluded CPU used: {used&excluded}')
                if row['placement']!='all-pages-verified':errors.append(f'{path.name}: placement failed')
                if row['scope'].startswith('thread:'):threads+=1
                if row.get('schema_version')=='2' and row['scope'].startswith('thread:') and row['kind'] in ('bandwidth','latency'):
                    unit=2**21 if row['page_policy']=='huge' else int(row['base_page_bytes'])
                    pages=(int(row['per_thread_array_bytes'])+unit-1)//unit
                    for role in ('source','destination'):
                        nodes=[int(x) for x in row[role+'_memory_nodes'].split(',') if x]
                        op=row['operation']
                        arrays=(0 if op=='write' else 2 if op=='triad' else 1) if role=='source' else (0 if op in ('read','chase') else 1)
                        expected={n:arrays*(pages//len(nodes)+(i<pages%len(nodes))) for i,n in enumerate(nodes)} if nodes else {}
                        counts=dict(tuple(map(int,x.split(':'))) for x in row[role+'_page_counts'].split('|') if x)
                        if counts!=expected or bool(nodes)!=bool(arrays):errors.append(f'{path.name}: cyclic {role} page counts mismatch')
                switches+=int(row['context_switches']) if row['scope'].startswith('thread:') else 0
                begin,end=int(row['start_ns']),int(row['end_ns'])
                seconds=float(row['elapsed_seconds'])
                if not math.isclose(seconds,(end-begin)*1e-9,rel_tol=2e-10,abs_tol=1e-12):errors.append(f'{path.name}: elapsed envelope mismatch')
                if row['kind']=='bandwidth':
                    rate=(int(row['read_bytes'])+int(row['write_bytes']))/seconds/1e9
                    if not math.isclose(rate,float(row['total_GB_s']),rel_tol=2e-10):errors.append(f'{path.name}: bandwidth arithmetic mismatch')
                else:
                    if not math.isclose((end-begin)/int(row['accesses']),float(row['ns_per_access']),rel_tol=2e-10):errors.append(f'{path.name}: latency arithmetic mismatch')
                by_sample[(row['sample'],row['attempt'])].append(row)
            for sample in by_sample.values():
                total=next((r for r in sample if r['scope']=='total'),None)
                workers=[r for r in sample if r['kind']=='bandwidth' and r['scope'].startswith('thread:')]
                if total and workers:
                    if int(total['start_ns'])!=min(int(r['start_ns']) for r in workers) or int(total['end_ns'])!=max(int(r['end_ns']) for r in workers):errors.append(f'{path.name}: total is not full thread envelope')
                    for field in ('read_bytes','write_bytes'):
                        if int(total[field])!=sum(int(r[field]) for r in workers):errors.append(f'{path.name}: aggregate {field} mismatch')
                    if total.get('schema_version')=='2':
                        for role in ('source','destination'):
                            expected=collections.Counter()
                            for worker in workers:
                                for pair in worker[role+'_page_counts'].split('|'):
                                    if pair:
                                        node,count=map(int,pair.split(':'));expected[node]+=count
                            actual=dict(tuple(map(int,x.split(':'))) for x in total[role+'_page_counts'].split('|') if x)
                            if actual!=dict(expected):errors.append(f'{path.name}: aggregate {role} page counts mismatch')
    if len(hashes)!=1:errors.append('Unexpected number of benchmark source hashes: '+str(hashes))
    result=dict(files=files,rows=rows,thread_rows=threads,exit_codes=dict(exits),source_hashes=sorted(hashes),errors=errors)
    (args.root/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
