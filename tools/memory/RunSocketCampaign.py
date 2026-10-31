# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Reproducible socket-1 memory campaign for the 920f-4 topology.

Every case retains its complete command, raw CSV, terminal log and exit status.
The default excludes CPUs 306-309, the occupied pytest affinity range.
Full 304-core cases are a separate, explicitly requested phase.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def cpu_spec(n):
    if n>34:
        return f"0-{n-1}"
    if n == 38:
        return "0-37"
    return "4" if n == 1 else f"4-{n + 3}"


def make_cases():
    cases = []
    def add(name, family, targets, execution, threads=1, op="read", access="temporal",
            kernel="vec4", size="4MiB", pages="base", command="bandwidth", **extra):
        case = dict(name=name, family=family, targets=list(targets), execution=list(execution),
                    threads_per_group=threads, operation=op, access=access, kernel=kernel,
                    size=size, pages=pages, command=command, **extra)
        case["id"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:14]
        cases.append(case)
    ops = ("read", "write", "copy", "triad")
    a, b = list(range(8, 12)), list(range(12, 16))
    whole = a + b
    for kind,target,pages,size in (("ddr",12,"base","4MiB"),("hbm",28,"base","4MiB"),
                                   ("hbm_huge",28,"huge","16MiB")):
        for n in (1,2,4,8,15,24,32,36,37,38):
            for op in ops:
                for access in ("temporal","stream"):
                    add(f"scaling_{kind}","scaling",[target],[12],n,op,access,size=size,pages=pages)
    for n in (1,8,15):
        for pages in ('base','huge'):
            for op in ('read','triad'):
                for access in ('temporal','stream'):
                    add('page_control','page_control',[28],[12],n,op,access,size='16MiB',pages=pages)
    # Representative single-NUMA paths, both directions for cross-die paths.
    single = [
        ("ddr_a", [8], [8]), ("ddr_b", [12], [12]),
        ("hbm_a", [24], [8]), ("hbm_b", [28], [12]),
        ("peer_hbm_a", [25], [8]), ("peer_hbm_b", [29], [12]),
        ("cross_ddr_ab", [12], [8]), ("cross_ddr_ba", [8], [12]),
        ("cross_hbm_ab", [28], [8]), ("cross_hbm_ba", [24], [12]),
    ]
    for name, targets, execution in single:
        for n in (1, 8, 16, 37):
            for op in ops:
                for access in ("temporal", "stream"):
                    add(name, "single", targets, execution, n, op, access)
    aggregate = [
        ("die_a_ddr", a, a), ("die_b_ddr", b, b), ("socket_ddr", whole, whole),
        ("die_a_hbm", [24,25,26,27], a), ("die_b_hbm", [28,29,30,31], b),
        ("socket_hbm", list(range(24,32)), whole),
        ("die_a_peer_hbm", [25,26,27,24], a), ("die_b_peer_hbm", [29,30,31,28], b),
        ("socket_peer_hbm", [25,26,27,24,29,30,31,28], whole),
        ("die_cross_ddr_ab", b, a), ("die_cross_ddr_ba", a, b),
        ("die_cross_hbm_ab", [28,29,30,31], a), ("die_cross_hbm_ba", [24,25,26,27], b),
        ("socket_cross_ddr", b+a, whole),
        ("socket_cross_hbm", [28,29,30,31,24,25,26,27], whole),
    ]
    for name, targets, execution in aggregate:
        for n in (8, 16, 37):
            for op in ops:
                for access in ("temporal", "stream"):
                    add(name, "aggregate", targets, execution, n, op, access)
    # Exact full-NUMA and full-die-B coverage avoids the occupied die-A core.
    for name, targets, execution in (("ddr_b_38", [12], [12]), ("hbm_b_38", [28], [12]),
                                     ("die_b_ddr_152", b, b), ("die_b_hbm_152", [28,29,30,31], b)):
        for op in ops:
            for access in ("temporal", "stream"):
                add(name, "full_die_b", targets, execution, 38, op, access)
    for name,targets,execution in (("available_die_a_ddr",a,a),("available_die_a_hbm",[24,25,26,27],a),
                                  ("available_socket_ddr",whole,whole),("available_socket_hbm",list(range(24,32)),whole)):
        for op in ops:
            for access in ("temporal","stream"):
                add(name,"available",targets,execution,38,op,access)
    for node in whole:
        for kind, target in (("ddr", node), ("hbm", node+16)):
            for access in ("temporal", "stream"):
                add(f"balance_{kind}_{node}", "balance", [target], [node], 16, "read", access)
    # Isolate the six work-per-iteration choices at one core and at 32 cores.
    for kind, target in (("ddr", 12), ("hbm", 28)):
        for n in (1, 32):
            for kernel in ("scalar16", "scalar32", "scalar64", "vec1", "vec2", "vec4"):
                for op in ops:
                    for access in ("temporal", "stream"):
                        add(f"kernel_{kind}", "kernels", [target], [12], n, op, access, kernel)
    # Keep all active arrays below each private cache's capacity.
    for cache, size in (("l1", "4KiB"), ("l2", "128KiB")):
        for kernel in ("scalar16", "scalar32", "scalar64", "vec1", "vec2", "vec4"):
            for op in ops:
                add(f"cache_{cache}", "cache_kernels", [12], [12], 1, op, kernel=kernel, size=size)
        for scale, targets in (("numa", [12]), ("die", b), ("socket", whole)):
            for op in ops:
                add(f"cache_{cache}_{scale}", "cache_aggregate", targets, targets, 37, op, size=size)
        for placement in ('ddr_base','hbm_base','hbm_huge'):
            for scale,execution,n in (('single',[12],1),('numa',[12],37),('die',b,37),('socket',whole,37)):
                targets=execution if placement=='ddr_base' else [x+16 for x in execution]
                for op in ops:
                    add(f'cache_control_{cache}_{placement}_{scale}','cache_control',targets,execution,n,op,
                        size=size,pages='huge' if placement=='hbm_huge' else 'base')
    sizes = ("4KiB","8KiB","16KiB","24KiB","32KiB","48KiB","64KiB","128KiB",
             "256KiB","512KiB","768KiB","1MiB","2MiB","4MiB","8MiB","16MiB")
    for size in sizes:
        for op in ops:
            add("cache_curve", "cache_curve", [12], [12], 1, op, size=size)
        add("cache_chase", "cache_latency", [12], [12], command="latency", size=size)
    # Confirm off-cache working-set stability and page-size sensitivity.
    for kind, target in (("ddr",12),("hbm",28)):
        for size in ("2MiB","4MiB","8MiB","16MiB","32MiB","64MiB"):
            for op in ("read","triad"):
                add(f"size_{kind}", "working_set", [target], [12], 1, op, size=size)
    for name, targets, execution in (("huge_hbm_single",[28],[12]),
                                     ("huge_hbm_die",[28,29,30,31],b),
                                     ("huge_hbm_socket",list(range(24,32)),whole),
                                     ("huge_hbm_cross",[28,29,30,31],a)):
        for n in (1, 16, 37):
            for op in ops:
                for access in ("temporal","stream"):
                    add(name,"huge_hbm",targets,execution,n,op,access,size="16MiB",pages="huge")
    # Full initiator-to-memory-target unloaded latency matrices.
    for execution in whole:
        for target in whole + list(range(24,32)):
            add("latency_matrix", "latency_matrix", [target], [execution], command="latency",size="16MiB")
        for target in range(24,32):
            add("latency_huge", "latency_huge", [target], [execution],command="latency",size="16MiB",pages="huge")
    for name, target in (("local_ddr",12),("local_hbm",28),("peer_hbm",29),("cross_ddr",8),("cross_hbm",24)):
        add(f"idle_{name}","loaded_baseline",[target],[12],command="latency",cpu_offset=0)
        for n in (4,16,32):
            for op in ("read","triad"):
                add(f"loaded_{name}","loaded_latency",[target],[12],n,op,command="latency",loaded=True)
    for name, route, execution, peer in (("same_numa","12",12,5),
                                         ("same_die","12:13",12,4),
                                         ("cross_die","12:8",12,4)):
        add(name,"pingpong",[12],[execution],command="pingpong",route=route,peer_cpu=peer)
    # Distinguish remote access from an actual source-to-destination transfer.
    for name,route in (("ddr_to_hbm","12:28"),("hbm_to_ddr","28:12"),
                       ("peer_hbm_forward","28:29"),("peer_hbm_reverse","29:28"),
                       ("cross_ddr_forward","12:8"),("cross_ddr_reverse","8:12"),
                       ("cross_hbm_forward","28:24"),("cross_hbm_reverse","24:28")):
        for n in (1,16,37):
            for op in ('copy','triad'):
                for access in ('temporal','stream'):
                    add(name,'transfer',[route],[12],n,op,access)
    for name,routes,execution in (
            ('die_ddr_to_hbm',[f'{n}:{n+16}' for n in b],b),
            ('die_hbm_to_ddr',[f'{n+16}:{n}' for n in b],b),
            ('die_peer_hbm',['28:29','29:30','30:31','31:28'],b),
            ('die_cross_ddr',[f'{n}:{n-4}' for n in b],b),
            ('die_cross_hbm',[f'{n+16}:{n+12}' for n in b],b)):
        for n in (16,37):
            for op in ('copy','triad'):
                for access in ('temporal','stream'):
                    add(name,'transfer',routes,execution,n,op,access)
    # Deferred until the occupied socket-1 core is released.
    for name, targets, execution in (("full_die_a_ddr",a,a),("full_die_a_hbm",[24,25,26,27],a),
                                     ("full_socket_ddr",whole,whole),("full_socket_hbm",list(range(24,32)),whole)):
        for op in ops:
            for access in ("temporal","stream"):
                add(name,"full",targets,execution,38,op,access)
    return cases


def snapshot():
    nodes={}
    for n in range(8,32):
        p=Path(f"/sys/devices/system/node/node{n}")
        nodes[n]={"cpulist":(p/"cpulist").read_text().strip(),
                  "meminfo":(p/"meminfo").read_text(),"distance":(p/"distance").read_text().strip()}
    freq={}
    for n in range(8,16):
        p=Path(f"/sys/devices/system/cpu/cpu{n*38}/cpufreq")
        freq[n]={f:(p/f).read_text().strip() for f in ("scaling_governor","scaling_cur_freq","scaling_min_freq","scaling_max_freq") if(p/f).exists()}
    return dict(time=datetime.now(timezone.utc).isoformat(),nodes=nodes,frequency=freq,
                load=Path('/proc/loadavg').read_text(),cpu_stat=Path('/proc/stat').read_text(),
                allowed_cpus=sorted(os.sched_getaffinity(0)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',default='all')
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--seconds',type=float,default=.2)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--retries',type=int,default=0)
    parser.add_argument('--exclude-cpus',default='306-309')
    parser.add_argument('--case-ids',type=Path,help='JSON array of case IDs for targeted reruns')
    parser.add_argument('--replay-manifest',type=Path,help='Replay the exact retained case definitions')
    parser.add_argument('--default-memory-bind',default='',help='Optional process default memory mask via numactl; include huge-page target nodes')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    excluded=set()
    for part in args.exclude_cpus.split(',') if args.exclude_cpus else []:
        bounds=list(map(int,part.split('-')))
        excluded.update(range(bounds[0],bounds[-1]+1))
    os.sched_setaffinity(0,set(range(304,608))-excluded)
    cases=json.loads(args.replay_manifest.read_text())['cases'] if args.replay_manifest else make_cases()
    phases=args.phase.split(',')
    if phases==['scaling']:phases+=['page_control','loaded_baseline','transfer','cache_control']
    selected=cases if args.replay_manifest else [c for c in cases if (c['family']!='full' if phases==['all'] else c['family'] in phases)]
    if args.case_ids:
        requested=set(json.loads(args.case_ids.read_text()))
        selected=[c for c in cases if c['id'] in requested]
        missing=requested-{c['id'] for c in selected}
        if missing:raise ValueError(f'Unknown case IDs: {sorted(missing)}')
    if args.pilot:
        selected=[]
        for n,targets,execution in ((1,[28],[12]),(37,[28],[12]),
                                    (37,[28,29,30,31],[12,13,14,15]),
                                    (37,list(range(24,32)),list(range(8,16)))):
            selected.append(dict(name='pilot',family='pilot',targets=targets,execution=execution,
                                 threads_per_group=n,operation='read',access='stream',kernel='vec4',
                                 size='4MiB',pages='base',command='bandwidth',id=f'pilot-{len(execution)}-{n}'))
    manifest=dict(schema=1,created=datetime.now(timezone.utc).isoformat(),
                  binary=str(args.binary.resolve()),samples=args.samples,seconds=args.seconds,
                  excluded_cpus=sorted(excluded),die_a=[8,9,10,11],die_b=[12,13,14,15],cases=selected)
    manifest['default_memory_bind']=args.default_memory_bind
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and (args.output/'states.jsonl').exists():
        previous=json.loads(manifest_path.read_text())
        for key in ('binary','samples','seconds','excluded_cpus','default_memory_bind'):
            if previous.get(key,'' if key=='default_memory_bind' else None)!=manifest.get(key):
                raise ValueError(f'Resume conditions changed for {key}; use a separate output directory')
        retained={c['id']:c for c in previous['cases']}
        for case in selected:
            if case['id'] in retained and retained[case['id']]!=case:
                raise ValueError('Resumed case definition changed: '+case['id'])
            retained[case['id']]=case
        # A resumed subset must not orphan previously completed state records.
        manifest['cases']=list(retained.values())
        manifest['created']=previous.get('created',manifest['created'])
    manifest_path.write_text(json.dumps(manifest,indent=2))
    if not (args.output/'before.json').exists():
        (args.output/'before.json').write_text(json.dumps(snapshot(),indent=2))
    states=args.output/'states.jsonl'
    completed={}
    if states.exists():
        for line in states.read_text().splitlines():
            r=json.loads(line); completed[r['id']]=r
    for i,c in enumerate(selected,1):
        if c['id'] in completed:
            continue
        if c['family']=='full' and any(node*38+offset in excluded for node in c['execution'] for offset in range(38)):
            raise ValueError('Full-core cases require every selected CPU; pass an empty exclusion list after checking availability')
        stem=f"{c['family']}-{c['name']}-{c['id']}"
        csv_path=args.output/(stem+'.csv')
        offsets=[]
        for node in c['execution']:
            order=list(range(4,38))+list(range(4))
            permitted=[x for x in order if node*38+x not in excluded]
            offsets.append([c['cpu_offset']] if 'cpu_offset' in c else permitted[:c['threads_per_group']])
        group_cpus='|'.join(','.join(map(str,x)) for x in offsets)
        command=[str(args.binary.resolve()),c['command'],'--numa',c.get('route',','.join(map(str,c['targets']))),
                 '--exec-node',','.join(map(str,c['execution'])),'--cpu',cpu_spec(c['threads_per_group']),
                 '--group-cpus',group_cpus,
                 '--size',c['size'],'--pages',c['pages'],'--seconds',str(c.get('seconds',args.seconds)),
                 '--samples',str(args.samples),'--retries',str(args.retries),'--warmup','.05','--csv',str(csv_path.resolve())]
        if c['command']=='bandwidth' or c.get('loaded'):
            command+=['--op',c['operation'],'--kernel',c['kernel'],'--access',c['access']]
        if c.get('interleave'):
            command+=['--interleave']
        if c.get('loaded'):
            # Reserve offset 0 for chase; background starts at offset 4.
            command+=['--loaded','--latency-cpu','0']
        if c['command']=='pingpong':
            peer=c['peer_cpu']
            target_node=int(c.get('route','12').split(':')[-1])
            while target_node*38+peer in excluded:peer+=1
            command+=['--peer-cpu',str(peer)]
        if args.default_memory_bind:
            command=['numactl','--membind='+args.default_memory_bind,*command]
        (args.output/(stem+'.command.json')).write_text(json.dumps(command,indent=2))
        started=time.monotonic()
        with (args.output/(stem+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        rows=[]
        if csv_path.exists():
            with csv_path.open(newline='') as stream: rows=list(csv.DictReader(stream))
        metric=[r for r in rows if r['scope'] in ('total','chase:g0','round-trip')]
        valid=sum(r['status']=='ok' for r in metric)
        state=dict(id=c['id'],index=i,total=len(selected),name=c['name'],family=c['family'],
                   exit=result.returncode,elapsed=round(time.monotonic()-started,3),
                   valid_rows=valid,raw_rows=len(rows),file=csv_path.name,
                   time=datetime.now(timezone.utc).isoformat())
        with states.open('a') as stream:stream.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if result.returncode not in (0,1):
            print('CASE_ERROR: '+(args.output/(stem+'.log')).read_text()[-2000:],flush=True)
        if i%20==0:(args.output/'latest-health.json').write_text(json.dumps(snapshot(),indent=2))
    (args.output/'after.json').write_text(json.dumps(snapshot(),indent=2))
    print('CAMPAIGN_COMPLETE',flush=True)


if __name__=='__main__':main()
