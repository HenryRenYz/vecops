# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scheduled-node SDMA campaign; explicit devices, socket-1 arrays/controllers.

Retains every command, exit, stdout and CSV. Resumes only with an identical ELF.
Does not change driver configuration or global page pools. Default statistics
accept synchronized, placement/data-verified, fault-free observations including
context switches with the full elapsed time. Insufficient cases are retained.
"""
import argparse
import ctypes
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

SCRIPT_SOURCE = Path(__file__).read_bytes()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--phase', default='all', choices=['all', 'quick'])
    a = p.parse_args()
    binary = a.binary.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    records = []
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['binary_sha256'] != sha(binary):
            raise RuntimeError('Cannot resume measurements from a different binary')
        records = previous['cases']
    visible = sorted(int(x.name[4:]) for x in Path('/sys/class/sdma').glob('sdma[0-9]*'))
    if not visible:
        raise RuntimeError('No driver-visible SDMA devices')
    saved_affinity = os.sched_getaffinity(0)
    packages = {str(n): sorted({read(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id') for c in range(n*38,n*38+37)}) for n in range(8,16)}
    if any(v != ['1'] for v in packages.values()):
        raise RuntimeError('Expected socket-1 CPU/node mapping was not verified: '+str(packages))
    die_ids = {str(n): read(f'/sys/devices/system/cpu/cpu{n*38}/topology/die_id') for n in range(8,16)}
    recommended = {}
    sdk = ctypes.CDLL('libsdma_dk.so')
    sdk.sdma_nearest_id.restype = ctypes.c_int
    for node in range(8, 16):
        cpus = list(range(node * 38, node * 38 + 37))
        if not set(cpus).issubset(saved_affinity):
            raise RuntimeError('Allocation must include all 296 socket-1 compute cores')
        os.sched_setaffinity(0, {cpus[0]})
        recommended[node] = sdk.sdma_nearest_id()
    os.sched_setaffinity(0, saved_affinity)
    if any(d not in visible for d in recommended.values()):
        raise RuntimeError('SDK recommended an unenumerated device')
    # Keep the campaign controller outside every selected benchmark controller list.
    os.sched_setaffinity(0, {606})
    child_affinity = set(saved_affinity)
    def child_setup():
        os.sched_setaffinity(0, child_affinity)
    metadata = dict(host=platform.node(), kernel=platform.release(), started=stamp(),
                    binary=str(binary), binary_sha256=sha(binary), visible_devices=visible,
                    physical_engine_count=None, cpu_packages=packages, cpu_die_ids=die_ids, device_numa='unknown unless explicit sysfs attribute',
                    recommended_devices={str(k): v for k, v in recommended.items()},
                    recommendation_source='installed sdma_nearest_id SDK / driver ioctl, sampled at one CPU per NUMA',
                    affinity=sorted(saved_affinity), scope='socket 1 memory/controllers; all visible SDMA devices tested',
                    share_chns=read('/sys/module/sdma_dae/parameters/share_chns'),
                    safe_mode=read('/sys/module/sdma_dae/parameters/safe_mode'),
                    scheduler_job=os.environ.get('BATCH_JOB_ID', os.environ.get('LSB_JOBID', '')),
                    packages=subprocess.getoutput('rpm -q sdma-dk'),
                    sdk_sha256=sha('/usr/lib64/libsdma_dk.so'),
                    header_sha256=sha('/usr/include/mdk_sdma.h'),
                    memory={str(i): dict(meminfo=read(f'/sys/devices/system/node/node{i}/meminfo'),
                        huge_free=read(f'/sys/devices/system/node/node{i}/hugepages/hugepages-2048kB/free_hugepages')) for i in range(32)},
                    load_before=read('/proc/loadavg'))
    if previous:
        for key in ['host','sdk_sha256','header_sha256','share_chns','safe_mode','affinity','recommended_devices']:
            if metadata[key]!=previous[key]:raise RuntimeError('Resume environment changed: '+key)
        metadata['started']=previous['started']
        metadata['resume_history']=previous.get('resume_history',[])+[stamp()]
        if 'repair_history' in previous:metadata['repair_history']=previous['repair_history']
    (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
    # Reserve no additional huge pages. Use only capacity already configured.
    huge = all(int(metadata['memory'][str(i)]['huge_free'] or '0') >= 192 for i in list(range(8,16))+list(range(24,32)))
    hbm_pages = 'huge' if huge else 'base'
    cases = []
    def add(name, family, routes='8:24', devices=None, cpus='0', size='64MiB',
            channels=4, depth=16, batch=8, chunk='64KiB', pages='base', seconds=.5,
            samples=5, interleave=False, engine='sdma', access=None, exec_nodes=None, size_scope='thread-array'):
        route_list = routes.split(',')
        nodes = [int(r.split(':')[0]) for r in route_list]
        execution = exec_nodes or ','.join(str(n if n < 16 else n - 16) for n in nodes)
        if devices is None:
            devices = '|'.join(str(recommended[int(n)]) for n in execution.split(','))
        command = [str(binary), 'bandwidth', '--engine', engine, '--op', 'copy',
                   '--numa', routes, '--exec-node', execution, '--cpu', cpus,
                   '--size', size, '--size-scope', size_scope, '--pages', pages, '--seconds', str(seconds),
                   '--warmup', '.05', '--samples', str(samples), '--retries', '0', '--max-skew', '.001']
        if engine == 'sdma':
            command += ['--sdma-device', str(devices), '--sdma-mode', 'shared',
                        '--sdma-channels', str(channels), '--sdma-depth', str(depth),
                        '--sdma-batch', str(batch), '--sdma-chunk', chunk]
        else:
            command += ['--kernel', 'vec4', '--access', access or 'stream']
        if interleave:
            command += ['--interleave']
        cases.append(dict(id=f'{len(cases):04d}-{name}', name=name, family=family,
                          routes=routes, devices=devices, controllers=cpus, size=size,
                          channels=channels, depth=depth, batch=batch, chunk=chunk,
                          pages=pages, seconds=seconds, samples=samples, interleave=interleave,
                          engine=engine, command=command))
    # Edge and partial-chunk verification; each sample changes source generation.
    for d in visible:
        for routes in ['8:8', '8:24', '24:8', '24:24']:
            add(f'correct-d{d}-{routes}', 'correctness', routes, d, size='65540',
                channels=1, depth=4, batch=2, chunk='16KiB', seconds=.05, samples=2)
    if huge:
        add('correct-huge-tail', 'correctness', size='2097156', channels=1,
            depth=4, batch=2, chunk='64KiB', pages='huge', seconds=.05, samples=2)
    # Same route/controller on all devices: device IDs imply no locality.
    for d in visible:
        for routes in ['8:8', '8:24', '24:8', '24:24']:
            for ch in [1, 2, 4, 8, 16, 32]:
                add(f'device-{d}-{routes}-ch{ch}', 'device-scaling', routes, d, channels=ch)
    # Descriptor and queue-depth behavior, including a single outstanding request.
    for routes in ['8:24', '24:24']:
        for chunk in ['4KiB', '16KiB', '64KiB', '256KiB', '1MiB']:
            for depth in [1, 4, 16, 64]:
                add(f'queue-{routes}-{chunk}-q{depth}', 'queue', routes,
                    channels=1, depth=depth, batch=min(8, depth), chunk=chunk)
        for batch in [1, 2, 4, 8, 16]:
            add(f'batch-{routes}-b{batch}', 'batch', routes, channels=4, depth=16, batch=batch)
    # Separate controller scalability from channel count.
    for n in [1, 2, 4, 8]:
        add(f'controllers-{n}', 'controllers', cpus='0' if n == 1 else f'0-{n-1}', channels=16)
    # Wide-channel/large-request saturation with fixed group working set.
    for d in visible:
        for routes in ['8:8', '8:24', '24:8', '24:24']:
            for channels in [16, 64, 128]:
                for n in [1, 4, 16]:
                    add(f'saturation-{d}-{routes}-ch{channels}-t{n}', 'saturation', routes, d,
                        cpus='0' if n==1 else f'0-{n-1}',size='128MiB',size_scope='group-array',
                        channels=channels,depth=4,batch=4,chunk='256KiB',
                        pages=hbm_pages if '24' in routes else 'base')
    # Every socket-1 DDR/HBM NUMA pair; recommended device follows controller.
    for src in [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]:
        for dst in [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]:
            add(f'route-{src}-{dst}', 'route-matrix', f'{src}:{dst}', channels=16,chunk='256KiB',
                pages=hbm_pages if src >= 24 or dst >= 24 else 'base')
    # Controlled two-way comparisons, with disjoint arrays in each direction.
    for src, dst in [(8, 9), (8, 12), (24, 25), (24, 28), (8, 24), (8, 28), (24, 12)]:
        for direction, route in [('forward',f'{src}:{dst}'), ('reverse',f'{dst}:{src}'), ('both',f'{src}:{dst},{dst}:{src}')]:
            # Same execution NUMA needs distinct controller CPUs.
            ex=[int(r.split(':')[0]) % 16 for r in route.split(',')]
            extra=[]
            if len(ex)==2 and ex[0]==ex[1]:
                extra=['--group-cpus','0|1']
            add(f'direction-{src}-{dst}-{direction}', 'bidirectional', route,
                channels=8,pages=hbm_pages if src>=24 or dst>=24 else 'base')
            cases[-1]['command']+=extra
    # Four NUMAs per die and eight NUMAs per socket, bind and page-interleave.
    for nodes, label in [([8,9,10,11],'die0'),([12,13,14,15],'die1'),(list(range(8,16)),'socket')]:
        for memory in ['ddr-ddr','ddr-hbm','hbm-ddr','hbm-hbm','hbm-ring','hbm-crossdie','ddr-crossdie']:
            routes=[]
            for i,n in enumerate(nodes):
                src=n+16 if memory.startswith('hbm') else n
                dst=n+16 if memory.endswith('hbm') else n
                if memory=='hbm-ring':dst=nodes[(i+1)%len(nodes)]+16
                if memory=='hbm-crossdie':dst=(n+4 if n<12 else n-4)+16
                if memory=='ddr-crossdie':dst=n+4 if n<12 else n-4
                routes.append(f'{src}:{dst}')
            for interleave in [False,True]:
                add(f'aggregate-{label}-{memory}-i{int(interleave)}','aggregate',','.join(routes),
                    channels=8,chunk='256KiB',interleave=interleave,pages=hbm_pages if 'hbm' in memory else 'base',seconds=1)
    # All devices over identical socket-1 memory pools, then separated HBM pools.
    for n in range(1,len(visible)+1):
        chosen=visible[:n]
        for memory in ['ddr','hbm']:
            for placement in ['same','split']:
                nodes=[8 if placement=='same' else 8+i for i in range(n)]
                routes=','.join(f'{x+16 if memory=="hbm" else x}:{x+16 if memory=="hbm" else x}' for x in nodes)
                add(f'all-devices-{n}-{memory}-{placement}','all-devices',routes,
                    '|'.join(map(str,chosen)),channels=16,chunk='256KiB',seconds=1,
                    pages=hbm_pages if memory=='hbm' else 'base',
                    exec_nodes=','.join(str(8+i) for i in range(n)))
    # Matched CPU-copy controls. They are observations from this host, not old report values.
    for routes in ['8:8','8:24','24:8','24:24','24:25','24:28']:
        for cpus in ['0','0-3','0-15','0-36']:
            add(f'cpu-{routes}-{cpus}','cpu-control',routes,cpus=cpus,
                size='4MiB',engine='cpu',pages='base',seconds=.5)
    # Equal-total-working-set controls for DMA comparisons.
    for routes in ['8:8','8:24','24:8','24:24']:
        add(f'cpu-equal-array-{routes}','cpu-equal-array',routes,size='64MiB',engine='cpu',seconds=.5)
    # Saturate both socket-local recommended devices over all eight memory NUMAs.
    for memory in ['ddr-ddr','ddr-hbm','hbm-ddr','hbm-hbm']:
        routes=','.join(f'{n+16 if memory.startswith("hbm") else n}:{n+16 if memory.endswith("hbm") else n}' for n in range(8,16))
        for channels in [16,32]:
            for n in [1,4,16]:
                for interleave in [False,True]:
                    add(f'socket-saturation-{memory}-ch{channels}-t{n}-i{int(interleave)}','socket-saturation',routes,
                        cpus='0' if n==1 else f'0-{n-1}',channels=channels,depth=4,batch=4,
                        chunk='256KiB',interleave=interleave,pages=hbm_pages if 'hbm' in memory else 'base',seconds=2)
    # Spread four visible devices across both socket-1 NUMA groups as another pool control.
    for n in range(1,len(visible)+1):
        for memory in ['ddr','hbm']:
            nodes=[8,12,9,13][:n]
            routes=','.join(f'{x+16 if memory=="hbm" else x}:{x+16 if memory=="hbm" else x}' for x in nodes)
            add(f'all-devices-{n}-{memory}-spread','all-devices-spread',routes,
                '|'.join(map(str,visible[:n])),channels=16,chunk='256KiB',seconds=1,
                pages=hbm_pages if memory=='hbm' else 'base',exec_nodes=','.join(map(str,nodes)))
    if a.phase=='quick':
        cases=[x for x in cases if x['family']=='correctness']
    metadata['script_sha256']=hashlib.sha256(SCRIPT_SOURCE).hexdigest()
    (out/'campaign-script.py').write_bytes(SCRIPT_SOURCE)
    metadata['planned_cases']=len(cases)
    (out/'plan.json').write_text(json.dumps(cases,indent=2))
    completed={r['id']:r for r in records}
    metadata['cases']=records
    def save():
        metadata['updated']=stamp()
        temporary=manifest_path.with_suffix('.new')
        temporary.write_text(json.dumps(metadata,indent=2))
        temporary.replace(manifest_path)
    for c in cases:
        if c['id'] in completed:
            if completed[c['id']]['command']!=c['command']:
                raise RuntimeError('Case command changed on resume: '+c['id'])
            retained=completed[c['id']]
            if retained['exit'] not in [0,1] or not (out/retained['csv']).exists() or sha(out/retained['csv'])!=retained['csv_sha256']:
                raise RuntimeError('Cannot resume a failed/missing/modified case: '+c['id'])
            continue
        csv_path=out/(c['id']+'.csv')
        command=c['command']+['--csv',str(csv_path)]
        print(stamp(),len(records)+1,'/',len(cases),c['id'],flush=True)
        started=time.monotonic()
        with (out/(c['id']+'.log')).open('w') as log:
            r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240,preexec_fn=child_setup)
        c['exit']=r.returncode;c['wall_seconds']=time.monotonic()-started;c['finished']=stamp()
        c['csv']=csv_path.name;c['csv_sha256']=sha(csv_path) if csv_path.exists() else None
        records.append(c);save()
        if r.returncode not in [0,1]:
            raise RuntimeError(f'{c["id"]} failed ({r.returncode}); inspect retained log')
    metadata['finished']=stamp();metadata['load_after']=read('/proc/loadavg');save()
    print('CAMPAIGN COMPLETE',len(records),flush=True)


if __name__=='__main__':
    main()
