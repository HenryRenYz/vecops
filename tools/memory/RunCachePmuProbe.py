# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Inspect L2 allocation sensitivity between serialized benchmark campaigns.

PMU events cover the entire child process, including initialization, warmup
and teardown. They are diagnostic counters, not counters gated to CSV samples.
No other benchmark may run concurrently with this probe.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',default='128KiB');p.add_argument('--default-memory-bind',default='');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,set(range(456,608)))
    records=[]
    for placement,targets,pages in (('ddr_base','12,13,14,15','base'),('hbm_base','28,29,30,31','base'),('hbm_huge','28,29,30,31','huge')):
        for trial in range(1,4):
            stem=f'{placement}-{trial}';csv_path=args.output/(stem+'.bench.csv');pmu_path=args.output/(stem+'.perf.csv')
            command=['perf','stat','-x,','-o',str(pmu_path),'-e','cycles:u,armv8_pmuv3_0/l2d_cache/u,armv8_pmuv3_0/l2d_cache_refill/u',
                     '--',str(args.binary.resolve()),'bandwidth','--numa',targets,'--exec-node','12,13,14,15',
                     '--group-cpus','|'.join(['0-2,4-37']*4),'--op','triad','--kernel','vec4','--access','temporal',
                     '--size',args.size,'--pages',pages,'--seconds','.3','--samples','3','--warmup','.1','--retries','0','--csv',str(csv_path)]
            if args.default_memory_bind:
                start=command.index('--')+1;command[start:start]=['numactl','--membind='+args.default_memory_bind]
            (args.output/(stem+'.command.json')).write_text(json.dumps(command,indent=2))
            with (args.output/(stem+'.log')).open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            records.append(dict(placement=placement,trial=trial,exit=r.returncode,bench_csv=csv_path.name,perf_csv=pmu_path.name))
            print(json.dumps(records[-1]),flush=True)
            if r.returncode not in (0,1):raise RuntimeError('PMU probe failed; inspect the retained log')
    (args.output/'results.json').write_text(json.dumps(dict(events='User-only whole-process cycles, L2D_CACHE and L2D_CACHE_REFILL',size=args.size,default_memory_bind=args.default_memory_bind,records=records),indent=2))
    print('PMU_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
