# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Serially rerun undersampled cases, retaining the original skew threshold.

A short repeat first addresses transient OS disturbances. Persistent multi-core
completion tails receive a longer common window instead of a looser skew
threshold. Reruns always live in distinct directories; the report prioritizes
the prescribed last rerun, never whichever run has the largest bandwidth.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from RunSocketCampaign import make_cases
from SummarizeSocketCampaign import summarize


def poor_cases(root):
    data=summarize(root)
    chosen={}
    for row in data['summaries']:
        if row['scope'] not in ('total','chase:g0','round-trip'):continue
        key=(row['case_id'],row['scope'])
        rank=(row['run'].startswith('repair'),row['run']=='scaling',row['run'])
        if key not in chosen or rank>chosen[key][0]:chosen[key]=(rank,row)
    # Full-core runs require a separate fresh availability check. A reduced
    # rerun must never replace a 304-core case under the same identity.
    known={c['id'] for c in make_cases() if c['family']!='full'}
    return sorted({row['case_id'] for _,row in chosen.values() if row['samples_usable']<3 and row['case_id'] in known})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--wait-pid',type=int);p.add_argument('--wait-log',type=Path);p.add_argument('--wait-marker',default='FOLLOWUP_COMPLETE');p.add_argument('--default-memory-bind',default='');args=p.parse_args()
    os.sched_setaffinity(0,set(range(304,608))-{306,307,308,309})
    if args.wait_pid:
        while True:
            try:status=Path(f'/proc/{args.wait_pid}/status').read_text()
            except FileNotFoundError:break
            if 'State:\tZ' in status:break
            time.sleep(5)
        if args.wait_log and args.wait_marker not in args.wait_log.read_text():raise RuntimeError('Predecessor failed; inspect its state rather than overlapping a retry')
    for directory,seconds,samples in (('repair-01-short',.5,5),('repair-02-long',20,3)):
        ids=poor_cases(args.root)
        selection=args.root/(directory+'-ids.json');selection.write_text(json.dumps(ids,indent=2))
        print(json.dumps(dict(phase=directory,cases=len(ids),seconds=seconds,samples=samples)),flush=True)
        if not ids:continue
        subprocess.run([sys.executable,'-u','tools/memory/RunSocketCampaign.py','--binary',str(args.binary),
                        '--output',str(args.root/directory),'--phase','all','--case-ids',str(selection),
                        '--samples',str(samples),'--seconds',str(seconds),'--retries','0','--exclude-cpus','306-309',
                        '--default-memory-bind',args.default_memory_bind],check=True)
    print('REPAIR_COMPLETE',flush=True)


if __name__=='__main__':main()
