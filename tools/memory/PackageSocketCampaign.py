# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Package retained measurements, figures, tool sources and verification logs.

The artifact manifest hashes every payload file. A dedicated license sidecar
lists exact paths of generated data/figures whose formats lack comments.
The measured remote executable and its build metadata should be copied into
provenance/remote before packaging; this script never mutates remote state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('Archive output must be outside the campaign directory')
    source=Path(__file__).resolve().parents[2]
    provenance=args.root/'provenance';tool=provenance/'tools'/'memory';tool.mkdir(parents=True,exist_ok=True)
    for path in (source/'tools'/'memory').iterdir():
        if path.name!='MemoryInfo.cpp' and path.is_file() and path.suffix in ('.h','.cpp','.py','.md','.txt'):shutil.copy2(path,tool/path.name)
    shutil.copy2(source/'LICENSE',provenance/'LICENSE')
    build=source/'build'/'memory-bench'
    local=provenance/'local';local.mkdir(exist_ok=True)
    for name in ('assembly-check.log','validation.log','smoke-campaign-final.log','group-cpus.csv'):
        path=build/name
        if path.exists():shutil.copy2(path,local/name)
    for name in ('SHA256SUMS.json','ARTIFACTS.license'):
        path=args.root/name
        if path.exists():path.unlink()
    payload=sorted(p for p in args.root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    uncommented=['SHA256SUMS.json']+[str(p.relative_to(args.root)) for p in payload if (p.suffix in ('.csv','.json','.jsonl','.png','.log','.pid') or p.name in ('vecops-memory-bench','CMakeCache.txt','flags.make','link.txt','build.make'))]
    (args.root/'ARTIFACTS.license').write_text('Files:\n'+''.join('  '+name+'\n' for name in uncommented)+
        '\nSPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\nSPDX-License-Identifier: MIT\n\n'
        'Scope: project-owned measurement records, generated figures and metadata listed above.\n'
        'Source files retain their own notices; the original license text is in provenance/LICENSE.\n')
    payload.append(args.root/'ARTIFACTS.license')
    manifest={str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
    (args.root/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    payload.append(args.root/'SHA256SUMS.json')
    with tarfile.open(args.output,'w:gz',compresslevel=6) as archive:
        for path in payload:archive.add(path,arcname=args.root.name+'/'+str(path.relative_to(args.root)),recursive=False)
    args.output.with_name(args.output.name+'.license').write_text(
        'SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\n'
        'SPDX-License-Identifier: MIT\n\n'
        'Scope: project-owned archive composition and generated measurements.\n'
        'Preserved third-party compiler inputs retain their original notices and licenses inside the archive.\n')
    print(json.dumps(dict(archive=str(args.output),bytes=args.output.stat().st_size,files=len(payload),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
