# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Retain the measured ELF, build settings and project compiler inputs.

Run between measurements. Dependency mtimes are compared with object mtimes;
any source edited after compilation is explicitly recorded, not silently
presented as the original compiled input. Upstream notices/licenses remain
unchanged in the copied project inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    project=Path(__file__).resolve().parents[2];args.output.mkdir(parents=True,exist_ok=True)
    build=args.build.resolve();target=build/'tools/memory/CMakeFiles/vecops-memory-bench.dir'
    dependencies={};changes=[]
    for depfile in target.glob('*.o.d'):
        obj=depfile.with_suffix('')
        parts=shlex.split(depfile.read_text().replace('\\\n',''))[1:]
        for name in parts:
            path=Path(name)
            if not path.is_absolute():path=build/path
            path=path.resolve()
            if not path.is_relative_to(project) or not path.is_file():continue
            relative=str(path.relative_to(project))
            dependencies[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_mtime_ns>obj.stat().st_mtime_ns:changes.append(dict(source=relative,object=str(obj.relative_to(build))))
            dest=args.output/'compiler-inputs'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ('flags.make','link.txt','build.make'):
        if (target/name).exists():shutil.copy2(target/name,args.output/name)
    shutil.copy2(build/'CMakeCache.txt',args.output/'CMakeCache.txt')
    binary=build/'tools/memory/vecops-memory-bench';shutil.copy2(binary,args.output/binary.name)
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
        if (project/name).exists():shutil.copy2(project/name,args.output/name)
    if (project/'LICENSES').is_dir():shutil.copytree(project/'LICENSES',args.output/'LICENSES',dirs_exist_ok=True)
    result=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),project_dependencies=dependencies,
                sources_modified_after_object=changes,build_dir=str(build),compiler_input_count=len(dependencies))
    (args.output/'build-inputs.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(project_inputs=len(dependencies),edited_after_compile=len(changes),binary_sha256=result['binary_sha256']),indent=2))


if __name__=='__main__':main()
