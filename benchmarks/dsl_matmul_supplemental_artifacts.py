# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact supplemental source/ABI/capture executable binding, without compilation."""
import hashlib
import json
import re
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True)


def artifact_inventory(directory):
    import vecops
    from dsl_matmul_artifact_ab import adapter_contract
    records = []
    for adapter in sorted((directory / 'build').glob('kernel-batch-*/tasks/*/input/vecops_adapter.cpp')):
        text = adapter.read_text()
        source = re.search(r'#include "([^"\n]+\.cpp)"', text)
        if source is None:
            continue
        spec = adapter.with_name('vecops_spec.h').read_text()
        captures = {name: int(value) for name, value in re.findall(
            r'inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;', spec)}
        captures.update(re.findall(r'using (\w+) = ::vecops::(\w+);', spec))
        for binary in (adapter.parents[3] / 'artifacts' / adapter.parents[1].name).glob('*.so'):
            records.append((Path(source.group(1)).resolve(), captures, adapter_contract(text),
                            vecops._C.Executable.load(str(binary)), str(binary.resolve()),
                            hashlib.sha256(binary.read_bytes()).hexdigest()))
    return records

def bind_artifact(template, source, records):
    import vecops
    from dsl_matmul_artifact_ab import template_contract
    contract, captures = template_contract(template)
    matches = [record for record in records if record[0] == Path(source).resolve()
               and canonical(record[2]) == canonical(contract)
               and all(record[1].get(name) == value for name, value in captures.items())]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one exact source/ABI/capture artifact, found {len(matches)}: {source}')
    executable = matches[0][3]
    access = [parameter[7] for parameter in contract[0] if parameter[0] == 'TENSOR']
    def invoke(values, repeat):
        arguments = vecops._C.ArgumentMetadata([
            vecops._C.TensorView.from_numpy(value, writable=bool(mode & 2))
            for value, mode in zip(values, access)] + [repeat])
        if not executable.can_invoke(arguments):
            raise RuntimeError('Exact artifact rejects invocation metadata')
        executable.invoke(arguments)
    return invoke, {'binary': matches[0][4], 'binary_sha256': matches[0][5],
                    'source': str(Path(source).resolve()), 'contract': contract, 'captures': captures}
