#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build all resource-runtime contracts with strict diagnostics and ASan/UBSan."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
output = root / 'build/memory-migration'
objects = output / 'runtime-sanitized-objects'
objects.mkdir(parents=True, exist_ok=True)
sources = sorted((root / 'src/memory').glob('*.cpp'))
tests = sorted((root / 'tests/memory').glob('*Test.cpp'))
commands = []
for source in [*sources, *tests]:
    relative = source.relative_to(root)
    target = objects / (str(relative).replace('/', '_') + '.o')
    command = ['g++', '-std=c++20', '-g', '-O0', '-fno-omit-frame-pointer',
               '-fsanitize=address,undefined', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
               '-Iinclude', '-isystem', 'build/_deps/googletest-src/googletest/include',
               '-c', str(relative), '-o', str(target.relative_to(root))]
    commands.append(command)
link = ['g++', '-fsanitize=address,undefined', '-g',
        *[command[-1] for command in commands],
        'build/memory-migration-cmake/lib/libgtest.a', 'build/memory-migration-cmake/lib/libgtest_main.a',
        '-pthread', '-lrt', '-o', 'build/memory-migration/runtime-sanitized-test']
(output / 'runtime-sanitized-command.json').write_text(json.dumps([*commands, link], indent=2))


def compile_one(command):
    with (root / (command[-1] + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'compilation failed; see {command[-1]}.log')


with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(compile_one, commands))
with (output / 'runtime-sanitized-link.log').open('w') as log:
    subprocess.run(link, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
with (output / 'runtime-sanitized-tests.log').open('w') as log:
    subprocess.run([str(output / 'runtime-sanitized-test'),
        '--gtest_output=json:build/memory-migration/runtime-sanitized-tests.json'],
        cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('ASan/UBSan runtime tests passed', flush=True)
