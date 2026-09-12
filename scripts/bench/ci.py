#!/usr/bin/env python3
"""Validate manual benchmark requests and dispatch the existing benchmark runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import load_catalog
import runner
import settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms as native_platforms
from cross_build import TARGETS


def platforms(value: str, run_id: str) -> dict:
    rows = []
    for row in native_platforms(value, run_id)['include']:
        if row['platform'].startswith('x86_64-'):
            for vendor in ('intel', 'amd'):
                rows.append({**row, 'name': row['platform'] + '-' + vendor,
                             'runner': row['runner'].replace('-intel/', '-' + vendor + '/')})
        else:
            rows.append({**row, 'name': row['platform']})
    for row in rows:
        row['timeout'] = max(row['timeout'], 120)
        target = row['platform'].removesuffix('-linux') + '-unknown-linux-gnu'
        if row['platform'] == 'riscv64-linux':
            target = 'riscv64gc-unknown-linux-gnu'
        if target in TARGETS:
            row['target'] = target
    return {'include': rows}


def arguments(env: dict) -> list[str]:
    selection = env.get('INPUT_SELECTION', '').strip()
    if selection.startswith('bench='):
        names = selection.removeprefix('bench=').split(',')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name.strip()) for name in names):
            raise ValueError('bench= requires comma-separated catalog target names')
        args = ['bench=' + ','.join(name.strip() for name in names)]
    else:
        args = selection.replace(',', ' ').split()
        if not args or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in args):
            raise ValueError('selection requires catalog algorithms/groups, all, or bench=<targets>')
    args += ['output_dir=target/bench', 'diag=' + env.get('INPUT_DIAGNOSTIC', 'false')]
    for name in ('filter', 'sample_size', 'warmup_ms', 'measure_ms'):
        value = env.get('INPUT_' + name.upper(), '')
        if value:
            args.append(name + '=' + value)
    parsed = runner.parse(['bench', *args])
    runner.requests(parsed, load_catalog())
    settings.load({key: getattr(parsed, key) for key in ('sample_size', 'warmup_ms', 'measure_ms')
                   if getattr(parsed, key) is not None})
    return args


def main() -> int:
    args = arguments(os.environ)
    if sys.argv[1:] == ['plan']:
        matrix = platforms(os.environ.get('INPUT_ARCHITECTURES', ''), os.environ['GITHUB_RUN_ID'])
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
            builds = [{'target': row['target']} for row in matrix['include'] if 'target' in row]
            output.write('cross=' + str(bool(builds)).lower() + '\n')
            output.write('builds=' + json.dumps({'include': builds}, separators=(',', ':')) + '\n')
        print('Platforms: ' + ', '.join(row['name'] for row in matrix['include']))
        print('Benchmark arguments: ' + json.dumps(args))
        return 0
    if len(sys.argv) == 4 and sys.argv[1] in ('prepare', 'measure'):
        operation, target, archive = sys.argv[1:]
        if target not in TARGETS:
            raise ValueError('unsupported cross-build target')
        args += ['target=' + target, ('prepare_archive=' if operation == 'prepare' else 'run_archive=') + archive]
        return subprocess.run(['just', 'bench', *args], check=False).returncode
    if sys.argv[1:] == ['run']:
        return subprocess.run(['just', 'bench', *args], check=False).returncode
    raise ValueError('usage: scripts/bench/ci.py plan|run|{prepare|measure} TARGET ARCHIVE')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
