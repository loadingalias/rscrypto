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

# Benchmark hardware is fixed where the provider allows it. Donated runners
# retain their provider labels; their actual machine identity is recorded by bench.
PLATFORMS = {
    'x86_64-linux': ('c8i.2xlarge', 'ubuntu24-minimal-x64', 90),
    'aarch64-linux': ('c8g.2xlarge', 'ubuntu24-minimal-arm64', 90),
    'x86_64-win': ('c8i.2xlarge', 'windows25-full-x64', 90),
    's390x-linux': ('', 'ubuntu-24.04-s390x', 90),
    'powerpc64le-linux': ('', 'ubuntu-24.04-ppc64le-p10', 90),
    'riscv64-linux': ('', 'ubuntu-24.04-riscv', 180),
}


def platforms(value: str, run_id: str) -> dict:
    names = list(PLATFORMS) if value.strip() == 'all' else list(dict.fromkeys(value.replace(',', ' ').split()))
    if not names or any(name not in PLATFORMS for name in names):
        raise ValueError('architectures must be all or a list of: ' + ', '.join(PLATFORMS))
    if not run_id.isdecimal():
        raise ValueError('GITHUB_RUN_ID must be numeric')
    rows = []
    for name in names:
        family, image, timeout = PLATFORMS[name]
        label = (f'runs-on={run_id}/family={family}/cpu=8/image={image}/spot=false/volume=100gb:gp3/env=production'
                 if family else image)
        rows.append({'platform': name, 'runner': label, 'timeout': timeout})
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
        print('Platforms: ' + ', '.join(row['platform'] for row in matrix['include']))
        print('Benchmark arguments: ' + json.dumps(args))
        return 0
    if sys.argv[1:] == ['run']:
        return subprocess.run(['just', 'bench', *args], check=False).returncode
    raise ValueError('usage: scripts/bench/ci.py plan|run')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
