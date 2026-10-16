#!/usr/bin/env -S uv run python
"""Validate manual benchmark requests and plan one bounded benchmark run per platform and target."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import load_catalog, resolve_selector, scope
import runner
import settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import PLATFORMS, platforms as native_platforms
from cross_build import TARGETS

OPTIONS = ('filter', 'sample_size', 'warmup_ms', 'measure_ms')
INPUTS = {'benchmarks', 'diagnostic', *OPTIONS, *PLATFORMS}
# GitHub rejects a matrix with more than 256 jobs.
MAX_MATRIX = 256


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


def dispatch_inputs(env: dict) -> dict:
    inputs = json.loads(env.get('DISPATCH_INPUTS') or '{}')
    if not isinstance(inputs, dict):
        raise ValueError('DISPATCH_INPUTS must be a JSON object')
    if set(inputs) - INPUTS:
        raise ValueError('unexpected workflow inputs: ' + ', '.join(sorted(set(inputs) - INPUTS)))
    # An omitted optional input is blank.
    inputs = {name: value for name, value in inputs.items() if value is not None}
    if any(type(inputs.get(name, False)) is not bool for name in ('diagnostic', *PLATFORMS)):
        raise ValueError('platform and diagnostic inputs must be Boolean')
    if any(type(inputs.get(name, '')) is not str for name in ('benchmarks', *OPTIONS)):
        raise ValueError('benchmarks and sampling inputs must be strings')
    return inputs


def selected_platforms(inputs: dict) -> str:
    names = [name for name in PLATFORMS if inputs.get(name)]
    if not names:
        raise ValueError('select at least one platform')
    return ','.join(names)


def selection(value: str) -> dict[str, list[str]]:
    """Group the selected target and algorithm names by the benchmark target that runs them."""
    names = value.replace(',', ' ').split()
    if not names or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        raise ValueError('benchmarks requires catalog algorithms, groups, targets, or all')
    catalog = load_catalog()
    runs: dict[str, list[str]] = {}
    for name in names:
        selected = resolve_selector(catalog, name)
        if selected is None:
            raise ValueError(f'unknown benchmark selector: {name}')
        for member in selected:
            bench, _ = scope(catalog, member)
            external = catalog['benches'][bench].get('external')
            if external:
                raise ValueError(f'{bench} needs {external}, which CI does not provision; run it locally')
            runs.setdefault(bench, [])
            if member not in runs[bench]:
                runs[bench].append(member)
    # A selected whole target already contains its algorithms.
    return {bench: [bench] if bench in members else members for bench, members in runs.items()}


def options(inputs: dict) -> list[str]:
    args = ['output_dir=target/bench', 'diag=' + str(inputs.get('diagnostic', False)).lower()]
    args += [name + '=' + inputs[name] for name in OPTIONS if inputs.get(name)]
    return args


def arguments(names: list[str], common: list[str]) -> list[str]:
    args = [*names, *common]
    parsed = runner.parse(['bench', *args])
    runner.requests(parsed, load_catalog())
    settings.load({key: getattr(parsed, key) for key in ('sample_size', 'warmup_ms', 'measure_ms')
                   if getattr(parsed, key) is not None})
    return args


def plan(inputs: dict, run_id: str) -> tuple[dict, dict, dict]:
    runs = selection(inputs.get('benchmarks', ''))
    commands = {bench: arguments(names, options(inputs)) for bench, names in runs.items()}
    rows = platforms(selected_platforms(inputs), run_id)['include']
    # Target-major order starts every platform's first runs in the first parallel wave.
    measurements = [{**row, 'bench': bench} for bench in runs for row in rows]
    builds = [{'platform': row['platform'], 'target': row['target'], 'bench': bench}
              for bench in runs for row in rows if 'target' in row]
    if max(len(measurements), len(builds)) > MAX_MATRIX:
        raise ValueError(f'the selection needs more than {MAX_MATRIX} jobs; select fewer targets or platforms')
    return {'include': measurements}, {'include': builds}, commands


def bench_arguments(inputs: dict, bench: str) -> list[str]:
    runs = selection(inputs.get('benchmarks', ''))
    if bench not in runs:
        raise ValueError(f'{bench} is not a target of this selection')
    return arguments(runs[bench], options(inputs))


def main() -> int:
    inputs = dispatch_inputs(os.environ)
    command = sys.argv[1:]
    if command == ['plan']:
        measurements, builds, commands = plan(inputs, os.environ.get('GITHUB_RUN_ID', ''))
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('measurements=' + json.dumps(measurements, separators=(',', ':')) + '\n')
            output.write('cross=' + str(bool(builds['include'])).lower() + '\n')
            output.write('builds=' + json.dumps(builds, separators=(',', ':')) + '\n')
        print('Platforms: ' + ', '.join(dict.fromkeys(row['name'] for row in measurements['include'])))
        for bench, args in commands.items():
            print(f'{bench}: just bench ' + ' '.join(args))
        print(f'Cross-builds: {len(builds["include"])}; measurements: {len(measurements["include"])}')
        return 0
    if len(command) == 2 and command[0] == 'run':
        args = bench_arguments(inputs, command[1])
        return subprocess.run(['just', 'bench', *args], check=False).returncode
    if len(command) == 4 and command[0] in ('prepare', 'measure'):
        operation, bench, target, archive = command
        if target not in TARGETS:
            raise ValueError('unsupported cross-build target')
        args = bench_arguments(inputs, bench)
        args += ['target=' + target, ('prepare_archive=' if operation == 'prepare' else 'run_archive=') + archive]
        return subprocess.run(['just', 'bench', *args], check=False).returncode
    raise ValueError('usage: scripts/bench/ci.py plan|run BENCH|{prepare|measure} BENCH TARGET ARCHIVE')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
