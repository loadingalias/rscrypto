#!/usr/bin/env -S uv run python
"""Validate manual benchmark requests and plan one job per platform and primitive family."""
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
PREPARE_MINUTES = 30
# Each further target in a family job adds one bounded `just bench` run.
TARGET_MINUTES = 10


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


def jobs(inputs: dict) -> dict[str, dict[str, list[str]]]:
    """Group the selected targets and their `just bench` arguments by the primitive family that runs them."""
    benches = load_catalog()['benches']
    common = options(inputs)
    grouped: dict[str, dict[str, list[str]]] = {}
    for bench, names in selection(inputs.get('benchmarks', '')).items():
        grouped.setdefault(benches[bench]['family'], {})[bench] = arguments(names, common)
    return grouped


def label(runner: str, family: str) -> str:
    # Donated runners keep their provider labels.
    return runner.replace('/', f'-{family}/', 1) if runner.startswith('runs-on=') else runner


def plan(inputs: dict, run_id: str) -> tuple[dict, dict, dict]:
    families = jobs(inputs)
    rows = platforms(selected_platforms(inputs), run_id)['include']
    # RunsOn starts a host for every queued job, whatever `max-parallel` says, and a job can take any
    # host whose label matches. One job per family bounds the hosts; the family label keeps each job on its own.
    measurements = [{**row, 'family': family, 'runner': label(row['runner'], family),
                     'timeout': row['timeout'] + TARGET_MINUTES * (len(runs) - 1)}
                    for family, runs in families.items() for row in rows]
    builds = [{'platform': row['platform'], 'target': row['target'], 'family': family,
               'timeout': PREPARE_MINUTES + TARGET_MINUTES * (len(runs) - 1)}
              for family, runs in families.items() for row in rows if 'target' in row]
    return {'include': measurements}, {'include': builds}, families


def run(inputs: dict, family: str, transfer=lambda bench: []) -> int:
    """Run each selected target of one family as its own `just bench`; the first failure ends the job."""
    runs = jobs(inputs).get(family)
    if runs is None:
        raise ValueError(f'{family} is not a family of this selection')
    for bench, args in runs.items():
        args = [*args, *transfer(bench)]
        print(f'{family} / {bench}: just bench ' + ' '.join(args), flush=True)
        status = subprocess.run(['just', 'bench', *args], check=False).returncode
        if status:
            return status
    return 0


def main() -> int:
    inputs = dispatch_inputs(os.environ)
    command = sys.argv[1:]
    if command == ['plan']:
        measurements, builds, families = plan(inputs, os.environ.get('GITHUB_RUN_ID', ''))
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('request=' + json.dumps(inputs, separators=(',', ':')) + '\n')
            output.write('measurements=' + json.dumps(measurements, separators=(',', ':')) + '\n')
            output.write('cross=' + str(bool(builds['include'])).lower() + '\n')
            output.write('builds=' + json.dumps(builds, separators=(',', ':')) + '\n')
        print('Platforms: ' + ', '.join(dict.fromkeys(row['name'] for row in measurements['include'])))
        for family, runs in families.items():
            for bench, args in runs.items():
                print(f'{family} / {bench}: just bench ' + ' '.join(args))
        print(f'Cross-builds: {len(builds["include"])}; measurements: {len(measurements["include"])}')
        return 0
    if len(command) == 2 and command[0] == 'run':
        return run(inputs, command[1])
    if len(command) == 4 and command[0] in ('prepare', 'measure'):
        operation, family, target, directory = command
        if target not in TARGETS:
            raise ValueError('unsupported cross-build target')
        # One sealed archive per target, named for it, in the family's transfer directory.
        directory = Path(directory)
        flag = 'prepare_archive=' if operation == 'prepare' else 'run_archive='
        if operation == 'prepare':
            directory.mkdir(parents=True, exist_ok=True)
        return run(inputs, family, lambda bench: ['target=' + target, flag + str(directory / f'{bench}.tar.gz')])
    raise ValueError('usage: scripts/bench/ci.py plan|run FAMILY|{prepare|measure} FAMILY TARGET DIRECTORY')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
