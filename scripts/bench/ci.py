#!/usr/bin/env -S uv run python
"""Validate manual benchmark requests and plan one job per platform and primitive family."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import applies, architecture, load_catalog, resolve_selector, scope
import runner
import settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import PLATFORMS, platforms as native_platforms
from cross_build import TARGETS

OPTIONS = ('filter', 'sample_size', 'warmup_ms', 'measure_ms')
INPUTS = {'benchmarks', 'diagnostic', 'quick', *OPTIONS, *PLATFORMS}
PREPARE_MINUTES = 30
# Each further run in a family job adds one bounded `just bench`.
RUN_MINUTES = 10


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
    if any(type(inputs.get(name, False)) is not bool for name in ('diagnostic', 'quick', *PLATFORMS)):
        raise ValueError('platform, diagnostic, and quick inputs must be Boolean')
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
    if inputs.get('quick'):
        args.append('quick=true')
    return args


def arguments(names: list[str], common: list[str]) -> list[str]:
    args = [*names, *common]
    parsed = runner.parse(['bench', *args])
    # Every platform job receives the same plan; each job skips the runs its architecture cannot execute.
    runner.requests(parsed, load_catalog(), every_architecture=True)
    settings.load({key: getattr(parsed, key) for key in ('sample_size', 'warmup_ms', 'measure_ms', 'quick')
                   if getattr(parsed, key) is not None})
    return args


def jobs(inputs: dict) -> dict[str, dict[str, list[str]]]:
    """Group the selected runs and their `just bench` arguments by the primitive family that runs them."""
    catalog = load_catalog()
    common = options(inputs)
    grouped: dict[str, dict[str, list[str]]] = {}
    for bench, names in selection(inputs.get('benchmarks', '')).items():
        entry = catalog['benches'][bench]
        if entry.get('split') == 'algorithms':
            # Each algorithm of a split target is its own run; no benchmark group spans two runs.
            if names == [bench]:
                names = [name for name, algorithm in catalog['algorithms'].items() if algorithm['bench'] == bench]
            runs = {name: arguments([name], common) for name in names}
        else:
            runs = {bench: arguments(names, common)}
        grouped.setdefault(entry['family'], {}).update(runs)
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
                     'timeout': row['timeout'] + RUN_MINUTES * (len(runs) - 1)}
                    for family, runs in families.items() for row in rows]
    builds = [{'platform': row['platform'], 'target': row['target'], 'family': family,
               'timeout': PREPARE_MINUTES + RUN_MINUTES * (len(runs) - 1)}
              for family, runs in families.items() for row in rows if 'target' in row]
    return {'include': measurements}, {'include': builds}, families


def run(inputs: dict, family: str, transfer=lambda name: [], arch: str | None = None) -> int:
    """Run each selected run of one family as its own `just bench`; the first failure ends the job."""
    runs = jobs(inputs).get(family)
    if runs is None:
        raise ValueError(f'{family} is not a family of this selection')
    catalog = load_catalog()
    arch = arch or architecture()
    for name, args in runs.items():
        bench, _ = scope(catalog, name)
        if not applies(catalog['benches'][bench], arch):
            print(f'{family} / {name}: skipped; it runs only on '
                  + ', '.join(catalog['benches'][bench]['architectures']), flush=True)
            continue
        args = [*args, *transfer(name)]
        print(f'{family} / {name}: just bench ' + ' '.join(args), flush=True)
        status = subprocess.run(['just', 'bench', *args], check=False).returncode
        if status:
            return status
    return 0


def evidence(source: Path, archive: Path) -> int:
    """Pack one measurement job's evidence into one file, which upload-artifact sends without zipping.

    Zipped uploads of these directories crashed upload-artifact on the RISC-V runners
    (`source.copy is not a function`). A raw file upload never enters that zip stream.
    """
    if not source.is_dir():
        raise ValueError(f'no benchmark evidence at {source}')
    with tarfile.open(archive, 'x:gz') as output:
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            # The preparation artifact keeps the transferred binaries; input/bundle.json keeps their digests.
            if relative.parts[:1] == ('criterion',) and relative.parts[2:4] == ('input', 'bin'):
                continue
            if path.is_symlink():
                raise ValueError(f'unexpected link in benchmark evidence: {relative.as_posix()}')
            if path.is_file():
                output.add(path, arcname=relative.as_posix(), recursive=False)
    return 0


def score(directory: Path, output: Path, inputs: dict) -> int:
    """Score every measurement archive of this run attempt; each archive name carries its platform label."""
    import compare
    import evidence_bundle
    catalog = load_catalog()
    families = sorted({bench['family'] for bench in catalog['benches'].values() if 'family' in bench}, key=len, reverse=True)
    suffix = f"-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    runs = []
    # download-artifact saves each raw archive inside a directory with the artifact's name.
    for archive in sorted(path for path in {*directory.glob('*.tar.gz'), *directory.glob('*/*.tar.gz')} if path.is_file()):
        name = archive.name.removesuffix('.tar.gz')
        # Measurement archives are bench-PLATFORM-FAMILY-RUN-ATTEMPT; preparation evidence is scored nowhere.
        if not (name.startswith('bench-') and name.endswith(suffix)) or name.startswith(('bench-prepare-', 'bench-score-')):
            continue
        stem = name[len('bench-'):-len(suffix)]
        family = next((family for family in families if stem.endswith('-' + family)), None)
        if family is None:
            raise ValueError(f'unexpected benchmark artifact: {name}')
        extracted = directory / name
        evidence_bundle.unpack(archive, extracted)
        runs += [f'{stem[:-len(family) - 1]}={status.parent}' for status in sorted(extracted.glob('criterion/*/status.txt'))]
    if not runs:
        raise ValueError('no measurement artifacts to score')
    # A case filter can cut comparison groups, so its missing points are notes rather than errors.
    return compare.main([*runs, '--json', str(output), *(['--partial'] if inputs.get('filter') else [])])


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
            for name, args in runs.items():
                print(f'{family} / {name}: just bench ' + ' '.join(args))
        print(f'Cross-builds: {len(builds["include"])}; measurements: {len(measurements["include"])}')
        return 0
    if len(command) == 2 and command[0] == 'run':
        return run(inputs, command[1])
    if len(command) == 3 and command[0] == 'score':
        return score(Path(command[1]), Path(command[2]), inputs)
    if len(command) == 3 and command[0] == 'evidence':
        return evidence(Path(command[1]), Path(command[2]))
    if len(command) == 4 and command[0] in ('prepare', 'measure'):
        operation, family, target, directory = command
        if target not in TARGETS:
            raise ValueError('unsupported cross-build target')
        # One sealed archive per run, named for its target or algorithm, in the family's transfer directory.
        directory = Path(directory)
        flag = 'prepare_archive=' if operation == 'prepare' else 'run_archive='
        if operation == 'prepare':
            directory.mkdir(parents=True, exist_ok=True)
        return run(inputs, family, lambda name: ['target=' + target, flag + str(directory / f'{name}.tar.gz')],
                   architecture(target))
    raise ValueError('usage: scripts/bench/ci.py plan|run FAMILY|{prepare|measure} FAMILY TARGET DIRECTORY'
                     '|score ARTIFACTS OUTPUT|evidence SOURCE ARCHIVE')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
