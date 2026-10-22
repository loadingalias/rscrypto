#!/usr/bin/env -S uv run python
"""Select native CT evidence lanes, and decide a run from their reports."""
import json
import os
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms
from cross_build import TARGETS
from manifest import primitive_supports_physical_timing, replay_cases
from provenance import load_toml


def main():
    if sys.argv[1:] == ['plan']:
        case_name = os.environ.get('DIAGNOSTIC_CASE', '')
        selected_primitives = []
        if case_name:
            ct = load_toml(Path(__file__).resolve().parents[2] / 'ct.toml')
            cases = {row['name']: row for row in ct['dudect_case']}
            primitives = {row['id']: row for row in ct['primitive']}
            selected_primitives = [primitives[cases[name]['primitive']]
                                   for name in replay_cases(cases, case_name)]
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'], runner_prefix='ct')
        for row in matrix['include']:
            row['timeout'] = 360
        builds = []
        for row in matrix['include']:
            target = row['platform'].removesuffix('-linux') + '-unknown-linux-gnu'
            if row['platform'] == 'riscv64-linux':
                target = 'riscv64gc-unknown-linux-gnu'
            if case_name:
                if not row['platform'].endswith('-linux'):
                    raise ValueError('CT diagnostic cases run on Linux platforms')
                if any(not primitive_supports_physical_timing(primitive, target)
                       for primitive in selected_primitives):
                    raise ValueError(f'CT diagnostic case {case_name} does not support {target}')
                if target not in TARGETS:
                    # Native rows measure through ct-full; cross rows replay sealed archives.
                    row['diagnostic_cases'] = ' '.join(replay_cases(cases, case_name, target))
            if target in TARGETS:
                row['target'] = target
                builds.append({'target': target})
        # RISC-V executes in a dispatched run that the workflow skips when no runner starts it.
        riscv = [row for row in matrix['include'] if row['platform'] == 'riscv64-linux']
        matrix['include'] = [row for row in matrix['include'] if row['platform'] != 'riscv64-linux']
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
            output.write('native=' + str(bool(matrix['include'])).lower() + '\n')
            output.write('riscv=' + (json.dumps(riscv[0], separators=(',', ':')) if riscv else '') + '\n')
            output.write('cross=' + str(bool(builds)).lower() + '\n')
            output.write('builds=' + json.dumps({'include': builds}, separators=(',', ':')) + '\n')
        return
    if sys.argv[1:2] == ['verdict'] and len(sys.argv) == 3:
        matrix = json.loads(os.environ['MATRIX'])
        platforms_run = [row['platform'] for row in matrix['include']]
        status = verdict(Path(sys.argv[2]), platforms_run, os.environ.get('RISCV_RESULT', ''))
        raise SystemExit(status)
    raise ValueError('usage: scripts/ct/ci.py plan | verdict DIRECTORY')


def verdict(directory, platforms_run, riscv_result):
    """Fail the run unless every platform's retained report passed.

    Matrix jobs fail only when their evidence is incomplete; a report with findings
    still uploads, so the run's verdict is decided here from every report at once.
    """
    rows = []
    for platform in platforms_run:
        archive = directory / f'ct-{platform}-full' / f'ct-{platform}-full.tar.gz'
        if not archive.exists():
            rows.append((platform, 'missing', 'no retained evidence'))
            continue
        with tarfile.open(archive) as bundle:
            reports = [member for member in bundle.getmembers()
                       if member.name.endswith('/release/ct-report.json') and member.name.startswith('ct/')]
            if len(reports) != 1:
                rows.append((platform, 'missing', f'{len(reports)} reports in the archive'))
                continue
            report = json.load(bundle.extractfile(reports[0]))
        rows.append((platform, report['status'], f"{report['failure_count']} failure(s); evidence complete: "
                                                   f"{report.get('evidence_complete', False)}"))
    if riscv_result:
        # The RISC-V run reports through its dispatching job; skipped means no runner started it.
        rows.append(('riscv64-linux', {'success': 'pass', 'skipped': 'skipped'}.get(riscv_result, 'fail'),
                     f'dispatched run: {riscv_result}'))
    table = ['| Platform | Verdict | Detail |', '| --- | --- | --- |', *(f'| {a} | {b} | {c} |' for a, b, c in rows)]
    print('\n'.join(table))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as summary:
            summary.write('## Constant-time verdict\n\n' + '\n'.join(table) + '\n')
    return 0 if all(state in ('pass', 'skipped') for _, state, _ in rows) else 1


if __name__ == '__main__':
    main()
