#!/usr/bin/env python3
"""Validate and dispatch one curated cross-target CI profile request."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Embedded Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ci as bench_ci
import runner
import settings
from benchmark_catalog import load_catalog

TARGETS = {
  'aarch64-linux': ('aarch64-linux', 'aarch64-unknown-linux-gnu'),
  'powerpc64le-linux': ('powerpc64le-linux', 'powerpc64le-unknown-linux-gnu'),
  'riscv64-linux': ('riscv64-linux', 'riscv64gc-unknown-linux-gnu'),
  's390x-linux': ('s390x-linux', 's390x-unknown-linux-gnu'),
  'x86_64-linux-amd': ('x86_64-linux', 'x86_64-unknown-linux-gnu'),
  'x86_64-linux-intel': ('x86_64-linux', 'x86_64-unknown-linux-gnu'),
}
ARCHITECTURES = set(TARGETS)


def profile_platform(architecture: str, run_id: str) -> dict:
  platform, target = TARGETS[architecture]
  rows = [row for row in bench_ci.platforms(platform, run_id)['include'] if row['name'] == architecture]
  if len(rows) != 1:
    raise ValueError(f'architecture must resolve to one native profile runner: {architecture}')
  return rows[0] | {'target': target}


def request(env: dict, run_id: str) -> dict:
  architecture = env.get('INPUT_ARCHITECTURE', '').strip()
  if architecture not in ARCHITECTURES:
    raise ValueError('architecture must be one of: ' + ', '.join(sorted(ARCHITECTURES)))
  catalog = load_catalog()
  primitive = env.get('INPUT_PRIMITIVE', '').strip()
  preset = catalog['profile_presets'].get(primitive)
  if preset is None:
    raise ValueError('primitive must be one of: ' + ', '.join(catalog['profile_presets']))
  benchmark = preset['bench']
  enabled = preset['diagnostic']
  entry = runner.target(catalog, benchmark, enabled)
  row = profile_platform(architecture, run_id)
  return {
    'architecture': architecture,
    'tooling_platform': row['platform'],
    'runner': row['runner'],
    'target': row['target'],
    'primitive': primitive,
    'benchmark': benchmark,
    'case': preset['case'],
    'seconds': settings.PROFILE_CAPTURE_DEFAULT_SECONDS,
    'diagnostic': enabled,
    'binary': entry['binary'],
    'features': entry['features'],
    'prepare_timeout': settings.PROFILE_PREPARE_TIMEOUT_MINUTES,
    'capture_timeout': settings.PROFILE_CAPTURE_TIMEOUT_MINUTES,
  }


def command(selection: dict, operation: str, target: str, archive: str) -> list[str]:
  if target != selection['target']:
    raise ValueError('requested target differs from the validated architecture target')
  if operation not in {'prepare', 'capture'}:
    raise ValueError('profile operation must be prepare or capture')
  result = ['just', 'profile', selection['benchmark'], selection['case'], str(selection['seconds'])]
  if selection['diagnostic']:
    result.append('--diag')
  result += ['--target', target,
             '--prepare-archive' if operation == 'prepare' else '--run-archive', archive]
  return result


def publish_summary(selection: dict) -> None:
  reports = list(Path('target/profiles').glob('*/perf-report.txt'))
  if len(reports) != 1:
    raise ValueError('native capture must produce exactly one perf report')
  report = reports[0].read_text()
  summary = (
    f"## Profile `{selection['architecture']}` / `{selection['primitive']}`\n\n"
    f"Case: `{selection['case']}` · Capture: `{selection['seconds']}s`\n\n"
    f"```text\n{report.rstrip()}\n```\n"
  )
  print(summary)
  if output := os.environ.get('GITHUB_STEP_SUMMARY'):
    with Path(output).open('a', encoding='utf-8') as destination:
      destination.write(summary)


def main() -> int:
  selection = request(os.environ, os.environ['GITHUB_RUN_ID'])
  if sys.argv[1:] == ['plan']:
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
      output.writelines(
        f'{name}={selection[name]}\n'
        for name in ('architecture', 'tooling_platform', 'runner', 'target', 'primitive',
                     'prepare_timeout', 'capture_timeout')
      )
    print('Profile request: ' + json.dumps(selection, sort_keys=True))
    return 0
  if len(sys.argv) == 4 and sys.argv[1] in {'prepare', 'capture'}:
    operation, target, archive = sys.argv[1:]
    result = subprocess.run(command(selection, operation, target, archive), check=False)
    if result.returncode == 0 and operation == 'capture':
      publish_summary(selection)
    return result.returncode
  raise ValueError('usage: scripts/bench/profile_ci.py plan|{prepare|capture} TARGET ARCHIVE')


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except (ValueError, OSError) as error:
    print(f'error: {error}', file=sys.stderr)
    raise SystemExit(2)
