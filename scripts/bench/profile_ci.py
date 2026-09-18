#!/usr/bin/env python3
"""Validate and dispatch one curated cross-target CI profile request."""

from __future__ import annotations

import json
import os
import re
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
  workload = env.get('INPUT_WORKLOAD', '').strip()
  preset = catalog['profile_presets'].get(workload)
  if preset is None:
    raise ValueError('workload must be one of: ' + ', '.join(catalog['profile_presets']))
  benchmark = preset['bench']
  if 'case' in preset:
    cases = [preset['case']]
  else:
    cases_by_architecture = preset['cases_by_architecture']
    if set(cases_by_architecture) != ARCHITECTURES:
      raise ValueError('architecture profile cases must cover exactly: ' + ', '.join(sorted(ARCHITECTURES)))
    cases = cases_by_architecture[architecture]
  enabled = preset['diagnostic']
  entry = runner.target(catalog, benchmark, enabled)
  raw_seconds = env.get('INPUT_SECONDS', '')
  if not re.fullmatch(r'[1-9][0-9]*', raw_seconds):
    raise ValueError('seconds must be a positive integer')
  seconds = int(raw_seconds)
  if seconds > settings.PROFILE_CAPTURE_MAX_SECONDS:
    raise ValueError(f'seconds must be at most {settings.PROFILE_CAPTURE_MAX_SECONDS}')
  row = profile_platform(architecture, run_id)
  return {
    'architecture': architecture,
    'tooling_platform': row['platform'],
    'runner': row['runner'],
    'target': row['target'],
    'workload': workload,
    'benchmark': benchmark,
    'cases': cases,
    'seconds': seconds,
    'diagnostic': enabled,
    'binary': entry['binary'],
    'features': entry['features'],
    'prepare_timeout': settings.PROFILE_PREPARE_TIMEOUT_MINUTES,
    'capture_timeout': settings.PROFILE_CAPTURE_TIMEOUT_MINUTES,
  }


def command(selection: dict, operation: str, target: str, archive: str, case: str) -> list[str]:
  if target != selection['target']:
    raise ValueError('requested target differs from the validated architecture target')
  if operation not in {'prepare', 'capture'}:
    raise ValueError('profile operation must be prepare or capture')
  if case not in selection['cases']:
    raise ValueError('profile case is not part of the validated workload')
  result = ['just', 'profile', selection['benchmark'], case, str(selection['seconds'])]
  if selection['diagnostic']:
    result.append('--diag')
  result += ['--target', target,
             '--prepare-archive' if operation == 'prepare' else '--run-archive', archive]
  return result


def main() -> int:
  selection = request(os.environ, os.environ['GITHUB_RUN_ID'])
  if sys.argv[1:] == ['plan']:
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
      output.writelines(
        f'{name}={selection[name]}\n'
        for name in ('architecture', 'tooling_platform', 'runner', 'target', 'workload',
                     'prepare_timeout', 'capture_timeout')
      )
    print('Profile request: ' + json.dumps(selection, sort_keys=True))
    return 0
  if len(sys.argv) == 4 and sys.argv[1] in {'prepare', 'capture'}:
    operation, target, archive = sys.argv[1:]
    cases = selection['cases'][:1] if operation == 'prepare' else selection['cases']
    status = 0
    for case in cases:
      result = subprocess.run(command(selection, operation, target, archive, case), check=False)
      if result.returncode and status == 0:
        status = result.returncode
    return status
  raise ValueError('usage: scripts/bench/profile_ci.py plan|{prepare|capture} TARGET ARCHIVE')


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except (ValueError, OSError) as error:
    print(f'error: {error}', file=sys.stderr)
    raise SystemExit(2)
