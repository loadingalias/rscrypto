#!/usr/bin/env python3
"""Validate and dispatch one exact cross-target CI profile request."""

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

ARCHITECTURES = {'riscv64-linux', 's390x-linux', 'powerpc64le-linux'}


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
  case = preset['case']
  enabled = preset['diagnostic']
  entry = runner.target(catalog, benchmark, enabled)
  raw_seconds = env.get('INPUT_SECONDS', '')
  if not re.fullmatch(r'[1-9][0-9]*', raw_seconds):
    raise ValueError('seconds must be a positive integer')
  seconds = int(raw_seconds)
  if seconds > settings.PROFILE_CAPTURE_MAX_SECONDS:
    raise ValueError(f'seconds must be at most {settings.PROFILE_CAPTURE_MAX_SECONDS}')
  row, = bench_ci.platforms(architecture, run_id)['include']
  return {
    'architecture': architecture,
    'runner': row['runner'],
    'target': row['target'],
    'workload': workload,
    'benchmark': benchmark,
    'case': case,
    'seconds': seconds,
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


def main() -> int:
  selection = request(os.environ, os.environ['GITHUB_RUN_ID'])
  if sys.argv[1:] == ['plan']:
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
      output.writelines(
        f'{name}={selection[name]}\n'
        for name in ('architecture', 'runner', 'target', 'workload', 'prepare_timeout', 'capture_timeout')
      )
    print('Profile request: ' + json.dumps(selection, sort_keys=True))
    return 0
  if len(sys.argv) == 4 and sys.argv[1] in {'prepare', 'capture'}:
    operation, target, archive = sys.argv[1:]
    return subprocess.run(command(selection, operation, target, archive), check=False).returncode
  raise ValueError('usage: scripts/bench/profile_ci.py plan|{prepare|capture} TARGET ARCHIVE')


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except (ValueError, OSError) as error:
    print(f'error: {error}', file=sys.stderr)
    raise SystemExit(2)
