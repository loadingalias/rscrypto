#!/usr/bin/env python3
"""Verify Nextest forwarding, repository policy, and fail-closed planning."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/test/test.sh', 'scripts/lib/python.sh', 'scripts/lib/rail-plan.sh', 'Cargo.toml'):
      path = root / name
      path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, path)
    toolchain = root / 'scripts/lib/toolchain.sh'
    toolchain.write_text('#!/bin/sh\necho fixture-toolchain\n')
    toolchain.chmod(0o755)
    binary = root / 'bin'
    binary.mkdir()
    for name in ('bash', 'cat', 'dirname', 'jq', 'mktemp', 'rm'):
      (binary / name).symlink_to(shutil.which(name))
    cargo = binary / 'cargo'
    cargo.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ['TEST_LOG'], 'a') as log:
  log.write(json.dumps({'args': args, 'threads': os.environ.get('NEXTEST_TEST_THREADS')}) + '\\n')
if args[:2] == ['rail', 'plan']:
  if os.environ.get('PLAN_FAIL'): sys.exit(23)
  if '--json' in args:
    state = os.environ.get('PLAN_STATE', 'required')
    row = {'state': state, 'scope': {'kind': 'cargo', 'selection': {'kind': 'packages', 'cargo_args': ['-p', 'rscrypto']}}}
    print(json.dumps({'plan_contract_version': 8, 'identity': 'plan-v8:sha256:fixture', 'required': [],
                      'work': {'cargo.test': row, 'cargo.doctest': row}}))
else:
  sys.exit(int(os.environ.get('RUN_EXIT', '0')))
''')
    cargo.chmod(0o755)
    nextest = binary / 'cargo-nextest'
    nextest.symlink_to(cargo)
    log = root / 'log.jsonl'
    env = {key: value for key, value in os.environ.items()
           if key not in ('BASH_ENV', 'ENV') and not key.startswith(('BASH_FUNC_', 'RSCRYPTO_', 'RAIL_', 'NEXTEST_'))}
    env.update(PATH=str(binary), PYTHON=sys.executable, TEST_LOG=str(log))
    bash = shutil.which('bash')

    def run(args, **extra):
      log.write_text('')
      result = subprocess.run([bash, str(root / 'scripts/test/test.sh'), *args], cwd=root,
                              env={**env, **extra}, capture_output=True, text=True, timeout=20)
      rows = [json.loads(line) for line in log.read_text().splitlines()]
      return result, rows

    forwarded = ['--release', '--no-run', '--test', 'two words', '-E', r'test(/foo|bar/)',
                 '--test-threads', '3', '--', '--skip', "quote'and\"space", '--exact', 'test name']
    result, rows = run(['--portable', '--', *forwarded], RSCRYPTO_TEST_THREADS='1')
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1 and rows[0]['args'][:2] == ['nextest', 'run'], rows
    assert rows[0]['args'][-len(forwarded):] == forwarded, rows
    assert '--all-features' in rows[0]['args'] and rows[0]['threads'] == '1'
    assert not any(row['args'][0] == 'test' for row in rows)
    for args in (['--release'], ['--no-run'], ['--lib', 'name'], ['--', '--', '--skip', 'slow'], ['--', '']):
      result, rows = run(args)
      assert result.returncode == 0 and len(rows) == 1, (args, result.stderr, rows)
      expected = args[1:] if args[0] == '--' else args
      assert rows[0]['args'][-len(expected):] == expected, rows
      features = rows[0]['args'][rows[0]['args'].index('--features') + 1].split(',')
      assert 'portable-only' not in features
    result, rows = run([])
    assert result.returncode == 0, result.stderr
    assert len([row for row in rows if row['args'][:2] == ['rail', 'plan']]) == 2
    runners = [row['args'] for row in rows if row['args'][0] != 'rail']
    assert len(runners) == 2 and runners[0][:2] == ['nextest', 'run'] and '--doc' in runners[1]
    assert all('-p' in args and 'rscrypto' in args for args in runners)
    result, rows = run(['--all'], RSCRYPTO_TEST_THREADS='1')
    assert result.returncode == 0 and len(rows) == 2
    assert all('--workspace' in row['args'] for row in rows)
    assert rows[0]['threads'] == '1'
    result, rows = run([], PLAN_STATE='skipped')
    assert result.returncode == 0 and all(row['args'][0] == 'rail' for row in rows)
    result, rows = run([], PLAN_FAIL='1')
    assert result.returncode != 0 and all(row['args'][0] == 'rail' for row in rows)
    result, rows = run(['--all'], RUN_EXIT='31')
    assert result.returncode == 31 and len(rows) == 1
    result, rows = run(['--all'], RSCRYPTO_SKIP_DOCTESTS='1')
    assert result.returncode == 0 and len(rows) == 1
    nextest.unlink()
    result, rows = run(['--all'])
    assert result.returncode == 127 and not rows
    assert 'cargo-nextest is required' in result.stderr
  print('Test runner regressions passed')


if __name__ == '__main__':
  main()
