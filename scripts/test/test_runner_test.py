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


def check_evidence_recipe():
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('justfile', 'scripts/lib/python.sh', 'scripts/ct/internal.py',
                 'scripts/ct/provenance.py', 'scripts/test/evidence_suite.py'):
      destination = root / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, destination)
    binary = root / 'bin'
    binary.mkdir()
    fake = binary / 'just'
    fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
with open(os.environ['TEST_LOG'], 'a') as log:
  print(json.dumps({'args': sys.argv[1:], 'flags': os.environ['CARGO_ENCODED_RUSTFLAGS']}), file=log)
if not os.environ.get('NO_BACKEND_EVIDENCE'):
  portable = '--portable' in sys.argv
  unavailable = bool(os.environ.get('BACKEND_UNAVAILABLE')) and not portable
  zero_execution = bool(os.environ.get('BACKEND_ZERO_EXECUTION')) and not portable
  for test in ('counter-zero', 'arbitrary-counters', 'self-inverse'):
    print('RSCRYPTO_BACKEND_EVIDENCE=' + json.dumps({
      'schema': 1,
      'kind': 'rscrypto.backend-execution',
      'primitive': 'chacha20',
      'test': test,
      'dispatch': 'portable-only' if portable else 'production-auto',
      'target_arch': 'fixture',
      'compiled': [{'id': 'fixture/backend', 'required_features': ['fixture'],
                    'runtime_available': None if portable else not unavailable}],
      'executed_backend_ids': [] if portable or unavailable or zero_execution else ['fixture/backend'],
      'executed_case_count': 0 if portable or unavailable or zero_execution else 1,
      'kernel_call_count': 0 if portable or unavailable or zero_execution else 1,
      'result': 'not-selected' if portable else ('unavailable' if unavailable or zero_execution else 'pass'),
    }))
sys.exit(int(os.environ.get('RUN_EXIT', '0')))
''')
    fake.chmod(0o755)
    log = root / 'commands.jsonl'
    flags = '-C\x1flink-arg=path with spaces'
    env = {**{key: value for key, value in os.environ.items() if key not in ('BASH_ENV', 'ENV')},
           'PATH': str(binary) + os.pathsep + os.environ['PATH'],
           'PYTHON': sys.executable, 'TEST_LOG': str(log), 'CARGO_BUILD_TARGET': 's390x-unknown-linux-gnu',
           'CARGO_ENCODED_RUSTFLAGS': flags}
    for status, count in ((0, 2), (17, 1)):
      log.write_text('')
      result = subprocess.run([shutil.which('just'), '--justfile', str(root / 'justfile'), 'test-evidence'],
                              cwd=root, env={**env, 'RUN_EXIT': str(status)}, capture_output=True, text=True)
      assert (result.returncode == 0) == (status == 0), result.stderr
      rows = [json.loads(line) for line in log.read_text().splitlines()]
      assert len(rows) == count, rows
      assert '--native' in rows[0]['args']
      if status == 0:
        assert '--portable' in rows[1]['args']
      assert all(row['flags'] == flags + '\x1f--cfg\x1frscrypto_internal' for row in rows), rows
    log.write_text('')
    result = subprocess.run([shutil.which('just'), '--justfile', str(root / 'justfile'), 'test-evidence'],
                            cwd=root, env={**env, 'BACKEND_UNAVAILABLE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert len(log.read_text().splitlines()) == 2
    for extra in ({'NO_BACKEND_EVIDENCE': '1'}, {'BACKEND_ZERO_EXECUTION': '1'}):
      log.write_text('')
      result = subprocess.run([shutil.which('just'), '--justfile', str(root / 'justfile'), 'test-evidence'],
                              cwd=root, env={**env, **extra}, capture_output=True, text=True)
      assert result.returncode != 0, (extra, result.stdout, result.stderr)
      assert len(log.read_text().splitlines()) == 1


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
    print(json.dumps({'plan_contract_version': 9, 'identity': 'plan-v9:sha256:fixture', 'required': [],
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
    for args in (['--no-run'], ['--lib', 'name'], ['--', '--', '--skip', 'slow'], ['--', '']):
      result, rows = run(args)
      assert result.returncode == 0 and len(rows) == 1, (args, result.stderr, rows)
      expected = args[1:] if args[0] == '--' else args
      assert rows[0]['args'][-len(expected):] == expected, rows
      features = rows[0]['args'][rows[0]['args'].index('--features') + 1].split(',')
      assert 'portable-only' not in features
    result, rows = run([])
    assert result.returncode == 0, result.stderr
    plan_rows = [row for row in rows if row['args'][:2] == ['rail', 'plan']]
    assert len(plan_rows) == 2 and plan_rows[1]['args'][-2:] == ['--verify', '-'], plan_rows
    runners = [row['args'] for row in rows if row['args'][0] != 'rail']
    assert len(runners) == 2 and runners[0][:2] == ['nextest', 'run'] and '--doc' in runners[1]
    assert all('-p' in args and 'rscrypto' in args for args in runners)
    result, rows = run(['--all'], RSCRYPTO_TEST_THREADS='1')
    assert result.returncode == 0 and len(rows) == 2
    assert all('--workspace' in row['args'] for row in rows)
    assert rows[0]['threads'] == '1'
    assert 'Dispatch profile: production-auto' in result.stdout
    for feature_args in (['--all-features'],
                         ['--features', 'portable-only'], ['--features=portable-only'],
                         ['-F', 'portable-only'], ['-Fportable-only']):
      result, rows = run(['--native', '--', *feature_args])
      assert result.returncode == 2 and not rows, (feature_args, result.stderr, rows)
      assert 'Cargo feature selection must use the repository dispatch profile' in result.stderr
    for dispatch in ('--native', '--portable'):
      result, rows = run(['--all', '--release', dispatch], PLAN_FAIL='1')
      assert result.returncode == 0 and len(rows) == 2, (result.stderr, rows)
      assert all('--release' in row['args'] for row in rows)
      assert '--doc' in rows[1]['args']
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
  check_evidence_recipe()
  print('Test runner regressions passed')


if __name__ == '__main__':
  main()
