#!/usr/bin/env python3
"""Exercise fuzz discovery and cleanup without building or running fuzz targets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
  inventory = (ROOT / 'fuzz/committed-seeds.txt').read_text().splitlines()
  tracked = subprocess.check_output(
    ['git', 'ls-files', '-z', '--', 'fuzz/corpus/*', 'fuzz-packages/*/corpus/*'],
    cwd=ROOT).decode().rstrip('\0').split('\0')
  assert inventory == sorted(set(tracked)), 'committed seed inventory must match tracked corpus inputs'
  assert all((ROOT / path).is_file() for path in inventory), 'committed seeds must exist'
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/test/test-fuzz.sh', 'scripts/test/test-fuzz-asan.sh',
                 'scripts/lib/fuzz-packages.sh'):
      destination = root / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, destination)
    toolchain = root / 'scripts/lib/toolchain.sh'
    toolchain.write_text('#!/bin/sh\necho fixture-nightly\n')
    toolchain.chmod(0o755)
    (root / '.config').mkdir()
    (root / '.config/toolchains.toml').touch()
    packages = [root / 'fuzz', root / 'fuzz-packages/scoped']
    for package in packages:
      package.mkdir(parents=True)
      (package / 'Cargo.toml').write_text('[package.metadata]\ncargo-fuzz = true\n')
      for directory in ('corpus/fixture', 'artifacts', 'coverage'):
        (package / directory).mkdir(parents=True)
      (package / 'corpus/fixture/seed').write_bytes(b'regression input')
      (package / 'corpus/fixture/local-discovery').write_bytes(b'new input')
    binary = root / 'bin'
    binary.mkdir()
    cargo = binary / 'cargo'
    cargo.write_text('#!' + sys.executable + '''
import json, os, sys
with open(os.environ['FUZZ_LOG'], 'a') as output:
    output.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['fuzz', 'list']:
    if os.environ.get('DISCOVERY') == 'fail':
        print('target discovery failed', file=sys.stderr)
        sys.exit(23)
    if os.environ.get('DISCOVERY') != 'empty':
        print('fixture')
        if os.environ.get('DISCOVERY') == 'extra-target': print('unrelated')
elif sys.argv[1] == 'test':
    replay = os.environ.get('REPLAY', 'ok')
    if '--list' in sys.argv:
        if replay == 'list-fail':
            sys.exit(24)
        if replay not in ('missing', 'renamed'):
            print('replay_fixture_corpus: test')
        if replay in ('extra', 'renamed'):
            print('replay_unexpected_corpus: test')
    elif replay == 'run-fail':
        sys.exit(25)
''')
    cargo.chmod(0o755)
    rustc = binary / 'rustc'
    rustc.write_text('#!/bin/sh\necho "host: fixture-host"\n')
    rustc.chmod(0o755)
    log = root / 'commands.jsonl'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('FUZZ_', 'RSCRYPTO_', 'BASH_FUNC_'))
                   and key not in ('BASH_ENV', 'ENV')}
    environment.update(PATH=f'{binary}:{os.environ["PATH"]}', FUZZ_LOG=str(log),
                       RSCRYPTO_FUZZ_TARGET_CONCURRENCY='1')

    def run(args, discovery='ok', script='test-fuzz.sh', replay='ok', **extra):
      log.write_text('')
      result = subprocess.run(['bash', str(root / 'scripts/test' / script), *args],
                              cwd=root, env={**environment, 'DISCOVERY': discovery, 'REPLAY': replay, **extra},
                              capture_output=True, text=True, timeout=30)
      commands = [json.loads(line) for line in log.read_text().splitlines()]
      return result, commands

    for script in ('test-fuzz.sh', 'test-fuzz-asan.sh'):
      selections = (['--all'],) if script.endswith('asan.sh') else (
        ['--full'], ['--all'], ['--list'], ['fixture'], ['--targets', 'fixture'])
      for args in selections:
        for discovery in ('fail', 'empty'):
          result, commands = run(args, discovery, script)
          assert result.returncode != 0, (script, args, discovery, result.stdout)
          assert result.stderr, (script, args, discovery)
          assert not any(c[:2] == ['fuzz', 'run'] or c[0] == 'test' for c in commands), commands
    for scope, count in (('--full', 1), ('--scoped', 1), ('--all', 2)):
      result, commands = run([scope], script='test-fuzz-asan.sh')
      assert result.returncode == 0, result.stderr
      assert f'passed for {count} targets' in result.stdout, result.stdout
      lists = [c for c in commands if c[0] == 'test' and '--list' in c]
      runs = [c for c in commands if c[0] == 'test' and '--list' not in c]
      assert len(lists) == len(runs) == count, commands
      assert all('--include-ignored' in c for c in runs), commands
      for replay in ('missing', 'extra', 'renamed', 'list-fail', 'run-fail'):
        result, commands = run([scope], script='test-fuzz-asan.sh', replay=replay)
        assert result.returncode != 0, (scope, replay, result.stdout)
        assert 'replay passed' not in result.stdout, result.stdout
        if replay in ('missing', 'extra', 'renamed'):
          assert 'inventory mismatch' in result.stderr, result.stderr
        if replay != 'run-fail':
          assert not any(c[0] == 'test' and '--list' not in c for c in commands), commands
    for args in (['--build', '--all'], ['--all', '--build']):
      result, commands = run(args)
      assert result.returncode == 0, result.stderr
      assert len([c for c in commands if c[:2] == ['fuzz', 'build']]) == 2, commands
      assert not any(c[:2] == ['fuzz', 'run'] for c in commands), commands
    for args in (['--list', '--build'], ['--targets', 'fixture', '--build'],
                 ['--build', 'fixture'], ['--unknown'], ['fixture', '0'],
                 ['fixture', '1', 'extra'], ['--full', '--scoped']):
      result, commands = run(args)
      assert result.returncode == 2 and not commands, (args, result.stderr, commands)
    result, commands = run(['--all'])
    assert result.returncode == 0, result.stderr
    assert len([c for c in commands if c[:2] == ['fuzz', 'run']]) == 2, commands
    result, commands = run(['--all'], RSCRYPTO_FUZZ_BUDGET_SECS='60')
    assert result.returncode == 2 and 'exceed' in result.stderr
    assert not any(c[:2] == ['fuzz', 'run'] for c in commands)
    result, commands = run(['--all'], RSCRYPTO_FUZZ_BUDGET_SECS='120', RSCRYPTO_FUZZ_VALIDATE_ONLY='1')
    assert result.returncode == 0, result.stderr
    assert not any(c[:2] == ['fuzz', 'run'] for c in commands)
    result, commands = run(['--targets', 'fixture'])
    assert result.returncode == 0, result.stderr
    runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
    assert len(runs) == 1 and str(packages[0]) in runs[0], commands
    for scope, package in (('--full', packages[0]), ('--scoped', packages[1])):
      result, commands = run(['--targets', 'fixture', scope])
      runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
      assert result.returncode == 0 and len(runs) == 1 and str(package) in runs[0], commands
    for scope, selected in (('--all', packages), ('--full', packages[:1]), ('--scoped', packages[1:])):
      for args in ([scope], [scope, '--targets', 'fixture'], ['--targets', 'fixture', scope],
                   [scope, 'fixture'], ['fixture', scope]):
        result, commands = run(args)
        runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
        lists = [c for c in commands if c[:2] == ['fuzz', 'list']]
        assert result.returncode == 0, (args, result.stderr)
        assert [c[c.index('--fuzz-dir') + 1] for c in runs] == [str(p) for p in selected], (args, commands)
        assert len(lists) == len(selected), (args, commands)
    result, commands = run(['--all', '--targets', 'fixture'], discovery='extra-target')
    runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
    assert result.returncode == 0 and len(runs) == 2
    assert all('fixture' in c and 'unrelated' not in c for c in runs), commands
    for selection in ('fixture,missing', 'fixture,fixture', 'fixture,', ',fixture'):
      result, commands = run(['--all', '--targets', selection])
      assert result.returncode != 0, selection
      assert not any(c[:2] == ['fuzz', 'run'] for c in commands), commands
    result, _ = run(['--clean'])
    assert result.returncode == 0, result.stderr
    for package in packages:
      assert (package / 'corpus/fixture/seed').read_bytes() == b'regression input'
      assert (package / 'corpus/fixture/local-discovery').read_bytes() == b'new input'
      assert not (package / 'artifacts').exists() and not (package / 'coverage').exists()
    (packages[1] / 'Cargo.toml').unlink()
    result, _ = run(['--scoped'])
    assert result.returncode != 0, result.stdout
  print('Fuzz runner regressions passed')


if __name__ == '__main__':
  main()
