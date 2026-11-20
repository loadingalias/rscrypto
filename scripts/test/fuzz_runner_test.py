#!/usr/bin/env -S uv run python
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
    ['git', 'ls-files', '-z', '--', 'fuzz/corpus/*'],
    cwd=ROOT).decode().rstrip('\0').split('\0')
  assert inventory == sorted(set(tracked)), 'committed seed inventory must match tracked corpus inputs'
  assert all((ROOT / path).is_file() for path in inventory), 'committed seeds must exist'
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/test/test-fuzz.sh', 'scripts/test/test-fuzz-asan.sh', 'scripts/lib/fuzz.sh'):
      destination = root / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, destination)
    toolchain = root / 'scripts/lib/toolchain.sh'
    toolchain.write_text('#!/bin/sh\necho fixture-nightly\n')
    toolchain.chmod(0o755)
    package = root / 'fuzz'
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
if sys.argv[1] == 'test' and os.environ.get('CHECK_ASAN_FLAGS'):
    flags = os.environ.get('RUSTFLAGS', '')
    assert '-Zsanitizer=address' in flags
    assert ('-Ctarget-feature=-outline-atomics' in flags) == (
        os.environ.get('FIXTURE_HOST') == 'aarch64-unknown-linux-gnu')
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
    rustc.write_text('#!/bin/sh\necho "host: ${FIXTURE_HOST:-fixture-host}"\n')
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
      selections = ([],) if script.endswith('asan.sh') else ([], ['--list'], ['fixture'], ['--targets', 'fixture'])
      for args in selections:
        for discovery in ('fail', 'empty'):
          result, commands = run(args, discovery, script)
          assert result.returncode != 0, (script, args, discovery, result.stdout)
          assert result.stderr, (script, args, discovery)
          assert not any(c[:2] == ['fuzz', 'run'] or c[0] == 'test' for c in commands), commands
    result, commands = run([], script='test-fuzz-asan.sh')
    assert result.returncode == 0, result.stderr
    assert 'passed for 1 targets' in result.stdout, result.stdout
    lists = [c for c in commands if c[0] == 'test' and '--list' in c]
    runs = [c for c in commands if c[0] == 'test' and '--list' not in c]
    assert len(lists) == len(runs) == 1 and '--include-ignored' in runs[0], commands
    for replay in ('missing', 'extra', 'renamed', 'list-fail', 'run-fail'):
      result, commands = run([], script='test-fuzz-asan.sh', replay=replay)
      assert result.returncode != 0, (replay, result.stdout)
      assert 'replay passed' not in result.stdout, result.stdout
      if replay in ('missing', 'extra', 'renamed'):
        assert 'inventory mismatch' in result.stderr, result.stderr
      if replay != 'run-fail':
        assert not any(c[0] == 'test' and '--list' not in c for c in commands), commands
    for host in ('aarch64-unknown-linux-gnu', 'x86_64-unknown-linux-gnu'):
      result, commands = run([], script='test-fuzz-asan.sh', FIXTURE_HOST=host, CHECK_ASAN_FLAGS='1')
      assert result.returncode == 0, (host, result.stderr)
      assert len([c for c in commands if c[0] == 'test']) == 2, commands
    result, commands = run(['--build'])
    assert result.returncode == 0, result.stderr
    assert len([c for c in commands if c[:2] == ['fuzz', 'build']]) == 1, commands
    assert not any(c[:2] == ['fuzz', 'run'] for c in commands), commands
    for args in (['--list', '--build'], ['--targets', 'fixture', '--build'],
                 ['--build', 'fixture'], ['--unknown'], ['fixture', '0'], ['fixture', '1', 'extra']):
      result, commands = run(args)
      assert result.returncode == 2 and not commands, (args, result.stderr, commands)
    result, commands = run(['--unknown'], script='test-fuzz-asan.sh')
    assert result.returncode == 2 and not commands, (result.stderr, commands)
    for args in ([], ['fixture'], ['--targets', 'fixture']):
      result, commands = run(args)
      runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
      assert result.returncode == 0, (args, result.stderr)
      assert [c[c.index('--fuzz-dir') + 1] for c in runs] == [str(package)], (args, commands)
    result, commands = run([], RSCRYPTO_FUZZ_BUDGET_SECS='59')
    assert result.returncode == 2 and 'exceed' in result.stderr
    assert not any(c[:2] == ['fuzz', 'run'] for c in commands)
    result, commands = run([], RSCRYPTO_FUZZ_BUDGET_SECS='60', RSCRYPTO_FUZZ_VALIDATE_ONLY='1')
    assert result.returncode == 0, result.stderr
    assert not any(c[:2] == ['fuzz', 'run'] for c in commands)
    result, commands = run(['--targets', 'fixture'], discovery='extra-target')
    runs = [c for c in commands if c[:2] == ['fuzz', 'run']]
    assert result.returncode == 0 and len(runs) == 1
    assert 'fixture' in runs[0] and 'unrelated' not in runs[0], commands
    for selection in ('fixture,missing', 'fixture,fixture', 'fixture,', ',fixture'):
      result, commands = run(['--targets', selection])
      assert result.returncode != 0, selection
      assert not any(c[:2] == ['fuzz', 'run'] for c in commands), commands
    result, _ = run(['--clean'])
    assert result.returncode == 0, result.stderr
    assert (package / 'corpus/fixture/seed').read_bytes() == b'regression input'
    assert (package / 'corpus/fixture/local-discovery').read_bytes() == b'new input'
    assert not (package / 'artifacts').exists() and not (package / 'coverage').exists()
  print('Fuzz runner regressions passed')


if __name__ == '__main__':
  main()
