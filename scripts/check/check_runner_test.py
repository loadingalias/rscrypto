#!/usr/bin/env python3
"""Verify command effects and coverage by executing the real check entry point."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


def main():
  source = Path(__file__).resolve().parents[2]
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/check/check.sh', 'scripts/lib/toolchain.sh', 'scripts/lib/toolchain.py', 'scripts/lib/python.sh', 'Cargo.toml',
                 'rust-toolchain.toml', '.config/toolchains.toml', '.config/target-matrix.json'):
      destination = root / name
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source / name, destination)
    binary = root / 'bin'
    binary.mkdir()
    log = root / 'commands.jsonl'
    stub = binary / 'stub'
    stub.write_text('''#!''' + sys.executable + '''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CHECK_LOG'], 'a') as output:
    output.write(json.dumps([name, *args]) + '\\n')
if name == 'rustc': print('host: ' + os.environ['CHECK_HOST'])
if name == 'rustup':
    if os.environ.get('CHECK_INVENTORY_EXIT'): sys.exit(int(os.environ['CHECK_INVENTORY_EXIT']))
    if args[0] == 'target':
        targets = json.loads(Path('.config/target-matrix.json').read_text())['targets']
        print('\\n'.join(t for t in targets if t != os.environ.get('CHECK_MISSING')))
    else: print('clippy-fixture (installed)')
if name == 'cargo' and 'clippy' in args:
    sys.exit(int(os.environ.get('CHECK_CLIPPY_EXIT', '0')))
''')
    stub.chmod(0o755)
    for name in ('cargo', 'rustc', 'rustup'):
      (binary / name).symlink_to(stub)
    for name in ('lint-independent-workspaces.sh',):
      (root / 'scripts/check' / name).symlink_to(stub)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
    environment.update(PATH=f'{binary}:{os.environ["PATH"]}', CHECK_PYTHON=sys.executable,
                       CHECK_LOG=str(log), CHECK_HOST='aarch64-apple-darwin')

    def run(mode, **extra):
      log.write_text('')
      result = subprocess.run(['bash', str(root / 'scripts/check/check.sh'), mode],
                              cwd=root, env={**environment, **extra}, capture_output=True,
                              text=True, timeout=30)
      return result, [json.loads(line) for line in log.read_text().splitlines()]

    targets = json.loads((root / '.config/target-matrix.json').read_text())['targets']
    feature_graph = tomllib.loads((root / 'Cargo.toml').read_text())['features']
    features = set(feature_graph)

    def expanded_features(selected):
      expanded = set(selected)
      pending = list(selected)
      while pending:
        for dependency in feature_graph.get(pending.pop(), []):
          if dependency in feature_graph and dependency not in expanded:
            expanded.add(dependency)
            pending.append(dependency)
      return expanded

    stable = tomllib.loads((root / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    nightly = tomllib.loads((root / '.config/toolchains.toml').read_text())['nightly']
    for mode in ('fix', 'local', 'native'):
      result, commands = run(mode)
      assert result.returncode == 0, result.stderr
      inventories = [tuple(c) for c in commands if c[0] == 'rustup']
      assert len(inventories) == len(set(inventories)) == (2 if mode == 'native' else 4), inventories
      clippy = [c for c in commands if c[0] == 'cargo' and 'clippy' in c]
      expected = targets if mode != 'native' else ['aarch64-apple-darwin']
      assert len(clippy) == 2 * len(expected), clippy
      for target in expected:
        rows = [c for c in clippy if c[c.index('--target') + 1] == target]
        assert len(rows) == 2
        for command in rows:
          selected = set(command[command.index('--features') + 1].split(','))
          assert ('--release' in command) != ('portable-only' in selected)
          assert ('--fix' in command) == (mode == 'fix')
          assert ('--allow-dirty' in command) == (mode == 'fix')
          assert ('--allow-staged' in command) == (mode == 'fix')
          assert '--locked' in command and '--no-default-features' in command
          assert ('--all-targets' in command) == (target == 'aarch64-apple-darwin')
          assert ('--lib' in command) == (target != 'aarch64-apple-darwin')
          assert (command[1] == '+' + nightly) == target.startswith(('powerpc64le-', 's390x-', 'riscv32', 'riscv64gc-'))
          if target == 'aarch64-apple-darwin':
            assert selected - {'portable-only'} == features - {'portable-only'}
          if '-none' in target or target == 'wasm32-unknown-unknown':
            assert not expanded_features(selected) & {'std', 'parallel', 'getrandom', 'default'}, command
            assert 'full' in selected
          if target == 'wasm32-wasip1':
            assert {'std', 'getrandom', 'full'} <= selected and 'parallel' not in selected
      fmt = [c for c in commands if c[:3] == ['cargo', '+' + stable, 'fmt']]
      assert fmt == ([['cargo', '+' + stable, 'fmt', '--all']] * 2 if mode == 'fix' else
                     [['cargo', '+' + stable, 'fmt', '--all', '--', '--check']])
      assert not any('plan' in c for c in commands)
      deny = [c for c in commands if c[:2] == ['cargo', 'deny']]
      assert len(deny) == (0 if mode == 'fix' else 1)
      if deny:
        assert ('--target' in deny[0]) == (mode == 'native')
      assert (['lint-independent-workspaces.sh'] in commands) == (mode != 'fix')
    result, commands = run('check')
    assert result.returncode == 0, result.stderr
    inventories = [c for c in commands if c[0] == 'rustup']
    assert len(inventories) == 4, inventories
    clippy = [c for c in commands if c[0] == 'cargo' and 'clippy' in c]
    assert len(clippy) == 4 * len(targets)
    assert sum('--fix' in c for c in clippy) == 2 * len(targets)
    tooling = tomllib.loads((source / '.config/tooling.toml').read_text())
    hosts = {row['rust-host'] for row in tooling.values() if isinstance(row, dict) and 'rust-host' in row}
    hosts.add('aarch64-apple-darwin')
    for host in sorted(hosts):
      result, commands = run('native', CHECK_HOST=host, RUSTUP_TOOLCHAIN='wrong-ambient-channel')
      assert result.returncode == 0, result.stderr
      expected = nightly if host in {'powerpc64le-unknown-linux-gnu', 's390x-unknown-linux-gnu', 'riscv64gc-unknown-linux-gnu'} else stable
      assert all(c[1] == '+' + expected for c in commands if c[0] == 'cargo' and 'clippy' in c)
    result, commands = run('fix', CHECK_MISSING='thumbv6m-none-eabi')
    assert result.returncode != 0 and not any(c[0] == 'cargo' for c in commands)
    result, commands = run('native', CHECK_CLIPPY_EXIT='7')
    assert result.returncode == 7 and not any(c[:2] == ['cargo', 'deny'] for c in commands)
    result, commands = run('fix', CHECK_INVENTORY_EXIT='19')
    assert result.returncode == 19 and not any(c[0] == 'cargo' for c in commands)
    result, commands = run('bogus')
    assert result.returncode == 2 and not commands
  print('Check runner regressions passed')


if __name__ == '__main__':
  main()
