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


def check_vendored_packages(source):
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    script = root / 'scripts/check/lint-independent-workspaces.sh'
    script.parent.mkdir(parents=True)
    shutil.copy2(source / 'scripts/check/lint-independent-workspaces.sh', script)
    shutil.copy2(source / 'Cargo.toml', root / 'Cargo.toml')
    workspace = root / 'tools/harness'
    workspace.mkdir(parents=True)
    (workspace / 'Cargo.toml').write_text('[workspace]\n')
    binary = root / 'bin'
    binary.mkdir()
    cargo = binary / 'cargo'
    cargo.write_text('#!' + sys.executable + '''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if Path(sys.argv[0]).name == 'jq':
    result = subprocess.run([os.environ['REAL_JQ'], *args], capture_output=True)
    sys.stdout.buffer.write(result.stdout.replace(b'\\n', b'\\r\\n'))
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
manifest = Path(args[args.index('--manifest-path') + 1])
if args[0] == 'metadata':
    prefix = os.environ['METADATA_PREFIX']
    separator = os.environ['METADATA_SEPARATOR']
    print(json.dumps({'workspace_root': str(manifest.parent), 'packages': [
        {'name': name, 'manifest_path': prefix + separator.join(path.split('/'))}
        for name, path in [('upstream', 'vendor/upstream/Cargo.toml'),
                           ('harness', 'Cargo.toml'),
                           ('vendor-tools', 'vendor-tools/Cargo.toml')]
    ]}))
else:
    Path(os.environ['CHECK_LOG']).write_text(json.dumps(args))
    if manifest.parent.name == os.environ.get('CHECK_FAIL_WORKSPACE'):
        sys.exit(7)
''')
    cargo.chmod(0o755)
    (binary / 'jq').symlink_to(cargo)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
    log = root / 'commands.json'
    for prefix, separator in [('/repo/tools/harness/', '/'), ('C:\\repo\\tools\\harness\\', '\\')]:
      result = subprocess.run(['bash', str(script)], cwd=root, capture_output=True, text=True, timeout=30,
                              env={**environment, 'PATH': f'{binary}:{os.environ["PATH"]}',
                                   'CHECK_LOG': str(log), 'METADATA_PREFIX': prefix,
                                   'REAL_JQ': shutil.which('jq'),
                                   'METADATA_SEPARATOR': separator})
      assert result.returncode == 0, result.stderr
      command = json.loads(log.read_text())
      excluded = [command[i + 1] for i, arg in enumerate(command) if arg == '--exclude']
      assert excluded == ['upstream'], (prefix, command)
      assert '--workspace' in command and '--all-targets' in command and '--no-deps' in command

    later = root / 'tools/later'
    later.mkdir()
    (later / 'Cargo.toml').write_text('[workspace]\n')
    result = subprocess.run(['bash', str(script)], cwd=root, capture_output=True, text=True, timeout=30,
                            env={**environment, 'PATH': f'{binary}:{os.environ["PATH"]}',
                                 'CHECK_LOG': str(log), 'METADATA_PREFIX': '/repo/tools/harness/',
                                 'REAL_JQ': shutil.which('jq'), 'METADATA_SEPARATOR': '/',
                                 'CHECK_FAIL_WORKSPACE': 'harness'})
    assert result.returncode == 7, (result.returncode, result.stderr)
    command = json.loads(log.read_text())
    assert Path(command[command.index('--manifest-path') + 1]).resolve() == (workspace / 'Cargo.toml').resolve(), command
    assert 'Linting independent workspace: tools/later/' not in result.stdout


def main():
  source = Path(__file__).resolve().parents[2]
  check_vendored_packages(source)
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/check/check.sh', 'scripts/check/dependencies.sh', 'scripts/lib/toolchain.sh',
                 'scripts/lib/toolchain.py', 'scripts/lib/cross_build.py', 'scripts/lib/python.sh', 'Cargo.toml',
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

    def run(mode, *args, **extra):
      log.write_text('')
      result = subprocess.run(['bash', str(root / 'scripts/check/check.sh'), mode, *args],
                              cwd=root, env={**environment, **extra}, capture_output=True,
                              text=True, timeout=30)
      return result, [json.loads(line) for line in log.read_text().splitlines()]

    targets = json.loads((root / '.config/target-matrix.json').read_text())['targets']
    assert set(tomllib.loads((source / 'deny.toml').read_text())['graph']['targets']) == set(targets)
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
      assert len(deny) == (0 if mode in ('fix', 'native') else 1)
      assert (['cargo', 'audit'] in commands) == (mode not in ('fix', 'native'))
      if deny:
        assert '--target' not in deny[0]
      assert (['lint-independent-workspaces.sh'] in commands) == (mode != 'fix')
    for target in ('riscv64gc-unknown-linux-gnu', 'powerpc64le-unknown-linux-gnu', 's390x-unknown-linux-gnu'):
      result, commands = run('target', target)
      assert result.returncode == 0, result.stderr
      cross = [c for c in commands if c[0] == 'cargo' and 'clippy' in c]
      assert len(cross) == 2
      assert all('--all-targets' in c and c[c.index('--target') + 1] == target for c in cross)
      assert all(c[1] == '+' + nightly for c in cross)
      assert sum('--release' in c for c in cross) == 1
      assert ['lint-independent-workspaces.sh'] in commands
      assert not any('--fix' in c for c in commands)
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
