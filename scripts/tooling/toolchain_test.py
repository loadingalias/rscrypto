#!/usr/bin/env python3
"""Check provisioning and native execution command generation for every host."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
  catalog = tomllib.loads((ROOT / '.config/tooling.toml').read_text())
  hosts = {row['rust-host']: row['components'] for row in catalog.values() if isinstance(row, dict) and 'rust-host' in row}
  hosts['aarch64-apple-darwin'] = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['components']
  stable = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
  nightly = tomllib.loads((ROOT / '.config/toolchains.toml').read_text())['nightly']
  nightly_hosts = {'riscv64gc-unknown-linux-gnu', 's390x-unknown-linux-gnu', 'powerpc64le-unknown-linux-gnu'}
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/lib/toolchain.py', 'scripts/lib/toolchain.sh', 'scripts/lib/python.sh',
                 'scripts/lib/rail-plan.sh', 'scripts/test/test.sh', 'scripts/test/test-examples.sh', 'rust-toolchain.toml',
                 '.config/toolchains.toml', 'Cargo.toml'):
      path = root / name
      path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, path)
    binary = root / 'bin'
    binary.mkdir()
    stub = binary / 'stub'
    stub.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'rustc':
  print('host: ' + os.environ['FIXTURE_HOST'])
else:
  with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(json.dumps({'command': [name, *sys.argv[1:]], 'channel': os.environ.get('RUSTUP_TOOLCHAIN')}) + '\\n')
  if name == 'cargo' and sys.argv[1] == 'metadata':
    print(json.dumps({'packages': [{'name': 'rscrypto', 'targets': [
      {'name': 'aead_seal_open', 'kind': ['example'], 'crate_types': ['bin'],
       'required-features': ['alloc', 'chacha20poly1305', 'getrandom']}]}]}))
''')
    stub.chmod(0o755)
    for name in ('rustc', 'cargo', 'cargo-nextest', 'rustup'):
      (binary / name).symlink_to(stub)
    log = root / 'commands.jsonl'
    env = {key: value for key, value in os.environ.items()
           if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
    env.update(PATH=str(binary) + os.pathsep + os.environ['PATH'], PYTHON=sys.executable,
               COMMAND_LOG=str(log), RUSTUP_TOOLCHAIN='wrong-ambient-channel')

    def run(command, host):
      log.write_text('')
      result = subprocess.run(command, cwd=root, env={**env, 'FIXTURE_HOST': host},
                              text=True, capture_output=True, timeout=20)
      assert result.returncode == 0, result.stderr
      return [json.loads(line) for line in log.read_text().splitlines()]

    for host in sorted(hosts):
      channel = nightly if host in nightly_hosts else stable
      for requested in ([], hosts[host]):
        components = [arg for component in requested for arg in ('--component', component)]
        commands = run([sys.executable, 'scripts/lib/toolchain.py', '--install', host, *components], host)
        expected = [(stable, ['rustfmt']), (nightly, ['clippy'])] if host in nightly_hosts else [
          (stable, ['clippy', 'rustfmt'])]
        assert [row['command'] for row in commands] == [
          ['rustup', 'toolchain', 'install', value, '--profile', 'minimal',
           *[arg for component in dict.fromkeys([*defaults, *requested]) for arg in ('--component', component)]]
          for value, defaults in expected], (host, requested, commands)
      commands = run(['bash', 'scripts/test/test.sh', '--all', '--portable', '--lib', 'two words'], host)
      assert commands and all(row['channel'] == channel for row in commands), (host, commands)
      assert commands[-1]['command'][-1] == 'two words'
      commands = run([sys.executable, 'scripts/lib/toolchain.py', '--exec', 'cargo', 'build', 'two words'], host)
      assert commands == [{'command': ['cargo', 'build', 'two words'], 'channel': channel}]
      commands = run(['bash', 'scripts/test/test-examples.sh'], host)
      assert len(commands) == 2 and all(row['channel'] == channel for row in commands), (host, commands)
      assert commands[-1]['command'] == ['cargo', 'run', '--locked', '--quiet', '--no-default-features',
                                        '--example', 'aead_seal_open', '--features', 'alloc,chacha20poly1305,getrandom']
    # Cross-check every supported target, including the non-host RISC-V lane.
    targets = json.loads((ROOT / '.config/target-matrix.json').read_text())['targets']
    for target in targets:
      result = subprocess.check_output([sys.executable, str(ROOT / 'scripts/lib/toolchain.py'), '--target', target], text=True).strip()
      assert result == (nightly if target in nightly_hosts or target == 'riscv32imac-unknown-none-elf' else stable), target
  print(f'Toolchain provisioning and execution regressions passed for {len(hosts)} hosts')


if __name__ == '__main__':
  main()
