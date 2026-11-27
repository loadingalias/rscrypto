"""Build the linked CT library test and record the executable reported by Cargo."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def build_library_test(target, target_dir, profile, features, rustc, env=None):
  # Cargo's bench mode builds the library test and passes trailing flags only to its rustc invocation.
  profiles = tomllib.loads((ROOT / 'Cargo.toml').read_text())['profile']
  settings = {'debug': False, 'strip': 'none'} | profiles['release'] | profiles[profile]
  panic_flags = ['-Zpanic-abort-tests'] if profile == 'release' else []
  command = ['cargo', 'rustc', *panic_flags, '--locked', '-p', 'rscrypto', '--lib', '--profile', 'bench',
             '--config', f'profile.bench.inherits="{profile}"']
  for key in sorted(profiles['bench'].keys() - {'inherits'}):
    command.extend(['--config', f'profile.bench.{key}={json.dumps(settings[key])}'])
  command.extend(['--no-default-features', '--features', ','.join(features), '--target', target,
                  '--target-dir', str(target_dir), '--message-format=json-render-diagnostics', '--', *rustc])
  binaries = []
  with subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE) as build:
    for line in build.stdout:
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        print(line, end='')
        continue
      if row.get('reason') == 'compiler-artifact' and row['target']['name'] == 'rscrypto' and row['profile']['test'] and row.get('executable'):
        binaries.append(Path(row['executable']))
    if status := build.wait():
      raise subprocess.CalledProcessError(status, command)
  if len(binaries) != 1:
    raise ValueError(f'expected one rscrypto library test executable, got {binaries}')
  return binaries[0]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--target', required=True)
  parser.add_argument('--target-dir', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('rustc', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  manifest = tomllib.loads((ROOT / 'ct.toml').read_text())
  features = manifest['equality_evidence']['release_binary']['features']
  flags = args.rustc[1:] if args.rustc[:1] == ['--'] else args.rustc
  try:
    binary = build_library_test(args.target, args.target_dir, 'release', features, flags)
  except subprocess.CalledProcessError as error:
    return error.returncode
  shutil.copy2(binary, args.output)
  return 0


if __name__ == '__main__':
  sys.exit(main())
