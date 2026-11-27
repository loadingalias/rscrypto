"""Build the linked CT library test and record the executable reported by Cargo."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--target', required=True)
  parser.add_argument('--target-dir', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('rustc', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  manifest = tomllib.loads((ROOT / 'ct.toml').read_text())
  features = manifest['equality_evidence']['release_binary']['features']
  # Cargo's bench profile builds the library test and accepts flags for its final rustc invocation.
  # It inherits release. Override the two local bench settings and preserve release panic behavior.
  command = ['cargo', 'rustc', '-Zpanic-abort-tests', '--locked', '-p', 'rscrypto', '--lib', '--profile', 'bench',
             '--config', 'profile.bench.debug=false', '--config', 'profile.bench.strip="debuginfo"',
             '--no-default-features', '--features', ','.join(features), '--target', args.target,
             '--target-dir', str(args.target_dir), '--message-format=json-render-diagnostics',
             *args.rustc]
  binaries = []
  with subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE) as build:
    for line in build.stdout:
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        print(line, end='')
        continue
      if row.get('reason') == 'compiler-artifact' and row['target']['name'] == 'rscrypto' and row['profile']['test'] and row.get('executable'):
        binaries.append(Path(row['executable']))
    if status := build.wait():
      return status
  if len(binaries) != 1:
    raise ValueError(f'expected one rscrypto library test executable, got {binaries}')
  shutil.copy2(binaries[0], args.output)
  return 0


if __name__ == '__main__':
  sys.exit(main())
