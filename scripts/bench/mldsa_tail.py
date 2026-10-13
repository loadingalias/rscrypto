#!/usr/bin/env python3
"""Retain an ML-DSA signing-tail baseline: per-signature latency and attempt counts.

Builds tools/mldsa-tail twice: the production build times each signature, and a
`--cfg rscrypto_mldsa_attempts` build counts rejection-sampling attempts (its
latency is not production latency). Both sign the same deterministic inputs.
Both outputs and the provenance are printed so a remote `just ssh-just` log carries them home.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from execution import build_environment, build_identity, source_evidence

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tools/mldsa-tail/Cargo.toml'
BINARY = 'rscrypto-mldsa-tail'


def build(output, name, rustflags, portable):
  target_dir = ROOT / 'target' / 'mldsa-tail' / name
  command = ['cargo', 'build', '--locked', '--release', '--manifest-path', str(MANIFEST), '--target-dir', str(target_dir)]
  if portable:
    command += ['--features', 'portable-only']
  environment = dict(os.environ)
  if rustflags:
    environment['RUSTFLAGS'] = ' '.join(filter(None, [environment.get('RUSTFLAGS', ''), rustflags]))
  with (output / f'build-{name}.log').open('wb') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200, env=environment)
  binary = target_dir / 'release' / (BINARY + ('.exe' if os.name == 'nt' else ''))
  return command, binary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--keys', type=int, default=16)
  parser.add_argument('--messages', type=int, default=64)
  parser.add_argument('--portable', action='store_true', help='select portable-only dispatch')
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  if not 1 <= args.keys <= 1024 or not 1 <= args.messages <= 4096:
    parser.error('keys must be 1..=1024 and messages 1..=4096')
  stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  output = (args.output or ROOT / 'benchmark_results' / f'mldsa-tail-{stamp}').resolve()
  output.mkdir(parents=True, exist_ok=False)
  os.chdir(ROOT)
  build_environment()
  source_evidence(output)
  provenance = {'identity': build_identity(), 'keys': args.keys, 'messages': args.messages,
                'portable': args.portable, 'runs': {}}
  for name, rustflags in (('production', ''), ('attempts', '--cfg rscrypto_mldsa_attempts')):
    command, binary = build(output, name, rustflags, args.portable)
    run = [str(binary), str(args.keys), str(args.messages)]
    result = subprocess.run(run, capture_output=True, check=True, timeout=3600)
    (output / f'{name}.txt').write_bytes(result.stdout)
    provenance['runs'][name] = {'build': command, 'rustflags': rustflags, 'run': run,
                                'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    print(f'== {name}')
    print(result.stdout.decode(), end='')
  (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
  print('== provenance')
  print(json.dumps(provenance, indent=2))
  (output / 'SHA256SUMS').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(output)) + '\n'
                                          for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS'))
  print(f'ML-DSA signing-tail evidence: {output}')


if __name__ == '__main__':
  main()
