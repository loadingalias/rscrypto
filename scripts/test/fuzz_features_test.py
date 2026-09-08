#!/usr/bin/env python3
"""Check each scoped fuzz workspace's resolved rscrypto features independently."""

import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
  features = tomllib.loads((ROOT / 'Cargo.toml').read_text())['features']
  manifests = sorted(ROOT.glob('fuzz-packages/*/Cargo.toml'))
  assert manifests, 'no scoped fuzz packages found'
  for manifest in manifests:
    dependency = tomllib.loads(manifest.read_text())['dependencies']['rscrypto']
    pending = ['std', *dependency.get('features', [])]
    if dependency.get('default-features', True):
      pending.append('default')
    expected = set()
    while pending:
      feature = pending.pop()
      if feature in expected or feature not in features:
        continue
      expected.add(feature)
      pending.extend(features[feature])
    metadata = json.loads(subprocess.check_output([
      'cargo', 'metadata', '--offline', '--locked', '--format-version', '1',
      '--manifest-path', str(manifest)], cwd=ROOT))
    package_id = next(p['id'] for p in metadata['packages']
                      if Path(p['manifest_path']) == ROOT / 'Cargo.toml')
    actual = set(next(n['features'] for n in metadata['resolve']['nodes'] if n['id'] == package_id))
    assert actual == expected, (
      f'{manifest.parent.name}: unexpected={sorted(actual - expected)}, '
      f'missing={sorted(expected - actual)}')
    print(f'{manifest.parent.name}: {",".join(sorted(actual))}')
  print(f'Scoped fuzz feature isolation passed for {len(manifests)} packages')


if __name__ == '__main__':
  main()
