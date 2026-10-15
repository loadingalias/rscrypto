#!/usr/bin/env -S uv run python
"""Retain one bounded resource/latency observation of the catalog TLS workload."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from execution import build_environment, build_identity, source_evidence

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend')
    parser.add_argument('credential')
    parser.add_argument('group')
    parser.add_argument('suite')
    parser.add_argument('mode', choices=('full', 'resumed'))
    parser.add_argument('--iterations', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10_000 or not 1 <= args.workers <= 16:
        parser.error('iterations must be 1..=10000 and workers 1..=16')
    output = (args.output or ROOT / 'benchmark_results' / ('provider-scorecard-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    build_environment()
    manifest = ROOT / 'tools/provider-bench/Cargo.toml'
    build = ['cargo', 'build', '--locked', '--release', '--manifest-path', str(manifest), '--features', 'ml-dsa', '--bin', 'scorecard']
    source_evidence(output)
    with (output / 'build.log').open('wb') as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--no-deps', '--format-version=1', '--manifest-path', str(manifest)]))
    binary = Path(metadata['target_directory']) / 'release' / ('scorecard.exe' if os.name == 'nt' else 'scorecard')
    command = [str(binary), args.backend, args.credential, args.group, args.suite, args.mode, str(args.iterations), str(args.workers)]
    provenance = {'build': build, 'run': command, 'identity': build_identity(),
                  'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    shutil.copyfile(binary, output / binary.name)
    with (output / 'scorecard.json').open('wb') as result, (output / 'stderr.log').open('wb') as errors:
        subprocess.run(command, stdout=result, stderr=errors, check=True, timeout=605)
    (output / 'SHA256SUMS').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(output)) + '\n'
                                            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS'))
    print(f'Provider scorecard evidence: {output}')


if __name__ == '__main__':
    main()
