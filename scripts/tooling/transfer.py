#!/usr/bin/env python3
"""Build pinned runner tools on x86-64 and verify them before native installation."""

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import evidence_bundle as bundle
from cross_build import TARGETS, environment, require_host, verify_elf

KIND = 'rscrypto.cross.tools'


def prepare(target, archive):
    if (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
        raise ValueError('runner tools must be cross-built on Linux x86-64')
    env = environment(target)
    identity = bundle.source_identity(ROOT)
    pins = tomllib.loads((ROOT / '.config/tooling.toml').read_text())['cargo']
    directory = ROOT / 'target/cross-tools' / target
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=directory) as temporary:
        out = Path(temporary)
        for name in ('just', 'cargo-nextest'):
            subprocess.run(['cargo', 'install', '--locked', '--version', pins[name], '--target', target,
                            '--root', str(out), name], cwd=ROOT, env=env, check=True)
            verify_elf(out / 'bin' / name, target)
        bundle.seal(ROOT, out, KIND, target, identity, {name: pins[name] for name in ('just', 'cargo-nextest')})
        bundle.pack(out, archive)


def install(target, archive, destination):
    require_host(target)
    pins = tomllib.loads((ROOT / '.config/tooling.toml').read_text())['cargo']
    bundle.unpack(archive, destination)
    manifest = bundle.verify(ROOT, destination, KIND, target)
    if manifest['metadata'] != {name: pins[name] for name in ('just', 'cargo-nextest')}:
        raise ValueError('runner tool versions do not match this checkout')
    for name in ('just', 'cargo-nextest'):
        verify_elf(destination / 'bin' / name, target)
    # Only verified tool directories enter PATH, after the caller sees success.
    print(destination / 'bin')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'install'))
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', nargs='?', type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        if args.destination:
            parser.error('prepare does not take a destination')
        prepare(args.target, args.archive.resolve())
    else:
        if not args.destination:
            parser.error('install requires a fresh destination')
        install(args.target, args.archive.resolve(), args.destination.resolve())


if __name__ == '__main__':
    main()
