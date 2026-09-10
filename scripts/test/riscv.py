#!/usr/bin/env python3
"""Build complete RISC-V suites on a fast host; run sealed suites on RISC-V."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import evidence_bundle as bundle
from riscv_build import environment
import doctest_bundle

TARGET = 'riscv64gc-unknown-linux-gnu'

def features(mode):
    graph = tomllib.loads((ROOT / 'Cargo.toml').read_text())['features']
    selected = sorted(set(graph) - {'portable-only'})
    if any('portable-only' in graph[name] for name in selected):
        raise ValueError('native features indirectly enable portable-only')
    return ['--all-features'] if mode == 'portable' else ['--no-default-features', '--features', ','.join(selected)]


def nextest_version():
    actual = subprocess.check_output(['cargo', 'nextest', '--version'], text=True).strip()
    pin = tomllib.loads((ROOT / '.config/tooling.toml').read_text())['cargo']['cargo-nextest']
    if actual.split()[1] != pin:
        raise ValueError(f'Nextest version mismatch: {actual}; expected {pin}')
    return actual


def prepare(archive):
    if archive.exists():
        raise ValueError(f'refusing to overwrite existing evidence: {archive}')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('RISC-V CI preparation requires the Ubuntu x86-64 cross-toolchain host')
    env = environment()
    compiler_path = shutil.which('riscv64-linux-gnu-gcc')
    if compiler_path is None:
        raise FileNotFoundError('install the ci-riscv-build tooling before preparing RISC-V tests')
    compiler = Path(compiler_path).resolve(strict=True)
    identity = bundle.source_identity(ROOT)
    version = nextest_version()
    directory = ROOT / 'target/riscv-transfer'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=directory) as temporary:
        out = Path(temporary)
        # Same all-targets native/portable Clippy, independent workspaces, and docs.
        subprocess.run(['just', 'ci-check-target', TARGET], cwd=ROOT, env=env, check=True)
        metadata = {'nextest': version, 'rustc': subprocess.check_output(['rustc', '-vV'], env=env, text=True),
                    'profile': tomllib.loads((ROOT / 'Cargo.toml').read_text())['profile']['release'],
                    'checks': 'ci-check-target', 'modes': {}}
        metadata['linker'] = {'path': str(compiler), 'sha256': bundle.digest(compiler),
                             'version': subprocess.check_output([str(compiler), '--version'], text=True)}
        for mode in ('native', 'portable'):
            args = ['--target', TARGET, *features(mode)]
            command = ['cargo', 'nextest', 'archive', '--locked', '--workspace', '--release', *args,
                       '--archive-file', str(out / f'{mode}.tar.zst')]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            plan = doctest_bundle.prepare(ROOT, out / f'{mode}-docs', args, env)
            metadata['modes'][mode] = {'command': command, 'doctests': plan['total']}
        bundle.seal(ROOT, out, 'rscrypto.riscv.tests', TARGET, identity, metadata)
        bundle.pack(out, archive)


def execute(archive):
    if platform.system() != 'Linux' or platform.machine() != 'riscv64':
        raise ValueError('RISC-V tests must execute on the physical RISC-V Linux runner')
    environment()  # Reject inherited filters/profile overrides on the consumer too.
    version = nextest_version()
    directory = ROOT / 'target/riscv-results'
    directory.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-', dir=directory))
    incoming = out / 'input'
    bundle.unpack(archive, incoming)
    manifest = bundle.verify(ROOT, incoming, 'rscrypto.riscv.tests', TARGET)
    if manifest['metadata']['nextest'] != version:
        raise ValueError('producer and consumer Nextest versions differ')
    if set(manifest['metadata']['modes']) != {'native', 'portable'}:
        raise ValueError('both dispatch modes are required')
    results = {}
    for mode in ('native', 'portable'):
        with (out / f'{mode}-nextest.log').open('w') as log:
            subprocess.run(['cargo', 'nextest', 'run', '--archive-file', str(incoming / f'{mode}.tar.zst'),
                            '--workspace-remap', str(ROOT), '--config-file', str(ROOT / '.config/nextest.toml'),
                            '--no-tests', 'fail'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[mode] = doctest_bundle.execute(ROOT, incoming / f'{mode}-docs', out / f'{mode}-docs')
    # Detect accidental changes to inputs throughout execution as well as before it.
    bundle.verify(ROOT, incoming, 'rscrypto.riscv.tests', TARGET)
    (out / 'summary.json').write_text(json.dumps({'status': 'pass', 'source': manifest['source'],
        'archive_sha256': bundle.digest(archive), 'host': platform.uname()._asdict(), 'modes': results}, indent=2) + '\n')
    print(f'RISC-V native/portable suites and doctests passed: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    (prepare if args.operation == 'prepare' else execute)(args.archive.resolve())


if __name__ == '__main__':
    main()
