#!/usr/bin/env python3
"""Build complete target suites on x86-64; execute sealed suites on native hardware."""

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
from cross_build import TARGETS, environment, require_host
import doctest_bundle


def features(mode):
    graph = tomllib.loads((ROOT / 'Cargo.toml').read_text())['features']
    selected = sorted(set(graph) - {'portable-only'})
    if any('portable-only' in graph[name] for name in selected):
        raise ValueError('native features indirectly enable portable-only')
    return ['--all-features'] if mode == 'portable' else ['--no-default-features', '--features', ','.join(selected)]


def nextest_identity(report):
    lines = report.splitlines()
    pin = tomllib.loads((ROOT / '.config/tooling.toml').read_text())['cargo']['cargo-nextest']
    if not lines or lines[0].split()[:2] != ['cargo-nextest', pin]:
        raise ValueError(f'Nextest version mismatch: {report}; expected {pin}')
    # Cross-architecture binaries intentionally report different hosts. Keep
    # every other field, including the full source commit, in the comparison.
    return [line for line in lines if not line.startswith('host: ')]


def nextest_version():
    actual = subprocess.check_output(['cargo', 'nextest', '--version'], text=True).strip()
    nextest_identity(actual)
    return actual


def prepare(target, archive):
    if archive.exists():
        raise ValueError(f'refusing to overwrite existing evidence: {archive}')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('cross-compiled CI preparation requires the Ubuntu x86-64 cross-toolchain host')
    env = environment(target)
    compiler_path = shutil.which(TARGETS[target][1] + '-gcc')
    if compiler_path is None:
        raise FileNotFoundError('install the ci-cross-build tooling before preparing cross-compiled tests')
    compiler = Path(compiler_path).resolve(strict=True)
    identity = bundle.source_identity(ROOT)
    version = nextest_version()
    directory = ROOT / 'target/cross-transfer' / target
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=directory) as temporary:
        out = Path(temporary)
        # Same all-targets native/portable Clippy, independent workspaces, and docs.
        subprocess.run(['just', 'ci-check-target', target], cwd=ROOT, env=env, check=True)
        metadata = {'nextest': version, 'rustc': subprocess.check_output(['rustc', '-vV'], env=env, text=True),
                    'profile': tomllib.loads((ROOT / 'Cargo.toml').read_text())['profile']['release'],
                    'checks': 'ci-check-target', 'modes': {}}
        metadata['linker'] = {'path': str(compiler), 'sha256': bundle.digest(compiler),
                             'version': subprocess.check_output([str(compiler), '--version'], text=True)}
        for mode in ('native', 'portable'):
            args = ['--target', target, *features(mode)]
            command = ['cargo', 'nextest', 'archive', '--locked', '--workspace', '--release', *args,
                       '--archive-file', str(out / f'{mode}.tar.zst')]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            plan = doctest_bundle.prepare(ROOT, out / f'{mode}-docs', args, env)
            metadata['modes'][mode] = {'command': command, 'doctests': plan['total']}
        bundle.seal(ROOT, out, 'rscrypto.cross.tests', target, identity, metadata)
        bundle.pack(out, archive)


def execute(target, archive):
    require_host(target)
    environment(target)  # Reject inherited filters/profile overrides on the consumer too.
    version = nextest_version()
    directory = ROOT / 'target/cross-results' / target
    directory.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-', dir=directory))
    incoming = out / 'input'
    bundle.unpack(archive, incoming)
    manifest = bundle.verify(ROOT, incoming, 'rscrypto.cross.tests', target)
    if nextest_identity(manifest['metadata']['nextest']) != nextest_identity(version):
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
    bundle.verify(ROOT, incoming, 'rscrypto.cross.tests', target)
    (out / 'summary.json').write_text(json.dumps({'status': 'pass', 'source': manifest['source'],
        'archive_sha256': bundle.digest(archive), 'host': platform.uname()._asdict(),
        'nextest': version, 'modes': results}, indent=2) + '\n')
    print(f'{target} native/portable suites and doctests passed: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    (prepare if args.operation == 'prepare' else execute)(args.target, args.archive.resolve())


if __name__ == '__main__':
    main()
