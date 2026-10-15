#!/usr/bin/env -S uv run python
"""Build target test suites on x86-64; execute the sealed suites on native hardware.

Doctests run only on the native CI rows. Each cross suite runs under the `cross` or
`cross-internal` Nextest profile, which skips work the native rows or another suite
already cover on the same target (`.config/nextest.toml`).
"""

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
from workspace_features import native_features
from evidence_suite import TARGET_ARGS, validate_backend_evidence

MODES = ("native", "portable", "internal-native", "internal-portable")


def features(mode):
    return (['--all-features'] if mode.endswith('portable') else
            ['--no-default-features', '--features', ','.join(native_features(ROOT))])


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
        for mode in MODES:
            internal = mode.startswith('internal-')
            mode_env = env.copy()
            if internal:
                flags = subprocess.check_output([sys.executable, '-B', str(ROOT / 'scripts/ct/internal.py'),
                    '--target', target, '--print-encoded-rustflags'], cwd=ROOT, env=env, text=True)
                mode_env['CARGO_ENCODED_RUSTFLAGS'] = flags
            args = ['--target', target, *features(mode)]
            command = ['cargo', 'nextest', 'archive', '--locked', '--workspace', '--release', *args,
                       '--archive-file', str(out / f'{mode}.tar.zst')]
            if internal:
                command += TARGET_ARGS
            subprocess.run(command, cwd=ROOT, env=mode_env, check=True)
            metadata['modes'][mode] = {'command': command, 'internal': internal,
                'encoded_rustflags': mode_env.get('CARGO_ENCODED_RUSTFLAGS')}
        bundle.seal(ROOT, out, 'rscrypto.cross.tests', target, identity, metadata)
        bundle.pack(out, archive, compression='xz')


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
    if set(manifest['metadata']['modes']) != set(MODES):
        raise ValueError('ordinary and internal suites require both dispatch modes')
    for mode, record in manifest['metadata']['modes'].items():
        internal = mode.startswith('internal-')
        flags = (record.get('encoded_rustflags') or '').split('\x1f')
        if record.get('internal') != internal or ('rscrypto_internal' in flags) != internal:
            raise ValueError('test archive internal-build provenance differs from its suite')
    results = {}
    for mode in MODES:
        internal = mode.startswith('internal-')
        with (out / f'{mode}-nextest.log').open('w') as log:
            subprocess.run(['cargo', 'nextest', 'run', '--archive-file', str(incoming / f'{mode}.tar.zst'),
                            '--workspace-remap', str(ROOT), '--config-file', str(ROOT / '.config/nextest.toml'),
                            '--profile', 'cross-internal' if internal else 'cross',
                            '--no-tests', 'fail'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[mode] = {'status': 'pass'}
        if internal:
            results[mode]['backend_evidence'] = validate_backend_evidence((out / f'{mode}-nextest.log').read_text(),
                'portable-only' if mode.endswith('portable') else 'production-auto')
    # Detect accidental changes to inputs throughout execution as well as before it.
    bundle.verify(ROOT, incoming, 'rscrypto.cross.tests', target)
    (out / 'summary.json').write_text(json.dumps({'status': 'pass', 'source': manifest['source'],
        'archive_sha256': bundle.digest(archive), 'host': platform.uname()._asdict(),
        'nextest': version, 'modes': results}, indent=2) + '\n')
    print(f'{target} ordinary and internal suites passed: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    (prepare if args.operation == 'prepare' else execute)(args.target, args.archive.resolve())


if __name__ == '__main__':
    main()
