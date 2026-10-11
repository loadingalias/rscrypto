#!/usr/bin/env python3
"""Check provider features and its own normal/build dependency graph."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import toolchain

PACKAGE = 'rscrypto-rustls'


def feature_matrix(root=ROOT):
    core = tomllib.loads((root / 'Cargo.toml').read_text())
    provider = tomllib.loads((root / PACKAGE / 'Cargo.toml').read_text())
    if provider['package']['rust-version'] != core['package']['rust-version']:
        raise ValueError('the provider rust-version must match rscrypto')
    graph = provider['features']
    for feature, edges in graph.items():
        # OS entropy is delegated to the same getrandom crate as rscrypto.
        expected = ['rscrypto/' + feature, *(['dep:getrandom'] if feature == 'getrandom' else [])]
        if feature != 'default' and (feature not in core['features'] or edges != expected):
            raise ValueError(f'provider feature {feature} must forward one-to-one to rscrypto')
    native = sorted(set(graph) - {'default', 'portable-only'})
    return [('empty', []), ('default', ['default']), ('native', native),
            ('all', sorted(graph)),
            *[(feature, [feature]) for feature in sorted(set(graph) - {'default'})]]


def feature_args(features):
    return ['--no-default-features', *(['--features', ','.join(features)] if features else [])]


def supports_std(target):
    return '-none' not in target and target != 'wasm32-unknown-unknown'


def run(command, root=ROOT):
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def no_c(features, root=ROOT):
    command = ['cargo', '+' + toolchain.channel(), 'tree', '--locked', '-p', PACKAGE,
               '-e', 'normal,build', '--target', 'all', '--prefix', 'none', '--format', '{p}',
               *feature_args(features)]
    print('+ ' + ' '.join(command), flush=True)
    tree = subprocess.check_output(command, cwd=root, text=True)
    packages = {line.split()[0] for line in tree.splitlines() if line.strip()}
    if PACKAGE not in packages:
        raise ValueError('Cargo returned no provider dependency graph')
    forbidden = {name for name in packages if name in {'ring', 'openssl-sys', 'boring-sys', 'cc', 'cmake'}
                 or name.startswith('aws-lc-')}
    if forbidden:
        raise ValueError('provider normal/build graph contains forbidden packages: ' + ', '.join(sorted(forbidden)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lint', 'features', 'no-c'))
    parser.add_argument('--target')
    parser.add_argument('--fix', action='store_true')
    parser.add_argument('--all-targets', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    matrix = feature_matrix()
    if args.list:
        print(json.dumps(matrix, indent=2))
        return
    if args.mode == 'no-c':
        # Cargo dependency features are additive. The all row is a superset of
        # every combination, including combinations beyond the compile matrix.
        for name, features in matrix:
            print('Provider no-C graph: ' + name, flush=True)
            no_c(features)
        return
    target = args.target or toolchain.host()
    if not supports_std(target):
        parser.error('the provider requires a std target')
    if args.mode == 'lint':
        for name, features in matrix[2:4]:
            run(['cargo', '+' + toolchain.channel(), 'clippy', '--locked', '-p', PACKAGE,
                 '--target', target, '--all-targets' if args.all_targets else '--lib',
                 *(['--release'] if name == 'native' else []), *feature_args(features),
                 *(['--fix', '--allow-dirty', '--allow-staged'] if args.fix else [])])
        return
    for channel in dict.fromkeys([toolchain.channel(), toolchain.msrv_channel()]):
        for name, features in matrix:
            print(f'Provider features: {channel} / {target} / {name}', flush=True)
            run(['cargo', '+' + channel, 'check', '--locked', '-p', PACKAGE, '--lib',
                 '--target', target, *feature_args(features)])


if __name__ == '__main__':
    main()
