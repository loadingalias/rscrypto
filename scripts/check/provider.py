#!/usr/bin/env -S uv run python
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


def runtime_matrix():
    return [
        ('tls-x25519-p256-aes', ['getrandom', 'hkdf', 'aes-gcm', 'x25519', 'ecdsa-p256']),
        ('tls-p256-rsa-aes', ['getrandom', 'hkdf', 'aes-gcm', 'p256-ecdh', 'rsa']),
        ('tls-p384-p384-aes', ['getrandom', 'hkdf', 'aes-gcm', 'p384-ecdh', 'ecdsa-p384']),
        ('tls-x25519-ed25519-chacha', ['getrandom', 'hkdf', 'chacha20poly1305', 'x25519', 'ed25519']),
        ('tls-hybrid-p256-aes', ['getrandom', 'hkdf', 'aes-gcm', 'x25519', 'ml-kem', 'ecdsa-p256']),
        ('tls-hybrid-ed25519-chacha', ['getrandom', 'hkdf', 'chacha20poly1305', 'x25519', 'ml-kem', 'ed25519']),
        ('tls-hybrid-mldsa-aes', ['getrandom', 'hkdf', 'aes-gcm', 'x25519', 'ml-kem', 'ml-dsa']),
        ('tls-x25519-mldsa-chacha', ['getrandom', 'hkdf', 'chacha20poly1305', 'x25519', 'ml-dsa']),
        ('quinn-x25519-p256', ['quinn', 'x25519', 'ecdsa-p256']),
        ('quinn-hybrid-mldsa', ['quinn', 'x25519', 'ml-kem', 'ml-dsa', 'ecdsa-p256', 'chacha20poly1305']),
    ]


def feature_matrix(root=ROOT):
    core = tomllib.loads((root / 'Cargo.toml').read_text())
    provider = tomllib.loads((root / PACKAGE / 'Cargo.toml').read_text())
    if provider['package']['rust-version'] != core['package']['rust-version']:
        raise ValueError('the provider rust-version must match rscrypto')
    graph = provider['features']
    for feature, edges in graph.items():
        if feature == 'quinn':
            if edges != ['dep:quinn-proto', 'dep:bytes', 'aes-gcm', 'hkdf', 'getrandom']:
                raise ValueError('quinn integration must retain its fixed Initial/token primitives and backend-free dependencies')
            continue
        # OS entropy is delegated to the same getrandom crate as rscrypto.
        expected = ['rscrypto/' + feature, *(['dep:getrandom'] if feature == 'getrandom' else [])]
        if feature != 'default' and (feature not in core['features'] or edges != expected):
            raise ValueError(f'provider feature {feature} must forward one-to-one to rscrypto')
    native = sorted(set(graph) - {'default', 'portable-only'})
    return [('empty', []), ('default', ['default']), ('native', native),
            ('all', sorted(graph)),
            *[('kx-' + feature, ['getrandom', feature])
              for feature in ('x25519', 'p256-ecdh', 'p384-ecdh')],
            ('kx-hybrid', ['getrandom', 'x25519', 'ml-kem']),
            *[('sign-' + feature, ['getrandom', feature])
              for feature in ('ecdsa-p256', 'ecdsa-p384', 'ed25519', 'rsa', 'ml-dsa')],
            *runtime_matrix(),
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
    parser.add_argument('mode', choices=('lint', 'features', 'runtime', 'no-c'))
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
    if args.mode == 'runtime':
        if target != toolchain.host():
            parser.error('runtime feature checks require execution on the native host')
        for name, features in runtime_matrix():
            for portable in (False, True):
                print(f'Provider runtime: {name} / portable={portable}', flush=True)
                selected = [*features, *(['portable-only'] if portable else [])]
                run(['cargo', '+' + toolchain.channel(), 'nextest', 'run', '--locked',
                     '-p', PACKAGE, '--lib', '--release', *feature_args(selected),
                     '-E', 'test(tests::profiles::) | test(quinn::)', '--success-output', 'final'])
        return
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
