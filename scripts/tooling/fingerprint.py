#!/usr/bin/env -S uv run python
"""Fingerprint what one platform's installation depends on, so a converged host or image can trust it.

usage: fingerprint.py PLATFORM OPERATION

Only the installers run this. It reads their own files, which no other lane needs.
"""
import hashlib
import sys

import catalog

# Cross-build and cross-run installations depend on a target or a transferred
# archive, so they always converge and have no fingerprint.
OPERATIONS = {'linux': ('dev', 'ct-proof', 'ci', 'ci-compat', 'ci-package', 'ci-fuzz', 'ci-miri', 'ci-ct',
                        'ci-ct-full', 'ci-bench'),
              'win': ('dev', 'ci', 'ci-bench', 'ci-ct')}


def inputs(platform, operation):
    """The files one platform's installer reads for an operation, relative to the repository root."""
    family = platform.rsplit('-', 1)[1]
    front_door = f'scripts/tooling/{platform}.' + ('ps1' if family == 'win' else 'sh')
    own = {'linux': ['scripts/tooling/linux.sh', 'scripts/tooling/uv.sh'], 'win': ['scripts/tooling/windows.ps1']}[family]
    # The compatibility and package lanes install their own toolchains and targets.
    lane = {'ci-compat': ['scripts/check/compat.py', '.config/target-matrix.json'],
            'ci-package': ['scripts/check/package.py', 'scripts/check/compat.py']}.get(operation, [])
    # Every installer provisions the Python that .python-version names, through uv.
    return sorted([front_door, 'scripts/tooling/fingerprint.py', 'scripts/tooling/catalog.py',
                   'scripts/lib/toolchain.py', '.config/tooling.toml', 'rust-toolchain.toml', '.python-version',
                   *own, *lane])


def fingerprint(data, platform, operation):
    """Digest the installer's inputs, the operation, the required OS release and the platform.

    An edit to any other file, such as a test or another platform's installer,
    leaves the platform's hosts and images current.
    """
    family = platform.rsplit('-', 1)[1]
    if operation not in OPERATIONS[family]:
        raise ValueError(f'{platform} has no reusable {operation} installation')
    digest = hashlib.sha256()
    for name in inputs(platform, operation):
        content = (catalog.ROOT / name).read_bytes()
        digest.update(f'{name}\0{len(content)}\0'.encode() + content)
    if operation in ('ci-compat', 'ci-package'):
        # Of the manifest, those lanes read only the MSRV.
        msrv = catalog.read(catalog.ROOT / 'Cargo.toml')['package']['rust-version']
        digest.update(f'Cargo.toml\0rust-version\0{msrv}\0'.encode())
    if family == 'linux':
        release = data['linux-ci' if operation.startswith('ci') else 'linux']['ubuntu']
    else:
        release = data[platform]['os']
    digest.update(f'{platform}\0{operation}\0{release}'.encode())
    return digest.hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__.split('\n\n')[1])
    try:
        print(fingerprint(catalog.read(), *sys.argv[1:]))
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f'tooling: {error}')
