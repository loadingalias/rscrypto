#!/usr/bin/env -S uv run python
"""Build and cache the pinned OpenSSL peer that interop tests and fixture generators run.

The peer is an independent comparison implementation outside every shipped graph. It is built
once per host from the release tarball, whose SHA-256 both the OpenSSL GitHub release and the
Homebrew formula (`openssl@4` at homebrew-core `b07caeb2bf`) publish. The build is static, so the
binary needs no library path. Print the binary with `python3 scripts/lib/openssl_peer.py`.
"""

import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = '4.0.3'
URL = f'https://github.com/openssl/openssl/releases/download/openssl-{VERSION}/openssl-{VERSION}.tar.gz'
SHA256 = '325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9'


def cache_root():
    """`RSCRYPTO_OPENSSL_CACHE`, else the XDG cache, else `~/.cache`."""
    configured = os.environ.get('RSCRYPTO_OPENSSL_CACHE')
    if configured:
        return Path(configured)
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'rscrypto'


def prefix(root=None):
    host = f'{platform.system()}-{platform.machine()}'.lower()
    return (root or cache_root()) / f'openssl-{VERSION}-{host}'


def download(destination, url=URL, expected=SHA256):
    """Fetch the tarball into `destination` and refuse any other content."""
    with urllib.request.urlopen(url, timeout=120) as response, destination.open('wb') as output:
        shutil.copyfileobj(response, output)
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != expected:
        destination.unlink()
        raise ValueError(f'OpenSSL {VERSION} tarball SHA-256 is {actual}, expected {expected}')


def build(target, fetch=download):
    """Build the static `openssl` binary into `target`, atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='openssl-peer-', dir=target.parent) as temporary:
        work = Path(temporary)
        archive = work / f'openssl-{VERSION}.tar.gz'
        fetch(archive)
        with tarfile.open(archive) as source:
            source.extractall(work, filter='data')
        tree = work / f'openssl-{VERSION}'
        # The binary records its prefix and configuration directory, so configure the final
        # location and stage the install under DESTDIR.
        destdir = work / 'destdir'
        staged = destdir / target.relative_to(target.anchor)
        log = target.parent / f'openssl-{VERSION}-build.log'
        with log.open('wb') as output:
            def run(*command):
                subprocess.run(command, cwd=tree, check=True, stdout=output, stderr=subprocess.STDOUT)
            run('perl', './Configure', f'--prefix={target}', f'--openssldir={target}/ssl', '--libdir=lib',
                'no-shared', 'no-tests', 'no-docs')
            run('make', f'-j{os.cpu_count() or 1}')
            run('make', f'DESTDIR={destdir}', 'install_sw', 'install_ssldirs')
        version = subprocess.check_output([staged / 'bin/openssl', 'version'], text=True)
        if not version.startswith(f'OpenSSL {VERSION} '):
            raise RuntimeError(f'built OpenSSL reports {version.strip()}')
        staged.rename(target)


def binary(root=None, fetch=download):
    """Return the pinned `openssl` binary, building it on first use."""
    target = prefix(root)
    executable = target / 'bin/openssl'
    if executable.exists():
        return executable
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / f'openssl-{VERSION}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not executable.exists():
            print(f'Building OpenSSL {VERSION} into {target} (once per host)', file=sys.stderr, flush=True)
            build(target, fetch)
    return executable


if __name__ == '__main__':
    print(binary())
