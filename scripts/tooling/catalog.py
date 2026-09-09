#!/usr/bin/env python3
"""Read the tooling catalog and install its verified, platform-specific archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / '.config/tooling.toml'
NATIVE_SOURCE_PLATFORMS = ('riscv64-linux', 's390x-linux', 'powerpc64le-linux')
PROFILING_PLATFORMS = ('aarch64-linux', 'x86_64-linux')
PLATFORMS = (*PROFILING_PLATFORMS, 'aarch64-win', 'x86_64-win', *NATIVE_SOURCE_PLATFORMS)


def read(path=CATALOG):
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)


def rust_channel():
    return read(ROOT / 'rust-toolchain.toml')['toolchain']['channel']


def download(url, destination, checksum):
    if not url.startswith('https://') or len(checksum) != 64:
        raise ValueError(f'invalid pinned download: {url}')
    request = urllib.request.Request(url, headers={'User-Agent': 'rscrypto-tooling'})
    with urllib.request.urlopen(request, timeout=120) as response, Path(destination).open('wb') as output:
        if not response.url.startswith('https://'):
            raise ValueError('download redirected away from HTTPS')
        digest = hashlib.sha256()
        while block := response.read(1024 * 1024):
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != checksum.lower():
        Path(destination).unlink()
        raise ValueError(f'checksum mismatch: {url}')


def unpack(archive, destination):
    destination = Path(destination)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                resolved = (destination / member.filename).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError(f'archive path escapes destination: {member.filename}')
            bundle.extractall(destination)
            if os.name != 'nt':
                for member in bundle.infolist():
                    mode = (member.external_attr >> 16) & 0o777
                    if mode and not member.is_dir():
                        (destination / member.filename).chmod(mode)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter='data')


def install_archive(name, asset, prefix):
    """Keep complete tool distributions; their adjacent libraries are required."""
    prefix = Path(prefix)
    destination = prefix / name / asset['sha256'][:16]
    receipt = destination / '.rscrypto-installed'
    if not receipt.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            stage = Path(temporary)
            archive = stage / 'download'
            download(asset['url'], archive, asset['sha256'])
            unpack(archive, stage / 'unpacked')
            contents = stage / 'unpacked'
            children = list(contents.iterdir())
            if len(children) == 1 and children[0].is_dir():
                contents = children[0]
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(contents), destination)
            receipt.write_text(asset['sha256'] + '\n')
    return destination


def validate(data):
    if not isinstance(data['ci-compat']['workers'], int) or data['ci-compat']['workers'] < 1:
        raise ValueError('ci-compat: workers must be a positive integer')
    if any(asset not in data['x86_64-linux']['assets'] for asset in data['ci-compat']['assets']):
        raise ValueError('ci-compat: missing pinned archive')
    for profile, required in (('ci', {'just', 'cargo-nextest'}),
                              ('ci-policy', {'cargo-deny', 'cargo-audit'}),
                              ('ci-compat', {'just'})):
        if set(data[profile]['cargo']) != required:
            raise ValueError(f'{profile}: incorrect CI tool set')
        if any(tool not in data['cargo'] for tool in data[profile]['cargo']):
            raise ValueError(f'{profile}: missing Cargo tool version')
    for platform in PLATFORMS:
        config = data[platform]
        if 'miri' in config['components']:
            raise ValueError(f'{platform}: Miri belongs to the separate nightly lane')
        required = {'cargo-nextest', 'just', 'ripgrep', 'cargo-audit', 'cargo-deny'}
        if not required <= set(config['cargo']):
            raise ValueError(f'{platform}: missing check/test/bench tools')
        for tool in config['cargo']:
            if tool not in data['cargo']:
                raise ValueError(f'{platform}: no version for {tool}')
        assets = config['assets']
        if 'rustup' not in assets:
            raise ValueError(f'{platform}: missing native rustup archive')
        if platform == 'x86_64-win' and 'nasm' not in assets:
            raise ValueError(f'{platform}: missing NASM for native dependency assembly')
        if platform not in NATIVE_SOURCE_PLATFORMS and not {'cargo-rail', 'cargo-binstall', 'cmake', 'llvm'} <= assets.keys():
            raise ValueError(f'{platform}: missing native tool archives')
        for name, asset in assets.items():
            if not asset['url'].startswith('https://') or not __import__('re').fullmatch('[0-9a-f]{64}', asset['sha256']):
                raise ValueError(f'{platform}: invalid {name} asset')
        profiling = {'samply', 'gungraun-runner'}
        packages = set(data['linux']['packages']) | set(config.get('packages', [])) if platform.endswith('-linux') else set()
        if platform in PROFILING_PLATFORMS:
            if not profiling <= set(config['cargo']) or not {'valgrind', 'linux-tools-generic'} <= packages:
                raise ValueError(f'{platform}: incomplete profiling tools')
        elif profiling & set(config['cargo']) or {'valgrind', 'linux-tools-generic'} & packages:
            raise ValueError(f'{platform}: profiling is not enabled')
        if platform in NATIVE_SOURCE_PLATFORMS and not {'cmake', 'clang', 'libclang-dev'} <= packages:
            raise ValueError(f'{platform}: missing native build prerequisites')


def main():
    data = read()
    command, *args = sys.argv[1:]
    if command == 'get':
        value = data
        for key in args:
            value = value[key]
        if isinstance(value, list):
            print('\n'.join(value))
        elif isinstance(value, dict):
            print(json.dumps(value))
        else:
            print(value)
    elif command == 'json':
        print(json.dumps(data))
    elif command == 'rust-channel':
        print(rust_channel())
    elif command == 'validate':
        validate(data)
        print('Tooling catalog passed')
    elif command == 'install-archive':
        platform, name, prefix = args
        print(install_archive(name, data[platform]['assets'][name], prefix))
    elif command == 'install-archives':
        platform, prefix = args
        for name, asset in data[platform]['assets'].items():
            if name == 'rustup':
                continue
            directory = install_archive(name, asset, prefix)
            print(f'{name}\t{directory}')
    elif command == 'download':
        platform, name, destination = args
        asset = data[platform]['assets'][name]
        download(asset['url'], destination, asset['sha256'])
    else:
        raise ValueError(f'unknown catalog operation: {command}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f'tooling: {error}')
