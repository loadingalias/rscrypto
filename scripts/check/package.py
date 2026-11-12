#!/usr/bin/env -S uv run python
"""Verify the packages Cargo will publish, then consume their unpacked contents."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compat import ROOT, boundary_features, read, toolchain

PROVIDER = 'rscrypto-rustls'


def git_patches(manifest):
    """Return `--config` arguments that patch each Git-pinned dependency to its pin.

    Packaging keeps only a dependency's version, so the archive would otherwise
    resolve the crates.io release of that version. A registry pin needs no patch.
    """
    arguments = []
    for table in (manifest, *manifest.get('target', {}).values()):
        for kind in ('dependencies', 'dev-dependencies', 'build-dependencies'):
            for key, dependency in table.get(kind, {}).items():
                if isinstance(dependency, dict) and 'git' in dependency:
                    name = dependency.get('package', key)
                    for field in ('git', 'rev', 'branch', 'tag'):
                        if field in dependency:
                            arguments += ['--config', f'patch.crates-io.{name}.{field}={json.dumps(dependency[field])}']
    return arguments


def check_contents(package, source):
    """Require each archive file to be the source file of the same name; Cargo writes the rest."""
    for path in sorted(path for path in package.rglob('*') if path.is_file()):
        name = path.relative_to(package).as_posix()
        if name in ('Cargo.toml', 'Cargo.lock', '.cargo_vcs_info.json'):
            continue
        origin = source / ('Cargo.toml' if name == 'Cargo.toml.orig' else name)
        if not origin.is_file() or origin.read_bytes() != path.read_bytes():
            raise SystemExit(f'{package.name} holds {name}, which is not {origin.relative_to(ROOT)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--allow-dirty', action='store_true', help='local validation of uncommitted work')
    args = parser.parse_args()
    manifest = read('Cargo.toml')
    channels = list(dict.fromkeys([toolchain.channel(), toolchain.msrv_channel()]))
    if args.install:
        for channel in channels:
            subprocess.run(['rustup', 'toolchain', 'install', channel, '--profile', 'minimal',
                            '--target', 'thumbv6m-none-eabi'], check=True)
        return
    provider = read(PROVIDER + '/Cargo.toml')
    patches = git_patches(provider)
    output = ROOT / 'target/package-evidence'
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'CARGO_TARGET_DIR': str(output / 'build'), 'CARGO_RAIL_CACHE': 'off'}

    def run(command, cwd=ROOT):
        print('+ ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=cwd, env=environment, check=True)

    run(['just', 'test-examples'])
    # One invocation, so the provider verifies against the packaged rscrypto.
    run(['cargo', '+' + channels[0], *patches, 'package', '-p', 'rscrypto', '-p', PROVIDER, '--locked',
         *(['--allow-dirty'] if args.allow_dirty else [])])
    name = manifest['package']['name'] + '-' + manifest['package']['version']
    provider_name = PROVIDER + '-' + provider['package']['version']
    with tempfile.TemporaryDirectory(prefix='rscrypto-consumers-') as temporary:
        directory = Path(temporary)
        for archive in (name, provider_name):
            with tarfile.open(output / 'build/package' / (archive + '.crate')) as source:
                source.extractall(directory, filter='data')
        package = directory / name
        for required in ('Cargo.toml', 'src/lib.rs', 'LICENSE', 'README.md'):
            if not (package / required).is_file():
                raise SystemExit('package missing ' + required)
        (output / 'contents.txt').write_text('\n'.join(sorted(str(p.relative_to(package)) for p in package.rglob('*') if p.is_file())) + '\n')
        # The provider's library tests, built from its archive against the packaged rscrypto.
        check_contents(directory / provider_name, ROOT / PROVIDER)
        provider_features = sorted(set(provider['features']) - {'default', 'portable-only'})
        packaged_rscrypto = 'patch.crates-io.rscrypto.path=' + json.dumps(str(package))
        run(['cargo', '+' + channels[0], *patches, '--config', packaged_rscrypto, 'test', '--release', '--lib',
             '--no-default-features', '--features', ','.join(provider_features)], directory / provider_name)
        for boundary in ('std', 'core', 'alloc'):
            consumer = directory / boundary
            (consumer / 'src').mkdir(parents=True)
            features = ['full', 'std'] if boundary == 'std' else boundary_features(manifest['features'], boundary)
            (consumer / 'Cargo.toml').write_text(
                '[package]\nname = "package_consumer"\nversion = "0.0.0"\nedition = "2024"\n'
                '[dependencies.rscrypto]\npath = ' + json.dumps(str(package)) + '\n'
                'default-features = false\nfeatures = ' + json.dumps(features) + '\n')
            vector = ('use rscrypto::{Digest, Sha256};\n'
                      'pub fn verify() { assert_eq!(<Sha256 as Digest>::digest(b"abc"), '
                      '[0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,'
                      '0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad]); }\n')
            (consumer / 'src/lib.rs').write_text(('#![no_std]\n' if boundary != 'std' else '') + vector)
            (consumer / 'src/main.rs').write_text('fn main() { package_consumer::verify(); }\n')
            run(['cargo', '+' + channels[-1], 'generate-lockfile'], consumer)
            for channel in channels:
                run(['cargo', '+' + channel, 'run', '--locked', '--release'], consumer)
                if boundary != 'std':
                    run(['cargo', '+' + channel, 'check', '--locked', '--lib', '--target', 'thumbv6m-none-eabi'], consumer)
    print('Package verification and external consumers passed', flush=True)


if __name__ == '__main__':
    main()
