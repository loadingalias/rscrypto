#!/usr/bin/env -S uv run python
"""Measure the linked size of a minimal rustls client on each top-secret host target."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import toolchain

MANIFEST = ROOT / 'tools/provider-size/Cargo.toml'
BINARY = 'rscrypto-provider-size'
# top-secret's hosts. Apple builds natively; the others link with Zig or xwin.
TARGETS = {
    'aarch64-apple-darwin': ['build'],
    'x86_64-unknown-linux-gnu': ['zigbuild'],
    'aarch64-unknown-linux-gnu': ['zigbuild'],
    'x86_64-unknown-linux-musl': ['zigbuild'],
    'aarch64-unknown-linux-musl': ['zigbuild'],
    'x86_64-pc-windows-msvc': ['xwin', 'build'],
}
VARIANTS = {'ClientProvider': [], 'default_provider': ['--features', 'default-provider']}


def llvm_tool(name):
    sysroot = subprocess.check_output(['rustc', '+' + toolchain.channel(), '--print', 'sysroot'], text=True).strip()
    return str(Path(sysroot) / 'lib/rustlib' / toolchain.host() / 'bin' / name)


def packages(features):
    tree = subprocess.check_output(
        ['cargo', '+' + toolchain.channel(), 'tree', '--locked', '--manifest-path', str(MANIFEST), '-e', 'normal,build',
         '--target', 'all', '--prefix', 'none', '--format', '{p}', *features], cwd=ROOT, text=True)
    return sorted({line.split()[0] for line in tree.splitlines() if line.strip()} - {BINARY})


def measure(target, command, features, output):
    subprocess.run(['cargo', '+' + toolchain.channel(), *command, '--locked', '--release', '--manifest-path',
                    str(MANIFEST), '--target', target, *features], cwd=ROOT, check=True)
    suffix = '.exe' if 'windows' in target else ''
    built = MANIFEST.parent / 'target' / target / 'release' / (BINARY + suffix)
    linked = output / f'{target}-{"default" if features else "client"}{suffix}'
    shutil.copyfile(built, linked)
    stripped = linked.with_name(linked.name + '.stripped')
    subprocess.run([llvm_tool('llvm-strip'), '-o', str(stripped), str(linked)], check=True)
    return {'linked': linked.stat().st_size, 'stripped': stripped.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory for binaries and results')
    parser.add_argument('--target', action='append', choices=sorted(TARGETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    results = {'graph': {name: packages(features) for name, features in VARIANTS.items()}, 'targets': {}}
    for target in args.target or TARGETS:
        results['targets'][target] = {name: measure(target, TARGETS[target], features, args.output)
                                      for name, features in VARIANTS.items()}
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    for name, graph in results['graph'].items():
        print(f'{name}: {len(graph)} packages: {" ".join(graph)}')
    print('| Target | ClientProvider | default_provider() | Saved |')
    print('|---|---:|---:|---:|')
    for target, sizes in results['targets'].items():
        client, default = sizes['ClientProvider']['stripped'], sizes['default_provider']['stripped']
        print(f'| {target} | {client:,} | {default:,} | {default - client:,} ({(default - client) / default:.1%}) |')


if __name__ == '__main__':
    main()
