#!/usr/bin/env python3
"""Run host tests and corpus replay once, then report their combined Rust coverage."""

import argparse
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib


def run(args, root, env, *, output=None, capture_errors=False):
    command = shlex.join(map(str, args))
    print('+ ' + (command if len(command) <= 300 else command[:300] + ' …'), flush=True)
    result = subprocess.run(args, cwd=root, env=env, check=True, stdout=output,
                            stderr=subprocess.PIPE if capture_errors else None, text=True)
    if result.stderr is not None:
        result.stderr = re.sub(r'\x1b\[[0-9;]*m', '', result.stderr)
    return result


def capture(args, root, env):
    return run(args, root, env, output=subprocess.PIPE).stdout


def instrument(root, env, manifest):
    flags = ['--manifest-path', str(manifest)]
    if manifest.parent != root:
        package = tomllib.loads((root / 'Cargo.toml').read_text())['package']['name']
        flags += ['--dep-coverage', package]
    exports = capture(['cargo', 'llvm-cov', 'show-env', '--sh', '--no-cfg-coverage', *flags], root, env)
    result = env.copy()
    for line in exports.splitlines():
        words = shlex.split(line)
        if len(words) != 2 or words[0] != 'export':
            raise RuntimeError(f'unexpected cargo-llvm-cov environment output: {line}')
        key, value = words[1].split('=', 1)
        result[key] = value
    result['LLVM_PROFILE_FILE'] = env['LLVM_PROFILE_FILE']
    return result


def collect(root, work, env):
    """Instrument each workspace and rscrypto, including code inlined into replay tests."""
    features = tomllib.loads((root / 'Cargo.toml').read_text())['features']
    native = sorted(set(features) - {'portable-only'})
    if any('portable-only' in features[name] for name in native):
        raise RuntimeError('native test features indirectly enable portable-only')
    suites = [
        ('native', root / 'Cargo.toml', ['--workspace', '--no-default-features', '--features', ','.join(native)]),
        ('portable', root / 'Cargo.toml', ['--workspace', '--all-features']),
    ]
    manifests = [root / 'fuzz/Cargo.toml', *sorted(root.glob('fuzz-packages/*/Cargo.toml'))]
    suites.extend((str(p.parent.relative_to(root)), p, ['--all-features', '--test', 'corpus_replay']) for p in manifests)
    objects = set()
    threads = ['--test-threads', env['RSCRYPTO_TEST_THREADS']] if env.get('RSCRYPTO_TEST_THREADS') else []
    for index, (name, manifest, flags) in enumerate(suites):
        print(f'\nCoverage: {name}', flush=True)
        config = ['--config-file', str(root / '.config/nextest.toml')] if index < 2 else []
        config += ['-P', 'default']
        suite_env = instrument(root, env, manifest)
        metadata = work / f'binaries-{index}.json'
        listing = capture(['cargo', 'nextest', 'list', '--locked', '--manifest-path', str(manifest),
                           *flags, *config, '--message-format', 'json', '--list-type', 'binaries-only'], root, suite_env)
        metadata.write_text(listing)
        binaries = json.loads(listing)['rust-binaries'].values()
        paths = {binary['binary-path'] for binary in binaries}
        if not paths:
            raise RuntimeError(f'no test executables found for {name}')
        # Nextest reuses the exact binaries whose coverage mappings we will export.
        run(['cargo', 'nextest', 'run', '--manifest-path', str(manifest),
             '--binaries-metadata', str(metadata), *config, *threads], root, suite_env)
        objects.update(paths)
    return sorted(objects)


def verify_mappings(cov, args, root, work, env):
    # Rust emits zero-hash unused-function placeholders. LLVM can warn when another
    # binary supplies the real mapping. Accept these only if that mapping is loaded.
    # https://github.com/rust-lang/rust/blob/1.98.1/compiler/rustc_codegen_llvm/src/coverageinfo/mapgen/covfun.rs
    mapping = work / 'mappings.json'
    with mapping.open('w') as stream:
        result = run([cov, 'export', *args, '-dump', '-num-threads=1', '-skip-expansions',
                      '-skip-branches'], root, env, output=stream, capture_errors=True)
    missing = []
    expected = ''
    count = 0
    for line in result.stderr.splitlines():
        if match := re.fullmatch(r"hash-mismatch: No profile record found for '(.*)' with hash = (\S+)", line):
            if match[2] != '0x0':
                raise RuntimeError(line)
            missing.append(match[1])
        elif match := re.fullmatch(r'warning: (\d+) functions have mismatched data', line):
            expected = line + '\n'
            count = int(match[1])
        else:
            raise RuntimeError(line)
    text = mapping.read_text()
    # LLVM's -dump writes diagnostic lines before its JSON document.
    data = json.loads(text[text.index('{"data":'):])
    names = {function['name'] for unit in data['data'] for function in unit['functions']}
    if len(missing) != count or not set(missing) <= names:
        raise RuntimeError('coverage contains function mismatches without matching loaded mappings')
    return expected


def report(root, work, output, objects, llvm_bin, env):
    profiles = sorted(work.glob('*.profraw'))
    if not profiles:
        raise RuntimeError('tests produced no coverage profiles')
    profile_list = work / 'profiles.txt'
    profile_list.write_text(''.join(f'{path}\n' for path in profiles))
    merged = output / 'merged.profdata'
    (output / 'objects.json').write_text(json.dumps(objects))
    result = run([str(llvm_bin / ('llvm-profdata.exe' if os.name == 'nt' else 'llvm-profdata')),
                  'merge', '-sparse', '-f', str(profile_list), '-o', str(merged)], root, env, capture_errors=True)
    if result.stderr:
        raise RuntimeError(result.stderr.strip())
    # Explicit objects are essential: root-only discovery misses independent replay binaries.
    args = [f'-instr-profile={merged}', *[f'-object={path}' for path in objects]]
    cov = str(llvm_bin / ('llvm-cov.exe' if os.name == 'nt' else 'llvm-cov'))
    expected = verify_mappings(cov, args, root, work, env)
    args += ['--sources', *map(str, sorted((root / 'src').rglob('*.rs')))]

    def export(command, stream):
        result = run([cov, *command, *args], root, env, output=stream, capture_errors=True)
        if result.stderr != expected:
            raise RuntimeError(result.stderr or 'coverage diagnostics changed between exports')

    with (work / 'total.lcov').open('w') as stream:
        export(['export', '-format=lcov'], stream)
    with (work / 'total.lcov').open() as stream:
        if not any(line.startswith('DA:') for line in stream):
            raise RuntimeError('coverage export contains no source lines')
    with (work / 'SUMMARY.txt').open('w') as stream:
        export(['report'], stream)
    export(['show', '-format=html', f'-output-dir={work / "html"}'], subprocess.DEVNULL)
    # Publish only after every test and export succeeds; failed runs leave no success report.
    for name in ('total.lcov', 'SUMMARY.txt', 'html'):
        (work / name).rename(output / name)
    print(next(line for line in (output / 'SUMMARY.txt').read_text().splitlines() if line.startswith('TOTAL')))
    print(f'Coverage: {output / "html/index.html"}\nLCOV: {output / "total.lcov"}')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['RUSTUP_TOOLCHAIN'] = tomllib.loads((root / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    version = capture(['rustc', '--version', '--verbose'], root, env)
    if 'nightly' in version or 'dev' in version.splitlines()[0]:
        raise RuntimeError('coverage requires a stable development toolchain')
    host = next(line.removeprefix('host: ') for line in version.splitlines() if line.startswith('host: '))
    sysroot = Path(capture(['rustc', '--print', 'sysroot'], root, env).strip())
    llvm_bin = sysroot / 'lib/rustlib' / host / 'bin'
    for tool in ('llvm-cov', 'llvm-profdata'):
        if not (llvm_bin / (tool + ('.exe' if os.name == 'nt' else ''))).is_file():
            raise RuntimeError('install llvm-tools-preview for the development toolchain: rustup component add llvm-tools-preview')
    run(['cargo', 'llvm-cov', '--version'], root, env)
    run(['cargo', 'nextest', '--version'], root, env)
    output = root / 'coverage'
    output.mkdir(exist_ok=True)
    # Serialize this command so another run cannot overwrite its binaries or reports.
    with (output / '.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('total.lcov', 'nextest.lcov', 'fuzz.lcov', 'SUMMARY.md', 'SUMMARY.txt', 'html', 'merged.profdata', 'objects.json'):
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix='.run-', dir=output) as directory:
            work = Path(directory)
            build = root / 'target/coverage'
            env.update(CARGO_TARGET_DIR=str(build), CARGO_BUILD_BUILD_DIR=str(build), CARGO_BUILD_TARGET=host,
                       CARGO_LLVM_COV_TARGET_DIR=str(build), CARGO_LLVM_COV_BUILD_DIR=str(build))
            env['LLVM_PROFILE_FILE'] = str(work / '%p-%m.profraw')
            objects = collect(root, work, env)
            report(root, work, output, objects, llvm_bin, env)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f'coverage failed: {error}')
