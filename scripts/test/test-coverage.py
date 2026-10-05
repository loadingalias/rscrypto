#!/usr/bin/env python3
"""Run host tests and corpus replay once, then report their combined Rust coverage."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/bench'), str(ROOT / 'scripts/lib')]
import evidence as execution_evidence
import evidence_bundle as bundle


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


def host_identity():
    if sys.platform == 'darwin':
        model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    elif sys.platform.startswith('linux'):
        rows = (Path('/proc/cpuinfo').read_text(errors='replace').splitlines()
                if Path('/proc/cpuinfo').is_file() else [])
        model = next((value.strip() for key, _, value in (row.partition(':') for row in rows)
                      if key.strip() in {'model name', 'Hardware', 'Processor'}), platform.processor())
    else:
        model = platform.processor()
    return {
        'platform': platform.platform(),
        'cpu': {'architecture': platform.machine(), 'model': model},
    }


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


def file_identity(path):
    return {'bytes': path.stat().st_size, 'sha256': bundle.digest(path)}


def verify_objects(objects):
    for path, identity in objects.items():
        if file_identity(Path(path)) != identity:
            raise RuntimeError(f'test executable changed during coverage collection: {path}')


def corpus_inventory(root, manifest, mode):
    """Hash the replay candidates; the Rust replay helper records actual consumption."""
    corpus = manifest.parent / 'corpus'
    if not corpus.is_dir():
        return {}
    if mode not in ('committed', 'local'):
        raise RuntimeError('RSCRYPTO_FUZZ_CORPUS must be committed or local')
    seeds = [root / name for name in (root / 'fuzz/committed-seeds.txt').read_text().splitlines()
             if (root / name).is_relative_to(corpus)]
    paths = seeds if mode == 'committed' else [
        path for directory in sorted({path.parent for path in seeds})
        for path in sorted(directory.iterdir()) if path.is_file() or path.is_symlink()]
    return {str(path): file_identity(path) for path in paths}


def execution_inventory(listing, directory, root, corpus):
    """After nextest succeeds, reconcile test launches and profiles with its full list."""
    expected = {}
    tests = []
    for binary_id, binary in listing['rust-suites'].items():
        if binary['status'] != 'listed':
            raise RuntimeError(f'test binary was not listed: {binary_id}')
        for name, case in binary['testcases'].items():
            row = {'binary_id': binary_id, 'test': name, 'ignored': case['ignored'],
                   'filter_match': case['filter-match'], 'executed': False}
            tests.append(row)
            if case['filter-match']['status'] == 'matches':
                expected[binary_id, name] = (row, str(Path(binary['binary-path']).resolve()))
    if not expected:
        raise RuntimeError('coverage suite has no selected tests')
    profiles = []
    consumed = set()
    for result in sorted(directory.glob('test-*/execution.json')):
        record = json.loads(result.read_text())
        key = record['binary_id'], record['test']
        if key not in expected:
            raise RuntimeError(f'unexpected or duplicate test execution: {key}')
        row, binary = expected.pop(key)
        if record['binary'] != binary:
            raise RuntimeError(f'mismatched executable: {key}')
        raw = sorted(result.parent.glob('*.profraw'))
        if not raw or any(path.stat().st_size == 0 for path in raw):
            raise RuntimeError(f'test produced no usable coverage profiles: {key}')
        row.update(executed=True, raw_profile_count=len(raw))
        receipt = result.parent / 'corpus-inputs'
        if receipt.exists():
            payload = receipt.read_bytes()
            if not payload or not payload.endswith(b'\0'):
                raise RuntimeError('invalid coverage corpus receipt')
            names = payload[:-1].decode('utf-8').split('\0')
            if any(name not in corpus for name in names):
                raise RuntimeError('replayed corpus input was not present before execution')
            row['corpus_inputs'] = [Path(name).relative_to(root).as_posix() for name in names]
            consumed.update(names)
        profiles.extend(raw)
    if expected:
        raise RuntimeError(f'missing test executions: {sorted(expected)}')
    if consumed != set(corpus):
        raise RuntimeError('replayed corpus inputs differ from the selected inventory')
    return tests, profiles


def collect_suite(root, directory, env, manifest, flags, config):
    directory.mkdir()
    corpus = corpus_inventory(root, manifest, env.get('RSCRYPTO_FUZZ_CORPUS', 'committed'))
    suite_env = instrument(root, env, manifest)
    # Cargo runner arrays preserve paths with spaces on every supported host.
    runner = [sys.executable, str(ROOT / 'scripts/test/coverage_run.py'), str(directory)]
    runner_config = ['--config', f'target.{env["CARGO_BUILD_TARGET"]}.runner={json.dumps(runner)}']
    config = [*config, '--user-config-file', 'none', *runner_config]
    metadata = directory / 'binaries.json'
    metadata.write_text(capture([
        'cargo', 'nextest', 'list', '--locked', '--manifest-path', str(manifest),
        *flags, *config, '--message-format', 'json', '--list-type', 'binaries-only'], root, suite_env))
    binaries = json.loads(metadata.read_text())['rust-binaries']
    objects = {binary['binary-path']: file_identity(Path(binary['binary-path'])) for binary in binaries.values()}
    if not objects:
        raise RuntimeError(f'no test executables found for {manifest}')
    reuse = ['--manifest-path', str(manifest), '--binaries-metadata', str(metadata), *config]
    listing = json.loads(capture(['cargo', 'nextest', 'list', *reuse,
                                  '--message-format', 'json'], root, suite_env))
    if (set(listing['rust-suites']) != set(binaries)
            or any(binary['binary-path'] != binaries[key]['binary-path']
                   for key, binary in listing['rust-suites'].items())):
        raise RuntimeError('test discovery changed the executable inventory')
    threads = ['--test-threads', env['RSCRYPTO_TEST_THREADS']] if env.get('RSCRYPTO_TEST_THREADS') else []
    run(['cargo', 'nextest', 'run', *reuse, *threads, '--no-tests', 'fail', '--retries', '0'], root, suite_env)
    verify_objects(objects)
    tests, profiles = execution_inventory(listing, directory, root, corpus)
    if corpus != corpus_inventory(root, manifest, env.get('RSCRYPTO_FUZZ_CORPUS', 'committed')):
        raise RuntimeError('corpus changed during coverage collection')
    return objects, tests, profiles, {Path(path).relative_to(root).as_posix(): identity
                                      for path, identity in corpus.items()}


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
    objects = {}
    profiles = []
    collected = []
    for index, (name, manifest, flags) in enumerate(suites):
        print(f'\nCoverage: {name}', flush=True)
        config = ['--config-file', str(root / '.config/nextest.toml')] if index < 2 else []
        config += ['-P', 'default']
        binaries, tests, raw, corpus = collect_suite(root, work / f'suite-{index}', env, manifest, flags, config)
        for path, identity in binaries.items():
            if path in objects and objects[path] != identity:
                raise RuntimeError(f'test executable reused with different contents: {path}')
            objects[path] = identity
        profiles.extend(raw)
        collected.append({
            'name': name,
            'manifest': manifest.relative_to(root).as_posix(),
            'cargo_arguments': flags,
            'binaries': binaries,
            'tests': tests,
            'raw_profile_count': len(raw),
            'profile_paths': [str(path.relative_to(work)) for path in raw],
            'corpus_inputs': corpus,
        })
    return objects, collected, profiles


def verify_mappings(cov, args, root, work, env):
    # Rust emits zero-hash unused-function placeholders. LLVM can warn when another
    # binary supplies the real mapping. Accept these only if that mapping is loaded.
    # https://github.com/rust-lang/rust/blob/f7575a9da8e4a4fca3b5668d5a2ea7476db44b3f/compiler/rustc_codegen_llvm/src/coverageinfo/mapgen/covfun.rs
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


def coverage_arguments(root, work, objects, profiles, llvm_bin, env):
    verify_objects(objects)
    if not profiles:
        raise RuntimeError('tests produced no coverage profiles')
    profile_list = work / 'profiles.txt'
    profile_list.write_text(''.join(f'{path}\n' for path in profiles))
    merged = work / 'merged.profdata'
    result = run([str(llvm_bin / ('llvm-profdata.exe' if os.name == 'nt' else 'llvm-profdata')),
                  'merge', '-sparse', '-f', str(profile_list), '-o', str(merged)], root, env, capture_errors=True)
    if result.stderr:
        raise RuntimeError(result.stderr.strip())
    # Explicit objects are essential: root-only discovery misses independent replay binaries.
    args = [f'-instr-profile={merged}', *[f'-object={path}' for path in objects]]
    cov = str(llvm_bin / ('llvm-cov.exe' if os.name == 'nt' else 'llvm-cov'))
    expected = verify_mappings(cov, args, root, work, env)
    args += ['--sources', *map(str, sorted((root / 'src').rglob('*.rs')))]

    return cov, args, expected


def export(cov, args, expected, command, root, env, stream):
    result = run([cov, *command, *args], root, env, output=stream, capture_errors=True)
    if result.stderr != expected:
        raise RuntimeError(result.stderr or 'coverage diagnostics changed between exports')


def export_lcov(root, work, cov, args, expected, env):
    with (work / 'total.lcov').open('w') as stream:
        export(cov, args, expected, ['export', '-format=lcov'], root, env, stream)
    with (work / 'total.lcov').open() as stream:
        if not any(line.startswith('DA:') for line in stream):
            raise RuntimeError('coverage export contains no source lines')


def report(root, work, output, objects, profiles, llvm_bin, env, provenance):
    contributions = work / 'suites'
    contributions.mkdir()
    for index, suite in enumerate(provenance.get('suites', [])):
        directory = work / f'suite-report-{index}'
        directory.mkdir()
        raw = [work / path for path in suite.pop('profile_paths')]
        cov, args, expected = coverage_arguments(root, directory, suite['binaries'], raw, llvm_bin, env)
        export_lcov(root, directory, cov, args, expected, env)
        name = f'suites/{index}.lcov'
        (directory / 'total.lcov').rename(work / name)
        suite['coverage'] = name
    cov, args, expected = coverage_arguments(root, work, objects, profiles, llvm_bin, env)
    (work / 'objects.json').write_text(json.dumps(objects, indent=2) + '\n')
    export_lcov(root, work, cov, args, expected, env)
    with (work / 'SUMMARY.txt').open('w') as stream:
        export(cov, args, expected, ['report'], root, env, stream)
    export(cov, args, expected, ['show', '-format=html', f'-output-dir={work / "html"}'],
           root, env, subprocess.DEVNULL)
    # Publish only after every test and export succeeds; failed runs leave no success report.
    if bundle.source_identity(root) != provenance['source']:
        raise RuntimeError('source changed during coverage collection')
    verify_objects(objects)
    totals = [line for line in (work / 'SUMMARY.txt').read_text().splitlines() if line.startswith('TOTAL')]
    if len(totals) != 1:
        raise RuntimeError('coverage summary must contain exactly one TOTAL row')
    total = totals[0]
    provenance.update({
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_profile_count': len(profiles),
        'object_count': len(objects),
        'summary_total': total,
        'artifacts': {
            name: file_identity(work / name)
            for name in ('total.lcov', 'SUMMARY.txt', 'merged.profdata', 'objects.json', 'html/index.html',
                         *[suite['coverage'] for suite in provenance.get('suites', [])])
        },
    })
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    # The provenance file is the completion marker and is always published last.
    try:
        for name in ('merged.profdata', 'objects.json', 'total.lcov', 'SUMMARY.txt', 'html', 'suites', 'provenance.json'):
            (work / name).rename(output / name)
    except BaseException:
        clear_reports(output)
        raise
    print(total)
    print(f'Coverage: {output / "html/index.html"}\nLCOV: {output / "total.lcov"}')


def clear_reports(output):
    for name in ('total.lcov', 'nextest.lcov', 'fuzz.lcov', 'SUMMARY.md', 'SUMMARY.txt', 'html',
                 'merged.profdata', 'objects.json', 'suites', 'provenance.json'):
        path = output / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = ROOT
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
        clear_reports(output)
        env = os.environ.copy()
        env['RUSTUP_TOOLCHAIN'] = tomllib.loads((root / 'rust-toolchain.toml').read_text())['toolchain']['channel']
        rustc_version = capture(['rustc', '--version', '--verbose'], root, env)
        if 'dev' in rustc_version.splitlines()[0]:
            raise RuntimeError('coverage requires the pinned development toolchain, not a local compiler build')
        host = next(line.removeprefix('host: ') for line in rustc_version.splitlines() if line.startswith('host: '))
        sysroot = Path(capture(['rustc', '--print', 'sysroot'], root, env).strip())
        llvm_bin = sysroot / 'lib/rustlib' / host / 'bin'
        for tool in ('llvm-cov', 'llvm-profdata'):
            if not (llvm_bin / (tool + ('.exe' if os.name == 'nt' else ''))).is_file():
                raise RuntimeError('install llvm-tools-preview for the development toolchain: rustup component add llvm-tools-preview')
        llvm_cov_version = capture(['cargo', 'llvm-cov', '--version'], root, env).strip()
        nextest_version = capture(['cargo', 'nextest', '--version'], root, env).strip()
        with tempfile.TemporaryDirectory(prefix='.run-', dir=output) as directory:
            work = Path(directory)
            build = root / 'target/coverage'
            env.update(CARGO_TARGET_DIR=str(build), CARGO_BUILD_BUILD_DIR=str(build), CARGO_BUILD_TARGET=host,
                       CARGO_LLVM_COV_TARGET_DIR=str(build), CARGO_LLVM_COV_BUILD_DIR=str(build))
            env['LLVM_PROFILE_FILE'] = str(work / '%p-%m.profraw')
            source = bundle.source_identity(root)
            objects, suites, profiles = collect(root, work, env)
            provenance = {
                'schema': 2,
                'kind': 'rscrypto.coverage',
                'command': ['just', 'test-coverage'],
                'source': source,
                'target': host,
                'host': host_identity(),
                'tools': {
                    'rustc': rustc_version.strip(),
                    'cargo-llvm-cov': llvm_cov_version,
                    'cargo-nextest': nextest_version,
                },
                'execution_environment': execution_evidence.collect(env),
                'llvm_profile_pattern': 'suite-N/test-*/%p-%m.profraw',
                'suites': suites,
            }
            report(root, work, output, objects, profiles, llvm_bin, env, provenance)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f'coverage failed: {error}')
