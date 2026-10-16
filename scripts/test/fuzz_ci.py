#!/usr/bin/env -S uv run python
"""Plan bounded CI fuzzing and retain corpus and source-coverage evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import evidence_bundle as bundle

BUDGET = 1800
CONCURRENCY = 8
DAILY_TARGETS = 8
NAME = re.compile(r'[a-z0-9_]+')


def inventory(root):
    result = []
    for manifest in [root / 'fuzz/Cargo.toml', *sorted(root.glob('fuzz-packages/*/Cargo.toml'))]:
        data = tomllib.loads(manifest.read_text())
        if not data['package'].get('metadata', {}).get('cargo-fuzz'):
            continue
        names = [entry['name'] for entry in data.get('bin', [])]
        if not names or len(names) != len(set(names)) or any(not NAME.fullmatch(name) for name in names):
            raise ValueError(f'invalid fuzz target inventory: {manifest}')
        result.extend({'package': manifest.parent.relative_to(root).as_posix(), 'target': name}
                      for name in sorted(names))
    if not result:
        raise ValueError('no fuzz targets found')
    return result


def affected(rows, paths):
    """Only exact harness/seed edits and documentation can narrow the selection.

    Product, shared harness, dependency, build, and unknown changes select everything:
    this deliberately avoids guessing Rust's transitive dependency graph.
    """
    names = {row['target'] for row in rows}
    selected = set()
    for path in paths:
        parts = Path(path).parts
        if (path.startswith(('docs/', 'benchmark_results/', '.changes/'))
                or (len(parts) == 1 and path.endswith('.md'))):
            continue
        if path.startswith(('fuzz/target_impls/', 'fuzz/fuzz_targets/')) and path.endswith('.rs'):
            name = Path(path).stem
            if name in names:
                selected.add(name)
                continue
        matches = [row['target'] for row in rows
                   if path.startswith(f"{row['package']}/corpus/{row['target']}/")
                   or path == f"{row['package']}/fuzz_targets/{row['target']}.rs"]
        if matches:
            selected.update(matches)
        else:
            return sorted(names)
    return sorted(selected)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def changed_paths(root, event, payload):
    base = payload.get('before') if event == 'push' else payload.get('pull_request', {}).get('base', {}).get('sha')
    if not base or not re.fullmatch(r'[0-9a-f]{40,64}', base) or not base.strip('0'):
        return None, None
    try:
        if event == 'pull_request':
            base = git(root, 'merge-base', base, 'HEAD')
        paths = subprocess.check_output(
            ['git', 'diff', '--name-only', '-z', '--no-renames', base, 'HEAD'], cwd=root).decode().split('\0')
        return base, [path for path in paths if path]
    except subprocess.CalledProcessError:
        return None, None


def plan(root, env, now):
    rows = inventory(root)
    names = sorted({row['target'] for row in rows})
    event = env.get('GITHUB_EVENT_NAME', 'workflow_dispatch')
    campaign = env.get('FUZZ_CAMPAIGN', '') or 'smoke'
    if campaign not in ('smoke', 'daily'):
        raise ValueError('campaign must be smoke or daily')
    daily = event == 'schedule' or campaign == 'daily'
    requested = env.get('FUZZ_TARGETS', '')
    base, paths = None, None
    if requested:
        selected = requested.split(',')
        if len(selected) != len(set(selected)) or not set(selected) <= set(names):
            raise ValueError('targets must be distinct, exact fuzz target names')
    elif daily:
        start = (now.date().toordinal() * DAILY_TARGETS) % len(names)
        selected = [names[(start + offset) % len(names)] for offset in range(min(DAILY_TARGETS, len(names)))]
    elif event in ('push', 'pull_request'):
        payload = json.loads(Path(env['GITHUB_EVENT_PATH']).read_text())
        base, paths = changed_paths(root, event, payload)
        selected = names if paths is None else affected(rows, paths)
    else:
        selected = names
    selected = sorted(selected)
    pairs = [row for row in rows if row['target'] in selected]
    batches = math.ceil(len(pairs) / CONCURRENCY)
    duration = BUDGET // max(1, batches) if daily else int(env.get('FUZZ_DURATION', '') or '60')
    if duration <= 0 or batches * duration > BUDGET:
        raise ValueError(f'selection exceeds the {BUDGET}s live-fuzzing budget or has an invalid duration')
    return {
        'schema': 1, 'event': event, 'campaign': 'daily' if daily else 'smoke', 'source': bundle.source_identity(root),
        'created_at_utc': now.isoformat(), 'base': base, 'changed_paths': paths,
        'targets': selected, 'pairs': pairs, 'duration_seconds': duration,
        'concurrency': CONCURRENCY, 'budget_seconds': BUDGET,
        'planned_live_seconds': batches * duration,
    }


def corpus_path(row):
    return Path(row['package']) / 'corpus' / row['target']


def inputs(directory):
    """Read only flat, regular inputs; never follow a cached symlink."""
    if any(parent.is_symlink() for parent in [directory, *directory.parents]):
        raise ValueError(f'symlink in corpus path: {directory}')
    if not directory.exists():
        return {}
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'not a regular corpus input: {path}')
        data = path.read_bytes()
        result[hashlib.sha256(data).hexdigest()] = data
    return result


def write_inputs(directory, values):
    directory.mkdir(parents=True, exist_ok=True)
    for digest, data in values.items():
        (directory / digest).write_bytes(data)


def restore(root, cache, rows):
    for row in rows:
        source = cache / corpus_path(row)
        values = inputs(source)
        if source.exists() and set(values) != {path.name for path in source.iterdir()}:
            raise ValueError(f'cached corpus content does not match its SHA256 names: {source}')
        destination = root / corpus_path(row)
        existing = inputs(destination)  # Reject symlinks; committed seed names stay intact.
        destination.mkdir(parents=True, exist_ok=True)
        for digest, data in values.items():
            path = destination / f'ci-{digest}'
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'corpus hash collision: {path}')
            if digest not in existing:
                path.write_bytes(data)


def snapshot(root, cache, rows):
    """Replace only the generated cache after every source corpus has been read."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fuzz-corpus-', dir=cache.parent) as temporary:
        staged = Path(temporary) / 'corpus'
        staged.mkdir()
        for row in rows:
            values = inputs(root / corpus_path(row))
            if not values:
                raise ValueError(f'empty corpus: {corpus_path(row)}')
            write_inputs(staged / corpus_path(row), values)
        if cache.exists():
            shutil.rmtree(cache)
        staged.rename(cache)


def execute(args, root, env, log, *, stdout=None):
    command = shlex.join(map(str, args))
    print('+ ' + (command if len(command) <= 300 else command[:300] + ' …'), flush=True)
    log.write('+ ' + command + '\n')
    log.flush()
    subprocess.run(args, cwd=root, env=env, stdout=stdout or log, stderr=log,
                   check=True, timeout=900)


def branch_gaps(data, root):
    files, branches, missing = 0, 0, []
    for unit in data['data']:
        for entry in unit['files']:
            path = Path(entry['filename'])
            if not path.is_relative_to(root):
                continue
            relative = path.relative_to(root).as_posix()
            if not relative.startswith(('src/', 'rscrypto-rustls/src/')):
                continue
            files += 1
            for branch in entry['branches']:
                branches += 1
                line, column, end_line, end_column, true_count, false_count = branch[:6]
                if true_count == 0 or false_count == 0:
                    missing.append({'file': relative, 'line': line, 'column': column,
                                    'end_line': end_line, 'end_column': end_column,
                                    'true': true_count, 'false': false_count})
    if not files:
        raise ValueError('coverage contains no production source files')
    return {'source_files': files, 'instrumented_conditions': branches, 'uncovered_outcomes': missing,
            'limit': 'Zero counts identify unobserved outcomes, not proof that those outcomes are feasible.'}


def retain(root, cache, logs, selected, env):
    """Minimize and measure each selected package/target before publishing it."""
    env = {**env, 'RUSTUP_TOOLCHAIN': tomllib.loads((root / 'rust-toolchain.toml').read_text())['toolchain']['channel'],
           'CARGO_RAIL_CACHE': 'off'}
    if env.get('CARGO_ENCODED_RUSTFLAGS'):
        raise ValueError('unset CARGO_ENCODED_RUSTFLAGS so cargo-fuzz can install its instrumentation')
    compiler = subprocess.check_output(['rustc', '-vV'], cwd=root, env=env, text=True)
    host = next(line.removeprefix('host: ') for line in compiler.splitlines() if line.startswith('host: '))
    sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], cwd=root, env=env, text=True).strip())
    llvm = sysroot / 'lib/rustlib' / host / 'bin/llvm-cov'
    if not llvm.is_file():
        raise ValueError('install the pinned toolchain llvm-tools component for fuzz coverage')
    build = root / 'fuzz/target'
    coverage_build = root / 'target/fuzz-coverage'
    # A fresh completion marker distinguishes successful evidence from partial logs.
    completion = logs / 'campaign.json'
    completion.unlink(missing_ok=True)
    records = []
    for row in selected['pairs']:
        package, target = root / row['package'], row['target']
        output = logs / 'coverage' / row['package'] / target
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'commands.log').open('w') as log, tempfile.TemporaryDirectory(prefix='corpus-', dir=output) as temporary:
            work = Path(temporary)
            original, minimized = work / 'original', work / 'minimized'
            values = inputs(root / corpus_path(row))
            if not values:
                raise ValueError(f'empty corpus: {corpus_path(row)}')
            write_inputs(original, values)
            minimized.mkdir()
            execute(['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1',
                     '--manifest-path', package / 'Cargo.toml'], root, env, log)
            execute(['cargo', 'fuzz', 'build', '--fuzz-dir', package, '--target', host,
                     '--target-dir', build, target], root, env, log)
            binary = build / host / 'release' / target
            artifacts = package / 'artifacts' / target
            artifacts.mkdir(parents=True, exist_ok=True)
            limits = ['-timeout=30', '-rss_limit_mb=2048', f'-artifact_prefix={artifacts}/']
            # cargo-fuzz 0.13.2 exec_cmin prints a merge failure but returns Ok(()).
            # Run the production fuzzer directly so a failed merge cannot publish success.
            execute([binary, '-merge=1', *limits, minimized, original], root, env, log)
            values = inputs(minimized)
            if not values:
                raise ValueError(f'minimization produced an empty corpus: {target}')
            # Preserve the committed regression inputs even if edge minimization drops them.
            seeds = (root / 'fuzz/committed-seeds.txt').read_text().splitlines()
            prefix = corpus_path(row).as_posix() + '/'
            for seed in seeds:
                if seed.startswith(prefix):
                    data = (root / seed).read_bytes()
                    values[hashlib.sha256(data).hexdigest()] = data
            shutil.rmtree(minimized)
            write_inputs(minimized, values)
            cov_env = {**env, 'RUSTFLAGS': env.get('RUSTFLAGS', '') + ' --cfg rscrypto_internal_fuzzing -Zcoverage-options=branch'}
            raw = package / 'coverage' / target
            if raw.exists():
                shutil.rmtree(raw)  # Generated output: stale profiles must never enter this report.
            execute(['cargo', 'fuzz', 'coverage', '--fuzz-dir', package, '--target', host,
                     '--target-dir', coverage_build, '-j', '1', target, minimized, '--', *limits], root, cov_env, log)
            profile = output / 'coverage.profdata'
            shutil.copyfile(raw / 'coverage.profdata', profile)
            cov_binary = coverage_build / host / 'release' / target
            sources = sorted((root / 'src').rglob('*.rs')) + sorted((root / 'rscrypto-rustls/src').rglob('*.rs'))
            arguments = [str(cov_binary), f'-instr-profile={profile}', '--sources', *map(str, sources)]
            with (output / 'coverage.json').open('w') as stream:
                execute([llvm, 'export', '-skip-functions', '-skip-expansions', *arguments], root, env, log, stdout=stream)
            gaps = branch_gaps(json.loads((output / 'coverage.json').read_text()), root)
            (output / 'branch-gaps.json').write_text(json.dumps(gaps, indent=2) + '\n')
            with (output / 'SUMMARY.txt').open('w') as stream:
                execute([llvm, 'report', *arguments], root, env, log, stdout=stream)
            if inputs(minimized) != values:
                raise ValueError('corpus changed during coverage collection')
            destination = cache / corpus_path(row)
            if destination.exists():
                shutil.rmtree(destination)
            write_inputs(destination, values)
            records.append({**row, 'inputs': {digest: len(data) for digest, data in sorted(values.items())},
                            'binary_sha256': hashlib.sha256(cov_binary.read_bytes()).hexdigest(),
                            'instrumented_conditions': gaps['instrumented_conditions'],
                            'uncovered_outcomes': len(gaps['uncovered_outcomes']),
                            'coverage': output.relative_to(logs).as_posix()})
    if bundle.source_identity(root) != selected['source']:
        raise ValueError('source changed during fuzz coverage collection')
    completion.write_text(json.dumps({'plan': selected, 'compiler': compiler, 'corpora': records}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'restore', 'snapshot', 'retain'])
    args = parser.parse_args()
    cache = ROOT / 'target/fuzz-corpus'
    logs = Path(os.environ.get('RSCRYPTO_FUZZ_LOG_DIR', ROOT / 'target/fuzz-logs'))
    logs.mkdir(parents=True, exist_ok=True)
    plan_path = logs / 'plan.json'
    if args.action == 'plan':
        result = plan(ROOT, os.environ, datetime.now(timezone.utc))
        plan_path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        if os.environ.get('GITHUB_OUTPUT'):
            with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
                output.write(f"targets={','.join(result['targets'])}\nduration={result['duration_seconds']}\n")
                output.write(f"live={'true' if result['pairs'] else 'false'}\n")
                output.write(f"retain={'true' if result['campaign'] == 'daily' else 'false'}\n")
                output.write(f"concurrency={result['concurrency']}\nbudget={result['budget_seconds']}\n")
    elif args.action == 'restore':
        restore(ROOT, cache, inventory(ROOT))
    elif args.action == 'snapshot':
        snapshot(ROOT, cache, inventory(ROOT))
    else:
        selected = json.loads(plan_path.read_text())
        if selected['source'] != bundle.source_identity(ROOT):
            raise ValueError('source changed after fuzz planning')
        retain(ROOT, cache, logs, selected, os.environ)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        sys.exit(f'fuzz CI failed: {error}')
