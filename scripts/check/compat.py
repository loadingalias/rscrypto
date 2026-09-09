#!/usr/bin/env python3
"""Bounded, fail-fast feature/target checks and real WASM/WASI execution."""
import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('toolchain', ROOT / 'scripts/lib/toolchain.py')
toolchain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toolchain)


def read(path):
    return tomllib.loads((ROOT / path).read_text())


def targets():
    return [t for t in json.loads((ROOT / '.config/target-matrix.json').read_text())['targets']
            if '-none' in t or t.startswith('wasm32-')]


def install():
    channels = {toolchain.stable(): set(targets())}
    for target in targets():
        channel = toolchain.for_target(target)
        if channel != toolchain.stable():
            channels[toolchain.stable()].remove(target)
            channels.setdefault(channel, set()).add(target)
    channels.setdefault(read('Cargo.toml')['package']['rust-version'], set()).add('thumbv6m-none-eabi')
    for channel, selected in channels.items():
        subprocess.run(['rustup', 'toolchain', 'install', channel, '--profile', 'minimal',
                        '--target', ','.join(sorted(selected))], check=True)


def boundary_features(graph, boundary):
    """Maximal feature set that respects the core-only or alloc boundary."""
    def closure(feature):
        result, pending = set(), [feature]
        while pending:
            name = pending.pop()
            if name not in result:
                result.add(name)
                pending.extend(graph.get(name, []))
        return result
    forbidden = {'std', 'getrandom'} | ({'alloc'} if boundary == 'core' else set())
    return [feature for feature in graph if not closure(feature) & forbidden]


def cases():
    manifest = read('Cargo.toml')
    graph = manifest['features']
    stable = toolchain.stable()
    msrv = manifest['package']['rust-version']
    host = toolchain.host()

    def check(channel, target, features, operation='check'):
        command = ['cargo', '+' + channel, operation, '--locked', '--lib', '--target', target,
                   '--no-default-features']
        if operation == 'build':
            command.append('--release')
        if features:
            command += ['--features', ','.join(features)]
        return command

    # Standalone features include umbrella aliases: they are also public contracts.
    for channel in dict.fromkeys([stable, msrv]):
        for feature in ['', *sorted(graph)]:
            yield f'{channel}-{feature or "empty"}', [check(channel, host, [feature] if feature else [])], {}
        for boundary in ('core', 'alloc'):
            selected = boundary_features(graph, boundary)
            yield f'{channel}-thumb-{boundary}', [check(channel, 'thumbv6m-none-eabi', selected)], {}
        native = sorted(set(graph) - {'portable-only'})
        yield f'{channel}-broad', [check(channel, host, native), check(channel, host, [*native, 'portable-only'])], {}

    for target in targets():
        features = ['full', 'serde', 'serde-secrets', 'websocket-sha1']
        if target == 'wasm32-wasip1':
            features += ['std', 'diag', 'getrandom']
        yield target, [check(toolchain.for_target(target), target, features, 'build')], {}

    # Execute the same independent vectors in bare WASM and WASI, scalar and SIMD.
    for target in ('wasm32-unknown-unknown', 'wasm32-wasip1'):
        for simd in (False, True):
            name = f'{target}-{"simd" if simd else "scalar"}'
            build = ['cargo', '+' + stable, 'build', '--locked', '--release', '--manifest-path',
                     'tools/wasm-runtime-vectors/Cargo.toml', '--target', target]
            invoke = ['--invoke', 'run_vectors'] if target.endswith('unknown-unknown') else []
            run = ['wasmtime', 'run', '-W', f'simd={"y" if simd else "n"},relaxed-simd=n',
                   *invoke, '{target_dir}/' + target + '/release/rscrypto-wasm-runtime-vectors.wasm']
            yield name, [build, run], {'RUSTFLAGS': '-Ctarget-feature=' + ('+simd128,-relaxed-simd' if simd else '-simd128,-relaxed-simd')}


def execute(work, workers, directory):
    """One Cargo process per worker directory; terminate process groups on failure."""
    directory.mkdir(parents=True, exist_ok=True)
    pending = deque(work)
    active = {}
    free = list(range(workers))
    jobs = max(1, (os.cpu_count() or 1) // workers)

    def stop():
        for process, _, _, _, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        for process, output, _, _, _, _ in active.values():
            process.wait()
            output.close()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            while pending and free:
                worker = free.pop()
                name, commands, extra = pending.popleft()
                target_dir = directory / f'worker-{worker}'
                environment = {**os.environ, 'CARGO_TARGET_DIR': str(target_dir), 'CARGO_BUILD_JOBS': str(jobs),
                               'CARGO_RAIL_CACHE': 'off', **extra}
                commands = [[arg.replace('{target_dir}', str(target_dir)) for arg in command] for command in commands]
                path = directory / f'{name}.log'
                output = path.open('w')
                print(f'Starting {name}', flush=True)
                process = subprocess.Popen(commands.pop(0), cwd=ROOT, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[worker] = (process, output, name, commands, environment, path)
            for worker, (process, output, name, commands, environment, path) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                if status:
                    output.flush()
                    print(path.read_text(), flush=True)
                    raise SystemExit(f'{name} failed ({status}); log: {path}')
                if commands:
                    process = subprocess.Popen(commands.pop(0), cwd=ROOT, env=environment, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    active[worker] = (process, output, name, commands, environment, path)
                else:
                    output.close()
                    del active[worker]
                    free.append(worker)
                    print(f'Passed {name}', flush=True)
            if active:
                time.sleep(0.1)
    finally:
        stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.install:
        install()
        return
    work = list(cases())
    if args.list:
        print(json.dumps(work, indent=2))
        return
    workers = min(read('.config/tooling.toml')['ci-compat']['workers'], os.cpu_count() or 1)
    execute(work, workers, ROOT / 'target/compat')


if __name__ == '__main__':
    main()
