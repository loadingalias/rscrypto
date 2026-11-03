#!/usr/bin/env -S uv run python
"""Run pinned QUIC Interop Runner cases in bounded local Docker campaigns."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = '740c05a10b61d65e8abd3ad38d60898004d335d9'
ENDPOINT = '82c845f6e5399ef15fac217ba2799c2a233d875b'
# Tag h3-quinn-v0.0.10; its h3 sources equal h3-v0.0.8.
H3 = '2dc3412bdf6083451920d5bfd7a9484d054c1859'
GO = 'martenseemann/quic-go-interop@sha256:633266d83b816442975eb7ccb9123ba95b7f8356b71338e0ed319536f192c584'
SIMULATOR = 'martenseemann/quic-network-simulator@sha256:c23d82a55caffe681b1bdae65d4d30d23e1283141a414a7f02ee56cf15f9c6b9'
ENDPOINT_BASE = 'martenseemann/quic-network-simulator-endpoint@sha256:3b2b9e6fa317da238c8140a7b6d2bf8d4d45a8464b873eecac3ff2a32520b71a'
UBUNTU = 'ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7'
UV = 'ghcr.io/astral-sh/uv@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168'
PROJECT = 'rscrypto_provider_interop'
GO_ECN_ENTRYPOINT = '''#!/bin/sh
set -eu
if [ "${TESTCASE:-}" = ecn ]; then
    export TESTCASE=transfer
fi
exec ./run_endpoint.sh "$@"
'''


def run(output, name, command, *, cwd=ROOT, data=None, timeout=600, check=True):
    record = {'cwd': str(cwd), 'argv': list(map(str, command)), 'timeout_seconds': timeout}
    (output / (name + '-command.json')).write_text(json.dumps(record, indent=2) + '\n')
    with (output / (name + '.log')).open('wb') as log:
        result = subprocess.run(command, cwd=cwd, input=data, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}: {output}')
    return result.returncode


def checkout(output, name, url, revision):
    directory = output / name
    run(output, name + '-init', ['git', 'init', str(directory)])
    run(output, name + '-fetch', ['git', 'fetch', '--depth=1', url, revision], cwd=directory)
    run(output, name + '-checkout', ['git', 'checkout', '--detach', revision], cwd=directory)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
    if actual != revision:
        raise RuntimeError(f'{name} differs from its source pin')
    return directory


def ensure_image(output, name, image):
    # A content-addressed local image is already the requested artifact. Do not
    # make reproducibility depend on registry availability or its rate limit.
    if '@sha256:' not in image:
        raise ValueError('cached image reuse requires a digest')
    if run(output, name + '-inspect', ['docker', 'image', 'inspect', image], check=False):
        run(output, name + '-pull', ['docker', 'pull', image])


def prepare(output, *, adapt_go_ecn=False, stock=False):
    runner = checkout(output, 'runner', 'https://github.com/quic-interop/quic-interop-runner.git', RUNNER)
    endpoint = checkout(output, 'endpoint', 'https://github.com/quinn-rs/quinn-interop.git', ENDPOINT)
    h3 = checkout(output, 'h3', 'https://github.com/hyperium/h3.git', H3)
    run(output, 'h3-patch', ['git', 'apply', '-'], cwd=h3,
        data=(ROOT / 'scripts/test/provider_h3_quinn.patch').read_bytes())
    patch_name = 'provider_quinn_stock.patch' if stock else 'provider_quinn_interop.patch'
    patch = (ROOT / 'scripts/test' / patch_name).read_text()
    patch = patch.replace('@RSCRYPTO_ROOT@', str(ROOT)).replace('@H3_ROOT@', str(h3))
    run(output, 'endpoint-patch', ['git', 'apply', '-'], cwd=endpoint, data=patch.encode())
    # One selected share makes the two stock controls and the provider's PQ
    # flight comparable. These are harness settings, not provider defaults.
    client = endpoint / 'quinn-interop/src/bin/client.rs'
    source = client.read_text()
    marker = '    if let CipherSuite::Chacha20 = suites {'
    if source.count(marker) != 1:
        raise RuntimeError('upstream client group insertion point changed')
    source = source.replace(marker, '''    if let Ok(group) = std::env::var("RSCRYPTO_QUIC_GROUP") {
        provider.kx_groups = provider.kx_groups.iter().copied()
            .filter(|g| format!("{:?}", g.name()) == group).collect::<Vec<_>>().into();
        anyhow::ensure!(!provider.kx_groups.is_empty(), "unavailable control group");
    }
''' + marker)
    client.write_text(source)
    architecture = subprocess.check_output(['docker', 'info', '--format', '{{.Architecture}}'], text=True).strip()
    targets = {'aarch64': 'aarch64-unknown-linux-gnu', 'arm64': 'aarch64-unknown-linux-gnu',
               'x86_64': 'x86_64-unknown-linux-gnu', 'amd64': 'x86_64-unknown-linux-gnu'}
    if architecture not in targets:
        raise RuntimeError('the pinned endpoint build supports Linux ARM64 or x86-64 Docker engines')
    target = targets[architecture]
    manifest = endpoint / 'quinn-interop/Cargo.toml'
    run(output, 'endpoint-resolve', ['cargo', 'metadata', '--format-version=1', '--manifest-path', str(manifest)])
    run(output, 'endpoint-build', ['cargo', 'zigbuild', '--locked', '--release', '--manifest-path', str(manifest),
                                   '--target', target + '.2.39'])
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version=1',
                                                  '--manifest-path', str(manifest)], cwd=ROOT))
    build = output / 'build-endpoint'
    build.mkdir()
    for name in ('client', 'server'):
        shutil.copyfile(Path(metadata['target_directory']) / target / 'release' / name, build / name)
    shutil.copyfile(endpoint / 'quinn-interop/run_endpoint.sh', build / 'run_endpoint.sh')
    (build / 'Dockerfile').write_text(f'FROM {ENDPOINT_BASE}\nWORKDIR /h3-quinn\nCOPY client server run_endpoint.sh ./\n'
                                    'RUN chmod +x client server run_endpoint.sh\nENTRYPOINT ["./run_endpoint.sh"]\n')
    endpoint_image = 'rscrypto-quinn-interop:' + output.name.lower()
    run(output, 'endpoint-image', ['docker', 'build', '-t', endpoint_image, str(build)])
    driver = output / 'build-driver'
    driver.mkdir()
    shutil.copyfile(ROOT / 'scripts/test/provider_quic_runner_requirements.txt', driver / 'requirements.txt')
    (driver / 'Dockerfile').write_text(
        f'FROM {UBUNTU}\nCOPY --from={UV} /uv /usr/local/bin/uv\n'
        'ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python\n'
        'RUN apt-get update && apt-get install -y --no-install-recommends docker.io docker-compose-v2 '
        'tshark openssl iproute2 ca-certificates && rm -rf /var/lib/apt/lists/*\n'
        'COPY requirements.txt /tmp/requirements.txt\n'
        'RUN uv python install 3.13.9 && uv venv --python 3.13.9 /opt/runner && '
        'uv pip install --python /opt/runner/bin/python -r /tmp/requirements.txt\nENV PATH=/opt/runner/bin:$PATH\n')
    driver_image = 'rscrypto-quic-runner:' + output.name.lower()
    run(output, 'driver-image', ['docker', 'build', '-t', driver_image, str(driver)])
    for name, image in (('go', GO), ('simulator', SIMULATOR)):
        ensure_image(output, name, image)
    go_image = GO
    if adapt_go_ecn:
        # The pinned Go interop application omits the ECN case from its switch.
        # Use its ordinary transfer path; the runner still verifies ECT and
        # ACK-ECN packets in both directions with its original assertions.
        adapter = output / 'build-go-ecn'
        adapter.mkdir()
        (adapter / 'entrypoint.sh').write_text(GO_ECN_ENTRYPOINT)
        (adapter / 'Dockerfile').write_text(
            f'FROM {GO}\nCOPY entrypoint.sh /rscrypto-ecn-entrypoint\n'
            'RUN chmod +x /rscrypto-ecn-entrypoint\nENTRYPOINT ["/rscrypto-ecn-entrypoint"]\n')
        go_image = 'rscrypto-quic-go-ecn:' + output.name.lower()
        run(output, 'go-ecn-image', ['docker', 'build', '-t', go_image, str(adapter)])
    compose = runner / 'docker-compose.yml'
    source = compose.read_text()
    source = source.replace('image: martenseemann/quic-network-simulator\n', f'image: {SIMULATOR}\n')
    compose.write_text(source)
    provenance = {'runner': RUNNER, 'endpoint': ENDPOINT, 'h3': H3, 'target': target, 'stock': stock,
                  'endpoint_image': endpoint_image, 'driver_image': driver_image,
                  'go_image': go_image, 'go_base_image': GO,
                  'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    for role, image in (('endpoint', endpoint_image), ('driver', driver_image), ('go', go_image)):
        provenance[role + '_image_id'] = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    shutil.copytree(ROOT / 'rscrypto-rustls', output / 'source/rscrypto-rustls')
    shutil.copytree(ROOT / 'src', output / 'source/src')
    for relative in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'scripts/test/provider_quic_runner.py',
                     '.cargo/config.toml',
                     'scripts/test/provider_quic_runner_requirements.txt', 'scripts/test/provider_quinn_interop.patch',
                     'scripts/test/provider_h3_quinn.patch', 'scripts/test/provider_quinn_stock.patch'):
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    run(output, 'images', ['docker', 'image', 'inspect', endpoint_image, driver_image, go_image, GO, SIMULATOR])
    run(output, 'rustc', ['rustc', '-Vv'])
    run(output, 'cargo', ['cargo', '-V'])
    return provenance


def cleanup(name):
    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    for kind in ('container', 'network'):
        ids = subprocess.check_output(['docker', kind, 'ls', '-aq' if kind == 'container' else '-q',
                                       '--filter', 'label=com.docker.compose.project=' + PROJECT], text=True).split()
        if ids:
            subprocess.run(['docker', kind, 'rm', *(['-f'] if kind == 'container' else []), *ids],
                           stdout=subprocess.DEVNULL, check=True)


def summarize(result, requested, pair_count):
    rows = [row for pair in result['results'] for row in pair]
    failures = [row for row in rows if row['result'] != 'succeeded']
    complete = (len(result['results']) == pair_count and
                all(len(pair) == len(requested) and {row['name'] for row in pair} == set(requested)
                    for pair in result['results']))
    return {'complete': complete, 'passed': len(rows) - len(failures), 'other': failures}


def campaign_budget(value):
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('budget must be an integer number of seconds') from error
    if not 1 <= seconds <= 600:
        raise argparse.ArgumentTypeError('budget must be between 1 and 600 seconds')
    return seconds


def go_image_for_bundle(provenance):
    image = provenance.get('go_image', GO)
    if image != GO and (provenance.get('go_base_image') != GO or not provenance.get('go_image_id')):
        raise RuntimeError('the prepared Go ECN adapter lacks its pinned base and image identity')
    return image


def campaign_sources(prepared, output, seed, rng_run, credentials, group, trace):
    """Keep each campaign's settings and retained credentials out of the reusable bundle."""
    runner = output / 'campaign-runner'
    shutil.copytree(prepared / 'runner', runner, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    temporary = output / 'tmp'
    temporary.mkdir(exist_ok=True)
    # Test subclasses also allocate credentials (notably amplificationlimit).
    # Every bind source must exist at the same path on the Docker host. Keep
    # each campaign independent even when it reuses an older prepared bundle.
    for path in runner.glob('*.py'):
        source = path.read_text().replace('dir="/tmp"', f'dir={str(temporary)!r}')
        source = source.replace(f'dir={str(prepared / "tmp")!r}', f'dir={str(temporary)!r}')
        path.write_text(source)
    path = runner / 'interop.py'
    source = path.read_text()
    marker = ').format(test.scenario())'
    if source.count(marker) != 1:
        raise RuntimeError('upstream scenario insertion point changed')
    source = source.replace(marker, f').format(test.scenario() + " --RngSeed={seed} --RngRun={rng_run}")')
    # Keep credentials and logs even when an endpoint reports unsupported.
    source = source.replace('if status == TestResult.FAILED or status == TestResult.SUCCEEDED:', 'if True:')
    marker = '            shutil.copyfile(log_file.name, log_dir + "/output.txt")'
    if source.count(marker) != 1:
        raise RuntimeError('upstream credential-retention insertion point changed')
    source = source.replace(marker, marker + '\n            shutil.copytree(test.certs_dir(), log_dir + "/certs")')
    path.write_text(source)
    if credentials:
        retained = output / 'credentials'
        shutil.copytree(credentials, retained)
        path = runner / 'testcase.py'
        source = path.read_text()
        marker = 'def generate_cert_chain(directory: str, length: int = 1):\n'
        if source.count(marker) != 1:
            raise RuntimeError('upstream credential insertion point changed')
        # Amplification's long chain must still come from its original fixture.
        source = source.replace(marker, marker +
            f'    if length == 1:\n        import shutil\n        shutil.copytree({str(retained)!r}, directory, dirs_exist_ok=True)\n        return\n')
        path.write_text(source)
    path = runner / 'docker-compose.yml'
    environment = f'      - RSCRYPTO_QUIC_GROUP={group}\n' if group else ''
    if trace:
        environment += '      - RUST_LOG=quinn_proto=trace,quinn=trace,h3_quinn_interop=debug\n'
    source = path.read_text().replace('      - ROLE=', environment + '      - ROLE=')
    path.write_text(source)
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, help='reuse the source-bound images in an earlier prepared bundle')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stock', action='store_true', help='prepare the pinned Quinn stock rustls/AWS-LC control')
    parser.add_argument('--prepare-only', action='store_true', help='build and retain images without starting a campaign')
    parser.add_argument('--group', choices=('X25519MLKEM768', 'X25519'),
                        help='restrict the Quinn client to one group; otherwise keep its defaults')
    parser.add_argument('--seed', type=int, default=1, help='ns-3 RngSeed; TLS entropy remains independent')
    parser.add_argument('--rng-run', type=int, default=1, help='ns-3 RngRun substream')
    parser.add_argument('--credentials', type=Path, help='reuse retained runner credentials (except the long amplification chain)')
    parser.add_argument('--trace', action='store_true', help='retain Quinn production transport tracing')
    parser.add_argument('--tests', default='handshake,transfer,chacha20,retry,resumption,zerortt,keyupdate')
    parser.add_argument('--budget-seconds', type=campaign_budget, default=600,
                        help='campaign wall-clock limit, at most 600 seconds (default: 600)')
    parser.add_argument('--server', choices=('quinn', 'quic-go', 'quinn,quic-go'), default='quinn,quic-go')
    parser.add_argument('--client', choices=('quinn', 'quic-go', 'quinn,quic-go'), default='quinn,quic-go')
    args = parser.parse_args()
    requested = args.tests.split(',')
    if not all(requested) or len(set(requested)) != len(requested):
        parser.error('--tests requires distinct, nonempty case names')
    if not 1 <= args.seed <= 0xffffffff or not 1 <= args.rng_run <= 0xffffffffffffffff:
        parser.error('seed/run must be nonzero uint32/uint64 values')
    output = (args.output or ROOT / 'benchmark_results' / ('provider-quic-runner-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared.resolve() if args.prepared else output
    provenance = (json.loads((prepared / 'provenance.json').read_text()) if args.prepared
                  else prepare(output, adapt_go_ecn='ecn' in requested, stock=args.stock))
    if provenance['runner'] != RUNNER or provenance['endpoint'] != ENDPOINT:
        raise RuntimeError('prepared sources differ from the selected upstream pins')
    # Verify the endpoint and driver artifacts before using a prepared bundle.
    endpoint_image, driver_image = provenance['endpoint_image'], provenance['driver_image']
    go_image = go_image_for_bundle(provenance)
    identities = [('endpoint', endpoint_image), ('driver', driver_image)]
    if 'go_image_id' in provenance:
        identities.append(('go', go_image))
    for role, image in identities:
        actual = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
        if actual != provenance[role + '_image_id']:
            raise RuntimeError(f'the prepared {role} image changed')
    run(output, 'images', ['docker', 'image', 'inspect', endpoint_image, driver_image, go_image, GO, SIMULATOR])
    (output / 'prepared.json').write_text(json.dumps({'path': str(prepared), 'provenance': provenance}, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared source-bound images: {prepared}')
        return 0
    active = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=com.docker.compose.project=' + PROJECT], text=True).strip()
    if active:
        raise RuntimeError('another provider runner campaign owns the Compose project')
    name = 'rscrypto-quic-driver-' + output.name.lower()
    runner = campaign_sources(prepared, output, args.seed, args.rng_run, args.credentials,
                              args.group, args.trace)
    (output / 'settings.json').write_text(json.dumps({
        'seed': args.seed, 'rng_run': args.rng_run, 'group': args.group, 'trace': args.trace,
        'tests': requested, 'server': args.server, 'client': args.client,
        'budget_seconds': args.budget_seconds, 'credentials': str(args.credentials),
    }, indent=2) + '\n')
    mounts = list(dict.fromkeys((prepared, output)))
    command = ['docker', 'run', '--rm', '--name', name, '-e', 'COMPOSE_PROJECT_NAME=' + PROJECT,
               '-v', '/var/run/docker.sock:/var/run/docker.sock']
    for path in mounts:
        command += ['-v', f'{path}:{path}']
    selected = ','.join(dict.fromkeys(args.server.split(',') + args.client.split(',')))
    command += ['-w', str(runner), driver_image, 'python', 'run.py', '-d',
                '-s', args.server, '-c', args.client, '-n', selected, '-r', f'quinn={endpoint_image},quic-go={go_image}',
                '-t', args.tests, '-l', str(output / 'logs'), '-j', str(output / 'results.json')]
    began = time.monotonic()
    status = 1
    try:
        status = run(output, 'runner', command, timeout=args.budget_seconds, check=False)
    except subprocess.TimeoutExpired:
        (output / 'timeout.json').write_text(json.dumps({'budget_seconds': args.budget_seconds, 'status': 'incomplete'}) + '\n')
        status = 124
    finally:
        cleanup(name)
        (output / 'elapsed_seconds.txt').write_text(str(time.monotonic() - began) + '\n')
    if (output / 'results.json').exists():
        result = json.loads((output / 'results.json').read_text())
        pair_count = len(args.server.split(',')) * len(args.client.split(','))
        summary = summarize(result, requested, pair_count)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        status = status or int(bool(summary['other']) or not summary['complete'])
    else:
        status = status or 1
    (output / 'SHA256SUMS').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(output)) + '\n'
                                            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS' and '.git' not in path.parts and 'target' not in path.parts))
    print(f'QUIC runner exit={status}; evidence: {output}')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
