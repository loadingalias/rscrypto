#!/usr/bin/env python3
"""Exercise the rscrypto Quinn adapter against the pinned, C-free quic-go peer."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from provider_interop import ROOT, FIXTURES, CREDENTIALS, MLDSA_CREDENTIALS, SUITES, GROUPS, ready, results


def run_case(output, binary, peer, kind, suite, group, scenario, ours_client):
    name = '-'.join([kind, suite, group, scenario, 'ours-client' if ours_client else 'ours-server'])
    directory = output / name
    directory.mkdir()
    logs = directory / 'qlog'
    logs.mkdir()
    control = directory / 'reset-control'
    control.mkdir()
    env = {**os.environ, 'QLOGDIR': str(logs), 'RSCRYPTO_QUIC_RESET_CONTROL': str(control)}
    count = 1 if scenario == 'reset' else 2
    common = [str(FIXTURES), kind, suite, group, str(count), scenario]
    ours = [str(binary), 'client' if ours_client else 'server', '{address}' if ours_client else '127.0.0.1:0', *common]
    theirs = [str(peer), 'server' if ours_client else 'client', '127.0.0.1:0' if ours_client else '{address}', *common]
    server_command, client_command = (theirs, ours) if ours_client else (ours, theirs)
    (directory / 'commands.json').write_text(json.dumps({'server': server_command, 'client': client_command, 'QLOGDIR': str(logs), 'RSCRYPTO_QUIC_RESET_CONTROL': str(control)}, indent=2) + '\n')
    prelude = b''
    with (directory / 'server.stderr').open('wb') as errors:
        server = subprocess.Popen(server_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        try:
            prelude, line = ready(server, b'LISTEN ')
            client_command = [arg.replace('{address}', line.split()[1]) for arg in client_command]
            with (directory / 'client.stderr').open('wb') as client_errors:
                client = subprocess.run(client_command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=client_errors, timeout=35)
            (directory / 'client.stdout').write_bytes(client.stdout)
            if client.returncode:
                raise RuntimeError(f'client exited {client.returncode}: {directory}')
            remaining, _ = server.communicate(timeout=35)
            server_output = prelude + remaining
            (directory / 'server.stdout').write_bytes(server_output)
            if server.returncode:
                raise RuntimeError(f'server exited {server.returncode}: {directory}')
            ours_output, peer_output = (client.stdout, server_output) if ours_client else (server_output, client.stdout)
            ours_rows, peer_rows = results(ours_output), results(peer_output)
            if len(ours_rows) != count or len(peer_rows) != count or 'Full' not in ours_rows[0] or (count == 2 and 'Resumed' not in ours_rows[1]):
                raise AssertionError(f'full/resumed QUIC mismatch: {ours_rows} / {peer_rows}')
            if any(suite not in row or f'Some({group})' not in row for row in ours_rows):
                raise AssertionError('QUIC negotiated a different suite or group')
            if [row.split()[-1] for row in ours_rows] != [row.split()[-1] for row in peer_rows]:
                raise AssertionError('independent QUIC exporters differ')
            if scenario == 'early' and 'early=true' not in peer_rows[1]:
                raise AssertionError('independent peer did not confirm 0-RTT')
            if scenario == 'retry' and b'RETRY\n' not in server_output:
                raise AssertionError('Retry scenario did not issue Retry')
            if scenario == 'reset' and b'RESET_RECEIVED\n' not in client.stdout:
                raise AssertionError('client did not authenticate the stateless reset')
            # Qlog observes the peer's real packet-key transitions, including
            # phase > 0; calling force_key_update alone would not prove one.
            events = [json.loads(line) for path in logs.glob('*.sqlog') for line in path.read_text().split('\x1e') if line.strip()]
            updates = [event for event in events if event.get('name') == 'security:key_updated' and event.get('data', {}).get('key_phase', 0) > 0]
            if not updates:
                raise AssertionError('quic-go did not observe a packet-key update')
            print('PASS ' + name, flush=True)
            return {'case': name, 'key_updates': len(updates), 'connections': count}
        finally:
            if server.poll() is None:
                server.kill()
            if not (directory / 'server.stdout').exists():
                remaining, _ = server.communicate(timeout=5)
                (directory / 'server.stdout').write_bytes(prelude + remaining)
            server.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--reset-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / 'benchmark_results' / ('provider-quic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output(['go', 'version'], text=True).strip()
    if not version.startswith('go version go1.27.2 '):
        raise RuntimeError('quic-go peer requires Go 1.27.2')
    provenance = {'host': platform.platform(), 'go': version, 'quic-go': 'v0.61.0', 'commit': '579ee19d5b54c4f9320ffca668113c3513a138e5',
                  'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    shutil.copytree(ROOT / 'rscrypto-rustls', output / 'source/rscrypto-rustls')
    shutil.copytree(ROOT / 'scripts/test/provider_quic_peer', output / 'source/scripts/test/provider_quic_peer')
    for path in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'scripts/test/provider_quic.py', 'scripts/test/provider_interop.py']:
        destination = output / 'source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    with (output / 'build.log').open('wb') as log:
        subprocess.run(['cargo', 'build', '--locked', '-p', 'rscrypto-rustls', '--release', '--features', 'quinn,ml-dsa', '--example', 'quic_interop'], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        subprocess.run(['go', 'build', '-mod=readonly', '-o', str(output / 'quic-go-peer'), '.'],
                       cwd=ROOT / 'scripts/test/provider_quic_peer', env={**os.environ, 'CGO_ENABLED': '0'}, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version=1'], cwd=ROOT))
    binary = Path(metadata['target_directory']) / 'release/examples/quic_interop'
    credentials = CREDENTIALS + MLDSA_CREDENTIALS
    cases = [(kind, suite, group, 'normal') for kind, suite, group in itertools.product(credentials, SUITES, GROUPS)]
    cases += [(kind, suite, 'X25519MLKEM768', scenario) for kind, suite, scenario in itertools.product(credentials, SUITES, ('retry', 'early'))]
    if args.quick:
        cases = [('p256', SUITES[0], 'X25519MLKEM768', scenario) for scenario in ('normal', 'retry', 'early')]
    if args.reset_only:
        cases = []
    cases.append(('p256', SUITES[0], 'X25519MLKEM768', 'reset'))
    completed = []
    for case, ours_client in itertools.product(cases, (True, False)):
        completed.append(run_case(output, binary, output / 'quic-go-peer', *case, ours_client))
        (output / 'results.json').write_text(json.dumps(completed, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(output)) + '\n'
                                           for path in sorted(output.rglob('*')) if path.is_file() and path != output / 'SHA256SUMS'))
    print(f'{len(completed)} cases / {sum(row["connections"] for row in completed)} connections; evidence: {output}')


if __name__ == '__main__':
    main()
