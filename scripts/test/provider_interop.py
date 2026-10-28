#!/usr/bin/env -S uv run python
"""Execute bounded TLS 1.3 and opt-in TLS 1.2 interoperability with pinned OpenSSL and Go peers."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import select
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import openssl_peer

FIXTURES = ROOT / 'rscrypto-rustls/tests/vectors/credentials'
PAYLOAD = b'rscrypto provider interoperability\n'
SUITES = ['TLS13_AES_128_GCM_SHA256', 'TLS13_AES_256_GCM_SHA384', 'TLS13_CHACHA20_POLY1305_SHA256']
GROUPS = {'X25519': 'X25519', 'secp256r1': 'P-256', 'secp384r1': 'P-384', 'X25519MLKEM768': 'X25519MLKEM768'}
# TLS 1.2 suites and their OpenSSL names. Hybrid groups are TLS 1.3 only.
TLS12_SUITES = {
    'TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384': 'ECDHE-ECDSA-AES256-GCM-SHA384',
    'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256': 'ECDHE-ECDSA-AES128-GCM-SHA256',
    'TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256': 'ECDHE-ECDSA-CHACHA20-POLY1305',
    'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384': 'ECDHE-RSA-AES256-GCM-SHA384',
    'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256': 'ECDHE-RSA-AES128-GCM-SHA256',
    'TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256': 'ECDHE-RSA-CHACHA20-POLY1305',
}
TLS12_GROUPS = ['X25519', 'secp256r1', 'secp384r1']
# In TLS 1.2, supported_groups also lists the curves an ECDSA certificate may use (RFC 8422
# section 5.1), so both endpoints list the certificate's curve after the group under test.
CERTIFICATE_CURVES = {'p256': 'secp256r1', 'p384': 'secp384r1'}
CREDENTIALS = ['p256', 'p384', 'ed25519', 'rsa3072']
MLDSA_CREDENTIALS = ['mldsa44', 'mldsa65', 'mldsa87']


def pem(source, destination, label):
    encoded = base64.b64encode(source.read_bytes()).decode()
    destination.write_text(f'-----BEGIN {label}-----\n' +
                           '\n'.join(encoded[i:i + 64] for i in range(0, len(encoded), 64)) +
                           f'\n-----END {label}-----\n')


def ready(process, prefix, timeout=10):
    """Read unbuffered startup lines without opening a spurious TLS connection."""
    deadline = time.monotonic() + timeout
    captured, line = bytearray(), bytearray()
    while time.monotonic() < deadline and len(captured) < 8192:
        if not select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))[0]:
            break
        byte = process.stdout.read(1)
        if not byte:
            raise RuntimeError(f'peer exited before readiness: {captured.decode(errors="replace")}')
        captured.extend(byte)
        line.extend(byte)
        if byte == b'\n':
            if line.startswith(prefix):
                return bytes(captured), line.decode().strip()
            line.clear()
    raise TimeoutError('peer did not publish its bounded readiness line')


def results(output):
    return [line for line in output.decode().splitlines() if line.startswith('RESULT ')]


def tls12(suite):
    return suite in TLS12_SUITES


def check_results(output, suite, group):
    rows = results(output)
    if len(rows) != 2 or 'Full' not in rows[0] or 'Resumed' not in rows[1]:
        raise AssertionError(f'expected a full and resumed connection: {rows}')
    # A resumed TLS 1.2 session performs no key exchange, so it reports no group.
    groups = [f'Some({group})', 'None' if tls12(suite) else f'Some({group})']
    if any(f'Some({suite})' not in row or expected not in row for row, expected in zip(rows, groups)):
        raise AssertionError(f'negotiation differs from the selected profile: {rows}')
    return [row.split()[-1] for row in rows]


def run_case(output, name, server_command, client_commands, *, suite, group, peer, ours_client, address=None):
    directory = output / name
    directory.mkdir()
    (directory / 'commands.json').write_text(json.dumps(
        {'server': server_command, 'clients': client_commands}, indent=2) + '\n')
    prefix = b'ACCEPT' if peer == 'openssl' and ours_client else b'LISTEN '
    prelude = b''
    with (directory / 'server.stderr').open('wb') as errors:
        server = subprocess.Popen(server_command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=errors, bufsize=0, stdin=subprocess.DEVNULL)
        try:
            prelude, line = ready(server, prefix)
            address = address or line.split()[1]
            clients = []
            for index, command in enumerate(client_commands):
                command = [arg.replace('{address}', address) for arg in command]
                with (directory / f'client-{index}.stderr').open('wb') as client_errors:
                    result = subprocess.run(command, cwd=ROOT, input=PAYLOAD * 2,
                                            stdout=subprocess.PIPE, stderr=client_errors, timeout=30)
                (directory / f'client-{index}.stdout').write_bytes(result.stdout)
                if result.returncode:
                    raise RuntimeError(f'client exited {result.returncode}: {directory}')
                clients.append(result.stdout)
            remaining, _ = server.communicate(timeout=15)
            server_output = prelude + remaining
            (directory / 'server.stdout').write_bytes(server_output)
            if server.returncode:
                raise RuntimeError(f'server exited {server.returncode}: {directory}')
            ours = clients[0] if ours_client else server_output
            exported = check_results(ours, suite, group)
            if peer == 'go':
                theirs = server_output if ours_client else clients[0]
                rows = results(theirs)
                if len(rows) != 2 or 'Full' not in rows[0] or 'Resumed' not in rows[1]:
                    raise AssertionError(f'Go resumption mismatch: {rows}')
                if exported != [row.split()[-1] for row in rows]:
                    raise AssertionError('independent Go exporters differ')
            elif not ours_client and any(value != PAYLOAD * 2 for value in clients):
                raise AssertionError('OpenSSL application payload differs')
            print('PASS ' + name, flush=True)
        finally:
            if server.poll() is None:
                server.kill()
            if not (directory / 'server.stdout').exists():
                remaining, _ = server.communicate(timeout=5)
                (directory / 'server.stdout').write_bytes(prelude + remaining)
            server.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer', choices=('all', 'openssl', 'go'), default='all')
    parser.add_argument('--profile', choices=('all', 'classical', 'mldsa'), default='all')
    parser.add_argument('--protocol', choices=('all', 'tls13', 'tls12'), default='all',
                        help='TLS 1.2 runs only the classical credentials')
    parser.add_argument('--quick', action='store_true', help='one profile for harness diagnosis')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('the process harness requires POSIX pipe readiness; Rust tests cover other native hosts')
    output = args.output or ROOT / 'benchmark_results' / ('provider-interop-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    versions = {'host': platform.platform(), 'profile': args.profile, 'protocol': args.protocol,
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    peers = ['openssl', 'go'] if args.peer == 'all' else [args.peer]
    # The repository builds and caches its pinned OpenSSL peer; Go comes from PATH.
    executables = {'openssl': str(openssl_peer.binary()) if 'openssl' in peers else 'openssl', 'go': 'go'}
    for peer in peers:
        version = subprocess.check_output([executables[peer], 'version'], text=True).strip()
        expected = 'OpenSSL 4.0.3 ' if peer == 'openssl' else 'go version go1.27.3 '
        if not version.startswith(expected):
            raise RuntimeError(f'{peer}: expected {expected}, found {version}')
        versions[peer] = version
    (output / 'provenance.json').write_text(json.dumps(versions, indent=2) + '\n')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    shutil.copytree(ROOT / 'rscrypto-rustls', output / 'source/rscrypto-rustls')
    for path in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'scripts/test/provider_peer.go', 'scripts/test/provider_interop.py'):
        destination = output / 'source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    with (output / 'build.log').open('wb') as log:
        subprocess.run(['cargo', 'build', '--locked', '-p', 'rscrypto-rustls', '--release', '--features', 'ml-dsa', '--example', 'interop'],
                       cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if 'go' in peers:
            subprocess.run(['go', 'build', '-o', str(output / 'go-peer'), 'scripts/test/provider_peer.go'],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version=1'], cwd=ROOT))
    binary = str(Path(metadata['target_directory']) / 'release/examples/interop')
    credentials = ((CREDENTIALS if args.profile != 'mldsa' else []) +
                   (MLDSA_CREDENTIALS if args.profile != 'classical' else []))
    cases = []
    if args.protocol != 'tls12':
        cases += itertools.product(credentials, SUITES, GROUPS)
    if args.protocol != 'tls13':
        cases += [(kind, suite, group) for kind in credentials if kind in CREDENTIALS
                  for suite in TLS12_SUITES if ('_RSA_' in suite) == (kind == 'rsa3072')
                  for group in TLS12_GROUPS]
    if args.quick:
        cases = cases[:1] if args.protocol == 'tls12' else [(credentials[0], SUITES[0], 'X25519MLKEM768')]
    converted = output / 'pem'
    converted.mkdir()
    for kind in credentials:
        for role in ('root', 'leaf'):
            pem(FIXTURES / f'{kind}-{role}.der', converted / f'{kind}-{role}.pem', 'CERTIFICATE')
        pem(FIXTURES / f'{kind}-leaf-key.der', converted / f'{kind}-key.pem', 'PRIVATE KEY')
    completed = []
    for peer, (kind, suite, group), ours_client in itertools.product(peers, cases, (True, False)):
        name = f'{peer}-{kind}-{suite}-{group}-ours-' + ('client' if ours_client else 'server')
        common = [str(FIXTURES), kind, suite, group, '2']
        curve = CERTIFICATE_CURVES.get(kind) if tls12(suite) else None
        extra = [curve] if curve not in (None, group) else []
        # Servers choose in their own order, so the group under test stays the negotiated group.
        offered = '+'.join([group, *extra])
        ours = [binary, 'client' if ours_client else 'server', '{address}' if ours_client else '127.0.0.1:0',
                str(FIXTURES), kind, suite, offered, '2', 'reverse' if peer == 'openssl' and ours_client else 'echo']
        if peer == 'go':
            theirs = [str(output / 'go-peer'), 'server' if ours_client else 'client',
                      '127.0.0.1:0' if ours_client else '{address}', *common]
            run_case(output, name, theirs if ours_client else ours, [ours if ours_client else theirs],
                     suite=suite, group=group, peer=peer, ours_client=ours_client)
        else:
            protocol = (['-tls1_2', '-cipher', TLS12_SUITES[suite]] if tls12(suite) else
                        ['-tls1_3', '-ciphersuites', suite.replace('TLS13_', 'TLS_')])
            credentials = ['-cert', str(converted / f'{kind}-leaf.pem'), '-key', str(converted / f'{kind}-key.pem'),
                           '-CAfile', str(converted / f'{kind}-root.pem'), '-verify_return_error',
                           '-no-CApath', '-no-CAstore', '-alpn', 'provider-test',
                           *protocol]
            groups = ['-groups', ':'.join(GROUPS[name] for name in [group, *extra])]
            if ours_client:
                with socket.socket() as available:
                    available.bind(('127.0.0.1', 0))
                    address = f'127.0.0.1:{available.getsockname()[1]}'
                server = [executables['openssl'], 's_server', '-accept', address, '-rev', '-naccept', '2', '-Verify', '1',
                          *credentials, *groups, '-serverpref']
                run_case(output, name, server, [ours], suite=suite, group=group, peer=peer,
                         ours_client=True, address=address)
            else:
                session = str(output / (name + '.session.pem'))
                client = [executables['openssl'], 's_client', '-connect', '{address}', '-servername', 'localhost',
                          '-verify_hostname', 'localhost', '-quiet', '-ign_eof', *credentials, *groups]
                run_case(output, name, ours, [client + ['-sess_out', session], client + ['-sess_in', session]],
                         suite=suite, group=group, peer=peer, ours_client=False)
        completed.append(name)
        (output / 'results.json').write_text(json.dumps({'passed': completed, 'connections': len(completed) * 2}, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(output)) + '\n'
        for path in sorted(output.rglob('*')) if path.is_file() and path != output / 'SHA256SUMS'))
    print(f'{len(completed)} profiles, {len(completed) * 2} full/resumed connections; evidence: {output}')


if __name__ == '__main__':
    main()
