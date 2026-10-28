#!/usr/bin/env -S uv run python
"""Generate independent TLS credential/signature fixtures with OpenSSL 4.0.3.

The output must be a new directory. Keys and randomized signatures are fresh;
this reproduces the fixture contracts, not their pinned bytes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/lib'))
import openssl_peer


def generate(openssl, output, profile):
    version = subprocess.check_output([openssl, 'version'], text=True).strip()
    if not version.startswith('OpenSSL 4.0.3 '):
        raise ValueError('fixture generation requires OpenSSL 4.0.3')
    if output.with_suffix('.commands.json').exists():
        raise ValueError('command provenance output already exists')
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    with tempfile.TemporaryDirectory(prefix='provider-credentials-') as temporary:
        work = Path(temporary)

        def run(*args):
            commands.append(['openssl', *args])
            subprocess.run([openssl, *args], cwd=work, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        (work / 'message.txt').write_bytes(b'rscrypto-rustls signature adapter fixture')
        profiles = [
            ('p256', 'EC', ['-pkeyopt', 'ec_paramgen_curve:P-256']),
            ('p384', 'EC', ['-pkeyopt', 'ec_paramgen_curve:P-384']),
            ('ed25519', 'ED25519', []),
            ('rsa3072', 'RSA', ['-pkeyopt', 'rsa_keygen_bits:3072']),
        ]
        if profile == 'mldsa':
            profiles = [(f'mldsa{size}', f'ML-DSA-{size}', []) for size in (44, 65, 87)]
        if profile == 'tls-bench':
            profiles += [(f'mldsa{size}', f'ML-DSA-{size}', []) for size in (44, 65, 87)]
            profiles = [(kind + '-chain', algorithm, options) for kind, algorithm, options in profiles]
            profiles += [('rsa4096', 'RSA', ['-pkeyopt', 'rsa_keygen_bits:4096'])]
        for kind, algorithm, options in profiles:
            chain = kind.endswith('-chain')
            roles = ('root', 'intermediate', 'leaf') if chain else ('root', 'leaf')
            for serial, role in enumerate(roles, 1):
                root = role == 'root'
                ca = role != 'leaf'
                run('genpkey', '-algorithm', algorithm, *options,
                    '-out', f'{kind}-{role}-key.pem')
                args = ['req', '-new', '-x509', '-key', f'{kind}-{role}-key.pem',
                        '-subj', f'/CN=rscrypto provider test {kind} {role}',
                        '-set_serial', str(serial),
                        '-not_before', '20260101000000Z', '-not_after', '20360101000000Z',
                        '-addext', f'basicConstraints=critical,CA:TRUE,pathlen:{int(root and chain)}' if ca
                        else 'basicConstraints=critical,CA:FALSE',
                        '-addext', 'keyUsage=critical,keyCertSign,cRLSign' if ca
                        else 'keyUsage=critical,digitalSignature',
                        '-out', f'{kind}-{role}.pem']
                if not root:
                    issuer = 'intermediate' if chain and not ca else 'root'
                    args += ['-CA', f'{kind}-{issuer}.pem', '-CAkey', f'{kind}-{issuer}-key.pem']
                if not ca:
                    args += ['-addext', 'subjectAltName=DNS:localhost',
                             '-addext', 'extendedKeyUsage=serverAuth,clientAuth']
                run(*args)
                run('x509', '-in', f'{kind}-{role}.pem', '-outform', 'DER',
                    '-out', f'{kind}-{role}.der')
            key = f'{kind}-leaf-key.pem'
            run('pkcs8', '-topk8', '-nocrypt', '-in', key, '-outform', 'DER',
                '-out', f'{kind}-leaf-key.der')
            for purpose in ('sslserver', 'sslclient'):
                intermediates = ['-untrusted', f'{kind}-intermediate.pem'] if chain else []
                run('verify', '-CAfile', f'{kind}-root.pem', '-purpose', purpose,
                    *intermediates, f'{kind}-leaf.pem')
            if profile == 'tls-bench':
                continue
            run('pkey', '-in', key, '-pubout', '-outform', 'DER', '-out', f'{kind}-spki.der')
            if kind in ('p256', 'p384'):
                run('ec', '-in', key, '-outform', 'DER', '-out', f'{kind}-sec1.der')
                for digest in ('sha256', 'sha384'):
                    run('dgst', '-' + digest, '-sign', key, '-out', f'{kind}-{digest}.sig',
                        'message.txt')
            elif kind == 'ed25519' or kind.startswith('mldsa'):
                run('pkeyutl', '-sign', '-inkey', key, '-rawin', '-in', 'message.txt',
                    '-out', f'{kind}.sig')
                if kind.startswith('mldsa'):
                    run('pkey', '-in', key, '-provparam', 'ml-dsa.output_formats=seed-only',
                        '-outform', 'DER', '-out', f'{kind}-leaf-seed-key.der')
                    run('pkeyutl', '-sign', '-inkey', key, '-rawin', '-in', 'message.txt',
                        '-pkeyopt', 'context-string:wrong context', '-out', f'{kind}-context.sig')
            else:
                run('rsa', '-in', key, '-traditional', '-outform', 'DER',
                    '-out', f'{kind}-pkcs1.der')
                run('rsa', '-in', key, '-RSAPublicKey_out', '-outform', 'DER',
                    '-out', f'{kind}-public.der')
                for digest in ('sha256', 'sha384', 'sha512'):
                    for padding in ('pkcs1', 'pss'):
                        options = ['-sigopt', 'rsa_padding_mode:' + padding]
                        if padding == 'pss':
                            options += ['-sigopt', 'rsa_pss_saltlen:digest']
                        run('dgst', '-' + digest, '-sign', key, *options,
                            '-out', f'{kind}-{padding}-{digest}.sig', 'message.txt')
        if profile == 'classical':
            run('genpkey', '-algorithm', 'RSA', '-pkeyopt', 'rsa_keygen_bits:2048',
                '-out', 'rsa2048-key.pem')
            run('pkcs8', '-topk8', '-nocrypt', '-in', 'rsa2048-key.pem', '-outform', 'DER',
                '-out', 'rsa2048-key.der')
            run('rsa', '-in', 'rsa2048-key.pem', '-RSAPublicKey_out', '-outform', 'DER',
                '-out', 'rsa2048-public.der')
            run('dgst', '-sha256', '-sign', 'rsa2048-key.pem', '-sigopt', 'rsa_padding_mode:pss',
                '-sigopt', 'rsa_pss_saltlen:digest', '-out', 'rsa2048-pss-sha256.sig', 'message.txt')
        for path in sorted(work.iterdir()):
            if path.suffix in ('.der', '.sig') or (path.name == 'message.txt' and profile != 'tls-bench'):
                shutil.copyfile(path, output / path.name)
    (output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n'
        for path in sorted(output.iterdir())))
    # Keep command provenance outside the fixture payload inventory.
    output.with_suffix('.commands.json').write_text(json.dumps(
        {'openssl': version, 'commands': commands}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--openssl', help='OpenSSL 4.0.3 binary; default: the repository-built peer')
    parser.add_argument('--profile', choices=('classical', 'mldsa', 'tls-bench'), default='classical')
    args = parser.parse_args()
    generate(args.openssl or str(openssl_peer.binary()), args.output, args.profile)


if __name__ == '__main__':
    main()
