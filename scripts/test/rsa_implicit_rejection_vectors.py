#!/usr/bin/env -S uv run python
"""Write RSAES-PKCS1-v1_5 implicit-rejection vectors from independent sources.

Appendix B of the checksum-pinned draft-irtf-cfrg-rsa-guidance-10 text supplies
the published vectors. OpenSSL 4.0.3, which applies the draft's implicit
rejection by default, decrypts the Wycheproof invalid-padding cases and seeded
random ciphertexts, and must reproduce every Appendix B message. This script
performs no RSA arithmetic of its own.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/lib'))
import openssl_peer

ROOT = Path(__file__).resolve().parents[2]
DRAFT = 'draft-irtf-cfrg-rsa-guidance-10'
DRAFT_URL = 'https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-10.txt'
DRAFT_SHA256 = '9e2af1e3e15744ef3b6575184d97dcc5ca22af427f6c93235e1bf23bca9952e6'
OUTPUT = ROOT / 'testdata/rsa/implicit_rejection'
WYCHEPROOF = ROOT / 'testdata/rsa/wycheproof'
WYCHEPROOF_FILES = ('rsa_pkcs1_2048_test.json', 'rsa_pkcs1_3072_test.json', 'rsa_pkcs1_4096_test.json')
RANDOM_CIPHERTEXTS_PER_KEY = 32
RANDOM_SEED = 0x52534149  # "RSAI"


def appendix_b(source):
    text = source[source.index('\nAppendix B.  Test Vectors\n'):]
    # Drop page footers, headers, and form feeds before splitting sections.
    lines = [line for line in text.splitlines()
             if not line.startswith(('Kario ', 'Internet-Draft ')) and line != '\f']
    sections, current = [], None
    for line in lines:
        match = re.match(r'^B\.(\d+)\.(\d+)\.  (.+)$', line)
        if match:
            current = {'key': int(match[1]), 'case': int(match[2]), 'name': match[3], 'lines': []}
            sections.append(current)
        elif current is not None:
            current['lines'].append(line.strip())
    keys = {}
    for section in sections:
        body = section['lines']
        if section['case'] == 1:
            pem = body[body.index('-----BEGIN PRIVATE KEY-----') + 1:body.index('-----END PRIVATE KEY-----')]
            keys[section['key']] = {'section': f'B.{section["key"]}', 'pkcs8': base64.b64decode(''.join(pem)).hex(),
                                    'cases': []}
            continue
        ciphertext = hex_block(body, 'Hex encoded ciphertext:')
        if 'The result of decryption is a message of length 0.' in body:
            message = ''
        elif any(label.startswith('ASCII encoded') for label in body):
            label = next(index for index, line in enumerate(body) if line.startswith('ASCII encoded'))
            message = next(line for line in body[label + 1:] if line).encode().hex()
        else:
            label = next(line for line in body if re.fullmatch(r'Hex encoded (decrypted )?message:', line))
            message = hex_block(body, label)
        keys[section['key']]['cases'].append({'section': f'B.{section["key"]}.{section["case"]}',
                                              'name': section['name'], 'ciphertext': ciphertext, 'message': message})
    if sorted(keys) != [1, 2, 3, 4] or any(len(key['cases']) != 12 for key in keys.values()):
        raise ValueError('Appendix B must contain four keys with twelve cases each')
    return [keys[index] for index in sorted(keys)]


def hex_block(body, label):
    start = body.index(label) + 1
    while not body[start]:
        start += 1
    end = start
    while end < len(body) and re.fullmatch(r'[0-9a-f]+', body[end]):
        end += 1
    if end == start:
        raise ValueError(f'no hex block after {label!r}')
    return ''.join(body[start:end])


class OpenSsl:
    def __init__(self, directory):
        self.directory = directory
        self.executable = str(openssl_peer.binary())
        version = subprocess.run([self.executable, 'version'], check=True, capture_output=True, text=True).stdout
        if not version.startswith('OpenSSL 4.0.3 '):
            raise ValueError('vector generation requires OpenSSL 4.0.3')
        self.version = version.strip()

    def key(self, name, pkcs8_hex):
        path = self.directory / f'{name}.der'
        path.write_bytes(bytes.fromhex(pkcs8_hex))
        modulus = subprocess.run([self.executable, 'pkey', '-inform', 'DER', '-in', path, '-noout', '-text'],
                                 check=True, capture_output=True, text=True).stdout
        bits = int(re.search(r'Private-Key: \((\d+) bit', modulus)[1])
        return path, bits

    def decrypt(self, key, ciphertext_hex, *, raw=False):
        ciphertext = self.directory / 'ciphertext.bin'
        ciphertext.write_bytes(bytes.fromhex(ciphertext_hex))
        padding = ['-pkeyopt', 'rsa_padding_mode:none'] if raw else [
            '-pkeyopt', 'rsa_padding_mode:pkcs1', '-pkeyopt', 'rsa_pkcs1_implicit_rejection:1']
        result = subprocess.run([self.executable, 'pkeyutl', '-decrypt', '-inkey', key, '-keyform', 'DER',
                                 *padding, '-in', ciphertext],
                                check=True, capture_output=True)
        return result.stdout.hex()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True, help=f'{DRAFT}.txt from {DRAFT_URL}')
    args = parser.parse_args()
    source = args.draft.read_bytes()
    if hashlib.sha256(source).hexdigest() != DRAFT_SHA256:
        raise ValueError(f'{args.draft} differs from the pinned {DRAFT} text')
    keys = appendix_b(source.decode())

    with tempfile.TemporaryDirectory(prefix='rscrypto-rsa-ir-') as temporary:
        openssl = OpenSsl(Path(temporary))
        rng = random.Random(RANDOM_SEED)
        random_cases = []
        for key in keys:
            path, bits = openssl.key(key['section'], key['pkcs8'])
            key['modulus_bits'] = bits
            for case in key['cases']:
                if openssl.decrypt(path, case['ciphertext']) != case['message']:
                    raise ValueError(f'OpenSSL disagrees with {DRAFT} {case["section"]}')
                case['encoded_message'] = openssl.decrypt(path, case['ciphertext'], raw=True)
            k = (bits + 7) // 8
            for _ in range(RANDOM_CIPHERTEXTS_PER_KEY):
                # A zero leading octet keeps the representative below the modulus.
                ciphertext = (b'\0' + rng.randbytes(k - 1)).hex()
                random_cases.append({'key': key['section'], 'ciphertext': ciphertext,
                                     'message': openssl.decrypt(path, ciphertext)})

        wycheproof_cases, sources = [], []
        for name in WYCHEPROOF_FILES:
            data = (WYCHEPROOF / name).read_bytes()
            sources.append({'file': name, 'sha256': hashlib.sha256(data).hexdigest()})
            suite = json.loads(data)
            for index, group in enumerate(suite['testGroups']):
                path, _ = openssl.key(f'{name}-{index}', group['privateKeyPkcs8'])
                for test in group['tests']:
                    if 'InvalidPkcs1Padding' in test['flags']:
                        wycheproof_cases.append({'file': name, 'tcId': test['tcId'],
                                                 'message': openssl.decrypt(path, test['ct'])})
        version = openssl.version

    OUTPUT.mkdir(parents=True, exist_ok=True)
    draft = {'source': {'document': DRAFT, 'url': DRAFT_URL, 'sha256': DRAFT_SHA256}, 'keys': keys}
    oracle = {'oracle': version, 'command': 'openssl pkeyutl -decrypt -pkeyopt rsa_padding_mode:pkcs1 '
                                           '-pkeyopt rsa_pkcs1_implicit_rejection:1',
              'random_seed': RANDOM_SEED, 'wycheproof_sources': sources,
              'random': random_cases, 'wycheproof': wycheproof_cases}
    (OUTPUT / 'appendix_b.json').write_text(json.dumps(draft, indent=1) + '\n')
    (OUTPUT / 'openssl_4_0_3.json').write_text(json.dumps(oracle, indent=1) + '\n')
    print(f'{sum(len(key["cases"]) for key in keys)} Appendix B cases match OpenSSL; '
          f'{len(random_cases)} random and {len(wycheproof_cases)} Wycheproof cases written to {OUTPUT}')


if __name__ == '__main__':
    main()
