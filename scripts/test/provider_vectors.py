#!/usr/bin/env python3
"""Extract provider vectors from the pinned RFC 8448 and RFC 9001 text files."""

import argparse
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE_HASHES = {
    8448: '6564d1376d1ec744fc7a9993da15ebc1b9be361908b166091f47ef605c537fba',
    9001: '3bbaecdf5afd278052a2c48348ce118c4ff8d0cf6b9915549858171b3f98a591',
}


def read_source(directory, number):
    data = (directory / f'rfc{number}.txt').read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[number]:
        raise ValueError(f'RFC {number} differs from the pinned source')
    return data.decode()


def tls_trace(source):
    section = source[source.index('\n3.  Simple 1-RTT Handshake\n'):
                     source.index('\n4.  Resumed 0-RTT Handshake\n')]
    section = '\n'.join(line for line in section.splitlines()
                        if not line.startswith(('Thomson', 'RFC 8448')))
    operations = []
    for part in re.split(r'\n   (?=\{(?:client|server)\})', section)[1:]:
        fields = []
        for line in part.splitlines():
            match = re.match(r'^      (.+?) \((\d+) octets\):[ ]*(.*)$', line)
            if match:
                name, length, first = match.groups()
                fields.append([name, int(length), first.replace('(empty)', '')])
            elif fields and re.fullmatch(r' +(?:[0-9a-f]{2} ?)+', line):
                fields[-1][2] += ' ' + line.strip()
        if not fields:
            continue
        result = ['# ' + ' '.join(part.split('\n\n')[0].split())]
        for name, length, text in fields:
            data = bytes.fromhex(text)
            if len(data) != length:
                raise ValueError(f'{name}: expected {length} bytes, found {len(data)}')
            result.append(name + ' = ' + data.hex())
        operations.append('\n'.join(result))
    if len(operations) != 38:
        raise ValueError('the complete simple handshake trace must contain 38 populated steps')
    return '\n\n'.join(operations) + '\n'


def quic_packets(source, start, end):
    section = source[source.index('\n' + start + '\n'):source.index('\n' + end + '\n')]
    blocks = re.findall(r'(?:^   [0-9a-f]+(?: [0-9a-f]+)*\n)+', section, re.MULTILINE)
    if len(blocks) != 3:
        raise ValueError('each Initial vector must contain a frame, header, and protected packet')
    return ''.join(name + ' = ' + bytes.fromhex(block).hex() + '\n'
                   for name, block in zip(('frame', 'header', 'packet'), blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, help='directory containing rfc8448.txt and rfc9001.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'rscrypto-rustls/tests/vectors')
    args = parser.parse_args()
    quic = read_source(args.sources, 9001)
    payloads = {
        'rfc8448.txt': tls_trace(read_source(args.sources, 8448)),
        'rfc9001-client.txt': quic_packets(quic, 'A.2.  Client Initial', 'A.3.  Server Initial'),
        'rfc9001-server.txt': quic_packets(quic, 'A.3.  Server Initial', 'A.4.  Retry'),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        (args.output / name).write_text(text)
    (args.output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(text.encode()).hexdigest() + '  ' + name + '\n'
        for name, text in sorted(payloads.items())))


if __name__ == '__main__':
    main()
