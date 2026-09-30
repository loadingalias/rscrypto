"""Verify restored APT index state against the catalog's pinned Ubuntu snapshot.

Usage: apt_state.py verify LISTS PREFIX ARCHITECTURE SECTION

APT trusts every index file already present in its lists directory; it only
verifies signatures and hashes while `apt-get update` downloads them. A restored
lists directory therefore needs the same proof before APT may use it offline:

1. each pinned suite's InRelease matches the catalog's SHA-256 pin and Ubuntu's
   archive signature;
2. every other lists file is an index named by its suite's signed InRelease,
   with the signed size and SHA-256; and
3. each suite provides the main and universe package indexes for ARCHITECTURE.

APT itself rejects any cached .deb whose size or SHA-256 differs from these
verified indexes. Exit status 1 means the state is unusable: the caller discards
the lists and fetches them from the network.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import catalog

KEYRING = Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg')
COMPONENTS = ('main', 'universe')
# APT's own bookkeeping inside the lists directory; never index content.
BOOKKEEPING = {'lock', 'partial', 'auxfiles'}


class Unusable(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def signed_body(data):
    """Return the clearsigned text of an InRelease file, rejecting unsigned additions."""
    text = data.decode('utf-8')
    header = '-----BEGIN PGP SIGNED MESSAGE-----\n'
    start = '-----BEGIN PGP SIGNATURE-----\n'
    end = '-----END PGP SIGNATURE-----'
    if not text.startswith(header) or text.count(start) != 1 or not text.rstrip().endswith(end):
        raise Unusable('InRelease is not a single clearsigned message')
    armor, _, rest = text.partition('\n\n')
    if not armor.startswith(header.rstrip('\n')):
        raise Unusable('InRelease has no signed-message header')
    return rest.partition(start)[0]


def signed_indexes(body):
    """Map each path in the signed SHA256 field to (size, digest)."""
    indexes = {}
    in_field = False
    for line in body.splitlines():
        if line.startswith('SHA256:'):
            in_field = True
            continue
        if in_field and line.startswith(' '):
            digest, size, path = line.split()
            indexes[path] = (int(size), digest)
        elif in_field:
            break
    if not indexes:
        raise Unusable('InRelease signs no SHA256 indexes')
    return indexes


def verify_signature(path):
    result = subprocess.run(['gpgv', '--keyring', str(KEYRING), str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise Unusable(f'{path.name}: Ubuntu archive signature did not verify')


def verify(lists, prefix, architecture, pins):
    """Raise Unusable unless LISTS is a complete, verified copy of the pinned suites."""
    suites = sorted(pins, key=len, reverse=True)
    signed = {}
    for suite in suites:
        path = lists / f'{prefix}_dists_{suite}_InRelease'
        if not path.is_file():
            raise Unusable(f'missing InRelease for {suite}')
        if sha256(path) != pins[suite]:
            raise Unusable(f'{suite}: InRelease differs from the catalog pin')
        verify_signature(path)
        signed[suite] = {name.replace('/', '_'): entry
                         for name, entry in signed_indexes(signed_body(path.read_bytes())).items()}

    present = {suite: set() for suite in suites}
    for path in sorted(lists.iterdir()):
        if path.name in BOOKKEEPING:
            continue
        if not path.is_file() or path.is_symlink():
            raise Unusable(f'{path.name}: not a regular index file')
        head = f'{prefix}_dists_'
        if not path.name.startswith(head):
            raise Unusable(f'{path.name}: not an index for this snapshot source')
        remainder = path.name[len(head):]
        suite = next((suite for suite in suites if remainder.startswith(f'{suite}_')), None)
        if suite is None:
            raise Unusable(f'{path.name}: not an index of a pinned suite')
        name = remainder[len(suite) + 1:]
        if name == 'InRelease':
            continue
        entry = signed[suite].get(name)
        if entry is None:
            raise Unusable(f'{path.name}: not signed by its InRelease')
        if path.stat().st_size != entry[0] or sha256(path) != entry[1]:
            raise Unusable(f'{path.name}: differs from its signed size or SHA-256')
        present[suite].add(name)

    for suite in suites:
        for component in COMPONENTS:
            if not any(is_package_index(name, component, architecture) for name in present[suite]):
                raise Unusable(f'{suite}: missing {component} package index for {architecture}')


def is_package_index(name, component, architecture):
    """Whether NAME is COMPONENT's package index for ARCHITECTURE or one of its variants.

    APT on an x86-64-v3 host with architecture variants enabled fetches
    binary-amd64v3 instead of binary-amd64; both are signed indexes of the suite.
    """
    head = f'{component}_binary-{architecture}'
    if not name.startswith(head) or not name.endswith('_Packages'):
        return False
    variant = name[len(head):-len('_Packages')]
    return variant == '' or (variant.startswith('v') and variant[1:].isdigit())


def main():
    command, *args = sys.argv[1:]
    if command != 'verify' or len(args) != 4:
        raise ValueError('usage: apt_state.py verify LISTS PREFIX ARCHITECTURE SECTION')
    lists, prefix, architecture, section = args
    pins = catalog.read()[section]['inrelease']
    try:
        verify(Path(lists), prefix, architecture, pins)
    except Unusable as reason:
        print(f'APT state rejected: {reason}', file=sys.stderr)
        return 1
    print(f'APT state verified against the pinned {section} snapshot')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f'apt_state: {error}')
