"""Check that committed assembly matches its provenance manifests.

  provenance.py check    fail if a manifest output differs from its committed file, or if
                         assembly under `src/` that rscrypto does not own has no manifest output

A manifest is `src/**/*_assembly_provenance.tsv`. An `output` row names an assembly input and
pins its SHA-256, either as `output PATH LINES BYTES SHA256` or as `output PATH SHA256 ...`.
A `variant OUTPUT SOURCE elf|macho` row selects the platform branches in a shared source.
Upstream `archive` and `member` hashes are recorded for review; this check does not download them.
Assembly that rscrypto owns carries `rscrypto contributors` in its copyright header.
"""

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHA256 = re.compile(r'[0-9a-f]{64}')
OWNED = 'Copyright (c) 2026 rscrypto contributors'


def outputs(manifest):
    """Yield each pinned output, its physical source and target, and its expected counts and digest."""
    rows = [line.split('\t') for line in manifest.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
    if ['schema', '1'] not in rows:
        raise ValueError(f'{manifest.relative_to(ROOT)}: missing `schema 1`')
    variants = {row[1]: (row[2], row[3]) for row in rows if row[0] == 'variant'}
    for row in rows:
        if row[0] != 'output':
            continue
        if len(row) >= 5 and row[2].isdigit() and row[3].isdigit():
            counts = int(row[2]), int(row[3]), row[4]
        else:
            counts = None, None, row[2]
        source, target = variants.pop(row[1], (row[1], None))
        yield row[1], source, target, *counts
    if variants:
        raise ValueError(f'{manifest}: variants without output rows: {list(variants)}')


def assembly_input(file, target):
    """Reconstruct the pinned bytes, including blank lines and the original table placement."""
    data = file.read_bytes()
    if target is not None:
        if target not in ('elf', 'macho'):
            raise ValueError(f'{file}: unknown assembly variant {target}')
        out = bytearray()
        branch = None
        for line in data.splitlines(keepends=True):
            if line == b'.if {macho}\n' and branch is None:
                branch = 'macho'
            elif line == b'.else // macho\n' and branch == 'macho':
                branch = 'elf'
            elif line == b'.endif // macho\n' and branch == 'elf':
                branch = None
            elif branch in (None, target):
                out.extend(line)
        if branch is not None:
            raise ValueError(f'{file}: unterminated platform branch')
        data = bytes(out)

    def table(match):
        name = match[1]
        fragment = file.with_name(name.decode() + '.inc').read_bytes()
        definition = re.fullmatch(rb'(?://[^\n]*\n|\n)*\.macro ' + name + rb'\n(.*?)\.endm\n', fragment, re.S)
        if not definition:
            raise ValueError(f'{file}: invalid table definition for {name.decode()}')
        return definition[1]

    return re.sub(rb'(?m)^ +((?:rscrypto_ed25519|rscrypto_x25519)_base_table)\n', table, data)


def check():
    errors = []
    covered = set()
    for manifest in sorted((ROOT / 'src').rglob('*_assembly_provenance.tsv')):
        for path, source, target, lines, size, digest in outputs(manifest):
            covered.add(source)
            file = ROOT / source
            if not SHA256.fullmatch(digest):
                errors.append(f'{path}: malformed SHA-256 in {manifest.name}')
            elif not file.is_file():
                errors.append(f'{path}: listed in {manifest.name} but missing')
            else:
                data = assembly_input(file, target)
                if hashlib.sha256(data).hexdigest() != digest:
                    errors.append(f'{path}: SHA-256 differs from {manifest.name}')
                if lines is not None and (data.count(b'\n'), len(data)) != (lines, size):
                    errors.append(f'{path}: line or byte count differs from {manifest.name}')
    for file in sorted(p for p in (ROOT / 'src').rglob('*') if p.suffix in ('.s', '.S')):
        path = file.relative_to(ROOT).as_posix()
        if path not in covered and OWNED not in file.read_text(encoding='utf-8').split('\n', 1)[0]:
            errors.append(f'{path}: derived assembly has no provenance manifest output')
    for error in errors:
        print(f'assembly provenance: {error}', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['check']):
        raise SystemExit('usage: provenance.py [check]')
    sys.exit(check())
