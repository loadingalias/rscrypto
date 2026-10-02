"""Check that committed assembly matches its provenance manifests.

  provenance.py check    fail if a manifest output differs from its committed file, or if
                         assembly under `src/` that rscrypto does not own has no manifest output

A manifest is `src/**/*_assembly_provenance.tsv`. An `output` row names a committed file and
pins its SHA-256, either as `output PATH LINES BYTES SHA256` or as `output PATH SHA256 ...`.
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
    """Yield `(path, lines, bytes, sha256)` for every output row; counts are None when absent."""
    rows = [line.split('\t') for line in manifest.read_text().splitlines() if line and not line.startswith('#')]
    if ['schema', '1'] not in rows:
        raise ValueError(f'{manifest.relative_to(ROOT)}: missing `schema 1`')
    for row in rows:
        if row[0] != 'output':
            continue
        if len(row) >= 5 and row[2].isdigit() and row[3].isdigit():
            yield row[1], int(row[2]), int(row[3]), row[4]
        else:
            yield row[1], None, None, row[2]


def check():
    errors = []
    covered = set()
    for manifest in sorted((ROOT / 'src').rglob('*_assembly_provenance.tsv')):
        for path, lines, size, digest in outputs(manifest):
            covered.add(path)
            file = ROOT / path
            if not SHA256.fullmatch(digest):
                errors.append(f'{path}: malformed SHA-256 in {manifest.name}')
            elif not file.is_file():
                errors.append(f'{path}: listed in {manifest.name} but missing')
            else:
                data = file.read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    errors.append(f'{path}: SHA-256 differs from {manifest.name}')
                if lines is not None and (data.count(b'\n'), len(data)) != (lines, size):
                    errors.append(f'{path}: line or byte count differs from {manifest.name}')
    for file in sorted(p for p in (ROOT / 'src').rglob('*') if p.suffix in ('.s', '.S')):
        path = file.relative_to(ROOT).as_posix()
        if path not in covered and OWNED not in file.read_text().split('\n', 1)[0]:
            errors.append(f'{path}: derived assembly has no provenance manifest output')
    for error in errors:
        print(f'assembly provenance: {error}', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['check']):
        raise SystemExit('usage: provenance.py [check]')
    sys.exit(check())
