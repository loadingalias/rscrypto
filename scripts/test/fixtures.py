#!/usr/bin/env python3
"""Verify the complete checked-in vector payload inventory and its pinned hashes."""

import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METADATA = {"README.md", "NIST-NOTICE.txt"}


def verify(root):
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
         "testdata", "tests/vectors"], cwd=root, text=True
    ).split("\0")
    files = {Path(name) for name in names if name}
    manifests = sorted(path for path in files if path.name == "SHA256SUMS")
    if not manifests:
        raise ValueError("no vector checksum manifests found")
    for path in sorted(files):
        if any((root / part).is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"vector input is a symlink: {path}")
        if not (root / path).is_file():
            raise ValueError(f"missing vector input: {path}")
    expected = {}
    for manifest in manifests:
        lines = (root / manifest).read_text().splitlines()
        if not lines:
            raise ValueError(f"empty vector manifest: {manifest}")
        for number, line in enumerate(lines, 1):
            match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", line)
            if match is None or match[2] in {".", ".."}:
                raise ValueError(f"invalid vector checksum entry: {manifest}:{number}")
            path = manifest.parent / match[2]
            if path in expected:
                raise ValueError(f"duplicate vector checksum entry: {path}")
            expected[path] = match[1]
    payloads = {path for path in files if path.name not in METADATA | {"SHA256SUMS"}}
    missing = sorted(payloads - expected.keys())
    extra = sorted(expected.keys() - payloads)
    if missing or extra:
        raise ValueError(f"vector inventory mismatch: unlisted={list(map(str, missing))}; "
                         f"missing or non-payload={list(map(str, extra))}")
    for path, expected_digest in expected.items():
        with (root / path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_digest:
            raise ValueError(f"vector checksum mismatch: {path}")
    return len(manifests), len(payloads)


if __name__ == "__main__":
    try:
        manifests, payloads = verify(ROOT)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"vector fixtures failed: {error}")
    print(f"Vector fixtures: {payloads} payloads verified across {manifests} manifests")
