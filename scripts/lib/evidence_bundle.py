"""Bind transferred build evidence to the effective source tree and exact files.

The CI artifact service is the transport trust boundary. These hashes detect
mix-ups and corruption; they do not authenticate a compromised build runner.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def source_identity(root):
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root
    ).decode().split('\0')
    rows = []
    for name in sorted(set(names) - {''}):
        path = root / name
        if path.is_symlink():
            raise ValueError(f'source symlink is unsupported: {name}')
        rows.append([name, digest(path) if path.is_file() else None,
                     bool(path.stat().st_mode & 0o111) if path.exists() else False])
    return {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'sha256': hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest(),
    }


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError(f'unsafe evidence path: {name!r}')
    return path


def files(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'evidence symlink is unsupported: {path}')
        if path.is_file() and path != directory / 'bundle.json':
            result[path.relative_to(directory).as_posix()] = {
                'sha256': digest(path), 'bytes': path.stat().st_size,
                'executable': bool(path.stat().st_mode & 0o111),
            }
    return result


def seal(root, directory, kind, target, identity, metadata):
    if source_identity(root) != identity:
        raise ValueError('source changed during preparation')
    manifest = {'schema': 1, 'kind': kind, 'target': target, 'source': identity,
                'metadata': metadata, 'files': files(directory)}
    (directory / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(root, directory, kind, target):
    manifest = json.loads((directory / 'bundle.json').read_text())
    if (manifest.get('schema'), manifest.get('kind'), manifest.get('target')) != (1, kind, target):
        raise ValueError('wrong evidence bundle kind, schema, or target')
    if manifest.get('source') != source_identity(root):
        raise ValueError('evidence bundle source does not match this checkout')
    for name in manifest['files']:
        relative_path(name)
    if not manifest['files'] or files(directory) != manifest['files']:
        raise ValueError('evidence bundle files changed, are missing, or were added')
    return manifest


def pack(directory, archive):
    with tarfile.open(archive, 'x:gz') as output:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                output.add(path, arcname=path.relative_to(directory).as_posix(), recursive=False)


def unpack(archive, destination):
    # No links, devices, duplicate names, or path traversal, even in trusted CI artifacts.
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        seen = set()
        for member in members:
            relative_path(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError(f'invalid evidence archive member: {member.name}')
            seen.add(member.name)
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as data, path.open('xb') as output:
                shutil.copyfileobj(data, output)
            os.chmod(path, 0o755 if member.mode & 0o111 else 0o644)
