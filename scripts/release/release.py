#!/usr/bin/env python3
"""Publish a prepared, qualified commit; reconcile retries by package checksum."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def run(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def registry(version):
    request = urllib.request.Request(
        f'https://crates.io/api/v1/crates/rscrypto/{version}',
        headers={'User-Agent': 'rscrypto-release (https://github.com/loadingalias/rscrypto)'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)['version']
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def github(path, payload=None):
    request = urllib.request.Request(
        f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}{path}',
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': f'Bearer {os.environ["GH_TOKEN"]}',
                 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if payload is None and error.code == 404:
            return None
        raise


def candidate():
    if os.environ.get('GITHUB_REF') != 'refs/heads/main':
        raise ValueError('Release must be dispatched on main')
    sha = os.environ['GITHUB_SHA']
    if run('git', 'rev-parse', 'HEAD') != sha:
        raise ValueError('Checkout differs from the triggering commit')
    if run('git', 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Release checkout has tracked changes')
    package = tomllib.loads((ROOT / 'Cargo.toml').read_text())['package']
    version = package['version']
    if package['name'] != 'rscrypto' or not re.fullmatch(r'\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?', version):
        raise ValueError('Invalid release package/version')
    sections = re.split(r'^## ', (ROOT / 'CHANGELOG.md').read_text(), flags=re.MULTILINE)
    if len(sections) < 2 or not sections[1].startswith(f'[{version}]'):
        raise ValueError('The first changelog entry must match Cargo.toml')
    notes = sections[1].partition('\n')[2].strip()
    if not notes:
        raise ValueError('Release notes are empty')
    pending = list((ROOT / '.changes').glob('*.md'))
    if any(path.name != 'README.md' for path in pending):
        raise ValueError('Prepare the release and consume pending .changes entries first')
    tag = f'v{version}'
    refs = run('git', 'ls-remote', '--tags', 'origin', f'refs/tags/{tag}', f'refs/tags/{tag}^{{}}')
    if refs:
        objects = dict(line.split()[::-1] for line in refs.splitlines())
        target = objects.get(f'refs/tags/{tag}^{{}}', objects.get(f'refs/tags/{tag}'))
        if target != sha:
            raise ValueError(f'{tag} already identifies a different commit')
    return version, tag, sha, notes


def published(version, archive):
    existing = registry(version)
    if existing is None:
        return False
    if existing['yanked'] or existing['checksum'] != hashlib.sha256(archive.read_bytes()).hexdigest():
        raise ValueError('Existing crates.io version is yanked or has different package bytes')
    return True


def main():
    operation, = sys.argv[1:]
    if operation not in ('preflight', 'package', 'publish'):
        raise ValueError('usage: release.py {preflight|package|publish}')
    version, tag, sha, notes = candidate()
    if operation == 'preflight':
        print(f'Qualify rscrypto {version} at {sha}')
        return
    archive = ROOT / 'target/package' / f'rscrypto-{version}.crate'
    if operation == 'package':
        run('cargo', 'package', '--locked')
        # Package generation must not have altered the candidate's tracked files.
        candidate()
        exists = published(version, archive)
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
            output.write(f'published={str(exists).lower()}\n')
        return
    if not published(version, archive):
        # The package was verified before minting the short-lived registry token.
        run('cargo', 'publish', '--locked', '--no-verify', '--registry', 'crates-io')
        if not published(version, archive):
            raise ValueError('Registry has not exposed the published checksum; rerun failed jobs')
    # Create the immutable source tag only after the registry upload is confirmed.
    refs = run('git', 'ls-remote', '--tags', 'origin', f'refs/tags/{tag}')
    if not refs:
        github('/git/refs', {'ref': f'refs/tags/{tag}', 'sha': sha})
    # A retry must also reject tags changed since initial candidate validation.
    candidate()
    existing = github(f'/releases/tags/{tag}')
    if existing:
        if (existing['draft'] or (existing['body'] or '').strip() != notes
                or existing['prerelease'] != ('-' in version)):
            raise ValueError('Existing GitHub Release differs; inspect it before retrying')
        print(existing['html_url'])
        return
    release = github('/releases', {'tag_name': tag, 'target_commitish': sha, 'name': tag, 'body': notes,
                                   'prerelease': '-' in version})
    print(release['html_url'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        raise SystemExit(f'Release failed: {error}') from error
