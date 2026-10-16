#!/usr/bin/env -S uv run python
"""Require the release source commit's Mac qualification before cargo-rail releases it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/check'))
from qualified import MACOS_CONTEXT, compiler, macos_description


def run(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def github(path):
    request = urllib.request.Request(
        f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}{path}',
        headers={'Authorization': f'Bearer {os.environ["GH_TOKEN"]}', 'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
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
    return sha


def require_macos(sha):
    expected = macos_description(run('git', 'rev-parse', f'{sha}^{{tree}}'), compiler())
    # GitHub returns newest statuses first. Never fall back to an older success
    # after the latest result for this context failed or used another compiler.
    page = 1
    while True:
        statuses = github(f'/commits/{sha}/statuses?per_page=100&page={page}') or []
        for status in statuses:
            if status['context'].casefold() == MACOS_CONTEXT.casefold():
                if status['state'] != 'success' or status['description'] != expected:
                    raise ValueError('The latest Mac qualification did not pass for this tree and compiler')
                return
        if len(statuses) < 100:
            raise ValueError('Missing Mac qualification; run just qualify-macos on the physical Mac and wait for its status')
        page += 1


def main():
    operation, = sys.argv[1:]
    if operation != 'preflight':
        raise ValueError('usage: release.py preflight')
    sha = candidate()
    require_macos(sha)
    print(f'Release source {sha} has its Mac qualification')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        raise SystemExit(f'Release failed: {error}') from error
