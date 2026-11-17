#!/usr/bin/env -S uv run python
"""Reuse local Mac qualification passes by Git tree and compiler.

A tree is qualified when it passed with the current compiler, or when the nearest
first-parent ancestor of HEAD with a pass differs from it only in paths that no
local check reads. Any other difference, and any compiler change, requires a new run.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MACOS_CONTEXT = 'rscrypto/macos'
# Paths that `just check-macos` never compiles, tests, or reads, apart from the Mac's
# uv pin, which scripts/lib/python.sh reads from .config/tooling.toml. The uv version was never part of
# a pass's identity, and CI runs every new uv release.
# README.md is not here: `src/lib.rs` includes it as crate documentation, so doctests run it.
INERT = re.compile(
  r'^(\.changes/|\.github/|benchmark_results/|docs/|\.config/tooling\.toml$'
  r'|CHANGELOG\.md$|CONTRIBUTING\.md$|SECURITY\.md$|THREAT_MODEL\.md$)'
)
# Bound the ancestor walk; an older pass is not worth the lookup.
ANCESTORS = 64
# `uv run` exports these for the project that started this process; it also puts VIRTUAL_ENV/bin first on PATH.
UV_RUN = ('UV', 'UV_LOCKED', 'UV_PROJECT', 'UV_RUN_RECURSION_DEPTH', 'VIRTUAL_ENV')


def git(*args):
  return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def environment():
  """Return this process's environment for a check in another worktree.

  The `uv run` variables would run its Python with the developer's project, lock, and virtual
  environment, which may change or disappear before the check ends.
  """
  env = {name: value for name, value in os.environ.items() if name not in UV_RUN}
  if venv := os.environ.get('VIRTUAL_ENV'):
    env['PATH'] = os.pathsep.join(entry for entry in env.get('PATH', '').split(os.pathsep)
                                  if entry != os.path.join(venv, 'bin'))
  return env


def compiler():
  channel = subprocess.check_output([ROOT / 'scripts/lib/toolchain.sh'], cwd=ROOT, text=True).strip()
  return subprocess.check_output(['rustc', f'+{channel}', '-vV'], cwd=ROOT, text=True).strip()


def records():
  return Path(git('rev-parse', '--path-format=absolute', '--git-common-dir')) / 'rscrypto-macos-qualified'


def macos_description(tree, rustc):
  """Bind a GitHub status to the tree and compiler distribution across host triples."""
  if not re.fullmatch(r'[0-9a-f]{40}', tree) or not re.search(r'^commit-hash: [0-9a-f]{40}$', rustc, re.MULTILINE):
    raise ValueError('Mac qualification needs a Git tree and a versioned Rust compiler')
  # Release verification runs on Linux. Every rustc -vV field except its host must match.
  identity = '\n'.join(line for line in rustc.splitlines() if not line.startswith('host: '))
  return f'tree={tree} rustc={hashlib.sha256(identity.encode()).hexdigest()}'


def passed(tree, rustc):
  record = records() / tree
  return record.is_file() and record.read_text() == rustc


def qualified(tree):
  rustc = compiler()
  if passed(tree, rustc):
    return f'Mac qualification already passed for tree {tree}'
  if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'HEAD'], cwd=ROOT, capture_output=True).returncode:
    return None
  for commit in git('rev-list', '--first-parent', f'--max-count={ANCESTORS}', 'HEAD').splitlines():
    ancestor = git('rev-parse', f'{commit}^{{tree}}')
    if not passed(ancestor, rustc):
      continue
    changed = [path for path in git('diff', '--name-only', ancestor, tree).splitlines() if path]
    if all(INERT.match(path) for path in changed):
      return f'Mac qualification passed for ancestor {commit[:12]}; this tree changes only unchecked paths'
    return None
  return None
