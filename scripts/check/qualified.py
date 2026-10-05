#!/usr/bin/env python3
"""Record and reuse local qualification passes by Git tree and compiler.

  qualified.py check KIND TREE    exit 0 if TREE is qualified for KIND, else 1
  qualified.py record KIND TREE   record that TREE passed KIND with the current compiler

A tree is qualified when it passed with the current compiler, or when the nearest
first-parent ancestor of HEAD with a pass differs from it only in paths that no
local check reads. Any other difference, and any compiler change, requires a new run.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KINDS = {'ci-check', 'macos'}
MACOS_CONTEXT = 'rscrypto/macos'
# Paths that `just ci-check` and `just check-macos` never compile, test, or read.
# README.md is not here: `src/lib.rs` includes it as crate documentation, so doctests run it.
INERT = re.compile(
  r'^(\.changes/|\.github/|benchmark_results/|docs/|\.config/tooling\.toml$'
  r'|CHANGELOG\.md$|CONTRIBUTING\.md$|SECURITY\.md$|THREAT_MODEL\.md$)'
)
# Bound the ancestor walk; an older pass is not worth the lookup.
ANCESTORS = 64


def git(*args):
  return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def compiler():
  channel = subprocess.check_output([ROOT / 'scripts/lib/toolchain.sh'], cwd=ROOT, text=True).strip()
  return subprocess.check_output(['rustc', f'+{channel}', '-vV'], cwd=ROOT, text=True).strip()


def records(kind):
  return Path(git('rev-parse', '--path-format=absolute', '--git-common-dir')) / f'rscrypto-{kind}-qualified'


def macos_description(tree, rustc):
  """Bind a GitHub status to the tree and compiler distribution across host triples."""
  if not re.fullmatch(r'[0-9a-f]{40}', tree) or not re.search(r'^commit-hash: [0-9a-f]{40}$', rustc, re.MULTILINE):
    raise ValueError('Mac qualification needs a Git tree and a versioned Rust compiler')
  # Release verification runs on Linux. Every rustc -vV field except its host must match.
  identity = '\n'.join(line for line in rustc.splitlines() if not line.startswith('host: '))
  return f'tree={tree} rustc={hashlib.sha256(identity.encode()).hexdigest()}'


def passed(kind, tree, rustc):
  record = records(kind) / tree
  return record.is_file() and record.read_text() == rustc


def qualified(kind, tree):
  rustc = compiler()
  if passed(kind, tree, rustc):
    return f'{kind} already passed for tree {tree}'
  if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'HEAD'], cwd=ROOT, capture_output=True).returncode:
    return None
  for commit in git('rev-list', '--first-parent', f'--max-count={ANCESTORS}', 'HEAD').splitlines():
    ancestor = git('rev-parse', f'{commit}^{{tree}}')
    if not passed(kind, ancestor, rustc):
      continue
    changed = [path for path in git('diff', '--name-only', ancestor, tree).splitlines() if path]
    if all(INERT.match(path) for path in changed):
      return f'{kind} passed for ancestor {commit[:12]}; this tree changes only unchecked paths'
    return None
  return None


def main():
  if len(sys.argv) != 4 or sys.argv[1] not in ('check', 'record') or sys.argv[2] not in KINDS:
    raise SystemExit(__doc__)
  operation, kind, tree = sys.argv[1:]
  tree = git('rev-parse', '--verify', f'{tree}^{{tree}}')
  if operation == 'record':
    directory = records(kind)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / tree).write_text(compiler())
    return 0
  reason = qualified(kind, tree)
  if reason is None:
    return 1
  print(reason)
  return 0


if __name__ == '__main__':
  sys.exit(main())
