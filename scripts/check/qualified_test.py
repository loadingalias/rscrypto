#!/usr/bin/env python3
"""Test qualification reuse: exact passes, inert-only descendants, and every invalidation."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import qualified


def run(trees, changed, passes, rustc='rustc 1'):
  """Evaluate `qualified('macos', 'candidate')` against fake history.

  `trees` maps first-parent commits (newest first) to trees, `changed` maps an ancestor
  tree to the paths that differ from the candidate, and `passes` maps passed trees to compilers.
  """
  with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    for tree, compiler in passes.items():
      (directory / tree).write_text(compiler)

    def git(*args):
      if args[0] == 'rev-list':
        return '\n'.join(trees)
      if args[0] == 'rev-parse':
        return trees[args[1].removesuffix('^{tree}')]
      if args[0] == 'diff':
        return '\n'.join(changed[args[2]])
      raise AssertionError(args)

    with patch.object(qualified, 'git', side_effect=git), patch.object(qualified, 'compiler', return_value=rustc), \
         patch.object(qualified, 'records', return_value=directory):
      return qualified.qualified('macos', 'candidate')


def main():
  history = {'c2': 'T2', 'c1': 'T1'}
  # An exact pass with the same compiler is reused; a compiler change invalidates it.
  assert run(history, {}, {'candidate': 'rustc 1'}) is not None
  assert run(history, {}, {'candidate': 'rustc 0'}) is None
  # A descendant that changes only unchecked paths reuses the nearest passing ancestor.
  inert = ['docs/platforms.md', '.changes/x.md', '.github/workflows/ci.yml', 'benchmark_results/OVERVIEW.md',
           '.config/tooling.toml', 'CHANGELOG.md', 'CONTRIBUTING.md', 'SECURITY.md', 'THREAT_MODEL.md']
  assert run(history, {'T2': inert}, {'T2': 'rustc 1'}) is not None
  assert run(history, {'T2': inert}, {'T2': 'rustc 0'}) is None
  # Any checked path forces a new run: source, tests, manifests, scripts, crate docs, and nested Markdown.
  for path in ('src/lib.rs', 'tests/x.rs', 'Cargo.toml', 'Cargo.lock', 'scripts/check/check.sh', 'justfile',
               'README.md', 'examples/README.md', 'src/notes.md', '.config/rail.toml', 'rust-toolchain.toml',
               'docs.rs', 'tools/ct-harness/Cargo.toml'):
    assert run(history, {'T2': [*inert, path]}, {'T2': 'rustc 1'}) is None, path
  # Only the nearest passing ancestor counts; an older pass cannot skip a relevant newer change.
  assert run(history, {'T2': ['src/lib.rs'], 'T1': inert}, {'T2': 'rustc 1', 'T1': 'rustc 1'}) is None
  # Ancestors without a pass are skipped until one has a pass.
  assert run(history, {'T1': inert}, {'T1': 'rustc 1'}) is not None
  # No pass anywhere means a full run.
  assert run(history, {}, {}) is None
  print('Qualification reuse regressions passed')


if __name__ == '__main__':
  main()
