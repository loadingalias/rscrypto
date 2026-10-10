#!/usr/bin/env python3
"""Run `just ci-check` on the staged tree for the pre-commit hook.

Unstaged edits and untracked files stay in the developer's checkout and never enter the
check, so several sessions can commit disjoint paths from one checkout. The staged tree is
written into one persistent detached worktree in the common Git directory. Git rewrites only
the files that differ from the previous check; their new timestamps are later than every
earlier build, so the worktree's own Cargo target directory rebuilds exactly what changed.
No other checkout may use that target directory: Cargo decides freshness by timestamp.
"""

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

import qualified

ROOT = qualified.ROOT


def main():
    # The hook environment selects the index being committed: `git commit -a` and
    # `git commit PATH` use a temporary index, which `write-tree` reads.
    hook = dict(os.environ)
    tree = qualified.git('write-tree')
    head = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'HEAD^{commit}'], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    if not head:
        raise SystemExit('The staged check starts from an existing commit; create the first commit separately.')
    # Hook-local Git variables must not redirect commands in the check worktree
    # back to the developer's index, HEAD, or working directory.
    for name in qualified.git('rev-parse', '--local-env-vars').splitlines():
        os.environ.pop(name, None)
    state = Path(qualified.git('rev-parse', '--path-format=absolute', '--git-common-dir')) / 'rscrypto-ci-check'
    state.mkdir(exist_ok=True)
    checkout = state / 'checkout'
    with (state / 'lock').open('w') as lock:
        # A second session's commit waits here; the worktree holds one tree at a time.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (checkout / '.git').is_file():
            shutil.rmtree(checkout, ignore_errors=True)
            subprocess.run(['git', 'worktree', 'prune'], cwd=ROOT, check=True)
            subprocess.run(['git', 'worktree', 'add', '--quiet', '--detach', '--no-checkout', str(checkout), head],
                           cwd=ROOT, check=True)
        qualified.ROOT = checkout
        # HEAD only supplies the ancestors whose passes can be reused. With --reset, read-tree
        # also restores tracked files changed in the worktree; clean removes everything else.
        qualified.git('update-ref', '--no-deref', 'HEAD', head)
        qualified.git('read-tree', '--reset', '-u', tree)
        qualified.git('clean', '-ffdxq')
        rustc = qualified.compiler()
        reason = qualified.qualified('ci-check', tree)
        if reason:
            print(reason)
            return 0
        target = state / 'target'
        env = {**os.environ, 'CARGO_TARGET_DIR': str(target),
               'RSCRYPTO_INDEPENDENT_LINT_TARGET_DIR': str(target / 'independent-lints')}
        status = subprocess.run(['just', 'ci-check'], cwd=checkout, env=env).returncode
        if status:
            return status
        staged = subprocess.check_output(['git', 'write-tree'], cwd=ROOT, env=hook, text=True).strip()
        if (staged != tree or qualified.git('diff', '--name-only')
                or qualified.git('ls-files', '--others', '--exclude-standard') or qualified.compiler() != rustc):
            print('Staged source, checked files, or compiler changed during native checks; review and retry the commit.',
                  file=sys.stderr)
            return 1
        records = qualified.records('ci-check')
        records.mkdir(parents=True, exist_ok=True)
        (records / tree).write_text(rustc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
