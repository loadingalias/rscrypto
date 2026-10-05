#!/usr/bin/env python3
"""Qualification preserves evidence while pushes only wait for a durable background job."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import macos_async

ROOT = Path(__file__).resolve().parents[2]


def install_qualification(root):
    """Give a fixture repository the real qualification helper and a pinned compiler."""
    for name in ('scripts/check/qualified.py', 'scripts/check/macos_async.py', 'scripts/lib/python.sh'):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / name)
    toolchain = root / 'scripts/lib/toolchain.sh'
    toolchain.write_text('#!/bin/sh\necho pinned\n')
    toolchain.chmod(0o755)
    rustc = root / '.git/rustc'
    rustc.write_text('#!/bin/sh\necho "rustc nightly\ncommit-hash: ' + '1' * 40 + '\nhost: aarch64-apple-darwin\nrelease: ${COMPILER:-one}"\ncat "$(dirname "$0")/compiler-change" 2>/dev/null || true\n')
    rustc.chmod(0o755)


class MacOSCommit(unittest.TestCase):
    def test_gate_preserves_ci_modes_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts/check').mkdir(parents=True)
            (root / 'scripts/lib').mkdir()
            script = root / 'scripts/check/macos.sh'
            shutil.copy2(ROOT / 'scripts/check/macos.sh', script)
            for name, body in {
                'scripts/lib/toolchain.sh': 'echo aarch64-apple-darwin',
                'uname': 'if [ "$1" = -s ]; then echo Darwin; else echo arm64; fi',
                'just': '[ "$RUSTUP_TOOLCHAIN" = aarch64-apple-darwin ] && [ -z "${RSCRYPTO_SKIP_DOCTESTS:-}" ] || exit 9\necho "$*" >> calls; [ "$*" != "${FAIL_COMMAND:-}" ]',
            }.items():
                tool = root / name
                tool.write_text('#!/bin/sh\n' + body + '\n')
                tool.chmod(0o755)
            env = {**os.environ, 'BASH_ENV': '/dev/null',
                   'RUSTUP_TOOLCHAIN': 'wrong', 'RSCRYPTO_SKIP_DOCTESTS': 'true',
                   'PATH': str(root) + os.pathsep + os.environ['PATH']}
            self.assertEqual(subprocess.run([str(script)], env=env).returncode, 0)
            self.assertEqual((root / 'calls').read_text().splitlines(), [
                'ci-check', 'test --all --release', 'test --all --release --portable',
                'test-evidence', 'test-rsa-macos-asm'])
            (root / 'calls').unlink()
            result = subprocess.run([str(script)], env={**env, 'FAIL_COMMAND': 'test --all --release'})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / 'calls').read_text().splitlines(),
                             ['ci-check', 'test --all --release'])

    def test_hook_checks_source_and_propagates_validation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', directory], check=True)
            hook = root / '.git/hooks/pre-commit'
            shutil.copy2(ROOT / '.githooks/pre-commit', hook)
            install_qualification(root)
            subprocess.run(['git', '-C', directory, 'add', 'scripts'], check=True)
            tool = root / '.git/just'
            tool.write_text('#!/bin/sh\necho "$*" >> .git/calls\nexit "${CHECK_STATUS:-0}"\n')
            tool.chmod(0o755)
            env = {**os.environ, 'BASH_ENV': '/dev/null', 'PATH': str(root / '.git') + os.pathsep + os.environ['PATH']}
            source = root / 'source'
            source.write_text('staged')
            subprocess.run(['git', '-C', directory, 'add', 'source'], check=True)

            def run(**extra):
                return subprocess.run([str(hook)], cwd=root, env={**env, **extra},
                                      capture_output=True, text=True).returncode

            # A failing check propagates and records nothing; a passing check records its tree.
            self.assertEqual(run(CHECK_STATUS='7'), 7)
            self.assertEqual(run(), 0)
            source.write_text('unstaged')
            self.assertNotEqual(run(), 0)
            source.write_text('staged')
            (root / 'untracked').write_text('new source')
            self.assertNotEqual(run(), 0)
            self.assertEqual((root / '.git/calls').read_text().splitlines(),
                             ['ci-check', 'ci-check'])

            # A recorded tree, and a descendant that changes only unchecked paths, reuse the pass.
            (root / 'untracked').unlink()
            self.assertEqual(run(), 0)
            git = ['git', '-C', directory, '-c', 'user.name=t', '-c', 'user.email=t@t']
            subprocess.run([*git, 'commit', '-qm', 'checked', '--no-verify'], check=True)
            self.assertEqual(run(), 0)
            (root / 'docs').mkdir()
            (root / 'docs/guide.md').write_text('prose')
            subprocess.run([*git, 'add', 'docs'], check=True)
            self.assertEqual(run(), 0)
            self.assertEqual((root / '.git/calls').read_text().splitlines(), ['ci-check', 'ci-check'])
            # Checked paths and compiler changes run the check again.
            source.write_text('changed')
            subprocess.run([*git, 'add', 'source'], check=True)
            self.assertEqual(run(), 0)
            self.assertEqual(run(COMPILER='two'), 0)
            self.assertEqual(len((root / '.git/calls').read_text().splitlines()), 4)

class AsyncMacOS(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git('init', '-q')
        for name, value in [('user.name', 'Qualification Test'), ('user.email', 'test@example.invalid'),
                            ('commit.gpgsign', 'false')]:
            self.git('config', name, value)
        self.common = self.root / '.git'
        self.hook = self.common / 'hooks/pre-push'
        shutil.copy2(ROOT / '.githooks/pre-push', self.hook)
        install_qualification(self.root)
        (self.root / '.gitignore').write_text('__pycache__/\n')
        (self.root / 'source').write_text('first')
        for name, body in {
            'uname': 'echo Darwin arm64',
            'just': '''
common=$(git rev-parse --path-format=absolute --git-common-dir)
printf '%s %s\\n' "$*" "$(git rev-parse HEAD)" >> "$common/calls"
touch "$common/started"
while [ -f "$common/hold" ]; do sleep 0.05; done
[ ! -f "$common/interrupt" ] || { kill -KILL "$PPID"; exit 1; }
[ ! -f "$common/dirty" ] || echo changed > source
[ ! -f "$common/change-compiler" ] || echo changed > "$common/compiler-change"
exit "${CHECK_STATUS:-0}"
''',
        }.items():
            tool = self.common / name
            tool.write_text('#!/bin/sh\n' + body + '\n')
            tool.chmod(0o755)
        self.env = {**os.environ, 'BASH_ENV': '/dev/null', 'PATH': str(self.common) + os.pathsep + os.environ['PATH']}
        self.commit('first')
        # Release held workers before removing their fixture, even on an assertion failure.
        self.addCleanup(self.finish_workers)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def push(self, sha=None, remote=None, **extra):
        sha = sha or self.git('rev-parse', 'HEAD')
        update = f'refs/heads/main {sha} refs/heads/main {"0" * 40}\n'
        return subprocess.run([str(self.hook), 'origin', remote or str(self.root)], cwd=self.root,
                              env={**self.env, **extra}, input=update, capture_output=True, text=True, timeout=10)

    def jobs(self):
        return list((self.common / 'rscrypto-macos').glob('*/state.json'))

    def wait(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        logs = [p.with_name('log').read_text() for p in self.jobs()]
        self.fail(f'Worker did not reach expected state: {logs}')

    def finish_workers(self):
        (self.common / 'hold').unlink(missing_ok=True)
        def finished():
            for record in self.jobs():
                with record.with_name('lock').open('w') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return False
            return True
        self.wait(finished)

    def phases(self):
        return [json.loads(path.read_text())['phase'] for path in self.jobs()]

    def calls(self):
        path = self.common / 'calls'
        return path.read_text().splitlines() if path.exists() else []

    def test_push_returns_while_worker_is_held_and_deduplicates(self):
        (self.common / 'hold').touch()
        first = self.git('rev-parse', 'HEAD')
        self.assertEqual(self.push(GIT_DIR=str(self.common), GIT_WORK_TREE=str(self.root)).returncode, 0)
        self.wait(lambda: (self.common / 'started').exists())
        self.assertEqual(self.phases(), ['running'])
        self.assertEqual(self.push().returncode, 0)
        self.assertEqual(self.calls(), [f'check-macos {first}'])
        # Advancing the main checkout cannot change the source the held job qualifies.
        (self.root / 'source').write_text('second')
        second = self.commit('second')
        self.assertEqual(self.push().returncode, 0)
        self.assertEqual(self.calls(), [f'check-macos {first}'])
        self.finish_workers()
        self.assertEqual(self.phases(), ['passed', 'passed'])
        self.assertEqual(self.calls(), [f'check-macos {first}', f'check-macos {second}'])
        self.assertEqual(self.git('status', '--porcelain'), '')
        self.assertEqual(len(self.git('worktree', 'list', '--porcelain').split('worktree ')), 2)

    def test_failures_record_no_pass_and_can_be_retried(self):
        for marker, environment in [(None, {'CHECK_STATUS': '7'}), ('dirty', {}), ('change-compiler', {})]:
            with self.subTest(marker=marker):
                if marker:
                    (self.common / marker).touch()
                self.assertEqual(self.push(**environment).returncode, 0)
                self.finish_workers()
                self.assertEqual(self.phases(), ['failed'])
                self.assertFalse(list((self.common / 'rscrypto-macos-qualified').glob('*')))
                if marker:
                    (self.common / marker).unlink()
                (self.common / 'compiler-change').unlink(missing_ok=True)
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        self.assertEqual(self.phases(), ['passed'])

    def test_reuses_exact_and_inert_pass_but_invalidates_source_and_compiler(self):
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        for _ in range(2):
            self.assertEqual(self.push().returncode, 0)
            self.finish_workers()
        (self.root / 'docs').mkdir()
        (self.root / 'docs/note.md').write_text('unchecked prose')
        self.commit('note')
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        self.assertEqual(len(self.calls()), 1)
        (self.root / 'source').write_text('changed source')
        self.commit('source')
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        self.assertEqual(self.push(COMPILER='two').returncode, 0)
        self.finish_workers()
        self.assertEqual(len(self.calls()), 3)

    def test_hook_rejects_dirty_or_other_source_and_ignores_deletions(self):
        self.assertNotEqual(self.push('1' * 40).returncode, 0)
        stale = subprocess.run(['scripts/lib/python.sh', 'scripts/check/macos_async.py', 'queue', str(self.root), '1' * 40],
                               cwd=self.root, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn('Checkout changed', stale.stderr)
        (self.root / 'source').write_text('dirty')
        self.assertNotEqual(self.push().returncode, 0)
        (self.root / 'source').write_text('first')
        (self.root / 'untracked').touch()
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual(self.push('0' * 40).returncode, 0)
        self.assertEqual(self.jobs(), [])

    def test_interrupted_worker_is_reported_and_recovered(self):
        (self.common / 'interrupt').touch()
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        result = subprocess.run(['scripts/lib/python.sh', 'scripts/check/macos_async.py', 'status'],
                                cwd=self.root, env=self.env, capture_output=True, text=True, check=True)
        self.assertIn('interrupted', result.stdout)
        self.assertFalse(list((self.common / 'rscrypto-macos-qualified').glob('*')))
        (self.common / 'interrupt').unlink()
        self.assertEqual(self.push().returncode, 0)
        self.finish_workers()
        self.assertEqual(self.phases(), ['passed'])
        self.assertEqual(len(self.calls()), 2)

    def test_queued_compiler_change_fails_before_testing(self):
        directory = self.common / 'rscrypto-macos'
        directory.mkdir()
        with (directory / 'build.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.assertEqual(self.push().returncode, 0)
            (self.common / 'compiler-change').write_text('changed')
        self.finish_workers()
        self.assertEqual(self.phases(), ['failed'])
        self.assertEqual(self.calls(), [])

    def test_publication_failure_retains_local_pass_for_retry(self):
        gh = self.common / 'gh'
        gh.write_text('#!/bin/sh\necho "HTTP 403" >&2\nexit 1\n')
        gh.chmod(0o755)
        remote = 'git@github.com:example/rscrypto.git'
        self.assertEqual(self.push(remote=remote).returncode, 0)
        self.finish_workers()
        self.assertEqual(self.phases(), ['publication-failed'])
        self.assertTrue(json.loads(self.jobs()[0].read_text())['qualified'])
        gh.write_text('#!/bin/sh\nif [ "$2" = "--method" ]; then cat > "$(dirname "$0")/status-payload"; fi\n')
        self.assertEqual(self.push(remote=remote).returncode, 0)
        self.finish_workers()
        self.assertEqual(self.phases(), ['published'])
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(json.loads((self.common / 'status-payload').read_text())['state'], 'success')
        (self.root / 'source').write_text('failing source')
        self.commit('failing source')
        self.assertEqual(self.push(remote=remote, CHECK_STATUS='7').returncode, 0)
        self.finish_workers()
        self.assertEqual(json.loads((self.common / 'status-payload').read_text())['state'], 'failure')
        self.assertFalse((self.common / 'rscrypto-macos-qualified' / self.git('rev-parse', 'HEAD^{tree}')).exists())

    def test_github_visibility_retry_and_exact_status(self):
        state = {'repository': 'example/rscrypto', 'commit': '2' * 40, 'tree': '3' * 40,
                 'compiler': 'rustc nightly\ncommit-hash: ' + '1' * 40 + '\nhost: aarch64-apple-darwin'}
        missing = subprocess.CompletedProcess([], 1, stderr='HTTP 404')
        found = subprocess.CompletedProcess([], 0)
        with patch.object(macos_async.subprocess, 'run', side_effect=[missing, found, found]) as run, \
             patch.object(macos_async.time, 'sleep') as sleep:
            macos_async.publish(state, True)
        sleep.assert_called_once_with(5)
        args, kwargs = run.call_args
        self.assertIn(f'repos/example/rscrypto/statuses/{"2" * 40}', args[0])
        payload = json.loads(kwargs['input'])
        self.assertEqual(payload['state'], 'success')
        self.assertEqual(payload['context'], 'rscrypto/macos')
        self.assertTrue(payload['description'].startswith('tree=' + '3' * 40 + ' rustc='))
        denied = subprocess.CompletedProcess([], 1, stderr='HTTP 403')
        with patch.object(macos_async.subprocess, 'run', return_value=denied) as run, \
             self.assertRaisesRegex(ValueError, 'HTTP 403'):
            macos_async.publish(state, True)
        run.assert_called_once()
        for remote in ('git@github.com:example/rscrypto.git', 'https://github.com/example/rscrypto',
                       'ssh://git@github.com/example/rscrypto.git'):
            self.assertEqual(macos_async.repository(remote), 'example/rscrypto')
        self.assertIsNone(macos_async.repository('https://github.com.evil.invalid/example/rscrypto.git'))


if __name__ == '__main__':
    unittest.main()
