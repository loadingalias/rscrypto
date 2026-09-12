#!/usr/bin/env python3
"""Commits must fail when macOS validation fails or the staged source differs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


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
                'just': 'echo "$*" >> calls; [ "$*" != "${FAIL_COMMAND:-}" ]',
            }.items():
                tool = root / name
                tool.write_text('#!/bin/sh\n' + body + '\n')
                tool.chmod(0o755)
            env = {**os.environ, 'BASH_ENV': '/dev/null',
                   'PATH': str(root) + os.pathsep + os.environ['PATH']}
            self.assertEqual(subprocess.run([str(script)], env=env).returncode, 0)
            self.assertEqual((root / 'calls').read_text().splitlines(), [
                'ci-check', 'test --all --release', 'test --all --release --portable',
                'test-rsa-macos-asm'])
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

            self.assertEqual(run(), 0)
            self.assertEqual((root / '.git/calls').read_text(), 'check-macos\n')
            self.assertEqual(run(CHECK_STATUS='7'), 7)
            source.write_text('unstaged')
            self.assertNotEqual(run(), 0)
            source.write_text('staged')
            (root / 'untracked').write_text('new source')
            self.assertNotEqual(run(), 0)
            self.assertEqual((root / '.git/calls').read_text().splitlines(),
                             ['check-macos', 'check-macos'])


if __name__ == '__main__':
    unittest.main()
