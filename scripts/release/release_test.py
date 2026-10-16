#!/usr/bin/env -S uv run python
"""Exercise the release preflight without forge writes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('release', Path(__file__).with_name('release.py'))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Release(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.git('config', 'commit.gpgsign', 'false')
        self.git('config', 'user.name', 'Release Test')
        self.git('config', 'user.email', 'release@example.invalid')
        (self.root / 'Cargo.toml').write_text('[package]\nname="rscrypto"\nversion="1.2.3"\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'source')
        self.sha = self.git('rev-parse', 'HEAD')
        self.enterContext(patch.object(release, 'ROOT', self.root))
        self.enterContext(patch.dict(os.environ, GITHUB_REF='refs/heads/main', GITHUB_SHA=self.sha))
        self.rustc = 'rustc nightly\ncommit-hash: ' + '1' * 40 + '\nhost: x86_64-unknown-linux-gnu'
        self.enterContext(patch.object(release, 'compiler', return_value=self.rustc))
        self.status_path = f'/commits/{self.sha}/statuses?per_page=100&page=1'
        self.mac_status = {'context': 'rscrypto/macos', 'state': 'success',
                           'description': release.macos_description(self.git('rev-parse', 'HEAD^{tree}'), self.rustc)}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def test_macos_gate_binds_source_compiler_and_latest_result(self):
        for statuses in ([], [{**self.mac_status, 'state': state} for state in ('failure', 'success')],
                         [{**self.mac_status, 'state': 'pending'}],
                         [{**self.mac_status, 'description': release.macos_description('0' * 40, self.rustc)}],
                         [{**self.mac_status, 'description': release.macos_description(self.git('rev-parse', 'HEAD^{tree}'), self.rustc + '\nLLVM version: other')}],
                         [{**self.mac_status, 'context': 'unrelated'}]):
            with self.subTest(statuses=statuses), \
                 patch.object(release, 'github', return_value=statuses) as github, \
                 patch('sys.argv', ['release.py', 'preflight']), self.assertRaisesRegex(ValueError, 'Mac qualification'):
                release.main()
            github.assert_called_once_with(self.status_path)
        with patch.object(release, 'github', return_value=[self.mac_status]), patch('sys.argv', ['release.py', 'preflight']):
            release.main()
            # A matching compiler from a different host is valid; a changed distribution is not.
            with patch.object(release, 'compiler', return_value=self.rustc.replace('x86_64-unknown-linux-gnu', 'aarch64-apple-darwin')):
                release.require_macos(self.sha)
            with patch.object(release, 'compiler', return_value=self.rustc + '\nLLVM version: changed'), \
                 self.assertRaisesRegex(ValueError, 'compiler'):
                release.require_macos(self.sha)
        unrelated = [{'context': 'other', 'state': 'success'}] * 100
        with patch.object(release, 'github', side_effect=[unrelated, [self.mac_status]]) as github:
            release.require_macos(self.sha)
        self.assertEqual(github.call_args.args, (f'/commits/{self.sha}/statuses?per_page=100&page=2',))

    def test_wrong_ref_sha_and_dirty_checkout_are_rejected(self):
        self.assertEqual(release.candidate(), self.sha)
        for key, value in [('GITHUB_REF', 'refs/heads/feature'), ('GITHUB_SHA', '0' * 40)]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}), self.assertRaises(ValueError):
                release.candidate()
        (self.root / 'Cargo.toml').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'tracked changes'):
            release.candidate()

    def test_unknown_operation_is_rejected_before_forge_access(self):
        for operation in ('package', 'publish'):
            with self.subTest(operation=operation), patch.object(release, 'github') as github, \
                 patch('sys.argv', ['release.py', operation]), self.assertRaisesRegex(ValueError, 'usage'):
                release.main()
            github.assert_not_called()


if __name__ == '__main__':
    unittest.main()
