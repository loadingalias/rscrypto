#!/usr/bin/env python3
"""Exercise release refusal and recovery without registry or forge writes."""
import hashlib
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
        self.git('config', 'tag.gpgsign', 'false')
        self.git('config', 'user.name', 'Release Test')
        self.git('config', 'user.email', 'release@example.invalid')
        (self.root / 'Cargo.toml').write_text('[package]\nname="rscrypto"\nversion="1.2.3"\n')
        (self.root / 'CHANGELOG.md').write_text('# Changelog\n\n## [1.2.3]\n\n- Reviewed change.\n')
        (self.root / '.gitignore').write_text('/target\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'candidate')
        self.sha = self.git('rev-parse', 'HEAD')
        self.git('remote', 'add', 'origin', str(self.root))
        self.enterContext(patch.object(release, 'ROOT', self.root))
        self.enterContext(patch.dict(os.environ, GITHUB_REF='refs/heads/main', GITHUB_SHA=self.sha))
        self.archive = self.root / 'target/package/rscrypto-1.2.3.crate'
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b'verified crate bytes')
        self.record = {'yanked': False, 'checksum': hashlib.sha256(self.archive.read_bytes()).hexdigest()}
        self.rustc = 'rustc nightly\ncommit-hash: ' + '1' * 40 + '\nhost: x86_64-unknown-linux-gnu'
        self.enterContext(patch.object(release, 'compiler', return_value=self.rustc))
        self.status_path = f'/commits/{self.sha}/statuses?per_page=100&page=1'
        self.mac_status = {'context': 'rscrypto/macos', 'state': 'success',
                           'description': release.macos_description(self.git('rev-parse', 'HEAD^{tree}'), self.rustc)}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def forge(self, effect=None):
        def request(path, payload=None):
            if path == self.status_path and payload is None:
                return [self.mac_status]
            return effect(path, payload) if effect else None
        return request

    def test_macos_gate_binds_source_compiler_and_latest_result(self):
        for statuses in ([], [{**self.mac_status, 'state': state} for state in ('failure', 'success')],
                         [{**self.mac_status, 'state': 'pending'}],
                         [{**self.mac_status, 'description': release.macos_description('0' * 40, self.rustc)}],
                         [{**self.mac_status, 'description': release.macos_description(self.git('rev-parse', 'HEAD^{tree}'), self.rustc + '\nLLVM version: other')}],
                         [{**self.mac_status, 'context': 'unrelated'}]):
            for operation in ('preflight', 'package', 'publish'):
                with self.subTest(statuses=statuses, operation=operation), \
                     patch.object(release, 'github', return_value=statuses) as github, \
                     patch.object(release, 'registry') as registry, \
                     patch('sys.argv', ['release.py', operation]), self.assertRaisesRegex(ValueError, 'Mac qualification'):
                    release.main()
                registry.assert_not_called()
                github.assert_called_once_with(self.status_path)
        with patch.object(release, 'github', side_effect=self.forge()):
            release.require_macos(self.sha)
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

    def test_candidate_and_annotated_tag(self):
        self.assertEqual(release.candidate(), ('1.2.3', 'v1.2.3', self.sha, '- Reviewed change.'))
        self.git('tag', '-am', 'release', 'v1.2.3')
        self.assertEqual(release.candidate()[2], self.sha)

    def test_requested_version_must_match_the_package(self):
        with patch.dict(os.environ, RELEASE_VERSION='1.2.3'):
            self.assertEqual(release.candidate()[0], '1.2.3')
        for requested in ('1.2.4', 'v1.2.3', ''):
            with self.subTest(requested=requested), patch.dict(os.environ, RELEASE_VERSION=requested), \
                 self.assertRaisesRegex(ValueError, 'does not match Cargo.toml'):
                release.candidate()

    def test_wrong_ref_sha_dirty_notes_pending_and_tag_are_rejected(self):
        for key, value in [('GITHUB_REF', 'refs/heads/feature'), ('GITHUB_SHA', '0' * 40)]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}), self.assertRaises(ValueError):
                release.candidate()
        (self.root / 'CHANGELOG.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'tracked changes'):
            release.candidate()
        self.git('checkout', '--', 'CHANGELOG.md')
        (self.root / '.changes').mkdir()
        pending = self.root / '.changes/change.md'
        pending.write_text('pending')
        with self.assertRaisesRegex(ValueError, 'pending'):
            release.candidate()
        pending.unlink()
        self.git('commit', '--allow-empty', '-qm', 'different candidate')
        self.git('tag', 'v1.2.3')
        self.git('checkout', '--detach', '-q', self.sha)
        with self.assertRaisesRegex(ValueError, 'different commit'):
            release.candidate()

    def test_registry_absence_match_mismatch_and_yank(self):
        for record, expected in [(None, False), (self.record, True)]:
            with patch.object(release, 'registry', return_value=record):
                self.assertEqual(release.published('1.2.3', self.archive), expected)
        for change in [{'checksum': '0' * 64}, {'yanked': True}]:
            with patch.object(release, 'registry', return_value={**self.record, **change}), self.assertRaises(ValueError):
                release.published('1.2.3', self.archive)

    def test_retry_finishes_github_without_republishing(self):
        calls = []

        def github(path, payload=None):
            calls.append((path, payload))
            if path == '/git/refs':
                self.git('tag', 'v1.2.3', payload['sha'])
            if path == '/releases':
                return {'html_url': 'https://example.invalid/release'}
            return None

        with patch.object(release, 'registry', return_value=self.record), patch.object(release, 'github', side_effect=self.forge(github)), patch('sys.argv', ['release.py', 'publish']):
            # Real command execution: accidentally invoking Cargo would fail here.
            release.main()
        self.assertEqual([path for path, _ in calls], ['/git/refs', '/releases/tags/v1.2.3', '/releases'])
        self.assertEqual(self.git('rev-parse', 'v1.2.3'), self.sha)

    def test_publish_failure_has_no_forge_side_effect(self):
        original = release.run

        def run(*command):
            if command[:2] == ('cargo', 'publish'):
                raise subprocess.CalledProcessError(1, command)
            return original(*command)

        with patch.object(release, 'registry', return_value=None), patch.object(release, 'run', side_effect=run), patch.object(release, 'github', side_effect=self.forge()) as github, patch('sys.argv', ['release.py', 'publish']):
            with self.assertRaises(subprocess.CalledProcessError):
                release.main()
            github.assert_called_once_with(self.status_path)

    def test_successful_upload_precedes_forge_effects(self):
        events = []
        original = release.run

        def run(*command):
            if command[:2] == ('cargo', 'publish'):
                events.append('upload')
                return ''
            return original(*command)

        def github(path, payload=None):
            events.append(path)
            if path == '/git/refs':
                self.git('tag', 'v1.2.3', payload['sha'])
            if path == '/releases':
                return {'html_url': 'https://example.invalid/release'}
            return None

        with patch.object(release, 'registry', side_effect=[None, self.record]), patch.object(release, 'run', side_effect=run), patch.object(release, 'github', side_effect=self.forge(github)), patch('sys.argv', ['release.py', 'publish']):
            release.main()
        self.assertEqual(events, ['upload', '/git/refs', '/releases/tags/v1.2.3', '/releases'])

    def test_completed_release_is_noop_and_conflicting_notes_fail(self):
        self.git('tag', 'v1.2.3')
        existing = {'draft': False, 'prerelease': False, 'body': '- Reviewed change.',
                    'html_url': 'https://example.invalid/release'}
        with patch.object(release, 'registry', return_value=self.record), patch.object(release, 'github', side_effect=self.forge(lambda *_: existing)) as github, patch('sys.argv', ['release.py', 'publish']):
            release.main()
            self.assertEqual([call.args for call in github.call_args_list], [(self.status_path,), ('/releases/tags/v1.2.3',)])
            existing['body'] = 'different notes'
            with self.assertRaisesRegex(ValueError, 'differs'):
                release.main()

    def test_mismatched_upload_never_creates_release(self):
        with patch.object(release, 'registry', return_value={**self.record, 'checksum': '0' * 64}), patch.object(release, 'github', side_effect=self.forge()) as github, patch('sys.argv', ['release.py', 'publish']):
            with self.assertRaises(ValueError):
                release.main()
            github.assert_called_once_with(self.status_path)


if __name__ == '__main__':
    unittest.main()
