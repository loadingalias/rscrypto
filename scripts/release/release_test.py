#!/usr/bin/env -S uv run python
"""Exercise the release preflight without forge writes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import call, patch

import yaml

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
        self.fuzz_path = f'/actions/workflows/fuzz.yml/runs?head_sha={self.sha}&event=workflow_dispatch&per_page=1'
        self.fuzz_run = {'id': 42, 'run_attempt': 2, 'head_sha': self.sha, 'event': 'workflow_dispatch',
                         'path': '.github/workflows/fuzz.yml', 'status': 'completed', 'conclusion': 'success',
                         'html_url': 'https://github.com/example/fixture/actions/runs/42'}
        self.fuzz_job = {'name': 'Release qualification', 'status': 'completed', 'conclusion': 'success'}
        self.jobs_path = '/actions/runs/42/attempts/2/jobs?per_page=100&page=1'

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
        with patch.object(release, 'github', return_value=[self.mac_status]):
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

    def test_preflight_requires_both_qualifications(self):
        with patch.object(release, 'github', side_effect=[
                [self.mac_status], {'workflow_runs': [self.fuzz_run]}, {'jobs': [self.fuzz_job]}]) as github, \
             patch('sys.argv', ['release.py', 'preflight']):
            release.main()
        self.assertEqual(github.call_args_list, [call(self.status_path), call(self.fuzz_path), call(self.jobs_path)])

    def test_fuzz_gate_rejects_missing_mismatched_or_unfinished_runs(self):
        for result in (None, {}, {'workflow_runs': []}):
            with self.subTest(result=result), patch.object(release, 'github', return_value=result) as github, \
                 self.assertRaisesRegex(ValueError, 'Missing manual Fuzz'):
                release.require_fuzz(self.sha)
            github.assert_called_once_with(self.fuzz_path)
        changes = [
            {'head_sha': '0' * 40}, {'event': 'push'}, {'event': 'schedule'},
            {'path': '.github/workflows/ci.yml'}, {'status': 'in_progress', 'conclusion': None},
            {'status': 'queued', 'conclusion': None},
            *({'conclusion': state} for state in ('failure', 'cancelled', 'timed_out', 'skipped')),
        ]
        for changed in changes:
            with self.subTest(changed=changed), patch.object(release, 'github', return_value={
                    'workflow_runs': [{**self.fuzz_run, **changed}]}) as github, self.assertRaisesRegex(ValueError, 'Fuzz'):
                release.require_fuzz(self.sha)
            # No search for an older success or jobs from an earlier attempt.
            github.assert_called_once_with(self.fuzz_path)

    def test_fuzz_gate_requires_full_workload_success_in_current_attempt(self):
        results = [None, {'jobs': []}, {'jobs': [{'name': 'fuzz (x86_64-linux)', 'conclusion': 'success'}]},
                   *({'jobs': [{**self.fuzz_job, 'conclusion': state}]} for state in ('skipped', 'failure', 'cancelled')),
                   {'jobs': [{**self.fuzz_job, 'status': 'in_progress', 'conclusion': None}]}]
        for result in results:
            with self.subTest(result=result), patch.object(release, 'github', side_effect=[
                    {'workflow_runs': [self.fuzz_run]}, result]) as github, self.assertRaisesRegex(ValueError, 'Fuzz'):
                release.require_fuzz(self.sha)
            self.assertEqual(github.call_args_list, [call(self.fuzz_path), call(self.jobs_path)])
        # A large matrix must not hide the qualifying job on the next page.
        with patch.object(release, 'github', side_effect=[
                {'workflow_runs': [self.fuzz_run]}, {'jobs': [{'name': 'unrelated'}] * 100},
                {'jobs': [self.fuzz_job]}]) as github:
            release.require_fuzz(self.sha)
        self.assertEqual(github.call_args.args, ('/actions/runs/42/attempts/2/jobs?per_page=100&page=2',))

    def test_release_workflow_requires_full_manual_fuzz_without_starting_it(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.load((root / '.github/workflows/fuzz.yml').read_text(), Loader=yaml.BaseLoader)
        inputs = workflow['on']['workflow_dispatch']['inputs']
        self.assertEqual(inputs['campaign']['default'], 'smoke')
        self.assertEqual(inputs['architectures']['default'], 'all')
        self.assertEqual(inputs['targets'].get('default', ''), '')
        self.assertGreaterEqual(int(inputs['duration']['default']), 120)
        matrix = workflow['jobs']['fuzz']['strategy']['matrix']
        self.assertIn('["x86_64-linux","aarch64-linux"]', matrix['platform'])
        self.assertIn({'platform': 'x86_64-linux', 'task': 'miri'}, matrix['include'])
        gate = workflow['jobs']['release-qualification']
        self.assertEqual(gate['name'], 'Release qualification')
        self.assertEqual(gate['needs'], 'fuzz')
        self.assertEqual(gate['if'], "inputs.architectures == 'all' && inputs.campaign == 'smoke' && inputs.targets == '' && inputs.duration >= 120")
        self.assertNotIn('continue-on-error', workflow['jobs']['fuzz'])
        workflow = yaml.load((root / '.github/workflows/release.yml').read_text(), Loader=yaml.BaseLoader)
        jobs = workflow['jobs']
        self.assertNotIn('fuzz', jobs)
        self.assertEqual(jobs['preflight']['permissions']['actions'], 'read')
        self.assertTrue(any('scripts/release/release.py preflight' in step.get('run', '')
                            for step in jobs['preflight']['steps']))
        self.assertEqual(jobs['ci']['needs'], 'preflight')
        self.assertEqual(jobs['ct']['needs'], 'preflight')
        self.assertEqual(set(jobs['release']['needs']), {'ci', 'ct'})

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
