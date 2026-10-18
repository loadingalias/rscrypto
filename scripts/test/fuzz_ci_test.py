#!/usr/bin/env -S uv run python
"""Exercise CI fuzz selection, corpus retention, and failures at external tool boundaries."""

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

spec = importlib.util.spec_from_file_location('fuzz_ci', Path(__file__).with_name('fuzz_ci.py'))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class FuzzCI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fuzz CI ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / 'target/fuzz-corpus'
        self.logs = self.root / 'target/fuzz-logs'
        self.logs.mkdir(parents=True)
        self.now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        self.source = {'commit': 'a' * 40, 'sha256': 'b' * 64}
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ('BASH_ENV', 'ENV', 'CARGO_ENCODED_RUSTFLAGS')
                    and not key.startswith('BASH_FUNC_')}
        source = patch.object(ci.bundle, 'source_identity', return_value=self.source)
        source.start()
        self.addCleanup(source.stop)
        self.package('fuzz', ['parser', 'decrypt'])
        self.package('fuzz-packages/parser', ['parser'])
        (self.root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "fixture-nightly"\n')
        (self.root / 'fuzz/committed-seeds.txt').write_text(
            'fuzz/corpus/parser/seed\nfuzz/corpus/decrypt/seed\nfuzz-packages/parser/corpus/parser/seed\n')

    def package(self, name, targets):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'Cargo.toml').write_text('[package.metadata]\ncargo-fuzz = true\n' + ''.join(
            f'[[bin]]\nname = "{target}"\n' for target in targets))
        for target in targets:
            corpus = directory / 'corpus' / target
            corpus.mkdir(parents=True, exist_ok=True)
            (corpus / 'seed').write_bytes(b'abc')

    def test_manual_selection_covers_every_package_and_rejects_automatic_events(self):
        env = {'GITHUB_EVENT_NAME': 'workflow_dispatch'}
        self.assertEqual(ci.plan(self.root, env, self.now)['targets'], ['decrypt', 'parser'])
        selected = ci.plan(self.root, {**env, 'FUZZ_TARGETS': 'parser'}, self.now)
        self.assertEqual(selected['pairs'], [
            {'package': 'fuzz', 'target': 'parser'},
            {'package': 'fuzz-packages/parser', 'target': 'parser'},
        ])
        for event in ('push', 'pull_request', 'schedule', 'workflow_call'):
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, 'manual dispatch'):
                ci.plan(self.root, {'GITHUB_EVENT_NAME': event}, self.now)

    def test_daily_rotation_covers_every_target_with_bounded_work(self):
        self.package('fuzz', [f'target_{i:02}' for i in range(67)])
        reached = set()
        for day in range(68):
            plan = ci.plan(self.root, {'FUZZ_CAMPAIGN': 'daily'}, self.now + timedelta(days=day))
            reached.update(plan['targets'])
            self.assertEqual(len(plan['targets']), 8)
            self.assertLessEqual(plan['planned_live_seconds'], 1800)
            self.assertGreaterEqual(plan['duration_seconds'], 900)
        self.assertEqual(reached, {row['target'] for row in ci.inventory(self.root)})
        manual = ci.plan(self.root, {'GITHUB_EVENT_NAME': 'workflow_dispatch', 'FUZZ_CAMPAIGN': 'daily'}, self.now)
        self.assertEqual(manual['campaign'], 'daily')
        with self.assertRaisesRegex(ValueError, 'campaign'):
            ci.plan(self.root, {'FUZZ_CAMPAIGN': 'unknown'}, self.now)
        for requested in ('unknown', 'parser,parser', 'parser,'):
            with self.assertRaises(ValueError):
                ci.plan(self.root, {'FUZZ_TARGETS': requested}, self.now)
        with self.assertRaisesRegex(ValueError, 'budget'):
            ci.plan(self.root, {'FUZZ_DURATION': '1801'}, self.now)

    def test_cache_is_content_addressed_and_keeps_committed_seeds(self):
        rows = ci.inventory(self.root)
        discovery = self.root / 'fuzz/corpus/parser/discovery'
        discovery.write_bytes(b'new discovery')
        ci.snapshot(self.root, self.cache, rows)
        discovery.unlink()
        ci.restore(self.root, self.cache, rows)
        self.assertEqual((self.root / 'fuzz/corpus/parser/seed').read_bytes(), b'abc')
        self.assertEqual(len(list((self.root / 'fuzz/corpus/parser').iterdir())), 2)
        self.assertEqual((self.root / 'fuzz/corpus/parser' / ('ci-' + hashlib.sha256(b'new discovery').hexdigest())).read_bytes(),
                         b'new discovery')
        cached = self.cache / 'fuzz/corpus/parser'
        (cached / 'bad-hash').write_bytes(b'bad input')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            ci.restore(self.root, self.cache, rows)
        (cached / 'bad-hash').unlink()
        (cached / 'link').symlink_to(discovery.parent / 'seed')
        with self.assertRaisesRegex(ValueError, 'regular'):
            ci.restore(self.root, self.cache, rows)
        (cached / 'link').unlink()
        shutil.rmtree(cached)
        cached.symlink_to(self.root / 'fuzz/corpus/parser', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            ci.restore(self.root, self.cache, rows)

    def test_generated_corpus_inputs_stay_outside_source_identity(self):
        # Source identity hashes untracked, unignored files, so restored and discovered inputs must be ignored.
        directories = [ci.corpus_path(row).as_posix() for row in ci.inventory(ci.ROOT)]
        generated = {f'{directory}/{name}' for directory in directories
                     for name in ('ci-' + hashlib.sha256(b'restored').hexdigest(),
                                  hashlib.sha1(b'discovered').hexdigest())}
        promoted = {f'{directory}/seed-regression' for directory in directories}
        result = subprocess.run(['git', '-c', 'core.excludesFile=/dev/null', 'check-ignore', '--stdin', '-z'],
                                cwd=ci.ROOT, input='\0'.join(sorted(generated | promoted)).encode(),
                                capture_output=True)
        self.assertIn(result.returncode, (0, 1), result.stderr.decode())
        ignored = set(result.stdout.decode().split('\0')) - {''}
        self.assertEqual(sorted(generated - ignored), [])
        self.assertEqual(sorted(promoted & ignored), [])

    def test_failed_snapshot_preserves_previous_checkpoint(self):
        rows = ci.inventory(self.root)
        ci.snapshot(self.root, self.cache, rows)
        old = ci.inputs(self.cache / 'fuzz/corpus/parser')
        (self.root / 'fuzz/corpus/parser/seed').unlink()
        with self.assertRaisesRegex(ValueError, 'empty'):
            ci.snapshot(self.root, self.cache, rows)
        self.assertEqual(ci.inputs(self.cache / 'fuzz/corpus/parser'), old)

    def test_branch_report_distinguishes_unobserved_outcomes_from_feasibility(self):
        data = {'data': [{'files': [
            {'filename': str(self.root / 'src/auth/rsa.rs'), 'branches': [
                [12, 3, 12, 18, 7, 0, 0, 0, 4], [20, 3, 20, 18, 1, 2, 0, 0, 4],
                [30, 3, 30, 18, 0, 0, 0, 0, 4]]},
            {'filename': '/external/dependency.rs', 'branches': [[1, 1, 1, 2, 0, 0]]},
        ]}]}
        result = ci.branch_gaps(data, self.root)
        self.assertEqual(result['source_files'], 1)
        self.assertEqual(result['instrumented_conditions'], 3)
        self.assertEqual([(row['line'], row['true'], row['false']) for row in result['uncovered_outcomes']],
                         [(12, 7, 0), (30, 0, 0)])
        with self.assertRaisesRegex(ValueError, 'no production'):
            ci.branch_gaps({'data': [{'files': []}]}, self.root)

    def test_merge_or_coverage_failure_cannot_publish_a_corpus(self):
        tools = self.root / 'bin'
        tools.mkdir()
        llvm = self.root / 'sysroot/lib/rustlib/fixture-host/bin/llvm-cov'
        llvm.parent.mkdir(parents=True)
        llvm.write_text('#!/bin/sh\nexit 0\n')
        llvm.chmod(0o755)
        rustc = tools / 'rustc'
        rustc.write_text('#!' + sys.executable + '\nimport sys\nprint(' +
                         repr(str(self.root / 'sysroot')) + ' if "--print" in sys.argv else "host: fixture-host")\n')
        rustc.chmod(0o755)
        cargo = tools / 'cargo'
        cargo.write_text('#!' + sys.executable + '''
import os, pathlib, sys
if sys.argv[1:3] == ['fuzz', 'build']:
    directory = pathlib.Path(sys.argv[sys.argv.index('--target-dir') + 1]) / 'fixture-host/release'
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / sys.argv[-1]
    binary.write_text('#!' + sys.executable + '\\nimport os, pathlib, shutil, sys\\n'
        + 'sys.exit(17) if os.environ.get("FAIL_MERGE") else None\\n'
        + 'shutil.copyfile(next(pathlib.Path(sys.argv[-1]).iterdir()), pathlib.Path(sys.argv[-2]) / "merged")\\n')
    binary.chmod(0o755)
elif sys.argv[1:3] == ['fuzz', 'coverage']:
    sys.exit(18)
''')
        cargo.chmod(0o755)
        selected = {'pairs': [{'package': 'fuzz', 'target': 'parser'}], 'source': self.source}
        ci.snapshot(self.root, self.cache, ci.inventory(self.root))
        old = ci.inputs(self.cache / 'fuzz/corpus/parser')
        env = {**self.env, 'PATH': f'{tools}:{os.environ["PATH"]}'}
        for fail_merge, code in [('1', 17), ('', 18)]:
            with self.subTest(fail_merge=fail_merge), self.assertRaises(subprocess.CalledProcessError) as error:
                ci.retain(self.root, self.cache, self.logs, selected, {**env, 'FAIL_MERGE': fail_merge})
            self.assertEqual(error.exception.returncode, code)
            self.assertFalse((self.logs / 'campaign.json').exists())
            self.assertEqual(ci.inputs(self.cache / 'fuzz/corpus/parser'), old)

    def test_only_explicit_manual_workflows_build_and_replay_before_fuzzing(self):
        workflow = yaml.load((ci.ROOT / '.github/workflows/fuzz.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow['on']), {'workflow_dispatch'})
        for path in (ci.ROOT / '.github/workflows').glob('*.yml'):
            caller = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            self.assertFalse(any(job.get('uses') == './.github/workflows/fuzz.yml' for job in caller['jobs'].values()), path)
        self.assertEqual(workflow['on']['workflow_dispatch']['inputs']['campaign']['options'], ['smoke', 'daily'])
        job = workflow['jobs']['fuzz']
        self.assertEqual(job['strategy']['fail-fast'], 'true')
        self.assertIn("inputs.architectures == 'all'", job['strategy']['matrix']['platform'])
        self.assertNotIn('github.event_name', job['strategy']['matrix']['platform'])
        steps = job['steps']
        build, = [step for step in steps if 'just test-fuzz --build --all' in step.get('run', '')]
        self.assertEqual(build['if'], "matrix.task == 'fuzz'")
        self.assertIn('RSCRYPTO_FUZZ_CORPUS=local just test-fuzz-asan --all', build['run'])
        live, = [step for step in steps if step.get('name') == 'Fuzz selected targets']
        self.assertEqual(live['env']['RSCRYPTO_FUZZ_TARGET_CONCURRENCY'], '${{ steps.plan.outputs.concurrency }}')
        self.assertEqual(live['env']['RSCRYPTO_FUZZ_BUDGET_SECS'], '${{ steps.plan.outputs.budget }}')
        retain, = [step for step in steps if 'just test-fuzz-ci retain' in step.get('run', '')]
        self.assertIn("steps.plan.outputs.retain == 'true'", retain['if'])
        save, = [step for step in steps if step.get('uses', '').startswith('actions/cache/save@')]
        self.assertIn("github.ref == 'refs/heads/main'", save['if'])
        self.assertNotIn('always()', save['if'])
        upload, = [step for step in steps if step.get('uses', '').startswith('actions/upload-artifact@')]
        self.assertIn('always()', upload['if'])
        self.assertIn('fuzz/artifacts/', upload['with']['path'])
        self.assertTrue(all('continue-on-error' not in step for step in steps))
        # Execute the actual lane shell with Just as the external boundary.
        binary = self.root / 'just'
        binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
                          'case "$*" in *--build*) exit "${BUILD_EXIT:-0}";; esac\n')
        binary.chmod(0o755)
        for status in ('0', '19'):
            commands = self.root / 'commands'
            commands.write_text('')
            env = {**self.env, 'PATH': f'{self.root}:{os.environ["PATH"]}',
                   'COMMAND_LOG': str(commands), 'BUILD_EXIT': status}
            result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c',
                                     'source() { :; }\n' + build['run']],
                                    cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, int(status), result.stderr)
            self.assertEqual('test-fuzz-asan --all' in commands.read_text(), status == '0')


if __name__ == '__main__':
    unittest.main()
