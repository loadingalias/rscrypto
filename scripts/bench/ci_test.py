#!/usr/bin/env -S uv run python
"""Manual selection must neither launch extra machines nor broaden measured work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci

ROOT = Path(__file__).resolve().parents[2]


def request(benchmarks='sha256', **values):
    return {'DISPATCH_INPUTS': json.dumps({'benchmarks': benchmarks, 'x86_64-linux': True, **values})}


def plan(benchmarks='sha256', **values):
    measurements, builds, _ = ci.plan(ci.dispatch_inputs(request(benchmarks, **values)), '42')
    return measurements['include'], builds['include']


class ManualBench(unittest.TestCase):
    def test_one_many_all_platforms(self):
        self.assertEqual([r['platform'] for r in ci.platforms('riscv64-linux', '42')['include']], ['riscv64-linux'])
        rows = ci.platforms('s390x-linux, x86_64-win s390x-linux', '42')['include']
        self.assertEqual([r['platform'] for r in rows], ['s390x-linux', 'x86_64-win', 'x86_64-win'])
        self.assertIn('measure-x86_64-win-intel', rows[1]['runner'])
        self.assertIn('measure-x86_64-win-amd', rows[2]['runner'])
        self.assertNotEqual(rows[1]['name'], rows[2]['name'])
        self.assertEqual(len(ci.platforms('all', '42')['include']), 8)
        for bad in ('', 'all,x86_64-linux', 'apple-arm64', 'x86_64-linux,typo'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ci.platforms(bad, '42')

    def test_checkboxes_select_platforms(self):
        measurements, _ = plan(**{'x86_64-linux': False, 'aarch64-linux': True, 's390x-linux': True})
        self.assertEqual([row['name'] for row in measurements], ['aarch64-linux', 's390x-linux'])
        with self.assertRaisesRegex(ValueError, 'at least one platform'):
            plan(**{'x86_64-linux': False})
        everything = {name: True for name in ci.PLATFORMS}
        self.assertEqual(len(plan(**everything)[0]), 8)

    def test_one_job_per_platform_and_target(self):
        measurements, builds = plan('sha256,sha512,blake3,crc64', **{'riscv64-linux': True})
        jobs = [(row['name'], row['bench']) for row in measurements]
        self.assertEqual(jobs, [(name, bench) for bench in ('sha2', 'blake3', 'crc')
                                for name in ('x86_64-linux-intel', 'x86_64-linux-amd', 'riscv64-linux')])
        self.assertEqual([(row['target'], row['bench']) for row in builds],
                         [('riscv64gc-unknown-linux-gnu', bench) for bench in ('sha2', 'blake3', 'crc')])
        runs = ci.selection('sha256,sha512,blake3,crc64')
        self.assertEqual(runs, {'sha2': ['sha256', 'sha512'], 'blake3': ['blake3'], 'crc': ['crc64-xz', 'crc64-nvme']})
        # A whole target subsumes the algorithms it contains.
        self.assertEqual(ci.selection('auth,mlkem,ed25519'), {'auth': ['auth']})

    def test_all_runs_every_default_target_without_external_peers(self):
        catalog = ci.load_catalog()
        defaults = [name for name, bench in catalog['benches'].items()
                    if bench['kind'] == 'criterion' and bench['required']]
        self.assertEqual(ci.selection('all'), {name: [name] for name in defaults})
        self.assertNotIn('tls_connections', defaults)
        everything = {name: True for name in ci.PLATFORMS}
        measurements, builds = plan('all', **everything)
        self.assertEqual(len(measurements), 8 * len(defaults))
        self.assertEqual(len(builds), 3 * len(defaults))
        self.assertLessEqual(len(measurements), ci.MAX_MATRIX)

    def test_external_targets_fail_before_runners_start(self):
        for selection in ('tls-provider', 'tls_connections', 'sha256,tls-provider'):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'OpenSSL 4.0.3'):
                plan(selection)

    def test_catalog_selection_preserves_algorithm_scope(self):
        inputs = ci.dispatch_inputs(request('sha256, blake3-mixed', filter='^.*/rscrypto/64$'))
        for bench, names in ci.selection(inputs['benchmarks']).items():
            args = ci.arguments(names, ci.options(inputs))
            rows = ci.runner.requests(ci.runner.parse(['bench', *args]), ci.load_catalog())
            self.assertEqual({r['binary'] for r in rows}, {bench})
            self.assertTrue(all(r['scope'] for r in rows))
            self.assertEqual({r['pattern'] for r in rows}, {'^.*/rscrypto/64$'})
        for selection in ('hashes', 'checksums', 'auth', 'aead', 'mldsa', 'password-hashing', 'all'):
            with self.subTest(selection=selection):
                self.assertTrue(plan(selection)[0])

    def test_rejects_unknown_selection_options_and_bad_inputs(self):
        for selection in ('', 'sha999', '--output-dir=/tmp', 'bench=sha2', 'sha256;echo pwned', 'structural'):
            with self.subTest(selection=selection), self.assertRaises((ValueError, SystemExit)):
                plan(selection)
        with self.assertRaises(ValueError):
            plan(sample_size='1')
        for inputs in ({'benchmarks': 'sha256', 'architectures': 'all'}, {'benchmarks': 'sha256', 'x86_64-linux': 'true'},
                       {'benchmarks': 'sha256', 'diagnostic': 'false'}, {'benchmarks': 7}, []):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                ci.dispatch_inputs({'DISPATCH_INPUTS': json.dumps(inputs)})

    def test_run_forwards_regex_literally_and_propagates_failure(self):
        env = request(filter="$(touch /tmp/not-executed);'|.*")
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['ci.py', 'run', 'sha2']), \
             patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as execute:
            self.assertEqual(ci.main(), 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[:3], ['just', 'bench', 'sha256'])
            self.assertIn('filter=' + json.loads(env['DISPATCH_INPUTS'])['filter'], command)
            self.assertNotIn('shell', execute.call_args.kwargs)
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['ci.py', 'run', 'blake3']), \
             self.assertRaisesRegex(ValueError, 'not a target of this selection'):
            ci.main()

    def test_transfer_dispatch_keeps_selection_and_separate_invocations(self):
        env = request('sha256,sha512,crc32c', filter='^sha256/rscrypto/64$', sample_size='24')
        for operation, flag in (('prepare', 'prepare_archive'), ('measure', 'run_archive')):
            with self.subTest(operation=operation), patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, 'argv', ['ci.py', operation, 'sha2', 's390x-unknown-linux-gnu', 'archive with spaces.tar.gz']), \
                 patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 9)) as execute:
                self.assertEqual(ci.main(), 9)
                command = execute.call_args.args[0]
                self.assertEqual(command[:4], ['just', 'bench', 'sha256', 'sha512'])
                self.assertNotIn('crc32c', command)
                self.assertIn(flag + '=archive with spaces.tar.gz', command)
                self.assertIn('sample_size=24', command)
                self.assertIn('target=s390x-unknown-linux-gnu', command)
        for name in ('riscv64-linux', 's390x-linux', 'powerpc64le-linux'):
            row, = ci.platforms(name, '42')['include']
            self.assertIn('target', row)
            self.assertGreaterEqual(row['timeout'], 120)
        self.assertTrue(all('target' not in row for row in ci.platforms('x86_64-linux', '42')['include']))

    def test_isolated_python_entry_points(self):
        for command in ([sys.executable, '-I', str(ROOT / 'scripts/bench/runner.py'), '--help'],
                        [sys.executable, '-I', str(ROOT / 'scripts/bench/bounded.py'), sys.executable, '-c', 'pass']):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            inputs = {'benchmarks': 'sha256,blake3', **{name: True for name in ci.PLATFORMS}}
            env = {**os.environ, 'DISPATCH_INPUTS': json.dumps(inputs), 'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bench/ci.py'), 'plan'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split('=', 1) for line in output.read_text().splitlines())
            self.assertEqual(len(json.loads(values['measurements'])['include']), 16)
            self.assertEqual(values['cross'], 'true')
            self.assertEqual({(row['target'], row['bench']) for row in json.loads(values['builds'])['include']},
                             {(target, bench) for bench in ('sha2', 'blake3') for target in
                              ('riscv64gc-unknown-linux-gnu', 'powerpc64le-unknown-linux-gnu', 's390x-unknown-linux-gnu')})
            self.assertIn('sha2: just bench sha256 output_dir=target/bench diag=false', result.stdout)


if __name__ == '__main__':
    unittest.main()
