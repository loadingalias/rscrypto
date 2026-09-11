#!/usr/bin/env python3
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


class ManualBench(unittest.TestCase):
    def test_one_many_all_platforms(self):
        self.assertEqual([r['platform'] for r in ci.platforms('riscv64-linux', '42')['include']], ['riscv64-linux'])
        rows = ci.platforms('s390x-linux, x86_64-win s390x-linux', '42')['include']
        self.assertEqual([r['platform'] for r in rows], ['s390x-linux', 'x86_64-win'])
        self.assertIn('windows25-full-x64', rows[1]['runner'])
        self.assertEqual(len(ci.platforms('all', '42')['include']), 6)
        for bad in ('', 'all,x86_64-linux', 'apple-arm64', 'x86_64-linux,typo'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ci.platforms(bad, '42')

    def test_catalog_selection_preserves_algorithm_scope(self):
        args = ci.arguments({'INPUT_SELECTION': 'sha256, blake3', 'INPUT_FILTER': '^.*/rscrypto/64$'})
        rows = ci.runner.requests(ci.runner.parse(['bench', *args]), ci.load_catalog())
        self.assertEqual({r['binary'] for r in rows}, {'sha2', 'blake3'})
        self.assertTrue(all(r['scope'] for r in rows))
        self.assertEqual({r['pattern'] for r in rows}, {'^.*/rscrypto/64$'})
        for selection in ('hashes', 'auth', 'all', 'bench=sha2,auth'):
            with self.subTest(selection=selection):
                self.assertTrue(ci.arguments({'INPUT_SELECTION': selection}))

    def test_rejects_unknown_selection_options_and_bad_sampling(self):
        for selection in ('', 'sha999', '--output-dir=/tmp', 'bench=sha2,', 'sha256;echo pwned'):
            with self.subTest(selection=selection), self.assertRaises((ValueError, SystemExit)):
                ci.arguments({'INPUT_SELECTION': selection})
        with self.assertRaises(ValueError):
            ci.arguments({'INPUT_SELECTION': 'sha256', 'INPUT_SAMPLE_SIZE': '1'})

    def test_run_forwards_regex_literally_and_propagates_failure(self):
        env = {'INPUT_SELECTION': 'sha256', 'INPUT_FILTER': "$(touch /tmp/not-executed);'|.*"}
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['ci.py', 'run']), \
             patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as execute:
            self.assertEqual(ci.main(), 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[:3], ['just', 'bench', 'sha256'])
            self.assertIn('filter=' + env['INPUT_FILTER'], command)
            self.assertNotIn('shell', execute.call_args.kwargs)

    def test_isolated_python_entry_points(self):
        root = Path(__file__).resolve().parents[2]
        for command in ([sys.executable, '-I', str(root / 'scripts/bench/runner.py'), '--help'],
                        [sys.executable, '-I', str(root / 'scripts/bench/bounded.py'), sys.executable, '-c', 'pass']):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            env = {**os.environ, 'INPUT_SELECTION': 'sha256', 'INPUT_ARCHITECTURES': 'all',
                   'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
            result = subprocess.run([sys.executable, '-I', str(root / 'scripts/bench/ci.py'), 'plan'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(output.read_text().removeprefix('matrix='))['include']), 6)


if __name__ == '__main__':
    unittest.main()
