#!/usr/bin/env python3
"""Exercise profiling scope and shared builds without sampling the host."""

import contextlib
import hashlib
import io
import json
import profile as profile_runner
import unittest
from pathlib import Path
from unittest.mock import patch

import run_test
import runner


class ProfileTests(unittest.TestCase):
  def setUp(self):
    self.fixture = run_test.RunnerTests()
    self.fixture.setUp()
    self.addCleanup(self.fixture.doCleanups)
    self.root = self.fixture.root

  def profile(self, *args, **env):
    return self.fixture.invoke('profile', *args, **env)

  def test_default_capture_interval_is_five_seconds(self):
    args = runner.parse(['profile', 'sha2', 'sha256/rscrypto/4096'])
    self.assertEqual(args.seconds, 5)

  def metadata(self):
    return json.loads(next((self.root / 'target/profiles').glob('*/metadata.json')).read_text())

  def test_invalid_scope_never_captures(self):
    for args in (('sha2',), ('sha2', '', '1'), ('sha2', 'sha256', '1'),
                 ('sha2', 'sha256/rscrypto/64', '0'), ('sha2', 'sha256/rscrypto/64', 'nan')):
      self.assertNotEqual(self.profile(*args).returncode, 0)
      self.assertFalse((self.root / 'target/profiles').exists())
    for env in ({'FAIL_BUILD': '1'}, {'FAIL_LIST': '1'}, {'DUPLICATE_CASES': '1'}):
      self.assertNotEqual(self.profile('sha2', 'sha256/rscrypto/64', '1', **env).returncode, 0)
      self.assertFalse((self.root / 'target/profiles').exists())

  def test_preparation_surfaces_failed_command_diagnostics(self):
    for args in (('sha2', '--list'), ('sha2', 'sha256/rscrypto/64', '1')):
      for env, code, diagnostic in (
        ({'FAIL_BUILD': '1'}, 8, 'fixture compiler detail: rejected benchmark source'),
        ({'FAIL_LIST': '1'}, 6, 'fixture discovery detail: cannot enumerate cases'),
      ):
        with self.subTest(args=args, env=env):
          result = self.profile(*args, **env)
          self.assertEqual(result.returncode, code)
          self.assertIn(diagnostic, result.stderr)
    self.assertFalse((self.root / 'target/profiles').exists())

  def test_discovery_and_diagnostic_build_parity(self):
    for env in ({}, {'RUSTFLAGS': ''}, {'RUSTFLAGS': '-C target-cpu=generic'},
                {'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-cpu=generic'}, {'DEV_MACHINE_TARGET': 'fixture'}):
      self.fixture.ok(self.fixture.bench('blake3', '--diag', '--list', **env))
      self.fixture.ok(self.profile('blake3', '--diag', '--list', **env))
      builds = [json.loads(row) for row in (self.root / 'builds.jsonl').read_text().splitlines()]
      self.assertEqual(builds[-2], builds[-1])
      self.assertIn('diag', builds[-1][builds[-1].index('--features') + 1].split(','))
    self.assertFalse((self.root / 'target/profiles').exists())
    self.assertFalse(self.fixture.runs())

  def test_exact_capture_and_shared_evidence(self):
    controls = {'RAYON_NUM_THREADS': '3', 'RSCRYPTO_FORCE_AVX512': '1'}
    self.fixture.ok(self.fixture.bench('sha256', **controls))
    self.fixture.ok(self.profile('sha2', 'sha256/rscrypto/64', '1', **controls))
    metadata = self.metadata()
    self.assertEqual(metadata['status'], 'complete')
    self.assertEqual(metadata['compatibility'], self.fixture.plan()[0]['compatibility'])
    self.assertEqual(metadata['artifact']['sha256'], hashlib.sha256(Path(metadata['artifact']['path']).read_bytes()).hexdigest())
    self.assertEqual(metadata['command'], [metadata['artifact']['path'], '--bench', '--profile-time', '1.0', '--noplot'])
    root = next((self.root / 'target/profiles').iterdir())
    self.assertEqual(json.loads((root / 'cases.json').read_text()), ['sha256/rscrypto/64'])
    for name in ('source.json', 'source-state.json', 'profile.json.gz', 'output.txt'):
      self.assertTrue((root / name).is_file())
    self.assertEqual(len(self.fixture.calls()), 2)

  def test_capture_failure_is_recorded(self):
    result = self.profile('sha2', 'sha256/rscrypto/64', '1', FAIL_CAPTURE='1')
    self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
    self.assertEqual(self.metadata()['status'], 'failed')
    self.assertEqual(self.metadata()['exit_code'], 7)

  def perf(self, **environment):
    root = self.root / 'target/perf-unit'
    root.mkdir(parents=True, exist_ok=True)
    case = 'criterion-fixture/rscrypto/64'
    cases = root / 'cases.json'
    cases.write_text(json.dumps([case]))
    env = self.fixture.env | environment | {
      'CRITERION_HOME': str(root / 'criterion'),
      'RSCRYPTO_BENCH_CASES': str(cases),
    }
    command = [str(self.root / 'bin/criterion-fixture'), '--bench', '--profile-time', '1', '--noplot']
    with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      return root, profile_runner.perf_capture(root, command, env)

  def test_perf_capture_retains_native_report_and_raw_evidence(self):
    root, (status, cause, collector) = self.perf()
    self.assertEqual((status, cause), ('complete', None))
    self.assertEqual(collector['event'], 'cycles:u')
    self.assertEqual(collector['callchain'], 'dwarf')
    self.assertEqual(collector['access'], 'runner')
    self.assertEqual(collector['version'], 'perf fixture')
    for name in ('host.json', 'capabilities.json', 'perf-stat.txt', 'perf.data', 'perf-report.txt',
                 'perf-script.txt', 'perf-buildids.txt', 'output.txt'):
      self.assertTrue((root / name).is_file(), name)

  def test_perf_capture_falls_back_and_keeps_partial_failures(self):
    _, (status, _, collector) = self.perf(FAIL_PERF_DWARF='1')
    self.assertEqual(status, 'complete')
    self.assertEqual(collector['callchain'], 'flat')
    for environment in ({'FAIL_PERF_STAT': '1'}, {'FAIL_PERF_CAPTURE': '1'}, {'FAIL_PERF_REPORT': '1'},
                        {'FAIL_PERF_SCRIPT': '1'}, {'ZERO_PERF_SAMPLES': '1'}):
      with self.subTest(environment=environment):
        root = self.root / 'target/perf-unit'
        for path in root.iterdir():
          if path.is_file():
            path.unlink()
        _, (status, cause, _) = self.perf(**environment)
        self.assertEqual(status, 'partial')
        self.assertTrue(cause)

  def test_missing_or_blocked_perf_is_unavailable(self):
    with patch.object(profile_runner.shutil, 'which', return_value=None), \
         self.assertRaises(profile_runner.ProfileUnavailable):
      self.perf()
    host = json.loads((self.root / 'target/perf-unit/host.json').read_text())
    self.assertEqual(host['perf']['exit_code'], 127)
    self.assertEqual(host['uname']['exit_code'], 0)
    with self.assertRaisesRegex(profile_runner.ProfileUnavailable,
                                'perf_event_paranoid=4.*CAP_PERFMON'):
      self.perf(FAIL_PERF_PROBE='1')


if __name__ == '__main__':
  unittest.main()
