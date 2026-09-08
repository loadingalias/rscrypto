#!/usr/bin/env python3
"""Exercise profiling scope and shared builds without sampling the host."""

import hashlib
import json
from pathlib import Path
import unittest

import run_test


class ProfileTests(unittest.TestCase):
  def setUp(self):
    self.fixture = run_test.RunnerTests()
    self.fixture.setUp()
    self.addCleanup(self.fixture.doCleanups)
    self.root = self.fixture.root

  def profile(self, *args, **env):
    return self.fixture.invoke('profile', *args, **env)

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


if __name__ == '__main__':
  unittest.main()
