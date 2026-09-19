#!/usr/bin/env python3
"""Keep manual profile requests curated, bounded, and confined to one native runner."""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_ci

ROOT = Path(__file__).resolve().parents[2]


def workflow_input(name: str) -> tuple[str, list[str]]:
  text = (ROOT / '.github/workflows/profile.yml').read_text()
  match = re.search(rf'^      {re.escape(name)}:\n(?P<body>(?: {{8,}}.*\n)+)', text, re.MULTILINE)
  if match is None:
    raise AssertionError(f'missing workflow input: {name}')
  body = match.group('body')
  default = re.search(r'^        default: ["\']?(.*?)["\']?$', body, re.MULTILINE)
  if default is None:
    raise AssertionError(f'missing workflow default: {name}')
  options = re.findall(r'^          - ["\']?(.*?)["\']?$', body, re.MULTILINE)
  return default.group(1), options


class ManualProfile(unittest.TestCase):
  def environment(self, **overrides):
    return {
      'INPUT_ARCHITECTURE': 'x86_64-linux-intel',
      'INPUT_PRIMITIVE': 'aead/aes',
    } | overrides

  def test_request_resolves_one_exact_catalog_preset_and_runner(self):
    selection = profile_ci.request(self.environment(), '42')
    self.assertEqual(selection['architecture'], 'x86_64-linux-intel')
    self.assertEqual(selection['target'], 'x86_64-unknown-linux-gnu')
    self.assertEqual(selection['runner'], 'runs-on=42/runner=measure-x86_64-linux-intel/env=production')
    self.assertEqual(selection['primitive'], 'aead/aes')
    self.assertEqual(selection['benchmark'], 'aead')
    self.assertEqual(selection['case'], 'aes-128-gcm/copy-and-encrypt/rscrypto/4096')
    self.assertEqual(selection['seconds'], 5)
    self.assertEqual(selection['binary'], 'aead')
    self.assertNotIn('diag', selection['features'])
    self.assertEqual(selection['prepare_timeout'], 20)
    self.assertEqual(selection['capture_timeout'], 20)

  def test_request_resolves_standard_linux_architectures(self):
    expected = {
      'aarch64-linux': (
        'aarch64-unknown-linux-gnu',
        'runs-on=42/runner=measure-aarch64-linux/env=production',
        'aarch64-linux',
      ),
      'x86_64-linux-amd': (
        'x86_64-unknown-linux-gnu',
        'runs-on=42/runner=measure-x86_64-linux-amd/env=production',
        'x86_64-linux',
      ),
      'x86_64-linux-intel': (
        'x86_64-unknown-linux-gnu',
        'runs-on=42/runner=measure-x86_64-linux-intel/env=production',
        'x86_64-linux',
      ),
    }
    for architecture, (target, runner, tooling_platform) in expected.items():
      with self.subTest(architecture=architecture):
        selection = profile_ci.request(self.environment(INPUT_ARCHITECTURE=architecture), '42')
        self.assertEqual(selection['target'], target)
        self.assertEqual(selection['runner'], runner)
        self.assertEqual(selection['tooling_platform'], tooling_platform)

  def test_request_rejects_broad_or_unbounded_input(self):
    invalid = (
      {'INPUT_ARCHITECTURE': 'all'},
      {'INPUT_ARCHITECTURE': 'x86_64-linux'},
      {'INPUT_ARCHITECTURE': 'aarch64-win'},
      {'INPUT_PRIMITIVE': ''},
      {'INPUT_PRIMITIVE': 'aead'},
      {'INPUT_PRIMITIVE': 'aead/$(touch never)'},
    )
    for override in invalid:
      with self.subTest(override=override), self.assertRaises(ValueError):
        profile_ci.request(self.environment(**override), '42')

  def test_dispatch_uses_the_preset_case_as_one_literal_argument(self):
    selection = profile_ci.request(self.environment(INPUT_PRIMITIVE='hashes/blake3'), '42')
    for operation, flag in (('prepare', '--prepare-archive'), ('capture', '--run-archive')):
      with self.subTest(operation=operation):
        command = profile_ci.command(selection, operation, selection['target'], 'archive with spaces.tar.gz')
        self.assertEqual(command[:3], ['just', 'profile', 'blake3'])
        self.assertIn('blake3/rscrypto/4096', command)
        self.assertIn('--diag', command)
        self.assertIn(flag, command)
        self.assertEqual(command[-1], 'archive with spaces.tar.gz')

  def test_capture_runs_one_case_and_publishes_its_report(self):
    selection = profile_ci.request(self.environment(INPUT_PRIMITIVE='hashes/sha256'), '42')
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      report = root / 'target/profiles/sha2/perf-report.txt'
      report.parent.mkdir(parents=True)
      report.write_text('99.00% sha2 rscrypto::sha256\n')
      summary = root / 'summary.md'
      env = self.environment(INPUT_PRIMITIVE='hashes/sha256') | {
        'GITHUB_RUN_ID': '42',
        'GITHUB_STEP_SUMMARY': str(summary),
      }
      with contextlib.chdir(root), patch.dict(os.environ, env, clear=True), \
           patch.object(sys, 'argv', ['profile_ci.py', 'capture', selection['target'], 'archive.tar.gz']), \
           patch.object(profile_ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
           contextlib.redirect_stdout(io.StringIO()):
        self.assertEqual(profile_ci.main(), 0)
      run.assert_called_once_with(
        profile_ci.command(selection, 'capture', selection['target'], 'archive.tar.gz'), check=False)
      text = summary.read_text()
      self.assertIn('x86_64-linux-intel` / `hashes/sha256', text)
      self.assertIn('sha256/rscrypto/4096', text)
      self.assertIn('rscrypto::sha256', text)

  def test_workflow_choices_match_the_validated_policy(self):
    workflow = (ROOT / '.github/workflows/profile.yml').read_text()
    architecture_default, architectures = workflow_input('architecture')
    primitive_default, primitives = workflow_input('primitive')
    self.assertEqual(architecture_default, 'x86_64-linux-intel')
    self.assertEqual(set(architectures), profile_ci.ARCHITECTURES)
    self.assertEqual(primitive_default, 'aead/aes')
    self.assertEqual(primitives, list(profile_ci.load_catalog()['profile_presets']))
    self.assertNotIn('      seconds:', workflow)
    self.assertIn('RSCRYPTO_REQUIRE_PERF: "1"', workflow)
    self.assertNotIn('RSCRYPTO_PERF_SUDO', workflow)

  def test_plan_emits_only_the_selected_runner(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'output'
      env = self.environment() | {'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
      with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['profile_ci.py', 'plan']), \
           contextlib.redirect_stdout(io.StringIO()):
        self.assertEqual(profile_ci.main(), 0)
      values = dict(line.split('=', 1) for line in output.read_text().splitlines())
      self.assertEqual(values['architecture'], 'x86_64-linux-intel')
      self.assertEqual(values['target'], 'x86_64-unknown-linux-gnu')
      self.assertEqual(values['runner'], 'runs-on=42/runner=measure-x86_64-linux-intel/env=production')
      self.assertEqual(values['primitive'], 'aead/aes')
      self.assertEqual(values['prepare_timeout'], '20')
      self.assertEqual(values['capture_timeout'], '20')

  def test_isolated_entry_point(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'output'
      env = os.environ | self.environment() | {'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
      result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bench/profile_ci.py'), 'plan'],
                              env=env, text=True, capture_output=True, check=False)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertEqual(json.loads(result.stdout.removeprefix('Profile request: '))['primitive'], 'aead/aes')


if __name__ == '__main__':
  unittest.main()
