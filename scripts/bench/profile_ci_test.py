#!/usr/bin/env python3
"""Keep manual profile requests exact, bounded, and confined to one native runner."""

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
      'INPUT_ARCHITECTURE': 's390x-linux',
      'INPUT_WORKLOAD': 'aead/aes',
      'INPUT_SECONDS': '5',
    } | overrides

  def test_request_resolves_one_exact_catalog_preset_and_runner(self):
    selection = profile_ci.request(self.environment(), '42')
    self.assertEqual(selection['architecture'], 's390x-linux')
    self.assertEqual(selection['target'], 's390x-unknown-linux-gnu')
    self.assertEqual(selection['runner'], 'ubuntu-24.04-s390x')
    self.assertEqual(selection['workload'], 'aead/aes')
    self.assertEqual(selection['benchmark'], 'aead')
    self.assertEqual(selection['case'], 'aes-128-gcm/copy-and-encrypt/rscrypto/4096')
    self.assertEqual(selection['binary'], 'aead')
    self.assertNotIn('diag', selection['features'])
    self.assertEqual(selection['prepare_timeout'], 30)
    self.assertEqual(selection['capture_timeout'], 20)

  def test_request_rejects_broad_or_unbounded_input(self):
    invalid = (
      {'INPUT_ARCHITECTURE': 'all'},
      {'INPUT_ARCHITECTURE': 'x86_64-linux'},
      {'INPUT_WORKLOAD': ''},
      {'INPUT_WORKLOAD': 'aead'},
      {'INPUT_WORKLOAD': 'aead/$(touch never)'},
      {'INPUT_SECONDS': '0'},
      {'INPUT_SECONDS': '16'},
      {'INPUT_SECONDS': '1.5'},
    )
    for override in invalid:
      with self.subTest(override=override), self.assertRaises(ValueError):
        profile_ci.request(self.environment(**override), '42')

  def test_dispatch_uses_the_preset_case_as_one_literal_argument(self):
    selection = profile_ci.request(self.environment(INPUT_WORKLOAD='hashes/blake3'), '42')
    for operation, flag in (('prepare', '--prepare-archive'), ('capture', '--run-archive')):
      with self.subTest(operation=operation):
        command = profile_ci.command(selection, operation, selection['target'], 'archive with spaces.tar.gz')
        self.assertEqual(command[:3], ['just', 'profile', 'blake3'])
        self.assertIn('blake3/rscrypto/4096', command)
        self.assertIn('--diag', command)
        self.assertIn(flag, command)
        self.assertEqual(command[-1], 'archive with spaces.tar.gz')

  def test_workflow_choices_match_the_validated_policy(self):
    architecture_default, architectures = workflow_input('architecture')
    workload_default, workloads = workflow_input('workload')
    seconds_default, seconds = workflow_input('seconds')
    self.assertEqual(architecture_default, 's390x-linux')
    self.assertEqual(set(architectures), profile_ci.ARCHITECTURES)
    self.assertEqual(workload_default, 'aead/aes')
    self.assertEqual(workloads, list(profile_ci.load_catalog()['profile_presets']))
    self.assertEqual(seconds_default, str(profile_ci.settings.PROFILE_CAPTURE_DEFAULT_SECONDS))
    self.assertEqual(seconds, ['3', '5', '10', '15'])
    self.assertEqual(int(seconds[-1]), profile_ci.settings.PROFILE_CAPTURE_MAX_SECONDS)

  def test_plan_emits_only_the_selected_runner(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'output'
      env = self.environment() | {'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
      with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['profile_ci.py', 'plan']), \
           contextlib.redirect_stdout(io.StringIO()):
        self.assertEqual(profile_ci.main(), 0)
      values = dict(line.split('=', 1) for line in output.read_text().splitlines())
      self.assertEqual(values['architecture'], 's390x-linux')
      self.assertEqual(values['target'], 's390x-unknown-linux-gnu')
      self.assertEqual(values['runner'], 'ubuntu-24.04-s390x')
      self.assertEqual(values['workload'], 'aead/aes')
      self.assertEqual(values['prepare_timeout'], '30')
      self.assertEqual(values['capture_timeout'], '20')

  def test_isolated_entry_point(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'output'
      env = os.environ | self.environment() | {'GITHUB_RUN_ID': '42', 'GITHUB_OUTPUT': str(output)}
      result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bench/profile_ci.py'), 'plan'],
                              env=env, text=True, capture_output=True, check=False)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertEqual(json.loads(result.stdout.removeprefix('Profile request: '))['workload'], 'aead/aes')


if __name__ == '__main__':
  unittest.main()
