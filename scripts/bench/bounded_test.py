#!/usr/bin/env -S uv run python
"""Verify timeout enforcement on a real child process tree."""

import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from bounded import main, run


class BudgetTests(unittest.TestCase):
  def test_exit_status_and_limit_validation(self):
    self.assertEqual(run([sys.executable, "-c", "raise SystemExit(7)"], 2), 7)
    self.assertEqual(run([sys.executable, "-c", "pass"], 600), 0)
    for limit in (0, -1, 601):
      with self.assertRaises(ValueError):
        run([sys.executable, "-c", "pass"], limit)
    self.assertEqual(run([sys.executable, "-c", "pass"], 1800, maximum=1800), 0)
    for seconds, maximum in ((1801, 1800), (1801, 1801)):
      with self.assertRaises(ValueError):
        run([sys.executable, "-c", "pass"], seconds, maximum=maximum)

  def test_front_door_extends_only_a_complete_slow_comparison(self):
    selector = 'slhdsa-sha2-128s-sign'
    for args, expected in (([selector], 1800), (['hash-slhdsa-shake-256s-sign'], 1800),
                           (['slhdsa-sha2-128s-verify'], 600), (['slhdsa-sha2-128f-sign'], 600),
                           (['rsa-4096-keygen'], 1800), (['rsa-3072-keygen'], 600), (['rsa_keygen'], 600),
                           (['slhdsa'], 600), (['slhdsa-sha2-128s'], 600),
                           ([selector, 'slhdsa-sha2-128s-verify'], 600),
                           ([selector, '--filter', 'rscrypto'], 600),
                           ([selector, '--quick'], 600),
                           ([selector, '--prepare-archive', 'fixture.tar'], 600),
                           ([selector, '--run-archive', 'fixture.tar'], 1800)):
      with self.subTest(args=args), patch.dict(os.environ, {}, clear=True), patch('bounded.run', return_value=0) as execute:
        self.assertEqual(main(['--bench', *args]), 0)
        self.assertEqual(execute.call_args.args[1], expected)
        self.assertEqual(execute.call_args.args[0][2:], ['bench', *args])
        self.assertEqual(execute.call_args.kwargs, {'maximum': 1800})
    with patch.dict(os.environ, {'BENCH_QUICK': 'true'}, clear=True), patch('bounded.run', return_value=0) as execute:
      self.assertEqual(main(['--bench', selector]), 0)
      self.assertEqual(execute.call_args.args[1], 600)

  def test_native_measurement_builds_each_target_under_its_own_bound(self):
    # A cold build must not spend the measurement budget, so each target first builds and lists alone.
    args = ['hmac-sha256', 'sha2', '--measure-ms', '200']
    with patch.dict(os.environ, {}, clear=True), patch('bounded.run', return_value=0) as execute:
      self.assertEqual(main(['--bench', *args]), 0)
    self.assertEqual([call.args[0][2:] for call in execute.call_args_list],
                     [['bench', *args, '--list', '--list-target', name] for name in ('auth', 'sha2')]
                     + [['bench', *args]])
    # Builds take the longest bound; the measurement keeps its ordinary budget.
    self.assertEqual([call.args[1] for call in execute.call_args_list], [1800, 1800, 600])
    self.assertEqual([call.kwargs for call in execute.call_args_list], [{'maximum': 1800}] * 3)
    # A failed build ends the invocation before measurement.
    with patch.dict(os.environ, {}, clear=True), patch('bounded.run', side_effect=[124]) as execute:
      self.assertEqual(main(['--bench', *args]), 124)
    self.assertEqual(execute.call_count, 1)
    # Prepared binaries need no build: transfer preparation and consumption keep one bounded run.
    for transfer in ('--prepare-archive', '--run-archive'):
      with self.subTest(transfer=transfer), patch.dict(os.environ, {}, clear=True), \
          patch('bounded.run', return_value=0) as execute:
        self.assertEqual(main(['--bench', 'sha2', transfer, 'fixture.tar']), 0)
        self.assertEqual(execute.call_count, 1)

  def test_listing_bounds_each_target_separately(self):
    # Listing builds every selected target, so each target's build and discovery gets the longest bound.
    args = ['sha2', 'crc', 'slhdsa-sha2-128s-sign', '--list']
    with patch.dict(os.environ, {}, clear=True), patch('bounded.run', return_value=0) as execute:
      self.assertEqual(main(['--bench', *args]), 0)
    self.assertEqual([call.args[0][2:] for call in execute.call_args_list],
                     [['bench', *args, '--list-target', name] for name in ('sha2', 'crc', 'slhdsa')])
    self.assertEqual([call.args[1] for call in execute.call_args_list], [1800] * 3)
    self.assertEqual([call.kwargs for call in execute.call_args_list], [{'maximum': 1800}] * 3)
    # The first failed target ends the listing.
    with patch.dict(os.environ, {}, clear=True), patch('bounded.run', side_effect=[0, 124, 0]) as execute:
      self.assertEqual(main(['--bench', *args]), 124)
    self.assertEqual(execute.call_count, 2)

  @unittest.skipUnless(os.name == "posix", "signal forwarding proof requires POSIX")
  def test_sigterm_stops_the_child(self):
    child = "import time; print('ready', flush=True); time.sleep(30)"
    command = [sys.executable, str(Path(__file__).with_name("bounded.py")), sys.executable, "-c", child]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
      self.assertEqual(process.stdout.readline().strip(), "ready")
      process.send_signal(signal.SIGTERM)
      self.assertEqual(process.wait(timeout=2), 143)
      self.assertEqual(process.stdout.read(), "")

  @unittest.skipUnless(os.name == "posix", "process-group termination proof requires POSIX")
  def test_timeout_kills_grandchild_that_ignores_sigterm(self):
    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / "escaped"
      child = "import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(1); pathlib.Path(%r).touch()" % str(marker)
      parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(30)" % child
      start = time.monotonic()
      self.assertEqual(run([sys.executable, "-c", parent], 0.3), 124)
      self.assertLess(time.monotonic() - start, 1)
      time.sleep(1)
      self.assertFalse(marker.exists(), "grandchild outlived the run budget")


if __name__ == "__main__":
  unittest.main()
