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
                           (['slhdsa'], 600), (['slhdsa-sha2-128s'], 600),
                           ([selector, 'slhdsa-sha2-128s-verify'], 600),
                           ([selector, '--filter', 'rscrypto'], 600), ([selector, '--list'], 600),
                           ([selector, '--quick'], 600),
                           ([selector, '--prepare-archive', 'fixture.tar'], 600),
                           ([selector, '--run-archive', 'fixture.tar'], 1800)):
      with self.subTest(args=args), patch.dict(os.environ, {}, clear=True), patch('bounded.run', return_value=7) as execute:
        self.assertEqual(main(['--bench', *args]), 7)
        self.assertEqual(execute.call_args.args[1], expected)
        self.assertEqual(execute.call_args.args[0][2:], ['bench', *args])
        self.assertEqual(execute.call_args.kwargs, {'maximum': 1800})
    with patch.dict(os.environ, {'BENCH_QUICK': 'true'}, clear=True), patch('bounded.run', return_value=0) as execute:
      self.assertEqual(main(['--bench', selector]), 0)
      self.assertEqual(execute.call_args.args[1], 600)

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
