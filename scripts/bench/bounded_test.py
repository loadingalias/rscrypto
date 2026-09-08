#!/usr/bin/env python3
"""Verify timeout enforcement on a real child process tree."""

import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest

from bounded import run


class BudgetTests(unittest.TestCase):
  def test_exit_status_and_limit_validation(self):
    self.assertEqual(run([sys.executable, "-c", "raise SystemExit(7)"], 2), 7)
    for limit in (0, -1, 3601):
      with self.assertRaises(ValueError):
        run([sys.executable, "-c", "pass"], limit)

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
