#!/usr/bin/env python3
"""Bound a whole benchmark/profile process tree, reserving time for shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from settings import load


class Terminated(Exception):
  """A termination request received by the watchdog itself."""


def terminate(signum, _frame):
  raise Terminated(signum)


def stop(process: subprocess.Popen, sig: int) -> None:
  try:
    if os.name == "posix":
      os.killpg(process.pid, sig)
    elif process.poll() is None:
      subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except ProcessLookupError:
    pass


def run(command: list[str], seconds: float) -> int:
  if not 0 < seconds <= 3600:
    raise ValueError("run budget must be positive and at most one hour")
  grace = min(5.0, seconds / 10)
  with subprocess.Popen(command, start_new_session=os.name == "posix") as process:
    try:
      return process.wait(timeout=seconds - grace)
    except subprocess.TimeoutExpired:
      print(f"error: {seconds:g}s run budget exhausted; stopping the process tree", file=sys.stderr, flush=True)
      status = 124
      stop(process, signal.SIGTERM)
    except (KeyboardInterrupt, Terminated) as error:
      sig = error.args[0] if isinstance(error, Terminated) else signal.SIGINT
      status = 128 + sig
      stop(process, sig)
    try:
      process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
      pass
    finally:
      stop(process, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
    process.wait()
    return status


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, terminate)
  try:
    raise SystemExit(run(sys.argv[1:], load()["max_run_seconds"]))
  except (ValueError, OSError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)
