#!/usr/bin/env -S uv run python
"""Bound a whole benchmark/profile process tree, reserving time for shutdown."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from settings import MAX_RUN_SECONDS, load


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


def run(command: list[str], seconds: float, *, maximum: int = MAX_RUN_SECONDS) -> int:
  if not 0 < seconds <= maximum <= load()["max_comparison_seconds"]:
    raise ValueError(f"run budget must be positive and at most {maximum} seconds")
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


def main(arguments: list[str]) -> int:
  if arguments[:1] == ["--bench"]:
    import runner
    args = runner.parse(["bench", *arguments[1:]])
    catalog = runner.load_catalog()
    command = [sys.executable, str(Path(runner.__file__)), "bench", *arguments[1:]]
    # Each target's build and discovery runs alone, under the longest permitted bound: a cold build of the
    # auth target took over 600 s on Windows (Bench 39867778994 and 39880626790). Measurement keeps its own
    # budget and finds the build current.
    build = load()["max_comparison_seconds"]
    targets = dict.fromkeys(row["bench"] for row in runner.requests(args, catalog))
    if args.list and not args.list_target:
      for bench in targets:
        status = run([*command, "--list-target", bench], build, maximum=build)
        if status:
          return status
      return 0
    if not (args.prepare_archive or args.run_archive):
      for bench in targets:
        status = run([*command, "--list", "--list-target", bench], build, maximum=build)
        if status:
          return status
    return run(command, runner.run_budget(args, catalog), maximum=load()["max_comparison_seconds"])
  return run(arguments, load()["max_run_seconds"])


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, terminate)
  try:
    raise SystemExit(main(sys.argv[1:]))
  except (ValueError, OSError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)
