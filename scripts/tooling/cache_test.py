#!/usr/bin/env python3
"""Exercise the repository cache front door against a recorded Cargo boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tooling/cache.sh"
REMOTE = "r2://0123456789abcdef0123456789abcdef/rscrypto-cache/rscrypto/shared"


with tempfile.TemporaryDirectory(prefix="cache setup ") as temporary:
  root = Path(temporary)
  cargo = root / "cargo"
  log = root / "calls.jsonl"
  cargo.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
with open(os.environ["CACHE_CALL_LOG"], "a") as log:
  log.write(json.dumps(sys.argv[1:]) + "\\n")
if "--check" in sys.argv:
  raise SystemExit(int(os.environ.get("CACHE_CHECK_EXIT", "0")))
if sys.argv[1:] == ["rail", "cache", "ready"] and (
    os.environ.get("CARGO_RAIL_CACHE_REMOTE") or os.environ.get("CARGO_RAIL_CACHE_MODE")
):
  raise SystemExit(2)
''')
  cargo.chmod(0o755)
  base = {key: value for key, value in os.environ.items()
          if key not in ("BASH_ENV", "ENV") and not key.startswith("BASH_FUNC_")}
  base.update(PATH=str(root) + os.pathsep + os.environ["PATH"], CACHE_CALL_LOG=str(log))

  def run(extra=None, *args):
    log.write_text("")
    result = subprocess.run([str(SCRIPT), *args], cwd=ROOT, env={**base, **(extra or {})},
                            text=True, capture_output=True, timeout=20)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, calls

  result, calls = run(None, "--max-size", "10GiB")
  assert result.returncode == 0, result.stderr
  expected = [
    ["rail", "cache", "setup", "--check", "--local-only", "--max-size", "10GiB"],
    ["rail", "cache", "setup", "--local-only", "--max-size", "10GiB"],
    ["rail", "cache", "ready"],
  ]
  assert calls == expected, (calls, result.stdout, result.stderr)

  remote_env = {"CARGO_RAIL_CACHE_REMOTE": REMOTE, "CARGO_RAIL_CACHE_MODE": "read-write"}
  result, calls = run(remote_env, "--max-size", "10GiB")
  assert result.returncode == 0, result.stderr
  authority = ["--remote", REMOTE, "--remote-mode", "read-write", "--root-portability", "remap"]
  expected = [
    ["rail", "cache", "setup", "--check", *authority, "--max-size", "10GiB"],
    ["rail", "cache", "setup", "--local-only", "--max-size", "10GiB"],
    ["rail", "cache", "ready"],
    ["rail", "cache", "setup", *authority, "--max-size", "10GiB"],
    ["rail", "cache", "probe"],
  ]
  assert calls == expected, (calls, result.stdout, result.stderr)

  result, calls = run({"CACHE_CHECK_EXIT": "1"})
  assert result.returncode == 0, result.stderr
  assert len(calls) == 3

  result, calls = run({"CACHE_CHECK_EXIT": "2"})
  assert result.returncode == 2
  assert calls == [["rail", "cache", "setup", "--check", "--local-only"]]

  result, calls = run({"CARGO_RAIL_CACHE_MODE": "read"})
  assert result.returncode == 2
  assert not calls
  assert "requires CARGO_RAIL_CACHE_REMOTE" in result.stderr

  result, calls = run({"CARGO_RAIL_CACHE_REMOTE": REMOTE})
  assert result.returncode == 2
  assert not calls
  assert "CARGO_RAIL_CACHE_MODE is required" in result.stderr

print("Cache setup regressions passed")
