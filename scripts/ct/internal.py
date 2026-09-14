#!/usr/bin/env python3
"""Enable repository-only evidence hooks for a Cargo invocation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import resolved_rustflags

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_CFG = ["--cfg", "rscrypto_internal"]


def host_command(
    command: list[str], platform: str = os.name, environment: Mapping[str, str] = os.environ,
) -> list[str]:
  """Make repository shell wrappers executable by Windows subprocesses."""
  if platform == "nt" and Path(command[0]).suffix == ".sh":
    bash = environment.get("RSCRYPTO_BASH")
    if not bash:
      raise RuntimeError("RSCRYPTO_BASH is unset; run the Windows tooling installer first")
    return [bash, *command]
  return command


def build_environment(target: str, extra_flags: list[str] | None = None) -> tuple[dict[str, str], list[str]]:
  """Preserve resolved target flags and make the actual evidence flags recordable."""
  flags = [*resolved_rustflags(ROOT, target)[2], *(extra_flags or []), *INTERNAL_CFG]
  environment = os.environ.copy()
  # Encoded arguments preserve spaces and take precedence over ambient flag sources.
  environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
  return environment, flags


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--print-encoded-rustflags", action="store_true")
  parser.add_argument("command", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command[1:] if args.command[:1] == ["--"] else args.command
  if args.print_encoded_rustflags:
    if command:
      parser.error("--print-encoded-rustflags cannot be combined with a command")
    environment, _ = build_environment(args.target)
    print(environment["CARGO_ENCODED_RUSTFLAGS"], end="")
    return 0
  if not command:
    parser.error("a Cargo command is required after --")
  environment, _ = build_environment(args.target)
  return subprocess.run(host_command(command), cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
  raise SystemExit(main())
