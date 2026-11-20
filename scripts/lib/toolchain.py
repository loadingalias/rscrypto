#!/usr/bin/env -S uv run python
"""Resolve the canonical Rust toolchain for provisioning and execution.

`rust-toolchain.toml` owns the only development compiler for every host,
target, and tool lane. The MSRV lane uses the released toolchain that the
manifest's `rust-version` names.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def channel():
  return tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]


def msrv():
  """Released toolchain that validates the declared MSRV: the manifest's `rust-version`."""
  return tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["rust-version"]


def host():
  output = subprocess.check_output(["rustc", "+" + channel(), "-vV"], text=True)
  for line in output.splitlines():
    if line.startswith("host: "):
      return line.removeprefix("host: ").strip()
  raise ValueError("cannot determine Rust host")


def select():
  os.environ["RUSTUP_TOOLCHAIN"] = channel()
  return channel()


def install_command(components):
  return ["rustup", "toolchain", "install", channel(), "--profile", "minimal",
          *[arg for component in dict.fromkeys(["clippy", "rustfmt", *components])
            for arg in ("--component", component)]]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--msrv", action="store_true", help="print the toolchain for the MSRV lane")
  mode.add_argument("--print-host", action="store_true")
  mode.add_argument("--install", action="store_true", help="install the canonical toolchain")
  mode.add_argument("--exec", dest="command", nargs=argparse.REMAINDER)
  parser.add_argument("--component", action="append", default=[])
  argv = sys.argv[1:]
  command = None
  if '--exec' in argv:
    index = argv.index('--exec')
    # The child owns every token after --exec, including its own -- delimiter.
    # argparse otherwise consumes that delimiter even with REMAINDER.
    command = argv[index + 1:]
    argv = argv[:index + 1]
  args = parser.parse_args(argv)
  if command is not None:
    args.command = command
  if args.print_host:
    print(host())
  elif args.command is not None:
    if not args.command:
      parser.error("--exec requires a command")
    select()
    return subprocess.run(args.command).returncode
  elif args.install:
    subprocess.run(install_command(args.component), check=True)
  elif args.msrv:
    print(msrv())
  else:
    print(channel())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
