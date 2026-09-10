#!/usr/bin/env python3
"""Resolve pinned Rust toolchains for provisioning and execution."""

import argparse
import fnmatch
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def stable():
  return tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]


def contracts():
  return tomllib.loads((ROOT / ".config/toolchains.toml").read_text())


def for_target(target):
  policy = contracts()
  return policy["nightly"] if any(fnmatch.fnmatchcase(target, pattern) for pattern in policy["nightly_targets"]) else stable()


def host():
  output = subprocess.check_output(["rustc", "+" + stable(), "-vV"], text=True)
  for line in output.splitlines():
    if line.startswith("host: "):
      return line.removeprefix("host: ").strip()
  raise ValueError("cannot determine Rust host")


def select_host():
  channel = for_target(host())
  os.environ["RUSTUP_TOOLCHAIN"] = channel
  return channel


def install_commands(target, components):
  development = stable()
  native = for_target(target)
  return [["rustup", "toolchain", "install", channel, "--profile", "minimal",
           *[arg for component in dict.fromkeys([
             *(["clippy"] if channel == native else []),
             *(["rustfmt"] if channel == development else []), *components])
             for arg in ("--component", component)]]
          for channel in dict.fromkeys([development, native])]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--nightly", action="store_true")
  mode.add_argument("--target")
  mode.add_argument("--host", action="store_true")
  mode.add_argument("--print-host", action="store_true")
  mode.add_argument("--install", metavar="HOST")
  mode.add_argument("--exec", dest="command", nargs=argparse.REMAINDER)
  parser.add_argument("--component", action="append", default=[])
  args = parser.parse_args()
  if args.print_host:
    print(host())
    return 0
  if args.command is not None:
    if not args.command:
      parser.error("--exec requires a command")
    select_host()
    return subprocess.run(args.command).returncode
  if args.install:
    for command in install_commands(args.install, args.component):
      subprocess.run(command, check=True)
    return 0
  if args.nightly:
    print(contracts()["nightly"])
  elif args.target or args.host:
    print(for_target(args.target or host()))
  else:
    print(stable())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
