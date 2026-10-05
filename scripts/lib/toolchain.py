#!/usr/bin/env python3
"""Resolve the canonical Rust toolchain for provisioning and execution.

`rust-toolchain.toml` owns the only development compiler for every host,
target, and tool lane. The MSRV lane uses the manifest's `rust-version`,
except while that release is unpublished: a canonical nightly of the same
release runs the MSRV lane itself, and a nightly one release ahead means the
MSRV is in beta, so the lane runs on the exact `MSRV_PREVIEW` beta.
"""

import argparse
import functools
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
# Exact beta of the unpublished MSRV release. Ignored once the canonical nightly
# is two releases past the MSRV, because the MSRV is then a stable release.
MSRV_PREVIEW = "1.100.0-beta.1"


def channel():
  return tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]


def msrv():
  return tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["rust-version"]


@functools.cache
def release(selected):
  """Return the `rustc` release string (for example `1.100.0-nightly`) of a toolchain."""
  output = subprocess.check_output(["rustc", "+" + selected, "-vV"], text=True)
  for line in output.splitlines():
    if line.startswith("release: "):
      return line.removeprefix("release: ").strip()
  raise ValueError(f"cannot determine the {selected} release")


def minor(version):
  return int(version.split(".")[1])


def msrv_channel():
  """Toolchain that validates the declared MSRV.

  Rust releases follow a train: when the canonical nightly is release N, the
  beta is N-1 and every release up to N-2 is stable. A preview MSRV is
  accepted only on its own nightly or its exact beta; otherwise the MSRV must
  name an installable released toolchain.
  """
  declared = msrv()
  canonical = channel()
  if not canonical.startswith("nightly-"):
    return declared
  current = release(canonical)
  if current == declared + "-nightly":
    return canonical
  if minor(current) == minor(declared) + 1:
    if not MSRV_PREVIEW.startswith(declared + "-beta."):
      raise ValueError(f"MSRV {declared} is in beta under {canonical}; "
                       f"set MSRV_PREVIEW to an exact {declared}-beta.N")
    return MSRV_PREVIEW
  return declared


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
    print(msrv_channel())
  else:
    print(channel())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
