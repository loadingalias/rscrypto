"""Shared Cargo builds, Criterion discovery, and execution evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from toolchain import select_host

from evidence import collect


def read_json(path: Path):
  return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
  path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path: Path) -> str:
  with path.open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def identity(value) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def exit_code(error) -> int:
  if isinstance(error, subprocess.CalledProcessError):
    return error.returncode if error.returncode > 0 else 128 - error.returncode
  if isinstance(error, SystemExit):
    return int(error.code or 0)
  return 130 if isinstance(error, KeyboardInterrupt) else 1


def execute(command: list[str], output: Path, *, env=None, capture=False) -> str:
  print(f"Running: {shlex.join(command)}", flush=True)
  captured = []
  with output.open("a+") as log:
    start = log.tell()
    log.write(f"\nRunning: {shlex.join(command)}\n")
    log.flush()
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                          stderr=log if capture else subprocess.STDOUT, text=True) as process:
      for line in process.stdout:
        log.write(line)
        if capture:
          captured.append(line)
        else:
          print(line, end="", flush=True)
      status = process.wait()
    if status:
      if capture:
        log.flush()
        log.seek(start)
        shutil.copyfileobj(log, sys.stderr)
        sys.stderr.flush()
      raise subprocess.CalledProcessError(status, command)
  return "".join(captured)


def build_environment() -> None:
  select_host()
  local = os.environ.get("RSCRYPTO_BENCH_MODE", "remote" if os.environ.get("DEV_MACHINE_TARGET") else "local") == "local"
  if local and platform.system() == "Darwin" and not {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"} & os.environ.keys():
    os.environ["RUSTFLAGS"] = "-C target-cpu=native"


def build_command(binary: str, features: list[str]) -> list[str]:
  return ["cargo", "bench", "--locked", "--profile", "bench", "--features", ",".join(sorted(set(features))),
          "--no-default-features", "--bench", binary]



def hardware() -> dict:
  host = {"system": platform.system(), "machine": platform.machine(), "cpus": os.cpu_count()}
  if platform.system() == "Darwin":
    host["model"] = subprocess.check_output(["sysctl", "-n", "hw.model", "machdep.cpu.brand_string"], text=True).strip()
  elif Path("/proc/cpuinfo").is_file():
    fields = {"vendor_id", "model name", "cpu family", "model", "stepping", "siblings", "cpu cores", "Features", "flags", "CPU architecture", "CPU implementer", "CPU part", "machine", "processor"}
    host["cpu"] = sorted(set(line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                             if ":" in line and line.split(":", 1)[0].strip() in fields))
  else:
    host["model"] = platform.processor()
  return host

def build_identity() -> dict:
  root = Path.cwd()
  paths = [directory / ".cargo" / name for directory in (root, *root.parents) for name in ("config", "config.toml")]
  cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
  paths += [cargo_home / name for name in ("config", "config.toml")]
  # File contents, rather than checkout paths, identify configuration. Missing
  # files are omitted; the manifest's source revision is the A/B variable.
  configs = [digest(path) for path in dict.fromkeys(paths) if path.is_file()]
  return {
    "environment": collect(),
    "rustc": subprocess.check_output([os.environ.get("RUSTC", "rustc"), "-Vv"], text=True).strip(),
    "cargo": subprocess.check_output(["cargo", "-V"], text=True).strip(),
    "profiles": tomllib.loads((root / "Cargo.toml").read_text()).get("profile", {}),
    "cargo_config_sha256": configs,
    "host": hardware(),
  }


def build(command: list[str], log: Path, env: dict) -> dict:
  binary = command[command.index("--bench") + 1]
  messages = execute([*command, "--no-run", "--message-format=json"], log, env=env, capture=True)
  artifacts = [message for line in messages.splitlines() if
               (message := json.loads(line)).get("reason") == "compiler-artifact"
               and message.get("executable") and message["target"]["name"] == binary
               and "bench" in message["target"]["kind"]]
  if len(artifacts) != 1:
    raise ValueError(f"expected one Cargo executable for {binary}, got {len(artifacts)}")
  path = Path(artifacts[0]["executable"]).resolve()
  if not path.is_file() or not os.access(path, os.X_OK):
    raise ValueError(f"missing benchmark executable: {path}")
  return {"path": str(path), "sha256": digest(path), "cargo": artifacts[0], "command": command}


def unchanged(artifact: dict) -> None:
  if digest(Path(artifact["path"])) != artifact["sha256"]:
    raise ValueError(f"benchmark executable changed: {artifact['path']}")


def discover(artifact: dict, pattern: str, log: Path, env: dict) -> list[str]:
  unchanged(artifact)
  command = [artifact["path"], "--bench", "--list", "--format", "terse"]
  if pattern:
    command += ["--", pattern]
  listing = execute(command, log, env=env, capture=True)
  cases = [line.removesuffix(": benchmark") for line in listing.splitlines() if line.endswith(": benchmark")]
  if len(cases) != len(set(cases)):
    raise ValueError("ambiguous duplicate case identities")
  unchanged(artifact)
  return cases


def match_cases(artifact, cases, patterns, log, env):
  unchanged(artifact)
  with tempfile.TemporaryDirectory(prefix="rscrypto-filters-") as directory:
    request = Path(directory) / "request.json"
    write_json(request, {"cases": cases, "patterns": list(dict.fromkeys(patterns))})
    output = execute([artifact["path"], "--rscrypto-filter-cases", str(request)], log, env=env, capture=True)
  unchanged(artifact)
  return json.loads(output)


def source_evidence(root: Path) -> None:
  names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
                                   "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml", ".cargo", ".config", "src", "benches", "scripts"])
  write_json(root / "source.json", {name: digest(Path(name)) if Path(name).is_file() else None
                                   for name in sorted(set(names.decode().split("\0")) - {""})})
  write_json(root / "source-state.json", {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "status": subprocess.check_output(["git", "status", "--short"], text=True),
  })
