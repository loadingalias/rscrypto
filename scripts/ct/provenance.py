#!/usr/bin/env python3
"""Write CT provenance and primitive evidence index files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toml_compat import tomllib


def run(args: list[str], *, cwd: Path) -> str:
  return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def optional_version(args: list[str], *, cwd: Path) -> str | None:
  try:
    return run(args, cwd=cwd).strip() or None
  except (OSError, subprocess.CalledProcessError):
    return None


def llvm_tool(root: Path, name: str) -> str:
  sysroot = Path(run(["rustc", "--print", "sysroot"], cwd=root).strip())
  host = parse_rustc_verbose(run(["rustc", "-vV"], cwd=root)).get("host", "")
  candidate = sysroot / "lib" / "rustlib" / host / "bin" / name
  if candidate.exists():
    return str(candidate)
  return shutil.which(name) or name


def load_toml(path: Path) -> dict[str, Any]:
  with path.open("rb") as fh:
    return tomllib.load(fh)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def parse_rustc_verbose(text: str) -> dict[str, str]:
  out: dict[str, str] = {}
  for line in text.splitlines():
    if ":" in line:
      key, value = line.split(":", 1)
      out[key.strip().replace("-", "_")] = value.strip()
  return out


def rustc_channel(verbose: dict[str, str]) -> str:
  release = verbose.get("release", "")
  if "nightly" in release:
    return "nightly"
  if "beta" in release:
    return "beta"
  if release:
    return "stable"
  return "unknown"


def target_env_key(target: str, suffix: str) -> str:
  normalized = re.sub(r"[^A-Za-z0-9]", "_", target).upper()
  return f"CARGO_TARGET_{normalized}_{suffix}"


def target_rustflags(root: Path, target: str) -> list[str]:
  config_path = root / ".cargo" / "config.toml"
  if not config_path.exists():
    return []
  config = load_toml(config_path)
  flags = config.get("target", {}).get(target, {}).get("rustflags", [])
  if isinstance(flags, str):
    return flags.split()
  return list(flags)


def resolved_rustflags(root: Path, target: str) -> tuple[list[str], list[str], list[str], str]:
  configured = target_rustflags(root, target)
  if value := os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
    environment = [part for part in value.split("\x1f") if part]
    return configured, environment, environment, "CARGO_ENCODED_RUSTFLAGS"
  if value := os.environ.get("RUSTFLAGS"):
    environment = shlex.split(value)
    return configured, environment, environment, "RUSTFLAGS"
  target_key = target_env_key(target, "RUSTFLAGS")
  if value := os.environ.get(target_key):
    environment = shlex.split(value)
    return configured, environment, configured + environment, target_key
  if configured:
    return configured, [], configured, ".cargo/config.toml"
  if value := os.environ.get("CARGO_BUILD_RUSTFLAGS"):
    environment = shlex.split(value)
    return configured, environment, environment, "CARGO_BUILD_RUSTFLAGS"
  return configured, [], [], "none"


def codegen_value(flags: list[str], key: str) -> str:
  prefix = f"{key}="
  value = "unspecified"
  for idx, flag in enumerate(flags):
    if flag == "-C" and idx + 1 < len(flags) and flags[idx + 1].startswith(prefix):
      value = flags[idx + 1][len(prefix) :]
    elif flag.startswith(f"-C{prefix}"):
      value = flag[len(f"-C{prefix}") :]
  return value


def codegen_values(flags: list[str], key: str) -> list[str]:
  prefix = f"{key}="
  values: list[str] = []
  for idx, flag in enumerate(flags):
    if flag == "-C" and idx + 1 < len(flags) and flags[idx + 1].startswith(prefix):
      values.append(flags[idx + 1][len(prefix) :])
    elif flag.startswith(f"-C{prefix}"):
      values.append(flag[len(f"-C{prefix}") :])
  return values


def cfg_target_features(cfg_text: str) -> list[str]:
  features = []
  for line in cfg_text.splitlines():
    if match := re.fullmatch(r'target_feature="([^"]+)"', line.strip()):
      features.append(match.group(1))
  return sorted(features)


def resolve_linker(root: Path, target: str) -> tuple[str, str]:
  env_key = target_env_key(target, "LINKER")
  if value := os.environ.get(env_key):
    return value, env_key
  if value := os.environ.get("CARGO_TARGET_LINKER"):
    return value, "CARGO_TARGET_LINKER"

  config_path = root / ".cargo" / "config.toml"
  if config_path.exists():
    config = load_toml(config_path)
    linker = config.get("target", {}).get(target, {}).get("linker")
    if linker:
      return str(linker), ".cargo/config.toml"

  return "cc", "rustc-target-default"


def resolve_executable(command: str) -> Path | None:
  parts = shlex.split(command)
  if not parts:
    return None
  executable = "zig" if Path(parts[0]).name == "zig-cc.sh" else parts[0]
  found = shutil.which(executable)
  if found is None:
    return None
  return Path(found).resolve()


def resolved_linker_version(path: Path | None, configured_linker: str, root: Path) -> str | None:
  if path is None:
    return None
  parts = shlex.split(configured_linker)
  if parts and Path(parts[0]).name == "zig-cc.sh":
    zig_version = optional_version([str(path), "version"], cwd=root)
    return f"zig {zig_version}" if zig_version else None
  return optional_version([str(path), "--version"], cwd=root)


def artifact_kind(path: Path) -> str:
  name = path.name
  if name == "rscrypto-ct-evidence.link-map.txt":
    return "linked_binary_link_map"
  if name.endswith(".binary.raw-disasm.txt"):
    return "linked_binary_raw_disassembly"
  if name.endswith(".binary.nm-symbols.txt"):
    return "linked_binary_nm_symbol_map"
  if name.endswith(".binary.indirect-symbols.txt"):
    return "linked_binary_indirect_symbol_map"
  if name.endswith(".binary.disasm.txt"):
    return "linked_binary_disassembly"
  if name.endswith(".binary.symbols.txt"):
    return "linked_binary_symbol_map"
  if name.endswith(".binary.symbols.txt.rustfilt.txt"):
    return "linked_binary_demangled_symbol_map"
  if name.endswith(".binary.size.txt"):
    return "linked_binary_size"
  if name == "linker-command.txt":
    return "linker_command"
  if name in {"rscrypto-ct-evidence", "rscrypto-ct-evidence.exe"}:
    return "linked_binary"
  if name.endswith((".o.raw-symbols.txt", ".obj.raw-symbols.txt")):
    return "raw_symbol_map"
  if name.endswith((".o.disasm.txt", ".obj.disasm.txt")):
    return "object_disassembly"
  if name.endswith((".o.symbols.txt.rustfilt.txt", ".obj.symbols.txt.rustfilt.txt")):
    return "demangled_symbol_map"
  if name.endswith((".o.symbols.txt", ".obj.symbols.txt")):
    return "symbol_map"
  if name.endswith((".o.size.txt", ".obj.size.txt")):
    return "object_size"
  if name.endswith(".ll"):
    return "llvm_ir"
  if name.endswith(".s"):
    return "assembly"
  if name.endswith((".o", ".obj")):
    return "object"
  return "unknown"


def artifact_records(artifact_dir: Path) -> list[dict[str, Any]]:
  records = []
  for path in sorted(artifact_dir.iterdir()):
    if not path.is_file():
      continue
    records.append(
      {
        "path": f"artifacts/{path.name}",
        "name": path.name,
        "kind": artifact_kind(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
      }
    )
  return records


def write_artifact_hashes(out_dir: Path, artifacts: list[dict[str, Any]]) -> None:
  lines = [f"{artifact['sha256']}  {artifact['name']}" for artifact in artifacts]
  (out_dir / "artifact-hashes.txt").write_text("\n".join(lines) + "\n")


def component_lockfiles(root: Path) -> list[dict[str, str]]:
  paths = (
    "Cargo.lock",
    "tools/ct-harness/Cargo.lock",
    "tools/ct-dudect/Cargo.lock",
    "tools/ct-binsec-harness/Cargo.lock",
  )
  return [{"path": path, "sha256": sha256_file(root / path)} for path in paths]


def report_records(out_dir: Path) -> list[dict[str, Any]]:
  reports = []
  for name, kind in (
    ("asm-heuristics.json", "asm_heuristics"),
    ("asm-heuristics.md", "asm_heuristics_summary"),
  ):
    path = out_dir / name
    if path.exists():
      reports.append(
        {
          "path": name,
          "name": name,
          "kind": kind,
          "sha256": sha256_file(path),
          "bytes": path.stat().st_size,
        }
      )
  return reports


def symbol_objects(artifact_dir: Path) -> dict[str, list[dict[str, str]]]:
  symbols: dict[str, list[dict[str, str]]] = {}
  symbol_maps = sorted([*artifact_dir.glob("*.o.symbols.txt"), *artifact_dir.glob("*.obj.symbols.txt")])
  symbol_maps.extend(sorted(artifact_dir.glob("*.binary.symbols.txt")))
  for path in symbol_maps:
    object_name = path.name.removesuffix(".symbols.txt")
    for line in path.read_text().splitlines():
      if match := re.search(r"\b_?(ct_entry_[A-Za-z0-9_]+)\b", line):
        symbols.setdefault(match.group(1), []).append(
          {
            "object": object_name,
            "symbol_map": f"artifacts/{path.name}",
          }
        )
  return symbols


def primitive_evidence(ct: dict[str, Any], symbol_map: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
  rows = []
  for primitive in ct.get("primitive", []):
    harness = primitive.get("harness", {})
    symbol_rows = []
    for symbol in harness.get("symbols", []):
      locations = symbol_map.get(symbol, [])
      symbol_rows.append(
        {
          "name": symbol,
          "present": bool(locations),
          "locations": locations,
        }
      )
    rows.append(
      {
        "id": primitive.get("id"),
        "tier": primitive.get("tier"),
        "claim": primitive.get("claim"),
        "required": primitive.get("required", []),
        "secrets": primitive.get("secrets", []),
        "public": primitive.get("public", []),
        "may_leak": primitive.get("may_leak", []),
        "must_not_leak": primitive.get("must_not_leak", []),
        "must_not_leak_ref": primitive.get("must_not_leak_ref"),
        "harness": {
          "status": harness.get("status"),
          "coverage": harness.get("coverage"),
          "symbols": symbol_rows,
        },
      }
    )
  return rows


def git_status(root: Path) -> list[str]:
  text = run(["git", "status", "--short", "--untracked-files=all"], cwd=root)
  return [line for line in text.splitlines() if line]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", required=True)
  parser.add_argument("--artifact-dir", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  parser.add_argument("--build-target-dir", required=True, type=Path)
  parser.add_argument("--backend", default="llvm")
  parser.add_argument("--features", default="std,full")
  parser.add_argument("--linker-command-log", required=True, type=Path)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  cargo = load_toml(root / "Cargo.toml")
  harness_manifest = load_toml(root / "tools" / "ct-harness" / "Cargo.toml")
  ct = load_toml(root / "ct.toml")

  rustc_verbose_text = run(["rustc", "-vV"], cwd=root)
  rustc_verbose = parse_rustc_verbose(rustc_verbose_text)

  configured_rustflags, environment_rustflags, effective_rustflags, rustflags_source = resolved_rustflags(
    root, args.target
  )
  target_cfg_text = run(["rustc", "--print", "cfg", "--target", args.target, *effective_rustflags], cwd=root)
  linker, linker_source = resolve_linker(root, args.target)
  linker_path = resolve_executable(linker)
  artifacts = artifact_records(args.artifact_dir)
  reports = report_records(args.out_dir)
  write_artifact_hashes(args.out_dir, artifacts)

  dependency_lockfile = root / "tools" / "ct-harness" / "Cargo.lock"
  workspace_lockfile = root / "Cargo.lock"
  profile = harness_manifest.get("profile", {}).get(args.profile, {})
  status_entries = git_status(root)
  feature_list = [feature for feature in args.features.split(",") if feature]

  provenance = {
    "schema_version": 2,
    "kind": "rscrypto.ct.provenance",
    "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "crate": cargo.get("package", {}).get("name"),
    "crate_version": cargo.get("package", {}).get("version"),
    "manifest": "ct.toml",
    "ct_manifest_sha256": sha256_file(root / "ct.toml"),
    "package_manifest_sha256": sha256_file(root / "Cargo.toml"),
    "harness_manifest": "tools/ct-harness/Cargo.toml",
    "harness_manifest_sha256": sha256_file(root / "tools" / "ct-harness" / "Cargo.toml"),
    "harness_package": harness_manifest.get("package", {}).get("name"),
    "harness_version": harness_manifest.get("package", {}).get("version"),
    "harness_crate_type": harness_manifest.get("lib", {}).get("crate-type", []),
    "git_commit": run(["git", "rev-parse", "HEAD"], cwd=root).strip(),
    "git_dirty": bool(status_entries),
    "git_status_entries": len(status_entries),
    "git_status": status_entries,
    "rustc": run(["rustc", "--version"], cwd=root).strip(),
    "rustc_verbose": rustc_verbose_text,
    "rustc_channel": rustc_channel(rustc_verbose),
    "rustc_commit_hash": rustc_verbose.get("commit_hash"),
    "rustc_commit_date": rustc_verbose.get("commit_date"),
    "rustc_host": rustc_verbose.get("host"),
    "llvm_version": rustc_verbose.get("LLVM version"),
    "tools": {
      "python": sys.version,
      "cargo": run(["cargo", "-V"], cwd=root).strip(),
      "rustc": rustc_verbose_text.strip(),
      "llvm_objdump": optional_version([llvm_tool(root, "llvm-objdump"), "--version"], cwd=root),
      "llvm_nm": optional_version([llvm_tool(root, "llvm-nm"), "--version"], cwd=root),
      "llvm_size": optional_version([llvm_tool(root, "llvm-size"), "--version"], cwd=root),
      "rustfilt": optional_version([shutil.which("rustfilt") or "rustfilt", "--version"], cwd=root),
    },
    "backend": args.backend,
    "target": args.target,
    "target_triple": args.target,
    "target_cfg": target_cfg_text,
    "target_cfg_features": cfg_target_features(target_cfg_text),
    "target_cpu": codegen_value(effective_rustflags, "target-cpu"),
    "target_features": codegen_values(effective_rustflags, "target-feature"),
    "configured_rustflags": configured_rustflags,
    "environment_rustflags": environment_rustflags,
    "effective_rustflags": effective_rustflags,
    "rustflags_source": rustflags_source,
    "linker": linker,
    "linker_source": linker_source,
    "linker_path": str(linker_path) if linker_path is not None else None,
    "linker_sha256": sha256_file(linker_path) if linker_path is not None and linker_path.is_file() else None,
    "linker_version": resolved_linker_version(linker_path, linker, root),
    "linker_command": f"artifacts/{args.linker_command_log.name}",
    "linker_command_sha256": sha256_file(args.linker_command_log),
    "link_args": codegen_values(effective_rustflags, "link-arg"),
    "profile": args.profile,
    "opt_level": str(profile.get("opt-level", "unknown")),
    "lto": profile.get("lto", "unknown"),
    "panic": profile.get("panic", "unwind"),
    "codegen_units": profile.get("codegen-units", "unknown"),
    "overflow_checks": profile.get("overflow-checks", "unknown"),
    "debug": profile.get("debug", "unknown"),
    "strip": profile.get("strip", "unknown"),
    "features": feature_list,
    "default_features": False,
    "dependency_lockfile": "tools/ct-harness/Cargo.lock",
    "dependency_lockfile_sha256": sha256_file(dependency_lockfile),
    "workspace_lockfile": "Cargo.lock",
    "workspace_lockfile_sha256": sha256_file(workspace_lockfile),
    "component_lockfiles": component_lockfiles(root),
    "host_runner": socket.gethostname(),
    "host_uname": run(["uname", "-a"], cwd=root),
    "host_os": sys.platform,
    "build_target_dir": str(args.build_target_dir.relative_to(root)),
    "artifact_dir": str(args.out_dir.relative_to(root)),
    "artifacts": artifacts,
    "reports": reports,
  }

  provenance_path = args.out_dir / "provenance.json"
  provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

  symbols = symbol_objects(args.artifact_dir)
  expected_symbols = {symbol for harness in ct.get("harness", []) for symbol in harness.get("symbols", [])}
  evidence_index = {
    "schema_version": 2,
    "kind": "rscrypto.ct.evidence-index",
    "provenance": "provenance.json",
    "provenance_sha256": sha256_file(provenance_path),
    "crate": provenance["crate"],
    "crate_version": provenance["crate_version"],
    "backend": args.backend,
    "target": args.target,
    "target_triple": args.target,
    "profile": args.profile,
    "features": feature_list,
    "artifacts": artifacts,
    "reports": reports,
    "primitives": primitive_evidence(ct, symbols),
    "unmanifested_ct_symbols": sorted(symbol for symbol in symbols if symbol not in expected_symbols),
  }
  (args.out_dir / "evidence-index.json").write_text(json.dumps(evidence_index, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
