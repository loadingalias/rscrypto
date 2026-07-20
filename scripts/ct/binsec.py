#!/usr/bin/env python3
"""Run BINSEC/Rel checks for rscrypto CT kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from toml_compat import tomllib


ROOT = Path(__file__).resolve().parents[2]
HARNESS_MANIFEST = ROOT / "tools" / "ct-binsec-harness" / "Cargo.toml"
HARNESS_BIN = "rscrypto-ct-binsec-harness"


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


@lru_cache(maxsize=1)
def candidate_identity() -> dict[str, object]:
  with (ROOT / "Cargo.toml").open("rb") as source:
    crate = tomllib.load(source)
  with HARNESS_MANIFEST.open("rb") as source:
    harness = tomllib.load(source)
  dependency = harness["dependencies"]["rscrypto"]
  rustc_verbose = run(["rustc", "-vV"]).stdout.strip()
  status = run(["git", "status", "--short", "--untracked-files=all"]).stdout.splitlines()
  return {
    "crate_version": crate["package"]["version"],
    "git_commit": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
    "git_dirty": bool(status),
    "git_status": status,
    "ct_manifest_sha256": sha256_file(ROOT / "ct.toml"),
    "harness_manifest_sha256": sha256_file(HARNESS_MANIFEST),
    "harness_lockfile_sha256": sha256_file(ROOT / "tools" / "ct-binsec-harness" / "Cargo.lock"),
    "rustc_verbose": rustc_verbose,
    "cargo": run(["cargo", "-V"]).stdout.strip(),
    "features": dependency["features"],
    "default_features": dependency.get("default-features", True),
    "profile_settings": harness.get("profile", {}).get("release", {}),
  }


def run(
  cmd: list[str],
  *,
  cwd: Path = ROOT,
  capture: bool = True,
  timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
  try:
    return subprocess.run(
      cmd,
      cwd=cwd,
      text=True,
      stdout=subprocess.PIPE if capture else None,
      stderr=subprocess.PIPE if capture else None,
      check=False,
      timeout=timeout,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
    stderr += f"\ncommand timed out after {timeout} seconds\n"
    return subprocess.CompletedProcess(cmd, 124, stdout, stderr)


def cargo_target_env_name(target: str, suffix: str) -> str:
  return f"CARGO_TARGET_{target.upper().replace('-', '_')}_{suffix}"


def rustc_host() -> str:
  result = run(["rustc", "-vV"])
  if result.returncode != 0:
    raise SystemExit(result.stderr)
  for line in result.stdout.splitlines():
    if line.startswith("host:"):
      return line.split(":", 1)[1].strip()
  raise SystemExit("could not determine rustc host target")


def load_manifest() -> dict:
  with (ROOT / "ct.toml").open("rb") as fh:
    return tomllib.load(fh)


def find_binsec() -> str | None:
  found = shutil.which("binsec")
  if found is not None:
    return found
  home = Path.home()
  for candidate in sorted((home / ".opam").glob("*/bin/binsec")):
    if candidate.is_file() and os.access(candidate, os.X_OK):
      return str(candidate)
  return None


def kernel_targets(kernel: dict, ct: dict) -> set[str]:
  targets = kernel.get("targets", [])
  if "*" in targets:
    return binsec_required_targets(ct)
  return set(targets)


def target_record(ct: dict, target: str) -> dict | None:
  for row in ct.get("target", []):
    if row.get("name") == target:
      return row
  return None


def target_binsec_policy(ct: dict, target: str) -> str:
  row = target_record(ct, target)
  if row is None:
    return "unsupported"
  return str(row.get("binsec", "unsupported"))


def binsec_required_targets(ct: dict) -> set[str]:
  return {
    target.get("name", "")
    for target in ct.get("target", [])
    if target.get("claim") in {"ct-intended", "ct-claimed"} and target.get("binsec") == "required"
  }


def kernels(ct: dict, kernel_filter: str | None, target: str) -> list[dict]:
  rows = ct.get("binsec_kernel", [])
  rows = [row for row in rows if not row.get("targets") or target in kernel_targets(row, ct)]
  if kernel_filter is not None:
    rows = [row for row in rows if row.get("id") == kernel_filter or row.get("symbol") == kernel_filter]
  return rows


def target_is_claimed(ct: dict, target: str) -> bool:
  for row in ct.get("target", []):
    if row.get("name") == target:
      return row.get("claim") in {"ct-intended", "ct-claimed"}
  return False


def llvm_tool(name: str) -> str:
  env_name = name.upper().replace("-", "_")
  override = os.environ.get(env_name)
  if override:
    return override

  sysroot = run(["rustc", "--print", "sysroot"]).stdout.strip()
  host = rustc_host()
  candidate = Path(sysroot) / "lib" / "rustlib" / host / "bin" / name
  if candidate.exists():
    return str(candidate)
  if Path(str(candidate) + ".exe").exists():
    return str(candidate) + ".exe"
  return name


def configure_cross_linker(env: dict[str, str], target: str) -> None:
  host = rustc_host()
  if "linux" not in target:
    return

  linker_env = cargo_target_env_name(target, "LINKER")
  if env.get(linker_env):
    return

  host_arch = host.split("-", 1)[0]
  target_arch = target.split("-", 1)[0]
  if target.endswith("-linux-musl") and host_arch == target_arch:
    musl = shutil.which("musl-gcc")
    if musl is not None:
      env[linker_env] = musl
      return

  if "linux" in host:
    return

  zig = shutil.which("zig")
  wrapper = ROOT / "scripts" / "check" / "zig-cc.sh"
  if zig is None or not wrapper.exists():
    return

  env[linker_env] = str(wrapper)
  env.setdefault("ZIG_CC_TARGET", target)


def default_target_rustflags(target: str) -> list[str]:
  if target.startswith("x86_64-"):
    return ["-C", "target-cpu=x86-64"]
  if target.startswith("aarch64-"):
    return ["-C", "target-cpu=generic"]
  if target.startswith("powerpc64le-"):
    return ["-C", "target-feature=+altivec,+vsx,+power8-vector"]
  if target.startswith("s390x-") or target.startswith("riscv"):
    return ["-C", "target-cpu=generic"]
  return []


def binsec_proof_rustflags(target: str) -> list[str]:
  if target.endswith("-unknown-linux-gnu"):
    return ["-C", "relocation-model=static", "-C", "link-arg=-no-pie"]
  return []


def elf_type(binary: Path) -> str | None:
  try:
    header = binary.read_bytes()[:18]
  except OSError:
    return None
  if len(header) < 18 or header[:4] != b"\x7fELF":
    return None
  byteorder = "little" if header[5] == 1 else "big" if header[5] == 2 else None
  if byteorder is None:
    return None
  value = int.from_bytes(header[16:18], byteorder)
  names = {
    1: "relocatable",
    2: "exec",
    3: "dyn",
    4: "core",
  }
  return names.get(value, f"unknown-{value}")


def build_harness(target: str, profile: str, rustflags: list[str]) -> tuple[Path, list[str]]:
  if profile != "release":
    raise SystemExit("only --profile release is supported for BINSEC today")

  target_dir = ROOT / "target" / "ct-binsec-build"
  cmd = [
    "cargo",
    "build",
    "--locked",
    "--manifest-path",
    str(HARNESS_MANIFEST),
    "--target-dir",
    str(target_dir),
    "--target",
    target,
    "--release",
  ]
  env = os.environ.copy()
  configure_cross_linker(env, target)
  effective_rustflags = list(rustflags)
  if not rustflags:
    effective_rustflags.extend(default_target_rustflags(target))
  effective_rustflags.extend(binsec_proof_rustflags(target))
  if effective_rustflags:
    existing = env.get("RUSTFLAGS", "")
    env["RUSTFLAGS"] = " ".join([existing, *effective_rustflags]).strip()
  result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, check=False)
  if result.returncode != 0:
    raise SystemExit(result.returncode)

  suffix = ".exe" if "windows" in target else ""
  binary = target_dir / target / "release" / f"{HARNESS_BIN}{suffix}"
  if not binary.exists():
    raise SystemExit(f"BINSEC harness binary missing: {binary}")
  return binary, effective_rustflags


def symbol_names(binary: Path) -> set[str]:
  nm = llvm_tool("llvm-nm")
  result = run([nm, "--defined-only", str(binary)])
  if result.returncode != 0:
    return set()
  names: set[str] = set()
  for line in result.stdout.splitlines():
    parts = line.split()
    if parts:
      names.add(parts[-1])
  return names


def resolve_symbol(names: set[str], symbol: str) -> str | None:
  if symbol in names:
    return symbol
  prefixed = f"_{symbol}"
  if prefixed in names:
    return prefixed
  return None


def disassemble(binary: Path, out: Path) -> None:
  objdump = llvm_tool("llvm-objdump")
  result = run([objdump, "--disassemble", "--reloc", "--demangle", str(binary)])
  if result.returncode != 0:
    out.write_text(result.stdout + result.stderr)
    raise SystemExit(f"llvm-objdump failed for {binary}")
  out.write_text(result.stdout)


def elf_sections(binary: Path) -> set[str]:
  objdump = llvm_tool("llvm-objdump")
  result = run([objdump, "--section-headers", str(binary)])
  if result.returncode != 0:
    return set()
  names: set[str] = set()
  for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[0].isdigit():
      names.add(parts[1])
  return names


def binsec_load_sections(binary: Path) -> list[str]:
  available = elf_sections(binary)
  preferred = [".text", ".rodata", ".data.rel.ro", ".got", ".got.plt", ".data", ".bss"]
  selected = [section for section in preferred if section in available]
  return selected or [".text", ".rodata", ".data", ".bss"]


def binsec_script(kernel: dict, *, start_symbol: str, done_symbol: str, load_sections: list[str]) -> str:
  secret_lines = []
  for secret in kernel.get("secrets", []):
    if secret.get("kind") == "global":
      secret_lines.append(f"secret global {secret['name']}")

  public_lines = []
  for public in kernel.get("public", []):
    if public.get("kind") == "global":
      public_lines.append(f"public global {public['name']}")

  assumptions = [f"assume {assumption}" for assumption in kernel.get("assumptions", [])]
  body = [
    f"load sections {', '.join(load_sections)} from file",
    f"starting from <{start_symbol}>",
    "with concrete stack pointer",
    *secret_lines,
    *public_lines,
    *assumptions,
    f"halt at <{done_symbol}>",
    "explore all",
    "",
  ]
  return "\n".join(body)


def parse_status(stdout: str, returncode: int) -> tuple[str, str]:
  lowered = stdout.lower()
  if returncode != 0:
    if "native bitwuzla binding is required" in lowered or "solver" in lowered:
      return "unknown", "binsec SMT solver unavailable"
    if "can not resolve decoder" in lowered:
      return "unknown", "binsec decoder unavailable for target"
    return "unknown", "binsec exited nonzero"
  if "[checkct:result] program status is : secure" in lowered:
    return "secure", "binsec reported secure"
  if "[checkct:result] program status is : insecure" in lowered:
    return "insecure", "binsec reported insecure"
  if "[checkct:result] program status is : unknown" in lowered:
    if "smt-timeout" in lowered or "solver timeout" in lowered:
      return "unknown", "binsec exploration incomplete: SMT solver timeout"
    if "exploration is incomplete" in lowered:
      return "unknown", "binsec exploration incomplete"
    return "unknown", "binsec reported unknown"
  return "unknown", "binsec did not emit a recognized checkct result"


def write_report(
  path: Path,
  *,
  kernel: dict,
  target: str,
  profile: str,
  status: str,
  reason: str,
  artifacts: dict[str, str],
  binsec_version: str | None,
  timeout_seconds: int | None,
  smt_timeout_seconds: int | None,
  smt_solver: str | None,
  rustflags: list[str] | None = None,
  harness_elf_type: str | None = None,
  load_sections: list[str] | None = None,
) -> None:
  report = {
    "schema_version": 1,
    "kind": "rscrypto.ct.binsec",
    "crate": "rscrypto",
    **candidate_identity(),
    "backend": "llvm",
    "target": target,
    "target_triple": target,
    "profile": profile,
    "kernel": kernel.get("id"),
    "primitive": kernel.get("primitive"),
    "symbol": kernel.get("symbol"),
    "required": bool(kernel.get("required", False)),
    "status": status,
    "reason": reason,
    "timeout_seconds": timeout_seconds,
    "smt_timeout_seconds": smt_timeout_seconds,
    "smt_solver": smt_solver,
    "rustflags": rustflags or [],
    "harness_elf_type": harness_elf_type,
    "load_sections": load_sections or [],
    "sse_depth": kernel.get("sse_depth"),
    "binsec_version": binsec_version,
    "artifacts": artifacts,
  }
  path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", default=None, help="target triple")
  parser.add_argument("--profile", default="release", help="build profile")
  parser.add_argument("--kernel", default=None, help="kernel id or symbol to run")
  parser.add_argument("--timeout", default="120", help="BINSEC timeout in seconds")
  parser.add_argument("--smt-timeout", default=None, help="SMT solver timeout in seconds; defaults to --timeout")
  parser.add_argument(
    "--smt-solver",
    default=os.environ.get("BINSEC_SMT_SOLVER", "bitwuzla:builtin"),
    help="BINSEC SMT solver backend; default: BINSEC_SMT_SOLVER or bitwuzla:builtin",
  )
  parser.add_argument(
    "--allow-missing-binsec",
    action="store_true",
    help="write blocked reports instead of failing when binsec is absent",
  )
  args = parser.parse_args()

  target = args.target or rustc_host()
  profile = args.profile
  ct = load_manifest()
  if target_binsec_policy(ct, target) != "required":
    print(f"BINSEC is not required for {target} by ct.toml policy")
    return 0

  out_root = ROOT / "target" / "ct" / target / profile / "binsec"
  selected = kernels(ct, args.kernel, target)
  if not selected:
    if out_root.exists():
      shutil.rmtree(out_root)
    if args.kernel is None and target_is_claimed(ct, target):
      print(f"no BINSEC kernels selected for claimed target {target}", file=sys.stderr)
      return 1
    print(f"no BINSEC kernels selected for {target}")
    return 0

  binsec = find_binsec()
  binsec_version = None
  if binsec is not None:
    version = run([binsec, "-version"])
    binsec_version = (version.stdout or version.stderr).strip() or None

  if binsec is None and not args.allow_missing_binsec:
    print(
      "binsec not found; install BINSEC/Rel or pass --allow-missing-binsec to emit blocked reports",
      file=sys.stderr,
    )
    return 2

  if args.kernel is None and out_root.exists():
    shutil.rmtree(out_root)
  out_root.mkdir(parents=True, exist_ok=True)

  failures = 0
  for kernel in selected:
    kernel_id = kernel["id"]
    out_dir = out_root / kernel_id.replace("/", "_").replace(":", "_")
    if out_dir.exists():
      shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    report_path = out_dir / "binsec-report.json"
    artifacts: dict[str, str] = {}

    if binsec is None:
      write_report(
        report_path,
        kernel=kernel,
        target=target,
        profile=profile,
        status="blocked",
        reason="binsec not found",
        artifacts=artifacts,
        binsec_version=binsec_version,
        timeout_seconds=None,
        smt_timeout_seconds=None,
        smt_solver=args.smt_solver,
        rustflags=[],
        harness_elf_type=None,
        load_sections=[],
      )
      continue

    rustflags = [str(flag) for flag in kernel.get("rustflags", [])]
    kernel_timeout = int(kernel.get("timeout", args.timeout))
    kernel_smt_timeout = int(kernel.get("smt_timeout", args.smt_timeout if args.smt_timeout is not None else kernel_timeout))
    binary, effective_rustflags = build_harness(target, profile, rustflags)
    harness_elf_type = elf_type(binary)
    names = symbol_names(binary)

    assert binary is not None
    start_symbol = resolve_symbol(names, kernel["symbol"])
    done_symbol = resolve_symbol(names, "ct_binsec_done")
    if start_symbol is None:
      write_report(
        report_path,
        kernel=kernel,
        target=target,
        profile=profile,
        status="unknown",
        reason=f"missing symbol {kernel['symbol']}",
        artifacts=artifacts,
        binsec_version=binsec_version,
        timeout_seconds=kernel_timeout,
        smt_timeout_seconds=kernel_smt_timeout,
        smt_solver=args.smt_solver,
        rustflags=effective_rustflags,
        harness_elf_type=harness_elf_type,
        load_sections=[],
      )
      failures += 1
      continue
    if done_symbol is None:
      write_report(
        report_path,
        kernel=kernel,
        target=target,
        profile=profile,
        status="unknown",
        reason="missing symbol ct_binsec_done",
        artifacts=artifacts,
        binsec_version=binsec_version,
        timeout_seconds=kernel_timeout,
        smt_timeout_seconds=kernel_smt_timeout,
        smt_solver=args.smt_solver,
        rustflags=effective_rustflags,
        harness_elf_type=harness_elf_type,
        load_sections=[],
      )
      failures += 1
      continue
    if target.endswith("-unknown-linux-gnu") and harness_elf_type == "dyn":
      write_report(
        report_path,
        kernel=kernel,
        target=target,
        profile=profile,
        status="unknown",
        reason="BINSEC harness is PIE; static ELF proof binary required",
        artifacts=artifacts,
        binsec_version=binsec_version,
        timeout_seconds=kernel_timeout,
        smt_timeout_seconds=kernel_smt_timeout,
        smt_solver=args.smt_solver,
        rustflags=effective_rustflags,
        harness_elf_type=harness_elf_type,
        load_sections=[],
      )
      failures += 1
      continue

    driver = out_dir / "driver.elf"
    shutil.copy2(binary, driver)
    artifacts["driver.elf"] = sha256_file(driver)
    load_sections = binsec_load_sections(driver)

    disasm = out_dir / "driver.disasm"
    disassemble(driver, disasm)
    artifacts["driver.disasm"] = sha256_file(disasm)

    cfg = out_dir / "checkct.cfg"
    cfg.write_text(
      binsec_script(kernel, start_symbol=start_symbol, done_symbol=done_symbol, load_sections=load_sections)
    )
    artifacts["checkct.cfg"] = sha256_file(cfg)

    stats = out_dir / "binsec-stats.toml"
    log = out_dir / "binsec.log"
    binsec_cmd = [
      binsec,
      "-sse",
      "-checkct",
      "-sse-script",
      str(cfg),
      "-checkct-stats-file",
      str(stats),
      "-smt-solver",
      str(args.smt_solver),
      "-smt-timeout",
      str(kernel_smt_timeout),
      "-sse-timeout",
      str(kernel_timeout),
    ]
    if kernel.get("sse_depth") is not None:
      binsec_cmd.extend(["-sse-depth", str(kernel["sse_depth"])])
    binsec_cmd.append(str(driver))
    result = run(binsec_cmd, timeout=kernel_timeout + 60)
    log.write_text(result.stdout + result.stderr)
    artifacts["binsec.log"] = sha256_file(log)
    if stats.exists():
      artifacts["binsec-stats.toml"] = sha256_file(stats)

    status, reason = parse_status(result.stdout + result.stderr, result.returncode)
    write_report(
      report_path,
      kernel=kernel,
      target=target,
      profile=profile,
      status=status,
      reason=reason,
      artifacts=artifacts,
      binsec_version=binsec_version,
      timeout_seconds=kernel_timeout,
      smt_timeout_seconds=kernel_smt_timeout,
      smt_solver=args.smt_solver,
      rustflags=effective_rustflags,
      harness_elf_type=harness_elf_type,
      load_sections=load_sections,
    )
    if status != "secure" and kernel.get("required", False):
      failures += 1

  print(f"BINSEC artifacts written to {out_root}")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
