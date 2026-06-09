#!/usr/bin/env python3
"""Scan CT harness disassembly for high-signal constant-time hazards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BRANCH_CONDS_AARCH64 = {
  "b.eq",
  "b.ne",
  "b.cs",
  "b.hs",
  "b.cc",
  "b.lo",
  "b.mi",
  "b.pl",
  "b.vs",
  "b.vc",
  "b.hi",
  "b.ls",
  "b.ge",
  "b.lt",
  "b.gt",
  "b.le",
  "cbz",
  "cbnz",
  "tbz",
  "tbnz",
}
BRANCH_CONDS_X86 = {
  "ja",
  "jae",
  "jb",
  "jbe",
  "jc",
  "je",
  "jg",
  "jge",
  "jl",
  "jle",
  "jna",
  "jnae",
  "jnb",
  "jnbe",
  "jnc",
  "jne",
  "jng",
  "jnge",
  "jnl",
  "jnle",
  "jno",
  "jnp",
  "jns",
  "jnz",
  "jo",
  "jp",
  "jpe",
  "jpo",
  "js",
  "jz",
  "loop",
  "loope",
  "loopne",
}
DIV_MNEMONICS = {"div", "idiv", "udiv", "sdiv"}
INDIRECT_JUMP_MNEMONICS = {"br"}
INDIRECT_CALL_MNEMONICS = {"blr"}
SUSPICIOUS_CALL_TARGETS = (
  "panic",
  "panic_bounds_check",
  "__rust_alloc",
  "__rust_dealloc",
  "__rust_realloc",
  "__rust_alloc_error_handler",
  "alloc::alloc",
  "memcmp",
  "bcmp",
  "strcmp",
)


def load_toml(path: Path) -> dict[str, Any]:
  with path.open("rb") as fh:
    return tomllib.load(fh)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def normalize_symbol(symbol: str) -> str:
  return symbol[1:] if symbol.startswith("_ct_entry_") else symbol


def expected_symbols(root: Path) -> set[str]:
  ct = load_toml(root / "ct.toml")
  return {symbol for harness in ct.get("harness", []) for symbol in harness.get("symbols", [])}


def waivers(root: Path) -> list[dict[str, Any]]:
  ct = load_toml(root / "ct.toml")
  return list(ct.get("asm_waiver", []))


def waiver_matches(waiver: dict[str, Any], finding: dict[str, Any], target: str) -> bool:
  if waiver.get("target") not in (None, "*", target):
    return False
  if waiver.get("symbol") not in (None, "*", finding["symbol"]):
    return False
  if waiver.get("kind") not in (None, "*", finding["kind"]):
    return False
  if contains := waiver.get("instruction_contains"):
    if contains not in finding.get("text", ""):
      return False
  return True


def apply_waivers(findings: list[dict[str, Any]], configured: list[dict[str, Any]], target: str) -> None:
  for finding in findings:
    matched = [waiver for waiver in configured if waiver_matches(waiver, finding, target)]
    finding["waived"] = bool(matched)
    if matched:
      finding["waiver"] = {
        "reason": matched[0].get("reason"),
        "reviewed_by": matched[0].get("reviewed_by"),
      }


def parse_symbol_addresses(artifact_dir: Path) -> dict[str, int]:
  symbols: dict[str, int] = {}
  for path in sorted([*artifact_dir.glob("*.o.symbols.txt"), *artifact_dir.glob("*.obj.symbols.txt")]):
    for line in path.read_text(errors="replace").splitlines():
      if match := re.match(r"^([0-9a-fA-F]+)\s+\S+\s+(_?ct_entry_[A-Za-z0-9_]+)\b", line.strip()):
        symbols[normalize_symbol(match.group(2))] = int(match.group(1), 16)
  return symbols


def parse_asm_aliases(artifact_dir: Path) -> list[tuple[str, str]]:
  aliases: list[tuple[str, str]] = []
  alias_re = re.compile(r"^\s*(_?ct_entry_[A-Za-z0-9_]+)\s*=\s*(_?ct_entry_[A-Za-z0-9_]+)\s*$")
  for path in sorted(artifact_dir.glob("*.s")):
    for line in path.read_text(errors="replace").splitlines():
      if match := alias_re.match(line):
        aliases.append((normalize_symbol(match.group(1)), normalize_symbol(match.group(2))))
  return aliases


def apply_alias_bodies(functions: dict[str, list[tuple[int, str]]], aliases: list[tuple[str, str]]) -> None:
  parent: dict[str, str] = {}

  def find(symbol: str) -> str:
    parent.setdefault(symbol, symbol)
    while parent[symbol] != symbol:
      parent[symbol] = parent[parent[symbol]]
      symbol = parent[symbol]
    return symbol

  def union(left: str, right: str) -> None:
    left_root = find(left)
    right_root = find(right)
    if left_root != right_root:
      parent[right_root] = left_root

  for alias, target in aliases:
    union(alias, target)

  components: dict[str, list[str]] = {}
  for symbol in parent:
    components.setdefault(find(symbol), []).append(symbol)

  for symbols in components.values():
    body = next((functions[symbol] for symbol in symbols if symbol in functions), None)
    if body is None:
      continue
    for symbol in symbols:
      functions.setdefault(symbol, body)


def parse_disassembly(path: Path) -> tuple[dict[str, list[tuple[int, str]]], dict[int, list[list[tuple[int, str]]]]]:
  functions_by_label: dict[str, list[tuple[int, str]]] = {}
  functions_by_address: dict[int, list[list[tuple[int, str]]]] = {}
  current_lines: list[tuple[int, str]] | None = None
  label_re = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:$")
  for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
    if match := label_re.match(line.strip()):
      address_text = line.strip().split(maxsplit=1)[0]
      symbol = normalize_symbol(match.group(1))
      address = int(address_text, 16)
      current_lines = []
      functions_by_address.setdefault(address, []).append(current_lines)
      if symbol.startswith("ct_entry_"):
        functions_by_label.setdefault(symbol, current_lines)
      continue
    if current_lines is not None:
      current_lines.append((line_number, line))
  return functions_by_label, functions_by_address


def mnemonic(line: str) -> str | None:
  # llvm-objdump lines look like: "  6d14: b4000280     cbz x0, ..."
  if "\t" in line:
    fields = [field.strip() for field in line.split("\t") if field.strip()]
    if len(fields) >= 2:
      return fields[1].split(maxsplit=1)[0].lower()
  match = re.match(r"^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2,16}\s+)+([A-Za-z.][A-Za-z0-9_.]*)\b", line)
  if match:
    return match.group(1).lower()
  return None


def is_s390x_return(target: str, inst: str, line: str) -> bool:
  return target.startswith("s390x-") and inst == "br" and re.search(r"\bbr\s+%r14\b", line.lower()) is not None


def is_s390x_register_branch(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst == "br"


def is_indirect_jump(target: str, inst: str) -> bool:
  if inst not in INDIRECT_JUMP_MNEMONICS:
    return False
  if target.startswith("s390x-"):
    return False
  return True


def finding(
  symbol: str,
  path: Path,
  line_number: int,
  line: str,
  kind: str,
  severity: str,
  rationale: str,
) -> dict[str, Any]:
  return {
    "symbol": symbol,
    "file": f"artifacts/{path.name}",
    "line": line_number,
    "kind": kind,
    "severity": severity,
    "mnemonic": mnemonic(line),
    "text": line.strip(),
    "rationale": rationale,
    "waived": False,
  }


def scan_symbol(target: str, symbol: str, path: Path, lines: list[tuple[int, str]]) -> list[dict[str, Any]]:
  findings: list[dict[str, Any]] = []
  for line_number, line in lines:
    inst = mnemonic(line)
    if not inst:
      lowered = line.lower()
      if any(target in lowered for target in SUSPICIOUS_CALL_TARGETS):
        findings.append(
          finding(
            symbol,
            path,
            line_number,
            line,
            "suspicious_relocation_target",
            "warn",
            "Relocation or symbol text references a panic, allocation, or C comparison routine.",
          )
        )
      continue

    if inst in DIV_MNEMONICS:
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "variable_latency_division",
          "fail",
          "Division/remainder-family instructions are variable-latency on common targets and require explicit review.",
        )
      )
    elif is_s390x_return(target, inst, line):
      continue
    elif is_s390x_register_branch(target, inst):
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "register_branch",
          "warn",
          "s390x register branch found inside a CT harness symbol. "
          "It may be public jump-table control flow, but needs review.",
        )
      )
    elif is_indirect_jump(target, inst):
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "indirect_jump",
          "fail",
          "Indirect jump inside a CT harness symbol can hide secret-dependent control flow.",
        )
      )
    elif inst in BRANCH_CONDS_AARCH64 or inst in BRANCH_CONDS_X86:
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "conditional_branch",
          "warn",
          "Conditional branch found inside CT harness symbol. It may be public-shape control flow, but needs review.",
        )
      )
    elif inst in {"bl", "call", "callq"}:
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "call",
          "warn",
          "Call found inside CT harness symbol. Leaf extraction or callee review may be needed.",
        )
      )
    elif inst in INDIRECT_CALL_MNEMONICS:
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "indirect_call",
          "warn",
          "Indirect call found inside CT harness symbol. Usually backend dispatch, but dispatch source must be public.",
        )
      )
    elif re.search(r"\[[xrw][0-9]+,\s*[xrw][0-9]+", line) or re.search(r"\([^)]*,\s*%[a-z0-9]+", line):
      findings.append(
        finding(
          symbol,
          path,
          line_number,
          line,
          "register_indexed_memory",
          "warn",
          "Register-indexed memory access may be table lookup or public indexing; review operand provenance.",
        )
      )
  return findings


def summarize(
  symbols: set[str],
  functions: dict[str, list[tuple[int, str]]],
  findings: list[dict[str, Any]],
) -> dict[str, Any]:
  findings_by_symbol: dict[str, list[dict[str, Any]]] = {}
  for item in findings:
    findings_by_symbol.setdefault(item["symbol"], []).append(item)

  rows = {}
  for symbol in sorted(symbols):
    symbol_findings = findings_by_symbol.get(symbol, [])
    unwaived = [item for item in symbol_findings if not item.get("waived")]
    rows[symbol] = {
      "present": symbol in functions,
      "instruction_lines": len(functions.get(symbol, [])),
      "finding_count": len(symbol_findings),
      "unwaived_fail_count": sum(1 for item in unwaived if item["severity"] == "fail"),
      "unwaived_warn_count": sum(1 for item in unwaived if item["severity"] == "warn"),
    }
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", required=True)
  parser.add_argument("--artifact-dir", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  symbols = expected_symbols(root)
  configured_waivers = waivers(root)
  functions: dict[str, list[tuple[int, str]]] = {}
  findings: list[dict[str, Any]] = []
  disassembly_files = sorted([*args.artifact_dir.glob("*.o.disasm.txt"), *args.artifact_dir.glob("*.obj.disasm.txt")])
  symbol_addresses = parse_symbol_addresses(args.artifact_dir)
  aliases = parse_asm_aliases(args.artifact_dir)

  for path in disassembly_files:
    by_label, by_address = parse_disassembly(path)
    functions.update(by_label)
    for symbol in sorted(symbols):
      if symbol in functions:
        continue
      address = symbol_addresses.get(symbol)
      bodies = by_address.get(address, []) if address is not None else []
      if len(bodies) == 1:
        functions[symbol] = bodies[0]
    apply_alias_bodies(functions, aliases)

  for path in disassembly_files:
    for symbol in sorted(symbols):
      lines = functions.get(symbol)
      if lines:
        findings.extend(scan_symbol(args.target, symbol, path, lines))
    break

  apply_waivers(findings, configured_waivers, args.target)
  missing_symbols = sorted(symbol for symbol in symbols if symbol not in functions)
  unwaived_failures = [item for item in findings if item["severity"] == "fail" and not item.get("waived")]
  unwaived_warnings = [item for item in findings if item["severity"] == "warn" and not item.get("waived")]

  report = {
    "schema_version": 1,
    "kind": "rscrypto.ct.asm-heuristics",
    "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "target": args.target,
    "profile": args.profile,
    "policy": {
      "fail": ["variable_latency_division", "indirect_jump"],
      "warn": [
        "conditional_branch",
        "call",
        "indirect_call",
        "register_branch",
        "register_indexed_memory",
        "suspicious_relocation_target",
      ],
    },
    "disassembly_files": [
      {
        "path": f"artifacts/{path.name}",
        "sha256": sha256_file(path),
      }
      for path in disassembly_files
    ],
    "symbol_summary": summarize(symbols, functions, findings),
    "missing_symbols": missing_symbols,
    "finding_count": len(findings),
    "unwaived_fail_count": len(unwaived_failures),
    "unwaived_warn_count": len(unwaived_warnings),
    "waiver_count": sum(1 for item in findings if item.get("waived")),
    "findings": findings,
  }

  out = args.out_dir / "asm-heuristics.json"
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  if missing_symbols or unwaived_failures:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
