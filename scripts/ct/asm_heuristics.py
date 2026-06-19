#!/usr/bin/env python3
"""Scan CT harness disassembly for high-signal constant-time hazards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
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
S390X_DIV_MNEMONICS = {
  "d",
  "dd",
  "ddb",
  "ddbr",
  "ddr",
  "de",
  "der",
  "dl",
  "dlg",
  "dlgr",
  "dlr",
  "dr",
  "dsg",
  "dsgf",
  "dsgfr",
  "dsgr",
  "dxbr",
  "dxr",
}
S390X_MUL_MNEMONICS = {
  "m",
  "mfy",
  "mh",
  "mhi",
  "mhy",
  "ml",
  "mlg",
  "mlgf",
  "mlgfr",
  "mlgr",
  "mlr",
  "mr",
  "ms",
  "msc",
  "msfi",
  "msg",
  "msgf",
  "msgfi",
  "msgfr",
  "msgr",
  "msr",
  "msy",
}
S390X_VECTOR_MUL_PREFIXES = ("vma", "vme", "vmh", "vml", "vmo")
BRANCH_CONDS_S390X = {
  "ber",
  "bher",
  "bhr",
  "bler",
  "blr",
  "bner",
  "bnher",
  "bnhr",
  "bnler",
  "bnlr",
  "bnor",
  "bor",
  "brc",
  "brcl",
  "brct",
  "brctg",
  "brxh",
  "brxhg",
  "brxle",
  "brxlg",
  "je",
  "jge",
  "jh",
  "jhe",
  "jl",
  "jle",
  "jne",
  "jno",
  "jnp",
  "jo",
  "jp",
}
DIRECT_CALL_MNEMONICS = {"bl", "brasl", "call", "callq"}
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
MLKEM_ARITHMETIC_SYMBOL_FRAGMENTS = (
  "rscrypto::auth::mlkem::portable::s390x::",
  "rscrypto::auth::mlkem::portable::base_case",
  "rscrypto::auth::mlkem::portable::barrett",
  "rscrypto::auth::mlkem::portable::compress",
  "rscrypto::auth::mlkem::portable::decode_decompress",
  "rscrypto::auth::mlkem::portable::decompress",
  "rscrypto::auth::mlkem::portable::div_q_compress",
  "rscrypto::auth::mlkem::portable::inverse_ntt",
  "rscrypto::auth::mlkem::portable::matrix_accumulate",
  "rscrypto::auth::mlkem::portable::montgomery",
  "rscrypto::auth::mlkem::portable::mul_",
  "rscrypto::auth::mlkem::portable::multiply_ntts",
  "rscrypto::auth::mlkem::portable::ntt",
  "rscrypto::auth::mlkem::portable::poly_",
  "rscrypto::auth::mlkem::portable::subtract_compress",
)


@dataclass
class FunctionBody:
  symbol: str
  path: Path
  address: int
  lines: list[tuple[int, str]]


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


def primitive_symbols(root: Path) -> dict[str, list[str]]:
  ct = load_toml(root / "ct.toml")
  symbols: dict[str, set[str]] = {}
  for primitive in ct.get("primitive", []):
    primitive_id = primitive.get("id")
    if not isinstance(primitive_id, str):
      continue
    for symbol in primitive.get("harness", {}).get("symbols", []):
      symbols.setdefault(symbol, set()).add(primitive_id)
  return {symbol: sorted(ids) for symbol, ids in symbols.items()}


def ct_intended_roots_by_primitive(root: Path) -> dict[str, set[str]]:
  ct = load_toml(root / "ct.toml")
  roots: dict[str, set[str]] = {}
  for primitive in ct.get("primitive", []):
    if primitive.get("claim") != "ct-intended":
      continue
    primitive_id = primitive.get("id")
    if not isinstance(primitive_id, str):
      continue
    symbols = set(primitive.get("harness", {}).get("symbols", []))
    if symbols:
      roots[primitive_id] = symbols
  return roots


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


def apply_alias_bodies(functions: dict[str, FunctionBody], aliases: list[tuple[str, str]]) -> None:
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


def parse_disassembly(path: Path) -> tuple[dict[str, FunctionBody], dict[int, list[FunctionBody]]]:
  functions_by_label: dict[str, FunctionBody] = {}
  functions_by_address: dict[int, list[FunctionBody]] = {}
  current: FunctionBody | None = None
  label_re = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
  for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
    if match := label_re.match(line.strip()):
      symbol = normalize_symbol(match.group(2))
      address = int(match.group(1), 16)
      current = FunctionBody(symbol=symbol, path=path, address=address, lines=[])
      functions_by_address.setdefault(address, []).append(current)
      functions_by_label.setdefault(symbol, current)
      continue
    if current is not None:
      current.lines.append((line_number, line))
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


def symbol_target_base(symbol: str) -> str:
  symbol = symbol.strip()
  symbol = symbol.split("+", 1)[0]
  symbol = symbol.split("@", 1)[0]
  return normalize_symbol(symbol)


def direct_call_targets(line: str) -> set[str]:
  targets = set()
  if match := re.search(r"<(.+)>", line):
    targets.add(symbol_target_base(match.group(1)))
  targets.update(
    symbol_target_base(match.group(1))
    for match in re.finditer(r"\bR_[A-Z0-9_]+\s+(.+?)(?:\+0x[0-9a-fA-F]+)?\s*$", line)
  )
  return {target for target in targets if target and not target.startswith((".", "Ltmp", "LBB"))}


def direct_callees(body: FunctionBody, functions: dict[str, FunctionBody]) -> set[str]:
  callees: set[str] = set()
  for index, (_, line) in enumerate(body.lines):
    if mnemonic(line) not in DIRECT_CALL_MNEMONICS:
      continue
    targets = direct_call_targets(line)
    if index + 1 < len(body.lines):
      targets.update(direct_call_targets(body.lines[index + 1][1]))
    callees.update(target for target in targets if target in functions)
  callees.discard(body.symbol)
  return callees


def build_call_graph(functions: dict[str, FunctionBody]) -> dict[str, set[str]]:
  return {symbol: direct_callees(body, functions) for symbol, body in functions.items()}


def reachable_closure(root_symbol: str, call_graph: dict[str, set[str]]) -> set[str]:
  seen: set[str] = set()
  stack = [root_symbol]
  while stack:
    symbol = stack.pop()
    if symbol in seen:
      continue
    seen.add(symbol)
    stack.extend(sorted(call_graph.get(symbol, set()) - seen, reverse=True))
  return seen


def closure_roots(
  roots_by_primitive: dict[str, set[str]],
  call_graph: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
  by_primitive: dict[str, dict[str, set[str]]] = {}
  for primitive_id, roots in roots_by_primitive.items():
    by_primitive[primitive_id] = {root: reachable_closure(root, call_graph) for root in sorted(roots)}
  return by_primitive


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


def is_mlkem_scope(primitive_ids: list[str]) -> bool:
  return any("mlkem" in primitive_id for primitive_id in primitive_ids)


def is_mlkem_arithmetic_symbol(symbol: str) -> bool:
  return symbol.startswith("ct_entry_mlkem") or any(fragment in symbol for fragment in MLKEM_ARITHMETIC_SYMBOL_FRAGMENTS)


def is_s390x_division(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_DIV_MNEMONICS


def is_s390x_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_MUL_MNEMONICS


def is_s390x_vector_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and any(inst.startswith(prefix) for prefix in S390X_VECTOR_MUL_PREFIXES)


def is_s390x_conditional_branch(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in BRANCH_CONDS_S390X


def finding(
  symbol: str,
  path: Path,
  line_number: int,
  line: str,
  kind: str,
  severity: str,
  rationale: str,
  *,
  scope: str,
  primitive_ids: list[str],
  roots: list[str],
) -> dict[str, Any]:
  return {
    "symbol": symbol,
    "scope": scope,
    "primitive_ids": primitive_ids,
    "roots": roots,
    "file": f"artifacts/{path.name}",
    "line": line_number,
    "kind": kind,
    "severity": severity,
    "mnemonic": mnemonic(line),
    "text": line.strip(),
    "rationale": rationale,
    "waived": False,
  }


def scan_symbol(
  target: str,
  symbol: str,
  body: FunctionBody,
  local_symbols: set[str],
  *,
  scope: str,
  primitive_ids: list[str],
  roots: list[str],
) -> list[dict[str, Any]]:
  findings: list[dict[str, Any]] = []
  for line_number, line in body.lines:
    inst = mnemonic(line)
    if not inst:
      lowered = line.lower()
      if any(target in lowered for target in SUSPICIOUS_CALL_TARGETS):
        findings.append(
          finding(
            symbol,
            body.path,
            line_number,
            line,
            "suspicious_relocation_target",
            "warn",
            "Relocation or symbol text references a panic, allocation, or C comparison routine.",
            scope=scope,
            primitive_ids=primitive_ids,
            roots=roots,
          )
        )
      continue

    if inst in DIV_MNEMONICS or (
      is_s390x_division(target, inst) and is_mlkem_scope(primitive_ids) and is_mlkem_arithmetic_symbol(symbol)
    ):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_division",
          "fail",
          "Division/remainder-family instructions are variable-latency on common targets and require explicit review.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif is_s390x_division(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "s390x_division_review",
          "warn",
          "s390x division-family instruction found in a CT evidence closure outside ML-KEM arithmetic. "
          "Review operand provenance before treating it as release evidence.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif is_mlkem_scope(primitive_ids) and is_mlkem_arithmetic_symbol(symbol) and is_s390x_scalar_multiply(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_multiply",
          "fail",
          "s390x multiply-family instruction found in an ML-KEM CT evidence scope. "
          "Secret-fed products must use fixed-work arithmetic unless this is proven public.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif is_mlkem_scope(primitive_ids) and is_mlkem_arithmetic_symbol(symbol) and is_s390x_vector_multiply(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "s390x_vector_multiply_review",
          "warn",
          "s390x vector multiply found in an ML-KEM CT evidence scope. This is allowed only for "
          "z/Vector-gated arithmetic with native IBM Z DudeCT coverage and no scalar multiply fallback.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif is_s390x_return(target, inst, line):
      continue
    elif is_s390x_register_branch(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "register_branch",
          "warn",
          "s390x register branch found inside a CT harness symbol. "
          "It may be public jump-table control flow, but needs review.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif is_indirect_jump(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "indirect_jump",
          "fail",
          "Indirect jump inside a CT harness symbol can hide secret-dependent control flow.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif inst in BRANCH_CONDS_AARCH64 or inst in BRANCH_CONDS_X86 or is_s390x_conditional_branch(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "conditional_branch",
          "warn",
          "Conditional branch found inside CT harness symbol. It may be public-shape control flow, but needs review.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif inst in DIRECT_CALL_MNEMONICS:
      callees = direct_call_targets(line)
      if callees and callees <= local_symbols:
        continue
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "call",
          "warn",
          "Unresolved direct call found inside CT evidence scope. Callee review may be needed.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif inst in INDIRECT_CALL_MNEMONICS:
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "indirect_call",
          "warn",
          "Indirect call found inside CT harness symbol. Usually backend dispatch, but dispatch source must be public.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
    elif re.search(r"\[[xrw][0-9]+,\s*[xrw][0-9]+", line) or re.search(r"\([^)]*,\s*%[a-z0-9]+", line):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "register_indexed_memory",
          "warn",
          "Register-indexed memory access may be table lookup or public indexing; review operand provenance.",
          scope=scope,
          primitive_ids=primitive_ids,
          roots=roots,
        )
      )
  return findings


def summarize(
  symbols: set[str],
  functions: dict[str, FunctionBody],
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
      "instruction_lines": len(functions[symbol].lines) if symbol in functions else 0,
      "finding_count": len(symbol_findings),
      "unwaived_fail_count": sum(1 for item in unwaived if item["severity"] == "fail"),
      "unwaived_warn_count": sum(1 for item in unwaived if item["severity"] == "warn"),
    }
  return rows


def summarize_closure(
  closures: dict[str, dict[str, set[str]]],
  functions: dict[str, FunctionBody],
  findings: list[dict[str, Any]],
) -> dict[str, Any]:
  unwaived = [item for item in findings if not item.get("waived")]
  by_primitive: dict[str, dict[str, Any]] = {}
  for primitive_id, roots in sorted(closures.items()):
    reachable = sorted({symbol for symbols in roots.values() for symbol in symbols})
    primitive_findings = [item for item in findings if primitive_id in item.get("primitive_ids", [])]
    primitive_unwaived = [item for item in unwaived if primitive_id in item.get("primitive_ids", [])]
    by_primitive[primitive_id] = {
      "root_symbols": sorted(roots),
      "reachable_symbol_count": len(reachable),
      "reachable_symbols": reachable,
      "missing_root_symbols": sorted(root for root in roots if root not in functions),
      "finding_count": len(primitive_findings),
      "unwaived_fail_count": sum(1 for item in primitive_unwaived if item["severity"] == "fail"),
      "unwaived_warn_count": sum(1 for item in primitive_unwaived if item["severity"] == "warn"),
    }
  return by_primitive


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", required=True)
  parser.add_argument("--artifact-dir", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  symbols = expected_symbols(root)
  symbol_primitives = primitive_symbols(root)
  ct_roots_by_primitive = ct_intended_roots_by_primitive(root)
  configured_waivers = waivers(root)
  functions: dict[str, FunctionBody] = {}
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

  call_graph = build_call_graph(functions)
  closures = closure_roots(ct_roots_by_primitive, call_graph)
  closure_symbols = {symbol for roots in closures.values() for symbols_for_root in roots.values() for symbol in symbols_for_root}
  root_symbols_by_symbol: dict[str, set[str]] = {}
  primitive_ids_by_symbol: dict[str, set[str]] = {symbol: set(ids) for symbol, ids in symbol_primitives.items()}
  for primitive_id, roots in closures.items():
    for root_symbol, reachable in roots.items():
      for symbol in reachable:
        root_symbols_by_symbol.setdefault(symbol, set()).add(root_symbol)
        primitive_ids_by_symbol.setdefault(symbol, set()).add(primitive_id)

  local_symbols = set(functions)
  for symbol in sorted(symbols | closure_symbols):
    body = functions.get(symbol)
    if body is None:
      continue
    scope = "entry" if symbol in symbols else "ct_intended_call_closure"
    findings.extend(
      scan_symbol(
        args.target,
        symbol,
        body,
        local_symbols,
        scope=scope,
        primitive_ids=sorted(primitive_ids_by_symbol.get(symbol, [])),
        roots=sorted(root_symbols_by_symbol.get(symbol, [])),
      )
    )

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
    "scan_scope": {
      "entry_symbols": "all manifest ct_entry_* symbols",
      "call_closure": "direct local call closure for ct-intended primitive harness roots",
      "non_ct_intended_primitives": "entry-symbol scan only",
    },
    "policy": {
      "fail": ["variable_latency_division", "variable_latency_multiply", "indirect_jump"],
      "warn": [
        "conditional_branch",
        "call",
        "indirect_call",
        "register_branch",
        "register_indexed_memory",
        "s390x_division_review",
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
    "ct_intended_call_closure": {
      "primitive_summary": summarize_closure(closures, functions, findings),
      "root_symbol_count": sum(len(roots) for roots in ct_roots_by_primitive.values()),
      "reachable_symbol_count": len(closure_symbols),
    },
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
