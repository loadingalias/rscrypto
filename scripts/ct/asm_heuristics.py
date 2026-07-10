#!/usr/bin/env python3
"""Scan CT harness disassembly for high-signal constant-time hazards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toml_compat import tomllib


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
RISCV_MUL_MNEMONICS = {"mul", "mulh", "mulhsu", "mulhu", "mulw"}
RISCV_DIV_MNEMONICS = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
POWER_DIV_MNEMONICS = {
  "divd",
  "divde",
  "divdeu",
  "divdu",
  "divw",
  "divwe",
  "divweu",
  "divwu",
  "modsd",
  "modsw",
  "modud",
  "moduw",
}
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
WAIVER_REQUIRED_FIELDS = {
  "target",
  "primitive",
  "symbol",
  "kind",
  "artifact",
  "locator",
  "function_sha256",
  "source",
  "classification",
  "rationale",
  "reviewer",
  "reviewed_at",
}


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


def validate_waivers(configured: list[dict[str, Any]]) -> list[str]:
  errors = []
  identities = set()
  for index, waiver in enumerate(configured):
    missing = sorted(WAIVER_REQUIRED_FIELDS - set(waiver))
    extra = sorted(set(waiver) - WAIVER_REQUIRED_FIELDS)
    if missing:
      errors.append(f"asm_waiver[{index}] missing field(s): {', '.join(missing)}")
    if extra:
      errors.append(f"asm_waiver[{index}] has unknown field(s): {', '.join(extra)}")
    for field in ("target", "primitive", "symbol", "kind", "artifact", "locator", "function_sha256"):
      if waiver.get(field) in (None, "", "*"):
        errors.append(f"asm_waiver[{index}] field {field} must be exact")
    if waiver.get("classification") != "public":
      errors.append(f"asm_waiver[{index}] classification must be 'public'; secret or uncertain findings cannot be waived")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(waiver.get("reviewed_at", ""))):
      errors.append(f"asm_waiver[{index}] reviewed_at must use YYYY-MM-DD")
    for field in ("source", "rationale", "reviewer"):
      if not str(waiver.get(field, "")).strip():
        errors.append(f"asm_waiver[{index}] field {field} must not be empty")
    identity = tuple(waiver.get(field) for field in sorted(WAIVER_REQUIRED_FIELDS))
    if identity in identities:
      errors.append(f"asm_waiver[{index}] duplicates an earlier waiver")
    identities.add(identity)
  return errors


def waiver_matches(waiver: dict[str, Any], finding: dict[str, Any], target: str, primitive_id: str) -> bool:
  return all(
    (
      waiver.get("target") == target,
      waiver.get("primitive") == primitive_id,
      waiver.get("symbol") == finding["symbol"],
      waiver.get("kind") == finding["kind"],
      waiver.get("artifact") == finding["file"],
      waiver.get("locator") == finding["locator"],
      waiver.get("function_sha256") == finding["function_sha256"],
    )
  )


def apply_waivers(findings: list[dict[str, Any]], configured: list[dict[str, Any]], target: str) -> list[str]:
  used = set()
  for finding in findings:
    accepted = []
    for primitive_id in finding.get("primitive_ids", []):
      matched = [
        (index, waiver)
        for index, waiver in enumerate(configured)
        if waiver_matches(waiver, finding, target, primitive_id)
      ]
      if len(matched) > 1:
        raise ValueError(f"multiple waivers match {primitive_id} {finding['locator']}")
      if matched:
        index, waiver = matched[0]
        used.add(index)
        accepted.append(
          {
            "primitive": primitive_id,
            "classification": waiver["classification"],
            "source": waiver["source"],
            "rationale": waiver["rationale"],
            "reviewer": waiver["reviewer"],
            "reviewed_at": waiver["reviewed_at"],
          }
        )

    accepted_primitives = {row["primitive"] for row in accepted}
    unresolved = sorted(set(finding.get("primitive_ids", [])) - accepted_primitives)
    finding["waivers"] = accepted
    finding["unresolved_primitive_ids"] = unresolved
    finding["waived"] = not unresolved and bool(accepted)
    if finding["waived"]:
      finding["operand_class"] = "public"
      finding["disposition"] = "accepted"

  return [f"asm_waiver[{index}] did not match generated disassembly" for index in range(len(configured)) if index not in used]


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
      if symbol.startswith((".", "LBB", "Ltmp")):
        if current is not None:
          current.lines.append((line_number, line))
        continue
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


def is_call_relocation(line: str) -> bool:
  return bool(re.search(r"\b(?:R_[A-Z0-9_]*(?:CALL|PLT32)|ARM64_RELOC_BRANCH26|X86_64_RELOC_BRANCH)\b", line))


def direct_callees(body: FunctionBody, functions: dict[str, FunctionBody]) -> set[str]:
  callees: set[str] = set()
  for index, (_, line) in enumerate(body.lines):
    inst = mnemonic(line)
    if inst not in DIRECT_CALL_MNEMONICS and not is_call_relocation(line):
      continue
    targets = direct_call_targets(line)
    if inst in DIRECT_CALL_MNEMONICS and index + 1 < len(body.lines):
      targets.update(direct_call_targets(body.lines[index + 1][1]))
    callees.update(
      target
      for target in targets
      if target in functions and target.startswith(("rscrypto::", "rscrypto_", "ct_entry_", "<rscrypto::"))
    )
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


def is_ecdsa_scope(primitive_ids: list[str]) -> bool:
  return any(primitive_id.startswith("signature.ecdsa_") for primitive_id in primitive_ids)


def is_s390x_division(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_DIV_MNEMONICS


def is_s390x_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_MUL_MNEMONICS


def is_s390x_vector_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and any(inst.startswith(prefix) for prefix in S390X_VECTOR_MUL_PREFIXES)


def is_riscv_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith(("riscv32", "riscv64")) and inst in RISCV_MUL_MNEMONICS


def is_target_division_review(target: str, inst: str) -> bool:
  normalized = inst.rstrip(".")
  return (
    target.startswith(("riscv32", "riscv64"))
    and normalized in RISCV_DIV_MNEMONICS
    or target.startswith(("powerpc", "ppc"))
    and normalized in POWER_DIV_MNEMONICS
  )


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
  address_match = re.match(r"^\s*([0-9a-fA-F]+):", line)
  offset = int(address_match.group(1), 16) if address_match else None
  normalized_text = line.strip()
  text_hash = hashlib.sha256(normalized_text.encode()).hexdigest()[:16]
  locator_offset = f"0x{offset:x}" if offset is not None else f"line-{line_number}"
  return {
    "symbol": symbol,
    "scope": scope,
    "primitive_ids": primitive_ids,
    "roots": roots,
    "file": f"artifacts/{path.name}",
    "line": line_number,
    "offset": offset,
    "locator": f"{symbol}+{locator_offset}:{kind}:{text_hash}",
    "function_sha256": None,
    "kind": kind,
    "severity": severity,
    "operand_class": "unproven",
    "disposition": "needs-fix" if severity == "fail" else "needs-binsec",
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
  for line_index, (line_number, line) in enumerate(body.lines):
    inst = mnemonic(line)
    if not inst:
      callees = direct_call_targets(line)
      if is_call_relocation(line) and callees and not callees <= local_symbols:
        findings.append(
          finding(
            symbol,
            body.path,
            line_number,
            line,
            "call",
            "warn",
            "Direct-call relocation leaves the local CT artifact; callee operand provenance is unproven.",
            scope=scope,
            primitive_ids=primitive_ids,
            roots=roots,
          )
        )
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
    elif (
      is_ecdsa_scope(primitive_ids)
      and (is_s390x_scalar_multiply(target, inst) or is_riscv_scalar_multiply(target, inst))
    ):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_multiply",
          "fail",
          "Scalar multiply is forbidden in s390x/RISC-V ECDSA secret arithmetic; use the fixed-work limb multiplier.",
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
    elif is_target_division_review(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_division_review",
          "warn",
          "RISC-V/POWER division or remainder has implementation-dependent latency. "
          "Operand provenance or binary proof is required.",
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
    elif is_s390x_scalar_multiply(target, inst) or is_riscv_scalar_multiply(target, inst):
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_multiply_review",
          "warn",
          "Scalar multiply in a s390x/RISC-V CT closure has implementation-dependent latency. "
          "Operand provenance or binary proof is required; native timing evidence alone is not proof.",
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
          "Register-indexed memory operand provenance is unproven.",
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
    elif inst == "jalr":
      relocation_window = [previous_line for _, previous_line in body.lines[max(0, line_index - 2) : line_index]]
      if any("R_RISCV_CALL" in previous_line for previous_line in relocation_window):
        continue
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "indirect_call",
          "warn",
          "RISC-V indirect call target provenance is unproven.",
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
  function_text = "\n".join(item.strip() for _, item in body.lines)
  function_sha256 = hashlib.sha256(function_text.encode()).hexdigest()
  for item in findings:
    item["function_sha256"] = function_sha256
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
      "needs_fix_count": sum(1 for item in unwaived if item["disposition"] == "needs-fix"),
      "needs_binsec_count": sum(1 for item in unwaived if item["disposition"] == "needs-binsec"),
      "accepted_count": sum(1 for item in symbol_findings if item["disposition"] == "accepted"),
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
      "needs_fix_count": sum(1 for item in primitive_unwaived if item["disposition"] == "needs-fix"),
      "needs_binsec_count": sum(1 for item in primitive_unwaived if item["disposition"] == "needs-binsec"),
      "accepted_count": sum(1 for item in primitive_findings if primitive_id not in item["unresolved_primitive_ids"]),
    }
  return by_primitive


def grouped_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
  groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
  for item in findings:
    for primitive_id in item.get("primitive_ids", []):
      key = (primitive_id, item["symbol"], item["kind"], item["file"])
      row = groups.setdefault(
        key,
        {
          "primitive": primitive_id,
          "symbol": item["symbol"],
          "kind": item["kind"],
          "artifact": item["file"],
          "finding_count": 0,
          "needs_fix_count": 0,
          "needs_binsec_count": 0,
          "accepted_count": 0,
        },
      )
      row["finding_count"] += 1
      if primitive_id not in item.get("unresolved_primitive_ids", []):
        row["accepted_count"] += 1
      elif item["disposition"] == "needs-fix":
        row["needs_fix_count"] += 1
      else:
        row["needs_binsec_count"] += 1
  kind_priority = {
    "register_indexed_memory": 0,
    "conditional_branch": 1,
    "register_branch": 1,
    "indirect_jump": 2,
    "indirect_call": 2,
    "call": 2,
  }
  ordered = sorted(
    groups,
    key=lambda key: (key[0], kind_priority.get(key[2], 3), key[1], key[2], key[3]),
  )
  return [groups[key] for key in ordered]


def markdown_report(report: dict[str, Any]) -> str:
  lines = [
    "# Constant-time assembly triage",
    "",
    f"- Target: `{report['target']}`",
    f"- Profile: `{report['profile']}`",
    f"- Unique findings: `{report['finding_count']}`",
    f"- Needs fix: `{report['needs_fix_count']}`",
    f"- Needs BINSEC/manual proof: `{report['needs_binsec_count']}`",
    f"- Accepted by exact waiver: `{report['accepted_count']}`",
    f"- Unclassified: `{report['unclassified_count']}`",
    "",
    "`needs-binsec` means operand provenance is unproven. It is not a constant-time waiver or a pass.",
    "Grouped counts are per primitive and may reference the same instruction from multiple primitive closures.",
    "",
    "## Grouped findings",
    "",
    "| Primitive | Reachable symbol | Kind | Artifact | Findings | Needs fix | Needs BINSEC | Accepted |",
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
  ]
  for row in report["grouped_findings"]:
    values = [
      row["primitive"],
      row["symbol"],
      row["kind"],
      row["artifact"],
      str(row["finding_count"]),
      str(row["needs_fix_count"]),
      str(row["needs_binsec_count"]),
      str(row["accepted_count"]),
    ]
    escaped = [value.replace("|", "\\|") for value in values]
    lines.append("| " + " | ".join(f"`{value}`" for value in escaped) + " |")

  needs_fix = [item for item in report["findings"] if not item["waived"] and item["disposition"] == "needs-fix"]
  if needs_fix:
    lines.extend(["", "## Blocking findings", ""])
    for item in needs_fix:
      lines.append(f"- `{item['locator']}` in `{item['file']}`: `{item['text'].replace('`', '')}`")
  lines.append("")
  return "\n".join(lines)


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
  waiver_errors = validate_waivers(configured_waivers)
  if waiver_errors:
    for error in waiver_errors:
      print(error)
    return 2
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

  try:
    waiver_match_errors = apply_waivers(findings, configured_waivers, args.target)
  except ValueError as exc:
    print(f"invalid asm waiver set: {exc}")
    return 2
  if waiver_match_errors:
    for error in waiver_match_errors:
      print(error)
    return 2
  missing_symbols = sorted(symbol for symbol in symbols if symbol not in functions)
  unwaived_failures = [item for item in findings if item["severity"] == "fail" and not item.get("waived")]
  unwaived_warnings = [item for item in findings if item["severity"] == "warn" and not item.get("waived")]
  needs_fix = [item for item in findings if item["disposition"] == "needs-fix" and not item.get("waived")]
  needs_binsec = [item for item in findings if item["disposition"] == "needs-binsec" and not item.get("waived")]
  accepted = [item for item in findings if item["disposition"] == "accepted"]
  unclassified = [
    item
    for item in findings
    if item.get("operand_class") not in {"public", "secret", "unproven"}
    or item.get("disposition") not in {"accepted", "needs-fix", "needs-binsec"}
  ]

  report = {
    "schema_version": 2,
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
      "needs_fix": ["variable_latency_division", "variable_latency_multiply", "indirect_jump"],
      "needs_binsec": [
        "conditional_branch",
        "call",
        "indirect_call",
        "register_branch",
        "register_indexed_memory",
        "s390x_division_review",
        "s390x_vector_multiply_review",
        "suspicious_relocation_target",
        "variable_latency_division_review",
        "variable_latency_multiply_review",
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
    "needs_fix_count": len(needs_fix),
    "needs_binsec_count": len(needs_binsec),
    "accepted_count": len(accepted),
    "unclassified_count": len(unclassified),
    "unwaived_fail_count": len(unwaived_failures),
    "unwaived_warn_count": len(unwaived_warnings),
    "waiver_count": sum(1 for item in findings if item.get("waived")),
    "grouped_findings": grouped_findings(findings),
    "findings": findings,
  }

  out = args.out_dir / "asm-heuristics.json"
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  (args.out_dir / "asm-heuristics.md").write_text(markdown_report(report))
  if missing_symbols or needs_fix or unclassified:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
