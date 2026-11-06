#!/usr/bin/env -S uv run python
"""Scan CT harness disassembly for high-signal constant-time hazards."""

from __future__ import annotations

import argparse
import sys
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linked_image import LinkedImage, load_linked_image
from provenance import load_toml, sha256_file
from symbolize_linked_binary import demangle_table

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
  "jcxz",
  "je",
  "jecxz",
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
  "jrcxz",
  "js",
  "jz",
  "loop",
  "loope",
  "loopne",
}
DIV_MNEMONICS = {"div", "idiv", "udiv", "sdiv"}
RISCV_MUL_MNEMONICS = {"mul", "mulh", "mulhsu", "mulhu", "mulw"}
RISCV_DIV_MNEMONICS = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
RISCV_BRANCH_CONDS = {
  "beq",
  "beqz",
  "bge",
  "bgez",
  "bgeu",
  "bgt",
  "bgtz",
  "bgtu",
  "ble",
  "bleu",
  "blez",
  "blt",
  "bltz",
  "bltu",
  "bne",
  "bnez",
  "c.beqz",
  "c.bnez",
}
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
DIRECT_CALL_MNEMONICS = {"bl", "brasl", "call", "callq", "jal"}
# Optimized wrappers often tail-call production code without a link instruction.
# Intra-function branches are removed by symbol identity when constructing edges.
DIRECT_TAIL_MNEMONICS = {"b", "ba", "j", "jg", "jmp", "jmpq", "c.j"}
INDIRECT_JUMP_MNEMONICS = {"br"}
INDIRECT_CALL_MNEMONICS = {"blr", "jalr"}
RESOLVED_RISCV_TRANSFERS = {"jalr", "c.jalr", "jr", "c.jr"}
# s390x condition-code suffixes for relative jumps and compare-and-branch forms.
S390X_CONDITION = r"(?:e|ne|h|nh|l|nl|lh|nlh|le|nle|he|nhe|o|no|p|np|m|nm|z|nz)"
S390X_CONDITIONAL_JUMP = re.compile(rf"^(?:j|jg){S390X_CONDITION}$")
S390X_COMPARE_AND_BRANCH = re.compile(rf"^cl?g?(?:r|i|f|fi|ih)?[jb]{S390X_CONDITION}$")
POWER_BRANCH_CONDS = {
  "bc",
  "bdnz",
  "bdnzf",
  "bdnzt",
  "bdz",
  "bdzf",
  "bdzt",
  "beq",
  "bf",
  "bge",
  "bgt",
  "ble",
  "blt",
  "bne",
  "bng",
  "bnl",
  "bns",
  "bnu",
  "bso",
  "bt",
  "bun",
}
CONTROL_FLOW_STOP_MNEMONICS = {
  "basr",
  "balr",
  "bcctr",
  "bcctrl",
  "bclr",
  "bctr",
  "bctrl",
  "bdnzlr",
  "bdzlr",
  "bflr",
  "blr",
  "br",
  "btlr",
  "brk",
  "c.ebreak",
  "c.jalr",
  "c.jr",
  "c.unimp",
  "ebreak",
  "int3",
  "jr",
  "ret",
  "retq",
  "trap",
  "ud2",
  "udf",
  "unimp",
}
# Longest straight-line arm of a select-shaped branch. On riscv64, LLVM's
# branchy lowering of ML-KEM's 32-byte implicit-rejection select spans 39
# instructions.
SELECT_BRANCH_MAX_ARM = 64
RELOCATION_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+(?:R_|IMAGE_REL_|ARM64_RELOC_|X86_64_RELOC_)")
INSTRUCTION_ADDRESS = re.compile(r"^\s*([0-9a-fA-F]+):")
BRANCH_TARGET_SYMBOL = re.compile(r"\s<(.+?)(?:\+0x([0-9a-fA-F]+))?>\s*$")
BRANCH_TARGET_ADDRESS = re.compile(r"\b0x([0-9a-fA-F]+)\s*$")
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
  "rscrypto::auth::mlkem::operations::s390x::",
  "rscrypto::auth::mlkem::operations::base_case",
  "rscrypto::auth::mlkem::operations::barrett",
  "rscrypto::auth::mlkem::operations::compress",
  "rscrypto::auth::mlkem::operations::decode_decompress",
  "rscrypto::auth::mlkem::operations::decompress",
  "rscrypto::auth::mlkem::operations::div_q_compress",
  "rscrypto::auth::mlkem::operations::inverse_ntt",
  "rscrypto::auth::mlkem::operations::matrix_accumulate",
  "rscrypto::auth::mlkem::operations::montgomery",
  "rscrypto::auth::mlkem::operations::mul_",
  "rscrypto::auth::mlkem::operations::multiply_ntts",
  "rscrypto::auth::mlkem::operations::ntt",
  "rscrypto::auth::mlkem::operations::poly_",
  "rscrypto::auth::mlkem::operations::subtract_compress",
)
WAIVER_REQUIRED_FIELDS = {
  "target",
  "primitive",
  "symbol",
  "kind",
  "slice_sha256",
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
  symbol_names: dict[str, str] = field(default_factory=dict, repr=False)


def normalize_symbol(symbol: str) -> str:
  return symbol[1:] if symbol.startswith(("_ct_entry_", "_rscrypto_", "_OUTLINED_FUNCTION_")) else symbol


def scoped_symbol(symbol: str, path: Path) -> str:
  # LLVM reuses outlined helper names in independent objects. Only the linked
  # artifact provides one namespace for them.
  if symbol.startswith("OUTLINED_FUNCTION_") and path.name.endswith((".o.disasm.txt", ".obj.disasm.txt")):
    return f"{path.name}::{symbol}"
  return symbol


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


def equality_release_roots(root: Path) -> set[str]:
  ct = load_toml(root / "ct.toml")
  release_binary = ct.get("equality_evidence", {}).get("release_binary", {})
  return set(release_binary.get("owner_symbols", [])) | set(release_binary.get("public_len_symbols", []))


def waivers(root: Path) -> list[dict[str, Any]]:
  ct = load_toml(root / "ct.toml")
  return list(ct.get("asm_waiver", []))


def select_branch_enforced_primitives(root: Path) -> set[str]:
  ct = load_toml(root / "ct.toml")
  return set(ct.get("asm_select_branch", {}).get("enforced_primitives", []))


def enforced_unreachable_symbols(root: Path) -> list[str]:
  ct = load_toml(root / "ct.toml")
  return list(ct.get("asm_select_branch", {}).get("unreachable_symbols", []))


def apply_select_branch_enforcement(findings: list[dict[str, Any]], enforced: set[str]) -> None:
  """Keep select-shaped branches blocking only where a primitive enforces them."""
  for finding in findings:
    if finding["kind"] != "select_branch" or enforced.intersection(finding.get("primitive_ids", [])):
      continue
    finding["severity"] = "warn"
    finding["disposition"] = "needs-binsec"


def public_operand_rules(root: Path) -> list[dict[str, Any]]:
  ct = load_toml(root / "ct.toml")
  return list(ct.get("asm_public_operand", []))


def apply_public_operand_rules(
  findings: list[dict[str, Any]],
  configured: list[dict[str, Any]],
  target: str | None = None,
) -> list[str]:
  """Accept findings whose operands a rule classifies as public.

  A rule with `targets` applies only to those targets, so its `max_count`
  bounds the findings of each listed target rather than of every target.
  """
  required = {"primitives", "roots", "symbol", "kind", "max_count", "source", "rationale"}
  errors: list[str] = []
  match_counts = [0] * len(configured)
  for index, rule in enumerate(configured):
    if not required <= set(rule) <= required | {"targets"}:
      errors.append(
        f"asm_public_operand[{index}] must contain exactly: {', '.join(sorted(required))}, and optionally targets"
      )
      continue
    if "targets" in rule and (
      not isinstance(rule["targets"], list)
      or not rule["targets"]
      or any(not isinstance(value, str) or not value.strip() for value in rule["targets"])
      or len(rule["targets"]) != len(set(rule["targets"]))
    ):
      errors.append(f"asm_public_operand[{index}] targets must be a non-empty list of unique strings")
      continue
    if any(
      not isinstance(rule.get(field), str) or not rule.get(field, "").strip()
      for field in required - {"max_count", "primitives", "roots"}
    ):
      errors.append(f"asm_public_operand[{index}] string fields must be non-empty")
    for field in ("primitives", "roots"):
      values = rule.get(field)
      if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(values) != len(set(values))
      ):
        errors.append(f"asm_public_operand[{index}] {field} must be a non-empty list of unique strings")
    max_count = rule.get("max_count")
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count <= 0:
      errors.append(f"asm_public_operand[{index}] max_count must be a positive integer")
    if rule.get("kind") not in {"variable_latency_division", "variable_latency_multiply"}:
      errors.append(f"asm_public_operand[{index}] kind requires an exact asm_waiver")

  if errors:
    return errors

  for finding in findings:
    matches = [
      index
      for index, rule in enumerate(configured)
      if rule["symbol"] == finding["symbol"]
      and rule["kind"] == finding["kind"]
      and ("targets" not in rule or target in rule["targets"])
      and set(rule["primitives"]).issubset(finding.get("primitive_ids", []))
      and set(finding.get("roots", [])).issubset(rule["roots"])
    ]
    if len(matches) > 1:
      errors.append(f"multiple asm_public_operand rules match {finding['locator']}")
      continue
    if not matches:
      continue
    index = matches[0]
    rule = configured[index]
    match_counts[index] += 1
    finding["operand_class"] = "public"
    finding["disposition"] = "accepted"
    finding["public_classification"] = {
      "primitives": rule["primitives"],
      "roots": rule["roots"],
      "source": rule["source"],
      "rationale": rule["rationale"],
    }

  for index, count in enumerate(match_counts):
    max_count = configured[index]["max_count"]
    if count > max_count:
      errors.append(f"asm_public_operand[{index}] matched {count} findings, exceeding max_count {max_count}")
  return errors


def validate_waivers(configured: list[dict[str, Any]]) -> list[str]:
  errors = []
  identities = set()
  for index, waiver in enumerate(configured):
    missing = sorted(WAIVER_REQUIRED_FIELDS - set(waiver))
    extra = sorted(set(waiver) - WAIVER_REQUIRED_FIELDS - {"contract"})
    if missing:
      errors.append(f"asm_waiver[{index}] missing field(s): {', '.join(missing)}")
    if extra:
      errors.append(f"asm_waiver[{index}] has unknown field(s): {', '.join(extra)}")
    for field in ("target", "primitive", "symbol", "kind", "slice_sha256"):
      if waiver.get(field) in (None, "", "*"):
        errors.append(f"asm_waiver[{index}] field {field} must be exact")
    if not re.fullmatch(r"[0-9a-f]{64}", str(waiver.get("slice_sha256", ""))):
      errors.append(f"asm_waiver[{index}] slice_sha256 must be a SHA-256 hex digest")
    classification = waiver.get("classification")
    if classification not in {"public", "outside-claim", "allowed-disclosure"}:
      errors.append(f"asm_waiver[{index}] classification must be public, outside-claim, or allowed-disclosure")
    if classification in {"outside-claim", "allowed-disclosure"} and not str(waiver.get("contract", "")).strip():
      errors.append(f"asm_waiver[{index}] requires the existing claim contract")
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
      waiver.get("slice_sha256") == finding.get("slice_sha256"),
    )
  )


SLICE_IDENTITY_VERSION = "rscrypto.ct.condition-slice.v1"
# Exit status of a check whose evidence is complete and whose findings fail the verdict. Other nonzero
# statuses mean the evidence itself is missing or inconsistent.
FINDINGS_EXIT = 3


def condition_identity(canonical: str, reasons: list[str], function_sha256: str) -> str:
  """The waiver identity of a finding: the backward slice of its condition.

  A closed slice ends only at function inputs, entry memory, immediates, read-only data, and the stores that
  can reach its loads, so code outside it cannot change the reviewed condition. An open slice also binds the
  whole function, because a call or an unknown effect can change what its leaves hold.
  """
  text = f"{SLICE_IDENTITY_VERSION}\n{canonical}"
  if reasons:
    text += f"\nfunction {function_sha256}"
  return hashlib.sha256(text.encode()).hexdigest()


def attach_condition_identities(
  target: str,
  findings: list[dict[str, Any]],
  functions: dict[str, FunctionBody],
  final_file: str,
  local_symbols: set[str],
  image: Any,
  kinds: set[str],
) -> None:
  """Give each linked-binary finding of a waivable kind its `slice_sha256`."""
  from condition_slice import FunctionSlices

  analyses: dict[str, Any] = {}
  for item in findings:
    if item["file"] != final_file or item["kind"] not in kinds:
      continue
    symbol = item["symbol"]
    if symbol not in analyses:
      analyses[symbol] = FunctionSlices(target, functions[symbol], image, local_symbols)
    result = analyses[symbol].slice(item["line"])
    item["slice_sha256"] = condition_identity(result.canonical, result.reasons, item["function_sha256"])
    item["slice_open"] = bool(result.reasons)


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
            **({"contract": waiver["contract"]} if "contract" in waiver else {}),
          }
        )

    accepted_primitives = {row["primitive"] for row in accepted}
    public_classification = finding.get("public_classification")
    public_primitives = (
      set(public_classification["primitives"])
      if isinstance(public_classification, dict)
      and isinstance(public_classification.get("primitives"), list)
      else set()
    )
    resolved_primitives = accepted_primitives | public_primitives
    unresolved = sorted(set(finding.get("primitive_ids", [])) - resolved_primitives)
    finding["waivers"] = accepted
    finding["unresolved_primitive_ids"] = unresolved
    finding["waived"] = not unresolved and bool(accepted)
    if not unresolved and resolved_primitives:
      # An existing disclosure or claim exclusion does not make an operand public.
      finding["operand_class"] = "public" if all(row["classification"] == "public" for row in accepted) else "unproven"
      finding["disposition"] = "accepted"
    elif unresolved and finding["disposition"] == "accepted":
      finding["operand_class"] = "unproven"
      finding["disposition"] = "needs-fix" if finding["severity"] == "fail" else "needs-binsec"

  return [
    f"asm_waiver[{index}] matches no finding: its reviewed condition changed or disappeared"
    for index, waiver in enumerate(configured)
    if waiver.get("target") == target and index not in used
  ]


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


def parse_disassembly(
  path: Path, symbol_names: dict[str, str] | None = None,
) -> tuple[dict[str, FunctionBody], dict[int, list[FunctionBody]]]:
  functions_by_label: dict[str, FunctionBody] = {}
  functions_by_address: dict[int, list[FunctionBody]] = {}
  current: FunctionBody | None = None
  label_re = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
  for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
    if line.startswith("Disassembly of section "):
      current = None
      continue
    if match := label_re.match(line.strip()):
      symbol = scoped_symbol(normalize_symbol(match.group(2)), path)
      if symbol.startswith((".", "LBB", "Ltmp")):
        if current is not None:
          current.lines.append((line_number, line))
        continue
      address = int(match.group(1), 16)
      current = FunctionBody(symbol=symbol, path=path, address=address, lines=[], symbol_names=symbol_names or {})
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


def direct_call_targets(line: str, symbol_names: dict[str, str] | None = None) -> set[str]:
  targets = set()
  if match := re.search(r"<(.+)>", line):
    targets.add(symbol_target_base(match.group(1)))
  targets.update(
    symbol_target_base(match.group(1))
    for match in re.finditer(
      r"\b(?:R_[A-Z0-9_]+|ARM64_RELOC_BRANCH26|X86_64_RELOC_BRANCH|IMAGE_REL_[A-Z0-9_]+)\s+(.+?)(?:\+0x[0-9a-fA-F]+)?\s*$",
      line,
    )
  )
  if symbol_names:
    targets = {normalize_symbol(symbol_names.get(target, target)) for target in targets}
  return {target for target in targets if target and not target.startswith(("Ltmp", "LBB"))}


def is_call_relocation(line: str) -> bool:
  return bool(re.search(r"\b(?:R_[A-Z0-9_]*(?:CALL(?:_PLT)?|PLT32)|ARM64_RELOC_BRANCH26|X86_64_RELOC_BRANCH)\b", line))


def all_direct_callees(body: FunctionBody) -> set[str]:
  callees: set[str] = set()
  for index, (_, line) in enumerate(body.lines):
    inst = mnemonic(line)
    if inst not in DIRECT_CALL_MNEMONICS | DIRECT_TAIL_MNEMONICS | RESOLVED_RISCV_TRANSFERS and not is_call_relocation(line):
      continue
    targets = direct_call_targets(line, body.symbol_names)
    if inst in DIRECT_CALL_MNEMONICS | DIRECT_TAIL_MNEMONICS and index + 1 < len(body.lines):
      next_line = body.lines[index + 1][1]
      # A relocation immediately following this instruction supplies its target.
      # Tail transfers use architecture-specific jump/relative relocations, not
      # necessarily the CALL relocations recognized on standalone lines.
      if is_call_relocation(next_line) or re.search(r"\b(?:R_|IMAGE_REL_)[A-Z0-9_]+\b", next_line):
        targets.update(direct_call_targets(next_line, body.symbol_names))
    callees.update(scoped_symbol(target, body.path) for target in targets)
  callees.discard(body.symbol)
  return callees


def direct_callees(body: FunctionBody, functions: dict[str, FunctionBody]) -> set[str]:
  return {
    target
    for target in all_direct_callees(body)
    if target in functions and is_ct_internal_symbol(target)
  }


def is_ct_internal_symbol(symbol: str) -> bool:
  return (
    "rscrypto::" in symbol or "::OUTLINED_FUNCTION_" in symbol
    or symbol.startswith(("rscrypto_", "ct_entry_", "OUTLINED_FUNCTION_"))
  )


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


def final_equality_closure(
  roots: set[str], functions: dict[str, FunctionBody], artifact: Path, target: str | None = None,
) -> dict[str, Any]:
  call_graph = build_call_graph(functions)
  by_root = {root: reachable_closure(root, call_graph) for root in sorted(roots) if root in functions}
  reachable = sorted({symbol for symbols in by_root.values() for symbol in symbols})
  terminal_calls: dict[str, list[str]] = {}
  unresolved_internal_calls = []
  for symbol in reachable:
    callees = all_direct_callees(functions[symbol])
    terminal = sorted(target for target in callees if not is_ct_internal_symbol(target))
    if terminal:
      terminal_calls[symbol] = terminal
    for callee in callees:
      if ((is_ct_internal_symbol(callee) and callee not in functions) or callee.startswith(".text")
          or (target == "powerpc64le-unknown-linux-gnu" and callee.startswith("."))):
        unresolved_internal_calls.append({"caller": symbol, "target": callee})
    if target == "x86_64-unknown-linux-gnu":
      for _, line in functions[symbol].lines:
        if re.search(r"\b(?:callq?|jmpq?)\s+\*[^\s]*\(%rip\)", line):
          targets = direct_call_targets(line)
          if (not re.search(r"# GOT 0x[0-9a-fA-F]+\s+<.+>\s*$", line)
              or not targets or any(name.startswith(".") for name in targets)):
            unresolved_internal_calls.append({"caller": symbol, "target": f"unresolved GOT transfer: {line.strip()}"})
  return {
    "artifact": f"artifacts/{artifact.name}",
    "artifact_sha256": sha256_file(artifact),
    "root_symbols": sorted(roots),
    "missing_root_symbols": sorted(roots - set(functions)),
    "reachable_symbols_by_root": {root: sorted(symbols) for root, symbols in by_root.items()},
    "reachable_symbol_count": len(reachable),
    "reachable_symbols": reachable,
    "terminal_calls_by_symbol": terminal_calls,
    "unresolved_internal_calls": unresolved_internal_calls,
  }


def unreachable_symbol_errors(
  roots: set[str], unreachable: list[str], functions: dict[str, FunctionBody],
) -> list[str]:
  """Prove that the linked enforced closure cannot reach the listed symbols.

  Each symbol must exist in the linked binary. A renamed or fully inlined
  symbol would otherwise satisfy the check without evidence.
  """
  call_graph = build_call_graph(functions)
  parents: dict[str, str | None] = {}
  queue = [root for root in sorted(roots) if root in functions]
  for root in queue:
    parents.setdefault(root, None)
  for symbol in queue:
    for callee in sorted(call_graph.get(symbol, set())):
      if callee not in parents:
        parents[callee] = symbol
        queue.append(callee)
  errors = []
  for symbol in unreachable:
    if symbol not in functions:
      errors.append(f"unreachable symbol {symbol} is absent from the linked binary")
    elif symbol in parents:
      path = [symbol]
      while (parent := parents[path[-1]]) is not None:
        path.append(parent)
      errors.append(f"enforced CT closure reaches {symbol}: {' -> '.join(reversed(path))}")
  return errors


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


def is_rsa_inverse_scope(primitive_ids: list[str], symbol: str) -> bool:
  if "rsa.private_ops" not in primitive_ids:
    return False
  return any(
    fragment in symbol
    for fragment in (
      "diag_rsa_blinding_factor_inverse_with_scratch",
      "private_modular_inverse_odd",
      "private_i31_",
    )
  )


def is_s390x_division(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_DIV_MNEMONICS


def is_s390x_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and inst in S390X_MUL_MNEMONICS


def is_s390x_vector_multiply(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and any(inst.startswith(prefix) for prefix in S390X_VECTOR_MUL_PREFIXES)


def is_riscv_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith(("riscv32", "riscv64")) and inst in RISCV_MUL_MNEMONICS


def is_riscv32_scalar_multiply(target: str, inst: str) -> bool:
  return target.startswith("riscv32") and inst in RISCV_MUL_MNEMONICS


def is_target_division_review(target: str, inst: str) -> bool:
  normalized = inst.rstrip(".")
  return (
    target.startswith(("riscv32", "riscv64"))
    and normalized in RISCV_DIV_MNEMONICS
    or target.startswith(("powerpc", "ppc"))
    and normalized in POWER_DIV_MNEMONICS
  )


def is_s390x_conditional_branch(target: str, inst: str) -> bool:
  return target.startswith("s390x-") and (
    inst in BRANCH_CONDS_S390X
    or S390X_CONDITIONAL_JUMP.match(inst) is not None
    or S390X_COMPARE_AND_BRANCH.match(inst) is not None
  )


def is_riscv_conditional_branch(target: str, inst: str) -> bool:
  return target.startswith(("riscv32", "riscv64")) and inst in RISCV_BRANCH_CONDS


def is_power_conditional_branch(target: str, inst: str) -> bool:
  return target.startswith(("powerpc", "ppc")) and inst.rstrip("+-") in POWER_BRANCH_CONDS


def is_conditional_branch(target: str, inst: str) -> bool:
  # s390x `jg` is an unconditional long jump, not x86 "jump if greater".
  if target.startswith("s390x-"):
    return is_s390x_conditional_branch(target, inst)
  return (
    inst in BRANCH_CONDS_AARCH64
    or inst in BRANCH_CONDS_X86
    or is_riscv_conditional_branch(target, inst)
    or is_power_conditional_branch(target, inst)
  )


@dataclass(frozen=True)
class Instruction:
  offset: int
  mnemonic: str
  local_target: int | None


def parse_instructions(body: FunctionBody) -> list[Instruction]:
  """Decode instruction lines with offsets and same-function branch targets.

  llvm-objdump names most branch targets as `<symbol+offset>`; s390x prints
  only the address. An address operand followed by a relocation is not local.
  """
  instructions = []
  lines = [line for _, line in body.lines]
  for index, line in enumerate(lines):
    address = INSTRUCTION_ADDRESS.match(line)
    inst = mnemonic(line)
    if address is None or inst is None or RELOCATION_LINE.match(line):
      continue
    local_target = None
    if symbol_target := BRANCH_TARGET_SYMBOL.search(line):
      if scoped_symbol(normalize_symbol(symbol_target.group(1)), body.path) == body.symbol:
        local_target = int(symbol_target.group(2) or "0", 16)
    elif address_target := BRANCH_TARGET_ADDRESS.search(line):
      relocated = index + 1 < len(lines) and RELOCATION_LINE.match(lines[index + 1]) is not None
      if not relocated:
        local_target = int(address_target.group(1), 16) - body.address
    instructions.append(Instruction(int(address.group(1), 16) - body.address, inst, local_target))
  return instructions


def select_branch_arm(
  target: str,
  instructions: list[Instruction],
  index_by_offset: dict[int, int],
  index: int,
) -> int | None:
  """Return the arm length when a conditional branch only chooses between data values.

  LLVM lowers a `select` to a short branch whose paths rejoin after straight-line
  code when the target lacks a usable conditional move, or when it judges the
  branch cheaper. The shapes are a skipped block, an arm that jumps to the
  branch target, an out-of-line arm that jumps back, and a two-arm diamond.
  """
  branch = instructions[index]
  if branch.local_target is None or index + 1 >= len(instructions):
    return None
  taken = index_by_offset.get(branch.local_target)
  fall = index + 1

  def is_stop(instruction: Instruction) -> bool:
    return (
      is_conditional_branch(target, instruction.mnemonic)
      or instruction.mnemonic in DIRECT_TAIL_MNEMONICS
      or instruction.mnemonic in DIRECT_CALL_MNEMONICS
      or instruction.mnemonic in INDIRECT_CALL_MNEMONICS
      or instruction.mnemonic in CONTROL_FLOW_STOP_MNEMONICS
    )

  def straight(start: int, end: int) -> bool:
    return 0 < end - start <= SELECT_BRANCH_MAX_ARM and not any(map(is_stop, instructions[start:end]))

  def arm_jump(start: int) -> tuple[int, int | None] | None:
    """Return the length and destination of a straight-line arm ending in an unconditional jump."""
    for length, instruction in enumerate(instructions[start : start + SELECT_BRANCH_MAX_ARM + 1]):
      if instruction.mnemonic in DIRECT_TAIL_MNEMONICS and not is_conditional_branch(target, instruction.mnemonic):
        return length, instruction.local_target
      if is_stop(instruction):
        return None
    return None

  if taken is not None and taken > fall and straight(fall, taken):
    return taken - fall
  fall_arm = arm_jump(fall)
  if fall_arm is not None and fall_arm[0] > 0 and fall_arm[1] == branch.local_target:
    return fall_arm[0]
  if taken is None:
    return None
  taken_arm = arm_jump(taken)
  if taken_arm is not None and taken_arm[0] > 0 and taken_arm[1] == instructions[fall].offset:
    return taken_arm[0]
  if fall_arm is None or fall_arm[1] is None:
    return None
  join = index_by_offset.get(fall_arm[1])
  if join is None:
    return None
  if taken_arm is not None and taken_arm[1] == fall_arm[1] and fall_arm[0] + taken_arm[0] > 0:
    return fall_arm[0] + taken_arm[0]
  if join > taken and straight(taken, join):
    return fall_arm[0] + join - taken
  return None


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


def instruction_identity(target: str, body: FunctionBody, line: str, local_symbols: set[str]) -> str:
  """Bind instruction semantics without binding resolved transfers to placement.

  Keep non-control operands and bytes exact, except the symbol name that the
  disassembler prints for an absolute immediate (`without_immediate_symbol`).
  For a decoded direct transfer to a known symbol, retain its length, mnemonic,
  registers, condition, and exact symbol/offset, but omit the encoding of its
  layout-dependent displacement. Unknown encodings and unresolved destinations
  stay byte-exact. `address_reference_identities` applies the same rule to
  resolved data and code address references in the linked binary.
  """
  address = INSTRUCTION_ADDRESS.match(line)
  if address is None:
    return line.strip()
  offset = int(address.group(1), 16) - body.address
  exact = f"{offset:x}: {line[address.end():].strip()}"
  inst = mnemonic(line)
  if inst not in DIRECT_CALL_MNEMONICS | DIRECT_TAIL_MNEMONICS and not is_conditional_branch(target, inst or ""):
    return without_immediate_symbol(target, inst, exact)
  fields = [part.strip() for part in line.split("\t") if part.strip()]
  if len(fields) < 2 or ":" not in fields[0]:
    return exact
  encoded = fields[0].split(":", 1)[1].split()
  if not encoded or any(not re.fullmatch(r"(?:[0-9a-fA-F]{2}){1,8}", word) for word in encoded):
    return exact
  text = " ".join(" ".join(fields[1:]).split())
  destination = re.search(r"0x[0-9a-fA-F]+\s+<(.+)>$", text)
  if destination is None:
    return exact
  symbol = normalize_symbol(destination.group(1))
  # Versioned and PLT names (`memcpy@plt`, `free@@GLIBC_2.17`) are local symbols in full.
  if symbol not in local_symbols and symbol_target_base(symbol) not in local_symbols:
    return exact
  length = sum(len(word) // 2 for word in encoded)
  return f"{offset:x}: {length}B {text[:destination.start()]}<{POWER_PLT_STUB_GROUP.sub('', symbol)}>"


IDENTITY_VERSION = "rscrypto.ct.instruction-identity.v2"
COMMENT = re.compile(r"\s+(?:#\s|//|;)")
SIGNED_IMMEDIATE = r"[-+]?(?:0x[0-9a-fA-F]+|\d+)"
# The `<symbol+offset>` that llvm-objdump appends to an absolute immediate, located by the immediate's syntax:
# x86-64 `# imm = 0xfffff`, AArch64 `#0xffff` or `// =0x10000`, and the operand of a RISC-V `lui`.
X86_IMMEDIATE_SYMBOL = re.compile(rf"(# imm = {SIGNED_IMMEDIATE})\s+<.*>$")
AARCH64_IMMEDIATE_SYMBOL = re.compile(rf"((?:#|// =){SIGNED_IMMEDIATE})\s+<.*>$")
RISCV_IMMEDIATE_SYMBOL = re.compile(rf"(,\s*{SIGNED_IMMEDIATE})\s+<.*>$")
RISCV_ABSOLUTE_IMMEDIATE_MNEMONICS = {"lui", "c.lui"}
# GNU ld names a POWER call stub `<group>.plt_call.<import>`; the group number records where the stub
# was placed. Artifact collection checks each stub's sequence, TOC slot, and relocation.
POWER_PLT_STUB_GROUP = re.compile(r"^[0-9a-f]{8}\.(?=plt_call\.)")


def without_immediate_symbol(target: str, inst: str | None, identity: str) -> str:
  """Drop the symbol name that llvm-objdump appends to an absolute immediate.

  llvm-objdump names the symbol whose address range contains an immediate's value, even when the value is a
  mask or a constant: `lui a1, 0x33333 <Projective::add+0x841>`. The name changes whenever unrelated code
  moves, and the encoding already binds the value, so it records placement, not semantics. PC-relative
  operands keep their names because those identify real targets.
  """
  if target.startswith("x86_64-"):
    pattern = X86_IMMEDIATE_SYMBOL
  elif target.startswith("aarch64-"):
    pattern = AARCH64_IMMEDIATE_SYMBOL
  elif target.startswith("riscv64") and inst in RISCV_ABSOLUTE_IMMEDIATE_MNEMONICS:
    pattern = RISCV_IMMEDIATE_SYMBOL
  else:
    return identity
  return pattern.sub(r"\1", identity)
X86_RIP_OPERAND = re.compile(rf"(?P<disp>{SIGNED_IMMEDIATE})?\(%rip\)")
AARCH64_REGISTER = re.compile(r"\b(?:[xw](\d+)|sp|wsp|xzr|wzr|[qdsbhv]\d+(?:\.\w+)?)\b")
AARCH64_ADDRESS = re.compile(r"^(?P<reg>[xw]\d+), 0x(?P<address>[0-9a-fA-F]+)(?: <.*>)?$")
AARCH64_ADD = re.compile(rf"^(?P<dst>[xw]\d+|sp), (?P<src>x\d+|sp), #(?P<imm>{SIGNED_IMMEDIATE})$")
AARCH64_MEMORY = re.compile(rf"\[(?P<base>x\d+|sp)(?:, #(?P<imm>{SIGNED_IMMEDIATE}))?\](?P<writeback>!?)")
RISCV_REGISTER = re.compile(r"\b(zero|ra|sp|gp|tp|fp|t[0-6]|s(?:1[01]|[0-9])|a[0-7]|f(?:t(?:1[01]|[0-9])|s(?:1[01]|[0-9])|a[0-7]))\b")
RISCV_ADDI = re.compile(rf"^(?P<dst>\w+), (?P<src>\w+), (?P<imm>{SIGNED_IMMEDIATE})$")
RISCV_MEMORY = re.compile(rf"(?P<imm>{SIGNED_IMMEDIATE})?\((?P<base>\w+)\)")
RISCV_STORES = {"sb", "sh", "sw", "sd", "fsw", "fsd", "c.sw", "c.sd", "c.fsd", "c.swsp", "c.sdsp"}
POWER_ADD_IMMEDIATE = re.compile(rf"^(?P<dst>\d+), (?P<src>\d+), (?P<imm>{SIGNED_IMMEDIATE})$")
POWER_DISPLACEMENT = re.compile(rf"^(?P<reg>\d+), (?P<imm>{SIGNED_IMMEDIATE})\((?P<base>\d+)\)$")
# D-form and DS-form loads and stores, with the bytes each accesses.
POWER_DISPLACEMENT_FORMS = {
  "lbz": 1, "lha": 2, "lhz": 2, "lwa": 4, "lwz": 4, "ld": 8, "lfs": 4, "lfd": 8, "lxsd": 8, "lxssp": 4, "lxv": 16,
  "stb": 1, "sth": 2, "stw": 4, "std": 8, "stfs": 4, "stfd": 8, "stxsd": 8, "stxssp": 4, "stxv": 16,
}
RISCV_ACCESS_WIDTH = {
  "lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4, "lwu": 4, "ld": 8, "flw": 4, "fld": 8,
  "c.lw": 4, "c.ld": 8, "c.flw": 4, "c.fld": 8,
}
S390X_RELATIVE_LONG = {
  "larl", "lgrl", "lrl", "lgfrl", "llgfrl", "lghrl", "llghrl", "lhrl", "llhrl", "stgrl", "strl", "sthrl",
  "cgrl", "crl", "clgrl", "clrl", "chrl", "cghrl", "clhrl", "clghrl", "cgfrl", "clgfrl", "exrl", "pfdrl",
}


@dataclass(frozen=True)
class DecodedLine:
  number: int
  offset: int
  address: int
  length: int
  mnemonic: str
  operands: str
  comment: str
  encoded: tuple[str, ...]


def decode_lines(body: FunctionBody) -> list[DecodedLine]:
  decoded = []
  for number, line in body.lines:
    address = INSTRUCTION_ADDRESS.match(line)
    if address is None or RELOCATION_LINE.match(line):
      continue
    fields = [part.strip() for part in line.split("\t") if part.strip()]
    if len(fields) < 2 or ":" not in fields[0]:
      continue
    encoded = fields[0].split(":", 1)[1].split()
    if not encoded or any(not re.fullmatch(r"(?:[0-9a-fA-F]{2}){1,8}", word) for word in encoded):
      continue
    text = " ".join(" ".join(fields[1:]).split())
    comment = ""
    if match := COMMENT.search(text):
      text, comment = text[: match.start()], text[match.end() :].strip()
    inst, _, operands = text.partition(" ")
    value = int(address.group(1), 16)
    decoded.append(DecodedLine(number, value - body.address, value, sum(len(word) // 2 for word in encoded),
                               inst.lower(), operands.strip(), comment, tuple(encoded)))
  return decoded


def reference_identity(row: DecodedLine, operands: str) -> str:
  return f"{row.offset:x}: {row.length}B {row.mnemonic} {operands}".rstrip()


def code_target_descriptor(image: LinkedImage, body: FunctionBody, target_address: int, descriptor: str) -> str:
  """Bind a reference into another function's interior to the referenced bytes."""
  function = image.function_at(target_address)
  if function is None or function[2] == body.symbol or function[0] == target_address:
    return descriptor
  contents = image.read(target_address, 16)
  return f"{descriptor}#{contents.hex()}" if contents is not None else descriptor


def describe_reference(image: LinkedImage, body: FunctionBody, target_address: int, width: int = 0) -> str | None:
  """Describe a resolved reference that reads `width` bytes, or 0 for an address that is only formed."""
  sink = getattr(image, "reference_sink", None)
  if sink is not None:
    sink.add(target_address)
    return "reference"
  descriptor = image.describe(target_address, minimum_end=target_address + width if width else 0)
  if descriptor is not None and descriptor.startswith("code:"):
    descriptor = code_target_descriptor(image, body, target_address, descriptor)
  return descriptor


def prepare_linked_image(target: str, functions: dict[str, FunctionBody], image: LinkedImage) -> None:
  """Give the image its function ranges and the member starts of its global-merge units.

  A member starts at every address that some function forms into a global-merge
  unit, so the image resolves the references of every function once before any
  descriptor is made.
  """
  image.set_functions(function_ranges(functions))
  if not image.merged_unit_count():
    return
  sink: set[int] = set()
  image.reference_sink = sink
  try:
    for body in functions.values():
      address_reference_identities(target, body, image)
  finally:
    image.reference_sink = None
  image.add_member_starts(sink)


def local_branch_targets(body: FunctionBody) -> set[int]:
  return {instruction.local_target for instruction in parse_instructions(body) if instruction.local_target is not None}


def x86_reference_identities(rows: list[DecodedLine], body: FunctionBody, image: LinkedImage) -> dict[int, str]:
  identities = {}
  for row in rows:
    matches = list(X86_RIP_OPERAND.finditer(row.operands))
    if len(matches) != 1:
      continue
    match = matches[0]
    target_address = row.address + row.length + (int(match.group("disp"), 0) if match.group("disp") else 0)
    annotated = re.match(r"(?:GOT\s+)?0x([0-9a-fA-F]+)", row.comment)
    if annotated is not None and int(annotated.group(1), 16) != target_address:
      continue
    if (descriptor := describe_reference(image, body, target_address)) is None:
      continue
    operands = f"{row.operands[: match.start()]}<{descriptor}>(%rip){row.operands[match.end() :]}"
    identities[row.number] = reference_identity(row, operands)
  return identities


def is_return_or_trap(target: str, row: DecodedLine) -> bool:
  """Whether control leaves the function or stops at `row` without a local successor."""
  mnemonic = row.mnemonic
  if target.startswith(("x86_64-", "i686-")):
    return mnemonic in {"ret", "retq", "retl", "ud2", "int3", "hlt"}
  if target.startswith("aarch64-"):
    return mnemonic in {"ret", "retaa", "retab", "brk", "udf", "hlt"}
  if target.startswith("riscv64"):
    return mnemonic in {"ret", "ebreak", "c.ebreak", "unimp", "c.unimp"} or (
      mnemonic in {"jr", "c.jr"} and re.fullmatch(r"(?:0x0\()?ra\)?", row.operands.strip()) is not None)
  if target.startswith(("powerpc64", "ppc64")):
    return mnemonic.rstrip("+-") in {"blr", "trap"}
  if target.startswith("s390x-"):
    return (mnemonic == "br" and row.operands.strip() == "%r14") or mnemonic == "j" and row.operands.strip() == "0x0"
  return False


def is_register_jump(target: str, row: DecodedLine) -> bool:
  """Whether `row` jumps to a register target, such as a jump table entry."""
  mnemonic = row.mnemonic
  if target.startswith(("x86_64-", "i686-")):
    return mnemonic in {"jmp", "jmpq"} and "*" in row.operands
  if target.startswith("aarch64-"):
    return mnemonic in {"br", "braa", "brab"}
  if target.startswith("riscv64"):
    return mnemonic in {"jr", "c.jr"} and not is_return_or_trap(target, row) and "<" not in row.operands
  if target.startswith(("powerpc64", "ppc64")):
    return mnemonic.rstrip("+-") == "bctr"
  if target.startswith("s390x-"):
    return mnemonic in {"br", "bcr"} and not is_return_or_trap(target, row)
  return False


def row_successors(target: str, body: FunctionBody, rows: list[DecodedLine]) -> list[list[int] | None]:
  """Return local successor row indices, or None for an unknown intra-function edge.

  Calls fall through. Returns, traps, and direct transfers out of the function
  have no local successors; an indirect jump has unknown successors.
  """
  local = {instruction.offset: instruction.local_target for instruction in parse_instructions(body)}
  index_by_offset = {row.offset: index for index, row in enumerate(rows)}
  successors: list[list[int] | None] = []
  for index, row in enumerate(rows):
    if is_return_or_trap(target, row):
      successors.append([])
      continue
    if is_register_jump(target, row):
      successors.append(None)
      continue
    destination = local.get(row.offset)
    conditional = is_conditional_branch(target, row.mnemonic)
    unconditional = row.mnemonic in DIRECT_TAIL_MNEMONICS and not conditional
    if not conditional and not unconditional:
      successors.append([index + 1])
      continue
    following = [] if unconditional else [index + 1]
    if destination is not None:
      if destination not in index_by_offset:
        successors.append(None)
        continue
      following.append(index_by_offset[destination])
    successors.append(following)
  return successors


def page_pair_identities(
  target: str,
  rows: list[DecodedLine],
  body: FunctionBody,
  image: LinkedImage,
  *,
  base: Any,
  consumer: Any,
  classify: Any,
  page_name: str,
  defer_merged: bool = False,
) -> dict[int, str]:
  """Resolve two-instruction address formation such as `adrp` or `auipc` pairs.

  `base(row)` returns the high-part register and address. `consumer(row,
  register, high)` returns `(address, render, formed, width)`: the low-part
  target, replacement operands, the register that now holds the full address
  (or None for an access), and the bytes an access reads (0 for an address).
  `classify(row, register, consumed)` returns "read", "dead", or None.

  Consumers count only in the straight-line code that the high part
  dominates. The high part is described by its resolved targets: its value is
  a placement-dependent public address, so a later read of the register, such
  as an argument register at a call, cannot depend on anything else.

  A low part that forms the start of a global-merge unit is a base for member
  references (`addi rX, rBase, imm`, `ld rX, imm(rBase)`). Each such reference
  is described by the member it designates. When every read of the base is
  such a reference, the base binds no contents; otherwise the base escapes and
  binds the whole unit. With `defer_merged`, the caller resolves member
  references by dataflow instead, and a base's identity holds the placeholder
  `merged-base@ADDRESS` for that caller to settle.
  """
  identities = {}
  labels = local_branch_targets(body)
  successors = row_successors(target, body, rows)

  def dead_on_all_paths(starts: list[int], register: str) -> bool:
    pending, seen = list(starts), set()
    while pending:
      index = pending.pop()
      if index in seen:
        continue
      if index >= len(rows) or len(seen) > 4096:
        return False
      seen.add(index)
      state = classify(rows[index], register, False)
      if state == "read":
        return False
      if state == "dead":
        continue
      following = successors[index]
      if following is None:
        return False
      pending.extend(following)
    return True

  def resolve(index: int, register: str, address: int, members: bool) -> tuple[list[str], bool]:
    """Describe the consumers of `register`, which holds `address`, after rows[index]."""
    targets: list[str] = []
    dead = False
    position = index + 1
    while position < len(rows):
      following = rows[position]
      if following.offset in labels:
        dead = dead_on_all_paths([position], register)
        break
      result = consumer(following, register, address)
      if result is not None:
        target_address, render, formed, width = result
        unit = image.merged_unit(target_address) if formed is not None and not members else None
        if unit is not None and unit[0] == target_address and defer_merged:
          descriptor = f"merged-base@{target_address:x}"
        elif unit is not None and unit[0] == target_address:
          derived, consumed = resolve(position, formed, target_address, True)
          if getattr(image, "reference_sink", None) is not None:
            descriptor = describe_reference(image, body, target_address)
          else:
            descriptor = "merged-base" if derived and consumed else image.describe_unit(target_address)
        else:
          descriptor = describe_reference(image, body, target_address, width)
        if descriptor is None:
          break
        identities[following.number] = reference_identity(following, render(descriptor))
        targets.append(descriptor)
      state = classify(following, register, result is not None)
      if state is not None:
        dead = state == "dead"
        break
      if successors[position] != [position + 1]:
        dead = successors[position] is not None and dead_on_all_paths(successors[position], register)
        break
      position += 1
    return targets, dead

  for index, row in enumerate(rows):
    high = base(row)
    if high is None:
      continue
    register, high_address = high
    targets, _ = resolve(index, register, high_address, False)
    if targets:
      identities[row.number] = reference_identity(row, f"{register}, <{page_name}:{','.join(targets)}>")
  return identities


def aarch64_access_width(row: DecodedLine) -> int:
  """The bytes a load or store at a resolved address accesses."""
  data = AARCH64_REGISTER.findall(row.operands.split("[", 1)[0])
  first = re.match(r"\s*([xwqdshb])\d", row.operands)
  width = {"x": 8, "w": 4, "q": 16, "d": 8, "s": 4, "h": 2, "b": 1}.get(first.group(1), 16) if first else 16
  if row.mnemonic in {"ldrb", "ldrsb", "strb"}:
    width = 1
  elif row.mnemonic in {"ldrh", "ldrsh", "strh"}:
    width = 2
  elif row.mnemonic == "ldrsw":
    width = 4
  return width * (2 if row.mnemonic.startswith(("ldp", "stp", "ldnp", "stnp")) and len(data) >= 2 else 1)


def aarch64_reference_identities(
  target: str, rows: list[DecodedLine], body: FunctionBody, image: LinkedImage,
) -> dict[int, str]:
  identities = {}
  for row in rows:
    match = AARCH64_ADDRESS.match(row.operands)
    if match is None or "[" in row.operands:
      continue
    if row.mnemonic == "adr" or row.mnemonic in {"ldr", "ldrsw", "prfm"}:
      if (descriptor := describe_reference(image, body, int(match.group("address"), 16))) is not None:
        identities[row.number] = reference_identity(row, f"{match.group('reg')}, <{descriptor}>")

  def base(row: DecodedLine) -> tuple[str, int] | None:
    match = AARCH64_ADDRESS.match(row.operands) if row.mnemonic == "adrp" else None
    return (match.group("reg"), int(match.group("address"), 16)) if match else None

  def number(register: str) -> str | None:
    return register[1:] if re.fullmatch(r"[xw]\d+", register) else None

  def registers(operands: str) -> list[str | None]:
    return [match.group(1) for match in AARCH64_REGISTER.finditer(operands)]

  def consumer(row: DecodedLine, register: str, page: int):
    if row.mnemonic == "add" and (match := AARCH64_ADD.match(row.operands)) and match.group("src") == register:
      return page + int(match.group("imm"), 0), lambda d: f"{match.group('dst')}, {register}, <{d}>", match.group("dst"), 0
    memory = [match for match in AARCH64_MEMORY.finditer(row.operands) if match.group("base") == register]
    if len(memory) != 1 or memory[0].group("writeback") or row.operands[memory[0].end() :].strip():
      return None
    data = row.operands[: memory[0].start()]
    stored = registers(data) if row.mnemonic.startswith("st") else registers(data)[2 if row.mnemonic.startswith("ldp") else 1:]
    if number(register) in stored:
      return None
    match = memory[0]
    return (page + (int(match.group("imm"), 0) if match.group("imm") else 0),
            lambda d: f"{row.operands[: match.start()]}[{register}, <{d}>]", None, aarch64_access_width(row))

  def classify(row: DecodedLine, register: str, consumed: bool) -> str | None:
    value = number(register)
    tokens = registers(row.operands)
    if row.mnemonic.startswith(("st", "cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "prfm", "b", "cb", "tb", "ret")):
      writes = 0
    elif row.mnemonic.startswith(("ldp", "ldnp", "ldxp", "ldaxp")):
      writes = 2
    else:
      writes = 1
    reads_destination = row.mnemonic in {"movk", "bfi", "bfxil", "bfm", "bfc"}
    written = value in tokens[:writes] and not reads_destination
    read = value in tokens[writes:] or (reads_destination and value in tokens[:writes])
    if read and not consumed:
      return "read"
    index = int(value)
    if row.mnemonic in {"bl", "blr", "blraa", "blrab"}:
      # x0-x8 carry arguments, x9-x18 are clobbered, and x19-x28 survive.
      return "read" if index <= 8 else ("dead" if index <= 18 else None)
    if row.mnemonic in {"ret", "retaa", "retab"}:
      return "read" if index <= 7 else "dead"
    if row.mnemonic in {"br", "braa", "brab"}:
      return "read"
    if row.mnemonic == "b" and re.search(rf"<_?{re.escape(body.symbol)}(?:\+|>)", row.operands) is None:
      # A tail transfer leaves the function; only argument registers stay live.
      return "read" if index <= 8 or index >= 19 else "dead"
    return "dead" if written else None

  identities.update(page_pair_identities(target, rows, body, image, base=base,
                                         consumer=consumer, classify=classify, page_name="page"))
  return identities


def riscv_reference_identities(
  target: str, rows: list[DecodedLine], body: FunctionBody, image: LinkedImage,
) -> dict[int, str]:
  def base(row: DecodedLine) -> tuple[str, int] | None:
    match = re.fullmatch(r"(\w+), 0x([0-9a-fA-F]+)", row.operands) if row.mnemonic == "auipc" else None
    if match is None:
      return None
    immediate = int(match.group(2), 16) & 0xFFFFF
    if immediate & 0x80000:
      immediate -= 0x100000
    return match.group(1), row.address + (immediate << 12)

  def without_annotation(operands: str) -> str:
    return re.sub(r"\s+<.*>$", "", operands)

  def consumer(row: DecodedLine, register: str, high: int):
    operands = without_annotation(row.operands)
    if row.mnemonic in {"addi", "c.addi"} and (match := RISCV_ADDI.match(operands)) and match.group("src") == register:
      return high + int(match.group("imm"), 0), lambda d: f"{match.group('dst')}, {register}, <{d}>", match.group("dst"), 0
    memory = [match for match in RISCV_MEMORY.finditer(operands) if match.group("base") == register]
    if len(memory) != 1 or operands[memory[0].end() :].strip():
      return None
    data = operands[: memory[0].start()]
    if row.mnemonic in RISCV_STORES and register in RISCV_REGISTER.findall(data):
      return None
    match = memory[0]
    return (high + (int(match.group("imm"), 0) if match.group("imm") else 0),
            lambda d: f"{data}<{d}>({register})", None, RISCV_ACCESS_WIDTH.get(row.mnemonic, 8))

  def classify(row: DecodedLine, register: str, consumed: bool) -> str | None:
    operands = without_annotation(row.operands)
    tokens = RISCV_REGISTER.findall(operands)
    indirect_form = re.match(rf"(?:{SIGNED_IMMEDIATE})?\(", operands) is not None
    if row.mnemonic in RISCV_STORES or row.mnemonic.startswith(("b", "c.b")) or row.mnemonic in {"j", "c.j", "jr", "c.jr", "ret"}:
      written, read = False, register in tokens
    elif row.mnemonic in {"jalr", "c.jalr"} and indirect_form:
      written, read = register == "ra", register in tokens
    else:
      written, read = bool(tokens) and tokens[0] == register, register in tokens[1:]
    if read and not consumed:
      return "read"
    is_call = row.mnemonic in {"jal", "c.jal"} or (row.mnemonic in {"jalr", "c.jalr"} and (indirect_form or tokens[:1] == ["ra"]))
    if is_call:
      if consumed or written:
        return "dead"
      # a0-a7 carry arguments, s-registers survive, and temporaries are clobbered.
      if re.fullmatch(r"a[0-7]|sp|gp|tp", register):
        return "read"
      return None if re.fullmatch(r"s\d+|fp", register) else "dead"
    if row.mnemonic in {"ret", "jr", "c.jr"}:
      if row.mnemonic != "ret" and tokens != ["ra"] and "<" not in row.operands:
        return "read"
      return "read" if register in {"a0", "a1"} else "dead"
    return "dead" if written else None

  return page_pair_identities(target, rows, body, image, base=base, consumer=consumer,
                              classify=classify, page_name="high")


def power_reference_identities(
  target: str, rows: list[DecodedLine], body: FunctionBody, image: LinkedImage,
) -> dict[int, str]:
  toc = image.toc_base
  if toc is None:
    return {}
  identities = {}
  if (len(rows) >= 2 and rows[0].offset == 0 and rows[1].offset == 4 and rows[0].mnemonic == "addis"
      and rows[1].mnemonic == "addi"):
    high = POWER_ADD_IMMEDIATE.match(rows[0].operands)
    low = POWER_ADD_IMMEDIATE.match(rows[1].operands)
    if (high and low and (high.group("dst"), high.group("src")) == ("2", "12")
        and (low.group("dst"), low.group("src")) == ("2", "2")
        and body.address + (int(high.group("imm"), 0) << 16) + int(low.group("imm"), 0) == toc):
      identities[rows[0].number] = reference_identity(rows[0], "2, 12, <toc-base>")
      identities[rows[1].number] = reference_identity(rows[1], "2, 2, <toc-base>")

  def consumer(row: DecodedLine, register: str, high: int):
    if row.mnemonic == "addi" and (match := POWER_ADD_IMMEDIATE.match(row.operands)) and match.group("src") == register:
      return high + int(match.group("imm"), 0), lambda d: f"{match.group('dst')}, {register}, <{d}>", match.group("dst"), 0
    if row.mnemonic in POWER_DISPLACEMENT_FORMS and (match := POWER_DISPLACEMENT.match(row.operands)) and match.group("base") == register:
      if row.mnemonic.startswith("st") and match.group("reg") == register:
        return None
      return (high + int(match.group("imm"), 0), lambda d: f"{match.group('reg')}, <{d}>({register})", None,
              POWER_DISPLACEMENT_FORMS[row.mnemonic])
    return None

  for row in rows:
    if row.number in identities:
      continue
    single = consumer(row, "2", toc)
    if single is None:
      continue
    unit = image.merged_unit(single[0]) if single[2] is not None else None
    if unit is not None and unit[0] == single[0]:
      identities[row.number] = reference_identity(row, single[1](f"merged-base@{single[0]:x}"))
    elif (descriptor := describe_reference(image, body, single[0], single[3])) is not None:
      identities[row.number] = reference_identity(row, single[1](descriptor))

  def base(row: DecodedLine) -> tuple[str, int] | None:
    high = POWER_ADD_IMMEDIATE.match(row.operands) if row.mnemonic == "addis" else None
    if high is None or high.group("src") != "2" or high.group("dst") == "2":
      return None
    return high.group("dst"), toc + (int(high.group("imm"), 0) << 16)

  def classify(row: DecodedLine, register: str, consumed: bool) -> str | None:
    # Bare register numbers cannot be told from immediates without an operand
    # table, so any other mention of the register counts as a read.
    numbers = re.findall(r"(?<![\w.-])\d+(?![\w.])", re.sub(r"\s+<.*>$", "", row.operands))
    destination = numbers[0] if numbers else None
    writes = not row.mnemonic.startswith(("st", "cmp", "b", "mt", "tw", "td", "dcb", "icb"))
    index = int(register)
    if row.mnemonic in {"bl", "bla"}:
      # r3-r10 carry arguments, r0 and r11-r12 are clobbered, and r14-r31 survive.
      return "read" if 3 <= index <= 10 else ("dead" if index in {0, 11, 12} else None)
    if row.mnemonic in {"blr", "bctr", "bctrl"}:
      return "read" if 3 <= index <= 10 or row.mnemonic != "blr" else "dead"
    if consumed:
      return "dead" if writes and destination == register else None
    if register in numbers[1 if writes else 0 :]:
      return "read"
    if writes and destination == register:
      return "dead"
    if row.mnemonic == "b" and re.search(rf"<{re.escape(body.symbol)}(?:\+|>)", row.operands) is None:
      return "read" if 3 <= index <= 10 or index >= 14 else "dead"
    return None

  identities.update(page_pair_identities(target, rows, body, image, base=base, consumer=consumer,
                                         classify=classify, page_name="toc-high", defer_merged=True))
  if image.merged_unit_count():
    identities.update(power_member_identities(target, rows, body, image, identities))
  return identities


# X-form loads, `T, RA, RB` with EA = (RA|0) + RB, and the bytes each reads.
POWER_INDEXED_LOADS = {
  "lbzx": 1, "lhzx": 2, "lhax": 2, "lwzx": 4, "lwax": 4, "ldx": 8, "lfsx": 4, "lfdx": 8, "lxsdx": 8,
  "lxsiwzx": 4, "lxsiwax": 4, "lxsspx": 4, "lxvd2x": 16, "lxvw4x": 16, "lxvx": 16, "lxvdsx": 8, "lvx": 16,
}
# D-form vector and floating-point loads, `T, D(RA)`.
POWER_VECTOR_DISPLACEMENT_LOADS = {"lxv": 16, "lxsd": 8, "lxssp": 4, "lfd": 8, "lfs": 4}
MERGED_BASE = re.compile(r"merged-base@([0-9a-f]+)")


# POWER instructions with implicit general-register operands: load and store multiple, string moves, `sc`.
POWER_IMPLICIT_REGISTERS = {"lmw", "stmw", "lswi", "lswx", "stswi", "stswx", "sc", "scv"}


def power_member_identities(
  target: str, rows: list[DecodedLine], body: FunctionBody, image: LinkedImage, identities: dict[int, str],
) -> dict[int, str]:
  """Describe POWER references to global-merge members by forward dataflow.

  Register values are `("T", address)` for a TOC-relative partial address (r2,
  `addis rX, 2, hi`), `("A", unit, address)` for an address inside a global-merge
  unit, and `("C", value, rows)` for a constant with the rows that defined it.
  Every instruction that can produce a unit's start is recognized: a TOC-relative
  `addi` or `ld` from a slot that targets the start, in any block. That start, a
  raw base, is a compiler artifact that can reach any member.

  A load or an address formed from an `A` value plus constant offsets designates
  one member, so it is described by that member; a constant used only as such an
  offset (`li 7, 20440` before `lxvd2x 1, 23, 7`) is merged-unit layout, so its
  immediate is dropped. An interior `A` value designates the member described
  where it was formed. A raw base escapes, and binds its whole unit, when it
  reaches any use other than a resolved access or a copy. Its spill to a stack
  slot and the reload keep the value, because a spill slot belongs to this
  function alone; a store to the frame at an unknown offset drops tracked slots
  and escapes their bases. An access through an `A` value with an unknown index
  also escapes its unit.
  Returns replacement identities, including the settled `merged-base` placeholders.
  """
  from condition_slice import power_effects

  toc = image.toc_base
  successors = row_successors(target, body, rows)
  value_forms = {"li", "lis", "addi", "addis", "mr", "add", "ld"}
  number = re.compile(r"(?<![\w.])-?\d+(?![\w.])")

  def gpr(text: str) -> str | None:
    return None if text == "0" else f"r{text}"

  def numbers_of(row: DecodedLine) -> list[str]:
    return number.findall(re.sub(r"\s+<.*>$", "", row.operands))

  def row_effect(row: DecodedLine) -> tuple[set[str], set[str], bool]:
    """(uses, defs, all registers) for one row."""
    effect = power_effects(row, identities.get(row.number, ""))
    if not effect.unknown:
      return set(effect.uses), set(effect.defs), False
    if row.mnemonic.rstrip(".") in POWER_IMPLICIT_REGISTERS:
      return set(), set(), True
    named = {f"r{text}" for text in numbers_of(row)}
    return named, named, False

  def raw(value: tuple | None) -> bool:
    return value is not None and value[0] == "A" and value[1] == value[2]

  def located(address: int) -> tuple | None:
    """The value of an address: a unit start or member, or None for anything else."""
    unit = image.merged_unit(address)
    return ("A", unit[0], address) if unit is not None else None

  def store_slot(row: DecodedLine) -> tuple[str, int, int] | None:
    """(stored register, r1 displacement, width) for a D-form store to the frame."""
    match = POWER_DISPLACEMENT.match(row.operands)
    if row.mnemonic in POWER_DISPLACEMENT_FORMS and row.mnemonic.startswith("st") and match and match.group("base") == "1":
      return f"r{match.group('reg')}", int(match.group("imm"), 0), POWER_DISPLACEMENT_FORMS[row.mnemonic]
    return None

  def frame_store(row: DecodedLine, numbers: list[str]) -> bool:
    """Whether a store other than a D-form spill can write the frame through r1."""
    if "(" in row.operands:
      match = re.search(r"\((\d+)\)", row.operands)
      return match is not None and match.group(1) == "1"
    return "1" in numbers[1:3]

  def produced(index: int, state: dict[str, tuple]) -> tuple[str, tuple] | None:
    """The tracked value that rows[index] writes, as (register, value)."""
    row = rows[index]
    numbers = numbers_of(row)
    plain = row.mnemonic
    if plain not in value_forms or not numbers:
      return None
    destination = f"r{numbers[0]}"
    if plain in {"li", "lis"} and len(numbers) == 2:
      return destination, ("C", int(numbers[1], 0) << (16 if plain == "lis" else 0), frozenset({index}))
    if plain in {"addi", "addis"} and len(numbers) == 3:
      source = state.get(gpr(numbers[1]) or "") if numbers[1] != "0" else ("C", 0, frozenset())
      immediate = int(numbers[2], 0) << (16 if plain == "addis" else 0)
      if source is None:
        return None
      if source[0] == "T":
        return (destination, ("T", source[1] + immediate)) if plain == "addis" else (
          (destination, value) if (value := located(source[1] + immediate)) else None)
      if source[0] == "A":
        return destination, ("A", source[1], source[2] + immediate)
      return destination, ("C", source[1] + immediate, source[2] | {index})
    if plain == "mr" and len(numbers) == 2:
      value = state.get(f"r{numbers[1]}")
      return (destination, value) if value is not None else None
    if plain == "add" and len(numbers) == 3:
      left, right = state.get(f"r{numbers[1]}"), state.get(f"r{numbers[2]}")
      if left and right and {left[0], right[0]} == {"A", "C"}:
        pointer, offset = (left, right) if left[0] == "A" else (right, left)
        return destination, ("A", pointer[1], pointer[2] + offset[1])
      if left and right and left[0] == right[0] == "C":
        return destination, ("C", left[1] + right[1], left[2] | right[2] | {index})
      return None
    if plain == "ld" and (match := POWER_DISPLACEMENT.match(row.operands)):
      source = state.get(f"r{match.group('base')}")
      if match.group("base") == "1" and (spilled := state.get(f"s{int(match.group('imm'), 0)}")) is not None:
        return destination, spilled
      if source is not None and source[0] == "T":
        slot = image.slots.get(source[1] + int(match.group("imm"), 0))
        if slot is not None and slot.address is not None and (value := located(slot.address)):
          return destination, value
    return None

  def transfer(index: int, state: dict[str, tuple]) -> tuple[dict[str, tuple], set[int]]:
    row = rows[index]
    after = dict(state)
    lost: set[int] = set()
    uses, defs, everything = row_effect(row)
    slots = {key: value for key, value in state.items() if key.startswith("s")}
    if everything:
      return {"r2": ("T", toc)}, {value[1] for value in state.values() if value[0] == "A"}
    value = produced(index, state)
    for register in defs:
      after.pop(register, None)
    if "r1" in defs:
      numbers = numbers_of(row)
      epilogue = (row.mnemonic == "addi" and numbers[:2] == ["1", "1"]) or (
        row.mnemonic == "ld" and row.operands.replace(" ", "") == "1,0(1)")
      for key, slot_value in slots.items():
        after.pop(key, None)
        if not epilogue:
          lost.add(slot_value[1])
    stored = store_slot(row)
    if stored is not None:
      register, displacement, width = stored
      for key in slots:
        offset = int(key[1:])
        if offset < displacement + width and displacement < offset + 8:
          after.pop(key, None)
      if width == 8 and raw(state.get(register)):
        after[f"s{displacement}"] = state[register]
    elif row.mnemonic.startswith("st") and slots and frame_store(row, numbers_of(row)):
      for key, slot_value in slots.items():
        after.pop(key, None)
        lost.add(slot_value[1])
    if value is not None:
      after[value[0]] = value[1]
    after["r2"] = ("T", toc)
    return after, lost

  if not rows or toc is None:
    return {}
  states: list[dict[str, tuple] | None] = [None] * len(rows)
  states[0] = {"r2": ("T", toc)}
  escaped: set[int] = set()
  pending = [0]
  while pending:
    index = pending.pop()
    after, lost = transfer(index, states[index] or {})
    escaped |= lost
    for successor in successors[index] or []:
      if successor >= len(rows):
        continue
      current = states[successor]
      merged = after if current is None else {
        key: value for key, value in current.items() if after.get(key) == value}
      if merged != current:
        states[successor] = merged
        pending.append(successor)

  replacements: dict[int, str] = {}
  index_uses: set[int] = set()
  other_uses: set[int] = set()
  for index, row in enumerate(rows):
    state = states[index]
    if state is None:
      continue
    uses, _defs, everything = row_effect(row)
    value = produced(index, state)
    numbers = numbers_of(row)
    plain = row.mnemonic
    if value is not None and raw(value[1]):
      # A copy or a spill reload keeps its exact identity. A TOC-relative formation of the base
      # describes it by a placeholder, also where it is not a straight-line pair.
      source = state.get(f"r{numbers[1]}") if plain == "addi" else (
        state.get(f"r{match.group('base')}") if plain == "ld" and (match := POWER_DISPLACEMENT.match(row.operands)) else None)
      if source is not None and source[0] == "T" and "merged-base@" not in identities.get(row.number, ""):
        placeholder = f"merged-base@{value[1][1]:x}"
        operands = (f"{numbers[0]}, {numbers[1]}, <{placeholder}>" if plain == "addi"
                    else f"{numbers[0]}, <{placeholder}>({numbers[-1]})")
        replacements[row.number] = reference_identity(row, operands)
      continue
    held = {key: tracked for key, tracked in state.items()
            if key != "r2" and not key.startswith("s") and (everything or key in uses)}
    pointers = {register: tracked for register, tracked in held.items() if tracked[0] == "A"}
    constants = {register: tracked for register, tracked in held.items() if tracked[0] == "C"}
    resolved: tuple[int, str, set[str], int] | None = None
    stored = store_slot(row)
    if not pointers:
      pass
    elif everything:
      escaped.update(tracked[1] for tracked in pointers.values())
    elif plain in POWER_INDEXED_LOADS and len(numbers) == 3:
      left, right = gpr(numbers[1]), gpr(numbers[2])
      pointer = next((register for register in (left, right) if register in pointers), None)
      other = right if pointer == left else left
      if pointer is not None and (other is None or other in constants):
        offset = constants[other][1] if other is not None else 0
        resolved = (pointers[pointer][2] + offset, f"{row.operands}, <{{}}>",
                    {other} if other is not None else set(), POWER_INDEXED_LOADS[plain])
      else:
        escaped.update(tracked[1] for tracked in pointers.values())
    elif (plain in POWER_DISPLACEMENT_FORMS and not plain.startswith("st")) or plain in POWER_VECTOR_DISPLACEMENT_LOADS:
      match = POWER_DISPLACEMENT.match(row.operands)
      base_register = f"r{match.group('base')}" if match else None
      if match and base_register in pointers and set(pointers) == {base_register}:
        width = POWER_DISPLACEMENT_FORMS.get(plain) or POWER_VECTOR_DISPLACEMENT_LOADS[plain]
        resolved = (pointers[base_register][2] + int(match.group("imm"), 0),
                    f"{match.group('reg')}, <{{}}>({match.group('base')})", set(), width)
      else:
        escaped.update(tracked[1] for tracked in pointers.values())
    elif stored is not None and set(pointers) == {stored[0]}:
      # Storing a pointer is not an access through it; a raw base may only be spilled whole.
      if raw(pointers[stored[0]]) and stored[2] != 8:
        escaped.add(pointers[stored[0]][1])
    elif plain == "addi" and len(numbers) == 3 and f"r{numbers[1]}" in pointers:
      resolved = (pointers[f"r{numbers[1]}"][2] + int(numbers[2], 0), f"{numbers[0]}, {numbers[1]}, <{{}}>", set(), 0)
    elif plain == "add" and len(numbers) == 3 and len(pointers) == 1 and len(constants) == 1:
      pointer = next(iter(pointers.values()))
      resolved = (pointer[2] + next(iter(constants.values()))[1], f"{row.operands}, <{{}}>", set(constants), 0)
    elif plain == "mr":
      pass
    else:
      # An interior pointer designates the member described where it was formed; a raw base does not.
      escaped.update(tracked[1] for tracked in pointers.values() if raw(tracked))
    if resolved is not None:
      address, render, used_constants, width = resolved
      descriptor = describe_reference(image, body, address, width)
      if descriptor is None:
        escaped.update(tracked[1] for tracked in pointers.values())
      else:
        replacements[row.number] = reference_identity(row, render.format(descriptor))
        for register in used_constants:
          index_uses.update(constants[register][2])
        for register, tracked in constants.items():
          if register not in used_constants:
            other_uses.update(tracked[2])
      continue
    if value is None or value[1][0] != "C":
      for tracked in constants.values():
        other_uses.update(tracked[2])

  for index in index_uses - other_uses:
    row = rows[index]
    numbers = numbers_of(row)
    if row.number not in replacements and numbers:
      replacements[row.number] = reference_identity(row, f"{', '.join(numbers[:-1])}, <member-offset>")

  for line, identity in {**identities, **replacements}.items():
    if "merged-base@" in identity:
      replacements[line] = MERGED_BASE.sub(
        lambda match: (image.describe_unit(int(match.group(1), 16)) or "merged-unresolved")
        if int(match.group(1), 16) in escaped else "merged-base", identity)
  return replacements


def s390x_reference_identities(rows: list[DecodedLine], body: FunctionBody, image: LinkedImage) -> dict[int, str]:
  identities = {}
  for row in rows:
    if row.mnemonic not in S390X_RELATIVE_LONG or row.length != 6:
      continue
    match = re.fullmatch(r"(%r\d+|\d+), 0x([0-9a-fA-F]+)(?: <.*>)?", row.operands)
    if match is None:
      continue
    displacement = int("".join(row.encoded)[4:12], 16)
    if displacement & 0x80000000:
      displacement -= 1 << 32
    target_address = int(match.group(2), 16)
    if row.address + 2 * displacement != target_address:
      continue
    if (descriptor := describe_reference(image, body, target_address)) is not None:
      identities[row.number] = reference_identity(row, f"{match.group(1)}, <{descriptor}>")
  return identities


def address_reference_identities(target: str, body: FunctionBody, image: LinkedImage | None) -> dict[int, str]:
  """Return placement-independent identities for resolved address references."""
  if image is None:
    return {}
  rows = decode_lines(body)
  if target.startswith(("x86_64-", "i686-")):
    return x86_reference_identities(rows, body, image)
  if target.startswith("aarch64-"):
    return aarch64_reference_identities(target, rows, body, image)
  if target.startswith("riscv64"):
    return riscv_reference_identities(target, rows, body, image)
  if target.startswith(("powerpc64", "ppc64")):
    return power_reference_identities(target, rows, body, image)
  if target.startswith("s390x-"):
    return s390x_reference_identities(rows, body, image)
  return {}


def function_ranges(functions: dict[str, FunctionBody]) -> list[tuple[int, int, str]]:
  ranges = {}
  for symbol, body in functions.items():
    rows = decode_lines(body)
    end = max((row.address + row.length for row in rows), default=body.address + 1)
    if body.address not in ranges or symbol < ranges[body.address][2]:
      ranges[body.address] = (body.address, max(end, body.address + 1), symbol)
  return sorted(ranges.values())


def instruction_identities(
  target: str, body: FunctionBody, local_symbols: set[str], image: LinkedImage | None = None,
) -> dict[int, str]:
  """Map each disassembly line number to its placement-independent identity."""
  references = address_reference_identities(target, body, image)
  return {
    number: references.get(number) or instruction_identity(target, body, line, local_symbols)
    for number, line in body.lines
  }


def function_identity(identities: dict[int, str]) -> str:
  return hashlib.sha256((IDENTITY_VERSION + "\n" + "\n".join(identities.values())).encode()).hexdigest()


def scan_symbol(
  target: str,
  symbol: str,
  body: FunctionBody,
  local_symbols: set[str],
  *,
  scope: str,
  primitive_ids: list[str],
  roots: list[str],
  image: LinkedImage | None = None,
) -> list[dict[str, Any]]:
  findings: list[dict[str, Any]] = []
  instructions = parse_instructions(body)
  index_by_offset = {instruction.offset: position for position, instruction in enumerate(instructions)}
  for line_index, (line_number, line) in enumerate(body.lines):
    inst = mnemonic(line)
    if not inst:
      callees = direct_call_targets(line, body.symbol_names)
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
      (is_ecdsa_scope(primitive_ids) or is_rsa_inverse_scope(primitive_ids, symbol))
      and is_riscv32_scalar_multiply(target, inst)
    ) or (is_ecdsa_scope(primitive_ids) and is_s390x_scalar_multiply(target, inst)):
      # ECDSA and the RSA blinding inverse multiply secret limbs in hardware on riscv64, and the
      # inverse on s390x (owner decisions, 2026-10-21); they fall through to the multiply review
      # below, as ECDH and the RSA exponentiation do. s390x ECDSA keeps the fixed-work multiplier:
      # secret MLGR products failed native DudeCT (P-384 ECDH, Constant-Time 39717348652).
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "variable_latency_multiply",
          "fail",
          "Scalar multiply is forbidden in this secret-arithmetic scope; use the fixed-work limb multiplier.",
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
    elif is_conditional_branch(target, inst):
      address = INSTRUCTION_ADDRESS.match(line)
      position = index_by_offset.get(int(address.group(1), 16) - body.address) if address else None
      arm = select_branch_arm(target, instructions, index_by_offset, position) if position is not None else None
      if arm is not None:
        findings.append(
          finding(
            symbol,
            body.path,
            line_number,
            line,
            "select_branch",
            "fail",
            f"Conditional branch whose paths rejoin after {arm} straight-line instruction(s): the shape of a "
            "masked selection lowered to control flow. Keep the selection mask opaque, or classify the "
            "condition with an exact reviewed asm_waiver.",
            scope=scope,
            primitive_ids=primitive_ids,
            roots=roots,
          )
        )
      else:
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
    elif inst in DIRECT_CALL_MNEMONICS | DIRECT_TAIL_MNEMONICS:
      callees = direct_call_targets(line)
      if inst in DIRECT_TAIL_MNEMONICS and callees == {symbol}:
        continue
      if callees and callees <= local_symbols and all(is_ct_internal_symbol(callee) for callee in callees):
        continue
      findings.append(
        finding(
          symbol,
          body.path,
          line_number,
          line,
          "call",
          "warn",
          "Unresolved direct call or tail transfer found inside CT evidence scope. Callee review may be needed.",
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
  identities = instruction_identities(target, body, local_symbols, image)
  function_sha256 = function_identity(identities)
  for item in findings:
    item["function_sha256"] = function_sha256
    if item["offset"] is not None:
      item["offset"] -= body.address
      offset = f"0x{item['offset']:x}"
    else:
      offset = f"line-{item['line']}"
    text_hash = hashlib.sha256(identities[item["line"]].encode()).hexdigest()[:16]
    item["locator"] = f"{symbol}+{offset}:{item['kind']}:{text_hash}"
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
    unresolved = [
      item for item in symbol_findings if not item.get("waived") and item["disposition"] != "accepted"
    ]
    rows[symbol] = {
      "present": symbol in functions,
      "instruction_lines": len(functions[symbol].lines) if symbol in functions else 0,
      "finding_count": len(symbol_findings),
      "unwaived_fail_count": sum(1 for item in unresolved if item["severity"] == "fail"),
      "unwaived_warn_count": sum(1 for item in unresolved if item["severity"] == "warn"),
      "needs_fix_count": sum(1 for item in unresolved if item["disposition"] == "needs-fix"),
      "needs_binsec_count": sum(1 for item in unresolved if item["disposition"] == "needs-binsec"),
      "accepted_count": sum(1 for item in symbol_findings if item["disposition"] == "accepted"),
    }
  return rows


def summarize_closure(
  closures: dict[str, dict[str, set[str]]],
  functions: dict[str, FunctionBody],
  findings: list[dict[str, Any]],
) -> dict[str, Any]:
  by_primitive: dict[str, dict[str, Any]] = {}
  for primitive_id, roots in sorted(closures.items()):
    reachable = sorted({symbol for symbols in roots.values() for symbol in symbols})
    primitive_findings = [item for item in findings if primitive_id in item.get("primitive_ids", [])]
    primitive_unresolved = [
      item
      for item in primitive_findings
      if not item.get("waived")
      and primitive_id in item.get("unresolved_primitive_ids", item.get("primitive_ids", []))
    ]
    by_primitive[primitive_id] = {
      "root_symbols": sorted(roots),
      "reachable_symbol_count": len(reachable),
      "reachable_symbols": reachable,
      "missing_root_symbols": sorted(root for root in roots if root not in functions),
      "finding_count": len(primitive_findings),
      "unwaived_fail_count": sum(1 for item in primitive_unresolved if item["severity"] == "fail"),
      "unwaived_warn_count": sum(1 for item in primitive_unresolved if item["severity"] == "warn"),
      "needs_fix_count": sum(1 for item in primitive_unresolved if item["disposition"] == "needs-fix"),
      "needs_binsec_count": sum(1 for item in primitive_unresolved if item["disposition"] == "needs-binsec"),
      "accepted_count": sum(
        1
        for item in primitive_findings
        if primitive_id not in item.get("unresolved_primitive_ids", item.get("primitive_ids", []))
      ),
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
  configured_public_operands = public_operand_rules(root)
  equality_roots = equality_release_roots(root)
  waiver_errors = validate_waivers(configured_waivers)
  if waiver_errors:
    for error in waiver_errors:
      print(error)
    return 2
  functions: dict[str, FunctionBody] = {}
  findings: list[dict[str, Any]] = []
  disassembly_files = sorted([*args.artifact_dir.glob("*.o.disasm.txt"), *args.artifact_dir.glob("*.obj.disasm.txt")])
  final_disassembly_files = sorted(args.artifact_dir.glob("*.binary.disasm.txt"))
  if len(final_disassembly_files) != 1:
    print(f"expected exactly one final binary disassembly, found {len(final_disassembly_files)}")
    return 2
  disassembly_files.extend(final_disassembly_files)
  symbol_addresses = parse_symbol_addresses(args.artifact_dir)
  symbol_names = demangle_table(args.artifact_dir)
  aliases = parse_asm_aliases(args.artifact_dir)

  for path in disassembly_files:
    by_label, by_address = parse_disassembly(path, symbol_names)
    functions.update(by_label)
    for symbol in sorted(symbols):
      if symbol in functions:
        continue
      address = symbol_addresses.get(symbol)
      bodies = by_address.get(address, []) if address is not None else []
      if len(bodies) == 1:
        functions[symbol] = bodies[0]
    apply_alias_bodies(functions, aliases)

  final_functions, _ = parse_disassembly(final_disassembly_files[0])
  final_equality = final_equality_closure(equality_roots, final_functions, final_disassembly_files[0], args.target)
  if final_equality["missing_root_symbols"] or final_equality["unresolved_internal_calls"]:
    for symbol in final_equality["missing_root_symbols"]:
      print(f"final equality binary is missing root {symbol}")
    for call in final_equality["unresolved_internal_calls"]:
      print(f"final equality closure cannot resolve {call['caller']} -> {call['target']}")
    return 2

  enforced_roots = {
    symbol for primitive in select_branch_enforced_primitives(root)
    for symbol in ct_roots_by_primitive.get(primitive, set())
  }
  final_enforced = final_equality_closure(enforced_roots, final_functions, final_disassembly_files[0], args.target)
  final_enforced["unreachable_symbols"] = enforced_unreachable_symbols(root)
  unreachable_errors = unreachable_symbol_errors(enforced_roots, final_enforced["unreachable_symbols"], final_functions)
  if final_enforced["missing_root_symbols"] or final_enforced["unresolved_internal_calls"] or unreachable_errors:
    for symbol in final_enforced["missing_root_symbols"]:
      print(f"enforced CT closure is missing linked root {symbol}")
    for call in final_enforced["unresolved_internal_calls"]:
      print(f"enforced CT closure cannot resolve {call['caller']} -> {call['target']}")
    for error in unreachable_errors:
      print(error)
    return 2

  image = load_linked_image(args.artifact_dir)
  if image is None:
    print("cannot read the linked binary for placement-independent instruction identities")
    return 2
  prepare_linked_image(args.target, final_functions, image)
  final_path = final_disassembly_files[0]

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
        image=image if body.path == final_path else None,
      )
    )

  apply_select_branch_enforcement(findings, select_branch_enforced_primitives(root))
  attach_condition_identities(args.target, findings, functions, f"artifacts/{final_path.name}", local_symbols, image,
                              {"select_branch"} | {waiver["kind"] for waiver in configured_waivers})
  try:
    public_operand_errors = apply_public_operand_rules(findings, configured_public_operands, args.target)
    if public_operand_errors:
      for error in public_operand_errors:
        print(error)
      return 2
    stale_waivers = apply_waivers(findings, configured_waivers, args.target)
  except ValueError as exc:
    print(f"invalid asm waiver set: {exc}")
    return 2
  # A stale waiver waives nothing; it fails the verdict without stopping evidence collection.
  for error in stale_waivers:
    print(error)
  missing_symbols = sorted(symbol for symbol in symbols if symbol not in functions)
  unwaived_failures = [
    item for item in findings if item["severity"] == "fail" and item["disposition"] != "accepted" and not item.get("waived")
  ]
  unwaived_warnings = [
    item
    for item in findings
    if item["severity"] == "warn" and item["disposition"] != "accepted" and not item.get("waived")
  ]
  needs_fix = [item for item in findings if item["disposition"] == "needs-fix" and not item.get("waived")]
  needs_binsec = [item for item in findings if item["disposition"] == "needs-binsec" and not item.get("waived")]
  accepted = [item for item in findings if item["disposition"] == "accepted"]
  unclassified = [
    item
    for item in findings
    if item.get("operand_class") not in {"public", "secret", "unproven"}
    or item.get("disposition") not in {"accepted", "needs-fix", "needs-binsec"}
  ]
  final_artifact = final_disassembly_files[0].name
  final_reachable = set(final_equality["reachable_symbols"])
  final_equality["terminal_call_sites"] = [
    {
      "symbol": item["symbol"],
      "locator": item["locator"],
      "text": item["text"],
      "kind": item["kind"],
      "function_sha256": item["function_sha256"],
    }
    for item in findings
    if Path(item["file"]).name == final_artifact
    and item["symbol"] in final_reachable
    and item["kind"] in {"call", "indirect_call"}
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
      "needs_fix_for_enforced_primitives": ["select_branch"],
      "select_branch_enforced_primitives": sorted(select_branch_enforced_primitives(root)),
      "needs_binsec": [
        "conditional_branch",
        "select_branch",
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
    "final_equality_call_closure": final_equality,
    "final_enforced_call_closure": final_enforced,
    "missing_symbols": missing_symbols,
    "finding_count": len(findings),
    "needs_fix_count": len(needs_fix),
    "needs_binsec_count": len(needs_binsec),
    "accepted_count": len(accepted),
    "unclassified_count": len(unclassified),
    "unwaived_fail_count": len(unwaived_failures),
    "unwaived_warn_count": len(unwaived_warnings),
    "waiver_count": sum(1 for item in findings if item.get("waived")),
    "stale_waivers": stale_waivers,
    "grouped_findings": grouped_findings(findings),
    "findings": findings,
  }

  out = args.out_dir / "asm-heuristics.json"
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  (args.out_dir / "asm-heuristics.md").write_text(markdown_report(report))
  if missing_symbols or needs_fix or unclassified or stale_waivers:
    return FINDINGS_EXIT
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
