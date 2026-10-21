#!/usr/bin/env -S uv run python
"""Propose renewals of asm waivers whose branch condition is unchanged.

The gate binds each waiver to an exact function identity, so any change in a
waived function makes its waivers stale. For each stale waiver, this tool
compares the reviewed linked artifact with a new one. It proposes renewal only
when the new function has a select-shaped branch with the same branch identity
and the same backward slice of its condition, and when that slice ends only at
function inputs, immediates, or read-only data bound by contents. A slice with
a non-constant load, a call result, or an instruction of unknown effect needs
manual review, as does every changed slice and every enforced finding without
a waiver. The tool writes a review record; `--apply` renews only the proposals.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asm_heuristics as scanner
from condition_slice import condition_slice
from provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def linked_functions(artifact_dir: Path) -> tuple[dict[str, scanner.FunctionBody], Any, Path]:
  paths = sorted(artifact_dir.glob("*.binary.disasm.txt"))
  if len(paths) != 1:
    raise SystemExit(f"expected one linked disassembly in {artifact_dir}, found {len(paths)}")
  functions, _ = scanner.parse_disassembly(paths[0])
  image = scanner.load_linked_image(artifact_dir)
  if image is None:
    raise SystemExit(f"cannot read the linked binary in {artifact_dir}")
  image.set_functions(scanner.function_ranges(functions))
  return functions, image, paths[0]


def scan_new(target: str, artifact_dir: Path) -> tuple[list[dict[str, Any]], set[int]]:
  """Run the gate's scanner and return its findings and the indexes of matched waivers."""
  captured: dict[str, Any] = {}
  original = scanner.apply_waivers

  def recording(findings, configured, current_target):
    errors = original(findings, configured, current_target)
    captured["findings"] = findings
    captured["stale"] = {int(error.split("[", 1)[1].split("]", 1)[0]) for error in errors}
    return []

  scanner.apply_waivers = recording
  try:
    with tempfile.TemporaryDirectory() as out_dir:
      saved = sys.argv
      sys.argv = ["asm_heuristics.py", "--target", target, "--profile", "release",
                  "--artifact-dir", str(artifact_dir), "--out-dir", out_dir]
      try:
        status = scanner.main()
      finally:
        sys.argv = saved
  finally:
    scanner.apply_waivers = original
  if "findings" not in captured:
    raise SystemExit(f"the scanner stopped before waiver matching (status {status}); fix that failure first")
  return captured["findings"], captured["stale"]


def slice_record(target: str, functions: dict[str, scanner.FunctionBody], image: Any, symbol: str,
                 line: int) -> dict[str, Any]:
  result = condition_slice(target, functions[symbol], image, line, set(functions))
  return {"canonical": result.canonical, "reasons": result.reasons, "nodes": result.nodes}


def offset_of(finding: dict[str, Any]) -> int:
  return finding["offset"] if finding["offset"] is not None else -1


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--reviewed-artifacts", required=True, type=Path,
                      help="artifact directory that the current waivers were reviewed against")
  parser.add_argument("--artifacts", required=True, type=Path, help="new artifact directory")
  parser.add_argument("--out", required=True, type=Path, help="directory for the review record")
  parser.add_argument("--apply", action="store_true", help="renew the proposed waivers in ct.toml")
  parser.add_argument("--reviewed-at", default=datetime.now(UTC).strftime("%Y-%m-%d"))
  args = parser.parse_args()

  configured = scanner.waivers(ROOT)
  errors = scanner.validate_waivers(configured)
  if errors:
    for error in errors:
      print(error)
    return 2
  findings, stale_indexes = scan_new(args.target, args.artifacts)
  old_functions, old_image, old_disassembly = linked_functions(args.reviewed_artifacts)
  new_functions, new_image, new_disassembly = linked_functions(args.artifacts)
  enforced = scanner.select_branch_enforced_primitives(ROOT)

  rows: list[dict[str, Any]] = []
  stale_by_group: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
  for index, waiver in enumerate(configured):
    if waiver["target"] != args.target:
      continue
    row = {"index": index, "waiver": waiver, "decision": "current" if index not in stale_indexes else None,
           "reasons": []}
    rows.append(row)
    if index not in stale_indexes:
      continue
    body = old_functions.get(waiver["symbol"])
    reviewed = None
    if body is not None:
      for finding in scanner.scan_symbol(args.target, waiver["symbol"], body, set(old_functions), scope="renewal",
                                         primitive_ids=[waiver["primitive"]], roots=[], image=old_image):
        if (finding["locator"], finding["function_sha256"]) == (waiver["locator"], waiver["function_sha256"]):
          reviewed = finding
    if reviewed is None:
      row["decision"] = "manual"
      row["reasons"].append("the reviewed artifact does not contain this exact waived instruction")
      continue
    row["old"] = {"offset": offset_of(reviewed), "text": reviewed["text"],
                  "slice": slice_record(args.target, old_functions, old_image, waiver["symbol"], reviewed["line"])}
    if waiver["symbol"] not in new_functions:
      row["decision"] = "manual"
      row["reasons"].append("the function is absent from the new linked binary")
      continue
    stale_by_group[(waiver["primitive"], waiver["symbol"])].append(row)

  # A finding stays a candidate only for primitives that no current waiver already covers.
  waived = {(row["waiver"]["primitive"], row["waiver"]["locator"], row["waiver"]["function_sha256"])
            for row in rows if row["decision"] == "current"}
  slices: dict[int, dict[str, Any]] = {}
  proposed_findings: set[tuple[str, int]] = set()
  for (primitive, symbol), group in stale_by_group.items():
    candidates = [finding for finding in findings
                  if finding["symbol"] == symbol and finding["kind"] == group[0]["waiver"]["kind"]
                  and finding["file"] == group[0]["waiver"]["artifact"] and primitive in finding.get("primitive_ids", [])
                  and (primitive, finding["locator"], finding["function_sha256"]) not in waived]
    by_signature: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for candidate in sorted(candidates, key=offset_of):
      key = id(candidate)
      if key not in slices:
        slices[key] = slice_record(args.target, new_functions, new_image, symbol, candidate["line"])
      by_signature[slices[key]["canonical"]].append(candidate)
    old_by_signature: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in sorted(group, key=lambda item: item["old"]["offset"]):
      old_by_signature[row["old"]["slice"]["canonical"]].append(row)
    for signature, old_rows in old_by_signature.items():
      matches = by_signature.get(signature, [])
      for position, row in enumerate(old_rows):
        if not matches:
          row["decision"] = "manual"
          row["reasons"].append("no new select-shaped branch has the same branch and condition slice")
          continue
        if len(matches) != len(old_rows):
          row["decision"] = "manual"
          row["reasons"].append(f"{len(old_rows)} reviewed and {len(matches)} new branches share this slice")
          continue
        candidate = matches[position]
        row["new"] = {"offset": offset_of(candidate), "locator": candidate["locator"],
                      "function_sha256": candidate["function_sha256"], "text": candidate["text"],
                      "slice": slices[id(candidate)]}
        reasons = sorted(set(row["old"]["slice"]["reasons"]) | set(slices[id(candidate)]["reasons"]))
        if reasons:
          # The slice is unchanged, but a leaf needs a reviewer to confirm it is public.
          row["decision"] = "manual"
          row["slice_unchanged"] = True
          row["reasons"].extend(reasons)
        else:
          row["decision"] = "propose"
          proposed_findings.add((primitive, id(candidate)))
    for row in group:
      if row["decision"] is None:
        row["decision"] = "manual"
        row["reasons"].append("unmatched")
      if row["decision"] == "manual" and "new" not in row:
        row["candidates"] = [{"offset": offset_of(candidate), "locator": candidate["locator"], "text": candidate["text"],
                              "slice": slices[id(candidate)]} for candidate in sorted(candidates, key=offset_of)]

  unresolved = []
  for finding in findings:
    if finding["kind"] != "select_branch":
      continue
    missing = [primitive for primitive in finding.get("unresolved_primitive_ids", [])
               if primitive in enforced and (primitive, id(finding)) not in proposed_findings]
    if missing:
      unresolved.append({"symbol": finding["symbol"], "offset": offset_of(finding), "locator": finding["locator"],
                         "function_sha256": finding["function_sha256"], "primitives": missing,
                         "text": finding["text"],
                         "slice": slices.get(id(finding)) or slice_record(args.target, new_functions, new_image,
                                                                          finding["symbol"], finding["line"])})

  counts = collections.Counter(row["decision"] for row in rows)
  counts["manual_with_unchanged_slice"] = sum(1 for row in rows if row.get("slice_unchanged"))
  record = {
    "kind": "rscrypto.ct.waiver-renewal",
    "schema_version": 1,
    "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "target": args.target,
    "identity": scanner.IDENTITY_VERSION,
    "ct_manifest_sha256": sha256_file(ROOT / "ct.toml"),
    "reviewed_disassembly": {"path": str(old_disassembly), "sha256": sha256_file(old_disassembly)},
    "new_disassembly": {"path": str(new_disassembly), "sha256": sha256_file(new_disassembly)},
    "counts": dict(counts),
    "enforced_findings_needing_review": len(unresolved),
    "waivers": rows,
    "unwaived_enforced_findings": unresolved,
  }
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "renewal-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  (args.out / "renewal-record.md").write_text(markdown(record))
  print(f"{args.target}: {dict(counts)}; {len(unresolved)} enforced findings need review")
  if args.apply:
    renewed = apply_renewals(ROOT / "ct.toml", [row for row in rows if row["decision"] == "propose"], args.reviewed_at)
    print(f"renewed {renewed} waivers in ct.toml")
  return 0


def markdown(record: dict[str, Any]) -> str:
  lines = [
    f"# Waiver renewal record: `{record['target']}`",
    "",
    f"- Generated: {record['generated_at_utc']}",
    f"- Identity: `{record['identity']}`",
    f"- Reviewed disassembly SHA-256: `{record['reviewed_disassembly']['sha256']}`",
    f"- New disassembly SHA-256: `{record['new_disassembly']['sha256']}`",
    f"- Decisions: {', '.join(f'{name} {count}' for name, count in sorted(record['counts'].items()))}",
    f"- Enforced findings without a waiver or proposal: {record['enforced_findings_needing_review']}",
    "",
    "A proposal means that the branch and the backward slice of its condition are unchanged and that",
    "the slice ends only at function inputs, immediates, or read-only data bound by contents.",
    "It does not classify the branch. Every manual item needs a review of the new artifact.",
    "",
  ]
  sections = (
    ("Manual: changed or unmatched slice", lambda row: row["decision"] == "manual" and not row.get("slice_unchanged")),
    ("Manual: unchanged slice with a leaf to confirm", lambda row: row.get("slice_unchanged")),
    ("Proposed", lambda row: row["decision"] == "propose"),
  )
  for title, predicate in sections:
    selected = [row for row in record["waivers"] if predicate(row)]
    if not selected:
      continue
    lines.extend([f"## {title} ({len(selected)})", ""])
    for row in selected:
      waiver = row["waiver"]
      lines.append(f"### `{waiver['primitive']}` `{waiver['symbol']}`")
      lines.append("")
      lines.append(f"- Reviewed: `{waiver['locator']}` ({waiver['classification']})")
      if "new" in row:
        lines.append(f"- New: `{row['new']['locator']}`")
      for reason in row["reasons"]:
        lines.append(f"- {reason}")
      if "old" in row:
        lines.extend(["", "Reviewed slice:", "", "```text", row["old"]["slice"]["canonical"], "```"])
      if "new" in row and row["new"]["slice"]["canonical"] != row.get("old", {}).get("slice", {}).get("canonical"):
        lines.extend(["", "New slice:", "", "```text", row["new"]["slice"]["canonical"], "```"])
      lines.append("")
  if record["unwaived_enforced_findings"]:
    lines.extend([f"## Enforced findings without a waiver ({len(record['unwaived_enforced_findings'])})", ""])
    for finding in record["unwaived_enforced_findings"]:
      lines.extend([f"### `{finding['symbol']}` +0x{finding['offset']:x}", "",
                    f"- `{finding['locator']}` for {', '.join(finding['primitives'])}",
                    *[f"- {reason}" for reason in finding["slice"]["reasons"]],
                    "", "```text", finding["slice"]["canonical"], "```", ""])
  return "\n".join(lines)


def apply_renewals(path: Path, proposals: list[dict[str, Any]], reviewed_at: str) -> int:
  """Replace the locator, function hash, and review date of each proposed waiver."""
  import tomllib

  text = path.read_text()
  lines = text.split("\n")
  headers = [number for number, line in enumerate(lines) if line.startswith("[")]
  blocks = [(number, headers[position + 1] if position + 1 < len(headers) else len(lines))
            for position, number in enumerate(headers) if lines[number].strip() == "[[asm_waiver]]"]
  configured = tomllib.loads(text)["asm_waiver"]
  if len(configured) != len(blocks):
    raise SystemExit("cannot map asm_waiver tables to ct.toml lines")
  for row in proposals:
    waiver = configured[row["index"]]
    if waiver != row["waiver"]:
      raise SystemExit(f"asm_waiver[{row['index']}] changed since the renewal record was written")
    start, end = blocks[row["index"]]
    replacements = {
      f'locator = "{waiver["locator"]}"': f'locator = "{row["new"]["locator"]}"',
      f'function_sha256 = "{waiver["function_sha256"]}"': f'function_sha256 = "{row["new"]["function_sha256"]}"',
      f'reviewed_at = "{waiver["reviewed_at"]}"': f'reviewed_at = "{reviewed_at}"',
    }
    found = 0
    for number in range(start, end):
      if lines[number] in replacements:
        lines[number] = replacements[lines[number]]
        found += 1
    if found != 3:
      raise SystemExit(f"cannot rewrite asm_waiver[{row['index']}] in place")
  path.write_text("\n".join(lines))
  return len(proposals)


if __name__ == "__main__":
  raise SystemExit(main())
