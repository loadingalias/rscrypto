#!/usr/bin/env python3
"""Check one production SecretBytes<32> drop in final release-LTO LLVM IR.

This is a fixed-owner compiler regression gate, not whole-library cleanup proof.
It consumes the existing linked CT artifact build; it does not compile a second
implementation or make post-drop reads that could keep ordinary stores alive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SYMBOL = "zeroize_entry_secret_bytes_32"


def inspect_ir(ir: str) -> dict[str, object]:
    match = re.search(r"^define [^\n]*@" + SYMBOL + r"\([^\n]*\n(.*?)^}", ir, re.M | re.S)
    if match is None:
        raise ValueError(f"missing final-LTO definition: {SYMBOL}")
    body = match.group(1)
    # Deliberately accept only the simple straight-line sentinel. A compiler
    # shape change needs review, rather than a guessed loop/control-flow proof.
    if re.search(r"^\s*(?:br|switch|invoke|callbr|indirectbr)\b", body, re.M):
        raise ValueError("cleanup sentinel is not straight-line; review new compiler output")
    for line in body.splitlines():
        if re.search(r"\bcall\b", line) and not any(token in line for token in
                ("@llvm.lifetime.", "@llvm.memcpy.", 'asm sideeffect "",')):
            raise ValueError("unreviewed call in cleanup sentinel")
    owners = re.findall(r"(%[-\w.]+) = alloca \[32 x i8\]", body)
    if len(owners) != 1:
        raise ValueError("expected exactly one 32-byte owner allocation")
    offsets = {owners[0]: 0}
    covered: set[int] = set()
    fence_after_wipe = False
    returned = False
    for line in body.splitlines():
        if covered and re.search(r"\bcall\b", line) and "@llvm.lifetime.end." not in line:
            raise ValueError("unreviewed call after cleanup began")
        if covered and re.search(r"\bload\b", line):
            raise ValueError("post-wipe read could keep ordinary cleanup live")
        gep = re.search(r"(%[-\w.]+) = getelementptr(?: inbounds)?(?: nuw)? i(8|16|32|64), ptr (%[-\w.]+), i(?:32|64) (\d+)", line)
        if gep and gep[3] in offsets:
            offsets[gep[1]] = offsets[gep[3]] + int(gep[2]) // 8 * int(gep[4])
        array_gep = re.search(r"(%[-\w.]+) = getelementptr(?: inbounds)?(?: nuw)? \[32 x i8\], ptr (%[-\w.]+), i(?:32|64) 0, i(?:32|64) (\d+)", line)
        if array_gep and array_gep[2] in offsets:
            offsets[array_gep[1]] = offsets[array_gep[2]] + int(array_gep[3])
        store = re.search(r"store volatile i(8|16|32|64|128) 0, ptr (%[-\w.]+)", line)
        if store and store[2] in offsets:
            start = offsets[store[2]]
            covered.update(range(start, start + int(store[1]) // 8))
            fence_after_wipe = False
        elif re.search(r"\bstore\b", line) and any(re.search(r"ptr " + re.escape(pointer) + r"(?:,|$)", line) for pointer in offsets):
            if covered:
                raise ValueError("owner overwritten after cleanup began")
        if 'fence syncscope("singlethread") seq_cst' in line and covered == set(range(32)):
            fence_after_wipe = True
        if re.search(r"^\s*ret\b", line):
            if not fence_after_wipe:
                raise ValueError("return lacks a complete volatile wipe followed by compiler fence")
            returned = True
    if not returned:
        raise ValueError("missing checked return")
    return {"symbol": SYMBOL, "cleared_bytes": sorted(covered), "compiler_fence": True,
            "control_flow": "straight-line", "evidence": "final-release-LTO-IR"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directory = args.artifact_dir
    ir_files = list(directory.glob("rscrypto_ct_evidence*.ll"))
    binaries = [path for path in (directory / "rscrypto-ct-evidence", directory / "rscrypto-ct-evidence.exe") if path.is_file()]
    report: dict[str, object] = {"status": "fail", "scope": "8-byte-aligned SecretBytes<32> drop sentinel only",
        "limits": ["Other alignments, owners, operations, heap cleanup, error paths and compiler-created copies are not verified.",
                   "IR is checked automatically; retained final disassembly still requires machine-code review."]}
    try:
        if len(ir_files) != 1 or len(binaries) != 1:
            raise ValueError("expected one final linked binary and its emitted LLVM IR")
        binary = binaries[0]
        symbols = directory / (binary.name + ".binary.nm-symbols.txt")
        assembly = directory / (binary.name + ".binary.raw-disasm.txt")
        if not re.search(r"\b_?" + SYMBOL + r"\b", symbols.read_text()):
            raise ValueError("cleanup sentinel missing from linked binary symbol table")
        if not re.search(r"\b_?" + SYMBOL + r"\b", assembly.read_text()):
            raise ValueError("cleanup sentinel missing from final disassembly")
        report.update(inspect_ir(ir_files[0].read_text()))
        report["artifacts"] = {path.name: digest(path) for path in (binary, ir_files[0], symbols, assembly)}
        report["status"] = "pass"
    except (OSError, ValueError) as error:
        report["error"] = str(error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
