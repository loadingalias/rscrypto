#!/usr/bin/env python3
"""Derive the SLH-DSA ACVP sigGen and sigVer fixtures from the pinned upstream JSON.

The upstream files total about 68 MB, mostly hexadecimal signatures. This writes
every case's fields as length-prefixed binary instead, and replaces each expected
sigGen signature with its SHA-256. The format is documented in
testdata/slhdsa/acvp/README.md.

  scripts/test/slhdsa_acvp.py derive            # fetch the pinned files
  scripts/test/slhdsa_acvp.py derive --source DIR

Either way, every input must match its pinned SHA-256 before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

COMMIT = "975de31eb83d87039ec88934fdc47d8c312b892d"
UPSTREAM = f"https://raw.githubusercontent.com/usnistgov/ACVP-Server/{COMMIT}/gen-val/json-files"

# Local name -> (upstream directory/file, SHA-256 of the unmodified bytes).
SOURCES = {
  "sigGen-prompt.json": (
    "SLH-DSA-sigGen-FIPS205/prompt.json",
    "afa673eacdf0aec53512a159159b7632684adfcd0d88f8640a7f6f5796aacdc8",
  ),
  "sigGen-expectedResults.json": (
    "SLH-DSA-sigGen-FIPS205/expectedResults.json",
    "71e8e0f7e4b0cfd1747314299204d9d4d50968d200a4ae873921eaa7aabeaad1",
  ),
  "sigVer-prompt.json": (
    "SLH-DSA-sigVer-FIPS205/prompt.json",
    "4e7beb1233e47baa0acdd36417c66c45811aa40a4e32ffdb1a35d93b13b289fb",
  ),
  "sigVer-expectedResults.json": (
    "SLH-DSA-sigVer-FIPS205/expectedResults.json",
    "259f5e2a0665de0adc0fefa45b5db3a2a6ed13c3c44d14bdaf64a80aee12c687",
  ),
}

MAGIC = b"SLHACVP1"
OUTPUT = Path(__file__).resolve().parents[2] / "testdata" / "slhdsa" / "acvp"


def load(source: Path | None, name: str) -> dict:
  upstream, digest = SOURCES[name]
  if source is None:
    with urllib.request.urlopen(f"{UPSTREAM}/{upstream}") as response:
      data = response.read()
  else:
    data = (source / name).read_bytes()
  actual = hashlib.sha256(data).hexdigest()
  if actual != digest:
    raise SystemExit(f"{name}: SHA-256 {actual} does not match the pinned {digest}")
  return json.loads(data)


def expected_cases(results: dict) -> dict[tuple[int, int], dict]:
  return {(group["tgId"], case["tcId"]): case for group in results["testGroups"] for case in group["tests"]}


def text(value: object) -> bytes:
  if isinstance(value, bool):
    return b"true" if value else b"false"
  return str(value).encode("ascii")


def encode(cases: list[list[bytes]]) -> bytes:
  field_count = len(cases[0])
  out = bytearray(MAGIC)
  out += struct.pack("<II", field_count, len(cases))
  for fields in cases:
    assert len(fields) == field_count
    for field in fields:
      out += struct.pack("<I", len(field))
      out += field
  return bytes(out)


def signing_cases(prompt: dict, results: dict) -> list[list[bytes]]:
  expected = expected_cases(results)
  cases = []
  for group in prompt["testGroups"]:
    for case in group["tests"]:
      signature = bytes.fromhex(expected[(group["tgId"], case["tcId"])]["signature"])
      cases.append([
        text(group["parameterSet"]),
        text(group["signatureInterface"]),
        text(group.get("preHash", "")),
        text(group["deterministic"]),
        text(case.get("hashAlg", "")),
        text(group["tgId"]),
        text(case["tcId"]),
        bytes.fromhex(case["sk"]),
        bytes.fromhex(case["message"]),
        bytes.fromhex(case.get("context", "")),
        bytes.fromhex(case.get("additionalRandomness", "")),
        hashlib.sha256(signature).digest(),
      ])
  return cases


def verification_cases(prompt: dict, results: dict) -> list[list[bytes]]:
  expected = expected_cases(results)
  cases = []
  for group in prompt["testGroups"]:
    for case in group["tests"]:
      cases.append([
        text(group["parameterSet"]),
        text(group["signatureInterface"]),
        text(group.get("preHash", "")),
        text(case.get("hashAlg", "")),
        text(group["tgId"]),
        text(case["tcId"]),
        text(expected[(group["tgId"], case["tcId"])]["testPassed"]),
        bytes.fromhex(case["pk"]),
        bytes.fromhex(case["message"]),
        bytes.fromhex(case.get("context", "")),
        bytes.fromhex(case["signature"]),
      ])
  return cases


def derive(source: Path | None) -> None:
  outputs = {
    "sigGen.bin": encode(signing_cases(load(source, "sigGen-prompt.json"), load(source, "sigGen-expectedResults.json"))),
    "sigVer.bin": encode(
      verification_cases(load(source, "sigVer-prompt.json"), load(source, "sigVer-expectedResults.json"))
    ),
  }
  for name, data in outputs.items():
    (OUTPUT / name).write_bytes(data)
    print(f"{hashlib.sha256(data).hexdigest()}  {name}  ({len(data)} bytes)")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  commands = parser.add_subparsers(dest="command", required=True)
  derive_parser = commands.add_parser("derive", help="write sigGen.bin and sigVer.bin")
  derive_parser.add_argument("--source", type=Path, help="directory holding the pinned upstream files under their local names")
  args = parser.parse_args()
  if args.command == "derive":
    derive(args.source)
  return 0


if __name__ == "__main__":
  sys.exit(main())
