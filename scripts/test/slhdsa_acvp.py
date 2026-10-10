#!/usr/bin/env python3
"""Derive the SLH-DSA ACVP fixtures from the pinned upstream JSON.

The upstream files total about 68 MB, mostly hexadecimal signatures. This writes
every sigGen and sigVer case's fields as length-prefixed binary instead, and
replaces each expected sigGen signature with its SHA-256. It also writes a small
runtime subset, one record per parameter set with full expected signatures, for
targets that run vectors outside the unit tests. The format and the subset's
selection rule are documented in testdata/slhdsa/acvp/README.md.

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
  "keyGen-prompt.json": (
    "SLH-DSA-keyGen-FIPS205/prompt.json",
    "bce170976f257ee3dfc8c54ea46722ccb553539847daa6d8048f0216cc28b51c",
  ),
  "keyGen-expectedResults.json": (
    "SLH-DSA-keyGen-FIPS205/expectedResults.json",
    "f35f74b6676d6b369c87e88c36698f28c14d5929d31e507d910288c69258afee",
  ),
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


# The RFC 9909 HashSLH-DSA pre-hash pairing, by ACVP parameter-set name prefix.
PAIRINGS = {
  "SLH-DSA-SHA2-128": "SHA2-256",
  "SLH-DSA-SHA2-192": "SHA2-512",
  "SLH-DSA-SHA2-256": "SHA2-512",
  "SLH-DSA-SHAKE-128": "SHAKE-128",
  "SLH-DSA-SHAKE-192": "SHAKE-256",
  "SLH-DSA-SHAKE-256": "SHAKE-256",
}


def runtime_cases(key_prompt: dict, key_results: dict, sign_prompt: dict, sign_results: dict) -> list[list[bytes]]:
  """One record per parameter set, in keyGen order: the first keyGen case, the
  deterministic external pure case with the shortest message (first on ties),
  and the deterministic external pre-hash case that uses the RFC 9909 pairing."""
  key_expected = expected_cases(key_results)
  sign_expected = expected_cases(sign_results)
  records = []
  for key_group in key_prompt["testGroups"]:
    parameter_set = key_group["parameterSet"]
    key_case = key_group["tests"][0]
    key_want = key_expected[(key_group["tgId"], key_case["tcId"])]

    def group(pre_hash: str) -> dict:
      (match,) = [
        g
        for g in sign_prompt["testGroups"]
        if g["parameterSet"] == parameter_set
        and g["deterministic"]
        and g["signatureInterface"] == "external"
        and g.get("preHash") == pre_hash
      ]
      return match

    pure_group = group("pure")
    pure = min(pure_group["tests"], key=lambda case: len(case["message"]))
    prehash_group = group("preHash")
    (prehash,) = [case for case in prehash_group["tests"] if case["hashAlg"] == PAIRINGS[parameter_set[:-1]]]
    records.append([
      text(parameter_set),
      bytes.fromhex(key_case["skSeed"] + key_case["skPrf"] + key_case["pkSeed"]),
      bytes.fromhex(key_want["pk"]),
      bytes.fromhex(key_want["sk"]),
      bytes.fromhex(pure["sk"]),
      bytes.fromhex(pure["message"]),
      bytes.fromhex(pure["context"]),
      bytes.fromhex(sign_expected[(pure_group["tgId"], pure["tcId"])]["signature"]),
      bytes.fromhex(prehash["sk"]),
      bytes.fromhex(prehash["message"]),
      bytes.fromhex(prehash["context"]),
      bytes.fromhex(sign_expected[(prehash_group["tgId"], prehash["tcId"])]["signature"]),
    ])
  return records


def derive(source: Path | None) -> None:
  sign_prompt = load(source, "sigGen-prompt.json")
  sign_results = load(source, "sigGen-expectedResults.json")
  outputs = {
    "sigGen.bin": encode(signing_cases(sign_prompt, sign_results)),
    "sigVer.bin": encode(
      verification_cases(load(source, "sigVer-prompt.json"), load(source, "sigVer-expectedResults.json"))
    ),
    "runtime.bin": encode(
      runtime_cases(
        load(source, "keyGen-prompt.json"), load(source, "keyGen-expectedResults.json"), sign_prompt, sign_results
      )
    ),
  }
  for name, data in outputs.items():
    (OUTPUT / name).write_bytes(data)
    print(f"{hashlib.sha256(data).hexdigest()}  {name}  ({len(data)} bytes)")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  commands = parser.add_subparsers(dest="command", required=True)
  derive_parser = commands.add_parser("derive", help="write sigGen.bin, sigVer.bin, and runtime.bin")
  derive_parser.add_argument("--source", type=Path, help="directory holding the pinned upstream files under their local names")
  args = parser.parse_args()
  if args.command == "derive":
    derive(args.source)
  return 0


if __name__ == "__main__":
  sys.exit(main())
