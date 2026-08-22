#!/usr/bin/env python3
"""Contract tests for the benchmark catalog and Cargo benchmark targets."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / ".config" / "benchmark-matrix.json"
TOOL = ROOT / "scripts" / "bench" / "benchmark_catalog.py"


def fail(message: str) -> None:
  raise AssertionError(message)


def query(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [sys.executable, str(TOOL), *args],
    cwd=ROOT,
    check=check,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )


def main() -> None:
  query("validate")
  with CATALOG.open(encoding="utf-8") as source:
    catalog = json.load(source)
  with (ROOT / "Cargo.toml").open("rb") as source:
    cargo = tomllib.load(source)

  cargo_benches = {bench["name"]: bench for bench in cargo["bench"]}
  catalog_binaries = {bench["binary"] for bench in catalog["benches"].values()}
  if set(cargo_benches) != catalog_binaries:
    fail(
      "Cargo and benchmark catalog targets differ: "
      f"Cargo-only={sorted(set(cargo_benches) - catalog_binaries)}, "
      f"catalog-only={sorted(catalog_binaries - set(cargo_benches))}"
    )

  for name, bench in catalog["benches"].items():
    required = set(cargo_benches[bench["binary"]].get("required-features", []))
    selected = set(bench["features"])
    if not required <= selected:
      fail(f"catalog bench {name} omits required features: {sorted(required - selected)}")

  expected = {
    "checksum": "crc16-ccitt,crc16-ibm,crc24-openpgp,crc32-ieee,crc32c,crc64-xz,crc64-nvme",
    "mlkem": "mlkem512,mlkem768,mlkem1024",
    "sha512-256": "sha512-256",
  }
  for selector, algorithms in expected.items():
    actual = query("resolve-selector", selector).stdout.strip()
    if actual != algorithms:
      fail(f"selector {selector} resolved to {actual}, expected {algorithms}")

  unknown = query("resolve-selector", "raw-criterion-filter", check=False)
  if unknown.returncode != 3 or unknown.stdout:
    fail("unknown selectors must remain available as raw Criterion filters")

  if query("plan-algorithm", "sha512").stdout.strip() != "hashes|sha2|^sha512/":
    fail("SHA-512 benchmark identity changed")
  if query("plan-algorithm", "aead-diag").stdout.strip() != "aead|aead_diag|chacha20-poly1305/encrypt":
    fail("AEAD diagnostic benchmark identity changed")
  if query("binary", "aead_diag").stdout.strip() != "aead":
    fail("AEAD diagnostic selector must use the aead benchmark binary")

  expanded = query("expand-benches", "checksum_comp,auth_comp").stdout.strip()
  if expanded != "crc,auth":
    fail(f"bench aliases expanded to {expanded}")

  features = query("features", "sha2,aead").stdout.strip().split(",")
  if len(features) != len(set(features)) or not {"parallel", "sha2", "aes-gcm"} <= set(features):
    fail("bench feature union is incomplete or duplicated")

  required = set(query("required-benches").stdout.strip().split(","))
  expected_required = {name for name, bench in catalog["benches"].items() if bench["required"]}
  if required != expected_required:
    fail("required benchmark targets are not derived from the catalog")

  criterion = set(query("criterion-binaries").stdout.strip().split(","))
  if "structural" in criterion or "aead" not in criterion:
    fail("generic Criterion runs must exclude Gungraun and include the AEAD binary")

  print("benchmark catalog tests passed")


if __name__ == "__main__":
  main()
