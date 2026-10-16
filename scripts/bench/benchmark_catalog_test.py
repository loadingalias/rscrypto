#!/usr/bin/env -S uv run python
"""Check the benchmark catalog, then exercise benchmark and profiling front doors."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import tomllib
from benchmark_catalog import (
  CatalogError,
  case_class,
  resolve_selector,
  validate_catalog,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / ".config" / "benchmark-matrix.json"


def fail(message: str) -> None:
  raise AssertionError(message)


def main() -> None:
  with CATALOG.open(encoding="utf-8") as source:
    catalog = json.load(source)
  with (ROOT / "Cargo.toml").open("rb") as source:
    cargo = tomllib.load(source)

  validate_catalog(catalog)

  manifests = {entry.get('manifest', 'Cargo.toml') for entry in catalog['benches'].values()}
  cargo_benches = {
    (manifest, bench['name']): bench
    for manifest in manifests
    for bench in tomllib.loads((ROOT / manifest).read_text()).get('bench', [])
  }
  catalog_binaries = {(bench.get('manifest', 'Cargo.toml'), bench['binary']) for bench in catalog['benches'].values()}
  if set(cargo_benches) != catalog_binaries:
    fail(
      "Cargo and benchmark catalog targets differ: "
      f"Cargo-only={sorted(set(cargo_benches) - catalog_binaries)}, "
      f"catalog-only={sorted(catalog_binaries - set(cargo_benches))}"
    )

  for name, bench in catalog["benches"].items():
    required = set(cargo_benches[(bench.get('manifest', 'Cargo.toml'), bench["binary"])].get("required-features", []))
    selected = set(bench["features"])
    if not required <= selected:
      fail(f"catalog bench {name} omits required features: {sorted(required - selected)}")

  defaults = [name for name, bench in catalog["benches"].items() if bench["kind"] == "criterion" and bench["required"]]
  expected = {
    "all": ",".join(defaults),
    "checksums": "crc16-ccitt,crc16-ibm,crc24-openpgp,crc32-ieee,crc32c,crc64-xz,crc64-nvme",
    "mlkem": "mlkem512,mlkem768,mlkem1024",
    "sha512-256": "sha512-256",
    "SHA512_256": "sha512-256",
    "password-hashing": "password_hashing",
  }
  for selector, names in expected.items():
    actual = ",".join(resolve_selector(catalog, selector))
    if actual != names:
      fail(f"selector {selector} resolved to {actual}, expected {names}")
  if "tls_connections" in defaults or "structural" in defaults:
    fail("default runs include a target that needs an external peer or another harness")

  for selector in ("unknown-selector", "bench=sha2", "hash", "checksum"):
    if resolve_selector(catalog, selector) is not None:
      fail(f"unknown selector accepted: {selector}")

  for kind in ("algorithms", "benches"):
    for name in catalog[kind]:
      if resolve_selector(catalog, name) != [name]:
        fail(f"exact name {name} selects other names")
  for selector, collision in (("crc64nvme", "groups"), ("auth", "groups"), ("all", "groups"), ("blake3", "algorithms")):
    shadowed = copy.deepcopy(catalog)
    shadowed[collision][selector] = (["crc64-xz", "crc64-nvme"] if collision == "groups"
                                     else {"bench": "blake3", "filter": "blake3"})
    try:
      validate_catalog(shadowed)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted a duplicate selector name: {selector}")
  for change in ({"required": True}, {"external": ""}):
    broken = copy.deepcopy(catalog)
    broken["benches"]["tls_connections"].update(change)
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted an invalid external prerequisite: {change}")
  for name, family in (("sha2", None), ("sha2", ""), ("sha2", "Hashes"), ("sha2", "hashes/x"),
                       ("structural", "hashes")):
    broken = copy.deepcopy(catalog)
    broken["benches"][name]["family"] = family
    if family is None:
      del broken["benches"][name]["family"]
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted family {family!r} for {name}")
  for members in ([], ["sha256", "sha256"], ["structural"], ["missing"]):
    broken = copy.deepcopy(catalog)
    broken["groups"]["hashes"] = members
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted invalid group members: {members}")
  aes_profile = catalog["profile_presets"].get("aead/aes")
  if aes_profile != {
    "bench": "aead",
    "case": "aes-128-gcm/copy-and-encrypt/rscrypto/4096",
    "diagnostic": False,
  }:
    fail("aead/aes profile preset no longer names the representative production case")
  for field, value in (("bench", "structural"), ("case", ""), ("diagnostic", "false")):
    broken = copy.deepcopy(catalog)
    broken["profile_presets"]["aead/aes"][field] = value
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted invalid profile preset {field}")
  broken = copy.deepcopy(catalog)
  broken["profile_presets"]["aead/aes"]["cases_by_architecture"] = {"s390x-linux": ["case"]}
  try:
    validate_catalog(broken)
  except CatalogError:
    pass
  else:
    fail("catalog accepted an architecture-specific profile bundle")
  for binary, case, expected_class in (
    ("sha2", "sha256/rscrypto/64", "ordinary"),
    ("sha2", "sha256/internal/compress/rscrypto/64", "diagnostic"),
    ("blake3", "blake3/rscrypto-scalar/64", "diagnostic"),
    ("blake3", "blake3/keyed/rscrypto/64", "ordinary"),
    ("auth", "pbkdf2-sha256/iters=1000/rscrypto/32", "expensive"),
    ("auth", "pbkdf2-sha256/internal/iters=1000/rscrypto-oneshot/32", "diagnostic"),
    ("password_hashing", "argon2id-owasp/rscrypto/m=19MiB_t=2_p=1", "expensive"),
    ("rsa", "rsa-2048-private-signing/blinding-inverse-scratch-rscrypto", "diagnostic"),
    ("rsa", "rsa-2048-private-signing/sign-pss-sha256-caller-entropy-scratch-rscrypto", "expensive"),
    ("aead_kernels", "aead-kernel/poly1305-auth/dispatched/64", "diagnostic"),
    ("aead", "chacha20-poly1305/copy-and-encrypt/rscrypto/64", "ordinary"),
    ("aead", "chacha20-poly1305/copy-and-encrypt/rscrypto-owned/64", "diagnostic"),
    ("aead", "chacha20-poly1305/copy-and-decrypt/rscrypto-x86-asm/64", "diagnostic"),
    ("blake2", "blake2/short-oneshot/rscrypto/blake2b256/16", "ordinary"),
    ("blake2", "blake2/single-update/rscrypto/blake2b256/16", "ordinary"),
  ):
    if case_class(catalog, binary, case) != expected_class:
      fail(f"incorrect work class for {binary}: {case}")

  for name, bench in catalog["benches"].items():
    features = set(bench["features"])
    if 'manifest' in bench:
      continue  # Companion tools own their runtime features in their manifest.
    if "std" not in features or ("parallel" in features) != (name in {"blake3", "blake3_file", "password_hashing"}):
      fail(f"incorrect runtime feature scope for {name}")

  for manifest, binary in catalog_binaries:
    if binary == 'structural':
      continue
    source = (ROOT / Path(manifest).parent / "benches" / f"{binary}.rs").read_text()
    if 'bench_config::run(&[' not in source:
      fail(f"{binary} does not use the shared Criterion configuration")
    if re.search(r"\.(sample_size|measurement_time|warm_up_time|nresamples|confidence_level|significance_level|noise_threshold)\s*\(", source):
      fail(f"{binary} overrides the shared Criterion configuration")

  for manifest in ('../Cargo.toml', '/tmp/Cargo.toml', 'tools/missing/Cargo.toml'):
    broken = copy.deepcopy(catalog)
    broken['benches']['tls_connections']['manifest'] = manifest
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail('catalog accepted an invalid companion manifest')

  subprocess.run([sys.executable, str(ROOT / "scripts/bench/bounded_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/run_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/profile_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/profile_ci_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/transfer_test.py")], check=True)
  print("benchmark catalog and orchestration tests passed")


if __name__ == "__main__":
  main()
