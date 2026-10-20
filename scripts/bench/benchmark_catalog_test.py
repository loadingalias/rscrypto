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

  # The existing SLH-DSA names retain their workloads while CI separates complete operation groups.
  # Enumerate the harness's registered cases independently of the catalog filters: an omitted operation,
  # overlapping filter, or split between implementations would silently lose or distort comparisons.
  slhdsa = {name: re.compile(row["filter"]) for name, row in catalog["algorithms"].items()
            if row["bench"] == "slhdsa"}
  covered = set()
  for family in ("sha2", "shake"):
    for parameter in ("128s", "128f", "192s", "192f", "256s", "256f"):
      name = f"slhdsa-{family}-{parameter}"
      prehash = ("sha256" if parameter.startswith("128") else "sha512") if family == "sha2" else (
        "shake128" if parameter.startswith("128") else "shake256")
      operations = {
        "keygen": ("fixed-seeds", ("rscrypto", "rustcrypto")),
        "sign": ("deterministic-32", ("rscrypto", "rustcrypto", "fips205")),
        "verify": ("32", ("rscrypto", "rustcrypto", "fips205")),
        "import": ("raw-secret", ("rscrypto",)),
        "hash-sign": (f"{prehash}/deterministic-32", ("rscrypto", "fips205")),
        "hash-verify": (f"{prehash}/32", ("rscrypto", "fips205")),
      }
      for operation, (workload, implementations) in operations.items():
        selector = f"hash-{name}-{operation.removeprefix('hash-')}" if operation.startswith("hash-") else f"{name}-{operation}"
        for implementation in implementations:
          case = f"{name}/{operation}/{workload}/{implementation}"
          owners = [member for member, pattern in slhdsa.items() if pattern.search(case)]
          if owners != [selector]:
            fail(f"SLH-DSA comparison partition changed: {case} belongs to {owners}, expected {selector}")
          covered.add(selector)
      for alias, operations in ((name, ("keygen", "sign", "verify", "import")),
                                ("hash-" + name, ("sign", "verify"))):
        if resolve_selector(catalog, alias) != [f"{alias}-{operation}" for operation in operations]:
          fail(f"SLH-DSA selector no longer preserves its complete workload: {alias}")
  if covered != set(slhdsa):
    fail("SLH-DSA catalog contains selectors outside the operation partition")

  # RSA key generation runs one key size per invocation; each selector keeps every implementation.
  keygen = {name: re.compile(row["filter"]) for name, row in catalog["algorithms"].items()
            if row["bench"] == "rsa_keygen"}
  for bits in (2048, 3072, 4096):
    for implementation in ("rscrypto", "rustcrypto-rsa", "aws-lc-rs"):
      case = f"rsa-{bits}-keygen/{implementation}"
      owners = [name for name, pattern in keygen.items() if pattern.search(case)]
      if owners != [f"rsa-{bits}-keygen"]:
        fail(f"RSA key generation partition changed: {case} belongs to {owners}")
  if set(keygen) != {f"rsa-{bits}-keygen" for bits in (2048, 3072, 4096)}:
    fail("RSA key generation catalog contains selectors outside the key-size partition")

  extended = {f'{prefix}-{family}-{parameter}-sign': 1800
              for prefix in ('slhdsa', 'hash-slhdsa') for family in ('sha2', 'shake')
              for parameter in ('128s', '192s', '256s')} | {'rsa-4096-keygen': 1800}
  if {name: row['max_run_seconds'] for name, row in catalog['algorithms'].items()
      if 'max_run_seconds' in row} != extended:
    fail('extended budgets must cover exactly the twelve small-signature SLH-DSA signing comparisons '
         'and RSA-4096 key generation')
  for kind, selector, budget in (('algorithms', 'slhdsa-sha2-128s-sign', value)
                                for value in (True, 600, 1801, 1800.0, '1800')):
    broken = copy.deepcopy(catalog)
    broken[kind][selector]['max_run_seconds'] = budget
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f'catalog accepted invalid comparison budget: {budget}')
  for kind, selector in (('algorithms', 'sha256'), ('benches', 'slhdsa')):
    broken = copy.deepcopy(catalog)
    broken[kind][selector]['max_run_seconds'] = 1800
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f'catalog accepted an extended budget outside a split algorithm: {selector}')

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
  for name, split in (("slhdsa", "parameter-sets"), ("slhdsa", True), ("blake2", "algorithms"),
                      ("structural", "algorithms")):
    broken = copy.deepcopy(catalog)
    broken["benches"][name]["split"] = split
    try:
      validate_catalog(broken)
    except CatalogError:
      pass
    else:
      fail(f"catalog accepted split {split!r} for {name}")
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
    ("pbkdf2", "pbkdf2-sha256/iters=1000/rscrypto/32", "expensive"),
    ("pbkdf2", "pbkdf2-sha256/verify/iters=600000/rscrypto/32", "expensive"),
    ("pbkdf2", "pbkdf2-sha256/internal/iters=1000/rscrypto-oneshot/32", "diagnostic"),
    ("password_hashing", "argon2id-owasp/rscrypto/m=19MiB_t=2_p=1", "expensive"),
    ("rsa", "rsa-2048-private-signing/blinding-inverse-scratch-rscrypto", "diagnostic"),
    ("rsa", "rsa-2048-private-signing/sign-pss-sha256/caller-entropy/rscrypto", "expensive"),
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
    if not re.search(r'bench_config::run\(&', source):
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
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/compare_test.py")], check=True)
  print("benchmark catalog and orchestration tests passed")


if __name__ == "__main__":
  main()
