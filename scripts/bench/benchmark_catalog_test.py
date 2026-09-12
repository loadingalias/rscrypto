#!/usr/bin/env python3
"""Check the benchmark catalog, then exercise benchmark and profiling front doors."""

from __future__ import annotations

import json
import copy
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from benchmark_catalog import CatalogError, case_class, resolve_selector, validate_catalog


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
    actual = ",".join(resolve_selector(catalog, selector))
    if actual != algorithms:
      fail(f"selector {selector} resolved to {actual}, expected {algorithms}")

  if resolve_selector(catalog, "unknown-selector") is not None:
    fail("unknown selector accepted")

  for name in catalog["algorithms"]:
    if resolve_selector(catalog, name) != [name]:
      fail(f"exact algorithm {name} selects other algorithms")
  shadowed = copy.deepcopy(catalog)
  shadowed["selectors"]["crc64nvme"] = ["crc64-xz", "crc64-nvme"]
  try:
    validate_catalog(shadowed)
  except CatalogError:
    pass
  else:
    fail("catalog accepted a family selector shadowing an exact algorithm")
  for case in ("blake2/rscrypto/blake2b256/64", "blake2/dryoc/blake2b256/64",
               "blake2/keyed/dryoc/blake2b256/64", "blake2/streaming/dryoc/blake2b256/64B",
               "blake2/params/rscrypto/blake2b256/salt+personal/64"):
    if not re.search(catalog["algorithms"]["blake2"]["filter"], case):
      fail(f"BLAKE2 selector excludes {case}")
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
    if "std" not in features or ("parallel" in features) != (name in {"blake3", "password_hashing"}):
      fail(f"incorrect runtime feature scope for {name}")

  for binary in catalog_binaries - {"structural"}:
    source = (ROOT / "benches" / f"{binary}.rs").read_text()
    if 'bench_config::run(&[' not in source:
      fail(f"{binary} does not use the shared Criterion configuration")
    if re.search(r"\.(sample_size|measurement_time|warm_up_time|nresamples|confidence_level|significance_level|noise_threshold)\s*\(", source):
      fail(f"{binary} overrides the shared Criterion configuration")

  subprocess.run([sys.executable, str(ROOT / "scripts/bench/bounded_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/run_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/profile_test.py")], check=True)
  subprocess.run([sys.executable, str(ROOT / "scripts/bench/transfer_test.py")], check=True)
  print("benchmark catalog and orchestration tests passed")


if __name__ == "__main__":
  main()
