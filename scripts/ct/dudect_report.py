#!/usr/bin/env python3
"""Build a machine-readable report from dudect-bencher output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CASE_METADATA = {
  "constant_time_eq_equal_vs_first_diff": {
    "primitive": "ct.constant_time_eq",
    "left_class": "equal 64-byte inputs",
    "right_class": "same length with first byte different",
  },
  "hmac_sha256_valid_vs_invalid_tag": {
    "primitive": "mac.hmac_verify",
    "left_class": "valid tag",
    "right_class": "invalid first tag byte",
  },
  "kmac256_valid_vs_invalid_tag": {
    "primitive": "mac.kmac256_verify",
    "left_class": "valid tag",
    "right_class": "invalid first tag byte",
  },
  "blake3_keyed_valid_vs_invalid_tag": {
    "primitive": "keyed_hash.blake3_verify",
    "left_class": "valid keyed digest",
    "right_class": "invalid first digest byte",
  },
  "secret_wrappers_eq_and_debug_fixed_vs_random": {
    "primitive": "secret_wrappers.equality_and_display",
    "left_class": "fixed secret bytes",
    "right_class": "random secret bytes",
  },
  "xchacha20poly1305_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "valid open with fixed secret key",
    "right_class": "valid open with random secret key",
  },
  "aes128_gcm_siv_diag_derive_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "derive AES-128-GCM-SIV message keys from fixed master key",
    "right_class": "derive AES-128-GCM-SIV message keys from random master key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for per-message key derivation.",
  },
  "aes256_gcm_siv_diag_derive_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "derive AES-256-GCM-SIV message keys from fixed master key",
    "right_class": "derive AES-256-GCM-SIV message keys from random master key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for per-message key derivation.",
  },
  "gcm_siv_diag_polyval_fixed_vs_random_auth_key": {
    "primitive": "aead.open_authentication",
    "left_class": "POLYVAL with fixed authentication key",
    "right_class": "POLYVAL with random authentication key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for POLYVAL under secret H.",
  },
  "aes128_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "AES-128 tag block under fixed raw encryption key",
    "right_class": "AES-128 tag block under random raw encryption key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES.",
  },
  "aes256_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "AES-256 tag block under fixed raw encryption key",
    "right_class": "AES-256 tag block under random raw encryption key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES.",
  },
  "aes128_gcm_diag_ctr32_be_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM CTR with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM CTR with random secret key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for exact 44-byte CTR over the active AES backend.",
  },
  "aes256_gcm_diag_ctr32_be_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM CTR with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM CTR with random secret key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for exact 44-byte CTR over the active AES backend.",
  },
  "aes128_gcm_diag_ghash_fixed_vs_random_h": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM GHASH with fixed secret H",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM GHASH with random secret H",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for GHASH under the active carryless multiply backend.",
  },
  "aes256_gcm_diag_ghash_fixed_vs_random_h": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM GHASH with fixed secret H",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM GHASH with random secret H",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for GHASH under the active carryless multiply backend.",
  },
  "aes128_gcm_diag_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM final tag AES with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM final tag AES with random secret key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for final tag AES under secret key variation.",
  },
  "aes256_gcm_diag_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM final tag AES with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM final tag AES with random secret key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM timing trace for final tag AES under secret key variation.",
  },
  "aes128_gcm_siv_diag_ctr32_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM-SIV CTR with fixed raw encryption key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM-SIV CTR with random raw encryption key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for exact 44-byte CTR over the active AES backend.",
  },
  "aes256_gcm_siv_diag_ctr32_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM-SIV CTR with fixed raw encryption key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM-SIV CTR with random raw encryption key",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for exact 44-byte CTR over the active AES backend.",
  },
  "aes128_gcm_siv_diag_raw_tag_aes_varying_block": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM-SIV raw tag AES with fixed block",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM-SIV raw tag AES with random block",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES under block variation.",
  },
  "aes256_gcm_siv_diag_raw_tag_aes_varying_block": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM-SIV raw tag AES with fixed block",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM-SIV raw tag AES with random block",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES under block variation.",
  },
  "x25519_fixed_vs_random_scalar": {
    "primitive": "kx.x25519",
    "left_class": "fixed scalar",
    "right_class": "random scalar",
  },
  "ed25519_sign_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret",
    "right_class": "random signing secret",
  },
  "ed25519_public_key_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret public-key derivation",
    "right_class": "random signing secret public-key derivation",
  },
  "ed25519_keypair_sign_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed keypair signing secret",
    "right_class": "random keypair signing secret",
  },
  "ed25519_sha512_secret_expand_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret SHA-512 expansion",
    "right_class": "random signing secret SHA-512 expansion",
  },
  "rsa_pkcs1v15_fixed_vs_random_message": {
    "primitive": "rsa.private_ops",
    "left_class": "fixed message",
    "right_class": "random same-length message",
  },
  "rsa_private_key_pkcs8_import_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "valid PKCS#8 private key A",
    "right_class": "valid same-size same-shape PKCS#8 private key B",
    "gate": "diagnostic",
    "reason": "Classes differ in RSA public modulus; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_validate_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "valid PKCS#8 private key A validation",
    "right_class": "valid same-size same-shape PKCS#8 private key B validation",
    "gate": "diagnostic",
    "reason": "Full RSA private-key import validation performs value-dependent big-integer consistency checks over unrelated keys; required CT evidence is scoped to bounded validation leaves and steady-state private operations.",
  },
  "rsa_private_key_pkcs8_import_stage50_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "validated PKCS#8 private key A before materialization",
    "right_class": "validated same-size same-shape PKCS#8 private key B before materialization",
  },
  "rsa_private_key_pkcs8_import_stage51_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after public modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after public modulus setup",
    "gate": "diagnostic",
    "reason": "Includes RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage52_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after p modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after p modulus setup",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage53_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after q modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after q modulus setup",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage54_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after secret integer boxing",
    "right_class": "same-size same-shape PKCS#8 private key B after secret integer boxing",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_export_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "validated private key A",
    "right_class": "validated same-size same-shape private key B",
    "gate": "diagnostic",
    "reason": "Canonical DER private-key export is a secret-output formatting operation with value-dependent INTEGER shape.",
  },
  "hkdf_sha2_fixed_vs_random_ikm": {
    "primitive": "kdf.hkdf",
    "left_class": "fixed input key material",
    "right_class": "random input key material",
  },
  "pbkdf2_sha2_fixed_vs_random_password": {
    "primitive": "kdf.pbkdf2",
    "left_class": "fixed password",
    "right_class": "random same-length password",
  },
  "argon2i_fixed_vs_random_password": {
    "primitive": "password.argon2i",
    "left_class": "fixed password",
    "right_class": "random same-length password",
  },
  "chacha20poly1305_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "fixed secret key",
    "right_class": "random secret key",
  },
  "blake2b256_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2b-256 keyed-hash key",
    "right_class": "random BLAKE2b-256 keyed-hash key",
  },
  "blake2b512_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2b-512 keyed-hash key",
    "right_class": "random BLAKE2b-512 keyed-hash key",
  },
  "blake2s128_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2s-128 keyed-hash key",
    "right_class": "random BLAKE2s-128 keyed-hash key",
  },
  "blake2s256_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2s-256 keyed-hash key",
    "right_class": "random BLAKE2s-256 keyed-hash key",
  },
  "blake3_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE3 keyed-hash key",
    "right_class": "random BLAKE3 keyed-hash key",
  },
}

SEED_RE = re.compile(r"^bench\s+(?P<name>\S+)\s+seeded with (?P<seed>0x[0-9a-fA-F]+)$")
RESULT_RE = re.compile(
  r"^bench (?P<name>\S+)\s+\.\.\. : n == (?P<n>[+-]?[0-9.]+)M, "
  r"max t = (?P<t>[+-]?[0-9.]+), max tau = (?P<tau>[+-]?[0-9.]+), "
  r"\(5/tau\)\^2 = (?P<needed>[0-9]+)$"
)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def rustc_verbose() -> str:
  try:
    return subprocess.check_output(["rustc", "-vV"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unavailable"


def parse_stdout(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
  seeds: dict[str, str] = {}
  results: dict[str, dict] = {}
  for line in path.read_text(errors="replace").splitlines():
    if seed_match := SEED_RE.match(line):
      seeds[seed_match.group("name")] = seed_match.group("seed")
      continue
    if result_match := RESULT_RE.match(line):
      name = result_match.group("name")
      max_t = float(result_match.group("t"))
      max_tau = float(result_match.group("tau"))
      results[name] = {
        "samples_millions": float(result_match.group("n")),
        "max_t": max_t,
        "abs_max_t": abs(max_t),
        "max_tau": max_tau,
        "needed_samples_for_tau_threshold": int(result_match.group("needed")),
      }
  return seeds, results


def raw_csv_rows(path: Path) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  if not path.exists():
    return rows

  with path.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      name = row.get("benchname", "")
      class_name = row.get("class", "")
      if not name or not class_name:
        continue
      entry = rows.setdefault(name, {"row_count": 0, "labels": {}})
      entry["row_count"] += 1
      entry["labels"].setdefault(class_name, 0)
      entry["labels"][class_name] += 1
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stdout", required=True, type=Path)
  parser.add_argument("--csv", required=True, type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--threshold", type=float, default=10.0)
  parser.add_argument("--samples", type=int, required=True)
  parser.add_argument("--command", default="")
  args = parser.parse_args()

  seeds, results = parse_stdout(args.stdout)
  raw_rows = raw_csv_rows(args.csv)
  cases = []
  for name in sorted(results):
    result = results[name]
    metadata = CASE_METADATA.get(name, {})
    gate = metadata.get("gate", "required")
    passed = result["abs_max_t"] <= args.threshold
    diagnostic = gate == "diagnostic"
    cases.append(
      {
        "name": name,
        "primitive": metadata.get("primitive", "unknown"),
        "left_class": metadata.get("left_class", "unknown"),
        "right_class": metadata.get("right_class", "unknown"),
        "gate": gate,
        "diagnostic_reason": metadata.get("reason"),
        "seed": seeds.get(name),
        "requested_samples": args.samples,
        "raw_csv": raw_rows.get(name, {"row_count": 0, "labels": {}}),
        **result,
        "threshold_abs_max_t": args.threshold,
        "status": "pass" if passed else ("diagnostic-fail" if diagnostic else "fail"),
      }
    )

  report = {
    "schema_version": 1,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": args.target,
    "target_triple": args.target,
    "profile": args.profile,
    "threshold_abs_max_t": args.threshold,
    "requested_samples": args.samples,
    "command": args.command,
    "rustc_verbose": rustc_verbose(),
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "python": platform.python_version(),
    },
    "raw_stdout": str(args.stdout),
    "raw_stdout_sha256": sha256_file(args.stdout) if args.stdout.exists() else None,
    "raw_csv": str(args.csv),
    "raw_csv_sha256": sha256_file(args.csv) if args.csv.exists() else None,
    "cases": cases,
    "case_count": len(cases),
    "failure_count": sum(1 for case in cases if case["status"] == "fail"),
    "diagnostic_failure_count": sum(1 for case in cases if case["status"] == "diagnostic-fail"),
    "notes": [
      "DudeCT is empirical timing evidence, not a proof.",
      "A pass means no leakage was detected for this configuration, host, and input classification.",
      "dudect-bencher 0.7.0 writes both raw CSV classes with label 0; raw CSV labels are recorded for traceability, not class-balance proof.",
    ],
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"dudect report: {args.out}")
  return 1 if report["failure_count"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
