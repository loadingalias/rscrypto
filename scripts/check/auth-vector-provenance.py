#!/usr/bin/env python3
"""Verify the pinned authentication Wycheproof corpus."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "testdata/auth/wycheproof"
UPSTREAM_COMMIT = "b61843a9a5115bb758134b6a1f5d5e502d445342"
FILES = {
    "ecdsa_secp256r1_sha256_test.json": "182db4f3e230f6f9fa9f800d2a614dede30284b8e8438bbfe1171905402e9332",
    "ecdsa_secp384r1_sha384_test.json": "8a5b3ae1760975143414811f13588c24d951d9d8c904195087ba327591dfe9cc",
    "ed25519_test.json": "70471c053c711731f2195ef4875b60ea7f5d6793939d99058ac12da810cb8e00",
    "hkdf_sha256_test.json": "bb2b462a38b251cb52a2aede706d6d4b62b26864f4e80c95497507ddb07c5f1e",
    "hkdf_sha384_test.json": "69ff6ea3657bb9c1b8cdffbbb4e7832353d08fd15c0d9997b03f7a6b180e3678",
    "hmac_sha256_test.json": "2d201cfa61d1bf95e6f5d07d96634b4a348b31e8eaa277ad7c8d09677b7a743f",
    "hmac_sha384_test.json": "28b9776e979dd755d852ca471043ea6cedce8b15f7a28abdf6ea9efd982b43c0",
    "hmac_sha512_test.json": "b6c90477bdb4a6fc8ee3d1f7b2c0b69a8dfffab34718abaa6cabd71cc2ba1207",
    "kmac256_no_customization_test.json": "950b9e8f64bd4e614aa3d825f0cd0570ec33c6cdfc81cbd043c429265149a671",
    "pbkdf2_hmacsha256_test.json": "fa21062c95e385aab1714d607c320d534c75082f9594bdf965dba3d934fd17ef",
    "pbkdf2_hmacsha512_test.json": "3bc72b80f5c3d79cc2565b9b98dd982e7b1e1082df3a356d648a4d77470aa1d7",
    "x25519_test.json": "35c3f5231cf25cc640b524d403461deee9e49441d5d915a3a25b2c8ff5adbe7d",
}


def fail(message: str) -> None:
    raise SystemExit(f"auth-vector-provenance: {message}")


def read_pinned(path: Path, expected: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        fail(f"{path} has SHA-256 {actual}; expected {expected}")
    return data


def require_exact_checkout(root: Path) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot inspect upstream checkout {root}: {error}")
    if actual != UPSTREAM_COMMIT:
        fail(f"{root} is at {actual}; expected {UPSTREAM_COMMIT}")


def read_upstream_blob(root: Path, name: str, expected: str) -> bytes:
    try:
        data = subprocess.run(
            ["git", "-C", str(root), "show", f"{UPSTREAM_COMMIT}:testvectors_v1/{name}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot read pinned upstream blob testvectors_v1/{name}: {error}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        fail(f"upstream {name} has SHA-256 {actual}; expected {expected}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path)
    args = parser.parse_args()

    actual_files = {path.name for path in LOCAL_DIR.glob("*.json")}
    expected_files = set(FILES)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        fail(f"corpus coverage drift; missing={missing}, extra={extra}")

    for name, expected in FILES.items():
        local = read_pinned(LOCAL_DIR / name, expected)
        if args.upstream_root is not None:
            upstream = read_upstream_blob(args.upstream_root, name, expected)
            if upstream != local:
                fail(f"{name} differs from pinned upstream bytes")

    if args.upstream_root is not None:
        require_exact_checkout(args.upstream_root)

    print("auth-vector-provenance: committed corpus matches the pinned C2SP/Wycheproof commit")


if __name__ == "__main__":
    main()
