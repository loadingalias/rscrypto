#!/usr/bin/env python3
"""Verify the pinned authentication vector corpora and their transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "testdata/auth/wycheproof"
NIST_LOCAL = ROOT / "testdata/auth/nist/KAS_ECC_CDH_PrimitiveTest_P-256.rsp"
NIST_LOCAL_SHA256 = "5a7006d1ae4f7001ba7d6d45c2c2f1f8bc5e5d48e2021eb55c5995cd055eea32"
NIST_ARCHIVE_SHA256 = "5fff092551f2d72e89a3d9362711878708f9a14b502f0dfae819649105b0ea39"
NIST_ARCHIVE_MEMBER = "KAS_ECC_CDH_PrimitiveTest.txt"
UPSTREAM_COMMIT = "b61843a9a5115bb758134b6a1f5d5e502d445342"
UPSTREAM_FILES = {
    "ecdh_secp256r1_ecpoint_test.json": "648f16d077caf2400d02331ca51f44744c72c799830c8d0595d0b18b6dd9f886",
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
LOCAL_FILES = {
    "ecdh_secp256r1_ecpoint_test.json": "648f16d077caf2400d02331ca51f44744c72c799830c8d0595d0b18b6dd9f886",
    "ecdsa_secp256r1_sha256_test.json": "0fdda72545bff71a635255242a9efb76abb03378718d26e02d37ef7dec9713cd",
    "ecdsa_secp384r1_sha384_test.json": "d4f0c12f6f81f75d8895af1232a63c719913224e05b471823ea3c80aeba886dc",
    "ed25519_test.json": "62bea6ae2ac471a3307a2494813d39d4a82ca962fac29608617619ce47fc704e",
    "hkdf_sha256_test.json": "9e64dbd3f63e46ea505a8ee81e40c49188df5b6d5203afd92e6ede8a2c873d2a",
    "hkdf_sha384_test.json": "e35e585d05c7d81fe1ec3791cb4073460fbcc3c452aacaf7cc8790b83f612d48",
    "hmac_sha256_test.json": "cb5ae2081056393a6b95622e63870df023837212c4bcd16f80fb716f364f51bc",
    "hmac_sha384_test.json": "e78ab52be42de3b0122e933ce7c6afc1e0058157ae29754a6ea54daaf13f84cb",
    "hmac_sha512_test.json": "6f00cc2656af6c53cbf68b4646dbdb4503d4db0085ff69459ac7c9627b4999fe",
    "kmac256_no_customization_test.json": "c3e6acc89e304464dd6c4f8eb9a6c867088a29c7bfee3d4d3afda4e0188c6576",
    "pbkdf2_hmacsha256_test.json": "6bf02f2d48262d97e6ac62bf29e45549d8ef9d4be165aef54568b22262e5d260",
    "pbkdf2_hmacsha512_test.json": "212634c805075f1691e5add5c48c720b80784d3975ae9dc9eef06f77aa7c03f6",
    "x25519_test.json": "558cb1584593dcb204e1cea51fce82b49d10de94c3c822f63f91bfc0f24708ed",
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


def require_same_json(name: str, local: bytes, upstream: bytes) -> None:
    try:
        local_document = json.loads(local)
        upstream_document = json.loads(upstream)
    except json.JSONDecodeError as error:
        fail(f"cannot parse {name}: {error}")
    if local_document != upstream_document:
        fail(f"{name} differs semantically from the pinned upstream blob")


def read_nist_p256_section(archive_path: Path) -> bytes:
    archive = read_pinned(archive_path, NIST_ARCHIVE_SHA256)
    try:
        with zipfile.ZipFile(archive_path) as source:
            if source.namelist() != [NIST_ARCHIVE_MEMBER]:
                fail(f"unexpected NIST archive inventory: {source.namelist()}")
            document = source.read(NIST_ARCHIVE_MEMBER)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        fail(f"cannot read NIST archive {archive_path}: {error}")

    if len(archive) == 0:
        fail(f"NIST archive {archive_path} is empty")

    normalized = document.replace(b"\r\n", b"\n")
    start_marker = b"[P-256]\n"
    end_marker = b"[P-384]\n"
    try:
        start = normalized.index(start_marker)
        end = normalized.index(end_marker, start + len(start_marker))
    except ValueError as error:
        fail(f"cannot locate the exact P-256 section in {NIST_ARCHIVE_MEMBER}: {error}")
    return normalized[start:end].rstrip(b"\n") + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--nist-archive", type=Path)
    args = parser.parse_args()

    nist_local = read_pinned(NIST_LOCAL, NIST_LOCAL_SHA256)
    if args.nist_archive is not None:
        nist_upstream = read_nist_p256_section(args.nist_archive)
        if nist_local != nist_upstream:
            fail(f"{NIST_LOCAL} differs from the normalized P-256 section in {args.nist_archive}")

    actual_files = {path.name for path in LOCAL_DIR.glob("*.json")}
    expected_files = set(LOCAL_FILES)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        fail(f"corpus coverage drift; missing={missing}, extra={extra}")

    if set(UPSTREAM_FILES) != expected_files:
        fail("local and upstream provenance inventories differ")

    for name, local_expected in LOCAL_FILES.items():
        local = read_pinned(LOCAL_DIR / name, local_expected)
        if args.upstream_root is not None:
            upstream = read_upstream_blob(args.upstream_root, name, UPSTREAM_FILES[name])
            require_same_json(name, local, upstream)

    if args.upstream_root is not None:
        require_exact_checkout(args.upstream_root)

    print("auth-vector-provenance: committed NIST and Wycheproof corpora match pinned digests and transforms")


if __name__ == "__main__":
    main()
