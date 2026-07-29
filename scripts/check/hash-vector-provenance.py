#!/usr/bin/env python3
"""Verify pinned hash-vector corpora and their deterministic transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SHA2_COMMIT = "82c36a428f8d6f05f3bfccdedb243e9d1f85359d"
SHA3_COMMIT = "1637e892b5658941d04a4d895165b66780c7d7ab"
BLAKE2_COMMIT = "ed1974ea83433eba7b2d95c5dcd9ac33cb847913"
BLAKE3_COMMIT = "8aa5145039b972ba30e98e788752d37d14568824"

SHA2_FILES = {
    "sha224.blb": "59b185972521af418fd49a079de3d5f5bed74cd76d80473da51cab3faee6c7d0",
    "sha256.blb": "bb096934bb7e43e41ce143d211397afca6fcdfe243a39811688ea31aae6f800a",
    "sha384.blb": "e8fe66c07ba336fae2c0aa4c87cb768f41bd4ed318ee1a36fbde0a68581946ec",
    "sha512.blb": "1cc0e86571f2f4e3bc81438ce7b6c25c118d2d7437355240113f59cbb782c8d6",
    "sha512_256.blb": "95195b758e362d92ff0cebebac4cca696512ea5811b635243bc70e29164e5786",
}

SHA3_FILES = {
    "sha3_224.blb": "9c6676da06e149cf2f71be4b4554d042f7c5fa6d5f43696a30ca8d6747c85a23",
    "sha3_256.blb": "00e7834e0abc16614b772a0c6245a29c16807e79c54aa153b008f11cd26268d6",
    "sha3_384.blb": "cd4d9c607c5518a0274415b89512ea4cb9be3fd25edf8269aef566d904b797e9",
    "sha3_512.blb": "f551f332df7fc50b313544aadad361ce3ce5fd91f21259b93c64b35157904be3",
    "shake128.blb": "5900de7f0e09bfd290bee04b183f69fef8407a022491f5f6018cad737de53e4a",
    "shake256.blb": "4b65535c6e28e34f840df71b6dd0d99f51bac13d191e3769861e8560bf9d2373",
}

BLAKE2_SOURCE_SHA256 = "5031ac14800798ae15cee79c04d65e326a575f2c968c7e2846a79bd07a1c0e61"
BLAKE2_FILES = {
    "blake2b.blb": "00e098356d825dc75608e41a60c52b2e26ff06518dd84b262a781cb8b4a73d90",
    "blake2s.blb": "c9de6782932db24c4510cea1b24ad67cd7f0834e974fae893957c12b1dea773e",
}

BLAKE3_SOURCE_SHA256 = "dcb91ea8accc77e6d6e632af7cdc1a99a9f3ae78cf648da595c7d064db32f624"
BLAKE3_OUTPUT_SHA256 = "c56e08d48fc279088f99794e004bc774a76061a8705ed059a09dd9ea535e671d"


def fail(message: str) -> None:
    raise SystemExit(f"hash-vector-provenance: {message}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_digest(path: Path, expected: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    actual = digest(data)
    if actual != expected:
        fail(f"{path} has SHA-256 {actual}; expected {expected}")
    return data


def require_git_commit(root: Path, expected: str) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot inspect upstream checkout {root}: {error}")
    if actual != expected:
        fail(f"{root} is at {actual}; expected {expected}")


def require_exact_artifacts(directory: Path, expected: set[str]) -> None:
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".blb", ".json"}
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        fail(f"{directory} corpus coverage drift; missing={missing}, extra={extra}")


def encode_vlq(value: int) -> bytes:
    encoded = [0, 0, 0, 0]
    for index in (3, 2, 1, 0):
        if index == 3:
            encoded[index] = value & 0x7F
        else:
            value -= 1
            encoded[index] = 0x80 | (value & 0x7F)
        value >>= 7
        if value == 0:
            return bytes(encoded[index:])
    fail("vector blob is too large for the pinned VLQ format")


def encode_blobs_no_dedup(blobs: list[bytes]) -> bytes:
    output = bytearray(b"\0")
    for blob in blobs:
        output.extend(encode_vlq(len(blob) << 1))
        output.extend(blob)
    return bytes(output)


def verify_copied_family(
    local_dir: Path,
    source_dir: Path | None,
    source_commit: str,
    files: dict[str, str],
) -> None:
    for name, expected in files.items():
        local = require_digest(local_dir / name, expected)
        if source_dir is not None:
            source = require_digest(source_dir / name, expected)
            if source != local:
                fail(f"{local_dir / name} differs from pinned upstream bytes")
    if source_dir is not None:
        require_git_commit(source_dir.parents[2], source_commit)


def blake2_outputs(source: bytes) -> dict[str, bytes]:
    try:
        cases = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"invalid BLAKE2 source JSON: {error}")
    outputs: dict[str, bytes] = {}
    for family in ("blake2b", "blake2s"):
        selected = [case for case in cases if case.get("hash") == family]
        if len(selected) != 512:
            fail(f"BLAKE2 source has {len(selected)} {family} cases; expected 512")
        blobs: list[bytes] = []
        try:
            for case in selected:
                blobs.extend(bytes.fromhex(case[field]) for field in ("in", "key", "out"))
        except (KeyError, TypeError, ValueError) as error:
            fail(f"invalid BLAKE2 {family} case: {error}")
        outputs[f"{family}.blb"] = encode_blobs_no_dedup(blobs)
    return outputs


def blake3_output(source: bytes) -> bytes:
    try:
        vectors = json.loads(source)
        key = vectors["key"].encode()
        context = vectors["context_string"].encode()
        cases = vectors["cases"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, AttributeError) as error:
        fail(f"invalid BLAKE3 source JSON: {error}")
    if len(cases) != 35:
        fail(f"BLAKE3 source has {len(cases)} cases; expected 35")
    blobs: list[bytes] = []
    try:
        for case in cases:
            blobs.extend(
                (
                    key,
                    context,
                    int(case["input_len"]).to_bytes(8, "little"),
                    bytes.fromhex(case["hash"]),
                    bytes.fromhex(case["keyed_hash"]),
                    bytes.fromhex(case["derive_key"]),
                )
            )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        fail(f"invalid BLAKE3 case: {error}")
    return encode_blobs_no_dedup(blobs)


def verify_generated(output_dir: Path, generated: dict[str, bytes], expected: dict[str, str]) -> None:
    for name, data in generated.items():
        if digest(data) != expected[name]:
            fail(f"generated {name} has an unexpected digest")
        if require_digest(output_dir / name, expected[name]) != data:
            fail(f"{output_dir / name} differs from deterministic output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha2-root", type=Path)
    parser.add_argument("--sha3-root", type=Path)
    parser.add_argument("--blake2-root", type=Path)
    parser.add_argument("--blake3-root", type=Path)
    args = parser.parse_args()

    require_exact_artifacts(ROOT / "testdata/sha2", set(SHA2_FILES))
    require_exact_artifacts(ROOT / "testdata/sha3", set(SHA3_FILES))
    require_exact_artifacts(ROOT / "testdata/blake2", set(BLAKE2_FILES))
    require_exact_artifacts(
        ROOT / "testdata/blake3",
        {"test_vectors.blb", "test_vectors.json"},
    )

    verify_copied_family(
        ROOT / "testdata/sha2",
        args.sha2_root / "sha2/tests/data" if args.sha2_root else None,
        SHA2_COMMIT,
        SHA2_FILES,
    )
    verify_copied_family(
        ROOT / "testdata/sha3",
        args.sha3_root / "sha3/tests/data" if args.sha3_root else None,
        SHA3_COMMIT,
        SHA3_FILES,
    )

    for name, expected in BLAKE2_FILES.items():
        require_digest(ROOT / "testdata/blake2" / name, expected)
    if args.blake2_root:
        require_git_commit(args.blake2_root, BLAKE2_COMMIT)
        source = require_digest(
            args.blake2_root / "testvectors/blake2-kat.json",
            BLAKE2_SOURCE_SHA256,
        )
        verify_generated(ROOT / "testdata/blake2", blake2_outputs(source), BLAKE2_FILES)

    local_blake3 = require_digest(
        ROOT / "testdata/blake3/test_vectors.json",
        BLAKE3_SOURCE_SHA256,
    )
    generated_blake3 = blake3_output(local_blake3)
    verify_generated(
        ROOT / "testdata/blake3",
        {"test_vectors.blb": generated_blake3},
        {"test_vectors.blb": BLAKE3_OUTPUT_SHA256},
    )
    if args.blake3_root:
        require_git_commit(args.blake3_root, BLAKE3_COMMIT)
        source = require_digest(
            args.blake3_root / "test_vectors/test_vectors.json",
            BLAKE3_SOURCE_SHA256,
        )
        if source != local_blake3:
            fail("committed BLAKE3 JSON differs from pinned upstream bytes")

    print("hash-vector-provenance: committed corpora match pinned digests and transforms")


if __name__ == "__main__":
    main()
