#!/usr/bin/env python3
"""Generate binary hash testdata blobs under crates/hashes/testdata.

This is a manual regeneration utility for vector fixtures consumed by tests.
Inputs come from:
- checked-in sources (for BLAKE3/KMAC)
- locally cached cargo registry crates (for some Ascon/Blake2 vectors)

Run via:
  just gen-hashes-testdata
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def encode_git_vlq(val: int) -> bytes:
    if val < 0:
        raise ValueError("val must be non-negative")
    if val > 270_549_119:
        raise ValueError("val too large for blobby VLQ")
    buf = bytearray(4)
    for n in (3, 2, 1, 0):
        if n == 3:
            buf[n] = val & 0x7F
        else:
            val -= 1
            buf[n] = 0x80 | (val & 0x7F)
        val >>= 7
        if val == 0:
            return bytes(buf[n:])
    raise RuntimeError("unreachable")


def blb_encode(blobs: Iterable[bytes]) -> bytes:
    out = bytearray()
    out += encode_git_vlq(0)  # no de-duplicated blobs
    for blob in blobs:
        out += encode_git_vlq(len(blob) << 1)
        out += blob
    return bytes(out)


def write_blb(path: Path, blobs: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blb_encode(blobs))


def hex_to_bytes(s: str) -> bytes:
    s = s.strip()
    if s == "":
        return b""
    return bytes.fromhex(s)


def parse_nist_msg_md(path: Path) -> list[tuple[bytes, bytes]]:
    msg_re = re.compile(r"^Msg\s*=\s*(.*)$")
    md_re = re.compile(r"^MD\s*=\s*([0-9A-Fa-f]*)$")

    msg: bytes | None = None
    pairs: list[tuple[bytes, bytes]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = msg_re.match(line)
        if m is not None:
            msg = hex_to_bytes(m.group(1))
            continue
        m = md_re.match(line)
        if m is not None:
            if msg is None:
                raise ValueError(f"MD without Msg in {path}: {line}")
            md = hex_to_bytes(m.group(1))
            pairs.append((msg, md))
            msg = None
            continue

    if msg is not None:
        raise ValueError(f"trailing Msg without MD in {path}")
    return pairs


def generate_ascon(testdata_root: Path) -> None:
    src = glob.glob(os.path.expanduser("~/.cargo/registry/src/index.crates.io-*/ascon-hash256-0.5.0-rc.0/tests/data/asconhash.txt"))
    if not src:
        raise RuntimeError("could not find ascon-hash256 vector source in ~/.cargo/registry")
    hash_pairs = parse_nist_msg_md(Path(src[0]))
    write_blb(
        testdata_root / "ascon" / "asconhash.blb",
        [b for pair in hash_pairs for b in pair],
    )

    src = glob.glob(os.path.expanduser("~/.cargo/registry/src/index.crates.io-*/ascon-hash256-0.5.0-rc.0/tests/data/asconxof.txt"))
    if not src:
        raise RuntimeError("could not find ascon-hash256 xof vector source in ~/.cargo/registry")
    xof_pairs = parse_nist_msg_md(Path(src[0]))
    write_blb(
        testdata_root / "ascon" / "asconxof.blb",
        [b for pair in xof_pairs for b in pair],
    )


@dataclass(frozen=True)
class Blake3Case:
    input_len: int
    hash_xof: bytes
    keyed_xof: bytes
    derive_xof: bytes


def generate_blake3(testdata_root: Path) -> None:
    src = (testdata_root / "blake3" / "test_vectors.json").read_text(encoding="utf-8")
    vectors = json.loads(src)

    key = vectors["key"].encode("utf-8")
    if len(key) != 32:
        raise ValueError(f"unexpected blake3 key length: {len(key)}")
    context = vectors["context_string"].encode("utf-8")

    cases: list[Blake3Case] = []
    for case in vectors["cases"]:
        cases.append(
            Blake3Case(
                input_len=int(case["input_len"]),
                hash_xof=hex_to_bytes(case["hash"]),
                keyed_xof=hex_to_bytes(case["keyed_hash"]),
                derive_xof=hex_to_bytes(case["derive_key"]),
            )
        )

    blobs: list[bytes] = []
    for case in cases:
        blobs.append(key)
        blobs.append(context)
        blobs.append(int(case.input_len).to_bytes(8, byteorder="little", signed=False))
        blobs.append(case.hash_xof)
        blobs.append(case.keyed_xof)
        blobs.append(case.derive_xof)

    write_blb(testdata_root / "blake3" / "test_vectors.blb", blobs)


def generate_blake2(testdata_root: Path) -> None:
    # Source Blake2b test vectors from the blake2 crate
    src_b_fixed = glob.glob(os.path.expanduser("~/.cargo/registry/src/index.crates.io-*/blake2-0.10.6/tests/data/blake2b/fixed.blb"))
    if not src_b_fixed:
        raise RuntimeError("could not find blake2b fixed vector source in ~/.cargo/registry")

    # Copy the Blake2b fixed vectors (64-byte output)
    blake2b_data = Path(src_b_fixed[0]).read_bytes()
    (testdata_root / "blake2" / "blake2b_fixed.blb").parent.mkdir(parents=True, exist_ok=True)
    (testdata_root / "blake2" / "blake2b_fixed.blb").write_bytes(blake2b_data)

    # Generate Blake2s-256 test vectors using the blake2 crate as reference.
    # The upstream blake2s/variable.blb is too small, so we generate our own.
    try:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
            f.write("""
use blake2::{Blake2s256, Digest};

fn main() {
    let inputs = [
        b"".as_slice(),
        b"a",
        b"abc",
        b"message digest",
        b"abcdefghijklmnopqrstuvwxyz",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
    ];

    for input in inputs {
        let output = Blake2s256::digest(input);
        print!("{}:{}", hex::encode(input), hex::encode(output));
    }
}
""")
            temp_rs = f.name

        result = subprocess.run(
            ["cargo", "script", "--dep", "blake2@0.10", "--dep", "hex@0.4", temp_rs],
            capture_output=True,
            text=True,
            check=True,
        )
        pairs_hex = result.stdout.strip().split()

        blobs: list[bytes] = []
        for pair in pairs_hex:
            input_hex, output_hex = pair.split(":")
            blobs.append(bytes.fromhex(input_hex))
            blobs.append(bytes.fromhex(output_hex))

        write_blb(testdata_root / "blake2" / "blake2s_variable.blb", blobs)

    except Exception as e:
        print(f"Warning: Failed to generate Blake2s vectors via cargo-script: {e}")
        print("Falling back to hardcoded vectors...")

        # Fallback: hardcoded test vectors from the Blake2 specification
        blobs: list[bytes] = []
        # Empty string
        blobs.append(b"")
        blobs.append(hex_to_bytes("69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"))
        # "abc"
        blobs.append(b"abc")
        blobs.append(hex_to_bytes("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"))
        # Longer message
        blobs.append(b"The quick brown fox jumps over the lazy dog")
        blobs.append(hex_to_bytes("606beeec743ccbeff6cbcdf5d5302aa855c256c29b88c8ed331ea1a6bf3c8812"))

        write_blb(testdata_root / "blake2" / "blake2s_variable.blb", blobs)


def generate_kmac(testdata_root: Path) -> None:
    # Vectors mirrored from tiny-keccak 2.0.2 tests (SP800-185 KATs).
    key = bytes(range(0x40, 0x60))
    msg_short = bytes([0x00, 0x01, 0x02, 0x03])
    msg_long = bytes(range(0x00, 0xC8))  # 0x00..0xC7
    custom_empty = b""
    custom_tagged = b"My Tagged Application"

    def bl(hex_str: str) -> bytes:
        return hex_to_bytes(hex_str.replace(" ", "").replace("\n", ""))

    kmac128_fixed = [
        (key, custom_empty, msg_short, bl("E5780B0D3EA6F7D3A429C5706AA43A00FADBD7D49628839E3187243F456EE14E")),
        (key, custom_tagged, msg_short, bl("3B1FBA963CD8B0B59E8C1A6D71888B7143651AF8BA0A7070C0979E2811324AA5")),
        (key, custom_tagged, msg_long, bl("1F5B4E6CCA02209E0DCB5CA635B89A15E271ECC760071DFD805FAA38F9729230")),
    ]
    kmac128_xof = [
        (key, custom_empty, msg_short, bl("CD83740BBD92CCC8CF032B1481A0F4460E7CA9DD12B08A0C4031178BACD6EC35")),
        (key, custom_tagged, msg_short, bl("31A44527B4ED9F5C6101D11DE6D26F0620AA5C341DEF41299657FE9DF1A3B16C")),
        (key, custom_tagged, msg_long, bl("47026C7CD793084AA0283C253EF658490C0DB61438B8326FE9BDDF281B83AE0F")),
    ]
    kmac256_fixed = [
        (
            key,
            custom_tagged,
            msg_short,
            bl(
                "20C570C31346F703C9AC36C61C03CB64"
                "C3970D0CFC787E9B79599D273A68D2F7"
                "F69D4CC3DE9D104A351689F27CF6F595"
                "1F0103F33F4F24871024D9C27773A8DD"
            ),
        ),
        (
            key,
            custom_tagged,
            msg_long,
            bl(
                "B58618F71F92E1D56C1B8C55DDD7CD18"
                "8B97B4CA4D99831EB2699A837DA2E4D9"
                "70FBACFDE50033AEA585F1A2708510C3"
                "2D07880801BD182898FE476876FC8965"
            ),
        ),
    ]
    kmac256_xof = [
        (
            key,
            custom_tagged,
            msg_short,
            bl(
                "1755133F1534752AAD0748F2C706FB5C"
                "784512CAB835CD15676B16C0C6647FA9"
                "6FAA7AF634A0BF8FF6DF39374FA00FAD"
                "9A39E322A7C92065A64EB1FB0801EB2B"
            ),
        ),
        (
            key,
            custom_empty,
            msg_long,
            bl(
                "FF7B171F1E8A2B24683EED37830EE797"
                "538BA8DC563F6DA1E667391A75EDC02C"
                "A633079F81CE12A25F45615EC8997203"
                "1D18337331D24CEB8F8CA8E6A19FD98B"
            ),
        ),
        (
            key,
            custom_tagged,
            msg_long,
            bl(
                "D5BE731C954ED7732846BB59DBE3A8E3"
                "0F83E77A4BFF4459F2F1C2B4ECEBB8CE"
                "67BA01C62E8AB8578D2D499BD1BB2767"
                "68781190020A306A97DE281DCC30305D"
            ),
        ),
    ]

    def flatten(cases: list[tuple[bytes, bytes, bytes, bytes]]) -> list[bytes]:
        out: list[bytes] = []
        for k, c, m, d in cases:
            out.extend([k, c, m, d])
        return out

    write_blb(testdata_root / "sha3" / "kmac128_fixed.blb", flatten(kmac128_fixed))
    write_blb(testdata_root / "sha3" / "kmac128_xof.blb", flatten(kmac128_xof))
    write_blb(testdata_root / "sha3" / "kmac256_fixed.blb", flatten(kmac256_fixed))
    write_blb(testdata_root / "sha3" / "kmac256_xof.blb", flatten(kmac256_xof))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    testdata_root = repo_root / "crates" / "hashes" / "testdata"
    generate_ascon(testdata_root)
    generate_blake2(testdata_root)
    generate_blake3(testdata_root)
    generate_kmac(testdata_root)


if __name__ == "__main__":
    main()
